import pandas as pd
import unicodedata

# ── Helper: strip accents from text ──────────────────────────────────────────
def remove_accents(text):
    if pd.isna(text):
        return text
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(text))
        if unicodedata.category(c) != 'Mn'
    )

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load IA combined data and clean
# ══════════════════════════════════════════════════════════════════════════════
ia = pd.read_csv("CombinedIA_OwnersRenters.csv", low_memory=False)
print(f"IA combined rows loaded: {len(ia):,}")

# Drop Seminole Tribe
ia = ia[ia["disasterNumber"] != 4341].copy()

# Rename county_clean to county for consistency
ia = ia.rename(columns={"county_clean": "county"})

# Drop Statewide rows
ia = ia[ia["county"].str.lower() != "statewide"]

# Drop unexpected states
VALID_STATES = ["TX", "FL", "GA", "SC", "PR", "VI"]
ia = ia[ia["state"].isin(VALID_STATES)]

# Fix USVI county names
usvi_fix = {
    "St. Croix County":  "St. Croix",
    "St. Thomas County": "St. Thomas",
    "St. John County":   "St. John"
}
ia["county"] = ia["county"].replace(usvi_fix)

# Strip accents from PR county names only
ia["county"] = ia.apply(
    lambda row: remove_accents(row["county"]) if row["state"] == "PR" else row["county"],
    axis=1
)

print(f"After cleanup: {len(ia):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Apply denial rate multipliers to get d_i
# Source: National Low Income Housing Coalition (2019)
# Puerto Rico: 60% denied -> multiplier 2.50
# All others:  30% denied -> multiplier 1.43 (conservative, same as Texas)
# ══════════════════════════════════════════════════════════════════════════════
DENIAL_MULTIPLIERS = {
    4332: 1 / 0.70,  # Harvey - Texas         (30% denied, NLIHC 2019)
    4335: 1 / 0.70,  # Irma - USVI            (assumed same as TX)
    4336: 1 / 0.40,  # Irma - Puerto Rico     (60% denied, NLIHC 2019)
    4337: 1 / 0.70,  # Irma - Florida         (assumed same as TX)
    4338: 1 / 0.70,  # Irma - Georgia         (assumed same as TX)
    4339: 1 / 0.40,  # Maria - Puerto Rico    (60% denied, NLIHC 2019)
    4340: 1 / 0.70,  # Maria - USVI           (assumed same as TX)
    4346: 1 / 0.70,  # Irma - South Carolina  (assumed same as TX)
}

ia["denial_multiplier"] = ia["disasterNumber"].map(DENIAL_MULTIPLIERS)
ia["d_i"] = ia["ia_totalApprovedIhp"] * ia["denial_multiplier"]

print(f"\nDenial multipliers applied:")
print(f"  Puerto Rico: 2.50x (60% denial rate, NLIHC 2019)")
print(f"  All others:  1.43x (30% denial rate, NLIHC 2019)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load and clean population data
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a: US County Census file ─────────────────────────────────────────────────
county_pop = pd.read_csv("census_county_pop.csv", low_memory=False, encoding="latin-1")
county_pop = county_pop[county_pop["SUMLEV"] == 50].copy()
county_pop = county_pop[
    county_pop["STNAME"].isin(["Texas", "Florida", "Georgia", "South Carolina"])
]
county_pop = county_pop[["CTYNAME", "STNAME", "POPESTIMATE2017"]].copy()
county_pop = county_pop.rename(columns={
    "CTYNAME":         "county",
    "STNAME":          "state_name",
    "POPESTIMATE2017": "population"
})
state_abbrev = {
    "Texas": "TX", "Florida": "FL",
    "Georgia": "GA", "South Carolina": "SC"
}
county_pop["state"] = county_pop["state_name"].map(state_abbrev)
county_pop = county_pop[["county", "state", "population"]]
print(f"\nUS county population rows: {len(county_pop):,}")

# ── 3b: Puerto Rico municipio file ───────────────────────────────────────────
pr_pop = pd.read_csv("pr_municipio_pop.csv", header=None,
                     names=["raw_name", "population"])

# Strip leading periods then " Municipio, Puerto Rico"
pr_pop["county"] = pr_pop["raw_name"].str.lstrip(".")
pr_pop["county"] = pr_pop["county"].str.replace(
    r"\s*Municipio,?\s*Puerto Rico", "", regex=True
).str.strip()

# Strip accents to match IA data
pr_pop["county"] = pr_pop["county"].apply(remove_accents)

# Clean population column
pr_pop["population"] = (pr_pop["population"].astype(str)
                        .str.replace(",", "").str.strip())
pr_pop["population"] = pd.to_numeric(pr_pop["population"], errors="coerce")
pr_pop = pr_pop[["county", "population"]].dropna()
pr_pop["state"] = "PR"
print(f"Puerto Rico municipio rows: {len(pr_pop):,}")
print(f"Sample PR names: {pr_pop['county'].head(3).tolist()}")

# ── 3c: USVI hardcoded 2017 populations ──────────────────────────────────────
usvi_pop = pd.DataFrame({
    "county":     ["St. Croix", "St. Thomas", "St. John"],
    "state":      ["VI", "VI", "VI"],
    "population": [50601, 51634, 4170]
})
print(f"USVI hardcoded rows: {len(usvi_pop):,}")

# ── 3d: Combine all population sources ───────────────────────────────────────
all_pop = pd.concat([county_pop, pr_pop, usvi_pop], ignore_index=True)
print(f"Total population rows: {len(all_pop):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Match check before merging
# ══════════════════════════════════════════════════════════════════════════════
ia_pairs  = set(zip(ia["county"], ia["state"]))
pop_pairs = set(zip(all_pop["county"], all_pop["state"]))

matched   = ia_pairs & pop_pairs
unmatched = ia_pairs - pop_pairs

print(f"\n{'='*60}")
print("MATCH CHECK (county + state)")
print(f"{'='*60}")
print(f"IA county+state pairs:      {len(ia_pairs)}")
print(f"Population county+state:    {len(pop_pairs)}")
print(f"Matched:                    {len(matched)}")
print(f"Unmatched (no pop data):    {len(unmatched)}")

if unmatched:
    print(f"\nUnmatched counties:")
    for county, state in sorted(unmatched):
        print(f"  {county} ({state})")
else:
    print("\nAll counties matched!")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Merge population onto IA data using county + state
# ══════════════════════════════════════════════════════════════════════════════
final = pd.merge(
    ia,
    all_pop,
    on=["county", "state"],
    how="left"
)

if len(final) != len(ia):
    print(f"\nWARNING: row count changed! Before: {len(ia)}, After: {len(final)}")
    print("This suggests duplicate matches -- investigate before proceeding")
else:
    print(f"\nRow count unchanged after merge -- no duplicates introduced")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Compute n_i = d_i / p_i and FEMA per capita
# ══════════════════════════════════════════════════════════════════════════════
final["n_i"]             = final["d_i"] / final["population"]
final["fema_per_capita"] = final["ia_totalApprovedIhp"] / final["population"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Summary report
# ══════════════════════════════════════════════════════════════════════════════
DR_NAMES = {
    4332: "Harvey - Texas",
    4335: "Irma - US Virgin Islands",
    4336: "Irma - Puerto Rico",
    4337: "Irma - Florida",
    4338: "Irma - Georgia",
    4339: "Maria - Puerto Rico",
    4340: "Maria - US Virgin Islands",
    4346: "Irma - South Carolina",
}

print(f"\n{'='*65}")
print("SUMMARY: d_i AND n_i BY DISASTER")
print(f"{'='*65}")

for dr, label in DR_NAMES.items():
    sub = final[final["disasterNumber"] == dr]
    if len(sub) == 0:
        continue
    total_d    = sub["d_i"].sum()
    total_pop  = sub["population"].sum()
    avg_n      = sub["n_i"].mean()
    total_fema = sub["ia_totalApprovedIhp"].sum()
    pct_met    = (total_fema / total_d * 100) if total_d > 0 else 0

    print(f"\nDR-{dr} - {label}")
    print(f"  Counties/regions:     {len(sub)}")
    print(f"  Total d_i (need):     ${total_d:>15,.2f}")
    print(f"  Total population:     {total_pop:>15,.0f}")
    print(f"  Avg n_i (need/cap):   ${avg_n:>15,.4f}")
    print(f"  FEMA actual (raw):    ${total_fema:>15,.2f}")
    print(f"  FEMA met:             {pct_met:.1f}% of adjusted need")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Save ModelingInput.csv
# ══════════════════════════════════════════════════════════════════════════════
output_cols = [c for c in [
    "disasterNumber", "state", "county",
    "ia_totalApprovedIhp", "denial_multiplier",
    "d_i", "population", "n_i", "fema_per_capita"
] if c in final.columns]

final[output_cols].to_csv("ModelingInput.csv", index=False)

print(f"\n{'='*65}")
print(f"Saved to: ModelingInput.csv")
print(f"Total regions:                {len(final):,}")
print(f"Total d_i (adjusted need):    ${final['d_i'].sum():,.2f}")
print(f"Total budget B (FEMA actual): ${final['ia_totalApprovedIhp'].sum():,.2f}")
print(f"Overall FEMA met:             {final['ia_totalApprovedIhp'].sum() / final['d_i'].sum() * 100:.1f}% of total adjusted need")