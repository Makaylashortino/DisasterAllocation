import pandas as pd
import re

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("HousingAssistanceRenters.csv", low_memory=False)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# ── Filter to 9 DR numbers ────────────────────────────────────────────────────
DR_NUMBERS = {
    4332: "Harvey – Texas",
    4335: "Irma – US Virgin Islands",
    4336: "Irma – Puerto Rico",
    4337: "Irma – Florida",
    4338: "Irma – Georgia",
    4339: "Maria – Puerto Rico",
    4340: "Maria – US Virgin Islands",
    4341: "Irma – Seminole Tribe of Florida",
    4346: "Irma – South Carolina",
}

df_filtered = df[df["disasterNumber"].isin(DR_NUMBERS.keys())].copy()
print(f"After DR filter: {len(df_filtered):,} rows")

# ── Clean county names ────────────────────────────────────────────────────────
def clean_county(row):
    val = str(row["county"]).strip()
    
    # Remove ALL parenthetical expressions
    name = re.sub(r"\s*\(.*?\)", "", val).strip()
    
    val_lower = val.lower()
    if "municipio" in val_lower:
        return name
    elif "parish" in val_lower:
        return f"{name} Parish"
    elif "county" in val_lower:
        return f"{name} County"
    else:
        return name

df_filtered["county_clean"] = df_filtered.apply(clean_county, axis=1)

# ── Aggregate to county level ─────────────────────────────────────────────────
agg_cols = {
    "validRegistrations": "sum",
    "totalDamage": "sum",
    "approvedForFemaAssistance": "sum",
    "totalApprovedIhpAmount": "sum",
    "repairReplaceAmount": "sum",
    "rentalAmount": "sum",
    "otherNeedsAmount": "sum",
}
agg_cols = {k: v for k, v in agg_cols.items() if k in df_filtered.columns}

df_agg = df_filtered.groupby(
    ["disasterNumber", "state", "county_clean"]
).agg(agg_cols).reset_index()

print(f"\nAfter aggregation to county level: {len(df_agg):,} rows")

# ── Quick summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROWS PER DISASTER NUMBER")
print("="*60)
for dr, label in DR_NUMBERS.items():
    sub = df_agg[df_agg["disasterNumber"] == dr]
    if len(sub) == 0:
        print(f"DR-{dr} ({label}): NOT IN DATA")
    else:
        total_damage = sub["totalDamage"].sum() if "totalDamage" in sub.columns else 0
        total_approved = sub["totalApprovedIhpAmount"].sum() if "totalApprovedIhpAmount" in sub.columns else 0
        print(f"DR-{dr} — {label}")
        print(f"  Counties: {len(sub)}")
        print(f"  Total inspected damage: ${total_damage:,.2f}")
        print(f"  Total approved IHP:     ${total_approved:,.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "CleanedIA_Renters_2017Hurricanes.csv"
df_agg.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"Final row count: {len(df_agg):,}")