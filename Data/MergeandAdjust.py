import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
ia_owners  = pd.read_csv("CleanedIA_2017Hurricanes.csv", low_memory=False)
ia_renters = pd.read_csv("CleanedIA_Renters_2017Hurricanes.csv", low_memory=False)

print(f"Owners rows:  {len(ia_owners):,}")
print(f"Renters rows: {len(ia_renters):,}")

# ── Drop Seminole Tribe ───────────────────────────────────────────────────────
ia_owners  = ia_owners[ia_owners["disasterNumber"] != 4341]
ia_renters = ia_renters[ia_renters["disasterNumber"] != 4341]

# ── Rename columns to distinguish ────────────────────────────────────────────
ia_owners = ia_owners.rename(columns={
    "validRegistrations":     "owner_validRegistrations",
    "totalDamage":            "owner_totalDamage",
    "approvedForFemaAssistance": "owner_approved",
    "totalApprovedIhpAmount": "owner_totalApprovedIhp",
    "repairReplaceAmount":    "owner_repairReplaceAmount",
    "rentalAmount":           "owner_rentalAmount",
})

ia_renters = ia_renters.rename(columns={
    "validRegistrations":     "renter_validRegistrations",
    "approvedForFemaAssistance": "renter_approved",
    "totalApprovedIhpAmount": "renter_totalApprovedIhp",
    "repairReplaceAmount":    "renter_repairReplaceAmount",
    "rentalAmount":           "renter_rentalAmount",
    "otherNeedsAmount":       "renter_otherNeedsAmount",
})

# ── Merge owners and renters ──────────────────────────────────────────────────
ia_combined = pd.merge(
    ia_owners,
    ia_renters,
    on=["disasterNumber", "state", "county_clean"],
    how="outer"
).fillna(0)

print(f"\nAfter owners + renters merge: {len(ia_combined):,} rows")

# ── Check for duplicates ──────────────────────────────────────────────────────
dupes = ia_combined[
    ia_combined.duplicated(subset=["disasterNumber", "county_clean"], keep=False)
]
if len(dupes) > 0:
    print(f"\n⚠ WARNING: {len(dupes)} duplicate county rows found!")
    print(dupes[["disasterNumber", "state", "county_clean"]])
else:
    print("\n✓ No duplicate counties — merge looks clean")

# ── Compute combined IA totals ────────────────────────────────────────────────
ia_combined["ia_totalApprovedIhp"] = (
    ia_combined.get("owner_totalApprovedIhp", 0) +
    ia_combined.get("renter_totalApprovedIhp", 0)
)
ia_combined["ia_validRegistrations"] = (
    ia_combined.get("owner_validRegistrations", 0) +
    ia_combined.get("renter_validRegistrations", 0)
)
ia_combined["ia_owner_totalDamage"] = ia_combined.get("owner_totalDamage", 0)

# ── Summary by disaster ───────────────────────────────────────────────────────
DR_NAMES = {
    4332: "Harvey – Texas",
    4335: "Irma – US Virgin Islands",
    4336: "Irma – Puerto Rico",
    4337: "Irma – Florida",
    4338: "Irma – Georgia",
    4339: "Maria – Puerto Rico",
    4340: "Maria – US Virgin Islands",
    4346: "Irma – South Carolina",
}

print("\n" + "="*65)
print("COMBINED IA SUMMARY BY DISASTER")
print("="*65)

for dr, label in DR_NAMES.items():
    sub = ia_combined[ia_combined["disasterNumber"] == dr]
    if len(sub) == 0:
        print(f"\nDR-{dr} — {label}: NOT IN DATA")
        continue

    owner_ihp  = sub["owner_totalApprovedIhp"].sum() if "owner_totalApprovedIhp" in sub.columns else 0
    renter_ihp = sub["renter_totalApprovedIhp"].sum() if "renter_totalApprovedIhp" in sub.columns else 0
    combined   = sub["ia_totalApprovedIhp"].sum()
    regs       = sub["ia_validRegistrations"].sum()

    print(f"\nDR-{dr} — {label}")
    print(f"  Counties:              {len(sub)}")
    print(f"  Total registrations:   {regs:,.0f}")
    print(f"  Owner IHP approved:    ${owner_ihp:>18,.2f}")
    print(f"  Renter IHP approved:   ${renter_ihp:>18,.2f}")
    print(f"  Combined IHP approved: ${combined:>18,.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "CombinedIA_OwnersRenters.csv"
ia_combined.to_csv(output_path, index=False)
print(f"\n\nSaved to: {output_path}")
print(f"Total rows: {len(ia_combined):,}")