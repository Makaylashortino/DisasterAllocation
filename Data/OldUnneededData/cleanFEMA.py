import pandas as pd
 
df = pd.read_csv("CutDownPublicAssistanceFunded.csv", low_memory=False)
print(f"Loaded {len(df):,} rows")
 

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
 
# Dropping cancelled/withdrawn projects
# Keep: Active, Obligated, Project Closed, Pending Closeout
# Drop: Cancelled, Withdrawn (check what values actually exist first)
print("\nProjectStatus values:")
print(df_filtered["projectStatus"].value_counts())
 
drop_statuses = ["Cancelled", "Withdrawn"]
df_filtered = df_filtered[~df_filtered["projectStatus"].isin(drop_statuses)]
print(f"After status filter: {len(df_filtered):,} rows")
 
#Drop unneeded columns
cols_to_drop = [
    "pwNumber", "applicationTitle", "applicantId",
    "gmProjectId", "gmApplicantId", "lastRefresh",
    "mitigationAmount", "firstObligationDate", "lastObligationDate"
]
cols_to_drop = [c for c in cols_to_drop if c in df_filtered.columns]
df_filtered = df_filtered.drop(columns=cols_to_drop)
 
# Classify county field 
def classify_county(val):
    if pd.isna(val) or str(val).strip() == "":
        return "blank"
    elif str(val).strip().lower() == "statewide":
        return "statewide"
    else:
        return "county"
 
df_filtered["county_type"] = df_filtered["county"].apply(classify_county)
 
# Summary
print("\n" + "="*60)
print("SUMMARY REPORT BY DISASTER NUMBER")
print("="*60)
 
for dr, label in DR_NUMBERS.items():
    sub = df_filtered[df_filtered["disasterNumber"] == dr]
    if len(sub) == 0:
        print(f"\nDR-{dr} ({label}): NOT FOUND IN DATA")
        continue
 
    print(f"\nDR-{dr} — {label}")
    print(f"  Total projects: {len(sub):,}")
 
    # County counts
    county_rows = sub[sub["county_type"] == "county"]
    statewide_rows = sub[sub["county_type"] == "statewide"]
    blank_rows = sub[sub["county_type"] == "blank"]
 
    print(f"  Statewide rows: {len(statewide_rows):,}")
    print(f"  Blank county rows: {len(blank_rows):,}")
    print(f"  Named county rows: {len(county_rows):,}")
 
    if len(county_rows) > 0:
        print("  County breakdown:")
        county_counts = county_rows["county"].value_counts()
        for county, count in county_counts.items():
            print(f"    {county}: {count}")
 
# Save cleaned file
output_path = "CleanedFEMA_2017Hurricanes.csv"
df_filtered.to_csv(output_path, index=False)
print(f"\nCleaned file saved to: {output_path}")
print(f"Final row count: {len(df_filtered):,}")
 