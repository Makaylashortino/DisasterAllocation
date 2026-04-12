import pandas as pd

df = pd.read_csv("CleanedFEMA_2017Hurricanes.csv", low_memory=False)

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

# Statewide vs County Dollar Breakdown
print("="*65)
print("SECTION 1: STATEWIDE vs COUNTY DOLLAR BREAKDOWN BY DISASTER")
print("="*65)

total_statewide_all = 0
total_county_all = 0

for dr, label in DR_NAMES.items():
    sub = df[df["disasterNumber"] == dr]
    if len(sub) == 0:
        continue

    statewide = sub[sub["county_type"] == "statewide"]["projectAmount"].sum()
    county    = sub[sub["county_type"] == "county"]["projectAmount"].sum()
    total     = statewide + county
    pct       = (statewide / total * 100) if total > 0 else 0

    total_statewide_all += statewide
    total_county_all    += county

    print(f"\nDR-{dr} — {label}")
    print(f"  Named county $: ${county:>18,.2f}")
    print(f"  Statewide $:    ${statewide:>18,.2f}")
    print(f"  Total $:        ${total:>18,.2f}")
    print(f"  Statewide is {pct:.1f}% of total dollars")

grand_total = total_statewide_all + total_county_all
print("\n" + "="*65)
print("OVERALL")
print(f"  Named county $: ${total_county_all:>18,.2f}")
print(f"  Statewide $:    ${total_statewide_all:>18,.2f}")
print(f"  Grand total $:  ${grand_total:>18,.2f}")
print(f"  Statewide is {total_statewide_all/grand_total*100:.1f}% of all dollars")


# Proportional Distribution of Statewide Dollars 
print("\n\n" + "="*65)
print("SECTION 2: ADJUSTED d_i AFTER PROPORTIONAL STATEWIDE DISTRIBUTION")
print("="*65)
print("(Statewide dollars distributed proportionally by named county dollar share)")

adjusted_rows = []

for dr, label in DR_NAMES.items():
    sub = df[df["disasterNumber"] == dr]
    if len(sub) == 0:
        continue

    statewide_amt = sub[sub["county_type"] == "statewide"]["projectAmount"].sum()
    statewide_obl = sub[sub["county_type"] == "statewide"]["totalObligated"].sum()

    county_sub = sub[sub["county_type"] == "county"].copy()
    if len(county_sub) == 0:
        print(f"\nDR-{dr} — {label}: No named county rows, skipping distribution")
        continue

    county_grouped = county_sub.groupby("county").agg(
        named_projectAmount=("projectAmount", "sum"),
        named_totalObligated=("totalObligated", "sum")
    ).reset_index()

    total_named_amt = county_grouped["named_projectAmount"].sum()

    county_grouped["statewide_share"] = (
        county_grouped["named_projectAmount"] / total_named_amt
    )
    county_grouped["distributed_statewide_amt"] = (
        county_grouped["statewide_share"] * statewide_amt
    )
    county_grouped["distributed_statewide_obl"] = (
        county_grouped["statewide_share"] * statewide_obl
    )
    county_grouped["adjusted_projectAmount"] = (
        county_grouped["named_projectAmount"] +
        county_grouped["distributed_statewide_amt"]
    )
    county_grouped["adjusted_totalObligated"] = (
        county_grouped["named_totalObligated"] +
        county_grouped["distributed_statewide_obl"]
    )

    county_grouped["disasterNumber"] = dr
    county_grouped["disaster_label"] = label
    county_grouped["stateAbbreviation"] = sub["stateAbbreviation"].iloc[0]

    adjusted_rows.append(county_grouped)

    print(f"\nDR-{dr} — {label}")
    print(f"  {'County':<30} {'Adjusted Need $':>18} {'Adjusted Obligated $':>20}")
    print(f"  {'-'*30} {'-'*18} {'-'*20}")
    for _, row in county_grouped.iterrows():
        print(f"  {row['county']:<30} ${row['adjusted_projectAmount']:>17,.2f} ${row['adjusted_totalObligated']:>19,.2f}")


# Stated Need vs FEMA Obligated by Disaster
print("\n\n" + "="*65)
print("SECTION 3: STATED NEED vs FEMA OBLIGATED BY DISASTER")
print("="*65)

for dr, label in DR_NAMES.items():
    sub = df[df["disasterNumber"] == dr]
    if len(sub) == 0:
        continue

    need      = sub["projectAmount"].sum()
    obligated = sub["totalObligated"].sum()
    gap       = need - obligated
    pct_met   = (obligated / need * 100) if need > 0 else 0

    print(f"\nDR-{dr} — {label}")
    print(f"  Total stated need:     ${need:>18,.2f}")
    print(f"  Total FEMA obligated:  ${obligated:>18,.2f}")
    print(f"  Gap (unmet need):      ${gap:>18,.2f}")
    print(f"  FEMA met {pct_met:.1f}% of stated need")


# SECTION 4: Save adjusted county-level dataset
if adjusted_rows:
    final_df = pd.concat(adjusted_rows, ignore_index=True)

    output_cols = [
        "disasterNumber", "disaster_label", "stateAbbreviation",
        "county",
        "named_projectAmount", "distributed_statewide_amt", "adjusted_projectAmount",
        "named_totalObligated", "distributed_statewide_obl", "adjusted_totalObligated"
    ]
    final_df = final_df[output_cols]

    output_path = "AdjustedCountyNeed.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n\nAdjusted county-level dataset saved to: {output_path}")
    print(f"Total counties/regions: {len(final_df)}")
    print(f"Total adjusted need:    ${final_df['adjusted_projectAmount'].sum():,.2f}")
    print(f"Total adjusted obligated: ${final_df['adjusted_totalObligated'].sum():,.2f}")