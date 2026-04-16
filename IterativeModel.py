import pandas as pd

df = pd.read_csv("Data/ModelingInput.csv", low_memory=False)
B = df["ia_totalApprovedIhp"].sum()
print(f"Total budget B= {B}")
increment_pct = 0.01 #allocating 10%  of their need each time


allocation     = [0.0] * len(df)                # x_i starts at zero
remaining_budget = B
remaining_need = df["d_i"].values.copy()
population     = df["population"].values.copy()
d_i            = df["d_i"].values.copy()

iteration = 0
max_iterations = 1_000_000

while remaining_budget > 0.01 and iteration < max_iterations:

    priority = []
    for i in range(len(df)):
        if remaining_need[i] > 0 and population[i] > 0:
            priority.append(remaining_need[i] / population[i])
        else:
            priority.append(0.0)

    max_priority = max(priority)

    if max_priority <= 0:
            print(f"All regions fully funded at iteration {iteration}")
            break

    candidates = []
    for i, p in enumerate(priority): #gets me the index i and the p
        if abs(p - max_priority) < 1e-6:
            candidates.append(i)

    chosen = max(candidates, key=lambda i: d_i[i])

    increment = increment_pct * remaining_need[chosen]  # 10% of remaining need
    increment = min(increment, remaining_budget)         # can't be more than budget
    increment = min(increment, remaining_need[chosen])   # can't be more than remaining need

    if increment <= 0:
        break

    allocation[chosen]     += increment
    remaining_need[chosen] -= increment
    remaining_budget       -= increment
 
    iteration += 1
    
    # Progress update every 100k iterations (Claude's help)
    if iteration % 100_000 == 0:
        print(f"  Iteration {iteration:,} — remaining budget: ${remaining_budget:,.2f}")

print(f"\nCompleted in {iteration:,} iterations")
print(f"Remaining budget: ${remaining_budget:,.2f}")
print(f"Regions funded:   {sum(1 for a in allocation if a > 0)} of {len(df)}")
 
#Results (Claude's help)
results = df.copy()
results["x_i"]              = allocation
results["pct_need_met"]     = results["x_i"] / results["d_i"] * 100
results["per_capita_alloc"] = results["x_i"] / results["population"]
results["fema_per_capita"]  = results["ia_totalApprovedIhp"] / results["population"]
 
# Summary by disaster
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
print("ITERATIVE MODEL RESULTS BY DISASTER")
print(f"{'='*65}")
 
for dr, label in DR_NAMES.items():
    sub = results[results["disasterNumber"] == dr]
    if len(sub) == 0:
        continue
 
    total_alloc   = sub["x_i"].sum()
    total_need    = sub["d_i"].sum()
    total_fema    = sub["ia_totalApprovedIhp"].sum()
    avg_pct_met   = sub["pct_need_met"].mean()
    fema_pct_met = (total_fema / total_need * 100) if total_need > 0 else 0
    avg_per_cap   = sub["per_capita_alloc"].mean()
    fema_per_cap  = sub["fema_per_capita"].mean()
 
    print(f"\nDR-{dr} — {label}")
    print(f"  Regions:               {len(sub)}")
    print(f"  Total allocated:       ${total_alloc:>15,.2f}")
    print(f"  Total need (d_i):      ${total_need:>15,.2f}")
    print(f"  Avg % need met:        {avg_pct_met:>14.1f}%")
    print(f"  FEMA avg % need met:   {fema_pct_met:>14.1f}%")
    print(f"  Avg per capita:        ${avg_per_cap:>15.2f}")
    print(f"  FEMA actual per cap:   ${fema_per_cap:>15.2f}")
 
# Puerto Rico combined (Irma + Maria) 
pr_all = results[results["state"] == "PR"]
print(f"\n{'='*65}")
print(f"PUERTO RICO COMBINED (Irma DR-4336 + Maria DR-4339)")
print(f"{'='*65}")
print(f"  Total allocated:       ${pr_all['x_i'].sum():>15,.2f}")
print(f"  Total need:            ${pr_all['d_i'].sum():>15,.2f}")
print(f"  Avg % need met:        {pr_all['pct_need_met'].mean():>14.1f}%")
print(f"  FEMA actual % met:     {(pr_all['ia_totalApprovedIhp'].sum() / pr_all['d_i'].sum() * 100):>13.1f}%")
 
# Overall summary
print(f"\n{'='*65}")
print("OVERALL SUMMARY")
print(f"{'='*65}")
print(f"  Total budget:          ${B:>15,.2f}")
print(f"  Total allocated:       ${results['x_i'].sum():>15,.2f}")
print(f"  Total need:            ${results['d_i'].sum():>15,.2f}")
print(f"  Overall % need met:    {results['x_i'].sum() / results['d_i'].sum() * 100:>14.1f}%")
print(f"  Regions receiving aid: {sum(1 for a in allocation if a > 0)} of {len(df)}")

def gini(values):
    values = sorted([v for v in values if v >= 0])
    n = len(values)
    if n == 0:
        return 0
    cumsum = 0
    for i, v in enumerate(values):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(values)) if sum(values) > 0 else 0

model_gini = gini(results["per_capita_alloc"].tolist())
fema_gini  = gini(results["fema_per_capita"].tolist())

print(f"\n  Gini (iterative model): {model_gini:.4f}")
print(f"  Gini (FEMA actual):     {fema_gini:.4f}")

results.to_csv("Results/IterativeModel_Results.csv", index=False)
print(f"\nSaved to: IterativeModel_Results.csv")