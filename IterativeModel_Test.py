import pandas as pd
import os
 
df = pd.read_csv("Data/ModelingInput.csv", low_memory=False)
B = df["ia_totalApprovedIhp"].sum()
print(f"Total budget B= {B}")
increment_pct = 0.01
 
remaining_need        = df["d_i"].values.copy()
allocation            = [0.0] * len(df)
remaining_budget      = B
adjusted_applicants   = df["adjusted_applicants"].values.copy()  # NEW: use applicants not population
d_i                   = df["d_i"].values.copy()
 
iteration = 0
max_iterations = 1_000_000
 
# ── First pass: guarantee every region 10% of their need as a floor ──────────
print("\nApplying 10% floor allocation...")
for i in range(len(df)):
    floor = 0.10 * d_i[i]
    floor = min(floor, remaining_budget)
    allocation[i] += floor
    remaining_need[i] -= floor
    remaining_budget -= floor
 
print(f"After floor pass:")
print(f"  Remaining budget:      ${remaining_budget:,.2f}")
print(f"  Budget used on floors: ${B - remaining_budget:,.2f}")
 
# ── Main iterative loop ───────────────────────────────────────────────────────
while remaining_budget > 0.01 and iteration < max_iterations:
 
    # Priority = remaining need per adjusted applicant
    priority = []
    for i in range(len(df)):
        if remaining_need[i] > 0 and adjusted_applicants[i] > 0:
            priority.append(remaining_need[i] / adjusted_applicants[i])
        else:
            priority.append(0.0)
 
    max_priority = max(priority)
 
    if max_priority <= 0:
        print(f"All regions fully funded at iteration {iteration}")
        break
 
    candidates = []
    for i, p in enumerate(priority):
        if abs(p - max_priority) < 1e-6:
            candidates.append(i)
 
    chosen = max(candidates, key=lambda i: d_i[i])
 
    increment = increment_pct * remaining_need[chosen]
    increment = min(increment, remaining_budget)
    increment = min(increment, remaining_need[chosen])
 
    if increment <= 0:
        break
 
    allocation[chosen]     += increment
    remaining_need[chosen] -= increment
    remaining_budget       -= increment
 
    iteration += 1
 
    if iteration % 100_000 == 0:
        print(f"  Iteration {iteration:,} — remaining budget: ${remaining_budget:,.2f}")
 
print(f"\nCompleted in {iteration:,} iterations")
print(f"Remaining budget: ${remaining_budget:,.2f}")
print(f"Regions funded:   {sum(1 for a in allocation if a > 0)} of {len(df)}")
 
# ── Results ───────────────────────────────────────────────────────────────────
results = df.copy()
results["x_i"]                  = allocation
results["pct_need_met"]         = results["x_i"] / results["d_i"] * 100
results["per_applicant_alloc"]  = results["x_i"] / results["adjusted_applicants"]
results["fema_per_applicant"]   = results["ia_totalApprovedIhp"] / results["adjusted_applicants"]
 
# ── Summary by disaster ───────────────────────────────────────────────────────
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
 
    total_alloc      = sub["x_i"].sum()
    total_need       = sub["d_i"].sum()
    total_fema       = sub["ia_totalApprovedIhp"].sum()
    avg_pct_met      = sub["pct_need_met"].mean()
    avg_per_app      = sub["per_applicant_alloc"].mean()
    fema_per_app     = sub["fema_per_applicant"].mean()
    fema_pct_met     = (total_fema / total_need * 100) if total_need > 0 else 0
 
    print(f"\nDR-{dr} - {label}")
    print(f"  Regions:                    {len(sub)}")
    print(f"  Total allocated:            ${total_alloc:>15,.2f}")
    print(f"  Total need (d_i):           ${total_need:>15,.2f}")
    print(f"  Avg % need met:             {avg_pct_met:>14.1f}%")
    print(f"  FEMA avg % need met:        {fema_pct_met:>14.1f}%")
    print(f"  Avg per applicant (model):  ${avg_per_app:>15.2f}")
    print(f"  Avg per applicant (FEMA):   ${fema_per_app:>15.2f}")
 
# ── Puerto Rico combined ──────────────────────────────────────────────────────
pr_all = results[results["state"] == "PR"]
print(f"\n{'='*65}")
print(f"PUERTO RICO COMBINED (Irma DR-4336 + Maria DR-4339)")
print(f"{'='*65}")
print(f"  Total allocated:       ${pr_all['x_i'].sum():>15,.2f}")
print(f"  Total need:            ${pr_all['d_i'].sum():>15,.2f}")
print(f"  Avg % need met:        {pr_all['pct_need_met'].mean():>14.1f}%")
print(f"  FEMA actual % met:     {(pr_all['ia_totalApprovedIhp'].sum() / pr_all['d_i'].sum() * 100):>13.1f}%")
 
# ── Overall summary ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("OVERALL SUMMARY")
print(f"{'='*65}")
print(f"  Total budget:          ${B:>15,.2f}")
print(f"  Total allocated:       ${results['x_i'].sum():>15,.2f}")
print(f"  Total need:            ${results['d_i'].sum():>15,.2f}")
print(f"  Overall % need met:    {results['x_i'].sum() / results['d_i'].sum() * 100:>14.1f}%")
print(f"  Regions receiving aid: {sum(1 for a in allocation if a > 0)} of {len(df)}")
 
# ── Gini coefficient ──────────────────────────────────────────────────────────
def gini(values):
    values = sorted([v for v in values if v >= 0])
    n = len(values)
    if n == 0:
        return 0
    cumsum = 0
    for i, v in enumerate(values):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(values)) if sum(values) > 0 else 0
 
model_gini = gini(results["per_applicant_alloc"].tolist())
fema_gini  = gini(results["fema_per_applicant"].tolist())
 
print(f"\n  Gini (iterative model): {model_gini:.4f}")
print(f"  Gini (FEMA actual):     {fema_gini:.4f}")
print(f"  (Lower = more equal distribution)")
 
# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("Results", exist_ok=True)
results.to_csv("Results/IterativeModel_Test_Results.csv", index=False)
print(f"\nSaved to: Results/IterativeModel_Test_Results.csv")