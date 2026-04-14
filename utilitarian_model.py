"""
PROBLEM CONTEXT:
    We are reallocating $4.124 billion in FEMA Individual Assistance (IA) across 204 disaster-affected 
    regions from Hurricanes Harvey, Irma, and Maria. The goal is to maximize total weighted societal benefit, 
    where each region's weight is its need-per-capita (n_i).
 
MATHEMATICAL FORMULATION:
    Sets & Indices
        I = {1, 2, ..., 204} set of regions (counties / municipios / islands)
 
    Data / Parameters (all read from ModelingInput.csv)
        n_i = need per capita for region i      (= d_i / population_i)
        d_i = adjusted total need for region i  (dollars)
        p_i = population of region i
        B   = total budget = sum of ia_totalApprovedIhp = $4,124,000,952.74
 
    Decision Variables
        x_i  ∈ R+    continuous — dollars allocated to region i
        y_i  ∈ {0,1} binary     — 1 if region i receives any aid, 0 otherwise

    Objective (Utilitarian — maximize weighted benefit)
        max  Σ_{i ∈ I} n_i · x_i
        - We want to maximize the total benefit across all regions, where each dollar of aid is weighted by 
          how badly that region needs it on a per-capita basis. Regions with higher need-per-capita get a higher 
          value in the objective, so the optimizer naturally prioritizes them.
 
    Constraints
    1. Budget constraint: Total aid across all regions cannot exceed the budget.
            Σ_{i ∈ I}  x_i  ≤  B
 
    2. Non-negativity: No region can receive negative aid.
            x_i  ≥  0  for all i ∈ I
 
    3. Need cap: No region gets more than its stated need.
            x_i  ≤  d_i  for all i ∈ I
 
    4. Linking — upper bound: If the region i is not selected, then x_i is forced to 0.  If y_i = 1, then x_i 
                              can be up to d_i.
            x_i  ≤  d_i · y_i  for all i ∈ I

    5. Minimum floor: If a region receives aid at all, it must get at least 10% of its need.  This prevents the 
                      optimizer from giving a region a meaninglessly small amount.
            x_i  ≥  0.10 · d_i · y_i  for all i ∈ I
 
    6. Binary integrality: Whether or not region i recieves any aid.
            y_i  ∈ {0, 1}      for all i ∈ I
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
 
df = pd.read_csv("Data/ModelingInput.csv")

# Parameters
n = len(df)                                        # 204 regions
B = df["ia_totalApprovedIhp"].sum()                # total budget
d_i = df["d_i"].values                               # adjusted need per region
n_i = df["n_i"].values                             # need per capita per region
pop = df["population"].values                      # population per region
 
print("=" * 70)
print("MODEL 1 — UTILITARIAN (Weighted Social Welfare Maximization)")
print("=" * 70)
print(f"  Number of regions : {n}")
print(f"  Total budget  (B) : ${B:,.2f}")
print(f"  Total need (Σd_i) : ${d_i.sum():,.2f}")
print(f"  Budget / Need     : {B / d_i.sum():.4f}  (budget covers {100*B/d_i.sum():.2f}% of total need)")
print()
 
# =============================================================================
# 2.  BUILD THE GUROBI MODEL
# =============================================================================
model = gp.Model("Utilitarian_Disaster_Relief")
model.setParam("OutputFlag", 1)          # show solver log
model.setParam("MIPGap", 1e-6)          # tight optimality gap
 
# --- Decision variables ---
# x_i : continuous, dollars allocated to region i  (lower bound = 0)
x = {}
for i in range(n):
    x[i] = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                         name=f"x_{df.iloc[i]['county']}")
 
# y_i : binary, 1 if region i receives aid
y = {}
for i in range(n):
    y[i] = model.addVar(vtype=GRB.BINARY,
                         name=f"y_{df.iloc[i]['county']}")
 
model.update()
 
# --- Objective: max Σ n_i · x_i ---
model.setObjective(
    gp.quicksum(n_i[i] * x[i] for i in range(n)),
    GRB.MAXIMIZE
)
 
# --- Constraint (1): Budget ---
model.addConstr(
    gp.quicksum(x[i] for i in range(n)) <= B,
    name="Budget"
)
 
# --- Constraint (3): Need cap   x_i ≤ d_i ---
for i in range(n):
    model.addConstr(
        x[i] <= d_i[i],
        name=f"NeedCap_{i}"
    )
 
# --- Constraint (4): Linking upper   x_i ≤ d_i · y_i ---
for i in range(n):
    model.addConstr(
        x[i] <= d_i[i] * y[i],
        name=f"LinkUpper_{i}"
    )
 
# --- Constraint (5): Minimum floor   x_i ≥ 0.10 · d_i · y_i ---
for i in range(n):
    model.addConstr(
        x[i] >= 0.10 * d_i[i] * y[i],
        name=f"MinFloor_{i}"
    )
 
# (Constraints 2 and 6 are handled by variable bounds and types.)
 
print(f"  Variables   : {model.NumVars}  ({n} continuous x_i + {n} binary y_i)")
print(f"  Constraints : {model.NumConstrs}")
print()
 
# =============================================================================
# 3.  SOLVE
# =============================================================================
model.optimize()
 
if model.status != GRB.OPTIMAL:
    print(f"WARNING: Solver status = {model.status}")
else:
    print(f"\nOptimal objective value: {model.ObjVal:,.2f}")
 
# =============================================================================
# 4.  EXTRACT & ANALYZE RESULTS
# =============================================================================
results = df[["disasterNumber", "state", "county", "population", "d_i",
              "n_i", "ia_totalApprovedIhp", "fema_per_capita"]].copy()
 
results["x_i_utilitarian"] = [x[i].X for i in range(n)]
results["y_i"]             = [int(round(y[i].X)) for i in range(n)]
results["pct_need_met"]    = results["x_i_utilitarian"] / results["d_i"] * 100
results["per_capita_alloc"] = results["x_i_utilitarian"] / results["population"]
 
# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
total_alloc = results["x_i_utilitarian"].sum()
regions_funded = results["y_i"].sum()
 
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"  Total allocated          : ${total_alloc:,.2f}")
print(f"  Budget                   : ${B:,.2f}")
print(f"  Budget used              : {100 * total_alloc / B:.4f}%")
print(f"  Regions funded (y_i = 1) : {regions_funded} / {n}")
print(f"  Regions excluded (y_i=0) : {n - regions_funded}")
print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Per-state summary
# ─────────────────────────────────────────────────────────────────────────────
print("PER-STATE ALLOCATION SUMMARY:")
print("-" * 70)
state_summary = results.groupby("state").agg(
    num_regions=("county", "count"),
    total_need=("d_i", "sum"),
    total_allocation=("x_i_utilitarian", "sum"),
    total_fema_actual=("ia_totalApprovedIhp", "sum"),
    total_pop=("population", "sum"),
    regions_funded=("y_i", "sum")
).reset_index()
 
state_summary["pct_need_met"] = state_summary["total_allocation"] / state_summary["total_need"] * 100
state_summary["per_capita_util"] = state_summary["total_allocation"] / state_summary["total_pop"]
state_summary["per_capita_fema"] = state_summary["total_fema_actual"] / state_summary["total_pop"]
 
for _, row in state_summary.iterrows():
    print(f"  {row['state']:>2s}: {row['num_regions']:3.0f} regions | "
          f"Allocated ${row['total_allocation']:>15,.2f} | "
          f"FEMA Actual ${row['total_fema_actual']:>15,.2f} | "
          f"Need Met {row['pct_need_met']:6.2f}% | "
          f"Per Cap (Util) ${row['per_capita_util']:>8,.2f} | "
          f"Per Cap (FEMA) ${row['per_capita_fema']:>8,.2f}")
 
print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Key fairness metric: PR vs TX per-capita ratio
# ─────────────────────────────────────────────────────────────────────────────
pr_data = state_summary[state_summary["state"] == "PR"]
tx_data = state_summary[state_summary["state"] == "TX"]
 
if len(pr_data) > 0 and len(tx_data) > 0:
    pr_pc_util = pr_data["per_capita_util"].values[0]
    tx_pc_util = tx_data["per_capita_util"].values[0]
    pr_pc_fema = pr_data["per_capita_fema"].values[0]
    tx_pc_fema = tx_data["per_capita_fema"].values[0]
 
    print("HEADLINE FAIRNESS METRIC — PR vs TX Per-Capita Ratio:")
    print("-" * 70)
    print(f"  Utilitarian model : PR/TX = ${pr_pc_util:,.2f} / ${tx_pc_util:,.2f} = {pr_pc_util/tx_pc_util:.4f}")
    print(f"  FEMA actual       : PR/TX = ${pr_pc_fema:,.2f} / ${tx_pc_fema:,.2f} = {pr_pc_fema/tx_pc_fema:.4f}")
    print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Gini coefficient computation (on per-capita allocation)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
 
def gini_coefficient(values):
    """Compute the Gini coefficient of a numpy array of values."""
    values = np.array(values, dtype=float)
    values = values[values > 0]  # only funded regions for meaningful Gini
    if len(values) == 0:
        return float('nan')
    sorted_vals = np.sort(values)
    n_vals = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (2.0 * np.sum((np.arange(1, n_vals + 1) * sorted_vals))) / (n_vals * np.sum(sorted_vals)) - (n_vals + 1) / n_vals
    return gini
 
gini_util = gini_coefficient(results["per_capita_alloc"].values)
gini_fema = gini_coefficient(results["fema_per_capita"].values)
 
print("GINI COEFFICIENTS (per-capita allocation, lower = more equal):")
print("-" * 70)
print(f"  Utilitarian model : {gini_util:.4f}")
print(f"  FEMA actual       : {gini_fema:.4f}")
print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Top 10 and Bottom 10 regions by allocation
# ─────────────────────────────────────────────────────────────────────────────
print("TOP 10 REGIONS BY UTILITARIAN PER-CAPITA ALLOCATION:")
print("-" * 70)
top10 = results.nlargest(10, "per_capita_alloc")
for _, row in top10.iterrows():
    print(f"  {row['county']:>30s}, {row['state']} | "
          f"Per Cap ${row['per_capita_alloc']:>10,.2f} | "
          f"Need Met {row['pct_need_met']:6.2f}% | "
          f"n_i = {row['n_i']:>10,.2f}")
print()
 
print("BOTTOM 10 FUNDED REGIONS BY UTILITARIAN PER-CAPITA ALLOCATION:")
print("-" * 70)
funded = results[results["y_i"] == 1]
bottom10 = funded.nsmallest(10, "per_capita_alloc")
for _, row in bottom10.iterrows():
    print(f"  {row['county']:>30s}, {row['state']} | "
          f"Per Cap ${row['per_capita_alloc']:>10,.2f} | "
          f"Need Met {row['pct_need_met']:6.2f}% | "
          f"n_i = {row['n_i']:>10,.2f}")
print()
 
# ─────────────────────────────────────────────────────────────────────────────
# Excluded regions (if any)
# ─────────────────────────────────────────────────────────────────────────────
excluded = results[results["y_i"] == 0]
if len(excluded) > 0:
    print(f"EXCLUDED REGIONS (y_i = 0): {len(excluded)} regions")
    print("-" * 70)
    for _, row in excluded.iterrows():
        print(f"  {row['county']:>30s}, {row['state']} | "
              f"d_i = ${row['d_i']:>15,.2f} | "
              f"n_i = {row['n_i']:>10,.4f}")
    print()
 
# =============================================================================
# 5.  SAVE RESULTS
# =============================================================================
output_path = "Results/utilitarian_results.csv"
results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print("=" * 70)