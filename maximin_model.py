"""
PROBLEM CONTEXT:
    We are reallocating $4.124 billion in FEMA Individual Assistance (IA) across 204 disaster-affected 
    regions from Hurricanes Harvey, Irma, and Maria. The goal is to maximize the minimum per-capita aid 
    received across all regions.
 
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
        z    ∈ R     continuous — the minimum per-capita allocation across all funded regions

    Objective (Maximin — maximize the worst-off region)
        max  z
        - We maximize z, an auxiliary variable that represents the minimum per-capita allocation across all 
          regions. This forces the model to keep raising the floor for the least-served region before improving 
          anyone else's allocation. It is the fairness-focused counterpart to the utilitarian model.
 
    Constraints
    1. Budget constraint: Total aid across all regions must equal the budget. (If you do <= here, the model would 
                          just not spend to improve the objective, which is not what we want.)
            Σ_{i ∈ I}  x_i  =  B
 
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

    7. Maximin linking: z cannot exceed the per-capita allocation of any funded region. For unfunded regions 
                        the constraint is relaxed by adding a Big M term so that z is not forced 
                        to zero simply because a region was excluded.
            z  ≤  x_i / p_i  +  M · (1 - y_i)    for all i ∈ I
       So that Gurobi doesn't break...
            z · p_i  ≤  x_i  +  M · p_i · (1 - y_i)    for all i ∈ I
       We set M to be a safely large upper bound: the maximum possible per-capita allocation for any region, 
       which is max(n_i). This is the tightest valid Big M.

Important notes on data preprocessing:
    Several regions (mainly in Puerto Rico and USVI) appear multiple times in the raw data because they were struck by 
    more than one hurricane We aggregate by (state, county) so that each community is represented exactly once in the model,
    with its total need and population across all disasters. This is important to ensure that the model allocates aid at the 
    community level rather than splitting it across  multiple rows for the same place.
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
 
# 1. LOAD & AGGREGATE DATA
raw = pd.read_csv("Data/ModelingInput.csv")
 
# Aggregate by (state, county): sum dollars, keep population constant
df = raw.groupby(["state", "county"]).agg(
    d_i=("d_i", "sum"),
    ia_totalApprovedIhp=("ia_totalApprovedIhp", "sum"),
    population=("population", "first"),
    num_disasters=("disasterNumber", "count"),
    disaster_list=("disasterNumber", lambda x: list(x))
).reset_index()
 
# Recompute per-capita metrics on the aggregated data
df["n_i"] = df["d_i"] / df["population"]
df["fema_per_capita"] = df["ia_totalApprovedIhp"] / df["population"]
 
# Handle any regions with 0 population (shouldn't happen but be safe)
df["n_i"] = df["n_i"].fillna(0)
df["fema_per_capita"] = df["fema_per_capita"].fillna(0)
 
# Parameters
n = len(df)                                        # 183 unique regions
B = df["ia_totalApprovedIhp"].sum()                # total budget
d_i = df["d_i"].values                             # adjusted need per region
n_i = df["n_i"].values                             # need per capita per region
pop = df["population"].values                      # population per region
M = n_i.max()                                      # Big M = max per-capita need
 
print("MODEL 2 — MAXIMIN (Maximize Minimum Per-Capita Allocation)")
print(f"  Raw data rows     : {len(raw)}")
print(f"  After aggregation : {n} unique regions")
print(f"  Multi-disaster    : {(df['num_disasters'] > 1).sum()} regions hit by 2+ hurricanes")
print(f"  Total budget  (B) : ${B:,.2f}")
print(f"  Total need (Σd_i) : ${d_i.sum():,.2f}")
print(f"  Budget / Need     : {B / d_i.sum():.4f}  ({100*B/d_i.sum():.2f}% of total need)")
print(f"  Big-M value       : {M:,.2f} (= max n_i)")
print()
 
# 2.  BUILD THE GUROBI MODEL
model = gp.Model("Maximin_Disaster_Relief")
model.setParam("OutputFlag", 1)          # show solver log
model.setParam("MIPGap", 1e-6)          # tight optimality gap
 
# Decision variables
# x_i : continuous, dollars allocated to region i  (lb = 0)
x = {}
for i in range(n):
    x[i] = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                         name=f"x_{i}_{df.iloc[i]['county']}")
 
# y_i : binary, 1 if region i receives aid, 0 otherwise
y = {}
for i in range(n):
    y[i] = model.addVar(vtype=GRB.BINARY,
                         name=f"y_{i}_{df.iloc[i]['county']}")

# z : continuous, the minimum per-capita allocation across funded regions
z = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
 
model.update()
 
# Objective: max z
model.setObjective(z, GRB.MAXIMIZE)

# Constraint 1: Budget
model.addConstr(
    gp.quicksum(x[i] for i in range(n)) <= B,
    name="Budget"
)
 
# Constraint 3: Need cap
for i in range(n):
    model.addConstr(
        x[i] <= d_i[i],
        name=f"NeedCap_{i}"
    )

# Constraint 4: Linking upper
for i in range(n):
    model.addConstr(
        x[i] <= d_i[i] * y[i],
        name=f"LinkUpper_{i}"
    )
 
# Constraint 5: Minimum floor
for i in range(n):
    model.addConstr(
        x[i] >= 0.10 * d_i[i] * y[i],
        name=f"MinFloor_{i}"
    ) 

# Constraint 7: Maximin linking
for i in range(n):
    model.addConstr(
        z * pop[i] <= x[i] + M * pop[i] * (1 - y[i]),
        name=f"Maximin_{i}"
    )


# 3.  SOLVE
model.optimize()
if model.status != GRB.OPTIMAL:
    print(f"WARNING: Solver status = {model.status}")

# 4.  EXTRACT & ANALYZE RESULTS
results = df[["state", "county", "population", "d_i", "n_i",
              "ia_totalApprovedIhp", "fema_per_capita",
              "num_disasters", "disaster_list"]].copy()
 
results["x_i_maximin"]      = [x[i].X for i in range(n)]
results["y_i"]              = [int(round(y[i].X)) for i in range(n)]
results["pct_need_met"]     = results["x_i_maximin"] / results["d_i"] * 100
results["per_capita_alloc"] = results["x_i_maximin"] / results["population"]

results.to_csv("Results/Maximin/maximin_results.csv", index=False)
 
# Summary statistics
total_alloc = results["x_i_maximin"].sum()
regions_funded = results["y_i"].sum()
z_star = z.X

summary_df = pd.DataFrame({
    "Metric": [
        "Total Allocated",
        "Budget",
        "Budget Used (%)",
        "Regions Funded",
        "Regions Excluded",
        "z* (Min Per-Capita)"
    ],
    "Value": [
        total_alloc,
        B,
        100 * total_alloc / B,
        regions_funded / n,
        n - regions_funded,
        z_star
    ]
})
summary_df.to_csv("Results/Maximin/maximin_summary.csv", index=False)

# Per-state summary
state_summary = results.groupby("state").agg(
    num_regions=("county", "count"),
    total_need=("d_i", "sum"),
    total_allocation=("x_i_maximin", "sum"),
    total_fema_actual=("ia_totalApprovedIhp", "sum"),
    total_pop=("population", "sum"),
    regions_funded=("y_i", "sum")
).reset_index()
 
state_summary["pct_need_met"]      = state_summary["total_allocation"] / state_summary["total_need"] * 100
state_summary["per_capita_maximin"] = state_summary["total_allocation"] / state_summary["total_pop"]
state_summary["per_capita_fema"]   = state_summary["total_fema_actual"] / state_summary["total_pop"]

state_summary.to_csv("Results/Maximin/maximin_state_summary.csv", index=False)

# Gini coefficient computation (on per-capita allocation)
def gini_coefficient(values):
    """Compute the Gini coefficient of a numpy array of values."""
    values = np.array(values, dtype=float)
    values = values[values > 0]  # only funded regions
    if len(values) == 0:
        return float('nan')
    sorted_vals = np.sort(values)
    n_vals = len(sorted_vals)
    gini = (2.0 * np.sum((np.arange(1, n_vals + 1) * sorted_vals))) / (n_vals * np.sum(sorted_vals)) - (n_vals + 1) / n_vals
    return gini
 
gini_maximin = gini_coefficient(results["per_capita_alloc"].values)
gini_fema    = gini_coefficient(results["fema_per_capita"].values)
 
gini_df = pd.DataFrame({
    "Metric": ["Gini (Maximin)", "Gini (FEMA)"],
    "Value": [gini_maximin, gini_fema]
})
gini_df.to_csv("Results/Maximin/gini_coefficients.csv", index=False)
 
# Top 10 and Bottom 10 regions by allocation
top10 = results.nlargest(10, "per_capita_alloc")
top10.to_csv("Results/Maximin/top10_maximin.csv", index=False)
 
funded = results[results["y_i"] == 1]
bottom10 = funded.nsmallest(10, "per_capita_alloc")
bottom10.to_csv("Results/Maximin/bottom10_maximin.csv", index=False)

# Excluded regions (if any)
excluded = results[results["y_i"] == 0]
if len(excluded) > 0:
    excluded.to_csv("Results/Maximin/excluded_regions.csv", index=False)