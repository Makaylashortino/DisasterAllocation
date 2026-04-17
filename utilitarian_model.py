"""
PROBLEM CONTEXT:
    We are reallocating $4.124 billion in FEMA Individual Assistance (IA) across 204 disaster-affected 
    regions from Hurricanes Harvey, Irma, and Maria. The goal is to maximize total weighted societal benefit, 
    where each region's weight is its need-per-applicant (n_i).
 
MATHEMATICAL FORMULATION:
    Sets & Indices
        I = {1, 2, ..., 204} set of regions (counties / municipios / islands)
 
    Data / Parameters (all read from ModelingInput.csv)
        n_i = need per applicant for region i      (= d_i / a_i)
        d_i = adjusted total need for region i  (dollars)
        a_i = adjusted applicants in region i
        B   = total budget = sum of ia_totalApprovedIhp = $4,124,000,952.74
 
    Decision Variables
        x_i  ∈ R+    continuous — dollars allocated to region i
        y_i  ∈ {0,1} binary     — 1 if region i receives any aid, 0 otherwise

    Objective (Utilitarian — maximize weighted benefit)
        max  Σ_{i ∈ I} n_i · x_i
        - We want to maximize the total benefit across all regions, where each dollar of aid is weighted by 
          how badly that region needs it on a per-applicant basis. Regions with higher need-per-applicant get a higher 
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

Important notes on data preprocessing:
    Several regions (mainly in Puerto Rico and USVI) appear multiple times in the raw data because they were struck by 
    more than one hurricane We aggregate by (state, county) so that each community is represented exactly once in the model,
    with its total need and applicants across all disasters. This is important to ensure that the model allocates aid at the 
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
    applicants=("ia_validRegistrations", "sum"),
    num_disasters=("disasterNumber", "count"),
    disaster_list=("disasterNumber", lambda x: list(x))
).reset_index()
 
# Recompute per-applicant metrics on the aggregated data
df["n_i"] = df["d_i"] / df["applicants"]
df["fema_per_applicant"] = df["ia_totalApprovedIhp"] / df["applicants"]  # per capita based on applicants
 
# Handle any regions with 0 population (shouldn't happen but be safe)
df["n_i"] = df["n_i"].fillna(0)
df["fema_per_applicant"] = df["fema_per_applicant"].fillna(0)
 
# Parameters
n = len(df)                                        # 183 unique regions
B = df["ia_totalApprovedIhp"].sum()                # total budget
d_i = df["d_i"].values                             # adjusted need per region
n_i = df["n_i"].values                             # need per applicant per region
app = df["applicants"].values                      # applicants per region
 
print("MODEL 1 — UTILITARIAN (Weighted Social Welfare Maximization)")
print(f"  Raw data rows     : {len(raw)}")
print(f"  After aggregation : {n} unique regions")
print(f"  Multi-disaster    : {(df['num_disasters'] > 1).sum()} regions hit by 2+ hurricanes")
print(f"  Total budget  (B) : ${B:,.2f}")
print(f"  Total need (Σd_i) : ${d_i.sum():,.2f}")
print(f"  Budget / Need     : {B / d_i.sum():.4f}  ({100*B/d_i.sum():.2f}% of total need)")
print()
 
# 2.  BUILD THE GUROBI MODEL
model = gp.Model("Utilitarian_Disaster_Relief")
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
 
model.update()
 
# Objective: max Σ n_i * x_i 
model.setObjective(
    gp.quicksum(n_i[i] * x[i] for i in range(n)),
    GRB.MAXIMIZE
)

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


# 3.  SOLVE
model.optimize()
if model.status != GRB.OPTIMAL:
    print(f"WARNING: Solver status = {model.status}")
 

# 4.  EXTRACT & ANALYZE RESULTS
results = df[["state", "county", "population", "applicants", "d_i", "n_i",
              "ia_totalApprovedIhp", "fema_per_applicant",
              "num_disasters", "disaster_list"]].copy()
 
results["x_i_utilitarian"]  = [x[i].X for i in range(n)]
results["y_i"]              = [int(round(y[i].X)) for i in range(n)]
results["pct_need_met"]     = results["x_i_utilitarian"] / results["d_i"] * 100
results["per_applicant_alloc"] = results["x_i_utilitarian"] / results["applicants"]

results.to_csv("Results/Utilitarian/utilitarian_results.csv", index=False)
 
# Summary statistics
total_alloc = results["x_i_utilitarian"].sum()
regions_funded = results["y_i"].sum()

summary_df = pd.DataFrame({
    "Metric": [
        "Total Allocated",
        "Budget",
        "Budget Used (%)",
        "Regions Funded",
        "Regions Excluded"
    ],
    "Value": [
        total_alloc,
        B,
        100 * total_alloc / B,
        regions_funded,
        n - regions_funded
    ]
})
summary_df.to_csv("Results/Utilitarian/utilitarian_summary.csv", index=False)

# Per-state summary
state_summary = results.groupby("state").agg(
    num_regions=("county", "count"),
    total_need=("d_i", "sum"),
    total_allocation=("x_i_utilitarian", "sum"),
    total_fema_actual=("ia_totalApprovedIhp", "sum"),
    total_pop=("population", "sum"),
    total_applicants=("applicants", "sum"),
    regions_funded=("y_i", "sum")
).reset_index()
 
state_summary["pct_need_met"]    = state_summary["total_allocation"] / state_summary["total_need"] * 100
state_summary["per_applicant_util"] = state_summary["total_allocation"] / state_summary["total_applicants"]
state_summary["per_applicant_fema"] = state_summary["total_fema_actual"] / state_summary["total_applicants"]

state_summary.to_csv("Results/Utilitarian/utilitarian_state_summary.csv", index=False)
 
# PR vs TX per-applicant ratio (key fairness metric)
pr_data = state_summary[state_summary["state"] == "PR"]
tx_data = state_summary[state_summary["state"] == "TX"]
 
if len(pr_data) > 0 and len(tx_data) > 0:
    pr_pc_util = pr_data["per_applicant_util"].values[0]
    tx_pc_util = tx_data["per_applicant_util"].values[0]
    pr_pc_fema = pr_data["per_applicant_fema"].values[0]
    tx_pc_fema = tx_data["per_applicant_fema"].values[0]
 
pr_tx_df = pd.DataFrame({
    "Metric": [
        "PR per applicant (Utilitarian)",
        "TX per applicant (Utilitarian)",
        "PR/TX ratio (Utilitarian)",
        "PR per applicant (FEMA)",
        "TX per applicant (FEMA)",
        "PR/TX ratio (FEMA)"
    ],
    "Value": [
        pr_pc_util,
        tx_pc_util,
        pr_pc_util / tx_pc_util,
        pr_pc_fema,
        tx_pc_fema,
        pr_pc_fema / tx_pc_fema
    ]
})
pr_tx_df.to_csv("Results/Utilitarian/pr_vs_tx_fairness.csv", index=False)

 
# Gini coefficient computation (on per-applicant allocation)
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
 
gini_util = gini_coefficient(results["per_applicant_alloc"].values)
gini_fema = gini_coefficient(results["fema_per_applicant"].values)
 
gini_df = pd.DataFrame({
    "Metric": ["Gini (Utilitarian)", "Gini (FEMA)"],
    "Value": [gini_util, gini_fema]
})
gini_df.to_csv("Results/Utilitarian/gini_coefficients.csv", index=False)
 
# Top 10 and Bottom 10 regions by allocation
top10 = results.nlargest(10, "per_applicant_alloc")
top10.to_csv("Results/Utilitarian/top10_utilitarian.csv", index=False)
 
funded = results[results["y_i"] == 1]
bottom10 = funded.nsmallest(10, "per_applicant_alloc")
bottom10.to_csv("Results/Utilitarian/bottom10_utilitarian.csv", index=False)

# Excluded regions (if any)
excluded = results[results["y_i"] == 0]
if len(excluded) > 0:
    excluded.to_csv("Results/Utilitarian/excluded_regions.csv", index=False)


#####Graphs and visualizations########
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# REMAINING NEED ANALYSIS — UTILITARIAN MODEL
# Groups by state (aggregated model has no disaster-level rows)
# ══════════════════════════════════════════════════════════════════════════════

# Compute remaining need per applicant for three scenarios
results["starting_need_per_app"]   = results["d_i"] / results["applicants"]
results["fema_remaining_per_app"]  = (results["d_i"] - results["ia_totalApprovedIhp"]) / results["applicants"]
results["model_remaining_per_app"] = (results["d_i"] - results["x_i_utilitarian"]) / results["applicants"]

# Clip negatives to zero (fully funded regions)
results["fema_remaining_per_app"]  = results["fema_remaining_per_app"].clip(lower=0)
results["model_remaining_per_app"] = results["model_remaining_per_app"].clip(lower=0)

# ── State grouping ─────────────────────────────────────────────────────────
# State order: largest to smallest by total need, PR and VI last for emphasis
STATE_LABELS = {
    "TX": "Harvey\nTexas",
    "FL": "Irma\nFlorida",
    "GA": "Irma\nGeorgia",
    "PR": "Irma + Maria\nPuerto Rico",
    "VI": "Irma + Maria\nUSVI",
}

# ── Print summary table ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("REMAINING NEED PER APPLICANT — BEFORE vs AFTER ALLOCATION (BY STATE)")
print(f"{'='*80}")
print(f"{'State':<22} {'Starting':>12} {'After FEMA':>12} {'After Model':>12} {'FEMA Gap':>10} {'Model Gap':>10}")
print(f"{'-'*80}")

for state, label in STATE_LABELS.items():
    sub = results[results["state"] == state]
    if len(sub) == 0:
        continue
    # Use sum-weighted averages: aggregate dollars then divide
    total_app   = sub["applicants"].sum()
    start       = sub["d_i"].sum() / total_app
    fema_r      = max((sub["d_i"].sum() - sub["ia_totalApprovedIhp"].sum()) / total_app, 0)
    mod_r       = max((sub["d_i"].sum() - sub["x_i_utilitarian"].sum()) / total_app, 0)
    print(f"{label.replace(chr(10), ' '):<22} ${start:>10,.0f} ${fema_r:>10,.0f} ${mod_r:>10,.0f} "
          f"{fema_r/start*100:>9.1f}% {mod_r/start*100:>9.1f}%")

# ── Convergence metrics ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("CONVERGENCE METRICS (std deviation of remaining need per applicant)")
print(f"{'='*80}")

std_start = results["starting_need_per_app"].std()
std_fema  = results["fema_remaining_per_app"].std()
std_model = results["model_remaining_per_app"].std()

print(f"  Std dev before allocation:   ${std_start:>10,.2f}")
print(f"  Std dev after FEMA:          ${std_fema:>10,.2f}  (change: {(std_fema-std_start)/std_start*100:+.1f}%)")
print(f"  Std dev after model:         ${std_model:>10,.2f}  (change: {(std_model-std_start)/std_start*100:+.1f}%)")
print(f"\n  Lower std dev = more equal remaining need across regions")
print(f"  Model {'MORE' if std_model < std_fema else 'LESS'} equal than FEMA in remaining need")

# ── Build chart arrays ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — % OF NEED MET: MODEL VS FEMA BY STATE
# ══════════════════════════════════════════════════════════════════════════════

labels_g1  = []
fema_pct   = []
model_pct  = []

for state, label in STATE_LABELS.items():
    sub = results[results["state"] == state]
    if len(sub) == 0:
        continue
    total_need  = sub["d_i"].sum()
    labels_g1.append(label)
    fema_pct.append(sub["ia_totalApprovedIhp"].sum() / total_need * 100)
    model_pct.append(sub["x_i_utilitarian"].sum() / total_need * 100)

x     = np.arange(len(labels_g1))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars1 = ax.bar(x - width/2, fema_pct,  width, label="FEMA",        color="#CC3333", zorder=3)
bars2 = ax.bar(x + width/2, model_pct, width, label="Utilitarian",  color="#3366CC", zorder=3)

def label_bars_pct(bars, color):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color=color
        )

label_bars_pct(bars1, "#991111")
label_bars_pct(bars2, "#1A3D99")

ax.set_xticks(x)
ax.set_xticklabels(labels_g1, fontsize=10)
ax.set_ylabel("% of adjusted need met", fontsize=11)
ax.set_title(
    "% of need met: FEMA vs utilitarian model by state",
    fontsize=12, fontweight="normal", pad=14
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#CC3333", label="FEMA"),
    mpatches.Patch(color="#3366CC", label="Utilitarian model"),
]
ax.legend(handles=legend_patches, fontsize=10, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)

plt.tight_layout()
plt.savefig("Results/Charts/UTIL_pct_need_met.jpeg", dpi=150,
            bbox_inches="tight", format="jpeg")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — PER-APPLICANT ALLOCATION: MODEL VS FEMA BY STATE
# ══════════════════════════════════════════════════════════════════════════════

labels_g2      = []
fema_per_app   = []
model_per_app  = []

for state, label in STATE_LABELS.items():
    sub = results[results["state"] == state]
    if len(sub) == 0:
        continue
    total_app = sub["applicants"].sum()
    labels_g2.append(label)
    fema_per_app.append(sub["ia_totalApprovedIhp"].sum() / total_app)
    model_per_app.append(sub["x_i_utilitarian"].sum() / total_app)

x     = np.arange(len(labels_g2))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars1 = ax.bar(x - width/2, fema_per_app,  width, label="FEMA",       color="#CC3333", zorder=3)
bars2 = ax.bar(x + width/2, model_per_app, width, label="Utilitarian", color="#3366CC", zorder=3)

def label_bars_dollar(bars, color):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"${h:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color=color
        )

label_bars_dollar(bars1, "#991111")
label_bars_dollar(bars2, "#1A3D99")

ax.set_xticks(x)
ax.set_xticklabels(labels_g2, fontsize=10)
ax.set_ylabel("Dollars per applicant ($)", fontsize=11)
ax.set_title(
    "Per-applicant allocation: FEMA vs utilitarian model by state",
    fontsize=12, fontweight="normal", pad=14
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#CC3333", label="FEMA"),
    mpatches.Patch(color="#3366CC", label="Utilitarian model"),
]
ax.legend(handles=legend_patches, fontsize=10, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)

plt.tight_layout()
plt.savefig("Results/Charts/UTIL_per_applicant_allocation.jpeg", dpi=150,
            bbox_inches="tight", format="jpeg")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — REMAINING NEED PER APPLICANT: BEFORE vs AFTER FEMA vs AFTER MODEL
# ══════════════════════════════════════════════════════════════════════════════
# ── Build chart arrays ─────────────────────────────────────────────────────
labels  = []
start   = []
fema_r  = []
model_r = []

for state, label in STATE_LABELS.items():
    sub = results[results["state"] == state]
    if len(sub) == 0:
        continue
    total_app = sub["applicants"].sum()
    labels.append(label)
    start.append(sub["d_i"].sum() / total_app)
    fema_r.append(max((sub["d_i"].sum() - sub["ia_totalApprovedIhp"].sum()) / total_app, 0))
    model_r.append(max((sub["d_i"].sum() - sub["x_i_utilitarian"].sum()) / total_app, 0))

# ── Chart setup ───────────────────────────────────────────────────────────
x     = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Bars ──────────────────────────────────────────────────────────────────
bars1 = ax.bar(x - width, start,   width, label="Starting need",  color="#B4B2A9", zorder=3)
bars2 = ax.bar(x,         fema_r,  width, label="After FEMA",     color="#CC3333", zorder=3)
bars3 = ax.bar(x + width, model_r, width, label="After model",    color="#3366CC", zorder=3)

# ── Value labels ──────────────────────────────────────────────────────────
def label_bars(bars, color):
    for bar in bars:
        h = bar.get_height()
        if h > 50:
            ax.annotate(
                f"${h:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color=color
            )

label_bars(bars1, "#5F5E5A")
label_bars(bars2, "#991111")
label_bars(bars3, "#1A3D99")

# ── Convergence line ──────────────────────────────────────────────────────
avg_model = np.mean(model_r)
ax.axhline(avg_model, color="#3366CC", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
ax.annotate(
    f"Model convergence ≈ ${avg_model:,.0f}",
    xy=(len(labels) - 0.5, avg_model),
    xytext=(-8, 6),
    textcoords="offset points",
    ha="right", fontsize=9, color="#1A3D99"
)

# ── Axes and labels ───────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Remaining need per applicant ($)", fontsize=11)
ax.set_title(
    "Remaining need per applicant: before, after FEMA, and after utilitarian model",
    fontsize=12, fontweight="normal", pad=14
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="#B4B2A9", label="Starting need"),
    mpatches.Patch(color="#CC3333", label="After FEMA"),
    mpatches.Patch(color="#3366CC", label="After utilitarian model"),
]
ax.legend(handles=legend_patches, fontsize=10, frameon=False, 
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)

# ── Std dev annotation box ────────────────────────────────────────────────
annotation = (
    f"Std dev of remaining need:\n"
    f"  Before:      ${std_start:,.0f}\n"
    f"  After FEMA:  ${std_fema:,.0f}  ({(std_fema-std_start)/std_start*100:+.0f}%)\n"
    f"  After model: ${std_model:,.0f}  ({(std_model-std_start)/std_start*100:+.0f}%)"
)
ax.text(
    0.01, 0.97, annotation,
    transform=ax.transAxes,
    fontsize=8.5, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#DDEEFF", edgecolor="#3366CC", alpha=0.8)
)

plt.tight_layout()
plt.savefig("Results/Charts/UTIL_remaining_need_convergence.jpeg", dpi=150,
            bbox_inches="tight", format="jpeg")
plt.show()


# ══════════════════════════════════
# ════════════════════════════════════════════
# GRAPH 4 — GINI COEFFICIENT: MODEL VS FEMA
# ══════════════════════════════════════════════════════════════════════════════

gini_util = gini_coefficient(results["per_applicant_alloc"].values)
gini_fema = gini_coefficient(results["fema_per_applicant"].values)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(
    ["FEMA", "Utilitarian model"],
    [gini_fema, gini_util],
    color=["#CC3333", "#3366CC"],
    width=0.4,
    zorder=3
)

for bar, val in zip(bars, [gini_fema, gini_util]):
    ax.annotate(
        f"{val:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color=bar.get_facecolor()
    )

ax.set_ylabel("Gini coefficient", fontsize=11)
ax.set_title(
    "Gini coefficient: FEMA vs utilitarian model\n(per-applicant allocation)",
    fontsize=12, fontweight="normal", pad=14
)
ax.set_ylim(0, max(gini_fema, gini_util) * 1.3)
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Explanatory note
ax.text(
    0.5, 0.15,
    "Note: A higher Gini in the utilitarian model reflects\n"
    "equity (routing dollars to highest-need regions),\n"
    "not inequality.",
    transform=ax.transAxes,
    fontsize=9, ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#DDEEFF", edgecolor="#3366CC", alpha=0.8)
)

plt.tight_layout()
plt.savefig("Results/Charts/UTIL_gini_comparison.jpeg", dpi=150,
            bbox_inches="tight", format="jpeg")
plt.show()