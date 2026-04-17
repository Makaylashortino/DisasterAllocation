import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
 
# Output folder
OUTPUT = Path("Results/Iterative")
OUTPUT.mkdir(parents=True, exist_ok=True)
 
# Load data 
df = pd.read_csv("Data/ModelingInput.csv", low_memory=False)
B  = df["ia_totalApprovedIhp"].sum()
 
increment_pct  = 0.01
remaining_need = df["d_i"].values.copy()
allocation     = [0.0] * len(df)
remaining_budget = B
raw_applicants = df["ia_validRegistrations"].values.copy()
d_i            = df["d_i"].values.copy()
 
# Floor pass: every region gets 10% of need guaranteed
for i in range(len(df)):
    floor = min(0.10 * d_i[i], remaining_budget)
    allocation[i]     += floor
    remaining_need[i] -= floor
    remaining_budget  -= floor
 
# Main loop
iteration      = 0
max_iterations = 1_000_000
 
while remaining_budget > 0.01 and iteration < max_iterations:
 
    priority = []
    for i in range(len(df)):
        if remaining_need[i] > 0 and raw_applicants[i] > 0:
            priority.append(remaining_need[i] / raw_applicants[i])
        else:
            priority.append(0.0)
 
    max_priority = max(priority)
    if max_priority <= 0:
        break
 
    candidates = [i for i, p in enumerate(priority) if abs(p - max_priority) < 1e-6]
    chosen     = max(candidates, key=lambda i: d_i[i])
 
    increment = increment_pct * remaining_need[chosen]
    increment = min(increment, remaining_budget, remaining_need[chosen])
    if increment <= 0:
        break
 
    allocation[chosen]     += increment
    remaining_need[chosen] -= increment
    remaining_budget       -= increment
    iteration += 1
 
# Build results dataframe 
results = df.copy()
results["x_i"]                   = allocation
results["pct_need_met"]          = results["x_i"] / results["d_i"] * 100
results["per_applicant_alloc"]   = results["x_i"] / results["ia_validRegistrations"]
results["fema_per_applicant"]    = results["ia_totalApprovedIhp"] / results["ia_validRegistrations"]
results["starting_need_per_app"] = results["d_i"] / results["ia_validRegistrations"]
results["fema_remaining_per_app"]  = ((results["d_i"] - results["ia_totalApprovedIhp"]) / results["ia_validRegistrations"]).clip(lower=0)
results["model_remaining_per_app"] = ((results["d_i"] - results["x_i"]) / results["ia_validRegistrations"]).clip(lower=0)
 
# Gini 
def gini(values):
    v = sorted([x for x in values if x >= 0])
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0
    return sum((2*(i+1) - n - 1) * vi for i, vi in enumerate(v)) / (n * sum(v))
 
# Disaster labels 
DR_NAMES = {
    4332: "Harvey - Texas",
    4335: "Irma - USVI",
    4336: "Irma - Puerto Rico",
    4337: "Irma - Florida",
    4338: "Irma - Georgia",
    4339: "Maria - Puerto Rico",
    4340: "Maria - USVI",
    4346: "Irma - South Carolina",
}
 

# CSV 1: Region-level results
results.to_csv(OUTPUT / "region_results.csv", index=False)
 

# CSV 2: Disaster-level summary
summary_rows = []
for dr, label in DR_NAMES.items():
    sub = results[results["disasterNumber"] == dr]
    if len(sub) == 0:
        continue
    total_need  = sub["d_i"].sum()
    total_fema  = sub["ia_totalApprovedIhp"].sum()
    total_alloc = sub["x_i"].sum()
    summary_rows.append({
        "disaster":               label,
        "regions":                len(sub),
        "total_need":             round(total_need, 2),
        "total_fema_actual":      round(total_fema, 2),
        "total_model_alloc":      round(total_alloc, 2),
        "fema_pct_need_met":      round(total_fema / total_need * 100, 1),
        "model_pct_need_met":     round(total_alloc / total_need * 100, 1),
        "avg_per_app_fema":       round(sub["fema_per_applicant"].mean(), 2),
        "avg_per_app_model":      round(sub["per_applicant_alloc"].mean(), 2),
        "starting_need_per_app":  round(sub["starting_need_per_app"].mean(), 2),
        "fema_remaining_per_app": round(sub["fema_remaining_per_app"].mean(), 2),
        "model_remaining_per_app":round(sub["model_remaining_per_app"].mean(), 2),
    })
 
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT / "disaster_summary.csv", index=False)
 

# CSV 3: Overall stats

std_start = results["starting_need_per_app"].std()
std_fema  = results["fema_remaining_per_app"].std()
std_model = results["model_remaining_per_app"].std()
 
overall = pd.DataFrame([{
    "total_budget":          round(B, 2),
    "total_allocated":       round(results["x_i"].sum(), 2),
    "total_need":            round(results["d_i"].sum(), 2),
    "overall_pct_need_met":  round(results["x_i"].sum() / results["d_i"].sum() * 100, 1),
    "regions_funded":        sum(1 for a in allocation if a > 0),
    "total_regions":         len(df),
    "iterations":            iteration,
    "gini_model":            round(gini(results["per_applicant_alloc"].tolist()), 4),
    "gini_fema":             round(gini(results["fema_per_applicant"].tolist()), 4),
    "std_dev_before":        round(std_start, 2),
    "std_dev_after_fema":    round(std_fema, 2),
    "std_dev_after_model":   round(std_model, 2),
    "std_dev_pct_change_fema":  round((std_fema - std_start) / std_start * 100, 1),
    "std_dev_pct_change_model": round((std_model - std_start) / std_start * 100, 1),
}])
overall.to_csv(OUTPUT / "overall_stats.csv", index=False)
 
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: % of need met — model vs FEMA by disaster
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
labels_g1 = summary_df["disaster"].tolist()
x1        = np.arange(len(labels_g1))
w         = 0.35
 
b1 = ax.bar(x1 - w/2, summary_df["fema_pct_need_met"],  w, color="#378ADD", label="FEMA actual")
b2 = ax.bar(x1 + w/2, summary_df["model_pct_need_met"], w, color="#1D9E75", label="Iterative model")
 
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
 
ax.set_xticks(x1)
ax.set_xticklabels(labels_g1, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("% of adjusted need met", fontsize=10)
ax.set_title("Percentage of adjusted need met: FEMA actual vs iterative model", fontsize=11, fontweight="normal")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT / "graph1_pct_need_met.png", dpi=150, bbox_inches="tight")
plt.close()
 
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Per applicant allocation — model vs FEMA by disaster
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
b1 = ax.bar(x1 - w/2, summary_df["avg_per_app_fema"],  w, color="#378ADD", label="FEMA actual")
b2 = ax.bar(x1 + w/2, summary_df["avg_per_app_model"], w, color="#1D9E75", label="Iterative model")
 
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 50:
        ax.annotate(f"${h:,.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)
 
ax.set_xticks(x1)
ax.set_xticklabels(labels_g1, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Dollars per applicant ($)", fontsize=10)
ax.set_title("Average allocation per applicant: FEMA actual vs iterative model", fontsize=11, fontweight="normal")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT / "graph2_per_applicant.png", dpi=150, bbox_inches="tight")
plt.close()
 
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3: Remaining need convergence
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
x3    = np.arange(len(summary_df))
w3    = 0.25
 
bars1 = ax.bar(x3 - w3, summary_df["starting_need_per_app"],   w3, color="#B4B2A9", label="Starting need")
bars2 = ax.bar(x3,      summary_df["fema_remaining_per_app"],  w3, color="#378ADD", label="After FEMA")
bars3 = ax.bar(x3 + w3, summary_df["model_remaining_per_app"], w3, color="#1D9E75", label="After iterative model")
 
def annotate_bars(bars, color):
    for bar in bars:
        h = bar.get_height()
        if h > 50:
            ax.annotate(f"${h:,.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=color)
 
annotate_bars(bars1, "#5F5E5A")
annotate_bars(bars2, "#185FA5")
annotate_bars(bars3, "#0F6E56")
 
avg_model_remaining = summary_df["model_remaining_per_app"].mean()
ax.axhline(avg_model_remaining, color="#1D9E75", linewidth=1.2, linestyle="--", alpha=0.7)
ax.annotate(f"Model convergence ≈ ${avg_model_remaining:,.0f}",
            xy=(len(summary_df) - 0.5, avg_model_remaining),
            xytext=(-8, 6), textcoords="offset points",
            ha="right", fontsize=9, color="#0F6E56")
 
ax.set_xticks(x3)
ax.set_xticklabels(summary_df["disaster"].tolist(), rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Remaining need per applicant ($)", fontsize=10)
ax.set_title("Remaining need per applicant: before, after FEMA, and after iterative model", fontsize=11, fontweight="normal")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
 
legend_patches = [
    mpatches.Patch(color="#B4B2A9", label="Starting need"),
    mpatches.Patch(color="#378ADD", label="After FEMA"),
    mpatches.Patch(color="#1D9E75", label="After iterative model"),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=False, loc="upper right")
 
annotation = (
    f"Std dev of remaining need:\n"
    f"  Before:      ${std_start:,.0f}\n"
    f"  After FEMA:  ${std_fema:,.0f}  ({(std_fema-std_start)/std_start*100:+.0f}%)\n"
    f"  After model: ${std_model:,.0f}  ({(std_model-std_start)/std_start*100:+.0f}%)"
)
ax.text(0.01, 0.97, annotation, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF3DE", edgecolor="#639922", alpha=0.8))
 
plt.tight_layout()
plt.savefig(OUTPUT / "graph3_remaining_need.png", dpi=150, bbox_inches="tight")
plt.close()
 
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 4: Gini comparison
# ══════════════════════════════════════════════════════════════════════════════
model_gini = gini(results["per_applicant_alloc"].tolist())
fema_gini  = gini(results["fema_per_applicant"].tolist())
 
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
 
bars = ax.bar(["FEMA actual", "Iterative model"], [fema_gini, model_gini],
              color=["#378ADD", "#1D9E75"], width=0.4)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10)
 
ax.set_ylabel("Gini coefficient", fontsize=10)
ax.set_title("Gini coefficient: FEMA vs iterative model\n(measured on per-applicant allocation)", fontsize=10, fontweight="normal")
ax.set_ylim(0, max(fema_gini, model_gini) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.5, 0.05,
        "Note: higher Gini here reflects equity-based\nallocation, not unfairness",
        transform=ax.transAxes, ha="center", fontsize=8,
        color="#5F5E5A", style="italic")
plt.tight_layout()
plt.savefig(OUTPUT / "graph4_gini.png", dpi=150, bbox_inches="tight")
plt.close()
 
print(f"All outputs saved to {OUTPUT}/")
print(f"  region_results.csv")
print(f"  disaster_summary.csv")
print(f"  overall_stats.csv")
print(f"  graph1_pct_need_met.png")
print(f"  graph2_per_applicant.png")
print(f"  graph3_remaining_need.png")
print(f"  graph4_gini.png")