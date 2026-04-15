import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#d1d5db',
    'axes.labelcolor': '#374151',
    'text.color': '#1f2937',
    'xtick.color': '#374151',
    'ytick.color': '#374151',
    'grid.color': '#e5e7eb',
    'grid.alpha': 0.7,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#d1d5db',
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

# Red-themed accent palette
FEMA_C  = '#dc2626'   # bold red — primary accent / FEMA
UTIL_C  = '#2563eb'   # blue — utilitarian
MAXI_C  = '#059669'   # green — maximin
ACCENT  = '#dc2626'   # red accent for highlights
MUTED   = '#6b7280'   # grey for secondary text
BG      = '#ffffff'
CARD    = '#f9fafb'

STATE_C = {'PR': '#7c3aed', 'TX': '#ea580c', 'FL': '#0891b2', 'VI': '#db2777', 'GA': '#6b7280', 'SC': '#6b7280'}

def save(fig, name):
    fig.savefig(f"/Results/Charts/{name}.jpg", format='jpeg', facecolor=fig.get_facecolor())
    plt.close(fig)

util = pd.read_csv("/Results/Utilitarian/utilitarian_results.csv")
maxi = pd.read_csv("/Results/Maximin/maximin_results.csv")

m = util[["state","county","population","d_i","n_i","ia_totalApprovedIhp","fema_per_capita"]].copy()
m["util_alloc"] = util["x_i_utilitarian"]
m["util_pc"] = util["per_capita_alloc"]
m["util_y"] = util["y_i"]
m["maxi_alloc"] = maxi["x_i_maximin"]
m["maxi_pc"] = maxi["per_capita_alloc"]
m["maxi_y"] = maxi["y_i"]

st = m.groupby("state").agg(
    pop=("population","sum"), need=("d_i","sum"),
    fema_total=("ia_totalApprovedIhp","sum"),
    util_total=("util_alloc","sum"), maxi_total=("maxi_alloc","sum"),
    n_reg=("county","count"), util_funded=("util_y","sum"), maxi_funded=("maxi_y","sum")
).reset_index()
st["fema_pc"] = st["fema_total"]/st["pop"]
st["util_pc"] = st["util_total"]/st["pop"]
st["maxi_pc"] = st["maxi_total"]/st["pop"]
st["fema_pct"] = st["fema_total"]/st["need"]*100
st["util_pct"] = st["util_total"]/st["need"]*100
st["maxi_pct"] = st["maxi_total"]/st["need"]*100

order = ["VI","PR","TX","FL","GA"]
labels = ["FEMA Actual", "Utilitarian", "Maximin"]
colors = [FEMA_C, UTIL_C, MAXI_C]

def gini(values):
    v = np.array(values, dtype=float)
    v = v[v > 0]
    if len(v) == 0: return np.nan
    s = np.sort(v)
    nn = len(s)
    return (2*np.sum((np.arange(1,nn+1)*s)))/(nn*np.sum(s)) - (nn+1)/nn

g_fema = gini(m["fema_per_capita"].values)
g_util = gini(m["util_pc"].values)
g_maxi = gini(m["maxi_pc"].values)

pr_fema = st[st.state=="PR"]["fema_pc"].values[0]
tx_fema = st[st.state=="TX"]["fema_pc"].values[0]
pr_util = st[st.state=="PR"]["util_pc"].values[0]
tx_util = st[st.state=="TX"]["util_pc"].values[0]
pr_maxi = st[st.state=="PR"]["maxi_pc"].values[0]
tx_maxi = st[st.state=="TX"]["maxi_pc"].values[0]
ratios = [pr_fema/tx_fema, pr_util/tx_util, pr_maxi/tx_maxi]


# CHART 1: Per-Capita Allocation by State
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(order))
w = 0.25

bars1 = ax.bar(x - w, [st[st.state==s]["fema_pc"].values[0] for s in order], w, label="FEMA Actual", color=FEMA_C, zorder=3)
bars2 = ax.bar(x,     [st[st.state==s]["util_pc"].values[0] for s in order], w, label="Utilitarian", color=UTIL_C, zorder=3)
bars3 = ax.bar(x + w, [st[st.state==s]["maxi_pc"].values[0] for s in order], w, label="Maximin",     color=MAXI_C, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=13, fontweight='bold')
ax.set_ylabel("Dollars per Person")
ax.set_title("Per-Capita Allocation by State", pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 15, f'${h:,.0f}', ha='center', va='bottom', fontsize=8, color=MUTED)

save(fig, "01_per_capita_by_state")


# CHART 2: Budget Distribution by State (Billions)
fig, ax = plt.subplots(figsize=(10, 6))
fema_b = [st[st.state==s]["fema_total"].values[0]/1e9 for s in order]
util_b = [st[st.state==s]["util_total"].values[0]/1e9 for s in order]
maxi_b = [st[st.state==s]["maxi_total"].values[0]/1e9 for s in order]

bars1 = ax.bar(x - w, fema_b, w, label="FEMA Actual", color=FEMA_C, zorder=3)
bars2 = ax.bar(x,     util_b, w, label="Utilitarian", color=UTIL_C, zorder=3)
bars3 = ax.bar(x + w, maxi_b, w, label="Maximin",     color=MAXI_C, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=13, fontweight='bold')
ax.set_ylabel("Billions ($)")
ax.set_title("Total Budget Allocation by State", pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:.1f}B'))

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'${h:.2f}B', ha='center', va='bottom', fontsize=8, color=MUTED)

save(fig, "02_budget_by_state")


# CHART 3: Percent of Need Met by State
fig, ax = plt.subplots(figsize=(10, 6))
order2 = ["PR","TX","FL","VI","GA"]
x2 = np.arange(len(order2))

ax.bar(x2 - w, [st[st.state==s]["fema_pct"].values[0] for s in order2], w, label="FEMA Actual", color=FEMA_C, zorder=3)
ax.bar(x2,     [st[st.state==s]["util_pct"].values[0] for s in order2], w, label="Utilitarian", color=UTIL_C, zorder=3)
ax.bar(x2 + w, [st[st.state==s]["maxi_pct"].values[0] for s in order2], w, label="Maximin",     color=MAXI_C, zorder=3)

ax.set_xticks(x2)
ax.set_xticklabels(order2, fontsize=13, fontweight='bold')
ax.set_ylabel("% of Adjusted Need Met")
ax.set_title("Percent of Need Met by State", pad=15)
ax.set_ylim(0, 110)
ax.axhline(y=100, color=ACCENT, linestyle='--', alpha=0.4, linewidth=1)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))

save(fig, "03_pct_need_met")


# CHART 4: Gini Coefficient Comparison
fig, ax = plt.subplots(figsize=(8, 5))
vals = [g_fema, g_util, g_maxi]
bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3, edgecolor='white', linewidth=0.5)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}', ha='center', fontsize=13, fontweight='bold', color='#1f2937')

ax.set_ylabel("Gini Coefficient")
ax.set_title("Gini Coefficient — Per-Capita Allocation Inequality", pad=15)
ax.set_ylim(0, 0.7)
ax.grid(axis='y', linestyle='--')
ax.text(0.98, 0.95, "Lower = More Equal", transform=ax.transAxes, ha='right', va='top', fontsize=10, color=ACCENT, fontstyle='italic')

save(fig, "04_gini_comparison")

# CHART 5: PR vs TX Per-Capita Ratio
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, ratios, color=colors, width=0.5, zorder=3, edgecolor='white', linewidth=0.5)

for bar, v in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}x', ha='center', fontsize=14, fontweight='bold', color='#1f2937')

ax.set_ylabel("PR / TX Per-Capita Ratio")
ax.set_title("PR vs TX — Per-Capita Allocation Ratio", pad=15)
ax.grid(axis='y', linestyle='--')
ax.axhline(y=1, color=MUTED, linestyle=':', alpha=0.4)
ax.text(0.98, 0.95, "Higher = PR gets proportionally more", transform=ax.transAxes, ha='right', va='top', fontsize=10, color=ACCENT, fontstyle='italic')

save(fig, "05_pr_tx_ratio")


# CHART 6: Lorenz Curves
fig, ax = plt.subplots(figsize=(8, 8))

for label, col, color, ls, lw in [
    ("FEMA Actual", "fema_per_capita", FEMA_C, '-', 2.5),
    ("Utilitarian", "util_pc", UTIL_C, '-', 2.5),
    ("Maximin", "maxi_pc", MAXI_C, '--', 2.5),
]:
    vals = np.sort(m[col].values)
    vals = vals[vals > 0]
    cum = np.cumsum(vals) / vals.sum()
    x_pct = np.arange(1, len(vals)+1) / len(vals) * 100
    cum_pct = cum * 100
    ax.plot(np.concatenate([[0], x_pct]), np.concatenate([[0], cum_pct]),
            color=color, linewidth=lw, label=label, linestyle=ls)

ax.plot([0, 100], [0, 100], color=MUTED, linestyle=':', linewidth=1.5, label='Perfect Equality', alpha=0.5)
ax.fill_between([0, 100], [0, 100], 0, alpha=0.03, color=MUTED)

ax.set_xlabel("Cumulative % of Regions (lowest to highest per-capita)", fontsize=11)
ax.set_ylabel("Cumulative % of Total Allocation", fontsize=11)
ax.set_title("Lorenz Curve — Per-Capita Allocation Inequality", pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.grid(True, linestyle='--')

save(fig, "06_lorenz_curves")


# CHART 7: Regions Funded vs Excluded
fema_funded = len(m[m["ia_totalApprovedIhp"] > 0])

fig, ax = plt.subplots(figsize=(9, 4))
models = ["FEMA Actual", "Utilitarian", "Maximin"]
funded = [fema_funded, int(m["util_y"].sum()), int(m["maxi_y"].sum())]
excluded = [183 - f for f in funded]
y_pos = np.arange(len(models))

ax.barh(y_pos, funded, height=0.5, label="Funded", color=[FEMA_C, UTIL_C, MAXI_C], zorder=3)
ax.barh(y_pos, excluded, height=0.5, left=funded, label="Excluded", color='#d1d5db', zorder=3)

for i, (f, e) in enumerate(zip(funded, excluded)):
    ax.text(f/2, i, f'{f}', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    if e > 5:
        ax.text(f + e/2, i, f'{e}', ha='center', va='center', fontsize=13, fontweight='bold', color=MUTED)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=13, fontweight='bold')
ax.set_xlabel("Number of Regions")
ax.set_title("Regions Funded vs Excluded (out of 183)", pad=15)
ax.set_xlim(0, 195)
ax.grid(axis='x', linestyle='--')

save(fig, "07_regions_funded")


# CHART 8: Scatter — Need vs FEMA Allocation
fig, ax = plt.subplots(figsize=(9, 7))

for state in ["PR","TX","FL","VI"]:
    sub = m[(m["state"] == state) & (m["fema_per_capita"] > 0)]
    ax.scatter(sub["n_i"], sub["fema_per_capita"], c=STATE_C[state], s=40, alpha=0.7, label=state, zorder=3, edgecolors='white', linewidth=0.3)

maxval = max(m["n_i"].max(), m["fema_per_capita"].max()) * 1.05
ax.plot([0, maxval], [0, maxval], color=ACCENT, linestyle='--', linewidth=1.5, alpha=0.4, label='100% need met')

ax.set_xlabel("Need per Capita (n_i)")
ax.set_ylabel("FEMA Allocation per Capita ($)")
ax.set_title("Need vs FEMA Actual Allocation (Per Capita)", pad=15)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "08_scatter_fema")


# CHART 9: Scatter — Need vs Utilitarian Allocation
fig, ax = plt.subplots(figsize=(9, 7))

for state in ["PR","TX","FL","VI"]:
    sub = m[(m["state"] == state) & (m["util_pc"] > 0)]
    ax.scatter(sub["n_i"], sub["util_pc"], c=STATE_C[state], s=40, alpha=0.7, label=state, zorder=3, edgecolors='white', linewidth=0.3)

maxval = max(m["n_i"].max(), m["util_pc"].max()) * 1.05
ax.plot([0, maxval], [0, maxval], color=ACCENT, linestyle='--', linewidth=1.5, alpha=0.4, label='100% need met')

ax.set_xlabel("Need per Capita (n_i)")
ax.set_ylabel("Utilitarian Allocation per Capita ($)")
ax.set_title("Need vs Utilitarian Allocation (Per Capita)", pad=15)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "09_scatter_utilitarian")


# CHART 10: Scatter — Need vs Maximin Allocation
fig, ax = plt.subplots(figsize=(9, 7))

for state in ["PR","TX","FL","VI"]:
    sub = m[(m["state"] == state) & (m["maxi_pc"] > 0)]
    ax.scatter(sub["n_i"], sub["maxi_pc"], c=STATE_C[state], s=40, alpha=0.7, label=state, zorder=3, edgecolors='white', linewidth=0.3)

maxval = max(m["n_i"].max(), m["maxi_pc"].max()) * 1.05
ax.plot([0, maxval], [0, maxval], color=ACCENT, linestyle='--', linewidth=1.5, alpha=0.4, label='100% need met')
ax.axhline(y=465.68, color=MAXI_C, linestyle=':', linewidth=2, alpha=0.7)
ax.text(100, 520, 'z* = $465.68 (min floor)', fontsize=10, color=MAXI_C, fontstyle='italic')

ax.set_xlabel("Need per Capita (n_i)")
ax.set_ylabel("Maximin Allocation per Capita ($)")
ax.set_title("Need vs Maximin Allocation (Per Capita)", pad=15)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "10_scatter_maximin")


# CHART 11: Top 10 Regions — Utilitarian
top_util = m[m["util_y"]==1].nlargest(10, "util_pc")[["county","state","util_pc","n_i"]].reset_index(drop=True)
top_util["label"] = top_util["county"] + ", " + top_util["state"]
top_util = top_util.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [STATE_C.get(s, MUTED) for s in top_util["state"]]
bars = ax.barh(top_util["label"], top_util["util_pc"], color=bar_colors, height=0.6, zorder=3, edgecolor='white', linewidth=0.3)
for bar, v in zip(bars, top_util["util_pc"]):
    ax.text(v + 30, bar.get_y() + bar.get_height()/2, f'${v:,.0f}', va='center', fontsize=10, color=MUTED)

ax.set_xlabel("Per-Capita Allocation ($)")
ax.set_title("Top 10 Regions — Utilitarian Model", pad=15)
ax.grid(axis='x', linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "11_top10_utilitarian")


# CHART 12: Top 10 Regions — Maximin
top_maxi = m[m["maxi_y"]==1].nlargest(10, "maxi_pc")[["county","state","maxi_pc","n_i"]].reset_index(drop=True)
top_maxi["label"] = top_maxi["county"] + ", " + top_maxi["state"]
top_maxi = top_maxi.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [STATE_C.get(s, MUTED) for s in top_maxi["state"]]
bars = ax.barh(top_maxi["label"], top_maxi["maxi_pc"], color=bar_colors, height=0.6, zorder=3, edgecolor='white', linewidth=0.3)
for bar, v in zip(bars, top_maxi["maxi_pc"]):
    ax.text(v + 30, bar.get_y() + bar.get_height()/2, f'${v:,.0f}', va='center', fontsize=10, color=MUTED)

ax.set_xlabel("Per-Capita Allocation ($)")
ax.set_title("Top 10 Regions — Maximin Model", pad=15)
ax.grid(axis='x', linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "12_top10_maximin")


# CHART 13: PR Per-Capita Deep Dive
fig, ax = plt.subplots(figsize=(9, 4))
pr_vals = [pr_fema, pr_util, pr_maxi]
y_pos = np.arange(3)
bars = ax.barh(y_pos, pr_vals, height=0.5, color=colors, zorder=3, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, pr_vals):
    ax.text(v + 10, bar.get_y() + bar.get_height()/2, f'${v:,.0f}', va='center', fontsize=13, fontweight='bold', color='#1f2937')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=13, fontweight='bold')
ax.set_xlabel("Dollars per Person")
ax.set_title("Puerto Rico — Per-Capita Allocation Comparison", pad=15)
ax.grid(axis='x', linestyle='--')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "13_pr_deep_dive")


# CHART 14: PR vs TX Side by Side
fig, ax = plt.subplots(figsize=(9, 5))
x14 = np.arange(2)
w14 = 0.22

pr_v = [pr_fema, pr_util, pr_maxi]
tx_v = [tx_fema, tx_util, tx_maxi]

for j, (lbl, col) in enumerate(zip(labels, colors)):
    vals = [pr_v[j], tx_v[j]]
    b = ax.bar(x14 + (j-1)*w14, vals, w14, label=lbl, color=col, zorder=3, edgecolor='white', linewidth=0.5)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 15, f'${v:,.0f}', ha='center', fontsize=9, color=MUTED)

ax.set_xticks(x14)
ax.set_xticklabels(["Puerto Rico", "Texas"], fontsize=14, fontweight='bold')
ax.set_ylabel("Dollars per Person")
ax.set_title("PR vs TX — Per-Capita Allocation Side by Side", pad=15)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

save(fig, "14_pr_vs_tx_side_by_side")


# CHART 15: Summary Metrics Table
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

table_data = [
    ["Metric", "FEMA Actual", "Utilitarian", "Maximin"],
    ["Total Budget", "$4.12B", "$4.12B", "$4.12B"],
    ["Regions Funded", "179 / 183", "86 / 183", "84 / 183"],
    ["Gini Coefficient", f"{g_fema:.4f}", f"{g_util:.4f}", f"{g_maxi:.4f}"],
    ["PR Per Capita", f"${pr_fema:,.0f}", f"${pr_util:,.0f}", f"${pr_maxi:,.0f}"],
    ["TX Per Capita", f"${tx_fema:,.0f}", f"${tx_util:,.0f}", f"${tx_maxi:,.0f}"],
    ["PR/TX Ratio", f"{ratios[0]:.2f}x", f"{ratios[1]:.2f}x", f"{ratios[2]:.2f}x"],
    ["PR % Need Met", f"{st[st.state=='PR']['fema_pct'].values[0]:.0f}%", f"{st[st.state=='PR']['util_pct'].values[0]:.0f}%", f"{st[st.state=='PR']['maxi_pct'].values[0]:.0f}%"],
    ["Min Per-Capita (z*)", "N/A", "N/A", "$465.68"],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)

# Header row
for j in range(4):
    cell = table[0, j]
    if j == 0:
        cell.set_facecolor('#374151')
    elif j == 1:
        cell.set_facecolor(FEMA_C)
    elif j == 2:
        cell.set_facecolor(UTIL_C)
    else:
        cell.set_facecolor(MAXI_C)
    cell.set_text_props(fontweight='bold', color='white')

# Data rows
for i in range(1, len(table_data)):
    for j in range(4):
        cell = table[i, j]
        cell.set_facecolor('#ffffff' if i % 2 == 1 else '#f3f4f6')
        cell.set_edgecolor('#e5e7eb')
        cell.set_text_props(color='#1f2937')
        if j == 0:
            cell.set_text_props(fontweight='bold', color='#374151')

ax.set_title("Model Comparison Summary", fontsize=18, fontweight='bold', pad=20, color='#1f2937')

save(fig, "15_summary_table")