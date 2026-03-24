"""
Sensitivity & Specificity Analysis of the Reward-Model Predictor
on the LMArena Clean Summary Dataset.

Definitions (per LLM pair / group):
  - Positive class: winner == 1  (LLM_a wins)
  - Negative class: winner == 0  (LLM_b wins)
  - Sensitivity (TPR) = TP / (TP + FN)   — how often the predictor correctly predicts LLM_a wins
  - Specificity (TNR) = TN / (TN + FP)   — how often the predictor correctly predicts LLM_b wins
"""

import pathlib
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "clean_data" / "clean_summary.csv"
OUT_DIR = ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
})

PALETTE = sns.color_palette("mako", as_cmap=False, n_colors=5)
ACCENT  = "#E07A5F"   # warm accent for highlights
PRIMARY = "#3D405B"    # dark slate for primary marks
CMAP    = "mako_r"

# ── data loading & per-group metrics ─────────────────────────────────────────
df = pd.read_csv(DATA_PATH)


def compute_group_metrics(g):
    """Compute sensitivity, specificity, accuracy, and metadata for one group."""
    tp = ((g.winner == 1) & (g.prediction == 1)).sum()
    fn = ((g.winner == 1) & (g.prediction == 0)).sum()
    tn = ((g.winner == 0) & (g.prediction == 0)).sum()
    fp = ((g.winner == 0) & (g.prediction == 1)).sum()
    n = len(g)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    return pd.Series({
        "model_a": g.model_a.iloc[0],
        "model_b": g.model_b.iloc[0],
        "n": n,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "win_rate": g.winner.mean(),
        "pred_rate": g.prediction.mean(),
    })


metrics = df.groupby("group_id").apply(compute_group_metrics).reset_index()
metrics["pair_label"] = metrics.model_a.str[:12] + " v " + metrics.model_b.str[:12]

# sort by descending win rate for ranked plots
metrics_ranked = metrics.sort_values("win_rate", ascending=False).reset_index(drop=True)
metrics_ranked["rank"] = np.arange(1, len(metrics_ranked) + 1)

# ── overall (macro & micro) statistics ───────────────────────────────────────
macro_sens = metrics.sensitivity.mean()
macro_spec = metrics.specificity.mean()
macro_acc  = metrics.accuracy.mean()

total_tp = metrics.tp.sum()
total_fn = metrics.fn.sum()
total_tn = metrics.tn.sum()
total_fp = metrics.fp.sum()
micro_sens = total_tp / (total_tp + total_fn)
micro_spec = total_tn / (total_tn + total_fp)
micro_acc  = (total_tp + total_tn) / (total_tp + total_fn + total_tn + total_fp)

summary_text = textwrap.dedent(f"""\
    ====================================================================
    Predictor Sensitivity & Specificity Analysis — LMArena Clean Summary
    ====================================================================
    Dataset : {DATA_PATH.name}
    Rows    : {len(df):,}
    Groups  : {metrics.group_id.nunique()}
    Group sizes : min={int(metrics.n.min())}, median={int(metrics.n.median())}, max={int(metrics.n.max())}

    ── Micro-averaged (pooled across all rows) ────────────────────────
      Sensitivity : {micro_sens:.4f}
      Specificity : {micro_spec:.4f}
      Accuracy    : {micro_acc:.4f}

    ── Macro-averaged (mean across groups) ────────────────────────────
      Sensitivity : {macro_sens:.4f}  (std={metrics.sensitivity.std():.4f})
      Specificity : {macro_spec:.4f}  (std={metrics.specificity.std():.4f})
      Accuracy    : {macro_acc:.4f}  (std={metrics.accuracy.std():.4f})

    ── Confusion matrix (pooled) ──────────────────────────────────────
      TP = {int(total_tp):>6,}    FN = {int(total_fn):>6,}
      FP = {int(total_fp):>6,}    TN = {int(total_tn):>6,}

    ── Per-group distribution quantiles ───────────────────────────────
    Sensitivity: {metrics.sensitivity.describe().to_string()}
    Specificity: {metrics.specificity.describe().to_string()}
""")

print(summary_text)
(OUT_DIR / "summary_stats.txt").write_text(summary_text)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Scatter: Sensitivity vs Specificity per pair
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sc = ax.scatter(
    metrics.specificity, metrics.sensitivity,
    c=metrics.win_rate, cmap=CMAP, s=50, alpha=0.8,
    edgecolors="white", linewidth=0.4,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.82)
cbar.set_label("Ground-Truth Win Rate (LLM A)", fontsize=10)

# reference lines
ax.axhline(macro_sens, color=ACCENT, ls="--", lw=1, label=f"Macro Sens = {macro_sens:.2f}")
ax.axvline(macro_spec, color=PRIMARY, ls="--", lw=1, label=f"Macro Spec = {macro_spec:.2f}")
ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8, alpha=0.5)

ax.set_xlabel("Specificity (TNR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.set_title("Sensitivity vs Specificity per LLM Pair")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "plot1_sensitivity_vs_specificity.png")
plt.close(fig)
print("✓ Saved plot1_sensitivity_vs_specificity.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2a — Ranked: Specificity vs descending GT win-rate
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors_spec = plt.colormaps["mako_r"](
    (metrics_ranked.win_rate - metrics_ranked.win_rate.min()) /
    (metrics_ranked.win_rate.max() - metrics_ranked.win_rate.min())
)
ax.bar(metrics_ranked["rank"], metrics_ranked.specificity,
       color=colors_spec, width=1.0, edgecolor="none", alpha=0.85)
ax.axhline(macro_spec, color=ACCENT, ls="--", lw=1.2, label=f"Macro Spec = {macro_spec:.2f}")
ax.set_xlabel("LLM Pair Rank (by descending GT win rate of LLM A)")
ax.set_ylabel("Specificity")
ax.set_title("Specificity Ranked by Descending Ground-Truth Win Rate")
ax.legend(fontsize=9)
ax.set_xlim(0.5, len(metrics_ranked) + 0.5)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot2a_specificity_ranked.png")
plt.close(fig)
print("✓ Saved plot2a_specificity_ranked.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2b — Ranked: Sensitivity vs descending GT win-rate
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors_sens = plt.colormaps["mako_r"](
    (metrics_ranked.win_rate - metrics_ranked.win_rate.min()) /
    (metrics_ranked.win_rate.max() - metrics_ranked.win_rate.min())
)
ax.bar(metrics_ranked["rank"], metrics_ranked.sensitivity,
       color=colors_sens, width=1.0, edgecolor="none", alpha=0.85)
ax.axhline(macro_sens, color=ACCENT, ls="--", lw=1.2, label=f"Macro Sens = {macro_sens:.2f}")
ax.set_xlabel("LLM Pair Rank (by descending GT win rate of LLM A)")
ax.set_ylabel("Sensitivity")
ax.set_title("Sensitivity Ranked by Descending Ground-Truth Win Rate")
ax.legend(fontsize=9)
ax.set_xlim(0.5, len(metrics_ranked) + 0.5)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot2b_sensitivity_ranked.png")
plt.close(fig)
print("✓ Saved plot2b_sensitivity_ranked.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Joint distribution: 2D KDE + marginals (sensitivity & specificity)
# ══════════════════════════════════════════════════════════════════════════════
g = sns.JointGrid(data=metrics, x="specificity", y="sensitivity", height=6.5)
g.plot_joint(
    sns.kdeplot, fill=True, cmap="mako", levels=8, thresh=0.05, alpha=0.7,
)
g.plot_joint(
    sns.scatterplot, color=PRIMARY, s=20, alpha=0.5, edgecolor="white", linewidth=0.3,
)
g.plot_marginals(sns.histplot, kde=True, color=PRIMARY, alpha=0.4, bins=25)
g.ax_joint.set_xlabel("Specificity (TNR)")
g.ax_joint.set_ylabel("Sensitivity (TPR)")
g.ax_joint.set_xlim(-0.02, 1.02)
g.ax_joint.set_ylim(-0.02, 1.02)
g.figure.suptitle("Joint Distribution of Sensitivity & Specificity", y=1.01, fontweight="bold")
g.figure.tight_layout()
g.figure.savefig(OUT_DIR / "plot3_joint_distribution.png", bbox_inches="tight")
plt.close(g.figure)
print("✓ Saved plot3_joint_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Sensitivity & Specificity vs GT Win Rate (dual-axis smooth trends)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(metrics.win_rate, metrics.sensitivity, s=28, alpha=0.45,
           color="#457B9D", label="Sensitivity", edgecolors="white", linewidth=0.3)
ax.scatter(metrics.win_rate, metrics.specificity, s=28, alpha=0.45,
           color="#E76F51", label="Specificity", edgecolors="white", linewidth=0.3)

# LOWESS smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

for col, color, label in [
    ("sensitivity", "#457B9D", "Sensitivity (LOWESS)"),
    ("specificity", "#E76F51", "Specificity (LOWESS)"),
]:
    sm = sm_lowess(metrics[col].values, metrics.win_rate.values, frac=0.35)
    ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.5, label=label)

ax.set_xlabel("Ground-Truth Win Rate (LLM A)")
ax.set_ylabel("Rate")
ax.set_title("Sensitivity & Specificity vs Ground-Truth Win Rate")
ax.legend(fontsize=9, ncol=2, loc="upper center", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot4_sens_spec_vs_winrate.png")
plt.close(fig)
print("✓ Saved plot4_sens_spec_vs_winrate.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Heatmap: Accuracy, Sensitivity, Specificity binned by win-rate
# ══════════════════════════════════════════════════════════════════════════════
metrics["wr_bin"] = pd.cut(metrics.win_rate, bins=np.linspace(0, 1, 11), include_lowest=True)
bin_agg = metrics.groupby("wr_bin", observed=False)[["sensitivity", "specificity", "accuracy"]].mean()

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    bin_agg.T, annot=True, fmt=".2f", cmap="mako_r", linewidths=0.5,
    ax=ax, vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": "Rate"},
)
ax.set_xlabel("Ground-Truth Win Rate Bin")
ax.set_ylabel("")
ax.set_title("Mean Predictor Metrics by Win-Rate Bin")
# rotate x labels
ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=35, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot5_heatmap_by_winrate.png")
plt.close(fig)
print("✓ Saved plot5_heatmap_by_winrate.png")

print("\n✓ All plots saved to:", OUT_DIR)
