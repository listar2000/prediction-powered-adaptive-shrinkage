"""Build the LMArena CI two-panel benchmark plot from saved raw results.

Loads `raw.csv` produced by `run_lmarena_ci.py`, computes per-(alpha, method)
mean and SE, and renders a two-panel figure (CI Width / Miscoverage rate)
styled to match `figures/lmarena_ci_benchmarks.pdf` in the paper repo.

Iterating styling is decoupled from running the experiment; just rerun this
script after edits.

Usage:

    uv run python src/scripts/make_lmarena_plot.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Methods to plot, in legend order. Mirrors the paper figure (Classical,
# Prediction Mean, PPI, Power-Tuned PPI) plus the two PT-based rebiased
# estimators contributed in this paper.
PLOT_METHODS = [
    "mle_ci",
    "pred_mean_ci",
    # "ppi_ci",
    "pt_ci",
    "eb_pt_ci",
    "eb_npmle_pt_ci",
]

METHOD_LABELS = {
    "mle_ci":         "Classical",
    "pred_mean_ci":   "Prediction Mean",
    # "ppi_ci":         "PPI",
    "pt_ci":          "PT",
    "eb_pt_ci":       "PT Param (ours)",
    "eb_npmle_pt_ci": "PT NPMLE (ours)",
}

METHOD_COLORS = {
    "mle_ci":         "#1f77b4",  # blue
    "pred_mean_ci":   "#bcbd22",  # olive
    # "ppi_ci":         "#2ca02c",  # green
    "pt_ci":          "#d62728",  # red
    "eb_pt_ci":       "#9467bd",  # purple (PT-Param)
    "eb_npmle_pt_ci": "#e377c2",  # pink   (PT-NPMLE)
}

METHOD_MARKERS = {
    "mle_ci":         "o",
    "pred_mean_ci":   "v",
    # "ppi_ci":         "s",
    "pt_ci":          "D",
    "eb_pt_ci":       "P",
    "eb_npmle_pt_ci": "*",
}


XLIM = (0.007, 0.5)
YLIM_MISCOV = (0.005, 0.99)
YTICKS_MISCOV = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for alpha, group in raw.groupby("alpha"):
        for method in PLOT_METHODS:
            cov = group[f"{method}_coverage"].astype(float)
            wid = group[f"{method}_width"].astype(float)
            records.append({
                "alpha": float(alpha),
                "method": method,
                "mean_coverage": cov.mean(),
                "se_coverage": cov.sem(),
                "mean_width": wid.mean(),
                "se_width": wid.sem(),
            })
    return pd.DataFrame(records)


def make_plot(summary_df: pd.DataFrame, alphas, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))

    # --- Left panel: CI Width vs Alpha ---
    ax = axes[0]
    for method in PLOT_METHODS:
        df_m = summary_df[summary_df["method"] == method].sort_values("alpha")
        ax.errorbar(
            df_m["alpha"], df_m["mean_width"], yerr=1.96 * df_m["se_width"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], capsize=2, linewidth=1.2, markersize=5,
        )
    ax.set_xlabel(r"$\alpha$", fontsize=11)
    ax.set_ylabel("Average CI Width", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(XLIM)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")

    # --- Right panel: Miscoverage vs Alpha ---
    ax = axes[1]
    for method in PLOT_METHODS:
        df_m = summary_df[summary_df["method"] == method].sort_values("alpha")
        miscov = 1 - df_m["mean_coverage"].astype(float)
        ax.errorbar(
            df_m["alpha"], miscov,
            yerr=1.96 * df_m["se_coverage"].astype(float),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], capsize=2, linewidth=1.2, markersize=5,
        )
    diag = np.linspace(XLIM[0], XLIM[1], 100)
    ax.plot(diag, diag, "k--", linewidth=1)
    ax.set_xlabel(r"$\alpha$", fontsize=11)
    ax.set_ylabel("Miscoverage Rate", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM_MISCOV)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(YTICKS_MISCOV)
    ax.set_yticklabels([str(y) for y in YTICKS_MISCOV], fontsize=9)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")

    # Shared legend at the top, all in one row; nominal handle appended.
    handles, labels = axes[0].get_legend_handles_labels()
    nominal_handle = plt.Line2D([], [], color="k", linestyle="--", linewidth=1)
    handles.append(nominal_handle)
    labels.append(r"Nominal $\alpha$")
    fig.legend(
        handles, labels, loc="upper center", ncol=len(handles),
        fontsize=8, bbox_to_anchor=(0.5, 1.04), frameon=False,
        handletextpad=0.4, columnspacing=1.0,
    )

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    default = Path("/tmp/lmarena_ci_results")
    parser.add_argument("--in-dir", type=Path,
                        default=default)
    parser.add_argument("--out", type=Path,
                        default=default / "lmarena_ci_plot.pdf")
    args = parser.parse_args()

    raw = pd.read_csv(args.in_dir / "raw.csv")
    alphas = sorted(raw["alpha"].unique())
    summary_df = summarize(raw)
    make_plot(summary_df, alphas, args.out)


if __name__ == "__main__":
    main()
