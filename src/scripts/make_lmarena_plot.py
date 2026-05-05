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
# Prediction Mean, Power-Tuned PPI, Rebiased PPI) plus the additional
# Param/NPMLE x PPI/UniPT variants.
PLOT_METHODS = [
    "mle_ci",
    "pred_mean_ci",
    "pt_ci",
    "eb_ppi_ci",
    "eb_npmle_ppi_ci",
    "eb_unipt_ppi_ci",
    "eb_npmle_unipt_ppi_ci",
]

METHOD_LABELS = {
    "mle_ci":                "Classical (CLT)",
    "pred_mean_ci":          "Prediction Mean",
    "pt_ci":                 "Power-Tuned PPI",
    "eb_ppi_ci":             "Rebiased PPI Param (ours)",
    "eb_npmle_ppi_ci":       "Rebiased PPI NPMLE (ours)",
    "eb_unipt_ppi_ci":       "Rebiased UniPT Param (ours)",
    "eb_npmle_unipt_ppi_ci": "Rebiased UniPT NPMLE (ours)",
}

METHOD_COLORS = {
    "mle_ci":                "#1f77b4",  # blue
    "pred_mean_ci":          "#bcbd22",  # olive
    "pt_ci":                 "#d62728",  # red
    "eb_ppi_ci":             "#9467bd",  # purple (PPI-Param)
    "eb_npmle_ppi_ci":       "#e377c2",  # pink   (PPI-NPMLE)
    "eb_unipt_ppi_ci":       "#8c564b",  # brown  (UniPT-Param)
    "eb_npmle_unipt_ppi_ci": "#17becf",  # cyan   (UniPT-NPMLE)
}

METHOD_MARKERS = {
    "mle_ci":                "o",
    "pred_mean_ci":          "v",
    "pt_ci":                 "D",
    "eb_ppi_ci":             "P",
    "eb_npmle_ppi_ci":       "*",
    "eb_unipt_ppi_ci":       "X",
    "eb_npmle_unipt_ppi_ci": "h",
}


def _no_trailing_zero(x, _pos=None):
    """Tick formatter: 0.10 -> 0.1, 0.50 -> 0.5, etc."""
    s = ("%g" % x)
    return s


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
    # Paper figure has near-square panels with relatively bold markers.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # ---------------- Left: CI Width vs alpha (log x) ----------------
    ax = axes[0]
    for method in PLOT_METHODS:
        df_m = summary_df[summary_df["method"] == method].sort_values("alpha")
        ax.errorbar(
            df_m["alpha"], df_m["mean_width"],
            yerr=1.96 * df_m["se_width"],
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], capsize=3, linewidth=1.5,
            markersize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=13)
    ax.set_ylabel("Average CI Width", fontsize=13)
    ax.set_xticks(alphas)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(_no_trailing_zero))
    ax.minorticks_off()
    ax.grid(True, which="both", alpha=0.3)

    # ---------------- Right: Miscoverage vs alpha (log-log) ----------------
    ax = axes[1]
    for method in PLOT_METHODS:
        df_m = summary_df[summary_df["method"] == method].sort_values("alpha")
        miscov = 1 - df_m["mean_coverage"].astype(float)
        ax.errorbar(
            df_m["alpha"], miscov,
            yerr=1.96 * df_m["se_coverage"].astype(float),
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], capsize=3, linewidth=1.5,
            markersize=8,
        )

    # Nominal y=x reference (linear in log-log).
    diag = np.linspace(min(alphas) * 0.7, max(alphas) * 1.3, 50)
    ax.plot(diag, diag, "k--", linewidth=1.0, label=r"Nominal $\alpha$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=13)
    ax.set_ylabel("Miscoverage Rate", fontsize=13)
    ax.set_xticks(alphas)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(_no_trailing_zero))
    ax.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(_no_trailing_zero))
    ax.minorticks_off()
    ax.grid(True, which="both", alpha=0.3)

    # Single shared legend at the top, wrapped over 2 rows (paper figure
    # has 5 entries in 1 row; we have 8 -> use 4 columns x 2 rows).
    handles, labels = axes[1].get_legend_handles_labels()
    n_legend = len(handles)
    ncol = 4 if n_legend > 5 else n_legend
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=ncol, fontsize=9.5, frameon=False,
        columnspacing=1.4, handletextpad=0.5,
    )
    # Reserve more vertical space for 2-row legend.
    legend_pad = 0.84 if n_legend > 5 else 0.92
    fig.tight_layout(rect=(0, 0, 1, legend_pad))
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path,
                        default=Path(__file__).parent / "lmarena_ci_results")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "lmarena_ci_npmle.pdf")
    args = parser.parse_args()

    raw = pd.read_csv(args.in_dir / "raw.csv")
    alphas = sorted(raw["alpha"].unique())
    summary_df = summarize(raw)
    make_plot(summary_df, alphas, args.out)


if __name__ == "__main__":
    main()
