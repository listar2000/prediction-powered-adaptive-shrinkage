"""Combined LMArena + Amazon-Reviews CI benchmark plot.

Renders a 1x4 figure: ``[LMArena width][LMArena miscoverage][Amazon width]
[Amazon miscoverage]``. Both datasets share a single legend at the top.

Inputs
------
* Amazon: ``tmp/lmarena_ci_results/external/amazon_real.csv`` (colleague-supplied
  per-(alpha, method) summary with columns Setting, Method, SE_Coverage,
  Avg_Coverage, SE_Length, Avg_Length, SE_Length_Ratio, Avg_Length_Ratio, alpha).
  Method names follow ``_style.CSV_TO_INTERNAL``.
* LMArena: ``tmp/lmarena_ci_results/summary.csv`` (per-(alpha, method) summary
  produced by ``run_lmarena_ci.py`` with the internal method keys directly).

Output
------
* ``tmp/plots/amazon_lmarena_ci.pdf``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from _style import (
    CSV_TO_INTERNAL,
    METHOD_LABELS,
    METHOD_COLORS,
    METHOD_MARKERS,
    ERR_KW,
    LABEL_FS,
    TICK_FS,
    LEGEND_FS,
    COLTITLE_FS,
    SAVE_DPI,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP = PROJECT_ROOT / "tmp"

XLIM = (0.007, 0.5)
YLIM_MISCOV = (0.005, 0.99)
YTICKS_MISCOV = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# Method draw order for both panels of both datasets. PPI is dropped — the
# Power-Tuned PPI baseline (``pt_ci``) carries the same role at smaller width.
PLOT_METHODS = [
    "mle_ci",
    "pred_mean_ci",
    "pt_ci",
    "eb_pt_ci",
    "eb_npmle_pt_ci",
]


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------

def load_amazon_summary(path: Path) -> pd.DataFrame:
    """Translate colleague's amazon_real.csv to the columns we plot.

    Returns columns ``alpha, method, mean_coverage, se_coverage,
    mean_width, se_width``.
    """
    raw = pd.read_csv(path)
    raw["method"] = raw["Method"].map(CSV_TO_INTERNAL)
    if raw["method"].isna().any():
        unknown = raw.loc[raw["method"].isna(), "Method"].unique().tolist()
        raise ValueError(f"Unknown method names in {path}: {unknown}")
    return pd.DataFrame({
        "alpha": raw["alpha"].astype(float),
        "method": raw["method"],
        "mean_coverage": raw["Avg_Coverage"].astype(float),
        "se_coverage":   raw["SE_Coverage"].astype(float),
        "mean_width":    raw["Avg_Length"].astype(float),
        "se_width":      raw["SE_Length"].astype(float),
    })


def load_lmarena_summary(path: Path) -> pd.DataFrame:
    """LMArena summary.csv already uses internal method keys."""
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# Panel renderers
# ----------------------------------------------------------------------------

def _draw_width_panel(ax, df, alphas, *, label_for_legend=False, errorbars=False):
    for method in PLOT_METHODS:
        df_m = df[df["method"] == method].sort_values("alpha")
        if df_m.empty:
            continue
        kw = dict(
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            **ERR_KW,
        )
        if label_for_legend:
            kw["label"] = METHOD_LABELS[method]
        yerr = 1.96 * df_m["se_width"] if errorbars else None
        ax.errorbar(df_m["alpha"], df_m["mean_width"], yerr=yerr, **kw)
    ax.set_xlabel(r"$\alpha$", fontsize=LABEL_FS)
    ax.set_ylabel("Average Width", fontsize=LABEL_FS)
    ax.set_xscale("log")
    ax.set_xlim(XLIM)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")


def _draw_miscov_panel(ax, df, alphas, *, errorbars=False):
    for method in PLOT_METHODS:
        df_m = df[df["method"] == method].sort_values("alpha")
        if df_m.empty:
            continue
        miscov = 1 - df_m["mean_coverage"].astype(float)
        yerr = 1.96 * df_m["se_coverage"].astype(float) if errorbars else None
        ax.errorbar(
            df_m["alpha"], miscov, yerr=yerr,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], **ERR_KW,
        )
    diag = np.linspace(XLIM[0], XLIM[1], 100)
    ax.plot(diag, diag, "k--", linewidth=1)
    ax.set_xlabel(r"$\alpha$", fontsize=LABEL_FS)
    ax.set_ylabel("Miscoverage Rate", fontsize=LABEL_FS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM_MISCOV)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=TICK_FS)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(YTICKS_MISCOV)
    ax.set_yticklabels([str(y) for y in YTICKS_MISCOV], fontsize=TICK_FS)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")


# ----------------------------------------------------------------------------
# Figure assembly
# ----------------------------------------------------------------------------

def make_plot(amazon_df, lmarena_df, out_path: Path, *, errorbars: bool = False):
    alphas_amazon = sorted(amazon_df["alpha"].unique())
    alphas_lmarena = sorted(lmarena_df["alpha"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.2))

    # LMArena is shown first (matching the order in the main text);
    # legend handles are registered only on the first (LMArena-width) axis.
    _draw_width_panel(axes[0], lmarena_df, alphas_lmarena,
                      label_for_legend=True, errorbars=errorbars)
    _draw_miscov_panel(axes[1], lmarena_df, alphas_lmarena, errorbars=errorbars)
    _draw_width_panel(axes[2], amazon_df, alphas_amazon, errorbars=errorbars)
    _draw_miscov_panel(axes[3], amazon_df, alphas_amazon, errorbars=errorbars)

    # Shared top legend with the diagonal nominal-alpha entry appended.
    handles, labels = axes[0].get_legend_handles_labels()
    nominal_handle = plt.Line2D([], [], color="k", linestyle="--", linewidth=1)
    handles.append(nominal_handle)
    labels.append(r"Nominal $\alpha$")
    fig.legend(
        handles, labels, loc="upper center", ncol=len(handles),
        fontsize=LEGEND_FS, bbox_to_anchor=(0.5, 1.06), frameon=False,
        handletextpad=0.4, columnspacing=1.0,
    )

    # Reserve ~10% at the bottom of the figure for the column-pair titles.
    # Titles sit just below the per-panel x-axis labels and intentionally
    # reach a bit into the slack space matplotlib leaves there.
    fig.tight_layout(pad=1.0, rect=(0, 0.10, 1, 0.94))

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    pos3 = axes[3].get_position()
    x_lmarena = (pos0.x0 + pos1.x1) / 2
    x_amazon = (pos2.x0 + pos3.x1) / 2
    y_title = 0.1
    fig.text(x_lmarena, y_title, "(i) LMArena",
             ha="center", va="bottom",
             fontsize=COLTITLE_FS, fontweight="bold")
    fig.text(x_amazon, y_title, "(ii) Amazon",
             ha="center", va="bottom",
             fontsize=COLTITLE_FS, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight", dpi=SAVE_DPI)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amazon-csv", type=Path,
        default=TMP / "lmarena_ci_results" / "external" / "amazon_real.csv",
    )
    parser.add_argument(
        "--lmarena-csv", type=Path,
        default=TMP / "lmarena_ci_results" / "summary.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=TMP / "plots" / "amazon_lmarena_ci.pdf",
    )
    parser.add_argument(
        "--errorbars", action="store_true",
        help="Draw 1.96*SE error bars on the Average-Width and "
             "Miscoverage panels (default: off, since the SE values "
             "are typically <1% of the means).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    amazon_df = load_amazon_summary(args.amazon_csv)
    lmarena_df = load_lmarena_summary(args.lmarena_csv)
    make_plot(amazon_df, lmarena_df, args.out, errorbars=args.errorbars)


if __name__ == "__main__":
    main()
