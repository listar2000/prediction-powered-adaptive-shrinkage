"""Combined synthetic CI benchmark plot (normal prior + two-point prior).

Renders a 1x4 figure: ``[Normal width][Normal miscoverage][TwoPoint width]
[TwoPoint miscoverage]``. Both halves share a single legend at the top.

Inputs
------
* ``tmp/lmarena_ci_results/external/synthetic_normal.csv``  (x = sigma)
* ``tmp/lmarena_ci_results/external/synthetic_twopoint.csv`` (x = b0)

Both CSVs have columns: ``<x>, Method, SE_Coverage, Avg_Coverage, SE_Length,
Avg_Length, SE_Length_Ratio, Avg_Length_Ratio``. Method names follow the
colleague's convention (``classic``, ``classic_bias``, ``ppi``, ``pt``,
``pt_oracle``, ``pt_param``, ``pt_npmle``); we translate via
``_style.CSV_TO_INTERNAL``.

The width panel reports the *length-ratio* (relative to Classical, which is
constant across x). We use Avg_Length_Ratio / SE_Length_Ratio columns so the
values are dimensionless and comparable across panels. Miscoverage = 1 - cov.

Output
------
* ``tmp/plots/synthetic_ci.pdf``
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

# Method order. Includes Oracle (only present in the synthetic CSVs). PPI is
# dropped from this combined plot — Power-Tuned PPI (``pt_ci``) is the
# baseline of interest.
PLOT_METHODS = [
    "mle_ci",
    "pred_mean_ci",
    "pt_ci",
    "pt_oracle",
    "eb_pt_ci",
    "eb_npmle_pt_ci",
]

# All four CSV runs use a fixed nominal level alpha = 0.05 (Classical coverage
# in the supplied summaries is ~0.95). The synthetic plots therefore draw a
# horizontal nominal-line at this level instead of the y=x diagonal.
NOMINAL_ALPHA = 0.05

# Miscoverage range to display. Set wide enough so Classic-Bias (which can
# undercover heavily, e.g. miscov ~ 0.5 for two-point prior at b0=0.5) stays
# visible while still showing the cluster around alpha.
YLIM_MISCOV = (0.005, 0.99)
YTICKS_MISCOV = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------

def load_synthetic_summary(path: Path, x_col: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if x_col not in raw.columns:
        raise ValueError(f"{path}: expected column {x_col!r}, got {list(raw.columns)}")
    raw["method"] = raw["Method"].map(CSV_TO_INTERNAL)
    if raw["method"].isna().any():
        unknown = raw.loc[raw["method"].isna(), "Method"].unique().tolist()
        raise ValueError(f"Unknown method names in {path}: {unknown}")
    return pd.DataFrame({
        "x":              raw[x_col].astype(float),
        "method":         raw["method"],
        "mean_coverage":  raw["Avg_Coverage"].astype(float),
        "se_coverage":    raw["SE_Coverage"].astype(float),
        "mean_ratio":     raw["Avg_Length_Ratio"].astype(float),
        "se_ratio":       raw["SE_Length_Ratio"].astype(float),
    })


# ----------------------------------------------------------------------------
# Panel renderers
# ----------------------------------------------------------------------------

def _draw_width_panel(ax, df, xs, x_label, *,
                      label_for_legend=False, errorbars=False):
    for method in PLOT_METHODS:
        df_m = df[df["method"] == method].sort_values("x")
        if df_m.empty:
            continue
        kw = dict(
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            **ERR_KW,
        )
        if label_for_legend:
            kw["label"] = METHOD_LABELS[method]
        yerr = 1.96 * df_m["se_ratio"] if errorbars else None
        ax.errorbar(df_m["x"], df_m["mean_ratio"], yerr=yerr, **kw)
    ax.set_xlabel(x_label, fontsize=LABEL_FS)
    ax.set_ylabel("Average Width Ratio", fontsize=LABEL_FS)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")


def _draw_miscov_panel(ax, df, xs, x_label, *, errorbars=False):
    for method in PLOT_METHODS:
        df_m = df[df["method"] == method].sort_values("x")
        if df_m.empty:
            continue
        miscov = 1 - df_m["mean_coverage"].astype(float)
        yerr = 1.96 * df_m["se_coverage"].astype(float) if errorbars else None
        ax.errorbar(
            df_m["x"], miscov, yerr=yerr,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], **ERR_KW,
        )
    # Nominal level: horizontal line at alpha.
    ax.axhline(NOMINAL_ALPHA, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label, fontsize=LABEL_FS)
    ax.set_ylabel("Miscoverage Rate", fontsize=LABEL_FS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(YLIM_MISCOV)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=TICK_FS)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(YTICKS_MISCOV)
    ax.set_yticklabels([str(y) for y in YTICKS_MISCOV], fontsize=TICK_FS)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="both")


# ----------------------------------------------------------------------------
# Figure assembly
# ----------------------------------------------------------------------------

def make_plot(normal_df, twopoint_df, out_path: Path, *, errorbars: bool = False):
    xs_normal = sorted(normal_df["x"].unique())
    xs_twopoint = sorted(twopoint_df["x"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))

    _draw_width_panel(axes[0], normal_df, xs_normal, r"$A^{1/2}$",
                      label_for_legend=True, errorbars=errorbars)
    _draw_miscov_panel(axes[1], normal_df, xs_normal, r"$A^{1/2}$",
                       errorbars=errorbars)
    _draw_width_panel(axes[2], twopoint_df, xs_twopoint, r"$b_0$",
                      errorbars=errorbars)
    _draw_miscov_panel(axes[3], twopoint_df, xs_twopoint, r"$b_0$",
                       errorbars=errorbars)

    handles, labels = axes[0].get_legend_handles_labels()
    nominal_handle = plt.Line2D([], [], color="k", linestyle="--", linewidth=1)
    handles.append(nominal_handle)
    # labels.append(rf"Nominal $\alpha = {NOMINAL_ALPHA}$")
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
    x_normal = (pos0.x0 + pos1.x1) / 2
    x_twopt = (pos2.x0 + pos3.x1) / 2
    y_title = 0.1
    fig.text(x_normal, y_title, "(i) Normal prior",
             ha="center", va="bottom",
             fontsize=COLTITLE_FS, fontweight="bold")
    fig.text(x_twopt, y_title, "(ii) Two-point prior",
             ha="center", va="bottom",
             fontsize=COLTITLE_FS, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight", dpi=SAVE_DPI)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normal-csv", type=Path,
        default=TMP / "lmarena_ci_results" / "external" / "synthetic_normal.csv",
    )
    parser.add_argument(
        "--twopoint-csv", type=Path,
        default=TMP / "lmarena_ci_results" / "external" / "synthetic_twopoint.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=TMP / "plots" / "synthetic_ci.pdf",
    )
    parser.add_argument(
        "--errorbars", action="store_true",
        help="Draw 1.96*SE error bars on the Average-Width-Ratio and "
             "Miscoverage panels (default: off, since the SE values "
             "are typically small relative to the means).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    normal_df = load_synthetic_summary(args.normal_csv, x_col="sigma")
    twopoint_df = load_synthetic_summary(args.twopoint_csv, x_col="b0")
    make_plot(normal_df, twopoint_df, args.out, errorbars=args.errorbars)


if __name__ == "__main__":
    main()
