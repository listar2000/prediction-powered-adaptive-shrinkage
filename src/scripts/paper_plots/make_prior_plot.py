"""Bias-prior diagnostic plot (LMArena + Amazon + GWAS).

Renders a 1x3 figure showing, for each real-data application in the
paper, the empirical histogram of observed bias estimates alongside the
fitted parametric (Gaussian) prior and the NPMLE prior, plus the two
implied marginal densities of $\\hat b_i$. The Parametric/NPMLE legend
is shown only in panel (a); per-panel subtitles ``(a) LMArena`` /
``(b) Amazon`` / ``(c) GWAS`` sit at the top of each panel.

Inputs (all under ``tmp/lmarena_ci_results/external/``):
    {amazon,lmarena,gwas}_bias_se_pt.csv      columns: bias, bias_sd
    {amazon,lmarena,gwas}_npmle_prior_pt.csv  columns: atom, weight
    {amazon,lmarena,gwas}_param_prior_pt.csv  columns: hat_mu_b, hat_var_b

Output:
    tmp/plots/prior_plot.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm

from _style import (
    LABEL_FS,
    TICK_FS,
    LEGEND_FS,
    COLTITLE_FS,
    SAVE_DPI,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP = PROJECT_ROOT / "tmp"
EXT = TMP / "lmarena_ci_results" / "external"

PARAM_COLOR = "firebrick"
NPMLE_COLOR = "steelblue"

# Per-panel tuning (histogram bins / spike-height scaling) carried over
# from the original colleague notebook so the visuals stay comparable.
# ``param_spike_height`` only fires for GWAS, where the fitted Gaussian
# prior has near-zero variance and is best rendered as a single spike.
# ``sample_n`` down-samples the bias vector before computing the
# expensive per-row marginal density (only relevant for GWAS).
PANEL_CONFIG = [
    dict(
        key="lmarena",  title="(a) LMArena",
        bias_csv="lmarena_bias_se_pt.csv",
        npmle_csv="lmarena_npmle_prior_pt.csv",
        param_csv="lmarena_param_prior_pt.csv",
        hist_bins=25, npmle_spike_scale=5.0,
        param_spike_height=None, sample_n=None,
        # gamma_check=0.992,
    ),
    dict(
        key="amazon",   title="(b) Amazon",
        bias_csv="amazon_bias_se_pt.csv",
        npmle_csv="amazon_npmle_prior_pt.csv",
        param_csv="amazon_param_prior_pt.csv",
        hist_bins=20, npmle_spike_scale=20.0,
        param_spike_height=None, sample_n=None,
        # gamma_check=0.892,
    ),
    dict(
        key="gwas",     title="(c) GWAS",
        bias_csv="gwas_bias_se_pt.csv",
        npmle_csv="gwas_npmle_prior_pt.csv",
        param_csv="gwas_param_prior_pt.csv",
        hist_bins=25, npmle_spike_scale=25.0,
        param_spike_height=25.0, sample_n=10_000,
        # gamma_check=0.752,
    ),
]


# ----------------------------------------------------------------------------
# Panel renderer
# ----------------------------------------------------------------------------

def _draw_panel(ax, *, bias_csv, npmle_csv, param_csv,
                hist_bins, npmle_spike_scale, param_spike_height,
                sample_n, title, gamma_check=None, show_legend=False):
    bias_df = pd.read_csv(bias_csv)
    npmle_df = pd.read_csv(npmle_csv)
    param_df = pd.read_csv(param_csv)

    bias_full = bias_df["bias"].to_numpy(dtype=float)
    bias_sd_full = bias_df["bias_sd"].to_numpy(dtype=float)
    valid = (np.isfinite(bias_full) & np.isfinite(bias_sd_full)
             & (bias_sd_full > 0))
    bias_full = bias_full[valid]
    bias_sd_full = bias_sd_full[valid]

    # Down-sample large bias vectors so the per-row marginal-density
    # accumulation stays cheap; histogram still uses the full vector.
    if sample_n is not None and len(bias_full) > sample_n:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(bias_full), size=sample_n, replace=False)
        bias = bias_full[idx]
        bias_sd = bias_sd_full[idx]
    else:
        bias = bias_full
        bias_sd = bias_sd_full

    mu_b = float(param_df["hat_mu_b"].iloc[0])
    var_b = float(param_df["hat_var_b"].iloc[0])
    sd_b = float(np.sqrt(var_b))

    atoms = npmle_df["atom"].to_numpy(dtype=float)
    weights = npmle_df["weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    x_min = min(np.nanmin(bias), mu_b - 4 * sd_b, np.nanmin(atoms))
    x_max = max(np.nanmax(bias), mu_b + 4 * sd_b, np.nanmax(atoms))
    pad = 0.1 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 1000)

    # Parametric marginal: average over i of N(mu, var_b + bias_sd_i^2).
    param_marginal = np.mean(
        norm.pdf(x_grid[:, None], loc=mu_b,
                 scale=np.sqrt(var_b + bias_sd[None, :] ** 2)),
        axis=1,
    )
    # NPMLE marginal: average over i of the mixture sum_k w_k N(atom_k, bias_sd_i^2).
    npmle_marginal = np.zeros_like(x_grid)
    for sd_i in bias_sd:
        npmle_marginal += np.sum(
            weights[None, :] * norm.pdf(
                x_grid[:, None], loc=atoms[None, :], scale=sd_i),
            axis=1,
        )
    npmle_marginal /= len(bias_sd)

    param_prior_density = norm.pdf(x_grid, loc=mu_b, scale=sd_b)
    npmle_prior_spike_height = weights * npmle_spike_scale

    # Observed bias histogram (uses the full, un-sampled vector).
    ax.hist(bias_full, bins=hist_bins, density=True, color="gray",
            alpha=0.18, edgecolor="gray", linewidth=0.6)
    # Marginal densities of $\hat b_i$ -- dotted.
    ax.plot(x_grid, param_marginal, color=PARAM_COLOR,
            linewidth=1.6, linestyle=":")
    ax.plot(x_grid, npmle_marginal, color=NPMLE_COLOR,
            linewidth=1.6, linestyle=":")
    # Priors on $b_i$ -- solid for parametric, vlines for NPMLE atoms.
    ax.plot(x_grid, param_prior_density, color=PARAM_COLOR,
            linewidth=2.2, linestyle="-")
    if param_spike_height is not None:
        ax.vlines(mu_b, ymin=0, ymax=param_spike_height,
                  color=PARAM_COLOR, linewidth=2.2, alpha=0.9)
    ax.vlines(atoms, ymin=0, ymax=npmle_prior_spike_height,
              color=NPMLE_COLOR, linewidth=2.2, alpha=0.9)

    ax.set_xlabel(r"$\hat b_i$ / $b_i$", fontsize=LABEL_FS)
    ax.set_ylabel("Density", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(title, fontsize=COLTITLE_FS, fontweight="bold", pad=4)
    ax.grid(True, alpha=0.25, which="both")

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=PARAM_COLOR, lw=2.2, label="Normal"),
            Line2D([0], [0], color=NPMLE_COLOR, lw=2.2, label="NPMLE"),
        ]
        ax.legend(handles=legend_handles, frameon=False,
                  fontsize=LEGEND_FS, handletextpad=0.4, loc="best")

    # Bottom-right: estimated efficiency gain / shrinkage factor
    # gamma-check, set in axes-fraction coords so it sits a small fixed
    # margin from the right/bottom edges regardless of data range.
    if gamma_check is not None:
        ax.text(
            0.96, 0.08, rf"$\bar{{\gamma}}: {gamma_check:.3f}$",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=LABEL_FS, color="black",
        )


# ----------------------------------------------------------------------------
# Figure assembly
# ----------------------------------------------------------------------------

def make_plot(out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    for i, (ax, cfg) in enumerate(zip(axes, PANEL_CONFIG)):
        _draw_panel(
            ax,
            bias_csv=EXT / cfg["bias_csv"],
            npmle_csv=EXT / cfg["npmle_csv"],
            param_csv=EXT / cfg["param_csv"],
            hist_bins=cfg["hist_bins"],
            npmle_spike_scale=cfg["npmle_spike_scale"],
            param_spike_height=cfg["param_spike_height"],
            sample_n=cfg["sample_n"],
            title=cfg["title"],
            # gamma_check=cfg["gamma_check"],
            show_legend=(i == 0),
        )

    fig.tight_layout(pad=1.0)

    fig.savefig(out_path, bbox_inches="tight", dpi=SAVE_DPI)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=TMP / "plots" / "prior_plot.pdf",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.out)


if __name__ == "__main__":
    main()
