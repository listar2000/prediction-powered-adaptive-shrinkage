"""Visualization utilities for PAS statistical-test results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from pas.statistical_tests.base import StatisticalTestResult


def plot_permutation_null(
    result: StatisticalTestResult,
    *,
    ax=None,
    bins: int = 45,
):
    """Plot a standardized permutation null and the observed statistic."""
    if result.null_distribution is None:
        raise ValueError("result does not contain a null distribution")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.4))
    null = np.asarray(result.null_distribution, dtype=float)
    ax.hist(null, bins=bins, density=True, alpha=0.7)
    ax.axvline(result.statistic, linestyle="--", linewidth=2.0)
    ax.set_xlabel("Test statistic")
    ax.set_ylabel("Permutation density")
    ax.set_title(
        f"Permutation null; observed={result.statistic:.3f}, "
        f"p={result.p_value:.4g}"
    )
    return ax


def plot_bias_versus_target(
    summary: pd.DataFrame,
    bins: np.ndarray,
    *,
    ax=None,
    title: str = "A. Bias versus target\n(points; quartile means +/- 1.96 SE)",
):
    """Panel A: task-level bias against target, with per-bin means +/- 1.96 SE.

    This panel is purely descriptive, so it takes the bin labels directly and
    does not require a fitted :class:`StatisticalTestResult`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    theta = summary["theta"].to_numpy(dtype=float)
    bias = summary["bias"].to_numpy(dtype=float)
    bins = np.asarray(bins, dtype=int).reshape(-1)

    ax.scatter(theta, bias, s=22, alpha=0.55, edgecolors="none")
    centers, means, errors = [], [], []
    for bin_index in np.unique(bins):
        mask = bins == bin_index
        centers.append(float(np.mean(theta[mask])))
        means.append(float(np.mean(bias[mask])))
        errors.append(
            float(1.96 * np.std(bias[mask], ddof=1) / np.sqrt(np.sum(mask)))
        )
    ax.errorbar(centers, means, yerr=errors, marker="o", capsize=3, linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Full-corpus human win rate")
    ax.set_ylabel("Pseudo-oracle prediction bias")
    ax.set_title(title)
    return ax


def plot_bias_exchangeability(
    summary: pd.DataFrame,
    result: StatisticalTestResult,
    output_path: Optional[Union[str, Path]] = None,
):
    """Create a three-panel visualization for the target-bin diagnostic."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if result.name != "bias_exchangeability_by_target":
        raise ValueError("result is not a bias-exchangeability test result")
    theta = summary["theta"].to_numpy(dtype=float)
    bias = summary["bias"].to_numpy(dtype=float)
    bins = np.asarray(result.metadata["bin_index"], dtype=int) - 1
    unique_bins = np.unique(bins)

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)

    # A. Task-level relationship plus descriptive bin means.
    plot_bias_versus_target(summary, bins, ax=axes[0])

    # B. Entire empirical distributions; W1 compares their horizontal gaps.
    ax = axes[1]
    for bin_index in unique_bins:
        values = np.sort(bias[bins == bin_index])
        ecdf = np.arange(1, values.size + 1) / values.size
        ax.step(values, ecdf, where="post", label=f"Q{bin_index + 1}")
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Pseudo-oracle prediction bias")
    ax.set_ylabel("Empirical CDF")
    max_pair = result.metadata["max_pair"]
    ax.set_title(
        "B. Bias distributions by target quartile\n"
        f"largest gap: Q{max_pair[0]} vs Q{max_pair[1]}"
    )
    ax.legend(title="Target bin", fontsize=8)

    # C. Permutation calibration of the omnibus maximum.
    plot_permutation_null(result, ax=axes[2])
    axes[2].set_title(
        "C. Permutation calibration\n"
        f"T={result.statistic:.3f}; p={result.p_value:.4g}; "
        f"z_null={result.metadata['observed_null_sd_units']:.2f}"
    )

    figure.suptitle(
        "LM Arena: pseudo-oracle diagnostic for target-dependent bias",
        fontsize=14,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
    return figure


__all__ = [
    "plot_bias_exchangeability",
    "plot_bias_versus_target",
    "plot_permutation_null",
]
