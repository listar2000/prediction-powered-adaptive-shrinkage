"""Pseudo-oracle test for target-dependent violations of bias exchangeability.

The rebiasing model assumes that task biases are draws from one common law
``G`` that does not depend on the target ``theta_i``.  With one latent bias per
task, unrestricted exchangeability is not itself testable.  This module tests
a concrete observable implication using the full LM Arena corpus as the
allowed pseudo oracle:

    theta_tilde_i = mean(human preference in task i),
    b_tilde_i = mean(Bradley--Terry prediction in task i) - theta_tilde_i.

Tasks are split into equal-count target bins.  The statistic is the largest
pairwise one-dimensional Wasserstein distance between their pseudo-bias
distributions, normalized by the overall pseudo-bias standard deviation.
A permutation test randomly reassigns pseudo-biases to the fixed target bins.
Taking the maximum before permutation calibration automatically accounts for
searching over all bin pairs.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from pas.statistical_tests.base import StatisticalTestResult


def load_lmarena_pseudo_oracle(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Aggregate LM Arena row-level data into one pseudo-oracle row per task."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    required = {"group_id", "model_a", "model_b", "winner", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"LM Arena CSV is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("LM Arena columns used by the diagnostic contain NaNs")

    pair_counts = frame.groupby("group_id")[["model_a", "model_b"]].nunique()
    if (pair_counts.to_numpy() != 1).any():
        raise ValueError("each group_id must identify one fixed model pair")

    summary = (
        frame.groupby("group_id", sort=True)
        .agg(
            model_a=("model_a", "first"),
            model_b=("model_b", "first"),
            n=("winner", "size"),
            theta=("winner", "mean"),
            prediction_mean=("prediction", "mean"),
        )
        .reset_index()
    )
    summary["bias"] = summary["prediction_mean"] - summary["theta"]

    residual_frame = frame[["group_id"]].copy()
    residual_frame["prediction_error"] = (
        frame["prediction"].astype(float) - frame["winner"].astype(float)
    )
    residual_variance = residual_frame.groupby("group_id")[
        "prediction_error"
    ].var(ddof=1)
    summary["bias_se"] = np.sqrt(
        summary["group_id"].map(residual_variance).to_numpy(dtype=float)
        / summary["n"].to_numpy(dtype=float)
    )

    predictions = frame["prediction"].to_numpy(dtype=float)
    summary.attrs.update(
        {
            "csv_path": str(path),
            "prediction_n_unique": int(np.unique(predictions).size),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "prediction_is_binary": bool(
                np.unique(predictions).size <= 2
                and np.all(np.isin(np.unique(predictions), [0.0, 1.0]))
            ),
        }
    )
    return summary


def equal_count_bins(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Deterministically form approximately equal-size bins, including ties."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if values.size < n_bins:
        raise ValueError("n_bins cannot exceed the number of observations")
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.size, dtype=int)
    bins[order] = np.minimum(
        np.arange(values.size) * n_bins // values.size,
        n_bins - 1,
    )
    return bins


def _distribution_shift_statistic(
    bias: np.ndarray,
    bins: np.ndarray,
) -> tuple[float, float, tuple[int, int]]:
    scale = float(np.std(bias, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.0, 0.0, (0, 0)
    distances = [
        (
            float(wasserstein_distance(bias[bins == left], bias[bins == right])),
            (int(left), int(right)),
        )
        for left, right in combinations(np.unique(bins), 2)
    ]
    raw_distance, pair = max(distances, key=lambda item: item[0])
    return float(raw_distance / scale), raw_distance, pair


def _bin_summaries(
    theta: np.ndarray,
    bias: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, float]]:
    output = []
    for bin_index in np.unique(bins):
        mask = bins == bin_index
        output.append(
            {
                "bin": int(bin_index + 1),
                "n": int(np.sum(mask)),
                "theta_min": float(np.min(theta[mask])),
                "theta_max": float(np.max(theta[mask])),
                "theta_mean": float(np.mean(theta[mask])),
                "bias_mean": float(np.mean(bias[mask])),
                "bias_median": float(np.median(bias[mask])),
                "bias_sd": float(np.std(bias[mask], ddof=1)),
            }
        )
    return output


class BiasExchangeabilityTest:
    """Permutation test of bias-distribution invariance across target bins."""

    name = "bias_exchangeability_by_target"

    def run(
        self,
        summary: pd.DataFrame,
        *,
        n_bins: int = 4,
        permutations: int = 9999,
        seed: int = 8273,
    ) -> StatisticalTestResult:
        required = {"theta", "bias"}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"summary is missing columns: {sorted(missing)}")
        if permutations < 1:
            raise ValueError("permutations must be positive")

        theta = summary["theta"].to_numpy(dtype=float)
        bias = summary["bias"].to_numpy(dtype=float)
        if theta.size < 2 * n_bins:
            raise ValueError("each target bin must contain at least two tasks")
        if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(bias)):
            raise ValueError("theta and bias must contain only finite values")

        bins = equal_count_bins(theta, n_bins)
        observed, raw_distance, max_pair = _distribution_shift_statistic(
            bias, bins
        )
        rng = np.random.default_rng(seed)
        null = np.empty(permutations, dtype=float)
        for index in range(permutations):
            null[index] = _distribution_shift_statistic(
                rng.permutation(bias), bins
            )[0]
        p_value = float((1.0 + np.sum(null >= observed)) / (permutations + 1.0))
        null_mean = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1))
        standardized = (
            float((observed - null_mean) / null_sd) if null_sd > 0.0 else 0.0
        )

        metadata = {
            "n_bins": int(n_bins),
            "permutations": int(permutations),
            "seed": int(seed),
            "bin_index": (bins + 1).tolist(),
            "bin_summaries": _bin_summaries(theta, bias, bins),
            "bias_sd": float(np.std(bias, ddof=1)),
            "max_pair": [int(max_pair[0] + 1), int(max_pair[1] + 1)],
            "max_pair_wasserstein": float(raw_distance),
            "null_mean": null_mean,
            "null_sd": null_sd,
            "observed_null_sd_units": standardized,
            "permutation_percentile": float(
                (np.sum(null <= observed) + 0.5) / (permutations + 1.0)
            ),
            "pseudo_oracle_definition": {
                "theta": "full-task mean of winner",
                "bias": "full-task mean of prediction minus full-task mean of winner",
            },
        }
        return StatisticalTestResult(
            name=self.name,
            statistic=observed,
            p_value=p_value,
            null_hypothesis=(
                "the pseudo-bias distribution is invariant across the fixed "
                "equal-count target bins"
            ),
            alternative=(
                "at least one pair of target bins has a different pseudo-bias "
                "distribution"
            ),
            method=(
                "Monte Carlo permutation test using the maximum pairwise "
                "1-Wasserstein distance, normalized by the overall bias SD"
            ),
            n_observations=theta.size,
            effect_size=observed,
            effect_size_name="max pairwise Wasserstein distance / bias SD",
            metadata=metadata,
            null_distribution=null,
        )


def test_bias_exchangeability(
    summary: pd.DataFrame,
    **kwargs,
) -> StatisticalTestResult:
    """Functional wrapper for :class:`BiasExchangeabilityTest`."""
    return BiasExchangeabilityTest().run(summary, **kwargs)


__all__ = [
    "BiasExchangeabilityTest",
    "equal_count_bins",
    "load_lmarena_pseudo_oracle",
    "test_bias_exchangeability",
]
