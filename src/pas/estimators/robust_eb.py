"""Robust empirical-Bayes shrinkage of unbiased task estimators.

This is a Python translation of the default estimator produced by
``kolesarm/ebci::ebci`` (Armstrong, Kolesar, and Plagborg-Moller, 2022).
Unlike empirical-Bayes *rebiasing*, this baseline does not use a biased
prediction-only estimator.  It starts from an unbiased estimator ``Y_i`` and
shrinks it toward a weighted regression fit,

    theta_hat_i = mu_i + w_i (Y_i - mu_i),
    w_i = mu2 / (mu2 + se_i^2).

The PAS-facing default uses the power-tuned PPI estimator as ``Y_i``, shrinks
toward its inverse-variance-weighted grand mean (the R formula ``Y ~ 1``),
uses weights ``1 / se_i^2``, and applies the official posterior-mean
truncation finite-sample correction (``fs_correction='PMT'``).  Those defaults
match the configuration used by the first author for the rebuttal baseline.

The moment formulas are ported from ``R/eb.R`` in the MIT-licensed official
``ebci`` repository.  See ``THIRD_PARTY_NOTICES.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

from pas.estimators.ppi_estimators import (
    get_pt_ppi_estimators,
    get_vanilla_ppi_estimators,
)

ArrayLike = Union[Sequence[float], np.ndarray]
BaseEstimator = Literal["pt", "ppi", "classical"]
FiniteSampleCorrection = Literal["PMT", "FPLIB", "none"]


@dataclass
class RobustEBFit:
    """Fitted robust-EB shrinkage rule and its per-task quantities."""

    estimate: np.ndarray
    unshrunk: np.ndarray
    standard_error: np.ndarray
    regression_mean: np.ndarray
    residuals: np.ndarray
    shrinkage: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    coefficients: np.ndarray
    mu2: float
    raw_mu2: float
    kappa: float
    raw_kappa: float
    finite_sample_correction: str


def _as_1d(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("weights must have positive sum")
    return float(np.sum(weights * values) / denominator)


def _weighted_variance_estimate(values: np.ndarray, weights: np.ndarray) -> float:
    """Variance expression used by ``ebci`` for its FPLIB correction."""
    denominator = float(np.sum(weights) ** 2 - np.sum(weights**2))
    if denominator <= 0.0:
        return 0.0
    mean = _weighted_mean(values, weights)
    numerator = float(np.sum(weights**2 * (values**2 - mean**2)))
    return max(numerator / denominator, 0.0)


def _truncated_normal_mean(mean: float, variance: float) -> float:
    """Mean of ``N(mean, variance)`` conditional on being positive."""
    if variance <= 0.0:
        return max(float(mean), 0.0)
    sd = np.sqrt(variance)
    z = float(mean) / sd
    mills = np.exp(norm.logpdf(z) - log_ndtr(z))
    return float(mean + sd * mills)


def _moment_estimates(
    residuals: np.ndarray,
    standard_error: np.ndarray,
    weights: np.ndarray,
    *,
    fs_correction: str,
    kappa: Optional[float],
) -> tuple[float, float, float, float]:
    """Port of ``ebci:::moments``.

    Returns ``(mu2, kappa, raw_mu2, raw_kappa)``.
    """
    correction = str(fs_correction)
    if correction.lower() == "pmt":
        correction = "PMT"
    elif correction.lower() == "fplib":
        correction = "FPLIB"
    elif correction.lower() == "none":
        correction = "none"
    else:
        raise ValueError("fs_correction must be 'PMT', 'FPLIB', or 'none'")

    eps = np.asarray(residuals, dtype=float)
    se = np.asarray(standard_error, dtype=float)
    wgt = np.asarray(weights, dtype=float)
    w2 = eps**2 - se**2
    w4 = eps**4 - 6.0 * se**2 * eps**2 + 3.0 * se**4
    raw_mu2 = _weighted_mean(w2, wgt)
    raw_mu4 = _weighted_mean(w4, wgt)

    sum_weights = float(np.sum(wgt))
    denominator2 = sum_weights * float(np.mean(wgt * se**2))
    denominator4 = sum_weights * float(np.mean(wgt * se**4))
    pmt_trim2 = (
        2.0 * float(np.mean(wgt**2 * se**4)) / denominator2
        if denominator2 > 0.0
        else 0.0
    )
    pmt_trim4 = (
        32.0 * float(np.mean(wgt**2 * se**8)) / denominator4
        if denominator4 > 0.0
        else 0.0
    )

    if correction == "none":
        mu2 = max(raw_mu2, 0.0)
    elif correction == "PMT":
        mu2 = max(raw_mu2, pmt_trim2)
    else:
        mu2 = _truncated_normal_mean(
            raw_mu2, _weighted_variance_estimate(w2, wgt)
        )

    if raw_mu2 == 0.0:
        raw_kappa = float("nan")
    else:
        raw_kappa = float(raw_mu4 / raw_mu2**2)

    if kappa is not None:
        estimated_kappa = float(kappa)
        if np.isnan(estimated_kappa) or estimated_kappa < 1.0:
            raise ValueError("kappa must be at least one")
    elif mu2 <= 0.0:
        estimated_kappa = float("inf")
    elif correction == "none":
        estimated_kappa = max(float(raw_mu4 / mu2**2), 1.0)
    elif correction == "PMT":
        estimated_kappa = max(
            float(raw_mu4 / mu2**2),
            float(1.0 + pmt_trim4 / mu2**2),
        )
    else:
        centered_fourth = w4 - 2.0 * mu2 * w2
        estimated_kappa = float(
            1.0
            + _truncated_normal_mean(
                raw_mu4 - raw_mu2**2,
                _weighted_variance_estimate(centered_fourth, wgt),
            )
            / mu2**2
        )
        estimated_kappa = max(estimated_kappa, 1.0)

    return float(mu2), estimated_kappa, float(raw_mu2), raw_kappa


def _prepare_design(
    n: int,
    design: Optional[np.ndarray],
    shrink_to: Literal["grand_mean", "zero"],
) -> np.ndarray:
    if design is None:
        if shrink_to == "grand_mean":
            return np.ones((n, 1), dtype=float)
        if shrink_to == "zero":
            return np.empty((n, 0), dtype=float)
        raise ValueError("shrink_to must be 'grand_mean' or 'zero'")
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ValueError("design must have one row per task")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")
    return matrix


def _weighted_regression(
    y: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if design.shape[1] == 0:
        return np.empty(0, dtype=float), np.zeros_like(y)
    root_weights = np.sqrt(weights)
    weighted_x = design * root_weights[:, None]
    weighted_y = y * root_weights
    coefficients = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    return coefficients, design @ coefficients


def fit_robust_eb(
    y: ArrayLike,
    standard_error: ArrayLike,
    *,
    weights: Optional[ArrayLike] = None,
    design: Optional[np.ndarray] = None,
    shrink_to: Literal["grand_mean", "zero"] = "grand_mean",
    fs_correction: FiniteSampleCorrection = "PMT",
    kappa: Optional[float] = None,
) -> RobustEBFit:
    """Fit the robust-EB point estimator from unbiased normal-means inputs.

    Parameters mirror the core pieces of ``ebci::ebci``.  ``design=None`` and
    ``shrink_to='grand_mean'`` correspond to the R formula ``Y ~ 1``;
    ``shrink_to='zero'`` corresponds to ``Y ~ 0``.
    """
    y_array = _as_1d("y", y)
    se_array = _as_1d("standard_error", standard_error)
    if y_array.size != se_array.size:
        raise ValueError("y and standard_error must have the same length")
    if np.any(~np.isfinite(y_array)):
        raise ValueError("y must contain only finite values")
    if np.any(~np.isfinite(se_array)) or np.any(se_array < 0.0):
        raise ValueError("standard_error must be finite and non-negative")

    if weights is None:
        weight_array = np.ones_like(y_array)
    else:
        weight_array = _as_1d("weights", weights)
        if weight_array.size != y_array.size:
            raise ValueError("weights must have the same length as y")
        if np.any(~np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
            raise ValueError("weights must be finite and strictly positive")

    design_array = _prepare_design(y_array.size, design, shrink_to)
    coefficients, regression_mean = _weighted_regression(
        y_array, design_array, weight_array
    )
    residuals = y_array - regression_mean
    mu2, estimated_kappa, raw_mu2, raw_kappa = _moment_estimates(
        residuals,
        se_array,
        weight_array,
        fs_correction=fs_correction,
        kappa=kappa,
    )

    denominator = mu2 + se_array**2
    shrinkage = np.divide(
        mu2,
        denominator,
        out=np.zeros_like(se_array),
        where=denominator > 0.0,
    )
    estimate = regression_mean + shrinkage * residuals
    correction_name = (
        "PMT" if str(fs_correction).lower() == "pmt"
        else "FPLIB" if str(fs_correction).lower() == "fplib"
        else "none"
    )
    return RobustEBFit(
        estimate=estimate,
        unshrunk=y_array,
        standard_error=se_array,
        regression_mean=regression_mean,
        residuals=residuals,
        shrinkage=shrinkage,
        weights=weight_array,
        design=design_array,
        coefficients=coefficients,
        mu2=mu2,
        raw_mu2=raw_mu2,
        kappa=estimated_kappa,
        raw_kappa=raw_kappa,
        finite_sample_correction=correction_name,
    )


def get_unbiased_estimates_and_ses(
    data,
    *,
    base_estimator: BaseEstimator = "pt",
    share_var: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an unbiased PAS/PPI estimator and its plug-in standard error."""
    base_estimator = str(base_estimator).lower()
    if base_estimator not in {"pt", "ppi", "classical"}:
        raise ValueError("base_estimator must be 'pt', 'ppi', or 'classical'")

    if base_estimator == "pt":
        estimates, lambdas = get_pt_ppi_estimators(
            data, share_var=share_var, get_lambdas=True
        )
    elif base_estimator == "ppi":
        estimates = get_vanilla_ppi_estimators(data)
        lambdas = np.ones(data.M, dtype=float)
    else:
        estimates = np.asarray(
            [np.mean(np.asarray(y_i, dtype=float)) for y_i in data.y_labelled],
            dtype=float,
        )
        standard_errors = np.asarray(
            [
                np.std(np.asarray(y_i, dtype=float), ddof=1)
                / np.sqrt(np.asarray(y_i).size)
                for y_i in data.y_labelled
            ],
            dtype=float,
        )
        return estimates, standard_errors

    imputed_ses = np.asarray(
        [
            np.std(lam * np.asarray(pred_u, dtype=float), ddof=1) / np.sqrt(N)
            for lam, pred_u, N in zip(lambdas, data.pred_unlabelled, data.Ns)
        ],
        dtype=float,
    )
    rectifier_ses = np.asarray(
        [
            np.std(
                np.asarray(y_i, dtype=float)
                - lam * np.asarray(pred_l, dtype=float),
                ddof=1,
            )
            / np.sqrt(n)
            for lam, y_i, pred_l, n in zip(
                lambdas, data.y_labelled, data.pred_labelled, data.ns
            )
        ],
        dtype=float,
    )
    return np.asarray(estimates, dtype=float), np.sqrt(
        imputed_ses**2 + rectifier_ses**2
    )


def fit_robust_eb_dataset(
    data,
    *,
    base_estimator: BaseEstimator = "pt",
    share_var: bool = True,
    weights: Union[Literal["inverse_variance", "equal"], ArrayLike] = "inverse_variance",
    design: Optional[np.ndarray] = None,
    shrink_to: Literal["grand_mean", "zero"] = "grand_mean",
    fs_correction: FiniteSampleCorrection = "PMT",
    kappa: Optional[float] = None,
) -> RobustEBFit:
    """Fit robust EB to a :class:`pas.datasets.dataset.PasDataset` object."""
    estimates, standard_errors = get_unbiased_estimates_and_ses(
        data, base_estimator=base_estimator, share_var=share_var
    )
    if isinstance(weights, str):
        if weights == "inverse_variance":
            variances = standard_errors**2
            positive = variances > 0.0
            if not np.any(positive):
                fit_weights = np.ones_like(standard_errors)
            else:
                # A plug-in SE can be exactly zero in a small Bernoulli
                # sample.  The formal EB limit leaves that task unshrunk; for
                # the cross-task WLS/moment fit we cap its infinite precision
                # at the largest finite inverse-variance weight.
                fit_weights = np.empty_like(standard_errors)
                fit_weights[positive] = 1.0 / variances[positive]
                fit_weights[~positive] = np.max(fit_weights[positive])
        elif weights == "equal":
            fit_weights = np.ones_like(standard_errors)
        else:
            raise ValueError("weights must be 'inverse_variance', 'equal', or an array")
    else:
        fit_weights = np.asarray(weights, dtype=float)
    return fit_robust_eb(
        estimates,
        standard_errors,
        weights=fit_weights,
        design=design,
        shrink_to=shrink_to,
        fs_correction=fs_correction,
        kappa=kappa,
    )


def get_robust_eb_estimators(data, **kwargs) -> np.ndarray:
    """Return robust-EB estimates; registry-friendly PAS estimator wrapper."""
    return fit_robust_eb_dataset(data, **kwargs).estimate


__all__ = [
    "ArrayLike",
    "BaseEstimator",
    "FiniteSampleCorrection",
    "RobustEBFit",
    "fit_robust_eb",
    "get_unbiased_estimates_and_ses",
    "fit_robust_eb_dataset",
    "get_robust_eb_estimators",
]
