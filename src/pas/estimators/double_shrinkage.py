"""Rosenman--Dominici--Miratrix empirical-Bayes double shrinkage.

Implements the baseline from Rosenman, Dominici, and Miratrix,
"Empirical Bayes Double Shrinkage for Combining Biased and Unbiased Causal
Estimates" (arXiv:2309.06727).  The estimator combines an unbiased estimator
``u_i`` with an independent, possibly biased estimator ``v_i`` and then
applies a second shrinkage toward a fixed center.  This module implements all
four hyperparameter choices in the paper (MM1, MM2, marginal MLE, and URE);
the companion robust empirical-Bayes confidence intervals live in
:mod:`pas.intervals.double_shrinkage_cis`.

For PPI datasets the faithful mapping is deliberately *not* based on PT:
``u_i`` is the labelled outcome mean and ``v_i`` is the prediction mean on the
independent unlabelled sample.  This matches the baseline paper's conditional
independence model.  A fixed ``center=0.5`` is natural for LM Arena win rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize

DoubleShrinkageMethod = Literal["mm1", "mm2", "mle", "ure"]


@dataclass
class DoubleShrinkageFit:
    """Fitted double-shrinkage rule and per-task derived quantities."""

    method: str
    center: float
    eta2: float
    gamma2: float
    raw_eta2: float
    raw_gamma2: float
    finite_sample_correction: str
    unbiased: np.ndarray
    biased: np.ndarray
    var_unbiased: np.ndarray
    var_biased: np.ndarray
    valid: np.ndarray
    lambda_: np.ndarray
    a: np.ndarray
    estimate: np.ndarray
    sampling_variance: np.ndarray
    normalized_bias_m2: np.ndarray
    objective: Optional[float] = None
    optimizer_success: Optional[bool] = None


def _as_1d(name: str, value: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def independent_estimator_components(data):
    """Extract the original baseline's independent estimator pair from PPI.

    Returns
    -------
    unbiased:
        Labelled human-outcome means ``mean(Y_i)``.
    biased:
        Unlabelled prediction means ``mean(h(Xtilde_i))``.
    var_unbiased, var_biased:
        Usual plug-in variances of the two sample means.
    """
    unbiased, biased, var_unbiased, var_biased = [], [], [], []
    for y_i, pred_u_i in zip(data.y_labelled, data.pred_unlabelled):
        y_i = np.asarray(y_i, dtype=float)
        pred_u_i = np.asarray(pred_u_i, dtype=float)
        unbiased.append(float(np.mean(y_i)) if y_i.size else np.nan)
        biased.append(float(np.mean(pred_u_i)) if pred_u_i.size else np.nan)
        var_unbiased.append(
            float(np.var(y_i, ddof=1) / y_i.size) if y_i.size > 1 else np.nan
        )
        var_biased.append(
            float(np.var(pred_u_i, ddof=1) / pred_u_i.size)
            if pred_u_i.size > 1
            else np.nan
        )
    return tuple(
        np.asarray(x, dtype=float)
        for x in (unbiased, biased, var_unbiased, var_biased)
    )


def _valid_mask(
    unbiased: np.ndarray,
    biased: np.ndarray,
    var_unbiased: np.ndarray,
    var_biased: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(unbiased)
        & np.isfinite(biased)
        & np.isfinite(var_unbiased)
        & np.isfinite(var_biased)
        & (var_unbiased > 0.0)
        & (var_biased > 0.0)
    )


def _pmt_floor(measurement_variance: np.ndarray) -> float:
    """Equal-weight posterior moment-truncation floor used by ``ebci``."""
    v = np.asarray(measurement_variance, dtype=float)
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size == 0:
        return 0.0
    denominator = v.size * float(np.mean(v))
    if denominator <= 0.0:
        return 0.0
    return float(2.0 * np.mean(v**2) / denominator)


def _moment_hyperparameters(
    u: np.ndarray,
    b: np.ndarray,
    vu: np.ndarray,
    vb: np.ndarray,
    method: Literal["mm1", "mm2"],
):
    eta2 = max(float(np.mean(u**2 - vu)), 0.0)
    if method == "mm1":
        gamma2 = max(float(np.mean((u - b) ** 2 - vu - vb)), 0.0)
    else:
        gamma2 = max(float(np.mean(b**2 - u**2 + vu - vb)), 0.0)
    return eta2, gamma2


def _weights(
    eta2: float,
    gamma2: float,
    vu: np.ndarray,
    vb: np.ndarray,
):
    gamma_plus_vb = gamma2 + vb
    lambda_ = gamma_plus_vb / (gamma_plus_vb + vu)
    denominator = vu * gamma_plus_vb + eta2 * (gamma_plus_vb + vu)
    a = np.divide(
        eta2 * (gamma_plus_vb + vu),
        denominator,
        out=np.zeros_like(vu),
        where=denominator > 0.0,
    )
    return lambda_, a


def _double_shrinkage_estimate(
    eta2: float,
    gamma2: float,
    u: np.ndarray,
    b: np.ndarray,
    vu: np.ndarray,
    vb: np.ndarray,
):
    lambda_, a = _weights(eta2, gamma2, vu, vb)
    estimate = a * (lambda_ * u + (1.0 - lambda_) * b)
    return estimate, lambda_, a


def _optimization_upper_bound(
    u: np.ndarray,
    b: np.ndarray,
    vu: np.ndarray,
    vb: np.ndarray,
) -> float:
    scale = max(
        float(np.mean(u**2 + vu)),
        float(np.mean(b**2 + vb)),
        float(np.mean(vu + vb)),
        1e-10,
    )
    return float(max(1000.0 * scale, 1e-6))


def _multistart_minimize(objective, starts, upper: float):
    candidates = []
    bounds = [(0.0, upper), (0.0, upper)]
    for start in starts:
        x0 = np.clip(np.asarray(start, dtype=float), 0.0, upper)
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append((float(result.fun), result.x, bool(result.success)))
    # Include starts and corners because URE can attain its optimum at a bound.
    for point in starts + [(0.0, 0.0), (upper, 0.0), (0.0, upper), (upper, upper)]:
        x = np.clip(np.asarray(point, dtype=float), 0.0, upper)
        val = float(objective(x))
        if np.isfinite(val):
            candidates.append((val, x, True))
    if not candidates:
        raise RuntimeError("double-shrinkage hyperparameter optimization failed")
    value, x, success = min(candidates, key=lambda item: item[0])
    return float(x[0]), float(x[1]), value, success


def _mle_hyperparameters(
    u: np.ndarray,
    b: np.ndarray,
    vu: np.ndarray,
    vb: np.ndarray,
):
    """Optimize the factorized marginal objective stated in the baseline."""
    mm1 = _moment_hyperparameters(u, b, vu, vb, "mm1")
    mm2 = _moment_hyperparameters(u, b, vu, vb, "mm2")
    upper = _optimization_upper_bound(u, b, vu, vb)

    def objective(x: np.ndarray) -> float:
        eta2, gamma2 = map(float, x)
        var_u = eta2 + vu
        var_b = eta2 + gamma2 + vb
        if np.any(var_u <= 0.0) or np.any(var_b <= 0.0):
            return np.inf
        return float(
            0.5
            * np.sum(
                np.log(var_u)
                + u**2 / var_u
                + np.log(var_b)
                + b**2 / var_b
            )
        )

    starts = [
        mm1,
        mm2,
        (max(float(np.mean(u**2 - vu)), 0.0),
         max(float(np.mean((b - u) ** 2 - vu - vb)), 0.0)),
        (upper * 1e-4, upper * 1e-4),
    ]
    return _multistart_minimize(objective, starts, upper)


def _ure_hyperparameters(
    u: np.ndarray,
    b: np.ndarray,
    vu: np.ndarray,
    vb: np.ndarray,
):
    mm1 = _moment_hyperparameters(u, b, vu, vb, "mm1")
    mm2 = _moment_hyperparameters(u, b, vu, vb, "mm2")
    mle_eta, mle_gamma, _, _ = _mle_hyperparameters(u, b, vu, vb)
    upper = _optimization_upper_bound(u, b, vu, vb)

    def objective(x: np.ndarray) -> float:
        eta2, gamma2 = map(float, x)
        psi, lambda_, a = _double_shrinkage_estimate(
            eta2, gamma2, u, b, vu, vb
        )
        return float(
            np.sum(vu)
            + np.sum((psi - u) ** 2)
            - 2.0 * np.sum(vu * (1.0 - a * lambda_))
        )

    starts = [
        mm1,
        mm2,
        (mle_eta, mle_gamma),
        (upper * 1e-4, upper * 1e-4),
        (upper * 0.1, upper * 0.1),
    ]
    return _multistart_minimize(objective, starts, upper)


def fit_double_shrinkage_components(
    unbiased: Union[Sequence[float], np.ndarray],
    biased: Union[Sequence[float], np.ndarray],
    var_unbiased: Union[Sequence[float], np.ndarray],
    var_biased: Union[Sequence[float], np.ndarray],
    *,
    method: DoubleShrinkageMethod = "mle",
    center: float = 0.0,
    finite_sample_correction: Literal["pmt", "none"] = "pmt",
    min_eta2: float = 0.0,
    min_gamma2: float = 0.0,
) -> DoubleShrinkageFit:
    """Fit a double-shrinkage rule from summary statistics.

    The target prior is centered at ``center`` while the bias prior remains
    centered at zero.  ``finite_sample_correction='pmt'`` applies the equal-
    weight posterior moment-truncation floor to both latent second moments;
    ``'none'`` reproduces the raw hyperparameter rules, including possible
    collapse when ``eta2=0``.
    """
    method = str(method).lower()
    if method not in {"mm1", "mm2", "mle", "ure"}:
        raise ValueError("method must be one of 'mm1', 'mm2', 'mle', or 'ure'")
    if finite_sample_correction not in {"pmt", "none"}:
        raise ValueError("finite_sample_correction must be 'pmt' or 'none'")
    if min_eta2 < 0.0 or min_gamma2 < 0.0:
        raise ValueError("minimum variances must be non-negative")

    unbiased_arr = _as_1d("unbiased", unbiased)
    biased_arr = _as_1d("biased", biased)
    vu_arr = _as_1d("var_unbiased", var_unbiased)
    vb_arr = _as_1d("var_biased", var_biased)
    n = unbiased_arr.size
    if not (biased_arr.size == vu_arr.size == vb_arr.size == n):
        raise ValueError("all component arrays must have the same length")

    valid = _valid_mask(unbiased_arr, biased_arr, vu_arr, vb_arr)
    if valid.sum() < 2:
        raise ValueError("at least two tasks with finite positive variances are required")

    u = unbiased_arr[valid] - float(center)
    b = biased_arr[valid] - float(center)
    vu = vu_arr[valid]
    vb = vb_arr[valid]

    objective = None
    success = None
    if method in {"mm1", "mm2"}:
        raw_eta2, raw_gamma2 = _moment_hyperparameters(u, b, vu, vb, method)
    elif method == "mle":
        raw_eta2, raw_gamma2, objective, success = _mle_hyperparameters(
            u, b, vu, vb
        )
    else:
        raw_eta2, raw_gamma2, objective, success = _ure_hyperparameters(
            u, b, vu, vb
        )

    eta_floor = float(min_eta2)
    gamma_floor = float(min_gamma2)
    if finite_sample_correction == "pmt":
        eta_floor = max(eta_floor, _pmt_floor(vu))
        # The observable difference b-u estimates the latent bias and has
        # measurement variance vu+vb under the baseline's independence model.
        gamma_floor = max(gamma_floor, _pmt_floor(vu + vb))

    eta2 = max(float(raw_eta2), eta_floor)
    gamma2 = max(float(raw_gamma2), gamma_floor)

    estimate = unbiased_arr.copy()
    lambda_full = np.full(n, np.nan, dtype=float)
    a_full = np.full(n, np.nan, dtype=float)
    sampling_var = np.full(n, np.nan, dtype=float)
    normalized_m2 = np.full(n, np.nan, dtype=float)

    psi, lambda_, a = _double_shrinkage_estimate(eta2, gamma2, u, b, vu, vb)
    estimate[valid] = float(center) + psi
    lambda_full[valid] = lambda_
    a_full[valid] = a

    base_var = lambda_**2 * vu + (1.0 - lambda_) ** 2 * vb
    sample_var = a**2 * base_var
    sampling_var[valid] = sample_var

    bias_second_moment = (
        (a - 1.0) ** 2 * eta2
        + (a * (1.0 - lambda_)) ** 2 * gamma2
    )
    m2 = np.divide(
        bias_second_moment,
        sample_var,
        out=np.zeros_like(sample_var),
        where=sample_var > 0.0,
    )
    normalized_m2[valid] = m2

    return DoubleShrinkageFit(
        method=method,
        center=float(center),
        eta2=eta2,
        gamma2=gamma2,
        raw_eta2=float(raw_eta2),
        raw_gamma2=float(raw_gamma2),
        finite_sample_correction=finite_sample_correction,
        unbiased=unbiased_arr,
        biased=biased_arr,
        var_unbiased=vu_arr,
        var_biased=vb_arr,
        valid=valid,
        lambda_=lambda_full,
        a=a_full,
        estimate=estimate,
        sampling_variance=sampling_var,
        normalized_bias_m2=normalized_m2,
        objective=objective,
        optimizer_success=success,
    )


def fit_double_shrinkage(
    data,
    *,
    method: DoubleShrinkageMethod = "mle",
    center: float = 0.0,
    finite_sample_correction: Literal["pmt", "none"] = "pmt",
    min_eta2: float = 0.0,
    min_gamma2: float = 0.0,
) -> DoubleShrinkageFit:
    """Fit double shrinkage to a ``PasDataset``-shaped PPI object."""
    components = independent_estimator_components(data)
    return fit_double_shrinkage_components(
        *components,
        method=method,
        center=center,
        finite_sample_correction=finite_sample_correction,
        min_eta2=min_eta2,
        min_gamma2=min_gamma2,
    )


def get_double_shrinkage_estimators(
    data,
    *,
    method: DoubleShrinkageMethod = "mle",
    center: float = 0.0,
    finite_sample_correction: Literal["pmt", "none"] = "pmt",
    min_eta2: float = 0.0,
    min_gamma2: float = 0.0,
) -> np.ndarray:
    """Return the fitted double-shrinkage point estimators."""
    return fit_double_shrinkage(
        data,
        method=method,
        center=center,
        finite_sample_correction=finite_sample_correction,
        min_eta2=min_eta2,
        min_gamma2=min_gamma2,
    ).estimate


__all__ = [
    "DoubleShrinkageFit",
    "independent_estimator_components",
    "fit_double_shrinkage_components",
    "fit_double_shrinkage",
    "get_double_shrinkage_estimators",
]
