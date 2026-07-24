"""Robust empirical-Bayes confidence intervals for double shrinkage.

Implements the interval construction from Definition 1 of Rosenman,
Dominici, and Miratrix (arXiv:2309.06727),

    psi_k +/- cva_alpha(c_k) * a_k * sqrt(lambda_k^2 s_uk^2
                                          + (1-lambda_k)^2 s_bk^2),

where ``cva_alpha`` is the Armstrong--Kolesar--Plagborg-Moller robust
empirical-Bayes critical value under a second-moment bound on the
normalized conditional bias (see :mod:`pas.intervals.robust_eb`).
"""
from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
from scipy.stats import norm

from pas.estimators.double_shrinkage import (
    DoubleShrinkageMethod,
    fit_double_shrinkage,
)
from pas.intervals.robust_eb import robust_eb_critical_values


def get_double_shrinkage_cis(
    data,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    *,
    method: DoubleShrinkageMethod = "mle",
    center: float = 0.0,
    finite_sample_correction: Literal["pmt", "none"] = "pmt",
    min_eta2: float = 0.0,
    min_gamma2: float = 0.0,
    cv_mode: Literal["lookup", "exact"] = "lookup",
) -> np.ndarray:
    """Robust empirical-Bayes intervals for double shrinkage.

    The baseline paper derives symmetric two-sided robust EBCIs.  One-sided
    alternatives are therefore rejected rather than silently substituted by
    an unsupported construction.
    """
    if alternative != "two-sided":
        raise ValueError("double-shrinkage robust EBCIs support only 'two-sided'")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    fit = fit_double_shrinkage(
        data,
        method=method,
        center=center,
        finite_sample_correction=finite_sample_correction,
        min_eta2=min_eta2,
        min_gamma2=min_gamma2,
    )
    n = fit.estimate.size
    out = np.empty((n, 2), dtype=float)

    # A task excluded from cross-task fitting falls back to its valid unbiased
    # Wald interval whenever possible.
    fallback_se = np.sqrt(np.maximum(fit.var_unbiased, 0.0))
    fallback_z = norm.ppf(1.0 - alpha / 2.0)
    out[:, 0] = fit.unbiased - fallback_z * fallback_se
    out[:, 1] = fit.unbiased + fallback_z * fallback_se

    idx = np.flatnonzero(fit.valid)
    positive = fit.sampling_variance[idx] > 0.0
    if np.any(positive):
        active = idx[positive]
        critical = robust_eb_critical_values(
            fit.normalized_bias_m2[active], alpha=alpha, mode=cv_mode
        )
        half = critical * np.sqrt(fit.sampling_variance[active])
        out[active, 0] = fit.estimate[active] - half
        out[active, 1] = fit.estimate[active] + half

    # With correction disabled, eta2 can be exactly zero and the manuscript's
    # estimator/interval collapses to the center.  Preserve that behavior.
    collapsed = idx[~positive]
    if collapsed.size:
        warnings.warn(
            "double-shrinkage interval collapsed because estimated eta2 is zero; "
            "use finite_sample_correction='pmt' to apply the manuscript's "
            "finite-sample truncation idea",
            RuntimeWarning,
            stacklevel=2,
        )
        out[collapsed, 0] = fit.estimate[collapsed]
        out[collapsed, 1] = fit.estimate[collapsed]
    return out


# Registry-friendly wrappers -------------------------------------------------

def get_double_shrinkage_mm1_cis(data, alpha=0.1, alternative="two-sided", **kwargs):
    return get_double_shrinkage_cis(
        data, alpha, alternative, method="mm1", **kwargs
    )


def get_double_shrinkage_mm2_cis(data, alpha=0.1, alternative="two-sided", **kwargs):
    return get_double_shrinkage_cis(
        data, alpha, alternative, method="mm2", **kwargs
    )


def get_double_shrinkage_mle_cis(data, alpha=0.1, alternative="two-sided", **kwargs):
    return get_double_shrinkage_cis(
        data, alpha, alternative, method="mle", **kwargs
    )


def get_double_shrinkage_ure_cis(data, alpha=0.1, alternative="two-sided", **kwargs):
    return get_double_shrinkage_cis(
        data, alpha, alternative, method="ure", **kwargs
    )


__all__ = [
    "get_double_shrinkage_cis",
    "get_double_shrinkage_mm1_cis",
    "get_double_shrinkage_mm2_cis",
    "get_double_shrinkage_mle_cis",
    "get_double_shrinkage_ure_cis",
]
