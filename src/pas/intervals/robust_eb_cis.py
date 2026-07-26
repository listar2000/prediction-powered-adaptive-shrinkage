"""Robust empirical-Bayes confidence intervals for unbiased PAS estimators.

The intervals implement the default ``ebci::ebci`` construction from
Armstrong, Kolesar, and Plagborg-Moller (2022).  Given the fitted rule

    theta_hat_i = mu_i + w_i (Y_i - mu_i),
    w_i = mu2 / (mu2 + se_i^2),

its robust interval is

    theta_hat_i +/- cva_alpha(se_i^2 / mu2, kappa) * w_i * se_i.

The default PAS configuration uses PT as the unbiased preliminary estimator,
shrinks toward its inverse-variance-weighted grand mean, fits the moments with
weights ``1 / se_i^2``, and uses ``fs_correction='PMT'``.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union
import warnings

import numpy as np
from scipy.stats import norm

from pas.estimators.robust_eb import (
    BaseEstimator,
    FiniteSampleCorrection,
    fit_robust_eb_dataset,
)
from pas.intervals.robust_eb import robust_eb_critical_values


def get_robust_eb_cis(
    data,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    *,
    base_estimator: BaseEstimator = "pt",
    share_var: bool = True,
    weights: Union[
        Literal["inverse_variance", "equal"],
        Sequence[float],
        np.ndarray,
    ] = "inverse_variance",
    design: Optional[np.ndarray] = None,
    shrink_to: Literal["grand_mean", "zero"] = "grand_mean",
    fs_correction: FiniteSampleCorrection = "PMT",
    kappa: Optional[float] = None,
    cv_mode: Literal["lookup", "exact"] = "exact",
) -> np.ndarray:
    """Construct robust EBCIs by shrinking an unbiased estimator only.

    ``alternative`` is restricted to ``'two-sided'`` because the source method
    derives symmetric two-sided average-coverage intervals.
    """
    if alternative != "two-sided":
        raise ValueError("robust EBCIs support only alternative='two-sided'")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    fit = fit_robust_eb_dataset(
        data,
        base_estimator=base_estimator,
        share_var=share_var,
        weights=weights,
        design=design,
        shrink_to=shrink_to,
        fs_correction=fs_correction,
        kappa=kappa,
    )

    if fit.mu2 <= 0.0:
        warnings.warn(
            "The robust-EB second-moment estimate is zero; returning the "
            "unshrunk Wald intervals. Use fs_correction='PMT' to avoid this "
            "finite-sample collapse.",
            RuntimeWarning,
            stacklevel=2,
        )
        half_length = norm.ppf(1.0 - alpha / 2.0) * fit.standard_error
        return np.column_stack(
            [fit.unshrunk - half_length, fit.unshrunk + half_length]
        )

    normalized_bias_m2 = fit.standard_error**2 / fit.mu2
    critical_values = robust_eb_critical_values(
        normalized_bias_m2,
        alpha=alpha,
        kappa=fit.kappa,
        mode=cv_mode,
    )
    half_length = critical_values * fit.shrinkage * fit.standard_error
    return np.column_stack(
        [fit.estimate - half_length, fit.estimate + half_length]
    )


__all__ = ["get_robust_eb_cis"]
