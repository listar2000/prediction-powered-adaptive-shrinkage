"""
Empirical Bayes (EB) adjusted PPI confidence intervals for the mean.
"""
import numpy as np
from datasets.dataset import PasDataset
from estimators.eb_estimators import (
    _get_generic_ppi_estimators_plus_bias,
    _get_global_ppi_lambda,
    _param_prior_computation_for_bias,
)
from utils import _zconfint


def _eb_ppi_ci_from_components(ppi, bias, cov_matrix, corr_threshold,
                               alpha, alternative):
    """Shared logic for EB-adjusted PPI CIs (used by both lambda=1 and UniPT)."""
    var_prod = np.maximum(cov_matrix[:, 0, 0] * cov_matrix[:, 1, 1], 0.0)
    denom_corr = np.sqrt(var_prod)
    corr = np.divide(
        cov_matrix[:, 0, 1],
        denom_corr,
        out=np.zeros_like(cov_matrix[:, 0, 1], dtype=float),
        where=denom_corr > 0,
    )
    use_eb = corr < corr_threshold
    point_estimates = ppi.copy()
    variances = cov_matrix[:, 0, 0].copy()

    if np.any(use_eb):
        mu_0, var_0 = _param_prior_computation_for_bias(
            bias[use_eb], cov_matrix[use_eb],
        )
        denom_eb = cov_matrix[use_eb, 1, 1] + var_0
        point_estimates[use_eb] = (
            ppi[use_eb]
            - cov_matrix[use_eb, 0, 1] / denom_eb * (bias[use_eb] - mu_0)
        )
        variances[use_eb] = (
            cov_matrix[use_eb, 0, 0]
            - (cov_matrix[use_eb, 0, 1] ** 2) / denom_eb
        )

    variances = np.maximum(variances, 0.0)
    return _zconfint(point_estimates, np.sqrt(variances), alpha, alternative)


def get_eb_ppi_cis(data: PasDataset, alpha: float = 0.1,
                   alternative: str = "two-sided",
                   corr_threshold: float = 1.0) -> np.ndarray:
    """EB-adjusted PPI confidence interval (lambda=1) for each problem's mean.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, 1.0)
    return _eb_ppi_ci_from_components(
        ppi, bias, cov_matrix, corr_threshold, alpha, alternative)


def get_eb_unipt_ppi_cis(data: PasDataset, alpha: float = 0.1,
                         alternative: str = "two-sided",
                         corr_threshold: float = 1.0) -> np.ndarray:
    """EB-adjusted UniPT PPI confidence interval (global lambda) for each
    problem's mean.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    lambda_ = _get_global_ppi_lambda(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled)
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambda_)
    return _eb_ppi_ci_from_components(
        ppi, bias, cov_matrix, corr_threshold, alpha, alternative)
