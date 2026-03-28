"""
Empirical Bayes (EB) adjusted PPI estimators for compound mean estimation.
"""
import numpy as np
from pas.datasets.dataset import PasDataset
from typing import Tuple
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Helpers (also used by intervals/eb_cis.py)
# ---------------------------------------------------------------------------

def _get_generic_ppi_estimators_plus_bias(
    f_x_tilde: np.ndarray,
    f_x: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PPI estimator, bias term b_i, and their covariance matrix per problem.

    Estimator: ppi_i = ybar_i + lambda * (ztilde_i - zbar_i)
    Bias term: b_i   = ybar_i - lambda * zbar_i

    Args:
        f_x_tilde: list of prediction arrays for unlabelled data.
        f_x: list of prediction arrays for labelled data.
        y: list of response arrays for labelled data.
        lambda_: global scalar lambda.

    Returns:
        ppi: PPI estimates, shape (M,).
        bias: b_i values, shape (M,).
        cov_mats: covariance matrices, shape (M, 2, 2).
    """
    ppi = []
    bias = []
    cov_mats = []

    for i in range(len(f_x_tilde)):
        n_i = len(y[i])
        N_i = len(f_x_tilde[i])

        ppi_i = y[i].mean() + lambda_ * (f_x_tilde[i].mean() - f_x[i].mean())
        b_i = y[i].mean() - lambda_ * f_x[i].mean()

        if n_i > 1:
            var_y_i = np.var(y[i], ddof=1)
            var_z_i = np.var(f_x[i], ddof=1)
            cov_yz_i = np.cov(y[i], f_x[i], ddof=1)[0, 1]
            var_b_i = (var_y_i + lambda_**2 * var_z_i - 2 * lambda_ * cov_yz_i) / n_i
        else:
            var_b_i = np.nan

        if N_i > 1:
            var_ztilde_mean_i = np.var(f_x_tilde[i], ddof=1) / N_i
        else:
            var_ztilde_mean_i = np.nan

        if np.isnan(var_b_i) or np.isnan(var_ztilde_mean_i):
            var_ppi_i = np.nan
            cov_ppi_b_i = np.nan
        else:
            var_ppi_i = var_b_i + lambda_**2 * var_ztilde_mean_i
            cov_ppi_b_i = var_b_i

        Sigma_i = np.array([
            [var_ppi_i, cov_ppi_b_i],
            [cov_ppi_b_i, var_b_i]
        ])

        ppi.append(ppi_i)
        bias.append(b_i)
        cov_mats.append(Sigma_i)

    return np.array(ppi), np.array(bias), np.array(cov_mats)


def _get_global_ppi_lambda(
    f_x_tilde: np.ndarray,
    f_x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Estimate a single global lambda by minimizing the sum of estimated
    variances across all problems.

    Returns:
        lambda_hat: estimated global lambda.
    """
    numerator = 0.0
    denominator = 0.0

    for i in range(len(y)):
        n_i = len(y[i])
        N_i = len(f_x_tilde[i])

        cov_yz_i = np.cov(y[i], f_x[i], ddof=1)[0, 1]
        z_all_i = np.concatenate([f_x[i], f_x_tilde[i]])
        var_z_i = np.var(z_all_i, ddof=1)

        numerator += cov_yz_i / n_i
        denominator += (1.0 / n_i + 1.0 / N_i) * var_z_i

    if denominator <= 0:
        raise ValueError("Estimated denominator is non-positive, so lambda is undefined.")

    return float(numerator / denominator)


def _param_prior_computation_for_bias(bias, cov_mats):
    """Estimate parametric normal prior for latent b_i from observed b_hat_i
    via maximum likelihood.

    Returns:
        hat_mu_b: estimated prior mean.
        hat_var_b: estimated prior variance.
    """
    meas_var = cov_mats[:, 1, 1]

    init_mean = bias.mean()
    s2_obs = np.var(bias, ddof=1) if len(bias) > 1 else 0.0
    init_var = max(s2_obs - meas_var.mean(), 1e-4)

    def nll(par):
        mu_b, log_tau2 = par
        tau2 = np.exp(log_tau2)
        total_var = tau2 + meas_var
        resid2 = (bias - mu_b) ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * total_var) + resid2 / total_var)

    opt = minimize(
        nll,
        x0=np.array([init_mean, np.log(init_var)]),
        method="BFGS",
    )
    hat_mu_b = opt.x[0]
    hat_var_b = np.exp(opt.x[1])

    return hat_mu_b, hat_var_b


def _get_generic_eb_ppi_estimators(ppi, bias, cov_mats):
    """Compute EB-adjusted PPI estimators using the estimated prior on bias terms.

    Returns:
        eb_ppi: EB-adjusted estimates, shape (m,).
    """
    mu_0, var_0 = _param_prior_computation_for_bias(bias, cov_mats)
    denom = cov_mats[:, 1, 1] + var_0
    k = cov_mats[:, 0, 1] / denom
    return ppi - k * (bias - mu_0)


def _apply_eb_with_corr_filter(ppi, bias, cov_matrix, corr_threshold):
    """Apply EB adjustment only to problems where correlation is below threshold."""
    var_prod = np.maximum(cov_matrix[:, 0, 0] * cov_matrix[:, 1, 1], 0.0)
    denom_corr = np.sqrt(var_prod)
    corr = np.divide(
        cov_matrix[:, 0, 1],
        denom_corr,
        out=np.zeros_like(cov_matrix[:, 0, 1], dtype=float),
        where=denom_corr > 0,
    )
    use_eb = corr < corr_threshold
    out = ppi.copy()

    if np.any(use_eb):
        out[use_eb] = _get_generic_eb_ppi_estimators(
            ppi[use_eb], bias[use_eb], cov_matrix[use_eb],
        )
    return out


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------

def get_eb_ppi_estimators(data: PasDataset, corr_threshold: float = 1.0) -> np.ndarray:
    """EB-adjusted PPI estimator (lambda=1) for each problem's mean.

    Args:
        data: Dataset with M problems.
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.

    Returns:
        EB-adjusted estimates, shape (M,).
    """
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, 1.0)
    return _apply_eb_with_corr_filter(ppi, bias, cov_matrix, corr_threshold)


def get_eb_unipt_ppi_estimators(data: PasDataset, corr_threshold: float = 1.0) -> np.ndarray:
    """EB-adjusted UniPT PPI estimator (global lambda) for each problem's mean.

    Args:
        data: Dataset with M problems.
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.

    Returns:
        EB-adjusted estimates, shape (M,).
    """
    lambda_ = _get_global_ppi_lambda(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled)
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambda_)
    return _apply_eb_with_corr_filter(ppi, bias, cov_matrix, corr_threshold)
