"""
NPMLE empirical Bayes adjusted PPI equal-tail confidence intervals.
"""
import numpy as np
from typing import Union, Tuple
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.optimize import brentq

from pas.datasets.dataset import PasDataset
from pas.estimators.eb_estimators import (
    _get_generic_ppi_estimators_plus_bias,
    _get_global_ppi_lambda,
    _get_power_tuned_lambdas,
    _get_power_tuned_ppi_estimators_plus_bias,
)
from pas.estimators.eb_npmle_estimator import _npmle_prior_computation_for_bias
from pas.utils import _zconfint


def _posterior_weights_bias_npmle(
    bias: np.ndarray,
    cov_mats: np.ndarray,
    atoms: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute posterior weights P(b_i = atom_k | b_hat_i).

    Model:
        b_hat_i | b_i ~ N(b_i, Var(b_hat_i))
        b_i ~ sum_k weights[k] delta_{atoms[k]}

    Returns:
        post_prob: array of shape (m, K), where post_prob[i, k] is the
        posterior weight of atom k for problem i.
    """
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_mats = np.asarray(cov_mats, dtype=float)
    atoms = np.asarray(atoms, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    var_bias_hat = cov_mats[:, 1, 1]

    if np.any(~np.isfinite(var_bias_hat)) or np.any(var_bias_hat <= 0):
        raise ValueError("cov_mats[:, 1, 1] must be finite and strictly positive.")

    weights = weights / weights.sum()

    log_lik = (
        -0.5 * np.log(2 * np.pi * var_bias_hat[:, None])
        -0.5 * (bias[:, None] - atoms[None, :]) ** 2 / var_bias_hat[:, None]
    )

    log_prior = np.log(np.maximum(weights, 1e-300))[None, :]
    log_joint = log_lik + log_prior
    log_norm = logsumexp(log_joint, axis=1, keepdims=True)

    return np.exp(log_joint - log_norm)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile for a discrete distribution."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / weights.sum()

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cdf = np.cumsum(weights_sorted)

    return float(values_sorted[np.searchsorted(cdf, q, side="left")])


def _mixture_normal_quantile(
    q: float,
    means: np.ndarray,
    weights: np.ndarray,
    sd: float,
) -> float:
    """Quantile of sum_k weights[k] N(means[k], sd^2)."""
    means = np.asarray(means, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / weights.sum()

    if not (0.0 < q < 1.0):
        raise ValueError("q must be strictly between 0 and 1.")

    if sd <= 1e-12:
        return _weighted_quantile(means, weights, q)

    def cdf(x):
        return np.sum(weights * norm.cdf((x - means) / sd))

    def root_fun(x):
        return cdf(x) - q

    lower = np.min(means) - 10.0 * sd
    upper = np.max(means) + 10.0 * sd

    # Expand bracket if needed.
    while root_fun(lower) > 0:
        lower -= 10.0 * sd
    while root_fun(upper) < 0:
        upper += 10.0 * sd

    return float(brentq(root_fun, lower, upper))


def _npmle_equal_tail_ci_from_components(
    ppi: np.ndarray,
    bias: np.ndarray,
    cov_matrix: np.ndarray,
    corr_threshold: float,
    alpha: float,
    alternative: str,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """Shared logic for NPMLE EB equal-tail CIs.

    For selected problems, this constructs equal-tail intervals from the
    mixture distribution

        theta_i | ppi_i, b_hat_i
        ~ sum_k C_ik N(
              ppi_i - Sigma12_i / Sigma22_i * (b_hat_i - atom_k),
              Sigma11_i - Sigma12_i^2 / Sigma22_i
          ).

    For problems not selected by the correlation filter, it returns the
    original normal PPI CI based on ppi_i and Sigma11_i.
    """
    ppi = np.asarray(ppi, dtype=float).reshape(-1)
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'.")

    # Start with the unadjusted normal CI.
    base_var = np.maximum(cov_matrix[:, 0, 0], 0.0)
    ci = _zconfint(ppi, np.sqrt(base_var), alpha, alternative)

    var_prod = np.maximum(cov_matrix[:, 0, 0] * cov_matrix[:, 1, 1], 0.0)
    denom_corr = np.sqrt(var_prod)
    corr = np.divide(
        cov_matrix[:, 0, 1],
        denom_corr,
        out=np.zeros_like(cov_matrix[:, 0, 1], dtype=float),
        where=denom_corr > 0,
    )

    finite_cov = np.all(np.isfinite(cov_matrix), axis=(1, 2))
    valid = (
        np.isfinite(ppi)
        & np.isfinite(bias)
        & finite_cov
        & (cov_matrix[:, 0, 0] > 0)
        & (cov_matrix[:, 1, 1] > 0)
    )

    use_eb = valid & (corr < corr_threshold)

    if np.sum(use_eb) < 2:
        if verbose:
            print("Skipping NPMLE EB CI because fewer than 2 valid problems are selected.")
        return ci

    ppi_use = ppi[use_eb]
    bias_use = bias[use_eb]
    cov_use = cov_matrix[use_eb]

    atoms, weights = _npmle_prior_computation_for_bias(
        bias=bias_use,
        cov_mats=cov_use,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )

    post_prob = _posterior_weights_bias_npmle(
        bias=bias_use,
        cov_mats=cov_use,
        atoms=atoms,
        weights=weights,
    )

    sigma_11 = cov_use[:, 0, 0]
    sigma_12 = cov_use[:, 0, 1]
    sigma_22 = cov_use[:, 1, 1]

    cond_var = sigma_11 - (sigma_12 ** 2) / sigma_22
    cond_var = np.maximum(cond_var, 0.0)

    selected_indices = np.where(use_eb)[0]

    for local_idx, global_idx in enumerate(selected_indices):
        component_means = (
            ppi_use[local_idx]
            - (sigma_12[local_idx] / sigma_22[local_idx])
            * (bias_use[local_idx] - atoms)
        )
        component_weights = post_prob[local_idx]
        component_sd = np.sqrt(cond_var[local_idx])

        if alternative == "two-sided":
            lower_q = alpha / 2.0
            upper_q = 1.0 - alpha / 2.0
            ci[global_idx, 0] = _mixture_normal_quantile(
                lower_q, component_means, component_weights, component_sd
            )
            ci[global_idx, 1] = _mixture_normal_quantile(
                upper_q, component_means, component_weights, component_sd
            )

        elif alternative == "larger":
            ci[global_idx, 0] = _mixture_normal_quantile(
                alpha, component_means, component_weights, component_sd
            )
            ci[global_idx, 1] = np.inf

        elif alternative == "smaller":
            ci[global_idx, 0] = -np.inf
            ci[global_idx, 1] = _mixture_normal_quantile(
                1.0 - alpha, component_means, component_weights, component_sd
            )

    return ci


def get_npmle_eb_ppi_cis(
    data: PasDataset,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    corr_threshold: float = 1.0,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB equal-tail PPI confidence intervals with lambda=1.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1.
        alternative: "two-sided", "larger", or "smaller".
        corr_threshold: only apply NPMLE EB when correlation(ppi, b) < threshold.
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
        1.0,
    )

    return _npmle_equal_tail_ci_from_components(
        ppi=ppi,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        alpha=alpha,
        alternative=alternative,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )


def get_npmle_eb_unipt_ppi_cis(
    data: PasDataset,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    corr_threshold: float = 1.0,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB equal-tail UniPT PPI confidence intervals using global lambda.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1.
        alternative: "two-sided", "larger", or "smaller".
        corr_threshold: only apply NPMLE EB when correlation(ppi, b) < threshold.
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    lambda_ = _get_global_ppi_lambda(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
    )

    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
        lambda_,
    )

    return _npmle_equal_tail_ci_from_components(
        ppi=ppi,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        alpha=alpha,
        alternative=alternative,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )

def get_npmle_eb_power_tuned_cis(
    data: PasDataset,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    corr_threshold: float = 1.0,
    clip_lambda: Tuple[float, float] = (0.0, 1.0),
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB equal-tail confidence intervals for power-tuned PPI.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1.
        alternative: "two-sided", "larger", or "smaller".
        corr_threshold: only apply NPMLE EB when corr(theta_PT, b_hat) < threshold.
        clip_lambda: bounds for each problem-wise lambda_i. Default is (0, 1).
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    lambdas = _get_power_tuned_lambdas(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
        clip=clip_lambda,
    )

    _, bias, pt, cov_matrix = _get_power_tuned_ppi_estimators_plus_bias(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
        lambdas,
    )

    return _npmle_equal_tail_ci_from_components(
        ppi=pt,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        alpha=alpha,
        alternative=alternative,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )