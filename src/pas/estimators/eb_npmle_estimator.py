import numpy as np
from pas.datasets.dataset import PasDataset
from typing import Tuple, Union
from scipy.special import logsumexp
from pas.estimators.eb_estimators import (
    _get_global_ppi_lambda,
    _get_generic_ppi_estimators_plus_bias,
    _get_power_tuned_lambdas,
    _get_power_tuned_ppi_estimators_plus_bias,
)
def _npmle_prior_computation_for_bias(
    bias: np.ndarray,
    cov_mats: np.ndarray,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a nonparametric prior G for the latent bias b_i using NPMLE.

    Args:
        bias: Array of observed bias values, shape (m,).
        cov_mats:
            Array of covariance matrices, shape (m, 2, 2), where
            cov_mats[i, 1, 1] is Var(b_hat_i).
        u: If int, use an equally spaced grid from min(bias)-eps to max(bias)+eps
           with length u. If array-like, use it directly as the atom grid
        threshold: Drop NPMLE atoms whose estimated weights are <= threshold.
        eps: mall padding added to the grid endpoints when u is an integer.
        verbose: Whether to print progress messages.
        max_iter_em: Number of EM updates after solving weights on the initial grid.

    Returns:
        atoms: grid points
        weights: corresponding probability
    """
    from npeb import GLMixture
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_mats = np.asarray(cov_mats, dtype=float)

    if np.isscalar(u):
        n_grid = int(u)
        if n_grid < 2:
            raise ValueError("If u is an integer, it must be at least 2.")
        u_grid = np.linspace(bias.min() - eps, bias.max() + eps, n_grid)
    else:
        u_grid = np.asarray(u, dtype=float).reshape(-1)
        if u_grid.size < 2:
            raise ValueError("If u is array-like, it must contain at least 2 grid points.")

    X = bias.reshape(-1, 1)
    var_bias_hat = cov_mats[:, 1, 1] 
    prec = (1.0 / var_bias_hat).reshape(-1, 1)

    model = GLMixture(
        prec_type="diagonal",
        homoscedastic=False,
        atoms_init=u_grid.reshape(-1, 1),
    )

    if verbose:
        print("Solving NPMLE for bias prior using npeb...")

    model.fit(
        X,
        prec,
        max_iter_em=max_iter_em,
        weight_thresh=threshold,
        row_condition=True,
        solver="mosek",
    )

    atoms, weights = model.get_params()

    atoms = np.asarray(atoms, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    keep = weights > threshold
    atoms = atoms[keep]
    weights = weights[keep]

    if weights.sum() <= 0:
        raise RuntimeError("All NPMLE weights were thresholded out. Try lowering threshold.")

    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return atoms,weights

def _posterior_moments_bias_npmle(
    bias: np.ndarray,
    cov_mats: np.ndarray,
    atoms: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute posterior mean and variance of latent b_i under an NPMLE prior.

    Model:
        b_hat_i | b_i ~ N(b_i, Var(b_hat_i))
        b_i ~ sum_k weights[k] delta_{atoms[k]}

    Args:
        bias: observed bias values b_hat_i, shape (m,).
        cov_mats: covariance matrices, shape (m, 2, 2), where
            cov_mats[i, 1, 1] is Var(b_hat_i).
        atoms: NPMLE prior atoms, shape (K,).
        weights: NPMLE prior weights, shape (K,).

    Returns:
        post_mean: E[b_i | b_hat_i], shape (m,).
        post_var: Var(b_i | b_hat_i), shape (m,).
    """
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_mats = np.asarray(cov_mats, dtype=float)
    atoms = np.asarray(atoms, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    var_bias_hat = cov_mats[:, 1, 1]
    weights = weights / weights.sum()

    # log p(b_hat_i | b_i = atom_k)
    log_lik = (
        -0.5 * np.log(2 * np.pi * var_bias_hat[:, None])
        -0.5 * (bias[:, None] - atoms[None, :]) ** 2 / var_bias_hat[:, None]
    )

    log_prior = np.log(np.maximum(weights, 1e-300))[None, :]
    log_joint = log_lik + log_prior
    log_norm = logsumexp(log_joint, axis=1, keepdims=True)

    post_prob = np.exp(log_joint - log_norm)

    post_mean = post_prob @ atoms
    post_second = post_prob @ (atoms ** 2)
    post_var = np.maximum(post_second - post_mean ** 2, 0.0)

    return post_mean, post_var

def _get_generic_npmle_eb_ppi_estimators(
    ppi: np.ndarray,
    bias: np.ndarray,
    cov_mats: np.ndarray,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """Compute EB-adjusted PPI estimators using an NPMLE prior on bias terms.

    Args:
        ppi: PPI estimates, shape (m,).
        bias: observed bias values b_hat_i, shape (m,).
        cov_mats: covariance matrices, shape (m, 2, 2).
        u: grid size or grid points for NPMLE prior.
        threshold: drop NPMLE atoms with weights <= threshold.
        eps: small padding for grid endpoints if u is an integer.
        verbose: whether to print progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        npmle_eb_ppi: NPMLE EB-adjusted estimates, shape (m,).
    """
    ppi = np.asarray(ppi, dtype=float).reshape(-1)
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_mats = np.asarray(cov_mats, dtype=float)

    sigma_12 = cov_mats[:, 0, 1]
    sigma_22 = cov_mats[:, 1, 1]

    if np.any(~np.isfinite(ppi)) or np.any(~np.isfinite(bias)):
        raise ValueError("ppi and bias must be finite.")

    if np.any(~np.isfinite(sigma_12)) or np.any(~np.isfinite(sigma_22)):
        raise ValueError("cov_mats contains non-finite covariance entries.")

    if np.any(sigma_22 <= 0):
        raise ValueError("cov_mats[:, 1, 1] must be strictly positive.")

    atoms, weights = _npmle_prior_computation_for_bias(
        bias=bias,
        cov_mats=cov_mats,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )

    post_mean, _ = _posterior_moments_bias_npmle(
        bias=bias,
        cov_mats=cov_mats,
        atoms=atoms,
        weights=weights,
    )

    npmle_eb_ppi = ppi - (sigma_12 / sigma_22) * (bias - post_mean)

    return npmle_eb_ppi

def _apply_npmle_eb_with_corr_filter(
    ppi: np.ndarray,
    bias: np.ndarray,
    cov_matrix: np.ndarray,
    corr_threshold: float,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """Apply NPMLE EB adjustment only where correlation is below threshold.

    Args:
        ppi: PPI estimates, shape (M,).
        bias: observed bias values, shape (M,).
        cov_matrix: covariance matrices, shape (M, 2, 2).
        corr_threshold: only apply EB when corr(ppi, b_hat) < threshold.
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        out: estimates after applying NPMLE EB on selected problems.
    """
    ppi = np.asarray(ppi, dtype=float).reshape(-1)
    bias = np.asarray(bias, dtype=float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

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

    out = ppi.copy()

    if np.any(use_eb):
        out[use_eb] = _get_generic_npmle_eb_ppi_estimators(
            ppi=ppi[use_eb],
            bias=bias[use_eb],
            cov_mats=cov_matrix[use_eb],
            u=u,
            threshold=threshold,
            eps=eps,
            verbose=verbose,
            max_iter_em=max_iter_em,
        )

    return out

def get_npmle_eb_ppi_estimators(
    data: PasDataset,
    corr_threshold: float = 1.0,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB-adjusted PPI estimator (lambda=1) for each problem's mean.

    Args:
        data: Dataset with M problems.
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        NPMLE EB-adjusted estimates, shape (M,).
    """
    ppi, bias, cov_matrix = _get_generic_ppi_estimators_plus_bias(
        data.pred_unlabelled,
        data.pred_labelled,
        data.y_labelled,
        1.0,
    )

    return _apply_npmle_eb_with_corr_filter(
        ppi=ppi,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )

def get_npmle_eb_unipt_ppi_estimators(
    data: PasDataset,
    corr_threshold: float = 1.0,
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB-adjusted UniPT PPI estimator using global lambda.

    Args:
        data: Dataset with M problems.
        corr_threshold: only apply EB when correlation(ppi, b) < threshold.
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        NPMLE EB-adjusted UniPT PPI estimates, shape (M,).
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

    return _apply_npmle_eb_with_corr_filter(
        ppi=ppi,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )

def get_npmle_eb_power_tuned_estimators(
    data: PasDataset,
    corr_threshold: float = 1.0,
    clip_lambda: Tuple[float, float] = (0.0, 1.0),
    u: Union[int, np.ndarray] = 30,
    threshold: float = 1e-6,
    eps: float = 1e-5,
    verbose: bool = False,
    max_iter_em: int = 0,
) -> np.ndarray:
    """NPMLE EB-adjusted power-tuned PPI estimator.

    Args:
        data: Dataset with M problems.
        corr_threshold: only apply EB when corr(theta_PT, b_hat) < threshold.
        clip_lambda: bounds for each problem-wise lambda_i. Default is (0, 1).
        u: grid size or grid points for NPMLE prior.
        threshold: NPMLE atom weight threshold.
        eps: grid endpoint padding.
        verbose: whether to print NPMLE progress messages.
        max_iter_em: number of EM updates after solving weights on the initial grid.

    Returns:
        NPMLE EB-adjusted power-tuned estimates, shape (M,).
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

    return _apply_npmle_eb_with_corr_filter(
        ppi=pt,
        bias=bias,
        cov_matrix=cov_matrix,
        corr_threshold=corr_threshold,
        u=u,
        threshold=threshold,
        eps=eps,
        verbose=verbose,
        max_iter_em=max_iter_em,
    )