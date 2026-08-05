"""
PAS and UniPAS estimators
"""
import numpy as np
from typing import Union, Tuple
from pas.datasets.dataset import PasDataset
from pas.utils import _minimize_lbfgs
from pas.estimators.ppi_estimators import (
    MIN_PT_VARIANCE,
    estimate_second_moments,
    get_pt_ppi_estimators,
)


def get_shrinkage_only_estimators(data: PasDataset, get_lambdas: bool = False, share_var: bool = True, cutoff: float = 0.999) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Obtain the SURE-minimizing shrinkage estimator for the problem. No PPI is done, only the second stage.

    Let A_i denotes Var(Y_i) for the i^th product, θ_i be the MLE and μ_i be the prediction mean (to shrink towards).
    SURE(λ) = Σ_i (A_i / (A_i + λ)^2) * (A_i * (μ_i - θ_i)^2 + λ^2 - A_i^2)

    To minimize the above, we will resort to numerical optimization (via `scipy.optim`). The final estimator is given by:

    θ_i^SURE = λ_i * θ_i + (1 - λ_i) * μ_i, where λ_i = λ^* / (A_i + λ^*)

    Args:
        data (PasDataset): the dataset object   

        get_lambdas (bool): whether to return the shrinkage factor λ_i. Default to `False`.

        share_var (bool): whether to consider a single global variance (across products) for all problems. If `True`, a \
            global variance will be first computed based on the concatenation of all the observed Y_i, \
            and then divided by the n_i to get A_i for each problem. Default to `False`.

        cutoff (float): the cutoff value for the maximum value of λ_i. This is an *ad-hoc* way of restraining the search space \
            for the optimal λ since λ_i = λ^* / (A_i + λ^*), i.e. we can calculate the upper bound of λ^* based on the \
            cutoff value and the maximum A_i. Default to `0.99`.

    Returns:
        sure_estimates: the SURE-minimizing shrinkage estimator for each product. If `get_lambdas` is `True`, the shrinkage factors will \
        also be returned.

    References:
        [1] X. Xie, S. C. Kou, and L. D. Brown, “SURE Estimates for a Heteroscedastic Hierarchical Model”.
    """
    # prepare observations for minimizing SURE
    f_x_tilde_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])
    y_bar = np.array([data.y_labelled[i].mean() for i in range(data.M)])

    # A_j = Var(Ybar_j) = sigma_hat_j^2 / n_j. `share_var=True` pools the
    # sigma_hat_j^2 across problems, `False` keeps them per-problem -- the same
    # convention as `get_pt_ppi_estimators`. This branch used to be inverted.
    var_y, _, _ = estimate_second_moments(data, share_var)
    var_y = np.maximum(var_y / data.ns, MIN_PT_VARIANCE)

    def sure_fn(lambda_: float) -> float:
        return np.sum((var_y / (var_y + lambda_) ** 2)
                      * (var_y * (f_x_tilde_bar - y_bar) ** 2 + lambda_ ** 2 - var_y ** 2))

    # calculate upper search bound for lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    lbd_upper = cutoff / (1 - cutoff) * var_y.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    lambdas = optimal_lbd / (var_y + optimal_lbd)
    sure_estimates = lambdas * y_bar + (1 - lambdas) * f_x_tilde_bar

    return sure_estimates if not get_lambdas else (sure_estimates, lambdas)


def get_pas_estimators(data: PasDataset, get_lambdas: bool = False, get_omegas: bool = False, share_var: bool = True,
                       cutoff: float = 0.999):
    """
    The very core `PAS` estimator mentioned in the paper. It adopts a two-stage estimation procedure:

    1. We use the power tuned PPI (`get_pt_ppi_estimators`) to lower the variance and obtain a power-tuned estimator.
    2. We then use the SURE-minimizing shrinkage estimator (`get_shrinkage_only_estimators`) to shrink the power-tuned estimator towards prediction mean.

    Args:
        data (PasDataset): the dataset object.
        get_lambdas (bool): whether to return the shrinkage factor λ_i. Default to `False`.
        get_omegas (bool): whether to return the power-tuning parameter ω_i. Default to `False`.
        share_var (bool): whether to consider a single global variance (across products) for all problems. If `True`, a \
            global variance will be first computed based on the concatenation of all the observed Y_i, \
            and then divided by the n_i to get A_i for each problem. Default to `False`.

        cutoff (float): the cutoff value for the maximum value of λ_i. This is an *ad-hoc* way of restraining the search space \
            for the optimal λ since λ_i = λ^* / (A_i + λ^*), i.e. we can calculate the upper bound of λ^* based on the \
            cutoff value and the maximum A_i. Default to `0.99`.
    """

    f_x_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])

    pt_ppi_estimates, sure_lambdas = get_pt_ppi_estimators(
        data, get_lambdas=True, share_var=share_var)

    # `share_var` now reaches the second moments as well, not just the PT
    # lambdas: previously these were always pooled, so `share_var=False` mixed
    # per-problem lambdas with pooled moments.
    var_y, var_f_x, cov_y_f_x = estimate_second_moments(data, share_var)

    var_y = var_y / data.ns
    cov_y_f_x = cov_y_f_x / data.ns
    var_f_x_scaled = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    var_pt_ppi = var_y + sure_lambdas ** 2 * \
        var_f_x_scaled - 2 * sure_lambdas * cov_y_f_x
    # See MIN_PT_VARIANCE: this plug-in variance can be negative in small
    # samples, which would drive omega_j above 1 and anti-shrink.
    var_pt_ppi = np.maximum(var_pt_ppi, MIN_PT_VARIANCE)

    def sure_fn(lambda_: float) -> float:
        first_term = np.sum((var_pt_ppi / (var_pt_ppi + lambda_) ** 2)
                            * (var_pt_ppi * (f_x_bar - pt_ppi_estimates) ** 2 + lambda_ ** 2 - var_pt_ppi ** 2))

        # note here we use the original (not scaled) version of `var_f_x`
        extra_term = np.sum(
            2 * (var_pt_ppi / (var_pt_ppi + lambda_)) * sure_lambdas * var_f_x / data.Ns)
        return first_term + extra_term

    # calculate upper search bound for lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    lbd_upper = cutoff / (1 - cutoff) * var_pt_ppi.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    omegas = optimal_lbd / (var_pt_ppi + optimal_lbd)
    sure_estimates = omegas * pt_ppi_estimates + (1 - omegas) * f_x_bar

    if not get_lambdas and not get_omegas:
        return sure_estimates
    elif get_lambdas and not get_omegas:
        return sure_estimates, sure_lambdas
    elif not get_lambdas and get_omegas:
        return sure_estimates, omegas
    else:
        return sure_estimates, sure_lambdas, omegas


def get_shrinkage_to_mean_estimators(data: PasDataset, get_lambdas: bool = False, get_omegas: bool = False, share_var: bool = True,
                                     cutoff: float = 0.999):
    """
    PT-PPI estimator but shrink towards the average (grand mean) of the estimators themselves (across all m problems).

    Implements Eq. (29) / Algorithm 4 ("shrink-average"): the shrinkage target
    is θ̄^PT = m⁻¹ Σ_j θ̂_j^PT, the grand mean of the power-tuned estimates.

    References:
        [1] X. Xie, S. C. Kou, and L. D. Brown, “SURE Estimates for a Heteroscedastic Hierarchical Model”.
    """
    pt_ppi_estimates, sure_lambdas = get_pt_ppi_estimators(
        data, get_lambdas=True, share_var=share_var)

    # `share_var` now reaches the second moments as well, not just the PT
    # lambdas (previously these were always pooled).
    var_y, var_f_x, cov_y_f_x = estimate_second_moments(data, share_var)

    var_y = var_y / data.ns
    cov_y_f_x = cov_y_f_x / data.ns
    var_f_x_scaled = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    var_pt_ppi = var_y + sure_lambdas ** 2 * \
        var_f_x_scaled - 2 * sure_lambdas * cov_y_f_x
    # See MIN_PT_VARIANCE: this plug-in variance can be negative in small
    # samples, which would drive omega_j above 1 and anti-shrink.
    var_pt_ppi = np.maximum(var_pt_ppi, MIN_PT_VARIANCE)

    grand_mean = np.mean(pt_ppi_estimates)

    def sure_fn(omega_: float) -> float:
        # follow section 4 of Xie et al. (2012)
        omega_j = omega_ / (var_pt_ppi + omega_)
        return np.sum(((1 - omega_j) ** 2) * (grand_mean - pt_ppi_estimates) ** 2
                      + (1 - omega_j) * (omega_ + (2 / data.M - 1) * var_pt_ppi))

    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    lbd_upper = cutoff / (1 - cutoff) * var_pt_ppi.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    omegas = optimal_lbd / (var_pt_ppi + optimal_lbd)
    # Shrink toward `grand_mean`, the quantity the SURE objective above is built
    # around. This previously shrank toward the per-problem prediction mean
    # `f_x_bar`, which is the target of `get_shrinkage_only_estimators`.
    sure_estimates = omegas * pt_ppi_estimates + (1 - omegas) * grand_mean

    if not get_lambdas and not get_omegas:
        return sure_estimates
    elif get_lambdas and not get_omegas:
        return sure_estimates, sure_lambdas
    elif not get_lambdas and get_omegas:
        return sure_estimates, omegas
    else:
        return sure_estimates, sure_lambdas, omegas
