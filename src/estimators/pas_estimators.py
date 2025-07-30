"""
PAS and UniPAS estimators
"""
import numpy as np
from typing import Union, Tuple
from datasets.dataset import PasDataset
from utils import _minimize_lbfgs
from estimators.ppi_estimators import get_pt_ppi_estimators


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
    var_y, f_x_tilde_bar, y_bar = [], [], []
    for i in range(data.M):
        n = data.ns[i]
        # N = data.Ns[i]
        if share_var:
            # var_y.append(data.pred_labelled[i].var(ddof=1) / N)
            var_y.append(data.y_labelled[i].var(ddof=1) / n)
        f_x_tilde_bar.append(data.pred_unlabelled[i].mean())
        y_bar.append(data.y_labelled[i].mean())

    if not share_var:
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        var_y = var_y / data.ns

    if data.has_true_vars:
        var_y = data.true_vars

    f_x_tilde_bar, y_bar, var_y = np.array(
        f_x_tilde_bar), np.array(y_bar), np.array(var_y)

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

    if data.has_true_vars:
        var_f_x = data.true_fx_vars
        var_y = data.true_vars
        cov_y_f_x = data.true_covs
    else:
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        cov_y_f_x = np.cov(np.concatenate(data.y_labelled),
                           np.concatenate(data.pred_labelled), ddof=1)[0, 1]
        var_f_x = np.concatenate(data.pred_unlabelled).var(ddof=1)

    var_y = var_y / data.ns
    cov_y_f_x = cov_y_f_x / data.ns
    var_f_x_scaled = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    var_pt_ppi = var_y + sure_lambdas ** 2 * \
        var_f_x_scaled - 2 * sure_lambdas * cov_y_f_x

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

    References:
        [1] X. Xie, S. C. Kou, and L. D. Brown, “SURE Estimates for a Heteroscedastic Hierarchical Model”.
    """
    f_x_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])

    pt_ppi_estimates, sure_lambdas = get_pt_ppi_estimators(
        data, get_lambdas=True, share_var=share_var)

    if data.has_true_vars:
        var_f_x = data.true_fx_vars
        var_y = data.true_vars
        cov_y_f_x = data.true_covs
    else:
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        cov_y_f_x = np.cov(np.concatenate(data.y_labelled),
                           np.concatenate(data.pred_labelled), ddof=1)[0, 1]
        var_f_x = np.concatenate(data.pred_unlabelled).var(ddof=1)

    var_y = var_y / data.ns
    cov_y_f_x = cov_y_f_x / data.ns
    var_f_x_scaled = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    var_pt_ppi = var_y + sure_lambdas ** 2 * \
        var_f_x_scaled - 2 * sure_lambdas * cov_y_f_x

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
    sure_estimates = omegas * pt_ppi_estimates + (1 - omegas) * f_x_bar

    if not get_lambdas and not get_omegas:
        return sure_estimates
    elif get_lambdas and not get_omegas:
        return sure_estimates, sure_lambdas
    elif not get_lambdas and get_omegas:
        return sure_estimates, omegas
    else:
        return sure_estimates, sure_lambdas, omegas
