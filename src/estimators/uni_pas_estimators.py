"""
UniPAS estimators
"""
import numpy as np
from datasets.dataset import PasDataset
from estimators.ppi_estimators import _get_generic_ppi_estimators
from utils import _minimize_lbfgs


def get_uni_pt_estimators(data: PasDataset, get_lambda: bool = False):
    """ Obtain the univariate power-tuning PPI estimator for the PPI problem.

    In the univariate power-tuning PPI setting, our goal is to find a single 'lambda' (i.e. power-tuning parameter) that works well for all problems.
    That is, we add up the covariance and variance terms together to calculate the optimal power-tuning parameter. Note that in this
    setting it does not make too much sense to "share variance" like in other estimators.

    TO PAPER READERS: this estimator turns out to be not very interesting by itself, but it can be combined to build UniPAS. 

    Args:
        data (PasDataset): the dataset object.

        get_lambda (bool): whether to return the power-tuning parameter λ_i. Default to `False`.

    Returns:
        ppi_estimates: the univariate power-tuning PPI estimator for each product. If `get_lambda` is `True`, the power-tuning parameters will \
        also be returned.
    """
    # aggregate the sum of covariance and variance terms
    numerator, denominator = 0, 0
    for i in range(data.M):
        n, N = data.ns[i], data.Ns[i]
        var_bar = np.concatenate(
            [data.pred_labelled[i], data.pred_unlabelled[i]]).var(ddof=1)
        cov_bar = np.cov(
            data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1]
        # compute the lambda for each problem
        numerator += cov_bar / n
        denominator += var_bar * ((N + n) / (N * n))

    lambda_ = numerator / denominator
    ppi_estimates = _get_generic_ppi_estimators(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambda_)
    return ppi_estimates if not get_lambda else (ppi_estimates, lambda_)


def get_uni_pas_estimators(data: PasDataset, get_lambda: bool = False, get_omega: bool = False, cutoff: float = 0.999):
    """
    Obtain the univariate PAS estimator from the paper.
    """
    # step 1: obtain the compound PPI estimates and lambda
    compound_estimates, cp_lambda = get_uni_pt_estimators(
        data, get_lambda=True)

    f_x_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])

    # step 2: recompute the sample-based unbiased estimate of second moments
    var_fx_hats = np.zeros(data.M)
    var_y_hats = np.zeros(data.M)
    cov_hats = np.zeros(data.M)

    for i in range(data.M):
        # n, N = data.ns[i], data.Ns[i]
        var_fx_hat = np.concatenate(
            [data.pred_labelled[i], data.pred_unlabelled[i]]).var(ddof=1)
        var_y_hat = data.y_labelled[i].var(ddof=1)
        cov_hat = np.cov(
            data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1]

        var_fx_hats[i] = var_fx_hat
        var_y_hats[i] = var_y_hat
        cov_hats[i] = cov_hat

    # construct the variance of the compound power tuning estimator for each problem
    compound_pt_vars = var_y_hats / data.ns + (data.ns + data.Ns) / (data.Ns * data.ns) * cp_lambda ** 2 * var_fx_hats \
        - (2 / data.ns) * cp_lambda * cov_hats

    # construct the plug-in "i.e. should be considered fixed" compound var mean
    compound_pt_var_mean = np.mean(compound_pt_vars)

    # further calculate the unbiased estimator for the covariance term between the PT estimator and prediction mean
    cov_pt_pred_mean = cov_hats / (data.Ns * data.ns)

    # when calculating CURE, we will go back to using the original unbiased sample-based estimate of second moments
    def sure_fn(omega: float) -> float:
        omega_scaled = omega / (omega + compound_pt_var_mean)

        term_1 = (2 * omega_scaled - 1) * compound_pt_var_mean
        term_2 = 2 * (1 - omega_scaled) * cov_pt_pred_mean
        term_3 = ((1 - omega_scaled) * (compound_estimates - f_x_bar)) ** 2

        return np.sum(term_1 + term_2 + term_3)

    # step 3: compute the optimal lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    omega_upper = cutoff / (1 - cutoff) * compound_pt_var_mean
    optimal_omega = _minimize_lbfgs(sure_fn, bounds=(0, omega_upper))

    optim_omega_scaled = optimal_omega / (optimal_omega + compound_pt_var_mean)

    compound_sure_ppi_estimates = optim_omega_scaled * \
        compound_estimates + (1 - optim_omega_scaled) * f_x_bar

    if not get_lambda and not get_omega:
        return compound_sure_ppi_estimates
    elif get_lambda:
        return compound_sure_ppi_estimates, cp_lambda
    elif get_omega:
        return compound_sure_ppi_estimates, optim_omega_scaled
    else:
        return compound_sure_ppi_estimates, optim_omega_scaled, cp_lambda
