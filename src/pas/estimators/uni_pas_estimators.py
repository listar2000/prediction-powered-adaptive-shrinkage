"""
UniPAS estimators
"""
import numpy as np
from pas.datasets.dataset import PasDataset
from pas.estimators.ppi_estimators import (
    MIN_PT_VARIANCE,
    _get_generic_ppi_estimators,
    estimate_second_moments,
)
from pas.utils import _minimize_lbfgs


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

    # The paper clips the global tuning parameter before use:
    # lambda_hat_clip := clip(lambda_hat, [0, 1]) (Appendix C.2).
    lambda_ = float(np.clip(numerator / denominator, 0.0, 1.0))
    ppi_estimates = _get_generic_ppi_estimators(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambda_)
    return ppi_estimates if not get_lambda else (ppi_estimates, lambda_)


def get_uni_pas_estimators(data: PasDataset, get_lambda: bool = False, get_omega: bool = False, cutoff: float = 0.999):
    """
    Obtain the univariate PAS estimator from the paper (Appendix C.3, Definition C.4).

    Three distinct variance quantities appear, following Eqs. (25)-(26):

    - ``sigma_dot`` (σ̇_j²): the per-problem sample-based variance of the UniPT
      estimator, used inside the CURE objective itself;
    - ``gamma_dot`` (γ̇_j): ``lambda_hat_clip * tau_hat_j^2 / N_j``, the covariance
      between the UniPT estimator and the shrinkage target ``Ztilde_j^f``;
    - ``sigma_check`` (σ̌_j²): the same variance formula but built from moments
      *averaged across problems*, used only for the shrinkage weight ``omega_j``.
      It still varies with ``j`` through ``n_j`` and ``N_j``.
    """
    # step 1: obtain the compound PPI estimates and lambda
    compound_estimates, cp_lambda = get_uni_pt_estimators(
        data, get_lambda=True)

    f_x_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])

    # Step 2: always recompute the sample-based unbiased second moments. UniPAS
    # is fully data-driven in Appendix C.3, even when the dataset also exposes
    # oracle moments for PT/PAS (as the synthetic benchmark does).
    var_y_hats, var_fx_hats, cov_hats = estimate_second_moments(
        data, share_var=False, use_true_moments=False)

    n, N = data.ns, data.Ns
    var_fx_scale = (n + N) / (N * n)

    # σ̇_j²: per-problem variance of the UniPT estimator (CURE objective).
    sigma_dot = var_y_hats / n + var_fx_scale * cp_lambda ** 2 * var_fx_hats \
        - (2 / n) * cp_lambda * cov_hats

    # γ̇_j = lambda_hat_clip * tau_hat_j^2 / N_j. This is Cov(UniPT_j, Ztilde_j^f);
    # the labelled and unlabelled samples are independent, so only the unlabelled
    # prediction mean contributes.
    gamma_dot = cp_lambda * var_fx_hats / N

    # σ̌_j²: averaged moments, still problem-specific through n_j and N_j.
    # Floored for the same reason as MIN_PT_VARIANCE: it drives omega_j, which
    # must stay in [0, 1].
    sigma_check = np.maximum(
        var_y_hats.mean() / n
        + var_fx_scale * cp_lambda ** 2 * var_fx_hats.mean()
        - (2 / n) * cp_lambda * cov_hats.mean(),
        MIN_PT_VARIANCE,
    )

    def sure_fn(omega: float) -> float:
        omega_j = omega / (omega + sigma_check)

        term_1 = (2 * omega_j - 1) * sigma_dot
        term_2 = 2 * (1 - omega_j) * gamma_dot
        term_3 = ((1 - omega_j) * (compound_estimates - f_x_bar)) ** 2

        return np.sum(term_1 + term_2 + term_3)

    # step 3: compute the optimal lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    omega_upper = cutoff / (1 - cutoff) * sigma_check.max()
    optimal_omega = _minimize_lbfgs(sure_fn, bounds=(0, omega_upper))

    optim_omega_scaled = optimal_omega / (optimal_omega + sigma_check)

    compound_sure_ppi_estimates = optim_omega_scaled * \
        compound_estimates + (1 - optim_omega_scaled) * f_x_bar

    if get_lambda and get_omega:
        return compound_sure_ppi_estimates, cp_lambda, optim_omega_scaled
    elif get_lambda:
        return compound_sure_ppi_estimates, cp_lambda
    elif get_omega:
        return compound_sure_ppi_estimates, optim_omega_scaled
    return compound_sure_ppi_estimates
