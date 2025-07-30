"""
PPI estimators
"""
import numpy as np
from datasets.dataset import PasDataset
from typing import Union, Tuple


def _get_generic_ppi_estimators(f_x_tilde: np.ndarray, f_x: np.ndarray, y: np.ndarray, lambda_: Union[float, np.ndarray]) -> np.ndarray:
    """ Helper function to compute the PPI estimator for the PPI problem.

    Args:
        f_x_tilde (np.ndarray): the prediction mean of the unlabelled data for each product.

        f_x (np.ndarray): the prediction mean of the labelled data for each product.

        y (np.ndarray): the mean response (MLE) of the labelled data for each product.

        lambda (Union[float, np.ndarray]): the power-tuning parameter λ_i for each product.

    Returns:
        ppi: the PPI estimator for each product.
    """
    ppi = []
    flag = isinstance(lambda_, np.ndarray)
    for i in range(len(f_x_tilde)):
        lbd = lambda_[i] if flag else lambda_
        ppi_i = y[i].mean() + lbd * (f_x_tilde[i].mean() - f_x[i].mean())
        ppi.append(ppi_i)
    return np.array(ppi)


def get_vanilla_ppi_estimators(data: PasDataset) -> np.ndarray:
    """ Obtain the vanilla PPI estimator for the PPI problem. This estimator is **non-compound**.

    The vanilla PPI estimator is given by:

    θ_i^PPI = θ_i + (μ_i - κ_i)

    where `θ_i` is the MLE, `μ_i`/`κ_i` are the prediction mean of the unlabelled/labelled data for the i^th problem.

    Args:
        data (PasDataset): the dataset object.

    Returns:
        ppi_estimates: the vanilla PPI estimator for each product.

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic, “PPI++: Efficient Prediction-Powered Inference”.
    """
    return _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, 1.0)


def get_pt_ppi_estimators(
    data: PasDataset,
    share_var: bool = True,
    get_lambdas: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Obtain the power-tuned PPI estimator for the PPI problem. This estimator is **non-compound**.

    The power-tuning parameter λ_i for the i^th problem is given by:

    λ_i = (N_i / (n_i + N_i)) * Cov(Y_i, f(X_i)) / Var(f(X_i))

    where `Cov(Y_i, f(X_i))` is the sample covariance of the `n_i` paired labelled data and `Var(f(X_i))` \
        is the sample variance calculated from `n_i + N_i` unlabelled data. The final estimator is given by:

    θ_i^PPI = θ_i + λ_i * (μ_i - κ_i)

    where `θ_i` is the MLE, `μ_i`/`κ_i` are the prediction mean of the unlabelled/labelled data for the i^th problem.

    Args:
        data (PasDataset): the dataset object.

        share_var (bool): whether to share the variance & covariance across all problems. Default to `False`.

        get_lambdas (bool): whether to return the power-tuning parameter λ_i. Default to `False`.

    Returns:
        ppi_estimates: the power-tuned PPI estimator for each product. If `get_lambdas` is `True`, the power-tuning parameters will \
        also be returned.

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic, “PPI++: Efficient Prediction-Powered Inference”.
    """
    lambdas = []
    if share_var:
        # concatenate all the predictions
        all_pred_labelled = np.concatenate(data.pred_labelled)
        all_pred_unlabelled = np.concatenate(data.pred_unlabelled)
        all_y_labelled = np.concatenate(data.y_labelled)
        var_bar = np.concatenate(
            [all_pred_labelled, all_pred_unlabelled]).var(ddof=1)
        cov_bar = np.cov(all_pred_labelled, all_y_labelled, ddof=1)[0, 1]

    for i in range(data.M):
        n, N = data.ns[i], data.Ns[i]
        if data.has_true_vars:
            var_bar = data.true_vars[i]
            cov_bar = data.true_covs[i]
        elif not share_var:
            # need to calculate the variance and covariance for each problem
            var_bar = np.concatenate(
                [data.pred_labelled[i], data.pred_unlabelled[i]]).var(ddof=1)
            cov_bar = np.cov(
                data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1]
        # compute the lambda for each problem
        lambda_i = (N / (n + N)) * cov_bar / var_bar
        lambda_i = np.clip(lambda_i, 0, 1)
        lambdas.append(lambda_i)

    ppi_estimates = _get_generic_ppi_estimators(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, np.array(lambdas))
    return ppi_estimates if not get_lambdas else (ppi_estimates, np.array(lambdas))
