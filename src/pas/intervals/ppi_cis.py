"""
Prediction-Powered Inference (PPI) confidence intervals for the mean.
"""
import numpy as np
from pas.datasets.dataset import PasDataset
from pas.estimators.ppi_estimators import get_vanilla_ppi_estimators, get_pt_ppi_estimators
from pas.utils import _zconfint


def get_vanilla_ppi_cis(data: PasDataset, alpha: float = 0.1,
                        alternative: str = "two-sided") -> np.ndarray:
    """Vanilla PPI confidence interval (lambda=1) for each problem's mean.

    The point estimate is:  Y_i.mean() + (pred_unlabelled_i.mean() - pred_labelled_i.mean())

    The standard error combines:
      - Imputed SE: std(pred_unlabelled_i) / sqrt(N_i)
      - Rectifier SE: std(Y_i - pred_labelled_i) / sqrt(n_i)

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic,
            "PPI++: Efficient Prediction-Powered Inference".
    """
    point_estimates = get_vanilla_ppi_estimators(data)

    imputed_ses = np.array([
        pred_u.std(ddof=1) / np.sqrt(N)
        for pred_u, N in zip(data.pred_unlabelled, data.Ns)
    ])
    rectifier_ses = np.array([
        (y - pred_l).std(ddof=1) / np.sqrt(n)
        for y, pred_l, n in zip(data.y_labelled, data.pred_labelled, data.ns)
    ])
    combined_ses = np.sqrt(imputed_ses ** 2 + rectifier_ses ** 2)

    return _zconfint(point_estimates, combined_ses, alpha, alternative)


def get_pt_ppi_cis(data: PasDataset, alpha: float = 0.1,
                   alternative: str = "two-sided",
                   share_var: bool = True) -> np.ndarray:
    """Power-tuned PPI confidence interval for each problem's mean.

    Uses the power-tuning parameter lambda_i from get_pt_ppi_estimators to
    construct the CI with reduced width when predictions are informative.

    The point estimate is:  Y_i.mean() + lambda_i * (pred_unlabelled_i.mean() - pred_labelled_i.mean())

    The standard error combines:
      - Imputed SE: std(lambda_i * pred_unlabelled_i) / sqrt(N_i)
      - Rectifier SE: std(Y_i - lambda_i * pred_labelled_i) / sqrt(n_i)

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".
        share_var: Whether to share variance/covariance across problems for lambda tuning.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic,
            "PPI++: Efficient Prediction-Powered Inference".
    """
    pt_estimates, lambdas = get_pt_ppi_estimators(
        data, share_var=share_var, get_lambdas=True)

    imputed_ses = np.array([
        (lam * pred_u).std(ddof=1) / np.sqrt(N)
        for lam, pred_u, N in zip(lambdas, data.pred_unlabelled, data.Ns)
    ])
    rectifier_ses = np.array([
        (y - lam * pred_l).std(ddof=1) / np.sqrt(n)
        for lam, y, pred_l, n in zip(lambdas, data.y_labelled, data.pred_labelled, data.ns)
    ])
    combined_ses = np.sqrt(imputed_ses ** 2 + rectifier_ses ** 2)

    return _zconfint(pt_estimates, combined_ses, alpha, alternative)
