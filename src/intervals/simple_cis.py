"""
Classical confidence intervals for the mean.
"""
import numpy as np
from datasets.dataset import PasDataset
from utils import _zconfint


def get_mle_cis(data: PasDataset, alpha: float = 0.1,
                alternative: str = "two-sided") -> np.ndarray:
    """Classical CLT-based confidence interval for each problem's mean.

    For problem i, the CI is:  Y_i.mean() +/- z * Y_i.std(ddof=1) / sqrt(n_i)

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    means = np.array([y.mean() for y in data.y_labelled])
    ses = np.array([y.std(ddof=1) / np.sqrt(n)
                    for y, n in zip(data.y_labelled, data.ns)])
    return _zconfint(means, ses, alpha, alternative)
