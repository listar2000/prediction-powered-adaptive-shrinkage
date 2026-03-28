"""
Classical confidence intervals for the mean.
"""
import numpy as np
from pas.datasets.dataset import PasDataset
from pas.utils import _zconfint


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


def get_pred_mean_cis(data: PasDataset, alpha: float = 0.1,
                      alternative: str = "two-sided") -> np.ndarray:
    """Prediction-based CLT confidence interval for each problem's mean.

    Uses only the unlabelled predictions as if they were the true labels:
    CI = pred_unlabelled_i.mean() +/- z * pred_unlabelled_i.std(ddof=1) / sqrt(N_i)

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    means = np.array([pred.mean() for pred in data.pred_unlabelled])
    ses = np.array([pred.std(ddof=1) / np.sqrt(N)
                    for pred, N in zip(data.pred_unlabelled, data.Ns)])
    return _zconfint(means, ses, alpha, alternative)


def get_bootstrap_cis(data: PasDataset, alpha: float = 0.1,
                      alternative: str = "two-sided",
                      B: int = 1000, seed: int = None) -> np.ndarray:
    """Nonparametric bootstrap percentile confidence interval for each problem's mean.

    For problem i:
      1. Draw B resamples of size n_i (with replacement) from Y_i.
      2. Compute the mean of each resample.
      3. Use the empirical quantiles of the bootstrap means as CI bounds.

    This avoids the normal approximation used by get_mle_cis, which can matter
    when per-problem sample sizes are small or data are skewed.

    Args:
        data: Dataset with M problems.
        alpha: Error level; targets 1-alpha coverage. Default 0.1 (90% CI).
        alternative: "two-sided", "larger", or "smaller".
        B: Number of bootstrap resamples per problem. Default 1000.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (M, 2) with columns [lower, upper].
    """
    rng = np.random.default_rng(seed)
    M = data.M
    ci = np.empty((M, 2))

    for i in range(M):
        y = data.y_labelled[i]
        n = len(y)
        # Draw B bootstrap samples and compute their means
        boot_idx = rng.integers(0, n, size=(B, n))
        boot_means = y[boot_idx].mean(axis=1)

        if alternative == "two-sided":
            ci[i, 0] = np.percentile(boot_means, 100 * alpha / 2)
            ci[i, 1] = np.percentile(boot_means, 100 * (1 - alpha / 2))
        elif alternative == "larger":
            ci[i, 0] = np.percentile(boot_means, 100 * alpha)
            ci[i, 1] = np.inf
        elif alternative == "smaller":
            ci[i, 0] = -np.inf
            ci[i, 1] = np.percentile(boot_means, 100 * (1 - alpha))
        else:
            raise ValueError(
                f"alternative must be 'two-sided', 'larger', or 'smaller', got '{alternative}'"
            )

    return ci
