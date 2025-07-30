"""
Utility functions for calculating SURE, etc.
"""
from scipy.optimize import minimize_scalar
import numpy as np
from typing import NamedTuple, Tuple, List
import inspect


def _minimize_lbfgs(fun, args=(), bounds=None):
    return minimize_scalar(fun, bounds=bounds, method='bounded', args=args).x


# metrics
def get_mse(y_true: np.ndarray, esimates: np.ndarray) -> float:
    # Calculate the mean squared error between the true values and the estimates.
    return np.mean((y_true - esimates) ** 2)


def get_mean_improve_ratio(y_true: np.ndarray, esimates: np.ndarray, mle: np.ndarray) -> float:
    # filter out too small values in (y_true - mle) ** 2
    idx = (y_true - mle) ** 2 > 1e-6
    y_true, mle, esimates = y_true[idx], mle[idx], esimates[idx]
    mse_ratio = (y_true - esimates) ** 2 / (y_true - mle) ** 2
    return np.mean(mse_ratio)


def get_decrease_fraction(y_true: np.ndarray, esimates: np.ndarray, mle: np.ndarray) -> float:
    # calculate number of points where (y_true - mle) ** 2 > (y_true - esimates) ** 2
    idx = (y_true - mle) ** 2 > (y_true - esimates) ** 2
    return np.mean(idx)


# get function default arguments
def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


"""
Some visualization functions
"""


def plot_estimators_over_problems(estimators: List[np.ndarray], names: List[str] = None, links: List[Tuple[int, int]] = None):
    """
    Use seaborn to plot line plots of the estimators over the problems (id from 1 to len(n_problems)). Choose different colors and markers
    for each estimator. Link the 
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    n_estimators, n_problems = len(estimators), len(estimators[0])

    if names is None:
        names = [f'Estimator {i}' for i in range(len(estimators))]

    problem_idx = np.arange(1, n_problems + 1)
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'x']
    colors = sns.color_palette('husl', n_estimators)

    fig = plt.figure(figsize=(10, 6))
    for i, estimator in enumerate(estimators):
        sns.lineplot(x=problem_idx, y=estimator,
                     label=names[i], marker=markers[i], color=colors[i], linestyle='')

    for link in links:
        assert len(link) == 2, 'Link should be a Tuple of two integers'
        assert 0 <= link[0] <= n_problems and 1 <= link[1] <= n_problems, 'Link should be a Tuple of two integers'
        # plot vertical lines between estimators in `link` for each problem
        for problem in problem_idx:
            plt.vlines(problem, estimators[link[0] - 1][problem - 1],
                       estimators[link[1] - 1][problem - 1], colors='gray', linestyles='dotted')

    # show legend
    plt.legend()
    plt.xlabel('Problem ID')
    plt.ylabel('Estimated Value')
    plt.show()
    # return the plot so that it can be saved
    return fig


"""
Legacy utils: functions that help the estimator calculations in `estimators.py`
"""


class VarCovStats(NamedTuple):
    """Holds variance, covariance, and means for each problem."""
    var_pred: np.ndarray      # Variance of predictions for each problem
    cov_pred_y: np.ndarray    # Covariance of predictions and true y for each problem
    mean_y: np.ndarray        # MLE for each problem (i.e., y.mean())
    mean_pred_labelled: np.ndarray   # average of labelled predictions
    mean_pred_unlabelled: np.ndarray  # average of unlabelled predictions


def compute_var_cov(
    pred_labelled: List[np.ndarray],
    pred_unlabelled: List[np.ndarray],
    y_labelled: List[np.ndarray],
    share_var: bool = False
) -> VarCovStats:
    """
    Compute variance of predictions and covariance(pred_labelled, y_labelled) 
    for each problem. Optionally share a single global variance/cov across all.
    """

    M = len(pred_labelled)
    var_pred = np.zeros(M)
    cov_pred_y = np.zeros(M)
    mean_y = np.zeros(M)
    mean_pred_labelled = np.zeros(M)
    mean_pred_unlabelled = np.zeros(M)

    if share_var:
        # Concatenate across all problems
        all_pred_labelled = np.concatenate(pred_labelled)
        all_pred_unlabelled = np.concatenate(pred_unlabelled)
        all_y_labelled = np.concatenate(y_labelled)

        global_var = np.concatenate(
            [all_pred_labelled, all_pred_unlabelled]).var(ddof=1)
        global_cov = np.cov(all_pred_labelled, all_y_labelled, ddof=1)[0, 1]

        # For each problem, store the same var/cov
        for i in range(M):
            var_pred[i] = global_var
            cov_pred_y[i] = global_cov
            mean_y[i] = y_labelled[i].mean()
            mean_pred_labelled[i] = pred_labelled[i].mean()
            mean_pred_unlabelled[i] = pred_unlabelled[i].mean()

    else:
        # Problem-specific var/cov
        for i in range(M):
            preds_all = np.concatenate([pred_labelled[i], pred_unlabelled[i]])
            var_pred[i] = preds_all.var(ddof=1)
            cov_pred_y[i] = np.cov(
                pred_labelled[i], y_labelled[i], ddof=1)[0, 1]
            mean_y[i] = y_labelled[i].mean()
            mean_pred_labelled[i] = pred_labelled[i].mean()
            mean_pred_unlabelled[i] = pred_unlabelled[i].mean()

    return VarCovStats(
        var_pred=var_pred,
        cov_pred_y=cov_pred_y,
        mean_y=mean_y,
        mean_pred_labelled=mean_pred_labelled,
        mean_pred_unlabelled=mean_pred_unlabelled
    )


def combine_ppi_estimates(
    mean_y: np.ndarray,
    mean_pred_labelled: np.ndarray,
    mean_pred_unlabelled: np.ndarray,
    lambdas: np.ndarray
) -> np.ndarray:
    """
    Generic final-step function for PPI or PPI-like shrinkage:
        theta_hat_i = mean_y[i] + lambdas[i] * (mean_pred_unlabelled[i] - mean_pred_labelled[i])
    """
    # some checks that the inputs shape are correct
    assert mean_y.shape == mean_pred_labelled.shape == mean_pred_unlabelled.shape, \
        "mean_y, mean_pred_labelled, and mean_pred_unlabelled should have the same shape"
    # lambdas is either a scalar or a vector with the same shape as mean_y
    assert lambdas.shape == mean_y.shape or len(
        lambdas) == 1, "lambdas should be a scalar or a vector with the same shape as mean_y"
    return mean_y + lambdas * (mean_pred_unlabelled - mean_pred_labelled)


if __name__ == '__main__':
    fig = plot_estimators_over_problems([np.random.rand(10), np.random.rand(10)], [
                                        'Estimator 1', 'Estimator 2'], [(1, 2)])
    fig.savefig('test.png')
