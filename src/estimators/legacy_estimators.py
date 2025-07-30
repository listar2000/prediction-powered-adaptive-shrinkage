"""
All the estimators that are not being used in the current version of the code.
"""
import numpy as np
from datasets.dataset import PasDataset
from estimators.ppi_estimators import _get_generic_ppi_estimators, get_pt_ppi_estimators, get_uni_pt_estimators
from utils import _minimize_lbfgs
from copy import copy


def get_shrink_var_ppi_estimators(data: PasDataset, get_lambdas: bool = False):
    """ Obtain the shrinkage variance PPI estimator for the PPI problem.

    This estimator is very similar to the `power-tuned PPI` estimator, with the difference that we estimate the sample variance differently.
    The power-tuning parameter λ_i for the i^th problem is now given by:

    λ_i = (N_i / (n_i + N_i)) * Cov(Y_i, f(X_i)) / V^*_i

    V^*_i = λ_v * V_m + (1 - λ_v) * Var(f(X_i))

    and V_m is the median of the vector V = (SV(f(X_1))), ..., (SV(f(X_M))), where SV(f(X_1)) denotes the sample variance of the prediction for the i^th problem.
    The variance shrinkage parameter λ_v is estimated by:

    λ_v = min(1, sum(SV2_i) / sum( (SV(f(X_i)) - V_m) ** 2 )

    where SV2_i is the `sample variance` of the `sample variance` of the prediction for the i^th problem. See [1] for more details.

    Args:
        data (PasDataset): the dataset object.
        get_lambdas (bool): whether to return the power-tuning parameter λ_i after variance shrinkage. Default to `False`.

    Returns:
        shrink_var_ppi_estimates: the shrinkage variance PPI estimator for each product. If `get_lambdas` is `True`, the power-tuning parameters will \
        also be returned. Default to `False`.

    References:
        [1] R. Opgen-Rhein and K. Strimmer, “Accurate Ranking of Differentially Expressed Genes by a Distribution-Free Shrinkage Approach”.
    """
    # precedure of estimating the shrinkage variance parameter λ_v
    cov_bar = []
    # unbiased sample variance and sample variance of the sample variance
    sv, sv2 = [], []
    for i in range(data.M):
        N = data.Ns[i]
        all_pred = np.concatenate(
            [data.pred_labelled[i], data.pred_unlabelled[i]])

        bar_f_x = all_pred.mean()
        ws = (all_pred - bar_f_x) ** 2
        bar_ws = ws.mean()

        sv.append((N / (N - 1)) * bar_ws)
        sv2.append(N / (N - 1) ** 3 * np.sum((ws - bar_ws) ** 2))

        cov_bar.append(
            np.cov(data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1])

    # calculate the optimal λ_v
    sv, sv2 = np.array(sv), np.array(sv2)
    v_m = np.median(sv)
    lambda_v = np.sum(sv2) / np.sum((sv - v_m) ** 2)
    lambda_v = min(1, lambda_v)

    # calculate the individual λ_i from λ_v
    N, n = np.array(data.Ns), np.array(data.ns)
    lambdas = (N / (n + N)) * np.array(cov_bar) / \
        (lambda_v * v_m + (1 - lambda_v) * sv)
    lambdas = np.clip(lambdas, 0, 1)

    shrink_var_ppi_estimates = _get_generic_ppi_estimators(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambdas)

    return shrink_var_ppi_estimates if not get_lambdas else (shrink_var_ppi_estimates, lambdas)


def get_mixed_compound_pt_ppi_estimators(data: PasDataset, get_w: bool = False):
    """ TODO: add the description of the mixed compound-pt PPI estimator.
    """
    compound_ppi_estimates, lambda_cp = get_uni_pt_estimators(
        data, get_lambda=True)
    pt_ppi_estimates, lambda_pt = get_pt_ppi_estimators(
        data, get_lambdas=True)

    bias_square = np.array([(y.mean() - pred.mean()) ** 2 for y, pred in
                            zip(data.y_labelled, data.pred_labelled)])

    w_numerator = np.sum((lambda_cp * (lambda_cp - lambda_pt)) * bias_square)
    w_denominator = np.sum((lambda_cp - lambda_pt) ** 2 * bias_square)
    w = w_numerator / w_denominator
    w = np.clip(w, 0, 1)

    mixed_ppi_estimates = w * pt_ppi_estimates + \
        (1 - w) * compound_ppi_estimates
    return mixed_ppi_estimates if not get_w else (mixed_ppi_estimates, w)


def get_split_sure_ppi_estimators(data: PasDataset, split_ratio: float = 0.5, cutoff: float = 0.999, share_var: bool = True, get_lambdas: bool = False):
    """ SURE-PPI estimator with split unlabelled data.

    This estimator splits the unlabelled data into two parts for each problem:
    - Part 1 (split_ratio): used for power-tuned PPI estimation
    - Part 2 (1-split_ratio): used for shrinkage estimation

    This avoids using the same data for both PPI and EB shrinkage.

    Args:
        data: Dataset to estimate on
        split_ratio: Ratio of unlabelled data to use for PT-PPI (default: 0.5)
        cutoff: Upper bound cutoff for lambda search (default: 0.99)
        share_var: Whether to share variance in PT-PPI estimation (default: True)
        get_lambdas: Whether to return shrinkage factors (default: False)

    Returns:
        If get_lambdas is False, returns the SURE estimates.
        If get_lambdas is True, returns (estimates, lambdas).
    """
    # Create a copy of dataset to modify unlabelled data
    dataset_split = copy(data)

    # For storing split data
    pt_pred_unlabelled, pt_y_unlabelled = [], []  # for PT-PPI estimation
    sure_pred_unlabelled, sure_y_unlabelled = [], []  # for shrinkage

    # Split unlabelled data for each problem
    for i in range(data.M):
        N = len(data.pred_unlabelled[i])
        N1 = int(N * split_ratio)

        # Randomly select indices for PT-PPI part
        indices = np.random.choice(N, N1, replace=False)
        not_indices = np.setdiff1d(np.arange(N), indices)

        # Split predictions and responses
        pt_pred_unlabelled.append(data.pred_unlabelled[i][indices])
        pt_y_unlabelled.append(data.y_unlabelled[i][indices])

        # Store remaining data for shrinkage
        sure_pred_unlabelled.append(data.pred_unlabelled[i][not_indices])
        sure_y_unlabelled.append(data.y_unlabelled[i][not_indices])

    # Update split dataset with PT-PPI portion
    dataset_split.set_metadata(
        dataset_split.pred_labelled,
        dataset_split.y_labelled,
        pt_pred_unlabelled,
        pt_y_unlabelled,
        dataset_split.true_theta
    )

    # Get PT-PPI estimates using the first portion
    pt_ppi_estimates, sure_lambdas = get_pt_ppi_estimators(
        dataset_split, share_var=share_var, get_lambdas=True
    )

    # Calculate shrinkage target using the second portion
    sure_f_x_bar = np.array([sure_pred_unlabelled[i].mean()
                            for i in range(data.M)])

    # Calculate variances and covariances
    if data.has_true_vars:
        var_y = data.true_vars
        cov_y_f_x = data.true_covs
    else:
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        var_y = var_y / data.ns
        cov_y_f_x = np.cov(np.concatenate(data.y_labelled),
                           np.concatenate(data.pred_labelled), ddof=1)[0, 1]
        cov_y_f_x = cov_y_f_x / data.ns

    var_f_x = np.concatenate(data.pred_unlabelled).var(ddof=1)
    var_f_x = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    # Calculate PT-PPI variance
    var_pt_ppi = var_y + sure_lambdas ** 2 * var_f_x - 2 * sure_lambdas * cov_y_f_x

    # Define SURE function
    def sure_fn(lambda_: float) -> float:
        return np.sum((var_pt_ppi / (var_pt_ppi + lambda_) ** 2)
                      * (var_pt_ppi * (sure_f_x_bar - pt_ppi_estimates) ** 2 + lambda_ ** 2 - var_pt_ppi ** 2))

    # Calculate optimal lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    lbd_upper = cutoff / (1 - cutoff) * var_pt_ppi.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    # Calculate final estimates
    lambdas = optimal_lbd / (var_pt_ppi + optimal_lbd)
    sure_estimates = lambdas * pt_ppi_estimates + (1 - lambdas) * sure_f_x_bar

    return sure_estimates if not get_lambdas else (sure_estimates, lambdas)


def get_kfold_split_sure_ppi_estimators(data: PasDataset, n_folds: int = 2, cutoff: float = 0.999, share_var: bool = True, get_lambdas: bool = False):
    """ K-fold SURE-PPI estimator with split unlabelled data.

    This estimator performs K-fold splitting of unlabelled data and averages the estimates.
    For each fold k:
    - Part k is used for power-tuned PPI estimation
    - Remaining parts are used for shrinkage estimation
    Final estimate is the average across all K estimators.

    Args:
        data: Dataset to estimate on
        n_folds: Number of folds for splitting (default: 2)
        cutoff: Upper bound cutoff for lambda search (default: 0.99)
        share_var: Whether to share variance in PT-PPI estimation (default: True)
        get_lambdas: Whether to return shrinkage factors (default: False)

    Returns:
        If get_lambdas is False, returns the averaged SURE estimates.
        If get_lambdas is True, returns (estimates, list of lambdas from each fold).
    """
    assert n_folds >= 2, "Number of folds must be at least 2"

    # Store estimates and lambdas from each fold
    all_estimates = []
    all_lambdas = []

    # For each fold
    for k in range(n_folds):
        # Create a copy of dataset to modify unlabelled data
        dataset_split = copy(data)

        # For storing split data
        # for PT-PPI estimation (fold k)
        pt_pred_unlabelled, pt_y_unlabelled = [], []
        # for shrinkage (all other folds)
        sure_pred_unlabelled, sure_y_unlabelled = [], []

        # Split unlabelled data for each problem
        for i in range(data.M):
            N = len(data.pred_unlabelled[i])
            fold_size = N // n_folds

            # Get indices for current fold
            start_idx = k * fold_size
            end_idx = start_idx + fold_size if k < n_folds - 1 else N

            # Create boolean mask for current fold
            fold_mask = np.zeros(N, dtype=bool)
            fold_mask[start_idx:end_idx] = True

            # Split data into current fold (for PT-PPI) and other folds (for shrinkage)
            pt_pred_unlabelled.append(data.pred_unlabelled[i][fold_mask])
            pt_y_unlabelled.append(data.y_unlabelled[i][fold_mask])

            sure_pred_unlabelled.append(data.pred_unlabelled[i][~fold_mask])
            sure_y_unlabelled.append(data.y_unlabelled[i][~fold_mask])

        # Update split dataset with PT-PPI portion
        dataset_split.set_metadata(
            dataset_split.pred_labelled,
            dataset_split.y_labelled,
            pt_pred_unlabelled,
            pt_y_unlabelled,
            dataset_split.true_theta
        )

        # Get PT-PPI estimates using the current fold
        pt_ppi_estimates, sure_lambdas = get_pt_ppi_estimators(
            dataset_split, share_var=share_var, get_lambdas=True
        )

        # Calculate shrinkage target using other folds
        sure_f_x_bar = np.array([sure_pred_unlabelled[i].mean()
                                for i in range(data.M)])

        # Calculate variances and covariances
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        var_y = var_y / data.ns

        var_f_x = np.concatenate(data.pred_unlabelled).var(ddof=1)
        var_f_x = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

        cov_y_f_x = np.cov(np.concatenate(data.y_labelled),
                           np.concatenate(data.pred_labelled), ddof=1)[0, 1]
        cov_y_f_x = cov_y_f_x / data.ns

        # Calculate PT-PPI variance
        var_pt_ppi = var_y + sure_lambdas ** 2 * var_f_x - 2 * sure_lambdas * cov_y_f_x

        # Define SURE function
        def sure_fn(lambda_: float) -> float:
            return np.sum((var_pt_ppi / (var_pt_ppi + lambda_) ** 2)
                          * (var_pt_ppi * (sure_f_x_bar - pt_ppi_estimates) ** 2 + lambda_ ** 2 - var_pt_ppi ** 2))

        # Calculate optimal lambda
        assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
        lbd_upper = cutoff / (1 - cutoff) * var_pt_ppi.max()
        optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

        # Calculate estimates for this fold
        lambdas = optimal_lbd / (var_pt_ppi + optimal_lbd)
        sure_estimates = lambdas * pt_ppi_estimates + \
            (1 - lambdas) * sure_f_x_bar

        all_estimates.append(sure_estimates)
        all_lambdas.append(lambdas)

    # Average estimates across folds
    final_estimates = np.mean(all_estimates, axis=0)

    return final_estimates if not get_lambdas else (final_estimates, all_lambdas)
