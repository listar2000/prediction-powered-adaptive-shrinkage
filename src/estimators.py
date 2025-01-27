"""
All the estimators are represented as (pure) functions that in principle exerts no side-effects.
In the case that this shall be violated, define a higher-order function.
"""
import numpy as np
from datasets.dataset import PasDataset
from utils import _minimize_lbfgs
from typing import Union, Tuple
from copy import copy

# Trivial estimators
def get_mle_estimators(data: PasDataset) -> np.ndarray:
    """ Obtain the MLE estimator, i.e. the mean response for each problem.
    """
    return np.array([y.mean() for y in data.y_labelled])


def get_pred_mean_estimators(data: PasDataset) -> np.ndarray:
    """ Obtain the prediction mean estimator for each problem.
    """
    return np.array([pred.mean() for pred in data.pred_unlabelled])


# PPI estimators
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


def get_power_tuned_ppi_estimators(
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


# Empirical Bayes (Shrinkage) estimators

# Shrinkage-only estimator
def get_eb_sure_estimators(data: PasDataset, get_lambdas: bool = False, share_var: bool = True, cutoff: float = 0.999) \
        -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """ Obtain the SURE-minimizing shrinkage estimator for the PPI problem.

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

    f_x_tilde_bar, y_bar, var_y = np.array(f_x_tilde_bar), np.array(y_bar), np.array(var_y)

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


def get_sure_ppi_estimators(data: PasDataset, get_lambdas: bool = False, get_omegas: bool = False, share_var: bool = True, \
        cutoff: float = 0.999, old_ver: bool = False):
    
    f_x_bar = np.array([data.pred_unlabelled[i].mean() for i in range(data.M)])

    pt_ppi_estimates, sure_lambdas = get_power_tuned_ppi_estimators(
        data, get_lambdas=True, share_var=share_var)

    if data.has_true_vars:
        var_f_x = data.true_fx_vars
        var_y = data.true_vars
        cov_y_f_x = data.true_covs
    else:
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        cov_y_f_x = np.cov(np.concatenate(data.y_labelled), np.concatenate(data.pred_labelled), ddof=1)[0, 1]
        var_f_x = np.concatenate(data.pred_unlabelled).var(ddof=1)
    
    var_y = var_y / data.ns
    cov_y_f_x = cov_y_f_x / data.ns
    var_f_x_scaled = var_f_x * ((data.Ns + data.ns) / (data.Ns * data.ns))

    var_pt_ppi = var_y + sure_lambdas ** 2 * var_f_x_scaled - 2 * sure_lambdas * cov_y_f_x

    def sure_fn(lambda_: float) -> float:
        old_term = np.sum((var_pt_ppi / (var_pt_ppi + lambda_) ** 2)
                      * (var_pt_ppi * (f_x_bar - pt_ppi_estimates) ** 2 + lambda_ ** 2 - var_pt_ppi ** 2))
        if old_ver:
            return old_term
        else:
            # note here we use the original (not scaled) version of `var_f_x`
            extra_term = np.sum(2 * (var_pt_ppi / (var_pt_ppi + lambda_)) * sure_lambdas * var_f_x / data.Ns)
            return old_term + extra_term

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
    pt_ppi_estimates, sure_lambdas = get_power_tuned_ppi_estimators(
        dataset_split, share_var=share_var, get_lambdas=True
    )
    
    # Calculate shrinkage target using the second portion
    sure_f_x_bar = np.array([sure_pred_unlabelled[i].mean() for i in range(data.M)])
    
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


# Aliases for the estimators
ALL_ESTIMATORS = {
    "mle": get_mle_estimators,
    "pred_mean": get_pred_mean_estimators,
    "vanilla_ppi": get_vanilla_ppi_estimators,
    "power_tuned_ppi": get_power_tuned_ppi_estimators,
    "eb_sure": get_eb_sure_estimators,
    "sure_ppi": get_sure_ppi_estimators,
    "split_sure_ppi": get_split_sure_ppi_estimators,
}


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


def get_compound_ppi_estimators(data: PasDataset, get_lambda: bool = False):
    """ Obtain the compound PPI estimator for the PPI problem.

    In the compound PPI setting, our goal is to find a single 'lambda' (i.e. power-tuning parameter) that works well for all problems.
    That is, we add up the covariance and variance terms together to calculate the optimal power-tuning parameter. Note that in this
    setting it does not make too much sense to "share variance" like in other estimators.

    TO PAPER READERS: this estimator turns out to be not very interesting, so feel free to ignore it in benchmarking. 

    Args:
        data (PasDataset): the dataset object.

        get_lambda (bool): whether to return the power-tuning parameter λ_i. Default to `False`.

    Returns:
        ppi_estimates: the compound PPI estimator for each product. If `get_lambda` is `True`, the power-tuning parameters will \
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


def get_mixed_compound_pt_ppi_estimators(data: PasDataset, get_w: bool = False):
    compound_ppi_estimates, lambda_cp = get_compound_ppi_estimators(
        data, get_lambda=True)
    pt_ppi_estimates, lambda_pt = get_power_tuned_ppi_estimators(
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
        pt_pred_unlabelled, pt_y_unlabelled = [], []  # for PT-PPI estimation (fold k)
        sure_pred_unlabelled, sure_y_unlabelled = [], []  # for shrinkage (all other folds)
        
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
            
            # Store remaining data for shrinkage
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
        pt_ppi_estimates, sure_lambdas = get_power_tuned_ppi_estimators(
            dataset_split, share_var=share_var, get_lambdas=True
        )
        
        # Calculate shrinkage target using other folds
        sure_f_x_bar = np.array([sure_pred_unlabelled[i].mean() for i in range(data.M)])
        
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
        sure_estimates = lambdas * pt_ppi_estimates + (1 - lambdas) * sure_f_x_bar
        
        all_estimates.append(sure_estimates)
        all_lambdas.append(lambdas)
    
    # Average estimates across folds
    final_estimates = np.mean(all_estimates, axis=0)
    
    return final_estimates if not get_lambdas else (final_estimates, all_lambdas)