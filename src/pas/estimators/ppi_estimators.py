"""
PPI estimators
"""
import numpy as np
from pas.datasets.dataset import PasDataset
from typing import Union, Tuple

#: Floor for plug-in variance estimates of the PT estimator. That variance is a
#: difference of sample moments (`sigma^2/n + lambda^2 tau^2 (n+N)/(nN) -
#: 2 lambda gamma / n`), so in small samples it can come out negative -- on the
#: Amazon `BERT-tuned` corpus this happens for 3 of 200 problems once
#: `share_var=False` makes the moments per-problem. A negative A_j would push
#: the shrinkage weight `omega_j = lambda / (A_j + lambda)` above 1 and
#: extrapolate *away* from the shrinkage target. Flooring at a tiny positive
#: value sends `omega_j -> 1`, i.e. leaves such a problem's estimate unshrunk,
#: which is the correct zero-noise limit.
MIN_PT_VARIANCE = 1e-12

#: Ways of turning the labelled data into the second moments that the PT / PAS /
#: Shrink-classical / Shrink-average variance formulas need.
#:
#: The paper *assumes* `sigma_j^2, tau_j^2, gamma_j` are known for those
#: estimators, so how they are obtained in practice is an implementation choice
#: the paper does not specify -- hence a knob:
#:
#: - ``"per_problem"``: `sigmahat_j^2` etc. exactly as in Appendix C.1. The
#:   default, and the configuration behind the paper's reported tables. Keeps the
#:   heteroscedasticity that PAS's per-problem `omega_j` exists to exploit.
#: - ``"averaged"``: one shared value per moment, `m^-1 sum_j sigmahat_j^2`. Each
#:   problem's deviations are still measured from *its own* mean, so this
#:   estimates the common within-problem moment without contamination.
#: - ``"pooled"``: one shared value per moment, computed from the concatenated
#:   raw data against a single grand mean. Kept for backwards compatibility only:
#:   by the ANOVA identity this equals the within-problem component *plus* the
#:   between-problem dispersion of the problem means, so it systematically
#:   overstates the within-problem moment (~13% on the Amazon corpora) and gives
#:   9-27x the RMSE of ``"averaged"`` for the same target. Prefer ``"averaged"``.
#:
#: Deliberately NOT available to UniPT / UniPAS: their moment handling is fixed
#: by Appendix C.2 / C.3 and Algorithm 2, so it is part of the estimator's
#: definition rather than a free choice.
MOMENT_ESTIMATORS = ("per_problem", "averaged", "pooled")


def _resolve_moments(moments=None, share_var=None) -> str:
    """Resolve the moment estimator, accepting the legacy `share_var` bool.

    `share_var=False` maps to ``"per_problem"`` and `share_var=True` to
    ``"pooled"``, preserving the historical meaning of the flag.
    """
    # `share_var` used to be the second positional parameter of both this
    # module's helpers and `get_pt_ppi_estimators`; keep such calls working.
    if isinstance(moments, bool):
        moments, share_var = None, moments

    if share_var is not None:
        if moments is not None:
            raise ValueError(
                "pass either `moments` or the legacy `share_var`, not both")
        return "pooled" if share_var else "per_problem"
    if moments is None:
        return "per_problem"
    if moments not in MOMENT_ESTIMATORS:
        raise ValueError(
            f"`moments` must be one of {MOMENT_ESTIMATORS}, got {moments!r}")
    return moments


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


def estimate_second_moments(
    data: PasDataset,
    moments=None,
    share_var=None,
    use_true_moments: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Estimate the per-problem second moments used by the PT/PAS variance formulas.

    Returns ``(var_y, var_f, cov_yf)``, each of shape ``(M,)``, following the
    sample-based estimators of Appendix C.1:

    - ``var_y[j]``  = σ̂_j², variance of `Y` over the `n_j` labelled points;
    - ``var_f[j]``  = τ̂_j², variance of `f(X)` over **all** `n_j + N_j` predictions;
    - ``cov_yf[j]`` = γ̂_j, covariance of `Y` and `f(X)` over the labelled pairs.

    Args:
        data (PasDataset): the dataset object.

        moments (str): which plug-in to use -- one of `MOMENT_ESTIMATORS`; see \
            that constant for what each one does and why this is a knob at all. \
            Default `"per_problem"`.

        share_var (bool): deprecated alias for `moments`; `False` maps to \
            `"per_problem"` and `True` to `"pooled"`.

        use_true_moments (bool): if `True`, `data.true_vars` takes precedence \
            whenever it is available. UniPAS passes `False` because Appendix C.3 \
            defines it using the sample-based moments even when oracle moments \
            happen to be available to other estimators. When oracle moments are \
            used there is nothing to estimate, so `moments` is then ignored.

    Returns:
        (var_y, var_f, cov_yf): the three second-moment arrays, shape `(M,)` each.
    """
    M = data.M
    choice = _resolve_moments(moments, share_var)

    if use_true_moments and data.has_true_vars:
        return (
            np.broadcast_to(np.asarray(data.true_y_vars, dtype=float), (M,)).copy(),
            np.broadcast_to(np.asarray(data.true_vars, dtype=float), (M,)).copy(),
            np.broadcast_to(np.asarray(data.true_covs, dtype=float), (M,)).copy(),
        )

    if choice == "pooled":
        # One grand mean for the whole corpus, so each deviation also absorbs how
        # far its problem's mean sits from the global mean. See MOMENT_ESTIMATORS.
        all_pred_labelled = np.concatenate(data.pred_labelled)
        all_y_labelled = np.concatenate(data.y_labelled)
        var_y = all_y_labelled.var(ddof=1)
        var_f = np.concatenate(
            [all_pred_labelled, np.concatenate(data.pred_unlabelled)]).var(ddof=1)
        cov_yf = np.cov(all_y_labelled, all_pred_labelled, ddof=1)[0, 1]
        return np.full(M, var_y), np.full(M, var_f), np.full(M, cov_yf)

    var_y = np.empty(M)
    var_f = np.empty(M)
    cov_yf = np.empty(M)
    for j in range(M):
        var_y[j] = data.y_labelled[j].var(ddof=1)
        var_f[j] = np.concatenate(
            [data.pred_labelled[j], data.pred_unlabelled[j]]).var(ddof=1)
        cov_yf[j] = np.cov(data.y_labelled[j], data.pred_labelled[j], ddof=1)[0, 1]

    if choice == "averaged":
        # Each problem's deviations were measured from its own mean, so averaging
        # gives a shared value with no between-problem contamination.
        return tuple(np.full(M, float(a.mean())) for a in (var_y, var_f, cov_yf))
    return var_y, var_f, cov_yf


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
    share_var=None,
    get_lambdas: bool = False,
    clip_lambda: bool = True,
    *,
    moments=None,
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

        share_var (bool): deprecated alias for `moments`; `False` -> `"per_problem"`, `True` -> `"pooled"`.

        moments (str): which second-moment plug-in to build λ_i from -- one of `MOMENT_ESTIMATORS`. \
            Default `"per_problem"`, the estimator of Appendix C.1 and the configuration behind the \
            paper's tables. The paper assumes these moments known, so this choice is not specified by it.

        get_lambdas (bool): whether to return the power-tuning parameter λ_i. Default to `False`.

        clip_lambda (bool): whether to clip each λ_i to `[0, 1]`. Default to `True`. Eq. (13) of the paper \
            places no bound on the per-problem λ*_j -- clipping is the PPI++ convention, which keeps the \
            estimator a convex combination and guards against a noisy `Cov(Y, f)` sending λ_i outside the \
            interval (this binds for roughly 1-4% of (problem, replicate) pairs on the paper's datasets). \
            Pass `False` for the literal Eq. (13) estimator. Note the paper *does* clip the single global \
            λ̂ of UniPT (Appendix C.2).

    Returns:
        ppi_estimates: the power-tuned PPI estimator for each product. If `get_lambdas` is `True`, the power-tuning parameters will \
        also be returned.

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic, “PPI++: Efficient Prediction-Powered Inference”.
    """
    _, var_bar, cov_bar = estimate_second_moments(
        data, moments=moments, share_var=share_var)

    # Eq. (13): lambda*_j = (N_j / (n_j + N_j)) * gamma_j / tau_j^2
    lambdas = (data.Ns / (data.ns + data.Ns)) * cov_bar / var_bar
    if clip_lambda:
        lambdas = np.clip(lambdas, 0, 1)

    ppi_estimates = _get_generic_ppi_estimators(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambdas)
    return ppi_estimates if not get_lambdas else (ppi_estimates, lambdas)
