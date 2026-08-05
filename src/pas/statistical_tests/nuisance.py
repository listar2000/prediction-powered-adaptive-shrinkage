"""Per-problem nuisance parameters of the rebiasing model, for sanity checks.

The rebiasing model (paper Eq. 3) writes each problem as

    (theta_hat_b_i, b_hat_i) | (theta_i, b_i)
        ~ N( (theta_i + b_i, b_i),
             [[sigma_i^2,        rho_i sigma_i tau_i],
              [rho_i sigma_i tau_i,      tau_i^2    ]] ),      b_i ~ G,

so each problem carries a *known* nuisance triple

    sigma_i^2 = Var[theta_hat_b_i]   (variance of the biased estimator),
    tau_i^2   = Var[b_hat_i]         (variance of the bias estimator),
    rho_i     = Corr(theta_hat_b_i, b_hat_i),

alongside the latent bias ``b_i`` drawn from the shared prior ``G``.  Fitting
one ``G`` across problems presumes ``b_i`` is independent of that triple, so
plotting ``b_hat_i`` against each of ``rho_i``, ``tau_i^2`` and ``sigma_i^2``
is a direct sanity check on the modelling assumption.

Mapping onto the code
---------------------
:func:`pas.estimators.eb_estimators._get_power_tuned_ppi_estimators_plus_bias`
returns the covariance of ``(pt_i, b_hat_i)`` rather than of
``(theta_hat_b_i, b_hat_i)``, because that is the form the shrinkage step
consumes.  Since ``pt_i = theta_hat_b_i - b_hat_i``, the two parametrizations
are related exactly:

    sigma_i^2           = Sigma_11 + 2 Sigma_12 + Sigma_22,
    rho_i sigma_i tau_i = Sigma_12 + Sigma_22,
    tau_i^2             = Sigma_22.

For the power-tuned decomposition ``theta_hat_b_i`` is the unlabelled
prediction mean, so ``sigma_i^2`` reduces to ``Var(f)/N_i`` and
``rho_i sigma_i tau_i`` to ``(1 - lambda_i) Var(f)/N_i``; both identities are
checked in the test suite.  Note that ``rho_i = 0`` whenever
``lambda_i = 1`` -- vanilla PPI leaves the biased estimator and the bias
estimate independent -- and that the correlation is what makes the power-tuned
variant the interesting case for this diagnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pas.estimators.eb_estimators import (
    _get_power_tuned_lambdas,
    _get_power_tuned_ppi_estimators_plus_bias,
)

NUISANCE_COLUMNS = ("bias_hat", "rho", "tau2", "sigma2")


def _nuisance_from_pt_covariance(cov_mats: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Re-express Cov(pt, b_hat) as the paper's ``(sigma^2, tau^2, rho)``."""
    cov_mats = np.asarray(cov_mats, dtype=float)
    s11 = cov_mats[:, 0, 0]
    s12 = cov_mats[:, 0, 1]
    s22 = cov_mats[:, 1, 1]

    sigma2 = s11 + 2.0 * s12 + s22
    tau2 = s22
    covariance = s12 + s22

    scale = np.sqrt(np.maximum(sigma2, 0.0) * np.maximum(tau2, 0.0))
    rho = np.divide(
        covariance,
        scale,
        out=np.full_like(covariance, np.nan),
        where=scale > 0.0,
    )
    return sigma2, tau2, rho


def power_tuned_nuisance_summary(
    pred_unlabelled: Sequence[np.ndarray],
    pred_labelled: Sequence[np.ndarray],
    y_labelled: Sequence[np.ndarray],
    *,
    group_ids: Optional[Sequence] = None,
    clip_lambda: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """One row per problem: the observed ``b_hat_i`` and its nuisance triple.

    Uses the same power-tuned decomposition the rebiased intervals run on, so
    the numbers plotted are exactly the ones the estimator sees.
    """
    lambdas = _get_power_tuned_lambdas(
        pred_unlabelled, pred_labelled, y_labelled, clip=clip_lambda
    )
    theta_b, bias_hat, pt, cov_mats = _get_power_tuned_ppi_estimators_plus_bias(
        pred_unlabelled, pred_labelled, y_labelled, lambdas
    )
    sigma2, tau2, rho = _nuisance_from_pt_covariance(cov_mats)

    if group_ids is None:
        group_ids = np.arange(len(bias_hat))

    summary = pd.DataFrame(
        {
            "group_id": np.asarray(group_ids),
            "n": np.array([len(y) for y in y_labelled], dtype=int),
            "N": np.array([len(z) for z in pred_unlabelled], dtype=int),
            "lambda_value": lambdas,
            "theta_b": theta_b,
            "pt": pt,
            "bias_hat": bias_hat,
            "sigma2": sigma2,
            "tau2": tau2,
            "rho": rho,
        }
    )
    return summary


def load_lmarena_nuisance_summary(
    csv_path: Union[str, Path],
    *,
    train_test_split: float = 0.1,
    split_seed: int = 42,
    clip_lambda: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Nuisance summary for one LM Arena labelled/unlabelled split.

    ``train_test_split`` and ``split_seed`` default to the values the LMArena
    CI benchmark runs at, so the figure describes the data the reported
    experiment actually uses.
    """
    from pas.datasets.lmarena import LMArenaDataset

    dataset = LMArenaDataset(
        file_path=Path(csv_path),
        train_test_split=train_test_split,
        split_seed=split_seed,
    )
    summary = power_tuned_nuisance_summary(
        dataset.pred_unlabelled,
        dataset.pred_labelled,
        dataset.y_labelled,
        clip_lambda=clip_lambda,
    )
    summary.attrs.update(
        {
            "dataset": "lmarena",
            "source_path": str(dataset.file_path),
            "train_test_split": float(train_test_split),
            "split_seed": int(split_seed),
        }
    )
    return summary


def load_amazon_synthetic_nuisance_summary(**kwargs) -> pd.DataFrame:
    """Nuisance summary for one replicate of the Amazon-derived synthetic study.

    The synthetic generator works at the summary level -- it draws
    ``(Ybar_i, Zbar_i, Ztilde_i)`` from a known covariance -- so here
    ``sigma_i^2``, ``tau_i^2`` and ``rho_i`` are the exact values used to
    generate the data rather than plug-in estimates.  Keyword arguments are
    forwarded to
    :func:`pas.statistical_tests.exchangeability.load_amazon_synthetic_pseudo_oracle`.
    """
    from pas.statistical_tests.exchangeability import (
        load_amazon_synthetic_pseudo_oracle,
    )

    summary = load_amazon_synthetic_pseudo_oracle(**kwargs)
    missing = set(NUISANCE_COLUMNS).difference(summary.columns)
    if missing:
        raise ValueError(
            f"synthetic loader did not provide nuisance columns: {sorted(missing)}"
        )
    return summary


def nuisance_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of ``b_hat_i`` with each nuisance parameter.

    A drifting mean is what this diagnostic is looking for.  Note that spread
    is a different matter: ``b_hat_i = b_i + noise`` with noise variance
    ``tau_i^2``, so the vertical scatter *must* widen with ``tau_i^2`` even
    when the prior is shared, and only a trend in the centre is evidence
    against the model.
    """
    bias = summary["bias_hat"].to_numpy(dtype=float)
    records = []
    for column in ("rho", "tau2", "sigma2"):
        values = summary[column].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(bias)
        if finite.sum() < 3 or np.std(values[finite]) == 0.0:
            correlation = np.nan
        else:
            correlation = float(np.corrcoef(bias[finite], values[finite])[0, 1])
        records.append(
            {"nuisance": column, "n": int(finite.sum()), "pearson_r": correlation}
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "NUISANCE_COLUMNS",
    "load_amazon_synthetic_nuisance_summary",
    "load_lmarena_nuisance_summary",
    "nuisance_correlations",
    "power_tuned_nuisance_summary",
]
