"""Regression tests for the second-round audit fixes.

Companion to `test_estimator_fixes.py`, which pins the estimator formulas to the
paper's appendices. This module pins the *plumbing*: the dataset interface, the
argument defaults, and the guards around plug-in quantities that can go
negative. Each test names the defect it prevents.

Paper references (arXiv:2502.14166v3):
  Eq. (13)  per-problem lambda*_j -- no clipping in the paper
  C.2       global lambda_hat IS clipped: lambda_hat_clip := clip(lambda_hat, [0, 1])
  C.1       per-problem second moments (the `share_var=False` convention)
"""
from __future__ import annotations

import numpy as np
import pytest

from test_estimator_fixes import _ToyDataset  # shared fixture

from pas.config import DATA_PATHS
from pas.datasets.dataset import PasDataset
from pas.estimators import PAPER_TABLE3_ESTIMATORS
from pas.estimators.eb_estimators import (
    _get_global_ppi_lambda,
    _param_prior_computation_for_bias,
)
from pas.estimators.pas_estimators import (
    get_pas_estimators,
    get_shrinkage_only_estimators,
    get_shrinkage_to_mean_estimators,
)
from pas.estimators.ppi_estimators import get_pt_ppi_estimators
from pas.estimators.uni_pas_estimators import get_uni_pt_estimators
from pas.intervals.ppi_cis import get_pt_ppi_cis


@pytest.fixture
def toy():
    return _ToyDataset(seed=1)


# ---------------------------------------------------------------------------
# The documented dataset interface
# ---------------------------------------------------------------------------

class _MinimalDataset(PasDataset):
    """A dataset written exactly as the README documents it.

    `load_data` takes no arguments and reads `self.split_seed`; `reload_data` is
    *not* overridden. The base class used to call `self.load_data(split_seed)`
    positionally, which raised TypeError here -- and, for the shipped loaders
    (whose first parameter is `train_test_split`), would have passed the seed as
    a labelled fraction.
    """

    def __init__(self, M: int = 6):
        self._M = M
        super().__init__("minimal")

    def load_data(self):
        rng = np.random.default_rng(self.split_seed)
        pred_l, y_l, pred_u, y_u, truth = [], [], [], [], []
        for j in range(self._M):
            z_l = rng.normal(j * 0.1, 1.0, 9)
            z_u = rng.normal(j * 0.1, 1.0, 21)
            pred_l.append(z_l)
            pred_u.append(z_u)
            y_l.append(z_l + rng.normal(0.0, 0.4, 9))
            y_u.append(z_u + rng.normal(0.0, 0.4, 21))
            truth.append(j * 0.1)
        return pred_l, y_l, pred_u, y_u, np.array(truth)


def test_default_reload_data_supports_the_documented_load_data() -> None:
    data = _MinimalDataset()
    assert data.split_seed == 42, "base class must supply a split_seed default"

    first = data.y_labelled[0].copy()
    data.reload_data(split_seed=7)
    assert data.split_seed == 7, "reload_data must record the new seed"
    assert not np.array_equal(first, data.y_labelled[0]), (
        "reload_data must actually redraw the split"
    )


def test_run_benchmark_accepts_a_readme_style_dataset() -> None:
    from pas.experiments import run_benchmark

    frame = run_benchmark(
        _MinimalDataset(), trials=3, summary=False, verbose=False
    )
    assert len(frame) == 3
    assert np.all(np.isfinite(frame["pas"].astype(float)))


# ---------------------------------------------------------------------------
# Falsy-`or` argument defaults
# ---------------------------------------------------------------------------

def test_zero_split_arguments_are_not_silently_replaced() -> None:
    """`x or self.x` swallowed a caller's 0 / 0.0; `x is None` does not."""
    from pas.datasets.galaxy_zoo import GalaxyZooDataset

    if not DATA_PATHS["galaxy"]["predictions"].exists():
        pytest.skip("galaxy data not present")

    data = GalaxyZooDataset()
    baseline = data.y_labelled[0].copy()

    data.reload_data(split_seed=0)
    assert data.split_seed == 0
    assert not np.array_equal(baseline, data.y_labelled[0]), (
        "split_seed=0 must not fall back to the stored default"
    )

    data.reload_data(train_test_split=0.0, split_seed=0)
    assert data.train_test_split == 0.0
    assert np.all(data.ns == 0), (
        "train_test_split=0.0 must be honoured, not replaced by the default"
    )


def test_lmarena_config_default_matches_the_predictor_scripts_use() -> None:
    lmarena = DATA_PATHS["lmarena"]
    assert lmarena["cleaned"] == lmarena["bt_prob"], (
        "the config default must resolve to the Bradley-Terry predictor that "
        "the diagnostics and CI benchmarks use, not the binary one"
    )
    assert lmarena["binary"] != lmarena["bt_prob"]


# ---------------------------------------------------------------------------
# C.2: the global lambda_hat is clipped
# ---------------------------------------------------------------------------

def test_global_lambda_is_clipped_and_agrees_with_uni_pt() -> None:
    # negative_cov drives the raw global ratio below zero.
    data = _ToyDataset(seed=3, negative_cov=True)

    clipped = _get_global_ppi_lambda(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled
    )
    raw = _get_global_ppi_lambda(
        data.pred_unlabelled, data.pred_labelled, data.y_labelled, clip=None
    )
    assert raw < 0.0, "fixture should produce a negative unclipped ratio"
    assert clipped == 0.0

    # Both modules must implement the same C.2 estimator.
    _, uni_pt_lambda = get_uni_pt_estimators(data, get_lambda=True)
    np.testing.assert_allclose(clipped, uni_pt_lambda)


# ---------------------------------------------------------------------------
# Negative plug-in Var(b_hat) must not poison the prior fit
# ---------------------------------------------------------------------------

def test_bias_prior_survives_a_negative_measurement_variance() -> None:
    bias = np.array([0.4, -0.2, 0.1, 0.3, -0.5, 0.2])
    cov_mats = np.zeros((bias.size, 2, 2))
    cov_mats[:, 1, 1] = np.array([0.05, 0.04, -0.02, 0.06, 0.03, 0.05])
    cov_mats[:, 0, 1] = cov_mats[:, 1, 1]
    cov_mats[:, 0, 0] = np.abs(cov_mats[:, 1, 1]) + 0.01

    mu, var = _param_prior_computation_for_bias(bias, cov_mats)
    assert np.isfinite(mu), "log of a negative total variance used to give NaN"
    assert np.isfinite(var) and var > 0.0


# ---------------------------------------------------------------------------
# share_var defaults to the paper's per-problem moments (C.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn",
    [
        get_pt_ppi_estimators,
        get_shrinkage_only_estimators,
        get_shrinkage_to_mean_estimators,
        get_pas_estimators,
    ],
)
def test_share_var_defaults_to_per_problem(fn, toy) -> None:
    default = np.asarray(fn(toy), dtype=float)
    per_problem = np.asarray(fn(toy, share_var=False), dtype=float)
    pooled = np.asarray(fn(toy, share_var=True), dtype=float)

    np.testing.assert_allclose(default, per_problem)
    assert not np.allclose(default, pooled), (
        "fixture must distinguish the two settings for this test to bite"
    )


def test_pt_ci_shares_the_estimator_default(toy) -> None:
    np.testing.assert_allclose(
        get_pt_ppi_cis(toy, alpha=0.1),
        get_pt_ppi_cis(toy, alpha=0.1, share_var=False),
    )


# ---------------------------------------------------------------------------
# Eq. (13): the per-problem lambda clip is opt-out
# ---------------------------------------------------------------------------

def test_clip_lambda_is_optional_and_defaults_to_clipping(toy) -> None:
    _, clipped = get_pt_ppi_estimators(toy, get_lambdas=True)
    _, raw = get_pt_ppi_estimators(toy, get_lambdas=True, clip_lambda=False)

    assert np.all((clipped >= 0.0) & (clipped <= 1.0))
    assert np.any(raw > 1.0) or np.any(raw < 0.0), (
        "fixture must push some lambda_j outside [0, 1] for this test to bite"
    )
    np.testing.assert_allclose(clipped, np.clip(raw, 0.0, 1.0))


@pytest.mark.parametrize(
    "fn", [get_pas_estimators, get_shrinkage_to_mean_estimators]
)
def test_clip_lambda_reaches_the_pas_family(fn, toy) -> None:
    assert not np.allclose(
        np.asarray(fn(toy), dtype=float),
        np.asarray(fn(toy, clip_lambda=False), dtype=float),
    ), f"{fn.__name__} ignores clip_lambda"


# ---------------------------------------------------------------------------
# Legacy: the true-variance branch used tau^2 where sigma^2 was meant
# ---------------------------------------------------------------------------

def test_legacy_split_sure_uses_var_y_not_var_f() -> None:
    from pas.estimators.legacy_estimators import get_split_sure_ppi_estimators

    class _TrueVarDataset(_ToyDataset):
        def __init__(self, y_var_scale: float):
            self._y_var_scale = y_var_scale
            super().__init__(seed=5)
            self.has_true_vars = True
            var_f = np.array([
                np.concatenate([self.pred_labelled[j], self.pred_unlabelled[j]]).var(ddof=1)
                for j in range(self.M)
            ])
            self.true_vars = list(var_f)
            self.true_covs = list(0.5 * var_f)
            # Only sigma^2 = Var(Y) differs between the two instances.
            self.true_y_vars = list(y_var_scale * var_f)

    np.random.seed(0)
    small = get_split_sure_ppi_estimators(_TrueVarDataset(0.1))
    np.random.seed(0)
    large = get_split_sure_ppi_estimators(_TrueVarDataset(10.0))

    assert not np.allclose(small, large), (
        "the estimates must respond to Var(Y); the branch used to read "
        "`true_vars` (tau^2) and ignore `true_y_vars` (sigma^2) entirely"
    )


# ---------------------------------------------------------------------------
# The demo scripts cover every Table 3 row
# ---------------------------------------------------------------------------

def test_paper_table3_registry_has_all_nine_rows() -> None:
    assert list(PAPER_TABLE3_ESTIMATORS) == [
        "mle", "pred_mean", "ppi", "pt", "shrinkage_only",
        "shrinkage_mean", "pas", "uni_pt", "uni_pas",
    ], "run_benchmark measures '% Improved' against the first key, so mle leads"
