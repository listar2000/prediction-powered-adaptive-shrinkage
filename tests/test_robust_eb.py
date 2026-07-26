from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pas.estimators.robust_eb import (
    fit_robust_eb,
    fit_robust_eb_dataset,
    get_unbiased_estimates_and_ses,
)
from pas.intervals import CORE_CI_METHODS
from pas.intervals.robust_eb import (
    robust_eb_critical_value,
    robust_eb_critical_values,
)
from pas.intervals.robust_eb_cis import get_robust_eb_cis


def test_source_critical_values_match_official_ebci_examples() -> None:
    # Values used by the official package's cv tests.
    cases = [
        (25.0, 3.0, 0.05, 11.88358367),
        (1.0, 3.0, 0.20, 1.8494683),
        (100.0, 3.0, 0.05, 24.86172217),
        (2.8**2, 8.0, 0.10, 7.0779747158),
    ]
    for m2, kappa, alpha, expected in cases:
        actual = robust_eb_critical_value(m2, alpha=alpha, kappa=kappa)
        assert actual == pytest.approx(expected, rel=2e-8, abs=2e-8)


def test_finite_lookup_is_conservative_and_close_to_exact() -> None:
    cases = [
        (0.33, 2.60, 0.10),
        (0.77, 3.10, 0.05),
        (0.11, 2.05, 0.20),
    ]
    for m2, kappa, alpha in cases:
        exact = robust_eb_critical_value(m2, alpha=alpha, kappa=kappa)
        lookup = float(
            robust_eb_critical_values(
                np.asarray([m2]), alpha=alpha, kappa=kappa, mode="lookup"
            )[0]
        )
        assert lookup >= exact - 2e-10
        assert lookup <= exact * 1.04


def test_pmt_moments_and_estimator_match_r_formulas() -> None:
    y = np.asarray([1.0, -2.0, 0.5, 1.5])
    se = np.asarray([0.20, 0.30, 0.10, 0.40])
    weights = np.asarray([2.0, 1.0, 3.0, 0.5])

    fit = fit_robust_eb(
        y,
        se,
        weights=weights,
        shrink_to="zero",
        fs_correction="PMT",
    )

    w2 = y**2 - se**2
    w4 = y**4 - 6.0 * se**2 * y**2 + 3.0 * se**4
    raw_mu2 = np.average(w2, weights=weights)
    raw_mu4 = np.average(w4, weights=weights)
    trim2 = (
        2.0
        * np.mean(weights**2 * se**4)
        / (np.sum(weights) * np.mean(weights * se**2))
    )
    trim4 = (
        32.0
        * np.mean(weights**2 * se**8)
        / (np.sum(weights) * np.mean(weights * se**4))
    )
    mu2 = max(raw_mu2, trim2)
    kappa = max(raw_mu4 / mu2**2, 1.0 + trim4 / mu2**2)
    shrinkage = mu2 / (mu2 + se**2)

    assert fit.raw_mu2 == pytest.approx(raw_mu2)
    assert fit.mu2 == pytest.approx(mu2)
    assert fit.kappa == pytest.approx(kappa)
    np.testing.assert_allclose(fit.regression_mean, 0.0)
    np.testing.assert_allclose(fit.shrinkage, shrinkage)
    np.testing.assert_allclose(fit.estimate, shrinkage * y)


def test_inverse_variance_weighted_grand_mean() -> None:
    y = np.asarray([1.0, 2.0, 8.0])
    se = np.asarray([0.5, 1.0, 2.0])
    weights = 1.0 / se**2
    expected_mean = np.average(y, weights=weights)

    fit = fit_robust_eb(y, se, weights=weights, fs_correction="PMT")

    np.testing.assert_allclose(fit.regression_mean, expected_mean)
    np.testing.assert_allclose(fit.coefficients, [expected_mean])
    np.testing.assert_allclose(
        fit.estimate,
        expected_mean + fit.shrinkage * (y - expected_mean),
    )


def _synthetic_ppi_data(seed: int = 0, tasks: int = 12):
    rng = np.random.default_rng(seed)
    pred_labelled = []
    y_labelled = []
    pred_unlabelled = []
    y_unlabelled = []
    truth = []
    for index in range(tasks):
        theta = 0.25 + 0.04 * index
        prediction_bias = 0.04 * np.sin(index)
        labelled_y = rng.normal(theta, 0.35, size=35)
        labelled_pred = labelled_y + prediction_bias + rng.normal(0.0, 0.18, size=35)
        unlabelled_y = rng.normal(theta, 0.35, size=120)
        unlabelled_pred = (
            unlabelled_y + prediction_bias + rng.normal(0.0, 0.18, size=120)
        )
        pred_labelled.append(labelled_pred)
        y_labelled.append(labelled_y)
        pred_unlabelled.append(unlabelled_pred)
        y_unlabelled.append(unlabelled_y)
        truth.append(theta)
    return SimpleNamespace(
        pred_labelled=pred_labelled,
        y_labelled=y_labelled,
        pred_unlabelled=pred_unlabelled,
        y_unlabelled=y_unlabelled,
        true_theta=np.asarray(truth),
        ns=np.asarray([len(x) for x in pred_labelled]),
        Ns=np.asarray([len(x) for x in pred_unlabelled]),
        M=tasks,
        has_true_vars=False,
    )


def test_pt_adapter_and_ci_are_finite() -> None:
    data = _synthetic_ppi_data()
    estimates, ses = get_unbiased_estimates_and_ses(
        data, base_estimator="pt", share_var=False
    )
    assert estimates.shape == (data.M,)
    assert ses.shape == (data.M,)
    assert np.all(np.isfinite(estimates))
    assert np.all(np.isfinite(ses))
    assert np.all(ses >= 0.0)

    intervals = get_robust_eb_cis(
        data,
        alpha=0.10,
        base_estimator="pt",
        share_var=False,
        weights="inverse_variance",
        fs_correction="PMT",
        kappa=3.0,
        cv_mode="lookup",
    )
    assert intervals.shape == (data.M, 2)
    assert np.all(np.isfinite(intervals))
    assert np.all(intervals[:, 0] <= intervals[:, 1])


def test_zero_standard_error_has_stable_inverse_weight_limit() -> None:
    data = SimpleNamespace(
        pred_labelled=[np.ones(8), np.linspace(0.0, 1.0, 8)],
        y_labelled=[np.ones(8), np.linspace(0.0, 1.0, 8)],
        pred_unlabelled=[np.ones(12), np.linspace(0.0, 1.0, 12)],
        y_unlabelled=[np.ones(12), np.linspace(0.0, 1.0, 12)],
        ns=np.asarray([8, 8]),
        Ns=np.asarray([12, 12]),
        M=2,
        has_true_vars=False,
    )
    fit = fit_robust_eb_dataset(
        data,
        base_estimator="classical",
        weights="inverse_variance",
        fs_correction="PMT",
    )
    assert fit.standard_error[0] == 0.0
    assert np.all(np.isfinite(fit.weights))
    assert np.all(fit.weights > 0.0)
    assert fit.shrinkage[0] == pytest.approx(1.0)


def test_robust_eb_ci_is_registered() -> None:
    assert CORE_CI_METHODS["robust_eb_ci"] is get_robust_eb_cis
