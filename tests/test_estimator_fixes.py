"""Regression tests pinning the PAS/UniPAS estimators to the paper's formulas.

Each test corresponds to a defect found by comparing the implementation against
the ICML 2025 paper's appendices, so that a future refactor cannot silently
reintroduce one.

Paper references (camera-ready appendix):
  C.1  sample-based second moments sigma_hat^2, tau_hat^2, gamma_hat
  C.2  UniPT, including lambda_hat_clip := clip(lambda_hat, [0, 1])
  C.3  UniPAS: Eqs. (25)-(26), sigma_dot / gamma_dot / sigma_check
  E.4  pseudo ground-truth theta_dot_j := (1/T_j) sum_i Ydot_ij
  (29) shrink-average SURE and Algorithm 4 (target = grand mean of PT)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pas.datasets.dataset import PasDataset
from pas.estimators.pas_estimators import (
    get_pas_estimators,
    get_shrinkage_only_estimators,
    get_shrinkage_to_mean_estimators,
)
from pas.estimators.ppi_estimators import (
    estimate_second_moments,
    get_pt_ppi_estimators,
)
from pas.estimators.uni_pas_estimators import (
    get_uni_pas_estimators,
    get_uni_pt_estimators,
)


class _ToyDataset(PasDataset):
    """Small dataset with deliberately unequal n_j, N_j and heterogeneous moments.

    Unequal sample sizes are what make the pooled-vs-per-problem and the
    scalar-vs-vector variance distinctions observable.
    """

    def __init__(self, seed: int = 0, M: int = 12, negative_cov: bool = False):
        self._seed = seed
        self._M = M
        self._negative_cov = negative_cov
        super().__init__("toy", verbose=False, has_true_vars=False)

    def load_data(self):
        rng = np.random.default_rng(self._seed)
        pred_l, y_l, pred_u, y_u, truth = [], [], [], [], []
        for j in range(self._M):
            n = 8 + 3 * (j % 4)          # 8, 11, 14, 17 -- varies with j
            N = 20 + 7 * (j % 3)         # 20, 27, 34    -- varies with j
            slope = -1.0 if (self._negative_cov and j % 2 == 0) else 1.0
            mu = 0.2 * j
            z_l = rng.normal(mu, 1.0 + 0.3 * (j % 5), n)
            z_u = rng.normal(mu, 1.0 + 0.3 * (j % 5), N)
            pred_l.append(z_l)
            pred_u.append(z_u)
            y_l.append(slope * z_l + rng.normal(0.0, 0.5, n))
            y_u.append(slope * z_u + rng.normal(0.0, 0.5, N))
            truth.append(mu * slope)
        return pred_l, y_l, pred_u, y_u, np.array(truth)

    def reload_data(self, split_seed: int = 0):
        self._seed = split_seed
        self.set_metadata(*self.load_data())


@pytest.fixture
def toy():
    return _ToyDataset(seed=1)


# ---------------------------------------------------------------------------
# C.1 / share_var: second moments
# ---------------------------------------------------------------------------

def test_second_moments_respect_share_var(toy) -> None:
    pooled = estimate_second_moments(toy, share_var=True)
    per_problem = estimate_second_moments(toy, share_var=False)

    for arr in pooled:
        assert arr.shape == (toy.M,)
        assert np.allclose(arr, arr[0]), "share_var=True must pool to one value"
    for arr in per_problem:
        assert arr.shape == (toy.M,)
    assert not np.allclose(per_problem[0], per_problem[0][0]), (
        "share_var=False must vary across problems"
    )

    # tau_hat_j^2 uses all n_j + N_j predictions (C.1), not the unlabelled only.
    expected_var_f = np.array([
        np.concatenate([toy.pred_labelled[j], toy.pred_unlabelled[j]]).var(ddof=1)
        for j in range(toy.M)
    ])
    np.testing.assert_allclose(per_problem[1], expected_var_f)

    # sigma_hat_j^2 from the labelled Y; gamma_hat_j from the labelled pairs.
    np.testing.assert_allclose(
        per_problem[0], [y.var(ddof=1) for y in toy.y_labelled]
    )
    np.testing.assert_allclose(
        per_problem[2],
        [np.cov(toy.y_labelled[j], toy.pred_labelled[j], ddof=1)[0, 1]
         for j in range(toy.M)],
    )


def test_share_var_changes_pas_family_estimates(toy) -> None:
    """share_var must actually reach the SURE variance, not just the PT lambdas."""
    for fn in (get_pas_estimators, get_shrinkage_to_mean_estimators,
               get_shrinkage_only_estimators):
        pooled = np.asarray(fn(toy, share_var=True), dtype=float)
        per_problem = np.asarray(fn(toy, share_var=False), dtype=float)
        assert not np.allclose(pooled, per_problem), (
            f"{fn.__name__} ignores share_var"
        )


def test_shrink_classical_share_var_is_not_inverted(toy) -> None:
    """share_var=True pools sigma_hat^2, matching get_pt_ppi_estimators."""
    pooled_var_y = np.concatenate(toy.y_labelled).var(ddof=1) / toy.ns
    _, omegas = get_shrinkage_only_estimators(toy, share_var=True, get_lambdas=True)
    # With pooled moments the weight is a monotone function of 1/n_j alone, so
    # problems sharing an n_j must share a weight.
    for n_value in np.unique(toy.ns):
        same_n = omegas[toy.ns == n_value]
        assert np.allclose(same_n, same_n[0]), (
            "share_var=True should pool the variance across problems"
        )
    assert pooled_var_y.shape == (toy.M,)


# ---------------------------------------------------------------------------
# Eq. (29) / Algorithm 4: shrink-average target
# ---------------------------------------------------------------------------

def test_shrink_average_targets_the_grand_mean_of_pt(toy) -> None:
    pt = get_pt_ppi_estimators(toy, share_var=False)
    grand_mean = float(np.mean(pt))
    f_x_bar = np.array([z.mean() for z in toy.pred_unlabelled])

    estimates, omegas = get_shrinkage_to_mean_estimators(
        toy, share_var=False, get_omegas=True
    )
    np.testing.assert_allclose(estimates, omegas * pt + (1 - omegas) * grand_mean)

    # Guard against the previous behaviour (shrinking toward the prediction mean).
    assert not np.allclose(estimates, omegas * pt + (1 - omegas) * f_x_bar)

    # Every estimate must lie between its PT value and the single shared target.
    lower = np.minimum(pt, grand_mean) - 1e-12
    upper = np.maximum(pt, grand_mean) + 1e-12
    assert np.all(estimates >= lower) and np.all(estimates <= upper)


# ---------------------------------------------------------------------------
# C.2: UniPT lambda clipping
# ---------------------------------------------------------------------------

def test_unipt_lambda_is_clipped_to_unit_interval() -> None:
    # negative_cov flips Cov(Y, f) < 0 on half the problems, driving the
    # unclipped global lambda below zero.
    data = _ToyDataset(seed=3, negative_cov=True)
    _, lam = get_uni_pt_estimators(data, get_lambda=True)
    assert 0.0 <= lam <= 1.0

    numerator = denominator = 0.0
    for j in range(data.M):
        n, N = data.ns[j], data.Ns[j]
        var_bar = np.concatenate(
            [data.pred_labelled[j], data.pred_unlabelled[j]]).var(ddof=1)
        cov_bar = np.cov(data.pred_labelled[j], data.y_labelled[j], ddof=1)[0, 1]
        numerator += cov_bar / n
        denominator += var_bar * ((N + n) / (N * n))
    raw = numerator / denominator
    assert raw < 0.0, "fixture should produce a negative unclipped lambda"
    np.testing.assert_allclose(lam, np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# C.3 Eqs. (25)-(26): UniPAS CURE
# ---------------------------------------------------------------------------

def test_unipas_matches_the_paper_cure(toy) -> None:
    upt, lam = get_uni_pt_estimators(toy, get_lambda=True)
    f_x_bar = np.array([z.mean() for z in toy.pred_unlabelled])
    var_y, var_f, cov_yf = estimate_second_moments(toy, share_var=False)
    n, N = toy.ns, toy.Ns
    scale = (n + N) / (N * n)

    sigma_dot = var_y / n + scale * lam**2 * var_f - (2 / n) * lam * cov_yf
    gamma_dot = lam * var_f / N
    sigma_check = (var_y.mean() / n + scale * lam**2 * var_f.mean()
                   - (2 / n) * lam * cov_yf.mean())

    # sigma_check must stay problem-specific through n_j, N_j (not a scalar).
    assert not np.allclose(sigma_check, sigma_check[0])

    # gamma_dot is emphatically not the old Cov(Y, f) / (N_j n_j).
    assert not np.allclose(gamma_dot, cov_yf / (N * n))

    def cure(omega):
        w = omega / (omega + sigma_check)
        return np.sum((2 * w - 1) * sigma_dot + 2 * (1 - w) * gamma_dot
                      + ((1 - w) * (upt - f_x_bar)) ** 2)

    estimates, omegas = get_uni_pas_estimators(toy, get_omega=True)
    assert np.asarray(omegas).shape == (toy.M,), "omega_j must be per-problem"
    np.testing.assert_allclose(estimates, omegas * upt + (1 - omegas) * f_x_bar)

    # The returned omega must minimise CURE over the *feasible* interval, which
    # the implementation caps at cutoff/(1-cutoff) * max(sigma_check).
    implied = omegas * sigma_check / (1 - omegas)
    cutoff = 0.999
    omega_upper = cutoff / (1 - cutoff) * float(sigma_check.max())
    grid = np.linspace(1e-9, omega_upper, 500)
    assert cure(float(implied[0])) <= min(cure(g) for g in grid) + 1e-6


def test_unipas_get_lambda_and_get_omega_together(toy) -> None:
    estimates, lam, omegas = get_uni_pas_estimators(
        toy, get_lambda=True, get_omega=True
    )
    assert np.isscalar(lam) or np.asarray(lam).ndim == 0
    assert np.asarray(omegas).shape == (toy.M,)
    np.testing.assert_allclose(
        estimates, np.asarray(get_uni_pas_estimators(toy), dtype=float)
    )


# ---------------------------------------------------------------------------
# E.4: pseudo ground-truth uses all labels
# ---------------------------------------------------------------------------

def _dataset_available(key: str, sub: str) -> bool:
    from pas.config import DATA_PATHS

    return Path(DATA_PATHS[key][sub]).exists()


@pytest.mark.parametrize(
    "factory, key, sub",
    [
        ("amazon", "amazon", "tuned"),
        ("galaxy", "galaxy", "predictions"),
    ],
)
def test_pseudo_ground_truth_uses_all_labels(factory, key, sub) -> None:
    if not _dataset_available(key, sub):
        pytest.skip(f"{factory} data not present")

    if factory == "amazon":
        from pas.datasets.amazon_review import AmazonReviewDataset

        data = AmazonReviewDataset(tuned=True)
    else:
        from pas.datasets.galaxy_zoo import GalaxyZooDataset

        data = GalaxyZooDataset()

    expected = np.array([
        np.mean(np.concatenate([data.y_labelled[j], data.y_unlabelled[j]]))
        for j in range(data.M)
    ])
    np.testing.assert_allclose(data.true_theta, expected)

    # The estimand is a property of the corpus, so it must not move when the
    # labelled/unlabelled split is redrawn.
    first = np.asarray(data.true_theta, dtype=float).copy()
    data.reload_data(split_seed=first.size + 7)
    np.testing.assert_allclose(data.true_theta, first)
