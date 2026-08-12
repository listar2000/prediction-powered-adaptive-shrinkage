"""Validation of the synthetic model's closed-form second moments (Appendix E.1).

Model: X ~ N(eta, psi^2), Y | X ~ N(2 eta X - eta^2, c). The three moments the
estimators consume are sigma^2 = Var(Y), tau^2 = Var(f(X)), gamma = Cov(f(X), Y)
= 2 eta Cov(f(X), X).

Ground truth here is adaptive quadrature of the exact Gaussian integrals, so
these tests pin the *formulas*, independently of any sampling. The camera-ready
E.1 prints a wrong gamma_j for f(x) = |x| (a Gaussian-pdf factor where
(2 Phi(eta/psi) - 1) belongs) and a doubly mangled mu_j line; the corrected
forms implemented in `GaussianSyntheticDataset` are what is tested.
"""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from pas.datasets.synthetic_model import GaussianSyntheticDataset

PSI, C = 0.1, 0.05
ETAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, -0.05, -0.5, -1.0]


def quad_moments(f, eta):
    """(Var Y, Var f(X), Cov(f(X), Y)) by adaptive quadrature."""
    pdf = lambda x: norm.pdf(x, eta, PSI)
    lo, hi = eta - 12 * PSI, eta + 12 * PSI
    ef = integrate.quad(lambda x: f(x) * pdf(x), lo, hi)[0]
    ef2 = integrate.quad(lambda x: f(x) ** 2 * pdf(x), lo, hi)[0]
    exf = integrate.quad(lambda x: x * f(x) * pdf(x), lo, hi)[0]
    sigma2 = 4 * eta ** 2 * PSI ** 2 + C
    return sigma2, ef2 - ef ** 2, 2 * eta * (exf - eta * ef)


def closed_forms(good_f, eta):
    """The formulas implemented in `GaussianSyntheticDataset.load_data`."""
    k = 4 * eta ** 2 * PSI ** 2
    sigma2 = k + C
    if good_f:
        return sigma2, k + 2 * PSI ** 4, k
    a = eta / PSI
    phi_cdf = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
    mean_abs = PSI * math.sqrt(2.0 / math.pi) * math.exp(-a ** 2 / 2.0) \
        + eta * (2.0 * phi_cdf - 1.0)
    return sigma2, eta ** 2 + PSI ** 2 - mean_abs ** 2, \
        2.0 * eta * PSI ** 2 * (2.0 * phi_cdf - 1.0)


class _AnalyticSynthetic(GaussianSyntheticDataset):
    DEBUG_FLAG = False


@pytest.mark.parametrize("good_f,f", [(True, lambda x: x ** 2), (False, np.abs)])
@pytest.mark.parametrize("eta", ETAS)
def test_closed_forms_match_quadrature(good_f, f, eta):
    expected = quad_moments(f, eta)
    got = closed_forms(good_f, eta)
    for e, g in zip(expected, got):
        assert g == pytest.approx(e, rel=1e-6, abs=1e-12)


def test_paper_printed_gamma_for_abs_is_wrong():
    """The camera-ready gamma_j = 2 eta psi^2 sqrt(2/pi) e^{-eta^2/(2 psi^2)}
    disagrees with quadrature away from eta = 0 -- guard against 'fixing' the
    code back to the printed expression."""
    eta = 0.5
    printed = 2 * eta * PSI ** 2 * math.sqrt(2 / math.pi) \
        * math.exp(-eta ** 2 / (2 * PSI ** 2))
    truth = quad_moments(np.abs, eta)[2]
    assert truth == pytest.approx(0.01, rel=1e-4)
    assert abs(printed - truth) / truth > 0.99


@pytest.mark.parametrize("good_f", [True, False])
def test_analytic_dataset_stores_closed_forms(good_f):
    d = _AnalyticSynthetic(good_f=good_f, M=40, split_seed=7, has_true_vars=True)
    for j in range(d.M):
        s2, t2, g = closed_forms(good_f, d.mu_x_s[j])
        assert d.true_y_vars[j] == pytest.approx(s2, rel=1e-12)
        assert d.true_vars[j] == pytest.approx(t2, rel=1e-12)
        assert d.true_covs[j] == pytest.approx(g, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("good_f", [True, False])
def test_default_monte_carlo_path_targets_the_same_moments(good_f):
    """DEBUG_FLAG defaults to True (50k-draw MC), and those MC moments must be
    noisy estimates *of the closed forms* -- tight on the variances, absolute
    tolerance on gamma, whose true value passes through 0 at eta = 0. The
    50k-draw MC gives gamma a sampling SE of ~1e-4, so the absolute tolerance
    is ~6 SE."""
    assert GaussianSyntheticDataset.DEBUG_FLAG is True
    d = GaussianSyntheticDataset(good_f=good_f, M=40, split_seed=7,
                                 has_true_vars=True)
    for j in range(d.M):
        s2, t2, g = closed_forms(good_f, d.mu_x_s[j])
        assert d.true_y_vars[j] == pytest.approx(s2, rel=0.05)
        assert d.true_vars[j] == pytest.approx(t2, rel=0.05)
        assert d.true_covs[j] == pytest.approx(g, rel=0.05, abs=6e-4)


def test_analytic_path_draws_only_the_data():
    """With DEBUG_FLAG=False only n_j + N_j points are drawn per problem."""
    d = _AnalyticSynthetic(good_f=False, M=10, split_seed=7, has_true_vars=True)
    for j in range(d.M):
        assert len(d.pred_labelled[j]) + len(d.pred_unlabelled[j]) \
            == d.ns[j] + d.Ns[j]
