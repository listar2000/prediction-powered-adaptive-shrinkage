"""Critical values for robust empirical-Bayes confidence intervals.

Adapted from the MIT-licensed ``ebci`` implementation by Michal Kolesar;
see ``THIRD_PARTY_NOTICES.md`` for attribution.

This is a small Python port of the ``kappa = infinity`` (second-moment
constraint only) branch of ``ebci::cva`` from Armstrong, Kolesar, and
Plagborg-Moller's robust EBCI implementation.  The exact scalar solver is
provided for validation and uncommon confidence levels.  For the confidence
levels used by the paper's experiments, :func:`robust_eb_critical_values`
uses a packaged lookup table and rounds *up* to the next grid point, making
the interpolation conservative relative to the tabulated exact values.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Union
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

_TOL = 1e-12
_TABLE_NAME = "robust_eb_cv_table.npz"


def _rejection_probability(t: float, chi: float) -> float:
    """Noncoverage probability for squared normalized bias ``t``."""
    t = max(float(t), 0.0)
    root_t = np.sqrt(t)
    # Matches ebci's stable extreme-tail shortcut.
    if root_t - chi > 5.0:
        return 1.0
    return float(norm.cdf(-root_t - chi) + norm.cdf(root_t - chi))


def _rejection_derivative(t: float, chi: float) -> float:
    t = max(float(t), 0.0)
    if t >= 1e-8:
        root_t = np.sqrt(t)
        return float(
            (norm.pdf(root_t - chi) - norm.pdf(root_t + chi))
            / (2.0 * root_t)
        )
    return float(chi * norm.pdf(chi))


def _rejection_second_derivative(t: float, chi: float) -> float:
    t = max(float(t), 0.0)
    if t >= 2e-6:
        root_t = np.sqrt(t)
        numerator = (
            norm.pdf(root_t + chi) * (chi * root_t + t + 1.0)
            + norm.pdf(root_t - chi) * (chi * root_t - t - 1.0)
        )
        return float(numerator / (4.0 * t ** 1.5))
    return float(norm.pdf(chi) * chi * (chi**2 - 3.0) / 6.0)


@lru_cache(maxsize=8192)
def _tangent_point(chi: float) -> float:
    """Return the contact point defining the concave envelope ``rho``."""
    chi = float(chi)
    chi2 = chi * chi
    if chi2 < 3.0:
        return 0.0

    def f0(t: float) -> float:
        return (
            _rejection_probability(t, chi)
            - t * _rejection_derivative(t, chi)
            - _rejection_probability(0.0, chi)
        )

    midpoint = chi2 - 1.5
    if (
        abs(_rejection_second_derivative(midpoint, chi)) < _TOL
        or (chi2 - 3.0) == chi2
    ):
        inflection = midpoint
    else:
        lo_ip = max(chi2 - 3.0, 0.0)
        hi_ip = chi2
        try:
            inflection = brentq(
                lambda t: _rejection_second_derivative(t, chi),
                lo_ip,
                hi_ip,
                xtol=_TOL,
                rtol=4 * np.finfo(float).eps,
            )
        except ValueError:
            # Numerical fallback used only at extreme chi; the midpoint is the
            # asymptotically correct location and mirrors ebci's shortcut.
            inflection = midpoint

    lo = inflection
    up = max(2.0 * chi2, lo + 1.0)
    while f0(up) < 0.0:
        lo = up
        up *= 2.0
        if not np.isfinite(up):
            raise RuntimeError("failed to bracket robust-EBCI tangent point")

    f_lo = f0(lo)
    if f_lo < 0.0:
        return float(
            brentq(f0, lo, up, xtol=_TOL, rtol=4 * np.finfo(float).eps)
        )
    if f_lo > _TOL:
        warnings.warn(
            f"Numerical warning while solving robust-EBCI tangent point at chi={chi}",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(lo)


def _rho_second_moment(m2: float, chi: float) -> float:
    """Worst-case noncoverage under a second-moment bound ``m2``."""
    m2 = max(float(m2), 0.0)
    t0 = _tangent_point(float(chi))
    if m2 >= t0:
        return _rejection_probability(m2, chi)
    return float(
        _rejection_probability(t0, chi)
        + (m2 - t0) * _rejection_derivative(t0, chi)
    )


def _bounded_bias_critical_value(B: float, alpha: float) -> float:
    """Critical value when the absolute normalized bias is bounded by ``B``."""
    B = max(float(B), 0.0)
    if B < 10.0:
        return float(np.sqrt(ncx2.ppf(1.0 - alpha, df=1, nc=B * B)))
    return float(B + norm.ppf(1.0 - alpha))


@lru_cache(maxsize=32768)
def robust_eb_critical_value(m2: float, alpha: float = 0.05) -> float:
    """Exact robust-EBCI critical value under only a second-moment bound.

    Parameters
    ----------
    m2:
        Bound on the second moment of the normalized conditional bias.
    alpha:
        Two-sided noncoverage level.
    """
    m2 = float(m2)
    alpha = float(alpha)
    if not np.isfinite(m2) or m2 < 0.0:
        raise ValueError("m2 must be finite and non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if m2 == 0.0:
        return _bounded_bias_critical_value(0.0, alpha)

    lo = _bounded_bias_critical_value(np.sqrt(m2), alpha) - 0.01
    up = np.sqrt((1.0 + m2) / alpha)

    objective_up = _rho_second_moment(m2, up) - alpha
    if abs(objective_up) < 9e-6:
        return float(up)

    objective_lo = _rho_second_moment(m2, lo) - alpha
    if objective_lo < 0.0:
        # The theoretical lower bound should bracket the root.  Rounding in
        # noncentral-chi-square tails can rarely violate this by a few ulps.
        lo = max(norm.ppf(1.0 - alpha / 2.0), lo - 0.1)
        objective_lo = _rho_second_moment(m2, lo) - alpha
    if objective_lo * objective_up > 0.0:
        raise RuntimeError(
            "failed to bracket robust-EBCI critical value "
            f"(m2={m2}, alpha={alpha}, f(lo)={objective_lo}, f(up)={objective_up})"
        )
    return float(
        brentq(
            lambda chi: _rho_second_moment(m2, chi) - alpha,
            lo,
            up,
            xtol=_TOL,
            rtol=4 * np.finfo(float).eps,
        )
    )


@lru_cache(maxsize=1)
def _load_lookup_table() -> tuple:
    path = files("pas.intervals.data").joinpath(_TABLE_NAME)
    with path.open("rb") as fh:
        table = np.load(fh)
        m2_grid = np.asarray(table["m2_grid"], dtype=float)
        alphas = np.asarray(table["alphas"], dtype=float)
        critical_values = np.asarray(table["critical_values"], dtype=float)
    return m2_grid, alphas, critical_values


def robust_eb_critical_values(
    m2: Union[float, np.ndarray],
    alpha: float = 0.05,
    *,
    mode: str = "lookup",
) -> np.ndarray:
    """Vectorized robust-EBCI critical values.

    ``mode='lookup'`` is the benchmark default.  It uses an exact precomputed
    table for the paper's five alpha values and rounds each ``m2`` upward to
    the next grid point.  This is deliberately conservative and avoids solving
    hundreds of nested root-finding problems in every Monte Carlo replicate.
    ``mode='exact'`` calls :func:`robust_eb_critical_value` elementwise.
    """
    arr = np.asarray(m2, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("all m2 values must be finite and non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    flat = arr.reshape(-1)
    if mode == "exact":
        out = np.array(
            [robust_eb_critical_value(float(x), float(alpha)) for x in flat],
            dtype=float,
        )
        return out.reshape(arr.shape)
    if mode != "lookup":
        raise ValueError("mode must be 'lookup' or 'exact'")

    m2_grid, alphas, table = _load_lookup_table()
    alpha_idx = np.flatnonzero(np.isclose(alphas, alpha, rtol=0.0, atol=1e-12))
    if alpha_idx.size == 0:
        warnings.warn(
            "No packaged robust-EBCI lookup table for alpha="
            f"{alpha}; using the exact scalar solver.",
            RuntimeWarning,
            stacklevel=2,
        )
        out = np.array(
            [robust_eb_critical_value(float(x), float(alpha)) for x in flat],
            dtype=float,
        )
        return out.reshape(arr.shape)

    row = table[int(alpha_idx[0])]
    indices = np.searchsorted(m2_grid, flat, side="left")
    inside = indices < m2_grid.size
    out = np.empty_like(flat)
    out[inside] = row[indices[inside]]

    # Beyond the precomputed range, the Chebyshev bound used by ebci to
    # bracket its solver is a valid conservative critical value.
    if np.any(~inside):
        out[~inside] = np.sqrt((1.0 + flat[~inside]) / alpha)
    return out.reshape(arr.shape)


__all__ = [
    "robust_eb_critical_value",
    "robust_eb_critical_values",
]
