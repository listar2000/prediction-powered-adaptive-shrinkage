"""Critical values for robust empirical-Bayes confidence intervals.

This module is a Python port of the critical-value calculations in the
MIT-licensed R package :mod:`ebci` by Michal Kolesar.  It implements the
Armstrong--Kolesar--Plagborg-Moller (2022) critical value

    cva_alpha(m2, kappa),

where ``m2`` bounds the second moment of the normalized conditional bias and
``kappa`` bounds its kurtosis.  The previous PAS implementation covered only
``kappa = infinity``.  The finite-kurtosis branch below follows ``R/cv.R`` in
the official repository so that PAS can reproduce the default ``ebci()``
workflow, which estimates a finite ``kappa`` from the data.

For the confidence levels used by the paper's experiments,
:func:`robust_eb_critical_values` provides packaged lookup tables.  The
finite-kurtosis table covers the range encountered in the rebuttal
experiments; lookup rounds ``m2`` and ``kappa`` upward, while values outside
the table fall back to the exact scalar solver.  ``mode="exact"`` always uses
the source-equivalent solver.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Union
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr
from scipy.stats import ncx2, norm

_TOL = 1e-12
_TABLE_NAME = "robust_eb_cv_table.npz"
_FINITE_TABLE_NAME = "robust_eb_finite_cv_table.npz"
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _normal_pdf(x: float) -> float:
    """Fast scalar standard-normal density."""
    x = float(x)
    return float(_INV_SQRT_2PI * np.exp(-0.5 * x * x))


def _rejection_probability(t: float, chi: float) -> float:
    """Noncoverage probability at squared normalized bias ``t``."""
    t = max(float(t), 0.0)
    root_t = np.sqrt(t)
    # Matches ebci's stable extreme-tail shortcut.
    if root_t - chi > 5.0:
        return 1.0
    return float(ndtr(-root_t - chi) + ndtr(root_t - chi))


def _rejection_derivative(t: float, chi: float) -> float:
    """First derivative of :func:`_rejection_probability` in ``t``."""
    t = max(float(t), 0.0)
    if t >= 1e-8:
        root_t = np.sqrt(t)
        return float(
            (_normal_pdf(root_t - chi) - _normal_pdf(root_t + chi))
            / (2.0 * root_t)
        )
    # L'Hopital limit used by ebci.
    return float(chi * _normal_pdf(chi))


def _rejection_second_derivative(t: float, chi: float) -> float:
    """Second derivative of the rejection probability in ``t``."""
    t = max(float(t), 0.0)
    if t >= 2e-6:
        root_t = np.sqrt(t)
        numerator = (
            _normal_pdf(root_t + chi) * (chi * root_t + t + 1.0)
            + _normal_pdf(root_t - chi) * (chi * root_t - t - 1.0)
        )
        return float(numerator / (4.0 * t ** 1.5))
    return float(_normal_pdf(chi) * chi * (chi**2 - 3.0) / 6.0)


def _rejection_third_derivative(t: float, chi: float) -> float:
    """Third derivative of the rejection probability in ``t``."""
    t = max(float(t), 0.0)
    if t >= 2e-4:
        root_t = np.sqrt(t)
        first = _normal_pdf(chi - root_t) * (
            t**2
            - 2.0 * chi * t ** 1.5
            + (2.0 + chi**2) * t
            - 3.0 * chi * root_t
            + 3.0
        )
        second = _normal_pdf(chi + root_t) * (
            t**2
            + 2.0 * chi * t ** 1.5
            + (2.0 + chi**2) * t
            + 3.0 * chi * root_t
            + 3.0
        )
        return float((first - second) / (8.0 * t ** 2.5))
    return float(
        _normal_pdf(chi)
        * (chi**5 - 10.0 * chi**3 + 15.0 * chi)
        / 60.0
    )


@lru_cache(maxsize=8192)
def _tangent_and_inflection(chi: float) -> tuple[float, float]:
    """Return ``(t0, inflection)`` from ``ebci::rt0``."""
    chi = float(chi)
    chi2 = chi * chi
    if chi2 < 3.0:
        return 0.0, 0.0

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
            inflection = float(
                brentq(
                    lambda t: _rejection_second_derivative(t, chi),
                    lo_ip,
                    hi_ip,
                    xtol=_TOL,
                    rtol=4 * np.finfo(float).eps,
                )
            )
        except ValueError:
            # This mirrors the official large-chi numerical shortcut.
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
        t0 = float(
            brentq(f0, lo, up, xtol=_TOL, rtol=4 * np.finfo(float).eps)
        )
    else:
        if f_lo > _TOL:
            warnings.warn(
                "Numerical warning while solving robust-EBCI tangent point "
                f"at chi={chi}",
                RuntimeWarning,
                stacklevel=2,
            )
        t0 = float(lo)
    return t0, float(inflection)


@lru_cache(maxsize=8192)
def _tangent_point(chi: float) -> float:
    """Return the contact point defining the concave envelope ``rho0``."""
    return _tangent_and_inflection(float(chi))[0]


def _rho_second_moment(m2: float, chi: float) -> float:
    """Worst-case noncoverage under only a second-moment bound."""
    m2 = max(float(m2), 0.0)
    t0 = _tangent_point(float(chi))
    if m2 >= t0:
        return _rejection_probability(m2, chi)
    return float(
        _rejection_probability(t0, chi)
        + (m2 - t0) * _rejection_derivative(t0, chi)
    )


def _delta(x: float, x0: float, chi: float) -> float:
    """Second-order secant remainder ``delta(x; x0)`` from ``ebci``."""
    x = max(float(x), 0.0)
    x0 = max(float(x0), 0.0)
    difference = x - x0
    if abs(difference) < 1e-4:
        return float(_rejection_second_derivative(x0, chi) / 2.0)
    return float(
        (
            _rejection_probability(x, chi)
            - _rejection_probability(x0, chi)
            - _rejection_derivative(x0, chi) * difference
        )
        / difference**2
    )


def _delta_derivative(x: float, x0: float, chi: float) -> float:
    """Derivative of ``delta(x; x0)`` in its first argument."""
    x = max(float(x), 0.0)
    x0 = max(float(x0), 0.0)
    difference = x - x0
    if abs(difference) < 1e-3:
        return float(_rejection_third_derivative(x0, chi) / 6.0)
    return float(
        (
            _rejection_derivative(x, chi)
            + _rejection_derivative(x0, chi)
            - 2.0
            * (
                _rejection_probability(x, chi)
                - _rejection_probability(x0, chi)
            )
            / difference
        )
        / difference**2
    )


def _bounded_minimum(function, lower: float, upper: float) -> tuple[float, float]:
    """Minimize on a closed interval, explicitly checking both endpoints."""
    lower = float(lower)
    upper = float(upper)
    if upper <= lower + _TOL:
        value = float(function(lower))
        return lower, value
    result = minimize_scalar(
        function,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": _TOL, "maxiter": 500},
    )
    candidates = [
        (lower, float(function(lower))),
        (upper, float(function(upper))),
    ]
    if result.success and np.isfinite(result.fun):
        candidates.append((float(result.x), float(result.fun)))
    return min(candidates, key=lambda item: item[1])


@lru_cache(maxsize=32768)
def _maximize_delta(x0: float, chi: float) -> tuple[float, float]:
    """Return ``(max delta, argmax)``; port of ``ebci::lam``."""
    x0 = max(float(x0), 0.0)
    chi = float(chi)
    t0, inflection = _tangent_and_inflection(chi)
    base = np.sort(np.asarray([inflection, t0], dtype=float))
    if x0 >= base[0]:
        xs = np.asarray([0.0, base[0]], dtype=float)
    else:
        xs = np.unique(np.asarray([0.0, x0, base[0], base[1]], dtype=float))
    xs = np.sort(xs)

    values = np.asarray([_delta(x, x0, chi) for x in xs])
    derivatives = np.asarray([_delta_derivative(x, x0, chi) for x in xs])
    opt0 = (float(values[0]), 0.0)

    if np.all(derivatives <= 0.0) and int(np.argmax(values)) == 0:
        return opt0

    signs = derivatives >= 0.0
    monotone_signs = np.all(np.diff(signs.astype(int)) <= 0)
    if monotone_signs and derivatives[-1] <= 0.0:
        negative = np.flatnonzero(~signs)
        first_negative = int(negative[0]) if negative.size else 0
        right = max(first_negative, 1)
        right = min(right, xs.size - 1)
        interval = (float(xs[right - 1]), float(xs[right]))
    elif np.min(np.abs(derivatives)) < 1e-6:
        best = int(np.argmax(values))
        left = max(best - 1, 0)
        right = min(best + 1, xs.size - 1)
        interval = (float(xs[left]), float(xs[right]))
    else:
        # The official implementation stops here.  A dense deterministic
        # fallback is safer inside a benchmark while preserving the same
        # objective and emits a warning so a discrepancy is visible.
        warnings.warn(
            "Unexpected derivative pattern while maximizing robust-EBCI "
            f"delta(x, x0={x0}, chi={chi}); using a dense bracket fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        grid = np.linspace(0.0, max(t0, inflection, x0, 1e-12), 257)
        grid_values = np.asarray([_delta(x, x0, chi) for x in grid])
        best = int(np.argmax(grid_values))
        left = max(best - 1, 0)
        right = min(best + 1, grid.size - 1)
        interval = (float(grid[left]), float(grid[right]))

    argmin, objective = _bounded_minimum(
        lambda x: -_delta(x, x0, chi), interval[0], interval[1]
    )
    optimum = -objective
    if optimum > opt0[0]:
        return float(optimum), float(argmin)
    if optimum > opt0[0] - 1e3 * _TOL:
        return opt0
    warnings.warn(
        "Potential numerical error while maximizing robust-EBCI delta at "
        f"x0={x0}, chi={chi}; using the endpoint solution.",
        RuntimeWarning,
        stacklevel=2,
    )
    return opt0


def _rho_moment_constraints(m2: float, kappa: float, chi: float) -> float:
    """Worst-case noncoverage under second- and fourth-moment constraints."""
    m2 = max(float(m2), 0.0)
    kappa = float(kappa)
    chi = float(chi)
    t0 = _tangent_point(chi)
    r0 = _rho_second_moment(m2, chi)

    if kappa == 1.0:
        return _rejection_probability(m2, chi)
    if m2 >= t0:
        return r0
    if np.isinf(kappa) or m2 * kappa >= t0:
        return r0

    tbar = _maximize_delta(0.0, chi)[1]

    def lammax(x0: float) -> float:
        if x0 >= tbar:
            return _delta(0.0, x0, chi)
        return max(_maximize_delta(float(x0), chi)[0], 0.0)

    def objective(x0: float) -> float:
        x0 = float(x0)
        return float(
            _rejection_probability(x0, chi)
            + _rejection_derivative(x0, chi) * (m2 - x0)
            + lammax(x0) * (kappa * m2**2 - 2.0 * x0 * m2 + x0**2)
        )

    below = _bounded_minimum(objective, 0.0, tbar)
    above = _bounded_minimum(objective, tbar, t0)
    value = min(below[1], above[1])
    return float(np.clip(value, 0.0, 1.0))


def _bounded_bias_critical_value(B: float, alpha: float) -> float:
    """Critical value when absolute normalized bias is bounded by ``B``."""
    B = max(float(B), 0.0)
    if B < 10.0:
        return float(np.sqrt(ncx2.ppf(1.0 - alpha, df=1, nc=B * B)))
    return float(B + norm.ppf(1.0 - alpha))


@lru_cache(maxsize=65536)
def robust_eb_critical_value(
    m2: float,
    alpha: float = 0.05,
    kappa: float = np.inf,
) -> float:
    """Exact Armstrong--Kolesar--Plagborg-Moller robust-EBCI critical value.

    Parameters
    ----------
    m2:
        Bound on the second moment of the normalized conditional bias.
    alpha:
        Two-sided noncoverage level.
    kappa:
        Bound on kurtosis.  ``np.inf`` uses only the second-moment bound;
        finite values reproduce the default critical values used by
        ``ebci::ebci`` after its moment estimation step.
    """
    m2 = float(m2)
    alpha = float(alpha)
    kappa = float(kappa)
    if np.isnan(m2) or m2 < 0.0:
        raise ValueError("m2 must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if np.isnan(kappa) or kappa < 1.0:
        raise ValueError("kappa must be at least one")
    if np.isinf(m2):
        return float("nan")
    if m2 == 0.0 or kappa == 1.0:
        return _bounded_bias_critical_value(np.sqrt(m2), alpha)

    # For extremely large m2 the finite-kappa dual becomes numerically
    # indistinguishable from the second-moment-only problem, as in ebci::cva.
    if (1.0 / m2) < _TOL and not np.isinf(kappa):
        warnings.warn(
            "m2 is too large for a reliable finite-kappa calculation; "
            "assuming the kurtosis constraint is non-binding.",
            RuntimeWarning,
            stacklevel=2,
        )
        kappa = np.inf

    lo = _bounded_bias_critical_value(np.sqrt(m2), alpha) - 0.01
    up = np.sqrt((1.0 + m2) / alpha)

    objective_up = _rho_second_moment(m2, up) - alpha
    if abs(objective_up) < 9e-6:
        cv_inf = float(up)
    else:
        objective_lo = _rho_second_moment(m2, lo) - alpha
        if objective_lo < 0.0:
            lo = max(norm.ppf(1.0 - alpha / 2.0), lo - 0.1)
            objective_lo = _rho_second_moment(m2, lo) - alpha
        if objective_lo * objective_up > 0.0:
            raise RuntimeError(
                "failed to bracket robust-EBCI critical value "
                f"(m2={m2}, alpha={alpha}, f(lo)={objective_lo}, "
                f"f(up)={objective_up})"
            )
        cv_inf = float(
            brentq(
                lambda chi: _rho_second_moment(m2, chi) - alpha,
                lo,
                up,
                xtol=_TOL,
                rtol=4 * np.finfo(float).eps,
            )
        )

    if np.isinf(kappa):
        return cv_inf

    finite_at_upper = _rho_moment_constraints(m2, kappa, cv_inf) - alpha
    if finite_at_upper < -1e-5:
        finite_at_lower = _rho_moment_constraints(m2, kappa, lo) - alpha
        if finite_at_lower < 0.0:
            # Very small m2 may make the theoretical lower bound equal to the
            # solution up to floating-point error.
            return _bounded_bias_critical_value(np.sqrt(m2), alpha)
        return float(
            brentq(
                lambda chi: _rho_moment_constraints(m2, kappa, chi) - alpha,
                lo,
                cv_inf,
                xtol=_TOL,
                rtol=4 * np.finfo(float).eps,
            )
        )
    return cv_inf


@lru_cache(maxsize=1)
def _load_lookup_table() -> tuple:
    path = files("pas.intervals.data").joinpath(_TABLE_NAME)
    with path.open("rb") as fh:
        table = np.load(fh)
        m2_grid = np.asarray(table["m2_grid"], dtype=float)
        alphas = np.asarray(table["alphas"], dtype=float)
        critical_values = np.asarray(table["critical_values"], dtype=float)
    return m2_grid, alphas, critical_values

@lru_cache(maxsize=1)
def _load_finite_lookup_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the conservative finite-kurtosis critical-value grid."""
    path = files("pas.intervals.data").joinpath(_FINITE_TABLE_NAME)
    with path.open("rb") as fh:
        table = np.load(fh)
        m2_grid = np.asarray(table["m2_grid"], dtype=float)
        kappa_grid = np.asarray(table["kappa_grid"], dtype=float)
        alphas = np.asarray(table["alphas"], dtype=float)
        critical_values = np.asarray(table["critical_values"], dtype=float)
    return m2_grid, kappa_grid, alphas, critical_values


def robust_eb_critical_values(
    m2: Union[float, np.ndarray],
    alpha: float = 0.05,
    kappa: float = np.inf,
    *,
    mode: str = "lookup",
) -> np.ndarray:
    """Vectorized robust-EBCI critical values.

    ``mode='lookup'`` rounds ``m2`` and, for finite kurtosis, ``kappa``
    upward on packaged exact grids.  Monotonicity of the robust critical value
    makes this conservative.  Values outside the finite grid fall back to the
    exact scalar solver.  ``mode='exact'`` always calls
    :func:`robust_eb_critical_value` elementwise.
    """
    arr = np.asarray(m2, dtype=float)
    kappa = float(kappa)
    if np.any(np.isnan(arr)) or np.any(arr < 0.0):
        raise ValueError("all m2 values must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if np.isnan(kappa) or kappa < 1.0:
        raise ValueError("kappa must be at least one")

    flat = arr.reshape(-1)
    if mode not in {"lookup", "exact"}:
        raise ValueError("mode must be 'lookup' or 'exact'")
    if mode == "exact":
        out = np.array(
            [
                robust_eb_critical_value(
                    float(x), alpha=float(alpha), kappa=kappa
                )
                for x in flat
            ],
            dtype=float,
        )
        return out.reshape(arr.shape)

    if not np.isinf(kappa):
        m2_grid, kappa_grid, alphas, table = _load_finite_lookup_table()
        alpha_idx = np.flatnonzero(
            np.isclose(alphas, alpha, rtol=0.0, atol=1e-12)
        )
        if alpha_idx.size:
            kappa_index = int(np.searchsorted(kappa_grid, kappa, side="left"))
            if kappa_index < kappa_grid.size:
                m2_indices = np.searchsorted(m2_grid, flat, side="left")
                inside = m2_indices < m2_grid.size
                out = np.empty_like(flat)
                out[inside] = table[
                    int(alpha_idx[0]), kappa_index, m2_indices[inside]
                ]
                if np.any(~inside):
                    out[~inside] = np.array(
                        [
                            robust_eb_critical_value(
                                float(x), alpha=float(alpha), kappa=kappa
                            )
                            for x in flat[~inside]
                        ],
                        dtype=float,
                    )
                return out.reshape(arr.shape)

        # The packaged finite table deliberately covers the common experiment
        # range.  Outside that range we retain exact source-method behavior.
        out = np.array(
            [
                robust_eb_critical_value(
                    float(x), alpha=float(alpha), kappa=kappa
                )
                for x in flat
            ],
            dtype=float,
        )
        return out.reshape(arr.shape)

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
            [
                robust_eb_critical_value(float(x), float(alpha), np.inf)
                for x in flat
            ],
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
