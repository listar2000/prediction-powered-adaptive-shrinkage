"""Generate the packaged finite-kurtosis robust-EBCI lookup table.

The table contains exact ``ebci::cva`` translations on a compact grid covering
the range used by the PAS rebuttal experiments.  Runtime lookup rounds both
``m2`` and ``kappa`` upward, so the resulting critical value is conservative.
Values outside the grid are computed by the exact scalar solver.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pas.intervals.robust_eb import robust_eb_critical_value  # noqa: E402


def _one_curve(job: tuple[float, float, np.ndarray]) -> tuple[float, float, np.ndarray]:
    alpha, kappa, m2_grid = job
    values = np.asarray(
        [robust_eb_critical_value(float(m2), alpha, kappa) for m2 in m2_grid],
        dtype=float,
    )
    return alpha, kappa, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "src"
        / "pas"
        / "intervals"
        / "data"
        / "robust_eb_finite_cv_table.npz",
    )
    args = parser.parse_args()

    alphas = np.asarray([0.01, 0.05, 0.10, 0.20, 0.30], dtype=float)
    m2_grid = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.001, 0.005, 0.01, 0.02]),
                np.arange(0.05, 1.0001, 0.05),
                np.asarray([1.25, 1.5, 2.0]),
            ]
        )
    )
    kappa_grid = np.asarray(
        [1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 8.0, 10.0],
        dtype=float,
    )
    jobs = [
        (float(alpha), float(kappa), m2_grid)
        for alpha in alphas
        for kappa in kappa_grid
    ]
    table = np.empty((alphas.size, kappa_grid.size, m2_grid.size), dtype=float)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for alpha, kappa, values in executor.map(_one_curve, jobs):
            alpha_index = int(np.flatnonzero(alphas == alpha)[0])
            kappa_index = int(np.flatnonzero(kappa_grid == kappa)[0])
            table[alpha_index, kappa_index] = values

    # Enforce the theoretical monotonicity numerically.  This also ensures that
    # upward grid rounding is conservative even at floating-point tolerance.
    table = np.maximum.accumulate(table, axis=1)
    table = np.maximum.accumulate(table, axis=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        m2_grid=m2_grid,
        kappa_grid=kappa_grid,
        alphas=alphas,
        critical_values=table,
    )
    print(
        f"wrote {args.out}: shape={table.shape}, "
        f"m2=[{m2_grid[0]}, {m2_grid[-1]}], "
        f"kappa=[{kappa_grid[0]}, {kappa_grid[-1]}]"
    )


if __name__ == "__main__":
    main()
