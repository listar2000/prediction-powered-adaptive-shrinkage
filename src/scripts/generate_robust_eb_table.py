"""Generate the packaged robust-EBCI critical-value lookup table.

This script is not needed for normal use. It recomputes the table from the
exact scalar solver and is included so the numerical artifact is reproducible.

Usage (from repo root):

    uv run python src/scripts/generate_robust_eb_table.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np

from pas.intervals.robust_eb import robust_eb_critical_value

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "src/pas/intervals/data/robust_eb_cv_table.npz"


def _solve_one(args):
    alpha, m2 = args
    return robust_eb_critical_value(float(m2), float(alpha))


def main() -> None:
    alphas = np.array([0.01, 0.05, 0.10, 0.20, 0.30], dtype=float)
    # Dense on a log scale over the practically relevant range. Values above
    # 1e4 use the valid Chebyshev upper bound at runtime. At 2000 points the
    # adjacent-grid ratio is ~1.4%, so the conservative round-up to the next
    # grid point inflates interval widths by well under 1%.
    m2_grid = np.concatenate(
        [np.array([0.0]), np.geomspace(1e-8, 1e4, num=2000)]
    )
    jobs = [(float(alpha), float(m2)) for alpha in alphas for m2 in m2_grid]
    workers = min(16, os.cpu_count() or 1)
    print(f"solving {len(jobs)} values with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(_solve_one, jobs, chunksize=2))
    table = np.asarray(values, dtype=float).reshape(alphas.size, m2_grid.size)
    table = np.maximum.accumulate(table, axis=1)

    np.savez_compressed(
        OUT_PATH,
        m2_grid=m2_grid,
        alphas=alphas,
        critical_values=table,
    )
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
