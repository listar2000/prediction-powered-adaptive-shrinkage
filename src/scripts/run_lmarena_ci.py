"""LMArena CI benchmark — run experiment and save raw per-trial results.

Saves three files in `/tmp/lmarena_ci_results/` (so the outputs are not tracked
by git):
    raw.csv      -- one row per (alpha, trial), columns are
                    {method}_coverage and {method}_width for every method
    summary.csv  -- mean/SE per (alpha, method)
    meta.json    -- run config (trials, num_workers, methods, etc.)

The plot and the appendix table are produced by separate scripts that load
`raw.csv`, so plot/table iteration does not require re-running the
experiment.

Usage (from repo root):

    MOSEKLM_LICENSE_FILE=/path/to/mosek.lic \
        uv run python src/scripts/run_lmarena_ci.py \
            --trials 200 --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pas.datasets.lmarena import LMArenaDataset
from pas.experiments import run_ci_benchmark
from pas.intervals import CORE_CI_METHODS


# Methods to run. Superset of plot methods and table methods. The current
# focus is the PT-based rebiased estimators, so we run a slim set:
#
#   - Classic             : mle_ci
#   - Prediction Mean     : pred_mean_ci
#   - PPI                 : ppi_ci
#   - PT                  : pt_ci  (power-tuned PPI baseline)
#   - Robust EB           : robust_eb_ci (Armstrong et al., applied to PT)
#   - PT-Param  (ours)    : eb_pt_ci
#   - PT-NPMLE  (ours)    : eb_npmle_pt_ci
#
# The earlier PPI- / UniPT-flavored EB methods are kept in CORE_CI_METHODS but
# commented out below so they do not eat compute on every run.
RUN_METHODS = [
    "mle_ci",            # Classic
    "pred_mean_ci",      # Prediction Mean
    "ppi_ci",            # Vanilla PPI
    "pt_ci",             # Power-Tuned PPI baseline
    "robust_eb_ci",      # Armstrong et al. robust EBCI on unbiased PT
    "double_shrinkage_mle_ci",  # Rosenman et al. (2023), robust EBCI
    "eb_pt_ci",          # PT-Param  (ours)
    "eb_npmle_pt_ci",    # PT-NPMLE  (ours)
    # Other double-shrinkage hyperparameter variants / rebiased flavours
    # kept for ablations; uncomment to enable:
    # "double_shrinkage_mm1_ci",
    # "double_shrinkage_mm2_ci",
    # "double_shrinkage_ure_ci",
    # "eb_ppi_ci",             # PPI-Param   (older PPI-flavored variant)
    # "eb_npmle_ppi_ci",       # PPI-NPMLE
    # "eb_unipt_ppi_ci",       # UniPT-Param
    # "eb_npmle_unipt_ppi_ci", # UniPT-NPMLE
]


def _summarize(raw: pd.DataFrame, methods) -> pd.DataFrame:
    records = []
    for alpha, group in raw.groupby("alpha"):
        for method in methods:
            cov = group[f"{method}_coverage"].astype(float)
            wid = group[f"{method}_width"].astype(float)
            records.append({
                "alpha": float(alpha),
                "method": method,
                "mean_coverage": cov.mean(),
                "se_coverage": cov.sem(),
                "mean_width": wid.mean(),
                "se_width": wid.sem(),
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.01, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--train-test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes (default 4).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/tmp/lmarena_ci_results"))
    parser.add_argument("--csv", type=Path, default=None,
                        help="LMArena cleaned CSV; defaults to DATA_PATHS. "
                             "Use clean_summary_v2.csv for the Bradley-Terry "
                             "probability artifact.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = (
        LMArenaDataset(file_path=args.csv, train_test_split=args.train_test_split)
        if args.csv is not None
        else LMArenaDataset(train_test_split=args.train_test_split)
    )
    print(f"LMArena source: {dataset.file_path}")
    print(f"LMArena: M={dataset.M} problems, "
          f"n=[{dataset.ns.min()},{dataset.ns.max()}], "
          f"N=[{dataset.Ns.min()},{dataset.Ns.max()}]")

    missing = [k for k in RUN_METHODS if k not in CORE_CI_METHODS]
    if missing:
        raise RuntimeError(
            f"Missing CI methods (install npmle extra?): {missing}. "
            "Run `uv sync --extra npmle` and set MOSEKLM_LICENSE_FILE."
        )
    ci_methods = {k: CORE_CI_METHODS[k] for k in RUN_METHODS}
    ci_kwargs = {
        "pt_ci": {"share_var": False},
        # Match ebci(Y ~ 1, se=se, weights=1/se^2,
        #            fs_correction="PMT") using PT as the unbiased Y.
        #
        # `cv_mode="exact"` solves cva() per task exactly as ebci() does. The
        # packaged lookup tables round m2 (and kappa) *up*, which would widen
        # these two competitor baselines by ~1% on average (up to ~2.4% for
        # robust EB) relative to their true intervals -- a systematic bias in
        # our own favour. Exact mode costs ~12s per replicate, which is small
        # next to the NPMLE method, so we pay it.
        "robust_eb_ci": {
            "base_estimator": "pt",
            "share_var": False,
            "weights": "inverse_variance",
            "shrink_to": "grand_mean",
            "fs_correction": "PMT",
            "cv_mode": "exact",
        },
        # The double-shrinkage prior shrinks the target toward a fixed center.
        # For a pairwise win probability, 0.5 is the natural no-preference
        # center.
        "double_shrinkage_mle_ci": {
            "center": 0.5,
            "finite_sample_correction": "pmt",
            "cv_mode": "exact",
        },
    }

    raw_frames = []
    for alpha in args.alphas:
        print(f"\n=== alpha = {alpha} ===")
        ci_results = run_ci_benchmark(
            dataset,
            trials=args.trials,
            alpha=alpha,
            summary=False,
            ci_methods=ci_methods,
            ci_kwargs=ci_kwargs,
            num_workers=args.num_workers,
            verbose=True,
        )
        ci_results = ci_results.reset_index().rename(columns={"index": "trial"})
        ci_results.insert(0, "alpha", float(alpha))
        raw_frames.append(ci_results)

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(args.out_dir / "raw.csv", index=False)
    print(f"\nSaved raw -> {args.out_dir / 'raw.csv'} ({len(raw)} rows)")

    summary = _summarize(raw, RUN_METHODS)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(f"Saved summary -> {args.out_dir / 'summary.csv'}")
    print(summary.round(4).to_string(index=False))

    meta = {
        "csv": str(dataset.file_path),
        "trials": args.trials,
        "alphas": list(args.alphas),
        "train_test_split": args.train_test_split,
        "num_workers": args.num_workers,
        "methods": RUN_METHODS,
        "ci_kwargs": ci_kwargs,
    }
    with open(args.out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved meta -> {args.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
