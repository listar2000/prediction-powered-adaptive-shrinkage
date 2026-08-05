"""Post-fix reproduction of Table 3 of the PAS paper (share_var off and on).

Re-runs the paper's real-data benchmark after the estimator/dataset corrections
described in ``README.md`` in this directory. Results are NOT comparable
row-for-row with the published table: the pseudo-ground-truth fix changes the
estimand, which rescales every MSE (see the README).


Nine estimator rows (Classical ... UniPAS) x three dataset columns (Amazon
``BERT-base``, Amazon ``BERT-tuned``, Galaxy), K = 200 replicates at the default
split seed -- the configuration described in Appendix E.4.

Each (dataset, share_var) cell is an independent job, so the six jobs run in
parallel subprocesses. Datasets are constructed inside the worker rather than
pickled across the process boundary.

``share_var`` reaches only the estimators whose signature accepts it (PT,
Shrink Classical, Shrink Avg, PAS). Classical / Prediction Avg / PPI have no
second-moment estimate to share, and UniPT / UniPAS deliberately do not take the
flag -- the paper notes that sharing variance is not meaningful once a single
global lambda is used. Those five rows are therefore identical across the two
runs, which doubles as a consistency check on the parallel harness.

Usage::

    uv run python src/scripts/run_pas_paper_tables.py
    uv run python src/scripts/run_pas_paper_tables.py --trials 5 --num-workers 2
"""
from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
HERE = Path(__file__).resolve().parent

# (paper row label, registry key)
ROWS = [
    ("Classical", "mle"),
    ("Prediction Avg", "pred_mean"),
    ("PPI", "ppi"),
    ("PT", "pt"),
    ("Shrink Classical", "shrinkage_only"),
    ("Shrink Avg", "shrinkage_mean"),
    ("PAS (ours)", "pas"),
    ("UniPT (ours)", "uni_pt"),
    ("UniPAS (ours)", "uni_pas"),
]

DATASET_LABELS = {
    "amazon_base": "Amazon (base f)",
    "amazon_tuned": "Amazon (tuned f)",
    "galaxy": "Galaxy",
}

# Published Table 3, as (mse_mean, mse_se, improved_mean, improved_se).
# `None` for the Classical baseline's "% Improved" cell.
PUBLISHED = {
    "amazon_base": {
        "mle": (24.305, 0.189, None, None),
        "pred_mean": (41.332, 0.050, 30.7, 0.2),
        "ppi": (11.063, 0.085, 62.4, 0.2),
        "pt": (10.633, 0.089, 70.3, 0.2),
        "shrinkage_only": (15.995, 0.121, 56.4, 0.3),
        "shrinkage_mean": (9.276, 0.078, 70.4, 0.2),
        "pas": (8.517, 0.071, 71.4, 0.2),
        "uni_pt": (10.272, 0.084, 70.0, 0.2),
        "uni_pas": (8.879, 0.073, 69.5, 0.2),
    },
    "amazon_tuned": {
        "mle": (24.305, 0.189, None, None),
        "pred_mean": (3.945, 0.011, 75.4, 0.2),
        "ppi": (7.565, 0.066, 70.4, 0.2),
        "pt": (6.289, 0.050, 76.0, 0.2),
        "shrinkage_only": (3.828, 0.039, 78.9, 0.2),
        "shrinkage_mean": (6.280, 0.058, 77.1, 0.2),
        "pas": (3.287, 0.024, 80.8, 0.2),
        "uni_pt": (6.489, 0.053, 76.2, 0.2),
        "uni_pas": (3.356, 0.031, 77.6, 0.2),
    },
    "galaxy": {
        "mle": (2.073, 0.028, None, None),
        "pred_mean": (7.195, 0.008, 17.0, 0.2),
        "ppi": (1.149, 0.017, 59.4, 0.3),
        "pt": (1.026, 0.015, 67.7, 0.3),
        "shrinkage_only": (1.522, 0.016, 48.8, 0.4),
        "shrinkage_mean": (0.976, 0.014, 68.9, 0.3),
        "pas": (0.893, 0.011, 67.3, 0.4),
        "uni_pt": (1.017, 0.015, 67.7, 0.3),
        "uni_pas": (0.909, 0.011, 66.7, 0.3),
    },
}


def build_dataset(key: str):
    """Construct a dataset by key. Called inside the worker process."""
    from pas.datasets.amazon_review import AmazonReviewDataset
    from pas.datasets.galaxy_zoo import GalaxyZooDataset

    if key == "amazon_base":
        return AmazonReviewDataset(tuned=False)
    if key == "amazon_tuned":
        return AmazonReviewDataset(tuned=True)
    if key == "galaxy":
        return GalaxyZooDataset()
    raise ValueError(f"unknown dataset {key!r}")


def _accepts_share_var(func) -> bool:
    try:
        return "share_var" in inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C functions
        return False


def run_cell(args) -> dict:
    """One (dataset, share_var) cell: K replicates over all nine estimators."""
    dataset_key, share_var, trials = args

    warnings.filterwarnings("ignore")
    from pas.estimators import ALL_ESTIMATORS
    from pas.experiments import run_benchmark

    # `mle` must come first: run_benchmark measures "% Improved" against it.
    estimators = {key: ALL_ESTIMATORS[key] for _, key in ROWS}
    estimator_kwargs = {
        key: {"share_var": share_var}
        for key in estimators
        if _accepts_share_var(estimators[key])
    }

    dataset = build_dataset(dataset_key)
    frame = run_benchmark(
        dataset,
        trials=trials,
        estimators=estimators,
        estimator_kwargs=estimator_kwargs,
        summary=False,
        verbose=False,
    )

    records = []
    for label, key in ROWS:
        mse = frame[key].astype(float)
        frac = frame[f"{key}_frac"].astype(float)
        records.append({
            "dataset": dataset_key,
            "share_var": share_var,
            "estimator": key,
            "label": label,
            "trials": int(len(frame)),
            "mse_mean_e3": float(mse.mean() * 1e3),
            "mse_se_e3": float(mse.sem() * 1e3),
            "improved_pct_mean": float(frac.mean() * 100.0),
            "improved_pct_se": float(frac.sem() * 100.0),
        })

    return {
        "dataset": dataset_key,
        "share_var": share_var,
        "M": int(dataset.M),
        "share_var_applied_to": sorted(estimator_kwargs),
        "records": records,
        "raw": frame.astype(float).to_dict(orient="list"),
    }


def render_text_table(summary: pd.DataFrame, share_var: bool) -> str:
    sub = summary[summary["share_var"] == share_var]
    lines = [
        f"=== share_var={share_var} ===",
        f"{'Estimator':18s}" + "".join(
            f"{DATASET_LABELS[d]:>34s}" for d in DATASET_LABELS
        ),
        f"{'':18s}" + "".join(f"{'MSE x1e-3':>18s}{'% Impr':>16s}"
                              for _ in DATASET_LABELS),
    ]
    for label, key in ROWS:
        cells = ""
        for dataset in DATASET_LABELS:
            row = sub[(sub["dataset"] == dataset) & (sub["estimator"] == key)]
            if row.empty:
                cells += f"{'-':>18s}{'-':>16s}"
                continue
            row = row.iloc[0]
            mse = f"{row['mse_mean_e3']:.3f}+-{row['mse_se_e3']:.3f}"
            if key == "mle":
                impr = "baseline"
            else:
                impr = f"{row['improved_pct_mean']:.1f}+-{row['improved_pct_se']:.1f}"
            cells += f"{mse:>18s}{impr:>16s}"
        lines.append(f"{label:18s}{cells}")
    return "\n".join(lines)


def render_comparison(summary: pd.DataFrame) -> str:
    """MSE only: published vs the two runs, so drift is easy to eyeball."""
    lines = ["=== MSE (x1e-3): published vs re-run ===",
             f"{'Dataset':16s}{'Estimator':18s}{'published':>12s}"
             f"{'share_var=F':>14s}{'share_var=T':>14s}"]
    for dataset in DATASET_LABELS:
        for label, key in ROWS:
            pub = PUBLISHED[dataset][key][0]
            def _get(sv):
                row = summary[(summary["dataset"] == dataset)
                              & (summary["estimator"] == key)
                              & (summary["share_var"] == sv)]
                return f"{row.iloc[0]['mse_mean_e3']:.3f}" if not row.empty else "-"
            lines.append(f"{dataset:16s}{label:18s}{pub:>12.3f}"
                         f"{_get(False):>14s}{_get(True):>14s}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_LABELS),
                        default=sorted(DATASET_LABELS))
    parser.add_argument("--share-var", choices=("both", "on", "off"), default="both")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    share_var_values = {"both": [False, True], "on": [True], "off": [False]}[
        args.share_var
    ]
    jobs = [(d, sv, args.trials) for d in args.datasets for sv in share_var_values]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs ({len(args.datasets)} datasets x "
          f"{len(share_var_values)} share_var) on "
          f"{min(args.num_workers, len(jobs))} workers, K={args.trials}")

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.num_workers, len(jobs)), mp_context=ctx
    ) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for future in as_completed(futures):
            dataset_key, share_var, _ = futures[future]
            payload = future.result()
            results.append(payload)
            pd.DataFrame(payload["raw"]).to_csv(
                args.out_dir / f"raw_{dataset_key}_sharevar-{share_var}.csv",
                index_label="trial",
            )
            print(f"  done: {dataset_key} share_var={share_var} "
                  f"(M={payload['M']})", flush=True)

    summary = pd.DataFrame([r for p in results for r in p["records"]])
    summary = summary.sort_values(["dataset", "share_var", "estimator"])
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    print()
    for share_var in share_var_values:
        print(render_text_table(summary, share_var))
        print()
    print(render_comparison(summary))

    # Rows that ignore share_var must match exactly across the two runs.
    if len(share_var_values) == 2:
        insensitive = [k for _, k in ROWS
                       if k not in results[0]["share_var_applied_to"]]
        mismatches = []
        for dataset in args.datasets:
            for key in insensitive:
                vals = [
                    summary[(summary["dataset"] == dataset)
                            & (summary["estimator"] == key)
                            & (summary["share_var"] == sv)]["mse_mean_e3"].iloc[0]
                    for sv in (False, True)
                ]
                if not np.isclose(vals[0], vals[1], rtol=1e-12):
                    mismatches.append((dataset, key, vals))
        print(f"share_var-insensitive rows ({', '.join(insensitive)}): "
              f"{'MATCH' if not mismatches else f'MISMATCH {mismatches}'}")

    with open(args.out_dir / "meta.json", "w") as fh:
        json.dump({
            "trials": args.trials,
            "datasets": args.datasets,
            "share_var_values": share_var_values,
            "rows": [k for _, k in ROWS],
            "share_var_applied_to": results[0]["share_var_applied_to"],
            "split_seed": "dataset default (42); replicate k uses seed 42 + k",
        }, fh, indent=2)
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
