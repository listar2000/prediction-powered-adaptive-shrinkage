"""Post-fix reproduction of Tables 2 and 3 of the PAS paper.

Re-runs the paper's benchmarks after the code corrections described in
``README.md`` in this directory.

- **Table 3** (real data): nine estimator rows x three dataset columns (Amazon
  ``BERT-base``, Amazon ``BERT-tuned``, Galaxy), K = 200 replicates at the
  default split seed -- the configuration of Appendix E.4. Results are NOT
  comparable row-for-row with the published table: the pseudo-ground-truth fix
  changes the estimand, so every estimator is evaluated against a different
  target (see the README).
- **Table 2** (synthetic): seven estimator rows x two predictors
  (f1(x) = x^2, f2(x) = |x|), m = 200, n_j = 20, N_j = 80. The true second
  moments are evaluated from their corrected closed forms rather than by Monte
  Carlo. The revised experiment uses psi = 0.2 rather than the published
  psi = 0.1, so the published table is included only as a reference.

Every cell uses per-problem second moments (Appendix C.1), which the tables
report as ``share_var=False`` -- the configuration behind the paper's numbers.
The per-problem PT, shrink-average, and PAS rows use the unrestricted
problem-level power-tuning parameter in Eq. (13). UniPT and UniPAS retain the
explicit ``[0, 1]`` clipping in Appendix C.2.
Each dataset is an independent job, so they run in parallel subprocesses;
datasets are constructed inside the worker rather than pickled across the
process boundary.

Usage::

    uv run python src/scripts/post-fix/run_post_fix_tables.py
    uv run python src/scripts/post-fix/run_post_fix_tables.py --trials 5 --num-workers 2
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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
HERE = Path(__file__).resolve().parent

#: The second-moment plug-in used for every cell. ``share_var=False`` is the
#: historical spelling of this and is what the table captions report.
MOMENTS = "per_problem"

# (paper row label, registry key). `mle` must come first: `run_benchmark`
# measures "% Improved" against it.
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

# Table 2 omits UniPT/UniPAS: the paper includes them "only for the real-world
# experiments, as they are specifically designed for settings where the second
# moments are unknown".
SYNTHETIC_ROWS = [row for row in ROWS if row[1] not in {"uni_pt", "uni_pas"}]

# Eq. (13) does not constrain the per-problem lambda_j^*. All paper rows whose
# first stage is that estimator therefore opt out of the library's defensive
# PPI++-style clipping. The single shared lambda in UniPT/UniPAS remains clipped
# exactly as specified in Appendix C.2.
UNCLIPPED_PT_ROWS = {"pt", "shrinkage_mean", "pas"}

REAL_DATASETS = {
    "amazon_base": "Amazon (base f)",
    "amazon_tuned": "Amazon (tuned f)",
    "galaxy": "Galaxy",
}
SYNTHETIC_DATASETS = {
    "synthetic_f1": "Synthetic f1=x^2",
    "synthetic_f2": "Synthetic f2=|x|",
}
DATASET_LABELS = {**REAL_DATASETS, **SYNTHETIC_DATASETS}

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

# Published Table 2 at psi=0.1, as (mse_mean, mse_se). The paper reports no
# "% Improved" column for the synthetic study. The revised run uses psi=0.2.
PUBLISHED_SYNTHETIC = {
    "synthetic_f1": {
        "mle": (3.142, 0.033), "pred_mean": (0.273, 0.004),
        "ppi": (2.689, 0.027), "pt": (2.642, 0.027),
        "shrinkage_only": (0.272, 0.003), "shrinkage_mean": (2.486, 0.026),
        "pas": (0.272, 0.003),
    },
    "synthetic_f2": {
        "mle": (3.142, 0.033), "pred_mean": (34.335, 0.147),
        "ppi": (2.756, 0.027), "pt": (2.659, 0.026),
        "shrinkage_only": (2.863, 0.030), "shrinkage_mean": (2.537, 0.026),
        "pas": (2.466, 0.026),
    },
}


def build_dataset(key: str):
    """Construct a dataset by key. Called inside the worker process."""
    from pas.datasets.amazon_review import AmazonReviewDataset
    from pas.datasets.galaxy_zoo import GalaxyZooDataset
    from pas.datasets.synthetic_model import GaussianSyntheticDataset

    if key == "amazon_base":
        return AmazonReviewDataset(tuned=False)
    if key == "amazon_tuned":
        return AmazonReviewDataset(tuned=True)
    if key == "galaxy":
        return GalaxyZooDataset()
    if key in SYNTHETIC_DATASETS:
        # Table 2 / Figure 3 configuration. Select the analytical path
        # explicitly: DEBUG_FLAG=True is the legacy 50k-draw Monte Carlo
        # approximation retained as the dataset-wide default for backwards
        # reproducibility, whereas Table 2 now uses the corrected closed forms.
        GaussianSyntheticDataset.DEBUG_FLAG = False
        return GaussianSyntheticDataset(
            good_f=(key == "synthetic_f1"), M=200,
            has_true_vars=True, split_seed=4321, sigma_x=0.2,
        )
    raise ValueError(f"unknown dataset {key!r}")


def _accepts(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C functions
        return False


def run_cell(args) -> dict:
    """One dataset: K replicates over that table's estimator rows."""
    dataset_key, trials = args

    warnings.filterwarnings("ignore")
    from pas.estimators import ALL_ESTIMATORS
    from pas.experiments import run_benchmark

    rows = SYNTHETIC_ROWS if dataset_key in SYNTHETIC_DATASETS else ROWS
    estimators = {key: ALL_ESTIMATORS[key] for _, key in rows}
    # UniPT / UniPAS deliberately take no moment argument: Appendix C.2 / C.3
    # fix their moment handling, so it is part of the estimator, not a knob.
    estimator_kwargs = {}
    for key, func in estimators.items():
        kwargs = {}
        if _accepts(func, "moments"):
            kwargs["moments"] = MOMENTS
        if key in UNCLIPPED_PT_ROWS:
            assert _accepts(func, "clip_lambda")
            kwargs["clip_lambda"] = False
        if kwargs:
            estimator_kwargs[key] = kwargs

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
    for label, key in rows:
        mse = frame[key].astype(float)
        frac = frame[f"{key}_frac"].astype(float)
        records.append({
            "dataset": dataset_key,
            "share_var": False,          # i.e. moments="per_problem"
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
        "M": int(dataset.M),
        "moments": MOMENTS,
        "moments_applied_to": sorted(estimator_kwargs),
        "unclipped_pt_rows": sorted(UNCLIPPED_PT_ROWS & set(estimators)),
        "records": records,
        "raw": frame.astype(float).to_dict(orient="list"),
    }


def render_text_table(summary: pd.DataFrame, datasets: dict, rows: list,
                      title: str, with_improved: bool = True) -> str:
    width = 34 if with_improved else 18
    lines = [
        f"=== {title} ===",
        f"{'Estimator':18s}" + "".join(f"{datasets[d]:>{width}s}" for d in datasets),
        f"{'':18s}" + "".join(
            (f"{'MSE x1e-3':>18s}{'% Impr':>16s}" if with_improved
             else f"{'MSE x1e-3':>18s}") for _ in datasets),
    ]
    for label, key in rows:
        cells = ""
        for dataset in datasets:
            row = summary[(summary["dataset"] == dataset)
                          & (summary["estimator"] == key)]
            if row.empty:
                cells += f"{'-':>18s}" + ("" if not with_improved else f"{'-':>16s}")
                continue
            row = row.iloc[0]
            cells += f"{f'{row.mse_mean_e3:.3f}+-{row.mse_se_e3:.3f}':>18s}"
            if with_improved:
                impr = ("baseline" if key == "mle" else
                        f"{row.improved_pct_mean:.1f}+-{row.improved_pct_se:.1f}")
                cells += f"{impr:>16s}"
        lines.append(f"{label:18s}{cells}")
    return "\n".join(lines)


def render_comparison(summary: pd.DataFrame) -> str:
    """MSE only: published vs re-run, so drift is easy to eyeball."""
    lines = ["=== MSE (x1e-3): published vs re-run ===",
             f"{'Dataset':16s}{'Estimator':18s}{'published':>12s}{'re-run':>12s}"
             f"{'ratio':>9s}"]
    for dataset in DATASET_LABELS:
        rows = SYNTHETIC_ROWS if dataset in SYNTHETIC_DATASETS else ROWS
        for label, key in rows:
            pub = (PUBLISHED_SYNTHETIC[dataset][key][0]
                   if dataset in SYNTHETIC_DATASETS
                   else PUBLISHED[dataset][key][0])
            row = summary[(summary["dataset"] == dataset)
                          & (summary["estimator"] == key)]
            if row.empty:
                continue
            new = float(row.iloc[0]["mse_mean_e3"])
            lines.append(f"{dataset:16s}{label:18s}{pub:>12.3f}{new:>12.3f}"
                         f"{new / pub:>9.3f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_LABELS),
                        default=sorted(DATASET_LABELS))
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--preserve-unselected",
        action="store_true",
        help="Keep rows and raw files for datasets not named by --datasets. "
             "This supports rerunning only Table 2 while retaining Table 3.",
    )
    args = parser.parse_args()

    jobs = [(d, args.trials) for d in args.datasets]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs on {min(args.num_workers, len(jobs))} workers, "
          f"K={args.trials}, moments={MOMENTS}")

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.num_workers, len(jobs)), mp_context=ctx
    ) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for future in as_completed(futures):
            dataset_key, _ = futures[future]
            payload = future.result()
            results.append(payload)
            pd.DataFrame(payload["raw"]).to_csv(
                args.out_dir / f"raw_{dataset_key}.csv", index_label="trial")
            print(f"  done: {dataset_key} (M={payload['M']})", flush=True)

    summary = pd.DataFrame([r for p in results for r in p["records"]])
    summary_path = args.out_dir / "summary.csv"
    if args.preserve_unselected and summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous[~previous["dataset"].isin(args.datasets)]
        summary = pd.concat([previous, summary], ignore_index=True)
    summary = summary.sort_values(["dataset", "estimator"])
    summary.to_csv(summary_path, index=False)

    print()
    present = set(summary["dataset"])
    real = {k: v for k, v in REAL_DATASETS.items() if k in present}
    synth = {k: v for k, v in SYNTHETIC_DATASETS.items() if k in present}
    if real:
        print(render_text_table(summary, real, ROWS,
                                "Table 3 (real data), share_var=False"))
        print()
    if synth:
        print(render_text_table(summary, synth, SYNTHETIC_ROWS,
                                "Table 2 (synthetic model)", with_improved=False))
        print()
    print(render_comparison(summary))

    with open(args.out_dir / "meta.json", "w") as fh:
        json.dump({
            "trials": args.trials,
            "datasets": [k for k in DATASET_LABELS if k in present],
            "moments": MOMENTS,
            "moments_applied_to": results[0]["moments_applied_to"],
            "unclipped_pt_rows": sorted(UNCLIPPED_PT_ROWS),
            "unipt_unipas_lambda": "clipped to [0, 1] per Appendix C.2",
            "synthetic_moment_source": "corrected analytical closed forms",
            "table3_rows": [k for _, k in ROWS],
            "table2_rows": [k for _, k in SYNTHETIC_ROWS],
            "split_seed": "dataset default (42 real, 4321 synthetic); "
                          "replicate k uses seed + k",
        }, fh, indent=2)
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
