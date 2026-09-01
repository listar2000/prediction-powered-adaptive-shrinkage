"""Reproduce the corrected real-data split-ratio curves for PAS Figure 6.

The experiment follows Appendix E.4--E.5: for each labelled fraction from 1%
through 40%, redraw K=200 labelled/unlabelled splits using seeds 42,...,241,
evaluate against the mean of all responses, and report average MSE +/- 1 SE.

Run from the repository root:

    uv run python src/scripts/paper_plots/plot_split_ratios.py
"""
from __future__ import annotations

import argparse
import inspect
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_DIR = HERE / "results"

BASE_SEED = 42
MOMENTS = "per_problem"

# Figure 6 predates the addition of Shrink Avg and contains these eight curves.
ESTIMATORS = [
    ("mle", "Classical"),
    ("pred_mean", "Prediction Avg"),
    ("ppi", "PPI"),
    ("pt", "PT"),
    ("shrinkage_only", "Shrink Classical"),
    ("pas", "PAS"),
    ("uni_pt", "UniPT"),
    ("uni_pas", "UniPAS"),
]

# Appendix C.1's sample variance/covariance estimators require n_j >= 2.
REQUIRES_SECOND_MOMENTS = {
    "pt", "shrinkage_only", "pas", "uni_pt", "uni_pas",
}

DATASETS = {
    "amazon_base": {
        "title": r"Amazon Review (base $f$)",
        "filename": "split_ratio_amazon_base.pdf",
    },
    "amazon_tuned": {
        "title": r"Amazon Review (tuned $f$)",
        "filename": "split_ratio_amazon_tuned.pdf",
    },
    "galaxy": {
        "title": "Galaxy Zoo",
        "filename": "split_ratio_galaxy_zoo.pdf",
    },
}


def build_dataset(key: str, labelled_fraction: float):
    """Construct one of the three Figure 6 datasets."""
    from pas.datasets.amazon_review import AmazonReviewDataset
    from pas.datasets.galaxy_zoo import GalaxyZooDataset

    if key == "amazon_base":
        return AmazonReviewDataset(
            tuned=False, train_test_split=labelled_fraction,
            split_seed=BASE_SEED,
        )
    if key == "amazon_tuned":
        return AmazonReviewDataset(
            tuned=True, train_test_split=labelled_fraction,
            split_seed=BASE_SEED,
        )
    if key == "galaxy":
        return GalaxyZooDataset(
            train_test_split=labelled_fraction, split_seed=BASE_SEED,
        )
    raise ValueError(f"unknown dataset {key!r}")


def _accepts(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def run_cell(job: tuple[str, int, int]) -> list[dict]:
    """Run one (dataset, labelled percentage) cell for all K replicates."""
    dataset_key, ratio_pct, trials = job
    warnings.filterwarnings("ignore")

    from pas.estimators import ALL_ESTIMATORS
    estimator_functions = {
        key: ALL_ESTIMATORS[key] for key, _ in ESTIMATORS
    }
    estimator_kwargs = {
        key: {"moments": MOMENTS}
        for key, func in estimator_functions.items()
        if _accepts(func, "moments")
    }
    data = build_dataset(dataset_key, ratio_pct / 100.0)
    mse_values = {key: [] for key, _ in ESTIMATORS}
    for trial in range(trials):
        data.reload_data(split_seed=BASE_SEED + trial)
        for key, _ in ESTIMATORS:
            # At the 1% Galaxy split, 10 problems have n_j=1. The moment-based
            # estimators are undefined there under Appendix C.1; preserve that
            # fact instead of silently raising the nominal labeled fraction.
            if key in REQUIRES_SECOND_MOMENTS and np.any(data.ns < 2):
                mse_values[key].append(np.nan)
                continue
            estimates = estimator_functions[key](
                data, **estimator_kwargs.get(key, {}))
            mse_values[key].append(
                float(np.mean((np.asarray(estimates) - data.true_theta) ** 2)))

    records = []
    for key, label in ESTIMATORS:
        values = pd.Series(mse_values[key], dtype=float)
        records.append({
            "dataset": dataset_key,
            "ratio_pct": ratio_pct,
            "labelled_fraction": ratio_pct / 100.0,
            "estimator": key,
            "label": label,
            "trials": trials,
            "valid_trials": int(values.notna().sum()),
            "mse_mean": float(values.mean()),
            "mse_se": float(values.sem()),
        })
    return records


def plot_results(summary: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "mle": "#4C78A8",
        "pred_mean": "#F58518",
        "ppi": "#54A24B",
        "pt": "#E45756",
        "shrinkage_only": "#7A5195",
        "pas": "#9C755F",
        "uni_pt": "#E377C2",
        "uni_pas": "#262626",
    }
    linestyles = {
        "mle": "-",
        "pred_mean": "--",
        "ppi": "-",
        "pt": "--",
        "shrinkage_only": "-.",
        "pas": "-",
        "uni_pt": ":",
        "uni_pas": "--",
    }
    markers = {
        "mle": "o", "pred_mean": "s", "ppi": "^", "pt": "v",
        "shrinkage_only": "D", "pas": "P", "uni_pt": "X", "uni_pas": "*",
    }

    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key, metadata in DATASETS.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
        dataset_rows = summary[summary["dataset"] == dataset_key]
        for key, label in ESTIMATORS:
            rows = dataset_rows[dataset_rows["estimator"] == key].sort_values(
                "ratio_pct")
            x = rows["ratio_pct"].to_numpy(dtype=float)
            mean = rows["mse_mean"].to_numpy(dtype=float)
            se = rows["mse_se"].to_numpy(dtype=float)
            color = colors[key]
            ax.plot(
                x, mean, label=label, color=color,
                linestyle=linestyles[key], linewidth=1.8,
                marker=markers[key], markersize=3.5, markevery=4,
            )
            finite_band = np.isfinite(mean) & np.isfinite(se)
            if finite_band.any():
                ax.fill_between(
                    x[finite_band],
                    np.maximum(0.0, mean[finite_band] - se[finite_band]),
                    mean[finite_band] + se[finite_band],
                    color=color, alpha=0.10, linewidth=0,
                )

        ax.set_title(metadata["title"], fontsize=12, loc="left")
        ax.set_xlabel("Labeled fraction (%)")
        ax.set_ylabel("Mean squared error")
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
        # Zoom independently for each dataset. Put the nearly horizontal
        # Prediction Avg curve about three quarters of the way up the panel,
        # while retaining at least 5% headroom above its highest point. This
        # intentionally clips the very large small-n errors of other methods so
        # the scientifically relevant separation is visible.
        prediction_mse = dataset_rows[
            dataset_rows["estimator"] == "pred_mean"
        ]["mse_mean"].to_numpy(dtype=float)
        y_upper = max(
            float(np.mean(prediction_mse)) / 0.75,
            float(np.max(prediction_mse)) / 0.95,
        )
        ax.set_ylim(0, y_upper)
        ax.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            ncol=2, fontsize=8, frameon=False, loc="upper right",
            columnspacing=1.0, handlelength=2.4,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        fig.savefig(plot_dir / metadata["filename"], format="pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--ratios", type=int, nargs="+", default=list(range(1, 41)),
        help="Labeled percentages to evaluate (default: every integer 1,...,40).",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--num-workers", type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=RESULTS_DIR / "split_ratio_results.csv",
    )
    parser.add_argument(
        "--plot-dir", type=Path,
        default=REPO_ROOT / "images" / "varying_ratios",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Re-render PDFs from --output-csv without rerunning experiments.",
    )
    args = parser.parse_args()

    if args.plot_only:
        summary = pd.read_csv(args.output_csv)
        plot_results(summary, args.plot_dir)
        for key in args.datasets:
            print(f"Saved plot -> {args.plot_dir / DATASETS[key]['filename']}",
                  flush=True)
        return

    ratios = sorted(set(args.ratios))
    if not ratios or ratios[0] < 1 or ratios[-1] > 40:
        parser.error("ratios must lie in [1, 40]")
    jobs = [
        (dataset_key, ratio_pct, args.trials)
        for dataset_key in args.datasets for ratio_pct in ratios
    ]
    print(
        f"Running {len(jobs)} cells on {min(args.num_workers, len(jobs))} "
        f"workers: K={args.trials}, moments={MOMENTS}",
        flush=True,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    partial_csv = args.output_csv.with_suffix(".partial.csv")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.num_workers, len(jobs)), mp_context=ctx,
    ) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            dataset_key, ratio_pct, _ = futures[future]
            records.extend(future.result())
            pd.DataFrame(records).sort_values(
                ["dataset", "ratio_pct", "estimator"]
            ).to_csv(partial_csv, index=False)
            print(
                f"  [{completed:3d}/{len(jobs)}] {dataset_key} "
                f"labeled={ratio_pct}%",
                flush=True,
            )

    summary = pd.DataFrame(records).sort_values(
        ["dataset", "ratio_pct", "estimator"])
    summary.to_csv(args.output_csv, index=False)
    partial_csv.unlink(missing_ok=True)
    plot_results(summary, args.plot_dir)
    print(f"Saved numerical results -> {args.output_csv}", flush=True)
    for key in args.datasets:
        print(f"Saved plot -> {args.plot_dir / DATASETS[key]['filename']}", flush=True)


if __name__ == "__main__":
    main()
