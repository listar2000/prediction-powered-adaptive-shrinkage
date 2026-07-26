"""Run the LM Arena pseudo-oracle bias-exchangeability diagnostic.

Examples
--------
From the repository root::

    python src/scripts/run_lmarena_exchangeability.py

To render only panel A (bias versus target) without running the permutation
test, which is the expensive part::

    python src/scripts/run_lmarena_exchangeability.py --panel a

The default input is the Bradley--Terry probability artifact
``data/lmarena/clean_data/clean_summary_v2.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pas.statistical_tests import (  # noqa: E402
    load_lmarena_pseudo_oracle,
    plot_bias_exchangeability,
    plot_bias_versus_target,
    run_statistical_test,
)
from pas.statistical_tests.exchangeability import equal_count_bins  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether the pseudo-bias distribution varies across target bins."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "lmarena"
        / "clean_data"
        / "clean_summary_v2.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "lmarena_exchangeability",
    )
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=8273)
    parser.add_argument(
        "--panel",
        choices=("all", "a"),
        default="all",
        help="'all' runs the permutation test and writes the three-panel "
             "figure; 'a' writes only the descriptive bias-versus-target "
             "panel and skips the test.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_lmarena_pseudo_oracle(args.csv)

    if args.panel == "a":
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        bins = equal_count_bins(summary["theta"].to_numpy(dtype=float), args.bins)
        figure, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
        plot_bias_versus_target(summary, bins, ax=ax)
        for suffix in ("png", "pdf"):
            path = args.out_dir / f"exchangeability_panel_a.{suffix}"
            figure.savefig(path, dpi=220, bbox_inches="tight")
            print(f"Wrote {path}")
        plt.close(figure)
        return

    result = run_statistical_test(
        "bias_exchangeability_by_target",
        summary,
        n_bins=args.bins,
        permutations=args.permutations,
        seed=args.seed,
    )

    task_summary = summary.copy()
    task_summary["target_bin"] = np.asarray(result.metadata["bin_index"], dtype=int)
    task_summary.to_csv(args.out_dir / "task_summary.csv", index=False)

    result.metadata["data"] = {
        "csv": str(args.csv),
        "number_of_tasks": int(len(summary)),
        "bias_mean": float(summary["bias"].mean()),
        "bias_sd": float(summary["bias"].std(ddof=1)),
        "bias_min": float(summary["bias"].min()),
        "bias_max": float(summary["bias"].max()),
        "median_task_size": float(summary["n"].median()),
        "min_task_size": int(summary["n"].min()),
        "max_task_size": int(summary["n"].max()),
        "prediction_n_unique": summary.attrs.get("prediction_n_unique"),
        "prediction_min": summary.attrs.get("prediction_min"),
        "prediction_max": summary.attrs.get("prediction_max"),
        "prediction_is_binary": summary.attrs.get("prediction_is_binary"),
    }
    result.to_json(args.out_dir / "diagnostics.json")

    for suffix in ("png", "pdf"):
        figure = plot_bias_exchangeability(
            summary,
            result,
            args.out_dir / f"exchangeability_diagnostic.{suffix}",
        )
        import matplotlib.pyplot as plt

        plt.close(figure)

    print(result.format())
    print(f"\nWrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
