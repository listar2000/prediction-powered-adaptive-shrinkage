"""Reproduce the illustrative synthetic-model figure used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "images" / "algo_06_04.png"
DEFAULT_DATA_OUTPUT = Path(__file__).resolve().parent / "results" / "synthetic_example.csv"
C = 0.05
PSI = 0.2
FIGURE_SEED = 13509


def make_data(seed: int = FIGURE_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for problem, eta in enumerate(np.linspace(-1, 1, 10), start=1):
        x_labelled = rng.normal(eta, PSI, 10)
        y_labelled = (
            eta**2
            + 2 * eta * (x_labelled - eta)
            + rng.normal(0, np.sqrt(C), 10)
        )
        x_unlabelled = rng.normal(eta, PSI, 20)
        for observation, (x, y) in enumerate(zip(x_labelled, y_labelled), start=1):
            rows.append((problem, eta, "labelled", observation, x, y))
        for observation, x in enumerate(x_unlabelled, start=1):
            rows.append(
                (problem, eta, "unlabelled_prediction", observation, x, abs(x))
            )
    return pd.DataFrame(
        rows,
        columns=["problem", "eta_j", "sample", "observation", "x", "value"],
    )


def _draw_mean_segment(
    ax,
    *,
    mean_x: float,
    mean_y: float,
    xmin: float,
    xmax: float,
    color,
    linestyle: str,
) -> None:
    span = xmax - xmin
    if mean_x < 0:
        ax.plot([xmin, mean_x], [mean_y, mean_y], color=color,
                alpha=0.55, linestyle=linestyle)
        ax.text(xmin + 0.012 * span, mean_y, f"{mean_y:.2f}", color=color,
                fontsize=14, ha="left", va="bottom")
    else:
        ax.plot([mean_x, xmax], [mean_y, mean_y], color=color,
                alpha=0.55, linestyle=linestyle)
        ax.text(xmax - 0.012 * span, mean_y, f"{mean_y:.2f}", color=color,
                fontsize=14, ha="right", va="bottom")


def plot_example(data: pd.DataFrame, output: Path) -> None:
    sns.set(style="whitegrid", context="talk")
    fig, (labelled_ax, prediction_ax) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    x_min, x_max = float(data["x"].min()), float(data["x"].max())
    padding = 0.08 * (x_max - x_min)
    xmin, xmax = x_min - padding, x_max + padding
    labelled_ax.set_xlim(xmin, xmax)

    for problem, color in zip(range(1, 11), colors):
        labelled = data[
            (data["problem"] == problem) & (data["sample"] == "labelled")
        ]
        predicted = data[
            (data["problem"] == problem)
            & (data["sample"] == "unlabelled_prediction")
        ]
        labelled_ax.scatter(
            labelled["x"], labelled["value"], color=color, alpha=0.7, s=90
        )
        prediction_ax.scatter(
            predicted["x"], predicted["value"], color=color,
            alpha=0.7, s=90, marker="v",
        )
        _draw_mean_segment(
            labelled_ax,
            mean_x=float(labelled["x"].mean()),
            mean_y=float(labelled["value"].mean()),
            xmin=xmin,
            xmax=xmax,
            color=color,
            linestyle="--",
        )
        _draw_mean_segment(
            prediction_ax,
            mean_x=float(predicted["x"].mean()),
            mean_y=float(predicted["value"].mean()),
            xmin=xmin,
            xmax=xmax,
            color=color,
            linestyle="-.",
        )

    labelled_ax.plot([], [], color="black", linestyle="--", label=r"$\bar{Y}_j$")
    prediction_ax.plot(
        [], [], color="black", linestyle="-.", label=r"$\tilde{Z}^f_j$"
    )
    labelled_ax.set_title(
        r"$n_j=10$ labeled points from each of $m=10$ problems", fontsize=20
    )
    labelled_ax.set_xlabel(r"$X_{ij}$", fontsize=20)
    labelled_ax.set_ylabel(r"$Y_{ij}$", fontsize=20)
    prediction_ax.set_title(
        r"$N_j=20$ unlabeled points projected onto $y=|x|$", fontsize=20
    )
    prediction_ax.set_xlabel(r"$\tilde X_{ij}$", fontsize=20)
    prediction_ax.set_ylabel(r"$f(\tilde X_{ij})$", fontsize=20)
    labelled_ax.legend(loc="upper center", fontsize=20, frameon=True)
    prediction_ax.legend(loc="upper center", fontsize=20, frameon=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    args = parser.parse_args()

    data = make_data()
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False)
    plot_example(data, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.data_output}")


if __name__ == "__main__":
    main()
