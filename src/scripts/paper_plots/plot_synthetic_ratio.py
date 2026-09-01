"""Reproduce the covariance-to-variance ratio plot in Appendix E.1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "images" / "ratio.pdf"
C = 0.05
PSI = 0.2


def plot_ratio(output: Path) -> None:
    eta = np.linspace(-1.0, 1.0, 1001)
    ratio = 2.0 * np.abs(eta) * PSI**2 / (4.0 * eta**2 * PSI**2 + C)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(
        eta,
        ratio,
        color=sns.color_palette("deep")[0],
        linewidth=1.5,
        label=r"$\dfrac{2|\eta_j|\psi^2}{4\eta_j^2\psi^2+c}$",
    )
    ax.set_title(
        r"Ratio between $|\mathrm{Cov}_{\eta_j}[X_{ij}, Y_{ij}]|$ and "
        r"$\mathrm{Var}_{\eta_j}[Y_{ij}]$",
        fontsize=18,
    )
    ax.set_xlabel(r"$\eta_j$", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower right", frameon=True, fontsize=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_ratio(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
