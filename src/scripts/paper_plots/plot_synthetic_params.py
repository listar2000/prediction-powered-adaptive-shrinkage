"""Reproduce Figure 3 and its PT-variance diagnostic.

The figure shows PAS's first-stage power-tuning parameters and second-stage
adaptive-shrinkage weights for both synthetic predictors.  It uses the same
configuration as post-fix Table 2: m=200, n_j=20, N_j=80, covariate standard
deviation psi=0.2, additional response variance c=0.05, split seed 4321, and
the corrected analytical second moments from Appendix E.1. The problem-level power-tuning parameters are the
unrestricted minimizers in Eq. (13). A separate diagnostic plots the resulting
per-problem PT variances that turn PAS's fitted global shrinkage parameter into
local weights.

Usage::

    uv run python src/scripts/paper_plots/plot_synthetic_params.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pas.datasets.synthetic_model import GaussianSyntheticDataset
from pas.estimators.pas_estimators import get_pas_estimators
from pas.estimators.ppi_estimators import MIN_PT_VARIANCE, estimate_second_moments

M = 200
SPLIT_SEED = 4321
CUTOFF = 0.999
DEFAULT_OUTPUT = REPO_ROOT / "images" / "synthetic_params.pdf"
DEFAULT_DATA_OUTPUT = (
    REPO_ROOT / "src" / "scripts" / "post-fix" / "results"
    / "synthetic_params.csv"
)
DEFAULT_ZOOMED_OMEGA_OUTPUT = REPO_ROOT / "images" / "synthetic_omegas_zoomed.pdf"
DEFAULT_VARIANCE_OUTPUT = REPO_ROOT / "images" / "synthetic_variances.pdf"
DEFAULT_VARIANCE_DATA_OUTPUT = (
    REPO_ROOT / "src" / "scripts" / "post-fix" / "results"
    / "synthetic_variances.csv"
)


def _make_datasets() -> Tuple[GaussianSyntheticDataset, GaussianSyntheticDataset]:
    """Build the paired f1/f2 datasets using analytical true moments."""
    old_debug_flag = GaussianSyntheticDataset.DEBUG_FLAG
    try:
        # DEBUG_FLAG=True is the legacy 50k-draw Monte Carlo approximation.
        # Figure 3 now follows post-fix Table 2 and uses the corrected closed
        # forms for sigma_j^2, tau_j^2, and gamma_j instead.
        GaussianSyntheticDataset.DEBUG_FLAG = False
        good = GaussianSyntheticDataset(
            good_f=True, M=M, has_true_vars=True, split_seed=SPLIT_SEED,
            sigma_x=0.2,
        )
        bad = GaussianSyntheticDataset(
            good_f=False, M=M, has_true_vars=True, split_seed=SPLIT_SEED,
            sigma_x=0.2,
        )
    finally:
        GaussianSyntheticDataset.DEBUG_FLAG = old_debug_flag

    # With a shared seed and analytical moments, both predictors must be
    # evaluated on exactly the same 200 latent problem means and observations.
    np.testing.assert_array_equal(good.mu_x_s, bad.mu_x_s)
    assert good.M == bad.M == M
    assert np.all(good.ns == 20) and np.all(bad.ns == 20)
    assert np.all(good.Ns == 80) and np.all(bad.Ns == 80)
    assert good.additional_y_variance == bad.additional_y_variance == 0.05
    assert good.sigma_x == bad.sigma_x == 0.2
    return good, bad


def make_parameter_data() -> pd.DataFrame:
    """Return the parameters and PT variances for the two synthetic predictors."""
    datasets = _make_datasets()
    frames = []

    for predictor, dataset in zip(("f1", "f2"), datasets):
        _, lambdas, omegas = get_pas_estimators(
            dataset,
            get_lambdas=True,
            get_omegas=True,
            moments="per_problem",
            cutoff=CUTOFF,
            clip_lambda=False,
        )
        lambdas = np.asarray(lambdas, dtype=float)
        omegas = np.asarray(omegas, dtype=float)
        var_y, var_f, cov_yf = estimate_second_moments(
            dataset, moments="per_problem"
        )
        var_y = np.asarray(var_y, dtype=float)
        var_f = np.asarray(var_f, dtype=float)
        cov_yf = np.asarray(cov_yf, dtype=float)
        # This is exactly Eq. (14)'s PT variance used by PAS to turn the fitted
        # global shrinkage parameter into each local omega_j:
        #   omega_j = omega / (omega + tilde_sigma_j^2).
        tilde_sigma2 = (
            var_y / dataset.ns
            + lambdas**2 * var_f
            * (dataset.Ns + dataset.ns) / (dataset.Ns * dataset.ns)
            - 2 * lambdas * cov_yf / dataset.ns
        )
        tilde_sigma2 = np.maximum(tilde_sigma2, MIN_PT_VARIANCE)

        # Independently reduce Eqs. (13)--(14) at the analytical optimum. This
        # catches a mismatch in either the lambda implementation or the general
        # quadratic variance expression above.
        expected_lambdas = (
            dataset.Ns / (dataset.Ns + dataset.ns) * cov_yf / var_f
        )
        expected_tilde_sigma2 = (
            var_y / dataset.ns
            - dataset.Ns / (dataset.ns * (dataset.Ns + dataset.ns))
            * cov_yf**2 / var_f
        )
        np.testing.assert_allclose(
            lambdas, expected_lambdas, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            tilde_sigma2,
            np.maximum(expected_tilde_sigma2, MIN_PT_VARIANCE),
            rtol=1e-12,
            atol=1e-14,
        )
        assert (expected_tilde_sigma2 > MIN_PT_VARIANCE).all()
        assert (tilde_sigma2 <= var_y / dataset.ns + 1e-14).all()

        assert lambdas.shape == omegas.shape == (M,)
        assert tilde_sigma2.shape == (M,)
        assert (
            np.isfinite(lambdas).all()
            and np.isfinite(omegas).all()
            and np.isfinite(tilde_sigma2).all()
        )
        # Both synthetic covariances are nonnegative. Unlike UniPT's shared
        # parameter, Eq. (13)'s per-problem lambda_j^* need not be at most one.
        assert (lambdas >= 0).all()
        assert ((0 <= omegas) & (omegas <= 1)).all()
        assert (tilde_sigma2 > 0).all()

        # Recovering one constant from every (omega_j, variance_j) pair checks
        # that this is the same variance vector used for localization.
        fitted_global_omega = omegas * tilde_sigma2 / (1 - omegas)
        np.testing.assert_allclose(
            fitted_global_omega,
            np.repeat(fitted_global_omega[0], M),
            rtol=1e-10,
            atol=1e-14,
        )

        frames.append(
            pd.DataFrame(
                {
                    "problem": np.arange(1, M + 1),
                    "eta_j": dataset.mu_x_s,
                    "predictor": predictor,
                    "lambda_j": lambdas,
                    "omega_j": omegas,
                    "tilde_sigma2_j": tilde_sigma2,
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    assert result.groupby("predictor").size().to_dict() == {"f1": M, "f2": M}
    return result


def plot_parameter_data(data: pd.DataFrame, output: Path) -> None:
    """Render the two-panel plot in the style of the original Figure 3."""
    sns.set(style="whitegrid", context="talk")
    fig, (lambda_ax, omega_ax) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True
    )

    good = data[data["predictor"] == "f1"]
    bad = data[data["predictor"] == "f2"]
    f1_label = r"$f_1(x) = x^2$"
    f2_label = r"$f_2(x) = |x|$"

    lambda_ax.scatter(
        good["eta_j"], good["lambda_j"], color="red", marker="o", s=80,
        alpha=0.5, label=f1_label,
    )
    lambda_ax.scatter(
        bad["eta_j"], bad["lambda_j"], color="blue", marker="o", s=80,
        alpha=0.4, label=f2_label,
    )
    lambda_ax.set_ylabel(r"$\lambda_j^*$", fontsize=27)
    lambda_ax.set_title("Power-Tuning Parameters", fontsize=27)
    lambda_ax.legend(fontsize=20, loc="lower left")
    lambda_ax.grid(True)

    omega_ax.scatter(
        good["eta_j"], good["omega_j"], color="red", marker="v", s=120,
        alpha=0.5, label=f1_label,
    )
    omega_ax.scatter(
        bad["eta_j"], bad["omega_j"], color="blue", marker="v", s=120,
        alpha=0.4, label=f2_label,
    )
    omega_ax.set_xlabel(
        r"Problems indexed by $\eta_j = \mathrm{E}[X_{ij}]$", fontsize=27
    )
    omega_ax.set_ylabel(r"$\hat \omega_j$", fontsize=27)
    omega_ax.set_title("Adaptive Shrinkage Parameters", fontsize=27)
    omega_ax.legend(fontsize=20, loc="lower left", bbox_to_anchor=(0, 0.06))
    omega_ax.grid(True)

    lambda_ax.set_xlim(-1, 1)
    # Leave a small margin below zero so the near-zero f1 triangles are not
    # clipped by the bottom axis boundary.
    omega_ax.set_ylim(-0.025, 1.1)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, format=output.suffix.lstrip(".") or "pdf", bbox_inches="tight"
    )
    plt.close(fig)


def _padded_limits(values: pd.Series, fraction: float = 0.15) -> tuple[float, float]:
    """Return close limits that preserve a small margin around one series."""
    lower = float(values.min())
    upper = float(values.max())
    padding = fraction * (upper - lower)
    return lower - padding, upper + padding


def plot_zoomed_omega_data(data: pd.DataFrame, output: Path) -> None:
    """Plot local shrinkage factors on a broken y-axis."""
    sns.set(style="whitegrid", context="talk")
    fig, (upper_ax, lower_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1), "hspace": 0.08},
    )

    good = data[data["predictor"] == "f1"]
    bad = data[data["predictor"] == "f2"]
    upper_ax.scatter(
        bad["eta_j"], bad["omega_j"],
        color="blue", marker="v", s=120, alpha=0.4,
        label=r"$f_2(x) = |x|$",
    )
    lower_ax.scatter(
        good["eta_j"], good["omega_j"],
        color="red", marker="v", s=120, alpha=0.5,
        label=r"$f_1(x) = x^2$",
    )

    upper_ax.set_ylim(*_padded_limits(bad["omega_j"]))
    lower_ax.set_ylim(*_padded_limits(good["omega_j"]))
    lower_ax.set_xlim(-1, 1)
    lower_ax.set_xlabel(
        r"Problems indexed by $\eta_j = \mathrm{E}[X_{ij}]$", fontsize=27
    )
    fig.supylabel(r"$\hat \omega_j$", fontsize=27, x=0.025)
    upper_ax.set_title("Adaptive Shrinkage Parameters (Zoomed In)", fontsize=27)

    # Make the omitted middle of the y-axis explicit.
    upper_ax.spines["bottom"].set_visible(False)
    lower_ax.spines["top"].set_visible(False)
    upper_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower_ax.xaxis.tick_bottom()
    break_size = 0.5
    break_kwargs = {
        "marker": [(-1, -break_size), (1, break_size)],
        "markersize": 12,
        "linestyle": "none",
        "color": "black",
        "mec": "black",
        "mew": 1.5,
        "clip_on": False,
    }
    upper_ax.plot([0, 1], [0, 0], transform=upper_ax.transAxes, **break_kwargs)
    lower_ax.plot([0, 1], [1, 1], transform=lower_ax.transAxes, **break_kwargs)

    upper_ax.legend(fontsize=20, loc="lower center")
    lower_ax.legend(fontsize=20, loc="lower center")
    upper_ax.grid(True)
    lower_ax.grid(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, format=output.suffix.lstrip(".") or "pdf", bbox_inches="tight"
    )
    plt.close(fig)


def plot_variance_data(data: pd.DataFrame, output: Path) -> None:
    """Plot the PT variance that localizes PAS's global shrinkage parameter."""
    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    good = data[data["predictor"] == "f1"]
    bad = data[data["predictor"] == "f2"]
    ax.scatter(
        good["eta_j"], 1e3 * good["tilde_sigma2_j"],
        color="red", marker="o", s=80, alpha=0.5,
        label=r"$f_1(x) = x^2$",
    )
    ax.scatter(
        bad["eta_j"], 1e3 * bad["tilde_sigma2_j"],
        color="blue", marker="o", s=80, alpha=0.4,
        label=r"$f_2(x) = |x|$",
    )
    ax.set_xlim(-1, 1)
    ax.set_xlabel(
        r"Problems indexed by $\eta_j = \mathrm{E}[X_{ij}]$", fontsize=27
    )
    ax.set_ylabel(r"$\tilde{\sigma}_j^2$ ($\times 10^{-3}$)", fontsize=27)
    ax.set_title("Variance of the Power-Tuned Estimator", fontsize=27)
    ax.legend(fontsize=20, loc="upper center")
    ax.grid(True)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, format=output.suffix.lstrip(".") or "pdf", bbox_inches="tight"
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"figure destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--data-output", type=Path, default=DEFAULT_DATA_OUTPUT,
        help=f"plotted-data destination (default: {DEFAULT_DATA_OUTPUT})",
    )
    parser.add_argument(
        "--zoomed-omega-output", type=Path, default=DEFAULT_ZOOMED_OMEGA_OUTPUT,
        help=(
            "broken-axis shrinkage-factor figure destination "
            f"(default: {DEFAULT_ZOOMED_OMEGA_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--variance-output", type=Path, default=DEFAULT_VARIANCE_OUTPUT,
        help=f"PT-variance figure destination (default: {DEFAULT_VARIANCE_OUTPUT})",
    )
    parser.add_argument(
        "--variance-data-output", type=Path, default=DEFAULT_VARIANCE_DATA_OUTPUT,
        help=(
            "PT-variance plotted-data destination "
            f"(default: {DEFAULT_VARIANCE_DATA_OUTPUT})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = make_parameter_data()

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False)
    plot_parameter_data(data, args.output)
    plot_zoomed_omega_data(data, args.zoomed_omega_output)
    variance_data = data[
        ["problem", "eta_j", "predictor", "tilde_sigma2_j"]
    ].copy()
    args.variance_data_output.parent.mkdir(parents=True, exist_ok=True)
    variance_data.to_csv(args.variance_data_output, index=False)
    plot_variance_data(variance_data, args.variance_output)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.data_output} ({len(data)} rows; {M} per predictor)")
    print(f"Wrote {args.zoomed_omega_output}")
    print(f"Wrote {args.variance_output}")
    print(
        f"Wrote {args.variance_data_output} "
        f"({len(variance_data)} rows; {M} per predictor)"
    )
    for predictor, group in data.groupby("predictor", sort=True):
        print(
            f"{predictor}: lambda=[{group.lambda_j.min():.6f}, "
            f"{group.lambda_j.max():.6f}], omega=[{group.omega_j.min():.6f}, "
            f"{group.omega_j.max():.6f}], "
            f"tilde_sigma2=[{group.tilde_sigma2_j.min():.9f}, "
            f"{group.tilde_sigma2_j.max():.9f}]"
        )


if __name__ == "__main__":
    main()
