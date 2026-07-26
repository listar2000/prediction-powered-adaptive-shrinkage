from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pas.statistical_tests import (
    STATISTICAL_TESTS,
    BiasExchangeabilityTest,
    load_lmarena_pseudo_oracle,
    plot_bias_exchangeability,
    run_statistical_test,
)
from pas.statistical_tests.exchangeability import equal_count_bins


def test_equal_count_bins_are_balanced_and_monotone() -> None:
    values = np.asarray([5.0, 1.0, 9.0, 4.0, 8.0, 0.0, 2.0, 7.0, 3.0, 6.0])
    bins = equal_count_bins(values, n_bins=4)
    counts = np.bincount(bins, minlength=4)
    assert counts.max() - counts.min() <= 1
    ordered = bins[np.argsort(values, kind="mergesort")]
    assert np.all(np.diff(ordered) >= 0)


def test_target_dependent_bias_is_detected() -> None:
    rng = np.random.default_rng(19)
    theta = np.linspace(0.05, 0.95, 96)
    target_bin = equal_count_bins(theta, n_bins=4)
    bias = 0.12 * target_bin + rng.normal(0.0, 0.025, size=theta.size)
    summary = pd.DataFrame({"theta": theta, "bias": bias})

    result = run_statistical_test(
        "bias_exchangeability_by_target",
        summary,
        n_bins=4,
        permutations=999,
        seed=12,
    )

    assert result.p_value <= 0.002
    assert result.statistic > 1.0
    assert result.metadata["max_pair"] == [1, 4]
    assert result.null_distribution is not None
    assert result.null_distribution.size == 999
    assert "Wasserstein" in result.method


def test_result_json_and_registry() -> None:
    summary = pd.DataFrame(
        {
            "theta": np.linspace(0.0, 1.0, 24),
            "bias": np.sin(np.linspace(0.0, 4.0, 24)),
        }
    )
    result = BiasExchangeabilityTest().run(
        summary, n_bins=4, permutations=49, seed=1
    )
    parsed = json.loads(result.to_json())
    assert parsed["name"] == "bias_exchangeability_by_target"
    assert "null_distribution" not in parsed
    assert parsed["metadata"]["n_bins"] == 4
    assert "bias_exchangeability_by_target" in STATISTICAL_TESTS


def test_lmarena_loader_aggregates_pseudo_oracle(tmp_path: Path) -> None:
    rows = []
    for group_id, (model_a, model_b) in enumerate([("a", "b"), ("c", "d")]):
        for index in range(5):
            rows.append(
                {
                    "group_id": group_id,
                    "model_a": model_a,
                    "model_b": model_b,
                    "winner": float((index + group_id) % 2),
                    "prediction": 0.1 + 0.15 * index + 0.05 * group_id,
                }
            )
    csv_path = tmp_path / "lmarena.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = load_lmarena_pseudo_oracle(csv_path)

    assert list(summary["group_id"]) == [0, 1]
    assert np.all(summary["n"].to_numpy() == 5)
    np.testing.assert_allclose(
        summary["bias"], summary["prediction_mean"] - summary["theta"]
    )
    assert summary.attrs["prediction_is_binary"] is False
    assert np.all(np.isfinite(summary["bias_se"]))


def test_exchangeability_plot_is_written(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "theta": np.linspace(0.05, 0.95, 40),
            "bias": np.cos(np.linspace(0.0, 3.0, 40)) / 10.0,
        }
    )
    result = BiasExchangeabilityTest().run(
        summary, n_bins=4, permutations=49, seed=8
    )
    output = tmp_path / "diagnostic.png"
    figure = plot_bias_exchangeability(summary, result, output)
    try:
        assert output.exists()
        assert output.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
