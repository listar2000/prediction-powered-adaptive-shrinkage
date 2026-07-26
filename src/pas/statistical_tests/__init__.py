"""Registry and built-in statistical diagnostics for PAS."""
from __future__ import annotations

from typing import Any

from pas.statistical_tests.base import StatisticalTest, StatisticalTestResult
from pas.statistical_tests.exchangeability import (
    BiasExchangeabilityTest,
    load_lmarena_pseudo_oracle,
    test_bias_exchangeability,
)
from pas.statistical_tests.plotting import (
    plot_bias_exchangeability,
    plot_bias_versus_target,
    plot_permutation_null,
)

STATISTICAL_TESTS: dict[str, StatisticalTest] = {
    BiasExchangeabilityTest.name: BiasExchangeabilityTest(),
}


def register_statistical_test(name: str, test: StatisticalTest, *, replace: bool = False) -> None:
    """Register a test object under a stable user-facing name."""
    if name in STATISTICAL_TESTS and not replace:
        raise KeyError(f"a statistical test named {name!r} is already registered")
    STATISTICAL_TESTS[name] = test


def run_statistical_test(name: str, *args: Any, **kwargs: Any) -> StatisticalTestResult:
    """Run a registered test and return a standard result object."""
    try:
        test = STATISTICAL_TESTS[name]
    except KeyError as error:
        available = ", ".join(sorted(STATISTICAL_TESTS))
        raise KeyError(f"unknown statistical test {name!r}; available: {available}") from error
    return test.run(*args, **kwargs)


__all__ = [
    "BiasExchangeabilityTest",
    "STATISTICAL_TESTS",
    "StatisticalTest",
    "StatisticalTestResult",
    "load_lmarena_pseudo_oracle",
    "plot_bias_exchangeability",
    "plot_bias_versus_target",
    "plot_permutation_null",
    "register_statistical_test",
    "run_statistical_test",
    "test_bias_exchangeability",
]
