"""Common interfaces and result objects for statistical diagnostics in PAS."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class StatisticalTestResult:
    """Standard output shared by PAS statistical tests.

    ``null_distribution`` is retained for visualization but omitted from the
    default JSON summary to keep result files compact.
    """

    name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    alternative: str
    method: str
    n_observations: int
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    null_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.statistic = float(self.statistic)
        self.p_value = float(self.p_value)
        self.n_observations = int(self.n_observations)
        if self.effect_size is not None:
            self.effect_size = float(self.effect_size)
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("p_value must lie in [0, 1]")
        if self.n_observations <= 0:
            raise ValueError("n_observations must be positive")
        if self.null_distribution is not None:
            self.null_distribution = np.asarray(
                self.null_distribution, dtype=float
            ).reshape(-1)

    def to_dict(self, *, include_null_distribution: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "null_hypothesis": self.null_hypothesis,
            "alternative": self.alternative,
            "method": self.method,
            "n_observations": self.n_observations,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "metadata": self.metadata,
        }
        if include_null_distribution and self.null_distribution is not None:
            output["null_distribution"] = self.null_distribution
        return _jsonable(output)

    def to_json(
        self,
        path: Optional[Path] = None,
        *,
        include_null_distribution: bool = False,
        indent: int = 2,
    ) -> str:
        text = json.dumps(
            self.to_dict(include_null_distribution=include_null_distribution),
            indent=indent,
        )
        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
        return text

    def format(self, digits: int = 4) -> str:
        effect = ""
        if self.effect_size is not None:
            label = self.effect_size_name or "effect size"
            effect = f"\n{label}: {self.effect_size:.{digits}g}"
        return (
            f"{self.name}\n"
            f"method: {self.method}\n"
            f"null: {self.null_hypothesis}\n"
            f"statistic: {self.statistic:.{digits}g}\n"
            f"p-value: {self.p_value:.{digits}g}"
            f"{effect}\n"
            f"n: {self.n_observations}"
        )

    def __str__(self) -> str:
        return self.format()


@runtime_checkable
class StatisticalTest(Protocol):
    """Protocol for test objects accepted by the PAS test registry."""

    name: str

    def run(self, *args: Any, **kwargs: Any) -> StatisticalTestResult:
        ...


__all__ = ["StatisticalTest", "StatisticalTestResult"]
