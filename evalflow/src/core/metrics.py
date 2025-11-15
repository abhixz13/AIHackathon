from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

# Import for type checking only (avoids circular import at runtime)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataframe import EvalDataFrame


class Metric(ABC):
    """Base class for evaluation metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        higher_is_better: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.version = version
        self.higher_is_better = higher_is_better

    @abstractmethod
    def compute(self, eval_df: "EvalDataFrame") -> float:
        """Compute the metric value from an :class:`EvalDataFrame`."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Metric(name={self.name!r}, version={self.version!r})"


class SimpleMetric(Metric):
    """Metric defined by a Python callable.

    Parameters
    ----------
    compute_fn
        Callable that receives two sequences: ``outputs`` and ``targets``,
        extracted from the dataframe using ``output_col`` and
        ``target_col``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        compute_fn: Callable,
        output_col: str = "output",
        target_col: str = "target",
        higher_is_better: bool = True,
        version: str = "1.0.0",
    ) -> None:
        super().__init__(name=name, description=description, version=version, higher_is_better=higher_is_better)
        self._compute_fn = compute_fn
        self.output_col = output_col
        self.target_col = target_col

    def compute(self, eval_df: "EvalDataFrame") -> float:
        df = eval_df.to_pandas()
        if self.output_col not in df.columns or self.target_col not in df.columns:
            return float("nan")
        outputs = df[self.output_col].to_list()
        targets = df[self.target_col].to_list()
        try:
            value = self._compute_fn(outputs, targets)
        except Exception:
            value = float("nan")
        if value is None:
            return float("nan")
        return float(value)


class MetricRegistry:
    """Global registry for evaluation metrics."""

    _instance: Optional["MetricRegistry"] = None

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    @classmethod
    def get_instance(cls) -> "MetricRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Basic operations -------------------------------------------------
    def register(self, metric: Metric) -> None:
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise KeyError(f"Metric '{name}' is not registered") from exc

    def list_metrics(self) -> Dict[str, Metric]:
        return dict(self._metrics)


def register_metric(metric: Metric) -> None:
    """Convenience wrapper to register a metric in the global registry."""

    MetricRegistry.get_instance().register(metric)
