from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .dataframe import EvalDataFrame, ColumnExpression


TransformFn = Callable[[EvalDataFrame], EvalDataFrame]


@dataclass
class PipelineStep:
    name: str
    transform: TransformFn


@dataclass
class Pipeline:
    """Composable pipeline for evaluation data transformations."""

    name: Optional[str] = None
    _steps: List[PipelineStep] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------
    def add_step(self, name: str, transform: TransformFn) -> "Pipeline":
        self._steps.append(PipelineStep(name=name, transform=transform))
        return self

    def filter_step(self, name: str, predicate: ColumnExpression) -> "Pipeline":
        def _transform(df: EvalDataFrame) -> EvalDataFrame:
            return df.filter(predicate)

        return self.add_step(name, _transform)

    def transform_step(self, name: str, column: str, expression: ColumnExpression) -> "Pipeline":
        def _transform(df: EvalDataFrame) -> EvalDataFrame:
            return df.with_column(column, expression)

        return self.add_step(name, _transform)

    def metric_step(self, name: str, metrics, group_by=None) -> "Pipeline":
        def _transform(df: EvalDataFrame) -> EvalDataFrame:
            return df.compute_metrics(metrics, group_by=group_by)

        return self.add_step(name, _transform)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, df: EvalDataFrame) -> EvalDataFrame:
        current = df
        for step in self._steps:
            current = step.transform(current)
        return current

    def describe(self) -> str:
        """Human-readable description of the pipeline steps."""

        lines = [f"Pipeline(name={self.name!r}, steps=["]
        for step in self._steps:
            lines.append(f"  - {step.name}")
        lines.append("])")
        return "\n".join(lines)
