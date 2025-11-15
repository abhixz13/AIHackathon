from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from .metrics import MetricRegistry


class ColumnExpression:
    """A simple expression object for column operations.

    Instances of this class are callable on a pandas.DataFrame and return
    a pandas.Series. They support comparison operators and logical
    composition so that expressions like::

        df.filter((col("model") == "gpt-4") & (col("accuracy") > 0.8))

    work as expected.
    """

    def __init__(self, func: Callable[[pd.DataFrame], pd.Series]):
        self._func = func

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._func(df)

    # ------------------------------------------------------------------
    # Binary operators
    # ------------------------------------------------------------------
    def _binary_op(self, other: Any, op: Callable) -> "ColumnExpression":
        if isinstance(other, ColumnExpression):
            return ColumnExpression(lambda df: op(self(df), other(df)))
        else:
            return ColumnExpression(lambda df: op(self(df), other))

    def __eq__(self, other: Any) -> "ColumnExpression":  # type: ignore[override]
        import operator

        return self._binary_op(other, operator.eq)

    def __ne__(self, other: Any) -> "ColumnExpression":  # type: ignore[override]
        import operator

        return self._binary_op(other, operator.ne)

    def __gt__(self, other: Any) -> "ColumnExpression":
        import operator

        return self._binary_op(other, operator.gt)

    def __ge__(self, other: Any) -> "ColumnExpression":
        import operator

        return self._binary_op(other, operator.ge)

    def __lt__(self, other: Any) -> "ColumnExpression":
        import operator

        return self._binary_op(other, operator.lt)

    def __le__(self, other: Any) -> "ColumnExpression":
        import operator

        return self._binary_op(other, operator.le)

    # ------------------------------------------------------------------
    # Logical composition
    # ------------------------------------------------------------------
    def __and__(self, other: "ColumnExpression") -> "ColumnExpression":
        return ColumnExpression(lambda df: self(df) & other(df))

    def __or__(self, other: "ColumnExpression") -> "ColumnExpression":
        return ColumnExpression(lambda df: self(df) | other(df))

    def __invert__(self) -> "ColumnExpression":
        return ColumnExpression(lambda df: ~self(df))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def is_in(self, values: Iterable[Any]) -> "ColumnExpression":
        return ColumnExpression(lambda df: self(df).isin(list(values)))

    @property
    def str(self) -> "StringMethods":
        return StringMethods(self)


class StringMethods:
    """String helpers for :class:`ColumnExpression`."""

    def __init__(self, expr: ColumnExpression):
        self._expr = expr

    def contains(self, pattern: str, case: bool = True, na: bool = False) -> ColumnExpression:
        def _func(df: pd.DataFrame) -> pd.Series:
            return (
                self._expr(df)
                .astype(str)
                .str.contains(pattern, case=case, na=na)
            )

        return ColumnExpression(_func)


def col(name: str) -> ColumnExpression:
    """Create a column expression referring to *name*.

    This is the main entrypoint for building expressions in filters and
    transformations.
    """

    return ColumnExpression(lambda df: df[name])


DataLike = Union[pd.DataFrame, "EvalDataFrame"]


class EvalDataFrame:
    """High-level wrapper around :class:`pandas.DataFrame`.

    The real EvalFlow library is built on Daft for distributed, lazy
    execution. In this reference implementation we use pandas for
    simplicity while keeping the public API shape the same.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
        else:
            self._df = pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "EvalDataFrame":
        return cls(df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()

    # ------------------------------------------------------------------
    # Core dataframe-style operations
    # ------------------------------------------------------------------
    def filter(self, predicate: ColumnExpression) -> "EvalDataFrame":
        mask = predicate(self._df)
        return EvalDataFrame(self._df[mask])

    def select(self, *columns: str) -> "EvalDataFrame":
        cols = list(columns)
        return EvalDataFrame(self._df[cols])

    def with_column(self, column_name: str, expression: Union[ColumnExpression, Any]) -> "EvalDataFrame":
        if isinstance(expression, ColumnExpression):
            self._df[column_name] = expression(self._df)
        else:
            self._df[column_name] = expression
        return EvalDataFrame(self._df)

    def group_by(self, *columns: str) -> "GroupedEvalDataFrame":
        return GroupedEvalDataFrame(self._df, list(columns))

    def join(
        self,
        other: "EvalDataFrame",
        on: Union[str, Sequence[str]],
        how: str = "inner",
    ) -> "EvalDataFrame":
        left = self._df
        right = other._df
        result = left.merge(right, on=on, how=how)
        return EvalDataFrame(result)

    def sort(self, *columns: str, desc: bool = False) -> "EvalDataFrame":
        ascending = not desc
        result = self._df.sort_values(list(columns), ascending=ascending)
        return EvalDataFrame(result)

    # ------------------------------------------------------------------
    # Evaluation-centric helpers
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        metrics: Union[str, List[str], Dict[str, str]],
        group_by: Optional[List[str]] = None,
    ) -> "EvalDataFrame":
        """Compute one or more metrics.

        Parameters
        ----------
        metrics
            Either a single metric name, a list of metric names, or a
            mapping from output column name to metric name.
        group_by
            Optional list of columns to group by before computing metrics.
        """

        registry = MetricRegistry.get_instance()

        if isinstance(metrics, str):
            metric_names = [metrics]
        elif isinstance(metrics, dict):
            metric_names = list(metrics.values())
        else:
            metric_names = list(metrics)

        if group_by:
            grouped = self._df.groupby(group_by, dropna=False)
            rows: List[Dict[str, Any]] = []
            for group_key, group_df in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row: Dict[str, Any] = {}
                for col_name, value in zip(group_by, group_key):
                    row[col_name] = value
                tmp_eval = EvalDataFrame(group_df)
                for name in metric_names:
                    metric = registry.get(name)
                    row[name] = metric.compute(tmp_eval)
                rows.append(row)
            return EvalDataFrame(pd.DataFrame(rows))
        else:
            tmp_eval = EvalDataFrame(self._df)
            row = {name: MetricRegistry.get_instance().get(name).compute(tmp_eval) for name in metric_names}
            return EvalDataFrame(pd.DataFrame([row]))

    def compare_models(
        self,
        model_col: str = "model",
        metric_col: str = "accuracy",
        baseline: Optional[str] = None,
    ) -> "EvalDataFrame":
        """Compare models on a given metric.

        This is a convenience helper that computes the mean value of
        *metric_col* per *model_col* and optionally adds a ``lift`` column
        relative to a baseline model.
        """

        df = self._df
        summary = df.groupby(model_col, dropna=False)[metric_col].mean().reset_index()
        summary = summary.rename(columns={metric_col: f"{metric_col}_mean"})
        if baseline is not None and baseline in summary[model_col].values:
            base_val = float(summary.loc[summary[model_col] == baseline, f"{metric_col}_mean"].iloc[0])
            summary[f"{metric_col}_lift_vs_{baseline}"] = summary[f"{metric_col}_mean"] - base_val
        return EvalDataFrame(summary)

    def filter_experiments(self, experiment_ids: Union[str, List[str]]) -> "EvalDataFrame":
        if isinstance(experiment_ids, str):
            ids = [experiment_ids]
        else:
            ids = list(experiment_ids)
        if "experiment_id" not in self._df.columns:
            raise KeyError("Column 'experiment_id' not found in dataframe.")
        return EvalDataFrame(self._df[self._df["experiment_id"].isin(ids)])

    def filter_benchmarks(self, benchmarks: Union[str, List[str]]) -> "EvalDataFrame":
        if isinstance(benchmarks, str):
            bms = [benchmarks]
        else:
            bms = list(benchmarks)
        if "benchmark" not in self._df.columns:
            raise KeyError("Column 'benchmark' not found in dataframe.")
        return EvalDataFrame(self._df[self._df["benchmark"].isin(bms)])

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def collect(self) -> pd.DataFrame:
        """Trigger computation and return a pandas DataFrame.

        In a real distributed implementation this would trigger the
        underlying execution graph. Here it simply returns a copy of the
        in-memory dataframe.
        """

        return self.to_pandas()

    def write_parquet(self, path: str) -> None:
        self._df.to_parquet(path, index=False)

    def write_csv(self, path: str) -> None:
        self._df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    def count(self) -> int:
        return int(len(self._df))

    def columns(self) -> List[str]:
        return list(self._df.columns)

    def describe(self) -> pd.DataFrame:
        return self._df.describe(include="all")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, *args: Any, **kwargs: Any):
        """Quick plotting helper.

        This uses pandas' built-in plotting and returns a :class:`Plot`
        wrapper from :mod:`evalflow.viz` for consistency with the rest of
        the API.
        """

        import matplotlib.pyplot as plt
        from ..viz.plots import Plot

        ax = self._df.plot(*args, **kwargs)
        fig = ax.get_figure()
        return Plot(fig, ax)

    # Representation helpers
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"EvalDataFrame(shape={self._df.shape}, columns={list(self._df.columns)})"


@dataclass
class GroupedEvalDataFrame:
    """Grouped variant of :class:`EvalDataFrame` for aggregations."""

    _df: pd.DataFrame
    _group_cols: List[str]

    def agg(self, aggregations: Dict[str, Union[str, List[str]]]) -> EvalDataFrame:
        grouped = self._df.groupby(self._group_cols, dropna=False).agg(aggregations)
        # Flatten multi-index columns if present
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                "{}_{}".format(col, agg) for col, agg in grouped.columns.to_flat_index()
            ]
        grouped = grouped.reset_index()
        return EvalDataFrame(grouped)

    def compute_metrics(self, metrics: List[str]) -> EvalDataFrame:
        eval_df = EvalDataFrame(self._df)
        return eval_df.compute_metrics(metrics, group_by=self._group_cols)
