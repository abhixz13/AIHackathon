from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from ..core.dataframe import EvalDataFrame


@dataclass
class Plot:
    """Lightweight wrapper around a Matplotlib figure/axes pair."""

    fig: "plt.Figure"
    ax: "plt.Axes"

    def save(self, path: str) -> None:
        self.fig.tight_layout()
        self.fig.savefig(path)


def _to_pandas(eval_df) -> pd.DataFrame:
    if isinstance(eval_df, EvalDataFrame):
        return eval_df.to_pandas()
    elif isinstance(eval_df, pd.DataFrame):
        return eval_df.copy()
    else:
        raise TypeError("Expected EvalDataFrame or pandas.DataFrame")


def plot_comparison(
    eval_df,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str = "Model Comparison",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Plot:
    df = _to_pandas(eval_df)
    if x is None or y is None:
        raise ValueError("Both x and y must be provided for plot_comparison")
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[x], df[y], **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_xticklabels(df[x], rotation=45, ha="right")
    return Plot(fig, ax)


def plot_trends(
    eval_df,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: str = "Performance Trends",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Plot:
    df = _to_pandas(eval_df)
    if x is None or y is None:
        raise ValueError("Both x and y must be provided for plot_trends")
    fig, ax = plt.subplots(figsize=figsize)
    if hue is None:
        ax.plot(df[x], df[y], marker="o", **kwargs)
    else:
        for key, group in df.groupby(hue):
            ax.plot(group[x], group[y], marker="o", label=str(key), **kwargs)
        ax.legend(title=hue)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    return Plot(fig, ax)


def plot_scatter(
    eval_df,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: str = "Scatter Plot",
    figsize: Tuple[int, int] = (8, 6),
    **kwargs,
) -> Plot:
    df = _to_pandas(eval_df)
    fig, ax = plt.subplots(figsize=figsize)
    if hue is None:
        ax.scatter(df[x], df[y], **kwargs)
    else:
        for key, group in df.groupby(hue):
            ax.scatter(group[x], group[y], label=str(key), **kwargs)
        ax.legend(title=hue)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return Plot(fig, ax)


def plot_heatmap(
    eval_df,
    index: str,
    columns: str,
    values: str,
    title: str = "Heatmap",
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Plot:
    import numpy as np

    df = _to_pandas(eval_df)
    pivot = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(pivot.values, aspect="auto", **kwargs)
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(cax, ax=ax)
    return Plot(fig, ax)


def plot_distribution(
    eval_df,
    column: str,
    bins: int = 30,
    title: str = "Distribution",
    figsize: Tuple[int, int] = (8, 6),
    **kwargs,
) -> Plot:
    df = _to_pandas(eval_df)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[column], bins=bins, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    return Plot(fig, ax)
