"""EvalFlow - Model Evaluation & Analytics Platform

This is a lightweight, self-contained reference implementation of the
EvalFlow API described in the accompanying documentation. It focuses on
providing a clean, easy-to-use interface for model evaluation analytics
on top of pandas, while keeping the public surface area consistent with
the documented design.
"""

from .core.dataframe import EvalDataFrame, GroupedEvalDataFrame, col
from .core.pipeline import Pipeline
from .io.loaders import (
    load_results,
    load_from_parquet,
    load_from_csv,
    load_from_json,
    load_from_pandas,
    load_from_dict,
    load_experiment_results,
    load_benchmark_results,
)
from .metrics import (
    # string constants for metric names
    accuracy,
    precision,
    recall,
    f1_score,
    bleu_score,
    rouge_score,
    perplexity,
    gpt4_win_rate,
    preference_score,
    coherence_score,
    toxicity_score,
    hallucination_rate,
    instruction_following,
)
from .viz.plots import (
    Plot,
    plot_comparison,
    plot_trends,
    plot_scatter,
    plot_heatmap,
    plot_distribution,
)

__all__ = [
    # core
    "EvalDataFrame",
    "GroupedEvalDataFrame",
    "Pipeline",
    "col",
    # loaders
    "load_results",
    "load_from_parquet",
    "load_from_csv",
    "load_from_json",
    "load_from_pandas",
    "load_from_dict",
    "load_experiment_results",
    "load_benchmark_results",
    # metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "bleu_score",
    "rouge_score",
    "perplexity",
    "gpt4_win_rate",
    "preference_score",
    "coherence_score",
    "toxicity_score",
    "hallucination_rate",
    "instruction_following",
    # viz
    "Plot",
    "plot_comparison",
    "plot_trends",
    "plot_scatter",
    "plot_heatmap",
    "plot_distribution",
]
