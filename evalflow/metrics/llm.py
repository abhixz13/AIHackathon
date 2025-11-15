"""LLM-specific helper metrics.

These implementations are intentionally simple and rely on common column
conventions used in many evaluation pipelines. If the expected columns
are missing, the metric returns ``NaN`` instead of raising.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.metrics import SimpleMetric, register_metric


def _mean_column(outputs: Sequence, _targets: Sequence) -> float:
    arr = np.asarray(list(outputs), dtype="float64")
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _binary_rate(outputs: Sequence, _targets: Sequence) -> float:
    arr = np.asarray(list(outputs), dtype="float64")
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


# GPT-4 win rate -------------------------------------------------------
register_metric(
    SimpleMetric(
        name="gpt4_win_rate",
        description="Win rate against GPT-4 in pairwise comparisons (expects column 'gpt4_win' in [0,1]).",
        compute_fn=_binary_rate,
        output_col="gpt4_win",
    )
)

# Human preference score -----------------------------------------------
register_metric(
    SimpleMetric(
        name="preference_score",
        description="Average human preference score (expects column 'preference_score').",
        compute_fn=_mean_column,
        output_col="preference_score",
    )
)

# Coherence score ------------------------------------------------------
register_metric(
    SimpleMetric(
        name="coherence_score",
        description="Average coherence score (expects column 'coherence_score').",
        compute_fn=_mean_column,
        output_col="coherence_score",
    )
)

# Toxicity score -------------------------------------------------------
register_metric(
    SimpleMetric(
        name="toxicity_score",
        description="Average toxicity score (expects column 'toxicity_score').",
        compute_fn=_mean_column,
        output_col="toxicity_score",
        higher_is_better=False,
    )
)

# Hallucination rate ---------------------------------------------------
register_metric(
    SimpleMetric(
        name="hallucination_rate",
        description="Rate of hallucinations (expects column 'is_hallucination' in [0,1]).",
        compute_fn=_binary_rate,
        output_col="is_hallucination",
        higher_is_better=False,
    )
)

# Instruction following ------------------------------------------------
register_metric(
    SimpleMetric(
        name="instruction_following",
        description="Average instruction-following score (expects column 'instruction_following').",
        compute_fn=_mean_column,
        output_col="instruction_following",
    )
)
