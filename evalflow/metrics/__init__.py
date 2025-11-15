"""Metric name constants and default registrations."""

from ..core.metrics import MetricRegistry, register_metric, SimpleMetric
from . import standard as _standard  # noqa: F401
from . import llm as _llm  # noqa: F401

# Metric name constants (useful for autocompletion)
accuracy = "accuracy"
precision = "precision"
recall = "recall"
f1_score = "f1_score"
bleu_score = "bleu_score"
rouge_score = "rouge_score"
perplexity = "perplexity"

gpt4_win_rate = "gpt4_win_rate"
preference_score = "preference_score"
coherence_score = "coherence_score"
toxicity_score = "toxicity_score"
hallucination_rate = "hallucination_rate"
instruction_following = "instruction_following"

__all__ = [
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
]
