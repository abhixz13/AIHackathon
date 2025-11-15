"""Standard scalar evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.metrics import SimpleMetric, register_metric


def _safe_arrays(outputs: Sequence, targets: Sequence):
    o = np.asarray(list(outputs))
    t = np.asarray(list(targets))
    mask = ~(np.isnan(o.astype("float64", copy=False, casting="unsafe", subok=False)) |  # type: ignore
             np.isnan(t.astype("float64", copy=False, casting="unsafe", subok=False)))  # type: ignore
    try:
        return o[mask], t[mask]
    except Exception:
        # Fallback for non-numeric labels
        return o, t


def _accuracy(outputs, targets) -> float:
    o, t = _safe_arrays(outputs, targets)
    if len(o) == 0:
        return float("nan")
    return float((o == t).mean())


def _precision(outputs, targets) -> float:
    o, t = _safe_arrays(outputs, targets)
    if len(o) == 0:
        return float("nan")
    positive = 1
    tp = ((o == positive) & (t == positive)).sum()
    fp = ((o == positive) & (t != positive)).sum()
    if tp + fp == 0:
        return float("nan")
    return float(tp / (tp + fp))


def _recall(outputs, targets) -> float:
    o, t = _safe_arrays(outputs, targets)
    if len(o) == 0:
        return float("nan")
    positive = 1
    tp = ((o == positive) & (t == positive)).sum()
    fn = ((o != positive) & (t == positive)).sum()
    if tp + fn == 0:
        return float("nan")
    return float(tp / (tp + fn))


def _f1(outputs, targets) -> float:
    p = _precision(outputs, targets)
    r = _recall(outputs, targets)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def _bleu(outputs, targets) -> float:
    # Extremely small and naive BLEU-style score: unigram overlap
    scores = []
    for o, t in zip(outputs, targets):
        o_tokens = str(o).split()
        t_tokens = str(t).split()
        if not o_tokens or not t_tokens:
            continue
        overlap = len(set(o_tokens) & set(t_tokens))
        scores.append(overlap / len(o_tokens))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _rouge(outputs, targets) -> float:
    # Naive ROUGE-1 recall
    scores = []
    for o, t in zip(outputs, targets):
        o_tokens = str(o).split()
        t_tokens = str(t).split()
        if not o_tokens or not t_tokens:
            continue
        overlap = len(set(o_tokens) & set(t_tokens))
        scores.append(overlap / len(t_tokens))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _perplexity(outputs, targets) -> float:
    # Assume `outputs` is a sequence of negative log likelihoods
    import math

    losses = [float(x) for x in outputs if x is not None]
    if not losses:
        return float("nan")
    mean_nll = sum(losses) / len(losses)
    return float(math.exp(mean_nll))


# Register default metrics in the global registry
register_metric(
    SimpleMetric(
        name="accuracy",
        description="Classification accuracy (outputs == targets).",
        compute_fn=_accuracy,
    )
)

register_metric(
    SimpleMetric(
        name="precision",
        description="Binary precision assuming positive label 1.",
        compute_fn=_precision,
    )
)

register_metric(
    SimpleMetric(
        name="recall",
        description="Binary recall assuming positive label 1.",
        compute_fn=_recall,
    )
)

register_metric(
    SimpleMetric(
        name="f1_score",
        description="Binary F1 score assuming positive label 1.",
        compute_fn=_f1,
    )
)

register_metric(
    SimpleMetric(
        name="bleu_score",
        description="Naive unigram BLEU-style score.",
        compute_fn=_bleu,
    )
)

register_metric(
    SimpleMetric(
        name="rouge_score",
        description="Naive ROUGE-1-style score.",
        compute_fn=_rouge,
    )
)

register_metric(
    SimpleMetric(
        name="perplexity",
        description="Perplexity from negative log-likelihoods.",
        compute_fn=_perplexity,
        output_col="nll",
    )
)
