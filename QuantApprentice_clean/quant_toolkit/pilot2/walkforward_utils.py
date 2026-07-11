"""Utilities for Pilot 2.1 yearly walk-forward bucket evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

TRAIN_BUCKET_QUANTILES = (0.2, 0.4, 0.6, 0.8)


def compute_train_thresholds(
    train_scores: Iterable[float],
    *,
    quantiles: tuple[float, float, float, float] = TRAIN_BUCKET_QUANTILES,
) -> np.ndarray:
    """Compute the four train-derived bucket thresholds for Q1~Q5."""
    scores = np.asarray(train_scores, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("train_scores must contain at least one finite value")
    thresholds = np.quantile(scores, quantiles).astype(float)
    return thresholds


def assign_prediction_buckets(scores: Iterable[float], thresholds: Iterable[float]) -> np.ndarray:
    """Assign scores to Q1~Q5 using train-derived thresholds only."""
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    threshold_array = np.asarray(thresholds, dtype=float).reshape(-1)
    if threshold_array.shape != (4,):
        raise ValueError("thresholds must contain exactly four values")
    if not np.all(np.isfinite(threshold_array)):
        raise ValueError("thresholds must be finite")
    return np.searchsorted(threshold_array, score_array, side="right").astype(np.int16) + 1
