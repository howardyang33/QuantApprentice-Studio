"""Smoke tests for Pilot 2.1 walk-forward helper functions."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit.pilot2.walkforward_utils import assign_prediction_buckets, compute_train_thresholds


def test_compute_train_thresholds_returns_four_ordered_cutoffs():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    thresholds = compute_train_thresholds(scores)

    assert thresholds.shape == (4,)
    assert np.all(np.diff(thresholds) >= 0)
    assert thresholds[0] == pytest.approx(1.8)
    assert thresholds[-1] == pytest.approx(4.2)


def test_assign_prediction_buckets_uses_train_thresholds_only():
    thresholds = np.array([10.0, 20.0, 30.0, 40.0])
    scores = np.array([5.0, 10.0, 19.0, 20.0, 39.0, 40.0, 50.0])

    buckets = assign_prediction_buckets(scores, thresholds)

    assert buckets.tolist() == [1, 2, 2, 3, 4, 5, 5]
