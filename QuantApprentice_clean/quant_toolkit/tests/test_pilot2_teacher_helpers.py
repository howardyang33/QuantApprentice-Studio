"""Smoke tests for Pilot 2 first-teacher helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit.pilot2.teacher_utils import (
    REVERSAL_FEATURE_COLUMNS,
    build_executable_label_frame,
    compute_reversal_features,
)


def _make_base_df(n_rows: int = 130) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    close = np.linspace(10.0, 12.5, n_rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "close": close,
            "high": close + 0.10,
            "low": close - 0.15,
            "volume": np.linspace(1000.0, 2000.0, n_rows),
            "amount": np.linspace(10000.0, 25000.0, n_rows),
        }
    )


def test_build_executable_label_frame_uses_stock_specific_executable_days():
    df = _make_base_df()
    df.loc[121, ["open", "high", "low", "close", "volume", "amount"]] = 0.0

    labels = build_executable_label_frame(df, min_history=120, horizon=5)

    assert len(labels) > 0
    first = labels.iloc[0]
    assert first["signal_date"] == df.loc[120, "date"]
    assert first["entry_date"] == df.loc[122, "date"]
    assert first["exit_date"] == df.loc[126, "date"]

    expected = df.loc[126, "close"] / df.loc[122, "open"] - 1.0
    assert first["future_return_5d"] == pytest.approx(expected)


def test_compute_reversal_features_produces_selected_columns():
    df = _make_base_df(n_rows=40)
    features = compute_reversal_features(df)

    for col in REVERSAL_FEATURE_COLUMNS:
        assert col in features.columns
        assert np.isfinite(features[col].iloc[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
