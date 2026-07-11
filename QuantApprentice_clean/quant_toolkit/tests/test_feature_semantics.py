"""Semantic guards for Pilot 1.5 contract-sensitive behavior."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit.features.factor_utils import calculate_market_cap_features
from quant_toolkit.signals.kd_strategy import construct_samples


def test_construct_samples_uses_executable_trade_days():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
        ]),
        "open": [10.0, 10.1, 0.0, 10.5, 10.8],
        "close": [10.1, 9.9, 9.9, 10.7, 11.0],
        "high": [10.2, 10.2, 0.0, 10.8, 11.1],
        "low": [9.9, 9.8, 0.0, 10.4, 10.7],
        "volume": [1000.0, 1200.0, 0.0, 1400.0, 1600.0],
        "amount": [10100.0, 11880.0, 0.0, 14980.0, 17600.0],
        "turnover": [0.01, 0.012, 0.0, 0.014, 0.016],
        "J": [20.0, 10.0, 9.0, 15.0, 18.0],
    })

    samples = construct_samples(df, "000001", holding_days=2)

    assert len(samples) == 1
    sample = samples[0]
    assert sample["signal_date"] == pd.Timestamp("2020-01-02")
    assert sample["entry_date"] == pd.Timestamp("2020-01-06")
    assert sample["exit_date"] == pd.Timestamp("2020-01-07")
    assert sample["holding_days"] == 2


def test_market_cap_features_accept_fractional_turnover():
    df = pd.DataFrame({
        "close": [10.0, 10.0, 10.0],
        "volume": [1000.0, 1000.0, 1000.0],
        "turnover": [0.1, 0.1, 0.1],
    })

    out = calculate_market_cap_features(df)
    expected_float_mcap = 10.0 * 1000.0 / 0.1

    assert np.isclose(out.loc[0, "log_market_cap"], np.log(expected_float_mcap))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
