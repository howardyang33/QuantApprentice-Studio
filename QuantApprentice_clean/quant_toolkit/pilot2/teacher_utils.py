"""Pilot 2 contract-safe helpers for the first reversal teacher."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..data.trading_calendar import executable_trade_indices
from ..features.technical import calculate_kdj, calculate_kdj_derivatives, compute_technical_features

REVERSAL_FEATURE_COLUMNS = [
    "K",
    "D",
    "J",
    "dJ_1",
    "dJ_3",
    "J_minus_D",
    "oversold_depth",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "close_to_ma5",
    "close_to_ma10",
    "close_to_ma20",
    "pos_20",
    "body_pct",
    "amplitude",
    "lower_shadow",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "dist_to_20d_high",
    "dist_to_20d_low",
    "volume_ma5_ratio",
    "volume_ma20_ratio",
    "volume_zscore_20",
    "amt_ma5_ratio",
    "amt_ma20_ratio",
    "amt_zscore_20",
]


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def compute_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the selected Pilot 2 stock-only reversal features.

    All features are computed from the current row and its history only.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out = _ensure_numeric(out, ["open", "high", "low", "close", "volume", "amount"])

    out = calculate_kdj(out)
    out = calculate_kdj_derivatives(out)
    out = compute_technical_features(out)

    rolling_high_20 = out["high"].rolling(20, min_periods=1).max()
    rolling_low_20 = out["low"].rolling(20, min_periods=1).min()
    out["dist_to_20d_high"] = out["close"] / (rolling_high_20 + 1e-12) - 1.0
    out["dist_to_20d_low"] = out["close"] / (rolling_low_20 + 1e-12) - 1.0

    volume_ma5 = out["volume"].rolling(5, min_periods=1).mean()
    volume_ma20 = out["volume"].rolling(20, min_periods=1).mean()
    volume_std20 = out["volume"].rolling(20, min_periods=1).std()
    out["volume_ma5_ratio"] = out["volume"] / (volume_ma5 + 1e-12)
    out["volume_ma20_ratio"] = out["volume"] / (volume_ma20 + 1e-12)
    out["volume_zscore_20"] = (out["volume"] - volume_ma20) / (volume_std20 + 1e-12)

    for col in REVERSAL_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        arr = out[col].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        arr[~np.isfinite(arr)] = 0.0
        out[col] = arr

    return out


def build_executable_label_frame(
    df: pd.DataFrame,
    *,
    min_history: int = 120,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Build Pilot 2 labels from stock-specific executable trading days.

    `signal_date` is the current executable row, `entry_date` is the next
    executable row, and `exit_date` is the `horizon`th executable row counting
    entry as day 1.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out = _ensure_numeric(out, ["open", "close", "high", "low", "volume", "amount"])
    executable_indices = executable_trade_indices(out)

    if len(executable_indices) <= (min_history + horizon):
        return pd.DataFrame(
            columns=[
                "signal_index",
                "entry_index",
                "exit_index",
                "signal_date",
                "entry_date",
                "exit_date",
                "entry_open",
                "exit_close",
                "future_return_5d",
            ]
        )

    rows = []
    for exec_pos in range(min_history, len(executable_indices) - horizon):
        signal_idx = int(executable_indices[exec_pos])
        entry_idx = int(executable_indices[exec_pos + 1])
        exit_idx = int(executable_indices[exec_pos + horizon])

        entry_open = float(out.iloc[entry_idx]["open"])
        exit_close = float(out.iloc[exit_idx]["close"])
        if not np.isfinite(entry_open) or not np.isfinite(exit_close):
            continue
        if entry_open <= 0 or exit_close <= 0:
            continue

        rows.append(
            {
                "signal_index": signal_idx,
                "entry_index": entry_idx,
                "exit_index": exit_idx,
                "signal_date": out.iloc[signal_idx]["date"],
                "entry_date": out.iloc[entry_idx]["date"],
                "exit_date": out.iloc[exit_idx]["date"],
                "entry_open": entry_open,
                "exit_close": exit_close,
                "future_return_5d": exit_close / entry_open - 1.0,
            }
        )

    return pd.DataFrame(rows)


def build_reversal_candidate_mask(df: pd.DataFrame) -> pd.Series:
    """
    Keep the first teacher focused on short-horizon weak-state observations.
    """
    mask = (df["J"] <= 35.0) | ((df["ret_3"] <= -0.03) & (df["pos_20"] <= 0.35))
    return mask.fillna(False)


def compute_reversal_threshold_score(df: pd.DataFrame) -> pd.Series:
    """
    Simple hard-threshold score aligned with the reversal hypothesis.

    The threshold hit dominates; small tie-break terms preserve an ordering for
    decile analysis without turning the baseline into a complex model.
    """
    threshold_hit = (
        (df["J"] <= 15.0)
        & (df["ret_5"] <= -0.04)
        & (df["pos_20"] <= 0.20)
        & (df["amt_zscore_20"] >= 0.0)
    )

    score = np.where(threshold_hit, 100.0, 0.0)
    score = score + np.clip(-df["ret_5"].to_numpy(dtype=float), 0.0, None) * 10.0
    score = score + np.clip(20.0 - df["J"].to_numpy(dtype=float), 0.0, None) / 20.0
    score = score - np.clip(df["dist_to_20d_low"].to_numpy(dtype=float), 0.0, None)
    return pd.Series(score, index=df.index, name="hard_threshold_score")
