"""Utilities for stock-specific executable trading-day calculations."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def executable_trade_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for rows that look executable for a single stock.

    The local CSVs occasionally contain zero-volume or zero-amount rows with
    placeholder prices. Those rows should not count as executable holding days.
    """
    mask = pd.Series(True, index=df.index)

    for col in ("open", "close", "high", "low"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mask &= values.notna() & (values > 0)

    for col in ("volume", "amount"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mask &= values.notna() & (values > 0)

    return mask.fillna(False)


def executable_trade_indices(df: pd.DataFrame) -> np.ndarray:
    """Return row indices that qualify as executable trade days."""
    return np.flatnonzero(executable_trade_mask(df).to_numpy())


def nth_executable_trade_index(
    executable_indices: Sequence[int],
    anchor_idx: int,
    offset: int = 0,
    *,
    include_anchor: bool = False,
) -> Optional[int]:
    """
    Resolve the row index of the target executable trade day.

    Args:
        executable_indices: Monotonic row indices for executable days.
        anchor_idx: Reference row index.
        offset: Number of executable days to move forward from the first match.
        include_anchor: When True, an executable anchor row counts as offset 0.

    Returns:
        Row index or None if the target day does not exist.
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")

    exec_idx = np.asarray(executable_indices, dtype=np.int64)
    if exec_idx.size == 0:
        return None

    side = "left" if include_anchor else "right"
    pos = int(np.searchsorted(exec_idx, anchor_idx, side=side))
    target_pos = pos + offset
    if target_pos >= exec_idx.size:
        return None
    return int(exec_idx[target_pos])
