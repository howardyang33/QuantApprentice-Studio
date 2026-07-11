"""Feature registry utilities for autonomous teacher construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..features.technical import get_v1_feature_columns

BASE_FEATURE_NAMES: List[str] = get_v1_feature_columns()
DERIVED_FEATURE_NAMES: List[str] = [
    "dist_to_20d_high",
    "dist_to_20d_low",
    "volume_ma5_ratio",
    "volume_ma20_ratio",
    "volume_zscore_20",
    "ret_1_clip",
    "ret_3_clip",
    "gap_pct",
    "volume_log",
    "close_log",
]
ALL_REGISTERED_FEATURES: List[str] = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES

_BASE_FEATURE_META: Dict[str, Dict[str, Any]] = {
    "ret_1": {"category": "returns", "formula": "close.pct_change(1)", "lookback": 1},
    "ret_3": {"category": "returns", "formula": "close.pct_change(3)", "lookback": 3},
    "ret_5": {"category": "returns", "formula": "close.pct_change(5)", "lookback": 5},
    "ret_10": {"category": "returns", "formula": "close.pct_change(10)", "lookback": 10},
    "ret_20": {"category": "returns", "formula": "close.pct_change(20)", "lookback": 20},
    "ret_60": {"category": "returns", "formula": "close.pct_change(60)", "lookback": 60},
    "close_to_ma5": {"category": "moving_average", "formula": "close / ma5", "lookback": 5},
    "close_to_ma10": {"category": "moving_average", "formula": "close / ma10", "lookback": 10},
    "close_to_ma20": {"category": "moving_average", "formula": "close / ma20", "lookback": 20},
    "close_to_ma60": {"category": "moving_average", "formula": "close / ma60", "lookback": 60},
    "pos_20": {"category": "position", "formula": "(close-min20)/(max20-min20)", "lookback": 20},
    "pos_60": {"category": "position", "formula": "(close-min60)/(max60-min60)", "lookback": 60},
    "K": {"category": "kdj", "formula": "KDJ K line", "lookback": 9},
    "D": {"category": "kdj", "formula": "KDJ D line", "lookback": 9},
    "J": {"category": "kdj", "formula": "KDJ J line", "lookback": 9},
    "dK_1": {"category": "kdj", "formula": "K.diff(1)", "lookback": 10},
    "dD_1": {"category": "kdj", "formula": "D.diff(1)", "lookback": 10},
    "dJ_1": {"category": "kdj", "formula": "J.diff(1)", "lookback": 10},
    "dJ_3": {"category": "kdj", "formula": "J.diff(3)", "lookback": 12},
    "J_minus_D": {"category": "kdj", "formula": "J-D", "lookback": 9},
    "oversold_depth": {"category": "kdj", "formula": "clip(13-J, lower=0)", "lookback": 9},
    "days_J_below_20_last_10": {"category": "kdj", "formula": "count(J<20, last10)", "lookback": 10},
    "days_J_below_10_last_10": {"category": "kdj", "formula": "count(J<10, last10)", "lookback": 10},
    "body_pct": {"category": "candle", "formula": "(close-open)/open", "lookback": 1},
    "amplitude": {"category": "candle", "formula": "(high-low)/prev_close", "lookback": 2},
    "upper_shadow": {
        "category": "candle",
        "formula": "(high-max(close,open))/prev_close",
        "lookback": 2,
    },
    "lower_shadow": {
        "category": "candle",
        "formula": "(min(close,open)-low)/prev_close",
        "lookback": 2,
    },
    "volatility_5": {"category": "volatility", "formula": "std(ret1, 5)", "lookback": 5},
    "volatility_10": {"category": "volatility", "formula": "std(ret1, 10)", "lookback": 10},
    "volatility_20": {"category": "volatility", "formula": "std(ret1, 20)", "lookback": 20},
    "vol_ratio_5_20": {"category": "volatility", "formula": "volatility_5/volatility_20", "lookback": 20},
    "amt_log": {"category": "amount", "formula": "log1p(amount)", "lookback": 1},
    "amt_ma5_ratio": {"category": "amount", "formula": "amount/ma(amount,5)", "lookback": 5},
    "amt_ma20_ratio": {"category": "amount", "formula": "amount/ma(amount,20)", "lookback": 20},
    "amt_zscore_20": {"category": "amount", "formula": "zscore(amount,20)", "lookback": 20},
}

_DERIVED_FEATURE_META: Dict[str, Dict[str, Any]] = {
    "dist_to_20d_high": {
        "category": "position",
        "formula": "close / rolling_high_20 - 1",
        "lookback": 20,
        "depends_on": ["close", "high"],
    },
    "dist_to_20d_low": {
        "category": "position",
        "formula": "close / rolling_low_20 - 1",
        "lookback": 20,
        "depends_on": ["close", "low"],
    },
    "volume_ma5_ratio": {
        "category": "volume",
        "formula": "volume / ma(volume,5)",
        "lookback": 5,
        "depends_on": ["volume"],
    },
    "volume_ma20_ratio": {
        "category": "volume",
        "formula": "volume / ma(volume,20)",
        "lookback": 20,
        "depends_on": ["volume"],
    },
    "volume_zscore_20": {
        "category": "volume",
        "formula": "zscore(volume,20)",
        "lookback": 20,
        "depends_on": ["volume"],
    },
    "ret_1_clip": {
        "category": "returns",
        "formula": "clip(ret_1, -0.20, 0.20)",
        "lookback": 1,
        "depends_on": ["ret_1"],
    },
    "ret_3_clip": {
        "category": "returns",
        "formula": "clip(ret_3, -0.30, 0.30)",
        "lookback": 3,
        "depends_on": ["ret_3"],
    },
    "gap_pct": {
        "category": "candle",
        "formula": "open/prev_close - 1",
        "lookback": 2,
        "depends_on": ["open", "close"],
    },
    "volume_log": {
        "category": "volume",
        "formula": "log1p(volume)",
        "lookback": 1,
        "depends_on": ["volume"],
    },
    "close_log": {
        "category": "price_level",
        "formula": "log(close)",
        "lookback": 1,
        "depends_on": ["close"],
    },
}


def _feature_card(name: str, meta: Dict[str, Any], *, is_base_feature: bool) -> Dict[str, Any]:
    return {
        "feature_name": name,
        "category": meta["category"],
        "formula": meta["formula"],
        "lookback_window": meta["lookback"],
        "required_raw_fields": sorted(set(meta.get("depends_on", []) or ["open", "high", "low", "close"])),
        "implemented_in": "quant_toolkit.features.technical" if is_base_feature else "quant_toolkit.teacher_loop.generic",
        "leakage_risk": "low_if_shifted_correctly",
        "status": "implemented",
        "is_base_feature": is_base_feature,
    }


def build_feature_registry() -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for name in BASE_FEATURE_NAMES:
        features.append(_feature_card(name, _BASE_FEATURE_META[name], is_base_feature=True))
    for name in DERIVED_FEATURE_NAMES:
        features.append(_feature_card(name, _DERIVED_FEATURE_META[name], is_base_feature=False))
    return {
        "schema_version": "1.0",
        "feature_count": len(features),
        "base_feature_count": len(BASE_FEATURE_NAMES),
        "derived_feature_count": len(DERIVED_FEATURE_NAMES),
        "features": features,
    }


def ensure_feature_registry(memory_dir: str | Path) -> Path:
    root = Path(memory_dir).expanduser().resolve()
    target = root / "indexes" / "feature_registry.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_feature_registry(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def registry_prompt_block(registry: Dict[str, Any]) -> str:
    lines = [
        f"Available registered features: {registry['feature_count']} total.",
        f"Base features: {registry['base_feature_count']}; derived features: {registry['derived_feature_count']}.",
        "Use only these feature names unless you explicitly request a new feature implementation attempt:",
    ]
    for feature in registry["features"]:
        lines.append(
            f"- {feature['feature_name']} ({feature['category']}): {feature['formula']}"
        )
    return "\n".join(lines)
