from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .agents.scoring import SignalScoringAgent
from .data_adapters.kline_downloader import KlineDownloader, parse_stock_codes
from .provenance import write_json
from .registry import StudioRegistry


REQUIRED_FEATURE_KEYS = [
    "J_minus_D",
    "amplitude",
    "amt_log",
    "amt_ma20_ratio",
    "amt_zscore_20",
    "body_pct",
    "close_to_ma10",
    "close_to_ma20",
    "dJ_3",
    "dK_1",
    "pos_20",
    "ret_10",
    "ret_20",
    "ret_5",
    "vol_ratio_5_20",
    "volatility_10",
    "volatility_20",
    "volume_log",
    "volume_ma20_ratio",
    "volume_zscore_20",
]


def default_recent_kline_earliest() -> str:
    # 120 trading days is roughly 170-180 calendar days.
    return (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    den = _finite_float(denominator, 0.0)
    if abs(den) < 1e-12:
        return float(default)
    return _finite_float(numerator, 0.0) / den


def _zscore(value: Any, mean: Any, std: Any) -> float:
    std_value = _finite_float(std, 0.0)
    if abs(std_value) < 1e-12:
        return 0.0
    return (_finite_float(value, 0.0) - _finite_float(mean, 0.0)) / std_value


def _normalize_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"K-line data is missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("K-line data is empty after numeric/date normalization.")
    out["volume"] = out["volume"].fillna(0.0)
    out["amount"] = out["amount"].fillna(0.0)
    return out


def _append_kdj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_n = out["low"].rolling(9, min_periods=1).min()
    high_n = out["high"].rolling(9, min_periods=1).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((out["close"] - low_n) / denom * 100.0).replace([np.inf, -np.inf], np.nan).fillna(50.0)
    k_values: List[float] = []
    d_values: List[float] = []
    prev_k = 50.0
    prev_d = 50.0
    for value in rsv:
        prev_k = (2.0 / 3.0) * prev_k + (1.0 / 3.0) * _finite_float(value, 50.0)
        prev_d = (2.0 / 3.0) * prev_d + (1.0 / 3.0) * prev_k
        k_values.append(prev_k)
        d_values.append(prev_d)
    out["K"] = k_values
    out["D"] = d_values
    out["J"] = 3.0 * out["K"] - 2.0 * out["D"]
    return out


def build_signal_record_from_kline(
    df: pd.DataFrame,
    *,
    symbol: str,
    source_note: str = "simple_stock_code_live",
) -> Dict[str, Any]:
    out = _append_kdj(_normalize_kline_frame(df))
    if len(out) < 25:
        raise ValueError(
            f"Need at least 25 K-line rows to build the canonical scoring features; got {len(out)}."
        )

    out["pre_close"] = out["close"].shift(1)
    out["daily_ret"] = out["close"].pct_change()
    out["ma10"] = out["close"].rolling(10, min_periods=1).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=1).mean()
    out["low20"] = out["low"].rolling(20, min_periods=1).min()
    out["high20"] = out["high"].rolling(20, min_periods=1).max()
    out["volume_ma5"] = out["volume"].rolling(5, min_periods=1).mean()
    out["volume_ma20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["amount_ma20"] = out["amount"].rolling(20, min_periods=1).mean()
    out["volume_std20"] = out["volume"].rolling(20, min_periods=2).std()
    out["amount_std20"] = out["amount"].rolling(20, min_periods=2).std()
    out["amount_mean20"] = out["amount"].rolling(20, min_periods=1).mean()
    out["volume_mean20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["volatility_10"] = out["daily_ret"].rolling(10, min_periods=2).std()
    out["volatility_20"] = out["daily_ret"].rolling(20, min_periods=2).std()

    row = out.iloc[-1]
    idx = len(out) - 1

    pre_close = _finite_float(row.get("pre_close"), _finite_float(row.get("open"), 0.0))
    position_denominator = _finite_float(row.get("high20"), 0.0) - _finite_float(row.get("low20"), 0.0)
    features = {
        "J_minus_D": _finite_float(row["J"] - row["D"]),
        "amplitude": _safe_divide(row["high"] - row["low"], pre_close, 0.0),
        "amt_log": math.log1p(max(_finite_float(row["amount"], 0.0), 0.0)),
        "amt_ma20_ratio": _safe_divide(row["amount"], row["amount_ma20"], 1.0),
        "amt_zscore_20": _zscore(row["amount"], row["amount_mean20"], row["amount_std20"]),
        "body_pct": _safe_divide(row["close"] - row["open"], row["open"], 0.0),
        "close_to_ma10": _safe_divide(row["close"], row["ma10"], 1.0),
        "close_to_ma20": _safe_divide(row["close"], row["ma20"], 1.0),
        "dJ_3": _finite_float(row["J"] - out.iloc[max(0, idx - 3)]["J"], 0.0),
        "dK_1": _finite_float(row["K"] - out.iloc[max(0, idx - 1)]["K"], 0.0),
        "pos_20": _safe_divide(row["close"] - row["low20"], position_denominator, 0.5),
        "ret_10": _safe_divide(row["close"], out.iloc[max(0, idx - 10)]["close"], 1.0) - 1.0,
        "ret_20": _safe_divide(row["close"], out.iloc[max(0, idx - 20)]["close"], 1.0) - 1.0,
        "ret_5": _safe_divide(row["close"], out.iloc[max(0, idx - 5)]["close"], 1.0) - 1.0,
        "vol_ratio_5_20": _safe_divide(row["volume_ma5"], row["volume_ma20"], 1.0),
        "volatility_10": _finite_float(row["volatility_10"], 0.0),
        "volatility_20": _finite_float(row["volatility_20"], 0.0),
        "volume_log": math.log1p(max(_finite_float(row["volume"], 0.0), 0.0)),
        "volume_ma20_ratio": _safe_divide(row["volume"], row["volume_ma20"], 1.0),
        "volume_zscore_20": _zscore(row["volume"], row["volume_mean20"], row["volume_std20"]),
    }
    clean_features = {
        key: round(_finite_float(features.get(key), 0.0), 6)
        for key in REQUIRED_FEATURE_KEYS
    }
    signal_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    return {
        "symbol": str(symbol).zfill(6),
        "signal_date": signal_date,
        "entry_date": signal_date,
        "features": clean_features,
        "_studio_source": source_note,
        "_studio_feature_note": (
            "Features were built from recent local K-line data for Simple Mode stock-code scoring. "
            "The canonical teacher-library scoring schema is preserved; 60d/120d views are requested in the GPT output."
        ),
    }


def _default_lesson_alias(registry: StudioRegistry) -> str:
    catalog = registry.load_runtime_catalog()
    return str(catalog.get("defaults", {}).get("alignment_seed_alias") or "alignment_seed0005")


def _default_market_run_alias(registry: StudioRegistry) -> str:
    catalog = registry.load_runtime_catalog()
    return str(catalog.get("defaults", {}).get("market_run_alias") or "market_2025_lseed20250705")


def _pct_label(value: Any) -> str:
    return f"{_finite_float(value, 0.0) * 100:.2f}%"


def _feature_diagnostics(signal_record: Mapping[str, Any]) -> Dict[str, Any]:
    features = dict(signal_record.get("features") or {})
    cues: List[str] = []
    risks: List[str] = []

    pos20 = _finite_float(features.get("pos_20"), 0.5)
    if pos20 >= 0.82:
        cues.append(f"20日价格位置 pos_20={pos20:.2f}，接近阶段高位，属于趋势延续/突破观察区。")
        risks.append("价格已经靠近20日区间上沿，若量能不能继续确认，容易出现高位回撤。")
    elif pos20 <= 0.25:
        cues.append(f"20日价格位置 pos_20={pos20:.2f}，靠近阶段低位，更像超跌修复或弱势反弹场景。")
    else:
        cues.append(f"20日价格位置 pos_20={pos20:.2f}，处在区间中部，方向性需要动量和量能进一步确认。")

    close_ma10 = _finite_float(features.get("close_to_ma10"), 1.0)
    close_ma20 = _finite_float(features.get("close_to_ma20"), 1.0)
    if close_ma10 > 1.035 and close_ma20 > 1.045:
        cues.append(f"收盘价显著高于 MA10/MA20（{close_ma10:.3f}/{close_ma20:.3f}），趋势强但短线乖离偏大。")
        risks.append("MA10/MA20 乖离偏大，追高的风险收益比会变差。")
    elif close_ma10 >= 0.99 and close_ma20 >= 0.99:
        cues.append(f"收盘价仍贴近或站上 MA10/MA20（{close_ma10:.3f}/{close_ma20:.3f}），均线结构没有明显破坏。")
    else:
        cues.append(f"收盘价低于 MA10/MA20（{close_ma10:.3f}/{close_ma20:.3f}），趋势承接偏弱。")

    j_minus_d = _finite_float(features.get("J_minus_D"), 0.0)
    dj3 = _finite_float(features.get("dJ_3"), 0.0)
    dk1 = _finite_float(features.get("dK_1"), 0.0)
    if j_minus_d >= 12 and dj3 > 0:
        cues.append(f"KDJ 动量偏强：J-D={j_minus_d:.2f}，dJ_3={dj3:.2f}，短线资金情绪有扩张。")
    elif j_minus_d <= -8 or dj3 < -8:
        cues.append(f"KDJ 转弱或背离：J-D={j_minus_d:.2f}，dJ_3={dj3:.2f}，短线动量在降温。")
        risks.append("KDJ 动量转弱时，即使价格形态尚可，也要防止反弹失败。")
    else:
        cues.append(f"KDJ 信号中性：J-D={j_minus_d:.2f}，dK_1={dk1:.2f}，没有强烈单边确认。")

    volume_ratio = _finite_float(features.get("volume_ma20_ratio"), 1.0)
    vol_ratio_5_20 = _finite_float(features.get("vol_ratio_5_20"), 1.0)
    if volume_ratio >= 1.45 or vol_ratio_5_20 >= 1.35:
        cues.append(f"量能放大：volume/MA20={volume_ratio:.2f}，5/20量比={vol_ratio_5_20:.2f}，说明有资金确认。")
    elif volume_ratio <= 0.75 and vol_ratio_5_20 <= 0.85:
        cues.append(f"量能收缩：volume/MA20={volume_ratio:.2f}，5/20量比={vol_ratio_5_20:.2f}，突破或反弹持续性不足。")
        risks.append("缩量环境下，价格信号更容易变成假突破或弱反弹。")
    else:
        cues.append(f"量能处于常态区：volume/MA20={volume_ratio:.2f}，5/20量比={vol_ratio_5_20:.2f}。")

    volatility10 = _finite_float(features.get("volatility_10"), 0.0)
    volatility20 = _finite_float(features.get("volatility_20"), 0.0)
    if volatility20 > 0 and volatility10 > volatility20 * 1.25:
        cues.append(f"短周期波动率抬升：volatility_10={volatility10:.4f} 高于 volatility_20={volatility20:.4f}，说明分歧正在放大。")
        risks.append("波动率扩张期需要控制仓位，容易出现大幅日内回撤。")
    elif volatility20 > 0 and volatility10 < volatility20 * 0.78:
        cues.append(f"短周期波动率收敛：volatility_10={volatility10:.4f} 低于 volatility_20={volatility20:.4f}，形态更偏蓄势。")
    else:
        cues.append(f"波动率结构中性：volatility_10={volatility10:.4f}，volatility_20={volatility20:.4f}。")

    ret5 = _finite_float(features.get("ret_5"), 0.0)
    ret10 = _finite_float(features.get("ret_10"), 0.0)
    ret20 = _finite_float(features.get("ret_20"), 0.0)
    cues.append(f"收益率结构：ret_5={_pct_label(ret5)}，ret_10={_pct_label(ret10)}，ret_20={_pct_label(ret20)}。")
    if ret5 < 0 < ret20:
        cues.append("短线回撤但20日仍为正，更接近趋势回调后的再选择场景。")
    elif ret5 > 0 and ret10 > 0 and ret20 > 0:
        cues.append("5/10/20日收益同向为正，动量一致性较好。")
    elif ret5 < 0 and ret10 < 0 and ret20 < 0:
        risks.append("5/10/20日收益均为负，当前更像弱势状态，需等待修复确认。")

    amplitude = _finite_float(features.get("amplitude"), 0.0)
    body_pct = _finite_float(features.get("body_pct"), 0.0)
    if amplitude >= 0.06:
        cues.append(f"日内振幅 amplitude={_pct_label(amplitude)} 偏大，说明多空分歧明显。")
        risks.append("高振幅环境下，次日延续性不稳定。")
    if body_pct >= 0.035:
        cues.append(f"实体涨幅 body_pct={_pct_label(body_pct)}，K线实体偏强。")
    elif body_pct <= -0.035:
        cues.append(f"实体跌幅 body_pct={_pct_label(body_pct)}，当日卖压较重。")
        risks.append("实体阴线偏大时，短线承接需要重新验证。")

    return {
        "cues_zh": cues[:10],
        "risk_flags_zh": list(dict.fromkeys(risks))[:6],
        "feature_snapshot": {
            key: features.get(key)
            for key in [
                "pos_20",
                "close_to_ma10",
                "close_to_ma20",
                "J_minus_D",
                "dJ_3",
                "dK_1",
                "volume_ma20_ratio",
                "vol_ratio_5_20",
                "volatility_10",
                "volatility_20",
                "ret_5",
                "ret_10",
                "ret_20",
                "amplitude",
                "body_pct",
            ]
        },
    }


def _trading_reference_zh(scoring: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    total = _finite_float(scoring.get("total_score"), 50.0)
    score60 = _finite_float(scoring.get("score_60d"), total)
    score120 = _finite_float(scoring.get("score_120d"), total)
    risks = list(diagnostics.get("risk_flags_zh") or [])
    if total >= 72:
        stance = "研究结论偏强：可以列入重点观察池，适合等待量价继续确认后的机会。"
    elif total >= 58:
        stance = "研究结论中性偏强：可以观察，但更适合等回踩均线、量能确认或波动率收敛后再提高权重。"
    elif total >= 45:
        stance = "研究结论中性：当前信号不够干净，更适合等待新的突破、缩量回调或KDJ重新走强。"
    else:
        stance = "研究结论偏谨慎：暂时不像老师模型的高胜率舒适区，优先等待结构修复。"
    if score60 - score120 >= 8:
        window = "短线评分明显高于中期评分，说明最近修复快，但中期趋势基础还需要确认。"
    elif score120 - score60 >= 8:
        window = "中期评分高于短线评分，说明趋势底子尚可，但最近几天动量偏弱。"
    else:
        window = "短线与中期评分接近，说明两个时间窗口没有明显冲突。"
    risk_text = f"主要风险：{'；'.join(risks[:3])}。" if risks else "主要风险：暂未发现特别突出的单项硬伤，但仍需控制仓位和验证成交量。"
    return f"{stance}{window}{risk_text}"


def _teacher_domain_name_zh(row: Mapping[str, Any]) -> str:
    round_id = str(row.get("round_id") or "").lower()
    title = str(row.get("title") or "").lower()
    merged = f"{round_id} {title}"
    if "038" in merged or "breakout" in merged:
        return "突破延续老师"
    if "042" in merged:
        return "均线回调老师"
    if "050" in merged:
        return "动量回调老师"
    if "026" in merged:
        return "量能-KDJ 回调老师"
    if "pullback" in merged:
        return "趋势回调老师"
    return "综合技术形态老师"


def _score_band_zh(score: Any) -> str:
    try:
        value = float(score)
    except Exception:
        return "匹配度未知"
    if value >= 70:
        return "高度匹配"
    if value >= 55:
        return "中等偏强"
    if value >= 40:
        return "部分匹配"
    if value >= 25:
        return "匹配偏弱"
    return "明显不匹配"


def _teacher_note_zh(row: Mapping[str, Any]) -> str:
    name = _teacher_domain_name_zh(row)
    band = _score_band_zh(row.get("score"))
    if "突破" in name:
        focus = "关注趋势突破后的延续性、量能确认和波动率配合"
    elif "均线" in name:
        focus = "关注贴近 MA20 后的回调承接、波动率收敛和量价配合"
    elif "动量回调" in name:
        focus = "关注短线回调后动量能否重新转强"
    elif "量能-KDJ" in name:
        focus = "关注量能动量和 KDJ 回调结构是否同时落在舒适区"
    else:
        focus = "关注多因子技术形态是否落在该老师的舒适区"
    return f"{band}。{focus}。"


def _localize_teacher_scores(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        localized = dict(row)
        localized["display_name_zh"] = _teacher_domain_name_zh(row)
        localized["note_zh"] = _teacher_note_zh(row)
        out.append(localized)
    return out


def _result_case_root(contract: Mapping[str, Any], *, symbol: str, signal_record: Mapping[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha1(json.dumps(dict(signal_record), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    return Path(str(contract["scoring_root"])) / f"stock_code_live_{timestamp}_{str(symbol).zfill(6)}_{digest}"


def score_stock_code_live(
    *,
    profile_id: str,
    contract: Mapping[str, Any],
    stock_codes: str | List[str],
    earliest_date: str = "",
    adjust_type: str = "qfq",
    lesson_alias: str = "",
    update_indexes: bool = False,
    full_refresh: bool = False,
    prompt_only: bool = False,
) -> Dict[str, Any]:
    codes = parse_stock_codes(stock_codes)
    if not codes:
        raise ValueError("No valid 6-digit A-share stock code was provided.")
    if len(codes) > 1:
        raise ValueError("Simple Mode live stock scoring currently handles one stock at a time.")
    code = codes[0]
    earliest = str(earliest_date or default_recent_kline_earliest()).replace("-", "")
    registry = StudioRegistry(profile_id)
    registry.ensure_bootstrapped()
    lesson = str(lesson_alias or _default_lesson_alias(registry))
    market_alias = _default_market_run_alias(registry)

    downloader = KlineDownloader(
        cache_dir=str(contract["dataset_stock_klines_root"]),
        index_cache_dir=str(contract["dataset_index_klines_root"]),
        listed_dates_cache_path=str(Path(str(contract["dataset_cache_root"])) / "listed_dates_cache.csv"),
        earliest_date=earliest,
        history_days=180,
        full_refresh=bool(full_refresh),
        adjust_type=str(adjust_type or "qfq"),
    )
    progress_events: List[Dict[str, Any]] = []

    def _progress(phase: str, payload: Dict[str, Any]) -> None:
        compact = dict(payload)
        compact["phase"] = phase
        progress_events.append(compact)

    stock_job = downloader.update_all_kline_cache([code], progress_callback=_progress)
    index_job = downloader.update_indexes(progress_callback=_progress) if update_indexes else {"skipped": True}
    if stock_job.get("failed_codes"):
        raise RuntimeError(f"K-line download failed for {code}: {stock_job.get('failed_codes')}")

    csv_path = Path(str(contract["dataset_stock_klines_root"])) / f"{code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"K-line cache file was not created: {csv_path}")
    df = pd.read_csv(csv_path)
    signal_record = build_signal_record_from_kline(
        df,
        symbol=code,
        source_note="simple_mode_online_kline_stock_code_scoring",
    )

    agent = SignalScoringAgent(registry)
    live_payload = agent.score_live(
        lesson_alias=lesson,
        signal_record=signal_record,
        prompt_only=bool(prompt_only),
        reuse_cache=False,
        persist_run=not bool(prompt_only),
        run_label=f"{str(contract.get('run_id') or 'simple')}_stock_code_live_{code}",
        source_tag="simple_mode_online_kline_stock_code",
        schema_market_run_alias=market_alias,
    )
    result_payload = dict(live_payload.get("result") or {})
    if not result_payload:
        result_payload = {
            "total_score": live_payload.get("total_score"),
            "score_60d": live_payload.get("score_60d"),
            "score_120d": live_payload.get("score_120d"),
            "window_score_note": live_payload.get("window_score_note", ""),
            "short_reason": live_payload.get("short_reason", ""),
            "teacher_scores": live_payload.get("teacher_scores", []),
            "parsed_payload": live_payload.get("parsed_payload", {}),
        }
    result_payload["teacher_scores"] = _localize_teacher_scores(result_payload.get("teacher_scores") or [])
    validation = dict(live_payload.get("signal_schema_validation") or {})
    case_root = _result_case_root(contract, symbol=code, signal_record=signal_record)
    case_root.mkdir(parents=True, exist_ok=True)

    generated_signal_path = case_root / "generated_signal_record.json"
    raw_live_path = case_root / "live_score_payload.json"
    result_path = case_root / "simple_stock_code_live_result.json"
    provenance_path = case_root / "scoring_provenance.json"
    write_json(generated_signal_path, signal_record)
    write_json(raw_live_path, live_payload)

    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "live" if not prompt_only else "prompt_only",
        "model_called": bool(live_payload.get("model_called", False)),
        "result_valid_for_research": bool(live_payload.get("model_called", False) and not prompt_only),
        "lesson_source": "imported_final_asset",
        "teacher_source": "imported_frozen_teacher_zoo",
        "teacher_library_id": "paper_ashare_gptoss20b_v7",
        "teacher_library_name_zh": "A股技术形态基准老师库",
        "teacher_library_source_type": "built_in_baseline",
        "lesson_alias": lesson,
        "schema_market_run_alias": market_alias,
        "fallback_used": False,
        "fallback_reason": "",
        "imported_final_asset": True,
        "current_workflow_asset": False,
        "demo_asset": False,
        "internet_required": True,
        "external_api_called": True,
        "local_runtime_used": True,
        "stock_kline_csv": str(csv_path),
        "generated_signal_record_json": str(generated_signal_path),
        "live_cache_json": str(live_payload.get("cache_path") or ""),
        "live_saved_run_json": str(live_payload.get("saved_run_path") or ""),
        "raw_live_payload_json": str(raw_live_path),
    }
    write_json(provenance_path, provenance)
    feature_diagnostics = _feature_diagnostics(signal_record)

    summary_zh = (
        f"{code} 已完成 live scoring。综合评分={result_payload.get('total_score', '-')}; "
        f"近60日视角={result_payload.get('score_60d', '-')}; "
        f"近120日视角={result_payload.get('score_120d', '-')}。"
    )
    scoring_result = {
        "result_type": "simple_stock_code_live_scoring",
        "mode": "live" if not prompt_only else "prompt_only",
        "summary_zh": summary_zh,
        "symbol": code,
        "signal_date": signal_record.get("signal_date", ""),
        "teacher_library_id": provenance["teacher_library_id"],
        "teacher_library_name_zh": provenance["teacher_library_name_zh"],
        "teacher_library_source_type": provenance["teacher_library_source_type"],
        "total_score": result_payload.get("total_score"),
        "score_60d": result_payload.get("score_60d"),
        "score_120d": result_payload.get("score_120d"),
        "window_score_note": result_payload.get("window_score_note", ""),
        "short_reason": result_payload.get("short_reason", ""),
        "teacher_scores": result_payload.get("teacher_scores", []),
        "feature_diagnostics_zh": feature_diagnostics,
        "trading_reference_zh": _trading_reference_zh(result_payload, feature_diagnostics),
        "model_called": bool(live_payload.get("model_called", False)),
        "signal_record": signal_record,
        "download_summary": {
            "stock_job": stock_job,
            "index_job": index_job,
            "progress_events_tail": progress_events[-10:],
            "adjust_type": str(adjust_type or "qfq"),
            "earliest_date": earliest,
            "stock_kline_csv": str(csv_path),
        },
        "signal_input_manifest": {
            "valid": bool(validation.get("valid", True)),
            "record_count": 1,
            "missing_top_level_keys": validation.get("missing_top_level_keys", []),
            "missing_feature_keys": validation.get("missing_feature_keys", []),
            "non_numeric_feature_keys": validation.get("non_numeric_feature_keys", []),
            "required_feature_count": validation.get("required_feature_count", len(REQUIRED_FEATURE_KEYS)),
            "provided_feature_count": validation.get("provided_feature_count", len(signal_record.get("features") or {})),
        },
        "scoring_provenance": provenance,
        "artifact_paths": {
            "generated_signal_record.json": str(generated_signal_path),
            "simple_stock_code_live_result.json": str(result_path),
            "scoring_provenance.json": str(provenance_path),
            "live_score_payload.json": str(raw_live_path),
            "live_cache_json": str(live_payload.get("cache_path") or ""),
            "live_saved_run_json": str(live_payload.get("saved_run_path") or ""),
        },
        "raw_live_payload": live_payload,
    }
    write_json(result_path, scoring_result)
    return scoring_result
