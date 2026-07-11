#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal LLM apprentice replay framework for single-/multi-teacher pilots.

This module intentionally starts with a relaxed setup:
1. Teachers are already trained on the full available history.
2. Replay starts from 2020 and can focus on a short 2-month window.
3. The apprentice is allowed to see teacher scores/buckets in the first pilot.
4. Holding horizon is fixed to 5 executable trading days, matching teacher labels.

The first goal is not signal-free autonomy. The first goal is to verify that an
LLM can imitate one teacher, then a small teacher ensemble, under realistic NAV
constraints with one API call per decision day.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import joblib
import matplotlib
import requests

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._paths import env_path, project_root
from ..backtest.nav_curve_backtest import compute_nav_curve_fast, load_hs300

PROJECT_ROOT = env_path("QUANT_PROJECT_ROOT", project_root())
REPORT_ROOT = env_path("APPRENTICE_REPORT_ROOT", PROJECT_ROOT / "reports" / "apprentice_loop")
TEACHER_REPORT_ROOT = env_path("TEACHER_LOOP_REPORT_ROOT", PROJECT_ROOT / "reports" / "teacher_loop")
MEMORY_ROOT = env_path("QUANT_MEMORY_DIR", PROJECT_ROOT / "research_memory")
TEACHER_ARTIFACT_ROOT = env_path("TEACHER_LOOP_ARTIFACT_ROOT", MEMORY_ROOT / "artifacts" / "teacher_loop")
TRADER_LESSON_MEMORY_PATH = env_path(
    "APPRENTICE_TRADER_LESSON_PATH",
    MEMORY_ROOT / "trader_lessons" / "apprentice_replay_lessons.jsonl",
)
MASTER_CACHE_PATH = env_path(
    "APPRENTICE_MASTER_CACHE_PATH",
    TEACHER_ARTIFACT_ROOT / "_shared_cache" / "master_feature_label_20260605_v2.joblib",
)
REPLAY_BUNDLE_CACHE_ROOT = env_path(
    "APPRENTICE_REPLAY_BUNDLE_CACHE_ROOT",
    MASTER_CACHE_PATH.parent / "apprentice_replay_bundle_cache",
)
REPLAY_BUNDLE_CACHE_DISABLED = os.environ.get("APPRENTICE_DISABLE_REPLAY_BUNDLE_CACHE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
API_URL = os.environ.get("APPRENTICE_API_URL", os.environ.get("LLM_API_URL", "https://api.chatanywhere.tech/v1/chat/completions")).strip()
API_MAX_RETRIES = int(os.environ.get("APPRENTICE_API_MAX_RETRIES", "12"))
API_RETRY_BASE_SLEEP = float(os.environ.get("APPRENTICE_API_RETRY_BASE_SLEEP", "3.0"))
API_RETRY_MAX_SLEEP = float(os.environ.get("APPRENTICE_API_RETRY_MAX_SLEEP", "90.0"))
API_TIMEOUT_SECONDS = float(os.environ.get("APPRENTICE_API_TIMEOUT_SECONDS", "120"))
RESEARCH_EXPERIENCE_PATH = os.environ.get("APPRENTICE_RESEARCH_EXPERIENCE_PATH", "").strip()


def _apprentice_api_key() -> str:
    return os.environ.get("APPRENTICE_API_KEY", os.environ.get("CHATANYWHERE_API_KEY", "")).strip()


def _progress(message: str) -> None:
    print(f"[apprentice_replay] {message}", flush=True)


@dataclass
class ApprenticeReplayConfig:
    mode: str
    teacher_round_ids: List[str]
    negative_teacher_round_ids: List[str] = field(default_factory=list)
    start_date: str = "2020-01-02"
    end_date: str = "2020-02-28"
    candidate_pool_size: int = 12
    teacher_daily_pick_count: int = 4
    llm_max_daily_picks: int = 4
    lock_days: int = 5
    api_model: str = "deepseek-v4-flash"
    api_temperature: float = 0.0
    api_max_tokens: int = 450
    force_local_qwen_no_thinking: bool = True
    use_line_answer_prefix: bool = False
    private_reasoning_target_tokens: int = 0
    private_reasoning_max_tokens_hint: int = 0
    prompt_recipe: str = "standard"
    include_teacher_signal: bool = True
    candidate_source: str = "teacher_ranked"
    prompt_feature_count: int = 8
    lesson_feature_count: int = 12
    reuse_api_cache: bool = True
    ignore_holdings_context: bool = False
    api_parallel_workers: int = 1
    api_failed_rerun_rounds: int = 1
    api_failed_rerun_workers: int = 0
    api_request_max_retries: int = API_MAX_RETRIES
    inline_day_retry_enabled: bool = True
    summary_variant: str = "simple_v1"
    warmup_sample_count: int = 0
    warmup_start_date: str = "2020-01-02"
    warmup_end_date: str = "2022-12-30"
    warmup_curriculum: str = "legacy_v1"
    warmup_batch_size: int = 10
    warmup_signal_pool_per_day: int = 15
    warmup_review_memory_limit: int = 40
    warmup_lesson_zone_max_lines: int = 12
    warmup_lesson_rewrite_max_tokens: int = 700
    warmup_signal_score_max_tokens: int = 2048
    warmup_retained_case_max_per_tier: int = 4
    scorefit_variant: str = "v1"
    warmup_only: bool = False
    warmup_state_json: str = ""
    run_tag: str = ""
    sample_seed: int = 0
    llm_decision_seed: int = 0

    def run_id(self) -> str:
        if self.mode == "single":
            teacher_part = self.teacher_round_ids[0]
        else:
            teacher_part = "_".join(self.teacher_round_ids)
        tag = f"_{self.run_tag}" if self.run_tag else ""
        return (
            f"{self.mode}_{teacher_part}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}{tag}"
        )


@dataclass
class ReplaySummary:
    run_id: str
    mode: str
    start_date: str
    end_date: str
    api_model: str
    summary_variant: str
    teacher_round_ids: List[str]
    negative_teacher_round_ids: List[str]
    decision_days: int
    api_calls: int
    api_cache_hits: int
    parse_fallback_days: int
    parse_failure_days: int
    query_invoked_days: int
    query_success_days: int
    abstain_days: int
    retry_invoked_days: int
    retry_success_days: int
    prompt_tokens_max: int
    prompt_tokens_mean: float
    total_tokens_max: int
    total_tokens_mean: float
    max_request_chars: int
    llm_selected_rows: int
    teacher_target_rows: int
    teacher_full_rows: int
    mean_daily_jaccard: float
    mean_daily_precision: float
    mean_daily_recall: float
    exact_match_days: int
    exact_match_rate: float
    llm_selected_mean_return: float
    llm_not_selected_mean_return: float
    teacher_selected_mean_return: float
    uplift_vs_not_selected: float
    uplift_vs_teacher_selected: float
    llm_final_nav: float
    llm_total_return: float
    llm_cagr: float
    llm_max_drawdown: float
    teacher_target_final_nav: float
    teacher_target_total_return: float
    teacher_target_cagr: float
    teacher_target_max_drawdown: float
    teacher_full_final_nav: float
    teacher_full_total_return: float
    teacher_full_cagr: float
    teacher_full_max_drawdown: float
    hs300_final_nav: float
    hs300_total_return: float
    llm_vs_target_tracking_gap: float
    llm_vs_full_tracking_gap: float
    nav_overlay_path: str
    nav_curve_csv_path: str
    yearly_returns_csv_path: str
    agreement_csv_path: str


def _normalize_symbol(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"\D", "", text)
    return text.zfill(6)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_existing_scope_batch_records(
    *,
    warmup_report_dir: Path,
    scope_index: int,
    scope_round_id: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    state_paths = sorted(
        warmup_report_dir.glob(f"scope_{scope_index:02d}_{scope_round_id}_batch_*_state.json")
    )
    expected_batch_index = 1
    for state_path in state_paths:
        match = re.search(r"_batch_(\d+)_state\.json$", state_path.name)
        if not match:
            continue
        batch_index = int(match.group(1))
        if batch_index != expected_batch_index:
            break
        review_path = state_path.with_name(
            state_path.name.replace("_state.json", "_review_entries.json")
        )
        if not review_path.exists():
            break
        try:
            state_payload = _load_json(state_path)
            review_payload = _load_json(review_path)
        except Exception:
            break
        if str(state_payload.get("scope_round_id", "")) != str(scope_round_id):
            break
        entries = review_payload.get("entries", [])
        if not isinstance(entries, list):
            break
        records.append(
            {
                "batch_index": batch_index,
                "state": state_payload,
                "review_entries": entries,
                "state_path": str(state_path),
                "review_path": str(review_path),
            }
        )
        expected_batch_index += 1
    return records


def _restore_scope_retained_review_entries(
    *,
    batch_records: Sequence[Dict[str, Any]],
    limit: int,
    max_per_tier: int,
) -> List[Dict[str, Any]]:
    if not batch_records:
        return []
    candidate_entries: List[Dict[str, Any]] = []
    for record in batch_records:
        candidate_entries.extend(list(record.get("review_entries", []) or []))
    if not candidate_entries:
        return []
    candidate_entries = _annotate_review_entry_tiers(candidate_entries)
    last_state = dict(batch_records[-1].get("state", {}) or {})
    retained_case_ids = (
        list(last_state.get("lesson_artifact", {}).get("retained_case_ids", []) or [])
        if isinstance(last_state.get("lesson_artifact", {}), dict)
        else []
    )
    by_id = {
        str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
        for entry in candidate_entries
    }
    retained_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    if retained_entries:
        return _annotate_review_entry_tiers(retained_entries)
    return _fallback_retained_review_entries(
        candidate_entries,
        limit=limit,
        max_per_tier=max_per_tier,
    )


def _load_warmup_state_override(path_text: str) -> Optional[Dict[str, Any]]:
    path_text = (path_text or "").strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"warmup_state_json not found: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"warmup_state_json must be a JSON object: {path}")
    return payload


def _load_research_experience_digest() -> str:
    if not RESEARCH_EXPERIENCE_PATH:
        return ""
    path = Path(RESEARCH_EXPERIENCE_PATH).expanduser().resolve()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^#\s+Main Brain[^\n]*\n+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _is_deepseek_model(model: str) -> bool:
    lower = model.lower()
    return "deepseek" in lower


def _is_gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _prefers_line_id_protocol(model: str) -> bool:
    lower = model.lower()
    return "deepseek" in lower or "qwen" in lower or "minimax" in lower or "gpt-oss" in lower


def _use_local_qwen_no_thinking(model: str) -> bool:
    lower = model.lower()
    api_url = API_URL.lower()
    return "qwen" in lower and ("127.0.0.1" in api_url or "localhost" in api_url)


def _compact_reasoning_instruction(config: ApprenticeReplayConfig) -> str:
    target = int(config.private_reasoning_target_tokens or 0)
    hard_cap = int(config.private_reasoning_max_tokens_hint or 0)
    parts: List[str] = []
    if target > 0 and hard_cap > 0:
        parts.append(
            f"Keep your private scratchpad compact: aim to finish within roughly {target} tokens and stop refining by around {hard_cap} tokens."
        )
    elif hard_cap > 0:
        parts.append(
            f"Keep your private scratchpad compact and stop refining by around {hard_cap} tokens."
        )
    parts.append("Trading is a fuzzy art, not a proof.")
    parts.append(
        "If several candidates are inside a reasonable zone, choose the best approximate set and let expected value work."
    )
    parts.append("Do not over-optimize tiny conflicts or chase perfect certainty.")
    parts.append("When the evidence is good enough, act; when it clearly fails, ABSTAIN.")
    return " ".join(parts)


def _initial_line_answer_prefix(config: ApprenticeReplayConfig) -> Optional[Dict[str, str]]:
    if not config.use_line_answer_prefix:
        return None
    if not _prefers_line_id_protocol(config.api_model):
        return None
    return {"role": "assistant", "content": "SELECT: "}


def _retry_completion_budget(config: ApprenticeReplayConfig) -> int:
    # GPT-OSS occasionally spends a large fraction of a small completion budget on
    # hidden reasoning and returns empty visible content. Give the compact retry
    # prompt a bit more room so the whole run does not die on a single bad day.
    if _is_gpt_oss_model(config.api_model):
        return min(config.api_max_tokens, 1024)
    return min(config.api_max_tokens, 512)


def _is_local_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _extract_symbols_from_text(text: str, allowed_symbols: Sequence[str], limit: int) -> List[str]:
    allowed = {_normalize_symbol(symbol) for symbol in allowed_symbols}
    found = re.findall(r"\b\d{6}\b", text)
    selected: List[str] = []
    for token in found:
        symbol = _normalize_symbol(token)
        if symbol in allowed and symbol not in selected:
            selected.append(symbol)
        if len(selected) >= limit:
            break
    return selected


def _extract_candidate_ids_from_text(text: str, limit: int) -> List[str]:
    found = re.findall(r"\bC\d{2,3}\b", text.upper())
    selected: List[str] = []
    for token in found:
        if token not in selected:
            selected.append(token)
        if len(selected) >= limit:
            break
    return selected


def _extract_numeric_candidate_ids_from_text(
    text: str, *, candidate_id_map: Optional[Dict[str, str]], limit: int
) -> List[str]:
    if not candidate_id_map:
        return []
    max_index = len(candidate_id_map)
    patterns = re.findall(r"(?<![\d.])(?:\d{1,3})(?![\d.])", text)
    selected: List[str] = []
    for token in patterns:
        try:
            idx = int(token)
        except Exception:
            continue
        if idx <= 0 or idx > max_index:
            continue
        cid = f"C{idx:02d}"
        if cid in candidate_id_map and cid not in selected:
            selected.append(cid)
        if len(selected) >= limit:
            break
    return selected


def _looks_like_explicit_abstain(text: str) -> bool:
    cleaned = (text or "").strip().upper()
    return bool(
        re.match(r"^(SELECT:\s*)?(ABSTAIN|NO[_ -]?TRADE|SKIP|PASS)\b", cleaned)
        or re.search(r"\bABSTAIN\b", cleaned)
    )


def _classify_parse_failure_text(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "empty"
    if _looks_like_prompt_echo(t):
        return "prompt_echo"
    if _looks_like_explicit_abstain(t):
        return "abstain_text"
    if re.search(r"\bC\d{2,3}\b", t.upper()):
        return "contains_candidate_ids"
    if re.fullmatch(r"[\d,\s]+", t):
        return "numeric_ids_only"
    if re.search(r"\b202\d-\d{2}-\d{2}\b", t):
        return "date_dump"
    if (
        low.startswith("#")
        or "analysis" in low
        or "verification" in low
        or "根据提供" in low
        or "关键指标" in low
    ):
        return "verbose_analysis"
    return "other_text"


def _parse_feature_list(text: str) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    cleaned = re.sub(r"^(FOCUS|FEATURES|NEED|CHECK)\s*[:=]\s*", "", cleaned, flags=re.I)
    cleaned = cleaned.replace(" and ", ",")
    raw_parts = re.split(r"[,\|;/]+", cleaned)
    out: List[str] = []
    for part in raw_parts:
        token = part.strip().strip(".")
        if not token:
            continue
        token = re.sub(r"^[\-\d\.\)\s]+", "", token)
        if token and token not in out:
            out.append(token)
    return out[:8]


def _parse_query_request_from_text(
    text: str,
    *,
    candidate_id_map: Optional[Dict[str, str]],
    allowed_symbols: Sequence[str],
) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fences(text).strip()
    if not cleaned:
        return None
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and str(payload.get("action", "")).strip().lower() == "query":
            focus = payload.get("focus_features") or payload.get("features") or payload.get("need_features") or []
            if isinstance(focus, str):
                focus_features = _parse_feature_list(focus)
            elif isinstance(focus, list):
                focus_features = [str(item).strip() for item in focus if str(item).strip()]
            else:
                focus_features = []
            target = (
                payload.get("query_candidate_id")
                or payload.get("candidate_id")
                or payload.get("query_symbol")
                or payload.get("symbol")
            )
            if target is None:
                return None
            target_text = str(target).strip().upper()
            if target_text.startswith("C") and target_text in (candidate_id_map or {}):
                candidate_id = target_text
                symbol = (candidate_id_map or {}).get(candidate_id, "")
            elif re.fullmatch(r"\d{6}", target_text) and target_text in {_normalize_symbol(s) for s in allowed_symbols}:
                symbol = _normalize_symbol(target_text)
                inverse = {str(v): str(k) for k, v in (candidate_id_map or {}).items()}
                candidate_id = inverse.get(symbol, "")
            else:
                candidate_id = ""
                symbol = ""
            if not candidate_id and not symbol:
                return None
            return {
                "action": "query",
                "query_candidate_id": candidate_id,
                "query_symbol": symbol,
                "focus_features": focus_features,
                "reason": str(payload.get("reason", "")).strip(),
            }
    except Exception:
        pass

    match = re.search(r"\bQUERY\b\s*[:=]?\s*(C\d{2,3}|\d{6}|\d{1,3})", cleaned, flags=re.I)
    if not match:
        return None
    target = match.group(1).strip().upper()
    candidate_id = ""
    symbol = ""
    if target.startswith("C"):
        candidate_id = target
        symbol = (candidate_id_map or {}).get(candidate_id, "")
    elif re.fullmatch(r"\d{6}", target):
        symbol = _normalize_symbol(target)
        inverse = {str(v): str(k) for k, v in (candidate_id_map or {}).items()}
        candidate_id = inverse.get(symbol, "")
    else:
        idx = int(target)
        candidate_id = f"C{idx:02d}"
        symbol = (candidate_id_map or {}).get(candidate_id, "")
    if not candidate_id and not symbol:
        return None
    focus_match = re.search(r"(?:FOCUS|FEATURES|NEED|CHECK)\s*[:=]\s*([^|;]+)", cleaned, flags=re.I)
    focus_features = _parse_feature_list(focus_match.group(1)) if focus_match else []
    return {
        "action": "query",
        "query_candidate_id": candidate_id,
        "query_symbol": symbol,
        "focus_features": focus_features,
        "reason": "",
    }


def _feature_percentile_rank(values: pd.Series, value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return float((series <= float(value)).mean())


def _build_feature_query_context(
    *,
    day_candidates: pd.DataFrame,
    query_candidate_id: str,
    focus_features: Sequence[str],
) -> str:
    df = day_candidates.copy().reset_index(drop=True)
    if "_candidate_id" not in df.columns:
        df["_candidate_id"] = [f"C{i:02d}" for i in range(1, len(df) + 1)]
    row = df[df["_candidate_id"].astype(str) == str(query_candidate_id)].head(1)
    if row.empty:
        return f"Feature query result: candidate {query_candidate_id} not found in current pool."
    row = row.iloc[0]
    feature_cols = _lesson_feature_columns(df)
    numeric_features = [col for col in feature_cols if col in df.columns]
    requested = [feat for feat in focus_features if feat in numeric_features]
    extra_ranked: List[Tuple[str, float, float, Optional[float], float, float, float]] = []
    for feat in numeric_features:
        series = pd.to_numeric(df[feat], errors="coerce").dropna()
        if series.empty:
            continue
        value = row.get(feat)
        if pd.isna(value):
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            z = 0.0
        else:
            z = (float(value) - mean) / std
        percentile = _feature_percentile_rank(series, value)
        extra_ranked.append(
            (
                feat,
                abs(float(z)),
                float(value),
                percentile,
                float(series.median()),
                float(series.quantile(0.25)),
                float(series.quantile(0.75)),
            )
        )
    extra_ranked.sort(key=lambda item: item[1], reverse=True)
    extra_features: List[str] = []
    for feat, *_ in extra_ranked:
        if feat in requested:
            continue
        extra_features.append(feat)
        if len(extra_features) >= 5:
            break
    combined_features = requested + extra_features
    feature_lines = []
    for feat in combined_features:
        series = pd.to_numeric(df[feat], errors="coerce").dropna()
        if series.empty:
            continue
        value = row.get(feat)
        if pd.isna(value):
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        z = 0.0 if not np.isfinite(std) or std <= 1e-12 else (float(value) - mean) / std
        percentile = _feature_percentile_rank(series, value)
        percentile_text = f"{percentile:.2f}" if percentile is not None else "nan"
        feature_lines.append(
            f"- {feat}: value={_format_float(value)}, pct_rank={percentile_text}, "
            f"median={_format_float(series.median())}, p25={_format_float(series.quantile(0.25))}, "
            f"p75={_format_float(series.quantile(0.75))}, z={z:.2f}"
        )
    if not feature_lines:
        feature_lines.append("- no numeric feature evidence available")
    symbol = str(row.get("symbol", "")).strip()
    entry_date = pd.Timestamp(row.get("entry_date")).strftime("%Y-%m-%d") if pd.notna(row.get("entry_date")) else "na"
    exit_date = pd.Timestamp(row.get("exit_date")).strftime("%Y-%m-%d") if pd.notna(row.get("exit_date")) else "na"
    signal_date = pd.Timestamp(row.get("signal_date")).strftime("%Y-%m-%d") if pd.notna(row.get("signal_date")) else "na"
    top_band = extra_ranked[:3]
    band_lines = []
    for feat, _, value, percentile, median, p25, p75 in top_band:
        if percentile is None:
            continue
        side = "upper" if percentile >= 0.5 else "lower"
        band_lines.append(
            f"- {feat}: {side} band, value={_format_float(value)}, percentile={percentile:.2f}, median={_format_float(median)}"
        )
    if not band_lines:
        band_lines.append("- no reliable band summary")
    return "\n".join(
        [
            f"Feature query result for {query_candidate_id} / {symbol}:",
            f"- signal_date={signal_date} | entry_date={entry_date} | exit_date={exit_date}",
            f"- requested_features={', '.join(requested) if requested else 'none'}",
            "Requested feature statistics:",
            *feature_lines,
            "Extra band summary:",
            *band_lines,
            "Use this extra evidence once, then make the final decision. Do not ask for a second query.",
        ]
    )


def _parse_model_action_reply(
    *,
    content: str,
    model: str,
    allowed_symbols: Sequence[str],
    limit: int,
    candidate_id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    query = _parse_query_request_from_text(content, candidate_id_map=candidate_id_map, allowed_symbols=allowed_symbols)
    if query is not None:
        return query
    parsed = _parse_model_selection_reply(
        content=content,
        model=model,
        allowed_symbols=allowed_symbols,
        limit=limit,
        candidate_id_map=candidate_id_map,
    )
    parsed["action"] = "abstain" if parsed.get("abstain") else ("select" if not parsed.get("parse_failed") else "failure")
    return parsed


def _looks_like_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    bad_phrases = [
        "user will provide a query",
        "the assistant's response must",
        "must be exactly one line",
        "reply with exactly one line",
        "the assistant must not output anything else",
    ]
    return any(phrase in lowered for phrase in bad_phrases)


def _load_master_dataset() -> pd.DataFrame:
    payload = joblib.load(MASTER_CACHE_PATH, mmap_mode="r")
    dataset = payload["dataset"]
    df = dataset.copy()
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def _multi_teacher_bundle_cache_spec(config: ApprenticeReplayConfig) -> Tuple[Path, Dict[str, Any]]:
    payload = {
        "cache_schema_version": "multi_teacher_replay_bundle_v1",
        "teacher_report_root": str(TEACHER_REPORT_ROOT),
        "master_cache_path": str(MASTER_CACHE_PATH),
        "teacher_round_ids": list(config.teacher_round_ids),
        "negative_teacher_round_ids": list(config.negative_teacher_round_ids),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "candidate_pool_size": int(config.candidate_pool_size),
        "teacher_daily_pick_count": int(config.teacher_daily_pick_count),
        "prompt_feature_count": int(config.prompt_feature_count),
        "lesson_feature_count": int(config.lesson_feature_count),
        "candidate_source": str(config.candidate_source),
        "summary_variant": str(config.summary_variant),
        "warmup_start_date": config.warmup_start_date if config.summary_variant == "enriched_v2" else "",
        "warmup_end_date": config.warmup_end_date if config.summary_variant == "enriched_v2" else "",
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    filename = f"multi_{config.start_date.replace('-', '')}_{config.end_date.replace('-', '')}_{digest}.joblib"
    return REPLAY_BUNDLE_CACHE_ROOT / filename, payload


def _load_or_build_multi_teacher_bundle(config: ApprenticeReplayConfig) -> Dict[str, Any]:
    cache_path, cache_meta = _multi_teacher_bundle_cache_spec(config)
    meta_path = cache_path.with_suffix(".meta.json")
    if not REPLAY_BUNDLE_CACHE_DISABLED and cache_path.exists():
        try:
            bundle = joblib.load(cache_path)
            required = {"metas", "prompt_features", "negative_metas", "candidate_pool_df", "teacher_target_df", "teacher_full_df"}
            if not isinstance(bundle, dict) or not required.issubset(bundle):
                raise ValueError(f"missing bundle keys: expected={sorted(required)}")
            _progress(f"multi replay bundle cache hit path={cache_path}")
            return bundle
        except Exception as exc:
            _progress(f"multi replay bundle cache invalid path={cache_path} reason={exc}; rebuilding")

    master_df = _load_master_dataset()
    _progress(f"master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")
    merged, metas, prompt_features = _build_multi_teacher_frame(config, master_df)
    _progress(
        "multi teacher frame built "
        f"rows={len(merged)} teachers={len(metas)} prompt_features={len(prompt_features)}"
    )
    negative_metas = [
        _negative_teacher_meta(round_id, max(4, config.prompt_feature_count // 2))
        for round_id in config.negative_teacher_round_ids
    ]
    if config.summary_variant == "enriched_v2":
        for meta in metas:
            meta["preference_bands"] = _derive_preference_bands(
                round_id=meta["round_id"],
                master_df=master_df,
                feature_cols=meta["top_prompt_features"],
                start_date=config.warmup_start_date,
                end_date=config.warmup_end_date,
            )
    candidate_pool_df, teacher_target_df = _multi_teacher_target(merged, config)
    teacher_full_df = teacher_target_df.copy()
    _progress(
        "candidate pool ready "
        f"pool_rows={len(candidate_pool_df)} target_rows={len(teacher_target_df)} "
        f"decision_days={candidate_pool_df['signal_date'].nunique()}"
    )
    bundle = {
        "metas": metas,
        "prompt_features": prompt_features,
        "negative_metas": negative_metas,
        "candidate_pool_df": candidate_pool_df,
        "teacher_target_df": teacher_target_df,
        "teacher_full_df": teacher_full_df,
    }
    if not REPLAY_BUNDLE_CACHE_DISABLED:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, cache_path)
        _write_json(meta_path, cache_meta)
        _progress(f"multi replay bundle cache saved path={cache_path}")
    return bundle


def _load_teacher_predictions(round_id: str, *, start_date: str, end_date: str) -> pd.DataFrame:
    pred_path = TEACHER_ARTIFACT_ROOT / round_id / "test_predictions.csv.gz"
    df = pd.read_csv(pred_path, parse_dates=["signal_date", "entry_date", "exit_date"])
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df = df[(df["signal_date"] >= pd.Timestamp(start_date)) & (df["signal_date"] <= pd.Timestamp(end_date))].copy()
    df["bucket"] = df["bucket"].astype(int)
    df["quintile"] = df["bucket"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"})
    df["return_20d"] = df["future_return_5d"].astype(float)
    return df


def _load_teacher_spec(round_id: str) -> Dict[str, Any]:
    return _load_json(TEACHER_REPORT_ROOT / round_id / "selected_spec.json")


def _load_teacher_nav_summary(round_id: str) -> Dict[str, Any]:
    return _load_json(TEACHER_REPORT_ROOT / round_id / "nav_summary.json")


def _load_teacher_feature_importance(round_id: str) -> pd.DataFrame:
    path = TEACHER_REPORT_ROOT / round_id / "feature_importance.csv"
    return pd.read_csv(path)


def _load_teacher_factor_analysis_summary(round_id: str) -> Dict[str, Any]:
    path = TEACHER_REPORT_ROOT / round_id / "factor_analysis_summary.json"
    summary: Dict[str, Any] = {}
    if path.exists():
        summary = _load_json(path)

    branch_path = TEACHER_REPORT_ROOT / round_id / "branch_rule_cards.json"
    if branch_path.exists():
        branch_payload = _load_json(branch_path)
        field_map = {
            "branch_cards": "branch_rule_cards",
            "soft_rules": "soft_rules",
            "hard_veto_rules": "hard_veto_rules",
            "meta_rules": "meta_rules",
            "ambiguous_combo_contexts": "ambiguous_combo_contexts",
            "false_positive_contrast_pairs": "false_positive_contrast_pairs",
            "archetypes": "archetypes",
            "pdp_effect_summaries": "pdp_effect_summaries",
        }
        if not summary:
            summary = {}
        summary.setdefault("report_schema_version", branch_payload.get("report_schema_version", "branch_oriented_v2"))
        for branch_key, summary_key in field_map.items():
            if not summary.get(summary_key):
                summary[summary_key] = list(branch_payload.get(branch_key) or [])
        artifact_files = dict(summary.get("artifact_files") or {})
        artifact_files.setdefault("branch_rule_cards_json", branch_path.name)
        summary["artifact_files"] = artifact_files
    return summary


def _source_round_id_for_round(round_id: str) -> str:
    try:
        payload = _load_teacher_spec(round_id)
    except FileNotFoundError:
        text = str(round_id).strip()
        fallback = re.sub(r"_frozen_\d{4}$", "", text)
        return fallback or text
    source_round_id = payload.get("source_round_id")
    if source_round_id:
        return str(source_round_id)
    return round_id


def _find_memory_item_by_round_id(directory: Path, round_id: str) -> Optional[Dict[str, Any]]:
    for path in sorted(directory.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if str(payload.get("round_id", "")) == round_id:
            return payload
    return None


def _memory_lookup_round_ids(round_id: str) -> List[str]:
    candidates = [str(round_id)]
    source_round_id = _source_round_id_for_round(round_id)
    if source_round_id and source_round_id not in candidates:
        candidates.append(source_round_id)
    return candidates


def _kb_context_for_round_ids(round_ids: Sequence[str]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for round_id in round_ids:
        lookup_round_ids = _memory_lookup_round_ids(round_id)
        lesson = None
        teacher_model = None
        matched_round_id = round_id
        for lookup_round_id in lookup_round_ids:
            if teacher_model is None:
                teacher_model = _find_memory_item_by_round_id(MEMORY_ROOT / "teacher_models", lookup_round_id)
                if teacher_model is not None:
                    matched_round_id = lookup_round_id
            if lesson is None:
                lesson = _find_memory_item_by_round_id(MEMORY_ROOT / "research_lessons", lookup_round_id)
                if lesson is not None:
                    matched_round_id = lookup_round_id
            if teacher_model is not None and lesson is not None:
                break
        card: Dict[str, Any] = {
            "round_id": round_id,
            "memory_lookup_round_ids": lookup_round_ids,
            "memory_matched_round_id": matched_round_id,
        }
        if teacher_model:
            card.update(
                {
                    "teacher_model_title": teacher_model.get("title"),
                    "teacher_model_summary": teacher_model.get("summary"),
                    "sample_template": teacher_model.get("sample_template"),
                    "research_family": teacher_model.get("research_family"),
                    "feature_count": teacher_model.get("feature_count"),
                    "accepted_as_teacher": teacher_model.get("accepted_as_teacher"),
                    "zoo_partition": teacher_model.get("zoo_partition"),
                    "mean_alpha": teacher_model.get("mean_alpha"),
                    "median_alpha": teacher_model.get("median_alpha"),
                    "nav_cagr": teacher_model.get("nav_cagr"),
                    "nav_max_drawdown": teacher_model.get("nav_max_drawdown"),
                    "factor_analysis_summary": teacher_model.get("factor_analysis_summary") or {},
                }
            )
        if lesson:
            card.update(
                {
                    "lesson_summary": lesson.get("lesson_summary"),
                    "recommended_action": lesson.get("recommended_action"),
                    "applies_to": lesson.get("applies_to"),
                }
            )
        contexts.append(card)
    return contexts


def _compact_kb_hint(item: Dict[str, Any], max_len: int = 120) -> str:
    parts: List[str] = []
    partition = item.get("zoo_partition")
    if partition:
        parts.append(str(partition))
    if item.get("accepted_as_teacher") is True:
        parts.append("accepted")
    summary = item.get("teacher_model_summary") or item.get("lesson_summary") or ""
    summary = " ".join(str(summary).split())
    if summary:
        if len(summary) > max_len:
            summary = summary[: max_len - 3].rstrip() + "..."
        parts.append(summary)
    factor_summary = item.get("factor_analysis_summary") or {}
    top_features = factor_summary.get("top_global_features") or []
    if top_features:
        top = top_features[0]
        parts.append(f"factor={top.get('feature')}:{top.get('preferred_direction')}")
    return " | ".join(parts) if parts else "no extra memory hint"


def _band_lines_for_metas(metas: Sequence[Dict[str, Any]]) -> List[str]:
    detailed: List[str] = []
    no_preference_ids: List[str] = []
    for meta in metas:
        hint = _compact_band_hint(meta.get("preference_bands", []))
        if hint == "no strong band preference derived":
            no_preference_ids.append(meta["round_id"])
        else:
            detailed.append(f"Bands {meta['round_id']}: {hint}")
    if detailed:
        if no_preference_ids:
            detailed.append(f"Bands none-derived: {','.join(no_preference_ids)}")
        return detailed
    return ["Bands: none derived across current positive teachers"]


def _format_float(value: Any) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    value = float(value)
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _derive_preference_bands(
    *,
    round_id: str,
    master_df: pd.DataFrame,
    feature_cols: Sequence[str],
    start_date: str,
    end_date: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    feature_cols = [col for col in feature_cols if col in master_df.columns]
    if not feature_cols:
        return []
    pred_df = _load_teacher_predictions(round_id, start_date=start_date, end_date=end_date)
    if pred_df.empty:
        return []
    joined = _join_features(pred_df, master_df, feature_cols)
    selected = joined[joined["bucket"] == 5].copy()
    if selected.empty:
        selected = joined[joined["bucket"] >= 4].copy()
    if selected.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for feat in feature_cols:
        pool_vals = joined[feat].dropna()
        sel_vals = selected[feat].dropna()
        if len(pool_vals) < 20 or len(sel_vals) < 12:
            continue
        pool_med = float(pool_vals.median())
        sel_med = float(sel_vals.median())
        sel_q25 = float(sel_vals.quantile(0.25))
        sel_q75 = float(sel_vals.quantile(0.75))
        std = float(pool_vals.std())
        effect = abs(sel_med - pool_med) / (std + 1e-9)
        direction = "higher" if sel_med > pool_med else "lower"
        rows.append(
            {
                "feature": feat,
                "direction": direction,
                "selected_q25": sel_q25,
                "selected_median": sel_med,
                "selected_q75": sel_q75,
                "pool_median": pool_med,
                "effect_size": effect,
            }
        )
    rows.sort(key=lambda item: item["effect_size"], reverse=True)
    return rows[:top_k]


def _compact_band_hint(bands: Sequence[Dict[str, Any]]) -> str:
    if not bands:
        return "no strong band preference derived"
    parts = []
    for band in bands:
        parts.append(
            f"{band['feature']} {band['direction']} [{_format_float(band['selected_q25'])},{_format_float(band['selected_q75'])}]"
        )
    return "; ".join(parts)


def _compact_factor_rule_hint(summary: Mapping[str, Any], *, top_features: int = 3, top_combos: int = 1) -> str:
    if not summary:
        return "no factor-rule summary"
    parts: List[str] = []
    for row in list(summary.get("top_global_features") or [])[:top_features]:
        feature = row.get("feature")
        if not feature:
            continue
        direction = row.get("preferred_direction", "mixed")
        q25 = row.get("selected_q25")
        q75 = row.get("selected_q75")
        if q25 is not None and q75 is not None:
            band_text = f"[{_format_float(q25)},{_format_float(q75)}]"
        else:
            band_text = "[]"
        shape = row.get("shape_hint") or "na"
        parts.append(f"{feature} {direction} {band_text} {shape}")
    for row in list(summary.get("top_feature_combos") or [])[:top_combos]:
        left = row.get("feature_left")
        right = row.get("feature_right")
        if not left or not right:
            continue
        lift = row.get("lift_favored_vs_opposite")
        parts.append(f"combo {left}+{right} lift={_format_float(lift)}")
    return "; ".join(parts) if parts else "no factor-rule summary"


def _uses_explainability_only_prompt(config: ApprenticeReplayConfig) -> bool:
    return str(getattr(config, "prompt_recipe", "standard")).strip().lower() == "explainability_only"


def _uses_report_v2_with_lessons_prompt(config: ApprenticeReplayConfig) -> bool:
    return str(getattr(config, "prompt_recipe", "standard")).strip().lower() == "report_v2_with_lessons"


def _uses_branch_report_context(config: ApprenticeReplayConfig) -> bool:
    return _uses_explainability_only_prompt(config) or _uses_report_v2_with_lessons_prompt(config)


def _compact_branch_rule_lines(
    summary: Mapping[str, Any],
    *,
    max_branches: int = 2,
    max_soft_rules: int = 5,
    max_veto_rules: int = 3,
    max_meta_rules: int = 3,
    max_trap_pairs: int = 2,
) -> List[str]:
    if not summary:
        return ["none"]

    branch_cards = list(summary.get("branch_rule_cards") or summary.get("branch_cards") or [])
    soft_rules = list(summary.get("soft_rules") or [])
    hard_veto_rules = list(summary.get("hard_veto_rules") or [])
    meta_rules = list(summary.get("meta_rules") or [])
    trap_pairs = list(summary.get("false_positive_contrast_pairs") or [])
    archetypes = list(summary.get("archetypes") or [])
    ambiguous_contexts = list(summary.get("ambiguous_combo_contexts") or [])

    lines: List[str] = []
    for row in branch_cards[:max_branches]:
        anchor = row.get("anchor_pair") or {}
        left = f"{anchor.get('left_feature', '?')} {anchor.get('left_direction', '?')} {anchor.get('left_band', '[]')}"
        right = f"{anchor.get('right_feature', '?')} {anchor.get('right_direction', '?')} {anchor.get('right_band', '[]')}"
        evidence = row.get("evidence") or {}
        favored = _format_float(evidence.get("favored_return"))
        mixed = _format_float(evidence.get("mixed_return"))
        opposite = _format_float(evidence.get("opposite_return"))
        lines.append(
            f"{row.get('branch_id', 'BRANCH')} {row.get('strength', 'medium')}: {left} + {right}; "
            f"full={favored}, partial={mixed}, opposite={opposite}"
        )
    for row in soft_rules[:max_soft_rules]:
        lines.append(
            f"{row.get('rule_id', 'SOFT')} {row.get('strength', 'medium')}: prefer "
            f"{row.get('feature', '?')} {row.get('direction', '?')} {row.get('band', '[]')} "
            f"({row.get('usage_note', 'fuzzy preference')})"
        )
    for row in hard_veto_rules[:max_veto_rules]:
        lines.append(
            f"{row.get('rule_id', 'VETO')} {row.get('strength', 'medium')}: avoid when "
            f"{row.get('trigger', 'veto condition')}; why={row.get('reason', '')}"
        )
    for row in meta_rules[:max_meta_rules]:
        lines.append(f"{row.get('rule_id', 'META')}: {row.get('guidance', '')}")
    for row in ambiguous_contexts[:1]:
        lines.append(
            f"{row.get('context_id', 'AMBIG')}: {row.get('guidance', '')} "
            f"(favored={_format_float(row.get('favored_return'))}, mixed={_format_float(row.get('mixed_return'))})"
        )
    for row in trap_pairs[:max_trap_pairs]:
        shared = ",".join(list(row.get("shared_positive_features") or [])[:3]) or "none"
        winner_extra = ",".join(list(row.get("winner_extra_features") or [])[:3]) or "none"
        trap_drag = ",".join(list(row.get("trap_negative_features") or [])[:3]) or "none"
        lines.append(
            f"{row.get('pair_id', 'PAIR')}: shared={shared}; winner_extra={winner_extra}; trap_drag={trap_drag}"
        )
    for row in archetypes[:1]:
        core = ",".join(list(row.get("core_positive_features") or [])[:4]) or "none"
        offsets = ",".join(list(row.get("common_negative_offsets") or [])[:3]) or "none"
        lines.append(
            f"{row.get('archetype_id', 'ARCH')}: {row.get('name', 'archetype')} core={core}; common_offsets={offsets}"
        )
    return lines or ["none"]


def _teacher_explainability_payload(
    summary: Mapping[str, Any],
    *,
    max_branches: int = 2,
    max_soft_rules: int = 5,
    max_veto_rules: int = 3,
    max_meta_rules: int = 3,
    max_trap_pairs: int = 2,
) -> Dict[str, Any]:
    if not summary:
        return {}
    return {
        "report_schema_version": summary.get("report_schema_version", "legacy"),
        "branch_rule_cards": list(summary.get("branch_rule_cards") or summary.get("branch_cards") or [])[:max_branches],
        "soft_rules": list(summary.get("soft_rules") or [])[:max_soft_rules],
        "hard_veto_rules": list(summary.get("hard_veto_rules") or [])[:max_veto_rules],
        "meta_rules": list(summary.get("meta_rules") or [])[:max_meta_rules],
        "ambiguous_combo_contexts": list(summary.get("ambiguous_combo_contexts") or [])[:1],
        "false_positive_contrast_pairs": list(summary.get("false_positive_contrast_pairs") or [])[:max_trap_pairs],
        "archetypes": list(summary.get("archetypes") or [])[:1],
        "pdp_effect_summaries": list(summary.get("pdp_effect_summaries") or [])[:4],
    }


def _summary_has_explainability_content(summary: Optional[Mapping[str, Any]]) -> bool:
    if not summary:
        return False
    keys = [
        "branch_rule_cards",
        "branch_cards",
        "soft_rules",
        "hard_veto_rules",
        "meta_rules",
        "ambiguous_combo_contexts",
        "false_positive_contrast_pairs",
        "archetypes",
        "pdp_effect_summaries",
    ]
    for key in keys:
        value = summary.get(key) if isinstance(summary, Mapping) else None
        if isinstance(value, list) and value:
            return True
    return False


def _resolve_explainability_summary(
    *,
    preferred_round_ids: Sequence[str],
    fallback_summary: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    for round_id in preferred_round_ids:
        round_text = str(round_id or "").strip()
        if not round_text:
            continue
        try:
            summary = _load_teacher_factor_analysis_summary(round_text)
        except Exception:
            continue
        if _summary_has_explainability_content(summary):
            return dict(summary), round_text
    if _summary_has_explainability_content(fallback_summary):
        return dict(fallback_summary or {}), ""
    return dict(fallback_summary or {}), ""


def _sample_uniform_dates(
    dates: Sequence[pd.Timestamp], count: int, *, sample_seed: int = 0
) -> List[pd.Timestamp]:
    unique_dates = sorted(pd.Timestamp(dt) for dt in set(dates))
    if count <= 0 or len(unique_dates) <= count:
        return unique_dates
    if int(sample_seed) <= 0:
        positions = np.linspace(0, len(unique_dates) - 1, count)
        idxs = sorted({int(round(pos)) for pos in positions})
        while len(idxs) < count:
            for i in range(len(unique_dates)):
                if i not in idxs:
                    idxs.append(i)
                if len(idxs) >= count:
                    break
        idxs = sorted(idxs[:count])
        return [unique_dates[i] for i in idxs]

    # Seeded stratified jitter: keep the full time-span coverage but let each
    # run perturb which day is chosen inside each temporal slice.
    rng = np.random.default_rng(int(sample_seed))
    bucket_edges = np.linspace(0, len(unique_dates), count + 1)
    selected_idxs: List[int] = []
    used_idxs: set[int] = set()
    for bucket_idx in range(count):
        lo = int(math.floor(bucket_edges[bucket_idx]))
        hi = int(math.ceil(bucket_edges[bucket_idx + 1])) - 1
        lo = max(0, min(lo, len(unique_dates) - 1))
        hi = max(lo, min(hi, len(unique_dates) - 1))
        candidates = [idx for idx in range(lo, hi + 1) if idx not in used_idxs]
        if not candidates:
            continue
        chosen = int(rng.choice(candidates))
        selected_idxs.append(chosen)
        used_idxs.add(chosen)

    if len(selected_idxs) < count:
        remaining = [idx for idx in range(len(unique_dates)) if idx not in used_idxs]
        if remaining:
            extra = rng.permutation(remaining).tolist()
            selected_idxs.extend(int(idx) for idx in extra[: count - len(selected_idxs)])
    selected_idxs = sorted(selected_idxs[:count])
    if len(selected_idxs) < count:
        for idx in range(len(unique_dates)):
            if idx not in selected_idxs:
                selected_idxs.append(idx)
            if len(selected_idxs) >= count:
                break
        selected_idxs = sorted(selected_idxs[:count])
    return [unique_dates[i] for i in selected_idxs]


def _split_count_evenly(total: int, bucket_count: int) -> List[int]:
    if bucket_count <= 0:
        return []
    if total <= 0:
        return [0] * bucket_count
    base = total // bucket_count
    extra = total % bucket_count
    return [base + (1 if idx < extra else 0) for idx in range(bucket_count)]


def _lesson_feature_columns(candidate_pool_df: pd.DataFrame) -> List[str]:
    exclude = {
        "symbol",
        "signal_date",
        "entry_date",
        "exit_date",
        "future_return_5d",
        "return_20d",
        "sample_template",
        "teacher_round_id",
        "teacher_present_count",
        "ensemble_score",
        "_candidate_id",
        "teacher_rank",
        "teacher_rank_pct",
        "score",
        "bucket",
        "quintile",
    }
    cols = []
    for col in candidate_pool_df.columns:
        if col in exclude:
            continue
        if col.endswith("_score") or col.endswith("_bucket") or col.endswith("_rank_pct"):
            continue
        if pd.api.types.is_numeric_dtype(candidate_pool_df[col]):
            cols.append(col)
    return cols


def _load_prior_trader_lessons(round_ids: Sequence[str], limit: int = 20) -> List[Dict[str, Any]]:
    if not TRADER_LESSON_MEMORY_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in TRADER_LESSON_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        lesson_rounds = set(payload.get("teacher_round_ids", []))
        if lesson_rounds and lesson_rounds.intersection(round_ids):
            rows.append(payload)
    return rows[-limit:]


def _append_trader_lessons(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    TRADER_LESSON_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRADER_LESSON_MEMORY_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _lesson_case_records(day_pool: pd.DataFrame, feature_cols: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    day_pool = day_pool.copy().reset_index(drop=True)
    day_pool["_candidate_id"] = [f"C{i:02d}" for i in range(1, len(day_pool) + 1)]
    for rec in _candidate_rows_to_records(day_pool, feature_cols, teacher_fields=[]):
        records.append(rec)
    return records


def _filter_case_records_by_symbols(records: Sequence[Dict[str, Any]], symbols: Sequence[str]) -> List[Dict[str, Any]]:
    allowed = {_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()}
    if not allowed:
        return []
    kept: List[Dict[str, Any]] = []
    for record in records:
        record_symbol = _normalize_symbol(record.get("symbol", ""))
        if record_symbol in allowed:
            kept.append(dict(record))
    return kept


def _feature_values_from_case_records(records: Sequence[Dict[str, Any]], feature: str) -> List[float]:
    values: List[float] = []
    for record in records:
        features = record.get("features", {}) if isinstance(record, Mapping) else {}
        if not isinstance(features, Mapping):
            continue
        value = features.get(feature)
        if value is None or pd.isna(value):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _feature_distribution_from_values(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray([float(value) for value in values], dtype=float)
    if arr.size == 0:
        return None
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _feature_distribution_from_case_records(
    records: Sequence[Dict[str, Any]],
    feature_cols: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for feature in feature_cols:
        stats = _feature_distribution_from_values(_feature_values_from_case_records(records, feature))
        if stats is not None:
            summary[feature] = stats
    return summary


def _feature_gap_rows(
    primary_summary: Mapping[str, Mapping[str, float]],
    secondary_summary: Mapping[str, Mapping[str, float]],
    *,
    feature_order: Sequence[str],
    top_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feature in feature_order:
        primary = primary_summary.get(feature)
        secondary = secondary_summary.get(feature)
        if not primary or not secondary:
            continue
        primary_median = float(primary.get("median", float("nan")))
        secondary_median = float(secondary.get("median", float("nan")))
        if math.isnan(primary_median) or math.isnan(secondary_median):
            continue
        gap = primary_median - secondary_median
        rows.append(
            {
                "feature": feature,
                "direction": "higher" if gap > 0 else "lower",
                "median_gap": float(gap),
                "abs_median_gap": abs(float(gap)),
                "primary": dict(primary),
                "secondary": dict(secondary),
            }
        )
    rows.sort(key=lambda item: item["abs_median_gap"], reverse=True)
    return rows[:top_n]


def _interval_text(stats: Mapping[str, Any]) -> str:
    return f"[{_format_float(stats.get('q25'))}, {_format_float(stats.get('q75'))}]"


def _build_interval_review_card(
    entry: Mapping[str, Any],
    *,
    feature_priority: Sequence[str],
    max_features: int = 2,
) -> str:
    teacher_only_summary = _feature_distribution_from_case_records(
        entry.get("teacher_only_feature_records", []),
        feature_priority,
    )
    llm_only_summary = _feature_distribution_from_case_records(
        entry.get("llm_only_feature_records", []),
        feature_priority,
    )
    gap_rows = _feature_gap_rows(
        teacher_only_summary,
        llm_only_summary,
        feature_order=feature_priority,
        top_n=max_features,
    )
    compare_label = "teacher_only_vs_llm_only"
    if not gap_rows:
        teacher_selected_summary = _feature_distribution_from_case_records(
            entry.get("teacher_target_feature_records", []),
            feature_priority,
        )
        llm_selected_summary = _feature_distribution_from_case_records(
            entry.get("llm_selected_feature_records", []),
            feature_priority,
        )
        gap_rows = _feature_gap_rows(
            teacher_selected_summary,
            llm_selected_summary,
            feature_order=feature_priority,
            top_n=max_features,
        )
        compare_label = "teacher_selected_vs_llm_selected"
    feature_bits: List[str] = []
    for row in gap_rows:
        feature_bits.append(
            f"{row['feature']} {row['direction']} T{_interval_text(row['primary'])} vs L{_interval_text(row['secondary'])}"
        )
    feature_text = "; ".join(feature_bits) if feature_bits else "feature gap not available"
    card = (
        f"CID={str(entry.get('case_id', entry.get('decision_date', ''))).strip()} | "
        f"tier={str(entry.get('teacher_eval_tier', 'unknown')).strip()} | "
        f"alpha={float(entry.get('teacher_alpha', 0.0))*100:+.2f}% | "
        f"J={float(entry.get('jaccard', 0.0)):.2f} | "
        f"compare={compare_label} | "
        f"{feature_text}"
    )
    return card[:260]


def _select_interval_review_cards_for_prompt(
    review_entries: Sequence[Dict[str, Any]],
    *,
    feature_priority: Sequence[str],
    limit: int,
) -> List[str]:
    if limit <= 0 or not review_entries:
        return []
    cards: List[str] = []
    seen = set()
    for entry in review_entries:
        case_id = str(entry.get("case_id", entry.get("decision_date", ""))).strip()
        if not case_id or case_id in seen:
            continue
        card = _build_interval_review_card(entry, feature_priority=feature_priority)
        if not card:
            continue
        cards.append(card)
        seen.add(case_id)
        if len(cards) >= limit:
            break
    return cards[:limit]


def _case_delta_fallback(
    *,
    day_text: str,
    day_pool: pd.DataFrame,
    teacher_symbols: Sequence[str],
    llm_symbols: Sequence[str],
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    teacher_only = day_pool[day_pool["symbol"].astype(str).isin(set(teacher_symbols) - set(llm_symbols))].copy()
    llm_only = day_pool[day_pool["symbol"].astype(str).isin(set(llm_symbols) - set(teacher_symbols))].copy()
    deltas: List[Tuple[str, float, str, float, float]] = []
    for feat in feature_cols:
        if feat not in day_pool.columns:
            continue
        t_vals = teacher_only[feat].dropna()
        l_vals = llm_only[feat].dropna()
        if len(t_vals) == 0 or len(l_vals) == 0:
            continue
        t_med = float(t_vals.median())
        l_med = float(l_vals.median())
        gap = t_med - l_med
        direction = "higher" if gap > 0 else "lower"
        deltas.append((feat, abs(gap), direction, t_med, l_med))
    deltas.sort(key=lambda item: item[1], reverse=True)
    top = deltas[:3]
    if not top:
        return {
            "verdict": "success" if set(teacher_symbols) == set(llm_symbols) else "mistake",
            "concise_memory": f"{day_text}: align more tightly with teacher picks and avoid drifting from the teacher-preferred reversal corridor.",
            "teacher_preference": "teacher preferred the stronger overlap set in the reversal corridor",
            "llm_mistake": "llm drifted from the teacher-preferred candidate subset",
            "correction_rule": "promote candidates nearer the teacher-preferred reversal corridor",
            "trigger_pattern": "teacher agreement on reversal structure",
        }
    pref_parts = []
    for feat, _, direction, t_med, l_med in top:
        pref_parts.append(f"{feat} {direction} (teacher={_format_float(t_med)} vs llm={_format_float(l_med)})")
    pref_text = "; ".join(pref_parts)
    return {
        "verdict": "success" if set(teacher_symbols) == set(llm_symbols) else "mistake",
        "concise_memory": f"{day_text}: prefer candidates with {pref_text}.",
        "teacher_preference": f"teacher-only picks tended to have {pref_text}",
        "llm_mistake": f"llm-only picks drifted away from {pref_text}",
        "correction_rule": f"when teacher and llm disagree, rank names with {pref_text} first",
        "trigger_pattern": "teacher-only versus llm-only median feature gap",
    }


def _top_prompt_features(round_id: str, top_n: int) -> List[str]:
    importance_df = _load_teacher_feature_importance(round_id)
    agg = (
        importance_df.groupby("feature", as_index=False)["importance_abs"]
        .mean()
        .sort_values("importance_abs", ascending=False)
        .reset_index(drop=True)
    )
    return agg["feature"].head(top_n).tolist()


def _top_teacher_features(round_id: str, top_n: int) -> List[str]:
    return _top_prompt_features(round_id, top_n)


def _teacher_meta(round_id: str, top_feature_count: int) -> Dict[str, Any]:
    spec = _load_teacher_spec(round_id)
    nav = _load_teacher_nav_summary(round_id)
    factor_summary = _load_teacher_factor_analysis_summary(round_id)
    return {
        "round_id": round_id,
        "title": spec["title"],
        "teacher_role": spec["teacher_role"],
        "research_family": spec["research_family"],
        "sample_template": spec["sample_template"],
        "model_family": spec["model_family"],
        "target_kind": spec["target_kind"],
        "top_prompt_features": _top_prompt_features(round_id, top_feature_count),
        "nav_cagr": float(nav["cagr"]),
        "nav_max_drawdown": float(nav["max_drawdown"]),
        "nav_positive_years": int(nav["positive_years"]),
        "nav_total_years": int(nav["total_years"]),
        "factor_analysis_summary": factor_summary,
    }


def _negative_teacher_meta(round_id: str, top_feature_count: int) -> Dict[str, Any]:
    teacher_model = _find_memory_item_by_round_id(MEMORY_ROOT / "teacher_models", round_id) or {}
    lesson = _find_memory_item_by_round_id(MEMORY_ROOT / "research_lessons", round_id) or {}
    spec = _load_teacher_spec(round_id)
    factor_summary = _load_teacher_factor_analysis_summary(round_id)
    try:
        nav = _load_teacher_nav_summary(round_id)
    except Exception:
        nav = {"cagr": float("nan"), "max_drawdown": float("nan"), "positive_years": 0, "total_years": 0}
    return {
        "round_id": round_id,
        "title": spec.get("title", teacher_model.get("title", round_id)),
        "teacher_role": spec.get("teacher_role", ""),
        "research_family": spec.get("research_family", teacher_model.get("research_family", "")),
        "sample_template": spec.get("sample_template", teacher_model.get("sample_template", "")),
        "model_family": spec.get("model_family", teacher_model.get("model_family", "")),
        "top_prompt_features": _top_prompt_features(round_id, top_feature_count),
        "nav_cagr": float(nav.get("cagr", float("nan"))),
        "nav_max_drawdown": float(nav.get("max_drawdown", float("nan"))),
        "nav_positive_years": int(nav.get("positive_years", 0) or 0),
        "nav_total_years": int(nav.get("total_years", 0) or 0),
        "accepted_as_teacher": teacher_model.get("accepted_as_teacher"),
        "rejection_reason": teacher_model.get("rejection_reason") or lesson.get("lesson_summary") or lesson.get("summary") or "",
        "lesson_summary": lesson.get("lesson_summary") or lesson.get("summary") or "",
        "recommended_action": lesson.get("recommended_action") or "",
        "factor_analysis_summary": factor_summary,
    }


def _join_features(pred_df: pd.DataFrame, master_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    subset_cols = ["symbol", "signal_date", *feature_cols]
    subset_cols = list(dict.fromkeys(subset_cols))
    feature_df = master_df.loc[
        master_df["signal_date"].between(pred_df["signal_date"].min(), pred_df["signal_date"].max()),
        subset_cols,
    ].copy()
    merged = pred_df.merge(feature_df, on=["symbol", "signal_date"], how="left")
    return merged


def _baseline_sort_spec(sample_template: str) -> Tuple[List[str], List[bool]]:
    if sample_template == "weak_state_reversal_pool":
        return (
            ["J", "ret_3_clip", "pos_20", "dist_to_20d_low", "days_J_below_20_last_10", "oversold_depth"],
            [True, True, True, True, False, False],
        )
    if sample_template == "hard_threshold_reversal_gate":
        return (["J", "ret_5", "pos_20", "amt_zscore_20", "oversold_depth"], [True, True, True, False, False])
    if sample_template == "trend_breakout_pool":
        return (["ret_20", "close_to_ma20", "pos_20", "amt_zscore_20", "body_pct"], [False, False, False, False, False])
    if sample_template == "trend_pullback_pool":
        return (["ret_20", "close_to_ma20", "pos_20", "ret_5", "amt_zscore_20"], [False, False, False, True, False])
    return (["symbol"], [True])


def _template_style_hint(sample_template: str) -> str:
    mapping = {
        "weak_state_reversal_pool": "weak-state snapback reversal",
        "hard_threshold_reversal_gate": "oversold hard-gate reversal",
        "trend_breakout_pool": "trend breakout continuation",
        "trend_pullback_pool": "trend pullback continuation",
    }
    return mapping.get(sample_template, sample_template)


def _basic_filter_hint(sample_template: str) -> str:
    sort_cols, ascending = _baseline_sort_spec(sample_template)
    if not sort_cols or sort_cols == ["symbol"]:
        return "no explicit baseline filter"
    return ", ".join(
        f"{col}:{'low-first' if asc else 'high-first'}"
        for col, asc in zip(sort_cols, ascending)
    )


def _scope_domain_card(
    meta: Mapping[str, Any],
    *,
    scope_round_id: Optional[str] = None,
    source_round_id: Optional[str] = None,
) -> Dict[str, Any]:
    sample_template = str(meta.get("sample_template", "") or "")
    explainability_round_id = str(scope_round_id or meta.get("round_id", "")).strip()
    return {
        "round_id": str(scope_round_id or meta.get("round_id", "")).strip(),
        "source_round_id": str(source_round_id or meta.get("round_id", "")).strip(),
        "explainability_round_id": explainability_round_id,
        "title": str(meta.get("title", "")).strip(),
        "teacher_role": str(meta.get("teacher_role", "")).strip(),
        "family": str(meta.get("research_family", "")).strip(),
        "template": sample_template,
        "style_hint": _template_style_hint(sample_template),
        "basic_filter": _basic_filter_hint(sample_template),
        "top_features": list(meta.get("top_prompt_features", []) or []),
    }


def _teacher_scope_entries(scoped_warmup_state: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(scoped_warmup_state, Mapping):
        return []
    raw = scoped_warmup_state.get("teacher_scopes", [])
    if isinstance(raw, Mapping):
        scopes = list(raw.values())
    elif isinstance(raw, list):
        scopes = raw
    else:
        scopes = []
    clean: List[Dict[str, Any]] = []
    for item in scopes:
        if isinstance(item, Mapping):
            clean.append(dict(item))
    return clean


def _find_teacher_scope_entry(
    scoped_warmup_state: Optional[Mapping[str, Any]], round_id: str
) -> Optional[Dict[str, Any]]:
    target = str(round_id).strip()
    if not target:
        return None
    for scope in _teacher_scope_entries(scoped_warmup_state):
        candidates = {
            str(scope.get("round_id", "")).strip(),
            str(scope.get("source_round_id", "")).strip(),
            str(scope.get("warmup_source_round_id", "")).strip(),
        }
        if target in candidates:
            return scope
    return None


def _prompt_global_lesson_lines(
    *,
    scoped_warmup_state: Optional[Mapping[str, Any]],
    fallback_lines: Optional[Sequence[str]],
    limit: int,
) -> List[str]:
    raw: Sequence[str] = []
    if isinstance(scoped_warmup_state, Mapping):
        raw = (
            scoped_warmup_state.get("global_lesson_zone_lines")
            or scoped_warmup_state.get("global_lesson_zone")
            or []
        )
    if not raw:
        raw = list(fallback_lines or [])
    return [str(line).strip() for line in raw if str(line).strip()][:limit]


def _prompt_scope_lesson_lines(
    *,
    scoped_warmup_state: Optional[Mapping[str, Any]],
    round_id: str,
    limit: int,
) -> List[str]:
    scope = _find_teacher_scope_entry(scoped_warmup_state, round_id)
    if not scope:
        return []
    raw = scope.get("scope_lesson_zone_lines") or scope.get("scope_lesson_zone") or scope.get("lesson_zone") or []
    return [str(line).strip() for line in raw if str(line).strip()][:limit]


def _prompt_scope_review_cards(
    *,
    scoped_warmup_state: Optional[Mapping[str, Any]],
    round_id: str,
    fallback_cards: Optional[Sequence[str]],
    limit: int,
) -> List[str]:
    scope = _find_teacher_scope_entry(scoped_warmup_state, round_id)
    raw: Sequence[str] = []
    if scope:
        raw = scope.get("review_cards_for_prompt") or scope.get("retained_review_cards") or []
    if not raw:
        raw = list(fallback_cards or [])
    return [str(line).strip() for line in raw if str(line).strip()][:limit]


def _aggregate_scope_review_cards(
    scoped_warmup_state: Optional[Mapping[str, Any]], *, limit: int
) -> List[str]:
    cards: List[str] = []
    seen = set()
    for scope in _teacher_scope_entries(scoped_warmup_state):
        for card in list(scope.get("review_cards_for_prompt") or [])[: max(1, limit // max(len(_teacher_scope_entries(scoped_warmup_state)), 1))]:
            text = str(card).strip()
            if not text or text in seen:
                continue
            cards.append(text)
            seen.add(text)
            if len(cards) >= limit:
                return cards
    return cards[:limit]


def _render_all_scope_lesson_lines(
    *, scoped_warmup_state: Optional[Mapping[str, Any]], lesson_limit_per_scope: int
) -> List[str]:
    lines: List[str] = []
    for scope in _teacher_scope_entries(scoped_warmup_state):
        header = (
            f"Scope {str(scope.get('round_id', '')).strip()}: "
            f"family={str(scope.get('family', '')).strip()}, "
            f"template={str(scope.get('template', '')).strip()}, "
            f"style={str(scope.get('style_hint', '')).strip()}, "
            f"basic_filter={str(scope.get('basic_filter', '')).strip()}"
        )
        lines.append(header)
        top_features = ",".join(list(scope.get("top_features") or [])[:6]) or "none"
        lines.append(f"Scope top_features: {top_features}")
        for lesson in list(scope.get("scope_lesson_zone_lines") or scope.get("scope_lesson_zone") or [])[:lesson_limit_per_scope]:
            text = str(lesson).strip()
            if text:
                lines.append(text)
    return lines


def _candidate_rows_to_records(df: pd.DataFrame, feature_cols: Sequence[str], *, teacher_fields: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        features = {}
        for col in feature_cols:
            value = row.get(col)
            if pd.isna(value):
                features[col] = None
            else:
                features[col] = round(float(value), 4)
        record = {
            "symbol": row["symbol"],
            "candidate_id": row.get("_candidate_id"),
            "signal_date": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"),
            "entry_date": pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d"),
            "features": features,
        }
        for field in teacher_fields:
            value = row.get(field)
            if pd.isna(value):
                record[field] = None
            elif isinstance(value, (int, np.integer)):
                record[field] = int(value)
            else:
                try:
                    record[field] = round(float(value), 6)
                except (TypeError, ValueError):
                    record[field] = value
        records.append(record)
    return records


def _current_holdings(selected_rows: pd.DataFrame, decision_date: pd.Timestamp) -> List[Dict[str, Any]]:
    if selected_rows.empty:
        return []
    active = selected_rows[
        (pd.to_datetime(selected_rows["entry_date"]) <= decision_date)
        & (pd.to_datetime(selected_rows["exit_date"]) > decision_date)
    ].copy()
    holdings = []
    for _, row in active.sort_values(["exit_date", "symbol"]).iterrows():
        days_left = (pd.Timestamp(row["exit_date"]) - decision_date).days
        holdings.append(
            {
                "symbol": row["symbol"],
                "entry_date": pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d"),
                "days_to_exit": int(days_left),
                "source_signal_date": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"),
            }
        )
    return holdings


def _chat_completion(
    *,
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    force_local_qwen_no_thinking: bool = True,
    fail_fast_on_empty_content: bool = False,
    max_retries: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    request_body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    if force_local_qwen_no_thinking and _use_local_qwen_no_thinking(model):
        # Local vLLM-served Qwen3 obeys this and stops emitting long <think> traces.
        request_body["chat_template_kwargs"] = {"enable_thinking": False}
    if model.startswith("gpt-"):
        request_body["reasoning_effort"] = "low" if _is_gpt_oss_model(model) else "minimal"
        if not _prefers_line_id_protocol(model):
            request_body["response_format"] = {"type": "json_object"}
    if seed is not None and int(seed) > 0:
        request_body["seed"] = int(seed)
    headers = {
        key: value
        for key, value in {
            "Authorization": f"Bearer {api_key}" if api_key else None,
            "Content-Type": "application/json",
        }.items()
        if value is not None
    }
    last_exc: Optional[Exception] = None
    retryable_http_codes = {429, 502, 503, 504}
    redirect_http_codes = {301, 302, 303, 307, 308}
    effective_max_retries = max(1, int(API_MAX_RETRIES if max_retries is None else max_retries))
    for attempt in range(effective_max_retries):
        try:
            request_url = API_URL
            redirect_hops = 0
            while True:
                session = requests.Session()
                if _is_local_http_url(request_url):
                    session.trust_env = False
                response = session.post(
                    request_url,
                    headers=headers,
                    json=request_body,
                    timeout=API_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
                if response.status_code in redirect_http_codes:
                    location = response.headers.get("Location", "").strip()
                    if not location:
                        raise RuntimeError(
                            f"API redirect {response.status_code} without Location header: {response.text[:500]}"
                        )
                    request_url = location
                    redirect_hops += 1
                    if redirect_hops > 5:
                        raise RuntimeError(f"API redirect loop detected for {API_URL}")
                    continue
                if response.status_code >= 400:
                    body = response.text
                    last_exc = RuntimeError(f"API HTTPError {response.status_code}: {body}")
                    if response.status_code not in retryable_http_codes or attempt >= effective_max_retries - 1:
                        raise last_exc
                    break
                payload = response.json()
                try:
                    choice0 = payload.get("choices", [{}])[0]
                    message0 = choice0.get("message", {}) or {}
                    content = str(message0.get("content", "") or "").strip()
                except Exception:
                    choice0 = {}
                    message0 = {}
                    content = ""
                if not content:
                    finish_reason = str(choice0.get("finish_reason", "") or "").strip()
                    reasoning = str(message0.get("reasoning_content", "") or "")
                    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
                    last_exc = RuntimeError(
                        "APIEmptyContent "
                        f"model={model} "
                        f"finish_reason={finish_reason or 'unknown'} "
                        f"reasoning_len={len(reasoning)} "
                        f"content_len={len(content)} "
                        f"usage={json.dumps(usage, ensure_ascii=False)}"
                    )
                    if fail_fast_on_empty_content:
                        raise last_exc
                    # For GPT-OSS on local vLLM, repeating the exact same long prompt after
                    # a 200 OK with empty visible content is usually pointless: the model has
                    # already consumed the completion budget inside hidden reasoning. Fail fast
                    # so the caller can downgrade to the compact retry prompt instead of
                    # burning many minutes on identical retries.
                    if _is_gpt_oss_model(model) and finish_reason.lower() == "length" and len(reasoning) > 0:
                        raise last_exc
                    if attempt >= effective_max_retries - 1:
                        raise last_exc
                    break
                return payload
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= effective_max_retries - 1:
                raise last_exc from exc
        except Exception as exc:  # network timeout / transient transport failures
            last_exc = exc
            if attempt >= effective_max_retries - 1:
                break
        sleep_seconds = min(API_RETRY_BASE_SLEEP * (attempt + 1), API_RETRY_MAX_SLEEP)
        time.sleep(sleep_seconds)
    raise RuntimeError(f"API request failed after retries: {last_exc}") from last_exc


def _compute_yearly_returns(nav: pd.Series) -> Dict[int, float]:
    if nav is None or len(nav) == 0:
        return {}
    yearly_returns: Dict[int, float] = {}
    prev_nav = 1.0
    for year in sorted(nav.index.year.unique()):
        year_nav = nav[nav.index.year == year]
        if len(year_nav) == 0:
            continue
        year_end_nav = float(year_nav.iloc[-1])
        yearly_returns[int(year)] = year_end_nav / prev_nav - 1.0
        prev_nav = year_end_nav
    return yearly_returns


def _compute_max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) == 0:
        return 0.0
    peaks = nav.cummax()
    dd = nav / peaks - 1.0
    return float(dd.min())


def _compute_nav_metrics(nav: pd.Series) -> Dict[str, Any]:
    if nav is None or len(nav) == 0:
        return {
            "final_nav": 1.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "yearly_returns": {},
        }
    total_return = float(nav.iloc[-1] - 1.0)
    cagr = float(nav.iloc[-1] ** (252.0 / len(nav)) - 1.0) if len(nav) else float("nan")
    return {
        "final_nav": float(nav.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": _compute_max_drawdown(nav),
        "yearly_returns": _compute_yearly_returns(nav),
    }


def _plot_overlay(
    *,
    output_path: Path,
    title: str,
    llm_nav: pd.Series,
    teacher_target_nav: pd.Series,
    teacher_full_nav: pd.Series,
    hs300_nav: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(llm_nav.index, llm_nav.values, color="#1f6feb", linewidth=2.2, label="LLM Apprentice NAV")
    if np.allclose(teacher_target_nav.values, teacher_full_nav.values):
        ax.plot(
            teacher_target_nav.index,
            teacher_target_nav.values,
            color="#16a34a",
            linewidth=2.0,
            linestyle="--",
            label="Teacher Reference NAV",
        )
    else:
        ax.plot(
            teacher_target_nav.index,
            teacher_target_nav.values,
            color="#16a34a",
            linewidth=2.0,
            linestyle="--",
            label="Teacher Target NAV",
        )
        ax.plot(
            teacher_full_nav.index,
            teacher_full_nav.values,
            color="#f59e0b",
            linewidth=1.8,
            linestyle="-.",
            label="Teacher Full Q5 NAV",
        )
    ax.plot(hs300_nav.index, hs300_nav.values, color="#7f8c8d", linewidth=1.5, linestyle=":", label="HS300 NAV")
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", framealpha=0.92)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _teacher_weight(meta: Dict[str, Any]) -> float:
    cagr = max(float(meta["nav_cagr"]), 0.01)
    return cagr


def _build_single_teacher_frame(config: ApprenticeReplayConfig, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    round_id = config.teacher_round_ids[0]
    meta = _teacher_meta(round_id, config.prompt_feature_count)
    pred_df = _load_teacher_predictions(round_id, start_date=config.start_date, end_date=config.end_date)
    prompt_features = list(meta["top_prompt_features"])
    lesson_features = _top_teacher_features(round_id, max(config.lesson_feature_count, config.prompt_feature_count))
    baseline_sort_features, _ = _baseline_sort_spec(meta["sample_template"])
    join_features = list(dict.fromkeys(prompt_features + lesson_features + baseline_sort_features))
    merged = _join_features(pred_df, master_df, join_features)
    merged["sample_template"] = meta["sample_template"]
    merged = merged.sort_values(["signal_date", "score"], ascending=[True, False]).reset_index(drop=True)
    return merged, meta


def _baseline_daily_pool(day_df: pd.DataFrame, sample_template: str, candidate_pool_size: int) -> pd.DataFrame:
    frame = day_df.copy()
    sort_cols, ascending = _baseline_sort_spec(sample_template)
    existing_cols = [col for col in sort_cols if col in frame.columns]
    existing_ascending = [ascending[sort_cols.index(col)] for col in existing_cols]
    if existing_cols:
        frame = frame.sort_values(existing_cols, ascending=existing_ascending, kind="mergesort")
    return frame.head(candidate_pool_size).copy()


def _build_multi_teacher_frame(
    config: ApprenticeReplayConfig, master_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
    metas = [_teacher_meta(round_id, max(4, config.prompt_feature_count // 2)) for round_id in config.teacher_round_ids]
    prompt_feature_union: List[str] = []
    lesson_feature_union: List[str] = []
    for meta in metas:
        for feat in meta["top_prompt_features"]:
            if feat not in prompt_feature_union:
                prompt_feature_union.append(feat)
        for feat in _top_teacher_features(meta["round_id"], max(config.lesson_feature_count, config.prompt_feature_count)):
            if feat not in lesson_feature_union:
                lesson_feature_union.append(feat)
    prompt_feature_union = prompt_feature_union[: config.prompt_feature_count]
    lesson_feature_union = lesson_feature_union[: max(config.lesson_feature_count, config.prompt_feature_count)]

    per_teacher_daily: List[pd.DataFrame] = []
    weights = {meta["round_id"]: _teacher_weight(meta) for meta in metas}
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    for meta in metas:
        pred_df = _load_teacher_predictions(meta["round_id"], start_date=config.start_date, end_date=config.end_date)
        joined = _join_features(pred_df, master_df, lesson_feature_union)
        joined = joined.sort_values(["signal_date", "score"], ascending=[True, False]).reset_index(drop=True)
        joined["teacher_round_id"] = meta["round_id"]
        daily_parts = []
        for _, day_df in joined.groupby("signal_date", sort=True):
            preferred = day_df[day_df["bucket"] >= 4].copy()
            if preferred.empty:
                preferred = day_df.copy()
            preferred = preferred.head(config.candidate_pool_size).copy()
            if preferred.empty:
                continue
            preferred["teacher_rank"] = np.arange(1, len(preferred) + 1)
            preferred["teacher_rank_pct"] = 1.0 - (preferred["teacher_rank"] - 1) / max(len(preferred), 1)
            daily_parts.append(preferred)
        if daily_parts:
            per_teacher_daily.append(pd.concat(daily_parts, ignore_index=True))

    if not per_teacher_daily:
        return pd.DataFrame(), metas, prompt_feature_union

    stacked = pd.concat(per_teacher_daily, ignore_index=True)
    merge_keys = ["symbol", "signal_date", "entry_date", "exit_date", "future_return_5d", "return_20d", *lesson_feature_union]
    agg_rows: List[Dict[str, Any]] = []
    for keys, group in stacked.groupby(merge_keys, dropna=False, sort=True):
        row = {
            "symbol": keys[0],
            "signal_date": keys[1],
            "entry_date": keys[2],
            "exit_date": keys[3],
            "future_return_5d": keys[4],
            "return_20d": keys[5],
        }
        for idx, feat in enumerate(lesson_feature_union, start=6):
            row[feat] = keys[idx]
        ensemble_score = 0.0
        present_count = 0
        for meta in metas:
            rid = meta["round_id"]
            sub = group[group["teacher_round_id"] == rid]
            if sub.empty:
                row[f"{rid}_score"] = np.nan
                row[f"{rid}_bucket"] = np.nan
                row[f"{rid}_rank_pct"] = 0.0
                continue
            present_count += 1
            score = float(sub.iloc[0]["score"])
            bucket = int(sub.iloc[0]["bucket"])
            rank_pct = float(sub.iloc[0]["teacher_rank_pct"])
            row[f"{rid}_score"] = score
            row[f"{rid}_bucket"] = bucket
            row[f"{rid}_rank_pct"] = rank_pct
            ensemble_score += weights[rid] * rank_pct
        row["teacher_present_count"] = present_count
        row["ensemble_score"] = ensemble_score
        agg_rows.append(row)

    merged = pd.DataFrame(agg_rows).sort_values(["signal_date", "ensemble_score"], ascending=[True, False]).reset_index(drop=True)

    daily_keep = []
    for _, day_df in merged.groupby("signal_date", sort=True):
        daily_keep.append(day_df.head(config.candidate_pool_size).copy())
    return pd.concat(daily_keep, ignore_index=True), metas, prompt_feature_union


def _single_teacher_target(frame: pd.DataFrame, config: ApprenticeReplayConfig) -> pd.DataFrame:
    target_parts = []
    pool_parts = []
    sample_template = str(frame["sample_template"].iloc[0]) if "sample_template" in frame.columns and len(frame) else ""
    for _, day_df in frame.groupby("signal_date", sort=True):
        if config.candidate_source == "baseline_signal":
            pool = _baseline_daily_pool(day_df, sample_template, config.candidate_pool_size)
        else:
            preferred = day_df[day_df["bucket"] >= 4].copy()
            if preferred.empty:
                preferred = day_df.copy()
            pool = preferred.head(config.candidate_pool_size).copy()
        pool_parts.append(pool)
        target = pool.sort_values(["score", "bucket"], ascending=[False, False]).head(config.teacher_daily_pick_count).copy()
        target_parts.append(target)
    return pd.concat(pool_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def _multi_teacher_target(frame: pd.DataFrame, config: ApprenticeReplayConfig) -> pd.DataFrame:
    if frame.empty or "signal_date" not in frame.columns:
        raise ValueError(
            "multi-teacher candidate frame is empty or missing signal_date. "
            "This usually means the chosen teacher pack has no predictions in the requested replay/warmup window."
        )
    target_parts = []
    pool_parts = []
    for _, day_df in frame.groupby("signal_date", sort=True):
        pool = day_df.sort_values(["ensemble_score", "teacher_present_count"], ascending=[False, False]).head(
            config.candidate_pool_size
        )
        pool_parts.append(pool)
        target = pool.sort_values(["ensemble_score", "teacher_present_count"], ascending=[False, False]).head(
            config.teacher_daily_pick_count
        )
        target_parts.append(target)
    return pd.concat(pool_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def _daily_prompt_single(
    *,
    config: ApprenticeReplayConfig,
    decision_date: pd.Timestamp,
    meta: Dict[str, Any],
    negative_metas: Optional[Sequence[Dict[str, Any]]] = None,
    candidate_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    warmup_lessons: Optional[Sequence[str]] = None,
    warmup_review_cards: Optional[Sequence[str]] = None,
    scoped_warmup_state: Optional[Mapping[str, Any]] = None,
    current_scope_round_id: Optional[str] = None,
) -> Tuple[str, str]:
    teacher_fields = ["score", "bucket"] if config.include_teacher_signal else []
    prompt_features = meta["top_prompt_features"]
    candidates = _candidate_rows_to_records(candidate_df, prompt_features, teacher_fields=teacher_fields)
    holdings = _current_holdings(selected_rows, decision_date)
    explainability_only = _uses_explainability_only_prompt(config)
    branch_report_context = _uses_branch_report_context(config)
    kb_context = [] if explainability_only else _kb_context_for_round_ids([meta["round_id"]])
    research_experience_digest = "" if explainability_only else _load_research_experience_digest()
    scope_round_id = str(current_scope_round_id or meta["round_id"]).strip()
    global_lesson_lines = (
        []
        if explainability_only
        else _prompt_global_lesson_lines(
            scoped_warmup_state=scoped_warmup_state,
            fallback_lines=warmup_lessons,
            limit=config.warmup_lesson_zone_max_lines,
        )
    )
    current_scope_lesson_lines = (
        []
        if explainability_only
        else _prompt_scope_lesson_lines(
            scoped_warmup_state=scoped_warmup_state,
            round_id=scope_round_id,
            limit=config.warmup_lesson_zone_max_lines,
        )
    )
    current_scope_review_cards = (
        []
        if explainability_only
        else _prompt_scope_review_cards(
            scoped_warmup_state=scoped_warmup_state,
            round_id=scope_round_id,
            fallback_cards=warmup_review_cards,
            limit=config.warmup_review_memory_limit,
        )
    )
    current_scope_entry = _find_teacher_scope_entry(scoped_warmup_state, scope_round_id)
    explainability_preferred_round_ids = [
        str(
            (current_scope_entry or {}).get("explainability_round_id")
            or (current_scope_entry or {}).get("round_id")
            or scope_round_id
            or meta.get("round_id", "")
        ).strip(),
        str(meta.get("round_id", "")).strip(),
    ]
    explainability_summary, explainability_round_id = _resolve_explainability_summary(
        preferred_round_ids=explainability_preferred_round_ids,
        fallback_summary=meta.get("factor_analysis_summary", {}),
    )
    reasoning_instruction = _compact_reasoning_instruction(config)
    mode_guidance = (
        "You are only given the teacher's branch-oriented explainability pack. "
        "Use branch cards as conditional templates, soft rules as fuzzy preferences, hard veto rules as avoid zones, "
        "and meta rules as conflict-resolution hints. "
        "Trading is fuzzy and relative: do not overfit tiny threshold differences, and do not wait for perfect agreement. "
        if explainability_only
        else (
            "Teacher explainability pack is theory guidance. "
            "Global Lesson Zone and Current Teacher Scope Lesson Zone are practice-corrected memory distilled from warmup. "
            "When theory and lesson disagree, prefer the lesson if the warmup evidence clearly supports it. "
            "Use the report to understand branch logic, but use the lesson zones to operationalize ranking and abstain behavior. "
            if branch_report_context
            else "When Global Lesson Zone and Current Teacher Scope Lesson Zone both exist, treat the global zone as cross-teacher meta rules "
            "and the current scope zone as the active evaluation regime for this single-teacher warmup. "
        )
    )
    if _prefers_line_id_protocol(config.api_model):
        teacher_line = (
            f"Teacher {meta['round_id']}: family={meta['research_family']}, template={meta['sample_template']}, "
            f"nav_cagr={meta['nav_cagr']:.4f}, nav_mdd={meta['nav_max_drawdown']:.4f}, "
            f"top_features={','.join(prompt_features)}"
        )
        teacher_band_line = None
        teacher_factor_line = None
        explainability_lines: List[str] = []
        explainability_ref_line = ""
        if explainability_only:
            explainability_lines = _compact_branch_rule_lines(explainability_summary)
            if explainability_round_id:
                explainability_ref_line = (
                    f"Explainability reference: {explainability_round_id} | scoring source: {meta['round_id']}"
                )
        else:
            if _uses_report_v2_with_lessons_prompt(config):
                explainability_lines = _compact_branch_rule_lines(explainability_summary)
                if explainability_round_id:
                    explainability_ref_line = (
                        f"Explainability reference: {explainability_round_id} | scoring source: {meta['round_id']}"
                    )
            else:
                teacher_band_line = (
                    "Teacher bands: "
                    + (
                        _compact_band_hint(meta.get("preference_bands", []))
                        if _compact_band_hint(meta.get("preference_bands", [])) != "no strong band preference derived"
                        else "none derived for this teacher"
                    )
                )
                teacher_factor_line = "Teacher factor rules: " + _compact_factor_rule_hint(meta.get("factor_analysis_summary", {}))
        kb_lines = []
        for item in kb_context:
            kb_lines.append(f"KB {item['round_id']}: {_compact_kb_hint(item)}")
        negative_lines = []
        for neg in list(negative_metas or [])[:6]:
            negative_lines.append(
                f"Avoid {neg['round_id']}: family={neg['research_family']}, template={neg['sample_template']}, "
                f"nav_cagr={_format_float(neg['nav_cagr'])}, weak_note={str(neg.get('rejection_reason') or neg.get('lesson_summary') or '')[:140]}"
            )
        candidate_lines = []
        for rec in candidates:
            feats = rec["features"]
            feat_text = ", ".join(f"{k}={feats[k]}" for k in prompt_features if k in feats)
            candidate_lines.append(
                f"{rec.get('candidate_id')} | symbol={rec['symbol']} | entry={rec['entry_date']} | exit={rec['exit_date']} | {feat_text}"
            )
        holding_lines = [
            f"{h['symbol']} exit={h['exit_date']} days_to_exit={h['days_to_exit']}" for h in holdings
        ] or ["none"]
        system = "".join(
            [
                "You are the QuantApprentice trader apprentice. ",
                "This is a real daily trading decision, not a research summary task. ",
                "Choose exactly 4 candidate IDs that best match the teacher, or ABSTAIN if none are attractive enough. ",
                mode_guidance,
                f"{reasoning_instruction} ",
                "If you need more evidence, you may issue at most one QUERY for a single candidate before the final decision. ",
                "QUERY format: QUERY: C03 | FOCUS: close_to_ma20, volatility_20 | REASON: short phrase. ",
                "After the code-agent returns extra feature evidence, you must make the final SELECT or ABSTAIN decision. ",
                "Do not write next-step research plans, do not discuss future experiments, and do not summarize hypotheses. ",
                "Reply with exactly one line in one of these formats: ",
                "SELECT: C01,C04,C09,C11 | REASON: short phrase. ",
                "ABSTAIN | REASON: short phrase. ",
                "QUERY: C03 | FOCUS: close_to_ma20, volatility_20 | REASON: short phrase. ",
                "Never output 0, NONE, or the prompt text. ",
                "Do not explain your reasoning.",
            ]
        )
        user = "\n".join(
            [
                f"Decision date: {decision_date.strftime('%Y-%m-%d')}",
                (
                    f"Current learning scope: round_id={scope_round_id}, "
                    f"family={str(current_scope_entry.get('family', meta.get('research_family', ''))).strip()}, "
                    f"template={str(current_scope_entry.get('template', meta.get('sample_template', ''))).strip()}, "
                    f"style={str(current_scope_entry.get('style_hint', _template_style_hint(meta.get('sample_template', '')))).strip()}, "
                    f"basic_filter={str(current_scope_entry.get('basic_filter', _basic_filter_hint(meta.get('sample_template', '')))).strip()}"
                )
                if current_scope_entry
                else (
                    f"Current learning scope: round_id={scope_round_id}, "
                    f"family={meta.get('research_family', '')}, template={meta.get('sample_template', '')}, "
                    f"style={_template_style_hint(meta.get('sample_template', ''))}, "
                    f"basic_filter={_basic_filter_hint(meta.get('sample_template', ''))}"
                )
                ,
                teacher_line,
                *([teacher_band_line] if teacher_band_line else []),
                *([teacher_factor_line] if teacher_factor_line else []),
                *(
                    [
                        *([explainability_ref_line] if explainability_ref_line else []),
                        "Teacher explainability pack (Report v2 branch-oriented):",
                        *(explainability_lines or ["none"]),
                    ]
                    if explainability_only
                    else (
                        [
                            *([explainability_ref_line] if explainability_ref_line else []),
                            "Teacher explainability pack (Report v2 branch-oriented):",
                            *(explainability_lines or ["none"]),
                            *kb_lines,
                            "Negative teacher memory:",
                            *(negative_lines or ["none"]),
                            "Global Lesson Zone:",
                            *(global_lesson_lines or ["none"]),
                            "Current Teacher Scope Lesson Zone:",
                            *(current_scope_lesson_lines or ["none"]),
                            "Current Scope Practice Review Area:",
                            *(current_scope_review_cards or ["none"]),
                            "Outer-loop background memory (context only, not a direct rulebook):",
                            *(research_experience_digest.splitlines() if research_experience_digest else ["none"]),
                        ]
                        if _uses_report_v2_with_lessons_prompt(config)
                        else [
                            *kb_lines,
                            "Negative teacher memory:",
                            *(negative_lines or ["none"]),
                            "Global Lesson Zone:",
                            *(global_lesson_lines or ["none"]),
                            "Current Teacher Scope Lesson Zone:",
                            *(current_scope_lesson_lines or ["none"]),
                            "Current Scope Practice Review Area:",
                            *(current_scope_review_cards or ["none"]),
                            "Outer-loop background memory (context only, not a direct rulebook):",
                            *(research_experience_digest.splitlines() if research_experience_digest else ["none"]),
                        ]
                    )
                ),
                f"Current holdings: {'; '.join(holding_lines)}",
                "Feature-query policy:",
                "- You may ask one QUERY only if the current candidates do not provide enough evidence for a confident trade.",
                "- Use QUERY to inspect one candidate's missing feature dimensions, then finalize the decision.",
                "Candidates:",
                *candidate_lines,
            ]
        )
        return system, user
    system = (
        "You are the QuantApprentice trader apprentice. "
        "Your task is to imitate the provided teacher's next-slot stock selection behavior. "
        "Return strict JSON only. "
        "Do not include markdown. "
        "Do not spend tokens on hidden reasoning. "
        "Answer with one compact JSON object immediately."
    )
    user_payload = {
        "task": "single_teacher_imitation",
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "prompt_recipe": (
            "explainability_only"
            if explainability_only
            else ("report_v2_with_lessons" if _uses_report_v2_with_lessons_prompt(config) else "standard")
        ),
        "selection_rule": {
            "choose_up_to": config.llm_max_daily_picks,
            "mode": "buy_only_for_tomorrow_slot",
            "existing_holdings_auto_hold_until_exit": True,
            "objective": "match the teacher's likely top selections for this decision day",
        },
        "teacher_summary": {
            "round_id": meta["round_id"],
            "title": meta["title"],
            "role": meta["teacher_role"],
            "family": meta["research_family"],
            "template": meta["sample_template"],
            "nav_cagr": round(meta["nav_cagr"], 6),
            "nav_max_drawdown": round(meta["nav_max_drawdown"], 6),
            "nav_positive_years": f"{meta['nav_positive_years']}/{meta['nav_total_years']}",
            "top_features": prompt_features,
            "preferred_feature_bands": []
            if (explainability_only or _uses_report_v2_with_lessons_prompt(config))
            else meta.get("preference_bands", []),
            "factor_rules": {} if (explainability_only or branch_report_context) else meta.get("factor_analysis_summary", {}),
            "teacher_explainability_reference_round_id": explainability_round_id or scope_round_id,
            "teacher_explainability_pack": (
                _teacher_explainability_payload(explainability_summary)
                if branch_report_context
                else {}
            ),
            "teacher_signal_visible": config.include_teacher_signal,
        },
        "feature_query_policy": {
            "allowed": True,
            "max_queries": 1,
            "scope": "single candidate per decision day",
            "purpose": "inspect missing feature dimensions before final trade selection",
        },
        "knowledge_base_context": kb_context,
        "trading_memory_no_signal": research_experience_digest,
        "negative_teacher_memory": [] if explainability_only else list(negative_metas or []),
        "global_lesson_zone": list(global_lesson_lines),
        "teacher_scope_domain": current_scope_entry or _scope_domain_card(meta, scope_round_id=scope_round_id),
        "teacher_scope_review_memory": list(current_scope_review_cards or []),
        "teacher_scope_lessons": list(current_scope_lesson_lines or []),
        "current_holdings": holdings,
        "candidates": candidates,
        "required_output": {
            "selected_symbols": ["000001", "600519"],
            "brief_reason": "one short sentence",
        },
    }
    return system, json.dumps(user_payload, ensure_ascii=False)


def _daily_prompt_multi(
    *,
    config: ApprenticeReplayConfig,
    decision_date: pd.Timestamp,
    metas: List[Dict[str, Any]],
    negative_metas: Optional[Sequence[Dict[str, Any]]] = None,
    prompt_features: Sequence[str],
    candidate_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    warmup_lessons: Optional[Sequence[str]] = None,
    warmup_review_cards: Optional[Sequence[str]] = None,
    scoped_warmup_state: Optional[Mapping[str, Any]] = None,
    current_scope_round_id: Optional[str] = None,
) -> Tuple[str, str]:
    teacher_ids = [meta["round_id"] for meta in metas]
    teacher_fields: List[str] = []
    if config.include_teacher_signal:
        for rid in teacher_ids:
            teacher_fields.extend([f"{rid}_score", f"{rid}_bucket", f"{rid}_rank_pct"])
        teacher_fields.extend(["ensemble_score", "teacher_present_count"])
    candidates = _candidate_rows_to_records(candidate_df, prompt_features, teacher_fields=teacher_fields)
    holdings = _current_holdings(selected_rows, decision_date)
    explainability_only = _uses_explainability_only_prompt(config)
    branch_report_context = _uses_branch_report_context(config)
    kb_context = [] if explainability_only else _kb_context_for_round_ids(teacher_ids)
    research_experience_digest = "" if explainability_only else _load_research_experience_digest()
    global_lesson_lines = (
        []
        if explainability_only
        else _prompt_global_lesson_lines(
            scoped_warmup_state=scoped_warmup_state,
            fallback_lines=warmup_lessons,
            limit=config.warmup_lesson_zone_max_lines,
        )
    )
    scope_lesson_lines = (
        []
        if explainability_only
        else _render_all_scope_lesson_lines(
            scoped_warmup_state=scoped_warmup_state,
            lesson_limit_per_scope=min(config.warmup_lesson_zone_max_lines, 6),
        )
    )
    review_cards_for_prompt = (
        []
        if explainability_only
        else (
            _aggregate_scope_review_cards(scoped_warmup_state, limit=config.warmup_review_memory_limit)
            if scoped_warmup_state
            else list(warmup_review_cards or [])
        )
    )
    reasoning_instruction = _compact_reasoning_instruction(config)
    mode_guidance = (
        "You are only given each teacher's branch-oriented explainability pack. "
        "Use branch cards as conditional templates, soft rules as fuzzy preferences, hard veto rules as avoid zones, "
        "and meta rules as conflict-resolution hints. Trading is fuzzy and relative: do not overfit tiny threshold differences. "
        if explainability_only
        else (
            "Teacher explainability pack is theory guidance. "
            "Global Lesson Zone contains cross-teacher meta rules refined by warmup. "
            "Teacher-Scoped Lesson Zones contain family/template-specific practical corrections distilled from warmup. "
            "If theory and lesson disagree, prefer the lesson when the warmup evidence clearly supports it. "
            "Use negative teacher memory only as avoid-style constraints, not as primary targets. "
            "Outer-loop background memory is context only: use it to understand what each pool or teacher was trying to capture, not as a direct threshold rulebook. "
            "For each decision, infer which scope each candidate most resembles and use those scope lessons as fuzzy guidance rather than requiring exact threshold matches. "
            if branch_report_context
            else "Use negative teacher memory only as avoid-style constraints, not as primary targets. "
            "Global Lesson Zone contains cross-teacher meta rules. "
            "Teacher-Scoped Lesson Zones contain family/template-specific evaluation standards. "
            "Outer-loop background memory is context only: use it to understand what each pool or teacher was trying to capture, not as a direct threshold rulebook. "
            "For each decision, infer which scope each candidate most resembles and use those scope lessons as fuzzy guidance rather than requiring exact threshold matches. "
        )
    )
    if _prefers_line_id_protocol(config.api_model):
        teacher_lines = []
        explainability_lines = []
        for meta in metas:
            explainability_summary, explainability_round_id = _resolve_explainability_summary(
                preferred_round_ids=[str(meta.get("round_id", "")).strip()],
                fallback_summary=meta.get("factor_analysis_summary", {}),
            )
            base = (
                f"Teacher {meta['round_id']}: family={meta['research_family']}, template={meta['sample_template']}, "
                f"nav_cagr={meta['nav_cagr']:.4f}, nav_mdd={meta['nav_max_drawdown']:.4f}, "
                f"top_features={','.join(meta['top_prompt_features'])}"
            )
            if not branch_report_context:
                base += f", factor_rules={_compact_factor_rule_hint(meta.get('factor_analysis_summary', {}), top_features=2, top_combos=1)}"
            teacher_lines.append(base)
            if branch_report_context:
                if explainability_round_id:
                    explainability_lines.append(
                        f"Explainability reference: {explainability_round_id} | scoring source: {meta['round_id']}"
                    )
                explainability_lines.append(f"Teacher {meta['round_id']} explainability pack:")
                explainability_lines.extend(_compact_branch_rule_lines(explainability_summary))
        band_lines = [] if (explainability_only or branch_report_context) else _band_lines_for_metas(metas)
        kb_lines = []
        for item in kb_context:
            kb_lines.append(f"KB {item['round_id']}: {_compact_kb_hint(item)}")
        negative_lines = []
        for neg in list(negative_metas or [])[:8]:
            negative_lines.append(
                f"Avoid {neg['round_id']}: family={neg['research_family']}, template={neg['sample_template']}, "
                f"nav_cagr={_format_float(neg['nav_cagr'])}, top_features={','.join(neg['top_prompt_features'])}, "
                f"factor_rules={_compact_factor_rule_hint(neg.get('factor_analysis_summary', {}), top_features=2, top_combos=0)}, "
                f"weak_note={str(neg.get('rejection_reason') or neg.get('lesson_summary') or '')[:140]}"
            )
        candidate_lines = []
        for rec in candidates:
            feats = rec["features"]
            feat_text = ", ".join(f"{k}={feats[k]}" for k in prompt_features if k in feats)
            candidate_lines.append(
                f"{rec.get('candidate_id')} | symbol={rec['symbol']} | entry={rec['entry_date']} | exit={rec['exit_date']} | {feat_text}"
            )
        holding_lines = [
            f"{h['symbol']} exit={h['exit_date']} days_to_exit={h['days_to_exit']}" for h in holdings
        ] or ["none"]
        system = "".join(
            [
                "You are the QuantApprentice trader apprentice. ",
                "This is a real daily trading decision, not a research summary task. ",
                "Choose exactly 4 candidate IDs that best match the positive teacher ensemble unless fewer than 4 candidates exist. ",
                mode_guidance,
                f"{reasoning_instruction} ",
                "If you need more evidence, you may issue at most one QUERY for a single candidate before the final decision. ",
                "QUERY format: QUERY: C03 | FOCUS: close_to_ma20, volatility_20 | REASON: short phrase. ",
                "After the code-agent returns extra feature evidence, you must make the final SELECT or ABSTAIN decision. ",
                "Do not write next-step research plans, do not discuss future experiments, and do not summarize hypotheses. ",
                "You may ABSTAIN if none are attractive enough. ",
                "Reply with exactly one line in one of these formats: ",
                "SELECT: C01,C04,C09,C11 | REASON: short phrase. ",
                "ABSTAIN | REASON: short phrase. ",
                "QUERY: C03 | FOCUS: close_to_ma20, volatility_20 | REASON: short phrase. ",
                "Never output 0, NONE, or the prompt text. ",
                "Do not explain your reasoning.",
            ]
        )
        user = "\n".join(
            [
                f"Decision date: {decision_date.strftime('%Y-%m-%d')}",
                *teacher_lines,
                *band_lines,
                *(
                    ["Teacher explainability pack (Report v2 branch-oriented):", *(explainability_lines or ["none"])]
                    if explainability_only
                    else (
                        [
                            "Teacher explainability pack (Report v2 branch-oriented):",
                            *(explainability_lines or ["none"]),
                            *kb_lines,
                            "Global Lesson Zone:",
                            *(global_lesson_lines or ["none"]),
                            "Teacher-Scoped Lesson Zones:",
                            *(scope_lesson_lines or ["none"]),
                            "Outer-loop background memory (context only, not a direct rulebook):",
                            *(research_experience_digest.splitlines() if research_experience_digest else ["none"]),
                            "Negative teacher memory:",
                            *(negative_lines or ["none"]),
                            "Practice review area:",
                            *(review_cards_for_prompt or ["none"]),
                        ]
                        if _uses_report_v2_with_lessons_prompt(config)
                        else [
                            *kb_lines,
                            "Global Lesson Zone:",
                            *(global_lesson_lines or ["none"]),
                            "Teacher-Scoped Lesson Zones:",
                            *(scope_lesson_lines or ["none"]),
                            "Outer-loop background memory (context only, not a direct rulebook):",
                            *(research_experience_digest.splitlines() if research_experience_digest else ["none"]),
                            "Negative teacher memory:",
                            *(negative_lines or ["none"]),
                            "Practice review area:",
                            *(review_cards_for_prompt or ["none"]),
                        ]
                    )
                ),
                f"Current holdings: {'; '.join(holding_lines)}",
                "Feature-query policy:",
                "- You may ask one QUERY only if the current candidates do not provide enough evidence for a confident trade.",
                "- Use QUERY to inspect one candidate's missing feature dimensions, then finalize the decision.",
                "Candidates:",
                *candidate_lines,
            ]
        )
        return system, user
    system = (
        "You are the QuantApprentice trader apprentice. "
        "Your task is to imitate a small teacher ensemble's combined next-slot selection behavior. "
        "Use the provided teacher summaries and candidate-level teacher views. "
        "Return strict JSON only. "
        "Do not include markdown. "
        "Do not spend tokens on hidden reasoning. "
        "Answer with one compact JSON object immediately."
    )
    user_payload = {
        "task": "multi_teacher_imitation",
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "prompt_recipe": (
            "explainability_only"
            if explainability_only
            else ("report_v2_with_lessons" if _uses_report_v2_with_lessons_prompt(config) else "standard")
        ),
        "selection_rule": {
            "choose_up_to": config.llm_max_daily_picks,
            "mode": "buy_only_for_tomorrow_slot",
            "existing_holdings_auto_hold_until_exit": True,
            "objective": "match the ensemble's likely top selections for this decision day",
        },
        "teachers": [
            {
                "round_id": meta["round_id"],
                "title": meta["title"],
                "family": meta["research_family"],
                "template": meta["sample_template"],
                "nav_cagr": round(meta["nav_cagr"], 6),
                "nav_max_drawdown": round(meta["nav_max_drawdown"], 6),
                "nav_positive_years": f"{meta['nav_positive_years']}/{meta['nav_total_years']}",
                "top_features": meta["top_prompt_features"],
                "preferred_feature_bands": []
                if (explainability_only or branch_report_context)
                else meta.get("preference_bands", []),
                "factor_rules": {} if (explainability_only or branch_report_context) else meta.get("factor_analysis_summary", {}),
                "teacher_explainability_pack": (
                    _teacher_explainability_payload(meta.get("factor_analysis_summary", {}))
                    if branch_report_context
                    else {}
                ),
            }
            for meta in metas
        ],
        "feature_query_policy": {
            "allowed": True,
            "max_queries": 1,
            "scope": "single candidate per decision day",
            "purpose": "inspect missing feature dimensions before final trade selection",
        },
        "knowledge_base_context": kb_context,
        "trading_memory_no_signal": research_experience_digest,
        "negative_teacher_memory": [] if explainability_only else list(negative_metas or []),
        "global_lesson_zone": list(global_lesson_lines),
        "teacher_scoped_lesson_zones": [] if explainability_only else _teacher_scope_entries(scoped_warmup_state),
        "warmup_review_memory": list(review_cards_for_prompt or []),
        "current_holdings": holdings,
        "candidate_feature_schema": list(prompt_features),
        "candidates": candidates,
        "required_output": {
            "selected_symbols": ["000001", "600519"],
            "brief_reason": "one short sentence",
        },
    }
    return system, json.dumps(user_payload, ensure_ascii=False)


def _parse_selected_symbols(payload: Dict[str, Any], allowed_symbols: Sequence[str], limit: int) -> Tuple[List[str], bool, bool]:
    if payload.get("abstain") is True:
        return [], False, True
    raw = payload.get("selected_symbols", [])
    if not isinstance(raw, list):
        return [], True, False
    allowed = {_normalize_symbol(symbol) for symbol in allowed_symbols}
    selected: List[str] = []
    for value in raw:
        symbol = _normalize_symbol(value)
        if symbol in allowed and symbol not in selected:
            selected.append(symbol)
        if len(selected) >= limit:
            break
    if selected:
        return selected, False, False
    if payload.get("selected_symbols") == []:
        return [], False, True
    return [], True, False


def _parse_model_selection_reply(
    *,
    content: str,
    model: str,
    allowed_symbols: Sequence[str],
    limit: int,
    candidate_id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not content.strip():
        return {"selected_symbols": [], "parse_failed": True, "abstain": False, "parse_mode": "empty", "failure_reason": "empty"}
    if _looks_like_prompt_echo(content):
        return {"selected_symbols": [], "parse_failed": True, "abstain": False, "parse_mode": "prompt_echo", "failure_reason": "prompt_echo"}
    if _looks_like_explicit_abstain(content):
        return {"selected_symbols": [], "parse_failed": False, "abstain": True, "parse_mode": "abstain_text", "failure_reason": ""}
    try:
        payload = _extract_json_payload(content)
        selected, parse_failed, abstain = _parse_selected_symbols(payload, allowed_symbols, limit)
        return {
            "selected_symbols": selected,
            "parse_failed": parse_failed,
            "abstain": abstain,
            "parse_mode": "json" if not parse_failed else "json_invalid",
            "failure_reason": "" if not parse_failed else "json_invalid",
        }
    except Exception:
        pass
    if _prefers_line_id_protocol(model):
        candidate_ids = _extract_candidate_ids_from_text(content, limit)
        if candidate_ids:
            mapped: List[str] = []
            for cid in candidate_ids:
                symbol = (candidate_id_map or {}).get(cid)
                if symbol and symbol not in mapped:
                    mapped.append(symbol)
                if len(mapped) >= limit:
                    break
            if mapped:
                return {"selected_symbols": mapped, "parse_failed": False, "abstain": False, "parse_mode": "candidate_ids", "failure_reason": ""}
        numeric_ids = _extract_numeric_candidate_ids_from_text(content, candidate_id_map=candidate_id_map, limit=limit)
        if numeric_ids:
            mapped = []
            for cid in numeric_ids:
                symbol = (candidate_id_map or {}).get(cid)
                if symbol and symbol not in mapped:
                    mapped.append(symbol)
                if len(mapped) >= limit:
                    break
            if mapped:
                return {"selected_symbols": mapped, "parse_failed": False, "abstain": False, "parse_mode": "numeric_candidate_ids", "failure_reason": ""}
    selected = _extract_symbols_from_text(content, allowed_symbols, limit)
    if selected:
        return {"selected_symbols": selected, "parse_failed": False, "abstain": False, "parse_mode": "symbols", "failure_reason": ""}
    return {
        "selected_symbols": [],
        "parse_failed": True,
        "abstain": False,
        "parse_mode": "failure",
        "failure_reason": _classify_parse_failure_text(content),
    }


def _build_trade_calendar(start_entry: pd.Timestamp, end_exit: pd.Timestamp) -> Tuple[pd.DatetimeIndex, pd.Series]:
    hs300_close = load_hs300()
    trade_dates = hs300_close.index[(hs300_close.index >= start_entry) & (hs300_close.index <= end_exit)]
    hs300_nav = (hs300_close.loc[trade_dates] / hs300_close.loc[trade_dates].iloc[0]).copy()
    return trade_dates, hs300_nav


def _nav_from_selected(
    selected_df: pd.DataFrame,
    *,
    trade_dates: pd.DatetimeIndex,
    lock_days: int,
) -> pd.Series:
    if selected_df.empty:
        return pd.Series(1.0, index=trade_dates)
    return compute_nav_curve_fast(selected_df.copy(), "Q5", trade_dates, pre_filtered=selected_df.copy(), lock_days=lock_days)


def _agreement_stats(
    llm_decisions: Dict[pd.Timestamp, List[str]],
    teacher_targets: Dict[pd.Timestamp, List[str]],
) -> pd.DataFrame:
    rows = []
    all_dates = sorted(set(llm_decisions) | set(teacher_targets))
    for dt in all_dates:
        llm_set = set(llm_decisions.get(dt, []))
        tgt_set = set(teacher_targets.get(dt, []))
        inter = len(llm_set & tgt_set)
        union = len(llm_set | tgt_set)
        precision = inter / len(llm_set) if llm_set else 0.0
        recall = inter / len(tgt_set) if tgt_set else 0.0
        jaccard = inter / union if union else 1.0
        exact = int(llm_set == tgt_set)
        rows.append(
            {
                "decision_date": dt.strftime("%Y-%m-%d"),
                "llm_count": len(llm_set),
                "teacher_target_count": len(tgt_set),
                "intersection_count": inter,
                "precision": precision,
                "recall": recall,
                "jaccard": jaccard,
                "exact_match": exact,
            }
        )
    return pd.DataFrame(rows)


def _serialize_daily_response(path: Path, *, request_payload: Dict[str, Any], response_payload: Dict[str, Any]) -> None:
    _write_json(path, {"request": request_payload, "response": response_payload})


def _request_chars(payload: Dict[str, Any]) -> int:
    messages = payload.get("messages", [])
    total = 0
    for msg in messages:
        total += len(str(msg.get("content", "")))
    return total


def _api_usage_stats(api_cache_dir: Path) -> Dict[str, Any]:
    prompt_tokens: List[int] = []
    total_tokens: List[int] = []
    request_chars: List[int] = []
    for path in sorted(api_cache_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        for request_key, response_key in [
            ("request", "response"),
            ("query_request", "query_response"),
            ("retry_request", "retry_response"),
        ]:
            request_payload = payload.get(request_key)
            response_payload = payload.get(response_key)
            if not isinstance(request_payload, dict) or not isinstance(response_payload, dict):
                continue
            usage = response_payload.get("usage") or {}
            prompt = usage.get("prompt_tokens")
            total = usage.get("total_tokens")
            if isinstance(prompt, (int, float)):
                prompt_tokens.append(int(prompt))
            if isinstance(total, (int, float)):
                total_tokens.append(int(total))
            request_chars.append(_request_chars(request_payload))
    return {
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else 0,
        "prompt_tokens_mean": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
        "total_tokens_max": max(total_tokens) if total_tokens else 0,
        "total_tokens_mean": float(np.mean(total_tokens)) if total_tokens else 0.0,
        "max_request_chars": max(request_chars) if request_chars else 0,
    }


def _retry_prompt_single(
    *,
    decision_date: pd.Timestamp,
    meta: Dict[str, Any],
    candidate_df: pd.DataFrame,
) -> Tuple[str, str]:
    prompt_features = meta["top_prompt_features"][: max(3, min(4, len(meta["top_prompt_features"])))]
    candidate_lines = []
    for rec in _candidate_rows_to_records(candidate_df, prompt_features, teacher_fields=[]):
        feats = rec["features"]
        feat_text = ", ".join(f"{k}={feats[k]}" for k in prompt_features if k in feats)
        candidate_lines.append(f"{rec.get('candidate_id')} {feat_text}")
    system = (
        "Return either ABSTAIN or exactly 4 candidate IDs separated by commas only. "
        "No extra words. Examples: C01,C04,C09,C11 or ABSTAIN or 1,4,9,11"
    )
    user = "\n".join(
        [
            f"Date: {decision_date.strftime('%Y-%m-%d')}",
            f"Style: {_template_style_hint(meta['sample_template'])}",
            "Candidates:",
            *candidate_lines,
        ]
    )
    return system, user


def _retry_prompt_multi(
    *,
    decision_date: pd.Timestamp,
    metas: List[Dict[str, Any]],
    prompt_features: Sequence[str],
    candidate_df: pd.DataFrame,
) -> Tuple[str, str]:
    use_features = list(prompt_features[: max(3, min(4, len(prompt_features)))])
    style_line = " + ".join(_template_style_hint(meta["sample_template"]) for meta in metas)
    candidate_lines = []
    for rec in _candidate_rows_to_records(candidate_df, use_features, teacher_fields=[]):
        feats = rec["features"]
        feat_text = ", ".join(f"{k}={feats[k]}" for k in use_features if k in feats)
        candidate_lines.append(f"{rec.get('candidate_id')} {feat_text}")
    system = (
        "Return either ABSTAIN or exactly 4 candidate IDs separated by commas only. "
        "No extra words. Examples: C01,C04,C09,C11 or ABSTAIN or 1,4,9,11"
    )
    user = "\n".join(
        [
            f"Date: {decision_date.strftime('%Y-%m-%d')}",
            f"Styles: {style_line}",
            "Candidates:",
            *candidate_lines,
        ]
    )
    return system, user


def _query_prompt_followup(
    *,
    decision_date: pd.Timestamp,
    original_context: str,
    query_action_text: str,
    query_context: str,
) -> Tuple[str, str]:
    system = (
        "You are the QuantApprentice trader apprentice. "
        "A code-agent has returned extra feature evidence for one queried candidate. "
        "Use it once, then make the final trade decision. "
        "Do not ask another QUERY. "
        "Reply with exactly one line in one of these formats: "
        "SELECT: C01,C04,C09,C11 | REASON: short phrase. "
        "ABSTAIN | REASON: short phrase."
    )
    user = "\n".join(
        [
            f"Decision date: {decision_date.strftime('%Y-%m-%d')}",
            f"Previous action: {query_action_text}",
            "Feature query result:",
            query_context,
            "Original decision context:",
            original_context,
        ]
    )
    return system, user


def _request_one_day(
    *,
    day_candidates: pd.DataFrame,
    decision_date: pd.Timestamp,
    config: ApprenticeReplayConfig,
    prompt_builder,
    prompt_builder_kwargs: Dict[str, Any],
    api_key: str,
    api_cache_dir: Path,
    empty_rows: pd.DataFrame,
    reuse_api_cache_override: Optional[bool] = None,
) -> Dict[str, Any]:
    day_candidates = day_candidates.copy().reset_index(drop=True)
    day_candidates["_candidate_id"] = [f"C{i:02d}" for i in range(1, len(day_candidates) + 1)]
    candidate_id_map = dict(zip(day_candidates["_candidate_id"], day_candidates["symbol"]))
    selected_rows = empty_rows if config.ignore_holdings_context else empty_rows
    system, user_content = prompt_builder(
        config=config,
        decision_date=decision_date,
        candidate_df=day_candidates,
        selected_rows=selected_rows,
        **prompt_builder_kwargs,
    )
    request_payload = {
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": config.api_max_tokens,
        "temperature": config.api_temperature,
    }
    initial_prefix = _initial_line_answer_prefix(config)
    if initial_prefix is not None:
        request_payload["messages"].append(initial_prefix)
    cache_path = api_cache_dir / f"{decision_date.strftime('%Y%m%d')}.json"
    api_calls = 0
    api_cache_hits = 0
    retry_invoked = False
    retry_success = False
    query_invoked = False
    query_success = False
    reuse_api_cache = config.reuse_api_cache if reuse_api_cache_override is None else bool(reuse_api_cache_override)
    cached: Optional[Dict[str, Any]] = None
    cache_payload: Dict[str, Any]
    response_payload: Optional[Dict[str, Any]] = None
    if reuse_api_cache and cache_path.exists():
        cached = _load_json(cache_path)
        cache_payload = dict(cached)
        response_payload = cached.get("response")
        if response_payload is not None:
            api_cache_hits = 1
    else:
        cache_payload = {"request": request_payload}
        try:
            response_payload = _chat_completion(
                messages=request_payload["messages"],
                api_key=api_key,
                model=config.api_model,
                max_tokens=config.api_max_tokens,
                temperature=config.api_temperature,
                force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=not config.inline_day_retry_enabled,
                max_retries=config.api_request_max_retries,
            )
            cache_payload["response"] = response_payload
            _serialize_daily_response(cache_path, request_payload=request_payload, response_payload=response_payload)
            api_calls = 1
        except Exception as exc:
            cache_payload["request_error"] = str(exc)
    content = ""
    action_result: Dict[str, Any] = {
        "action": "failure",
        "query_candidate_id": "",
        "query_symbol": "",
        "focus_features": [],
    }
    if response_payload is not None:
        content = response_payload["choices"][0]["message"].get("content", "") or ""
        action_result = _parse_model_action_reply(
            content=content,
            model=config.api_model,
            allowed_symbols=day_candidates["symbol"].tolist(),
            limit=config.llm_max_daily_picks,
            candidate_id_map=candidate_id_map,
        )
    selected_symbols: List[str] = []
    parse_fallback = True
    abstain = False
    parse_mode = "failure"
    failure_reason = ""
    if response_payload is None:
        parse_mode = "api_error"
        failure_reason = str(cache_payload.get("request_error", "") or "primary_api_error")
    if response_payload is not None and action_result.get("action") == "query" and _prefers_line_id_protocol(config.api_model):
        query_invoked = True
        query_candidate_id = str(action_result.get("query_candidate_id", "")).strip().upper()
        query_symbol = str(action_result.get("query_symbol", "")).strip()
        if not query_candidate_id and query_symbol:
            inverse_map = {str(v): str(k) for k, v in candidate_id_map.items()}
            query_candidate_id = inverse_map.get(_normalize_symbol(query_symbol), "")
        focus_features = list(action_result.get("focus_features") or [])
        query_action_text = f"QUERY: {query_candidate_id or query_symbol or 'unknown'} | FOCUS: {', '.join(focus_features) or 'none'}"
        query_context = _build_feature_query_context(
            day_candidates=day_candidates,
            query_candidate_id=query_candidate_id or (candidate_id_map and next((cid for cid, sym in candidate_id_map.items() if sym == query_symbol), "")) or "",
            focus_features=focus_features,
        )
        if cached and "query_response" in cached:
            query_response = cached["query_response"]
            api_cache_hits += 1
        else:
            query_system, query_user = _query_prompt_followup(
                decision_date=decision_date,
                original_context=user_content,
                query_action_text=query_action_text,
                query_context=query_context,
            )
            query_request = {
                "model": config.api_model,
                "messages": [
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": query_user},
                    {"role": "assistant", "content": "SELECT: "},
                ],
                "max_tokens": config.api_max_tokens,
                "temperature": config.api_temperature,
            }
            try:
                query_response = _chat_completion(
                    messages=query_request["messages"],
                    api_key=api_key,
                    model=config.api_model,
                    max_tokens=config.api_max_tokens,
                    temperature=config.api_temperature,
                    force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                    fail_fast_on_empty_content=not config.inline_day_retry_enabled,
                    max_retries=config.api_request_max_retries,
                )
                api_calls += 1
                cache_payload.update({"query_request": query_request, "query_response": query_response})
                _write_json(cache_path, cache_payload)
            except Exception as exc:
                cache_payload.update({"query_request": query_request, "query_error": str(exc)})
                _write_json(cache_path, cache_payload)
                query_response = None
                failure_reason = (failure_reason + " | " if failure_reason else "") + f"query_api_error={exc}"
        if query_response is not None:
            query_content = query_response["choices"][0]["message"].get("content", "") or ""
            content = f"{content}\n[query] {query_content}"
            query_parse_result = _parse_model_selection_reply(
                content=query_content,
                model=config.api_model,
                allowed_symbols=day_candidates["symbol"].tolist(),
                limit=config.llm_max_daily_picks,
                candidate_id_map=candidate_id_map,
            )
            if (query_parse_result["selected_symbols"] or query_parse_result["abstain"]) and not query_parse_result["parse_failed"]:
                query_success = True
                selected_symbols = list(query_parse_result["selected_symbols"])
                parse_fallback = bool(query_parse_result["parse_failed"])
                abstain = bool(query_parse_result["abstain"])
                parse_mode = str(query_parse_result["parse_mode"])
                failure_reason = str(query_parse_result.get("failure_reason", ""))
            else:
                selected_symbols = list(query_parse_result["selected_symbols"])
                parse_fallback = bool(query_parse_result["parse_failed"])
                abstain = bool(query_parse_result["abstain"])
                parse_mode = str(query_parse_result["parse_mode"])
                failure_reason = str(query_parse_result.get("failure_reason", ""))
        else:
            parse_fallback = True
            abstain = False
            parse_mode = "query_api_error"
    else:
        parse_result = _parse_model_selection_reply(
            content=content,
            model=config.api_model,
            allowed_symbols=day_candidates["symbol"].tolist(),
            limit=config.llm_max_daily_picks,
            candidate_id_map=candidate_id_map,
        )
        selected_symbols = list(parse_result["selected_symbols"])
        parse_fallback = bool(parse_result["parse_failed"])
        abstain = bool(parse_result["abstain"])
        parse_mode = str(parse_result["parse_mode"])
        failure_reason = str(parse_result.get("failure_reason", ""))
    if config.inline_day_retry_enabled and parse_fallback and _prefers_line_id_protocol(config.api_model):
        retry_invoked = True
        if cached and "retry_response" in cached:
            retry_response = cached["retry_response"]
            api_cache_hits += 1
        else:
            if config.mode == "single":
                retry_system, retry_user = _retry_prompt_single(
                    decision_date=decision_date,
                    meta=prompt_builder_kwargs["meta"],
                    candidate_df=day_candidates,
                )
            else:
                retry_system, retry_user = _retry_prompt_multi(
                    decision_date=decision_date,
                    metas=prompt_builder_kwargs["metas"],
                    prompt_features=prompt_builder_kwargs["prompt_features"],
                    candidate_df=day_candidates,
                )
            retry_request = {
                "model": config.api_model,
                "messages": [
                    {"role": "system", "content": retry_system},
                    {"role": "user", "content": retry_user},
                    {"role": "assistant", "content": "C01,C04,C09,C11"},
                ],
                "max_tokens": _retry_completion_budget(config),
                "temperature": config.api_temperature,
            }
            try:
                retry_response = _chat_completion(
                    messages=retry_request["messages"],
                    api_key=api_key,
                    model=config.api_model,
                    max_tokens=retry_request["max_tokens"],
                    temperature=config.api_temperature,
                    force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                    fail_fast_on_empty_content=not config.inline_day_retry_enabled,
                    max_retries=config.api_request_max_retries,
                )
                api_calls += 1
                cache_payload.update({"retry_request": retry_request, "retry_response": retry_response})
                _write_json(cache_path, cache_payload)
            except Exception as exc:
                cache_payload.update({"retry_request": retry_request, "retry_error": str(exc)})
                _write_json(cache_path, cache_payload)
                retry_response = None
                failure_reason = (failure_reason + " | " if failure_reason else "") + f"retry_api_error={exc}"
        if retry_response is not None:
            retry_content = retry_response["choices"][0]["message"].get("content", "") or ""
            retry_parse_result = _parse_model_selection_reply(
                content=retry_content,
                model=config.api_model,
                allowed_symbols=day_candidates["symbol"].tolist(),
                limit=config.llm_max_daily_picks,
                candidate_id_map=candidate_id_map,
            )
            content = f"{content}\n[retry] {retry_content}"
            if (retry_parse_result["selected_symbols"] or retry_parse_result["abstain"]) and not retry_parse_result["parse_failed"]:
                retry_success = True
                selected_symbols = list(retry_parse_result["selected_symbols"])
                parse_fallback = bool(retry_parse_result["parse_failed"])
                abstain = bool(retry_parse_result["abstain"])
                parse_mode = str(retry_parse_result["parse_mode"])
                failure_reason = str(retry_parse_result.get("failure_reason", ""))
        else:
            parse_fallback = True
            abstain = False
            parse_mode = "retry_api_error"
    return {
        "decision_date": decision_date,
        "day_candidates": day_candidates,
        "selected_symbols": selected_symbols,
        "parse_fallback": parse_fallback,
        "abstain": abstain,
        "parse_mode": parse_mode,
        "failure_reason": failure_reason,
        "query_invoked": query_invoked,
        "query_success": query_success,
        "retry_invoked": retry_invoked,
        "retry_success": retry_success,
        "content": content,
        "api_calls": api_calls,
        "api_cache_hits": api_cache_hits,
    }


def _parallel_result_needs_rerun(result: Mapping[str, Any]) -> bool:
    return bool(result.get("parse_fallback", False)) and not bool(result.get("abstain", False))


def _request_day_batch_parallel(
    *,
    day_batch: Sequence[Tuple[pd.Timestamp, pd.DataFrame]],
    config: ApprenticeReplayConfig,
    prompt_builder,
    prompt_builder_kwargs: Dict[str, Any],
    api_key: str,
    api_cache_dir: Path,
    empty_rows: pd.DataFrame,
    workers: int,
    reuse_api_cache_override: Optional[bool],
    parallel_pass_index: int,
    batch_index: int,
) -> List[Dict[str, Any]]:
    if not day_batch:
        return []
    future_map = {}
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for decision_date, day_candidates in day_batch:
            future = executor.submit(
                _request_one_day,
                day_candidates=day_candidates,
                decision_date=decision_date,
                config=config,
                prompt_builder=prompt_builder,
                prompt_builder_kwargs=prompt_builder_kwargs,
                api_key=api_key,
                api_cache_dir=api_cache_dir,
                empty_rows=empty_rows,
                reuse_api_cache_override=reuse_api_cache_override,
            )
            future_map[future] = pd.Timestamp(decision_date)
        for future in as_completed(future_map):
            result = future.result()
            result["parallel_pass_index"] = int(parallel_pass_index)
            result["parallel_batch_index"] = int(batch_index)
            results.append(result)
    results.sort(key=lambda item: item["decision_date"])
    return results


def _run_parallel_wave_replay(
    *,
    day_groups: Sequence[Tuple[pd.Timestamp, pd.DataFrame]],
    teacher_target_by_date: Mapping[pd.Timestamp, List[str]],
    config: ApprenticeReplayConfig,
    prompt_builder,
    prompt_builder_kwargs: Dict[str, Any],
    api_key: str,
    api_cache_dir: Path,
    candidate_pool_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, List[str]], List[Dict[str, Any]], Dict[str, int]]:
    empty_rows = pd.DataFrame(columns=list(candidate_pool_df.columns))
    selected_parts: List[pd.DataFrame] = []
    llm_decisions: Dict[pd.Timestamp, List[str]] = {}
    teacher_decisions: Dict[pd.Timestamp, List[str]] = {}
    prompt_log_rows: List[Dict[str, Any]] = []
    stats = {
        "api_calls": 0,
        "api_cache_hits": 0,
        "parse_fallback_days": 0,
        "parse_failure_days": 0,
        "query_invoked_days": 0,
        "query_success_days": 0,
        "abstain_days": 0,
        "retry_invoked_days": 0,
        "retry_success_days": 0,
    }

    pending = [(pd.Timestamp(decision_date), day_candidates.copy().reset_index(drop=True)) for decision_date, day_candidates in day_groups]
    final_results_by_date: Dict[pd.Timestamp, Dict[str, Any]] = {}
    total_passes = max(1, 1 + int(max(config.api_failed_rerun_rounds, 0)))

    for pass_index in range(total_passes):
        if not pending:
            break
        pass_workers = int(config.api_parallel_workers) if pass_index == 0 else int(
            config.api_failed_rerun_workers or max(1, config.api_parallel_workers // 2)
        )
        pass_workers = max(1, pass_workers)
        pass_reuse_cache = config.reuse_api_cache if pass_index == 0 else False
        _progress(
            "parallel wave pass start "
            f"run_id={config.run_id()} pass={pass_index + 1}/{total_passes} "
            f"pending_days={len(pending)} workers={pass_workers} reuse_cache={int(pass_reuse_cache)}"
        )

        next_pending: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
        pass_results: List[Dict[str, Any]] = []
        for batch_index, start_idx in enumerate(range(0, len(pending), pass_workers), start=1):
            day_batch = pending[start_idx : start_idx + pass_workers]
            batch_results = _request_day_batch_parallel(
                day_batch=day_batch,
                config=config,
                prompt_builder=prompt_builder,
                prompt_builder_kwargs=prompt_builder_kwargs,
                api_key=api_key,
                api_cache_dir=api_cache_dir,
                empty_rows=empty_rows,
                workers=pass_workers,
                reuse_api_cache_override=pass_reuse_cache,
                parallel_pass_index=pass_index,
                batch_index=batch_index,
            )
            pass_results.extend(batch_results)
            batch_failures = sum(1 for item in batch_results if _parallel_result_needs_rerun(item))
            _progress(
                "parallel wave batch done "
                f"run_id={config.run_id()} pass={pass_index + 1}/{total_passes} "
                f"batch={batch_index} size={len(day_batch)} failures={batch_failures}"
            )
            if pass_index + 1 < total_passes:
                for item in batch_results:
                    if _parallel_result_needs_rerun(item):
                        next_pending.append((pd.Timestamp(item["decision_date"]), item["day_candidates"].copy().reset_index(drop=True)))

        _progress(
            "parallel wave pass done "
            f"run_id={config.run_id()} pass={pass_index + 1}/{total_passes} "
            f"completed_days={len(pass_results)} rerun_next={len(next_pending)}"
        )
        for item in pass_results:
            final_results_by_date[pd.Timestamp(item["decision_date"])] = item
        pending = next_pending

    ordered_results = [final_results_by_date[key] for key in sorted(final_results_by_date)]
    for result in ordered_results:
        decision_date = pd.Timestamp(result["decision_date"])
        day_candidates = result["day_candidates"]
        selected_symbols = result["selected_symbols"]
        parse_fallback = bool(result["parse_fallback"])
        abstain = bool(result.get("abstain", False))
        parse_mode = str(result.get("parse_mode", ""))
        failure_reason = str(result.get("failure_reason", ""))
        query_invoked = bool(result.get("query_invoked", False))
        query_success = bool(result.get("query_success", False))
        retry_invoked = bool(result.get("retry_invoked", False))
        retry_success = bool(result.get("retry_success", False))
        content = str(result["content"])
        stats["api_calls"] += int(result["api_calls"])
        stats["api_cache_hits"] += int(result["api_cache_hits"])
        if query_invoked:
            stats["query_invoked_days"] += 1
        if query_success:
            stats["query_success_days"] += 1
        if retry_invoked:
            stats["retry_invoked_days"] += 1
        if retry_success:
            stats["retry_success_days"] += 1
        if abstain:
            stats["abstain_days"] += 1
        if parse_fallback:
            stats["parse_fallback_days"] += 1
            stats["parse_failure_days"] += 1

        llm_decisions[decision_date] = selected_symbols
        teacher_target_symbols = teacher_target_by_date.get(decision_date, [])
        teacher_decisions[decision_date] = teacher_target_symbols
        day_selected = _select_day_rows_by_symbols(day_candidates, selected_symbols)
        if not day_selected.empty:
            selected_parts.append(day_selected)

        prompt_log_rows.append(
            {
                "decision_date": decision_date.strftime("%Y-%m-%d"),
                "candidate_count": len(day_candidates),
                "teacher_target_count": len(teacher_target_symbols),
                "llm_selected_count": len(selected_symbols),
                "abstain": int(abstain),
                "parse_fallback": int(parse_fallback),
                "feature_query_invoked": int(query_invoked),
                "feature_query_success": int(query_success),
                "retry_invoked": int(retry_invoked),
                "retry_success": int(retry_success),
                "parse_mode": parse_mode,
                "failure_reason": failure_reason,
                "parallel_pass_index": int(result.get("parallel_pass_index", 0)),
                "parallel_batch_index": int(result.get("parallel_batch_index", 0)),
                "selected_symbols": ",".join(selected_symbols),
                "teacher_target_symbols": ",".join(teacher_target_symbols),
                "brief_reason": content[:240],
            }
        )

    selected_rows = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=list(candidate_pool_df.columns))
    return selected_rows, llm_decisions, teacher_decisions, prompt_log_rows, stats


def _finalize_replay_outputs(
    *,
    config: ApprenticeReplayConfig,
    report_dir: Path,
    api_cache_dir: Path,
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
    teacher_full_df: pd.DataFrame,
    llm_selected_df: pd.DataFrame,
    llm_decisions: Dict[pd.Timestamp, List[str]],
    teacher_decisions: Dict[pd.Timestamp, List[str]],
    prompt_log_rows: List[Dict[str, Any]],
    api_calls: int,
    api_cache_hits: int,
    parse_fallback_days: int,
    parse_failure_days: int,
    query_invoked_days: int,
    query_success_days: int,
    abstain_days: int,
    retry_invoked_days: int,
    retry_success_days: int,
) -> ReplaySummary:
    llm_selected_df = llm_selected_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    teacher_target_df = teacher_target_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    teacher_full_df = teacher_full_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)

    selected_key_cols = [col for col in ["symbol", "signal_date", "entry_date", "exit_date"] if col in candidate_pool_df.columns]
    if selected_key_cols:
        llm_selected_keys = llm_selected_df[selected_key_cols].drop_duplicates() if not llm_selected_df.empty else pd.DataFrame(columns=selected_key_cols)
        llm_not_selected_df = candidate_pool_df.merge(
            llm_selected_keys.assign(_selected_flag=1),
            on=selected_key_cols,
            how="left",
        )
        llm_not_selected_df = llm_not_selected_df[llm_not_selected_df["_selected_flag"].isna()].drop(columns=["_selected_flag"])
    else:
        llm_not_selected_df = candidate_pool_df.iloc[0:0].copy()

    llm_selected_mean_return = float(llm_selected_df["future_return_5d"].mean()) if not llm_selected_df.empty and "future_return_5d" in llm_selected_df.columns else 0.0
    llm_not_selected_mean_return = float(llm_not_selected_df["future_return_5d"].mean()) if not llm_not_selected_df.empty and "future_return_5d" in llm_not_selected_df.columns else 0.0
    teacher_selected_mean_return = float(teacher_target_df["future_return_5d"].mean()) if not teacher_target_df.empty and "future_return_5d" in teacher_target_df.columns else 0.0
    uplift_vs_not_selected = float(llm_selected_mean_return - llm_not_selected_mean_return)
    uplift_vs_teacher_selected = float(llm_selected_mean_return - teacher_selected_mean_return)

    start_entry = min(pd.to_datetime(candidate_pool_df["entry_date"]).min(), pd.to_datetime(teacher_full_df["entry_date"]).min())
    end_exit = max(pd.to_datetime(candidate_pool_df["exit_date"]).max(), pd.to_datetime(teacher_full_df["exit_date"]).max())
    trade_dates, hs300_nav = _build_trade_calendar(start_entry, end_exit)

    llm_nav = _nav_from_selected(llm_selected_df, trade_dates=trade_dates, lock_days=config.lock_days)
    teacher_target_nav = _nav_from_selected(teacher_target_df, trade_dates=trade_dates, lock_days=config.lock_days)
    if "quintile" in teacher_full_df.columns:
        teacher_full_nav = compute_nav_curve_fast(teacher_full_df.copy(), "Q5", trade_dates, lock_days=config.lock_days)
    else:
        teacher_full_nav = _nav_from_selected(teacher_full_df, trade_dates=trade_dates, lock_days=config.lock_days)

    llm_metrics = _compute_nav_metrics(llm_nav)
    teacher_target_metrics = _compute_nav_metrics(teacher_target_nav)
    teacher_full_metrics = _compute_nav_metrics(teacher_full_nav)
    hs300_metrics = _compute_nav_metrics(hs300_nav)

    agreement_df = _agreement_stats(llm_decisions, teacher_decisions)

    nav_curve_df = pd.DataFrame(
        {
            "date": trade_dates,
            "llm_nav": llm_nav.reindex(trade_dates, fill_value=1.0).values,
            "teacher_target_nav": teacher_target_nav.reindex(trade_dates, fill_value=1.0).values,
            "teacher_full_nav": teacher_full_nav.reindex(trade_dates, fill_value=1.0).values,
            "hs300_nav": hs300_nav.reindex(trade_dates, fill_value=1.0).values,
        }
    )

    yearly_rows = []
    for label, metrics in [
        ("llm", llm_metrics),
        ("teacher_target", teacher_target_metrics),
        ("teacher_full", teacher_full_metrics),
        ("hs300", hs300_metrics),
    ]:
        for year, ret in metrics["yearly_returns"].items():
            yearly_rows.append({"curve": label, "year": int(year), "annual_return": float(ret)})
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["curve", "year"]).reset_index(drop=True)

    title = (
        f"{config.mode.upper()} Apprentice Replay\n"
        f"{', '.join(config.teacher_round_ids)} | {config.start_date} -> {config.end_date}"
    )
    overlay_path = report_dir / "nav_overlay.png"
    _plot_overlay(
        output_path=overlay_path,
        title=title,
        llm_nav=llm_nav,
        teacher_target_nav=teacher_target_nav,
        teacher_full_nav=teacher_full_nav,
        hs300_nav=hs300_nav,
    )

    nav_curve_path = report_dir / "nav_curves.csv"
    yearly_returns_path = report_dir / "yearly_returns.csv"
    agreement_path = report_dir / "daily_agreement.csv"
    prompt_log_path = report_dir / "daily_prompt_log.csv"
    llm_selected_path = report_dir / "llm_selected_signals.csv"
    teacher_target_path = report_dir / "teacher_target_signals.csv"
    teacher_full_path = report_dir / "teacher_full_signals.csv"
    candidate_pool_path = report_dir / "candidate_pool_signals.csv"

    nav_curve_df.to_csv(nav_curve_path, index=False)
    yearly_df.to_csv(yearly_returns_path, index=False)
    agreement_df.to_csv(agreement_path, index=False)
    prompt_log_df = pd.DataFrame(prompt_log_rows)
    if not prompt_log_df.empty and "decision_date" in prompt_log_df.columns:
        sort_cols = ["decision_date"] + [col for col in ["parallel_batch_index"] if col in prompt_log_df.columns]
        prompt_log_df = prompt_log_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    prompt_log_df.to_csv(prompt_log_path, index=False)
    candidate_pool_df.to_csv(candidate_pool_path, index=False)
    llm_selected_df.to_csv(llm_selected_path, index=False)
    teacher_target_df.to_csv(teacher_target_path, index=False)
    teacher_full_df.to_csv(teacher_full_path, index=False)

    mean_daily_jaccard = float(agreement_df["jaccard"].mean()) if not agreement_df.empty else 0.0
    mean_daily_precision = float(agreement_df["precision"].mean()) if not agreement_df.empty else 0.0
    mean_daily_recall = float(agreement_df["recall"].mean()) if not agreement_df.empty else 0.0
    exact_match_days = int(agreement_df["exact_match"].sum()) if not agreement_df.empty else 0
    exact_match_rate = float(agreement_df["exact_match"].mean()) if not agreement_df.empty else 0.0
    api_usage = _api_usage_stats(api_cache_dir)

    summary = ReplaySummary(
        run_id=config.run_id(),
        mode=config.mode,
        start_date=config.start_date,
        end_date=config.end_date,
        api_model=config.api_model,
        summary_variant=config.summary_variant,
        teacher_round_ids=list(config.teacher_round_ids),
        negative_teacher_round_ids=list(config.negative_teacher_round_ids),
        decision_days=int(candidate_pool_df["signal_date"].nunique()),
        api_calls=api_calls,
        api_cache_hits=api_cache_hits,
        parse_fallback_days=parse_fallback_days,
        parse_failure_days=parse_failure_days,
        query_invoked_days=query_invoked_days,
        query_success_days=query_success_days,
        abstain_days=abstain_days,
        retry_invoked_days=retry_invoked_days,
        retry_success_days=retry_success_days,
        prompt_tokens_max=int(api_usage["prompt_tokens_max"]),
        prompt_tokens_mean=float(api_usage["prompt_tokens_mean"]),
        total_tokens_max=int(api_usage["total_tokens_max"]),
        total_tokens_mean=float(api_usage["total_tokens_mean"]),
        max_request_chars=int(api_usage["max_request_chars"]),
        llm_selected_rows=int(len(llm_selected_df)),
        teacher_target_rows=int(len(teacher_target_df)),
        teacher_full_rows=int(len(teacher_full_df)),
        mean_daily_jaccard=mean_daily_jaccard,
        mean_daily_precision=mean_daily_precision,
        mean_daily_recall=mean_daily_recall,
        exact_match_days=exact_match_days,
        exact_match_rate=exact_match_rate,
        llm_selected_mean_return=llm_selected_mean_return,
        llm_not_selected_mean_return=llm_not_selected_mean_return,
        teacher_selected_mean_return=teacher_selected_mean_return,
        uplift_vs_not_selected=uplift_vs_not_selected,
        uplift_vs_teacher_selected=uplift_vs_teacher_selected,
        llm_final_nav=float(llm_metrics["final_nav"]),
        llm_total_return=float(llm_metrics["total_return"]),
        llm_cagr=float(llm_metrics["cagr"]),
        llm_max_drawdown=float(llm_metrics["max_drawdown"]),
        teacher_target_final_nav=float(teacher_target_metrics["final_nav"]),
        teacher_target_total_return=float(teacher_target_metrics["total_return"]),
        teacher_target_cagr=float(teacher_target_metrics["cagr"]),
        teacher_target_max_drawdown=float(teacher_target_metrics["max_drawdown"]),
        teacher_full_final_nav=float(teacher_full_metrics["final_nav"]),
        teacher_full_total_return=float(teacher_full_metrics["total_return"]),
        teacher_full_cagr=float(teacher_full_metrics["cagr"]),
        teacher_full_max_drawdown=float(teacher_full_metrics["max_drawdown"]),
        hs300_final_nav=float(hs300_metrics["final_nav"]),
        hs300_total_return=float(hs300_metrics["total_return"]),
        llm_vs_target_tracking_gap=float(llm_metrics["final_nav"] - teacher_target_metrics["final_nav"]),
        llm_vs_full_tracking_gap=float(llm_metrics["final_nav"] - teacher_full_metrics["final_nav"]),
        nav_overlay_path=_relative(overlay_path),
        nav_curve_csv_path=_relative(nav_curve_path),
        yearly_returns_csv_path=_relative(yearly_returns_path),
        agreement_csv_path=_relative(agreement_path),
    )

    summary_path = report_dir / "summary.json"
    summary_md_path = report_dir / "SUMMARY.md"
    _write_json(summary_path, asdict(summary))

    lines = [
        f"# Apprentice Replay Summary - {summary.run_id}",
        "",
        "Replay assumptions:",
        "",
        "- Teachers are already trained on full-history teacher-loop artifacts.",
        "- Replay uses the configured date range below; this run does not retrain teachers inside the replay window.",
        f"- teacher_signal_visible: `{config.include_teacher_signal}`",
        f"- candidate_source: `{config.candidate_source}`",
        "- Fixed 5-day holding horizon, no early exit, no leverage, no transaction cost.",
        "",
        "## Configuration",
        "",
        f"- mode: `{config.mode}`",
        f"- teachers: `{', '.join(config.teacher_round_ids)}`",
        f"- date_range: `{config.start_date}` -> `{config.end_date}`",
        f"- candidate_pool_size: `{config.candidate_pool_size}`",
        f"- teacher_daily_pick_count: `{config.teacher_daily_pick_count}`",
        f"- llm_max_daily_picks: `{config.llm_max_daily_picks}`",
        f"- teacher_signal_visible: `{config.include_teacher_signal}`",
        f"- candidate_source: `{config.candidate_source}`",
        f"- summary_variant: `{config.summary_variant}`",
        f"- api_model: `{config.api_model}`",
        f"- warmup_sample_count: `{config.warmup_sample_count}`",
        f"- warmup_curriculum: `{config.warmup_curriculum}`",
        f"- warmup_batch_size: `{config.warmup_batch_size}`",
        f"- warmup_review_memory_limit: `{config.warmup_review_memory_limit}`",
        f"- warmup_lesson_zone_max_lines: `{config.warmup_lesson_zone_max_lines}`",
        "",
        "## Imitation Metrics",
        "",
        f"- decision_days: `{summary.decision_days}`",
        f"- api_calls: `{summary.api_calls}`",
        f"- api_cache_hits: `{summary.api_cache_hits}`",
        f"- parse_fallback_days: `{summary.parse_fallback_days}`",
        f"- parse_failure_days: `{summary.parse_failure_days}`",
        f"- query_invoked_days: `{summary.query_invoked_days}`",
        f"- query_success_days: `{summary.query_success_days}`",
        f"- abstain_days: `{summary.abstain_days}`",
        f"- retry_invoked_days: `{summary.retry_invoked_days}`",
        f"- retry_success_days: `{summary.retry_success_days}`",
        f"- prompt_tokens_max: `{summary.prompt_tokens_max}`",
        f"- prompt_tokens_mean: `{summary.prompt_tokens_mean:.1f}`",
        f"- total_tokens_max: `{summary.total_tokens_max}`",
        f"- total_tokens_mean: `{summary.total_tokens_mean:.1f}`",
        f"- max_request_chars: `{summary.max_request_chars}`",
        f"- mean_daily_jaccard: `{summary.mean_daily_jaccard:.4f}`",
        f"- mean_daily_precision: `{summary.mean_daily_precision:.4f}`",
        f"- mean_daily_recall: `{summary.mean_daily_recall:.4f}`",
        f"- exact_match_days: `{summary.exact_match_days}` / `{summary.decision_days}`",
        f"- llm_selected_mean_5d_return: `{summary.llm_selected_mean_return:.4%}`",
        f"- llm_not_selected_mean_5d_return: `{summary.llm_not_selected_mean_return:.4%}`",
        f"- teacher_selected_mean_5d_return: `{summary.teacher_selected_mean_return:.4%}`",
        f"- uplift_vs_not_selected: `{summary.uplift_vs_not_selected:.4%}`",
        f"- uplift_vs_teacher_selected: `{summary.uplift_vs_teacher_selected:.4%}`",
        "",
        "## NAV Metrics",
        "",
        f"- llm_final_nav: `{summary.llm_final_nav:.4f}`",
        f"- llm_total_return: `{summary.llm_total_return:.2%}`",
        f"- llm_cagr: `{summary.llm_cagr:.2%}`",
        f"- llm_max_drawdown: `{summary.llm_max_drawdown:.2%}`",
        f"- teacher_target_final_nav: `{summary.teacher_target_final_nav:.4f}`",
        f"- teacher_target_total_return: `{summary.teacher_target_total_return:.2%}`",
        f"- teacher_target_cagr: `{summary.teacher_target_cagr:.2%}`",
        f"- teacher_full_final_nav: `{summary.teacher_full_final_nav:.4f}`",
        f"- teacher_full_total_return: `{summary.teacher_full_total_return:.2%}`",
        f"- teacher_full_cagr: `{summary.teacher_full_cagr:.2%}`",
        f"- hs300_total_return: `{summary.hs300_total_return:.2%}`",
        "",
        "## Artifacts",
        "",
        f"- nav_overlay: `{summary.nav_overlay_path}`",
        f"- nav_curves_csv: `{summary.nav_curve_csv_path}`",
        f"- yearly_returns_csv: `{summary.yearly_returns_csv_path}`",
        f"- agreement_csv: `{summary.agreement_csv_path}`",
        "",
    ]
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def _format_symbol_brief(symbols: Sequence[str], max_items: int = 4) -> str:
    cleaned = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        return "none"
    text = ",".join(cleaned[:max_items])
    if len(cleaned) > max_items:
        text += f"+{len(cleaned) - max_items}"
    return text


def _select_day_rows_by_symbols(day_candidates: pd.DataFrame, selected_symbols: Sequence[str]) -> pd.DataFrame:
    if day_candidates.empty or not selected_symbols:
        return day_candidates.iloc[0:0].copy()
    norm_selected = {_normalize_symbol(symbol) for symbol in selected_symbols if str(symbol).strip()}
    if not norm_selected:
        return day_candidates.iloc[0:0].copy()
    day_selected = day_candidates.copy()
    day_selected["_symbol_norm"] = day_selected["symbol"].astype(str).map(_normalize_symbol)
    day_selected = day_selected[day_selected["_symbol_norm"].isin(norm_selected)].drop(columns=["_symbol_norm"])
    return day_selected.copy()


def _feature_snapshot_for_symbol(
    *,
    day_pool: pd.DataFrame,
    symbol: str,
    feature_cols: Sequence[str],
    max_features: int = 4,
) -> str:
    if not symbol:
        return "none"
    norm_symbol = _normalize_symbol(symbol)
    df = day_pool.copy()
    df["symbol"] = df["symbol"].astype(str).map(_normalize_symbol)
    row = df[df["symbol"] == norm_symbol].head(1)
    if row.empty:
        return norm_symbol
    record = row.iloc[0]
    parts = []
    for col in list(feature_cols)[:max_features]:
        if col not in record.index:
            continue
        value = record.get(col)
        if pd.isna(value):
            continue
        parts.append(f"{col}={_format_float(value)}")
    return f"{norm_symbol}[{', '.join(parts) if parts else 'no_feat'}]"


def _build_warmup_review_card(
    *,
    decision_date: str,
    jaccard: float,
    precision: float,
    recall: float,
    teacher_symbols: Sequence[str],
    llm_symbols: Sequence[str],
    day_pool: pd.DataFrame,
    prompt_features: Sequence[str],
) -> Dict[str, Any]:
    teacher_symbols = [_normalize_symbol(symbol) for symbol in teacher_symbols]
    llm_symbols = [_normalize_symbol(symbol) for symbol in llm_symbols]
    teacher_miss = [symbol for symbol in teacher_symbols if symbol not in llm_symbols]
    false_positive = [symbol for symbol in llm_symbols if symbol not in teacher_symbols]
    teacher_focus_symbol = teacher_miss[0] if teacher_miss else (teacher_symbols[0] if teacher_symbols else "")
    llm_focus_symbol = false_positive[0] if false_positive else (llm_symbols[0] if llm_symbols else "")
    teacher_focus = _feature_snapshot_for_symbol(
        day_pool=day_pool,
        symbol=teacher_focus_symbol,
        feature_cols=prompt_features,
    )
    llm_focus = _feature_snapshot_for_symbol(
        day_pool=day_pool,
        symbol=llm_focus_symbol,
        feature_cols=prompt_features,
    )
    review_card = (
        f"{decision_date} | J={jaccard:.2f} P={precision:.2f} R={recall:.2f} | "
        f"T={_format_symbol_brief(teacher_symbols)} | L={_format_symbol_brief(llm_symbols)} | "
        f"miss={_format_symbol_brief(teacher_miss, max_items=2)} | fp={_format_symbol_brief(false_positive, max_items=2)} | "
        f"Tfocus={teacher_focus} | Lfocus={llm_focus}"
    )[:280]
    return {
        "decision_date": decision_date,
        "jaccard": float(jaccard),
        "precision": float(precision),
        "recall": float(recall),
        "teacher_target_symbols": teacher_symbols,
        "llm_selected_symbols": llm_symbols,
        "teacher_miss_symbols": teacher_miss,
        "llm_false_positive_symbols": false_positive,
        "teacher_focus": teacher_focus,
        "llm_focus": llm_focus,
        "review_card": review_card,
    }


def _safe_mean_return(values: Sequence[Any]) -> float:
    arr = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not arr:
        return 0.0
    return float(np.mean(arr))


def _teacher_alpha_strong_cut(review_entries: Sequence[Dict[str, Any]]) -> float:
    positives = sorted(float(entry.get("teacher_alpha", 0.0)) for entry in review_entries if float(entry.get("teacher_alpha", 0.0)) > 0)
    if not positives:
        return 0.004
    if len(positives) >= 4:
        return max(0.004, float(np.quantile(positives, 0.7)))
    return max(0.004, positives[-1])


def _teacher_eval_tier(alpha: float, *, strong_cut: float) -> str:
    if alpha >= strong_cut and alpha > 0:
        return "must_do"
    if alpha > 0:
        return "pass_positive"
    if alpha >= -0.0015:
        return "neutral_optional"
    return "avoid"


def _render_retained_review_card(entry: Dict[str, Any]) -> str:
    case_id = str(entry.get("case_id", entry.get("decision_date", ""))).strip()
    tier = str(entry.get("teacher_eval_tier", "unknown")).strip()
    alpha = float(entry.get("teacher_alpha", 0.0))
    teacher_symbols = entry.get("teacher_target_symbols", [])
    llm_symbols = entry.get("llm_selected_symbols", [])
    card = (
        f"CID={case_id} | tier={tier} | alpha={alpha*100:+.2f}% | "
        f"J={float(entry.get('jaccard', 0.0)):.2f} | "
        f"T={_format_symbol_brief(teacher_symbols)} | L={_format_symbol_brief(llm_symbols)} | "
        f"Tfocus={str(entry.get('teacher_focus', 'none')).strip()} | "
        f"Lfocus={str(entry.get('llm_focus', 'none')).strip()}"
    )
    return card[:240]


def _annotate_review_entry_tiers(review_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not review_entries:
        return []
    strong_cut = _teacher_alpha_strong_cut(review_entries)
    annotated: List[Dict[str, Any]] = []
    for raw in review_entries:
        entry = dict(raw)
        alpha = float(entry.get("teacher_alpha", 0.0))
        entry["teacher_eval_tier"] = _teacher_eval_tier(alpha, strong_cut=strong_cut)
        entry["review_card"] = _render_retained_review_card(entry)
        annotated.append(entry)
    return annotated


def _retention_case_payload(entry: Dict[str, Any], *, origin: str) -> Dict[str, Any]:
    return {
        "case_id": str(entry.get("case_id", entry.get("decision_date", ""))).strip(),
        "origin": origin,
        "tier": str(entry.get("teacher_eval_tier", "unknown")).strip(),
        "teacher_alpha": round(float(entry.get("teacher_alpha", 0.0)), 6),
        "teacher_target_mean_return": round(float(entry.get("teacher_target_mean_return", 0.0)), 6),
        "pool_mean_return": round(float(entry.get("candidate_pool_mean_return", 0.0)), 6),
        "jaccard": round(float(entry.get("jaccard", 0.0)), 4),
        "precision": round(float(entry.get("precision", 0.0)), 4),
        "recall": round(float(entry.get("recall", 0.0)), 4),
        "teacher_target_symbols": list(entry.get("teacher_target_symbols", []))[:4],
        "llm_selected_symbols": list(entry.get("llm_selected_symbols", []))[:4],
        "teacher_focus": str(entry.get("teacher_focus", "")).strip(),
        "llm_focus": str(entry.get("llm_focus", "")).strip(),
        "review_card": str(entry.get("review_card", "")).strip(),
    }


def _interval_retention_case_payload(
    entry: Dict[str, Any],
    *,
    origin: str,
    feature_priority: Sequence[str],
) -> Dict[str, Any]:
    return {
        "case_id": str(entry.get("case_id", entry.get("decision_date", ""))).strip(),
        "origin": origin,
        "tier": str(entry.get("teacher_eval_tier", "unknown")).strip(),
        "teacher_alpha": round(float(entry.get("teacher_alpha", 0.0)), 6),
        "teacher_target_mean_return": round(float(entry.get("teacher_target_mean_return", 0.0)), 6),
        "pool_mean_return": round(float(entry.get("candidate_pool_mean_return", 0.0)), 6),
        "llm_selected_mean_return": round(float(entry.get("llm_selected_mean_return", 0.0)), 6),
        "jaccard": round(float(entry.get("jaccard", 0.0)), 4),
        "precision": round(float(entry.get("precision", 0.0)), 4),
        "recall": round(float(entry.get("recall", 0.0)), 4),
        "interval_review_card": _build_interval_review_card(entry, feature_priority=feature_priority),
    }


def _parse_retained_case_ids(payload: Dict[str, Any], *, allowed_case_ids: Sequence[str], limit: int) -> List[str]:
    raw = payload.get("retained_case_ids")
    if raw is None:
        raw = payload.get("keep_case_ids")
    if raw is None:
        raw = payload.get("selected_case_ids")
    if not isinstance(raw, list):
        return []
    allowed = {str(item).strip() for item in allowed_case_ids if str(item).strip()}
    kept: List[str] = []
    for item in raw:
        case_id = str(item).strip()
        if case_id in allowed and case_id not in kept:
            kept.append(case_id)
        if len(kept) >= limit:
            break
    return kept


def _fallback_retained_review_entries(
    review_entries: Sequence[Dict[str, Any]],
    *,
    limit: int,
    max_per_tier: int,
) -> List[Dict[str, Any]]:
    if not review_entries:
        return []
    entries = _annotate_review_entry_tiers(review_entries)
    tier_order = ["must_do", "pass_positive", "neutral_optional", "avoid"]
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for tier in tier_order:
        tier_entries = [entry for entry in entries if entry.get("teacher_eval_tier") == tier]
        tier_entries = sorted(
            tier_entries,
            key=lambda entry: (
                0 if str(entry.get("case_origin", "")) == "latest_batch" else 1,
                float(entry.get("jaccard", 1.0)),
                -abs(float(entry.get("teacher_alpha", 0.0))),
                str(entry.get("decision_date", "")),
            ),
        )
        keep_n = min(max_per_tier, max(0, limit - len(selected)))
        for entry in tier_entries[:keep_n]:
            case_id = str(entry.get("case_id", entry.get("decision_date", ""))).strip()
            if case_id and case_id not in selected_ids:
                selected.append(entry)
                selected_ids.add(case_id)
        if len(selected) >= limit:
            return selected[:limit]
    if len(selected) < limit:
        remaining = sorted(
            entries,
            key=lambda entry: (
                0 if str(entry.get("case_origin", "")) == "latest_batch" else 1,
                float(entry.get("jaccard", 1.0)),
                -abs(float(entry.get("teacher_alpha", 0.0))),
            ),
        )
        for entry in remaining:
            case_id = str(entry.get("case_id", entry.get("decision_date", ""))).strip()
            if case_id and case_id not in selected_ids:
                selected.append(entry)
                selected_ids.add(case_id)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _select_review_cards_for_prompt(review_entries: Sequence[Dict[str, Any]], limit: int) -> List[str]:
    if limit <= 0 or not review_entries:
        return []
    if len(review_entries) <= limit:
        return [str(entry.get("review_card", "")).strip() for entry in review_entries if str(entry.get("review_card", "")).strip()]
    hard_target = min(len(review_entries), max(2, limit // 2))
    recent_target = max(0, limit - hard_target)
    hard_cases = sorted(
        review_entries,
        key=lambda entry: (
            float(entry.get("jaccard", 1.0)),
            float(entry.get("precision", 1.0)),
            float(entry.get("recall", 1.0)),
            str(entry.get("decision_date", "")),
        ),
    )[:hard_target]
    recent_cases = list(review_entries)[-recent_target:] if recent_target > 0 else []
    selected: List[str] = []
    seen = set()
    for entry in [*hard_cases, *recent_cases]:
        card = str(entry.get("review_card", "")).strip()
        key = str(entry.get("decision_date", "")).strip()
        if not card or key in seen:
            continue
        selected.append(card)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _distribution_payload_from_summary(
    summary: Mapping[str, Mapping[str, float]],
    *,
    feature_order: Sequence[str],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for feature in feature_order:
        stats = summary.get(feature)
        if not stats:
            continue
        payload.append(
            {
                "feature": feature,
                "count": int(stats.get("count", 0)),
                "min": round(float(stats.get("min", 0.0)), 6),
                "q25": round(float(stats.get("q25", 0.0)), 6),
                "median": round(float(stats.get("median", 0.0)), 6),
                "q75": round(float(stats.get("q75", 0.0)), 6),
                "max": round(float(stats.get("max", 0.0)), 6),
                "mean": round(float(stats.get("mean", 0.0)), 6),
            }
        )
    return payload


def _scope_interval_evidence(
    *,
    review_entries: Sequence[Dict[str, Any]],
    feature_priority: Sequence[str],
) -> Dict[str, Any]:
    entries = list(review_entries)
    if not entries:
        return {
            "feature_priority": list(feature_priority),
            "teacher_selected_profile": [],
            "llm_selected_profile": [],
            "candidate_pool_profile": [],
            "teacher_only_vs_llm_only_gaps": [],
            "teacher_selected_vs_pool_gaps": [],
            "tier_profiles": {},
        }

    def _collect(record_key: str, subset: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in subset:
            rows.extend(list(entry.get(record_key, []) or []))
        return rows

    teacher_selected_records = _collect("teacher_target_feature_records", entries)
    llm_selected_records = _collect("llm_selected_feature_records", entries)
    candidate_pool_records = _collect("candidate_full_input_features", entries)
    teacher_only_records = _collect("teacher_only_feature_records", entries)
    llm_only_records = _collect("llm_only_feature_records", entries)

    teacher_selected_summary = _feature_distribution_from_case_records(teacher_selected_records, feature_priority)
    llm_selected_summary = _feature_distribution_from_case_records(llm_selected_records, feature_priority)
    candidate_pool_summary = _feature_distribution_from_case_records(candidate_pool_records, feature_priority)
    teacher_only_summary = _feature_distribution_from_case_records(teacher_only_records, feature_priority)
    llm_only_summary = _feature_distribution_from_case_records(llm_only_records, feature_priority)

    tier_profiles: Dict[str, Any] = {}
    for tier in ["must_do", "pass_positive", "neutral_optional", "avoid"]:
        tier_entries = [entry for entry in entries if str(entry.get("teacher_eval_tier", "")).strip() == tier]
        if not tier_entries:
            continue
        tier_profiles[tier] = {
            "case_count": len(tier_entries),
            "teacher_selected_profile": _distribution_payload_from_summary(
                _feature_distribution_from_case_records(
                    _collect("teacher_target_feature_records", tier_entries),
                    feature_priority,
                ),
                feature_order=feature_priority,
            ),
            "llm_selected_profile": _distribution_payload_from_summary(
                _feature_distribution_from_case_records(
                    _collect("llm_selected_feature_records", tier_entries),
                    feature_priority,
                ),
                feature_order=feature_priority,
            ),
        }

    return {
        "feature_priority": list(feature_priority),
        "teacher_selected_profile": _distribution_payload_from_summary(
            teacher_selected_summary,
            feature_order=feature_priority,
        ),
        "llm_selected_profile": _distribution_payload_from_summary(
            llm_selected_summary,
            feature_order=feature_priority,
        ),
        "candidate_pool_profile": _distribution_payload_from_summary(
            candidate_pool_summary,
            feature_order=feature_priority,
        ),
        "teacher_only_vs_llm_only_gaps": [
            {
                "feature": row["feature"],
                "direction": row["direction"],
                "median_gap": round(float(row["median_gap"]), 6),
                "teacher_range": {
                    "q25": round(float(row["primary"]["q25"]), 6),
                    "median": round(float(row["primary"]["median"]), 6),
                    "q75": round(float(row["primary"]["q75"]), 6),
                },
                "llm_range": {
                    "q25": round(float(row["secondary"]["q25"]), 6),
                    "median": round(float(row["secondary"]["median"]), 6),
                    "q75": round(float(row["secondary"]["q75"]), 6),
                },
            }
            for row in _feature_gap_rows(
                teacher_only_summary,
                llm_only_summary,
                feature_order=feature_priority,
                top_n=min(6, len(feature_priority)),
            )
        ],
        "teacher_selected_vs_pool_gaps": [
            {
                "feature": row["feature"],
                "direction": row["direction"],
                "median_gap": round(float(row["median_gap"]), 6),
                "teacher_range": {
                    "q25": round(float(row["primary"]["q25"]), 6),
                    "median": round(float(row["primary"]["median"]), 6),
                    "q75": round(float(row["primary"]["q75"]), 6),
                },
                "pool_range": {
                    "q25": round(float(row["secondary"]["q25"]), 6),
                    "median": round(float(row["secondary"]["median"]), 6),
                    "q75": round(float(row["secondary"]["q75"]), 6),
                },
            }
            for row in _feature_gap_rows(
                teacher_selected_summary,
                candidate_pool_summary,
                feature_order=feature_priority,
                top_n=min(6, len(feature_priority)),
            )
        ],
        "tier_profiles": tier_profiles,
    }


def _contains_exemplar_pattern(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"\b20\d{2}-\d{2}-\d{2}\b",
        r"\b20\d{6}\b",
        r"\b\d{6}\b",
        r"\bC\d{2}\b",
        r"\bfavor\b",
        r"\bover\b",
        r"overlap with teacher picks is weak",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _looks_like_interval_rule(text: str) -> bool:
    lowered = text.lower()
    has_action = "action=" in lowered or "prefer when" in lowered or "avoid if" in lowered or "downweight if" in lowered
    has_interval = bool(re.search(r"\bin\s*\[[^\]]+\]", text)) or any(op in text for op in ["<=", ">=", " < ", " > "])
    return has_action and has_interval


def _sanitize_global_lesson_zone_lines(
    *,
    candidate_lines: Sequence[str],
    old_lines: Sequence[str],
    fallback_lines: Sequence[str],
    max_lines: int,
) -> List[str]:
    clean: List[str] = []
    seen = set()
    for raw in list(candidate_lines) + list(old_lines) + list(fallback_lines):
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line):
            continue
        if line in seen:
            continue
        clean.append(line)
        seen.add(line)
        if len(clean) >= max_lines:
            break
    return clean[:max_lines]


def _fallback_global_interval_lesson_lines() -> List[str]:
    return [
        "Use current-scope feature bands as the first filter; treat practice cards as supporting evidence rather than literal templates.",
        "Prefer candidates matching multiple teacher-style intervals simultaneously instead of chasing one extreme feature.",
        "Downweight candidates whose feature profile resembles repeated llm-only drift bands more than teacher-only bands.",
        "If fewer than two scope-core constraints are satisfied, abstain or keep the decision conservative.",
    ]


def _fallback_global_rulecard_lines() -> List[str]:
    return [
        "Use digest background for teacher intent, but let the current scope rule card decide the actual pick standard.",
        "Require multiple scope-core conditions together; one attractive feature alone is not enough.",
        "Treat veto and drift rules as stronger than tie-break rules whenever they conflict.",
        "If a candidate matches llm-drift patterns more than teacher-core patterns, downweight it even if one core rule looks good.",
        "When evidence is mixed, prefer conservative selection or abstain instead of forcing a low-conviction pick.",
    ]


def _fallback_interval_scope_lesson_lines(
    *,
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    teacher_vs_pool = list(interval_evidence.get("teacher_selected_vs_pool_gaps", []) or [])
    teacher_vs_llm = list(interval_evidence.get("teacher_only_vs_llm_only_gaps", []) or [])
    lines: List[str] = []

    def _teacher_band(item: Mapping[str, Any]) -> str:
        return (
            f"{item['feature']} in "
            f"[{_format_float(item['teacher_range']['q25'])}, {_format_float(item['teacher_range']['q75'])}]"
        )

    def _llm_band(item: Mapping[str, Any]) -> str:
        return (
            f"{item['feature']} in "
            f"[{_format_float(item['llm_range']['q25'])}, {_format_float(item['llm_range']['q75'])}]"
        )

    core_items = teacher_vs_pool[:4]
    drift_items = teacher_vs_llm[:4]
    if len(core_items) >= 2:
        lines.append(f"L01: ACTION=upweight | prefer when {_teacher_band(core_items[0])} and {_teacher_band(core_items[1])}.")
    if len(core_items) >= 4:
        lines.append(f"L02: ACTION=upweight | strengthen only if {_teacher_band(core_items[2])} and {_teacher_band(core_items[3])}.")
    elif len(core_items) >= 3:
        lines.append(f"L02: ACTION=upweight | strengthen only if {_teacher_band(core_items[2])}.")
    if len(drift_items) >= 2:
        lines.append(f"L03: ACTION=avoid | downweight when {_llm_band(drift_items[0])} and {_llm_band(drift_items[1])}.")
    elif len(drift_items) >= 1:
        lines.append(f"L03: ACTION=avoid | downweight when {_llm_band(drift_items[0])}.")
    if core_items:
        key_feature = core_items[0]["feature"]
        lines.append(
            f"L04: ACTION=conditional | if fewer than two teacher-style bands match, prefer names that stay inside the teacher band for {key_feature}; otherwise abstain."
        )
    if not lines:
        first_two = list(feature_priority)[:2]
        if len(first_two) == 2:
            lines = [
                f"L01: ACTION=upweight | prefer when {first_two[0]} is inside the teacher-style middle band and {first_two[1]} is also inside the teacher-style middle band.",
                f"L02: ACTION=avoid | downweight when either {first_two[0]} or {first_two[1]} moves into repeated llm-drift territory.",
                "L03: ACTION=conditional | require at least two aligned scope features before taking a strong trade.",
                "L04: ACTION=avoid | abstain when the candidate only matches one isolated feature but misses the rest of the scope profile.",
            ]
    return lines[:max_lines]


def _sanitize_interval_scope_lesson_lines(
    *,
    candidate_lines: Sequence[str],
    old_lines: Sequence[str],
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    clean: List[str] = []
    seen = set()
    for idx, raw in enumerate(list(candidate_lines), start=1):
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line) or not _looks_like_interval_rule(line):
            continue
        if not re.match(r"^L\d{2}:", line):
            line = f"L{idx:02d}: {line}"
        if line in seen:
            continue
        clean.append(line)
        seen.add(line)
        if len(clean) >= max_lines:
            break
    if len(clean) >= 4:
        return clean[:max_lines]
    fallback = _fallback_interval_scope_lesson_lines(
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=max_lines,
    )
    for idx, raw in enumerate(list(old_lines) + fallback, start=len(clean) + 1):
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line):
            continue
        if not re.match(r"^L\d{2}:", line):
            line = f"L{idx:02d}: {line}"
        if line in seen:
            continue
        clean.append(line)
        seen.add(line)
        if len(clean) >= max_lines:
            break
    return clean[:max_lines]


def _looks_like_rulecard_rule(text: str) -> bool:
    lowered = str(text).lower()
    has_action = "action=" in lowered or any(
        token in lowered
        for token in [
            "prefer when",
            "reject when",
            "avoid when",
            "downweight when",
            "abstain when",
            "break ties",
            "require at least",
        ]
    )
    has_condition = (
        bool(re.search(r"\bin\s*\[[^\]]+\]", str(text)))
        or any(op in str(text) for op in ["<=", ">=", " < ", " > "])
        or "teacher band" in lowered
        or "drift" in lowered
        or "fewer than two" in lowered
        or "at least two" in lowered
    )
    return has_action and has_condition


def _infer_rulecard_prefix(text: str) -> str:
    lowered = str(text).lower()
    if "abstain" in lowered:
        return "ABSTAIN"
    if "tie" in lowered:
        return "TIE"
    if "reject" in lowered or "avoid" in lowered or "downweight" in lowered:
        if "drift" in lowered or "false friend" in lowered:
            return "DRIFT"
        return "VETO"
    if "check" in lowered or "require at least" in lowered:
        return "CHECK"
    return "CORE"


def _fallback_rulecard_scope_lesson_lines(
    *,
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    teacher_vs_pool = list(interval_evidence.get("teacher_selected_vs_pool_gaps", []) or [])
    teacher_vs_llm = list(interval_evidence.get("teacher_only_vs_llm_only_gaps", []) or [])
    lines: List[str] = []

    def _teacher_band(item: Mapping[str, Any]) -> str:
        return (
            f"{item['feature']} in "
            f"[{_format_float(item['teacher_range']['q25'])}, {_format_float(item['teacher_range']['q75'])}]"
        )

    def _llm_band(item: Mapping[str, Any]) -> str:
        return (
            f"{item['feature']} in "
            f"[{_format_float(item['llm_range']['q25'])}, {_format_float(item['llm_range']['q75'])}]"
        )

    core_items = teacher_vs_pool[:5]
    drift_items = teacher_vs_llm[:4]
    if len(core_items) >= 2:
        lines.append(f"CORE01: ACTION=upweight | prefer when {_teacher_band(core_items[0])} and {_teacher_band(core_items[1])}.")
    if len(core_items) >= 3:
        lines.append(f"CORE02: ACTION=upweight | add confidence when {_teacher_band(core_items[2])}.")
    if len(core_items) >= 4:
        lines.append(f"CORE03: ACTION=upweight | keep priority only if {_teacher_band(core_items[3])}.")
    if len(core_items) >= 5:
        lines.append(f"CORE04: ACTION=upweight | strongest names usually also keep {_teacher_band(core_items[4])}.")
    if len(core_items) >= 3:
        lines.append(
            f"CORE05: ACTION=upweight | prefer multi-feature agreement when {_teacher_band(core_items[0])} and {_teacher_band(core_items[2])} appear together."
        )
    if len(core_items) >= 4:
        lines.append(
            f"CORE06: ACTION=upweight | keep high conviction only if {_teacher_band(core_items[1])} stays aligned with {_teacher_band(core_items[3])}."
        )
    if len(drift_items) >= 2:
        lines.append(f"VETO01: ACTION=avoid | reject when {_llm_band(drift_items[0])} and {_llm_band(drift_items[1])}.")
    elif len(drift_items) >= 1:
        lines.append(f"VETO01: ACTION=avoid | reject when {_llm_band(drift_items[0])}.")
    if len(drift_items) >= 3:
        lines.append(f"DRIFT01: ACTION=avoid | repeated llm-only drift appears when {_llm_band(drift_items[2])}.")
    if len(drift_items) >= 4:
        lines.append(f"VETO02: ACTION=avoid | reject secondary drift when {_llm_band(drift_items[3])}.")
        lines.append(
            f"DRIFT02: ACTION=avoid | if {_llm_band(drift_items[2])} and {_llm_band(drift_items[3])} stack together, treat the candidate as low quality."
        )
    if len(core_items) >= 2:
        lines.append(
            f"TIE01: ACTION=tiebreak | between survivors, prefer names closer to the teacher band for {core_items[0]['feature']} and {core_items[1]['feature']}."
        )
    elif len(core_items) >= 1:
        lines.append(
            f"TIE01: ACTION=tiebreak | between survivors, prefer names closer to the teacher band for {core_items[0]['feature']}."
        )
    if len(core_items) >= 3:
        lines.append(
            f"TIE02: ACTION=tiebreak | if candidates look similar, use {core_items[2]['feature']} as the second tie-break."
        )
    if len(core_items) >= 4:
        lines.append(
            f"TIE03: ACTION=tiebreak | if the first two tie-breaks remain close, prefer the candidate closer to the teacher band for {core_items[3]['feature']}."
        )
    lines.append("CHECK01: ACTION=check | require at least two core rules before treating a candidate as high conviction.")
    lines.append("CHECK02: ACTION=check | if a veto rule triggers, require at least one extra core confirmation before keeping the name.")
    lines.append("ABSTAIN01: ACTION=abstain | abstain when only one core rule matches or when any veto rule triggers together with weak confirmation.")
    lines.append("ABSTAIN02: ACTION=abstain | abstain when candidates all sit outside the teacher-style middle bands and only weak tie-breaks remain.")
    lines.append("DRIFT03: ACTION=avoid | do not chase single-feature extremes outside the teacher-style middle bands.")
    if not lines:
        first_two = list(feature_priority)[:2]
        if len(first_two) == 2:
            lines = [
                f"CORE01: ACTION=upweight | prefer when {first_two[0]} stays inside the teacher-style middle band and {first_two[1]} also stays inside the teacher-style middle band.",
                f"CORE02: ACTION=upweight | add confidence only when both {first_two[0]} and {first_two[1]} remain aligned together.",
                f"CORE03: ACTION=upweight | rank higher when {first_two[0]} and {first_two[1]} confirm the same teacher-style direction.",
                f"VETO01: ACTION=avoid | reject when either {first_two[0]} or {first_two[1]} moves into repeated llm-drift territory.",
                f"VETO02: ACTION=avoid | reject when {first_two[0]} and {first_two[1]} both leave the teacher-style middle band together.",
                f"TIE01: ACTION=tiebreak | between survivors, prefer names that stay closer to the middle band for {first_two[0]}.",
                "CHECK01: ACTION=check | require at least two aligned scope features before taking a strong trade.",
                "ABSTAIN01: ACTION=abstain | abstain when the candidate only matches one isolated feature but misses the rest of the scope profile.",
                "DRIFT01: ACTION=avoid | do not overreact to one noisy feature without cross-feature confirmation.",
            ]
    return lines[:max_lines]


def _sanitize_rulecard_scope_lesson_lines(
    *,
    candidate_lines: Sequence[str],
    old_lines: Sequence[str],
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    clean: List[str] = []
    seen = set()
    prefix_counter: Dict[str, int] = {}

    def _normalize(raw_line: str) -> str:
        line = str(raw_line).strip()
        if not line:
            return ""
        match = re.match(r"^(CORE|VETO|TIE|DRIFT|CHECK|ABSTAIN)\d{2}:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            rule_index = int(re.search(r"\d{2}", match.group(0)).group(0))
            prefix_counter[prefix] = max(prefix_counter.get(prefix, 0), rule_index)
            return f"{prefix}{rule_index:02d}: {match.group(2).strip()}"
        prefix = _infer_rulecard_prefix(line)
        prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1
        return f"{prefix}{prefix_counter[prefix]:02d}: {line}"

    for raw in list(candidate_lines):
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line) or not _looks_like_rulecard_rule(line):
            continue
        normalized = _normalize(line)
        if not normalized or normalized in seen:
            continue
        clean.append(normalized)
        seen.add(normalized)
        if len(clean) >= max_lines:
            break
    if len(clean) >= 6:
        return clean[:max_lines]
    fallback = _fallback_rulecard_scope_lesson_lines(
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=max_lines,
    )
    for raw in list(old_lines) + fallback:
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line):
            continue
        normalized = _normalize(line)
        if not normalized or normalized in seen:
            continue
        clean.append(normalized)
        seen.add(normalized)
        if len(clean) >= max_lines:
            break
    return clean[:max_lines]


def _fallback_global_weighted_soft_rule_lines() -> List[str]:
    return [
        "Let digest explain teacher intent, but let the current scope weighted rule card decide final ranking.",
        "Treat PREFER rules as weighted evidence, not exact pass-fail gates; prefer stacked agreement over one isolated extreme.",
        "Any hard veto should override soft preferences; soft veto only lowers conviction unless multiple strong preferences agree.",
        "When strong and medium rules conflict, resolve by asking which side better matches the teacher-style band cluster instead of one exact threshold.",
        "If no strong preference survives and the case depends on weak or conflicting evidence, abstain rather than forcing a trade.",
    ]


def _looks_like_weighted_soft_rule(text: str) -> bool:
    lowered = str(text).lower()
    inferred_prefix = _infer_weighted_soft_prefix(text)
    has_role = "type=" in lowered or "action=" in lowered or any(
        token in lowered
        for token in [
            "soft_preference",
            "hard_veto",
            "soft_veto",
            "conflict_resolution",
            "prefer ",
            "reject ",
            "downweight ",
            "abstain ",
            "count strong",
        ]
    )
    has_strength = "strength=" in lowered or any(token in lowered for token in ["strong", "medium", "weak"])
    has_condition = (
        "teacher band" in lowered
        or "drift band" in lowered
        or "inside or slightly" in lowered
        or "near the teacher band" in lowered
        or "well outside" in lowered
        or "if" in lowered
    )
    if inferred_prefix in {"META", "CHECK", "ABSTAIN"}:
        return has_role and has_condition
    return has_role and has_strength and has_condition


def _infer_weighted_soft_prefix(text: str) -> str:
    lowered = str(text).lower()
    if "abstain" in lowered:
        return "ABSTAIN"
    if "conflict" in lowered or "resolve" in lowered or "meta" in lowered:
        return "META"
    if "check" in lowered or "count strong" in lowered or "count medium" in lowered:
        return "CHECK"
    if "veto" in lowered or "reject" in lowered or "downweight" in lowered:
        return "VETO"
    return "PREFER"


def _fallback_weighted_soft_scope_lesson_lines(
    *,
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    teacher_vs_pool = list(interval_evidence.get("teacher_selected_vs_pool_gaps", []) or [])
    teacher_vs_llm = list(interval_evidence.get("teacher_only_vs_llm_only_gaps", []) or [])
    lines: List[str] = []

    def _teacher_soft_band(item: Mapping[str, Any], *, softness: str = "inside or slightly beyond the") -> str:
        return (
            f"{item['feature']} stays {softness} teacher band "
            f"[{_format_float(item['teacher_range']['q25'])}, {_format_float(item['teacher_range']['q75'])}]"
        )

    def _llm_drift_band(item: Mapping[str, Any], *, closeness: str = "inside or close to the") -> str:
        return (
            f"{item['feature']} falls {closeness} repeated llm-drift band "
            f"[{_format_float(item['llm_range']['q25'])}, {_format_float(item['llm_range']['q75'])}]"
        )

    core_items = teacher_vs_pool[:5]
    drift_items = teacher_vs_llm[:4]
    if len(core_items) >= 2:
        lines.append(
            f"PREFER01: TYPE=soft_preference | STRENGTH=strong | favor when {_teacher_soft_band(core_items[0])} and {_teacher_soft_band(core_items[1], softness='near the')} for most of the case."
        )
    if len(core_items) >= 3:
        lines.append(
            f"PREFER02: TYPE=soft_preference | STRENGTH=strong | add conviction when {_teacher_soft_band(core_items[2], softness='comfortably inside the')}."
        )
    if len(core_items) >= 4:
        lines.append(
            f"PREFER03: TYPE=soft_preference | STRENGTH=medium | rank higher when {_teacher_soft_band(core_items[3], softness='inside or just above the')}."
        )
    if len(core_items) >= 5:
        lines.append(
            f"PREFER04: TYPE=soft_preference | STRENGTH=weak | small bonus if {_teacher_soft_band(core_items[4], softness='still near the')}."
        )
    if len(core_items) >= 3:
        lines.append(
            f"PREFER05: TYPE=soft_preference | STRENGTH=strong | strongest setups usually show agreement between {_teacher_soft_band(core_items[0])} and {_teacher_soft_band(core_items[2], softness='near the same')} teacher-style zone."
        )
    if len(core_items) >= 4:
        lines.append(
            f"PREFER06: TYPE=soft_preference | STRENGTH=medium | keep the name above average if {_teacher_soft_band(core_items[1])} while {_teacher_soft_band(core_items[3], softness='not too far from the')}."
        )
    if len(drift_items) >= 2:
        lines.append(
            f"VETO01: TYPE=hard_veto | STRENGTH=hard | reject when {_llm_drift_band(drift_items[0])} and {_llm_drift_band(drift_items[1])} at the same time."
        )
    elif len(drift_items) >= 1:
        lines.append(
            f"VETO01: TYPE=hard_veto | STRENGTH=hard | reject when {_llm_drift_band(drift_items[0], closeness='deep inside the')}."
        )
    if len(drift_items) >= 3:
        lines.append(
            f"VETO02: TYPE=soft_veto | STRENGTH=medium | downweight when {_llm_drift_band(drift_items[2])}, even if one medium preference still looks fine."
        )
    if len(drift_items) >= 4:
        lines.append(
            f"VETO03: TYPE=soft_veto | STRENGTH=weak | lower conviction when {_llm_drift_band(drift_items[3], closeness='near the')} unless multiple strong preferences align."
        )
    lines.append("META01: ACTION=conflict_resolution | if any hard veto triggers, it beats all soft preferences and the candidate should usually be removed.")
    lines.append("META02: ACTION=conflict_resolution | if two strong preferences align against one soft veto, keep the name but lower conviction instead of deleting it immediately.")
    lines.append("META03: ACTION=conflict_resolution | if the choice depends on weak preferences only, prefer the candidate with the tighter multi-feature teacher-style cluster.")
    lines.append("CHECK01: ACTION=check | count strong preferences first, then medium preferences, and treat weak preferences as tie-break support only.")
    lines.append("ABSTAIN01: ACTION=abstain | abstain when no strong preference survives and the decision relies on weak evidence plus conflict-heavy signals.")
    lines.append("ABSTAIN02: ACTION=abstain | abstain when hard veto is absent but the candidate still matches more drift clues than teacher-style clues.")
    if not lines:
        first_two = list(feature_priority)[:2]
        if len(first_two) == 2:
            lines = [
                f"PREFER01: TYPE=soft_preference | STRENGTH=strong | favor when {first_two[0]} and {first_two[1]} both stay near the teacher-style middle zone instead of one isolated extreme.",
                f"PREFER02: TYPE=soft_preference | STRENGTH=medium | add confidence when {first_two[0]} and {first_two[1]} point in the same teacher-style direction.",
                f"VETO01: TYPE=hard_veto | STRENGTH=hard | reject when both {first_two[0]} and {first_two[1]} move into repeated llm-drift territory together.",
                f"VETO02: TYPE=soft_veto | STRENGTH=medium | downweight when one of {first_two[0]} or {first_two[1]} leaves the teacher-style zone while the other is only weakly supportive.",
                "META01: ACTION=conflict_resolution | two strong aligned preferences can survive a soft veto, but not a hard veto.",
                "CHECK01: ACTION=check | count strong evidence first and do not let weak evidence dominate the decision.",
                "ABSTAIN01: ACTION=abstain | abstain when the setup is driven by weak hints rather than multi-feature agreement.",
            ]
    return lines[:max_lines]


def _sanitize_weighted_soft_scope_lesson_lines(
    *,
    candidate_lines: Sequence[str],
    old_lines: Sequence[str],
    feature_priority: Sequence[str],
    interval_evidence: Mapping[str, Any],
    max_lines: int,
) -> List[str]:
    clean: List[str] = []
    seen = set()
    prefix_counter: Dict[str, int] = {}

    def _prefix_counts(lines: Sequence[str]) -> Dict[str, int]:
        counts = {"PREFER": 0, "VETO": 0, "META": 0, "CHECK": 0, "ABSTAIN": 0}
        for item in lines:
            match = re.match(r"^(PREFER|VETO|META|CHECK|ABSTAIN)\d{2}:", str(item).strip(), flags=re.IGNORECASE)
            if match:
                counts[match.group(1).upper()] += 1
        return counts

    def _normalize(raw_line: str) -> str:
        line = str(raw_line).strip()
        if not line:
            return ""
        match = re.match(r"^(PREFER|VETO|META|CHECK|ABSTAIN)\d{2}:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            rule_index = int(re.search(r"\d{2}", match.group(0)).group(0))
            prefix_counter[prefix] = max(prefix_counter.get(prefix, 0), rule_index)
            return f"{prefix}{rule_index:02d}: {match.group(2).strip()}"
        prefix = _infer_weighted_soft_prefix(line)
        prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1
        return f"{prefix}{prefix_counter[prefix]:02d}: {line}"

    for raw in list(candidate_lines):
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line) or not _looks_like_weighted_soft_rule(line):
            continue
        normalized = _normalize(line)
        if not normalized or normalized in seen:
            continue
        clean.append(normalized)
        seen.add(normalized)
        if len(clean) >= max_lines:
            break
    counts = _prefix_counts(clean)
    if (
        len(clean) >= 8
        and counts["PREFER"] >= 4
        and counts["VETO"] >= 2
        and counts["META"] >= 2
        and (counts["CHECK"] + counts["ABSTAIN"]) >= 2
    ):
        return clean[:max_lines]
    fallback = _fallback_weighted_soft_scope_lesson_lines(
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=max_lines,
    )
    for raw in list(old_lines) + fallback:
        line = str(raw).strip()
        if not line or _contains_exemplar_pattern(line):
            continue
        normalized = _normalize(line)
        if not normalized or normalized in seen:
            continue
        clean.append(normalized)
        seen.add(normalized)
        if len(clean) >= max_lines:
            break
    return clean[:max_lines]


def _parse_lesson_zone_lines(payload: Dict[str, Any], *, max_lines: int) -> List[str]:
    raw = payload.get("lesson_zone")
    if raw is None:
        raw = payload.get("lessons")
    if raw is None:
        raw = payload.get("lesson_zone_lines")
    if isinstance(raw, str):
        candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    elif isinstance(raw, list):
        candidates = [str(item).strip() for item in raw if str(item).strip()]
    else:
        candidates = []
    lines: List[str] = []
    seen = set()
    for idx, item in enumerate(candidates, start=1):
        line = item
        if not re.match(r"^L\d{2}:", line):
            line = f"L{idx:02d}: {line}"
        if line not in seen:
            lines.append(line)
            seen.add(line)
        if len(lines) >= max_lines:
            break
    return lines


def _parse_named_lesson_lines(payload: Dict[str, Any], *, keys: Sequence[str], max_lines: int) -> List[str]:
    raw: Any = None
    for key in keys:
        if key in payload:
            raw = payload.get(key)
            break
    if raw is None:
        return []
    return _parse_lesson_zone_lines({keys[0]: raw}, max_lines=max_lines)


def _fallback_lesson_zone_lines(
    *,
    old_lesson_zone: Sequence[str],
    review_entries: Sequence[Dict[str, Any]],
    max_lines: int,
) -> List[str]:
    if old_lesson_zone:
        return [str(line).strip() for line in old_lesson_zone if str(line).strip()][:max_lines]
    lines: List[str] = []
    for idx, entry in enumerate(review_entries, start=1):
        teacher_focus = str(entry.get("teacher_focus", "none")).strip()
        llm_focus = str(entry.get("llm_focus", "none")).strip()
        line = (
            f"L{idx:02d}: On {entry.get('decision_date')}, favor {teacher_focus} "
            f"over {llm_focus} when overlap with teacher picks is weak."
        )
        lines.append(line[:260])
        if len(lines) >= max_lines:
            break
    return lines


def _extract_warmup_review_entries(
    *,
    report_dir: Path,
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
    prompt_features: Sequence[str],
    lesson_feature_cols: Sequence[str],
    teacher_round_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    agreement_path = report_dir / "daily_agreement.csv"
    llm_path = report_dir / "llm_selected_signals.csv"
    if not (agreement_path.exists() and llm_path.exists()):
        return []
    agreement_df = pd.read_csv(agreement_path)
    llm_df = pd.read_csv(llm_path)
    if agreement_df.empty:
        return []
    entries: List[Dict[str, Any]] = []
    for _, row in agreement_df.sort_values("decision_date").iterrows():
        day = pd.Timestamp(row["decision_date"])
        day_text = day.strftime("%Y-%m-%d")
        case_id = day.strftime("%Y%m%d")
        day_pool = candidate_pool_df[candidate_pool_df["signal_date"] == day].copy()
        if day_pool.empty:
            continue
        teacher_rows = teacher_target_df[teacher_target_df["signal_date"] == day].copy()
        llm_rows = llm_df[pd.to_datetime(llm_df["signal_date"]) == day].copy() if "signal_date" in llm_df.columns else pd.DataFrame()
        teacher_symbols = teacher_rows["symbol"].astype(str).tolist()
        llm_symbols = llm_rows["symbol"].astype(str).tolist()
        case_records = _lesson_case_records(day_pool, lesson_feature_cols)
        teacher_target_feature_records = _filter_case_records_by_symbols(case_records, teacher_symbols)
        llm_selected_feature_records = _filter_case_records_by_symbols(case_records, llm_symbols)
        teacher_only_symbols = [symbol for symbol in teacher_symbols if _normalize_symbol(symbol) not in {_normalize_symbol(item) for item in llm_symbols}]
        llm_only_symbols = [symbol for symbol in llm_symbols if _normalize_symbol(symbol) not in {_normalize_symbol(item) for item in teacher_symbols}]
        teacher_only_feature_records = _filter_case_records_by_symbols(case_records, teacher_only_symbols)
        llm_only_feature_records = _filter_case_records_by_symbols(case_records, llm_only_symbols)
        review_entry = _build_warmup_review_card(
            decision_date=day_text,
            jaccard=float(row["jaccard"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            teacher_symbols=teacher_symbols,
            llm_symbols=llm_symbols,
            day_pool=day_pool,
            prompt_features=prompt_features,
        )
        review_entry.update(
            {
                "case_id": case_id,
                "teacher_round_ids": list(teacher_round_ids),
                "teacher_target_mean_return": _safe_mean_return(teacher_rows.get("future_return_5d", pd.Series(dtype=float)).tolist()),
                "candidate_pool_mean_return": _safe_mean_return(day_pool.get("future_return_5d", pd.Series(dtype=float)).tolist()),
                "llm_selected_mean_return": _safe_mean_return(llm_rows.get("future_return_5d", pd.Series(dtype=float)).tolist()),
                "candidate_full_input_features": case_records,
                "teacher_target_feature_records": teacher_target_feature_records,
                "llm_selected_feature_records": llm_selected_feature_records,
                "teacher_only_feature_records": teacher_only_feature_records,
                "llm_only_feature_records": llm_only_feature_records,
            }
        )
        review_entry["teacher_alpha"] = float(review_entry["teacher_target_mean_return"] - review_entry["candidate_pool_mean_return"])
        entries.append(review_entry)
    return entries


def _rewrite_lesson_zone_iterative(
    *,
    config: ApprenticeReplayConfig,
    metas: Sequence[Dict[str, Any]],
    review_entries: Sequence[Dict[str, Any]],
    old_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], Dict[str, Any]]:
    teacher_cards = [
        (
            f"{meta['round_id']} | family={meta['research_family']} | template={meta['sample_template']} | "
            f"bands={_compact_band_hint(meta.get('preference_bands', []))} | "
            f"factor_rules={_compact_factor_rule_hint(meta.get('factor_analysis_summary', {}), top_features=2, top_combos=1)}"
        )
        for meta in metas
    ]
    review_cards = [str(entry.get("review_card", "")).strip() for entry in review_entries if str(entry.get("review_card", "")).strip()]
    latest_batch_cases = [
        {
            "decision_date": entry.get("decision_date"),
            "jaccard": entry.get("jaccard"),
            "precision": entry.get("precision"),
            "recall": entry.get("recall"),
            "teacher_target_symbols": entry.get("teacher_target_symbols", []),
            "llm_selected_symbols": entry.get("llm_selected_symbols", []),
            "teacher_focus": entry.get("teacher_focus", ""),
            "llm_focus": entry.get("llm_focus", ""),
        }
        for entry in batch_entries
    ]
    latest_batch_case_details = [
        {
            "decision_date": entry.get("decision_date"),
            "teacher_focus": entry.get("teacher_focus", ""),
            "llm_focus": entry.get("llm_focus", ""),
            "candidate_full_input_features": entry.get("candidate_full_input_features", []),
        }
        for entry in batch_entries
    ]
    system = (
        "You are maintaining the mutable Lesson Zone for QuantApprentice. "
        "This Lesson Zone will be injected directly into future no-teacher-signal trading prompts. "
        "You have full authority to rewrite, merge, delete, or replace old lessons if review evidence contradicts them. "
        "Do not preserve old lines unless they still help. "
        "Do not write research plans or process summaries. "
        "Return strict JSON with keys: lesson_zone, revision_notes. "
        f"lesson_zone must be a list of 4 to {config.warmup_lesson_zone_max_lines} concise, feature-aware, directly actionable lines."
    )
    user = json.dumps(
        {
            "objective": "rewrite the lesson zone for future stock-selection decisions without current teacher signal",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "teacher_cards": teacher_cards,
            "old_lesson_zone": list(old_lesson_zone),
            "review_area_cards": review_cards,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    parsed_payload: Dict[str, Any]
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {"lesson_zone": _fallback_lesson_zone_lines(old_lesson_zone=old_lesson_zone, review_entries=batch_entries, max_lines=config.warmup_lesson_zone_max_lines)}
    lesson_zone_lines = _parse_lesson_zone_lines(parsed_payload, max_lines=config.warmup_lesson_zone_max_lines)
    if not lesson_zone_lines:
        lesson_zone_lines = _fallback_lesson_zone_lines(
            old_lesson_zone=old_lesson_zone,
            review_entries=batch_entries,
            max_lines=config.warmup_lesson_zone_max_lines,
        )
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "old_lesson_zone": list(old_lesson_zone),
        "review_area_count": len(review_cards),
        "latest_batch_case_count": len(latest_batch_cases),
        "latest_batch_case_details": latest_batch_case_details,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "lesson_zone": lesson_zone_lines,
    }
    return lesson_zone_lines, artifact


def _generate_warmup_lessons_iterative(
    *,
    config: ApprenticeReplayConfig,
    warm_cfg: ApprenticeReplayConfig,
    metas: Sequence[Dict[str, Any]],
    negative_metas: Sequence[Dict[str, Any]],
    prompt_features: Sequence[str],
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
) -> Dict[str, Any]:
    warmup_report_dir = REPORT_ROOT / warm_cfg.run_id()
    warmup_report_dir.mkdir(parents=True, exist_ok=True)
    _progress(
        "iterative_v3 warmup start "
        f"run_id={warm_cfg.run_id()} sample_count={config.warmup_sample_count} "
        f"batch_size={config.warmup_batch_size}"
    )
    lesson_feature_cols = _lesson_feature_columns(candidate_pool_df)
    sample_dates = _sample_uniform_dates(
        candidate_pool_df["signal_date"].tolist(),
        config.warmup_sample_count,
        sample_seed=config.sample_seed,
    )
    sample_dates = sorted(pd.Timestamp(item) for item in sample_dates)
    batch_size = max(1, int(config.warmup_batch_size))
    date_batches = [sample_dates[idx : idx + batch_size] for idx in range(0, len(sample_dates), batch_size)]
    review_bank: List[Dict[str, Any]] = []
    lesson_zone_lines: List[str] = []
    batch_history: List[Dict[str, Any]] = []
    batch_report_dirs: List[str] = []
    for batch_index, batch_dates in enumerate(date_batches, start=1):
        _progress(
            "iterative_v3 warmup batch start "
            f"batch={batch_index}/{len(date_batches)} dates={len(batch_dates)} "
            f"date_from={pd.Timestamp(batch_dates[0]).strftime('%Y-%m-%d')} "
            f"date_to={pd.Timestamp(batch_dates[-1]).strftime('%Y-%m-%d')}"
        )
        batch_cfg = replace(
            warm_cfg,
            run_tag=f"{warm_cfg.run_tag}_batch{batch_index:02d}of{len(date_batches)}",
        )
        batch_pool = candidate_pool_df[candidate_pool_df["signal_date"].isin(batch_dates)].copy()
        batch_target = teacher_target_df[teacher_target_df["signal_date"].isin(batch_dates)].copy()
        batch_full = batch_target.copy()
        _run_replay(
            config=batch_cfg,
            candidate_pool_df=batch_pool,
            teacher_target_df=batch_target,
            teacher_full_df=batch_full,
            prompt_builder=_daily_prompt_multi,
            prompt_builder_kwargs={
                "metas": list(metas),
                "negative_metas": list(negative_metas),
                "prompt_features": prompt_features,
                "warmup_lessons": lesson_zone_lines,
                "warmup_review_cards": _select_review_cards_for_prompt(review_bank, config.warmup_review_memory_limit),
            },
        )
        batch_report_dir = REPORT_ROOT / batch_cfg.run_id()
        batch_report_dirs.append(_relative(batch_report_dir))
        batch_entries = _extract_warmup_review_entries(
            report_dir=batch_report_dir,
            candidate_pool_df=batch_pool,
            teacher_target_df=batch_target,
            prompt_features=prompt_features,
            lesson_feature_cols=lesson_feature_cols,
            teacher_round_ids=config.teacher_round_ids,
        )
        review_bank.extend(batch_entries)
        lesson_zone_lines, lesson_artifact = _rewrite_lesson_zone_iterative(
            config=config,
            metas=metas,
            review_entries=review_bank,
            old_lesson_zone=lesson_zone_lines,
            batch_entries=batch_entries,
            batch_index=batch_index,
            total_batches=len(date_batches),
        )
        batch_payload = {
            "batch_index": batch_index,
            "batch_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in batch_dates],
            "batch_report_dir": _relative(batch_report_dir),
            "batch_review_count": len(batch_entries),
            "cumulative_review_count": len(review_bank),
            "lesson_zone": lesson_zone_lines,
            "lesson_artifact": lesson_artifact,
        }
        batch_history.append(batch_payload)
        _write_json(warmup_report_dir / f"batch_{batch_index:02d}_state.json", batch_payload)
        _write_json(
            warmup_report_dir / f"batch_{batch_index:02d}_review_entries.json",
            {"entries": batch_entries},
        )
        _progress(
            "iterative_v3 warmup batch done "
            f"batch={batch_index}/{len(date_batches)} batch_review={len(batch_entries)} "
            f"cumulative_review={len(review_bank)} lesson_lines={len(lesson_zone_lines)}"
        )

    review_area_payload = {
        "curriculum": "iterative_v3",
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": batch_size,
        "review_entry_count": len(review_bank),
        "entries": review_bank,
        "batch_report_dirs": batch_report_dirs,
    }
    lesson_payload = {
        "curriculum": "iterative_v3",
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": batch_size,
        "lesson_zone": lesson_zone_lines,
        "batch_history": batch_history,
        "review_entry_count": len(review_bank),
        "review_cards_for_prompt": _select_review_cards_for_prompt(review_bank, config.warmup_review_memory_limit),
    }
    _write_json(warmup_report_dir / "warmup_review_area.json", review_area_payload)
    with (warmup_report_dir / "warmup_review_area.jsonl").open("w", encoding="utf-8") as f:
        for entry in review_bank:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _write_json(warmup_report_dir / "warmup_lessons.json", lesson_payload)
    _write_json(warmup_report_dir / "warmup_lesson_bank.json", {"lesson_count": len(review_bank), "lessons": review_bank})
    (warmup_report_dir / "warmup_lessons.md").write_text(
        "\n".join(
            [
                "# Warmup Lessons",
                "",
                "## Lesson Zone",
                "",
                *(lesson_zone_lines or ["none"]),
                "",
                "## Practice Review Area",
                "",
                *(_select_review_cards_for_prompt(review_bank, config.warmup_review_memory_limit) or ["none"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "lesson_zone_lines": lesson_zone_lines,
        "review_cards_for_prompt": _select_review_cards_for_prompt(review_bank, config.warmup_review_memory_limit),
        "review_bank": review_bank,
        "artifact_dir": _relative(warmup_report_dir),
    }


def _rewrite_lesson_zone_iterative_v4(
    *,
    config: ApprenticeReplayConfig,
    metas: Sequence[Dict[str, Any]],
    retained_review_entries: Sequence[Dict[str, Any]],
    old_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    teacher_cards = [
        (
            f"{meta['round_id']} | family={meta['research_family']} | template={meta['sample_template']} | "
            f"bands={_compact_band_hint(meta.get('preference_bands', []))} | "
            f"factor_rules={_compact_factor_rule_hint(meta.get('factor_analysis_summary', {}), top_features=2, top_combos=1)}"
        )
        for meta in metas
    ]
    retained_entries = [dict(entry, case_origin="retained") for entry in retained_review_entries]
    latest_entries = [dict(entry, case_origin="latest_batch") for entry in batch_entries]
    candidate_entries = _annotate_review_entry_tiers([*retained_entries, *latest_entries])
    retained_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "retained"]
    latest_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "latest_batch"]
    available_case_ids = [str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in candidate_entries]
    current_review_cases = [_retention_case_payload(entry, origin="retained") for entry in retained_entries]
    latest_batch_cases = [_retention_case_payload(entry, origin="latest_batch") for entry in latest_entries]
    system = (
        "You are maintaining two bounded memories for QuantApprentice: the mutable Lesson Zone and the Practice Review Area. "
        "The Practice Review Area cannot keep every warmup case. "
        "You must actively decide which cases remain as Golden cases for future no-teacher-signal trading decisions. "
        "Cases not selected will be removed from memory. "
        "Prefer a compact, high-information memory with non-redundant examples across four tiers: "
        "must_do, pass_positive, neutral_optional, avoid. "
        "Keep cases that best teach decision boundaries, especially where teacher alpha, overlap error, or avoid-style signal is instructive. "
        "Return strict JSON with keys: lesson_zone, retained_case_ids, revision_notes. "
        f"lesson_zone must be a list of 4 to {config.warmup_lesson_zone_max_lines} concise, feature-aware, directly actionable lines. "
        f"retained_case_ids must contain at most {config.warmup_review_memory_limit} case ids."
    )
    user = json.dumps(
        {
            "objective": "rewrite the lesson zone and prune the practice review area into a compact golden-case memory",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "teacher_cards": teacher_cards,
            "old_lesson_zone": list(old_lesson_zone),
            "memory_budget": {
                "max_retained_cases": config.warmup_review_memory_limit,
                "preferred_cross_tier_coverage": ["must_do", "pass_positive", "neutral_optional", "avoid"],
                "max_cases_per_tier_fallback": config.warmup_retained_case_max_per_tier,
            },
            "current_review_area_cases": current_review_cases,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    parsed_payload: Dict[str, Any]
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {
            "lesson_zone": _fallback_lesson_zone_lines(
                old_lesson_zone=old_lesson_zone,
                review_entries=latest_entries or candidate_entries,
                max_lines=config.warmup_lesson_zone_max_lines,
            )
        }
    lesson_zone_lines = _parse_lesson_zone_lines(parsed_payload, max_lines=config.warmup_lesson_zone_max_lines)
    if not lesson_zone_lines:
        lesson_zone_lines = _fallback_lesson_zone_lines(
            old_lesson_zone=old_lesson_zone,
            review_entries=latest_entries or candidate_entries,
            max_lines=config.warmup_lesson_zone_max_lines,
        )
    retained_case_ids = _parse_retained_case_ids(
        parsed_payload,
        allowed_case_ids=available_case_ids,
        limit=config.warmup_review_memory_limit,
    )
    if retained_case_ids:
        by_id = {
            str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
            for entry in candidate_entries
        }
        next_review_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    else:
        next_review_entries = _fallback_retained_review_entries(
            candidate_entries,
            limit=config.warmup_review_memory_limit,
            max_per_tier=config.warmup_retained_case_max_per_tier,
        )
        retained_case_ids = [
            str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in next_review_entries
        ]
    next_review_entries = _annotate_review_entry_tiers(next_review_entries)
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "old_lesson_zone": list(old_lesson_zone),
        "candidate_review_case_count": len(candidate_entries),
        "current_review_area_count": len(current_review_cases),
        "latest_batch_case_count": len(latest_batch_cases),
        "retained_case_count": len(next_review_entries),
        "retained_case_ids": retained_case_ids,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "lesson_zone": lesson_zone_lines,
        "retained_review_cards": [str(entry.get("review_card", "")).strip() for entry in next_review_entries],
        "latest_batch_case_details": [
            {
                "case_id": entry.get("case_id"),
                "teacher_eval_tier": entry.get("teacher_eval_tier"),
                "teacher_alpha": entry.get("teacher_alpha"),
                "teacher_focus": entry.get("teacher_focus"),
                "llm_focus": entry.get("llm_focus"),
                "candidate_full_input_features": entry.get("candidate_full_input_features", []),
            }
            for entry in latest_entries
        ],
    }
    return lesson_zone_lines, next_review_entries, artifact


def _generate_warmup_lessons_iterative_v4(
    *,
    config: ApprenticeReplayConfig,
    warm_cfg: ApprenticeReplayConfig,
    metas: Sequence[Dict[str, Any]],
    negative_metas: Sequence[Dict[str, Any]],
    prompt_features: Sequence[str],
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
) -> Dict[str, Any]:
    warmup_report_dir = REPORT_ROOT / warm_cfg.run_id()
    warmup_report_dir.mkdir(parents=True, exist_ok=True)
    _progress(
        "iterative_v4 warmup start "
        f"run_id={warm_cfg.run_id()} sample_count={config.warmup_sample_count} "
        f"batch_size={config.warmup_batch_size} retained_limit={config.warmup_review_memory_limit}"
    )
    lesson_feature_cols = _lesson_feature_columns(candidate_pool_df)
    sample_dates = _sample_uniform_dates(
        candidate_pool_df["signal_date"].tolist(),
        config.warmup_sample_count,
        sample_seed=config.sample_seed,
    )
    sample_dates = sorted(pd.Timestamp(item) for item in sample_dates)
    batch_size = max(1, int(config.warmup_batch_size))
    date_batches = [sample_dates[idx : idx + batch_size] for idx in range(0, len(sample_dates), batch_size)]
    retained_review_entries: List[Dict[str, Any]] = []
    lesson_zone_lines: List[str] = []
    batch_history: List[Dict[str, Any]] = []
    batch_report_dirs: List[str] = []
    all_seen_entries_count = 0
    for batch_index, batch_dates in enumerate(date_batches, start=1):
        _progress(
            "iterative_v4 warmup batch start "
            f"batch={batch_index}/{len(date_batches)} dates={len(batch_dates)} "
            f"date_from={pd.Timestamp(batch_dates[0]).strftime('%Y-%m-%d')} "
            f"date_to={pd.Timestamp(batch_dates[-1]).strftime('%Y-%m-%d')}"
        )
        batch_cfg = replace(
            warm_cfg,
            run_tag=f"{warm_cfg.run_tag}_batch{batch_index:02d}of{len(date_batches)}",
        )
        batch_pool = candidate_pool_df[candidate_pool_df["signal_date"].isin(batch_dates)].copy()
        batch_target = teacher_target_df[teacher_target_df["signal_date"].isin(batch_dates)].copy()
        batch_full = batch_target.copy()
        _run_replay(
            config=batch_cfg,
            candidate_pool_df=batch_pool,
            teacher_target_df=batch_target,
            teacher_full_df=batch_full,
            prompt_builder=_daily_prompt_multi,
            prompt_builder_kwargs={
                "metas": list(metas),
                "negative_metas": list(negative_metas),
                "prompt_features": prompt_features,
                "warmup_lessons": lesson_zone_lines,
                "warmup_review_cards": [str(entry.get("review_card", "")).strip() for entry in retained_review_entries][: config.warmup_review_memory_limit],
            },
        )
        batch_report_dir = REPORT_ROOT / batch_cfg.run_id()
        batch_report_dirs.append(_relative(batch_report_dir))
        batch_entries = _extract_warmup_review_entries(
            report_dir=batch_report_dir,
            candidate_pool_df=batch_pool,
            teacher_target_df=batch_target,
            prompt_features=prompt_features,
            lesson_feature_cols=lesson_feature_cols,
            teacher_round_ids=config.teacher_round_ids,
        )
        all_seen_entries_count += len(batch_entries)
        lesson_zone_lines, retained_review_entries, lesson_artifact = _rewrite_lesson_zone_iterative_v4(
            config=config,
            metas=metas,
            retained_review_entries=retained_review_entries,
            old_lesson_zone=lesson_zone_lines,
            batch_entries=batch_entries,
            batch_index=batch_index,
            total_batches=len(date_batches),
        )
        batch_payload = {
            "batch_index": batch_index,
            "batch_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in batch_dates],
            "batch_report_dir": _relative(batch_report_dir),
            "batch_review_count": len(batch_entries),
            "all_seen_entries_count": all_seen_entries_count,
            "retained_review_count": len(retained_review_entries),
            "lesson_zone": lesson_zone_lines,
            "lesson_artifact": lesson_artifact,
        }
        batch_history.append(batch_payload)
        _write_json(warmup_report_dir / f"batch_{batch_index:02d}_state.json", batch_payload)
        _write_json(
            warmup_report_dir / f"batch_{batch_index:02d}_review_entries.json",
            {"entries": batch_entries},
        )
        _progress(
            "iterative_v4 warmup batch done "
            f"batch={batch_index}/{len(date_batches)} batch_review={len(batch_entries)} "
            f"retained_review={len(retained_review_entries)} lesson_lines={len(lesson_zone_lines)}"
        )

    retained_review_entries = _annotate_review_entry_tiers(retained_review_entries)
    review_area_payload = {
        "curriculum": "iterative_v4",
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": batch_size,
        "all_seen_entries_count": all_seen_entries_count,
        "retained_review_entry_count": len(retained_review_entries),
        "entries": retained_review_entries,
        "batch_report_dirs": batch_report_dirs,
    }
    lesson_payload = {
        "curriculum": "iterative_v4",
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": batch_size,
        "lesson_zone": lesson_zone_lines,
        "batch_history": batch_history,
        "retained_review_entry_count": len(retained_review_entries),
        "review_cards_for_prompt": [str(entry.get("review_card", "")).strip() for entry in retained_review_entries][: config.warmup_review_memory_limit],
    }
    _write_json(warmup_report_dir / "warmup_review_area.json", review_area_payload)
    with (warmup_report_dir / "warmup_review_area.jsonl").open("w", encoding="utf-8") as f:
        for entry in retained_review_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _write_json(warmup_report_dir / "warmup_lessons.json", lesson_payload)
    _write_json(
        warmup_report_dir / "warmup_lesson_bank.json",
        {"lesson_count": len(retained_review_entries), "lessons": retained_review_entries},
    )
    (warmup_report_dir / "warmup_lessons.md").write_text(
        "\n".join(
            [
                "# Warmup Lessons",
                "",
                "## Lesson Zone",
                "",
                *(lesson_zone_lines or ["none"]),
                "",
                "## Retained Golden Cases",
                "",
                *([str(entry.get("review_card", "")).strip() for entry in retained_review_entries] or ["none"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "lesson_zone_lines": lesson_zone_lines,
        "review_cards_for_prompt": [str(entry.get("review_card", "")).strip() for entry in retained_review_entries][: config.warmup_review_memory_limit],
        "review_bank": retained_review_entries,
        "artifact_dir": _relative(warmup_report_dir),
    }


def _rewrite_scoped_lesson_zone_iterative_v5(
    *,
    config: ApprenticeReplayConfig,
    meta: Dict[str, Any],
    scope_round_id: str,
    source_round_id: str,
    retained_review_entries: Sequence[Dict[str, Any]],
    old_global_lesson_zone: Sequence[str],
    old_scope_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    scope_card = _scope_domain_card(meta, scope_round_id=scope_round_id, source_round_id=source_round_id)
    retained_entries = [dict(entry, case_origin="retained") for entry in retained_review_entries]
    latest_entries = [dict(entry, case_origin="latest_batch") for entry in batch_entries]
    candidate_entries = _annotate_review_entry_tiers([*retained_entries, *latest_entries])
    retained_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "retained"]
    latest_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "latest_batch"]
    available_case_ids = [str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in candidate_entries]
    current_review_cases = [_retention_case_payload(entry, origin="retained") for entry in retained_entries]
    latest_batch_cases = [_retention_case_payload(entry, origin="latest_batch") for entry in latest_entries]
    system = (
        "You are maintaining three bounded memories for QuantApprentice warmup: "
        "Global Lesson Zone, Current Teacher Scope Lesson Zone, and Current Scope Practice Review Area. "
        "Global Lesson Zone should contain only cross-teacher meta rules that are likely reusable across strategies. "
        "Current Teacher Scope Lesson Zone should contain rules specific to the active teacher family/template. "
        "Do not move current-scope thresholds into Global Lesson Zone unless they clearly generalize. "
        "Do not delete other hidden teacher scopes; you only control the current scope and the global zone in this task. "
        "Practice Review Area cannot keep every case. "
        "Return strict JSON with keys: global_lesson_zone, scope_lesson_zone, retained_case_ids, revision_notes. "
        f"global_lesson_zone must be a list of 2 to {config.warmup_lesson_zone_max_lines} concise lines. "
        f"scope_lesson_zone must be a list of 4 to {config.warmup_lesson_zone_max_lines} concise, feature-aware, directly actionable lines. "
        f"retained_case_ids must contain at most {config.warmup_review_memory_limit} case ids."
    )
    user = json.dumps(
        {
            "objective": "rewrite the global lesson zone and the current teacher scope lesson zone after one single-teacher warmup batch",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "scope_card": scope_card,
            "old_global_lesson_zone": list(old_global_lesson_zone),
            "old_scope_lesson_zone": list(old_scope_lesson_zone),
            "memory_budget": {
                "max_retained_cases": config.warmup_review_memory_limit,
                "preferred_cross_tier_coverage": ["must_do", "pass_positive", "neutral_optional", "avoid"],
                "max_cases_per_tier_fallback": config.warmup_retained_case_max_per_tier,
            },
            "current_scope_review_area_cases": current_review_cases,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {}
    global_lesson_zone_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["global_lesson_zone", "global_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    if not global_lesson_zone_lines:
        global_lesson_zone_lines = [str(line).strip() for line in old_global_lesson_zone if str(line).strip()][
            : config.warmup_lesson_zone_max_lines
        ]
    scope_lesson_zone_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["scope_lesson_zone", "lesson_zone", "scope_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    if not scope_lesson_zone_lines:
        scope_lesson_zone_lines = _fallback_lesson_zone_lines(
            old_lesson_zone=old_scope_lesson_zone,
            review_entries=latest_entries or candidate_entries,
            max_lines=config.warmup_lesson_zone_max_lines,
        )
    retained_case_ids = _parse_retained_case_ids(
        parsed_payload,
        allowed_case_ids=available_case_ids,
        limit=config.warmup_review_memory_limit,
    )
    if retained_case_ids:
        by_id = {
            str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
            for entry in candidate_entries
        }
        next_review_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    else:
        next_review_entries = _fallback_retained_review_entries(
            candidate_entries,
            limit=config.warmup_review_memory_limit,
            max_per_tier=config.warmup_retained_case_max_per_tier,
        )
        retained_case_ids = [
            str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in next_review_entries
        ]
    next_review_entries = _annotate_review_entry_tiers(next_review_entries)
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "scope_round_id": scope_round_id,
        "source_round_id": source_round_id,
        "scope_card": scope_card,
        "old_global_lesson_zone": list(old_global_lesson_zone),
        "old_scope_lesson_zone": list(old_scope_lesson_zone),
        "candidate_review_case_count": len(candidate_entries),
        "current_review_area_count": len(current_review_cases),
        "latest_batch_case_count": len(latest_batch_cases),
        "retained_case_count": len(next_review_entries),
        "retained_case_ids": retained_case_ids,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "global_lesson_zone": global_lesson_zone_lines,
        "scope_lesson_zone": scope_lesson_zone_lines,
        "retained_review_cards": [str(entry.get("review_card", "")).strip() for entry in next_review_entries],
        "latest_batch_case_details": [
            {
                "case_id": entry.get("case_id"),
                "teacher_eval_tier": entry.get("teacher_eval_tier"),
                "teacher_alpha": entry.get("teacher_alpha"),
                "teacher_focus": entry.get("teacher_focus"),
                "llm_focus": entry.get("llm_focus"),
                "candidate_full_input_features": entry.get("candidate_full_input_features", []),
            }
            for entry in latest_entries
        ],
    }
    return global_lesson_zone_lines, scope_lesson_zone_lines, next_review_entries, artifact


def _rewrite_scoped_lesson_zone_iterative_v6_interval(
    *,
    config: ApprenticeReplayConfig,
    meta: Dict[str, Any],
    scope_round_id: str,
    source_round_id: str,
    retained_review_entries: Sequence[Dict[str, Any]],
    old_global_lesson_zone: Sequence[str],
    old_scope_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    scope_card = _scope_domain_card(meta, scope_round_id=scope_round_id, source_round_id=source_round_id)
    feature_priority = list(scope_card.get("top_features") or [])[: max(4, config.prompt_feature_count)]
    retained_entries = [dict(entry, case_origin="retained") for entry in retained_review_entries]
    latest_entries = [dict(entry, case_origin="latest_batch") for entry in batch_entries]
    candidate_entries = _annotate_review_entry_tiers([*retained_entries, *latest_entries])
    retained_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "retained"]
    latest_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "latest_batch"]
    available_case_ids = [str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in candidate_entries]
    interval_evidence = _scope_interval_evidence(review_entries=candidate_entries, feature_priority=feature_priority)
    current_review_cases = [
        _interval_retention_case_payload(entry, origin="retained", feature_priority=feature_priority)
        for entry in retained_entries
    ]
    latest_batch_cases = [
        _interval_retention_case_payload(entry, origin="latest_batch", feature_priority=feature_priority)
        for entry in latest_entries
    ]
    system = (
        "You are maintaining three bounded memories for QuantApprentice warmup: "
        "Global Lesson Zone, Current Teacher Scope Lesson Zone, and Current Scope Practice Review Area. "
        "This version uses interval-rule lessons only. "
        "Global Lesson Zone should contain only cross-teacher meta rules that generalize across strategies. "
        "Current Teacher Scope Lesson Zone must contain only abstract interval rules for the active teacher family/template. "
        "Never mention dates, stock codes, candidate IDs, or pairwise 'favor A over B' examples. "
        "Never write 'when overlap with teacher picks is weak'. "
        "Each scope lesson line must be an interval-style rule with explicit feature bands or inequalities and an action tag. "
        "Good format examples: "
        "'L01: ACTION=upweight | prefer when dJ_3 in [-60,-35] and close_to_ma20 in [1.01,1.07].' "
        "'L02: ACTION=avoid | downweight when volatility_20 >= 0.055 and volume_zscore_20 in [1.2,2.5].' "
        "Practice Review Area cannot keep every case. "
        "Return strict JSON with keys: global_lesson_zone, scope_lesson_zone, retained_case_ids, revision_notes. "
        f"global_lesson_zone must be a list of 2 to {config.warmup_lesson_zone_max_lines} concise lines. "
        f"scope_lesson_zone must be a list of 4 to {config.warmup_lesson_zone_max_lines} concise interval-rule lines. "
        f"retained_case_ids must contain at most {config.warmup_review_memory_limit} case ids."
    )
    user = json.dumps(
        {
            "objective": "rewrite the global lesson zone and the current teacher scope lesson zone after one single-teacher warmup batch using interval-rule schema only",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "scope_card": scope_card,
            "old_global_lesson_zone": list(old_global_lesson_zone),
            "old_scope_lesson_zone": list(old_scope_lesson_zone),
            "memory_budget": {
                "max_retained_cases": config.warmup_review_memory_limit,
                "preferred_cross_tier_coverage": ["must_do", "pass_positive", "neutral_optional", "avoid"],
                "max_cases_per_tier_fallback": config.warmup_retained_case_max_per_tier,
            },
            "interval_evidence": interval_evidence,
            "current_scope_review_area_cases": current_review_cases,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {}

    raw_global_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["global_lesson_zone", "global_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    global_lesson_zone_lines = _sanitize_global_lesson_zone_lines(
        candidate_lines=raw_global_lines,
        old_lines=old_global_lesson_zone,
        fallback_lines=_fallback_global_interval_lesson_lines(),
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    raw_scope_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["scope_lesson_zone", "lesson_zone", "scope_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    scope_lesson_zone_lines = _sanitize_interval_scope_lesson_lines(
        candidate_lines=raw_scope_lines,
        old_lines=old_scope_lesson_zone,
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    retained_case_ids = _parse_retained_case_ids(
        parsed_payload,
        allowed_case_ids=available_case_ids,
        limit=config.warmup_review_memory_limit,
    )
    if retained_case_ids:
        by_id = {
            str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
            for entry in candidate_entries
        }
        next_review_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    else:
        next_review_entries = _fallback_retained_review_entries(
            candidate_entries,
            limit=config.warmup_review_memory_limit,
            max_per_tier=config.warmup_retained_case_max_per_tier,
        )
        retained_case_ids = [
            str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in next_review_entries
        ]
    next_review_entries = _annotate_review_entry_tiers(next_review_entries)
    next_review_cards = _select_interval_review_cards_for_prompt(
        next_review_entries,
        feature_priority=feature_priority,
        limit=config.warmup_review_memory_limit,
    )
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "scope_round_id": scope_round_id,
        "source_round_id": source_round_id,
        "scope_card": scope_card,
        "feature_priority": feature_priority,
        "old_global_lesson_zone": list(old_global_lesson_zone),
        "old_scope_lesson_zone": list(old_scope_lesson_zone),
        "interval_evidence": interval_evidence,
        "candidate_review_case_count": len(candidate_entries),
        "current_review_area_count": len(current_review_cases),
        "latest_batch_case_count": len(latest_batch_cases),
        "retained_case_count": len(next_review_entries),
        "retained_case_ids": retained_case_ids,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "global_lesson_zone": global_lesson_zone_lines,
        "scope_lesson_zone": scope_lesson_zone_lines,
        "retained_review_cards": next_review_cards,
        "latest_batch_case_details": latest_batch_cases,
    }
    return global_lesson_zone_lines, scope_lesson_zone_lines, next_review_entries, artifact


def _rewrite_scoped_lesson_zone_iterative_v7_rulecard(
    *,
    config: ApprenticeReplayConfig,
    meta: Dict[str, Any],
    scope_round_id: str,
    source_round_id: str,
    retained_review_entries: Sequence[Dict[str, Any]],
    old_global_lesson_zone: Sequence[str],
    old_scope_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    scope_card = _scope_domain_card(meta, scope_round_id=scope_round_id, source_round_id=source_round_id)
    feature_priority = list(scope_card.get("top_features") or [])[: max(4, config.prompt_feature_count)]
    retained_entries = [dict(entry, case_origin="retained") for entry in retained_review_entries]
    latest_entries = [dict(entry, case_origin="latest_batch") for entry in batch_entries]
    candidate_entries = _annotate_review_entry_tiers([*retained_entries, *latest_entries])
    retained_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "retained"]
    latest_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "latest_batch"]
    available_case_ids = [str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in candidate_entries]
    interval_evidence = _scope_interval_evidence(review_entries=candidate_entries, feature_priority=feature_priority)
    current_review_cases = [
        _interval_retention_case_payload(entry, origin="retained", feature_priority=feature_priority)
        for entry in retained_entries
    ]
    latest_batch_cases = [
        _interval_retention_case_payload(entry, origin="latest_batch", feature_priority=feature_priority)
        for entry in latest_entries
    ]
    system = (
        "You are maintaining three bounded memories for QuantApprentice warmup: "
        "Global Lesson Zone, Current Teacher Scope Lesson Zone, and Current Scope Practice Review Area. "
        "This version uses structured rule-card lessons only. "
        "Global Lesson Zone should contain only cross-teacher meta rules. "
        "Current Teacher Scope Lesson Zone must be an ordered rule card for the active teacher family/template. "
        "Never mention dates, stock codes, candidate IDs, or pairwise examples. "
        "Never write 'overlap with teacher picks is weak'. "
        "Each scope lesson line must use one of these prefixes: CORE, VETO, TIE, DRIFT, CHECK, ABSTAIN. "
        "Each line must keep an ACTION tag and an explicit condition, ideally with feature bands or inequalities. "
        "Prefer a denser rule card instead of a thin summary; include enough distinct rules to cover entry preference, veto conditions, tie-break logic, drift warnings, and abstain conditions. "
        "Good examples: "
        "'CORE01: ACTION=upweight | prefer when dJ_3 in [-60,-35] and close_to_ma20 in [1.01,1.07].' "
        "'VETO01: ACTION=avoid | reject when volatility_20 >= 0.055 and volume_zscore_20 in [1.2,2.5].' "
        "'TIE01: ACTION=tiebreak | between survivors, prefer names closer to the teacher band for amt_zscore_20.' "
        "'ABSTAIN01: ACTION=abstain | abstain when fewer than two core rules match.' "
        "Practice Review Area cannot keep every case. "
        "Return strict JSON with keys: global_lesson_zone, scope_lesson_zone, retained_case_ids, revision_notes. "
        f"global_lesson_zone must be a list of 3 to {config.warmup_lesson_zone_max_lines} concise lines. "
        f"scope_lesson_zone must be a list of 10 to {config.warmup_lesson_zone_max_lines} concise rule-card lines. "
        f"retained_case_ids must contain at most {config.warmup_review_memory_limit} case ids."
    )
    user = json.dumps(
        {
            "objective": "rewrite the global lesson zone and the current teacher scope lesson zone after one single-teacher warmup batch using expanded rule-card schema",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "scope_card": scope_card,
            "old_global_lesson_zone": list(old_global_lesson_zone),
            "old_scope_lesson_zone": list(old_scope_lesson_zone),
            "required_rulecard_mix": {
                "target_prefixes": ["CORE", "VETO", "TIE", "DRIFT", "CHECK", "ABSTAIN"],
                "minimum_core_rules": 4,
                "minimum_veto_rules": 3,
                "minimum_tiebreak_rules": 2,
            },
            "memory_budget": {
                "max_retained_cases": config.warmup_review_memory_limit,
                "preferred_cross_tier_coverage": ["must_do", "pass_positive", "neutral_optional", "avoid"],
                "max_cases_per_tier_fallback": config.warmup_retained_case_max_per_tier,
            },
            "interval_evidence": interval_evidence,
            "current_scope_review_area_cases": current_review_cases,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {}

    raw_global_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["global_lesson_zone", "global_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    global_lesson_zone_lines = _sanitize_global_lesson_zone_lines(
        candidate_lines=raw_global_lines,
        old_lines=old_global_lesson_zone,
        fallback_lines=_fallback_global_rulecard_lines(),
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    raw_scope_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["scope_lesson_zone", "lesson_zone", "scope_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    scope_lesson_zone_lines = _sanitize_rulecard_scope_lesson_lines(
        candidate_lines=raw_scope_lines,
        old_lines=old_scope_lesson_zone,
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    retained_case_ids = _parse_retained_case_ids(
        parsed_payload,
        allowed_case_ids=available_case_ids,
        limit=config.warmup_review_memory_limit,
    )
    if retained_case_ids:
        by_id = {
            str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
            for entry in candidate_entries
        }
        next_review_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    else:
        next_review_entries = _fallback_retained_review_entries(
            candidate_entries,
            limit=config.warmup_review_memory_limit,
            max_per_tier=config.warmup_retained_case_max_per_tier,
        )
        retained_case_ids = [
            str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in next_review_entries
        ]
    next_review_entries = _annotate_review_entry_tiers(next_review_entries)
    next_review_cards = _select_interval_review_cards_for_prompt(
        next_review_entries,
        feature_priority=feature_priority,
        limit=config.warmup_review_memory_limit,
    )
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "scope_round_id": scope_round_id,
        "source_round_id": source_round_id,
        "scope_card": scope_card,
        "feature_priority": feature_priority,
        "old_global_lesson_zone": list(old_global_lesson_zone),
        "old_scope_lesson_zone": list(old_scope_lesson_zone),
        "interval_evidence": interval_evidence,
        "candidate_review_case_count": len(candidate_entries),
        "current_review_area_count": len(current_review_cases),
        "latest_batch_case_count": len(latest_batch_cases),
        "retained_case_count": len(next_review_entries),
        "retained_case_ids": retained_case_ids,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "global_lesson_zone": global_lesson_zone_lines,
        "scope_lesson_zone": scope_lesson_zone_lines,
        "retained_review_cards": next_review_cards,
        "latest_batch_case_details": latest_batch_cases,
    }
    return global_lesson_zone_lines, scope_lesson_zone_lines, next_review_entries, artifact


def _rewrite_scoped_lesson_zone_iterative_v8_weighted_soft_rules(
    *,
    config: ApprenticeReplayConfig,
    meta: Dict[str, Any],
    scope_round_id: str,
    source_round_id: str,
    retained_review_entries: Sequence[Dict[str, Any]],
    old_global_lesson_zone: Sequence[str],
    old_scope_lesson_zone: Sequence[str],
    batch_entries: Sequence[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    scope_card = _scope_domain_card(meta, scope_round_id=scope_round_id, source_round_id=source_round_id)
    feature_priority = list(scope_card.get("top_features") or [])[: max(4, config.prompt_feature_count)]
    theory_summary, theory_round_id = _resolve_explainability_summary(
        preferred_round_ids=[
            str(scope_card.get("explainability_round_id", "")).strip(),
            str(scope_round_id or "").strip(),
            str(meta.get("round_id", "")).strip(),
        ],
        fallback_summary=meta.get("factor_analysis_summary", {}),
    )
    theory_report_v2 = _teacher_explainability_payload(
        theory_summary,
        max_branches=4,
        max_soft_rules=8,
        max_veto_rules=5,
        max_meta_rules=5,
        max_trap_pairs=3,
    )
    theory_report_v2_lines = _compact_branch_rule_lines(
        theory_summary,
        max_branches=4,
        max_soft_rules=8,
        max_veto_rules=5,
        max_meta_rules=5,
        max_trap_pairs=3,
    )
    retained_entries = [dict(entry, case_origin="retained") for entry in retained_review_entries]
    latest_entries = [dict(entry, case_origin="latest_batch") for entry in batch_entries]
    candidate_entries = _annotate_review_entry_tiers([*retained_entries, *latest_entries])
    retained_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "retained"]
    latest_entries = [entry for entry in candidate_entries if str(entry.get("case_origin", "")) == "latest_batch"]
    available_case_ids = [str(entry.get("case_id", entry.get("decision_date", ""))).strip() for entry in candidate_entries]
    interval_evidence = _scope_interval_evidence(review_entries=candidate_entries, feature_priority=feature_priority)
    current_review_cases = [
        _interval_retention_case_payload(entry, origin="retained", feature_priority=feature_priority)
        for entry in retained_entries
    ]
    latest_batch_cases = [
        _interval_retention_case_payload(entry, origin="latest_batch", feature_priority=feature_priority)
        for entry in latest_entries
    ]
    system = (
        "You are maintaining three bounded memories for QuantApprentice warmup: "
        "Global Lesson Zone, Current Teacher Scope Lesson Zone, and Current Scope Practice Review Area. "
        "This version uses weighted soft-rule lessons only. "
        "Global Lesson Zone should contain only cross-teacher meta rules. "
        "Current Teacher Scope Lesson Zone must be an ordered weighted rule card for the active teacher family/template. "
        "You are also given a branch-oriented Report v2 theory pack for this teacher. "
        "Treat that report as the starting prior, then revise it using warmup evidence. "
        "Keep theory that survives practice, soften theory that overfires, and add lesson lines for repeated warmup drift patterns. "
        "Never mention dates, stock codes, candidate IDs, or pairwise examples. "
        "Never write 'overlap with teacher picks is weak'. "
        "Do not write rigid exact-threshold rules that read like hard if-else code. "
        "Use branch conditions and evidence-derived ranges as soft reference zones: near, inside, slightly above, slightly below, comfortably inside, or well outside. "
        "Each scope lesson line must use one of these prefixes: PREFER, VETO, META, CHECK, ABSTAIN. "
        "Every PREFER or VETO line must include TYPE and STRENGTH. "
        "TYPE choices: soft_preference, hard_veto, soft_veto. "
        "STRENGTH choices: strong, medium, weak, hard. "
        "META lines must explain conflict resolution between preferences and vetoes. "
        "Good examples: "
        "'PREFER01: TYPE=soft_preference | STRENGTH=strong | favor when dJ_3 stays inside or slightly beyond teacher band [-60,-35] and close_to_ma20 also stays near [1.01,1.07].' "
        "'VETO01: TYPE=hard_veto | STRENGTH=hard | reject when volatility_20 falls deep inside repeated llm-drift band [0.055,0.081] together with heavy volume expansion.' "
        "'META01: ACTION=conflict_resolution | if any hard veto triggers, it beats all soft preferences.' "
        "'ABSTAIN01: ACTION=abstain | abstain when no strong preference survives and the case depends on weak evidence only.' "
        "Prefer a balanced rule card that explains weighted ranking, hard vetoes, soft vetoes, and conflict handling. "
        "Practice Review Area cannot keep every case. "
        "Return strict JSON with keys: global_lesson_zone, scope_lesson_zone, retained_case_ids, revision_notes. "
        f"global_lesson_zone must be a list of 3 to {config.warmup_lesson_zone_max_lines} concise lines. "
        f"scope_lesson_zone must be a list of 10 to {config.warmup_lesson_zone_max_lines} concise weighted rule-card lines. "
        f"retained_case_ids must contain at most {config.warmup_review_memory_limit} case ids."
    )
    user = json.dumps(
        {
            "objective": "rewrite the global lesson zone and the current teacher scope lesson zone after one single-teacher warmup batch using weighted soft-rule schema",
            "batch_index": batch_index,
            "total_batches": total_batches,
            "scope_card": scope_card,
            "theory_round_id": theory_round_id or scope_round_id,
            "theory_report_v2": theory_report_v2,
            "theory_report_v2_compact_lines": theory_report_v2_lines,
            "old_global_lesson_zone": list(old_global_lesson_zone),
            "old_scope_lesson_zone": list(old_scope_lesson_zone),
            "required_rule_mix": {
                "target_prefixes": ["PREFER", "VETO", "META", "CHECK", "ABSTAIN"],
                "minimum_prefer_rules": 4,
                "minimum_veto_rules": 2,
                "minimum_meta_rules": 2,
                "minimum_abstain_or_check_rules": 2,
            },
            "style_constraints": {
                "soft_bands_not_exact_thresholds": True,
                "must_include_strength_tags": True,
                "must_distinguish_hard_veto_vs_soft_preference": True,
                "prefer_conflict_resolution_rules": True,
            },
            "memory_budget": {
                "max_retained_cases": config.warmup_review_memory_limit,
                "preferred_cross_tier_coverage": ["must_do", "pass_positive", "neutral_optional", "avoid"],
                "max_cases_per_tier_fallback": config.warmup_retained_case_max_per_tier,
            },
            "interval_evidence": interval_evidence,
            "current_scope_review_area_cases": current_review_cases,
            "latest_batch_cases": latest_batch_cases,
        },
        ensure_ascii=False,
    )
    response = _chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    try:
        parsed_payload = _extract_json_payload(content)
    except Exception:
        parsed_payload = {}

    raw_global_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["global_lesson_zone", "global_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    global_lesson_zone_lines = _sanitize_global_lesson_zone_lines(
        candidate_lines=raw_global_lines,
        old_lines=old_global_lesson_zone,
        fallback_lines=_fallback_global_weighted_soft_rule_lines(),
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    raw_scope_lines = _parse_named_lesson_lines(
        parsed_payload,
        keys=["scope_lesson_zone", "lesson_zone", "scope_lessons"],
        max_lines=config.warmup_lesson_zone_max_lines,
    )
    scope_lesson_zone_lines = _sanitize_weighted_soft_scope_lesson_lines(
        candidate_lines=raw_scope_lines,
        old_lines=old_scope_lesson_zone,
        feature_priority=feature_priority,
        interval_evidence=interval_evidence,
        max_lines=config.warmup_lesson_zone_max_lines,
    )

    retained_case_ids = _parse_retained_case_ids(
        parsed_payload,
        allowed_case_ids=available_case_ids,
        limit=config.warmup_review_memory_limit,
    )
    if retained_case_ids:
        by_id = {
            str(entry.get("case_id", entry.get("decision_date", ""))).strip(): entry
            for entry in candidate_entries
        }
        next_review_entries = [by_id[case_id] for case_id in retained_case_ids if case_id in by_id]
    else:
        next_review_entries = _fallback_retained_review_entries(
            candidate_entries,
            limit=config.warmup_review_memory_limit,
            max_per_tier=config.warmup_retained_case_max_per_tier,
        )
        retained_case_ids = [
            str(entry.get("case_id", entry.get("decision_date", ""))).strip()
            for entry in next_review_entries
        ]
    next_review_cards = _select_interval_review_cards_for_prompt(
        next_review_entries,
        feature_priority=feature_priority,
        limit=config.warmup_review_memory_limit,
    )
    artifact = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "scope_round_id": scope_round_id,
        "source_round_id": source_round_id,
        "scope_card": scope_card,
        "feature_priority": feature_priority,
        "old_global_lesson_zone": list(old_global_lesson_zone),
        "old_scope_lesson_zone": list(old_scope_lesson_zone),
        "interval_evidence": interval_evidence,
        "candidate_review_case_count": len(candidate_entries),
        "current_review_area_count": len(current_review_cases),
        "latest_batch_case_count": len(latest_batch_cases),
        "retained_case_count": len(next_review_entries),
        "retained_case_ids": retained_case_ids,
        "raw_response": content,
        "parsed_payload": parsed_payload,
        "global_lesson_zone": global_lesson_zone_lines,
        "scope_lesson_zone": scope_lesson_zone_lines,
        "retained_review_cards": next_review_cards,
        "latest_batch_case_details": latest_batch_cases,
    }
    return global_lesson_zone_lines, scope_lesson_zone_lines, next_review_entries, artifact


def _generate_warmup_lessons_iterative_v5_scoped(
    *,
    config: ApprenticeReplayConfig,
    warm_cfg: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
) -> Dict[str, Any]:
    warmup_report_dir = REPORT_ROOT / warm_cfg.run_id()
    warmup_report_dir.mkdir(parents=True, exist_ok=True)
    source_round_ids = list(warm_cfg.teacher_round_ids)
    final_round_ids = list(config.teacher_round_ids)
    if len(source_round_ids) != len(final_round_ids):
        raise ValueError("iterative_v5_scoped requires one source round per final teacher round")
    sample_counts = _split_count_evenly(config.warmup_sample_count, len(final_round_ids))
    global_lesson_zone_lines: List[str] = []
    teacher_scope_payloads: List[Dict[str, Any]] = []
    combined_batch_history: List[Dict[str, Any]] = []
    combined_review_entries: List[Dict[str, Any]] = []
    _progress(
        "iterative_v5_scoped warmup start "
        f"run_id={warm_cfg.run_id()} sample_count={config.warmup_sample_count} "
        f"teacher_scopes={len(final_round_ids)} batch_size={config.warmup_batch_size}"
    )
    for scope_index, (final_round_id, source_round_id, scope_sample_count) in enumerate(
        zip(final_round_ids, source_round_ids, sample_counts),
        start=1,
    ):
        scope_cfg = replace(
            warm_cfg,
            mode="single",
            teacher_round_ids=[source_round_id],
            negative_teacher_round_ids=[],
            candidate_source="baseline_signal",
            warmup_sample_count=0,
            run_tag=f"{warm_cfg.run_tag}_scope{scope_index:02d}_{source_round_id}",
        )
        scope_frame, scope_meta = _build_single_teacher_frame(scope_cfg, master_df)
        if config.summary_variant == "enriched_v2":
            scope_meta["preference_bands"] = _derive_preference_bands(
                round_id=source_round_id,
                master_df=master_df,
                feature_cols=scope_meta["top_prompt_features"],
                start_date=config.warmup_start_date,
                end_date=config.warmup_end_date,
            )
        scope_pool, scope_target = _single_teacher_target(scope_frame, scope_cfg)
        scope_domain = _scope_domain_card(scope_meta, scope_round_id=final_round_id, source_round_id=source_round_id)
        if scope_pool.empty or scope_target.empty or scope_sample_count <= 0:
            teacher_scope_payloads.append(
                {
                    **scope_domain,
                    "scope_index": scope_index,
                    "scope_sample_count": scope_sample_count,
                    "sampled_days": 0,
                    "scope_lesson_zone_lines": [],
                    "review_cards_for_prompt": [],
                    "retained_review_entry_count": 0,
                    "batch_history": [],
                }
            )
            continue
        sampled_dates = _sample_uniform_dates(
            scope_pool["signal_date"].tolist(),
            scope_sample_count,
            sample_seed=(config.sample_seed + scope_index) if int(config.sample_seed) > 0 else 0,
        )
        sampled_dates = sorted(pd.Timestamp(item) for item in sampled_dates)
        scope_pool = scope_pool[scope_pool["signal_date"].isin(sampled_dates)].copy()
        scope_target = scope_target[scope_target["signal_date"].isin(sampled_dates)].copy()
        lesson_feature_cols = _lesson_feature_columns(scope_pool)
        batch_size = max(1, int(config.warmup_batch_size))
        date_batches = [sampled_dates[idx : idx + batch_size] for idx in range(0, len(sampled_dates), batch_size)]
        retained_review_entries: List[Dict[str, Any]] = []
        scope_lesson_zone_lines: List[str] = []
        scope_batch_history: List[Dict[str, Any]] = []
        _progress(
            "iterative_v5_scoped teacher start "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} source_round={source_round_id} "
            f"sampled_days={len(sampled_dates)}"
        )
        for batch_index, batch_dates in enumerate(date_batches, start=1):
            _progress(
                "iterative_v5_scoped batch start "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"date_from={pd.Timestamp(batch_dates[0]).strftime('%Y-%m-%d')} "
                f"date_to={pd.Timestamp(batch_dates[-1]).strftime('%Y-%m-%d')}"
            )
            batch_cfg = replace(
                scope_cfg,
                run_tag=f"{warm_cfg.run_tag}_scope{scope_index:02d}_{source_round_id}_batch{batch_index:02d}of{len(date_batches)}",
            )
            batch_pool = scope_pool[scope_pool["signal_date"].isin(batch_dates)].copy()
            batch_target = scope_target[scope_target["signal_date"].isin(batch_dates)].copy()
            scope_prompt_state = {
                "global_lesson_zone_lines": list(global_lesson_zone_lines),
                "teacher_scopes": [
                    {
                        **scope_domain,
                        "scope_lesson_zone_lines": list(scope_lesson_zone_lines),
                        "review_cards_for_prompt": [
                            str(entry.get("review_card", "")).strip()
                            for entry in retained_review_entries
                        ][: config.warmup_review_memory_limit],
                    }
                ],
            }
            _run_replay(
                config=batch_cfg,
                candidate_pool_df=batch_pool,
                teacher_target_df=batch_target,
                teacher_full_df=batch_target.copy(),
                prompt_builder=_daily_prompt_single,
                prompt_builder_kwargs={
                    "meta": scope_meta,
                    "negative_metas": [],
                    "warmup_lessons": scope_lesson_zone_lines,
                    "warmup_review_cards": [
                        str(entry.get("review_card", "")).strip()
                        for entry in retained_review_entries
                    ][: config.warmup_review_memory_limit],
                    "scoped_warmup_state": scope_prompt_state,
                    "current_scope_round_id": final_round_id,
                },
            )
            batch_report_dir = REPORT_ROOT / batch_cfg.run_id()
            batch_entries = _extract_warmup_review_entries(
                report_dir=batch_report_dir,
                candidate_pool_df=batch_pool,
                teacher_target_df=batch_target,
                prompt_features=scope_meta["top_prompt_features"],
                lesson_feature_cols=lesson_feature_cols,
                teacher_round_ids=[final_round_id, source_round_id],
            )
            for entry in batch_entries:
                entry["scope_round_id"] = final_round_id
                entry["scope_source_round_id"] = source_round_id
            combined_review_entries.extend(batch_entries)
            global_lesson_zone_lines, scope_lesson_zone_lines, retained_review_entries, lesson_artifact = _rewrite_scoped_lesson_zone_iterative_v5(
                config=config,
                meta=scope_meta,
                scope_round_id=final_round_id,
                source_round_id=source_round_id,
                retained_review_entries=retained_review_entries,
                old_global_lesson_zone=global_lesson_zone_lines,
                old_scope_lesson_zone=scope_lesson_zone_lines,
                batch_entries=batch_entries,
                batch_index=batch_index,
                total_batches=len(date_batches),
            )
            batch_payload = {
                "scope_index": scope_index,
                "scope_round_id": final_round_id,
                "scope_source_round_id": source_round_id,
                "scope_domain": scope_domain,
                "batch_index": batch_index,
                "batch_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in batch_dates],
                "batch_report_dir": _relative(batch_report_dir),
                "batch_review_count": len(batch_entries),
                "retained_review_count": len(retained_review_entries),
                "global_lesson_zone": list(global_lesson_zone_lines),
                "scope_lesson_zone": list(scope_lesson_zone_lines),
                "lesson_artifact": lesson_artifact,
            }
            scope_batch_history.append(batch_payload)
            combined_batch_history.append(batch_payload)
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_state.json",
                batch_payload,
            )
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_review_entries.json",
                {"entries": batch_entries},
            )
            _progress(
                "iterative_v5_scoped batch done "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"retained_review={len(retained_review_entries)} global_lines={len(global_lesson_zone_lines)} "
                f"scope_lines={len(scope_lesson_zone_lines)}"
            )
        scope_payload = {
            **scope_domain,
            "scope_index": scope_index,
            "scope_sample_count": scope_sample_count,
            "sampled_days": len(sampled_dates),
            "scope_lesson_zone_lines": list(scope_lesson_zone_lines),
            "review_cards_for_prompt": [
                str(entry.get("review_card", "")).strip() for entry in retained_review_entries
            ][: config.warmup_review_memory_limit],
            "retained_review_entry_count": len(retained_review_entries),
            "batch_history": scope_batch_history,
        }
        teacher_scope_payloads.append(scope_payload)
        _write_json(warmup_report_dir / f"teacher_scope_{scope_index:02d}_{final_round_id}.json", scope_payload)
        _progress(
            "iterative_v5_scoped teacher done "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
            f"scope_lines={len(scope_lesson_zone_lines)} retained_review={len(retained_review_entries)}"
        )
    scoped_state = {
        "curriculum": "iterative_v5_scoped",
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": config.warmup_batch_size,
        "sample_seed": int(config.sample_seed),
        "global_lesson_zone_lines": list(global_lesson_zone_lines),
        "teacher_scope_order": list(final_round_ids),
        "teacher_scopes": teacher_scope_payloads,
        "review_cards_for_prompt": _select_review_cards_for_prompt(
            _annotate_review_entry_tiers(combined_review_entries),
            config.warmup_review_memory_limit,
        ),
        "batch_history": combined_batch_history,
        "artifact_dir": _relative(warmup_report_dir),
        "lesson_zone_lines": list(global_lesson_zone_lines),
    }
    _write_json(warmup_report_dir / "warmup_scoped_lessons.json", scoped_state)
    md_lines: List[str] = [
        "# Warmup Scoped Lessons",
        "",
        "## Global Lesson Zone",
        *(global_lesson_zone_lines or ["none"]),
        "",
        "## Teacher Scopes",
    ]
    for scope in teacher_scope_payloads:
        md_lines.extend(
            [
                f"### {scope.get('round_id')}",
                f"- family: {scope.get('family')}",
                f"- template: {scope.get('template')}",
                f"- style: {scope.get('style_hint')}",
                f"- basic_filter: {scope.get('basic_filter')}",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
            ]
        )
    (warmup_report_dir / "warmup_scoped_lessons.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scoped_state


def _scorefit_safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _scorefit_clipped_digest_text(text: str, max_chars: int = 3000) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _scorefit_sample_uniform_indices(total: int, count: int, *, sample_seed: int = 0) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    if int(sample_seed) <= 0:
        positions = np.linspace(0, total - 1, count)
        idxs = sorted({int(round(pos)) for pos in positions})
        while len(idxs) < count:
            for i in range(total):
                if i not in idxs:
                    idxs.append(i)
                if len(idxs) >= count:
                    break
        return sorted(idxs[:count])
    rng = np.random.default_rng(int(sample_seed))
    edges = np.linspace(0, total, count + 1)
    idxs: List[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = int(math.floor(left))
        hi = int(math.ceil(right)) - 1
        lo = max(0, min(lo, total - 1))
        hi = max(lo, min(hi, total - 1))
        if hi <= lo:
            idx = lo
        else:
            idx = int(rng.integers(lo, hi + 1))
        idxs.append(idx)
    idxs = sorted(dict.fromkeys(idxs))
    while len(idxs) < count:
        for i in range(total):
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= count:
                break
    return sorted(idxs[:count])


def _scorefit_signal_record_from_row(row: pd.Series, feature_cols: Sequence[str]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for col in feature_cols:
        value = row.get(col)
        if pd.isna(value):
            features[col] = None
        else:
            features[col] = round(float(value), 4)
    return {
        "symbol": str(row.get("symbol", "")).strip(),
        "signal_date": pd.Timestamp(row.get("signal_date")).strftime("%Y-%m-%d"),
        "entry_date": pd.Timestamp(row.get("entry_date")).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(row.get("exit_date")).strftime("%Y-%m-%d"),
        "features": features,
    }


def _scorefit_bucket_quota_map(x: int) -> Dict[int, int]:
    base = max(0, int(x)) // 5
    extra = max(0, int(x)) % 5
    quotas = {bucket: base for bucket in range(1, 6)}
    for bucket in range(1, extra + 1):
        quotas[bucket] += 1
    return quotas


def _scorefit_sample_day_signals(
    day_df: pd.DataFrame,
    *,
    x: int,
    sample_seed: int = 0,
) -> pd.DataFrame:
    if day_df.empty or x <= 0:
        return day_df.iloc[0:0].copy()
    parts: List[pd.DataFrame] = []
    quotas = _scorefit_bucket_quota_map(x)
    for bucket in range(1, 6):
        bucket_df = day_df[day_df["bucket"] == bucket].sort_values("score", ascending=False).reset_index(drop=True)
        if bucket_df.empty:
            continue
        keep = min(len(bucket_df), int(quotas.get(bucket, 0)))
        if keep <= 0:
            continue
        idxs = _scorefit_sample_uniform_indices(
            len(bucket_df),
            keep,
            sample_seed=(int(sample_seed) * 13 + bucket) if int(sample_seed) > 0 else 0,
        )
        picked = bucket_df.iloc[idxs].copy()
        picked["_teacher_bucket"] = int(bucket)
        parts.append(picked)
    if not parts:
        return day_df.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["bucket", "score"], ascending=[True, False]).reset_index(drop=True)
    out["_scorefit_candidate_id"] = [f"S{i:02d}" for i in range(1, len(out) + 1)]
    return out


def _scorefit_fallback_lesson_from_report(
    *,
    scope_domain: Mapping[str, Any],
    explainability_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    items: Dict[str, Dict[str, Any]] = {}
    next_index = 1
    strength_max = {"strong": 12, "medium": 8, "weak": 5}

    for branch in list(explainability_summary.get("branch_rule_cards") or explainability_summary.get("branch_cards") or [])[:4]:
        item_id = f"I{next_index:02d}"
        next_index += 1
        strength = str(branch.get("strength", "medium")).strip().lower() or "medium"
        items[item_id] = {
            "title": str(branch.get("branch_id", item_id)).strip() or item_id,
            "role": "branch_bonus",
            "score_range": f"0..{strength_max.get(strength, 8)}",
            "signals_to_check": [
                str((branch.get("anchor_pair") or {}).get("left_feature", "")).strip(),
                str((branch.get("anchor_pair") or {}).get("right_feature", "")).strip(),
            ],
            "rule": str(branch.get("branch_logic", "")).strip(),
            "interaction_note": str(branch.get("partial_alignment_note", "")).strip(),
        }
    for rule in list(explainability_summary.get("soft_rules") or [])[:10]:
        item_id = f"I{next_index:02d}"
        next_index += 1
        strength = str(rule.get("strength", "medium")).strip().lower() or "medium"
        items[item_id] = {
            "title": str(rule.get("rule_id", item_id)).strip() or item_id,
            "role": "soft_preference",
            "score_range": f"0..{strength_max.get(strength, 8)}",
            "signals_to_check": [str(rule.get("feature", "")).strip()],
            "rule": (
                f"Prefer {rule.get('feature')} {rule.get('direction')} near {rule.get('band')}. "
                f"{str(rule.get('usage_note', '')).strip()}"
            ).strip(),
            "interaction_note": str(rule.get("shape_hint", "")).strip(),
        }
    for veto in list(explainability_summary.get("hard_veto_rules") or [])[:4]:
        item_id = f"I{next_index:02d}"
        next_index += 1
        items[item_id] = {
            "title": str(veto.get("rule_id", item_id)).strip() or item_id,
            "role": "hard_veto",
            "score_range": "-12..0",
            "signals_to_check": [],
            "rule": str(veto.get("trigger", "")).strip(),
            "interaction_note": str(veto.get("reason", "")).strip(),
        }
    if not items:
        for idx, row in enumerate(list(explainability_summary.get("top_global_features") or [])[:8], start=1):
            item_id = f"I{idx:02d}"
            items[item_id] = {
                "title": str(row.get("feature", item_id)).strip() or item_id,
                "role": "soft_preference",
                "score_range": "0..8",
                "signals_to_check": [str(row.get("feature", "")).strip()],
                "rule": (
                    f"Prefer {row.get('feature')} {row.get('preferred_direction')} near "
                    f"[{_format_float(row.get('selected_q25'))},{_format_float(row.get('selected_q75'))}]"
                ),
                "interaction_note": str(row.get("shape_hint", "")).strip(),
            }
    return {
        "schema": "scorefit_v1_json",
        "lesson_name": "ScoreFit-v1 fallback lesson",
        "teacher_scope": dict(scope_domain),
        "items": items,
        "meta_rules": [
            str(item.get("guidance", "")).strip()
            for item in list(explainability_summary.get("meta_rules") or [])[:4]
            if str(item.get("guidance", "")).strip()
        ],
        "scoring_notes": [
            "Target the hidden teacher ranking, not the realized return of this single sample.",
            "Score approximately inside useful zones; do not overfit tiny conflicts.",
            "Let branch alignment add score and hard vetoes subtract score.",
        ],
    }


def _scorefit_variant_settings(config: ApprenticeReplayConfig) -> Dict[str, Any]:
    variant = str(getattr(config, "scorefit_variant", "v1") or "v1").strip().lower()
    settings: Dict[str, Any] = {
        "name": variant,
        "compact_items": False,
        "max_items": 18,
        "dedupe_signal_sets": False,
        "lesson0_extra_instruction": "",
        "revise_extra_instruction": "",
        "include_recent_batch_context": False,
        "suppress_revise_history_prompt": False,
        "best_checkpoint_guidance": False,
        "best_checkpoint_objective_lambda": 0.5,
        "best_checkpoint_objective_method": "raw_linear",
    }
    if variant == "v2_schemafix":
        settings["lesson0_extra_instruction"] = (
            "Keep the schema stable and machine-readable. "
            "Do not return prose outside JSON. "
            "Always populate meta_rules and scoring_notes as JSON arrays, never as a single string."
        )
        settings["revise_extra_instruction"] = (
            "Always return the full lesson object even if you keep most items unchanged. "
            "Never return a summary without lesson.items. "
            "Meta rules and scoring notes must stay as JSON arrays."
        )
    elif variant == "v3_tailaware":
        settings["lesson0_extra_instruction"] = (
            "Prefer branch-sensitive weighted rules over flat average scoring. "
            "Use score gaps large enough to separate teacher-Q5 style tail winners from mediocre middle cases."
        )
        settings["revise_extra_instruction"] = (
            "Primary objective is Spearman, but among similar Spearman candidates prefer lessons that improve top-bucket separation. "
            "Use penalties or vetoes when needed so weak middle-quality signals do not cluster near high-conviction signals."
        )
    elif variant == "v4_compact_tailaware":
        settings["compact_items"] = True
        settings["max_items"] = 10
        settings["dedupe_signal_sets"] = True
        settings["lesson0_extra_instruction"] = (
            "Build a compact scorecard with 8 to 10 items. "
            "Prefer 2 to 4 branch or interaction items, 2 to 4 soft preferences, and 1 to 2 veto or penalty items. "
            "Avoid repeated single-feature restatements and avoid duplicate feature pairs."
        )
        settings["revise_extra_instruction"] = (
            "Keep the lesson compact and branch-oriented. "
            "Primary objective is Spearman, but among similar Spearman candidates prefer better Q5 separation. "
            "Keep at most 10 items, avoid duplicate feature pairs, and avoid spending many items on the same single feature."
        )
    elif variant == "v5_stability_guard":
        settings["compact_items"] = True
        settings["max_items"] = 10
        settings["dedupe_signal_sets"] = True
        settings["include_recent_batch_context"] = True
        settings["lesson0_extra_instruction"] = (
            "Build a compact scorecard with 8 to 10 items. "
            "Prefer 2 to 4 branch or interaction items, 2 to 4 soft preferences, and 1 to 2 veto or penalty items. "
            "Avoid repeated single-feature restatements and avoid duplicate feature pairs."
        )
        settings["revise_extra_instruction"] = (
            "Treat one warmup batch as noisy evidence. "
            "Default under uncertainty is to keep the current lesson mostly unchanged. "
            "Prefer editing only 1 to 3 items per batch unless repeated evidence clearly supports a larger rewrite. "
            "Use recent_batch_context to distinguish stable signals from one-batch noise. "
            "Do not rewrite anchor items just because of one weak batch if their recent average ablation signal remains useful. "
            "When evidence is mixed, make small weight or range nudges instead of replacing the whole scorecard. "
            "Primary objective is Spearman, but among similar Spearman candidates prefer better Q5 separation. "
            "Keep at most 10 items, avoid duplicate feature pairs, and avoid spending many items on the same single feature."
        )
    elif variant == "v6_bestguard_explore":
        settings["compact_items"] = True
        settings["max_items"] = 10
        settings["dedupe_signal_sets"] = True
        settings["suppress_revise_history_prompt"] = True
        settings["best_checkpoint_guidance"] = True
        settings["best_checkpoint_objective_lambda"] = 1.0
        settings["best_checkpoint_objective_method"] = "raw_linear"
        settings["lesson0_extra_instruction"] = (
            "Build a compact scorecard with 8 to 10 items. "
            "Prefer a mix of branch items, interaction items, soft preferences, and a small number of veto or penalty items. "
            "Avoid repeated single-feature restatements and avoid duplicate feature pairs."
        )
        settings["revise_extra_instruction"] = (
            "Treat each warmup batch as noisy evidence for lesson-space exploration, not as an exact gradient step. "
            "First objective is to improve both teacher_score_spearman and llm_uplift_vs_not_selected together. "
            "Use current_batch item ablations to understand which lesson items helped or hurt, but do not assume one batch gives a perfect direction. "
            "Use current_best_checkpoint as a guardrail: try not to produce a candidate that is plausibly worse than the current best lesson on the hidden objective. "
            "When evidence is ambiguous, keep the best structural backbone and make a small coherent exploration change instead of rewriting the whole scorecard. "
            "You may borrow good structure from current_best_checkpoint even if current_lesson underperformed. "
            "Keep at most 10 items, avoid duplicate feature pairs, and avoid spending many items on the same single feature."
        )
    elif variant == "v7_bestguard_explore_longbatch":
        settings["compact_items"] = True
        settings["max_items"] = 10
        settings["dedupe_signal_sets"] = True
        settings["suppress_revise_history_prompt"] = True
        settings["best_checkpoint_guidance"] = True
        settings["best_checkpoint_objective_lambda"] = 1.0
        settings["best_checkpoint_objective_method"] = "zscore_sum"
        settings["lesson0_extra_instruction"] = (
            "Build a compact scorecard with 8 to 10 items. "
            "Prefer a mix of branch items, interaction items, soft preferences, and a small number of veto or penalty items. "
            "Avoid repeated single-feature restatements and avoid duplicate feature pairs."
        )
        settings["revise_extra_instruction"] = (
            "Treat each warmup batch as noisy evidence for lesson-space exploration, not as an exact gradient step. "
            "First objective is to improve both teacher_score_spearman and llm_uplift_vs_not_selected together. "
            "Use current_batch item ablations to understand which lesson items helped or hurt, but do not assume one batch gives a perfect direction. "
            "Use current_best_checkpoint as a guardrail, where the ranking is based on normalized z_spearman + z_uplift across explored lessons in this scope. "
            "Prefer exploring from the current z-score-best lesson backbone instead of blindly trusting the latest lesson. "
            "When evidence is ambiguous, keep the best structural backbone and make a small coherent exploration change instead of rewriting the whole scorecard. "
            "You may borrow good structure from current_best_checkpoint even if current_lesson underperformed. "
            "Keep at most 10 items, avoid duplicate feature pairs, and avoid spending many items on the same single feature."
        )
    return settings


def _scorefit_string_list(value: Any, *, limit: int) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        if str(value.get("guidance", "")).strip():
            return [str(value.get("guidance", "")).strip()]
        return [json.dumps(value, ensure_ascii=False)]
    output: List[str] = []
    for item in list(value):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            text = str(item.get("guidance", "")).strip() or json.dumps(item, ensure_ascii=False)
        else:
            text = str(item).strip()
        if text:
            output.append(text)
        if len(output) >= limit:
            break
    return output[:limit]


def _scorefit_item_role_priority(role: str) -> int:
    role_text = str(role or "").strip().lower()
    if "primary" in role_text or "anchor" in role_text:
        return 0
    if "secondary" in role_text or "support" in role_text:
        return 1
    if "veto" in role_text or "penalty" in role_text:
        return 2
    if "interaction" in role_text or "synergy" in role_text:
        return 3
    if "soft" in role_text or "preference" in role_text:
        return 4
    if "context" in role_text or "guidance" in role_text or "meta" in role_text:
        return 5
    return 6


def _scorefit_postprocess_lesson_payload(
    lesson_payload: Mapping[str, Any],
    *,
    config: ApprenticeReplayConfig,
) -> Dict[str, Any]:
    settings = _scorefit_variant_settings(config)
    items = dict(lesson_payload.get("items") or {})
    if settings.get("dedupe_signal_sets"):
        kept: Dict[str, Dict[str, Any]] = {}
        seen_pairs: set[Tuple[str, ...]] = set()
        ordered_ids = sorted(
            items.keys(),
            key=lambda item_id: (
                _scorefit_item_role_priority((items.get(item_id) or {}).get("role", "")),
                item_id,
            ),
        )
        for item_id in ordered_ids:
            item = dict(items.get(item_id) or {})
            signal_set = tuple(sorted(str(value).strip() for value in list(item.get("signals_to_check") or []) if str(value).strip()))
            if signal_set and signal_set in seen_pairs:
                continue
            kept[item_id] = item
            if signal_set:
                seen_pairs.add(signal_set)
        items = kept
    if settings.get("compact_items") and len(items) > int(settings.get("max_items", 10)):
        ordered_ids = sorted(
            items.keys(),
            key=lambda item_id: (
                _scorefit_item_role_priority((items.get(item_id) or {}).get("role", "")),
                item_id,
            ),
        )
        keep_ids = ordered_ids[: int(settings.get("max_items", 10))]
        items = {item_id: items[item_id] for item_id in keep_ids}
    return {
        "schema": "scorefit_v1_json",
        "lesson_name": str(lesson_payload.get("lesson_name", "ScoreFit-v1 lesson")).strip() or "ScoreFit-v1 lesson",
        "teacher_scope": dict(lesson_payload.get("teacher_scope") or {}),
        "items": items,
        "meta_rules": _scorefit_string_list(lesson_payload.get("meta_rules"), limit=8),
        "scoring_notes": _scorefit_string_list(
            lesson_payload.get("scoring_notes") or lesson_payload.get("usage_notes"),
            limit=8,
        ),
    }


def _scorefit_normalize_lesson_payload(
    payload: Mapping[str, Any],
    *,
    scope_domain: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_items = payload.get("items") or payload.get("score_items") or {}
    items: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_items, Mapping):
        iterable = raw_items.items()
    elif isinstance(raw_items, list):
        iterable = []
        for idx, item in enumerate(raw_items, start=1):
            if isinstance(item, Mapping):
                iterable.append((str(item.get("item_id", f"I{idx:02d}")).strip(), item))
    else:
        iterable = []
    for item_id, item in iterable:
        if not str(item_id).strip() or not isinstance(item, Mapping):
            continue
        clean_id = str(item_id).strip()
        items[clean_id] = {
            "title": str(item.get("title", clean_id)).strip() or clean_id,
            "role": str(item.get("role", "soft_preference")).strip() or "soft_preference",
            "score_range": str(item.get("score_range", "0..8")).strip() or "0..8",
            "signals_to_check": [
                str(value).strip()
                for value in list(item.get("signals_to_check") or item.get("features") or [])
                if str(value).strip()
            ][:8],
            "rule": str(item.get("rule", item.get("scoring_rule", ""))).strip(),
            "interaction_note": str(item.get("interaction_note", item.get("why", ""))).strip(),
        }
    if not items:
        raise ValueError("lesson payload missing usable items")
    return {
        "schema": "scorefit_v1_json",
        "lesson_name": str(payload.get("lesson_name", "ScoreFit-v1 lesson")).strip() or "ScoreFit-v1 lesson",
        "teacher_scope": dict(payload.get("teacher_scope") or dict(scope_domain)),
        "items": items,
        "meta_rules": _scorefit_string_list(payload.get("meta_rules"), limit=8),
        "scoring_notes": _scorefit_string_list(
            payload.get("scoring_notes") or payload.get("usage_notes"),
            limit=8,
        ),
    }


def _scorefit_render_lesson_lines(lesson_payload: Mapping[str, Any], *, limit: int) -> List[str]:
    items = lesson_payload.get("items") or {}
    if not isinstance(items, Mapping):
        return []
    lines: List[str] = []
    for item_id, item in list(items.items())[:limit]:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("title", item_id)).strip() or str(item_id)
        role = str(item.get("role", "")).strip()
        score_range = str(item.get("score_range", "")).strip()
        rule = " ".join(str(item.get("rule", "")).split())
        if len(rule) > 140:
            rule = rule[:137].rstrip() + "..."
        lines.append(f"{item_id} {role} {score_range}: {title}; {rule}")
    return lines[:limit]


def _scorefit_render_history_cards(
    revise_history: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> List[str]:
    cards: List[str] = []
    for row in list(revise_history)[-limit:]:
        batch_index = row.get("batch_index")
        spearman = row.get("spearman_before")
        summary = str(row.get("summary", "")).strip()
        cards.append(
            f"batch={batch_index} spearman_before={_format_float(spearman)} {summary}".strip()
        )
    return cards[-limit:]


def _scorefit_build_lesson0(
    *,
    config: ApprenticeReplayConfig,
    scope_domain: Mapping[str, Any],
    explainability_summary: Mapping[str, Any],
    background_digest: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    variant_settings = _scorefit_variant_settings(config)
    theory_payload = _teacher_explainability_payload(
        explainability_summary,
        max_branches=4,
        max_soft_rules=10,
        max_veto_rules=4,
        max_meta_rules=4,
        max_trap_pairs=3,
    )
    fallback = _scorefit_fallback_lesson_from_report(
        scope_domain=scope_domain,
        explainability_summary=theory_payload,
    )
    system = (
        "You are preparing lesson_0 for a trader imitation task. "
        "Your job is to convert a branch-oriented teacher explainability pack into a structured JSON scorecard. "
        "The scorecard will later be used to score one signal at a time and maximize Spearman correlation to the hidden teacher score. "
        "Return strict JSON only with keys: schema, lesson_name, teacher_scope, items, meta_rules, scoring_notes. "
        "items must be an object keyed by item id like I01, I02. "
        "Each item value must include: title, role, score_range, signals_to_check, rule, interaction_note. "
        "Use around 10 to 18 items. Allow soft preferences, branch bonuses, and hard veto penalties. "
        "Represent scoring as weighted sub-items instead of plain yes/no rules. "
        "If multiple lesson structures are similarly plausible, use sampling_seed only as a deterministic tie-breaker "
        "to choose one coherent lesson variant; do not mention the seed in the lesson itself. "
        f"{variant_settings.get('lesson0_extra_instruction', '')}"
    )
    user = json.dumps(
        {
            "teacher_scope": dict(scope_domain),
            "sampling_seed": int(config.sample_seed or 0),
            "research_background_digest": _scorefit_clipped_digest_text(background_digest, max_chars=2500),
            "report_v2": theory_payload,
        },
        ensure_ascii=False,
    )
    try:
        response = _chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            api_key=_apprentice_api_key(),
            model=config.api_model,
            max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
            temperature=0.0,
            force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
            fail_fast_on_empty_content=True,
            seed=int(config.sample_seed or 0) or None,
        )
        content = str(response["choices"][0]["message"].get("content", "") or "")
        parsed = _extract_json_payload(content)
        lesson = _scorefit_postprocess_lesson_payload(
            _scorefit_normalize_lesson_payload(parsed, scope_domain=scope_domain),
            config=config,
        )
        artifact = {
            "mode": "model_generated",
            "raw_response": content,
            "report_v2": theory_payload,
        }
        return lesson, artifact
    except Exception as exc:
        artifact = {
            "mode": "fallback_from_report_v2",
            "fallback_reason": str(exc),
            "report_v2": theory_payload,
        }
        return _scorefit_postprocess_lesson_payload(fallback, config=config), artifact


def _scorefit_signal_prompt(
    *,
    config: ApprenticeReplayConfig,
    scope_domain: Mapping[str, Any],
    lesson_payload: Mapping[str, Any],
    signal_record: Mapping[str, Any],
) -> Tuple[str, str]:
    reasoning_instruction = _compact_reasoning_instruction(config)
    item_ids = list((lesson_payload.get("items") or {}).keys())
    system = (
        "You are scoring one candidate signal to imitate a hidden tree-based teacher. "
        "The objective is teacher-score alignment, not short-term hindsight on this one sample. "
        f"{reasoning_instruction} "
        "Use the lesson JSON as the scoring rubric. "
        "Return strict JSON only with keys: total_score, subscores, short_reason. "
        "subscores must be an object containing every lesson item id exactly once. "
        "Each subscore must be numeric and can be positive or negative if the lesson item range allows it. "
        "Do not output markdown."
    )
    user = json.dumps(
        {
            "teacher_scope": {
                "round_id": scope_domain.get("round_id"),
                "source_round_id": scope_domain.get("source_round_id"),
                "family": scope_domain.get("family"),
                "template": scope_domain.get("template"),
                "style_hint": scope_domain.get("style_hint"),
                "basic_filter": scope_domain.get("basic_filter"),
            },
            "required_item_ids": item_ids,
            "lesson": lesson_payload,
            "signal": signal_record,
        },
        ensure_ascii=False,
    )
    return system, user


def _scorefit_parse_signal_reply(
    *,
    content: str,
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    try:
        payload = _extract_json_payload(content)
    except Exception as exc:
        return {
            "parse_failed": True,
            "failure_reason": f"scorefit_json_parse_error={exc}",
            "total_score": 0.0,
            "subscores": {item_id: 0.0 for item_id in item_ids},
            "short_reason": "",
        }
    raw_subscores = payload.get("subscores") or payload.get("item_scores") or {}
    if not isinstance(raw_subscores, Mapping):
        raw_subscores = {}
    subscores = {
        str(item_id): _scorefit_safe_float(raw_subscores.get(item_id, 0.0), 0.0)
        for item_id in item_ids
    }
    total_score = payload.get("total_score", payload.get("score"))
    if total_score is None:
        total_value = float(sum(subscores.values()))
    else:
        total_value = _scorefit_safe_float(total_score, float(sum(subscores.values())))
    return {
        "parse_failed": False,
        "failure_reason": "",
        "total_score": float(total_value),
        "subscores": subscores,
        "short_reason": str(payload.get("short_reason", payload.get("reason", ""))).strip(),
        "parsed_payload": payload,
    }


def _legacy_text_signal_prompt(
    *,
    config: ApprenticeReplayConfig,
    scope_domain: Mapping[str, Any],
    lesson_payload: Mapping[str, Any],
    signal_record: Mapping[str, Any],
) -> Tuple[str, str]:
    reasoning_instruction = _compact_reasoning_instruction(config)
    system = (
        "You are scoring one candidate signal to imitate a hidden tree-based teacher. "
        "The objective is teacher-score alignment and stable ranking, not hindsight on this one sample. "
        f"{reasoning_instruction} "
        "You are given a legacy warmup lesson written as text rules distilled from a small number of practice cases. "
        "Some lesson lines may be exemplar-like, fuzzy, or noisy; infer the broader preference pattern instead of copying dates or symbols literally. "
        "Use a 0-100 scale where 50 is neutral, higher means more teacher-like, and lower means less teacher-like. "
        "Return strict JSON only with keys: total_score, short_reason. "
        "Do not output markdown."
    )
    user = json.dumps(
        {
            "teacher_scope": {
                "round_id": scope_domain.get("round_id"),
                "source_round_id": scope_domain.get("source_round_id"),
                "family": scope_domain.get("family"),
                "template": scope_domain.get("template"),
                "style_hint": scope_domain.get("style_hint"),
                "basic_filter": scope_domain.get("basic_filter"),
            },
            "legacy_curriculum": str(lesson_payload.get("legacy_curriculum", "legacy_text")).strip(),
            "global_lesson_zone": list(lesson_payload.get("global_lesson_zone_lines") or []),
            "teacher_scope_lesson_zone": list(lesson_payload.get("scope_lesson_zone_lines") or []),
            "teacher_scope_review_memory": list(lesson_payload.get("review_cards_for_prompt") or []),
            "signal": signal_record,
        },
        ensure_ascii=False,
    )
    return system, user


def _legacy_text_parse_signal_reply(
    *,
    content: str,
) -> Dict[str, Any]:
    try:
        payload = _extract_json_payload(content)
    except Exception as exc:
        return {
            "parse_failed": True,
            "failure_reason": f"legacy_text_json_parse_error={exc}",
            "total_score": 0.0,
            "short_reason": "",
            "parsed_payload": {},
        }
    total_value = _scorefit_safe_float(payload.get("total_score", payload.get("score")), 0.0)
    return {
        "parse_failed": False,
        "failure_reason": "",
        "total_score": float(total_value),
        "short_reason": str(payload.get("short_reason", payload.get("reason", ""))).strip(),
        "parsed_payload": payload,
    }


def _legacy_text_request_one_signal(
    *,
    task: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    lesson_payload: Mapping[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    system, user = _legacy_text_signal_prompt(
        config=config,
        scope_domain=task["scope_domain"],
        lesson_payload=lesson_payload,
        signal_record=task["signal_record"],
    )
    request_payload = {
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": min(config.api_max_tokens, config.warmup_signal_score_max_tokens),
        "temperature": config.api_temperature,
    }
    cache_path = Path(str(task["cache_path"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    api_calls = 0
    api_cache_hits = 0
    response_payload = None
    content = ""
    failure_reason = ""
    try:
        if bool(task.get("reuse_api_cache", True)) and cache_path.exists():
            cached = _load_json(cache_path)
            response_payload = cached.get("response")
            api_cache_hits += 1
        else:
            response_payload = _chat_completion(
                messages=request_payload["messages"],
                api_key=api_key,
                model=config.api_model,
                max_tokens=request_payload["max_tokens"],
                temperature=config.api_temperature,
                force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=True,
                max_retries=1,
            )
            api_calls += 1
            _write_json(cache_path, {"request": request_payload, "response": response_payload})
        content = str(response_payload["choices"][0]["message"].get("content", "") or "") if response_payload else ""
        parsed = _legacy_text_parse_signal_reply(content=content)
        failure_reason = str(parsed.get("failure_reason", "")).strip()
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": bool(parsed.get("parse_failed", False)),
            "failure_reason": failure_reason,
            "total_score": float(parsed.get("total_score", 0.0)),
            "subscores": {},
            "short_reason": str(parsed.get("short_reason", "")).strip(),
            "parsed_payload": parsed.get("parsed_payload", {}),
        }
    except Exception as exc:
        failure_reason = str(exc)
        _write_json(
            cache_path,
            {
                "request": request_payload,
                "error": failure_reason,
            },
        )
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": True,
            "failure_reason": failure_reason,
            "total_score": 0.0,
            "subscores": {},
            "short_reason": "",
            "parsed_payload": {},
        }


def _legacy_text_request_signal_batch(
    *,
    task_batch: Sequence[Mapping[str, Any]],
    workers: int,
    config: ApprenticeReplayConfig,
    lesson_payload: Mapping[str, Any],
    api_key: str,
    global_batch_index: int,
) -> List[Dict[str, Any]]:
    if not task_batch:
        return []
    results: List[Dict[str, Any]] = []
    future_map = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for task in task_batch:
            future = executor.submit(
                _legacy_text_request_one_signal,
                task=task,
                config=config,
                lesson_payload=lesson_payload,
                api_key=api_key,
            )
            future_map[future] = task
        for future in as_completed(future_map):
            result = future.result()
            result["parallel_batch_index"] = int(global_batch_index)
            results.append(result)
    results.sort(key=lambda item: (str(item.get("signal_date")), str(item.get("task_key"))))
    return results


def _scorefit_request_one_signal(
    *,
    task: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    lesson_payload: Mapping[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    system, user = _scorefit_signal_prompt(
        config=config,
        scope_domain=task["scope_domain"],
        lesson_payload=lesson_payload,
        signal_record=task["signal_record"],
    )
    request_payload = {
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": min(config.api_max_tokens, config.warmup_signal_score_max_tokens),
        "temperature": config.api_temperature,
    }
    cache_path = Path(str(task["cache_path"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    api_calls = 0
    api_cache_hits = 0
    response_payload = None
    content = ""
    failure_reason = ""
    try:
        if bool(task.get("reuse_api_cache", True)) and cache_path.exists():
            cached = _load_json(cache_path)
            response_payload = cached.get("response")
            api_cache_hits += 1
        else:
            response_payload = _chat_completion(
                messages=request_payload["messages"],
                api_key=api_key,
                model=config.api_model,
                max_tokens=request_payload["max_tokens"],
                temperature=config.api_temperature,
                force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=True,
                max_retries=1,
            )
            api_calls += 1
            _write_json(cache_path, {"request": request_payload, "response": response_payload})
        content = str(response_payload["choices"][0]["message"].get("content", "") or "") if response_payload else ""
        parsed = _scorefit_parse_signal_reply(
            content=content,
            item_ids=list((lesson_payload.get("items") or {}).keys()),
        )
        failure_reason = str(parsed.get("failure_reason", "")).strip()
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": bool(parsed.get("parse_failed", False)),
            "failure_reason": failure_reason,
            "total_score": float(parsed.get("total_score", 0.0)),
            "subscores": dict(parsed.get("subscores", {})),
            "short_reason": str(parsed.get("short_reason", "")).strip(),
            "parsed_payload": parsed.get("parsed_payload", {}),
        }
    except Exception as exc:
        failure_reason = str(exc)
        _write_json(
            cache_path,
            {
                "request": request_payload,
                "error": failure_reason,
            },
        )
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": True,
            "failure_reason": failure_reason,
            "total_score": 0.0,
            "subscores": {
                str(item_id): 0.0 for item_id in list((lesson_payload.get("items") or {}).keys())
            },
            "short_reason": "",
            "parsed_payload": {},
        }


def _scorefit_request_signal_batch(
    *,
    task_batch: Sequence[Mapping[str, Any]],
    workers: int,
    config: ApprenticeReplayConfig,
    lesson_payload: Mapping[str, Any],
    api_key: str,
    global_batch_index: int,
) -> List[Dict[str, Any]]:
    if not task_batch:
        return []
    results: List[Dict[str, Any]] = []
    future_map = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for task in task_batch:
            future = executor.submit(
                _scorefit_request_one_signal,
                task=task,
                config=config,
                lesson_payload=lesson_payload,
                api_key=api_key,
            )
            future_map[future] = task
        for future in as_completed(future_map):
            result = future.result()
            result["parallel_batch_index"] = int(global_batch_index)
            results.append(result)
    results.sort(key=lambda item: (str(item.get("signal_date")), str(item.get("task_key"))))
    return results


def _scorefit_corr_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_s = pd.Series(list(left), dtype=float)
    right_s = pd.Series(list(right), dtype=float)
    value = left_s.corr(right_s, method="spearman")
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _scorefit_batch_metrics(
    *,
    results: Sequence[Mapping[str, Any]],
    lesson_payload: Mapping[str, Any],
    lock_days: int,
) -> Dict[str, Any]:
    usable = [dict(row) for row in results if not bool(row.get("parse_fallback", False))]
    usable.sort(key=lambda item: (str(item.get("signal_date")), str(item.get("task_key"))))
    item_ids = list((lesson_payload.get("items") or {}).keys())
    teacher_scores = [_scorefit_safe_float(row.get("teacher_score"), 0.0) for row in usable]
    llm_scores = [_scorefit_safe_float(row.get("total_score"), 0.0) for row in usable]
    base_spearman = _scorefit_corr_spearman(llm_scores, teacher_scores)

    total_success = len(usable)
    top_k = max(1, int(round(total_success * 0.2))) if total_success > 0 else 0
    top_selected = sorted(
        usable,
        key=lambda item: (
            -_scorefit_safe_float(item.get("total_score"), 0.0),
            str(item.get("signal_date")),
            str(item.get("symbol")),
        ),
    )[:top_k]
    selected_keys = {str(item.get("task_key")) for item in top_selected}
    not_selected = [item for item in usable if str(item.get("task_key")) not in selected_keys]
    teacher_q5_rows = [item for item in usable if int(item.get("teacher_bucket", 0) or 0) == 5]
    teacher_not_selected_rows = [item for item in usable if int(item.get("teacher_bucket", 0) or 0) != 5]

    all_days = sorted({str(item.get("signal_date")) for item in usable})
    daily_returns: List[Dict[str, Any]] = []
    nav = 1.0
    capital_frac = 1.0 / max(1, int(lock_days))
    for day_text in all_days:
        day_selected = [
            _scorefit_safe_float(item.get("future_return_5d"), 0.0)
            for item in usable
            if str(item.get("signal_date")) == day_text and str(item.get("task_key")) in selected_keys
        ]
        day_return = float(np.mean(day_selected)) if day_selected else 0.0
        nav *= 1.0 + capital_frac * day_return
        daily_returns.append(
            {
                "signal_date": day_text,
                "selected_count": len(day_selected),
                "day_mean_return": day_return,
                "nav_after_close": nav,
            }
        )

    teacher_bucket_stats: List[Dict[str, Any]] = []
    selected_bucket_stats: List[Dict[str, Any]] = []
    for bucket in range(1, 6):
        bucket_rows = [row for row in usable if int(row.get("teacher_bucket", 0) or 0) == bucket]
        if bucket_rows:
            teacher_bucket_stats.append(
                {
                    "bucket": bucket,
                    "count": len(bucket_rows),
                    "mean_teacher_score": float(np.mean([_scorefit_safe_float(row.get("teacher_score"), 0.0) for row in bucket_rows])),
                    "mean_llm_score": float(np.mean([_scorefit_safe_float(row.get("total_score"), 0.0) for row in bucket_rows])),
                    "mean_return": float(np.mean([_scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in bucket_rows])),
                }
            )
        sel_rows = [row for row in top_selected if int(row.get("teacher_bucket", 0) or 0) == bucket]
        if sel_rows:
            selected_bucket_stats.append(
                {
                    "bucket": bucket,
                    "count": len(sel_rows),
                    "mean_return": float(np.mean([_scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in sel_rows])),
                }
            )

    item_stats: List[Dict[str, Any]] = []
    base_uplift = (
        (
            float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in top_selected]))
            if top_selected
            else 0.0
        )
        - (
            float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in not_selected]))
            if not_selected
            else 0.0
        )
    )
    for item_id in item_ids:
        raw_scores = [_scorefit_safe_float((row.get("subscores") or {}).get(item_id), 0.0) for row in usable]
        ablated_scores = [
            _scorefit_safe_float(row.get("total_score"), 0.0) - _scorefit_safe_float((row.get("subscores") or {}).get(item_id), 0.0)
            for row in usable
        ]
        without_spearman = _scorefit_corr_spearman(ablated_scores, teacher_scores)
        ablated_ranked = sorted(
            zip(usable, ablated_scores),
            key=lambda pair: (
                -_scorefit_safe_float(pair[1], 0.0),
                str(pair[0].get("signal_date")),
                str(pair[0].get("symbol")),
            ),
        )[:top_k]
        ablated_selected_keys = {str(item.get("task_key")) for item, _ in ablated_ranked}
        ablated_top_selected = [item for item, _ in ablated_ranked]
        ablated_not_selected = [
            item for item in usable if str(item.get("task_key")) not in ablated_selected_keys
        ]
        uplift_without_item = (
            (
                float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in ablated_top_selected]))
                if ablated_top_selected
                else 0.0
            )
            - (
                float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in ablated_not_selected]))
                if ablated_not_selected
                else 0.0
            )
        )
        item_stats.append(
            {
                "item_id": item_id,
                "mean_subscore": float(np.mean(raw_scores)) if raw_scores else 0.0,
                "nonzero_rate": float(np.mean([abs(score) > 1e-12 for score in raw_scores])) if raw_scores else 0.0,
                "item_vs_teacher_spearman": _scorefit_corr_spearman(raw_scores, teacher_scores),
                "spearman_without_item": without_spearman,
                "ablation_delta": base_spearman - without_spearman,
                "uplift_without_item": uplift_without_item,
                "ablation_uplift_delta": base_uplift - uplift_without_item,
            }
        )
    item_stats.sort(key=lambda row: float(row.get("ablation_delta", 0.0)), reverse=True)

    return {
        "successful_signal_count": total_success,
        "teacher_score_spearman": base_spearman,
        "llm_top20_count": len(top_selected),
        "llm_top20_mean_return": (
            float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in top_selected]))
            if top_selected
            else 0.0
        ),
        "llm_not_selected_mean_return": (
            float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in not_selected]))
            if not_selected
            else 0.0
        ),
        "sample_mean_return": (
            float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in usable]))
            if usable
            else 0.0
        ),
        "teacher_q5_mean_return": (
            float(
                np.mean(
                    [
                        _scorefit_safe_float(item.get("future_return_5d"), 0.0)
                        for item in teacher_q5_rows
                    ]
                )
            )
            if teacher_q5_rows
            else 0.0
        ),
        "teacher_not_selected_mean_return": (
            float(
                np.mean(
                    [
                        _scorefit_safe_float(item.get("future_return_5d"), 0.0)
                        for item in teacher_not_selected_rows
                    ]
                )
            )
            if teacher_not_selected_rows
            else 0.0
        ),
        "llm_uplift_vs_not_selected": (
            (
                float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in top_selected]))
                if top_selected
                else 0.0
            )
            - (
                float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in not_selected]))
                if not_selected
                else 0.0
            )
        ),
        "teacher_q5_uplift_vs_not_selected": (
            (
                float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in teacher_q5_rows]))
                if teacher_q5_rows
                else 0.0
            )
            - (
                float(
                    np.mean(
                        [
                            _scorefit_safe_float(item.get("future_return_5d"), 0.0)
                            for item in teacher_not_selected_rows
                        ]
                    )
                )
                if teacher_not_selected_rows
                else 0.0
            )
        ),
        "gap_q5_uplift": (
            (
                (
                    float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in teacher_q5_rows]))
                    if teacher_q5_rows
                    else 0.0
                )
                - (
                    float(
                        np.mean(
                            [
                                _scorefit_safe_float(item.get("future_return_5d"), 0.0)
                                for item in teacher_not_selected_rows
                            ]
                        )
                    )
                    if teacher_not_selected_rows
                    else 0.0
                )
            )
            - (
                (
                    float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in top_selected]))
                    if top_selected
                    else 0.0
                )
                - (
                    float(np.mean([_scorefit_safe_float(item.get("future_return_5d"), 0.0) for item in not_selected]))
                    if not_selected
                    else 0.0
                )
            )
        ),
        "batch_nav_final": nav,
        "top20_selection_task_keys": sorted(selected_keys),
        "daily_returns": daily_returns,
        "teacher_bucket_stats": teacher_bucket_stats,
        "selected_bucket_stats": selected_bucket_stats,
        "item_stats": item_stats,
    }


def _scorefit_objective_summary(batch_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    teacher_bucket_stats = list(batch_metrics.get("teacher_bucket_stats") or [])
    selected_bucket_stats = list(batch_metrics.get("selected_bucket_stats") or [])
    q5_teacher = next((row for row in teacher_bucket_stats if int(row.get("bucket", 0) or 0) == 5), {})
    q5_selected = next((row for row in selected_bucket_stats if int(row.get("bucket", 0) or 0) == 5), {})
    return {
        "teacher_score_spearman": _scorefit_safe_float(batch_metrics.get("teacher_score_spearman"), 0.0),
        "llm_top20_mean_return": _scorefit_safe_float(batch_metrics.get("llm_top20_mean_return"), 0.0),
        "llm_not_selected_mean_return": _scorefit_safe_float(batch_metrics.get("llm_not_selected_mean_return"), 0.0),
        "llm_uplift_vs_not_selected": _scorefit_safe_float(batch_metrics.get("llm_uplift_vs_not_selected"), 0.0),
        "teacher_q5_mean_return": _scorefit_safe_float(batch_metrics.get("teacher_q5_mean_return"), 0.0),
        "teacher_not_selected_mean_return": _scorefit_safe_float(batch_metrics.get("teacher_not_selected_mean_return"), 0.0),
        "teacher_q5_uplift_vs_not_selected": _scorefit_safe_float(batch_metrics.get("teacher_q5_uplift_vs_not_selected"), 0.0),
        "gap_q5_uplift": _scorefit_safe_float(batch_metrics.get("gap_q5_uplift"), 0.0),
        "successful_signal_count": int(batch_metrics.get("successful_signal_count", 0) or 0),
        "llm_top20_count": int(batch_metrics.get("llm_top20_count", 0) or 0),
        "q5_selected_count_inside_llm_top20": int(q5_selected.get("count", 0) or 0),
        "q5_selected_mean_return_inside_llm_top20": _scorefit_safe_float(q5_selected.get("mean_return"), 0.0),
        "teacher_q5_sample_count": int(q5_teacher.get("count", 0) or 0),
        "teacher_q5_sample_mean_return": _scorefit_safe_float(q5_teacher.get("mean_return"), 0.0),
    }


def _scorefit_batch_composite_score(
    batch_metrics: Mapping[str, Any],
    *,
    composite_lambda: float,
) -> float:
    return (
        _scorefit_safe_float(batch_metrics.get("teacher_score_spearman"), 0.0)
        + float(composite_lambda) * _scorefit_safe_float(batch_metrics.get("llm_uplift_vs_not_selected"), 0.0)
    )


def _scorefit_objective_method_slug(value: Any) -> str:
    normalized = str(value or "raw_linear").strip().lower()
    if normalized == "zscore_sum":
        return "zscore_sum"
    return "raw_linear"


def _scorefit_checkpoint_context(
    *,
    lesson0_json: Mapping[str, Any],
    current_lesson: Mapping[str, Any],
    current_batch_metrics: Mapping[str, Any],
    scope_batch_history: Sequence[Mapping[str, Any]],
    composite_lambda: float,
    composite_method: str = "raw_linear",
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    scope_batch_history = [dict(row) for row in list(scope_batch_history or []) if isinstance(row, Mapping)]

    if scope_batch_history:
        first_metrics = dict(scope_batch_history[0].get("batch_metrics") or {})
        candidates.append(
            {
                "lesson_label": "Lesson0",
                "candidate_kind": "lesson0",
                "produced_after_batch": 0,
                "evaluated_on_batch": int(scope_batch_history[0].get("batch_index", 1) or 1),
                "lesson_json": dict(lesson0_json or {}),
                "batch_metrics": first_metrics,
                "raw_composite_score": _scorefit_batch_composite_score(
                    first_metrics,
                    composite_lambda=composite_lambda,
                ),
            }
        )
        for idx in range(1, len(scope_batch_history)):
            produced_payload = dict(scope_batch_history[idx - 1] or {})
            eval_payload = dict(scope_batch_history[idx] or {})
            eval_metrics = dict(eval_payload.get("batch_metrics") or {})
            candidates.append(
                {
                    "lesson_label": f"Lesson{idx}",
                    "candidate_kind": "warmup_checkpoint",
                    "produced_after_batch": int(produced_payload.get("batch_index", idx) or idx),
                    "evaluated_on_batch": int(eval_payload.get("batch_index", idx + 1) or (idx + 1)),
                    "lesson_json": dict(produced_payload.get("lesson_json") or {}),
                    "batch_metrics": eval_metrics,
                    "raw_composite_score": _scorefit_batch_composite_score(
                        eval_metrics,
                        composite_lambda=composite_lambda,
                    ),
                }
            )

    current_index = len(scope_batch_history)
    candidates.append(
        {
            "lesson_label": f"Lesson{current_index}",
            "candidate_kind": "current_lesson",
            "produced_after_batch": current_index,
            "evaluated_on_batch": current_index + 1,
            "lesson_json": dict(current_lesson or {}),
            "batch_metrics": dict(current_batch_metrics or {}),
            "raw_composite_score": _scorefit_batch_composite_score(
                current_batch_metrics,
                composite_lambda=composite_lambda,
            ),
        }
    )

    normalized_method = _scorefit_objective_method_slug(composite_method)
    spearman_values = [
        _scorefit_safe_float((row.get("batch_metrics") or {}).get("teacher_score_spearman"), 0.0)
        for row in candidates
    ]
    uplift_values = [
        _scorefit_safe_float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected"), 0.0)
        for row in candidates
    ]
    spearman_mean = float(np.mean(spearman_values)) if spearman_values else 0.0
    uplift_mean = float(np.mean(uplift_values)) if uplift_values else 0.0
    spearman_std = float(np.std(spearman_values)) if spearman_values else 0.0
    uplift_std = float(np.std(uplift_values)) if uplift_values else 0.0
    for row in candidates:
        spearman = _scorefit_safe_float((row.get("batch_metrics") or {}).get("teacher_score_spearman"), 0.0)
        uplift = _scorefit_safe_float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected"), 0.0)
        z_spearman = 0.0 if spearman_std <= 1e-12 else float((spearman - spearman_mean) / spearman_std)
        z_uplift = 0.0 if uplift_std <= 1e-12 else float((uplift - uplift_mean) / uplift_std)
        row["z_spearman"] = z_spearman
        row["z_uplift"] = z_uplift
        row["composite_score"] = (
            float(z_spearman + float(composite_lambda) * z_uplift)
            if normalized_method == "zscore_sum"
            else float(row.get("raw_composite_score", 0.0) or 0.0)
        )

    best = max(
        candidates,
        key=lambda row: (
            float(row.get("composite_score", 0.0)),
            _scorefit_safe_float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected"), 0.0),
            _scorefit_safe_float((row.get("batch_metrics") or {}).get("teacher_score_spearman"), 0.0),
            -int(row.get("produced_after_batch", 0) or 0),
        ),
    )

    leaderboard = sorted(
        [
            {
                "lesson_label": str(row.get("lesson_label", "")).strip(),
                "candidate_kind": str(row.get("candidate_kind", "")).strip(),
                "produced_after_batch": int(row.get("produced_after_batch", 0) or 0),
                "evaluated_on_batch": int(row.get("evaluated_on_batch", 0) or 0),
                "composite_score": float(row.get("composite_score", 0.0) or 0.0),
                "raw_composite_score": float(row.get("raw_composite_score", 0.0) or 0.0),
                "z_spearman": float(row.get("z_spearman", 0.0) or 0.0),
                "z_uplift": float(row.get("z_uplift", 0.0) or 0.0),
                "teacher_score_spearman": _scorefit_safe_float((row.get("batch_metrics") or {}).get("teacher_score_spearman"), 0.0),
                "llm_uplift_vs_not_selected": _scorefit_safe_float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected"), 0.0),
                "llm_top20_mean_return": _scorefit_safe_float((row.get("batch_metrics") or {}).get("llm_top20_mean_return"), 0.0),
                "teacher_q5_mean_return": _scorefit_safe_float((row.get("batch_metrics") or {}).get("teacher_q5_mean_return"), 0.0),
            }
            for row in candidates
        ],
        key=lambda row: (
            float(row.get("composite_score", 0.0)),
            float(row.get("llm_uplift_vs_not_selected", 0.0)),
            float(row.get("teacher_score_spearman", 0.0)),
        ),
        reverse=True,
    )[:6]

    best_batch_metrics = dict(best.get("batch_metrics") or {})
    return {
        "best_checkpoint": {
            "lesson_label": str(best.get("lesson_label", "")).strip(),
            "candidate_kind": str(best.get("candidate_kind", "")).strip(),
            "produced_after_batch": int(best.get("produced_after_batch", 0) or 0),
            "evaluated_on_batch": int(best.get("evaluated_on_batch", 0) or 0),
            "composite_score": float(best.get("composite_score", 0.0) or 0.0),
            "raw_composite_score": float(best.get("raw_composite_score", 0.0) or 0.0),
            "z_spearman": float(best.get("z_spearman", 0.0) or 0.0),
            "z_uplift": float(best.get("z_uplift", 0.0) or 0.0),
            "composite_method": normalized_method,
            "objective_summary": _scorefit_objective_summary(best_batch_metrics),
            "lesson": dict(best.get("lesson_json") or {}),
            "item_stats": list(best_batch_metrics.get("item_stats") or []),
        },
        "checkpoint_leaderboard": leaderboard,
    }


def _scorefit_recent_batch_context(
    previous_batch_metrics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    recent = [dict(row) for row in list(previous_batch_metrics or [])[-3:] if isinstance(row, Mapping)]
    if not recent:
        return {
            "recent_batch_count": 0,
            "recent_spearman_mean": 0.0,
            "recent_spearman_std": 0.0,
            "recent_llm_top20_mean_return": 0.0,
            "recent_teacher_q5_mean_return": 0.0,
            "stable_item_signals": [],
        }

    spearman_values = [
        _scorefit_safe_float(row.get("teacher_score_spearman"), 0.0)
        for row in recent
    ]
    llm_values = [
        _scorefit_safe_float(row.get("llm_top20_mean_return"), 0.0)
        for row in recent
    ]
    teacher_values = [
        _scorefit_safe_float(row.get("teacher_q5_mean_return"), 0.0)
        for row in recent
    ]
    item_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for batch_row in recent:
        for item in list(batch_row.get("item_stats") or []):
            if not isinstance(item, Mapping):
                continue
            item_id = str(item.get("item_id", "")).strip()
            if not item_id:
                continue
            item_rows.setdefault(item_id, []).append(item)

    stable_item_signals: List[Dict[str, Any]] = []
    for item_id, rows in item_rows.items():
        deltas = [_scorefit_safe_float(row.get("ablation_delta"), 0.0) for row in rows]
        ivt = [_scorefit_safe_float(row.get("item_vs_teacher_spearman"), 0.0) for row in rows]
        stable_item_signals.append(
            {
                "item_id": item_id,
                "recent_seen_batches": len(rows),
                "recent_ablation_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
                "recent_ablation_delta_positive_rate": float(np.mean([value > 0 for value in deltas])) if deltas else 0.0,
                "recent_item_vs_teacher_spearman_mean": float(np.mean(ivt)) if ivt else 0.0,
            }
        )
    stable_item_signals.sort(
        key=lambda row: (
            -_scorefit_safe_float(row.get("recent_ablation_delta_mean"), 0.0),
            -_scorefit_safe_float(row.get("recent_ablation_delta_positive_rate"), 0.0),
            str(row.get("item_id", "")),
        )
    )

    return {
        "recent_batch_count": len(recent),
        "recent_spearman_mean": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "recent_spearman_std": float(np.std(spearman_values)) if spearman_values else 0.0,
        "recent_llm_top20_mean_return": float(np.mean(llm_values)) if llm_values else 0.0,
        "recent_teacher_q5_mean_return": float(np.mean(teacher_values)) if teacher_values else 0.0,
        "stable_item_signals": stable_item_signals[:8],
    }


def _scorefit_revise_lesson(
    *,
    config: ApprenticeReplayConfig,
    scope_domain: Mapping[str, Any],
    explainability_summary: Mapping[str, Any],
    background_digest: str,
    current_lesson: Mapping[str, Any],
    lesson0_json: Mapping[str, Any],
    batch_metrics: Mapping[str, Any],
    revise_history: Sequence[Mapping[str, Any]],
    previous_batch_metrics: Sequence[Mapping[str, Any]],
    scope_batch_history: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    variant_settings = _scorefit_variant_settings(config)
    theory_payload = _teacher_explainability_payload(
        explainability_summary,
        max_branches=4,
        max_soft_rules=10,
        max_veto_rules=4,
        max_meta_rules=4,
        max_trap_pairs=3,
    )
    system = (
        "You are revising a structured JSON scorecard to imitate a hidden teacher score. "
        "Do not ask for raw samples. Use only the theory pack, the current lesson, the aggregate batch statistics, and the best-checkpoint context when provided. "
        "Keep useful item ids when possible so learning remains stable across batches. "
        "Return strict JSON only with keys: lesson, history_entry. "
        "lesson must be a full object with keys: schema, lesson_name, teacher_scope, items, meta_rules, scoring_notes. "
        "lesson.items must be a non-empty object keyed by item id like I01, I02. "
        "history_entry must include: summary, keep_item_ids, add_item_ids, drop_item_ids."
        " If multiple revisions are similarly plausible, use sampling_seed only as a deterministic tie-breaker "
        "to choose one coherent revision path."
        f" {variant_settings.get('revise_extra_instruction', '')}"
    )
    checkpoint_context = (
        _scorefit_checkpoint_context(
            lesson0_json=lesson0_json,
            current_lesson=current_lesson,
            current_batch_metrics=batch_metrics,
            scope_batch_history=scope_batch_history,
            composite_lambda=float(variant_settings.get("best_checkpoint_objective_lambda", 0.5)),
            composite_method=str(variant_settings.get("best_checkpoint_objective_method", "raw_linear")),
        )
        if bool(variant_settings.get("best_checkpoint_guidance"))
        else {}
    )
    user = json.dumps(
        {
            "teacher_scope": dict(scope_domain),
            "sampling_seed": int(config.sample_seed or 0),
            "research_background_digest": _scorefit_clipped_digest_text(background_digest, max_chars=2500),
            "report_v2": theory_payload,
            "current_lesson": current_lesson,
            "lesson_revise_history": (
                []
                if bool(variant_settings.get("suppress_revise_history_prompt"))
                else list(revise_history)[-8:]
            ),
            "current_best_checkpoint": dict(checkpoint_context.get("best_checkpoint") or {}),
            "checkpoint_leaderboard": list(checkpoint_context.get("checkpoint_leaderboard") or []),
            "batch_metrics": batch_metrics,
            "objective_summary": _scorefit_objective_summary(batch_metrics),
            "recent_batch_context": (
                _scorefit_recent_batch_context(previous_batch_metrics)
                if bool(variant_settings.get("include_recent_batch_context"))
                else {}
            ),
        },
        ensure_ascii=False,
    )
    try:
        response = _chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            api_key=_apprentice_api_key(),
            model=config.api_model,
            max_tokens=min(config.api_max_tokens, config.warmup_lesson_rewrite_max_tokens),
            temperature=0.0,
            force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
            fail_fast_on_empty_content=True,
            seed=int(config.sample_seed or 0) or None,
        )
        content = str(response["choices"][0]["message"].get("content", "") or "")
        parsed = _extract_json_payload(content)
        lesson_raw = parsed.get("lesson") if isinstance(parsed, Mapping) else None
        lesson_payload = _scorefit_postprocess_lesson_payload(
            _scorefit_normalize_lesson_payload(
                lesson_raw if isinstance(lesson_raw, Mapping) else parsed,
                scope_domain=scope_domain,
            ),
            config=config,
        )
        history_entry = parsed.get("history_entry") if isinstance(parsed, Mapping) else {}
        if not isinstance(history_entry, Mapping):
            history_entry = {}
        history_clean = {
            "summary": str(history_entry.get("summary", "")).strip() or "lesson revised from aggregate batch stats",
            "keep_item_ids": [str(item).strip() for item in list(history_entry.get("keep_item_ids") or []) if str(item).strip()],
            "add_item_ids": [str(item).strip() for item in list(history_entry.get("add_item_ids") or []) if str(item).strip()],
            "drop_item_ids": [str(item).strip() for item in list(history_entry.get("drop_item_ids") or []) if str(item).strip()],
            "raw_response": content,
        }
        return lesson_payload, history_clean
    except Exception as exc:
        return (
            _scorefit_postprocess_lesson_payload(dict(current_lesson), config=config),
            {
                "summary": f"fallback_keep_old_lesson reason={exc}",
                "keep_item_ids": list((current_lesson.get("items") or {}).keys()),
                "add_item_ids": [],
                "drop_item_ids": [],
                "raw_response": "",
            },
        )


def _load_existing_scorefit_scope_batch_records(
    *,
    warmup_report_dir: Path,
    scope_index: int,
    scope_round_id: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    state_paths = sorted(
        warmup_report_dir.glob(f"scope_{scope_index:02d}_{scope_round_id}_batch_*_state.json")
    )
    expected_batch_index = 1
    for state_path in state_paths:
        match = re.search(r"_batch_(\d+)_state\.json$", state_path.name)
        if not match:
            continue
        batch_index = int(match.group(1))
        if batch_index != expected_batch_index:
            break
        try:
            payload = _load_json(state_path)
        except Exception:
            break
        records.append({"batch_index": batch_index, "state": payload, "state_path": str(state_path)})
        expected_batch_index += 1
    return records


def _generate_warmup_lessons_scorefit_v1_json(
    *,
    config: ApprenticeReplayConfig,
    warm_cfg: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
    initial_scoped_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    warmup_report_dir = REPORT_ROOT / warm_cfg.run_id()
    warmup_report_dir.mkdir(parents=True, exist_ok=True)
    completed_state_path = warmup_report_dir / "warmup_scoped_lessons.json"
    if completed_state_path.exists():
        try:
            completed_state = _load_json(completed_state_path)
        except Exception:
            completed_state = {}
        if completed_state.get("curriculum") == "scorefit_v1_json":
            _progress(
                "scorefit_v1_json warmup state cache hit "
                f"run_id={warm_cfg.run_id()} path={_relative(completed_state_path)}"
            )
            return completed_state

    source_round_ids = list(warm_cfg.teacher_round_ids)
    final_round_ids = list(config.teacher_round_ids)
    if len(source_round_ids) != len(final_round_ids):
        raise ValueError("scorefit_v1_json requires one source round per final teacher round")

    sample_counts = _split_count_evenly(config.warmup_sample_count, len(final_round_ids))
    background_digest = _load_research_experience_digest()
    teacher_scope_payloads: List[Dict[str, Any]] = []
    combined_batch_history: List[Dict[str, Any]] = []
    warmup_csv_rows: List[Dict[str, Any]] = []
    _progress(
        "scorefit_v1_json warmup start "
        f"run_id={warm_cfg.run_id()} sample_count={config.warmup_sample_count} "
        f"teacher_scopes={len(final_round_ids)} batch_size={config.warmup_batch_size} "
        f"signal_pool_per_day={config.warmup_signal_pool_per_day} "
        f"variant={config.scorefit_variant}"
    )

    for scope_index, (final_round_id, source_round_id, scope_sample_count) in enumerate(
        zip(final_round_ids, source_round_ids, sample_counts),
        start=1,
    ):
        scope_cfg = replace(
            warm_cfg,
            mode="single",
            teacher_round_ids=[source_round_id],
            negative_teacher_round_ids=[],
            candidate_source="baseline_signal",
            warmup_sample_count=0,
            run_tag=f"{warm_cfg.run_tag}_scope{scope_index:02d}_{source_round_id}",
        )
        scope_frame, scope_meta = _build_single_teacher_frame(scope_cfg, master_df)
        scope_domain = _scope_domain_card(
            scope_meta,
            scope_round_id=final_round_id,
            source_round_id=source_round_id,
        )
        explainability_summary, explainability_round_id = _resolve_explainability_summary(
            preferred_round_ids=[final_round_id, source_round_id],
            fallback_summary=scope_meta.get("factor_analysis_summary", {}),
        )
        scope_domain["explainability_round_id"] = explainability_round_id or str(final_round_id)
        if scope_frame.empty or scope_sample_count <= 0:
            teacher_scope_payloads.append(
                {
                    **scope_domain,
                    "scope_index": scope_index,
                    "scope_sample_count": scope_sample_count,
                    "sampled_days": 0,
                    "scorefit_variant": config.scorefit_variant,
                    "scope_lesson_zone_lines": [],
                    "review_cards_for_prompt": [],
                    "scorefit_lesson_json": {},
                    "revise_history": [],
                    "batch_history": [],
                }
            )
            continue

        sampled_dates = _sample_uniform_dates(
            scope_frame["signal_date"].tolist(),
            scope_sample_count,
            sample_seed=(config.sample_seed + scope_index) if int(config.sample_seed) > 0 else 0,
        )
        sampled_dates = sorted(pd.Timestamp(item) for item in sampled_dates)
        scope_frame = scope_frame[scope_frame["signal_date"].isin(sampled_dates)].copy()
        feature_cols = _lesson_feature_columns(scope_frame)
        batch_size = max(1, int(config.warmup_batch_size))
        date_batches = [sampled_dates[idx : idx + batch_size] for idx in range(0, len(sampled_dates), batch_size)]

        initial_scope_entry = _find_teacher_scope_entry(initial_scoped_state, final_round_id)
        if initial_scope_entry is None:
            initial_scope_entry = _find_teacher_scope_entry(initial_scoped_state, source_round_id)
        if initial_scope_entry and isinstance(initial_scope_entry.get("scorefit_lesson_json"), Mapping):
            lesson_json = _scorefit_postprocess_lesson_payload(
                _scorefit_normalize_lesson_payload(
                    dict(initial_scope_entry.get("scorefit_lesson_json") or {}),
                    scope_domain=scope_domain,
                ),
                config=config,
            )
            lesson0_artifact = {
                "mode": "initial_scoped_state_override",
                "initial_scope_round_id": str(initial_scope_entry.get("round_id", "")).strip(),
                "initial_scope_source_round_id": str(initial_scope_entry.get("source_round_id", "")).strip(),
            }
        else:
            lesson_json, lesson0_artifact = _scorefit_build_lesson0(
                config=config,
                scope_domain=scope_domain,
                explainability_summary=explainability_summary,
                background_digest=background_digest,
            )
        lesson0_json = dict(lesson_json)
        revise_history: List[Dict[str, Any]] = []
        scope_batch_history: List[Dict[str, Any]] = []

        lesson0_path = warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_lesson0.json"
        if not lesson0_path.exists():
            _write_json(lesson0_path, {"lesson": lesson_json, "artifact": lesson0_artifact})

        resume_records = _load_existing_scorefit_scope_batch_records(
            warmup_report_dir=warmup_report_dir,
            scope_index=scope_index,
            scope_round_id=final_round_id,
        )
        if resume_records:
            last_state = dict(resume_records[-1]["state"])
            lesson_json = dict(last_state.get("lesson_json") or lesson_json)
            revise_history = list(last_state.get("revise_history") or [])
            scope_batch_history = [dict(record["state"]) for record in resume_records]
            combined_batch_history.extend(scope_batch_history)
            for batch_row in scope_batch_history:
                batch_metrics = dict(batch_row.get("batch_metrics") or {})
                warmup_csv_rows.append(
                    {
                        "scope_round_id": final_round_id,
                        "scope_index": scope_index,
                        "batch_index": int(batch_row.get("batch_index", 0) or 0),
                        "teacher_score_spearman": _scorefit_safe_float(batch_metrics.get("teacher_score_spearman"), 0.0),
                        "llm_top20_mean_return": _scorefit_safe_float(batch_metrics.get("llm_top20_mean_return"), 0.0),
                        "teacher_q5_mean_return": _scorefit_safe_float(batch_metrics.get("teacher_q5_mean_return"), 0.0),
                        "batch_nav_final": _scorefit_safe_float(batch_metrics.get("batch_nav_final"), 1.0),
                        "successful_signal_count": int(batch_metrics.get("successful_signal_count", 0) or 0),
                    }
                )
            _progress(
                "scorefit_v1_json resume detected "
                f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
                f"completed_batches={len(resume_records)}/{len(date_batches)}"
            )

        start_batch_index = len(scope_batch_history) + 1
        _progress(
            "scorefit_v1_json teacher start "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} source_round={source_round_id} "
            f"sampled_days={len(sampled_dates)}"
        )
        for batch_index, batch_dates in enumerate(date_batches[start_batch_index - 1 :], start=start_batch_index):
            batch_tasks: List[Dict[str, Any]] = []
            task_counter = 0
            for day_offset, decision_date in enumerate(batch_dates):
                day_df = scope_frame[scope_frame["signal_date"] == pd.Timestamp(decision_date)].copy()
                sampled_day = _scorefit_sample_day_signals(
                    day_df,
                    x=int(config.warmup_signal_pool_per_day),
                    sample_seed=(config.sample_seed * 10000 + scope_index * 100 + batch_index * 10 + day_offset)
                    if int(config.sample_seed) > 0
                    else 0,
                )
                for _, row in sampled_day.iterrows():
                    task_counter += 1
                    task_key = (
                        f"{final_round_id}::batch{batch_index:02d}::"
                        f"{pd.Timestamp(decision_date).strftime('%Y%m%d')}::{str(row.get('symbol', '')).strip()}::{task_counter:03d}"
                    )
                    batch_tasks.append(
                        {
                            "task_key": task_key,
                            "scope_round_id": final_round_id,
                            "scope_source_round_id": source_round_id,
                            "scope_domain": scope_domain,
                            "signal_date": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"),
                            "symbol": str(row.get("symbol", "")).strip(),
                            "teacher_score": _scorefit_safe_float(row.get("score"), 0.0),
                            "teacher_bucket": int(row.get("bucket", 0) or 0),
                            "future_return_5d": _scorefit_safe_float(row.get("future_return_5d"), 0.0),
                            "signal_record": _scorefit_signal_record_from_row(row, feature_cols),
                            "cache_path": str(
                                warmup_report_dir
                                / "api_calls"
                                / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_{task_counter:04d}.json"
                            ),
                            "reuse_api_cache": True,
                            "rerun_count": 0,
                        }
                    )
            if not batch_tasks:
                continue
            _progress(
                "scorefit_v1_json batch start "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"days={len(batch_dates)} tasks={len(batch_tasks)}"
            )

            pending = list(batch_tasks)
            finalized_results: Dict[str, Dict[str, Any]] = {}
            global_batch_counter = 0
            total_passes = 1 + max(0, int(config.api_failed_rerun_rounds))
            for pass_index in range(total_passes):
                if not pending:
                    break
                next_pending: List[Dict[str, Any]] = []
                batch_cursor = 0
                while batch_cursor < len(pending):
                    task_batch = pending[batch_cursor : batch_cursor + max(1, int(config.api_parallel_workers))]
                    batch_cursor += max(1, int(config.api_parallel_workers))
                    global_batch_counter += 1
                    results = _scorefit_request_signal_batch(
                        task_batch=task_batch,
                        workers=max(1, int(config.api_parallel_workers)),
                        config=config,
                        lesson_payload=lesson_json,
                        api_key=_apprentice_api_key(),
                        global_batch_index=global_batch_counter,
                    )
                    requeued = 0
                    finalized = 0
                    source_by_key = {str(task["task_key"]): task for task in task_batch}
                    for result in results:
                        if bool(result.get("parse_fallback", False)) and pass_index + 1 < total_passes:
                            rerun_task = dict(source_by_key[str(result["task_key"])])
                            rerun_task["reuse_api_cache"] = False
                            rerun_task["rerun_count"] = int(rerun_task.get("rerun_count", 0)) + 1
                            next_pending.append(rerun_task)
                            requeued += 1
                        else:
                            finalized_results[str(result["task_key"])] = result
                            finalized += 1
                    _progress(
                        "scorefit_v1_json signal batch done "
                        f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                        f"pass={pass_index + 1}/{total_passes} wave={global_batch_counter} size={len(task_batch)} "
                        f"requeued={requeued} finalized={finalized}"
                    )
                pending = next_pending
            if pending:
                raise RuntimeError(
                    f"scorefit_v1_json unresolved failed samples remain: "
                    f"scope={final_round_id} batch={batch_index} pending={len(pending)}"
                )

            batch_results = [finalized_results[str(task["task_key"])] for task in batch_tasks]
            batch_metrics = _scorefit_batch_metrics(
                results=batch_results,
                lesson_payload=lesson_json,
                lock_days=config.lock_days,
            )
            revised_lesson_json, history_entry = _scorefit_revise_lesson(
                config=config,
                scope_domain=scope_domain,
                explainability_summary=explainability_summary,
                background_digest=background_digest,
                current_lesson=lesson_json,
                lesson0_json=lesson0_json,
                batch_metrics=batch_metrics,
                revise_history=revise_history,
                previous_batch_metrics=[dict(row.get("batch_metrics") or {}) for row in scope_batch_history[-3:]],
                scope_batch_history=scope_batch_history,
            )
            history_clean = {
                "batch_index": batch_index,
                "spearman_before": _scorefit_safe_float(batch_metrics.get("teacher_score_spearman"), 0.0),
                "summary": str(history_entry.get("summary", "")).strip(),
                "keep_item_ids": list(history_entry.get("keep_item_ids") or []),
                "add_item_ids": list(history_entry.get("add_item_ids") or []),
                "drop_item_ids": list(history_entry.get("drop_item_ids") or []),
            }
            revise_history.append(history_clean)
            lesson_json = revised_lesson_json
            scope_lesson_zone_lines = _scorefit_render_lesson_lines(
                lesson_json,
                limit=config.warmup_lesson_zone_max_lines,
            )
            review_cards_for_prompt = _scorefit_render_history_cards(
                revise_history,
                limit=config.warmup_review_memory_limit,
            )
            batch_payload = {
                "scope_index": scope_index,
                "scope_round_id": final_round_id,
                "scope_source_round_id": source_round_id,
                "scope_domain": scope_domain,
                "batch_index": batch_index,
                "batch_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in batch_dates],
                "successful_signal_count": int(batch_metrics.get("successful_signal_count", 0) or 0),
                "scope_lesson_zone_lines": scope_lesson_zone_lines,
                "review_cards_for_prompt": review_cards_for_prompt,
                "lesson_json": lesson_json,
                "revise_history": revise_history,
                "batch_metrics": batch_metrics,
            }
            scope_batch_history.append(batch_payload)
            combined_batch_history.append(batch_payload)
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_state.json",
                batch_payload,
            )
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_signal_scores.json",
                {"results": batch_results},
            )
            warmup_csv_rows.append(
                {
                    "scope_round_id": final_round_id,
                    "scope_index": scope_index,
                    "batch_index": batch_index,
                    "teacher_score_spearman": _scorefit_safe_float(batch_metrics.get("teacher_score_spearman"), 0.0),
                    "llm_top20_mean_return": _scorefit_safe_float(batch_metrics.get("llm_top20_mean_return"), 0.0),
                    "teacher_q5_mean_return": _scorefit_safe_float(batch_metrics.get("teacher_q5_mean_return"), 0.0),
                    "batch_nav_final": _scorefit_safe_float(batch_metrics.get("batch_nav_final"), 1.0),
                    "successful_signal_count": int(batch_metrics.get("successful_signal_count", 0) or 0),
                }
            )
            _progress(
                "scorefit_v1_json batch done "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"spearman={_format_float(batch_metrics.get('teacher_score_spearman'))} "
                f"signals={batch_metrics.get('successful_signal_count', 0)}"
            )

        scope_payload = {
            **scope_domain,
            "scope_index": scope_index,
            "scope_sample_count": scope_sample_count,
            "sampled_days": len(sampled_dates),
            "scorefit_variant": config.scorefit_variant,
            "scope_lesson_zone_lines": _scorefit_render_lesson_lines(
                lesson_json,
                limit=config.warmup_lesson_zone_max_lines,
            ),
            "review_cards_for_prompt": _scorefit_render_history_cards(
                revise_history,
                limit=config.warmup_review_memory_limit,
            ),
            "scorefit_lesson_json": lesson_json,
            "revise_history": revise_history,
            "retained_review_entry_count": 0,
            "batch_history": scope_batch_history,
        }
        teacher_scope_payloads.append(scope_payload)
        _write_json(
            warmup_report_dir / f"teacher_scope_{scope_index:02d}_{final_round_id}.json",
            scope_payload,
        )
        _progress(
            "scorefit_v1_json teacher done "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
            f"scope_lines={len(scope_payload['scope_lesson_zone_lines'])} "
            f"history_entries={len(revise_history)}"
        )

    scoped_state = {
        "curriculum": "scorefit_v1_json",
        "scorefit_variant": config.scorefit_variant,
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": config.warmup_batch_size,
        "warmup_signal_pool_per_day": int(config.warmup_signal_pool_per_day),
        "sample_seed": int(config.sample_seed),
        "global_lesson_zone_lines": [],
        "teacher_scope_order": list(final_round_ids),
        "teacher_scopes": teacher_scope_payloads,
        "review_cards_for_prompt": [],
        "batch_history": combined_batch_history,
        "artifact_dir": _relative(warmup_report_dir),
        "lesson_zone_lines": [],
    }
    _write_json(warmup_report_dir / "warmup_scoped_lessons.json", scoped_state)
    if warmup_csv_rows:
        pd.DataFrame(warmup_csv_rows).sort_values(["scope_index", "batch_index"]).to_csv(
            warmup_report_dir / "warmup_scorefit_batch_metrics.csv",
            index=False,
        )
    md_lines: List[str] = [
        "# ScoreFit Warmup Lessons",
        "",
        "Structured JSON scorecards generated by scorefit_v1_json.",
        "",
    ]
    for scope in teacher_scope_payloads:
        md_lines.extend(
            [
                f"## {scope.get('round_id')}",
                f"- family: {scope.get('family')}",
                f"- template: {scope.get('template')}",
                f"- style: {scope.get('style_hint')}",
                f"- basic_filter: {scope.get('basic_filter')}",
                "",
                "### Final Lesson Lines",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
                "### Final Lesson JSON",
                "```json",
                json.dumps(scope.get("scorefit_lesson_json") or {}, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    (warmup_report_dir / "warmup_scoped_lessons.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scoped_state


def _generate_warmup_lessons_iterative_v6_scoped_interval(
    *,
    config: ApprenticeReplayConfig,
    warm_cfg: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
) -> Dict[str, Any]:
    curriculum_name = config.warmup_curriculum
    if curriculum_name not in {"iterative_v6_scoped_interval", "iterative_v7_scoped_rulecard", "iterative_v8_weighted_soft_rules"}:
        raise ValueError(f"unsupported scoped warmup curriculum for this generator: {curriculum_name}")
    warmup_report_dir = REPORT_ROOT / warm_cfg.run_id()
    warmup_report_dir.mkdir(parents=True, exist_ok=True)
    completed_state_path = warmup_report_dir / "warmup_scoped_lessons.json"
    if completed_state_path.exists():
        try:
            completed_state = _load_json(completed_state_path)
        except Exception:
            completed_state = {}
        if completed_state.get("curriculum") == curriculum_name:
            _progress(
                f"{curriculum_name} warmup state cache hit "
                f"run_id={warm_cfg.run_id()} path={_relative(completed_state_path)}"
            )
            return completed_state
    source_round_ids = list(warm_cfg.teacher_round_ids)
    final_round_ids = list(config.teacher_round_ids)
    if len(source_round_ids) != len(final_round_ids):
        raise ValueError(f"{curriculum_name} requires one source round per final teacher round")
    sample_counts = _split_count_evenly(config.warmup_sample_count, len(final_round_ids))
    global_lesson_zone_lines: List[str] = []
    teacher_scope_payloads: List[Dict[str, Any]] = []
    combined_batch_history: List[Dict[str, Any]] = []
    combined_review_entries: List[Dict[str, Any]] = []
    _progress(
        f"{curriculum_name} warmup start "
        f"run_id={warm_cfg.run_id()} sample_count={config.warmup_sample_count} "
        f"teacher_scopes={len(final_round_ids)} batch_size={config.warmup_batch_size}"
    )
    for scope_index, (final_round_id, source_round_id, scope_sample_count) in enumerate(
        zip(final_round_ids, source_round_ids, sample_counts),
        start=1,
    ):
        scope_cfg = replace(
            warm_cfg,
            mode="single",
            teacher_round_ids=[source_round_id],
            negative_teacher_round_ids=[],
            candidate_source="baseline_signal",
            warmup_sample_count=0,
            run_tag=f"{warm_cfg.run_tag}_scope{scope_index:02d}_{source_round_id}",
        )
        scope_frame, scope_meta = _build_single_teacher_frame(scope_cfg, master_df)
        if config.summary_variant == "enriched_v2":
            scope_meta["preference_bands"] = _derive_preference_bands(
                round_id=source_round_id,
                master_df=master_df,
                feature_cols=scope_meta["top_prompt_features"],
                start_date=config.warmup_start_date,
                end_date=config.warmup_end_date,
            )
        scope_pool, scope_target = _single_teacher_target(scope_frame, scope_cfg)
        scope_domain = _scope_domain_card(scope_meta, scope_round_id=final_round_id, source_round_id=source_round_id)
        scope_feature_priority = list(scope_domain.get("top_features") or [])[: max(4, config.prompt_feature_count)]
        if scope_pool.empty or scope_target.empty or scope_sample_count <= 0:
            teacher_scope_payloads.append(
                {
                    **scope_domain,
                    "scope_index": scope_index,
                    "scope_sample_count": scope_sample_count,
                    "sampled_days": 0,
                    "scope_lesson_zone_lines": [],
                    "review_cards_for_prompt": [],
                    "retained_review_entry_count": 0,
                    "batch_history": [],
                }
            )
            continue
        sampled_dates = _sample_uniform_dates(
            scope_pool["signal_date"].tolist(),
            scope_sample_count,
            sample_seed=(config.sample_seed + scope_index) if int(config.sample_seed) > 0 else 0,
        )
        sampled_dates = sorted(pd.Timestamp(item) for item in sampled_dates)
        scope_pool = scope_pool[scope_pool["signal_date"].isin(sampled_dates)].copy()
        scope_target = scope_target[scope_target["signal_date"].isin(sampled_dates)].copy()
        lesson_feature_cols = _lesson_feature_columns(scope_pool)
        batch_size = max(1, int(config.warmup_batch_size))
        date_batches = [sampled_dates[idx : idx + batch_size] for idx in range(0, len(sampled_dates), batch_size)]
        retained_review_entries: List[Dict[str, Any]] = []
        scope_lesson_zone_lines: List[str] = []
        scope_batch_history: List[Dict[str, Any]] = []
        resume_records = _load_existing_scope_batch_records(
            warmup_report_dir=warmup_report_dir,
            scope_index=scope_index,
            scope_round_id=final_round_id,
        )
        if resume_records:
            last_resume_state = dict(resume_records[-1]["state"])
            scope_batch_history = [dict(record["state"]) for record in resume_records]
            combined_batch_history.extend(scope_batch_history)
            for record in resume_records:
                combined_review_entries.extend(list(record.get("review_entries", []) or []))
            global_lesson_zone_lines = list(
                last_resume_state.get("global_lesson_zone", global_lesson_zone_lines)
            )
            scope_lesson_zone_lines = list(last_resume_state.get("scope_lesson_zone", []))
            retained_review_entries = _restore_scope_retained_review_entries(
                batch_records=resume_records,
                limit=config.warmup_review_memory_limit,
                max_per_tier=config.warmup_retained_case_max_per_tier,
            )
            _progress(
                f"{curriculum_name} resume detected "
                f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
                f"completed_batches={len(resume_records)}/{len(date_batches)} "
                f"retained_review={len(retained_review_entries)}"
            )
        _progress(
            f"{curriculum_name} teacher start "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} source_round={source_round_id} "
            f"sampled_days={len(sampled_dates)}"
        )
        if len(scope_batch_history) >= len(date_batches):
            scope_payload = {
                **scope_domain,
                "scope_index": scope_index,
                "scope_sample_count": scope_sample_count,
                "sampled_days": len(sampled_dates),
                "scope_lesson_zone_lines": list(scope_lesson_zone_lines),
                "review_cards_for_prompt": _select_interval_review_cards_for_prompt(
                    retained_review_entries,
                    feature_priority=scope_feature_priority,
                    limit=config.warmup_review_memory_limit,
                ),
                "retained_review_entry_count": len(retained_review_entries),
                "batch_history": scope_batch_history,
            }
            teacher_scope_payloads.append(scope_payload)
            teacher_scope_path = warmup_report_dir / f"teacher_scope_{scope_index:02d}_{final_round_id}.json"
            if not teacher_scope_path.exists():
                _write_json(teacher_scope_path, scope_payload)
            _progress(
                f"{curriculum_name} teacher restored "
                f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
                f"scope_lines={len(scope_lesson_zone_lines)} retained_review={len(retained_review_entries)}"
            )
            continue
        start_batch_index = len(scope_batch_history) + 1
        for batch_index, batch_dates in enumerate(
            date_batches[start_batch_index - 1 :],
            start=start_batch_index,
        ):
            _progress(
                f"{curriculum_name} batch start "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"date_from={pd.Timestamp(batch_dates[0]).strftime('%Y-%m-%d')} "
                f"date_to={pd.Timestamp(batch_dates[-1]).strftime('%Y-%m-%d')}"
            )
            batch_cfg = replace(
                scope_cfg,
                run_tag=f"{warm_cfg.run_tag}_scope{scope_index:02d}_{source_round_id}_batch{batch_index:02d}of{len(date_batches)}",
            )
            batch_pool = scope_pool[scope_pool["signal_date"].isin(batch_dates)].copy()
            batch_target = scope_target[scope_target["signal_date"].isin(batch_dates)].copy()
            scope_prompt_review_cards = _select_interval_review_cards_for_prompt(
                retained_review_entries,
                feature_priority=scope_feature_priority,
                limit=config.warmup_review_memory_limit,
            )
            scope_prompt_state = {
                "global_lesson_zone_lines": list(global_lesson_zone_lines),
                "teacher_scopes": [
                    {
                        **scope_domain,
                        "scope_lesson_zone_lines": list(scope_lesson_zone_lines),
                        "review_cards_for_prompt": list(scope_prompt_review_cards),
                    }
                ],
            }
            _run_replay(
                config=batch_cfg,
                candidate_pool_df=batch_pool,
                teacher_target_df=batch_target,
                teacher_full_df=batch_target.copy(),
                prompt_builder=_daily_prompt_single,
                prompt_builder_kwargs={
                    "meta": scope_meta,
                    "negative_metas": [],
                    "warmup_lessons": scope_lesson_zone_lines,
                    "warmup_review_cards": list(scope_prompt_review_cards),
                    "scoped_warmup_state": scope_prompt_state,
                    "current_scope_round_id": final_round_id,
                },
            )
            batch_report_dir = REPORT_ROOT / batch_cfg.run_id()
            batch_entries = _extract_warmup_review_entries(
                report_dir=batch_report_dir,
                candidate_pool_df=batch_pool,
                teacher_target_df=batch_target,
                prompt_features=scope_meta["top_prompt_features"],
                lesson_feature_cols=lesson_feature_cols,
                teacher_round_ids=[final_round_id, source_round_id],
            )
            for entry in batch_entries:
                entry["scope_round_id"] = final_round_id
                entry["scope_source_round_id"] = source_round_id
            combined_review_entries.extend(batch_entries)
            if curriculum_name == "iterative_v8_weighted_soft_rules":
                rewrite_fn = _rewrite_scoped_lesson_zone_iterative_v8_weighted_soft_rules
            elif curriculum_name == "iterative_v7_scoped_rulecard":
                rewrite_fn = _rewrite_scoped_lesson_zone_iterative_v7_rulecard
            else:
                rewrite_fn = _rewrite_scoped_lesson_zone_iterative_v6_interval
            global_lesson_zone_lines, scope_lesson_zone_lines, retained_review_entries, lesson_artifact = rewrite_fn(
                config=config,
                meta=scope_meta,
                scope_round_id=final_round_id,
                source_round_id=source_round_id,
                retained_review_entries=retained_review_entries,
                old_global_lesson_zone=global_lesson_zone_lines,
                old_scope_lesson_zone=scope_lesson_zone_lines,
                batch_entries=batch_entries,
                batch_index=batch_index,
                total_batches=len(date_batches),
            )
            scope_review_cards = _select_interval_review_cards_for_prompt(
                retained_review_entries,
                feature_priority=scope_feature_priority,
                limit=config.warmup_review_memory_limit,
            )
            batch_payload = {
                "scope_index": scope_index,
                "scope_round_id": final_round_id,
                "scope_source_round_id": source_round_id,
                "scope_domain": scope_domain,
                "batch_index": batch_index,
                "batch_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in batch_dates],
                "batch_report_dir": _relative(batch_report_dir),
                "batch_review_count": len(batch_entries),
                "retained_review_count": len(retained_review_entries),
                "global_lesson_zone": list(global_lesson_zone_lines),
                "scope_lesson_zone": list(scope_lesson_zone_lines),
                "review_cards_for_prompt": scope_review_cards,
                "lesson_artifact": lesson_artifact,
            }
            scope_batch_history.append(batch_payload)
            combined_batch_history.append(batch_payload)
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_state.json",
                batch_payload,
            )
            _write_json(
                warmup_report_dir / f"scope_{scope_index:02d}_{final_round_id}_batch_{batch_index:02d}_review_entries.json",
                {"entries": batch_entries},
            )
            _progress(
                f"{curriculum_name} batch done "
                f"scope={scope_index}/{len(final_round_ids)} batch={batch_index}/{len(date_batches)} "
                f"retained_review={len(retained_review_entries)} global_lines={len(global_lesson_zone_lines)} "
                f"scope_lines={len(scope_lesson_zone_lines)}"
            )
        scope_payload = {
            **scope_domain,
            "scope_index": scope_index,
            "scope_sample_count": scope_sample_count,
            "sampled_days": len(sampled_dates),
            "scope_lesson_zone_lines": list(scope_lesson_zone_lines),
            "review_cards_for_prompt": _select_interval_review_cards_for_prompt(
                retained_review_entries,
                feature_priority=scope_feature_priority,
                limit=config.warmup_review_memory_limit,
            ),
            "retained_review_entry_count": len(retained_review_entries),
            "batch_history": scope_batch_history,
        }
        teacher_scope_payloads.append(scope_payload)
        _write_json(warmup_report_dir / f"teacher_scope_{scope_index:02d}_{final_round_id}.json", scope_payload)
        _progress(
            f"{curriculum_name} teacher done "
            f"scope={scope_index}/{len(final_round_ids)} final_round={final_round_id} "
            f"scope_lines={len(scope_lesson_zone_lines)} retained_review={len(retained_review_entries)}"
        )
    combined_feature_priority = list(
        dict.fromkeys(
            feat
            for scope in teacher_scope_payloads
            for feat in list(scope.get("top_features", []) or [])
        )
    )[: max(4, config.prompt_feature_count)]
    scoped_state = {
        "curriculum": curriculum_name,
        "warmup_sample_count": config.warmup_sample_count,
        "warmup_batch_size": config.warmup_batch_size,
        "sample_seed": int(config.sample_seed),
        "global_lesson_zone_lines": list(global_lesson_zone_lines),
        "teacher_scope_order": list(final_round_ids),
        "teacher_scopes": teacher_scope_payloads,
        "review_cards_for_prompt": _select_interval_review_cards_for_prompt(
            _annotate_review_entry_tiers(combined_review_entries),
            feature_priority=combined_feature_priority,
            limit=config.warmup_review_memory_limit,
        ),
        "batch_history": combined_batch_history,
        "artifact_dir": _relative(warmup_report_dir),
        "lesson_zone_lines": list(global_lesson_zone_lines),
    }
    _write_json(warmup_report_dir / "warmup_scoped_lessons.json", scoped_state)
    md_lines: List[str] = [
        "# Warmup Scoped Lessons",
        "",
        "## Global Lesson Zone",
        *(global_lesson_zone_lines or ["none"]),
        "",
        "## Teacher Scopes",
    ]
    for scope in teacher_scope_payloads:
        md_lines.extend(
            [
                f"### {scope.get('round_id')}",
                f"- family: {scope.get('family')}",
                f"- template: {scope.get('template')}",
                f"- style: {scope.get('style_hint')}",
                f"- basic_filter: {scope.get('basic_filter')}",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
                "Review Cards:",
                *(list(scope.get("review_cards_for_prompt") or ["none"])),
                "",
            ]
        )
    (warmup_report_dir / "warmup_scoped_lessons.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scoped_state


def _generate_warmup_lessons(
    *,
    config: ApprenticeReplayConfig,
    warmup_report_dir: Path,
    metas: Sequence[Dict[str, Any]],
    prompt_features: Sequence[str],
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
) -> List[str]:
    agreement_path = warmup_report_dir / "daily_agreement.csv"
    llm_path = warmup_report_dir / "llm_selected_signals.csv"
    teacher_path = warmup_report_dir / "teacher_target_signals.csv"
    if not (agreement_path.exists() and llm_path.exists() and teacher_path.exists()):
        return []
    agreement_df = pd.read_csv(agreement_path)
    llm_df = pd.read_csv(llm_path)
    teacher_df = pd.read_csv(teacher_path)
    if agreement_df.empty:
        return []
    prompt_log_path = warmup_report_dir / "daily_prompt_log.csv"
    prompt_log = pd.read_csv(prompt_log_path) if prompt_log_path.exists() else pd.DataFrame()
    prompt_log = prompt_log.set_index("decision_date") if not prompt_log.empty else pd.DataFrame()
    lesson_feature_cols = _lesson_feature_columns(candidate_pool_df)
    teacher_lines = [
        f"{meta['round_id']} {meta['research_family']} {meta['sample_template']} bands={_compact_band_hint(meta.get('preference_bands', []))}"
        for meta in metas
    ]
    prior_memory = _load_prior_trader_lessons(config.teacher_round_ids, limit=12)
    prior_lines = [str(item.get("concise_memory", "")) for item in prior_memory if item.get("concise_memory")]
    lesson_bank: List[Dict[str, Any]] = []
    for _, row in agreement_df.sort_values("decision_date").iterrows():
        day = pd.Timestamp(row["decision_date"])
        day_text = day.strftime("%Y-%m-%d")
        day_pool = candidate_pool_df[candidate_pool_df["signal_date"] == day].copy()
        if day_pool.empty:
            continue
        teacher_rows = teacher_target_df[teacher_target_df["signal_date"] == day].copy()
        llm_rows = llm_df[pd.to_datetime(llm_df["signal_date"]) == day].copy() if "signal_date" in llm_df.columns else pd.DataFrame()
        case_records = _lesson_case_records(day_pool, lesson_feature_cols)
        teacher_symbols = teacher_rows["symbol"].astype(str).tolist()
        llm_symbols = llm_rows["symbol"].astype(str).tolist()
        recent_memory = (prior_lines + [entry["concise_memory"] for entry in lesson_bank if entry.get("concise_memory")])[-8:]
        system = (
            "You are building a trader lesson bank from mock exams. "
            "Given one full case with complete candidate input features, explain the error or success in a reusable way. "
            "Return strict JSON with keys: verdict, concise_memory, teacher_preference, llm_mistake, correction_rule, trigger_pattern."
        )
        user = json.dumps(
            {
                "teacher_cards": teacher_lines,
                "old_lesson_memory": recent_memory,
                "case": {
                    "decision_date": day_text,
                    "jaccard": float(row["jaccard"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "teacher_target_symbols": teacher_symbols,
                    "llm_selected_symbols": llm_symbols,
                    "candidate_full_input_features": case_records,
                },
            },
            ensure_ascii=False,
        )
        response = _chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            api_key=_apprentice_api_key(),
            model=config.api_model,
            max_tokens=min(config.api_max_tokens, 320),
            temperature=0.0,
        )
        content = response["choices"][0]["message"].get("content", "") or ""
        fallback_parsed = _case_delta_fallback(
            day_text=day_text,
            day_pool=day_pool,
            teacher_symbols=teacher_symbols,
            llm_symbols=llm_symbols,
            feature_cols=lesson_feature_cols,
        )
        try:
            parsed = _extract_json_payload(content)
        except Exception:
            parsed = fallback_parsed
        else:
            for key, value in fallback_parsed.items():
                if parsed.get(key) in (None, "", []):
                    parsed[key] = value
        lesson_entry = {
            "lesson_id": f"warmup_{len(lesson_bank)+1:03d}",
            "source_run_id": warmup_report_dir.name,
            "decision_date": day_text,
            "teacher_round_ids": list(config.teacher_round_ids),
            "jaccard": float(row["jaccard"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "teacher_target_symbols": teacher_symbols,
            "llm_selected_symbols": llm_symbols,
            "candidate_full_input_features": case_records,
            "raw_reflection": content,
            "verdict": parsed.get("verdict"),
            "teacher_preference": parsed.get("teacher_preference"),
            "llm_mistake": parsed.get("llm_mistake"),
            "correction_rule": parsed.get("correction_rule"),
            "trigger_pattern": parsed.get("trigger_pattern"),
            "concise_memory": parsed.get("concise_memory"),
        }
        lesson_bank.append(lesson_entry)

    bank_path = warmup_report_dir / "warmup_lesson_bank.json"
    bank_jsonl_path = warmup_report_dir / "warmup_lesson_bank.jsonl"
    _write_json(bank_path, {"teacher_cards": teacher_lines, "lesson_count": len(lesson_bank), "lessons": lesson_bank})
    with bank_jsonl_path.open("w", encoding="utf-8") as f:
        for row in lesson_bank:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _append_trader_lessons(lesson_bank)

    distilled_system = (
        "You are distilling a lesson bank into reusable trader lessons. "
        "Use the prior lesson memory and the new 50 case lessons. "
        "Return exactly 12 lines named L01: to L12:. Each line must be concrete, feature-aware, and non-generic."
    )
    distilled_user = json.dumps(
        {
            "teacher_cards": teacher_lines,
            "old_lesson_memory": prior_lines,
            "new_case_memories": [entry["concise_memory"] for entry in lesson_bank if entry.get("concise_memory")],
        },
        ensure_ascii=False,
    )
    distilled_response = _chat_completion(
        messages=[{"role": "system", "content": distilled_system}, {"role": "user", "content": distilled_user}],
        api_key=_apprentice_api_key(),
        model=config.api_model,
        max_tokens=min(config.api_max_tokens, 400),
        temperature=0.0,
    )
    distilled_content = distilled_response["choices"][0]["message"].get("content", "") or ""
    lesson_lines = []
    for line in distilled_content.splitlines():
        line = line.strip()
        if re.match(r"^L\d{2}:", line):
            lesson_lines.append(line)
    if not lesson_lines:
        lesson_lines = [f"L{i+1:02d}: {entry['concise_memory']}" for i, entry in enumerate(lesson_bank[:12]) if entry.get("concise_memory")]
    artifact = {
        "teacher_cards": teacher_lines,
        "prior_lesson_memory": prior_lines,
        "distilled_reflection_raw": distilled_content,
        "lessons": lesson_lines,
        "lesson_bank_path": str(bank_path),
    }
    _write_json(warmup_report_dir / "warmup_lessons.json", artifact)
    (warmup_report_dir / "warmup_lessons.md").write_text("\n".join(["# Warmup Lessons", "", *lesson_lines]) + "\n", encoding="utf-8")
    return lesson_lines


def _run_replay(
    *,
    config: ApprenticeReplayConfig,
    candidate_pool_df: pd.DataFrame,
    teacher_target_df: pd.DataFrame,
    teacher_full_df: pd.DataFrame,
    prompt_builder,
    prompt_builder_kwargs: Dict[str, Any],
) -> ReplaySummary:
    api_key = _apprentice_api_key()

    report_dir = REPORT_ROOT / config.run_id()
    api_cache_dir = report_dir / "api_calls"
    report_dir.mkdir(parents=True, exist_ok=True)
    api_cache_dir.mkdir(parents=True, exist_ok=True)
    _progress(
        "replay start "
        f"run_id={config.run_id()} mode={config.mode} days={candidate_pool_df['signal_date'].nunique()} "
        f"rows={len(candidate_pool_df)} model={config.api_model}"
    )

    selected_rows = pd.DataFrame(columns=list(candidate_pool_df.columns))
    llm_decisions: Dict[pd.Timestamp, List[str]] = {}
    teacher_decisions: Dict[pd.Timestamp, List[str]] = {}
    selected_parts: List[pd.DataFrame] = []
    api_calls = 0
    api_cache_hits = 0
    parse_fallback_days = 0
    parse_failure_days = 0
    query_invoked_days = 0
    query_success_days = 0
    abstain_days = 0
    retry_invoked_days = 0
    retry_success_days = 0

    prompt_log_rows = []
    day_groups = [
        (pd.Timestamp(decision_date), day_candidates.copy().reset_index(drop=True))
        for decision_date, day_candidates in candidate_pool_df.groupby("signal_date", sort=True)
    ]
    teacher_target_by_date = {
        pd.Timestamp(decision_date): group["symbol"].astype(str).tolist()
        for decision_date, group in teacher_target_df.groupby("signal_date", sort=True)
    }

    if config.ignore_holdings_context and config.api_parallel_workers > 1:
        (
            selected_rows,
            llm_decisions,
            teacher_decisions,
            prompt_log_rows,
            parallel_stats,
        ) = _run_parallel_wave_replay(
            day_groups=day_groups,
            teacher_target_by_date=teacher_target_by_date,
            config=config,
            prompt_builder=prompt_builder,
            prompt_builder_kwargs=prompt_builder_kwargs,
            api_key=api_key,
            api_cache_dir=api_cache_dir,
            candidate_pool_df=candidate_pool_df,
        )
        api_calls = int(parallel_stats["api_calls"])
        api_cache_hits = int(parallel_stats["api_cache_hits"])
        parse_fallback_days = int(parallel_stats["parse_fallback_days"])
        parse_failure_days = int(parallel_stats["parse_failure_days"])
        query_invoked_days = int(parallel_stats["query_invoked_days"])
        query_success_days = int(parallel_stats["query_success_days"])
        abstain_days = int(parallel_stats["abstain_days"])
        retry_invoked_days = int(parallel_stats["retry_invoked_days"])
        retry_success_days = int(parallel_stats["retry_success_days"])
    else:
        for decision_date, day_candidates in day_groups:
            day_candidates = day_candidates.copy().reset_index(drop=True)
            day_candidates["_candidate_id"] = [f"C{i:02d}" for i in range(1, len(day_candidates) + 1)]
            candidate_id_map = dict(zip(day_candidates["_candidate_id"], day_candidates["symbol"]))
            system, user_content = prompt_builder(
                config=config,
                decision_date=decision_date,
                candidate_df=day_candidates,
                selected_rows=selected_rows,
                **prompt_builder_kwargs,
            )
            request_payload = {
                "model": config.api_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": config.api_max_tokens,
                "temperature": config.api_temperature,
            }
            initial_prefix = _initial_line_answer_prefix(config)
            if initial_prefix is not None:
                request_payload["messages"].append(initial_prefix)
            cache_path = api_cache_dir / f"{decision_date.strftime('%Y%m%d')}.json"
            if config.reuse_api_cache and cache_path.exists():
                cached = _load_json(cache_path)
                response_payload = cached["response"]
                api_cache_hits += 1
            else:
                response_payload = _chat_completion(
                    messages=request_payload["messages"],
                    api_key=api_key,
                    model=config.api_model,
                    max_tokens=config.api_max_tokens,
                    temperature=config.api_temperature,
                    force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                )
                _serialize_daily_response(cache_path, request_payload=request_payload, response_payload=response_payload)
                api_calls += 1

            content = response_payload["choices"][0]["message"].get("content", "") or ""
            parse_result = _parse_model_selection_reply(
                content=content,
                model=config.api_model,
                allowed_symbols=day_candidates["symbol"].tolist(),
                limit=config.llm_max_daily_picks,
                candidate_id_map=candidate_id_map,
            )
            selected_symbols = list(parse_result["selected_symbols"])
            parse_fallback = bool(parse_result["parse_failed"])
            abstain = bool(parse_result["abstain"])
            parse_mode = str(parse_result["parse_mode"])
            failure_reason = str(parse_result.get("failure_reason", ""))
            query_invoked = False
            query_success = False
            retry_invoked = False
            retry_success = False
            if abstain:
                abstain_days += 1
            if parse_fallback:
                retry_invoked = True
                retry_invoked_days += 1
                parse_fallback_days += 1

            llm_decisions[decision_date] = selected_symbols
            teacher_target_symbols = teacher_target_by_date.get(decision_date, [])
            teacher_decisions[decision_date] = teacher_target_symbols

            if parse_fallback and _prefers_line_id_protocol(config.api_model):
                if config.mode == "single":
                    retry_system, retry_user = _retry_prompt_single(
                        decision_date=decision_date,
                        meta=prompt_builder_kwargs["meta"],
                        candidate_df=day_candidates,
                    )
                else:
                    retry_system, retry_user = _retry_prompt_multi(
                        decision_date=decision_date,
                        metas=prompt_builder_kwargs["metas"],
                        prompt_features=prompt_builder_kwargs["prompt_features"],
                        candidate_df=day_candidates,
                    )
                retry_request = {
                    "model": config.api_model,
                    "messages": [
                        {"role": "system", "content": retry_system},
                        {"role": "user", "content": retry_user},
                        {"role": "assistant", "content": "C01,C04,C09,C11"},
                    ],
                    "max_tokens": _retry_completion_budget(config),
                    "temperature": config.api_temperature,
                }
                try:
                    retry_response = _chat_completion(
                        messages=retry_request["messages"],
                        api_key=api_key,
                        model=config.api_model,
                        max_tokens=retry_request["max_tokens"],
                        temperature=config.api_temperature,
                        force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                    )
                    api_calls += 1
                    _write_json(
                        cache_path,
                        {
                            "request": request_payload,
                            "response": response_payload,
                            "retry_request": retry_request,
                            "retry_response": retry_response,
                        },
                    )
                except Exception as exc:
                    _write_json(
                        cache_path,
                        {
                            "request": request_payload,
                            "response": response_payload,
                            "retry_request": retry_request,
                            "retry_error": str(exc),
                        },
                    )
                    retry_response = None
                    failure_reason = (failure_reason + " | " if failure_reason else "") + f"retry_api_error={exc}"
                if retry_response is not None:
                    retry_content = retry_response["choices"][0]["message"].get("content", "") or ""
                    content = f"{content}\n[retry] {retry_content}"
                    retry_parse_result = _parse_model_selection_reply(
                        content=retry_content,
                        model=config.api_model,
                        allowed_symbols=day_candidates["symbol"].tolist(),
                        limit=config.llm_max_daily_picks,
                        candidate_id_map=candidate_id_map,
                    )
                    if (retry_parse_result["selected_symbols"] or retry_parse_result["abstain"]) and not retry_parse_result["parse_failed"]:
                        retry_success = True
                        retry_success_days += 1
                        parse_fallback_days -= 1
                        selected_symbols = list(retry_parse_result["selected_symbols"])
                    parse_fallback = bool(retry_parse_result["parse_failed"])
                    abstain = bool(retry_parse_result["abstain"])
                    parse_mode = str(retry_parse_result["parse_mode"])
                    failure_reason = str(retry_parse_result.get("failure_reason", ""))
                    if abstain:
                        abstain_days += 1
                else:
                    parse_mode = str(retry_parse_result["parse_mode"])
                    failure_reason = str(retry_parse_result.get("failure_reason", ""))
                    parse_failure_days += 1
            elif parse_fallback:
                parse_failure_days += 1

            day_selected = _select_day_rows_by_symbols(day_candidates, selected_symbols)
            if not day_selected.empty:
                selected_parts.append(day_selected)
                selected_rows = pd.concat(selected_parts, ignore_index=True)

            prompt_log_rows.append(
                {
                    "decision_date": decision_date.strftime("%Y-%m-%d"),
                    "candidate_count": len(day_candidates),
                    "teacher_target_count": len(teacher_target_symbols),
                    "llm_selected_count": len(selected_symbols),
                    "abstain": int(abstain),
                    "parse_fallback": int(parse_fallback),
                    "feature_query_invoked": int(query_invoked),
                    "feature_query_success": int(query_success),
                    "retry_invoked": int(retry_invoked),
                    "retry_success": int(retry_success),
                    "parse_mode": parse_mode,
                    "failure_reason": failure_reason,
                    "selected_symbols": ",".join(selected_symbols),
                    "teacher_target_symbols": ",".join(teacher_target_symbols),
                    "brief_reason": content[:240],
                }
            )

    llm_selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=candidate_pool_df.columns)
    return _finalize_replay_outputs(
        config=config,
        report_dir=report_dir,
        api_cache_dir=api_cache_dir,
        candidate_pool_df=candidate_pool_df,
        teacher_target_df=teacher_target_df,
        teacher_full_df=teacher_full_df,
        llm_selected_df=llm_selected_df,
        llm_decisions=llm_decisions,
        teacher_decisions=teacher_decisions,
        prompt_log_rows=prompt_log_rows,
        api_calls=api_calls,
        api_cache_hits=api_cache_hits,
        parse_fallback_days=parse_fallback_days,
        parse_failure_days=parse_failure_days,
        query_invoked_days=query_invoked_days,
        query_success_days=query_success_days,
        abstain_days=abstain_days,
        retry_invoked_days=retry_invoked_days,
        retry_success_days=retry_success_days,
    )


def run_single_teacher_replay(config: ApprenticeReplayConfig) -> ReplaySummary:
    if config.mode != "single":
        raise ValueError("single-teacher replay requires config.mode='single'")
    master_df = _load_master_dataset()
    merged, meta = _build_single_teacher_frame(config, master_df)
    negative_metas = [_negative_teacher_meta(round_id, max(4, config.prompt_feature_count // 2)) for round_id in config.negative_teacher_round_ids]
    if config.summary_variant == "enriched_v2":
        meta["preference_bands"] = _derive_preference_bands(
            round_id=meta["round_id"],
            master_df=master_df,
            feature_cols=meta["top_prompt_features"],
            start_date=config.warmup_start_date,
            end_date=config.warmup_end_date,
        )
    candidate_pool_df, teacher_target_df = _single_teacher_target(merged, config)
    teacher_full_df = merged[merged["bucket"] == 5].copy()
    return _run_replay(
        config=config,
        candidate_pool_df=candidate_pool_df,
        teacher_target_df=teacher_target_df,
        teacher_full_df=teacher_full_df,
        prompt_builder=_daily_prompt_single,
        prompt_builder_kwargs={"meta": meta, "negative_metas": negative_metas, "warmup_lessons": []},
    )


def _compute_multi_teacher_warmup_state(
    *, config: ApprenticeReplayConfig, master_df: pd.DataFrame, negative_metas: Sequence[Dict[str, Any]]
) -> Tuple[List[str], List[str], Optional[Dict[str, Any]], Dict[str, Any], ApprenticeReplayConfig]:
    warmup_teacher_round_ids = [_source_round_id_for_round(round_id) for round_id in config.teacher_round_ids]
    warm_cfg = replace(
        config,
        teacher_round_ids=warmup_teacher_round_ids,
        start_date=config.warmup_start_date,
        end_date=config.warmup_end_date,
        warmup_sample_count=0,
        run_tag=f"{config.run_tag}_warmup{config.warmup_sample_count}",
    )
    warmup_lessons: List[str] = []
    warmup_review_cards: List[str] = []
    scoped_warmup_state: Optional[Dict[str, Any]] = None
    if config.warmup_curriculum == "scorefit_v1_json":
        warmup_state = _generate_warmup_lessons_scorefit_v1_json(
            config=config,
            warm_cfg=warm_cfg,
            master_df=master_df,
        )
        scoped_warmup_state = warmup_state
        warmup_lessons = list(warmup_state.get("global_lesson_zone_lines", []))
        warmup_review_cards = list(warmup_state.get("review_cards_for_prompt", []))
        _progress(
            "scorefit_v1_json warmup complete "
            f"scope_count={len(_teacher_scope_entries(scoped_warmup_state))}"
        )
        return warmup_lessons, warmup_review_cards, scoped_warmup_state, warmup_state, warm_cfg
    if config.warmup_curriculum == "iterative_v5_scoped":
        warmup_state = _generate_warmup_lessons_iterative_v5_scoped(
            config=config,
            warm_cfg=warm_cfg,
            master_df=master_df,
        )
        scoped_warmup_state = warmup_state
        warmup_lessons = list(warmup_state.get("global_lesson_zone_lines", []))
        warmup_review_cards = list(warmup_state.get("review_cards_for_prompt", []))
        _progress(
            "iterative_v5_scoped warmup complete "
            f"global_lines={len(warmup_lessons)} scope_count={len(_teacher_scope_entries(scoped_warmup_state))} "
            f"review_cards={len(warmup_review_cards)}"
        )
        return warmup_lessons, warmup_review_cards, scoped_warmup_state, warmup_state, warm_cfg
    if config.warmup_curriculum in {"iterative_v6_scoped_interval", "iterative_v7_scoped_rulecard", "iterative_v8_weighted_soft_rules"}:
        warmup_state = _generate_warmup_lessons_iterative_v6_scoped_interval(
            config=config,
            warm_cfg=warm_cfg,
            master_df=master_df,
        )
        scoped_warmup_state = warmup_state
        warmup_lessons = list(warmup_state.get("global_lesson_zone_lines", []))
        warmup_review_cards = list(warmup_state.get("review_cards_for_prompt", []))
        _progress(
            f"{config.warmup_curriculum} warmup complete "
            f"global_lines={len(warmup_lessons)} scope_count={len(_teacher_scope_entries(scoped_warmup_state))} "
            f"review_cards={len(warmup_review_cards)}"
        )
        return warmup_lessons, warmup_review_cards, scoped_warmup_state, warmup_state, warm_cfg
    warm_merged, warm_metas, warm_prompt_features = _build_multi_teacher_frame(warm_cfg, master_df)
    _progress(
        "warmup frame built "
        f"rows={len(warm_merged)} teachers={len(warm_metas)} prompt_features={len(warm_prompt_features)}"
    )
    if config.summary_variant == "enriched_v2":
        for meta in warm_metas:
            meta["preference_bands"] = _derive_preference_bands(
                round_id=meta["round_id"],
                master_df=master_df,
                feature_cols=meta["top_prompt_features"],
                start_date=config.warmup_start_date,
                end_date=config.warmup_end_date,
            )
    warm_pool, warm_target = _multi_teacher_target(warm_merged, warm_cfg)
    sample_dates = _sample_uniform_dates(
        warm_pool["signal_date"].tolist(),
        config.warmup_sample_count,
        sample_seed=config.sample_seed,
    )
    warm_pool = warm_pool[warm_pool["signal_date"].isin(sample_dates)].copy()
    warm_target = warm_target[warm_target["signal_date"].isin(sample_dates)].copy()
    warm_full = warm_target.copy()
    _progress(
        "warmup pool sampled "
        f"sampled_days={warm_pool['signal_date'].nunique()} pool_rows={len(warm_pool)} target_rows={len(warm_target)}"
    )
    if config.warmup_curriculum == "iterative_v4":
        warmup_state = _generate_warmup_lessons_iterative_v4(
            config=config,
            warm_cfg=warm_cfg,
            metas=warm_metas,
            negative_metas=negative_metas,
            prompt_features=warm_prompt_features,
            candidate_pool_df=warm_pool,
            teacher_target_df=warm_target,
        )
        warmup_lessons = list(warmup_state.get("lesson_zone_lines", []))
        warmup_review_cards = list(warmup_state.get("review_cards_for_prompt", []))
        _progress(
            "iterative_v4 warmup complete "
            f"lesson_lines={len(warmup_lessons)} review_cards={len(warmup_review_cards)}"
        )
    elif config.warmup_curriculum == "iterative_v3":
        warmup_state = _generate_warmup_lessons_iterative(
            config=config,
            warm_cfg=warm_cfg,
            metas=warm_metas,
            negative_metas=negative_metas,
            prompt_features=warm_prompt_features,
            candidate_pool_df=warm_pool,
            teacher_target_df=warm_target,
        )
        warmup_lessons = list(warmup_state.get("lesson_zone_lines", []))
        warmup_review_cards = list(warmup_state.get("review_cards_for_prompt", []))
        _progress(
            "iterative_v3 warmup complete "
            f"lesson_lines={len(warmup_lessons)} review_cards={len(warmup_review_cards)}"
        )
    else:
        _run_replay(
            config=warm_cfg,
            candidate_pool_df=warm_pool,
            teacher_target_df=warm_target,
            teacher_full_df=warm_full,
            prompt_builder=_daily_prompt_multi,
            prompt_builder_kwargs={
                "metas": warm_metas,
                "negative_metas": negative_metas,
                "prompt_features": warm_prompt_features,
                "warmup_lessons": [],
                "warmup_review_cards": [],
            },
        )
        warmup_lessons = _generate_warmup_lessons(
            config=config,
            warmup_report_dir=REPORT_ROOT / warm_cfg.run_id(),
            metas=warm_metas,
            prompt_features=warm_prompt_features,
            candidate_pool_df=warm_pool,
            teacher_target_df=warm_target,
        )
        warmup_state = {"lesson_zone_lines": warmup_lessons, "review_cards_for_prompt": []}
        _progress(f"legacy warmup complete lesson_lines={len(warmup_lessons)}")
    return warmup_lessons, warmup_review_cards, scoped_warmup_state, warmup_state, warm_cfg


def run_multi_teacher_scoped_warmup(config: ApprenticeReplayConfig) -> Dict[str, Any]:
    if config.mode != "multi":
        raise ValueError("scoped warmup helper requires config.mode='multi'")
    if config.warmup_curriculum not in {"iterative_v5_scoped", "iterative_v6_scoped_interval", "iterative_v7_scoped_rulecard", "iterative_v8_weighted_soft_rules", "scorefit_v1_json"}:
        raise ValueError("run_multi_teacher_scoped_warmup requires scoped warmup curriculum")
    _progress(
        "multi scoped warmup bootstrap "
        f"teachers={len(config.teacher_round_ids)} warmup_sample_count={config.warmup_sample_count}"
    )
    master_df = _load_master_dataset()
    _progress(f"master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")
    negative_metas = [_negative_teacher_meta(round_id, max(4, config.prompt_feature_count // 2)) for round_id in config.negative_teacher_round_ids]
    warmup_lessons, warmup_review_cards, scoped_warmup_state, warmup_state, warm_cfg = _compute_multi_teacher_warmup_state(
        config=config,
        master_df=master_df,
        negative_metas=negative_metas,
    )
    summary = {
        "run_id": warm_cfg.run_id(),
        "curriculum": config.warmup_curriculum,
        "teacher_round_ids": list(config.teacher_round_ids),
        "source_teacher_round_ids": list(warm_cfg.teacher_round_ids),
        "warmup_sample_count": int(config.warmup_sample_count),
        "warmup_signal_pool_per_day": int(config.warmup_signal_pool_per_day),
        "sample_seed": int(config.sample_seed),
        "global_lesson_count": len(warmup_lessons),
        "scope_count": len(_teacher_scope_entries(scoped_warmup_state)),
        "review_cards_for_prompt_count": len(warmup_review_cards),
        "artifact_dir": warmup_state.get("artifact_dir", _relative(REPORT_ROOT / warm_cfg.run_id())),
    }
    report_dir = REPORT_ROOT / warm_cfg.run_id()
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "WARMUP_ONLY_SUMMARY.json", summary)
    return summary


def run_multi_teacher_replay(config: ApprenticeReplayConfig) -> ReplaySummary:
    if config.mode != "multi":
        raise ValueError("multi-teacher replay requires config.mode='multi'")
    _progress(
        "multi replay bootstrap "
        f"run_id={config.run_id()} warmup_curriculum={config.warmup_curriculum} "
        f"warmup_sample_count={config.warmup_sample_count}"
    )
    bundle = _load_or_build_multi_teacher_bundle(config)
    metas = list(bundle["metas"])
    prompt_features = list(bundle["prompt_features"])
    negative_metas = list(bundle["negative_metas"])
    candidate_pool_df = bundle["candidate_pool_df"].copy()
    teacher_target_df = bundle["teacher_target_df"].copy()
    teacher_full_df = bundle["teacher_full_df"].copy()
    _progress(
        "multi replay bundle ready "
        f"pool_rows={len(candidate_pool_df)} target_rows={len(teacher_target_df)} "
        f"decision_days={candidate_pool_df['signal_date'].nunique()} "
        f"teachers={len(metas)} prompt_features={len(prompt_features)}"
    )
    warmup_lessons: List[str] = []
    warmup_review_cards: List[str] = []
    scoped_warmup_state: Optional[Dict[str, Any]] = None
    warmup_override = _load_warmup_state_override(config.warmup_state_json)
    if warmup_override is not None:
        scoped_warmup_state = warmup_override
        warmup_lessons = list(warmup_override.get("global_lesson_zone_lines", []))
        warmup_review_cards = list(warmup_override.get("review_cards_for_prompt", []))
        _progress(
            "warmup state override loaded "
            f"path={config.warmup_state_json} global_lines={len(warmup_lessons)} "
            f"scope_count={len(_teacher_scope_entries(scoped_warmup_state))} "
            f"review_cards={len(warmup_review_cards)}"
        )
    elif config.warmup_sample_count > 0:
        master_df = _load_master_dataset()
        _progress(f"warmup master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")
        warmup_lessons, warmup_review_cards, scoped_warmup_state, _, _ = _compute_multi_teacher_warmup_state(
            config=config,
            master_df=master_df,
            negative_metas=negative_metas,
        )
    _progress(
        "final replay dispatch "
        f"run_id={config.run_id()} warmup_lessons={len(warmup_lessons)} warmup_review_cards={len(warmup_review_cards)}"
    )
    return _run_replay(
        config=config,
        candidate_pool_df=candidate_pool_df,
        teacher_target_df=teacher_target_df,
        teacher_full_df=teacher_full_df,
        prompt_builder=_daily_prompt_multi,
        prompt_builder_kwargs={
            "metas": metas,
            "negative_metas": negative_metas,
            "prompt_features": prompt_features,
            "warmup_lessons": warmup_lessons,
            "warmup_review_cards": warmup_review_cards,
            "scoped_warmup_state": scoped_warmup_state,
        },
    )
