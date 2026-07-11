from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .local_service import describe_local_service_status, ensure_local_service_if_configured, local_api_url_if_enabled
from .paths import studio_root
from .provenance import read_json, write_json
from .registry import StudioRegistry


def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


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


def _template_desc(template: str) -> str:
    table = {
        "trend_breakout_pool": "Breakout-continuation candidates with trend expansion context.",
        "trend_pullback_pool": "Trend pullback candidates that try to resume an existing move after a controlled retracement.",
        "weak_state_reversal_pool": "Weak-state reversal candidates that require rebound confirmation after stress.",
        "hard_threshold_reversal_gate": "Hard-gated oversold reversal candidates with stricter rejection conditions.",
    }
    return table.get(str(template or "").strip(), str(template or "").strip())


def _style_label(sample_template: str, family: str, hypothesis: str) -> str:
    template = str(sample_template or "").strip().lower()
    family_text = str(family or "").strip().lower()
    hypo_text = str(hypothesis or "").strip().lower()
    merged = " ".join([template, family_text, hypo_text])
    if "breakout" in merged:
        return "breakout continuation"
    if "pullback" in merged and "reversal" in merged:
        return "pullback rebound inside trend"
    if "pullback" in merged:
        return "trend pullback continuation"
    if "reversal" in merged:
        return "reversal / rebound"
    return "mixed regime ranking"


def _compact_feature_cues(factor_summary: Mapping[str, Any], top_n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(factor_summary.get("top_global_features") or [])[:top_n]:
        feature = str(row.get("feature", "")).strip()
        if not feature:
            continue
        out.append(
            {
                "feature": feature,
                "preferred_direction": str(row.get("preferred_direction", "")).strip(),
                "shape_hint": str(row.get("shape_hint", "")).strip(),
            }
        )
    return out


def _compact_lesson_for_prompt(
    lesson_json: Mapping[str, Any],
    *,
    max_meta_rules: int,
    max_scoring_notes: int,
) -> Dict[str, Any]:
    items = {}
    for item_id, payload in dict(lesson_json.get("items") or {}).items():
        items[str(item_id)] = {
            "title": str(payload.get("title", "")).strip(),
            "role": str(payload.get("role", "")).strip(),
            "score_range": str(payload.get("score_range", "")).strip(),
            "signals_to_check": [str(x).strip() for x in list(payload.get("signals_to_check") or []) if str(x).strip()],
            "rule": str(payload.get("rule", "")).strip(),
            "interaction_note": str(payload.get("interaction_note", "")).strip(),
        }
    return {
        "schema": str(lesson_json.get("schema", "")).strip(),
        "lesson_name": str(lesson_json.get("lesson_name", "")).strip(),
        "teacher_scope": dict(lesson_json.get("teacher_scope") or {}),
        "items": items,
        "meta_rules": [str(x).strip() for x in list(lesson_json.get("meta_rules") or []) if str(x).strip()][:max_meta_rules],
        "scoring_notes": [str(x).strip() for x in list(lesson_json.get("scoring_notes") or []) if str(x).strip()][:max_scoring_notes],
    }


def _clip_digest_text(text: str, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _is_gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in str(model or "").lower()


def _normalize_api_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    if "/v1/" in trimmed:
        if trimmed.endswith("/chat"):
            return f"{trimmed}/completions"
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def _redact_secret(secret: str) -> str:
    text = str(secret or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}***{text[-4:]}"


def _signal_prompt(
    *,
    digest_text: str,
    teacher_cards: Sequence[Mapping[str, Any]],
    teacher_lessons: Mapping[str, Any],
    signal_record: Mapping[str, Any],
    reasoning_target_tokens: int,
    reasoning_max_tokens: int,
    decision_seed: int,
) -> Tuple[str, str]:
    reasoning_parts: List[str] = []
    if reasoning_target_tokens > 0 and reasoning_max_tokens > 0:
        reasoning_parts.append(
            f"Keep your private scratchpad compact: aim to finish within roughly {reasoning_target_tokens} tokens and stop refining by around {reasoning_max_tokens} tokens."
        )
    elif reasoning_max_tokens > 0:
        reasoning_parts.append(
            f"Keep your private scratchpad compact and stop refining by around {reasoning_max_tokens} tokens."
        )
    reasoning_parts.append("Trading is a fuzzy art, not a proof.")
    reasoning_parts.append("If several candidates are inside a reasonable zone, choose the best approximate set and let expected value work.")
    reasoning_parts.append("Do not over-optimize tiny conflicts or chase perfect certainty.")
    reasoning_parts.append("When the evidence is good enough, act; when it clearly fails, abstain.")
    reasoning_instruction = " ".join(reasoning_parts)

    has_lessons = bool(teacher_lessons)
    lesson_instruction = (
        "Use the final lessons as teacher-local scoring guides when the signal matches or partially matches that teacher's style. "
        "If a signal falls outside every teacher's filter, generalize softly and avoid fake precision. "
    ) if has_lessons else (
        "No teacher-local final lessons are available in this run. "
        "Rely only on the simplified hypothesis-validation digest, the teacher basic infos, and the raw signal features. "
        "If a signal falls outside every teacher's filter, score conservatively and avoid inventing teacher-local scoring detail. "
    )
    system = (
        "You are independently scoring one market signal for 5-trading-day expected value. "
        f"{reasoning_instruction} "
        "Use the simplified hypothesis-validation digest only as background about which styles were easy or hard to stratify. "
        "Use the teacher basic infos to understand each teacher's cluster, filter, style, and historical strength. "
        f"{lesson_instruction}"
        "Scoring scale: 0..50 means expected value is negative, 50..100 means expected value is positive. "
        "Anchor the scale approximately as: 25 ~ around -2% expected 5-day return, 50 ~ near flat, 75 ~ around +2% expected 5-day return. "
        "Return strict JSON only with keys: total_score, score_60d, score_120d, window_score_note, short_reason, teacher_scores. "
        "total_score must be a number from 0 to 100. "
        "score_60d must be a number from 0 to 100 representing the short-window view using recent 60-trading-day evidence. "
        "score_120d must be a number from 0 to 100 representing the medium-window view using recent 120-trading-day evidence. "
        "If the signal does not contain explicitly separated 60d/120d feature contexts, infer both from the provided feature snapshot, keep them reasonably close to total_score, and explain that limitation in window_score_note. "
        "teacher_scores must be an array with one object per teacher basic info, each using keys round_id, title, score, note. "
        "Each teacher-local score should represent how strongly this signal fits that teacher's local comfort zone on a 0..100 scale. "
    )
    user = json.dumps(
        {
            "simplified_hypothesis_validation_digest": digest_text,
            "teacher_basic_infos": list(teacher_cards),
            "teacher_final_lessons": dict(teacher_lessons),
            "decision_seed": int(decision_seed or 0),
            "tie_break_note": (
                "If several final scores feel nearly equal, use decision_seed only as a deterministic tie-breaker. "
                "Do not mention the seed in the answer."
            ),
            "signal": signal_record,
        },
        ensure_ascii=False,
    )
    return system, user


def _scorefit_parse_signal_reply(content: str) -> Dict[str, Any]:
    payload = _extract_json_payload(content)
    total_value = float(payload.get("total_score", payload.get("score", 0.0)) or 0.0)
    score_60d = float(payload.get("score_60d", total_value) or total_value)
    score_120d = float(payload.get("score_120d", total_value) or total_value)
    teacher_scores: List[Dict[str, Any]] = []
    for row in list(payload.get("teacher_scores") or []):
        if not isinstance(row, dict):
            continue
        teacher_scores.append(
            {
                "round_id": str(row.get("round_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "score": float(row.get("score", 0.0) or 0.0),
                "note": str(row.get("note", "")).strip(),
            }
        )
    return {
        "total_score": total_value,
        "score_60d": score_60d,
        "score_120d": score_120d,
        "window_score_note": str(payload.get("window_score_note", "")).strip(),
        "short_reason": str(payload.get("short_reason", payload.get("reason", ""))).strip(),
        "teacher_scores": teacher_scores,
        "parsed_payload": payload,
    }


@dataclass
class LiveModelConfig:
    api_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    reasoning_target_tokens: int
    reasoning_max_tokens: int
    prompt_digest_max_chars: int
    prompt_top_feature_count: int
    prompt_max_meta_rules: int
    prompt_max_scoring_notes: int
    decision_seed: int


def load_live_model_config() -> LiveModelConfig:
    raw_api_url = (
        os.environ.get("QA_STUDIO_API_URL", "")
        or os.environ.get("QA_LIVE_MODEL_API_URL", "")
        or os.environ.get("APPRENTICE_API_URL", "")
        or os.environ.get("LLM_API_URL", "")
    ).strip()
    api_key = (
        os.environ.get("QA_STUDIO_API_KEY", "")
        or os.environ.get("QA_LIVE_MODEL_API_KEY", "")
        or os.environ.get("APPRENTICE_API_KEY", "")
        or os.environ.get("CHATANYWHERE_API_KEY", "")
    ).strip()
    model = (
        os.environ.get("QA_STUDIO_LIVE_MODEL", "")
        or os.environ.get("QA_LIVE_MODEL_NAME", "")
        or "gpt-oss-20b"
    ).strip()
    normalized_api_url = _normalize_api_url(raw_api_url) or local_api_url_if_enabled()
    return LiveModelConfig(
        api_url=normalized_api_url,
        api_key=api_key,
        model=model,
        temperature=float(os.environ.get("QA_STUDIO_LIVE_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("QA_STUDIO_LIVE_MAX_TOKENS", "1024")),
        timeout_seconds=float(os.environ.get("QA_STUDIO_LIVE_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.environ.get("QA_STUDIO_LIVE_MAX_RETRIES", "3")),
        reasoning_target_tokens=int(os.environ.get("QA_STUDIO_REASONING_TARGET_TOKENS", "1600")),
        reasoning_max_tokens=int(os.environ.get("QA_STUDIO_REASONING_MAX_TOKENS", "2000")),
        prompt_digest_max_chars=int(os.environ.get("QA_STUDIO_PROMPT_DIGEST_MAX_CHARS", "3200")),
        prompt_top_feature_count=int(os.environ.get("QA_STUDIO_PROMPT_TOP_FEATURE_COUNT", "5")),
        prompt_max_meta_rules=int(os.environ.get("QA_STUDIO_PROMPT_MAX_META_RULES", "4")),
        prompt_max_scoring_notes=int(os.environ.get("QA_STUDIO_PROMPT_MAX_SCORING_NOTES", "4")),
        decision_seed=int(os.environ.get("QA_STUDIO_DECISION_SEED", "20250705")),
    )


def describe_live_model_config() -> Dict[str, Any]:
    config = load_live_model_config()
    payload = {
        "api_url": config.api_url,
        "api_url_configured": bool(config.api_url),
        "api_key_configured": bool(config.api_key),
        "api_key_preview": _redact_secret(config.api_key),
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "reasoning_target_tokens": config.reasoning_target_tokens,
        "reasoning_max_tokens": config.reasoning_max_tokens,
        "prompt_digest_max_chars": config.prompt_digest_max_chars,
        "prompt_top_feature_count": config.prompt_top_feature_count,
        "prompt_max_meta_rules": config.prompt_max_meta_rules,
        "prompt_max_scoring_notes": config.prompt_max_scoring_notes,
        "decision_seed": config.decision_seed,
    }
    try:
        payload["local_service"] = describe_local_service_status()
    except Exception as exc:
        payload["local_service"] = {"error": str(exc)}
    return payload


def _http_chat_completion(
    *,
    config: LiveModelConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    if not config.api_url.strip():
        raise RuntimeError("Live API URL is not configured. Set QA_STUDIO_API_URL or APPRENTICE_API_URL.")
    request_body: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": int(config.max_tokens),
        "temperature": float(config.temperature),
        "top_p": 1.0,
    }
    if config.model.startswith("gpt-"):
        request_body["reasoning_effort"] = "low" if _is_gpt_oss_model(config.model) else "minimal"
        request_body["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    retryable = {429, 502, 503, 504}
    body = json.dumps(request_body).encode("utf-8")
    last_error: Optional[Exception] = None
    for attempt in range(max(1, int(config.max_retries))):
        try:
            req = Request(config.api_url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=float(config.timeout_seconds)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choice0 = (payload.get("choices") or [{}])[0]
            message0 = choice0.get("message", {}) or {}
            content = str(message0.get("content", "") or "").strip()
            if not content:
                finish_reason = str(choice0.get("finish_reason", "") or "").strip()
                reasoning = str(message0.get("reasoning_content", "") or "")
                usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
                raise RuntimeError(
                    "APIEmptyContent "
                    f"model={config.model} "
                    f"finish_reason={finish_reason or 'unknown'} "
                    f"reasoning_len={len(reasoning)} "
                    f"usage={json.dumps(usage, ensure_ascii=False)}"
                )
            return payload
        except HTTPError as exc:
            code = getattr(exc, "code", None)
            text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            last_error = RuntimeError(f"HTTPError {code}: {text}")
            if code not in retryable or attempt >= int(config.max_retries) - 1:
                raise last_error
        except URLError as exc:
            last_error = exc
            if attempt >= int(config.max_retries) - 1:
                raise
        except Exception as exc:
            last_error = exc
            if attempt >= int(config.max_retries) - 1:
                raise
        time.sleep(min(3.0 * (attempt + 1), 15.0))
    raise RuntimeError(f"Live API request failed after retries: {last_error}") from last_error


def _signal_cache_path(lesson_alias: str, signal_record: Mapping[str, Any]) -> Path:
    signal_date = str(signal_record.get("signal_date", "")).strip() or "unknown_date"
    symbol = str(signal_record.get("symbol", "")).strip() or "unknown_symbol"
    text = json.dumps(signal_record, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return studio_root() / "runs" / "live_score_cache" / lesson_alias / f"{signal_date}__{symbol}__{digest}.json"


def _infer_shared_context_root_from_lesson(path: Path) -> Optional[Path]:
    current = path.expanduser().resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "clean_context":
            return parent
    return None


def _find_current_workflow_digest(shared_context_root: Path) -> Optional[Path]:
    candidates = sorted(shared_context_root.rglob("digest_simplified.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_teacher_report_info(
    *,
    catalog: Mapping[str, Any],
    round_id: str,
    shared_context_root: Optional[Path],
) -> Tuple[Dict[str, str], str]:
    imported = dict(catalog.get("teacher_reports", {})).get(round_id)
    if imported:
        return {
            "selected_spec_json": str(imported.get("selected_spec_json", "")),
            "nav_summary_json": str(imported.get("nav_summary_json", "")),
            "factor_analysis_summary_json": str(imported.get("factor_analysis_summary_json", "")),
        }, "imported_final_asset"

    if shared_context_root is not None:
        report_dir = shared_context_root / "reports" / "teacher_loop" / round_id
        spec = report_dir / "selected_spec.json"
        nav = report_dir / "nav_summary.json"
        factor_candidates = [
            report_dir / "factor_analysis_summary.json",
            report_dir / "factor_analysis_summary_v2.json",
        ]
        factor = next((path for path in factor_candidates if path.exists()), factor_candidates[0])
        if spec.exists():
            return {
                "selected_spec_json": str(spec),
                "nav_summary_json": str(nav),
                "factor_analysis_summary_json": str(factor),
            }, "current_workflow_asset"

    raise KeyError(f"teacher report bundle not found for round_id={round_id}")


def _selection_meta_for_bundle(
    *,
    shared_context_root: Optional[Path],
) -> Dict[str, Any]:
    if shared_context_root is None:
        return {
            "selection_resolution_source": "imported_final_asset_bundle",
            "fallback_reason": "",
            "fallback_used": False,
        }
    selection_artifact = shared_context_root / "studio_control" / "selected_teacher_for_inner_loop.json"
    if not selection_artifact.exists():
        return {
            "selection_resolution_source": "current_workflow_asset",
            "fallback_reason": "",
            "fallback_used": False,
        }
    payload = read_json(selection_artifact)
    fallback_reason = str(payload.get("fallback_reason", "")).strip()
    return {
        "selection_resolution_source": str(payload.get("resolution_source", "")).strip() or "current_workflow_asset",
        "fallback_reason": fallback_reason,
        "fallback_used": bool(fallback_reason),
        "selection_artifact_json": str(selection_artifact),
    }


def build_market_runtime_bundle(
    registry: StudioRegistry,
    *,
    lesson_alias: str = "",
    final_lesson_state_json: str = "",
    live_config: Optional[LiveModelConfig] = None,
) -> Dict[str, Any]:
    config = live_config or load_live_model_config()
    catalog = registry.load_runtime_catalog()
    explicit_path = str(final_lesson_state_json or "").strip()
    lesson_alias = str(lesson_alias or "").strip()
    if explicit_path:
        lesson_path = Path(explicit_path).expanduser().resolve()
        final_state = read_json(lesson_path)
        shared_context_root = _infer_shared_context_root_from_lesson(lesson_path)
        lesson_source = "explicit_final_lesson_state_json"
        resolved_lesson_alias = lesson_alias
    else:
        lesson_run = catalog["lesson_runs"][lesson_alias]
        lesson_path = Path(lesson_run["final_lesson_state_json"]).expanduser().resolve()
        final_state = read_json(lesson_path)
        shared_context_root = _infer_shared_context_root_from_lesson(lesson_path)
        lesson_source = "imported_final_asset"
        resolved_lesson_alias = lesson_alias
    digest_path = (
        _find_current_workflow_digest(shared_context_root)
        if shared_context_root is not None
        else None
    ) or Path(catalog["digest_file"]).expanduser().resolve()
    digest_text = _clip_digest_text(digest_path.read_text(encoding="utf-8"), config.prompt_digest_max_chars)

    teacher_cards: List[Dict[str, Any]] = []
    teacher_lessons: Dict[str, Any] = {}
    teacher_sources: List[str] = []
    for scope in list(final_state.get("teacher_scopes") or []):
        round_id = str(scope.get("round_id", "")).strip()
        if not round_id:
            continue
        report_info, report_source = _resolve_teacher_report_info(
            catalog=catalog,
            round_id=round_id,
            shared_context_root=shared_context_root,
        )
        teacher_sources.append(report_source)
        spec = read_json(Path(report_info["selected_spec_json"]))
        nav_summary = read_json(Path(report_info["nav_summary_json"])) if Path(report_info["nav_summary_json"]).exists() else {}
        factor_summary = (
            read_json(Path(report_info["factor_analysis_summary_json"]))
            if Path(report_info["factor_analysis_summary_json"]).exists()
            else {}
        )
        compact_lesson = _compact_lesson_for_prompt(
            dict(scope.get("scorefit_lesson_json") or {}),
            max_meta_rules=config.prompt_max_meta_rules,
            max_scoring_notes=config.prompt_max_scoring_notes,
        )
        teacher_cards.append(
            {
                "round_id": round_id,
                "source_round_id": str(spec.get("source_round_id") or scope.get("source_round_id") or "").strip(),
                "title": str(spec.get("title", "")).strip(),
                "style_family": _style_label(
                    str(spec.get("sample_template", "")).strip(),
                    str(spec.get("research_family", "")).strip(),
                    str(spec.get("hypothesis", "")).strip(),
                ),
                "research_family": str(spec.get("research_family", "")).strip(),
                "signal_cluster": str(spec.get("sample_template", "")).strip(),
                "basic_filter": _template_desc(str(spec.get("sample_template", "")).strip()),
                "hypothesis": str(spec.get("hypothesis", "")).strip(),
                "walkforward_final_nav": round(float(nav_summary.get("final_nav", 0.0) or 0.0), 6),
                "walkforward_total_return": round(float(nav_summary.get("total_return", 0.0) or 0.0), 6),
                "walkforward_cagr": round(float(nav_summary.get("cagr", 0.0) or 0.0), 6),
                "source_type": report_source,
                "important_feature_cues": _compact_feature_cues(
                    factor_summary,
                    config.prompt_top_feature_count,
                ),
            }
        )
        teacher_lessons[round_id] = compact_lesson
    selection_meta = _selection_meta_for_bundle(shared_context_root=shared_context_root)
    return {
        "digest_text": digest_text,
        "digest_file": str(digest_path),
        "teacher_cards": teacher_cards,
        "teacher_lessons": teacher_lessons,
        "lesson_alias": resolved_lesson_alias,
        "lesson_source": lesson_source,
        "teacher_source": teacher_sources[0] if len(set(teacher_sources)) == 1 and teacher_sources else ("mixed_asset" if teacher_sources else ""),
        "teacher_sources": sorted(set(teacher_sources)),
        "final_lesson_state_json": str(lesson_path),
        "shared_context_root": str(shared_context_root) if shared_context_root is not None else "",
        **selection_meta,
    }


def build_live_prompt(
    registry: StudioRegistry,
    *,
    lesson_alias: str = "",
    final_lesson_state_json: str = "",
    signal_record: Mapping[str, Any],
    live_config: Optional[LiveModelConfig] = None,
) -> Dict[str, Any]:
    config = live_config or load_live_model_config()
    bundle = build_market_runtime_bundle(
        registry,
        lesson_alias=lesson_alias,
        final_lesson_state_json=final_lesson_state_json,
        live_config=config,
    )
    system, user = _signal_prompt(
        digest_text=bundle["digest_text"],
        teacher_cards=bundle["teacher_cards"],
        teacher_lessons=bundle["teacher_lessons"],
        signal_record=signal_record,
        reasoning_target_tokens=config.reasoning_target_tokens,
        reasoning_max_tokens=config.reasoning_max_tokens,
        decision_seed=config.decision_seed,
    )
    return {
        "system": system,
        "user": user,
        "bundle_meta": {
            "lesson_alias": lesson_alias,
            "teacher_count": len(bundle["teacher_cards"]),
            "teacher_cards": list(bundle["teacher_cards"]),
            "final_lesson_state_json": bundle["final_lesson_state_json"],
            "lesson_source": bundle["lesson_source"],
            "teacher_source": bundle["teacher_source"],
            "teacher_sources": bundle["teacher_sources"],
            "selection_resolution_source": bundle["selection_resolution_source"],
            "fallback_reason": bundle["fallback_reason"],
            "fallback_used": bundle["fallback_used"],
            "digest_file": bundle["digest_file"],
            "shared_context_root": bundle["shared_context_root"],
        },
    }


def score_live_signal(
    registry: StudioRegistry,
    *,
    lesson_alias: str = "",
    final_lesson_state_json: str = "",
    signal_record: Mapping[str, Any],
    prompt_only: bool = False,
    reuse_cache: bool = True,
    live_config: Optional[LiveModelConfig] = None,
) -> Dict[str, Any]:
    config = live_config or load_live_model_config()
    local_status = ensure_local_service_if_configured(config.api_url)
    prompt = build_live_prompt(
        registry,
        lesson_alias=lesson_alias,
        final_lesson_state_json=final_lesson_state_json,
        signal_record=signal_record,
        live_config=config,
    )
    cache_alias = lesson_alias or ("explicit_" + hashlib.sha1(str(final_lesson_state_json).encode("utf-8")).hexdigest()[:12])
    cache_path = _signal_cache_path(cache_alias, signal_record)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_only:
        return {
            "mode": "prompt_only",
            "cache_path": str(cache_path),
            "local_service_status": local_status,
            **prompt,
        }
    if reuse_cache and cache_path.exists():
        cached = read_json(cache_path)
        bundle_meta = dict(cached.get("bundle_meta") or {})
        for key, value in dict(prompt.get("bundle_meta") or {}).items():
            bundle_meta.setdefault(key, value)
        cached["bundle_meta"] = bundle_meta
        cached.setdefault("teacher_scores", list((cached.get("parsed_payload") or {}).get("teacher_scores") or []))
        cached.setdefault("score_60d", float(cached.get("total_score", 0.0) or 0.0))
        cached.setdefault("score_120d", float(cached.get("total_score", 0.0) or 0.0))
        cached.setdefault("window_score_note", "Loaded from an older cache without explicit 60d/120d window scores; both window scores default to total_score.")
        cached["cache_hit"] = True
        return cached
    response = _http_chat_completion(
        config=config,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    )
    choice0 = (response.get("choices") or [{}])[0]
    content = str(choice0.get("message", {}).get("content", "") or "")
    finish_reason = str(choice0.get("finish_reason", "") or "")
    usage = dict(response.get("usage") or {})
    try:
        parsed = _scorefit_parse_signal_reply(content)
    except Exception as exc:
        failure_payload = {
            "mode": "live_runtime_parse_error",
            "cache_hit": False,
            "cache_path": str(cache_path),
            "model": config.model,
            "api_url": config.api_url,
            "local_service_status": local_status,
            "lesson_alias": lesson_alias,
            "final_lesson_state_json": str(final_lesson_state_json or prompt["bundle_meta"]["final_lesson_state_json"]).strip(),
            "signal_record": dict(signal_record),
            "system": prompt["system"],
            "user": prompt["user"],
            "bundle_meta": dict(prompt["bundle_meta"]),
            "content": content,
            "raw_response": response,
            "finish_reason": finish_reason,
            "usage": usage,
            "model_called": True,
            "result_valid_for_research": False,
            "parse_error": str(exc),
            "parse_error_type": exc.__class__.__name__,
        }
        write_json(cache_path, failure_payload)
        raise RuntimeError(
            "Live model response could not be parsed as strict scoring JSON. "
            f"finish_reason={finish_reason or 'unknown'} cache_path={cache_path} parse_error={exc}"
        ) from exc
    payload = {
        "mode": "live_runtime",
        "cache_hit": False,
        "cache_path": str(cache_path),
        "model": config.model,
        "api_url": config.api_url,
        "local_service_status": local_status,
        "lesson_alias": lesson_alias,
        "final_lesson_state_json": str(final_lesson_state_json or prompt["bundle_meta"]["final_lesson_state_json"]).strip(),
        "signal_record": dict(signal_record),
        "system": prompt["system"],
        "user": prompt["user"],
        "bundle_meta": dict(prompt["bundle_meta"]),
        "content": content,
        "raw_response": response,
        "finish_reason": finish_reason,
        "usage": usage,
        "model_called": True,
        "result_valid_for_research": True,
        "total_score": float(parsed["total_score"]),
        "score_60d": float(parsed["score_60d"]),
        "score_120d": float(parsed["score_120d"]),
        "window_score_note": str(parsed.get("window_score_note", "")),
        "short_reason": str(parsed["short_reason"]),
        "teacher_scores": list(parsed.get("teacher_scores") or []),
        "parsed_payload": dict(parsed["parsed_payload"]),
    }
    write_json(cache_path, payload)
    return payload
