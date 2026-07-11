from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .agents.scoring import SignalScoringAgent
from .console_views import build_lesson_set_view
from .live_runtime import build_live_prompt
from .provenance import read_json, write_json
from .registry import StudioRegistry


SIMPLE_REQUIRED_SIGNAL_KEYS = ["signal_id", "signal_date", "symbol", "signal_type"]
OHLCV_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
SIGNAL_META_KEYS = {
    "signal_id",
    "date",
    "signal_date",
    "symbol",
    "signal_type",
    "entry_date",
    "exit_date",
    "future_return_5d",
    "label",
    "target",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(value: str, fallback: str = "item") -> str:
    text = "".join(ch if ch.isalnum() else "-" for ch in str(value or "").strip()).strip("-").lower()
    return text or fallback


def _scoring_case_root(contract: Mapping[str, Any], mode: str, records: List[Dict[str, Any]]) -> Path:
    text = json.dumps(records, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = Path(str(contract["scoring_root"])) / f"{timestamp}_{_slug(mode, 'scoring')}_{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode_file_payload(file_payload: Mapping[str, Any] | None) -> tuple[str, bytes]:
    if not file_payload:
        return "", b""
    filename = str(file_payload.get("filename") or "uploaded_signal.json")
    content = file_payload.get("content") or ""
    encoding = str(file_payload.get("content_encoding") or "text").lower()
    if encoding == "base64":
        return filename, base64.b64decode(str(content).encode("utf-8"))
    return filename, str(content).encode("utf-8")


def _parse_json_records(raw: str) -> List[Dict[str, Any]]:
    payload = json.loads(raw)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [dict(payload)]
    raise ValueError("Signal JSON must be an object or an array of objects.")


def _parse_csv_records(raw: str) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(raw))
    return [dict(row) for row in reader]


def _parse_parquet_records(raw: bytes) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency.
        raise ValueError("Parquet upload requires pandas/pyarrow in the Studio environment.") from exc
    frame = pd.read_parquet(io.BytesIO(raw))
    return [dict(row) for row in frame.to_dict(orient="records")]


def parse_signal_inputs(
    *,
    file_payload: Mapping[str, Any] | None = None,
    pasted_payload: str = "",
    signal_record: Mapping[str, Any] | None = None,
    signal_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    if signal_records is not None:
        return {
            "source": "inline_signal_records",
            "filename": "",
            "records": [dict(row) for row in signal_records],
        }
    if signal_record is not None:
        return {"source": "inline_signal_record", "filename": "", "records": [dict(signal_record)]}
    filename, raw_bytes = _decode_file_payload(file_payload)
    if raw_bytes:
        lower = filename.lower()
        if lower.endswith(".csv"):
            records = _parse_csv_records(raw_bytes.decode("utf-8-sig"))
            return {"source": "uploaded_csv", "filename": filename, "records": records}
        if lower.endswith(".parquet"):
            records = _parse_parquet_records(raw_bytes)
            return {"source": "uploaded_parquet", "filename": filename, "records": records}
        records = _parse_json_records(raw_bytes.decode("utf-8-sig"))
        return {"source": "uploaded_json", "filename": filename, "records": records}
    if str(pasted_payload or "").strip():
        return {"source": "pasted_json", "filename": "", "records": _parse_json_records(str(pasted_payload))}
    raise ValueError("No structured signal input was provided.")


def _maybe_number(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return value
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _parse_features(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): _maybe_number(v) for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): _maybe_number(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_signal_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(record or {})
    features = _parse_features(row.get("features"))
    if not features:
        features = {
            str(key): _maybe_number(value)
            for key, value in row.items()
            if str(key) not in SIGNAL_META_KEYS and not isinstance(value, (dict, list))
        }
    signal_date = str(row.get("signal_date") or row.get("date") or "").strip()
    return {
        "signal_id": str(row.get("signal_id") or "").strip(),
        "date": str(row.get("date") or signal_date).strip(),
        "signal_date": signal_date,
        "symbol": str(row.get("symbol") or "").strip(),
        "signal_type": str(row.get("signal_type") or "").strip(),
        "entry_date": str(row.get("entry_date") or signal_date).strip(),
        "exit_date": str(row.get("exit_date") or "").strip(),
        "features": features,
        "raw_record": row,
    }


def _looks_like_ohlcv_record(record: Mapping[str, Any]) -> bool:
    return OHLCV_COLUMNS.issubset({str(key) for key in record.keys()})


def _validate_records(
    *,
    scoring_agent: SignalScoringAgent,
    records: List[Dict[str, Any]],
    schema_market_run_alias: str,
) -> Dict[str, Any]:
    items = []
    union_missing = set()
    union_missing_features = set()
    ohlcv_like_count = 0
    valid_count = 0
    normalized_records = []
    for idx, raw in enumerate(records):
        normalized = normalize_signal_record(raw)
        missing_simple = []
        if not normalized["signal_id"]:
            missing_simple.append("signal_id")
        if not normalized["signal_date"]:
            missing_simple.append("signal_date")
        if not normalized["symbol"]:
            missing_simple.append("symbol")
        if not normalized["signal_type"]:
            missing_simple.append("signal_type")
        if not normalized["features"]:
            missing_simple.append("features")
        canonical_record = {
            "symbol": normalized["symbol"],
            "signal_date": normalized["signal_date"],
            "entry_date": normalized["entry_date"],
            "exit_date": normalized["exit_date"],
            "features": normalized["features"],
        }
        canonical_validation: Dict[str, Any]
        if missing_simple:
            canonical_validation = {
                "valid": False,
                "missing_top_level_keys": [],
                "missing_feature_keys": [],
                "non_numeric_feature_keys": [],
                "normalized_signal_record": canonical_record,
            }
        else:
            canonical_validation = scoring_agent.validate_signal_record(
                canonical_record,
                market_run_alias=schema_market_run_alias,
            )
        missing_features = list(canonical_validation.get("missing_feature_keys") or [])
        non_numeric = list(canonical_validation.get("non_numeric_feature_keys") or [])
        valid = not missing_simple and bool(canonical_validation.get("valid", False))
        if valid:
            valid_count += 1
            normalized_records.append(dict(canonical_validation.get("normalized_signal_record") or canonical_record))
        union_missing.update(missing_simple)
        union_missing_features.update(missing_features)
        if _looks_like_ohlcv_record(raw) and not normalized["signal_id"]:
            ohlcv_like_count += 1
        items.append(
            {
                "index": idx,
                "signal_id": normalized["signal_id"],
                "symbol": normalized["symbol"],
                "signal_date": normalized["signal_date"],
                "signal_type": normalized["signal_type"],
                "valid": valid,
                "missing_simple_keys": missing_simple,
                "missing_feature_count": len(missing_features),
                "missing_feature_keys_sample": missing_features[:20],
                "non_numeric_feature_keys": non_numeric[:20],
                "canonical_validation": {
                    key: value
                    for key, value in canonical_validation.items()
                    if key != "normalized_signal_record"
                },
            }
        )
    readiness = "ready" if records and valid_count == len(records) else "failed"
    mismatch_type = "ohlcv_market_data_not_signal_candidates" if ohlcv_like_count and valid_count == 0 else ""
    return {
        "record_count": len(records),
        "valid_count": valid_count,
        "invalid_count": len(records) - valid_count,
        "data_readiness": readiness,
        "valid": readiness == "ready",
        "required_columns": SIMPLE_REQUIRED_SIGNAL_KEYS + ["features"],
        "missing_columns": sorted(union_missing),
        "missing_feature_keys_sample": sorted(union_missing_features)[:50],
        "mismatch_type": mismatch_type,
        "items": items,
        "normalized_signal_records": normalized_records,
    }


def _resolve_lesson_source(
    *,
    profile_id: str,
    contract: Mapping[str, Any],
    source_mode: str,
    lesson_alias: str,
    allow_lesson_fallback: bool,
) -> Dict[str, Any]:
    registry = StudioRegistry(profile_id)
    catalog = registry.load_runtime_catalog()
    default_alias = str(catalog.get("defaults", {}).get("alignment_seed_alias", "") or "alignment_seed0005").strip()
    resolved_alias = str(lesson_alias or default_alias).strip()
    current_lesson_path = ""
    try:
        lesson_set = build_lesson_set_view(
            project_id=str(contract["project_id"]),
            dataset_id=str(contract["dataset_id"]),
            run_id=str(contract["run_id"]),
        )
        current_lesson_path = str(lesson_set.get("final_lesson_state_json") or "").strip()
    except Exception:
        lesson_set = {}

    source_mode = str(source_mode or "auto").strip()
    if source_mode == "current_workflow":
        if current_lesson_path:
            return {
                "lesson_alias": "",
                "final_lesson_state_json": current_lesson_path,
                "lesson_source": "current_workflow_asset",
                "teacher_source": "current_workflow_asset",
                "teacher_library_id": f"user_{contract.get('project_id')}_{contract.get('dataset_id')}_{contract.get('run_id')}",
                "teacher_library_name_zh": f"{contract.get('run_id')} 老师库",
                "teacher_library_source_type": "user_trained",
                "fallback_used": False,
                "fallback_reason": "",
                "current_workflow_asset": True,
                "imported_final_asset": False,
                "demo_asset": False,
                "lesson_set": lesson_set,
            }
        if allow_lesson_fallback and bool(contract.get("allow_imported_fallback", False)):
            return {
                "lesson_alias": resolved_alias,
                "final_lesson_state_json": "",
                "lesson_source": "imported_final_asset",
                "teacher_source": "imported_frozen_teacher",
                "teacher_library_id": "paper_ashare_gptoss20b_v7",
                "teacher_library_name_zh": "A股技术形态基准老师库",
                "teacher_library_source_type": "built_in_baseline",
                "fallback_used": True,
                "fallback_reason": "Requested current workflow final_lesson_set, but none exists. User explicitly allowed fallback to the built-in teacher library.",
                "current_workflow_asset": False,
                "imported_final_asset": True,
                "demo_asset": False,
                "lesson_set": lesson_set,
            }
        raise ValueError("Current workflow final_lesson_set is not available. Choose the built-in teacher library or explicitly allow fallback.")

    if source_mode == "auto" and current_lesson_path:
        return {
            "lesson_alias": "",
            "final_lesson_state_json": current_lesson_path,
            "lesson_source": "current_workflow_asset",
            "teacher_source": "current_workflow_asset",
            "teacher_library_id": f"user_{contract.get('project_id')}_{contract.get('dataset_id')}_{contract.get('run_id')}",
            "teacher_library_name_zh": f"{contract.get('run_id')} 老师库",
            "teacher_library_source_type": "user_trained",
            "fallback_used": False,
            "fallback_reason": "",
            "current_workflow_asset": True,
            "imported_final_asset": False,
            "demo_asset": False,
            "lesson_set": lesson_set,
        }

    return {
        "lesson_alias": resolved_alias,
        "final_lesson_state_json": "",
        "lesson_source": "imported_final_asset",
        "teacher_source": "imported_frozen_teacher",
        "teacher_library_id": "paper_ashare_gptoss20b_v7",
        "teacher_library_name_zh": "A股技术形态基准老师库",
        "teacher_library_source_type": "built_in_baseline",
        "fallback_used": False,
        "fallback_reason": "",
        "current_workflow_asset": False,
        "imported_final_asset": True,
        "demo_asset": source_mode == "imported_demo",
        "lesson_set": lesson_set,
    }


def _write_common_artifacts(root: Path, payloads: Mapping[str, Any]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for filename, payload in payloads.items():
        path = root / filename
        write_json(path, payload)
        paths[filename] = str(path)
    return paths


def _summary_zh(payload: Mapping[str, Any]) -> str:
    mode = str(payload.get("mode") or "")
    validation = dict(payload.get("signal_input_manifest") or {})
    provenance = dict(payload.get("scoring_provenance") or {})
    if mode == "prompt_only":
        lead = "本次只完成输入校验和 prompt preview，没有调用模型，也没有真实评分。"
    elif mode == "dry_run":
        lead = "本次是 dry run：已校验输入和 lesson 来源，但没有构造真实模型评分结果。"
    elif mode == "mock":
        lead = "本次是 UI mock scoring：分数仅用于界面验收，不可用于研究结论。"
    elif mode == "archived_replay":
        lead = "本次是 archived recorded result replay：复用历史记录展示，不是新的模型调用。"
    else:
        lead = "本次未调用模型。"
    if not validation.get("valid", False):
        missing = validation.get("missing_columns") or []
        mismatch = validation.get("mismatch_type") or ""
        details = f"缺少字段：{', '.join(missing) if missing else '见 manifest'}。"
        if mismatch:
            details += " 这更像历史行情 OHLCV 数据，不是候选信号。"
        return f"{lead}\n输入暂时不能评分：{details}\n请上传包含 signal_id、signal_date、symbol、signal_type 和完整结构化 features 的候选信号。旧字段 date 可作为 signal_date 别名自动识别。"
    return (
        f"{lead}\n"
        f"信号数量：{validation.get('record_count', 0)}；lesson_source={provenance.get('lesson_source', '-') }；"
        f"teacher_source={provenance.get('teacher_source', '-') }；fallback_used={provenance.get('fallback_used', False)}。\n"
        "这是研究辅助评分流程，不构成投资建议，也不保证收益。"
    )


def _mock_result(bundle_meta: Mapping[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = records[0] if records else {}
    features = dict(first.get("features") or {})
    numeric_values = [float(v) for v in features.values() if isinstance(v, (int, float))]
    base = 50.0
    if numeric_values:
        base = max(5.0, min(95.0, 50.0 + sum(numeric_values[:20]) / max(1, len(numeric_values[:20]))))
    teacher_cards = list(bundle_meta.get("teacher_cards") or [])
    teacher_scores = []
    for idx, teacher in enumerate(teacher_cards):
        score = max(0.0, min(100.0, base + (idx - 1.5) * 3.0))
        teacher_scores.append(
            {
                "round_id": str(teacher.get("round_id") or ""),
                "title": str(teacher.get("title") or teacher.get("style_family") or "Teacher"),
                "score": round(score, 2),
                "note": "UI mock only: deterministic placeholder, not a real model judgement.",
            }
        )
    return {
        "total_score": round(base, 2),
        "short_reason": "Mock result for UI smoke test only. No model was called.",
        "teacher_scores": teacher_scores,
        "mock_result": True,
        "not_for_research_use": True,
    }


def run_simple_scoring(
    *,
    profile_id: str,
    contract: Mapping[str, Any],
    scoring_payload: Mapping[str, Any],
    file_payload: Mapping[str, Any] | None = None,
    signal_record: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    mode = str(scoring_payload.get("mode") or "prompt_only").strip()
    source_mode = str(scoring_payload.get("lesson_source_mode") or "auto").strip()
    schema_market_run_alias = str(scoring_payload.get("schema_market_run_alias") or "").strip()
    lesson_alias = str(scoring_payload.get("lesson_alias") or "alignment_seed0005").strip()
    allow_lesson_fallback = bool(scoring_payload.get("allow_lesson_fallback", False))
    pasted_payload = str(scoring_payload.get("pasted_payload") or "").strip()
    inline_records = scoring_payload.get("signal_records")
    archived_run = str(scoring_payload.get("archived_run") or "").strip()
    archived_signal_date = str(scoring_payload.get("archived_signal_date") or "").strip()
    archived_symbol = str(scoring_payload.get("archived_symbol") or "").strip()

    registry = StudioRegistry(profile_id)
    registry.ensure_bootstrapped()
    scoring_agent = SignalScoringAgent(registry)
    if not schema_market_run_alias:
        schema_market_run_alias = str(registry.load_runtime_catalog().get("defaults", {}).get("market_run_alias", "")).strip()

    parsed_source = {
        "source": "none",
        "filename": "",
        "records": [],
    }
    records: List[Dict[str, Any]]
    if mode == "archived_replay":
        archived_run = archived_run or schema_market_run_alias
        recorded = scoring_agent.score_recorded(
            archived_run,
            archived_signal_date or "",
            archived_symbol or "",
        ).__dict__ if archived_signal_date and archived_symbol else None
        if recorded is None:
            sample = scoring_agent.sample_signal_record(market_run_alias=archived_run)
            recorded = scoring_agent.score_recorded(
                archived_run,
                str(sample.get("signal_date") or ""),
                str(sample.get("symbol") or ""),
            ).__dict__
        records = [
            {
                "signal_id": f"archived::{recorded.get('signal_date')}::{recorded.get('symbol')}",
                "date": recorded.get("signal_date", ""),
                "symbol": recorded.get("symbol", ""),
                "signal_type": "archived_recorded_signal",
                **dict(recorded.get("signal_record") or {}),
            }
        ]
        parsed_source = {"source": "archived_recorded_result", "filename": "", "records": records}
    else:
        parsed_source = parse_signal_inputs(
            file_payload=file_payload,
            pasted_payload=pasted_payload,
            signal_record=signal_record,
            signal_records=inline_records if isinstance(inline_records, list) else None,
        )
        records = [dict(row) for row in parsed_source["records"]]

    validation = _validate_records(
        scoring_agent=scoring_agent,
        records=records,
        schema_market_run_alias=schema_market_run_alias,
    )
    root = _scoring_case_root(contract, mode, records)
    input_payload = {
        "created_at": _now_iso(),
        "source": parsed_source["source"],
        "filename": parsed_source["filename"],
        "records": records,
    }
    normalized_records = list(validation.get("normalized_signal_records") or [])

    try:
        lesson_resolution = _resolve_lesson_source(
            profile_id=profile_id,
            contract=contract,
            source_mode=source_mode,
            lesson_alias=lesson_alias,
            allow_lesson_fallback=allow_lesson_fallback,
        )
    except Exception as exc:
        lesson_resolution = {
            "lesson_alias": lesson_alias,
            "final_lesson_state_json": "",
            "lesson_source": "unresolved",
            "teacher_source": "unresolved",
            "fallback_used": False,
            "fallback_reason": "",
            "current_workflow_asset": False,
            "imported_final_asset": False,
            "demo_asset": False,
            "resolution_error": str(exc),
        }

    prompt_preview: Dict[str, Any] = {}
    if validation.get("valid") and not lesson_resolution.get("resolution_error"):
        prompt_items = []
        for idx, normalized in enumerate(normalized_records):
            prompt = build_live_prompt(
                registry,
                lesson_alias=str(lesson_resolution.get("lesson_alias") or ""),
                final_lesson_state_json=str(lesson_resolution.get("final_lesson_state_json") or ""),
                signal_record=normalized,
            )
            if idx == 0:
                prompt_preview = dict(prompt)
            prompt_items.append(
                {
                    "index": idx,
                    "symbol": normalized.get("symbol", ""),
                    "signal_date": normalized.get("signal_date", ""),
                    "system": prompt.get("system", ""),
                    "user": prompt.get("user", ""),
                    "bundle_meta": prompt.get("bundle_meta", {}),
                }
            )
        prompt_preview["items"] = prompt_items[:5]

    bundle_meta = dict(prompt_preview.get("bundle_meta") or {})
    if "lesson_source" in bundle_meta:
        bundle_meta["runtime_lesson_source"] = bundle_meta.get("lesson_source", "")
    if "teacher_source" in bundle_meta:
        bundle_meta["runtime_teacher_source"] = bundle_meta.get("teacher_source", "")
    bundle_meta["lesson_source"] = str(lesson_resolution.get("lesson_source") or bundle_meta.get("lesson_source") or "")
    bundle_meta["teacher_source"] = str(lesson_resolution.get("teacher_source") or bundle_meta.get("teacher_source") or "")
    bundle_meta["teacher_library_id"] = str(lesson_resolution.get("teacher_library_id") or bundle_meta.get("teacher_library_id") or "")
    bundle_meta["teacher_library_name_zh"] = str(lesson_resolution.get("teacher_library_name_zh") or bundle_meta.get("teacher_library_name_zh") or "")
    bundle_meta["teacher_library_source_type"] = str(lesson_resolution.get("teacher_library_source_type") or bundle_meta.get("teacher_library_source_type") or "")
    bundle_meta["fallback_used"] = bool(lesson_resolution.get("fallback_used", False) or bundle_meta.get("fallback_used", False))
    bundle_meta["fallback_reason"] = str(lesson_resolution.get("fallback_reason") or bundle_meta.get("fallback_reason") or "")

    result_payload: Dict[str, Any] = {
        "mode": mode,
        "result_type": {
            "prompt_only": "prompt_only_preview",
            "dry_run": "dry_run_summary",
            "mock": "mock_scoring_response",
            "archived_replay": "archived_recorded_replay",
        }.get(mode, "dry_run_summary"),
        "model_called": False,
        "result_valid_for_research": False,
        "signal_input_manifest": {
            **{key: value for key, value in validation.items() if key != "normalized_signal_records"},
            "source": parsed_source["source"],
            "filename": parsed_source["filename"],
            "schema_market_run_alias": schema_market_run_alias,
        },
        "bundle_meta": bundle_meta,
    }
    if mode == "mock" and validation.get("valid") and not lesson_resolution.get("resolution_error"):
        result_payload.update(_mock_result(bundle_meta, normalized_records))
    if mode == "archived_replay":
        result_payload.update(
            {
                "source": "archived_recorded_result",
                "replay_only": True,
                "recorded_run": archived_run,
            }
        )

    provenance = {
        "created_at": _now_iso(),
        "project_id": str(contract["project_id"]),
        "dataset_id": str(contract["dataset_id"]),
        "run_id": str(contract["run_id"]),
        "mode": mode,
        "model_called": False,
        "result_valid_for_research": False,
        "lesson_source": bundle_meta.get("lesson_source", ""),
        "teacher_source": bundle_meta.get("teacher_source", ""),
        "teacher_library_id": bundle_meta.get("teacher_library_id", ""),
        "teacher_library_name_zh": bundle_meta.get("teacher_library_name_zh", ""),
        "teacher_library_source_type": bundle_meta.get("teacher_library_source_type", ""),
        "fallback_used": bool(bundle_meta.get("fallback_used", False)),
        "fallback_reason": str(bundle_meta.get("fallback_reason") or ""),
        "imported_final_asset": bool(lesson_resolution.get("imported_final_asset", False)),
        "current_workflow_asset": bool(lesson_resolution.get("current_workflow_asset", False)),
        "demo_asset": bool(lesson_resolution.get("demo_asset", False)),
        "source": parsed_source["source"],
        "mock_result": mode == "mock",
        "replay_only": mode == "archived_replay",
        "not_for_research_use": mode == "mock",
        "vllm_started": False,
        "external_api_called": False,
    }
    summary_zh = {
        "created_at": _now_iso(),
        "summary_zh": _summary_zh(
            {
                "mode": mode,
                "signal_input_manifest": result_payload["signal_input_manifest"],
                "scoring_provenance": provenance,
            }
        ),
        "risk_boundaries_zh": [
            "这是研究辅助评分，不构成投资建议。",
            "本阶段没有调用本地 GPT-OSS，也没有调用外部 API。",
            "prompt_only / dry_run / mock / 历史样本复核都不能当作真实模型评分结论。",
        ],
    }

    artifact_payloads: Dict[str, Any] = {
        "signal_input_manifest.json": result_payload["signal_input_manifest"],
        "scoring_summary_zh.json": summary_zh,
        "scoring_provenance.json": provenance,
    }
    if len(records) == 1:
        artifact_payloads["scoring_input.json"] = input_payload
    else:
        artifact_payloads["scoring_input_batch.json"] = input_payload
    if mode == "prompt_only":
        artifact_payloads["scoring_prompt_preview.json"] = prompt_preview
    elif mode == "dry_run":
        artifact_payloads["scoring_dry_run_summary.json"] = result_payload
    elif mode == "mock":
        artifact_payloads["scoring_mock_result.json"] = result_payload
        artifact_payloads["scoring_prompt_preview.json"] = prompt_preview
    elif mode == "archived_replay":
        artifact_payloads["scoring_recorded_replay.json"] = result_payload

    paths = _write_common_artifacts(root, artifact_payloads)
    result_payload["artifact_paths"] = paths
    result_payload["scoring_root"] = str(root)
    result_payload["summary_zh"] = summary_zh["summary_zh"]
    result_payload["scoring_provenance"] = provenance
    result_payload["prompt_preview_available"] = bool(prompt_preview)
    return result_payload
