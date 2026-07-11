from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_apprentice_studio.agents.chief import ChiefResearchAgent
from quant_apprentice_studio.local_service import describe_local_service_status
from quant_apprentice_studio.provenance import write_json
from quant_apprentice_studio.registry import StudioRegistry


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _read_models(base_url: str) -> Dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        return {"error": str(exc)}


def _get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _nvidia_snapshot() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        return {"error": str(exc), "items": []}
    items: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        items.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return {
        "items": items,
        "gpu_1_memory_low": next((gpu["memory_used_mib"] < 512 for gpu in items if gpu["index"] == 1), False),
    }


def _save_case_artifacts(case_dir: Path, live_payload: Dict[str, Any]) -> Dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw_response.json"
    parsed_path = case_dir / "parsed_result.json"
    provenance_path = case_dir / "provenance.json"
    write_json(raw_path, live_payload.get("raw_response") or {})
    write_json(
        parsed_path,
        {
            "mode": live_payload.get("mode"),
            "total_score": live_payload.get("total_score"),
            "short_reason": live_payload.get("short_reason"),
            "teacher_scores": live_payload.get("teacher_scores") or [],
            "parsed_payload": live_payload.get("parsed_payload") or {},
            "finish_reason": live_payload.get("finish_reason"),
            "usage": live_payload.get("usage") or {},
        },
    )
    write_json(
        provenance_path,
        {
            "model_called": bool(live_payload.get("model_called")),
            "result_valid_for_research": bool(live_payload.get("result_valid_for_research")),
            "model": live_payload.get("model"),
            "api_url": live_payload.get("api_url"),
            "cache_hit": bool(live_payload.get("cache_hit")),
            "cache_path": live_payload.get("cache_path"),
            "saved_run_path": live_payload.get("saved_run_path"),
            "lesson_alias": live_payload.get("lesson_alias"),
            "final_lesson_state_json": live_payload.get("final_lesson_state_json"),
            "bundle_meta": live_payload.get("bundle_meta") or {},
            "local_service_status": live_payload.get("local_service_status") or {},
            "signal_schema_validation": live_payload.get("signal_schema_validation") or {},
        },
    )
    return {
        "raw_response_json": str(raw_path),
        "parsed_result_json": str(parsed_path),
        "provenance_json": str(provenance_path),
    }


def main() -> None:
    os.environ.setdefault("QA_STUDIO_LOCAL_CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("QA_STUDIO_LOCAL_TENSOR_PARALLEL_SIZE", "1")
    os.environ.setdefault("QA_STUDIO_TEACHER_CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("QA_STUDIO_LIVE_MAX_TOKENS", "2048")
    os.environ.setdefault("QA_STUDIO_LIVE_TIMEOUT_SECONDS", "240")

    stamp = _now_stamp()
    report_dir = ROOT / "test_reports" / f"phase9_live_gptoss_validation_artifacts_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "test_reports" / f"phase9_live_gptoss_validation_{stamp}.json"

    profile = "gpt_oss_20b_final"
    lesson_alias = "alignment_seed0005"
    market_run = "market_2025_lseed20250705"
    signal_date = "2025-01-02"
    symbol = "000151"

    failed_checks: List[str] = []
    fixed_issues: List[str] = [
        "live_runtime.py now persists raw_response on successful live calls",
        "live_runtime.py now persists parse-error raw_response artifacts before raising",
        "Simple Mode scoring controls are runtime-aware instead of hard-coded no-model",
        "README distinguishes no-model rehearsal from local GPT-OSS live validation",
    ]
    remaining_ux_issues: List[str] = []

    local_status = describe_local_service_status()
    if not local_status.get("service_healthy"):
        failed_checks.append("local_gptoss_service_not_healthy")
    if str(local_status.get("cuda_visible_devices")) != "0,2":
        failed_checks.append("local_service_config_not_phase9_cuda_0_2")
    if int(local_status.get("tensor_parallel_size") or 0) != 2:
        failed_checks.append("local_service_config_not_tensor_parallel_2")

    models_payload = _read_models(str(local_status.get("base_url") or "http://127.0.0.1:2310"))
    model_ids = [str(row.get("id")) for row in list(models_payload.get("data") or []) if isinstance(row, dict)]
    if "gpt-oss-20b" not in model_ids:
        failed_checks.append("served_model_identity_missing_gpt_oss_20b")

    registry = StudioRegistry(profile)
    chief = ChiefResearchAgent(registry)

    cases: List[Dict[str, Any]] = []
    model_called = False
    recorded_live = None
    try:
        recorded = chief.scoring.score_recorded(market_run, signal_date, symbol)
        recorded_live = chief.scoring.score_live(
            lesson_alias=lesson_alias,
            signal_record=recorded.signal_record,
            prompt_only=False,
            reuse_cache=False,
            persist_run=True,
            run_label=f"phase9_live_gptoss_validation_{stamp}",
            source_tag="phase9_recorded_reference",
            schema_market_run_alias=market_run,
        )
        model_called = bool(recorded_live.get("model_called"))
        artifact_paths = _save_case_artifacts(report_dir / "recorded_reference_live_score", recorded_live)
        cases.append(
            {
                "case_id": "recorded_reference_live_score",
                "passed": bool(recorded_live.get("result_valid_for_research") and recorded_live.get("raw_response")),
                "recorded_total_score": recorded.total_score,
                "live_total_score": recorded_live.get("total_score"),
                "score_delta": float(recorded_live.get("total_score", 0.0)) - float(recorded.total_score),
                "teacher_score_count": len(recorded_live.get("teacher_scores") or []),
                "finish_reason": recorded_live.get("finish_reason"),
                "usage": recorded_live.get("usage") or {},
                "cache_path": recorded_live.get("cache_path"),
                "saved_run_path": recorded_live.get("saved_run_path"),
                "artifact_paths": artifact_paths,
            }
        )
    except Exception as exc:
        failed_checks.append("recorded_reference_live_score_failed")
        cases.append({"case_id": "recorded_reference_live_score", "passed": False, "error": str(exc)})

    try:
        compare_payload = chief.scoring.compare_live_to_recorded(
            lesson_alias=lesson_alias,
            market_run_alias=market_run,
            signal_date=signal_date,
            symbol=symbol,
            prompt_only=False,
            reuse_cache=True,
        )
        compare_path = report_dir / "compare_live_to_recorded.json"
        write_json(compare_path, compare_payload)
        cases.append(
            {
                "case_id": "compare_live_to_recorded",
                "passed": "score_delta" in compare_payload,
                "score_delta": compare_payload.get("score_delta"),
                "recorded_total_score": (compare_payload.get("recorded") or {}).get("total_score"),
                "live_total_score": (compare_payload.get("live") or {}).get("total_score"),
                "artifact_path": str(compare_path),
            }
        )
    except Exception as exc:
        failed_checks.append("compare_live_to_recorded_failed")
        cases.append({"case_id": "compare_live_to_recorded", "passed": False, "error": str(exc)})

    try:
        if recorded_live:
            external_payload = chief.scoring.score_live(
                lesson_alias=lesson_alias,
                signal_record=recorded_live["signal_record"],
                prompt_only=False,
                reuse_cache=True,
                persist_run=True,
                run_label=f"phase9_external_signal_cached_{stamp}",
                source_tag="phase9_external_signal",
                schema_market_run_alias=market_run,
            )
            external_path = report_dir / "external_signal_live_result.json"
            write_json(external_path, external_payload)
            cases.append(
                {
                    "case_id": "external_signal_live_score",
                    "passed": bool(external_payload.get("total_score") is not None),
                    "cache_hit": bool(external_payload.get("cache_hit")),
                    "model_called": bool(external_payload.get("model_called")),
                    "total_score": external_payload.get("total_score"),
                    "artifact_path": str(external_path),
                }
            )
    except Exception as exc:
        failed_checks.append("external_signal_live_score_failed")
        cases.append({"case_id": "external_signal_live_score", "passed": False, "error": str(exc)})

    current_workflow_reason = (
        "No current workflow final_lesson_set was available in the default active run during Phase 9; "
        "validation used imported_final_asset alignment_seed0005 instead."
    )
    cases.append(
        {
            "case_id": "current_workflow_final_lesson_live_score",
            "passed": False,
            "skipped": True,
            "reason": current_workflow_reason,
        }
    )

    app_js = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    frontend_checks = {
        "live_option_runtime_aware": "live scoring (local GPT-OSS ready)" in app_js,
        "simple_mode_live_calls_score_live": 'fetchJson("/score/live"' in app_js,
        "simple_mode_live_calls_score_live_batch": 'fetchJson("/score/live-batch-external"' in app_js,
        "runtime_note_present": "simple-scoring-runtime-note" in index_html,
        "no_phase5_fixed_wording_removed": "Phase 5 no-model mode" not in index_html,
    }
    api_live_config = _get_json("http://127.0.0.1:8010/live-config?profile=gpt_oss_20b_final")
    api_cached_score = _post_json(
        "http://127.0.0.1:8010/score/live",
        {
            "profile": profile,
            "lesson_alias": lesson_alias,
            "from_recorded_run": market_run,
            "signal_date": signal_date,
            "symbol": symbol,
            "prompt_only": False,
            "reuse_cache": True,
            "persist_run": False,
        },
    )
    api_checks = {
        "api_live_config_available": not api_live_config.get("error"),
        "api_live_config_runtime_healthy": bool((api_live_config.get("local_service") or {}).get("service_healthy")),
        "api_score_live_cache_available": not api_cached_score.get("error"),
        "api_score_live_returns_raw_response": bool(api_cached_score.get("raw_response")),
        "api_score_live_model_called_true": bool(api_cached_score.get("model_called")),
    }
    readme_checks = {
        "runtime_aware_current_phase": "runtime-aware Research OS V1" in readme,
        "phase9_gpt_gpu_0_command_present": "QA_STUDIO_LOCAL_CUDA_VISIBLE_DEVICES=0" in readme,
        "phase9_teacher_gpu_1_command_present": "QA_STUDIO_TEACHER_CUDA_VISIBLE_DEVICES=1" in readme,
        "raw_response_documented": "raw_response" in readme,
    }
    for key, value in {**frontend_checks, **api_checks, **readme_checks}.items():
        if not value:
            failed_checks.append(key)

    gpu_snapshot = _nvidia_snapshot()
    report = {
        "phase": "phase9_live_gptoss_validation",
        "created_at": stamp,
        "model_called": bool(model_called),
        "vllm_started": bool(local_status.get("service_healthy")),
        "external_api_called": False,
        "clean_pipeline_modified": False,
        "local_endpoint": local_status.get("api_url"),
        "model_identity": model_ids,
        "runtime_status": local_status,
        "gpu_snapshot": gpu_snapshot,
        "gpu_policy": {
            "gptoss_cuda_visible_devices": "0",
            "teacher_xgboost_cuda_visible_devices": "1",
        },
        "cases": cases,
        "frontend_checks": frontend_checks,
        "api_checks": api_checks,
        "api_live_config_preview": {
            "api_url": api_live_config.get("api_url"),
            "model": api_live_config.get("model"),
            "local_service": api_live_config.get("local_service") or {},
        },
        "api_cached_score_preview": {
            "mode": api_cached_score.get("mode"),
            "cache_hit": api_cached_score.get("cache_hit"),
            "model_called": api_cached_score.get("model_called"),
            "total_score": api_cached_score.get("total_score"),
            "raw_response_available": bool(api_cached_score.get("raw_response")),
            "lesson_source": (api_cached_score.get("bundle_meta") or {}).get("lesson_source"),
            "teacher_source": (api_cached_score.get("bundle_meta") or {}).get("teacher_source"),
            "fallback_used": (api_cached_score.get("bundle_meta") or {}).get("fallback_used"),
            "error": api_cached_score.get("error"),
        },
        "readme_checks": readme_checks,
        "failed_checks": failed_checks,
        "fixed_issues": fixed_issues,
        "remaining_ux_issues": remaining_ux_issues,
        "report_artifact_dir": str(report_dir),
        "passed": not failed_checks,
    }
    write_json(report_path, report)
    print(json.dumps({"report_path": str(report_path), "passed": report["passed"], "failed_checks": failed_checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
