from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

from quant_apprentice_studio.agents.scoring import SignalScoringAgent
from quant_apprentice_studio.api.app import create_app
from quant_apprentice_studio.contracts import build_run_contract, ensure_contract_dirs, save_research_campaign, save_run_spec
from quant_apprentice_studio.provenance import read_json, write_json
from quant_apprentice_studio.registry import StudioRegistry


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "gpt_oss_20b_final"


def _client() -> TestClient:
    return TestClient(create_app())


def _ctx(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    return {
        "profile": PROFILE,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "allow_imported_fallback": True,
        "allow_demo_fallback": False,
    }


def _sample_external_signal() -> Dict[str, Any]:
    registry = StudioRegistry(PROFILE)
    agent = SignalScoringAgent(registry)
    catalog = registry.load_runtime_catalog()
    market_alias = str(catalog["defaults"]["market_run_alias"])
    sample = agent.sample_signal_record(market_run_alias=market_alias)
    return {
        "signal_id": "phase5-smoke-single-001",
        "date": str(sample.get("signal_date") or ""),
        "symbol": str(sample.get("symbol") or ""),
        "signal_type": "phase5_smoke_structured_signal",
        **sample,
    }


def _chat_message(client: TestClient, *, project_id: str, dataset_id: str, run_id: str, message: str) -> Dict[str, Any]:
    response = client.post(
        "/chat/message",
        json={
            **_ctx(project_id, dataset_id, run_id),
            "message": message,
            "mode": "simple",
        },
    )
    response.raise_for_status()
    return response.json()


def _score_action(
    client: TestClient,
    *,
    project_id: str,
    dataset_id: str,
    run_id: str,
    session_id: str,
    task_state: Dict[str, Any],
    scoring_payload: Dict[str, Any],
    file_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response = client.post(
        "/chat/action",
        json={
            **_ctx(project_id, dataset_id, run_id),
            "session_id": session_id,
            "action_id": "score_signal",
            "confirm": True,
            "task_state": task_state,
            "file_payload": file_payload,
            "scoring_payload": scoring_payload,
        },
    )
    response.raise_for_status()
    return response.json()


def _assert_artifacts(scoring: Dict[str, Any]) -> bool:
    paths = scoring.get("artifact_paths") or {}
    required = ["signal_input_manifest.json", "scoring_summary_zh.json", "scoring_provenance.json"]
    return all(Path(paths.get(name, "")).exists() for name in required)


def _make_current_workflow_lesson_run(project_id: str, dataset_id: str, run_id: str) -> str:
    contract = build_run_contract(
        profile_id=PROFILE,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=False,
    )
    ensure_contract_dirs(contract)
    catalog = StudioRegistry(PROFILE).load_runtime_catalog()
    imported_lesson = Path(catalog["lesson_runs"]["alignment_seed0005"]["final_lesson_state_json"])
    lesson_payload = read_json(imported_lesson)
    current_lesson = Path(contract["lesson_root"]) / "current_workflow_final_lesson.json"
    write_json(current_lesson, lesson_payload)
    run_spec = {
        **contract,
        "profile_id": PROFILE,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "mode": "scoring_only",
        "research_goal": "Phase 5 current workflow lesson smoke run.",
    }
    save_run_spec(contract, run_spec)
    save_research_campaign(contract, {"mode": "scoring_only", "run_id": contract["run_id"]})
    workflow_result = {
        "status": "completed",
        "mode": "scoring_only",
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "steps": [
            {
                "step_id": "S-inner",
                "owner": "Apprentice Agent",
                "wrapper_stage": "inner_loop_suite",
                "status": "completed",
                "agent_summary": {
                    "final_lesson_artifact_json": str(current_lesson),
                    "suite_summary_json": "",
                },
            }
        ],
    }
    write_json(Path(contract["workflow_root"]) / "workflow_result.json", workflow_result)
    return str(current_lesson)


def main() -> None:
    client = _client()
    signal = _sample_external_signal()
    signal2 = dict(signal)
    signal2["signal_id"] = "phase5-smoke-batch-002"
    cases = []

    chat = _chat_message(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="single-json",
        message="我有一个候选信号 JSON，帮我做 prompt preview 打分。",
    )
    single = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="single-json",
        session_id=chat["session"]["session_id"],
        task_state=chat["task_state"],
        scoring_payload={
            "mode": "dry_run",
            "lesson_source_mode": "imported",
            "pasted_payload": json.dumps(signal),
        },
    )
    scoring = single["scoring_result"]
    cases.append(
        {
            "name": "single_signal_schema_validation",
            "passed": scoring["model_called"] is False and scoring["signal_input_manifest"]["valid"] and _assert_artifacts(scoring),
            "mode": scoring["mode"],
            "artifact_paths": scoring["artifact_paths"],
        }
    )

    batch = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="batch-json",
        session_id=chat["session"]["session_id"],
        task_state={**chat["task_state"], "run_id": "batch-json"},
        scoring_payload={
            "mode": "dry_run",
            "lesson_source_mode": "imported",
            "pasted_payload": json.dumps([signal, signal2]),
        },
    )["scoring_result"]
    cases.append(
        {
            "name": "batch_signal_schema_validation",
            "passed": batch["model_called"] is False and batch["signal_input_manifest"]["record_count"] == 2 and batch["signal_input_manifest"]["valid"],
            "mode": batch["mode"],
            "artifact_paths": batch["artifact_paths"],
        }
    )

    prompt = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="prompt-only",
        session_id=chat["session"]["session_id"],
        task_state={**chat["task_state"], "run_id": "prompt-only"},
        scoring_payload={
            "mode": "prompt_only",
            "lesson_source_mode": "imported",
            "pasted_payload": json.dumps(signal),
        },
    )["scoring_result"]
    prompt_prov = prompt["scoring_provenance"]
    cases.append(
        {
            "name": "prompt_only_imported_lesson",
            "passed": prompt["result_type"] == "prompt_only_preview"
            and prompt_prov["model_called"] is False
            and prompt_prov["imported_final_asset"] is True
            and Path(prompt["artifact_paths"]["scoring_prompt_preview.json"]).exists(),
            "mode": prompt["mode"],
            "artifact_paths": prompt["artifact_paths"],
        }
    )

    current_lesson = _make_current_workflow_lesson_run("phase5-simple-scoring", "signals-a", "current-workflow")
    current_chat = _chat_message(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="current-workflow",
        message="我想用刚才生成的 lesson 给新信号打分。",
    )
    current = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="current-workflow",
        session_id=current_chat["session"]["session_id"],
        task_state=current_chat["task_state"],
        scoring_payload={
            "mode": "dry_run",
            "lesson_source_mode": "current_workflow",
            "allow_lesson_fallback": False,
            "pasted_payload": json.dumps(signal),
        },
    )["scoring_result"]
    cases.append(
        {
            "name": "current_workflow_lesson_source",
            "passed": current["scoring_provenance"]["current_workflow_asset"] is True
            and current["scoring_provenance"]["imported_final_asset"] is False
            and current["scoring_provenance"]["fallback_used"] is False
            and current_lesson in json.dumps(current, ensure_ascii=False),
            "mode": current["mode"],
            "artifact_paths": current["artifact_paths"],
        }
    )

    ohlcv = "date,symbol,open,high,low,close,volume,amount\n2025-01-02,600519,1,2,1,2,100,200\n"
    mismatch = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="mismatch",
        session_id=chat["session"]["session_id"],
        task_state={**chat["task_state"], "run_id": "mismatch"},
        file_payload={"filename": "ohlcv.csv", "content": ohlcv, "content_encoding": "text"},
        scoring_payload={"mode": "dry_run", "lesson_source_mode": "imported"},
    )["scoring_result"]
    cases.append(
        {
            "name": "ohlcv_mismatch_case",
            "passed": mismatch["signal_input_manifest"]["valid"] is False
            and mismatch["signal_input_manifest"]["mismatch_type"] == "ohlcv_market_data_not_signal_candidates"
            and "候选信号" in mismatch["summary_zh"],
            "mode": mismatch["mode"],
            "artifact_paths": mismatch["artifact_paths"],
        }
    )

    stock_only = _chat_message(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="stock-code-only",
        message="帮我看看 600519 最近怎么样？",
    )
    stock_text = json.dumps(stock_only, ensure_ascii=False)
    cases.append(
        {
            "name": "stock_code_only_guard",
            "passed": "只有股票代码不足以评分" in stock_text
            and stock_only["task_state"]["task_type"] == "scoring_only",
            "task_type": stock_only["task_state"]["task_type"],
        }
    )

    mock = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="mock-ui",
        session_id=chat["session"]["session_id"],
        task_state={**chat["task_state"], "run_id": "mock-ui"},
        scoring_payload={
            "mode": "mock",
            "lesson_source_mode": "imported",
            "pasted_payload": json.dumps(signal),
        },
    )["scoring_result"]
    cases.append(
        {
            "name": "mock_ui_result",
            "passed": mock.get("mock_result") is True
            and mock["model_called"] is False
            and bool(mock.get("teacher_scores"))
            and mock["scoring_provenance"]["not_for_research_use"] is True,
            "mode": mock["mode"],
            "artifact_paths": mock["artifact_paths"],
        }
    )

    archived = _score_action(
        client,
        project_id="phase5-simple-scoring",
        dataset_id="signals-a",
        run_id="archived-replay",
        session_id=chat["session"]["session_id"],
        task_state={**chat["task_state"], "run_id": "archived-replay"},
        scoring_payload={
            "mode": "archived_replay",
            "lesson_source_mode": "imported",
        },
    )["scoring_result"]
    cases.append(
        {
            "name": "archived_recorded_replay",
            "passed": archived["scoring_provenance"]["replay_only"] is True
            and archived["model_called"] is False
            and archived["source"] == "archived_recorded_result",
            "mode": archived["mode"],
            "artifact_paths": archived["artifact_paths"],
        }
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 5 Simple Mode Scoring Flow",
        "passed": all(row["passed"] for row in cases),
        "model_called": False,
        "vllm_started": False,
        "external_api_called": False,
        "real_research_scoring_result": False,
        "prompt_only_tests": [row["name"] for row in cases if row.get("mode") == "prompt_only"],
        "dry_run_tests": [row["name"] for row in cases if row.get("mode") == "dry_run"],
        "mock_tests": [row["name"] for row in cases if row.get("mode") == "mock"],
        "archived_replay_tests": [row["name"] for row in cases if row.get("mode") == "archived_replay"],
        "cases": cases,
    }
    out = ROOT / "test_reports" / f"phase5_simple_scoring_smoke_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(json.dumps({"report_path": str(out.relative_to(ROOT)), "passed": report["passed"], "cases": [(c["name"], c["passed"]) for c in cases]}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
