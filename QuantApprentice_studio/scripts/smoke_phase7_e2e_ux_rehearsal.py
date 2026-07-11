from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

from quant_apprentice_studio.agents.scoring import SignalScoringAgent
from quant_apprentice_studio.api.app import create_app
from quant_apprentice_studio.contracts import (
    build_run_contract,
    ensure_contract_dirs,
    save_dataset_manifest,
    save_project_config,
    save_research_campaign,
    save_run_spec,
)
from quant_apprentice_studio.provenance import read_json, write_json
from quant_apprentice_studio.registry import StudioRegistry


PROFILE = "gpt_oss_20b_final"
ROOT = Path(__file__).resolve().parents[1]


def _ctx(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    return {
        "profile": PROFILE,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "allow_imported_fallback": True,
        "allow_demo_fallback": False,
    }


def _client() -> TestClient:
    return TestClient(create_app())


def _post(client: TestClient, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    return response.json()


def _get(client: TestClient, endpoint: str, **params: Any) -> Dict[str, Any]:
    response = client.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


def _sample_signal() -> Dict[str, Any]:
    registry = StudioRegistry(PROFILE)
    agent = SignalScoringAgent(registry)
    market_alias = str(registry.load_runtime_catalog()["defaults"]["market_run_alias"])
    sample = agent.sample_signal_record(market_run_alias=market_alias)
    return {
        "signal_id": "phase7-signal-001",
        "date": str(sample.get("signal_date") or ""),
        "symbol": str(sample.get("symbol") or ""),
        "signal_type": "phase7_structured_signal",
        **sample,
    }


def _chat(client: TestClient, project_id: str, dataset_id: str, run_id: str, message: str) -> Dict[str, Any]:
    return _post(
        client,
        "/chat/message",
        {
            **_ctx(project_id, dataset_id, run_id),
            "mode": "simple",
            "message": message,
            "attachments": [],
        },
    )


def _action(
    client: TestClient,
    project_id: str,
    dataset_id: str,
    run_id: str,
    action_id: str,
    chat_payload: Dict[str, Any],
    *,
    file_payload: Dict[str, Any] | None = None,
    scoring_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _post(
        client,
        "/chat/action",
        {
            **_ctx(project_id, dataset_id, run_id),
            "session_id": chat_payload["session"]["session_id"],
            "action_id": action_id,
            "confirm": True,
            "task_state": chat_payload["task_state"],
            "file_payload": file_payload,
            "scoring_payload": scoring_payload or {},
        },
    )


def _create_no_model_full_pipeline_smoke(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=PROFILE,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=False,
    )
    ensure_contract_dirs(contract)
    save_project_config(contract, {"profile_id": PROFILE, "project_id": contract["project_id"], "dataset_id": contract["dataset_id"], "run_id": contract["run_id"]})
    save_dataset_manifest(
        contract,
        {
            "source_type": "upload_local",
            "market": "A-share",
            "row_count": 64,
            "symbol_count": 8,
            "date_range": {"start": "2020-01-02", "end": "2022-12-30"},
            "required_columns": ["date", "symbol", "open", "high", "low", "close", "volume", "amount"],
            "missing_columns": [],
            "data_readiness": "ready_for_full_research_pipeline",
            "data_isolation_status": contract["data_isolation"],
        },
    )
    save_run_spec(
        contract,
        {
            **contract,
            "profile_id": PROFILE,
            "mode": "full_pipeline",
            "research_goal": "Phase 7 no-model full pipeline smoke rehearsal.",
            "api_model": "gpt-oss-20b",
        },
    )
    save_research_campaign(contract, {"mode": "full_pipeline", "research_goal": "Phase 7 no-model full pipeline smoke rehearsal."})
    studio_control = Path(contract["shared_context_root"]) / "studio_control"
    studio_control.mkdir(parents=True, exist_ok=True)
    research_spec = studio_control / "research_spec.json"
    write_json(research_spec, {"research_goal": "Phase 7 no-model full pipeline smoke rehearsal.", "model_called": False})
    imported_lesson = Path(StudioRegistry(PROFILE).load_runtime_catalog()["lesson_runs"]["alignment_seed0005"]["final_lesson_state_json"])
    final_lesson = Path(contract["lesson_root"]) / "phase7_final_lesson_set.json"
    write_json(final_lesson, read_json(imported_lesson))
    suite_summary = Path(contract["lesson_root"]) / "phase7_inner_loop_summary.json"
    write_json(suite_summary, {"final_lesson_artifact_json": str(final_lesson), "model_called": False})
    selection = {
        "resolution_source": "current_workflow_asset",
        "fallback_reason": "",
        "teachers": [
            {
                "round_id": "phase7_frozen_round",
                "title": "Phase 7 current workflow frozen teacher",
                "report_dir": str(Path(contract["teacher_zoo_root"]) / "phase7_frozen_round" / "report_v2"),
                "mean_alpha": 0.01,
                "nav_cagr": 0.2,
            }
        ],
    }
    write_json(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json", selection)
    write_json(
        Path(contract["workflow_root"]) / "workflow_result.json",
        {
            "status": "completed",
            "executed_steps": 6,
            "failed_steps": 0,
            "manual_steps": 0,
            "steps": [
                {"step_id": "S0", "owner": "PlannerAgent", "wrapper_stage": "research_spec", "status": "completed", "payload": {"artifact_json": str(research_spec)}},
                {"step_id": "S1", "owner": "outer_loop", "wrapper_stage": "outer_loop", "status": "completed", "payload": {"artifact_json": str(Path(contract["workflow_root"]) / "outer_loop_summary.json")}},
                {
                    "step_id": "S2",
                    "owner": "VerificationAgent",
                    "wrapper_stage": "VerificationAgent",
                    "status": "completed",
                    "payload": {
                        "candidate_teachers": [{"round_id": "phase7_candidate_round", "title": "Phase 7 candidate"}],
                        "validated_teachers": [{"round_id": "phase7_validated_round", "title": "Phase 7 validated"}],
                    },
                },
                {
                    "step_id": "S3",
                    "owner": "teacher_frozen_eval",
                    "wrapper_stage": "teacher_frozen_eval",
                    "status": "completed",
                    "agent_summary": {"teachers": [{"round_id": "phase7_frozen_round", "title": "Phase 7 frozen"}]},
                },
                {"step_id": "S4", "owner": "TeacherSelectionAgent", "wrapper_stage": "TeacherSelectionAgent", "status": "completed", "payload": {"selected_spec_json": str(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json")}},
                {"step_id": "S5", "owner": "ApprenticeAgent", "wrapper_stage": "inner_loop_suite", "status": "completed", "agent_summary": {"final_lesson_artifact_json": str(final_lesson), "suite_summary_json": str(suite_summary)}},
            ],
        },
    )
    return contract


def _ensure_scoring_run_contract(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=PROFILE,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=False,
    )
    ensure_contract_dirs(contract)
    save_project_config(contract, {"profile_id": PROFILE, "project_id": contract["project_id"], "dataset_id": contract["dataset_id"], "run_id": contract["run_id"]})
    save_dataset_manifest(
        contract,
        {
            "source_type": "structured_signal_upload",
            "task_type": "scoring_only",
            "row_count": 1,
            "symbol_count": 1,
            "required_columns": ["signal_id", "date", "symbol", "signal_type", "features"],
            "missing_columns": [],
            "data_readiness": "ready_for_scoring_only",
            "data_isolation_status": contract["data_isolation"],
        },
    )
    save_run_spec(
        contract,
        {
            **contract,
            "profile_id": PROFILE,
            "mode": "scoring_only",
            "research_goal": "Phase 7 scoring-only UX rehearsal.",
            "lesson_alias": "alignment_seed0005",
            "api_model": "gpt-oss-20b",
        },
    )
    save_research_campaign(contract, {"mode": "scoring_only", "research_goal": "Phase 7 scoring-only UX rehearsal."})
    write_json(
        Path(contract["workflow_root"]) / "workflow_result.json",
        {
            "status": "completed",
            "executed_steps": 1,
            "failed_steps": 0,
            "manual_steps": 0,
            "steps": [
                {"step_id": "S-score", "owner": "SignalScoringAgent", "wrapper_stage": "SignalScoringAgent", "status": "completed", "payload": {}}
            ],
        },
    )
    write_json(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json", {"resolution_source": "", "fallback_reason": "", "teachers": []})
    return contract


def main() -> None:
    client = _client()
    signal = _sample_signal()
    cases: list[Dict[str, Any]] = []
    fixed_issues = [
        "空状态文案改为中文下一步指引：dataset_manifest / run_spec / workflow_result / final_lesson_set / scoring artifact。",
        "live scoring unavailable 文案明确说明当前没有 GPU / GPT-OSS runtime，不会启动模型。",
        "Simple Mode 增加术语说明卡，统一 Teacher Zoo、final_lesson_set、imported_final_asset、current_workflow_asset、fallback、no-model 模式解释。",
    ]
    remaining_ux_issues = [
        "Phase 7 仍是静态 HTML/JS；后续如果交互继续复杂化，建议拆分前端模块或迁移组件框架。",
        "Agent Activity Timeline 仍是 stage-level 状态，不代表每个 Agent 都有真实独立实时事件。",
        "no-model scoring 只能验证输入、来源和 artifact，不代表真实 GPT-OSS 打分质量。",
    ]

    imported_chat = _chat(client, "phase7-e2e", "imported-assets", "imported-demo", "我想用导入的 A 股 paper assets 做一个演示，不上传数据。")
    imported_action = _action(client, "phase7-e2e", "imported-assets", "imported-demo", "use_imported_demo_assets", imported_chat)
    imported_manifest = _get(client, "/console/dataset-manifest", **_ctx("phase7-e2e", "imported-assets", "imported-demo"))
    cases.append(
        {
            "name": "imported_asset_demo_path",
            "passed": imported_action["dataset_manifest"]["source_type"] in {"imported_assets", "imported_paper_assets"}
            and imported_manifest["source_type"] in {"imported_assets", "imported_paper_assets"}
            and imported_action["task_state"]["dataset_manifest_path"] == imported_manifest["dataset_manifest_json"],
            "task_type": imported_chat["task_state"]["task_type"],
            "dataset_manifest_path": imported_manifest["dataset_manifest_json"],
        }
    )

    full_contract = _create_no_model_full_pipeline_smoke("phase7-e2e", "full-pipeline", "smoke-full")
    run_monitor = _get(client, "/console/run-monitor", project_id=full_contract["project_id"], dataset_id=full_contract["dataset_id"], run_id=full_contract["run_id"])
    lesson_set = _get(client, "/console/lesson-set", project_id=full_contract["project_id"], dataset_id=full_contract["dataset_id"], run_id=full_contract["run_id"])
    run_status = _get(client, "/chat/run-status", profile=PROFILE, project_id=full_contract["project_id"], dataset_id=full_contract["dataset_id"], run_id=full_contract["run_id"])
    cases.append(
        {
            "name": "full_research_pipeline_no_model_smoke_path",
            "passed": run_monitor["workflow_status"] == "completed"
            and lesson_set["final_lesson_source"] == "current_workflow_asset"
            and run_status["task_card"]["latest_artifact"] == lesson_set["final_lesson_state_json"],
            "workflow_result": run_monitor["workflow_result_json"],
            "final_lesson_set": lesson_set["final_lesson_state_json"],
            "run_id_consistency": [run_monitor["contract"]["run_id"], lesson_set["contract"]["run_id"], full_contract["run_id"]],
        }
    )

    _ensure_scoring_run_contract("phase7-e2e", "signals", "scoring-dry-run")
    scoring_chat = _chat(client, "phase7-e2e", "signals", "scoring-dry-run", "我有一批结构化候选信号，想用导入 lesson 做 dry_run。")
    scoring_action = _action(
        client,
        "phase7-e2e",
        "signals",
        "scoring-dry-run",
        "score_signal",
        scoring_chat,
        scoring_payload={"mode": "dry_run", "lesson_source_mode": "imported", "pasted_payload": json.dumps(signal)},
    )
    scoring_result = scoring_action["scoring_result"]
    scoring_provenance_path = scoring_result["artifact_paths"]["scoring_provenance.json"]
    scoring_console = _get(client, "/console/provenance", project_id="phase7-e2e", dataset_id="signals", run_id="scoring-dry-run")
    scoring_raw = _get(client, "/console/artifact-json", path=scoring_provenance_path)
    cases.append(
        {
            "name": "scoring_only_path",
            "passed": scoring_result["signal_input_manifest"]["valid"] is True
            and scoring_raw["payload"]["model_called"] is False
            and scoring_raw["payload"]["lesson_source"] == "imported_final_asset"
            and scoring_provenance_path in scoring_console["artifact_files"],
            "task_type": scoring_chat["task_state"]["task_type"],
            "scoring_provenance_path": scoring_provenance_path,
            "simple_expert_same_run": scoring_console["contract"]["run_id"] == scoring_action["task_state"]["run_id"],
        }
    )

    _ensure_scoring_run_contract("phase7-e2e", "signals", "ohlcv-mismatch")
    ohlcv_chat = _chat(client, "phase7-e2e", "signals", "ohlcv-mismatch", "我想给候选信号打分。")
    ohlcv_csv = "date,symbol,open,high,low,close,volume,amount\n2025-01-02,600519,1,2,1,2,100,200\n"
    ohlcv_action = _action(
        client,
        "phase7-e2e",
        "signals",
        "ohlcv-mismatch",
        "score_signal",
        ohlcv_chat,
        file_payload={"filename": "ohlcv.csv", "content": ohlcv_csv, "content_encoding": "text"},
        scoring_payload={"mode": "dry_run", "lesson_source_mode": "imported"},
    )
    ohlcv_result = ohlcv_action["scoring_result"]
    cases.append(
        {
            "name": "ohlcv_mismatch_path",
            "passed": ohlcv_result["signal_input_manifest"]["valid"] is False
            and ohlcv_result["signal_input_manifest"]["mismatch_type"] == "ohlcv_market_data_not_signal_candidates"
            and "候选信号" in ohlcv_result["summary_zh"],
            "missing_columns": ohlcv_result["signal_input_manifest"]["missing_columns"],
            "summary_zh": ohlcv_result["summary_zh"],
        }
    )

    stock_guard = _chat(client, "phase7-e2e", "signals", "stock-code-only", "帮我看看 600519 最近还能不能买？")
    stock_text = json.dumps(stock_guard, ensure_ascii=False)
    cases.append(
        {
            "name": "stock_code_only_guard_path",
            "passed": stock_guard["task_state"]["task_type"] == "scoring_only"
            and "只有股票代码不足以评分" in stock_text
            and any(action["action_id"] == "score_signal" and not action["enabled"] for action in stock_guard["recommended_actions"]),
            "task_type": stock_guard["task_state"]["task_type"],
            "missing_fields": stock_guard["task_state"].get("missing_fields", []),
            "recommended_actions": stock_guard["recommended_actions"],
        }
    )

    _ensure_scoring_run_contract("phase7-e2e", "signals", "fallback-scoring")
    fallback_chat = _chat(client, "phase7-e2e", "signals", "fallback-scoring", "我想优先用当前 workflow lesson 打分，如果没有就明确 fallback 到导入 lesson。")
    fallback_action = _action(
        client,
        "phase7-e2e",
        "signals",
        "fallback-scoring",
        "score_signal",
        fallback_chat,
        scoring_payload={
            "mode": "dry_run",
            "lesson_source_mode": "current_workflow",
            "allow_lesson_fallback": True,
            "pasted_payload": json.dumps(signal),
        },
    )
    fallback_result = fallback_action["scoring_result"]
    fallback_provenance = fallback_result["scoring_provenance"]
    fallback_console = _get(client, "/console/provenance", project_id="phase7-e2e", dataset_id="signals", run_id="fallback-scoring")
    fallback_raw = _get(client, "/console/artifact-json", path=fallback_result["artifact_paths"]["scoring_provenance.json"])
    cases.append(
        {
            "name": "fallback_artifact_path",
            "passed": fallback_provenance["fallback_used"] is True
            and fallback_provenance["lesson_source"] == "imported_final_asset"
            and fallback_provenance["fallback_reason"] == fallback_raw["payload"]["fallback_reason"]
            and fallback_result["artifact_paths"]["scoring_provenance.json"] in fallback_console["artifact_files"],
            "fallback_reason": fallback_provenance["fallback_reason"],
            "simple_source": {
                "lesson_source": fallback_provenance["lesson_source"],
                "teacher_source": fallback_provenance["teacher_source"],
                "fallback_used": fallback_provenance["fallback_used"],
            },
            "expert_artifact": fallback_result["artifact_paths"]["scoring_provenance.json"],
        }
    )

    html = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "app.js").read_text(encoding="utf-8")
    terminology_terms = [
        "Teacher / Teacher Zoo",
        "final_lesson_set",
        "imported_final_asset",
        "current_workflow_asset",
        "fallback",
        "prompt_only / dry_run / mock / archived_replay",
    ]
    empty_state_terms = [
        "还没有 dataset_manifest",
        "请先生成 run_spec",
        "还没有 workflow_result",
        "当前 workflow 尚未生成 final_lesson_set",
        "当前 run 尚未产生 scoring artifacts",
        "当前没有可用 GPU / GPT-OSS runtime",
    ]
    cases.append(
        {
            "name": "terminology_and_empty_state_polish",
            "passed": all(term in html for term in terminology_terms)
            and all(term in js for term in empty_state_terms)
            and js.count("{") == js.count("}")
            and js.count("(") == js.count(")"),
            "terminology_terms": terminology_terms,
            "empty_state_terms": empty_state_terms,
        }
    )

    failed_checks = [case for case in cases if not case["passed"]]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 7 End-to-End UX Rehearsal + Consistency Polish",
        "passed": not failed_checks,
        "model_called": False,
        "vllm_started": False,
        "external_api_called": False,
        "clean_pipeline_modified": False,
        "walkthrough_cases": cases,
        "failed_checks": failed_checks,
        "fixed_issues": fixed_issues,
        "remaining_ux_issues": remaining_ux_issues,
    }
    report_dir = ROOT / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase7_e2e_ux_rehearsal_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    write_json(report_path, report)
    print(f"report={report_path}")
    print(f"passed={report['passed']}")
    if failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
