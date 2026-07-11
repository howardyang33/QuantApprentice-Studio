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
from quant_apprentice_studio.provenance import write_json
from quant_apprentice_studio.registry import StudioRegistry


PROFILE = "gpt_oss_20b_final"
ROOT = Path(__file__).resolve().parents[1]


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


def _post(client: TestClient, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    return response.json()


def _get(client: TestClient, endpoint: str, **params: Any) -> Dict[str, Any]:
    response = client.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


def _chat(client: TestClient, project_id: str, dataset_id: str, run_id: str, message: str) -> Dict[str, Any]:
    return _post(client, "/chat/message", {**_ctx(project_id, dataset_id, run_id), "mode": "simple", "message": message, "attachments": []})


def _action(client: TestClient, project_id: str, dataset_id: str, run_id: str, action_id: str, chat_payload: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return _post(
        client,
        "/chat/action",
        {
            **_ctx(project_id, dataset_id, run_id),
            "session_id": chat_payload["session"]["session_id"],
            "action_id": action_id,
            "confirm": True,
            "task_state": chat_payload["task_state"],
            **extra,
        },
    )


def _sample_signal() -> Dict[str, Any]:
    registry = StudioRegistry(PROFILE)
    agent = SignalScoringAgent(registry)
    market_alias = str(registry.load_runtime_catalog()["defaults"]["market_run_alias"])
    sample = agent.sample_signal_record(market_run_alias=market_alias)
    return {
        "signal_id": "phase8-signal-001",
        "date": str(sample.get("signal_date") or ""),
        "symbol": str(sample.get("symbol") or ""),
        "signal_type": "phase8_structured_signal",
        **sample,
    }


def _ensure_scoring_contract(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
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
            "research_goal": "Phase 8 UI polish scoring rehearsal.",
            "lesson_alias": "alignment_seed0005",
            "api_model": "gpt-oss-20b",
        },
    )
    save_research_campaign(contract, {"mode": "scoring_only", "research_goal": "Phase 8 UI polish scoring rehearsal."})
    write_json(
        Path(contract["workflow_root"]) / "workflow_result.json",
        {
            "status": "completed",
            "executed_steps": 1,
            "failed_steps": 0,
            "manual_steps": 0,
            "steps": [{"step_id": "S-score", "owner": "SignalScoringAgent", "wrapper_stage": "SignalScoringAgent", "status": "completed", "payload": {}}],
        },
    )
    write_json(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json", {"resolution_source": "", "fallback_reason": "", "teachers": []})
    return contract


def main() -> None:
    client = _client()
    html = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "styles.css").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    signal = _sample_signal()
    cases: list[Dict[str, Any]] = []

    app_html = client.get("/app")
    app_html.raise_for_status()
    cases.append(
        {
            "name": "simple_mode_first_screen_empty_state",
            "passed": "你今天想做什么？" in app_html.text
            and app_html.text.count("quick-entry-card") >= 4
            and "中文智能研究助手" in app_html.text,
            "quick_entry_count": app_html.text.count("quick-entry-card"),
        }
    )

    chat = _chat(client, "phase8-ui", "imported-demo", "entry", "我想使用论文导入资产做一个演示。")
    cases.append(
        {
            "name": "chinese_chat_bubble_and_actions",
            "passed": chat["task_state"]["task_type"] == "imported_asset_demo"
            and bool(chat["recommended_actions"])
            and ".chat-bubble.user" in css
            and ".chat-bubble.assistant" in css
            and "chat-bubble assistant typing" in js,
            "task_type": chat["task_state"]["task_type"],
            "recommended_actions": [row["action_id"] for row in chat["recommended_actions"]],
        }
    )

    imported_action = _action(client, "phase8-ui", "imported-demo", "entry", "use_imported_demo_assets", chat)
    cases.append(
        {
            "name": "recommended_actions_button_execution",
            "passed": imported_action["dataset_manifest"]["source_type"] in {"imported_paper_assets", "imported_assets"}
            and imported_action["task_state"]["next_action"] in {"generate_run_spec", "score_signal", "open_expert_monitor"},
            "dataset_manifest_path": imported_action["task_state"]["dataset_manifest_path"],
        }
    )

    cases.append(
        {
            "name": "file_upload_card_display",
            "passed": "simple-chat-upload-file" in html
            and "composer-file-button" in html
            and "simple-upload-file" in html
            and "simple-chat-upload-name" in js,
            "upload_controls": ["simple-chat-upload-file", "simple-upload-file"],
        }
    )

    fallback_contract = _ensure_scoring_contract("phase8-ui", "signals", "fallback")
    fallback_chat = _chat(client, "phase8-ui", "signals", "fallback", "我想优先用当前流程资产打分，如果没有最终经验规则集就回退到导入论文资产。")
    fallback_action = _action(
        client,
        "phase8-ui",
        "signals",
        "fallback",
        "score_signal",
        fallback_chat,
        scoring_payload={
            "mode": "dry_run",
            "lesson_source_mode": "current_workflow",
            "allow_lesson_fallback": True,
            "pasted_payload": json.dumps(signal),
        },
    )
    provenance = fallback_action["scoring_result"]["scoring_provenance"]
    expert_provenance = _get(client, "/console/provenance", project_id=fallback_contract["project_id"], dataset_id=fallback_contract["dataset_id"], run_id=fallback_contract["run_id"])
    cases.append(
        {
            "name": "fallback_warning_display_and_source_transparency",
            "passed": provenance["fallback_used"] is True
            and provenance["lesson_source"] == "imported_final_asset"
            and "fallback-card" in css
            and "Fallback Reason" in js
            and fallback_action["scoring_result"]["artifact_paths"]["scoring_provenance.json"] in expert_provenance["artifact_files"],
            "fallback_reason": provenance["fallback_reason"],
            "lesson_source": provenance["lesson_source"],
            "teacher_source": provenance["teacher_source"],
        }
    )

    cases.append(
        {
            "name": "timeline_style_and_detail_readability",
            "passed": "timeline-agent-node.running" in css
            and "timeline-agent-node.completed" in css
            and "timeline-agent-node.failed" in css
            and "timeline-agent-node.fallback" in css
            and "timeline-detail-head" in js
            and "阶段级映射" in html,
            "timeline_classes": ["running", "completed", "failed", "fallback"],
        }
    )

    demo_paths = [
        "Imported asset demo scoring dry run",
        "Upload OHLCV mismatch / full pipeline data guidance",
        "Existing run timeline + Teacher Zoo + Lesson Lab + Audit Trail review",
    ]
    cases.append(
        {
            "name": "demo_paths_documented",
            "passed": all(path in readme for path in demo_paths)
            and "prompt_only" in readme
            and "dry_run" in readme
            and "archived_replay" in readme,
            "demo_paths": demo_paths,
        }
    )

    expert_ids = [
        "dataset-lab-summary",
        "workflow-lab-chain",
        "teacher-lab-imported",
        "lesson-lab-current-summary",
        "scoring-lab-summary",
        "audit-chain",
        "runspec-form",
    ]
    project_view = _get(client, "/console/project", profile=PROFILE, project_id="phase8-ui", dataset_id="signals", run_id="fallback")
    cases.append(
        {
            "name": "expert_mode_not_broken",
            "passed": all(f'id="{dom_id}"' in html for dom_id in expert_ids)
            and project_view["project_id"] == "phase8-ui"
            and "Advanced Console 仅供研究员和开发调试使用" in html,
            "expert_ids": expert_ids,
        }
    )

    failed_checks = [case for case in cases if not case["passed"]]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 8 Product UI Polish + Demo Readiness",
        "passed": not failed_checks,
        "model_called": False,
        "vllm_started": False,
        "external_api_called": False,
        "clean_pipeline_modified": False,
        "walkthrough_cases": cases,
        "failed_checks": failed_checks,
        "fixed_issues": [
            "Simple Mode first screen now offers four clear task entry cards.",
            "Chat composer includes adjacent file upload control and typing/loading state.",
            "Right task state rail is grouped into task/data/workflow/scoring/next-action sections.",
            "Artifact links render as readable cards instead of bare paths.",
            "README documents no-model demo paths and current limitations.",
        ],
        "remaining_ux_issues": [
            "Still a single static HTML/JS file; future large UX work should split modules.",
            "File upload is client-side to API payload; large production uploads should move to streaming/multipart.",
            "Live GPT-OSS validation remains disabled until GPU/runtime is available.",
        ],
    }
    report_dir = ROOT / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase8_ui_polish_demo_readiness_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    write_json(report_path, report)
    print(f"report={report_path}")
    print(f"passed={report['passed']}")
    if failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
