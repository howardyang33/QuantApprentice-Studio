from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

from quant_apprentice_studio.api.app import create_app
from quant_apprentice_studio.contracts import (
    build_run_contract,
    ensure_contract_dirs,
    save_research_campaign,
    save_run_spec,
)
from quant_apprentice_studio.provenance import write_json
from quant_apprentice_studio.simple_chat import build_agent_timeline_status_from_views


def _create_contract_run(
    *,
    project_id: str,
    dataset_id: str,
    run_id: str,
    mode: str,
    workflow_result: Dict[str, Any],
    teacher_selection_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id="gpt_oss_20b_final",
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=False,
    )
    ensure_contract_dirs(contract)
    save_run_spec(
        contract,
        {
            **contract,
            "profile_id": "gpt_oss_20b_final",
            "mode": mode,
            "research_goal": f"phase4 smoke {run_id}",
            "lesson_alias": "alignment_seed0005",
            "api_model": "gpt-oss-20b",
        },
    )
    save_research_campaign(
        contract,
        {
            "profile_id": "gpt_oss_20b_final",
            "project_id": contract["project_id"],
            "dataset_id": contract["dataset_id"],
            "run_id": contract["run_id"],
            "mode": mode,
            "research_goal": f"phase4 smoke {run_id}",
        },
    )
    payload = {
        **workflow_result,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "run_contract": contract,
        "workflow_root": contract["workflow_root"],
        "shared_context_root": contract["shared_context_root"],
        "profile_id": contract["profile_id"],
    }
    write_json(Path(contract["workflow_root"]) / "workflow_result.json", payload)
    write_json(
        Path(contract["workflow_root"]) / "teacher_selection_summary_final.json",
        teacher_selection_summary or {"resolution_source": "current_workflow_asset", "fallback_reason": "", "teachers": []},
    )
    return contract


def _get(client: TestClient, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    response = client.get(
        "/chat/run-status",
        params={
            "profile": "gpt_oss_20b_final",
            "project_id": project_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    client = TestClient(create_app())
    report_dir = Path("test_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase4_agent_timeline_smoke_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"

    cases: list[Dict[str, Any]] = []

    completed = _get(client, "console-probe", "dataset-a", "cockpit-smoke-001")
    cases.append(
        {
            "name": "completed_workflow_timeline",
            "passed": completed["workflow_status"] == "completed" and len(completed["timeline_nodes"]) == 10,
            "workflow_status": completed["workflow_status"],
            "completed_stages": completed["task_card"]["completed_stages"],
            "node_statuses": {node["agent_name"]: node["status"] for node in completed["timeline_nodes"]},
        }
    )

    _create_contract_run(
        project_id="phase4-smoke-fallback",
        dataset_id="dataset-a",
        run_id="fallback-001",
        mode="full_pipeline",
        workflow_result={
            "status": "completed",
            "executed_steps": 1,
            "failed_steps": 0,
            "manual_steps": 0,
            "steps": [
                {
                    "step_id": "S1",
                    "owner": "TeacherSelectionAgent",
                    "wrapper_stage": "TeacherSelectionAgent",
                    "status": "completed",
                    "payload": {"selected_spec_json": "/tmp/imported_selection.json"},
                }
            ],
        },
        teacher_selection_summary={
            "resolution_source": "imported_final_asset",
            "fallback_reason": "No current workflow frozen teacher passed smoke criteria; imported frozen teacher fallback used.",
            "teachers": [{"round_id": "round_038_frozen_2022", "report_dir": "/tmp/imported_report"}],
        },
    )
    fallback = _get(client, "phase4-smoke-fallback", "dataset-a", "fallback-001")
    cases.append(
        {
            "name": "fallback_artifact_timeline",
            "passed": fallback["task_card"]["fallback_used"] is True and "imported" in fallback["task_card"]["fallback_reason"].lower(),
            "workflow_status": fallback["workflow_status"],
            "fallback_reason": fallback["task_card"]["fallback_reason"],
            "node_statuses": {node["agent_name"]: node["status"] for node in fallback["timeline_nodes"]},
        }
    )

    _create_contract_run(
        project_id="phase4-smoke-failed",
        dataset_id="dataset-a",
        run_id="failed-001",
        mode="full_pipeline",
        workflow_result={
            "status": "failed",
            "executed_steps": 1,
            "failed_steps": 1,
            "manual_steps": 0,
            "steps": [
                {
                    "step_id": "S2",
                    "owner": "outer_loop",
                    "wrapper_stage": "outer_loop",
                    "status": "failed",
                    "error": "Synthetic smoke failure: missing benchmark index file.",
                    "payload": {"artifact_json": "/tmp/outer_loop_failed.json"},
                }
            ],
        },
    )
    failed = _get(client, "phase4-smoke-failed", "dataset-a", "failed-001")
    failed_nodes = [node for node in failed["timeline_nodes"] if node["status"] == "failed"]
    cases.append(
        {
            "name": "failed_workflow_timeline",
            "passed": failed["workflow_status"] == "failed" and bool(failed_nodes) and "missing benchmark" in failed_nodes[0]["error_message"],
            "workflow_status": failed["workflow_status"],
            "failed_stage": failed["task_card"]["failed_stage"],
            "error_message": failed_nodes[0]["error_message"] if failed_nodes else "",
            "node_statuses": {node["agent_name"]: node["status"] for node in failed["timeline_nodes"]},
        }
    )

    running_mock = build_agent_timeline_status_from_views(
        run_monitor={
            "contract": {"project_id": "ui-mock", "dataset_id": "dataset-a", "run_id": "running-001", "mode": "full_pipeline"},
            "workflow_status": "running",
            "nodes": [
                {"node_id": "research_spec", "label": "Research Spec", "status": "completed", "payload": {"artifact_json": "/tmp/research_spec.json"}},
                {"node_id": "outer_loop", "label": "Outer Loop", "status": "running", "payload": {"artifact_json": "/tmp/outer_loop_progress.json"}},
            ],
        },
        provenance={
            "contract": {"project_id": "ui-mock", "dataset_id": "dataset-a", "run_id": "running-001", "mode": "full_pipeline"},
            "workflow_status": "running",
            "run_spec_json": "/tmp/run_spec.json",
            "research_campaign_json": "/tmp/research_campaign.json",
            "artifact_files": [],
            "teacher_selection_summary": {},
        },
        lesson_set={"final_lesson_source": "", "final_lesson_state_json": "", "teacher_scopes": []},
        ui_mock=True,
    )
    cases.append(
        {
            "name": "running_ui_mock_timeline",
            "passed": running_mock["ui_mock"] is True and any(node["status"] == "running" for node in running_mock["timeline_nodes"]),
            "ui_mock": True,
            "note": "UI mock only. This case is not written to formal workflow provenance.",
            "node_statuses": {node["agent_name"]: node["status"] for node in running_mock["timeline_nodes"]},
        }
    )

    valid_workspaces = {"simple", "full-pipeline", "library", "provenance", "advanced", "scoring"}
    click_case_nodes = completed["timeline_nodes"]
    click_links_ok = all(node.get("expert_link") in valid_workspaces for node in click_case_nodes)
    cases.append(
        {
            "name": "timeline_node_expert_links",
            "passed": click_links_ok,
            "validated_links": {node["agent_name"]: node.get("expert_link") for node in click_case_nodes},
            "note": "Frontend click handler uses data-expert-link to call setWorkspace().",
        }
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 4 Simple Mode Agent Activity Timeline",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "passed": report["passed"], "case_results": [(case["name"], case["passed"]) for case in cases]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
