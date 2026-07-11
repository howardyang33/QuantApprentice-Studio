from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

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


def _contract(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=PROFILE,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=False,
    )
    ensure_contract_dirs(contract)
    return contract


def _imported_lesson_payload() -> Dict[str, Any]:
    catalog = StudioRegistry(PROFILE).load_runtime_catalog()
    imported_path = Path(catalog["lesson_runs"]["alignment_seed0005"]["final_lesson_state_json"])
    return read_json(imported_path)


def _write_common_run_files(contract: Dict[str, Any], *, mode: str, research_goal: str) -> None:
    save_project_config(
        contract,
        {
            "profile_id": PROFILE,
            "project_id": contract["project_id"],
            "dataset_id": contract["dataset_id"],
            "run_id": contract["run_id"],
            "allow_imported_fallback": contract["allow_imported_fallback"],
            "allow_demo_fallback": contract["allow_demo_fallback"],
        },
    )
    save_run_spec(
        contract,
        {
            **contract,
            "profile_id": PROFILE,
            "mode": mode,
            "research_goal": research_goal,
            "lesson_alias": "alignment_seed0005",
            "api_model": "gpt-oss-20b",
        },
    )
    save_research_campaign(
        contract,
        {
            "profile_id": PROFILE,
            "project_id": contract["project_id"],
            "dataset_id": contract["dataset_id"],
            "run_id": contract["run_id"],
            "mode": mode,
            "research_goal": research_goal,
            "pipeline_plan": {"steps": ["outer_loop", "teacher_frozen_eval", "selection", "inner_loop", "scoring"]},
        },
    )


def _create_completed_run() -> Dict[str, Any]:
    contract = _contract("phase6-expert-mode", "dataset-a", "reorg-001")
    _write_common_run_files(contract, mode="full_pipeline", research_goal="Phase 6 Expert Mode complete smoke run.")
    save_dataset_manifest(
        contract,
        {
            "source_type": "upload_local",
            "market": "A-share",
            "project_id": contract["project_id"],
            "dataset_id": contract["dataset_id"],
            "row_count": 128,
            "symbol_count": 12,
            "date_range": {"start": "2020-01-02", "end": "2022-12-30"},
            "required_columns": ["date", "symbol", "open", "high", "low", "close", "volume", "amount"],
            "missing_columns": [],
            "warning_reasons": [],
            "fail_reasons": [],
            "data_readiness": "ready_for_full_research_pipeline",
            "data_isolation_status": contract["data_isolation"],
        },
    )
    studio_control = Path(contract["shared_context_root"]) / "studio_control"
    studio_control.mkdir(parents=True, exist_ok=True)
    research_spec = studio_control / "research_spec.json"
    write_json(research_spec, {"research_goal": "Phase 6 Expert Mode complete smoke run.", "source": "smoke_test"})

    current_lesson = Path(contract["lesson_root"]) / "phase6_current_workflow_final_lesson.json"
    write_json(current_lesson, _imported_lesson_payload())
    suite_summary = Path(contract["lesson_root"]) / "phase6_inner_loop_suite_summary.json"
    write_json(suite_summary, {"teacher_scope_count": 4, "final_lesson_artifact_json": str(current_lesson)})

    scoring_dir = Path(contract["scoring_root"]) / "phase6_prompt_only"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    scoring_provenance = scoring_dir / "scoring_provenance.json"
    write_json(
        scoring_provenance,
        {
            "mode": "prompt_only",
            "model_called": False,
            "result_valid_for_research": False,
            "lesson_source": "current_workflow_asset",
            "teacher_source": "current_workflow_asset",
            "fallback_used": False,
            "fallback_reason": "",
            "imported_final_asset": False,
            "current_workflow_asset": True,
            "demo_asset": False,
        },
    )
    write_json(scoring_dir / "signal_input_manifest.json", {"valid": True, "record_count": 1, "missing_columns": []})
    write_json(scoring_dir / "scoring_prompt_preview.json", {"model_called": False, "prompt_preview": "not executed"})
    write_json(scoring_dir / "scoring_summary_zh.json", {"summary_zh": "Phase 6 smoke no-model scoring artifact."})

    candidate = {
        "round_id": "phase6_candidate_round",
        "title": "Smoke Candidate Teacher",
        "research_family": "Breakout smoke",
        "sample_template": "top25bottom25",
        "target_kind": "ret_5",
        "mean_alpha": 0.0123,
        "nav_cagr": 0.24,
        "selection_reason": "synthetic smoke candidate",
    }
    frozen = {
        **candidate,
        "round_id": "phase6_frozen_round",
        "frozen_round_id": "phase6_frozen_round",
        "positive_years": 3,
        "total_years": 3,
        "nav_max_drawdown": -0.08,
        "uplift_mean": 0.006,
        "report_dir": str(Path(contract["teacher_zoo_root"]) / "phase6_frozen_round" / "report_v2"),
    }
    selection_summary = {
        "resolution_source": "current_workflow_asset",
        "fallback_reason": "",
        "teachers": [
            {
                **frozen,
                "source_round_id": "phase6_candidate_round",
                "report_dir": str(Path(contract["teacher_zoo_root"]) / "phase6_frozen_round" / "report_v2"),
            }
        ],
    }
    write_json(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json", selection_summary)
    workflow_result = {
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
                "payload": {"candidate_teachers": [candidate], "validated_teachers": [{**candidate, "round_id": "phase6_validated_round"}]},
            },
            {
                "step_id": "S3",
                "owner": "teacher_frozen_eval",
                "wrapper_stage": "teacher_frozen_eval",
                "status": "completed",
                "agent_summary": {"teachers": [frozen], "frozen_eval_artifact_json": str(Path(contract["workflow_root"]) / "frozen_eval.json")},
            },
            {"step_id": "S4", "owner": "TeacherSelectionAgent", "wrapper_stage": "TeacherSelectionAgent", "status": "completed", "payload": {"selected_spec_json": str(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json")}},
            {"step_id": "S5", "owner": "ApprenticeAgent", "wrapper_stage": "inner_loop_suite", "status": "completed", "agent_summary": {"final_lesson_artifact_json": str(current_lesson), "suite_summary_json": str(suite_summary)}},
            {"step_id": "S6", "owner": "SignalScoringAgent", "wrapper_stage": "SignalScoringAgent", "status": "completed", "payload": {"artifact_json": str(scoring_provenance)}},
        ],
    }
    write_json(Path(contract["workflow_root"]) / "workflow_result.json", workflow_result)
    return contract


def _create_fallback_run() -> Dict[str, Any]:
    contract = _contract("phase6-expert-mode", "dataset-fallback", "fallback-001")
    _write_common_run_files(contract, mode="full_pipeline", research_goal="Phase 6 fallback smoke run.")
    save_dataset_manifest(contract, {"source_type": "upload_local", "row_count": 8, "missing_columns": [], "data_readiness": "ready", "data_isolation_status": contract["data_isolation"]})
    imported_report = str(Path(contract["imported_asset_root"]) / "teacher_zoo" / "round_038" / "report_v2")
    write_json(
        Path(contract["workflow_root"]) / "teacher_selection_summary_final.json",
        {
            "resolution_source": "imported_final_asset",
            "fallback_reason": "No current workflow frozen teacher passed smoke criteria; imported frozen teacher fallback used.",
            "teachers": [{"round_id": "round_038", "title": "Imported fallback teacher", "report_dir": imported_report}],
        },
    )
    write_json(
        Path(contract["workflow_root"]) / "workflow_result.json",
        {
            "status": "completed",
            "executed_steps": 1,
            "failed_steps": 0,
            "manual_steps": 0,
            "steps": [
                {"step_id": "S4", "owner": "TeacherSelectionAgent", "wrapper_stage": "TeacherSelectionAgent", "status": "completed", "payload": {"selected_spec_json": str(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json")}},
            ],
        },
    )
    return contract


def _create_failed_run() -> Dict[str, Any]:
    contract = _contract("phase6-expert-mode", "dataset-failed", "failed-001")
    _write_common_run_files(contract, mode="full_pipeline", research_goal="Phase 6 failed smoke run.")
    save_dataset_manifest(contract, {"source_type": "upload_local", "row_count": 4, "missing_columns": ["amount"], "data_readiness": "not_ready", "data_isolation_status": contract["data_isolation"]})
    write_json(Path(contract["workflow_root"]) / "teacher_selection_summary_final.json", {"resolution_source": "", "fallback_reason": "", "teachers": []})
    write_json(
        Path(contract["workflow_root"]) / "workflow_result.json",
        {
            "status": "failed",
            "executed_steps": 1,
            "failed_steps": 1,
            "manual_steps": 0,
            "steps": [
                {
                    "step_id": "S1",
                    "owner": "outer_loop",
                    "wrapper_stage": "outer_loop",
                    "status": "failed",
                    "error": "Synthetic Phase 6 smoke failure: missing dataset factor columns.",
                    "payload": {"artifact_json": str(Path(contract["workflow_root"]) / "outer_loop_failed.json")},
                }
            ],
        },
    )
    return contract


def _get(client: TestClient, endpoint: str, **params: Any) -> Dict[str, Any]:
    response = client.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


def main() -> None:
    completed_contract = _create_completed_run()
    fallback_contract = _create_fallback_run()
    failed_contract = _create_failed_run()
    client = TestClient(create_app())
    ctx = {
        "profile": PROFILE,
        "project_id": completed_contract["project_id"],
        "dataset_id": completed_contract["dataset_id"],
        "run_id": completed_contract["run_id"],
    }

    cases: list[Dict[str, Any]] = []
    manifest = _get(client, "/console/dataset-manifest", **ctx)
    cases.append(
        {
            "name": "dataset_lab_manifest",
            "passed": manifest.get("data_readiness") == "ready_for_full_research_pipeline"
            and manifest.get("missing_columns") == []
            and manifest.get("data_isolation_status", {}).get("isolated_from_imported_assets") is True,
            "manifest_path": manifest.get("dataset_manifest_json"),
            "readiness": manifest.get("data_readiness"),
            "missing_columns": manifest.get("missing_columns"),
        }
    )

    monitor = _get(client, "/console/run-monitor", project_id=ctx["project_id"], dataset_id=ctx["dataset_id"], run_id=ctx["run_id"])
    failed_monitor = _get(client, "/console/run-monitor", project_id=failed_contract["project_id"], dataset_id=failed_contract["dataset_id"], run_id=failed_contract["run_id"])
    fallback_zoo = _get(client, "/console/teacher-zoo", profile=PROFILE, project_id=fallback_contract["project_id"], dataset_id=fallback_contract["dataset_id"], run_id=fallback_contract["run_id"])
    cases.append(
        {
            "name": "workflow_monitor_completed_failed_fallback",
            "passed": monitor.get("workflow_status") == "completed"
            and failed_monitor.get("workflow_status") == "failed"
            and bool(fallback_zoo.get("fallback_reason")),
            "completed_status": monitor.get("workflow_status"),
            "failed_status": failed_monitor.get("workflow_status"),
            "fallback_reason": fallback_zoo.get("fallback_reason"),
        }
    )

    zoo = _get(client, "/console/teacher-zoo", **ctx)
    cases.append(
        {
            "name": "teacher_zoo_lifecycle",
            "passed": bool(zoo.get("imported_teachers"))
            and bool(zoo.get("current_workflow_candidate_teachers"))
            and bool(zoo.get("current_workflow_validated_teachers"))
            and bool(zoo.get("current_workflow_frozen_teachers"))
            and bool(zoo.get("selected_teachers_for_inner_loop")),
            "imported_count": len(zoo.get("imported_teachers") or []),
            "candidate_count": len(zoo.get("current_workflow_candidate_teachers") or []),
            "validated_count": len(zoo.get("current_workflow_validated_teachers") or []),
            "frozen_count": len(zoo.get("current_workflow_frozen_teachers") or []),
            "selected_count": len(zoo.get("selected_teachers_for_inner_loop") or []),
        }
    )

    lessons = _get(client, "/lessons", profile=PROFILE, alias="alignment_seed0005")
    lesson_set = _get(client, "/console/lesson-set", project_id=ctx["project_id"], dataset_id=ctx["dataset_id"], run_id=ctx["run_id"])
    cases.append(
        {
            "name": "lesson_lab_imported_and_current",
            "passed": lessons.get("alias") == "alignment_seed0005"
            and lesson_set.get("final_lesson_source") == "current_workflow_asset"
            and lesson_set.get("teacher_scope_count", 0) > 0,
            "imported_alias": lessons.get("alias"),
            "current_source": lesson_set.get("final_lesson_source"),
            "teacher_scope_count": lesson_set.get("teacher_scope_count"),
        }
    )

    provenance = _get(client, "/console/provenance", project_id=ctx["project_id"], dataset_id=ctx["dataset_id"], run_id=ctx["run_id"])
    scoring_paths = [path for path in provenance.get("artifact_files", []) if "scoring_provenance" in path]
    scoring_json = _get(client, "/console/artifact-json", path=scoring_paths[0]) if scoring_paths else {}
    scoring_payload = scoring_json.get("payload") or {}
    cases.append(
        {
            "name": "scoring_lab_no_model_artifacts",
            "passed": bool(scoring_paths)
            and scoring_payload.get("model_called") is False
            and scoring_payload.get("result_valid_for_research") is False,
            "scoring_provenance_path": scoring_paths[0] if scoring_paths else "",
            "model_called": scoring_payload.get("model_called"),
            "result_valid_for_research": scoring_payload.get("result_valid_for_research"),
        }
    )

    raw_workflow = _get(client, "/console/artifact-json", path=provenance["workflow_result_json"])
    cases.append(
        {
            "name": "audit_trail_chain_and_raw_json",
            "passed": bool(provenance.get("artifact_files"))
            and provenance.get("workflow_status") == "completed"
            and raw_workflow.get("payload", {}).get("status") == "completed",
            "artifact_count": len(provenance.get("artifact_files") or []),
            "workflow_result_json": provenance.get("workflow_result_json"),
        }
    )

    html = (ROOT / "src" / "quant_apprentice_studio" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    required_ids = [
        "dataset-lab-summary",
        "workflow-lab-chain",
        "teacher-lab-imported",
        "teacher-lab-candidate",
        "lesson-lab-current-summary",
        "scoring-lab-summary",
        "audit-chain",
        "audit-raw",
        "runspec-form",
    ]
    cases.append(
        {
            "name": "expert_mode_static_dom",
            "passed": all(f'id="{dom_id}"' in html for dom_id in required_ids) and "Advanced Console 仅供研究员和开发调试使用" in html,
            "required_ids": required_ids,
        }
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 6 Expert Mode Reorganization",
        "passed": all(row["passed"] for row in cases),
        "model_called": False,
        "vllm_started": False,
        "external_api_called": False,
        "clean_pipeline_modified": False,
        "completed_run": {
            "project_id": completed_contract["project_id"],
            "dataset_id": completed_contract["dataset_id"],
            "run_id": completed_contract["run_id"],
        },
        "fallback_run": {
            "project_id": fallback_contract["project_id"],
            "dataset_id": fallback_contract["dataset_id"],
            "run_id": fallback_contract["run_id"],
        },
        "failed_run": {
            "project_id": failed_contract["project_id"],
            "dataset_id": failed_contract["dataset_id"],
            "run_id": failed_contract["run_id"],
        },
        "cases": cases,
    }
    report_dir = ROOT / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase6_expert_mode_reorg_smoke_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    write_json(report_path, report)
    print(f"report={report_path}")
    print(f"passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
