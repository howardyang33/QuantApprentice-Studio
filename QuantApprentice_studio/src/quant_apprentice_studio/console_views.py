from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .contracts import (
    build_run_contract,
    dataset_manifest_path,
    list_run_specs,
    project_config_path,
    research_campaign_path,
    run_spec_path,
    save_research_campaign,
    save_run_spec,
)
from .paths import import_root, studio_root
from .provenance import read_json


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _workflow_result_path(contract: Mapping[str, Any]) -> Path:
    return Path(str(contract["workflow_root"])).expanduser().resolve() / "workflow_result.json"


def _workflow_launch_status_path(contract: Mapping[str, Any]) -> Path:
    return Path(str(contract["workflow_root"])).expanduser().resolve() / "workflow_launch_status.json"


def _teacher_selection_summary_path(contract: Mapping[str, Any]) -> Path:
    return Path(str(contract["workflow_root"])).expanduser().resolve() / "teacher_selection_summary_final.json"


def _step_lookup(workflow_result: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(step.get("wrapper_stage") or step.get("owner") or step.get("step_id")): dict(step)
        for step in list(workflow_result.get("steps") or [])
    }


def _source_type(path: str, *, workflow_root: Path, imported_root: Path) -> str:
    if not path:
        return ""
    p = Path(path).expanduser().resolve()
    if imported_root == p or imported_root in p.parents:
        return "imported_final_asset"
    if workflow_root == p or workflow_root in p.parents:
        return "current_workflow_asset"
    return "external_asset"


def build_project_view(
    *,
    profile_id: str,
    project_id: str = "default-project",
    dataset_id: str = "default-dataset",
    run_id: str = "draft-run",
    allow_imported_fallback: bool = True,
    allow_demo_fallback: bool = False,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    imported_root = Path(contract["imported_asset_root"]).expanduser().resolve()
    existing_runs = list_run_specs()
    return {
        "profile_id": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "draft_run_id": contract["run_id"],
        "dataset_root": contract["dataset_root"],
        "dataset_raw_root": contract["dataset_raw_root"],
        "dataset_stock_klines_root": contract["dataset_stock_klines_root"],
        "dataset_index_klines_root": contract["dataset_index_klines_root"],
        "dataset_cache_root": contract["dataset_cache_root"],
        "dataset_jobs_root": contract["dataset_jobs_root"],
        "dataset_upload_root": contract["dataset_upload_root"],
        "asset_root": contract["asset_root"],
        "teacher_zoo_root": contract["teacher_zoo_root"],
        "lesson_root": contract["lesson_root"],
        "scoring_root": contract["scoring_root"],
        "workflow_root": contract["workflow_root"],
        "shared_context_root": contract["shared_context_root"],
        "imported_asset_root": str(imported_root),
        "allow_imported_fallback": bool(contract["allow_imported_fallback"]),
        "allow_demo_fallback": bool(contract["allow_demo_fallback"]),
        "data_isolation": dict(contract["data_isolation"]),
        "project_config_json": str(project_config_path(contract)),
        "dataset_manifest_json": str(dataset_manifest_path(contract)),
        "existing_run_count": len(existing_runs),
        "existing_runs": existing_runs[-8:],
    }


def create_run_spec(
    *,
    profile_id: str,
    mode: str,
    research_goal: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    plan: Mapping[str, Any],
    selection_json: str = "",
    final_lesson_state_json: str = "",
    lesson_alias: str = "",
    api_model: str = "",
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    run_spec = {
        **contract,
        "profile_id": profile_id,
        "mode": str(mode).strip(),
        "research_goal": str(research_goal).strip(),
        "selection_json_hint": str(selection_json).strip(),
        "final_lesson_state_json_hint": str(final_lesson_state_json).strip(),
        "lesson_alias": str(lesson_alias).strip(),
        "api_model": str(api_model).strip(),
    }
    campaign = {
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "profile_id": profile_id,
        "mode": str(mode).strip(),
        "research_goal": str(research_goal).strip(),
        "pipeline_plan": dict(plan),
        "policy": {
            "allow_imported_fallback": bool(allow_imported_fallback),
            "allow_demo_fallback": bool(allow_demo_fallback),
        },
        "artifact_contract": {
            key: run_spec[key]
            for key in [
                "asset_root",
                "teacher_zoo_root",
                "lesson_root",
                "scoring_root",
                "workflow_root",
                "shared_context_root",
            ]
        },
    }
    run_spec_json = save_run_spec(contract, run_spec)
    research_campaign_json = save_research_campaign(contract, campaign)
    return {
        "run_spec": run_spec,
        "run_spec_json": run_spec_json,
        "research_campaign_json": research_campaign_json,
        "pipeline_plan": dict(plan),
    }


def load_run_bundle(*, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    lookup_contract = build_run_contract(
        profile_id="gpt_oss_20b_final",
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=True,
        allow_demo_fallback=True,
    )
    stored_run_spec = _safe_read_json(run_spec_path(lookup_contract))
    if not stored_run_spec:
        raise FileNotFoundError(f"Run spec not found for {project_id}/{dataset_id}/{run_id}")
    # Older studio smoke-test run_spec files predate the full project/dataset
    # contract and may miss fields such as dataset_root. Rehydrate the contract
    # from the URL identity, then overlay stored run-specific metadata.
    contract = {
        **build_run_contract(
            profile_id=str(stored_run_spec.get("profile_id") or stored_run_spec.get("profile") or "gpt_oss_20b_final"),
            project_id=str(stored_run_spec.get("project_id") or project_id),
            dataset_id=str(stored_run_spec.get("dataset_id") or dataset_id),
            run_id=str(stored_run_spec.get("run_id") or run_id),
            allow_imported_fallback=bool(stored_run_spec.get("allow_imported_fallback", True)),
            allow_demo_fallback=bool(stored_run_spec.get("allow_demo_fallback", True)),
        ),
        **stored_run_spec,
    }
    workflow_result = _safe_read_json(_workflow_result_path(contract))
    workflow_launch_status = _safe_read_json(_workflow_launch_status_path(contract))
    teacher_selection_summary = _safe_read_json(_teacher_selection_summary_path(contract))
    research_campaign = _safe_read_json(research_campaign_path(contract))
    return {
        "contract": contract,
        "research_campaign": research_campaign,
        "workflow_result": workflow_result,
        "workflow_launch_status": workflow_launch_status,
        "teacher_selection_summary": teacher_selection_summary,
    }


def build_run_monitor(*, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    bundle = load_run_bundle(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    contract = dict(bundle["contract"])
    workflow_result = dict(bundle["workflow_result"])
    workflow_launch_status = dict(bundle.get("workflow_launch_status") or {})
    step_map = _step_lookup(workflow_result)
    launch_status = str(workflow_launch_status.get("status", "")).strip()
    has_workflow_result = bool(workflow_result)
    stages = [
        ("research_spec", "Research Spec", "PlannerAgent"),
        ("outer_loop", "Outer Loop", "outer_loop"),
        ("teacher_frozen_eval", "Frozen Eval", "teacher_frozen_eval"),
        ("TeacherSelectionAgent", "Selection", "TeacherSelectionAgent"),
        ("inner_loop_suite", "Inner Loop", "inner_loop_suite"),
        ("final_lesson_set", "Final Lesson Set", "ApprenticeAgent"),
        ("SignalScoringAgent", "Scoring", "SignalScoringAgent"),
    ]
    nodes: List[Dict[str, Any]] = []
    for node_id, label, key in stages:
        payload = dict(step_map.get(key, {}))
        status = str(payload.get("status", "")).strip() or "not_started"
        if node_id == "research_spec":
            research_spec_path = Path(str(contract.get("shared_context_root", ""))) / "studio_control" / "research_spec.json"
            status = "completed" if research_spec_path.exists() else "not_started"
            payload = {"artifact_json": str(research_spec_path) if research_spec_path.exists() else ""}
        elif not has_workflow_result and launch_status == "running":
            if node_id == "outer_loop":
                status = "running"
                payload = {"summary_path": str(_workflow_launch_status_path(contract))}
            elif status == "not_started":
                status = "pending"
        elif not has_workflow_result and launch_status == "failed":
            if node_id == "outer_loop":
                status = "failed"
                payload = {"summary_path": str(_workflow_launch_status_path(contract)), "error_message": workflow_launch_status.get("error_message", "")}
        if node_id == "final_lesson_set":
            inner_payload = dict(step_map.get("inner_loop_suite", {}).get("agent_summary", {}))
            lesson_path = str(inner_payload.get("final_lesson_artifact_json", "")).strip()
            status = "completed" if lesson_path else ("failed" if step_map.get("inner_loop_suite", {}).get("status") == "failed" else "not_started")
            payload = {"artifact_json": lesson_path, "suite_summary_json": inner_payload.get("suite_summary_json", "")}
            if not has_workflow_result and launch_status == "running":
                status = "pending"
        nodes.append(
            {
                "node_id": node_id,
                "label": label,
                "status": status,
                "payload": payload,
            }
        )
    return {
        "contract": contract,
        "workflow_status": str(workflow_result.get("status", "")).strip() or launch_status or "spec_only",
        "workflow_launch_status": workflow_launch_status,
        "executed_steps": int(workflow_result.get("executed_steps", 0) or 0),
        "manual_steps": int(workflow_result.get("manual_steps", 0) or 0),
        "failed_steps": int(workflow_result.get("failed_steps", 0) or 0),
        "nodes": nodes,
        "workflow_result_json": str(_workflow_result_path(contract)),
        "workflow_launch_status_json": str(_workflow_launch_status_path(contract)),
    }


def build_teacher_zoo_view(*, profile_chief: Any, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    bundle = load_run_bundle(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    contract = dict(bundle["contract"])
    workflow_result = dict(bundle["workflow_result"])
    imported_root = Path(contract["imported_asset_root"]).expanduser().resolve()
    workflow_root = Path(contract["workflow_root"]).expanduser().resolve()
    current_run_root = Path(contract["run_root"]).expanduser().resolve()
    step_map = _step_lookup(workflow_result)
    verification = dict(step_map.get("VerificationAgent", {}).get("payload", {}))
    frozen_eval = dict(step_map.get("teacher_frozen_eval", {}).get("agent_summary", {}))
    selection = dict(bundle["teacher_selection_summary"])

    imported_items = [
        {
            **item.__dict__,
            "teacher_state": "imported_frozen_teacher",
            "source_type": "imported_final_asset",
        }
        for item in profile_chief.teacher_zoo.list_teachers()
    ]
    candidate_items = [
        {
            **dict(row),
            "teacher_state": "candidate_teacher",
            "source_type": "current_workflow_asset",
        }
        for row in list(verification.get("candidate_teachers") or [])
    ]
    validated_items = [
        {
            **dict(row),
            "teacher_state": "validated_teacher",
            "source_type": "current_workflow_asset",
        }
        for row in list(verification.get("validated_teachers") or [])
    ]
    frozen_items = [
        {
            **dict(row),
            "teacher_state": "frozen_teacher",
            "source_type": "current_workflow_asset",
        }
        for row in list(frozen_eval.get("teachers") or [])
    ]
    selected_items = []
    for row in list(selection.get("teachers") or []):
        report_dir = str(row.get("report_dir", "")).strip()
        selected_items.append(
            {
                **dict(row),
                "teacher_state": "selected_teacher_for_inner_loop",
                "source_type": _source_type(report_dir, workflow_root=current_run_root, imported_root=imported_root),
            }
        )
    return {
        "contract": contract,
        "selection_resolution_source": str(selection.get("resolution_source", "")).strip(),
        "fallback_reason": str(selection.get("fallback_reason", "")).strip(),
        "imported_teachers": imported_items,
        "current_workflow_candidate_teachers": candidate_items,
        "current_workflow_validated_teachers": validated_items,
        "current_workflow_frozen_teachers": frozen_items,
        "selected_teachers_for_inner_loop": selected_items,
    }


def build_lesson_set_view(*, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    bundle = load_run_bundle(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    contract = dict(bundle["contract"])
    workflow_result = dict(bundle["workflow_result"])
    imported_root = Path(contract["imported_asset_root"]).expanduser().resolve()
    workflow_root = Path(contract["workflow_root"]).expanduser().resolve()
    current_run_root = Path(contract["run_root"]).expanduser().resolve()
    step_map = _step_lookup(workflow_result)
    apprentice_summary = dict(step_map.get("inner_loop_suite", {}).get("agent_summary", {}))
    lesson_path = str(apprentice_summary.get("final_lesson_artifact_json", "")).strip()
    lesson_payload = _safe_read_json(Path(lesson_path)) if lesson_path else {}
    teacher_scopes = list(lesson_payload.get("teacher_scopes") or [])
    lesson_rows = []
    for scope in teacher_scopes:
        lesson = dict(scope.get("scorefit_lesson_json") or {})
        lesson_rows.append(
            {
                "round_id": str(scope.get("round_id", "")).strip(),
                "source_round_id": str(scope.get("source_round_id", "")).strip(),
                "lesson_name": str(lesson.get("lesson_name", "")).strip(),
                "item_count": len(lesson.get("items", {})),
                "meta_rule_count": len(lesson.get("meta_rules", [])),
                "source_type": _source_type(lesson_path, workflow_root=current_run_root, imported_root=imported_root),
            }
        )
    return {
        "contract": contract,
        "final_lesson_state_json": lesson_path,
        "final_lesson_source": _source_type(lesson_path, workflow_root=current_run_root, imported_root=imported_root),
        "suite_summary_json": str(apprentice_summary.get("suite_summary_json", "")).strip(),
        "teacher_scope_count": len(teacher_scopes),
        "teacher_scopes": lesson_rows,
        "raw_teacher_scopes": teacher_scopes,
    }


def build_provenance_view(*, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    bundle = load_run_bundle(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    contract = dict(bundle["contract"])
    workflow_result = dict(bundle["workflow_result"])
    research_campaign = dict(bundle["research_campaign"])
    teacher_selection_summary = dict(bundle["teacher_selection_summary"])
    workflow_root = Path(contract["workflow_root"]).expanduser().resolve()
    files = []
    for static_path in [
        project_config_path(contract),
        dataset_manifest_path(contract),
        run_spec_path(contract),
        research_campaign_path(contract),
        _teacher_selection_summary_path(contract),
    ]:
        if Path(static_path).exists():
            files.append(str(static_path))
    for path in sorted(workflow_root.rglob("*.json")):
        files.append(str(path))
    scoring_root = Path(str(contract.get("scoring_root") or "")).expanduser()
    if scoring_root.exists():
        for path in sorted(scoring_root.rglob("*.json")):
            files.append(str(path))
    return {
        "contract": contract,
        "project_config_json": str(project_config_path(contract)),
        "dataset_manifest_json": str(dataset_manifest_path(contract)),
        "run_spec_json": str(run_spec_path(contract)),
        "research_campaign_json": str(research_campaign_path(contract)),
        "workflow_result_json": str(_workflow_result_path(contract)),
        "teacher_selection_summary_json": str(_teacher_selection_summary_path(contract)),
        "research_campaign": research_campaign,
        "teacher_selection_summary": teacher_selection_summary,
        "workflow_status": str(workflow_result.get("status", "")).strip() or "spec_only",
        "artifact_files": files,
    }
