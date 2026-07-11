from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping


def _safe_catalog(chief: Any) -> Dict[str, Any]:
    try:
        return dict(chief.registry.load_runtime_catalog())
    except Exception:
        return {}


def _safe_teachers(chief: Any) -> List[Dict[str, Any]]:
    try:
        return [dict(item.__dict__) for item in chief.teacher_zoo.list_teachers()]
    except Exception:
        return []


def _safe_lessons(chief: Any) -> List[Dict[str, Any]]:
    try:
        return [dict(item.__dict__) for item in chief.lesson.list_lesson_runs()]
    except Exception:
        return []


def _safe_markets(chief: Any) -> List[Dict[str, Any]]:
    try:
        return [dict(item.__dict__) for item in chief.backtest.list_market_runs()]
    except Exception:
        return []


def _safe_current_library(*, project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any] | None:
    if not (project_id and dataset_id and run_id):
        return None
    try:
        from .console_views import load_run_bundle

        bundle = load_run_bundle(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    except Exception:
        return None
    workflow = dict(bundle.get("workflow_result") or {})
    selection = dict(bundle.get("teacher_selection_summary") or {})
    campaign = dict(bundle.get("research_campaign") or {})
    contract = dict(bundle.get("contract") or {})
    steps = list(workflow.get("steps") or [])
    selected = list(selection.get("teachers") or [])
    final_lesson_path = ""
    for step in steps:
        if step.get("step_id") == "inner_loop_suite":
            summary = dict(step.get("agent_summary") or {})
            final_lesson_path = str(summary.get("final_lesson_artifact_json") or "").strip()
            break
    if not selected and not final_lesson_path:
        return None
    display_name = str(
        selection.get("teacher_library_display_name_zh")
        or campaign.get("teacher_library_display_name_zh")
        or contract.get("teacher_library_display_name_zh")
        or ""
    ).strip()
    if not display_name:
        display_name = f"{run_id} 老师库"
    return {
        "teacher_library_id": f"user_{project_id}_{dataset_id}_{run_id}",
        "display_name_zh": display_name,
        "market": "用户数据",
        "source_type": "user_trained",
        "status": "ready" if final_lesson_path else "training_or_partial",
        "teacher_count": len(selected),
        "lesson_set_status": "ready" if final_lesson_path else "not_ready",
        "default_for_market": False,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "final_lesson_state_json": final_lesson_path,
        "description_zh": "由用户上传数据通过当前 QuantApprentice workflow 训练得到。",
        "teacher_items": selected,
    }


def build_teacher_library_registry(
    *,
    chief: Any,
    profile_id: str,
    project_id: str = "",
    dataset_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    """Product-level registry of teacher libraries.

    This wrapper intentionally keeps legacy artifact field names internal. The UI can present
    existing paper-run assets as a built-in frozen teacher library rather than as "imported" files.
    """

    catalog = _safe_catalog(chief)
    defaults = dict(catalog.get("defaults") or {})
    teachers = _safe_teachers(chief)
    lessons = _safe_lessons(chief)
    markets = _safe_markets(chief)
    default_lesson_alias = str(defaults.get("alignment_seed_alias") or (lessons[0].get("alias") if lessons else "") or "").strip()
    default_market_alias = str(defaults.get("market_run_alias") or (markets[0].get("alias") if markets else "") or "").strip()
    final_lesson_path = ""
    for lesson in lessons:
        if lesson.get("alias") == default_lesson_alias:
            final_lesson_path = str(lesson.get("final_lesson_state_json") or "").strip()
            break

    built_in = {
        "teacher_library_id": "paper_ashare_gptoss20b_v7",
        "display_name_zh": "A股趋势回调与突破教师库",
        "market": "A股",
        "source_type": "built_in_baseline",
        "status": "ready" if teachers and final_lesson_path else "not_ready",
        "teacher_count": len(teachers),
        "lesson_set_status": "ready" if final_lesson_path else "not_ready",
        "default_for_market": True,
        "profile_id": profile_id,
        "lesson_alias": default_lesson_alias,
        "market_run_alias": default_market_alias,
        "final_lesson_state_json": final_lesson_path,
        "artifact_root": str(Path(str(catalog.get("import_root") or "")).expanduser()) if catalog.get("import_root") else "",
        "description_zh": "基于 QuantApprentice 论文配置训练得到，覆盖 A 股趋势突破、均线回调、动量回调与量能-KDJ 回调等短线技术形态。",
        "teacher_items": teachers,
    }

    libraries: List[Dict[str, Any]] = [built_in]
    current = _safe_current_library(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    if current:
        libraries.append(current)
    active_id = current["teacher_library_id"] if current else built_in["teacher_library_id"]

    return {
        "profile_id": profile_id,
        "active_teacher_library_id": active_id,
        "default_teacher_library_id": built_in["teacher_library_id"],
        "items": libraries,
        "ui_terms": {
            "teacher_library": "老师库",
            "built_in_baseline": "系统内置老师库",
            "user_trained": "用户训练老师库",
            "final_lesson_set": "最终经验规则集",
        },
    }


def teacher_library_source_label_zh(source: str) -> str:
    value = str(source or "").strip()
    if value in {"built_in_baseline", "imported_final_asset", "imported_frozen_teacher", "imported_frozen_teacher_zoo"}:
        return "A股趋势回调与突破教师库"
    if value in {"user_trained", "current_workflow_asset", "explicit_final_lesson_state_json"}:
        return "本次训练老师库"
    if value in {"lesson_alias"}:
        return "系统已有经验规则集"
    if value in {"demo_asset", "imported_demo"}:
        return "开发测试资产"
    if value in {"unavailable", "unresolved", ""}:
        return "未生成"
    return value
