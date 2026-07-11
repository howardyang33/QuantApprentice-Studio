from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .paths import import_root, studio_root
from .provenance import read_json, write_json


def _slug(value: str, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip()).strip("-").lower()
    return text or fallback


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def projects_root() -> Path:
    return studio_root() / "runs" / "projects"


def dataset_root(project_id: str, dataset_id: str) -> Path:
    return (
        projects_root()
        / _slug(project_id, "default-project")
        / "datasets"
        / _slug(dataset_id, "default-dataset")
    )


def run_root(project_id: str, dataset_id: str, run_id: str) -> Path:
    return dataset_root(project_id, dataset_id) / "runs" / _slug(run_id, f"run-{_timestamp()}")


def build_run_contract(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
) -> Dict[str, Any]:
    root = run_root(project_id, dataset_id, run_id)
    dataset_root_path = dataset_root(project_id, dataset_id)
    dataset_raw_root = dataset_root_path / "raw"
    dataset_stock_klines_root = dataset_raw_root / "stock_klines"
    dataset_index_klines_root = dataset_raw_root / "index_klines"
    dataset_cache_root = dataset_root_path / "cache"
    dataset_jobs_root = dataset_root_path / "jobs"
    dataset_upload_root = dataset_root_path / "uploads"
    asset_root = root / "assets"
    teacher_zoo_root = asset_root / "teacher_zoo"
    lesson_root = asset_root / "lessons"
    scoring_root = asset_root / "scoring"
    workflow_root = root / "workflow"
    shared_context_root = workflow_root / "clean_context"
    provenance_root = root / "provenance"
    imported_root = import_root(profile_id).resolve()
    current_asset_roots = [
        dataset_root_path,
        dataset_raw_root,
        dataset_stock_klines_root,
        dataset_index_klines_root,
        dataset_cache_root,
        dataset_jobs_root,
        dataset_upload_root,
        asset_root,
        teacher_zoo_root,
        lesson_root,
        scoring_root,
        workflow_root,
        provenance_root,
    ]
    return {
        "profile_id": str(profile_id).strip(),
        "project_id": _slug(project_id, "default-project"),
        "dataset_id": _slug(dataset_id, "default-dataset"),
        "run_id": _slug(run_id, f"run-{_timestamp()}"),
        "dataset_root": str(dataset_root_path),
        "dataset_raw_root": str(dataset_raw_root),
        "dataset_stock_klines_root": str(dataset_stock_klines_root),
        "dataset_index_klines_root": str(dataset_index_klines_root),
        "dataset_cache_root": str(dataset_cache_root),
        "dataset_jobs_root": str(dataset_jobs_root),
        "dataset_upload_root": str(dataset_upload_root),
        "run_root": str(root),
        "asset_root": str(asset_root),
        "teacher_zoo_root": str(teacher_zoo_root),
        "lesson_root": str(lesson_root),
        "scoring_root": str(scoring_root),
        "workflow_root": str(workflow_root),
        "shared_context_root": str(shared_context_root),
        "provenance_root": str(provenance_root),
        "imported_asset_root": str(imported_root),
        "allow_imported_fallback": bool(allow_imported_fallback),
        "allow_demo_fallback": bool(allow_demo_fallback),
        "data_isolation": {
            "isolated_from_imported_assets": all(imported_root not in Path(path).resolve().parents for path in map(str, current_asset_roots)),
            "run_root_exists": root.exists(),
            "dataset_root_exists": dataset_root_path.exists(),
            "imported_asset_root_exists": imported_root.exists(),
        },
    }


def ensure_contract_dirs(contract: Dict[str, Any]) -> None:
    for key in [
        "dataset_root",
        "dataset_raw_root",
        "dataset_stock_klines_root",
        "dataset_index_klines_root",
        "dataset_cache_root",
        "dataset_jobs_root",
        "dataset_upload_root",
        "run_root",
        "asset_root",
        "teacher_zoo_root",
        "lesson_root",
        "scoring_root",
        "workflow_root",
        "shared_context_root",
        "provenance_root",
    ]:
        Path(contract[key]).mkdir(parents=True, exist_ok=True)


def run_spec_path(contract: Dict[str, Any]) -> Path:
    return Path(contract["run_root"]) / "run_spec.json"


def project_config_path(contract: Dict[str, Any]) -> Path:
    return Path(contract["run_root"]) / "project_config.json"


def dataset_manifest_path(contract: Dict[str, Any]) -> Path:
    return Path(contract["dataset_root"]) / "dataset_manifest.json"


def dataset_job_path(contract: Dict[str, Any], job_id: str) -> Path:
    return Path(contract["dataset_jobs_root"]) / f"{_slug(job_id, 'dataset-job')}.json"


def research_campaign_path(contract: Dict[str, Any]) -> Path:
    return Path(contract["run_root"]) / "research_campaign.json"


def save_project_config(contract: Dict[str, Any], payload: Dict[str, Any]) -> str:
    ensure_contract_dirs(contract)
    path = project_config_path(contract)
    write_json(path, payload)
    return str(path)


def save_dataset_manifest(contract: Dict[str, Any], payload: Dict[str, Any]) -> str:
    ensure_contract_dirs(contract)
    path = dataset_manifest_path(contract)
    write_json(path, payload)
    return str(path)


def save_run_spec(contract: Dict[str, Any], payload: Dict[str, Any]) -> str:
    ensure_contract_dirs(contract)
    path = run_spec_path(contract)
    write_json(path, payload)
    return str(path)


def save_research_campaign(contract: Dict[str, Any], payload: Dict[str, Any]) -> str:
    ensure_contract_dirs(contract)
    path = research_campaign_path(contract)
    write_json(path, payload)
    return str(path)


def load_run_spec(project_id: str, dataset_id: str, run_id: str) -> Dict[str, Any]:
    path = run_spec_path(
        build_run_contract(
            profile_id="gpt_oss_20b_final",
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=True,
            allow_demo_fallback=True,
        )
    )
    if not path.exists():
        raise FileNotFoundError(f"run_spec not found: {path}")
    return read_json(path)


def list_run_specs() -> List[Dict[str, Any]]:
    root = projects_root()
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("run_spec.json")):
        try:
            payload = read_json(path)
            workflow_result = path.parent / "workflow" / "workflow_result.json"
            rows.append(
                {
                    "project_id": str(payload.get("project_id", "")).strip(),
                    "dataset_id": str(payload.get("dataset_id", "")).strip(),
                    "run_id": str(payload.get("run_id", "")).strip(),
                    "profile_id": str(payload.get("profile_id", "")).strip(),
                    "research_goal": str(payload.get("research_goal", "")).strip(),
                    "mode": str(payload.get("mode", "")).strip(),
                    "status": "completed" if workflow_result.exists() else "spec_only",
                    "run_root": str(path.parent),
                    "run_spec_json": str(path),
                    "workflow_result_json": str(workflow_result) if workflow_result.exists() else "",
                    "allow_imported_fallback": bool(payload.get("allow_imported_fallback", True)),
                    "allow_demo_fallback": bool(payload.get("allow_demo_fallback", True)),
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda row: (row["project_id"], row["dataset_id"], row["run_id"]))
    return rows
