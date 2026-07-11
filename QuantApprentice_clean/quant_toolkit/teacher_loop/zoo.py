"""Teacher-zoo indexing and partition inference for autonomous loop control."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ..memory.store import MemoryStore
from ..pilot2.teacher_utils import REVERSAL_FEATURE_COLUMNS
from .registry import ALL_REGISTERED_FEATURES, DERIVED_FEATURE_NAMES, ensure_feature_registry

PILOT2_2_FEATURE_COLUMNS = [
    "ret_1_clip",
    "ret_3_clip",
    "body_pct",
    "amplitude",
    "lower_shadow",
    "upper_shadow",
    "volume_zscore_20",
    "amt_zscore_20",
    "gap_pct",
    "volatility_5",
    "volatility_20",
    "vol_ratio_5_20",
    "amt_log",
    "volume_log",
    "close_log",
    "oversold_depth",
    "J",
    "pos_20",
    "dist_to_20d_low",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _teacher_loop_report_root(project_root: Path) -> Path:
    default = project_root / "reports" / "teacher_loop"
    return Path(os.environ.get("TEACHER_LOOP_REPORT_ROOT", str(default))).expanduser().resolve()


def infer_round_id(item: Dict[str, Any]) -> str | None:
    if item.get("round_id"):
        return str(item["round_id"])
    title = str(item.get("title", ""))
    if title.startswith("round_"):
        return title.split()[0]
    return None


def infer_nav_metrics(item: Dict[str, Any], *, project_root: Path) -> Dict[str, Any]:
    fields = [
        "nav_final_nav",
        "nav_total_return",
        "nav_cagr",
        "nav_max_drawdown",
        "nav_hs300_total_return",
        "nav_excess_total_return",
        "nav_positive_years",
        "nav_total_years",
    ]
    if any(item.get(field) is not None for field in fields):
        return {field: item.get(field) for field in fields}

    round_id = infer_round_id(item)
    if not round_id:
        return {field: None for field in fields}

    summary_path = _teacher_loop_report_root(project_root) / round_id / "nav_summary.json"
    if not summary_path.exists():
        return {field: None for field in fields}

    payload = _load_json(summary_path)
    return {
        "nav_final_nav": payload.get("final_nav"),
        "nav_total_return": payload.get("total_return"),
        "nav_cagr": payload.get("cagr"),
        "nav_max_drawdown": payload.get("max_drawdown"),
        "nav_hs300_total_return": payload.get("hs300_total_return"),
        "nav_excess_total_return": payload.get("excess_total_return"),
        "nav_positive_years": payload.get("positive_years"),
        "nav_total_years": payload.get("total_years"),
    }


def infer_zoo_partition(item: Dict[str, Any]) -> str:
    if item.get("zoo_partition"):
        return str(item["zoo_partition"])
    if item.get("mean_incremental_alpha") is not None and item.get("accepted_as_teacher") is True:
        return "main"
    if item.get("accepted_as_teacher") is True:
        return "try"
    if item.get("status") == "rejected" and item.get("artifact_refs"):
        return "try"
    return "rejected"


def infer_sample_template(item: Dict[str, Any]) -> str:
    if item.get("sample_template"):
        return str(item["sample_template"])
    tags = set(item.get("tags", []))
    title = str(item.get("title", "")).lower()
    if "within_gate" in tags or "within_gate" in title:
        return "hard_threshold_reversal_gate"
    if "walkforward" in tags or "q5_selector" in tags:
        return "weak_state_reversal_pool"
    return "weak_state_reversal_pool"


def infer_research_family(item: Dict[str, Any]) -> str:
    if item.get("research_family"):
        return str(item["research_family"])
    tags = set(item.get("tags", []))
    title = str(item.get("title", "")).lower()
    if "breakout" in tags or "breakout" in title:
        return "breakout"
    if "defensive" in tags or "defensive" in title:
        return "defensive"
    return "reversal"


def infer_feature_columns(item: Dict[str, Any]) -> List[str]:
    if item.get("feature_columns"):
        return [str(x) for x in item["feature_columns"]]
    tags = set(item.get("tags", []))
    title = str(item.get("title", "")).lower()
    if "within_gate" in tags or "within_gate" in title:
        return list(PILOT2_2_FEATURE_COLUMNS)
    return list(REVERSAL_FEATURE_COLUMNS)


def build_teacher_zoo_index(memory_dir: str | Path) -> Path:
    root = Path(memory_dir).expanduser().resolve()
    project_root = root.parent
    target = root / "indexes" / "teacher_zoo_index.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root)
    entries = store.list_items(item_type="teacher_model", limit=None)
    teachers: List[Dict[str, Any]] = []

    for entry in entries:
        item = store.get_item(path=entry["storage_path"])
        feature_columns = infer_feature_columns(item)
        nav_metrics = infer_nav_metrics(item, project_root=project_root)
        teacher = {
            "memory_id": item["memory_id"],
            "model_id": item["model_id"],
            "round_id": item.get("round_id"),
            "round_index": item.get("round_index"),
            "title": item["title"],
            "summary": item["summary"],
            "status": item["status"],
            "zoo_partition": infer_zoo_partition(item),
            "research_family": infer_research_family(item),
            "sample_template": infer_sample_template(item),
            "model_family": item.get("model_family", "unknown"),
            "accepted_as_teacher": bool(item.get("accepted_as_teacher", False)),
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "feature_overlap_basis": sorted(set(feature_columns).intersection(ALL_REGISTERED_FEATURES)),
            "artifact_refs": item.get("artifact_refs", []),
            "linked_ids": item.get("linked_ids", []),
            "tags": item.get("tags", []),
            "metrics": {
                "q5_win_rate": item.get("q5_win_rate"),
                "mean_alpha": item.get("mean_alpha"),
                "median_alpha": item.get("median_alpha"),
                "mean_incremental_alpha": item.get("mean_incremental_alpha"),
                "median_incremental_alpha": item.get("median_incremental_alpha"),
                **nav_metrics,
            },
        }
        teachers.append(teacher)

    partition_counts: Dict[str, int] = {}
    for teacher in teachers:
        partition = teacher["zoo_partition"]
        partition_counts[partition] = partition_counts.get(partition, 0) + 1

    payload = {
        "schema_version": "1.0",
        "teacher_count": len(teachers),
        "partition_counts": partition_counts,
        "teachers": teachers,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def ensure_teacher_loop_indexes(memory_dir: str | Path) -> Dict[str, str]:
    feature_registry_path = ensure_feature_registry(memory_dir)
    zoo_index_path = build_teacher_zoo_index(memory_dir)
    loop_manifest_path = Path(memory_dir).expanduser().resolve() / "indexes" / "teacher_loop_manifest.jsonl"
    loop_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not loop_manifest_path.exists():
        loop_manifest_path.write_text("", encoding="utf-8")
    loop_state_path = Path(memory_dir).expanduser().resolve() / "indexes" / "teacher_loop_state.json"
    if not loop_state_path.exists():
        default_target_rounds = int(os.environ.get("TEACHER_LOOP_TARGET_ROUNDS", "10"))
        loop_state_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "loop_name": "teacher_construction_loop",
                    "target_rounds": default_target_rounds,
                    "rounds_launched": 0,
                    "rounds_completed": 0,
                    "active_round_id": None,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "feature_registry_path": str(feature_registry_path),
        "teacher_zoo_index_path": str(zoo_index_path),
        "teacher_loop_manifest_path": str(loop_manifest_path),
        "teacher_loop_state_path": str(loop_state_path),
    }
