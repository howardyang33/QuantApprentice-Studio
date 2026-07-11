"""Minimal file-backed research memory store for QuantApprentice Pilot 1."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "1.0"
MANIFEST_FILE_NAME = "kb_manifest.jsonl"

ITEM_TYPE_TO_DIR: Dict[str, str] = {
    "hypothesis": "hypotheses",
    "experiment": "experiments",
    "teacher_model": "teacher_models",
    "factor_card": "factor_cards",
    "regime_card": "regime_cards",
    "failure_case": "failure_cases",
    "success_case": "success_cases",
    "research_lesson": "research_lessons",
    "trader_lesson": "trader_lessons",
}

ITEM_TYPE_TO_ID_FIELD: Dict[str, str] = {
    "hypothesis": "hypothesis_id",
    "experiment": "experiment_id",
    "teacher_model": "model_id",
    "factor_card": "factor_id",
    "regime_card": "regime_id",
    "failure_case": "case_id",
    "success_case": "case_id",
    "research_lesson": "lesson_id",
    "trader_lesson": "lesson_id",
}

ITEM_TYPE_TO_ID_PREFIX: Dict[str, str] = {
    "hypothesis": "hyp",
    "experiment": "exp",
    "teacher_model": "tm",
    "factor_card": "fac",
    "regime_card": "reg",
    "failure_case": "fail",
    "success_case": "succ",
    "research_lesson": "rless",
    "trader_lesson": "tless",
}

TYPE_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "hypothesis": ["hypothesis_statement", "rationale"],
    "experiment": ["hypothesis_ids", "objective", "design", "conclusion"],
    "teacher_model": ["model_family", "intended_use", "training_data_refs"],
    "factor_card": ["factor_name", "factor_definition"],
    "regime_card": ["regime_name", "regime_definition"],
    "failure_case": ["failure_summary", "root_cause"],
    "success_case": ["success_summary", "why_it_worked"],
    "research_lesson": ["lesson_summary", "recommended_action"],
    "trader_lesson": ["lesson_summary", "recommended_action"],
}

GENERAL_STATUS_VALUES = [
    "draft",
    "active",
    "completed",
    "rejected",
    "archived",
    "dummy",
]

EXPERIMENT_EXECUTION_STATUS_VALUES = [
    "planned",
    "not_run",
    "running",
    "completed",
    "failed",
    "cancelled",
]

README_TEMPLATE = """# QuantApprentice Research Memory

This directory stores the minimal file-backed research memory created in Pilot 1.

Key rules:
- One knowledge item per JSON file.
- One append-only manifest record per item in `kb_manifest.jsonl`.
- Schema registry lives in `indexes/schema_registry.json`.
- Artifact files should be placed under `artifacts/` and referenced by path from item files.
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_timestamp() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def id_timestamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str, max_length: int = 40) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "item"
    return slug[:max_length]


def dedupe_keep_order(values: Optional[Iterable[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values or []:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_schema_registry() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_fields": [
            "memory_id",
            "item_type",
            "item_id",
            "title",
            "summary",
            "status",
            "created_at",
            "updated_at",
            "tags",
            "linked_ids",
            "storage_path",
            "is_dummy",
            "bootstrap_example",
            "schema_version",
        ],
        "id_rules": {
            "memory_id": "mem_<item_type>_<YYYYMMDDTHHMMSSZ>_<slug>_<suffix4>",
            "hypothesis_id": "hyp_<YYYYMMDDTHHMMSSZ>_<slug>_<suffix4>",
            "experiment_id": "exp_<YYYYMMDDTHHMMSSZ>_<slug>_<suffix4>",
            "model_id": "tm_<YYYYMMDDTHHMMSSZ>_<slug>_<suffix4>",
            "factor_id": "fac_<YYYYMMDDTHHMMSSZ>_<slug>_<suffix4>",
        },
        "timestamp_rule": "UTC RFC3339 without fractional seconds, for example 2026-06-03T08:15:30Z",
        "general_status_values": GENERAL_STATUS_VALUES,
        "experiment_execution_status_values": EXPERIMENT_EXECUTION_STATUS_VALUES,
        "item_type_directories": ITEM_TYPE_TO_DIR,
        "required_fields": TYPE_REQUIRED_FIELDS,
    }


@dataclass
class CreatedItem:
    item: Dict[str, Any]
    manifest_entry: Dict[str, Any]
    path: Path


class MemoryStore:
    """Simple JSON file storage with a JSONL manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE_NAME

    @property
    def schema_registry_path(self) -> Path:
        return self.root / "indexes" / "schema_registry.json"

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"

    def init_storage(self) -> Dict[str, Any]:
        created_directories: List[str] = []
        created_files: List[str] = []

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            created_directories.append(".")

        for directory in ITEM_TYPE_TO_DIR.values():
            path = self.root / directory
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_directories.append(path.relative_to(self.root).as_posix())

        for directory in ("indexes", "artifacts"):
            path = self.root / directory
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_directories.append(path.relative_to(self.root).as_posix())

        if not self.manifest_path.exists():
            self.manifest_path.write_text("", encoding="utf-8")
            created_files.append(self.manifest_path.relative_to(self.root).as_posix())

        if not self.schema_registry_path.exists():
            schema_registry = json.dumps(build_schema_registry(), indent=2, ensure_ascii=False) + "\n"
            self.schema_registry_path.write_text(schema_registry, encoding="utf-8")
            created_files.append(self.schema_registry_path.relative_to(self.root).as_posix())

        if not self.readme_path.exists():
            self.readme_path.write_text(README_TEMPLATE, encoding="utf-8")
            created_files.append(self.readme_path.relative_to(self.root).as_posix())

        return {
            "memory_dir": str(self.root),
            "created_directories": created_directories,
            "created_files": created_files,
            "manifest_path": str(self.manifest_path),
            "schema_registry_path": str(self.schema_registry_path),
        }

    def create_item(
        self,
        item_type: str,
        title: str,
        summary: str,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        linked_ids: Optional[Iterable[str]] = None,
        source_label: str = "cli",
        source_type: str = "manual_cli",
        created_by: str = "quant_toolkit.memory.cli",
        is_dummy: bool = False,
        bootstrap_example: bool = False,
        item_id: Optional[str] = None,
    ) -> CreatedItem:
        self.init_storage()

        normalized_type = item_type.strip().lower()
        if normalized_type not in ITEM_TYPE_TO_DIR:
            raise ValueError(f"Unsupported item_type: {item_type}")

        payload = dict(payload or {})
        self._check_reserved_fields(payload)

        id_field = ITEM_TYPE_TO_ID_FIELD[normalized_type]
        payload_item_id = payload.pop(id_field, None)
        final_item_id = item_id or payload_item_id or self._generate_item_id(normalized_type, title)
        final_memory_id = self._generate_memory_id(normalized_type, title)
        while self._manifest_contains(final_memory_id) or (self.root / ITEM_TYPE_TO_DIR[normalized_type] / f"{final_memory_id}.json").exists():
            final_memory_id = self._generate_memory_id(normalized_type, title)

        now = utc_timestamp()
        relative_path = f"{ITEM_TYPE_TO_DIR[normalized_type]}/{final_memory_id}.json"
        item: Dict[str, Any] = {
            "memory_id": final_memory_id,
            "item_type": normalized_type,
            "schema_version": SCHEMA_VERSION,
            "title": title.strip(),
            "summary": summary.strip(),
            "status": status.strip(),
            "created_at": now,
            "updated_at": now,
            "tags": dedupe_keep_order(tags),
            "linked_ids": dedupe_keep_order(linked_ids),
            "storage_path": relative_path,
            "is_dummy": bool(is_dummy),
            "bootstrap_example": bool(bootstrap_example),
            "provenance": {
                "source_label": source_label,
                "source_type": source_type,
                "created_by": created_by,
            },
            id_field: final_item_id,
        }
        item.update(payload)
        self._apply_type_defaults(item)
        self._validate_item(item)

        output_path = self.root / relative_path
        output_path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest_entry = self._build_manifest_entry(item)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

        return CreatedItem(item=item, manifest_entry=manifest_entry, path=output_path)

    def get_item(self, memory_id: Optional[str] = None, path: Optional[str | Path] = None) -> Dict[str, Any]:
        if bool(memory_id) == bool(path):
            raise ValueError("Provide exactly one of memory_id or path")

        if memory_id:
            manifest_entry = self._find_manifest_entry(memory_id)
            if manifest_entry is None:
                raise FileNotFoundError(f"memory_id not found: {memory_id}")
            return self.get_item(path=manifest_entry["storage_path"])

        resolved_path = Path(path) if isinstance(path, Path) else Path(str(path))
        if not resolved_path.is_absolute():
            resolved_path = (self.root / resolved_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"item file not found: {resolved_path}")
        return json.loads(resolved_path.read_text(encoding="utf-8"))

    def list_items(self, item_type: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        entries = self._load_manifest_entries()
        if item_type:
            normalized_type = item_type.strip().lower()
            entries = [entry for entry in entries if entry["item_type"] == normalized_type]

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for entry in reversed(entries):
            memory_id = entry["memory_id"]
            if memory_id in seen:
                continue
            seen.add(memory_id)
            deduped.append(entry)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped

    def _load_manifest_entries(self) -> List[Dict[str, Any]]:
        self.init_storage()
        entries: List[Dict[str, Any]] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
        return entries

    def _manifest_contains(self, memory_id: str) -> bool:
        return self._find_manifest_entry(memory_id) is not None

    def _find_manifest_entry(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for entry in reversed(self._load_manifest_entries()):
            if entry["memory_id"] == memory_id:
                return entry
        return None

    def _generate_item_id(self, item_type: str, title: str) -> str:
        prefix = ITEM_TYPE_TO_ID_PREFIX[item_type]
        return f"{prefix}_{id_timestamp()}_{slugify(title)}_{secrets.token_hex(2)}"

    def _generate_memory_id(self, item_type: str, title: str) -> str:
        return f"mem_{item_type}_{id_timestamp()}_{slugify(title)}_{secrets.token_hex(2)}"

    def _check_reserved_fields(self, payload: Dict[str, Any]) -> None:
        reserved = {
            "memory_id",
            "item_type",
            "schema_version",
            "created_at",
            "updated_at",
            "storage_path",
            "provenance",
        }
        overlap = sorted(reserved.intersection(payload))
        if overlap:
            raise ValueError(f"Payload contains reserved fields: {overlap}")

    def _apply_type_defaults(self, item: Dict[str, Any]) -> None:
        item_type = item["item_type"]
        if item_type == "hypothesis":
            item.setdefault("assumptions", [])
            item.setdefault("expected_signal", "")
            item.setdefault("linked_experiment_ids", [])
        elif item_type == "experiment":
            item.setdefault("execution_status", "planned")
            item.setdefault("result_summary", "")
            item.setdefault("artifact_refs", [])
        elif item_type == "teacher_model":
            item.setdefault("training_status", "planned")
            item.setdefault("artifact_refs", [])
        elif item_type in {"research_lesson", "trader_lesson"}:
            item.setdefault("applies_to", [])
        elif item_type in {"failure_case", "success_case"}:
            item.setdefault("artifact_refs", [])

    def _validate_item(self, item: Dict[str, Any]) -> None:
        required_common = [
            "memory_id",
            "item_type",
            "schema_version",
            "title",
            "summary",
            "status",
            "created_at",
            "updated_at",
            "tags",
            "linked_ids",
            "storage_path",
        ]
        missing_common = [field for field in required_common if field not in item]
        if missing_common:
            raise ValueError(f"Missing common fields: {missing_common}")

        if item["status"] not in GENERAL_STATUS_VALUES:
            raise ValueError(f"Unsupported status: {item['status']}")

        item_type = item["item_type"]
        id_field = ITEM_TYPE_TO_ID_FIELD[item_type]
        if id_field not in item or not item[id_field]:
            raise ValueError(f"Missing type-specific id field: {id_field}")

        required_fields = TYPE_REQUIRED_FIELDS[item_type]
        missing_required = [field for field in required_fields if field not in item or item[field] in (None, "", [])]
        if missing_required:
            raise ValueError(f"Missing required {item_type} fields: {missing_required}")

        if item_type == "experiment":
            if item["execution_status"] not in EXPERIMENT_EXECUTION_STATUS_VALUES:
                raise ValueError(f"Unsupported execution_status: {item['execution_status']}")
            if not isinstance(item["hypothesis_ids"], list):
                raise ValueError("experiment.hypothesis_ids must be a list")

    def _build_manifest_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        id_field = ITEM_TYPE_TO_ID_FIELD[item["item_type"]]
        return {
            "memory_id": item["memory_id"],
            "item_type": item["item_type"],
            "item_id": item[id_field],
            "title": item["title"],
            "summary": item["summary"],
            "status": item["status"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "tags": item["tags"],
            "linked_ids": item["linked_ids"],
            "storage_path": item["storage_path"],
            "is_dummy": item["is_dummy"],
            "bootstrap_example": item["bootstrap_example"],
            "schema_version": item["schema_version"],
        }
