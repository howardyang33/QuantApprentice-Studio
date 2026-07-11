from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Iterable, List

from ..paths import original_quant_root, studio_root
from ..provenance import write_json
from ..registry import StudioRegistry
from ..schemas import AssetRecord, AssetSpec, ImportManifest


LESSON_INCLUDE = [
    "selected_final_lesson_best_composite_zscore_sum.json",
    "warmup_scoped_lessons.json",
]

MARKET_INCLUDE = [
    "summary.json",
    "llm_signal_scores.json",
    "llm_daily_nav.json",
    "teacher_daily_nav.json",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_profile_config(profile_id: str = "gpt_oss_20b_final") -> Dict[str, Any]:
    registry = StudioRegistry(profile_id)
    return registry.load_profile()


def _glob_match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def build_asset_specs(profile_id: str = "gpt_oss_20b_final") -> List[AssetSpec]:
    profile = load_profile_config(profile_id)
    specs: List[AssetSpec] = [
        AssetSpec(
            asset_id="teacher_selection_dir",
            kind="teacher_selection",
            source_path=profile["selection_dir"],
            recursive=True,
            include_globs=["*.json", "*.csv", "*.md"],
            notes="Frozen GPT-OSS best4 teacher selection bundle.",
        ),
        AssetSpec(
            asset_id="digest_simplified",
            kind="digest",
            source_path=profile["digest_file"],
            notes="Simplified digest used by the final GPT-OSS line.",
        ),
    ]
    for round_id, path in profile["teacher_report_dirs"].items():
        specs.append(
            AssetSpec(
                asset_id=f"teacher_report::{round_id}",
                kind="teacher_report",
                source_path=path,
                recursive=True,
                include_globs=["*.json", "*.csv", "*.md", "*.png"],
                notes=f"Frozen report bundle for {round_id}.",
            )
        )
    for round_id, path in profile["teacher_artifact_dirs"].items():
        specs.append(
            AssetSpec(
                asset_id=f"teacher_artifact::{round_id}",
                kind="teacher_artifact",
                source_path=path,
                recursive=True,
                include_globs=["*.joblib", "*.csv", "*.csv.gz"],
                notes=f"Model and threshold artifacts for {round_id}.",
            )
        )
    for alias, payload in profile["lesson_runs"].items():
        specs.append(
            AssetSpec(
                asset_id=f"lesson_run::{alias}",
                kind="lesson_run",
                source_path=payload["path"],
                recursive=True,
                include_globs=LESSON_INCLUDE,
                notes=f"Final lesson state and warmup state for {alias}.",
            )
        )
    for alias, payload in profile["market_runs"].items():
        specs.append(
            AssetSpec(
                asset_id=f"market_run::{alias}",
                kind="market_run",
                source_path=payload["path"],
                recursive=True,
                include_globs=MARKET_INCLUDE,
                notes=f"Archived market replay bundle for {alias}.",
            )
        )
    for alias, path in profile["paper_tables"].items():
        specs.append(
            AssetSpec(
                asset_id=f"paper_table::{alias}",
                kind="paper_table",
                source_path=path,
                notes=f"Derived summary table: {alias}.",
            )
        )
    return specs


def _iter_selected_files(source: Path, include_globs: List[str], exclude_globs: List[str]) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        rel_name = str(rel)
        if exclude_globs and _glob_match_any(rel_name, exclude_globs):
            continue
        if include_globs and not _glob_match_any(rel.name, include_globs) and not _glob_match_any(rel_name, include_globs):
            continue
        yield path


def _copy_asset(spec: AssetSpec, registry: StudioRegistry, *, overwrite: bool) -> AssetRecord:
    source = Path(spec.source_path).expanduser().resolve()
    if not source.exists():
        return AssetRecord(
            asset_id=spec.asset_id,
            kind=spec.kind,
            source_path=str(source),
            target_path="",
            status="missing_source",
            copied_files=0,
            copied_bytes=0,
            recursive=spec.recursive,
            notes=spec.notes,
        )
    quant_root = original_quant_root()
    target_root = registry.imported_profile_root / "original_quant"
    if source == quant_root:
        raise ValueError("Importing the whole original quant root is not allowed.")
    target = target_root / source.relative_to(quant_root)
    copied_files = 0
    copied_bytes = 0
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            copy2(source, target)
        copied_files = 1
        copied_bytes = target.stat().st_size
    else:
        for file_path in _iter_selected_files(source, spec.include_globs, spec.exclude_globs):
            rel = file_path.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not dest.exists():
                copy2(file_path, dest)
            copied_files += 1
            copied_bytes += dest.stat().st_size
    return AssetRecord(
        asset_id=spec.asset_id,
        kind=spec.kind,
        source_path=str(source),
        target_path=str(target),
        status="copied",
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        recursive=spec.recursive,
        notes=spec.notes,
    )


def _build_runtime_catalog(profile: Dict[str, Any], registry: StudioRegistry) -> Dict[str, Any]:
    selection_dir = registry.imported_path_for_source(profile["selection_dir"])
    digest_file = registry.imported_path_for_source(profile["digest_file"])
    teacher_reports = {
        round_id: {
            "report_dir": str(registry.imported_path_for_source(path)),
            "selected_spec_json": str(registry.imported_path_for_source(path) / "selected_spec.json"),
            "nav_summary_json": str(registry.imported_path_for_source(path) / "nav_summary.json"),
            "factor_analysis_summary_json": str(registry.imported_path_for_source(path) / "factor_analysis_summary.json"),
            "branch_rule_cards_json": str(registry.imported_path_for_source(path) / "branch_rule_cards.json"),
        }
        for round_id, path in profile["teacher_report_dirs"].items()
    }
    teacher_artifacts = {
        round_id: {
            "artifact_dir": str(registry.imported_path_for_source(path)),
            "joblib_files": [str(p) for p in sorted(registry.imported_path_for_source(path).glob("models/*.joblib"))],
            "threshold_csv_files": [str(p) for p in sorted(registry.imported_path_for_source(path).glob("**/*threshold*.csv"))],
        }
        for round_id, path in profile["teacher_artifact_dirs"].items()
    }
    lesson_runs = {}
    for alias, payload in profile["lesson_runs"].items():
        imported_dir = registry.imported_path_for_source(payload["path"])
        lesson_runs[alias] = {
            "seed_label": payload["seed_label"],
            "imported_dir": str(imported_dir),
            "final_lesson_state_json": str(imported_dir / "selected_final_lesson_best_composite_zscore_sum.json"),
            "warmup_state_json": str(imported_dir / "warmup_scoped_lessons.json"),
        }
    market_runs = {}
    for alias, payload in profile["market_runs"].items():
        imported_dir = registry.imported_path_for_source(payload["path"])
        market_runs[alias] = {
            "window": payload["window"],
            "imported_dir": str(imported_dir),
            "summary_json": str(imported_dir / "summary.json"),
            "llm_signal_scores_json": str(imported_dir / "llm_signal_scores.json"),
            "llm_daily_nav_json": str(imported_dir / "llm_daily_nav.json"),
            "teacher_daily_nav_json": str(imported_dir / "teacher_daily_nav.json"),
        }
    paper_tables = {
        alias: str(registry.imported_path_for_source(path))
        for alias, path in profile["paper_tables"].items()
    }
    return {
        "profile_id": profile["profile_id"],
        "backbone": profile["backbone"],
        "generated_at": _utc_now(),
        "selection_dir": str(selection_dir),
        "selection_json": str(selection_dir / "selection.json"),
        "digest_file": str(digest_file),
        "teacher_reports": teacher_reports,
        "teacher_artifacts": teacher_artifacts,
        "lesson_runs": lesson_runs,
        "market_runs": market_runs,
        "paper_tables": paper_tables,
        "defaults": profile.get("defaults", {}),
    }


def bootstrap_profile(profile_id: str = "gpt_oss_20b_final", *, overwrite: bool = False) -> Dict[str, Any]:
    registry = StudioRegistry(profile_id)
    profile = registry.load_profile()
    specs = build_asset_specs(profile_id)
    records = [_copy_asset(spec, registry, overwrite=overwrite) for spec in specs]
    manifest = ImportManifest(
        profile_id=profile_id,
        imported_at=_utc_now(),
        studio_root=str(studio_root()),
        original_quant_root=str(original_quant_root()),
        import_root=str(registry.imported_profile_root),
        records=records,
    )
    runtime_catalog = _build_runtime_catalog(profile, registry)
    write_json(registry.manifest_path, manifest)
    write_json(registry.runtime_catalog_path, runtime_catalog)
    return {
        "manifest_path": str(registry.manifest_path),
        "runtime_catalog_path": str(registry.runtime_catalog_path),
        "copied_asset_count": len(records),
        "copied_file_count": sum(item.copied_files for item in records),
        "copied_bytes": sum(item.copied_bytes for item in records),
    }
