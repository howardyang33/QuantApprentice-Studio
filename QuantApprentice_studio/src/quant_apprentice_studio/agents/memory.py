from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..paths import original_quant_root
from ..provenance import read_json
from .base import BaseAgent


class MemoryAgent(BaseAgent):
    def profile_summary(self) -> Dict:
        catalog = self.registry.load_runtime_catalog()
        return {
            "profile_id": catalog["profile_id"],
            "backbone": catalog["backbone"],
            "selection_json": catalog["selection_json"],
            "digest_file": catalog["digest_file"],
            "default_alignment_seed_alias": str(catalog.get("defaults", {}).get("alignment_seed_alias", "")).strip(),
            "default_market_run_alias": str(catalog.get("defaults", {}).get("market_run_alias", "")).strip(),
            "teacher_report_count": len(catalog.get("teacher_reports", {})),
            "lesson_run_count": len(catalog.get("lesson_runs", {})),
            "market_run_count": len(catalog.get("market_runs", {})),
        }

    def resolve_selection_json(self, selection_json: str = "") -> str:
        candidate = str(selection_json).strip()
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"selection_json not found: {path}")
            return str(path)
        return str(Path(self.registry.load_runtime_catalog()["selection_json"]).resolve())

    def resolve_default_lesson_alias(self, lesson_alias: str = "") -> str:
        if str(lesson_alias).strip():
            return str(lesson_alias).strip()
        catalog = self.registry.load_runtime_catalog()
        alias = str(catalog.get("defaults", {}).get("alignment_seed_alias", "")).strip()
        if alias:
            return alias
        lesson_runs = dict(catalog.get("lesson_runs", {}))
        if lesson_runs:
            return sorted(lesson_runs.keys())[0]
        raise ValueError("No lesson alias available in runtime catalog.")

    def resolve_default_api_model(self) -> str:
        backbone = str(self.registry.load_runtime_catalog().get("backbone", "")).strip().lower()
        if backbone:
            return backbone
        return "gpt-oss-20b"

    def resolve_imported_quant_path(self, relative_to_original_quant: str) -> str:
        source = (original_quant_root() / relative_to_original_quant).resolve()
        return str(self.registry.imported_path_for_source(source))

    def load_selection_payload(self, selection_json: str = "") -> Dict:
        return read_json(Path(self.resolve_selection_json(selection_json)))

    def resolve_imported_teacher_report_root(self) -> str:
        catalog = self.registry.load_runtime_catalog()
        roots = sorted({str(Path(v["report_dir"]).expanduser().resolve().parent) for v in catalog.get("teacher_reports", {}).values()})
        if not roots:
            raise ValueError("No imported teacher report roots found in runtime catalog.")
        return roots[0]

    def resolve_imported_teacher_artifact_root(self) -> str:
        catalog = self.registry.load_runtime_catalog()
        roots = sorted({str(Path(v["artifact_dir"]).expanduser().resolve().parent) for v in catalog.get("teacher_artifacts", {}).values()})
        if not roots:
            raise ValueError("No imported teacher artifact roots found in runtime catalog.")
        return roots[0]

    def resolve_shared_master_cache_path(self) -> str:
        candidates = [
            original_quant_root() / "research_memory" / "artifacts" / "teacher_loop" / "_shared_cache" / "master_feature_label_20260605_v2.joblib",
            original_quant_root() / "research_memory" / "artifacts" / "teacher_loop" / "_shared_cache" / "master_feature_label_day_klines_20190101_20260601_y2020_2026.joblib",
            original_quant_root() / "research_memory_exp2012" / "artifacts" / "teacher_loop" / "_shared_cache" / "master_feature_label_day_klines_2012_20120101_20260605_y2013_2022.joblib",
            original_quant_root() / "research_memory_exp2012_dsv4" / "artifacts" / "teacher_loop" / "_shared_cache" / "master_feature_label_day_klines_2012_20120101_20260605_y2013_2026_dsv4_frozen.joblib",
        ]
        for path in candidates:
            resolved = Path(path).expanduser().resolve()
            if resolved.exists():
                return str(resolved)
        raise FileNotFoundError("No shared master feature-label cache was found under original quant roots.")

    def resolve_hs300_index_path(self) -> str:
        candidate = (original_quant_root() / "index_klines" / "000300.csv").resolve()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"HS300 index file not found: {candidate}")

    def resolve_original_stock_data_dir(self) -> str:
        candidate = (original_quant_root() / "day_klines").resolve()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Stock data dir not found: {candidate}")

    def resolve_original_teacher_report_root(self) -> str:
        profile = self.registry.load_profile()
        roots = sorted({str(Path(path).expanduser().resolve().parent) for path in profile.get("teacher_report_dirs", {}).values()})
        if not roots:
            raise ValueError("No original teacher report roots found in profile config.")
        return roots[0]

    def resolve_original_teacher_artifact_root(self) -> str:
        profile = self.registry.load_profile()
        roots = sorted({str(Path(path).expanduser().resolve().parent) for path in profile.get("teacher_artifact_dirs", {}).values()})
        if not roots:
            raise ValueError("No original teacher artifact roots found in profile config.")
        return roots[0]

    def resolve_workflow_shared_context_root(self, path_hint: str = "") -> str:
        candidate = str(path_hint).strip()
        if not candidate:
            return ""
        path = Path(candidate).expanduser().resolve()
        for parent in [path, *path.parents]:
            if parent.name != "clean_context":
                continue
            if parent.parent.name and parent.parent.parent.name == "workflows":
                return str(parent)
        return ""
