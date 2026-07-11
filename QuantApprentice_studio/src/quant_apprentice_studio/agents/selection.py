from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..provenance import read_json, write_json
from .base import BaseAgent


class TeacherSelectionAgent(BaseAgent):
    def _normalize_text(self, value: str) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
        return " ".join(part for part in text.split() if part)

    def _load_research_spec(self, shared_context_root: str) -> Dict[str, Any]:
        path = self._studio_control_dir(shared_context_root) / "research_spec.json"
        if not path.exists():
            return {}
        return read_json(path)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or str(value).strip() == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None or str(value).strip() == "":
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _zscore_map(self, rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
        values = [self._safe_float(row.get(key, 0.0), 0.0) for row in rows]
        if not values:
            return {}
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        score_map: Dict[str, float] = {}
        for row, value in zip(rows, values):
            round_id = str(row.get("round_id", "")).strip()
            if not round_id:
                continue
            score_map[round_id] = 0.0 if std <= 1e-12 else (value - mean) / std
        return score_map

    def _family_priority_bonus(self, candidate: Mapping[str, Any], family_priority: Sequence[str]) -> float:
        family_text = self._normalize_text(candidate.get("research_family", ""))
        template_text = self._normalize_text(candidate.get("sample_template", ""))
        title_text = self._normalize_text(candidate.get("title", ""))
        bonus = 0.0
        for idx, raw in enumerate(family_priority):
            token = self._normalize_text(raw)
            if not token:
                continue
            weight = max(0.0, 0.18 - 0.03 * idx)
            if token in family_text or token in template_text or token in title_text:
                bonus += weight
        return bonus

    def _template_priority_bonus(self, candidate: Mapping[str, Any], preferred_templates: Sequence[str]) -> float:
        template_text = self._normalize_text(candidate.get("sample_template", ""))
        bonus = 0.0
        for idx, raw in enumerate(preferred_templates):
            token = self._normalize_text(raw)
            if not token:
                continue
            weight = max(0.0, 0.18 - 0.04 * idx)
            if token == template_text:
                bonus += weight
        return bonus

    def _attach_selection_scores(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        shared_context_root: str,
    ) -> List[Dict[str, Any]]:
        rows = [dict(row) for row in candidates]
        if not rows:
            return []
        research_spec = self._load_research_spec(shared_context_root)
        family_priority = list(research_spec.get("family_priority") or [])
        preferred_templates = list(research_spec.get("preferred_sample_templates") or [])
        for row in rows:
            positive_years = self._safe_int(row.get("positive_years", 0), 0)
            total_years = max(self._safe_int(row.get("total_years", 0), 0), 1)
            row["positive_rate"] = positive_years / total_years
            row["drawdown_quality"] = -abs(self._safe_float(row.get("nav_max_drawdown", 0.0), 0.0))
        z_nav_cagr = self._zscore_map(rows, "nav_cagr")
        z_alpha = self._zscore_map(rows, "mean_alpha")
        z_uplift = self._zscore_map(rows, "uplift_mean")
        z_positive = self._zscore_map(rows, "positive_rate")
        z_drawdown = self._zscore_map(rows, "drawdown_quality")
        for row in rows:
            round_id = str(row.get("round_id", "")).strip()
            explainability_bonus = 0.12 if bool(row.get("explainability_report_exists", False)) else 0.0
            family_bonus = self._family_priority_bonus(row, family_priority)
            template_bonus = self._template_priority_bonus(row, preferred_templates)
            base_score = (
                1.35 * z_nav_cagr.get(round_id, 0.0)
                + 1.10 * z_alpha.get(round_id, 0.0)
                + 0.80 * z_uplift.get(round_id, 0.0)
                + 0.60 * z_positive.get(round_id, 0.0)
                + 0.40 * z_drawdown.get(round_id, 0.0)
                + explainability_bonus
                + family_bonus
                + template_bonus
            )
            row["selection_base_score"] = float(base_score)
            row["selection_base_score_breakdown"] = {
                "z_nav_cagr": z_nav_cagr.get(round_id, 0.0),
                "z_mean_alpha": z_alpha.get(round_id, 0.0),
                "z_uplift_mean": z_uplift.get(round_id, 0.0),
                "z_positive_rate": z_positive.get(round_id, 0.0),
                "z_drawdown_quality": z_drawdown.get(round_id, 0.0),
                "explainability_bonus": explainability_bonus,
                "family_priority_bonus": family_bonus,
                "preferred_template_bonus": template_bonus,
            }
        return rows

    def _diversity_bonus(self, candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> float:
        if not selected:
            return 0.0
        cand_family = self._normalize_text(candidate.get("research_family", ""))
        cand_template = self._normalize_text(candidate.get("sample_template", ""))
        cand_model = self._normalize_text(candidate.get("model_family", ""))
        selected_families = {self._normalize_text(row.get("research_family", "")) for row in selected}
        selected_templates = {self._normalize_text(row.get("sample_template", "")) for row in selected}
        selected_models = {self._normalize_text(row.get("model_family", "")) for row in selected}
        bonus = 0.0
        if cand_family and cand_family not in selected_families:
            bonus += 0.35
        else:
            bonus -= 0.18
        if cand_template and cand_template not in selected_templates:
            bonus += 0.25
        else:
            bonus -= 0.12
        if cand_model and cand_model not in selected_models:
            bonus += 0.08
        if cand_family and cand_template:
            for row in selected:
                if (
                    cand_family == self._normalize_text(row.get("research_family", ""))
                    and cand_template == self._normalize_text(row.get("sample_template", ""))
                ):
                    bonus -= 0.20
                    break
        return bonus

    def _choose_diverse_subset(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        max_teachers: int,
    ) -> List[Dict[str, Any]]:
        pool = [dict(row) for row in candidates if str(row.get("round_id", "")).strip()]
        if len(pool) <= max_teachers:
            return sorted(pool, key=lambda row: float(row.get("selection_base_score", 0.0)), reverse=True)
        selected: List[Dict[str, Any]] = []
        remaining = list(pool)
        while remaining and len(selected) < max_teachers:
            best_idx = 0
            best_score = float("-inf")
            for idx, row in enumerate(remaining):
                composite = float(row.get("selection_base_score", 0.0)) + self._diversity_bonus(row, selected)
                if composite > best_score:
                    best_score = composite
                    best_idx = idx
            chosen = dict(remaining.pop(best_idx))
            chosen["selection_composite_score"] = best_score
            selected.append(chosen)
        return selected

    def _load_shared_zoo_map(self, shared_context_root: str) -> Dict[str, Dict[str, Any]]:
        zoo_index_path = Path(shared_context_root).expanduser().resolve() / "research_memory" / "indexes" / "teacher_zoo_index.json"
        if not zoo_index_path.exists():
            return {}
        payload = read_json(zoo_index_path)
        return {
            str(row.get("round_id", "")).strip(): dict(row)
            for row in list(payload.get("teachers") or [])
            if str(row.get("round_id", "")).strip()
        }

    def _load_selected_spec(self, report_dir: Path) -> Dict[str, Any]:
        spec_path = report_dir / "selected_spec.json"
        if not spec_path.exists():
            return {}
        return read_json(spec_path)

    def _build_frozen_eval_diverse_selection(
        self,
        *,
        shared_context_root: str,
        frozen_eval_summary: Mapping[str, Any],
        max_teachers: int,
    ) -> Dict[str, Any]:
        report_dir = Path(str(frozen_eval_summary.get("report_dir", "")).strip()).expanduser().resolve()
        selection_json = Path(str(frozen_eval_summary.get("selection_json", "")).strip()).expanduser().resolve()
        if not selection_json.exists():
            raise FileNotFoundError(f"Frozen eval selection_json not found: {selection_json}")
        selection_payload = read_json(selection_json)
        summary_csv = Path(str(frozen_eval_summary.get("summary_csv", "")).strip()).expanduser().resolve()
        rows: List[Dict[str, Any]] = []
        if summary_csv.exists():
            with summary_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        by_frozen_id = {str(row.get("frozen_round_id", "")).strip(): dict(row) for row in rows if str(row.get("frozen_round_id", "")).strip()}
        shared_zoo_map = self._load_shared_zoo_map(shared_context_root)
        candidates: List[Dict[str, Any]] = []
        for frozen_round_id in list(selection_payload.get("frozen_round_ids") or []):
            frozen_round_id = str(frozen_round_id).strip()
            if not frozen_round_id:
                continue
            row = dict(by_frozen_id.get(frozen_round_id, {}))
            source_round_id = str(row.get("source_round_id", "")).strip()
            round_report_dir = report_dir.parent / frozen_round_id
            spec = self._load_selected_spec(round_report_dir)
            source_zoo = dict(shared_zoo_map.get(source_round_id, {}))
            metrics = dict(source_zoo.get("metrics") or {})
            candidates.append(
                {
                    "round_id": frozen_round_id,
                    "frozen_round_id": frozen_round_id,
                    "source_round_id": source_round_id,
                    "title": str(spec.get("title", row.get("title", ""))).strip(),
                    "research_family": str(spec.get("research_family", row.get("research_family", ""))).strip(),
                    "sample_template": str(spec.get("sample_template", row.get("sample_template", ""))).strip(),
                    "model_family": str(row.get("model_family", spec.get("model_family", ""))).strip(),
                    "mean_alpha": self._safe_float(row.get("mean_alpha", metrics.get("mean_alpha", 0.0)), 0.0),
                    "uplift_mean": self._safe_float(row.get("mean_alpha", metrics.get("mean_alpha", 0.0)), 0.0),
                    "positive_years": self._safe_int(row.get("positive_years", source_zoo.get("positive_years", 0)), 0),
                    "total_years": self._safe_int(row.get("total_years", source_zoo.get("total_years", 0)), 0),
                    "nav_cagr": self._safe_float(row.get("nav_cagr", metrics.get("nav_cagr", 0.0)), 0.0),
                    "nav_total_return": self._safe_float(row.get("nav_total_return", metrics.get("nav_total_return", 0.0)), 0.0),
                    "nav_max_drawdown": self._safe_float(row.get("nav_max_drawdown", metrics.get("nav_max_drawdown", 0.0)), 0.0),
                    "accepted_as_teacher": bool(source_zoo.get("accepted_as_teacher", True)),
                    "status": str(source_zoo.get("status", "completed")).strip() or "completed",
                    "zoo_partition": str(source_zoo.get("zoo_partition", "frozen_eval")).strip() or "frozen_eval",
                    "explainability_report_exists": (
                        (round_report_dir / "FACTOR_ANALYSIS_REPORT.md").exists()
                        or (round_report_dir / "branch_rule_cards.json").exists()
                        or (round_report_dir / "factor_analysis_summary_v2.json").exists()
                    ),
                    "report_dir": str(round_report_dir),
                }
            )
        scored = self._attach_selection_scores(candidates, shared_context_root=shared_context_root)
        chosen = self._choose_diverse_subset(scored, max_teachers=max_teachers)
        if not chosen:
            raise RuntimeError("No frozen-eval teacher candidates were available for studio selection.")
        path = self._studio_control_dir(shared_context_root) / "selection_from_frozen_eval_diverse.json"
        payload = {
            "positive_round_ids": [str(row.get("source_round_id", "")).strip() for row in chosen if str(row.get("source_round_id", "")).strip()],
            "negative_round_ids": list(selection_payload.get("negative_round_ids") or []),
            "frozen_round_ids": [str(row.get("frozen_round_id", "")).strip() for row in chosen if str(row.get("frozen_round_id", "")).strip()],
            "selection_mode": "current_workflow_frozen_eval_diverse",
            "selection_generated_by": "TeacherSelectionAgent",
            "selection_note": (
                "Studio formal selection generated from current-workflow frozen-eval results. "
                "Candidates are ranked by real frozen metrics with a soft diversity bonus over research family/template."
            ),
            "selection_source_selection_json": str(selection_json),
            "selection_source_summary_csv": str(summary_csv) if summary_csv.exists() else "",
            "selection_candidate_count": len(scored),
            "selection_candidates": chosen,
            "train_end_year": selection_payload.get("train_end_year"),
            "test_start_year": selection_payload.get("test_start_year"),
            "test_end_year": selection_payload.get("test_end_year"),
            "fallback_reason": "",
        }
        write_json(path, payload)
        return {
            "selection_json": str(path),
            "selection_mode": "current_workflow_frozen_eval_diverse",
            "selected_rows": chosen,
            "candidate_count": len(scored),
            "source_selection_json": str(selection_json),
        }

    def _build_likely_teacher_diverse_selection(
        self,
        *,
        shared_context_root: str,
        likely_teachers: Sequence[Mapping[str, Any]],
        negative_round_ids: Sequence[str],
        max_teachers: int,
    ) -> Dict[str, Any]:
        candidates = self._attach_selection_scores(
            [
                {
                    "round_id": str(row.get("round_id", "")).strip(),
                    "source_round_id": str(row.get("round_id", "")).strip(),
                    "title": str(row.get("title", "")).strip(),
                    "research_family": str(row.get("research_family", "")).strip(),
                    "sample_template": str(row.get("sample_template", "")).strip(),
                    "model_family": "",
                    "mean_alpha": self._safe_float(row.get("mean_alpha", 0.0), 0.0),
                    "uplift_mean": self._safe_float(row.get("uplift_mean", row.get("mean_alpha", 0.0)), 0.0),
                    "positive_years": self._safe_int(row.get("positive_years", 0), 0),
                    "total_years": self._safe_int(row.get("total_years", 0), 0),
                    "nav_cagr": self._safe_float(row.get("nav_cagr", 0.0), 0.0),
                    "nav_total_return": self._safe_float(row.get("nav_total_return", 0.0), 0.0),
                    "nav_max_drawdown": self._safe_float(row.get("nav_max_drawdown", 0.0), 0.0),
                    "accepted_as_teacher": bool(row.get("accepted_as_teacher", False)),
                    "status": str(row.get("status", "")).strip(),
                    "zoo_partition": str(row.get("zoo_partition", "")).strip(),
                    "explainability_report_exists": bool(row.get("explainability_report_exists", False)),
                    "report_dir": "",
                }
                for row in likely_teachers
                if str(row.get("round_id", "")).strip()
            ],
            shared_context_root=shared_context_root,
        )
        chosen = self._choose_diverse_subset(candidates, max_teachers=max_teachers)
        path = self._studio_control_dir(shared_context_root) / "selection_from_outer_loop.json"
        payload = {
            "positive_round_ids": [str(row.get("round_id", "")).strip() for row in chosen if str(row.get("round_id", "")).strip()],
            "negative_round_ids": [str(x).strip() for x in negative_round_ids if str(x).strip()],
            "frozen_round_ids": [str(row.get("round_id", "")).strip() for row in chosen if str(row.get("round_id", "")).strip()],
            "selection_mode": "current_workflow_likely_teacher_diverse_transition",
            "selection_generated_by": "TeacherSelectionAgent",
            "selection_note": (
                "Studio-generated transitional selection from current validated teachers. "
                "Used only when no formal frozen-eval bundle is available."
            ),
            "selection_candidate_count": len(candidates),
            "selection_candidates": chosen,
            "fallback_reason": "No current-workflow frozen-eval teachers were available; using current validated teachers directly as a transition source.",
        }
        write_json(path, payload)
        return {
            "selection_json": str(path),
            "selection_mode": "current_workflow_likely_teacher_diverse_transition",
            "selected_rows": chosen,
            "candidate_count": len(candidates),
        }

    def resolve_workflow_selection_json(self, selection_json_hint: str = "") -> Dict:
        candidate = str(selection_json_hint).strip()
        source = "registry_default"
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"selection_json not found: {path}")
            source = "explicit_hint"
        else:
            path = Path(self.registry.load_runtime_catalog()["selection_json"]).expanduser().resolve()
        return {
            "selection_json": str(path),
            "resolution_source": source,
        }

    def _studio_control_dir(self, shared_context_root: str) -> Path:
        return Path(shared_context_root).expanduser().resolve() / "studio_control"

    def _load_negative_rounds(self, selection_json: Path, *, limit: int = 4) -> List[str]:
        if not selection_json.exists() or not selection_json.is_file():
            return []
        payload = read_json(selection_json)
        return [str(x).strip() for x in list(payload.get("negative_round_ids") or []) if str(x).strip()][:limit]

    def _write_selection_artifact(
        self,
        *,
        shared_context_root: str,
        payload: Dict[str, Any],
    ) -> str:
        path = self._studio_control_dir(shared_context_root) / "selected_teacher_for_inner_loop.json"
        write_json(path, payload)
        return str(path)

    def _build_transition_selection(
        self,
        *,
        shared_context_root: str,
        chosen_round_ids: List[str],
        negative_round_ids: List[str],
        selection_mode: str,
        selection_note: str,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        path = self._studio_control_dir(shared_context_root) / "selection_from_outer_loop.json"
        payload = {
            "positive_round_ids": list(chosen_round_ids),
            "negative_round_ids": list(negative_round_ids),
            "frozen_round_ids": list(chosen_round_ids),
            "selection_mode": selection_mode,
            "selection_generated_by": "TeacherSelectionAgent",
            "selection_note": selection_note,
            "fallback_reason": fallback_reason,
        }
        write_json(path, payload)
        return {
            "selection_json": str(path),
            "selection_mode": selection_mode,
        }

    def _build_demo_fallback_selection(
        self,
        *,
        shared_context_root: str,
        max_teachers: int,
    ) -> Dict[str, Any]:
        catalog = self.registry.load_runtime_catalog()
        round_ids = sorted(str(x).strip() for x in dict(catalog.get("teacher_reports", {})).keys() if str(x).strip())[:max_teachers]
        path = self._studio_control_dir(shared_context_root) / "demo_fallback_selection.json"
        payload = {
            "positive_round_ids": list(round_ids),
            "negative_round_ids": [],
            "frozen_round_ids": list(round_ids),
            "selection_mode": "demo_fallback_teacher",
            "selection_generated_by": "TeacherSelectionAgent",
            "selection_note": "Demo fallback selection generated from the imported runtime catalog because no stronger source was available.",
            "fallback_reason": "Neither current-workflow nor imported confirmed frozen selections were usable.",
        }
        write_json(path, payload)
        return {
            "selection_json": str(path),
            "selection_mode": "demo_fallback_teacher",
        }

    def materialize_workflow_selection(
        self,
        *,
        shared_context_root: str,
        verification_review: Mapping[str, Any],
        frozen_eval_summary: Mapping[str, Any],
        fallback_selection_json: str,
        fallback_resolution_source: str = "registry_default",
        max_teachers: int = 4,
        allow_imported_fallback: bool = True,
        allow_demo_fallback: bool = True,
    ) -> Dict[str, Any]:
        fallback_path = Path(fallback_selection_json).expanduser().resolve() if str(fallback_selection_json).strip() else Path()
        priority_chain = [
            "current workflow frozen-eval teacher",
            "current workflow likely teacher",
            "imported confirmed frozen teacher",
            "demo fallback teacher",
        ]
        frozen_eval_selection_json = Path(str(frozen_eval_summary.get("selection_json", "")).strip()).expanduser().resolve()
        if bool(frozen_eval_summary.get("frozen_eval_available", False)) and frozen_eval_selection_json.exists():
            generated = self._build_frozen_eval_diverse_selection(
                shared_context_root=shared_context_root,
                frozen_eval_summary=frozen_eval_summary,
                max_teachers=max_teachers,
            )
            artifact_json = self._write_selection_artifact(
                shared_context_root=shared_context_root,
                payload={
                    "teacher_priority_applied": priority_chain,
                    "resolution_source": "current_workflow_frozen_eval",
                    "resolved_selection_json": str(generated["selection_json"]),
                    "selection_mode": str(generated["selection_mode"]),
                    "fallback_reason": "",
                    "source_teacher_state": "frozen_teacher",
                    "studio_selected_state": "selected_teacher_for_inner_loop",
                    "selected_teacher_state": "frozen_teacher",
                    "frozen_eval_summary_name": str(frozen_eval_summary.get("summary_name", "")).strip(),
                    "frozen_eval_source_selection_json": str(frozen_eval_selection_json),
                    "selection_candidate_count": int(generated.get("candidate_count", 0) or 0),
                    "selected_round_ids": [
                        str(x).strip() for x in list(read_json(Path(str(generated["selection_json"])).expanduser().resolve()).get("frozen_round_ids") or []) if str(x).strip()
                    ],
                },
            )
            return {
                "selection_json": str(generated["selection_json"]),
                "resolution_source": "current_workflow_frozen_eval",
                "selection_mode": str(generated["selection_mode"]),
                "fallback_reason": "",
                "selection_artifact_json": artifact_json,
            }

        likely_teachers = list(verification_review.get("likely_teachers") or [])
        chosen_round_ids = [
            str(row.get("round_id", "")).strip()
            for row in likely_teachers
            if str(row.get("round_id", "")).strip()
        ][:max_teachers]
        if chosen_round_ids:
            negative_round_ids = self._load_negative_rounds(fallback_path, limit=max_teachers)
            generated = self._build_likely_teacher_diverse_selection(
                shared_context_root=shared_context_root,
                likely_teachers=likely_teachers,
                negative_round_ids=negative_round_ids,
                max_teachers=max_teachers,
            )
            artifact_json = self._write_selection_artifact(
                shared_context_root=shared_context_root,
                payload={
                    "teacher_priority_applied": priority_chain,
                    "resolution_source": "current_workflow_likely_teacher",
                    "resolved_selection_json": str(generated["selection_json"]),
                    "selection_mode": str(generated["selection_mode"]),
                    "fallback_reason": "No current-workflow frozen-eval teachers were available; using current validated teachers directly as a transition source.",
                    "source_teacher_state": "validated_teacher",
                    "studio_selected_state": "selected_teacher_for_inner_loop",
                    "selected_teacher_state": "validated_teacher",
                    "selection_candidate_count": int(generated.get("candidate_count", 0) or 0),
                    "selected_round_ids": [
                        str(x).strip()
                        for x in list(read_json(Path(str(generated["selection_json"])).expanduser().resolve()).get("frozen_round_ids") or [])
                        if str(x).strip()
                    ],
                },
            )
            return {
                "selection_json": str(generated["selection_json"]),
                "resolution_source": "current_workflow_likely_teacher",
                "selection_mode": str(generated["selection_mode"]),
                "fallback_reason": "No current-workflow frozen-eval teachers were available; using current validated teachers directly as a transition source.",
                "selection_artifact_json": artifact_json,
            }

        if allow_imported_fallback and fallback_path.exists() and fallback_path.is_file():
            fallback_reason = "No current-workflow frozen-eval or validated teachers were available; falling back to imported confirmed frozen teachers."
            artifact_json = self._write_selection_artifact(
                shared_context_root=shared_context_root,
                payload={
                    "teacher_priority_applied": priority_chain,
                    "resolution_source": "imported_confirmed_frozen_teacher",
                    "resolved_selection_json": str(fallback_path),
                    "selection_mode": "imported_confirmed_frozen_teacher",
                    "fallback_reason": fallback_reason,
                    "source_teacher_state": "frozen_teacher",
                    "studio_selected_state": "selected_teacher_for_inner_loop",
                    "selected_teacher_state": "selected_teacher_for_inner_loop",
                    "selected_round_ids": [
                        str(x).strip() for x in list(read_json(fallback_path).get("frozen_round_ids") or []) if str(x).strip()
                    ],
                },
            )
            return {
                "selection_json": str(fallback_path),
                "resolution_source": "imported_confirmed_frozen_teacher",
                "selection_mode": "imported_confirmed_frozen_teacher",
                "fallback_reason": fallback_reason,
                "selection_artifact_json": artifact_json,
            }

        if not allow_demo_fallback:
            artifact_json = self._write_selection_artifact(
                shared_context_root=shared_context_root,
                payload={
                    "teacher_priority_applied": priority_chain,
                    "resolution_source": "no_usable_teacher_source",
                    "resolved_selection_json": "",
                    "selection_mode": "none",
                    "fallback_reason": (
                        "No current-workflow teachers were usable and fallback policy disallowed both imported and demo fallbacks."
                    ),
                    "source_teacher_state": "none",
                    "studio_selected_state": "selected_teacher_for_inner_loop",
                    "selected_teacher_state": "none",
                    "selected_round_ids": [],
                },
            )
            return {
                "selection_json": "",
                "resolution_source": "no_usable_teacher_source",
                "selection_mode": "none",
                "fallback_reason": "No current-workflow teachers were usable and fallback policy disallowed both imported and demo fallbacks.",
                "selection_artifact_json": artifact_json,
            }

        demo = self._build_demo_fallback_selection(
            shared_context_root=shared_context_root,
            max_teachers=max_teachers,
        )
        demo_path = Path(str(demo["selection_json"])).expanduser().resolve()
        artifact_json = self._write_selection_artifact(
            shared_context_root=shared_context_root,
            payload={
                "teacher_priority_applied": priority_chain,
                "resolution_source": "demo_fallback_teacher",
                "resolved_selection_json": str(demo_path),
                "selection_mode": "demo_fallback_teacher",
                "fallback_reason": "Neither current-workflow nor imported confirmed frozen teachers were usable.",
                "source_teacher_state": "demo_fallback_teacher",
                "studio_selected_state": "selected_teacher_for_inner_loop",
                "selected_teacher_state": "selected_teacher_for_inner_loop",
                "selected_round_ids": [
                    str(x).strip() for x in list(read_json(demo_path).get("frozen_round_ids") or []) if str(x).strip()
                ],
            },
        )
        return {
            "selection_json": str(demo_path),
            "resolution_source": "demo_fallback_teacher",
            "selection_mode": "demo_fallback_teacher",
            "fallback_reason": "Neither current-workflow nor imported confirmed frozen teachers were usable.",
            "selection_artifact_json": artifact_json,
        }

    def summarize_selection(self, selection_json: str, *, shared_context_root: str = "") -> Dict:
        if not str(selection_json).strip():
            return {
                "selection_json": "",
                "selection_mode": "none",
                "fallback_reason": "No selection_json was resolved.",
                "frozen_count": 0,
                "negative_count": 0,
                "family_count": 0,
                "teachers": [],
            }
        path = Path(selection_json).expanduser().resolve()
        payload = read_json(path)
        frozen = list(payload.get("frozen_round_ids") or [])
        positive = list(payload.get("positive_round_ids") or [])
        negative = list(payload.get("negative_round_ids") or [])
        catalog = self.registry.load_runtime_catalog()
        report_map = dict(catalog.get("teacher_reports", {}))
        imported_profile_root = self.registry.imported_profile_root
        shared_report_root = (
            Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
            if str(shared_context_root).strip()
            else None
        )
        shared_zoo_index_path = (
            Path(shared_context_root).expanduser().resolve() / "research_memory" / "indexes" / "teacher_zoo_index.json"
            if str(shared_context_root).strip()
            else None
        )
        shared_zoo_payload = (
            read_json(shared_zoo_index_path)
            if shared_zoo_index_path is not None and shared_zoo_index_path.exists()
            else {"teachers": []}
        )
        shared_zoo_map = {
            str(row.get("round_id", "")).strip(): dict(row)
            for row in list(shared_zoo_payload.get("teachers") or [])
            if str(row.get("round_id", "")).strip()
        }

        teachers: List[Dict] = []
        families: List[str] = []
        for idx, round_id in enumerate(frozen):
            report_info = dict(report_map.get(str(round_id), {}))
            selected_spec = {}
            nav_summary = {}
            report_dir = None
            if report_info.get("selected_spec_json"):
                spec_path = Path(report_info["selected_spec_json"]).expanduser().resolve()
                selected_spec = read_json(spec_path)
                report_dir = spec_path.parent
                nav_path = Path(str(report_info.get("nav_summary_json", ""))).expanduser().resolve()
                if nav_path.exists():
                    nav_summary = read_json(nav_path)
            elif shared_report_root is not None:
                shared_spec_path = shared_report_root / str(round_id).strip() / "selected_spec.json"
                if shared_spec_path.exists():
                    selected_spec = read_json(shared_spec_path)
                    report_dir = shared_spec_path.parent
                    nav_path = report_dir / "nav_summary.json"
                    if nav_path.exists():
                        nav_summary = read_json(nav_path)
            family = str(selected_spec.get("research_family", "")).strip()
            template = str(selected_spec.get("sample_template", "")).strip()
            title = str(selected_spec.get("title", "")).strip()
            source_round_id = str(positive[idx]).strip() if idx < len(positive) else ""
            uplift_mean = 0.0
            positive_years = 0
            total_years = 0
            if report_dir is not None:
                summary_csv = report_dir / "walkforward_yearly_summary.csv"
                if summary_csv.exists():
                    try:
                        rows: List[Dict[str, str]] = []
                        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
                            rows = [dict(row) for row in csv.DictReader(handle)]
                        values: List[float] = []
                        for row in rows:
                            raw = str(row.get("q5_alpha_vs_baseline", "")).strip()
                            if not raw:
                                continue
                            try:
                                values.append(float(raw))
                            except ValueError:
                                continue
                        if values:
                            uplift_mean = float(sum(values) / len(values))
                            positive_years = int(sum(1 for value in values if value > 0))
                            total_years = int(len(values))
                    except Exception:
                        uplift_mean = 0.0
            shared_teacher = dict(shared_zoo_map.get(str(round_id).strip(), {}))
            metrics = dict(shared_teacher.get("metrics") or {})
            explainability_exists = False
            if report_dir is not None:
                explainability_exists = (
                    (report_dir / "FACTOR_ANALYSIS_REPORT.md").exists()
                    or (report_dir / "branch_rule_cards.json").exists()
                    or (report_dir / "factor_analysis_summary_v2.json").exists()
                )
            if family:
                families.append(family)
            nav_positive_years = int(nav_summary.get("positive_years", metrics.get("nav_positive_years", 0)) or 0)
            nav_total_years = int(nav_summary.get("total_years", metrics.get("nav_total_years", 0)) or 0)
            mean_alpha = float(
                shared_teacher.get(
                    "mean_alpha",
                    metrics.get("mean_alpha", uplift_mean),
                )
                or uplift_mean
            )
            if positive_years <= 0 and nav_positive_years > 0:
                positive_years = nav_positive_years
            if total_years <= 0 and nav_total_years > 0:
                total_years = nav_total_years
            teachers.append(
                {
                    "round_id": str(round_id),
                    "source_round_id": source_round_id,
                    "title": title,
                    "research_family": family,
                    "sample_template": template,
                    "teacher_state": (
                        "frozen_teacher" if bool(selected_spec.get("frozen_eval", False)) or "_frozen_" in str(round_id)
                        else "validated_teacher" if str(round_id).strip() in shared_zoo_map
                        else "selected_teacher_for_inner_loop"
                    ),
                    "accepted_as_teacher": bool(shared_teacher.get("accepted_as_teacher", False)) if shared_teacher else bool(selected_spec.get("frozen_eval", False)),
                    "status": str(shared_teacher.get("status", "")).strip() or "completed",
                    "zoo_partition": str(shared_teacher.get("zoo_partition", "")).strip() or str(nav_summary.get("partition", "")).strip(),
                    "mean_alpha": mean_alpha,
                    "uplift_mean": uplift_mean,
                    "nav_cagr": float(nav_summary.get("cagr", metrics.get("nav_cagr", 0.0)) or 0.0),
                    "nav_final": float(nav_summary.get("final_nav", metrics.get("nav_final_nav", 0.0)) or 0.0),
                    "nav_total_return": float(nav_summary.get("total_return", metrics.get("nav_total_return", 0.0)) or 0.0),
                    "nav_max_drawdown": float(nav_summary.get("max_drawdown", metrics.get("nav_max_drawdown", 0.0)) or 0.0),
                    "positive_years": int(shared_teacher.get("positive_years", positive_years) or positive_years),
                    "total_years": int(shared_teacher.get("total_years", total_years) or total_years),
                    "nav_positive_years": nav_positive_years,
                    "nav_total_years": nav_total_years,
                    "explainability_report_exists": explainability_exists,
                    "report_dir": str(report_dir) if report_dir is not None else "",
                }
            )
        return {
            "selection_json": str(path),
            "selection_mode": str(payload.get("selection_mode", "")).strip() or "standard",
            "fallback_reason": str(payload.get("fallback_reason", "")).strip(),
            "frozen_count": len(frozen),
            "negative_count": len(negative),
            "family_count": len({x for x in families if x}),
            "teachers": teachers,
        }
