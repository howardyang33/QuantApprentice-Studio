from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from ..provenance import read_json
from .base import BaseAgent


class TeacherAcceptanceAgent(BaseAgent):
    def _extract_available_test_years(self, report_dir: Path) -> List[int]:
        summary_csv = report_dir / "walkforward_yearly_summary.csv"
        if not summary_csv.exists():
            return []
        years: List[int] = []
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = str(row.get("test_year", "")).strip()
                if not raw:
                    continue
                try:
                    years.append(int(raw))
                except ValueError:
                    continue
        return sorted({year for year in years})

    def _infer_frozen_eval_window(
        self,
        *,
        verification_review: Mapping[str, Any],
        shared_context_root: str,
        default_train_end_year: int,
        default_test_start_year: int,
        default_test_end_year: int,
    ) -> Tuple[int, int, int]:
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        teacher_year_sets: List[List[int]] = []
        for row in list(verification_review.get("likely_teachers") or []):
            round_id = str(row.get("round_id", "")).strip()
            if not round_id:
                continue
            years = self._extract_available_test_years(report_root / round_id)
            if years:
                teacher_year_sets.append(years)

        if not teacher_year_sets:
            return default_train_end_year, default_test_start_year, default_test_end_year

        shared_years = set(teacher_year_sets[0])
        for years in teacher_year_sets[1:]:
            shared_years &= set(years)
        ordered = sorted(shared_years) if shared_years else sorted(set().union(*[set(x) for x in teacher_year_sets]))
        if not ordered:
            return default_train_end_year, default_test_start_year, default_test_end_year
        if len(ordered) >= 3:
            return ordered[-3], ordered[-2], ordered[-1]
        if len(ordered) == 2:
            return ordered[0], ordered[1], ordered[1]
        only_year = ordered[0]
        return only_year, only_year, only_year

    def build_teacher_frozen_eval_args(
        self,
        *,
        verification_review: Mapping[str, Any],
        shared_context_root: str,
        fallback_selection_json: str,
        summary_name: str,
        train_end_year: int = 2022,
        test_start_year: int = 2023,
        test_end_year: int = 2026,
        top_k: int = 8,
        negative_k: int = 4,
    ) -> List[str]:
        likely_teachers = list(verification_review.get("likely_teachers") or [])
        teacher_rounds = [
            str(row.get("round_id", "")).strip()
            for row in likely_teachers
            if str(row.get("round_id", "")).strip()
        ][:top_k]

        train_end_year, test_start_year, test_end_year = self._infer_frozen_eval_window(
            verification_review=verification_review,
            shared_context_root=shared_context_root,
            default_train_end_year=train_end_year,
            default_test_start_year=test_start_year,
            default_test_end_year=test_end_year,
        )

        negative_rounds: List[str] = []
        fallback_path = Path(fallback_selection_json).expanduser().resolve()
        shared_report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        if fallback_path.exists():
            payload = read_json(fallback_path)
            negative_rounds = [
                str(x).strip()
                for x in list(payload.get("negative_round_ids") or [])
                if str(x).strip() and (shared_report_root / str(x).strip() / "selected_spec.json").exists()
            ][:negative_k]

        args: List[str] = [
            "--summary-name",
            summary_name,
            "--train-end-year",
            str(int(train_end_year)),
            "--test-start-year",
            str(int(test_start_year)),
            "--test-end-year",
            str(int(test_end_year)),
        ]
        if teacher_rounds:
            args.extend(["--teacher-rounds", *teacher_rounds])
        if negative_rounds:
            args.extend(["--negative-rounds", *negative_rounds])
        if not teacher_rounds:
            args.extend(["--top-k", str(int(top_k))])
        if not negative_rounds:
            args.extend(["--negative-k", str(int(negative_k))])
        return args

    def summarize_frozen_eval_outputs(
        self,
        *,
        shared_context_root: str,
        summary_name: str,
    ) -> Dict[str, Any]:
        report_dir = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop" / summary_name
        selection_json = report_dir / "selection.json"
        summary_csv = report_dir / "frozen_post2022_summary.csv"
        if not selection_json.exists():
            return {
                "summary_name": summary_name,
                "report_dir": str(report_dir),
                "selection_json": str(selection_json),
                "frozen_eval_available": False,
                "frozen_teacher_count": 0,
                "teachers": [],
            }

        selection_payload = read_json(selection_json)
        summary_rows: List[Dict[str, Any]] = []
        if summary_csv.exists():
            with summary_csv.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = [dict(row) for row in csv.DictReader(handle)]
        teachers: List[Dict[str, Any]] = []
        by_frozen_id = {str(row.get("frozen_round_id", "")).strip(): row for row in summary_rows}
        for frozen_round_id in list(selection_payload.get("frozen_round_ids") or []):
            frozen_round_id = str(frozen_round_id).strip()
            row = dict(by_frozen_id.get(frozen_round_id, {}))
            round_report_dir = report_dir.parent / frozen_round_id
            teachers.append(
                {
                    "frozen_round_id": frozen_round_id,
                    "source_round_id": str(row.get("source_round_id", "")).strip(),
                    "mean_alpha": float(row.get("mean_alpha", 0.0) or 0.0),
                    "uplift_mean": float(row.get("mean_alpha", 0.0) or 0.0),
                    "positive_years": int(row.get("positive_years", 0) or 0),
                    "nav_cagr": float(row.get("nav_cagr", 0.0) or 0.0),
                    "nav_max_drawdown": float(row.get("nav_max_drawdown", 0.0) or 0.0),
                    "explainability_report_exists": (round_report_dir / "FACTOR_ANALYSIS_REPORT.md").exists(),
                    "branch_rule_cards_exists": (round_report_dir / "branch_rule_cards.json").exists(),
                    "selected_spec_json": str(round_report_dir / "selected_spec.json"),
                }
            )
        return {
            "summary_name": summary_name,
            "report_dir": str(report_dir),
            "selection_json": str(selection_json),
            "selection_payload": selection_payload,
            "summary_csv": str(summary_csv),
            "frozen_eval_available": True,
            "frozen_teacher_count": len(teachers),
            "teachers": teachers,
        }

    def build_acceptance_review(
        self,
        *,
        selection_resolution: Mapping[str, Any],
        teacher_selection_summary: Mapping[str, Any],
        verification_review: Mapping[str, Any],
        frozen_eval_summary: Mapping[str, Any],
    ) -> Dict[str, Any]:
        resolved_selection_json = str(selection_resolution.get("selection_json", "")).strip()
        resolution_source = str(selection_resolution.get("resolution_source", "")).strip()
        fallback_reason = str(selection_resolution.get("fallback_reason", "")).strip()
        teacher_count = int(teacher_selection_summary.get("frozen_count", 0) or 0)
        family_count = int(teacher_selection_summary.get("family_count", 0) or 0)
        verified_teacher_count = int(verification_review.get("verified_teacher_count", 0) or 0)
        preferred_template_hit_rate = float(verification_review.get("preferred_template_hit_rate", 0.0) or 0.0)
        frozen_eval_available = bool(frozen_eval_summary.get("frozen_eval_available", False))
        frozen_teacher_count = int(frozen_eval_summary.get("frozen_teacher_count", 0) or 0)
        likely_teacher_count = int(verification_review.get("likely_teacher_count", 0) or 0)

        if resolution_source == "current_workflow_frozen_eval":
            acceptance_status = "accepted_from_current_workflow_frozen_eval"
        elif resolution_source == "current_workflow_likely_teacher":
            acceptance_status = "accepted_from_current_workflow_likely_teacher_transition"
        elif resolution_source == "imported_confirmed_frozen_teacher":
            acceptance_status = "accepted_from_imported_confirmed_frozen_teacher"
        elif resolution_source == "demo_fallback_teacher":
            acceptance_status = "accepted_from_demo_fallback_teacher"
        else:
            acceptance_status = (
                "accepted_from_existing_selection_hint"
                if resolution_source == "explicit_hint"
                else "accepted_from_registry_default_selection"
            )

        return {
            "resolved_selection_json": resolved_selection_json,
            "selection_resolution_source": resolution_source,
            "fallback_reason": fallback_reason,
            "accepted_teacher_count": teacher_count,
            "accepted_family_count": family_count,
            "verified_teacher_count_in_shared_context": verified_teacher_count,
            "research_spec_loaded_for_outer_loop": bool(verification_review.get("research_spec_loaded", False)),
            "preferred_template_hit_rate": preferred_template_hit_rate,
            "frozen_eval_available": frozen_eval_available,
            "frozen_teacher_count": frozen_teacher_count,
            "likely_teacher_count": likely_teacher_count,
            "acceptance_status": acceptance_status,
            "acceptance_principles": [
                "Priority order: current workflow frozen-eval teacher > current workflow likely teacher > imported confirmed frozen teacher > demo fallback teacher.",
                "Prefer strong teachers that also span complementary research domains.",
                "Check whether the generated teachers actually followed the studio-side research mandate before trusting them as fresh discoveries.",
                "Preserve later inner-loop coverage rather than maximizing one single family only.",
                "Allow the studio to reuse proven frozen teachers when the outer loop is not generating new accepted rounds yet.",
            ],
        }
