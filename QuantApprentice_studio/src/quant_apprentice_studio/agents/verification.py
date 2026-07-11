from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..provenance import read_json
from .base import BaseAgent


class VerificationAgent(BaseAgent):
    def _is_current_workflow_likely_teacher(self, row: Mapping[str, Any]) -> bool:
        if bool(row.get("accepted_as_teacher", False)):
            return True
        if not bool(row.get("explainability_report_exists", False)):
            return False
        if not bool(row.get("branch_rule_cards_exists", False)):
            return False

        positive_years = int(row.get("positive_years", 0) or 0)
        total_years = max(int(row.get("total_years", 0) or 0), 1)
        nav_positive_years = int(row.get("nav_positive_years", 0) or 0)
        nav_total_years = max(int(row.get("nav_total_years", 0) or 0), 1)
        mean_alpha = float(row.get("mean_alpha", 0.0) or 0.0)
        nav_cagr = float(row.get("nav_cagr", 0.0) or 0.0)
        nav_max_drawdown = float(row.get("nav_max_drawdown", 0.0) or 0.0)

        stable_alpha = mean_alpha > 0.0 and positive_years >= total_years
        stable_nav = nav_cagr > 0.0 and nav_positive_years >= nav_total_years
        drawdown_not_extreme = nav_max_drawdown >= -0.20
        return drawdown_not_extreme and (stable_alpha or stable_nav)

    def review_teacher_outputs(
        self,
        *,
        shared_context_root: str,
        teacher_training_summary: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary = dict(teacher_training_summary or {})
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        memory_root = Path(shared_context_root).expanduser().resolve() / "research_memory"
        zoo_index_path = memory_root / "indexes" / "teacher_zoo_index.json"
        zoo_payload = read_json(zoo_index_path) if zoo_index_path.exists() else {"teachers": []}

        candidate_teachers: List[Dict[str, Any]] = []
        if report_root.exists():
            for spec_path in sorted(report_root.rglob("selected_spec.json")):
                round_dir = spec_path.parent
                round_id = round_dir.name
                if "_frozen_" in round_id:
                    continue
                selected_spec = read_json(spec_path)
                candidate_teachers.append(
                    {
                        "round_id": round_id,
                        "title": str(selected_spec.get("title", "")).strip(),
                        "research_family": str(selected_spec.get("research_family", "")).strip(),
                        "sample_template": str(selected_spec.get("sample_template", "")).strip(),
                        "has_execution_report": (round_dir / "EXECUTION_REPORT.md").exists(),
                        "has_factor_analysis_summary": (round_dir / "factor_analysis_summary.json").exists(),
                        "has_branch_rule_cards": (round_dir / "branch_rule_cards.json").exists(),
                    }
                )

        validated_teachers: List[Dict[str, Any]] = []
        for row in list(zoo_payload.get("teachers") or []):
            round_id = str(row.get("round_id", "")).strip()
            if not round_id or "_frozen_" in round_id:
                continue
            metrics = dict(row.get("metrics") or {})
            report_dir = report_root / round_id
            nav_positive_years = int(metrics.get("nav_positive_years", 0) or 0)
            nav_total_years = int(metrics.get("nav_total_years", 0) or 0)
            positive_years = int(row.get("positive_years", 0) or 0)
            total_years = int(row.get("total_years", 0) or 0)
            if positive_years <= 0 and nav_positive_years > 0:
                positive_years = nav_positive_years
            if total_years <= 0 and nav_total_years > 0:
                total_years = nav_total_years
            teacher_row = {
                "round_id": round_id,
                "title": str(row.get("title", "")).strip(),
                "research_family": str(row.get("research_family", "")).strip(),
                "sample_template": str(row.get("sample_template", "")).strip(),
                "model_family": str(row.get("model_family", "")).strip(),
                "accepted_as_teacher": bool(row.get("accepted_as_teacher", False)),
                "status": str(row.get("status", "")).strip(),
                "zoo_partition": str(row.get("zoo_partition", "")).strip(),
                "mean_alpha": float(metrics.get("mean_alpha", 0.0) or 0.0),
                "uplift_mean": float(metrics.get("mean_alpha", 0.0) or 0.0),
                "nav_cagr": float(metrics.get("nav_cagr", 0.0) or 0.0),
                "nav_total_return": float(metrics.get("nav_total_return", 0.0) or 0.0),
                "nav_max_drawdown": float(metrics.get("nav_max_drawdown", 0.0) or 0.0),
                "positive_years": positive_years,
                "total_years": total_years,
                "nav_positive_years": nav_positive_years,
                "nav_total_years": nav_total_years,
                "feature_count": int(row.get("feature_count", 0) or 0),
                "explainability_report_exists": (report_dir / "FACTOR_ANALYSIS_REPORT.md").exists(),
                "branch_rule_cards_exists": (report_dir / "branch_rule_cards.json").exists(),
                "selected_spec_json": str(report_dir / "selected_spec.json"),
            }
            teacher_row["is_current_workflow_likely_teacher"] = self._is_current_workflow_likely_teacher(teacher_row)
            validated_teachers.append(teacher_row)

        rounds: List[Dict[str, Any]] = []
        for row in list(summary.get("rounds") or []):
            round_id = str(row.get("round_id", "")).strip()
            round_dir = report_root / round_id
            nav_summary_path = round_dir / "nav_curve_backtest" / "nav_summary.json"
            nav_summary = read_json(nav_summary_path) if nav_summary_path.exists() else {}
            rounds.append(
                {
                    "round_id": round_id,
                    "title": str(row.get("title", "")).strip(),
                    "research_family": str(row.get("research_family", "")).strip(),
                    "sample_template": str(row.get("sample_template", "")).strip(),
                    "walkforward_final_nav": float(nav_summary.get("final_nav", 0.0) or 0.0),
                }
            )

        sorted_rounds = sorted(rounds, key=lambda x: float(x.get("walkforward_final_nav", 0.0)), reverse=True)
        likely_validated_teachers = sorted(
            [
                row
                for row in validated_teachers
                if row.get("accepted_as_teacher") or row.get("is_current_workflow_likely_teacher")
            ],
            key=lambda x: (
                float(x.get("nav_cagr", 0.0)),
                float(x.get("mean_alpha", 0.0)),
                -float(x.get("nav_max_drawdown", 0.0)),
                int(x.get("positive_years", 0)),
            ),
            reverse=True,
        )
        preferred_template_hit_count = int(summary.get("preferred_template_hit_count", 0) or 0)
        preferred_template_hit_rate = (
            preferred_template_hit_count / len(rounds)
            if rounds
            else 0.0
        )
        return {
            "shared_teacher_report_root": str(report_root),
            "verified_teacher_count": len(sorted_rounds),
            "research_spec_loaded": bool(summary.get("research_spec_loaded", False)),
            "preferred_sample_templates": list(summary.get("preferred_sample_templates") or []),
            "preferred_template_hit_count": preferred_template_hit_count,
            "preferred_template_hit_rate": preferred_template_hit_rate,
            "candidate_teacher_count": len(candidate_teachers),
            "candidate_teachers": candidate_teachers,
            "validated_teacher_count": len(validated_teachers),
            "validated_teachers": validated_teachers,
            "likely_teacher_count": len(likely_validated_teachers),
            "likely_teachers": likely_validated_teachers[:8],
            "current_workflow_family_count": len(
                {str(row.get("research_family", "")).strip() for row in likely_validated_teachers if str(row.get("research_family", "")).strip()}
            ),
            "top_rounds_by_final_nav": sorted_rounds[:5],
            "verification_status": (
                "no_new_teacher_outputs_detected"
                if not candidate_teachers and not validated_teachers
                else "teacher_outputs_detected_and_ranked"
            ),
            "verification_notes": [
                "Verification here is studio-side review of wrapper outputs, not a reimplementation of clean walk-forward logic.",
                "A nonzero preferred-template hit rate indicates the clean teacher proposal step responded to the studio-side research mandate.",
                "Accepted teachers should later be chosen for both performance and domain complementarity.",
            ],
        }
