from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .base import BaseAgent


class ExplainabilityAgent(BaseAgent):
    def _has_required_factor_artifacts(self, round_dir: Path) -> bool:
        required_names = [
            "factor_analysis_summary.json",
            "FACTOR_ANALYSIS_REPORT.md",
            "shap_global_summary.csv",
            "feature_group_effectiveness.csv",
            "feature_redundant_pairs.csv",
            "feature_combo_effects.csv",
            "shap_local_examples.csv",
            "feature_pdp_curves.csv",
        ]
        return all((round_dir / name).exists() for name in required_names)

    def build_refresh_args(
        self,
        *,
        shared_context_root: str,
        verification_review: Mapping[str, Any],
    ) -> List[str]:
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        round_ids = [
            str(row.get("round_id", "")).strip()
            for row in list(verification_review.get("candidate_teachers") or [])
            if (
                str(row.get("round_id", "")).strip()
                and "_frozen_" not in str(row.get("round_id", "")).strip()
                and self._has_required_factor_artifacts(report_root / str(row.get("round_id", "")).strip())
            )
        ]
        args: List[str] = []
        if report_root.exists() and round_ids:
            args.extend(["--report-root", str(report_root)])
            for round_id in round_ids:
                args.extend(["--round-id", round_id])
        return args

    def has_teacher_reports(self, *, shared_context_root: str) -> bool:
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        return report_root.exists() and any(report_root.rglob("factor_analysis_summary.json"))

    def summarize_outputs(self, *, shared_context_root: str) -> Dict[str, Any]:
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        v2_paths = sorted(report_root.rglob("factor_analysis_summary_v2.json")) if report_root.exists() else []
        branch_cards = sorted(report_root.rglob("branch_rule_cards.json")) if report_root.exists() else []
        return {
            "shared_teacher_report_root": str(report_root),
            "report_v2_count": len(v2_paths),
            "branch_rule_card_count": len(branch_cards),
            "sample_report_v2_json": str(v2_paths[0]) if v2_paths else "",
            "sample_branch_rule_cards_json": str(branch_cards[0]) if branch_cards else "",
        }
