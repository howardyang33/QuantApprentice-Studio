from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..provenance import read_json
from .base import BaseAgent


class TeacherTrainingAgent(BaseAgent):
    def build_research_spec(
        self,
        *,
        research_goal: str,
        planning_brief: Mapping[str, Any],
        hypothesis_brief: Mapping[str, Any],
        factor_design_brief: Mapping[str, Any],
    ) -> Dict[str, Any]:
        goal = str(research_goal or "").strip()
        tags = [str(x).strip() for x in list(planning_brief.get("focus_tags") or []) if str(x).strip()]
        target_style = str(hypothesis_brief.get("target_research_style", "")).strip() or "mixed technical edge"
        preferred_templates = [
            str(x).strip()
            for x in list(factor_design_brief.get("preferred_sample_templates") or [])
            if str(x).strip()
        ]
        candidate_factor_families = [
            str(x).strip()
            for x in list(factor_design_brief.get("candidate_factor_families") or [])
            if str(x).strip()
        ]
        regime_hints = [
            str(x).strip()
            for x in list(hypothesis_brief.get("regime_hints") or [])
            if str(x).strip()
        ]
        design_constraints = [
            str(x).strip()
            for x in list(factor_design_brief.get("design_constraints") or [])
            if str(x).strip()
        ]
        diversification_objective = [
            str(x).strip()
            for x in list(factor_design_brief.get("teacher_diversification_objective") or [])
            if str(x).strip()
        ]
        validation_principles = [
            str(x).strip()
            for x in list(hypothesis_brief.get("validation_principles") or [])
            if str(x).strip()
        ]

        family_priority: List[str] = []
        if "breakout" in tags or "breakout" in target_style.lower():
            family_priority.extend(["breakout", "trend_continuation"])
        if "momentum" in tags and "pullback" in tags:
            family_priority.extend(["momentum_pullback", "trend_pullback"])
        elif "pullback" in tags:
            family_priority.extend(["trend_pullback", "trend_reentry"])
        if "reversal" in tags:
            family_priority.extend(["oversold_reversal", "mean_reversion"])
        if not family_priority:
            family_priority.extend(["trend_pullback", "breakout"])

        fallback_template = preferred_templates[0] if preferred_templates else "trend_pullback_pool"
        fallback_model_family = "logistic_regression" if fallback_template == "trend_breakout_pool" else "ridge_regression"
        fallback_target_kind = (
            "future_return_positive_5d"
            if fallback_template == "trend_breakout_pool"
            else "future_return_5d"
        )

        studio_mandate_lines = [
            f"Research goal: {goal}" if goal else "Research goal: not provided",
            f"Target style: {target_style}",
            "Prefer proposals that match the sample-template priority unless novelty evidence strongly argues otherwise.",
            "Preserve teacher-zoo diversification rather than cloning an existing family.",
            "Favor feature sets that can later be verbalized into explainable lesson items.",
        ]

        return {
            "research_goal": goal,
            "planning_focus_tags": tags,
            "target_research_style": target_style,
            "primary_hypothesis": str(hypothesis_brief.get("primary_hypothesis", "")).strip(),
            "alternative_hypotheses": [
                str(x).strip()
                for x in list(hypothesis_brief.get("alternative_hypotheses") or [])
                if str(x).strip()
            ],
            "regime_hints": regime_hints,
            "preferred_sample_templates": preferred_templates,
            "candidate_factor_families": candidate_factor_families,
            "design_constraints": design_constraints,
            "diversification_objective": diversification_objective,
            "validation_principles": validation_principles,
            "family_priority": family_priority,
            "fallback_preferences": {
                "sample_template": fallback_template,
                "model_family": fallback_model_family,
                "target_kind": fallback_target_kind,
            },
            "studio_mandate_lines": studio_mandate_lines,
        }

    def build_outer_loop_args(
        self,
        *,
        research_goal: str,
        run_tag_base: str,
    ) -> List[str]:
        # The clean outer-loop entrypoint is still mostly environment-driven.
        # We keep the argv surface minimal here and let studio-side planning
        # metadata carry the richer intent.
        _ = (research_goal, run_tag_base)
        return []

    def summarize_outputs(
        self,
        *,
        shared_context_root: str,
        research_spec_json: str = "",
    ) -> Dict[str, Any]:
        report_root = Path(shared_context_root).expanduser().resolve() / "reports" / "teacher_loop"
        research_spec: Dict[str, Any] = {}
        if str(research_spec_json).strip():
            path = Path(str(research_spec_json)).expanduser().resolve()
            if path.exists():
                try:
                    research_spec = dict(read_json(path) or {})
                except Exception:
                    research_spec = {}
        preferred_templates = {
            str(x).strip()
            for x in list(research_spec.get("preferred_sample_templates") or [])
            if str(x).strip()
        }
        rounds: List[Dict[str, Any]] = []
        if report_root.exists():
            for spec_path in sorted(report_root.rglob("selected_spec.json")):
                round_dir = spec_path.parent
                round_id = round_dir.name
                selected_spec = read_json(spec_path)
                factor_summary_path = round_dir / "factor_analysis_summary.json"
                nav_summary_path = round_dir / "nav_curve_backtest" / "nav_summary.json"
                sample_template = str(selected_spec.get("sample_template", "")).strip()
                rounds.append(
                    {
                        "round_id": round_id,
                        "title": str(selected_spec.get("title", "")).strip(),
                        "research_family": str(selected_spec.get("research_family", "")).strip(),
                        "sample_template": sample_template,
                        "has_factor_analysis_summary": factor_summary_path.exists(),
                        "has_nav_summary": nav_summary_path.exists(),
                        "matches_studio_template_preference": sample_template in preferred_templates if preferred_templates else None,
                    }
                )
        matched_rounds = [
            row for row in rounds
            if row.get("matches_studio_template_preference") is True
        ]
        return {
            "shared_teacher_report_root": str(report_root),
            "research_spec_json": str(research_spec_json).strip(),
            "research_spec_loaded": bool(research_spec),
            "preferred_sample_templates": sorted(preferred_templates),
            "teacher_report_count": len(rounds),
            "preferred_template_hit_count": len(matched_rounds),
            "rounds": rounds,
        }
