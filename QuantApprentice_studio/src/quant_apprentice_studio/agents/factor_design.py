from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .base import BaseAgent


class FactorDesignAgent(BaseAgent):
    def build_factor_design_brief(
        self,
        *,
        research_goal: str,
        planning_brief: Mapping[str, Any],
        hypothesis_brief: Mapping[str, Any],
    ) -> Dict[str, Any]:
        goal = str(research_goal or "").strip()
        tags = [str(x).strip() for x in list(planning_brief.get("focus_tags") or []) if str(x).strip()]
        target_style = str(hypothesis_brief.get("target_research_style", "")).strip()

        factor_families: List[str] = ["price position", "short-horizon returns", "realized volatility"]
        if "breakout" in tags:
            factor_families.extend(["moving-average distance", "range expansion", "volume confirmation"])
        if "pullback" in tags:
            factor_families.extend(["pullback depth", "rebound readiness", "trend context"])
        if "reversal" in tags:
            factor_families.extend(["oversold state", "reversal candle texture"])
        if "kdj" in tags:
            factor_families.append("oscillator divergence / rebound")
        if "volume" in tags:
            factor_families.append("volume participation")
        if "volatility" in tags:
            factor_families.append("volatility regime switching")

        preferred_templates: List[str] = []
        if "breakout" in tags:
            preferred_templates.append("trend_breakout_pool")
        if "pullback" in tags:
            preferred_templates.append("trend_pullback_pool")
        if "reversal" in tags:
            preferred_templates.append("weak_state_reversal_pool")
        if not preferred_templates:
            preferred_templates = ["trend_pullback_pool", "trend_breakout_pool"]

        return {
            "research_goal": goal,
            "target_research_style": target_style,
            "preferred_sample_templates": preferred_templates,
            "candidate_factor_families": factor_families,
            "design_constraints": [
                "Favor factors that can later be verbalized into explainable lesson items.",
                "Avoid overfitting to one narrow indicator unless it survives walk-forward verification.",
                "Leave room for multiple teacher comfort zones rather than one single canonical rule.",
            ],
            "teacher_diversification_objective": [
                "capture at least one continuation-style teacher if available",
                "capture at least one pullback/reversal-style teacher if available",
                "preserve domain diversity for later multi-teacher warmup",
            ],
        }
