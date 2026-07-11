from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .base import BaseAgent


class HypothesisAgent(BaseAgent):
    def build_hypothesis_brief(
        self,
        *,
        research_goal: str,
        planning_brief: Mapping[str, Any],
    ) -> Dict[str, Any]:
        goal = str(research_goal or "").strip()
        tags = [str(x).strip() for x in list(planning_brief.get("focus_tags") or []) if str(x).strip()]

        primary_style = "mixed technical edge"
        if "breakout" in tags:
            primary_style = "breakout continuation"
        elif "pullback" in tags and "momentum" in tags:
            primary_style = "momentum pullback continuation"
        elif "pullback" in tags:
            primary_style = "trend pullback continuation"
        elif "reversal" in tags:
            primary_style = "oversold reversal"

        regime_hints: List[str] = []
        if "volatility" in tags:
            regime_hints.append("volatility regime segmentation matters")
        if "volume" in tags:
            regime_hints.append("volume confirmation should affect confidence")
        if "kdj" in tags:
            regime_hints.append("oscillator state may matter near turning zones")
        if not regime_hints:
            regime_hints.append("test multiple regime branches rather than one monolithic rule")

        return {
            "research_goal": goal,
            "primary_hypothesis": (
                f"Signals in the '{primary_style}' family may contain a teacher-trainable ranking edge "
                "when evaluated with regime-aware technical factors."
            ),
            "alternative_hypotheses": [
                "The edge may exist only in a narrow volatility/position subregime.",
                "The edge may require multi-factor confirmation instead of single-factor thresholding.",
            ],
            "target_research_style": primary_style,
            "regime_hints": regime_hints,
            "validation_principles": [
                "Prefer walk-forward stability over one-shot in-sample strength.",
                "Preserve teacher explainability so the inner loop can later internalize the standard.",
                "Keep candidate styles diversified enough for later multi-teacher internalization.",
            ],
            "focus_tags": tags,
        }
