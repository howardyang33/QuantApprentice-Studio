from __future__ import annotations

from typing import Dict, List, Optional

from ..llm.backend import StudioLLMBackend
from .base import BaseAgent


def _goal_tags(text: str) -> List[str]:
    lowered = str(text or "").lower()
    table = [
        ("breakout", ["breakout", "突破"]),
        ("pullback", ["pullback", "回调"]),
        ("reversal", ["reversal", "反转", "超跌"]),
        ("momentum", ["momentum", "动量"]),
        ("volatility", ["volatility", "波动率"]),
        ("volume", ["volume", "量能"]),
        ("kdj", ["kdj"]),
        ("market_backtest", ["backtest", "回测"]),
        ("alignment", ["alignment", "内化", "lesson"]),
    ]
    tags: List[str] = []
    for tag, clues in table:
        if any(clue in lowered for clue in clues):
            tags.append(tag)
    return tags


class PlannerAgent(BaseAgent):
    def __init__(self, registry, llm_backend: Optional[StudioLLMBackend] = None) -> None:
        super().__init__(registry)
        self.llm_backend = llm_backend or StudioLLMBackend()

    def build_research_brief(self, *, research_goal: str, mode: str) -> Dict:
        text = str(research_goal or "").strip()
        tags = _goal_tags(text)
        return {
            "research_goal": text,
            "requested_mode": str(mode).strip(),
            "focus_tags": tags,
            "recommended_execution_style": (
                "teacher_construction_then_internalization"
                if mode == "full_pipeline"
                else "single_branch_execution"
            ),
            "notes": [
                "Use frozen/imported artifacts whenever possible before triggering expensive recomputation.",
                "Prefer wrapper-driven execution so the clean research tree remains unchanged.",
            ],
        }

    def llm_backend_status(self) -> Dict:
        return self.llm_backend.describe()
