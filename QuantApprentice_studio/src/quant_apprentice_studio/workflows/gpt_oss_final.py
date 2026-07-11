from __future__ import annotations

from ..agents.chief import ChiefResearchAgent
from ..registry import StudioRegistry


class GPTOSSFinalWorkflow:
    def __init__(self, profile_id: str = "gpt_oss_20b_final") -> None:
        self.registry = StudioRegistry(profile_id)
        self.chief = ChiefResearchAgent(self.registry)

    def runtime_probe(self, *, lesson_alias: str, market_run_alias: str, signal_date: str, symbol: str) -> dict:
        return {
            "overview": self.chief.overview(),
            "teachers": [item.__dict__ for item in self.chief.teacher_zoo.list_teachers()],
            "lesson_summary": self.chief.lesson.summarize_lesson_run(lesson_alias),
            "alignment_mean": self.chief.alignment.get_after_warmup_result("Mean"),
            "recorded_signal_score": self.chief.scoring.score_recorded(
                market_run_alias=market_run_alias,
                signal_date=signal_date,
                symbol=symbol,
            ).__dict__,
        }
