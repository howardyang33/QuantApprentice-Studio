from __future__ import annotations

from .acceptance import TeacherAcceptanceAgent
from .apprentice import ApprenticeAgent
from .alignment import AlignmentAgent
from .backtest import BacktestAgent
from .evaluation import EvaluationAgent
from .explainability import ExplainabilityAgent
from .factor_design import FactorDesignAgent
from .hypothesis import HypothesisAgent
from .lesson import LessonAgent
from .memory import MemoryAgent
from .planner import PlannerAgent
from .scoring import SignalScoringAgent
from .selection import TeacherSelectionAgent
from .teacher_training import TeacherTrainingAgent
from .teacher_zoo import TeacherZooAgent
from .verification import VerificationAgent


class ChiefResearchAgent:
    def __init__(self, registry) -> None:
        self.registry = registry
        self.memory = MemoryAgent(registry)
        self.planner = PlannerAgent(registry)
        self.hypothesis = HypothesisAgent(registry)
        self.factor_design = FactorDesignAgent(registry)
        self.teacher_zoo = TeacherZooAgent(registry)
        self.teacher_training = TeacherTrainingAgent(registry)
        self.verification = VerificationAgent(registry)
        self.explainability = ExplainabilityAgent(registry)
        self.acceptance = TeacherAcceptanceAgent(registry)
        self.selection = TeacherSelectionAgent(registry)
        self.apprentice = ApprenticeAgent(registry)
        self.lesson = LessonAgent(registry)
        self.alignment = AlignmentAgent(registry)
        self.backtest = BacktestAgent(registry)
        self.evaluation = EvaluationAgent(registry)
        self.scoring = SignalScoringAgent(registry)

    def overview(self) -> dict:
        catalog = self.registry.load_runtime_catalog()
        return {
            "profile_id": catalog["profile_id"],
            "backbone": catalog["backbone"],
            "teacher_count": len(self.teacher_zoo.list_teachers()),
            "lesson_run_count": len(self.lesson.list_lesson_runs()),
            "market_run_count": len(self.backtest.list_market_runs()),
            "default_alignment_seed_alias": catalog["defaults"].get("alignment_seed_alias", ""),
            "default_market_run_alias": catalog["defaults"].get("market_run_alias", ""),
        }
