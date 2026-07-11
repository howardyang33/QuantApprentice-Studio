from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..tools.clean_pipeline import CleanPipelineWrapper


@dataclass
class PipelineStep:
    step_id: str
    title: str
    owner: str
    stage_type: str
    status: str = "planned"
    description: str = ""
    wrapper_stage: str = ""
    notes: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePlan:
    mode: str
    research_goal: str
    run_label: str
    steps: List[PipelineStep]
    system_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "research_goal": self.research_goal,
            "run_label": self.run_label,
            "system_note": self.system_note,
            "steps": [asdict(step) for step in self.steps],
        }


class QuantPipelineOrchestrator:
    """High-level workflow planner for the full QuantApprentice pipeline.

    This class does not replace the clean research code. Instead, it maps the
    user-facing studio modes onto wrapper-friendly clean entrypoints.
    """

    def __init__(self) -> None:
        self.clean = CleanPipelineWrapper()

    def describe_modes(self) -> List[Dict[str, Any]]:
        return [
            {
                "mode": "full_pipeline",
                "summary": "Research goal -> outer loop -> teacher explainability -> inner loop -> final lesson -> new signal scoring",
            },
            {
                "mode": "outer_loop_only",
                "summary": "Research goal -> teacher construction and validation only",
            },
            {
                "mode": "inner_loop_only",
                "summary": "Existing teacher selection -> warmup / alignment / final lesson only",
            },
            {
                "mode": "scoring_only",
                "summary": "Existing final lesson -> score new candidate signals only",
            },
        ]

    def plan_full_pipeline(
        self,
        *,
        research_goal: str,
        run_label: str,
        selection_json_hint: str = "",
        final_lesson_state_hint: str = "",
    ) -> PipelinePlan:
        steps = [
            PipelineStep(
                step_id="S1",
                title="Research Planning",
                owner="PlannerAgent",
                stage_type="planning",
                description="Interpret the research goal, inspect prior memory, and select a workflow branch.",
                notes=["This is the studio-native top-level planning step."],
            ),
            PipelineStep(
                step_id="S2",
                title="Hypothesis Framing",
                owner="HypothesisAgent",
                stage_type="planning",
                description="Translate the research goal into a concrete market hypothesis family and validation intent.",
            ),
            PipelineStep(
                step_id="S3",
                title="Factor Design Scoping",
                owner="FactorDesignAgent",
                stage_type="planning",
                description="Define candidate factor families, templates, and diversification intent for teacher construction.",
            ),
            PipelineStep(
                step_id="S4",
                title="Outer Loop Teacher Construction",
                owner="TeacherTrainingAgent",
                stage_type="clean_wrapper",
                wrapper_stage="outer_loop",
                description="Build or extend the teacher zoo from the research goal.",
            ),
            PipelineStep(
                step_id="S5",
                title="Teacher Verification Review",
                owner="VerificationAgent",
                stage_type="review",
                description="Review outer-loop teacher outputs and summarize what was actually produced.",
            ),
            PipelineStep(
                step_id="S6",
                title="Explainability Report Refresh",
                owner="ExplainabilityAgent",
                stage_type="clean_wrapper",
                wrapper_stage="explainability_report_v2",
                description="Rebuild or refresh branch-oriented teacher explainability reports for accepted teachers.",
            ),
            PipelineStep(
                step_id="S7",
                title="Teacher Frozen Evaluation",
                owner="TeacherAcceptanceAgent",
                stage_type="clean_wrapper",
                wrapper_stage="teacher_frozen_eval",
                description="Freeze current-workflow candidate teachers, evaluate them post-cutoff, and emit a formal selection.json.",
            ),
            PipelineStep(
                step_id="S8",
                title="Teacher Acceptance Review",
                owner="TeacherAcceptanceAgent",
                stage_type="review",
                description="Decide which teacher source should feed the inner loop, using formal frozen-eval results whenever available.",
                inputs={"selection_json_hint": selection_json_hint},
            ),
            PipelineStep(
                step_id="S9",
                title="Teacher Selection for Internalization",
                owner="TeacherSelectionAgent",
                stage_type="selection",
                description="Load the accepted frozen-teacher set and summarize the internalization targets.",
                inputs={"selection_json_hint": selection_json_hint},
            ),
            PipelineStep(
                step_id="S10",
                title="Inner Loop Warmup + Alignment",
                owner="ApprenticeAgent",
                stage_type="clean_wrapper",
                wrapper_stage="inner_loop_suite",
                description="Run scoped warmup, checkpoint exploration, and lesson selection.",
            ),
            PipelineStep(
                step_id="S11",
                title="Long-Range Market Backtest",
                owner="EvaluationAgent",
                stage_type="clean_wrapper",
                wrapper_stage="market_backtest",
                description="Evaluate final lessons on non-teacher market signal pools.",
                inputs={"final_lesson_state_hint": final_lesson_state_hint},
            ),
            PipelineStep(
                step_id="S12",
                title="Scoring Service Activation",
                owner="SignalScoringAgent",
                stage_type="studio_runtime",
                description="Expose the selected final lesson for live single-signal and batch scoring.",
            ),
        ]
        return PipelinePlan(
            mode="full_pipeline",
            research_goal=research_goal,
            run_label=run_label,
            steps=steps,
            system_note=(
                "Full Pipeline is the canonical QuantApprentice mode: it connects teacher discovery, "
                "teacher internalization, and scoring deployment instead of treating them as isolated tasks."
            ),
        )

    def plan_outer_loop_only(
        self,
        *,
        research_goal: str,
        run_label: str,
    ) -> PipelinePlan:
        steps = [
            PipelineStep(
                step_id="S1",
                title="Research Planning",
                owner="PlannerAgent",
                stage_type="planning",
                description="Parse the research goal into a teacher-construction objective.",
            ),
            PipelineStep(
                step_id="S2",
                title="Hypothesis Framing",
                owner="HypothesisAgent",
                stage_type="planning",
                description="Translate the research goal into a concrete market hypothesis family and validation intent.",
            ),
            PipelineStep(
                step_id="S3",
                title="Factor Design Scoping",
                owner="FactorDesignAgent",
                stage_type="planning",
                description="Define candidate factor families, templates, and diversification intent for teacher construction.",
            ),
            PipelineStep(
                step_id="S4",
                title="Outer Loop Teacher Construction",
                owner="TeacherTrainingAgent",
                stage_type="clean_wrapper",
                wrapper_stage="outer_loop",
                description="Run the autonomous teacher loop only.",
            ),
            PipelineStep(
                step_id="S5",
                title="Teacher Verification Review",
                owner="VerificationAgent",
                stage_type="review",
                description="Review outer-loop teacher outputs and summarize what was actually produced.",
            ),
            PipelineStep(
                step_id="S6",
                title="Explainability Report Refresh",
                owner="ExplainabilityAgent",
                stage_type="clean_wrapper",
                wrapper_stage="explainability_report_v2",
                description="Refresh explainability assets for the accepted teachers.",
            ),
            PipelineStep(
                step_id="S7",
                title="Teacher Frozen Evaluation",
                owner="TeacherAcceptanceAgent",
                stage_type="clean_wrapper",
                wrapper_stage="teacher_frozen_eval",
                description="Freeze current-workflow candidate teachers and emit a formal post-cutoff selection bundle.",
            ),
            PipelineStep(
                step_id="S8",
                title="Teacher Acceptance Review",
                owner="TeacherAcceptanceAgent",
                stage_type="review",
                description="Summarize which accepted teacher set is available after the outer-loop and frozen-eval pass.",
            ),
        ]
        return PipelinePlan(
            mode="outer_loop_only",
            research_goal=research_goal,
            run_label=run_label,
            steps=steps,
            system_note="Outer Loop Only stops after teacher construction and report generation.",
        )

    def plan_inner_loop_only(
        self,
        *,
        research_goal: str,
        run_label: str,
        selection_json: str,
        api_model: str,
    ) -> PipelinePlan:
        steps = [
            PipelineStep(
                step_id="S1",
                title="Teacher Intake",
                owner="TeacherSelectionAgent",
                stage_type="selection",
                description="Load an existing teacher selection and associated explainability reports.",
                inputs={"selection_json": selection_json},
            ),
            PipelineStep(
                step_id="S2",
                title="Inner Loop Warmup + Alignment",
                owner="ApprenticeAgent",
                stage_type="clean_wrapper",
                wrapper_stage="inner_loop_suite",
                description="Run scoped warmup and final lesson selection from the existing teacher set.",
                inputs={"api_model": api_model},
            ),
            PipelineStep(
                step_id="S3",
                title="Scope Alignment Review",
                owner="EvaluationAgent",
                stage_type="clean_wrapper",
                wrapper_stage="scope_alignment",
                description="Optional standalone alignment replay for the selected final lesson state.",
            ),
        ]
        return PipelinePlan(
            mode="inner_loop_only",
            research_goal=research_goal,
            run_label=run_label,
            steps=steps,
            system_note="Inner Loop Only assumes the teacher zoo already exists and starts from teacher selection.",
        )

    def plan_scoring_only(
        self,
        *,
        research_goal: str,
        run_label: str,
        lesson_alias: str,
    ) -> PipelinePlan:
        steps = [
            PipelineStep(
                step_id="S1",
                title="Lesson Routing",
                owner="LessonAgent",
                stage_type="studio_runtime",
                description="Resolve the final lesson and teacher-scope lesson cards used for scoring.",
                inputs={"lesson_alias": lesson_alias},
            ),
            PipelineStep(
                step_id="S2",
                title="Signal Validation and Normalization",
                owner="SignalScoringAgent",
                stage_type="studio_runtime",
                description="Validate incoming signals against the canonical archived schema before scoring.",
            ),
            PipelineStep(
                step_id="S3",
                title="Live Signal Scoring",
                owner="SignalScoringAgent",
                stage_type="studio_runtime",
                description="Call the local GPT-OSS runtime with the final lesson and structured factor payload.",
            ),
        ]
        return PipelinePlan(
            mode="scoring_only",
            research_goal=research_goal,
            run_label=run_label,
            steps=steps,
            system_note="Scoring Only is the application stage. It does not retrain teachers or lessons.",
        )

    def build_plan(self, *, mode: str, research_goal: str, run_label: str, **kwargs: Any) -> Dict[str, Any]:
        mode_key = str(mode).strip().lower()
        if mode_key == "full_pipeline":
            return self.plan_full_pipeline(research_goal=research_goal, run_label=run_label, **kwargs).to_dict()
        if mode_key == "outer_loop_only":
            return self.plan_outer_loop_only(research_goal=research_goal, run_label=run_label).to_dict()
        if mode_key == "inner_loop_only":
            return self.plan_inner_loop_only(research_goal=research_goal, run_label=run_label, **kwargs).to_dict()
        if mode_key == "scoring_only":
            return self.plan_scoring_only(research_goal=research_goal, run_label=run_label, **kwargs).to_dict()
        raise KeyError(f"unknown pipeline mode: {mode}")
