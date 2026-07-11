from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..agents.chief import ChiefResearchAgent
from ..provenance import write_json
from ..tools.clean_pipeline import CleanPipelineWrapper


@dataclass
class WorkflowExecutionContext:
    mode: str
    research_goal: str
    run_label: str
    workflow_root: Path
    shared_context_root: Path
    resolved_selection_json: str
    resolved_api_model: str
    lesson_alias: str
    final_lesson_state_json: str
    data_dir: str
    selection_resolution: Dict[str, Any]
    planning_brief: Dict[str, Any]
    teacher_selection_summary: Dict[str, Any]
    run_contract: Dict[str, Any]
    project_id: str
    dataset_id: str
    allow_imported_fallback: bool
    allow_demo_fallback: bool
    global_env: Dict[str, str]
    stage_args: Dict[str, list[str]]
    stage_env: Dict[str, Dict[str, str]]
    step_status_by_stage: Dict[str, str] = field(default_factory=dict)
    generated_final_lesson_artifact_json: str = ""
    research_spec_json: str = ""
    agent_payloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class WorkflowStepExecutor:
    def __init__(
        self,
        *,
        chief: ChiefResearchAgent,
        clean: CleanPipelineWrapper,
    ) -> None:
        self.chief = chief
        self.clean = clean

    def execute(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
        check: bool = False,
        allow_manual_steps: bool = True,
    ) -> Dict[str, Any]:
        stage_type = str(step.get("stage_type", "")).strip()
        owner = str(step.get("owner", "")).strip()
        wrapper_stage = str(step.get("wrapper_stage", "")).strip()
        if stage_type in {"planning", "selection", "review"}:
            return self._execute_native_agent_step(step=step, context=context, allow_manual_steps=allow_manual_steps)
        if stage_type == "studio_runtime":
            return self._execute_studio_runtime_step(step=step, context=context)
        if stage_type == "clean_wrapper" and wrapper_stage:
            return self._execute_clean_wrapper_step(step=step, context=context, check=check)
        status = "manual_pending" if allow_manual_steps else "skipped"
        return {
            "step_id": str(step.get("step_id", "")).strip(),
            "title": str(step.get("title", "")).strip(),
            "owner": owner,
            "stage_type": stage_type,
            "wrapper_stage": wrapper_stage,
            "status": status,
            "notes": list(step.get("notes", [])),
            "inputs": dict(step.get("inputs", {})),
            "description": str(step.get("description", "")),
            "payload": {},
        }

    def _artifact_path_for_step(self, *, step: Mapping[str, Any]) -> Path:
        step_id = str(step.get("step_id", "")).strip() or "step"
        owner = str(step.get("owner", "")).strip() or "agent"
        safe_owner = owner.replace("Agent", "").replace(" ", "_").lower()
        return Path(step_id + f"_{safe_owner}.json")

    def _persist_native_payload(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        rel = Path("agent_artifacts") / self._artifact_path_for_step(step=step)
        path = context.workflow_root / rel
        write_json(path, payload)
        out = dict(payload)
        out["artifact_json"] = str(path)
        return out

    def _completed_step(self, step: Mapping[str, Any], *, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step_id": str(step.get("step_id", "")).strip(),
            "title": str(step.get("title", "")).strip(),
            "owner": str(step.get("owner", "")).strip(),
            "stage_type": str(step.get("stage_type", "")).strip(),
            "wrapper_stage": str(step.get("wrapper_stage", "")).strip(),
            "status": "completed",
            "notes": list(step.get("notes", [])),
            "inputs": dict(step.get("inputs", {})),
            "description": str(step.get("description", "")),
            "payload": payload,
        }

    def _execute_native_agent_step(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
        allow_manual_steps: bool,
    ) -> Dict[str, Any]:
        owner = str(step.get("owner", "")).strip()

        if owner == "PlannerAgent":
            payload = dict(context.planning_brief)
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["PlannerAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        if owner == "HypothesisAgent":
            payload = self.chief.hypothesis.build_hypothesis_brief(
                research_goal=context.research_goal,
                planning_brief=context.planning_brief,
            )
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["HypothesisAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        if owner == "FactorDesignAgent":
            payload = self.chief.factor_design.build_factor_design_brief(
                research_goal=context.research_goal,
                planning_brief=context.planning_brief,
                hypothesis_brief=context.agent_payloads.get("HypothesisAgent", {}),
            )
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["FactorDesignAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        if owner in {"MemoryAgent", "TeacherSelectionAgent"}:
            selection_resolution = self.chief.selection.materialize_workflow_selection(
                shared_context_root=str(context.shared_context_root),
                verification_review=context.agent_payloads.get("VerificationAgent", {}),
                frozen_eval_summary=context.agent_payloads.get("TeacherFrozenEval", {}),
                fallback_selection_json=context.resolved_selection_json,
                fallback_resolution_source=str(context.selection_resolution.get("resolution_source", "")).strip() or "registry_default",
                allow_imported_fallback=context.allow_imported_fallback,
                allow_demo_fallback=context.allow_demo_fallback,
            )
            context.selection_resolution = dict(selection_resolution)
            context.resolved_selection_json = str(selection_resolution.get("selection_json", "")).strip()
            context.teacher_selection_summary = self.chief.selection.summarize_selection(
                context.resolved_selection_json,
                shared_context_root=str(context.shared_context_root),
            )
            context.teacher_selection_summary["resolution_source"] = str(
                selection_resolution.get("resolution_source", "")
            ).strip()
            context.teacher_selection_summary["fallback_reason"] = str(
                selection_resolution.get("fallback_reason", "")
            ).strip()
            context.teacher_selection_summary["selection_artifact_json"] = str(
                selection_resolution.get("selection_artifact_json", "")
            ).strip()
            payload = dict(context.teacher_selection_summary)
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["TeacherSelectionAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        if owner == "VerificationAgent":
            payload = self.chief.verification.review_teacher_outputs(
                shared_context_root=str(context.shared_context_root),
                teacher_training_summary=context.agent_payloads.get("TeacherTrainingAgent", {}),
            )
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["VerificationAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        if owner == "TeacherAcceptanceAgent":
            selection_resolution = self.chief.selection.materialize_workflow_selection(
                shared_context_root=str(context.shared_context_root),
                verification_review=context.agent_payloads.get("VerificationAgent", {}),
                frozen_eval_summary=context.agent_payloads.get("TeacherFrozenEval", {}),
                fallback_selection_json=context.resolved_selection_json,
                fallback_resolution_source=str(context.selection_resolution.get("resolution_source", "")).strip() or "registry_default",
                allow_imported_fallback=context.allow_imported_fallback,
                allow_demo_fallback=context.allow_demo_fallback,
            )
            context.selection_resolution = dict(selection_resolution)
            context.resolved_selection_json = str(selection_resolution.get("selection_json", "")).strip()
            context.teacher_selection_summary = self.chief.selection.summarize_selection(
                context.resolved_selection_json,
                shared_context_root=str(context.shared_context_root),
            )
            context.teacher_selection_summary["resolution_source"] = str(
                selection_resolution.get("resolution_source", "")
            ).strip()
            context.teacher_selection_summary["fallback_reason"] = str(
                selection_resolution.get("fallback_reason", "")
            ).strip()
            context.teacher_selection_summary["selection_artifact_json"] = str(
                selection_resolution.get("selection_artifact_json", "")
            ).strip()
            payload = self.chief.acceptance.build_acceptance_review(
                selection_resolution=context.selection_resolution,
                teacher_selection_summary=context.teacher_selection_summary,
                verification_review=context.agent_payloads.get("VerificationAgent", {}),
                frozen_eval_summary=context.agent_payloads.get("TeacherFrozenEval", {}),
            )
            payload = self._persist_native_payload(step=step, context=context, payload=payload)
            context.agent_payloads["TeacherAcceptanceAgent"] = dict(payload)
            return self._completed_step(step, payload=payload)

        status = "manual_pending" if allow_manual_steps else "skipped"
        return {
            "step_id": str(step.get("step_id", "")).strip(),
            "title": str(step.get("title", "")).strip(),
            "owner": owner,
            "stage_type": str(step.get("stage_type", "")).strip(),
            "wrapper_stage": str(step.get("wrapper_stage", "")).strip(),
            "status": status,
            "notes": list(step.get("notes", [])),
            "inputs": dict(step.get("inputs", {})),
            "description": str(step.get("description", "")),
            "payload": {},
        }

    def _run_clean_stage(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
        wrapper_stage: str,
        default_stage_args: Optional[Sequence[str]] = None,
        stage_specific_env: Optional[Mapping[str, str]] = None,
        check: bool = False,
    ) -> Dict[str, Any]:
        merged_env = dict(context.global_env)
        merged_env.update({str(k): str(v) for k, v in dict(stage_specific_env or {}).items()})
        merged_env.update(context.stage_env.get(wrapper_stage, {}))
        if str(context.data_dir).strip():
            # User-uploaded / project-isolated datasets must win over imported
            # paper defaults from _base_clean_env().
            merged_env["TEACHER_LOOP_DATA_DIR"] = str(context.data_dir).strip()
            merged_env["APPRENTICE_MASTER_CACHE_PATH"] = str(
                context.shared_context_root / "cache" / "apprentice_loop" / "master_feature_label_studio.joblib"
            )
        stage_run_label = f"{context.run_label}__{str(step.get('step_id', '')).strip()}_{wrapper_stage}"
        payload = self.clean.run_stage(
            wrapper_stage,
            extra_args=[*(default_stage_args or []), *context.stage_args.get(wrapper_stage, [])],
            run_label=stage_run_label,
            context_root=str(context.shared_context_root),
            data_dir=context.data_dir,
            extra_env=merged_env,
            check=check,
        )
        status = "completed" if payload.get("ok") else "failed"
        context.step_status_by_stage[wrapper_stage] = status
        return {
            "step_id": str(step.get("step_id", "")).strip(),
            "title": str(step.get("title", "")).strip(),
            "owner": str(step.get("owner", "")).strip(),
            "stage_type": str(step.get("stage_type", "")).strip(),
            "wrapper_stage": wrapper_stage,
            "status": status,
            "payload": payload,
        }

    def _base_clean_env(self) -> Dict[str, str]:
        return {
            "TEACHER_LOOP_DATA_DIR": self.chief.memory.resolve_original_stock_data_dir(),
            "APPRENTICE_MASTER_CACHE_PATH": self.chief.memory.resolve_shared_master_cache_path(),
            "NAV_CURVE_HS300_INDEX_FILE": self.chief.memory.resolve_hs300_index_path(),
        }

    def _teacher_gpu_env(self) -> Dict[str, str]:
        return {
            "TEACHER_LOOP_REQUIRE_GPU": os.environ.get("QA_STUDIO_TEACHER_LOOP_REQUIRE_GPU", "true"),
            "TEACHER_LOOP_GPU_DEVICE": os.environ.get("QA_STUDIO_TEACHER_LOOP_GPU_DEVICE", "cuda"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("QA_STUDIO_TEACHER_CUDA_VISIBLE_DEVICES", "1"),
        }

    def _imported_teacher_env(self) -> Dict[str, str]:
        return {
            "TEACHER_LOOP_REPORT_ROOT": self.chief.memory.resolve_original_teacher_report_root(),
            "TEACHER_LOOP_ARTIFACT_ROOT": self.chief.memory.resolve_original_teacher_artifact_root(),
            **self._base_clean_env(),
        }

    def _external_workflow_teacher_env(self, *, external_shared_context_root: str) -> Dict[str, str]:
        external_root = Path(external_shared_context_root).expanduser().resolve()
        return {
            "TEACHER_LOOP_REPORT_ROOT": str(external_root / "reports" / "teacher_loop"),
            "TEACHER_LOOP_ARTIFACT_ROOT": str(external_root / "research_memory" / "artifacts" / "teacher_loop"),
            **self._base_clean_env(),
        }

    def _prepare_research_spec(self, *, context: WorkflowExecutionContext) -> Dict[str, Any]:
        payload = self.chief.teacher_training.build_research_spec(
            research_goal=context.research_goal,
            planning_brief=context.planning_brief,
            hypothesis_brief=context.agent_payloads.get("HypothesisAgent", {}),
            factor_design_brief=context.agent_payloads.get("FactorDesignAgent", {}),
        )
        path = context.shared_context_root / "studio_control" / "research_spec.json"
        write_json(path, payload)
        context.research_spec_json = str(path)
        payload_with_path = dict(payload)
        payload_with_path["artifact_json"] = str(path)
        context.agent_payloads["TeacherResearchSpec"] = dict(payload_with_path)
        return payload_with_path

    def _selection_runtime_env(self, *, context: WorkflowExecutionContext) -> Dict[str, str]:
        selection_path = Path(str(context.resolved_selection_json).strip()).expanduser().resolve()
        shared_root = context.shared_context_root.expanduser().resolve()
        if selection_path == shared_root or shared_root in selection_path.parents:
            return self._base_clean_env()
        external_shared_root = self.chief.memory.resolve_workflow_shared_context_root(str(selection_path))
        if external_shared_root:
            return self._external_workflow_teacher_env(external_shared_context_root=external_shared_root)
        return self._imported_teacher_env()

    def _execute_clean_wrapper_step(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
        check: bool,
    ) -> Dict[str, Any]:
        wrapper_stage = str(step.get("wrapper_stage", "")).strip()
        if wrapper_stage == "outer_loop":
            research_spec_payload = self._prepare_research_spec(context=context)
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.teacher_training.build_outer_loop_args(
                    research_goal=context.research_goal,
                    run_tag_base=context.run_label,
                ),
                stage_specific_env={
                    **self._base_clean_env(),
                    **self._teacher_gpu_env(),
                    "QA_STUDIO_RESEARCH_SPEC_JSON": context.research_spec_json,
                },
                check=check,
            )
            result["agent_summary"] = self.chief.teacher_training.summarize_outputs(
                shared_context_root=str(context.shared_context_root),
                research_spec_json=context.research_spec_json,
            )
            result["research_spec"] = research_spec_payload
            context.agent_payloads["TeacherTrainingAgent"] = dict(result["agent_summary"])
            return result

        if wrapper_stage == "explainability_report_v2":
            if not self.chief.explainability.has_teacher_reports(shared_context_root=str(context.shared_context_root)):
                context.step_status_by_stage[wrapper_stage] = "skipped_no_teacher_reports"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_no_teacher_reports",
                    "message": "No teacher_loop reports were found in the shared workflow context.",
                }
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.explainability.build_refresh_args(
                    shared_context_root=str(context.shared_context_root),
                    verification_review=context.agent_payloads.get("VerificationAgent", {}),
                ),
                stage_specific_env=self._base_clean_env(),
                check=check,
            )
            result["agent_summary"] = self.chief.explainability.summarize_outputs(
                shared_context_root=str(context.shared_context_root)
            )
            context.agent_payloads["ExplainabilityAgent"] = dict(result["agent_summary"])
            return result

        if wrapper_stage == "teacher_frozen_eval":
            if int(context.agent_payloads.get("VerificationAgent", {}).get("likely_teacher_count", 0) or 0) <= 0:
                context.step_status_by_stage[wrapper_stage] = "skipped_no_likely_teachers"
                result = {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_no_likely_teachers",
                    "message": "No current-workflow likely teachers were available for frozen evaluation.",
                }
                context.agent_payloads["TeacherFrozenEval"] = {
                    "frozen_eval_available": False,
                    "frozen_teacher_count": 0,
                    "teachers": [],
                    "skip_reason": "no_current_workflow_likely_teachers",
                }
                return result
            summary_name = f"{context.run_label}_frozen_eval"
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.acceptance.build_teacher_frozen_eval_args(
                    verification_review=context.agent_payloads.get("VerificationAgent", {}),
                    shared_context_root=str(context.shared_context_root),
                    fallback_selection_json=context.resolved_selection_json,
                    summary_name=summary_name,
                ),
                stage_specific_env={
                    **self._base_clean_env(),
                    **self._teacher_gpu_env(),
                },
                check=check,
            )
            result["agent_summary"] = self.chief.acceptance.summarize_frozen_eval_outputs(
                shared_context_root=str(context.shared_context_root),
                summary_name=summary_name,
            )
            context.agent_payloads["TeacherFrozenEval"] = dict(result["agent_summary"])
            return result

        if wrapper_stage == "inner_loop_suite":
            selection_json = str(context.resolved_selection_json or "").strip()
            if not selection_json or not Path(selection_json).expanduser().exists():
                context.step_status_by_stage[wrapper_stage] = "skipped_no_selection_json"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_no_selection_json",
                    "message": "No selected teacher set was available for inner-loop warmup.",
                }
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.apprentice.build_inner_loop_suite_args(
                    selection_json=context.resolved_selection_json,
                    run_tag_base=context.run_label,
                    api_model=context.resolved_api_model,
                ),
                stage_specific_env=self._selection_runtime_env(context=context),
                check=check,
            )
            agent_summary = self.chief.apprentice.summarize_outputs(
                shared_context_root=str(context.shared_context_root),
                run_tag_base=context.run_label,
            )
            context.generated_final_lesson_artifact_json = str(agent_summary.get("final_lesson_artifact_json", "")).strip()
            result["agent_summary"] = agent_summary
            context.agent_payloads["ApprenticeAgent"] = dict(agent_summary)
            return result

        if wrapper_stage == "scope_alignment":
            if context.step_status_by_stage.get("inner_loop_suite") == "failed":
                context.step_status_by_stage[wrapper_stage] = "skipped_previous_failure"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_previous_failure",
                    "message": "inner_loop_suite failed, so scope_alignment was not executed.",
                }
            lesson_artifact_json = context.generated_final_lesson_artifact_json or context.final_lesson_state_json
            if not lesson_artifact_json:
                context.step_status_by_stage[wrapper_stage] = "skipped_no_final_lesson_artifact"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_no_final_lesson_artifact",
                    "message": "No final lesson artifact was found under the shared workflow context.",
                }
            run_tag = f"{context.run_label}_scope_alignment"
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.evaluation.build_scope_alignment_args(
                    selection_json=context.resolved_selection_json,
                    run_tag_base=run_tag,
                    api_model=context.resolved_api_model,
                    lesson_artifact_json=lesson_artifact_json,
                ),
                stage_specific_env=self._selection_runtime_env(context=context),
                check=check,
            )
            result["agent_summary"] = self.chief.evaluation.summarize_scope_alignment_outputs(
                shared_context_root=str(context.shared_context_root),
                run_tag_base=run_tag,
            )
            context.agent_payloads["EvaluationAgent_scope_alignment"] = dict(result["agent_summary"])
            return result

        if wrapper_stage == "market_backtest":
            if str(context.global_env.get("QA_STUDIO_SKIP_MARKET_BACKTEST", "")).strip().lower() in {"1", "true", "yes"}:
                context.step_status_by_stage[wrapper_stage] = "skipped_by_studio_preset"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_by_studio_preset",
                    "message": "Skipped by Studio workflow preset because this demo only needs outer-loop and inner-loop artifact generation.",
                }
            lesson_state_json = context.generated_final_lesson_artifact_json or context.final_lesson_state_json
            if not lesson_state_json:
                context.step_status_by_stage[wrapper_stage] = "skipped_no_final_lesson_artifact"
                return {
                    "step_id": str(step.get("step_id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "owner": str(step.get("owner", "")).strip(),
                    "stage_type": str(step.get("stage_type", "")).strip(),
                    "wrapper_stage": wrapper_stage,
                    "status": "skipped_no_final_lesson_artifact",
                    "message": "No final lesson artifact was available for market_backtest.",
                }
            run_tag = f"{context.run_label}_market_backtest"
            result = self._run_clean_stage(
                step=step,
                context=context,
                wrapper_stage=wrapper_stage,
                default_stage_args=self.chief.evaluation.build_market_backtest_args(
                    selection_json=context.resolved_selection_json,
                    final_lesson_state_json=lesson_state_json,
                    run_tag=run_tag,
                    api_model=context.resolved_api_model,
                ),
                stage_specific_env=self._selection_runtime_env(context=context),
                check=check,
            )
            result["agent_summary"] = self.chief.evaluation.summarize_market_backtest_outputs(
                shared_context_root=str(context.shared_context_root),
                run_tag=run_tag,
            )
            context.agent_payloads["EvaluationAgent_market_backtest"] = dict(result["agent_summary"])
            return result

        return self._run_clean_stage(
            step=step,
            context=context,
            wrapper_stage=wrapper_stage,
            stage_specific_env=self._base_clean_env(),
            check=check,
        )

    def _execute_studio_runtime_step(
        self,
        *,
        step: Mapping[str, Any],
        context: WorkflowExecutionContext,
    ) -> Dict[str, Any]:
        owner = str(step.get("owner", "")).strip()
        title = str(step.get("title", "")).strip().lower()
        if owner == "LessonAgent":
            payload = self.chief.lesson.resolve_runtime_lesson(
                lesson_alias=context.lesson_alias,
                final_lesson_state_json=context.generated_final_lesson_artifact_json or context.final_lesson_state_json,
            )
            return self._completed_step(step, payload=payload)
        if owner == "SignalScoringAgent":
            payload = self.chief.scoring.runtime_activation_summary(
                lesson_alias=context.lesson_alias,
                final_lesson_state_json=context.generated_final_lesson_artifact_json or context.final_lesson_state_json,
            )
            if "validation" in title:
                payload["signal_schema"] = self.chief.scoring.describe_signal_schema()
            return self._completed_step(step, payload=payload)
        return self._completed_step(step, payload={})
