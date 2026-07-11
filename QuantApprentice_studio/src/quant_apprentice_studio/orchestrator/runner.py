from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..agents.chief import ChiefResearchAgent
from ..contracts import build_run_contract, ensure_contract_dirs, save_research_campaign, save_run_spec
from ..llm.backend import StudioLLMBackend
from ..paths import studio_root
from ..provenance import write_json
from ..registry import StudioRegistry
from ..tools.clean_pipeline import CleanPipelineWrapper
from .pipeline import QuantPipelineOrchestrator
from .step_executor import WorkflowExecutionContext, WorkflowStepExecutor


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _normalize_stage_args_map(payload: Optional[Mapping[str, Sequence[str] | str]]) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    for key, value in dict(payload or {}).items():
        stage = str(key)
        if isinstance(value, (list, tuple)):
            out[stage] = [str(item) for item in value]
        elif value is None:
            out[stage] = []
        else:
            out[stage] = [str(value)]
    return out


class WorkflowRunner:
    """Execute studio workflow modes through wrapper-friendly steps.

    This runner intentionally keeps the historical research implementation in
    ``QuantApprentice_clean`` untouched. It only stitches together studio-level
    planning metadata plus wrapper-executed clean stages.
    """

    def __init__(self, profile_id: str = "gpt_oss_20b_final") -> None:
        self.profile_id = profile_id
        self.registry = StudioRegistry(profile_id)
        self.chief = ChiefResearchAgent(self.registry)
        self.orchestrator = QuantPipelineOrchestrator()
        self.clean = CleanPipelineWrapper()
        self.llm = StudioLLMBackend()
        self.executor = WorkflowStepExecutor(chief=self.chief, clean=self.clean)

    def _workflow_root(self, run_label: str) -> Path:
        label = str(run_label).strip() or f"workflow_{_timestamp()}"
        return studio_root() / "runs" / "workflows" / label

    def _build_plan(
        self,
        *,
        mode: str,
        research_goal: str,
        run_label: str,
        selection_json: str = "",
        final_lesson_state_json: str = "",
        lesson_alias: str = "alignment_seed0005",
        api_model: str = "gpt-oss-20b",
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if mode == "full_pipeline":
            kwargs = {
                "selection_json_hint": selection_json,
                "final_lesson_state_hint": final_lesson_state_json,
            }
        elif mode == "inner_loop_only":
            kwargs = {
                "selection_json": selection_json,
                "api_model": api_model,
            }
        elif mode == "scoring_only":
            kwargs = {
                "lesson_alias": lesson_alias,
            }
        return self.orchestrator.build_plan(
            mode=mode,
            research_goal=research_goal,
            run_label=run_label,
            **kwargs,
        )

    def run(
        self,
        *,
        mode: str,
        research_goal: str,
        run_label: str = "",
        project_id: str = "default-project",
        dataset_id: str = "default-dataset",
        selection_json: str = "",
        final_lesson_state_json: str = "",
        lesson_alias: str = "alignment_seed0005",
        api_model: str = "gpt-oss-20b",
        allow_imported_fallback: bool = True,
        allow_demo_fallback: bool = False,
        data_dir: str = "",
        global_env: Optional[Mapping[str, str]] = None,
        stage_args: Optional[Mapping[str, Sequence[str]]] = None,
        stage_env: Optional[Mapping[str, Mapping[str, str]]] = None,
        check: bool = False,
        allow_manual_steps: bool = True,
    ) -> Dict[str, Any]:
        effective_label = str(run_label).strip() or f"{mode}_{_timestamp()}"
        contract = build_run_contract(
            profile_id=self.profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=effective_label,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
        )
        ensure_contract_dirs(contract)
        llm_status = self.llm.ensure_ready()
        selection_resolution = self.chief.selection.resolve_workflow_selection_json(selection_json)
        resolved_selection_json = str(selection_resolution["selection_json"]).strip()
        resolved_api_model = str(api_model).strip() or self.chief.memory.resolve_default_api_model()
        planning_brief = self.chief.planner.build_research_brief(research_goal=research_goal, mode=mode)
        selection_shared_context_root = self.chief.memory.resolve_workflow_shared_context_root(resolved_selection_json)
        teacher_selection_summary = self.chief.selection.summarize_selection(
            resolved_selection_json,
            shared_context_root=selection_shared_context_root,
        )
        teacher_selection_summary["resolution_source"] = str(selection_resolution.get("resolution_source", "")).strip()
        plan = self._build_plan(
            mode=mode,
            research_goal=research_goal,
            run_label=effective_label,
            selection_json=resolved_selection_json,
            final_lesson_state_json=final_lesson_state_json,
            lesson_alias=lesson_alias,
            api_model=resolved_api_model,
        )
        workflow_root = Path(contract["workflow_root"]).expanduser().resolve()
        workflow_root.mkdir(parents=True, exist_ok=True)
        shared_context_root = Path(contract["shared_context_root"]).expanduser().resolve()
        shared_context_root.mkdir(parents=True, exist_ok=True)
        write_json(workflow_root / "planning_brief.json", planning_brief)
        write_json(workflow_root / "teacher_selection_summary.json", teacher_selection_summary)
        write_json(workflow_root / "workflow_plan.json", plan)
        save_run_spec(
            contract,
            {
                **contract,
                "profile_id": self.profile_id,
                "mode": mode,
                "project_id": contract["project_id"],
                "dataset_id": contract["dataset_id"],
                "run_id": contract["run_id"],
                "research_goal": research_goal,
                "selection_json_hint": selection_json,
                "resolved_selection_json_initial": resolved_selection_json,
                "final_lesson_state_json_hint": final_lesson_state_json,
                "lesson_alias": lesson_alias,
                "api_model": resolved_api_model,
                "data_dir": str(data_dir).strip(),
                "global_env": {str(k): str(v) for k, v in dict(global_env or {}).items()},
                "stage_args": {str(k): list(v) if isinstance(v, (list, tuple)) else [str(v)] for k, v in dict(stage_args or {}).items()},
                "stage_env": {
                    str(stage): {str(k): str(v) for k, v in dict(env_map).items()}
                    for stage, env_map in dict(stage_env or {}).items()
                },
                "allow_imported_fallback": bool(allow_imported_fallback),
                "allow_demo_fallback": bool(allow_demo_fallback),
            },
        )
        save_research_campaign(
            contract,
            {
                "profile_id": self.profile_id,
                "project_id": contract["project_id"],
                "dataset_id": contract["dataset_id"],
                "run_id": contract["run_id"],
                "research_goal": research_goal,
                "mode": mode,
                "pipeline_plan": plan,
                "data_dir": str(data_dir).strip(),
                "global_env": {str(k): str(v) for k, v in dict(global_env or {}).items()},
                "stage_args": {str(k): list(v) if isinstance(v, (list, tuple)) else [str(v)] for k, v in dict(stage_args or {}).items()},
                "stage_env": {
                    str(stage): {str(k): str(v) for k, v in dict(env_map).items()}
                    for stage, env_map in dict(stage_env or {}).items()
                },
                "policy": {
                    "allow_imported_fallback": bool(allow_imported_fallback),
                    "allow_demo_fallback": bool(allow_demo_fallback),
                },
            },
        )

        global_env_map = dict(self.llm.clean_env_overrides())
        global_env_map.update({str(k): str(v) for k, v in dict(global_env or {}).items()})
        stage_args_map = _normalize_stage_args_map(stage_args)
        stage_env_map = {
            str(stage): {str(k): str(v) for k, v in dict(env_map).items()}
            for stage, env_map in dict(stage_env or {}).items()
        }
        exec_context = WorkflowExecutionContext(
            mode=mode,
            research_goal=research_goal,
            run_label=effective_label,
            workflow_root=workflow_root,
            shared_context_root=shared_context_root,
            resolved_selection_json=resolved_selection_json,
            resolved_api_model=resolved_api_model,
            lesson_alias=str(lesson_alias).strip(),
            final_lesson_state_json=str(final_lesson_state_json).strip(),
            data_dir=str(data_dir).strip(),
            selection_resolution=dict(selection_resolution),
            planning_brief=planning_brief,
            teacher_selection_summary=teacher_selection_summary,
            run_contract=dict(contract),
            project_id=str(contract["project_id"]),
            dataset_id=str(contract["dataset_id"]),
            allow_imported_fallback=bool(allow_imported_fallback),
            allow_demo_fallback=bool(allow_demo_fallback),
            global_env=global_env_map,
            stage_args=stage_args_map,
            stage_env=stage_env_map,
        )

        step_results = []
        manual_steps = 0
        failed_steps = 0
        executed_steps = 0

        for step in plan.get("steps", []):
            step_result = self.executor.execute(
                step=step,
                context=exec_context,
                check=check,
                allow_manual_steps=allow_manual_steps,
            )
            step_results.append(step_result)
            if str(step_result.get("stage_type", "")).strip() == "clean_wrapper" and str(step_result.get("status", "")).strip() in {
                "completed",
                "failed",
            }:
                executed_steps += 1
            if str(step_result.get("status", "")).strip() == "failed":
                failed_steps += 1
            if str(step_result.get("status", "")).strip() == "manual_pending":
                manual_steps += 1

        overall_status = "completed"
        if failed_steps:
            overall_status = "failed"
        elif manual_steps:
            overall_status = "partial"

        write_json(workflow_root / "teacher_selection_summary_final.json", exec_context.teacher_selection_summary)

        result = {
            "mode": mode,
            "research_goal": research_goal,
            "run_label": effective_label,
            "project_id": str(contract["project_id"]),
            "dataset_id": str(contract["dataset_id"]),
            "run_id": str(contract["run_id"]),
            "run_contract": dict(contract),
            "workflow_root": str(workflow_root),
            "shared_context_root": str(shared_context_root),
            "profile_id": self.profile_id,
            "selection_resolution": exec_context.selection_resolution,
            "resolved_selection_json": exec_context.resolved_selection_json,
            "teacher_selection_summary": exec_context.teacher_selection_summary,
            "resolved_api_model": resolved_api_model,
            "llm_backend": self.llm.describe(),
            "llm_ready_status": llm_status,
            "status": overall_status,
            "executed_steps": executed_steps,
            "manual_steps": manual_steps,
            "failed_steps": failed_steps,
            "plan_path": str(workflow_root / "workflow_plan.json"),
            "steps": step_results,
        }
        write_json(workflow_root / "workflow_result.json", result)
        return result
