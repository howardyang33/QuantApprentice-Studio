from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..paths import clean_root, studio_root
from ..provenance import write_json


@dataclass(frozen=True)
class CleanStageSpec:
    key: str
    title: str
    mode: str
    description: str
    module: str = ""
    script_relpath: str = ""
    callable_hint: str = ""
    outputs: List[str] = field(default_factory=list)
    example_args: List[str] = field(default_factory=list)


STAGE_SPECS: Dict[str, CleanStageSpec] = {
    "outer_loop": CleanStageSpec(
        key="outer_loop",
        title="Outer Loop Teacher Construction",
        mode="outer_only",
        description=(
            "Launch one or more autonomous teacher-construction rounds, including hypothesis proposal, "
            "factor registration, teacher training, walk-forward validation, explainability generation, "
            "and teacher-zoo writeback."
        ),
        module="quant_toolkit.teacher_loop.run_autonomous_teacher_loop",
        callable_hint="quant_toolkit.teacher_loop.loop.launch_until_target(max_new_rounds=...)",
        outputs=[
            "research_memory/indexes/teacher_loop_manifest.jsonl",
            "reports/teacher_loop/<round_id>/selected_spec.json",
            "reports/teacher_loop/<round_id>/factor_analysis_summary.json",
            "research_memory/artifacts/teacher_loop/<round_id>/",
        ],
        example_args=[],
    ),
    "explainability_report_v2": CleanStageSpec(
        key="explainability_report_v2",
        title="Rebuild Teacher Explainability Report v2",
        mode="outer_only",
        description="Rebuild branch-oriented teacher explainability reports in place from existing CSV explainability artifacts.",
        script_relpath="scripts/rebuild_teacher_factor_reports_v2.py",
        callable_hint="scripts/rebuild_teacher_factor_reports_v2.py main()",
        outputs=[
            "reports/teacher_loop/<round_id>/factor_analysis_summary_v2.json",
            "reports/teacher_loop/<round_id>/branch_rule_cards.json",
        ],
        example_args=["--help"],
    ),
    "teacher_frozen_eval": CleanStageSpec(
        key="teacher_frozen_eval",
        title="Teacher Frozen Evaluation and Formal Selection",
        mode="outer_only",
        description=(
            "Freeze current-workflow candidate teachers at a train cutoff, evaluate them on the post-cutoff window, "
            "and emit a formal selection.json for inner-loop intake."
        ),
        module="quant_toolkit.examples.run_teacher_frozen_eval",
        callable_hint="quant_toolkit.examples.run_teacher_frozen_eval.main()",
        outputs=[
            "reports/teacher_loop/<summary_name>/selection.json",
            "reports/teacher_loop/<summary_name>/frozen_post2022_summary.csv",
            "reports/teacher_loop/<summary_name>/SUMMARY.md",
            "reports/teacher_loop/<round_id>_frozen_<train_end_year>/",
        ],
        example_args=["--help"],
    ),
    "inner_loop_suite": CleanStageSpec(
        key="inner_loop_suite",
        title="Inner Loop Warmup + Alignment Suite",
        mode="inner_only",
        description=(
            "Run the packaged lesson-evolution suite, including warmup lesson generation, checkpoint selection, "
            "and alignment evaluation."
        ),
        module="quant_toolkit.examples.run_scorefit_scope_experiment_suite",
        callable_hint=(
            "quant_toolkit.apprentice_loop.run_multi_teacher_scoped_warmup(config) and "
            "quant_toolkit.apprentice_loop.run_multi_teacher_replay(config)"
        ),
        outputs=[
            "reports/apprentice_loop/<run_id>/",
            "reports/apprentice_loop/<run_id>/selected_final_lesson_*.json",
            "reports/apprentice_loop/<run_id>/*summary*.md",
        ],
        example_args=["--scorefit-variant", "v7_bestguard_explore_longbatch"],
    ),
    "scope_alignment": CleanStageSpec(
        key="scope_alignment",
        title="Scope Alignment Only",
        mode="inner_only",
        description="Run standalone scope-alignment evaluation from an existing teacher selection and optional final lesson state.",
        module="quant_toolkit.examples.run_scope_alignment_test",
        callable_hint="quant_toolkit.examples.run_scope_alignment_test.main()",
        outputs=[
            "reports/apprentice_loop/<run_id>/SUMMARY.md",
            "reports/apprentice_loop/<run_id>/llm_signal_scores.json",
            "reports/apprentice_loop/<run_id>/teacher_signal_scores.json",
        ],
        example_args=["--help"],
    ),
    "market_backtest": CleanStageSpec(
        key="market_backtest",
        title="Long-Range Market Backtest",
        mode="scoring_only",
        description="Run market-wide scorefit backtest from an existing teacher selection plus final lesson state.",
        module="quant_toolkit.examples.run_market_scorefit_backtest",
        callable_hint="quant_toolkit.examples.run_market_scorefit_backtest.main()",
        outputs=[
            "reports/apprentice_loop/<run_id>/summary.json",
            "reports/apprentice_loop/<run_id>/llm_signal_scores.json",
            "reports/apprentice_loop/<run_id>/llm_daily_nav.json",
        ],
        example_args=["--help"],
    ),
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _normalize_extra_args(extra_args: Optional[Sequence[str]]) -> List[str]:
    values = list(extra_args or [])
    if values and values[0] == "--":
        values = values[1:]
    return [str(x) for x in values]


def _env_value(value: str) -> str:
    return str(Path(value).expanduser().resolve()) if value else ""


class CleanPipelineWrapper:
    def __init__(self) -> None:
        self._clean_root = clean_root()
        self._studio_root = studio_root()

    @property
    def clean_repo_root(self) -> Path:
        return self._clean_root

    @property
    def studio_repo_root(self) -> Path:
        return self._studio_root

    def stage_map(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for spec in STAGE_SPECS.values():
            row = asdict(spec)
            row["clean_repo_root"] = str(self.clean_repo_root)
            row["studio_wrapper_default_root"] = str(self._base_run_root() / spec.key)
            rows.append(row)
        return rows

    def _base_run_root(self) -> Path:
        return self.studio_repo_root / "runs" / "clean_pipeline"

    def build_run_root(self, stage: str, run_label: str = "") -> Path:
        label = str(run_label).strip() or _timestamp()
        return self._base_run_root() / stage / label

    def build_env(
        self,
        *,
        run_root: Path,
        context_root: Optional[Path] = None,
        data_dir: str = "",
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        env = dict(os.environ)
        context = Path(context_root or run_root)
        clean_root_str = str(self.clean_repo_root)
        old_pythonpath = env.get("PYTHONPATH", "").strip()
        if old_pythonpath:
            env["PYTHONPATH"] = clean_root_str + os.pathsep + old_pythonpath
        else:
            env["PYTHONPATH"] = clean_root_str
        env["QUANT_PROJECT_ROOT"] = clean_root_str
        env["QUANT_MEMORY_DIR"] = str(context / "research_memory")
        env["TEACHER_LOOP_REPORT_ROOT"] = str(context / "reports" / "teacher_loop")
        env["APPRENTICE_REPORT_ROOT"] = str(context / "reports" / "apprentice_loop")
        env["TEACHER_LOOP_ARTIFACT_ROOT"] = str(context / "research_memory" / "artifacts" / "teacher_loop")
        env["TEACHER_LOOP_NAV_REPORT_ROOT"] = str(context / "reports" / "teacher_loop_nav_backtest")
        env["NAV_CURVE_OUTPUT_DIR"] = str(context / "reports" / "nav_curve_backtest")
        env["TEACHER_LOOP_CACHE_ROOT"] = str(context / "cache" / "teacher_loop")
        env["APPRENTICE_MASTER_CACHE_PATH"] = str(
            context / "cache" / "apprentice_loop" / "master_feature_label_studio.joblib"
        )
        env["APPRENTICE_REPLAY_BUNDLE_CACHE_ROOT"] = str(context / "cache" / "apprentice_loop" / "bundle_cache")
        if str(data_dir).strip():
            env["TEACHER_LOOP_DATA_DIR"] = _env_value(str(data_dir).strip())
        if extra_env:
            for key, value in extra_env.items():
                env[str(key)] = str(value)
        return env

    def build_command(self, stage: str, *, extra_args: Optional[Sequence[str]] = None) -> List[str]:
        if stage not in STAGE_SPECS:
            raise KeyError(f"unknown clean pipeline stage: {stage}")
        spec = STAGE_SPECS[stage]
        args = _normalize_extra_args(extra_args)
        if spec.module:
            return [sys.executable, "-m", spec.module, *args]
        if spec.script_relpath:
            return [sys.executable, str(self.clean_repo_root / spec.script_relpath), *args]
        raise ValueError(f"stage has no module or script: {stage}")

    def plan_stage(
        self,
        stage: str,
        *,
        extra_args: Optional[Sequence[str]] = None,
        run_label: str = "",
        context_root: str = "",
        data_dir: str = "",
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        if stage not in STAGE_SPECS:
            raise KeyError(f"unknown clean pipeline stage: {stage}")
        run_root = self.build_run_root(stage, run_label=run_label)
        resolved_context_root = Path(str(context_root).strip()).expanduser().resolve() if str(context_root).strip() else run_root
        env = self.build_env(
            run_root=run_root,
            context_root=resolved_context_root,
            data_dir=data_dir,
            extra_env=extra_env,
        )
        command = self.build_command(stage, extra_args=extra_args)
        spec = STAGE_SPECS[stage]
        interesting_env_keys = [
            "QUANT_PROJECT_ROOT",
            "TEACHER_LOOP_DATA_DIR",
            "QUANT_MEMORY_DIR",
            "TEACHER_LOOP_REPORT_ROOT",
            "APPRENTICE_REPORT_ROOT",
            "TEACHER_LOOP_ARTIFACT_ROOT",
            "TEACHER_LOOP_NAV_REPORT_ROOT",
            "NAV_CURVE_OUTPUT_DIR",
            "TEACHER_LOOP_CACHE_ROOT",
            "APPRENTICE_MASTER_CACHE_PATH",
            "APPRENTICE_REPLAY_BUNDLE_CACHE_ROOT",
            "QA_STUDIO_RESEARCH_SPEC_JSON",
        ]
        return {
            "stage": stage,
            "title": spec.title,
            "mode": spec.mode,
            "description": spec.description,
            "callable_hint": spec.callable_hint,
            "clean_repo_root": str(self.clean_repo_root),
            "cwd": str(self.clean_repo_root),
            "run_root": str(run_root),
            "context_root": str(resolved_context_root),
            "command": command,
            "command_shell": " ".join(shlex.quote(part) for part in command),
            "env": {key: env.get(key, "") for key in interesting_env_keys if env.get(key, "")},
            "outputs": list(spec.outputs),
            "example_args": list(spec.example_args),
        }

    def run_stage(
        self,
        stage: str,
        *,
        extra_args: Optional[Sequence[str]] = None,
        run_label: str = "",
        context_root: str = "",
        data_dir: str = "",
        extra_env: Optional[Mapping[str, str]] = None,
        check: bool = False,
    ) -> Dict[str, Any]:
        plan = self.plan_stage(
            stage,
            extra_args=extra_args,
            run_label=run_label,
            context_root=context_root,
            data_dir=data_dir,
            extra_env=extra_env,
        )
        run_root = Path(plan["run_root"])
        context_root_path = Path(plan["context_root"])
        run_root.mkdir(parents=True, exist_ok=True)
        context_root_path.mkdir(parents=True, exist_ok=True)
        write_json(run_root / "studio_wrapper_plan.json", plan)
        env = self.build_env(
            run_root=run_root,
            context_root=context_root_path,
            data_dir=data_dir,
            extra_env=extra_env,
        )
        command = list(plan["command"])
        started_at = datetime.now().isoformat(timespec="seconds")
        proc = subprocess.run(
            command,
            cwd=str(self.clean_repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = run_root / "studio_wrapper_stdout.log"
        stderr_path = run_root / "studio_wrapper_stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        result = {
            **plan,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "returncode": int(proc.returncode),
            "ok": proc.returncode == 0,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        write_json(run_root / "studio_wrapper_result.json", result)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"clean stage failed: {stage} rc={proc.returncode} stderr={stderr_path}"
            )
        return result
