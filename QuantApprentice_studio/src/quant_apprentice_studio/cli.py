from __future__ import annotations

import argparse
import json

from .local_service import describe_local_service_status, start_local_service, stop_local_service
from .agents.chief import ChiefResearchAgent
from .importers.gpt_oss_final import bootstrap_profile
from .orchestrator.pipeline import QuantPipelineOrchestrator
from .orchestrator.runner import WorkflowRunner
from .provenance import to_jsonable
from .registry import StudioRegistry
from .tools.clean_pipeline import CleanPipelineWrapper, STAGE_SPECS
from .workflows.gpt_oss_final import GPTOSSFinalWorkflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantApprentice Studio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap", help="Import the final GPT-OSS asset set into the studio workspace")
    p_boot.add_argument("--profile", default="gpt_oss_20b_final")
    p_boot.add_argument("--overwrite", action="store_true")

    p_overview = sub.add_parser("overview", help="Show imported runtime overview")
    p_overview.add_argument("--profile", default="gpt_oss_20b_final")

    p_teachers = sub.add_parser("teachers", help="List imported frozen teachers")
    p_teachers.add_argument("--profile", default="gpt_oss_20b_final")

    p_lessons = sub.add_parser("lessons", help="List imported final lesson runs")
    p_lessons.add_argument("--profile", default="gpt_oss_20b_final")
    p_lessons.add_argument("--alias", default="")

    p_alignment = sub.add_parser("alignment", help="Show archived GPT alignment result")
    p_alignment.add_argument("--profile", default="gpt_oss_20b_final")
    p_alignment.add_argument("--seed", default="Mean")

    p_market = sub.add_parser("market", help="Show archived GPT market summary")
    p_market.add_argument("--profile", default="gpt_oss_20b_final")
    p_market.add_argument("--run", required=True)

    p_score = sub.add_parser("score-recorded", help="Lookup a recorded single-signal score from an archived market run")
    p_score.add_argument("--profile", default="gpt_oss_20b_final")
    p_score.add_argument("--run", required=True)
    p_score.add_argument("--signal-date", required=True)
    p_score.add_argument("--symbol", required=True)

    p_live_cfg = sub.add_parser("live-config", help="Show current live runtime API/model configuration")
    p_live_cfg.add_argument("--profile", default="gpt_oss_20b_final")

    p_schema = sub.add_parser("signal-schema", help="Show the canonical single-signal input schema used by live scoring")
    p_schema.add_argument("--profile", default="gpt_oss_20b_final")
    p_schema.add_argument("--run", default="")
    p_schema.add_argument("--signal-date", default="")
    p_schema.add_argument("--symbol", default="")

    p_template = sub.add_parser(
        "signal-template",
        help="Export a canonical signal JSON template based on an archived signal record",
    )
    p_template.add_argument("--profile", default="gpt_oss_20b_final")
    p_template.add_argument("--run", default="")
    p_template.add_argument("--signal-date", default="")
    p_template.add_argument("--symbol", default="")
    p_template.add_argument("--output", required=True)

    sub.add_parser("local-model-status", help="Show local GPT-OSS vLLM service status")

    p_local_start = sub.add_parser("start-local-model", help="Start the local GPT-OSS vLLM service")
    p_local_start.add_argument("--force-restart", action="store_true")

    sub.add_parser("stop-local-model", help="Stop the local GPT-OSS vLLM service")

    p_live = sub.add_parser("score-live", help="Run live GPT-style scoring for one signal using the imported final lesson")
    p_live.add_argument("--profile", default="gpt_oss_20b_final")
    p_live.add_argument("--lesson-alias", default="alignment_seed0005")
    p_live.add_argument("--signal-json", default="")
    p_live.add_argument("--signal-inline", default="")
    p_live.add_argument("--from-recorded-run", default="")
    p_live.add_argument("--signal-date", default="")
    p_live.add_argument("--symbol", default="")
    p_live.add_argument("--prompt-only", action="store_true")
    p_live.add_argument("--no-cache", action="store_true")
    p_live.add_argument("--schema-from-run", default="")
    p_live.add_argument("--run-label", default="")
    p_live.add_argument("--no-persist-run", action="store_true")

    p_compare = sub.add_parser("compare-live", help="Compare one live score against its recorded archived score")
    p_compare.add_argument("--profile", default="gpt_oss_20b_final")
    p_compare.add_argument("--lesson-alias", default="alignment_seed0005")
    p_compare.add_argument("--run", required=True)
    p_compare.add_argument("--signal-date", required=True)
    p_compare.add_argument("--symbol", required=True)
    p_compare.add_argument("--prompt-only", action="store_true")
    p_compare.add_argument("--no-cache", action="store_true")

    p_live_batch = sub.add_parser(
        "score-live-batch",
        help="Run live scoring on a batch of archived signals and compare with recorded scores",
    )
    p_live_batch.add_argument("--profile", default="gpt_oss_20b_final")
    p_live_batch.add_argument("--lesson-alias", default="alignment_seed0005")
    p_live_batch.add_argument("--run", required=True)
    p_live_batch.add_argument("--limit", type=int, default=5)
    p_live_batch.add_argument("--offset", type=int, default=0)
    p_live_batch.add_argument("--signal-date", default="")
    p_live_batch.add_argument("--prompt-only", action="store_true")
    p_live_batch.add_argument("--no-cache", action="store_true")

    p_api = sub.add_parser("serve-api", help="Run the FastAPI backend server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8010)
    p_api.add_argument("--reload", action="store_true")

    p_probe = sub.add_parser("probe", help="Run the minimal archived GPT-OSS workflow probe")
    p_probe.add_argument("--profile", default="gpt_oss_20b_final")
    p_probe.add_argument("--lesson-alias", default="alignment_seed0005")
    p_probe.add_argument("--run", default="market_2025_lseed20250705")
    p_probe.add_argument("--signal-date", default="2025-01-02")
    p_probe.add_argument("--symbol", default="000151")

    sub.add_parser("clean-pipeline-map", help="Show the studio-to-clean pipeline stage mapping")

    p_clean_plan = sub.add_parser(
        "clean-stage-plan",
        help="Preview how the studio wrapper would invoke one QuantApprentice_clean stage",
    )
    p_clean_plan.add_argument("stage", choices=sorted(STAGE_SPECS.keys()))
    p_clean_plan.add_argument("--run-label", default="")
    p_clean_plan.add_argument("--data-dir", default="")
    p_clean_plan.add_argument("--env", action="append", default=[], help="Extra environment override, e.g. KEY=VALUE")

    p_clean_run = sub.add_parser(
        "clean-stage-run",
        help="Execute one QuantApprentice_clean stage through the studio wrapper",
    )
    p_clean_run.add_argument("stage", choices=sorted(STAGE_SPECS.keys()))
    p_clean_run.add_argument("--run-label", default="")
    p_clean_run.add_argument("--data-dir", default="")
    p_clean_run.add_argument("--env", action="append", default=[], help="Extra environment override, e.g. KEY=VALUE")
    p_clean_run.add_argument("--check", action="store_true")

    sub.add_parser("pipeline-modes", help="Describe the four system-level workflow modes")

    p_pipeline_plan = sub.add_parser("pipeline-plan", help="Build a studio-level workflow plan without executing it")
    p_pipeline_plan.add_argument("--mode", required=True, choices=["full_pipeline", "outer_loop_only", "inner_loop_only", "scoring_only"])
    p_pipeline_plan.add_argument("--research-goal", required=True)
    p_pipeline_plan.add_argument("--run-label", default="")
    p_pipeline_plan.add_argument("--selection-json", default="")
    p_pipeline_plan.add_argument("--final-lesson-state-json", default="")
    p_pipeline_plan.add_argument("--lesson-alias", default="alignment_seed0005")
    p_pipeline_plan.add_argument("--api-model", default="gpt-oss-20b")

    p_workflow_run = sub.add_parser("workflow-run", help="Execute a studio workflow mode through wrapper-driven steps")
    p_workflow_run.add_argument("--profile", default="gpt_oss_20b_final")
    p_workflow_run.add_argument("--mode", required=True, choices=["full_pipeline", "outer_loop_only", "inner_loop_only", "scoring_only"])
    p_workflow_run.add_argument("--research-goal", required=True)
    p_workflow_run.add_argument("--run-label", default="")
    p_workflow_run.add_argument("--selection-json", default="")
    p_workflow_run.add_argument("--final-lesson-state-json", default="")
    p_workflow_run.add_argument("--lesson-alias", default="alignment_seed0005")
    p_workflow_run.add_argument("--api-model", default="gpt-oss-20b")
    p_workflow_run.add_argument("--data-dir", default="")
    p_workflow_run.add_argument("--env", action="append", default=[], help="Global environment override, e.g. KEY=VALUE")
    p_workflow_run.add_argument(
        "--stage-args-json",
        default="{}",
        help='JSON mapping from wrapper stage to argv list, e.g. {"scope_alignment":["--help"]}',
    )
    p_workflow_run.add_argument(
        "--stage-env-json",
        default="{}",
        help='JSON mapping from wrapper stage to env override map, e.g. {"outer_loop":{"TEACHER_LOOP_MAX_ROUNDS":"0"}}',
    )
    p_workflow_run.add_argument("--check", action="store_true")
    p_workflow_run.add_argument("--no-manual-steps", action="store_true")

    return parser


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"invalid --env override, expected KEY=VALUE, got: {text}")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid --env override with empty key: {text}")
        out[key] = value
    return out


def _normalize_passthrough_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return list(values)


def _parse_json_object(raw: str, *, field_name: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {field_name} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{field_name} must decode to a JSON object")
    return payload


def _print_json(payload) -> None:
    print(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))


def main() -> None:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()

    if unknown and args.command not in {"clean-stage-plan", "clean-stage-run"}:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.command == "clean-pipeline-map":
        wrapper = CleanPipelineWrapper()
        _print_json(wrapper.stage_map())
        return

    if args.command == "clean-stage-plan":
        wrapper = CleanPipelineWrapper()
        payload = wrapper.plan_stage(
            args.stage,
            extra_args=_normalize_passthrough_args(unknown),
            run_label=args.run_label,
            data_dir=args.data_dir,
            extra_env=_parse_env_overrides(args.env),
        )
        _print_json(payload)
        return

    if args.command == "clean-stage-run":
        wrapper = CleanPipelineWrapper()
        payload = wrapper.run_stage(
            args.stage,
            extra_args=_normalize_passthrough_args(unknown),
            run_label=args.run_label,
            data_dir=args.data_dir,
            extra_env=_parse_env_overrides(args.env),
            check=bool(args.check),
        )
        _print_json(payload)
        return

    if args.command == "pipeline-modes":
        orchestrator = QuantPipelineOrchestrator()
        _print_json(orchestrator.describe_modes())
        return

    if args.command == "pipeline-plan":
        orchestrator = QuantPipelineOrchestrator()
        kwargs = {}
        if args.mode == "full_pipeline":
            kwargs = {
                "selection_json_hint": args.selection_json,
                "final_lesson_state_hint": args.final_lesson_state_json,
            }
        elif args.mode == "inner_loop_only":
            kwargs = {
                "selection_json": args.selection_json,
                "api_model": args.api_model,
            }
        elif args.mode == "scoring_only":
            kwargs = {
                "lesson_alias": args.lesson_alias,
            }
        payload = orchestrator.build_plan(
            mode=args.mode,
            research_goal=args.research_goal,
            run_label=args.run_label or args.mode,
            **kwargs,
        )
        _print_json(payload)
        return

    if args.command == "workflow-run":
        runner = WorkflowRunner(profile_id=args.profile)
        payload = runner.run(
            mode=args.mode,
            research_goal=args.research_goal,
            run_label=args.run_label or args.mode,
            selection_json=args.selection_json,
            final_lesson_state_json=args.final_lesson_state_json,
            lesson_alias=args.lesson_alias,
            api_model=args.api_model,
            data_dir=args.data_dir,
            global_env=_parse_env_overrides(args.env),
            stage_args=_parse_json_object(args.stage_args_json, field_name="stage-args-json"),
            stage_env=_parse_json_object(args.stage_env_json, field_name="stage-env-json"),
            check=bool(args.check),
            allow_manual_steps=not bool(args.no_manual_steps),
        )
        _print_json(payload)
        return

    if args.command == "bootstrap":
        payload = bootstrap_profile(args.profile, overwrite=args.overwrite)
        _print_json(payload)
        return

    if args.command == "serve-api":
        import uvicorn

        uvicorn.run(
            "quant_apprentice_studio.api.app:app",
            host=args.host,
            port=args.port,
            reload=bool(args.reload),
        )
        return

    if args.command == "local-model-status":
        _print_json(describe_local_service_status())
        return

    if args.command == "start-local-model":
        _print_json(start_local_service(force_restart=bool(args.force_restart)))
        return

    if args.command == "stop-local-model":
        _print_json(stop_local_service())
        return

    registry = StudioRegistry(args.profile)
    chief = ChiefResearchAgent(registry)

    if args.command == "overview":
        _print_json(chief.overview())
        return

    if args.command == "teachers":
        payload = [item.__dict__ for item in chief.teacher_zoo.list_teachers()]
        _print_json(payload)
        return

    if args.command == "lessons":
        if args.alias:
            payload = chief.lesson.summarize_lesson_run(args.alias)
        else:
            payload = [item.__dict__ for item in chief.lesson.list_lesson_runs()]
        _print_json(payload)
        return

    if args.command == "alignment":
        payload = chief.alignment.get_after_warmup_result(args.seed)
        _print_json(payload)
        return

    if args.command == "market":
        payload = chief.backtest.load_market_summary(args.run)
        _print_json(payload)
        return

    if args.command == "score-recorded":
        payload = chief.scoring.score_recorded(args.run, args.signal_date, args.symbol).__dict__
        _print_json(payload)
        return

    if args.command == "live-config":
        payload = chief.scoring.live_config_status()
        _print_json(payload)
        return

    if args.command == "signal-schema":
        payload = chief.scoring.describe_signal_schema(
            market_run_alias=args.run,
            signal_date=args.signal_date,
            symbol=args.symbol,
        )
        _print_json(payload)
        return

    if args.command == "signal-template":
        payload = chief.scoring.write_signal_template(
            args.output,
            market_run_alias=args.run,
            signal_date=args.signal_date,
            symbol=args.symbol,
        )
        _print_json(payload)
        return

    if args.command == "score-live":
        if args.signal_json:
            with open(args.signal_json, "r", encoding="utf-8") as f:
                signal_record = json.load(f)
        elif args.signal_inline:
            signal_record = json.loads(args.signal_inline)
        elif args.from_recorded_run and args.signal_date and args.symbol:
            signal_record = chief.scoring.build_signal_record_from_recorded(
                args.from_recorded_run, args.signal_date, args.symbol
            )
        else:
            raise SystemExit(
                "Provide one of: --signal-json PATH, --signal-inline JSON, or "
                "--from-recorded-run RUN --signal-date YYYY-MM-DD --symbol 000001"
            )
        payload = chief.scoring.score_live(
            lesson_alias=args.lesson_alias,
            signal_record=signal_record,
            prompt_only=bool(args.prompt_only),
            reuse_cache=not bool(args.no_cache),
            persist_run=not bool(args.no_persist_run),
            run_label=args.run_label,
            source_tag="recorded_reference" if args.from_recorded_run else "external_signal",
            schema_market_run_alias=args.schema_from_run or args.from_recorded_run,
        )
        _print_json(payload)
        return

    if args.command == "compare-live":
        payload = chief.scoring.compare_live_to_recorded(
            lesson_alias=args.lesson_alias,
            market_run_alias=args.run,
            signal_date=args.signal_date,
            symbol=args.symbol,
            prompt_only=bool(args.prompt_only),
            reuse_cache=not bool(args.no_cache),
        )
        _print_json(payload)
        return

    if args.command == "score-live-batch":
        payload = chief.scoring.score_live_batch_from_recorded(
            lesson_alias=args.lesson_alias,
            market_run_alias=args.run,
            limit=args.limit,
            offset=args.offset,
            signal_date=args.signal_date,
            prompt_only=bool(args.prompt_only),
            reuse_cache=not bool(args.no_cache),
        )
        _print_json(payload)
        return

    if args.command == "probe":
        workflow = GPTOSSFinalWorkflow(args.profile)
        payload = workflow.runtime_probe(
            lesson_alias=args.lesson_alias,
            market_run_alias=args.run,
            signal_date=args.signal_date,
            symbol=args.symbol,
        )
        _print_json(payload)
        return


if __name__ == "__main__":
    main()
