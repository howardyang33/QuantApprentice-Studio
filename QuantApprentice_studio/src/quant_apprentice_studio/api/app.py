from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..agents.chief import ChiefResearchAgent
from ..console_views import (
    build_lesson_set_view,
    build_project_view,
    build_provenance_view,
    build_run_monitor,
    build_teacher_zoo_view,
    create_run_spec,
)
from ..data_jobs import KlineDownloadJobManager
from ..guided_entry import (
    analyze_task_intake,
    build_dataset_manifest,
    build_imported_asset_manifest,
    create_guided_run_bundle,
    dataset_requirements,
)
from ..contracts import build_run_contract, dataset_manifest_path, list_run_specs
from ..local_service import describe_local_service_status, start_local_service, stop_local_service
from ..orchestrator.pipeline import QuantPipelineOrchestrator
from ..orchestrator.runner import WorkflowRunner
from ..paths import import_root, studio_root
from ..registry import StudioRegistry
from ..provenance import read_json
from ..simple_chat import build_chat_run_status, handle_chat_action, handle_chat_message
from ..teacher_libraries import build_teacher_library_registry
from ..tools.clean_pipeline import CleanPipelineWrapper
from .models import (
    ChatActionRequest,
    ChatMessageRequest,
    CleanStagePlanRequest,
    CleanStageRunRequest,
    CompareLiveRequest,
    DatasetOnboardingRequest,
    GuidedRunWizardRequest,
    ImportedAssetManifestRequest,
    KlineDownloadStartRequest,
    PipelinePlanRequest,
    PipelineRunRequest,
    RunSpecRequest,
    ScoreLiveExternalBatchRequest,
    ScoreLiveBatchRequest,
    ScoreLiveRequest,
    TaskIntakeRequest,
)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@lru_cache(maxsize=8)
def _chief_for_profile(profile: str) -> ChiefResearchAgent:
    registry = StudioRegistry(profile)
    registry.ensure_bootstrapped()
    return ChiefResearchAgent(registry)


def _resolve_signal_record(chief: ChiefResearchAgent, request: ScoreLiveRequest) -> Dict[str, Any]:
    if request.signal_record is not None:
        return _normalize_signal_date_alias(dict(request.signal_record))
    return chief.scoring.build_signal_record_from_recorded(
        request.from_recorded_run,
        request.signal_date,
        request.symbol,
    )


def _normalize_signal_date_alias(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record or {})
    if not normalized.get("signal_date") and normalized.get("date"):
        normalized["signal_date"] = normalized.get("date")
        normalized["_studio_normalization_note"] = "date was accepted as an alias and normalized to signal_date."
    return normalized


@lru_cache(maxsize=1)
def _kline_job_manager() -> KlineDownloadJobManager:
    return KlineDownloadJobManager()


def create_app() -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="QuantApprentice Studio API",
        version="0.1.0",
        description="Backend API for archived QuantApprentice runtime browsing and live signal scoring.",
    )
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "quant_apprentice_studio_api",
            "docs": "/docs",
            "health": "/health",
            "app": "/app",
            "default_profile": "gpt_oss_20b_final",
        }

    @app.get("/app")
    def app_shell() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/overview")
    def overview(profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            return _chief_for_profile(profile).overview()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/teacher-libraries")
    def teacher_libraries(
        profile: str = Query(default="gpt_oss_20b_final"),
        project_id: str = Query(default=""),
        dataset_id: str = Query(default=""),
        run_id: str = Query(default=""),
    ) -> Dict[str, Any]:
        try:
            return build_teacher_library_registry(
                chief=_chief_for_profile(profile),
                profile_id=profile,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/project")
    def console_project(
        profile: str = Query(default="gpt_oss_20b_final"),
        project_id: str = Query(default="default-project"),
        dataset_id: str = Query(default="default-dataset"),
        run_id: str = Query(default="draft-run"),
        allow_imported_fallback: bool = Query(default=True),
        allow_demo_fallback: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            return build_project_view(
                profile_id=profile,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/runs")
    def console_runs() -> Dict[str, Any]:
        try:
            return {"items": list_run_specs()}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/dataset-manifest")
    def console_dataset_manifest(
        profile: str = Query(default="gpt_oss_20b_final"),
        project_id: str = Query(default="default-project"),
        dataset_id: str = Query(default="default-dataset"),
        run_id: str = Query(default="draft-run"),
        allow_imported_fallback: bool = Query(default=True),
        allow_demo_fallback: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            contract = build_run_contract(
                profile_id=profile,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
            )
            path = dataset_manifest_path(contract)
            if not path.exists():
                return {
                    "exists": False,
                    "dataset_manifest_json": str(path),
                    "project_id": contract["project_id"],
                    "dataset_id": contract["dataset_id"],
                    "run_id": contract["run_id"],
                    "data_isolation_status": contract["data_isolation"],
                }
            payload = read_json(path)
            if isinstance(payload, dict):
                payload.setdefault("exists", True)
                payload.setdefault("dataset_manifest_json", str(path))
                payload.setdefault("data_isolation_status", contract["data_isolation"])
                return payload
            return {"exists": True, "dataset_manifest_json": str(path), "payload": payload}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/artifact-json")
    def console_artifact_json(path: str = Query(...)) -> Dict[str, Any]:
        try:
            target = Path(path).expanduser().resolve()
            allowed_roots = [studio_root().resolve(), import_root("gpt_oss_20b_final").resolve()]
            if not any(root == target or root in target.parents for root in allowed_roots):
                raise ValueError("artifact path is outside allowed Studio/import roots")
            if target.suffix.lower() != ".json":
                raise ValueError("only JSON artifacts can be opened through this endpoint")
            return {"path": str(target), "payload": read_json(target)}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/guided/requirements")
    def guided_requirements(task_type: str = Query(default="full_research_pipeline")) -> Dict[str, Any]:
        try:
            return dataset_requirements(task_type)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/guided/task-intake")
    def guided_task_intake(request: TaskIntakeRequest) -> Dict[str, Any]:
        try:
            return analyze_task_intake(
                user_request=request.user_request,
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/chat/message")
    def chat_message(request: ChatMessageRequest) -> Dict[str, Any]:
        try:
            return handle_chat_message(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                session_id=request.session_id,
                mode=request.mode,
                message=request.message,
                attachments=request.attachments,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/chat/action")
    def chat_action(request: ChatActionRequest) -> Dict[str, Any]:
        try:
            return handle_chat_action(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                session_id=request.session_id,
                action_id=request.action_id,
                confirm=request.confirm,
                task_state_payload=request.task_state,
                file_payload=request.file_payload,
                kline_params=request.kline_params,
                signal_record=request.signal_record,
                scoring_payload=request.scoring_payload,
                api_model=request.api_model,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/chat/run-status")
    def chat_run_status(
        profile: str = Query(default="gpt_oss_20b_final"),
        project_id: str = Query(...),
        dataset_id: str = Query(...),
        run_id: str = Query(...),
    ) -> Dict[str, Any]:
        try:
            return build_chat_run_status(
                profile_id=profile,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/guided/dataset-onboarding")
    def guided_dataset_onboarding(request: DatasetOnboardingRequest) -> Dict[str, Any]:
        try:
            return build_dataset_manifest(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                task_type=request.task_type,
                filename=request.filename,
                content=request.content,
                content_encoding=request.content_encoding,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/guided/dataset-onboarding/imported-assets")
    def guided_dataset_onboarding_imported_assets(request: ImportedAssetManifestRequest) -> Dict[str, Any]:
        try:
            return build_imported_asset_manifest(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                task_type=request.task_type,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/guided/dataset-onboarding/online-kline/start")
    def guided_dataset_onboarding_online_kline_start(request: KlineDownloadStartRequest) -> Dict[str, Any]:
        try:
            return _kline_job_manager().start_job(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                task_type=request.task_type,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                stock_codes=request.stock_codes,
                earliest_date=request.earliest_date,
                adjust_type=request.adjust_type,
                full_refresh=request.full_refresh,
                update_indexes=request.update_indexes,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/guided/dataset-onboarding/online-kline/job")
    def guided_dataset_onboarding_online_kline_job(
        profile: str = Query(default="gpt_oss_20b_final"),
        project_id: str = Query(default="default-project"),
        dataset_id: str = Query(default="default-dataset"),
        run_id: str = Query(default="guided-run"),
        job_id: str = Query(...),
    ) -> Dict[str, Any]:
        try:
            return _kline_job_manager().get_job(
                profile_id=profile,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                job_id=job_id,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/guided/run-wizard")
    def guided_run_wizard(request: GuidedRunWizardRequest) -> Dict[str, Any]:
        try:
            return create_guided_run_bundle(
                profile_id=request.profile,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id,
                task_intake=request.task_intake,
                dataset_manifest=request.dataset_manifest,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                api_model=request.api_model,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/console/run-spec")
    def console_run_spec(request: RunSpecRequest) -> Dict[str, Any]:
        try:
            orchestrator = QuantPipelineOrchestrator()
            kwargs: Dict[str, Any] = {}
            if request.mode == "full_pipeline":
                kwargs = {
                    "selection_json_hint": request.selection_json,
                    "final_lesson_state_hint": request.final_lesson_state_json,
                }
            elif request.mode == "inner_loop_only":
                kwargs = {
                    "selection_json": request.selection_json,
                    "api_model": request.api_model,
                }
            elif request.mode == "scoring_only":
                kwargs = {
                    "lesson_alias": request.lesson_alias,
                }
            plan = orchestrator.build_plan(
                mode=request.mode,
                research_goal=request.research_goal,
                run_label=request.run_id or request.mode,
                **kwargs,
            )
            return create_run_spec(
                profile_id=request.profile,
                mode=request.mode,
                research_goal=request.research_goal,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                run_id=request.run_id or request.mode,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                plan=plan,
                selection_json=request.selection_json,
                final_lesson_state_json=request.final_lesson_state_json,
                lesson_alias=request.lesson_alias,
                api_model=request.api_model,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/run-monitor")
    def console_run_monitor(
        project_id: str,
        dataset_id: str,
        run_id: str,
    ) -> Dict[str, Any]:
        try:
            return build_run_monitor(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/teacher-zoo")
    def console_teacher_zoo(
        project_id: str,
        dataset_id: str,
        run_id: str,
        profile: str = Query(default="gpt_oss_20b_final"),
    ) -> Dict[str, Any]:
        try:
            return build_teacher_zoo_view(
                profile_chief=_chief_for_profile(profile),
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/lesson-set")
    def console_lesson_set(
        project_id: str,
        dataset_id: str,
        run_id: str,
    ) -> Dict[str, Any]:
        try:
            return build_lesson_set_view(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/console/provenance")
    def console_provenance(
        project_id: str,
        dataset_id: str,
        run_id: str,
    ) -> Dict[str, Any]:
        try:
            return build_provenance_view(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/pipeline/modes")
    def pipeline_modes() -> Dict[str, Any]:
        try:
            return {"items": QuantPipelineOrchestrator().describe_modes()}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/pipeline/plan")
    def pipeline_plan(request: PipelinePlanRequest) -> Dict[str, Any]:
        try:
            orchestrator = QuantPipelineOrchestrator()
            kwargs: Dict[str, Any] = {}
            if request.mode == "full_pipeline":
                kwargs = {
                    "selection_json_hint": request.selection_json,
                    "final_lesson_state_hint": request.final_lesson_state_json,
                }
            elif request.mode == "inner_loop_only":
                kwargs = {
                    "selection_json": request.selection_json,
                    "api_model": request.api_model,
                }
            elif request.mode == "scoring_only":
                kwargs = {
                    "lesson_alias": request.lesson_alias,
                }
            plan = orchestrator.build_plan(
                mode=request.mode,
                research_goal=request.research_goal,
                run_label=request.run_label or request.mode,
                **kwargs,
            )
            return {
                **plan,
                "suggested_run_contract": build_run_contract(
                    profile_id=request.profile,
                    project_id=request.project_id,
                    dataset_id=request.dataset_id,
                    run_id=request.run_id or request.run_label or request.mode,
                    allow_imported_fallback=request.allow_imported_fallback,
                    allow_demo_fallback=request.allow_demo_fallback,
                ),
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/pipeline/run")
    def pipeline_run(request: PipelineRunRequest) -> Dict[str, Any]:
        try:
            runner = WorkflowRunner(profile_id=request.profile)
            return runner.run(
                mode=request.mode,
                research_goal=request.research_goal,
                run_label=request.run_label or request.mode,
                project_id=request.project_id,
                dataset_id=request.dataset_id,
                selection_json=request.selection_json,
                final_lesson_state_json=request.final_lesson_state_json,
                lesson_alias=request.lesson_alias,
                api_model=request.api_model,
                allow_imported_fallback=request.allow_imported_fallback,
                allow_demo_fallback=request.allow_demo_fallback,
                data_dir=request.data_dir,
                global_env=request.global_env,
                stage_args=request.stage_args,
                stage_env=request.stage_env,
                check=request.check,
                allow_manual_steps=request.allow_manual_steps,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/clean/stages")
    def clean_stages() -> Dict[str, Any]:
        try:
            return {"items": CleanPipelineWrapper().stage_map()}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/clean/plan")
    def clean_plan(request: CleanStagePlanRequest) -> Dict[str, Any]:
        try:
            wrapper = CleanPipelineWrapper()
            return wrapper.plan_stage(
                request.stage,
                extra_args=request.extra_args,
                run_label=request.run_label,
                data_dir=request.data_dir,
                extra_env=request.extra_env,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/clean/run")
    def clean_run(request: CleanStageRunRequest) -> Dict[str, Any]:
        try:
            wrapper = CleanPipelineWrapper()
            return wrapper.run_stage(
                request.stage,
                extra_args=request.extra_args,
                run_label=request.run_label,
                data_dir=request.data_dir,
                extra_env=request.extra_env,
                check=request.check,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/live-config")
    def live_config(profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            return _chief_for_profile(profile).scoring.live_config_status()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/local-model/status")
    def local_model_status() -> Dict[str, Any]:
        try:
            return describe_local_service_status()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/local-model/start")
    def local_model_start(force_restart: bool = False) -> Dict[str, Any]:
        try:
            return start_local_service(force_restart=force_restart)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/local-model/stop")
    def local_model_stop() -> Dict[str, Any]:
        try:
            return stop_local_service()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/markets")
    def markets(profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return {
                "profile": profile,
                "items": [item.__dict__ for item in chief.backtest.list_market_runs()],
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/teachers")
    def teachers(profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return {
                "profile": profile,
                "items": [item.__dict__ for item in chief.teacher_zoo.list_teachers()],
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/lessons")
    def lessons(
        profile: str = Query(default="gpt_oss_20b_final"),
        alias: str = Query(default=""),
    ) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            if alias:
                return {
                    "profile": profile,
                    "alias": alias,
                    "summary": chief.lesson.summarize_lesson_run(alias),
                    "scopes": chief.lesson.load_scope_lessons(alias),
                }
            return {
                "profile": profile,
                "items": [item.__dict__ for item in chief.lesson.list_lesson_runs()],
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/alignment")
    def alignment_rows(profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return {
                "profile": profile,
                "items": [item.__dict__ for item in chief.alignment.list_after_warmup_results()],
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/alignment/{seed}")
    def alignment(seed: str, profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            return _chief_for_profile(profile).alignment.get_after_warmup_result(seed)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/market/{run}")
    def market(run: str, profile: str = Query(default="gpt_oss_20b_final")) -> Dict[str, Any]:
        try:
            return _chief_for_profile(profile).backtest.load_market_summary(run)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/market/{run}/signals")
    def market_signals(
        run: str,
        profile: str = Query(default="gpt_oss_20b_final"),
        signal_date: str = Query(default=""),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return chief.scoring.list_recorded_signal_keys_window(
                run,
                signal_date=signal_date,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/score/recorded")
    def score_recorded(
        run: str,
        signal_date: str,
        symbol: str,
        profile: str = Query(default="gpt_oss_20b_final"),
    ) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return chief.scoring.score_recorded(run, signal_date, symbol).__dict__
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/score/schema")
    def score_schema(
        profile: str = Query(default="gpt_oss_20b_final"),
        run: str = Query(default=""),
        signal_date: str = Query(default=""),
        symbol: str = Query(default=""),
    ) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(profile)
            return chief.scoring.describe_signal_schema(
                market_run_alias=run,
                signal_date=signal_date,
                symbol=symbol,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/score/live")
    def score_live(request: ScoreLiveRequest) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(request.profile)
            signal_record = _resolve_signal_record(chief, request)
            return chief.scoring.score_live(
                lesson_alias=request.lesson_alias,
                final_lesson_state_json=request.final_lesson_state_json,
                signal_record=signal_record,
                prompt_only=request.prompt_only,
                reuse_cache=request.reuse_cache,
                persist_run=request.persist_run,
                run_label=request.run_label,
                source_tag="recorded_reference" if request.from_recorded_run else "external_signal",
                schema_market_run_alias=request.schema_from_run or request.from_recorded_run,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/score/live-batch-external")
    def score_live_batch_external(request: ScoreLiveExternalBatchRequest) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(request.profile)
            items = []
            for idx, signal_record in enumerate(request.signal_records):
                payload = chief.scoring.score_live(
                    lesson_alias=request.lesson_alias,
                    final_lesson_state_json=request.final_lesson_state_json,
                    signal_record=_normalize_signal_date_alias(dict(signal_record)),
                    prompt_only=request.prompt_only,
                    reuse_cache=request.reuse_cache,
                    persist_run=request.persist_run,
                    run_label=request.run_label or f"batch_{idx:04d}",
                    source_tag="external_signal_batch",
                    schema_market_run_alias=request.schema_from_run,
                )
                items.append(payload)
            return {
                "summary": {
                    "count": len(items),
                    "profile": request.profile,
                    "lesson_alias": request.lesson_alias,
                    "final_lesson_state_json": request.final_lesson_state_json,
                    "prompt_only": request.prompt_only,
                    "used_current_workflow_final_lesson_set": bool(request.final_lesson_state_json),
                    "used_imported_lesson": bool(request.lesson_alias and not request.final_lesson_state_json),
                },
                "items": items,
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/score/compare")
    def score_compare(request: CompareLiveRequest) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(request.profile)
            return chief.scoring.compare_live_to_recorded(
                lesson_alias=request.lesson_alias,
                final_lesson_state_json=request.final_lesson_state_json,
                market_run_alias=request.run,
                signal_date=request.signal_date,
                symbol=request.symbol,
                prompt_only=request.prompt_only,
                reuse_cache=request.reuse_cache,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/score/batch")
    def score_batch(request: ScoreLiveBatchRequest) -> Dict[str, Any]:
        try:
            chief = _chief_for_profile(request.profile)
            return chief.scoring.score_live_batch_from_recorded(
                lesson_alias=request.lesson_alias,
                final_lesson_state_json=request.final_lesson_state_json,
                market_run_alias=request.run,
                limit=request.limit,
                offset=request.offset,
                signal_date=request.signal_date,
                prompt_only=request.prompt_only,
                reuse_cache=request.reuse_cache,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    return app


app = create_app()
