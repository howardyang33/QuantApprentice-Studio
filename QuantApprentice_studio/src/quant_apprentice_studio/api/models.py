from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class CleanStagePlanRequest(BaseModel):
    stage: str
    run_label: str = ""
    data_dir: str = ""
    extra_args: list[str] = Field(default_factory=list)
    extra_env: Dict[str, str] = Field(default_factory=dict)


class CleanStageRunRequest(CleanStagePlanRequest):
    check: bool = False


class PipelinePlanRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    mode: str
    research_goal: str
    run_label: str = ""
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = ""
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    selection_json: str = ""
    final_lesson_state_json: str = ""
    lesson_alias: str = "alignment_seed0005"
    api_model: str = "gpt-oss-20b"


class PipelineRunRequest(PipelinePlanRequest):
    data_dir: str = ""
    global_env: Dict[str, str] = Field(default_factory=dict)
    stage_args: Dict[str, list[str]] = Field(default_factory=dict)
    stage_env: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    check: bool = False
    allow_manual_steps: bool = True


class ScoreLiveRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    lesson_alias: str = "alignment_seed0005"
    final_lesson_state_json: str = ""
    signal_record: Optional[Dict[str, Any]] = None
    from_recorded_run: str = ""
    signal_date: str = ""
    symbol: str = ""
    prompt_only: bool = False
    reuse_cache: bool = True
    persist_run: bool = True
    run_label: str = ""
    schema_from_run: str = ""

    @model_validator(mode="after")
    def validate_source(self) -> "ScoreLiveRequest":
        has_signal_record = self.signal_record is not None
        has_recorded_ref = bool(self.from_recorded_run and self.signal_date and self.symbol)
        if has_signal_record == has_recorded_ref:
            raise ValueError(
                "Provide exactly one source: either signal_record, or from_recorded_run + signal_date + symbol."
            )
        return self


class CompareLiveRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    lesson_alias: str = "alignment_seed0005"
    final_lesson_state_json: str = ""
    run: str
    signal_date: str
    symbol: str
    prompt_only: bool = False
    reuse_cache: bool = True


class ScoreLiveBatchRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    lesson_alias: str = "alignment_seed0005"
    final_lesson_state_json: str = ""
    run: str
    limit: int = Field(default=5, ge=1, le=512)
    offset: int = Field(default=0, ge=0)
    signal_date: str = ""
    prompt_only: bool = False
    reuse_cache: bool = True


class RunSpecRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    mode: str = "full_pipeline"
    research_goal: str
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = ""
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    selection_json: str = ""
    final_lesson_state_json: str = ""
    lesson_alias: str = "alignment_seed0005"
    api_model: str = "gpt-oss-20b"


class TaskIntakeRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    user_request: str


class ChatMessageRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    session_id: str = ""
    mode: str = "simple"
    message: str
    attachments: list[Dict[str, Any]] = Field(default_factory=list)


class ChatActionRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    session_id: str = ""
    action_id: str
    confirm: bool = False
    task_state: Dict[str, Any] = Field(default_factory=dict)
    file_payload: Optional[Dict[str, Any]] = None
    kline_params: Dict[str, Any] = Field(default_factory=dict)
    signal_record: Optional[Dict[str, Any]] = None
    scoring_payload: Dict[str, Any] = Field(default_factory=dict)
    api_model: str = "gpt-oss-20b"


class DatasetOnboardingRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    task_type: str
    filename: str
    content: str
    content_encoding: str = "text"


class GuidedRunWizardRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    api_model: str = "gpt-oss-20b"
    task_intake: Dict[str, Any]
    dataset_manifest: Optional[Dict[str, Any]] = None


class ImportedAssetManifestRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    task_type: str


class KlineDownloadStartRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    project_id: str = "default-project"
    dataset_id: str = "default-dataset"
    run_id: str = "guided-run"
    allow_imported_fallback: bool = True
    allow_demo_fallback: bool = False
    task_type: str = "full_research_pipeline"
    stock_codes: str
    earliest_date: str = "20190101"
    adjust_type: str = "qfq"
    full_refresh: bool = False
    update_indexes: bool = True

    @model_validator(mode="after")
    def validate_adjust_type(self) -> "KlineDownloadStartRequest":
        if str(self.adjust_type).strip() not in {"qfq", "hfq"}:
            raise ValueError("adjust_type must be qfq or hfq.")
        if not str(self.stock_codes).strip():
            raise ValueError("stock_codes cannot be empty.")
        return self


class ScoreLiveExternalBatchRequest(BaseModel):
    profile: str = "gpt_oss_20b_final"
    lesson_alias: str = "alignment_seed0005"
    final_lesson_state_json: str = ""
    signal_records: list[Dict[str, Any]] = Field(default_factory=list)
    prompt_only: bool = False
    reuse_cache: bool = True
    persist_run: bool = False
    run_label: str = ""
    schema_from_run: str = ""

    @model_validator(mode="after")
    def validate_records(self) -> "ScoreLiveExternalBatchRequest":
        if not self.signal_records:
            raise ValueError("Provide at least one signal record for batch scoring.")
        return self
