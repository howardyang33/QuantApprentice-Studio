from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AssetSpec:
    asset_id: str
    kind: str
    source_path: str
    relative_target: str = ""
    recursive: bool = False
    include_globs: List[str] = field(default_factory=list)
    exclude_globs: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class AssetRecord:
    asset_id: str
    kind: str
    source_path: str
    target_path: str
    status: str
    copied_files: int
    copied_bytes: int
    recursive: bool
    notes: str = ""


@dataclass
class ImportManifest:
    profile_id: str
    imported_at: str
    studio_root: str
    original_quant_root: str
    import_root: str
    records: List[AssetRecord]


@dataclass
class TeacherDescriptor:
    round_id: str
    source_round_id: str
    title: str
    family: str
    template: str
    walkforward_final_nav: float
    factor_analysis_path: str
    branch_rule_cards_path: str
    selected_spec_path: str
    model_artifact_path: str


@dataclass
class LessonRunDescriptor:
    alias: str
    seed_label: str
    final_lesson_state_json: str
    warmup_state_json: str


@dataclass
class AlignmentSeedResult:
    seed: str
    signals_mean_return_pct: float
    teacher_selected_mean_return_pct: float
    teacher_uplift_vs_not_selected_pct: float
    teacher_score_spearman: float
    selected_mean_return_pct: float
    uplift_vs_not_selected_pct: float
    gap_to_teacher_uplift_pct: float
    batch_nav_final: float


@dataclass
class MarketRunDescriptor:
    alias: str
    window: str
    summary_json: str
    llm_signal_scores_json: str
    llm_daily_nav_json: str
    teacher_daily_nav_json: str


@dataclass
class SignalScore:
    market_run_alias: str
    signal_date: str
    symbol: str
    total_score: float
    short_reason: str
    future_return_5d: float
    signal_record: Dict[str, Any]
    parsed_payload: Dict[str, Any]
    subscores: Dict[str, Any]
