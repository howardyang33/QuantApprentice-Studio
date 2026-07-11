"""Autonomous multi-round teacher-construction loop."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .._paths import env_path, project_root
from ..backtest.teacher_loop_nav_backtest import run_round_nav_backtest
from ..data.loaders import load_daily_data, load_universe
from ..memory.store import MemoryStore
from ..pilot2.teacher_utils import (
    build_executable_label_frame,
    build_reversal_candidate_mask,
    compute_reversal_features,
)
from ..pilot2.walkforward_utils import (
    TRAIN_BUCKET_QUANTILES,
    assign_prediction_buckets,
    compute_train_thresholds,
)
from .factor_analysis import run_teacher_factor_analysis
from .registry import ALL_REGISTERED_FEATURES, DERIVED_FEATURE_NAMES, registry_prompt_block
from .zoo import build_teacher_zoo_index, ensure_teacher_loop_indexes


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not str(value).strip() else int(value)


def _env_timestamp(name: str, default: str) -> pd.Timestamp:
    value = os.environ.get(name, default)
    return pd.Timestamp(str(value))


def _parse_year_list(value: Optional[str], default: List[int]) -> List[int]:
    if value is None or not str(value).strip():
        return list(default)
    text = str(value).strip()
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start_year = int(start_text.strip())
        end_year = int(end_text.strip())
        return list(range(start_year, end_year + 1))
    return [int(text)]


PROJECT_ROOT = env_path("QUANT_PROJECT_ROOT", project_root())
MEMORY_DIR = env_path("QUANT_MEMORY_DIR", PROJECT_ROOT / "research_memory")
REPORT_ROOT = env_path("TEACHER_LOOP_REPORT_ROOT", PROJECT_ROOT / "reports" / "teacher_loop")
ARTIFACT_ROOT = env_path("TEACHER_LOOP_ARTIFACT_ROOT", MEMORY_DIR / "artifacts" / "teacher_loop")
STOCK_DATA_DIR = env_path("TEACHER_LOOP_DATA_DIR", PROJECT_ROOT / "day_klines")
CACHE_ROOT = env_path("TEACHER_LOOP_CACHE_ROOT", ARTIFACT_ROOT / "_shared_cache")

WINDOW_START = _env_timestamp("TEACHER_LOOP_WINDOW_START", "2019-01-01")
WINDOW_END = _env_timestamp("TEACHER_LOOP_WINDOW_END", "2026-06-01")
LOAD_START = WINDOW_START - pd.Timedelta(days=_env_int("TEACHER_LOOP_LOAD_PADDING_DAYS", 400))
LOAD_END = WINDOW_END + pd.Timedelta(days=_env_int("TEACHER_LOOP_LOAD_FORWARD_DAYS", 20))
MIN_HISTORY = _env_int("TEACHER_LOOP_MIN_HISTORY", 120)
HORIZON = _env_int("TEACHER_LOOP_HORIZON", 5)
TEST_YEARS = _parse_year_list(os.environ.get("TEACHER_LOOP_TEST_YEARS"), list(range(2020, 2027)))
MASTER_CACHE_VERSION = os.environ.get(
    "TEACHER_LOOP_CACHE_TAG",
    f"{STOCK_DATA_DIR.name}_{WINDOW_START.strftime('%Y%m%d')}_{WINDOW_END.strftime('%Y%m%d')}_y{TEST_YEARS[0]}_{TEST_YEARS[-1]}",
)
MASTER_CACHE_VERSION = re.sub(r"[^a-zA-Z0-9_.-]+", "_", MASTER_CACHE_VERSION)
MASTER_CACHE_PATH = CACHE_ROOT / f"master_feature_label_{MASTER_CACHE_VERSION}.joblib"

FEATURE_IMPORTANCE_SAMPLE_SIZE = 8000
FEATURE_IMPORTANCE_REPEATS = 3
RANDOM_SEED = 42

SUPPORTED_SAMPLE_TEMPLATES = {
    "weak_state_reversal_pool": "Weak-state reversal candidate pool: J<=35 or (ret_3<=-3% and pos_20<=0.35).",
    "hard_threshold_reversal_gate": "Strict reversal gate: J<=15 and ret_5<=-4% and pos_20<=0.20 and amt_zscore_20>=0.",
    "trend_breakout_pool": "Trend breakout pool: ret_20>=8%, close_to_ma20>=1.02, pos_20>=0.75, amt_zscore_20>=0.",
    "trend_pullback_pool": "Trend pullback pool: ret_20>=5%, close_to_ma20>=1.00, ret_5<=2%, pos_20 in [0.35,0.85], amt_zscore_20>=-0.5.",
}
SUPPORTED_MODEL_FAMILIES = {
    "ridge_regression": "StandardScaler + Ridge on continuous future_return_5d.",
    "logistic_regression": "StandardScaler + LogisticRegression on binary future_return_positive_5d.",
    "hist_gbdt_regression": "HistGradientBoostingRegressor on continuous future_return_5d.",
    "xgb_regression_gpu": "XGBoost regressor with CUDA device on continuous future_return_5d.",
    "xgb_classification_gpu": "XGBoost classifier with CUDA device on binary future_return_positive_5d.",
}
SUPPORTED_EVALUATION_CONTRACTS = {
    "yearly_q5_majority": "Q5 alpha positive in >=60% years; mean and median alpha must be positive.",
    "yearly_q5_strict_incremental": "Q5 alpha nonnegative in every year; no statistically significant underperformance years.",
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _gpu_required() -> bool:
    return _env_flag("TEACHER_LOOP_REQUIRE_GPU", default=False)


def _gpu_device() -> str:
    return os.environ.get("TEACHER_LOOP_GPU_DEVICE", "cuda").strip() or "cuda"


def _supported_model_families(*, require_gpu: bool) -> Dict[str, str]:
    if not require_gpu:
        return SUPPORTED_MODEL_FAMILIES
    return {name: desc for name, desc in SUPPORTED_MODEL_FAMILIES.items() if name.startswith("xgb_")}


def _training_backend_label(spec: "TeacherSpec") -> str:
    if spec.model_family.startswith("xgb_"):
        return f"xgboost:{_gpu_device()}"
    return "sklearn:cpu"


def _build_worker_count() -> int:
    explicit = os.environ.get("TEACHER_LOOP_BUILD_WORKERS")
    if explicit:
        return max(1, int(explicit))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_timestamp() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "teacher"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_studio_research_spec() -> Dict[str, Any]:
    path_text = os.environ.get("QA_STUDIO_RESEARCH_SPEC_JSON", "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class TeacherSpec:
    title: str
    teacher_role: str
    research_family: str
    hypothesis: str
    sample_template: str
    model_family: str
    target_kind: str
    evaluation_contract: str
    feature_columns: List[str]
    novelty_rationale: str

    def validate(self) -> None:
        if self.sample_template not in SUPPORTED_SAMPLE_TEMPLATES:
            raise ValueError(f"Unsupported sample_template: {self.sample_template}")
        if self.model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"Unsupported model_family: {self.model_family}")
        if self.evaluation_contract not in SUPPORTED_EVALUATION_CONTRACTS:
            raise ValueError(f"Unsupported evaluation_contract: {self.evaluation_contract}")
        if self.target_kind not in {"future_return_5d", "future_return_positive_5d"}:
            raise ValueError(f"Unsupported target_kind: {self.target_kind}")
        if self.model_family == "logistic_regression" and self.target_kind != "future_return_positive_5d":
            raise ValueError("logistic_regression requires target_kind=future_return_positive_5d")
        if self.model_family == "xgb_classification_gpu" and self.target_kind != "future_return_positive_5d":
            raise ValueError("xgb_classification_gpu requires target_kind=future_return_positive_5d")
        if self.model_family != "logistic_regression" and self.target_kind != "future_return_5d":
            if self.model_family != "xgb_classification_gpu":
                raise ValueError("regression families require target_kind=future_return_5d")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        unknown = sorted(set(self.feature_columns) - set(ALL_REGISTERED_FEATURES))
        if unknown:
            raise ValueError(f"Unknown feature columns: {unknown}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NoveltyResult:
    too_similar: bool
    max_feature_overlap: float
    nearest_teachers: List[Dict[str, Any]]
    novelty_reason: str


@dataclass
class LoopJudgement:
    judgement_label: str
    accepted: bool
    zoo_partition: str
    positive_years: int
    total_years: int
    positive_rate: float
    mean_alpha: float
    median_alpha: float
    statistically_worse_years: int
    single_year_dominated: bool
    interpretation: str


def build_registered_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = compute_reversal_features(raw_df)
    out["ret_1_clip"] = out["ret_1"].clip(-0.20, 0.20)
    out["ret_3_clip"] = out["ret_3"].clip(-0.30, 0.30)
    out["gap_pct"] = out["open"] / (out["prev_close"] + 1e-12) - 1.0
    out["volume_log"] = np.log1p(out["volume"].clip(lower=0.0))
    out["close_log"] = np.log(out["close"].clip(lower=1e-12))
    for col in ALL_REGISTERED_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        arr = out[col].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        arr[~np.isfinite(arr)] = 0.0
        out[col] = arr
    return out


def mask_weak_state_reversal_pool(df: pd.DataFrame) -> pd.Series:
    return build_reversal_candidate_mask(df)


def mask_hard_threshold_reversal_gate(df: pd.DataFrame) -> pd.Series:
    mask = (
        (df["J"] <= 15.0)
        & (df["ret_5"] <= -0.04)
        & (df["pos_20"] <= 0.20)
        & (df["amt_zscore_20"] >= 0.0)
    )
    return mask.fillna(False)


def mask_trend_breakout_pool(df: pd.DataFrame) -> pd.Series:
    mask = (
        (df["ret_20"] >= 0.08)
        & (df["close_to_ma20"] >= 1.02)
        & (df["pos_20"] >= 0.75)
        & (df["amt_zscore_20"] >= 0.0)
    )
    return mask.fillna(False)


def mask_trend_pullback_pool(df: pd.DataFrame) -> pd.Series:
    mask = (
        (df["ret_20"] >= 0.05)
        & (df["close_to_ma20"] >= 1.00)
        & (df["ret_5"] <= 0.02)
        & (df["pos_20"] >= 0.35)
        & (df["pos_20"] <= 0.85)
        & (df["amt_zscore_20"] >= -0.5)
    )
    return mask.fillna(False)


SAMPLE_TEMPLATE_TO_MASK = {
    "weak_state_reversal_pool": mask_weak_state_reversal_pool,
    "hard_threshold_reversal_gate": mask_hard_threshold_reversal_gate,
    "trend_breakout_pool": mask_trend_breakout_pool,
    "trend_pullback_pool": mask_trend_pullback_pool,
}


def load_loop_state(memory_dir: Path) -> Dict[str, Any]:
    state_path = memory_dir / "indexes" / "teacher_loop_state.json"
    return _load_json(state_path)


def save_loop_state(memory_dir: Path, state: Dict[str, Any]) -> None:
    state_path = memory_dir / "indexes" / "teacher_loop_state.json"
    _write_json(state_path, state)


def append_loop_manifest(memory_dir: Path, payload: Dict[str, Any]) -> None:
    manifest_path = memory_dir / "indexes" / "teacher_loop_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def next_round_id(memory_dir: Path) -> Tuple[str, int]:
    state = load_loop_state(memory_dir)
    round_index = int(state.get("rounds_launched", 0)) + 1
    return f"round_{round_index:03d}", round_index


def _feature_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def compare_spec_to_zoo(spec: TeacherSpec, zoo_payload: Dict[str, Any]) -> NoveltyResult:
    comparisons: List[Dict[str, Any]] = []
    for teacher in zoo_payload.get("teachers", []):
        overlap = _feature_jaccard(spec.feature_columns, teacher.get("feature_columns", []))
        same_family = spec.research_family == teacher.get("research_family")
        same_template = spec.sample_template == teacher.get("sample_template")
        same_model_family = spec.model_family == teacher.get("model_family")
        comparisons.append(
            {
                "memory_id": teacher["memory_id"],
                "title": teacher["title"],
                "zoo_partition": teacher["zoo_partition"],
                "feature_overlap": overlap,
                "same_family": same_family,
                "same_sample_template": same_template,
                "same_model_family": same_model_family,
            }
        )
    comparisons.sort(key=lambda row: (row["feature_overlap"], row["same_sample_template"], row["same_model_family"]), reverse=True)
    nearest = comparisons[:5]
    too_similar = any(
        row["same_family"] and row["same_sample_template"] and row["same_model_family"] and row["feature_overlap"] >= 0.85
        for row in comparisons
    )
    if too_similar:
        reason = "Selected spec is too similar to an existing teacher attempt on family + template + model family + feature overlap."
    elif nearest:
        reason = f"Nearest feature overlap is {nearest[0]['feature_overlap']:.2%}; acceptable novelty."
    else:
        reason = "No prior teachers in zoo; novelty is trivially acceptable."
    return NoveltyResult(
        too_similar=too_similar,
        max_feature_overlap=max((row["feature_overlap"] for row in comparisons), default=0.0),
        nearest_teachers=nearest,
        novelty_reason=reason,
    )


def fallback_spec(*, require_gpu: bool = False, studio_research_spec: Optional[Dict[str, Any]] = None) -> TeacherSpec:
    studio_research_spec = dict(studio_research_spec or {})
    preferred_templates = [
        str(x).strip()
        for x in list(studio_research_spec.get("preferred_sample_templates") or [])
        if str(x).strip() in SUPPORTED_SAMPLE_TEMPLATES
    ]
    fallback_preferences = dict(studio_research_spec.get("fallback_preferences") or {})
    sample_template = str(fallback_preferences.get("sample_template", "")).strip()
    if sample_template not in SUPPORTED_SAMPLE_TEMPLATES:
        sample_template = preferred_templates[0] if preferred_templates else "trend_breakout_pool"

    if sample_template == "trend_breakout_pool":
        research_family = "breakout"
        title = "Studio-guided fallback breakout selector"
        teacher_role = "Use breakout candidates and let a compact classifier score which breakouts hold for 5 executable trading days."
        hypothesis = str(studio_research_spec.get("primary_hypothesis", "")).strip() or (
            "A compact breakout pool with probability ranking may provide a distinct route from reversal-focused tries and yield a useful Q5 selector."
        )
        model_family = "xgb_classification_gpu" if require_gpu else "logistic_regression"
        target_kind = "future_return_positive_5d"
        feature_columns = [
            "ret_5",
            "ret_10",
            "ret_20",
            "close_to_ma10",
            "close_to_ma20",
            "pos_20",
            "body_pct",
            "amplitude",
            "volume_log",
            "amt_log",
            "amt_zscore_20",
            "volatility_10",
            "volatility_20",
        ]
    elif sample_template == "trend_pullback_pool":
        research_family = "trend_pullback"
        title = "Studio-guided fallback trend pullback ranker"
        teacher_role = "Use trend-pullback candidates and rank which pullbacks are most likely to resume in the next 5 executable trading days."
        hypothesis = str(studio_research_spec.get("primary_hypothesis", "")).strip() or (
            "A trend-pullback pool with regime-aware ranking may separate constructive pullbacks from weak continuation setups."
        )
        model_family = "xgb_regression_gpu" if require_gpu else "ridge_regression"
        target_kind = "future_return_5d"
        feature_columns = [
            "ret_5",
            "ret_10",
            "ret_20",
            "close_to_ma10",
            "close_to_ma20",
            "pos_20",
            "J",
            "D",
            "amplitude",
            "volume_log",
            "amt_zscore_20",
            "volatility_10",
            "volatility_20",
        ]
    else:
        research_family = "reversal"
        title = "Studio-guided fallback reversal ranker"
        teacher_role = "Use weak-state reversal candidates and rank which oversold setups are most likely to rebound over 5 executable trading days."
        hypothesis = str(studio_research_spec.get("primary_hypothesis", "")).strip() or (
            "A weak-state reversal pool with oscillator and volatility context may isolate higher-quality rebound candidates."
        )
        model_family = "xgb_regression_gpu" if require_gpu else "ridge_regression"
        target_kind = "future_return_5d"
        feature_columns = [
            "ret_3",
            "ret_5",
            "pos_20",
            "J",
            "D",
            "body_pct",
            "amplitude",
            "volume_log",
            "amt_zscore_20",
            "volatility_10",
            "volatility_20",
        ]

    novelty_rationale = (
        "Fallback spec chosen under the current studio research mandate, preserving the preferred sample-template direction "
        "while remaining within the supported execution DSL."
        if studio_research_spec
        else "Moves away from reversal and within-gate reranking toward a trend-following family using a classifier instead of a regressor."
    )

    spec = TeacherSpec(
        title=title,
        teacher_role=teacher_role,
        research_family=research_family,
        hypothesis=hypothesis,
        sample_template=sample_template,
        model_family=model_family,
        target_kind=target_kind,
        evaluation_contract="yearly_q5_majority",
        feature_columns=feature_columns,
        novelty_rationale=novelty_rationale,
    )
    spec.validate()
    return spec


def _teacher_loop_api_url() -> str:
    return os.environ.get("TEACHER_LOOP_API_URL", os.environ.get("LLM_API_URL", "https://api.chatanywhere.tech/v1/chat/completions")).strip()


def _teacher_loop_api_key() -> str:
    return os.environ.get("TEACHER_LOOP_API_KEY", os.environ.get("CHATANYWHERE_API_KEY", "")).strip()


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _normalize_candidate_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(candidate)
    alias_maps = {
        "target_kind": {
            "continuous_future_return_5d": "future_return_5d",
            "future_return": "future_return_5d",
            "regression_future_return_5d": "future_return_5d",
            "regression": "future_return_5d",
            "continuous": "future_return_5d",
            "continuous future_return_5d": "future_return_5d",
            "binary_future_return_positive_5d": "future_return_positive_5d",
            "binary_positive_future_return_5d": "future_return_positive_5d",
            "future_return_positive": "future_return_positive_5d",
            "binary": "future_return_positive_5d",
            "classification": "future_return_positive_5d",
            "binary_classification": "future_return_positive_5d",
            "positive_5d": "future_return_positive_5d",
            "binary future_return_positive_5d": "future_return_positive_5d",
        },
        "evaluation_contract": {
            "yearly_q5_incremental": "yearly_q5_strict_incremental",
            "yearly_q5_all_positive": "yearly_q5_strict_incremental",
            "yearly_majority_q5": "yearly_q5_majority",
        },
        "sample_template": {
            "Weak-state reversal candidate pool: J<=35 or (ret_3<=-3% and pos_20<=0.35)": "weak_state_reversal_pool",
            "Hard-threshold reversal gate: J<=15 and ret_5<=-4% and pos_20<=0.20 and amt_zscore_20>=0": "hard_threshold_reversal_gate",
            "Trend breakout pool: ret_20>=8%, close_to_ma20>=1.02, pos_20>=0.75, amt_zscore_20>=0": "trend_breakout_pool",
            "Trend pullback pool: ret_20>=5%, close_to_ma20>=1.00, ret_5<=2%, pos_20 in [0.35,0.85], amt_zscore_20>=-0.5": "trend_pullback_pool",
        },
    }
    for field, aliases in alias_maps.items():
        value = normalized.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            normalized[field] = aliases.get(stripped, stripped)
    return normalized


def _chat_completion(messages: List[Dict[str, str]], *, model: str, api_key: str, max_tokens: int = 5000) -> Dict[str, Any]:
    url = _teacher_loop_api_url()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    model_lower = model.lower()
    if "gpt-oss" in model_lower:
        payload["reasoning_effort"] = "low"
    elif model.startswith("gpt-"):
        payload["reasoning_effort"] = "minimal"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={key: value for key, value in {
            "Authorization": f"Bearer {api_key}" if api_key else None,
            "Content-Type": "application/json",
        }.items() if value is not None},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"URLError: {exc}") from exc


def propose_next_spec(
    memory_dir: Path,
    round_dir: Path,
    *,
    model_name: Optional[str] = None,
) -> Tuple[TeacherSpec, NoveltyResult, Dict[str, Any], Dict[str, Any]]:
    api_key = _teacher_loop_api_key()
    model_name = model_name or os.environ.get("TEACHER_LOOP_API_MODEL", "gpt-5").strip()
    require_gpu = _gpu_required()
    studio_research_spec = _load_studio_research_spec()
    supported_model_families = _supported_model_families(require_gpu=require_gpu)
    feature_registry = _load_json(memory_dir / "indexes" / "feature_registry.json")
    zoo_payload = _load_json(memory_dir / "indexes" / "teacher_zoo_index.json")
    store = MemoryStore(memory_dir)
    recent_lessons = [store.get_item(path=entry["storage_path"]) for entry in store.list_items(item_type="research_lesson", limit=6)]
    recent_teachers = [store.get_item(path=entry["storage_path"]) for entry in store.list_items(item_type="teacher_model", limit=6)]
    recent_factor_cards = [store.get_item(path=entry["storage_path"]) for entry in store.list_items(item_type="factor_card", limit=10)]

    zoo_summary_lines = []
    for teacher in zoo_payload.get("teachers", []):
        metrics = teacher["metrics"]
        zoo_summary_lines.append(
            f"- {teacher['title']} | partition={teacher['zoo_partition']} | family={teacher['research_family']} | "
            f"template={teacher['sample_template']} | model={teacher['model_family']} | features={teacher['feature_count']} | "
            f"alpha(mean={metrics.get('mean_alpha')}, median={metrics.get('median_alpha')}) | "
            f"nav(cagr={metrics.get('nav_cagr')}, mdd={metrics.get('nav_max_drawdown')}, "
            f"nav_positive_years={metrics.get('nav_positive_years')}/{metrics.get('nav_total_years')})"
        )

    lesson_lines = []
    for lesson in recent_lessons:
        lesson_lines.append(f"- {lesson['title']}: {lesson['summary']}")

    teacher_factor_lines = []
    for teacher in recent_teachers:
        factor_summary = teacher.get("factor_analysis_summary") or {}
        top_features = factor_summary.get("top_global_features") or []
        top_combos = factor_summary.get("top_feature_combos") or []
        feature_text = ", ".join(
            f"{row.get('feature')}({row.get('preferred_direction')},{row.get('shape_hint')})"
            for row in top_features[:4]
            if row.get("feature")
        ) or "na"
        combo_text = ", ".join(
            f"{row.get('feature_left')}+{row.get('feature_right')} lift={float(row.get('lift_favored_vs_opposite')):.4f}"
            for row in top_combos[:2]
            if row.get("feature_left") and row.get("feature_right") and row.get("lift_favored_vs_opposite") is not None
        ) or "na"
        teacher_factor_lines.append(
            f"- {teacher['title']} | partition={teacher.get('zoo_partition')} | top_factor_rules={feature_text} | combo_lifts={combo_text}"
        )

    factor_card_lines = []
    for card in recent_factor_cards:
        factor_card_lines.append(
            f"- {card.get('factor_name', card['title'])}: {card.get('summary', '')}"
        )

    system = (
        "You are the autonomous LLM Researcher for QuantApprentice. "
        "Do not reveal reasoning. Return strict JSON only, with no markdown fences or extra prose. "
        "The first character of your response must be '{' and the last character must be '}'."
    )
    studio_lines: List[str] = []
    if studio_research_spec:
        studio_lines.extend(
            [
                "Studio-side research mandate:",
                f"- Research goal: {str(studio_research_spec.get('research_goal', '')).strip() or 'not provided'}",
                f"- Target research style: {str(studio_research_spec.get('target_research_style', '')).strip() or 'not provided'}",
            ]
        )
        preferred_templates = [
            str(x).strip()
            for x in list(studio_research_spec.get("preferred_sample_templates") or [])
            if str(x).strip()
        ]
        if preferred_templates:
            studio_lines.append(
                f"- Preferred sample-template priority: {', '.join(preferred_templates)}"
            )
        factor_families = [
            str(x).strip()
            for x in list(studio_research_spec.get("candidate_factor_families") or [])
            if str(x).strip()
        ]
        if factor_families:
            studio_lines.append(
                f"- Candidate factor families to emphasize: {', '.join(factor_families)}"
            )
        regime_hints = [
            str(x).strip()
            for x in list(studio_research_spec.get("regime_hints") or [])
            if str(x).strip()
        ]
        if regime_hints:
            studio_lines.append(
                f"- Regime hints: {'; '.join(regime_hints)}"
            )
        design_constraints = [
            str(x).strip()
            for x in list(studio_research_spec.get("design_constraints") or [])
            if str(x).strip()
        ]
        if design_constraints:
            studio_lines.append(
                f"- Design constraints: {'; '.join(design_constraints)}"
            )
        diversification_objective = [
            str(x).strip()
            for x in list(studio_research_spec.get("diversification_objective") or [])
            if str(x).strip()
        ]
        if diversification_objective:
            studio_lines.append(
                f"- Diversification objective: {'; '.join(diversification_objective)}"
            )
        studio_lines.append(
            "- Treat this mandate as the planning prior. Among otherwise valid candidates, prefer the one that best satisfies the mandate without duplicating an existing teacher."
        )
    user = "\n".join(
        [
            "Design the next teacher-construction experiment.",
            "Goal: diversify the teacher zoo, avoid repeating similar failed or weaker routes, and stay within the supported execution DSL.",
            *(["", *studio_lines] if studio_lines else []),
            "",
            "Supported sample templates:",
            *[f"- {name}: {desc}" for name, desc in SUPPORTED_SAMPLE_TEMPLATES.items()],
            "",
            "Supported model families:",
            *[f"- {name}: {desc}" for name, desc in supported_model_families.items()],
            "",
            "Supported evaluation contracts:",
            *[f"- {name}: {desc}" for name, desc in SUPPORTED_EVALUATION_CONTRACTS.items()],
            "",
            registry_prompt_block(feature_registry),
            "",
            "Current teacher zoo:",
            *zoo_summary_lines,
            "",
            "Recent factor-behavior memory from prior teachers:",
            *(teacher_factor_lines or ["- none"]),
            "",
            "Recent factor cards:",
            *(factor_card_lines or ["- none"]),
            "",
            "Recent research lessons:",
            *lesson_lines,
            "",
            "Rules:",
            "- Do not propose a route that is too similar to an existing teacher in family + sample template + model family + feature set.",
            "- Prefer a new family or at least a clearly different template/model/feature composition.",
            "- Read the factor-behavior evidence to decide whether an old route is saturated, should be avoided, or can be adjusted with a new interaction/regime angle.",
            "- Use only registered feature names.",
            "- For accepted-but-weaker historical routes in try partition, do not just trivially tweak one feature.",
            f"- GPU requirement is {'enabled' if require_gpu else 'disabled'}; when enabled, choose only GPU-compatible XGBoost model families.",
            "- Output one raw minified JSON object only. No explanation, no prefatory text, no trailing text.",
            "",
            "Return minified JSON with schema:",
            '{"candidates":[{"candidate_id":"c1","title":"","teacher_role":"","research_family":"","hypothesis":"","sample_template":"","model_family":"","target_kind":"","evaluation_contract":"","feature_columns":[""],"novelty_rationale":""}],"selected_candidate_id":"","selection_rationale":""}',
        ]
    )

    prompt_payload = {"system": system, "user": user}
    _write_json(round_dir / "prompt.json", prompt_payload)
    if studio_research_spec:
        _write_json(round_dir / "studio_research_spec_snapshot.json", studio_research_spec)

    if not model_name:
        spec = fallback_spec(require_gpu=require_gpu, studio_research_spec=studio_research_spec)
        novelty = compare_spec_to_zoo(spec, zoo_payload)
        api_raw = {"mode": "fallback_no_model_name"}
        proposal_payload = {
            "candidates": [spec.to_dict()],
            "selected_candidate_id": "fallback",
            "selection_rationale": "Fallback spec used because no teacher-loop API model name was configured.",
        }
        return spec, novelty, api_raw, proposal_payload

    raw_response: Dict[str, Any] | None = None
    try:
        api_raw = _chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model_name,
            api_key=api_key,
        )
        raw_response = api_raw
        _write_json(round_dir / "api_raw.json", api_raw)

        content = api_raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        proposal_payload = _extract_json_payload(content)
        _write_json(round_dir / "proposal.json", proposal_payload)
        by_id = {row["candidate_id"]: row for row in proposal_payload["candidates"]}
        selected = _normalize_candidate_fields(by_id[proposal_payload["selected_candidate_id"]])
        spec = TeacherSpec(
            title=selected["title"],
            teacher_role=selected["teacher_role"],
            research_family=selected["research_family"],
            hypothesis=selected["hypothesis"],
            sample_template=selected["sample_template"],
            model_family=selected["model_family"],
            target_kind=selected["target_kind"],
            evaluation_contract=selected["evaluation_contract"],
            feature_columns=list(selected["feature_columns"]),
            novelty_rationale=selected["novelty_rationale"],
        )
        spec.validate()
        if require_gpu and not spec.model_family.startswith("xgb_"):
            raise ValueError("GPU-required loop received non-GPU model_family from proposal")
    except Exception as exc:
        api_raw = {
            "mode": "fallback_api_parse_error",
            "error": str(exc),
            "api_url": _teacher_loop_api_url(),
            "model_name": model_name,
        }
        if raw_response is not None:
            api_raw["raw_response"] = raw_response
        proposal_payload = {"candidates": [], "selected_candidate_id": None, "selection_rationale": "fallback after API error"}
        _write_json(round_dir / "api_raw.json", api_raw)
        _write_json(round_dir / "proposal.json", proposal_payload)
        spec = fallback_spec(require_gpu=require_gpu, studio_research_spec=studio_research_spec)

    novelty = compare_spec_to_zoo(spec, zoo_payload)
    if novelty.too_similar:
        spec = fallback_spec(require_gpu=require_gpu, studio_research_spec=studio_research_spec)
        novelty = compare_spec_to_zoo(spec, zoo_payload)
    _write_json(round_dir / "selected_spec.json", spec.to_dict())
    _write_json(
        round_dir / "novelty_report.json",
        {
            "too_similar": novelty.too_similar,
            "max_feature_overlap": novelty.max_feature_overlap,
            "novelty_reason": novelty.novelty_reason,
            "nearest_teachers": novelty.nearest_teachers,
        },
    )
    return spec, novelty, api_raw, proposal_payload


def build_dataset_for_spec(spec: TeacherSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    master_dataset, master_stock_summary = _load_or_build_master_dataset_cache()
    sample_mask = SAMPLE_TEMPLATE_TO_MASK[spec.sample_template](master_dataset)
    dataset = master_dataset.loc[sample_mask.to_numpy()].copy()
    if dataset.empty:
        raise RuntimeError(f"No samples were constructed for template {spec.sample_template}")
    keep_cols = [
        "symbol",
        "signal_date",
        "entry_date",
        "exit_date",
        "future_return_5d",
        "future_return_positive_5d",
        "signal_year",
        *spec.feature_columns,
    ]
    dataset = dataset.loc[:, keep_cols].reset_index(drop=True)
    symbol_kept_rows = dataset.groupby("symbol").size()
    stock_summary = master_stock_summary.copy()
    stock_summary["kept_rows"] = stock_summary["symbol"].map(symbol_kept_rows).fillna(0).astype(int)
    stock_summary.loc[(stock_summary["status"] == "ok") & (stock_summary["kept_rows"] == 0), "status"] = "template_filtered"
    return dataset, stock_summary


def _build_master_frame_for_symbol(symbol: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    raw_df = load_daily_data(
        symbol,
        STOCK_DATA_DIR,
        start_date=LOAD_START.strftime("%Y-%m-%d"),
        end_date=LOAD_END.strftime("%Y-%m-%d"),
    )
    if raw_df is None:
        return None, {"symbol": symbol, "status": "missing", "rows": 0, "eligible_rows": 0, "kept_rows": 0}
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    feat_df = build_registered_feature_frame(raw_df)
    label_df = build_executable_label_frame(feat_df, min_history=MIN_HISTORY, horizon=HORIZON)
    if label_df.empty:
        return None, {"symbol": symbol, "status": "no_labels", "rows": len(feat_df), "eligible_rows": 0, "kept_rows": 0}
    label_df["signal_date"] = pd.to_datetime(label_df["signal_date"])
    label_df["entry_date"] = pd.to_datetime(label_df["entry_date"])
    label_df["exit_date"] = pd.to_datetime(label_df["exit_date"])
    label_df = label_df[(label_df["signal_date"] >= WINDOW_START) & (label_df["signal_date"] <= WINDOW_END)].copy()
    if label_df.empty:
        return None, {"symbol": symbol, "status": "outside_window", "rows": len(feat_df), "eligible_rows": 0, "kept_rows": 0}

    signal_indices = label_df["signal_index"].to_numpy(dtype=int)
    signal_features = (
        feat_df.loc[signal_indices, ALL_REGISTERED_FEATURES]
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    stock_frame = pd.DataFrame(
        {
            "symbol": symbol,
            "signal_date": pd.to_datetime(label_df["signal_date"]),
            "entry_date": pd.to_datetime(label_df["entry_date"]),
            "exit_date": pd.to_datetime(label_df["exit_date"]),
            "future_return_5d": label_df["future_return_5d"].to_numpy(dtype=np.float32),
        }
    )
    stock_frame["future_return_positive_5d"] = (stock_frame["future_return_5d"] > 0).astype(np.int8)
    stock_frame["signal_year"] = stock_frame["signal_date"].dt.year.astype(np.int16)
    stock_frame = pd.concat([stock_frame, signal_features], axis=1)
    stat = {
        "symbol": symbol,
        "status": "ok",
        "rows": len(feat_df),
        "eligible_rows": len(label_df),
        "kept_rows": len(stock_frame),
    }
    return stock_frame, stat


def _build_master_feature_label_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_universe(STOCK_DATA_DIR, universe="all")
    per_stock_frames: List[pd.DataFrame] = []
    stock_stats: List[Dict[str, Any]] = []
    total_rows = 0
    workers = _build_worker_count()
    print(f"[master_cache] worker_count={workers}", flush=True)

    def _consume_result(idx: int, symbol: str, stock_frame: Optional[pd.DataFrame], stat: Dict[str, Any]) -> None:
        nonlocal total_rows
        stock_stats.append(stat)
        if stock_frame is not None and not stock_frame.empty:
            per_stock_frames.append(stock_frame)
            total_rows += len(stock_frame)
        if idx % 200 == 0:
            print(f"[master_cache] processed {idx}/{len(symbols)} symbols, rows={total_rows}", flush=True)

    if workers == 1:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                stock_frame, stat = _build_master_frame_for_symbol(symbol)
            except Exception as exc:
                stock_frame = None
                stat = {"symbol": symbol, "status": "error", "rows": 0, "eligible_rows": 0, "kept_rows": 0}
                print(f"[master_cache_error] {symbol}: {exc}", flush=True)
            _consume_result(idx, symbol, stock_frame, stat)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, (symbol, result) in enumerate(zip(symbols, executor.map(_build_master_frame_for_symbol, symbols, chunksize=16)), start=1):
                try:
                    stock_frame, stat = result
                except Exception as exc:
                    stock_frame = None
                    stat = {"symbol": symbol, "status": "error", "rows": 0, "eligible_rows": 0, "kept_rows": 0}
                    print(f"[master_cache_error] {symbol}: {exc}", flush=True)
                _consume_result(idx, symbol, stock_frame, stat)

    dataset = pd.concat(per_stock_frames, ignore_index=True) if per_stock_frames else pd.DataFrame()
    if dataset.empty:
        raise RuntimeError("Master feature-label dataset build produced no rows")
    for col in ALL_REGISTERED_FEATURES:
        dataset[col] = dataset[col].astype(np.float32)
    return dataset, pd.DataFrame(stock_stats)


def _load_or_build_master_dataset_cache() -> Tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if MASTER_CACHE_PATH.exists():
        print(f"[master_cache] loading {MASTER_CACHE_PATH.name}", flush=True)
        payload = joblib.load(MASTER_CACHE_PATH, mmap_mode="r")
        return payload["dataset"], payload["stock_summary"]
    print(f"[master_cache] building {MASTER_CACHE_PATH.name}", flush=True)
    dataset, stock_summary = _build_master_feature_label_dataset()
    joblib.dump(
        {"dataset": dataset, "stock_summary": stock_summary, "cache_version": MASTER_CACHE_VERSION},
        MASTER_CACHE_PATH,
        compress=0,
    )
    print(f"[master_cache] saved {MASTER_CACHE_PATH.name} rows={len(dataset)}", flush=True)
    return dataset, stock_summary


def _feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    return df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def _fit_model(spec: TeacherSpec, train_df: pd.DataFrame):
    x = _feature_matrix(train_df, spec.feature_columns)
    if spec.target_kind == "future_return_positive_5d":
        y = train_df["future_return_positive_5d"].to_numpy(dtype=np.int32)
    else:
        y = train_df["future_return_5d"].to_numpy(dtype=np.float32)

    if spec.model_family == "ridge_regression":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=RANDOM_SEED))])
    elif spec.model_family == "logistic_regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=500,
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    elif spec.model_family == "hist_gbdt_regression":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            max_leaf_nodes=63,
            min_samples_leaf=120,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
            early_stopping=False,
        )
    elif spec.model_family == "xgb_regression_gpu":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=_gpu_device(),
            max_depth=6,
            n_estimators=260,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=6.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            max_bin=256,
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
        )
    elif spec.model_family == "xgb_classification_gpu":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=_gpu_device(),
            max_depth=6,
            n_estimators=240,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            max_bin=256,
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
        )
    else:
        raise ValueError(spec.model_family)
    model.fit(x, y)
    return model


def _score_model(model, spec: TeacherSpec, df: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(df, spec.feature_columns)
    if spec.model_family in {"logistic_regression", "xgb_classification_gpu"}:
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def _per_date_corr(group: pd.DataFrame, score_col: str, label_col: str, method: str) -> float:
    if len(group) < 5:
        return np.nan
    score = group[score_col]
    label = group[label_col]
    if score.nunique() < 2 or label.nunique() < 2:
        return np.nan
    return float(score.corr(label, method=method))


def _score_diagnostics(df: pd.DataFrame, score_col: str, label_col: str = "future_return_5d") -> Dict[str, float]:
    grouped = df.groupby("signal_date", sort=True)
    ic_values = grouped.apply(_per_date_corr, score_col=score_col, label_col=label_col, method="pearson").dropna()
    rank_ic_values = grouped.apply(_per_date_corr, score_col=score_col, label_col=label_col, method="spearman").dropna()
    return {
        "ic_mean": float(ic_values.mean()) if len(ic_values) else np.nan,
        "rank_ic_mean": float(rank_ic_values.mean()) if len(rank_ic_values) else np.nan,
    }


def _build_daily_alpha_frame(test_scored: pd.DataFrame) -> pd.DataFrame:
    base_daily = test_scored.groupby("signal_date", sort=True)["future_return_5d"].mean().rename("baseline_return").reset_index()
    q5_daily = (
        test_scored[test_scored["bucket"] == 5]
        .groupby("signal_date", sort=True)["future_return_5d"]
        .mean()
        .rename("q5_return")
        .reset_index()
    )
    merged = base_daily.merge(q5_daily, on="signal_date", how="left")
    merged = merged[merged["q5_return"].notna()].copy()
    merged["daily_alpha"] = merged["q5_return"] - merged["baseline_return"]
    return merged.reset_index(drop=True)


def _negative_alpha_pvalue(alpha_series: pd.Series) -> float:
    clean = alpha_series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if clean.size < 5 or np.allclose(clean.std(ddof=1), 0.0):
        return np.nan
    t_stat, two_sided_p = stats.ttest_1samp(clean, popmean=0.0, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(two_sided_p):
        return np.nan
    if t_stat < 0:
        return float(two_sided_p / 2.0)
    return float(1.0 - two_sided_p / 2.0)


def _assess_single_year_dominance(alpha_series: pd.Series) -> bool:
    positive = alpha_series[alpha_series > 0].sort_values(ascending=False).reset_index(drop=True)
    if len(positive) <= 1:
        return False
    share = float(positive.iloc[0] / (positive.sum() + 1e-12))
    ratio = float(positive.iloc[0] / (positive.iloc[1] + 1e-12))
    return share > 0.75 and ratio > 2.5


def _extract_importance(model, spec: TeacherSpec, test_df: pd.DataFrame, *, test_year: int) -> pd.DataFrame:
    if spec.model_family in {"ridge_regression", "logistic_regression"}:
        if spec.model_family == "ridge_regression":
            coeffs = model.named_steps["ridge"].coef_.reshape(-1)
        else:
            coeffs = model.named_steps["logit"].coef_.reshape(-1)
        return pd.DataFrame(
            {
                "test_year": test_year,
                "feature": spec.feature_columns,
                "importance": coeffs,
                "importance_abs": np.abs(coeffs),
                "importance_type": "coefficient",
            }
        )
    if spec.model_family in {"xgb_regression_gpu", "xgb_classification_gpu"}:
        booster = model.get_booster()
        gains = booster.get_score(importance_type="gain")
        importances = [float(gains.get(f"f{idx}", 0.0)) for idx in range(len(spec.feature_columns))]
        return pd.DataFrame(
            {
                "test_year": test_year,
                "feature": spec.feature_columns,
                "importance": importances,
                "importance_abs": np.abs(importances),
                "importance_type": "gain",
            }
        )
    x = _feature_matrix(test_df, spec.feature_columns)
    y = test_df["future_return_5d"].to_numpy(dtype=np.float32)
    sample_size = min(len(test_df), FEATURE_IMPORTANCE_SAMPLE_SIZE)
    sample_idx = np.arange(len(test_df))
    if len(test_df) > sample_size:
        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = np.sort(rng.choice(sample_idx, size=sample_size, replace=False))
    result = permutation_importance(
        model,
        x[sample_idx],
        y[sample_idx],
        scoring="neg_mean_squared_error",
        n_repeats=FEATURE_IMPORTANCE_REPEATS,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "test_year": test_year,
            "feature": spec.feature_columns,
            "importance": result.importances_mean,
            "importance_abs": np.abs(result.importances_mean),
            "importance_type": "permutation",
        }
    )


def _make_judgement(spec: TeacherSpec, summary_df: pd.DataFrame) -> LoopJudgement:
    alpha_series = summary_df["q5_alpha_vs_baseline"]
    positive_years = int((alpha_series > 0).sum())
    total_years = int(len(summary_df))
    positive_rate = float(positive_years / total_years) if total_years else 0.0
    mean_alpha = float(alpha_series.mean()) if total_years else np.nan
    median_alpha = float(alpha_series.median()) if total_years else np.nan
    statistically_worse_years = int(
        ((summary_df["q5_alpha_vs_baseline"] < 0) & (summary_df["negative_alpha_pvalue"] < 0.05)).sum()
    )
    single_year_dominated = _assess_single_year_dominance(alpha_series)

    if spec.evaluation_contract == "yearly_q5_strict_incremental":
        accepted = total_years > 0 and positive_years == total_years and statistically_worse_years == 0 and mean_alpha > 0
    else:
        accepted = (
            total_years > 0
            and positive_rate >= 0.60
            and mean_alpha > 0
            and median_alpha > 0
            and not single_year_dominated
        )

    if accepted:
        zoo_partition = "main" if spec.evaluation_contract == "yearly_q5_strict_incremental" else "try"
    else:
        zoo_partition = "try" if (positive_years > 0 or mean_alpha > -0.002) else "rejected"

    judgement_label = "candidate_teacher" if accepted else "not_yet_teacher"
    if accepted:
        interpretation = "The autonomous round produced a teacher candidate that satisfies the configured acceptance contract."
    elif zoo_partition == "try":
        interpretation = "The round did not clear the main acceptance gate, but it remains informative enough to keep in the try partition."
    else:
        interpretation = "The round failed strongly enough that it should remain only as a rejected reference."

    return LoopJudgement(
        judgement_label=judgement_label,
        accepted=accepted,
        zoo_partition=zoo_partition,
        positive_years=positive_years,
        total_years=total_years,
        positive_rate=positive_rate,
        mean_alpha=mean_alpha,
        median_alpha=median_alpha,
        statistically_worse_years=statistically_worse_years,
        single_year_dominated=single_year_dominated,
        interpretation=interpretation,
    )


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    view = df.loc[:, list(columns)].copy()
    rows: List[List[str]] = []
    headers = list(view.columns)
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            elif isinstance(value, bool):
                values.append("yes" if value else "no")
            elif isinstance(value, pd.Timestamp):
                values.append(value.strftime("%Y-%m-%d"))
            else:
                values.append(str(value))
        rows.append(values)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def execute_teacher_spec(
    spec: TeacherSpec,
    memory_dir: Path,
    report_dir: Path,
    artifact_dir: Path,
) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "models").mkdir(parents=True, exist_ok=True)

    dataset, stock_summary = build_dataset_for_spec(spec)
    dataset.to_csv(artifact_dir / "dataset.csv.gz", index=False, compression="gzip")
    stock_summary.to_csv(artifact_dir / "stock_build_summary.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    thresholds_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    daily_alpha_frames: List[pd.DataFrame] = []
    model_paths: List[Path] = []
    models_by_year: Dict[int, Any] = {}
    scored_frames_by_year: Dict[int, pd.DataFrame] = {}

    for test_year in [year for year in TEST_YEARS if year in dataset["signal_year"].unique()]:
        train_df = dataset[dataset["signal_year"] < test_year].copy()
        test_df = dataset[dataset["signal_year"] == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        model = _fit_model(spec, train_df)
        models_by_year[int(test_year)] = model
        model_path = artifact_dir / "models" / f"{_slugify(spec.title)}_{test_year}.joblib"
        joblib.dump(model, model_path)
        model_paths.append(model_path)

        train_scores = _score_model(model, spec, train_df)
        test_scores = _score_model(model, spec, test_df)
        thresholds = compute_train_thresholds(train_scores, quantiles=TRAIN_BUCKET_QUANTILES)

        train_scored = train_df.copy()
        train_scored["score"] = train_scores
        train_scored["bucket"] = assign_prediction_buckets(train_scores, thresholds)
        test_scored = test_df.copy()
        test_scored["score"] = test_scores
        test_scored["bucket"] = assign_prediction_buckets(test_scores, thresholds)
        scored_frames_by_year[int(test_year)] = test_scored.copy()

        train_diag = _score_diagnostics(train_scored, "score")
        test_diag = _score_diagnostics(test_scored, "score")
        baseline_return = float(test_scored["future_return_5d"].mean())
        q5_df = test_scored[test_scored["bucket"] == 5]
        q5_return = float(q5_df["future_return_5d"].mean()) if len(q5_df) else np.nan
        alpha = q5_return - baseline_return if np.isfinite(q5_return) else np.nan
        daily_alpha = _build_daily_alpha_frame(test_scored).assign(test_year=test_year)
        negative_p = _negative_alpha_pvalue(daily_alpha["daily_alpha"])

        summary_rows.append(
            {
                "test_year": test_year,
                "train_years_used": ", ".join(str(y) for y in sorted(train_df["signal_year"].unique().tolist())),
                "train_sample_count": int(len(train_df)),
                "test_sample_count": int(len(test_df)),
                "train_threshold_q20": float(thresholds[0]),
                "train_threshold_q40": float(thresholds[1]),
                "train_threshold_q60": float(thresholds[2]),
                "train_threshold_q80": float(thresholds[3]),
                "q5_sample_count": int(len(q5_df)),
                "baseline_avg_return": baseline_return,
                "q5_avg_return": q5_return,
                "q5_alpha_vs_baseline": alpha,
                "negative_alpha_pvalue": negative_p,
                "rank_ic_mean": test_diag["rank_ic_mean"],
                "ic_mean": test_diag["ic_mean"],
                "train_rank_ic_mean": train_diag["rank_ic_mean"],
                "year_note": "partial" if test_year == 2026 else "full",
            }
        )
        thresholds_rows.append(
            {
                "test_year": test_year,
                "threshold_source": "training predictions only",
                "train_threshold_q20": float(thresholds[0]),
                "train_threshold_q40": float(thresholds[1]),
                "train_threshold_q60": float(thresholds[2]),
                "train_threshold_q80": float(thresholds[3]),
                "train_score_mean": float(np.mean(train_scores)),
                "train_score_std": float(np.std(train_scores)),
            }
        )
        for split_label, frame in (("train", train_scored), ("test", test_scored)):
            baseline = float(frame["future_return_5d"].mean())
            for bucket in range(1, 6):
                bucket_df = frame[frame["bucket"] == bucket]
                avg_return = float(bucket_df["future_return_5d"].mean()) if len(bucket_df) else np.nan
                bucket_rows.append(
                    {
                        "test_year": test_year,
                        "data_split": split_label,
                        "bucket": bucket,
                        "bucket_label": f"Q{bucket}",
                        "sample_count": int(len(bucket_df)),
                        "sample_share": float(len(bucket_df) / len(frame)) if len(frame) else np.nan,
                        "avg_return": avg_return,
                        "alpha_vs_baseline": avg_return - baseline if np.isfinite(avg_return) else np.nan,
                    }
                )
        prediction_frames.append(
            test_scored[["symbol", "signal_date", "entry_date", "exit_date", "future_return_5d", "score", "bucket"]]
            .assign(test_year=test_year)
            .sort_values(["signal_date", "symbol"])
            .reset_index(drop=True)
        )
        importance_frames.append(_extract_importance(model, spec, test_scored, test_year=test_year))
        daily_alpha_frames.append(daily_alpha)

    summary_df = pd.DataFrame(summary_rows).sort_values("test_year").reset_index(drop=True)
    thresholds_df = pd.DataFrame(thresholds_rows).sort_values("test_year").reset_index(drop=True)
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["test_year", "data_split", "bucket"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(["test_year", "signal_date", "symbol"]).reset_index(drop=True)
    importance_df = pd.concat(importance_frames, ignore_index=True).sort_values(["test_year", "importance_abs"], ascending=[True, False]).reset_index(drop=True)
    daily_alpha_df = pd.concat(daily_alpha_frames, ignore_index=True).sort_values(["test_year", "signal_date"]).reset_index(drop=True)
    judgement = _make_judgement(spec, summary_df)

    summary_df.to_csv(report_dir / "walkforward_yearly_summary.csv", index=False)
    thresholds_df.to_csv(report_dir / "walkforward_thresholds.csv", index=False)
    bucket_df.to_csv(report_dir / "walkforward_bucket_returns.csv", index=False)
    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)
    daily_alpha_df.to_csv(report_dir / "daily_alpha_summary.csv", index=False)

    summary_df.to_csv(artifact_dir / "walkforward_yearly_summary.csv", index=False)
    thresholds_df.to_csv(artifact_dir / "walkforward_thresholds.csv", index=False)
    importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    daily_alpha_df.to_csv(artifact_dir / "daily_alpha_summary.csv", index=False)
    predictions_df.to_csv(artifact_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    nav_result = run_round_nav_backtest(
        round_id=report_dir.name,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
        partition=judgement.zoo_partition,
        status=judgement.judgement_label,
    )
    factor_analysis = run_teacher_factor_analysis(
        spec_title=spec.title,
        model_family=spec.model_family,
        feature_columns=spec.feature_columns,
        models_by_year=models_by_year,
        scored_frames_by_year=scored_frames_by_year,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
    )

    report_lines = [
        f"# {spec.title}",
        "",
        f"- teacher_role: {spec.teacher_role}",
        f"- research_family: `{spec.research_family}`",
        f"- sample_template: `{spec.sample_template}`",
        f"- model_family: `{spec.model_family}`",
        f"- training_backend: `{_training_backend_label(spec)}`",
        f"- target_kind: `{spec.target_kind}`",
        f"- evaluation_contract: `{spec.evaluation_contract}`",
        f"- feature_count: `{len(spec.feature_columns)}`",
        f"- novelty_rationale: {spec.novelty_rationale}",
        "",
        "## Outcome",
        "",
        f"- judgement: `{judgement.judgement_label}`",
        f"- accepted: {'yes' if judgement.accepted else 'no'}",
        f"- zoo_partition: `{judgement.zoo_partition}`",
        f"- positive_years: `{judgement.positive_years}/{judgement.total_years}`",
        f"- mean_alpha: `{judgement.mean_alpha:.6f}`",
        f"- median_alpha: `{judgement.median_alpha:.6f}`",
        f"- statistically_worse_years: `{judgement.statistically_worse_years}`",
        f"- interpretation: {judgement.interpretation}",
        "",
        "## NAV Backtest",
        "",
        f"- nav_final: `{nav_result.final_nav:.4f}`",
        f"- nav_total_return: `{nav_result.total_return:.2%}`",
        f"- nav_cagr: `{nav_result.cagr:.2%}`",
        f"- nav_max_drawdown: `{nav_result.max_drawdown:.2%}`",
        f"- hs300_total_return: `{nav_result.hs300_total_return:.2%}`",
        f"- excess_total_return: `{nav_result.excess_total_return:.2%}`",
        f"- nav_positive_years: `{nav_result.positive_years}/{nav_result.total_years}`",
        f"- nav_curve_path: `{_relative(Path(nav_result.plot_path))}`",
        "",
        "## Detailed Factor Analysis",
        "",
        f"- factor_analysis_report: `{_relative(report_dir / 'FACTOR_ANALYSIS_REPORT.md')}`",
        f"- local_explainability_method: `{factor_analysis.summary.get('local_explainability_method', 'unknown')}`",
        f"- top_factor_features: `{', '.join(row['feature'] for row in factor_analysis.summary.get('top_global_features', [])[:6])}`",
        "",
        "## Walk-Forward Summary",
        "",
        _markdown_table(
            summary_df,
            [
                "test_year",
                "train_years_used",
                "test_sample_count",
                "q5_sample_count",
                "baseline_avg_return",
                "q5_avg_return",
                "q5_alpha_vs_baseline",
                "negative_alpha_pvalue",
                "rank_ic_mean",
            ],
        ),
        "",
    ]
    _write_text(report_dir / "EXECUTION_REPORT.md", "\n".join(report_lines))

    return {
        "summary_df": summary_df,
        "thresholds_df": thresholds_df,
        "bucket_df": bucket_df,
        "importance_df": importance_df,
        "daily_alpha_df": daily_alpha_df,
        "nav_result": nav_result,
        "factor_analysis": {
            "summary": factor_analysis.summary,
            "artifact_paths": factor_analysis.artifact_paths,
        },
        "judgement": judgement,
        "model_paths": model_paths,
    }


def write_memory_for_round(
    spec: TeacherSpec,
    round_id: str,
    round_index: int,
    report_dir: Path,
    artifact_dir: Path,
    result: Dict[str, Any],
    novelty: NoveltyResult,
) -> List[Dict[str, Any]]:
    store = MemoryStore(MEMORY_DIR)
    summary_df = result["summary_df"]
    judgement: LoopJudgement = result["judgement"]
    model_paths: List[Path] = result["model_paths"]
    nav_result = result["nav_result"]
    factor_analysis = result.get("factor_analysis", {})
    factor_summary = factor_analysis.get("summary", {}) if isinstance(factor_analysis, dict) else {}
    factor_artifact_paths = factor_analysis.get("artifact_paths", []) if isinstance(factor_analysis, dict) else []
    artifact_refs = [
        _relative(report_dir / "EXECUTION_REPORT.md"),
        _relative(report_dir / "selected_spec.json"),
        _relative(report_dir / "proposal.json"),
        _relative(report_dir / "novelty_report.json"),
        _relative(report_dir / "walkforward_yearly_summary.csv"),
        _relative(report_dir / "walkforward_thresholds.csv"),
        _relative(report_dir / "walkforward_bucket_returns.csv"),
        _relative(report_dir / "feature_importance.csv"),
        _relative(report_dir / "daily_alpha_summary.csv"),
        _relative(report_dir / "nav_curve.csv"),
        _relative(report_dir / "nav_yearly_returns.csv"),
        _relative(report_dir / "nav_summary.json"),
        _relative(report_dir / "nav_curve.png"),
        _relative(report_dir / "prompt.json"),
        _relative(artifact_dir / "dataset.csv.gz"),
        _relative(artifact_dir / "stock_build_summary.csv"),
        _relative(artifact_dir / "walkforward_yearly_summary.csv"),
        _relative(artifact_dir / "walkforward_thresholds.csv"),
        _relative(artifact_dir / "feature_importance.csv"),
        _relative(artifact_dir / "daily_alpha_summary.csv"),
        _relative(artifact_dir / "test_predictions.csv.gz"),
        _relative(artifact_dir / "nav_curve.csv"),
        _relative(artifact_dir / "nav_yearly_returns.csv"),
        _relative(artifact_dir / "nav_summary.json"),
        _relative(artifact_dir / "nav_curve.png"),
    ]
    artifact_refs.extend(_relative(path) for path in model_paths)
    artifact_refs.extend(_relative(Path(path)) for path in factor_artifact_paths)

    common_kwargs = {
        "source_label": f"teacher_loop_{round_id}",
        "source_type": "autonomous_loop",
        "created_by": "quant_toolkit.teacher_loop.loop",
    }
    teacher_title = f"{round_id} {spec.title}"
    if not teacher_title.lower().endswith("teacher"):
        teacher_title = f"{teacher_title} teacher"

    hypothesis = store.create_item(
        item_type="hypothesis",
        title=f"{round_id} {spec.title} hypothesis",
        summary=spec.hypothesis,
        status="active",
        payload={
            "hypothesis_statement": spec.hypothesis,
            "rationale": spec.novelty_rationale,
            "expected_signal": f"{spec.evaluation_contract} under {spec.sample_template}",
            "linked_experiment_ids": [],
            "feature_columns": spec.feature_columns,
            "sample_template": spec.sample_template,
            "research_family": spec.research_family,
            "round_id": round_id,
            "round_index": round_index,
            "artifact_refs": artifact_refs,
            "factor_analysis_summary": factor_summary,
        },
        tags=["teacher_loop", spec.research_family, spec.sample_template, spec.model_family, f"round_{round_index:03d}"],
        linked_ids=[],
        **common_kwargs,
    )
    experiment = store.create_item(
        item_type="experiment",
        title=f"{round_id} {spec.title} execution",
        summary=f"Autonomous teacher-loop round {round_index} execution for {spec.title}.",
        status="completed",
        payload={
            "hypothesis_ids": [hypothesis.item["hypothesis_id"]],
            "objective": "Autonomously propose, execute, and evaluate one new teacher-construction experiment.",
            "design": json.dumps(spec.to_dict(), ensure_ascii=False),
            "conclusion": judgement.judgement_label,
            "execution_status": "completed",
            "result_summary": (
                f"positive_years={judgement.positive_years}/{judgement.total_years}, "
                f"mean_alpha={judgement.mean_alpha:.6f}, median_alpha={judgement.median_alpha:.6f}, "
                f"zoo_partition={judgement.zoo_partition}, "
                f"nav_cagr={nav_result.cagr:.4f}, nav_mdd={nav_result.max_drawdown:.4f}"
            ),
            "artifact_refs": artifact_refs,
            "round_id": round_id,
            "round_index": round_index,
            "novelty_report": {
                "too_similar": novelty.too_similar,
                "max_feature_overlap": novelty.max_feature_overlap,
                "nearest_teachers": novelty.nearest_teachers,
            },
            "spec": spec.to_dict(),
            "factor_analysis_summary": factor_summary,
        },
        tags=["teacher_loop", "experiment", spec.research_family, spec.sample_template, spec.model_family],
        linked_ids=[hypothesis.item["memory_id"]],
        **common_kwargs,
    )
    lesson = store.create_item(
        item_type="research_lesson",
        title=f"{round_id} {spec.title} lesson",
        summary=judgement.interpretation,
        status="completed",
        payload={
            "lesson_summary": judgement.interpretation,
            "recommended_action": (
                "Promote this route into the active zoo partition and avoid duplicating it too closely in the next round."
                if judgement.accepted
                else "Keep this route in try/rejected memory and use it as a novelty constraint for the next round."
            ),
            "applies_to": ["teacher_loop", spec.research_family, spec.sample_template],
            "artifact_refs": artifact_refs,
            "round_id": round_id,
            "round_index": round_index,
            "nav_snapshot": {
                "cagr": nav_result.cagr,
                "max_drawdown": nav_result.max_drawdown,
                "positive_years": nav_result.positive_years,
                "total_years": nav_result.total_years,
            },
            "factor_analysis_summary": factor_summary,
        },
        tags=["teacher_loop", "research_lesson", spec.research_family, judgement.zoo_partition],
        linked_ids=[hypothesis.item["memory_id"], experiment.item["memory_id"]],
        **common_kwargs,
    )
    teacher = store.create_item(
        item_type="teacher_model",
        title=teacher_title,
        summary=f"Autonomous teacher-loop output for {spec.title}.",
        status="completed" if judgement.zoo_partition in {"main", "try"} else "rejected",
        payload={
            "model_family": spec.model_family,
            "intended_use": spec.teacher_role,
            "training_data_refs": [
                _relative(STOCK_DATA_DIR) + "/*.csv",
                _relative(report_dir / "walkforward_yearly_summary.csv"),
                _relative(report_dir / "walkforward_thresholds.csv"),
            ],
            "training_status": "completed",
            "accepted_as_teacher": judgement.accepted,
            "artifact_refs": artifact_refs,
            "zoo_partition": judgement.zoo_partition,
            "teacher_role": spec.teacher_role,
            "research_family": spec.research_family,
            "sample_template": spec.sample_template,
            "target_kind": spec.target_kind,
            "evaluation_contract": spec.evaluation_contract,
            "feature_columns": spec.feature_columns,
            "round_id": round_id,
            "round_index": round_index,
            "positive_years": judgement.positive_years,
            "total_years": judgement.total_years,
            "positive_rate": judgement.positive_rate,
            "mean_alpha": judgement.mean_alpha,
            "median_alpha": judgement.median_alpha,
            "statistically_worse_years": judgement.statistically_worse_years,
            "nav_final_nav": nav_result.final_nav,
            "nav_total_return": nav_result.total_return,
            "nav_cagr": nav_result.cagr,
            "nav_max_drawdown": nav_result.max_drawdown,
            "nav_hs300_total_return": nav_result.hs300_total_return,
            "nav_excess_total_return": nav_result.excess_total_return,
            "nav_positive_years": nav_result.positive_years,
            "nav_total_years": nav_result.total_years,
            "nav_yearly_returns": {str(year): ret for year, ret in nav_result.yearly_returns.items()},
            "factor_analysis_summary": factor_summary,
            "novelty_report": {
                "too_similar": novelty.too_similar,
                "max_feature_overlap": novelty.max_feature_overlap,
                "nearest_teachers": novelty.nearest_teachers,
            },
        },
        tags=["teacher_loop", "teacher_model", judgement.zoo_partition, spec.research_family, spec.sample_template, spec.model_family],
        linked_ids=[hypothesis.item["memory_id"], experiment.item["memory_id"], lesson.item["memory_id"]],
        **common_kwargs,
    )

    feature_registry = _load_json(MEMORY_DIR / "indexes" / "feature_registry.json")
    feature_meta = {row["feature_name"]: row for row in feature_registry.get("features", [])}
    factor_created_items: List[Dict[str, Any]] = []
    for feature_row in list(factor_summary.get("top_global_features", []))[:8]:
        feature_name = str(feature_row.get("feature", "")).strip()
        if not feature_name:
            continue
        meta = feature_meta.get(feature_name, {})
        q25 = feature_row.get("selected_q25")
        q75 = feature_row.get("selected_q75")
        has_band = isinstance(q25, (int, float)) and isinstance(q75, (int, float))
        band_text = f"[{float(q25):.4f}, {float(q75):.4f}]" if has_band else "na"
        factor_item = store.create_item(
            item_type="factor_card",
            title=f"{round_id} {feature_name} factor card",
            summary=(
                f"{feature_name} is a leading driver in {round_id}; "
                f"preferred_direction={feature_row.get('preferred_direction')}, preferred_band={band_text}, "
                f"shape={feature_row.get('shape_hint')}"
            ),
            status="completed",
            payload={
                "factor_name": feature_name,
                "factor_definition": meta.get("formula", feature_name),
                "category": meta.get("category"),
                "lookback_window": meta.get("lookback_window"),
                "required_raw_fields": meta.get("required_raw_fields", []),
                "implemented_in": meta.get("implemented_in"),
                "leakage_risk": meta.get("leakage_risk"),
                "round_id": round_id,
                "round_index": round_index,
                "research_family": spec.research_family,
                "sample_template": spec.sample_template,
                "teacher_model_id": teacher.item["model_id"],
                "teacher_memory_id": teacher.item["memory_id"],
                "preferred_direction": feature_row.get("preferred_direction"),
                "selected_q25": feature_row.get("selected_q25"),
                "selected_median": feature_row.get("selected_median"),
                "selected_q75": feature_row.get("selected_q75"),
                "pool_median": feature_row.get("pool_median"),
                "mean_abs_contribution": feature_row.get("mean_abs_contribution"),
                "mean_contribution": feature_row.get("mean_contribution"),
                "band_effect_size": feature_row.get("band_effect_size"),
                "best_bin_return": feature_row.get("best_bin_return"),
                "shape_hint": feature_row.get("shape_hint"),
                "artifact_refs": artifact_refs,
            },
            tags=["teacher_loop", "factor_card", spec.research_family, spec.sample_template, feature_name],
            linked_ids=[teacher.item["memory_id"], experiment.item["memory_id"], lesson.item["memory_id"]],
            **common_kwargs,
        )
        factor_created_items.append(
            {"item_type": factor_item.item["item_type"], "memory_id": factor_item.item["memory_id"], "path": str(factor_item.path)}
        )

    return [
        {"item_type": hypothesis.item["item_type"], "memory_id": hypothesis.item["memory_id"], "path": str(hypothesis.path)},
        {"item_type": experiment.item["item_type"], "memory_id": experiment.item["memory_id"], "path": str(experiment.path)},
        {"item_type": lesson.item["item_type"], "memory_id": lesson.item["memory_id"], "path": str(lesson.path)},
        {"item_type": teacher.item["item_type"], "memory_id": teacher.item["memory_id"], "path": str(teacher.path)},
        *factor_created_items,
    ]


def launch_next_round(*, model_name: Optional[str] = None) -> Dict[str, Any]:
    ensure_teacher_loop_indexes(MEMORY_DIR)
    round_id, round_index = next_round_id(MEMORY_DIR)
    round_dir = REPORT_ROOT / round_id
    artifact_dir = ARTIFACT_ROOT / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    state = load_loop_state(MEMORY_DIR)
    state["rounds_launched"] = round_index
    state["active_round_id"] = round_id
    save_loop_state(MEMORY_DIR, state)
    append_loop_manifest(
        MEMORY_DIR,
        {
            "created_at": _utc_timestamp(),
            "round_id": round_id,
            "round_index": round_index,
            "phase": "launched",
            "status": "running",
            "report_dir": _relative(round_dir),
        },
    )

    try:
        spec, novelty, api_raw, proposal_payload = propose_next_spec(MEMORY_DIR, round_dir, model_name=model_name)
        _write_json(round_dir / "selected_spec.json", spec.to_dict())
        _write_json(
            round_dir / "novelty_report.json",
            {
                "too_similar": novelty.too_similar,
                "max_feature_overlap": novelty.max_feature_overlap,
                "novelty_reason": novelty.novelty_reason,
                "nearest_teachers": novelty.nearest_teachers,
            },
        )
        result = execute_teacher_spec(spec, MEMORY_DIR, round_dir, artifact_dir)
        created_items = write_memory_for_round(spec, round_id, round_index, round_dir, artifact_dir, result, novelty)
        build_teacher_zoo_index(MEMORY_DIR)

        judgement: LoopJudgement = result["judgement"]
        _write_text(
            round_dir / "MEMORY_WRITEBACK.md",
            "\n".join(
                [
                    f"# {round_id} Memory Writeback",
                    "",
                    f"- zoo_partition: `{judgement.zoo_partition}`",
                    f"- accepted: {'yes' if judgement.accepted else 'no'}",
                    f"- created_items: {json.dumps(created_items, ensure_ascii=False)}",
                ]
            ),
        )

        state = load_loop_state(MEMORY_DIR)
        state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
        state["active_round_id"] = None
        save_loop_state(MEMORY_DIR, state)
        append_loop_manifest(
            MEMORY_DIR,
            {
                "created_at": _utc_timestamp(),
                "round_id": round_id,
                "round_index": round_index,
                "phase": "completed",
                "status": judgement.judgement_label,
                "zoo_partition": judgement.zoo_partition,
                "spec_title": spec.title,
                "report_dir": _relative(round_dir),
                "teacher_memory_id": next(item["memory_id"] for item in created_items if item["item_type"] == "teacher_model"),
            },
        )

        return {
            "round_id": round_id,
            "round_index": round_index,
            "spec": spec.to_dict(),
            "novelty": asdict(novelty),
            "judgement": asdict(judgement),
            "created_items": created_items,
            "report_dir": _relative(round_dir),
        }
    except Exception as exc:
        state = load_loop_state(MEMORY_DIR)
        state["active_round_id"] = None
        save_loop_state(MEMORY_DIR, state)
        append_loop_manifest(
            MEMORY_DIR,
            {
                "created_at": _utc_timestamp(),
                "round_id": round_id,
                "round_index": round_index,
                "phase": "failed",
                "status": "error",
                "report_dir": _relative(round_dir),
                "error": repr(exc),
            },
        )
        raise


def launch_until_target(*, model_name: Optional[str] = None, max_new_rounds: int = 1) -> List[Dict[str, Any]]:
    ensure_teacher_loop_indexes(MEMORY_DIR)
    results: List[Dict[str, Any]] = []
    for _ in range(max_new_rounds):
        state = load_loop_state(MEMORY_DIR)
        if int(state.get("rounds_completed", 0)) >= int(state.get("target_rounds", 10)):
            break
        results.append(launch_next_round(model_name=model_name))
    return results
