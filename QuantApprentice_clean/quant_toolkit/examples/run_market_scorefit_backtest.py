#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from quant_toolkit._paths import env_path, project_root
from quant_toolkit.apprentice_loop import ApprenticeReplayConfig
from quant_toolkit.apprentice_loop import replay as replay_mod
from quant_toolkit.pilot2.walkforward_utils import assign_prediction_buckets
from quant_toolkit.teacher_loop import loop as teacher_loop_mod


PROJECT_ROOT = env_path("QUANT_PROJECT_ROOT", project_root())
REPORT_ROOT = env_path("APPRENTICE_REPORT_ROOT", PROJECT_ROOT / "reports" / "apprentice_loop")
TEACHER_REPORT_ROOT = env_path("TEACHER_LOOP_REPORT_ROOT", PROJECT_ROOT / "reports" / "teacher_loop")
MEMORY_ROOT = env_path("QUANT_MEMORY_DIR", PROJECT_ROOT / "research_memory")
TEACHER_ARTIFACT_ROOT = env_path("TEACHER_LOOP_ARTIFACT_ROOT", MEMORY_ROOT / "artifacts" / "teacher_loop")


@dataclass
class TeacherContext:
    round_id: str
    source_round_id: str
    title: str
    family: str
    hypothesis: str
    sample_template: str
    sample_template_desc: str
    style_label: str
    feature_columns: List[str]
    lesson_prompt: Dict[str, Any]
    top_feature_cues: List[Dict[str, Any]]
    walkforward_final_nav: float
    walkforward_q5_mean_return: float
    walkforward_q5_win_rate: float
    walkforward_q5_alpha_mean: float
    walkforward_rank_ic_mean: float
    walkforward_baseline_mean_return: float
    model_path: str
    threshold_path: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clip_score_0_100(value: float) -> float:
    return float(max(0.0, min(100.0, float(value))))


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batched market-wide scorefit backtest with final scoped lessons")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--final-lesson-state-json", required=True)
    parser.add_argument("--disable-final-lessons", action="store_true")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--daily-sample-size", type=int, default=40)
    parser.add_argument("--daily-top-pct", type=float, default=0.10)
    parser.add_argument("--lock-days", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20250701)
    parser.add_argument("--llm-decision-seed", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--prompt-recipe", default="standard")
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-max-tokens", type=int, default=384)
    parser.add_argument("--api-parallel-workers", type=int, default=256)
    parser.add_argument("--api-failed-rerun-rounds", type=int, default=4)
    parser.add_argument("--api-request-max-retries", type=int, default=1)
    parser.add_argument("--private-reasoning-target-tokens", type=int, default=0)
    parser.add_argument("--private-reasoning-max-tokens-hint", type=int, default=0)
    parser.add_argument("--force-local-qwen-no-thinking", action="store_true")
    parser.add_argument("--enable-local-qwen-thinking", action="store_true")
    parser.add_argument("--prompt-digest-max-chars", type=int, default=3200)
    parser.add_argument("--prompt-top-feature-count", type=int, default=5)
    parser.add_argument("--prompt-max-meta-rules", type=int, default=4)
    parser.add_argument("--prompt-max-scoring-notes", type=int, default=4)
    parser.add_argument("--report-subdir", default="market_scorefit_backtest_2025")
    parser.add_argument("--digest-output-path", default="")
    parser.add_argument("--teacher-profile-start-date", default="2023-01-01")
    parser.add_argument("--teacher-profile-end-date", default="2026-12-31")
    return parser


def _load_selection(selection_json: Path) -> Dict[str, Any]:
    payload = _load_json(selection_json)
    frozen = [str(x).strip() for x in list(payload.get("frozen_round_ids") or []) if str(x).strip()]
    positive = [str(x).strip() for x in list(payload.get("positive_round_ids") or []) if str(x).strip()]
    negative = [str(x).strip() for x in list(payload.get("negative_round_ids") or []) if str(x).strip()]
    if not frozen:
        raise ValueError(f"selection json missing frozen_round_ids: {selection_json}")
    scope_specs: List[Dict[str, str]] = []
    for idx, frozen_round_id in enumerate(frozen):
        if idx < len(positive):
            source_round_id = positive[idx]
        else:
            source_round_id = replay_mod._source_round_id_for_round(frozen_round_id)
        scope_specs.append(
            {
                "final_round_id": frozen_round_id,
                "source_round_id": source_round_id,
            }
        )
    return {
        "frozen_round_ids": frozen,
        "positive_round_ids": positive,
        "negative_round_ids": negative,
        "scope_specs": scope_specs,
        "raw": payload,
    }


def _load_final_lesson_state(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid final lesson state json: {path}")
    scopes = list(payload.get("teacher_scopes") or [])
    if not scopes:
        raise ValueError(f"final lesson state has no teacher_scopes: {path}")
    return payload


def _sample_template_desc(template: str) -> str:
    return str(teacher_loop_mod.SUPPORTED_SAMPLE_TEMPLATES.get(template, template)).strip()


def _style_label(sample_template: str, family: str, hypothesis: str) -> str:
    template = str(sample_template or "").strip().lower()
    family_text = str(family or "").strip().lower()
    hypo_text = str(hypothesis or "").strip().lower()
    merged = " ".join([template, family_text, hypo_text])
    if "trend_breakout_pool" in merged or "breakout" in merged:
        return "breakout continuation"
    if "hard_threshold_reversal_gate" in merged:
        return "hard-gated oversold reversal"
    if "weak_state_reversal_pool" in merged:
        return "weak-state reversal"
    if "pullback" in merged and "reversal" in merged:
        return "pullback rebound inside trend"
    if "pullback" in merged:
        return "trend pullback continuation"
    if "reversal" in merged:
        return "reversal / rebound"
    return "mixed regime ranking"


def _compact_feature_cues(factor_summary: Mapping[str, Any], top_n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(factor_summary.get("top_global_features") or [])[:top_n]:
        feature = str(row.get("feature", "")).strip()
        if not feature:
            continue
        out.append(
            {
                "feature": feature,
                "preferred_direction": str(row.get("preferred_direction", "")).strip(),
                "shape_hint": str(row.get("shape_hint", "")).strip(),
            }
        )
    return out


def _compact_lesson_for_prompt(
    lesson_json: Mapping[str, Any],
    *,
    max_meta_rules: int,
    max_scoring_notes: int,
) -> Dict[str, Any]:
    items = {}
    for item_id, payload in dict(lesson_json.get("items") or {}).items():
        items[str(item_id)] = {
            "title": str(payload.get("title", "")).strip(),
            "role": str(payload.get("role", "")).strip(),
            "score_range": str(payload.get("score_range", "")).strip(),
            "signals_to_check": [str(x).strip() for x in list(payload.get("signals_to_check") or []) if str(x).strip()],
            "rule": str(payload.get("rule", "")).strip(),
            "interaction_note": str(payload.get("interaction_note", "")).strip(),
        }
    return {
        "schema": str(lesson_json.get("schema", "")).strip(),
        "lesson_name": str(lesson_json.get("lesson_name", "")).strip(),
        "teacher_scope": dict(lesson_json.get("teacher_scope") or {}),
        "items": items,
        "meta_rules": [str(x).strip() for x in list(lesson_json.get("meta_rules") or []) if str(x).strip()][:max_meta_rules],
        "scoring_notes": [str(x).strip() for x in list(lesson_json.get("scoring_notes") or []) if str(x).strip()][:max_scoring_notes],
    }


def _lesson_signal_features(lesson_json: Mapping[str, Any]) -> List[str]:
    feats: List[str] = []
    for payload in dict(lesson_json.get("items") or {}).values():
        for feat in list(payload.get("signals_to_check") or []):
            text = str(feat).strip()
            if text and text not in feats:
                feats.append(text)
    return feats


def _load_thresholds(round_id: str) -> np.ndarray:
    threshold_path = TEACHER_REPORT_ROOT / round_id / "walkforward_thresholds.csv"
    df = pd.read_csv(threshold_path)
    first = df.iloc[0]
    return np.array(
        [
            float(first["train_threshold_q20"]),
            float(first["train_threshold_q40"]),
            float(first["train_threshold_q60"]),
            float(first["train_threshold_q80"]),
        ],
        dtype=float,
    )


def _load_teacher_model(round_id: str):
    model_path = TEACHER_ARTIFACT_ROOT / round_id / "models" / f"{round_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"missing teacher model: {model_path}")
    return joblib.load(model_path)


def _teacher_profile(round_id: str, start_date: str, end_date: str) -> Dict[str, float]:
    pred_df = replay_mod._load_teacher_predictions(
        round_id,
        start_date=start_date,
        end_date=end_date,
    )
    yearly_path = TEACHER_REPORT_ROOT / round_id / "walkforward_yearly_summary.csv"
    yearly = pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()
    q5 = pred_df[pred_df["bucket"] == 5].copy()
    baseline_mean_return = float(pred_df["future_return_5d"].mean()) if not pred_df.empty else 0.0
    q5_mean_return = float(q5["future_return_5d"].mean()) if not q5.empty else 0.0
    q5_win_rate = float((q5["future_return_5d"] > 0).mean()) if not q5.empty else 0.0
    q5_alpha_mean = float(yearly["q5_alpha_vs_baseline"].mean()) if not yearly.empty else 0.0
    rank_ic_mean = float(yearly["rank_ic_mean"].mean()) if not yearly.empty else 0.0
    return {
        "walkforward_baseline_mean_return": baseline_mean_return,
        "walkforward_q5_mean_return": q5_mean_return,
        "walkforward_q5_win_rate": q5_win_rate,
        "walkforward_q5_alpha_mean": q5_alpha_mean,
        "walkforward_rank_ic_mean": rank_ic_mean,
    }


def _collect_teacher_contexts(
    *,
    selection: Mapping[str, Any],
    final_lesson_state: Mapping[str, Any],
    prompt_top_feature_count: int,
    prompt_max_meta_rules: int,
    prompt_max_scoring_notes: int,
    teacher_profile_start_date: str,
    teacher_profile_end_date: str,
) -> List[TeacherContext]:
    out: List[TeacherContext] = []
    for scope_spec in list(selection["scope_specs"]):
        round_id = str(scope_spec["final_round_id"]).strip()
        source_round_id = str(scope_spec["source_round_id"]).strip()
        spec = replay_mod._load_teacher_spec(round_id)
        nav_summary = replay_mod._load_teacher_nav_summary(round_id)
        factor_summary = replay_mod._load_teacher_factor_analysis_summary(round_id)
        profile = _teacher_profile(round_id, teacher_profile_start_date, teacher_profile_end_date)
        scope_entry = replay_mod._find_teacher_scope_entry(final_lesson_state, round_id)
        if not scope_entry:
            raise ValueError(f"final lesson state missing scope for {round_id}")
        lesson_json = dict(scope_entry.get("scorefit_lesson_json") or {})
        if not lesson_json:
            raise ValueError(f"final lesson state has empty lesson json for {round_id}")
        out.append(
            TeacherContext(
                round_id=round_id,
                source_round_id=source_round_id,
                title=str(spec.get("title", "")).strip(),
                family=str(spec.get("research_family", "")).strip(),
                hypothesis=str(spec.get("hypothesis", "")).strip(),
                sample_template=str(spec.get("sample_template", "")).strip(),
                sample_template_desc=_sample_template_desc(str(spec.get("sample_template", "")).strip()),
                style_label=_style_label(
                    str(spec.get("sample_template", "")).strip(),
                    str(spec.get("research_family", "")).strip(),
                    str(spec.get("hypothesis", "")).strip(),
                ),
                feature_columns=[str(x).strip() for x in list(spec.get("feature_columns") or []) if str(x).strip()],
                lesson_prompt=_compact_lesson_for_prompt(
                    lesson_json,
                    max_meta_rules=prompt_max_meta_rules,
                    max_scoring_notes=prompt_max_scoring_notes,
                ),
                top_feature_cues=_compact_feature_cues(
                    factor_summary,
                    prompt_top_feature_count,
                ),
                walkforward_final_nav=float(nav_summary.get("final_nav", 0.0) or 0.0),
                walkforward_q5_mean_return=float(profile["walkforward_q5_mean_return"]),
                walkforward_q5_win_rate=float(profile["walkforward_q5_win_rate"]),
                walkforward_q5_alpha_mean=float(profile["walkforward_q5_alpha_mean"]),
                walkforward_rank_ic_mean=float(profile["walkforward_rank_ic_mean"]),
                walkforward_baseline_mean_return=float(profile["walkforward_baseline_mean_return"]),
                model_path=str(TEACHER_ARTIFACT_ROOT / round_id / "models" / f"{round_id}.joblib"),
                threshold_path=str(TEACHER_REPORT_ROOT / round_id / "walkforward_thresholds.csv"),
            )
        )
    return out


def _collect_negative_contexts(
    *,
    selection: Mapping[str, Any],
    prompt_top_feature_count: int,
    teacher_profile_start_date: str,
    teacher_profile_end_date: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_id in list(selection.get("negative_round_ids") or []):
        spec = replay_mod._load_teacher_spec(round_id)
        nav_summary = replay_mod._load_teacher_nav_summary(round_id)
        factor_summary = replay_mod._load_teacher_factor_analysis_summary(round_id)
        profile = _teacher_profile(round_id, teacher_profile_start_date, teacher_profile_end_date)
        rows.append(
            {
                "round_id": round_id,
                "title": str(spec.get("title", "")).strip(),
                "family": str(spec.get("research_family", "")).strip(),
                "sample_template": str(spec.get("sample_template", "")).strip(),
                "sample_template_desc": _sample_template_desc(str(spec.get("sample_template", "")).strip()),
                "style_label": _style_label(
                    str(spec.get("sample_template", "")).strip(),
                    str(spec.get("research_family", "")).strip(),
                    str(spec.get("hypothesis", "")).strip(),
                ),
                "hypothesis": str(spec.get("hypothesis", "")).strip(),
                "walkforward_final_nav": float(nav_summary.get("final_nav", 0.0) or 0.0),
                "walkforward_q5_alpha_mean": float(profile["walkforward_q5_alpha_mean"]),
                "walkforward_rank_ic_mean": float(profile["walkforward_rank_ic_mean"]),
                "top_feature_cues": _compact_feature_cues(factor_summary, prompt_top_feature_count),
            }
        )
    return rows


def _aggregate_feature_frequency(teachers: Sequence[TeacherContext]) -> List[str]:
    counts: Dict[str, int] = {}
    for teacher in teachers:
        for cue in teacher.top_feature_cues:
            feature = str(cue.get("feature", "")).strip()
            if not feature:
                continue
            counts[feature] = counts.get(feature, 0) + 1
    return [feature for feature, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]


def _build_simplified_hypval_digest(
    *,
    teacher_contexts: Sequence[TeacherContext],
    negative_contexts: Sequence[Mapping[str, Any]],
) -> str:
    repeated_features = _aggregate_feature_frequency(teacher_contexts)
    lines: List[str] = [
        "Simplified Hypothesis-Validation Loop Digest",
        "",
        "1. Outer-loop takeaway",
        "- The research loop tested multiple signal clusters instead of assuming one universal ranking rule.",
        "- Some clusters can be stratified clearly and repeatedly by local feature structure; others remain noisy and only support weak transfer.",
        "- When a new market signal resembles a teacher's cluster or style family, that teacher's lesson is useful. When it only partially matches, reuse the idea softly instead of forcing exact rule copying.",
        "",
        "2. Clusters that showed clearer stratification and produced usable teachers",
    ]
    for teacher in teacher_contexts:
        cue_text = ", ".join(
            f"{cue['feature']}({cue['preferred_direction'] or 'context-dependent'})"
            for cue in teacher.top_feature_cues[:4]
            if cue.get("feature")
        ) or "no stable cue summary"
        lines.append(
            "- "
            f"{teacher.round_id}: {teacher.style_label}; cluster={teacher.sample_template}; "
            f"family={teacher.family}; q5_mean_return={teacher.walkforward_q5_mean_return:+.4%}; "
            f"q5_win_rate={teacher.walkforward_q5_win_rate:.2%}; "
            f"q5_alpha_mean={teacher.walkforward_q5_alpha_mean:+.4%}; "
            f"important cues={cue_text}"
        )
    lines.extend(
        [
            "",
            "3. Clusters that looked harder or less stable to reuse",
        ]
    )
    if negative_contexts:
        for row in negative_contexts:
            cue_text = ", ".join(
                f"{cue['feature']}({cue['preferred_direction'] or 'context-dependent'})"
                for cue in list(row.get("top_feature_cues") or [])[:3]
                if cue.get("feature")
            ) or "no stable cue summary"
            lines.append(
                "- "
                f"{row['round_id']}: style={row['style_label']}; cluster={row['sample_template']}; "
                f"final_nav={float(row.get('walkforward_final_nav', 0.0)):.4f}; "
                f"q5_alpha_mean={float(row.get('walkforward_q5_alpha_mean', 0.0)):+.4%}; "
                f"rank_ic_mean={float(row.get('walkforward_rank_ic_mean', 0.0)):+.4f}; "
                f"treat as noisy or fragile unless a new signal strongly matches the same setting; cues={cue_text}"
            )
    else:
        lines.append("- No negative teacher shortlist was provided.")
    lines.extend(
        [
            "",
            "4. Cross-teacher background that kept recurring",
            "- Repeatedly useful families included volatility regime, KDJ slope / divergence, distance to recent highs or moving averages, volume acceleration, and amplitude / candlestick pressure.",
            "- These recurring families help with transfer, but the exact useful subrange is teacher-local and belongs in the final lessons rather than in this digest.",
            "- If a new signal is far from every teacher's basic filter, score it conservatively and treat teacher lessons as analogies instead of exact templates.",
            "",
            "5. Usage rule",
            "- This digest is background only. It explains where the outer loop found transferable structure and where it found noisy pools.",
            "- Do not turn this digest into hard interval rules by itself. Use the final lessons for teacher-local scoring detail.",
        ]
    )
    if repeated_features:
        lines.extend(
            [
                "",
                "6. Repeated feature families across successful teachers",
                "- " + ", ".join(repeated_features),
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _build_teacher_prompt_cards(teacher_contexts: Sequence[TeacherContext]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for teacher in teacher_contexts:
        cards.append(
            {
                "round_id": teacher.round_id,
                "source_round_id": teacher.source_round_id,
                "title": teacher.title,
                "style_family": teacher.style_label,
                "research_family": teacher.family,
                "signal_cluster": teacher.sample_template,
                "basic_filter": teacher.sample_template_desc,
                "hypothesis": teacher.hypothesis,
                "walkforward_q5_mean_return": round(float(teacher.walkforward_q5_mean_return), 6),
                "walkforward_q5_win_rate": round(float(teacher.walkforward_q5_win_rate), 6),
                "walkforward_q5_alpha_mean": round(float(teacher.walkforward_q5_alpha_mean), 6),
                "walkforward_rank_ic_mean": round(float(teacher.walkforward_rank_ic_mean), 6),
                "important_feature_cues": teacher.top_feature_cues,
            }
        )
    return cards


def _prompt_feature_union(
    teacher_contexts: Sequence[TeacherContext],
    available_columns: Sequence[str],
    *,
    include_lesson_features: bool,
) -> Tuple[List[str], List[str]]:
    feature_set: List[str] = []
    for teacher in teacher_contexts:
        for feature in teacher.feature_columns:
            if feature not in feature_set:
                feature_set.append(feature)
        if include_lesson_features:
            for feature in _lesson_signal_features(teacher.lesson_prompt):
                if feature not in feature_set:
                    feature_set.append(feature)
    available = set(str(col) for col in available_columns)
    used = [feature for feature in feature_set if feature in available]
    missing = [feature for feature in feature_set if feature not in available]
    return used, missing


def _sample_market_signals(
    market_df: pd.DataFrame,
    *,
    daily_sample_size: int,
    sample_seed: int,
    max_days: int,
) -> pd.DataFrame:
    day_frames: List[pd.DataFrame] = []
    grouped = market_df.groupby("signal_date", sort=True)
    for day_index, (signal_date, day_df) in enumerate(grouped, start=1):
        if max_days > 0 and day_index > max_days:
            break
        day_df = day_df.sort_values(["symbol"], kind="mergesort").reset_index(drop=True)
        take = min(len(day_df), int(daily_sample_size))
        if take <= 0:
            continue
        if len(day_df) <= take:
            picked = day_df.copy()
        else:
            rng = np.random.default_rng(int(sample_seed) * 10000 + day_index)
            idx = np.sort(rng.choice(len(day_df), size=take, replace=False))
            picked = day_df.iloc[idx].copy()
        picked["_sample_day_index"] = int(day_index)
        picked["_sample_count_for_day"] = int(len(picked))
        picked["_market_day_count"] = int(len(day_df))
        picked["_signal_date_text"] = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
        day_frames.append(picked)
    if not day_frames:
        return market_df.iloc[0:0].copy()
    out = pd.concat(day_frames, ignore_index=True)
    out = out.sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)
    return out


def _load_teacher_runtime_objects(
    teacher_contexts: Sequence[TeacherContext],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for teacher in teacher_contexts:
        out[teacher.round_id] = {
            "spec": teacher_loop_mod._load_json(TEACHER_REPORT_ROOT / teacher.round_id / "selected_spec.json"),
            "thresholds": _load_thresholds(teacher.round_id),
            "model": _load_teacher_model(teacher.round_id),
        }
    return out


def _annotate_teacher_scores(
    sampled_df: pd.DataFrame,
    teacher_contexts: Sequence[TeacherContext],
    teacher_runtime: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    out = sampled_df.copy()
    for teacher in teacher_contexts:
        runtime = dict(teacher_runtime[teacher.round_id])
        spec_payload = runtime["spec"]
        spec = teacher_loop_mod.TeacherSpec(
            title=str(spec_payload.get("title", "")),
            teacher_role=str(spec_payload.get("teacher_role", "")),
            research_family=str(spec_payload.get("research_family", "")),
            hypothesis=str(spec_payload.get("hypothesis", "")),
            sample_template=str(spec_payload.get("sample_template", "")),
            model_family=str(spec_payload.get("model_family", "")),
            target_kind=str(spec_payload.get("target_kind", "")),
            evaluation_contract=str(spec_payload.get("evaluation_contract", "")),
            feature_columns=[str(x) for x in list(spec_payload.get("feature_columns") or [])],
            novelty_rationale=str(spec_payload.get("novelty_rationale", "")),
        )
        scores = teacher_loop_mod._score_model(runtime["model"], spec, out)
        buckets = assign_prediction_buckets(scores, runtime["thresholds"])
        out[f"teacher_score__{teacher.round_id}"] = scores.astype(float)
        out[f"teacher_bucket__{teacher.round_id}"] = buckets.astype(int)
    return out


def _signal_prompt(
    *,
    config: ApprenticeReplayConfig,
    digest_text: str,
    teacher_cards: Sequence[Mapping[str, Any]],
    teacher_lessons: Mapping[str, Any],
    signal_record: Mapping[str, Any],
) -> Tuple[str, str]:
    reasoning_instruction = replay_mod._compact_reasoning_instruction(config)
    recipe = str(getattr(config, "prompt_recipe", "standard") or "standard").strip().lower()
    style_families = [str(row.get("style_family", "")).strip().lower() for row in teacher_cards]
    no_breakout_teacher = not any("breakout" in style for style in style_families)
    guardrail_extra = ""
    if recipe == "qwen_guardrail_v1":
        parts = [
            "Calibration for a random market-wide pool: most signals should score below 55.",
            "Reserve 70+ only for rare cases with a clear teacher-cluster match, a passed basic filter, and no major contradiction.",
            "If no teacher basic filter clearly matches, cap the score around 20..45. Use 50..55 only for mixed but not clearly negative evidence.",
            "For pullback or rebound teachers, high D/J, high pos_20, price near recent highs, close well above MA20, and volume spikes are not bullish by themselves. Without a real pullback anchor they often indicate late extension, crowding, or exhaustion risk.",
            "Single feature coincidences never override style mismatch. Check style and basic filter first, then local features.",
        ]
        if no_breakout_teacher:
            parts.append(
                "There is no dedicated breakout-chasing teacher in this prompt. Do not give high scores just because momentum is already hot or extended."
            )
        guardrail_extra = " ".join(parts)
    has_lessons = bool(teacher_lessons)
    lesson_instruction = (
        "Use the final lessons as teacher-local scoring guides when the signal matches or partially matches that teacher's style. "
        "If a signal falls outside every teacher's filter, generalize softly and avoid fake precision. "
    ) if has_lessons else (
        "No teacher-local final lessons are available in this run. "
        "Rely only on the simplified hypothesis-validation digest, the teacher basic infos, and the raw signal features. "
        "If a signal falls outside every teacher's filter, score conservatively and avoid inventing teacher-local scoring detail. "
    )
    system = (
        "You are independently scoring one market signal for 5-trading-day expected value. "
        f"{reasoning_instruction} "
        "Use the simplified hypothesis-validation digest only as background about which styles were easy or hard to stratify. "
        "Use the teacher basic infos to understand each teacher's cluster, filter, style, and historical strength. "
        f"{lesson_instruction}"
        "Scoring scale: 0..50 means expected value is negative, 50..100 means expected value is positive. "
        "Anchor the scale approximately as: 25 ~ around -2% expected 5-day return, 50 ~ near flat, 75 ~ around +2% expected 5-day return. "
        "Return strict JSON only with keys: total_score, short_reason. total_score must be a number from 0 to 100. "
        f"{guardrail_extra}"
    )
    user = json.dumps(
        {
            "simplified_hypothesis_validation_digest": digest_text,
            "teacher_basic_infos": list(teacher_cards),
            "teacher_final_lessons": dict(teacher_lessons),
            "decision_seed": int(config.llm_decision_seed or 0),
            "tie_break_note": (
                "If several final scores feel nearly equal, use decision_seed only as a deterministic tie-breaker. "
                "Do not mention the seed in the answer."
            ),
            "signal": signal_record,
        },
        ensure_ascii=False,
    )
    return system, user


def _score_one_signal(
    *,
    task: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    digest_text: str,
    teacher_cards: Sequence[Mapping[str, Any]],
    teacher_lessons: Mapping[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    system, user = _signal_prompt(
        config=config,
        digest_text=digest_text,
        teacher_cards=teacher_cards,
        teacher_lessons=teacher_lessons,
        signal_record=task["signal_record"],
    )
    request_payload = {
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": config.api_max_tokens,
        "temperature": config.api_temperature,
    }
    cache_path = Path(str(task["cache_path"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    api_calls = 0
    api_cache_hits = 0
    content = ""
    try:
        if bool(task.get("reuse_api_cache", True)) and cache_path.exists():
            cached = _load_json(cache_path)
            response_payload = cached.get("response")
            api_cache_hits += 1
        else:
            response_payload = replay_mod._chat_completion(
                messages=request_payload["messages"],
                api_key=api_key,
                model=config.api_model,
                max_tokens=config.api_max_tokens,
                temperature=config.api_temperature,
                force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=True,
                max_retries=max(1, int(config.api_request_max_retries)),
                seed=int(config.llm_decision_seed or 0) or None,
            )
            api_calls += 1
            _write_json(cache_path, {"request": request_payload, "response": response_payload})
        content = str(response_payload["choices"][0]["message"].get("content", "") or "") if response_payload else ""
        parsed = replay_mod._scorefit_parse_signal_reply(content=content, item_ids=[])
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": bool(parsed.get("parse_failed", False)),
            "failure_reason": str(parsed.get("failure_reason", "")).strip(),
            "total_score": _clip_score_0_100(float(parsed.get("total_score", 0.0))),
            "short_reason": str(parsed.get("short_reason", "")).strip(),
            "subscores": {},
            "parsed_payload": parsed.get("parsed_payload", {}),
        }
    except Exception as exc:
        _write_json(cache_path, {"request": request_payload, "error": str(exc)})
        return {
            **dict(task),
            "content": content,
            "api_calls": api_calls,
            "api_cache_hits": api_cache_hits,
            "parse_fallback": True,
            "failure_reason": str(exc),
            "total_score": 0.0,
            "short_reason": "",
            "subscores": {},
            "parsed_payload": {},
        }


def _score_signal_batch(
    *,
    task_batch: Sequence[Mapping[str, Any]],
    workers: int,
    config: ApprenticeReplayConfig,
    digest_text: str,
    teacher_cards: Sequence[Mapping[str, Any]],
    teacher_lessons: Mapping[str, Any],
    api_key: str,
    global_batch_index: int,
) -> List[Dict[str, Any]]:
    if not task_batch:
        return []
    results: List[Dict[str, Any]] = []
    future_map = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for task in task_batch:
            future = executor.submit(
                _score_one_signal,
                task=task,
                config=config,
                digest_text=digest_text,
                teacher_cards=teacher_cards,
                teacher_lessons=teacher_lessons,
                api_key=api_key,
            )
            future_map[future] = task
        for future in as_completed(future_map):
            result = future.result()
            result["parallel_batch_index"] = int(global_batch_index)
            results.append(result)
    results.sort(key=lambda row: (str(row.get("signal_date")), str(row.get("task_key"))))
    return results


def _build_signal_tasks(
    sampled_df: pd.DataFrame,
    *,
    prompt_feature_cols: Sequence[str],
    run_dir: Path,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for task_index, row in enumerate(sampled_df.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        signal_date = pd.Timestamp(row_series["signal_date"]).strftime("%Y-%m-%d")
        symbol = str(row_series.get("symbol", "")).strip()
        task_key = f"{signal_date}::{symbol}::{task_index:05d}"
        task_payload = {
            "task_key": task_key,
            "signal_date": signal_date,
            "symbol": symbol,
            "future_return_5d": replay_mod._scorefit_safe_float(row_series.get("future_return_5d"), 0.0),
            "signal_record": replay_mod._scorefit_signal_record_from_row(row_series, prompt_feature_cols),
            "cache_path": str(run_dir / "api_calls" / f"{task_index:05d}.json"),
            "reuse_api_cache": True,
            "rerun_count": 0,
        }
        tasks.append(task_payload)
    return tasks


def _run_llm_scoring(
    *,
    sampled_df: pd.DataFrame,
    config: ApprenticeReplayConfig,
    digest_text: str,
    teacher_cards: Sequence[Mapping[str, Any]],
    teacher_lessons: Mapping[str, Any],
    prompt_feature_cols: Sequence[str],
    run_dir: Path,
) -> List[Dict[str, Any]]:
    tasks = _build_signal_tasks(sampled_df, prompt_feature_cols=prompt_feature_cols, run_dir=run_dir)
    replay_mod._progress(
        "market scorefit run start "
        f"model={config.api_model} sample_seed={config.sample_seed} "
        f"llm_seed={int(config.llm_decision_seed or 0)} "
        f"tasks={len(tasks)} workers={config.api_parallel_workers}"
    )
    pending = list(tasks)
    finalized: Dict[str, Dict[str, Any]] = {}
    total_passes = 1 + max(0, int(config.api_failed_rerun_rounds))
    api_key = replay_mod._apprentice_api_key()
    global_batch_counter = 0
    for pass_index in range(total_passes):
        if not pending:
            break
        next_pending: List[Dict[str, Any]] = []
        batch_cursor = 0
        while batch_cursor < len(pending):
            task_batch = pending[batch_cursor : batch_cursor + max(1, int(config.api_parallel_workers))]
            batch_cursor += max(1, int(config.api_parallel_workers))
            global_batch_counter += 1
            results = _score_signal_batch(
                task_batch=task_batch,
                workers=max(1, int(config.api_parallel_workers)),
                config=config,
                digest_text=digest_text,
                teacher_cards=teacher_cards,
                teacher_lessons=teacher_lessons,
                api_key=api_key,
                global_batch_index=global_batch_counter,
            )
            source_by_key = {str(task["task_key"]): dict(task) for task in task_batch}
            requeued = 0
            finalized_count = 0
            for result in results:
                if bool(result.get("parse_fallback", False)) and pass_index + 1 < total_passes:
                    rerun_task = dict(source_by_key[str(result["task_key"])])
                    rerun_task["reuse_api_cache"] = False
                    rerun_task["rerun_count"] = int(rerun_task.get("rerun_count", 0)) + 1
                    next_pending.append(rerun_task)
                    requeued += 1
                else:
                    finalized[str(result["task_key"])] = result
                    finalized_count += 1
            replay_mod._progress(
                "market scorefit batch done "
                f"model={config.api_model} pass={pass_index + 1}/{total_passes} "
                f"wave={global_batch_counter} size={len(task_batch)} requeued={requeued} finalized={finalized_count}"
            )
        pending = next_pending
    if pending:
        raise RuntimeError(f"unresolved failed samples remain: pending={len(pending)}")
    return [finalized[str(task["task_key"])] for task in tasks]


def _compute_llm_metrics(
    *,
    results: Sequence[Mapping[str, Any]],
    lock_days: int,
    daily_top_pct: float,
) -> Dict[str, Any]:
    usable = [dict(row) for row in results if not bool(row.get("parse_fallback", False))]
    failure_count = int(len(results) - len(usable))
    usable.sort(key=lambda row: (str(row.get("signal_date")), str(row.get("symbol"))))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in usable:
        grouped.setdefault(str(row["signal_date"]), []).append(row)

    capital_frac = 1.0 / max(1, int(lock_days))
    nav = 1.0
    selected_keys: set[str] = set()
    daily_rows: List[Dict[str, Any]] = []
    for signal_date, rows in grouped.items():
        day_rows = sorted(
            rows,
            key=lambda row: (-replay_mod._scorefit_safe_float(row.get("total_score"), 0.0), str(row.get("symbol", ""))),
        )
        pick_count = max(1, int(round(len(day_rows) * float(daily_top_pct))))
        picked = day_rows[:pick_count]
        picked_returns = [replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in picked]
        day_return = float(np.mean(picked_returns)) if picked_returns else 0.0
        nav *= 1.0 + capital_frac * day_return
        for row in picked:
            selected_keys.add(str(row["task_key"]))
        daily_rows.append(
            {
                "signal_date": signal_date,
                "pick_count": int(len(picked)),
                "selected_mean_return": day_return,
                "nav_after_close": nav,
            }
        )

    selected = [row for row in usable if str(row["task_key"]) in selected_keys]
    not_selected = [row for row in usable if str(row["task_key"]) not in selected_keys]
    return {
        "llm_parse_failure_count": failure_count,
        "successful_signal_count": int(len(usable)),
        "decision_days": int(len(grouped)),
        "llm_selected_count": int(len(selected)),
        "llm_selected_mean_return": (
            float(np.mean([replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in selected]))
            if selected
            else 0.0
        ),
        "llm_not_selected_mean_return": (
            float(np.mean([replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in not_selected]))
            if not_selected
            else 0.0
        ),
        "llm_uplift_vs_not_selected": (
            (
                float(np.mean([replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in selected]))
                if selected
                else 0.0
            )
            - (
                float(np.mean([replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0) for row in not_selected]))
                if not_selected
                else 0.0
            )
        ),
        "llm_nav_final": float(nav),
        "daily_rows": daily_rows,
        "selected_task_keys": sorted(selected_keys),
    }


def _compute_teacher_metrics(
    *,
    sampled_df: pd.DataFrame,
    teacher_contexts: Sequence[TeacherContext],
    llm_results: Sequence[Mapping[str, Any]],
    lock_days: int,
) -> Dict[str, Any]:
    llm_score_by_key = {str(row["task_key"]): replay_mod._scorefit_safe_float(row.get("total_score"), 0.0) for row in llm_results}
    task_key_by_symbol_day = {
        (str(row["symbol"]).strip(), pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d")): str(row["task_key"])
        for row in llm_results
    }
    capital_frac = 1.0 / max(1, int(lock_days))
    teacher_rows: Dict[str, Dict[str, Any]] = {}
    for teacher in teacher_contexts:
        score_col = f"teacher_score__{teacher.round_id}"
        bucket_col = f"teacher_bucket__{teacher.round_id}"
        working = sampled_df.copy()
        working["task_key"] = [
            task_key_by_symbol_day[(str(sym).strip(), pd.Timestamp(dt).strftime("%Y-%m-%d"))]
            for sym, dt in zip(working["symbol"], working["signal_date"])
        ]
        usable = working[working["task_key"].isin(llm_score_by_key)].copy()
        if usable.empty:
            teacher_rows[teacher.round_id] = {
                "round_id": teacher.round_id,
                "teacher_selected_mean_return": 0.0,
                "teacher_not_selected_mean_return": 0.0,
                "teacher_uplift_vs_not_selected": 0.0,
                "teacher_nav_final": 1.0,
                "teacher_q5_count": 0,
                "teacher_score_spearman_with_llm": 0.0,
            }
            continue
        usable["llm_total_score"] = usable["task_key"].map(llm_score_by_key).astype(float)
        usable["teacher_score"] = usable[score_col].astype(float)
        usable["teacher_bucket"] = usable[bucket_col].astype(int)
        q5 = usable[usable["teacher_bucket"] == 5].copy()
        not_q5 = usable[usable["teacher_bucket"] != 5].copy()
        nav = 1.0
        daily_rows: List[Dict[str, Any]] = []
        for signal_date, day_df in usable.groupby("signal_date", sort=True):
            picked = day_df[day_df["teacher_bucket"] == 5]
            returns = picked["future_return_5d"].astype(float).tolist()
            day_return = float(np.mean(returns)) if returns else 0.0
            nav *= 1.0 + capital_frac * day_return
            daily_rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                    "q5_count": int(len(picked)),
                    "selected_mean_return": day_return,
                    "nav_after_close": nav,
                }
            )
        teacher_rows[teacher.round_id] = {
            "round_id": teacher.round_id,
            "title": teacher.title,
            "style_family": teacher.style_label,
            "teacher_selected_mean_return": float(q5["future_return_5d"].mean()) if not q5.empty else 0.0,
            "teacher_not_selected_mean_return": float(not_q5["future_return_5d"].mean()) if not not_q5.empty else 0.0,
            "teacher_uplift_vs_not_selected": (
                (float(q5["future_return_5d"].mean()) if not q5.empty else 0.0)
                - (float(not_q5["future_return_5d"].mean()) if not not_q5.empty else 0.0)
            ),
            "teacher_nav_final": float(nav),
            "teacher_q5_count": int(len(q5)),
            "teacher_score_spearman_with_llm": replay_mod._scorefit_corr_spearman(
                usable["llm_total_score"].astype(float).tolist(),
                usable["teacher_score"].astype(float).tolist(),
            ),
            "daily_rows": daily_rows,
        }
    return teacher_rows


def _summary_markdown(
    *,
    run_tag: str,
    model_name: str,
    start_date: str,
    end_date: str,
    prompt_feature_cols: Sequence[str],
    llm_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Mapping[str, Any]],
    digest_path: Path,
    final_lesson_state_json: Path,
    disable_final_lessons: bool,
    daily_sample_size: int,
    daily_top_pct: float,
    api_parallel_workers: int,
) -> str:
    teacher_rows = [dict(row) for row in teacher_metrics.values()]
    teacher_rows_sorted = sorted(
        teacher_rows,
        key=lambda row: float(row.get("teacher_uplift_vs_not_selected", 0.0)),
        reverse=True,
    )
    lines: List[str] = [
        "# Market Scorefit Backtest",
        "",
        f"- run_tag: `{run_tag}`",
        f"- model: `{model_name}`",
        f"- window: `{start_date}` -> `{end_date}`",
        f"- daily_sample_size: `{daily_sample_size}`",
        f"- llm_daily_top_pct: `{daily_top_pct:.2%}`",
        f"- prompt_feature_count: `{len(prompt_feature_cols)}`",
        f"- api_parallel_workers: `{api_parallel_workers}`",
        f"- digest_path: `{digest_path}`",
        f"- final_lesson_state_json: `{final_lesson_state_json}`",
        f"- disable_final_lessons: `{disable_final_lessons}`",
        "",
        "## LLM",
        f"- successful_signal_count: `{int(llm_metrics.get('successful_signal_count', 0) or 0)}`",
        f"- decision_days: `{int(llm_metrics.get('decision_days', 0) or 0)}`",
        f"- llm_selected_mean_return: `{float(llm_metrics.get('llm_selected_mean_return', 0.0)):+.4%}`",
        f"- llm_not_selected_mean_return: `{float(llm_metrics.get('llm_not_selected_mean_return', 0.0)):+.4%}`",
        f"- llm_uplift_vs_not_selected: `{float(llm_metrics.get('llm_uplift_vs_not_selected', 0.0)):+.4%}`",
        f"- llm_nav_final: `{float(llm_metrics.get('llm_nav_final', 1.0)):.4f}`",
        "",
        "## Teachers",
    ]
    for row in teacher_rows_sorted:
        lines.extend(
            [
                f"### {row.get('round_id')}",
                f"- title: {row.get('title', '')}",
                f"- style_family: {row.get('style_family', '')}",
                f"- teacher_selected_mean_return: {float(row.get('teacher_selected_mean_return', 0.0)):+.4%}",
                f"- teacher_not_selected_mean_return: {float(row.get('teacher_not_selected_mean_return', 0.0)):+.4%}",
                f"- teacher_uplift_vs_not_selected: {float(row.get('teacher_uplift_vs_not_selected', 0.0)):+.4%}",
                f"- teacher_nav_final: {float(row.get('teacher_nav_final', 1.0)):.4f}",
                f"- teacher_score_spearman_with_llm: {float(row.get('teacher_score_spearman_with_llm', 0.0)):+.4f}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    selection_json = Path(args.selection_json).expanduser().resolve()
    final_lesson_state_json = Path(args.final_lesson_state_json).expanduser().resolve()
    run_root = (REPORT_ROOT / args.report_subdir / args.run_tag).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(selection_json)
    final_lesson_state = _load_final_lesson_state(final_lesson_state_json)

    teacher_contexts = _collect_teacher_contexts(
        selection=selection,
        final_lesson_state=final_lesson_state,
        prompt_top_feature_count=int(args.prompt_top_feature_count),
        prompt_max_meta_rules=int(args.prompt_max_meta_rules),
        prompt_max_scoring_notes=int(args.prompt_max_scoring_notes),
        teacher_profile_start_date=args.teacher_profile_start_date,
        teacher_profile_end_date=args.teacher_profile_end_date,
    )
    negative_contexts = _collect_negative_contexts(
        selection=selection,
        prompt_top_feature_count=int(args.prompt_top_feature_count),
        teacher_profile_start_date=args.teacher_profile_start_date,
        teacher_profile_end_date=args.teacher_profile_end_date,
    )

    digest_text = _build_simplified_hypval_digest(
        teacher_contexts=teacher_contexts,
        negative_contexts=negative_contexts,
    )
    digest_path = (
        Path(args.digest_output_path).expanduser().resolve()
        if str(args.digest_output_path).strip()
        else run_root / "simplified_hypval_digest.txt"
    )
    _write_text(digest_path, digest_text)

    base_config = ApprenticeReplayConfig(
        mode="multi",
        teacher_round_ids=[teacher.round_id for teacher in teacher_contexts],
        negative_teacher_round_ids=[str(x) for x in list(selection.get("negative_round_ids") or [])],
        start_date=args.start_date,
        end_date=args.end_date,
        lock_days=int(args.lock_days),
        api_model=args.api_model,
        prompt_recipe=str(args.prompt_recipe or "standard").strip(),
        api_temperature=float(args.api_temperature),
        api_max_tokens=int(args.api_max_tokens),
        force_local_qwen_no_thinking=(not args.enable_local_qwen_thinking),
        private_reasoning_target_tokens=int(args.private_reasoning_target_tokens),
        private_reasoning_max_tokens_hint=int(args.private_reasoning_max_tokens_hint),
        api_parallel_workers=int(args.api_parallel_workers),
        api_failed_rerun_rounds=int(args.api_failed_rerun_rounds),
        api_request_max_retries=int(args.api_request_max_retries),
        sample_seed=int(args.sample_seed),
        llm_decision_seed=(int(args.llm_decision_seed) if int(args.llm_decision_seed) > 0 else int(args.sample_seed)),
    )
    if args.force_local_qwen_no_thinking:
        base_config.force_local_qwen_no_thinking = True

    master_df = replay_mod._load_master_dataset()
    replay_mod._progress(f"master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")
    market_df = master_df[
        (master_df["signal_date"] >= pd.Timestamp(args.start_date))
        & (master_df["signal_date"] <= pd.Timestamp(args.end_date))
    ].copy()
    if market_df.empty:
        raise RuntimeError(f"no market rows between {args.start_date} and {args.end_date}")
    market_df["symbol"] = market_df["symbol"].map(replay_mod._normalize_symbol)
    market_df = market_df.sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)

    prompt_feature_cols, missing_prompt_features = _prompt_feature_union(
        teacher_contexts,
        market_df.columns,
        include_lesson_features=(not bool(args.disable_final_lessons)),
    )
    if not prompt_feature_cols:
        raise RuntimeError("no prompt feature columns survived intersection with market dataset")

    sampled_df = _sample_market_signals(
        market_df,
        daily_sample_size=int(args.daily_sample_size),
        sample_seed=int(args.sample_seed),
        max_days=int(args.max_days),
    )
    if sampled_df.empty:
        raise RuntimeError("sampled market frame is empty")

    teacher_runtime = _load_teacher_runtime_objects(teacher_contexts)
    sampled_df = _annotate_teacher_scores(
        sampled_df,
        teacher_contexts=teacher_contexts,
        teacher_runtime=teacher_runtime,
    )

    teacher_cards = _build_teacher_prompt_cards(teacher_contexts)
    teacher_lessons = {} if bool(args.disable_final_lessons) else {
        teacher.round_id: teacher.lesson_prompt for teacher in teacher_contexts
    }

    _write_json(run_root / "teacher_basic_infos.json", {"rows": teacher_cards})
    _write_json(run_root / "teacher_final_lessons.json", teacher_lessons)
    _write_json(
        run_root / "prompt_feature_manifest.json",
        {
            "prompt_feature_cols": prompt_feature_cols,
            "missing_prompt_features": missing_prompt_features,
        },
    )
    sampled_df.to_csv(run_root / "sampled_market_signals.csv.gz", index=False)

    llm_results = _run_llm_scoring(
        sampled_df=sampled_df,
        config=base_config,
        digest_text=replay_mod._scorefit_clipped_digest_text(digest_text, int(args.prompt_digest_max_chars)),
        teacher_cards=teacher_cards,
        teacher_lessons=teacher_lessons,
        prompt_feature_cols=prompt_feature_cols,
        run_dir=run_root,
    )
    llm_metrics = _compute_llm_metrics(
        results=llm_results,
        lock_days=int(args.lock_days),
        daily_top_pct=float(args.daily_top_pct),
    )
    teacher_metrics = _compute_teacher_metrics(
        sampled_df=sampled_df,
        teacher_contexts=teacher_contexts,
        llm_results=llm_results,
        lock_days=int(args.lock_days),
    )

    teacher_mean_uplift = float(np.mean([row["teacher_uplift_vs_not_selected"] for row in teacher_metrics.values()])) if teacher_metrics else 0.0
    teacher_best_uplift = float(max([row["teacher_uplift_vs_not_selected"] for row in teacher_metrics.values()])) if teacher_metrics else 0.0
    teacher_mean_nav = float(np.mean([row["teacher_nav_final"] for row in teacher_metrics.values()])) if teacher_metrics else 1.0
    teacher_best_nav = float(max([row["teacher_nav_final"] for row in teacher_metrics.values()])) if teacher_metrics else 1.0
    mean_teacher_spearman = float(np.mean([row["teacher_score_spearman_with_llm"] for row in teacher_metrics.values()])) if teacher_metrics else 0.0

    summary = {
        "run_tag": args.run_tag,
        "model": args.api_model,
        "prompt_recipe": str(args.prompt_recipe or "standard").strip(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "daily_sample_size": int(args.daily_sample_size),
        "daily_top_pct": float(args.daily_top_pct),
        "lock_days": int(args.lock_days),
        "sample_seed": int(args.sample_seed),
        "llm_decision_seed": (int(args.llm_decision_seed) if int(args.llm_decision_seed) > 0 else int(args.sample_seed)),
        "prompt_feature_count": int(len(prompt_feature_cols)),
        "api_parallel_workers": int(args.api_parallel_workers),
        "api_failed_rerun_rounds": int(args.api_failed_rerun_rounds),
        "digest_path": str(digest_path.resolve()),
        "final_lesson_state_json": str(final_lesson_state_json.resolve()),
        "disable_final_lessons": bool(args.disable_final_lessons),
        "selection_json": str(selection_json.resolve()),
        "llm_metrics": llm_metrics,
        "teacher_metrics": teacher_metrics,
        "teacher_mean_uplift_vs_not_selected": teacher_mean_uplift,
        "teacher_best_uplift_vs_not_selected": teacher_best_uplift,
        "teacher_mean_nav_final": teacher_mean_nav,
        "teacher_best_nav_final": teacher_best_nav,
        "mean_teacher_score_spearman_with_llm": mean_teacher_spearman,
        "missing_prompt_features": missing_prompt_features,
    }

    _write_json(run_root / "summary.json", summary)
    _write_json(run_root / "llm_signal_scores.json", {"results": llm_results})
    _write_json(run_root / "llm_daily_nav.json", {"rows": list(llm_metrics.get("daily_rows") or [])})
    _write_json(run_root / "teacher_daily_nav.json", {"teachers": teacher_metrics})
    _write_text(
        run_root / "summary.md",
        _summary_markdown(
            run_tag=args.run_tag,
            model_name=args.api_model,
            start_date=args.start_date,
            end_date=args.end_date,
            prompt_feature_cols=prompt_feature_cols,
            llm_metrics=llm_metrics,
            teacher_metrics=teacher_metrics,
            digest_path=digest_path,
            final_lesson_state_json=final_lesson_state_json,
            disable_final_lessons=bool(args.disable_final_lessons),
            daily_sample_size=int(args.daily_sample_size),
            daily_top_pct=float(args.daily_top_pct),
            api_parallel_workers=int(args.api_parallel_workers),
        ),
    )
    replay_mod._progress(
        "market scorefit complete "
        f"model={args.api_model} llm_nav={float(llm_metrics.get('llm_nav_final', 1.0)):.4f} "
        f"llm_uplift={float(llm_metrics.get('llm_uplift_vs_not_selected', 0.0)):+.4%}"
    )


if __name__ == "__main__":
    main()
