#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run single-teacher scope alignment tests on teacher-comfort-zone dates."""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from quant_toolkit.apprentice_loop import ApprenticeReplayConfig
from quant_toolkit.apprentice_loop import replay as replay_mod


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_replay_summary(path: Path) -> replay_mod.ReplaySummary:
    payload = _load_json(path)
    return replay_mod.ReplaySummary(**payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scope alignment tests for QuantApprentice")
    parser.add_argument("--selection-json", required=True, help="Path to teacher selection.json with frozen_round_ids")
    parser.add_argument("--lesson-artifact-json", default="", help="Optional warmup_scoped_lessons.json for scope lessons")
    parser.add_argument("--start-date", default="2020-01-02")
    parser.add_argument("--end-date", default="2022-12-30")
    parser.add_argument("--per-scope-sample-count", type=int, default=50)
    parser.add_argument(
        "--alignment-sampling-strategy",
        default="teacher_selected_mean_return_top_bottom_v2",
        choices=[
            "teacher_selected_mean_return_top_bottom_v2",
            "neutral36_topq5_v1",
        ],
    )
    parser.add_argument("--signal-pool-per-day", type=int, default=15)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--teacher-daily-pick-count", type=int, default=4)
    parser.add_argument("--llm-max-daily-picks", type=int, default=4)
    parser.add_argument("--prompt-feature-count", type=int, default=8)
    parser.add_argument("--lesson-feature-count", type=int, default=20)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-max-tokens", type=int, default=1024)
    parser.add_argument("--force-local-qwen-thinking", action="store_true")
    parser.add_argument("--use-line-answer-prefix", action="store_true")
    parser.add_argument("--private-reasoning-target-tokens", type=int, default=0)
    parser.add_argument("--private-reasoning-max-tokens-hint", type=int, default=0)
    parser.add_argument("--prompt-recipe", choices=["standard", "explainability_only", "report_v2_with_lessons"], default="standard")
    parser.add_argument("--summary-variant", choices=["simple_v1", "enriched_v2"], default="enriched_v2")
    parser.add_argument("--candidate-source", choices=["teacher_ranked", "baseline_signal"], default="baseline_signal")
    parser.add_argument("--include-teacher-signal", action="store_true")
    parser.add_argument("--ignore-holdings-context", action="store_true")
    parser.add_argument("--api-parallel-workers", type=int, default=1)
    parser.add_argument(
        "--api-failed-rerun-rounds",
        type=int,
        default=1,
        help="Number of flat-batch rerun rounds for APIEmptyContent/parse-fallback tasks; default keeps failed days in batch mode instead of stopping at one pass.",
    )
    parser.add_argument("--api-failed-rerun-workers", type=int, default=0)
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--bundle-cache-dir", default="", help="Optional joblib cache directory for per-scope frames")
    return parser


def _load_selection(selection_json: Path) -> Dict[str, Any]:
    payload = _load_json(selection_json)
    frozen = list(payload.get("frozen_round_ids") or [])
    positive = list(payload.get("positive_round_ids") or [])
    negative = list(payload.get("negative_round_ids") or [])
    if not frozen:
        raise ValueError(f"selection_json has no frozen_round_ids: {selection_json}")
    return {
        "frozen_teachers": frozen,
        "positive_teachers": positive,
        "negative_teachers": negative,
        "raw": payload,
    }


def _load_scoped_warmup_state(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"lesson artifact not found: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict) or "teacher_scopes" not in payload:
        raise ValueError(f"lesson artifact is not a scoped warmup payload: {path}")
    return payload


def _resolve_scope_specs(selection: Dict[str, Any], scoped_warmup_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if scoped_warmup_state:
        scopes = []
        for item in list(scoped_warmup_state.get("teacher_scopes") or []):
            scopes.append(
                {
                    "final_round_id": str(item.get("round_id", "")).strip(),
                    "source_round_id": str(item.get("source_round_id") or item.get("round_id") or "").strip(),
                    "scope_entry": dict(item),
                }
            )
        scopes = [item for item in scopes if item["final_round_id"] and item["source_round_id"]]
        if scopes:
            return scopes
    frozen = list(selection.get("frozen_teachers") or [])
    positive = list(selection.get("positive_teachers") or [])
    if positive and len(positive) != len(frozen):
        raise ValueError("selection.json positive_round_ids and frozen_round_ids length mismatch")
    scopes = []
    for idx, frozen_round_id in enumerate(frozen):
        source_round_id = positive[idx] if idx < len(positive) else replay_mod._source_round_id_for_round(frozen_round_id)
        scopes.append(
            {
                "final_round_id": str(frozen_round_id),
                "source_round_id": str(source_round_id),
                "scope_entry": {},
            }
        )
    return scopes


def _teacher_not_selected_rows(candidate_pool_df: pd.DataFrame, teacher_target_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [col for col in ["symbol", "signal_date", "entry_date", "exit_date"] if col in candidate_pool_df.columns]
    if not key_cols:
        return candidate_pool_df.iloc[0:0].copy()
    target_keys = teacher_target_df[key_cols].drop_duplicates() if not teacher_target_df.empty else pd.DataFrame(columns=key_cols)
    pool_not_selected = candidate_pool_df.merge(
        target_keys.assign(_teacher_selected=1),
        on=key_cols,
        how="left",
    )
    pool_not_selected = pool_not_selected[pool_not_selected["_teacher_selected"].isna()].drop(columns=["_teacher_selected"])
    return pool_not_selected


def _teacher_day_alpha_table(candidate_pool_df: pd.DataFrame, teacher_target_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    teacher_not_selected_df = _teacher_not_selected_rows(candidate_pool_df, teacher_target_df)
    grouped_pool = {pd.Timestamp(dt): df.copy() for dt, df in candidate_pool_df.groupby("signal_date", sort=True)}
    grouped_target = {pd.Timestamp(dt): df.copy() for dt, df in teacher_target_df.groupby("signal_date", sort=True)}
    grouped_not_selected = {pd.Timestamp(dt): df.copy() for dt, df in teacher_not_selected_df.groupby("signal_date", sort=True)}
    for decision_date in sorted(grouped_pool):
        target_df = grouped_target.get(decision_date)
        not_selected_df = grouped_not_selected.get(decision_date)
        if target_df is None or target_df.empty or not_selected_df is None or not_selected_df.empty:
            continue
        teacher_mean = float(target_df["future_return_5d"].mean())
        not_selected_mean = float(not_selected_df["future_return_5d"].mean())
        rows.append(
            {
                "signal_date": decision_date,
                "teacher_selected_count": int(len(target_df)),
                "teacher_not_selected_count": int(len(not_selected_df)),
                "teacher_selected_mean_return": teacher_mean,
                "teacher_not_selected_mean_return": not_selected_mean,
                "teacher_uplift_vs_not_selected": float(teacher_mean - not_selected_mean),
            }
        )
    return pd.DataFrame(rows)


def _sample_scope_days(teacher_alpha_df: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if len(teacher_alpha_df) < sample_count:
        raise ValueError(
            f"scope has only {len(teacher_alpha_df)} teacher-evaluable days, fewer than requested {sample_count}"
        )
    high_count = sample_count // 2
    low_count = sample_count - high_count
    ranked_high = teacher_alpha_df.sort_values(
        ["teacher_selected_mean_return", "teacher_uplift_vs_not_selected", "signal_date"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top_days_df = ranked_high.head(high_count).copy()
    top_days_df["sample_side"] = "top_teacher_return"

    remaining_df = teacher_alpha_df[~teacher_alpha_df["signal_date"].isin(top_days_df["signal_date"])].copy()
    ranked_low = remaining_df.sort_values(
        ["teacher_selected_mean_return", "teacher_uplift_vs_not_selected", "signal_date"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    bottom_days_df = ranked_low.head(low_count).copy()
    bottom_days_df["sample_side"] = "bottom_teacher_return"

    sampled_days_df = pd.concat([top_days_df, bottom_days_df], ignore_index=True)
    sampled_days_df = sampled_days_df.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    return sampled_days_df


def _scope_bundle_cache_path(
    *,
    cache_dir: Optional[Path],
    source_round_id: str,
    final_round_id: str,
    config: ApprenticeReplayConfig,
    sample_count: int,
    alignment_sampling_strategy: str,
    signal_pool_per_day: int,
) -> Optional[Path]:
    if cache_dir is None:
        return None
    sampling_strategy = str(alignment_sampling_strategy).strip() or "teacher_return_top_bottom_v2"
    name = (
        f"{final_round_id}_{source_round_id}_{config.start_date.replace('-', '')}_{config.end_date.replace('-', '')}"
        f"_pool{config.candidate_pool_size}_pick{config.teacher_daily_pick_count}"
        f"_sample{sample_count}_{sampling_strategy}_signals{int(signal_pool_per_day)}_{config.summary_variant}.joblib"
    )
    return cache_dir / name


def _prepare_scope_bundle(
    *,
    master_df: pd.DataFrame,
    base_config: ApprenticeReplayConfig,
    final_round_id: str,
    source_round_id: str,
    sample_count: int,
    cache_dir: Optional[Path],
    alignment_sampling_strategy: str,
    signal_pool_per_day: int,
) -> Dict[str, Any]:
    cache_path = _scope_bundle_cache_path(
        cache_dir=cache_dir,
        source_round_id=source_round_id,
        final_round_id=final_round_id,
        config=base_config,
        sample_count=sample_count,
        alignment_sampling_strategy=alignment_sampling_strategy,
        signal_pool_per_day=signal_pool_per_day,
    )
    if cache_path and cache_path.exists():
        replay_mod._progress(f"scope alignment cache hit path={cache_path}")
        return joblib.load(cache_path)

    scope_cfg = replace(
        base_config,
        mode="single",
        teacher_round_ids=[source_round_id],
        negative_teacher_round_ids=[],
        candidate_source="baseline_signal",
        warmup_sample_count=0,
        run_tag="",
    )
    merged, meta = replay_mod._build_single_teacher_frame(scope_cfg, master_df)
    if scope_cfg.summary_variant == "enriched_v2":
        meta["preference_bands"] = replay_mod._derive_preference_bands(
            round_id=source_round_id,
            master_df=master_df,
            feature_cols=meta["top_prompt_features"],
            start_date=scope_cfg.start_date,
            end_date=scope_cfg.end_date,
        )

    if alignment_sampling_strategy == "teacher_selected_mean_return_top_bottom_v2":
        candidate_pool_all_df, teacher_target_all_df = replay_mod._single_teacher_target(merged, scope_cfg)
        teacher_alpha_df = _teacher_day_alpha_table(candidate_pool_all_df, teacher_target_all_df)
        sampled_days_df = _sample_scope_days(teacher_alpha_df, sample_count)
        sampled_dates = sorted(pd.Timestamp(item) for item in sampled_days_df["signal_date"].tolist())
        sampled_date_set = set(sampled_dates)
        candidate_pool_df = candidate_pool_all_df[candidate_pool_all_df["signal_date"].isin(sampled_date_set)].copy()
        teacher_target_df = teacher_target_all_df[teacher_target_all_df["signal_date"].isin(sampled_date_set)].copy()
        teacher_alpha_sampled_df = teacher_alpha_df[teacher_alpha_df["signal_date"].isin(sampled_date_set)].copy()
        sampling_summary = {
            "top_teacher_return_days": int((sampled_days_df["sample_side"] == "top_teacher_return").sum()),
            "bottom_teacher_return_days": int((sampled_days_df["sample_side"] == "bottom_teacher_return").sum()),
            "signal_pool_per_day": int(signal_pool_per_day),
        }
    elif alignment_sampling_strategy == "neutral36_topq5_v1":
        sampled_pool_parts: List[pd.DataFrame] = []
        sampled_target_parts: List[pd.DataFrame] = []
        day_rows: List[Dict[str, Any]] = []
        for signal_date, day_df in merged.groupby("signal_date", sort=True):
            sampled_day = replay_mod._scorefit_sample_day_signals(
                day_df.copy(),
                x=signal_pool_per_day,
                sample_seed=0,
            )
            if sampled_day.empty or len(sampled_day) < int(scope_cfg.teacher_daily_pick_count):
                continue
            sampled_target = sampled_day.sort_values(
                ["score", "bucket"],
                ascending=[False, False],
            ).head(scope_cfg.teacher_daily_pick_count).copy()
            if sampled_target.empty:
                continue
            q5_rows = sampled_day[sampled_day["bucket"] == 5]
            sampled_pool_parts.append(sampled_day)
            sampled_target_parts.append(sampled_target)
            day_rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date),
                    "sampled_signal_count": int(len(sampled_day)),
                    "day_sample_mean_return": float(sampled_day["future_return_5d"].mean()),
                    "day_q5_mean_return": float(q5_rows["future_return_5d"].mean()) if not q5_rows.empty else 0.0,
                }
            )
        if not sampled_pool_parts or not sampled_target_parts:
            raise ValueError(
                f"scope {final_round_id} has no evaluable neutral36_topq5_v1 days "
                f"with signal_pool_per_day={signal_pool_per_day}"
            )
        candidate_pool_all_df = pd.concat(sampled_pool_parts, ignore_index=True)
        teacher_target_all_df = pd.concat(sampled_target_parts, ignore_index=True)
        day_table = pd.DataFrame(day_rows).sort_values("signal_date").reset_index(drop=True)
        if len(day_table) < int(sample_count):
            raise ValueError(
                f"scope {final_round_id} has only {len(day_table)} evaluable days, "
                f"fewer than requested {sample_count}"
            )
        neutral_pool_size = max(36, int(sample_count))
        if len(day_table) < neutral_pool_size:
            raise ValueError(
                f"scope {final_round_id} has only {len(day_table)} evaluable days, "
                f"fewer than neutral pool size {neutral_pool_size}"
            )
        day_table["abs_day_sample_mean_return"] = day_table["day_sample_mean_return"].abs()
        neutral_pool = day_table.sort_values(
            ["abs_day_sample_mean_return", "signal_date"],
            ascending=[True, True],
            kind="mergesort",
        ).head(neutral_pool_size).copy()
        neutral_pool = neutral_pool.reset_index(drop=True)
        neutral_pool["neutral_pool_rank"] = list(range(1, len(neutral_pool) + 1))
        sampled_days_df = neutral_pool.sort_values(
            ["day_q5_mean_return", "abs_day_sample_mean_return", "signal_date"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(sample_count).copy()
        sampled_days_df = sampled_days_df.reset_index(drop=True)
        sampled_days_df["q5_rank_within_neutral_pool"] = list(range(1, len(sampled_days_df) + 1))
        sampled_days_df = sampled_days_df.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
        sampled_dates = sorted(pd.Timestamp(item) for item in sampled_days_df["signal_date"].tolist())
        sampled_date_set = set(sampled_dates)
        candidate_pool_df = candidate_pool_all_df[candidate_pool_all_df["signal_date"].isin(sampled_date_set)].copy()
        teacher_target_df = teacher_target_all_df[teacher_target_all_df["signal_date"].isin(sampled_date_set)].copy()
        teacher_alpha_df = _teacher_day_alpha_table(candidate_pool_all_df, teacher_target_all_df)
        teacher_alpha_sampled_df = _teacher_day_alpha_table(candidate_pool_df, teacher_target_df)
        sampling_summary = {
            "neutral_pool_size": int(neutral_pool_size),
            "selected_day_sample_mean_return_mean": float(sampled_days_df["day_sample_mean_return"].mean()),
            "selected_day_q5_mean_return_mean": float(sampled_days_df["day_q5_mean_return"].mean()),
            "signal_pool_per_day": int(signal_pool_per_day),
        }
    else:
        raise ValueError(f"unsupported alignment_sampling_strategy: {alignment_sampling_strategy}")

    teacher_full_df = teacher_target_df.copy()
    teacher_not_selected_df = _teacher_not_selected_rows(candidate_pool_df, teacher_target_df)
    teacher_not_selected_mean_return = (
        float(teacher_not_selected_df["future_return_5d"].mean())
        if not teacher_not_selected_df.empty and "future_return_5d" in teacher_not_selected_df.columns
        else 0.0
    )
    teacher_selected_mean_return = (
        float(teacher_target_df["future_return_5d"].mean())
        if not teacher_target_df.empty and "future_return_5d" in teacher_target_df.columns
        else 0.0
    )
    bundle = {
        "final_round_id": final_round_id,
        "source_round_id": source_round_id,
        "meta": meta,
        "candidate_pool_df": candidate_pool_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True),
        "teacher_target_df": teacher_target_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True),
        "teacher_full_df": teacher_full_df.sort_values(["signal_date", "symbol"]).reset_index(drop=True),
        "teacher_alpha_df": teacher_alpha_df.sort_values("signal_date").reset_index(drop=True),
        "teacher_alpha_sampled_df": teacher_alpha_sampled_df.sort_values("signal_date").reset_index(drop=True),
        "sampled_dates": sampled_dates,
        "sampling_strategy": alignment_sampling_strategy,
        "sampling_summary": sampling_summary,
        "teacher_selected_mean_return": teacher_selected_mean_return,
        "teacher_not_selected_mean_return": teacher_not_selected_mean_return,
        "teacher_uplift_vs_not_selected": float(teacher_selected_mean_return - teacher_not_selected_mean_return),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, cache_path)
        replay_mod._progress(f"scope alignment cache saved path={cache_path}")
    return bundle


def _scoped_state_for_teacher(scope_entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not scope_entry:
        return None
    return {
        "curriculum": "iterative_v5_scoped_scope_only",
        "global_lesson_zone_lines": [],
        "teacher_scope_order": [scope_entry.get("round_id")],
        "teacher_scopes": [dict(scope_entry)],
        "review_cards_for_prompt": list(scope_entry.get("review_cards_for_prompt") or []),
    }


def _run_one_scope(
    *,
    base_config: ApprenticeReplayConfig,
    bundle: Dict[str, Any],
    scope_entry: Optional[Dict[str, Any]],
    run_tag: str,
    with_lesson: bool,
) -> replay_mod.ReplaySummary:
    config = replace(
        base_config,
        mode="single",
        teacher_round_ids=[bundle["source_round_id"]],
        negative_teacher_round_ids=[],
        candidate_source="baseline_signal",
        warmup_sample_count=0,
        run_tag=run_tag,
    )
    prompt_state = _scoped_state_for_teacher(scope_entry) if with_lesson else None
    warmup_review_cards = list(scope_entry.get("review_cards_for_prompt") or []) if (with_lesson and scope_entry) else []
    return replay_mod._run_replay(
        config=config,
        candidate_pool_df=bundle["candidate_pool_df"],
        teacher_target_df=bundle["teacher_target_df"],
        teacher_full_df=bundle["teacher_full_df"],
        prompt_builder=replay_mod._daily_prompt_single,
        prompt_builder_kwargs={
            "meta": bundle["meta"],
            "negative_metas": [],
            "warmup_lessons": [],
            "warmup_review_cards": warmup_review_cards,
            "scoped_warmup_state": prompt_state,
            "current_scope_round_id": bundle["final_round_id"],
        },
    )


def _resume_or_run_one_scope(
    *,
    base_config: ApprenticeReplayConfig,
    bundle: Dict[str, Any],
    scope_entry: Optional[Dict[str, Any]],
    run_tag: str,
    with_lesson: bool,
) -> replay_mod.ReplaySummary:
    config = replace(
        base_config,
        mode="single",
        teacher_round_ids=[bundle["source_round_id"]],
        negative_teacher_round_ids=[],
        candidate_source="baseline_signal",
        warmup_sample_count=0,
        run_tag=run_tag,
    )
    report_dir = replay_mod.REPORT_ROOT / config.run_id()
    summary_path = report_dir / "summary.json"
    if summary_path.exists():
        replay_mod._progress(
            f"scope alignment reuse existing summary run_id={config.run_id()} path={summary_path}"
        )
        return _load_replay_summary(summary_path)
    return _run_one_scope(
        base_config=base_config,
        bundle=bundle,
        scope_entry=scope_entry,
        run_tag=run_tag,
        with_lesson=with_lesson,
    )


def _scope_result_row(
    *,
    bundle: Dict[str, Any],
    summary: replay_mod.ReplaySummary,
    with_lesson: bool,
) -> Dict[str, Any]:
    llm_vs_teacher_pool_not_selected = float(summary.llm_selected_mean_return - bundle["teacher_not_selected_mean_return"])
    gap_to_teacher_pool_uplift = float(bundle["teacher_uplift_vs_not_selected"] - llm_vs_teacher_pool_not_selected)
    return {
        "final_round_id": bundle["final_round_id"],
        "source_round_id": bundle["source_round_id"],
        "teacher_family": bundle["meta"].get("research_family"),
        "teacher_template": bundle["meta"].get("sample_template"),
        "sampled_days": len(bundle["sampled_dates"]),
        "sampling_strategy": bundle.get("sampling_strategy", ""),
        "signal_pool_per_day": int(bundle.get("sampling_summary", {}).get("signal_pool_per_day", 0)),
        "top_teacher_return_days": int(bundle.get("sampling_summary", {}).get("top_teacher_return_days", 0)),
        "bottom_teacher_return_days": int(bundle.get("sampling_summary", {}).get("bottom_teacher_return_days", 0)),
        "neutral_pool_size": int(bundle.get("sampling_summary", {}).get("neutral_pool_size", 0)),
        "with_lesson": bool(with_lesson),
        "teacher_selected_mean_return": bundle["teacher_selected_mean_return"],
        "teacher_not_selected_mean_return": bundle["teacher_not_selected_mean_return"],
        "teacher_uplift_vs_not_selected": bundle["teacher_uplift_vs_not_selected"],
        "llm_selected_mean_return": float(summary.llm_selected_mean_return),
        "llm_vs_teacher_pool_not_selected": llm_vs_teacher_pool_not_selected,
        "gap_to_teacher_pool_uplift": gap_to_teacher_pool_uplift,
        "llm_vs_llm_not_selected": float(summary.uplift_vs_not_selected),
        "llm_vs_teacher_selected": float(summary.uplift_vs_teacher_selected),
        "mean_daily_jaccard": float(summary.mean_daily_jaccard),
        "mean_daily_precision": float(summary.mean_daily_precision),
        "mean_daily_recall": float(summary.mean_daily_recall),
        "exact_match_rate": float(summary.exact_match_rate),
        "llm_final_nav": float(summary.llm_final_nav),
        "teacher_target_final_nav": float(summary.teacher_target_final_nav),
        "run_id": summary.run_id,
        "report_dir": str((replay_mod.REPORT_ROOT / summary.run_id).resolve()),
    }


def _mean_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        "sampled_days",
        "teacher_selected_mean_return",
        "teacher_not_selected_mean_return",
        "teacher_uplift_vs_not_selected",
        "llm_selected_mean_return",
        "llm_vs_teacher_pool_not_selected",
        "gap_to_teacher_pool_uplift",
        "llm_vs_llm_not_selected",
        "llm_vs_teacher_selected",
        "mean_daily_jaccard",
        "mean_daily_precision",
        "mean_daily_recall",
        "exact_match_rate",
        "llm_final_nav",
        "teacher_target_final_nav",
    ]
    mean_payload: Dict[str, Any] = {"teacher_count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        mean_payload[key] = float(sum(values) / len(values)) if values else 0.0
    return mean_payload


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4%}"


def _fmt_num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _sampling_note(
    *,
    alignment_sampling_strategy: str,
    per_scope_sample_count: int,
    signal_pool_per_day: int,
) -> str:
    if alignment_sampling_strategy == "neutral36_topq5_v1":
        neutral_pool_size = max(36, int(per_scope_sample_count))
        return (
            f"Each scope first samples {int(signal_pool_per_day)} candidates per day with cross-bucket quotas, "
            f"then keeps the {neutral_pool_size} days whose sampled day mean return is closest to zero, "
            f"and finally chooses the top {int(per_scope_sample_count)} days by teacher-Q5 mean return within that neutral pool."
        )
    return (
        f"Each scope samples the highest teacher-return days and the lowest teacher-return days, "
        f"split {int(per_scope_sample_count // 2)}/{int(per_scope_sample_count - (per_scope_sample_count // 2))} "
        f"when per_scope_sample_count={int(per_scope_sample_count)}."
    )


def _comparison_md(
    *,
    title: str,
    per_teacher_rows: List[Dict[str, Any]],
    aggregate_no_lesson: Dict[str, Any],
    aggregate_with_lesson: Optional[Dict[str, Any]],
) -> str:
    with_enabled = bool(aggregate_with_lesson)
    with_uplift = None if not with_enabled else aggregate_with_lesson.get("llm_vs_teacher_pool_not_selected")
    with_gap = None if not with_enabled else aggregate_with_lesson.get("gap_to_teacher_pool_uplift")
    with_jaccard = None if not with_enabled else aggregate_with_lesson.get("mean_daily_jaccard")
    delta_uplift = None if not with_enabled else float(with_uplift) - float(aggregate_no_lesson["llm_vs_teacher_pool_not_selected"])
    lines = [
        f"# {title}",
        "",
        f"- sampling_strategy: `{per_teacher_rows[0]['sampling_strategy'] if per_teacher_rows else ''}`",
        f"- per_scope_day_mix: `top={per_teacher_rows[0]['top_teacher_return_days'] if per_teacher_rows else 0}, bottom={per_teacher_rows[0]['bottom_teacher_return_days'] if per_teacher_rows else 0}`",
        f"- signal_pool_per_day: `{per_teacher_rows[0]['signal_pool_per_day'] if per_teacher_rows else 0}`",
        f"- neutral_pool_size: `{per_teacher_rows[0]['neutral_pool_size'] if per_teacher_rows else 0}`",
        "",
        "## Aggregate Mean Across 4 Teachers",
        "",
        f"- teacher_uplift_vs_not_selected: `{aggregate_no_lesson['teacher_uplift_vs_not_selected']:.4%}`",
        f"- no_lesson llm_vs_teacher_pool_not_selected: `{aggregate_no_lesson['llm_vs_teacher_pool_not_selected']:.4%}`",
        f"- with_lesson llm_vs_teacher_pool_not_selected: `{_fmt_pct(with_uplift)}`",
        f"- no_lesson gap_to_teacher_pool_uplift: `{aggregate_no_lesson['gap_to_teacher_pool_uplift']:.4%}`",
        f"- with_lesson gap_to_teacher_pool_uplift: `{_fmt_pct(with_gap)}`",
        f"- delta llm_vs_teacher_pool_not_selected: `{_fmt_pct(delta_uplift)}`",
        f"- no_lesson mean_daily_jaccard: `{aggregate_no_lesson['mean_daily_jaccard']:.4f}`",
        f"- with_lesson mean_daily_jaccard: `{_fmt_num(with_jaccard)}`",
        "",
        "## Per Teacher",
        "",
        "| teacher | family | template | teacher_vs_not_selected | no_lesson_llm_vs_teacher_pool_not_selected | with_lesson_llm_vs_teacher_pool_not_selected | delta | no_lesson_jaccard | with_lesson_jaccard |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows_by_teacher = {}
    for row in per_teacher_rows:
        rows_by_teacher.setdefault(row["final_round_id"], {})["with" if row["with_lesson"] else "without"] = row
    for teacher_id, item in rows_by_teacher.items():
        pre = item["without"]
        post = item.get("with")
        post_uplift = None if post is None else post["llm_vs_teacher_pool_not_selected"]
        post_jaccard = None if post is None else post["mean_daily_jaccard"]
        post_delta = None if post is None else post["llm_vs_teacher_pool_not_selected"] - pre["llm_vs_teacher_pool_not_selected"]
        lines.append(
            "| "
            + " | ".join(
                [
                    teacher_id,
                    str(pre["teacher_family"]),
                    str(pre["teacher_template"]),
                    f"{pre['teacher_uplift_vs_not_selected']:.4%}",
                    f"{pre['llm_vs_teacher_pool_not_selected']:.4%}",
                    _fmt_pct(post_uplift),
                    _fmt_pct(post_delta),
                    f"{pre['mean_daily_jaccard']:.4f}",
                    _fmt_num(post_jaccard),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _scope_prompt_kwargs(
    *,
    bundle: Dict[str, Any],
    scope_entry: Optional[Dict[str, Any]],
    with_lesson: bool,
) -> Dict[str, Any]:
    prompt_state = _scoped_state_for_teacher(scope_entry) if with_lesson else None
    warmup_review_cards = list(scope_entry.get("review_cards_for_prompt") or []) if (with_lesson and scope_entry) else []
    return {
        "meta": bundle["meta"],
        "negative_metas": [],
        "warmup_lessons": [],
        "warmup_review_cards": warmup_review_cards,
        "scoped_warmup_state": prompt_state,
        "current_scope_round_id": bundle["final_round_id"],
    }


def _request_scope_task_batch(
    *,
    task_batch: List[Dict[str, Any]],
    workers: int,
    global_batch_index: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {}
        for task in task_batch:
            future = executor.submit(
                replay_mod._request_one_day,
                day_candidates=task["day_candidates"],
                decision_date=task["decision_date"],
                config=task["config"],
                prompt_builder=replay_mod._daily_prompt_single,
                prompt_builder_kwargs=task["prompt_builder_kwargs"],
                api_key=task["api_key"],
                api_cache_dir=task["api_cache_dir"],
                empty_rows=task["empty_rows"],
                reuse_api_cache_override=bool(task.get("reuse_api_cache", True)),
            )
            future_map[future] = task
        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            result["task_key"] = task["task_key"]
            result["run_key"] = task["run_key"]
            result["rerun_count"] = int(task.get("rerun_count", 0))
            result["parallel_batch_index"] = int(global_batch_index)
            results.append(result)
    results.sort(key=lambda item: (str(item["run_key"]), pd.Timestamp(item["decision_date"])))
    return results


def _run_scope_suite_flat_batches(
    *,
    scope_specs: List[Dict[str, Any]],
    base_config: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
    scope_suite_dir: Path,
    bundle_cache_dir: Optional[Path],
    run_tag_base: str,
    per_scope_sample_count: int,
    alignment_sampling_strategy: str,
    signal_pool_per_day: int,
    with_lesson_modes: List[bool],
) -> Dict[str, Any]:
    api_key = replay_mod._apprentice_api_key()
    run_jobs: List[Dict[str, Any]] = []
    pending: deque[Dict[str, Any]] = deque()
    global_task_count = 0

    for scope_index, spec in enumerate(scope_specs, start=1):
        final_round_id = spec["final_round_id"]
        source_round_id = spec["source_round_id"]
        scope_entry = dict(spec.get("scope_entry") or {})
        replay_mod._progress(
            "scope alignment prepare "
            f"scope={scope_index}/{len(scope_specs)} final_round={final_round_id} source_round={source_round_id}"
        )
        bundle = _prepare_scope_bundle(
            master_df=master_df,
            base_config=base_config,
            final_round_id=final_round_id,
            source_round_id=source_round_id,
            sample_count=per_scope_sample_count,
            cache_dir=bundle_cache_dir,
            alignment_sampling_strategy=alignment_sampling_strategy,
            signal_pool_per_day=signal_pool_per_day,
        )
        teacher_days_csv = scope_suite_dir / f"{final_round_id}_teacher_comfort_days.csv"
        bundle["teacher_alpha_sampled_df"].to_csv(teacher_days_csv, index=False)
        replay_mod._progress(
            "scope alignment sample ready "
            f"scope={scope_index}/{len(scope_specs)} final_round={final_round_id} sampled_days={len(bundle['sampled_dates'])} "
            f"teacher_uplift={bundle['teacher_uplift_vs_not_selected']:.4%}"
        )

        for with_lesson in with_lesson_modes:
            run_tag = f"{run_tag_base}_{final_round_id}_{'withlesson' if with_lesson else 'nolesson'}"
            config = replace(
                base_config,
                mode="single",
                teacher_round_ids=[bundle["source_round_id"]],
                negative_teacher_round_ids=[],
                candidate_source="baseline_signal",
                warmup_sample_count=0,
                inline_day_retry_enabled=False,
                run_tag=run_tag,
            )
            report_dir = replay_mod.REPORT_ROOT / config.run_id()
            api_cache_dir = report_dir / "api_calls"
            report_dir.mkdir(parents=True, exist_ok=True)
            api_cache_dir.mkdir(parents=True, exist_ok=True)

            prompt_kwargs = _scope_prompt_kwargs(bundle=bundle, scope_entry=scope_entry, with_lesson=with_lesson)
            day_groups = [
                (pd.Timestamp(decision_date), day_candidates.copy().reset_index(drop=True))
                for decision_date, day_candidates in bundle["candidate_pool_df"].groupby("signal_date", sort=True)
            ]
            teacher_target_by_date = {
                pd.Timestamp(decision_date): group["symbol"].astype(str).tolist()
                for decision_date, group in bundle["teacher_target_df"].groupby("signal_date", sort=True)
            }
            run_key = config.run_id()
            job = {
                "run_key": run_key,
                "config": config,
                "bundle": bundle,
                "with_lesson": bool(with_lesson),
                "teacher_target_by_date": teacher_target_by_date,
                "task_keys": [],
            }
            run_jobs.append(job)

            for decision_date, day_candidates in day_groups:
                task_key = f"{run_key}::{decision_date.strftime('%Y-%m-%d')}"
                job["task_keys"].append(task_key)
                pending.append(
                    {
                        "task_key": task_key,
                        "run_key": run_key,
                        "decision_date": pd.Timestamp(decision_date),
                        "day_candidates": day_candidates,
                        "config": config,
                        "prompt_builder_kwargs": prompt_kwargs,
                        "api_key": api_key,
                        "api_cache_dir": api_cache_dir,
                        "empty_rows": bundle["candidate_pool_df"].iloc[0:0].copy(),
                        "reuse_api_cache": True,
                        "rerun_count": 0,
                        "api_calls_cum": 0,
                        "api_cache_hits_cum": 0,
                    }
                )
                global_task_count += 1

    global_batch_index = 0
    finalized_results: Dict[str, Dict[str, Any]] = {}
    max_reruns = max(0, int(base_config.api_failed_rerun_rounds))
    batch_workers = max(1, int(base_config.api_parallel_workers))

    replay_mod._progress(
        "scope alignment flat scheduler start "
        f"run_tag={run_tag_base} total_tasks={global_task_count} batch_workers={batch_workers} max_reruns={max_reruns}"
    )

    while pending:
        batch_size = min(batch_workers, len(pending))
        task_batch = [pending.popleft() for _ in range(batch_size)]
        global_batch_index += 1
        results = _request_scope_task_batch(
            task_batch=task_batch,
            workers=batch_workers,
            global_batch_index=global_batch_index,
        )
        requeued = 0
        finalized = 0
        for result in results:
            needs_rerun = replay_mod._parallel_result_needs_rerun(result)
            rerun_count = int(result.get("rerun_count", 0))
            source_task = next(task for task in task_batch if task["task_key"] == result["task_key"])
            api_calls_total = int(source_task.get("api_calls_cum", 0)) + int(result.get("api_calls", 0))
            api_cache_hits_total = int(source_task.get("api_cache_hits_cum", 0)) + int(result.get("api_cache_hits", 0))
            if needs_rerun and rerun_count < max_reruns:
                pending.append(
                    {
                        "task_key": result["task_key"],
                        "run_key": result["run_key"],
                        "decision_date": pd.Timestamp(result["decision_date"]),
                        "day_candidates": result["day_candidates"].copy().reset_index(drop=True),
                        "config": source_task["config"],
                        "prompt_builder_kwargs": source_task["prompt_builder_kwargs"],
                        "api_key": api_key,
                        "api_cache_dir": source_task["api_cache_dir"],
                        "empty_rows": source_task["empty_rows"],
                        "reuse_api_cache": False,
                        "rerun_count": rerun_count + 1,
                        "api_calls_cum": api_calls_total,
                        "api_cache_hits_cum": api_cache_hits_total,
                    }
                )
                requeued += 1
            else:
                result["api_calls_total"] = api_calls_total
                result["api_cache_hits_total"] = api_cache_hits_total
                finalized_results[result["task_key"]] = result
                finalized += 1
        replay_mod._progress(
            "scope alignment flat batch done "
            f"run_tag={run_tag_base} batch={global_batch_index} size={len(task_batch)} "
            f"requeued={requeued} finalized={finalized} pending={len(pending)}"
        )

    if len(finalized_results) != global_task_count:
        raise RuntimeError(
            f"flat scheduler finalized {len(finalized_results)} tasks, expected {global_task_count}"
        )

    per_teacher_rows: List[Dict[str, Any]] = []
    run_summary_by_key: Dict[str, replay_mod.ReplaySummary] = {}
    scheduler_stats = {
        "total_tasks": int(global_task_count),
        "total_batches": int(global_batch_index),
        "batch_workers": int(batch_workers),
        "max_reruns": int(max_reruns),
    }

    for job in run_jobs:
        bundle = job["bundle"]
        config = job["config"]
        report_dir = replay_mod.REPORT_ROOT / config.run_id()
        api_cache_dir = report_dir / "api_calls"
        selected_parts: List[pd.DataFrame] = []
        llm_decisions: Dict[pd.Timestamp, List[str]] = {}
        teacher_decisions: Dict[pd.Timestamp, List[str]] = {}
        prompt_log_rows: List[Dict[str, Any]] = []
        api_calls = 0
        api_cache_hits = 0
        parse_fallback_days = 0
        parse_failure_days = 0
        query_invoked_days = 0
        query_success_days = 0
        abstain_days = 0
        retry_invoked_days = 0
        retry_success_days = 0

        task_results = [finalized_results[key] for key in job["task_keys"]]
        task_results.sort(key=lambda item: pd.Timestamp(item["decision_date"]))
        for result in task_results:
            decision_date = pd.Timestamp(result["decision_date"])
            day_candidates = result["day_candidates"]
            selected_symbols = list(result["selected_symbols"])
            parse_fallback = bool(result["parse_fallback"])
            abstain = bool(result.get("abstain", False))
            parse_mode = str(result.get("parse_mode", ""))
            failure_reason = str(result.get("failure_reason", ""))
            query_invoked = bool(result.get("query_invoked", False))
            query_success = bool(result.get("query_success", False))
            retry_invoked = bool(result.get("retry_invoked", False))
            retry_success = bool(result.get("retry_success", False))
            content = str(result.get("content", ""))

            api_calls += int(result.get("api_calls_total", result.get("api_calls", 0)))
            api_cache_hits += int(result.get("api_cache_hits_total", result.get("api_cache_hits", 0)))
            if query_invoked:
                query_invoked_days += 1
            if query_success:
                query_success_days += 1
            if retry_invoked:
                retry_invoked_days += 1
            if retry_success:
                retry_success_days += 1
            if abstain:
                abstain_days += 1
            if parse_fallback:
                parse_fallback_days += 1
                parse_failure_days += 1

            llm_decisions[decision_date] = selected_symbols
            teacher_target_symbols = job["teacher_target_by_date"].get(decision_date, [])
            teacher_decisions[decision_date] = teacher_target_symbols
            day_selected = replay_mod._select_day_rows_by_symbols(day_candidates, selected_symbols)
            if not day_selected.empty:
                selected_parts.append(day_selected)

            prompt_log_rows.append(
                {
                    "decision_date": decision_date.strftime("%Y-%m-%d"),
                    "candidate_count": len(day_candidates),
                    "teacher_target_count": len(teacher_target_symbols),
                    "llm_selected_count": len(selected_symbols),
                    "abstain": int(abstain),
                    "parse_fallback": int(parse_fallback),
                    "feature_query_invoked": int(query_invoked),
                    "feature_query_success": int(query_success),
                    "retry_invoked": int(retry_invoked),
                    "retry_success": int(retry_success),
                    "parse_mode": parse_mode,
                    "failure_reason": failure_reason,
                    "parallel_pass_index": int(result.get("rerun_count", 0)),
                    "parallel_batch_index": int(result.get("parallel_batch_index", 0)),
                    "selected_symbols": ",".join(selected_symbols),
                    "teacher_target_symbols": ",".join(teacher_target_symbols),
                    "brief_reason": content[:240],
                }
            )

        llm_selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=bundle["candidate_pool_df"].columns)
        summary = replay_mod._finalize_replay_outputs(
            config=config,
            report_dir=report_dir,
            api_cache_dir=api_cache_dir,
            candidate_pool_df=bundle["candidate_pool_df"],
            teacher_target_df=bundle["teacher_target_df"],
            teacher_full_df=bundle["teacher_full_df"],
            llm_selected_df=llm_selected_df,
            llm_decisions=llm_decisions,
            teacher_decisions=teacher_decisions,
            prompt_log_rows=prompt_log_rows,
            api_calls=api_calls,
            api_cache_hits=api_cache_hits,
            parse_fallback_days=parse_fallback_days,
            parse_failure_days=parse_failure_days,
            query_invoked_days=query_invoked_days,
            query_success_days=query_success_days,
            abstain_days=abstain_days,
            retry_invoked_days=retry_invoked_days,
            retry_success_days=retry_success_days,
        )
        run_summary_by_key[job["run_key"]] = summary
        per_teacher_rows.append(_scope_result_row(bundle=bundle, summary=summary, with_lesson=job["with_lesson"]))

    return {
        "per_teacher_rows": per_teacher_rows,
        "run_summary_by_key": run_summary_by_key,
        "scheduler_stats": scheduler_stats,
    }


def main() -> None:
    args = _build_parser().parse_args()
    selection_json = Path(args.selection_json).expanduser().resolve()
    lesson_artifact_json = Path(args.lesson_artifact_json).expanduser().resolve() if args.lesson_artifact_json else None
    bundle_cache_dir = Path(args.bundle_cache_dir).expanduser().resolve() if args.bundle_cache_dir else None
    if bundle_cache_dir:
        bundle_cache_dir.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(selection_json)
    scoped_warmup_state = _load_scoped_warmup_state(lesson_artifact_json)
    scope_specs = _resolve_scope_specs(selection, scoped_warmup_state)

    base_config = ApprenticeReplayConfig(
        mode="single",
        teacher_round_ids=["placeholder"],
        negative_teacher_round_ids=[],
        start_date=args.start_date,
        end_date=args.end_date,
        candidate_pool_size=args.candidate_pool_size,
        teacher_daily_pick_count=args.teacher_daily_pick_count,
        llm_max_daily_picks=args.llm_max_daily_picks,
        prompt_feature_count=args.prompt_feature_count,
        lesson_feature_count=args.lesson_feature_count,
        api_model=args.api_model,
        api_temperature=args.api_temperature,
        api_max_tokens=args.api_max_tokens,
        force_local_qwen_no_thinking=not args.force_local_qwen_thinking,
        use_line_answer_prefix=args.use_line_answer_prefix,
        private_reasoning_target_tokens=args.private_reasoning_target_tokens,
        private_reasoning_max_tokens_hint=args.private_reasoning_max_tokens_hint,
        prompt_recipe=args.prompt_recipe,
        include_teacher_signal=args.include_teacher_signal,
        candidate_source=args.candidate_source,
        reuse_api_cache=True,
        ignore_holdings_context=args.ignore_holdings_context,
        api_parallel_workers=args.api_parallel_workers,
        api_failed_rerun_rounds=args.api_failed_rerun_rounds,
        api_failed_rerun_workers=args.api_failed_rerun_workers,
        api_request_max_retries=1,
        inline_day_retry_enabled=False,
        summary_variant=args.summary_variant,
        warmup_sample_count=0,
        warmup_start_date=args.start_date,
        warmup_end_date=args.end_date,
        run_tag="",
    )

    master_df = replay_mod._load_master_dataset()
    replay_mod._progress(f"scope alignment master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")

    scope_suite_dir = replay_mod.REPORT_ROOT / f"scope_alignment_suite_{args.run_tag_base}"
    scope_suite_dir.mkdir(parents=True, exist_ok=True)
    with_lesson_modes = [False, True] if scoped_warmup_state else [False]
    flat_run = _run_scope_suite_flat_batches(
        scope_specs=scope_specs,
        base_config=base_config,
        master_df=master_df,
        scope_suite_dir=scope_suite_dir,
        bundle_cache_dir=bundle_cache_dir,
        run_tag_base=args.run_tag_base,
        per_scope_sample_count=args.per_scope_sample_count,
        alignment_sampling_strategy=args.alignment_sampling_strategy,
        signal_pool_per_day=args.signal_pool_per_day,
        with_lesson_modes=with_lesson_modes,
    )
    per_teacher_rows = list(flat_run["per_teacher_rows"])

    no_lesson_rows = [row for row in per_teacher_rows if not row["with_lesson"]]
    with_lesson_rows = [row for row in per_teacher_rows if row["with_lesson"]]
    aggregate_no_lesson = _mean_row(no_lesson_rows)
    aggregate_with_lesson = _mean_row(with_lesson_rows) if with_lesson_rows else None

    comparison = {
        "suite_dir": str(scope_suite_dir.resolve()),
        "selection_json": str(selection_json),
        "lesson_artifact_json": str(lesson_artifact_json) if lesson_artifact_json else "",
        "api_model": args.api_model,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "per_scope_sample_count": int(args.per_scope_sample_count),
        "candidate_source": args.candidate_source,
        "prompt_recipe": args.prompt_recipe,
        "with_lesson_modes": with_lesson_modes,
        "sampling_strategy": args.alignment_sampling_strategy,
        "sampling_note": _sampling_note(
            alignment_sampling_strategy=args.alignment_sampling_strategy,
            per_scope_sample_count=args.per_scope_sample_count,
            signal_pool_per_day=args.signal_pool_per_day,
        ),
        "signal_pool_per_day": int(args.signal_pool_per_day),
        "scheduler_mode": "flat_global_batch_queue_v1",
        "scheduler_stats": flat_run["scheduler_stats"],
        "per_teacher_rows": per_teacher_rows,
        "aggregate_no_lesson": aggregate_no_lesson,
        "aggregate_with_lesson": aggregate_with_lesson,
    }
    comparison_json = scope_suite_dir / "scope_alignment_comparison.json"
    comparison_md = scope_suite_dir / "scope_alignment_comparison.md"
    per_teacher_csv = scope_suite_dir / "scope_alignment_teacher_rows.csv"
    _write_json(comparison_json, comparison)
    comparison_md.write_text(
        _comparison_md(
            title=f"Scope Alignment Comparison - {args.run_tag_base}",
            per_teacher_rows=per_teacher_rows,
            aggregate_no_lesson=aggregate_no_lesson,
            aggregate_with_lesson=aggregate_with_lesson,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(per_teacher_rows).to_csv(per_teacher_csv, index=False)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
