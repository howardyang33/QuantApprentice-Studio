#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import pandas as pd

from quant_toolkit._paths import env_path, project_root
from quant_toolkit.apprentice_loop import ApprenticeReplayConfig
from quant_toolkit.apprentice_loop import replay as replay_mod


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_seeds(text: str) -> List[int]:
    values = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("seeds must not be empty")
    return values


def _load_selection(selection_json: Path) -> Dict[str, Any]:
    payload = _load_json(selection_json)
    frozen = list(payload.get("frozen_round_ids") or [])
    positive = list(payload.get("positive_round_ids") or [])
    negative = list(payload.get("negative_round_ids") or [])
    if not frozen:
        raise ValueError(f"selection_json has no frozen_round_ids: {selection_json}")
    scopes: List[Dict[str, Any]] = []
    for idx, frozen_round_id in enumerate(frozen):
        if idx < len(positive) and str(positive[idx]).strip():
            source_round_id = str(positive[idx]).strip()
        else:
            source_round_id = replay_mod._source_round_id_for_round(str(frozen_round_id))
        scopes.append(
            {
                "final_round_id": str(frozen_round_id).strip(),
                "source_round_id": str(source_round_id).strip(),
            }
        )
    return {
        "frozen_teachers": [str(x).strip() for x in frozen],
        "negative_teachers": [str(x).strip() for x in negative],
        "scope_specs": scopes,
        "raw": payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scorefit-style scope alignment experiment suite")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--suite-root", default="")
    parser.add_argument("--bundle-cache-dir", default="")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--start-date", default="2020-01-02")
    parser.add_argument("--end-date", default="2022-12-30")
    parser.add_argument("--alignment-day-count", type=int, default=30)
    parser.add_argument(
        "--alignment-sampling-strategy",
        default="day_mean_return_high_to_low_even_spacing_v1",
        choices=[
            "day_mean_return_high_to_low_even_spacing_v1",
            "neutral36_topq5_v1",
        ],
    )
    parser.add_argument("--signal-pool-per-day", type=int, default=15)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--teacher-daily-pick-count", type=int, default=4)
    parser.add_argument("--llm-max-daily-picks", type=int, default=4)
    parser.add_argument("--prompt-feature-count", type=int, default=8)
    parser.add_argument("--lesson-feature-count", type=int, default=20)
    parser.add_argument("--lock-days", type=int, default=5)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-max-tokens", type=int, default=8192)
    parser.add_argument("--api-parallel-workers", type=int, default=128)
    parser.add_argument("--api-failed-rerun-rounds", type=int, default=4)
    parser.add_argument("--api-request-max-retries", type=int, default=1)
    parser.add_argument("--summary-variant", default="enriched_v2")
    parser.add_argument("--private-reasoning-target-tokens", type=int, default=0)
    parser.add_argument("--private-reasoning-max-tokens-hint", type=int, default=0)
    parser.add_argument("--force-local-qwen-no-thinking", action="store_true")
    parser.add_argument("--enable-local-qwen-thinking", action="store_true")
    parser.add_argument("--warmup-sample-count", type=int, default=256)
    parser.add_argument("--warmup-batch-size", type=int, default=8)
    parser.add_argument("--warmup-signal-pool-per-day", type=int, default=15)
    parser.add_argument("--warmup-review-memory-limit", type=int, default=12)
    parser.add_argument("--warmup-lesson-zone-max-lines", type=int, default=16)
    parser.add_argument("--warmup-lesson-rewrite-max-tokens", type=int, default=8192)
    parser.add_argument("--warmup-signal-score-max-tokens", type=int, default=2048)
    parser.add_argument("--warmup-retained-case-max-per-tier", type=int, default=4)
    parser.add_argument(
        "--scorefit-variant",
        choices=[
            "v1",
            "v2_schemafix",
            "v3_tailaware",
            "v4_compact_tailaware",
            "v5_stability_guard",
            "v6_bestguard_explore",
            "v7_bestguard_explore_longbatch",
        ],
        default="v1",
    )
    parser.add_argument("--reuse-warmup-run-tag-base", default="")
    parser.add_argument("--reuse-lesson0-suite-root", default="")
    parser.add_argument("--reuse-baseline-suite-root", default="")
    parser.add_argument("--skip-no-lesson", action="store_true")
    parser.add_argument("--skip-lesson0-alignment", action="store_true")
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument(
        "--final-lesson-selection",
        default="best_composite",
        choices=["last", "best_composite", "llm_history_synthesis"],
    )
    parser.add_argument("--final-lesson-composite-lambda", type=float, default=1.0)
    parser.add_argument(
        "--final-lesson-composite-method",
        default="zscore_sum",
        choices=["raw_linear", "zscore_sum"],
    )
    return parser


def _base_config_from_args(args: argparse.Namespace, selection: Mapping[str, Any], *, sample_seed: int) -> ApprenticeReplayConfig:
    return ApprenticeReplayConfig(
        mode="multi",
        teacher_round_ids=list(selection["frozen_teachers"]),
        negative_teacher_round_ids=list(selection["negative_teachers"]),
        start_date=args.start_date,
        end_date=args.end_date,
        candidate_pool_size=args.candidate_pool_size,
        teacher_daily_pick_count=args.teacher_daily_pick_count,
        llm_max_daily_picks=args.llm_max_daily_picks,
        lock_days=args.lock_days,
        api_model=args.api_model,
        api_temperature=args.api_temperature,
        api_max_tokens=args.api_max_tokens,
        force_local_qwen_no_thinking=(not args.enable_local_qwen_thinking) if "qwen" in args.api_model.lower() else True,
        private_reasoning_target_tokens=args.private_reasoning_target_tokens,
        private_reasoning_max_tokens_hint=args.private_reasoning_max_tokens_hint,
        include_teacher_signal=False,
        candidate_source="teacher_ranked",
        prompt_feature_count=args.prompt_feature_count,
        lesson_feature_count=args.lesson_feature_count,
        ignore_holdings_context=True,
        api_parallel_workers=args.api_parallel_workers,
        api_failed_rerun_rounds=args.api_failed_rerun_rounds,
        api_request_max_retries=args.api_request_max_retries,
        inline_day_retry_enabled=False,
        summary_variant=args.summary_variant,
        warmup_sample_count=args.warmup_sample_count,
        warmup_start_date=args.start_date,
        warmup_end_date=args.end_date,
        warmup_curriculum="scorefit_v1_json",
        warmup_batch_size=args.warmup_batch_size,
        warmup_signal_pool_per_day=args.warmup_signal_pool_per_day,
        warmup_review_memory_limit=args.warmup_review_memory_limit,
        warmup_lesson_zone_max_lines=args.warmup_lesson_zone_max_lines,
        warmup_lesson_rewrite_max_tokens=args.warmup_lesson_rewrite_max_tokens,
        warmup_signal_score_max_tokens=args.warmup_signal_score_max_tokens,
        warmup_retained_case_max_per_tier=args.warmup_retained_case_max_per_tier,
        scorefit_variant=args.scorefit_variant,
        sample_seed=int(sample_seed),
    )


def _teacher_explainability_bundle(
    *,
    base_config: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
    final_round_id: str,
    source_round_id: str,
    alignment_day_count: int,
    signal_pool_per_day: int,
    alignment_sampling_strategy: str,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = (
            f"{final_round_id}_{source_round_id}_{base_config.start_date.replace('-', '')}_{base_config.end_date.replace('-', '')}"
            f"_align{alignment_day_count}_x{signal_pool_per_day}_{alignment_sampling_strategy}_scorefit_v2.joblib"
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            replay_mod._progress(f"scorefit scope bundle cache hit path={cache_path}")
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
    scope_frame, scope_meta = replay_mod._build_single_teacher_frame(scope_cfg, master_df)
    scope_domain = replay_mod._scope_domain_card(
        scope_meta,
        scope_round_id=final_round_id,
        source_round_id=source_round_id,
    )
    explainability_summary, explainability_round_id = replay_mod._resolve_explainability_summary(
        preferred_round_ids=[final_round_id, source_round_id],
        fallback_summary=scope_meta.get("factor_analysis_summary", {}),
    )
    scope_domain["explainability_round_id"] = explainability_round_id or str(final_round_id)
    feature_cols = replay_mod._lesson_feature_columns(scope_frame)

    day_rows: List[Dict[str, Any]] = []
    for signal_date, day_df in scope_frame.groupby("signal_date", sort=True):
        sampled_day = replay_mod._scorefit_sample_day_signals(day_df.copy(), x=signal_pool_per_day, sample_seed=0)
        if sampled_day.empty:
            continue
        q5_rows = sampled_day[sampled_day["bucket"] == 5]
        day_rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "sampled_signal_count": int(len(sampled_day)),
                "day_sample_mean_return": float(sampled_day["future_return_5d"].mean()),
                "day_q5_mean_return": float(q5_rows["future_return_5d"].mean()) if not q5_rows.empty else 0.0,
            }
        )
    day_table = pd.DataFrame(day_rows).sort_values("signal_date").reset_index(drop=True)
    if len(day_table) < alignment_day_count:
        raise ValueError(
            f"scope {final_round_id} has only {len(day_table)} evaluable days, fewer than requested {alignment_day_count}"
        )
    day_table["abs_day_sample_mean_return"] = day_table["day_sample_mean_return"].abs()
    if alignment_sampling_strategy == "day_mean_return_high_to_low_even_spacing_v1":
        ranked = day_table.sort_values(
            ["day_sample_mean_return", "signal_date"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        keep_idxs = replay_mod._scorefit_sample_uniform_indices(len(ranked), alignment_day_count, sample_seed=0)
        selected_days_df = ranked.iloc[keep_idxs].copy().reset_index(drop=True)
        selected_days_df["rank_position_by_day_mean"] = list(keep_idxs)
        selected_days_df = selected_days_df.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    elif alignment_sampling_strategy == "neutral36_topq5_v1":
        neutral_pool_size = max(36, int(alignment_day_count))
        if len(day_table) < neutral_pool_size:
            raise ValueError(
                f"scope {final_round_id} has only {len(day_table)} evaluable days, fewer than neutral pool size {neutral_pool_size}"
            )
        neutral_pool = day_table.sort_values(
            ["abs_day_sample_mean_return", "signal_date"],
            ascending=[True, True],
            kind="mergesort",
        ).head(neutral_pool_size).copy()
        neutral_pool = neutral_pool.reset_index(drop=True)
        neutral_pool["neutral_pool_rank"] = list(range(1, len(neutral_pool) + 1))
        selected_days_df = neutral_pool.sort_values(
            ["day_q5_mean_return", "abs_day_sample_mean_return", "signal_date"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(alignment_day_count).copy()
        selected_days_df = selected_days_df.reset_index(drop=True)
        selected_days_df["q5_rank_within_neutral_pool"] = list(range(1, len(selected_days_df) + 1))
        selected_days_df = selected_days_df.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    else:
        raise ValueError(f"unsupported alignment sampling strategy: {alignment_sampling_strategy}")

    bundle = {
        "final_round_id": final_round_id,
        "source_round_id": source_round_id,
        "scope_frame": scope_frame,
        "scope_meta": scope_meta,
        "scope_domain": scope_domain,
        "explainability_summary": explainability_summary,
        "feature_cols": feature_cols,
        "selected_days_df": selected_days_df,
        "alignment_sampling_strategy": alignment_sampling_strategy,
    }
    if cache_path is not None:
        joblib.dump(bundle, cache_path)
        replay_mod._progress(f"scorefit scope bundle cache saved path={cache_path}")
    return bundle


def _build_lesson0_scoped_state(
    *,
    base_config: ApprenticeReplayConfig,
    scope_bundles: Sequence[Mapping[str, Any]],
    output_dir: Path,
    reuse_source_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output_path = output_dir / "warmup_scoped_lessons.json"
    if output_path.exists():
        payload = _load_json(output_path)
        if payload.get("curriculum") == "scorefit_v1_json":
            replay_mod._progress(f"lesson0 state cache hit path={output_path}")
            return payload
    if reuse_source_dir is not None:
        reuse_json = reuse_source_dir / "warmup_scoped_lessons.json"
        if reuse_json.exists():
            payload = _load_json(reuse_json)
            if payload.get("curriculum") == "scorefit_v1_json":
                replay_mod._progress(f"lesson0 state reuse hit source={reuse_json} target={output_path}")
                _write_json(output_path, payload)
                reuse_md = reuse_source_dir / "warmup_scoped_lessons.md"
                if reuse_md.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "warmup_scoped_lessons.md").write_text(
                        reuse_md.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                return payload

    background_digest = replay_mod._load_research_experience_digest()
    teacher_scopes: List[Dict[str, Any]] = []
    for scope_index, bundle in enumerate(scope_bundles, start=1):
        lesson_json, lesson0_artifact = replay_mod._scorefit_build_lesson0(
            config=base_config,
            scope_domain=bundle["scope_domain"],
            explainability_summary=bundle["explainability_summary"],
            background_digest=background_digest,
        )
        scope_payload = {
            **dict(bundle["scope_domain"]),
            "scope_index": scope_index,
            "scope_sample_count": 0,
            "sampled_days": 0,
            "scorefit_variant": base_config.scorefit_variant,
            "scope_lesson_zone_lines": replay_mod._scorefit_render_lesson_lines(
                lesson_json,
                limit=base_config.warmup_lesson_zone_max_lines,
            ),
            "review_cards_for_prompt": [],
            "scorefit_lesson_json": lesson_json,
            "revise_history": [],
            "retained_review_entry_count": 0,
            "batch_history": [],
            "lesson0_artifact": lesson0_artifact,
        }
        teacher_scopes.append(scope_payload)

    state = {
        "curriculum": "scorefit_v1_json",
        "scorefit_variant": base_config.scorefit_variant,
        "warmup_sample_count": 0,
        "warmup_batch_size": 0,
        "warmup_signal_pool_per_day": int(base_config.warmup_signal_pool_per_day),
        "sample_seed": int(base_config.sample_seed),
        "global_lesson_zone_lines": [],
        "teacher_scope_order": [bundle["final_round_id"] for bundle in scope_bundles],
        "teacher_scopes": teacher_scopes,
        "review_cards_for_prompt": [],
        "batch_history": [],
        "artifact_dir": str(output_dir.resolve()),
        "lesson_zone_lines": [],
        "mode": "lesson0_only",
    }
    _write_json(output_path, state)

    md_lines: List[str] = [
        "# ScoreFit Lesson0 State",
        "",
        f"- sample_seed: `{base_config.sample_seed}`",
        f"- scorefit_variant: `{base_config.scorefit_variant}`",
        "",
    ]
    for scope in teacher_scopes:
        md_lines.extend(
            [
                f"## {scope.get('round_id')}",
                f"- family: {scope.get('family')}",
                f"- template: {scope.get('template')}",
                f"- style: {scope.get('style_hint')}",
                f"- basic_filter: {scope.get('basic_filter')}",
                "",
                "### Lesson0 Lines",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
            ]
        )
    (output_dir / "warmup_scoped_lessons.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return state


def _run_warmup_from_initial_state(
    *,
    base_config: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
    initial_scoped_state: Mapping[str, Any],
    warmup_run_tag_base: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    warmup_teacher_round_ids = [replay_mod._source_round_id_for_round(round_id) for round_id in base_config.teacher_round_ids]
    run_tag_base = str(warmup_run_tag_base or base_config.run_tag)
    warm_cfg = replace(
        base_config,
        teacher_round_ids=warmup_teacher_round_ids,
        start_date=base_config.warmup_start_date,
        end_date=base_config.warmup_end_date,
        warmup_sample_count=0,
        run_tag=f"{run_tag_base}_warmup{base_config.warmup_sample_count}",
    )
    state = replay_mod._generate_warmup_lessons_scorefit_v1_json(
        config=base_config,
        warm_cfg=warm_cfg,
        master_df=master_df,
        initial_scoped_state=initial_scoped_state,
    )
    return state, replay_mod.REPORT_ROOT / warm_cfg.run_id()


def _scorefit_checkpoint_composite(
    batch_metrics: Mapping[str, Any],
    *,
    composite_lambda: float,
) -> float:
    spearman = float(batch_metrics.get("teacher_score_spearman", 0.0) or 0.0)
    uplift = float(batch_metrics.get("llm_uplift_vs_not_selected", 0.0) or 0.0)
    return float(spearman + float(composite_lambda) * uplift)


def _scorefit_population_std(values: Sequence[float]) -> float:
    nums = [float(v) for v in values]
    if len(nums) <= 1:
        return 0.0
    mean = float(sum(nums) / len(nums))
    variance = float(sum((x - mean) ** 2 for x in nums) / len(nums))
    if variance <= 1e-12:
        return 0.0
    return float(sqrt(variance))


def _scorefit_composite_method_slug(composite_method: str) -> str:
    normalized = str(composite_method).strip().lower()
    return normalized if normalized else "raw_linear"


def _scorefit_annotate_checkpoint_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    composite_lambda: float,
    composite_method: str,
) -> List[Dict[str, Any]]:
    normalized_method = str(composite_method).strip().lower()
    rows = [dict(row) for row in candidates]
    spearman_values = [
        float((row.get("batch_metrics") or {}).get("teacher_score_spearman", 0.0) or 0.0)
        for row in rows
    ]
    uplift_values = [
        float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected", 0.0) or 0.0)
        for row in rows
    ]
    spearman_mean = float(sum(spearman_values) / len(spearman_values)) if spearman_values else 0.0
    uplift_mean = float(sum(uplift_values) / len(uplift_values)) if uplift_values else 0.0
    spearman_std = _scorefit_population_std(spearman_values)
    uplift_std = _scorefit_population_std(uplift_values)

    annotated: List[Dict[str, Any]] = []
    for row, spearman, uplift in zip(rows, spearman_values, uplift_values):
        raw_composite = float(spearman + float(composite_lambda) * uplift)
        z_spearman = float((spearman - spearman_mean) / spearman_std) if spearman_std > 0 else 0.0
        z_uplift = float((uplift - uplift_mean) / uplift_std) if uplift_std > 0 else 0.0
        if normalized_method == "zscore_sum":
            composite_score = float(z_spearman + float(composite_lambda) * z_uplift)
        else:
            composite_score = raw_composite
        enriched = dict(row)
        enriched["raw_composite_score"] = raw_composite
        enriched["z_spearman"] = z_spearman
        enriched["z_uplift"] = z_uplift
        enriched["composite_score"] = composite_score
        enriched["composite_method"] = normalized_method
        annotated.append(enriched)
    return annotated


def _scorefit_selected_state_md(state: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# Selected Final Lesson State",
        "",
        f"- selection_mode: `{state.get('final_lesson_selection_mode', 'last')}`",
        f"- composite_lambda: `{state.get('final_lesson_composite_lambda', 0.0)}`",
        f"- composite_method: `{state.get('final_lesson_composite_method', 'raw_linear')}`",
        "",
    ]
    for scope in list(state.get("teacher_scopes") or []):
        checkpoint = dict(scope.get("selected_checkpoint") or {})
        lines.extend(
            [
                f"## {scope.get('round_id')}",
                f"- selected_batch_index: `{checkpoint.get('selected_batch_index', 'last')}`",
                f"- evaluated_on_batch_index: `{checkpoint.get('evaluated_on_batch_index', 'n/a')}`",
                f"- composite_score: `{float(checkpoint.get('composite_score', 0.0)):.6f}`",
                f"- raw_composite_score: `{float(checkpoint.get('raw_composite_score', 0.0)):.6f}`",
                f"- spearman: `{float(checkpoint.get('teacher_score_spearman', 0.0)):.6f}`",
                f"- uplift_vs_not_selected: `{float(checkpoint.get('llm_uplift_vs_not_selected', 0.0)):.6%}`",
                f"- z_spearman: `{float(checkpoint.get('z_spearman', 0.0)):.6f}`",
                f"- z_uplift: `{float(checkpoint.get('z_uplift', 0.0)):.6f}`",
                "",
                "### Lesson Lines",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _scorefit_recompute_batch_metrics(
    *,
    warmup_run_dir: Path,
    scope_index: int,
    round_id: str,
    batch_index: int,
    lesson_json: Mapping[str, Any],
    lock_days: int,
) -> Dict[str, Any]:
    signal_scores_path = (
        warmup_run_dir
        / f"scope_{int(scope_index):02d}_{round_id}_batch_{int(batch_index):02d}_signal_scores.json"
    )
    if not signal_scores_path.exists():
        return {}
    payload = _load_json(signal_scores_path)
    results = list(payload.get("results") or [])
    if not results:
        return {}
    return replay_mod._scorefit_batch_metrics(
        results=results,
        lesson_payload=dict(lesson_json or {}),
        lock_days=int(lock_days),
    )


def _scorefit_lesson_item_deltas(
    *,
    lesson_json: Mapping[str, Any],
    batch_metrics: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    items = dict(lesson_json.get("items") or {})
    out: List[Dict[str, Any]] = []
    for stat in list(batch_metrics.get("item_stats") or []):
        item_id = str(stat.get("item_id", "")).strip()
        if not item_id:
            continue
        item_payload = dict(items.get(item_id) or {})
        out.append(
            {
                "item_id": item_id,
                "title": str(item_payload.get("title", "")).strip(),
                "role": str(item_payload.get("role", "")).strip(),
                "signals_to_check": list(item_payload.get("signals_to_check") or []),
                "delta_spearman": float(stat.get("ablation_delta", 0.0) or 0.0),
                "delta_uplift_vs_not_selected": float(stat.get("ablation_uplift_delta", 0.0) or 0.0),
                "item_vs_teacher_spearman": float(stat.get("item_vs_teacher_spearman", 0.0) or 0.0),
                "nonzero_rate": float(stat.get("nonzero_rate", 0.0) or 0.0),
                "mean_subscore": float(stat.get("mean_subscore", 0.0) or 0.0),
            }
        )
    return out


def _scorefit_compact_lesson_for_history(lesson_json: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": str(lesson_json.get("schema", "")).strip(),
        "lesson_name": str(lesson_json.get("lesson_name", "")).strip(),
        "items": dict(lesson_json.get("items") or {}),
        "meta_rules": list(lesson_json.get("meta_rules") or []),
        "scoring_notes": list(lesson_json.get("scoring_notes") or []),
    }


def _scorefit_lesson_eval_summary(
    batch_metrics: Mapping[str, Any],
    *,
    composite_lambda: float,
) -> Dict[str, Any]:
    if not batch_metrics:
        return {
            "heldout_metrics_available": False,
            "teacher_score_spearman": 0.0,
            "llm_uplift_vs_not_selected": 0.0,
            "llm_top20_mean_return": 0.0,
            "llm_not_selected_mean_return": 0.0,
            "teacher_q5_mean_return": 0.0,
            "teacher_not_selected_mean_return": 0.0,
            "teacher_q5_uplift_vs_not_selected": 0.0,
            "batch_nav_final": 1.0,
            "successful_signal_count": 0,
            "composite_score": 0.0,
        }
    return {
        "heldout_metrics_available": True,
        "teacher_score_spearman": float(batch_metrics.get("teacher_score_spearman", 0.0) or 0.0),
        "llm_uplift_vs_not_selected": float(batch_metrics.get("llm_uplift_vs_not_selected", 0.0) or 0.0),
        "llm_top20_mean_return": float(batch_metrics.get("llm_top20_mean_return", 0.0) or 0.0),
        "llm_not_selected_mean_return": float(batch_metrics.get("llm_not_selected_mean_return", 0.0) or 0.0),
        "teacher_q5_mean_return": float(batch_metrics.get("teacher_q5_mean_return", 0.0) or 0.0),
        "teacher_not_selected_mean_return": float(batch_metrics.get("teacher_not_selected_mean_return", 0.0) or 0.0),
        "teacher_q5_uplift_vs_not_selected": float(batch_metrics.get("teacher_q5_uplift_vs_not_selected", 0.0) or 0.0),
        "batch_nav_final": float(batch_metrics.get("batch_nav_final", 1.0) or 1.0),
        "successful_signal_count": int(batch_metrics.get("successful_signal_count", 0) or 0),
        "composite_score": _scorefit_checkpoint_composite(
            batch_metrics,
            composite_lambda=composite_lambda,
        ),
    }


def _scorefit_scope_lesson_timeline(
    *,
    lesson0_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    warmup_run_dir: Path,
    round_id: str,
    lock_days: int,
    composite_lambda: float,
    base_config: ApprenticeReplayConfig,
) -> Dict[str, Any]:
    lesson0_scope = replay_mod._find_teacher_scope_entry(lesson0_state, round_id) or {}
    final_scope = replay_mod._find_teacher_scope_entry(final_state, round_id) or {}
    scope_index = int(final_scope.get("scope_index") or lesson0_scope.get("scope_index") or 0)
    scope_domain = dict(final_scope.get("scope_domain") or lesson0_scope.get("scope_domain") or {})
    batch_history = [dict(row) for row in list(final_scope.get("batch_history") or [])]
    timeline: List[Dict[str, Any]] = []

    lesson0_json = dict(lesson0_scope.get("scorefit_lesson_json") or {})
    if lesson0_json:
        eval_metrics = {}
        if batch_history:
            eval_metrics = _scorefit_recompute_batch_metrics(
                warmup_run_dir=warmup_run_dir,
                scope_index=scope_index,
                round_id=round_id,
                batch_index=int(batch_history[0].get("batch_index", 1) or 1),
                lesson_json=lesson0_json,
                lock_days=lock_days,
            )
        timeline.append(
            {
                "lesson_index": 0,
                "lesson_label": "Lesson0",
                "lesson_origin": "initial_lesson0",
                "produced_after_batch_index": 0,
                "evaluated_on_batch_index": (
                    int(batch_history[0].get("batch_index", 1) or 1) if batch_history else None
                ),
                "lesson": _scorefit_compact_lesson_for_history(lesson0_json),
                "eval_summary": _scorefit_lesson_eval_summary(
                    eval_metrics,
                    composite_lambda=composite_lambda,
                ),
                "item_deltas": _scorefit_lesson_item_deltas(
                    lesson_json=lesson0_json,
                    batch_metrics=eval_metrics,
                ),
            }
        )

    for produced_idx, batch_payload in enumerate(batch_history, start=1):
        lesson_json = dict(batch_payload.get("lesson_json") or {})
        eval_metrics = {}
        evaluated_on_batch_index: Optional[int] = None
        if produced_idx < len(batch_history):
            evaluated_on_batch_index = int(batch_history[produced_idx].get("batch_index", produced_idx + 1) or (produced_idx + 1))
            eval_metrics = _scorefit_recompute_batch_metrics(
                warmup_run_dir=warmup_run_dir,
                scope_index=scope_index,
                round_id=round_id,
                batch_index=evaluated_on_batch_index,
                lesson_json=lesson_json,
                lock_days=lock_days,
            )
        timeline.append(
            {
                "lesson_index": produced_idx,
                "lesson_label": f"Lesson{produced_idx}",
                "lesson_origin": f"post_batch_{produced_idx}_revision",
                "produced_after_batch_index": int(batch_payload.get("batch_index", produced_idx) or produced_idx),
                "evaluated_on_batch_index": evaluated_on_batch_index,
                "lesson": _scorefit_compact_lesson_for_history(lesson_json),
                "eval_summary": _scorefit_lesson_eval_summary(
                    eval_metrics,
                    composite_lambda=composite_lambda,
                ),
                "item_deltas": _scorefit_lesson_item_deltas(
                    lesson_json=lesson_json,
                    batch_metrics=eval_metrics,
                ),
            }
        )

    return {
        "scope_index": scope_index,
        "scope_domain": scope_domain,
        "timeline": timeline,
        "revise_history": list(final_scope.get("revise_history") or []),
        "scope_lesson_zone_lines": list(final_scope.get("scope_lesson_zone_lines") or []),
    }


def _scorefit_history_synthesis_md(state: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# LLM History Synthesis Final Lesson State",
        "",
        f"- selection_mode: `{state.get('final_lesson_selection_mode', 'llm_history_synthesis')}`",
        f"- composite_lambda: `{state.get('final_lesson_composite_lambda', 0.0)}`",
        "",
    ]
    for scope in list(state.get("teacher_scopes") or []):
        checkpoint = dict(scope.get("selected_checkpoint") or {})
        lines.extend(
            [
                f"## {scope.get('round_id')}",
                f"- summary: `{str(checkpoint.get('summary', '')).strip()}`",
                f"- evaluated_lessons: `{checkpoint.get('evaluated_lesson_count', 0)}`",
                f"- latest_lesson_included: `{checkpoint.get('latest_lesson_index', 0)}`",
                f"- fallback_used: `{bool(checkpoint.get('fallback_used', False))}`",
                "",
                "### Lesson Lines",
                *(list(scope.get("scope_lesson_zone_lines") or ["none"])),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _scorefit_item_signature(item_delta: Mapping[str, Any]) -> str:
    title = str(item_delta.get("title", "")).strip()
    role = str(item_delta.get("role", "")).strip()
    signals = ",".join(sorted(str(x).strip() for x in list(item_delta.get("signals_to_check") or []) if str(x).strip()))
    return f"{title} | role={role} | signals={signals}"


def _scorefit_history_synthesis_summary(
    lesson_history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    evaluated_rows = [
        dict(row)
        for row in lesson_history
        if bool(((row.get("eval_summary") or {}).get("heldout_metrics_available", False)))
    ]
    skeleton_candidates = sorted(
        [
            {
                "lesson_index": int(row.get("lesson_index", 0) or 0),
                "lesson_label": str(row.get("lesson_label", "")).strip(),
                "llm_uplift_vs_not_selected": float(((row.get("eval_summary") or {}).get("llm_uplift_vs_not_selected", 0.0)) or 0.0),
                "teacher_score_spearman": float(((row.get("eval_summary") or {}).get("teacher_score_spearman", 0.0)) or 0.0),
                "composite_score": float(((row.get("eval_summary") or {}).get("composite_score", 0.0)) or 0.0),
                "item_count": int(len(dict((row.get("lesson") or {}).get("items") or {}))),
            }
            for row in evaluated_rows
        ],
        key=lambda row: (
            float(row.get("llm_uplift_vs_not_selected", 0.0)),
            float(row.get("teacher_score_spearman", 0.0)),
            float(row.get("composite_score", 0.0)),
        ),
        reverse=True,
    )[:4]

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in evaluated_rows:
        lesson_label = str(row.get("lesson_label", "")).strip()
        for item_delta in list(row.get("item_deltas") or []):
            signature = _scorefit_item_signature(item_delta)
            bucket = grouped.setdefault(
                signature,
                {
                    "signature": signature,
                    "title": str(item_delta.get("title", "")).strip(),
                    "role": str(item_delta.get("role", "")).strip(),
                    "signals_to_check": list(item_delta.get("signals_to_check") or []),
                    "lessons": [],
                    "delta_spearman_values": [],
                    "delta_uplift_values": [],
                    "both_positive_count": 0,
                    "positive_uplift_count": 0,
                    "negative_uplift_count": 0,
                    "nonnegative_spearman_count": 0,
                    "negative_spearman_count": 0,
                },
            )
            delta_s = float(item_delta.get("delta_spearman", 0.0) or 0.0)
            delta_u = float(item_delta.get("delta_uplift_vs_not_selected", 0.0) or 0.0)
            bucket["lessons"].append(lesson_label)
            bucket["delta_spearman_values"].append(delta_s)
            bucket["delta_uplift_values"].append(delta_u)
            if delta_u > 0:
                bucket["positive_uplift_count"] += 1
            if delta_u < 0:
                bucket["negative_uplift_count"] += 1
            if delta_s >= 0:
                bucket["nonnegative_spearman_count"] += 1
            if delta_s < 0:
                bucket["negative_spearman_count"] += 1
            if delta_u > 0 and delta_s >= 0:
                bucket["both_positive_count"] += 1

    alpha_items: List[Dict[str, Any]] = []
    risk_items: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        deltas_s = list(bucket.get("delta_spearman_values") or [])
        deltas_u = list(bucket.get("delta_uplift_values") or [])
        count = max(1, len(deltas_s))
        row = {
            "signature": bucket["signature"],
            "title": bucket["title"],
            "role": bucket["role"],
            "signals_to_check": bucket["signals_to_check"],
            "support_count": len(deltas_s),
            "support_lessons": list(bucket["lessons"]),
            "mean_delta_spearman": float(sum(deltas_s) / count),
            "mean_delta_uplift_vs_not_selected": float(sum(deltas_u) / count),
            "positive_uplift_count": int(bucket["positive_uplift_count"]),
            "negative_uplift_count": int(bucket["negative_uplift_count"]),
            "nonnegative_spearman_count": int(bucket["nonnegative_spearman_count"]),
            "negative_spearman_count": int(bucket["negative_spearman_count"]),
            "both_positive_count": int(bucket["both_positive_count"]),
        }
        alpha_items.append(row)
        risk_items.append(row)

    alpha_items = sorted(
        alpha_items,
        key=lambda row: (
            int(row.get("positive_uplift_count", 0)),
            int(row.get("both_positive_count", 0)),
            float(row.get("mean_delta_uplift_vs_not_selected", 0.0)),
            float(row.get("mean_delta_spearman", 0.0)),
            int(row.get("support_count", 0)),
        ),
        reverse=True,
    )[:12]
    risk_items = sorted(
        risk_items,
        key=lambda row: (
            int(row.get("negative_uplift_count", 0)),
            float(row.get("mean_delta_uplift_vs_not_selected", 0.0)),
            int(row.get("negative_spearman_count", 0)),
            float(row.get("mean_delta_spearman", 0.0)),
        ),
    )[:12]
    return {
        "skeleton_candidates": skeleton_candidates,
        "alpha_item_summary": alpha_items,
        "risk_item_summary": risk_items,
    }


def _scorefit_synthesize_final_lesson_state(
    *,
    lesson0_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    fallback_state: Mapping[str, Any],
    output_dir: Path,
    composite_lambda: float,
    base_config: ApprenticeReplayConfig,
) -> Tuple[Dict[str, Any], Path]:
    synthesized_state = dict(final_state)
    synthesized_scopes: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []

    for scope in list(final_state.get("teacher_scopes") or []):
        round_id = str(scope.get("round_id", "")).strip()
        replay_mod._progress(
            "scorefit final history synthesis start "
            f"seed={int(base_config.sample_seed or 0)} scope={round_id}"
        )
        timeline_payload = _scorefit_scope_lesson_timeline(
            lesson0_state=lesson0_state,
            final_state=final_state,
            warmup_run_dir=output_dir,
            round_id=round_id,
            lock_days=int(base_config.lock_days),
            composite_lambda=float(composite_lambda),
            base_config=base_config,
        )
        scope_domain = dict(timeline_payload.get("scope_domain") or {})
        lesson_history = list(timeline_payload.get("timeline") or [])
        history_summary = _scorefit_history_synthesis_summary(lesson_history)
        fallback_scope = replay_mod._find_teacher_scope_entry(fallback_state, round_id) or dict(scope)
        synthesized_scope = dict(scope)
        fallback_used = False
        synthesis_summary = ""
        try:
            system = (
                "You are synthesizing one final structured JSON scorecard from a history of lesson checkpoints. "
                "Your task is to synthesize one final lesson from historical lesson checkpoints. "
                "Priority order is strict and must be obeyed. "
                "Primary objective: higher uplift_vs_not_selected is better. "
                "Secondary objective: higher teacher_score_spearman is better. "
                "Composite is only a weak tie-breaker, not the main target. "
                "Step 1: choose one strong evaluated lesson as the skeleton. Prefer higher uplift first, then higher Spearman, then composite. "
                "Do not start from the latest unevaluated lesson unless the evidence is overwhelming. "
                "Step 2: remove side-effect items from the skeleton. A side-effect item is one that repeatedly hurts uplift, or hurts Spearman, or shows unstable mixed behavior across batches without enough upside. Negative uplift is the more serious side effect. "
                "Step 3: add extra items only when they show repeated multi-batch evidence of helping, especially repeated positive delta_uplift_vs_not_selected, with non-negative or acceptable delta_spearman. Do not add one-batch lucky spikes. "
                "Step 4: keep the final lesson compact and coherent; do not union together every rule that ever looked good once. "
                "Each batch used different samples, so judge stability from repeated support, not from one lucky batch. "
                "Treat the latest unevaluated lesson as tentative because it has no held-out batch result yet. "
                "You may combine the best parts from multiple lessons and rewrite wording to be clearer and more stable, but the backbone should come from one strong uplift-first skeleton lesson. "
                "Return strict JSON only with keys: lesson, synthesis_entry. "
                "lesson must be a full object with keys: schema, lesson_name, teacher_scope, items, meta_rules, scoring_notes. "
                "lesson.items must be a non-empty object keyed by item id like I01, I02. "
                "synthesis_entry must include: summary, skeleton_lesson_label, kept_or_merged_from_lessons, dropped_or_deemphasized_lessons, merged_alpha_item_signatures, stability_notes."
            )
            user = json.dumps(
                {
                    "teacher_scope": scope_domain,
                    "sampling_seed": int(base_config.sample_seed or 0),
                    "objective": {
                        "primary": "maximize uplift_vs_not_selected",
                        "secondary": "maximize teacher_score_spearman",
                        "tie_breaker": f"composite = spearman + {float(composite_lambda):.3f} * uplift_vs_not_selected",
                        "stability_note": (
                            "Different batches use different sampled days and signals. "
                            "Look for stable utility across checkpoints instead of chasing one batch."
                        ),
                        "item_policy": {
                            "remove_side_effect_items_first": True,
                            "side_effect_definition": (
                                "Repeated negative delta_uplift_vs_not_selected, or repeated negative delta_spearman, "
                                "or unstable mixed signs without enough upside. Negative uplift is worse than negative Spearman."
                            ),
                            "merge_policy": (
                                "Only merge items with repeated multi-batch alpha support. Prefer repeated positive uplift contribution first, "
                                "then acceptable or positive Spearman contribution."
                            ),
                        },
                    },
                    "history_summary": history_summary,
                    "lesson_history": lesson_history,
                },
                ensure_ascii=False,
            )
            response = replay_mod._chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                api_key=os.environ.get("APPRENTICE_API_KEY", ""),
                model=base_config.api_model,
                max_tokens=min(int(base_config.api_max_tokens), int(base_config.warmup_lesson_rewrite_max_tokens)),
                temperature=0.0,
                force_local_qwen_no_thinking=base_config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=True,
                max_retries=1,
                seed=int(base_config.sample_seed or 0) or None,
            )
            content = str(response["choices"][0]["message"].get("content", "") or "")
            parsed = replay_mod._extract_json_payload(content)
            lesson_raw = parsed.get("lesson") if isinstance(parsed, Mapping) else None
            synthesis_entry = parsed.get("synthesis_entry") if isinstance(parsed, Mapping) else {}
            lesson_payload = replay_mod._scorefit_postprocess_lesson_payload(
                replay_mod._scorefit_normalize_lesson_payload(
                    lesson_raw if isinstance(lesson_raw, Mapping) else parsed,
                    scope_domain=scope_domain,
                ),
                config=base_config,
            )
            synthesized_scope["scorefit_lesson_json"] = dict(lesson_payload or {})
            synthesized_scope["scope_lesson_zone_lines"] = replay_mod._scorefit_render_lesson_lines(
                lesson_payload,
                limit=base_config.warmup_lesson_zone_max_lines,
            )
            synthesis_summary = (
                str((synthesis_entry or {}).get("summary", "")).strip()
                if isinstance(synthesis_entry, Mapping)
                else ""
            ) or "llm_history_synthesis"
            replay_mod._progress(
                "scorefit final history synthesis done "
                f"seed={int(base_config.sample_seed or 0)} scope={round_id} fallback=False"
            )
        except Exception as exc:
            fallback_used = True
            synthesized_scope = dict(fallback_scope)
            synthesis_summary = f"fallback_best_composite reason={exc}"
            replay_mod._progress(
                "scorefit final history synthesis done "
                f"seed={int(base_config.sample_seed or 0)} scope={round_id} fallback=True"
            )

        selected_checkpoint = {
            "selection_mode": "llm_history_synthesis",
            "summary": synthesis_summary,
            "evaluated_lesson_count": int(
                sum(
                    1
                    for row in lesson_history
                    if bool(((row.get("eval_summary") or {}).get("heldout_metrics_available", False)))
                )
            ),
            "latest_lesson_index": int(max([int(row.get("lesson_index", 0) or 0) for row in lesson_history] or [0])),
            "fallback_used": bool(fallback_used),
        }
        synthesized_scope["selected_checkpoint"] = selected_checkpoint
        synthesized_scopes.append(synthesized_scope)
        selection_rows.append(
            {
                "round_id": round_id,
                **selected_checkpoint,
            }
        )

    synthesized_state["teacher_scopes"] = synthesized_scopes
    synthesized_state["final_lesson_selection_mode"] = "llm_history_synthesis"
    synthesized_state["final_lesson_composite_lambda"] = float(composite_lambda)
    synthesized_state["selected_checkpoint_rows"] = selection_rows
    selected_path = output_dir / "selected_final_lesson_llm_history_synthesis.json"
    _write_json(selected_path, synthesized_state)
    (output_dir / "selected_final_lesson_llm_history_synthesis.md").write_text(
        _scorefit_history_synthesis_md(synthesized_state),
        encoding="utf-8",
    )
    return synthesized_state, selected_path


def _select_final_lesson_state(
    *,
    lesson0_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    output_dir: Path,
    selection_mode: str,
    composite_lambda: float,
    composite_method: str,
    base_config: ApprenticeReplayConfig,
) -> Tuple[Dict[str, Any], Path]:
    normalized_mode = str(selection_mode).strip().lower()
    if normalized_mode == "last":
        path = output_dir / "warmup_scoped_lessons.json"
        return dict(final_state), path
    if normalized_mode == "llm_history_synthesis":
        fallback_state, _ = _select_final_lesson_state(
            lesson0_state=lesson0_state,
            final_state=final_state,
            output_dir=output_dir,
            selection_mode="best_composite",
            composite_lambda=composite_lambda,
            composite_method=composite_method,
            base_config=base_config,
        )
        return _scorefit_synthesize_final_lesson_state(
            lesson0_state=lesson0_state,
            final_state=final_state,
            fallback_state=fallback_state,
            output_dir=output_dir,
            composite_lambda=composite_lambda,
            base_config=base_config,
        )

    selected_state = dict(final_state)
    selected_scopes: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    teacher_scopes = list(final_state.get("teacher_scopes") or [])
    for scope in teacher_scopes:
        round_id = str(scope.get("round_id", "")).strip()
        lesson0_scope = replay_mod._find_teacher_scope_entry(lesson0_state, round_id) or {}
        batch_history = [dict(row) for row in list(scope.get("batch_history") or [])]
        revise_history = list(scope.get("revise_history") or [])
        candidates: List[Dict[str, Any]] = []

        if batch_history and isinstance(lesson0_scope.get("scorefit_lesson_json"), Mapping):
            eval_metrics = dict(batch_history[0].get("batch_metrics") or {})
            candidates.append(
                {
                    "candidate_kind": "lesson0",
                    "produced_after_batch": 0,
                    "evaluated_on_batch": int(batch_history[0].get("batch_index", 1) or 1),
                    "lesson_json": dict(lesson0_scope.get("scorefit_lesson_json") or {}),
                    "scope_lesson_zone_lines": list(lesson0_scope.get("scope_lesson_zone_lines") or []),
                    "review_cards_for_prompt": [],
                    "revise_history": [],
                    "batch_history": [],
                    "batch_metrics": eval_metrics,
                }
            )

        for produced_idx, batch_payload in enumerate(batch_history[:-1], start=1):
            eval_payload = batch_history[produced_idx]
            eval_metrics = dict(eval_payload.get("batch_metrics") or {})
            lesson_json = dict(batch_payload.get("lesson_json") or {})
            candidates.append(
                {
                    "candidate_kind": "warmup_checkpoint",
                    "produced_after_batch": int(batch_payload.get("batch_index", produced_idx) or produced_idx),
                    "evaluated_on_batch": int(eval_payload.get("batch_index", produced_idx + 1) or (produced_idx + 1)),
                    "lesson_json": lesson_json,
                    "scope_lesson_zone_lines": replay_mod._scorefit_render_lesson_lines(
                        lesson_json,
                        limit=base_config.warmup_lesson_zone_max_lines,
                    ),
                    "review_cards_for_prompt": replay_mod._scorefit_render_history_cards(
                        revise_history[:produced_idx],
                        limit=base_config.warmup_review_memory_limit,
                    ),
                    "revise_history": revise_history[:produced_idx],
                    "batch_history": batch_history[:produced_idx],
                    "batch_metrics": eval_metrics,
                }
            )

        if not candidates:
            fallback_scope = dict(scope)
            fallback_scope["selected_checkpoint"] = {
                "selection_mode": "fallback_last_no_candidates",
                "selected_batch_index": "last",
                "evaluated_on_batch_index": "n/a",
                "composite_score": 0.0,
                "teacher_score_spearman": 0.0,
                "llm_uplift_vs_not_selected": 0.0,
            }
            selected_scopes.append(fallback_scope)
            selection_rows.append(
                {
                    "round_id": round_id,
                    "selected_batch_index": "last",
                    "evaluated_on_batch_index": "n/a",
                    "composite_score": 0.0,
                    "teacher_score_spearman": 0.0,
                    "llm_uplift_vs_not_selected": 0.0,
                }
            )
            continue

        candidates = _scorefit_annotate_checkpoint_candidates(
            candidates,
            composite_lambda=composite_lambda,
            composite_method=composite_method,
        )

        best = max(
            candidates,
            key=lambda row: (
                float(row.get("composite_score", 0.0)),
                float((row.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected", 0.0) or 0.0),
                float((row.get("batch_metrics") or {}).get("teacher_score_spearman", 0.0) or 0.0),
                -int(row.get("produced_after_batch", 0) or 0),
            ),
        )

        selected_scope = dict(scope)
        selected_scope["scorefit_lesson_json"] = dict(best.get("lesson_json") or {})
        selected_scope["scope_lesson_zone_lines"] = list(best.get("scope_lesson_zone_lines") or [])
        selected_scope["review_cards_for_prompt"] = list(best.get("review_cards_for_prompt") or [])
        selected_scope["revise_history"] = list(best.get("revise_history") or [])
        selected_scope["batch_history"] = list(best.get("batch_history") or [])
        selected_scope["selected_checkpoint"] = {
            "selection_mode": "best_composite",
            "selected_batch_index": int(best.get("produced_after_batch", 0) or 0),
            "evaluated_on_batch_index": int(best.get("evaluated_on_batch", 0) or 0),
            "candidate_kind": str(best.get("candidate_kind", "")),
            "composite_score": float(best.get("composite_score", 0.0) or 0.0),
            "raw_composite_score": float(best.get("raw_composite_score", 0.0) or 0.0),
            "composite_method": _scorefit_composite_method_slug(composite_method),
            "teacher_score_spearman": float((best.get("batch_metrics") or {}).get("teacher_score_spearman", 0.0) or 0.0),
            "llm_uplift_vs_not_selected": float((best.get("batch_metrics") or {}).get("llm_uplift_vs_not_selected", 0.0) or 0.0),
            "llm_top20_mean_return": float((best.get("batch_metrics") or {}).get("llm_top20_mean_return", 0.0) or 0.0),
            "teacher_q5_mean_return": float((best.get("batch_metrics") or {}).get("teacher_q5_mean_return", 0.0) or 0.0),
            "z_spearman": float(best.get("z_spearman", 0.0) or 0.0),
            "z_uplift": float(best.get("z_uplift", 0.0) or 0.0),
        }
        selected_scopes.append(selected_scope)
        selection_rows.append(
            {
                "round_id": round_id,
                **dict(selected_scope["selected_checkpoint"]),
            }
        )

    selected_state["teacher_scopes"] = selected_scopes
    selected_state["final_lesson_selection_mode"] = "best_composite"
    selected_state["final_lesson_composite_lambda"] = float(composite_lambda)
    selected_state["final_lesson_composite_method"] = _scorefit_composite_method_slug(composite_method)
    selected_state["selected_checkpoint_rows"] = selection_rows
    selected_path = output_dir / (
        f"selected_final_lesson_best_composite_{_scorefit_composite_method_slug(composite_method)}.json"
    )
    _write_json(selected_path, selected_state)
    (output_dir / f"selected_final_lesson_best_composite_{_scorefit_composite_method_slug(composite_method)}.md").write_text(
        _scorefit_selected_state_md(selected_state),
        encoding="utf-8",
    )
    return selected_state, selected_path


def _clip_score_0_100(value: float) -> float:
    return float(max(0.0, min(100.0, float(value))))


def _nolesson_signal_prompt(
    *,
    config: ApprenticeReplayConfig,
    scope_domain: Mapping[str, Any],
    signal_record: Mapping[str, Any],
) -> Tuple[str, str]:
    reasoning_instruction = replay_mod._compact_reasoning_instruction(config)
    system = (
        "You are scoring one candidate signal inside a fixed teacher scope. "
        "The objective is hidden-teacher ranking alignment, not hindsight on this one sample. "
        f"{reasoning_instruction} "
        "No report and no lesson rubric are available. Use only the teacher-scope context and the raw signal features. "
        "Return strict JSON only with keys: total_score, short_reason. "
        "total_score must be a number from 0 to 100. Do not output markdown."
    )
    user = json.dumps(
        {
            "teacher_scope": {
                "round_id": scope_domain.get("round_id"),
                "source_round_id": scope_domain.get("source_round_id"),
                "family": scope_domain.get("family"),
                "template": scope_domain.get("template"),
                "style_hint": scope_domain.get("style_hint"),
                "basic_filter": scope_domain.get("basic_filter"),
            },
            "decision_seed": int(config.sample_seed or 0),
            "tie_break_note": (
                "If several signals feel similarly plausible, use decision_seed only as a deterministic tie-breaker. "
                "Do not mention the seed in the answer."
            ),
            "signal": signal_record,
        },
        ensure_ascii=False,
    )
    return system, user


def _request_one_signal_nolesson(
    *,
    task: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    api_key: str,
) -> Dict[str, Any]:
    system, user = _nolesson_signal_prompt(
        config=config,
        scope_domain=task["scope_domain"],
        signal_record=task["signal_record"],
    )
    request_payload = {
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": min(config.api_max_tokens, config.warmup_signal_score_max_tokens),
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
                max_tokens=request_payload["max_tokens"],
                temperature=config.api_temperature,
                force_local_qwen_no_thinking=config.force_local_qwen_no_thinking,
                fail_fast_on_empty_content=True,
                max_retries=1,
                seed=int(config.sample_seed or 0) or None,
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
            "subscores": {},
            "short_reason": str(parsed.get("short_reason", "")).strip(),
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
            "subscores": {},
            "short_reason": "",
            "parsed_payload": {},
        }


def _request_signal_batch_nolesson(
    *,
    task_batch: Sequence[Mapping[str, Any]],
    workers: int,
    config: ApprenticeReplayConfig,
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
                _request_one_signal_nolesson,
                task=task,
                config=config,
                api_key=api_key,
            )
            future_map[future] = task
        for future in as_completed(future_map):
            result = future.result()
            result["parallel_batch_index"] = int(global_batch_index)
            results.append(result)
    results.sort(key=lambda item: (str(item.get("signal_date")), str(item.get("task_key"))))
    return results


def _alignment_signal_tasks(
    *,
    bundle: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    signal_pool_per_day: int,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    scope_frame = bundle["scope_frame"]
    feature_cols = list(bundle["feature_cols"])
    tasks: List[Dict[str, Any]] = []
    task_counter = 0
    for day_offset, day_row in enumerate(bundle["selected_days_df"].itertuples(index=False), start=1):
        decision_date = pd.Timestamp(day_row.signal_date)
        day_df = scope_frame[scope_frame["signal_date"] == decision_date].copy()
        sampled_day = replay_mod._scorefit_sample_day_signals(
            day_df,
            x=signal_pool_per_day,
            sample_seed=(config.sample_seed * 10000 + day_offset) if int(config.sample_seed) > 0 else 0,
        )
        for _, row in sampled_day.iterrows():
            task_counter += 1
            task_key = (
                f"{bundle['final_round_id']}::seed{config.sample_seed:04d}::"
                f"{decision_date.strftime('%Y%m%d')}::{str(row.get('symbol', '')).strip()}::{task_counter:04d}"
            )
            tasks.append(
                {
                    "task_key": task_key,
                    "scope_round_id": bundle["final_round_id"],
                    "scope_source_round_id": bundle["source_round_id"],
                    "scope_domain": bundle["scope_domain"],
                    "signal_date": decision_date.strftime("%Y-%m-%d"),
                    "symbol": str(row.get("symbol", "")).strip(),
                    "teacher_score": replay_mod._scorefit_safe_float(row.get("score"), 0.0),
                    "teacher_bucket": int(row.get("bucket", 0) or 0),
                    "future_return_5d": replay_mod._scorefit_safe_float(row.get("future_return_5d"), 0.0),
                    "signal_record": replay_mod._scorefit_signal_record_from_row(row, feature_cols),
                    "cache_path": str(run_dir / "api_calls" / f"{task_counter:05d}.json"),
                    "reuse_api_cache": True,
                    "rerun_count": 0,
                }
            )
    return tasks


def _run_alignment_once(
    *,
    bundle: Mapping[str, Any],
    config: ApprenticeReplayConfig,
    run_dir: Path,
    lesson_payload: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        cached_summary = _load_json(summary_path)
        if cached_summary.get("alignment_sampling_strategy") == bundle["alignment_sampling_strategy"]:
            replay_mod._progress(f"alignment cache hit path={summary_path}")
            return cached_summary

    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = _alignment_signal_tasks(
        bundle=bundle,
        config=config,
        signal_pool_per_day=int(config.warmup_signal_pool_per_day),
        run_dir=run_dir,
    )
    replay_mod._progress(
        "scorefit scope alignment run start "
        f"teacher={bundle['final_round_id']} seed={config.sample_seed} "
        f"mode={'lesson' if lesson_payload is not None else 'nolesson'} tasks={len(tasks)}"
    )
    legacy_text_mode = bool((lesson_payload or {}).get("_legacy_text_mode", False))
    pending = list(tasks)
    finalized_results: Dict[str, Dict[str, Any]] = {}
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
            if lesson_payload is None:
                results = _request_signal_batch_nolesson(
                    task_batch=task_batch,
                    workers=max(1, int(config.api_parallel_workers)),
                    config=config,
                    api_key=api_key,
                    global_batch_index=global_batch_counter,
                )
            else:
                if legacy_text_mode:
                    results = replay_mod._legacy_text_request_signal_batch(
                        task_batch=task_batch,
                        workers=max(1, int(config.api_parallel_workers)),
                        config=config,
                        lesson_payload=lesson_payload,
                        api_key=api_key,
                        global_batch_index=global_batch_counter,
                    )
                else:
                    results = replay_mod._scorefit_request_signal_batch(
                        task_batch=task_batch,
                        workers=max(1, int(config.api_parallel_workers)),
                        config=config,
                        lesson_payload=lesson_payload,
                        api_key=api_key,
                        global_batch_index=global_batch_counter,
                    )
            source_by_key = {str(task["task_key"]): task for task in task_batch}
            requeued = 0
            finalized = 0
            for result in results:
                if bool(result.get("parse_fallback", False)) and pass_index + 1 < total_passes:
                    rerun_task = dict(source_by_key[str(result["task_key"])])
                    rerun_task["reuse_api_cache"] = False
                    rerun_task["rerun_count"] = int(rerun_task.get("rerun_count", 0)) + 1
                    next_pending.append(rerun_task)
                    requeued += 1
                else:
                    finalized_results[str(result["task_key"])] = result
                    finalized += 1
            replay_mod._progress(
                "scorefit scope alignment batch done "
                f"teacher={bundle['final_round_id']} seed={config.sample_seed} "
                f"mode={'lesson' if lesson_payload is not None else 'nolesson'} "
                f"pass={pass_index + 1}/{total_passes} wave={global_batch_counter} size={len(task_batch)} "
                f"requeued={requeued} finalized={finalized}"
            )
        pending = next_pending
    if pending:
        raise RuntimeError(
            f"alignment unresolved failed samples remain teacher={bundle['final_round_id']} seed={config.sample_seed} pending={len(pending)}"
        )

    ordered_results = [finalized_results[str(task["task_key"])] for task in tasks]
    metrics = replay_mod._scorefit_batch_metrics(
        results=ordered_results,
        lesson_payload=lesson_payload or {"items": {}},
        lock_days=config.lock_days,
    )
    summary = {
        "teacher_scope_round_id": bundle["final_round_id"],
        "teacher_scope_source_round_id": bundle["source_round_id"],
        "teacher_family": bundle["scope_domain"].get("family"),
        "teacher_template": bundle["scope_domain"].get("template"),
        "alignment_sampling_strategy": bundle["alignment_sampling_strategy"],
        "alignment_day_count": int(len(bundle["selected_days_df"])),
        "successful_signal_count": int(metrics.get("successful_signal_count", 0) or 0),
        "teacher_score_spearman": float(metrics.get("teacher_score_spearman", 0.0)),
        "llm_selected_mean_return": float(metrics.get("llm_top20_mean_return", 0.0)),
        "llm_not_selected_mean_return": float(metrics.get("llm_not_selected_mean_return", 0.0)),
        "llm_uplift_vs_not_selected": float(metrics.get("llm_uplift_vs_not_selected", 0.0)),
        "teacher_selected_mean_return": float(metrics.get("teacher_q5_mean_return", 0.0)),
        "teacher_not_selected_mean_return": float(metrics.get("teacher_not_selected_mean_return", 0.0)),
        "teacher_uplift_vs_not_selected": float(metrics.get("teacher_q5_uplift_vs_not_selected", 0.0)),
        "gap_to_teacher_uplift": float(metrics.get("gap_q5_uplift", 0.0)),
        "batch_nav_final": float(metrics.get("batch_nav_final", 1.0)),
        "llm_top20_count": int(metrics.get("llm_top20_count", 0) or 0),
        "run_dir": str(run_dir.resolve()),
    }
    _write_json(summary_path, summary)
    selected_day_rows = []
    for row in bundle["selected_days_df"].to_dict(orient="records"):
        clean = dict(row)
        clean["signal_date"] = pd.Timestamp(clean["signal_date"]).strftime("%Y-%m-%d")
        selected_day_rows.append(clean)
    _write_json(run_dir / "selected_days.json", {"rows": selected_day_rows})
    _write_json(run_dir / "batch_metrics.json", metrics)
    _write_json(run_dir / "signal_scores.json", {"results": ordered_results})
    return summary


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        "successful_signal_count",
        "teacher_score_spearman",
        "llm_selected_mean_return",
        "llm_not_selected_mean_return",
        "llm_uplift_vs_not_selected",
        "teacher_selected_mean_return",
        "teacher_not_selected_mean_return",
        "teacher_uplift_vs_not_selected",
        "gap_to_teacher_uplift",
        "batch_nav_final",
        "llm_top20_count",
    ]
    out: Dict[str, Any] = {"teacher_count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = float(sum(values) / len(values)) if values else 0.0
    return out


def _load_reused_mode_rows(
    *,
    baseline_suite_root: Path,
    mode_key: str,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    summary_path = baseline_suite_root / "suite_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing baseline suite summary: {summary_path}")
    payload = _load_json(summary_path)
    mode_payload = dict(payload.get(mode_key) or {})
    rows = list(mode_payload.get("per_seed_rows") or [])
    row_by_seed = {
        int(row.get("sample_seed", 0) or 0): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    missing = [seed for seed in seeds if int(seed) not in row_by_seed]
    if missing:
        raise ValueError(
            f"baseline suite {baseline_suite_root} missing {mode_key} rows for seeds={missing}"
        )
    return [row_by_seed[int(seed)] for seed in seeds]


def _mode_md(title: str, seed_rows: Sequence[Mapping[str, Any]], mode_aggregate: Mapping[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        "## Mean Across Seeds",
        "",
        f"- teacher_score_spearman: `{mode_aggregate['teacher_score_spearman']:.4f}`",
        f"- llm_selected_mean_return: `{mode_aggregate['llm_selected_mean_return']:.4%}`",
        f"- llm_not_selected_mean_return: `{mode_aggregate['llm_not_selected_mean_return']:.4%}`",
        f"- llm_uplift_vs_not_selected: `{mode_aggregate['llm_uplift_vs_not_selected']:.4%}`",
        f"- teacher_selected_mean_return: `{mode_aggregate['teacher_selected_mean_return']:.4%}`",
        f"- teacher_not_selected_mean_return: `{mode_aggregate['teacher_not_selected_mean_return']:.4%}`",
        f"- teacher_uplift_vs_not_selected: `{mode_aggregate['teacher_uplift_vs_not_selected']:.4%}`",
        f"- gap_to_teacher_uplift: `{mode_aggregate['gap_to_teacher_uplift']:.4%}`",
        f"- batch_nav_final: `{mode_aggregate['batch_nav_final']:.4f}`",
        "",
        "## Per Seed",
        "",
        "| seed | spearman | llm_selected | llm_not_selected | llm_uplift | teacher_selected | teacher_not_selected | teacher_uplift | gap_to_teacher_uplift | nav |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_seed"]),
                    f"{float(row['teacher_score_spearman']):.4f}",
                    f"{float(row['llm_selected_mean_return']):.4%}",
                    f"{float(row['llm_not_selected_mean_return']):.4%}",
                    f"{float(row['llm_uplift_vs_not_selected']):.4%}",
                    f"{float(row['teacher_selected_mean_return']):.4%}",
                    f"{float(row['teacher_not_selected_mean_return']):.4%}",
                    f"{float(row['teacher_uplift_vs_not_selected']):.4%}",
                    f"{float(row['gap_to_teacher_uplift']):.4%}",
                    f"{float(row['batch_nav_final']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _summarize_warmup_run(
    *,
    sample_seed: int,
    warmup_run_dir: Path,
    lesson0_dir: Path,
    final_state: Mapping[str, Any],
) -> Dict[str, Any]:
    csv_path = warmup_run_dir / "warmup_scorefit_batch_metrics.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    out: Dict[str, Any] = {
        "sample_seed": int(sample_seed),
        "warmup_run_dir": str(warmup_run_dir.resolve()),
        "lesson0_state_json": str((lesson0_dir / "warmup_scoped_lessons.json").resolve()),
        "final_state_json": str((warmup_run_dir / "warmup_scoped_lessons.json").resolve()),
        "scope_count": int(len(list(final_state.get("teacher_scopes") or []))),
        "batch_count": int(len(df)),
    }
    if df.empty:
        out.update(
            {
                "teacher_score_spearman": 0.0,
                "llm_selected_mean_return": 0.0,
                "teacher_selected_mean_return": 0.0,
                "batch_nav_final": 1.0,
                "successful_signal_count": 0.0,
            }
        )
        return out

    out.update(
        {
            "teacher_score_spearman": float(df["teacher_score_spearman"].mean()),
            "llm_selected_mean_return": float(df["llm_top20_mean_return"].mean()),
            "teacher_selected_mean_return": float(df["teacher_q5_mean_return"].mean()),
            "batch_nav_final": float(df["batch_nav_final"].mean()),
            "successful_signal_count": float(df["successful_signal_count"].mean()),
        }
    )
    return out


def _warmup_only_md(seed_rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# Warmup Only",
        "",
        "## Mean Across Seeds",
        "",
        f"- teacher_score_spearman: `{float(aggregate['teacher_score_spearman']):.4f}`",
        f"- llm_selected_mean_return: `{float(aggregate['llm_selected_mean_return']):.4%}`",
        f"- teacher_selected_mean_return: `{float(aggregate['teacher_selected_mean_return']):.4%}`",
        f"- batch_nav_final: `{float(aggregate['batch_nav_final']):.4f}`",
        f"- successful_signal_count: `{float(aggregate['successful_signal_count']):.2f}`",
        "",
        "## Per Seed",
        "",
        "| seed | spearman | llm_selected | teacher_selected | batch_nav | successful_signals |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_seed"]),
                    f"{float(row['teacher_score_spearman']):.4f}",
                    f"{float(row['llm_selected_mean_return']):.4%}",
                    f"{float(row['teacher_selected_mean_return']):.4%}",
                    f"{float(row['batch_nav_final']):.4f}",
                    f"{float(row['successful_signal_count']):.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    selection = _load_selection(Path(args.selection_json).resolve())
    suite_root = (
        Path(args.suite_root).expanduser().resolve()
        if str(args.suite_root).strip()
        else env_path("APPRENTICE_REPORT_ROOT", project_root() / "reports")
        / "scopefit_scope_alignment_suite"
        / args.run_tag_base
    )
    bundle_cache_dir = Path(args.bundle_cache_dir).expanduser().resolve() if str(args.bundle_cache_dir).strip() else None
    suite_root.mkdir(parents=True, exist_ok=True)

    replay_mod._progress(f"suite start root={suite_root} seeds={seeds} model={args.api_model}")
    master_df = replay_mod._load_master_dataset()
    replay_mod._progress(f"suite master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")

    baseline_suite_root = (
        Path(args.reuse_baseline_suite_root).expanduser().resolve()
        if str(args.reuse_baseline_suite_root).strip()
        else None
    )

    prep_cfg = _base_config_from_args(args, selection, sample_seed=0)
    scope_bundles: List[Dict[str, Any]] = []
    for spec in selection["scope_specs"]:
        bundle = _teacher_explainability_bundle(
            base_config=prep_cfg,
            master_df=master_df,
            final_round_id=spec["final_round_id"],
            source_round_id=spec["source_round_id"],
            alignment_day_count=args.alignment_day_count,
            signal_pool_per_day=args.signal_pool_per_day,
            alignment_sampling_strategy=args.alignment_sampling_strategy,
            cache_dir=bundle_cache_dir,
        )
        scope_bundles.append(bundle)

    no_lesson_seed_rows: List[Dict[str, Any]] = []
    lesson0_seed_rows: List[Dict[str, Any]] = []
    final_seed_rows: List[Dict[str, Any]] = []
    warmup_only_seed_rows: List[Dict[str, Any]] = []

    if args.warmup_only:
        for sample_seed in seeds:
            base_cfg = _base_config_from_args(args, selection, sample_seed=sample_seed)
            lesson0_dir = suite_root / "lesson0_states" / f"seed_{sample_seed:04d}"
            lesson0_state = _build_lesson0_scoped_state(
                base_config=base_cfg,
                scope_bundles=scope_bundles,
                output_dir=lesson0_dir,
            )
            warmup_cfg = replace(base_cfg, run_tag=f"{args.run_tag_base}_warmup_seed{sample_seed:04d}")
            final_state, warmup_run_dir = _run_warmup_from_initial_state(
                base_config=warmup_cfg,
                master_df=master_df,
                initial_scoped_state=lesson0_state,
            )
            warmup_only_seed_rows.append(
                _summarize_warmup_run(
                    sample_seed=sample_seed,
                    warmup_run_dir=warmup_run_dir,
                    lesson0_dir=lesson0_dir,
                    final_state=final_state,
                )
            )

        warmup_mean = _aggregate_rows(warmup_only_seed_rows)
        suite_summary = {
            "model": args.api_model,
            "run_tag_base": args.run_tag_base,
            "seeds": seeds,
            "warmup_sample_count": int(args.warmup_sample_count),
            "warmup_batch_size": int(args.warmup_batch_size),
            "warmup_signal_pool_per_day": int(args.warmup_signal_pool_per_day),
            "scorefit_variant": args.scorefit_variant,
            "alignment_sampling_strategy": args.alignment_sampling_strategy,
            "warmup_only": True,
            "warmup_final_lesson": {
                "per_seed_rows": warmup_only_seed_rows,
                "mean": warmup_mean,
            },
        }
        _write_json(suite_root / "suite_summary.json", suite_summary)
        (suite_root / "warmup_only_summary.md").write_text(
            _warmup_only_md(warmup_only_seed_rows, warmup_mean),
            encoding="utf-8",
        )
        replay_mod._progress(f"suite complete path={suite_root / 'suite_summary.json'}")
        return

    if args.skip_no_lesson:
        if baseline_suite_root is None:
            raise ValueError("--skip-no-lesson requires --reuse-baseline-suite-root")
        no_lesson_seed_rows = _load_reused_mode_rows(
            baseline_suite_root=baseline_suite_root,
            mode_key="no_lesson",
            seeds=seeds,
        )
        replay_mod._progress(
            f"reused no_lesson baseline root={baseline_suite_root}"
        )

    if args.skip_lesson0_alignment:
        if baseline_suite_root is None:
            raise ValueError("--skip-lesson0-alignment requires --reuse-baseline-suite-root")
        lesson0_seed_rows = _load_reused_mode_rows(
            baseline_suite_root=baseline_suite_root,
            mode_key="lesson0",
            seeds=seeds,
        )
        replay_mod._progress(
            f"reused lesson0 baseline root={baseline_suite_root}"
        )

    for sample_seed in seeds:
        base_cfg = _base_config_from_args(args, selection, sample_seed=sample_seed)
        reuse_lesson0_root = (
            Path(args.reuse_lesson0_suite_root).expanduser().resolve()
            if str(args.reuse_lesson0_suite_root).strip()
            else None
        )
        reuse_lesson0_dir = (
            reuse_lesson0_root / "lesson0_states" / f"seed_{sample_seed:04d}"
            if reuse_lesson0_root is not None
            else None
        )
        warmup_run_tag_base = str(args.reuse_warmup_run_tag_base).strip() or args.run_tag_base

        if not args.skip_no_lesson:
            no_lesson_scope_rows: List[Dict[str, Any]] = []
            for bundle in scope_bundles:
                run_dir = (
                    suite_root
                    / "no_lesson"
                    / args.alignment_sampling_strategy
                    / f"seed_{sample_seed:04d}"
                    / bundle["final_round_id"]
                )
                row = _run_alignment_once(
                    bundle=bundle,
                    config=replace(base_cfg, run_tag=f"{args.run_tag_base}_nolesson_seed{sample_seed:04d}_{bundle['final_round_id']}"),
                    run_dir=run_dir,
                    lesson_payload=None,
                )
                no_lesson_scope_rows.append(row)
            seed_row = _aggregate_rows(no_lesson_scope_rows)
            seed_row["sample_seed"] = sample_seed
            no_lesson_seed_rows.append(seed_row)

        lesson0_dir = suite_root / "lesson0_states" / f"seed_{sample_seed:04d}"
        lesson0_state = _build_lesson0_scoped_state(
            base_config=base_cfg,
            scope_bundles=scope_bundles,
            output_dir=lesson0_dir,
            reuse_source_dir=reuse_lesson0_dir,
        )
        if not args.skip_lesson0_alignment:
            lesson0_scope_rows: List[Dict[str, Any]] = []
            for bundle in scope_bundles:
                scope_entry = replay_mod._find_teacher_scope_entry(lesson0_state, bundle["final_round_id"]) or {}
                lesson_payload = dict(scope_entry.get("scorefit_lesson_json") or {})
                run_dir = (
                    suite_root
                    / "lesson0_alignment"
                    / args.alignment_sampling_strategy
                    / f"seed_{sample_seed:04d}"
                    / bundle["final_round_id"]
                )
                row = _run_alignment_once(
                    bundle=bundle,
                    config=replace(base_cfg, run_tag=f"{args.run_tag_base}_lesson0_seed{sample_seed:04d}_{bundle['final_round_id']}"),
                    run_dir=run_dir,
                    lesson_payload=lesson_payload,
                )
                lesson0_scope_rows.append(row)
            seed_row = _aggregate_rows(lesson0_scope_rows)
            seed_row["sample_seed"] = sample_seed
            seed_row["lesson0_state_json"] = str((lesson0_dir / "warmup_scoped_lessons.json").resolve())
            lesson0_seed_rows.append(seed_row)

        warmup_cfg = replace(base_cfg, run_tag=f"{warmup_run_tag_base}_warmup_seed{sample_seed:04d}")
        final_state, warmup_run_dir = _run_warmup_from_initial_state(
            base_config=warmup_cfg,
            master_df=master_df,
            initial_scoped_state=lesson0_state,
            warmup_run_tag_base=f"{warmup_run_tag_base}_warmup_seed{sample_seed:04d}",
        )
        selected_final_state, selected_final_state_path = _select_final_lesson_state(
            lesson0_state=lesson0_state,
            final_state=final_state,
            output_dir=warmup_run_dir,
            selection_mode=args.final_lesson_selection,
            composite_lambda=float(args.final_lesson_composite_lambda),
            composite_method=str(args.final_lesson_composite_method),
            base_config=base_cfg,
        )
        final_scope_rows: List[Dict[str, Any]] = []
        for bundle in scope_bundles:
            scope_entry = replay_mod._find_teacher_scope_entry(selected_final_state, bundle["final_round_id"]) or {}
            lesson_payload = dict(scope_entry.get("scorefit_lesson_json") or {})
            run_dir = (
                suite_root
                / "final_alignment"
                / args.alignment_sampling_strategy
                / f"seed_{sample_seed:04d}"
                / bundle["final_round_id"]
            )
            row = _run_alignment_once(
                bundle=bundle,
                config=replace(base_cfg, run_tag=f"{args.run_tag_base}_final_seed{sample_seed:04d}_{bundle['final_round_id']}"),
                run_dir=run_dir,
                lesson_payload=lesson_payload,
            )
            final_scope_rows.append(row)
        seed_row = _aggregate_rows(final_scope_rows)
        seed_row["sample_seed"] = sample_seed
        seed_row["final_state_json"] = str(selected_final_state_path.resolve())
        seed_row["final_lesson_selection_mode"] = str(args.final_lesson_selection)
        final_seed_rows.append(seed_row)

    suite_summary = {
        "model": args.api_model,
        "run_tag_base": args.run_tag_base,
        "seeds": seeds,
        "alignment_day_count": int(args.alignment_day_count),
        "signal_pool_per_day": int(args.signal_pool_per_day),
        "warmup_sample_count": int(args.warmup_sample_count),
        "warmup_batch_size": int(args.warmup_batch_size),
        "warmup_signal_pool_per_day": int(args.warmup_signal_pool_per_day),
        "scorefit_variant": args.scorefit_variant,
        "alignment_sampling_strategy": args.alignment_sampling_strategy,
        "reuse_warmup_run_tag_base": str(args.reuse_warmup_run_tag_base).strip(),
        "reuse_lesson0_suite_root": str(args.reuse_lesson0_suite_root).strip(),
        "reuse_baseline_suite_root": str(args.reuse_baseline_suite_root).strip(),
        "skip_no_lesson": bool(args.skip_no_lesson),
        "skip_lesson0_alignment": bool(args.skip_lesson0_alignment),
        "final_lesson_selection": str(args.final_lesson_selection),
        "final_lesson_composite_lambda": float(args.final_lesson_composite_lambda),
        "final_lesson_composite_method": str(args.final_lesson_composite_method),
        "no_lesson": {
            "per_seed_rows": no_lesson_seed_rows,
            "mean": _aggregate_rows(no_lesson_seed_rows),
        },
        "lesson0": {
            "per_seed_rows": lesson0_seed_rows,
            "mean": _aggregate_rows(lesson0_seed_rows),
        },
        "final_lesson": {
            "per_seed_rows": final_seed_rows,
            "mean": _aggregate_rows(final_seed_rows),
        },
    }
    _write_json(suite_root / "suite_summary.json", suite_summary)
    (suite_root / "no_lesson_summary.md").write_text(
        _mode_md("No Lesson", no_lesson_seed_rows, suite_summary["no_lesson"]["mean"]),
        encoding="utf-8",
    )
    (suite_root / "lesson0_summary.md").write_text(
        _mode_md("Fixed Shared Lesson0", lesson0_seed_rows, suite_summary["lesson0"]["mean"]),
        encoding="utf-8",
    )
    (suite_root / "final_lesson_summary.md").write_text(
        _mode_md("Warmup Final Lesson", final_seed_rows, suite_summary["final_lesson"]["mean"]),
        encoding="utf-8",
    )
    replay_mod._progress(f"suite complete path={suite_root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
