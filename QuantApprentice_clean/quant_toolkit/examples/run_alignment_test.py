#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run fixed-date pre/post lesson alignment tests for multi-teacher apprentice prompts."""

from __future__ import annotations

import argparse
import json
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-date alignment pre/post tests for QuantApprentice")
    parser.add_argument("--selection-json", required=True, help="Path to teacher selection.json with frozen_round_ids")
    parser.add_argument("--lesson-artifact-json", default="", help="Path to warmup_scoped_lessons.json used for post-lesson run")
    parser.add_argument("--start-date", default="2020-01-02")
    parser.add_argument("--end-date", default="2022-12-30")
    parser.add_argument("--alignment-sample-count", type=int, default=200)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--teacher-daily-pick-count", type=int, default=4)
    parser.add_argument("--llm-max-daily-picks", type=int, default=4)
    parser.add_argument("--prompt-feature-count", type=int, default=8)
    parser.add_argument("--lesson-feature-count", type=int, default=20)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-max-tokens", type=int, default=768)
    parser.add_argument("--summary-variant", choices=["simple_v1", "enriched_v2"], default="enriched_v2")
    parser.add_argument("--candidate-source", choices=["teacher_ranked", "baseline_signal"], default="teacher_ranked")
    parser.add_argument("--include-teacher-signal", action="store_true")
    parser.add_argument("--ignore-holdings-context", action="store_true")
    parser.add_argument("--api-parallel-workers", type=int, default=1)
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--no-reuse-api-cache", action="store_true")
    parser.add_argument("--bundle-cache-path", default="", help="Optional shared joblib cache for teacher frames and sampled dates")
    return parser


def _load_selection(selection_json: Path) -> Dict[str, Any]:
    payload = _load_json(selection_json)
    frozen_teachers = list(payload.get("frozen_round_ids") or [])
    negative = list(payload.get("negative_round_ids") or [])
    if not frozen_teachers:
        raise ValueError(f"selection_json has no frozen_round_ids: {selection_json}")
    teachers = [replay_mod._source_round_id_for_round(round_id) for round_id in frozen_teachers]
    return {
        "teachers": teachers,
        "frozen_teachers": frozen_teachers,
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


def _maybe_enrich_teacher_meta(
    *,
    config: ApprenticeReplayConfig,
    master_df: pd.DataFrame,
    metas: List[Dict[str, Any]],
) -> None:
    if config.summary_variant != "enriched_v2":
        return
    for meta in metas:
        meta["preference_bands"] = replay_mod._derive_preference_bands(
            round_id=meta["round_id"],
            master_df=master_df,
            feature_cols=meta["top_prompt_features"],
            start_date=config.start_date,
            end_date=config.end_date,
        )


def _sample_alignment_frames(
    *,
    config: ApprenticeReplayConfig,
    bundle_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if bundle_cache_path and bundle_cache_path.exists():
        replay_mod._progress(f"alignment bundle cache hit path={bundle_cache_path}")
        return joblib.load(bundle_cache_path)
    master_df = replay_mod._load_master_dataset()
    replay_mod._progress(f"alignment master dataset loaded rows={len(master_df)} cols={len(master_df.columns)}")
    merged, metas, prompt_features = replay_mod._build_multi_teacher_frame(config, master_df)
    replay_mod._progress(
        "alignment teacher frame built "
        f"rows={len(merged)} teachers={len(metas)} prompt_features={len(prompt_features)}"
    )
    _maybe_enrich_teacher_meta(config=config, master_df=master_df, metas=metas)
    negative_metas = [
        replay_mod._negative_teacher_meta(round_id, max(4, config.prompt_feature_count // 2))
        for round_id in config.negative_teacher_round_ids
    ]
    candidate_pool_df, teacher_target_df = replay_mod._multi_teacher_target(merged, config)
    teacher_full_df = teacher_target_df.copy()
    sampled_dates = replay_mod._sample_uniform_dates(candidate_pool_df["signal_date"].tolist(), config.warmup_sample_count)
    sampled_dates = [pd.Timestamp(dt) for dt in sampled_dates]
    sampled_date_set = set(sampled_dates)
    candidate_pool_df = candidate_pool_df[candidate_pool_df["signal_date"].isin(sampled_date_set)].copy()
    teacher_target_df = teacher_target_df[teacher_target_df["signal_date"].isin(sampled_date_set)].copy()
    teacher_full_df = teacher_full_df[teacher_full_df["signal_date"].isin(sampled_date_set)].copy()
    candidate_pool_df = candidate_pool_df.sort_values(["signal_date", "ensemble_score"], ascending=[True, False]).reset_index(drop=True)
    teacher_target_df = teacher_target_df.sort_values(["signal_date", "ensemble_score"], ascending=[True, False]).reset_index(drop=True)
    teacher_full_df = teacher_full_df.sort_values(["signal_date", "ensemble_score"], ascending=[True, False]).reset_index(drop=True)
    replay_mod._progress(
        "alignment sampled pool ready "
        f"sampled_days={candidate_pool_df['signal_date'].nunique()} pool_rows={len(candidate_pool_df)} "
        f"target_rows={len(teacher_target_df)}"
    )
    bundle = {
        "metas": metas,
        "prompt_features": prompt_features,
        "negative_metas": negative_metas,
        "candidate_pool_df": candidate_pool_df,
        "teacher_target_df": teacher_target_df,
        "teacher_full_df": teacher_full_df,
        "sampled_dates": sampled_dates,
    }
    if bundle_cache_path:
        bundle_cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, bundle_cache_path)
        replay_mod._progress(f"alignment bundle cache saved path={bundle_cache_path}")
    return bundle


def _summary_to_dict(summary: replay_mod.ReplaySummary) -> Dict[str, Any]:
    return dict(summary.__dict__)


def _run_one(
    *,
    base_config: ApprenticeReplayConfig,
    run_tag: str,
    frames: Dict[str, Any],
    scoped_warmup_state: Optional[Dict[str, Any]],
) -> replay_mod.ReplaySummary:
    config = replace(base_config, warmup_sample_count=0, run_tag=run_tag)
    warmup_lessons = list((scoped_warmup_state or {}).get("global_lesson_zone_lines", []))
    warmup_review_cards = list((scoped_warmup_state or {}).get("review_cards_for_prompt", []))
    return replay_mod._run_replay(
        config=config,
        candidate_pool_df=frames["candidate_pool_df"],
        teacher_target_df=frames["teacher_target_df"],
        teacher_full_df=frames["teacher_full_df"],
        prompt_builder=replay_mod._daily_prompt_multi,
        prompt_builder_kwargs={
            "metas": frames["metas"],
            "negative_metas": frames["negative_metas"],
            "prompt_features": frames["prompt_features"],
            "warmup_lessons": warmup_lessons,
            "warmup_review_cards": warmup_review_cards,
            "scoped_warmup_state": scoped_warmup_state,
        },
    )


def _comparison_md(
    *,
    title: str,
    sample_dates: List[pd.Timestamp],
    no_lesson: Dict[str, Any],
    with_lesson: Optional[Dict[str, Any]],
    lesson_artifact_json: str,
) -> str:
    def _line(metric: str, fmt: str = ".4f") -> str:
        pre = no_lesson.get(metric)
        post = with_lesson.get(metric) if with_lesson else None
        if post is None or pre is None:
            return f"- {metric}: pre=`{pre}` post=`{post}`"
        delta = post - pre
        return f"- {metric}: pre=`{format(pre, fmt)}` post=`{format(post, fmt)}` delta=`{format(delta, fmt)}`"

    lines = [
        f"# {title}",
        "",
        "## Setup",
        "",
        f"- sampled_decision_days: `{len(sample_dates)}`",
        f"- first_sampled_date: `{sample_dates[0].strftime('%Y-%m-%d') if sample_dates else 'na'}`",
        f"- last_sampled_date: `{sample_dates[-1].strftime('%Y-%m-%d') if sample_dates else 'na'}`",
        f"- lesson_artifact_json: `{lesson_artifact_json or 'none'}`",
        "",
        "## Alignment Metrics",
        "",
        _line("mean_daily_jaccard"),
        _line("mean_daily_precision"),
        _line("mean_daily_recall"),
        _line("exact_match_rate"),
        "",
        "## Return-Side Diagnostics",
        "",
        _line("llm_selected_mean_return", ".6f"),
        _line("teacher_selected_mean_return", ".6f"),
        _line("uplift_vs_teacher_selected", ".6f"),
        _line("uplift_vs_not_selected", ".6f"),
        "",
        "## Path Diagnostics",
        "",
        _line("llm_final_nav"),
        _line("teacher_target_final_nav"),
        _line("parse_fallback_days", ".0f"),
        _line("parse_failure_days", ".0f"),
        _line("abstain_days", ".0f"),
        "",
        "## Artifacts",
        "",
        f"- no_lesson_summary: [{no_lesson['run_id']}]({replay_mod.PROJECT_ROOT / replay_mod.REPORT_ROOT.relative_to(replay_mod.PROJECT_ROOT) / no_lesson['run_id'] / 'SUMMARY.md'})",
    ]
    if with_lesson:
        lines.append(
            f"- with_lesson_summary: [{with_lesson['run_id']}]({replay_mod.PROJECT_ROOT / replay_mod.REPORT_ROOT.relative_to(replay_mod.PROJECT_ROOT) / with_lesson['run_id'] / 'SUMMARY.md'})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    selection_json = Path(args.selection_json).expanduser().resolve()
    lesson_artifact_json = Path(args.lesson_artifact_json).expanduser().resolve() if args.lesson_artifact_json else None
    bundle_cache_path = Path(args.bundle_cache_path).expanduser().resolve() if args.bundle_cache_path else None
    selection = _load_selection(selection_json)
    scoped_warmup_state = _load_scoped_warmup_state(lesson_artifact_json) if lesson_artifact_json else None

    base_config = ApprenticeReplayConfig(
        mode="multi",
        teacher_round_ids=selection["teachers"],
        negative_teacher_round_ids=selection["negative_teachers"],
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
        include_teacher_signal=args.include_teacher_signal,
        candidate_source=args.candidate_source,
        reuse_api_cache=not args.no_reuse_api_cache,
        ignore_holdings_context=args.ignore_holdings_context,
        api_parallel_workers=args.api_parallel_workers,
        summary_variant=args.summary_variant,
        warmup_sample_count=args.alignment_sample_count,
        run_tag=args.run_tag_base,
    )

    frames = _sample_alignment_frames(config=base_config, bundle_cache_path=bundle_cache_path)
    suite_dir = replay_mod.REPORT_ROOT / f"alignment_suite_{args.run_tag_base}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"signal_date": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in frames["sampled_dates"]]}
    ).to_csv(suite_dir / "alignment_sample_dates.csv", index=False)

    no_lesson_summary = _run_one(
        base_config=base_config,
        run_tag=f"{args.run_tag_base}_nolesson",
        frames=frames,
        scoped_warmup_state=None,
    )
    with_lesson_summary = None
    if scoped_warmup_state is not None:
        with_lesson_summary = _run_one(
            base_config=base_config,
            run_tag=f"{args.run_tag_base}_withlesson",
            frames=frames,
            scoped_warmup_state=scoped_warmup_state,
        )

    no_lesson_payload = _summary_to_dict(no_lesson_summary)
    with_lesson_payload = _summary_to_dict(with_lesson_summary) if with_lesson_summary is not None else None

    comparison = {
        "suite_dir": str(suite_dir),
        "selection_json": str(selection_json),
        "lesson_artifact_json": str(lesson_artifact_json) if lesson_artifact_json else "",
        "bundle_cache_path": str(bundle_cache_path) if bundle_cache_path else "",
        "teacher_round_ids": list(base_config.teacher_round_ids),
        "frozen_teacher_round_ids": list(selection.get("frozen_teachers") or []),
        "negative_teacher_round_ids": list(base_config.negative_teacher_round_ids),
        "api_model": base_config.api_model,
        "include_teacher_signal": bool(base_config.include_teacher_signal),
        "sampled_decision_days": len(frames["sampled_dates"]),
        "sampled_dates_csv": str(suite_dir / "alignment_sample_dates.csv"),
        "no_lesson": no_lesson_payload,
        "with_lesson": with_lesson_payload,
    }
    if with_lesson_payload is not None:
        comparison["delta"] = {
            metric: with_lesson_payload.get(metric, 0.0) - no_lesson_payload.get(metric, 0.0)
            for metric in [
                "mean_daily_jaccard",
                "mean_daily_precision",
                "mean_daily_recall",
                "exact_match_rate",
                "llm_selected_mean_return",
                "uplift_vs_teacher_selected",
                "uplift_vs_not_selected",
                "llm_final_nav",
                "parse_fallback_days",
                "parse_failure_days",
                "abstain_days",
            ]
        }

    _write_json(suite_dir / "alignment_comparison.json", comparison)
    (suite_dir / "ALIGNMENT_COMPARISON.md").write_text(
        _comparison_md(
            title=f"Alignment Test - {args.run_tag_base}",
            sample_dates=frames["sampled_dates"],
            no_lesson=no_lesson_payload,
            with_lesson=with_lesson_payload,
            lesson_artifact_json=str(lesson_artifact_json) if lesson_artifact_json else "",
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
