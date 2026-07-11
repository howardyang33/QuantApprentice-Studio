#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze selected teacher-loop rounds at train_end_year and evaluate post-cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from quant_toolkit.backtest.teacher_loop_nav_backtest import run_round_nav_backtest
from quant_toolkit.teacher_loop.loop import (
    ARTIFACT_ROOT,
    MEMORY_DIR,
    REPORT_ROOT,
    TEST_YEARS,
    TeacherSpec,
    _build_daily_alpha_frame,
    _extract_importance,
    _fit_model,
    _markdown_table,
    _negative_alpha_pvalue,
    _relative,
    _score_diagnostics,
    _score_model,
    build_dataset_for_spec,
)
from quant_toolkit.teacher_loop.factor_analysis import run_teacher_factor_analysis
from quant_toolkit.pilot2.walkforward_utils import TRAIN_BUCKET_QUANTILES, assign_prediction_buckets, compute_train_thresholds


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_zoo() -> Dict[str, Any]:
    return _load_json(MEMORY_DIR / "indexes" / "teacher_zoo_index.json")


def _teacher_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
    metrics = row.get("metrics", {})
    partition = row.get("zoo_partition", "")
    accepted = 1 if row.get("accepted_as_teacher") else 0
    partition_rank = 2 if partition == "main" else 1 if partition == "try" else 0
    nav_cagr = float(metrics.get("nav_cagr") or float("-inf"))
    mean_alpha = float(metrics.get("mean_alpha") or float("-inf"))
    positive_rate = 0.0
    nav_pos = metrics.get("nav_positive_years")
    nav_total = metrics.get("nav_total_years")
    if nav_pos is not None and nav_total:
        positive_rate = float(nav_pos) / float(nav_total)
    return (accepted + partition_rank, nav_cagr, mean_alpha, positive_rate)


def shortlist_teachers(top_k: int, negative_k: int) -> Tuple[List[str], List[str], pd.DataFrame]:
    zoo = _load_zoo()
    teachers = zoo.get("teachers", [])
    teacher_df = pd.DataFrame(teachers)
    if teacher_df.empty:
        raise RuntimeError("teacher_zoo_index.json contains no teachers")

    def metrics_value(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
        metrics = row.get("metrics", {})
        value = metrics.get(key)
        return default if value is None else float(value)

    positive_rows = [
        row
        for row in teachers
        if str(row.get("zoo_partition", "")) in {"main", "try"} and bool(row.get("accepted_as_teacher", False))
    ]
    positive_rows.sort(key=_teacher_sort_key, reverse=True)
    selected_positive = [str(row["round_id"]) for row in positive_rows[:top_k] if row.get("round_id")]

    negative_rows = []
    for row in teachers:
        partition = str(row.get("zoo_partition", ""))
        accepted = bool(row.get("accepted_as_teacher", False))
        mean_alpha = metrics_value(row, "mean_alpha", default=0.0)
        nav_cagr = metrics_value(row, "nav_cagr", default=0.0)
        if partition == "rejected" or not accepted or mean_alpha <= 0:
            negative_rows.append((mean_alpha, nav_cagr, str(row["round_id"])))
    negative_rows.sort(key=lambda item: (item[0], item[1]))
    selected_negative = [round_id for _, _, round_id in negative_rows[:negative_k]]
    if len(selected_negative) < negative_k:
        positive_set = set(selected_positive)
        negative_set = set(selected_negative)
        fallback_rows = []
        for row in teachers:
            round_id = str(row.get("round_id") or "")
            if not round_id or round_id in positive_set or round_id in negative_set:
                continue
            mean_alpha = metrics_value(row, "mean_alpha", default=0.0)
            nav_cagr = metrics_value(row, "nav_cagr", default=0.0)
            positive_rate = 0.0
            metrics = row.get("metrics", {})
            nav_pos = metrics.get("nav_positive_years")
            nav_total = metrics.get("nav_total_years")
            if nav_pos is not None and nav_total:
                positive_rate = float(nav_pos) / float(nav_total)
            fallback_rows.append((nav_cagr, mean_alpha, positive_rate, round_id))
        fallback_rows.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, _, round_id in fallback_rows:
            if round_id in negative_set:
                continue
            selected_negative.append(round_id)
            negative_set.add(round_id)
            if len(selected_negative) >= negative_k:
                break

    summary_rows = []
    for row in positive_rows[:top_k]:
        metrics = row.get("metrics", {})
        summary_rows.append(
            {
                "kind": "positive",
                "round_id": row.get("round_id"),
                "title": row.get("title"),
                "partition": row.get("zoo_partition"),
                "mean_alpha": metrics.get("mean_alpha"),
                "nav_cagr": metrics.get("nav_cagr"),
                "nav_total_return": metrics.get("nav_total_return"),
            }
        )
    for _, _, round_id in negative_rows[:negative_k]:
        row = next(item for item in teachers if str(item.get("round_id")) == round_id)
        metrics = row.get("metrics", {})
        summary_rows.append(
            {
                "kind": "negative",
                "round_id": row.get("round_id"),
                "title": row.get("title"),
                "partition": row.get("zoo_partition"),
                "mean_alpha": metrics.get("mean_alpha"),
                "nav_cagr": metrics.get("nav_cagr"),
                "nav_total_return": metrics.get("nav_total_return"),
            }
        )
    return selected_positive, selected_negative, pd.DataFrame(summary_rows)


def _load_spec(round_id: str) -> TeacherSpec:
    payload = _load_json(REPORT_ROOT / round_id / "selected_spec.json")
    spec = TeacherSpec(
        title=str(payload["title"]),
        teacher_role=str(payload["teacher_role"]),
        research_family=str(payload["research_family"]),
        hypothesis=str(payload["hypothesis"]),
        sample_template=str(payload["sample_template"]),
        model_family=str(payload["model_family"]),
        target_kind=str(payload["target_kind"]),
        evaluation_contract=str(payload["evaluation_contract"]),
        feature_columns=list(payload["feature_columns"]),
        novelty_rationale=str(payload.get("novelty_rationale", "")),
    )
    spec.validate()
    return spec


def _frozen_round_id(source_round_id: str, train_end_year: int) -> str:
    return f"{source_round_id}_frozen_{train_end_year}"


def freeze_teacher_round(source_round_id: str, *, train_end_year: int, test_start_year: int, test_end_year: int) -> Dict[str, Any]:
    spec = _load_spec(source_round_id)
    dataset, stock_summary = build_dataset_for_spec(spec)
    train_df = dataset[dataset["signal_year"] <= train_end_year].copy()
    test_df = dataset[(dataset["signal_year"] >= test_start_year) & (dataset["signal_year"] <= test_end_year)].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"{source_round_id}: empty train/test after freeze split")

    frozen_round_id = _frozen_round_id(source_round_id, train_end_year)
    report_dir = REPORT_ROOT / frozen_round_id
    artifact_dir = ARTIFACT_ROOT / frozen_round_id
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "models").mkdir(parents=True, exist_ok=True)

    model = _fit_model(spec, train_df)
    model_path = artifact_dir / "models" / f"{frozen_round_id}.joblib"
    joblib.dump(model, model_path)

    train_scores = _score_model(model, spec, train_df)
    thresholds = compute_train_thresholds(train_scores, quantiles=TRAIN_BUCKET_QUANTILES)
    test_scores = _score_model(model, spec, test_df)

    train_scored = train_df.copy()
    train_scored["score"] = train_scores
    train_scored["bucket"] = assign_prediction_buckets(train_scores, thresholds)
    test_scored = test_df.copy()
    test_scored["score"] = test_scores
    test_scored["bucket"] = assign_prediction_buckets(test_scores, thresholds)

    summary_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    daily_alpha_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    scored_frames_by_year: Dict[int, pd.DataFrame] = {}
    for test_year in sorted(test_scored["signal_year"].unique().tolist()):
        year_df = test_scored[test_scored["signal_year"] == test_year].copy()
        if year_df.empty:
            continue
        scored_frames_by_year[int(test_year)] = year_df.copy()
        diag = _score_diagnostics(year_df, "score")
        baseline_return = float(year_df["future_return_5d"].mean())
        q5_df = year_df[year_df["bucket"] == 5]
        q5_return = float(q5_df["future_return_5d"].mean()) if len(q5_df) else np.nan
        alpha = q5_return - baseline_return if np.isfinite(q5_return) else np.nan
        daily_alpha = _build_daily_alpha_frame(year_df).assign(test_year=test_year)
        negative_p = _negative_alpha_pvalue(daily_alpha["daily_alpha"])
        summary_rows.append(
            {
                "test_year": test_year,
                "train_end_year": train_end_year,
                "train_sample_count": int(len(train_df)),
                "test_sample_count": int(len(year_df)),
                "q5_sample_count": int(len(q5_df)),
                "baseline_avg_return": baseline_return,
                "q5_avg_return": q5_return,
                "q5_alpha_vs_baseline": alpha,
                "negative_alpha_pvalue": negative_p,
                "rank_ic_mean": diag["rank_ic_mean"],
                "ic_mean": diag["ic_mean"],
            }
        )
        threshold_rows.append(
            {
                "test_year": test_year,
                "threshold_source": f"frozen_train_<=_{train_end_year}",
                "train_threshold_q20": float(thresholds[0]),
                "train_threshold_q40": float(thresholds[1]),
                "train_threshold_q60": float(thresholds[2]),
                "train_threshold_q80": float(thresholds[3]),
                "train_score_mean": float(np.mean(train_scores)),
                "train_score_std": float(np.std(train_scores)),
            }
        )
        for bucket in range(1, 6):
            bucket_df = year_df[year_df["bucket"] == bucket]
            avg_return = float(bucket_df["future_return_5d"].mean()) if len(bucket_df) else np.nan
            bucket_rows.append(
                {
                    "test_year": test_year,
                    "bucket": bucket,
                    "sample_count": int(len(bucket_df)),
                    "sample_share": float(len(bucket_df) / len(year_df)) if len(year_df) else np.nan,
                    "avg_return": avg_return,
                    "alpha_vs_baseline": avg_return - baseline_return if np.isfinite(avg_return) else np.nan,
                }
            )
        daily_alpha_frames.append(daily_alpha)
        importance_frames.append(_extract_importance(model, spec, year_df, test_year=test_year))

    summary_df = pd.DataFrame(summary_rows).sort_values("test_year").reset_index(drop=True)
    threshold_df = pd.DataFrame(threshold_rows).sort_values("test_year").reset_index(drop=True)
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["test_year", "bucket"]).reset_index(drop=True)
    importance_df = pd.concat(importance_frames, ignore_index=True).sort_values(["test_year", "importance_abs"], ascending=[True, False]).reset_index(drop=True)
    daily_alpha_df = pd.concat(daily_alpha_frames, ignore_index=True).sort_values(["test_year", "signal_date"]).reset_index(drop=True)
    predictions_df = (
        test_scored[["symbol", "signal_date", "entry_date", "exit_date", "future_return_5d", "score", "bucket"]]
        .assign(test_year=test_scored["signal_year"].astype(int))
        .sort_values(["signal_date", "symbol"])
        .reset_index(drop=True)
    )

    selected_spec_payload = spec.to_dict()
    selected_spec_payload.update(
        {
            "title": f"{spec.title} frozen <= {train_end_year}",
            "source_round_id": source_round_id,
            "train_end_year": train_end_year,
            "test_start_year": test_start_year,
            "test_end_year": test_end_year,
            "frozen_eval": True,
        }
    )
    _write_json(report_dir / "selected_spec.json", selected_spec_payload)
    _write_json(report_dir / "prompt.json", {"mode": "frozen_eval", "source_round_id": source_round_id})
    _write_json(report_dir / "proposal.json", {"mode": "frozen_eval", "source_round_id": source_round_id})
    _write_json(report_dir / "novelty_report.json", {"mode": "frozen_eval", "source_round_id": source_round_id})

    stock_summary.to_csv(artifact_dir / "stock_build_summary.csv", index=False)
    summary_df.to_csv(report_dir / "walkforward_yearly_summary.csv", index=False)
    threshold_df.to_csv(report_dir / "walkforward_thresholds.csv", index=False)
    bucket_df.to_csv(report_dir / "walkforward_bucket_returns.csv", index=False)
    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)
    daily_alpha_df.to_csv(report_dir / "daily_alpha_summary.csv", index=False)
    summary_df.to_csv(artifact_dir / "walkforward_yearly_summary.csv", index=False)
    threshold_df.to_csv(artifact_dir / "walkforward_thresholds.csv", index=False)
    importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
    daily_alpha_df.to_csv(artifact_dir / "daily_alpha_summary.csv", index=False)
    predictions_df.to_csv(artifact_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    nav_result = run_round_nav_backtest(
        round_id=frozen_round_id,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
        partition="frozen_eval",
        status="completed",
    )
    factor_analysis = run_teacher_factor_analysis(
        spec_title=selected_spec_payload["title"],
        model_family=spec.model_family,
        feature_columns=spec.feature_columns,
        models_by_year={int(year): model for year in scored_frames_by_year.keys()},
        scored_frames_by_year=scored_frames_by_year,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
    )

    report_lines = [
        f"# {frozen_round_id}",
        "",
        f"- source_round_id: `{source_round_id}`",
        f"- teacher_role: {spec.teacher_role}",
        f"- research_family: `{spec.research_family}`",
        f"- sample_template: `{spec.sample_template}`",
        f"- model_family: `{spec.model_family}`",
        f"- target_kind: `{spec.target_kind}`",
        f"- train_end_year: `{train_end_year}`",
        f"- test_years: `{test_start_year}..{test_end_year}`",
        f"- feature_count: `{len(spec.feature_columns)}`",
        "",
        "## Post-2022 Frozen Summary",
        "",
        _markdown_table(
            summary_df,
            [
                "test_year",
                "train_sample_count",
                "test_sample_count",
                "q5_sample_count",
                "baseline_avg_return",
                "q5_avg_return",
                "q5_alpha_vs_baseline",
                "rank_ic_mean",
            ],
        ),
        "",
        "## NAV Backtest",
        "",
        f"- total_return: `{nav_result.total_return:.2%}`",
        f"- cagr: `{nav_result.cagr:.2%}`",
        f"- max_drawdown: `{nav_result.max_drawdown:.2%}`",
        f"- hs300_total_return: `{nav_result.hs300_total_return:.2%}`",
        f"- positive_years: `{nav_result.positive_years}/{nav_result.total_years}`",
        f"- nav_curve_path: `{_relative(Path(nav_result.plot_path))}`",
        "",
        "## Detailed Factor Analysis",
        "",
        f"- factor_analysis_report: `{_relative(report_dir / 'FACTOR_ANALYSIS_REPORT.md')}`",
        f"- local_explainability_method: `{factor_analysis.summary.get('local_explainability_method', 'unknown')}`",
    ]
    (report_dir / "EXECUTION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "source_round_id": source_round_id,
        "frozen_round_id": frozen_round_id,
        "title": spec.title,
        "research_family": spec.research_family,
        "sample_template": spec.sample_template,
        "model_family": spec.model_family,
        "train_end_year": train_end_year,
        "test_start_year": test_start_year,
        "test_end_year": test_end_year,
        "mean_alpha": float(summary_df["q5_alpha_vs_baseline"].mean()),
        "positive_years": int((summary_df["q5_alpha_vs_baseline"] > 0).sum()),
        "total_years": int(len(summary_df)),
        "nav_total_return": nav_result.total_return,
        "nav_cagr": nav_result.cagr,
        "nav_max_drawdown": nav_result.max_drawdown,
        "report_dir": _relative(report_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze teacher-loop rounds at 2022 and evaluate post-2022")
    parser.add_argument("--teacher-rounds", nargs="*", default=[], help="Explicit positive teacher round ids")
    parser.add_argument("--negative-rounds", nargs="*", default=[], help="Explicit negative teacher round ids")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--negative-k", type=int, default=4)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-start-year", type=int, default=2023)
    parser.add_argument("--test-end-year", type=int, default=2026)
    parser.add_argument("--summary-name", default="frozen_teacher_selection_summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive_rounds = list(args.teacher_rounds)
    negative_rounds = list(args.negative_rounds)
    selection_df = pd.DataFrame()
    if not positive_rounds or not negative_rounds:
        auto_positive, auto_negative, selection_df = shortlist_teachers(args.top_k, args.negative_k)
        if not positive_rounds:
            positive_rounds = auto_positive
        if not negative_rounds:
            negative_rounds = auto_negative

    report_dir = REPORT_ROOT / args.summary_name
    report_dir.mkdir(parents=True, exist_ok=True)
    if not selection_df.empty:
        selection_df.to_csv(report_dir / "selection_candidates.csv", index=False)

    frozen_rows = [
        freeze_teacher_round(
            round_id,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
        )
        for round_id in positive_rounds
    ]
    summary_df = pd.DataFrame(frozen_rows).sort_values(["nav_cagr", "mean_alpha"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(report_dir / "frozen_post2022_summary.csv", index=False)
    payload = {
        "positive_round_ids": positive_rounds,
        "negative_round_ids": negative_rounds,
        "frozen_round_ids": summary_df["frozen_round_id"].tolist(),
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
    }
    _write_json(report_dir / "selection.json", payload)
    md_lines = [
        "# Frozen Teacher Selection",
        "",
        f"- positive_round_ids: `{', '.join(positive_rounds)}`",
        f"- negative_round_ids: `{', '.join(negative_rounds)}`",
        f"- frozen_round_ids: `{', '.join(summary_df['frozen_round_id'].tolist())}`",
        "",
        _markdown_table(summary_df, ["frozen_round_id", "source_round_id", "mean_alpha", "positive_years", "nav_cagr", "nav_max_drawdown"]),
    ]
    (report_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
