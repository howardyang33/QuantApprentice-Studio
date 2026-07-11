from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseAgent
from ..provenance import read_json


class EvaluationAgent(BaseAgent):

    def build_scope_alignment_args(
        self,
        *,
        selection_json: str,
        run_tag_base: str,
        api_model: str,
        lesson_artifact_json: str = "",
    ) -> List[str]:
        args = [
            "--selection-json",
            str(selection_json),
            "--run-tag-base",
            str(run_tag_base),
            "--api-model",
            str(api_model),
            "--alignment-sampling-strategy",
            "neutral36_topq5_v1",
            "--api-parallel-workers",
            "128",
            "--api-failed-rerun-rounds",
            "4",
            "--api-failed-rerun-workers",
            "128",
            "--prompt-recipe",
            "report_v2_with_lessons",
            "--candidate-source",
            "baseline_signal",
            "--summary-variant",
            "enriched_v2",
        ]
        if str(lesson_artifact_json).strip():
            args.extend(["--lesson-artifact-json", str(lesson_artifact_json)])
        return args

    def build_market_backtest_args(
        self,
        *,
        selection_json: str,
        final_lesson_state_json: str,
        run_tag: str,
        api_model: str,
    ) -> List[str]:
        return [
            "--selection-json",
            str(selection_json),
            "--final-lesson-state-json",
            str(final_lesson_state_json),
            "--run-tag",
            str(run_tag),
            "--api-model",
            str(api_model),
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-12-31",
            "--daily-sample-size",
            "40",
            "--daily-top-pct",
            "0.10",
            "--lock-days",
            "5",
            "--sample-seed",
            "20250701",
            "--llm-decision-seed",
            "0",
            "--prompt-recipe",
            "standard",
            "--api-max-tokens",
            "384",
            "--api-parallel-workers",
            "128",
            "--api-failed-rerun-rounds",
            "4",
            "--api-request-max-retries",
            "1",
            "--private-reasoning-target-tokens",
            "0",
            "--private-reasoning-max-tokens-hint",
            "0",
        ]

    def summarize_scope_alignment_outputs(self, *, shared_context_root: str, run_tag_base: str) -> Dict:
        suite_root = (
            Path(shared_context_root).expanduser().resolve()
            / "reports"
            / "apprentice_loop"
            / f"scope_alignment_suite_{str(run_tag_base).strip()}"
        )
        comparison_json = suite_root / "scope_alignment_comparison.json"
        payload = read_json(comparison_json) if comparison_json.exists() else {}
        return {
            "scope_alignment_root": str(suite_root),
            "scope_alignment_comparison_json": str(comparison_json) if comparison_json.exists() else "",
            "aggregate_no_lesson": dict(payload.get("aggregate_no_lesson") or {}),
            "aggregate_with_lesson": dict(payload.get("aggregate_with_lesson") or {}),
        }

    def summarize_market_backtest_outputs(self, *, shared_context_root: str, run_tag: str) -> Dict:
        report_root = (
            Path(shared_context_root).expanduser().resolve()
            / "reports"
            / "apprentice_loop"
            / "market_scorefit_backtest_2025"
            / str(run_tag).strip()
        )
        market_summary_json = report_root / "summary.json"
        payload = read_json(market_summary_json) if market_summary_json.exists() else {}
        return {
            "market_backtest_root": str(report_root),
            "market_summary_json": str(market_summary_json) if market_summary_json.exists() else "",
            "market_summary": payload,
        }
