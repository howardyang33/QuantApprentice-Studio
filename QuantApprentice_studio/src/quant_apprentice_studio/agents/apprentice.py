from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..provenance import read_json
from .base import BaseAgent


class ApprenticeAgent(BaseAgent):
    def inner_suite_root(self, *, shared_context_root: str, run_tag_base: str) -> Path:
        context = Path(shared_context_root).expanduser().resolve()
        return context / "reports" / "apprentice_loop" / "scopefit_scope_alignment_suite" / str(run_tag_base).strip()

    def build_inner_loop_suite_args(
        self,
        *,
        selection_json: str,
        run_tag_base: str,
        api_model: str,
    ) -> List[str]:
        return [
            "--selection-json",
            str(selection_json),
            "--run-tag-base",
            str(run_tag_base),
            "--api-model",
            str(api_model),
            "--alignment-sampling-strategy",
            "neutral36_topq5_v1",
            "--scorefit-variant",
            "v7_bestguard_explore_longbatch",
            "--final-lesson-selection",
            "best_composite",
            "--final-lesson-composite-method",
            "zscore_sum",
            "--api-parallel-workers",
            "128",
            "--api-failed-rerun-rounds",
            "4",
            "--api-request-max-retries",
            "1",
        ]

    def locate_final_lesson_artifact(self, *, inner_suite_root: str) -> Optional[str]:
        suite_root = Path(inner_suite_root).expanduser().resolve()
        suite_summary = suite_root / "suite_summary.json"
        if suite_summary.exists():
            try:
                payload = read_json(suite_summary)
                final_rows = list(((payload.get("final_lesson") or {}).get("per_seed_rows") or []))
                for row in final_rows:
                    path = str(row.get("final_state_json", "")).strip()
                    if path and Path(path).exists():
                        return str(Path(path).resolve())
                warmup_rows = list(((payload.get("warmup_final_lesson") or {}).get("per_seed_rows") or []))
                for row in warmup_rows:
                    path = str(row.get("final_state_json", "")).strip()
                    if path and Path(path).exists():
                        return str(Path(path).resolve())
            except Exception:
                pass
        candidates = sorted(suite_root.rglob("selected_final_lesson*.json"))
        if candidates:
            return str(candidates[-1].resolve())
        fallback = sorted(suite_root.rglob("warmup_scoped_lessons.json"))
        if fallback:
            return str(fallback[-1].resolve())
        return None

    def summarize_outputs(self, *, shared_context_root: str, run_tag_base: str) -> Dict[str, Any]:
        suite_root = self.inner_suite_root(shared_context_root=shared_context_root, run_tag_base=run_tag_base)
        suite_summary = suite_root / "suite_summary.json"
        final_lesson_artifact = self.locate_final_lesson_artifact(inner_suite_root=str(suite_root))
        return {
            "inner_suite_root": str(suite_root),
            "suite_summary_json": str(suite_summary) if suite_summary.exists() else "",
            "final_lesson_artifact_json": str(final_lesson_artifact or ""),
            "suite_summary_exists": suite_summary.exists(),
        }
