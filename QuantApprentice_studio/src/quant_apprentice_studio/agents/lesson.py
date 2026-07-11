from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..provenance import read_json
from ..schemas import LessonRunDescriptor
from .base import BaseAgent


class LessonAgent(BaseAgent):
    def list_lesson_runs(self) -> List[LessonRunDescriptor]:
        catalog = self.registry.load_runtime_catalog()
        out: List[LessonRunDescriptor] = []
        for alias, payload in catalog["lesson_runs"].items():
            out.append(
                LessonRunDescriptor(
                    alias=alias,
                    seed_label=str(payload["seed_label"]),
                    final_lesson_state_json=str(payload["final_lesson_state_json"]),
                    warmup_state_json=str(payload["warmup_state_json"]),
                )
            )
        return out

    def load_final_lesson_state(self, alias: str) -> Dict:
        catalog = self.registry.load_runtime_catalog()
        payload = catalog["lesson_runs"][alias]
        return read_json(Path(payload["final_lesson_state_json"]))

    def load_scope_lessons(self, alias: str) -> Dict[str, Dict]:
        state = self.load_final_lesson_state(alias)
        output: Dict[str, Dict] = {}
        for scope in state.get("teacher_scopes", []):
            round_id = str(scope.get("round_id", "")).strip()
            lesson = dict(scope.get("scorefit_lesson_json") or {})
            if round_id:
                output[round_id] = lesson
        return output

    def summarize_lesson_run(self, alias: str) -> List[Dict]:
        state = self.load_final_lesson_state(alias)
        rows = []
        for scope in state.get("teacher_scopes", []):
            lesson = dict(scope.get("scorefit_lesson_json") or {})
            rows.append(
                {
                    "round_id": scope.get("round_id", ""),
                    "source_round_id": scope.get("source_round_id", ""),
                    "lesson_name": lesson.get("lesson_name", ""),
                    "item_count": len(lesson.get("items", {})),
                    "meta_rule_count": len(lesson.get("meta_rules", [])),
                }
            )
        return rows

    def resolve_runtime_lesson(
        self,
        *,
        lesson_alias: str = "",
        final_lesson_state_json: str = "",
    ) -> Dict:
        alias = str(lesson_alias).strip()
        path_hint = str(final_lesson_state_json).strip()
        if path_hint:
            path = Path(path_hint).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"final_lesson_state_json not found: {path}")
            state = read_json(path)
            source = "explicit_final_lesson_state_json"
        else:
            if not alias:
                catalog = self.registry.load_runtime_catalog()
                alias = str(catalog.get("defaults", {}).get("alignment_seed_alias", "")).strip()
            if not alias:
                raise ValueError("No lesson alias or final_lesson_state_json was provided.")
            catalog = self.registry.load_runtime_catalog()
            payload = catalog["lesson_runs"][alias]
            path = Path(payload["final_lesson_state_json"]).expanduser().resolve()
            state = read_json(path)
            source = "lesson_alias"
        teacher_scopes = list(state.get("teacher_scopes") or [])
        return {
            "lesson_alias": alias,
            "final_lesson_state_json": str(path),
            "resolution_source": source,
            "teacher_scope_count": len(teacher_scopes),
            "teacher_scope_round_ids": [str(scope.get("round_id", "")).strip() for scope in teacher_scopes if str(scope.get("round_id", "")).strip()],
        }
