from __future__ import annotations

from pathlib import Path
from typing import List

from ..provenance import read_json
from ..schemas import TeacherDescriptor
from .base import BaseAgent


class TeacherZooAgent(BaseAgent):
    def list_teachers(self) -> List[TeacherDescriptor]:
        catalog = self.registry.load_runtime_catalog()
        selection = read_json(Path(catalog["selection_json"]))
        frozen = list(selection.get("frozen_round_ids") or [])
        positive = list(selection.get("positive_round_ids") or [])
        items: List[TeacherDescriptor] = []
        for idx, round_id in enumerate(frozen):
            report_info = catalog["teacher_reports"][round_id]
            selected_spec = read_json(Path(report_info["selected_spec_json"]))
            nav_summary = read_json(Path(report_info["nav_summary_json"]))
            joblibs = catalog["teacher_artifacts"][round_id]["joblib_files"]
            source_round_id = str(selected_spec.get("source_round_id") or (positive[idx] if idx < len(positive) else round_id))
            items.append(
                TeacherDescriptor(
                    round_id=round_id,
                    source_round_id=source_round_id,
                    title=str(selected_spec.get("title", "")),
                    family=str(selected_spec.get("research_family", "")),
                    template=str(selected_spec.get("sample_template", "")),
                    walkforward_final_nav=float(nav_summary.get("final_nav", 0.0)),
                    factor_analysis_path=report_info["factor_analysis_summary_json"],
                    branch_rule_cards_path=report_info["branch_rule_cards_json"],
                    selected_spec_path=report_info["selected_spec_json"],
                    model_artifact_path=joblibs[0] if joblibs else "",
                )
            )
        return items
