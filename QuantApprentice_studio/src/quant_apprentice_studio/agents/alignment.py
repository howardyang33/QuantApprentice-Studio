from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..markdown_tables import load_section_tables
from ..schemas import AlignmentSeedResult
from .base import BaseAgent


def _pct(text: str) -> float:
    return float(str(text).replace("%", "").strip())


class AlignmentAgent(BaseAgent):
    SECTION = "GPT-OSS 20B"

    def _rows(self) -> List[Dict[str, str]]:
        catalog = self.registry.load_runtime_catalog()
        path = Path(catalog["paper_tables"]["alignment_after_warmup"])
        tables = load_section_tables(path)
        return list(tables[self.SECTION])

    def list_after_warmup_results(self) -> List[AlignmentSeedResult]:
        rows: List[AlignmentSeedResult] = []
        for row in self._rows():
            rows.append(
                AlignmentSeedResult(
                    seed=str(row["Seed"]),
                    signals_mean_return_pct=_pct(row["Signals Mean Return (%)"]),
                    teacher_selected_mean_return_pct=_pct(row["Teacher Selected Mean Return (%)"]),
                    teacher_uplift_vs_not_selected_pct=_pct(row["Teacher Uplift vs Not Selected (%)"]),
                    teacher_score_spearman=float(row["After Warmup Teacher Score Spearman"]),
                    selected_mean_return_pct=_pct(row["After Warmup Selected Mean Return (%)"]),
                    uplift_vs_not_selected_pct=_pct(row["After Warmup Uplift vs Not Selected (%)"]),
                    gap_to_teacher_uplift_pct=_pct(row["After Warmup Gap to Teacher Uplift (%)"]),
                    batch_nav_final=float(row["After Warmup Batch NAV Final"]),
                )
            )
        return rows

    def get_after_warmup_result(self, seed: str) -> Dict:
        for item in self.list_after_warmup_results():
            if str(item.seed) == str(seed):
                return item.__dict__
        raise KeyError(f"seed not found in GPT alignment table: {seed}")
