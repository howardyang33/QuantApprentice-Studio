from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..markdown_tables import load_section_tables
from ..provenance import read_json
from ..schemas import MarketRunDescriptor
from .base import BaseAgent


def _pct(text: str) -> float:
    return float(str(text).replace("%", "").strip())


class BacktestAgent(BaseAgent):
    SECTION = "GPT-OSS 20B"

    def _rows(self) -> List[Dict[str, str]]:
        catalog = self.registry.load_runtime_catalog()
        path = Path(catalog["paper_tables"]["market_after_warmup"])
        tables = load_section_tables(path)
        return list(tables[self.SECTION])

    def list_market_runs(self) -> List[MarketRunDescriptor]:
        catalog = self.registry.load_runtime_catalog()
        out: List[MarketRunDescriptor] = []
        for alias, payload in catalog["market_runs"].items():
            out.append(
                MarketRunDescriptor(
                    alias=alias,
                    window=str(payload["window"]),
                    summary_json=str(payload["summary_json"]),
                    llm_signal_scores_json=str(payload["llm_signal_scores_json"]),
                    llm_daily_nav_json=str(payload["llm_daily_nav_json"]),
                    teacher_daily_nav_json=str(payload["teacher_daily_nav_json"]),
                )
            )
        return out

    def load_market_summary(self, alias: str) -> Dict:
        catalog = self.registry.load_runtime_catalog()
        return read_json(Path(catalog["market_runs"][alias]["summary_json"]))

    def market_table_rows(self) -> List[Dict]:
        return self._rows()
