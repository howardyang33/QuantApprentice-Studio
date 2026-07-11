#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teacher-loop NAV backtest utilities and summary builder.

This module serves two purposes:
1. Compute realistic per-round NAV backtests for autonomous teacher-loop runs.
2. Build aggregated NAV summaries across all completed rounds.

The per-round utility is imported by `teacher_loop.loop` so every new teacher
round automatically gets a realistic non-levered NAV evaluation in addition to
the walk-forward Q5-alpha diagnostics.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .._paths import env_path, project_root
from .nav_curve_backtest import compute_nav_curve_fast, load_hs300

PROJECT_ROOT = env_path("QUANT_PROJECT_ROOT", project_root())
MEMORY_ROOT = env_path("QUANT_MEMORY_DIR", PROJECT_ROOT / "research_memory")
TEACHER_REPORT_ROOT = env_path("TEACHER_LOOP_REPORT_ROOT", PROJECT_ROOT / "reports" / "teacher_loop")
TEACHER_ARTIFACT_ROOT = env_path("TEACHER_LOOP_ARTIFACT_ROOT", MEMORY_ROOT / "artifacts" / "teacher_loop")
MANIFEST_PATH = MEMORY_ROOT / "indexes" / "teacher_loop_manifest.jsonl"
REPORT_ROOT = env_path("TEACHER_LOOP_NAV_REPORT_ROOT", PROJECT_ROOT / "reports" / "teacher_loop_nav_backtest")
PLOTS_DIR = REPORT_ROOT / "plots"


@dataclass
class RoundNavBacktestResult:
    round_id: str
    round_index: int
    title: str
    family: str
    template: str
    model_family: str
    partition: str
    status: str
    start_date: str
    end_date: str
    trade_days: int
    final_nav: float
    total_return: float
    cagr: float
    max_drawdown: float
    hs300_total_return: float
    excess_total_return: float
    positive_years: int
    total_years: int
    yearly_returns: Dict[int, float]
    plot_path: str
    nav_curve_path: str
    yearly_returns_path: str
    summary_path: str


def _read_completed_rounds() -> List[Dict[str, object]]:
    rows = []
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("phase") == "completed":
            rows.append(payload)
    rows.sort(key=lambda row: int(row["round_index"]))
    return rows


def _round_index_from_round_id(round_id: str) -> int:
    match = re.search(r"(\d+)", round_id)
    return int(match.group(1)) if match else 0


def _load_teacher_payload(round_dir: Path) -> Dict[str, object]:
    return json.loads((round_dir / "selected_spec.json").read_text(encoding="utf-8"))


def _load_scored_predictions(round_id: str) -> pd.DataFrame:
    pred_path = TEACHER_ARTIFACT_ROOT / round_id / "test_predictions.csv.gz"
    df = pd.read_csv(pred_path, parse_dates=["signal_date", "entry_date", "exit_date"])
    quintile_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"}
    df["quintile"] = df["bucket"].map(quintile_map)
    df["return_20d"] = df["future_return_5d"].astype(float)
    return df


def _compute_yearly_returns(nav: pd.Series) -> Dict[int, float]:
    yearly_returns: Dict[int, float] = {}
    prev_nav = 1.0
    for year in sorted(nav.index.year.unique()):
        year_nav = nav[nav.index.year == year]
        if len(year_nav) == 0:
            continue
        year_end_nav = float(year_nav.iloc[-1])
        yearly_returns[int(year)] = year_end_nav / prev_nav - 1.0
        prev_nav = year_end_nav
    return yearly_returns


def _compute_max_drawdown(nav: pd.Series) -> float:
    running_peak = nav.cummax()
    drawdown = nav / running_peak - 1.0
    return float(drawdown.min())


def _plot_teacher_nav(
    *,
    round_id: str,
    title: str,
    nav: pd.Series,
    hs300_nav: pd.Series,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(nav.index, nav.values, color="#1f6feb", linewidth=2.0, label=f"{round_id} Q5 NAV")
    ax.plot(hs300_nav.index, hs300_nav.values, color="#7f8c8d", linewidth=1.8, linestyle="--", label="HS300 NAV")
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.set_title(f"{round_id} NAV Curve\n{title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.annotate(
        f"{nav.iloc[-1]:.2f}",
        xy=(nav.index[-1], nav.iloc[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color="#1f6feb",
        va="center",
    )
    ax.annotate(
        f"HS300 {hs300_nav.iloc[-1]:.2f}",
        xy=(hs300_nav.index[-1], hs300_nav.iloc[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color="#7f8c8d",
        va="center",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_round_nav_backtest(
    *,
    round_id: str,
    report_dir: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
    partition: str = "",
    status: str = "",
    lock_days: int = 5,
) -> RoundNavBacktestResult:
    """Compute realistic NAV metrics for one completed teacher-loop round."""

    if report_dir is None:
        report_dir = TEACHER_REPORT_ROOT / round_id
    if artifact_dir is None:
        artifact_dir = TEACHER_ARTIFACT_ROOT / round_id

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    spec = _load_teacher_payload(report_dir)
    scored_df = _load_scored_predictions(round_id)
    hs300_close = load_hs300()

    min_entry = pd.to_datetime(scored_df["entry_date"]).min()
    max_exit = pd.to_datetime(scored_df["exit_date"]).max()
    hs300_idx = hs300_close.index
    trade_dates = hs300_idx[(hs300_idx >= min_entry) & (hs300_idx <= max_exit)]
    if len(trade_dates) == 0:
        raise RuntimeError(f"{round_id}: empty trade calendar for NAV backtest")

    nav = compute_nav_curve_fast(scored_df, "Q5", trade_dates, lock_days=lock_days)
    if len(nav) == 0:
        raise RuntimeError(f"{round_id}: empty NAV curve")

    hs300_sub = hs300_close.loc[trade_dates]
    hs300_nav = hs300_sub / hs300_sub.iloc[0]

    total_return = float(nav.iloc[-1] - 1.0)
    hs300_total_return = float(hs300_nav.iloc[-1] - 1.0)
    excess_total_return = total_return - hs300_total_return
    n_trade_days = int(len(nav))
    cagr = float(nav.iloc[-1] ** (252.0 / n_trade_days) - 1.0) if n_trade_days > 0 else float("nan")
    max_drawdown = _compute_max_drawdown(nav)
    yearly_returns = _compute_yearly_returns(nav)
    positive_years = int(sum(1 for ret in yearly_returns.values() if ret > 0))
    total_years = int(len(yearly_returns))

    nav_curve_df = pd.DataFrame(
        {
            "date": nav.index,
            "teacher_nav": nav.values,
            "hs300_nav": hs300_nav.reindex(nav.index).values,
        }
    )
    yearly_df = pd.DataFrame(
        [{"year": int(year), "annual_return": float(ret)} for year, ret in sorted(yearly_returns.items())]
    )

    round_plot_path = report_dir / "nav_curve.png"
    shared_plot_path = PLOTS_DIR / f"{round_id}_nav_curve.png"
    _plot_teacher_nav(
        round_id=round_id,
        title=str(spec["title"]),
        nav=nav,
        hs300_nav=hs300_nav,
        output_path=round_plot_path,
    )
    if shared_plot_path != round_plot_path:
        shared_plot_path.write_bytes(round_plot_path.read_bytes())

    nav_curve_csv = report_dir / "nav_curve.csv"
    yearly_csv = report_dir / "nav_yearly_returns.csv"
    summary_json = report_dir / "nav_summary.json"
    nav_curve_df.to_csv(nav_curve_csv, index=False)
    yearly_df.to_csv(yearly_csv, index=False)

    result = RoundNavBacktestResult(
        round_id=round_id,
        round_index=_round_index_from_round_id(round_id),
        title=str(spec["title"]),
        family=str(spec["research_family"]),
        template=str(spec["sample_template"]),
        model_family=str(spec["model_family"]),
        partition=partition,
        status=status,
        start_date=trade_dates[0].strftime("%Y-%m-%d"),
        end_date=trade_dates[-1].strftime("%Y-%m-%d"),
        trade_days=n_trade_days,
        final_nav=float(nav.iloc[-1]),
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        hs300_total_return=hs300_total_return,
        excess_total_return=excess_total_return,
        positive_years=positive_years,
        total_years=total_years,
        yearly_returns=yearly_returns,
        plot_path=str(round_plot_path),
        nav_curve_path=str(nav_curve_csv),
        yearly_returns_path=str(yearly_csv),
        summary_path=str(summary_json),
    )
    summary_json.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Keep artifact copies alongside training outputs for memory references.
    (artifact_dir / "nav_curve.csv").write_bytes(nav_curve_csv.read_bytes())
    (artifact_dir / "nav_yearly_returns.csv").write_bytes(yearly_csv.read_bytes())
    (artifact_dir / "nav_summary.json").write_bytes(summary_json.read_bytes())
    (artifact_dir / "nav_curve.png").write_bytes(round_plot_path.read_bytes())

    return result


def build_teacher_loop_nav_summary(round_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Rebuild the aggregated NAV summary across completed autonomous rounds."""

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    completed = _read_completed_rounds()
    if round_ids is not None:
        allow = set(round_ids)
        completed = [row for row in completed if str(row["round_id"]) in allow]

    summary_rows: List[Dict[str, object]] = []
    yearly_rows: List[Dict[str, object]] = []

    for row in completed:
        round_id = str(row["round_id"])
        round_dir = PROJECT_ROOT / str(row["report_dir"])
        artifact_dir = PROJECT_ROOT / "research_memory" / "artifacts" / "teacher_loop" / round_id
        nav_result = run_round_nav_backtest(
            round_id=round_id,
            report_dir=round_dir,
            artifact_dir=artifact_dir,
            partition=str(row.get("zoo_partition", "")),
            status=str(row.get("status", "")),
        )

        summary_row = {
            "round_id": nav_result.round_id,
            "round_index": nav_result.round_index,
            "title": nav_result.title,
            "family": nav_result.family,
            "template": nav_result.template,
            "model_family": nav_result.model_family,
            "partition": nav_result.partition,
            "status": nav_result.status,
            "start_date": nav_result.start_date,
            "end_date": nav_result.end_date,
            "trade_days": nav_result.trade_days,
            "final_nav": nav_result.final_nav,
            "total_return": nav_result.total_return,
            "cagr": nav_result.cagr,
            "max_drawdown": nav_result.max_drawdown,
            "hs300_total_return": nav_result.hs300_total_return,
            "excess_total_return": nav_result.excess_total_return,
            "positive_years": nav_result.positive_years,
            "total_years": nav_result.total_years,
            "plot_path": nav_result.plot_path,
        }
        for year, ret in nav_result.yearly_returns.items():
            summary_row[f"return_{year}"] = ret
            yearly_rows.append(
                {
                    "round_id": nav_result.round_id,
                    "round_index": nav_result.round_index,
                    "title": nav_result.title,
                    "year": int(year),
                    "annual_return": ret,
                }
            )
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("round_index").reset_index(drop=True)
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["round_index", "year"]).reset_index(drop=True)

    summary_csv = REPORT_ROOT / "teacher_loop_nav_summary.csv"
    yearly_csv = REPORT_ROOT / "teacher_loop_nav_yearly_returns.csv"
    summary_md = REPORT_ROOT / "teacher_loop_nav_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    yearly_df.to_csv(yearly_csv, index=False)

    view = summary_df.copy()
    for col in ["final_nav"]:
        view[col] = view[col].map(lambda x: f"{x:.4f}")
    for col in ["total_return", "cagr", "max_drawdown", "hs300_total_return", "excess_total_return"]:
        view[col] = summary_df[col].map(lambda x: f"{x:.2%}")
    view["plot_link"] = view["plot_path"].map(lambda p: f"[plot]({p})")

    year_cols = sorted([col for col in summary_df.columns if col.startswith("return_")])
    for col in year_cols:
        view[col] = summary_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    cols = [
        "round_id",
        "family",
        "template",
        "model_family",
        "partition",
        "final_nav",
        "total_return",
        "cagr",
        "max_drawdown",
        "hs300_total_return",
        "excess_total_return",
        "positive_years",
        "total_years",
        *year_cols,
        "plot_link",
    ]
    lines = [
        "# Teacher Loop NAV Backtest Summary",
        "",
        "Backtest assumptions:",
        "",
        "- Strategy uses each teacher's `Q5` signals only.",
        "- Max gross exposure is 100% with fixed 5-slot rolling allocation (`lock_days=5`).",
        "- No leverage, no transaction cost, no slippage.",
        "- NAV is built from the first walk-forward test year to the last available test year.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, record in view[cols].iterrows():
        lines.append("| " + " | ".join(str(record[col]) for col in cols) + " |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary_df


def main() -> None:
    summary_df = build_teacher_loop_nav_summary()
    print(f"wrote {REPORT_ROOT / 'teacher_loop_nav_summary.csv'}")
    print(f"wrote {REPORT_ROOT / 'teacher_loop_nav_yearly_returns.csv'}")
    print(f"wrote {REPORT_ROOT / 'teacher_loop_nav_summary.md'}")
    print(f"rounds={len(summary_df)} plots_dir={PLOTS_DIR}")


if __name__ == "__main__":
    main()
