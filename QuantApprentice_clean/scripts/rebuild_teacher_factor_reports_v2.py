#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bulk rebuild branch-oriented factor-analysis Report v2 for frozen teachers.

This script upgrades existing frozen teacher report directories in place, using
already-generated explainability CSV artifacts. It does not retrain any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from quant_toolkit.teacher_loop.factor_analysis import (
    _derive_branch_report_v2,
    _factor_report_markdown,
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _selection_payloads(selection_jsons: Iterable[Path]) -> List[Tuple[Path, List[str]]]:
    payloads: List[Tuple[Path, List[str]]] = []
    for selection_json in selection_jsons:
        selection_json = selection_json.expanduser().resolve()
        payload = _load_json(selection_json)
        frozen_round_ids = [str(x).strip() for x in list(payload.get("frozen_round_ids") or []) if str(x).strip()]
        if not frozen_round_ids:
            raise ValueError(f"selection_json has no frozen_round_ids: {selection_json}")
        report_root = selection_json.parent.parent
        payloads.append((report_root, frozen_round_ids))
    return payloads


def _unique_round_targets(
    selection_jsons: Iterable[Path],
    report_root: Path | None,
    round_ids: Iterable[str],
) -> List[Tuple[Path, str]]:
    ordered: List[Tuple[Path, str]] = []
    seen = set()
    for root, frozen_round_ids in _selection_payloads(selection_jsons):
        for round_id in frozen_round_ids:
            key = (str(root.resolve()), round_id)
            if key not in seen:
                ordered.append((root, round_id))
                seen.add(key)
    if report_root is not None:
        root = report_root.expanduser().resolve()
        for round_id in round_ids:
            rid = str(round_id).strip()
            if not rid:
                continue
            key = (str(root), rid)
            if key not in seen:
                ordered.append((root, rid))
                seen.add(key)
    return ordered


def _required_paths(report_dir: Path) -> Dict[str, Path]:
    return {
        "summary_json": report_dir / "factor_analysis_summary.json",
        "report_md": report_dir / "FACTOR_ANALYSIS_REPORT.md",
        "global_csv": report_dir / "shap_global_summary.csv",
        "group_csv": report_dir / "feature_group_effectiveness.csv",
        "redundant_csv": report_dir / "feature_redundant_pairs.csv",
        "combo_csv": report_dir / "feature_combo_effects.csv",
        "local_csv": report_dir / "shap_local_examples.csv",
        "pdp_csv": report_dir / "feature_pdp_curves.csv",
        "branch_json": report_dir / "branch_rule_cards.json",
    }


def _require_files(paths: Dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if name != "branch_json" and not path.exists()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"missing required factor-analysis artifacts: {detail}")


def _spec_title(summary: Dict, report_md_path: Path, round_id: str) -> str:
    title = str(summary.get("spec_title") or "").strip()
    if title:
        return title
    if report_md_path.exists():
        first_line = report_md_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if first_line.startswith("# "):
            first_line = first_line[2:].strip()
        if first_line.endswith(" Factor Analysis"):
            first_line = first_line[: -len(" Factor Analysis")].rstrip()
        if first_line:
            return first_line
    return round_id


def _rebuild_one(report_root: Path, round_id: str) -> Dict[str, object]:
    report_dir = report_root / round_id
    if not report_dir.exists():
        raise FileNotFoundError(f"report dir not found: {report_dir}")
    paths = _required_paths(report_dir)
    _require_files(paths)

    summary = _load_json(paths["summary_json"])
    spec_title = _spec_title(summary, paths["report_md"], round_id)
    global_summary_df = pd.read_csv(paths["global_csv"])
    group_df = pd.read_csv(paths["group_csv"])
    redundant_df = pd.read_csv(paths["redundant_csv"])
    combo_df = pd.read_csv(paths["combo_csv"])
    local_df = pd.read_csv(paths["local_csv"])
    pdp_df = pd.read_csv(paths["pdp_csv"])

    branch_v2 = _derive_branch_report_v2(
        global_summary_df=global_summary_df,
        group_df=group_df,
        redundant_df=redundant_df,
        combo_df=combo_df,
        local_df=local_df,
        pdp_df=pdp_df,
    )
    _write_json(paths["branch_json"], branch_v2)

    report_text = _factor_report_markdown(
        spec_title=spec_title,
        oos_rows=int(summary.get("out_of_sample_rows", 0) or 0),
        method=str(summary.get("local_explainability_method", "unknown")),
        global_summary_df=global_summary_df,
        group_df=group_df,
        redundant_df=redundant_df,
        combo_df=combo_df,
        local_df=local_df,
        branch_v2=branch_v2,
    )
    paths["report_md"].write_text(report_text, encoding="utf-8")

    artifact_files = dict(summary.get("artifact_files") or {})
    artifact_files["branch_rule_cards_json"] = paths["branch_json"].name
    summary["spec_title"] = spec_title
    summary["report_schema_version"] = branch_v2.get("report_schema_version", "branch_oriented_v2")
    summary["branch_rule_cards"] = list(branch_v2.get("branch_cards") or [])
    summary["soft_rules"] = list(branch_v2.get("soft_rules") or [])
    summary["hard_veto_rules"] = list(branch_v2.get("hard_veto_rules") or [])
    summary["meta_rules"] = list(branch_v2.get("meta_rules") or [])
    summary["ambiguous_combo_contexts"] = list(branch_v2.get("ambiguous_combo_contexts") or [])
    summary["false_positive_contrast_pairs"] = list(branch_v2.get("false_positive_contrast_pairs") or [])
    summary["archetypes"] = list(branch_v2.get("archetypes") or [])
    summary["pdp_effect_summaries"] = list(branch_v2.get("pdp_effect_summaries") or [])
    summary["artifact_files"] = artifact_files
    _write_json(paths["summary_json"], summary)

    return {
        "report_root": str(report_root),
        "round_id": round_id,
        "report_dir": str(report_dir),
        "report_schema_version": summary.get("report_schema_version"),
        "branch_cards": len(summary["branch_rule_cards"]),
        "ambiguous_combo_contexts": len(summary["ambiguous_combo_contexts"]),
        "soft_rules": len(summary["soft_rules"]),
        "hard_veto_rules": len(summary["hard_veto_rules"]),
        "meta_rules": len(summary["meta_rules"]),
        "false_positive_contrast_pairs": len(summary["false_positive_contrast_pairs"]),
        "archetypes": len(summary["archetypes"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk rebuild teacher factor-analysis Report v2 in place")
    parser.add_argument(
        "--selection-json",
        action="append",
        default=[],
        help="Selection JSON whose frozen_round_ids should be rebuilt. Can be passed multiple times.",
    )
    parser.add_argument("--report-root", default="", help="Optional report root when passing --round-id manually.")
    parser.add_argument("--round-id", action="append", default=[], help="Optional round_id under --report-root. Can repeat.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    selection_jsons = [Path(x) for x in list(args.selection_json or []) if str(x).strip()]
    report_root = Path(args.report_root) if str(args.report_root).strip() else None
    round_ids = [str(x).strip() for x in list(args.round_id or []) if str(x).strip()]
    targets = _unique_round_targets(selection_jsons, report_root, round_ids)
    if not targets:
        raise SystemExit("no targets provided; pass --selection-json and/or --report-root with --round-id")

    results = []
    for root, round_id in targets:
        result = _rebuild_one(root, round_id)
        results.append(result)
        print(
            f"[rebuild_report_v2] round={round_id} schema={result['report_schema_version']} "
            f"branch={result['branch_cards']} ambiguous={result['ambiguous_combo_contexts']} "
            f"soft={result['soft_rules']} meta={result['meta_rules']}",
            flush=True,
        )
    print(json.dumps({"rebuilt": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
