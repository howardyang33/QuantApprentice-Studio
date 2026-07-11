from __future__ import annotations

import base64
import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .contracts import (
    build_run_contract,
    dataset_manifest_path,
    project_config_path,
    research_campaign_path,
    run_spec_path,
    save_dataset_manifest,
    save_project_config,
    save_research_campaign,
    save_run_spec,
)
from .orchestrator.pipeline import QuantPipelineOrchestrator


TASK_TYPES = [
    "full_research_pipeline",
    "scoring_only",
    "imported_asset_demo",
    "artifact_review",
]

DEMO_LOOP_PRESET_ID = "demo_loop_only_v1"
DEMO_LOOP_PRESET_ZH = "内外循环小演示"
KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
    "high_limit",
    "low_limit",
]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _slug_hint(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").strip()).strip("-").lower()
    return cleaned[:48] or fallback


def dataset_requirements(task_type: str) -> Dict[str, Any]:
    if task_type == "full_research_pipeline":
        return {
            "dataset_kind": "ohlcv_market_panel",
            "required_columns": ["symbol", "date", "open", "high", "low", "close", "volume", "amount"],
            "optional_columns": ["adj_close", "industry", "market_cap"],
            "accepted_formats": ["csv", "json", "parquet"],
            "guidance": (
                "Upload historical market data for validation and planning, or explicitly choose the A-share online K-line downloader. The system will not silently fetch market data unless you select that data source."
            ),
        }
    if task_type == "scoring_only":
        return {
            "dataset_kind": "signal_records",
            "required_columns": ["signal_id", "signal_date", "symbol", "signal_type"],
            "aliases": {"date": "signal_date"},
            "optional_columns": ["entry_date", "exit_date"],
            "accepted_formats": ["csv", "json", "parquet"],
            "guidance": (
                "Upload scored signal candidates with structured factor fields. The legacy date field is accepted as an alias for signal_date. A stock code alone is not enough."
            ),
        }
    if task_type == "imported_asset_demo":
        return {
            "dataset_kind": "demo_or_signal_records",
            "required_columns": ["signal_id", "signal_date", "symbol", "signal_type"],
            "aliases": {"date": "signal_date"},
            "optional_columns": ["entry_date", "exit_date"],
            "accepted_formats": ["csv", "json", "parquet"],
            "guidance": (
                "You may skip upload for a pure demo, use imported paper assets directly, or upload a small signal file if you want the imported lesson to score your own signals."
            ),
        }
    return {
        "dataset_kind": "artifact_bundle",
        "required_columns": [],
        "optional_columns": [],
        "accepted_formats": ["none", "json", "csv"],
        "guidance": "Artifact review usually does not require fresh market data upload.",
    }


def analyze_task_intake(
    *,
    user_request: str,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
) -> Dict[str, Any]:
    text = _compact_text(user_request)
    lowered = text.lower()

    scoring_hits = sum(
        kw in lowered
        for kw in [
            "score",
            "scoring",
            "rank",
            "signal",
            "signals",
            "打分",
            "评分",
            "排序",
            "信号",
            "选股",
        ]
    )
    full_hits = sum(
        kw in lowered
        for kw in [
            "teacher",
            "teachers",
            "full pipeline",
            "research",
            "hypothesis",
            "outer loop",
            "warmup",
            "alignment",
            "train",
            "backtest",
            "研究",
            "假设",
            "外循环",
            "内循环",
            "训练",
            "回测",
        ]
    )
    demo_hits = sum(
        kw in lowered
        for kw in [
            "demo",
            "imported",
            "replay",
            "复用",
            "演示",
            "导入",
            "现成",
        ]
    )
    review_hits = sum(
        kw in lowered
        for kw in [
            "review",
            "artifact",
            "provenance",
            "lesson set",
            "teacher zoo",
            "审计",
            "复盘",
            "报告",
            "结果",
            "产物",
        ]
    )

    task_type = "artifact_review"
    confidence = 0.52
    if full_hits >= max(scoring_hits, demo_hits, review_hits) and full_hits > 0:
        task_type = "full_research_pipeline"
        confidence = 0.82
    elif scoring_hits >= max(demo_hits, review_hits) and scoring_hits > 0:
        task_type = "scoring_only"
        confidence = 0.78
    elif demo_hits > 0:
        task_type = "imported_asset_demo"
        confidence = 0.74
    elif review_hits > 0:
        task_type = "artifact_review"
        confidence = 0.72

    mentions_data = any(
        kw in lowered
        for kw in [
            "csv",
            "json",
            "dataset",
            "data",
            "ohlc",
            "open",
            "close",
            "volume",
            "amount",
            "factor",
            "features",
            "上传",
            "数据",
            "因子",
        ]
    )
    mentions_live_fetch = any(
        kw in lowered
        for kw in [
            "real-time",
            "realtime",
            "live price",
            "fetch",
            "crawl",
            "联网",
            "实时",
            "自动抓",
            "自动获取",
        ]
    )
    mentions_only_symbol = (
        any(kw in lowered for kw in ["ticker", "stock code", "股票代码", "代码"])
        and not mentions_data
    )

    questions: List[str] = []
    if task_type == "full_research_pipeline":
        questions.append("Will you upload historical OHLCV + amount data, or do you need help preparing it into the required schema?")
        questions.append("Do you want to build a brand-new teacher zoo for this dataset, or just run a smaller rehearsal first?")
    elif task_type == "scoring_only":
        questions.append("Will you upload a single signal, a batch of signals, or a signal CSV/JSON file with structured factors?")
        if not mentions_data:
            questions.append("Do you already have the factor values for each signal, or do you still need to prepare them offline?")
    elif task_type == "imported_asset_demo":
        questions.append("Do you want a pure imported-asset walkthrough, or do you also want the imported lesson to score your own uploaded signals?")
    elif task_type == "artifact_review":
        questions.append("Which artifact family do you want to inspect first: teacher zoo, final lessons, scoring provenance, or workflow logs?")

    limitations = [
        "The current system does not automatically fetch market data from the internet unless you explicitly choose the online downloader data source.",
        "If you only provide stock codes or a natural-language request, the system will not invent missing prices or factors.",
    ]
    if task_type == "full_research_pipeline":
        limitations.append(
            "Guided Entry V1 can validate uploaded OHLCV panels, but an arbitrary local upload still does not auto-transform into the clean multi-file training layout required by the full teacher-construction loop. The online A-share K-line downloader is the direct path for that layout."
        )
    if mentions_live_fetch or mentions_only_symbol:
        limitations.append("You will need to upload local structured data before a research run or external scoring job can proceed.")

    requirements = dataset_requirements(task_type)
    ready_for_run_spec = task_type in {"imported_asset_demo", "artifact_review"} or mentions_data

    concise_summary = {
        "full_research_pipeline": "Build teachers, validate them, internalize their standards, and then score signals.",
        "scoring_only": "Score new signals with existing lessons and surface provenance / fallback clearly.",
        "imported_asset_demo": "Use imported frozen teachers and lessons for a guided demo or lightweight scoring workflow.",
        "artifact_review": "Inspect existing workflow artifacts and provenance rather than launching new training.",
    }[task_type]

    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id or _slug_hint(text, "guided-project"),
        dataset_id=dataset_id or _slug_hint(task_type, "guided-dataset"),
        run_id=run_id or _slug_hint(task_type, "guided-run"),
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )

    return {
        "task_type": task_type,
        "confidence": confidence,
        "user_request": text,
        "summary": concise_summary,
        "clarifying_questions": questions,
        "limitations": limitations,
        "dataset_requirements": requirements,
        "ready_for_run_spec": ready_for_run_spec,
        "recommended_next_step": "dataset_onboarding" if task_type != "artifact_review" else "run_spec_wizard",
        "contract_preview": contract,
    }


def _detect_format(filename: str, content: str) -> str:
    lower_name = str(filename or "").lower()
    stripped = str(content or "").lstrip()
    if lower_name.endswith(".parquet"):
        return "parquet"
    if lower_name.endswith(".json") or stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "csv"


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    return value


def _flatten_signal_record_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    flattened: List[Dict[str, Any]] = []
    factor_columns: List[str] = []
    factor_set = set()
    for row in rows:
        item = dict(row)
        features = dict(item.pop("features", {}) or {})
        merged = {str(k): _coerce_scalar(v) for k, v in item.items()}
        for key, value in features.items():
            flat_key = f"factor::{key}"
            merged[flat_key] = _coerce_scalar(value)
            if flat_key not in factor_set:
                factor_set.add(flat_key)
                factor_columns.append(flat_key)
        flattened.append(merged)
    return flattened, factor_columns


def _load_rows(filename: str, content: str, content_encoding: str = "text") -> Tuple[str, List[Dict[str, Any]], List[str]]:
    fmt = _detect_format(filename, content)
    if fmt == "parquet":
        binary = base64.b64decode(content) if content_encoding == "base64" else str(content).encode("utf-8")
        frame = pd.read_parquet(io.BytesIO(binary))
        rows = [{str(k): _coerce_scalar(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
        return fmt, rows, [str(col) for col in frame.columns]
    if fmt == "json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = [dict(item) for item in payload if isinstance(item, dict)]
        else:
            raise ValueError("JSON dataset must be an object or a list of objects.")
        flattened, factor_columns = _flatten_signal_record_rows(rows)
        columns = sorted({key for row in flattened for key in row.keys()})
        return fmt, flattened, columns + [col for col in factor_columns if col not in columns]
    reader = csv.DictReader(io.StringIO(content))
    rows = [{str(k): _coerce_scalar(v) for k, v in dict(row).items()} for row in reader]
    return fmt, rows, list(reader.fieldnames or [])


def _string_values(rows: Iterable[Mapping[str, Any]], key: str) -> List[str]:
    values = []
    for row in rows:
        raw = row.get(key)
        text = str(raw or "").strip()
        if text:
            values.append(text)
    return values


def _date_range(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    keys = ["date", "signal_date", "entry_date"]
    values: List[str] = []
    for key in keys:
        values.extend(_string_values(rows, key))
    values = sorted(set(values))
    return {
        "start": values[0] if values else "",
        "end": values[-1] if values else "",
    }


def _preview_rows(rows: Sequence[Mapping[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    return [dict(row) for row in list(rows)[:limit]]


def _normalize_symbol_code(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6) if digits else text


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _write_clean_kline_bundle(
    *,
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    stock_root = Path(str(contract["dataset_stock_klines_root"])).expanduser().resolve()
    stock_root.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw_row in rows:
        row = {str(k).strip(): v for k, v in dict(raw_row).items()}
        symbol = _normalize_symbol_code(row.get("symbol"))
        date = _date_text(row.get("date"))
        if not symbol or not date:
            continue
        item: Dict[str, Any] = {"date": date}
        for column in KLINE_COLUMNS:
            if column == "date":
                continue
            item[column] = row.get(column, "")
        grouped.setdefault(symbol, []).append(item)

    written_files: List[str] = []
    total_rows = 0
    min_date = ""
    max_date = ""
    for symbol, symbol_rows in sorted(grouped.items()):
        symbol_rows = sorted(symbol_rows, key=lambda item: str(item.get("date", "")))
        if not symbol_rows:
            continue
        path = stock_root / f"{symbol}.csv"
        present_columns = [
            column
            for column in KLINE_COLUMNS
            if column == "date" or any(str(row.get(column, "")).strip() for row in symbol_rows)
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=present_columns)
            writer.writeheader()
            for row in symbol_rows:
                writer.writerow({column: row.get(column, "") for column in present_columns})
        written_files.append(str(path))
        total_rows += len(symbol_rows)
        first_date = str(symbol_rows[0].get("date") or "")
        last_date = str(symbol_rows[-1].get("date") or "")
        if first_date and (not min_date or first_date < min_date):
            min_date = first_date
        if last_date and (not max_date or last_date > max_date):
            max_date = last_date

    return {
        "stock_kline_root": str(stock_root),
        "stock_kline_file_count": len(written_files),
        "stock_kline_files": written_files,
        "clean_kline_row_count": total_rows,
        "clean_kline_date_range": {"start": min_date, "end": max_date},
    }


def _demo_loop_pipeline_inputs(*, contract: Mapping[str, Any], stock_kline_root: str) -> Dict[str, Any]:
    cache_tag = f"{contract['project_id']}_{contract['dataset_id']}_{contract['run_id']}_{DEMO_LOOP_PRESET_ID}"
    # Keep the demo intentionally small but still faithful to the real outer-loop
    # and inner-loop contract: build teachers, evaluate/select them, then warm up lessons.
    return {
        "data_dir": str(stock_kline_root),
        "workflow_preset": DEMO_LOOP_PRESET_ID,
        "workflow_preset_zh": DEMO_LOOP_PRESET_ZH,
        "skip_final_scoring": True,
        "auto_generate_run_spec": True,
        "auto_launch_workflow": True,
        "global_env": {
            "TEACHER_LOOP_MAX_ROUNDS": "2",
            "TEACHER_LOOP_TARGET_ROUNDS": "2",
            "TEACHER_LOOP_WINDOW_START": "2020-01-01",
            "TEACHER_LOOP_WINDOW_END": "2023-12-31",
            "TEACHER_LOOP_TEST_YEARS": "2022,2023",
            "TEACHER_LOOP_BUILD_WORKERS": "8",
            "TEACHER_LOOP_CACHE_TAG": cache_tag,
            "QA_STUDIO_TEACHER_CUDA_VISIBLE_DEVICES": "1",
            "QA_STUDIO_SKIP_MARKET_BACKTEST": "true",
        },
        "stage_args": {
            "teacher_frozen_eval": [
                "--train-end-year",
                "2022",
                "--test-start-year",
                "2023",
                "--test-end-year",
                "2023",
                "--top-k",
                "4",
                "--negative-k",
                "2",
            ],
            "inner_loop_suite": [
                "--seeds",
                "1",
                "--start-date",
                "2020-01-02",
                "--end-date",
                "2023-12-29",
                "--alignment-day-count",
                "8",
                "--signal-pool-per-day",
                "8",
                "--warmup-sample-count",
                "80",
                "--warmup-batch-size",
                "40",
                "--warmup-signal-pool-per-day",
                "8",
                "--warmup-only",
                "--skip-no-lesson",
                "--skip-lesson0-alignment",
                "--api-parallel-workers",
                "32",
                "--api-failed-rerun-rounds",
                "2",
                "--api-request-max-retries",
                "1",
            ],
        },
        "stage_env": {
            "outer_loop": {"CUDA_VISIBLE_DEVICES": "1"},
            "teacher_frozen_eval": {"CUDA_VISIBLE_DEVICES": "1"},
            "inner_loop_suite": {},
        },
    }


def build_dataset_manifest(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    task_type: str,
    filename: str,
    content: str,
    content_encoding: str = "text",
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
) -> Dict[str, Any]:
    if not str(content or "").strip():
        raise ValueError("Dataset content is empty.")

    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    format_name, rows, columns = _load_rows(filename, content, content_encoding)
    requirements = dataset_requirements(task_type)
    required_columns = list(requirements["required_columns"])

    normalized_columns = [str(col).strip() for col in columns if str(col).strip()]
    missing_columns = [col for col in required_columns if col not in normalized_columns]
    warnings: List[str] = []
    fail_reasons: List[str] = []

    row_count = len(rows)
    if row_count == 0:
        fail_reasons.append("Dataset contains zero rows after parsing.")

    symbol_key = "symbol"
    if symbol_key in normalized_columns:
        symbol_count = len(set(_string_values(rows, symbol_key)))
    else:
        symbol_count = 0

    if task_type in {"full_research_pipeline", "scoring_only"} and missing_columns:
        fail_reasons.append(f"Missing required columns: {', '.join(missing_columns)}")

    if task_type == "scoring_only":
        factor_cols = [col for col in normalized_columns if col.startswith("factor::")]
        if format_name == "csv":
            factor_cols = [col for col in normalized_columns if col not in required_columns and col not in {"entry_date", "exit_date"}]
        if not factor_cols:
            warnings.append("No explicit factor columns were detected. Scoring will still need structured factor values.")
    if task_type == "full_research_pipeline" and symbol_count < 2:
        warnings.append("Very few symbols were detected. A research pipeline is usually more meaningful with a wider market panel.")

    stored_dir = Path(contract["dataset_upload_root"])
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / (Path(filename).name or "uploaded_dataset.txt")
    if content_encoding == "base64" and str(filename).lower().endswith(".parquet"):
        stored_path.write_bytes(base64.b64decode(content))
    else:
        stored_path.write_text(content, encoding="utf-8")

    kline_bundle: Dict[str, Any] = {}
    pipeline_inputs: Dict[str, Any] = {}
    full_pipeline_ready = False
    data_readiness = "ready" if not fail_reasons else "invalid"
    generated_clean_layout_note = ""
    required_ohlcv = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
    if task_type == "full_research_pipeline" and not fail_reasons and required_ohlcv.issubset(set(normalized_columns)):
        kline_bundle = _write_clean_kline_bundle(contract=contract, rows=rows)
        if int(kline_bundle.get("stock_kline_file_count") or 0) > 0:
            pipeline_inputs = _demo_loop_pipeline_inputs(
                contract=contract,
                stock_kline_root=str(kline_bundle.get("stock_kline_root") or ""),
            )
            full_pipeline_ready = True
            data_readiness = "ready_for_demo_full_research_pipeline"
            generated_clean_layout_note = (
                "Uploaded panel CSV was converted into the clean per-symbol daily K-line layout under the isolated dataset root."
            )

    manifest = {
        "profile_id": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "task_type": task_type,
        "source_type": "upload_local_dataset",
        "internet_required": False,
        "dataset_kind": requirements["dataset_kind"],
        "source_filename": Path(filename).name or "uploaded_dataset.txt",
        "stored_dataset_path": str(stored_path),
        "dataset_format": format_name,
        "row_count": row_count,
        "symbol_count": symbol_count,
        "date_range": _date_range(rows),
        "columns": normalized_columns,
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "preview_rows": _preview_rows(rows),
        "warning_reasons": warnings,
        "fail_reasons": fail_reasons,
        "valid": not fail_reasons,
        "data_isolation_status": dict(contract["data_isolation"]),
        "accepted_formats": list(requirements["accepted_formats"]),
        "guidance": requirements["guidance"],
        "data_readiness": data_readiness,
        "full_pipeline_ready": full_pipeline_ready,
        "pipeline_inputs": pipeline_inputs,
        "clean_kline_bundle": kline_bundle,
        "workflow_preset": pipeline_inputs.get("workflow_preset", ""),
        "workflow_preset_zh": pipeline_inputs.get("workflow_preset_zh", ""),
        "skip_final_scoring": bool(pipeline_inputs.get("skip_final_scoring", False)),
        "generated_clean_layout_note": generated_clean_layout_note,
        "generated_paths": {
            "dataset_manifest_json": str(dataset_manifest_path(contract)),
            "project_config_json": str(project_config_path(contract)),
            "run_spec_json": str(run_spec_path(contract)),
            "research_campaign_json": str(research_campaign_path(contract)),
        },
    }
    save_dataset_manifest(contract, manifest)
    return manifest


def build_imported_asset_manifest(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    task_type: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    requirements = dataset_requirements(task_type)
    warnings = []
    if task_type == "full_research_pipeline":
        warnings.append("Imported paper assets are suitable for demo / review / lightweight scoring, not for building a brand-new teacher zoo.")
    manifest = {
        "profile_id": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "task_type": task_type,
        "source_type": "imported_paper_assets",
        "internet_required": False,
        "dataset_kind": requirements["dataset_kind"],
        "source_filename": "",
        "stored_dataset_path": str(contract["imported_asset_root"]),
        "dataset_format": "imported_assets",
        "row_count": 0,
        "symbol_count": 0,
        "date_range": {"start": "", "end": ""},
        "columns": [],
        "required_columns": list(requirements["required_columns"]),
        "missing_columns": [],
        "preview_rows": [],
        "warning_reasons": warnings,
        "fail_reasons": [],
        "valid": True,
        "data_isolation_status": dict(contract["data_isolation"]),
        "accepted_formats": list(requirements["accepted_formats"]),
        "guidance": "This dataset source reuses imported paper assets already bundled into the studio profile.",
        "data_readiness": "ready",
        "full_pipeline_ready": False,
        "pipeline_inputs": {},
        "generated_paths": {
            "dataset_manifest_json": str(dataset_manifest_path(contract)),
            "project_config_json": str(project_config_path(contract)),
            "run_spec_json": str(run_spec_path(contract)),
            "research_campaign_json": str(research_campaign_path(contract)),
        },
    }
    save_dataset_manifest(contract, manifest)
    return manifest


def build_online_kline_dataset_manifest(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    task_type: str,
    adjust_type: str,
    earliest_date: str,
    update_indexes: bool,
    stock_kline_root: str,
    index_kline_root: str,
    failed_codes: Sequence[str],
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    requirements = dataset_requirements(task_type)
    stock_root = Path(stock_kline_root).expanduser().resolve()
    index_root = Path(index_kline_root).expanduser().resolve()
    stock_files = sorted(stock_root.glob("*.csv")) if stock_root.exists() else []
    index_files = sorted(index_root.glob("*.csv")) if index_root.exists() else []
    observed_columns = set()
    min_date = ""
    max_date = ""
    missing_columns = set()
    preview_rows: List[Dict[str, Any]] = []
    total_rows = 0

    required_columns = ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg", "high_limit", "low_limit"]
    for idx, path in enumerate(stock_files):
        try:
            with path.open("r", encoding="utf-8") as handle:
                total_rows += max(sum(1 for _ in handle) - 1, 0)
        except Exception:
            pass
        if idx >= 31:
            continue
        try:
            frame = _load_rows(path.name, path.read_text(encoding="utf-8"))[1]
        except Exception:
            continue
        if frame:
            row_keys = set(frame[0].keys())
            observed_columns.update(row_keys)
            dr = _date_range(frame)
            start = dr.get("start", "")
            end = dr.get("end", "")
            if start and (not min_date or start < min_date):
                min_date = start
            if end and (not max_date or end > max_date):
                max_date = end
            if not preview_rows:
                preview_rows = _preview_rows(frame)
    if observed_columns:
        missing_columns = {col for col in required_columns if col not in observed_columns}

    benchmark_path = index_root / "000300.csv"
    warning_reasons: List[str] = []
    fail_reasons: List[str] = []
    if not stock_files:
        fail_reasons.append("No stock K-line CSV files were generated.")
    if missing_columns:
        fail_reasons.append(f"Missing required K-line columns: {', '.join(sorted(missing_columns))}")
    if not update_indexes:
        warning_reasons.append("Index download was skipped. Full pipeline launch remains blocked until benchmark index files are available.")
    elif not benchmark_path.exists():
        warning_reasons.append("HS300 benchmark file 000300.csv is missing under index_klines.")

    full_pipeline_ready = bool(stock_files) and not missing_columns and benchmark_path.exists()
    if task_type != "full_research_pipeline":
        warning_reasons.append("Online K-line downloader mainly prepares raw market data for full research workflows; scoring-only tasks still need structured signal factors.")

    manifest = {
        "profile_id": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "task_type": task_type,
        "source_type": "online_kline_downloader",
        "internet_required": True,
        "market": "A-share",
        "dataset_kind": "kline_panel_dataset",
        "dataset_format": "csv_directory_bundle",
        "adjust_type": adjust_type,
        "start_date": earliest_date,
        "end_date": max_date or "",
        "date_range": {"start": min_date or earliest_date, "end": max_date or ""},
        "stock_kline_root": str(stock_root),
        "index_kline_root": str(index_root),
        "stored_dataset_path": str(stock_root.parent),
        "source_filename": "",
        "row_count": total_rows,
        "symbol_count": len(stock_files),
        "index_count": len(index_files),
        "columns": sorted(observed_columns),
        "required_columns": required_columns,
        "missing_columns": sorted(missing_columns),
        "preview_rows": preview_rows,
        "failed_codes": list(failed_codes),
        "warning_reasons": warning_reasons,
        "fail_reasons": fail_reasons,
        "valid": bool(stock_files) and not missing_columns,
        "task_compatible": task_type == "full_research_pipeline",
        "data_readiness": "ready" if full_pipeline_ready else ("partial" if stock_files else "invalid"),
        "full_pipeline_ready": full_pipeline_ready and task_type == "full_research_pipeline",
        "pipeline_inputs": {
            "data_dir": str(stock_root),
            "global_env": {"NAV_CURVE_HS300_INDEX_FILE": str(benchmark_path)} if benchmark_path.exists() else {},
        },
        "data_isolation_status": dict(contract["data_isolation"]),
        "accepted_formats": list(requirements["accepted_formats"]),
        "guidance": (
            "This dataset was generated by the studio online A-share K-line downloader. "
            "Only when you explicitly choose this source will the system access the network."
        ),
        "generated_paths": {
            "dataset_manifest_json": str(dataset_manifest_path(contract)),
            "project_config_json": str(project_config_path(contract)),
            "run_spec_json": str(run_spec_path(contract)),
            "research_campaign_json": str(research_campaign_path(contract)),
        },
    }
    save_dataset_manifest(contract, manifest)
    return manifest


def _build_artifact_review_plan(research_goal: str, run_label: str) -> Dict[str, Any]:
    return {
        "mode": "artifact_review",
        "research_goal": research_goal,
        "run_label": run_label,
        "system_note": "Artifact review mode does not launch teacher training or inner-loop learning.",
        "steps": [
            {
                "step_id": "S1",
                "title": "Artifact Scope Selection",
                "owner": "PlannerAgent",
                "stage_type": "review",
                "description": "Decide whether the user wants teacher-zoo inspection, lesson inspection, scoring provenance, or workflow logs.",
            },
            {
                "step_id": "S2",
                "title": "Artifact Browsing",
                "owner": "EvaluationAgent",
                "stage_type": "review",
                "description": "Open the selected artifacts inside the Advanced Research Console without launching a new pipeline.",
            },
        ],
    }


def create_guided_run_bundle(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    task_intake: Mapping[str, Any],
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    dataset_manifest: Mapping[str, Any] | None = None,
    api_model: str = "gpt-oss-20b",
    teacher_library_display_name_zh: str = "",
) -> Dict[str, Any]:
    task_type = str(task_intake.get("task_type", "")).strip() or "artifact_review"
    user_request = _compact_text(str(task_intake.get("user_request", "")).strip())
    summary = str(task_intake.get("summary", "")).strip()
    teacher_library_display_name_zh = _compact_text(
        teacher_library_display_name_zh or str(task_intake.get("teacher_library_display_name_zh") or "")
    )

    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )

    dataset_payload = dict(dataset_manifest or {})
    dataset_valid = bool(dataset_payload.get("valid", False))
    dataset_source_type = str(dataset_payload.get("source_type", "")).strip()
    pipeline_inputs = dict(dataset_payload.get("pipeline_inputs") or {})
    workflow_preset = str(pipeline_inputs.get("workflow_preset") or dataset_payload.get("workflow_preset") or "").strip()
    workflow_preset_zh = str(pipeline_inputs.get("workflow_preset_zh") or dataset_payload.get("workflow_preset_zh") or "").strip()
    skip_final_scoring = bool(pipeline_inputs.get("skip_final_scoring") or dataset_payload.get("skip_final_scoring", False))
    task_compatible = bool(dataset_payload.get("task_compatible", True))

    mode = "artifact_review"
    launchable = False
    if task_type == "full_research_pipeline":
        mode = "full_pipeline"
        launchable = bool(dataset_payload.get("full_pipeline_ready", False))
    elif task_type == "scoring_only":
        mode = "scoring_only"
        launchable = dataset_valid and task_compatible
    elif task_type == "imported_asset_demo":
        mode = "scoring_only"
        launchable = True
    elif task_type == "artifact_review":
        mode = "artifact_review"
        launchable = False

    if mode == "artifact_review":
        plan = _build_artifact_review_plan(user_request or summary, contract["run_id"])
    else:
        orchestrator = QuantPipelineOrchestrator()
        if mode == "full_pipeline":
            plan = orchestrator.build_plan(
                mode=mode,
                research_goal=user_request or summary,
                run_label=contract["run_id"],
                selection_json_hint="",
                final_lesson_state_hint="",
            )
        elif mode == "scoring_only":
            lesson_alias = "alignment_seed0005"
            plan = orchestrator.build_plan(
                mode=mode,
                research_goal=user_request or summary,
                run_label=contract["run_id"],
                lesson_alias=lesson_alias,
            )
        else:
            raise ValueError(f"Unsupported guided wizard mode: {mode}")

    project_config = {
        "profile_id": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "task_type": task_type,
        "mode": mode,
        "user_request": user_request,
        "summary": summary,
        "teacher_library_display_name_zh": teacher_library_display_name_zh,
        "allow_imported_fallback": bool(allow_imported_fallback),
        "allow_demo_fallback": bool(allow_demo_fallback),
        "dataset_manifest_json": str(dataset_manifest_path(contract)) if dataset_payload else "",
        "dataset_source_type": dataset_source_type,
        "workflow_preset": workflow_preset,
        "workflow_preset_zh": workflow_preset_zh,
        "skip_final_scoring": skip_final_scoring,
        "launchable": bool(launchable),
    }

    run_allows_imported_fallback = bool(allow_imported_fallback)
    run_allows_demo_fallback = bool(allow_demo_fallback)
    if workflow_preset == DEMO_LOOP_PRESET_ID:
        run_allows_imported_fallback = False
        run_allows_demo_fallback = False

    run_spec = {
        **contract,
        "profile_id": profile_id,
        "mode": mode,
        "research_goal": user_request or summary,
        "selection_json_hint": "",
        "final_lesson_state_json_hint": "",
        "lesson_alias": "alignment_seed0005" if mode == "scoring_only" else "",
        "api_model": str(api_model or "gpt-oss-20b").strip(),
        "task_type": task_type,
        "teacher_library_display_name_zh": teacher_library_display_name_zh,
        "dataset_manifest_json": str(dataset_manifest_path(contract)) if dataset_payload else "",
        "dataset_source_type": dataset_source_type,
        "data_dir": str(pipeline_inputs.get("data_dir", "")).strip(),
        "global_env": dict(pipeline_inputs.get("global_env") or {}),
        "stage_args": dict(pipeline_inputs.get("stage_args") or {}),
        "stage_env": dict(pipeline_inputs.get("stage_env") or {}),
        "workflow_preset": workflow_preset,
        "workflow_preset_zh": workflow_preset_zh,
        "skip_final_scoring": skip_final_scoring,
        "allow_imported_fallback": run_allows_imported_fallback,
        "allow_demo_fallback": run_allows_demo_fallback,
        "launchable": bool(launchable),
        "wizard_generated": True,
    }

    research_campaign = {
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "profile_id": profile_id,
        "mode": mode,
        "task_type": task_type,
        "research_goal": user_request or summary,
        "task_summary": summary,
        "teacher_library_display_name_zh": teacher_library_display_name_zh,
        "dataset_ready": bool(dataset_valid),
        "launchable": bool(launchable),
        "workflow_preset": workflow_preset,
        "workflow_preset_zh": workflow_preset_zh,
        "skip_final_scoring": skip_final_scoring,
        "pipeline_inputs": pipeline_inputs,
        "pipeline_plan": dict(plan),
        "policy": {
            "allow_imported_fallback": bool(run_allows_imported_fallback),
            "allow_demo_fallback": bool(run_allows_demo_fallback),
        },
        "limitations": list(task_intake.get("limitations") or []),
        "clarifying_questions": list(task_intake.get("clarifying_questions") or []),
    }

    save_project_config(contract, project_config)
    if dataset_payload:
        save_dataset_manifest(contract, dataset_payload)
    save_run_spec(contract, run_spec)
    save_research_campaign(contract, research_campaign)

    natural_language_summary = [
        f"Task type: {task_type}.",
        f"Recommended workflow mode: {mode}.",
        "This system does not fetch live market data from the internet automatically.",
    ]
    if dataset_source_type == "online_kline_downloader":
        natural_language_summary.append("This run uses an explicitly requested online A-share K-line dataset source rather than a local upload.")
    if workflow_preset == DEMO_LOOP_PRESET_ID:
        natural_language_summary.append(
            "This run uses the Studio demo-loop preset: a small outer-loop plus inner-loop rehearsal with imported/demo fallback disabled."
        )
    if dataset_payload:
        if dataset_valid:
            natural_language_summary.append(
                f"The uploaded dataset passed validation with {dataset_payload.get('row_count', 0)} rows and {dataset_payload.get('symbol_count', 0)} symbols."
            )
        else:
            natural_language_summary.append(
                "The uploaded dataset still has validation issues, so launch should stay blocked until they are fixed."
            )
    else:
        natural_language_summary.append("No dataset manifest is attached to this wizard run yet.")
    if not launchable:
        natural_language_summary.append("This wizard output is not launchable yet; review the warnings and required inputs first.")
    if task_type == "full_research_pipeline":
        if launchable:
            natural_language_summary.append("This dataset is already aligned with the clean daily K-line layout, so the full pipeline can be launched directly from the wizard.")
        else:
            natural_language_summary.append(
                "For full research runs, a generic local upload still stops at validated planning artifacts. Direct launch is enabled only when the dataset source already matches the clean daily K-line layout."
            )

    return {
        "project_config": project_config,
        "dataset_manifest": dataset_payload,
        "run_spec": run_spec,
        "research_campaign": research_campaign,
        "pipeline_plan": dict(plan),
        "launchable": bool(launchable),
        "natural_language_summary": " ".join(natural_language_summary),
        "paths": {
            "project_config_json": str(project_config_path(contract)),
            "dataset_manifest_json": str(dataset_manifest_path(contract)),
            "run_spec_json": str(run_spec_path(contract)),
            "research_campaign_json": str(research_campaign_path(contract)),
        },
    }
