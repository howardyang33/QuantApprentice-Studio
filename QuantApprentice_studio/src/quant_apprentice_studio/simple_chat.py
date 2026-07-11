from __future__ import annotations

import json
import re
import threading
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from .contracts import (
    build_run_contract,
    dataset_manifest_path,
    ensure_contract_dirs,
    project_config_path,
    research_campaign_path,
    run_spec_path,
)
from .console_views import build_lesson_set_view, build_provenance_view, build_run_monitor
from .data_jobs import KlineDownloadJobManager
from .guided_entry import (
    DEMO_LOOP_PRESET_ID,
    DEMO_LOOP_PRESET_ZH,
    analyze_task_intake,
    build_dataset_manifest,
    build_imported_asset_manifest,
    create_guided_run_bundle,
    dataset_requirements,
)
from .orchestrator.runner import WorkflowRunner
from .provenance import read_json, write_json
from .simple_scoring import run_simple_scoring
from .simple_stock_scoring import default_recent_kline_earliest, score_stock_code_live


TASK_TYPE_ZH = {
    "full_research_pipeline": "完整研究流程",
    "scoring_only": "信号打分",
    "imported_asset_demo": "系统已有老师库演示",
    "artifact_review": "产物复盘",
    "online_kline_download": "在线下载 A 股 K 线",
    "unclear": "待澄清任务",
}

AGENT_TIMELINE_DEFS = [
    {
        "agent_name": "Planner Agent",
        "mapped_stage": "research_spec / task_intake / run_spec",
        "stage_ids": ["research_spec"],
        "expert_workspace": "full-pipeline",
        "relevant_modes": ["full_pipeline", "outer_loop_only", "inner_loop_only", "scoring_only", "artifact_review"],
    },
    {
        "agent_name": "Hypothesis Agent",
        "mapped_stage": "research_spec / hypothesis artifact",
        "stage_ids": ["research_spec", "outer_loop"],
        "expert_workspace": "full-pipeline",
        "relevant_modes": ["full_pipeline", "outer_loop_only"],
    },
    {
        "agent_name": "FactorDesign Agent",
        "mapped_stage": "factor_design / teacher_policy / factor section",
        "stage_ids": ["research_spec", "outer_loop"],
        "expert_workspace": "full-pipeline",
        "relevant_modes": ["full_pipeline", "outer_loop_only"],
    },
    {
        "agent_name": "TeacherTraining Agent",
        "mapped_stage": "outer_loop / teacher_training / candidate_teacher",
        "stage_ids": ["outer_loop"],
        "expert_workspace": "full-pipeline",
        "relevant_modes": ["full_pipeline", "outer_loop_only"],
    },
    {
        "agent_name": "Verification Agent",
        "mapped_stage": "validation / frozen_eval / verified metrics",
        "stage_ids": ["teacher_frozen_eval", "outer_loop"],
        "expert_workspace": "library",
        "relevant_modes": ["full_pipeline", "outer_loop_only"],
    },
    {
        "agent_name": "Explainability Agent",
        "mapped_stage": "explainability_report",
        "stage_ids": ["teacher_frozen_eval"],
        "expert_workspace": "library",
        "relevant_modes": ["full_pipeline", "outer_loop_only"],
    },
    {
        "agent_name": "TeacherSelection Agent",
        "mapped_stage": "teacher_selection / formal_selection",
        "stage_ids": ["TeacherSelectionAgent"],
        "expert_workspace": "library",
        "relevant_modes": ["full_pipeline", "inner_loop_only"],
    },
    {
        "agent_name": "Apprentice Agent",
        "mapped_stage": "inner_loop / warmup / final_lesson_set",
        "stage_ids": ["inner_loop_suite", "final_lesson_set"],
        "expert_workspace": "library",
        "relevant_modes": ["full_pipeline", "inner_loop_only"],
    },
    {
        "agent_name": "Evaluation Agent",
        "mapped_stage": "alignment / market_backtest / evaluation",
        "stage_ids": ["final_lesson_set", "inner_loop_suite"],
        "expert_workspace": "provenance",
        "relevant_modes": ["full_pipeline", "inner_loop_only"],
    },
    {
        "agent_name": "SignalScoring Agent",
        "mapped_stage": "scoring / live_score / batch_score",
        "stage_ids": ["SignalScoringAgent"],
        "expert_workspace": "scoring",
        "relevant_modes": ["full_pipeline", "scoring_only"],
    },
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_recent_kline_earliest() -> str:
    # Simple Mode uses a lightweight recent context: about 120 trading days.
    return default_recent_kline_earliest()


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _visible_user_text(text: str) -> str:
    raw = str(text or "")
    for marker in [
        "\n\n用户允许普通模式",
        "\n\n用户已在普通模式",
        "\n\nstock_codes=",
    ]:
        if marker in raw:
            raw = raw.split(marker, 1)[0]
            break
    return _compact(raw)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _language(text: str) -> str:
    has_cjk = _contains_cjk(text)
    has_ascii = bool(re.search(r"[a-zA-Z]", str(text or "")))
    if has_cjk and has_ascii:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def _teacher_library_display_name_zh(text: str) -> str:
    """Extract a user-facing Chinese teacher-library name from natural language.

    The name is product metadata only. Project/dataset/run ids remain generated
    by Studio, so users do not need to manage paths or ids themselves.
    """

    compact = _compact(text)
    patterns = [
        r"老师库[，,：:\s]*(?:叫|命名为|名称是|名字叫)\s*[「“\"']?([A-Za-z0-9_\-\u4e00-\u9fff（）()· ]{2,40})[」”\"']?",
        r"(?:命名为|叫做|叫|名字叫)\s*[「“\"']?([A-Za-z0-9_\-\u4e00-\u9fff（）()· ]{2,40}?老师库)[」”\"']?",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            value = match.group(1).strip(" 「」“”\"'，,。.;；：:")
            return value[:40]
    return ""


def _is_new_teacher_library_request(text: str) -> bool:
    lowered = _compact(text).lower()
    return (
        any(kw in lowered for kw in ["训练", "新老师", "老师库", "teacher library", "teacher zoo", "new teacher"])
        and any(kw in lowered for kw in ["老师库", "teacher", "teacher zoo", "teacher library"])
    )


def _chat_paths(contract: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(contract["run_root"]) / "chat"
    return {
        "root": root,
        "session_json": root / "session.json",
        "task_state_json": root / "task_state.json",
        "messages_jsonl": root / "messages.jsonl",
    }


def _read_optional(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _safe_path_exists(path: str) -> bool:
    try:
        return bool(path) and Path(path).expanduser().exists()
    except Exception:
        return False


def _existing_artifacts(contract: Dict[str, Any]) -> Dict[str, Any]:
    workflow_result = Path(contract["workflow_root"]) / "workflow_result.json"
    paths = {
        "project_config_json": project_config_path(contract),
        "dataset_manifest_json": dataset_manifest_path(contract),
        "run_spec_json": run_spec_path(contract),
        "research_campaign_json": research_campaign_path(contract),
        "workflow_result_json": workflow_result,
    }
    return {
        key: {"path": str(path), "exists": path.exists(), "payload": _read_optional(path)}
        for key, path in paths.items()
    }


def _task_type_from_message(message: str, guided: Dict[str, Any]) -> tuple[str, float]:
    text = _compact(message)
    lowered = text.lower()
    if not text:
        return "unclear", 0.0
    stock_code_like = _data_mentions(text)["mentions_stock_code_only"]
    explicit_download = any(kw in lowered for kw in ["下载", "更新", "拉取", "获取", "缓存", "download", "update", "fetch"]) and any(
        kw in lowered for kw in ["kline", "k-line", "k线", "k 线", "行情"]
    )
    if stock_code_like and not explicit_download:
        return "scoring_only", 0.86
    online_hits = any(
        kw in lowered
        for kw in [
            "kline",
            "k-line",
            "akshare",
            "tushare",
            "qfq",
            "hfq",
            "前复权",
            "后复权",
            "k线",
            "行情下载",
            "下载行情",
            "下载k",
            "下载 k",
            "下载 k 线",
            "k 线",
            "取 k 线",
            "联网取",
        ]
    )
    if online_hits:
        return "online_kline_download", 0.84

    obvious_hits = any(
        kw in lowered
        for kw in [
            "score",
            "signal",
            "teacher",
            "pipeline",
            "research",
            "artifact",
            "lesson",
            "打分",
            "信号",
            "研究",
            "训练",
            "回测",
            "复盘",
            "产物",
            "导入",
        ]
    )
    guided_type = str(guided.get("task_type") or "artifact_review")
    confidence = float(guided.get("confidence") or 0.0)
    if not obvious_hits and confidence <= 0.55:
        return "unclear", 0.38
    return guided_type, confidence


def _canonical_user_goal_en(task_type: str, message: str) -> str:
    message = _compact(message)
    templates = {
        "full_research_pipeline": "Run a full QuantApprentice research pipeline on an isolated user-provided dataset.",
        "scoring_only": "Score structured signal candidates using QuantApprentice final lessons and expose provenance.",
        "imported_asset_demo": "Use the built-in trained teacher library and final lessons for a controlled demonstration or lightweight scoring task.",
        "artifact_review": "Review existing QuantApprentice workflow artifacts, teacher zoo, lesson set, scoring outputs, and provenance.",
        "online_kline_download": "Prepare A-share OHLCV data through the explicit online K-line downloader before any research workflow.",
        "unclear": "Clarify the user's QuantApprentice task before creating a run specification.",
    }
    return f"{templates.get(task_type, templates['unclear'])} User request: {message}"


def _normalized_goal_zh(task_type: str, message: str) -> str:
    compact = _compact(message)
    if _contains_cjk(compact):
        return compact
    zh_prefix = {
        "full_research_pipeline": "用户希望运行完整研究流程",
        "scoring_only": "用户希望对结构化信号进行打分",
        "imported_asset_demo": "用户希望使用系统已有老师库做演示或轻量打分",
        "artifact_review": "用户希望查看已有工作流产物",
        "online_kline_download": "用户希望显式在线下载 A 股 K 线数据",
        "unclear": "用户意图尚不清晰，需要继续澄清",
    }.get(task_type, "用户意图尚不清晰")
    return f"{zh_prefix}：{compact}" if compact else zh_prefix


def _data_mentions(message: str) -> Dict[str, bool]:
    lowered = message.lower()
    return {
        "mentions_dataset": any(
            kw in lowered
            for kw in [
                "csv",
                "json",
                "parquet",
                "dataset",
                "data",
                "ohlc",
                "open",
                "close",
                "volume",
                "amount",
                "上传",
                "数据",
                "因子",
                "文件",
            ]
        ),
        "mentions_stock_code_only": bool(re.search(r"(?<!\d)\d{6}(?!\d)", message)) and not any(
            kw in lowered for kw in ["csv", "json", "parquet", "factor", "因子", "特征", "open", "close"]
        ),
        "mentions_date": bool(re.search(r"(?<!\d)20\d{6}(?!\d)|(?<!\d)20\d{2}[-/]?\d{0,4}(?!\d)", message)),
        "mentions_adjust": any(kw in lowered for kw in ["qfq", "hfq", "前复权", "后复权"]),
    }


def _stock_codes_from_text(message: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"(?<!\d)\d{6}(?!\d)", str(message or ""))))


def _is_stock_review_request(message: str) -> bool:
    text = _compact(message)
    if not _stock_codes_from_text(text):
        return False
    lowered = text.lower()
    intent_words = [
        "看看",
        "看一下",
        "怎么样",
        "走势",
        "趋势",
        "评分",
        "打分",
        "分析",
        "能不能",
        "要不要",
        "继续",
        "股票",
        "stock",
        "ticker",
        "score",
        "rate",
        "analyze",
        "trend",
    ]
    return any(word in lowered for word in intent_words) or len(text) <= 18


def _is_confirmation_message(message: str) -> bool:
    text = _compact(message).lower()
    if not text:
        return False
    positive_patterns = [
        "开始",
        "确认",
        "可以",
        "继续",
        "同意",
        "启动",
        "执行",
        "run it",
        "start",
        "go ahead",
        "confirm",
        "yes",
        "ok",
    ]
    negative_patterns = ["不要", "取消", "先别", "不用", "stop", "cancel", "no"]
    if any(pattern in text for pattern in negative_patterns):
        return False
    return any(pattern in text for pattern in positive_patterns)


def _stock_code_live_action(
    *,
    stock_codes: List[str],
    enabled: bool = True,
    reason_zh: str = "",
) -> Dict[str, Any]:
    code_text = ", ".join(stock_codes)
    return {
        "action_id": "score_stock_code_live",
        "label_zh": "查看这只股票",
        "type": "auto_agent_action",
        "enabled": bool(enabled and stock_codes),
        "reason_zh": reason_zh
        or f"已识别股票代码 {code_text}，会尝试补齐近 120 个交易日左右 K 线并调用本地 GPT-OSS。",
        "target_api": "/chat/action",
        "expert_link": "scoring",
    }


def _friendly_stock_reason_zh(scoring: Dict[str, Any]) -> str:
    total = scoring.get("total_score")
    score_60 = scoring.get("score_60d")
    score_120 = scoring.get("score_120d")
    try:
        total_value = float(total)
    except Exception:
        total_value = 50.0
    try:
        diff = float(score_60) - float(score_120)
    except Exception:
        diff = 0.0

    if total_value >= 70:
        base = "四个老师模型整体给分偏高，说明它比较接近历史上更容易跑出正期望的技术形态。"
    elif total_value >= 55:
        base = "整体评分略偏正面，说明它有一些可取的技术特征，但还不是特别强的一致性机会。"
    elif total_value >= 40:
        base = "整体评分中性偏谨慎，说明它有部分信号可看，但还没有明显落入老师模型最舒服的区域。"
    else:
        base = "整体评分偏低，说明它和历史上更占优的突破/回调形态匹配度不高。"

    if diff >= 6:
        window = "近 60 日视角明显好于 120 日，短线有修复迹象，但中期确认还不够。"
    elif diff <= -6:
        window = "近 120 日视角好于 60 日，说明中期结构略稳，但最近短线表现偏弱。"
    else:
        window = "近 60 日和 120 日视角差异不大，短线和中期判断比较一致。"
    cues = list((scoring.get("feature_diagnostics_zh") or {}).get("cues_zh") or [])
    clean_cues = [str(item).rstrip("。；; ") for item in cues[:3] if str(item).strip()]
    cue_text = f"核心因子线索：{'；'.join(clean_cues)}。" if clean_cues else ""
    return f"{base}{window}{cue_text}"


def _teacher_domain_name_zh(row: Dict[str, Any]) -> str:
    round_id = str(row.get("round_id") or "").lower()
    title = str(row.get("title") or "").lower()
    merged = f"{round_id} {title}"
    if "038" in merged or "breakout" in merged:
        return "突破延续老师"
    if "042" in merged:
        return "均线回调老师"
    if "050" in merged:
        return "动量回调老师"
    if "026" in merged:
        return "量能-KDJ 回调老师"
    if "breakout" in merged:
        return "突破延续老师"
    if "volume" in merged and "kdj" in merged:
        return "量能-KDJ 老师"
    if "pullback" in merged:
        return "趋势回调老师"
    return "综合技术形态老师"


def _score_band_zh(score: Any) -> str:
    try:
        value = float(score)
    except Exception:
        return "匹配度未知"
    if value >= 70:
        return "高度匹配"
    if value >= 55:
        return "中等偏强"
    if value >= 40:
        return "部分匹配"
    if value >= 25:
        return "匹配偏弱"
    return "明显不匹配"


def _teacher_domain_focus_zh(domain_name: str) -> str:
    if "突破" in domain_name:
        return "主要看趋势突破后的延续性、量能确认和波动率是否支持继续上行"
    if "均线" in domain_name:
        return "主要看价格贴近 MA20 后的回调承接、波动率收敛和量价配合"
    if "动量回调" in domain_name:
        return "主要看短线回调后动量是否重新转强，以及 KDJ/量能是否给出承接信号"
    if "量能-KDJ" in domain_name:
        return "主要看量能动量和 KDJ 回调结构是否同时落在舒适区"
    return "主要看多因子技术形态是否落在该老师的舒适区"


def _teacher_note_zh(row: Dict[str, Any]) -> str:
    domain_name = _teacher_domain_name_zh(row)
    score = row.get("score", "-")
    band = _score_band_zh(score)
    focus = _teacher_domain_focus_zh(domain_name)
    return f"{band}。{focus}。"


def _format_teacher_score_lines(scoring: Dict[str, Any]) -> List[str]:
    rows = list(scoring.get("teacher_scores") or [])
    lines: List[str] = []
    for row in rows[:4]:
        name = str(row.get("display_name_zh") or "").strip() or _teacher_domain_name_zh(row)
        score = row.get("score", "-")
        note = str(row.get("note_zh") or "").strip() or _teacher_note_zh(row)
        lines.append(f"- {name}: {score}/100。{note}")
    return lines


def _format_feature_diagnostic_lines(scoring: Dict[str, Any]) -> List[str]:
    diagnostics = dict(scoring.get("feature_diagnostics_zh") or {})
    cues = list(diagnostics.get("cues_zh") or [])
    risks = list(diagnostics.get("risk_flags_zh") or [])
    lines = [f"- {cue}" for cue in cues[:7]]
    if risks:
        risk_text = "；".join(str(item).rstrip("。；; ") for item in risks[:3] if str(item).strip())
        lines.append(f"- 风险项：{risk_text}。")
    return lines


def _missing_fields(task_type: str, message: str, artifacts: Dict[str, Any]) -> List[str]:
    mentions = _data_mentions(message)
    missing: List[str] = []
    has_manifest = bool(artifacts["dataset_manifest_json"]["exists"])
    has_run_spec = bool(artifacts["run_spec_json"]["exists"])

    if task_type == "full_research_pipeline":
        if not has_manifest and not mentions["mentions_dataset"]:
            missing.append("market_dataset_or_online_kline_source")
        if not has_run_spec:
            missing.append("research_goal_for_run_spec")
    elif task_type == "scoring_only":
        if mentions["mentions_stock_code_only"] or not mentions["mentions_dataset"]:
            missing.append("structured_signal_record_with_factor_columns")
    elif task_type == "online_kline_download":
        if not re.search(r"\b\d{6}\b", message):
            missing.append("stock_codes")
        if not mentions["mentions_date"]:
            missing.append("earliest_date")
        if not mentions["mentions_adjust"]:
            missing.append("adjust_type_qfq_or_hfq")
    elif task_type == "unclear":
        missing.append("task_goal")
    return missing


def _requirements_zh(task_type: str) -> List[str]:
    if task_type == "online_kline_download":
        return [
            "需要用户显式选择在线下载数据源。",
            "需要 stock codes、earliest date、adjust type(qfq/hfq)，可选择 full refresh 或 incremental。",
            "依赖服务器网络、akshare、tushare 与 TUSHARE_TOKEN；不可用时仍可上传本地数据。",
        ]
    req = dataset_requirements(task_type if task_type in {"full_research_pipeline", "scoring_only", "imported_asset_demo", "artifact_review"} else "artifact_review")
    cols = req.get("required_columns") or []
    if not cols:
        return [str(req.get("guidance") or "该任务通常不需要上传新的结构化行情数据。")]
    return [
        f"需要字段：{', '.join(cols)}。",
        str(req.get("guidance") or ""),
    ]


def _clarifying_questions_zh(task_type: str, missing: List[str]) -> List[str]:
    if task_type == "full_research_pipeline":
        return [
            "你要上传本地 OHLCV 数据，还是显式使用 A 股 K 线在线下载器？",
            "这次是正式完整流程，还是先跑一个缩小 rehearsal？",
        ]
    if task_type == "scoring_only":
        return [
            "你会上传单个信号 JSON，还是一批信号文件？",
            "每个信号是否已经包含 lesson 打分需要的结构化因子？",
        ]
    if task_type == "online_kline_download":
        return [
            "请提供股票代码列表，例如 000001, 600519。",
            "请确认 earliest date 与复权方式 qfq/hfq。",
        ]
    if task_type == "imported_asset_demo":
        return ["你想直接用系统已有老师库评分，还是先查看老师库和最终经验规则集？"]
    if task_type == "artifact_review":
        return ["你想优先查看 Teacher Zoo、Lesson Set、Scoring 结果，还是完整 provenance？"]
    return ["你希望做完整研究流程、信号打分、使用系统已有老师库演示，还是查看已有产物？"]


def _stage_from_artifacts(artifacts: Dict[str, Any]) -> str:
    workflow_payload = artifacts["workflow_result_json"].get("payload")
    if workflow_payload:
        status = workflow_payload.get("status") or workflow_payload.get("workflow_status")
        return f"workflow_{status or 'available'}"
    if artifacts["run_spec_json"]["exists"]:
        return "run_spec_ready"
    if artifacts["dataset_manifest_json"]["exists"]:
        return "dataset_manifest_ready"
    if artifacts["project_config_json"]["exists"]:
        return "project_initialized"
    return "task_intake"


def _artifact_links(contract: Dict[str, Any], artifacts: Dict[str, Any], chat_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    rows = [
        ("project_config.json", artifacts["project_config_json"]),
        ("dataset_manifest.json", artifacts["dataset_manifest_json"]),
        ("run_spec.json", artifacts["run_spec_json"]),
        ("research_campaign.json", artifacts["research_campaign_json"]),
        ("workflow_result.json", artifacts["workflow_result_json"]),
        ("chat/session.json", {"path": str(chat_paths["session_json"]), "exists": chat_paths["session_json"].exists()}),
        ("chat/task_state.json", {"path": str(chat_paths["task_state_json"]), "exists": chat_paths["task_state_json"].exists()}),
        ("chat/messages.jsonl", {"path": str(chat_paths["messages_jsonl"]), "exists": chat_paths["messages_jsonl"].exists()}),
    ]
    return [
        {
            "label": label,
            "path": row["path"],
            "exists": bool(row["exists"]),
            "source": "current_workflow_asset" if str(row["path"]).startswith(str(contract["run_root"])) else "dataset_asset",
        }
        for label, row in rows
    ]


def _append_user_message(
    *,
    chat_paths: Dict[str, Path],
    session_id: str,
    text: str,
    task_state: Dict[str, Any],
    interpreted_as: str = "",
) -> None:
    row = {
        "timestamp": _now_iso(),
        "session_id": session_id,
        "role": "user",
        "content_original": text,
        "normalized_user_goal_zh": task_state.get("normalized_user_goal_zh", ""),
        "canonical_user_goal_en": task_state.get("canonical_user_goal_en", ""),
    }
    if interpreted_as:
        row["interpreted_as"] = interpreted_as
    with chat_paths["messages_jsonl"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recommended_actions(task_type: str, missing: List[str], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    has_manifest = bool(artifacts["dataset_manifest_json"]["exists"])
    has_run_spec = bool(artifacts["run_spec_json"]["exists"])
    has_workflow_result = bool(artifacts["workflow_result_json"]["exists"])
    has_structured_signal = "structured_signal_record_with_factor_columns" not in set(missing)
    can_generate_spec = task_type in {"full_research_pipeline", "imported_asset_demo", "artifact_review"} and (
        has_manifest or task_type in {"imported_asset_demo", "artifact_review"}
    )
    return [
        {
            "action_id": "upload_dataset",
            "label_zh": "上传并生成数据清单",
            "type": "confirm_then_api",
            "enabled": task_type in {"full_research_pipeline", "scoring_only"},
            "reason_zh": "上传本地 CSV/JSON/Parquet 并生成 dataset_manifest.json。",
            "target_api": "/guided/dataset-onboarding",
            "expert_link": "simple",
        },
        {
            "action_id": "start_online_kline_download",
            "label_zh": "确认联网下载 A 股 K 线",
            "type": "confirm_then_api",
            "enabled": task_type == "online_kline_download",
            "reason_zh": "只有点击后才会联网下载；依赖 akshare、tushare 与 TUSHARE_TOKEN。",
            "target_api": "/guided/dataset-onboarding/online-kline/start",
            "expert_link": "simple",
        },
        {
            "action_id": "create_imported_asset_manifest",
            "label_zh": "创建系统老师库数据清单",
            "type": "confirm_then_api",
            "enabled": task_type in {"imported_asset_demo", "artifact_review"},
            "reason_zh": "复用系统内已经训练好的老师库，并在 provenance 中保留来源。",
            "target_api": "/guided/dataset-onboarding/imported-assets",
            "expert_link": "library",
        },
        {
            "action_id": "generate_run_spec",
            "label_zh": "生成 Run Spec",
            "type": "confirm_then_api",
            "enabled": bool(can_generate_spec),
            "reason_zh": "已有足够上下文后，可生成英文 canonical run_spec / research_campaign。",
            "target_api": "/guided/run-wizard",
            "expert_link": "full-pipeline",
        },
        {
            "action_id": "launch_workflow",
            "label_zh": "启动 Workflow",
            "type": "confirm_then_api",
            "enabled": has_run_spec and not has_workflow_result,
            "reason_zh": "只有明确点击后才会启动 pipeline，不会由聊天回复暗中执行。",
            "target_api": "/pipeline/run",
            "expert_link": "full-pipeline",
        },
        {
            "action_id": "open_expert_monitor",
            "label_zh": "查看专业复盘",
            "type": "navigate",
            "enabled": True,
            "reason_zh": "查看同一个 run_id 的 Workflow Monitor / Teacher Zoo / Lesson Set / Provenance。",
            "target_api": "/console/run-monitor",
            "expert_link": "full-pipeline",
        },
        {
            "action_id": "score_signal",
            "label_zh": "开始信号评分",
            "type": "confirm_then_api",
            "enabled": (task_type == "scoring_only" and has_structured_signal) or task_type == "imported_asset_demo",
            "reason_zh": "上传或粘贴结构化 signal 后，可按当前模式生成校验、预览或 live scoring 结果；只有股票代码时会走单股在线评分流程。",
            "target_api": "/chat/action",
            "expert_link": "scoring",
        },
        {
            "action_id": "use_imported_demo_assets",
            "label_zh": "使用系统已有老师库",
            "type": "confirm_then_api",
            "enabled": task_type in {"imported_asset_demo", "artifact_review", "scoring_only"},
            "reason_zh": "快捷入口：使用已经训练好的 A 股老师库；用户新训练的老师库会单独隔离。",
            "target_api": "/guided/dataset-onboarding/imported-assets",
            "expert_link": "library",
        },
    ]


def _scoring_followup_actions(task_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not task_state.get("last_scoring_result_path"):
        return []
    imported = str(task_state.get("lesson_source") or "") == "imported_final_asset"
    model_called = bool(task_state.get("model_called", False))
    return [
        {
            "action_id": "open_scoring_detail",
            "label_zh": "查看评分详情",
            "type": "navigate",
            "enabled": True,
            "reason_zh": "打开 Expert Scoring Lab，查看同一个 run_id 下的 scoring artifact。",
            "target_api": "/score/live",
            "expert_link": "scoring",
        },
        {
            "action_id": "open_scoring_provenance",
            "label_zh": "查看来源追溯",
            "type": "navigate",
            "enabled": True,
            "reason_zh": "核对 model_called、lesson_source、teacher_source、fallback_reason 和 artifact path。",
            "target_api": "/console/provenance",
            "expert_link": "provenance",
        },
        {
            "action_id": "continue_upload_signal",
            "label_zh": "继续上传新信号",
            "type": "local_ui",
            "enabled": True,
            "reason_zh": "可以继续在 Simple Mode 粘贴 JSON 或上传结构化候选信号文件。",
            "target_api": "",
            "expert_link": "simple",
        },
        {
            "action_id": "batch_score_signal",
            "label_zh": "批量评分",
            "type": "local_ui",
            "enabled": True,
            "reason_zh": "上传 JSON array / CSV 多行 / Parquet 多行后生成 batch scoring artifact。",
            "target_api": "/chat/action",
            "expert_link": "simple",
        },
        {
            "action_id": "open_lesson_source",
            "label_zh": "查看 Teacher / Lesson 来源",
            "type": "navigate",
            "enabled": True,
            "reason_zh": "确认当前使用系统已有老师库，还是本次 workflow 生成的 final_lesson_set。",
            "target_api": "/console/lesson-set",
            "expert_link": "library",
        },
        {
            "action_id": "open_prompt_preview",
            "label_zh": "查看 prompt preview",
            "type": "navigate",
            "enabled": bool(task_state.get("prompt_preview_path")),
            "reason_zh": "prompt_only 会保存将要发送给 GPT-OSS 的 prompt；当前没有真实模型调用。",
            "target_api": "",
            "expert_link": "provenance",
        },
        {
            "action_id": "run_project_specific_pipeline",
            "label_zh": "运行自己的 full pipeline",
            "type": "confirm_then_api",
            "enabled": imported,
            "reason_zh": "当前使用系统已有老师库；如果要训练自己的老师库，需要先跑完整研究流程。",
            "target_api": "/guided/run-wizard",
            "expert_link": "full-pipeline",
        },
        {
            "action_id": "future_live_validation",
            "label_zh": "切换到 live scoring",
            "type": "disabled_info",
            "enabled": False,
            "reason_zh": "如果本地 GPT-OSS runtime 健康，可在 Simple Scoring 的 Mode 中选择 live；live 会真实调用模型。",
            "target_api": "/score/live",
            "expert_link": "scoring",
        },
    ]


def _boundary_warnings_zh(task_type: str, message: str) -> List[str]:
    warnings = [
        "如果你明确要求查看某只股票，我会尝试联网补齐最近 K 线；不会在后台无缘无故抓取数据。",
        "评分结果是研究辅助，不构成投资建议。",
        "如果使用的是系统已有老师库、用户本次训练老师库或发生回退，我会在结果来源里说明。",
    ]
    mentions = _data_mentions(message)
    if task_type == "scoring_only" and mentions["mentions_stock_code_only"]:
        warnings.insert(
            1,
            "我会默认拉取近 120 个交易日左右的 K 线，并在结果里同时给出近 60 日与近 120 日两个视角。",
        )
    return warnings


def _assistant_message_zh(task_state: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
    missing = task_state.get("missing_fields") or []
    enabled_actions = [row["label_zh"] for row in actions if row.get("enabled")]
    mentions = _data_mentions(str(task_state.get("user_message_original") or ""))
    lines = [
        f"我理解你现在想做的是：{task_state['task_type_zh']}。",
    ]
    pending_codes = list(task_state.get("pending_stock_codes") or [])
    if task_state.get("task_type") == "scoring_only" and pending_codes:
        code_text = ", ".join(map(str, pending_codes))
        lines.append(
            f"我已经识别到股票代码：{code_text}。我会先补齐最近 K 线，再给出综合分、近 60 日/120 日视角，以及四个老师模型的看法。"
        )
    elif task_state.get("task_type") == "scoring_only" and mentions["mentions_stock_code_only"]:
        lines.append(
            "我已经识别到股票代码，会尽量自动补齐行情并评分；如果数据源不可用，我会告诉你缺什么。"
        )
    elif task_state.get("workflow_preset") == DEMO_LOOP_PRESET_ID:
        library_name = str(task_state.get("teacher_library_display_name_zh") or "新的老师库").strip()
        lines.append(
            f"我会把这次设置为“{DEMO_LOOP_PRESET_ZH}”，训练完成后的老师库显示名为「{library_name}」。"
        )
        lines.append("你只需要上传一份 OHLCV CSV；我会自动检查数据、生成配置，并尝试启动外循环和内循环小演示。")
    if missing and not pending_codes:
        lines.append(f"还缺少这些关键信息：{', '.join(missing)}。")
    elif pending_codes:
        lines.append("你不需要手工整理因子，我会从 K 线自动生成需要的技术特征。")
    else:
        lines.append("信息基本够用，我会继续推进；如果不够，我会直接问你缺的那一项。")
    if enabled_actions:
        lines.append(f"建议下一步：{enabled_actions[0]}。")
    lines.append("结果会说明数据和经验规则来自哪里。")
    return "\n".join(lines)


def handle_chat_message(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    session_id: str,
    mode: str,
    message: str,
    attachments: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    text = _visible_user_text(message)
    session_id = session_id.strip() or f"chat-{uuid.uuid4().hex[:12]}"
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    ensure_contract_dirs(contract)
    chat_paths = _chat_paths(contract)
    chat_paths["root"].mkdir(parents=True, exist_ok=True)
    previous_task_state = _read_optional(chat_paths["task_state_json"]) or {}

    if (
        _is_confirmation_message(text)
        and previous_task_state.get("pending_natural_action") == "score_stock_code_live"
        and previous_task_state.get("pending_stock_codes")
    ):
        _append_user_message(
            chat_paths=chat_paths,
            session_id=session_id,
            text=text,
            task_state=previous_task_state,
            interpreted_as="confirm_score_stock_code_live",
        )
        codes = list(previous_task_state.get("pending_stock_codes") or [])
        return handle_chat_action(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
            session_id=session_id,
            action_id="score_stock_code_live",
            confirm=True,
            task_state_payload=previous_task_state,
            kline_params={
                "stock_codes": ",".join(map(str, codes)),
                "earliest_date": str(previous_task_state.get("pending_kline_earliest_date") or _default_recent_kline_earliest()),
                "adjust_type": str(previous_task_state.get("pending_adjust_type") or "qfq"),
                "update_indexes": False,
            },
            api_model="gpt-oss-20b",
        )

    guided = analyze_task_intake(
        user_request=text,
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    artifacts = _existing_artifacts(contract)
    task_type, confidence = _task_type_from_message(text, guided)
    stock_codes = _stock_codes_from_text(text)
    if _is_stock_review_request(text):
        task_type = "scoring_only"
        confidence = max(float(confidence or 0.0), 0.92)
    missing = _missing_fields(task_type, text, artifacts)
    actions = _recommended_actions(task_type, missing, artifacts)
    boundary_warnings = _boundary_warnings_zh(task_type, text)
    teacher_library_display_name_zh = _teacher_library_display_name_zh(text) or str(
        previous_task_state.get("teacher_library_display_name_zh") or ""
    ).strip()

    task_state = {
        "session_id": session_id,
        "profile": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "mode": mode or "simple",
        "task_type": task_type,
        "task_type_zh": TASK_TYPE_ZH.get(task_type, "待澄清任务"),
        "confidence": confidence,
        "language": _language(text),
        "user_message_original": text,
        "normalized_user_goal_zh": _normalized_goal_zh(task_type, text),
        "canonical_user_goal_en": _canonical_user_goal_en(task_type, text),
        "teacher_library_display_name_zh": teacher_library_display_name_zh,
        "translation_policy": "Chinese is used only for interaction and explanation. Canonical run specs, research campaigns, JSON keys, teacher_id, lesson_id, factor columns, file paths, and scoring prompts remain English.",
        "missing_fields": missing,
        "clarifying_questions_zh": _clarifying_questions_zh(task_type, missing),
        "dataset_requirements_zh": _requirements_zh(task_type),
        "current_stage": _stage_from_artifacts(artifacts),
        "allow_imported_fallback": allow_imported_fallback,
        "allow_demo_fallback": allow_demo_fallback,
        "fallback_used": False,
        "fallback_reason": "",
        "data_isolation_status": contract.get("data_isolation", {}),
        "artifacts": {
            "project_config_json": artifacts["project_config_json"]["path"],
            "dataset_manifest_json": artifacts["dataset_manifest_json"]["path"],
            "run_spec_json": artifacts["run_spec_json"]["path"],
            "research_campaign_json": artifacts["research_campaign_json"]["path"],
            "workflow_result_json": artifacts["workflow_result_json"]["path"],
        },
        "artifact_exists": {key: bool(row["exists"]) for key, row in artifacts.items()},
        "boundary_warnings_zh": boundary_warnings,
        "attachments": attachments or [],
        "updated_at": _now_iso(),
    }
    if task_type == "scoring_only" and stock_codes:
        task_state["pending_stock_codes"] = stock_codes[:1]
        task_state["pending_kline_earliest_date"] = _default_recent_kline_earliest()
        task_state["pending_adjust_type"] = "qfq"
        task_state["next_action"] = "score_stock_code_live"
        task_state["auto_executed_action"] = "score_stock_code_live"
        missing = [item for item in missing if item != "structured_signal_record_with_factor_columns"]
        task_state["missing_fields"] = missing
        actions.insert(0, _stock_code_live_action(stock_codes=stock_codes[:1]))
        _append_user_message(
            chat_paths=chat_paths,
            session_id=session_id,
            text=text,
            task_state=task_state,
            interpreted_as="auto_score_stock_code_live",
        )
        return handle_chat_action(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
            session_id=session_id,
            action_id="score_stock_code_live",
            confirm=True,
            task_state_payload=task_state,
            kline_params={
                "stock_codes": ",".join(map(str, stock_codes[:1])),
                "earliest_date": _default_recent_kline_earliest(),
                "adjust_type": "qfq",
                "update_indexes": False,
            },
            api_model="gpt-oss-20b",
        )
    if task_type == "full_research_pipeline" and _is_new_teacher_library_request(text):
        task_state["workflow_preset"] = DEMO_LOOP_PRESET_ID
        task_state["workflow_preset_zh"] = DEMO_LOOP_PRESET_ZH
        task_state["skip_final_scoring"] = True
        task_state["auto_generate_run_spec_after_upload"] = True
        task_state["auto_launch_workflow_after_upload"] = True
        task_state["fallback_policy_zh"] = "本次训练不混用系统已有老师库；如果小训练未产出可用老师，会明确提示失败或需要扩大数据/轮数。"
        task_state["next_action"] = "upload_dataset"
        task_state["missing_fields"] = [item for item in task_state.get("missing_fields", []) if item != "research_goal_for_run_spec"]
    assistant_message = _assistant_message_zh(task_state, actions)
    task_state["recommended_next_action_zh"] = next((row["label_zh"] for row in actions if row.get("enabled")), "继续补充任务信息")
    task_state["recommended_actions"] = actions

    previous_session = _read_optional(chat_paths["session_json"]) or {}
    session = {
        "session_id": session_id,
        "profile": profile_id,
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "created_at": previous_session.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "mode": mode or "simple",
        "message_count": int(previous_session.get("message_count") or 0) + 2,
        "chat_root": str(chat_paths["root"]),
    }
    write_json(chat_paths["session_json"], session)
    write_json(chat_paths["task_state_json"], task_state)

    with chat_paths["messages_jsonl"].open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": _now_iso(),
                    "session_id": session_id,
                    "role": "user",
                    "content_original": text,
                    "normalized_user_goal_zh": task_state["normalized_user_goal_zh"],
                    "canonical_user_goal_en": task_state["canonical_user_goal_en"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": _now_iso(),
                    "session_id": session_id,
                    "role": "assistant",
                    "content_zh": assistant_message,
                    "task_type": task_type,
                    "recommended_actions": actions,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    artifact_links = _artifact_links(contract, artifacts, chat_paths)
    return {
        "session": session,
        "task_state": task_state,
        "assistant_message_zh": assistant_message,
        "recommended_actions": actions,
        "artifact_links": artifact_links,
        "chat_paths": {key: str(path) for key, path in chat_paths.items() if key != "root"},
        "boundary_warnings_zh": boundary_warnings,
        "guided_task_intake": guided,
    }


def _manifest_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": manifest.get("source_type", ""),
        "dataset_id": manifest.get("dataset_id", ""),
        "rows": manifest.get("row_count", 0),
        "symbols": manifest.get("symbol_count", 0),
        "date_range": manifest.get("date_range", {}),
        "missing_columns": manifest.get("missing_columns", []),
        "required_columns": manifest.get("required_columns", []),
        "readiness": manifest.get("data_readiness", ""),
        "valid": bool(manifest.get("valid", False)),
        "full_pipeline_ready": bool(manifest.get("full_pipeline_ready", False)),
        "task_compatible": bool(manifest.get("task_compatible", True)),
        "manifest_path": str((manifest.get("generated_paths") or {}).get("dataset_manifest_json", "")),
        "data_isolation_status": manifest.get("data_isolation_status", {}),
        "warning_reasons": manifest.get("warning_reasons", []),
        "fail_reasons": manifest.get("fail_reasons", []),
    }


def _looks_like_ohlcv(manifest: Dict[str, Any]) -> bool:
    columns = set(manifest.get("columns") or [])
    return {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}.issubset(columns)


def _looks_like_signal_data(manifest: Dict[str, Any]) -> bool:
    columns = set(manifest.get("columns") or [])
    return {"signal_id", "date", "symbol", "signal_type"}.issubset(columns) or {"signal_id", "signal_date", "symbol", "signal_type"}.issubset(columns)


def _manifest_explanation_zh(manifest: Dict[str, Any], task_type: str) -> str:
    summary = _manifest_summary(manifest)
    valid_text = "可用" if summary["valid"] else "不可直接使用"
    missing = summary["missing_columns"] or []
    warnings = summary["warning_reasons"] or []
    failures = summary["fail_reasons"] or []
    can_run: List[str] = []
    cannot_run: List[str] = []
    if task_type == "full_research_pipeline":
        if summary["full_pipeline_ready"]:
            can_run.append("full_research_pipeline")
        else:
            can_run.append("generate_run_spec / planning artifacts")
            cannot_run.append("direct full_pipeline launch until clean K-line layout is ready")
    elif task_type == "scoring_only":
        if summary["valid"] and _looks_like_signal_data(manifest):
            can_run.append("scoring_only")
        else:
            cannot_run.append("scoring_only until structured signal factor columns are provided")
    elif task_type == "imported_asset_demo":
        can_run.append("imported_asset_demo / lightweight scoring")
    else:
        can_run.append("artifact_review")

    mismatch_note = ""
    if task_type == "scoring_only" and _looks_like_ohlcv(manifest) and not _looks_like_signal_data(manifest):
        mismatch_note = (
            "这份数据更像完整研究流程使用的 OHLCV 行情面板，不是已经构造好的候选信号。"
            "你可以选择跑完整流程，或者上传包含 signal_id、signal_date、symbol、signal_type 和结构化因子的候选信号。"
        )
    elif task_type == "full_research_pipeline" and _looks_like_signal_data(manifest) and not _looks_like_ohlcv(manifest):
        mismatch_note = "这份数据更像 scoring_only 的候选信号数据，不是训练 teacher 所需的 OHLCV 行情面板。"

    parts = [
        f"我已经生成 dataset_manifest.json。数据状态：{valid_text}；data_readiness={summary['readiness'] or '-'}。",
        f"行数={summary['rows']}，标的数={summary['symbols']}，日期范围={summary['date_range'].get('start', '-') or '-'} -> {summary['date_range'].get('end', '-') or '-'}。",
        f"manifest path: {summary['manifest_path'] or '-'}。",
    ]
    if missing:
        parts.append(f"缺少字段：{', '.join(map(str, missing))}。")
    if warnings:
        parts.append(f"Warnings：{'；'.join(map(str, warnings))}。")
    if failures:
        parts.append(f"Fail reasons：{'；'.join(map(str, failures))}。")
    if mismatch_note:
        parts.append(mismatch_note)
    parts.append(f"可以继续的任务：{', '.join(can_run) if can_run else '-'}。")
    if cannot_run:
        parts.append(f"暂时不能运行：{', '.join(cannot_run)}。")
    isolation = summary["data_isolation_status"] or {}
    parts.append(f"数据隔离：isolated_from_imported_assets={isolation.get('isolated_from_imported_assets', '-') }。")
    return "\n".join(parts)


def _task_intake_from_state(task_state: Dict[str, Any]) -> Dict[str, Any]:
    task_type = str(task_state.get("task_type") or "artifact_review")
    return {
        "task_type": task_type if task_type in {"full_research_pipeline", "scoring_only", "imported_asset_demo", "artifact_review"} else "full_research_pipeline",
        "confidence": float(task_state.get("confidence") or 0.0),
        "user_request": str(task_state.get("canonical_user_goal_en") or task_state.get("user_message_original") or ""),
        "summary": str(task_state.get("canonical_user_goal_en") or ""),
        "teacher_library_display_name_zh": str(task_state.get("teacher_library_display_name_zh") or ""),
        "clarifying_questions": list(task_state.get("clarifying_questions_en") or []),
        "limitations": list(task_state.get("boundary_warnings_zh") or []),
        "dataset_requirements": dataset_requirements(
            task_type if task_type in {"full_research_pipeline", "scoring_only", "imported_asset_demo", "artifact_review"} else "full_research_pipeline"
        ),
        "ready_for_run_spec": True,
        "recommended_next_step": "run_spec_wizard",
    }


def _load_task_state(contract: Dict[str, Any], session_id: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    paths = _chat_paths(contract)
    stored = _read_optional(paths["task_state_json"])
    if fallback:
        merged = dict(stored or {})
        merged.update(dict(fallback))
        return merged
    if isinstance(stored, dict) and stored:
        return stored
    return {"session_id": session_id, "task_type": "unclear", "confidence": 0.0}


def _persist_chat_action(
    *,
    contract: Dict[str, Any],
    session_id: str,
    action_id: str,
    task_state: Dict[str, Any],
    assistant_message_zh: str,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    paths = _chat_paths(contract)
    paths["root"].mkdir(parents=True, exist_ok=True)
    artifacts = _existing_artifacts(contract)
    task_state["recommended_actions"] = actions
    task_state["recommended_next_action_zh"] = next((row["label_zh"] for row in actions if row.get("enabled")), "继续补充任务信息")
    task_state["updated_at"] = _now_iso()

    previous_session = _read_optional(paths["session_json"]) or {}
    session = {
        "session_id": session_id or str(task_state.get("session_id") or f"chat-{uuid.uuid4().hex[:12]}"),
        "profile": contract["profile_id"],
        "project_id": contract["project_id"],
        "dataset_id": contract["dataset_id"],
        "run_id": contract["run_id"],
        "created_at": previous_session.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "mode": str(task_state.get("mode") or "simple"),
        "message_count": int(previous_session.get("message_count") or 0) + 1,
        "chat_root": str(paths["root"]),
    }
    write_json(paths["session_json"], session)
    write_json(paths["task_state_json"], task_state)
    with paths["messages_jsonl"].open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": _now_iso(),
                    "session_id": session["session_id"],
                    "role": "assistant",
                    "action_id": action_id,
                    "content_zh": assistant_message_zh,
                    "task_type": task_state.get("task_type", ""),
                    "recommended_actions": actions,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return {
        "session": session,
        "task_state": task_state,
        "assistant_message_zh": assistant_message_zh,
        "recommended_actions": actions,
        "artifact_links": _artifact_links(contract, artifacts, paths),
        "chat_paths": {key: str(path) for key, path in paths.items() if key != "root"},
    }


def _update_task_state_with_manifest(task_state: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    summary = _manifest_summary(manifest)
    task_state["dataset_manifest_path"] = summary["manifest_path"]
    task_state["data_readiness"] = summary["readiness"]
    task_state["missing_columns"] = summary["missing_columns"]
    task_state["dataset_summary"] = summary
    task_state["data_isolation_status"] = summary["data_isolation_status"]
    task_state["current_stage"] = "dataset_manifest_ready"
    artifact_exists = dict(task_state.get("artifact_exists") or {})
    artifact_exists["dataset_manifest_json"] = bool(summary["manifest_path"])
    task_state["artifact_exists"] = artifact_exists
    task_type = str(task_state.get("task_type") or "")
    if task_type == "full_research_pipeline":
        task_state["next_action"] = "generate_run_spec"
    elif task_type == "scoring_only" and summary["valid"]:
        task_state["next_action"] = "score_signal"
    elif task_type == "imported_asset_demo":
        task_state["next_action"] = "generate_run_spec"
    else:
        task_state["next_action"] = "open_expert_monitor"
    return task_state


def _update_task_state_with_scoring(task_state: Dict[str, Any], scoring: Dict[str, Any]) -> Dict[str, Any]:
    provenance = dict(scoring.get("scoring_provenance") or {})
    manifest = dict(scoring.get("signal_input_manifest") or {})
    paths = dict(scoring.get("artifact_paths") or {})
    task_state["current_stage"] = "scoring_artifact_ready"
    task_state["scoring_status"] = "completed" if manifest.get("valid") else "validation_failed"
    task_state["scoring_mode"] = str(scoring.get("mode") or "")
    task_state["model_called"] = bool(provenance.get("model_called", False))
    task_state["result_valid_for_research"] = bool(provenance.get("result_valid_for_research", False))
    task_state["scored_signal_count"] = int(manifest.get("record_count") or 0)
    task_state["lesson_source"] = str(provenance.get("lesson_source") or "")
    task_state["teacher_source"] = str(provenance.get("teacher_source") or "")
    task_state["fallback_used"] = bool(provenance.get("fallback_used", False))
    task_state["fallback_reason"] = str(provenance.get("fallback_reason") or "")
    task_state["last_score"] = scoring.get("total_score")
    task_state["last_score_60d"] = scoring.get("score_60d")
    task_state["last_score_120d"] = scoring.get("score_120d")
    task_state["last_scored_symbol"] = str(scoring.get("symbol") or "")
    task_state["last_signal_date"] = str(scoring.get("signal_date") or "")
    task_state["last_scoring_result_path"] = (
        paths.get("scoring_mock_result.json")
        or paths.get("scoring_recorded_replay.json")
        or paths.get("scoring_prompt_preview.json")
        or paths.get("scoring_dry_run_summary.json")
        or paths.get("simple_stock_code_live_result.json")
        or paths.get("live_saved_run_json")
        or paths.get("live_cache_json")
        or paths.get("scoring_provenance.json")
        or ""
    )
    task_state["last_scoring_mode"] = str(scoring.get("mode") or "")
    task_state["last_scoring_summary_zh"] = str(scoring.get("summary_zh") or "")
    task_state["prompt_preview_path"] = paths.get("scoring_prompt_preview.json", "")
    task_state["dry_run_summary_path"] = paths.get("scoring_dry_run_summary.json", "")
    task_state["signal_input_manifest_path"] = paths.get("signal_input_manifest.json", "")
    task_state["scoring_provenance_path"] = paths.get("scoring_provenance.json", "")
    task_state["next_action"] = "open_expert_monitor"
    task_state.pop("pending_natural_action", None)
    task_state.pop("pending_stock_codes", None)
    return task_state


def _run_workflow_from_spec(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    api_model: str,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    run_spec = read_json(run_spec_path(contract))
    return WorkflowRunner(profile_id=profile_id).run(
        mode=str(run_spec.get("mode") or "scoring_only"),
        research_goal=str(run_spec.get("research_goal") or ""),
        run_label=str(run_spec.get("run_id") or run_id),
        project_id=project_id,
        dataset_id=dataset_id,
        selection_json=str(run_spec.get("selection_json_hint") or ""),
        final_lesson_state_json=str(run_spec.get("final_lesson_state_json_hint") or ""),
        lesson_alias=str(run_spec.get("lesson_alias") or "alignment_seed0005"),
        api_model=str(run_spec.get("api_model") or api_model),
        allow_imported_fallback=bool(run_spec.get("allow_imported_fallback", allow_imported_fallback)),
        allow_demo_fallback=bool(run_spec.get("allow_demo_fallback", allow_demo_fallback)),
        data_dir=str(run_spec.get("data_dir") or ""),
        global_env=dict(run_spec.get("global_env") or {}),
        stage_args=dict(run_spec.get("stage_args") or {}),
        stage_env=dict(run_spec.get("stage_env") or {}),
        check=False,
        allow_manual_steps=True,
    )


def _workflow_launch_status_path(contract: Dict[str, Any]) -> Path:
    return Path(contract["workflow_root"]) / "workflow_launch_status.json"


def _write_workflow_launch_status(contract: Dict[str, Any], payload: Dict[str, Any]) -> str:
    path = _workflow_launch_status_path(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"updated_at": _now_iso(), **payload}
    write_json(path, row)
    return str(path)


def _launch_workflow_background(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    api_model: str,
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    status_path = _write_workflow_launch_status(
        contract,
        {
            "status": "running",
            "run_id": contract["run_id"],
            "message_zh": "workflow 已在后台启动。",
            "model_called": False,
            "external_api_called": False,
        },
    )

    def _target() -> None:
        try:
            result = _run_workflow_from_spec(
                profile_id=profile_id,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
                api_model=api_model,
            )
            _write_workflow_launch_status(
                contract,
                {
                    "status": str(result.get("status") or "completed"),
                    "run_id": contract["run_id"],
                    "workflow_result_json": str((result.get("workflow_result") or {}).get("workflow_result_json") or ""),
                    "message_zh": "workflow 后台执行已结束。",
                },
            )
        except Exception as exc:
            _write_workflow_launch_status(
                contract,
                {
                    "status": "failed",
                    "run_id": contract["run_id"],
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "message_zh": "workflow 后台执行失败，请进入专业模式查看错误。",
                },
            )

    thread = threading.Thread(target=_target, name=f"qa-workflow-{contract['run_id']}", daemon=True)
    thread.start()
    return {
        "status": "running",
        "background": True,
        "status_path": status_path,
        "thread_name": thread.name,
        "run_id": contract["run_id"],
    }
    run_spec = read_json(run_spec_path(contract))
    return WorkflowRunner(profile_id=profile_id).run(
        mode=str(run_spec.get("mode") or "scoring_only"),
        research_goal=str(run_spec.get("research_goal") or ""),
        run_label=str(run_spec.get("run_id") or run_id),
        project_id=project_id,
        dataset_id=dataset_id,
        selection_json=str(run_spec.get("selection_json_hint") or ""),
        final_lesson_state_json=str(run_spec.get("final_lesson_state_json_hint") or ""),
        lesson_alias=str(run_spec.get("lesson_alias") or "alignment_seed0005"),
        api_model=str(run_spec.get("api_model") or api_model),
        allow_imported_fallback=bool(run_spec.get("allow_imported_fallback", allow_imported_fallback)),
        allow_demo_fallback=bool(run_spec.get("allow_demo_fallback", allow_demo_fallback)),
        data_dir=str(run_spec.get("data_dir") or ""),
        global_env=dict(run_spec.get("global_env") or {}),
        stage_args=dict(run_spec.get("stage_args") or {}),
        stage_env=dict(run_spec.get("stage_env") or {}),
        check=False,
        allow_manual_steps=True,
    )


def handle_chat_action(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
    allow_imported_fallback: bool,
    allow_demo_fallback: bool,
    session_id: str,
    action_id: str,
    confirm: bool,
    task_state_payload: Dict[str, Any] | None = None,
    file_payload: Dict[str, Any] | None = None,
    kline_params: Dict[str, Any] | None = None,
    signal_record: Dict[str, Any] | None = None,
    scoring_payload: Dict[str, Any] | None = None,
    api_model: str = "gpt-oss-20b",
) -> Dict[str, Any]:
    contract = build_run_contract(
        profile_id=profile_id,
        project_id=project_id,
        dataset_id=dataset_id,
        run_id=run_id,
        allow_imported_fallback=allow_imported_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    ensure_contract_dirs(contract)
    task_state = _load_task_state(contract, session_id, task_state_payload)
    session_id = session_id.strip() or str(task_state.get("session_id") or f"chat-{uuid.uuid4().hex[:12]}")
    task_type = str(task_state.get("task_type") or "artifact_review")
    artifacts_before = _existing_artifacts(contract)
    assistant_message = "动作已执行。"
    extra_payload: Dict[str, Any] = {}

    if action_id == "upload_dataset":
        if task_type == "online_kline_download":
            raise ValueError("online_kline_download should use start_online_kline_download with stock codes and adjust type.")
        if task_type == "imported_asset_demo":
            manifest = build_imported_asset_manifest(
                profile_id=profile_id,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                task_type=task_type,
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
            )
        else:
            if not file_payload:
                raise ValueError("upload_dataset requires file_payload with filename, content, and content_encoding.")
            manifest = build_dataset_manifest(
                profile_id=profile_id,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                task_type=task_type,
                filename=str(file_payload.get("filename") or "uploaded_dataset.txt"),
                content=str(file_payload.get("content") or ""),
                content_encoding=str(file_payload.get("content_encoding") or "text"),
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
            )
        task_state = _update_task_state_with_manifest(task_state, manifest)
        assistant_message = _manifest_explanation_zh(manifest, task_type)
        extra_payload["dataset_manifest"] = manifest
        if (
            task_type == "full_research_pipeline"
            and bool(manifest.get("full_pipeline_ready", False))
            and str(task_state.get("workflow_preset") or manifest.get("workflow_preset") or "") == DEMO_LOOP_PRESET_ID
            and bool(task_state.get("auto_generate_run_spec_after_upload", True))
        ):
            bundle = create_guided_run_bundle(
                profile_id=profile_id,
                project_id=project_id,
                dataset_id=dataset_id,
                run_id=run_id,
                task_intake=_task_intake_from_state(task_state),
                dataset_manifest=manifest,
                allow_imported_fallback=allow_imported_fallback,
                allow_demo_fallback=allow_demo_fallback,
                api_model=api_model,
                teacher_library_display_name_zh=str(task_state.get("teacher_library_display_name_zh") or "").strip(),
            )
            task_state["current_stage"] = "run_spec_ready"
            task_state["run_spec_path"] = str(bundle.get("paths", {}).get("run_spec_json", ""))
            task_state["research_campaign_path"] = str(bundle.get("paths", {}).get("research_campaign_json", ""))
            task_state["launchable"] = bool(bundle.get("launchable", False))
            task_state["workflow_preset"] = DEMO_LOOP_PRESET_ID
            task_state["workflow_preset_zh"] = DEMO_LOOP_PRESET_ZH
            task_state["skip_final_scoring"] = True
            task_state["next_action"] = "launch_workflow" if bundle.get("launchable") else "open_expert_monitor"
            artifact_exists = dict(task_state.get("artifact_exists") or {})
            artifact_exists["run_spec_json"] = True
            artifact_exists["research_campaign_json"] = True
            task_state["artifact_exists"] = artifact_exists
            extra_payload["wizard_bundle"] = bundle
            assistant_message += (
                f"\n\n我已经按「{DEMO_LOOP_PRESET_ZH}」自动生成训练配置，老师库名称会显示为「"
                f"{task_state.get('teacher_library_display_name_zh') or '新的老师库'}」。"
                "\n本次训练不会混用系统已有老师库；如果小训练没有产出可用老师，会明确显示失败原因。"
            )
            if bool(bundle.get("launchable", False)) and bool(task_state.get("auto_launch_workflow_after_upload", True)):
                launch = _launch_workflow_background(
                    profile_id=profile_id,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    run_id=run_id,
                    allow_imported_fallback=allow_imported_fallback,
                    allow_demo_fallback=allow_demo_fallback,
                    api_model=api_model,
                )
                task_state["current_stage"] = "workflow_running"
                task_state["workflow_launch_status_path"] = str(launch.get("status_path") or "")
                task_state["next_action"] = "open_expert_monitor"
                extra_payload["workflow_launch"] = launch
                assistant_message += "\n\n我已经在后台启动内外循环小演示。你可以继续看页面，右侧进度和专业模式会读取真实阶段产物。"

    elif action_id in {"create_imported_asset_manifest", "use_imported_demo_assets"}:
        manifest = build_imported_asset_manifest(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            task_type=task_type if task_type in {"imported_asset_demo", "artifact_review", "scoring_only", "full_research_pipeline"} else "imported_asset_demo",
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
        )
        task_state = _update_task_state_with_manifest(task_state, manifest)
        assistant_message = _manifest_explanation_zh(manifest, task_type)
        extra_payload["dataset_manifest"] = manifest

    elif action_id == "start_online_kline_download":
        params = dict(kline_params or {})
        job = KlineDownloadJobManager().start_job(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            task_type="full_research_pipeline",
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
            stock_codes=str(params.get("stock_codes") or ""),
            earliest_date=str(params.get("earliest_date") or _default_recent_kline_earliest()),
            adjust_type=str(params.get("adjust_type") or "qfq"),
            full_refresh=bool(params.get("full_refresh", False)),
            update_indexes=bool(params.get("update_indexes", True)),
        )
        task_state["current_stage"] = "online_kline_job_started"
        task_state["kline_job_id"] = job.get("job_id", "")
        task_state["next_action"] = "open_expert_monitor"
        assistant_message = (
            f"在线 K 线下载任务已启动：job_id={job.get('job_id', '-')}。\n"
            "这一步是显式联网动作，依赖服务器网络、akshare、tushare 和 TUSHARE_TOKEN。\n"
            "普通模式默认使用近 120 交易日左右的 K 线，并在后续评分解释里同时呈现近 60 日与近 120 日两个窗口。"
        )
        extra_payload["kline_job"] = job

    elif action_id == "sync_dataset_manifest":
        manifest_path = dataset_manifest_path(contract)
        if not manifest_path.exists():
            raise FileNotFoundError(f"dataset_manifest not found: {manifest_path}")
        manifest = read_json(manifest_path)
        task_state = _update_task_state_with_manifest(task_state, dict(manifest))
        assistant_message = _manifest_explanation_zh(dict(manifest), str(task_state.get("task_type") or "full_research_pipeline"))
        extra_payload["dataset_manifest"] = manifest

    elif action_id == "generate_run_spec":
        manifest_path = dataset_manifest_path(contract)
        manifest = read_json(manifest_path) if manifest_path.exists() else None
        bundle = create_guided_run_bundle(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            task_intake=_task_intake_from_state(task_state),
            dataset_manifest=manifest,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
            api_model=api_model,
            teacher_library_display_name_zh=str(task_state.get("teacher_library_display_name_zh") or "").strip(),
        )
        task_state["current_stage"] = "run_spec_ready"
        task_state["run_spec_path"] = str(bundle.get("paths", {}).get("run_spec_json", ""))
        task_state["research_campaign_path"] = str(bundle.get("paths", {}).get("research_campaign_json", ""))
        task_state["launchable"] = bool(bundle.get("launchable", False))
        task_state["next_action"] = "launch_workflow" if bundle.get("launchable") else "open_expert_monitor"
        artifact_exists = dict(task_state.get("artifact_exists") or {})
        artifact_exists["run_spec_json"] = True
        artifact_exists["research_campaign_json"] = True
        task_state["artifact_exists"] = artifact_exists
        assistant_message = (
            "Run Spec 和 research_campaign 已生成，仍保持英文 canonical。\n"
            f"launchable={bool(bundle.get('launchable', False))}。\n"
            f"run_spec path: {task_state['run_spec_path']}"
        )
        extra_payload["wizard_bundle"] = bundle

    elif action_id == "launch_workflow":
        if not confirm:
            raise ValueError("launch_workflow requires explicit confirm=true.")
        launch = _launch_workflow_background(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
            api_model=api_model,
        )
        task_state["current_stage"] = "workflow_running"
        task_state["workflow_launch_status_path"] = str(launch.get("status_path") or "")
        task_state["next_action"] = "open_expert_monitor"
        assistant_message = "Workflow 已在后台启动。请在专业复盘中查看真实阶段状态。"
        extra_payload["workflow_launch"] = launch

    elif action_id == "score_signal":
        payload = dict(scoring_payload or {})
        scoring = run_simple_scoring(
            profile_id=profile_id,
            contract=contract,
            scoring_payload=payload,
            file_payload=file_payload,
            signal_record=signal_record,
        )
        task_state = _update_task_state_with_scoring(task_state, scoring)
        assistant_message = (
            str(scoring.get("summary_zh") or "信号评分流程已完成。")
            + "\n\n请注意：这不是投资建议。本阶段不会启动本地 GPT-OSS，也不会调用外部 API；"
            "prompt_only / dry_run / mock / 历史样本复核都不能作为真实研究评分结论。"
        )
        extra_payload["scoring_result"] = scoring

    elif action_id == "score_stock_code_live":
        params = dict(kline_params or {})
        codes = str(params.get("stock_codes") or ",".join(map(str, task_state.get("pending_stock_codes") or []))).strip()
        try:
            scoring = score_stock_code_live(
                profile_id=profile_id,
                contract=contract,
                stock_codes=codes,
                earliest_date=str(params.get("earliest_date") or task_state.get("pending_kline_earliest_date") or _default_recent_kline_earliest()),
                adjust_type=str(params.get("adjust_type") or task_state.get("pending_adjust_type") or "qfq"),
                update_indexes=bool(params.get("update_indexes", False)),
                full_refresh=bool(params.get("full_refresh", False)),
                prompt_only=bool(params.get("prompt_only", False)),
            )
        except Exception as exc:
            task_state["task_type"] = "scoring_only"
            task_state["task_type_zh"] = TASK_TYPE_ZH["scoring_only"]
            task_state["current_stage"] = "stock_scoring_failed"
            task_state["scoring_status"] = "failed"
            task_state["last_scored_symbol"] = codes
            task_state["fallback_used"] = False
            task_state["fallback_reason"] = ""
            task_state["model_called"] = "Live model response" in str(exc) or "model" in str(exc).lower()
            task_state["next_action"] = "retry_or_upload_signal"
            task_state.pop("pending_natural_action", None)
            task_state.pop("pending_stock_codes", None)
            assistant_message = (
                f"我刚才尝试查看 {codes or '这只股票'}，但这次没有成功。\n"
                "可能原因有三类：1）行情源暂时取不到；2）服务器网络、akshare、tushare 或 TUSHARE_TOKEN 不可用；3）本地 GPT-OSS 返回了空内容或非标准 JSON。\n"
                "如果你想继续，我建议先检查模型 runtime 和 Tushare token；也可以上传这只股票最近 120 个交易日左右的 K 线文件，我会用同一套老师模型继续评分。"
            )
            extra_payload["scoring_error"] = {
                "symbol": codes,
                "error_message": str(exc),
                "error_type": exc.__class__.__name__,
                "user_facing_summary_zh": assistant_message,
            }
            scoring = {}
        if not scoring:
            pass
        else:
            task_state["task_type"] = "scoring_only"
            task_state["task_type_zh"] = TASK_TYPE_ZH["scoring_only"]
            task_state = _update_task_state_with_scoring(task_state, scoring)
            teacher_lines = _format_teacher_score_lines(scoring)
            feature_lines = _format_feature_diagnostic_lines(scoring)
            trading_reference = str(scoring.get("trading_reference_zh") or "").strip()
            signal_date = str(scoring.get("signal_date") or "").strip()
            core_reason = _friendly_stock_reason_zh(scoring)
            reference_text = trading_reference or core_reason
            assistant_message = (
                f"我看完了 {scoring.get('symbol', '')}（信号日 {signal_date or '-'}，默认前复权 qfq，近 120 交易日上下文）。\n\n"
                f"综合评分：{scoring.get('total_score', '-')} / 100\n"
                f"近 60 日评分：{scoring.get('score_60d', '-')} / 100\n"
                f"近 120 日评分：{scoring.get('score_120d', '-')} / 100\n\n"
                f"核心判断：{core_reason}\n\n"
                "技术结构：\n"
                + ("\n".join(feature_lines) if feature_lines else "- 暂未生成足够的技术因子诊断。")
                + "\n\n四个老师模型分项：\n"
                + ("\n".join(teacher_lines) if teacher_lines else "- 模型没有返回 teacher-level breakdown。")
                + f"\n\n参考意见：{reference_text}"
                + "\n\n这只是研究辅助评分，不构成投资建议。"
            )
            extra_payload["scoring_result"] = scoring

    elif action_id == "open_expert_monitor":
        task_state["next_action"] = "open_expert_monitor"
        assistant_message = "已准备跳转专业模式。专业模式会读取同一个 project_id / dataset_id / run_id 下的真实 artifact。"

    else:
        raise ValueError(f"Unsupported chat action_id: {action_id}")

    artifacts_after = _existing_artifacts(contract)
    missing = _missing_fields(str(task_state.get("task_type") or "artifact_review"), str(task_state.get("user_message_original") or ""), artifacts_after)
    if action_id == "score_stock_code_live":
        missing = []
    actions = _recommended_actions(str(task_state.get("task_type") or "artifact_review"), missing, artifacts_after)
    actions.extend(_scoring_followup_actions(task_state))
    task_state["missing_fields"] = missing
    task_state["artifact_exists"] = {key: bool(row["exists"]) for key, row in artifacts_after.items()}
    response = _persist_chat_action(
        contract=contract,
        session_id=session_id,
        action_id=action_id,
        task_state=task_state,
        assistant_message_zh=assistant_message,
        actions=actions,
    )
    response.update(extra_payload)
    response["previous_artifact_exists"] = {key: bool(row["exists"]) for key, row in artifacts_before.items()}
    return response


def _normalize_timeline_status(raw_status: str, *, mode: str, relevant_modes: List[str], has_workflow: bool) -> str:
    raw = str(raw_status or "").strip().lower()
    if mode and mode not in relevant_modes:
        return "skipped"
    if raw in {"completed", "done", "success"}:
        return "completed"
    if raw in {"failed", "error"}:
        return "failed"
    if raw in {"running", "in_progress"}:
        return "running"
    if raw in {"manual_pending", "planned", "partial", "queued"}:
        return "pending"
    if not has_workflow:
        return "pending" if "scoring_only" in relevant_modes or "full_pipeline" in relevant_modes else "not_started"
    return "not_started"


def _extract_payload_artifacts(payload: Dict[str, Any]) -> List[str]:
    artifacts: List[str] = []
    for key in ["artifact_json", "suite_summary_json", "selected_spec_json", "report_dir", "final_lesson_artifact_json"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            artifacts.append(value.strip())
    agent_summary = payload.get("agent_summary")
    if isinstance(agent_summary, dict):
        artifacts.extend(_extract_payload_artifacts(agent_summary))
    inner_payload = payload.get("payload")
    if isinstance(inner_payload, dict):
        artifacts.extend(_extract_payload_artifacts(inner_payload))
        for nested_key in ["final_lesson_state_json", "selection_json", "workflow_result_json"]:
            nested = inner_payload.get(nested_key)
            if isinstance(nested, str) and nested.strip():
                artifacts.append(nested.strip())
    return list(dict.fromkeys(artifacts))


def _monitor_node_lookup(run_monitor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for node in run_monitor.get("nodes") or []:
        row = dict(node)
        lookup[str(row.get("node_id") or "")] = row
        lookup[str(row.get("label") or "")] = row
    return lookup


def _first_node_for_agent(agent_def: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for stage_id in agent_def.get("stage_ids") or []:
        if stage_id in lookup:
            return dict(lookup[stage_id])
    return {}


def build_agent_timeline_status_from_views(
    *,
    run_monitor: Dict[str, Any],
    provenance: Dict[str, Any],
    lesson_set: Dict[str, Any],
    ui_mock: bool = False,
) -> Dict[str, Any]:
    contract = dict(run_monitor.get("contract") or provenance.get("contract") or {})
    mode = str(contract.get("mode") or provenance.get("research_campaign", {}).get("mode") or run_monitor.get("mode") or "").strip()
    workflow_status = str(run_monitor.get("workflow_status") or provenance.get("workflow_status") or "spec_only").strip() or "spec_only"
    workflow_result_json = str(run_monitor.get("workflow_result_json") or provenance.get("workflow_result_json") or "").strip()
    has_workflow = workflow_status not in {"", "spec_only"} or _safe_path_exists(workflow_result_json)
    lookup = _monitor_node_lookup(run_monitor)
    selection_summary = dict(provenance.get("teacher_selection_summary") or {})
    selection_fallback_reason = str(selection_summary.get("fallback_reason") or "").strip()
    selection_source = str(selection_summary.get("resolution_source") or "").strip()
    lesson_source = str(lesson_set.get("final_lesson_source") or "").strip()
    lesson_path = str(lesson_set.get("final_lesson_state_json") or "").strip()
    artifact_files = [str(path) for path in provenance.get("artifact_files") or []]

    timeline_nodes: List[Dict[str, Any]] = []
    for idx, agent_def in enumerate(AGENT_TIMELINE_DEFS, start=1):
        base_node = _first_node_for_agent(agent_def, lookup)
        payload = dict(base_node.get("payload") or {})
        raw_status = str(base_node.get("status") or "").strip()
        status = _normalize_timeline_status(
            raw_status,
            mode=mode,
            relevant_modes=list(agent_def.get("relevant_modes") or []),
            has_workflow=has_workflow,
        )
        if agent_def["agent_name"] == "Planner Agent" and _safe_path_exists(str(provenance.get("run_spec_json") or "")):
            status = "completed"
        if agent_def["agent_name"] in {"Hypothesis Agent", "FactorDesign Agent"} and mode == "scoring_only":
            status = "skipped"
        if agent_def["agent_name"] == "TeacherSelection Agent" and selection_summary:
            status = "fallback" if selection_fallback_reason else "completed"
        if agent_def["agent_name"] == "Apprentice Agent" and lesson_path:
            status = "fallback" if lesson_source == "imported_final_asset" else "completed"
        if agent_def["agent_name"] == "SignalScoring Agent" and mode == "scoring_only" and workflow_status == "completed":
            status = "completed"

        output_artifacts = _extract_payload_artifacts(base_node)
        if agent_def["agent_name"] == "Planner Agent":
            output_artifacts.extend([str(provenance.get("run_spec_json") or ""), str(provenance.get("research_campaign_json") or "")])
        if agent_def["agent_name"] == "TeacherSelection Agent":
            output_artifacts.append(str(provenance.get("teacher_selection_summary_json") or ""))
        if agent_def["agent_name"] == "Apprentice Agent":
            output_artifacts.extend([lesson_path, str(lesson_set.get("suite_summary_json") or "")])
        output_artifacts = [path for path in dict.fromkeys(output_artifacts) if path]
        summary_path = output_artifacts[0] if output_artifacts else ""
        if not summary_path:
            for path in artifact_files:
                lowered = path.lower()
                if agent_def["agent_name"].lower().split()[0] in lowered:
                    summary_path = path
                    break

        fallback_used = status == "fallback"
        fallback_reason = ""
        fallback_source = ""
        if agent_def["agent_name"] == "TeacherSelection Agent" and selection_fallback_reason:
            fallback_used = True
            fallback_reason = selection_fallback_reason
            fallback_source = selection_source or "unknown"
        if agent_def["agent_name"] == "Apprentice Agent" and lesson_source == "imported_final_asset":
            fallback_used = True
            fallback_reason = "当前 run 尚未生成自己的 final_lesson_set，因此使用系统已有老师库的最终经验规则集。"
            fallback_source = lesson_source

        timeline_nodes.append(
            {
                "node_id": f"agent_{idx:02d}",
                "agent_name": agent_def["agent_name"],
                "mapped_stage": agent_def["mapped_stage"],
                "status": status,
                "raw_status": raw_status or "暂无",
                "input_artifacts": [],
                "output_artifacts": output_artifacts,
                "summary_path": summary_path,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "fallback_source": fallback_source,
                "error_message": str(base_node.get("payload", {}).get("error") or base_node.get("error") or "暂无"),
                "started_at": str(base_node.get("started_at") or payload.get("started_at") or "暂无"),
                "finished_at": str(base_node.get("finished_at") or payload.get("finished_at") or "暂无"),
                "expert_link": agent_def["expert_workspace"],
                "source_note": "stage_level_mapping",
            }
        )

    running = [node for node in timeline_nodes if node["status"] == "running"]
    failed = [node for node in timeline_nodes if node["status"] == "failed"]
    completed = [node for node in timeline_nodes if node["status"] == "completed"]
    fallback_nodes = [node for node in timeline_nodes if node["fallback_used"] or node["status"] == "fallback"]
    latest_artifact = next((node["summary_path"] for node in reversed(timeline_nodes) if node.get("summary_path")), "")
    current_stage = running[0]["agent_name"] if running else (failed[0]["agent_name"] if failed else (completed[-1]["agent_name"] if completed else "暂无"))
    failed_stage = failed[0]["agent_name"] if failed else ""
    workflow_terminal = workflow_status in {"completed", "failed", "cancelled", "partial"}
    if workflow_status == "completed":
        next_action = "查看 Teacher Zoo / Final Lesson Set，或上传新信号进行 scoring"
    elif workflow_status == "failed":
        next_action = "查看专业日志，修改 run_spec 或重新上传数据"
    elif workflow_status in {"running", "pending", "queued"}:
        next_action = "等待当前 workflow 阶段完成"
    else:
        next_action = "继续补齐数据或生成 run_spec"

    return {
        "project_id": str(contract.get("project_id", "")),
        "dataset_id": str(contract.get("dataset_id", "")),
        "run_id": str(contract.get("run_id", "")),
        "mode": mode,
        "workflow_status": workflow_status,
        "workflow_terminal": workflow_terminal,
        "stage_level_mapping": True,
        "stage_level_note_zh": "当前显示的是阶段级状态，由 workflow stage 映射到 Agent，不代表每个 Agent 都有独立实时事件。",
        "timeline_nodes": timeline_nodes,
        "task_card": {
            "run_id": str(contract.get("run_id", "")),
            "task_type": str(contract.get("task_type", "")),
            "current_stage": current_stage,
            "completed_stages": [node["agent_name"] for node in completed],
            "failed_stage": failed_stage,
            "fallback_used": bool(fallback_nodes),
            "fallback_reason": "；".join([node["fallback_reason"] for node in fallback_nodes if node.get("fallback_reason")]) or "",
            "fallback_source": "；".join([node["fallback_source"] for node in fallback_nodes if node.get("fallback_source")]) or "",
            "latest_artifact": latest_artifact,
            "next_action": next_action,
            "workflow_status": workflow_status,
        },
        "post_workflow_actions": {
            "completed": [
                {"label_zh": "查看 Teacher Zoo", "expert_link": "library"},
                {"label_zh": "查看 Final Lesson Set", "expert_link": "library"},
                {"label_zh": "上传新信号进行 scoring", "expert_link": "scoring"},
                {"label_zh": "查看专业模式 provenance", "expert_link": "provenance"},
            ],
            "failed": [
                {"label_zh": "查看专业日志", "expert_link": "full-pipeline"},
                {"label_zh": "打开 Audit Trail", "expert_link": "provenance"},
                {"label_zh": "返回修改 run_spec", "expert_link": "advanced"},
                {"label_zh": "重新上传数据", "expert_link": "simple"},
                {"label_zh": "重新启动 workflow", "expert_link": "full-pipeline"},
            ],
        },
        "ui_mock": bool(ui_mock),
    }


def build_chat_run_status(
    *,
    profile_id: str,
    project_id: str,
    dataset_id: str,
    run_id: str,
) -> Dict[str, Any]:
    run_monitor = build_run_monitor(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    provenance = build_provenance_view(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    lesson_set = build_lesson_set_view(project_id=project_id, dataset_id=dataset_id, run_id=run_id)
    return build_agent_timeline_status_from_views(
        run_monitor=run_monitor,
        provenance=provenance,
        lesson_set=lesson_set,
        ui_mock=False,
    )
