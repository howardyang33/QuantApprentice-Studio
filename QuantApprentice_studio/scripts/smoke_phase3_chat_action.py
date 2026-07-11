from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

from quant_apprentice_studio.api.app import create_app


def post(client: TestClient, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post(path, json=payload)
    content_type = response.headers.get("content-type", "")
    body: Any = response.json() if content_type.startswith("application/json") else response.text
    return {"status_code": response.status_code, "json": body}


def main() -> None:
    client = TestClient(create_app())
    report_dir = Path("test_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase3_chat_action_smoke_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"

    csv_ohlcv = (
        "symbol,date,open,high,low,close,volume,amount\n"
        "000001,2025-01-02,10,10.5,9.8,10.2,100000,1020000\n"
        "000002,2025-01-02,20,21,19.8,20.5,200000,4100000\n"
        "000001,2025-01-03,10.2,10.8,10.1,10.6,120000,1272000\n"
    )

    base = {
        "profile": "gpt_oss_20b_final",
        "dataset_id": "dataset-a",
        "allow_imported_fallback": True,
        "allow_demo_fallback": False,
    }
    cases: list[Dict[str, Any]] = []

    msg1 = {
        **base,
        "project_id": "phase3-smoke-full",
        "run_id": "run-upload-001",
        "message": "我有一批A股OHLCV CSV，想跑完整研究流程，不要默认联网抓行情。",
    }
    chat1 = post(client, "/chat/message", msg1)
    session1 = chat1["json"]["session"]["session_id"]
    upload1 = post(
        client,
        "/chat/action",
        {
            **base,
            "project_id": "phase3-smoke-full",
            "run_id": "run-upload-001",
            "session_id": session1,
            "action_id": "upload_dataset",
            "confirm": True,
            "task_state": chat1["json"]["task_state"],
            "file_payload": {"filename": "mini_ohlcv.csv", "content": csv_ohlcv, "content_encoding": "text"},
        },
    )
    run_spec1 = post(
        client,
        "/chat/action",
        {
            **base,
            "project_id": "phase3-smoke-full",
            "run_id": "run-upload-001",
            "session_id": session1,
            "action_id": "generate_run_spec",
            "confirm": True,
            "task_state": upload1["json"]["task_state"],
        },
    )
    cases.append(
        {
            "name": "full_research_upload_manifest_runspec",
            "passed": (
                chat1["status_code"] == 200
                and upload1["status_code"] == 200
                and run_spec1["status_code"] == 200
                and upload1["json"]["dataset_manifest"]["row_count"] == 3
                and bool(run_spec1["json"]["task_state"].get("run_spec_path"))
            ),
            "input": msg1,
            "task_type": chat1["json"]["task_state"]["task_type"],
            "recommended_actions_after_chat": [a["action_id"] for a in chat1["json"]["recommended_actions"]],
            "recommended_actions_after_upload": [a["action_id"] for a in upload1["json"]["recommended_actions"]],
            "manifest_path": upload1["json"]["task_state"].get("dataset_manifest_path"),
            "run_spec_path": run_spec1["json"]["task_state"].get("run_spec_path"),
            "chat_paths": chat1["json"].get("chat_paths"),
        }
    )

    msg2 = {
        **base,
        "project_id": "phase3-smoke-mismatch",
        "run_id": "run-mismatch-001",
        "message": "我想给一批信号打分，我先上传这个CSV。",
    }
    chat2 = post(client, "/chat/message", msg2)
    session2 = chat2["json"]["session"]["session_id"]
    upload2 = post(
        client,
        "/chat/action",
        {
            **base,
            "project_id": "phase3-smoke-mismatch",
            "run_id": "run-mismatch-001",
            "session_id": session2,
            "action_id": "upload_dataset",
            "confirm": True,
            "task_state": chat2["json"]["task_state"],
            "file_payload": {"filename": "wrong_for_scoring_ohlcv.csv", "content": csv_ohlcv, "content_encoding": "text"},
        },
    )
    mismatch_text = upload2["json"].get("assistant_message_zh", "") if upload2["status_code"] == 200 else ""
    cases.append(
        {
            "name": "scoring_upload_ohlcv_mismatch_guard",
            "passed": upload2["status_code"] == 200 and "更像完整研究流程" in mismatch_text,
            "input": msg2,
            "task_type": chat2["json"]["task_state"]["task_type"],
            "recommended_actions_after_chat": [a["action_id"] for a in chat2["json"]["recommended_actions"]],
            "manifest_path": upload2["json"]["task_state"].get("dataset_manifest_path") if upload2["status_code"] == 200 else "",
            "assistant_excerpt": mismatch_text[:300],
        }
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 3 Chat File Upload + Action Execution",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "passed": report["passed"], "cases": cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
