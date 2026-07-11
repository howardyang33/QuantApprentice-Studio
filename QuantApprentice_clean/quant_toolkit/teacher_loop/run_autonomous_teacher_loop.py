#!/usr/bin/env python3
"""Canonical entrypoint for launching one autonomous teacher-loop round."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from quant_toolkit.teacher_loop.loop import REPORT_ROOT, launch_until_target


def append_error(message: str) -> None:
    path = REPORT_ROOT / "error_log.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def main() -> None:
    max_new_rounds = int(os.environ.get("TEACHER_LOOP_MAX_ROUNDS", "1"))
    results = launch_until_target(max_new_rounds=max_new_rounds)
    print(json.dumps({"launched_rounds": len(results), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        append_error(traceback.format_exc())
        raise
