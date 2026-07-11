from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import config_root


def _secret_file_candidates() -> list[Path]:
    root = config_root()
    return [
        root / "studio_secrets.json",
        root / "secrets.json",
    ]


def resolve_secret(name: str, default: str = "") -> str:
    env_value = os.environ.get(str(name).strip(), "")
    if env_value.strip():
        return env_value.strip()
    for path in _secret_file_candidates():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            value = str(payload.get(name, "") or "").strip()
            if value:
                return value
    return str(default or "").strip()
