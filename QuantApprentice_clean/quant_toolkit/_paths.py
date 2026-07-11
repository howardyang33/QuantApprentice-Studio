"""Shared path helpers for the clean QuantApprentice package."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return Path(default).expanduser().resolve()
    return Path(str(value)).expanduser().resolve()
