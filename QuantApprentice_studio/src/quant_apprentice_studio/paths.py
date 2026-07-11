from __future__ import annotations

import os
import sys
from pathlib import Path


def studio_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_root() -> Path:
    value = os.environ.get("QA_STUDIO_CLEAN_ROOT", "")
    if value.strip():
        return Path(value).expanduser().resolve()
    return (studio_root().parent / "QuantApprentice_clean").resolve()


def original_quant_root() -> Path:
    value = os.environ.get("QA_STUDIO_ORIGINAL_QUANT_ROOT", "")
    if value.strip():
        return Path(value).expanduser().resolve()
    return (studio_root().parent / "quant").resolve()


def config_root() -> Path:
    return studio_root() / "configs"


def docs_root() -> Path:
    return studio_root() / "docs"


def import_root(profile_id: str | None = None) -> Path:
    root = studio_root() / "imports"
    return root if profile_id is None else root / profile_id


def provenance_root() -> Path:
    return studio_root() / "provenance"


def ensure_clean_repo_on_path() -> Path:
    root = clean_root()
    if not root.exists():
        raise FileNotFoundError(f"QuantApprentice_clean not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
