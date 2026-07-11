from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .paths import config_root, import_root, original_quant_root, provenance_root
from .provenance import read_json


class StudioRegistry:
    def __init__(self, profile_id: str = "gpt_oss_20b_final") -> None:
        self.profile_id = profile_id

    @property
    def profile_path(self) -> Path:
        return config_root() / "profiles" / f"{self.profile_id}.json"

    @property
    def manifest_path(self) -> Path:
        return provenance_root() / f"{self.profile_id}_import_manifest.json"

    @property
    def runtime_catalog_path(self) -> Path:
        return provenance_root() / f"{self.profile_id}_runtime_catalog.json"

    @property
    def imported_profile_root(self) -> Path:
        return import_root(self.profile_id)

    def load_profile(self) -> Dict[str, Any]:
        return read_json(self.profile_path)

    def load_manifest(self) -> Dict[str, Any]:
        return read_json(self.manifest_path)

    def load_runtime_catalog(self) -> Dict[str, Any]:
        return read_json(self.runtime_catalog_path)

    def ensure_bootstrapped(self) -> None:
        if not self.runtime_catalog_path.exists():
            raise FileNotFoundError(
                f"Runtime catalog not found for profile {self.profile_id}. Run the bootstrap command first."
            )

    def imported_path_for_source(self, source_path: str | Path) -> Path:
        source = Path(source_path).expanduser().resolve()
        quant_root = original_quant_root()
        return self.imported_profile_root / "original_quant" / source.relative_to(quant_root)
