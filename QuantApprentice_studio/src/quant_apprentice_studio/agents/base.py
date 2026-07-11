from __future__ import annotations

from dataclasses import dataclass

from ..registry import StudioRegistry


@dataclass
class BaseAgent:
    registry: StudioRegistry

    def __post_init__(self) -> None:
        self.registry.ensure_bootstrapped()
