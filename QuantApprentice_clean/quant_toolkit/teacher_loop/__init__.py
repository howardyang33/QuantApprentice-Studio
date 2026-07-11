"""Autonomous multi-round teacher-construction loop utilities."""

from .registry import (
    ALL_REGISTERED_FEATURES,
    BASE_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    build_feature_registry,
    ensure_feature_registry,
)
from .zoo import build_teacher_zoo_index, ensure_teacher_loop_indexes

__all__ = [
    "ALL_REGISTERED_FEATURES",
    "BASE_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
    "build_feature_registry",
    "ensure_feature_registry",
    "build_teacher_zoo_index",
    "ensure_teacher_loop_indexes",
]
