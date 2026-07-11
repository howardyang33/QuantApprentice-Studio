"""信号生成与打分模块"""
from .kd_strategy import generate_kd13_signal, construct_samples, process_all_stocks
from .scoring import (
    compute_train_thresholds,
    assign_quintile_by_thresholds,
    score_signals,
    bucket_signals_by_score,
)

__all__ = [
    "generate_kd13_signal",
    "construct_samples",
    "process_all_stocks",
    "compute_train_thresholds",
    "assign_quintile_by_thresholds",
    "score_signals",
    "bucket_signals_by_score",
]
