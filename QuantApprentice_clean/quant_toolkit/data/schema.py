"""
数据Schema定义与验证

定义所有数据结构的字段名、类型和约束，用于运行时验证。
"""

from typing import List, Set
import pandas as pd
import numpy as np


class DailyDataSchema:
    """日线数据Schema"""

    REQUIRED_COLUMNS: List[str] = ["date", "open", "close", "high", "low"]
    OPTIONAL_COLUMNS: List[str] = ["volume", "amount", "turnover", "high_limit", "low_limit", "pct_chg"]
    ALL_COLUMNS: List[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    NUMERIC_COLUMNS: List[str] = ["open", "close", "high", "low", "volume", "amount", "turnover", "high_limit", "low_limit", "pct_chg"]
    DATETIME_COLUMNS: List[str] = ["date"]

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        """验证DataFrame是否符合日线数据Schema"""
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        return True


class SignalSchema:
    """信号数据Schema"""

    COLUMNS: List[str] = [
        "symbol",
        "signal_date",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "holding_days",
        "return_20d",
    ]

    # 标签列（未来数据，仅用于训练）
    LABEL_COLUMNS: List[str] = ["exit_price", "return_20d"]

    # 元数据列（非特征）
    META_COLUMNS: List[str] = ["symbol", "signal_date", "entry_date", "exit_date", "entry_price", "exit_price", "holding_days", "return_20d"]


class SampleSchema:
    """样本数据Schema（信号+特征）"""

    @classmethod
    def get_feature_columns(cls, df: pd.DataFrame) -> List[str]:
        """从样本DataFrame中识别特征列（排除元数据列）"""
        exclude = set(SignalSchema.META_COLUMNS)
        return [c for c in df.columns if c not in exclude]


class IndexDataSchema:
    """指数数据Schema"""

    REQUIRED_COLUMNS: List[str] = ["date", "open", "close", "high", "low"]
    OPTIONAL_COLUMNS: List[str] = ["volume", "amount", "turnover", "pct_chg"]

    ALL_INDICES: List[str] = ["000016", "000300", "000688", "399006", "000905", "000852"]

    INDEX_NAMES: dict = {
        "000016": "上证50",
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
        "399006": "创业板指",
        "000688": "科创50",
    }


class BacktestSchema:
    """回测结果Schema"""

    EQUITY_CURVE_COLUMNS: List[str] = ["date", "nav", "hs300_nav"]
    TRADE_COLUMNS: List[str] = [
        "symbol", "entry_date", "exit_date",
        "entry_price", "exit_price", "return", "weight"
    ]
    METRICS_COLUMNS: List[str] = [
        "total_return", "annualized_return", "max_drawdown",
        "sharpe_ratio", "calmar_ratio", "win_rate",
        "profit_factor", "num_trades"
    ]
