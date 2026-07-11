"""
数据集构建模块

提供从样本DataFrame构建机器学习数据集的标准接口。
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def build_ml_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_config: Optional[dict] = None,
) -> dict:
    """
    构建机器学习数据集

    Args:
        df: 样本DataFrame（信号+特征）
        feature_cols: 特征列名列表，None则自动识别
        label_config: 标签配置
            {
                "horizon": 5,           # 标签 horizon
                "type": "future_return", # 标签类型
                "col": "return_20d",    # 标签列名
            }

    Returns:
        数据集字典: {
            "X": DataFrame,           # 特征矩阵
            "y": ndarray,             # 标签向量
            "feature_cols": list,     # 特征列名
            "meta": DataFrame,        # 元数据（symbol, date等）
        }

    Example:
        >>> dataset = build_ml_dataset(df, label_config={"horizon": 20, "col": "return_20d"})
        >>> X, y = dataset["X"], dataset["y"]
    """
    df = df.copy()

    # 自动识别特征列
    if feature_cols is None:
        from ..data.schema import SignalSchema
        exclude = set(SignalSchema.META_COLUMNS)
        feature_cols = [c for c in df.columns if c not in exclude]

    # 标签配置
    label_config = label_config or {"horizon": 20, "type": "future_return", "col": "return_20d"}
    label_col = label_config.get("col", "return_20d")

    # 提取特征和标签
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[label_col].values if label_col in df.columns else np.zeros(len(df))

    # 元数据
    meta_cols = ["symbol", "signal_date", "entry_date", "exit_date", "holding_days"]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()

    return {
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "meta": meta,
    }


def split_by_date(
    dataset: dict,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    date_col: str = "signal_date",
) -> Tuple[dict, dict]:
    """
    按日期划分训练集和测试集

    Args:
        dataset: build_ml_dataset 输出的数据集字典
        train_start: 训练集开始日期
        train_end: 训练集结束日期
        test_start: 测试集开始日期
        test_end: 测试集结束日期
        date_col: 日期列名

    Returns:
        (train_dataset, test_dataset)

    Example:
        >>> train_ds, test_ds = split_by_date(dataset, train_end="2021-12-31", test_start="2022-01-01")
    """
    meta = dataset["meta"]
    X = dataset["X"]
    y = dataset["y"]

    # 确保日期格式正确
    meta[date_col] = pd.to_datetime(meta[date_col])

    # 训练集掩码
    train_mask = pd.Series(True, index=meta.index)
    if train_start:
        train_mask &= meta[date_col] >= pd.to_datetime(train_start)
    if train_end:
        train_mask &= meta[date_col] <= pd.to_datetime(train_end)

    # 测试集掩码
    test_mask = pd.Series(True, index=meta.index)
    if test_start:
        test_mask &= meta[date_col] >= pd.to_datetime(test_start)
    if test_end:
        test_mask &= meta[date_col] <= pd.to_datetime(test_end)

    train_dataset = {
        "X": X[train_mask].reset_index(drop=True),
        "y": y[train_mask],
        "feature_cols": dataset["feature_cols"],
        "meta": meta[train_mask].reset_index(drop=True),
    }

    test_dataset = {
        "X": X[test_mask].reset_index(drop=True),
        "y": y[test_mask],
        "feature_cols": dataset["feature_cols"],
        "meta": meta[test_mask].reset_index(drop=True),
    }

    return train_dataset, test_dataset
