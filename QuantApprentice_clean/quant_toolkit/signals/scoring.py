"""
信号打分与分层模块

提供模型打分、分位阈值计算和信号分层功能。
核心原则: 分位阈值来自训练集，不使用测试集信息。
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def compute_train_thresholds(y_pred_train: np.ndarray) -> dict:
    """
    根据训练集预测值计算 Q1~Q5 的分界阈值

    Args:
        y_pred_train: 训练集预测值数组

    Returns:
        阈值字典: {Q1_upper, Q2_upper, Q3_upper, Q4_upper}

    Note:
        阈值仅来自训练集，模拟真实交易时无法预知测试集分布的场景。

    Example:
        >>> thresholds = compute_train_thresholds(model.predict(X_train))
    """
    pct = [20, 40, 60, 80]
    boundaries = np.percentile(y_pred_train, pct)
    return {
        "Q1_upper": boundaries[0],
        "Q2_upper": boundaries[1],
        "Q3_upper": boundaries[2],
        "Q4_upper": boundaries[3],
    }


def assign_quintile_by_thresholds(y_pred: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    用训练集学到的阈值对预测值数组分档

    Args:
        y_pred: 预测值数组
        thresholds: 阈值字典

    Returns:
        字符串数组，值为 Q1/Q2/Q3/Q4/Q5

    Note:
        不依赖测试集分布，严格使用训练集阈值。
    """
    labels = np.full(len(y_pred), "Q5", dtype=object)
    labels[y_pred < thresholds["Q4_upper"]] = "Q4"
    labels[y_pred < thresholds["Q3_upper"]] = "Q3"
    labels[y_pred < thresholds["Q2_upper"]] = "Q2"
    labels[y_pred < thresholds["Q1_upper"]] = "Q1"
    return labels


def score_signals(
    df: pd.DataFrame,
    feature_cols: List[str],
    model,
) -> pd.DataFrame:
    """
    使用训练好的模型对信号进行打分

    Args:
        df: 样本DataFrame
        feature_cols: 特征列名列表
        model: 训练好的模型（需有predict方法）

    Returns:
        包含 model_score 列的DataFrame

    Example:
        >>> df = score_signals(df, feature_cols, model)
    """
    df = df.copy()
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["model_score"] = model.predict(X)
    return df


def bucket_signals_by_score(
    df: pd.DataFrame,
    score_col: str = "model_score",
    n_bins: int = 5,
    method: str = "train_thresholds",
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    按分数对信号分档

    Args:
        df: 包含分数列的DataFrame
        score_col: 分数列名
        n_bins: 分档数量，默认5（Q1~Q5）
        method: 分档方法
            - "train_thresholds": 使用训练集阈值（需传入thresholds）
            - "qcut": 使用等频分位（仅用于分析，有数据泄漏风险）
        thresholds: 训练集阈值字典（method="train_thresholds"时必需）

    Returns:
        包含 quintile 列的DataFrame

    Note:
        推荐使用 "train_thresholds" 方法，避免数据泄漏。
        "qcut" 方法仅用于探索性分析，不应用于正式回测。

    Example:
        >>> df = bucket_signals_by_score(df, thresholds=thresholds)
        >>> df = bucket_signals_by_score(df, method="qcut")  # 仅分析用
    """
    df = df.copy()

    if method == "train_thresholds":
        if thresholds is None:
            raise ValueError("method='train_thresholds' 需要提供 thresholds 参数")
        df["quintile"] = assign_quintile_by_thresholds(df[score_col].values, thresholds)
    elif method == "qcut":
        # 警告: 此方法使用全量数据分位，存在数据泄漏风险
        labels = [f"Q{i+1}" for i in range(n_bins)]
        df["quintile"] = pd.qcut(df[score_col], q=n_bins, labels=labels)
    else:
        raise ValueError(f"不支持的分档方法: {method}")

    return df


def walk_forward_scoring(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    label_col: str = "return_20d",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Walk-forward 打分: 每年用之前所有年份训练，对当年测试集预测+划档

    Args:
        df: 样本DataFrame
        feature_cols: 特征列名列表
        model_factory: 模型工厂函数，返回可fit/predict的模型实例
        label_col: 标签列名
        year_col: 年份列名（如果不存在，自动从signal_date提取）

    Returns:
        附加了 quintile 和 pred_score 列的DataFrame

    Note:
        严格的无泄漏验证框架。训练集阈值不依赖测试集。

    Example:
        >>> def make_model():
        ...     return xgb.XGBRegressor(n_estimators=200, max_depth=6)
        >>> scored_df = walk_forward_scoring(df, feature_cols, make_model)
    """
    df = df.copy()

    # 确保年份列存在
    if year_col not in df.columns:
        df[year_col] = pd.to_datetime(df["signal_date"]).dt.year

    df_sorted = df.sort_values("signal_date").reset_index(drop=True)
    years = sorted(df_sorted[year_col].unique())

    scored_parts = []

    for year in years:
        train_df = df_sorted[df_sorted[year_col] < year]
        test_df = df_sorted[df_sorted[year_col] == year].copy()

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[label_col].values

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        thresholds = compute_train_thresholds(y_pred_train)

        y_pred_test = model.predict(X_test)
        test_df["quintile"] = assign_quintile_by_thresholds(y_pred_test, thresholds)
        test_df["pred_score"] = y_pred_test

        scored_parts.append(test_df)

    if not scored_parts:
        return pd.DataFrame()

    return pd.concat(scored_parts, ignore_index=True)
