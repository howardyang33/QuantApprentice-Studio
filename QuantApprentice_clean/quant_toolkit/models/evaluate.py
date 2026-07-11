"""
模型评估模块

提供模型评估和预测功能。
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def evaluate_teacher_model(
    model,
    dataset: dict,
    metrics: Optional[List[str]] = None,
) -> dict:
    """
    评估教师模型

    Args:
        model: 训练好的模型
        dataset: 数据集字典（需包含X, y）
        metrics: 评估指标列表，默认 ["mse", "rmse", "mae", "r2"]

    Returns:
        评估指标字典

    Example:
        >>> metrics = evaluate_teacher_model(model, test_dataset)
        >>> print(metrics["r2"])
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    metrics = metrics or ["mse", "rmse", "mae", "r2"]

    X = dataset["X"]
    y_true = dataset["y"]
    y_pred = model.predict(X)

    result = {}
    if "mse" in metrics:
        result["mse"] = mean_squared_error(y_true, y_pred)
    if "rmse" in metrics:
        result["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    if "mae" in metrics:
        result["mae"] = mean_absolute_error(y_true, y_pred)
    if "r2" in metrics:
        result["r2"] = r2_score(y_true, y_pred)

    return result


def predict_teacher_scores(
    model,
    dataset: dict,
) -> np.ndarray:
    """
    使用教师模型进行预测

    Args:
        model: 训练好的模型
        dataset: 数据集字典（需包含X）

    Returns:
        预测分数数组

    Example:
        >>> scores = predict_teacher_scores(model, dataset)
    """
    X = dataset["X"]
    return model.predict(X)


def compute_feature_importance(model, feature_cols: List[str]) -> pd.DataFrame:
    """
    计算特征重要性

    Args:
        model: 训练好的模型（需有feature_importances_属性）
        feature_cols: 特征列名列表

    Returns:
        特征重要性DataFrame

    Example:
        >>> importance = compute_feature_importance(model, feature_cols)
        >>> print(importance.head(10))
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("模型不支持特征重要性")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance
