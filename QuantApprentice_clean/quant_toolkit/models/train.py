"""
模型训练模块

XGBoost模型训练，支持GPU/CPU自动降级。
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


def get_default_xgb_params(device: str = "auto") -> dict:
    """
    获取默认XGBoost参数

    Args:
        device: 计算设备，"auto"/"cuda"/"cpu"
            - "auto": 自动检测GPU，不可用则降级到CPU
            - "cuda": 强制使用GPU
            - "cpu": 强制使用CPU

    Returns:
        XGBoost参数字典
    """
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": 4,
        "verbosity": 0,
    }

    if device == "cuda":
        params["device"] = "cuda"
    elif device == "cpu":
        pass  # 默认就是CPU
    elif device == "auto":
        # 尝试检测GPU
        try:
            import xgboost as xgb
            # 简单测试GPU是否可用
            test_params = params.copy()
            test_params["device"] = "cuda"
            test_params["n_estimators"] = 1
            test_model = xgb.XGBRegressor(**test_params)
            # 如果这里不报错，说明GPU可用
            params["device"] = "cuda"
        except Exception:
            pass  # 使用CPU

    return params


def train_teacher_model(
    dataset: dict,
    model_config: Optional[dict] = None,
) -> dict:
    """
    训练教师模型（XGBoost回归）

    Args:
        dataset: build_ml_dataset 输出的数据集字典
        model_config: 模型配置
            {
                "type": "xgboost",
                "device": "auto",      # auto/cuda/cpu
                "params": {},          # 额外XGBoost参数
            }

    Returns:
        训练结果字典: {
            "model": 训练好的模型,
            "feature_cols": 特征列名,
            "train_metrics": 训练指标,
        }

    Note:
        自动处理GPU不可用的情况，降级到CPU。

    Example:
        >>> result = train_teacher_model(dataset, model_config={"device": "auto"})
        >>> model = result["model"]
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("需要安装 xgboost: pip install xgboost")

    model_config = model_config or {}
    device = model_config.get("device", "auto")
    extra_params = model_config.get("params", {})

    # 获取基础参数
    params = get_default_xgb_params(device)
    params.update(extra_params)

    X_train = dataset["X"]
    y_train = dataset["y"]

    # 训练模型
    try:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        model_type = f"xgboost_{params.get('device', 'cpu')}"
    except Exception as e:
        if "cuda" in str(e).lower() or "gpu" in str(e).lower():
            warnings.warn(f"GPU训练失败: {e}，降级到CPU")
            params.pop("device", None)
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            model_type = "xgboost_cpu"
        else:
            raise

    # 训练集指标
    y_pred_train = model.predict(X_train)
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y_train, y_pred_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)

    return {
        "model": model,
        "feature_cols": dataset["feature_cols"],
        "model_type": model_type,
        "train_metrics": {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
        },
    }
