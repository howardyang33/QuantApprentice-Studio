"""
技术指标计算模块

包含KDJ指标和基础技术指标计算。
所有计算仅使用历史数据，无未来函数。
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算KDJ指标

    标准日线KDJ: RSV窗口9, K平滑3, D平滑3
    通达信标准初始化: K=D=50

    Args:
        df: DataFrame，必须包含 high, low, close 列
        n: RSV计算窗口，默认9
        m1: K平滑参数，默认3
        m2: D平滑参数，默认3

    Returns:
        包含 K, D, J 列的DataFrame

    Note:
        无未来函数。滚动窗口计算，当日结果仅依赖当日及之前n日数据。
    """
    df = df.copy()

    # 计算RSV
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()

    # 避免除零
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-12) * 100

    # 初始化K, D - 通达信标准: K=D=50
    k = np.ones(len(df)) * 50
    d = np.ones(len(df)) * 50

    # 递归计算K, D (从第n天开始，前n-1天保持50)
    for i in range(n, len(df)):
        k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv.iloc[i]
        d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]

    df["K"] = k
    df["D"] = d
    df["J"] = 3 * k - 2 * d

    # 处理NaN
    df["K"] = df["K"].fillna(50)
    df["D"] = df["D"].fillna(50)
    df["J"] = df["J"].fillna(50)

    return df


def calculate_kdj_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算KDJ派生指标

    Args:
        df: 包含 K, D, J 列的DataFrame

    Returns:
        包含派生指标的DataFrame

    Note:
        无未来函数。所有diff操作仅使用历史数据。
    """
    df = df.copy()

    # KDJ变化率
    df["dK_1"] = df["K"].diff(1)
    df["dD_1"] = df["D"].diff(1)
    df["dJ_1"] = df["J"].diff(1)
    df["dJ_3"] = df["J"].diff(3)

    # J与D差值
    df["J_minus_D"] = df["J"] - df["D"]

    # 超卖深度
    df["oversold_depth"] = 13 - df["J"]
    df["oversold_depth"] = df["oversold_depth"].clip(lower=0)

    # J在低位的天数 (最近10天)
    df["days_J_below_20_last_10"] = df["J"].rolling(10, min_periods=1).apply(
        lambda x: (x < 20).sum(), raw=False
    )
    df["days_J_below_10_last_10"] = df["J"].rolling(10, min_periods=1).apply(
        lambda x: (x < 10).sum(), raw=False
    )

    # 填充NaN
    df = df.fillna(0)

    return df


def _calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算收益率特征"""
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)
    return df


def _calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """计算移动均线相关特征"""
    df["ma5"] = df["close"].rolling(5, min_periods=1).mean()
    df["ma10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()
    df["close_to_ma5"] = df["close"] / (df["ma5"] + 1e-12)
    df["close_to_ma10"] = df["close"] / (df["ma10"] + 1e-12)
    df["close_to_ma20"] = df["close"] / (df["ma20"] + 1e-12)
    df["close_to_ma60"] = df["close"] / (df["ma60"] + 1e-12)
    return df


def _calculate_position_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算位置指标 (价格在N日区间中的相对位置)"""
    for n in [20, 60]:
        rolling_min = df["close"].rolling(n, min_periods=1).min()
        rolling_max = df["close"].rolling(n, min_periods=1).max()
        df[f"pos_{n}"] = (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-12)
    return df


def _calculate_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算K线形态特征"""
    df["prev_close"] = df["close"].shift(1)
    df["body_pct"] = (df["close"] - df["open"]) / (df["open"] + 1e-12)
    df["amplitude"] = (df["high"] - df["low"]) / (df["prev_close"] + 1e-12)
    df["upper_shadow"] = (df["high"] - np.maximum(df["close"], df["open"])) / (df["prev_close"] + 1e-12)
    df["lower_shadow"] = (np.minimum(df["close"], df["open"]) - df["low"]) / (df["prev_close"] + 1e-12)
    return df


def _calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算波动率特征"""
    ret = df["close"].pct_change()
    df["volatility_5"] = ret.rolling(5, min_periods=1).std()
    df["volatility_10"] = ret.rolling(10, min_periods=1).std()
    df["volatility_20"] = ret.rolling(20, min_periods=1).std()
    df["vol_ratio_5_20"] = df["volatility_5"] / (df["volatility_20"] + 1e-12)
    return df


def _calculate_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交额相关特征"""
    if "amount" not in df.columns:
        return df
    df["amt_log"] = np.log1p(df["amount"])
    df["amt_ma5"] = df["amount"].rolling(5, min_periods=1).mean()
    df["amt_ma20"] = df["amount"].rolling(20, min_periods=1).mean()
    df["amt_ma5_ratio"] = df["amount"] / (df["amt_ma5"] + 1e-12)
    df["amt_ma20_ratio"] = df["amount"] / (df["amt_ma20"] + 1e-12)
    amt_mean20 = df["amount"].rolling(20, min_periods=1).mean()
    amt_std20 = df["amount"].rolling(20, min_periods=1).std()
    df["amt_zscore_20"] = (df["amount"] - amt_mean20) / (amt_std20 + 1e-12)
    return df


def compute_technical_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    计算所有基础技术指标（v1特征）

    Args:
        df: 日K DataFrame，必须包含 open, close, high, low
        config: 可选配置字典，当前未使用（预留）

    Returns:
        包含v1特征的DataFrame

    Note:
        无未来函数。所有特征仅使用signal_date及之前的历史数据。
        该函数应在 calculate_kdj 和 calculate_kdj_derivatives 之后调用。

    Example:
        >>> df = calculate_kdj(df)
        >>> df = calculate_kdj_derivatives(df)
        >>> df = compute_technical_features(df)
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = _calculate_returns(df)
    df = _calculate_moving_averages(df)
    df = _calculate_position_indicators(df)
    df = _calculate_candle_features(df)
    df = _calculate_volatility(df)
    df = _calculate_amount_features(df)

    # 清理inf/nan
    for col in df.select_dtypes(include=[np.number]).columns:
        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0.0
            df[col] = arr

    return df


def get_v1_feature_columns() -> list:
    """返回v1特征列名（35个）"""
    return [
        # 收益率 (6)
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
        # 均线比值 (4)
        "close_to_ma5", "close_to_ma10", "close_to_ma20", "close_to_ma60",
        # 位置指标 (2)
        "pos_20", "pos_60",
        # KDJ (11)
        "K", "D", "J", "dK_1", "dD_1", "dJ_1", "dJ_3", "J_minus_D",
        "oversold_depth", "days_J_below_20_last_10", "days_J_below_10_last_10",
        # K线形态 (4)
        "body_pct", "amplitude", "upper_shadow", "lower_shadow",
        # 波动率 (4)
        "volatility_5", "volatility_10", "volatility_20", "vol_ratio_5_20",
        # 成交额 (4)
        "amt_log", "amt_ma5_ratio", "amt_ma20_ratio", "amt_zscore_20",
    ]
