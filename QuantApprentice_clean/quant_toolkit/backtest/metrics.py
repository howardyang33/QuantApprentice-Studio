"""
回测指标计算模块

计算常见的回测绩效指标。
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_backtest_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    计算回测绩效指标

    Args:
        equity_curve: 净值曲线Series（index=日期）
        trades: 交易明细DataFrame（可选）
        risk_free_rate: 无风险利率，默认3%

    Returns:
        指标字典，包含:
            - total_return: 总收益率
            - annualized_return: 年化收益率
            - max_drawdown: 最大回撤
            - sharpe_ratio: Sharpe比率
            - calmar_ratio: Calmar比率
            - win_rate: 胜率（如有trades）
            - profit_factor: 盈亏比（如有trades）
            - num_trades: 交易次数（如有trades）

    Example:
        >>> metrics = compute_backtest_metrics(equity_curve)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    if len(equity_curve) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
        }

    nav = equity_curve.values
    dates = equity_curve.index

    # 日收益率
    daily_ret = np.diff(nav) / nav[:-1]

    # 总收益率
    total_return = nav[-1] / nav[0] - 1

    # 年化收益率
    n_days = len(nav)
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    # 最大回撤
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    max_drawdown = np.max(drawdown)

    # Sharpe比率
    if len(daily_ret) > 1 and np.std(daily_ret) > 0:
        excess_ret = daily_ret - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_ret) / np.std(daily_ret) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Calmar比率
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # 波动率
    volatility = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 0.0

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "calmar_ratio": float(calmar_ratio),
        "volatility": float(volatility),
        "num_days": int(n_days),
    }

    # 交易级指标
    if trades is not None and len(trades) > 0:
        returns = trades["return"].values if "return" in trades.columns else trades["return_20d"].values
        metrics["num_trades"] = int(len(trades))
        metrics["win_rate"] = float(np.mean(returns > 0))
        metrics["profit_factor"] = float(
            np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
            if np.sum(returns < 0) != 0 else float("inf")
        )
        metrics["avg_return"] = float(np.mean(returns))
        metrics["median_return"] = float(np.median(returns))

    return metrics
