"""
回测引擎模块

固定批次滚动窗口法（Fixed Batch Rolling Window）回测引擎。
核心逻辑:
  - 固定N个批次（N=lock_days），每个批次占 1/N 仓位
  - 按交易日顺序滚动分配信号到各批次
  - 批次内多信号等权分配该批次的仓位
  - 将收益均匀分布到持有期的每一天

无未来函数。回测使用已知的entry/exit信息，这是回测阶段的预期行为。
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def run_backtest(
    signals: pd.DataFrame,
    price_data: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    运行回测

    Args:
        signals: 信号DataFrame，必须包含:
            - entry_date, exit_date: 进出日期
            - return_20d: 持有期收益
            - quintile (可选): 策略分档
        price_data: 价格数据（用于计算基准净值，可选）
        config: 回测配置
            {
                "strategy": "Baseline",   # 策略名称或列表
                "lock_days": 20,         # 持有期/批次数量
                "initial_cash": 1000000,
                "transaction_cost": 0.001,
            }

    Returns:
        回测结果字典: {
            "equity_curve": pd.Series,    # 净值曲线
            "trades": pd.DataFrame,       # 交易明细
            "metrics": dict,              # 回测指标
        }

    Example:
        >>> result = run_backtest(signals, config={"lock_days": 20})
        >>> print(result["metrics"]["total_return"])
    """
    config = config or {}
    strategy = config.get("strategy", "Baseline")
    lock_days = config.get("lock_days", 20)
    transaction_cost = config.get("transaction_cost", 0.001)

    # 筛选策略对应的信号
    if isinstance(strategy, str):
        if strategy == "Baseline":
            sel = signals.copy()
        elif strategy == "Q4+Q5":
            sel = signals[signals["quintile"].isin(["Q4", "Q5"])].copy()
        elif "quintile" in signals.columns:
            sel = signals[signals["quintile"] == strategy].copy()
        else:
            sel = signals.copy()
    else:
        sel = signals.copy()

    if len(sel) == 0:
        return {
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "metrics": {},
        }

    sel = sel.sort_values("entry_date").reset_index(drop=True)

    # 获取所有交易日
    all_trade_dates = pd.date_range(
        start=sel["entry_date"].min(),
        end=sel["exit_date"].max(),
        freq="B",  # 工作日
    )

    # 计算净值曲线
    nav = _compute_nav_fixed_batch(sel, all_trade_dates, lock_days, transaction_cost)

    # 计算指标
    from .metrics import compute_backtest_metrics
    metrics = compute_backtest_metrics(nav)

    # 构建交易明细
    trades = sel[["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return_20d", "quintile"]].copy() if "quintile" in sel.columns else sel[["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return_20d"]].copy()
    trades["return"] = trades["return_20d"]

    return {
        "equity_curve": nav,
        "trades": trades,
        "metrics": metrics,
    }


def _compute_nav_fixed_batch(
    sel: pd.DataFrame,
    all_trade_dates: pd.DatetimeIndex,
    lock_days: int,
    transaction_cost: float,
) -> pd.Series:
    """
    固定批次滚动窗口法净值计算

    Args:
        sel: 筛选后的信号DataFrame
        all_trade_dates: 所有交易日索引
        lock_days: 持有期天数（批次数量）
        transaction_cost: 交易成本

    Returns:
        净值Series
    """
    n_batches = lock_days
    batch_size = 1.0 / n_batches

    n_dates = len(all_trade_dates)
    all_dates_arr = all_trade_dates.to_numpy().astype("datetime64[D]")

    # date -> index 映射
    date_to_idx = {}
    for i, d in enumerate(all_dates_arr):
        date_to_idx[int(np.int64(d))] = i

    # 各槽位每日收益
    slot_daily_returns = np.zeros((n_batches, n_dates), dtype=np.float64)

    # 按 entry_date 分组处理
    sorted_dates = sorted(sel["entry_date"].unique())

    for slot_idx, entry_date in enumerate(sorted_dates):
        slot_id = slot_idx % n_batches

        day_signals = sel[sel["entry_date"] == entry_date]
        if len(day_signals) == 0:
            continue

        # 计算批次内平均收益（扣除交易成本）
        signal_returns = day_signals["return_20d"].values
        avg_ret = np.mean(signal_returns) - transaction_cost * 2  # 买卖各一次

        batch_total_weight = batch_size

        # 获取持有期
        first_signal = day_signals.iloc[0]
        entry_dt = np.datetime64(first_signal["entry_date"], "D")
        exit_dt = np.datetime64(first_signal["exit_date"], "D")

        entry_int = int(np.int64(entry_dt))
        exit_int = int(np.int64(exit_dt))

        entry_idx = date_to_idx.get(entry_int, -1)
        exit_idx = date_to_idx.get(exit_int, -1)

        if entry_idx < 0 or exit_idx < 0 or entry_idx > exit_idx:
            continue

        holding_days = exit_idx - entry_idx + 1
        if holding_days <= 0:
            holding_days = 1

        # 等效日收益（复利方式）
        daily_equiv = (1.0 + avg_ret) ** (1.0 / holding_days) - 1.0
        daily_contribution = daily_equiv * batch_total_weight

        # 均匀分布到持有期每一天
        for day_idx in range(entry_idx, min(exit_idx + 1, n_dates)):
            slot_daily_returns[slot_id, day_idx] += daily_contribution

    # 组合净值
    portfolio_ret = np.sum(slot_daily_returns, axis=0)
    nav = np.cumprod(1.0 + portfolio_ret)

    return pd.Series(nav, index=all_trade_dates)


def export_backtest_results(
    result: dict,
    output_dir: Union[str, Path],
    prefix: str = "backtest",
) -> dict:
    """
    导出回测结果到文件

    Args:
        result: run_backtest 输出
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        导出的文件路径字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 净值曲线
    if len(result["equity_curve"]) > 0:
        equity_df = pd.DataFrame({
            "date": result["equity_curve"].index,
            "nav": result["equity_curve"].values,
        })
        equity_path = output_dir / f"{prefix}_equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        paths["equity_curve"] = str(equity_path)

    # 交易明细
    if len(result["trades"]) > 0:
        trades_path = output_dir / f"{prefix}_trades.csv"
        result["trades"].to_csv(trades_path, index=False)
        paths["trades"] = str(trades_path)

    # 指标
    import json
    metrics_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2, default=str)
    paths["metrics"] = str(metrics_path)

    return paths
