"""
KDJ策略信号生成模块

核心规则: KDJ J值 < 13 且前一交易日 J >= 13（下穿）
次日开盘价买入，持有固定天数后收盘价卖出。

无未来函数。所有信号判断仅使用当日及之前数据。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import warnings
import tempfile

warnings.filterwarnings("ignore")

from ..data.loaders import load_daily_data, load_universe
from ..data.trading_calendar import (
    executable_trade_indices,
    executable_trade_mask,
    nth_executable_trade_index,
)
from ..features.technical import calculate_kdj, calculate_kdj_derivatives, compute_technical_features
from ..features.factor_utils import calculate_all_features, get_feature_columns, load_index_data as load_index_features


def generate_kd13_signal(df: pd.DataFrame, j_threshold: int = 13) -> pd.Series:
    """
    生成KD13买点信号

    规则: 当日J < threshold 且 前一交易日J >= threshold（下穿）

    Args:
        df: 包含 J 列的DataFrame
        j_threshold: J值阈值，默认13

    Returns:
        bool Series，True表示产生信号

    Note:
        无未来函数。仅使用当日及前一日数据。

    Example:
        >>> df = calculate_kdj(df)
        >>> signals = generate_kd13_signal(df, j_threshold=13)
    """
    if "J" not in df.columns:
        return pd.Series(False, index=df.index)

    signal = (df["J"] < j_threshold) & (df["J"].shift(1) >= j_threshold)
    return signal


def calculate_trend_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算短期/长期趋势线，用于多头区间筛选

    Args:
        df: 日K DataFrame

    Returns:
        包含 short_trend, long_trend, bull_trend 列的DataFrame

    Note:
        无未来函数。EMA和MA仅使用历史数据。
    """
    df = df.copy()
    # 短期趋势线: EMA(EMA(C,10),10)
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    df["short_trend"] = ema10.ewm(span=10, adjust=False).mean()
    # 长期趋势线: (MA14 + MA28 + MA57 + MA114) / 4
    ma14 = df["close"].rolling(window=14, min_periods=1).mean()
    ma28 = df["close"].rolling(window=28, min_periods=1).mean()
    ma57 = df["close"].rolling(window=57, min_periods=1).mean()
    ma114 = df["close"].rolling(window=114, min_periods=1).mean()
    df["long_trend"] = (ma14 + ma28 + ma57 + ma114) / 4
    # 多头区间标记
    df["bull_trend"] = df["short_trend"] > df["long_trend"]
    return df

def construct_samples(
    df: pd.DataFrame,
    symbol: str,
    holding_days: int = 20,
    multi_holding_days: Optional[List[int]] = None,
    enable_bull_trend_filter: bool = False,
) -> List[dict]:
    """
    构造买点样本

    Args:
        df: 包含KDJ和特征的日K DataFrame
        symbol: 股票代码
        holding_days: 单一持有天数（默认20）
        multi_holding_days: 若传入list，则为每个持有期生成一行样本
        enable_bull_trend_filter: 是否启用多头趋势过滤

    Returns:
        样本字典列表

    Note:
        无未来函数。entry_date为signal_date后的第1个该股票可执行交易日，
        exit_date基于该股票自身可执行交易日计数。
        return_20d为标签（未来数据），仅用于训练。

    Example:
        >>> samples = construct_samples(df, "000001", holding_days=20)
    """
    samples = []

    # 多头趋势过滤（可选）
    if enable_bull_trend_filter:
        df = calculate_trend_lines(df)

    # 生成信号
    signals = generate_kd13_signal(df, j_threshold=13)

    # 决定持有期列表
    hd_list = multi_holding_days if multi_holding_days else [holding_days]
    tradeable_mask = executable_trade_mask(df)
    tradeable_indices = executable_trade_indices(df)

    for idx in range(len(df)):
        if not signals.iloc[idx]:
            continue
        if not tradeable_mask.iloc[idx]:
            continue

        # 多头过滤
        if enable_bull_trend_filter and "bull_trend" in df.columns:
            if not df.iloc[idx]["bull_trend"]:
                continue

        signal_date = df.iloc[idx]["date"]
        current_idx = idx

        # 下一可执行交易日买入
        entry_idx = nth_executable_trade_index(
            tradeable_indices,
            current_idx,
            offset=0,
            include_anchor=False,
        )
        if entry_idx is None:
            continue

        entry_date = df.iloc[entry_idx]["date"]
        entry_price = df.iloc[entry_idx]["open"]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # 为每个持有期生成一行样本
        for hd in hd_list:
            exit_idx = nth_executable_trade_index(
                tradeable_indices,
                entry_idx,
                offset=hd - 1,
                include_anchor=True,
            )
            if exit_idx is None:
                continue

            exit_date = df.iloc[exit_idx]["date"]
            exit_price = df.iloc[exit_idx]["close"]

            if pd.isna(exit_price) or exit_price <= 0:
                continue

            return_20d = exit_price / entry_price - 1

            samples.append({
                "symbol": symbol,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_days": hd,
                "return_20d": return_20d,
            })

    return samples


def _extract_sample_features(df: pd.DataFrame, signal_idx: int) -> dict:
    """
    提取样本特征（在signal_date上）
    只使用signal_date及之前的历史数据
    """
    features = {}
    for col in get_feature_columns():
        if col in df.columns:
            val = df.iloc[signal_idx][col]
            features[col] = val if pd.notna(val) else 0.0
    return features


def process_all_stocks(
    data_dir: Union[str, Path],
    index_dir: Optional[Union[str, Path]] = None,
    max_stocks: Optional[int] = None,
    holding_days: int = 20,
    multi_holding_days: Optional[List[int]] = None,
    enable_bull_trend_filter: bool = False,
    min_length: int = 70,
    batch_size: int = 200,
) -> pd.DataFrame:
    """
    处理所有股票，生成样本

    Args:
        data_dir: 股票数据目录
        index_dir: 指数数据目录（可选）
        max_stocks: 最大处理股票数（None表示全部）
        holding_days: 单一持有天数
        multi_holding_days: 多持有期列表
        enable_bull_trend_filter: 是否启用多头过滤
        min_length: 最小数据长度要求
        batch_size: 每批写入的股票数量

    Returns:
        样本DataFrame

    Example:
        >>> df = process_all_stocks("./day_klines", index_dir="./index_klines", max_stocks=100)
    """
    data_dir = Path(data_dir)
    csv_files = sorted(list(data_dir.glob("*.csv")))

    if max_stocks:
        csv_files = csv_files[:max_stocks]

    print(f"开始处理 {len(csv_files)} 只股票...")

    # 加载指数数据
    index_cache = None
    if index_dir:
        print("加载指数数据...")
        index_cache = load_index_features(index_dir)
        broad_count = len([k for k in index_cache if k != "_merged"])
        print(f"  已加载 {broad_count} 个指数数据")

    # 分批写入临时文件以控制内存
    tmp_dir = Path(tempfile.gettempdir())
    tmp_files = []
    batch_samples = []
    total_count = 0
    first_write = True

    def flush_batch(samples, first):
        if not samples:
            return first
        chunk_df = pd.DataFrame(samples)
        tmp_f = tmp_dir / f"qt_chunk_{len(tmp_files)}.csv"
        chunk_df.to_csv(tmp_f, index=False)
        tmp_files.append(tmp_f)
        return False

    for i, f in enumerate(csv_files):
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(csv_files)} 只股票，样本数: {total_count + len(batch_samples)}...")

        symbol = f.stem

        # 加载数据
        df = load_daily_data(symbol, data_dir)
        if df is None or len(df) < min_length:
            continue

        # 计算KDJ
        df = calculate_kdj(df)
        df = calculate_kdj_derivatives(df)

        # 计算股票级特征
        df = compute_technical_features(df)

        # 添加指数走势特征
        if index_cache:
            df = calculate_all_features(df, index_cache=index_cache)
        else:
            df = calculate_all_features(df)

        # 生成样本
        samples = construct_samples(
            df, symbol,
            holding_days=holding_days,
            multi_holding_days=multi_holding_days,
            enable_bull_trend_filter=enable_bull_trend_filter,
        )

        # 提取特征
        from collections import defaultdict
        feature_cache = defaultdict(dict)

        for sample in samples:
            key = (sample["symbol"], sample["signal_date"])
            if key not in feature_cache:
                signal_date = sample["signal_date"]
                signal_idx = df[df["date"] == signal_date].index
                if len(signal_idx) == 0:
                    continue
                signal_idx = signal_idx[0]
                features = _extract_sample_features(df, signal_idx)
                feature_cache[key] = features

            sample.update(feature_cache[key])
            batch_samples.append(sample)

        # 达到 batch_size 就写盘
        if len(batch_samples) >= batch_size * 20:
            total_count += len(batch_samples)
            first_write = flush_batch(batch_samples, first_write)
            batch_samples = []

    # 写入剩余
    if batch_samples:
        total_count += len(batch_samples)
        flush_batch(batch_samples, first_write)
        batch_samples = []

    print(f"\n共生成 {total_count} 个有效样本，正在合并...")

    if not tmp_files:
        return pd.DataFrame()

    # 合并
    merged_csv = tmp_dir / "qt_merged_all.csv"
    header_written = False
    for tf in tmp_files:
        chunk = pd.read_csv(tf)
        chunk.to_csv(merged_csv, mode="a" if header_written else "w",
                     header=not header_written, index=False)
        header_written = True
        tf.unlink()

    result = pd.read_csv(merged_csv)
    merged_csv.unlink()
    print(f"合并完成: {len(result)} 个样本")
    return result
