"""
数据加载模块

提供标准化的日线数据加载、验证和交易日对齐功能。
所有路径通过参数传入，不依赖任何硬编码路径。
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from .schema import DailyDataSchema, IndexDataSchema


def load_daily_data(
    symbol: str,
    data_dir: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    加载单只股票的日线数据

    Args:
        symbol: 股票代码，如 "000001"
        data_dir: 数据目录路径，包含 {symbol}.csv 文件
        start_date: 起始日期，格式 "YYYY-MM-DD"，可选
        end_date: 结束日期，格式 "YYYY-MM-DD"，可选

    Returns:
        预处理后的 DataFrame，如果文件不存在或数据无效则返回 None

    Example:
        >>> df = load_daily_data("000001", data_dir="./day_klines")
        >>> df = load_daily_data("000001", data_dir="./data", start_date="2020-01-01", end_date="2023-12-31")
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"{symbol}.csv"

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)

        # 标准化日期
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 去重
        df = df.drop_duplicates(subset=["date"], keep="first")

        # 验证必需字段
        DailyDataSchema.validate(df)

        # 确保数值类型
        for col in DailyDataSchema.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 日期过滤
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if len(df) == 0:
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        warnings.warn(f"加载股票 {symbol} 失败: {e}")
        return None


def load_universe(
    data_dir: Union[str, Path],
    universe: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    加载股票列表

    Args:
        data_dir: 数据目录路径
        universe: 股票列表或特殊标记
            - "all": 加载目录下所有CSV文件（默认）
            - List[str]: 指定股票代码列表
            - None: 同 "all"

    Returns:
        股票代码列表

    Example:
        >>> symbols = load_universe("./day_klines", universe="all")
        >>> symbols = load_universe("./data", universe=["000001", "000002"])
    """
    data_dir = Path(data_dir)

    if universe is None or universe == "all":
        csv_files = sorted(data_dir.glob("*.csv"))
        return [f.stem for f in csv_files]
    elif isinstance(universe, list):
        return universe
    else:
        raise ValueError(f"universe 参数不支持: {universe}")


def validate_daily_data(df: pd.DataFrame) -> dict:
    """
    验证日线数据质量

    Args:
        df: 日线DataFrame

    Returns:
        验证结果字典，包含状态、错误信息和统计

    Example:
        >>> result = validate_daily_data(df)
        >>> print(result["status"])  # "ok" 或 "error"
    """
    result = {"status": "ok", "errors": [], "warnings": [], "stats": {}}

    # 检查必需列
    try:
        DailyDataSchema.validate(df)
    except ValueError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        return result

    # 检查空值
    for col in ["open", "close", "high", "low"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            result["warnings"].append(f"{col} 有 {null_count} 个空值")

    # 检查价格异常
    price_invalid = (
        (df["open"] <= 0).sum()
        + (df["close"] <= 0).sum()
        + (df["high"] <= 0).sum()
        + (df["low"] <= 0).sum()
    )
    if price_invalid > 0:
        result["warnings"].append(f"有 {price_invalid} 行价格 <= 0")

    # 检查 high >= low
    hl_invalid = (df["high"] < df["low"]).sum()
    if hl_invalid > 0:
        result["errors"].append(f"有 {hl_invalid} 行 high < low")
        result["status"] = "error"

    # 检查日期连续性（仅警告，A股有停牌）
    df_sorted = df.sort_values("date")
    date_diff = df_sorted["date"].diff().dt.days
    large_gaps = (date_diff > 30).sum()  # 超过30天的间隔
    if large_gaps > 0:
        result["warnings"].append(f"有 {large_gaps} 处超过30天的数据间隔（可能长期停牌）")

    # 统计信息
    result["stats"] = {
        "n_rows": len(df),
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),
        "n_days": (df["date"].max() - df["date"].min()).days,
    }

    return result


def align_trading_calendar(
    df_list: List[pd.DataFrame],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    多股票交易日对齐，返回共同的交易日索引

    Args:
        df_list: 多个股票DataFrame列表
        date_col: 日期列名

    Returns:
        共同交易日的DataFrame（仅含日期列）

    Note:
        这是一个工具函数，用于识别多股票的共同交易日。
        实际对齐操作应在各模块中根据业务需求进行。

    Example:
        >>> common_dates = align_trading_calendar([df1, df2, df3])
    """
    if not df_list:
        return pd.DataFrame(columns=[date_col])

    common_dates = set(df_list[0][date_col])
    for df in df_list[1:]:
        common_dates &= set(df[date_col])

    return pd.DataFrame({date_col: sorted(common_dates)})


def load_index_data(
    index_dir: Union[str, Path],
    indices: Optional[List[str]] = None,
) -> dict:
    """
    加载指数数据

    Args:
        index_dir: 指数数据目录
        indices: 指定指数代码列表，默认加载所有6个指数

    Returns:
        字典: {code: DataFrame}

    Example:
        >>> index_data = load_index_data("./data/index_klines")
        >>> hs300 = index_data["000300"]
    """
    index_dir = Path(index_dir)
    indices = indices or IndexDataSchema.ALL_INDICES

    result = {}
    for code in indices:
        file_path = index_dir / f"{code}.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            result[code] = df
        except Exception as e:
            warnings.warn(f"加载指数 {code} 失败: {e}")

    return result
