"""
特征工程工具模块（v2新增特征 + 指数特征）

包含4组v2特征：
  1. 成交量深度特征
  2. 市值相关特征
  3. 关键K线特征
  4. 指数走势特征

以及特征汇总函数。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import warnings

warnings.filterwarnings("ignore")

from .technical import get_v1_feature_columns


# ============================================================
# 第1组: 成交量深度特征
# ============================================================

def calculate_volume_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    第1组: 成交量深度特征
    使用 amount 列（成交额）为主力，避免依赖 volume 的单位差异

    Note:
        无未来函数。所有rolling操作使用当日及之前数据。
    """
    EPS = 1e-12

    _zero_cols = [
        "amount_ratio_3", "amount_ratio_5", "amount_ratio_10", "amount_ratio_20",
        "amount_percentile_20", "amount_percentile_60",
        "high_volume_days_5", "high_volume_days_10", "high_volume_days_20",
        "low_volume_days_5", "low_volume_days_10",
        "up_volume_pressure_5", "up_volume_pressure_10", "up_volume_pressure_20",
        "down_volume_pressure_5", "down_volume_pressure_10", "down_volume_pressure_20",
        "price_volume_impulse_5", "price_volume_impulse_10", "price_volume_impulse_20",
        "high_volume_upbar_ratio_10", "high_volume_downbar_ratio_10",
        "low_vol_surge_ratio",
    ]

    if "amount" not in df.columns:
        for col in _zero_cols:
            df[col] = 0.0
        return df

    amt = df["amount"].replace(0, np.nan).ffill().fillna(0)
    amt_ma3 = amt.rolling(3, min_periods=1).mean()
    amt_ma5 = amt.rolling(5, min_periods=1).mean()
    amt_ma10 = amt.rolling(10, min_periods=1).mean()
    amt_ma20 = amt.rolling(20, min_periods=1).mean()

    # 1.1 当日成交额 / 过去N日均成交额
    df["amount_ratio_3"] = amt / (amt_ma3 + EPS)
    df["amount_ratio_5"] = amt / (amt_ma5 + EPS)
    df["amount_ratio_10"] = amt / (amt_ma10 + EPS)
    df["amount_ratio_20"] = amt / (amt_ma20 + EPS)

    # 1.2 单日成交额在历史窗口中的分位
    def rolling_percentile(series, window):
        def rank_in_window(x):
            return (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1)
        return series.rolling(window, min_periods=2).apply(rank_in_window, raw=True)

    df["amount_percentile_20"] = rolling_percentile(amt, 20)
    df["amount_percentile_60"] = rolling_percentile(amt, 60)

    # 1.3 近N日放量天数
    is_high_vol = (amt > 1.5 * amt_ma20).astype(float)
    df["high_volume_days_5"] = is_high_vol.rolling(5, min_periods=1).sum()
    df["high_volume_days_10"] = is_high_vol.rolling(10, min_periods=1).sum()
    df["high_volume_days_20"] = is_high_vol.rolling(20, min_periods=1).sum()

    # 1.4 近N日缩量天数
    is_low_vol = (amt < 0.7 * amt_ma20).astype(float)
    df["low_volume_days_5"] = is_low_vol.rolling(5, min_periods=1).sum()
    df["low_volume_days_10"] = is_low_vol.rolling(10, min_periods=1).sum()

    # 1.5 上涨放量 / 下跌放量拆分
    ar20 = df["amount_ratio_20"].values
    ret_1 = df["close"].pct_change().values

    n = len(df)
    up_ar20 = np.where(ret_1 > 0, ar20, 0.0)
    down_ar20 = np.where(ret_1 < 0, ar20, 0.0)

    up_s = pd.Series(up_ar20, index=df.index)
    down_s = pd.Series(down_ar20, index=df.index)

    df["up_volume_pressure_5"] = up_s.rolling(5, min_periods=1).sum()
    df["up_volume_pressure_10"] = up_s.rolling(10, min_periods=1).sum()
    df["up_volume_pressure_20"] = up_s.rolling(20, min_periods=1).sum()

    df["down_volume_pressure_5"] = down_s.rolling(5, min_periods=1).sum()
    df["down_volume_pressure_10"] = down_s.rolling(10, min_periods=1).sum()
    df["down_volume_pressure_20"] = down_s.rolling(20, min_periods=1).sum()

    # 1.6 量价冲量
    impulse = pd.Series(ret_1 * ar20, index=df.index)
    df["price_volume_impulse_5"] = impulse.rolling(5, min_periods=1).sum()
    df["price_volume_impulse_10"] = impulse.rolling(10, min_periods=1).sum()
    df["price_volume_impulse_20"] = impulse.rolling(20, min_periods=1).sum()

    # 1.7 放量阳线/阴线占比
    is_yang = (df["close"] > df["open"]).astype(float)
    is_yin = (df["close"] < df["open"]).astype(float)
    df["high_volume_upbar_ratio_10"] = (is_yang * is_high_vol).rolling(10, min_periods=1).mean()
    df["high_volume_downbar_ratio_10"] = (is_yin * is_high_vol).rolling(10, min_periods=1).mean()

    # 1.8 低位放量特征
    amt_ma30 = amt.rolling(30, min_periods=5).mean()
    ar30 = amt / (amt_ma30 + EPS)
    pos20 = df["pos_20"] if "pos_20" in df.columns else pd.Series(0.5, index=df.index)
    low_pos_ar30 = np.where(pos20 < 0.3, ar30, np.nan)
    low_pos_s = pd.Series(low_pos_ar30, index=df.index)
    df["low_vol_surge_ratio"] = low_pos_s.rolling(20, min_periods=1).mean().fillna(0)

    return df


# ============================================================
# 第2组: 市值相关特征
# ============================================================

def calculate_market_cap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    第2组: 市值特征

    Note:
        单股计算时无法做横截面分桶，故用时序分位数近似（60日历史分位）。
        这不是真实的横截面市值分位，存在近似误差。
    """
    EPS = 1e-12

    has_turnover = "turnover" in df.columns and df["turnover"].abs().sum() > 0
    has_volume = "volume" in df.columns

    if has_turnover and has_volume:
        turnover_safe = df["turnover"].replace(0, np.nan).ffill().fillna(0.001)
        # Local stock CSVs use fractional turnover (0.02 == 2%), but some
        # external datasets store percentages. Detect both safely.
        turnover_scale = 100.0 if float(turnover_safe.median()) > 1.0 else 1.0
        turnover_ratio = turnover_safe / turnover_scale
        float_mcap = df["close"] * df["volume"] / (turnover_ratio + EPS)
    elif has_volume:
        float_mcap = df["close"] * df["volume"]
    else:
        df["log_market_cap"] = 0.0
        df["market_cap_bucket"] = 2.0
        return df

    float_mcap = float_mcap.clip(lower=1)
    df["log_market_cap"] = np.log(float_mcap)

    # 市值分桶（用滚动60日历史分位近似横截面）
    def bucket_by_history(series, window=60):
        def _bucket(x):
            val = x[-1]
            hist = x[:-1]
            if len(hist) == 0:
                return 2.0
            pct = (hist < val).mean()
            if pct < 0.1:
                return 0.0
            elif pct < 0.3:
                return 1.0
            elif pct < 0.6:
                return 2.0
            elif pct < 0.9:
                return 3.0
            else:
                return 4.0
        return series.rolling(window, min_periods=2).apply(_bucket, raw=True)

    df["market_cap_bucket"] = bucket_by_history(df["log_market_cap"], window=60).fillna(2.0)

    return df


# ============================================================
# 第3组: 关键K线特征
# ============================================================

def calculate_key_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    第3组: 关键K线特征
    过去20日最大阳/阴/放量阳/放量阴线 + 突破/破位日

    Note:
        突破日/破位日判断使用窗口内简化逻辑，非精确的20日rolling_max突破。
        这是已知的近似处理，在审计报告中已标注。
    """
    EPS = 1e-12
    W = 20

    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    ret_1 = np.zeros(n)
    ret_1[1:] = close[1:] / close[:-1] - 1

    if "amount" in df.columns:
        amt = df["amount"].values.copy()
        amt[np.isnan(amt)] = 0
        amt_ma20 = np.zeros(n)
        for i in range(n):
            start = max(0, i - 19)
            w_amt = amt[start: i + 1]
            amt_ma20[i] = w_amt.mean() if len(w_amt) > 0 else 1
        ar20 = amt / (amt_ma20 + EPS)
    else:
        ar20 = np.ones(n)
        amt = np.ones(n)

    body_pct = (close - open_) / (open_ + EPS)
    body_abs = np.abs(body_pct)
    upper_shadow = (high - np.maximum(close, open_)) / (np.maximum(close, open_) + EPS)
    lower_shadow = (np.minimum(close, open_) - low) / (np.minimum(close, open_) + EPS)
    cp = (close - low) / (high - low + EPS)

    cols_init = {
        "largest_up_day_return": 0.0,
        "largest_up_day_amount_ratio": 0.0,
        "largest_up_day_body_pct": 0.0,
        "largest_up_day_upper_shadow": 0.0,
        "largest_up_day_lower_shadow": 0.0,
        "days_since_largest_up_day": float(W),
        "largest_down_day_return": 0.0,
        "largest_down_day_amount_ratio": 0.0,
        "largest_down_day_body_pct": 0.0,
        "largest_down_day_upper_shadow": 0.0,
        "largest_down_day_lower_shadow": 0.0,
        "days_since_largest_down_day": float(W),
        "largest_volume_up_day_return": 0.0,
        "largest_volume_up_day_body_pct": 0.0,
        "largest_volume_up_day_close_position": 0.0,
        "days_since_largest_volume_up_day": float(W),
        "largest_volume_down_day_return": 0.0,
        "largest_volume_down_day_body_pct": 0.0,
        "largest_volume_down_day_close_position": 0.0,
        "days_since_largest_volume_down_day": float(W),
        "has_breakout_day": 0.0,
        "days_since_breakout_day": 999.0,
        "breakout_day_return": 0.0,
        "breakout_day_amount_ratio": 0.0,
        "has_breakdown_day": 0.0,
        "days_since_breakdown_day": 999.0,
        "breakdown_day_return": 0.0,
        "breakdown_day_amount_ratio": 0.0,
    }
    result = {k: np.full(n, v) for k, v in cols_init.items()}

    for i in range(W - 1, n):
        start = i - W + 1
        w_ret = ret_1[start: i + 1]
        w_ar20 = ar20[start: i + 1]
        w_body = body_pct[start: i + 1]
        w_upper = upper_shadow[start: i + 1]
        w_lower = lower_shadow[start: i + 1]
        w_cp = cp[start: i + 1]
        w_amt = amt[start: i + 1]
        wlen = W

        # 3.1 最大阳线
        up_mask = w_ret > 0
        if up_mask.any():
            k_up = np.argmax(w_ret)
            result["largest_up_day_return"][i] = w_ret[k_up]
            result["largest_up_day_amount_ratio"][i] = w_ar20[k_up]
            result["largest_up_day_body_pct"][i] = w_body[k_up]
            result["largest_up_day_upper_shadow"][i] = w_upper[k_up]
            result["largest_up_day_lower_shadow"][i] = w_lower[k_up]
            result["days_since_largest_up_day"][i] = float(wlen - 1 - k_up)

        # 3.2 最大阴线
        down_mask = w_ret < 0
        if down_mask.any():
            k_dn = np.argmin(w_ret)
            result["largest_down_day_return"][i] = w_ret[k_dn]
            result["largest_down_day_amount_ratio"][i] = w_ar20[k_dn]
            result["largest_down_day_body_pct"][i] = w_body[k_dn]
            result["largest_down_day_upper_shadow"][i] = w_upper[k_dn]
            result["largest_down_day_lower_shadow"][i] = w_lower[k_dn]
            result["days_since_largest_down_day"][i] = float(wlen - 1 - k_dn)

        # 3.3 最大放量阳线
        yang_mask = w_body > 0
        if yang_mask.any():
            yang_amt = np.where(yang_mask, w_amt, -1)
            k_vu = np.argmax(yang_amt)
            result["largest_volume_up_day_return"][i] = w_ret[k_vu]
            result["largest_volume_up_day_body_pct"][i] = w_body[k_vu]
            result["largest_volume_up_day_close_position"][i] = w_cp[k_vu]
            result["days_since_largest_volume_up_day"][i] = float(wlen - 1 - k_vu)

        # 3.4 最大放量阴线
        yin_mask = w_body < 0
        if yin_mask.any():
            yin_amt = np.where(yin_mask, w_amt, -1)
            k_vd = np.argmax(yin_amt)
            result["largest_volume_down_day_return"][i] = w_ret[k_vd]
            result["largest_volume_down_day_body_pct"][i] = w_body[k_vd]
            result["largest_volume_down_day_close_position"][i] = w_cp[k_vd]
            result["days_since_largest_volume_down_day"][i] = float(wlen - 1 - k_vd)

        # 3.5 突破日 / 破位日（简化判断）
        found_breakout = False
        found_breakdown = False
        w_high = high[start: i + 1]
        w_low = low[start: i + 1]

        for j in range(1, wlen):
            prev_max_high = w_high[:j].max()
            prev_min_low = w_low[:j].min()
            c_j = close[start + j]
            ar_j = w_ar20[j]
            ret_j = w_ret[j]

            if not found_breakout and c_j > prev_max_high and ar_j > 1.5:
                result["has_breakout_day"][i] = 1.0
                result["days_since_breakout_day"][i] = float(wlen - 1 - j)
                result["breakout_day_return"][i] = ret_j
                result["breakout_day_amount_ratio"][i] = ar_j
                found_breakout = True

            if not found_breakdown and c_j < prev_min_low and ar_j > 1.5:
                result["has_breakdown_day"][i] = 1.0
                result["days_since_breakdown_day"][i] = float(wlen - 1 - j)
                result["breakdown_day_return"][i] = ret_j
                result["breakdown_day_amount_ratio"][i] = ar_j
                found_breakdown = True

            if found_breakout and found_breakdown:
                break

    for col, arr in result.items():
        df[col] = arr

    return df


# ============================================================
# 第4组: 指数走势特征
# ============================================================

ALL_INDICES = ["000016", "000300", "000688", "399006", "000905", "000852"]

INDEX_NAMES = {
    "000016": "上证50",
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000688": "科创50",
}


def load_index_data(index_dir: Union[str, Path]) -> dict:
    """
    加载6个指数数据，预计算各指数特征

    Args:
        index_dir: 指数数据目录路径

    Returns:
        dict: code -> DataFrame(带日期索引)
              同时包含 '_merged' 键，为合并宽表

    Note:
        路径通过参数传入，无硬编码路径。
    """
    index_dir = Path(index_dir)
    index_data = {}

    for code in ALL_INDICES:
        fpath = index_dir / f"{code}.csv"
        if not fpath.exists():
            continue
        try:
            idf = pd.read_csv(fpath)
            idf["date"] = pd.to_datetime(idf["date"])
            idf = idf.sort_values("date").reset_index(drop=True)

            # 基础特征
            idf[f"ret_1_{code}"] = idf["close"].pct_change(1)
            idf[f"ret_5_{code}"] = idf["close"].pct_change(5)
            idf[f"ret_10_{code}"] = idf["close"].pct_change(10)

            ma20 = idf["close"].rolling(20, min_periods=1).mean()
            ma60 = idf["close"].rolling(60, min_periods=1).mean()
            idf[f"idx_close_to_ma20_{code}"] = idf["close"] / (ma20 + 1e-12)
            idf[f"idx_close_to_ma60_{code}"] = idf["close"] / (ma60 + 1e-12)

            rmin20 = idf["close"].rolling(20, min_periods=1).min()
            rmax20 = idf["close"].rolling(20, min_periods=1).max()
            idf[f"idx_pos_20_{code}"] = (idf["close"] - rmin20) / (rmax20 - rmin20 + 1e-12)

            rmin60 = idf["close"].rolling(60, min_periods=1).min()
            rmax60 = idf["close"].rolling(60, min_periods=1).max()
            idf[f"idx_pos_60_{code}"] = (idf["close"] - rmin60) / (rmax60 - rmin60 + 1e-12)

            ret = idf["close"].pct_change()
            idf[f"idx_volatility_10_{code}"] = ret.rolling(10, min_periods=1).std()

            index_data[code] = idf
        except Exception as e:
            warnings.warn(f"加载指数 {code} 失败: {e}")

    # 构建合并宽表
    if len(index_data) >= 2:
        all_dates = sorted(set().union(*[set(v["date"]) for v in index_data.values()]))
        date_df = pd.DataFrame({"date": all_dates})

        merged = date_df.copy()
        for code, idf in index_data.items():
            cols = ["date"] + [c for c in idf.columns if c.startswith((
                "ret_1_", "ret_5_", "ret_10_",
                "idx_close_to_ma20_", "idx_close_to_ma60_",
                "idx_pos_20_", "idx_pos_60_",
                "idx_volatility_10_",
            ))]
            merged = merged.merge(idf[cols], on="date", how="left")

        # 4.1 各指数均值特征
        def avg_by_prefix(merged, prefix):
            cols = [c for c in merged.columns if c.startswith(prefix)]
            if not cols:
                return pd.Series(0.0, index=merged.index)
            return merged[cols].mean(axis=1)

        merged["index_ret_1"] = avg_by_prefix(merged, "ret_1_")
        merged["index_ret_5"] = avg_by_prefix(merged, "ret_5_")
        merged["index_ret_10"] = avg_by_prefix(merged, "ret_10_")
        merged["index_close_to_ma20"] = avg_by_prefix(merged, "idx_close_to_ma20_")
        merged["index_close_to_ma60"] = avg_by_prefix(merged, "idx_close_to_ma60_")
        merged["index_pos_20"] = avg_by_prefix(merged, "idx_pos_20_")
        merged["index_pos_60"] = avg_by_prefix(merged, "idx_pos_60_")
        merged["index_volatility_10"] = avg_by_prefix(merged, "idx_volatility_10_")

        # 4.2 风格扩散特征
        def get_ret5(code):
            col = f"ret_5_{code}"
            return merged[col] if col in merged.columns else pd.Series(0.0, index=merged.index)

        merged["small_vs_large_ret5"] = get_ret5("000852") - get_ret5("000300")
        merged["mid_vs_large_ret5"] = get_ret5("000905") - get_ret5("000016")

        growth_ret5 = (get_ret5("399006") + get_ret5("000688")) / 2.0
        merged["growth_vs_bluechip_ret5"] = growth_ret5 - get_ret5("000016")

        # 4.3 指数同步性特征
        up_cols = [f"ret_1_{code}" for code in ALL_INDICES if f"ret_1_{code}" in merged.columns]
        if up_cols:
            up_matrix = (merged[up_cols] > 0).astype(float)
            merged["index_up_pct_1"] = up_matrix.mean(axis=1)
            merged["index_up_pct_2"] = up_matrix.rolling(2, min_periods=1).mean().mean(axis=1)
            merged["index_up_pct_3"] = up_matrix.rolling(3, min_periods=1).mean().mean(axis=1)
        else:
            merged["index_up_pct_1"] = 0.0
            merged["index_up_pct_2"] = 0.0
            merged["index_up_pct_3"] = 0.0

        index_data["_merged"] = merged

    return index_data


def get_sector_index(symbol: str) -> str:
    """股票代码 -> 板块指数代码映射"""
    if symbol.startswith("688"):
        return "000688"
    elif symbol.startswith("300"):
        return "399006"
    elif symbol.startswith(("600", "601", "603", "605")):
        return "000016"
    else:
        return "000852"


def add_index_features(df: pd.DataFrame, index_cache: dict, symbol: str) -> pd.DataFrame:
    """
    将指数走势特征合并到股票DataFrame

    Note:
        通过日期merge加入，使用how='left'，无未来数据风险。
        需在 calculate_all_features() 之后调用（需要ret_5, ret_20已存在）。
    """
    idx_feat_cols = [
        "index_ret_1", "index_ret_5", "index_ret_10",
        "index_close_to_ma20", "index_close_to_ma60",
        "index_pos_20", "index_pos_60",
        "index_volatility_10",
        "small_vs_large_ret5", "mid_vs_large_ret5", "growth_vs_bluechip_ret5",
        "index_up_pct_1", "index_up_pct_2", "index_up_pct_3",
    ]
    stock_relative_cols = [
        "stock_vs_hs300_ret5", "stock_vs_zz1000_ret5",
        "stock_vs_hs300_ret20", "stock_vs_zz1000_ret20",
    ]

    if not index_cache or "_merged" not in index_cache:
        for col in idx_feat_cols + stock_relative_cols:
            df[col] = 0.0
        return df

    merged = index_cache["_merged"]

    # 合并宽表特征
    merge_cols = ["date"] + [c for c in idx_feat_cols if c in merged.columns]
    df = df.merge(merged[merge_cols], on="date", how="left")
    for col in idx_feat_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 个股相对指数强弱
    hs300 = index_cache.get("000300")
    zz1000 = index_cache.get("000852")

    def get_idx_ret(idf, n_col):
        if idf is None or n_col not in idf.columns:
            return pd.Series(0.0, index=df.index)
        tmp = idf[["date", n_col]].rename(columns={n_col: "_idx_ret_tmp"})
        merged_tmp = df[["date"]].merge(tmp, on="date", how="left")
        return merged_tmp["_idx_ret_tmp"].fillna(0.0).values

    hs300_ret5 = get_idx_ret(hs300, "ret_5_000300")
    zz1000_ret5 = get_idx_ret(zz1000, "ret_5_000852")

    # 补充ret_20
    for code in ["000300", "000852"]:
        idf = index_cache.get(code)
        if idf is not None and "close" in idf.columns and f"ret_20_{code}" not in idf.columns:
            idf[f"ret_20_{code}"] = idf["close"].pct_change(20)

    hs300_ret20_arr = np.zeros(len(df))
    zz1000_ret20_arr = np.zeros(len(df))
    if hs300 is not None and "ret_20_000300" in hs300.columns:
        tmp = hs300[["date", "ret_20_000300"]].rename(columns={"ret_20_000300": "_v"})
        merged_tmp = df[["date"]].merge(tmp, on="date", how="left")
        hs300_ret20_arr = merged_tmp["_v"].fillna(0.0).values
    if zz1000 is not None and "ret_20_000852" in zz1000.columns:
        tmp = zz1000[["date", "ret_20_000852"]].rename(columns={"ret_20_000852": "_v"})
        merged_tmp = df[["date"]].merge(tmp, on="date", how="left")
        zz1000_ret20_arr = merged_tmp["_v"].fillna(0.0).values

    stock_ret5 = df["ret_5"].values if "ret_5" in df.columns else np.zeros(len(df))
    stock_ret20 = df["ret_20"].values if "ret_20" in df.columns else np.zeros(len(df))

    df["stock_vs_hs300_ret5"] = stock_ret5 - hs300_ret5
    df["stock_vs_zz1000_ret5"] = stock_ret5 - zz1000_ret5
    df["stock_vs_hs300_ret20"] = stock_ret20 - hs300_ret20_arr
    df["stock_vs_zz1000_ret20"] = stock_ret20 - zz1000_ret20_arr

    return df


# ============================================================
# 汇总函数
# ============================================================

def calculate_all_features(df: pd.DataFrame, index_cache: Optional[dict] = None) -> pd.DataFrame:
    """
    计算所有股票级特征（v1 + v2）

    Args:
        df: 日K DataFrame，必须包含基础OHLCV列
        index_cache: 可选，指数数据缓存（通过 load_index_data() 加载）

    Returns:
        包含所有特征的DataFrame

    Note:
        无未来函数。所有特征仅使用signal_date及之前的历史数据。
        调用顺序: calculate_kdj -> calculate_kdj_derivatives -> compute_technical_features -> calculate_all_features

    Example:
        >>> from quant_toolkit.features.technical import calculate_kdj, calculate_kdj_derivatives, compute_technical_features
        >>> df = calculate_kdj(df)
        >>> df = calculate_kdj_derivatives(df)
        >>> df = compute_technical_features(df)
        >>> df = calculate_all_features(df, index_cache=index_cache)
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # v2新增特征
    df = calculate_volume_depth_features(df)
    df = pd.DataFrame(df)
    df = calculate_market_cap_features(df)
    df = pd.DataFrame(df)
    df = calculate_key_candlestick_features(df)
    df = pd.DataFrame(df)

    # 指数特征（可选）
    if index_cache is not None:
        symbol = df.get("symbol", pd.Series([""])).iloc[0]
        df = add_index_features(df, index_cache, symbol)

    # 清理inf/nan
    for col in df.select_dtypes(include=[np.number]).columns:
        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0.0
            df[col] = arr

    return df


def get_v2_feature_columns() -> list:
    """返回v2新增特征列名（不含v1）"""
    return [
        # 第1组: 成交量深度 (23个)
        "amount_ratio_3", "amount_ratio_5", "amount_ratio_10", "amount_ratio_20",
        "amount_percentile_20", "amount_percentile_60",
        "high_volume_days_5", "high_volume_days_10", "high_volume_days_20",
        "low_volume_days_5", "low_volume_days_10",
        "up_volume_pressure_5", "up_volume_pressure_10", "up_volume_pressure_20",
        "down_volume_pressure_5", "down_volume_pressure_10", "down_volume_pressure_20",
        "price_volume_impulse_5", "price_volume_impulse_10", "price_volume_impulse_20",
        "high_volume_upbar_ratio_10", "high_volume_downbar_ratio_10",
        "low_vol_surge_ratio",
        # 第2组: 市值 (2个)
        "log_market_cap", "market_cap_bucket",
        # 第3组: 关键K线 (28个)
        "largest_up_day_return", "largest_up_day_amount_ratio",
        "largest_up_day_body_pct", "largest_up_day_upper_shadow",
        "largest_up_day_lower_shadow", "days_since_largest_up_day",
        "largest_down_day_return", "largest_down_day_amount_ratio",
        "largest_down_day_body_pct", "largest_down_day_upper_shadow",
        "largest_down_day_lower_shadow", "days_since_largest_down_day",
        "largest_volume_up_day_return", "largest_volume_up_day_body_pct",
        "largest_volume_up_day_close_position", "days_since_largest_volume_up_day",
        "largest_volume_down_day_return", "largest_volume_down_day_body_pct",
        "largest_volume_down_day_close_position", "days_since_largest_volume_down_day",
        "has_breakout_day", "days_since_breakout_day",
        "breakout_day_return", "breakout_day_amount_ratio",
        "has_breakdown_day", "days_since_breakdown_day",
        "breakdown_day_return", "breakdown_day_amount_ratio",
        # 第4组: 指数走势 (18个)
        "index_ret_1", "index_ret_5", "index_ret_10",
        "index_close_to_ma20", "index_close_to_ma60",
        "index_pos_20", "index_pos_60",
        "index_volatility_10",
        "small_vs_large_ret5", "mid_vs_large_ret5", "growth_vs_bluechip_ret5",
        "index_up_pct_1", "index_up_pct_2", "index_up_pct_3",
        "stock_vs_hs300_ret5", "stock_vs_zz1000_ret5",
        "stock_vs_hs300_ret20", "stock_vs_zz1000_ret20",
    ]


def get_feature_columns() -> list:
    """返回所有特征列名 (v1: 35个 + v2: 71个 = 106个)"""
    return get_v1_feature_columns() + get_v2_feature_columns()
