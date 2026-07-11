# -*- coding: utf-8 -*-
"""
净值曲线回测 (NAV Curve Backtest)

核心逻辑：
  - 与 enhanced_yearly_holdout 完全相同的 walk-forward 方式训练 XGBoost
  - 用训练集阈值对测试集划分 Q1~Q5
  - 策略：选定档位的所有信号，在各自 entry_date 均等买入，exit_date 卖出，无损耗
  - 净值曲线：每个交易日统计当日所有持仓的组合净值变化

输出：7张图（Baseline/Q1/Q2/Q3/Q4/Q5/Q4+Q5），每张图包含5d/10d/15d/20d + 沪深300 共5条线
额外：Q5图上增加一条 "Q5_Top4_MultiQ 5d" 曲线（5d Q5中按4期Qx加权评分选前4，T+5卖出，批次锁仓）

================================================================================
固定批次滚动窗口法（Fixed Batch Rolling Window）
================================================================================

一、批次划分逻辑（Batch Allocation）
------------------------------------
1. 批次数量 = 持有期天数（lock_days）
   - 5d持有期 → 5个槽位（slot_0 ~ slot_4）
   - 10d持有期 → 10个槽位（slot_0 ~ slot_9）
   - 以此类推...

2. 槽位分配规则（循环分配）
   - 按交易日的出现顺序依次分配槽位
   - 第1个交易日 → slot_0
   - 第2个交易日 → slot_1
   - ...
   - 第N+1个交易日 → slot_0（循环回到起点）
   - 公式：slot_id = (交易日序号) % n_batches

3. 每个槽位的仓位占比
   - 每个槽位占总仓位的 1/n_batches
   - 例如：5d持有期 → 每个槽位占20%仓位

二、批次内信号处理（Intra-Batch Signal Handling）
-------------------------------------------------
1. 同一交易日的所有信号作为一组
   - 这些信号共享同一个槽位
   - 槽位内的信号等权分配该槽位的仓位

2. 权重计算
   - 假设某日有M个信号，分配到slot_k
   - 每个信号的权重 = (1/n_batches) / M = batch_size / M
   - 例如：5d持有期，某日有4个信号
     → 每个信号权重 = 20% / 4 = 5%

3. 收益分布方式（线性分布）
   - 将信号的总收益均匀分布到持有期的每一天
   - 每日贡献 = 总收益 / 持有天数 × 信号权重
   - 例如：某信号收益10%，持有5天，权重5%
     → 每日贡献 = 10% / 5 × 5% = 0.1%

三、组合净值计算（Portfolio NAV Calculation）
--------------------------------------------
1. 每日组合收益 = sum(所有活跃槽位当日的收益贡献)
   - 遍历所有槽位，累加每个槽位在该日的收益

2. 净值累积
   - NAV[t] = NAV[t-1] × (1 + portfolio_ret[t])
   - 初始净值 = 1.0

四、可选的标的去重（Optional Symbol Deduplication）
--------------------------------------------------
- enable_dedup=True 时启用
- 如果新信号与当前槽位已持有的标的重复，则过滤掉该信号
- 目的：避免同一标的在同一槽位重复持仓

五、示例说明（5d持有期）
-----------------------
交易日序列：
  Day 1: 信号A1, A2 → slot_0（各占10%仓位，共20%）
  Day 2: 信号B1, B2, B3 → slot_1（各占6.67%仓位，共20%）
  Day 3: 信号C1 → slot_2（占20%仓位）
  Day 4: 信号D1, D2 → slot_3（各占10%仓位，共20%）
  Day 5: 信号E1, E2, E3, E4 → slot_4（各占5%仓位，共20%）
  Day 6: 信号F1, F2 → slot_0（循环，各占10%仓位，共20%）
  ...

注意：
  - 每个槽位的仓位固定在20%，不随信号数量变化
  - 信号越多，单个信号的权重越小（批次内等权）
  - 槽位之间独立，互不影响
================================================================================
"""

import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except ImportError:  # optional for replay-only consumers
    xgb = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
from .._paths import env_path, project_root

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
PROJECT_ROOT = env_path("QUANT_PROJECT_ROOT", project_root())
INDEX_FILE = env_path("NAV_CURVE_HS300_INDEX_FILE", PROJECT_ROOT / 'index_klines' / '000300.csv')   # 沪深300
OUTPUT_DIR = env_path("NAV_CURVE_OUTPUT_DIR", PROJECT_ROOT / 'reports' / 'nav_curve_backtest')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDING_DAYS_LIST = [5, 10, 15, 20]
HOLDING_DIRS = {
    5:  PROJECT_ROOT / 'outputs_hold_5d',
    10: PROJECT_ROOT / 'outputs_hold_10d',
    15: PROJECT_ROOT / 'outputs_hold_15d',
    20: PROJECT_ROOT / 'outputs_hold_20d',
}

# ==================== XGBoost 配置 ====================
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': 4,
    'verbosity': 0,
}

STRATEGY_LABELS = ['Baseline', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q4+Q5']

# 各策略在图上的颜色
STRATEGY_COLORS = {
    'Baseline': '#888888',
    'Q1':       '#e74c3c',
    'Q2':       '#e67e22',
    'Q3':       '#f1c40f',
    'Q4':       '#2ecc71',
    'Q5':       '#27ae60',
    'Q4+Q5':    '#1a6b38',
}

# 持有期线型区分
HOLDING_LINESTYLES = {5: '-', 10: '--', 15: '-.', 20: ':'}
HOLDING_LINEWIDTHS = {5: 1.6, 10: 1.6, 15: 1.8, 20: 1.8}

# 沪深300基准线
HS300_COLOR = '#2980b9'
HS300_LINEWIDTH = 2.5

# Q5_Top4_MultiQ 曲线样式
Q5_TOP4_MULTIQ_COLOR = '#e91e63'
Q5_TOP4_MULTIQ_LINEWIDTH = 2.5


# ==================== 工具函数 ====================

def compute_train_thresholds(y_pred_train: np.ndarray) -> dict:
    pct = [20, 40, 60, 80]
    boundaries = np.percentile(y_pred_train, pct)
    return {
        'Q1_upper': boundaries[0],
        'Q2_upper': boundaries[1],
        'Q3_upper': boundaries[2],
        'Q4_upper': boundaries[3],
    }


def assign_quintile_by_thresholds(y_pred: np.ndarray, thresholds: dict) -> np.ndarray:
    labels = np.full(len(y_pred), 'Q5', dtype=object)
    labels[y_pred < thresholds['Q4_upper']] = 'Q4'
    labels[y_pred < thresholds['Q3_upper']] = 'Q3'
    labels[y_pred < thresholds['Q2_upper']] = 'Q2'
    labels[y_pred < thresholds['Q1_upper']] = 'Q1'
    return labels


def load_hs300() -> pd.Series:
    """加载沪深300收盘价，返回 pd.Series(index=date str)"""
    df = pd.read_csv(INDEX_FILE, parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    return df['close']


def load_samples(holding_days: int) -> pd.DataFrame:
    """加载指定持有期的样本文件"""
    path = HOLDING_DIRS[holding_days] / 'trade_samples_full.csv'
    print(f'  加载 {holding_days}d 样本: {path} ...', end=' ', flush=True)
    df = pd.read_csv(path, parse_dates=['signal_date', 'entry_date', 'exit_date'])
    df['year'] = df['signal_date'].dt.year
    print(f'{len(df):,} 条')
    return df


def build_scored_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward 打分：每年用之前所有年份训练，对当年测试集预测+划档。
    返回附加了 quintile 和 pred_score 列的 DataFrame（只包含被验证的年份行）。
    
    pred_score 为模型对该样本的原始预测值（不包含任何真实收益信息），
    用于后续增强策略的同分排序 tie-break。
    """
    exclude_cols = {'symbol', 'signal_date', 'entry_date', 'exit_date',
                    'entry_price', 'exit_price', 'holding_days', 'return_20d'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_sorted = df.sort_values('signal_date').reset_index(drop=True)
    years = sorted(df_sorted['year'].unique())

    scored_parts = []

    if xgb is None:
        raise ImportError("xgboost is required for build_scored_test_set but is not installed in the current environment.")

    for year in years:
        train_df = df_sorted[df_sorted['year'] < year]
        test_df  = df_sorted[df_sorted['year'] == year].copy()

        if len(train_df) < 500 or len(test_df) < 50:
            print(f'    {year}: 样本不足，跳过')
            continue

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test  = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df['return_20d'].values

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        thresholds   = compute_train_thresholds(y_pred_train)

        y_pred_test  = model.predict(X_test)
        test_df['quintile'] = assign_quintile_by_thresholds(y_pred_test, thresholds)
        test_df['pred_score'] = y_pred_test

        scored_parts.append(test_df)
        n_q5 = (test_df['quintile'] == 'Q5').sum()
        print(f'    {year}: 测试 {len(test_df):,} 条, Q5={n_q5} 条')

    return pd.concat(scored_parts, ignore_index=True)


def build_q5_top4_signals(scored_sets: dict) -> pd.DataFrame:
    """
    Q5_Top4_MultiQ 策略信号筛选（无未来函数版本）：
      1. 取 5d 的 Q5 信号（quintile == 'Q5'）
      2. 对同一(symbol, signal_date)，查找 10d/15d/20d 的 quintile
      3. 计算多持有期平均 Qx 分数 multi_q_score（仅使用 quintile 档位信息）
      4. 按 signal_date 分组，组内排序：
         主键：multi_q_score 降序
         次键：pred_score_5d 降序（模型预测值，非真实收益）
      5. 每个 signal_date 取前 4 名
      6. 返回带 batch_id、rank_within_date、multi_q_score、pred_score 的 DataFrame
    
    严禁使用的字段（未来信息）：return_20d, exit_price, exit_date（排序时）
    """
    QX_SCORE = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'Q5': 5}
    FUTURE_COLS = {'return_20d', 'exit_price', 'exit_date'}

    scored_5d = scored_sets[5]
    
    # 自检断言：scored_sets[5] 包含 pred_score 列
    assert 'pred_score' in scored_5d.columns, \
        "scored_sets[5] 缺少 pred_score 列，请确认 build_scored_test_set 已更新"
    
    q5_5d = scored_5d[scored_5d['quintile'] == 'Q5'].copy()

    if len(q5_5d) == 0:
        print('  [Q5_Top4_MultiQ] 5d Q5 信号为空，跳过')
        return pd.DataFrame()

    # 构建跨持有期 quintile 查找表（仅查 quintile，不查收益）
    qx_lookup = {}
    for hd in [10, 15, 20]:
        s = scored_sets[hd]
        lookup = s.set_index(['symbol', 'signal_date'])['quintile'].to_dict()
        qx_lookup[hd] = lookup

    # Step 3: 计算多持有期平均 Qx 分数
    scores = []
    for _, row in q5_5d.iterrows():
        key = (row['symbol'], row['signal_date'])
        
        # 5d 自身 quintile 分数
        total = QX_SCORE.get(row['quintile'], 3)
        n_periods = 1
        
        # 跨持有期查找 quintile 并累加分数
        for hd in [10, 15, 20]:
            qx = qx_lookup[hd].get(key, None)
            if qx is not None:
                total += QX_SCORE.get(qx, 3)
                n_periods += 1
            # 该持有期没有此信号则跳过
        
        avg_score = total / n_periods if n_periods > 0 else 3
        scores.append(avg_score)

    q5_5d['multi_q_score'] = scores

    # Step 4-5: 按 signal_date 分组排序，取 Top4（仅用 quintile 信息 + pred_score 排序）
    top4_rows = []
    for sig_date, group in q5_5d.groupby('signal_date'):
        sorted_group = group.sort_values(
            ['multi_q_score', 'pred_score'],
            ascending=[False, False]
        )
        sorted_group['rank_within_date'] = range(1, len(sorted_group) + 1)
        top4_rows.append(sorted_group.head(4))

    if len(top4_rows) == 0:
        return pd.DataFrame()
    
    result = pd.concat(top4_rows).reset_index(drop=True)

    # Step 6: 生成 batch_id（每个 signal_date 一个 batch）
    sig_dates_unique = result['signal_date'].unique()
    date_to_batch = {d: i for i, d in enumerate(sig_dates_unique)}
    result['batch_id'] = result['signal_date'].map(date_to_batch)

    # 自检断言：排序未使用任何未来收益列
    sort_cols_used = {'multi_q_score', 'pred_score'}
    assert sort_cols_used.isdisjoint(FUTURE_COLS), \
        f"错误！排序中使用了未来收益列: {sort_cols_used & FUTURE_COLS}"

    # 自检断言：每个 batch 最多 4 只
    max_per_batch = result.groupby('batch_id').size().max()
    assert max_per_batch <= 4, f"错误！存在超过 4 只的 batch: max={max_per_batch}"

    # 自检断言：全部来自 5d Q5
    assert (result['quintile'] == 'Q5').all(), "错误！存在非 Q5 的信号"

    n_batches = result['batch_id'].nunique()
    avg_per_batch = len(result) / n_batches if n_batches > 0 else 0
    print(f'  [Q5_Top4_MultiQ] 5d Q5 信号: {len(q5_5d):,} 条 → '
          f'{n_batches} 批次 → {len(result)} 条股票 (平均 {avg_per_batch:.1f} 只/批)')
    print(f'    score 分布: min={result["multi_q_score"].min():.2f}, '
          f'max={result["multi_q_score"].max():.2f}, '
          f'mean={result["multi_q_score"].mean():.2f}')
    
    return result


def compute_nav_curve_top4_batches(top4_df: pd.DataFrame,
                                   all_trade_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Q5_Top4_MultiQ 批次净值曲线计算。

    正确逻辑：
      - 同一 batch（同 signal_date）最多 4 只，等权同时开仓
      - batch 互斥：一个 batch 未清仓时不开新 batch
      - 使用交易日长度（非自然日）计算等效日收益
      - 仅使用 5d 样本的真实收益（不混入 10d/15d/20d 收益）

    Args:
        top4_df: build_q5_top4_signals 输出的 DataFrame（含 batch_id 等）
        all_trade_dates: 交易日 DatetimeIndex

    Returns:
        净值 Series (index=all_trade_dates)
    """
    if len(top4_df) == 0:
        return pd.Series(dtype=float)

    n_dates = len(all_trade_dates)
    all_dates_arr = all_trade_dates.to_numpy().astype('datetime64[D]')
    
    # 构建 date -> index 映射（D 精度）
    date_to_idx = {}
    for i, d in enumerate(all_dates_arr):
        date_to_idx[int(np.int64(d))] = i

    # 按 batch_id（即 entry_date）排序
    sel = top4_df.copy().sort_values('entry_date').reset_index(drop=True)

    # 计算每只股票的交易日持有长度和等效日收益
    entry_indices = []
    exit_indices = []
    daily_equivs = []
    
    holding_trading_days_list = []
    
    for _, row in sel.iterrows():
        entry_dt = np.datetime64(row['entry_date'], 'D')
        entry_int = int(np.int64(entry_dt))
        entry_idx = date_to_idx.get(entry_int, -1)
        
        exit_dt = np.datetime64(row['exit_date'], 'D')
        exit_int = int(np.int64(exit_dt))
        exit_idx = date_to_idx.get(exit_int, -1)
        
        if entry_idx < 0 or exit_idx < 0 or entry_idx >= exit_idx:
            continue
        
        # 交易日持有长度（基于 trade_dates index 差）
        ht_days = exit_idx - entry_idx + 1
        if ht_days <= 0:
            ht_days = 1
        
        # 等效日收益：严格按交易日长度展开
        ret = float(row['return_20d'])
        de = (1.0 + ret) ** (1.0 / ht_days) - 1.0
        
        entry_indices.append(entry_idx)
        exit_indices.append(exit_idx)
        daily_equivs.append(de)
        holding_trading_days_list.append(ht_days)
    
    if len(entry_indices) == 0:
        return pd.Series(1.0, index=all_trade_dates)

    entry_arr = np.array(entry_indices, dtype=np.int32)
    exit_arr = np.array(exit_indices, dtype=np.int32)
    daily_r = np.array(daily_equivs, dtype=np.float64)

    # === 批次锁仓：按 batch_id 过滤 ===
    # 一个 batch 内的所有股票一起进入；当前 batch 未退出时不开启下一个 batch
    batch_ids = sel.loc[:len(entry_arr)-1]['batch_id'].values
    
    # 找出每个 batch 的最大退出索引
    batch_max_exit = {}
    for i, bid in enumerate(batch_ids):
        eidx = exit_arr[i]
        if bid not in batch_max_exit or eidx > batch_max_exit[bid]:
            batch_max_exit[bid] = eidx
    
    # 按顺序遍历 batch，应用互斥过滤
    active_batches = set()
    batch_order = []  # 保留激活的 batch 顺序
    next_allowed_idx = 0
    
    seen_batches = set()
    for i, bid in enumerate(batch_ids):
        if bid in seen_batches:
            continue  # 同一批次的第2~4只自动跟随
        seen_batches.add(bid)
        
        # 这批的第一只股票的 entry_idx 代表该批的开仓日
        first_entry_in_batch = entry_arr[i]
        
        if first_entry_in_batch >= next_allowed_idx:
            # 允许开此 batch
            batch_order.append(bid)
            next_allowed_idx = batch_max_exit[bid]  # 此 batch 退出后才允许下一批

    # 过滤出属于激活 batch 的信号
    valid_mask = np.array([bid in set(batch_order) for bid in batch_ids], dtype=bool)
    
    lo_v = entry_arr[valid_mask]
    hi_v = exit_arr[valid_mask]
    dr_v = daily_r[valid_mask]

    if len(lo_v) == 0:
        return pd.Series(1.0, index=all_trade_dates)

    # 向量化差分累积
    sum_diff = np.zeros(n_dates + 1, dtype=np.float64)
    cnt_diff = np.zeros(n_dates + 1, dtype=np.int32)

    np.add.at(sum_diff, lo_v, dr_v)
    np.add.at(sum_diff, hi_v, -dr_v)
    np.add.at(cnt_diff, lo_v, 1)
    np.add.at(cnt_diff, hi_v, -1)

    sum_ret = np.cumsum(sum_diff)[:n_dates]
    count_ret = np.cumsum(cnt_diff)[:n_dates]

    portfolio_ret = np.where(count_ret > 0, sum_ret / count_ret, 0.0)
    nav = np.cumprod(1.0 + portfolio_ret)

    return pd.Series(nav, index=all_trade_dates)


def compute_nav_curve_fast(scored_df: pd.DataFrame,
                            strategy: str,
                            all_trade_dates: pd.DatetimeIndex,
                            pre_filtered: pd.DataFrame = None,
                            lock_days: int = None,
                            enable_dedup: bool = False) -> pd.Series:
    """
    净值曲线计算（固定批次滚动窗口法）
    
    核心逻辑：
      - 固定N个批次（N=lock_days），每个批次占 1/N 仓位
      - 按交易日顺序滚动：第1天→批次1, 第2天→批次2, ..., 第N+1天→批次1（循环）
      - 批次内多信号等权分配该批次的仓位
      - 可选：过滤与已有持仓重复的标的
      
    Args:
        scored_df: 打分后的样本数据
        strategy: 策略名称 (Baseline/Q1-Q5/Q4+Q5)
        all_trade_dates: 交易日索引
        pre_filtered: 预过滤的数据（用于特殊策略如Q5_Top4）
        lock_days: 持有期天数，也作为批次数量
        enable_dedup: 是否启用标的去重（过滤与已有持仓重复的信号）
    """
    # 1. 筛选策略对应的信号
    if pre_filtered is not None:
        sel = pre_filtered.copy()
    elif strategy == 'Baseline':
        sel = scored_df.copy()
    elif strategy == 'Q4+Q5':
        sel = scored_df[scored_df['quintile'].isin(['Q4', 'Q5'])].copy()
    else:
        sel = scored_df[scored_df['quintile'] == strategy].copy()

    if len(sel) == 0:
        return pd.Series(dtype=float)

    # 2. 按 entry_date 排序
    sel = sel.sort_values('entry_date').reset_index(drop=True)
    
    # 3. 如果没有指定 lock_days，使用默认逻辑（向后兼容）
    if lock_days is None or lock_days <= 0:
        return _compute_nav_legacy(sel, all_trade_dates)
    
    # 4. 固定批次滚动窗口法
    n_batches = lock_days  # 批次数量 = 持有期
    batch_size = 1.0 / n_batches  # 每个批次占总仓位的比例
    
    n_dates = len(all_trade_dates)
    all_dates_arr = all_trade_dates.to_numpy().astype('datetime64[D]')
    
    # 构建 date -> index 映射
    date_to_idx = {}
    for i, d in enumerate(all_dates_arr):
        date_to_idx[int(np.int64(d))] = i
    
    # 5. 按 entry_date 分组，每个交易日的信号作为一个子批次
    # 子批次按交易日顺序分配到 N 个槽位（循环）
    # slot_daily_returns[slot_id, day_idx] = 该槽位在该日的等效日收益
    slot_daily_returns = np.zeros((n_batches, n_dates), dtype=np.float64)
    
    # 跟踪每个槽位的当前持仓（用于去重）
    slot_holdings = {i: set() for i in range(n_batches)}  # {slot_id: set(symbols)}
    
    # 按 entry_date 排序后遍历
    sorted_dates = sorted(sel['entry_date'].unique())
    
    for slot_idx, entry_date in enumerate(sorted_dates):
        # 计算该 entry_date 对应的槽位索引（循环分配）
        slot_id = slot_idx % n_batches
        
        # 获取该日的所有信号
        day_signals = sel[sel['entry_date'] == entry_date]
        
        if len(day_signals) == 0:
            continue
        
        # 如果需要去重，过滤掉已在该槽位持仓的标的
        if enable_dedup:
            day_signals = day_signals[~day_signals['symbol'].isin(slot_holdings[slot_id])]
            if len(day_signals) == 0:
                continue  # 该槽位无有效信号
        
        # === 核心修正：批次内先计算平均收益，再转换为等效日收益 ===
        
        # 1. 计算批次内所有信号的平均收益
        # 注意：这里假设所有信号的持有期相同（都是lock_days）
        signal_returns = day_signals['return_20d'].values
        avg_ret = np.mean(signal_returns)
        
        # 2. 批次总权重 = 该槽位的仓位占比（固定为batch_size）
        batch_total_weight = batch_size
        
        # 3. 取第一个信号的持有期作为代表（同一批次的信号持有期应该相同）
        first_signal = day_signals.iloc[0]
        entry_dt = np.datetime64(first_signal['entry_date'], 'D')
        exit_dt = np.datetime64(first_signal['exit_date'], 'D')
        
        entry_int = int(np.int64(entry_dt))
        exit_int = int(np.int64(exit_dt))
        
        entry_idx = date_to_idx.get(entry_int, -1)
        exit_idx = date_to_idx.get(exit_int, -1)
        
        if entry_idx < 0 or exit_idx < 0 or entry_idx > exit_idx:
            continue
        
        # 4. 计算持有天数（交易日）
        holding_days = exit_idx - entry_idx + 1
        if holding_days <= 0:
            holding_days = 1
        
        # 5. 将批次平均收益转换为等效日收益（复利方式）
        # 保证：(1 + daily_equiv)^holding_days = 1 + avg_ret
        daily_equiv = (1.0 + avg_ret) ** (1.0 / holding_days) - 1.0
        
        # 6. 每日贡献 = 等效日收益 × 批次总权重
        daily_contribution = daily_equiv * batch_total_weight
        
        # 7. 均匀分布到持有期的每一天
        for day_idx in range(entry_idx, exit_idx + 1):
            if day_idx < n_dates:
                slot_daily_returns[slot_id, day_idx] += daily_contribution
        
        # 8. 更新槽位持仓（用于去重）
        if enable_dedup:
            for _, signal in day_signals.iterrows():
                slot_holdings[slot_id].add(signal['symbol'])
    
    # 6. 计算组合净值
    # 每日组合收益 = sum(各槽位当日的收益)
    portfolio_ret = np.sum(slot_daily_returns, axis=0)
    
    # 7. 累积净值
    nav = np.cumprod(1.0 + portfolio_ret)
    
    return pd.Series(nav, index=all_trade_dates)


def _compute_nav_legacy(sel: pd.DataFrame, all_trade_dates: pd.DatetimeIndex) -> pd.Series:
    """
    旧版净值计算方法（向后兼容）
    无锁仓逻辑，所有信号独立开仓
    """
    if len(sel) == 0:
        return pd.Series(dtype=float)

    sel = sel.copy()
    sel['holding_days_actual'] = (sel['exit_date'] - sel['entry_date']).dt.days.clip(lower=1)
    sel['daily_equiv'] = (1.0 + sel['return_20d']) ** (1.0 / sel['holding_days_actual']) - 1.0

    n_dates = len(all_trade_dates)
    all_dates_arr = all_trade_dates.to_numpy().astype('datetime64[D]')
    date_to_idx = {}
    for i, d in enumerate(all_dates_arr):
        date_to_idx[int(np.int64(d))] = i

    sel = sel.sort_values('entry_date').reset_index(drop=True)

    entry_arr = sel['entry_date'].values.astype('datetime64[D]')
    exit_arr  = sel['exit_date'].values.astype('datetime64[D]')
    daily_r   = sel['daily_equiv'].values

    lo_idx = np.searchsorted(all_dates_arr, entry_arr, side='left')
    hi_idx = np.searchsorted(all_dates_arr, exit_arr,  side='left')

    sum_diff  = np.zeros(n_dates + 1, dtype=np.float64)
    cnt_diff  = np.zeros(n_dates + 1, dtype=np.int32)

    lo_idx = np.clip(lo_idx, 0, n_dates)
    hi_idx = np.clip(hi_idx, 0, n_dates)

    valid = lo_idx < hi_idx
    lo_v = lo_idx[valid]
    hi_v = hi_idx[valid]
    dr_v = daily_r[valid]

    np.add.at(sum_diff, lo_v, dr_v)
    np.add.at(sum_diff, hi_v, -dr_v)
    np.add.at(cnt_diff, lo_v,  1)
    np.add.at(cnt_diff, hi_v, -1)

    sum_ret   = np.cumsum(sum_diff)[:n_dates]
    count_ret = np.cumsum(cnt_diff)[:n_dates]

    portfolio_ret = np.where(count_ret > 0, sum_ret / count_ret, 0.0)
    nav = np.cumprod(1.0 + portfolio_ret)

    return pd.Series(nav, index=all_trade_dates)


def plot_nav_chart(strategy: str,
                   nav_curves: dict,       # {holding_days: pd.Series}
                   hs300_nav: pd.Series,
                   output_path: Path,
                   extra_lines: list = None):  # [(label, pd.Series, color, linewidth, linestyle), ...]
    """绘制单张净值曲线图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 沪深300
    hs300_common = hs300_nav.copy()
    ax.plot(hs300_nav.index, hs300_nav.values,
            color=HS300_COLOR, linewidth=HS300_LINEWIDTH,
            linestyle='-', label='沪深300', zorder=5, alpha=0.9)

    # 各持有期策略线
    for hd in HOLDING_DAYS_LIST:
        nav = nav_curves.get(hd)
        if nav is None or len(nav) == 0:
            continue
        ax.plot(nav.index, nav.values,
                color=STRATEGY_COLORS.get(strategy, '#333333'),
                linewidth=HOLDING_LINEWIDTHS[hd],
                linestyle=HOLDING_LINESTYLES[hd],
                label=f'{strategy} {hd}d',
                alpha=0.85, zorder=4)

    # 参考线
    ax.axhline(y=1.0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)

    # 额外曲线（如 Q5_Top4_MultiQ）
    if extra_lines:
        for label, nav, color, lw, ls in extra_lines:
            if nav is not None and len(nav) > 0:
                ax.plot(nav.index, nav.values,
                        color=color, linewidth=lw, linestyle=ls,
                        label=label, alpha=0.95, zorder=6)
                final_nav = nav.iloc[-1]
                ax.annotate(f'{final_nav:.2f}',
                            xy=(nav.index[-1], final_nav),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=9, color=color, fontweight='bold',
                            va='center')

    # 标题与标签
    strategy_desc = {
        'Baseline': '基础策略（全量信号）',
        'Q1': 'Q1档（预测分最低20%）',
        'Q2': 'Q2档',
        'Q3': 'Q3档',
        'Q4': 'Q4档',
        'Q5': 'Q5档（预测分最高20%）',
        'Q4+Q5': 'Q4+Q5档（实盘推荐）',
    }
    desc = strategy_desc.get(strategy, strategy)
    ax.set_title(f'KDJ评分模型净值曲线 — {desc}\n（Walk-forward 无损耗等权满仓，训练阈值划档）',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('净值（初始=1.0）', fontsize=11)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=9)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.grid(True, alpha=0.25, linewidth=0.7)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)

    # 最终净值注释
    for hd in HOLDING_DAYS_LIST:
        nav = nav_curves.get(hd)
        if nav is not None and len(nav) > 0:
            final_nav = nav.iloc[-1]
            ax.annotate(f'{final_nav:.2f}',
                        xy=(nav.index[-1], final_nav),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=STRATEGY_COLORS.get(strategy, '#333333'),
                        va='center')

    final_hs300 = hs300_nav.iloc[-1]
    ax.annotate(f'HS300: {final_hs300:.2f}',
                xy=(hs300_nav.index[-1], final_hs300),
                xytext=(5, 0), textcoords='offset points',
                fontsize=8, color=HS300_COLOR, va='center')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {output_path.name}')


def main():
    print('=' * 70)
    print('净值曲线回测 (NAV Curve Backtest)')
    print('=' * 70)
    print()

    # 1. 加载沪深300
    print('加载沪深300...')
    hs300_close = load_hs300()
    print(f'  日期范围: {hs300_close.index.min()} ~ {hs300_close.index.max()}, {len(hs300_close)} 条')

    # 2. 对每个持有期加载数据并 walk-forward 打分
    print()
    scored_sets = {}
    for hd in HOLDING_DAYS_LIST:
        print(f'处理 {hd}d 样本...')
        df = load_samples(hd)
        print(f'  Walk-forward 打分中...')
        scored_df = build_scored_test_set(df)
        scored_sets[hd] = scored_df
        print(f'  打分完成: {len(scored_df):,} 条 (年份 {scored_df["signal_date"].dt.year.min()}-{scored_df["signal_date"].dt.year.max()})')
        print()

    # 3. 确定回测日期范围（所有持有期的交集）
    min_entry = max(scored_sets[hd]['entry_date'].min() for hd in HOLDING_DAYS_LIST)
    max_exit  = min(scored_sets[hd]['exit_date'].max() for hd in HOLDING_DAYS_LIST)
    print(f'回测区间: {min_entry.date()} ~ {max_exit.date()}')

    # 以沪深300的交易日为基准日历
    hs300_idx = hs300_close.index  # DatetimeIndex
    mask = (hs300_idx >= min_entry) & (hs300_idx <= max_exit)
    trade_dates = hs300_idx[mask]
    print(f'交易日数: {len(trade_dates)}')
    print()

    # 4. 沪深300净值曲线（标准化到1）
    hs300_sub = hs300_close.loc[trade_dates]
    hs300_nav = hs300_sub / hs300_sub.iloc[0]

    # 5. 构建 Q5_Top4_MultiQ 信号集
    print('构建 Q5_Top4_MultiQ 信号集...')
    q5_top4_df = build_q5_top4_signals(scored_sets)
    print()

    # 6. 导出增强策略明细 CSV
    if len(q5_top4_df) > 0:
        export_cols = ['symbol', 'signal_date', 'entry_date', 'exit_date',
                       'quintile', 'pred_score', 'multi_q_score',
                       'rank_within_date', 'batch_id', 'return_20d']
        export_df = q5_top4_df[[c for c in export_cols if c in q5_top4_df.columns]].copy()
        csv_path = OUTPUT_DIR / 'q5_top4_multiq_signals.csv'
        export_df.to_csv(csv_path, index=False)
        print(f'  已导出增强策略明细: {csv_path.name} ({len(export_df)} 行)')

    # 7. 对每个策略，计算净值曲线并绘图
    print('\n计算净值曲线并绘图...')
    for strategy in STRATEGY_LABELS:
        print(f'\n[策略: {strategy}]')
        nav_curves = {}
        for hd in HOLDING_DAYS_LIST:
            print(f'  计算 {hd}d ...', end=' ', flush=True)
            # 修复：添加 lock_days 参数，防止信号叠加导致的虚高收益
            # lock_days=hd 确保前一批次卖出后才能开新仓，符合实盘资金约束
            nav = compute_nav_curve_fast(scored_sets[hd], strategy, trade_dates, lock_days=hd)
            nav_curves[hd] = nav
            if len(nav) > 0:
                print(f'终值={nav.iloc[-1]:.3f}')
            else:
                print('无数据')

        extra_lines = None
        # Q5 图上额外绘制 Q5_Top4_MultiQ 曲线（新版本：批次锁仓、无未来函数）
        if strategy == 'Q5' and len(q5_top4_df) > 0:
            print(f'  计算 Q5_Top4_MultiQ 5d (批次锁仓) ...', end=' ', flush=True)
            nav_top4 = compute_nav_curve_top4_batches(q5_top4_df, trade_dates)
            if len(nav_top4) > 0:
                print(f'终值={nav_top4.iloc[-1]:.3f}')
                extra_lines = [('Q5_Top4_MultiQ 5d', nav_top4, Q5_TOP4_MULTIQ_COLOR,
                                Q5_TOP4_MULTIQ_LINEWIDTH, '-')]
                
                # 额外自检：检查是否有同一持仓窗口重复开 batch
                n_batches = q5_top4_df['batch_id'].nunique()
                print(f'    [自检] 总批次数: {n_batches}, '
                      f'总信号数: {len(q5_top4_df)}, '
                      f'平均每批: {len(q5_top4_df)/n_batches:.2f}')
            else:
                print('无数据')

        fname = f'nav_{strategy.replace("+", "plus")}.png'
        plot_nav_chart(
            strategy=strategy,
            nav_curves=nav_curves,
            hs300_nav=hs300_nav,
            output_path=OUTPUT_DIR / fname,
            extra_lines=extra_lines,
        )

    print()
    print('=' * 70)
    print(f'所有图表已保存到: {OUTPUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    # 设置中文字体
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    main()
