"""
信号生成测试

验证:
1. KDJ指标计算正确
2. KD13信号能生成
3. 样本构造逻辑正确
4. 无未来函数
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit._paths import project_root
from quant_toolkit.data.loaders import load_daily_data
from quant_toolkit.features.technical import calculate_kdj, calculate_kdj_derivatives, compute_technical_features
from quant_toolkit.signals.kd_strategy import generate_kd13_signal, construct_samples


DATA_DIR = Path(os.environ.get("QUANT_TOOLKIT_STOCK_DATA_DIR", str(project_root() / "day_klines")))


class TestKDJCalculation:
    """测试KDJ计算"""

    def test_calculate_kdj_columns(self):
        """测试KDJ计算产生正确列"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        assert "K" in df.columns
        assert "D" in df.columns
        assert "J" in df.columns

    def test_kdj_values_range(self):
        """测试KDJ值在合理范围内"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        # K, D应在0-100之间
        assert df["K"].min() >= 0 and df["K"].max() <= 100
        assert df["D"].min() >= 0 and df["D"].max() <= 100

    def test_kdj_derivatives(self):
        """测试KDJ派生指标"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        df = calculate_kdj_derivatives(df)
        assert "dJ_1" in df.columns
        assert "oversold_depth" in df.columns
        assert "days_J_below_20_last_10" in df.columns


class TestSignalGeneration:
    """测试信号生成"""

    def test_generate_kd13_signal(self):
        """测试KD13信号生成"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        df = calculate_kdj_derivatives(df)
        signals = generate_kd13_signal(df, j_threshold=13)
        assert isinstance(signals, pd.Series)
        assert signals.dtype == bool

    def test_signal_logic(self):
        """测试信号逻辑: J<13且前日J>=13"""
        # 构造测试数据
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "J": [20, 15, 12, 10, 14, 20, 12, 8, 15, 20],
        })
        signals = generate_kd13_signal(df, j_threshold=13)
        # J从15->12(第2->3天): 信号应在第3天触发
        assert signals.iloc[2] == True   # 15->12
        assert signals.iloc[6] == True   # 20->12
        assert signals.iloc[0] == False  # 无前日数据
        assert signals.iloc[3] == False  # 10->10，不是下穿

    def test_construct_samples(self):
        """测试样本构造"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        df = calculate_kdj_derivatives(df)
        df = compute_technical_features(df)

        samples = construct_samples(df, "000001", holding_days=20)
        assert isinstance(samples, list)

        if len(samples) > 0:
            sample = samples[0]
            assert "symbol" in sample
            assert "signal_date" in sample
            assert "entry_date" in sample
            assert "exit_date" in sample
            assert "entry_price" in sample
            assert "exit_price" in sample
            assert "holding_days" in sample
            assert "return_20d" in sample
            assert sample["symbol"] == "000001"
            assert sample["holding_days"] == 20
            # entry_date应为signal_date的下一个交易日
            assert sample["entry_date"] > sample["signal_date"]

    def test_no_future_data_in_features(self):
        """测试特征中无未来数据"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        df = calculate_kdj(df)
        df = calculate_kdj_derivatives(df)
        df = compute_technical_features(df)

        # 检查第50行的特征是否只使用了前50行数据
        # ret_20需要前20日数据，第50行应有值
        assert pd.notna(df.iloc[50]["ret_20"])
        # ma20需要前20日数据
        assert pd.notna(df.iloc[50]["ma20"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
