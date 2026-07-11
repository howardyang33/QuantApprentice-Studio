"""
数据加载测试

验证:
1. 数据能正确读取
2. 数据验证能发现异常
3. 路径参数化工作正常
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit._paths import project_root
from quant_toolkit.data.loaders import load_daily_data, validate_daily_data, load_universe
from quant_toolkit.data.schema import DailyDataSchema


DATA_DIR = Path(os.environ.get("QUANT_TOOLKIT_STOCK_DATA_DIR", str(project_root() / "day_klines")))


class TestDataLoader:
    """测试数据加载功能"""

    def test_load_daily_data_success(self):
        """测试成功加载单只股票数据"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        assert len(df) > 0
        assert "date" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns

    def test_load_daily_data_date_filter(self):
        """测试日期过滤功能"""
        df = load_daily_data("000001", data_dir=DATA_DIR, start_date="2020-01-01", end_date="2020-12-31")
        assert df is not None
        assert df["date"].min() >= pd.to_datetime("2020-01-01")
        assert df["date"].max() <= pd.to_datetime("2020-12-31")

    def test_load_daily_data_not_found(self):
        """测试文件不存在时返回None"""
        df = load_daily_data("999999", data_dir=DATA_DIR)
        assert df is None

    def test_load_daily_data_numeric_types(self):
        """测试数值列类型正确"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        for col in ["open", "close", "high", "low"]:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_validate_daily_data_ok(self):
        """测试数据验证通过"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        result = validate_daily_data(df)
        assert result["status"] == "ok"
        assert result["stats"]["n_rows"] > 0

    def test_validate_daily_data_missing_column(self):
        """测试缺少列时验证失败"""
        df = pd.DataFrame({"date": ["2020-01-01"], "open": [10.0]})
        result = validate_daily_data(df)
        assert result["status"] == "error"
        assert len(result["errors"]) > 0

    def test_load_universe_all(self):
        """测试加载全部股票列表"""
        symbols = load_universe(DATA_DIR, universe="all")
        assert len(symbols) > 0
        assert "000001" in symbols

    def test_load_universe_subset(self):
        """测试加载指定股票列表"""
        subset = ["000001", "000002"]
        symbols = load_universe(DATA_DIR, universe=subset)
        assert symbols == subset

    def test_schema_validation(self):
        """测试Schema验证"""
        df = load_daily_data("000001", data_dir=DATA_DIR)
        assert df is not None
        # 不应抛出异常
        assert DailyDataSchema.validate(df) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
