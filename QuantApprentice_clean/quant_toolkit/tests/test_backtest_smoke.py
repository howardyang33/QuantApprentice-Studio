"""
回测引擎冒烟测试

验证:
1. 回测能跑通
2. 输出指标中包含收益、最大回撤、Sharpe
3. 净值曲线合理
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_toolkit.backtest.engine import run_backtest
from quant_toolkit.backtest.metrics import compute_backtest_metrics


class TestBacktestSmoke:
    """回测引擎冒烟测试"""

    def _create_mock_signals(self, n: int = 100) -> pd.DataFrame:
        """创建模拟信号数据"""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        np.random.seed(42)

        signals = []
        for i in range(0, n - 20, 5):  # 每5天一个信号
            entry_date = dates[i]
            exit_date = dates[i + 19]
            ret = np.random.normal(0.02, 0.05)  # 随机收益
            signals.append({
                "symbol": f"{600000 + i:06d}",
                "signal_date": dates[max(0, i - 1)],
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 10.0,
                "exit_price": 10.0 * (1 + ret),
                "holding_days": 20,
                "return_20d": ret,
            })

        return pd.DataFrame(signals)

    def test_backtest_runs(self):
        """测试回测能正常跑通"""
        signals = self._create_mock_signals()
        result = run_backtest(signals, config={"lock_days": 20})

        assert "equity_curve" in result
        assert "trades" in result
        assert "metrics" in result

    def test_metrics_contains_required_fields(self):
        """测试指标包含必需字段"""
        signals = self._create_mock_signals()
        result = run_backtest(signals, config={"lock_days": 20})
        metrics = result["metrics"]

        required_fields = ["total_return", "annualized_return", "max_drawdown", "sharpe_ratio"]
        for field in required_fields:
            assert field in metrics, f"缺少指标: {field}"
            assert isinstance(metrics[field], (int, float)), f"{field} 不是数值类型"

    def test_equity_curve_reasonable(self):
        """测试净值曲线合理"""
        signals = self._create_mock_signals()
        result = run_backtest(signals, config={"lock_days": 20})
        nav = result["equity_curve"]

        assert len(nav) > 0
        assert np.isfinite(nav.iloc[0])
        assert nav.iloc[0] > 0
        assert all(nav > 0)  # 净值不应为负

    def test_backtest_with_quintiles(self):
        """测试带分层的回测"""
        signals = self._create_mock_signals()
        # 添加quintile列
        signals["quintile"] = np.random.choice(["Q1", "Q2", "Q3", "Q4", "Q5"], len(signals))

        for strategy in ["Baseline", "Q5", "Q4+Q5"]:
            result = run_backtest(signals, config={"strategy": strategy, "lock_days": 20})
            assert len(result["equity_curve"]) > 0 or strategy == "Q5"

    def test_metrics_computation(self):
        """测试指标计算正确性"""
        # 构造已知净值曲线
        nav = pd.Series(
            [1.0, 1.1, 1.2, 1.15, 1.3],
            index=pd.date_range("2020-01-01", periods=5)
        )
        metrics = compute_backtest_metrics(nav)

        assert np.isclose(metrics["total_return"], 0.3)  # 1.3/1.0 - 1 = 0.3
        assert metrics["max_drawdown"] > 0  # 1.2->1.15有回撤
        assert metrics["sharpe_ratio"] != 0

    def test_empty_signals(self):
        """测试空信号处理"""
        empty_signals = pd.DataFrame(columns=["entry_date", "exit_date", "return_20d"])
        result = run_backtest(empty_signals, config={"lock_days": 20})

        assert len(result["equity_curve"]) == 0
        assert len(result["trades"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
