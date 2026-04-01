"""
Integration test: QlibBacktester with real market data.
集成测试：使用真实 Qlib 美股数据验证回测引擎。

This test requires Qlib US data to be installed at ~/.qlib/qlib_data/us_data.
Skip gracefully if data is not available.
"""

import pytest
import numpy as np
from pathlib import Path

# Skip all tests if Qlib data is not available
QLIB_DATA_PATH = Path.home() / ".qlib" / "qlib_data" / "us_data"
SKIP_REASON = f"Qlib US data not found at {QLIB_DATA_PATH}"
pytestmark = pytest.mark.skipif(not QLIB_DATA_PATH.exists(), reason=SKIP_REASON)

from src.qlib_backtester import QlibBacktester, BacktestConfig, BacktestResult, create_backtest_objective


# ── 测试用因子代码 ──

MOMENTUM_FACTOR = """
import numpy as np
import pandas as pd

def calculate_factor(df):
    \"\"\"20-day momentum factor: pct_change over 20 days.\"\"\"
    return df['close'].pct_change(20)
"""

VOLUME_FACTOR = """
import numpy as np
import pandas as pd

def calculate_factor(df):
    \"\"\"Volume-price divergence: volume z-score * price direction.\"\"\"
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    vol_zscore = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    price_dir = np.sign(df['close'].pct_change(5))
    return vol_zscore * price_dir
"""

REVERSAL_FACTOR = """
import numpy as np
import pandas as pd

def calculate_factor(df):
    \"\"\"5-day mean reversion factor: negative short-term return.\"\"\"
    return -df['close'].pct_change(5)
"""

BAD_FACTOR = """
def calculate_factor(df):
    raise ValueError("Intentional error for testing")
"""

CONSTANT_FACTOR = """
import pandas as pd

def calculate_factor(df):
    \"\"\"Returns constant value — should produce zero IC.\"\"\"
    return pd.Series(1.0, index=df.index)
"""


@pytest.fixture(scope="module")
def backtester():
    """Create a backtester with small stock universe for fast tests."""
    config = BacktestConfig(
        eval_start="2020-01-01",
        eval_end="2020-06-01",
        max_stocks=30,  # 小范围加速测试
    )
    return QlibBacktester(config)


class TestQlibBacktester:
    """Test real backtesting with Qlib data."""

    def test_momentum_factor_produces_real_ic(self, backtester):
        """动量因子应产出非零 IC"""
        result = backtester.run(MOMENTUM_FACTOR)
        assert result.error is None, f"Backtest failed: {result.error}"
        assert result._simulated is False
        assert result.n_stocks > 0
        assert result.n_days > 0
        assert not np.isnan(result.IC)
        assert result.IC != 0.0  # 动量因子不太可能精确为 0
        print(f"Momentum IC={result.IC:.4f} ICIR={result.ICIR:.4f} "
              f"Sharpe={result.sharpe_ratio:.4f} ({result.n_stocks} stocks, {result.n_days} days)")

    def test_volume_factor(self, backtester):
        """Volume-price 因子应成功运行"""
        result = backtester.run(VOLUME_FACTOR)
        assert result.error is None, f"Backtest failed: {result.error}"
        assert result.n_stocks > 0
        assert not np.isnan(result.IC)

    def test_reversal_factor(self, backtester):
        """反转因子应成功运行"""
        result = backtester.run(REVERSAL_FACTOR)
        assert result.error is None, f"Backtest failed: {result.error}"
        assert not np.isnan(result.IC)

    def test_bad_factor_returns_error(self, backtester):
        """错误因子应返回 error，不崩溃"""
        result = backtester.run(BAD_FACTOR)
        assert result.error is not None
        assert np.isnan(result.IC)

    def test_constant_factor_low_ic(self, backtester):
        """常数因子的 IC 应接近 0"""
        result = backtester.run(CONSTANT_FACTOR)
        # 常数因子可能报错（rank correlation undefined）或 IC=0
        if result.error is None:
            assert abs(result.IC) < 0.05, f"Constant factor IC should be ~0, got {result.IC}"

    def test_result_to_dict(self, backtester):
        """to_dict() 应返回完整指标字典"""
        result = backtester.run(MOMENTUM_FACTOR)
        d = result.to_dict()
        assert "IC" in d
        assert "ICIR" in d
        assert "annual_return" in d
        assert "max_drawdown" in d
        assert "sharpe_ratio" in d
        assert d["_simulated"] is False

    def test_data_caching(self, backtester):
        """第二次调用应使用缓存数据（更快）"""
        import time
        # First run loads data
        backtester.run(MOMENTUM_FACTOR)
        # Second run should use cache
        start = time.time()
        backtester.run(REVERSAL_FACTOR)
        elapsed = time.time() - start
        # Cached run should be fast (< 10s for 30 stocks)
        assert elapsed < 30, f"Cached run too slow: {elapsed:.1f}s"


class TestBacktestObjective:
    """Test the objective function for ParameterOptimizer."""

    def test_objective_returns_float(self, backtester):
        """目标函数应返回 float"""
        objective = create_backtest_objective(backtester)
        score = objective(MOMENTUM_FACTOR)
        assert isinstance(score, float)
        assert score != -999.0  # 成功的因子不应返回错误分数

    def test_objective_bad_code_returns_penalty(self, backtester):
        """失败的因子应返回惩罚分数"""
        objective = create_backtest_objective(backtester)
        score = objective(BAD_FACTOR)
        assert score == -999.0
