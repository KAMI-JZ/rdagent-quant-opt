"""
Tests for Investment Principles — 经典数学投资逻辑
"""

import sys
import os
import importlib.util

# 绕过 src/__init__.py 的重依赖链
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if "src" not in sys.modules:
    sys.modules["src"] = type(sys)("src")
    sys.modules["src"].__path__ = [_src_dir]
    sys.modules["src"].__package__ = "src"

spec = importlib.util.spec_from_file_location(
    "src.investment_principles", os.path.join(_src_dir, "investment_principles.py")
)
ip_mod = importlib.util.module_from_spec(spec)
sys.modules["src.investment_principles"] = ip_mod
spec.loader.exec_module(ip_mod)

kelly_criterion = ip_mod.kelly_criterion
mean_variance_optimize = ip_mod.mean_variance_optimize
risk_parity = ip_mod.risk_parity
black_litterman = ip_mod.black_litterman
drawdown_position_control = ip_mod.drawdown_position_control
compute_position_advice = ip_mod.compute_position_advice
KellyResult = ip_mod.KellyResult
MVOResult = ip_mod.MVOResult
RiskParityResult = ip_mod.RiskParityResult
BlackLittermanResult = ip_mod.BlackLittermanResult
DrawdownControlResult = ip_mod.DrawdownControlResult
PositionAdvice = ip_mod.PositionAdvice

import pytest
import numpy as np


# ──────────────────────────────────────────────
# Kelly Criterion Tests
# ──────────────────────────────────────────────

class TestKellyCriterion:
    def test_positive_edge(self):
        """胜率55%、盈亏比1.5 → 正期望，有仓位"""
        result = kelly_criterion(0.55, 1.5)
        assert result.full_kelly > 0
        assert result.fraction > 0
        assert result.edge > 0

    def test_negative_edge(self):
        """胜率30%、盈亏比1.0 → 负期望，不下注"""
        result = kelly_criterion(0.30, 1.0)
        assert result.full_kelly < 0
        assert result.fraction == 0.0
        assert result.edge < 0

    def test_half_kelly_is_half(self):
        """半Kelly = 完整Kelly * 0.5"""
        result = kelly_criterion(0.6, 2.0, kelly_fraction=0.5)
        assert result.half_kelly == pytest.approx(result.full_kelly * 0.5, abs=1e-6)

    def test_max_position_cap(self):
        """仓位不超过 max_position"""
        result = kelly_criterion(0.8, 3.0, max_position=0.10)
        assert result.fraction <= 0.10

    def test_breakeven_edge(self):
        """胜率50%、盈亏比1.0 → 零期望"""
        result = kelly_criterion(0.50, 1.0)
        assert abs(result.edge) < 1e-6
        assert result.fraction == 0.0

    def test_win_rate_clamped(self):
        """胜率超界被截断"""
        result = kelly_criterion(1.5, 2.0)
        assert result.full_kelly > 0  # p=1.0

    def test_explanation_present(self):
        """结果包含中文解释"""
        result = kelly_criterion(0.55, 1.5)
        assert len(result.explanation) > 0


# ──────────────────────────────────────────────
# Mean-Variance Optimization Tests
# ──────────────────────────────────────────────

class TestMeanVariance:
    def setup_method(self):
        self.returns = np.array([0.10, 0.05, 0.03])
        self.cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01],
        ])

    def test_weights_sum_to_one(self):
        """权重和为1"""
        result = mean_variance_optimize(self.returns, self.cov)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_max_sharpe(self):
        """最大 Sharpe 模式返回正 Sharpe"""
        result = mean_variance_optimize(self.returns, self.cov, risk_free_rate=0.02)
        assert result.sharpe_ratio > 0

    def test_target_return(self):
        """指定目标收益"""
        result = mean_variance_optimize(self.returns, self.cov, target_return=0.06)
        # 组合收益应接近目标
        assert abs(result.expected_return - 0.06) < 0.02

    def test_two_assets(self):
        """两资产简单情况"""
        ret = np.array([0.10, 0.05])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        result = mean_variance_optimize(ret, cov)
        assert len(result.weights) == 2
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_explanation_has_stats(self):
        """解释包含关键统计"""
        result = mean_variance_optimize(self.returns, self.cov)
        assert "Markowitz" in result.explanation
        assert "Sharpe" in result.explanation


# ──────────────────────────────────────────────
# Risk Parity Tests
# ──────────────────────────────────────────────

class TestRiskParity:
    def setup_method(self):
        self.cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01],
        ])

    def test_weights_sum_to_one(self):
        result = risk_parity(self.cov)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_risk_contributions_roughly_equal(self):
        """风险贡献应接近均等 (1/n)"""
        result = risk_parity(self.cov)
        target = 1.0 / 3
        for rc in result.risk_contributions:
            assert abs(rc - target) < 0.05  # 允许5%偏差

    def test_higher_vol_gets_lower_weight(self):
        """高波动资产获得更低权重"""
        result = risk_parity(self.cov)
        # asset 0 vol=0.20, asset 2 vol=0.10 → weight[0] < weight[2]
        assert result.weights[0] < result.weights[2]

    def test_positive_volatility(self):
        result = risk_parity(self.cov)
        assert result.portfolio_volatility > 0

    def test_two_assets_equal_vol(self):
        """两资产等波动率 → 等权"""
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        result = risk_parity(cov)
        assert abs(result.weights[0] - 0.5) < 0.01
        assert abs(result.weights[1] - 0.5) < 0.01


# ──────────────────────────────────────────────
# Black-Litterman Tests
# ──────────────────────────────────────────────

class TestBlackLitterman:
    def setup_method(self):
        self.market_weights = np.array([0.5, 0.3, 0.2])
        self.cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01],
        ])

    def test_no_views_returns_equilibrium(self):
        """无观点 → 返回市场均衡"""
        result = black_litterman(self.market_weights, self.cov, views=[])
        np.testing.assert_array_almost_equal(result.weights, self.market_weights)
        assert "无主观观点" in result.explanation

    def test_bullish_view_increases_weight(self):
        """看涨某资产 → 权重增加"""
        views = [{"asset": 2, "return": 0.15, "confidence": 0.8}]
        result = black_litterman(self.market_weights, self.cov, views=views)
        # Asset 2 权重应大于均衡权重
        assert result.weights[2] > self.market_weights[2]

    def test_weights_sum_to_one(self):
        views = [{"asset": 0, "return": 0.12, "confidence": 0.6}]
        result = black_litterman(self.market_weights, self.cov, views=views)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_posterior_differs_from_prior(self):
        """有观点时后验收益应不同于先验"""
        views = [{"asset": 1, "return": 0.20, "confidence": 0.9}]
        result = black_litterman(self.market_weights, self.cov, views=views)
        assert not np.allclose(result.posterior_returns, result.prior_returns)

    def test_multiple_views(self):
        """多个观点"""
        views = [
            {"asset": 0, "return": 0.15, "confidence": 0.7},
            {"asset": 2, "return": 0.02, "confidence": 0.5},
        ]
        result = black_litterman(self.market_weights, self.cov, views=views)
        assert len(result.weights) == 3
        assert result.expected_return != 0


# ──────────────────────────────────────────────
# Drawdown Control Tests
# ──────────────────────────────────────────────

class TestDrawdownControl:
    def test_no_drawdown(self):
        """无回撤 → 满仓"""
        curve = [100, 101, 102, 103, 104]
        result = drawdown_position_control(curve)
        assert result.position_scale == 1.0
        assert not result.is_halted

    def test_mild_drawdown_within_limit(self):
        """小回撤在安全线内 → 满仓"""
        curve = [100, 105, 103]  # dd = -1.9%
        result = drawdown_position_control(curve, max_drawdown_limit=0.20)
        assert result.position_scale == 1.0

    def test_moderate_drawdown_reduces_position(self):
        """中等回撤 → 缩减仓位"""
        curve = [100, 110, 85]  # dd = -22.7%
        result = drawdown_position_control(curve, max_drawdown_limit=0.20, halt_drawdown=0.30)
        assert 0 < result.position_scale < 1.0

    def test_severe_drawdown_halts(self):
        """严重回撤 → 停止交易"""
        curve = [100, 110, 70]  # dd = -36.4%
        result = drawdown_position_control(curve, halt_drawdown=0.30)
        assert result.position_scale == 0.0
        assert result.is_halted

    def test_exponential_scaling(self):
        """指数缩放比线性更激进"""
        curve = [100, 110, 85]  # dd = -22.7%
        linear = drawdown_position_control(curve, scaling_method="linear")
        expo = drawdown_position_control(curve, scaling_method="exponential")
        assert expo.position_scale < linear.position_scale

    def test_short_curve(self):
        """数据不足 → 满仓"""
        result = drawdown_position_control([100])
        assert result.position_scale == 1.0

    def test_drawdown_value_is_negative(self):
        """回撤值为负"""
        curve = [100, 90]
        result = drawdown_position_control(curve)
        assert result.current_drawdown < 0


# ──────────────────────────────────────────────
# Position Advice (综合建议) Tests
# ──────────────────────────────────────────────

class TestPositionAdvice:
    def test_basic_advice(self):
        """基本建议 (只有 Kelly)"""
        advice = compute_position_advice(win_rate=0.55, win_loss_ratio=1.5)
        assert advice.recommended_size > 0
        assert advice.method_used == "Kelly"

    def test_advice_with_drawdown(self):
        """有回撤控制的建议"""
        curve = [100, 105, 95]  # 小回撤
        advice = compute_position_advice(
            win_rate=0.55, win_loss_ratio=1.5, equity_curve=curve,
        )
        assert "DrawdownControl" in advice.method_used

    def test_advice_halted_by_drawdown(self):
        """回撤触发暂停"""
        curve = [100, 110, 65]  # 大回撤
        advice = compute_position_advice(
            win_rate=0.60, win_loss_ratio=2.0, equity_curve=curve,
        )
        assert advice.recommended_size == 0.0

    def test_negative_edge_zero_position(self):
        """负期望 → 不交易"""
        advice = compute_position_advice(win_rate=0.30, win_loss_ratio=0.8)
        assert advice.recommended_size == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
