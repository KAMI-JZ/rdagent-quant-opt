"""Tests for PositionGuard — 仓位风控测试"""

import importlib
import sys
import os
import types
import pytest

_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [_src_path]
    sys.modules["src"] = src_pkg

spec = importlib.util.spec_from_file_location("src.position_guard",
                                               os.path.join(_src_path, "position_guard.py"))
_mod = importlib.util.module_from_spec(spec)
sys.modules["src.position_guard"] = _mod
spec.loader.exec_module(_mod)

PositionGuard = _mod.PositionGuard
GuardConfig = _mod.GuardConfig
Position = _mod.Position
ViolationType = _mod.ViolationType


# ──────── Position Tests ────────

class TestPosition:
    def test_pnl_positive(self):
        p = Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=120)
        assert abs(p.pnl_pct - 0.20) < 1e-6

    def test_pnl_negative(self):
        p = Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=90)
        assert abs(p.pnl_pct - (-0.10)) < 1e-6

    def test_pnl_zero_entry(self):
        p = Position(ticker="AAPL", weight=0.05, entry_price=0, current_price=100)
        assert p.pnl_pct == 0.0

    def test_drawdown_from_peak(self):
        p = Position(ticker="AAPL", weight=0.05, peak_price=150, current_price=120)
        assert abs(p.drawdown_from_peak - (-0.20)) < 1e-6


# ──────── Stop Loss Tests ────────

class TestStopLoss:
    def setup_method(self):
        self.guard = PositionGuard(GuardConfig(stop_loss_pct=-0.08))

    def test_stop_loss_triggered(self):
        positions = [
            Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=91),  # -9%
        ]
        result = self.guard.check(positions)
        assert "AAPL" in result.forced_exits
        assert any(v.type == ViolationType.STOP_LOSS for v in result.violations)

    def test_stop_loss_not_triggered(self):
        positions = [
            Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=95),  # -5%
        ]
        result = self.guard.check(positions)
        assert "AAPL" not in result.forced_exits

    def test_stop_loss_no_entry_price(self):
        positions = [
            Position(ticker="AAPL", weight=0.05, entry_price=0, current_price=50),
        ]
        result = self.guard.check(positions)
        assert "AAPL" not in result.forced_exits  # can't calculate P&L


# ──────── Take Profit Tests ────────

class TestTakeProfit:
    def setup_method(self):
        self.guard = PositionGuard(GuardConfig(take_profit_pct=0.20))

    def test_take_profit_triggered(self):
        positions = [
            Position(ticker="TSLA", weight=0.05, entry_price=100, current_price=125),  # +25%
        ]
        result = self.guard.check(positions)
        assert "TSLA" in result.forced_exits
        assert any(v.type == ViolationType.TAKE_PROFIT for v in result.violations)

    def test_take_profit_not_triggered(self):
        positions = [
            Position(ticker="TSLA", weight=0.05, entry_price=100, current_price=115),  # +15%
        ]
        result = self.guard.check(positions)
        assert "TSLA" not in result.forced_exits


# ──────── Trailing Stop Tests ────────

class TestTrailingStop:
    def test_trailing_stop_triggered(self):
        guard = PositionGuard(GuardConfig(trailing_stop_pct=0.10))
        positions = [
            Position(ticker="GOOG", weight=0.05, entry_price=100,
                    current_price=108, peak_price=125),  # -13.6% from peak
        ]
        result = guard.check(positions)
        assert "GOOG" in result.forced_exits

    def test_trailing_stop_disabled(self):
        guard = PositionGuard(GuardConfig(trailing_stop_pct=0.0))
        positions = [
            Position(ticker="GOOG", weight=0.05, entry_price=100,
                    current_price=108, peak_price=125),
        ]
        result = guard.check(positions)
        assert "GOOG" not in result.forced_exits


# ──────── Holding Period Tests ────────

class TestHoldingPeriod:
    def setup_method(self):
        self.guard = PositionGuard(GuardConfig(min_holding_days=3, max_holding_days=60))

    def test_max_holding_exceeded(self):
        positions = [
            Position(ticker="MSFT", weight=0.05, holding_days=65),
        ]
        result = self.guard.check(positions)
        assert "MSFT" in result.forced_exits
        assert any(v.type == ViolationType.MAX_HOLDING for v in result.violations)

    def test_min_holding_violation(self):
        positions = [
            Position(ticker="MSFT", weight=0.05, holding_days=1),
        ]
        result = self.guard.check(positions)
        assert "MSFT" not in result.forced_exits  # not forced out, just flagged
        assert any(v.type == ViolationType.MIN_HOLDING for v in result.violations)

    def test_normal_holding(self):
        positions = [
            Position(ticker="MSFT", weight=0.05, holding_days=10),
        ]
        result = self.guard.check(positions)
        assert result.n_violations == 0


# ──────── Position Limits Tests ────────

class TestPositionLimits:
    def test_max_position_capped(self):
        guard = PositionGuard(GuardConfig(max_single_position=0.10))
        positions = [
            Position(ticker="NVDA", weight=0.15),
        ]
        result = guard.check(positions)
        assert result.adjusted_positions[0].weight == 0.10
        assert any(v.type == ViolationType.MAX_POSITION for v in result.violations)


# ──────── Sector Limits Tests ────────

class TestSectorLimits:
    def test_sector_limit_triggered(self):
        guard = PositionGuard(GuardConfig(max_sector_exposure=0.30))
        positions = [
            Position(ticker="AAPL", weight=0.15, sector="Tech"),
            Position(ticker="MSFT", weight=0.15, sector="Tech"),
            Position(ticker="GOOG", weight=0.10, sector="Tech"),  # total 40%
        ]
        result = guard.check(positions)
        assert any(v.type == ViolationType.MAX_SECTOR for v in result.violations)
        # Weights should be scaled down
        tech_total = sum(p.weight for p in result.adjusted_positions if p.sector == "Tech")
        assert tech_total <= 0.30 + 1e-6

    def test_sector_under_limit(self):
        guard = PositionGuard(GuardConfig(max_sector_exposure=0.30))
        positions = [
            Position(ticker="AAPL", weight=0.10, sector="Tech"),
            Position(ticker="JPM", weight=0.10, sector="Finance"),
        ]
        result = guard.check(positions)
        assert not any(v.type == ViolationType.MAX_SECTOR for v in result.violations)


# ──────── Turnover Tests ────────

class TestTurnover:
    def test_turnover_warning(self):
        guard = PositionGuard(GuardConfig(max_daily_turnover=0.20))
        # High turnover scenario: all positions change
        positions = [
            Position(ticker="NEW1", weight=0.25),
            Position(ticker="NEW2", weight=0.25),
        ]
        result = guard.check(positions)
        # Turnover = 0.5 (50%), exceeds 20%
        if result.estimated_turnover > 0.20:
            assert any(v.type == ViolationType.TURNOVER_LIMIT for v in result.violations)


# ──────── Cost Estimation Tests ────────

class TestCostEstimation:
    def test_cost_drag(self):
        guard = PositionGuard(GuardConfig(commission_bps=5, slippage_bps=10))
        # 12x annual turnover (monthly full rebalance)
        drag = guard.compute_cost_drag(12.0)
        # Cost = 12 × (5+10)/10000 × 2 = 0.036 = 3.6%
        assert abs(drag - 0.036) < 1e-6

    def test_zero_turnover_cost(self):
        guard = PositionGuard()
        drag = guard.compute_cost_drag(0.0)
        assert drag == 0.0


# ──────── Output Tests ────────

class TestGuardOutput:
    def test_to_dict(self):
        guard = PositionGuard()
        positions = [
            Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=85),  # stop loss
        ]
        result = guard.check(positions)
        d = result.to_dict()
        assert "violations" in d
        assert "forced_exits" in d

    def test_to_markdown(self):
        guard = PositionGuard()
        positions = [
            Position(ticker="AAPL", weight=0.05, entry_price=100, current_price=85),
        ]
        result = guard.check(positions)
        md = result.to_markdown()
        assert "Position Guard Report" in md

    def test_empty_positions(self):
        guard = PositionGuard()
        result = guard.check([])
        assert result.n_violations == 0
        assert result.estimated_turnover == 0.0
