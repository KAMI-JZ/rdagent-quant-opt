"""Tests for FactorTranslator — 因子翻译器测试"""

import importlib
import sys
import os
import types
import pytest

# 影响: 注入空 src 包，避免 __init__.py 拉起 litellm 等重依赖
_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [_src_path]
    sys.modules["src"] = src_pkg

spec = importlib.util.spec_from_file_location("src.factor_translator",
                                               os.path.join(_src_path, "factor_translator.py"))
_mod = importlib.util.module_from_spec(spec)
sys.modules["src.factor_translator"] = _mod
spec.loader.exec_module(_mod)

FactorTranslator = _mod.FactorTranslator
FactorAnalyzer = _mod.FactorAnalyzer
FactorType = _mod.FactorType
SignalDirection = _mod.SignalDirection
PositionMethod = _mod.PositionMethod
TradingStrategy = _mod.TradingStrategy
EntryRule = _mod.EntryRule
ExitRule = _mod.ExitRule
PositionRule = _mod.PositionRule
translate_run_results = _mod.translate_run_results


# ──────── Sample Factor Codes ────────

MOMENTUM_FACTOR = """
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    returns = df['close'].pct_change(20)
    momentum = returns.rolling(5).mean()
    return momentum
"""

MEAN_REVERSION_FACTOR = """
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    z_score = (df['close'] - ma20) / std20
    return -z_score  # negative z-score = oversold = buy signal
"""

VOLATILITY_FACTOR = """
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    high_low = df['high'] - df['low']
    atr = high_low.rolling(14).mean()
    volatility = df['close'].rolling(20).std()
    return atr / volatility
"""

VOLUME_FACTOR = """
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    avg_volume = df['volume'].rolling(20).mean()
    volume_ratio = df['volume'] / avg_volume
    price_change = df['close'].pct_change()
    return volume_ratio * price_change
"""

COMPOSITE_FACTOR = """
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    momentum = df['close'].pct_change(10)
    vol = df['close'].rolling(20).std()
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
    z_score = (df['close'] - df['close'].rolling(20).mean()) / vol
    return momentum * vol_ratio - z_score
"""

GOOD_METRICS = {
    "IC": 0.052,
    "ICIR": 0.245,
    "annual_return": 0.834,
    "max_drawdown": -0.103,
    "sharpe_ratio": 3.991,
    "n_stocks": 50,
    "n_days": 247,
}

MEDIOCRE_METRICS = {
    "IC": 0.015,
    "ICIR": 0.09,
    "annual_return": 0.03,
    "max_drawdown": -0.45,
    "sharpe_ratio": 0.07,
    "n_stocks": 50,
    "n_days": 200,
}


# ──────── FactorAnalyzer Tests ────────

class TestFactorAnalyzer:
    def setup_method(self):
        self.analyzer = FactorAnalyzer()

    def test_momentum_detection(self):
        result = self.analyzer.analyze(MOMENTUM_FACTOR)
        assert result["factor_type"] == FactorType.MOMENTUM

    def test_mean_reversion_detection(self):
        result = self.analyzer.analyze(MEAN_REVERSION_FACTOR)
        assert result["factor_type"] == FactorType.MEAN_REVERSION

    def test_volatility_detection(self):
        result = self.analyzer.analyze(VOLATILITY_FACTOR)
        assert result["factor_type"] == FactorType.VOLATILITY

    def test_volume_detection(self):
        result = self.analyzer.analyze(VOLUME_FACTOR)
        assert result["factor_type"] == FactorType.VOLUME
        assert result["uses_volume"] is True

    def test_composite_detection(self):
        result = self.analyzer.analyze(COMPOSITE_FACTOR)
        # Should detect multiple types → composite
        assert result["factor_type"] in (FactorType.COMPOSITE, FactorType.MOMENTUM,
                                          FactorType.VOLUME, FactorType.MEAN_REVERSION)

    def test_window_extraction(self):
        result = self.analyzer.analyze(MOMENTUM_FACTOR)
        assert 20 in result["windows"]
        assert 5 in result["windows"]

    def test_window_extraction_volatility(self):
        result = self.analyzer.analyze(VOLATILITY_FACTOR)
        assert 14 in result["windows"]
        assert 20 in result["windows"]

    def test_volume_usage(self):
        result = self.analyzer.analyze(MOMENTUM_FACTOR)
        assert result["uses_volume"] is False

        result = self.analyzer.analyze(VOLUME_FACTOR)
        assert result["uses_volume"] is True

    def test_complexity(self):
        result = self.analyzer.analyze(MOMENTUM_FACTOR)
        assert result["complexity"] > 0

    def test_empty_code(self):
        result = self.analyzer.analyze("")
        assert result["factor_type"] == FactorType.UNKNOWN
        assert result["windows"] == []

    def test_invalid_syntax(self):
        result = self.analyzer.analyze("this is not python {{{{")
        assert result["complexity"] > 0  # fallback to line count


# ──────── FactorTranslator Tests ────────

class TestFactorTranslator:
    def setup_method(self):
        self.translator = FactorTranslator()

    def test_translate_momentum(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        assert strategy.factor_type == FactorType.MOMENTUM
        assert strategy.name.startswith("Momentum")
        assert strategy.backtest_metrics == GOOD_METRICS

    def test_translate_mean_reversion(self):
        strategy = self.translator.translate(MEAN_REVERSION_FACTOR, MEDIOCRE_METRICS)
        assert strategy.factor_type == FactorType.MEAN_REVERSION
        assert strategy.signal_direction == SignalDirection.LONG_SHORT

    def test_translate_with_hypothesis(self):
        strategy = self.translator.translate(
            MOMENTUM_FACTOR, GOOD_METRICS,
            hypothesis="20-day momentum captures intermediate trend"
        )
        assert "hypothesis" in strategy.narrative.lower() or "momentum" in strategy.narrative.lower()

    def test_entry_rule_generated(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        assert strategy.entry.condition != ""
        assert strategy.entry.lookback_days > 0

    def test_exit_rule_high_sharpe(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        # High Sharpe → more generous take profit + trailing stop
        assert strategy.exit.take_profit_pct >= 0.15
        assert strategy.exit.trailing_stop_pct > 0

    def test_exit_rule_low_sharpe(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, MEDIOCRE_METRICS)
        # Low Sharpe → tighter take profit, no trailing
        assert strategy.exit.take_profit_pct <= 0.15
        assert strategy.exit.trailing_stop_pct == 0.0

    def test_stop_loss_based_on_mdd(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        # Stop loss should be related to historical MDD
        assert strategy.exit.stop_loss_pct < 0  # negative = loss

    def test_position_rule_high_sharpe(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        # High Sharpe → concentrated, factor weighted
        assert strategy.position.method == PositionMethod.FACTOR_WEIGHTED
        assert strategy.position.n_holdings <= 20

    def test_position_rule_low_sharpe(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, MEDIOCRE_METRICS)
        # Low Sharpe → diversified, equal weight
        assert strategy.position.method == PositionMethod.EQUAL_WEIGHT

    def test_volatility_position_method(self):
        strategy = self.translator.translate(VOLATILITY_FACTOR, GOOD_METRICS)
        assert strategy.position.method == PositionMethod.INVERSE_VOL

    def test_rebalance_frequency(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        assert strategy.rebalance_frequency in ("daily", "weekly", "monthly")

    def test_no_metrics(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR)
        assert strategy.factor_type == FactorType.MOMENTUM
        assert strategy.exit.stop_loss_pct < 0

    def test_narrative_generated(self):
        strategy = self.translator.translate(MOMENTUM_FACTOR, GOOD_METRICS)
        assert len(strategy.narrative) > 50
        assert "momentum" in strategy.narrative.lower() or "strategy" in strategy.narrative.lower()


# ──────── TradingStrategy Tests ────────

class TestTradingStrategy:
    def test_to_dict(self):
        strategy = TradingStrategy(
            name="TestMomentum",
            factor_type=FactorType.MOMENTUM,
            signal_direction=SignalDirection.LONG_SHORT,
            entry=EntryRule(condition="test", threshold=0.5, lookback_days=20),
            exit=ExitRule(stop_loss_pct=-0.05, take_profit_pct=0.15),
            position=PositionRule(method=PositionMethod.EQUAL_WEIGHT, n_holdings=20),
        )
        d = strategy.to_dict()
        assert d["name"] == "TestMomentum"
        assert d["factor_type"] == "momentum"
        assert d["signal_direction"] == "long_short"
        assert d["entry"]["threshold"] == 0.5
        assert d["exit"]["stop_loss_pct"] == -0.05
        assert d["position"]["method"] == "equal_weight"

    def test_to_markdown(self):
        strategy = TradingStrategy(
            name="TestMomentum",
            factor_type=FactorType.MOMENTUM,
            entry=EntryRule(condition="Buy top 20%", lookback_days=20),
            exit=ExitRule(stop_loss_pct=-0.05, take_profit_pct=0.15, max_holding_days=20),
            position=PositionRule(method=PositionMethod.EQUAL_WEIGHT, n_holdings=20),
            backtest_metrics=GOOD_METRICS,
            narrative="Test narrative",
        )
        md = strategy.to_markdown()
        assert "# Trading Strategy: TestMomentum" in md
        assert "Stop Loss" in md
        assert "Test narrative" in md
        assert "3.99" in md  # Sharpe

    def test_to_markdown_no_trailing_stop(self):
        strategy = TradingStrategy(
            exit=ExitRule(trailing_stop_pct=0.0),
        )
        md = strategy.to_markdown()
        assert "Trailing Stop" not in md


# ──────── Batch Translation Tests ────────

class TestBatchTranslation:
    def test_translate_run_results(self):
        results = [
            {
                "iteration": 0,
                "skipped": True,
                "factor_code": "",
                "backtest_metrics": {},
            },
            {
                "iteration": 1,
                "skipped": False,
                "factor_code": MOMENTUM_FACTOR,
                "backtest_metrics": GOOD_METRICS,
                "hypothesis": "momentum captures trend",
            },
            {
                "iteration": 2,
                "skipped": False,
                "factor_code": MEAN_REVERSION_FACTOR,
                "backtest_metrics": MEDIOCRE_METRICS,
            },
        ]
        strategies = translate_run_results(results)
        assert len(strategies) == 2
        assert strategies[0].factor_type == FactorType.MOMENTUM
        assert strategies[1].factor_type == FactorType.MEAN_REVERSION

    def test_empty_results(self):
        strategies = translate_run_results([])
        assert strategies == []

    def test_all_skipped(self):
        results = [{"iteration": 0, "skipped": True, "factor_code": "", "backtest_metrics": {}}]
        strategies = translate_run_results(results)
        assert strategies == []
