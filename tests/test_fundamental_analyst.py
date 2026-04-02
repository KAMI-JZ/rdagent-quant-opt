"""
Tests for Fundamental Analyst — 基本面分析 + 信号转化
"""

import sys
import os
import importlib.util

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if "src" not in sys.modules:
    sys.modules["src"] = type(sys)("src")
    sys.modules["src"].__path__ = [_src_dir]
    sys.modules["src"].__package__ = "src"

spec = importlib.util.spec_from_file_location(
    "src.fundamental_analyst", os.path.join(_src_dir, "fundamental_analyst.py")
)
fa_mod = importlib.util.module_from_spec(spec)
sys.modules["src.fundamental_analyst"] = fa_mod
spec.loader.exec_module(fa_mod)

FundamentalData = fa_mod.FundamentalData
FundamentalAnalyzer = fa_mod.FundamentalAnalyzer
SignalGenerator = fa_mod.SignalGenerator
FundamentalLLMAnalyst = fa_mod.FundamentalLLMAnalyst
FundamentalSignal = fa_mod.FundamentalSignal
Signal = fa_mod.Signal
ValuationScore = fa_mod.ValuationScore
QualityScore = fa_mod.QualityScore
GrowthScore = fa_mod.GrowthScore

import pytest
from unittest.mock import MagicMock


# ──────────────────────────────────────────────
# FundamentalData Tests
# ──────────────────────────────────────────────

class TestFundamentalData:
    def test_default_values(self):
        data = FundamentalData()
        assert data.pe_ratio is None
        assert data.roe is None
        assert data.ticker == ""

    def test_with_values(self):
        data = FundamentalData(
            ticker="AAPL", pe_ratio=25.0, roe=30.0,
            revenue_growth=15.0, debt_equity=0.8,
        )
        assert data.ticker == "AAPL"
        assert data.pe_ratio == 25.0


# ──────────────────────────────────────────────
# Valuation Scoring Tests
# ──────────────────────────────────────────────

class TestValuation:
    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_low_pe_high_score(self):
        """低 PE → 高估值分"""
        data = FundamentalData(pe_ratio=8.0)
        val = self.analyzer._score_valuation(data)
        assert val.pe_score >= 80

    def test_high_pe_low_score(self):
        """高 PE → 低估值分"""
        data = FundamentalData(pe_ratio=50.0)
        val = self.analyzer._score_valuation(data)
        assert val.pe_score <= 30

    def test_negative_pe(self):
        """亏损 (PE < 0) → 最低分"""
        data = FundamentalData(pe_ratio=-5.0)
        val = self.analyzer._score_valuation(data)
        assert val.pe_score <= 15

    def test_low_pb(self):
        """PB < 1 → 破净，高分"""
        data = FundamentalData(pb_ratio=0.8)
        val = self.analyzer._score_valuation(data)
        assert val.pb_score >= 80

    def test_peg_ratio(self):
        """PEG < 1 → 成长+便宜"""
        data = FundamentalData(pe_ratio=15.0, earnings_growth=20.0)
        val = self.analyzer._score_valuation(data)
        assert val.peg_score >= 70  # PEG = 0.75

    def test_graham_number(self):
        """Graham Number 计算"""
        data = FundamentalData(eps=5.0, bps=30.0)
        val = self.analyzer._score_valuation(data)
        import math
        expected = math.sqrt(22.5 * 5.0 * 30.0)
        assert abs(val.graham_number - expected) < 0.01

    def test_sector_comparison(self):
        """行业对比修正"""
        # PE=10, 行业 PE=20 → PE 远低于行业 → 加分
        data = FundamentalData(pe_ratio=10.0, sector_pe=20.0)
        val = self.analyzer._score_valuation(data)
        assert val.pe_score >= 80


# ──────────────────────────────────────────────
# Quality Scoring Tests
# ──────────────────────────────────────────────

class TestQuality:
    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_high_roe(self):
        """高 ROE → 高质量分"""
        data = FundamentalData(roe=25.0)
        qual = self.analyzer._score_quality(data)
        assert qual.roe_score >= 85

    def test_low_debt(self):
        """低负债 → 健康"""
        data = FundamentalData(debt_equity=0.2)
        qual = self.analyzer._score_quality(data)
        assert qual.health_score >= 85

    def test_high_debt(self):
        """高负债 → 风险"""
        data = FundamentalData(debt_equity=2.0)
        qual = self.analyzer._score_quality(data)
        assert qual.health_score <= 30

    def test_piotroski_f_score(self):
        """Piotroski F-Score: 全满分"""
        data = FundamentalData(
            roa=10.0, fcf=1e6, net_margin=15.0,
            debt_equity=0.3, current_ratio=2.0,
            gross_margin=40.0, revenue_growth=10.0,
        )
        f = self.analyzer._piotroski_f_score(data)
        assert f >= 7  # 大部分条件满足

    def test_piotroski_f_score_zero(self):
        """全不满足 → 0"""
        data = FundamentalData(
            roa=-5.0, fcf=-1e6, net_margin=-10.0,
            debt_equity=3.0, current_ratio=0.3,
            gross_margin=10.0, revenue_growth=-5.0,
        )
        f = self.analyzer._piotroski_f_score(data)
        assert f <= 1


# ──────────────────────────────────────────────
# Growth Scoring Tests
# ──────────────────────────────────────────────

class TestGrowth:
    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_high_growth(self):
        """高增长"""
        data = FundamentalData(revenue_growth=35.0, earnings_growth=40.0)
        grow = self.analyzer._score_growth(data)
        assert grow.score >= 90

    def test_negative_growth(self):
        """负增长"""
        data = FundamentalData(revenue_growth=-10.0, earnings_growth=-5.0)
        grow = self.analyzer._score_growth(data)
        assert grow.score <= 30

    def test_no_data(self):
        """无数据 → 默认 50"""
        data = FundamentalData()
        grow = self.analyzer._score_growth(data)
        assert grow.score == 50.0


# ──────────────────────────────────────────────
# Composite Signal Tests
# ──────────────────────────────────────────────

class TestCompositeSignal:
    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_strong_buy_signal(self):
        """优质低估成长股 → STRONG_BUY"""
        data = FundamentalData(
            pe_ratio=8.0, pb_ratio=0.9, roe=25.0,
            net_margin=20.0, debt_equity=0.2,
            revenue_growth=30.0, earnings_growth=25.0,
        )
        signal = self.analyzer.analyze(data)
        assert signal.signal in (Signal.STRONG_BUY, Signal.BUY)
        assert signal.composite_score >= 70

    def test_strong_sell_signal(self):
        """高估低质衰退股 → SELL 或 STRONG_SELL"""
        data = FundamentalData(
            pe_ratio=80.0, pb_ratio=5.0, roe=2.0,
            net_margin=1.0, debt_equity=3.0,
            revenue_growth=-15.0, earnings_growth=-20.0,
        )
        signal = self.analyzer.analyze(data)
        assert signal.signal in (Signal.SELL, Signal.STRONG_SELL)
        assert signal.composite_score <= 35

    def test_hold_signal(self):
        """中等数据 → HOLD"""
        data = FundamentalData(
            pe_ratio=18.0, roe=12.0, net_margin=8.0,
            revenue_growth=5.0,
        )
        signal = self.analyzer.analyze(data)
        assert signal.signal in (Signal.HOLD, Signal.BUY)

    def test_signal_to_dict(self):
        """to_dict 输出"""
        data = FundamentalData(pe_ratio=15.0, roe=15.0)
        signal = self.analyzer.analyze(data)
        d = signal.to_dict()
        assert "signal" in d
        assert "composite_score" in d
        assert "explanation" in d


# ──────────────────────────────────────────────
# SignalGenerator Tests
# ──────────────────────────────────────────────

class TestSignalGenerator:
    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_buy_action(self):
        """BUY 信号 → 建仓"""
        data = FundamentalData(
            pe_ratio=8.0, roe=25.0, revenue_growth=30.0,
        )
        signal = self.analyzer.analyze(data)
        action = SignalGenerator.to_trading_action(signal, current_position=0.0)
        assert action["action"] == "buy"
        assert action["target_position"] > 0

    def test_sell_action(self):
        """SELL 信号 → 减仓"""
        data = FundamentalData(
            pe_ratio=80.0, roe=2.0, revenue_growth=-20.0,
        )
        signal = self.analyzer.analyze(data)
        action = SignalGenerator.to_trading_action(signal, current_position=0.5)
        assert action["action"] == "sell"
        assert action["target_position"] < 0.5

    def test_hold_no_change(self):
        """HOLD → 仓位不变"""
        data = FundamentalData(pe_ratio=18.0)
        signal = self.analyzer.analyze(data)
        if signal.signal == Signal.HOLD:
            action = SignalGenerator.to_trading_action(signal, current_position=0.3)
            assert action["target_position"] == 0.3

    def test_batch_rank(self):
        """批量排名"""
        signals = {
            "AAPL": self.analyzer.analyze(FundamentalData(pe_ratio=10.0, roe=25.0)),
            "TSLA": self.analyzer.analyze(FundamentalData(pe_ratio=80.0, roe=5.0)),
        }
        ranked = SignalGenerator.batch_rank(signals)
        assert ranked[0][0] == "AAPL"  # AAPL 应排第一
        assert ranked[0][1] > ranked[1][1]


# ──────────────────────────────────────────────
# FundamentalLLMAnalyst Tests
# ──────────────────────────────────────────────

class TestFundamentalLLMAnalyst:
    def test_without_router(self):
        """无 router → 只返回量化分析"""
        analyst = FundamentalLLMAnalyst()
        data = FundamentalData(pe_ratio=15.0, roe=20.0)
        result = analyst.deep_analysis(data)
        assert result["signal"].signal is not None
        assert result["llm_analysis"] == ""

    def test_with_mock_router(self):
        """有 router → 返回 LLM 分析"""
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "content": "AAPL shows strong fundamentals with reasonable valuation.",
            "model": "deepseek/deepseek-reasoner",
            "stage": "analysis",
            "usage": {"input": 200, "output": 100},
            "cost_usd": 0.001,
        }
        analyst = FundamentalLLMAnalyst(router=mock_router)
        data = FundamentalData(ticker="AAPL", pe_ratio=15.0, roe=20.0)
        result = analyst.deep_analysis(data)
        assert "strong fundamentals" in result["llm_analysis"]
        mock_router.route.assert_called_once()

    def test_build_prompt(self):
        """Prompt 包含关键数据"""
        analyst = FundamentalLLMAnalyst()
        data = FundamentalData(ticker="MSFT", pe_ratio=30.0, roe=40.0)
        signal = analyst.analyzer.analyze(data)
        prompt = analyst._build_prompt(data, signal, "Tech sector strong")
        assert "MSFT" in prompt
        assert "PE:" in prompt
        assert "ROE:" in prompt
        assert "Tech sector strong" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
