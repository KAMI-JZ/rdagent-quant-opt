"""Tests for the Market Regime Detector."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.market_regime import (
    MarketRegimeDetector, MarketRegime, RegimeAwareSynthesis,
    MarketDirection, Volatility, MarketStyle,
)


class TestIndicatorDetection:

    def setup_method(self):
        self.detector = MarketRegimeDetector()

    def test_bull_low_trend(self):
        regime = self.detector.detect_from_indicators(
            returns_20d=0.05, volatility_20d=0.12, ma_cross=0.04, vol_percentile=0.3,
        )
        assert regime.direction == MarketDirection.BULL
        assert regime.volatility == Volatility.LOW
        assert regime.style == MarketStyle.TREND

    def test_bear_high_mean_revert(self):
        regime = self.detector.detect_from_indicators(
            returns_20d=-0.03, volatility_20d=0.30, ma_cross=0.01, vol_percentile=0.8,
        )
        assert regime.direction == MarketDirection.BEAR
        assert regime.volatility == Volatility.HIGH
        assert regime.style == MarketStyle.MEAN_REVERT

    def test_label_format(self):
        regime = self.detector.detect_from_indicators(
            returns_20d=0.02, volatility_20d=0.15, ma_cross=0.05, vol_percentile=0.5,
        )
        assert regime.label == "bull_low_trend"

    def test_confidence_bounded(self):
        regime = self.detector.detect_from_indicators(
            returns_20d=0.50, volatility_20d=0.15, ma_cross=0.05, vol_percentile=0.5,
        )
        assert 0.0 <= regime.confidence <= 1.0


class TestResponseParsing:

    def test_parse_valid_response(self):
        detector = MarketRegimeDetector()
        text = (
            "DIRECTION: bear\n"
            "VOLATILITY: high\n"
            "STYLE: mean_revert\n"
            "CONFIDENCE: 0.85\n"
            "DESCRIPTION: market showing strong reversal signals"
        )
        regime = detector._parse_response(text)
        assert regime.direction == MarketDirection.BEAR
        assert regime.volatility == Volatility.HIGH
        assert regime.style == MarketStyle.MEAN_REVERT
        assert regime.confidence == pytest.approx(0.85)

    def test_parse_defaults_on_missing(self):
        detector = MarketRegimeDetector()
        regime = detector._parse_response("")
        assert regime.direction == MarketDirection.BULL
        assert regime.confidence == 0.5


class TestRegimeAwareSynthesis:

    def test_augment_injects_context(self):
        regime = MarketRegime(
            MarketDirection.BULL, Volatility.LOW, MarketStyle.TREND,
            confidence=0.8, description="uptrend",
        )
        synth = RegimeAwareSynthesis()
        synth.current_regime = regime
        messages = [
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "Generate a factor hypothesis."},
        ]
        augmented = synth.augment_prompt(messages)
        assert "[Market Context]" in augmented[1]["content"]
        assert augmented[0] == messages[0]  # system unchanged

    def test_no_regime_returns_original(self):
        synth = RegimeAwareSynthesis()
        messages = [{"role": "user", "content": "test"}]
        assert synth.augment_prompt(messages) == messages

    def test_prompt_context_content(self):
        regime = MarketRegime(
            MarketDirection.BEAR, Volatility.HIGH, MarketStyle.MEAN_REVERT,
            confidence=0.9, description="volatile downturn",
        )
        ctx = regime.to_prompt_context()
        assert "bear" in ctx.lower()
        assert "high" in ctx.lower()
        assert "mean_revert" in ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
