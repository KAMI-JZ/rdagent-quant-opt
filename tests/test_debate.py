"""Tests for the Bull-Bear Adversarial Debate module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.debate_agents import DebateAnalyzer, Verdict, DebateResult


class TestVerdictParsing:

    def setup_method(self):
        self.analyzer = DebateAnalyzer()

    def test_parse_continue(self):
        text = "VERDICT: CONTINUE\nCONFIDENCE: 0.8\nACTION: Refine momentum window"
        verdict, conf, action = self.analyzer._parse_verdict(text)
        assert verdict == Verdict.CONTINUE
        assert conf == pytest.approx(0.8)
        assert "momentum" in action.lower()

    def test_parse_pivot(self):
        text = "VERDICT: PIVOT\nCONFIDENCE: 0.9\nACTION: Try value factors instead"
        verdict, conf, action = self.analyzer._parse_verdict(text)
        assert verdict == Verdict.PIVOT
        assert conf == pytest.approx(0.9)

    def test_parse_unknown_defaults_neutral(self):
        text = "VERDICT: MAYBE\nCONFIDENCE: 0.3\nACTION: Unclear"
        verdict, conf, _ = self.analyzer._parse_verdict(text)
        assert verdict == Verdict.NEUTRAL

    def test_parse_malformed_confidence(self):
        text = "VERDICT: CONTINUE\nCONFIDENCE: high\nACTION: Keep going"
        verdict, conf, _ = self.analyzer._parse_verdict(text)
        assert verdict == Verdict.CONTINUE
        assert conf == 0.5  # default

    def test_confidence_clamped(self):
        text = "VERDICT: CONTINUE\nCONFIDENCE: 1.5\nACTION: Go"
        _, conf, _ = self.analyzer._parse_verdict(text)
        assert conf == 1.0

        text2 = "VERDICT: PIVOT\nCONFIDENCE: -0.3\nACTION: Stop"
        _, conf2, _ = self.analyzer._parse_verdict(text2)
        assert conf2 == 0.0


class TestDebateResult:

    def test_dataclass_fields(self):
        result = DebateResult(
            verdict=Verdict.CONTINUE, confidence=0.7,
            bull_argument="good", bear_argument="bad",
            synthesis="mixed", recommended_action="refine",
        )
        assert result.verdict == Verdict.CONTINUE
        assert result.confidence == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
