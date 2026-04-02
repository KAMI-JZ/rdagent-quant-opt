"""
Tests for the upgraded Bull-Bear Debate system (V2).
Tests router integration, improved prompts, and history-aware debate.
"""

import sys
import os
import importlib.util

# 影响: 绕过 src/__init__.py 的重依赖链 (litellm等)，直接加载目标模块
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if "src" not in sys.modules:
    sys.modules["src"] = type(sys)("src")
    sys.modules["src"].__path__ = [_src_dir]
    sys.modules["src"].__package__ = "src"

# Mock litellm before importing debate_agents
from unittest.mock import MagicMock
sys.modules["litellm"] = MagicMock()

spec = importlib.util.spec_from_file_location(
    "src.debate_agents", os.path.join(_src_dir, "debate_agents.py")
)
debate_mod = importlib.util.module_from_spec(spec)
sys.modules["src.debate_agents"] = debate_mod
spec.loader.exec_module(debate_mod)

DebateAnalyzer = debate_mod.DebateAnalyzer
DebateResult = debate_mod.DebateResult
Verdict = debate_mod.Verdict

import pytest


# ──────────────────────────────────────────────
# 基础测试 (向后兼容 — 现有 6 个测试的逻辑)
# ──────────────────────────────────────────────

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

    def test_confidence_clamped_high(self):
        text = "VERDICT: CONTINUE\nCONFIDENCE: 1.5\nACTION: Go"
        _, conf, _ = self.analyzer._parse_verdict(text)
        assert conf == 1.0

    def test_confidence_clamped_low(self):
        text = "VERDICT: PIVOT\nCONFIDENCE: -0.3\nACTION: Stop"
        _, conf, _ = self.analyzer._parse_verdict(text)
        assert conf == 0.0


class TestDebateResult:
    def test_dataclass_fields(self):
        result = DebateResult(
            verdict=Verdict.CONTINUE, confidence=0.7,
            bull_argument="good", bear_argument="bad",
            synthesis="mixed", recommended_action="refine",
        )
        assert result.verdict == Verdict.CONTINUE
        assert result.confidence == 0.7
        assert result.model_used == ""  # 默认空

    def test_model_used_field(self):
        """V2 新增: model_used 字段记录实际使用的模型"""
        result = DebateResult(
            verdict=Verdict.PIVOT, confidence=0.9,
            bull_argument="bull", bear_argument="bear",
            synthesis="synth", recommended_action="pivot",
            model_used="anthropic/claude-opus-4-6",
        )
        assert result.model_used == "anthropic/claude-opus-4-6"


# ──────────────────────────────────────────────
# V2 新增: Router 集成测试
# ──────────────────────────────────────────────

class TestRouterIntegration:
    """测试 DebateAnalyzer 通过 router 自动选模型"""

    def test_init_without_router(self):
        """无 router 时保持原有行为"""
        analyzer = DebateAnalyzer()
        assert analyzer.router is None
        assert analyzer.debate_model == "deepseek/deepseek-reasoner"

    def test_init_with_router(self):
        """传入 router 后记录"""
        mock_router = MagicMock()
        analyzer = DebateAnalyzer(router=mock_router)
        assert analyzer.router is mock_router

    def test_call_llm_uses_router_when_available(self):
        """有 router 时 _call_llm 走 router.route()"""
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "content": "Bull argument text",
            "model": "anthropic/claude-opus-4-6",
            "stage": "analysis",
            "usage": {"input": 100, "output": 50},
            "cost_usd": 0.01,
        }
        analyzer = DebateAnalyzer(router=mock_router)
        text, model = analyzer._call_llm("system", "user", "any-model")
        assert text == "Bull argument text"
        assert model == "anthropic/claude-opus-4-6"
        mock_router.route.assert_called_once()

    def test_call_llm_falls_back_to_litellm_without_router(self):
        """无 router 时走 litellm.completion()"""
        analyzer = DebateAnalyzer()
        # litellm 是 mocked，配置返回值
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  direct llm response  "
        debate_mod.litellm.completion.return_value = mock_response

        text, model = analyzer._call_llm("sys", "usr", "deepseek/deepseek-reasoner")
        assert text == "direct llm response"
        assert model == "deepseek/deepseek-reasoner"

    def test_router_route_called_three_times_in_debate(self):
        """debate() 应调用 router 3 次: Bull + Bear + Judge"""
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "content": "VERDICT: CONTINUE\nCONFIDENCE: 0.75\nACTION: Refine params",
            "model": "deepseek/deepseek-reasoner",
            "stage": "analysis",
            "usage": {"input": 100, "output": 50},
            "cost_usd": 0.001,
        }
        analyzer = DebateAnalyzer(router=mock_router)
        result = analyzer.debate("momentum hypothesis", "df.pct_change(20)", {"IC": 0.03}, 1)

        assert mock_router.route.call_count == 3
        assert result.verdict == Verdict.CONTINUE
        assert result.model_used == "deepseek/deepseek-reasoner"


# ──────────────────────────────────────────────
# V2 新增: 改进 Prompt 测试
# ──────────────────────────────────────────────

class TestImprovedPrompts:
    """验证 V2 prompt 要求量化证据引用"""

    def test_bull_prompt_requires_metrics(self):
        """Bull prompt 要求引用具体指标"""
        assert "MUST cite specific metric" in debate_mod.BULL_SYSTEM
        assert "IC" in debate_mod.BULL_SYSTEM

    def test_bear_prompt_requires_metrics(self):
        """Bear prompt 要求引用具体指标"""
        assert "MUST cite specific metric" in debate_mod.BEAR_SYSTEM
        assert "MDD" in debate_mod.BEAR_SYSTEM

    def test_judge_prompt_requires_evidence(self):
        """Judge prompt 要求基于证据决策"""
        assert "quantitative evidence" in debate_mod.JUDGE_SYSTEM
        assert "metrics" in debate_mod.JUDGE_SYSTEM

    def test_bull_prompt_has_benchmarks(self):
        """Bull prompt 包含基准阈值供参考"""
        assert "IC>0.02" in debate_mod.BULL_SYSTEM
        assert "Sharpe>1.0" in debate_mod.BULL_SYSTEM

    def test_bear_prompt_has_failure_thresholds(self):
        """Bear prompt 包含失败阈值"""
        assert "IC<0.02" in debate_mod.BEAR_SYSTEM
        assert "Sharpe<0" in debate_mod.BEAR_SYSTEM


# ──────────────────────────────────────────────
# V2 新增: 历史辩论上下文测试
# ──────────────────────────────────────────────

class TestDebateHistory:
    """测试 debate_with_history() 的历史注入功能"""

    def setup_method(self):
        self.mock_router = MagicMock()
        self.mock_router.route.return_value = {
            "content": "VERDICT: PIVOT\nCONFIDENCE: 0.85\nACTION: Try mean reversion",
            "model": "deepseek/deepseek-reasoner",
            "stage": "analysis",
            "usage": {"input": 200, "output": 100},
            "cost_usd": 0.002,
        }
        self.analyzer = DebateAnalyzer(router=self.mock_router)

    def test_build_history_context_empty(self):
        """空历史返回空字符串"""
        assert self.analyzer._build_history_context([]) == ""

    def test_build_history_context_single(self):
        """单条历史正确格式化"""
        history = [DebateResult(
            verdict=Verdict.CONTINUE, confidence=0.7,
            bull_argument="bull", bear_argument="bear",
            synthesis="synth", recommended_action="refine params",
        )]
        ctx = self.analyzer._build_history_context(history)
        assert "DEBATE HISTORY" in ctx
        assert "continue" in ctx
        assert "0.70" in ctx
        assert "refine params" in ctx

    def test_build_history_context_max_three(self):
        """最多引用最近3轮历史"""
        history = [
            DebateResult(
                verdict=Verdict.CONTINUE, confidence=0.5 + i * 0.1,
                bull_argument=f"bull{i}", bear_argument=f"bear{i}",
                synthesis=f"synth{i}", recommended_action=f"action{i}",
            )
            for i in range(5)
        ]
        ctx = self.analyzer._build_history_context(history)
        # 应该只有 Round 1, 2, 3 (最近3轮)
        assert "Round 1" in ctx
        assert "Round 2" in ctx
        assert "Round 3" in ctx
        assert ctx.count("Round") == 3

    def test_debate_with_history_injects_context(self):
        """debate_with_history 将历史注入到 prompt 中"""
        history = [DebateResult(
            verdict=Verdict.PIVOT, confidence=0.9,
            bull_argument="prev bull", bear_argument="prev bear",
            synthesis="prev synth", recommended_action="pivoted to value",
        )]
        result = self.analyzer.debate_with_history(
            hypothesis="momentum factor",
            code="df.pct_change(20)",
            metrics={"IC": 0.01, "Sharpe": -0.5},
            iteration=2,
            history=history,
        )
        # 验证 router 被调用了 3 次
        assert self.mock_router.route.call_count == 3
        # 验证第一次调用 (Bull) 的 prompt 包含历史
        first_call_messages = self.mock_router.route.call_args_list[0][0][0]
        user_content = first_call_messages[1]["content"]
        assert "DEBATE HISTORY" in user_content
        assert "pivoted to value" in user_content

    def test_debate_with_history_no_history(self):
        """无历史时 debate_with_history 行为等同 debate"""
        result = self.analyzer.debate_with_history(
            hypothesis="test", code="pass", metrics={"IC": 0.0},
            iteration=0, history=None,
        )
        assert result.verdict == Verdict.PIVOT
        # prompt 中不应有 DEBATE HISTORY
        first_call_messages = self.mock_router.route.call_args_list[0][0][0]
        user_content = first_call_messages[1]["content"]
        assert "DEBATE HISTORY" not in user_content


# ──────────────────────────────────────────────
# 边界情况测试
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_build_context_with_empty_metrics(self):
        analyzer = DebateAnalyzer()
        ctx = analyzer._build_context("hyp", "code", {}, 0)
        assert "Hypothesis: hyp" in ctx
        assert "Backtest Metrics:" in ctx

    def test_build_context_with_rich_metrics(self):
        analyzer = DebateAnalyzer()
        metrics = {"IC": 0.035, "Sharpe": 1.8, "MDD": -0.15, "ICIR": 0.42}
        ctx = analyzer._build_context("momentum", "df.pct_change()", metrics, 5)
        assert "IC: 0.035" in ctx
        assert "Sharpe: 1.8" in ctx
        assert "Iteration: 5" in ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
