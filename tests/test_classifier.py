"""
Unit tests for the Stage Classifier.

Tests that prompt patterns from RD-Agent's actual templates are correctly
classified into pipeline stages. Patterns derived from arXiv:2505.15155v2
Appendix E (Prompt Design).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_router import StageClassifier, PipelineStage


@pytest.fixture
def classifier():
    return StageClassifier()


# ──────── Synthesis Stage Tests ────────

class TestSynthesisClassification:
    """Tests for hypothesis generation prompt classification."""
    
    def test_factor_hypothesis_generation(self, classifier):
        messages = [
            {"role": "system", "content": "You are a quantitative researcher."},
            {"role": "user", "content": (
                "Based on the previous experiment results and SOTA factor library, "
                "generate a new factor hypothesis. The hypothesis should include "
                "economic rationale, mathematical formula, and expected signal direction."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.SYNTHESIS
    
    def test_model_proposal(self, classifier):
        messages = [
            {"role": "user", "content": (
                "Based on historical experiment feedback, propose a novel model "
                "architecture. The current SOTA model uses LSTM with attention. "
                "Suggest an improvement based on the experiment trajectory."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.SYNTHESIS
    
    def test_chinese_factor_hypothesis(self, classifier):
        """RD-Agent supports Chinese financial reports."""
        messages = [
            {"role": "user", "content": (
                "根据之前的实验反馈和当前SOTA因子列表，请生成新的因子假设。"
                "需要包含经济学原理和信号方向。"
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.SYNTHESIS
    
    def test_knowledge_forest_reference(self, classifier):
        messages = [
            {"role": "user", "content": (
                "The idea forest contains the following branches. "
                "Generate a new hypothesis that explores an unexplored direction."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.SYNTHESIS


# ──────── Implementation Stage Tests ────────

class TestImplementationClassification:
    """Tests for code generation and debugging prompt classification."""
    
    def test_factor_code_generation(self, classifier):
        messages = [
            {"role": "system", "content": "You are a Python developer."},
            {"role": "user", "content": (
                "Implement the following factor in Python using pandas and qlib:\n"
                "Factor: 20-day momentum deviation\n"
                "Formula: (close / close.shift(20) - 1) - rolling_mean(ret, 60)\n"
                "def calculate_factor(df):"
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.IMPLEMENTATION
    
    def test_debug_error(self, classifier):
        messages = [
            {"role": "user", "content": (
                "The previous code execution produced the following error:\n"
                "Traceback (most recent call last):\n"
                "  File 'factor.py', line 15\n"
                "TypeError: unsupported operand type(s)\n"
                "Please fix the bug and provide corrected code."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.IMPLEMENTATION
    
    def test_code_refinement(self, classifier):
        messages = [
            {"role": "user", "content": (
                "Refine the implementation based on the knowledge base. "
                "A similar task was successfully implemented as follows:\n"
                "import pandas as pd\nimport numpy as np\n"
                "def compute_factor(data):\n    return data['close'].pct_change(20)"
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.IMPLEMENTATION


# ──────── Analysis Stage Tests ────────

class TestAnalysisClassification:
    """Tests for result analysis prompt classification."""
    
    def test_performance_analysis(self, classifier):
        messages = [
            {"role": "user", "content": (
                "Analyze the following experiment result:\n"
                "IC: 0.032, ICIR: 0.28, ARR: 5.2%, MDD: -12%\n"
                "Compare with SOTA: IC decreased by 0.005\n"
                "Diagnose the failure and suggest improvements."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.ANALYSIS
    
    def test_strategy_evaluation(self, classifier):
        messages = [
            {"role": "user", "content": (
                "Evaluate whether the experiment outperforms the current SOTA. "
                "The Sharpe ratio improved from 1.2 to 1.5, but maximum drawdown "
                "worsened from -8% to -15%. Recommend next direction."
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.ANALYSIS
    
    def test_chinese_analysis(self, classifier):
        messages = [
            {"role": "user", "content": (
                "请分析本轮回测结果。IC指标为0.045，年化收益率12%。"
                "需要诊断改进方向。"
            )}
        ]
        assert classifier.classify(messages) == PipelineStage.ANALYSIS


# ──────── Edge Cases ────────

class TestEdgeCases:
    """Tests for ambiguous or mixed-content prompts."""
    
    def test_empty_messages(self, classifier):
        assert classifier.classify([]) == PipelineStage.UNKNOWN
    
    def test_generic_message(self, classifier):
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        assert classifier.classify(messages) == PipelineStage.UNKNOWN
    
    def test_mixed_content_prefers_stronger_signal(self, classifier):
        """When a prompt mentions both code and analysis, the stronger signal wins."""
        messages = [
            {"role": "user", "content": (
                "The implementation produced an error. Traceback:\n"
                "File 'factor.py', line 10\n"
                "Fix the code implementation."
            )}
        ]
        # Should classify as IMPLEMENTATION (stronger signal)
        result = classifier.classify(messages)
        assert result == PipelineStage.IMPLEMENTATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
