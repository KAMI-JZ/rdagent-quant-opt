"""Tests for the Anti-Alpha-Decay Filter."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.alpha_filter import (
    ASTSimilarityChecker, ComplexityChecker, AlphaDecayFilter, FilterResult,
)


# ──────── AST Similarity Tests ────────

class TestASTSimilarity:

    def test_identical_code_max_similarity(self):
        checker = ASTSimilarityChecker(similarity_threshold=0.85)
        code = "def f(df): return df['close'].pct_change(20)"
        checker.add_factor("momentum_20d", code)
        sim, name = checker.check(code)
        assert sim == 1.0
        assert name == "momentum_20d"

    def test_different_code_low_similarity(self):
        checker = ASTSimilarityChecker()
        checker.add_factor("simple", "x = 1 + 2")
        sim, _ = checker.check(
            "def calculate(df):\n"
            "    a = df['close'].rolling(20).mean()\n"
            "    b = df['volume'].shift(5)\n"
            "    return a / b"
        )
        assert sim < 0.85

    def test_empty_library_returns_zero(self):
        checker = ASTSimilarityChecker()
        sim, name = checker.check("x = 1")
        assert sim == 0.0
        assert name is None

    def test_syntax_error_returns_zero(self):
        checker = ASTSimilarityChecker()
        sim, name = checker.check("def broken(:")
        assert sim == 0.0


# ──────── Complexity Tests ────────

class TestComplexity:

    def test_simple_assignment(self):
        checker = ComplexityChecker(max_depth=5)
        assert checker.check("x = 1") <= 5

    def test_deeply_nested(self):
        checker = ComplexityChecker(max_depth=3)
        code = "x = f(g(h(j(k(1)))))"  # 5 nested calls
        depth = checker.check(code)
        assert depth > 3

    def test_syntax_error_returns_negative(self):
        checker = ComplexityChecker()
        assert checker.check("def broken(:") == -1


# ──────── Composite Filter Tests ────────

class TestAlphaDecayFilter:

    def test_novel_simple_factor_passes(self):
        filt = AlphaDecayFilter(similarity_threshold=0.85, max_complexity_depth=5)
        result = filt.evaluate("x = df['close'].pct_change(10)", check_alignment=False)
        assert result.passed
        assert result.rejection_reasons == []

    def test_duplicate_factor_rejected(self):
        filt = AlphaDecayFilter(similarity_threshold=0.85)
        code = "def f(df): return df['close'].pct_change(20)"
        filt.add_existing_factor("momentum_20d", code)
        result = filt.evaluate(code, check_alignment=False)
        assert not result.passed
        assert any("similar" in r.lower() for r in result.rejection_reasons)

    def test_overly_complex_factor_rejected(self):
        filt = AlphaDecayFilter(max_complexity_depth=2)
        code = "x = f(g(h(j(1))))"
        result = filt.evaluate(code, check_alignment=False)
        assert not result.passed
        assert any("complex" in r.lower() for r in result.rejection_reasons)

    def test_filter_result_fields(self):
        filt = AlphaDecayFilter()
        result = filt.evaluate("x = 1", check_alignment=False)
        assert isinstance(result, FilterResult)
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.complexity_depth, int)


# ──────── Hyperbolic Decay Tests ────────

class TestHyperbolicDecay:

    def test_recent_duplicate_still_rejected(self):
        """刚注册的因子（age=0）应该被正常拒绝。"""
        checker = ASTSimilarityChecker(similarity_threshold=0.85, decay_lambda=0.1)
        code = "def f(df): return df['close'].pct_change(20)"
        checker.add_factor("momentum_20d", code, iteration=0)
        checker.set_iteration(0)  # age = 0
        sim, name = checker.check(code)
        assert sim >= 0.85  # 相同代码应该被拒绝
        assert name == "momentum_20d"

    def test_old_duplicate_relaxed(self):
        """10轮前注册的因子，有效相似度应降低。"""
        checker = ASTSimilarityChecker(similarity_threshold=0.85, decay_lambda=0.1)
        code = "def f(df): return df['close'].pct_change(20)"
        checker.add_factor("momentum_20d", code, iteration=0)
        checker.set_iteration(10)  # age = 10, decay = 1 + 0.1*10 = 2.0
        sim, name = checker.check(code)
        # 原始相似度 1.0，衰减后 1.0/2.0 = 0.5 < 0.85 → 放行
        assert sim < 0.85

    def test_decay_lambda_zero_no_decay(self):
        """lambda=0 时不应有衰减。"""
        checker = ASTSimilarityChecker(similarity_threshold=0.85, decay_lambda=0.0)
        code = "def f(df): return df['close'].pct_change(20)"
        checker.add_factor("old", code, iteration=0)
        checker.set_iteration(100)
        sim, _ = checker.check(code)
        assert sim == 1.0  # 无衰减，相同代码永远是 1.0

    def test_filter_with_decay_passes_old_similar(self):
        """通过 AlphaDecayFilter，老因子的相似代码应放行。"""
        filt = AlphaDecayFilter(similarity_threshold=0.85, max_complexity_depth=5)
        code = "def f(df): return df['close'].pct_change(20)"
        filt.add_existing_factor("momentum_20d", code, iteration=0)
        filt.set_iteration(15)
        result = filt.evaluate(code, check_alignment=False)
        # age=15, decay=1+0.1*15=2.5, effective_sim = 1.0/2.5 = 0.4 < 0.85
        assert result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
