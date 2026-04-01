"""Tests for FactorReviewer — 因子审视系统测试"""

import importlib
import sys
import os
import types
import pytest

# 影响: 直接加载模块，避免 __init__.py 重依赖
_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [_src_path]
    sys.modules["src"] = src_pkg

spec = importlib.util.spec_from_file_location("src.factor_reviewer",
                                               os.path.join(_src_path, "factor_reviewer.py"))
_mod = importlib.util.module_from_spec(spec)
sys.modules["src.factor_reviewer"] = _mod
spec.loader.exec_module(_mod)

FactorReviewer = _mod.FactorReviewer
ReturnAttribution = _mod.ReturnAttribution
StabilityAnalyzer = _mod.StabilityAnalyzer
FactorReview = _mod.FactorReview
ReviewGrade = _mod.ReviewGrade
RiskFlag = _mod.RiskFlag
ReturnDecomposition = _mod.ReturnDecomposition
StabilityMetrics = _mod.StabilityMetrics
review_run_results = _mod.review_run_results

# ──────── Test Data ────────

EXCELLENT_METRICS = {
    "IC": 0.052, "ICIR": 0.245, "annual_return": 0.834,
    "max_drawdown": -0.103, "sharpe_ratio": 3.991,
    "n_stocks": 50, "n_days": 247,
}

GOOD_METRICS = {
    "IC": 0.030, "ICIR": 0.19, "annual_return": 0.12,
    "max_drawdown": -0.07, "sharpe_ratio": 1.3,
    "n_stocks": 50, "n_days": 200,
}

MEDIOCRE_METRICS = {
    "IC": 0.015, "ICIR": 0.09, "annual_return": 0.03,
    "max_drawdown": -0.45, "sharpe_ratio": 0.07,
    "n_stocks": 50, "n_days": 200,
}

POOR_METRICS = {
    "IC": 0.005, "ICIR": 0.03, "annual_return": -0.05,
    "max_drawdown": -0.30, "sharpe_ratio": -0.5,
    "n_stocks": 50, "n_days": 100,
}

FAIL_METRICS = {
    "IC": -0.01, "ICIR": -0.05, "annual_return": -0.20,
    "max_drawdown": -0.50, "sharpe_ratio": -2.0,
    "n_stocks": 10, "n_days": 30,
}

# 稳定的 IC 序列（大多数正）
STABLE_IC = [0.03, 0.05, 0.02, 0.04, -0.01, 0.06, 0.03, 0.04, 0.02, 0.05,
             0.03, 0.04, 0.01, 0.05, 0.02, 0.03, 0.04, 0.06, 0.01, 0.03] * 5

# 衰减的 IC 序列（前半好，后半差）
DECAYING_IC = [0.06, 0.05, 0.04, 0.05, 0.03] * 10 + [-0.01, 0.00, -0.02, 0.01, -0.01] * 10

# 不稳定的 IC 序列
UNSTABLE_IC = [0.08, -0.05, 0.07, -0.06, 0.09, -0.08] * 10


# ──────── ReturnAttribution Tests ────────

class TestReturnAttribution:
    def setup_method(self):
        self.attr = ReturnAttribution()

    def test_high_return_decomposition(self):
        decomp = self.attr.decompose(EXCELLENT_METRICS)
        assert decomp.total_return == 0.834
        assert decomp.factor_alpha > 0
        assert decomp.market_beta_return >= 0

    def test_alpha_ratio_high_sharpe(self):
        decomp = self.attr.decompose(EXCELLENT_METRICS)
        # High Sharpe + low MDD → high alpha ratio
        assert decomp.alpha_ratio > 0.5

    def test_negative_return(self):
        decomp = self.attr.decompose(POOR_METRICS)
        assert decomp.total_return < 0
        assert decomp.factor_alpha < 0

    def test_zero_return(self):
        decomp = self.attr.decompose({"annual_return": 0, "sharpe_ratio": 0, "max_drawdown": 0})
        assert decomp.alpha_ratio == 0.0


# ──────── StabilityAnalyzer Tests ────────

class TestStabilityAnalyzer:
    def setup_method(self):
        self.analyzer = StabilityAnalyzer()

    def test_stable_ic(self):
        result = self.analyzer.analyze(STABLE_IC, EXCELLENT_METRICS)
        assert result.ic_mean > 0
        assert result.ic_positive_ratio > 0.8

    def test_decaying_ic(self):
        result = self.analyzer.analyze(DECAYING_IC, GOOD_METRICS)
        assert result.ic_decay_rate < 0  # negative = decaying

    def test_unstable_ic(self):
        result = self.analyzer.analyze(UNSTABLE_IC, MEDIOCRE_METRICS)
        # Unstable → ic_positive_ratio around 0.5
        assert 0.3 < result.ic_positive_ratio < 0.7

    def test_empty_ic(self):
        result = self.analyzer.analyze([], GOOD_METRICS)
        assert result.ic_mean == 0
        assert result.ic_positive_ratio == 0

    def test_hit_rate(self):
        result = self.analyzer.analyze(STABLE_IC, EXCELLENT_METRICS)
        assert result.hit_rate == result.ic_positive_ratio

    def test_consecutive_negative(self):
        ic = [0.03, 0.02, -0.01, -0.02, -0.03, -0.01, -0.02, 0.03]
        result = self.analyzer.analyze(ic, GOOD_METRICS)
        assert result.max_consecutive_negative == 5

    def test_monthly_stats(self):
        result = self.analyzer.analyze(STABLE_IC, EXCELLENT_METRICS)
        assert result.best_month_ic > result.worst_month_ic


# ──────── FactorReviewer Tests ────────

class TestFactorReviewer:
    def setup_method(self):
        self.reviewer = FactorReviewer()

    def test_excellent_grade(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        assert review.grade == ReviewGrade.EXCELLENT
        assert review.score >= 80

    def test_good_grade(self):
        review = self.reviewer.review(backtest_metrics=GOOD_METRICS, ic_series=STABLE_IC)
        assert review.grade in (ReviewGrade.EXCELLENT, ReviewGrade.GOOD)
        assert review.score >= 60

    def test_fail_grade(self):
        review = self.reviewer.review(backtest_metrics=FAIL_METRICS)
        assert review.grade == ReviewGrade.FAIL

    def test_strengths_identified(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        assert len(review.strengths) >= 2

    def test_weaknesses_identified(self):
        review = self.reviewer.review(backtest_metrics=MEDIOCRE_METRICS, ic_series=UNSTABLE_IC)
        assert len(review.weaknesses) >= 1

    def test_risk_flags_data_snooping(self):
        review = self.reviewer.review(backtest_metrics=FAIL_METRICS)
        assert RiskFlag.DATA_SNOOPING in review.risk_flags

    def test_risk_flags_low_capacity(self):
        review = self.reviewer.review(backtest_metrics=FAIL_METRICS)
        assert RiskFlag.LOW_CAPACITY in review.risk_flags

    def test_risk_flags_overfitting(self):
        review = self.reviewer.review(backtest_metrics=GOOD_METRICS, ic_series=DECAYING_IC)
        # Decaying IC should trigger overfitting flag
        assert RiskFlag.OVERFITTING in review.risk_flags

    def test_recommendations_generated(self):
        review = self.reviewer.review(backtest_metrics=GOOD_METRICS, ic_series=STABLE_IC)
        assert len(review.recommendations) >= 1

    def test_summary_generated(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        assert len(review.summary) > 20
        assert "A" in review.summary

    def test_no_metrics(self):
        review = self.reviewer.review()
        assert review.grade == ReviewGrade.FAIL

    def test_score_range(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        assert 0 <= review.score <= 100

    def test_return_decomposition_populated(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS)
        assert review.return_decomposition.total_return == 0.834
        assert review.return_decomposition.factor_alpha != 0


# ──────── Output Format Tests ────────

class TestReviewOutput:
    def setup_method(self):
        self.reviewer = FactorReviewer()

    def test_to_dict(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        d = review.to_dict()
        assert "grade" in d
        assert "score" in d
        assert "strengths" in d
        assert "risk_flags" in d
        assert d["grade"] == "A"

    def test_to_markdown(self):
        review = self.reviewer.review(backtest_metrics=EXCELLENT_METRICS, ic_series=STABLE_IC)
        md = review.to_markdown()
        assert "# Factor Review Report" in md
        assert "Grade" in md
        assert "Strengths" in md
        assert "Recommendations" in md

    def test_to_markdown_with_risks(self):
        review = self.reviewer.review(backtest_metrics=FAIL_METRICS)
        md = review.to_markdown()
        assert "data_snooping" in md or "low_capacity" in md


# ──────── Batch Review Tests ────────

class TestBatchReview:
    def test_review_run_results(self):
        results = [
            {"iteration": 0, "skipped": True, "backtest_metrics": {}},
            {"iteration": 1, "skipped": False, "backtest_metrics": EXCELLENT_METRICS,
             "ic_series": STABLE_IC,
             "factor_code": "def calculate_factor(df): return df['close'].pct_change(20)"},
            {"iteration": 2, "skipped": False, "backtest_metrics": POOR_METRICS},
        ]
        reviews = review_run_results(results)
        assert len(reviews) == 2
        assert reviews[0].grade == ReviewGrade.EXCELLENT

    def test_empty_results(self):
        assert review_run_results([]) == []
