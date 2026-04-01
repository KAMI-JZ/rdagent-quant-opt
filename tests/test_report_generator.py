"""Tests for ReportGenerator — 报告生成器测试"""

import json
import os
import tempfile
import pytest

from src.report_generator import ReportGenerator, ReportConfig
from src.pipeline import PipelineReport, IterationResult
from src.alpha_filter import FilterResult
from src.debate_agents import DebateResult, Verdict


# ──────── Helpers ────────

def _make_result(iteration, ic=0.03, icir=0.15, sharpe=0.85,
                 verdict=Verdict.CONTINUE, skipped=False, skip_reason="",
                 hypothesis="test hypothesis", simulated=True):
    """Create an IterationResult for testing."""
    r = IterationResult(iteration=iteration)
    if skipped:
        r.skipped = True
        r.skip_reason = skip_reason
        r.filter_result = FilterResult(
            passed=False, factor_code="", rejection_reasons=[skip_reason]
        ) if "similar" in skip_reason or "complex" in skip_reason else None
    else:
        r.hypothesis = hypothesis
        r.factor_code = f"def factor_{iteration}(df): return df['close'].pct_change()"
        r.backtest_metrics = {
            "IC": ic, "ICIR": icir, "sharpe_ratio": sharpe,
            "annual_return": 0.08, "max_drawdown": -0.12,
            "_simulated": simulated,
        }
        r.debate_result = DebateResult(
            bull_argument="good", bear_argument="bad",
            verdict=verdict, confidence=0.7,
            synthesis="test synthesis",
            recommended_action="continue exploring",
        )
    r.cost_usd = 0.01
    r.duration_sec = 5.0
    return r


def _make_report(n_completed=5, n_skipped=2):
    """Create a PipelineReport for testing."""
    report = PipelineReport()
    results = []

    for i in range(n_completed):
        r = _make_result(
            iteration=i,
            ic=0.02 + i * 0.005,
            verdict=[Verdict.CONTINUE, Verdict.PIVOT, Verdict.NEUTRAL][i % 3],
        )
        results.append(r)
        report.iterations_completed += 1
        v = r.debate_result.verdict.value.lower()
        report.verdicts[v] = report.verdicts.get(v, 0) + 1

    for i in range(n_skipped):
        r = _make_result(
            iteration=n_completed + i,
            skipped=True,
            skip_reason="Too similar to existing factor",
        )
        results.append(r)
        report.iterations_skipped += 1

    report.results = results
    report.total_cost_usd = 0.07
    report.total_duration_sec = 35.0
    report.best_metrics = {"IC": 0.04, "iteration": 4}

    return report


# ──────── Tests ────────

class TestReportGenerator:

    def test_generate_creates_md_file(self, tmp_path):
        """报告生成器创建 .md 文件"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        filepath = gen.generate(report, run_label="test_run")
        assert os.path.exists(filepath)
        assert filepath.endswith(".md")

    def test_generate_creates_json_file(self, tmp_path):
        """同时生成 JSON 文件"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        gen.generate(report, run_label="test_run")
        json_path = os.path.join(str(tmp_path), "report_test_run.json")
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["run_label"] == "test_run"
        assert data["summary"]["iterations_completed"] == 5

    def test_overview_section(self, tmp_path):
        """总览包含关键指标"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        filepath = gen.generate(report, run_label="overview_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "5 / 7" in content       # 完成数 / 总数
        assert "$0.07" in content        # 成本
        assert "Best IC" in content      # 最佳 IC

    def test_iteration_table(self, tmp_path):
        """迭代明细表格存在且行数正确"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=3, n_skipped=1)
        filepath = gen.generate(report, run_label="table_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Iteration Details" in content
        # 3 completed + 1 skipped = 4 data rows
        assert content.count("| 0 |") >= 1
        assert content.count("SKIP") >= 1

    def test_verdict_analysis(self, tmp_path):
        """辩论裁决分布"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=6, n_skipped=0)
        filepath = gen.generate(report, run_label="verdict_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Debate Verdicts" in content
        assert "CONTINUE" in content
        assert "PIVOT" in content

    def test_filter_analysis(self, tmp_path):
        """过滤器分析"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=3, n_skipped=4)
        filepath = gen.generate(report, run_label="filter_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Filter Analysis" in content

    def test_cost_breakdown_with_summary(self, tmp_path):
        """有成本摘要时显示分阶段成本"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        cost_summary = {
            "total_cost_usd": 0.07,
            "total_calls": 15,
            "daily_budget_usd": 5.0,
            "budget_remaining_usd": 4.93,
            "stage_breakdown": {
                "synthesis": {"calls": 5, "cost_usd": 0.04, "input_tokens": 5000, "output_tokens": 3000},
                "implementation": {"calls": 10, "cost_usd": 0.03, "input_tokens": 8000, "output_tokens": 4000},
            },
            "cost_distribution": {"synthesis": "57.1%", "implementation": "42.9%"},
        }
        filepath = gen.generate(report, cost_summary=cost_summary, run_label="cost_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Cost Breakdown" in content
        assert "synthesis" in content
        assert "57.1%" in content

    def test_cost_breakdown_without_summary(self, tmp_path):
        """无成本摘要时仍正常生成"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        filepath = gen.generate(report, run_label="no_cost_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Cost Breakdown" in content
        assert "not available" in content

    def test_best_worst_comparison(self, tmp_path):
        """最佳/最差迭代对比"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=5, n_skipped=0)
        filepath = gen.generate(report, run_label="bw_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Best vs Worst" in content

    def test_recommendations_simulated_data(self, tmp_path):
        """模拟数据时给出警告"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report()
        filepath = gen.generate(report, run_label="rec_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Simulated data" in content

    def test_recommendations_low_pass_rate(self, tmp_path):
        """通过率低时建议降低阈值"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=2, n_skipped=5)
        filepath = gen.generate(report, run_label="low_pass_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Low pass rate" in content

    def test_metric_trends(self, tmp_path):
        """指标趋势分析"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=8, n_skipped=0)
        filepath = gen.generate(report, run_label="trend_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Metric Trends" in content
        assert "IC range" in content

    def test_sparkline(self):
        """文本迷你图生成"""
        line = ReportGenerator._sparkline([0, 0.25, 0.5, 0.75, 1.0])
        assert len(line) == 5
        assert line[0] in " ▁"   # 最低值
        assert line[-1] in "▇█"  # 最高值

    def test_sparkline_empty(self):
        """空数据的迷你图"""
        assert ReportGenerator._sparkline([]) == ""

    def test_sparkline_constant(self):
        """常数值的迷你图"""
        line = ReportGenerator._sparkline([0.5, 0.5, 0.5])
        assert len(line) == 3

    def test_categorize_reason(self):
        """拦截原因归类"""
        assert "similar" in ReportGenerator._categorize_reason("Too similar to factor_3").lower()
        assert "complex" in ReportGenerator._categorize_reason("Nesting depth exceeds 5").lower()
        assert "align" in ReportGenerator._categorize_reason("Alignment score too low").lower()
        assert "other" in ReportGenerator._categorize_reason("Unknown error").lower()

    def test_auto_run_label(self, tmp_path):
        """不提供 label 时自动生成时间戳标签"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=2, n_skipped=0)
        filepath = gen.generate(report)
        assert os.path.exists(filepath)
        assert "report_20" in filepath  # starts with year

    def test_json_metrics(self, tmp_path):
        """JSON 包含正确的指标统计"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=3, n_skipped=0)
        gen.generate(report, run_label="json_metrics")
        json_path = os.path.join(str(tmp_path), "report_json_metrics.json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["metrics"]["ic_max"] >= data["metrics"]["ic_min"]
        assert len(data["metrics"]["ic_values"]) == 3
        assert len(data["iterations"]) == 3

    def test_empty_report(self, tmp_path):
        """空报告不报错"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = PipelineReport()
        filepath = gen.generate(report, run_label="empty_test")
        assert os.path.exists(filepath)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Pipeline Run Report" in content

    def test_high_pivot_warning(self, tmp_path):
        """PIVOT 率过高时给出警告"""
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        report = _make_report(n_completed=0, n_skipped=0)
        # 手动构建高 PIVOT 率
        for i in range(10):
            r = _make_result(iteration=i, verdict=Verdict.PIVOT)
            report.results.append(r)
            report.iterations_completed += 1
        report.verdicts = {"continue": 1, "pivot": 8, "neutral": 1}
        filepath = gen.generate(report, run_label="pivot_test")
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "PIVOT rate > 60%" in content
