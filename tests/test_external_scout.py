"""
Tests for External Scout — 外部搜寻引擎
所有外部 API 调用均 mock，不会产生真实网络请求。
"""

import sys
import os
import json
import importlib.util
import tempfile
import shutil

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if "src" not in sys.modules:
    sys.modules["src"] = type(sys)("src")
    sys.modules["src"].__path__ = [_src_dir]
    sys.modules["src"].__package__ = "src"

spec = importlib.util.spec_from_file_location(
    "src.external_scout", os.path.join(_src_dir, "external_scout.py")
)
es_mod = importlib.util.module_from_spec(spec)
sys.modules["src.external_scout"] = es_mod
spec.loader.exec_module(es_mod)

ArxivScout = es_mod.ArxivScout
GitHubScout = es_mod.GitHubScout
ScoutPipeline = es_mod.ScoutPipeline
ScoutConfig = es_mod.ScoutConfig
ScoutResult = es_mod.ScoutResult
ScoutStatus = es_mod.ScoutStatus
SourceType = es_mod.SourceType

import pytest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# 样本数据
# ──────────────────────────────────────────────

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Alpha Factor Generation with LLM Agents</title>
    <summary>We propose a novel approach to generate formulaic alpha factors using
    large language models and multi-agent debate.</summary>
    <published>2026-03-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2603.12345" type="text/html"/>
    <id>http://arxiv.org/abs/2603.12345</id>
  </entry>
  <entry>
    <title>Deep Learning for Stock Prediction</title>
    <summary>A transformer-based model for financial market prediction.</summary>
    <published>2026-03-10T00:00:00Z</published>
    <author><name>Charlie Brown</name></author>
    <link href="http://arxiv.org/abs/2603.99999" type="text/html"/>
    <id>http://arxiv.org/abs/2603.99999</id>
  </entry>
</feed>"""

SAMPLE_GITHUB_JSON = {
    "items": [
        {
            "full_name": "user/alpha-factor-lib",
            "html_url": "https://github.com/user/alpha-factor-lib",
            "description": "A library for quantitative alpha factor generation and backtesting",
            "updated_at": "2026-03-20T10:00:00Z",
            "stargazers_count": 500,
            "language": "Python",
            "topics": ["quantitative-finance", "alpha", "factor"],
        },
        {
            "full_name": "user/trading-bot",
            "html_url": "https://github.com/user/trading-bot",
            "description": "Simple trading bot",
            "updated_at": "2026-03-18T10:00:00Z",
            "stargazers_count": 50,
            "language": "Python",
            "topics": [],
        },
    ]
}


# ──────────────────────────────────────────────
# ArxivScout Tests
# ──────────────────────────────────────────────

class TestArxivScout:
    def test_parse_arxiv_xml(self):
        """解析 arXiv XML 响应"""
        scout = ArxivScout()
        results = scout._parse_arxiv_xml(SAMPLE_ARXIV_XML)
        assert len(results) == 2
        assert results[0].title == "Alpha Factor Generation with LLM Agents"
        assert results[0].source == SourceType.ARXIV
        assert "Alice Smith" in results[0].authors
        assert results[0].published_date == "2026-03-15"

    def test_relevance_scoring(self):
        """相关度评分: 含高相关关键词的论文得分更高"""
        scout = ArxivScout()
        results = scout._parse_arxiv_xml(SAMPLE_ARXIV_XML)
        # 第一篇含 "alpha factor" → 高分
        # 第二篇只含 "financial market" → 较低分
        assert results[0].relevance_score > results[1].relevance_score

    def test_tag_extraction(self):
        """标签提取"""
        scout = ArxivScout()
        results = scout._parse_arxiv_xml(SAMPLE_ARXIV_XML)
        assert "alpha" in results[0].tags or "factor" in results[0].tags
        assert "multi-agent" in results[0].tags

    def test_empty_xml(self):
        """空 XML 不崩溃"""
        scout = ArxivScout()
        results = scout._parse_arxiv_xml("<feed xmlns='http://www.w3.org/2005/Atom'/>")
        assert results == []

    def test_malformed_xml(self):
        """错误 XML 不崩溃"""
        scout = ArxivScout()
        results = scout._parse_arxiv_xml("not xml at all")
        assert results == []

    @patch("src.external_scout.urllib.request.urlopen")
    def test_search_with_mock(self, mock_urlopen):
        """Mock API 调用测试"""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = SAMPLE_ARXIV_XML.encode("utf-8")
        mock_urlopen.return_value = mock_response

        scout = ArxivScout(max_results_per_query=5)
        results = scout.search("alpha factor")
        assert len(results) == 2


# ──────────────────────────────────────────────
# GitHubScout Tests
# ──────────────────────────────────────────────

class TestGitHubScout:
    def test_parse_github_response(self):
        """解析 GitHub API 响应"""
        scout = GitHubScout()
        results = scout._parse_github_response(SAMPLE_GITHUB_JSON)
        assert len(results) == 2
        assert results[0].title == "user/alpha-factor-lib"
        assert results[0].source == SourceType.GITHUB
        assert results[0].relevance_score == 0.5  # 500 stars / 1000

    def test_tags_from_topics(self):
        """从 GitHub topics 提取标签"""
        scout = GitHubScout()
        results = scout._parse_github_response(SAMPLE_GITHUB_JSON)
        assert "python" in results[0].tags
        assert "quantitative-finance" in results[0].tags

    def test_empty_response(self):
        """空响应"""
        scout = GitHubScout()
        results = scout._parse_github_response({"items": []})
        assert results == []


# ──────────────────────────────────────────────
# ScoutResult Tests
# ──────────────────────────────────────────────

class TestScoutResult:
    def test_to_dict_from_dict(self):
        """序列化/反序列化"""
        result = ScoutResult(
            source=SourceType.ARXIV,
            title="Test Paper",
            url="http://example.com",
            relevance_score=0.8,
            tags=["alpha", "factor"],
        )
        d = result.to_dict()
        restored = ScoutResult.from_dict(d)
        assert restored.title == "Test Paper"
        assert restored.source == SourceType.ARXIV
        assert restored.relevance_score == 0.8

    def test_status_transitions(self):
        """状态流转"""
        result = ScoutResult(
            source=SourceType.GITHUB, title="Repo", url="http://gh.com",
        )
        assert result.status == ScoutStatus.NEW
        result.status = ScoutStatus.EVALUATED
        assert result.status == ScoutStatus.EVALUATED
        result.status = ScoutStatus.APPROVED
        assert result.status == ScoutStatus.APPROVED


# ──────────────────────────────────────────────
# ScoutPipeline Tests
# ──────────────────────────────────────────────

class TestScoutPipeline:
    def test_init_default(self):
        """默认配置"""
        pipeline = ScoutPipeline()
        assert pipeline.arxiv is not None
        assert pipeline.github is not None

    def test_init_disable_sources(self):
        """禁用搜索源"""
        config = ScoutConfig(enable_arxiv=False, enable_github=False)
        pipeline = ScoutPipeline(config)
        assert pipeline.arxiv is None
        assert pipeline.github is None

    def test_search_with_mock_scouts(self):
        """Mock 搜索流程"""
        pipeline = ScoutPipeline(ScoutConfig(min_relevance=0.0))
        pipeline.arxiv = MagicMock()
        pipeline.github = MagicMock()

        pipeline.arxiv.search.return_value = [
            ScoutResult(SourceType.ARXIV, "Paper 1", "url1", relevance_score=0.8),
        ]
        pipeline.github.search.return_value = [
            ScoutResult(SourceType.GITHUB, "Repo 1", "url2", relevance_score=0.3),
        ]

        results = pipeline.search()
        assert len(results) == 2
        assert results[0].relevance_score == 0.8  # sorted by relevance

    def test_relevance_filter(self):
        """相关度过滤"""
        pipeline = ScoutPipeline(ScoutConfig(
            min_relevance=0.5, enable_github=False,
        ))
        pipeline.arxiv = MagicMock()
        pipeline.arxiv.search.return_value = [
            ScoutResult(SourceType.ARXIV, "High", "url1", relevance_score=0.9),
            ScoutResult(SourceType.ARXIV, "Low", "url2", relevance_score=0.1),
        ]

        results = pipeline.search()
        assert len(results) == 1
        assert results[0].title == "High"

    def test_evaluate_with_reviewer(self):
        """评估流程 (mock reviewer)"""
        mock_reviewer = MagicMock()
        mock_review = MagicMock()
        mock_review.grade.value = "B"
        mock_review.score = 75.0
        mock_reviewer.review.return_value = mock_review

        pipeline = ScoutPipeline(reviewer=mock_reviewer)

        results = [ScoutResult(
            SourceType.ARXIV, "Paper", "url",
            factor_code="df.pct_change(20)",
            status=ScoutStatus.NEW,
        )]

        evaluated = pipeline.evaluate(results)
        assert evaluated[0].status == ScoutStatus.APPROVED  # score >= 70
        assert "75" in evaluated[0].evaluation_notes

    def test_evaluate_low_score_stays_evaluated(self):
        """低分不自动通过"""
        mock_reviewer = MagicMock()
        mock_review = MagicMock()
        mock_review.grade.value = "D"
        mock_review.score = 40.0
        mock_reviewer.review.return_value = mock_review

        pipeline = ScoutPipeline(reviewer=mock_reviewer)
        results = [ScoutResult(
            SourceType.ARXIV, "Weak Paper", "url",
            factor_code="bad_code",
            status=ScoutStatus.NEW,
        )]

        evaluated = pipeline.evaluate(results)
        assert evaluated[0].status == ScoutStatus.EVALUATED  # 不是 APPROVED

    def test_integrate_with_library(self):
        """入库流程 (mock library)"""
        mock_library = MagicMock()

        pipeline = ScoutPipeline(library=mock_library)
        results = [ScoutResult(
            SourceType.GITHUB, "Factor Lib", "url",
            factor_code="df.rolling(20).mean()",
            status=ScoutStatus.APPROVED,
            tags=["alpha"],
        )]

        count = pipeline.integrate(results)
        assert count == 1
        assert results[0].status == ScoutStatus.INTEGRATED
        mock_library.add.assert_called_once()

    def test_run_full_pipeline(self):
        """完整流程 run()"""
        pipeline = ScoutPipeline(ScoutConfig(
            min_relevance=0.0,
            auto_evaluate=False,
            auto_integrate=False,
        ))
        pipeline.arxiv = MagicMock()
        pipeline.github = MagicMock()
        pipeline.arxiv.search.return_value = [
            ScoutResult(SourceType.ARXIV, "Paper", "url1", relevance_score=0.5),
        ]
        pipeline.github.search.return_value = []

        report = pipeline.run()
        assert report["total_found"] == 1
        assert report["by_source"]["arxiv"] == 1
        assert report["by_source"]["github"] == 0

    def test_save_load_history(self):
        """保存和加载搜索历史"""
        tmpdir = tempfile.mkdtemp()
        fp = os.path.join(tmpdir, "history.json")

        pipeline = ScoutPipeline()
        pipeline._results = [
            ScoutResult(SourceType.ARXIV, "Paper 1", "url1", relevance_score=0.7),
        ]
        pipeline.save_history(fp)

        # 新 pipeline 加载
        pipeline2 = ScoutPipeline()
        loaded = pipeline2.load_history(fp)
        assert len(loaded) == 1
        assert loaded[0].title == "Paper 1"

        shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────
# ScoutConfig Tests
# ──────────────────────────────────────────────

class TestScoutConfig:
    def test_defaults(self):
        config = ScoutConfig()
        assert config.enable_arxiv is True
        assert config.enable_github is True
        assert config.min_relevance == 0.1

    def test_custom(self):
        config = ScoutConfig(
            enable_arxiv=False,
            min_relevance=0.5,
            auto_evaluate=True,
        )
        assert config.enable_arxiv is False
        assert config.min_relevance == 0.5
        assert config.auto_evaluate is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
