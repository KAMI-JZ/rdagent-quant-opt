"""
External Scout — 外部搜寻引擎

定期通过标准化流程搜寻权威社区（arXiv、GitHub、量化论坛）的
新因子、工具和研究成果，纳入因子审核流程。

核心流程: 搜索 → 解析 → 评估 → 审核 → 入库
Skill 互调: ScoutResult → FactorReviewer → FactorLibrary

基于用户需求: "固定频率通过标准化流程去权威社区搜寻可使用和更新的功能或因子，
将其纳入到因子审核的流程中，然后决定是否加入因子库，允许skill之间互相调用。"
"""

import json
import logging
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """搜索源类型"""
    ARXIV = "arxiv"
    GITHUB = "github"
    CUSTOM = "custom"


class ScoutStatus(Enum):
    """搜索结果状态"""
    NEW = "new"                 # 新发现
    EVALUATED = "evaluated"     # 已评估
    APPROVED = "approved"       # 已审核通过
    REJECTED = "rejected"       # 已拒绝
    INTEGRATED = "integrated"   # 已入库


@dataclass
class ScoutResult:
    """搜索结果的标准化格式"""
    source: SourceType
    title: str
    url: str
    description: str = ""
    authors: list[str] = field(default_factory=list)
    published_date: str = ""
    relevance_score: float = 0.0  # 相关度评分 0-1
    factor_code: str = ""          # 提取的因子代码 (如有)
    tags: list[str] = field(default_factory=list)
    status: ScoutStatus = ScoutStatus.NEW
    evaluation_notes: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "authors": self.authors,
            "published_date": self.published_date,
            "relevance_score": self.relevance_score,
            "factor_code": self.factor_code,
            "tags": self.tags,
            "status": self.status.value,
            "evaluation_notes": self.evaluation_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScoutResult":
        return cls(
            source=SourceType(data.get("source", "custom")),
            title=data.get("title", ""),
            url=data.get("url", ""),
            description=data.get("description", ""),
            authors=data.get("authors", []),
            published_date=data.get("published_date", ""),
            relevance_score=data.get("relevance_score", 0.0),
            factor_code=data.get("factor_code", ""),
            tags=data.get("tags", []),
            status=ScoutStatus(data.get("status", "new")),
            evaluation_notes=data.get("evaluation_notes", ""),
        )


# ──────────────────────────────────────────────
# arXiv 搜索器
# ──────────────────────────────────────────────

class ArxivScout:
    """
    搜索 arXiv 论文，寻找量化因子相关研究。
    使用 arXiv API (无需 API key)。
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    # 影响: 关键词集合决定搜索范围，可扩展
    DEFAULT_QUERIES = [
        "quantitative factor alpha stock",
        "alpha factor mining machine learning",
        "formulaic alpha generation LLM",
    ]

    def __init__(self, max_results_per_query: int = 5):
        self.max_results = max_results_per_query

    def search(self, query: str | None = None, max_results: int | None = None) -> list[ScoutResult]:
        """
        搜索 arXiv。
        Args:
            query: 搜索关键词 (None = 使用默认关键词组)
            max_results: 每个查询返回最多几条
        """
        max_r = max_results or self.max_results
        queries = [query] if query else self.DEFAULT_QUERIES
        all_results = []

        for q in queries:
            try:
                results = self._fetch_arxiv(q, max_r)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"[ArxivScout] Search failed for '{q}': {e}")

        # 去重 (按 URL)
        seen = set()
        unique = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info(f"[ArxivScout] Found {len(unique)} unique results")
        return unique

    def _fetch_arxiv(self, query: str, max_results: int) -> list[ScoutResult]:
        """调用 arXiv API 并解析 XML 响应。"""
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{self.BASE_URL}?{params}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read().decode("utf-8")

        return self._parse_arxiv_xml(xml_data)

    def _parse_arxiv_xml(self, xml_data: str) -> list[ScoutResult]:
        """解析 arXiv Atom XML 响应。"""
        results = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return results

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
            published = published_el.text.strip()[:10] if published_el is not None and published_el.text else ""

            # 提取 URL
            url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "text/html":
                    url = link.get("href", "")
                    break
            if not url:
                id_el = entry.find("atom:id", ns)
                url = id_el.text.strip() if id_el is not None and id_el.text else ""

            # 提取作者
            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # 计算相关度 (简单关键词匹配)
            relevance = self._compute_relevance(title + " " + summary)

            # 提取标签
            tags = self._extract_tags(title + " " + summary)

            results.append(ScoutResult(
                source=SourceType.ARXIV,
                title=title,
                url=url,
                description=summary[:500],
                authors=authors,
                published_date=published,
                relevance_score=relevance,
                tags=tags,
            ))

        return results

    @staticmethod
    def _compute_relevance(text: str) -> float:
        """计算文本与量化因子的相关度。"""
        text_lower = text.lower()
        # 影响: 这些关键词决定相关度评分
        high_relevance = ["alpha factor", "formulaic alpha", "quantitative factor",
                          "factor mining", "alpha generation", "stock prediction"]
        medium_relevance = ["trading strategy", "portfolio", "backtest",
                            "financial", "market", "investment", "return prediction"]

        score = 0.0
        for kw in high_relevance:
            if kw in text_lower:
                score += 0.15
        for kw in medium_relevance:
            if kw in text_lower:
                score += 0.05

        return min(1.0, score)

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """从文本提取标签。"""
        keywords = {
            "momentum", "reversal", "volatility", "value", "quality",
            "sentiment", "factor", "alpha", "deep learning", "transformer",
            "reinforcement learning", "LLM", "GPT", "multi-agent",
        }
        text_lower = text.lower()
        return [k for k in keywords if k in text_lower]


# ──────────────────────────────────────────────
# GitHub 搜索器
# ──────────────────────────────────────────────

class GitHubScout:
    """
    搜索 GitHub 仓库，寻找量化因子工具和库。
    使用 GitHub Search API (无需 token，但有速率限制)。
    """

    BASE_URL = "https://api.github.com/search/repositories"
    DEFAULT_QUERIES = [
        "alpha factor generation python",
        "quantitative trading factor",
    ]

    def __init__(self, max_results_per_query: int = 5, token: str = ""):
        self.max_results = max_results_per_query
        self.token = token  # 可选 GitHub token 提高速率限制

    def search(self, query: str | None = None, max_results: int | None = None) -> list[ScoutResult]:
        max_r = max_results or self.max_results
        queries = [query] if query else self.DEFAULT_QUERIES
        all_results = []

        for q in queries:
            try:
                results = self._fetch_github(q, max_r)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"[GitHubScout] Search failed for '{q}': {e}")

        # 去重
        seen = set()
        unique = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info(f"[GitHubScout] Found {len(unique)} unique results")
        return unique

    def _fetch_github(self, query: str, max_results: int) -> list[ScoutResult]:
        """调用 GitHub Search API。"""
        params = urllib.parse.urlencode({
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": max_results,
        })
        url = f"{self.BASE_URL}?{params}"

        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "RD-Agent-Quant-Optimizer")
        if self.token:
            req.add_header("Authorization", f"token {self.token}")

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        return self._parse_github_response(data)

    def _parse_github_response(self, data: dict) -> list[ScoutResult]:
        """解析 GitHub API 响应。"""
        results = []
        for item in data.get("items", []):
            results.append(ScoutResult(
                source=SourceType.GITHUB,
                title=item.get("full_name", ""),
                url=item.get("html_url", ""),
                description=item.get("description", "") or "",
                published_date=item.get("updated_at", "")[:10],
                relevance_score=min(1.0, item.get("stargazers_count", 0) / 1000),
                tags=self._extract_tags(item),
            ))
        return results

    @staticmethod
    def _extract_tags(item: dict) -> list[str]:
        """从 GitHub repo 信息提取标签。"""
        tags = []
        if item.get("language"):
            tags.append(item["language"].lower())
        for topic in item.get("topics", []):
            tags.append(topic)
        return tags[:10]


# ──────────────────────────────────────────────
# Scout Pipeline — 完整搜索→评估→入库流程
# 影响: 这是 skill 互调的核心，连接 Scout → Reviewer → Library
# ──────────────────────────────────────────────

@dataclass
class ScoutConfig:
    """搜寻配置"""
    enable_arxiv: bool = True
    enable_github: bool = True
    arxiv_max_results: int = 5
    github_max_results: int = 5
    min_relevance: float = 0.1      # 最低相关度阈值
    auto_evaluate: bool = False      # 是否自动评估 (需要 FactorReviewer)
    auto_integrate: bool = False     # 是否自动入库 (需要 FactorLibrary)


class ScoutPipeline:
    """
    完整的外部搜寻流程:
    1. 搜索: ArxivScout + GitHubScout
    2. 过滤: 相关度阈值
    3. 评估: (可选) 调用 FactorReviewer
    4. 入库: (可选) 调用 FactorLibrary

    Skill 互调示例:
        pipeline = ScoutPipeline(config, reviewer=my_reviewer, library=my_library)
        report = pipeline.run()
    """

    def __init__(
        self,
        config: ScoutConfig | None = None,
        reviewer=None,      # FactorReviewer 实例 (可选)
        library=None,        # FactorLibrary 实例 (可选)
    ):
        self.config = config or ScoutConfig()
        self.reviewer = reviewer
        self.library = library

        self.arxiv = ArxivScout(self.config.arxiv_max_results) if self.config.enable_arxiv else None
        self.github = GitHubScout(self.config.github_max_results) if self.config.enable_github else None

        self._results: list[ScoutResult] = []
        self._history_file = "data/scout_history.json"

    def search(self, query: str | None = None) -> list[ScoutResult]:
        """执行搜索，返回过滤后的结果。"""
        results = []

        if self.arxiv:
            results.extend(self.arxiv.search(query))
        if self.github:
            results.extend(self.github.search(query))

        # 过滤低相关度
        filtered = [r for r in results if r.relevance_score >= self.config.min_relevance]
        filtered.sort(key=lambda r: r.relevance_score, reverse=True)

        self._results = filtered
        logger.info(f"[ScoutPipeline] {len(results)} found, {len(filtered)} after filter")
        return filtered

    def evaluate(self, results: list[ScoutResult] | None = None) -> list[ScoutResult]:
        """
        评估搜索结果 (需要 FactorReviewer)。
        对有因子代码的结果进行审核评分。
        """
        targets = results or self._results
        if not self.reviewer:
            logger.warning("[ScoutPipeline] No reviewer configured, skipping evaluation")
            return targets

        for result in targets:
            if result.factor_code and result.status == ScoutStatus.NEW:
                try:
                    # Skill 互调: 调用 FactorReviewer
                    review = self.reviewer.review(
                        factor_code=result.factor_code,
                        metrics={},
                        hypothesis=result.title,
                    )
                    result.status = ScoutStatus.EVALUATED
                    result.evaluation_notes = (
                        f"Grade: {review.grade.value}, Score: {review.score:.0f}/100"
                    )
                    # 高分自动审核通过
                    if review.score >= 70:
                        result.status = ScoutStatus.APPROVED
                except Exception as e:
                    logger.warning(f"[ScoutPipeline] Review failed for '{result.title}': {e}")

        return targets

    def integrate(self, results: list[ScoutResult] | None = None) -> int:
        """
        将审核通过的结果入库 (需要 FactorLibrary)。
        返回成功入库的数量。
        """
        targets = results or self._results
        if not self.library:
            logger.warning("[ScoutPipeline] No library configured, skipping integration")
            return 0

        integrated = 0
        for result in targets:
            if result.status == ScoutStatus.APPROVED and result.factor_code:
                try:
                    # Skill 互调: 调用 FactorLibrary
                    self.library.add(
                        code=result.factor_code,
                        hypothesis=result.title,
                        metrics={},
                        tags=result.tags + ["external", result.source.value],
                    )
                    result.status = ScoutStatus.INTEGRATED
                    integrated += 1
                except Exception as e:
                    logger.warning(f"[ScoutPipeline] Integration failed for '{result.title}': {e}")

        logger.info(f"[ScoutPipeline] Integrated {integrated} factors into library")
        return integrated

    def run(self, query: str | None = None) -> dict:
        """
        运行完整流程: 搜索 → 评估 → 入库。
        返回运行报告。
        """
        results = self.search(query)

        if self.config.auto_evaluate and self.reviewer:
            self.evaluate(results)

        if self.config.auto_integrate and self.library:
            integrated = self.integrate(results)
        else:
            integrated = 0

        report = {
            "total_found": len(results),
            "by_source": {
                "arxiv": len([r for r in results if r.source == SourceType.ARXIV]),
                "github": len([r for r in results if r.source == SourceType.GITHUB]),
            },
            "status_counts": {},
            "integrated": integrated,
            "top_results": [r.to_dict() for r in results[:5]],
        }

        for r in results:
            report["status_counts"][r.status.value] = \
                report["status_counts"].get(r.status.value, 0) + 1

        return report

    def save_history(self, filepath: str | None = None):
        """保存搜索历史到 JSON。"""
        fp = filepath or self._history_file
        import os
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        data = {"results": [r.to_dict() for r in self._results], "timestamp": time.time()}
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_history(self, filepath: str | None = None) -> list[ScoutResult]:
        """加载搜索历史。"""
        fp = filepath or self._history_file
        import os
        if not os.path.exists(fp):
            return []
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._results = [ScoutResult.from_dict(r) for r in data.get("results", [])]
        return self._results
