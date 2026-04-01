"""
Factor Library — 因子库：存储、版本、标签、排行、搜索

Centralized storage for discovered factors. Supports versioning, tagging,
ranking, and similarity-based deduplication.

因子库是所有发现因子的中央仓库，提供：
1. 存储：因子代码 + 假设 + 回测指标 + 审视报告
2. 版本：同一因子的多个版本（参数优化后）
3. 标签：自动打标（momentum/mean_reversion/volume/...）
4. 排行：按 IC/Sharpe/综合分排序
5. 搜索：按关键词、标签、指标范围检索
6. 去重：基于 AST 相似度防止重复因子

Usage:
    library = FactorLibrary()
    library.add(factor_code, metrics, hypothesis="20-day momentum")
    top = library.get_top_factors(n=5, sort_by="sharpe_ratio")
    similar = library.find_similar(new_code, threshold=0.85)
"""

import ast
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

@dataclass
class FactorEntry:
    """因子库中的一条记录"""
    id: str = ""                    # 唯一 ID (基于代码 hash)
    name: str = ""                  # 因子名称
    version: int = 1                # 版本号
    factor_code: str = ""           # Python 代码
    hypothesis: str = ""            # 原始假设
    tags: list[str] = field(default_factory=list)
    backtest_metrics: dict = field(default_factory=dict)
    review_grade: str = ""          # A/B/C/D/F
    review_score: float = 0.0       # 0-100
    created_at: float = 0.0         # timestamp
    iteration: int = 0              # 来自哪次迭代
    run_id: str = ""                # 来自哪次运行
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "factor_code": self.factor_code,
            "hypothesis": self.hypothesis,
            "tags": self.tags,
            "backtest_metrics": self.backtest_metrics,
            "review_grade": self.review_grade,
            "review_score": self.review_score,
            "created_at": self.created_at,
            "iteration": self.iteration,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────── Core Library ────────

class FactorLibrary:
    """
    Centralized factor storage with search, ranking, and deduplication.
    因子库：集中管理所有发现的因子。

    Persistence: JSON file (factors.json in data/ directory)
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._factors: dict[str, FactorEntry] = {}  # id → entry
        self._storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "factor_library.json"
        )
        self._load()

    def _load(self):
        """从磁盘加载因子库"""
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, "r") as f:
                    data = json.load(f)
                for entry_dict in data:
                    entry = FactorEntry.from_dict(entry_dict)
                    self._factors[entry.id] = entry
                logger.info(f"[Library] Loaded {len(self._factors)} factors from {self._storage_path}")
            except Exception as e:
                logger.warning(f"[Library] Failed to load: {e}")

    def _save(self):
        """保存因子库到磁盘"""
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        data = [entry.to_dict() for entry in self._factors.values()]
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _compute_id(factor_code: str) -> str:
        """基于代码内容生成唯一 ID"""
        # 影响: 去除空白后 hash，使格式差异不影响 ID
        normalized = "".join(factor_code.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    @staticmethod
    def _auto_tag(factor_code: str) -> list[str]:
        """自动从代码中提取标签"""
        tags = []
        code_lower = factor_code.lower()

        tag_keywords = {
            "momentum": ["pct_change", "momentum", "trend", "breakout"],
            "mean_reversion": ["z_score", "zscore", "bollinger", "rsi", "revert", "oversold"],
            "volatility": ["std", "volatility", "atr", "true_range"],
            "volume": ["volume", "vwap", "obv", "money_flow"],
            "moving_average": ["rolling", "ewm", "ma_", "sma", "ema"],
            "cross_sectional": ["rank", "percentile", "quintile"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in code_lower for kw in keywords):
                tags.append(tag)

        return tags if tags else ["unknown"]

    def add(
        self,
        factor_code: str,
        backtest_metrics: Optional[dict] = None,
        hypothesis: str = "",
        name: str = "",
        review_grade: str = "",
        review_score: float = 0.0,
        iteration: int = 0,
        run_id: str = "",
        metadata: Optional[dict] = None,
    ) -> FactorEntry:
        """
        添加因子到库中。如果已存在相同代码的因子，创建新版本。

        Returns:
            FactorEntry 新添加/更新的记录
        """
        factor_id = self._compute_id(factor_code)

        # 影响: 如果已存在，递增版本号
        version = 1
        if factor_id in self._factors:
            version = self._factors[factor_id].version + 1

        entry = FactorEntry(
            id=factor_id,
            name=name or f"Factor_{factor_id[:6]}",
            version=version,
            factor_code=factor_code,
            hypothesis=hypothesis,
            tags=self._auto_tag(factor_code),
            backtest_metrics=backtest_metrics or {},
            review_grade=review_grade,
            review_score=review_score,
            created_at=time.time(),
            iteration=iteration,
            run_id=run_id,
            metadata=metadata or {},
        )

        self._factors[factor_id] = entry
        self._save()

        logger.info(f"[Library] Added {entry.name} v{version} (ID={factor_id}, tags={entry.tags})")
        return entry

    def get(self, factor_id: str) -> Optional[FactorEntry]:
        """按 ID 获取因子"""
        return self._factors.get(factor_id)

    def remove(self, factor_id: str) -> bool:
        """删除因子"""
        if factor_id in self._factors:
            del self._factors[factor_id]
            self._save()
            return True
        return False

    @property
    def size(self) -> int:
        return len(self._factors)

    def get_all(self) -> list[FactorEntry]:
        """获取所有因子"""
        return list(self._factors.values())

    def get_top_factors(
        self,
        n: int = 10,
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> list[FactorEntry]:
        """
        按指标排行，返回前 N 个因子。

        Args:
            n: 返回数量
            sort_by: 排序字段 (IC, sharpe_ratio, annual_return, review_score)
            ascending: 是否升序
        """
        factors = list(self._factors.values())

        if sort_by == "review_score":
            factors.sort(key=lambda f: f.review_score, reverse=not ascending)
        else:
            factors.sort(
                key=lambda f: f.backtest_metrics.get(sort_by, float("-inf")),
                reverse=not ascending,
            )

        return factors[:n]

    def search(
        self,
        keyword: str = "",
        tags: Optional[list[str]] = None,
        min_ic: Optional[float] = None,
        min_sharpe: Optional[float] = None,
        min_grade: Optional[str] = None,
    ) -> list[FactorEntry]:
        """
        搜索因子库。

        Args:
            keyword: 在假设和代码中搜索
            tags: 按标签过滤
            min_ic: 最低 IC
            min_sharpe: 最低 Sharpe
            min_grade: 最低评级 (A > B > C > D > F)
        """
        results = list(self._factors.values())

        if keyword:
            kw_lower = keyword.lower()
            results = [f for f in results
                       if kw_lower in f.hypothesis.lower()
                       or kw_lower in f.factor_code.lower()
                       or kw_lower in f.name.lower()]

        if tags:
            results = [f for f in results if any(t in f.tags for t in tags)]

        if min_ic is not None:
            results = [f for f in results if f.backtest_metrics.get("IC", 0) >= min_ic]

        if min_sharpe is not None:
            results = [f for f in results if f.backtest_metrics.get("sharpe_ratio", 0) >= min_sharpe]

        if min_grade:
            grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "": 0}
            min_val = grade_order.get(min_grade, 0)
            results = [f for f in results if grade_order.get(f.review_grade, 0) >= min_val]

        return results

    def find_similar(self, factor_code: str, threshold: float = 0.85) -> list[tuple[FactorEntry, float]]:
        """
        查找与给定代码相似的已有因子（基于 AST 结构）。

        Returns:
            list of (entry, similarity_score) where similarity >= threshold
        """
        try:
            new_nodes = self._get_ast_node_types(factor_code)
        except SyntaxError:
            return []

        if not new_nodes:
            return []

        results = []
        for entry in self._factors.values():
            try:
                existing_nodes = self._get_ast_node_types(entry.factor_code)
                if not existing_nodes:
                    continue
                similarity = self._jaccard_similarity(new_nodes, existing_nodes)
                if similarity >= threshold:
                    results.append((entry, similarity))
            except SyntaxError:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _get_ast_node_types(code: str) -> list[str]:
        """提取代码的 AST 节点类型序列"""
        try:
            tree = ast.parse(code)
            return [type(node).__name__ for node in ast.walk(tree)]
        except SyntaxError:
            return []

    @staticmethod
    def _jaccard_similarity(a: list[str], b: list[str]) -> float:
        """计算两个列表的 Jaccard 相似度"""
        set_a = set(a)
        set_b = set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def to_markdown_table(self, factors: Optional[list[FactorEntry]] = None) -> str:
        """生成因子库的 Markdown 排行表"""
        entries = factors or self.get_top_factors(n=20)
        if not entries:
            return "No factors in library."

        lines = ["| Rank | Name | Tags | IC | Sharpe | Grade | Score |",
                 "|------|------|------|-----|--------|-------|-------|"]
        for i, f in enumerate(entries, 1):
            ic = f.backtest_metrics.get("IC", 0)
            sharpe = f.backtest_metrics.get("sharpe_ratio", 0)
            tags = ", ".join(f.tags[:3])
            lines.append(
                f"| {i} | {f.name} | {tags} | {ic:.4f} | {sharpe:.2f} | "
                f"{f.review_grade or '-'} | {f.review_score:.0f} |"
            )
        return "\n".join(lines)
