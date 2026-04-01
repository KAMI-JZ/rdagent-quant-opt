"""
Experience Memory — 经验记忆库

Stores structured records of past iterations (hypothesis, code, metrics, verdict)
and retrieves relevant experiences to guide future factor generation.

存储每轮迭代的结构化记录（假设、代码、指标、裁决），
检索相关经验来指导后续因子生成，避免重复失败。

Based on FactorMiner (arXiv:2602.14670) "Ralph Loop" concept:
retrieve → generate → evaluate → distill.

Design:
- Zero LLM cost: all retrieval is keyword/similarity based, no API calls
- Local JSON storage: persists across runs
- Lightweight: no database dependency
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """
    A single iteration's experience record.
    单轮迭代的经验记录。
    """
    iteration: int
    hypothesis: str
    factor_code: str
    metrics: dict = field(default_factory=dict)
    verdict: str = ""              # "continue" / "pivot" / "neutral"
    outcome: str = ""              # "success" / "filtered" / "failed"
    lesson: str = ""               # 经验教训（自动生成或手动标注）
    tags: list[str] = field(default_factory=list)  # 关键词标签，用于检索
    timestamp: float = 0.0
    run_id: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        # 影响: 自动从假设中提取关键词标签，用于后续检索
        if not self.tags and self.hypothesis:
            self.tags = self._extract_tags(self.hypothesis)

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """Extract keyword tags from hypothesis text. 从假设文本提取关键词。"""
        # 功能: 简单的关键词提取，不需要 NLP 库
        keywords = {
            "momentum", "reversal", "volatility", "volume", "value",
            "quality", "growth", "size", "liquidity", "earnings",
            "dividend", "beta", "correlation", "mean_revert", "trend",
            "rsi", "macd", "bollinger", "moving_average", "breakout",
            "sector", "rotation", "sentiment", "options", "skew",
            "turnover", "short_interest", "insider", "buyback",
        }
        text_lower = text.lower()
        return [k for k in keywords if k in text_lower]


class ExperienceMemory:
    """
    Persistent experience store with keyword-based retrieval.
    持久化经验库，支持关键词检索。

    Usage:
        memory = ExperienceMemory("data/experience.json")
        memory.add(Experience(iteration=0, hypothesis="...", factor_code="...", ...))
        relevant = memory.retrieve("momentum volatility", top_k=3)
    """

    def __init__(self, filepath: str = "data/experience.json"):
        self.filepath = filepath
        self.experiences: list[Experience] = []
        self._load()

    def _load(self):
        """Load experiences from disk. 从磁盘加载经验记录。"""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.experiences = [
                Experience(**record) for record in data.get("experiences", [])
            ]
            logger.info(f"[Memory] Loaded {len(self.experiences)} experiences from {self.filepath}")
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"[Memory] Failed to load {self.filepath}: {e}, starting fresh")
            self.experiences = []

    def save(self):
        """Persist experiences to disk. 将经验记录保存到磁盘。"""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        data = {"experiences": [asdict(exp) for exp in self.experiences]}
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[Memory] Saved {len(self.experiences)} experiences to {self.filepath}")

    def add(self, experience: Experience):
        """Add a new experience and persist. 添加新经验并保存。"""
        self.experiences.append(experience)
        self.save()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        outcome_filter: Optional[str] = None,
    ) -> list[Experience]:
        """
        Retrieve most relevant experiences by keyword matching.
        通过关键词匹配检索最相关的经验。

        Args:
            query: 检索关键词（空格分隔）
            top_k: 返回最多几条
            outcome_filter: 可选，只返回某种结果 ("success" / "filtered" / "failed")

        Returns:
            按相关度排序的经验列表
        """
        query_tags = Experience._extract_tags(query)
        query_words = set(query.lower().split())

        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]

        scored = []
        for exp in candidates:
            # 功能: 两层匹配——标签精确匹配 + 假设文本模糊匹配
            tag_overlap = len(set(exp.tags) & set(query_tags))
            word_overlap = sum(1 for w in query_words if w in exp.hypothesis.lower())
            score = tag_overlap * 2 + word_overlap  # 标签匹配权重更高
            if score > 0:
                scored.append((score, exp))

        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [exp for _, exp in scored[:top_k]]

    def get_success_patterns(self, top_k: int = 5) -> list[Experience]:
        """Get top successful experiences (by IC). 获取最成功的经验（按 IC 排序）。"""
        successes = [e for e in self.experiences if e.outcome == "success"]
        successes.sort(key=lambda e: e.metrics.get("IC", 0), reverse=True)
        return successes[:top_k]

    def get_failure_patterns(self, top_k: int = 5) -> list[Experience]:
        """Get recent failures to avoid. 获取最近的失败经验以避免重蹈覆辙。"""
        failures = [e for e in self.experiences if e.outcome in ("filtered", "failed")]
        failures.sort(key=lambda e: -e.timestamp)
        return failures[:top_k]

    def build_context_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Build a prompt context string from relevant experiences.
        从相关经验构建 prompt 上下文字符串。

        影响: 这个字符串会被注入到 synthesis prompt 中，
        让 LLM 知道之前哪些尝试成功了、哪些失败了。
        不需要额外 LLM 调用，零成本。
        """
        relevant = self.retrieve(query, top_k=top_k)
        if not relevant:
            return ""

        lines = ["[EXPERIENCE MEMORY — Past iterations for reference]"]

        successes = [e for e in relevant if e.outcome == "success"]
        failures = [e for e in relevant if e.outcome != "success"]

        if successes:
            lines.append("\nSuccessful approaches:")
            for exp in successes:
                ic = exp.metrics.get("IC", "?")
                lines.append(f"  - Iter {exp.iteration}: {exp.hypothesis[:80]}... (IC={ic})")
                if exp.lesson:
                    lines.append(f"    Lesson: {exp.lesson}")

        if failures:
            lines.append("\nFailed/filtered approaches (AVOID these):")
            for exp in failures:
                lines.append(f"  - Iter {exp.iteration}: {exp.hypothesis[:80]}...")
                reason = exp.lesson or exp.outcome
                lines.append(f"    Reason: {reason}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Get memory statistics. 获取记忆库统计信息。"""
        outcomes = {}
        for exp in self.experiences:
            outcomes[exp.outcome] = outcomes.get(exp.outcome, 0) + 1
        return {
            "total": len(self.experiences),
            "outcomes": outcomes,
            "unique_tags": len({t for e in self.experiences for t in e.tags}),
        }
