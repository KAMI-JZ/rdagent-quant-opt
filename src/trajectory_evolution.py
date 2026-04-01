"""
Trajectory Evolution — 轨迹级进化

Treats consecutive iterations as "trajectories" and applies evolutionary
operations (mutation, crossover) on successful trajectories to produce
better starting points for future runs.

将连续迭代看作"轨迹"，对成功轨迹进行进化操作（变异、交叉），
为后续运行生成更好的起点。

Based on QuantaAlpha (arXiv:2602.07085) trajectory-level evolution concept.

Design:
- Zero LLM cost for evolution operations (string manipulation only)
- LLM cost only when generating new hypotheses from evolved templates
- Uses ExperienceMemory as the storage backend
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from .experience_memory import Experience, ExperienceMemory

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    A sequence of related iterations forming a trajectory.
    一组相关迭代形成的轨迹。
    """
    experiences: list[Experience] = field(default_factory=list)
    total_ic: float = 0.0  # 轨迹累计 IC
    verdict_sequence: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.experiences)

    @property
    def avg_ic(self) -> float:
        if not self.experiences:
            return 0.0
        ics = [e.metrics.get("IC", 0) for e in self.experiences]
        return sum(ics) / len(ics)

    @property
    def hypotheses(self) -> list[str]:
        return [e.hypothesis for e in self.experiences]

    @property
    def best_experience(self) -> Optional[Experience]:
        if not self.experiences:
            return None
        return max(self.experiences, key=lambda e: e.metrics.get("IC", 0))


class TrajectoryBuilder:
    """
    Segments experience history into trajectories based on debate verdicts.
    根据辩论裁决将经验历史分段为轨迹。

    规则:
    - CONTINUE verdict → 继续当前轨迹
    - PIVOT verdict → 结束当前轨迹，开始新的
    - NEUTRAL → 继续当前轨迹
    """

    def build(self, experiences: list[Experience]) -> list[Trajectory]:
        """
        Segment experiences into trajectories.
        将经验按辩论裁决分段为轨迹。
        """
        if not experiences:
            return []

        trajectories = []
        current = Trajectory()

        for exp in sorted(experiences, key=lambda e: e.iteration):
            if exp.verdict == "pivot" and current.length > 0:
                # PIVOT = 结束当前轨迹，开始新的
                trajectories.append(current)
                current = Trajectory()

            current.experiences.append(exp)
            current.verdict_sequence.append(exp.verdict)
            current.total_ic += exp.metrics.get("IC", 0)

        if current.length > 0:
            trajectories.append(current)

        logger.info(
            f"[Evolution] Built {len(trajectories)} trajectories "
            f"from {len(experiences)} experiences"
        )
        return trajectories


class TrajectoryEvolver:
    """
    Applies evolutionary operations on trajectories to generate new starting points.
    对轨迹进行进化操作以生成新的起点。

    Operations:
    1. Mutation: 修改成功轨迹中的假设关键词
    2. Crossover: 组合两条成功轨迹的最佳元素
    3. Selection: 按 IC 排序选择最佳轨迹

    零 LLM 成本: 所有操作都是字符串级别的变换。
    """

    # 功能: 变异关键词表——将一个策略关键词替换为同类型的另一个
    MUTATION_MAP = {
        "momentum": ["reversal", "trend", "breakout"],
        "reversal": ["momentum", "mean_revert", "contrarian"],
        "volatility": ["volume", "liquidity", "beta"],
        "volume": ["volatility", "turnover", "liquidity"],
        "value": ["quality", "growth", "earnings"],
        "quality": ["value", "profitability", "stability"],
        "growth": ["value", "earnings", "revenue"],
        "short_term": ["long_term", "medium_term"],
        "long_term": ["short_term", "medium_term"],
        "5": ["10", "20"],
        "10": ["5", "20", "30"],
        "20": ["10", "30", "60"],
        "30": ["20", "60"],
    }

    def __init__(self, mutation_rate: float = 0.3, seed: Optional[int] = None):
        """
        Args:
            mutation_rate: 变异概率 (0-1)，每个关键词被替换的概率
            seed: 随机种子，用于可复现的测试
        """
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)

    def select_top(self, trajectories: list[Trajectory], top_k: int = 3) -> list[Trajectory]:
        """
        Select top trajectories by average IC.
        按平均 IC 选择最优轨迹。
        """
        sorted_t = sorted(trajectories, key=lambda t: t.avg_ic, reverse=True)
        return sorted_t[:top_k]

    def mutate(self, hypothesis: str) -> str:
        """
        Mutate a hypothesis by replacing strategy keywords.
        通过替换策略关键词变异假设。

        影响: 零 LLM 成本，纯字符串操作。
        保留假设的整体结构，只改变个别关键词。
        """
        words = hypothesis.split()
        mutated = []
        for word in words:
            word_lower = word.lower().strip(".,;:!?()[]")
            if word_lower in self.MUTATION_MAP and self.rng.random() < self.mutation_rate:
                alternatives = self.MUTATION_MAP[word_lower]
                replacement = self.rng.choice(alternatives)
                # 保持原始大小写
                if word[0].isupper():
                    replacement = replacement.capitalize()
                mutated.append(replacement)
            else:
                mutated.append(word)
        return " ".join(mutated)

    def crossover(self, hyp_a: str, hyp_b: str) -> str:
        """
        Crossover two hypotheses — take first half of A, second half of B.
        交叉两个假设——取 A 的前半段和 B 的后半段。

        影响: 零 LLM 成本，纯字符串操作。
        可能产生语法不通的句子，但会被 LLM 在 synthesis 阶段重新理解。
        """
        words_a = hyp_a.split()
        words_b = hyp_b.split()
        mid_a = len(words_a) // 2
        mid_b = len(words_b) // 2
        return " ".join(words_a[:mid_a] + words_b[mid_b:])

    def evolve(
        self,
        trajectories: list[Trajectory],
        n_offspring: int = 3,
    ) -> list[str]:
        """
        Generate evolved hypothesis templates from top trajectories.
        从最优轨迹生成进化后的假设模板。

        Args:
            trajectories: 所有轨迹
            n_offspring: 生成几个后代假设

        Returns:
            进化后的假设模板列表（零 LLM 成本）
        """
        top = self.select_top(trajectories, top_k=max(2, n_offspring))
        if not top:
            return []

        offspring = []

        # 功能: 对最佳轨迹的最佳假设进行变异
        for traj in top[:n_offspring]:
            best = traj.best_experience
            if best:
                mutated = self.mutate(best.hypothesis)
                offspring.append(mutated)
                logger.debug(f"[Evolution] Mutated: {best.hypothesis[:50]}... → {mutated[:50]}...")

        # 功能: 如果有多条轨迹，做交叉
        if len(top) >= 2:
            for i in range(min(n_offspring, len(top) - 1)):
                best_a = top[i].best_experience
                best_b = top[i + 1].best_experience
                if best_a and best_b:
                    crossed = self.crossover(best_a.hypothesis, best_b.hypothesis)
                    offspring.append(crossed)
                    logger.debug(f"[Evolution] Crossover: {crossed[:80]}...")

        logger.info(f"[Evolution] Generated {len(offspring)} evolved hypotheses")
        return offspring[:n_offspring * 2]  # 返回变异 + 交叉的结果

    def build_evolution_prompt(self, evolved_hypotheses: list[str]) -> str:
        """
        Build a prompt context from evolved hypotheses.
        从进化后的假设构建 prompt 上下文。

        影响: 注入到 synthesis prompt 中，让 LLM 参考进化后的模板。
        """
        if not evolved_hypotheses:
            return ""

        lines = [
            "[EVOLUTIONARY HINTS — Promising directions from past trajectories]",
            "Consider these evolved starting points (you may modify freely):",
        ]
        for i, hyp in enumerate(evolved_hypotheses[:5]):
            lines.append(f"  {i+1}. {hyp}")

        return "\n".join(lines)
