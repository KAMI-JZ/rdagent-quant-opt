"""
Unit tests for trajectory evolution module.
测试轨迹级进化。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experience_memory import Experience
from src.trajectory_evolution import (
    Trajectory, TrajectoryBuilder, TrajectoryEvolver,
)


# ──────── Trajectory Tests ────────

class TestTrajectory:

    def test_empty_trajectory(self):
        """空轨迹的属性应正确。"""
        t = Trajectory()
        assert t.length == 0
        assert t.avg_ic == 0.0
        assert t.hypotheses == []
        assert t.best_experience is None

    def test_avg_ic(self):
        """平均 IC 应正确计算。"""
        t = Trajectory(experiences=[
            Experience(iteration=0, hypothesis="a", factor_code="", metrics={"IC": 0.02}),
            Experience(iteration=1, hypothesis="b", factor_code="", metrics={"IC": 0.06}),
        ])
        assert abs(t.avg_ic - 0.04) < 1e-6

    def test_best_experience(self):
        """应返回 IC 最高的经验。"""
        exp_high = Experience(iteration=1, hypothesis="high", factor_code="",
                              metrics={"IC": 0.08})
        t = Trajectory(experiences=[
            Experience(iteration=0, hypothesis="low", factor_code="", metrics={"IC": 0.01}),
            exp_high,
        ])
        assert t.best_experience == exp_high


# ──────── TrajectoryBuilder Tests ────────

class TestTrajectoryBuilder:

    def test_single_trajectory(self):
        """所有 CONTINUE → 一条轨迹。"""
        builder = TrajectoryBuilder()
        exps = [
            Experience(iteration=0, hypothesis="a", factor_code="", verdict="continue"),
            Experience(iteration=1, hypothesis="b", factor_code="", verdict="continue"),
            Experience(iteration=2, hypothesis="c", factor_code="", verdict="continue"),
        ]
        trajs = builder.build(exps)
        assert len(trajs) == 1
        assert trajs[0].length == 3

    def test_pivot_splits_trajectory(self):
        """PIVOT 应分割轨迹。"""
        builder = TrajectoryBuilder()
        exps = [
            Experience(iteration=0, hypothesis="a", factor_code="", verdict="continue"),
            Experience(iteration=1, hypothesis="b", factor_code="", verdict="pivot"),
            Experience(iteration=2, hypothesis="c", factor_code="", verdict="continue"),
        ]
        trajs = builder.build(exps)
        assert len(trajs) == 2
        assert trajs[0].length == 1  # 第一条只有 "a"
        assert trajs[1].length == 2  # pivot 后 "b" 开始新轨迹（包含 b 和 c）

    def test_empty_input(self):
        """空输入应返回空列表。"""
        builder = TrajectoryBuilder()
        assert builder.build([]) == []

    def test_neutral_continues(self):
        """NEUTRAL 应继续当前轨迹。"""
        builder = TrajectoryBuilder()
        exps = [
            Experience(iteration=0, hypothesis="a", factor_code="", verdict="neutral"),
            Experience(iteration=1, hypothesis="b", factor_code="", verdict="neutral"),
        ]
        trajs = builder.build(exps)
        assert len(trajs) == 1


# ──────── TrajectoryEvolver Tests ────────

class TestTrajectoryEvolver:

    def _make_trajectory(self, ic: float, hypothesis: str) -> Trajectory:
        return Trajectory(experiences=[
            Experience(iteration=0, hypothesis=hypothesis, factor_code="",
                       metrics={"IC": ic}, outcome="success"),
        ])

    def test_select_top(self):
        """应按 IC 排序选择。"""
        evolver = TrajectoryEvolver(seed=42)
        trajs = [
            self._make_trajectory(0.01, "bad"),
            self._make_trajectory(0.08, "best"),
            self._make_trajectory(0.05, "mid"),
        ]
        top = evolver.select_top(trajs, top_k=2)
        assert len(top) == 2
        assert top[0].avg_ic == 0.08

    def test_mutate_changes_hypothesis(self):
        """变异应改变假设内容。"""
        evolver = TrajectoryEvolver(mutation_rate=1.0, seed=42)
        original = "Use momentum and volatility for trend following"
        mutated = evolver.mutate(original)
        # 高变异率下应该有变化
        assert mutated != original

    def test_mutate_zero_rate_no_change(self):
        """mutation_rate=0 时不应有变化。"""
        evolver = TrajectoryEvolver(mutation_rate=0.0, seed=42)
        original = "Use momentum for trend"
        assert evolver.mutate(original) == original

    def test_crossover(self):
        """交叉应混合两个假设。"""
        evolver = TrajectoryEvolver(seed=42)
        a = "The first hypothesis about momentum and trends"
        b = "The second hypothesis about volatility and mean reversion"
        crossed = evolver.crossover(a, b)
        # 应包含 A 的前半部分和 B 的后半部分
        assert len(crossed.split()) > 0
        # 不应完全等于 A 或 B
        assert crossed != a
        assert crossed != b

    def test_evolve_returns_offspring(self):
        """进化应返回后代假设。"""
        evolver = TrajectoryEvolver(seed=42)
        trajs = [
            self._make_trajectory(0.05, "momentum based factor with 20 day window"),
            self._make_trajectory(0.03, "volatility breakout strategy for growth stocks"),
        ]
        offspring = evolver.evolve(trajs, n_offspring=3)
        assert len(offspring) > 0
        assert all(isinstance(h, str) for h in offspring)

    def test_evolve_empty_trajectories(self):
        """空轨迹应返回空列表。"""
        evolver = TrajectoryEvolver(seed=42)
        assert evolver.evolve([], n_offspring=3) == []

    def test_build_evolution_prompt(self):
        """应生成包含进化假设的 prompt。"""
        evolver = TrajectoryEvolver(seed=42)
        hypotheses = ["mutated hypothesis 1", "crossed hypothesis 2"]
        prompt = evolver.build_evolution_prompt(hypotheses)
        assert "EVOLUTIONARY HINTS" in prompt
        assert "mutated hypothesis 1" in prompt

    def test_build_evolution_prompt_empty(self):
        """空假设应返回空字符串。"""
        evolver = TrajectoryEvolver(seed=42)
        assert evolver.build_evolution_prompt([]) == ""

    def test_deterministic_with_seed(self):
        """相同种子应产生相同结果。"""
        trajs = [
            self._make_trajectory(0.05, "momentum factor with volatility filter"),
            self._make_trajectory(0.03, "value quality earnings growth"),
        ]
        r1 = TrajectoryEvolver(seed=123).evolve(trajs, n_offspring=2)
        r2 = TrajectoryEvolver(seed=123).evolve(trajs, n_offspring=2)
        assert r1 == r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
