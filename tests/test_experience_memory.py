"""
Unit tests for the experience memory module.
测试经验记忆库。
"""

import json
import os
import tempfile

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experience_memory import Experience, ExperienceMemory


# ──────── Experience Tests ────────

class TestExperience:

    def test_auto_tags_momentum(self):
        """假设包含 momentum 时应自动提取标签。"""
        exp = Experience(iteration=0, hypothesis="Use 20-day momentum as factor", factor_code="")
        assert "momentum" in exp.tags

    def test_auto_tags_multiple(self):
        """假设包含多个关键词时应全部提取。"""
        exp = Experience(
            iteration=0,
            hypothesis="Combine volatility and volume breakout with RSI",
            factor_code="",
        )
        assert "volatility" in exp.tags
        assert "volume" in exp.tags
        assert "rsi" in exp.tags

    def test_auto_tags_empty_hypothesis(self):
        """空假设应产生空标签。"""
        exp = Experience(iteration=0, hypothesis="", factor_code="")
        assert exp.tags == []

    def test_manual_tags_override(self):
        """手动指定标签时不应自动提取。"""
        exp = Experience(
            iteration=0, hypothesis="momentum factor",
            factor_code="", tags=["custom_tag"],
        )
        assert exp.tags == ["custom_tag"]

    def test_timestamp_auto_set(self):
        """时间戳应自动设置为当前时间。"""
        exp = Experience(iteration=0, hypothesis="test", factor_code="")
        assert exp.timestamp > 0


# ──────── ExperienceMemory Tests ────────

class TestExperienceMemory:

    def _tmp_path(self) -> str:
        return os.path.join(tempfile.mkdtemp(), "test_memory.json")

    def test_empty_memory(self):
        """新建的记忆库应为空。"""
        mem = ExperienceMemory(self._tmp_path())
        assert len(mem.experiences) == 0
        assert mem.stats()["total"] == 0

    def test_add_and_retrieve(self):
        """添加经验后应能检索到。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(
            iteration=0,
            hypothesis="20-day momentum factor for trend following",
            factor_code="df['close'].pct_change(20)",
            metrics={"IC": 0.05},
            outcome="success",
        ))
        results = mem.retrieve("momentum trend")
        assert len(results) == 1
        assert results[0].iteration == 0

    def test_retrieve_empty_query(self):
        """空查询应返回空结果。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(iteration=0, hypothesis="test", factor_code=""))
        assert mem.retrieve("xyznonexistent") == []

    def test_retrieve_with_outcome_filter(self):
        """按结果过滤应只返回对应类型。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(
            iteration=0, hypothesis="momentum factor",
            factor_code="", outcome="success",
        ))
        mem.add(Experience(
            iteration=1, hypothesis="momentum reversal",
            factor_code="", outcome="filtered",
        ))
        successes = mem.retrieve("momentum", outcome_filter="success")
        assert len(successes) == 1
        assert successes[0].outcome == "success"

    def test_retrieve_top_k(self):
        """top_k 应限制返回数量。"""
        path = self._tmp_path()
        mem = ExperienceMemory(path)
        for i in range(10):
            mem.add(Experience(
                iteration=i, hypothesis=f"momentum factor variant {i}",
                factor_code="", outcome="success",
            ))
        results = mem.retrieve("momentum", top_k=3)
        assert len(results) == 3

    def test_persistence(self):
        """保存后重新加载应保留所有数据。"""
        path = self._tmp_path()
        mem1 = ExperienceMemory(path)
        mem1.add(Experience(
            iteration=0, hypothesis="volatility breakout",
            factor_code="code_here", metrics={"IC": 0.03},
            outcome="success", lesson="Works in high vol",
        ))

        # 重新加载
        mem2 = ExperienceMemory(path)
        assert len(mem2.experiences) == 1
        assert mem2.experiences[0].hypothesis == "volatility breakout"
        assert mem2.experiences[0].lesson == "Works in high vol"

    def test_corrupted_file_recovery(self):
        """损坏的文件应能恢复（清空重来）。"""
        path = self._tmp_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        mem = ExperienceMemory(path)
        assert len(mem.experiences) == 0

    def test_get_success_patterns(self):
        """应按 IC 排序返回最成功的经验。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(iteration=0, hypothesis="a", factor_code="",
                           metrics={"IC": 0.02}, outcome="success"))
        mem.add(Experience(iteration=1, hypothesis="b", factor_code="",
                           metrics={"IC": 0.08}, outcome="success"))
        mem.add(Experience(iteration=2, hypothesis="c", factor_code="",
                           metrics={"IC": 0.05}, outcome="success"))
        top = mem.get_success_patterns(top_k=2)
        assert len(top) == 2
        assert top[0].metrics["IC"] == 0.08
        assert top[1].metrics["IC"] == 0.05

    def test_get_failure_patterns(self):
        """应返回最近的失败经验。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(iteration=0, hypothesis="bad", factor_code="",
                           outcome="filtered", timestamp=100))
        mem.add(Experience(iteration=1, hypothesis="worse", factor_code="",
                           outcome="failed", timestamp=200))
        mem.add(Experience(iteration=2, hypothesis="good", factor_code="",
                           outcome="success", timestamp=300))
        failures = mem.get_failure_patterns(top_k=5)
        assert len(failures) == 2
        assert failures[0].timestamp == 200  # 最近的失败在前

    def test_build_context_prompt_empty(self):
        """空记忆库应返回空字符串。"""
        mem = ExperienceMemory(self._tmp_path())
        assert mem.build_context_prompt("momentum") == ""

    def test_build_context_prompt_with_data(self):
        """有数据时应生成包含成功/失败信息的上下文。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(
            iteration=0, hypothesis="momentum trend factor",
            factor_code="code", metrics={"IC": 0.05},
            outcome="success", lesson="Good in bull market",
        ))
        mem.add(Experience(
            iteration=1, hypothesis="momentum reversal",
            factor_code="code2", outcome="filtered",
            lesson="Too similar to existing",
        ))
        context = mem.build_context_prompt("momentum")
        assert "EXPERIENCE MEMORY" in context
        assert "Successful" in context
        assert "AVOID" in context

    def test_stats(self):
        """统计信息应正确。"""
        mem = ExperienceMemory(self._tmp_path())
        mem.add(Experience(iteration=0, hypothesis="momentum", factor_code="",
                           outcome="success"))
        mem.add(Experience(iteration=1, hypothesis="volatility", factor_code="",
                           outcome="filtered"))
        stats = mem.stats()
        assert stats["total"] == 2
        assert stats["outcomes"]["success"] == 1
        assert stats["outcomes"]["filtered"] == 1
        assert stats["unique_tags"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
