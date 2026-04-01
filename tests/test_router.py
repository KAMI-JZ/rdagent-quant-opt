"""
Unit tests for the Multi-Model Router, Cost Tracker, and AdaptiveModelSelector.
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_router import (
    MultiModelRouter, CostTracker, RoutingDecision,
    PipelineStage, ModelConfig, PromptCacheOptimizer,
    AdaptiveModelSelector, MODEL_REGISTRY,
)


# ──────── Cost Tracker Tests ────────

class TestCostTracker:

    def test_empty_tracker(self):
        tracker = CostTracker(daily_budget_usd=5.0)
        assert tracker.get_today_cost() == 0.0
        assert not tracker.is_over_budget()

    def test_record_and_sum(self):
        tracker = CostTracker(daily_budget_usd=5.0)
        tracker.record(RoutingDecision(
            timestamp=time.time(),
            stage=PipelineStage.IMPLEMENTATION,
            model_used="deepseek/deepseek-chat",
            input_tokens=1000, output_tokens=500, cost_usd=0.001,
        ))
        tracker.record(RoutingDecision(
            timestamp=time.time(),
            stage=PipelineStage.SYNTHESIS,
            model_used="deepseek/deepseek-reasoner",
            input_tokens=5000, output_tokens=1000, cost_usd=0.005,
        ))
        assert tracker.get_today_cost() == pytest.approx(0.006, abs=1e-6)

    def test_budget_exceeded(self):
        tracker = CostTracker(daily_budget_usd=0.01)
        tracker.record(RoutingDecision(
            timestamp=time.time(),
            stage=PipelineStage.IMPLEMENTATION,
            model_used="test",
            cost_usd=0.02,
        ))
        assert tracker.is_over_budget()

    def test_stage_breakdown(self):
        tracker = CostTracker()
        for _ in range(10):
            tracker.record(RoutingDecision(
                timestamp=time.time(),
                stage=PipelineStage.IMPLEMENTATION,
                model_used="cheap", cost_usd=0.001,
            ))
        for _ in range(2):
            tracker.record(RoutingDecision(
                timestamp=time.time(),
                stage=PipelineStage.SYNTHESIS,
                model_used="strong", cost_usd=0.005,
            ))

        breakdown = tracker.get_stage_breakdown()
        assert breakdown["implementation"]["calls"] == 10
        assert breakdown["synthesis"]["calls"] == 2
        assert breakdown["implementation"]["cost_usd"] == pytest.approx(0.01)
        assert breakdown["synthesis"]["cost_usd"] == pytest.approx(0.01)


# ──────── Prompt Cache Optimizer Tests ────────

class TestPromptCacheOptimizer:

    def test_system_message_stays_first(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "Generate a hypothesis."},
        ]
        optimized = PromptCacheOptimizer.optimize_messages(
            messages, PipelineStage.SYNTHESIS
        )
        assert optimized[0]["role"] == "system"
        assert optimized[0]["content"] == "You are a researcher."

    def test_preserves_all_messages(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "User 2"},
        ]
        optimized = PromptCacheOptimizer.optimize_messages(
            messages, PipelineStage.IMPLEMENTATION
        )
        assert len(optimized) == len(messages)


# ──────── Router Configuration Tests ────────

class TestRouterConfig:

    def test_default_routing_table(self):
        router = MultiModelRouter()
        assert PipelineStage.SYNTHESIS in router.routing_table
        assert PipelineStage.IMPLEMENTATION in router.routing_table
        assert PipelineStage.ANALYSIS in router.routing_table

    def test_synthesis_uses_frontier_model(self):
        """Synthesis needs highest creativity → frontier tier."""
        router = MultiModelRouter()
        model_key = router.routing_table[PipelineStage.SYNTHESIS]
        assert model_key == "frontier"

    def test_implementation_uses_efficient_model(self):
        router = MultiModelRouter()
        model_key = router.routing_table[PipelineStage.IMPLEMENTATION]
        assert model_key == "efficient"

    def test_analysis_uses_strong_model(self):
        router = MultiModelRouter()
        model_key = router.routing_table[PipelineStage.ANALYSIS]
        assert model_key == "strong"

    def test_custom_routing(self):
        custom = {
            PipelineStage.SYNTHESIS: "premium",
            PipelineStage.IMPLEMENTATION: "efficient",
            PipelineStage.ANALYSIS: "premium",
        }
        router = MultiModelRouter(routing_table=custom)
        assert router.routing_table[PipelineStage.SYNTHESIS] == "premium"

    def test_model_price_ordering(self):
        router = MultiModelRouter()
        efficient = router.models["efficient"]
        strong = router.models["strong"]
        frontier = router.models["frontier"]
        assert frontier.input_price > strong.input_price
        assert strong.input_price > efficient.input_price


# ──────── AdaptiveModelSelector Tests ────────

class TestAdaptiveModelSelector:

    def test_synthesis_gets_frontier_in_premium_mode(self):
        """因子构思(高创造力+低调用量) → premium模式下选 frontier。"""
        selector = AdaptiveModelSelector(budget_mode="premium")
        cfg = selector.select(PipelineStage.SYNTHESIS)
        assert cfg.tier == "frontier"

    def test_synthesis_gets_strong_in_optimized_mode(self):
        """optimized 模式下 frontier 降级为 strong。"""
        selector = AdaptiveModelSelector(budget_mode="optimized")
        cfg = selector.select(PipelineStage.SYNTHESIS)
        assert cfg.tier == "strong"

    def test_synthesis_gets_efficient_in_budget_mode(self):
        """budget 模式全部降级为 efficient。"""
        selector = AdaptiveModelSelector(budget_mode="budget")
        cfg = selector.select(PipelineStage.SYNTHESIS)
        assert cfg.tier == "efficient"

    def test_implementation_always_efficient(self):
        """Implementation 高调用量+有重试 → 任何模式都选 efficient。"""
        for mode in ["budget", "optimized", "premium"]:
            selector = AdaptiveModelSelector(budget_mode=mode)
            cfg = selector.select(PipelineStage.IMPLEMENTATION)
            assert cfg.tier == "efficient", f"Failed for mode={mode}"

    def test_analysis_gets_strong(self):
        """Analysis 高推理需求 → strong 或更高。"""
        selector = AdaptiveModelSelector(budget_mode="premium")
        cfg = selector.select(PipelineStage.ANALYSIS)
        assert cfg.tier in ("strong", "frontier")

    def test_all_stages_return_valid_config(self):
        """所有 stage 都能返回有效的 ModelConfig。"""
        selector = AdaptiveModelSelector(budget_mode="optimized")
        for stage in PipelineStage:
            if stage in (PipelineStage.VALIDATION, PipelineStage.EMBEDDING):
                continue
            cfg = selector.select(stage)
            assert isinstance(cfg, ModelConfig)
            assert cfg.model_id

    def test_router_adaptive_mode(self):
        """Router 启用 adaptive_mode 后自动生成路由表。"""
        router = MultiModelRouter(adaptive_mode="optimized")
        # 应该有路由表
        assert PipelineStage.SYNTHESIS in router.routing_table
        assert PipelineStage.IMPLEMENTATION in router.routing_table
        # Implementation 应该指向 efficient 模型
        impl_key = router.routing_table[PipelineStage.IMPLEMENTATION]
        impl_model = router.models[impl_key]
        assert impl_model.tier == "efficient"


# ──────── Model Registry Tests ────────

class TestModelRegistry:

    def test_registry_has_latest_models(self):
        """验证注册表包含最新模型。"""
        assert "claude-opus-4-6" in MODEL_REGISTRY
        assert "claude-sonnet-4-6" in MODEL_REGISTRY
        assert "deepseek-reasoner" in MODEL_REGISTRY
        assert "deepseek-chat" in MODEL_REGISTRY

    def test_opus_is_frontier_tier(self):
        assert MODEL_REGISTRY["claude-opus-4-6"].tier == "frontier"

    def test_sonnet_is_strong_tier(self):
        assert MODEL_REGISTRY["claude-sonnet-4-6"].tier == "strong"

    def test_deepseek_chat_is_efficient_tier(self):
        assert MODEL_REGISTRY["deepseek-chat"].tier == "efficient"

    def test_no_outdated_sonnet_4(self):
        """确认没有过时的 Sonnet 4 (claude-sonnet-4-20250514)。"""
        for name, cfg in MODEL_REGISTRY.items():
            assert "20250514" not in cfg.model_id, f"Outdated model ID in {name}"


# ──────── Cost Estimation Tests ────────

class TestCostEstimation:
    """Verify cost calculations match known pricing."""

    def test_deepseek_chat_pricing(self):
        """DeepSeek V3: $0.27/M input (miss), $0.07/M (hit), $1.10/M output"""
        config = MODEL_REGISTRY["deepseek-chat"]
        # 10K input tokens, 2K output, 50% cache hit
        input_cost = (
            10000 * 0.5 * config.cache_price / 1_000_000
            + 10000 * 0.5 * config.input_price / 1_000_000
        )
        output_cost = 2000 * config.output_price / 1_000_000
        total = input_cost + output_cost
        assert total > 0
        assert total < 0.01  # Should be very cheap

    def test_30_iteration_budget_estimate(self):
        """Verify budget estimate: 30 iterations should cost ~$0.5-5.0 with DeepSeek"""
        config = MODEL_REGISTRY["deepseek-chat"]
        calls_per_iter = 15
        avg_input = 3000
        avg_output = 1000
        iterations = 30

        cost_per_call = (
            avg_input * 0.5 * config.cache_price / 1e6
            + avg_input * 0.5 * config.input_price / 1e6
            + avg_output * config.output_price / 1e6
        )
        total = cost_per_call * calls_per_iter * iterations
        assert 0.1 < total < 10.0

    def test_frontier_much_more_expensive_than_efficient(self):
        """Frontier 模型应该比 efficient 贵很多倍。"""
        frontier = MODEL_REGISTRY["claude-opus-4-6"]
        efficient = MODEL_REGISTRY["deepseek-chat"]
        assert frontier.input_price / efficient.input_price > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
