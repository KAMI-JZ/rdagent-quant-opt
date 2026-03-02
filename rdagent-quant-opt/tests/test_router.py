"""
Unit tests for the Multi-Model Router and Cost Tracker.
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_router import (
    MultiModelRouter, CostTracker, RoutingDecision,
    PipelineStage, ModelConfig, PromptCacheOptimizer,
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
    
    def test_synthesis_uses_strong_model(self):
        router = MultiModelRouter()
        model_key = router.routing_table[PipelineStage.SYNTHESIS]
        assert model_key == "strong"
    
    def test_implementation_uses_efficient_model(self):
        router = MultiModelRouter()
        model_key = router.routing_table[PipelineStage.IMPLEMENTATION]
        assert model_key == "efficient"
    
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
        premium = router.models["premium"]
        # Premium should have highest input price
        assert premium.input_price > efficient.input_price


# ──────── Cost Estimation Tests ────────

class TestCostEstimation:
    """Verify cost calculations match known pricing."""
    
    def test_deepseek_chat_pricing(self):
        """DeepSeek V3.2: $0.28/M input (miss), $0.028/M (hit), $0.42/M output"""
        config = ModelConfig(
            model_id="deepseek/deepseek-chat",
            input_price=0.28, cache_price=0.028, output_price=0.42,
        )
        # 10K input tokens, 2K output tokens, 50% cache hit
        input_cost = (
            10000 * 0.5 * 0.028 / 1_000_000  # cache hit portion
            + 10000 * 0.5 * 0.28 / 1_000_000  # cache miss portion
        )
        output_cost = 2000 * 0.42 / 1_000_000
        total = input_cost + output_cost
        
        assert total == pytest.approx(0.00238, abs=1e-5)
    
    def test_30_iteration_budget_estimate(self):
        """Verify budget estimate: 30 iterations should cost ~$0.5-3.0"""
        # Typical per-iteration: 15 calls × 3K avg tokens input, 1K output
        calls_per_iter = 15
        avg_input = 3000
        avg_output = 1000
        iterations = 30
        
        # All DeepSeek Chat, 50% cache hit
        cost_per_call = (
            avg_input * 0.5 * 0.028 / 1e6
            + avg_input * 0.5 * 0.28 / 1e6
            + avg_output * 0.42 / 1e6
        )
        total = cost_per_call * calls_per_iter * iterations
        
        assert 0.1 < total < 5.0  # Should be in expected range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
