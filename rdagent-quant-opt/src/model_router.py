"""
Multi-Model Router for RD-Agent

Routes different RD-Agent pipeline stages to different LLM backends based on
task characteristics and cost-performance analysis.

Architecture insight from paper analysis (arXiv:2505.15155v2):
- Implementation Unit (Co-STEER) consumes 60-70% of LLM budget but has self-correction
- Synthesis & Analysis Units consume 15-20% but determine iteration direction
- Validation Unit uses zero LLM calls (pure CPU backtesting)

Routing strategy: Use strong models for low-frequency high-impact calls (Synthesis, Analysis),
use cost-efficient models for high-frequency self-correcting calls (Implementation).
"""

import os
import re
import json
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """RD-Agent(Q) pipeline stages with their characteristics."""
    SPECIFICATION = "specification"   # Template assembly, ~0 LLM calls
    SYNTHESIS = "synthesis"           # Hypothesis generation, 1-2 calls, HIGH impact
    IMPLEMENTATION = "implementation" # Code generation + debug, 10-15 calls, self-correcting
    VALIDATION = "validation"         # Qlib backtesting, 0 LLM calls
    ANALYSIS = "analysis"             # Strategy evaluation, 1-2 calls, HIGH impact
    EMBEDDING = "embedding"           # Knowledge base retrieval
    UNKNOWN = "unknown"               # Fallback


@dataclass
class ModelConfig:
    """Configuration for a model endpoint."""
    model_id: str           # e.g. "deepseek/deepseek-chat"
    input_price: float      # $ per 1M tokens (cache miss)
    cache_price: float      # $ per 1M tokens (cache hit)
    output_price: float     # $ per 1M tokens
    max_tokens: int = 4096
    temperature: float = 0.8
    supports_cache: bool = True


@dataclass
class RoutingDecision:
    """Record of a routing decision for cost tracking."""
    timestamp: float
    stage: PipelineStage
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Stage Classifier: Identifies which pipeline stage a prompt belongs to
# ──────────────────────────────────────────────

class StageClassifier:
    """
    Classifies LLM requests into pipeline stages based on prompt content.
    
    This is necessary because RD-Agent's architecture passes all LLM calls
    through a single global API backend (rdagent.oai.backend.LiteLLMAPIBackend),
    without exposing which module initiated the call.
    
    Classification uses keyword/pattern matching on the system prompt and
    user message content. Patterns are derived from RD-Agent's actual prompt
    templates (documented in arXiv:2505.15155v2 Appendix E).
    """
    
    # Patterns derived from RD-Agent source code prompt templates
    STAGE_PATTERNS = {
        PipelineStage.SYNTHESIS: [
            # Hypothesis generation prompts
            r"generate.*(?:hypothesis|factor|idea|proposal)",
            r"(?:propose|suggest|create).*(?:new|novel).*(?:factor|model|feature)",
            r"based on.*(?:previous|historical).*(?:experiment|result|feedback)",
            r"(?:knowledge forest|idea forest|experiment trajectory)",
            r"(?:SOTA|state.of.the.art).*(?:factor|model).*(?:list|library)",
            r"formulate.*(?:hypothesis|research question)",
            r"经济学原理|因子假设|信号方向",  # Chinese prompts from fin_factor_report
        ],
        PipelineStage.IMPLEMENTATION: [
            # Code generation and debugging prompts
            r"(?:implement|code|write|generate).*(?:python|function|class|factor)",
            r"(?:error|traceback|exception|bug|fix|debug)",
            r"(?:qlib|pandas|numpy|dataframe).*(?:factor|feature|signal)",
            r"def (?:calculate|compute|get)_factor",
            r"(?:execution|runtime).*(?:error|failed|timeout)",
            r"(?:refine|improve|fix).*(?:code|implementation)",
            r"(?:knowledge base|similar task|previous implementation)",
            r"import (?:pandas|numpy|qlib)",
        ],
        PipelineStage.ANALYSIS: [
            # Result analysis and feedback prompts
            r"(?:analyze|evaluate|assess).*(?:result|performance|experiment)",
            r"(?:IC|ICIR|ARR|MDD|Sharpe|Calmar|drawdown).*\d",
            r"(?:outperform|underperform|compare).*(?:SOTA|baseline|benchmark)",
            r"(?:diagnose|explain).*(?:failure|success|improvement)",
            r"(?:suggest|recommend).*(?:next|direction|refinement)",
            r"回测结果|指标分析|改进方向",  # Chinese analysis prompts
        ],
        PipelineStage.SPECIFICATION: [
            r"(?:specification|scenario|constraint|interface|schema)",
            r"(?:data field|output format|backtesting environment)",
        ],
    }
    
    def classify(self, messages: list[dict]) -> PipelineStage:
        """Classify a chat completion request into a pipeline stage."""
        # Combine all message content for pattern matching
        full_text = " ".join(
            msg.get("content", "") for msg in messages 
            if isinstance(msg.get("content"), str)
        ).lower()
        
        # Score each stage
        scores = {}
        for stage, patterns in self.STAGE_PATTERNS.items():
            score = sum(
                1 for pattern in patterns 
                if re.search(pattern, full_text, re.IGNORECASE)
            )
            if score > 0:
                scores[stage] = score
        
        if not scores:
            return PipelineStage.UNKNOWN
        
        # Return highest scoring stage
        return max(scores, key=scores.get)


# ──────────────────────────────────────────────
# Cost Tracker: Real-time cost monitoring
# ──────────────────────────────────────────────

class CostTracker:
    """
    Tracks LLM usage costs across pipeline stages with daily budget control.
    
    Key insight: Implementation (Co-STEER) consumes 60-70% of budget.
    This tracker helps validate that routing actually shifts costs as intended.
    """
    
    def __init__(self, daily_budget_usd: float = 5.0):
        self.daily_budget = daily_budget_usd
        self.records: list[RoutingDecision] = []
        self._day_start = time.time()
    
    def record(self, decision: RoutingDecision):
        """Record a routing decision and its cost."""
        self.records.append(decision)
        logger.info(
            f"[CostTracker] {decision.stage.value}: "
            f"model={decision.model_used}, "
            f"tokens={decision.input_tokens}+{decision.output_tokens}, "
            f"cost=${decision.cost_usd:.6f}"
        )
    
    def get_today_cost(self) -> float:
        """Get total cost for today."""
        cutoff = self._day_start
        return sum(r.cost_usd for r in self.records if r.timestamp >= cutoff)
    
    def is_over_budget(self) -> bool:
        """Check if daily budget is exceeded."""
        return self.get_today_cost() >= self.daily_budget
    
    def get_stage_breakdown(self) -> dict:
        """Get cost breakdown by pipeline stage."""
        breakdown = {}
        for record in self.records:
            stage = record.stage.value
            if stage not in breakdown:
                breakdown[stage] = {
                    "calls": 0, "cost_usd": 0.0,
                    "input_tokens": 0, "output_tokens": 0
                }
            breakdown[stage]["calls"] += 1
            breakdown[stage]["cost_usd"] += record.cost_usd
            breakdown[stage]["input_tokens"] += record.input_tokens
            breakdown[stage]["output_tokens"] += record.output_tokens
        return breakdown
    
    def get_summary(self) -> dict:
        """Get full cost summary."""
        breakdown = self.get_stage_breakdown()
        total = sum(s["cost_usd"] for s in breakdown.values())
        total_calls = sum(s["calls"] for s in breakdown.values())
        
        return {
            "total_cost_usd": round(total, 6),
            "total_calls": total_calls,
            "daily_budget_usd": self.daily_budget,
            "budget_remaining_usd": round(self.daily_budget - self.get_today_cost(), 6),
            "stage_breakdown": breakdown,
            "cost_distribution": {
                stage: f"{(s['cost_usd']/total*100):.1f}%" if total > 0 else "0%"
                for stage, s in breakdown.items()
            }
        }
    
    def export_log(self, filepath: str):
        """Export full cost log to JSON."""
        records = [
            {
                "timestamp": r.timestamp,
                "stage": r.stage.value,
                "model": r.model_used,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "success": r.success,
            }
            for r in self.records
        ]
        with open(filepath, "w") as f:
            json.dump({"records": records, "summary": self.get_summary()}, f, indent=2)
        logger.info(f"Cost log exported to {filepath}")


# ──────────────────────────────────────────────
# Prompt Cache Optimizer
# ──────────────────────────────────────────────

class PromptCacheOptimizer:
    """
    Optimizes prompt structure for maximum cache hit rates.
    
    RD-Agent's iterative nature makes it ideal for prompt caching:
    - System prompt (quantitative researcher role): ~2000 tokens, SAME every call
    - Factor library / model description: ~5000 tokens, mostly SAME
    - History (past iterations): grows incrementally, PREFIX is SAME
    - Current instruction: only this part is NEW (~500 tokens)
    
    DeepSeek V3.2 automatic prefix caching: 90% discount on cache hits ($0.028 vs $0.28/M)
    Key: Keep prompt prefix consistent across calls within same module.
    """
    
    @staticmethod
    def optimize_messages(messages: list[dict], stage: PipelineStage) -> list[dict]:
        """
        Reorder and structure messages to maximize prefix cache hits.
        
        Strategy:
        1. System message always first (most stable, always cached)
        2. Historical context second (grows but prefix stable)
        3. Current task-specific content last (only new part)
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        history_msgs = [m for m in messages if m.get("role") == "assistant"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        
        # Ensure system prompt is first and complete
        optimized = []
        
        # 1. System messages (highest cache value)
        optimized.extend(system_msgs)
        
        # 2. For multi-turn: interleave history (stable prefix)
        # Keep chronological order for history
        non_system = [m for m in messages if m.get("role") != "system"]
        
        # 3. Reconstruct with system first
        if system_msgs:
            return system_msgs + non_system
        return messages  # Don't modify if no system message


# ──────────────────────────────────────────────
# Main Router: The orchestrator
# ──────────────────────────────────────────────

class MultiModelRouter:
    """
    Intelligent multi-model router for RD-Agent pipeline optimization.
    
    This is the core contribution: instead of RD-Agent's single CHAT_MODEL
    for all modules, route each pipeline stage to the optimal model.
    
    Default routing strategy (based on paper analysis):
    
    ┌─────────────────┬──────────────────────┬─────────────────────────────┐
    │ Stage           │ Model                │ Rationale                   │
    ├─────────────────┼──────────────────────┼─────────────────────────────┤
    │ Synthesis       │ deepseek-reasoner    │ High-impact, low-freq,      │
    │                 │ (thinking mode)      │ needs creativity + domain   │
    │                 │                      │ knowledge                   │
    ├─────────────────┼──────────────────────┼─────────────────────────────┤
    │ Implementation  │ deepseek-chat        │ High-freq, self-correcting, │
    │                 │ (non-thinking)       │ structured code tasks       │
    ├─────────────────┼──────────────────────┼─────────────────────────────┤
    │ Analysis        │ deepseek-reasoner    │ High-impact, low-freq,      │
    │                 │ (thinking mode)      │ needs causal reasoning      │
    ├─────────────────┼──────────────────────┼─────────────────────────────┤
    │ Specification   │ deepseek-chat        │ Template assembly, minimal  │
    │ Validation      │ (no LLM needed)      │ Pure computation            │
    │ Embedding       │ BAAI/bge-m3 (free)   │ Knowledge base retrieval    │
    └─────────────────┴──────────────────────┴─────────────────────────────┘
    """
    
    # Default model configurations
    DEFAULT_MODELS = {
        "strong": ModelConfig(
            model_id="deepseek/deepseek-reasoner",
            input_price=0.28, cache_price=0.028, output_price=0.42,
            max_tokens=8192, temperature=1.0
        ),
        "efficient": ModelConfig(
            model_id="deepseek/deepseek-chat",
            input_price=0.28, cache_price=0.028, output_price=0.42,
            max_tokens=4096, temperature=0.8
        ),
        "premium": ModelConfig(
            model_id="anthropic/claude-sonnet-4-20250514",
            input_price=3.0, cache_price=0.30, output_price=15.0,
            max_tokens=4096, temperature=0.7
        ),
    }
    
    # Default routing table
    DEFAULT_ROUTING = {
        PipelineStage.SPECIFICATION: "efficient",
        PipelineStage.SYNTHESIS: "strong",
        PipelineStage.IMPLEMENTATION: "efficient",
        PipelineStage.ANALYSIS: "strong",
        PipelineStage.UNKNOWN: "efficient",
    }
    
    def __init__(
        self,
        routing_table: Optional[dict] = None,
        models: Optional[dict] = None,
        daily_budget: float = 5.0,
        enable_cache_optimization: bool = True,
        fallback_model: str = "efficient",
    ):
        self.models = models or self.DEFAULT_MODELS
        self.routing_table = routing_table or self.DEFAULT_ROUTING
        self.fallback_model = fallback_model
        self.enable_cache_optimization = enable_cache_optimization
        
        self.classifier = StageClassifier()
        self.cost_tracker = CostTracker(daily_budget_usd=daily_budget)
        self.cache_optimizer = PromptCacheOptimizer()
        
        logger.info(f"MultiModelRouter initialized with {len(self.models)} models")
        for stage, model_key in self.routing_table.items():
            model = self.models[model_key]
            logger.info(f"  {stage.value:20s} → {model.model_id}")
    
    def route(self, messages: list[dict], **kwargs) -> dict:
        """
        Route an LLM request to the optimal model and return the response.
        
        This method is designed to be a drop-in replacement for
        rdagent.oai.backend.LiteLLMAPIBackend.create_chat_completion()
        
        Args:
            messages: OpenAI-format chat messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            dict with keys: content, model, usage, cost, stage
        """
        start_time = time.time()
        
        # 1. Classify the request
        stage = self.classifier.classify(messages)
        
        # 2. Select model based on routing table
        model_key = self.routing_table.get(stage, self.fallback_model)
        
        # Budget check: if over budget, downgrade to efficient
        if self.cost_tracker.is_over_budget() and model_key != "efficient":
            logger.warning(f"Daily budget exceeded, downgrading {stage.value} to efficient model")
            model_key = "efficient"
        
        model_config = self.models[model_key]
        
        # 3. Optimize prompts for caching
        if self.enable_cache_optimization:
            messages = self.cache_optimizer.optimize_messages(messages, stage)
        
        # 4. Make the API call
        try:
            response = litellm.completion(
                model=model_config.model_id,
                messages=messages,
                temperature=kwargs.get("temperature", model_config.temperature),
                max_tokens=kwargs.get("max_tokens", model_config.max_tokens),
                stream=False,
            )
            
            # Extract usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # Estimate cost (assume 50% cache hit rate for DeepSeek after warmup)
            cache_ratio = 0.5 if model_config.supports_cache else 0.0
            input_cost = (
                input_tokens * cache_ratio * model_config.cache_price / 1_000_000
                + input_tokens * (1 - cache_ratio) * model_config.input_price / 1_000_000
            )
            output_cost = output_tokens * model_config.output_price / 1_000_000
            total_cost = input_cost + output_cost
            
            # Record the decision
            decision = RoutingDecision(
                timestamp=time.time(),
                stage=stage,
                model_used=model_config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=total_cost,
                latency_ms=(time.time() - start_time) * 1000,
                success=True,
            )
            self.cost_tracker.record(decision)
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "model": model_config.model_id,
                "stage": stage.value,
                "usage": {"input": input_tokens, "output": output_tokens},
                "cost_usd": total_cost,
            }
            
        except Exception as e:
            logger.error(f"API call failed for {stage.value}: {e}")
            
            # Record failure
            decision = RoutingDecision(
                timestamp=time.time(),
                stage=stage,
                model_used=model_config.model_id,
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
            self.cost_tracker.record(decision)
            
            # Fallback to efficient model if not already using it
            if model_key != "efficient":
                logger.info(f"Falling back to efficient model for {stage.value}")
                fallback = self.models["efficient"]
                try:
                    response = litellm.completion(
                        model=fallback.model_id,
                        messages=messages,
                        temperature=fallback.temperature,
                        max_tokens=fallback.max_tokens,
                    )
                    content = response.choices[0].message.content
                    return {
                        "content": content,
                        "model": fallback.model_id,
                        "stage": stage.value,
                        "usage": {},
                        "cost_usd": 0.0,
                        "fallback": True,
                    }
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
            
            raise


# ──────────────────────────────────────────────
# RD-Agent Integration Patch
# ──────────────────────────────────────────────

class PatchedLiteLLMBackend:
    """
    Drop-in replacement for rdagent.oai.backend.litellm.LiteLLMAPIBackend
    that adds multi-model routing.
    
    Usage:
        # In your .env or startup script:
        # BACKEND=src.model_router.PatchedLiteLLMBackend
        
        # Or monkey-patch at runtime:
        from src.model_router import PatchedLiteLLMBackend
        import rdagent.oai.backend.litellm as backend_module
        backend_module.LiteLLMAPIBackend = PatchedLiteLLMBackend
    """
    
    def __init__(self):
        self.router = MultiModelRouter(
            daily_budget=float(os.environ.get("DAILY_BUDGET_USD", "5.0")),
        )
        
        # Override routing if env vars specify custom models
        custom_routing = {}
        if os.environ.get("SYNTHESIS_MODEL"):
            self.router.models["synthesis_custom"] = ModelConfig(
                model_id=os.environ["SYNTHESIS_MODEL"],
                input_price=0.28, cache_price=0.028, output_price=0.42
            )
            custom_routing[PipelineStage.SYNTHESIS] = "synthesis_custom"
        
        if os.environ.get("IMPLEMENTATION_MODEL"):
            self.router.models["impl_custom"] = ModelConfig(
                model_id=os.environ["IMPLEMENTATION_MODEL"],
                input_price=0.28, cache_price=0.028, output_price=0.42
            )
            custom_routing[PipelineStage.IMPLEMENTATION] = "impl_custom"
        
        if os.environ.get("ANALYSIS_MODEL"):
            self.router.models["analysis_custom"] = ModelConfig(
                model_id=os.environ["ANALYSIS_MODEL"],
                input_price=0.28, cache_price=0.028, output_price=0.42
            )
            custom_routing[PipelineStage.ANALYSIS] = "analysis_custom"
        
        self.router.routing_table.update(custom_routing)
    
    def create_chat_completion(self, messages, **kwargs):
        """Drop-in replacement for RD-Agent's LLM call method."""
        result = self.router.route(messages, **kwargs)
        return result["content"]
    
    def create_embedding(self, text, **kwargs):
        """Embedding calls routed to free/cheap models."""
        model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        # Try SiliconFlow free tier first
        try:
            response = litellm.embedding(
                model=model,
                input=[text] if isinstance(text, str) else text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed with {model}: {e}, trying ada-002")
            response = litellm.embedding(
                model="text-embedding-ada-002",
                input=[text] if isinstance(text, str) else text,
            )
            return response.data[0].embedding
    
    def get_cost_summary(self) -> dict:
        """Get cost tracking summary."""
        return self.router.cost_tracker.get_summary()
    
    def export_cost_log(self, filepath: str = "cost_log.json"):
        """Export detailed cost log."""
        self.router.cost_tracker.export_log(filepath)
