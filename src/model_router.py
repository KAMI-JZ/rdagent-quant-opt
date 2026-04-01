"""
Multi-Model Router for RD-Agent.

Routes pipeline stages to optimal LLM backends: strong models for high-impact
low-frequency calls (Synthesis/Analysis), efficient models for high-frequency
self-correcting calls (Implementation/Co-STEER).

Includes AdaptiveModelSelector: auto-selects the best model per task based on
task characteristics (creativity, reasoning depth, volume) and cost constraints.

Based on arXiv:2505.15155v2 architecture analysis.
"""

import os
import re
import json
import time
import logging
from datetime import date
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    SPECIFICATION = "specification"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    model_id: str
    input_price: float      # $/1M tokens (cache miss)
    cache_price: float      # $/1M tokens (cache hit)
    output_price: float     # $/1M tokens
    max_tokens: int = 4096
    temperature: float = 0.8
    supports_cache: bool = True
    tier: str = "efficient"  # 能力层级: "frontier" / "strong" / "efficient"


# ──────────────────────────────────────────────
# 模型注册表 (Model Registry)
# 功能: 集中管理所有可用模型及其能力标签和价格
# 影响: 新增模型只需在这里加一行，不用改其他代码
# ──────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # === Frontier 层: 最强推理/创造力，用于因子构思、假设生成 ===
    "claude-opus-4-6": ModelConfig(
        model_id="anthropic/claude-opus-4-6",
        input_price=15.0, cache_price=1.50, output_price=75.0,
        max_tokens=8192, temperature=0.7, tier="frontier",
    ),
    # === Strong 层: 高质量推理，性价比优于 frontier ===
    "claude-sonnet-4-6": ModelConfig(
        model_id="anthropic/claude-sonnet-4-6",
        input_price=3.0, cache_price=0.30, output_price=15.0,
        max_tokens=8192, temperature=0.7, tier="strong",
    ),
    "deepseek-reasoner": ModelConfig(
        model_id="deepseek/deepseek-reasoner",
        input_price=0.55, cache_price=0.14, output_price=2.19,
        max_tokens=8192, temperature=1.0, tier="strong",
    ),
    # === Efficient 层: 高吞吐低成本，用于代码生成/debug/批量处理 ===
    # DeepSeek V3.2 (Mar 2026): 统一 chat/reasoner 端点，缓存命中价极低
    "deepseek-chat": ModelConfig(
        model_id="deepseek/deepseek-chat",
        input_price=0.28, cache_price=0.028, output_price=0.42,
        max_tokens=4096, temperature=0.8, tier="efficient",
    ),
    # DeepSeek V4 (Mar 2026): 最新旗舰，能力接近 strong 但定价 efficient
    "deepseek-v4": ModelConfig(
        model_id="deepseek/deepseek-v4",
        input_price=0.30, cache_price=0.030, output_price=0.50,
        max_tokens=8192, temperature=0.8, tier="efficient",
    ),
}


@dataclass
class RoutingDecision:
    timestamp: float
    stage: PipelineStage
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class StageClassifier:
    """Classifies LLM requests into pipeline stages via prompt pattern matching."""

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
        full_text = " ".join(
            msg.get("content", "") for msg in messages
            if isinstance(msg.get("content"), str)
        ).lower()

        scores = {
            stage: sum(1 for p in patterns if re.search(p, full_text, re.IGNORECASE))
            for stage, patterns in self.STAGE_PATTERNS.items()
        }
        scores = {s: v for s, v in scores.items() if v > 0}
        return max(scores, key=scores.get) if scores else PipelineStage.UNKNOWN


class CostTracker:
    """Tracks LLM usage costs per pipeline stage with daily budget control."""

    def __init__(self, daily_budget_usd: float = 5.0):
        self.daily_budget = daily_budget_usd
        self.records: list[RoutingDecision] = []
        self._current_date = date.today()  # 影响: 用日历日期代替时间戳，跨天自动重置

    def _maybe_reset_day(self):
        """跨天自动重置预算计数。影响: 多天运行时每天从零开始计费。"""
        today = date.today()
        if today != self._current_date:
            logger.info(f"[Cost] New day detected ({today}), resetting daily budget")
            self._current_date = today

    def record(self, decision: RoutingDecision):
        self._maybe_reset_day()
        self.records.append(decision)
        logger.info(
            f"[Cost] {decision.stage.value}: {decision.model_used} "
            f"{decision.input_tokens}+{decision.output_tokens}tok ${decision.cost_usd:.6f}"
        )

    def get_today_cost(self) -> float:
        self._maybe_reset_day()
        # 影响: 只计算今天的成本，跨天的记录不算
        today_start = time.mktime(self._current_date.timetuple())
        return sum(r.cost_usd for r in self.records if r.timestamp >= today_start)

    def is_over_budget(self) -> bool:
        return self.get_today_cost() >= self.daily_budget

    def get_stage_breakdown(self) -> dict:
        breakdown: dict = {}
        for r in self.records:
            s = breakdown.setdefault(r.stage.value, {
                "calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0
            })
            s["calls"] += 1
            s["cost_usd"] += r.cost_usd
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
        return breakdown

    def get_summary(self) -> dict:
        breakdown = self.get_stage_breakdown()
        total = sum(s["cost_usd"] for s in breakdown.values())
        return {
            "total_cost_usd": round(total, 6),
            "total_calls": sum(s["calls"] for s in breakdown.values()),
            "daily_budget_usd": self.daily_budget,
            "budget_remaining_usd": round(self.daily_budget - self.get_today_cost(), 6),
            "stage_breakdown": breakdown,
            "cost_distribution": {
                stage: f"{(s['cost_usd']/total*100):.1f}%" if total > 0 else "0%"
                for stage, s in breakdown.items()
            },
        }

    def export_log(self, filepath: str):
        records = [
            {"timestamp": r.timestamp, "stage": r.stage.value, "model": r.model_used,
             "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
             "cost_usd": r.cost_usd, "latency_ms": r.latency_ms, "success": r.success}
            for r in self.records
        ]
        with open(filepath, "w") as f:
            json.dump({"records": records, "summary": self.get_summary()}, f, indent=2)
        logger.info(f"Cost log exported to {filepath}")


class PromptCacheOptimizer:
    """Reorders messages to maximize DeepSeek prefix cache hits (90% discount)."""

    @staticmethod
    def optimize_messages(messages: list[dict], stage: PipelineStage) -> list[dict]:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if not system_msgs:
            return messages
        non_system = [m for m in messages if m.get("role") != "system"]
        return system_msgs + non_system


# ──────────────────────────────────────────────
# 自适应模型选择器 (AdaptiveModelSelector)
# 功能: 根据任务特征自动选择最优模型
# 影响: 不改变现有路由逻辑，作为可选增强层
# ──────────────────────────────────────────────

# 每个 pipeline stage 的任务特征评分
# creativity: 需要创造力/新颖性 (0-1)
# reasoning: 需要深度推理/因果分析 (0-1)
# volume: 调用频次，越高越需要便宜模型 (0-1)
# retry_tolerance: 有重试机制，容错能力 (0-1)
STAGE_TRAITS: dict[PipelineStage, dict[str, float]] = {
    PipelineStage.SYNTHESIS: {
        "creativity": 0.9,       # 因子构思需要最强创造力
        "reasoning": 0.8,        # 需要经济学推理
        "volume": 0.1,           # 每轮只调1-2次
        "retry_tolerance": 0.0,  # 无重试，输出决定方向
    },
    PipelineStage.ANALYSIS: {
        "creativity": 0.3,       # 分析不需要太多创造力
        "reasoning": 0.9,        # 需要因果推理和诊断
        "volume": 0.1,           # 每轮只调1-2次
        "retry_tolerance": 0.0,  # 无重试
    },
    PipelineStage.IMPLEMENTATION: {
        "creativity": 0.2,       # 代码生成是结构化任务
        "reasoning": 0.4,        # 中等推理
        "volume": 0.9,           # 每轮10-15次，最大开支
        "retry_tolerance": 0.9,  # Co-STEER 最多重试10次
    },
    PipelineStage.SPECIFICATION: {
        "creativity": 0.1,       # 模板组装
        "reasoning": 0.1,
        "volume": 0.05,
        "retry_tolerance": 0.5,
    },
    PipelineStage.UNKNOWN: {
        "creativity": 0.3,
        "reasoning": 0.3,
        "volume": 0.5,
        "retry_tolerance": 0.5,
    },
}


class AdaptiveModelSelector:
    """
    根据任务特征和预算自动选择最优模型。

    决策逻辑:
    - 高创造力 + 低调用量 → frontier (如 Opus 4.6)
    - 高推理 + 低调用量  → strong (如 Reasoner / Sonnet 4.6)
    - 高调用量 + 有重试   → efficient (如 DeepSeek Chat)

    边际收益原则: 当升级模型带来的质量提升不足以抵消成本增加时，
    保持使用高性价比模型。
    """

    # 功能: 根据 tier 需求匹配注册表中的模型
    # 影响: 注册表更新后自动生效，不用改选择逻辑
    TIER_PRIORITY = {
        "frontier": ["claude-opus-4-6", "claude-sonnet-4-6", "deepseek-reasoner"],
        "strong": ["deepseek-reasoner", "claude-sonnet-4-6", "deepseek-v4"],
        "efficient": ["deepseek-chat", "deepseek-v4", "deepseek-reasoner"],
    }

    def __init__(
        self,
        registry: dict[str, ModelConfig] | None = None,
        budget_mode: str = "optimized",  # "budget" | "optimized" | "premium"
    ):
        self.registry = registry or MODEL_REGISTRY
        self.budget_mode = budget_mode

    def select(self, stage: PipelineStage) -> ModelConfig:
        """根据 stage 特征 + 预算模式，自动选最优模型。"""
        traits = STAGE_TRAITS.get(stage, STAGE_TRAITS[PipelineStage.UNKNOWN])
        needed_tier = self._decide_tier(traits)

        # 预算模式覆盖: budget 模式强制降级
        if self.budget_mode == "budget":
            needed_tier = "efficient"
        elif self.budget_mode == "optimized" and needed_tier == "frontier":
            needed_tier = "strong"  # optimized 模式最高用 strong

        return self._pick_from_tier(needed_tier)

    def _decide_tier(self, traits: dict[str, float]) -> str:
        """核心决策: 任务特征 → 需要什么层级的模型。"""
        # 高调用量 + 有重试 → 用便宜模型（边际收益低）
        if traits["volume"] > 0.7 and traits["retry_tolerance"] > 0.5:
            return "efficient"
        # 需要强创造力 + 低调用量 → 用最强模型
        if traits["creativity"] > 0.7 and traits["volume"] < 0.3:
            return "frontier"
        # 需要深度推理 + 低调用量 → 用 strong 模型
        if traits["reasoning"] > 0.7 and traits["volume"] < 0.3:
            return "strong"
        return "efficient"

    def _pick_from_tier(self, tier: str) -> ModelConfig:
        """从注册表中按优先级挑选可用模型。"""
        for model_name in self.TIER_PRIORITY.get(tier, ["deepseek-chat"]):
            if model_name in self.registry:
                return self.registry[model_name]
        # 最终兜底
        return self.registry.get("deepseek-chat", list(self.registry.values())[-1])

    def get_routing_table(self) -> dict[PipelineStage, str]:
        """生成完整路由表，供 MultiModelRouter 使用。"""
        table = {}
        for stage in [PipelineStage.SPECIFICATION, PipelineStage.SYNTHESIS,
                      PipelineStage.IMPLEMENTATION, PipelineStage.ANALYSIS,
                      PipelineStage.UNKNOWN]:
            cfg = self.select(stage)
            # 在 models dict 中找到对应 key
            table[stage] = cfg.model_id
        return table


class MultiModelRouter:
    """Routes each RD-Agent pipeline stage to the optimal model tier."""

    # 功能: 默认模型配置，用注册表中的最新模型
    # 影响: premium 从过时的 Sonnet 4 升级到 Opus 4.6
    DEFAULT_MODELS = {
        "frontier": MODEL_REGISTRY["claude-opus-4-6"],
        "strong": MODEL_REGISTRY["deepseek-reasoner"],
        "efficient": MODEL_REGISTRY["deepseek-chat"],
        "premium": MODEL_REGISTRY["claude-sonnet-4-6"],
    }
    
    DEFAULT_ROUTING = {
        PipelineStage.SPECIFICATION: "efficient",
        PipelineStage.SYNTHESIS: "frontier",   # 因子构思 → 最强模型
        PipelineStage.IMPLEMENTATION: "efficient",  # 代码生成 → 高性价比
        PipelineStage.ANALYSIS: "strong",      # 结果分析 → 推理型
        PipelineStage.UNKNOWN: "efficient",
    }

    def __init__(
        self,
        routing_table: Optional[dict] = None,
        models: Optional[dict] = None,
        daily_budget: float = 5.0,
        enable_cache_optimization: bool = True,
        fallback_model: str = "efficient",
        adaptive_mode: Optional[str] = None,   # "budget" | "optimized" | "premium"
    ):
        self.models = models or dict(self.DEFAULT_MODELS)
        self.fallback_model = fallback_model
        self.enable_cache_optimization = enable_cache_optimization

        # 功能: adaptive_mode 启用自适应选择器，自动生成路由表
        # 影响: 覆盖手动路由表，根据任务特征 + 预算模式选模型
        if adaptive_mode:
            selector = AdaptiveModelSelector(
                registry=MODEL_REGISTRY, budget_mode=adaptive_mode
            )
            # 用选择器结果构建路由表: stage → model_key in self.models
            self.routing_table = {}
            for stage in [PipelineStage.SPECIFICATION, PipelineStage.SYNTHESIS,
                          PipelineStage.IMPLEMENTATION, PipelineStage.ANALYSIS,
                          PipelineStage.UNKNOWN]:
                cfg = selector.select(stage)
                # 找到或创建 model key
                key = self._ensure_model_key(cfg)
                self.routing_table[stage] = key
            logger.info(f"AdaptiveModelSelector active (mode={adaptive_mode})")
        else:
            self.routing_table = routing_table or dict(self.DEFAULT_ROUTING)

        self.classifier = StageClassifier()
        self.cost_tracker = CostTracker(daily_budget_usd=daily_budget)
        self.cache_optimizer = PromptCacheOptimizer()

        logger.info(f"MultiModelRouter initialized with {len(self.models)} models")
        for stage, model_key in self.routing_table.items():
            model = self.models[model_key]
            logger.info(f"  {stage.value:20s} → {model.model_id}")

    def _ensure_model_key(self, cfg: ModelConfig) -> str:
        """确保 ModelConfig 在 self.models 中有对应的 key。"""
        for key, existing in self.models.items():
            if existing.model_id == cfg.model_id:
                return key
        # 不在 models 中，用 tier 名注册
        key = f"auto_{cfg.tier}"
        self.models[key] = cfg
        return key
    
    def _estimate_cost(self, cfg: ModelConfig, input_tok: int, output_tok: int) -> float:
        cache_ratio = 0.5 if cfg.supports_cache else 0.0
        input_cost = input_tok * (
            cache_ratio * cfg.cache_price + (1 - cache_ratio) * cfg.input_price
        ) / 1_000_000
        return input_cost + output_tok * cfg.output_price / 1_000_000

    # 影响: 默认超时 120 秒，防止 API 卡死挂起整个流水线
    DEFAULT_TIMEOUT = 120

    def _call_model(self, cfg: ModelConfig, messages: list[dict], **kwargs):
        return litellm.completion(
            model=cfg.model_id, messages=messages,
            temperature=kwargs.get("temperature", cfg.temperature),
            max_tokens=kwargs.get("max_tokens", cfg.max_tokens), stream=False,
            timeout=kwargs.get("timeout", self.DEFAULT_TIMEOUT),
        )

    def route(self, messages: list[dict], **kwargs) -> dict:
        """Route an LLM request to the optimal model. Drop-in for RD-Agent backend."""
        start_time = time.time()
        stage = self.classifier.classify(messages)
        model_key = self.routing_table.get(stage, self.fallback_model)

        if self.cost_tracker.is_over_budget() and model_key != "efficient":
            logger.warning(f"Budget exceeded, downgrading {stage.value} to efficient")
            model_key = "efficient"

        cfg = self.models[model_key]
        if self.enable_cache_optimization:
            messages = self.cache_optimizer.optimize_messages(messages, stage)

        try:
            response = self._call_model(cfg, messages, **kwargs)
        except Exception as e:
            logger.error(f"API call failed for {stage.value}: {e}")
            self.cost_tracker.record(RoutingDecision(
                timestamp=time.time(), stage=stage, model_used=cfg.model_id,
                latency_ms=(time.time() - start_time) * 1000, success=False, error=str(e),
            ))
            if model_key != "efficient":
                logger.info(f"Falling back to efficient model for {stage.value}")
                cfg = self.models["efficient"]
                response = self._call_model(cfg, messages, **kwargs)
            else:
                raise

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost = self._estimate_cost(cfg, in_tok, out_tok)

        self.cost_tracker.record(RoutingDecision(
            timestamp=time.time(), stage=stage, model_used=cfg.model_id,
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
            latency_ms=(time.time() - start_time) * 1000,
        ))
        return {
            "content": response.choices[0].message.content,
            "model": cfg.model_id, "stage": stage.value,
            "usage": {"input": in_tok, "output": out_tok}, "cost_usd": cost,
        }


class PatchedLiteLLMBackend:
    """Drop-in replacement for rdagent.oai.backend.litellm.LiteLLMAPIBackend."""

    _ENV_MODEL_MAP = {
        "SYNTHESIS_MODEL": PipelineStage.SYNTHESIS,
        "IMPLEMENTATION_MODEL": PipelineStage.IMPLEMENTATION,
        "ANALYSIS_MODEL": PipelineStage.ANALYSIS,
    }

    def __init__(self):
        self.router = MultiModelRouter(
            daily_budget=float(os.environ.get("DAILY_BUDGET_USD", "5.0")),
        )
        for env_key, stage in self._ENV_MODEL_MAP.items():
            model_id = os.environ.get(env_key)
            if model_id:
                key = f"{stage.value}_custom"
                self.router.models[key] = ModelConfig(
                    model_id=model_id, input_price=0.28, cache_price=0.028, output_price=0.42,
                )
                self.router.routing_table[stage] = key

    def create_chat_completion(self, messages, **kwargs):
        return self.router.route(messages, **kwargs)["content"]

    def create_embedding(self, text, **kwargs):
        text_input = [text] if isinstance(text, str) else text
        model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        try:
            return litellm.embedding(model=model, input=text_input).data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding {model} failed: {e}, falling back to ada-002")
            return litellm.embedding(model="text-embedding-ada-002", input=text_input).data[0].embedding

    def get_cost_summary(self) -> dict:
        return self.router.cost_tracker.get_summary()

    def export_cost_log(self, filepath: str = "cost_log.json"):
        self.router.cost_tracker.export_log(filepath)
