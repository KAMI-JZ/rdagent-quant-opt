"""
Bull-Bear Adversarial Debate for RD-Agent Analysis Unit.

Replaces single-LLM analysis with two adversarial agents to reduce
confirmation bias in iterative R&D loops.

V2 升级 (2026-04-02):
- 接入 MultiModelRouter: 根据预算模式自动选最优辩论模型
- 改进 prompt: 强制引用量化证据（IC/Sharpe/回撤等）
- 新增 debate_with_history(): 注入历史辩论记录，避免重复论点

Based on TradingAgents (arXiv:2412.20138) + AlphaAgents (arXiv:2508.11152).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


class Verdict(Enum):
    CONTINUE = "continue"    # Current direction has potential
    PIVOT = "pivot"          # Should explore new hypothesis
    NEUTRAL = "neutral"      # No strong signal either way


@dataclass
class DebateResult:
    verdict: Verdict
    confidence: float           # 0-1, strength of consensus
    bull_argument: str
    bear_argument: str
    synthesis: str              # Final merged recommendation
    recommended_action: str     # Concrete next step
    model_used: str = ""        # 影响: 记录实际使用的模型，方便成本追踪


# ──────────────────────────────────────────────
# Prompt 模板 (V2 — 强制量化证据引用)
# 影响: 要求 Bull/Bear 必须引用具体数值，减少空泛论证
# ──────────────────────────────────────────────

BULL_SYSTEM = (
    "You are a Bull Analyst in a quantitative R&D team. Your role is to argue "
    "that the current factor/strategy direction has potential and should be refined "
    "further.\n\n"
    "RULES:\n"
    "1. You MUST cite specific metric values from the data (e.g., 'IC=0.035 is above "
    "the 0.02 threshold', 'Sharpe=1.8 indicates strong risk-adjusted return').\n"
    "2. Identify what is working and suggest concrete refinements.\n"
    "3. Compare against typical benchmarks: IC>0.02 is decent, Sharpe>1.0 is good.\n"
    "4. Be specific about what parameter changes or logic tweaks could improve results."
)

BEAR_SYSTEM = (
    "You are a Bear Analyst in a quantitative R&D team. Your role is to critically "
    "challenge the current factor/strategy direction.\n\n"
    "RULES:\n"
    "1. You MUST cite specific metric values to support your critique (e.g., "
    "'MDD=-35% exceeds the -20% risk tolerance', 'IC=0.01 is below noise level').\n"
    "2. Check for: overfitting, economic implausibility, data-snooping, survivorship bias.\n"
    "3. Compare against failure patterns: IC<0.02 is weak, Sharpe<0 means losing money.\n"
    "4. If fundamentally flawed, argue for pivoting to a completely new hypothesis."
)

JUDGE_SYSTEM = (
    "You are the Chief Strategist synthesizing Bull and Bear arguments about a "
    "quantitative factor. You must weigh the quantitative evidence cited by both sides.\n\n"
    "Decide: CONTINUE (refine current direction), PIVOT "
    "(abandon for new hypothesis), or NEUTRAL. Respond in this exact format:\n"
    "VERDICT: <CONTINUE|PIVOT|NEUTRAL>\n"
    "CONFIDENCE: <0.0-1.0>\n"
    "ACTION: <concrete next step in one sentence>\n"
    "REASONING: <2-3 sentences citing the specific metrics that drove your decision>"
)


class DebateAnalyzer:
    """
    Adversarial debate between Bull and Bear agents for strategy evaluation.

    V2: 支持通过 MultiModelRouter 自动选模型。
    - 传入 router → 根据预算模式选最优模型 (premium=Opus, budget=DeepSeek Chat)
    - 不传 router → 保持原有行为，直接调 litellm (向后兼容)
    """

    def __init__(
        self,
        debate_model: str = "deepseek/deepseek-reasoner",
        judge_model: str = "deepseek/deepseek-chat",
        max_tokens: int = 1024,
        router=None,  # 影响: 传入 MultiModelRouter 实例即可接入先进 LLM
    ):
        self.debate_model = debate_model
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self.router = router
        if router:
            logger.info("[Debate] Router connected — model selection will be automatic")

    def _call_llm(self, system: str, user: str, model: str) -> tuple[str, str]:
        """
        调用 LLM，返回 (response_text, model_used)。
        影响: 有 router 时走 router.route()，自动选模型 + 记录成本；
              无 router 时直接调 litellm，保持向后兼容。
        """
        if self.router:
            # 功能: 通过 router 路由，利用 ANALYSIS stage 的模型选择逻辑
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            result = self.router.route(messages, max_tokens=self.max_tokens)
            return result["content"], result["model"]
        else:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=0.7,
                timeout=120,
            )
            return response.choices[0].message.content.strip(), model

    def _build_context(
        self, hypothesis: str, code: str, metrics: dict, iteration: int
    ) -> str:
        metrics_str = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
        return (
            f"Iteration: {iteration}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Factor Code:\n```python\n{code}\n```\n"
            f"Backtest Metrics:\n{metrics_str}"
        )

    def _build_history_context(self, history: list[DebateResult]) -> str:
        """
        将历史辩论结果构建为 prompt 上下文。
        影响: 让 Bull/Bear 知道之前讨论过什么，避免重复论点。
        """
        if not history:
            return ""
        lines = ["\n[DEBATE HISTORY — Previous rounds for reference, avoid repeating arguments]"]
        for i, h in enumerate(history[-3:], 1):  # 最多引用最近3轮
            lines.append(
                f"  Round {i}: verdict={h.verdict.value}, confidence={h.confidence:.2f}"
                f" — {h.recommended_action}"
            )
        return "\n".join(lines)

    def _parse_verdict(self, judge_text: str) -> tuple[Verdict, float, str]:
        verdict = Verdict.NEUTRAL
        confidence = 0.5
        action = "Review and reassess"

        for line in judge_text.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = {"CONTINUE": Verdict.CONTINUE, "PIVOT": Verdict.PIVOT}.get(
                    v, Verdict.NEUTRAL
                )
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = min(max(float(line.split(":", 1)[1].strip()), 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip()

        return verdict, confidence, action

    def debate(
        self,
        hypothesis: str,
        code: str,
        metrics: dict,
        iteration: int = 0,
    ) -> DebateResult:
        """Run Bull-Bear debate on current factor. Returns structured decision."""
        context = self._build_context(hypothesis, code, metrics, iteration)

        bull_arg, bull_model = self._call_llm(BULL_SYSTEM, context, self.debate_model)
        bear_arg, bear_model = self._call_llm(BEAR_SYSTEM, context, self.debate_model)

        judge_prompt = (
            f"{context}\n\n"
            f"=== Bull Argument ===\n{bull_arg}\n\n"
            f"=== Bear Argument ===\n{bear_arg}"
        )
        synthesis, judge_model_used = self._call_llm(JUDGE_SYSTEM, judge_prompt, self.judge_model)
        verdict, confidence, action = self._parse_verdict(synthesis)

        logger.info(
            f"[Debate] iter={iteration} verdict={verdict.value} "
            f"confidence={confidence:.2f} action={action} model={bull_model}"
        )

        return DebateResult(
            verdict=verdict,
            confidence=confidence,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            synthesis=synthesis,
            recommended_action=action,
            model_used=bull_model,
        )

    def debate_with_history(
        self,
        hypothesis: str,
        code: str,
        metrics: dict,
        iteration: int = 0,
        history: Optional[list[DebateResult]] = None,
    ) -> DebateResult:
        """
        带历史上下文的辩论。注入过往辩论记录，避免重复论点。
        影响: 多轮迭代时，辩论质量随迭代积累而提升。
        """
        context = self._build_context(hypothesis, code, metrics, iteration)
        history_context = self._build_history_context(history or [])
        full_context = context + history_context

        bull_arg, bull_model = self._call_llm(BULL_SYSTEM, full_context, self.debate_model)
        bear_arg, bear_model = self._call_llm(BEAR_SYSTEM, full_context, self.debate_model)

        judge_prompt = (
            f"{full_context}\n\n"
            f"=== Bull Argument ===\n{bull_arg}\n\n"
            f"=== Bear Argument ===\n{bear_arg}"
        )
        synthesis, judge_model_used = self._call_llm(JUDGE_SYSTEM, judge_prompt, self.judge_model)
        verdict, confidence, action = self._parse_verdict(synthesis)

        logger.info(
            f"[Debate] iter={iteration} verdict={verdict.value} "
            f"confidence={confidence:.2f} action={action} model={bull_model} (with history)"
        )

        return DebateResult(
            verdict=verdict,
            confidence=confidence,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            synthesis=synthesis,
            recommended_action=action,
            model_used=bull_model,
        )
