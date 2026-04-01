"""
Bull-Bear Adversarial Debate for RD-Agent Analysis Unit.

Replaces single-LLM analysis with two adversarial agents to reduce
confirmation bias in iterative R&D loops. Adds only 2 extra LLM calls
per iteration.

Based on TradingAgents (arXiv:2412.20138) + AlphaAgents (arXiv:2508.11152).
"""

import logging
from dataclasses import dataclass
from enum import Enum

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


BULL_SYSTEM = (
    "You are a Bull Analyst in a quantitative R&D team. Your role is to argue "
    "that the current factor/strategy direction has potential and should be refined "
    "further. Look for positive signals, suggest improvements, and advocate for "
    "continuation. Be specific about what refinements could unlock better performance."
)

BEAR_SYSTEM = (
    "You are a Bear Analyst in a quantitative R&D team. Your role is to critically "
    "challenge the current factor/strategy direction. Identify fundamental flaws, "
    "overfitting risks, economic implausibility, and data-snooping concerns. "
    "Argue for pivoting to a completely new hypothesis if the current one is weak."
)

JUDGE_SYSTEM = (
    "You are the Chief Strategist synthesizing Bull and Bear arguments about a "
    "quantitative factor. Decide: CONTINUE (refine current direction), PIVOT "
    "(abandon for new hypothesis), or NEUTRAL. Respond in this exact format:\n"
    "VERDICT: <CONTINUE|PIVOT|NEUTRAL>\n"
    "CONFIDENCE: <0.0-1.0>\n"
    "ACTION: <concrete next step in one sentence>\n"
    "REASONING: <2-3 sentences>"
)


class DebateAnalyzer:
    """Adversarial debate between Bull and Bear agents for strategy evaluation."""

    def __init__(
        self,
        debate_model: str = "deepseek/deepseek-reasoner",
        judge_model: str = "deepseek/deepseek-chat",
        max_tokens: int = 1024,
    ):
        self.debate_model = debate_model
        self.judge_model = judge_model
        self.max_tokens = max_tokens

    def _call_llm(self, system: str, user: str, model: str) -> str:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
            timeout=120,  # 影响: 防止 API 卡死挂起整个流水线
        )
        return response.choices[0].message.content.strip()

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

        # Bull and Bear argue (could be parallelized in production)
        bull_arg = self._call_llm(BULL_SYSTEM, context, self.debate_model)
        bear_arg = self._call_llm(BEAR_SYSTEM, context, self.debate_model)

        # Judge synthesizes
        judge_prompt = (
            f"{context}\n\n"
            f"=== Bull Argument ===\n{bull_arg}\n\n"
            f"=== Bear Argument ===\n{bear_arg}"
        )
        synthesis = self._call_llm(JUDGE_SYSTEM, judge_prompt, self.judge_model)
        verdict, confidence, action = self._parse_verdict(synthesis)

        logger.info(
            f"[Debate] iter={iteration} verdict={verdict.value} "
            f"confidence={confidence:.2f} action={action}"
        )

        return DebateResult(
            verdict=verdict,
            confidence=confidence,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            synthesis=synthesis,
            recommended_action=action,
        )
