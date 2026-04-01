"""
Market Regime-Aware RAG for RD-Agent Synthesis Unit.

Classifies current market state and injects regime context into hypothesis
generation, changing factor search from blind to directed exploration.

Regime dimensions: {bull/bear} × {high/low volatility} × {trend/mean-revert}

Based on AlphaAgent/LLMQuant MarketPulse + Financial RAG literature.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import litellm

logger = logging.getLogger(__name__)


class MarketDirection(Enum):
    BULL = "bull"
    BEAR = "bear"


class Volatility(Enum):
    HIGH = "high"
    LOW = "low"


class MarketStyle(Enum):
    TREND = "trend"
    MEAN_REVERT = "mean_revert"


@dataclass
class MarketRegime:
    direction: MarketDirection
    volatility: Volatility
    style: MarketStyle
    confidence: float = 0.0
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.direction.value}_{self.volatility.value}_{self.style.value}"

    def to_prompt_context(self) -> str:
        return (
            f"Current Market Regime: {self.label} (confidence={self.confidence:.2f})\n"
            f"- Direction: {self.direction.value} market\n"
            f"- Volatility: {self.volatility.value}\n"
            f"- Style: {self.style.value}\n"
            f"- Description: {self.description}\n\n"
            "Tailor factor hypotheses to this regime. "
            f"{'Momentum and trend-following factors tend to work well. ' if self.style == MarketStyle.TREND else 'Mean-reversion and contrarian factors tend to work well. '}"
            f"{'Consider defensive/low-vol factors. ' if self.direction == MarketDirection.BEAR else ''}"
            f"{'High-vol regime favors factors with stronger signals. ' if self.volatility == Volatility.HIGH else ''}"
        )


# Regime-specific factor guidance for RAG injection
REGIME_FACTOR_HINTS = {
    "bull_low_trend": [
        "Momentum factors (5/10/20-day returns)",
        "Volume breakout signals",
        "Earnings surprise continuation",
    ],
    "bull_high_trend": [
        "Strong momentum with volatility filter",
        "Sector rotation signals",
        "High-beta exposure factors",
    ],
    "bear_high_mean_revert": [
        "Mean-reversion on oversold RSI",
        "Quality factors (ROE, debt/equity)",
        "Low-volatility anomaly",
    ],
    "bear_low_mean_revert": [
        "Defensive value factors",
        "Dividend yield signals",
        "Low-turnover stability factors",
    ],
}


REGIME_DETECT_PROMPT = (
    "You are a market regime classifier. Given the market data summary below, "
    "classify the current regime. Respond in exactly this format:\n"
    "DIRECTION: <bull|bear>\n"
    "VOLATILITY: <high|low>\n"
    "STYLE: <trend|mean_revert>\n"
    "CONFIDENCE: <0.0-1.0>\n"
    "DESCRIPTION: <one sentence summary>"
)


class MarketRegimeDetector:
    """Detects market regime from data summaries or indicators."""

    def __init__(self, model: str = "deepseek/deepseek-chat"):
        self.model = model

    def detect_from_indicators(
        self,
        returns_20d: float,
        volatility_20d: float,
        ma_cross: float,
        vol_percentile: float = 0.5,
    ) -> MarketRegime:
        """Rule-based detection from numeric indicators (no LLM needed)."""
        direction = MarketDirection.BULL if returns_20d > 0 else MarketDirection.BEAR
        volatility = Volatility.HIGH if vol_percentile > 0.7 else Volatility.LOW

        # Trend vs mean-revert: check if price follows or reverts from MA
        if abs(ma_cross) > 0.02:
            style = MarketStyle.TREND
        else:
            style = MarketStyle.MEAN_REVERT

        confidence = min(abs(returns_20d) * 10, 1.0) * 0.5 + 0.5

        return MarketRegime(
            direction=direction,
            volatility=volatility,
            style=style,
            confidence=round(confidence, 2),
            description=(
                f"{'Uptrend' if direction == MarketDirection.BULL else 'Downtrend'} "
                f"with {'elevated' if volatility == Volatility.HIGH else 'subdued'} volatility, "
                f"{'trending' if style == MarketStyle.TREND else 'range-bound'} price action"
            ),
        )

    def detect_from_text(self, market_summary: str) -> MarketRegime:
        """LLM-based detection from text market summary."""
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": REGIME_DETECT_PROMPT},
                    {"role": "user", "content": market_summary},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_response(text)
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime(
                MarketDirection.BULL, Volatility.LOW, MarketStyle.TREND,
                confidence=0.0, description="Detection failed, using default",
            )

    def _parse_response(self, text: str) -> MarketRegime:
        fields: dict = {}
        for line in text.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                fields[key.strip().upper()] = val.strip().lower()

        direction = (
            MarketDirection.BEAR if fields.get("DIRECTION") == "bear"
            else MarketDirection.BULL
        )
        volatility = (
            Volatility.HIGH if fields.get("VOLATILITY") == "high"
            else Volatility.LOW
        )
        style = (
            MarketStyle.MEAN_REVERT if fields.get("STYLE") == "mean_revert"
            else MarketStyle.TREND
        )
        try:
            confidence = min(max(float(fields.get("CONFIDENCE", "0.5")), 0.0), 1.0)
        except ValueError:
            confidence = 0.5

        return MarketRegime(
            direction=direction, volatility=volatility, style=style,
            confidence=confidence,
            description=fields.get("DESCRIPTION", ""),
        )


class RegimeAwareSynthesis:
    """Injects market regime context into RD-Agent Synthesis prompts."""

    def __init__(self, detector: MarketRegimeDetector | None = None):
        self.detector = detector or MarketRegimeDetector()
        self.current_regime: MarketRegime | None = None

    def update_regime(self, **kwargs):
        """Update regime from indicators or text."""
        if "market_summary" in kwargs:
            self.current_regime = self.detector.detect_from_text(kwargs["market_summary"])
        else:
            self.current_regime = self.detector.detect_from_indicators(**kwargs)
        logger.info(f"[Regime] Updated to {self.current_regime.label}")

    def augment_prompt(self, messages: list[dict]) -> list[dict]:
        """Inject regime context into synthesis prompt messages."""
        if not self.current_regime:
            return messages

        context = self.current_regime.to_prompt_context()
        hints = REGIME_FACTOR_HINTS.get(self.current_regime.label, [])
        if hints:
            context += "\nSuggested factor directions:\n" + "\n".join(
                f"- {h}" for h in hints
            )

        # Prepend regime context to the first user message
        augmented = []
        injected = False
        for msg in messages:
            if msg.get("role") == "user" and not injected:
                augmented.append({
                    "role": "user",
                    "content": f"[Market Context]\n{context}\n\n{msg['content']}",
                })
                injected = True
            else:
                augmented.append(msg)

        return augmented
