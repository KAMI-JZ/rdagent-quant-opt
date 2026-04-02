"""
Fundamental Analyst — 基本面分析 + 基本面信号→交易信号

将基本面数据（PE/PB/ROE/FCF/收入增长）转化为量化交易信号:
1. 估值评分: PE/PB/PEG 相对评估
2. 质量评分: ROE/ROA/毛利率/负债率
3. 成长性评分: 收入增长/利润增长/FCF增长
4. 综合信号: 多维度加权 → 买入/卖出/持有
5. 可选 LLM 深度分析 (通过 model_router)

经典框架:
- Graham Number: sqrt(22.5 * EPS * BPS)
- Piotroski F-Score: 9项财务健康度检查
- PEG Ratio: PE / 预期增长率

基于用户需求: "专用基本面分析LLM" + "基本面信号→交易信号转化skill"
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Signal(Enum):
    """交易信号"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class FundamentalData:
    """
    标准化基本面数据结构。
    所有字段可选，分析器根据可用数据计算。
    """
    ticker: str = ""
    # 估值
    pe_ratio: float | None = None          # 市盈率
    pb_ratio: float | None = None          # 市净率
    ps_ratio: float | None = None          # 市销率
    ev_ebitda: float | None = None         # 企业价值/EBITDA
    # 每股数据
    eps: float | None = None               # 每股收益
    bps: float | None = None               # 每股净资产
    dps: float | None = None               # 每股股息
    price: float | None = None             # 当前股价
    # 盈利能力
    roe: float | None = None               # 净资产收益率 (%)
    roa: float | None = None               # 总资产收益率 (%)
    gross_margin: float | None = None      # 毛利率 (%)
    net_margin: float | None = None        # 净利率 (%)
    # 成长性
    revenue_growth: float | None = None    # 收入同比增长 (%)
    earnings_growth: float | None = None   # 利润同比增长 (%)
    fcf_growth: float | None = None        # 自由现金流增长 (%)
    # 财务健康
    debt_equity: float | None = None       # 资产负债率
    current_ratio: float | None = None     # 流动比率
    fcf: float | None = None               # 自由现金流 (绝对值)
    # 行业基准
    sector_pe: float | None = None         # 行业平均 PE
    sector_pb: float | None = None         # 行业平均 PB


@dataclass
class ValuationScore:
    """估值评分结果"""
    score: float                # 0-100, 越高越被低估
    pe_score: float = 0.0
    pb_score: float = 0.0
    peg_score: float = 0.0
    graham_number: float = 0.0  # Graham 安全边际价格
    explanation: str = ""


@dataclass
class QualityScore:
    """质量评分结果"""
    score: float                # 0-100
    roe_score: float = 0.0
    margin_score: float = 0.0
    health_score: float = 0.0
    piotroski_f: int = 0        # Piotroski F-Score (0-9)
    explanation: str = ""


@dataclass
class GrowthScore:
    """成长性评分结果"""
    score: float                # 0-100
    revenue_score: float = 0.0
    earnings_score: float = 0.0
    fcf_score: float = 0.0
    explanation: str = ""


@dataclass
class FundamentalSignal:
    """综合基本面交易信号"""
    signal: Signal
    strength: float             # 信号强度 0-1
    composite_score: float      # 综合评分 0-100
    valuation: ValuationScore
    quality: QualityScore
    growth: GrowthScore
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.value,
            "strength": round(self.strength, 3),
            "composite_score": round(self.composite_score, 1),
            "valuation_score": round(self.valuation.score, 1),
            "quality_score": round(self.quality.score, 1),
            "growth_score": round(self.growth.score, 1),
            "explanation": self.explanation,
        }


class FundamentalAnalyzer:
    """
    基本面分析器: 纯计算，零 LLM 成本。

    三维度评估:
    - 估值 (40%): PE/PB/PEG + Graham Number
    - 质量 (35%): ROE/利润率/Piotroski F-Score
    - 成长 (25%): 收入/利润/FCF 增长率
    """

    # 影响: 这些权重决定综合评分的构成
    WEIGHTS = {"valuation": 0.40, "quality": 0.35, "growth": 0.25}

    def analyze(self, data: FundamentalData) -> FundamentalSignal:
        """分析基本面数据，生成交易信号。"""
        val = self._score_valuation(data)
        qual = self._score_quality(data)
        grow = self._score_growth(data)

        composite = (
            val.score * self.WEIGHTS["valuation"]
            + qual.score * self.WEIGHTS["quality"]
            + grow.score * self.WEIGHTS["growth"]
        )

        signal, strength = self._composite_to_signal(composite)

        explanation = (
            f"综合评分={composite:.1f}/100 "
            f"(估值{val.score:.0f}×40% + 质量{qual.score:.0f}×35% + 成长{grow.score:.0f}×25%)。"
            f"信号: {signal.value}, 强度: {strength:.0%}。"
        )

        return FundamentalSignal(
            signal=signal,
            strength=strength,
            composite_score=composite,
            valuation=val,
            quality=qual,
            growth=grow,
            explanation=explanation,
        )

    def _score_valuation(self, data: FundamentalData) -> ValuationScore:
        """估值评分: 越低估 → 分越高"""
        scores = []

        # PE 评分: 行业相对估值
        pe_score = 50.0
        if data.pe_ratio is not None:
            if data.pe_ratio <= 0:
                pe_score = 10.0  # 亏损
            elif data.pe_ratio < 10:
                pe_score = 90.0
            elif data.pe_ratio < 15:
                pe_score = 75.0
            elif data.pe_ratio < 20:
                pe_score = 60.0
            elif data.pe_ratio < 30:
                pe_score = 40.0
            else:
                pe_score = 20.0

            # 行业对比修正
            if data.sector_pe and data.sector_pe > 0:
                ratio = data.pe_ratio / data.sector_pe
                if ratio < 0.7:
                    pe_score = min(100, pe_score + 15)
                elif ratio > 1.3:
                    pe_score = max(0, pe_score - 15)

            scores.append(pe_score)

        # PB 评分
        pb_score = 50.0
        if data.pb_ratio is not None:
            if data.pb_ratio < 1.0:
                pb_score = 85.0
            elif data.pb_ratio < 2.0:
                pb_score = 70.0
            elif data.pb_ratio < 3.0:
                pb_score = 50.0
            else:
                pb_score = 25.0
            scores.append(pb_score)

        # PEG 评分: PE / 增长率
        peg_score = 50.0
        if data.pe_ratio and data.earnings_growth and data.earnings_growth > 0:
            peg = data.pe_ratio / data.earnings_growth
            if peg < 0.5:
                peg_score = 95.0
            elif peg < 1.0:
                peg_score = 80.0
            elif peg < 1.5:
                peg_score = 60.0
            elif peg < 2.0:
                peg_score = 40.0
            else:
                peg_score = 20.0
            scores.append(peg_score)

        # Graham Number
        graham = 0.0
        if data.eps and data.bps and data.eps > 0 and data.bps > 0:
            graham = math.sqrt(22.5 * data.eps * data.bps)

        total = sum(scores) / len(scores) if scores else 50.0

        explanation = f"PE评分={pe_score:.0f}, PB评分={pb_score:.0f}, PEG评分={peg_score:.0f}"
        if graham > 0:
            explanation += f", Graham安全价={graham:.2f}"

        return ValuationScore(
            score=total, pe_score=pe_score, pb_score=pb_score,
            peg_score=peg_score, graham_number=graham,
            explanation=explanation,
        )

    def _score_quality(self, data: FundamentalData) -> QualityScore:
        """质量评分: 盈利能力 + 财务健康"""
        scores = []

        # ROE 评分
        roe_score = 50.0
        if data.roe is not None:
            if data.roe > 20:
                roe_score = 90.0
            elif data.roe > 15:
                roe_score = 75.0
            elif data.roe > 10:
                roe_score = 60.0
            elif data.roe > 5:
                roe_score = 40.0
            else:
                roe_score = 20.0
            scores.append(roe_score)

        # 利润率评分
        margin_score = 50.0
        if data.net_margin is not None:
            if data.net_margin > 20:
                margin_score = 90.0
            elif data.net_margin > 10:
                margin_score = 70.0
            elif data.net_margin > 5:
                margin_score = 50.0
            elif data.net_margin > 0:
                margin_score = 30.0
            else:
                margin_score = 10.0
            scores.append(margin_score)

        # 财务健康评分
        health_score = 50.0
        if data.debt_equity is not None:
            if data.debt_equity < 0.3:
                health_score = 90.0
            elif data.debt_equity < 0.5:
                health_score = 75.0
            elif data.debt_equity < 1.0:
                health_score = 55.0
            else:
                health_score = 25.0
            scores.append(health_score)

        # Piotroski F-Score (简化版)
        f_score = self._piotroski_f_score(data)

        total = sum(scores) / len(scores) if scores else 50.0
        # F-Score 修正
        if f_score >= 7:
            total = min(100, total + 10)
        elif f_score <= 2:
            total = max(0, total - 10)

        return QualityScore(
            score=total, roe_score=roe_score, margin_score=margin_score,
            health_score=health_score, piotroski_f=f_score,
            explanation=f"ROE={roe_score:.0f}, 利润率={margin_score:.0f}, 健康={health_score:.0f}, F-Score={f_score}",
        )

    def _piotroski_f_score(self, data: FundamentalData) -> int:
        """
        Piotroski F-Score: 9项财务健康检查。
        每项通过 +1，满分 9。
        """
        score = 0

        # 1. 正 ROA
        if data.roa is not None and data.roa > 0:
            score += 1
        # 2. 正经营现金流
        if data.fcf is not None and data.fcf > 0:
            score += 1
        # 3. ROA 增长 (简化: ROA > 5% 算增长)
        if data.roa is not None and data.roa > 5:
            score += 1
        # 4. 现金流 > 净利润 (用 net_margin 代替)
        if data.fcf is not None and data.fcf > 0 and data.net_margin is not None and data.net_margin > 0:
            score += 1
        # 5. 低杠杆 (debt_equity < 0.5)
        if data.debt_equity is not None and data.debt_equity < 0.5:
            score += 1
        # 6. 良好流动性 (current_ratio > 1)
        if data.current_ratio is not None and data.current_ratio > 1:
            score += 1
        # 7. 无稀释 (简化: 跳过)
        # 8. 毛利率提升 (简化: gross_margin > 30%)
        if data.gross_margin is not None and data.gross_margin > 30:
            score += 1
        # 9. 资产周转率提升 (简化: revenue_growth > 0)
        if data.revenue_growth is not None and data.revenue_growth > 0:
            score += 1

        return score

    def _score_growth(self, data: FundamentalData) -> GrowthScore:
        """成长性评分"""
        scores = []

        revenue_score = 50.0
        if data.revenue_growth is not None:
            if data.revenue_growth > 30:
                revenue_score = 95.0
            elif data.revenue_growth > 20:
                revenue_score = 80.0
            elif data.revenue_growth > 10:
                revenue_score = 65.0
            elif data.revenue_growth > 0:
                revenue_score = 45.0
            else:
                revenue_score = 20.0
            scores.append(revenue_score)

        earnings_score = 50.0
        if data.earnings_growth is not None:
            if data.earnings_growth > 30:
                earnings_score = 95.0
            elif data.earnings_growth > 20:
                earnings_score = 80.0
            elif data.earnings_growth > 10:
                earnings_score = 65.0
            elif data.earnings_growth > 0:
                earnings_score = 45.0
            else:
                earnings_score = 20.0
            scores.append(earnings_score)

        fcf_score = 50.0
        if data.fcf_growth is not None:
            if data.fcf_growth > 20:
                fcf_score = 90.0
            elif data.fcf_growth > 10:
                fcf_score = 70.0
            elif data.fcf_growth > 0:
                fcf_score = 50.0
            else:
                fcf_score = 25.0
            scores.append(fcf_score)

        total = sum(scores) / len(scores) if scores else 50.0

        return GrowthScore(
            score=total, revenue_score=revenue_score,
            earnings_score=earnings_score, fcf_score=fcf_score,
            explanation=f"收入={revenue_score:.0f}, 利润={earnings_score:.0f}, FCF={fcf_score:.0f}",
        )

    @staticmethod
    def _composite_to_signal(composite: float) -> tuple[Signal, float]:
        """综合评分→交易信号"""
        if composite >= 80:
            return Signal.STRONG_BUY, min(1.0, (composite - 80) / 20 * 0.5 + 0.5)
        elif composite >= 65:
            return Signal.BUY, (composite - 65) / 15 * 0.5 + 0.3
        elif composite >= 45:
            return Signal.HOLD, 0.2
        elif composite >= 30:
            return Signal.SELL, (45 - composite) / 15 * 0.5 + 0.3
        else:
            return Signal.STRONG_SELL, min(1.0, (30 - composite) / 30 * 0.5 + 0.5)


class SignalGenerator:
    """
    将基本面信号转化为具体交易指令。
    影响: 这是从"分析"到"交易"的桥梁。
    """

    @staticmethod
    def to_trading_action(signal: FundamentalSignal, current_position: float = 0.0) -> dict:
        """
        基本面信号 → 交易指令。

        Args:
            signal: 基本面分析结果
            current_position: 当前持仓比例 (0-1)

        Returns:
            {"action": "buy/sell/hold", "target_position": float, "reason": str}
        """
        sig = signal.signal
        strength = signal.strength

        if sig == Signal.STRONG_BUY:
            target = min(1.0, 0.5 + strength * 0.5)
            action = "buy"
        elif sig == Signal.BUY:
            target = min(0.8, 0.3 + strength * 0.3)
            action = "buy"
        elif sig == Signal.HOLD:
            target = current_position  # 维持现有仓位
            action = "hold"
        elif sig == Signal.SELL:
            target = max(0.0, current_position * (1.0 - strength))
            action = "sell"
        else:  # STRONG_SELL
            target = 0.0
            action = "sell"

        return {
            "action": action,
            "target_position": round(target, 3),
            "current_position": round(current_position, 3),
            "position_change": round(target - current_position, 3),
            "signal": sig.value,
            "strength": round(strength, 3),
            "composite_score": round(signal.composite_score, 1),
            "reason": signal.explanation,
        }

    @staticmethod
    def batch_rank(signals: dict[str, FundamentalSignal]) -> list[tuple[str, float]]:
        """
        批量排名: 多只股票按综合评分排序。
        Returns: [(ticker, score), ...] 从高到低
        """
        ranked = [(ticker, sig.composite_score) for ticker, sig in signals.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class FundamentalLLMAnalyst:
    """
    LLM 增强的基本面分析 (可选)。
    通过 model_router 调用专用模型生成深度分析报告。
    """

    def __init__(self, router=None):
        self.router = router
        self.analyzer = FundamentalAnalyzer()

    def deep_analysis(
        self,
        data: FundamentalData,
        context: str = "",
    ) -> dict:
        """
        深度分析: 先跑量化评分，再用 LLM 生成解读。

        Args:
            data: 基本面数据
            context: 额外上下文 (如行业趋势、宏观环境)

        Returns:
            {"signal": FundamentalSignal, "llm_analysis": str}
        """
        signal = self.analyzer.analyze(data)

        if not self.router:
            return {"signal": signal, "llm_analysis": ""}

        # 构建 LLM prompt
        prompt = self._build_prompt(data, signal, context)

        try:
            messages = [
                {"role": "system", "content": (
                    "You are a senior fundamental analyst. Analyze the given financial "
                    "data and provide actionable insights in 3-5 sentences. Focus on: "
                    "1) Whether the valuation is justified by growth "
                    "2) Key risks to watch "
                    "3) Catalysts that could change the thesis"
                )},
                {"role": "user", "content": prompt},
            ]
            result = self.router.route(messages, max_tokens=512)
            llm_analysis = result["content"]
        except Exception as e:
            logger.warning(f"[FundamentalLLM] Analysis failed: {e}")
            llm_analysis = ""

        return {"signal": signal, "llm_analysis": llm_analysis}

    def _build_prompt(self, data: FundamentalData, signal: FundamentalSignal, context: str) -> str:
        lines = [f"Ticker: {data.ticker}"]
        if data.pe_ratio is not None:
            lines.append(f"PE: {data.pe_ratio:.1f}")
        if data.pb_ratio is not None:
            lines.append(f"PB: {data.pb_ratio:.1f}")
        if data.roe is not None:
            lines.append(f"ROE: {data.roe:.1f}%")
        if data.revenue_growth is not None:
            lines.append(f"Revenue Growth: {data.revenue_growth:.1f}%")
        if data.debt_equity is not None:
            lines.append(f"Debt/Equity: {data.debt_equity:.2f}")
        lines.append(f"\nQuantitative Signal: {signal.signal.value} (score={signal.composite_score:.1f})")
        lines.append(f"Valuation: {signal.valuation.explanation}")
        lines.append(f"Quality: {signal.quality.explanation}")
        lines.append(f"Growth: {signal.growth.explanation}")
        if context:
            lines.append(f"\nAdditional Context: {context}")
        return "\n".join(lines)
