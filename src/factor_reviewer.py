"""
Factor Reviewer — 因子深度归因审视系统

Performs attribution analysis on factor performance: identifies WHY a factor
makes money, decomposes return sources, and generates actionable reviews.

因子审视系统：分析因子的收益来源，回答"这个因子为什么赚钱/亏钱"。

Components:
1. ReturnAttribution: 分解收益来源（市场beta/行业/因子特有alpha）
2. StabilityAnalyzer: 分析因子表现的稳定性（IC衰减、分段表现）
3. FactorReviewer: 综合审视，生成结构化报告 + LLM 增强解读

Usage:
    reviewer = FactorReviewer()
    review = reviewer.review(factor_code, metrics, ic_series)
    print(review.to_markdown())
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

class ReviewGrade(Enum):
    """因子评级"""
    EXCELLENT = "A"   # IC > 0.04, Sharpe > 2, stable
    GOOD = "B"        # IC > 0.02, Sharpe > 1
    MEDIOCRE = "C"    # IC > 0.01, Sharpe > 0
    POOR = "D"        # IC > 0, Sharpe < 0
    FAIL = "F"        # IC <= 0 or fatal issues


class RiskFlag(Enum):
    """风险标记"""
    OVERFITTING = "overfitting"           # IC 衰减严重
    REGIME_DEPENDENT = "regime_dependent" # 只在特定市场状态有效
    CROWDED = "crowded"                   # 因子拥挤
    DATA_SNOOPING = "data_snooping"       # 数据窥探风险
    TAIL_RISK = "tail_risk"               # 尾部风险大
    LOW_CAPACITY = "low_capacity"         # 容量有限


@dataclass
class ReturnDecomposition:
    """收益分解"""
    total_return: float = 0.0        # 总年化收益
    market_beta_return: float = 0.0  # 市场beta贡献
    factor_alpha: float = 0.0       # 因子特有alpha
    residual: float = 0.0           # 残差

    @property
    def alpha_ratio(self) -> float:
        """Alpha占总收益的比例"""
        if abs(self.total_return) < 1e-8:
            return 0.0
        return self.factor_alpha / self.total_return


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_positive_ratio: float = 0.0    # IC > 0 的天数占比
    ic_decay_rate: float = 0.0        # IC 衰减速率（每月衰减百分比）
    worst_month_ic: float = 0.0       # 最差月 IC
    best_month_ic: float = 0.0        # 最佳月 IC
    max_consecutive_negative: int = 0  # 最长连续负 IC 天数
    hit_rate: float = 0.0             # 因子方向正确率


@dataclass
class FactorReview:
    """完整的因子审视报告"""
    grade: ReviewGrade = ReviewGrade.FAIL
    score: float = 0.0                # 0-100 综合评分
    summary: str = ""                 # 一句话总结
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    return_decomposition: ReturnDecomposition = field(default_factory=ReturnDecomposition)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    recommendations: list[str] = field(default_factory=list)
    # LLM 增强分析
    deep_analysis: str = ""
    # 原始数据
    backtest_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "grade": self.grade.value,
            "score": round(self.score, 1),
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "risk_flags": [f.value for f in self.risk_flags],
            "return_decomposition": {
                "total_return": self.return_decomposition.total_return,
                "market_beta_return": self.return_decomposition.market_beta_return,
                "factor_alpha": self.return_decomposition.factor_alpha,
                "alpha_ratio": self.return_decomposition.alpha_ratio,
            },
            "stability": {
                "ic_mean": self.stability.ic_mean,
                "ic_positive_ratio": self.stability.ic_positive_ratio,
                "ic_decay_rate": self.stability.ic_decay_rate,
                "hit_rate": self.stability.hit_rate,
            },
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        ic = self.backtest_metrics.get("IC", 0)
        sharpe = self.backtest_metrics.get("sharpe_ratio", 0)
        annual = self.backtest_metrics.get("annual_return", 0)
        mdd = self.backtest_metrics.get("max_drawdown", 0)

        risk_str = ", ".join(f.value for f in self.risk_flags) if self.risk_flags else "None"
        strengths_str = "\n".join(f"- {s}" for s in self.strengths) if self.strengths else "- None identified"
        weaknesses_str = "\n".join(f"- {w}" for w in self.weaknesses) if self.weaknesses else "- None identified"
        rec_str = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.recommendations)) if self.recommendations else "No recommendations"

        md = f"""# Factor Review Report

## Overall Grade: {self.grade.value} ({self.score:.0f}/100)
{self.summary}

## Performance Snapshot
| Metric | Value |
|--------|-------|
| IC | {ic:.4f} |
| Sharpe | {sharpe:.2f} |
| Annual Return | {annual:.1%} |
| Max Drawdown | {mdd:.1%} |
| IC Positive Ratio | {self.stability.ic_positive_ratio:.1%} |
| Hit Rate | {self.stability.hit_rate:.1%} |

## Return Decomposition
| Source | Contribution |
|--------|-------------|
| Total Return | {self.return_decomposition.total_return:.1%} |
| Market Beta | {self.return_decomposition.market_beta_return:.1%} |
| Factor Alpha | {self.return_decomposition.factor_alpha:.1%} |
| Alpha Ratio | {self.return_decomposition.alpha_ratio:.1%} |

## Stability
- IC Decay Rate: {self.stability.ic_decay_rate:.2%}/month
- Worst Month IC: {self.stability.worst_month_ic:.4f}
- Best Month IC: {self.stability.best_month_ic:.4f}
- Max Consecutive Negative IC Days: {self.stability.max_consecutive_negative}

## Strengths
{strengths_str}

## Weaknesses
{weaknesses_str}

## Risk Flags
{risk_str}

## Recommendations
{rec_str}
"""
        if self.deep_analysis:
            md += f"\n## Deep Analysis\n{self.deep_analysis}\n"

        return md


# ──────── Return Attribution ────────

class ReturnAttribution:
    """
    分解因子收益来源。

    简化版本（不需要真实市场数据）：
    - 用 Sharpe 和年化收益估算 alpha/beta 分解
    - 用 IC 稳定性评估因子独立性
    """

    # 影响: S&P 500 长期年化收益和波动率（历史均值）
    MARKET_ANNUAL_RETURN = 0.10  # ~10%
    MARKET_ANNUAL_VOL = 0.16     # ~16%

    def decompose(self, metrics: dict) -> ReturnDecomposition:
        """
        分解收益来源。

        Args:
            metrics: 回测指标 dict (annual_return, sharpe_ratio, max_drawdown, etc.)
        """
        total = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        mdd = abs(metrics.get("max_drawdown", 0))

        # 影响: 估算 beta — 用回撤和市场波动率的比值近似
        # 多空策略 beta 通常接近 0，但高回撤暗示市场暴露
        estimated_vol = abs(total / sharpe) if sharpe != 0 else self.MARKET_ANNUAL_VOL
        beta_estimate = min(mdd / 0.35, 1.0)  # 35% 是大熊市级别回撤

        # 影响: beta 贡献 = beta × 市场收益
        beta_return = beta_estimate * self.MARKET_ANNUAL_RETURN

        # 影响: alpha = 总收益 - beta 贡献
        alpha = total - beta_return
        residual = 0.0

        return ReturnDecomposition(
            total_return=total,
            market_beta_return=beta_return,
            factor_alpha=alpha,
            residual=residual,
        )


# ──────── Stability Analyzer ────────

class StabilityAnalyzer:
    """
    分析因子表现的时间稳定性。
    用 IC 序列评估因子是否持续有效。
    """

    def analyze(self, ic_series: list[float], metrics: dict) -> StabilityMetrics:
        """
        分析 IC 序列的稳定性。

        Args:
            ic_series: 每日 IC 值列表
            metrics: 回测指标
        """
        result = StabilityMetrics()

        if not ic_series:
            return result

        arr = np.array(ic_series, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            return result

        result.ic_mean = float(np.mean(arr))
        result.ic_std = float(np.std(arr))
        result.ic_positive_ratio = float(np.sum(arr > 0) / len(arr))

        # 影响: 命中率 = IC > 0 的比例（好因子 > 55%）
        result.hit_rate = result.ic_positive_ratio

        # 影响: IC 衰减率 — 前半段 vs 后半段的 IC 均值差异
        mid = len(arr) // 2
        if mid > 0:
            first_half_ic = float(np.mean(arr[:mid]))
            second_half_ic = float(np.mean(arr[mid:]))
            if abs(first_half_ic) > 1e-8:
                # 每月衰减率（假设数据跨约 12 个月）
                n_months = max(len(arr) / 21, 1)  # 21 交易日/月
                decay_total = (second_half_ic - first_half_ic) / abs(first_half_ic)
                result.ic_decay_rate = decay_total / (n_months / 2)
            else:
                result.ic_decay_rate = 0.0

        # 影响: 按月分段统计
        month_size = 21  # 约 21 交易日/月
        monthly_ics = []
        for i in range(0, len(arr), month_size):
            chunk = arr[i:i + month_size]
            if len(chunk) >= 5:
                monthly_ics.append(float(np.mean(chunk)))

        if monthly_ics:
            result.worst_month_ic = min(monthly_ics)
            result.best_month_ic = max(monthly_ics)

        # 影响: 最长连续负 IC 天数
        max_neg = 0
        current_neg = 0
        for v in arr:
            if v < 0:
                current_neg += 1
                max_neg = max(max_neg, current_neg)
            else:
                current_neg = 0
        result.max_consecutive_negative = max_neg

        return result


# ──────── Core Reviewer ────────

class FactorReviewer:
    """
    综合因子审视器。

    Usage:
        reviewer = FactorReviewer()
        review = reviewer.review(factor_code, metrics, ic_series)
        print(review.grade, review.score)

    With LLM:
        reviewer = FactorReviewer(router=my_router)
        review = reviewer.review(factor_code, metrics, ic_series, enhance=True)
    """

    def __init__(self, router=None):
        self._attribution = ReturnAttribution()
        self._stability = StabilityAnalyzer()
        self._router = router

    def review(
        self,
        factor_code: str = "",
        backtest_metrics: Optional[dict] = None,
        ic_series: Optional[list[float]] = None,
        hypothesis: str = "",
        enhance: bool = False,
    ) -> FactorReview:
        """
        执行完整的因子审视。

        Args:
            factor_code: 因子 Python 代码
            backtest_metrics: 回测指标
            ic_series: 每日 IC 序列
            hypothesis: 因子假设
            enhance: 是否用 LLM 增强分析
        """
        metrics = backtest_metrics or {}
        ic_list = ic_series or []

        review = FactorReview(backtest_metrics=metrics)

        # 影响: 收益分解
        review.return_decomposition = self._attribution.decompose(metrics)

        # 影响: 稳定性分析
        review.stability = self._stability.analyze(ic_list, metrics)

        # 影响: 综合评分（0-100）
        review.score = self._compute_score(metrics, review.stability, review.return_decomposition)

        # 影响: 评级
        review.grade = self._assign_grade(review.score, metrics)

        # 影响: 优缺点
        review.strengths = self._identify_strengths(metrics, review.stability, review.return_decomposition)
        review.weaknesses = self._identify_weaknesses(metrics, review.stability, review.return_decomposition)

        # 影响: 风险标记
        review.risk_flags = self._detect_risks(metrics, review.stability)

        # 影响: 建议
        review.recommendations = self._generate_recommendations(review)

        # 影响: 一句话总结
        review.summary = self._generate_summary(review)

        # 影响: LLM 增强（可选）
        if enhance and self._router:
            review.deep_analysis = self._enhance_with_llm(review, factor_code, hypothesis)

        logger.info(f"[Reviewer] Grade={review.grade.value}, Score={review.score:.0f}, "
                    f"Risks={[f.value for f in review.risk_flags]}")

        return review

    def _compute_score(self, metrics: dict, stability: StabilityMetrics,
                       decomp: ReturnDecomposition) -> float:
        """
        计算综合评分（0-100）。

        权重分配：
        - IC 绝对值: 25分
        - Sharpe: 25分
        - IC 稳定性 (positive ratio): 20分
        - Alpha 纯度: 15分
        - 回撤控制: 15分
        """
        score = 0.0

        # IC 分 (0-25)
        ic = abs(metrics.get("IC", 0))
        score += min(ic / 0.05, 1.0) * 25

        # Sharpe 分 (0-25)
        sharpe = metrics.get("sharpe_ratio", 0)
        score += min(max(sharpe, 0) / 3.0, 1.0) * 25

        # IC 稳定性分 (0-20)
        score += min(stability.ic_positive_ratio / 0.65, 1.0) * 20

        # Alpha 纯度分 (0-15)
        alpha_ratio = decomp.alpha_ratio
        if alpha_ratio > 0:
            score += min(alpha_ratio, 1.0) * 15

        # 回撤控制分 (0-15)
        mdd = abs(metrics.get("max_drawdown", 1.0))
        if mdd < 0.05:
            score += 15
        elif mdd < 0.10:
            score += 12
        elif mdd < 0.20:
            score += 8
        elif mdd < 0.30:
            score += 4

        return min(score, 100)

    def _assign_grade(self, score: float, metrics: dict) -> ReviewGrade:
        """根据评分分配等级"""
        ic = metrics.get("IC", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        if ic <= 0 or sharpe < -1:
            return ReviewGrade.FAIL
        if score >= 80:
            return ReviewGrade.EXCELLENT
        if score >= 60:
            return ReviewGrade.GOOD
        if score >= 40:
            return ReviewGrade.MEDIOCRE
        if score >= 20:
            return ReviewGrade.POOR
        return ReviewGrade.FAIL

    def _identify_strengths(self, metrics: dict, stability: StabilityMetrics,
                           decomp: ReturnDecomposition) -> list[str]:
        strengths = []
        ic = metrics.get("IC", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        mdd = abs(metrics.get("max_drawdown", 1))

        if ic > 0.04:
            strengths.append(f"Exceptional IC ({ic:.4f}) — strong predictive power")
        elif ic > 0.02:
            strengths.append(f"Solid IC ({ic:.4f}) — meaningful signal")

        if sharpe > 2:
            strengths.append(f"Outstanding risk-adjusted returns (Sharpe={sharpe:.2f})")
        elif sharpe > 1:
            strengths.append(f"Good risk-adjusted returns (Sharpe={sharpe:.2f})")

        if mdd < 0.10:
            strengths.append(f"Well-controlled drawdown ({mdd:.1%})")

        if stability.ic_positive_ratio > 0.6:
            strengths.append(f"Consistent signal ({stability.ic_positive_ratio:.0%} positive IC days)")

        if decomp.alpha_ratio > 0.8:
            strengths.append("High alpha purity — returns are factor-driven, not market-driven")

        return strengths

    def _identify_weaknesses(self, metrics: dict, stability: StabilityMetrics,
                            decomp: ReturnDecomposition) -> list[str]:
        weaknesses = []
        ic = metrics.get("IC", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        mdd = abs(metrics.get("max_drawdown", 0))

        if ic < 0.01:
            weaknesses.append(f"Weak IC ({ic:.4f}) — signal barely above noise")

        if sharpe < 0:
            weaknesses.append(f"Negative Sharpe ({sharpe:.2f}) — loses money risk-adjusted")
        elif sharpe < 0.5:
            weaknesses.append(f"Low Sharpe ({sharpe:.2f}) — insufficient compensation for risk")

        if mdd > 0.20:
            weaknesses.append(f"Large drawdown ({mdd:.1%}) — painful to live through")

        if stability.ic_positive_ratio < 0.5:
            weaknesses.append(f"Inconsistent signal ({stability.ic_positive_ratio:.0%} positive IC days)")

        if stability.ic_decay_rate < -0.05:
            weaknesses.append(f"IC decaying ({stability.ic_decay_rate:.1%}/month) — alpha may be ephemeral")

        if stability.max_consecutive_negative > 20:
            weaknesses.append(f"Long losing streak ({stability.max_consecutive_negative} consecutive negative IC days)")

        if decomp.alpha_ratio < 0.3 and decomp.total_return > 0:
            weaknesses.append("Low alpha purity — most returns come from market exposure")

        return weaknesses

    def _detect_risks(self, metrics: dict, stability: StabilityMetrics) -> list[RiskFlag]:
        risks = []

        # 影响: 过拟合风险 — IC 衰减严重
        if stability.ic_decay_rate < -0.10:
            risks.append(RiskFlag.OVERFITTING)

        # 影响: 状态依赖 — 月度 IC 方差太大
        if stability.worst_month_ic < -0.03 and stability.best_month_ic > 0.05:
            risks.append(RiskFlag.REGIME_DEPENDENT)

        # 影响: 尾部风险 — 回撤相对年化收益过大
        annual = metrics.get("annual_return", 0)
        mdd = abs(metrics.get("max_drawdown", 0))
        if annual > 0 and mdd / annual > 0.5:
            risks.append(RiskFlag.TAIL_RISK)

        # 影响: 数据窥探 — 天数太少
        n_days = metrics.get("n_days", 0)
        if n_days < 60:
            risks.append(RiskFlag.DATA_SNOOPING)

        # 影响: 容量有限 — 股票数太少
        n_stocks = metrics.get("n_stocks", 0)
        if n_stocks < 20:
            risks.append(RiskFlag.LOW_CAPACITY)

        return risks

    def _generate_recommendations(self, review: FactorReview) -> list[str]:
        recs = []

        if RiskFlag.OVERFITTING in review.risk_flags:
            recs.append("Test on out-of-sample data to verify signal persistence")

        if RiskFlag.DATA_SNOOPING in review.risk_flags:
            recs.append("Extend backtest period to at least 3 years for statistical significance")

        if review.stability.ic_decay_rate < -0.05:
            recs.append("Consider adaptive parameter recalibration to combat alpha decay")

        if review.return_decomposition.alpha_ratio < 0.5 and review.return_decomposition.total_return > 0:
            recs.append("Hedge market beta to isolate pure alpha returns")

        grade = review.grade
        if grade == ReviewGrade.EXCELLENT:
            recs.append("Factor is production-ready — monitor for crowding and decay")
        elif grade == ReviewGrade.GOOD:
            recs.append("Factor is promising — consider combining with complementary signals")
        elif grade == ReviewGrade.MEDIOCRE:
            recs.append("Factor needs improvement — try different lookback windows or signal transforms")
        elif grade in (ReviewGrade.POOR, ReviewGrade.FAIL):
            recs.append("Factor should be abandoned or fundamentally rethought")

        return recs

    def _generate_summary(self, review: FactorReview) -> str:
        grade = review.grade.value
        score = review.score
        n_risks = len(review.risk_flags)

        ic = review.backtest_metrics.get("IC", 0)
        sharpe = review.backtest_metrics.get("sharpe_ratio", 0)

        if review.grade == ReviewGrade.EXCELLENT:
            return (f"Grade {grade} ({score:.0f}/100): Exceptional factor with IC={ic:.4f} "
                    f"and Sharpe={sharpe:.2f}. {n_risks} risk flags.")
        elif review.grade == ReviewGrade.GOOD:
            return (f"Grade {grade} ({score:.0f}/100): Solid factor with room for improvement. "
                    f"IC={ic:.4f}, Sharpe={sharpe:.2f}.")
        elif review.grade == ReviewGrade.MEDIOCRE:
            return (f"Grade {grade} ({score:.0f}/100): Mediocre factor — signal exists but weak. "
                    f"IC={ic:.4f}, Sharpe={sharpe:.2f}.")
        else:
            return (f"Grade {grade} ({score:.0f}/100): Factor underperforms. "
                    f"IC={ic:.4f}, Sharpe={sharpe:.2f}. Consider abandoning.")

    def _enhance_with_llm(self, review: FactorReview, factor_code: str,
                         hypothesis: str) -> str:
        """LLM 增强深度分析"""
        if not self._router:
            return ""

        try:
            from .model_router import PipelineStage

            prompt = (
                f"You are a senior quantitative researcher reviewing a factor.\n\n"
                f"Grade: {review.grade.value} ({review.score:.0f}/100)\n"
                f"IC: {review.backtest_metrics.get('IC', 0):.4f}\n"
                f"Sharpe: {review.backtest_metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Annual Return: {review.backtest_metrics.get('annual_return', 0):.1%}\n"
                f"Max Drawdown: {review.backtest_metrics.get('max_drawdown', 0):.1%}\n"
                f"Alpha Ratio: {review.return_decomposition.alpha_ratio:.1%}\n"
                f"IC Positive Ratio: {review.stability.ic_positive_ratio:.1%}\n"
                f"IC Decay Rate: {review.stability.ic_decay_rate:.2%}/month\n"
                f"Risk Flags: {[f.value for f in review.risk_flags]}\n"
            )
            if hypothesis:
                prompt += f"Hypothesis: {hypothesis}\n"
            if factor_code:
                prompt += f"\nFactor Code:\n```python\n{factor_code[:500]}\n```\n"

            prompt += (
                "\nProvide a 3-paragraph deep analysis:\n"
                "1. What economic mechanism drives this factor's returns?\n"
                "2. What are the key risks and how might they materialize?\n"
                "3. What specific improvements would increase the factor's Sharpe ratio?"
            )

            response = self._router.route_and_call(
                stage=PipelineStage.ANALYSIS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.get("content", "")

        except Exception as e:
            logger.warning(f"[Reviewer] LLM enhancement failed: {e}")
            return ""


# ──────── Batch Review ────────

def review_run_results(
    results: list[dict],
    router=None,
) -> list[FactorReview]:
    """
    批量审视一次运行的所有因子。

    Args:
        results: pipeline run 的 results.json 列表
        router: 可选 LLM router

    Returns:
        list[FactorReview]
    """
    reviewer = FactorReviewer(router=router)
    reviews = []

    for r in results:
        if r.get("skipped") or not r.get("backtest_metrics"):
            continue

        review = reviewer.review(
            factor_code=r.get("factor_code", ""),
            backtest_metrics=r["backtest_metrics"],
            ic_series=r.get("ic_series", []),
            hypothesis=r.get("hypothesis", ""),
        )
        reviews.append(review)

    logger.info(f"[Reviewer] Reviewed {len(reviews)} factors: "
                f"{sum(1 for r in reviews if r.grade in (ReviewGrade.EXCELLENT, ReviewGrade.GOOD))} good, "
                f"{sum(1 for r in reviews if r.grade == ReviewGrade.FAIL)} failed")
    return reviews
