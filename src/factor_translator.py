"""
Factor Translator — 因子代码 → 标准化交易策略语言

Converts raw factor code (calculate_factor) into structured trading rules:
entry/exit signals, position sizing, risk constraints.

因子代码是数学公式，但交易需要明确的进出场规则和仓位管理。
这个模块把因子代码翻译成可执行的交易策略描述。

Usage:
    translator = FactorTranslator()
    strategy = translator.translate(factor_code, backtest_metrics)
    print(strategy.to_markdown())

Design:
- AST analysis: extracts factor type (momentum/mean-revert/volatility/etc.)
- Rule generation: maps factor characteristics to trading rules
- LLM enhancement (optional): uses router for natural language explanation
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

class FactorType(Enum):
    """因子类型分类"""
    MOMENTUM = "momentum"           # 动量/趋势跟踪
    MEAN_REVERSION = "mean_reversion"  # 均值回复
    VOLATILITY = "volatility"       # 波动率相关
    VOLUME = "volume"               # 成交量相关
    COMPOSITE = "composite"         # 多因子复合
    UNKNOWN = "unknown"


class SignalDirection(Enum):
    """信号方向"""
    LONG_ONLY = "long_only"         # 只做多
    SHORT_ONLY = "short_only"       # 只做空
    LONG_SHORT = "long_short"       # 多空双向


class PositionMethod(Enum):
    """仓位分配方法"""
    EQUAL_WEIGHT = "equal_weight"           # 等权
    FACTOR_WEIGHTED = "factor_weighted"     # 因子值加权
    RISK_PARITY = "risk_parity"             # 风险平价
    INVERSE_VOL = "inverse_volatility"      # 反波动率加权


@dataclass
class EntryRule:
    """进场规则"""
    condition: str = ""     # 人类可读条件，如 "因子值 > 0 时买入"
    threshold: float = 0.0  # 阈值
    lookback_days: int = 0  # 回看天数
    description: str = ""


@dataclass
class ExitRule:
    """出场规则"""
    stop_loss_pct: float = -0.05       # 止损比例 (-5%)
    take_profit_pct: float = 0.15      # 止盈比例 (+15%)
    max_holding_days: int = 20         # 最大持仓天数
    trailing_stop_pct: float = 0.0     # 移动止损 (0=不用)
    description: str = ""


@dataclass
class PositionRule:
    """仓位规则"""
    method: PositionMethod = PositionMethod.EQUAL_WEIGHT
    max_position_pct: float = 0.05     # 单只最大仓位 5%
    max_sector_pct: float = 0.25       # 单行业最大 25%
    max_turnover_daily: float = 0.20   # 每日最大换手 20%
    n_holdings: int = 20               # 目标持仓数量
    description: str = ""


@dataclass
class TradingStrategy:
    """完整的交易策略描述"""
    name: str = ""
    factor_type: FactorType = FactorType.UNKNOWN
    signal_direction: SignalDirection = SignalDirection.LONG_SHORT
    entry: EntryRule = field(default_factory=EntryRule)
    exit: ExitRule = field(default_factory=ExitRule)
    position: PositionRule = field(default_factory=PositionRule)
    rebalance_frequency: str = "daily"   # daily / weekly / monthly
    universe: str = "S&P 500"
    # 原始数据
    factor_code: str = ""
    backtest_metrics: dict = field(default_factory=dict)
    # LLM 生成的自然语言描述
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "factor_type": self.factor_type.value,
            "signal_direction": self.signal_direction.value,
            "entry": {
                "condition": self.entry.condition,
                "threshold": self.entry.threshold,
                "lookback_days": self.entry.lookback_days,
            },
            "exit": {
                "stop_loss_pct": self.exit.stop_loss_pct,
                "take_profit_pct": self.exit.take_profit_pct,
                "max_holding_days": self.exit.max_holding_days,
                "trailing_stop_pct": self.exit.trailing_stop_pct,
            },
            "position": {
                "method": self.position.method.value,
                "max_position_pct": self.position.max_position_pct,
                "n_holdings": self.position.n_holdings,
                "max_turnover_daily": self.position.max_turnover_daily,
            },
            "rebalance_frequency": self.rebalance_frequency,
            "universe": self.universe,
            "narrative": self.narrative,
        }

    def to_markdown(self) -> str:
        """生成人类可读的 Markdown 策略报告"""
        ic = self.backtest_metrics.get("IC", 0)
        sharpe = self.backtest_metrics.get("sharpe_ratio", 0)
        annual_ret = self.backtest_metrics.get("annual_return", 0)
        mdd = self.backtest_metrics.get("max_drawdown", 0)

        md = f"""# Trading Strategy: {self.name}

## Factor Profile
- **Type**: {self.factor_type.value}
- **Direction**: {self.signal_direction.value}
- **Universe**: {self.universe}
- **Rebalance**: {self.rebalance_frequency}

## Entry Rules
- {self.entry.condition}
- Lookback: {self.entry.lookback_days} days

## Exit Rules
- Stop Loss: {self.exit.stop_loss_pct:.1%}
- Take Profit: {self.exit.take_profit_pct:.1%}
- Max Holding: {self.exit.max_holding_days} days
{f"- Trailing Stop: {self.exit.trailing_stop_pct:.1%}" if self.exit.trailing_stop_pct else ""}

## Position Sizing
- Method: {self.position.method.value}
- Holdings: {self.position.n_holdings} stocks
- Max Single Position: {self.position.max_position_pct:.1%}
- Max Daily Turnover: {self.position.max_turnover_daily:.1%}

## Backtest Performance
| Metric | Value |
|--------|-------|
| IC | {ic:.4f} |
| Sharpe | {sharpe:.2f} |
| Annual Return | {annual_ret:.1%} |
| Max Drawdown | {mdd:.1%} |

## Strategy Narrative
{self.narrative if self.narrative else "_No narrative generated_"}
"""
        return md


# ──────── AST-Based Factor Analyzer ────────

# 影响: 关键词 → 因子类型映射表
_FACTOR_KEYWORDS = {
    FactorType.MOMENTUM: [
        "pct_change", "diff", "shift", "momentum", "roc", "rate_of_change",
        "trend", "ma_cross", "breakout", "rolling.*mean",
    ],
    FactorType.MEAN_REVERSION: [
        "z_score", "zscore", "deviation", "mean_revert", "bollinger",
        "rsi", "overbought", "oversold", "revert",
    ],
    FactorType.VOLATILITY: [
        "std", "var", "volatility", "atr", "true_range",
        "rolling.*std", "ewm.*std", "vix",
    ],
    FactorType.VOLUME: [
        "volume", "vwap", "obv", "on_balance", "money_flow",
        "accumulation", "distribution", "turnover",
    ],
}

# 影响: 常见滚动窗口 → 换仓频率映射
_WINDOW_TO_REBALANCE = {
    (1, 5): "daily",
    (5, 20): "weekly",
    (20, 60): "weekly",
    (60, 252): "monthly",
}


class FactorAnalyzer:
    """
    AST-based factor code analyzer.
    通过代码分析提取因子特征：类型、窗口、方向。
    """

    def analyze(self, factor_code: str) -> dict:
        """
        分析因子代码，返回特征字典。

        Returns:
            dict with keys: factor_type, windows, uses_volume, complexity, keywords_found
        """
        result = {
            "factor_type": FactorType.UNKNOWN,
            "windows": [],
            "uses_volume": False,
            "complexity": 0,
            "keywords_found": [],
        }

        code_lower = factor_code.lower()

        # 影响: 关键词匹配确定因子类型（volume 列直接使用加权 +2）
        type_scores: dict[FactorType, int] = {}
        for ftype, keywords in _FACTOR_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if re.search(kw, code_lower):
                    score += 1
                    result["keywords_found"].append(kw)
            if score > 0:
                type_scores[ftype] = score

        # 影响: 如果代码直接操作 volume 列（df['volume']），volume 类型额外加权
        if re.search(r"df\[.volume.\]|df\.volume", code_lower):
            type_scores[FactorType.VOLUME] = type_scores.get(FactorType.VOLUME, 0) + 2

        if type_scores:
            max_score = max(type_scores.values())
            tied = [t for t, s in type_scores.items() if s == max_score]

            if len(tied) == 1:
                result["factor_type"] = tied[0]
            elif len(tied) >= 3:
                # 影响: 3种以上类型同分 → composite
                result["factor_type"] = FactorType.COMPOSITE
            else:
                # 影响: 2种类型同分 → 用优先级打破平局
                # 专用类型 (mean_reversion, volume) 优先于通用类型 (momentum, volatility)
                priority = [FactorType.MEAN_REVERSION, FactorType.VOLUME,
                           FactorType.MOMENTUM, FactorType.VOLATILITY]
                for p in priority:
                    if p in tied:
                        result["factor_type"] = p
                        break
                else:
                    result["factor_type"] = tied[0]

        # 影响: 提取滚动窗口大小
        window_pattern = r'rolling\s*\(\s*(\d+)\s*\)|shift\s*\(\s*(-?\d+)\s*\)|pct_change\s*\(\s*(\d+)\s*\)'
        for match in re.finditer(window_pattern, factor_code):
            for g in match.groups():
                if g is not None:
                    result["windows"].append(abs(int(g)))

        # 影响: 是否使用成交量
        if re.search(r'\bvolume\b', code_lower):
            result["uses_volume"] = True

        # 影响: AST 复杂度分析
        try:
            tree = ast.parse(factor_code)
            result["complexity"] = sum(1 for _ in ast.walk(tree))
        except SyntaxError:
            result["complexity"] = len(factor_code.split("\n"))

        return result


# ──────── Core Translator ────────

class FactorTranslator:
    """
    Translates factor code into structured trading strategies.
    将因子代码翻译为结构化的交易策略。

    Usage:
        translator = FactorTranslator()
        strategy = translator.translate(factor_code, metrics)
        print(strategy.to_markdown())

    With LLM enhancement:
        translator = FactorTranslator(router=my_router)
        strategy = translator.translate(factor_code, metrics, enhance=True)
    """

    def __init__(self, router=None):
        """
        Args:
            router: Optional MultiModelRouter for LLM-enhanced translation
        """
        self._analyzer = FactorAnalyzer()
        self._router = router

    def translate(
        self,
        factor_code: str,
        backtest_metrics: Optional[dict] = None,
        hypothesis: str = "",
        enhance: bool = False,
    ) -> TradingStrategy:
        """
        将因子代码翻译为交易策略。

        Args:
            factor_code: Python 因子代码 (含 calculate_factor 函数)
            backtest_metrics: 回测指标 dict (IC, Sharpe, etc.)
            hypothesis: 因子假设描述 (可选，帮助生成更好的叙述)
            enhance: 是否使用 LLM 增强叙述

        Returns:
            TradingStrategy 完整交易策略
        """
        metrics = backtest_metrics or {}
        analysis = self._analyzer.analyze(factor_code)

        strategy = TradingStrategy(
            factor_code=factor_code,
            backtest_metrics=metrics,
        )

        # 影响: 根据分析结果填充策略
        strategy.factor_type = analysis["factor_type"]
        strategy.name = self._generate_name(analysis)
        strategy.signal_direction = self._infer_direction(analysis, metrics)
        strategy.entry = self._generate_entry_rule(analysis, metrics)
        strategy.exit = self._generate_exit_rule(analysis, metrics)
        strategy.position = self._generate_position_rule(analysis, metrics)
        strategy.rebalance_frequency = self._infer_rebalance(analysis)

        # 影响: 生成基础叙述
        strategy.narrative = self._generate_narrative(strategy, analysis, hypothesis)

        # 影响: LLM 增强（可选，消耗 1 次 LLM 调用）
        if enhance and self._router:
            strategy.narrative = self._enhance_with_llm(strategy, hypothesis)

        logger.info(
            f"[Translator] {strategy.name}: {strategy.factor_type.value}, "
            f"{strategy.signal_direction.value}, rebalance={strategy.rebalance_frequency}"
        )

        return strategy

    def _generate_name(self, analysis: dict) -> str:
        """根据因子类型生成策略名称"""
        ftype = analysis["factor_type"]
        windows = analysis["windows"]
        window_str = f"_{max(windows)}d" if windows else ""

        names = {
            FactorType.MOMENTUM: f"Momentum{window_str}",
            FactorType.MEAN_REVERSION: f"MeanRevert{window_str}",
            FactorType.VOLATILITY: f"VolSignal{window_str}",
            FactorType.VOLUME: f"VolumeSignal{window_str}",
            FactorType.COMPOSITE: f"Composite{window_str}",
            FactorType.UNKNOWN: f"Alpha{window_str}",
        }
        return names.get(ftype, f"Alpha{window_str}")

    def _infer_direction(self, analysis: dict, metrics: dict) -> SignalDirection:
        """根据因子类型和回测结果推断信号方向"""
        ftype = analysis["factor_type"]
        ic = metrics.get("IC", 0)

        # 动量因子通常做多（正 IC = 高因子值 → 高收益）
        if ftype == FactorType.MOMENTUM and ic > 0:
            return SignalDirection.LONG_ONLY

        # 均值回复通常多空双向
        if ftype == FactorType.MEAN_REVERSION:
            return SignalDirection.LONG_SHORT

        # 默认多空双向（因为回测用的就是多空组合）
        return SignalDirection.LONG_SHORT

    def _generate_entry_rule(self, analysis: dict, metrics: dict) -> EntryRule:
        """生成进场规则"""
        ftype = analysis["factor_type"]
        windows = analysis["windows"]
        lookback = max(windows) if windows else 20

        rules = {
            FactorType.MOMENTUM: EntryRule(
                condition="Buy when factor value ranks in top 20% (strong upward momentum)",
                threshold=0.8,
                lookback_days=lookback,
                description="趋势跟踪：因子值排名前20%时做多",
            ),
            FactorType.MEAN_REVERSION: EntryRule(
                condition="Buy when factor z-score < -1.5 (oversold); short when > 1.5 (overbought)",
                threshold=1.5,
                lookback_days=lookback,
                description="均值回复：超卖时买入，超买时卖出",
            ),
            FactorType.VOLATILITY: EntryRule(
                condition="Buy when volatility factor signals regime shift (low→high transition)",
                threshold=0.0,
                lookback_days=lookback,
                description="波动率信号：状态转换时进场",
            ),
            FactorType.VOLUME: EntryRule(
                condition="Buy when volume factor exceeds 2x average (unusual activity)",
                threshold=2.0,
                lookback_days=lookback,
                description="成交量异动：放量时进场",
            ),
        }

        return rules.get(ftype, EntryRule(
            condition="Buy/sell based on factor quintile ranking (top 20% long, bottom 20% short)",
            threshold=0.0,
            lookback_days=lookback,
            description="分位数排名进场",
        ))

    def _generate_exit_rule(self, analysis: dict, metrics: dict) -> ExitRule:
        """根据回测指标生成出场规则"""
        mdd = abs(metrics.get("max_drawdown", 0.10))
        sharpe = metrics.get("sharpe_ratio", 0)
        windows = analysis["windows"]
        holding = max(windows) if windows else 20

        # 影响: 止损根据历史最大回撤设定（1.5x MDD）
        stop_loss = -min(mdd * 1.5, 0.20)

        # 影响: 止盈根据 Sharpe 调整（高 Sharpe → 可以更贪）
        if sharpe > 2:
            take_profit = 0.25
            trailing = 0.08
        elif sharpe > 1:
            take_profit = 0.15
            trailing = 0.05
        else:
            take_profit = 0.10
            trailing = 0.0

        return ExitRule(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_holding_days=holding,
            trailing_stop_pct=trailing,
            description=f"止损{stop_loss:.1%}, 止盈{take_profit:.1%}, 最长持仓{holding}天",
        )

    def _generate_position_rule(self, analysis: dict, metrics: dict) -> PositionRule:
        """生成仓位管理规则"""
        sharpe = metrics.get("sharpe_ratio", 0)
        n_stocks = metrics.get("n_stocks", 50)

        # 影响: 高 Sharpe 因子可以更集中持仓
        if sharpe > 2:
            n_holdings = min(20, n_stocks // 5)
            max_pos = 0.08
            method = PositionMethod.FACTOR_WEIGHTED
        elif sharpe > 1:
            n_holdings = min(30, n_stocks // 4)
            max_pos = 0.05
            method = PositionMethod.FACTOR_WEIGHTED
        else:
            n_holdings = min(50, n_stocks // 3)
            max_pos = 0.04
            method = PositionMethod.EQUAL_WEIGHT

        # 影响: 波动率类因子适合反波动率加权
        if analysis["factor_type"] == FactorType.VOLATILITY:
            method = PositionMethod.INVERSE_VOL

        return PositionRule(
            method=method,
            max_position_pct=max_pos,
            n_holdings=n_holdings,
            description=f"{method.value}, {n_holdings}只, 单只最大{max_pos:.0%}",
        )

    def _infer_rebalance(self, analysis: dict) -> str:
        """根据滚动窗口推断换仓频率"""
        windows = analysis["windows"]
        if not windows:
            return "daily"

        max_window = max(windows)
        for (lo, hi), freq in _WINDOW_TO_REBALANCE.items():
            if lo <= max_window < hi:
                return freq
        return "monthly" if max_window >= 252 else "daily"

    def _generate_narrative(self, strategy: TradingStrategy, analysis: dict,
                           hypothesis: str) -> str:
        """生成基础策略叙述（不使用 LLM）"""
        ic = strategy.backtest_metrics.get("IC", 0)
        sharpe = strategy.backtest_metrics.get("sharpe_ratio", 0)
        annual_ret = strategy.backtest_metrics.get("annual_return", 0)

        parts = []

        # 策略类型描述
        type_desc = {
            FactorType.MOMENTUM: "a momentum-based strategy that captures trending price movements",
            FactorType.MEAN_REVERSION: "a mean-reversion strategy that profits from price dislocations",
            FactorType.VOLATILITY: "a volatility-driven strategy that trades regime shifts",
            FactorType.VOLUME: "a volume-based strategy that identifies unusual trading activity",
            FactorType.COMPOSITE: "a composite strategy combining multiple signal sources",
            FactorType.UNKNOWN: "a quantitative strategy based on derived price/volume features",
        }
        parts.append(f"This is {type_desc.get(strategy.factor_type, 'a quantitative strategy')}.")

        # 原始假设
        if hypothesis:
            parts.append(f"Hypothesis: {hypothesis}")

        # 绩效总结
        if sharpe > 2:
            parts.append(f"Strong performance with Sharpe {sharpe:.2f} and IC {ic:.4f}.")
        elif sharpe > 1:
            parts.append(f"Solid performance with Sharpe {sharpe:.2f} and IC {ic:.4f}.")
        elif sharpe > 0:
            parts.append(f"Moderate performance with Sharpe {sharpe:.2f} and IC {ic:.4f}.")
        else:
            parts.append(f"Weak performance with Sharpe {sharpe:.2f} — needs improvement.")

        # 窗口信息
        windows = analysis["windows"]
        if windows:
            parts.append(f"Uses {', '.join(str(w) + '-day' for w in sorted(set(windows)))} lookback windows.")

        # 风控总结
        parts.append(
            f"Risk management: {strategy.exit.stop_loss_pct:.1%} stop-loss, "
            f"{strategy.exit.take_profit_pct:.1%} take-profit, "
            f"{strategy.position.n_holdings} holdings with {strategy.position.method.value} sizing."
        )

        return " ".join(parts)

    def _enhance_with_llm(self, strategy: TradingStrategy, hypothesis: str) -> str:
        """使用 LLM 生成增强版策略叙述"""
        if not self._router:
            return strategy.narrative

        try:
            from .model_router import PipelineStage

            prompt = (
                f"You are a quantitative analyst. Given this factor's trading strategy, "
                f"write a concise 3-paragraph explanation for a portfolio manager:\n\n"
                f"Factor type: {strategy.factor_type.value}\n"
                f"Direction: {strategy.signal_direction.value}\n"
                f"Entry: {strategy.entry.condition}\n"
                f"Exit: stop-loss {strategy.exit.stop_loss_pct:.1%}, "
                f"take-profit {strategy.exit.take_profit_pct:.1%}\n"
                f"Position: {strategy.position.method.value}, "
                f"{strategy.position.n_holdings} holdings\n"
                f"IC: {strategy.backtest_metrics.get('IC', 0):.4f}, "
                f"Sharpe: {strategy.backtest_metrics.get('sharpe_ratio', 0):.2f}\n"
            )
            if hypothesis:
                prompt += f"Hypothesis: {hypothesis}\n"

            prompt += (
                "\nParagraph 1: What this strategy does and why it works.\n"
                "Paragraph 2: Key risks and how they are managed.\n"
                "Paragraph 3: When this strategy performs best and worst."
            )

            response = self._router.route_and_call(
                stage=PipelineStage.ANALYSIS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.get("content", strategy.narrative)

        except Exception as e:
            logger.warning(f"[Translator] LLM enhancement failed: {e}")
            return strategy.narrative


# ──────── Batch Translation ────────

def translate_run_results(
    results: list[dict],
    router=None,
) -> list[TradingStrategy]:
    """
    批量翻译一次运行的所有因子。

    Args:
        results: pipeline run 的 results.json 列表
        router: 可选，用于 LLM 增强

    Returns:
        list[TradingStrategy] 所有成功因子的交易策略
    """
    translator = FactorTranslator(router=router)
    strategies = []

    for r in results:
        if r.get("skipped") or not r.get("backtest_metrics"):
            continue
        code = r.get("factor_code", "")
        if not code:
            continue

        strategy = translator.translate(
            factor_code=code,
            backtest_metrics=r["backtest_metrics"],
            hypothesis=r.get("hypothesis", ""),
        )
        strategies.append(strategy)

    logger.info(f"[Translator] Translated {len(strategies)} factors into trading strategies")
    return strategies
