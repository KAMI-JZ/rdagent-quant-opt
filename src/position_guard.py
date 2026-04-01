"""
Position Guard — 强制保仓 + 风控系统

Enforces position management and risk control rules on trading strategies:
minimum holding periods, stop-loss/take-profit, turnover limits, and slippage.

仓位守卫：在因子信号之上叠加风控约束，防止过度交易和极端损失。

Components:
1. HoldingConstraint: 最小持仓期、最大持仓期
2. RiskLimiter: 止损止盈、移动止损、波动率调仓
3. TurnoverController: 换手率上限、交易成本预估
4. PositionGuard: 综合风控引擎

Usage:
    guard = PositionGuard()
    adjusted = guard.apply(positions, prices, config)
    print(adjusted.turnover, adjusted.violations)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

class ViolationType(Enum):
    """违规类型"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLDING = "max_holding_exceeded"
    MIN_HOLDING = "min_holding_violated"
    TURNOVER_LIMIT = "turnover_limit"
    MAX_POSITION = "max_position_exceeded"
    MAX_SECTOR = "max_sector_exceeded"


@dataclass
class Violation:
    """单条违规记录"""
    type: ViolationType
    ticker: str = ""
    detail: str = ""
    action_taken: str = ""     # 系统自动采取的措施


@dataclass
class GuardConfig:
    """风控配置"""
    # 持仓约束
    min_holding_days: int = 3        # 最少持仓 3 天（防止日内频繁交易）
    max_holding_days: int = 60       # 最长持仓 60 天
    # 止损止盈
    stop_loss_pct: float = -0.08     # 个股止损 -8%
    take_profit_pct: float = 0.20    # 个股止盈 +20%
    trailing_stop_pct: float = 0.0   # 移动止损（0=不启用）
    # 换手率
    max_daily_turnover: float = 0.20  # 每日最大换手率 20%
    # 仓位限制
    max_single_position: float = 0.10 # 单只最大 10%
    max_sector_exposure: float = 0.30 # 单行业最大 30%
    # 交易成本
    commission_bps: float = 5.0       # 佣金 5bps (0.05%)
    slippage_bps: float = 10.0        # 滑点 10bps (0.10%)


@dataclass
class Position:
    """单只股票的持仓信息"""
    ticker: str
    weight: float              # 目标权重 (0.0 - 1.0)
    entry_price: float = 0.0   # 建仓价格
    current_price: float = 0.0 # 当前价格
    holding_days: int = 0      # 已持仓天数
    peak_price: float = 0.0    # 持仓期间最高价（用于移动止损）
    sector: str = ""           # 行业分类

    @property
    def pnl_pct(self) -> float:
        """持仓盈亏百分比"""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def drawdown_from_peak(self) -> float:
        """从最高点的回撤"""
        if self.peak_price <= 0:
            return 0.0
        return (self.current_price - self.peak_price) / self.peak_price


@dataclass
class GuardResult:
    """风控结果"""
    original_positions: list[Position] = field(default_factory=list)
    adjusted_positions: list[Position] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    forced_exits: list[str] = field(default_factory=list)     # 被强制平仓的 ticker
    blocked_entries: list[str] = field(default_factory=list)   # 被阻止进场的 ticker
    estimated_turnover: float = 0.0
    estimated_cost_bps: float = 0.0

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return {
            "n_original": len(self.original_positions),
            "n_adjusted": len(self.adjusted_positions),
            "n_violations": self.n_violations,
            "violations": [{"type": v.type.value, "ticker": v.ticker,
                           "detail": v.detail, "action": v.action_taken}
                          for v in self.violations],
            "forced_exits": self.forced_exits,
            "blocked_entries": self.blocked_entries,
            "estimated_turnover": round(self.estimated_turnover, 4),
            "estimated_cost_bps": round(self.estimated_cost_bps, 2),
        }

    def to_markdown(self) -> str:
        lines = ["# Position Guard Report", ""]
        lines.append(f"- Original positions: {len(self.original_positions)}")
        lines.append(f"- Adjusted positions: {len(self.adjusted_positions)}")
        lines.append(f"- Violations: {self.n_violations}")
        lines.append(f"- Forced exits: {len(self.forced_exits)}")
        lines.append(f"- Blocked entries: {len(self.blocked_entries)}")
        lines.append(f"- Estimated turnover: {self.estimated_turnover:.1%}")
        lines.append(f"- Estimated cost: {self.estimated_cost_bps:.1f} bps")

        if self.violations:
            lines.append("\n## Violations")
            lines.append("| Type | Ticker | Detail | Action |")
            lines.append("|------|--------|--------|--------|")
            for v in self.violations:
                lines.append(f"| {v.type.value} | {v.ticker} | {v.detail} | {v.action_taken} |")

        return "\n".join(lines)


# ──────── Core Guard ────────

class PositionGuard:
    """
    Position management and risk control engine.
    仓位管理和风控引擎。

    Usage:
        guard = PositionGuard(config)
        result = guard.check(positions)
        # result.violations — 所有违规
        # result.adjusted_positions — 调整后的仓位
        # result.forced_exits — 被强制平仓的股票
    """

    def __init__(self, config: Optional[GuardConfig] = None):
        self.config = config or GuardConfig()

    def check(self, positions: list[Position]) -> GuardResult:
        """
        对持仓列表执行全部风控检查。

        Args:
            positions: 当前持仓列表

        Returns:
            GuardResult with violations and adjusted positions
        """
        result = GuardResult(original_positions=list(positions))
        adjusted = list(positions)

        # 影响: 依次执行各项风控检查
        adjusted = self._check_stop_loss(adjusted, result)
        adjusted = self._check_take_profit(adjusted, result)
        adjusted = self._check_trailing_stop(adjusted, result)
        adjusted = self._check_holding_period(adjusted, result)
        adjusted = self._check_position_limits(adjusted, result)
        adjusted = self._check_sector_limits(adjusted, result)

        # 影响: 计算换手率和成本
        result.adjusted_positions = adjusted
        result.estimated_turnover = self._estimate_turnover(positions, adjusted)
        result.estimated_cost_bps = self._estimate_cost(result.estimated_turnover)

        # 影响: 检查换手率上限
        self._check_turnover_limit(result)

        logger.info(
            f"[Guard] {len(positions)} positions → {len(adjusted)} after guard, "
            f"{result.n_violations} violations, turnover={result.estimated_turnover:.1%}"
        )

        return result

    def _check_stop_loss(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """止损检查"""
        kept = []
        for pos in positions:
            if pos.pnl_pct <= self.config.stop_loss_pct and pos.entry_price > 0:
                result.violations.append(Violation(
                    type=ViolationType.STOP_LOSS,
                    ticker=pos.ticker,
                    detail=f"P&L {pos.pnl_pct:.1%} breached stop-loss {self.config.stop_loss_pct:.1%}",
                    action_taken="Forced exit",
                ))
                result.forced_exits.append(pos.ticker)
            else:
                kept.append(pos)
        return kept

    def _check_take_profit(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """止盈检查"""
        kept = []
        for pos in positions:
            if pos.pnl_pct >= self.config.take_profit_pct and pos.entry_price > 0:
                result.violations.append(Violation(
                    type=ViolationType.TAKE_PROFIT,
                    ticker=pos.ticker,
                    detail=f"P&L {pos.pnl_pct:.1%} hit take-profit {self.config.take_profit_pct:.1%}",
                    action_taken="Forced exit (profit taken)",
                ))
                result.forced_exits.append(pos.ticker)
            else:
                kept.append(pos)
        return kept

    def _check_trailing_stop(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """移动止损检查"""
        if self.config.trailing_stop_pct <= 0:
            return positions

        kept = []
        for pos in positions:
            if pos.peak_price > 0 and pos.drawdown_from_peak <= -self.config.trailing_stop_pct:
                result.violations.append(Violation(
                    type=ViolationType.STOP_LOSS,
                    ticker=pos.ticker,
                    detail=f"Trailing stop: {pos.drawdown_from_peak:.1%} from peak",
                    action_taken="Forced exit (trailing stop)",
                ))
                result.forced_exits.append(pos.ticker)
            else:
                kept.append(pos)
        return kept

    def _check_holding_period(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """持仓期限检查"""
        kept = []
        for pos in positions:
            if pos.holding_days > self.config.max_holding_days:
                result.violations.append(Violation(
                    type=ViolationType.MAX_HOLDING,
                    ticker=pos.ticker,
                    detail=f"Held {pos.holding_days} days (max={self.config.max_holding_days})",
                    action_taken="Forced exit (holding limit)",
                ))
                result.forced_exits.append(pos.ticker)
            elif pos.holding_days < self.config.min_holding_days:
                # 不强制平仓，但标记违规（阻止卖出）
                result.violations.append(Violation(
                    type=ViolationType.MIN_HOLDING,
                    ticker=pos.ticker,
                    detail=f"Only held {pos.holding_days} days (min={self.config.min_holding_days})",
                    action_taken="Hold blocked (min holding not met)",
                ))
                kept.append(pos)
            else:
                kept.append(pos)
        return kept

    def _check_position_limits(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """单只仓位上限检查"""
        for pos in positions:
            if pos.weight > self.config.max_single_position:
                old_weight = pos.weight
                pos.weight = self.config.max_single_position
                result.violations.append(Violation(
                    type=ViolationType.MAX_POSITION,
                    ticker=pos.ticker,
                    detail=f"Weight {old_weight:.1%} exceeds max {self.config.max_single_position:.1%}",
                    action_taken=f"Capped to {self.config.max_single_position:.1%}",
                ))
        return positions

    def _check_sector_limits(self, positions: list[Position], result: GuardResult) -> list[Position]:
        """行业集中度检查"""
        sector_weights: dict[str, float] = {}
        for pos in positions:
            if pos.sector:
                sector_weights[pos.sector] = sector_weights.get(pos.sector, 0) + pos.weight

        for sector, total_weight in sector_weights.items():
            if total_weight > self.config.max_sector_exposure:
                result.violations.append(Violation(
                    type=ViolationType.MAX_SECTOR,
                    ticker=sector,
                    detail=f"Sector {sector} weight {total_weight:.1%} exceeds {self.config.max_sector_exposure:.1%}",
                    action_taken="Scale down sector positions proportionally",
                ))
                # 影响: 按比例缩减该行业所有持仓
                scale = self.config.max_sector_exposure / total_weight
                for pos in positions:
                    if pos.sector == sector:
                        pos.weight *= scale

        return positions

    def _estimate_turnover(self, old: list[Position], new: list[Position]) -> float:
        """估算换手率"""
        old_weights = {p.ticker: p.weight for p in old}
        new_weights = {p.ticker: p.weight for p in new}

        all_tickers = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0) - old_weights.get(t, 0))
            for t in all_tickers
        ) / 2  # 单边换手

        return turnover

    def _estimate_cost(self, turnover: float) -> float:
        """估算交易成本 (bps)"""
        return turnover * (self.config.commission_bps + self.config.slippage_bps)

    def _check_turnover_limit(self, result: GuardResult):
        """换手率上限检查"""
        if result.estimated_turnover > self.config.max_daily_turnover:
            result.violations.append(Violation(
                type=ViolationType.TURNOVER_LIMIT,
                ticker="PORTFOLIO",
                detail=f"Turnover {result.estimated_turnover:.1%} exceeds max {self.config.max_daily_turnover:.1%}",
                action_taken="Warning — consider reducing rebalance frequency",
            ))

    def compute_cost_drag(self, annual_turnover: float) -> float:
        """
        计算年化交易成本拖累。

        Args:
            annual_turnover: 年化换手率 (e.g., 12.0 = 每月全部换仓)

        Returns:
            年化成本拖累（百分比）
        """
        cost_per_trade_pct = (self.config.commission_bps + self.config.slippage_bps) / 10000
        return annual_turnover * cost_per_trade_pct * 2  # 双边
