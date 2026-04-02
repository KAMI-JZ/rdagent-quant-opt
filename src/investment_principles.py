"""
Investment Principles — 经典数学投资逻辑

将资深量化投资的核心数学公式内化为可调用工具:
1. Kelly Criterion: 最优仓位比例
2. Mean-Variance Optimization (Markowitz): 最优资产配置
3. Risk Parity: 等风险贡献配置
4. Black-Litterman: 融合市场均衡 + 主观观点
5. Maximum Drawdown Control: 基于回撤的动态仓位

所有函数零 LLM 成本，纯数学计算。
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Kelly Criterion — 最优仓位比例
# 功能: 给定胜率和赔率，计算数学上的最优下注比例
# 影响: 防止过度押注导致爆仓
# ──────────────────────────────────────────────

@dataclass
class KellyResult:
    """Kelly 公式计算结果"""
    full_kelly: float           # 完整 Kelly 比例
    half_kelly: float           # 半 Kelly (实战常用，更保守)
    fraction: float             # 实际推荐比例 (考虑 max_position)
    edge: float                 # 期望优势 = p*b - q
    explanation: str = ""


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    max_position: float = 0.25,
    kelly_fraction: float = 0.5,
) -> KellyResult:
    """
    Kelly Criterion: f* = (bp - q) / b

    Args:
        win_rate: 胜率 p (0-1)
        win_loss_ratio: 盈亏比 b = avg_win / avg_loss
        max_position: 最大仓位上限 (默认25%)
        kelly_fraction: Kelly 缩放系数 (默认0.5 = 半Kelly)

    Returns:
        KellyResult with optimal fraction and explanation
    """
    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = max(0.001, win_loss_ratio)

    # f* = (bp - q) / b
    full_kelly = (b * p - q) / b
    edge = b * p - q

    if full_kelly <= 0:
        return KellyResult(
            full_kelly=full_kelly,
            half_kelly=full_kelly / 2,
            fraction=0.0,
            edge=edge,
            explanation=f"负期望值 (edge={edge:.4f})，Kelly建议不下注。"
            f"胜率{p:.1%}×盈亏比{b:.2f}不足以覆盖亏损。",
        )

    half_kelly = full_kelly * kelly_fraction
    fraction = min(half_kelly, max_position)

    explanation = (
        f"Kelly最优仓位={full_kelly:.1%}，半Kelly={half_kelly:.1%}，"
        f"受max_position={max_position:.0%}约束后={fraction:.1%}。"
        f"期望优势edge={edge:.4f}。"
    )

    return KellyResult(
        full_kelly=full_kelly,
        half_kelly=half_kelly,
        fraction=fraction,
        edge=edge,
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# Mean-Variance Optimization (Markowitz)
# 功能: 给定期望收益和协方差矩阵，计算最优权重
# 影响: 经典资产配置，但对输入敏感 → 实际中常用 Risk Parity 替代
# ──────────────────────────────────────────────

@dataclass
class MVOResult:
    """Mean-Variance Optimization 结果"""
    weights: np.ndarray            # 最优权重向量
    expected_return: float         # 组合期望收益
    expected_volatility: float     # 组合期望波动率
    sharpe_ratio: float            # 组合 Sharpe
    explanation: str = ""


def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    target_return: float | None = None,
) -> MVOResult:
    """
    Markowitz Mean-Variance Optimization (解析解).

    Args:
        expected_returns: 各资产期望收益率 (年化)
        cov_matrix: 协方差矩阵 (年化)
        risk_free_rate: 无风险利率
        target_return: 目标收益率 (None = 最大 Sharpe)

    Returns:
        MVOResult with optimal weights
    """
    n = len(expected_returns)
    mu = np.array(expected_returns, dtype=float)
    sigma = np.array(cov_matrix, dtype=float)

    # 正则化: 防止奇异矩阵
    sigma += np.eye(n) * 1e-8

    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        # 奇异矩阵 → 等权配置
        weights = np.ones(n) / n
        port_ret = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ sigma @ weights))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0
        return MVOResult(
            weights=weights, expected_return=port_ret,
            expected_volatility=port_vol, sharpe_ratio=sharpe,
            explanation="协方差矩阵奇异，退化为等权配置。",
        )

    ones = np.ones(n)

    if target_return is None:
        # 最大 Sharpe: w* ∝ Σ^(-1)(μ - rf)
        excess = mu - risk_free_rate
        raw_weights = inv_sigma @ excess
        weight_sum = raw_weights.sum()
        if abs(weight_sum) < 1e-10:
            weights = np.ones(n) / n
        else:
            weights = raw_weights / weight_sum
    else:
        # 给定目标收益: 拉格朗日求解
        A = float(ones @ inv_sigma @ ones)
        B = float(ones @ inv_sigma @ mu)
        C = float(mu @ inv_sigma @ mu)
        det = A * C - B * B
        if abs(det) < 1e-10:
            weights = np.ones(n) / n
        else:
            lam = (C - B * target_return) / det
            gam = (A * target_return - B) / det
            weights = inv_sigma @ (lam * ones + gam * mu)

    # 归一化确保和为 1
    if abs(weights.sum()) > 1e-10:
        weights = weights / weights.sum()
    else:
        weights = np.ones(n) / n

    port_ret = float(weights @ mu)
    port_vol = float(np.sqrt(max(0, weights @ sigma @ weights)))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    explanation = (
        f"Markowitz最优配置: 期望收益={port_ret:.2%}，"
        f"波动率={port_vol:.2%}，Sharpe={sharpe:.2f}。"
        f"权重范围=[{weights.min():.2%}, {weights.max():.2%}]。"
    )

    return MVOResult(
        weights=weights, expected_return=port_ret,
        expected_volatility=port_vol, sharpe_ratio=sharpe,
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# Risk Parity — 等风险贡献
# 功能: 让每个资产对组合风险的贡献相等
# 影响: 比 MVO 更稳健，不依赖期望收益估计
# ──────────────────────────────────────────────

@dataclass
class RiskParityResult:
    """Risk Parity 结果"""
    weights: np.ndarray
    risk_contributions: np.ndarray  # 各资产风险贡献 (应接近 1/n)
    portfolio_volatility: float
    explanation: str = ""


def risk_parity(
    cov_matrix: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> RiskParityResult:
    """
    Risk Parity: 让每个资产贡献等量风险。
    使用 Spinu (2013) 的迭代算法。

    Args:
        cov_matrix: 协方差矩阵
        max_iterations: 最大迭代次数
        tolerance: 收敛容差

    Returns:
        RiskParityResult with equal-risk-contribution weights
    """
    sigma = np.array(cov_matrix, dtype=float)
    n = sigma.shape[0]
    sigma += np.eye(n) * 1e-8

    # 初始权重: 反波动率加权
    vols = np.sqrt(np.diag(sigma))
    vols = np.maximum(vols, 1e-10)
    weights = (1.0 / vols)
    weights = weights / weights.sum()

    # 迭代求解
    for _ in range(max_iterations):
        port_vol = float(np.sqrt(weights @ sigma @ weights))
        if port_vol < 1e-10:
            break
        # 边际风险贡献
        marginal_risk = sigma @ weights / port_vol
        # 风险贡献
        risk_contrib = weights * marginal_risk
        target_contrib = port_vol / n

        # 更新权重
        new_weights = weights * (target_contrib / np.maximum(risk_contrib, 1e-10))
        new_weights = new_weights / new_weights.sum()

        if np.max(np.abs(new_weights - weights)) < tolerance:
            weights = new_weights
            break
        weights = new_weights

    port_vol = float(np.sqrt(weights @ sigma @ weights))
    if port_vol > 1e-10:
        marginal_risk = sigma @ weights / port_vol
        risk_contrib = weights * marginal_risk
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
    else:
        risk_contrib_pct = np.ones(n) / n

    explanation = (
        f"Risk Parity配置: 组合波动率={port_vol:.2%}，"
        f"各资产风险贡献偏差={np.std(risk_contrib_pct):.4f} "
        f"(越接近0越好，完美等风险=0)。"
    )

    return RiskParityResult(
        weights=weights, risk_contributions=risk_contrib_pct,
        portfolio_volatility=port_vol, explanation=explanation,
    )


# ──────────────────────────────────────────────
# Black-Litterman Model
# 功能: 融合市场隐含均衡收益 + 主观观点 (贝叶斯)
# 影响: 解决 MVO 对输入过于敏感的问题
# ──────────────────────────────────────────────

@dataclass
class BlackLittermanResult:
    """Black-Litterman 结果"""
    weights: np.ndarray
    posterior_returns: np.ndarray    # 贝叶斯后验收益
    prior_returns: np.ndarray       # 先验 (均衡) 收益
    expected_return: float
    expected_volatility: float
    explanation: str = ""


def black_litterman(
    market_weights: np.ndarray,
    cov_matrix: np.ndarray,
    views: list[dict],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.02,
) -> BlackLittermanResult:
    """
    Black-Litterman Model.

    Args:
        market_weights: 市场均衡权重 (如市值加权)
        cov_matrix: 协方差矩阵
        views: 主观观点列表, 每个 dict 包含:
            - "asset": int (资产索引)
            - "return": float (观点收益率)
            - "confidence": float (观点置信度 0-1)
        risk_aversion: 风险厌恶系数 δ
        tau: 不确定性缩放 (通常 0.01-0.1)
        risk_free_rate: 无风险利率

    Returns:
        BlackLittermanResult with posterior weights
    """
    n = len(market_weights)
    w_mkt = np.array(market_weights, dtype=float)
    sigma = np.array(cov_matrix, dtype=float)
    sigma += np.eye(n) * 1e-8

    # 先验: 逆优化得到隐含均衡收益 π = δΣw
    pi = risk_aversion * sigma @ w_mkt

    if not views:
        # 无观点 → 直接返回均衡配置
        port_ret = float(w_mkt @ pi)
        port_vol = float(np.sqrt(w_mkt @ sigma @ w_mkt))
        return BlackLittermanResult(
            weights=w_mkt, posterior_returns=pi, prior_returns=pi,
            expected_return=port_ret, expected_volatility=port_vol,
            explanation="无主观观点，返回市场均衡配置。",
        )

    # 构建 P (观点矩阵) 和 Q (观点收益向量)
    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, view in enumerate(views):
        asset_idx = view["asset"]
        P[i, asset_idx] = 1.0
        Q[i] = view["return"]
        conf = max(0.01, min(1.0, view.get("confidence", 0.5)))
        # Ω = diag(1/confidence * τ * σ_ii) — 置信度越高，Ω越小
        omega_diag[i] = (1.0 / conf - 1.0) * tau * sigma[asset_idx, asset_idx]

    Omega = np.diag(omega_diag)

    # 后验收益: μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)π + P'Ω^(-1)Q]
    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(Omega + np.eye(k) * 1e-10)

    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
    posterior_mean = np.linalg.inv(posterior_precision) @ (
        tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    )

    # 后验权重: w_BL ∝ Σ^(-1) μ_BL
    inv_sigma = np.linalg.inv(sigma)
    raw_weights = inv_sigma @ posterior_mean
    if abs(raw_weights.sum()) > 1e-10:
        weights = raw_weights / raw_weights.sum()
    else:
        weights = w_mkt

    port_ret = float(weights @ posterior_mean)
    port_vol = float(np.sqrt(max(0, weights @ sigma @ weights)))

    explanation = (
        f"Black-Litterman融合{k}个观点: 期望收益={port_ret:.2%}，"
        f"波动率={port_vol:.2%}。先验均衡收益范围="
        f"[{pi.min():.2%}, {pi.max():.2%}]，"
        f"后验收益范围=[{posterior_mean.min():.2%}, {posterior_mean.max():.2%}]。"
    )

    return BlackLittermanResult(
        weights=weights, posterior_returns=posterior_mean, prior_returns=pi,
        expected_return=port_ret, expected_volatility=port_vol,
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# Maximum Drawdown Control — 动态仓位缩放
# 功能: 根据当前回撤深度自动缩减仓位，保护本金
# 影响: 回撤越深 → 仓位越小 → 限制进一步亏损
# ──────────────────────────────────────────────

@dataclass
class DrawdownControlResult:
    """回撤控制结果"""
    position_scale: float       # 仓位缩放因子 (0-1)
    current_drawdown: float     # 当前回撤比例 (负值)
    is_halted: bool             # 是否触发交易暂停
    explanation: str = ""


def drawdown_position_control(
    equity_curve: list[float] | np.ndarray,
    max_drawdown_limit: float = 0.20,
    halt_drawdown: float = 0.30,
    scaling_method: str = "linear",
) -> DrawdownControlResult:
    """
    基于最大回撤的动态仓位缩放。

    Args:
        equity_curve: 权益曲线 (时间序列)
        max_drawdown_limit: 开始缩减仓位的回撤阈值
        halt_drawdown: 完全暂停交易的回撤阈值
        scaling_method: 缩放方法 ("linear" 或 "exponential")

    Returns:
        DrawdownControlResult with position scale
    """
    curve = np.array(equity_curve, dtype=float)
    if len(curve) < 2:
        return DrawdownControlResult(
            position_scale=1.0, current_drawdown=0.0, is_halted=False,
            explanation="权益曲线数据不足，维持满仓。",
        )

    peak = np.maximum.accumulate(curve)
    current_peak = peak[-1]
    current_value = curve[-1]

    if current_peak <= 0:
        return DrawdownControlResult(
            position_scale=0.0, current_drawdown=-1.0, is_halted=True,
            explanation="权益归零，暂停交易。",
        )

    current_dd = (current_value - current_peak) / current_peak  # 负值

    if abs(current_dd) >= halt_drawdown:
        return DrawdownControlResult(
            position_scale=0.0, current_drawdown=current_dd, is_halted=True,
            explanation=f"回撤{current_dd:.1%}超过暂停线{-halt_drawdown:.0%}，停止交易。",
        )

    if abs(current_dd) <= max_drawdown_limit:
        return DrawdownControlResult(
            position_scale=1.0, current_drawdown=current_dd, is_halted=False,
            explanation=f"回撤{current_dd:.1%}在安全范围内({-max_drawdown_limit:.0%})，维持满仓。",
        )

    # 缩减区间: max_drawdown_limit < |dd| < halt_drawdown
    dd_ratio = (abs(current_dd) - max_drawdown_limit) / (halt_drawdown - max_drawdown_limit)
    dd_ratio = min(1.0, max(0.0, dd_ratio))

    if scaling_method == "exponential":
        scale = math.exp(-3.0 * dd_ratio)  # 指数衰减，更快缩减
    else:
        scale = 1.0 - dd_ratio  # 线性缩减

    scale = max(0.0, min(1.0, scale))

    explanation = (
        f"回撤{current_dd:.1%}触发仓位缩减 "
        f"({scaling_method}): 仓位缩放至{scale:.0%}。"
        f"安全线={-max_drawdown_limit:.0%}，暂停线={-halt_drawdown:.0%}。"
    )

    return DrawdownControlResult(
        position_scale=scale, current_drawdown=current_dd, is_halted=False,
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# 综合仓位建议器
# 功能: 整合多种原则，给出统一的仓位建议
# ──────────────────────────────────────────────

@dataclass
class PositionAdvice:
    """综合仓位建议"""
    recommended_size: float         # 推荐仓位比例 (0-1)
    kelly_size: float               # Kelly 建议
    drawdown_scale: float           # 回撤缩放因子
    method_used: str                # 使用了哪些方法
    explanation: str = ""


def compute_position_advice(
    win_rate: float = 0.55,
    win_loss_ratio: float = 1.5,
    equity_curve: list[float] | np.ndarray | None = None,
    max_position: float = 0.25,
    max_drawdown_limit: float = 0.20,
) -> PositionAdvice:
    """
    综合仓位建议: Kelly × 回撤控制。

    Args:
        win_rate: 策略胜率
        win_loss_ratio: 盈亏比
        equity_curve: 权益曲线 (可选)
        max_position: 最大仓位
        max_drawdown_limit: 回撤开始缩减的阈值

    Returns:
        PositionAdvice with recommended size
    """
    kelly = kelly_criterion(win_rate, win_loss_ratio, max_position)
    kelly_size = kelly.fraction

    dd_scale = 1.0
    method = "Kelly"

    if equity_curve is not None and len(equity_curve) >= 2:
        dd = drawdown_position_control(equity_curve, max_drawdown_limit)
        dd_scale = dd.position_scale
        method = "Kelly × DrawdownControl"
        if dd.is_halted:
            return PositionAdvice(
                recommended_size=0.0, kelly_size=kelly_size,
                drawdown_scale=0.0, method_used=method,
                explanation=f"交易暂停: {dd.explanation}",
            )

    recommended = kelly_size * dd_scale

    explanation = (
        f"Kelly建议{kelly_size:.1%} × 回撤缩放{dd_scale:.0%} = {recommended:.1%}。"
        f"{kelly.explanation}"
    )

    return PositionAdvice(
        recommended_size=recommended, kelly_size=kelly_size,
        drawdown_scale=dd_scale, method_used=method,
        explanation=explanation,
    )
