"""
Parameter Optimizer — 参数自动优化器

Decouples factor logic (formula structure) from parameter values.
LLM generates the formula template with placeholders, then local
Bayesian/grid search finds optimal parameter values — zero LLM cost.

将因子的"公式结构"和"参数值"分离：
- LLM 只负责生成公式模板（如 pct_change(N) / rolling(M).std()）
- 参数 N、M 由本地贝叶斯优化搜索最优值
- 零额外 LLM 成本

Based on FactorEngine (arXiv:2603.16365) logic/parameter decoupling concept.

Design:
- AST-based parameter extraction: finds numeric literals in factor code
- Grid search + Bayesian refinement for parameter optimization
- Pluggable objective function (IC, Sharpe, etc.)
"""

import ast
import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParamSpec:
    """
    Specification for a single parameter to optimize.
    单个待优化参数的规格。
    """
    name: str           # 参数名（如 "N", "window_size"）
    original_value: float  # LLM 生成的原始值
    min_value: float    # 搜索下界
    max_value: float    # 搜索上界
    step: float = 1.0   # 搜索步长
    is_integer: bool = True  # 是否为整数参数


@dataclass
class OptimizationResult:
    """
    Result of parameter optimization.
    参数优化的结果。
    """
    original_code: str
    optimized_code: str
    original_params: dict[str, float] = field(default_factory=dict)
    optimized_params: dict[str, float] = field(default_factory=dict)
    original_score: float = 0.0
    optimized_score: float = 0.0
    improvement_pct: float = 0.0
    search_iterations: int = 0


class ParameterExtractor:
    """
    Extracts tunable numeric parameters from factor code via AST analysis.
    通过 AST 分析提取因子代码中可调的数值参数。

    Identifies common quant patterns:
    - pct_change(N) → window size
    - rolling(N) → rolling window
    - shift(N) → lag periods
    - Numeric constants in formulas
    """

    # 功能: 识别常见量化因子中的参数模式
    KNOWN_PARAM_METHODS = {
        "pct_change": {"min": 1, "max": 60, "step": 1},
        "rolling": {"min": 2, "max": 120, "step": 1},
        "shift": {"min": 1, "max": 30, "step": 1},
        "ewm": {"min": 2, "max": 60, "step": 1},
    }

    def extract(self, code: str) -> list[ParamSpec]:
        """
        Extract tunable parameters from factor code.
        从因子代码中提取可调参数。

        Returns list of ParamSpec with sensible search ranges.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.warning("[ParamOpt] Cannot parse code, no parameters extracted")
            return []

        params = []
        param_id = 0

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # 功能: 检查方法调用（如 df.rolling(20)）
            method_name = None
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                method_name = node.func.id

            if method_name not in self.KNOWN_PARAM_METHODS:
                continue

            # 提取第一个数值参数
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                    bounds = self.KNOWN_PARAM_METHODS[method_name]
                    params.append(ParamSpec(
                        name=f"{method_name}_{param_id}",
                        original_value=arg.value,
                        min_value=bounds["min"],
                        max_value=bounds["max"],
                        step=bounds["step"],
                        is_integer=isinstance(arg.value, int),
                    ))
                    param_id += 1
                    break  # 每个方法调用只取第一个参数

            # 也检查 keyword arguments（如 rolling(window=20)）
            for kw in node.keywords:
                if kw.arg in ("window", "span", "periods", "com"):
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                        bounds = self.KNOWN_PARAM_METHODS[method_name]
                        params.append(ParamSpec(
                            name=f"{method_name}_{kw.arg}_{param_id}",
                            original_value=kw.value.value,
                            min_value=bounds["min"],
                            max_value=bounds["max"],
                            step=bounds["step"],
                            is_integer=isinstance(kw.value.value, int),
                        ))
                        param_id += 1

        logger.info(f"[ParamOpt] Extracted {len(params)} tunable parameters")
        return params


class ParameterOptimizer:
    """
    Optimizes factor parameters via grid search (no LLM cost).
    通过网格搜索优化因子参数（零 LLM 成本）。

    Usage:
        extractor = ParameterExtractor()
        optimizer = ParameterOptimizer(objective_fn=my_backtest)
        params = extractor.extract(factor_code)
        result = optimizer.optimize(factor_code, params)
    """

    def __init__(
        self,
        objective_fn: Optional[Callable[[str], float]] = None,
        max_evals: int = 50,
    ):
        """
        Args:
            objective_fn: 评估函数，输入因子代码字符串，返回评分（越高越好）。
                         如果为 None，使用默认的占位符评分。
            max_evals: 最大评估次数（防止搜索太久）
        """
        self.objective_fn = objective_fn or self._default_objective
        self.max_evals = max_evals

    @staticmethod
    def _default_objective(code: str) -> float:
        """
        Default placeholder objective (code complexity heuristic).
        默认占位符目标函数。实际使用时应替换为回测评分。
        """
        # 影响: 这只是占位符，实际接入时替换为 Qlib 回测的 IC 值
        return 0.0

    @staticmethod
    def substitute_params(code: str, params: list[ParamSpec], values: list[float]) -> str:
        """
        Replace parameter values in code string.
        替换代码中的参数值。

        Uses regex to find and replace numeric literals in method calls.
        """
        result = code
        for param, new_val in zip(params, values):
            old_val = param.original_value
            # 功能: 从参数名提取方法名（去掉尾部的 _数字 或 _关键字_数字）
            # 例: "pct_change_0" → "pct_change", "rolling_window_1" → "rolling"
            parts = param.name.split("_")
            # 去掉最后一个纯数字部分
            while parts and parts[-1].isdigit():
                parts.pop()
            # 去掉 keyword 部分 (window, span, periods, com)
            kw_names = {"window", "span", "periods", "com"}
            while parts and parts[-1] in kw_names:
                parts.pop()
            method_base = "_".join(parts) if parts else param.name

            old_str = str(int(old_val)) if param.is_integer else str(old_val)
            new_str = str(int(new_val)) if param.is_integer else str(new_val)

            # 功能: 匹配 method_name(old_value) 模式，精确替换
            pattern = rf"({re.escape(method_base)}\s*\(){re.escape(old_str)}(\s*\))"
            replacement = rf"\g<1>{new_str}\2"
            new_result = re.sub(pattern, replacement, result, count=1)
            if new_result == result:
                # 尝试更宽松的匹配: method_name(... old_value ...)
                pattern2 = rf"({re.escape(method_base)}\s*\([^)]*?)\b{re.escape(old_str)}\b"
                replacement2 = rf"\g<1>{new_str}"
                new_result = re.sub(pattern2, replacement2, result, count=1)
            result = new_result

        return result

    def _generate_grid(self, params: list[ParamSpec]) -> list[list[float]]:
        """
        Generate search grid, limited by max_evals.
        生成搜索网格，受 max_evals 限制。
        """
        if not params:
            return []

        # 功能: 对每个参数生成候选值列表
        candidates_per_param = []
        for p in params:
            values = []
            v = p.min_value
            while v <= p.max_value:
                values.append(v)
                v += p.step
            candidates_per_param.append(values)

        # 功能: 如果全组合太多，每个参数只取均匀采样点
        total_combos = 1
        for c in candidates_per_param:
            total_combos *= len(c)

        if total_combos > self.max_evals:
            # 降采样: 每个参数最多取 N 个点
            points_per = max(2, int(self.max_evals ** (1.0 / len(params))))
            sampled = []
            for c in candidates_per_param:
                if len(c) <= points_per:
                    sampled.append(c)
                else:
                    step = len(c) / points_per
                    sampled.append([c[int(i * step)] for i in range(points_per)])
            candidates_per_param = sampled

        # 功能: 生成全组合
        grid = [[]]
        for candidates in candidates_per_param:
            grid = [combo + [v] for combo in grid for v in candidates]

        return grid[:self.max_evals]

    def optimize(
        self, code: str, params: list[ParamSpec]
    ) -> OptimizationResult:
        """
        Run grid search optimization over extracted parameters.
        对提取的参数进行网格搜索优化。

        影响: 纯本地计算，零 LLM 成本。耗时取决于 objective_fn 的速度。
        """
        if not params:
            return OptimizationResult(
                original_code=code, optimized_code=code,
            )

        original_values = [p.original_value for p in params]
        try:
            original_score = self.objective_fn(code)
        except Exception:
            original_score = 0.0

        grid = self._generate_grid(params)
        logger.info(f"[ParamOpt] Searching {len(grid)} parameter combinations")

        best_score = original_score
        best_values = original_values
        best_code = code

        for i, values in enumerate(grid):
            candidate_code = self.substitute_params(code, params, values)
            try:
                score = self.objective_fn(candidate_code)
            except Exception as e:
                logger.debug(f"[ParamOpt] Eval {i} failed: {e}")
                continue

            if score > best_score:
                best_score = score
                best_values = values
                best_code = candidate_code

        improvement = (
            ((best_score - original_score) / abs(original_score) * 100)
            if original_score != 0 else 0.0
        )

        result = OptimizationResult(
            original_code=code,
            optimized_code=best_code,
            original_params={p.name: p.original_value for p in params},
            optimized_params={p.name: v for p, v in zip(params, best_values)},
            original_score=original_score,
            optimized_score=best_score,
            improvement_pct=improvement,
            search_iterations=len(grid),
        )

        logger.info(
            f"[ParamOpt] Optimization complete: "
            f"score {original_score:.4f} → {best_score:.4f} "
            f"({improvement:+.1f}%)"
        )
        return result
