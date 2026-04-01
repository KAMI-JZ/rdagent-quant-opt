"""
Optimized Pipeline — 完整管道组装

Connects all components into a single execution flow:
Market Regime → Synthesis → Alpha Filter → Implementation → Validation → Bull-Bear Debate

将所有组件串联成完整管道：
市场状态检测 → 因子构思 → 抗衰减过滤 → 代码生成 → 回测验证 → 多空辩论

Each iteration:
1. Detect market regime (rule-based, no LLM cost)
2. Generate hypothesis with regime context (strong/frontier model, 1 call)
3. Filter for originality + complexity (AST analysis, no LLM cost)
4. Implement factor code (efficient model, up to 10 retries)
5. Backtest validation (pure CPU, no LLM cost)
6. Bull-Bear debate analysis (strong model, 3 calls)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .model_router import MultiModelRouter, CostTracker, PipelineStage
from .alpha_filter import AlphaDecayFilter, FilterResult
from .debate_agents import DebateAnalyzer, DebateResult, Verdict
from .market_regime import MarketRegimeDetector, RegimeAwareSynthesis, MarketRegime
from .experience_memory import ExperienceMemory, Experience
from .param_optimizer import ParameterExtractor, ParameterOptimizer
from .trajectory_evolution import TrajectoryBuilder, TrajectoryEvolver
from .report_generator import ReportGenerator, ReportConfig
from .qlib_backtester import QlibBacktester, BacktestConfig, BacktestResult
from .data_provider import HybridProvider
import re as _re

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

@dataclass
class IterationResult:
    """
    Result of a single pipeline iteration.
    单轮迭代的结果。
    """
    iteration: int
    hypothesis: str = ""
    factor_code: str = ""
    filter_result: Optional[FilterResult] = None
    backtest_metrics: dict = field(default_factory=dict)
    debate_result: Optional[DebateResult] = None
    market_regime: Optional[MarketRegime] = None
    skipped: bool = False
    skip_reason: str = ""
    cost_usd: float = 0.0
    duration_sec: float = 0.0

    @property
    def passed_filter(self) -> bool:
        """Did the factor pass the anti-decay filter? 因子是否通过了抗衰减过滤？"""
        return self.filter_result is not None and self.filter_result.passed

    @property
    def verdict(self) -> str:
        """Final debate verdict. 辩论最终裁决"""
        if self.debate_result is None:
            return "N/A"
        return self.debate_result.verdict.value


@dataclass
class PipelineReport:
    """
    Summary of a complete pipeline run.
    完整运行的汇总报告。
    """
    iterations_completed: int = 0
    iterations_skipped: int = 0
    total_cost_usd: float = 0.0
    total_duration_sec: float = 0.0
    best_metrics: dict = field(default_factory=dict)
    results: list[IterationResult] = field(default_factory=list)
    verdicts: dict = field(default_factory=lambda: {
        "continue": 0, "pivot": 0, "neutral": 0,
    })

    def summary(self) -> str:
        """Human-readable summary. 生成可读的汇总文本"""
        lines = [
            f"Pipeline Run Complete",
            f"  Iterations: {self.iterations_completed} completed, {self.iterations_skipped} skipped",
            f"  Total cost: ${self.total_cost_usd:.4f}",
            f"  Duration: {self.total_duration_sec:.0f}s",
            f"  Verdicts: {self.verdicts}",
        ]
        if self.best_metrics:
            lines.append(f"  Best metrics: {self.best_metrics}")
        return "\n".join(lines)


# ──────── Main Pipeline ────────

class OptimizedPipeline:
    """
    Complete quantitative factor R&D pipeline.
    完整的量化因子研发管道。

    Integrates:
    - MultiModelRouter: routes LLM calls to optimal models per stage
    - MarketRegimeDetector: injects market context into hypothesis generation
    - AlphaDecayFilter: prevents factor crowding and overfitting
    - DebateAnalyzer: Bull-Bear adversarial analysis for direction decisions

    Usage:
        pipeline = OptimizedPipeline("configs/default.yaml")
        report = pipeline.run(n_iterations=30)
        print(report.summary())
    """

    def __init__(self, config_path: str = "configs/default.yaml", config: dict = None):
        # 支持直接传入 config dict（用于 run_pipeline.py 覆盖 adaptive_mode）
        self.config = config if config is not None else self._load_config(config_path)

        # 功能: 初始化模型路由器（自动选择最合适的模型）
        adaptive_mode = self.config.get("adaptive_mode", "optimized")
        daily_budget = self.config.get("cost", {}).get("daily_budget_usd", 5.0)
        self.router = MultiModelRouter(
            adaptive_mode=adaptive_mode,
            daily_budget=daily_budget,
        )

        # 功能: 初始化市场状态检测器（纯数学计算，不花钱）
        regime_config = self.config.get("market_regime", {})
        regime_model = regime_config.get("model", "deepseek/deepseek-chat")
        self.regime_detector = MarketRegimeDetector(model=regime_model)
        self.regime_synthesis = RegimeAwareSynthesis(detector=self.regime_detector)
        self.regime_update_interval = regime_config.get("update_interval_iters", 5)

        # 功能: 初始化抗因子衰减过滤器（AST分析，不花钱；对齐检查花一次LLM调用）
        filter_config = self.config.get("alpha_filter", {})
        self.alpha_filter = AlphaDecayFilter(
            similarity_threshold=filter_config.get("similarity_threshold", 0.85),
            max_complexity_depth=filter_config.get("max_complexity_depth", 5),
            min_alignment_score=filter_config.get("min_alignment_score", 0.6),
            alignment_model=filter_config.get("alignment_model", "deepseek/deepseek-chat"),
        )

        # 功能: 初始化多空辩论分析器（每轮3次LLM调用）
        debate_config = self.config.get("debate", {})
        self.debate_analyzer = DebateAnalyzer(
            debate_model=debate_config.get("debate_model", "deepseek/deepseek-reasoner"),
            judge_model=debate_config.get("judge_model", "deepseek/deepseek-chat"),
            max_tokens=debate_config.get("max_tokens", 1024),
        )

        # 功能: 因子代码库（存储已接受的因子，用于相似度对比）
        self.accepted_factors: list[tuple[str, str]] = []  # (name, code)

        # 功能: 辩论反馈（上一轮的裁决和建议，注入下一轮 synthesis prompt）
        # 影响: 让辩论结果真正影响下一轮策略方向，而不是白辩论
        self._last_debate: Optional[DebateResult] = None

        # 功能: 经验记忆库（存储历史迭代的成功/失败经验，零 LLM 成本）
        # 影响: 让 synthesis 模型知道之前哪些方向成功了、哪些失败了
        memory_path = self.config.get("experience_memory", {}).get(
            "filepath", "data/experience.json"
        )
        self.memory = ExperienceMemory(filepath=memory_path)

        # 功能: 参数优化器（零 LLM 成本，本地网格搜索）
        # 影响: 将因子的公式结构和参数值分离，自动搜索最优参数
        self._param_extractor = ParameterExtractor()
        self._param_optimizer: Optional[ParameterOptimizer] = None  # 延迟初始化（需要 backtester）

        # 功能: 轨迹进化器（零 LLM 成本，字符串变异/交叉）
        # 影响: 从成功轨迹生成进化模板，注入 synthesis prompt
        self._trajectory_builder = TrajectoryBuilder()
        self._trajectory_evolver = TrajectoryEvolver(mutation_rate=0.3)
        self._evolution_interval = 5  # 每 5 轮进行一次轨迹进化

        # 功能: 成本追踪器
        self.cost_tracker = CostTracker(daily_budget_usd=daily_budget)

        # 功能: 真实回测引擎（Qlib）
        # 影响: 如果 Qlib 数据存在则使用真实回测，否则回退到占位符
        backtest_config = self.config.get("backtest", {})
        try:
            bt_cfg = BacktestConfig(
                qlib_data_path=backtest_config.get("qlib_data_path", ""),
                instrument=backtest_config.get("instrument", "sp500"),
                eval_start=backtest_config.get("eval_start", "2019-01-01"),
                eval_end=backtest_config.get("eval_end", "2020-06-01"),
                max_stocks=backtest_config.get("max_stocks", 100),
            )
            self._backtester = QlibBacktester(bt_cfg)
            logger.info("[Pipeline] QlibBacktester configured for REAL backtesting")
        except Exception as e:
            self._backtester = None
            logger.warning(f"[Pipeline] QlibBacktester unavailable ({e}), using fallback")

        logger.info(
            f"[Pipeline] Initialized: adaptive_mode={adaptive_mode}, "
            f"budget=${daily_budget}/day"
        )

    @staticmethod
    def _extract_python_code(llm_response: str) -> str:
        """
        Extract pure Python code from LLM response.
        LLM 常返回 markdown 包裹的代码（```python ... ```），此方法提取纯代码。
        """
        # 尝试提取 ```python ... ``` 块
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = _re.findall(pattern, llm_response, _re.DOTALL)
        if matches:
            # 取最长的代码块（通常是主要实现）
            return max(matches, key=len).strip()

        # 如果没有代码块，检查是否整个响应就是代码
        lines = llm_response.strip().split("\n")
        code_lines = [l for l in lines if not l.startswith("#") or l.startswith("# ")]
        if any("def " in l for l in code_lines):
            return llm_response.strip()

        return llm_response.strip()

    def _get_real_market_indicators(self) -> dict:
        """
        从 Qlib 获取真实市场指标（SPY/S&P500），用于 regime detection。
        如果 Qlib 不可用则返回默认值。
        """
        if self._backtester is None:
            return {"returns_20d": 0.0, "volatility_20d": 0.15,
                    "ma_cross": 0.01, "vol_percentile": 0.5}
        try:
            self._backtester._ensure_qlib_init()
            from qlib.data import D
            import numpy as np

            df = D.features(["^gspc"], fields=["$close"],
                            start_time="2019-06-01", end_time="2020-06-01")
            if df.empty:
                # ^gspc 可能不存在，用 SPY 代替
                df = D.features(["SPY"], fields=["$close"],
                                start_time="2019-06-01", end_time="2020-06-01")
            if df.empty:
                return {"returns_20d": 0.0, "volatility_20d": 0.15,
                        "ma_cross": 0.01, "vol_percentile": 0.5}

            close = df["$close"].droplevel(0)
            ret_20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
            vol_20 = float(close.pct_change().rolling(20).std().iloc[-1]) if len(close) > 20 else 0.15
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            ma_cross_val = float((ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]) if len(close) > 60 else 0.0
            vol_pct = float((vol_20 - close.pct_change().std()) / close.pct_change().std()) if close.pct_change().std() > 0 else 0.5
            vol_pct = max(0.0, min(1.0, 0.5 + vol_pct))

            return {"returns_20d": ret_20, "volatility_20d": vol_20,
                    "ma_cross": ma_cross_val, "vol_percentile": vol_pct}
        except Exception as e:
            logger.warning(f"[Pipeline] Qlib market data failed: {e}, trying DataProvider fallback")
            return self._get_market_indicators_from_provider()

    def _get_market_indicators_from_provider(self) -> dict:
        """影响: DataProvider fallback — 当 Qlib 不可用时用 Polygon/Yahoo 获取 SPY 数据"""
        defaults = {"returns_20d": 0.0, "volatility_20d": 0.15,
                    "ma_cross": 0.01, "vol_percentile": 0.5}
        try:
            import numpy as np
            provider = HybridProvider()
            df = provider.get_daily_ohlcv("SPY", "2019-06-01", "2020-06-01")
            if df is None or df.empty:
                return defaults
            close = df["close"]
            ret_20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
            vol_20 = float(close.pct_change().rolling(20).std().iloc[-1]) if len(close) > 20 else 0.15
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            ma_cross_val = float((ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]) if len(close) > 60 else 0.0
            vol_pct = float((vol_20 - close.pct_change().std()) / close.pct_change().std()) if close.pct_change().std() > 0 else 0.5
            vol_pct = max(0.0, min(1.0, 0.5 + vol_pct))
            logger.info("[Pipeline] Market indicators loaded via DataProvider fallback")
            return {"returns_20d": ret_20, "volatility_20d": vol_20,
                    "ma_cross": ma_cross_val, "vol_percentile": vol_pct}
        except Exception as e2:
            logger.warning(f"[Pipeline] DataProvider fallback also failed: {e2}")
            return defaults

    def _load_config(self, path: str) -> dict:
        """Load YAML config file. 加载配置文件"""
        if not os.path.exists(path):
            logger.warning(f"[Pipeline] Config not found: {path}, using defaults")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _run_backtest(self, factor_code: str, iteration: int) -> dict:
        """
        Run backtest for a factor using real Qlib data.
        使用真实 Qlib 数据回测因子。

        Args:
            factor_code: 因子的 Python 代码（须定义 calculate_factor(df) 函数）
            iteration: 当前迭代轮次

        Returns:
            dict with keys: IC, ICIR, annual_return, max_drawdown, sharpe_ratio
        """
        if self._backtester is None:
            # 影响: 未配置 Qlib 时回退到占位符，并发出警告
            logger.warning("[Pipeline] QlibBacktester not configured, using fallback metrics")
            return {
                "IC": 0.0, "ICIR": 0.0, "annual_return": 0.0,
                "max_drawdown": 0.0, "sharpe_ratio": 0.0,
                "_simulated": True,
            }

        logger.info(f"[Pipeline] Running REAL backtest for iteration {iteration}")
        result = self._backtester.run(factor_code)
        return result.to_dict()

    def run_iteration(self, iteration: int) -> IterationResult:
        """
        Execute a single pipeline iteration.
        执行单轮迭代。

        Flow:
        1. 市场状态检测 → 2. 因子构思 → 3. 抗衰减过滤 →
        4. 代码生成 → 5. 回测验证 → 6. 多空辩论
        """
        start_time = time.time()
        result = IterationResult(iteration=iteration)

        logger.info(f"[Pipeline] ═══ Iteration {iteration} ═══")

        # ── Step 1: 市场状态检测（每 N 轮更新一次，纯数学不花钱）──
        if iteration % self.regime_update_interval == 0:
            logger.info("[Pipeline] Step 1: Updating market regime...")
            # 影响: 这里用规则检测（不花钱），LLM检测可选
            # 实际运行时需要传入真实市场数据
            # 功能: 从真实市场数据获取指标（Qlib），或回退到默认值
            indicators = self._get_real_market_indicators()
            result.market_regime = self.regime_detector.detect_from_indicators(
                **indicators
            )
            self.regime_synthesis.update_regime(
                **indicators
            )
            logger.info(f"[Pipeline] Regime: {result.market_regime.label}")

        # ── Step 2: 因子构思（强模型，1次LLM调用）──
        logger.info("[Pipeline] Step 2: Generating hypothesis...")

        # 功能: 构建基础 prompt
        user_content = (
            f"Iteration {iteration}: Generate a novel alpha factor hypothesis "
            f"for the S&P 500. Include the economic rationale and expected signal direction."
        )

        # 功能: 如果上一轮有辩论结果，注入反馈让模型调整方向
        # 影响: PIVOT → 要求换全新方向；CONTINUE → 在上一轮基础上优化
        if self._last_debate is not None:
            verdict = self._last_debate.verdict.value.upper()
            action = self._last_debate.recommended_action
            confidence = self._last_debate.confidence
            if self._last_debate.verdict == Verdict.PIVOT:
                user_content += (
                    f"\n\n[DEBATE FEEDBACK — PIVOT (confidence={confidence:.2f})]\n"
                    f"The previous direction was rejected. Reason: {action}\n"
                    f"You MUST explore a completely different hypothesis category."
                )
            elif self._last_debate.verdict == Verdict.CONTINUE:
                user_content += (
                    f"\n\n[DEBATE FEEDBACK — CONTINUE (confidence={confidence:.2f})]\n"
                    f"The previous direction shows promise. Suggestion: {action}\n"
                    f"Refine and improve the previous approach."
                )
            else:  # NEUTRAL
                user_content += (
                    f"\n\n[DEBATE FEEDBACK — NEUTRAL (confidence={confidence:.2f})]\n"
                    f"No strong signal. Suggestion: {action}\n"
                    f"You may continue or explore a new direction at your discretion."
                )
            logger.info(f"[Pipeline] Injected debate feedback: {verdict}")

        # 功能: 注入经验记忆（零 LLM 成本，纯本地检索）
        # 影响: 让模型知道过去哪些方向成功了、哪些失败了
        memory_context = self.memory.build_context_prompt(user_content, top_k=3)
        if memory_context:
            user_content += f"\n\n{memory_context}"
            logger.info("[Pipeline] Injected experience memory context")

        # 功能: 轨迹进化注入（每 N 轮，零 LLM 成本）
        # 影响: 从成功轨迹变异/交叉出新假设模板，让 LLM 参考
        if iteration > 0 and iteration % self._evolution_interval == 0:
            try:
                all_exp = self.memory.get_all()
                if len(all_exp) >= 3:
                    trajectories = self._trajectory_builder.build(all_exp)
                    evolved = self._trajectory_evolver.evolve(trajectories, n_offspring=3)
                    evo_prompt = self._trajectory_evolver.build_evolution_prompt(evolved)
                    if evo_prompt:
                        user_content += f"\n\n{evo_prompt}"
                        logger.info(f"[Pipeline] Injected {len(evolved)} evolved hypotheses")
            except Exception as e:
                logger.warning(f"[Pipeline] Trajectory evolution failed (non-fatal): {e}")

        synthesis_messages = [
            {"role": "system", "content": "You are a quantitative researcher."},
            {"role": "user", "content": user_content},
        ]
        # 功能: 注入市场状态上下文，让模型知道当前是什么市场环境
        synthesis_messages = self.regime_synthesis.augment_prompt(synthesis_messages)

        try:
            synthesis_result = self.router.route(synthesis_messages)
            result.hypothesis = synthesis_result.get("content", "")
            logger.info(f"[Pipeline] Hypothesis generated ({len(result.hypothesis)} chars)")
        except Exception as e:
            logger.error(f"[Pipeline] Synthesis failed: {e}")
            result.skipped = True
            result.skip_reason = f"Synthesis failed: {e}"
            result.duration_sec = time.time() - start_time
            return result

        # ── Step 3: 抗衰减过滤（AST分析不花钱，对齐检查花一次LLM）──
        logger.info("[Pipeline] Step 3: Running alpha decay filter...")
        # 影响: 在实际集成中，factor_code 由 Step 4 生成
        # 这里先用假设文本做初步过滤（跳过对齐检查，因为还没有代码）
        # 真正的过滤在 Step 4 生成代码后进行

        # ── Step 4: 代码生成（便宜模型，有重试机制）──
        logger.info("[Pipeline] Step 4: Implementing factor code...")
        impl_messages = [
            {"role": "system", "content": (
                "You are a Python quant developer. "
                "You MUST output ONLY a Python function named `calculate_factor(df)`. "
                "No explanation, no markdown. Just the code.\n\n"
                "The function signature:\n"
                "  def calculate_factor(df: pd.DataFrame) -> pd.Series:\n"
                "      ...\n\n"
                "Input `df` has columns: open, high, low, close, volume (float64).\n"
                "Index is DatetimeIndex (trading days for one stock).\n"
                "Return a pd.Series of factor values with the same index.\n"
                "You may import numpy as np and pandas as pd at the top."
            )},
            {"role": "user", "content": (
                f"Implement this factor hypothesis:\n\n"
                f"{result.hypothesis}\n\n"
                f"Output ONLY the Python code. No markdown, no explanation."
            )},
        ]

        try:
            impl_result = self.router.route(impl_messages)
            raw_code = impl_result.get("content", "")
            # 功能: 提取纯 Python 代码（去除 LLM 返回的 markdown 包裹）
            result.factor_code = self._extract_python_code(raw_code)
            logger.info(f"[Pipeline] Code generated ({len(result.factor_code)} chars)")
        except Exception as e:
            logger.error(f"[Pipeline] Implementation failed: {e}")
            result.skipped = True
            result.skip_reason = f"Implementation failed: {e}"
            result.duration_sec = time.time() - start_time
            return result

        # ── Step 3b: 过滤生成的代码（AST检查 + 对齐检查）──
        filter_enabled = self.config.get("alpha_filter", {}).get("enabled", True)
        if filter_enabled:
            # 影响: 设置当前迭代数，让双曲衰减知道因子的"年龄"
            self.alpha_filter.set_iteration(iteration)
            result.filter_result = self.alpha_filter.evaluate(
                code=result.factor_code,
                hypothesis=result.hypothesis,
                check_alignment=False,  # 影响: 跳过LLM对齐检查以节省成本，AST检查就够了
            )
            if not result.filter_result.passed:
                logger.info(
                    f"[Pipeline] Factor rejected: {result.filter_result.rejection_reasons}"
                )
                # 影响: 记录失败经验，避免后续轮次重复同类错误
                self.memory.add(Experience(
                    iteration=iteration,
                    hypothesis=result.hypothesis,
                    factor_code=result.factor_code,
                    outcome="filtered",
                    lesson="; ".join(result.filter_result.rejection_reasons),
                ))
                result.skipped = True
                result.skip_reason = "; ".join(result.filter_result.rejection_reasons)
                result.duration_sec = time.time() - start_time
                return result
            logger.info("[Pipeline] Factor passed filter")

        # ── Step 5: 回测验证（纯CPU，不调LLM）──
        logger.info("[Pipeline] Step 5: Backtesting...")
        result.backtest_metrics = self._run_backtest(result.factor_code, iteration)

        # ── Step 5b: 参数优化（零 LLM 成本，本地网格搜索）──
        # 功能: 提取因子代码中的可调参数，网格搜索最优值
        # 影响: 可将 IC 提升 5-30%（取决于因子类型），零额外 LLM 成本
        try:
            params = self._param_extractor.extract(result.factor_code)
            if params and self._backtester is not None:
                logger.info(f"[Pipeline] Step 5b: Optimizing {len(params)} parameters...")
                # 延迟初始化参数优化器（需要 backtester 作为目标函数）
                if self._param_optimizer is None:
                    from .qlib_backtester import create_backtest_objective
                    self._param_optimizer = ParameterOptimizer(
                        objective_fn=create_backtest_objective(self._backtester),
                        max_evals=30,  # 限制搜索次数，平衡速度和精度
                    )
                opt_result = self._param_optimizer.optimize(result.factor_code, params)
                if opt_result.optimized_score > opt_result.original_score:
                    logger.info(
                        f"[Pipeline] Parameter optimization improved IC: "
                        f"{opt_result.original_score:.4f} → {opt_result.optimized_score:.4f} "
                        f"({opt_result.improvement_pct:+.1f}%)"
                    )
                    result.factor_code = opt_result.optimized_code
                    result.backtest_metrics = self._run_backtest(result.factor_code, iteration)
                    result.backtest_metrics["_param_optimized"] = True
                    result.backtest_metrics["_param_improvement_pct"] = opt_result.improvement_pct
                else:
                    logger.info("[Pipeline] Parameter optimization found no improvement")
            elif params:
                logger.info(f"[Pipeline] Found {len(params)} params but no backtester for optimization")
        except Exception as e:
            logger.warning(f"[Pipeline] Parameter optimization failed (non-fatal): {e}")

        # ── Step 6: 多空辩论（强模型，3次LLM调用）──
        debate_enabled = self.config.get("debate", {}).get("enabled", True)
        if debate_enabled:
            logger.info("[Pipeline] Step 6: Bull-Bear debate...")
            try:
                result.debate_result = self.debate_analyzer.debate(
                    hypothesis=result.hypothesis,
                    code=result.factor_code,
                    metrics=result.backtest_metrics,
                    iteration=iteration,
                )
                logger.info(
                    f"[Pipeline] Verdict: {result.debate_result.verdict.value} "
                    f"(confidence={result.debate_result.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(f"[Pipeline] Debate failed (non-fatal): {e}")

        # 功能: 保存辩论结果，供下一轮 synthesis 使用
        # 影响: 这是辩论反馈闭环的关键一步
        if result.debate_result is not None:
            self._last_debate = result.debate_result

        # ── 记录已接受的因子 ──
        factor_name = f"factor_iter{iteration}"
        self.accepted_factors.append((factor_name, result.factor_code))
        self.alpha_filter.add_existing_factor(factor_name, result.factor_code)

        # ── 记录经验到记忆库（零 LLM 成本）──
        # 影响: 后续迭代的 synthesis 可以检索到这轮的经验
        self.memory.add(Experience(
            iteration=iteration,
            hypothesis=result.hypothesis,
            factor_code=result.factor_code,
            metrics=result.backtest_metrics,
            verdict=result.verdict,
            outcome="success",
            lesson=(
                result.debate_result.recommended_action
                if result.debate_result else ""
            ),
        ))

        result.cost_usd = self.cost_tracker.get_today_cost()
        result.duration_sec = time.time() - start_time
        logger.info(
            f"[Pipeline] Iteration {iteration} complete: "
            f"{result.duration_sec:.1f}s, ${result.cost_usd:.4f} total"
        )
        return result

    def run(self, n_iterations: int = 30) -> PipelineReport:
        """
        Run the complete pipeline for N iterations.
        运行完整管道 N 轮。

        Args:
            n_iterations: Number of iterations to run

        Returns:
            PipelineReport with all results and summary statistics
        """
        report = PipelineReport()
        start_time = time.time()

        logger.info(f"[Pipeline] Starting {n_iterations}-iteration run...")

        for i in range(n_iterations):
            # 功能: 检查预算，超了就停
            if self.cost_tracker.is_over_budget():
                logger.warning(f"[Pipeline] Budget exceeded at iteration {i}, stopping")
                break

            result = self.run_iteration(i)
            report.results.append(result)

            if result.skipped:
                report.iterations_skipped += 1
            else:
                report.iterations_completed += 1

                # 功能: 统计辩论裁决分布
                if result.debate_result:
                    verdict_key = result.debate_result.verdict.value.lower()
                    report.verdicts[verdict_key] = report.verdicts.get(verdict_key, 0) + 1

                # 功能: 跟踪最佳指标
                ic = result.backtest_metrics.get("IC", 0)
                best_ic = report.best_metrics.get("IC", 0)
                if ic > best_ic:
                    report.best_metrics = result.backtest_metrics.copy()
                    report.best_metrics["iteration"] = i

        report.total_duration_sec = time.time() - start_time
        report.total_cost_usd = self.cost_tracker.get_today_cost()

        logger.info(f"[Pipeline] Run complete:\n{report.summary()}")

        # 功能: 自动生成 Markdown 报告（零 LLM 成本，纯数据分析）
        # 影响: 报告保存在 logs/ 目录，方便查看和跨次对比
        try:
            report_config = ReportConfig(
                output_dir=self.config.get("cost", {}).get("log_file", "logs").rsplit("/", 1)[0] or "logs"
            )
            gen = ReportGenerator(report_config)
            report_path = gen.generate(
                report,
                cost_summary=self.cost_tracker.get_summary(),
                run_label=f"{self.config.get('adaptive_mode', 'default')}_{n_iterations}iter",
            )
            logger.info(f"[Pipeline] Report generated: {report_path}")
        except Exception as e:
            logger.warning(f"[Pipeline] Report generation failed (non-fatal): {e}")

        return report

    def get_cost_summary(self) -> dict:
        """Get current cost tracking summary. 获取当前成本汇总"""
        return self.cost_tracker.get_summary()

    def export_results(self, report: PipelineReport, output_dir: str) -> None:
        """
        Export pipeline results to files.
        导出运行结果到文件。

        Creates:
        - results.json: All iteration results
        - cost_log.json: Detailed cost breakdown
        - summary.txt: Human-readable summary
        """
        import json

        os.makedirs(output_dir, exist_ok=True)

        # 汇总文本
        with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(report.summary())

        # 迭代结果（简化版，去掉不能序列化的对象）
        results_data = []
        for r in report.results:
            results_data.append({
                "iteration": r.iteration,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
                "hypothesis_length": len(r.hypothesis),
                "code_length": len(r.factor_code),
                "backtest_metrics": r.backtest_metrics,
                "verdict": r.verdict,
                "cost_usd": r.cost_usd,
                "duration_sec": r.duration_sec,
            })

        with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        # 成本日志
        cost_path = os.path.join(output_dir, "cost_log.json")
        self.cost_tracker.export_log(cost_path)

        logger.info(f"[Pipeline] Results exported to {output_dir}/")
