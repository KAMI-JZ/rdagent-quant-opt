"""
Qlib Backtester — 真实因子回测引擎

Executes factor code against real market data via Qlib, returning genuine
IC, ICIR, Sharpe, and MaxDrawdown metrics.

真实回测引擎：接收因子代码 → 在 Qlib 上运行 → 返回真实指标。
替换 pipeline._run_backtest() 中的占位符数据。

Design:
- Writes factor code to a temp file and executes it in an isolated namespace
- Computes factor values across the SP500 universe
- Calculates IC (rank correlation with next-day returns) per day
- Aggregates to IC, ICIR, annualized return, max drawdown, Sharpe ratio
- Falls back gracefully if factor code fails (returns NaN metrics)
"""

import logging
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_validator import DataValidator
from src.survivorship_bias import SurvivorshipBiasCorrector

logger = logging.getLogger(__name__)

# Qlib 数据范围（由 get_data 下载决定）
DEFAULT_TRAIN_START = "2015-01-01"
DEFAULT_TRAIN_END = "2020-06-01"
DEFAULT_EVAL_START = "2019-01-01"
DEFAULT_EVAL_END = "2020-06-01"


@dataclass
class BacktestConfig:
    """回测配置"""
    qlib_data_path: str = ""  # 自动检测: ~/.qlib/qlib_data/us_data
    instrument: str = "sp500"
    eval_start: str = DEFAULT_EVAL_START
    eval_end: str = DEFAULT_EVAL_END
    ic_method: str = "rank"  # "rank" (Spearman) or "normal" (Pearson)
    forward_days: int = 1  # 预测 N 天后收益
    timeout_seconds: int = 60
    max_stocks: int = 100  # 限制股票数，加速回测


@dataclass
class BacktestResult:
    """回测结果"""
    IC: float = float("nan")
    ICIR: float = float("nan")
    annual_return: float = float("nan")
    max_drawdown: float = float("nan")
    sharpe_ratio: float = float("nan")
    ic_series: list = field(default_factory=list)  # 每日 IC 序列
    n_stocks: int = 0
    n_days: int = 0
    error: Optional[str] = None
    _simulated: bool = False  # 标记: 这是真实数据

    def to_dict(self) -> dict:
        return {
            "IC": round(self.IC, 6) if not np.isnan(self.IC) else 0.0,
            "ICIR": round(self.ICIR, 6) if not np.isnan(self.ICIR) else 0.0,
            "annual_return": round(self.annual_return, 6) if not np.isnan(self.annual_return) else 0.0,
            "max_drawdown": round(self.max_drawdown, 6) if not np.isnan(self.max_drawdown) else 0.0,
            "sharpe_ratio": round(self.sharpe_ratio, 6) if not np.isnan(self.sharpe_ratio) else 0.0,
            "n_stocks": self.n_stocks,
            "n_days": self.n_days,
            "_simulated": self._simulated,
            "error": self.error,
        }


class QlibBacktester:
    """
    Real factor backtester using Qlib data.
    真实因子回测器，使用 Qlib 数据计算 IC 等指标。

    流程:
    1. 从 Qlib 加载 OHLCV 数据
    2. 执行因子代码计算因子值
    3. 计算每日 IC（因子值 vs 未来收益的 rank 相关系数）
    4. 聚合为 IC, ICIR, 年化收益, 最大回撤, Sharpe

    Usage:
        backtester = QlibBacktester(config)
        result = backtester.run(factor_code)
        print(result.IC, result.ICIR)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._qlib_initialized = False
        self._data_cache: Optional[pd.DataFrame] = None
        # 数据质量校验 + 生存者偏差修正
        self._validator = DataValidator()
        self._survivorship_corrector = SurvivorshipBiasCorrector()

    def _ensure_qlib_init(self):
        """初始化 Qlib（仅第一次调用时执行）"""
        if self._qlib_initialized:
            return

        import qlib

        # 自动检测数据路径（优先 Databento 交易所直连数据）
        if not self.config.qlib_data_path:
            databento_path = Path.home() / ".qlib" / "qlib_data" / "us_databento"
            default_path = Path.home() / ".qlib" / "qlib_data" / "us_data"
            if databento_path.exists():
                self.config.qlib_data_path = str(databento_path)
                logger.info("[Backtester] Using Databento exchange-direct data (2020-2026)")
            elif default_path.exists():
                self.config.qlib_data_path = str(default_path)
                logger.info("[Backtester] Using Qlib default data (2000-2020)")
            else:
                raise FileNotFoundError(
                    f"Qlib data not found. Run one of:\n"
                    f"  python scripts/download_databento.py  (recommended, exchange-direct)\n"
                    f"  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us"
                )

        qlib.init(provider_uri=self.config.qlib_data_path)
        self._qlib_initialized = True
        logger.info(f"[Backtester] Qlib initialized with data at {self.config.qlib_data_path}")

    def _load_stock_data(self) -> pd.DataFrame:
        """
        加载 SP500 股票的 OHLCV 数据。
        使用缓存避免重复加载。
        """
        if self._data_cache is not None:
            return self._data_cache

        self._ensure_qlib_init()
        from qlib.data import D

        # 尝试指定的 instrument set，失败则回退到 "all"
        try:
            instruments = D.instruments(self.config.instrument)
            stock_list = D.list_instruments(
                instruments,
                start_time=self.config.eval_start,
                end_time=self.config.eval_end,
            )
        except (ValueError, FileNotFoundError):
            logger.warning(
                f"[Backtester] Instrument set '{self.config.instrument}' not found, "
                f"falling back to 'all'"
            )
            instruments = D.instruments("all")
            stock_list = D.list_instruments(
                instruments,
                start_time=self.config.eval_start,
                end_time=self.config.eval_end,
            )

        # 限制股票数量加速回测
        tickers = sorted(stock_list.keys())[:self.config.max_stocks]

        # 影响: 生存者偏差修正 — 用历史成分股替代当前成分股
        try:
            self._survivorship_corrector.set_base_constituents(
                self.config.eval_end, tickers
            )
            pit_tickers = self._survivorship_corrector.get_point_in_time_constituents(
                self.config.eval_start
            )
            if pit_tickers:
                # 取交集：Qlib 有数据的 + 历史上存在的
                valid_pit = [t for t in pit_tickers if t in stock_list]
                if len(valid_pit) >= 10:
                    logger.info(
                        f"[Backtester] Survivorship correction: {len(tickers)} → "
                        f"{len(valid_pit)} stocks (point-in-time {self.config.eval_start})"
                    )
                    tickers = valid_pit[:self.config.max_stocks]
        except Exception as e:
            logger.warning(f"[Backtester] Survivorship correction skipped: {e}")

        logger.info(f"[Backtester] Loading data for {len(tickers)} stocks "
                     f"({self.config.eval_start} to {self.config.eval_end})")

        df = D.features(
            tickers,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time=self.config.eval_start,
            end_time=self.config.eval_end,
        )

        # 重命名列去掉 $ 前缀
        df.columns = ["open", "high", "low", "close", "volume"]

        # 影响: 数据质量校验 — 对每只股票的 OHLCV 做健康检查
        quality_scores = []
        bad_tickers = []
        for ticker in df.index.get_level_values(0).unique():
            ticker_df = df.loc[ticker].reset_index()
            ticker_df = ticker_df.rename(columns={"datetime": "date"})
            report = self._validator.validate_ohlcv(ticker_df)
            quality_scores.append(report.data_quality_score)
            if report.data_quality_score < 0.5:
                bad_tickers.append(ticker)

        if quality_scores:
            avg_score = sum(quality_scores) / len(quality_scores)
            logger.info(
                f"[Backtester] Data quality: avg={avg_score:.2f}, "
                f"{len(bad_tickers)} bad stocks removed"
            )
            # 移除质量太差的股票
            if bad_tickers:
                df = df.drop(bad_tickers, level=0, errors="ignore")

        self._data_cache = df
        logger.info(f"[Backtester] Loaded {len(df)} rows for {len(tickers)} stocks")
        return df

    def _execute_factor_code(self, factor_code: str, stock_df: pd.DataFrame) -> pd.Series:
        """
        执行因子代码，返回因子值 Series。

        因子代码应定义一个 calculate_factor(df) 函数，
        接收单只股票的 DataFrame (columns: open, high, low, close, volume)，
        返回一个 Series（因子值）。

        如果代码没有 calculate_factor 函数，尝试直接 eval 最后一行表达式。
        """
        # 构建执行命名空间：允许 import numpy/pandas，限制其他危险操作
        import builtins as _builtins
        allowed_builtins = dict(vars(_builtins))
        # 移除文件/网络/系统操作
        for dangerous in ["open", "exec", "eval", "compile", "exit", "quit",
                          "breakpoint", "input", "memoryview"]:
            allowed_builtins.pop(dangerous, None)

        safe_globals = {
            "__builtins__": allowed_builtins,
            "np": np,
            "pd": pd,
        }

        local_ns = {}

        try:
            exec(factor_code, safe_globals, local_ns)
        except Exception as e:
            raise RuntimeError(f"Factor code execution failed: {e}")

        # 查找 calculate_factor 函数
        calc_fn = local_ns.get("calculate_factor")
        if calc_fn is None:
            # 尝试找任何以 calculate/compute/get_factor 开头的函数
            for name, obj in local_ns.items():
                if callable(obj) and any(name.startswith(p) for p in
                                          ["calculate", "compute", "get_factor", "factor"]):
                    calc_fn = obj
                    break

        if calc_fn is None:
            raise RuntimeError(
                "Factor code must define a calculate_factor(df) function. "
                "df has columns: open, high, low, close, volume."
            )

        # 对每只股票计算因子值
        results = {}
        for ticker in stock_df.index.get_level_values(0).unique():
            try:
                ticker_df = stock_df.loc[ticker].copy()
                factor_values = calc_fn(ticker_df)
                if isinstance(factor_values, pd.Series):
                    for dt, val in factor_values.items():
                        results[(ticker, dt)] = val
                elif isinstance(factor_values, (int, float)):
                    # 标量值：最后一天
                    last_date = ticker_df.index[-1]
                    results[(ticker, last_date)] = factor_values
            except Exception:
                # 单只股票失败不中断整体
                continue

        if not results:
            raise RuntimeError("Factor code produced no valid values for any stock")

        idx = pd.MultiIndex.from_tuples(list(results.keys()), names=["instrument", "datetime"])
        return pd.Series(list(results.values()), index=idx, name="factor")

    def _compute_forward_returns(self, stock_df: pd.DataFrame) -> pd.Series:
        """计算 N 日后收益率"""
        results = {}
        n = self.config.forward_days

        for ticker in stock_df.index.get_level_values(0).unique():
            ticker_df = stock_df.loc[ticker]
            fwd_ret = ticker_df["close"].pct_change(n).shift(-n)
            for dt, val in fwd_ret.items():
                if not np.isnan(val):
                    results[(ticker, dt)] = val

        idx = pd.MultiIndex.from_tuples(list(results.keys()), names=["instrument", "datetime"])
        return pd.Series(list(results.values()), index=idx, name="forward_return")

    def _compute_ic_series(self, factor_values: pd.Series,
                            forward_returns: pd.Series) -> pd.Series:
        """
        计算每日 IC（因子值与未来收益的截面相关系数）。

        IC = rank_corr(factor, forward_return) per day
        """
        # 对齐因子值和收益率
        combined = pd.DataFrame({
            "factor": factor_values,
            "return": forward_returns,
        }).dropna()

        if combined.empty:
            return pd.Series(dtype=float)

        # 按日期分组计算截面 rank 相关
        dates = combined.index.get_level_values(1).unique()
        ic_list = []

        for dt in sorted(dates):
            try:
                day_data = combined.xs(dt, level=1)
                if len(day_data) < 5:  # 至少 5 只股票才有意义
                    continue

                if self.config.ic_method == "rank":
                    ic = day_data["factor"].rank().corr(day_data["return"].rank())
                else:
                    ic = day_data["factor"].corr(day_data["return"])

                if not np.isnan(ic):
                    ic_list.append({"date": dt, "IC": ic})
            except Exception:
                continue

        if not ic_list:
            return pd.Series(dtype=float)

        ic_df = pd.DataFrame(ic_list).set_index("date")
        return ic_df["IC"]

    def _compute_long_short_returns(self, factor_values: pd.Series,
                                     stock_df: pd.DataFrame) -> pd.Series:
        """
        计算多空组合日收益（做多因子值最高 20%，做空最低 20%）。
        用于计算年化收益、最大回撤、Sharpe。
        """
        combined = pd.DataFrame({"factor": factor_values})
        dates = combined.index.get_level_values(1).unique()

        daily_returns = []

        for dt in sorted(dates):
            try:
                day_factors = combined.xs(dt, level=1)["factor"].dropna()
                if len(day_factors) < 10:
                    continue

                # 分五组，做多最高组，做空最低组
                n = len(day_factors)
                quintile = n // 5

                sorted_tickers = day_factors.sort_values()
                short_tickers = sorted_tickers.index[:quintile].tolist()
                long_tickers = sorted_tickers.index[-quintile:].tolist()

                # 获取下一天的收益
                next_dates = [d for d in sorted(dates) if d > dt]
                if not next_dates:
                    continue
                next_dt = next_dates[0]

                long_ret = []
                short_ret = []
                for ticker in long_tickers:
                    try:
                        today_close = stock_df.loc[(ticker, dt), "close"]
                        next_close = stock_df.loc[(ticker, next_dt), "close"]
                        long_ret.append(next_close / today_close - 1)
                    except (KeyError, ZeroDivisionError):
                        continue

                for ticker in short_tickers:
                    try:
                        today_close = stock_df.loc[(ticker, dt), "close"]
                        next_close = stock_df.loc[(ticker, next_dt), "close"]
                        short_ret.append(next_close / today_close - 1)
                    except (KeyError, ZeroDivisionError):
                        continue

                if long_ret and short_ret:
                    ls_ret = np.mean(long_ret) - np.mean(short_ret)
                    daily_returns.append({"date": dt, "return": ls_ret})

            except Exception:
                continue

        if not daily_returns:
            return pd.Series(dtype=float)

        ret_df = pd.DataFrame(daily_returns).set_index("date")
        return ret_df["return"]

    def run(self, factor_code: str) -> BacktestResult:
        """
        执行完整回测流程。

        Args:
            factor_code: 因子的 Python 代码，必须定义 calculate_factor(df) 函数

        Returns:
            BacktestResult with real IC, ICIR, Sharpe, etc.
        """
        result = BacktestResult()

        try:
            # 1. 加载数据
            stock_df = self._load_stock_data()

            # 2. 执行因子代码
            logger.info("[Backtester] Executing factor code...")
            factor_values = self._execute_factor_code(factor_code, stock_df)
            result.n_stocks = factor_values.index.get_level_values(0).nunique()

            # 3. 计算前向收益
            forward_returns = self._compute_forward_returns(stock_df)

            # 4. 计算 IC 序列
            ic_series = self._compute_ic_series(factor_values, forward_returns)
            result.n_days = len(ic_series)
            result.ic_series = ic_series.tolist()

            if len(ic_series) == 0:
                result.error = "No valid IC values computed (factor may be constant)"
                logger.warning(f"[Backtester] {result.error}")
                return result

            # 5. 聚合指标
            result.IC = float(ic_series.mean())
            ic_std = float(ic_series.std())
            result.ICIR = result.IC / ic_std if ic_std > 0 else 0.0

            # 6. 多空组合收益
            ls_returns = self._compute_long_short_returns(factor_values, stock_df)
            if len(ls_returns) > 0:
                # 年化收益 (假设 252 交易日)
                mean_daily = float(ls_returns.mean())
                result.annual_return = mean_daily * 252

                # Sharpe (年化)
                std_daily = float(ls_returns.std())
                result.sharpe_ratio = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

                # 最大回撤
                cumulative = (1 + ls_returns).cumprod()
                peak = cumulative.expanding().max()
                drawdown = (cumulative - peak) / peak
                result.max_drawdown = float(drawdown.min())

            logger.info(
                f"[Backtester] Result: IC={result.IC:.4f} ICIR={result.ICIR:.4f} "
                f"Sharpe={result.sharpe_ratio:.4f} MDD={result.max_drawdown:.4f} "
                f"({result.n_stocks} stocks, {result.n_days} days)"
            )

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[Backtester] Backtest failed: {result.error}")
            logger.debug(traceback.format_exc())

        return result


def create_backtest_objective(backtester: QlibBacktester):
    """
    为 ParameterOptimizer 创建目标函数。

    返回一个函数: factor_code -> IC (float)
    可直接传入 ParameterOptimizer(objective_fn=...)

    Args:
        backtester: 已配置的 QlibBacktester 实例

    Returns:
        Callable[[str], float]: 输入因子代码，返回 IC 值
    """
    def objective(factor_code: str) -> float:
        result = backtester.run(factor_code)
        if result.error or np.isnan(result.IC):
            return -999.0  # 失败的因子得分极低
        return result.IC

    return objective
