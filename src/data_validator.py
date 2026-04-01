"""
Data Quality Validator — 数据质量校验器

Checks OHLCV market data for common errors before it enters the backtesting
pipeline. Think of it as a "health check" for your data.

检查内容：
1. 缺失交易日（比如某天数据不见了）
2. 价格异常（负价格、Open > High 等现实中不可能的情况）
3. 未调整的拆分（单日涨跌超50%，大概率是数据问题）
4. 停牌/数据冻结（连续多天价格完全相同）
5. 零成交量（可能停牌或数据缺失）

Usage:
    validator = DataValidator()
    report = validator.validate_ohlcv(df)
    if report.data_quality_score < 0.9:
        print("Data quality too low!", report.errors)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

@dataclass
class ValidationReport:
    """
    Result of a data quality check.
    数据质量检查报告。

    Attributes:
        errors: Serious problems that will break backtesting (必须修复的严重问题)
        warnings: Minor issues that may affect accuracy (可能影响准确性的小问题)
        data_quality_score: Overall score 0.0 (terrible) to 1.0 (perfect) (综合质量评分)
        row_count: Total rows checked (检查的数据行数)
        date_range: Tuple of (first_date, last_date) (数据时间范围)
    """
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    row_count: int = 0
    date_range: tuple[str, str] = ("", "")

    @property
    def is_clean(self) -> bool:
        """True if no errors found. 没有严重问题就算干净"""
        return len(self.errors) == 0


# ──────── Core Validator ────────

class DataValidator:
    """
    Validates OHLCV DataFrames for common data quality issues.
    校验 OHLCV 数据的常见质量问题。

    Usage:
        validator = DataValidator()
        report = validator.validate_ohlcv(df)
    """

    # 功能: 可调参数 — 根据不同市场的特性调整阈值
    DEFAULT_MAX_DAILY_CHANGE = 0.50       # 单日涨跌超过50%视为异常
    DEFAULT_MAX_CONSECUTIVE_SAME = 5      # 连续相同收盘价超过5天视为异常
    DEFAULT_MAX_MISSING_PCT = 0.05        # 缺失超过5%视为严重问题

    def __init__(
        self,
        max_daily_change: float = DEFAULT_MAX_DAILY_CHANGE,
        max_consecutive_same: int = DEFAULT_MAX_CONSECUTIVE_SAME,
        max_missing_pct: float = DEFAULT_MAX_MISSING_PCT,
    ):
        self.max_daily_change = max_daily_change
        self.max_consecutive_same = max_consecutive_same
        self.max_missing_pct = max_missing_pct

    def validate_ohlcv(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all checks on an OHLCV DataFrame.
        对 OHLCV 数据运行全部检查。

        Expected columns: date, open, high, low, close, volume

        Args:
            df: DataFrame with OHLCV data

        Returns:
            ValidationReport with errors, warnings, and quality score
        """
        report = ValidationReport()

        # 功能: 基本检查 — 数据是否为空、列是否齐全
        if df is None or df.empty:
            report.errors.append("DataFrame is empty")
            report.data_quality_score = 0.0
            return report

        required = {"date", "open", "high", "low", "close", "volume"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            report.errors.append(f"Missing columns: {missing_cols}")
            report.data_quality_score = 0.0
            return report

        report.row_count = len(df)
        report.date_range = (str(df["date"].iloc[0]), str(df["date"].iloc[-1]))

        # 运行所有检查，每个检查会往 report 里添加 errors/warnings
        penalty = 0.0
        penalty += self._check_negative_prices(df, report)
        penalty += self._check_ohlc_consistency(df, report)
        penalty += self._check_zero_volume(df, report)
        penalty += self._check_missing_dates(df, report)
        penalty += self._check_extreme_changes(df, report)
        penalty += self._check_consecutive_same_close(df, report)

        # 功能: 综合评分 = 1.0 减去各项扣分，最低0分
        report.data_quality_score = max(0.0, 1.0 - penalty)

        level = "CLEAN" if report.is_clean else "ISSUES FOUND"
        logger.info(
            f"[DataValidator] {level}: score={report.data_quality_score:.2f}, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings, "
            f"{report.row_count} rows"
        )
        return report

    # ──────── Individual Checks ────────

    def _check_negative_prices(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check for negative or NaN prices.
        检查负价格和空值 — 这在现实中不可能，说明数据有严重错误。
        """
        penalty = 0.0
        for col in ["open", "high", "low", "close"]:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                report.errors.append(f"{col}: {nan_count} NaN values")
                penalty += 0.1

            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                report.errors.append(f"{col}: {neg_count} negative values")
                penalty += 0.2

        return penalty

    def _check_ohlc_consistency(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check OHLC logical consistency.
        检查 OHLC 逻辑一致性：
        - High 应该 >= Open, Close, Low（最高价应该是当天最高的）
        - Low 应该 <= Open, Close, High（最低价应该是当天最低的）
        """
        penalty = 0.0
        clean = df.dropna(subset=["open", "high", "low", "close"])

        if clean.empty:
            return penalty

        # High 应该是当天最大值
        high_violations = (
            (clean["high"] < clean["open"]) |
            (clean["high"] < clean["close"]) |
            (clean["high"] < clean["low"])
        ).sum()
        if high_violations > 0:
            report.errors.append(
                f"OHLC inconsistency: {high_violations} rows where High < Open/Close/Low"
            )
            penalty += min(0.3, high_violations / len(clean))

        # Low 应该是当天最小值
        low_violations = (
            (clean["low"] > clean["open"]) |
            (clean["low"] > clean["close"]) |
            (clean["low"] > clean["high"])
        ).sum()
        if low_violations > 0:
            report.errors.append(
                f"OHLC inconsistency: {low_violations} rows where Low > Open/Close/High"
            )
            penalty += min(0.3, low_violations / len(clean))

        return penalty

    def _check_zero_volume(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check for zero or negative volume.
        检查零成交量 — 可能是停牌日或数据缺失。少量没问题，太多就有问题了。
        """
        zero_vol = (df["volume"] <= 0).sum()
        if zero_vol == 0:
            return 0.0

        pct = zero_vol / len(df)
        if pct > 0.1:
            report.errors.append(f"Volume: {zero_vol} rows ({pct:.1%}) with zero/negative volume")
            return 0.2
        elif zero_vol > 0:
            report.warnings.append(f"Volume: {zero_vol} rows ({pct:.1%}) with zero volume")
            return 0.05

        return 0.0

    def _check_missing_dates(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check for gaps in trading dates.
        检查缺失的交易日。把日期转换后看有没有跳过的工作日。
        """
        try:
            dates = pd.to_datetime(df["date"])
        except Exception:
            report.warnings.append("Cannot parse dates for gap check")
            return 0.05

        if len(dates) < 2:
            return 0.0

        # 功能: 生成日期范围内的所有工作日（周一到周五），对比实际数据
        expected = pd.bdate_range(start=dates.min(), end=dates.max())
        actual = set(dates.dt.normalize())
        missing = set(expected) - actual
        # 功能: 排除美国主要节假日的近似（周末已排除，剩余缺失可能是节假日）
        # 每年约有 9 个交易所休市日，用这个近似值过滤
        yearly_holidays = 9
        years = max(1, (dates.max() - dates.min()).days / 365)
        expected_holidays = int(years * yearly_holidays)
        real_missing = max(0, len(missing) - expected_holidays)

        if real_missing == 0:
            return 0.0

        missing_pct = real_missing / len(expected)
        if missing_pct > self.max_missing_pct:
            report.errors.append(
                f"Missing dates: ~{real_missing} trading days missing ({missing_pct:.1%})"
            )
            return 0.2
        elif real_missing > 0:
            report.warnings.append(
                f"Missing dates: ~{real_missing} trading days missing ({missing_pct:.1%})"
            )
            return 0.05

        return 0.0

    def _check_extreme_changes(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check for single-day price changes exceeding threshold.
        检查单日涨跌幅超过50%的记录 — 很可能是股票拆分但数据没调整。

        Example: Apple did a 4:1 split in 2020. If the data isn't adjusted,
        you'd see the price drop 75% in one day, which would mess up your factors.
        例如：苹果2020年做了4:1拆分，如果数据没调整，价格会显示单日跌75%。
        """
        clean = df.dropna(subset=["close"])
        if len(clean) < 2:
            return 0.0

        closes = clean["close"].values
        # 功能: 计算每日收益率，跳过第一天（没有前一天对比）
        returns = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)
        extreme = np.abs(returns) > self.max_daily_change
        extreme_count = int(extreme.sum())

        if extreme_count == 0:
            return 0.0

        # 找出具体哪些日期有异常
        extreme_indices = np.where(extreme)[0]
        dates = clean["date"].values
        samples = [
            f"{dates[i+1]}: {returns[i]:+.1%}"
            for i in extreme_indices[:5]  # 最多显示5条
        ]
        sample_str = ", ".join(samples)

        if extreme_count > 3:
            report.errors.append(
                f"Extreme changes: {extreme_count} days with >{self.max_daily_change:.0%} "
                f"change (likely unadjusted splits). Examples: {sample_str}"
            )
            return 0.3
        else:
            report.warnings.append(
                f"Extreme changes: {extreme_count} days with >{self.max_daily_change:.0%} "
                f"change. Examples: {sample_str}"
            )
            return 0.1

        return 0.0

    def _check_consecutive_same_close(self, df: pd.DataFrame, report: ValidationReport) -> float:
        """
        Check for consecutive identical closing prices.
        检查连续多天收盘价完全相同 — 要么停牌，要么数据有误。
        少量可以，太多就不正常了。
        """
        clean = df.dropna(subset=["close"])
        if len(clean) < self.max_consecutive_same:
            return 0.0

        closes = clean["close"].values
        max_run = 1
        current_run = 1

        for i in range(1, len(closes)):
            if closes[i] == closes[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        if max_run > self.max_consecutive_same:
            report.warnings.append(
                f"Stale prices: {max_run} consecutive days with identical close price "
                f"(possible trading halt or data error)"
            )
            return 0.1

        return 0.0


# ──────── Split Adjustment Helper ────────

def adjust_for_splits(
    df: pd.DataFrame, actions: pd.DataFrame
) -> pd.DataFrame:
    """
    Forward-adjust OHLCV prices using corporate action records.
    使用公司行为记录对价格做前复权。

    This is a safety net — most data providers already return adjusted data,
    but if you suspect the adjustments are wrong, you can re-apply them.
    这是一层保险：大部分数据源已经返回调整后的数据，但如果你怀疑调整有误，可以用这个重新调整。

    Args:
        df: OHLCV DataFrame with columns: date, open, high, low, close, volume
        actions: Corporate actions DataFrame with columns: date, action_type, value

    Returns:
        Adjusted DataFrame (copy, original is not modified)
    """
    if df.empty or actions.empty:
        return df.copy()

    result = df.copy()
    splits = actions[actions["action_type"] == "split"].sort_values("date")

    for _, split in splits.iterrows():
        split_date = str(split["date"])
        ratio = float(split["value"])

        if ratio <= 0 or ratio == 1.0:
            continue

        # 功能: 拆分日之前的所有价格都要除以拆分比例
        # 例如 4:1 拆分，之前的价格全部 ÷ 4，这样前后价格可比较
        mask = result["date"] < split_date
        for col in ["open", "high", "low", "close"]:
            result.loc[mask, col] = result.loc[mask, col] / ratio
        # 成交量要反过来，乘以拆分比例
        result.loc[mask, "volume"] = result.loc[mask, "volume"] * ratio

        logger.info(
            f"[DataValidator] Applied {ratio}:1 split on {split_date}, "
            f"adjusted {mask.sum()} rows"
        )

    return result
