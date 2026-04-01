"""
Unit tests for data quality validator.
测试数据质量校验器。
"""

import pytest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_validator import DataValidator, ValidationReport, adjust_for_splits


# ──────── Helper: Build test DataFrames ────────

def make_clean_df(days: int = 20) -> pd.DataFrame:
    """
    Generate a clean OHLCV DataFrame with realistic prices.
    生成一份干净的测试数据（20天，价格在100左右波动）。
    """
    dates = pd.bdate_range("2024-01-02", periods=days)
    np.random.seed(42)
    base = 100.0
    closes = base + np.cumsum(np.random.randn(days) * 1.5)
    closes = np.maximum(closes, 10.0)  # 防止价格太低

    opens = closes + np.random.randn(days) * 0.5
    # 功能: 确保 high >= max(open, close) 且 low <= min(open, close)
    highs = np.maximum(opens, closes) + abs(np.random.randn(days)) * 2 + 0.01
    lows = np.minimum(opens, closes) - abs(np.random.randn(days)) * 2 - 0.01

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(1_000_000, 50_000_000, days),
    })


# ──────── Clean Data Tests ────────

class TestCleanData:

    def test_clean_data_passes(self):
        """干净数据应该得高分且没有错误。"""
        v = DataValidator()
        report = v.validate_ohlcv(make_clean_df())
        assert report.is_clean
        assert report.data_quality_score >= 0.8

    def test_report_has_row_count(self):
        """报告应包含数据行数。"""
        v = DataValidator()
        df = make_clean_df(30)
        report = v.validate_ohlcv(df)
        assert report.row_count == 30

    def test_report_has_date_range(self):
        """报告应包含时间范围。"""
        v = DataValidator()
        report = v.validate_ohlcv(make_clean_df())
        assert report.date_range[0] != ""
        assert report.date_range[1] != ""


# ──────── Empty / Missing Column Tests ────────

class TestEdgeCases:

    def test_empty_dataframe(self):
        """空数据应报错，得0分。"""
        v = DataValidator()
        report = v.validate_ohlcv(pd.DataFrame())
        assert not report.is_clean
        assert report.data_quality_score == 0.0

    def test_none_dataframe(self):
        """None 数据应报错。"""
        v = DataValidator()
        report = v.validate_ohlcv(None)
        assert not report.is_clean

    def test_missing_columns(self):
        """缺少必要列应报错。"""
        v = DataValidator()
        df = pd.DataFrame({"date": ["2024-01-02"], "close": [100.0]})
        report = v.validate_ohlcv(df)
        assert not report.is_clean
        assert any("Missing columns" in e for e in report.errors)


# ──────── Negative Price Tests ────────

class TestNegativePrices:

    def test_negative_close_detected(self):
        """负价格应被标记为错误。"""
        v = DataValidator()
        df = make_clean_df()
        df.loc[5, "close"] = -10.0
        report = v.validate_ohlcv(df)
        assert any("negative" in e.lower() for e in report.errors)

    def test_nan_price_detected(self):
        """NaN 价格应被标记为错误。"""
        v = DataValidator()
        df = make_clean_df()
        df.loc[3, "open"] = float("nan")
        report = v.validate_ohlcv(df)
        assert any("NaN" in e for e in report.errors)


# ──────── OHLC Consistency Tests ────────

class TestOHLCConsistency:

    def test_high_less_than_low(self):
        """High < Low 是不可能的，应报错。"""
        v = DataValidator()
        df = make_clean_df()
        df.loc[2, "high"] = 50.0
        df.loc[2, "low"] = 200.0
        report = v.validate_ohlcv(df)
        assert any("inconsistency" in e.lower() for e in report.errors)

    def test_high_less_than_open(self):
        """High < Open 也不对。"""
        v = DataValidator()
        df = make_clean_df()
        df.loc[0, "high"] = 1.0
        df.loc[0, "open"] = 100.0
        report = v.validate_ohlcv(df)
        assert any("inconsistency" in e.lower() for e in report.errors)


# ──────── Zero Volume Tests ────────

class TestZeroVolume:

    def test_some_zero_volume_is_warning(self):
        """少量零成交量应该是警告，不是错误。"""
        v = DataValidator()
        df = make_clean_df()
        df.loc[0, "volume"] = 0
        report = v.validate_ohlcv(df)
        assert any("volume" in w.lower() for w in report.warnings)

    def test_many_zero_volume_is_error(self):
        """大量零成交量应该是错误（超过10%）。"""
        v = DataValidator()
        df = make_clean_df(20)
        df.loc[:4, "volume"] = 0  # 5 out of 20 = 25%
        report = v.validate_ohlcv(df)
        assert any("volume" in e.lower() for e in report.errors)


# ──────── Extreme Change Tests ────────

class TestExtremeChanges:

    def test_split_like_drop_detected(self):
        """
        模拟未调整的拆分：价格突然跌75%。
        应该被检测为可能的未调整拆分。
        """
        v = DataValidator()
        df = make_clean_df()
        # 模拟4:1拆分未调整：第10天价格突然变为1/4
        df.loc[10:, "close"] = df.loc[10:, "close"] / 4
        df.loc[10:, "open"] = df.loc[10:, "open"] / 4
        df.loc[10:, "high"] = df.loc[10:, "high"] / 4
        df.loc[10:, "low"] = df.loc[10:, "low"] / 4
        report = v.validate_ohlcv(df)
        has_extreme = (
            any("extreme" in e.lower() for e in report.errors) or
            any("extreme" in w.lower() for w in report.warnings)
        )
        assert has_extreme, "Should detect the split-like price drop"

    def test_normal_volatility_ok(self):
        """正常波动（日涨跌幅<50%）不应触发警告。"""
        v = DataValidator()
        report = v.validate_ohlcv(make_clean_df())
        assert not any("extreme" in e.lower() for e in report.errors)


# ──────── Consecutive Same Close Tests ────────

class TestConsecutiveSameClose:

    def test_stale_prices_detected(self):
        """连续10天相同收盘价应被警告（可能停牌或数据错误）。"""
        v = DataValidator()
        df = make_clean_df(20)
        df.loc[5:14, "close"] = 100.0  # 10 consecutive same price
        report = v.validate_ohlcv(df)
        assert any("stale" in w.lower() for w in report.warnings)

    def test_short_same_close_ok(self):
        """连续2-3天相同价格是正常的，不应报警。"""
        v = DataValidator()
        df = make_clean_df(20)
        df.loc[5:6, "close"] = 100.0  # Only 2 days same
        report = v.validate_ohlcv(df)
        assert not any("stale" in w.lower() for w in report.warnings)


# ──────── Split Adjustment Tests ────────

class TestSplitAdjustment:

    def test_4_to_1_split(self):
        """
        4:1拆分调整：拆分前的价格应该除以4。
        例如：拆分前收盘价400，调整后应变为100。
        """
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "open": [400.0, 400.0, 100.0, 100.0],
            "high": [410.0, 410.0, 105.0, 105.0],
            "low": [390.0, 390.0, 95.0, 95.0],
            "close": [400.0, 400.0, 100.0, 100.0],
            "volume": [1000, 1000, 4000, 4000],
        })
        actions = pd.DataFrame({
            "date": ["2024-01-04"],
            "action_type": ["split"],
            "value": [4.0],
        })
        adjusted = adjust_for_splits(df, actions)
        # 拆分前的价格应该被除以4
        assert adjusted.loc[0, "close"] == pytest.approx(100.0)
        assert adjusted.loc[1, "close"] == pytest.approx(100.0)
        # 拆分后的价格不变
        assert adjusted.loc[2, "close"] == pytest.approx(100.0)
        # 拆分前的成交量应该被乘以4
        assert adjusted.loc[0, "volume"] == 4000

    def test_empty_actions_no_change(self):
        """没有公司行为记录时，数据不变。"""
        df = make_clean_df(5)
        actions = pd.DataFrame(columns=["date", "action_type", "value"])
        adjusted = adjust_for_splits(df, actions)
        pd.testing.assert_frame_equal(adjusted, df)

    def test_ratio_one_no_change(self):
        """拆分比例为1:1时不应有变化。"""
        df = make_clean_df(5)
        actions = pd.DataFrame({
            "date": ["2024-01-04"],
            "action_type": ["split"],
            "value": [1.0],
        })
        adjusted = adjust_for_splits(df, actions)
        pd.testing.assert_frame_equal(adjusted, df)


# ──────── Quality Score Tests ────────

class TestQualityScore:

    def test_perfect_data_high_score(self):
        """干净数据应该得接近1.0的分数。"""
        v = DataValidator()
        report = v.validate_ohlcv(make_clean_df())
        assert report.data_quality_score >= 0.8

    def test_terrible_data_low_score(self):
        """多种问题叠加应该得低分。"""
        v = DataValidator()
        df = make_clean_df(20)
        df.loc[0, "close"] = -100.0       # 负价格
        df.loc[1, "high"] = 1.0           # OHLC 不一致
        df.loc[1, "low"] = 999.0
        df.loc[2:6, "volume"] = 0         # 大量零成交量
        report = v.validate_ohlcv(df)
        assert report.data_quality_score < 0.5

    def test_score_always_between_0_and_1(self):
        """分数应该在0到1之间。"""
        v = DataValidator()
        # 制造极端坏数据
        df = make_clean_df(10)
        df["close"] = -1.0
        df["volume"] = 0
        report = v.validate_ohlcv(df)
        assert 0.0 <= report.data_quality_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
