"""
Unit tests for survivorship bias correction.
测试生存者偏差修正模块。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.survivorship_bias import SurvivorshipBiasCorrector, ConstituentChange


# ──────── Helper: Build a corrector with known test data ────────

def make_test_corrector() -> SurvivorshipBiasCorrector:
    """
    Create a corrector with hand-crafted test data (no network needed).
    用手工数据创建测试用的修正器，不需要网络。

    Simulated timeline:
    - Base date: 2020-01-01 with tickers: AAPL, MSFT, GOOG, AMZN, TSLA
    - 2019-12-01: TSLA was added (so before this date, TSLA was NOT in the index)
    - 2015-03-01: XYZ Corp was removed (so before this date, XYZ WAS in the index)
    - 2008-09-15: LEH (Lehman Brothers) was removed (so before this, LEH was in)
    - 2008-01-01: LEH was added (it was in the index during 2008)
    - 2005-06-01: OLD_CO was removed
    """
    corrector = SurvivorshipBiasCorrector.__new__(SurvivorshipBiasCorrector)
    corrector._base_date = "2020-01-01"
    corrector._base_tickers = {"AAPL", "MSFT", "GOOG", "AMZN", "TSLA"}

    corrector._changes = [
        # 2005: OLD_CO removed
        ConstituentChange(date="2005-06-01", ticker="OLD_CO", action="removed", reason="Acquired"),
        # 2008-01: LEH added to index
        ConstituentChange(date="2008-01-01", ticker="LEH", action="added", reason="Market cap"),
        # 2008-09: LEH removed (bankruptcy)
        ConstituentChange(date="2008-09-15", ticker="LEH", action="removed", reason="Bankruptcy"),
        # 2015-03: XYZ removed
        ConstituentChange(date="2015-03-01", ticker="XYZ", action="removed", reason="Delisted"),
        # 2019-12: TSLA added
        ConstituentChange(date="2019-12-01", ticker="TSLA", action="added", reason="Market cap growth"),
    ]

    return corrector


# ──────── Core Tests ────────

class TestPointInTimeConstituents:
    """Test that we get the right stocks for each historical date."""

    def test_base_date_returns_base_set(self):
        """基准日期应返回基准列表本身。"""
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2020-01-01")
        assert set(result) == {"AAPL", "MSFT", "GOOG", "AMZN", "TSLA"}

    def test_lehman_brothers_in_2008_crisis(self):
        """
        2008年9月1日，雷曼兄弟还没倒闭，应该在标普500里。
        (LEH was added 2008-01-01, removed 2008-09-15)
        """
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2008-09-01")
        assert "LEH" in result, "Lehman Brothers should be in S&P 500 before bankruptcy"

    def test_lehman_gone_after_bankruptcy(self):
        """
        2008年10月1日，雷曼兄弟已经倒闭被移除。
        """
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2008-10-01")
        assert "LEH" not in result, "Lehman should be gone after 2008-09-15"

    def test_tesla_not_in_2015(self):
        """
        2015年1月1日，特斯拉还没加入标普500（2019年12月才加入）。
        """
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2015-01-01")
        assert "TSLA" not in result, "Tesla was not added until 2019-12"

    def test_tesla_in_2020(self):
        """
        2020年1月1日，特斯拉已在标普500中。
        """
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2020-01-01")
        assert "TSLA" in result

    def test_xyz_in_2014(self):
        """
        2014年，XYZ 还没被移除（2015-03-01才移除），应该在列表中。
        """
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2014-01-01")
        assert "XYZ" in result, "XYZ was not removed until 2015-03"

    def test_xyz_gone_after_2015(self):
        """2015年4月，XYZ 已被移除。"""
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2015-04-01")
        assert "XYZ" not in result

    def test_stable_tickers_always_present(self):
        """AAPL, MSFT, GOOG 在整个测试期间都应该在列表中。"""
        c = make_test_corrector()
        for date in ["2006-01-01", "2008-09-01", "2015-01-01", "2020-01-01"]:
            result = c.get_point_in_time_constituents(date)
            for ticker in ["AAPL", "MSFT", "GOOG"]:
                assert ticker in result, f"{ticker} should be in list on {date}"


# ──────── Universe Builder Tests ────────

class TestSurvivorshipFreeUniverse:
    """Test the full universe builder."""

    def test_universe_has_start_date(self):
        """生成的股票池应包含起始日期的快照。"""
        c = make_test_corrector()
        universe = c.build_survivorship_free_universe("2007-01-01", "2009-01-01")
        assert "2007-01-01" in universe

    def test_universe_captures_changes(self):
        """股票池应在变更发生的日期产生新快照。"""
        c = make_test_corrector()
        universe = c.build_survivorship_free_universe("2007-01-01", "2009-01-01")
        # LEH added 2008-01-01 and removed 2008-09-15 — both should be snapshot dates
        assert "2008-01-01" in universe
        assert "2008-09-15" in universe

    def test_universe_lehman_transition(self):
        """
        在股票池中验证雷曼兄弟的加入和退出：
        - 2008-01-01 快照应包含 LEH
        - 2008-09-15 快照不应包含 LEH
        """
        c = make_test_corrector()
        universe = c.build_survivorship_free_universe("2007-01-01", "2009-01-01")
        assert "LEH" in universe["2008-01-01"]
        assert "LEH" not in universe["2008-09-15"]


# ──────── Change Query Tests ────────

class TestChangeQueries:
    """Test querying change records."""

    def test_get_changes_in_2008(self):
        """2008年应该有2条变更记录（LEH 加入和移除）。"""
        c = make_test_corrector()
        changes = c.get_changes_in_range("2008-01-01", "2008-12-31")
        assert len(changes) == 2
        tickers = [ch.ticker for ch in changes]
        assert "LEH" in tickers

    def test_empty_range_no_changes(self):
        """没有变更的时间段应返回空列表。"""
        c = make_test_corrector()
        changes = c.get_changes_in_range("2010-01-01", "2010-12-31")
        assert len(changes) == 0


# ──────── Edge Cases ────────

class TestEdgeCases:

    def test_no_base_returns_empty(self):
        """没有设置基准列表时应返回空。"""
        c = SurvivorshipBiasCorrector.__new__(SurvivorshipBiasCorrector)
        c._changes = []
        c._base_tickers = set()
        c._base_date = ""
        result = c.get_point_in_time_constituents("2020-01-01")
        assert result == []

    def test_date_normalization(self):
        """日期格式标准化应支持多种格式。"""
        c = make_test_corrector()
        assert c._normalize_date("2020-01-15") == "2020-01-15"
        assert c._normalize_date("01/15/2020") == "2020-01-15"
        assert c._normalize_date("2020/01/15") == "2020-01-15"

    def test_invalid_date_returns_empty(self):
        """无法解析的日期应返回空字符串。"""
        c = make_test_corrector()
        assert c._normalize_date("not-a-date") == ""

    def test_results_are_sorted(self):
        """返回的股票列表应是字母排序的。"""
        c = make_test_corrector()
        result = c.get_point_in_time_constituents("2020-01-01")
        assert result == sorted(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
