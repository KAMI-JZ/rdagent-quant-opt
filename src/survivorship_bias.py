"""
Survivorship Bias Correction — 生存者偏差修正

Problem: If you only backtest on stocks that exist TODAY, you miss all the ones
that went bankrupt or were delisted. This makes your backtest results look
better than they really are.

Solution: Use historical S&P 500 membership records to know exactly which stocks
were in the index on any given date — including ones that later failed.

问题：只用"现在还活着"的股票做回测，会跳过倒闭/退市的股票，导致结果虚高。
方案：用标普500历史成分股变更记录，还原任意日期的真实股票池。

Data source: https://github.com/fja05680/sp500 (public, free)
"""

import csv
import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

@dataclass
class ConstituentChange:
    """A single S&P 500 membership change. 一条成分股变更记录"""
    date: str            # "YYYY-MM-DD"
    ticker: str          # e.g. "TSLA"
    action: str          # "added" or "removed"
    reason: str = ""     # e.g. "Market cap growth" or "Acquired by XYZ"


# ──────── Core Class ────────

class SurvivorshipBiasCorrector:
    """
    Provides point-in-time S&P 500 constituent lists.
    提供任意历史日期的标普500真实成分股列表。

    Data loading priority:
    1. Local CSV file (data/sp500_changes.csv) — fastest, works offline
    2. GitHub download — always up to date

    Usage:
        corrector = SurvivorshipBiasCorrector()
        # 2008年9月15日标普500里有哪些股票？（包括雷曼兄弟）
        tickers = corrector.get_point_in_time_constituents("2008-09-15")
    """

    # 功能: GitHub 上的公开数据源，记录了标普500所有历史成分股变更
    GITHUB_CSV_URL = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
    )

    LOCAL_CSV_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "sp500_changes.csv"
    )

    def __init__(self, csv_path: str | None = None):
        self._changes: list[ConstituentChange] = []
        # 功能: 基准成分股列表 — 从某个已知日期的完整列表开始，往前/往后推算
        self._base_tickers: set[str] = set()
        self._base_date: str = ""

        csv_path = csv_path or self.LOCAL_CSV_PATH
        self._load_data(csv_path)

    def _load_data(self, csv_path: str) -> None:
        """
        Load S&P 500 change history from local file or GitHub.
        加载标普500变更历史：优先本地文件，没有就从GitHub下载。
        """
        # 尝试本地文件
        if os.path.exists(csv_path):
            logger.info(f"[Survivorship] Loading local CSV: {csv_path}")
            self._parse_csv_file(csv_path)
            return

        # 尝试从GitHub下载
        logger.info("[Survivorship] Local CSV not found, downloading from GitHub...")
        try:
            self._download_and_parse()
        except Exception as e:
            logger.error(f"[Survivorship] Failed to download data: {e}")
            logger.warning("[Survivorship] Running without survivorship correction")

    def _parse_csv_file(self, path: str) -> None:
        """Parse the local CSV file. 解析本地CSV文件"""
        with open(path, "r", encoding="utf-8") as f:
            self._parse_csv_content(f.read())

    def _download_and_parse(self) -> None:
        """Download CSV from GitHub and parse it. 从GitHub下载并解析"""
        import requests

        resp = requests.get(self.GITHUB_CSV_URL, timeout=30)
        resp.raise_for_status()
        content = resp.text

        # 功能: 下载成功后保存到本地，下次直接读本地文件
        os.makedirs(os.path.dirname(self.LOCAL_CSV_PATH), exist_ok=True)
        with open(self.LOCAL_CSV_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[Survivorship] Saved to {self.LOCAL_CSV_PATH}")

        self._parse_csv_content(content)

    def _parse_csv_content(self, content: str) -> None:
        """
        Parse the fja05680/sp500 CSV format.
        解析CSV内容。该数据集格式为：date, ticker, action 等列。

        The CSV from fja05680/sp500 has columns:
        date, Added Ticker, Added Security, Removed Ticker, Removed Security, Reason
        """
        reader = csv.DictReader(io.StringIO(content))
        changes = []

        for row in reader:
            date_str = row.get("date", "").strip()
            if not date_str:
                continue

            # 功能: 标准化日期格式
            date_str = self._normalize_date(date_str)
            if not date_str:
                continue

            # 添加的股票
            added = row.get("Added Ticker", "").strip()
            if added:
                changes.append(ConstituentChange(
                    date=date_str,
                    ticker=added,
                    action="added",
                    reason=row.get("Reason", ""),
                ))

            # 移除的股票
            removed = row.get("Removed Ticker", "").strip()
            if removed:
                changes.append(ConstituentChange(
                    date=date_str,
                    ticker=removed,
                    action="removed",
                    reason=row.get("Reason", ""),
                ))

        # 按日期排序
        changes.sort(key=lambda c: c.date)
        self._changes = changes
        logger.info(f"[Survivorship] Loaded {len(changes)} change records")

    def _normalize_date(self, date_str: str) -> str:
        """
        Convert various date formats to YYYY-MM-DD.
        把各种日期格式统一为 YYYY-MM-DD。
        """
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        logger.debug(f"[Survivorship] Cannot parse date: {date_str}")
        return ""

    def set_base_constituents(self, date: str, tickers: list[str]) -> None:
        """
        Set a known-good constituent list as the starting point.
        设置基准成分股列表（通常是当前日期的列表），用作推算起点。

        Args:
            date: The date this list is valid for
            tickers: Complete list of S&P 500 tickers on that date
        """
        self._base_date = date
        self._base_tickers = set(tickers)
        logger.info(f"[Survivorship] Base set: {len(tickers)} tickers as of {date}")

    def get_point_in_time_constituents(self, date: str) -> list[str]:
        """
        Return the S&P 500 constituent list as it existed on a specific date.
        返回指定日期标普500的真实成分股列表。

        How it works:
        1. Start from the base constituent list (e.g. today's list)
        2. Reverse-apply all changes between the base date and the target date
        3. Result: the actual stocks that were in the index on that day

        原理：从基准列表出发，反向应用所有变更记录，还原出目标日期的真实列表。

        Args:
            date: Target date "YYYY-MM-DD"

        Returns:
            List of ticker symbols that were in S&P 500 on that date
        """
        if not self._base_tickers:
            logger.warning("[Survivorship] No base constituents set, returning empty list")
            return []

        current = set(self._base_tickers)

        if date < self._base_date:
            # 功能: 目标日期在基准之前 → 反向回滚变更（added变removed，removed变added）
            relevant = [
                c for c in self._changes
                if date < c.date <= self._base_date
            ]
            for change in reversed(relevant):
                if change.action == "added":
                    current.discard(change.ticker)
                elif change.action == "removed":
                    current.add(change.ticker)

        elif date > self._base_date:
            # 功能: 目标日期在基准之后 → 正向应用变更
            relevant = [
                c for c in self._changes
                if self._base_date < c.date <= date
            ]
            for change in relevant:
                if change.action == "added":
                    current.add(change.ticker)
                elif change.action == "removed":
                    current.discard(change.ticker)

        return sorted(current)

    def build_survivorship_free_universe(
        self, start: str, end: str
    ) -> dict[str, list[str]]:
        """
        Build a complete survivorship-free stock universe for a date range.
        构建一段时间内每个变更日期的真实股票池。

        Instead of returning every single calendar day (which would be huge),
        returns a dict where keys are dates when changes happened.
        To find the universe for any date, use the most recent key <= that date.

        不是返回每天的列表（太大了），而是返回每个变更日的列表。
        查某天的股票池时，找 <= 该日期的最近一条记录即可。

        Args:
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            Dict of {date: [tickers]} for each date when membership changed
        """
        universe: dict[str, list[str]] = {}

        # 起始日的成分股
        universe[start] = self.get_point_in_time_constituents(start)

        # 功能: 找出时间范围内的所有变更日期
        change_dates = sorted(set(
            c.date for c in self._changes
            if start < c.date <= end
        ))

        for change_date in change_dates:
            universe[change_date] = self.get_point_in_time_constituents(change_date)

        logger.info(
            f"[Survivorship] Built universe: {len(universe)} snapshots "
            f"({start}~{end})"
        )
        return universe

    def get_changes_in_range(self, start: str, end: str) -> list[ConstituentChange]:
        """
        Get all constituent changes in a date range.
        获取一段时间内的所有成分股变更记录。

        Useful for understanding what happened during a specific period.
        适合用来了解某段时间发生了什么（比如 2008 金融危机期间有哪些股票被踢出去了）。
        """
        return [
            c for c in self._changes
            if start <= c.date <= end
        ]
