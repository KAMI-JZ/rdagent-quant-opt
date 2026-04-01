"""
Unified Data Provider Layer — 统一数据接口层

Provides a single interface for fetching market data from multiple sources.
支持多数据源切换：Polygon.io（主）→ Yahoo Finance（备用）→ 混合模式（自动回退）

Target market: US S&P 500 (美股标普500)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# ──────── Data Models ────────

@dataclass
class CorporateAction:
    """A single corporate action (split or dividend). 公司行为记录（拆分或分红）"""
    date: str
    ticker: str
    action_type: str          # "split" or "dividend"
    value: float              # split ratio (e.g. 4.0 for 4:1) or dividend amount
    description: str = ""


# ──────── Abstract Base Class ────────

class DataProvider(ABC):
    """
    Abstract base class for all data providers.
    所有数据源必须实现这些接口，保证上层代码不需要关心数据来自哪里。
    """

    @abstractmethod
    def get_daily_ohlcv(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a single ticker.
        获取单只股票的日线数据。

        Args:
            ticker: Stock symbol, e.g. "AAPL"
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            All prices are split/dividend adjusted (前复权).
        """
        ...

    @abstractmethod
    def get_sp500_constituents(self, date: str) -> list[str]:
        """
        Return S&P 500 constituent tickers as of a specific date.
        返回指定日期的标普500成分股列表。
        """
        ...

    @abstractmethod
    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """
        Fetch corporate actions (splits, dividends) for a ticker.
        获取拆分和分红记录。

        Returns:
            DataFrame with columns: date, action_type, value, description
        """
        ...

    @abstractmethod
    def get_delisted_tickers(self, start: str, end: str) -> list[str]:
        """
        Return tickers that were delisted during the given period.
        返回指定时间段内退市的股票列表。
        """
        ...


# ──────── Polygon.io Provider ────────

class PolygonProvider(DataProvider):
    """
    Primary data source using Polygon.io REST API.
    主数据源：Polygon.io（专业级股票数据，免费层可用）

    Free tier limits:
    - 5 API calls per minute → built-in rate limiter (12s between calls)
    - End-of-day data only (no intraday on free tier)

    API key: read from POLYGON_API_KEY environment variable.
    """

    BASE_URL = "https://api.polygon.io"
    # 功能: 免费层限速 — 每次请求后等12秒，保证不超过5次/分钟
    RATE_LIMIT_SECONDS = 12

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY not set. "
                "Get a free key at https://polygon.io/dashboard/signup"
            )
        self._last_call_time: float = 0.0
        logger.info("[DataProvider] PolygonProvider initialized")

    def _rate_limit(self) -> None:
        """Wait if needed to respect API rate limit. 限速器：防止超过免费层调用限制"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            wait = self.RATE_LIMIT_SECONDS - elapsed
            logger.debug(f"[DataProvider] Rate limit: waiting {wait:.1f}s")
            time.sleep(wait)
        self._last_call_time = time.time()

    def _request(self, url: str, params: dict | None = None) -> dict:
        """Make an authenticated GET request to Polygon API. 发送API请求"""
        import requests  # 影响: 延迟导入，只在实际调用时才需要 requests

        params = params or {}
        params["apiKey"] = self.api_key
        self._rate_limit()

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_daily_ohlcv(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Fetch adjusted daily OHLCV from Polygon.io.
        从 Polygon 获取经过拆分/分红调整的日线数据。
        """
        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}"
            f"/range/1/day/{start}/{end}"
        )
        data = self._request(url, {"adjusted": "true", "sort": "asc", "limit": 50000})

        results = data.get("results", [])
        if not results:
            logger.warning(f"[DataProvider] No data from Polygon for {ticker} {start}~{end}")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(results)
        # 功能: Polygon 返回的 t 是毫秒时间戳，转换为日期字符串
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.reset_index(drop=True)

        logger.info(f"[DataProvider] Polygon: {ticker} {len(df)} rows ({start}~{end})")
        return df

    def get_sp500_constituents(self, date: str) -> list[str]:
        """
        Get S&P 500 tickers from Polygon snapshot.
        注意: 免费层只能获取当前成分股，历史成分股需要 survivorship_bias 模块补充。
        """
        url = f"{self.BASE_URL}/v3/reference/tickers"
        data = self._request(url, {
            "market": "stocks",
            "exchange": "XNYS,XNAS",  # NYSE + NASDAQ
            "active": "true",
            "limit": 1000,
        })
        tickers = [r["ticker"] for r in data.get("results", [])]
        logger.info(f"[DataProvider] Polygon: {len(tickers)} active tickers")
        return tickers

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Fetch splits and dividends from Polygon. 获取公司行为数据"""
        actions = []

        # 功能: 获取股票拆分记录
        url = f"{self.BASE_URL}/v3/reference/splits"
        data = self._request(url, {"ticker": ticker, "limit": 100})
        for r in data.get("results", []):
            actions.append({
                "date": r.get("execution_date", ""),
                "action_type": "split",
                "value": r.get("split_to", 1) / max(r.get("split_from", 1), 1),
                "description": f"{r.get('split_from', '?')}:{r.get('split_to', '?')} split",
            })

        # 功能: 获取分红记录
        url = f"{self.BASE_URL}/v3/reference/dividends"
        data = self._request(url, {"ticker": ticker, "limit": 100})
        for r in data.get("results", []):
            actions.append({
                "date": r.get("pay_date", r.get("ex_dividend_date", "")),
                "action_type": "dividend",
                "value": r.get("cash_amount", 0.0),
                "description": f"${r.get('cash_amount', 0):.4f} dividend",
            })

        df = pd.DataFrame(actions, columns=["date", "action_type", "value", "description"])
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"[DataProvider] Polygon: {ticker} {len(df)} corporate actions")
        return df

    def get_delisted_tickers(self, start: str, end: str) -> list[str]:
        """Fetch tickers delisted during the period. 获取退市股票"""
        url = f"{self.BASE_URL}/v3/reference/tickers"
        data = self._request(url, {
            "market": "stocks",
            "active": "false",
            "date.gte": start,
            "date.lte": end,
            "limit": 1000,
        })
        tickers = [r["ticker"] for r in data.get("results", [])]
        logger.info(f"[DataProvider] Polygon: {len(tickers)} delisted tickers ({start}~{end})")
        return tickers


# ──────── Yahoo Finance Fallback ────────

class YahooFallbackProvider(DataProvider):
    """
    Fallback data source using yfinance.
    备用数据源：Yahoo Finance（免费，无需注册，但数据质量较低）

    Limitations:
    - Split/dividend adjustments may be incomplete
    - No survivorship-free constituent history
    - Rate limits are undocumented and may change
    """

    def __init__(self):
        logger.info("[DataProvider] YahooFallbackProvider initialized")

    def get_daily_ohlcv(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch adjusted OHLCV from Yahoo Finance. 从Yahoo获取调整后日线数据"""
        import yfinance as yf  # 影响: 延迟导入，未安装时不会崩溃

        stock = yf.Ticker(ticker)
        # 功能: auto_adjust=True 确保价格已包含拆分和分红调整
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            logger.warning(f"[DataProvider] No data from Yahoo for {ticker} {start}~{end}")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.reset_index(drop=True)

        logger.info(f"[DataProvider] Yahoo: {ticker} {len(df)} rows ({start}~{end})")
        return df

    def get_sp500_constituents(self, date: str) -> list[str]:
        """
        Get current S&P 500 tickers from Wikipedia.
        从维基百科获取当前标普500成分股（注意：只有当前的，没有历史的）。
        """
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"[DataProvider] Yahoo/Wiki: {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"[DataProvider] Failed to fetch S&P 500 list: {e}")
            return []

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Fetch splits/dividends from Yahoo. 从Yahoo获取公司行为"""
        import yfinance as yf

        stock = yf.Ticker(ticker)
        actions = []

        # 功能: 获取拆分记录
        splits = stock.splits
        if splits is not None and not splits.empty:
            for dt, ratio in splits.items():
                actions.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "action_type": "split",
                    "value": float(ratio),
                    "description": f"{ratio}:1 split",
                })

        # 功能: 获取分红记录
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            for dt, amount in dividends.items():
                actions.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "action_type": "dividend",
                    "value": float(amount),
                    "description": f"${amount:.4f} dividend",
                })

        df = pd.DataFrame(actions, columns=["date", "action_type", "value", "description"])
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"[DataProvider] Yahoo: {ticker} {len(df)} corporate actions")
        return df

    def get_delisted_tickers(self, start: str, end: str) -> list[str]:
        """
        Yahoo cannot reliably provide delisted tickers.
        Yahoo 无法可靠地获取退市股票列表，返回空列表。
        需要配合 survivorship_bias 模块使用。
        """
        logger.warning("[DataProvider] Yahoo cannot provide delisted tickers")
        return []


# ──────── Hybrid Provider (Main Entry Point) ────────

class HybridProvider(DataProvider):
    """
    Production data provider: tries Polygon first, falls back to Yahoo.
    生产环境数据源：优先用 Polygon（数据质量高），失败时自动切换到 Yahoo。

    Usage:
        provider = HybridProvider()
        df = provider.get_daily_ohlcv("AAPL", "2023-01-01", "2024-01-01")
    """

    def __init__(self, polygon_api_key: str | None = None):
        self._polygon: PolygonProvider | None = None
        self._yahoo = YahooFallbackProvider()

        # 功能: 如果有 Polygon API key 就初始化主数据源，没有就只用 Yahoo
        api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY", "")
        if api_key:
            try:
                self._polygon = PolygonProvider(api_key=api_key)
                logger.info("[DataProvider] HybridProvider: Polygon + Yahoo ready")
            except ValueError:
                logger.warning("[DataProvider] Polygon key invalid, Yahoo-only mode")
        else:
            logger.info("[DataProvider] HybridProvider: Yahoo-only mode (no Polygon key)")

        self._fallback_count = 0  # 影响: 跟踪回退次数，用于数据质量监控

    def _try_polygon_then_yahoo(
        self, method_name: str, *args, **kwargs
    ):
        """
        Generic fallback logic: try Polygon, on failure try Yahoo.
        通用回退逻辑：先试 Polygon，失败就用 Yahoo。
        """
        if self._polygon is not None:
            try:
                result = getattr(self._polygon, method_name)(*args, **kwargs)
                # 功能: 检查返回值是否为空（DataFrame 或 list）
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                if isinstance(result, list) and len(result) > 0:
                    return result
                logger.info(f"[DataProvider] Polygon returned empty for {method_name}, falling back to Yahoo")
            except Exception as e:
                self._fallback_count += 1
                logger.warning(
                    f"[DataProvider] Polygon failed for {method_name}: {e}. "
                    f"Falling back to Yahoo (fallback #{self._fallback_count})"
                )

        return getattr(self._yahoo, method_name)(*args, **kwargs)

    def get_daily_ohlcv(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        return self._try_polygon_then_yahoo("get_daily_ohlcv", ticker, start, end)

    def get_sp500_constituents(self, date: str) -> list[str]:
        return self._try_polygon_then_yahoo("get_sp500_constituents", date)

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        return self._try_polygon_then_yahoo("get_corporate_actions", ticker)

    def get_delisted_tickers(self, start: str, end: str) -> list[str]:
        return self._try_polygon_then_yahoo("get_delisted_tickers", start, end)

    @property
    def fallback_count(self) -> int:
        """Number of times Yahoo was used as fallback. 回退次数统计"""
        return self._fallback_count
