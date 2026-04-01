#!/usr/bin/env python3
"""
Download S&P 500 OHLCV data from Databento and convert to Qlib format.
从 Databento（交易所直连）下载 S&P 500 日线数据，转为 Qlib 二进制格式。

Databento 日线 OHLCV 数据免费，无需额外付费。
数据源: XNAS.ITCH (NASDAQ) + XNYS.PILLAR (NYSE) 直连交易所数据。

Usage:
    # 下载 2020-2026 数据（默认）
    python scripts/download_databento.py

    # 自定义日期
    python scripts/download_databento.py --start 2018-06-01 --end 2026-03-31

    # 指定输出目录
    python scripts/download_databento.py --output ~/.qlib/qlib_data/us_databento

环境要求:
    - DATABENTO_API_KEY 环境变量已设置
    - pip install databento
"""

import argparse
import logging
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_databento")

# S&P 500 成分股（主要标的，覆盖两大交易所）
# 完整 500 只太多，先下载最活跃的前 100 只作为快速模式
# --full 模式下载全部
SP500_TOP100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "AVGO", "HD", "CVX", "MA", "MRK", "ABBV",
    "LLY", "COST", "PEP", "KO", "ADBE", "WMT", "BAC", "MCD", "CRM", "CSCO",
    "TMO", "ACN", "ABT", "NFLX", "AMD", "LIN", "DHR", "ORCL", "CMCSA", "TXN",
    "PM", "WFC", "NEE", "INTC", "RTX", "DIS", "AMGN", "HON", "UNP", "QCOM",
    "COP", "LOW", "INTU", "BMY", "MS", "SPGI", "GS", "ELV", "BA", "CAT",
    "ISRG", "GE", "PLD", "BLK", "AMAT", "SYK", "MDLZ", "ADP", "GILD", "VRTX",
    "ADI", "DE", "MMC", "REGN", "LRCX", "ETN", "PANW", "KLAC", "SCHW", "CI",
    "SNPS", "CDNS", "MO", "ZTS", "SO", "CME", "BDX", "BSX", "DUK", "FI",
    "SLB", "EOG", "ICE", "PNC", "MCK", "APD", "SHW", "CB", "USB", "AON",
]


def get_sp500_tickers_full():
    """获取完整 S&P 500 成分股列表（从 Wikipedia）"""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = table[0]["Symbol"].tolist()
        # 修正 BRK.B → BRK/B 等特殊 ticker
        tickers = [t.replace(".", "/") if "." in t else t for t in tickers]
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 list: {e}, using top 100")
        return SP500_TOP100


def _download_batch(client, dataset, tickers, start, end):
    """Download a batch of tickers from a single dataset."""
    all_data = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        logger.info(f"  [{dataset}] Batch {batch_num}/{total_batches}: {len(batch)} tickers")

        try:
            data = client.timeseries.get_range(
                dataset=dataset,
                schema="ohlcv-1d",
                stype_in="raw_symbol",
                symbols=batch,
                start=start,
                end=end,
            )
            df = data.to_df()
            if not df.empty:
                all_data.append(df)
                logger.info(f"    Got {len(df)} rows")
        except Exception as e:
            logger.warning(f"    Batch failed ({e}), trying individually...")
            for ticker in batch:
                try:
                    data = client.timeseries.get_range(
                        dataset=dataset,
                        schema="ohlcv-1d",
                        stype_in="raw_symbol",
                        symbols=[ticker],
                        start=start,
                        end=end,
                    )
                    df = data.to_df()
                    if not df.empty:
                        all_data.append(df)
                except Exception:
                    pass  # 静默跳过

    return all_data


def download_from_databento(tickers, start, end, api_key):
    """
    从 Databento 下载日线 OHLCV 数据。

    数据集选择策略:
    - DBEQ.BASIC: 2023-03-28 至今（跨交易所汇总，最简单）
    - XNAS.ITCH: 2018-05-01 至今（NASDAQ 直连，覆盖更早）
    - XNYS.PILLAR: 2018-05-01 至今（NYSE 直连）

    对于 2023 年之前的数据，使用 XNAS + XNYS 组合。
    """
    import databento as db

    client = db.Historical(api_key)
    all_data = []

    logger.info(f"Downloading {len(tickers)} tickers from Databento ({start} to {end})...")

    # DBEQ.BASIC 从 2023-03-28 开始
    dbeq_start = "2023-03-28"

    if start >= dbeq_start:
        # 全部用 DBEQ.BASIC（最简单）
        logger.info("Using DBEQ.BASIC (cross-exchange consolidated)")
        all_data = _download_batch(client, "DBEQ.BASIC", tickers, start, end)
    else:
        # 2023 之前用 XNAS.ITCH（NASDAQ 上市股票覆盖大部分 S&P 500）
        early_end = min(end, dbeq_start)
        logger.info(f"Phase 1: XNAS.ITCH for {start} to {early_end}")
        early_data = _download_batch(client, "XNAS.ITCH", tickers, start, early_end)
        all_data.extend(early_data)

        # 2023 之后用 DBEQ.BASIC
        if end > dbeq_start:
            logger.info(f"Phase 2: DBEQ.BASIC for {dbeq_start} to {end}")
            late_data = _download_batch(client, "DBEQ.BASIC", tickers, dbeq_start, end)
            all_data.extend(late_data)

    if not all_data:
        raise RuntimeError("No data downloaded from Databento")

    combined = pd.concat(all_data)
    logger.info(f"Total: {len(combined)} rows downloaded")
    return combined


def databento_to_qlib_format(df, output_dir):
    """
    将 Databento DataFrame 转为 Qlib 二进制格式。

    Qlib 数据目录结构:
    output_dir/
    ├── calendars/
    │   └── day.txt          # 交易日列表
    ├── instruments/
    │   └── all.txt          # 股票列表 + 有效日期范围
    │   └── sp500.txt        # SP500 成分股
    └── features/
        └── AAPL/
        │   ├── open.day.bin    # 开盘价二进制
        │   ├── high.day.bin
        │   ├── low.day.bin
        │   ├── close.day.bin
        │   └── volume.day.bin
        └── MSFT/
            └── ...
    """
    output_path = Path(output_dir)

    # Databento 的列名处理
    # ohlcv-1d schema: ts_event, open, high, low, close, volume, symbol
    logger.info("Converting to Qlib format...")

    # 提取日期和 symbol
    if "symbol" not in df.columns and df.index.name == "ts_event":
        df = df.reset_index()

    # 处理 ts_event（Databento 用纳秒时间戳）
    if "ts_event" in df.columns:
        df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    elif df.index.name is not None:
        df = df.reset_index()
        date_col = [c for c in df.columns if "time" in str(c).lower() or "date" in str(c).lower() or "ts" in str(c).lower()]
        if date_col:
            df["date"] = pd.to_datetime(df[date_col[0]]).dt.date
        else:
            df["date"] = pd.to_datetime(df.index).date

    # Databento DBEQ.BASIC 每只股票每天有多行（来自不同交易所）
    # 取成交量最大的那行（=主上市交易所数据，最权威）
    df = df.sort_values("volume", ascending=False).drop_duplicates(
        subset=["symbol", "date"], keep="first"
    )
    logger.info(f"After dedup (keep highest volume exchange): {len(df)} rows")

    # 生成交易日历
    all_dates = sorted(df["date"].unique())
    cal_dir = output_path / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "day.txt", "w") as f:
        for d in all_dates:
            f.write(f"{d}\n")
    logger.info(f"Calendar: {len(all_dates)} trading days ({all_dates[0]} to {all_dates[-1]})")

    # 生成股票列表
    inst_dir = output_path / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    tickers = sorted(df["symbol"].unique())
    with open(inst_dir / "all.txt", "w") as f:
        for ticker in tickers:
            ticker_data = df[df["symbol"] == ticker]
            start_date = ticker_data["date"].min()
            end_date = ticker_data["date"].max()
            f.write(f"{ticker}\t{start_date}\t{end_date}\n")

    # SP500 instrument file（与 all 相同）
    with open(inst_dir / "sp500.txt", "w") as f:
        for ticker in tickers:
            ticker_data = df[df["symbol"] == ticker]
            start_date = ticker_data["date"].min()
            end_date = ticker_data["date"].max()
            f.write(f"{ticker}\t{start_date}\t{end_date}\n")

    logger.info(f"Instruments: {len(tickers)} tickers")

    # 生成 features 二进制文件
    feat_dir = output_path / "features"
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)

    # Qlib 需要的字段: open, high, low, close, volume, change, factor
    # change = 日收益率 (close/prev_close - 1)
    # factor = 复权因子 (我们用 Databento adjusted data，设为 1.0)
    for ticker_idx, ticker in enumerate(tickers):
        ticker_dir = feat_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        ticker_data = df[df["symbol"] == ticker].sort_values("date").set_index("date")

        # 构建按日历对齐的数组（Qlib 要求长度 = 日历总天数）
        close_arr = np.full(n_days, np.nan, dtype=np.float32)
        open_arr = np.full(n_days, np.nan, dtype=np.float32)
        high_arr = np.full(n_days, np.nan, dtype=np.float32)
        low_arr = np.full(n_days, np.nan, dtype=np.float32)
        vol_arr = np.full(n_days, np.nan, dtype=np.float32)

        for date, row in ticker_data.iterrows():
            if date in date_to_idx:
                idx = date_to_idx[date]
                close_arr[idx] = float(row["close"]) if pd.notna(row["close"]) else np.nan
                open_arr[idx] = float(row["open"]) if pd.notna(row["open"]) else np.nan
                high_arr[idx] = float(row["high"]) if pd.notna(row["high"]) else np.nan
                low_arr[idx] = float(row["low"]) if pd.notna(row["low"]) else np.nan
                vol_arr[idx] = float(row["volume"]) if pd.notna(row["volume"]) else np.nan

        # 计算 change (日收益率)
        change_arr = np.full(n_days, np.nan, dtype=np.float32)
        for j in range(1, n_days):
            if not np.isnan(close_arr[j]) and not np.isnan(close_arr[j - 1]) and close_arr[j - 1] != 0:
                change_arr[j] = close_arr[j] / close_arr[j - 1] - 1.0

        # factor = 1.0 (Databento 数据已经是调整后价格)
        factor_arr = np.ones(n_days, dtype=np.float32)
        # 没有数据的日期 factor 也设为 NaN
        for j in range(n_days):
            if np.isnan(close_arr[j]):
                factor_arr[j] = np.nan

        # 写入二进制文件
        close_arr.tofile(str(ticker_dir / "close.day.bin"))
        open_arr.tofile(str(ticker_dir / "open.day.bin"))
        high_arr.tofile(str(ticker_dir / "high.day.bin"))
        low_arr.tofile(str(ticker_dir / "low.day.bin"))
        vol_arr.tofile(str(ticker_dir / "volume.day.bin"))
        change_arr.tofile(str(ticker_dir / "change.day.bin"))
        factor_arr.tofile(str(ticker_dir / "factor.day.bin"))

        if (ticker_idx + 1) % 50 == 0 or ticker_idx == len(tickers) - 1:
            logger.info(f"  Written {ticker_idx + 1}/{len(tickers)} tickers")

    logger.info(f"Qlib data written to {output_path}")
    return output_path


def save_as_csv(df, csv_dir):
    """
    将 DataFrame 保存为 Qlib dump_bin.py 期望的格式:
    每只股票一个 CSV 文件，列: date, open, high, low, close, volume, factor, change
    """
    csv_path = Path(csv_dir)
    csv_path.mkdir(parents=True, exist_ok=True)

    # 确保有 date 列
    if "date" not in df.columns:
        if "ts_event" in df.columns:
            df["date"] = pd.to_datetime(df["ts_event"]).dt.date
        elif df.index.name and "ts" in str(df.index.name).lower():
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["ts_event"]).dt.date

    # 去重：每只股票每天只保留成交量最大的那条（主交易所）
    df = df.sort_values("volume", ascending=False).drop_duplicates(
        subset=["symbol", "date"], keep="first"
    )

    tickers = sorted(df["symbol"].unique())
    for ticker in tickers:
        ticker_data = df[df["symbol"] == ticker].sort_values("date").copy()
        ticker_data = ticker_data[["date", "open", "high", "low", "close", "volume"]].copy()

        # 计算 change 和 factor
        ticker_data["change"] = ticker_data["close"].pct_change()
        ticker_data["factor"] = 1.0  # Databento 已经是调整后价格

        # Qlib dump_bin 要求 date 列格式为字符串
        ticker_data["date"] = ticker_data["date"].astype(str)

        # 文件名: 用 Qlib 的 code_to_fname 格式（大写 ticker）
        fname = f"{ticker}.csv"
        ticker_data.to_csv(csv_path / fname, index=False)

    logger.info(f"Saved {len(tickers)} CSV files to {csv_path}")
    return csv_path


def convert_csv_to_qlib(csv_dir, qlib_dir):
    """
    用 Qlib 官方 dump_bin.py 将 CSV 转为 Qlib 二进制格式。
    """
    dump_bin_path = Path("/tmp/qlib_repo/scripts/dump_bin.py")
    if not dump_bin_path.exists():
        raise FileNotFoundError(
            "Qlib dump_bin.py not found. Clone qlib repo first:\n"
            "  git clone https://github.com/microsoft/qlib.git /tmp/qlib_repo"
        )

    import subprocess
    cmd = [
        sys.executable, str(dump_bin_path),
        "dump_all",
        f"--data_path={csv_dir}",
        f"--qlib_dir={qlib_dir}",
        "--freq=day",
        "--date_field_name=date",
        "--file_suffix=.csv",
        "--include_fields=open,high,low,close,volume,change,factor",
    ]
    logger.info(f"Running dump_bin.py: {' '.join(cmd[-5:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"dump_bin.py failed:\n{result.stderr}")
        raise RuntimeError("dump_bin.py conversion failed")

    logger.info(f"Qlib binary data written to {qlib_dir}")
    return Path(qlib_dir)


def verify_qlib_data(data_path):
    """验证生成的 Qlib 数据能被正常加载。"""
    try:
        import qlib
        qlib.init(provider_uri=str(data_path))
        from qlib.data import D

        instruments = D.instruments("all")
        stock_list = D.list_instruments(instruments)
        logger.info(f"Qlib verification: {len(stock_list)} instruments loaded")

        test_ticker = list(stock_list.keys())[0]
        df = D.features(
            [test_ticker],
            fields=["$close", "$volume"],
            start_time="2025-01-01",
            end_time="2026-03-31",
        )
        logger.info(f"Verification: {test_ticker} has {len(df)} rows of data")
        if not df.empty:
            logger.info(f"  Close range: {df['$close'].min():.2f} - {df['$close'].max():.2f}")
        return True
    except Exception as e:
        logger.error(f"Qlib verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 data from Databento → Qlib format"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (default: 2020-01-01)")
    parser.add_argument("--end", default="2026-03-31", help="End date (default: 2026-03-31)")
    parser.add_argument("--output", default=None, help="Output directory (default: ~/.qlib/qlib_data/us_databento)")
    parser.add_argument("--full", action="store_true", help="Download all 500 stocks (slower)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip Qlib verification")
    args = parser.parse_args()

    # 检查 API key
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY 未设置。请在 .env 文件中添加或运行:")
        logger.error("  export DATABENTO_API_KEY='db-...'")
        sys.exit(1)

    # 输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = str(Path.home() / ".qlib" / "qlib_data" / "us_databento")

    # 选择股票列表
    if args.full:
        tickers = get_sp500_tickers_full()
    else:
        tickers = SP500_TOP100
        logger.info(f"Quick mode: downloading top {len(tickers)} S&P 500 stocks")
        logger.info("Use --full for all 500 stocks")

    # 下载
    start_time = time.time()
    df = download_from_databento(tickers, args.start, args.end, api_key)

    # 保存为 CSV（中间格式）
    csv_dir = str(Path(output_dir).parent / "databento_csv")
    save_as_csv(df, csv_dir)

    # 用 Qlib 官方 dump_bin.py 转换
    try:
        qlib_path = convert_csv_to_qlib(csv_dir, output_dir)
    except FileNotFoundError:
        logger.warning("dump_bin.py not found, falling back to manual conversion")
        qlib_path = databento_to_qlib_format(df, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"Download + conversion completed in {elapsed:.0f}s")

    # 验证
    if not args.skip_verify:
        logger.info("Verifying Qlib data...")
        if verify_qlib_data(qlib_path):
            logger.info("=" * 50)
            logger.info("SUCCESS! Databento data ready for backtesting.")
            logger.info(f"Data path: {qlib_path}")
            logger.info(f"To use: update configs/default.yaml:")
            logger.info(f'  qlib_data_path: "{qlib_path}"')
            logger.info("=" * 50)
        else:
            logger.warning("Qlib verification had issues. Data may still work.")
    else:
        logger.info(f"Data saved to: {qlib_path}")


if __name__ == "__main__":
    main()
