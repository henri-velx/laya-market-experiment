"""Download and load daily bars (Open, High, Low, Close, Volume)."""
from __future__ import annotations

import time

import pandas as pd

from . import config

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _fetch(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # network or Yahoo-side problem
            print(f"  {ticker}: attempt {attempt} failed: {exc}")
        if attempt < retries:  # no point sleeping after the last attempt
            time.sleep(2 * attempt)
    return None


def download(tickers=None, start: str | None = None, end: str | None = None) -> None:
    """Write one CSV per ticker to data/raw.

    `end` is inclusive here. yfinance treats its own end date as exclusive, so one day
    is added before the call.
    """
    tickers = tickers or config.UNIVERSE
    start = start or config.START
    end = end or config.END
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    loaded, failed = [], []
    for ticker in tickers:
        df = _fetch(ticker, start, end_exclusive)
        if df is None:
            failed.append(ticker)
            print(f"{ticker}: no data")
            continue
        df = df[COLUMNS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")].dropna()
        df.to_csv(config.RAW_DIR / f"{ticker}.csv")
        loaded.append(ticker)
        print(f"{ticker}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
        time.sleep(0.5)  # be gentle with the endpoint

    print(f"\n{len(loaded)} downloaded, {len(failed)} failed {failed if failed else ''}")


def load(ticker: str) -> pd.DataFrame:
    path = config.RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python run.py download` first.")
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
