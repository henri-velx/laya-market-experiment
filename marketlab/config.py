"""Every setting in one place. The stage modules read from here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Set MARKETLAB_DATA to reuse a data directory that already exists.
DATA_DIR = Path(os.environ.get("MARKETLAB_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("MARKETLAB_RESULTS", ROOT / "results"))
RAW_DIR = DATA_DIR / "raw"
BUILD_DIR = DATA_DIR / "built"

# Liquid, long-lived tickers: broad indices, sectors, bonds, gold and large caps.
# Research data only. A private EU investor would trade UCITS vehicles instead.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLF", "XLK",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "KO", "TSLA",
]

START = "2010-01-01"

# Pin the end of the download so a re-run reproduces the committed numbers instead of
# picking up whatever the last trading day happens to be. `run.py --end` overrides it.
END = "2026-09-22"

# Decision at the close of day t, entry at the open of day t+1.
HORIZON = 10      # at most this many trading days to wait
K = 1.0           # barrier width = K * daily vol * sqrt(HORIZON)
VOL_WINDOW = 20   # days used for the volatility estimate

# The test block was never used for any number in this repo.
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"

LABELS = ("buy", "hold", "sell")

# Rough round-trip cost: 0.15% per side, no spread, no slippage model.
COST = 0.003

FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "vol_20d", "dist_high_20d", "dist_high_252d",
    "vs_sma50", "vs_sma200", "vol_ratio", "rsi_14",
]
