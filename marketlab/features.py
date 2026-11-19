"""Features and triple-barrier labels.

For a decision on day t:
  - features use data up to and including day t (no future window),
  - the entry price is the open of day t+1,
  - the barriers sit at entry * (1 +/- K * vol * sqrt(HORIZON)),
  - the first barrier touched within HORIZON days decides the label,
    and "hold" covers "neither barrier" and "both on the same day".

The label knows the future. Models later see only the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # A window with no down days is RSI 100 by definition, not undefined.
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    # A window that did not move at all has no direction: report the neutral midpoint.
    out = out.mask((loss == 0) & (gain == 0) & gain.notna(), 50.0)
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ret1 = close.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = ret1
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)
    f["vol_20d"] = ret1.rolling(config.VOL_WINDOW).std()
    f["dist_high_20d"] = close / df["High"].rolling(20).max() - 1
    f["dist_high_252d"] = close / df["High"].rolling(252).max() - 1
    f["vs_sma50"] = close / close.rolling(50).mean() - 1
    f["vs_sma200"] = close / close.rolling(200).mean() - 1
    vol_mean = df["Volume"].rolling(20).mean().replace(0, np.nan)
    f["vol_ratio"] = df["Volume"] / vol_mean
    f["rsi_14"] = rsi(close)
    return f


def triple_barrier(df: pd.DataFrame, vol: np.ndarray):
    """Return (label, forward return, exit date) per decision day."""
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    dates = df.index.to_numpy()
    n = len(df)
    label = np.full(n, None, dtype=object)
    ret = np.full(n, np.nan)
    t1 = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    for i in range(n - config.HORIZON):  # needs days i+1 ... i+HORIZON
        v = vol[i]
        if not np.isfinite(v) or v <= 0:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:  # a broken print, not a decision day
            continue
        width = config.K * v * np.sqrt(config.HORIZON)
        up, lo = entry * (1 + width), entry * (1 - width)

        lab, exit_px, j_end = "hold", c[i + config.HORIZON], i + config.HORIZON
        for j in range(i + 1, i + config.HORIZON + 1):
            # A gap through a barrier fills at the open, before that day can trade anywhere
            # else; the far barrier is unreachable once the open is beyond the near one.
            if o[j] >= up:
                lab, exit_px, j_end = "buy", o[j], j
                break
            if o[j] <= lo:
                lab, exit_px, j_end = "sell", o[j], j
                break
            hit_up, hit_lo = h[j] >= up, l[j] <= lo
            if hit_up and hit_lo:  # both barriers inside one bar: not separable, call it hold
                lab, exit_px, j_end = "hold", c[j], j
                break
            if hit_up:
                lab, exit_px, j_end = "buy", up, j
                break
            if hit_lo:
                lab, exit_px, j_end = "sell", lo, j
                break
        label[i] = lab
        ret[i] = exit_px / entry - 1
        t1[i] = dates[j_end]
    return label, ret, t1


def build(tickers=None) -> pd.DataFrame:
    """Label every ticker and write data/built/labels.csv."""
    tickers = tickers or config.UNIVERSE
    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    parts = []

    for ticker in tickers:
        path = config.RAW_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"{ticker}: no file, skipped")
            continue
        df = data.load(ticker)
        feats = make_features(df)
        label, ret, t1 = triple_barrier(df, feats["vol_20d"].to_numpy())
        feats["label"], feats["fwd_ret"], feats["t1"] = label, ret, t1
        feats["close"] = df["Close"]
        feats["ticker"] = ticker
        # infinity is not a number a scaler can handle, and dropna() does not remove it
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna(subset=config.FEATURES + ["label"])
        parts.append(feats.reset_index())
        print(f"{ticker}: {len(feats)} labelled days")

    if not parts:
        raise SystemExit("No data found. Run `python run.py download` first.")

    out = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])
    out.to_csv(config.BUILD_DIR / "labels.csv", index=False)

    print(f"\n{len(out)} rows, {out['date'].min().date()} to {out['date'].max().date()}")
    print("\nClass share:")
    print(out["label"].value_counts(normalize=True).round(3).to_string())
    print("\nMean forward return per label (barrier returns, before costs):")
    print(out.groupby("label")["fwd_ret"].mean().round(4).to_string())
    return out
