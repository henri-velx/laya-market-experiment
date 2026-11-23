"""Independent check of the reported "enriched features + regression target" result.

Not part of the study pipeline. It rebuilds a comparable feature set from the raw bars,
trains one gradient-boosting regressor on the training split and scores it once on the
sealed 2024+ block, so a claim made outside this repo can be checked against the data.

    python experiments/verify_enriched.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from marketlab import config, data  # noqa: E402
from marketlab.features import rsi  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ENRICHED = [
    "ret_2d", "ret_10d", "ret_60d", "ret_120d", "vol_20d_x", "vol_60d", "vol_chg",
    "rsi_7", "rsi_28", "macd_hist", "bb_pctb", "dist_sma20", "sma20_slope", "dist_sma50",
    "atr_pct", "gap", "intraday", "vol_z", "spy_ret_20d", "spy_vol_20d", "spy_dist_sma50",
    "rel_spy_20", "rel_xs_20",
]


def ticker_features(ticker: str) -> pd.DataFrame | None:
    path = config.RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = data.load(ticker)
    c, h, l, o, v = (df[k] for k in ("Close", "High", "Low", "Open", "Volume"))
    ret1 = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_2d"], f["ret_10d"], f["ret_20d"], f["ret_60d"], f["ret_120d"] = (
        c.pct_change(n) for n in (2, 10, 20, 60, 120)
    )
    f["vol_20d_x"] = ret1.rolling(20).std()
    f["vol_60d"] = ret1.rolling(60).std()
    f["vol_chg"] = f["vol_20d_x"] / f["vol_60d"] - 1
    f["rsi_7"], f["rsi_28"] = rsi(c, 7), rsi(c, 28)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    f["dist_sma20"] = c / ma20 - 1
    f["sma20_slope"] = ma20 / ma20.shift(5) - 1
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["gap"] = o / c.shift() - 1
    f["intraday"] = c / o - 1
    f["vol_z"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["ticker"] = ticker
    return f.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    base = pd.read_csv(config.BUILD_DIR / "dataset.csv", parse_dates=["date", "t1"])
    frames = [f for t in config.UNIVERSE if (f := ticker_features(t)) is not None]
    feats = pd.concat(frames).reset_index().rename(columns={"index": "date"})

    spy = feats[feats.ticker == "SPY"].set_index("date")[["ret_20d", "ret_60d", "vol_20d_x"]]
    spy.columns = ["spy_ret_20d", "spy_ret_60d", "spy_vol_20d"]
    feats = feats.merge(spy, left_on="date", right_index=True, how="left")
    feats = feats.merge(
        base[base.ticker == "SPY"].set_index("date")["vs_sma50"].rename("spy_dist_sma50"),
        left_on="date", right_index=True, how="left",
    )

    df = base.merge(feats, on=["ticker", "date"], how="inner", suffixes=("", "_raw"))
    df["rel_spy_20"] = df["ret_20d"] - df["spy_ret_20d"]
    df["rel_xs_20"] = df["ret_20d"] - df.groupby("date")["ret_20d"].transform("mean")
    tr = df[(df.split == "train")].dropna(subset=ENRICHED)
    te = df[(df.split == "test")].dropna(subset=ENRICHED)
    print(f"train {len(tr)} rows, test {len(te)} rows, {len(ENRICHED)} features")

    model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
    )
    model.fit(tr[ENRICHED], tr["fwd_ret"])
    pred = model.predict(te[ENRICHED])
    fwd = te["fwd_ret"].to_numpy()
    mean_all = fwd.mean()

    order = np.argsort(-pred)
    print("\nsealed test block 2024+ (trained only on data before 2024)")
    for frac in (0.10, 0.20):
        k = int(round(frac * len(te)))
        sel = np.zeros(len(te), bool)
        sel[order[:k]] = True
        raw, edge = fwd[sel].mean(), fwd[sel].mean() - mean_all
        print(f"  top-{frac:.0%}: n={k:5d}  return {raw:+.2%}  edge {edge:+.2%}  net {raw - config.COST:+.2%}")

    sel20 = np.zeros(len(te), bool)
    sel20[order[: int(round(0.20 * len(te)))]] = True
    print("\n  per year (top-20%):")
    for year, g in te[sel20].groupby(te.loc[sel20, "date"].dt.year):
        base_year = fwd[te["date"].dt.year == year].mean()
        print(f"    {year}: return {g['fwd_ret'].mean():+.2%}  edge {g['fwd_ret'].mean() - base_year:+.2%}")

    per_ticker = [g["fwd_ret"].mean() for _, g in te[sel20].groupby("ticker") if len(g) >= 40]
    print(f"\n  tickers with >=40 signals: {sum(r > 0 for r in per_ticker)} of {len(per_ticker)} positive")


if __name__ == "__main__":
    main()
