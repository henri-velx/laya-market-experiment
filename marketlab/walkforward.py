"""Walk-forward test.

For every year Y, train only on data before Y (and only on rows whose label has
resolved before Y starts) and predict Y. One good year is not a signal; a real
edge shows up in most years. The test block from 2024 on is never read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, metrics
from .baselines import make_models

MIN_TRAIN_YEARS = 3


def run(dataset: pd.DataFrame, last_date: str = config.VAL_END) -> pd.DataFrame:
    df = dataset[dataset["date"] <= pd.Timestamp(last_date)]
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    years = sorted(df["date"].dt.year.unique())
    if not years:
        raise SystemExit("No rows to walk forward over. Check the data and the config dates.")
    test_years = [y for y in years if y >= years[0] + MIN_TRAIN_YEARS]
    if not test_years:
        raise SystemExit(f"Not enough history to walk forward over: only {years} present.")
    print(f"Data {years[0]} to {years[-1]}, test years {test_years[0]} to {test_years[-1]}\n")

    rows, pooled = [], {}
    for year in test_years:
        start = pd.Timestamp(f"{year}-01-01")
        train = df[(df["date"] < start) & (df["t1"] < start)]  # purging
        test = df[df["date"].dt.year == year]
        if train.empty or test.empty:
            continue
        market = float(test["fwd_ret"].mean())
        for name, model in make_models().items():
            model.fit(train[config.FEATURES], train["label"])
            pred = model.predict(test[config.FEATURES])
            row = metrics.evaluate(name, test["label"], pred, test["fwd_ret"])
            row.update(year=int(year), n_test=int(len(test)), market=market)
            rows.append(row)
            pooled.setdefault(name, []).append(
                pd.DataFrame({"pred": pred, "fwd": test["fwd_ret"].to_numpy()})
            )

    result = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.RESULTS_DIR / "walkforward.csv", index=False)

    show = result[["year", "name", "n_test", "f1_macro", "n_buy", "ret_buy", "edge", "net", "se", "market"]].copy()
    for c in ("ret_buy", "edge", "net", "se", "market"):
        show[c] = (show[c] * 100).round(2)
    show["f1_macro"] = show["f1_macro"].round(3)
    print("Per year (percentages are per trade; market = mean of ALL days that year):\n")
    print(show.to_string(index=False))

    print("\nSummary over all years:")
    for name, parts in pooled.items():
        all_rows = pd.concat(parts)
        buys = all_rows[all_rows["pred"] == "buy"]["fwd"]
        edge_pooled = buys.mean() - all_rows["fwd"].mean()
        per_year = result[result["name"] == name]["edge"]
        print(
            f"  {name}: positive edge in {(per_year > 0).sum()} of {len(per_year)} years, "
            f"mean yearly edge {per_year.mean() * 100:+.2f}%, worst year {per_year.min() * 100:+.2f}%, "
            f"pooled edge {edge_pooled * 100:+.2f}%, pooled net {(buys.mean() - config.COST) * 100:+.2f}% "
            f"over {len(buys)} buy signals"
        )
    print("\nA real edge shows up in most years, not just in one.")
    return result
