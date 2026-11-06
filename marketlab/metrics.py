"""Shared evaluation, so every model in the study is measured the same way.

The numbers describe barrier outcomes per trade, not a portfolio. Labels overlap
in time, so the standard errors below are optimistic even when computed honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from . import config


def evaluate(name: str, y_true, y_pred, fwd_ret) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fwd = np.asarray(fwd_ret, dtype=float)
    rec = recall_score(y_true, y_pred, labels=list(config.LABELS), average=None, zero_division=0)
    buys = y_pred == "buy"
    n_buy = int(buys.sum())
    nan = float("nan")
    ret_buy = float(fwd[buys].mean()) if n_buy else nan
    return {
        "name": name,
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=list(config.LABELS), average="macro", zero_division=0),
        "rec_buy": rec[0], "rec_hold": rec[1], "rec_sell": rec[2],
        "n_buy": n_buy,
        "ret_buy": ret_buy,
        "edge": ret_buy - float(fwd.mean()) if n_buy else nan,
        "net": ret_buy - config.COST if n_buy else nan,
        "se": float(fwd[buys].std(ddof=1) / np.sqrt(n_buy)) if n_buy > 1 else nan,
    }


def print_table(rows) -> None:
    df = pd.DataFrame(rows).set_index("name")
    for c in ("ret_buy", "edge", "net", "se"):
        df[c] = (df[c] * 100).round(2)
    for c in ("acc", "f1_macro", "rec_buy", "rec_hold", "rec_sell"):
        df[c] = df[c].round(3)
    print(df.to_string())
    print("\nacc = accuracy, f1_macro = mean F1 over the three classes,")
    print("rec_* = recall per class, n_buy = number of buy signals.")
    print("ret_buy, edge, net, se in percent per trade:")
    print("  ret_buy = mean return of the buy signals,")
    print("  edge    = ret_buy minus the mean return of all days,")
    print("  net     = ret_buy minus round-trip costs,")
    print("  se      = standard error (optimistic: labels overlap).")
