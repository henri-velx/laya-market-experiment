"""Render report.html from the saved result files.

Standard library only, on purpose: the report is the artefact a visitor sees
first, and it must open with no environment set up at all.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import config

LABELS = ("buy", "hold", "sell")
MODELS = ("logistic regression", "gradient boosting")
COLORS = {"logistic regression": "#5eead4", "gradient boosting": "#fbbf24"}
DEFAULT_COST = 0.003


# --------------------------------------------------------------------------- io

def _rows(name: str) -> list[dict]:
    path = config.RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _summary() -> dict:
    path = config.RESULTS_DIR / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- numbers

def _stats(rows: list[dict], cost: float) -> dict | None:
    """Accuracy, macro F1 and the per-trade numbers, without sklearn."""
    pairs = [(r.get("label"), r.get("pred"), _num(r, "fwd_ret")) for r in rows]
    pairs = [(a, p, f) for a, p, f in pairs if a in LABELS and p in LABELS]
    n = len(pairs)
    if not n:
        return None

    hits = sum(1 for a, p, _ in pairs if a == p)
    f1s = []
    for label in LABELS:
        tp = sum(1 for a, p, _ in pairs if a == label and p == label)
        fp = sum(1 for a, p, _ in pairs if a != label and p == label)
        fn = sum(1 for a, p, _ in pairs if a == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    buys = [f for _, p, f in pairs if p == "buy"]
    mean_all = sum(f for _, _, f in pairs) / n
    ret_buy = sum(buys) / len(buys) if buys else float("nan")
    edge = ret_buy - mean_all if buys else float("nan")
    se = float("nan")
    if len(buys) > 1:
        mean_buy = sum(buys) / len(buys)
        var = sum((x - mean_buy) ** 2 for x in buys) / (len(buys) - 1)
        se = math.sqrt(var) / math.sqrt(len(buys))

    pred_share = {label: sum(1 for _, p, _ in pairs if p == label) / n for label in LABELS}
    actual_share = {label: sum(1 for a, _, _ in pairs if a == label) / n for label in LABELS}
    return {
        "n": n, "acc": hits / n, "f1": sum(f1s) / 3,
        "n_buy": len(buys), "ret_buy": ret_buy, "edge": edge,
        "net": ret_buy - cost if buys else float("nan"), "se": se,
        "pred_share": pred_share, "actual_share": actual_share,
        "majority": max(actual_share.values()),
    }


def _walk_forward_stats(wf: list[dict], model: str) -> dict | None:
    per_year = [_num(r, "edge") * 100 for r in wf if r.get("name") == model]
    per_year = [v for v in per_year if not math.isnan(v)]
    if not per_year:
        return None
    return {
        "years": len(per_year),
        "positive": sum(1 for v in per_year if v > 0),
        "mean": sum(per_year) / len(per_year),
        "worst": min(per_year),
        "best": max(per_year),
    }


def _confusion(rows: list[dict]) -> dict:
    counts = {(a, p): 0 for a in LABELS for p in LABELS}
    for r in rows:
        a, p = r.get("label"), r.get("pred")
        if a in LABELS and p in LABELS:
            counts[(a, p)] += 1
    return counts


# ----------------------------------------------------------------------- format

def _pct(v: float, digits: int = 2, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"


def _f3(v: float) -> str:
    return "—" if isinstance(v, float) and math.isnan(v) else f"{v:.3f}"


# ------------------------------------------------------------------------- chart

def _edge_chart(wf: list[dict]) -> str:
    years = sorted({int(_num(r, "year")) for r in wf if r.get("year")})
    if not years:
        return '<p class="muted">No walk-forward results yet. Run <code>python run.py walkforward</code>.</p>'

    series = {
        m: {int(_num(r, "year")): _num(r, "edge") * 100 for r in wf if r.get("name") == m}
        for m in MODELS
    }
    values = [v for s in series.values() for v in s.values() if not math.isnan(v)]
    max_abs = max([abs(v) for v in values] + [0.35])
    step = 0.5 if max_abs <= 1.2 else 1.0
    top = step * math.ceil(max_abs / step)

    width, height = 900, 330
    left, right, pad_top, bottom = 54, 150, 20, 42
    plot_w, plot_h = width - left - right, height - pad_top - bottom
    zero = pad_top + plot_h / 2
    scale = (plot_h / 2) / top
    group = plot_w / len(years)
    bar_w = min(13.0, group / 3.4)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Edge of each model per walk-forward year">']

    ticks = int(round(top / step))
    for k in range(-ticks, ticks + 1):
        value = k * step
        y = zero - value * scale
        cls = "zero" if k == 0 else "grid"
        out.append(f'<line class="{cls}" x1="{left}" y1="{y:.1f}" '
                   f'x2="{left + plot_w}" y2="{y:.1f}"/>')
        if k:
            out.append(f'<text class="tick" x="{left - 10}" y="{y + 3.5:.1f}" '
                       f'text-anchor="end">{value:+.1f}</text>')

    out.append(f'<text class="ylab" transform="translate(16 {pad_top + plot_h / 2:.1f}) rotate(-90)" '
               f'text-anchor="middle">edge per trade (pp)</text>')

    for i, year in enumerate(years):
        cx = left + group * (i + 0.5)
        out.append(f'<text class="tick" x="{cx:.1f}" y="{height - bottom + 22}" '
                   f'text-anchor="middle">{year}</text>')
        for k, model in enumerate(MODELS):
            value = series[model].get(year)
            if value is None or math.isnan(value):
                continue
            height_px = abs(value) * scale
            x = cx + (k - (len(MODELS) - 1) / 2) * (bar_w + 4) - bar_w / 2
            y = zero - height_px if value >= 0 else zero
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(height_px, 0.9):.1f}" rx="1.5" fill="{COLORS[model]}" '
                f'opacity="0.9"><title>{model}, {year}: {value:+.2f} pp</title></rect>'
            )

    for k, model in enumerate(MODELS):
        y = pad_top + 14 + k * 22
        out.append(f'<rect x="{left + plot_w + 18}" y="{y - 9}" width="10" height="10" rx="2" '
                   f'fill="{COLORS[model]}"/>')
        out.append(f'<text class="tick" x="{left + plot_w + 34}" y="{y}" '
                   f'style="fill:#c3d0dc;font-size:12px">{model}</text>')

    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------- rendering

CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0f14;color:#e6edf3;
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:56px 24px 80px}
.kicker{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.16em;
  text-transform:uppercase;color:#5eead4}
h1{font-size:clamp(30px,4.6vw,46px);line-height:1.1;letter-spacing:-.02em;margin:14px 0 0}
h2{font-size:23px;letter-spacing:-.01em;margin:58px 0 10px}
h3{font-size:14px;text-transform:uppercase;letter-spacing:.09em;color:#93a4b5;margin:34px 0 10px}
p{margin:12px 0;color:#c3d0dc}
.lede{font-size:19px;line-height:1.6;color:#dbe6ef;max-width:64ch;margin-top:18px}
.muted{color:#8b9aab}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#151d26;
  border:1px solid #223040;border-radius:5px;padding:.1em .35em;font-size:.88em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px;margin:36px 0 10px}
.card{background:linear-gradient(180deg,#121a23,#0e141b);border:1px solid #1f2b38;
  border-radius:14px;padding:18px 18px 16px}
.card .v{font:600 26px/1.15 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.card .l{font-size:13px;color:#8b9aab;margin-top:8px;line-height:1.45}
.bad{color:#fca5a5}.warn{color:#fbbf24}.good{color:#5eead4}
figure{margin:26px 0 0}
svg{width:100%;height:auto;display:block;background:#0e141b;border:1px solid #1f2b38;border-radius:14px}
svg .grid{stroke:#1b2734;stroke-width:1}
svg .zero{stroke:#3d5268;stroke-width:1.5}
svg .tick{fill:#7f8fa1;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}
svg .ylab{fill:#7f8fa1;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.06em}
figcaption{font-size:13.5px;color:#8b9aab;margin-top:12px;max-width:76ch}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}
th,td{text-align:right;padding:10px;border-bottom:1px solid #1b2734;
  font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{color:#8b9aab;font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}
tbody th{font-weight:500;color:#dbe6ef}
.matrix th{color:#8b9aab;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}
.matrix td{background:rgba(94,234,212,calc(var(--i) * .5));border:1px solid #1b2734;
  border-radius:8px;text-align:left;padding:10px 12px}
.matrix td b{display:block;font:600 17px ui-monospace,SFMono-Regular,Menlo,monospace;color:#eaf6f4}
.matrix td span{display:block;font-size:11px;color:#9fb8b5}
ul{padding-left:20px}li{margin:8px 0;color:#c3d0dc}
.foot{margin-top:64px;padding-top:22px;border-top:1px solid #1b2734;font-size:13px;color:#7f8fa1}
"""


def _val_table(base: list[dict], laya_row: dict | None) -> str:
    head = ("<tr><th>model</th><th>accuracy</th><th>macro F1</th><th>buy signals</th>"
            "<th>return per buy</th><th>edge</th><th>net of cost</th><th>std. error</th></tr>")
    body = []
    for r in base:
        body.append(
            "<tr><th>{}</th><td>{}</td><td>{}</td><td>{:,}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                r.get("name", ""), _f3(_num(r, "acc")), _f3(_num(r, "f1_macro")),
                int(_num(r, "n_buy")) if not math.isnan(_num(r, "n_buy")) else 0,
                _pct(_num(r, "ret_buy")), _pct(_num(r, "edge")),
                _pct(_num(r, "net")), _pct(_num(r, "se"), 2, sign=False),
            )
        )
    if laya_row:
        body.append(
            "<tr><th>{}</th><td>{}</td><td>{}</td><td>{:,}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                laya_row["name"], _f3(laya_row["acc"]), _f3(laya_row["f1"]),
                laya_row["n_buy"], _pct(laya_row["ret_buy"]), _pct(laya_row["edge"]),
                _pct(laya_row["net"]), _pct(laya_row["se"], 2, sign=False),
            )
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _matrix(counts: dict) -> str:
    head = "<tr><th></th>" + "".join(f"<th>said {p}</th>" for p in LABELS) + "</tr>"
    body = []
    for actual in LABELS:
        row_total = sum(counts[(actual, p)] for p in LABELS) or 1
        cells = []
        for pred in LABELS:
            n = counts[(actual, pred)]
            share = n / row_total
            cells.append(f'<td style="--i:{share:.3f}"><b>{n:,}</b><span>{share * 100:.0f}% of {actual} days</span></td>')
        body.append(f"<tr><th>actually {actual}</th>{''.join(cells)}</tr>")
    return f'<table class="matrix"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def _render(wf, base, laya, summary) -> str:
    cost = float(summary.get("params", {}).get("cost_round_trip", DEFAULT_COST))
    laya_stats = _stats(laya, cost)
    best_val = max(
        (r for r in base if not math.isnan(_num(r, "edge"))),
        key=lambda r: _num(r, "edge"), default=None,
    )

    wf_gb = _walk_forward_stats(wf, "gradient boosting")
    wf_lr = _walk_forward_stats(wf, "logistic regression")

    cards = []
    if laya_stats:
        cards.append(("warn", _pct(laya_stats["edge"]), "Laya, zero-shot: extra return per buy signal"))
    if best_val is not None:
        cards.append(("good", _pct(_num(best_val, "edge")),
                      f"best model on the validation block ({best_val.get('name')})"))
    if wf_gb:
        cards.append(("bad", f"{wf_gb['positive']}/{wf_gb['years']}",
                      "walk-forward years with a positive edge (gradient boosting)"))
        cards.append(("bad", f"{wf_gb['mean']:+.2f} pp",
                      "mean yearly edge once you roll the model forward"))
    card_html = "".join(
        f'<div class="card"><div class="v {cls}">{value}</div><div class="l">{label}</div></div>'
        for cls, value, label in cards
    )

    laya_section = ""
    if laya_stats:
        ps = laya_stats["pred_share"]
        counts = _confusion(laya)
        sell_calls = sum(counts[(a, "sell")] for a in LABELS) or 1
        sell_days = sum(counts[("sell", p)] for p in LABELS) or 1
        sell_precision = counts[("sell", "sell")] / sell_calls
        buy_on_sell_days = counts[("sell", "buy")] / sell_days
        laya_section = f"""
    <h2>What Laya actually said</h2>
    <p>Laya is a small decision model run locally: one ~0.8 GB checkpoint, one forward pass,
    no text generation, no API call &mdash; a classifier with a fixed interface, not an LLM.
    Zero-shot, with no training at all, it answered the question for all
    {laya_stats['n']:,} validation rows. The model was asked to pick one of three outcomes for
    the next {summary.get('params', {}).get('horizon_days', 10)} trading days, and it answered
    <b>buy</b> on {ps['buy'] * 100:.1f}% of them, <b>hold</b> on {ps['hold'] * 100:.1f}% and
    <b>sell</b> on {ps['sell'] * 100:.1f}%.</p>
    <p>Its accuracy was {laya_stats['acc'] * 100:.1f}%, against {laya_stats['majority'] * 100:.1f}%
    for always answering the most common class. That is the whole story in two numbers: a
    confident-sounding model that is less accurate than a constant.</p>
    <figure>
      {_matrix(counts)}
      <figcaption>Rows are what actually happened, columns what Laya said; the diagonal is where
      it was right. It said <b>buy</b> on {buy_on_sell_days * 100:.0f}% of the days that went on
      to fall sharply, and {sell_precision * 100:.0f}% of its sell calls were correct against a
      {laya_stats['actual_share']['sell'] * 100:.0f}% base rate.</figcaption>
    </figure>
    <p>The interesting part is not that it is wrong. It is that the probabilities it emits are
    tidy, ordered and useless &mdash; a model that picks the most likely label still sounds
    certain when the ranking behind it carries no information.</p>
"""

    wf_lr_line = ""
    if wf_lr:
        wf_lr_line = (f"Logistic regression did slightly better on the same test: "
                      f"{wf_lr['positive']} of {wf_lr['years']} positive years, mean "
                      f"{wf_lr['mean']:+.2f} pp.")

    body = f"""
  <div class="wrap">
    <div class="kicker">market signal study &middot; negative result</div>
    <h1>Can a foundation model pick stocks?</h1>
    <p class="lede">I gave a small local decision model sixteen years of daily market data and
    asked it one question. It answered confidently, and it was wrong. This is the whole
    experiment, the numbers, and the part of the pipeline I would keep.</p>

    <div class="cards">{card_html}</div>

    <h2>The validation block said yes</h2>
    <p>Train on 2010&ndash;2021, validate on 2022&ndash;2023. Every model is scored the same way:
    it either says <b>buy</b>, <b>hold</b> or <b>sell</b>; a buy is scored by the mean return of
    the barrier trade that follows; <b>edge</b> is that return minus the mean return of every
    day in the block. Costs are 0.30% per round trip.</p>
    <figure>
      {_val_table(base, ({"name": "Laya, zero-shot", **laya_stats} if laya_stats else None))}
      <figcaption>An edge of {_pct(_num(best_val, "edge")) if best_val else "+0.3%"} per trade
      looks like a signal. It is also exactly the size of a result that can disappear when you
      stop choosing the window.</figcaption>
    </figure>

    <h2>The walk-forward said no</h2>
    <p>So I did the boring test. For every year from 2013 to 2023, train only on the years before
    it &mdash; and purge any training row whose ten-day label had not resolved before the test
    year began &mdash; then predict that year. Same features, same models, same costs. One good
    year proves nothing; a real edge has to show up again and again.</p>
    <figure>
      {_edge_chart(wf)}
      <figcaption>Edge per trade, by year. The bars cross zero in both directions and stay
      small. {wf_lr_line} The validation block was one of the good windows, and it was not
      special.</figcaption>
    </figure>
    <p>That is the result: <b>no stable edge</b>. Not "a weak edge" &mdash; an edge that changes
    sign year to year and averages to nothing, measured on data that the model never saw. The
    honest conclusion is that ten technical features on twenty liquid tickers do not contain a
    signal this model can extract, not that a better hyperparameter would fix it.</p>

    {laya_section}

    <h2>Why publish a negative result</h2>
    <ul>
      <li><b>Most trading repos show a backtest, not a test.</b> A single holdout block, chosen
      after the fact, will always contain a profitable-looking window. The walk-forward is what
      separates a result from a coincidence, and it is cheap to run.</li>
      <li><b>The numbers here are barrier returns, not portfolio returns.</b> They ignore
      position sizing, overlapping labels, liquidity and the fact that a real book holds many
      names at once. Every one of those would make the picture worse, not better.</li>
      <li><b>The interesting artefact is the interface.</b> Laya cannot read a
      feature vector, so the state is rendered as words ("above 50d and 200d average",
      "sharp drop", "near high"). Same information, different surface. That translation step is
      the reusable idea in this repo, and it lives in one small file.</li>
      <li><b>Nothing in the test block was touched.</b> The numbers stop at the end of 2023. The
      2024+ bars are downloaded and never scored, so anyone can reuse them for their own test.</li>
    </ul>

    <h2>What it would take to be wrong</h2>
    <p>A real signal here would need all of: a positive edge in most walk-forward years, a
    standard error small enough to separate it from zero, survival after costs, and a
    portfolio-level simulation instead of per-trade barrier returns. Any one of those on its own
    is easy to produce by accident, which is exactly why this repo reports all of them.</p>

    <div class="foot">
      Generated from <code>results/*.csv</code> by <code>python run.py report</code> &middot;
      no dependencies, no build step. Data: Yahoo daily bars, 20 tickers, 2010&ndash;2026,
      labels scored through 2023. Code and text: MIT.
    </div>
  </div>
"""

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Can a foundation model pick stocks? A negative result</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>\n"
    )


def build() -> Path:
    wf = _rows("walkforward.csv")
    base = _rows("baselines_val.csv")
    laya = _rows("laya_val.csv")
    html = _render(wf, base, laya, _summary())
    path = config.ROOT / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
