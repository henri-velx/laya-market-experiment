# Can a foundation model pick stocks?

**A small, honest experiment — and a negative result.**

I took a small local decision model — not an LLM, no API call — gave it sixteen years of
daily market data rendered as plain English, and asked it to call the next ten trading days
on twenty liquid US tickers. Then I checked the answer the boring way: one year at a time,
always training only on the past.

It held up on the validation block and fell apart when rolled forward. This repo is the
experiment, the numbers, and the few pieces worth keeping.

**Start here:** [read the report rendered](https://henri-velx.github.io/laya-market-experiment/report.html) — no install, no server. It is one self-contained file with the charts and the tables, also [in the repo](report.html).

---

## The result in four numbers

| | value |
|---|---|
| Laya, zero-shot, validation block (2022–2023) | **+0.15%** edge per buy signal |
| Best model on the same block (gradient boosting) | **+0.35%** edge per buy signal |
| Walk-forward years with a positive edge (2013–2023) | **5 of 11** |
| Mean yearly edge over those 11 years | **−0.07%** |

The validation block looked like a signal. The walk-forward says it was a window.

---

## What the experiment does

1. **Download** daily bars for 20 tickers (indices, sectors, bonds, gold, large caps), 2010
   onward.
2. **Label** each decision day with a triple barrier: entry at the next open, barriers at
   `±1.0 × 20-day volatility × √10`, whichever is touched first within ten trading days
   decides `buy` / `hold` / `sell`. Roughly 79k labelled name-days.
3. **Split** chronologically — train to 2021, validate 2022–2023, test 2024+ — and *purge*
   every training row whose ten-day label had not resolved before the next block began.
4. **Compare** four dumb baselines and two learned models on the validation block, all scored
   identically.
5. **Walk forward**: for each year 2013–2023, train only on prior years and predict that year.
6. **Ask Laya** the same question in words, zero-shot, and score its answers with the same
   code as everything else.

The test block from 2024 on is never scored in this repo. The bars are downloaded; the
numbers stop at 2023.

## What Laya is

Laya is **not** a chat model and not an LLM in the usual sense. It is a small decision model
you run locally: one ~0.8 GB checkpoint (`convaiinnovations/laya`, `pip install laya`), one
forward pass, no text generation, no sampling, no API call. You hand it a flat JSON state and
a question with named options, and it returns a probability for each option — closer to a
classifier with a fixed interface than to a chatbot with a prompt.

That constraint is what the experiment is really about. A classifier cannot read a price
series, so each day is first rendered into words (`marketlab/state.py`), and Laya is asked one
`choice` question: *price rises clearly / stays in a range / falls clearly within ten days*.
Its answer is the argmax over those three options, plus a probability per option. Nothing is
trained or fine-tuned — this is a zero-shot read of a pretrained model that has never seen
this data.

So the honest claim is narrower than "an AI failed at trading": a 0.8 GB decision model,
reading a hand-rendered summary of a price series, came out less accurate than always
answering `hold`. The interesting part is therefore the interface, not the weights — see
**The part worth stealing** below.

## What I actually measured

Validation block (train = 2010–2021, validation = 2022–2023, 9,838 name-days):

| model | accuracy | macro F1 | buy signals | return per buy | edge | net of cost |
|---|---:|---:|---:|---:|---:|---:|
| always buy (market drift) | 0.301 | 0.154 | 9,838 | +0.13% | 0.00% | −0.17% |
| always most common class | 0.401 | 0.191 | 0 | — | — | — |
| random by class prior | 0.342 | 0.332 | 3,062 | +0.10% | −0.03% | −0.20% |
| rule: SMA + momentum | 0.332 | 0.330 | 3,192 | +0.22% | +0.09% | −0.08% |
| logistic regression | 0.422 | 0.406 | 3,642 | +0.45% | **+0.32%** | +0.15% |
| gradient boosting | 0.430 | 0.411 | 3,410 | +0.49% | **+0.35%** | +0.19% |
| Laya, zero-shot | 0.324 | 0.316 | 4,463 | +0.29% | +0.15% | −0.01% |

Walk-forward, per model over eleven years:

| model | positive years | mean yearly edge | worst year | pooled edge |
|---|---:|---:|---:|---:|
| logistic regression | 6 / 11 | −0.02% | −0.89% | −0.03% |
| gradient boosting | 5 / 11 | −0.07% | −0.40% | −0.07% |

Laya's own confusion matrix is in the report. Short version: it answered `buy` on 45% of days,
and its accuracy (32.4%) is *below* always answering `hold` (40.1%). The probabilities it
returns are clean and ordered and carry no usable ranking.

## The part worth stealing

Laya cannot read a feature vector, so each day is rendered as a short English
state, with moves measured in units of the ticker's own volatility:

```python
{
  "trend": "above 50d and 200d average",
  "move_1d": "slightly down",
  "move_20d": "sharp rise",
  "volatility": "high",
  "vs_52w_high": "below high",
  "rsi": "neutral",
  "volume": "normal",
}
```

Same information, different surface. It lives in [`marketlab/state.py`](marketlab/state.py)
and is the one design choice in this study I would reuse elsewhere.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py all          # download, label, baselines, walk-forward, report
```

Or one stage at a time: `download`, `label`, `baselines`, `walkforward`, `report`. Use
`--tickers SPY QQQ IWM`, `--start 2015-01-01` and `--end 2023-12-31` to narrow it down. The
end date is pinned in `marketlab/config.py` (`END`), so a re-run reproduces the committed
numbers instead of quietly picking up the latest trading day.

Regenerating the report needs nothing but Python:

```bash
python run.py report
```

The unit tests need neither data nor network:

```bash
python -m unittest discover -s tests
```

The optional Laya arm is not part of `all`:

```bash
pip install laya           # needs a CUDA GPU; downloads ~0.8 GB of weights
python run.py laya
```

It writes `results/laya_val.csv`, and the report picks it up automatically.

## Honest limitations

- **Barrier returns are not portfolio returns.** No position sizing, cash constraint, or
  concurrent book. A real portfolio would look worse.
- **Labels overlap in time** (ten-day horizons on consecutive days), so the standard errors
  printed here are optimistic even when computed honestly.
- **Twenty tickers, chosen in hindsight.** These are survivors and mostly US large caps, so
  absolute levels are flattered. Only relative comparisons are meaningful.
- **Costs are a guess**: 0.30% round trip, no spread, no slippage, no market impact.
- **One model, one interface, one question.** Laya may do better as a ranker than as a
  three-way classifier; this experiment cannot tell.
- **Data comes from Yahoo** and is not redistributed here. `data/` is gitignored.

## Background

This repo is a distilled, cleaned-up version of a larger study that reached the same verdict
with more machinery: pre-registered success criteria, block bootstraps, a closed-loop
simulator, and a sealed hold-out. The conclusion there was also *no edge* — and the most
useful thing it produced was the discipline to say so.

## License

MIT. See [LICENSE](LICENSE).
