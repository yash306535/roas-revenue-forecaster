# Probabilistic Revenue Forecasting for E-commerce Marketing

An AI-assisted forecasting utility that predicts e-commerce **revenue** and
**ROAS** as probabilistic ranges (P10 / P50 / P90) for 30 / 60 / 90-day planning
windows, broken down by channel and campaign type, with budget simulation and
LLM-generated insights.

![System architecture](docs/assets/architecture.png)

## How to run locally

```bash
# 1. create the environment (Python 3.11)
pip install -r requirements.txt

# 2. run the full pipeline (feature generation + prediction)
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

Output is written to `output/predictions.csv`.

`run.sh` accepts three optional positional arguments and falls back to defaults:

```
./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
```

## Interactive demo (frontend + AI insights)

The demo is **separate** from the scored pipeline (it is the only part that uses
the internet / LLM). To run it:

```bash
# extra demo-only dependencies
pip install -r requirements-app.txt

# add your OpenAI key (optional — without it, a rule-based summary is shown)
cp .env.example .env        # then edit .env and paste your key

streamlit run app.py
```

The app shows the forecast ranges, a confidence-band chart, channel/campaign-type
breakdowns, a **budget-simulation slider**, and an **AI insights** panel.

## Documentation

- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — methodology, model selection, validation.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — stack, modules, pipeline, LLM workflow.

## Repository structure

```
.
├── run.sh                    # single entry point (features -> predictions)
├── requirements.txt          # pinned dependencies (scored pipeline)
├── requirements-app.txt      # extra deps for the demo (Streamlit/LLM)
├── app.py                    # Streamlit demo UI
├── .env.example              # template for the OpenAI key
├── data/                     # input CSVs (overwritten with test data at scoring)
├── pickle/
│   └── model.pkl             # pre-trained, committed model artifact
├── src/
│   ├── config.py             # canonical schema + shared mappings
│   ├── data_loader.py        # per-platform adapters -> one unified table
│   ├── features.py           # monthly aggregation + lag/seasonality features
│   ├── generate_features.py  # entry: data/ -> features.parquet
│   ├── model.py              # quantile (P10/P50/P90) model definition
│   ├── train.py              # OFFLINE training + backtest -> model.pkl
│   ├── forecast.py           # recursive roll-forward + budget simulation
│   ├── predict.py            # entry: features + model -> predictions.csv
│   └── insights.py           # anomaly detection + LLM explanation (demo)
├── docs/                     # technical + architecture documentation
└── README.md
```

## Output format (`predictions.csv`)

One row per (level, channel, campaign_type, horizon):

| column | meaning |
|--------|---------|
| `level` | `total`, `channel`, or `channel_type` |
| `channel` | `google` / `bing` / `ALL` |
| `campaign_type` | e.g. `search`, `performance_max`, `ALL` |
| `horizon_days` | 30 / 60 / 90 |
| `planned_spend` | assumed spend over the horizon |
| `revenue_p10/p50/p90` | revenue forecast range |
| `roas_p10/p50/p90` | ROAS forecast range |

## Method (short version)

- **Unification:** three platform CSVs are normalized to one schema (Google
  spend converted from micros; campaign-type spellings unified).
- **Features:** daily data aggregated to monthly buckets per channel x campaign
  type; lag, rolling-mean, and seasonality (holiday) features; no leakage.
- **Model:** three gradient-boosted quantile regressors (P10/P50/P90) on
  `log1p(revenue)`, pooled across channels with channel/type one-hot features.
- **Forecast:** recursive monthly roll-forward to 30/60/90 days; ROAS derived as
  revenue / spend; rolled up to channel and total.
- **Validation:** time-based backtest. At the **total** (scored) level: ~13%
  MAPE with well-calibrated P10-P90 coverage. Finer grains are noisier and are
  communicated as honest, wider ranges.

## Assumptions & limitations

- **Meta has no revenue or campaign-type data**, so revenue forecasts cover the
  channels that report revenue (Google, Bing). Meta is treated as spend-only.
- Forecasts are **aggregate-period** (not daily) and **probabilistic ranges**,
  per the brief.
- Existing channel attribution is treated as the source of truth.

## Reproducibility

- Random seeds fixed (`random_state=42`).
- No absolute paths; no network calls during the scored run.
- The LLM insight layer (separate from `run.sh`) requires an API key supplied
  via a local `.env` and is used only in the interactive demo.

Python version: **3.11**
