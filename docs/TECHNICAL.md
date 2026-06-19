# Technical Documentation

## 1. Problem framing

Forecast e-commerce **revenue** and **ROAS** as probabilistic ranges
(P10 / P50 / P90) over 30 / 60 / 90-day windows, broken down by channel and
campaign type, with support for future-budget simulation and AI-assisted
explanation.

We deliberately frame this as an **aggregate-period, probabilistic** problem
(per the brief): the business decision needs a credible *range* over a planning
window, not a precise daily point estimate.

## 2. Data & preprocessing

Three platform exports (Google Ads, Microsoft/Bing Ads, Meta Ads), each with a
different schema. They are normalized to one canonical schema via per-platform
**adapters**:

| Issue | Handling |
|-------|----------|
| Different column names | adapter renames to canonical schema |
| Google spend in **micros** | divided by 1,000,000 to dollars |
| Campaign-type spellings (`PERFORMANCE_MAX` vs `PerformanceMax`) | mapped to one label |
| **Meta has no revenue / type** | revenue left missing; Meta excluded from revenue model (treated spend-only) |
| Trailing **partial month** | dropped (`filter_complete_months`) — this was the single biggest accuracy fix |
| All-missing monthly sums | `min_count=1` keeps them missing instead of a fake `$0` |

Platform detection is by **column signature**, not filename, so renamed test
files still route correctly.

## 3. Feature engineering

Daily rows are aggregated to **monthly** buckets per (channel, campaign_type),
then enriched with clues built **only from the past** (no leakage):

- **Seasonality:** month, quarter, holiday-season flag (Nov/Dec).
- **Lags:** revenue / spend / ROAS at t-1, t-2, t-3.
- **Rolling:** 3-month trailing mean of revenue / spend / ROAS.
- **Spend / log(spend):** current-period spend is an allowed input (a budget
  decision); `log1p(spend)` lets the model express diminishing returns.
- **Categoricals:** channel and campaign_type one-hot encoded.

## 4. Model selection

**Chosen:** three gradient-boosted regressors (`GradientBoostingRegressor`,
`loss="quantile"`, alpha = 0.1 / 0.5 / 0.9) trained on `log1p(revenue)`.

Rationale:
- **Quantile loss** directly produces calibrated P10/P50/P90 — the required
  probabilistic output — without distributional assumptions.
- **Gradient boosting** captures non-linear spend response and seasonality
  interactions that a linear model would miss.
- **Pooled** single model with channel/type features (rather than one model per
  channel) because the dataset is small (~102 trainable monthly rows); pooling
  shares signal and reduces variance.
- Trees kept **shallow and regularized** (`max_depth=2`, `min_samples_leaf=5`,
  `subsample=0.8`, 150 estimators) to resist overfitting on small data.

ROAS is **derived** as predicted revenue / planned spend per quantile (spend is
known/decided), keeping revenue and ROAS internally consistent. Quantile
crossing is removed by sorting P10 <= P50 <= P90 per row.

## 5. Forecasting procedure

Recursive monthly roll-forward: build the next month's feature row from recent
actuals + known calendar + assumed spend, predict, then feed the P50 back as the
lag for the following month. Months are summed into 30/60/90-day horizons and
rolled up to channel and total.

## 6. Validation (backtest)

Time-based split (train on earlier months, test on the most recent). Metrics at
three grains:

| Grain | MAPE (P50) | P10–P90 coverage |
|-------|-----------|------------------|
| **Total (scored)** | **~12.8%** | ~100% |
| Per channel | ~152% | ~75% |
| Per cell (channelxtype) | ~84% | ~67% |

The total/blended forecast — the primary business and scoring metric — is
accurate and well-calibrated. Finer grains are inherently noisy (e.g. Bing
revenue swings 10x+ month to month); we communicate this honestly as wider
ranges rather than overstating precision.

## 7. AI integration

A grounded LLM layer (OpenAI), separate from the scored pipeline:
- compute concrete **facts** (recent revenue/ROAS, per-channel ROAS, the 90-day
  forecast),
- detect **anomalies** with explainable rules (ROAS swings, spend-up/revenue-down,
  Meta no-revenue),
- prompt the LLM to explain drivers and risks **using only those facts**.

A rule-based fallback guarantees the demo works with no key, and any API error
degrades gracefully.

## 8. Assumptions & limitations

- Meta is spend-only (no revenue attribution in the data).
- Aggregate-period, probabilistic forecasts (not daily, not deterministic).
- Existing channel attribution is the source of truth.
- Horizon quantiles are summed across months (a practical approximation of the
  horizon interval).
- Small data caps fine-grained accuracy; the design prioritizes a calibrated
  total forecast plus honest uncertainty.
