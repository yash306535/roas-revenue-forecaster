"""
Forecasting roll-forward + budget simulation.

We have no "future rows", so we build them: known calendar + known recent lags +
an assumed SPEND (the budget decision). We predict month 1, feed that prediction
back as the lag for month 2 (recursive forecasting), and so on for 3 months.
Then we aggregate into 30/60/90-day horizons, derive ROAS, and roll up to
channel and total.

Meta has no revenue, so (per our documented Option A) revenue forecasting covers
the channels that actually report revenue (Google, Bing).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from features import HOLIDAY_MONTHS, LAG_PERIODS, ROLLING_WINDOW
from model import build_design_matrix, predict_quantiles

HORIZON_MONTHS = 3                 # we forecast 3 months -> 30/60/90 days
HORIZON_DAYS = {1: 30, 2: 60, 3: 90}


def _recent_history(panel: pd.DataFrame):
    """
    For each (channel, campaign_type) group that reports revenue, collect the
    recent monthly actuals we need to seed the forecast.
    Returns: dict[group] -> {rev: [...], spend: [...], roas: [...], last_period}
    """
    have_rev = panel[panel["revenue"].notna()].copy()
    have_rev = have_rev.sort_values("period")
    hist = {}
    for (ch, ct), g in have_rev.groupby(["channel", "campaign_type"]):
        hist[(ch, ct)] = {
            "rev": g["revenue"].tolist(),
            "spend": g["spend"].tolist(),
            "roas": g["roas"].tolist(),
            "last_period": g["period"].max(),
        }
    return hist


def _lag(values, k):
    """k-th lag from the end of a list, padding with the earliest value if short."""
    if len(values) >= k:
        return values[-k]
    return values[0] if values else 0.0


def _roll(values, w=ROLLING_WINDOW):
    """mean of the last w values (the 'past window')."""
    tail = values[-w:] if values else [0.0]
    return float(np.mean(tail)) if tail else 0.0


def _planned_spend(spend_hist, override, multiplier=1.0):
    """Default future monthly spend = avg of last 3 months, unless overridden."""
    if override is not None:
        return float(override) * multiplier
    return _roll(spend_hist, 3) * multiplier


def run_forecast(artifact: dict, panel: pd.DataFrame, spend_overrides: dict | None = None,
                 spend_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Recursive monthly forecast for each revenue-reporting group.

    spend_overrides: optional {(channel, campaign_type): monthly_spend} to
    simulate a different budget. Missing groups use their recent-average spend.
    spend_multiplier: scale all planned spend (the budget slider in the UI).

    Returns a long dataframe: one row per (group, month_ahead) with quantiles.
    """
    models = artifact["quantile_models"]
    feature_names = artifact["feature_names"]
    spend_overrides = spend_overrides or {}

    hist = _recent_history(panel)
    # working copies we extend as we walk forward
    work = {g: {k: list(v[k]) if isinstance(v[k], list) else v[k] for k in v} for g, v in hist.items()}

    rows = []
    for month_ahead in range(1, HORIZON_MONTHS + 1):
        # build one feature row per group for this future month
        batch = []
        meta = []
        for g, h in work.items():
            ch, ct = g
            future_period = h["last_period"] + pd.offsets.MonthBegin(month_ahead)
            spend = _planned_spend(h["spend"], spend_overrides.get(g), spend_multiplier)
            row = {
                "channel": ch,
                "campaign_type": ct,
                "spend": spend,
                "month": future_period.month,
                "quarter": (future_period.month - 1) // 3 + 1,
                "is_holiday_season": int(future_period.month in HOLIDAY_MONTHS),
            }
            for col, series in [("revenue", h["rev"]), ("spend", h["spend"]), ("roas", h["roas"])]:
                for lag in LAG_PERIODS:
                    row[f"{col}_lag{lag}"] = _lag(series, lag)
                row[f"{col}_roll{ROLLING_WINDOW}"] = _roll(series)
            # NOTE: spend lag/roll use actual spend history; current spend is the plan
            batch.append(row)
            meta.append((g, future_period, spend))

        bdf = pd.DataFrame(batch)
        X, _ = build_design_matrix(bdf)
        preds = predict_quantiles(models, X, feature_names)

        # record + push predictions back into history for the next month's lags
        for i, (g, future_period, spend) in enumerate(meta):
            p10, p50, p90 = preds.iloc[i][["p10", "p50", "p90"]]
            rows.append({
                "channel": g[0], "campaign_type": g[1],
                "month_ahead": month_ahead, "period": future_period,
                "spend": spend,
                "revenue_p10": p10, "revenue_p50": p50, "revenue_p90": p90,
            })
            # recursive update: use P50 as the realized revenue for lagging
            work[g]["rev"].append(float(p50))
            work[g]["spend"].append(float(spend))
            work[g]["roas"].append(float(p50 / spend) if spend > 0 else 0.0)

    return pd.DataFrame(rows)


def _add_roas(df, rev_col, spend_col, out_prefix):
    df[f"{out_prefix}_p10"] = np.where(df[spend_col] > 0, df["revenue_p10"] / df[spend_col], np.nan)
    df[f"{out_prefix}_p50"] = np.where(df[spend_col] > 0, df["revenue_p50"] / df[spend_col], np.nan)
    df[f"{out_prefix}_p90"] = np.where(df[spend_col] > 0, df["revenue_p90"] / df[spend_col], np.nan)
    return df


def summarize(forecast_long: pd.DataFrame) -> pd.DataFrame:
    """
    Roll the monthly group forecast into the deliverable table:
    levels (total / channel / channel+type) x horizons (30/60/90 days),
    with revenue and ROAS ranges.
    """
    results = []

    for horizon_m in [1, 2, 3]:
        days = HORIZON_DAYS[horizon_m]
        window = forecast_long[forecast_long["month_ahead"] <= horizon_m]

        rev_cols = ["revenue_p10", "revenue_p50", "revenue_p90"]
        agg = {c: "sum" for c in rev_cols}
        agg["spend"] = "sum"

        # level 1: campaign_type within channel
        lvl_ct = window.groupby(["channel", "campaign_type"]).agg(agg).reset_index()
        lvl_ct["level"] = "channel_type"

        # level 2: channel
        lvl_ch = window.groupby(["channel"]).agg(agg).reset_index()
        lvl_ch["campaign_type"] = "ALL"
        lvl_ch["level"] = "channel"

        # level 3: grand total (blended)
        lvl_tot = window.agg(agg).to_frame().T
        lvl_tot["channel"] = "ALL"
        lvl_tot["campaign_type"] = "ALL"
        lvl_tot["level"] = "total"

        for part in (lvl_ct, lvl_ch, lvl_tot):
            part["horizon_days"] = days
            results.append(part)

    out = pd.concat(results, ignore_index=True)
    out = _add_roas(out, "revenue", "spend", "roas")
    out = out.rename(columns={"spend": "planned_spend"})
    cols = [
        "level", "channel", "campaign_type", "horizon_days", "planned_spend",
        "revenue_p10", "revenue_p50", "revenue_p90",
        "roas_p10", "roas_p50", "roas_p90",
    ]
    out = out[cols].sort_values(["horizon_days", "level", "channel", "campaign_type"]).reset_index(drop=True)
    # round for readability
    money = ["planned_spend", "revenue_p10", "revenue_p50", "revenue_p90"]
    out[money] = out[money].round(2)
    roas = ["roas_p10", "roas_p50", "roas_p90"]
    out[roas] = out[roas].round(4)
    return out
