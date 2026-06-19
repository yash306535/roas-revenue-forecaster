"""
Feature engineering.

Turns the clean long table (from data_loader) into a monthly "panel" where each
row carries CLUES (features) built only from the PAST, plus the ANSWERS
(targets) we want to learn to predict.

Key ideas (see our notes):
  * aggregate noisy daily rows into steadier MONTHLY buckets
  * seasonality clues  -> month, quarter, holiday flag
  * lag clues          -> revenue/spend/roas from previous months
  * rolling clues      -> 3-month smoothed averages
  * NO leakage         -> revenue/roas clues come only from earlier months;
                          the only "current period" input is SPEND, which is a
                          budget DECISION you control (this is what powers
                          budget simulation later).
"""

import numpy as np
import pandas as pd

# Which lag distances (in months) to build for the history clues.
LAG_PERIODS = [1, 2, 3]
ROLLING_WINDOW = 3

# Months we treat as the e-commerce holiday surge.
HOLIDAY_MONTHS = {11, 12}


def _safe_roas(revenue, spend):
    """revenue / spend, but avoid divide-by-zero blowing up."""
    return np.where(spend > 0, revenue / spend.replace(0, np.nan), np.nan)


def filter_complete_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the trailing PARTIAL month.

    Data often ends mid-month (e.g. data stops on the 4th). That stub month
    looks like a catastrophic revenue collapse and wrecks both the lag features
    and any forecast. We keep only months that are fully covered, judged by the
    latest date present in the data.
    """
    df = df.copy()
    max_date = df["date"].max()
    month_end = max_date + pd.offsets.MonthEnd(0)
    if max_date < month_end:
        # the month containing max_date is incomplete -> drop it
        partial = max_date.to_period("M")
        df = df[df["date"].dt.to_period("M") != partial]
    return df


def aggregate_monthly(df: pd.DataFrame, group_keys) -> pd.DataFrame:
    """Sum daily rows up into one row per (group, month)."""
    df = filter_complete_months(df)
    df = df.copy()
    df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()  # first of month

    # min_count=1 => a month where EVERY value is NaN stays NaN (not 0).
    # This keeps Meta's missing revenue as "missing" instead of a fake $0.
    summed = (
        df.groupby(group_keys + ["period"], dropna=False)[
            ["spend", "revenue", "clicks", "impressions", "conversions"]
        ]
        .sum(min_count=1)
        .reset_index()
    )

    # efficiency metric
    summed["roas"] = _safe_roas(summed["revenue"], summed["spend"])
    return summed


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonality clues drawn from the calendar."""
    df = df.copy()
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["is_holiday_season"] = df["month"].isin(HOLIDAY_MONTHS).astype(int)
    return df


def add_history_features(df: pd.DataFrame, group_keys) -> pd.DataFrame:
    """
    Lag + rolling clues, computed WITHIN each series (each group), in time order.
    These describe the recent past so the model can extend the trend.
    """
    df = df.sort_values(group_keys + ["period"]).copy()
    grp = df.groupby(group_keys, dropna=False)

    for col in ["revenue", "spend", "roas"]:
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = grp[col].shift(lag)
        # rolling mean of the PAST window only (shift(1) first to exclude current)
        df[f"{col}_roll{ROLLING_WINDOW}"] = (
            grp[col].shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
    return df


def get_feature_columns():
    """The exact list of clue columns the model will read."""
    cols = ["month", "quarter", "is_holiday_season", "spend"]
    for base in ["revenue", "spend", "roas"]:
        for lag in LAG_PERIODS:
            cols.append(f"{base}_lag{lag}")
        cols.append(f"{base}_roll{ROLLING_WINDOW}")
    return cols


def build_panel(df: pd.DataFrame, group_keys=("channel", "campaign_type")) -> pd.DataFrame:
    """
    Full feature panel for one grain (default: channel + campaign_type).
    Returns every monthly row with features + targets (revenue, roas).
    Rows with no revenue (e.g. Meta) keep NaN targets and get dropped at
    training time.
    """
    group_keys = list(group_keys)
    panel = aggregate_monthly(df, group_keys)
    panel = add_calendar_features(panel)
    panel = add_history_features(panel, group_keys)

    # targets = the current month's actual revenue / roas (what we learn to hit)
    panel["target_revenue"] = panel["revenue"]
    panel["target_roas"] = panel["roas"]
    return panel


def training_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows usable for supervised training:
      * must have a real revenue target (drops Meta / no-revenue rows)
      * must have its lag clues filled (drops the earliest months of each series)
    """
    feature_cols = get_feature_columns()
    needed = feature_cols + ["target_revenue"]
    frame = panel.dropna(subset=needed).copy()
    # a valid ROAS target also needs positive spend
    frame = frame[frame["target_revenue"].notna()]
    return frame
