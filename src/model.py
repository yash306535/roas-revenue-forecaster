"""
The probabilistic forecasting model.

We train THREE small gradient-boosted models, one per quantile (P10/P50/P90),
each using the lopsided "quantile loss" so it aims low / middle / high. Together
they form the prediction RANGE.

Design choices for our tiny (~108 row) dataset:
  * one POOLED model with channel + campaign_type one-hot encoded as features
  * log1p(revenue) target so the skew is tamed and spend shows diminishing
    returns naturally (we also add log1p(spend) as a feature)
  * shallow, regularized trees to resist overfitting
  * predictions sorted so P10 <= P50 <= P90 (no "quantile crossing")
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import get_feature_columns

QUANTILES = [0.1, 0.5, 0.9]
RANDOM_SEED = 42

# the categorical clues we one-hot encode, and their full set of possible values
# (fixed here so train-time and predict-time columns always line up)
CHANNELS = ["google", "bing", "meta"]
CAMPAIGN_TYPES = [
    "search", "performance_max", "shopping", "video",
    "display", "demand_gen", "audience", "unknown",
]


def build_design_matrix(panel: pd.DataFrame):
    """
    Turn the feature panel into the numeric matrix X the model reads.
    Returns (X dataframe, list_of_feature_names).
    """
    df = panel.copy()

    # numeric clues from feature engineering + a log(spend) for diminishing returns
    numeric_cols = get_feature_columns()
    df["log_spend"] = np.log1p(df["spend"].clip(lower=0))

    X = df[numeric_cols + ["log_spend"]].copy()

    # one-hot encode channel and campaign_type against the FIXED value lists
    for ch in CHANNELS:
        X[f"channel_{ch}"] = (df["channel"] == ch).astype(int)
    for ct in CAMPAIGN_TYPES:
        X[f"type_{ct}"] = (df["campaign_type"] == ct).astype(int)

    feature_names = list(X.columns)
    return X, feature_names


def _make_regressor(alpha: float) -> GradientBoostingRegressor:
    """One small, regularized quantile regressor."""
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=150,
        max_depth=2,          # shallow trees -> less overfitting
        learning_rate=0.05,
        min_samples_leaf=5,
        subsample=0.8,        # row sampling -> more robust
        random_state=RANDOM_SEED,
    )


def train_quantile_models(X: pd.DataFrame, y_revenue: pd.Series) -> dict:
    """Fit one regressor per quantile on log1p(revenue)."""
    y_log = np.log1p(y_revenue.clip(lower=0))
    models = {}
    for q in QUANTILES:
        reg = _make_regressor(q)
        reg.fit(X, y_log)
        models[q] = reg
    return models


def predict_quantiles(models: dict, X: pd.DataFrame, feature_names) -> pd.DataFrame:
    """
    Predict revenue at each quantile. Returns a dataframe with columns
    p10 / p50 / p90 (in real dollars), with ordering enforced.
    """
    X = X.reindex(columns=feature_names, fill_value=0)  # align to training cols

    preds = {}
    for q in QUANTILES:
        log_pred = models[q].predict(X)
        preds[q] = np.expm1(log_pred).clip(min=0)  # back to dollars, no negatives

    out = pd.DataFrame({
        "p10": preds[0.1],
        "p50": preds[0.5],
        "p90": preds[0.9],
    })

    # enforce P10 <= P50 <= P90 row-wise (fixes any quantile crossing)
    sorted_vals = np.sort(out[["p10", "p50", "p90"]].values, axis=1)
    out["p10"], out["p50"], out["p90"] = sorted_vals[:, 0], sorted_vals[:, 1], sorted_vals[:, 2]
    return out
