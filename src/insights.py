"""
AI-assisted insight layer.

This is SEPARATE from run.sh (the scored pipeline must not call the internet).
It runs only in the interactive demo. We:
  1. compute grounded FACTS from the data + forecast (no hallucination room),
  2. detect simple ANOMALIES with plain rules,
  3. ask an LLM to explain/advise based ONLY on those facts.

If no API key is available (or the call fails) we fall back to a clear,
rule-based summary so the app always works.
"""

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Grounded facts
# ---------------------------------------------------------------------------
def compute_facts(panel: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Pull concrete numbers the LLM is allowed to talk about."""
    have_rev = panel[panel["revenue"].notna()].copy()
    monthly = (
        have_rev.groupby("period")
        .agg(revenue=("revenue", "sum"), spend=("spend", "sum"))
        .reset_index()
        .sort_values("period")
    )
    monthly["roas"] = np.where(monthly["spend"] > 0, monthly["revenue"] / monthly["spend"], np.nan)

    facts = {}
    if len(monthly) >= 2:
        last, prev = monthly.iloc[-1], monthly.iloc[-2]
        facts["last_month"] = str(last["period"].date())
        facts["last_month_revenue"] = round(float(last["revenue"]), 0)
        facts["prev_month_revenue"] = round(float(prev["revenue"]), 0)
        facts["revenue_change_pct"] = round(
            float((last["revenue"] - prev["revenue"]) / prev["revenue"] * 100) if prev["revenue"] else 0.0, 1
        )
        facts["last_month_roas"] = round(float(last["roas"]), 2)
        facts["prev_month_roas"] = round(float(prev["roas"]), 2)

    # per-channel recent ROAS
    ch_roas = {}
    for ch, g in have_rev.groupby("channel"):
        gm = g.groupby("period").agg(r=("revenue", "sum"), s=("spend", "sum")).sort_index()
        if len(gm) >= 1 and gm["s"].iloc[-1] > 0:
            ch_roas[ch] = round(float(gm["r"].iloc[-1] / gm["s"].iloc[-1]), 2)
    facts["recent_roas_by_channel"] = ch_roas

    # the 90-day total forecast
    tot = predictions[(predictions.level == "total") & (predictions.horizon_days == 90)]
    if not tot.empty:
        r = tot.iloc[0]
        facts["forecast_90d_revenue"] = [round(r.revenue_p10), round(r.revenue_p50), round(r.revenue_p90)]
        facts["forecast_90d_roas"] = [round(r.roas_p10, 2), round(r.roas_p50, 2), round(r.roas_p90, 2)]

    return facts


# ---------------------------------------------------------------------------
# 2. Anomaly detection (simple, explainable rules)
# ---------------------------------------------------------------------------
def detect_anomalies(panel: pd.DataFrame) -> list[str]:
    flags = []

    # Meta: spending money with no attributed revenue
    meta = panel[panel["channel"] == "meta"]
    if not meta.empty and meta["spend"].sum() > 0 and meta["revenue"].isna().all():
        flags.append(
            f"Meta spent ~${meta['spend'].sum():,.0f} total but reports no revenue "
            f"attribution, so it is excluded from the revenue forecast."
        )

    have_rev = panel[panel["revenue"].notna()]
    for ch, g in have_rev.groupby("channel"):
        gm = g.groupby("period").agg(r=("revenue", "sum"), s=("spend", "sum")).sort_index()
        if len(gm) >= 2:
            gm["roas"] = np.where(gm["s"] > 0, gm["r"] / gm["s"], np.nan)
            last, prev = gm["roas"].iloc[-1], gm["roas"].iloc[-2]
            if prev and not np.isnan(last) and not np.isnan(prev):
                change = (last - prev) / prev * 100
                if change <= -25:
                    flags.append(f"{ch.title()} ROAS dropped {abs(change):.0f}% last month ({prev:.1f} -> {last:.1f}).")
                elif change >= 40:
                    flags.append(f"{ch.title()} ROAS jumped {change:.0f}% last month ({prev:.1f} -> {last:.1f}).")
            # spend up but revenue down
            if len(gm) >= 2 and gm["s"].iloc[-1] > gm["s"].iloc[-2] * 1.1 and gm["r"].iloc[-1] < gm["r"].iloc[-2] * 0.9:
                flags.append(f"{ch.title()} spend rose while revenue fell last month — efficiency warning.")

    return flags


# ---------------------------------------------------------------------------
# 3. The LLM call (with safe fallback)
# ---------------------------------------------------------------------------
def _fallback_summary(facts: dict, anomalies: list[str]) -> str:
    lines = ["**Forecast summary (rule-based — no LLM key set):**", ""]
    if "forecast_90d_revenue" in facts:
        p10, p50, p90 = facts["forecast_90d_revenue"]
        lines.append(f"- Next 90 days: expected revenue ~${p50:,.0f} (range ${p10:,.0f}–${p90:,.0f}).")
    if "forecast_90d_roas" in facts:
        lo, mid, hi = facts["forecast_90d_roas"]
        lines.append(f"- Expected blended ROAS ~{mid} (range {lo}–{hi}).")
    if "revenue_change_pct" in facts:
        lines.append(f"- Last month revenue changed {facts['revenue_change_pct']}% vs the prior month.")
    if anomalies:
        lines.append("")
        lines.append("**Risks / anomalies:**")
        for a in anomalies:
            lines.append(f"- {a}")
    return "\n".join(lines)


def generate_insights(facts: dict, anomalies: list[str], api_key: str | None = None,
                      model: str = "gpt-4o-mini") -> str:
    """Ask the LLM to explain the forecast using ONLY the supplied facts."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_summary(facts, anomalies)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a marketing analytics advisor for a digital agency. "
            "Using ONLY the facts below, write a concise, business-focused summary "
            "(4-6 short bullet points) covering: the 90-day revenue and ROAS outlook, "
            "what is driving it, and the key operational risks. Do not invent numbers.\n\n"
            f"FACTS:\n{facts}\n\nANOMALIES DETECTED:\n{anomalies}\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # network/key/quota issues -> never crash the demo
        return _fallback_summary(facts, anomalies) + f"\n\n_(LLM unavailable: {e})_"
