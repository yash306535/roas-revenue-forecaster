"""
Interactive demo (Streamlit).

  streamlit run app.py

Shows the probabilistic forecast, lets you simulate budgets, and generates
AI insights. This is the DEMO layer -- it is independent of run.sh / scoring.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_unified_data            # noqa: E402
from features import build_panel                      # noqa: E402
from forecast import run_forecast, summarize, HORIZON_DAYS  # noqa: E402
from insights import compute_facts, detect_anomalies, generate_insights  # noqa: E402

load_dotenv()

st.set_page_config(page_title="E-commerce Revenue Forecast", layout="wide")


@st.cache_data(show_spinner=False)
def load_panel(data_dir="./data"):
    unified = load_unified_data(data_dir)
    return build_panel(unified, group_keys=("channel", "campaign_type"))


@st.cache_resource(show_spinner=False)
def load_model(model_path="./pickle/model.pkl"):
    return joblib.load(model_path)


def monthly_history(panel):
    h = panel[panel["revenue"].notna()]
    m = h.groupby("period").agg(revenue=("revenue", "sum"), spend=("spend", "sum")).reset_index()
    m["roas"] = np.where(m["spend"] > 0, m["revenue"] / m["spend"], np.nan)
    return m.sort_values("period")


def forecast_by_month(forecast_long):
    """Total revenue quantiles per future month (for the band chart)."""
    return (
        forecast_long.groupby("period")
        .agg(p10=("revenue_p10", "sum"), p50=("revenue_p50", "sum"), p90=("revenue_p90", "sum"))
        .reset_index()
        .sort_values("period")
    )


# --------------------------------------------------------------------------- UI
st.title("📈 Probabilistic Revenue & ROAS Forecast")
st.caption("E-commerce marketing forecasting utility — revenue & ROAS ranges, budget simulation, and AI insights.")

panel = load_panel()
artifact = load_model()

with st.sidebar:
    st.header("Controls")
    horizon = st.radio("Planning window", [30, 60, 90], index=2, horizontal=True)
    horizon_m = {30: 1, 60: 2, 90: 3}[horizon]

    st.subheader("Budget simulation")
    multiplier = st.slider("Scale planned spend", 0.5, 2.0, 1.0, 0.05,
                           help="Simulate spending more or less than the recent average.")
    st.caption(f"Spending at **{multiplier:.0f}%**" if multiplier == 1 else f"Spending at **{multiplier*100:.0f}%** of recent average")

    st.subheader("AI insights")
    api_key = st.text_input("OpenAI API key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""),
                            help="Optional. Without a key, a rule-based summary is shown.")

# run forecast at the chosen budget
forecast_long = run_forecast(artifact, panel, spend_multiplier=multiplier)
predictions = summarize(forecast_long)

# headline numbers at the selected horizon
tot = predictions[(predictions.level == "total") & (predictions.horizon_days == horizon)].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric(f"{horizon}-day revenue (P50)", f"${tot.revenue_p50:,.0f}",
          help=f"Range ${tot.revenue_p10:,.0f} – ${tot.revenue_p90:,.0f}")
c2.metric(f"{horizon}-day blended ROAS (P50)", f"{tot.roas_p50:.2f}",
          help=f"Range {tot.roas_p10:.2f} – {tot.roas_p90:.2f}")
c3.metric("Planned spend", f"${tot.planned_spend:,.0f}")

st.markdown(f"**Expected revenue range:** ${tot.revenue_p10:,.0f} → **${tot.revenue_p50:,.0f}** → ${tot.revenue_p90:,.0f}")

# --------------------------------------------------------------------------- chart
st.subheader("Revenue: history & forecast range")
hist = monthly_history(panel)
fc = forecast_by_month(forecast_long).head(horizon_m)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["period"], y=hist["revenue"], name="Actual revenue",
                         mode="lines+markers", line=dict(color="#1f77b4")))
# forecast band
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p90"], name="P90", mode="lines",
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p10"], name="P10–P90 range", mode="lines",
                         line=dict(width=0), fill="tonexty", fillcolor="rgba(255,127,14,0.25)"))
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p50"], name="Forecast (P50)", mode="lines+markers",
                         line=dict(color="#ff7f0e", dash="dash")))
fig.update_layout(height=380, margin=dict(t=10), yaxis_title="Revenue ($)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- breakdowns
left, right = st.columns(2)
with left:
    st.subheader("By channel")
    ch = predictions[(predictions.level == "channel") & (predictions.horizon_days == horizon)]
    st.dataframe(
        ch[["channel", "planned_spend", "revenue_p10", "revenue_p50", "revenue_p90",
            "roas_p10", "roas_p50", "roas_p90"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )
with right:
    st.subheader("By campaign type")
    ct = predictions[(predictions.level == "channel_type") & (predictions.horizon_days == horizon)]
    st.dataframe(
        ct[["channel", "campaign_type", "revenue_p50", "roas_p50"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------- AI insights
st.subheader("🤖 AI-generated insights")
if st.button("Generate insights"):
    with st.spinner("Analyzing forecast and detecting anomalies..."):
        facts = compute_facts(panel, predictions)
        anomalies = detect_anomalies(panel)
        text = generate_insights(facts, anomalies, api_key=api_key or None)
    st.markdown(text)
    with st.expander("Show the underlying facts the AI was given"):
        st.json(facts)
        st.write("Anomalies:", anomalies)
else:
    st.caption("Click to generate a plain-English summary of the forecast, drivers, and risks.")
