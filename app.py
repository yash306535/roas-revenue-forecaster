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
from forecast import run_forecast, summarize          # noqa: E402
from insights import compute_facts, detect_anomalies, generate_insights  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Revenue & ROAS Forecast", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#f7f9fc 0%, #ffffff 320px); }
      .block-container { padding-top: 2rem; max-width: 1250px; }
      .hero {
        background: linear-gradient(110deg,#ff6b35 0%, #ff9558 55%, #ffb27a 100%);
        border-radius: 18px; padding: 26px 30px; color: #fff;
        box-shadow: 0 10px 30px rgba(255,107,53,.25); margin-bottom: 22px;
      }
      .hero h1 { font-size: 1.9rem; margin: 0; font-weight: 800; letter-spacing:-.5px; }
      .hero p  { margin: 6px 0 0; opacity: .95; font-size: .98rem; }
      .kpi {
        background:#fff; border:1px solid #eef1f6; border-radius:16px; padding:18px 20px;
        box-shadow:0 4px 18px rgba(20,30,60,.05); height:100%;
      }
      .kpi .label { font-size:.8rem; color:#7a8699; text-transform:uppercase; letter-spacing:.6px; font-weight:600; }
      .kpi .value { font-size:1.7rem; font-weight:800; color:#1c2433; margin-top:4px; }
      .kpi .range { font-size:.85rem; color:#ff6b35; font-weight:600; margin-top:2px; }
      .section-title { font-size:1.15rem; font-weight:700; color:#1c2433; margin:8px 0 4px; }
      .insight-card {
        background:#fff; border:1px solid #eef1f6; border-left:5px solid #ff6b35;
        border-radius:14px; padding:20px 24px; box-shadow:0 6px 22px rgba(20,30,60,.06);
        line-height:1.6;
      }
      .insight-card h3 { margin-top:0; color:#ff6b35; }
      .pill { display:inline-block; background:#fff4ee; color:#d9531e; border:1px solid #ffd9c4;
              padding:6px 12px; border-radius:20px; font-size:.85rem; margin:3px 6px 3px 0; font-weight:600; }
      .keyok { color:#1a8a4a; font-weight:600; }
      .keyno { color:#c0392b; font-weight:600; }
      div[data-testid="stMetric"] { display:none; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_panel(data_dir="./data"):
    return build_panel(load_unified_data(data_dir), group_keys=("channel", "campaign_type"))


@st.cache_resource(show_spinner=False)
def load_model(model_path="./pickle/model.pkl"):
    return joblib.load(model_path)


def monthly_history(panel):
    h = panel[panel["revenue"].notna()]
    m = h.groupby("period").agg(revenue=("revenue", "sum"), spend=("spend", "sum")).reset_index()
    return m.sort_values("period")


def forecast_by_month(forecast_long):
    return (
        forecast_long.groupby("period")
        .agg(p10=("revenue_p10", "sum"), p50=("revenue_p50", "sum"), p90=("revenue_p90", "sum"))
        .reset_index().sort_values("period")
    )


def kpi(label, value, rng=None):
    r = f'<div class="range">{rng}</div>' if rng else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value">{value}</div>{r}</div>', unsafe_allow_html=True)


def md_to_html(text: str) -> str:
    """Light markdown -> HTML so formatting survives inside a styled card."""
    import re

    html_lines, in_list = [], False
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"(?<!\*)\*(?!\s)(.+?)\*", r"<em>\1</em>", line)
        line = re.sub(r"^#{1,4}\s*", "", line)
        if line.startswith(("- ", "* ", "• ")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:].strip()}</li>")
        elif not line:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{line}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "".join(html_lines)


# --------------------------------------------------------------------------- data
panel = load_panel()
artifact = load_model()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    horizon = st.radio("Planning window", [30, 60, 90], index=2, horizontal=True)
    horizon_m = {30: 1, 60: 2, 90: 3}[horizon]

    st.markdown("#### 💰 Budget simulation")
    multiplier = st.slider("Scale planned spend", 0.5, 2.0, 1.0, 0.05,
                           help="Simulate spending more or less than the recent average.")
    st.caption(f"Spending at **{multiplier*100:.0f}%** of recent average")

    st.markdown("#### 🤖 AI insights")
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        st.markdown('<span class="keyok">🔑 API key detected (.env)</span>', unsafe_allow_html=True)
        api_key = env_key
    else:
        st.markdown('<span class="keyno">No key in .env</span>', unsafe_allow_html=True)
        api_key = st.text_input("Paste an OpenAI key (optional)", type="password",
                                value="", placeholder="sk-...",
                                help="Not stored. Leave blank to use the rule-based summary.")
    st.caption("Without a key, a clear rule-based summary is shown instead.")

# --------------------------------------------------------------------------- forecast
forecast_long = run_forecast(artifact, panel, spend_multiplier=multiplier)
predictions = summarize(forecast_long)
tot = predictions[(predictions.level == "total") & (predictions.horizon_days == horizon)].iloc[0]

# --------------------------------------------------------------------------- hero
st.markdown(
    f"""
    <div class="hero">
      <h1>📈 Probabilistic Revenue &amp; ROAS Forecast</h1>
      <p>Multi-channel e-commerce forecasting with uncertainty ranges, budget simulation, and AI insights ·
         <b>{horizon}-day outlook</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- KPIs
c1, c2, c3 = st.columns(3)
with c1:
    kpi(f"{horizon}-day revenue (P50)", f"${tot.revenue_p50:,.0f}",
        f"range ${tot.revenue_p10:,.0f} – ${tot.revenue_p90:,.0f}")
with c2:
    kpi(f"{horizon}-day blended ROAS", f"{tot.roas_p50:.2f}",
        f"range {tot.roas_p10:.2f} – {tot.roas_p90:.2f}")
with c3:
    kpi("Planned spend", f"${tot.planned_spend:,.0f}",
        f"{multiplier*100:.0f}% of recent average")

st.write("")

# --------------------------------------------------------------------------- chart
st.markdown('<div class="section-title">Revenue — history &amp; forecast range</div>', unsafe_allow_html=True)
hist = monthly_history(panel)
fc = forecast_by_month(forecast_long).head(horizon_m)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["period"], y=hist["revenue"], name="Actual",
                         mode="lines+markers", line=dict(color="#2b6cb0", width=2.5)))
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p90"], mode="lines",
                         line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p10"], name="P10–P90 range", mode="lines",
                         line=dict(width=0), fill="tonexty", fillcolor="rgba(255,107,53,0.18)"))
fig.add_trace(go.Scatter(x=fc["period"], y=fc["p50"], name="Forecast (P50)", mode="lines+markers",
                         line=dict(color="#ff6b35", width=2.5, dash="dash")))
fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                  yaxis_title="Revenue ($)", hovermode="x unified",
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#eef1f6")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- breakdowns
left, right = st.columns(2)
with left:
    st.markdown('<div class="section-title">By channel</div>', unsafe_allow_html=True)
    ch = predictions[(predictions.level == "channel") & (predictions.horizon_days == horizon)]
    st.dataframe(
        ch[["channel", "planned_spend", "revenue_p50", "roas_p50"]]
        .rename(columns={"planned_spend": "spend", "revenue_p50": "revenue (P50)", "roas_p50": "ROAS (P50)"}),
        use_container_width=True, hide_index=True,
    )
with right:
    st.markdown('<div class="section-title">By campaign type</div>', unsafe_allow_html=True)
    ct = predictions[(predictions.level == "channel_type") & (predictions.horizon_days == horizon)]
    st.dataframe(
        ct[["channel", "campaign_type", "revenue_p50", "roas_p50"]]
        .rename(columns={"revenue_p50": "revenue (P50)", "roas_p50": "ROAS (P50)"}),
        use_container_width=True, hide_index=True,
    )

st.write("")

# --------------------------------------------------------------------------- AI insights
st.markdown('<div class="section-title">🤖 AI-generated insights</div>', unsafe_allow_html=True)
if st.button("✨ Generate insights", type="primary"):
    with st.spinner("Analyzing forecast and detecting anomalies..."):
        facts = compute_facts(panel, predictions)
        anomalies = detect_anomalies(panel)
        text = generate_insights(facts, anomalies, api_key=api_key or None)

    st.markdown(f'<div class="insight-card"><h3>📋 Forecast briefing</h3>{md_to_html(text)}</div>',
                unsafe_allow_html=True)

    if anomalies:
        st.markdown('<div class="section-title">⚠️ Detected risks</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="pill">{a}</span>' for a in anomalies),
                    unsafe_allow_html=True)

    with st.expander("Show the underlying facts the AI was given"):
        st.json(facts)
else:
    st.caption("Click to generate a plain-English summary of the forecast, its drivers, and key risks.")
