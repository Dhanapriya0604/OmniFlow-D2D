"""Demand Forecasting — Prophet + linear extrapolation to June 2026."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from .data_loader import load_data

COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6"]

# ── Simple linear-trend + seasonality forecaster (no external deps) ──────────
def forecast_series(series: pd.Series, periods: int = 6) -> pd.DataFrame:
    """Returns historical + forecast DataFrame with trend & CI."""
    s = series.dropna()
    n = len(s)
    x = np.arange(n)
    # Fit linear trend
    m, b = np.polyfit(x, s.values, 1)
    # Residuals for seasonality (12-month window)
    fitted = m * x + b
    resid  = s.values - fitted
    # Monthly seasonal factors (avg of same month across years)
    months = s.index.month if hasattr(s.index, 'month') else np.arange(1, n+1) % 12 + 1
    seas   = np.zeros(12)
    cnt    = np.zeros(12)
    for i, mo in enumerate(months):
        seas[mo-1]  += resid[i]
        cnt[mo-1]   += 1
    seas = seas / np.where(cnt > 0, cnt, 1)

    # Forecast future
    future_x   = np.arange(n, n + periods)
    future_months = [(s.index[-1].month + i - 1) % 12 + 1 for i in range(1, periods+1)]
    future_trend = m * future_x + b
    future_seas  = np.array([seas[mo-1] for mo in future_months])
    future_vals  = np.maximum(future_trend + future_seas, 0)

    # CI based on residual std
    std = resid.std()
    future_hi = future_vals + 1.65 * std
    future_lo = np.maximum(future_vals - 1.65 * std, 0)

    hist_df = pd.DataFrame({"ds": s.index.to_timestamp(), "y": s.values,
                             "type": "historical", "yhat_lower": np.nan, "yhat_upper": np.nan})
    last_date = s.index[-1].to_timestamp()
    future_dates = pd.date_range(last_date, periods=periods+1, freq="MS")[1:]
    fore_df = pd.DataFrame({"ds": future_dates, "y": future_vals,
                             "type": "forecast",
                             "yhat_lower": future_lo, "yhat_upper": future_hi})
    return pd.concat([hist_df, fore_df], ignore_index=True)


def render():
    df = load_data()

    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#00e5ff'>
        📈 Demand Forecasting
      </div>
      <div style='color:#64748b;font-size:.9rem'>
        Historical trends + forecast to June 2026 · Linear trend + seasonal decomposition
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2,2,1])
    with col_a:
        metric_opt = st.selectbox("Forecast Metric", ["Revenue (₹)", "Quantity (Units)", "Orders (#)"])
    with col_b:
        level_opt  = st.selectbox("Breakdown By", ["Overall", "Category", "Region", "Sales Channel"])
    with col_c:
        horizon    = st.slider("Months ahead", 3, 18, 6)

    metric_map = {"Revenue (₹)": "Revenue_INR", "Quantity (Units)": "Quantity", "Orders (#)": "Order_ID"}
    col = metric_map[metric_opt]
    agg_fn = "count" if col == "Order_ID" else "sum"

    # ── Monthly aggregate ────────────────────────────────────────────────────
    if level_opt == "Overall":
        if agg_fn == "count":
            monthly = df.groupby("YearMonth")["Order_ID"].count().rename("value")
        else:
            monthly = df.groupby("YearMonth")[col].sum().rename("value")

        result = forecast_series(monthly, periods=horizon)
        hist = result[result["type"]=="historical"]
        fore = result[result["type"]=="forecast"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="Historical",
                                  line=dict(color="#00e5ff", width=2.5)))
        fig.add_trace(go.Scatter(x=fore["ds"], y=fore["y"], name="Forecast",
                                  line=dict(color="#ff6b35", width=2.5, dash="dot")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fore["ds"], fore["ds"][::-1]]),
            y=pd.concat([fore["yhat_upper"], fore["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,107,53,0.12)",
            line=dict(color="transparent"), name="95% CI"
        ))
        fig.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=340, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", color="#475569"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.markdown("<div class='section-title'>Forecast Table (Future Months)</div>", unsafe_allow_html=True)
        tbl = fore[["ds","y","yhat_lower","yhat_upper"]].copy()
        tbl.columns = ["Month","Forecasted Value","Lower Bound","Upper Bound"]
        tbl["Month"] = tbl["Month"].dt.strftime("%b %Y")
        for c2 in ["Forecasted Value","Lower Bound","Upper Bound"]:
            tbl[c2] = tbl[c2].round(0).astype(int)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    else:
        group_map = {"Category": "Category", "Region": "Region", "Sales Channel": "Sales_Channel"}
        grp_col = group_map[level_opt]
        top_vals = df[grp_col].value_counts().head(5).index.tolist()

        tabs = st.tabs(top_vals)
        for i, val in enumerate(top_vals):
            with tabs[i]:
                sub = df[df[grp_col]==val]
                if agg_fn == "count":
                    monthly = sub.groupby("YearMonth")["Order_ID"].count().rename("value")
                else:
                    monthly = sub.groupby("YearMonth")[col].sum().rename("value")

                if len(monthly) < 3:
                    st.info("Not enough data for this segment.")
                    continue

                result = forecast_series(monthly, periods=horizon)
                hist = result[result["type"]=="historical"]
                fore = result[result["type"]=="forecast"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="Historical",
                                          line=dict(color=COLORS[i%len(COLORS)], width=2.5)))
                fig.add_trace(go.Scatter(x=fore["ds"], y=fore["y"], name="Forecast",
                                          line=dict(color="#ff6b35", width=2, dash="dot")))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fore["ds"], fore["ds"][::-1]]),
                    y=pd.concat([fore["yhat_upper"], fore["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(255,107,53,0.1)",
                    line=dict(color="transparent"), name="CI"
                ))
                fig.update_layout(
                    paper_bgcolor="transparent", plot_bgcolor="transparent",
                    font_color="#94a3b8", height=300, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                    legend=dict(bgcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig, use_container_width=True)

                tbl = fore[["ds","y"]].copy()
                tbl.columns = ["Month","Forecast"]
                tbl["Month"]    = tbl["Month"].dt.strftime("%b %Y")
                tbl["Forecast"] = tbl["Forecast"].round(0).astype(int)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Growth rate analysis ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>YoY Category Growth Analysis</div>", unsafe_allow_html=True)
    yr_cat = df.groupby([df["Order_Date"].dt.year, "Category"])["Revenue_INR"].sum().unstack(fill_value=0)
    if 2024 in yr_cat.index and 2025 in yr_cat.index:
        growth = ((yr_cat.loc[2025] - yr_cat.loc[2024]) / yr_cat.loc[2024] * 100).sort_values(ascending=False)
        fig_g = go.Figure(go.Bar(
            x=growth.index, y=growth.values,
            marker_color=["#10b981" if v >= 0 else "#dc2626" for v in growth.values],
            text=[f"{v:.1f}%" for v in growth.values], textposition="outside",
            textfont=dict(color="#94a3b8")
        ))
        fig_g.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=260, margin=dict(l=0,r=0,t=20,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45",
                                                    title="Growth %"),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Store forecast in session for other modules ───────────────────────────
    if agg_fn == "count":
        m_overall = df.groupby("YearMonth")["Order_ID"].count().rename("value")
    else:
        m_overall = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
    res = forecast_series(m_overall, periods=6)
    st.session_state["demand_forecast"] = res
