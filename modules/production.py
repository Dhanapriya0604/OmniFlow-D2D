"""Production Planning — uses demand forecast + inventory to generate production schedule."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .data_loader import load_data
from .demand import forecast_series

COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6"]


def get_production_plan(df, capacity_multiplier=1.0, safety_buffer=0.15):
    """Generate monthly production targets per category using demand + inventory gap."""
    # Category monthly demand
    cat_monthly = (
        df.groupby(["YearMonth","Category"])["Quantity"].sum()
        .unstack(fill_value=0)
    )
    plans = []
    for cat in cat_monthly.columns:
        series = cat_monthly[cat].rename("value")
        fore   = forecast_series(series, periods=6)
        future = fore[fore["type"]=="forecast"].copy()

        # Estimated current stock (last 3-month avg * lead-time factor)
        current_stock = float(series.iloc[-3:].mean() * 1.5)

        for _, row in future.iterrows():
            net_demand  = max(row["y"] - current_stock / 6, 0)
            with_buffer = net_demand * (1 + safety_buffer)
            production  = with_buffer * capacity_multiplier

            plans.append({
                "Month":          row["ds"].strftime("%b %Y"),
                "Month_dt":       row["ds"],
                "Category":       cat,
                "Forecasted_Demand": round(row["y"], 0),
                "Net_Production": round(production, 0),
                "Safety_Buffer":  round(with_buffer - net_demand, 0),
                "CI_Low":         round(row["yhat_lower"], 0),
                "CI_High":        round(row["yhat_upper"], 0),
            })
        current_stock = 0  # reset for next pass
    return pd.DataFrame(plans)


def render():
    df = load_data()

    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#a78bfa'>
        🏭 Production Planning
      </div>
      <div style='color:#64748b;font-size:.9rem'>
        Monthly production targets · driven by demand forecast + inventory position
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameters ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        capacity_mult = st.slider("Production Capacity Multiplier", 0.5, 2.0, 1.0, 0.1,
                                  help="Scale planned production relative to demand")
    with col_b:
        safety_buf = st.slider("Safety Buffer %", 5, 40, 15,
                               help="Extra % over net demand as safety production")

    plan_df = get_production_plan(df, capacity_mult, safety_buf/100)

    # ── KPI ───────────────────────────────────────────────────────────────────
    total_prod = plan_df["Net_Production"].sum()
    total_fore = plan_df["Forecasted_Demand"].sum()
    avg_month  = plan_df.groupby("Month_dt")["Net_Production"].sum().mean()
    peak_month = plan_df.groupby("Month_dt")["Net_Production"].sum().idxmax()

    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (c1, "Total Production Target", f"{total_prod:,.0f} units", "#a78bfa"),
        (c2, "Total Demand (6 mo)",     f"{total_fore:,.0f} units", "#00e5ff"),
        (c3, "Avg Monthly Production",  f"{avg_month:,.0f}",        "#10b981"),
        (c4, "Peak Month",              peak_month.strftime("%b %Y"), "#ff6b35"),
    ]
    for col, lbl, val, clr in kpis:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{lbl}</div>
          <div class='metric-value' style='color:{clr}'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Production vs Demand chart ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Production Plan vs Forecasted Demand (All Categories)</div>",
                unsafe_allow_html=True)

    agg = plan_df.groupby("Month_dt")[["Net_Production","Forecasted_Demand"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Month_dt"], y=agg["Net_Production"],
        name="Production Target", marker_color="#7c3aed"
    ))
    fig.add_trace(go.Scatter(
        x=agg["Month_dt"], y=agg["Forecasted_Demand"],
        mode="lines+markers", name="Demand Forecast",
        line=dict(color="#00e5ff", width=2.5), marker=dict(size=8)
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font_color="#94a3b8", height=300, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Category breakdown ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Production by Category (Stacked)</div>", unsafe_allow_html=True)
    cats = plan_df["Category"].unique()
    fig2 = go.Figure()
    for i, cat in enumerate(cats):
        sub = plan_df[plan_df["Category"]==cat].sort_values("Month_dt")
        fig2.add_trace(go.Bar(
            x=sub["Month_dt"], y=sub["Net_Production"],
            name=cat, marker_color=COLORS[i % len(COLORS)]
        ))
    fig2.update_layout(
        barmode="stack",
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font_color="#94a3b8", height=320, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Detailed plan table ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Detailed Production Schedule</div>", unsafe_allow_html=True)
    cat_sel = st.selectbox("Filter Category", ["All"] + list(plan_df["Category"].unique()))
    disp = plan_df if cat_sel == "All" else plan_df[plan_df["Category"]==cat_sel]
    disp2 = disp[["Month","Category","Forecasted_Demand","Safety_Buffer",
                   "Net_Production","CI_Low","CI_High"]].copy()
    disp2.columns = ["Month","Category","Demand Forecast","Safety Buffer",
                     "Production Target","Demand Low","Demand High"]
    st.dataframe(disp2.sort_values("Month"), use_container_width=True, hide_index=True)

    # ── Inventory vs Production gap ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Demand–Production Gap Analysis</div>", unsafe_allow_html=True)
    agg["Gap"] = agg["Net_Production"] - agg["Forecasted_Demand"]
    fig3 = go.Figure(go.Bar(
        x=agg["Month_dt"], y=agg["Gap"],
        marker_color=["#10b981" if g >= 0 else "#dc2626" for g in agg["Gap"]],
        text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside",
        textfont=dict(color="#94a3b8")
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="#475569")
    fig3.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font_color="#94a3b8", height=220, margin=dict(l=0,r=0,t=20,b=0),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45",
                                                title="Units (Surplus / Deficit)")
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.session_state["production_plan"] = plan_df
