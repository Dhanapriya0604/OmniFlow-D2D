"""Inventory Optimization — EOQ, Safety Stock, Reorder Point based on demand forecast."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .data_loader import load_data
from .demand import forecast_series

COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6"]

# ── Inventory formulas ────────────────────────────────────────────────────────
def calc_eoq(annual_demand, order_cost=500, holding_cost_pct=0.2, unit_cost=500):
    holding = unit_cost * holding_cost_pct
    if holding <= 0 or annual_demand <= 0:
        return 0
    return int(np.sqrt(2 * annual_demand * order_cost / holding))

def calc_safety_stock(std_demand, lead_time=7, service_level_z=1.65):
    return int(service_level_z * std_demand * np.sqrt(lead_time / 30))

def calc_rop(avg_daily_demand, lead_time=7, safety_stock=0):
    return int(avg_daily_demand * lead_time + safety_stock)


def render():
    df = load_data()

    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#10b981'>
        📦 Inventory Optimization
      </div>
      <div style='color:#64748b;font-size:.9rem'>
        EOQ · Safety Stock · Reorder Point · driven by demand forecast output
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameters sidebar ────────────────────────────────────────────────────
    with st.expander("⚙️ Inventory Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        order_cost   = c1.number_input("Order Cost (₹)", 100, 5000, 500, 50)
        hold_pct     = c2.slider("Holding Cost %", 5, 40, 20) / 100
        lead_time    = c3.slider("Lead Time (days)", 1, 30, 7)
        service_z    = st.selectbox("Service Level", ["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"])
        z_map = {"90% (z=1.28)":1.28, "95% (z=1.65)":1.65, "99% (z=2.33)":2.33}
        z_val = z_map[service_z]

    # ── Build SKU-level inventory table ──────────────────────────────────────
    sku_stats = (
        df.groupby(["SKU_ID","Product_Name","Category","Sell_Price"])
        .agg(
            total_qty   = ("Quantity","sum"),
            monthly_qty = ("Quantity", lambda x: x.groupby(df.loc[x.index,"YearMonth"]).sum().mean()),
            std_qty     = ("Quantity", lambda x: x.groupby(df.loc[x.index,"YearMonth"]).sum().std()),
        ).reset_index()
    )
    sku_stats["std_qty"]     = sku_stats["std_qty"].fillna(0)
    sku_stats["unit_cost"]   = sku_stats["Sell_Price"]
    sku_stats["annual_demand"] = sku_stats["monthly_qty"] * 12
    sku_stats["EOQ"]           = sku_stats.apply(
        lambda r: calc_eoq(r["annual_demand"], order_cost, hold_pct, r["unit_cost"]), axis=1)
    sku_stats["Safety_Stock"]  = sku_stats.apply(
        lambda r: calc_safety_stock(r["std_qty"], lead_time, z_val), axis=1)
    sku_stats["Reorder_Point"] = sku_stats.apply(
        lambda r: calc_rop(r["monthly_qty"]/30, lead_time, r["Safety_Stock"]), axis=1)

    # ── Get demand forecast to project future need ────────────────────────────
    m_overall = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    demand_fore = forecast_series(m_overall, periods=6)
    future_demand = demand_fore[demand_fore["type"]=="forecast"]
    avg_future_growth = (future_demand["y"].mean() / m_overall.iloc[-3:].mean()) - 1

    sku_stats["Forecast_Adj_Annual"] = (sku_stats["annual_demand"] * (1 + avg_future_growth)).astype(int)
    sku_stats["Stock_Status"] = sku_stats.apply(
        lambda r: "🔴 Critical" if r["total_qty"] < r["Safety_Stock"]
                  else ("🟡 Low" if r["total_qty"] < r["Reorder_Point"]
                        else "🟢 Adequate"), axis=1)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_sku  = len(sku_stats)
    critical   = (sku_stats["Stock_Status"]=="🔴 Critical").sum()
    low        = (sku_stats["Stock_Status"]=="🟡 Low").sum()
    ok         = (sku_stats["Stock_Status"]=="🟢 Adequate").sum()

    c1,c2,c3,c4 = st.columns(4)
    for col, label, value, color in [
        (c1,"Total SKUs",   total_sku, "#00e5ff"),
        (c2,"Critical",     critical,  "#dc2626"),
        (c3,"Low Stock",    low,       "#f59e0b"),
        (c4,"Adequate",     ok,        "#10b981"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='color:{color}'>{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stock status chart ────────────────────────────────────────────────────
    col_l, col_r = st.columns([1,2])

    with col_l:
        st.markdown("<div class='section-title'>Stock Status Distribution</div>", unsafe_allow_html=True)
        status_counts = sku_stats["Stock_Status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=status_counts.index, values=status_counts.values,
            hole=.55, marker_colors=["#dc2626","#f59e0b","#10b981"],
            textinfo="label+percent"
        ))
        fig.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=260, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>EOQ vs Reorder Point by Category</div>", unsafe_allow_html=True)
        cat_inv = sku_stats.groupby("Category")[["EOQ","Safety_Stock","Reorder_Point"]].mean().reset_index()
        fig2 = go.Figure()
        for i, metric in enumerate(["EOQ","Safety_Stock","Reorder_Point"]):
            fig2.add_trace(go.Bar(
                name=metric, x=cat_inv["Category"], y=cat_inv[metric],
                marker_color=COLORS[i]
            ))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=260, margin=dict(l=0,r=0,t=10,b=40),
            xaxis=dict(showgrid=False, tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Demand-adjusted future inventory need ─────────────────────────────────
    st.markdown("<div class='section-title'>Future Inventory Need (Based on Demand Forecast)</div>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=future_demand["ds"], y=future_demand["y"],
        mode="lines+markers", name="Forecasted Demand",
        line=dict(color="#00e5ff", width=2.5),
        marker=dict(size=8, color="#00e5ff")
    ))
    fig3.add_trace(go.Scatter(
        x=pd.concat([future_demand["ds"], future_demand["ds"][::-1]]),
        y=pd.concat([future_demand["yhat_upper"], future_demand["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(0,229,255,0.08)",
        line=dict(color="transparent"), name="Confidence Band"
    ))
    fig3.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font_color="#94a3b8", height=240, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45",
                                                title="Units Required"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── SKU Inventory Table ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>SKU Inventory Recommendations</div>", unsafe_allow_html=True)
    cat_filter = st.multiselect("Filter by Category", df["Category"].unique().tolist(),
                                default=df["Category"].unique().tolist())
    disp = sku_stats[sku_stats["Category"].isin(cat_filter)][
        ["SKU_ID","Product_Name","Category","monthly_qty","EOQ","Safety_Stock","Reorder_Point",
         "Forecast_Adj_Annual","Stock_Status"]
    ].copy()
    disp.columns = ["SKU","Product","Category","Avg Monthly Demand",
                    "EOQ","Safety Stock","Reorder Point","Forecast Annual Demand","Status"]
    for c2 in ["Avg Monthly Demand","EOQ","Safety Stock","Reorder Point","Forecast Annual Demand"]:
        disp[c2] = disp[c2].round(0).astype(int)
    st.dataframe(disp.sort_values("Status"), use_container_width=True, hide_index=True)

    # Store for downstream modules
    st.session_state["inventory_data"] = sku_stats
