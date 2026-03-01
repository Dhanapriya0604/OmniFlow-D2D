"""Overview Page — Project introduction + KPI dashboard."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .data_loader import load_data

COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6"]

def render():
    df = load_data()

    st.markdown("""
    <div style='padding:32px 0 16px'>
      <div style='font-family:Syne,sans-serif;font-size:2.8rem;font-weight:800;
                  background:linear-gradient(90deg,#00e5ff,#7c3aed);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  line-height:1.1'>
        OmniFlow D2D Intelligence
      </div>
      <div style='color:#64748b;font-size:1rem;margin-top:8px'>
        End-to-end supply chain analytics · India Direct-to-Door · Jan 2024 – Jun 2026 forecast
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Description ──────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
                padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden'>
      <div style='position:absolute;top:0;left:0;right:0;height:3px;
                  background:linear-gradient(90deg,#00e5ff,#7c3aed,#ff6b35)'></div>
      <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                  color:#00e5ff;margin-bottom:12px'>📋 About This Platform</div>
      <p style='color:#cbd5e1;line-height:1.8;margin:0'>
        <b style='color:#e2e8f0'>OmniFlow</b> is an AI-driven supply chain intelligence platform built on
        <b>5,200 real-world D2D orders</b> across India (Jan 2024 – Dec 2025). It integrates six
        interconnected modules — each module's output feeds the next, creating a closed-loop
        decision engine from raw demand signals to last-mile delivery optimisation.
      </p>
      <div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:16px'>
        <span class='tag tag-blue'>Demand Forecasting → Jun 2026</span>
        <span class='tag tag-green'>Inventory Optimisation</span>
        <span class='tag' style='background:#7c3aed'>Production Planning</span>
        <span class='tag tag-orange'>Logistics Intelligence</span>
        <span class='tag tag-red'>AI Decision Chatbot</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    total_rev     = df["Revenue_INR"].sum()
    total_orders  = len(df)
    total_qty     = df["Quantity"].sum()
    return_rate   = df["Return_Flag"].mean() * 100
    avg_del_days  = df["Delivery_Days"].mean()
    categories    = df["Category"].nunique()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kpi(col, label, value, sub=""):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{value}</div>
          <div style='color:#64748b;font-size:.78rem'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "Total Revenue",   f"₹{total_rev/1e7:.1f}Cr",   "all-time")
    kpi(c2, "Orders",          f"{total_orders:,}",          "non-cancelled")
    kpi(c3, "Units Sold",      f"{total_qty:,}",             "quantities")
    kpi(c4, "Return Rate",     f"{return_rate:.1f}%",        "of delivered")
    kpi(c5, "Avg Delivery",    f"{avg_del_days:.1f}d",       "days")
    kpi(c6, "Categories",      f"{categories}",              "product types")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ────────────────────────────────────────────────────────
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("<div class='section-title'>Monthly Revenue Trend</div>", unsafe_allow_html=True)
        monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue_INR"].sum().reset_index()
        monthly["Order_Date"] = monthly["Order_Date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Order_Date"], y=monthly["Revenue_INR"],
            fill="tozeroy", line=dict(color="#00e5ff", width=2.5),
            fillcolor="rgba(0,229,255,0.08)", name="Revenue"
        ))
        fig.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=280, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", color="#475569",
                       tickformat=",.0f"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Revenue by Category</div>", unsafe_allow_html=True)
        cat_rev = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(
            labels=cat_rev.index, values=cat_rev.values,
            hole=.55, marker=dict(colors=COLORS),
            textinfo="label+percent", textfont=dict(size=10, color="#e2e8f0")
        ))
        fig2.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=280, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ────────────────────────────────────────────────────────
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("<div class='section-title'>Orders by Channel</div>", unsafe_allow_html=True)
        ch = df["Sales_Channel"].value_counts().head(6)
        fig3 = go.Figure(go.Bar(
            x=ch.values, y=ch.index, orientation="h",
            marker_color=COLORS[:len(ch)], text=ch.values, textposition="outside",
            textfont=dict(color="#94a3b8", size=11)
        ))
        fig3.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=240, margin=dict(l=0,r=40,t=0,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Top Regions by Revenue</div>", unsafe_allow_html=True)
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(8)
        fig4 = go.Figure(go.Bar(
            x=reg.index, y=reg.values,
            marker=dict(color=reg.values, colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]]),
        ))
        fig4.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=240, margin=dict(l=0,r=0,t=0,b=40),
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45")
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown("<div class='section-title'>Order Status Split</div>", unsafe_allow_html=True)
        all_status = load_data()  # include cancelled for this chart
        all_status = pd.read_csv(
            __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__file__)),
                                       "OmniFlow_D2D_India_Unified_5200.csv"))
        st_counts = all_status["Order_Status"].value_counts()
        fig5 = go.Figure(go.Bar(
            x=st_counts.index, y=st_counts.values,
            marker_color=["#10b981","#dc2626","#f59e0b","#3b82f6"][:len(st_counts)]
        ))
        fig5.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=240, margin=dict(l=0,r=0,t=0,b=40),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45")
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Module Dependency Map ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Module Dependency Flow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
                padding:24px;display:flex;align-items:center;justify-content:center;
                flex-wrap:wrap;gap:8px;font-family:Syne,sans-serif'>
      <div style='background:#1e2d45;border:1px solid #00e5ff;border-radius:8px;
                  padding:10px 18px;color:#00e5ff;font-weight:700;font-size:.85rem;text-align:center'>
        📈 Demand<br><span style='font-size:.7rem;color:#64748b'>Forecast to Jun'26</span></div>
      <div style='color:#7c3aed;font-size:1.5rem;font-weight:800'>→</div>
      <div style='background:#1e2d45;border:1px solid #10b981;border-radius:8px;
                  padding:10px 18px;color:#10b981;font-weight:700;font-size:.85rem;text-align:center'>
        📦 Inventory<br><span style='font-size:.7rem;color:#64748b'>Stock Optimisation</span></div>
      <div style='color:#7c3aed;font-size:1.5rem;font-weight:800'>→</div>
      <div style='background:#1e2d45;border:1px solid #7c3aed;border-radius:8px;
                  padding:10px 18px;color:#a78bfa;font-weight:700;font-size:.85rem;text-align:center'>
        🏭 Production<br><span style='font-size:.7rem;color:#64748b'>Plan based on D+I</span></div>
      <div style='color:#7c3aed;font-size:1.5rem;font-weight:800'>→</div>
      <div style='background:#1e2d45;border:1px solid #ff6b35;border-radius:8px;
                  padding:10px 18px;color:#ff6b35;font-weight:700;font-size:.85rem;text-align:center'>
        🚚 Logistics<br><span style='font-size:.7rem;color:#64748b'>Carrier & Region Intel</span></div>
      <div style='color:#7c3aed;font-size:1.5rem;font-weight:800'>→</div>
      <div style='background:#1e2d45;border:1px solid #f59e0b;border-radius:8px;
                  padding:10px 18px;color:#f59e0b;font-weight:700;font-size:.85rem;text-align:center'>
        🤖 Chatbot<br><span style='font-size:.7rem;color:#64748b'>Decision Intelligence</span></div>
    </div>
    """, unsafe_allow_html=True)
