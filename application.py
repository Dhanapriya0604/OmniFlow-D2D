"""
OmniFlow D2D Supply Chain Intelligence
Streamlit Cloud entry point: application.py
Fixes:
  - Inventory: proper SKU aggregation, realistic stock simulation → real Critical/Low distribution
  - Module interdependency: demand forecast growth feeds inventory & production
  - Production explicitly uses demand_forecast output
  - Logistics uses production plan volumes to prioritise carriers/warehouses
  - Chatbot: uses claude-sonnet-4-20250514 with x-api-key header for cloud
  - All chart_defaults use rgba strings (not keyword args that break Plotly)
"""

import streamlit as st

st.set_page_config(page_title="OmniFlow D2D Intelligence", page_icon="🔮",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root{
    --bg:#f7f9fc;
    --card:rgba(255,255,255,0.85);
    --border:#e6ebf2;
    --primary:#5b7cfa;        
    --primary-soft:#eef2ff;
    --text:#1e293b;
    --muted:#7b8794;
    --green:#4ade80;         
    --orange:#fb923c;      
    --red:#f87171;         
    --purple:#a78bfa;       
    --sky:#38bdf8;        
    --shadow:0 8px 24px rgba(0,0,0,0.05);
    --glass:blur(10px);
}
html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    background-color:var(--bg)!important;
    color:var(--text)!important;
}
section[data-testid="stSidebar"]{
    background:var(--card)!important;
    border-right:1px solid var(--border)!important;
}
h1{
    font-size:2rem!important;
    font-weight:700!important;
    color:var(--primary)!important;
}
h2, h3{
    font-weight:600!important;
    color:var(--text)!important;
}
.metric-card{
    background:var(--card);
    backdrop-filter:var(--glass);
    border:1px solid var(--border);
    border-radius:16px;
    padding:18px;
    box-shadow:var(--shadow);
    transition:all 0.3s ease;
}
.metric-card:hover{
    transform:translateY(-4px) scale(1.01);
    box-shadow:0 12px 30px rgba(0,0,0,0.08);
}
.metric-label{
    font-size:0.75rem;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:0.06em;
}
.metric-value{
    font-size:1.9rem;
    font-weight:700;
    margin-top:6px;
}
.alert-box{
    background:var(--card);
    border-left:4px solid var(--primary);
    padding:12px 16px;
    border-radius:10px;
    box-shadow:var(--shadow);
}
.stButton>button{
    background:linear-gradient(135deg,#5b7cfa,#38bdf8)!important;
    color:white!important;
    border:none!important;
    border-radius:10px!important;
    font-weight:600!important;
    transition:0.25s;
}
.stButton>button:hover{
    transform:scale(1.05);
    box-shadow:0 8px 20px rgba(91,124,250,0.25);
}
.stTextInput input, .stSelectbox div{
    border-radius:8px!important;
    border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab"]{
    color:var(--muted)!important;
    font-weight:500;
}
.stTabs [aria-selected="true"]{
    color:var(--primary)!important;
    border-bottom:2px solid var(--primary)!important;
}
[data-testid="stExpander"]{
    background:var(--card)!important;
    border:1px solid var(--border)!important;
    border-radius:10px!important;
}
.stDataFrame{
    background:var(--card)!important;
    border-radius:10px!important;
}
div[style*="background:#0"]{
    background:var(--card)!important;
    color:var(--text)!important;
}
.chat-user-bubble{
    background:#eef2ff;
    padding:10px 14px;
    border-radius:12px;
}
.chat-ai-bubble{
    background:var(--card);
    border:1px solid var(--border);
    padding:12px 14px;
    border-radius:12px;
}
.tag{
    padding:4px 10px;
    border-radius:20px;
    font-size:0.7rem;
    font-weight:600;
}
.tag-green{background:#dcfce7;color:#166534;}
.tag-orange{background:#ffedd5;color:#9a3412;}
.tag-red{background:#fee2e2;color:#7f1d1d;}
.tag-blue{background:#e0e7ff;color:#1e3a8a;}
.tag-purple{background:#f3e8ff;color:#6b21a8;}
.stSlider [role="slider"]{
    background:var(--primary)!important;
}
.js-plotly-plot{
    background:transparent!important;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, os

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "OmniFlow_D2D_India_Unified_5200.csv")
COLORS = ["#5b7cfa","#38bdf8","#4ade80","#fb923c","#f87171","#a78bfa"]
@st.cache_data(show_spinner="Loading supply chain data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df = df[df["Order_Status"] != "Cancelled"].copy()
    df["YearMonth"] = df["Order_Date"].dt.to_period("M")
    return df

@st.cache_data(show_spinner=False)
def load_all_statuses():
    return pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])

def forecast_series(series: pd.Series, periods: int = 6) -> pd.DataFrame:
    """Linear trend + monthly seasonality forecaster. Returns hist + future rows."""
    s = series.dropna()
    if len(s) < 3:
        return pd.DataFrame()
    n = len(s)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, s.values, 1)
    fitted = slope * x + intercept
    resid  = s.values - fitted
    seas = np.zeros(12); cnt = np.zeros(12)
    idx_months = s.index.month if hasattr(s.index, "month") else (np.arange(n) % 12 + 1)
    for i, mo in enumerate(idx_months):
        seas[mo-1] += resid[i]; cnt[mo-1] += 1
    seas /= np.where(cnt > 0, cnt, 1)
   
    fut_x      = np.arange(n, n + periods)
    fut_months = [(s.index[-1].month + i - 1) % 12 + 1 for i in range(1, periods + 1)]
    fut_vals   = np.maximum(slope * fut_x + intercept + np.array([seas[m-1] for m in fut_months]), 0)
    std        = resid.std()
    hist_df = pd.DataFrame({
        "ds": s.index.to_timestamp(), "y": s.values,
        "type": "historical", "yhat_lower": np.nan, "yhat_upper": np.nan
    })
    fut_dates = pd.date_range(s.index[-1].to_timestamp(), periods=periods + 1, freq="MS")[1:]
    fore_df = pd.DataFrame({
        "ds": fut_dates, "y": fut_vals, "type": "forecast",
        "yhat_lower": np.maximum(fut_vals - 1.65 * std, 0),
        "yhat_upper": fut_vals + 1.65 * std
    })
    return pd.concat([hist_df, fore_df], ignore_index=True)

@st.cache_data(show_spinner=False)
def compute_demand_forecast():
    """MODULE 1 OUTPUT: monthly qty forecast for 6 months."""
    df = load_data()
    m  = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    return forecast_series(m, 6)

@st.cache_data(show_spinner=False)
def compute_inventory_table(order_cost=500, hold_pct=0.20, lead_time=7, z=1.65):
    df = load_data()
    sku_agg = df.groupby(["SKU_ID", "Product_Name", "Category"]).agg(
        total_qty  = ("Quantity", "sum"),
        num_months = ("YearMonth", lambda x: x.nunique()),
        avg_price  = ("Sell_Price", "mean")
    ).reset_index()
    sku_agg["monthly_avg"] = sku_agg["total_qty"] / sku_agg["num_months"]
    sku_monthly = (
        df.groupby(["SKU_ID", "YearMonth"])["Quantity"]
        .sum().unstack(fill_value=0)
    )
    sku_std = sku_monthly.std(axis=1).rename("monthly_std").reset_index()
    sku_agg = sku_agg.merge(sku_std, on="SKU_ID", how="left")
    sku_agg["monthly_std"] = sku_agg["monthly_std"].fillna(0)

    def eoq(d, oc, h, uc):
        return int(np.sqrt(2 * d * oc / (uc * h))) if d > 0 and uc * h > 0 else 0

    sku_agg["annual_demand"] = sku_agg["monthly_avg"] * 12
    sku_agg["EOQ"] = sku_agg.apply(
        lambda r: eoq(r["annual_demand"], order_cost, hold_pct, r["avg_price"]), axis=1)
    sku_agg["SS"]  = (z * sku_agg["monthly_std"] * np.sqrt(lead_time / 30)).astype(int)
    sku_agg["ROP"] = (sku_agg["monthly_avg"] / 30 * lead_time + sku_agg["SS"]).astype(int)

    demand_fore = compute_demand_forecast()
    fut_demand  = demand_fore[demand_fore["type"] == "forecast"]
    m_qty       = df.groupby("YearMonth")["Quantity"].sum()
    recent_avg  = m_qty.iloc[-3:].mean()
    growth      = (fut_demand["y"].mean() / recent_avg - 1) if recent_avg > 0 else 0
    sku_agg["forecast_annual"] = (sku_agg["annual_demand"] * (1 + growth)).astype(int)
    sku_agg["demand_growth_pct"] = round(growth * 100, 1)

    np.random.seed(42)
    stock_factor = np.random.uniform(0.3, 2.8, len(sku_agg))
    sku_agg["current_stock"] = (sku_agg["monthly_avg"] * stock_factor).astype(int)

    def classify(row):
        if row["current_stock"] < row["SS"]:
            return "🔴 Critical"
        elif row["current_stock"] < row["ROP"]:
            return "🟡 Low"
        return "🟢 Adequate"
    sku_agg["Status"] = sku_agg.apply(classify, axis=1)
    return sku_agg

@st.cache_data(show_spinner=False)
def compute_production_plan(cap=1.0, buf=0.15):
    df = load_data()
    inv = compute_inventory_table()
    cat_critical = (
        inv[inv["Status"] == "🔴 Critical"]
        .groupby("Category")["monthly_avg"].sum()
        .rename("critical_monthly")
    )
    cat_monthly = df.groupby(["YearMonth", "Category"])["Quantity"].sum().unstack(fill_value=0)
    plans = []
    for cat in cat_monthly.columns:
        series = cat_monthly[cat].rename("value")
        fore   = forecast_series(series, 6)
        if fore.empty:
            continue
        fut = fore[fore["type"] == "forecast"]
        crit_extra = float(cat_critical.get(cat, 0)) * 0.5
        cur_stock  = float(series.iloc[-3:].mean() * 1.5)
        for _, row in fut.iterrows():
            net  = max(row["y"] - cur_stock / 6 + crit_extra, 0)
            prod = net * (1 + buf) * cap
            plans.append({
                "Month":      row["ds"].strftime("%b %Y"),
                "Month_dt":   row["ds"],
                "Category":   cat,
                "Demand":     round(row["y"], 0),
                "Inv_Boost":  round(crit_extra, 0),
                "Production": round(prod, 0),
                "Buffer":     round(prod - net * cap, 0),
                "CI_Lo":      round(row["yhat_lower"], 0),
                "CI_Hi":      round(row["yhat_upper"], 0),
            })
    return pd.DataFrame(plans)

def CD():
    return dict(
        paper_bgcolor="#f7f9fc",   
        plot_bgcolor="#ffffff",   
        font=dict(color="#1e293b"),
        margin=dict(l=0, r=0, t=16, b=0),
    )
def kpi(col, label, value, color="#00e5ff", sub=""):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{value}</div>
      <div style='color:#475569;font-size:.75rem'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def feed_badge(text):
    st.markdown(f"<span class='feed-badge'>⬆ feeds from {text}</span>", unsafe_allow_html=True)

def ci_band(fig, fore, color="rgba(255,107,53,0.12)"):
    ds_fwd  = list(fore["ds"]) + list(fore["ds"])[::-1]
    y_band  = list(fore["yhat_upper"]) + list(fore["yhat_lower"])[::-1]
    fig.add_trace(go.Scatter(
        x=ds_fwd, y=y_band,
        fill="toself", fillcolor=color,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=False
    ))

def page_overview():
    df  = load_data()
    raw = load_all_statuses()
    st.markdown("""
    <div style='padding:24px 0 10px'>
      <div style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;
           background:linear-gradient(90deg,#00e5ff,#7c3aed);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1'>
        OmniFlow D2D Intelligence</div>
      <div style='color:#64748b;font-size:.95rem;margin-top:6px'>
        Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System
      </div></div>""", unsafe_allow_html=True)
   st.markdown("""
      <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
           padding:22px 28px;margin-bottom:20px;position:relative;overflow:hidden'>
        <div style='position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#00e5ff,#7c3aed,#ff6b35)'></div>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
             color:#00e5ff;margin-bottom:8px'>About This Platform</div>
        <p style='color:#cbd5e1;line-height:1.8;margin:0'>
          <b style='color:#e2e8f0'>OmniFlow</b> is an AI-driven supply chain intelligence platform
          built on <b>5,200 D2D orders</b> across India (Jan 2024–Dec 2025). Six interconnected
          modules feed each other in sequence — demand signals drive inventory, which drives
          production, which informs logistics. The AI chatbot synthesises all module outputs.
        </p>
        <div style='margin-top:12px'>
          <span class='tag tag-blue'>Demand → Jun 2026</span>
          <span class='tag tag-green'>Inventory EOQ/ROP</span>
          <span class='tag' style='background:#7c3aed'>Production Plan</span>
          <span class='tag tag-orange'>Logistics Intel</span>
          <span class='tag tag-red'>AI Chatbot</span>
        </div></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, "Total Revenue",  f"₹{df['Revenue_INR'].sum()/1e7:.1f}Cr", "#00e5ff", "all time")
    kpi(c2, "Orders",         f"{len(df):,}",                           "#f59e0b", "non-cancelled")
    kpi(c3, "Units Sold",     f"{df['Quantity'].sum():,}",              "#10b981", "quantities")
    kpi(c4, "Return Rate",    f"{df['Return_Flag'].mean()*100:.1f}%",   "#dc2626", "delivered")
    kpi(c5, "Avg Delivery",   f"{df['Delivery_Days'].mean():.1f}d",     "#a78bfa", "days")
    kpi(c6, "SKU Categories", f"{df['Category'].nunique()}",            "#3b82f6", "types")
    st.markdown("<br>", unsafe_allow_html=True)

    c_l, c_r = st.columns([2, 1])
    with c_l:
        st.markdown("<div class='section-title'>Monthly Revenue Trend</div>", unsafe_allow_html=True)
        m = df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue_INR"].sum().reset_index()
        m["Order_Date"] = m["Order_Date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["Order_Date"], y=m["Revenue_INR"],
            fill="tozeroy", line=dict(color="#00e5ff", width=2.5),
            fillcolor="rgba(0,229,255,0.07)", name="Revenue"))
        fig.update_layout(**CD(), height=270,
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", color="#475569", tickformat=",.0f"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c_r:
        st.markdown("<div class='section-title'>Revenue by Category</div>", unsafe_allow_html=True)
        cat = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(labels=cat.index, values=cat.values, hole=.55,
            marker=dict(colors=COLORS), textinfo="label+percent",
            textfont=dict(size=9, color="#e2e8f0")))
        fig2.update_layout(**CD(), height=270, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3a, c3b, c3c = st.columns(3)
    with c3a:
        st.markdown("<div class='section-title'>Orders by Channel</div>", unsafe_allow_html=True)
        ch = df["Sales_Channel"].value_counts().head(6)
        fig3 = go.Figure(go.Bar(x=ch.values, y=ch.index, orientation="h",
            marker_color=COLORS[:len(ch)], text=ch.values, textposition="outside",
            textfont=dict(color="#94a3b8", size=10)))
        fig3.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)
    with c3b:
        st.markdown("<div class='section-title'>Top Regions Revenue</div>", unsafe_allow_html=True)
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(8)
        fig4 = go.Figure(go.Bar(x=reg.index, y=reg.values,
            marker=dict(color=list(reg.values), colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]])))
        fig4.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"))
        st.plotly_chart(fig4, use_container_width=True)
    with c3c:
        st.markdown("<div class='section-title'>Order Status Split</div>", unsafe_allow_html=True)
        sc = raw["Order_Status"].value_counts()
        colors_sc = ["#10b981","#dc2626","#f59e0b","#3b82f6"]
        fig5 = go.Figure(go.Bar(x=sc.index, y=sc.values,
            marker_color=colors_sc[:len(sc)]))
        fig5.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div class='section-title'>Module Dependency Flow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
         padding:20px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px'>
      <div style='background:#0a1520;border:1px solid #00e5ff;border-radius:8px;
           padding:10px 14px;color:#00e5ff;font-weight:700;font-family:Syne,sans-serif;
           font-size:.8rem;text-align:center;min-width:90px'>
        Demand<br><span style='font-size:.66rem;color:#64748b'>→ Jun 2026</span></div>
      <div style='color:#7c3aed;font-size:1.3rem;font-weight:800'>→</div>
      <div style='background:#0a1520;border:1px solid #10b981;border-radius:8px;
           padding:10px 14px;color:#10b981;font-weight:700;font-family:Syne,sans-serif;
           font-size:.8rem;text-align:center;min-width:90px'>
        Inventory<br><span style='font-size:.66rem;color:#64748b'>EOQ+ROP</span></div>
      <div style='color:#7c3aed;font-size:1.3rem;font-weight:800'>→</div>
      <div style='background:#0a1520;border:1px solid #7c3aed;border-radius:8px;
           padding:10px 14px;color:#a78bfa;font-weight:700;font-family:Syne,sans-serif;
           font-size:.8rem;text-align:center;min-width:90px'>
        Production<br><span style='font-size:.66rem;color:#64748b'>D+Inv driven</span></div>
      <div style='color:#7c3aed;font-size:1.3rem;font-weight:800'>→</div>
      <div style='background:#0a1520;border:1px solid #ff6b35;border-radius:8px;
           padding:10px 14px;color:#ff6b35;font-weight:700;font-family:Syne,sans-serif;
           font-size:.8rem;text-align:center;min-width:90px'>
        Logistics<br><span style='font-size:.66rem;color:#64748b'>Prod+carrier</span></div>
      <div style='color:#7c3aed;font-size:1.3rem;font-weight:800'>→</div>
      <div style='background:#0a1520;border:1px solid #f59e0b;border-radius:8px;
           padding:10px 14px;color:#f59e0b;font-weight:700;font-family:Syne,sans-serif;
           font-size:.8rem;text-align:center;min-width:90px'>
        Chatbot<br><span style='font-size:.66rem;color:#64748b'>All outputs</span></div>
    </div>""", unsafe_allow_html=True)

def page_demand():
    df = load_data()

    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#00e5ff'>
        Demand Forecasting</div>
      <div style='color:#64748b;font-size:.88rem'>
        Linear trend + seasonal decomposition · Historic Jan 2024–Dec 2025 · Forecast to Jun 2026
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:12px'>
      <span class='feed-badge'>OUTPUT feeds → Inventory · Production · Logistics · Chatbot</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    metric_opt = c1.selectbox("Metric", ["Revenue (₹)", "Quantity (Units)", "Orders (#)"])
    level_opt  = c2.selectbox("Breakdown", ["Overall", "Category", "Region", "Sales Channel"])
    horizon    = c3.slider("Months ahead", 3, 18, 6)
    col_map = {"Revenue (₹)": "Revenue_INR", "Quantity (Units)": "Quantity", "Orders (#)": "Order_ID"}
    col     = col_map[metric_opt]
    agg_fn  = "count" if col == "Order_ID" else "sum"

    def get_monthly(sub):
        if agg_fn == "count":
            return sub.groupby("YearMonth")["Order_ID"].count().rename("value")
        return sub.groupby("YearMonth")[col].sum().rename("value")

    def draw_forecast(series, color="#00e5ff", title=""):
        res  = forecast_series(series, periods=horizon)
        if res.empty:
            st.info("Not enough data.")
            return res
        hist = res[res["type"] == "historical"]
        fore = res[res["type"] == "forecast"]
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="Historical",
            line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=fore["ds"], y=fore["y"], name="Forecast",
            line=dict(color="#ff6b35", width=2.5, dash="dot")))
        ci_band(fig, fore)
        fig.update_layout(**CD(), height=300,
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text=title, font=dict(color="#94a3b8", size=12)))
        st.plotly_chart(fig, use_container_width=True)
        return res

    if level_opt == "Overall":
        res = draw_forecast(get_monthly(df))
        if not res.empty:
            fore = res[res["type"] == "forecast"][["ds","y","yhat_lower","yhat_upper"]].copy()
            fore.columns = ["Month", "Forecast", "Lower (95%)", "Upper (95%)"]
            fore["Month"] = fore["Month"].dt.strftime("%b %Y")
            for c2 in ["Forecast","Lower (95%)","Upper (95%)"]:
                fore[c2] = fore[c2].round(0).astype(int)
            st.markdown("<div class='section-title'>Forecast Table</div>", unsafe_allow_html=True)
            st.dataframe(fore, use_container_width=True, hide_index=True)
    else:
        grp_map = {"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}
        grp = grp_map[level_opt]
        top = df[grp].value_counts().head(5).index.tolist()
        tabs = st.tabs(top)
        for i, (tab, val) in enumerate(zip(tabs, top)):
            with tab:
                res = draw_forecast(get_monthly(df[df[grp] == val]), color=COLORS[i], title=val)
                if not res.empty:
                    fore = res[res["type"]=="forecast"][["ds","y"]].copy()
                    fore.columns = ["Month","Forecast"]
                    fore["Month"] = fore["Month"].dt.strftime("%b %Y")
                    fore["Forecast"] = fore["Forecast"].round(0).astype(int)
                    st.dataframe(fore, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>YoY Category Growth (2024 → 2025)</div>",
                unsafe_allow_html=True)
    yr = df.groupby([df["Order_Date"].dt.year, "Category"])["Revenue_INR"].sum().unstack(fill_value=0)
    if 2024 in yr.index and 2025 in yr.index:
        g = ((yr.loc[2025] - yr.loc[2024]) / yr.loc[2024] * 100).sort_values(ascending=False)
        fig_g = go.Figure(go.Bar(
            x=g.index, y=g.values,
            marker_color=["#10b981" if v >= 0 else "#dc2626" for v in g.values],
            text=[f"{v:.1f}%" for v in g.values], textposition="outside",
            textfont=dict(color="#94a3b8")))
        fig_g.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="YoY Growth %"))
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div class='section-title'>Category Demand Forecast (fed to Production & Inventory)</div>",
                unsafe_allow_html=True)
    cat_monthly = df.groupby(["YearMonth","Category"])["Quantity"].sum().unstack(fill_value=0)
    cat_fore_rows = []
    for cat in cat_monthly.columns:
        s = cat_monthly[cat].rename("value")
        f = forecast_series(s, 6)
        if f.empty: continue
        for _, row in f[f["type"]=="forecast"].iterrows():
            cat_fore_rows.append({
                "Month": row["ds"].strftime("%b %Y"),
                "Category": cat,
                "Forecast Units": int(round(row["y"], 0)),
                "Lower": int(round(row["yhat_lower"], 0)),
                "Upper": int(round(row["yhat_upper"], 0)),
            })
    cat_fore_df = pd.DataFrame(cat_fore_rows)
    st.dataframe(cat_fore_df, use_container_width=True, hide_index=True)

def page_inventory():
    df = load_data()

    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#10b981'>
        Inventory Optimization</div>
      <div style='color:#64748b;font-size:.88rem'>
        EOQ · Safety Stock · Reorder Point · stock status per SKU
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:12px'>
      <span class='feed-badge'>⬆ feeds from: Demand Forecast (growth rate)</span>
      <span class='feed-badge' style='border-color:#10b981;color:#10b981'>
        feeds → Production Planning · Chatbot</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Inventory Parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        order_cost = p1.number_input("Order Cost ₹", 100, 5000, 500, 50)
        hold_pct   = p2.slider("Holding Cost %", 5, 40, 20) / 100
        lead_time  = p3.slider("Lead Time (days)", 1, 30, 7)
        svc        = st.selectbox("Service Level",
                                  ["90% (z=1.28)", "95% (z=1.65)", "99% (z=2.33)"])
        z = {"90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65, "99% (z=2.33)": 2.33}[svc]

    inv = compute_inventory_table(order_cost, hold_pct, lead_time, z)
    n_crit = (inv["Status"] == "🔴 Critical").sum()
    n_low  = (inv["Status"] == "🟡 Low").sum()
    n_ok   = (inv["Status"] == "🟢 Adequate").sum()
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total SKUs",    len(inv),   "#00e5ff")
    kpi(c2, "🔴 Critical",   n_crit,     "#dc2626", "reorder NOW")
    kpi(c3, "🟡 Low Stock",  n_low,      "#f59e0b", "approaching ROP")
    kpi(c4, "🟢 Adequate",   n_ok,       "#10b981", "well-stocked")
    st.markdown("<br>", unsafe_allow_html=True)

    growth_pct = inv["demand_growth_pct"].iloc[0]
    st.markdown(f"""
    <div style='background:#0a1520;border:1px solid #00e5ff;border-radius:8px;
         padding:10px 16px;margin-bottom:16px;font-size:.85rem;color:#94a3b8'>
      <b style='color:#00e5ff'>Demand Forecast Impact:</b>
      Future demand is projected to grow <b style='color:#10b981'>{growth_pct:+.1f}%</b>
      over the next 6 months (from Demand module). Annual demand forecasts adjusted accordingly.
    </div>""", unsafe_allow_html=True)

    cl, cr = st.columns([1, 2])
    with cl:
        st.markdown("<div class='section-title'>Stock Status Distribution</div>",
                    unsafe_allow_html=True)
        sc = inv["Status"].value_counts()
        
        color_map = {"🔴 Critical": "#dc2626", "🟡 Low": "#f59e0b", "🟢 Adequate": "#10b981"}
        pie_colors = [color_map.get(s, "#64748b") for s in sc.index]
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=.55,
            marker_colors=pie_colors, textinfo="label+value",
            textfont=dict(size=10, color="#e2e8f0")))
        fig.update_layout(**CD(), height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("<div class='section-title'>EOQ / Safety Stock / ROP by Category</div>",
                    unsafe_allow_html=True)
        ci2 = inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2 = go.Figure()
        for i, (m2, lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(name=lbl, x=ci2["Category"], y=ci2[m2].round(1),
                                  marker_color=COLORS[i]))
        fig2.update_layout(**CD(), height=260, barmode="group",
            xaxis=dict(showgrid=False, tickangle=-15),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Future Inventory Need — from Demand Forecast</div>",
                unsafe_allow_html=True)
    demand_fore = compute_demand_forecast()
    fut = demand_fore[demand_fore["type"] == "forecast"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fut["ds"], y=fut["y"], mode="lines+markers",
        line=dict(color="#00e5ff", width=2.5), marker=dict(size=8), name="Qty Forecast"))
    ci_band(fig3, fut, "rgba(0,229,255,0.08)")
    fig3.update_layout(**CD(), height=230,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Units"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-title'>SKU-Level Inventory Recommendations</div>",
                unsafe_allow_html=True)
    cats = st.multiselect("Filter Category", sorted(df["Category"].unique().tolist()),
                          default=sorted(df["Category"].unique().tolist()))
    status_filter = st.multiselect("Filter Status",
                                   ["🔴 Critical", "🟡 Low", "🟢 Adequate"],
                                   default=["🔴 Critical", "🟡 Low", "🟢 Adequate"])
    disp = inv[(inv["Category"].isin(cats)) & (inv["Status"].isin(status_filter))][
        ["SKU_ID","Product_Name","Category","monthly_avg","current_stock",
         "EOQ","SS","ROP","forecast_annual","Status"]
    ].copy()
    disp.columns = ["SKU","Product","Category","Avg/Month","Current Stock",
                    "EOQ","Safety Stock","Reorder Point","Forecast Annual","Status"]
    for c2 in ["Avg/Month","Current Stock","EOQ","Safety Stock","Reorder Point","Forecast Annual"]:
        disp[c2] = disp[c2].round(0).astype(int)
    disp = disp.sort_values("Status")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.session_state["inventory_summary"] = {
        "critical": int(n_crit), "low": int(n_low), "adequate": int(n_ok),
        "growth_pct": float(growth_pct)
    }

def page_production():
    df = load_data()

    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#a78bfa'>
        Production Planning</div>
      <div style='color:#64748b;font-size:.88rem'>
        Monthly production targets · driven by demand forecast + inventory critical stock
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:12px'>
      <span class='feed-badge'>⬆ feeds from: Demand Forecast + Inventory (critical boost)</span>
      <span class='feed-badge' style='border-color:#a78bfa;color:#a78bfa'>
        feeds → Logistics · Chatbot</span>
    </div>""", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    cap = p1.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1,
                    help="Scale all production targets up/down")
    buf = p2.slider("Safety Buffer %", 5, 40, 15,
                    help="Extra % above net demand as production buffer") / 100

    plan = compute_production_plan(cap, buf)
    if plan.empty:
        st.warning("Not enough data to build production plan.")
        return

    agg = plan.groupby("Month_dt")[["Production","Demand","Inv_Boost"]].sum().reset_index()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Production",   f"{plan['Production'].sum():,.0f}", "#a78bfa", "units · 6 mo")
    kpi(c2, "Total Demand",       f"{plan['Demand'].sum():,.0f}",     "#00e5ff", "forecast units")
    kpi(c3, "Avg Monthly Target", f"{agg['Production'].mean():,.0f}", "#10b981", "units/month")
    peak = agg.loc[agg["Production"].idxmax(), "Month_dt"]
    kpi(c4, "Peak Month",         peak.strftime("%b %Y"),             "#ff6b35")
    st.markdown("<br>", unsafe_allow_html=True)

    inv = compute_inventory_table()
    n_crit = (inv["Status"] == "🔴 Critical").sum()
    total_boost = agg["Inv_Boost"].sum()
    st.markdown(f"""
    <div style='background:#0a1520;border:1px solid #dc2626;border-radius:8px;
         padding:10px 16px;margin-bottom:16px;font-size:.85rem;color:#94a3b8'>
      <b style='color:#dc2626'>Inventory Signal:</b>
      {n_crit} SKUs are <b>Critical</b> (from Inventory module).
      Production plan boosted by <b style='color:#f59e0b'>+{total_boost:,.0f} units</b>
      across 6 months to replenish critical stock.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Production Plan vs Demand Forecast</div>",
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Production"],
        name="Production Target", marker_color="#7c3aed"))
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Inv_Boost"],
        name="Inv. Replenishment Boost", marker_color="#dc2626", opacity=0.7))
    fig.add_trace(go.Scatter(x=agg["Month_dt"], y=agg["Demand"], name="Demand Forecast",
        mode="lines+markers", line=dict(color="#00e5ff", width=2.5), marker=dict(size=8)))
    fig.update_layout(**CD(), height=300, barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Production by Category (Stacked)</div>",
                unsafe_allow_html=True)
    fig2 = go.Figure()
    for i, cat in enumerate(plan["Category"].unique()):
        s = plan[plan["Category"] == cat].sort_values("Month_dt")
        fig2.add_trace(go.Bar(x=s["Month_dt"], y=s["Production"],
            name=cat, marker_color=COLORS[i % len(COLORS)]))
    fig2.update_layout(**CD(), height=300, barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Production – Demand Gap</div>",
                unsafe_allow_html=True)
    agg["Gap"] = agg["Production"] - agg["Demand"]
    fig3 = go.Figure(go.Bar(
        x=agg["Month_dt"], y=agg["Gap"],
        marker_color=["#10b981" if g >= 0 else "#dc2626" for g in agg["Gap"]],
        text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside",
        textfont=dict(color="#94a3b8")))
    fig3.add_hline(y=0, line_dash="dash", line_color="#475569")
    fig3.update_layout(**CD(), height=210,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Units Surplus / Deficit"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-title'>Detailed Production Schedule</div>",
                unsafe_allow_html=True)
    filt = st.selectbox("Category filter", ["All"] + list(plan["Category"].unique()))
    d = plan if filt == "All" else plan[plan["Category"] == filt]
    d2 = d[["Month","Category","Demand","Inv_Boost","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d2.columns = ["Month","Category","Demand Forecast","Inv Boost","Safety Buffer",
                  "Production Target","Demand Low","Demand High"]
    st.dataframe(d2.sort_values("Month"), use_container_width=True, hide_index=True)
    st.session_state["production_plan"] = plan

def page_logistics():
    df = load_data()

    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#ff6b35'>
        Logistics Intelligence</div>
      <div style='color:#64748b;font-size:.88rem'>
        Carrier performance · delay hotspots · warehouse demand forecast · region analysis
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:12px'>
      <span class='feed-badge'>⬆ feeds from: Demand Forecast + Production Plan volumes</span>
      <span class='feed-badge' style='border-color:#ff6b35;color:#ff6b35'>
        feeds → Chatbot</span>
    </div>""", unsafe_allow_html=True)

    plan = compute_production_plan()
    prod_by_cat = plan.groupby("Category")["Production"].sum() if not plan.empty else pd.Series()

    t1, t2, t3, t4 = st.tabs(["Carrier", "Delay Intel", "Warehouse Forecast", "Regions"])

    with t1:
        st.markdown("<div class='section-title'>Carrier Performance Scorecard</div>",
                    unsafe_allow_html=True)

        cs = df.groupby("Courier_Partner").agg(
            Orders    = ("Order_ID", "count"),
            Avg_Del   = ("Delivery_Days", "mean"),
            Avg_Cost  = ("Shipping_Cost_INR", "mean"),
            Returns   = ("Return_Flag", "mean"),
            Revenue   = ("Revenue_INR", "sum")
        ).reset_index()
        cs["Delay_Idx"] = (cs["Avg_Del"] / cs["Avg_Del"].min() * (1 + cs["Returns"])).round(2)

        fig = go.Figure()
        for i, row in cs.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Avg_Del"]], y=[row["Avg_Cost"]],
                mode="markers+text",
                marker=dict(size=max(row["Orders"]/40, 14),
                            color=COLORS[i % len(COLORS)], opacity=.85),
                text=[row["Courier_Partner"]], textposition="top center",
                name=row["Courier_Partner"],
                hovertemplate=(
                    f"<b>{row['Courier_Partner']}</b><br>"
                    f"Orders: {row['Orders']}<br>"
                    f"Avg Delivery: {row['Avg_Del']:.1f}d<br>"
                    f"Avg Cost: ₹{row['Avg_Cost']:.0f}<br>"
                    f"Returns: {row['Returns']*100:.1f}%<extra></extra>"
                )
            ))
        fig.update_layout(**CD(), height=320, showlegend=False,
            xaxis=dict(title="Avg Delivery Days", showgrid=True, gridcolor="#1e2d45"),
            yaxis=dict(title="Avg Shipping Cost ₹", showgrid=True, gridcolor="#1e2d45"))
        st.plotly_chart(fig, use_container_width=True)

        d2 = cs.copy()
        d2["Avg_Del"]  = d2["Avg_Del"].round(1)
        d2["Avg_Cost"] = d2["Avg_Cost"].round(1)
        d2["Returns"]  = (d2["Returns"] * 100).round(1).astype(str) + "%"
        d2.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Revenue","Delay Index"]
        st.dataframe(d2, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Historical Carrier Orders Trend</div>",
                    unsafe_allow_html=True)
        cm = df.groupby([df["Order_Date"].dt.to_period("M"), "Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        fig2 = go.Figure()
        for i, c in enumerate(cm.columns):
            fig2.add_trace(go.Scatter(
                x=cm.index.to_timestamp(), y=cm[c], name=c,
                line=dict(color=COLORS[i % len(COLORS)], width=2)))
        fig2.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-title'>Carrier Order Forecast → Jun 2026 (from Demand)</div>",
                    unsafe_allow_html=True)
        fig_fc = go.Figure()
        for i, c in enumerate(cm.columns):
            s = cm[c].rename("value")
            f = forecast_series(s, 6)
            if f.empty: continue
            fut = f[f["type"] == "forecast"]
            fig_fc.add_trace(go.Scatter(
                x=fut["ds"], y=fut["y"], name=c, mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2, dash="dot"),
                marker=dict(size=7)))
        fig_fc.update_layout(**CD(), height=240,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Orders"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_fc, use_container_width=True)

        if not prod_by_cat.empty:
            st.markdown("<div class='section-title'>Carrier Recommendation (based on Production Plan)</div>",
                        unsafe_allow_html=True)
            best_carrier = (
                df.groupby(["Category", "Courier_Partner"])["Delivery_Days"]
                .mean().reset_index()
                .sort_values("Delivery_Days")
                .groupby("Category").first()
                .reset_index()
            )
            best_carrier = best_carrier.merge(
                prod_by_cat.rename("Planned Units").reset_index(), on="Category", how="left")
            best_carrier["Avg Delivery Days"] = best_carrier["Delivery_Days"].round(1)
            best_carrier = best_carrier[["Category","Courier_Partner","Avg Delivery Days","Planned Units"]]
            best_carrier.columns = ["Category","Recommended Carrier","Avg Days","Planned Units (6mo)"]
            best_carrier["Planned Units (6mo)"] = best_carrier["Planned Units (6mo)"].fillna(0).astype(int)
            st.dataframe(best_carrier, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("<div class='section-title'>Delay Hotspot Analysis</div>",
                    unsafe_allow_html=True)
        thr = st.slider("Delay Threshold (days)", 3, 15, 7)
        df2 = df.copy()
        df2["Delayed"] = df2["Delivery_Days"] > thr
        rd = df2.groupby("Region").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
        rd["Rate"] = (rd["D"] / rd["T"] * 100).round(1)
        rd = rd.sort_values("Rate", ascending=False)
        cl, cr = st.columns(2)
        with cl:
            st.markdown("**Delay Rate by Region**")
            fig_r = go.Figure(go.Bar(
                x=rd["Rate"], y=rd["Region"], orientation="h",
                marker_color=[f"rgba(220,38,38,{min(v/100+.25,1):.2f})" for v in rd["Rate"]],
                text=[f"{v}%" for v in rd["Rate"]], textposition="outside",
                textfont=dict(color="#94a3b8")))
            fig_r.update_layout(**CD(), height=320,
                xaxis=dict(showgrid=False, title="Delay %"), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_r, use_container_width=True)
        with cr:
            st.markdown("**Delay Rate by Carrier**")
            cd = df2.groupby("Courier_Partner").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
            cd["Rate"] = (cd["D"] / cd["T"] * 100).round(1)
            fig_c = go.Figure(go.Bar(
                x=cd["Courier_Partner"], y=cd["Rate"],
                marker_color=["#dc2626" if v > 30 else "#f59e0b" if v > 15 else "#10b981"
                               for v in cd["Rate"]],
                text=[f"{v}%" for v in cd["Rate"]], textposition="outside",
                textfont=dict(color="#94a3b8")))
            fig_c.update_layout(**CD(), height=320,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Delay %"))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("<div class='section-title'>Carrier × Region Delay Heatmap</div>",
                    unsafe_allow_html=True)
        pv = df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0) * 100
        fig_h = go.Figure(go.Heatmap(
            z=pv.values, x=list(pv.columns), y=list(pv.index),
            colorscale=[[0,"#0f172a"],[0.5,"#7c3aed"],[1,"#dc2626"]],
            text=np.round(pv.values, 1), texttemplate="%{text}%",
            colorbar=dict(tickfont=dict(color="#94a3b8"))))
        fig_h.update_layout(**CD(), height=260,
            xaxis=dict(showgrid=False, tickangle=-30), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("<div class='section-title'>Delivery Delay Trend Forecast → Jun 2026</div>",
                    unsafe_allow_html=True)
        delay_monthly = df.groupby("YearMonth")["Delivery_Days"].mean().rename("value")
        f_delay = forecast_series(delay_monthly, 6)
        if not f_delay.empty:
            hist_d = f_delay[f_delay["type"]=="historical"]
            fut_d  = f_delay[f_delay["type"]=="forecast"]
            fig_delay = go.Figure()
            fig_delay.add_trace(go.Scatter(
                x=hist_d["ds"], y=hist_d["y"], name="Historical Avg Delay",
                line=dict(color="#64748b", width=2)))
            fig_delay.add_trace(go.Scatter(
                x=fut_d["ds"], y=fut_d["y"], name="Forecast",
                line=dict(color="#ff6b35", width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=8)))
            ci_band(fig_delay, fut_d, "rgba(255,107,53,0.1)")
            fig_delay.update_layout(**CD(), height=250,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Avg Delivery Days"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_delay, use_container_width=True)

    with t3:
        st.markdown("<div class='section-title'>Warehouse Shipment Trend (Historical)</div>",
                    unsafe_allow_html=True)
        wm = df.groupby([df["Order_Date"].dt.to_period("M"), "Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh = go.Figure()
        for i, wh in enumerate(wm.columns):
            fig_wh.add_trace(go.Bar(x=wm.index.to_timestamp(), y=wm[wh],
                name=wh, marker_color=COLORS[i % len(COLORS)]))
        fig_wh.update_layout(**CD(), height=280, barmode="stack",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_wh, use_container_width=True)

        st.markdown("<div class='section-title'>Warehouse Demand Forecast → Jun 2026</div>",
                    unsafe_allow_html=True)
        wf_rows = []
        for wh in wm.columns:
            s = wm[wh].rename("value")
            if len(s.dropna()) < 4: continue
            f = forecast_series(s, 6)
            fut = f[f["type"] == "forecast"]
            for _, r in fut.iterrows():
                wf_rows.append({"Month": r["ds"], "Warehouse": wh,
                                 "Forecast": r["y"], "CI_Hi": r["yhat_upper"]})
        if wf_rows:
            wfd = pd.DataFrame(wf_rows)
            fig_wf = go.Figure()
            for i, wh in enumerate(wfd["Warehouse"].unique()):
                s = wfd[wfd["Warehouse"] == wh]
                fig_wf.add_trace(go.Scatter(
                    x=s["Month"], y=s["Forecast"], name=wh, mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)], width=2.5, dash="dot"),
                    marker=dict(size=8)))
            fig_wf.update_layout(**CD(), height=260,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_wf, use_container_width=True)

            wfd_tbl = wfd.copy()
            wfd_tbl["Month"]    = wfd_tbl["Month"].dt.strftime("%b %Y")
            wfd_tbl["Forecast"] = wfd_tbl["Forecast"].round(0).astype(int)
            wfd_tbl["CI_Hi"]    = wfd_tbl["CI_Hi"].round(0).astype(int)
            wfd_tbl.columns     = ["Month","Warehouse","Forecast Units","Upper Bound"]
            st.dataframe(wfd_tbl.sort_values(["Month","Warehouse"]),
                         use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Top Products per Warehouse</div>",
                    unsafe_allow_html=True)
        wsel = st.selectbox("Warehouse", sorted(df["Warehouse"].unique()))
        tp = (df[df["Warehouse"] == wsel]
              .groupby("Product_Name")["Quantity"].sum()
              .sort_values(ascending=False).head(10))
        fig_tp = go.Figure(go.Bar(
            x=tp.values, y=tp.index, orientation="h",
            marker_color="#00e5ff", text=tp.values, textposition="outside",
            textfont=dict(color="#94a3b8")))
        fig_tp.update_layout(**CD(), height=310,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_tp, use_container_width=True)

    with t4:
        st.markdown("<div class='section-title'>Region Performance Overview</div>",
                    unsafe_allow_html=True)
        rs = df.groupby("Region").agg(
            Orders  = ("Order_ID", "count"),
            Revenue = ("Revenue_INR", "sum"),
            Qty     = ("Quantity", "sum"),
            Avg_Del = ("Delivery_Days", "mean"),
            Returns = ("Return_Flag", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)

        met = st.selectbox("Metric", ["Revenue","Orders","Qty","Avg_Del","Returns"])
        fig_r = go.Figure(go.Bar(
            x=rs["Region"], y=rs[met],
            marker=dict(color=list(rs[met].values),
                        colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]]),
            text=rs[met].round(1), textposition="outside",
            textfont=dict(color="#94a3b8")))
        fig_r.update_layout(**CD(), height=300,
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"))
        st.plotly_chart(fig_r, use_container_width=True)

        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown("<div class='section-title'>Best Carrier per Region</div>",
                        unsafe_allow_html=True)
            bc = (df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
                  .mean().reset_index().sort_values("Delivery_Days")
                  .groupby("Region").first().reset_index())
            bc.columns = ["Region","Best Carrier","Avg Days"]
            bc["Avg Days"] = bc["Avg Days"].round(1)
            st.dataframe(bc, use_container_width=True, hide_index=True)

        with c_r:
            st.markdown("<div class='section-title'>Region Return Rate Ranking</div>",
                        unsafe_allow_html=True)
            rr = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False) * 100
            fig_ret = go.Figure(go.Bar(
                x=rr.values, y=rr.index, orientation="h",
                marker_color=["#dc2626" if v > 20 else "#f59e0b" if v > 12 else "#10b981"
                               for v in rr.values],
                text=[f"{v:.1f}%" for v in rr.values], textposition="outside",
                textfont=dict(color="#94a3b8")))
            fig_ret.update_layout(**CD(), height=280,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("<div class='section-title'>Region Revenue Forecast → Jun 2026</div>",
                    unsafe_allow_html=True)
        top_reg = df["Region"].value_counts().head(5).index.tolist()
        fig_rf = go.Figure()
        for i, reg in enumerate(top_reg):
            s = df[df["Region"]==reg].groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f = forecast_series(s, 6)
            if f.empty: continue
            hist_r = f[f["type"]=="historical"]
            fut_r  = f[f["type"]=="forecast"]
            fig_rf.add_trace(go.Scatter(
                x=hist_r["ds"], y=hist_r["y"], name=f"{reg} (hist)",
                line=dict(color=COLORS[i], width=1.5, dash="solid"),
                opacity=0.4, showlegend=False))
            fig_rf.add_trace(go.Scatter(
                x=fut_r["ds"], y=fut_r["y"], name=reg,
                mode="lines+markers",
                line=dict(color=COLORS[i], width=2.5, dash="dot"),
                marker=dict(size=8)))
        fig_rf.update_layout(**CD(), height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rf, use_container_width=True)

def build_context(df: pd.DataFrame) -> str:
    m_qty   = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    d_fore  = forecast_series(m_qty, 6)
    fut_d   = d_fore[d_fore["type"]=="forecast"]
    demand_str = "; ".join([f"{r['ds'].strftime('%b%Y')}:{r['y']:.0f}u"
                             for _, r in fut_d.iterrows()])

    m_rev  = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
    r_fore = forecast_series(m_rev, 6)
    fut_r  = r_fore[r_fore["type"]=="forecast"]
    rev_str = "; ".join([f"{r['ds'].strftime('%b%Y')}:₹{r['y']/1e6:.1f}M"
                          for _, r in fut_r.iterrows()])

    inv = compute_inventory_table()
    n_crit = (inv["Status"]=="🔴 Critical").sum()
    n_low  = (inv["Status"]=="🟡 Low").sum()
    growth = inv["demand_growth_pct"].iloc[0]
    crit_skus = ", ".join(inv[inv["Status"]=="🔴 Critical"]["Product_Name"].head(5).tolist())

    plan = compute_production_plan()
    if not plan.empty:
        prod_sum = plan.groupby("Category")["Production"].sum()
        prod_str = ", ".join([f"{k}:{v:.0f}" for k,v in prod_sum.items()])
        peak_mo  = plan.groupby("Month_dt")["Production"].sum().idxmax().strftime("%b %Y")
    else:
        prod_str = "N/A"; peak_mo = "N/A"

    cs = df.groupby("Courier_Partner").agg(
        n=("Order_ID","count"), d=("Delivery_Days","mean"), r=("Return_Flag","mean"))
    carr_str = "; ".join([f"{r}:{d['n']}ord,{d['d']:.1f}d,{d['r']*100:.1f}%ret"
                           for r, d in cs.iterrows()])
    bc = (df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
          .mean().reset_index().sort_values("Delivery_Days")
          .groupby("Region").first().reset_index())
    bc_str = ", ".join([f"{row['Region']}→{row['Courier_Partner']}"
                         for _, row in bc.iterrows()])

    top_reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    reg_str = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in top_reg.items()])
    cat_rev = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
    cat_str = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in cat_rev.items()])
    wh_rev  = df.groupby("Warehouse")["Revenue_INR"].sum().sort_values(ascending=False)
    wh_str  = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in wh_rev.items()])
    ret_hot = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False).head(4)
    ret_str = ", ".join([f"{k}:{v*100:.1f}%" for k,v in ret_hot.items()])
    top_sku = df.groupby("Product_Name")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    sku_str = ", ".join(top_sku.index.tolist())

    return f"""=== OmniFlow D2D India — Live Supply Chain Intelligence Context ===
DATASET: 5,200 orders | Jan 2024–Dec 2025 | India D2D (Amazon, Flipkart, Meesho, B2B)
SUMMARY: Revenue ₹{df['Revenue_INR'].sum()/1e7:.2f}Cr | Orders {len(df):,} | Return {df['Return_Flag'].mean()*100:.1f}% | Avg Del {df['Delivery_Days'].mean():.1f}d

[MODULE 1 — DEMAND FORECAST → Jun 2026]
Qty Forecast: {demand_str}
Rev Forecast: {rev_str}
Demand Growth (6mo avg vs recent): +{growth:.1f}%

[MODULE 2 — INVENTORY STATUS]
Critical SKUs: {n_crit} | Low: {n_low} | Adequate: {inv['Status'].eq('🟢 Adequate').sum()}
Critical Products (reorder NOW): {crit_skus}
EOQ/SS/ROP computed per SKU using demand growth adjustment

[MODULE 3 — PRODUCTION PLAN (6-month targets by category)]
{prod_str}
Peak Production Month: {peak_mo}
(Production boosted for Critical inventory SKUs)

[MODULE 4 — LOGISTICS]
Carriers: {carr_str}
Best Carrier per Region: {bc_str}
Warehouses by Revenue: {wh_str}
High Return Regions: {ret_str}

CATEGORIES: {cat_str}
TOP REGIONS: {reg_str}
TOP PRODUCTS: {sku_str}"""

SUGGESTIONS = [
    "Which carrier should I use for Maharashtra orders?",
    "What products will peak in April 2026?",
    "Which regions have critical stock risk?",
    "How to adjust production for June 2026?",
    "Best warehouse for Electronics shipments?",
    "Which category grows fastest and why?",
    "How can we reduce the return rate?",
    "Optimal reorder strategy for Home & Kitchen?",
]

def call_claude(messages, system):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY not found. Add it in Streamlit secrets."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        
        if response.status_code == 429:
            return "⚠️ Rate limit hit. Please wait a few seconds and try again."
        
        if response.status_code != 200:
            return f"⚠️ Groq API Error ({response.status_code}): {response.text}"
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
def page_chatbot():
    df  = load_data()

    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#f59e0b'>
        Decision Intelligence Chatbot</div>
      <div style='color:#64748b;font-size:.88rem'>
        Ask anything · Claude AI · live context from all 4 upstream modules
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:12px'>
      <span class='feed-badge'>⬆ context from: Demand + Inventory + Production + Logistics</span>
    </div>""", unsafe_allow_html=True)

    ctx = build_context(df)

    with st.expander("Live Module Context (fed to AI)", expanded=False):
        st.code(ctx, language="text")

    system_prompt = f"""You are OmniFlow, an expert AI supply chain decision assistant for an India D2D e-commerce operation.
Your expertise spans demand forecasting, inventory management (EOQ, safety stock, reorder points),
production planning, and logistics optimization across Indian regions and courier partners.

Rules:
- Give specific, actionable, data-backed recommendations
- Always use ₹ for Indian Rupees
- Use bullet points for multi-part answers
- Reference exact numbers from the context
- Consider module interdependencies: Demand → Inventory → Production → Logistics
- Flag risks, stockout risks, delay risks, and seasonal patterns proactively
- Be concise but comprehensive; lead with the key recommendation

LIVE SUPPLY CHAIN CONTEXT (all 4 modules):
{ctx}"""

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    if not st.session_state.chat_msgs:
        st.markdown("<div class='section-title'>Quick Questions</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, s in enumerate(SUGGESTIONS):
            with cols[i % 4]:
                if st.button(s, key=f"q{i}", use_container_width=True):
                    st.session_state.chat_msgs.append({"role": "user", "content": s})
                    with st.spinner("OmniFlow AI thinking…"):
                        rep = call_claude([{"role":"user","content":s}], system_prompt)
                    st.session_state.chat_msgs.append({"role":"assistant","content":rep})
                    st.rerun()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_msgs:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='chat-user'>
                  <div class='chat-user-bubble'>{msg['content']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                content = (msg["content"]
                           .replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;")
                           .replace("\n", "<br>"))
                st.markdown(f"""
                <div class='chat-ai'>
                  <div class='chat-avatar'></div>
                  <div class='chat-ai-bubble'>{content}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_inp, c_btn, c_clr = st.columns([5, 1, 1])
    with c_inp:
        user_in = st.text_input(
            "Ask a supply chain question…", key="user_input",
            placeholder="e.g. What production adjustments are needed for Q2 2026?",
            label_visibility="collapsed")
    with c_btn:
        send = st.button("Send", use_container_width=True)
    with c_clr:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_msgs = []
            st.rerun()

    if send and user_in.strip():
        st.session_state.chat_msgs.append({"role":"user","content":user_in.strip()})
        api_msgs = st.session_state.chat_msgs[-14:]  
        with st.spinner("OmniFlow AI thinking…"):
            rep = call_claude(api_msgs, system_prompt)
        st.session_state.chat_msgs.append({"role":"assistant","content":rep})
        st.rerun()

    if not st.session_state.chat_msgs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚡ Live Decision Alerts</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**🔴 Critical Inventory SKUs (reorder immediately)**")
            inv = compute_inventory_table()
            crit = inv[inv["Status"] == "🔴 Critical"][["Product_Name","Category","current_stock","ROP"]].head(5)
            for _, row in crit.iterrows():
                st.markdown(f"""
                <div style='background:#111827;border-left:3px solid #dc2626;
                    padding:8px 12px;margin:5px 0;border-radius:0 6px 6px 0;font-size:.82rem'>
                  <b>{row['Product_Name']}</b> [{row['Category']}]<br>
                  <span style='color:#64748b'>Stock: {row['current_stock']} | ROP: {row['ROP']}</span>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown("**Revenue Forecast Alerts (next 3 months)**")
            m = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f = forecast_series(m, 3)
            fut = f[f["type"] == "forecast"]
            last = float(m.iloc[-1])
            for _, row in fut.iterrows():
                chg  = (row["y"] - last) / last * 100
                clr  = "#10b981" if chg >= 0 else "#dc2626"
                icon = "📈" if chg >= 0 else "📉"
                st.markdown(f"""
                <div style='background:#111827;border-left:3px solid {clr};
                    padding:8px 12px;margin:5px 0;border-radius:0 6px 6px 0;font-size:.82rem'>
                  {icon} <b>{row['ds'].strftime("%b %Y")}</b>
                  — ₹{row['y']/1e6:.1f}M forecast ({chg:+.1f}% vs prev)
                </div>""", unsafe_allow_html=True)
                last = row["y"]

        st.markdown("**⚠️ High Delay + Return Regions**")
        df["Delayed"] = df["Delivery_Days"] > 7
        risk = df.groupby("Region").agg(
            Delay_Rate=("Delayed","mean"), Return_Rate=("Return_Flag","mean")).reset_index()
        risk["Risk_Score"] = risk["Delay_Rate"] * 0.5 + risk["Return_Rate"] * 0.5
        risk = risk.sort_values("Risk_Score", ascending=False).head(4)
        cc = st.columns(4)
        for i, (_, row) in enumerate(risk.iterrows()):
            with cc[i]:
                clr = "#dc2626" if row["Risk_Score"] > 0.2 else "#f59e0b"
                st.markdown(f"""
                <div class='metric-card' style='border-color:{clr}'>
                  <div class='metric-label'>{row['Region']}</div>
                  <div style='color:{clr};font-size:1.2rem;font-weight:800'>
                    {row['Risk_Score']*100:.1f}%</div>
                  <div style='color:#64748b;font-size:.72rem'>
                    Del: {row['Delay_Rate']*100:.1f}% | Ret: {row['Return_Rate']*100:.1f}%
                  </div>
                </div>""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='padding:12px 0 20px'>
  <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#00e5ff'>
    OmniFlow D2D</div>
</div>""", unsafe_allow_html=True)

PAGES = {
    "Overview":               page_overview,
    "Demand Forecasting":     page_demand,
    "Inventory Optimization": page_inventory,
    "Production Planning":    page_production,
    "Logistics Intelligence": page_logistics,
    "Decision Chatbot":       page_chatbot,
}

sel = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:.72rem;color:#64748b;line-height:1.9'>
Data: Jan 2024 – Dec 2025<br>
Forecast: to Jun 2026<br>
5,200 orders | 50 SKUs<br>
🇮🇳 India D2D Supply Chain<br>
Powered by Claude AI<br><br>
<b style='color:#475569'>Module Flow:</b><br>
Demand → Inventory<br>
→ Production → Logistics<br>
→ Chatbot
</div>""", unsafe_allow_html=True)

PAGES[sel]()
