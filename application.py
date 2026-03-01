"""
OmniFlow D2D Supply Chain Intelligence
Streamlit Cloud entry point: application.py
REDESIGNED: Premium dark UI with glassmorphism, animations, and cohesive color system
"""

import streamlit as st

st.set_page_config(
    page_title="OmniFlow D2D Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800&display=swap');

:root {
    /* Core palette */
    --midnight: #080e1a;
    --deep:     #0d1829;
    --surface:  #111e30;
    --panel:    #162236;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.12);
    
    /* Accents */
    --amber:    #f5a623;
    --coral:    #ff6b6b;
    --teal:     #2ed8c3;
    --sky:      #5ba4e5;
    --lavender: #9b87d4;
    --mint:     #56e0a0;
    
    /* Text */
    --text-1:   #f0f4ff;
    --text-2:   #8a9dc0;
    --text-3:   #4a5e7a;
    
    /* Shadows */
    --glow-amber: 0 0 40px rgba(245,166,35,0.15);
    --glow-teal:  0 0 40px rgba(46,216,195,0.12);
    --card-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.05);
}

/* === RESET & BASE === */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--midnight) !important;
    color: var(--text-1) !important;
}

/* Gradient mesh background */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(91,164,229,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(46,216,195,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(245,166,35,0.03) 0%, transparent 70%),
        var(--midnight) !important;
    min-height: 100vh;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--deep) 0%, var(--midnight) 100%) !important;
    border-right: 1px solid var(--border2) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5) !important;
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--amber), var(--coral), var(--teal));
}

/* === TYPOGRAPHY === */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.03em;
}

/* === METRIC CARDS === */
.metric-card {
    background: linear-gradient(135deg, var(--panel) 0%, var(--surface) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
    transition: transform 0.4s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.4s ease;
    cursor: default;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.amber::before { background: linear-gradient(90deg, var(--amber), #ff8c42); box-shadow: 0 0 20px rgba(245,166,35,0.5); }
.metric-card.teal::before  { background: linear-gradient(90deg, var(--teal), #22b8a5); box-shadow: 0 0 20px rgba(46,216,195,0.5); }
.metric-card.coral::before { background: linear-gradient(90deg, var(--coral), #ff4f4f); box-shadow: 0 0 20px rgba(255,107,107,0.5); }
.metric-card.sky::before   { background: linear-gradient(90deg, var(--sky), #3d87d4); box-shadow: 0 0 20px rgba(91,164,229,0.5); }
.metric-card.lav::before   { background: linear-gradient(90deg, var(--lavender), #7b6bbf); box-shadow: 0 0 20px rgba(155,135,212,0.5); }
.metric-card.mint::before  { background: linear-gradient(90deg, var(--mint), #3ec47a); box-shadow: 0 0 20px rgba(86,224,160,0.5); }

.metric-card::after {
    content: "";
    position: absolute;
    bottom: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.02);
    transition: all 0.4s ease;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 48px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.08);
    border-color: var(--border2);
}
.metric-card:hover::after {
    bottom: -10px; right: -10px;
    width: 130px; height: 130px;
}

.metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    color: var(--text-3);
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace !important;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
}
.metric-sub {
    font-size: 0.72rem;
    color: var(--text-3);
    margin-top: 6px;
    font-family: 'DM Mono', monospace !important;
}

/* === SECTION HEADERS === */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
}
.section-title {
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-2);
    font-family: 'DM Mono', monospace !important;
}

/* === PAGE TITLE BLOCK === */
.page-title-block {
    padding: 28px 0 20px;
    position: relative;
}
.page-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 6px;
}
.page-subtitle {
    color: var(--text-3);
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace !important;
}

/* === FEED BADGES === */
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid;
    font-family: 'DM Mono', monospace !important;
    transition: all 0.2s ease;
}
.badge:hover { transform: scale(1.05); }
.badge-amber { background: rgba(245,166,35,0.1); color: var(--amber); border-color: rgba(245,166,35,0.3); }
.badge-teal  { background: rgba(46,216,195,0.1); color: var(--teal); border-color: rgba(46,216,195,0.3); }
.badge-coral { background: rgba(255,107,107,0.1); color: var(--coral); border-color: rgba(255,107,107,0.3); }
.badge-sky   { background: rgba(91,164,229,0.1); color: var(--sky); border-color: rgba(91,164,229,0.3); }
.badge-lav   { background: rgba(155,135,212,0.1); color: var(--lavender); border-color: rgba(155,135,212,0.3); }

/* === INFO BANNERS === */
.info-banner {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0 20px;
    font-size: 0.83rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.info-banner::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
}
.banner-teal   { background: rgba(46,216,195,0.06); border-color: rgba(46,216,195,0.2); color: var(--text-2); }
.banner-teal::before { background: var(--teal); }
.banner-coral  { background: rgba(255,107,107,0.06); border-color: rgba(255,107,107,0.2); color: var(--text-2); }
.banner-coral::before { background: var(--coral); }
.banner-amber  { background: rgba(245,166,35,0.06); border-color: rgba(245,166,35,0.2); color: var(--text-2); }
.banner-amber::before { background: var(--amber); }

/* === ABOUT CARD === */
.about-card {
    background: linear-gradient(135deg, rgba(22,34,54,0.9) 0%, rgba(17,30,48,0.9) 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.about-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--amber), var(--coral), var(--teal), var(--sky));
}
.about-card::after {
    content: "⬡";
    position: absolute;
    right: 28px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.04;
    color: white;
    pointer-events: none;
}

/* === FLOW DIAGRAM === */
.flow-wrap {
    background: linear-gradient(135deg, var(--panel), var(--surface));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
    margin-top: 8px;
}
.flow-node {
    background: var(--deep);
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 700;
    font-size: 0.8rem;
    text-align: center;
    min-width: 100px;
    position: relative;
    border: 1px solid var(--border2);
    transition: all 0.3s ease;
    font-family: 'DM Mono', monospace !important;
}
.flow-node:hover {
    transform: scale(1.06);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.flow-node span {
    display: block;
    font-size: 0.6rem;
    font-weight: 400;
    color: var(--text-3);
    margin-top: 3px;
    letter-spacing: 0.05em;
}
.flow-arrow {
    color: var(--text-3);
    font-size: 1.4rem;
    padding: 0 8px;
    font-weight: 300;
    opacity: 0.4;
}

/* === CHART CONTAINER === */
.chart-panel {
    background: linear-gradient(135deg, var(--panel) 0%, var(--surface) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 20px 12px;
    box-shadow: var(--card-shadow);
    margin-bottom: 20px;
    transition: border-color 0.3s ease;
}
.chart-panel:hover {
    border-color: var(--border2);
}

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, rgba(245,166,35,0.15), rgba(255,107,107,0.1)) !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,166,35,0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    font-family: 'DM Mono', monospace !important;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(245,166,35,0.25), rgba(255,107,107,0.15)) !important;
    border-color: rgba(245,166,35,0.6) !important;
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 20px rgba(245,166,35,0.2) !important;
    color: #ffe066 !important;
}

/* === INPUTS === */
.stTextInput input, .stSelectbox select {
    background: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput input:focus {
    border-color: rgba(245,166,35,0.5) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.1) !important;
}

/* === DATAFRAME === */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* === CHAT BUBBLES === */
.chat-user-bubble {
    background: linear-gradient(135deg, rgba(245,166,35,0.12), rgba(255,107,107,0.08));
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: var(--text-1);
    margin-left: 20%;
}
.chat-ai-bubble {
    background: linear-gradient(135deg, var(--panel), var(--surface));
    border: 1px solid var(--border2);
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: var(--text-2);
    line-height: 1.7;
    margin-right: 10%;
}
.chat-spacing { margin: 10px 0; }

/* === ALERT CARDS === */
.alert-item {
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.8rem;
    border-left: 3px solid;
    background: var(--panel);
    transition: all 0.25s ease;
    font-family: 'DM Mono', monospace !important;
}
.alert-item:hover {
    transform: translateX(4px);
    background: var(--surface);
}
.alert-critical { border-color: var(--coral); }
.alert-warn     { border-color: var(--amber); }
.alert-ok       { border-color: var(--mint); }

/* === ANIMATIONS === */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(245,166,35,0.2); }
    50%       { border-color: rgba(245,166,35,0.5); }
}

.metric-card     { animation: fadeUp 0.5s ease both; }
.chart-panel     { animation: fadeUp 0.6s ease both; }
.about-card      { animation: fadeIn 0.7s ease both; }
.badge           { animation: fadeIn 0.4s ease both; }
.flow-wrap       { animation: fadeUp 0.8s ease both; }

/* Stagger cards */
.metric-card:nth-child(1) { animation-delay: 0.05s; }
.metric-card:nth-child(2) { animation-delay: 0.10s; }
.metric-card:nth-child(3) { animation-delay: 0.15s; }
.metric-card:nth-child(4) { animation-delay: 0.20s; }
.metric-card:nth-child(5) { animation-delay: 0.25s; }
.metric-card:nth-child(6) { animation-delay: 0.30s; }

/* === SIDEBAR RADIO === */
.stRadio label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-2) !important;
    letter-spacing: 0.04em !important;
    padding: 8px 4px !important;
    transition: color 0.2s ease !important;
}
.stRadio label:hover { color: var(--amber) !important; }

/* === EXPANDER === */
.streamlit-expanderHeader {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* === COLUMNS SPACING === */
[data-testid="column"] > div { padding: 0 8px; }

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-3) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(245,166,35,0.2), rgba(255,107,107,0.1)) !important;
    color: var(--amber) !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--midnight); }
::-webkit-scrollbar-thumb { background: var(--panel); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--border2); }

/* === PLOTLY === */
.js-plotly-plot, .plot-container { background: transparent !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, os

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "OmniFlow_D2D_India_Unified_5200.csv")

# Refined plotly color palette — warm, earthy, harmonious
COLORS = ["#f5a623", "#56e0a0", "#ff6b6b", "#5ba4e5", "#e87adb", "#2ed8c3"]
COLORS_SOFT = ["#e8963f", "#4ecf94", "#e85c5c", "#4d90d4", "#cc68c4", "#28c4b0"]

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
    sku_agg["EOQ"] = sku_agg.apply(lambda r: eoq(r["annual_demand"], order_cost, hold_pct, r["avg_price"]), axis=1)
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
        .groupby("Category")["monthly_avg"].sum().rename("critical_monthly")
    )
    cat_monthly = df.groupby(["YearMonth", "Category"])["Quantity"].sum().unstack(fill_value=0)
    plans = []
    for cat in cat_monthly.columns:
        series = cat_monthly[cat].rename("value")
        fore   = forecast_series(series, 6)
        if fore.empty: continue
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a9dc0", family="DM Mono, monospace", size=11),
        margin=dict(l=10, r=10, t=36, b=16)
    )

def grid_y():
    return dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", gridwidth=1,
                zeroline=False, tickcolor="#4a5e7a")

def grid_x():
    return dict(showgrid=False, zeroline=False, tickcolor="#4a5e7a")

def kpi(col, label, value, accent_class="amber", sub=""):
    col.markdown(f"""
    <div class='metric-card {accent_class}'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def section_title(label, emoji=""):
    st.markdown(f"""
    <div class='section-header'>
      <div class='section-title'>{emoji} {label}</div>
      <div class='section-header-line'></div>
    </div>""", unsafe_allow_html=True)

def ci_band(fig, fore, color="rgba(245,166,35,0.08)"):
    ds_fwd  = list(fore["ds"]) + list(fore["ds"])[::-1]
    y_band  = list(fore["yhat_upper"]) + list(fore["yhat_lower"])[::-1]
    fig.add_trace(go.Scatter(
        x=ds_fwd, y=y_band,
        fill="toself", fillcolor=color,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=False
    ))

def legend_style():
    return dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)",
                borderwidth=1, font=dict(color="#8a9dc0", size=10))

def page_overview():
    df  = load_data()
    raw = load_all_statuses()

    st.markdown("""
    <div class='page-title-block'>
      <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#4a5e7a;
           letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px'>
        ⬡ D2D SUPPLY CHAIN INTELLIGENCE PLATFORM
      </div>
      <div class='page-title' style='background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:3rem'>
        OmniFlow</div>
      <div class='page-subtitle'>Predictive Logistics · AI-Powered · Demand-to-Delivery Optimization · India</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
      <div style='font-family:Outfit,sans-serif;font-size:1.05rem;font-weight:700;
           color:#f0f4ff;margin-bottom:10px'>About This Platform</div>
      <p style='color:#8a9dc0;line-height:1.85;margin:0;font-size:0.88rem'>
        <b style='color:#f5a623'>OmniFlow</b> is an AI-driven supply chain intelligence platform
        built on <b style='color:#f0f4ff'>5,200 D2D orders</b> across India (Jan 2024–Dec 2025).
        Six interconnected modules feed each other in sequence — demand signals drive inventory,
        which drives production, which informs logistics. The AI chatbot synthesises all module outputs.
      </p>
      <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:16px'>
        <span class='badge badge-amber'>Demand → Jun 2026</span>
        <span class='badge badge-teal'>Inventory EOQ/ROP</span>
        <span class='badge badge-lav'>Production Plan</span>
        <span class='badge badge-coral'>Logistics Intel</span>
        <span class='badge badge-sky'>AI Chatbot</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, "Total Revenue",  f"₹{df['Revenue_INR'].sum()/1e7:.1f}Cr", "amber",  "all time")
    kpi(c2, "Orders",         f"{len(df):,}",                           "teal",   "non-cancelled")
    kpi(c3, "Units Sold",     f"{df['Quantity'].sum():,}",              "sky",    "quantities")
    kpi(c4, "Return Rate",    f"{df['Return_Flag'].mean()*100:.1f}%",   "coral",  "of delivered")
    kpi(c5, "Avg Delivery",   f"{df['Delivery_Days'].mean():.1f}d",     "lav",    "avg days")
    kpi(c6, "SKU Categories", f"{df['Category'].nunique()}",            "mint",   "product types")
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    c_l, c_r = st.columns([3, 2], gap="large")
    with c_l:
        section_title("Monthly Revenue Trend", "📈")
        m = df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue_INR"].sum().reset_index()
        m["Order_Date"] = m["Order_Date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m["Order_Date"], y=m["Revenue_INR"],
            fill="tozeroy",
            line=dict(color="#f5a623", width=2.5),
            fillcolor="rgba(245,166,35,0.06)",
            name="Revenue",
            hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(**CD(), height=260,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y(), tickformat=",.0f"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        section_title("Revenue by Category", "🥧")
        cat = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(
            labels=cat.index, values=cat.values, hole=.6,
            marker=dict(colors=COLORS, line=dict(color="#080e1a", width=3)),
            textinfo="label+percent",
            textfont=dict(size=10, color="#f0f4ff"),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
        ))
        fig2.update_layout(**CD(), height=260,
            showlegend=False,
            annotations=[dict(text="Revenue", x=0.5, y=0.5, showarrow=False,
                             font=dict(size=11, color="#4a5e7a", family="DM Mono"))])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c3a, c3b, c3c = st.columns(3, gap="large")

    with c3a:
        section_title("Orders by Channel", "📦")
        ch = df["Sales_Channel"].value_counts().head(6)
        fig3 = go.Figure(go.Bar(
            x=ch.values, y=ch.index, orientation="h",
            marker=dict(
                color=COLORS[:len(ch)],
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=ch.values, textposition="outside",
            textfont=dict(color="#4a5e7a", size=10),
            hovertemplate="<b>%{y}</b><br>%{x:,} orders<extra></extra>"
        ))
        fig3.update_layout(**CD(), height=250,
            xaxis=dict(**grid_x()), yaxis=dict(showgrid=False, color="#8a9dc0"))
        st.plotly_chart(fig3, use_container_width=True)

    with c3b:
        section_title("Top Regions Revenue", "🗺️")
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(8)
        fig4 = go.Figure(go.Bar(
            x=reg.index, y=reg.values,
            marker=dict(
                color=COLORS_SOFT * 2,
                line=dict(color="rgba(0,0,0,0)")
            ),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
        ))
        fig4.update_layout(**CD(), height=250,
            xaxis=dict(**grid_x(), tickangle=-30),
            yaxis=dict(**grid_y()))
        st.plotly_chart(fig4, use_container_width=True)

    with c3c:
        section_title("Order Status Split", "🔄")
        sc = raw["Order_Status"].value_counts()
        colors_sc = ["#56e0a0", "#ff6b6b", "#f5a623", "#5ba4e5", "#2ed8c3"]
        fig5 = go.Figure(go.Bar(
            x=sc.index, y=sc.values,
            marker=dict(color=colors_sc[:len(sc)], line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>"
        ))
        fig5.update_layout(**CD(), height=250,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y()))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_title("Module Dependency Flow", "⬡")
    st.markdown("""
    <div class='flow-wrap'>
      <div class='flow-node' style='border-color:rgba(245,166,35,0.4);color:#f5a623'>
        Demand<span>→ Jun 2026</span></div>
      <div class='flow-arrow'>→</div>
      <div class='flow-node' style='border-color:rgba(86,224,160,0.4);color:#56e0a0'>
        Inventory<span>EOQ + ROP</span></div>
      <div class='flow-arrow'>→</div>
      <div class='flow-node' style='border-color:rgba(155,135,212,0.4);color:#9b87d4'>
        Production<span>D + Inv driven</span></div>
      <div class='flow-arrow'>→</div>
      <div class='flow-node' style='border-color:rgba(255,107,107,0.4);color:#ff6b6b'>
        Logistics<span>Prod + carrier</span></div>
      <div class='flow-arrow'>→</div>
      <div class='flow-node' style='border-color:rgba(91,164,229,0.4);color:#5ba4e5'>
        Chatbot<span>All outputs</span></div>
    </div>""", unsafe_allow_html=True)

def page_demand():
    df = load_data()

    st.markdown("""
    <div class='page-title-block'>
      <div class='page-title' style='color:#f5a623'>Demand Forecasting</div>
      <div class='page-subtitle'>Linear trend + seasonal decomposition · Historic Jan 2024–Dec 2025 · Forecast to Jun 2026</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='badge-row'>
      <span class='badge badge-amber'>OUTPUT feeds → Inventory</span>
      <span class='badge badge-teal'>→ Production</span>
      <span class='badge badge-coral'>→ Logistics</span>
      <span class='badge badge-sky'>→ Chatbot</span>
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

    def draw_forecast(series, color="#f5a623", title=""):
        res  = forecast_series(series, periods=horizon)
        if res.empty:
            st.info("Not enough data.")
            return res
        hist = res[res["type"] == "historical"]
        fore = res[res["type"] == "forecast"]
        fig  = go.Figure()
        ci_band(fig, fore, "rgba(245,166,35,0.06)")
        fig.add_trace(go.Scatter(
            x=hist["ds"], y=hist["y"], name="Historical",
            line=dict(color="#4a5e7a", width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=fore["ds"], y=fore["y"], name="Forecast",
            line=dict(color="#f5a623", width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=7, color="#f5a623", line=dict(color="#080e1a", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(**CD(), height=300,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y()),
            legend=dict(**legend_style()),
            title=dict(text=title, font=dict(color="#4a5e7a", size=11)))
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
            section_title("Forecast Table", "📊")
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

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_title("YoY Category Growth (2024 → 2025)", "📊")
    yr = df.groupby([df["Order_Date"].dt.year, "Category"])["Revenue_INR"].sum().unstack(fill_value=0)
    if 2024 in yr.index and 2025 in yr.index:
        g = ((yr.loc[2025] - yr.loc[2024]) / yr.loc[2024] * 100).sort_values(ascending=False)
        fig_g = go.Figure(go.Bar(
            x=g.index, y=g.values,
            marker=dict(
                color=["#56e0a0" if v >= 0 else "#ff6b6b" for v in g.values],
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=[f"{v:.1f}%" for v in g.values], textposition="outside",
            textfont=dict(color="#4a5e7a"),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% YoY<extra></extra>"
        ))
        fig_g.update_layout(**CD(), height=240,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y(), title="YoY Growth %"))
        st.plotly_chart(fig_g, use_container_width=True)

    section_title("Category Demand Forecast (fed to Production & Inventory)", "🔮")
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

    st.markdown("""
    <div class='page-title-block'>
      <div class='page-title' style='color:#56e0a0'>Inventory Optimization</div>
      <div class='page-subtitle'>EOQ · Safety Stock · Reorder Point · Stock status per SKU</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='badge-row'>
      <span class='badge badge-amber'>⬆ from: Demand Forecast (growth rate)</span>
      <span class='badge badge-teal'>feeds → Production Planning</span>
      <span class='badge badge-sky'>→ Chatbot</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("⚙ Inventory Parameters", expanded=False):
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
    kpi(c1, "Total SKUs",    len(inv),   "sky")
    kpi(c2, "🔴 Critical",   n_crit,     "coral", "reorder NOW")
    kpi(c3, "🟡 Low Stock",  n_low,      "amber", "approaching ROP")
    kpi(c4, "🟢 Adequate",   n_ok,       "mint",  "well-stocked")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    growth_pct = inv["demand_growth_pct"].iloc[0]
    st.markdown(f"""
    <div class='info-banner banner-teal'>
      <b style='color:#2ed8c3'>Demand Forecast Impact:</b>
      Future demand is projected to grow <b style='color:#56e0a0'>{growth_pct:+.1f}%</b>
      over the next 6 months (from Demand module). Annual demand forecasts adjusted accordingly.
    </div>""", unsafe_allow_html=True)

    cl, cr = st.columns([1, 2], gap="large")
    with cl:
        section_title("Stock Status Distribution", "🥧")
        color_map = {"🔴 Critical": "#ff6b6b", "🟡 Low": "#f5a623", "🟢 Adequate": "#56e0a0"}
        sc = inv["Status"].value_counts()
        pie_colors = [color_map.get(s, "#4a5e7a") for s in sc.index]
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=.62,
            marker=dict(colors=pie_colors, line=dict(color="#080e1a", width=3)),
            textinfo="label+value",
            textfont=dict(size=10, color="#f0f4ff"),
            hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<extra></extra>"
        ))
        fig.update_layout(**CD(), height=270, showlegend=False,
            annotations=[dict(text="SKUs", x=0.5, y=0.5, showarrow=False,
                             font=dict(size=11, color="#4a5e7a", family="DM Mono"))])
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        section_title("EOQ / Safety Stock / ROP by Category", "📦")
        ci2 = inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2 = go.Figure()
        bar_colors = ["#f5a623", "#2ed8c3", "#9b87d4"]
        for i, (m2, lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(
                name=lbl, x=ci2["Category"], y=ci2[m2].round(1),
                marker=dict(color=bar_colors[i], line=dict(color="rgba(0,0,0,0)")),
                hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y:,.1f}}<extra></extra>"
            ))
        fig2.update_layout(**CD(), height=270, barmode="group",
            xaxis=dict(**grid_x(), tickangle=-15),
            yaxis=dict(**grid_y()),
            legend=dict(**legend_style()))
        st.plotly_chart(fig2, use_container_width=True)

    section_title("Future Inventory Need — from Demand Forecast", "📈")
    demand_fore = compute_demand_forecast()
    fut = demand_fore[demand_fore["type"] == "forecast"]
    fig3 = go.Figure()
    ci_band(fig3, fut, "rgba(46,216,195,0.05)")
    fig3.add_trace(go.Scatter(
        x=fut["ds"], y=fut["y"], mode="lines+markers",
        line=dict(color="#2ed8c3", width=2.5),
        marker=dict(size=8, color="#2ed8c3", line=dict(color="#080e1a", width=2)),
        name="Qty Forecast",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>"
    ))
    fig3.update_layout(**CD(), height=220,
        xaxis=dict(**grid_x()),
        yaxis=dict(**grid_y(), title="Units"),
        legend=dict(**legend_style()))
    st.plotly_chart(fig3, use_container_width=True)

    section_title("SKU-Level Inventory Recommendations", "📋")
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
    st.dataframe(disp.sort_values("Status"), use_container_width=True, hide_index=True)

    st.session_state["inventory_summary"] = {
        "critical": int(n_crit), "low": int(n_low), "adequate": int(n_ok),
        "growth_pct": float(growth_pct)
    }

def page_production():
    df = load_data()

    st.markdown("""
    <div class='page-title-block'>
      <div class='page-title' style='color:#9b87d4'>Production Planning</div>
      <div class='page-subtitle'>Monthly production targets · Driven by demand forecast + inventory critical stock</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='badge-row'>
      <span class='badge badge-amber'>⬆ from: Demand Forecast</span>
      <span class='badge badge-coral'>⬆ from: Inventory (critical boost)</span>
      <span class='badge badge-lav'>feeds → Logistics</span>
      <span class='badge badge-sky'>→ Chatbot</span>
    </div>""", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    cap = p1.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1)
    buf = p2.slider("Safety Buffer %", 5, 40, 15) / 100

    plan = compute_production_plan(cap, buf)
    if plan.empty:
        st.warning("Not enough data to build production plan.")
        return

    agg = plan.groupby("Month_dt")[["Production","Demand","Inv_Boost"]].sum().reset_index()
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Production",   f"{plan['Production'].sum():,.0f}", "lav",   "units · 6 mo")
    kpi(c2, "Total Demand",       f"{plan['Demand'].sum():,.0f}",     "sky",   "forecast units")
    kpi(c3, "Avg Monthly Target", f"{agg['Production'].mean():,.0f}", "mint",  "units/month")
    peak = agg.loc[agg["Production"].idxmax(), "Month_dt"]
    kpi(c4, "Peak Month",         peak.strftime("%b %Y"),             "amber")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    inv = compute_inventory_table()
    n_crit = (inv["Status"] == "🔴 Critical").sum()
    total_boost = agg["Inv_Boost"].sum()
    st.markdown(f"""
    <div class='info-banner banner-coral'>
      <b style='color:#ff6b6b'>Inventory Signal:</b>
      {n_crit} SKUs are <b>Critical</b> (from Inventory module).
      Production plan boosted by <b style='color:#f5a623'>+{total_boost:,.0f} units</b>
      across 6 months to replenish critical stock.
    </div>""", unsafe_allow_html=True)

    section_title("Production Plan vs Demand Forecast", "🏭")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Month_dt"], y=agg["Production"],
        name="Production Target",
        marker=dict(color="#9b87d4", line=dict(color="rgba(0,0,0,0)"))
    ))
    fig.add_trace(go.Bar(
        x=agg["Month_dt"], y=agg["Inv_Boost"],
        name="Inv. Replenishment Boost",
        marker=dict(color="rgba(255,107,107,0.7)", line=dict(color="rgba(0,0,0,0)"))
    ))
    fig.add_trace(go.Scatter(
        x=agg["Month_dt"], y=agg["Demand"], name="Demand Forecast",
        mode="lines+markers",
        line=dict(color="#f5a623", width=2.5),
        marker=dict(size=8, color="#f5a623", line=dict(color="#080e1a", width=2))
    ))
    fig.update_layout(**CD(), height=300, barmode="stack",
        xaxis=dict(**grid_x()),
        yaxis=dict(**grid_y()),
        legend=dict(**legend_style()))
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2, gap="large")
    with cl:
        section_title("Production by Category (Stacked)", "📊")
        fig2 = go.Figure()
        for i, cat in enumerate(plan["Category"].unique()):
            s = plan[plan["Category"] == cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(
                x=s["Month_dt"], y=s["Production"],
                name=cat,
                marker=dict(color=COLORS[i % len(COLORS)], line=dict(color="rgba(0,0,0,0)"))
            ))
        fig2.update_layout(**CD(), height=280, barmode="stack",
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y()),
            legend=dict(**legend_style(), orientation="h", y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        section_title("Production – Demand Gap", "📉")
        agg["Gap"] = agg["Production"] - agg["Demand"]
        fig3 = go.Figure(go.Bar(
            x=agg["Month_dt"], y=agg["Gap"],
            marker=dict(
                color=["#56e0a0" if g >= 0 else "#ff6b6b" for g in agg["Gap"]],
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside",
            textfont=dict(color="#4a5e7a")
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.1)")
        fig3.update_layout(**CD(), height=280,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y(), title="Units Surplus / Deficit"))
        st.plotly_chart(fig3, use_container_width=True)

    section_title("Detailed Production Schedule", "📋")
    filt = st.selectbox("Category filter", ["All"] + list(plan["Category"].unique()))
    d = plan if filt == "All" else plan[plan["Category"] == filt]
    d2 = d[["Month","Category","Demand","Inv_Boost","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d2.columns = ["Month","Category","Demand Forecast","Inv Boost","Safety Buffer",
                  "Production Target","Demand Low","Demand High"]
    st.dataframe(d2.sort_values("Month"), use_container_width=True, hide_index=True)
    st.session_state["production_plan"] = plan

def page_logistics():
    df = load_data()

    st.markdown("""
    <div class='page-title-block'>
      <div class='page-title' style='color:#ff6b6b'>Logistics Intelligence</div>
      <div class='page-subtitle'>Carrier performance · Delay hotspots · Warehouse demand forecast · Region analysis</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='badge-row'>
      <span class='badge badge-amber'>⬆ from: Demand Forecast</span>
      <span class='badge badge-lav'>⬆ from: Production Plan volumes</span>
      <span class='badge badge-sky'>feeds → Chatbot</span>
    </div>""", unsafe_allow_html=True)

    plan = compute_production_plan()
    prod_by_cat = plan.groupby("Category")["Production"].sum() if not plan.empty else pd.Series()

    t1, t2, t3, t4 = st.tabs(["🚚 Carrier", "⚠ Delay Intel", "🏭 Warehouse Forecast", "🗺 Regions"])

    with t1:
        section_title("Carrier Performance Scorecard", "🚚")
        cs = df.groupby("Courier_Partner").agg(
            Orders   = ("Order_ID", "count"),
            Avg_Del  = ("Delivery_Days", "mean"),
            Avg_Cost = ("Shipping_Cost_INR", "mean"),
            Returns  = ("Return_Flag", "mean"),
            Revenue  = ("Revenue_INR", "sum")
        ).reset_index()
        cs["Delay_Idx"] = (cs["Avg_Del"] / cs["Avg_Del"].min() * (1 + cs["Returns"])).round(2)

        fig = go.Figure()
        for i, row in cs.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Avg_Del"]], y=[row["Avg_Cost"]],
                mode="markers+text",
                marker=dict(
                    size=max(row["Orders"]/40, 14),
                    color=COLORS[i % len(COLORS)],
                    opacity=0.9,
                    line=dict(color="#080e1a", width=2)
                ),
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
            xaxis=dict(**grid_y(), title="Avg Delivery Days"),
            yaxis=dict(**grid_y(), title="Avg Shipping Cost ₹"))
        st.plotly_chart(fig, use_container_width=True)

        d2 = cs.copy()
        d2["Avg_Del"]  = d2["Avg_Del"].round(1)
        d2["Avg_Cost"] = d2["Avg_Cost"].round(1)
        d2["Returns"]  = (d2["Returns"] * 100).round(1).astype(str) + "%"
        d2.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Revenue","Delay Index"]
        st.dataframe(d2, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        cl, cr = st.columns(2, gap="large")
        with cl:
            section_title("Historical Carrier Orders Trend", "📈")
            cm = df.groupby([df["Order_Date"].dt.to_period("M"), "Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
            fig2 = go.Figure()
            for i, c in enumerate(cm.columns):
                fig2.add_trace(go.Scatter(
                    x=cm.index.to_timestamp(), y=cm[c], name=c,
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                    hovertemplate=f"<b>{c}</b><br>%{{x|%b %Y}}<br>%{{y:,}} orders<extra></extra>"
                ))
            fig2.update_layout(**CD(), height=250,
                xaxis=dict(**grid_x()), yaxis=dict(**grid_y()),
                legend=dict(**legend_style()))
            st.plotly_chart(fig2, use_container_width=True)

        with cr:
            section_title("Carrier Order Forecast → Jun 2026", "🔮")
            fig_fc = go.Figure()
            for i, c in enumerate(cm.columns):
                s = cm[c].rename("value")
                f = forecast_series(s, 6)
                if f.empty: continue
                fut = f[f["type"] == "forecast"]
                fig_fc.add_trace(go.Scatter(
                    x=fut["ds"], y=fut["y"], name=c, mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)], width=2, dash="dot"),
                    marker=dict(size=7, line=dict(color="#080e1a", width=1.5))
                ))
            fig_fc.update_layout(**CD(), height=250,
                xaxis=dict(**grid_x()),
                yaxis=dict(**grid_y(), title="Orders"),
                legend=dict(**legend_style()))
            st.plotly_chart(fig_fc, use_container_width=True)

        if not prod_by_cat.empty:
            section_title("Carrier Recommendation (based on Production Plan)", "⭐")
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
        section_title("Delay Hotspot Analysis", "⚠️")
        thr = st.slider("Delay Threshold (days)", 3, 15, 7)
        df2 = df.copy()
        df2["Delayed"] = df2["Delivery_Days"] > thr
        rd = df2.groupby("Region").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
        rd["Rate"] = (rd["D"] / rd["T"] * 100).round(1)
        rd = rd.sort_values("Rate", ascending=False)

        cl, cr = st.columns(2, gap="large")
        with cl:
            section_title("Delay Rate by Region", "🗺️")
            fig_r = go.Figure(go.Bar(
                x=rd["Rate"], y=rd["Region"], orientation="h",
                marker=dict(
                    color=[f"rgba(255,107,107,{min(v/80+0.2,0.9):.2f})" for v in rd["Rate"]],
                    line=dict(color="rgba(0,0,0,0)")
                ),
                text=[f"{v}%" for v in rd["Rate"]], textposition="outside",
                textfont=dict(color="#4a5e7a")
            ))
            fig_r.update_layout(**CD(), height=320,
                xaxis=dict(**grid_x(), title="Delay %"),
                yaxis=dict(showgrid=False, color="#8a9dc0"))
            st.plotly_chart(fig_r, use_container_width=True)

        with cr:
            section_title("Delay Rate by Carrier", "🚚")
            cd = df2.groupby("Courier_Partner").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
            cd["Rate"] = (cd["D"] / cd["T"] * 100).round(1)
            fig_c = go.Figure(go.Bar(
                x=cd["Courier_Partner"], y=cd["Rate"],
                marker=dict(
                    color=["#ff6b6b" if v > 30 else "#f5a623" if v > 15 else "#56e0a0" for v in cd["Rate"]],
                    line=dict(color="rgba(0,0,0,0)")
                ),
                text=[f"{v}%" for v in cd["Rate"]], textposition="outside",
                textfont=dict(color="#4a5e7a")
            ))
            fig_c.update_layout(**CD(), height=320,
                xaxis=dict(**grid_x()),
                yaxis=dict(**grid_y(), title="Delay %"))
            st.plotly_chart(fig_c, use_container_width=True)

        section_title("Carrier × Region Delay Heatmap", "🌡️")
        pv = df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0) * 100
        fig_h = go.Figure(go.Heatmap(
            z=pv.values, x=list(pv.columns), y=list(pv.index),
            colorscale=[[0,"#0d1829"],[0.4,"#7c4fd0"],[0.7,"#e87adb"],[1,"#ff6b6b"]],
            text=np.round(pv.values, 1), texttemplate="%{text}%",
            textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color="#8a9dc0", size=10))
        ))
        fig_h.update_layout(**CD(), height=260,
            xaxis=dict(showgrid=False, tickangle=-30, color="#8a9dc0"),
            yaxis=dict(showgrid=False, color="#8a9dc0"))
        st.plotly_chart(fig_h, use_container_width=True)

        section_title("Delivery Delay Trend Forecast → Jun 2026", "📈")
        delay_monthly = df.groupby("YearMonth")["Delivery_Days"].mean().rename("value")
        f_delay = forecast_series(delay_monthly, 6)
        if not f_delay.empty:
            hist_d = f_delay[f_delay["type"]=="historical"]
            fut_d  = f_delay[f_delay["type"]=="forecast"]
            fig_delay = go.Figure()
            ci_band(fig_delay, fut_d, "rgba(255,107,107,0.06)")
            fig_delay.add_trace(go.Scatter(
                x=hist_d["ds"], y=hist_d["y"], name="Historical Avg Delay",
                line=dict(color="#4a5e7a", width=2)
            ))
            fig_delay.add_trace(go.Scatter(
                x=fut_d["ds"], y=fut_d["y"], name="Forecast",
                line=dict(color="#ff6b6b", width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=8, color="#ff6b6b", line=dict(color="#080e1a", width=2))
            ))
            fig_delay.update_layout(**CD(), height=250,
                xaxis=dict(**grid_x()),
                yaxis=dict(**grid_y(), title="Avg Delivery Days"),
                legend=dict(**legend_style()))
            st.plotly_chart(fig_delay, use_container_width=True)

    with t3:
        section_title("Warehouse Shipment Trend (Historical)", "🏭")
        wm = df.groupby([df["Order_Date"].dt.to_period("M"), "Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh = go.Figure()
        for i, wh in enumerate(wm.columns):
            fig_wh.add_trace(go.Bar(
                x=wm.index.to_timestamp(), y=wm[wh], name=wh,
                marker=dict(color=COLORS[i % len(COLORS)], line=dict(color="rgba(0,0,0,0)"))
            ))
        fig_wh.update_layout(**CD(), height=280, barmode="stack",
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y()),
            legend=dict(**legend_style()))
        st.plotly_chart(fig_wh, use_container_width=True)

        section_title("Warehouse Demand Forecast → Jun 2026", "🔮")
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
                    marker=dict(size=8, line=dict(color="#080e1a", width=2))
                ))
            fig_wf.update_layout(**CD(), height=260,
                xaxis=dict(**grid_x()),
                yaxis=dict(**grid_y()),
                legend=dict(**legend_style()))
            st.plotly_chart(fig_wf, use_container_width=True)
            wfd_tbl = wfd.copy()
            wfd_tbl["Month"]    = wfd_tbl["Month"].dt.strftime("%b %Y")
            wfd_tbl["Forecast"] = wfd_tbl["Forecast"].round(0).astype(int)
            wfd_tbl["CI_Hi"]    = wfd_tbl["CI_Hi"].round(0).astype(int)
            wfd_tbl.columns     = ["Month","Warehouse","Forecast Units","Upper Bound"]
            st.dataframe(wfd_tbl.sort_values(["Month","Warehouse"]),
                         use_container_width=True, hide_index=True)

        section_title("Top Products per Warehouse", "🏆")
        wsel = st.selectbox("Warehouse", sorted(df["Warehouse"].unique()))
        tp = (df[df["Warehouse"] == wsel]
              .groupby("Product_Name")["Quantity"].sum()
              .sort_values(ascending=False).head(10))
        fig_tp = go.Figure(go.Bar(
            x=tp.values, y=tp.index, orientation="h",
            marker=dict(
                color="#2ed8c3",
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=tp.values, textposition="outside",
            textfont=dict(color="#4a5e7a")
        ))
        fig_tp.update_layout(**CD(), height=310,
            xaxis=dict(**grid_x()), yaxis=dict(showgrid=False, color="#8a9dc0"))
        st.plotly_chart(fig_tp, use_container_width=True)

    with t4:
        section_title("Region Performance Overview", "🗺️")
        rs = df.groupby("Region").agg(
            Orders  = ("Order_ID", "count"),
            Revenue = ("Revenue_INR", "sum"),
            Qty     = ("Quantity", "sum"),
            Avg_Del = ("Delivery_Days", "mean"),
            Returns = ("Return_Flag", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)

        met = st.selectbox("Metric", ["Revenue","Orders","Qty","Avg_Del","Returns"])
        n = len(rs)
        bar_colors = [COLORS[i % len(COLORS)] for i in range(n)]
        fig_r = go.Figure(go.Bar(
            x=rs["Region"], y=rs[met],
            marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>"
        ))
        fig_r.update_layout(**CD(), height=300,
            xaxis=dict(**grid_x(), tickangle=-30),
            yaxis=dict(**grid_y()))
        st.plotly_chart(fig_r, use_container_width=True)

        c_l, c_r = st.columns(2, gap="large")
        with c_l:
            section_title("Best Carrier per Region", "⭐")
            bc = (df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
                  .mean().reset_index().sort_values("Delivery_Days")
                  .groupby("Region").first().reset_index())
            bc.columns = ["Region","Best Carrier","Avg Days"]
            bc["Avg Days"] = bc["Avg Days"].round(1)
            st.dataframe(bc, use_container_width=True, hide_index=True)

        with c_r:
            section_title("Region Return Rate Ranking", "🔄")
            rr = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False) * 100
            fig_ret = go.Figure(go.Bar(
                x=rr.values, y=rr.index, orientation="h",
                marker=dict(
                    color=["#ff6b6b" if v > 20 else "#f5a623" if v > 12 else "#56e0a0" for v in rr.values],
                    line=dict(color="rgba(0,0,0,0)")
                ),
                text=[f"{v:.1f}%" for v in rr.values], textposition="outside",
                textfont=dict(color="#4a5e7a")
            ))
            fig_ret.update_layout(**CD(), height=280,
                xaxis=dict(**grid_x()), yaxis=dict(showgrid=False, color="#8a9dc0"))
            st.plotly_chart(fig_ret, use_container_width=True)

        section_title("Region Revenue Forecast → Jun 2026", "📈")
        top_reg = df["Region"].value_counts().head(5).index.tolist()
        fig_rf = go.Figure()
        for i, reg in enumerate(top_reg):
            s = df[df["Region"]==reg].groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f = forecast_series(s, 6)
            if f.empty: continue
            hist_r = f[f["type"]=="historical"]
            fut_r  = f[f["type"]=="forecast"]
            fig_rf.add_trace(go.Scatter(
                x=hist_r["ds"], y=hist_r["y"], name=f"{reg}",
                line=dict(color=COLORS[i], width=1.5, dash="solid"),
                opacity=0.3, showlegend=False
            ))
            fig_rf.add_trace(go.Scatter(
                x=fut_r["ds"], y=fut_r["y"], name=reg, mode="lines+markers",
                line=dict(color=COLORS[i], width=2.5, dash="dot"),
                marker=dict(size=8, line=dict(color="#080e1a", width=2))
            ))
        fig_rf.update_layout(**CD(), height=280,
            xaxis=dict(**grid_x()),
            yaxis=dict(**grid_y()),
            legend=dict(**legend_style()))
        st.plotly_chart(fig_rf, use_container_width=True)

def build_context(df: pd.DataFrame) -> str:
    m_qty   = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    d_fore  = forecast_series(m_qty, 6)
    fut_d   = d_fore[d_fore["type"]=="forecast"]
    demand_str = "; ".join([f"{r['ds'].strftime('%b%Y')}:{r['y']:.0f}u" for _, r in fut_d.iterrows()])
    m_rev  = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
    r_fore = forecast_series(m_rev, 6)
    fut_r  = r_fore[r_fore["type"]=="forecast"]
    rev_str = "; ".join([f"{r['ds'].strftime('%b%Y')}:₹{r['y']/1e6:.1f}M" for _, r in fut_r.iterrows()])
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
    carr_str = "; ".join([f"{r}:{d['n']}ord,{d['d']:.1f}d,{d['r']*100:.1f}%ret" for r, d in cs.iterrows()])
    bc = (df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
          .mean().reset_index().sort_values("Delivery_Days")
          .groupby("Region").first().reset_index())
    bc_str = ", ".join([f"{row['Region']}→{row['Courier_Partner']}" for _, row in bc.iterrows()])
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

[MODULE 4 — LOGISTICS]
Carriers: {carr_str}
Best Carrier per Region: {bc_str}
Warehouses by Revenue: {wh_str}
High Return Regions: {ret_str}

CATEGORIES: {cat_str}
TOP REGIONS: {reg_str}
TOP PRODUCTS: {sku_str}"""

SUGGESTIONS = [
    "Which carrier for Maharashtra orders?",
    "Peak products in April 2026?",
    "Regions with critical stock risk?",
    "Production adjustments for Jun 2026?",
    "Best warehouse for Electronics?",
    "Fastest growing category?",
    "How to reduce return rate?",
    "Optimal reorder for Home & Kitchen?",
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
    df = load_data()

    st.markdown("""
    <div class='page-title-block'>
      <div class='page-title' style='color:#5ba4e5'>Decision Intelligence Chatbot</div>
      <div class='page-subtitle'>Ask anything · AI powered · Live context from all 4 upstream modules</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='badge-row'>
      <span class='badge badge-amber'>⬆ Demand context</span>
      <span class='badge badge-teal'>⬆ Inventory context</span>
      <span class='badge badge-lav'>⬆ Production context</span>
      <span class='badge badge-coral'>⬆ Logistics context</span>
    </div>""", unsafe_allow_html=True)

    ctx = build_context(df)
    with st.expander("📊 Live Module Context (fed to AI)", expanded=False):
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
        section_title("Quick Questions", "⚡")
        cols = st.columns(4)
        for i, s in enumerate(SUGGESTIONS):
            with cols[i % 4]:
                if st.button(s, key=f"q{i}", use_container_width=True):
                    st.session_state.chat_msgs.append({"role": "user", "content": s})
                    with st.spinner("OmniFlow thinking…"):
                        rep = call_claude([{"role":"user","content":s}], system_prompt)
                    st.session_state.chat_msgs.append({"role":"assistant","content":rep})
                    st.rerun()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_msgs:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='chat-spacing'>
                  <div class='chat-user-bubble'>{msg['content']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                content = (msg["content"]
                           .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                           .replace("\n","<br>"))
                st.markdown(f"""
                <div class='chat-spacing'>
                  <div class='chat-ai-bubble'>{content}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
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
        with st.spinner("OmniFlow thinking…"):
            rep = call_claude(api_msgs, system_prompt)
        st.session_state.chat_msgs.append({"role":"assistant","content":rep})
        st.rerun()

    if not st.session_state.chat_msgs:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section_title("Live Decision Alerts", "⚡")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("""<div style='font-size:0.78rem;font-weight:700;
                color:#ff6b6b;letter-spacing:0.06em;text-transform:uppercase;
                font-family:DM Mono,monospace;margin-bottom:10px'>
                🔴 Critical SKUs — Reorder Immediately</div>""", unsafe_allow_html=True)
            inv = compute_inventory_table()
            crit = inv[inv["Status"] == "🔴 Critical"][["Product_Name","Category","current_stock","ROP"]].head(5)
            for _, row in crit.iterrows():
                st.markdown(f"""
                <div class='alert-item alert-critical'>
                  <b style='color:#f0f4ff'>{row['Product_Name']}</b>
                  <span style='color:#4a5e7a'> [{row['Category']}]</span><br>
                  <span style='color:#4a5e7a;font-size:0.75rem'>
                    Stock: {row['current_stock']} · ROP: {row['ROP']}</span>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div style='font-size:0.78rem;font-weight:700;
                color:#f5a623;letter-spacing:0.06em;text-transform:uppercase;
                font-family:DM Mono,monospace;margin-bottom:10px'>
                📈 Revenue Forecast Alerts (next 3 months)</div>""", unsafe_allow_html=True)
            m = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f = forecast_series(m, 3)
            fut = f[f["type"] == "forecast"]
            last = float(m.iloc[-1])
            for _, row in fut.iterrows():
                chg  = (row["y"] - last) / last * 100
                clr  = "#56e0a0" if chg >= 0 else "#ff6b6b"
                icon = "📈" if chg >= 0 else "📉"
                status_cls = "alert-ok" if chg >= 0 else "alert-critical"
                st.markdown(f"""
                <div class='alert-item {status_cls}'>
                  {icon} <b style='color:#f0f4ff'>{row['ds'].strftime("%b %Y")}</b>
                  — <span style='color:{clr}'>₹{row['y']/1e6:.1f}M ({chg:+.1f}%)</span>
                </div>""", unsafe_allow_html=True)
                last = row["y"]

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.78rem;font-weight:700;
            color:#ff6b6b;letter-spacing:0.06em;text-transform:uppercase;
            font-family:DM Mono,monospace;margin-bottom:10px'>
            ⚠️ High Delay + Return Regions</div>""", unsafe_allow_html=True)
        df["Delayed"] = df["Delivery_Days"] > 7
        risk = df.groupby("Region").agg(
            Delay_Rate=("Delayed","mean"), Return_Rate=("Return_Flag","mean")).reset_index()
        risk["Risk_Score"] = risk["Delay_Rate"] * 0.5 + risk["Return_Rate"] * 0.5
        risk = risk.sort_values("Risk_Score", ascending=False).head(4)
        cc = st.columns(4)
        for i, (_, row) in enumerate(risk.iterrows()):
            with cc[i]:
                clr = "#ff6b6b" if row["Risk_Score"] > 0.2 else "#f5a623"
                accent = "coral" if row["Risk_Score"] > 0.2 else "amber"
                kpi(cc[i], row["Region"],
                    f"{row['Risk_Score']*100:.1f}%",
                    accent,
                    f"Del: {row['Delay_Rate']*100:.1f}% · Ret: {row['Return_Rate']*100:.1f}%")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:20px 0 28px'>
  <div style='font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.16em;
       text-transform:uppercase;color:#4a5e7a;margin-bottom:6px'>Supply Chain Platform</div>
  <div style='font-family:Outfit,sans-serif;font-size:1.8rem;font-weight:900;
       letter-spacing:-0.04em;
       background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    OmniFlow</div>
  <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5e7a;
       margin-top:2px;letter-spacing:0.05em'>D2D INTELLIGENCE</div>
</div>""", unsafe_allow_html=True)

PAGES = {
    "⬡  Overview":               page_overview,
    "📈  Demand Forecasting":     page_demand,
    "📦  Inventory Optimization": page_inventory,
    "🏭  Production Planning":    page_production,
    "🚚  Logistics Intelligence": page_logistics,
    "💬  Decision Chatbot":       page_chatbot,
}

sel = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='border-top:1px solid rgba(255,255,255,0.06);padding-top:20px'>
  <div style='font-family:DM Mono,monospace;font-size:0.65rem;
       color:#4a5e7a;line-height:2.2;letter-spacing:0.04em'>
    <span style='color:#8a9dc0'>DATA RANGE</span><br>
    Jan 2024 – Dec 2025<br>
    <span style='color:#8a9dc0'>FORECAST HORIZON</span><br>
    → Jun 2026<br>
    <span style='color:#8a9dc0'>DATASET</span><br>
    5,200 orders · 50 SKUs<br>
    🇮🇳 India D2D<br>
  </div>
  <div style='margin-top:16px;font-family:DM Mono,monospace;font-size:0.63rem;color:#4a5e7a'>
    <span style='color:#f5a623'>FLOW</span><br>
    Demand → Inventory<br>
    → Production → Logistics<br>
    → Chatbot
  </div>
</div>""", unsafe_allow_html=True)

PAGES[sel]()
