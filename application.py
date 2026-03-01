"""
OmniFlow D2D Supply Chain Intelligence
Streamlit Cloud entry point: application.py
Place OmniFlow_D2D_India_Unified_5200.csv in the same directory.
"""

import streamlit as st

st.set_page_config(
    page_title="OmniFlow D2D Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0a0e1a;--surface:#111827;--border:#1e2d45;--accent:#00e5ff;
      --accent2:#ff6b35;--accent3:#7c3aed;--text:#e2e8f0;--muted:#64748b;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;
  background-color:var(--bg)!important;color:var(--text)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;}
.stApp{background-color:var(--bg)!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important;
  border-right:1px solid var(--border)!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent3),var(--accent))!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-family:'Syne',sans-serif!important;font-weight:600!important;}
.metric-card{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:20px 24px;position:relative;overflow:hidden;margin-bottom:4px;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),var(--accent3));}
.metric-value{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:var(--accent);}
.metric-label{font-size:.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.section-title{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
  color:var(--accent);margin:18px 0 10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
.tag{display:inline-block;background:var(--accent3);color:white;padding:2px 10px;
  border-radius:20px;font-size:.72rem;font-weight:600;letter-spacing:.06em;margin:2px;}
.tag-green{background:#059669;}.tag-orange{background:#d97706;}
.tag-red{background:#dc2626;}.tag-blue{background:#2563eb;}
.chat-user{display:flex;justify-content:flex-end;margin:10px 0;}
.chat-user-bubble{background:#1e2d45;border:1px solid #2d3f55;border-radius:12px 12px 2px 12px;
  padding:10px 16px;max-width:75%;color:#e2e8f0;font-size:.88rem;line-height:1.6;}
.chat-ai{display:flex;justify-content:flex-start;margin:10px 0;gap:10px;align-items:flex-start;}
.chat-avatar{width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,#7c3aed,#00e5ff);
  display:flex;align-items:center;justify-content:center;font-size:.85rem;flex-shrink:0;}
.chat-ai-bubble{background:#111827;border:1px solid #1e2d45;
  border-radius:2px 12px 12px 12px;padding:12px 16px;max-width:78%;
  color:#e2e8f0;font-size:.88rem;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, os, json
from datetime import datetime

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "OmniFlow_D2D_India_Unified_5200.csv")
COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6","#ec4899"]

@st.cache_data(show_spinner="Loading supply chain data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df["YearMonth"] = df["Order_Date"].dt.to_period("M")
    return df

@st.cache_data(show_spinner=False)
def load_all():
    return pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])

# ── Forecast engine ────────────────────────────────────────────────────────────
def forecast_series(series: pd.Series, periods: int = 6) -> pd.DataFrame:
    s = series.dropna()
    if len(s) < 3:
        return pd.DataFrame()
    n = len(s)
    x = np.arange(n)
    m, b = np.polyfit(x, s.values, 1)
    fitted = m * x + b
    resid  = s.values - fitted
    seas   = np.zeros(12); cnt = np.zeros(12)
    months = s.index.month if hasattr(s.index, "month") else (np.arange(n) % 12)
    for i, mo in enumerate(months):
        seas[mo-1] += resid[i]; cnt[mo-1] += 1
    seas = seas / np.where(cnt > 0, cnt, 1)
    future_x  = np.arange(n, n + periods)
    fut_months= [(s.index[-1].month + i - 1) % 12 + 1 for i in range(1, periods+1)]
    fut_trend = m * future_x + b
    fut_seas  = np.array([seas[mo-1] for mo in fut_months])
    fut_vals  = np.maximum(fut_trend + fut_seas, 0)
    std = resid.std()
    hist_df = pd.DataFrame({"ds": s.index.to_timestamp(), "y": s.values,
                             "type":"historical","yhat_lower":np.nan,"yhat_upper":np.nan})
    fut_dates = pd.date_range(s.index[-1].to_timestamp(), periods=periods+1, freq="MS")[1:]
    fore_df   = pd.DataFrame({"ds": fut_dates, "y": fut_vals, "type":"forecast",
                              "yhat_lower": np.maximum(fut_vals-1.65*std,0),
                              "yhat_upper": fut_vals+1.65*std})
    return pd.concat([hist_df, fore_df], ignore_index=True)

def chart_defaults():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",   
        font=dict(color="#94a3b8"), margin=dict(l=0, r=0, t=10, b=0)
    )

def kpi_card(col, label, value, color="#00e5ff", sub=""):
    col.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{value}</div>
      <div style='color:#475569;font-size:.75rem'>{sub}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    df = load_data()
    raw = load_all()

    st.markdown("""
    <div style='padding:28px 0 12px'>
      <div style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;
           background:linear-gradient(90deg,#00e5ff,#7c3aed);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1'>
        OmniFlow D2D Intelligence</div>
      <div style='color:#64748b;font-size:.95rem;margin-top:6px'>
        End-to-end supply chain analytics · India Direct-to-Door · Jan 2024 – Jun 2026 forecast
      </div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
         padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden'>
      <div style='position:absolute;top:0;left:0;right:0;height:3px;
           background:linear-gradient(90deg,#00e5ff,#7c3aed,#ff6b35)'></div>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
           color:#00e5ff;margin-bottom:10px'>📋 About This Platform</div>
      <p style='color:#cbd5e1;line-height:1.8;margin:0'>
        <b style='color:#e2e8f0'>OmniFlow</b> is an AI-driven supply chain intelligence platform
        built on <b>5,200 real-world D2D orders</b> across India (Jan 2024 – Dec 2025). Six
        interconnected modules feed each other — from raw demand signals to last-mile optimisation.
      </p>
      <div style='margin-top:14px'>
        <span class='tag tag-blue'>📈 Demand → Jun 2026</span>
        <span class='tag tag-green'>📦 Inventory EOQ</span>
        <span class='tag' style='background:#7c3aed'>🏭 Production Plan</span>
        <span class='tag tag-orange'>🚚 Logistics Intel</span>
        <span class='tag tag-red'>🤖 AI Chatbot</span>
      </div></div>""", unsafe_allow_html=True)

    # KPIs
    delivered = df[df["Order_Status"]=="Delivered"]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_card(c1,"Total Revenue",   f"₹{df['Revenue_INR'].sum()/1e7:.1f}Cr","#00e5ff","all orders")
    kpi_card(c2,"Orders",          f"{len(df):,}","#f59e0b","non-cancelled")
    kpi_card(c3,"Units Sold",      f"{df['Quantity'].sum():,}","#10b981","quantities")
    kpi_card(c4,"Return Rate",     f"{df['Return_Flag'].mean()*100:.1f}%","#dc2626","delivered")
    kpi_card(c5,"Avg Delivery",    f"{df['Delivery_Days'].mean():.1f}d","#a78bfa","days")
    kpi_card(c6,"Categories",      f"{df['Category'].nunique()}","#3b82f6","product types")

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend + Category pie
    c_l, c_r = st.columns([2,1])
    with c_l:
        st.markdown("<div class='section-title'>Monthly Revenue Trend</div>",unsafe_allow_html=True)
        m = df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue_INR"].sum().reset_index()
        m["Order_Date"] = m["Order_Date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["Order_Date"],y=m["Revenue_INR"],fill="tozeroy",
            line=dict(color="#00e5ff",width=2.5),fillcolor="rgba(0,229,255,0.07)"))
        fig.update_layout(**chart_defaults(),height=280,
            xaxis=dict(showgrid=False,color="#475569"),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45",color="#475569",tickformat=",.0f"),
            showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c_r:
        st.markdown("<div class='section-title'>Revenue by Category</div>",unsafe_allow_html=True)
        cat = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(labels=cat.index,values=cat.values,hole=.55,
            marker=dict(colors=COLORS),textinfo="label+percent",
            textfont=dict(size=9,color="#e2e8f0")))
        fig2.update_layout(**chart_defaults(),height=280,showlegend=False)
        st.plotly_chart(fig2,use_container_width=True)

    # 3 bottom charts
    c3a,c3b,c3c = st.columns(3)
    with c3a:
        st.markdown("<div class='section-title'>Orders by Channel</div>",unsafe_allow_html=True)
        ch = df["Sales_Channel"].value_counts().head(6)
        fig3 = go.Figure(go.Bar(x=ch.values,y=ch.index,orientation="h",
            marker_color=COLORS[:len(ch)],text=ch.values,textposition="outside",
            textfont=dict(color="#94a3b8",size=10)))
        fig3.update_layout(**chart_defaults(),height=240,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
        st.plotly_chart(fig3,use_container_width=True)
    with c3b:
        st.markdown("<div class='section-title'>Top Regions Revenue</div>",unsafe_allow_html=True)
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(8)
        fig4 = go.Figure(go.Bar(x=reg.index,y=reg.values,
            marker=dict(color=reg.values,colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]])))
        fig4.update_layout(**chart_defaults(),height=240,
            xaxis=dict(showgrid=False,tickangle=-30),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45"))
        st.plotly_chart(fig4,use_container_width=True)
    with c3c:
        st.markdown("<div class='section-title'>Order Status Split</div>",unsafe_allow_html=True)
        sc = raw["Order_Status"].value_counts()
        fig5 = go.Figure(go.Bar(x=sc.index,y=sc.values,
            marker_color=["#10b981","#dc2626","#f59e0b","#3b82f6"][:len(sc)]))
        fig5.update_layout(**chart_defaults(),height=240,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"))
        st.plotly_chart(fig5,use_container_width=True)

    # Module flow
    st.markdown("<div class='section-title'>Module Dependency Flow</div>",unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1e2d45;border-radius:12px;
         padding:20px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px'>
      <div style='background:#0f2030;border:1px solid #00e5ff;border-radius:8px;
           padding:10px 16px;color:#00e5ff;font-weight:700;font-family:Syne,sans-serif;font-size:.82rem;text-align:center'>
        📈 Demand<br><span style='font-size:.68rem;color:#64748b'>→ Jun 2026</span></div>
      <div style='color:#7c3aed;font-size:1.4rem;font-weight:800'>→</div>
      <div style='background:#0f2030;border:1px solid #10b981;border-radius:8px;
           padding:10px 16px;color:#10b981;font-weight:700;font-family:Syne,sans-serif;font-size:.82rem;text-align:center'>
        📦 Inventory<br><span style='font-size:.68rem;color:#64748b'>EOQ + ROP</span></div>
      <div style='color:#7c3aed;font-size:1.4rem;font-weight:800'>→</div>
      <div style='background:#0f2030;border:1px solid #7c3aed;border-radius:8px;
           padding:10px 16px;color:#a78bfa;font-weight:700;font-family:Syne,sans-serif;font-size:.82rem;text-align:center'>
        🏭 Production<br><span style='font-size:.68rem;color:#64748b'>Demand+Inv</span></div>
      <div style='color:#7c3aed;font-size:1.4rem;font-weight:800'>→</div>
      <div style='background:#0f2030;border:1px solid #ff6b35;border-radius:8px;
           padding:10px 16px;color:#ff6b35;font-weight:700;font-family:Syne,sans-serif;font-size:.82rem;text-align:center'>
        🚚 Logistics<br><span style='font-size:.68rem;color:#64748b'>Carrier+WH</span></div>
      <div style='color:#7c3aed;font-size:1.4rem;font-weight:800'>→</div>
      <div style='background:#0f2030;border:1px solid #f59e0b;border-radius:8px;
           padding:10px 16px;color:#f59e0b;font-weight:700;font-family:Syne,sans-serif;font-size:.82rem;text-align:center'>
        🤖 Chatbot<br><span style='font-size:.68rem;color:#64748b'>AI Decisions</span></div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
def page_demand():
    df = load_data()
    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#00e5ff'>
        📈 Demand Forecasting</div>
      <div style='color:#64748b;font-size:.88rem'>
        Linear trend + seasonal decomposition · Forecast to June 2026</div></div>""",
        unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,2,1])
    metric_opt = c1.selectbox("Metric",["Revenue (₹)","Quantity (Units)","Orders (#)"])
    level_opt  = c2.selectbox("Breakdown",["Overall","Category","Region","Sales Channel"])
    horizon    = c3.slider("Months ahead",3,18,6)

    col_map  = {"Revenue (₹)":"Revenue_INR","Quantity (Units)":"Quantity","Orders (#)":"Order_ID"}
    col      = col_map[metric_opt]
    agg_fn   = "count" if col=="Order_ID" else "sum"

    def get_monthly(sub):
        if agg_fn=="count":
            return sub.groupby("YearMonth")["Order_ID"].count().rename("value")
        return sub.groupby("YearMonth")[col].sum().rename("value")

    def draw_forecast(series, title="", color="#00e5ff"):
        res  = forecast_series(series, periods=horizon)
        if res.empty:
            st.info("Not enough data."); return res
        hist = res[res["type"]=="historical"]
        fore = res[res["type"]=="forecast"]
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"],y=hist["y"],name="Historical",
            line=dict(color=color,width=2.5)))
        fig.add_trace(go.Scatter(x=fore["ds"],y=fore["y"],name="Forecast",
            line=dict(color="#ff6b35",width=2.5,dash="dot")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fore["ds"],fore["ds"][::-1]]),
            y=pd.concat([fore["yhat_upper"],fore["yhat_lower"][::-1]]),
            fill="toself",fillcolor="rgba(255,107,53,0.10)",
            line=dict(color="transparent"),name="95% CI"))
        fig.update_layout(**chart_defaults(),height=300,
            xaxis=dict(showgrid=False,color="#475569"),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text=title,font=dict(color="#94a3b8",size=13)))
        st.plotly_chart(fig,use_container_width=True)
        return res

    if level_opt=="Overall":
        res = draw_forecast(get_monthly(df))
        if not res.empty:
            fore = res[res["type"]=="forecast"][["ds","y","yhat_lower","yhat_upper"]].copy()
            fore.columns=["Month","Forecast","Lower","Upper"]
            fore["Month"]=fore["Month"].dt.strftime("%b %Y")
            for c2 in ["Forecast","Lower","Upper"]: fore[c2]=fore[c2].round(0).astype(int)
            st.markdown("<div class='section-title'>Forecast Table</div>",unsafe_allow_html=True)
            st.dataframe(fore,use_container_width=True,hide_index=True)
    else:
        grp = {"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}[level_opt]
        top = df[grp].value_counts().head(5).index.tolist()
        tabs = st.tabs(top)
        for i,(tab,val) in enumerate(zip(tabs,top)):
            with tab:
                draw_forecast(get_monthly(df[df[grp]==val]),color=COLORS[i])

    # YoY growth
    st.markdown("<div class='section-title'>YoY Category Growth (2024 → 2025)</div>",unsafe_allow_html=True)
    yr = df.groupby([df["Order_Date"].dt.year,"Category"])["Revenue_INR"].sum().unstack(fill_value=0)
    if 2024 in yr.index and 2025 in yr.index:
        g = ((yr.loc[2025]-yr.loc[2024])/yr.loc[2024]*100).sort_values(ascending=False)
        fig_g = go.Figure(go.Bar(x=g.index,y=g.values,
            marker_color=["#10b981" if v>=0 else "#dc2626" for v in g.values],
            text=[f"{v:.1f}%" for v in g.values],textposition="outside",
            textfont=dict(color="#94a3b8")))
        fig_g.update_layout(**chart_defaults(),height=240,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45",title="YoY Growth %"))
        st.plotly_chart(fig_g,use_container_width=True)

    # Save demand forecast to session
    m = get_monthly(df)
    st.session_state["demand_monthly"] = m
    res = forecast_series(m, periods=6)
    st.session_state["demand_forecast"] = res


# ──────────────────────────────────────────────────────────────────────────────
def page_inventory():
    df = load_data()
    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#10b981'>
        📦 Inventory Optimization</div>
      <div style='color:#64748b;font-size:.88rem'>
        EOQ · Safety Stock · Reorder Point · driven by demand forecast</div></div>""",
        unsafe_allow_html=True)

    with st.expander("⚙️ Parameters",expanded=False):
        p1,p2,p3 = st.columns(3)
        order_cost = p1.number_input("Order Cost ₹",100,5000,500,50)
        hold_pct   = p2.slider("Holding Cost %",5,40,20)/100
        lead_time  = p3.slider("Lead Time (days)",1,30,7)
        svc        = st.selectbox("Service Level",["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"])
        z          = {"90% (z=1.28)":1.28,"95% (z=1.65)":1.65,"99% (z=2.33)":2.33}[svc]

    # Build SKU stats
    grp = df.groupby(["SKU_ID","Product_Name","Category","Sell_Price"])
    rows=[]
    for (sku,prod,cat,price),g in grp:
        mq = g.groupby("YearMonth")["Quantity"].sum()
        rows.append(dict(SKU_ID=sku,Product_Name=prod,Category=cat,
                         unit_cost=price,monthly_avg=mq.mean(),monthly_std=mq.std() or 0,
                         total_qty=g["Quantity"].sum()))
    inv = pd.DataFrame(rows)

    def eoq(d,oc,h,uc): return int(np.sqrt(2*d*oc/(uc*h))) if d>0 and uc*h>0 else 0
    def ss(std,lt,z):   return int(z*std*np.sqrt(lt/30))
    def rop(avg,lt,ss): return int(avg/30*lt+ss)

    inv["annual"]  = inv["monthly_avg"]*12
    inv["EOQ"]     = inv.apply(lambda r: eoq(r["annual"],order_cost,hold_pct,r["unit_cost"]),axis=1)
    inv["SS"]      = inv.apply(lambda r: ss(r["monthly_std"],lead_time,z),axis=1)
    inv["ROP"]     = inv.apply(lambda r: rop(r["monthly_avg"],lead_time,r["SS"]),axis=1)

    # Demand growth factor
    m_qty = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    fore  = forecast_series(m_qty,6)
    fut   = fore[fore["type"]=="forecast"]
    growth= (fut["y"].mean()/m_qty.iloc[-3:].mean())-1 if len(m_qty)>3 else 0
    inv["Forecast_Annual"] = (inv["annual"]*(1+growth)).astype(int)
    inv["Status"] = inv.apply(lambda r:
        "🔴 Critical" if r["total_qty"]<r["SS"] else
        ("🟡 Low" if r["total_qty"]<r["ROP"] else "🟢 Adequate"), axis=1)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    kpi_card(c1,"Total SKUs",  len(inv),        "#00e5ff")
    kpi_card(c2,"Critical",    (inv["Status"]=="🔴 Critical").sum(),"#dc2626")
    kpi_card(c3,"Low Stock",   (inv["Status"]=="🟡 Low").sum(),     "#f59e0b")
    kpi_card(c4,"Adequate",    (inv["Status"]=="🟢 Adequate").sum(),"#10b981")
    st.markdown("<br>",unsafe_allow_html=True)

    cl,cr = st.columns([1,2])
    with cl:
        st.markdown("<div class='section-title'>Status Distribution</div>",unsafe_allow_html=True)
        sc = inv["Status"].value_counts()
        fig = go.Figure(go.Pie(labels=sc.index,values=sc.values,hole=.55,
            marker_colors=["#dc2626","#f59e0b","#10b981"],textinfo="label+percent"))
        fig.update_layout(**chart_defaults(),height=250,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with cr:
        st.markdown("<div class='section-title'>EOQ / Safety Stock by Category</div>",unsafe_allow_html=True)
        ci = inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2 = go.Figure()
        for i,(m2,lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(name=lbl,x=ci["Category"],y=ci[m2],marker_color=COLORS[i]))
        fig2.update_layout(**chart_defaults(),height=250,barmode="group",
            xaxis=dict(showgrid=False,tickangle=-20),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2,use_container_width=True)

    # Demand forecast chart
    st.markdown("<div class='section-title'>Future Inventory Need (Demand Forecast)</div>",unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fut["ds"],y=fut["y"],mode="lines+markers",
        line=dict(color="#00e5ff",width=2.5),marker=dict(size=8),name="Forecast Demand"))
    fig3.add_trace(go.Scatter(
        x=pd.concat([fut["ds"],fut["ds"][::-1]]),
        y=pd.concat([fut["yhat_upper"],fut["yhat_lower"][::-1]]),
        fill="toself",fillcolor="rgba(0,229,255,0.07)",
        line=dict(color="transparent"),name="CI"))
    fig3.update_layout(**chart_defaults(),height=230,
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3,use_container_width=True)

    st.markdown("<div class='section-title'>SKU Recommendations</div>",unsafe_allow_html=True)
    cats = st.multiselect("Filter Category",df["Category"].unique().tolist(),
                          default=df["Category"].unique().tolist())
    disp = inv[inv["Category"].isin(cats)][
        ["SKU_ID","Product_Name","Category","monthly_avg","EOQ","SS","ROP","Forecast_Annual","Status"]
    ].copy()
    disp.columns=["SKU","Product","Category","Avg/Month","EOQ","Safety Stock","Reorder Point","Forecast Annual","Status"]
    for c in ["Avg/Month","EOQ","Safety Stock","Reorder Point","Forecast Annual"]:
        disp[c]=disp[c].round(0).astype(int)
    st.dataframe(disp.sort_values("Status"),use_container_width=True,hide_index=True)

    st.session_state["inventory_data"] = inv


# ──────────────────────────────────────────────────────────────────────────────
def page_production():
    df = load_data()
    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#a78bfa'>
        🏭 Production Planning</div>
      <div style='color:#64748b;font-size:.88rem'>
        Monthly production targets · demand forecast + inventory position</div></div>""",
        unsafe_allow_html=True)

    p1,p2 = st.columns(2)
    cap   = p1.slider("Capacity Multiplier",0.5,2.0,1.0,0.1)
    buf   = p2.slider("Safety Buffer %",5,40,15)/100

    cat_monthly = df.groupby(["YearMonth","Category"])["Quantity"].sum().unstack(fill_value=0)
    plans=[]
    for cat in cat_monthly.columns:
        series = cat_monthly[cat].rename("value")
        fore   = forecast_series(series,6)
        if fore.empty: continue
        fut    = fore[fore["type"]=="forecast"]
        cur    = float(series.iloc[-3:].mean()*1.5)
        for _,row in fut.iterrows():
            net  = max(row["y"]-cur/6,0)
            prod = net*(1+buf)*cap
            plans.append(dict(Month=row["ds"].strftime("%b %Y"),Month_dt=row["ds"],
                Category=cat,Demand=round(row["y"],0),Production=round(prod,0),
                Buffer=round(prod-net*cap,0),
                CI_Lo=round(row["yhat_lower"],0),CI_Hi=round(row["yhat_upper"],0)))
    plan = pd.DataFrame(plans)
    if plan.empty:
        st.warning("Not enough data to build plan."); return

    agg = plan.groupby("Month_dt")[["Production","Demand"]].sum().reset_index()

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    kpi_card(c1,"Total Production",f"{plan['Production'].sum():,.0f}","#a78bfa","units (6 mo)")
    kpi_card(c2,"Total Demand",    f"{plan['Demand'].sum():,.0f}",    "#00e5ff","units (6 mo)")
    kpi_card(c3,"Avg Monthly",     f"{agg['Production'].mean():,.0f}","#10b981","units")
    peak = agg.loc[agg["Production"].idxmax(),"Month_dt"]
    kpi_card(c4,"Peak Month",      peak.strftime("%b %Y"),            "#ff6b35")
    st.markdown("<br>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Production Plan vs Demand Forecast</div>",unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Month_dt"],y=agg["Production"],name="Production",marker_color="#7c3aed"))
    fig.add_trace(go.Scatter(x=agg["Month_dt"],y=agg["Demand"],name="Demand",
        mode="lines+markers",line=dict(color="#00e5ff",width=2.5),marker=dict(size=8)))
    fig.update_layout(**chart_defaults(),height=290,barmode="group",
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("<div class='section-title'>Production by Category (Stacked)</div>",unsafe_allow_html=True)
    fig2 = go.Figure()
    for i,cat in enumerate(plan["Category"].unique()):
        s = plan[plan["Category"]==cat].sort_values("Month_dt")
        fig2.add_trace(go.Bar(x=s["Month_dt"],y=s["Production"],name=cat,marker_color=COLORS[i%len(COLORS)]))
    fig2.update_layout(**chart_defaults(),height=300,barmode="stack",
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.25))
    st.plotly_chart(fig2,use_container_width=True)

    # Gap chart
    st.markdown("<div class='section-title'>Production–Demand Gap</div>",unsafe_allow_html=True)
    agg["Gap"]=agg["Production"]-agg["Demand"]
    fig3=go.Figure(go.Bar(x=agg["Month_dt"],y=agg["Gap"],
        marker_color=["#10b981" if g>=0 else "#dc2626" for g in agg["Gap"]],
        text=[f"{g:+.0f}" for g in agg["Gap"]],textposition="outside",
        textfont=dict(color="#94a3b8")))
    fig3.add_hline(y=0,line_dash="dash",line_color="#475569")
    fig3.update_layout(**chart_defaults(),height=210,
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45",title="Units"))
    st.plotly_chart(fig3,use_container_width=True)

    st.markdown("<div class='section-title'>Detailed Schedule</div>",unsafe_allow_html=True)
    filt = st.selectbox("Category",["All"]+list(plan["Category"].unique()))
    d = plan if filt=="All" else plan[plan["Category"]==filt]
    d2 = d[["Month","Category","Demand","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d2.columns=["Month","Category","Demand","Buffer","Production Target","Demand Low","Demand High"]
    st.dataframe(d2.sort_values("Month"),use_container_width=True,hide_index=True)

    st.session_state["production_plan"]=plan


# ──────────────────────────────────────────────────────────────────────────────
def page_logistics():
    df = load_data()
    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#ff6b35'>
        🚚 Logistics Intelligence</div>
      <div style='color:#64748b;font-size:.88rem'>
        Carrier performance · delay hotspots · warehouse demand · region analysis</div></div>""",
        unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["🚛 Carrier","⚠️ Delay Intel","🏪 Warehouse Forecast","📍 Regions"])

    # ── TAB 1: CARRIER ────────────────────────────────────────────────────────
    with t1:
        st.markdown("<div class='section-title'>Carrier Performance Scorecard</div>",unsafe_allow_html=True)
        cs = df.groupby("Courier_Partner").agg(
            Orders=("Order_ID","count"),Avg_Del=("Delivery_Days","mean"),
            Avg_Cost=("Shipping_Cost_INR","mean"),Returns=("Return_Flag","mean"),
            Revenue=("Revenue_INR","sum")).reset_index()
        cs["Delay_Idx"]=(cs["Avg_Del"]/cs["Avg_Del"].min()*(1+cs["Returns"])).round(2)

        fig=go.Figure()
        for i,row in cs.iterrows():
            fig.add_trace(go.Scatter(x=[row["Avg_Del"]],y=[row["Avg_Cost"]],
                mode="markers+text",
                marker=dict(size=max(row["Orders"]/40,12),color=COLORS[i%len(COLORS)],opacity=.8),
                text=[row["Courier_Partner"]],textposition="top center",
                name=row["Courier_Partner"],
                hovertemplate=f"<b>{row['Courier_Partner']}</b><br>Orders:{row['Orders']}<br>"
                              f"Avg Del:{row['Avg_Del']:.1f}d<br>Cost:₹{row['Avg_Cost']:.0f}<br>"
                              f"Returns:{row['Returns']*100:.1f}%<extra></extra>"))
        fig.update_layout(**chart_defaults(),height=320,showlegend=False,
            xaxis=dict(title="Avg Delivery Days",showgrid=True,gridcolor="#1e2d45"),
            yaxis=dict(title="Avg Shipping Cost ₹",showgrid=True,gridcolor="#1e2d45"))
        st.plotly_chart(fig,use_container_width=True)

        d2=cs.copy()
        d2["Avg_Del"]=d2["Avg_Del"].round(1); d2["Avg_Cost"]=d2["Avg_Cost"].round(1)
        d2["Returns"]=(d2["Returns"]*100).round(1).astype(str)+"%"
        d2.columns=["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Revenue","Delay Index"]
        st.dataframe(d2,use_container_width=True,hide_index=True)

        st.markdown("<div class='section-title'>Carrier Orders Trend</div>",unsafe_allow_html=True)
        cm=df.groupby([df["Order_Date"].dt.to_period("M"),"Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        fig2=go.Figure()
        for i,c in enumerate(cm.columns):
            fig2.add_trace(go.Scatter(x=cm.index.to_timestamp(),y=cm[c],name=c,
                line=dict(color=COLORS[i%len(COLORS)],width=2)))
        fig2.update_layout(**chart_defaults(),height=250,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2,use_container_width=True)

    # ── TAB 2: DELAY INTELLIGENCE ─────────────────────────────────────────────
    with t2:
        st.markdown("<div class='section-title'>Delay Hotspot Analysis</div>",unsafe_allow_html=True)
        thr=st.slider("Delay Threshold (days)",3,15,7)
        df2=df.copy(); df2["Delayed"]=df2["Delivery_Days"]>thr

        rd=df2.groupby("Region").agg(T=("Order_ID","count"),D=("Delayed","sum")).reset_index()
        rd["Rate"]=(rd["D"]/rd["T"]*100).round(1)
        rd=rd.sort_values("Rate",ascending=False)

        cl,cr=st.columns(2)
        with cl:
            st.markdown("**By Region**")
            fig_r=go.Figure(go.Bar(x=rd["Rate"],y=rd["Region"],orientation="h",
                marker_color=[f"rgba(220,38,38,{v/100+.2})" for v in rd["Rate"]],
                text=[f"{v}%" for v in rd["Rate"]],textposition="outside",
                textfont=dict(color="#94a3b8")))
            fig_r.update_layout(**chart_defaults(),height=320,
                xaxis=dict(showgrid=False,title="Delay %"),yaxis=dict(showgrid=False))
            st.plotly_chart(fig_r,use_container_width=True)
        with cr:
            st.markdown("**By Carrier**")
            cd=df2.groupby("Courier_Partner").agg(T=("Order_ID","count"),D=("Delayed","sum")).reset_index()
            cd["Rate"]=(cd["D"]/cd["T"]*100).round(1)
            fig_c=go.Figure(go.Bar(x=cd["Courier_Partner"],y=cd["Rate"],
                marker_color=["#dc2626" if v>30 else "#f59e0b" if v>15 else "#10b981" for v in cd["Rate"]],
                text=[f"{v}%" for v in cd["Rate"]],textposition="outside",
                textfont=dict(color="#94a3b8")))
            fig_c.update_layout(**chart_defaults(),height=320,
                xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45",title="Delay %"))
            st.plotly_chart(fig_c,use_container_width=True)

        st.markdown("<div class='section-title'>Carrier × Region Delay Heatmap</div>",unsafe_allow_html=True)
        pv=df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0)*100
        fig_h=go.Figure(go.Heatmap(z=pv.values,x=pv.columns,y=pv.index,
            colorscale=[[0,"#0f172a"],[.5,"#7c3aed"],[1,"#dc2626"]],
            text=np.round(pv.values,1),texttemplate="%{text}%",
            colorbar=dict(tickfont=dict(color="#94a3b8"))))
        fig_h.update_layout(**chart_defaults(),height=260,
            xaxis=dict(showgrid=False,tickangle=-30),yaxis=dict(showgrid=False))
        st.plotly_chart(fig_h,use_container_width=True)

    # ── TAB 3: WAREHOUSE ──────────────────────────────────────────────────────
    with t3:
        st.markdown("<div class='section-title'>Warehouse Shipment Trend</div>",unsafe_allow_html=True)
        wm=df.groupby([df["Order_Date"].dt.to_period("M"),"Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh=go.Figure()
        for i,wh in enumerate(wm.columns):
            fig_wh.add_trace(go.Bar(x=wm.index.to_timestamp(),y=wm[wh],name=wh,marker_color=COLORS[i%len(COLORS)]))
        fig_wh.update_layout(**chart_defaults(),height=280,barmode="stack",
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_wh,use_container_width=True)

        st.markdown("<div class='section-title'>Warehouse Demand Forecast → Jun 2026</div>",unsafe_allow_html=True)
        wf_rows=[]
        for i,wh in enumerate(wm.columns):
            s=wm[wh].rename("value")
            if s.dropna().__len__()<4: continue
            f=forecast_series(s,6)
            fut=f[f["type"]=="forecast"]
            for _,r in fut.iterrows():
                wf_rows.append(dict(Month=r["ds"],WH=wh,Forecast=r["y"]))
        if wf_rows:
            wfd=pd.DataFrame(wf_rows)
            fig_wf=go.Figure()
            for i,wh in enumerate(wfd["WH"].unique()):
                s=wfd[wfd["WH"]==wh]
                fig_wf.add_trace(go.Scatter(x=s["Month"],y=s["Forecast"],name=wh,
                    mode="lines+markers",line=dict(color=COLORS[i%len(COLORS)],width=2,dash="dot"),
                    marker=dict(size=7)))
            fig_wf.update_layout(**chart_defaults(),height=260,
                xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_wf,use_container_width=True)

        st.markdown("<div class='section-title'>Top Products per Warehouse</div>",unsafe_allow_html=True)
        wsel=st.selectbox("Warehouse",df["Warehouse"].unique())
        tp=(df[df["Warehouse"]==wsel].groupby("Product_Name")["Quantity"].sum()
            .sort_values(ascending=False).head(10))
        fig_tp=go.Figure(go.Bar(x=tp.values,y=tp.index,orientation="h",
            marker_color="#00e5ff",text=tp.values,textposition="outside",
            textfont=dict(color="#94a3b8")))
        fig_tp.update_layout(**chart_defaults(),height=310,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
        st.plotly_chart(fig_tp,use_container_width=True)

    # ── TAB 4: REGIONS ────────────────────────────────────────────────────────
    with t4:
        st.markdown("<div class='section-title'>Region Performance</div>",unsafe_allow_html=True)
        rs=df.groupby("Region").agg(Orders=("Order_ID","count"),Revenue=("Revenue_INR","sum"),
            Qty=("Quantity","sum"),Avg_Del=("Delivery_Days","mean"),
            Returns=("Return_Flag","mean")).reset_index().sort_values("Revenue",ascending=False)

        met=st.selectbox("Metric",["Revenue","Orders","Qty","Avg_Del","Returns"])
        fig_r=go.Figure(go.Bar(x=rs["Region"],y=rs[met],
            marker=dict(color=rs[met],colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]]),
            text=rs[met].round(1),textposition="outside",textfont=dict(color="#94a3b8")))
        fig_r.update_layout(**chart_defaults(),height=300,
            xaxis=dict(showgrid=False,tickangle=-30),
            yaxis=dict(showgrid=True,gridcolor="#1e2d45"))
        st.plotly_chart(fig_r,use_container_width=True)

        st.markdown("<div class='section-title'>Best Carrier per Region</div>",unsafe_allow_html=True)
        bc=(df.groupby(["Region","Courier_Partner"])["Delivery_Days"].mean()
            .reset_index().sort_values("Delivery_Days").groupby("Region").first().reset_index())
        bc.columns=["Region","Best Carrier","Avg Delivery Days"]
        bc["Avg Delivery Days"]=bc["Avg Delivery Days"].round(1)
        st.dataframe(bc,use_container_width=True,hide_index=True)

        # Region demand forecast
        st.markdown("<div class='section-title'>Region Revenue Forecast → Jun 2026</div>",unsafe_allow_html=True)
        top_reg=df["Region"].value_counts().head(4).index.tolist()
        fig_rf=go.Figure()
        for i,reg in enumerate(top_reg):
            s=df[df["Region"]==reg].groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f=forecast_series(s,6)
            fut=f[f["type"]=="forecast"]
            if fut.empty: continue
            fig_rf.add_trace(go.Scatter(x=fut["ds"],y=fut["y"],name=reg,
                mode="lines+markers",line=dict(color=COLORS[i],width=2,dash="dot"),
                marker=dict(size=7)))
        fig_rf.update_layout(**chart_defaults(),height=260,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rf,use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
def build_context(df):
    total_rev    = df["Revenue_INR"].sum()
    cat_rev      = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
    cat_str      = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in cat_rev.items()])
    cs           = df.groupby("Courier_Partner").agg(n=("Order_ID","count"),
                       d=("Delivery_Days","mean"),r=("Return_Flag","mean"))
    carr_str     = "; ".join([f"{r}:{d['n']}orders,{d['d']:.1f}d,{d['r']*100:.1f}%ret"
                              for r,d in cs.iterrows()])
    top_reg      = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    reg_str      = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in top_reg.items()])
    m_qty        = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    fore         = forecast_series(m_qty,6)
    fut          = fore[fore["type"]=="forecast"]
    fore_str     = "; ".join([f"{r['ds'].strftime('%b%Y')}:{r['y']:.0f}units" for _,r in fut.iterrows()])
    wh_rev       = df.groupby("Warehouse")["Revenue_INR"].sum().sort_values(ascending=False)
    wh_str       = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in wh_rev.items()])
    top_sku      = df.groupby("Product_Name")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    sku_str      = ", ".join(top_sku.index.tolist())
    ret_hot      = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False).head(3)
    ret_str      = ", ".join([f"{k}:{v*100:.1f}%" for k,v in ret_hot.items()])
    return f"""=== OmniFlow D2D India Supply Chain Context ===
Dataset: 5,200 orders | Jan 2024–Dec 2025 | India D2D
Revenue: ₹{total_rev/1e7:.2f}Cr | Orders: {len(df):,} | Return Rate: {df['Return_Flag'].mean()*100:.1f}% | Avg Del: {df['Delivery_Days'].mean():.1f}d
CATEGORIES: {cat_str}
TOP REGIONS: {reg_str}
CARRIERS: {carr_str}
WAREHOUSES: {wh_str}
TOP PRODUCTS: {sku_str}
HIGH RETURN REGIONS: {ret_str}
DEMAND FORECAST (6mo): {fore_str}"""

def call_claude_api(messages, system):
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":1000,
                  "system":system,"messages":messages})
        d = r.json()
        if "content" in d:
            return "".join(b.get("text","") for b in d["content"] if b.get("type")=="text")
        return f"⚠️ API Error: {d.get('error',{}).get('message','Unknown')}"
    except Exception as e:
        return f"⚠️ Connection error: {e}"

SUGGESTIONS = [
    "Which carrier should I use for Maharashtra?",
    "What products will peak in April 2026?",
    "Which regions have critical stock risk?",
    "How to adjust production for June 2026?",
    "Best warehouse for Electronics shipments?",
    "Which category grows fastest?",
    "How to reduce the return rate?",
    "Optimal reorder strategy for Home & Kitchen?",
]

def page_chatbot():
    df = load_data()
    st.markdown("""<div style='padding:12px 0 6px'>
      <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#f59e0b'>
        🤖 Decision Intelligence Chatbot</div>
      <div style='color:#64748b;font-size:.88rem'>
        Ask anything about your supply chain · Claude AI · live context from all modules</div></div>""",
        unsafe_allow_html=True)

    ctx = build_context(df)
    with st.expander("📊 Live Data Context fed to AI",expanded=False):
        st.code(ctx,language="text")

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    system_prompt = f"""You are OmniFlow, an expert supply chain decision intelligence assistant
for an India D2D e-commerce operation (Amazon, Flipkart, Meesho, B2B channels).
You have deep expertise in demand forecasting, inventory management (EOQ, safety stock),
production planning, and logistics optimization across Indian regions and carriers.
Give specific, actionable, data-backed answers. Use ₹ for rupees. Use bullet points for lists.
Reference exact numbers from the context. Be concise but comprehensive.
Always consider interdependencies: demand→inventory→production→logistics.
Flag risks, opportunities, and seasonal patterns when relevant.

LIVE SUPPLY CHAIN CONTEXT:
{ctx}"""

    # Suggestions (shown only when no chat yet)
    if not st.session_state.chat_msgs:
        st.markdown("<div class='section-title'>💡 Quick Questions</div>",unsafe_allow_html=True)
        cols = st.columns(4)
        for i,s in enumerate(SUGGESTIONS):
            with cols[i%4]:
                if st.button(s,key=f"q{i}",use_container_width=True):
                    st.session_state.chat_msgs.append({"role":"user","content":s})
                    with st.spinner("Thinking…"):
                        rep = call_claude_api([{"role":"user","content":s}],system_prompt)
                    st.session_state.chat_msgs.append({"role":"assistant","content":rep})
                    st.rerun()

    # Display chat history
    for msg in st.session_state.chat_msgs:
        if msg["role"]=="user":
            st.markdown(f"""<div class='chat-user'>
              <div class='chat-user-bubble'>{msg['content']}</div></div>""",
              unsafe_allow_html=True)
        else:
            content = msg["content"].replace("\n","<br>").replace("**","<b>",1).replace("**","</b>",1)
            st.markdown(f"""<div class='chat-ai'>
              <div class='chat-avatar'>🤖</div>
              <div class='chat-ai-bubble'>{content}</div></div>""",
              unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c_inp,c_btn,c_clr = st.columns([5,1,1])
    with c_inp:
        user_in = st.text_input("Ask a supply chain question…",key="user_input",
            placeholder="e.g. What production adjustments needed for Q2 2026?",
            label_visibility="collapsed")
    with c_btn:
        send = st.button("Send 🚀",use_container_width=True)
    with c_clr:
        if st.button("Clear 🗑️",use_container_width=True):
            st.session_state.chat_msgs=[]
            st.rerun()

    if send and user_in.strip():
        st.session_state.chat_msgs.append({"role":"user","content":user_in.strip()})
        api_msgs = st.session_state.chat_msgs[-12:]
        with st.spinner("OmniFlow AI thinking…"):
            rep = call_claude_api(api_msgs, system_prompt)
        st.session_state.chat_msgs.append({"role":"assistant","content":rep})
        st.rerun()

    # Quick alerts (shown when no chat)
    if not st.session_state.chat_msgs:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚡ Live Decision Alerts</div>",unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🔴 High Return Rate Regions**")
            rr = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False).head(4)
            for reg,rate in rr.items():
                clr="#dc2626" if rate>0.2 else "#f59e0b"
                st.markdown(f"""<div style='background:#111827;border-left:3px solid {clr};
                    padding:8px 12px;margin:5px 0;border-radius:0 6px 6px 0;font-size:.83rem'>
                    ⚠️ <b>{reg}</b> — {rate*100:.1f}% returns</div>""",unsafe_allow_html=True)
        with c2:
            st.markdown("**📈 Demand Forecast Alerts**")
            m = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            f = forecast_series(m,3)
            fut = f[f["type"]=="forecast"]
            last = float(m.iloc[-1])
            for _,row in fut.iterrows():
                chg=(row["y"]-last)/last*100
                clr="#10b981" if chg>=0 else "#dc2626"
                icon="📈" if chg>=0 else "📉"
                st.markdown(f"""<div style='background:#111827;border-left:3px solid {clr};
                    padding:8px 12px;margin:5px 0;border-radius:0 6px 6px 0;font-size:.83rem'>
                    {icon} <b>{row['ds'].strftime("%b %Y")}</b> — ₹{row['y']/1e6:.1f}M ({chg:+.1f}%)</div>""",
                    unsafe_allow_html=True)
                last=row["y"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + ROUTING
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style='padding:12px 0 20px'>
  <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#00e5ff'>OmniFlow</div>
  <div style='font-size:.7rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase'>
    D2D Supply Intelligence</div>
</div>""", unsafe_allow_html=True)

PAGES = {
    "🏠  Overview":               page_overview,
    "📈  Demand Forecasting":     page_demand,
    "📦  Inventory Optimization": page_inventory,
    "🏭  Production Planning":    page_production,
    "🚚  Logistics Intelligence": page_logistics,
    "🤖  Decision Chatbot":       page_chatbot,
}

sel = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:.72rem;color:#64748b;line-height:1.8'>
📅 Data: Jan 2024 – Dec 2025<br>
🔮 Forecast: to Jun 2026<br>
📊 5,200 orders<br>
🇮🇳 India D2D Supply Chain<br>
🤖 Powered by Claude AI
</div>""", unsafe_allow_html=True)

PAGES[sel]()
