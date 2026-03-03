# ============================================================
# OmniFlow D2D Intelligence Platform  |  Final-Year Project
# Supply-chain-accurate · ML forecasting · API-powered chatbot
# ============================================================

import streamlit as st
st.set_page_config(
    page_title="OmniFlow D2D Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
:root{
  --midnight:#080e1a;--deep:#0d1829;--surface:#111e30;--panel:#162236;
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.12);
  --amber:#f5a623;--coral:#ff6b6b;--teal:#2ed8c3;--sky:#5ba4e5;
  --lav:#9b87d4;--mint:#56e0a0;
  --text1:#f0f4ff;--text2:#8a9dc0;--text3:#4a5e7a;
  --shadow:0 8px 32px rgba(0,0,0,0.4),0 1px 0 rgba(255,255,255,0.05);
}
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important;background:var(--midnight)!important;color:var(--text1)!important;}
.stApp{
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%,rgba(91,164,229,0.08) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%,rgba(46,216,195,0.06) 0%,transparent 60%),
    var(--midnight)!important;
}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--deep) 0%,var(--midnight) 100%)!important;
  border-right:1px solid var(--border2)!important;
  box-shadow:4px 0 24px rgba(0,0,0,0.5)!important;
}
section[data-testid="stSidebar"]::before{
  content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--amber),var(--coral),var(--teal));
}
.metric-card{
  background:linear-gradient(135deg,var(--panel) 0%,var(--surface) 100%);
  border:1px solid var(--border);border-radius:16px;padding:20px 22px;
  box-shadow:var(--shadow);position:relative;overflow:hidden;
  transition:transform 0.3s ease,box-shadow 0.3s ease;cursor:default;
}
.metric-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;}
.metric-card.amber::before{background:linear-gradient(90deg,var(--amber),#ff8c42);}
.metric-card.teal::before{background:linear-gradient(90deg,var(--teal),#22b8a5);}
.metric-card.coral::before{background:linear-gradient(90deg,var(--coral),#ff4f4f);}
.metric-card.sky::before{background:linear-gradient(90deg,var(--sky),#3d87d4);}
.metric-card.lav::before{background:linear-gradient(90deg,var(--lav),#7b6bbf);}
.metric-card.mint::before{background:linear-gradient(90deg,var(--mint),#3ec47a);}
.metric-card:hover{transform:translateY(-4px);box-shadow:0 20px 48px rgba(0,0,0,0.5);}
.metric-label{font-size:0.65rem;text-transform:uppercase;color:var(--text3);
  letter-spacing:0.12em;font-weight:600;margin-bottom:8px;font-family:'DM Mono',monospace!important;}
.metric-value{font-size:2rem;font-weight:800;line-height:1;letter-spacing:-0.03em;}
.metric-sub{font-size:0.7rem;color:var(--text3);margin-top:5px;font-family:'DM Mono',monospace!important;}
.section-header{display:flex;align-items:center;gap:10px;margin:24px 0 14px;}
.section-header-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border2),transparent);}
.section-title{font-weight:700;font-size:0.78rem;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--text2);font-family:'DM Mono',monospace!important;}
.page-title{font-family:'Outfit',sans-serif!important;font-size:2.4rem;font-weight:900;
  letter-spacing:-0.04em;line-height:1.1;margin-bottom:6px;padding:20px 0 6px;}
.page-subtitle{color:var(--text3);font-size:0.82rem;font-family:'DM Mono',monospace!important;margin-bottom:16px;}
.badge{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:999px;
  font-size:0.68rem;font-weight:600;letter-spacing:0.05em;border:1px solid;
  font-family:'DM Mono',monospace!important;margin:3px;}
.badge-amber{background:rgba(245,166,35,0.1);color:var(--amber);border-color:rgba(245,166,35,0.3);}
.badge-teal{background:rgba(46,216,195,0.1);color:var(--teal);border-color:rgba(46,216,195,0.3);}
.badge-coral{background:rgba(255,107,107,0.1);color:var(--coral);border-color:rgba(255,107,107,0.3);}
.badge-sky{background:rgba(91,164,229,0.1);color:var(--sky);border-color:rgba(91,164,229,0.3);}
.badge-lav{background:rgba(155,135,212,0.1);color:var(--lav);border-color:rgba(155,135,212,0.3);}
.badge-mint{background:rgba(86,224,160,0.1);color:var(--mint);border-color:rgba(86,224,160,0.3);}
.info-banner{border-radius:12px;padding:13px 17px;margin:10px 0 18px;
  font-size:0.82rem;border:1px solid;position:relative;overflow:hidden;}
.info-banner::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;}
.banner-teal{background:rgba(46,216,195,0.06);border-color:rgba(46,216,195,0.2);color:var(--text2);}
.banner-teal::before{background:var(--teal);}
.banner-amber{background:rgba(245,166,35,0.06);border-color:rgba(245,166,35,0.2);color:var(--text2);}
.banner-amber::before{background:var(--amber);}
.banner-coral{background:rgba(255,107,107,0.06);border-color:rgba(255,107,107,0.2);color:var(--text2);}
.banner-coral::before{background:var(--coral);}
.about-card{
  background:linear-gradient(135deg,rgba(22,34,54,0.9),rgba(17,30,48,0.9));
  border:1px solid var(--border2);border-radius:20px;padding:26px 30px;
  margin-bottom:24px;position:relative;overflow:hidden;
}
.about-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--amber),var(--coral),var(--teal),var(--sky));}
.alert-item{border-radius:10px;padding:9px 13px;margin:5px 0;font-size:0.78rem;
  border-left:3px solid;background:var(--panel);font-family:'DM Mono',monospace!important;
  transition:transform 0.2s ease;}
.alert-item:hover{transform:translateX(3px);background:var(--surface);}
.alert-critical{border-color:var(--coral);}
.alert-warn{border-color:var(--amber);}
.alert-ok{border-color:var(--mint);}
.chat-user-bubble{
  background:linear-gradient(135deg,rgba(245,166,35,0.12),rgba(255,107,107,0.08));
  border:1px solid rgba(245,166,35,0.2);border-radius:14px 14px 4px 14px;
  padding:11px 15px;font-size:0.87rem;margin-left:18%;}
.chat-ai-bubble{
  background:linear-gradient(135deg,var(--panel),var(--surface));
  border:1px solid var(--border2);border-radius:14px 14px 14px 4px;
  padding:13px 17px;font-size:0.87rem;color:var(--text2);line-height:1.7;margin-right:8%;}
.stTabs [data-baseweb="tab-list"]{
  background:var(--panel)!important;border-radius:14px!important;
  padding:5px 7px!important;gap:5px!important;border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border-radius:10px!important;color:var(--text3)!important;
  font-family:'DM Mono',monospace!important;font-size:0.74rem!important;font-weight:600!important;
  padding:9px 18px!important;transition:all 0.2s!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(245,166,35,0.2),rgba(255,107,107,0.1))!important;
  color:var(--amber)!important;box-shadow:0 2px 12px rgba(245,166,35,0.12)!important;
}
.stButton>button{
  background:linear-gradient(135deg,rgba(245,166,35,0.15),rgba(255,107,107,0.1))!important;
  color:var(--amber)!important;border:1px solid rgba(245,166,35,0.3)!important;
  border-radius:10px!important;font-weight:600!important;font-size:0.78rem!important;
  font-family:'DM Mono',monospace!important;transition:all 0.25s!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(245,166,35,0.28),rgba(255,107,107,0.18))!important;
  transform:translateY(-2px)!important;box-shadow:0 6px 18px rgba(245,166,35,0.2)!important;
}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--midnight);}
::-webkit-scrollbar-thumb{background:var(--panel);border-radius:99px;}
footer{visibility:hidden;}#MainMenu{visibility:hidden;}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.metric-card{animation:fadeUp 0.45s ease both;}
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import requests as _requests

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_ecommerce_orders.csv")

COLORS   = ["#f5a623","#56e0a0","#ff6b6b","#5ba4e5","#e87adb","#2ed8c3"]
COLORS_S = ["#e8963f","#4ecf94","#e85c5c","#4d90d4","#cc68c4","#28c4b0"]

def CD():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a9dc0", family="DM Mono, monospace", size=11),
        margin=dict(l=8,r=8,t=34,b=14)
    )

def gY(): return dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickcolor="#4a5e7a")
def gX(): return dict(showgrid=False, zeroline=False, tickcolor="#4a5e7a")
def leg(): return dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1, font=dict(color="#8a9dc0",size=10))

def kpi(col, label, value, cls="amber", sub=""):
    col.markdown(f"""<div class='metric-card {cls}'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(label, emoji=""):
    st.markdown(f"""<div class='section-header'>
      <div class='section-title'>{emoji} {label}</div>
      <div class='section-header-line'></div>
    </div>""", unsafe_allow_html=True)

def banner(html, cls="teal"):
    st.markdown(f"<div class='info-banner banner-{cls}'>{html}</div>", unsafe_allow_html=True)

def sp(n=1):
    st.markdown(f"<div style='height:{n*14}px'></div>", unsafe_allow_html=True)

# ─── DATA LOADING ───────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df["Region"]      = df["Region"].replace("Pune", "Maharashtra")
    df["YearMonth"]   = df["Order_Date"].dt.to_period("M")
    df["Year"]        = df["Order_Date"].dt.year
    df["Month_Num"]   = df["Order_Date"].dt.month
    df["Net_Revenue"] = np.where(df["Return_Flag"] == 1, 0.0, df["Revenue_INR"])
    df.loc[df["Order_Status"] == "Cancelled", "Delivery_Days"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def get_ops(df):
    return df[df["Order_Status"].isin(["Delivered", "Shipped"])].copy()

@st.cache_data(show_spinner=False)
def get_delivered(df):
    return df[df["Order_Status"] == "Delivered"].copy()

# ─── ML FORECASTING ─────────────────────────────────────────
def build_features(n_hist, n_future, ds_hist, regime_start_idx):
    all_t = np.arange(n_hist + n_future)

    # Safely extract month values regardless of index type
    try:
        ts = ds_hist.to_timestamp()
    except AttributeError:
        ts = pd.DatetimeIndex(ds_hist)

    hist_months = ts.month.values
    last_month  = int(hist_months[-1])
    fut_months  = np.array([(last_month + i - 1) % 12 + 1 for i in range(1, n_future + 1)])
    mn = np.concatenate([hist_months, fut_months])

    regime = (all_t >= regime_start_idx).astype(float)
    X = np.column_stack([
        all_t, all_t ** 2,
        np.sin(2 * np.pi * mn / 12), np.cos(2 * np.pi * mn / 12),
        np.sin(4 * np.pi * mn / 12), np.cos(4 * np.pi * mn / 12),
        regime, all_t * regime,
    ])
    return X

def ml_forecast(series_values, ds_index, n_future=6, alpha=0.5):
    n = len(series_values)
    if n < 6:
        return None

    regime_idx = 15
    X_all  = build_features(n, n_future, ds_index, regime_idx)
    X_hist = X_all[:n]
    X_fut  = X_all[n:]

    h = 4
    Xtr, Xte = X_hist[:-h], X_hist[-h:]
    ytr, yte = series_values[:-h], series_values[-h:]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    mdl_eval = Ridge(alpha=alpha)
    mdl_eval.fit(Xtr_s, ytr)
    ypred_eval = np.maximum(mdl_eval.predict(Xte_s), 0)

    rmse  = np.sqrt(mean_squared_error(yte, ypred_eval))
    nrmse = rmse / np.mean(yte) if np.mean(yte) > 0 else 0
    mae   = mean_absolute_error(yte, ypred_eval)

    sc2 = StandardScaler()
    X_hist_s = sc2.fit_transform(X_hist)
    mdl_full = Ridge(alpha=alpha)
    mdl_full.fit(X_hist_s, series_values)

    fitted    = np.maximum(mdl_full.predict(X_hist_s), 0)
    residuals = series_values - fitted
    resid_std = residuals.std()
    ss_res    = np.sum((series_values - fitted) ** 2)
    ss_tot    = np.sum((series_values - np.mean(series_values)) ** 2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    X_fut_s  = sc2.transform(X_fut)
    forecast = np.maximum(mdl_full.predict(X_fut_s), 0)

    try:
        last_dt = ds_index.to_timestamp().iloc[-1]
    except AttributeError:
        last_dt = pd.DatetimeIndex(ds_index)[-1]
    fut_dates = pd.date_range(last_dt + pd.offsets.MonthBegin(1), periods=n_future, freq="MS")

    return {
        "hist_ds":     ds_index.to_timestamp(),
        "hist_y":      series_values,
        "fitted":      fitted,
        "fut_ds":      fut_dates,
        "forecast":    forecast,
        "ci_lo":       np.maximum(forecast - 1.645 * resid_std, 0),
        "ci_hi":       forecast + 1.645 * resid_std,
        "rmse":        rmse,
        "nrmse":       nrmse,
        "mae":         mae,
        "r2":          r2,
        "resid_std":   resid_std,
        "eval_actual": yte,
        "eval_pred":   ypred_eval,
        "eval_ds":     ds_index.to_timestamp().iloc[-h:],
    }

# ─── INVENTORY ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_inventory(order_cost=500, hold_pct=0.20, lead_time=7, z=1.65):
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    sku_monthly = (ops.groupby(["SKU_ID","YM"])["Quantity"]
                   .sum().reset_index().sort_values(["SKU_ID","YM"]))
    sku_stats = (ops.groupby(["SKU_ID","Product_Name","Category"])
                 .agg(avg_price=("Sell_Price","mean"), total_qty=("Quantity","sum"))
                 .reset_index())

    rows = []
    for _, sk in sku_stats.iterrows():
        sku     = sk["SKU_ID"]
        skd     = sku_monthly[sku_monthly["SKU_ID"]==sku].sort_values("YM")
        demands = skd["Quantity"].values
        if len(demands) < 2:
            continue

        avg_d   = demands.mean()
        std_d   = demands.std() if len(demands) > 1 else avg_d * 0.2
        daily_d = avg_d / 30.0
        ann_d   = avg_d * 12
        uc      = max(sk["avg_price"], 1.0)

        eoq       = int(np.sqrt(2 * ann_d * order_cost / (uc * hold_pct))) if ann_d > 0 else 10
        eoq       = max(eoq, 1)
        daily_std = std_d / np.sqrt(30)
        ss        = int(z * daily_std * np.sqrt(lead_time))
        ss        = max(ss, 0)
        rop       = int(daily_d * lead_time + ss)
        rop       = max(rop, 1)

        stock   = eoq * 2
        pending = 0
        for demand in demands:
            stock  -= demand
            stock   = max(stock + pending, 0)
            pending = 0
            if stock < rop:
                n_orders = max(1, int(np.ceil((rop - stock + ss) / eoq)))
                pending  = n_orders * eoq

        current_stock = max(stock, 0)

        if current_stock < ss:
            status = "🔴 Critical"
        elif current_stock < rop:
            status = "🟡 Low"
        else:
            status = "🟢 Adequate"

        rows.append({
            "SKU_ID": sku, "Product_Name": sk["Product_Name"],
            "Category": sk["Category"],
            "Monthly_Avg": round(avg_d,1), "Monthly_Std": round(std_d,1),
            "Daily_Std": round(daily_std,2),
            "EOQ": eoq, "SS": ss, "ROP": rop,
            "Current_Stock": current_stock, "Status": status,
            "Unit_Price": round(uc,0), "Annual_Demand": round(ann_d,0),
            "Forecast_6M": int(avg_d * 6 * 1.05),
        })

    return pd.DataFrame(rows)

# ─── PRODUCTION ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_production(cap_mult=1.0, buffer_pct=0.15):
    df  = load_data()
    ops = get_ops(df)
    inv = compute_inventory()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    cat_monthly = ops.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
    ds_index    = cat_monthly.index

    rows = []
    for cat in cat_monthly.columns:
        vals = cat_monthly[cat].values.astype(float)
        res  = ml_forecast(vals, ds_index, n_future=6)
        if res is None:
            continue

        crit_boost = float(inv[(inv["Category"]==cat)&(inv["Status"]=="🔴 Critical")]["Monthly_Avg"].sum() * 0.5)
        low_boost  = float(inv[(inv["Category"]==cat)&(inv["Status"]=="🟡 Low")]["Monthly_Avg"].sum() * 0.25)

        for i, (dt, fc) in enumerate(zip(res["fut_ds"], res["forecast"])):
            net_prod = max(fc + crit_boost + low_boost, 0) * cap_mult
            prod     = net_prod * (1 + buffer_pct)
            rows.append({
                "Month_dt": dt, "Month": dt.strftime("%b %Y"),
                "Category": cat,
                "Demand_Forecast": round(fc,0),
                "Crit_Boost": round(crit_boost,0),
                "Low_Boost":  round(low_boost,0),
                "Buffer":     round(prod - net_prod,0),
                "Production": round(prod,0),
                "CI_Lo": round(res["ci_lo"][i],0),
                "CI_Hi": round(res["ci_hi"][i],0),
            })

    return pd.DataFrame(rows)

# ─── LOGISTICS ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_logistics():
    df     = load_data()
    del_df = get_delivered(df)

    carr = del_df.groupby("Courier_Partner").agg(
        Orders=("Order_ID","count"), Avg_Days=("Delivery_Days","mean"),
        Avg_Cost=("Shipping_Cost_INR","mean"), Total_Cost=("Shipping_Cost_INR","sum"),
        Return_Rate=("Return_Flag","mean"),
    ).reset_index()
    carr["Delay_Index"] = (carr["Avg_Days"] / carr["Avg_Days"].min()).round(2)
    carr["Cost_Score"]  = (carr["Avg_Cost"] / carr["Avg_Cost"].min()).round(2)
    carr["Perf_Score"]  = ((1/carr["Delay_Index"])*(1-carr["Return_Rate"])*(1/carr["Cost_Score"])).round(3)

    best = (del_df.groupby(["Region","Courier_Partner"])
            .agg(Avg_Days=("Delivery_Days","mean"), Orders=("Order_ID","count"),
                 Avg_Cost=("Shipping_Cost_INR","mean"))
            .reset_index().sort_values("Avg_Days").groupby("Region").first().reset_index())

    current = (del_df.groupby(["Region","Courier_Partner"])
               .agg(Orders=("Order_ID","count"), Total_Cost=("Shipping_Cost_INR","sum"))
               .reset_index())

    cheapest = (del_df.groupby(["Region","Courier_Partner"])
                .agg(avg_cost=("Shipping_Cost_INR","mean"), orders=("Order_ID","count"))
                .reset_index().sort_values("avg_cost").groupby("Region").first().reset_index()
                .rename(columns={"Courier_Partner":"Optimal_Carrier","avg_cost":"Min_Avg_Cost"}))

    region_costs = (del_df.groupby("Region")
                    .agg(Current_Avg_Cost=("Shipping_Cost_INR","mean"),
                         Orders=("Order_ID","count"), Total_Spend=("Shipping_Cost_INR","sum"))
                    .reset_index())

    opt = region_costs.merge(cheapest[["Region","Optimal_Carrier","Min_Avg_Cost"]], on="Region")
    opt["Potential_Saving"] = ((opt["Current_Avg_Cost"]-opt["Min_Avg_Cost"])*opt["Orders"]).round(0)
    opt["Saving_Pct"]       = ((opt["Current_Avg_Cost"]-opt["Min_Avg_Cost"])/opt["Current_Avg_Cost"]*100).round(1)

    return carr, best, opt, current

# ─── CHATBOT CONTEXT ────────────────────────────────────────
def build_context():
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    m_orders = ops.groupby("YM")["Order_ID"].count().rename("v")
    m_qty    = ops.groupby("YM")["Quantity"].sum().rename("v")
    m_rev    = ops.groupby("YM")["Net_Revenue"].sum().rename("v")

    r_ord = ml_forecast(m_orders.values.astype(float), m_orders.index, 6)
    r_rev = ml_forecast(m_rev.values.astype(float),    m_rev.index,    6)
    r_qty = ml_forecast(m_qty.values.astype(float),    m_qty.index,    6)

    def fc_str(r, fmt):
        if r is None: return "N/A"
        return "; ".join([f"{d.strftime('%b%Y')}:{fmt(v)}" for d,v in zip(r["fut_ds"], r["forecast"])])

    inv  = compute_inventory()
    plan = compute_production()
    carr, best_carr, opt, _ = compute_logistics()

    n_crit     = (inv["Status"]=="🔴 Critical").sum()
    n_low      = (inv["Status"]=="🟡 Low").sum()
    crit_prods = ", ".join(inv[inv["Status"]=="🔴 Critical"]["Product_Name"].head(5).tolist())
    prod_sum   = plan.groupby("Category")["Production"].sum().to_dict() if not plan.empty else {}
    prod_str   = ", ".join([f"{k}:{v:.0f}u" for k,v in prod_sum.items()])
    peak_mo    = plan.groupby("Month_dt")["Production"].sum().idxmax().strftime("%b %Y") if not plan.empty else "N/A"
    carr_str   = "; ".join([f"{r['Courier_Partner']}: {r['Orders']}ord, {r['Avg_Days']:.1f}d, ₹{r['Avg_Cost']:.0f}/ship" for _,r in carr.iterrows()])
    best_str   = ", ".join([f"{r['Region']}→{r['Courier_Partner']}" for _,r in best_carr.iterrows()])
    saving_total = opt["Potential_Saving"].sum()
    saving_str   = "; ".join([f"{r['Region']}: save ₹{r['Potential_Saving']:,.0f} with {r['Optimal_Carrier']}" for _,r in opt.iterrows() if r['Potential_Saving']>0])
    cat_rev  = ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
    cat_str  = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in cat_rev.items()])
    top_reg  = ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False).head(5)
    reg_str  = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in top_reg.items()])
    top_sku  = ops.groupby("Product_Name")["Net_Revenue"].sum().sort_values(ascending=False).head(8)
    sku_str  = ", ".join(top_sku.index.tolist())
    metric_str = f"Orders RMSE:{r_ord['rmse']:.1f}, NRMSE:{r_ord['nrmse']*100:.1f}%, R²:{r_ord['r2']:.2f}" if r_ord else ""

    return f"""=== OmniFlow D2D India — Live Supply Chain Intelligence ===
DATASET: 5,200 orders | Jan 2024–Dec 2025 | India D2D (Amazon, Flipkart, B2B)
SUMMARY: Net Revenue ₹{ops['Net_Revenue'].sum()/1e7:.2f}Cr | Active Orders {len(ops):,} |
         Return Rate {df[df['Order_Status']=='Returned'].shape[0]/len(ops)*100:.1f}% |
         Avg Delivery {ops['Delivery_Days'].mean():.1f}d (Delivered only)

[MODULE 1 — DEMAND FORECAST (Ridge Regression + Structural Break)]
{metric_str}
Order Forecast: {fc_str(r_ord, lambda v: f"{v:.0f}")}
Qty Forecast:   {fc_str(r_qty, lambda v: f"{v:.0f}u")}
Revenue Forecast: {fc_str(r_rev, lambda v: f"₹{v/1e6:.1f}M")}

[MODULE 2 — INVENTORY (EOQ / Safety Stock / ROP)]
Critical SKUs: {n_crit} | Low: {n_low} | Adequate: {inv['Status'].eq('🟢 Adequate').sum()}
Reorder IMMEDIATELY: {crit_prods}

[MODULE 3 — PRODUCTION PLAN (6-month)]
By Category: {prod_str}
Peak Month: {peak_mo}

[MODULE 4 — LOGISTICS]
Carriers: {carr_str}
Best Carrier by Region: {best_str}
Cost Saving: ₹{saving_total:,.0f} total | {saving_str}

CATEGORIES: {cat_str}
TOP REGIONS: {reg_str}
TOP PRODUCTS: {sku_str}"""

# ─── LLM CALL ───────────────────────────────────────────────
def call_llm(messages, system, api_key):
    hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 1800,
        "temperature": 0.4,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    try:
        r = _requests.post("https://api.groq.com/openai/v1/chat/completions",
                           headers=hdrs, json=body, timeout=50)
        if r.status_code == 401:
            return "❌ Invalid Groq API key. Get a free key at console.groq.com"
        if r.status_code == 429:
            return "⚠️ Rate limit reached. Wait a few seconds and retry."
        if r.status_code != 200:
            return f"⚠️ Groq error ({r.status_code}): {r.text[:300]}"
        return r.json()["choices"][0]["message"]["content"]
    except _requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please retry."
    except Exception as exc:
        return f"⚠️ Call failed: {exc}"

def build_system_prompt(ctx):
    return f"""You are OmniFlow, an expert AI supply chain analyst embedded inside a
Streamlit dashboard for an India D2D e-commerce business.

=== YOUR EXPERTISE ===
• Demand forecasting — Ridge Regression with structural break detection, Fourier seasonality
• Inventory management — Wilson EOQ, Safety Stock (z·σ_daily·√LT), ROP, (s,Q) policy
• Production planning — 6-month capacity scheduling, demand + inventory-driven targets
• Logistics optimisation — carrier selection, cost reduction, delay analysis
• Indian e-commerce channels — Amazon.in, Shiprocket, INCREFF B2B

=== RESPONSE RULES ===
1. Lead with one precise, data-backed insight sentence
2. Use bullet points (▸) — cite exact numbers (₹, %, units, days)
3. Keep answers concise — 4–8 bullets is the target
4. No generic closings or padding sentences
5. If information is not in the context, say so honestly

=== LIVE SUPPLY CHAIN CONTEXT ===
{ctx}"""

# ═══════════════════════════════════════════════════════════
# PAGE — CHATBOT
# ═══════════════════════════════════════════════════════════
def page_chatbot():
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#5ba4e5'>Decision Intelligence Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>LLaMA-3.3-70B via Groq · Full supply chain context · Multi-turn</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>⬆ Demand</span>
      <span class='badge badge-teal'>⬆ Inventory</span>
      <span class='badge badge-lav'>⬆ Production</span>
      <span class='badge badge-coral'>⬆ Logistics</span>
      <span class='badge badge-sky'>Groq API</span>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""<div style='margin-top:18px;border-top:1px solid rgba(255,255,255,0.06);
            padding-top:16px;font-family:DM Mono,monospace;font-size:0.65rem;
            color:#4a5e7a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>
            🤖 Groq AI Config</div>""", unsafe_allow_html=True)
        api_key = st.text_input(
            "Groq API Key", type="password", key="llm_api_key",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
            help="Free key at console.groq.com — no credit card needed"
        )
        if api_key and len(api_key.strip()) > 10:
            if api_key.strip().startswith("gsk_"):
                st.markdown("<div style='font-size:0.62rem;color:#56e0a0;font-family:DM Mono,monospace;margin-top:4px'>✅ Key looks valid</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.62rem;color:#ff6b6b;font-family:DM Mono,monospace;margin-top:4px'>⚠️ Key should start with gsk_</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.6rem;color:#4a5e7a;font-family:DM Mono,monospace;margin-top:4px'>console.groq.com — free tier</div>", unsafe_allow_html=True)

    ctx    = build_context()
    system = build_system_prompt(ctx)

    with st.expander("📊 Live Context fed to AI", expanded=False):
        st.code(ctx, language="text")

    key_ok = bool(api_key and len(api_key.strip()) > 10)
    if not key_ok:
        st.markdown("""<div class='info-banner banner-amber'>
          <b style='color:#f5a623'>⚠️ Groq API Key Required:</b>
          Go to the left sidebar → 🤖 Groq AI Config and paste your key (starts with <code>gsk_</code>).
          Get one free at <b style='color:#56e0a0'>console.groq.com</b> — no credit card needed.
        </div>""", unsafe_allow_html=True)

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    CHAT_SUGGESTIONS = [
        "Which product will have the highest demand next month?",
        "What is the reorder point for Home & Kitchen SKUs?",
        "Which region needs the most logistics support and why?",
        "How should I adjust production targets for next quarter?",
        "Which courier should I use for Maharashtra to minimise cost?",
        "Explain the safety stock formula and values for Electronics",
        "How can I reduce shipping costs across all regions?",
        "Which SKUs are at critical inventory risk right now?",
        "Compare all carriers across speed, cost and returns",
        "Give me a complete 6-month production plan summary",
        "What is the EOQ for Fashion & Apparel and why?",
        "Which warehouse handles the most volume and is it optimal?",
        "Calculate total logistics cost saving if I switch carriers",
        "What is the NRMSE of the demand forecast and is it acceptable?",
        "If I increase lead time to 14 days, how does ROP change?",
        "What are the top 5 revenue-generating products?",
    ]

    if not st.session_state.chat_msgs:
        sec("Quick Queries — click any to get started", "⚡")
        cols = st.columns(4)
        for i, s in enumerate(CHAT_SUGGESTIONS):
            with cols[i % 4]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    if not key_ok:
                        st.warning("⚠️ Enter your Groq API key in the sidebar first.")
                    else:
                        st.session_state.chat_msgs.append({"role":"user","content":s})
                        with st.spinner("OmniFlow analysing…"):
                            reply = call_llm([{"role":"user","content":s}], system, api_key.strip())
                        st.session_state.chat_msgs.append({"role":"assistant","content":reply})
                        st.rerun()

    import re as _re
    for msg in st.session_state.chat_msgs:
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(f"<div style='margin:10px 0'><div class='chat-user-bubble'>{content}</div></div>", unsafe_allow_html=True)
        else:
            safe = content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            safe = _re.sub(r'\*\*(.+?)\*\*', r'<span style="color:#f0f4ff;font-weight:700">\1</span>', safe)
            safe = _re.sub(r'\*(.+?)\*',     r'<span style="color:#8a9dc0;font-style:italic">\1</span>', safe)
            html_parts = []
            for line in safe.split("\n"):
                line = line.strip()
                if not line:
                    html_parts.append("<div style='height:5px'></div>")
                elif _re.match(r"^[▸\-•] ", line):
                    body = line[2:].strip()
                    html_parts.append(f"<div style='display:flex;gap:8px;margin:5px 0'><span style='color:#f5a623;flex-shrink:0;margin-top:2px'>▸</span><span style='color:#c8d4e8;line-height:1.65'>{body}</span></div>")
                elif line.startswith("**") and line.endswith("**"):
                    html_parts.append(f"<div style='color:#f0f4ff;font-weight:700;margin:8px 0 3px'>{line[2:-2]}</div>")
                else:
                    html_parts.append(f"<div style='color:#8a9dc0;line-height:1.65;margin:3px 0'>{line}</div>")
            st.markdown(f"<div style='margin:10px 0'><div class='chat-ai-bubble'>{chr(10).join(html_parts)}</div></div>", unsafe_allow_html=True)

    sp()
    ci, cb, cc = st.columns([5,1,1])
    with ci:
        user_in = st.text_input("Ask anything…", key="user_input",
            placeholder="e.g. Which SKU should I reorder first and what quantity?",
            label_visibility="collapsed")
    with cb:
        if st.button("Send ↗", use_container_width=True):
            if not key_ok:
                st.warning("⚠️ Enter your Groq API key in the sidebar first.")
            elif user_in.strip():
                st.session_state.chat_msgs.append({"role":"user","content":user_in.strip()})
                history = st.session_state.chat_msgs[-20:]
                with st.spinner("OmniFlow thinking…"):
                    reply = call_llm(history, system, api_key.strip())
                st.session_state.chat_msgs.append({"role":"assistant","content":reply})
                st.rerun()
    with cc:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_msgs = []
            st.rerun()

    if not st.session_state.chat_msgs:
        sp()
        sec("Live Decision Alerts", "⚡")
        al1, al2 = st.columns(2, gap="large")
        with al1:
            st.markdown("""<div style='font-size:0.74rem;font-weight:700;color:#ff6b6b;
                letter-spacing:0.06em;text-transform:uppercase;font-family:DM Mono,monospace;margin-bottom:10px'>
                🔴 Critical SKUs — Reorder NOW</div>""", unsafe_allow_html=True)
            inv = compute_inventory()
            for _, r in inv[inv["Status"]=="🔴 Critical"][["Product_Name","Category","Current_Stock","ROP"]].head(5).iterrows():
                st.markdown(f"<div class='alert-item alert-critical'><b style='color:#f0f4ff'>{r['Product_Name']}</b> <span style='color:#4a5e7a'>[{r['Category']}]</span><br><span style='color:#4a5e7a;font-size:0.71rem'>Stock: {r['Current_Stock']} · ROP: {r['ROP']}</span></div>", unsafe_allow_html=True)
        with al2:
            st.markdown("""<div style='font-size:0.74rem;font-weight:700;color:#f5a623;
                letter-spacing:0.06em;text-transform:uppercase;font-family:DM Mono,monospace;margin-bottom:10px'>
                💰 Cost Saving Opportunities</div>""", unsafe_allow_html=True)
            _, _, opt, _ = compute_logistics()
            for _, r in opt.sort_values("Potential_Saving", ascending=False).head(5).iterrows():
                if r["Potential_Saving"] > 0:
                    st.markdown(f"<div class='alert-item alert-warn'><b style='color:#f0f4ff'>{r['Region']}</b> → <b style='color:#56e0a0'>{r['Optimal_Carrier']}</b><br><span style='color:#4a5e7a;font-size:0.71rem'>Save ₹{r['Potential_Saving']:,.0f} ({r['Saving_Pct']:.1f}%)</span></div>", unsafe_allow_html=True)

        sp()
        sec("Revenue Forecast — Next 3 Months", "📈")
        m_rev = ops.groupby("YM")["Net_Revenue"].sum().rename("v")
        r_rev = ml_forecast(m_rev.values.astype(float), m_rev.index, 3)
        if r_rev is not None:
            last = float(m_rev.iloc[-1])
            rc   = st.columns(3)
            for i, (dt, fc, lo, hi) in enumerate(zip(r_rev["fut_ds"], r_rev["forecast"], r_rev["ci_lo"], r_rev["ci_hi"])):
                chg = (fc - last) / last * 100 if last > 0 else 0
                kpi(rc[i], f"{'📈' if chg>=0 else '📉'} {dt.strftime('%b %Y')}", f"₹{fc/1e6:.1f}M",
                    "mint" if chg >= 0 else "coral", f"{chg:+.1f}% | CI ₹{lo/1e6:.1f}M–₹{hi/1e6:.1f}M")
                last = fc

# ═══════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ═══════════════════════════════════════════════════════════
def page_overview():
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("""<div class='page-title' style='background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow D2D</div>
    <div class='page-subtitle'>Predictive Logistics & AI-Powered Demand-to-Delivery Intelligence · India</div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class='about-card'>
      <div style='font-size:1.0rem;font-weight:700;color:#f0f4ff;margin-bottom:10px'>About This Platform</div>
      <p style='color:#8a9dc0;line-height:1.85;margin:0;font-size:0.86rem'>
        <b style='color:#f5a623'>OmniFlow</b> is a full-stack supply-chain intelligence platform built on
        <b style='color:#f0f4ff'>5,200 D2D orders</b> across India (Jan 2024–Dec 2025).
        Modules are causally connected: Demand signals drive Inventory EOQ/SS,
        which drives Production targets, which informs Logistics optimisation.
        All forecasting uses <b style='color:#2ed8c3'>Ridge Regression with structural-break detection</b>.
        Revenue KPIs are net of returns. Geographic data is corrected (Pune → Maharashtra).
      </p>
      <div style='margin-top:14px'>
        <span class='badge badge-amber'>Demand Forecast</span>
        <span class='badge badge-teal'>Inventory EOQ/ROP</span>
        <span class='badge badge-lav'>Production Plan</span>
        <span class='badge badge-coral'>Logistics + Cost Opt</span>
        <span class='badge badge-sky'>AI Chatbot (Groq)</span>
        <span class='badge badge-mint'>RMSE · NRMSE · R²</span>
      </div>
    </div>""", unsafe_allow_html=True)

    delivered = df[df["Order_Status"]=="Delivered"]
    net_rev   = ops["Net_Revenue"].sum()
    ret_rate  = df[df["Order_Status"]=="Returned"].shape[0] / len(ops) * 100
    avg_del   = delivered["Delivery_Days"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1,"Net Revenue",    f"₹{net_rev/1e7:.1f}Cr",       "amber","excl. returns")
    kpi(c2,"Active Orders",  f"{len(ops):,}",                "teal", "Del+Shipped")
    kpi(c3,"Units Sold",     f"{ops['Quantity'].sum():,}",   "sky",  "all products")
    kpi(c4,"Return Rate",    f"{ret_rate:.1f}%",             "coral","of active orders")
    kpi(c5,"Avg Delivery",   f"{avg_del:.1f}d",              "lav",  "delivered only")
    kpi(c6,"SKU Categories", f"{df['Category'].nunique()}",  "mint", "product types")
    sp()

    c_l, c_r = st.columns([3,2], gap="large")
    with c_l:
        sec("Monthly Net Revenue Trend")
        m = ops.groupby(ops["Order_Date"].dt.to_period("M"))["Net_Revenue"].sum().reset_index()
        m["ds"] = m["Order_Date"].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["ds"], y=m["Net_Revenue"], fill="tozeroy",
            line=dict(color="#f5a623",width=2.5), fillcolor="rgba(245,166,35,0.06)",
            hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"))
        fig.update_layout(**CD(), height=260, xaxis=gX(), yaxis={**gY(),"tickformat":",.0f"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        sec("Net Revenue by Category")
        cat = ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(labels=cat.index, values=cat.values, hole=.58,
            marker=dict(colors=COLORS, line=dict(color="#080e1a",width=3)),
            textinfo="label+percent", textfont=dict(size=10,color="#f0f4ff")))
        fig2.update_layout(**CD(), height=260, showlegend=False,
            annotations=[dict(text="Net Rev",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#4a5e7a",family="DM Mono"))])
        st.plotly_chart(fig2, use_container_width=True)

    sp(0.5)
    c3a, c3b, c3c = st.columns(3, gap="large")
    with c3a:
        sec("Orders by Channel")
        ch = ops["Sales_Channel"].value_counts()
        fig3 = go.Figure(go.Bar(x=ch.values, y=ch.index, orientation="h",
            marker=dict(color=COLORS[:len(ch)], line=dict(color="rgba(0,0,0,0)")),
            text=ch.values, textposition="outside", textfont=dict(color="#4a5e7a",size=10)))
        fig3.update_layout(**CD(), height=240, xaxis=gX(), yaxis=dict(showgrid=False,color="#8a9dc0"))
        st.plotly_chart(fig3, use_container_width=True)

    with c3b:
        sec("Top Regions by Net Revenue")
        reg = ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False)
        fig4 = go.Figure(go.Bar(x=reg.index, y=reg.values,
            marker=dict(color=COLORS_S*2, line=dict(color="rgba(0,0,0,0)"))))
        fig4.update_layout(**CD(), height=240, xaxis={**gX(),"tickangle":-30}, yaxis=gY())
        st.plotly_chart(fig4, use_container_width=True)

    with c3c:
        sec("Order Status Split")
        sc2 = df["Order_Status"].value_counts()
        fig5 = go.Figure(go.Bar(x=sc2.index, y=sc2.values,
            marker=dict(color=["#56e0a0","#5ba4e5","#ff6b6b","#f5a623"][:len(sc2)], line=dict(color="rgba(0,0,0,0)"))))
        fig5.update_layout(**CD(), height=240, xaxis=gX(), yaxis=gY())
        st.plotly_chart(fig5, use_container_width=True)

    sp(0.5)
    sec("Module Dependency Pipeline")
    st.markdown("""<div style='background:linear-gradient(135deg,var(--panel),var(--surface));
        border:1px solid var(--border);border-radius:16px;padding:22px;
        display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:0'>
      <div style='background:var(--deep);border-radius:12px;padding:11px 17px;font-weight:700;font-size:0.78rem;
           text-align:center;min-width:95px;border:1px solid rgba(245,166,35,0.4);color:#f5a623;font-family:DM Mono,monospace'>
           Demand<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>Forecast</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:var(--deep);border-radius:12px;padding:11px 17px;font-weight:700;font-size:0.78rem;
           text-align:center;min-width:95px;border:1px solid rgba(86,224,160,0.4);color:#56e0a0;font-family:DM Mono,monospace'>
           Inventory<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>EOQ + ROP</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:var(--deep);border-radius:12px;padding:11px 17px;font-weight:700;font-size:0.78rem;
           text-align:center;min-width:95px;border:1px solid rgba(155,135,212,0.4);color:#9b87d4;font-family:DM Mono,monospace'>
           Production<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>6-Month Plan</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:var(--deep);border-radius:12px;padding:11px 17px;font-weight:700;font-size:0.78rem;
           text-align:center;min-width:95px;border:1px solid rgba(255,107,107,0.4);color:#ff6b6b;font-family:DM Mono,monospace'>
           Logistics<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>+ Cost Opt</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:var(--deep);border-radius:12px;padding:11px 17px;font-weight:700;font-size:0.78rem;
           text-align:center;min-width:95px;border:1px solid rgba(91,164,229,0.4);color:#5ba4e5;font-family:DM Mono,monospace'>
           AI Chatbot<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>Groq LLaMA</span></div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE — DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════
def page_demand():
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#f5a623'>Demand Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Ridge Regression · Structural Break · Fourier Seasonality · RMSE / NRMSE / R²</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>OUTPUT → Inventory</span>
      <span class='badge badge-teal'>→ Production</span>
      <span class='badge badge-coral'>→ Logistics</span>
      <span class='badge badge-sky'>→ Chatbot</span>
    </div>""", unsafe_allow_html=True)

    banner("""<b style='color:#f5a623'>Model:</b> Ridge Regression with trend (t, t²),
    Fourier seasonal terms (2 harmonics), structural-break dummy for business scale-up,
    and trend×regime interaction. 4-month hold-out evaluation. 90% CI from residual std.""", "amber")

    c1, c2, c3 = st.columns([2,2,1])
    metric_opt = c1.selectbox("Metric",    ["Orders (#)","Quantity (Units)","Net Revenue (₹)"])
    level_opt  = c2.selectbox("Breakdown", ["Overall","Category","Region","Sales Channel"])
    horizon    = c3.slider("Months ahead", 3, 12, 6)

    col_map = {"Orders (#)":"Order_ID","Quantity (Units)":"Quantity","Net Revenue (₹)":"Net_Revenue"}
    col = col_map[metric_opt]

    def get_series(sub):
        if col == "Order_ID":
            return sub.groupby("YM")["Order_ID"].count().rename("v")
        return sub.groupby("YM")[col].sum().rename("v")

    def draw(series, color="#f5a623", title=""):
        res = ml_forecast(series.values.astype(float), series.index, n_future=horizon)
        if res is None:
            st.info("Insufficient data."); return None

        fig = go.Figure()
        x_ci = list(res["fut_ds"]) + list(res["fut_ds"])[::-1]
        y_ci = list(res["ci_hi"])  + list(res["ci_lo"])[::-1]
        fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
            fillcolor="rgba(245,166,35,0.07)", line=dict(color="rgba(0,0,0,0)"), name="90% CI", showlegend=False))
        fig.add_trace(go.Scatter(x=res["hist_ds"], y=res["fitted"], name="Model Fit",
            line=dict(color=color,width=1.5,dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=res["hist_ds"], y=res["hist_y"], name="Actual",
            line=dict(color="#4a5e7a",width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=res["fut_ds"], y=res["forecast"], name="Forecast",
            line=dict(color=color,width=2.5,dash="dot"), mode="lines+markers",
            marker=dict(size=7,color=color,line=dict(color="#080e1a",width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=res["eval_ds"], y=res["eval_pred"], name="Eval Pred",
            mode="markers", marker=dict(size=10,color="#ff6b6b",symbol="x",line=dict(color="#080e1a",width=2))))
        fig.update_layout(**CD(), height=320, xaxis=gX(), yaxis=gY(), legend=leg(),
            title=dict(text=title, font=dict(color="#4a5e7a",size=11)))
        st.plotly_chart(fig, use_container_width=True)

        m1,m2,m3,m4 = st.columns(4)
        kpi(m1,"RMSE",     f"{res['rmse']:.1f}",      "coral","hold-out 4 mo")
        kpi(m2,"NRMSE",    f"{res['nrmse']*100:.1f}%","amber","normalised RMSE")
        kpi(m3,"MAE",      f"{res['mae']:.1f}",       "sky",  "mean abs error")
        kpi(m4,"R² Score", f"{res['r2']:.3f}",        "mint", "model fit")
        sp(0.5)
        return res

    if level_opt == "Overall":
        series = get_series(ops)
        res = draw(series)
        if res is not None:
            sec("Forecast Table")
            tbl = pd.DataFrame({
                "Month":     [d.strftime("%b %Y") for d in res["fut_ds"]],
                "Forecast":  res["forecast"].round(0).astype(int),
                "Lower 90%": res["ci_lo"].round(0).astype(int),
                "Upper 90%": res["ci_hi"].round(0).astype(int),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        grp_map = {"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}
        grp  = grp_map[level_opt]
        top  = ops[grp].value_counts().head(5).index.tolist()
        tabs = st.tabs(top)
        for tab, val, color in zip(tabs, top, COLORS):
            with tab:
                draw(get_series(ops[ops[grp]==val]), color=color, title=val)

    sp()
    sec("YoY Revenue Growth — Actual + Projected")
    yr_rev      = ops.groupby(["Year","Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    cat_monthly = ops.groupby(["YM","Category"])["Net_Revenue"].sum().unstack(fill_value=0)

    proj_next = {}
    for cat in cat_monthly.columns:
        vals = cat_monthly[cat].values.astype(float)
        r    = ml_forecast(vals, cat_monthly.index, n_future=12)
        if r is None: continue
        proj_next[cat] = sum(v for d,v in zip(r["fut_ds"], r["forecast"]) if d.year == r["fut_ds"][0].year)

    if 2024 in yr_rev.index and 2025 in yr_rev.index:
        rows_yoy = []
        for cat in yr_rev.columns:
            r24 = yr_rev.loc[2024,cat]; r25 = yr_rev.loc[2025,cat]
            r_proj = proj_next.get(cat,0)
            g25 = (r25-r24)/r24*100 if r24>0 else 0
            g_proj = (r_proj-r25)/r25*100 if r25>0 else 0
            rows_yoy.append({"Category":cat,
                "2024 (₹M)":round(r24/1e6,1), "2025 (₹M)":round(r25/1e6,1),
                "YoY 24→25":f"{g25:+.1f}%",
                "Projected (₹M)":round(r_proj/1e6,1), "YoY Growth (Proj)":f"{g_proj:+.1f}% ⟵"})
        st.dataframe(pd.DataFrame(rows_yoy).sort_values("Projected (₹M)", ascending=False),
                     use_container_width=True, hide_index=True)

    sp()
    sec("Category-Level Demand Forecast (Quantity)")
    tabs2 = st.tabs(list(cat_monthly.columns))
    for tab, cat, col2 in zip(tabs2, cat_monthly.columns, COLORS):
        with tab:
            vals = ops[ops["Category"]==cat].groupby("YM")["Quantity"].sum().rename("v")
            draw(vals, color=col2, title=cat)

# ═══════════════════════════════════════════════════════════
# PAGE — INVENTORY
# ═══════════════════════════════════════════════════════════
def page_inventory():
    df  = load_data()
    ops = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#56e0a0'>Inventory Optimisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Wilson EOQ · Safety Stock Formula · Simulated (s,Q) Policy · Data-Driven Stock Status</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>⬆ from Demand Forecast</span>
      <span class='badge badge-teal'>feeds → Production</span>
      <span class='badge badge-sky'>→ Chatbot</span>
    </div>""", unsafe_allow_html=True)

    banner("""<b style='color:#2ed8c3'>Methodology:</b>
    EOQ = √(2DS / hC) (Wilson formula).
    Safety Stock = z × σ_daily × √LT where σ_daily = monthly_std / √30.
    Stock levels simulated using (s,Q) policy over actual demand history.""", "teal")

    with st.expander("⚙ Inventory Parameters", expanded=False):
        p1,p2,p3,p4 = st.columns(4)
        order_cost = p1.number_input("Order Cost ₹", 100, 5000, 500, 50)
        hold_pct   = p2.slider("Holding Cost %", 5, 40, 20) / 100
        lead_time  = p3.slider("Lead Time (days)", 1, 30, 7)
        svc        = p4.selectbox("Service Level", ["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"])
        z_map      = {"90% (z=1.28)":1.28,"95% (z=1.65)":1.65,"99% (z=2.33)":2.33}
        z          = z_map[svc]

    inv    = compute_inventory(order_cost, hold_pct, lead_time, z)
    n_crit = (inv["Status"]=="🔴 Critical").sum()
    n_low  = (inv["Status"]=="🟡 Low").sum()
    n_ok   = (inv["Status"]=="🟢 Adequate").sum()

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Total SKUs",  len(inv), "sky")
    kpi(c2,"🔴 Critical", n_crit,   "coral","immediate reorder")
    kpi(c3,"🟡 Low",      n_low,    "amber","approaching ROP")
    kpi(c4,"🟢 Adequate", n_ok,     "mint", "well-stocked")
    sp()

    cl, cr = st.columns([1,2], gap="large")
    with cl:
        sec("Stock Status Distribution")
        sc_colors = {"🔴 Critical":"#ff6b6b","🟡 Low":"#f5a623","🟢 Adequate":"#56e0a0"}
        sc = inv["Status"].value_counts()
        fig = go.Figure(go.Pie(labels=sc.index, values=sc.values, hole=.6,
            marker=dict(colors=[sc_colors.get(s,"#4a5e7a") for s in sc.index], line=dict(color="#080e1a",width=3)),
            textinfo="label+value", textfont=dict(size=10,color="#f0f4ff")))
        fig.update_layout(**CD(), height=270, showlegend=False,
            annotations=[dict(text="SKUs",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#4a5e7a",family="DM Mono"))])
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        sec("EOQ / Safety Stock / ROP by Category")
        ci2 = inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2 = go.Figure()
        for i,(m2,lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(name=lbl, x=ci2["Category"], y=ci2[m2].round(1),
                marker=dict(color=["#f5a623","#2ed8c3","#9b87d4"][i], line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(), height=270, barmode="group",
            xaxis={**gX(),"tickangle":-10}, yaxis=gY(), legend=leg())
        st.plotly_chart(fig2, use_container_width=True)

    sec("Critical SKU Alerts — Reorder Immediately")
    crit_df = inv[inv["Status"]=="🔴 Critical"][["SKU_ID","Product_Name","Category","Current_Stock","SS","ROP","EOQ","Monthly_Avg","Unit_Price"]].copy()
    crit_df.columns = ["SKU","Product","Category","Current Stock","Safety Stock","ROP","Order Qty (EOQ)","Avg/Month","Unit Price ₹"]
    for c in ["Current Stock","Safety Stock","ROP","Order Qty (EOQ)"]:
        crit_df[c] = crit_df[c].astype(int)
    st.dataframe(crit_df.sort_values("Current Stock"), use_container_width=True, hide_index=True)

    sp()
    sec("Full SKU-Level Inventory Table")
    cat_f  = st.multiselect("Filter Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    stat_f = st.multiselect("Filter Status", ["🔴 Critical","🟡 Low","🟢 Adequate"], default=["🔴 Critical","🟡 Low","🟢 Adequate"])
    disp   = inv[(inv["Category"].isin(cat_f))&(inv["Status"].isin(stat_f))][["SKU_ID","Product_Name","Category","Monthly_Avg","Current_Stock","EOQ","SS","ROP","Unit_Price","Status"]].copy()
    disp.columns = ["SKU","Product","Category","Avg/Month","Current Stock","EOQ","Safety Stock","ROP","Price ₹","Status"]
    for c in ["Avg/Month","Current Stock","EOQ","Safety Stock","ROP"]:
        disp[c] = disp[c].astype(int)
    st.dataframe(disp.sort_values("Status"), use_container_width=True, hide_index=True)

    sp()
    sec("Category Demand Forecast → Inventory Needs")
    cat_monthly = ops.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
    fig3 = go.Figure()
    for i, cat in enumerate(cat_monthly.columns):
        r = ml_forecast(cat_monthly[cat].values.astype(float), cat_monthly.index, 6)
        if r is None: continue
        clr = COLORS[i % len(COLORS)]
        fig3.add_trace(go.Scatter(x=r["hist_ds"], y=r["hist_y"], line=dict(color=clr,width=1,dash="dot"), opacity=0.2, showlegend=False))
        fig3.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=cat,
            mode="lines+markers", line=dict(color=clr,width=2.5), marker=dict(size=7,color=clr,line=dict(color="#080e1a",width=2))))
    fig3.update_layout(**CD(), height=280, xaxis=gX(), yaxis={**gY(),"title":"Forecast Units"}, legend=leg())
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE — PRODUCTION
# ═══════════════════════════════════════════════════════════
def page_production():
    st.markdown("<div class='page-title' style='color:#9b87d4'>Production Planning</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>6-Month Targets · Demand + Inventory Signal-Driven · Capacity Simulation</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>⬆ from Demand Forecast</span>
      <span class='badge badge-coral'>⬆ from Inventory Status</span>
      <span class='badge badge-lav'>feeds → Logistics</span>
      <span class='badge badge-sky'>→ Chatbot</span>
    </div>""", unsafe_allow_html=True)

    p1,p2 = st.columns(2)
    cap = p1.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1)
    buf = p2.slider("Safety Buffer %", 5, 40, 15) / 100

    plan = compute_production(cap, buf)
    inv  = compute_inventory()
    n_crit = (inv["Status"]=="🔴 Critical").sum()

    if plan.empty:
        st.warning("Insufficient data for production plan."); return

    agg = plan.groupby("Month_dt")[["Production","Demand_Forecast","Crit_Boost","Low_Boost"]].sum().reset_index()

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Total Production", f"{plan['Production'].sum():,.0f}","lav","units · 6 months")
    kpi(c2,"Total Demand",     f"{plan['Demand_Forecast'].sum():,.0f}","sky","forecast units")
    kpi(c3,"Avg / Month",      f"{agg['Production'].mean():,.0f}","mint","units/month")
    peak = agg.loc[agg["Production"].idxmax(),"Month_dt"]
    kpi(c4,"Peak Month",       peak.strftime("%b %Y"),"amber","highest demand")
    sp()

    banner(f"""<b style='color:#ff6b6b'>Inventory Signal Active:</b>
    {n_crit} Critical SKUs detected. Production targets include replenishment boost of
    <b style='color:#f5a623'>+{agg['Crit_Boost'].sum():,.0f} units</b> (critical) and
    +{agg['Low_Boost'].sum():,.0f} units (low stock) over 6 months.""", "coral")

    sec("Production Target vs Demand Forecast")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Production"], name="Production Target",
        marker=dict(color="#9b87d4", line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Crit_Boost"]+agg["Low_Boost"],
        name="Inv. Replenishment", marker=dict(color="rgba(255,107,107,0.7)", line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Scatter(x=agg["Month_dt"], y=agg["Demand_Forecast"], name="Demand Forecast",
        mode="lines+markers", line=dict(color="#f5a623",width=2.5),
        marker=dict(size=8,color="#f5a623",line=dict(color="#080e1a",width=2))))
    fig.update_layout(**CD(), height=300, barmode="stack", xaxis=gX(), yaxis=gY(), legend=leg())
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2, gap="large")
    with cl:
        sec("Production by Category (Stacked)")
        fig2 = go.Figure()
        for i, cat in enumerate(plan["Category"].unique()):
            s = plan[plan["Category"]==cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(x=s["Month_dt"], y=s["Production"], name=cat,
                marker=dict(color=COLORS[i%len(COLORS)], line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(), height=280, barmode="stack",
            xaxis=gX(), yaxis=gY(), legend={**leg(),"orientation":"h","y":-0.3})
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        sec("Production – Demand Gap")
        agg["Gap"] = agg["Production"] - agg["Demand_Forecast"]
        fig3 = go.Figure(go.Bar(x=agg["Month_dt"], y=agg["Gap"],
            marker=dict(color=["#56e0a0" if g>=0 else "#ff6b6b" for g in agg["Gap"]], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside", textfont=dict(color="#4a5e7a")))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.1)")
        fig3.update_layout(**CD(), height=280, xaxis=gX(), yaxis={**gY(),"title":"Units Surplus / Deficit"})
        st.plotly_chart(fig3, use_container_width=True)

    sec("Detailed Production Schedule")
    cat_f = st.selectbox("Filter Category", ["All"] + list(plan["Category"].unique()))
    d2 = plan if cat_f=="All" else plan[plan["Category"]==cat_f]
    d3 = d2[["Month","Category","Demand_Forecast","Crit_Boost","Low_Boost","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d3.columns = ["Month","Category","Demand Fc","Crit Boost","Low Boost","Buffer","Production","Demand Lo","Demand Hi"]
    st.dataframe(d3.sort_values("Month"), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# PAGE — LOGISTICS
# ═══════════════════════════════════════════════════════════
def page_logistics():
    df     = load_data()
    ops    = get_ops(df)
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    del_df = get_delivered(df)

    st.markdown("<div class='page-title' style='color:#ff6b6b'>Logistics Optimisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Carrier Scorecard · Delay Intel · Cost Optimisation · Warehouse Forecast · Region Analysis</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>⬆ Demand Forecast</span>
      <span class='badge badge-lav'>⬆ Production Volumes</span>
      <span class='badge badge-sky'>feeds → Chatbot</span>
    </div>""", unsafe_allow_html=True)

    carr, best_carr, opt, curr_alloc = compute_logistics()
    t1,t2,t3,t4,t5 = st.tabs(["Carrier Scorecard","Cost Optimisation","⚠ Delay Intel","Warehouse Forecast","🗺 Regions"])

    with t1:
        sec("Carrier Performance Scorecard")
        banner("""<b style='color:#5ba4e5'>Performance Score</b> = (1/Delay_Index) × (1−Return_Rate) × (1/Cost_Score). Higher is better.""", "teal")
        fig = go.Figure()
        for i, (_, r) in enumerate(carr.iterrows()):
            fig.add_trace(go.Scatter(x=[r["Avg_Days"]], y=[r["Avg_Cost"]], mode="markers+text",
                marker=dict(size=max(r["Orders"]/35,16), color=COLORS[i], opacity=0.9, line=dict(color="#080e1a",width=2)),
                text=[r["Courier_Partner"]], textposition="top center", name=r["Courier_Partner"],
                hovertemplate=f"<b>{r['Courier_Partner']}</b><br>Orders:{r['Orders']}<br>Avg Del:{r['Avg_Days']:.1f}d<br>Avg Cost:₹{r['Avg_Cost']:.0f}<br>Score:{r['Perf_Score']:.3f}<extra></extra>"))
        fig.update_layout(**CD(), height=320, showlegend=False,
            xaxis={**gY(),"title":"Avg Delivery Days"}, yaxis={**gY(),"title":"Avg Shipping Cost ₹"})
        st.plotly_chart(fig, use_container_width=True)

        d2 = carr.copy()
        d2["Avg_Days"]    = d2["Avg_Days"].round(1)
        d2["Avg_Cost"]    = d2["Avg_Cost"].round(1)
        d2["Return_Rate"] = (d2["Return_Rate"]*100).round(1).astype(str)+"%"
        d2["Perf_Score"]  = d2["Perf_Score"].round(3)
        d2.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Total Cost ₹","Delay Index","Cost Score","Perf Score"]
        st.dataframe(d2[["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Delay Index","Perf Score"]], use_container_width=True, hide_index=True)

        sp()
        sec("Carrier Order Trend + Forecast")
        cm = del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        cl2, cr2 = st.columns(2, gap="large")
        with cl2:
            sec("Historical Trend")
            fig2 = go.Figure()
            for i, c in enumerate(cm.columns):
                fig2.add_trace(go.Scatter(x=cm.index.to_timestamp(), y=cm[c], name=c, line=dict(color=COLORS[i%len(COLORS)],width=2)))
            fig2.update_layout(**CD(), height=250, xaxis=gX(), yaxis=gY(), legend=leg())
            st.plotly_chart(fig2, use_container_width=True)
        with cr2:
            sec("Carrier Volume Forecast")
            fig3 = go.Figure()
            for i, c in enumerate(cm.columns):
                r = ml_forecast(cm[c].values.astype(float), cm.index, 6)
                if r is None: continue
                fig3.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=c,
                    mode="lines+markers", line=dict(color=COLORS[i%len(COLORS)],width=2,dash="dot"),
                    marker=dict(size=7,line=dict(color="#080e1a",width=1.5))))
            fig3.update_layout(**CD(), height=250, xaxis=gX(), yaxis={**gY(),"title":"Orders"}, legend=leg())
            st.plotly_chart(fig3, use_container_width=True)

        sec("Recommended Carrier per Category")
        plan = compute_production()
        if not plan.empty:
            prod_by_cat = plan.groupby("Category")["Production"].sum().reset_index()
            best_cat = (del_df.groupby(["Category","Courier_Partner"])["Delivery_Days"]
                .mean().reset_index().sort_values("Delivery_Days").groupby("Category").first().reset_index())
            best_cat = best_cat.merge(prod_by_cat.rename(columns={"Production":"Planned Units 6M"}), on="Category", how="left")
            best_cat["Delivery_Days"]    = best_cat["Delivery_Days"].round(1)
            best_cat["Planned Units 6M"] = best_cat["Planned Units 6M"].fillna(0).astype(int)
            best_cat.columns = ["Category","Recommended Carrier","Avg Days","Planned Units (6mo)"]
            st.dataframe(best_cat, use_container_width=True, hide_index=True)

    with t2:
        sec("Logistics Cost Optimisation Analysis", "💰")
        total_current = del_df["Shipping_Cost_INR"].sum()
        total_saving  = opt["Potential_Saving"].sum()
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1,"Current Spend",         f"₹{total_current:,.0f}","coral","all deliveries")
        kpi(c2,"Optimised Spend",       f"₹{total_current-total_saving:,.0f}","mint","with best carriers")
        kpi(c3,"Total Saving Potential",f"₹{total_saving:,.0f}","amber","by region switch")
        kpi(c4,"Saving %",              f"{total_saving/total_current*100:.1f}%","sky","of total spend")
        sp()
        banner(f"""<b style='color:#f5a623'>Optimisation Logic:</b>
        For each region, identify the carrier with lowest avg shipping cost.
        Potential saving = (Current Avg − Min Avg) × Orders.
        Total potential: <b style='color:#56e0a0'>₹{total_saving:,.0f}</b>""", "amber")

        sec("Region-Level Cost Comparison")
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(name="Current Avg ₹", x=opt["Region"], y=opt["Current_Avg_Cost"], marker=dict(color="#ff6b6b", line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.add_trace(go.Bar(name="Optimal Avg ₹", x=opt["Region"], y=opt["Min_Avg_Cost"],     marker=dict(color="#56e0a0", line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.update_layout(**CD(), height=280, barmode="group", xaxis={**gX(),"tickangle":-25}, yaxis=gY(), legend=leg())
        st.plotly_chart(fig_cost, use_container_width=True)

        sec("Saving by Region")
        s_sorted = opt.sort_values("Potential_Saving", ascending=False)
        fig_sav = go.Figure(go.Bar(x=s_sorted["Region"], y=s_sorted["Potential_Saving"],
            marker=dict(color="#f5a623", line=dict(color="rgba(0,0,0,0)")),
            text=[f"₹{v:,.0f}" for v in s_sorted["Potential_Saving"]], textposition="outside", textfont=dict(color="#4a5e7a")))
        fig_sav.update_layout(**CD(), height=250, xaxis={**gX(),"tickangle":-25}, yaxis=gY())
        st.plotly_chart(fig_sav, use_container_width=True)

        sec("Optimisation Recommendation Table")
        opt_disp = opt.copy()
        opt_disp["Current_Avg_Cost"] = opt_disp["Current_Avg_Cost"].round(1)
        opt_disp["Min_Avg_Cost"]     = opt_disp["Min_Avg_Cost"].round(1)
        opt_disp["Potential_Saving"] = opt_disp["Potential_Saving"].astype(int)
        opt_disp = opt_disp[["Region","Optimal_Carrier","Current_Avg_Cost","Min_Avg_Cost","Potential_Saving","Saving_Pct","Orders"]]
        opt_disp.columns = ["Region","Switch To","Current Avg ₹","Optimal Avg ₹","Saving ₹","Saving %","Orders"]
        st.dataframe(opt_disp.sort_values("Saving ₹", ascending=False), use_container_width=True, hide_index=True)

    with t3:
        sec("Delay Hotspot Analysis", "⚠️")
        thr = st.slider("Delay Threshold (days)", 3, 10, 7)
        del_df2 = del_df.copy()
        del_df2["Delayed"] = del_df2["Delivery_Days"] > thr

        cl3, cr3 = st.columns(2, gap="large")
        with cl3:
            sec("Delay Rate by Region")
            rd = del_df2.groupby("Region").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
            rd["Rate"] = (rd["D"]/rd["T"]*100).round(1)
            rd_s = rd.sort_values("Region")
            fig_r = go.Figure(go.Bar(x=rd_s["Rate"], y=rd_s["Region"], orientation="h",
                marker=dict(color=[f"rgba(255,107,107,{min(v/60+0.25,0.9):.2f})" for v in rd_s["Rate"]], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v}%" for v in rd_s["Rate"]], textposition="outside", textfont=dict(color="#4a5e7a")))
            fig_r.update_layout(**CD(), height=300, xaxis={**gX(),"title":"Delay %"}, yaxis=dict(showgrid=False,color="#8a9dc0"))
            st.plotly_chart(fig_r, use_container_width=True)
        with cr3:
            sec("Delay Rate by Carrier")
            cd = del_df2.groupby("Courier_Partner").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
            cd["Rate"] = (cd["D"]/cd["T"]*100).round(1)
            fig_c = go.Figure(go.Bar(x=cd["Courier_Partner"], y=cd["Rate"],
                marker=dict(color=["#ff6b6b" if v>35 else "#f5a623" if v>20 else "#56e0a0" for v in cd["Rate"]], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v}%" for v in cd["Rate"]], textposition="outside", textfont=dict(color="#4a5e7a")))
            fig_c.update_layout(**CD(), height=300, xaxis=gX(), yaxis={**gY(),"title":"Delay %"})
            st.plotly_chart(fig_c, use_container_width=True)

        sec("Carrier × Region Delay Heatmap")
        pv = del_df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0)*100
        fig_h = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index),
            colorscale=[[0,"#0d1829"],[0.4,"#7c4fd0"],[0.7,"#e87adb"],[1,"#ff6b6b"]],
            text=np.round(pv.values,1), texttemplate="%{text}%", textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color="#8a9dc0",size=10))))
        fig_h.update_layout(**CD(), height=260,
            xaxis=dict(showgrid=False,tickangle=-25,color="#8a9dc0"),
            yaxis=dict(showgrid=False,color="#8a9dc0"))
        st.plotly_chart(fig_h, use_container_width=True)

        sec("Avg Delivery Days Forecast")
        delay_m = del_df.groupby(del_df["Order_Date"].dt.to_period("M"))["Delivery_Days"].mean().rename("v")
        r_del   = ml_forecast(delay_m.values.astype(float), delay_m.index, 6)
        if r_del:
            fig_d = go.Figure()
            x_ci = list(r_del["fut_ds"])+list(r_del["fut_ds"])[::-1]
            y_ci = list(r_del["ci_hi"])+list(r_del["ci_lo"])[::-1]
            fig_d.add_trace(go.Scatter(x=x_ci,y=y_ci,fill="toself",fillcolor="rgba(255,107,107,0.07)",line=dict(color="rgba(0,0,0,0)"),showlegend=False))
            fig_d.add_trace(go.Scatter(x=r_del["hist_ds"],y=r_del["hist_y"],name="Historical",line=dict(color="#4a5e7a",width=2)))
            fig_d.add_trace(go.Scatter(x=r_del["fut_ds"],y=r_del["forecast"],name="Forecast",
                line=dict(color="#ff6b6b",width=2.5,dash="dot"),mode="lines+markers",
                marker=dict(size=8,color="#ff6b6b",line=dict(color="#080e1a",width=2))))
            fig_d.update_layout(**CD(), height=250, xaxis=gX(), yaxis={**gY(),"title":"Avg Delivery Days"}, legend=leg())
            st.plotly_chart(fig_d, use_container_width=True)

    with t4:
        sec("Warehouse Shipment Volume Trend")
        wm = del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh = go.Figure()
        for i, wh in enumerate(wm.columns):
            fig_wh.add_trace(go.Bar(x=wm.index.to_timestamp(), y=wm[wh], name=wh,
                marker=dict(color=COLORS[i%len(COLORS)], line=dict(color="rgba(0,0,0,0)"))))
        fig_wh.update_layout(**CD(), height=270, barmode="stack", xaxis=gX(), yaxis=gY(), legend=leg())
        st.plotly_chart(fig_wh, use_container_width=True)

        sec("Warehouse Demand Forecast")
        wf_rows = []
        for wh in wm.columns:
            r = ml_forecast(wm[wh].values.astype(float), wm.index, 6)
            if r is None: continue
            for dt, fc, hi in zip(r["fut_ds"], r["forecast"], r["ci_hi"]):
                wf_rows.append({"Month":dt,"Warehouse":wh,"Forecast":fc,"Upper":hi})
        if wf_rows:
            wfd = pd.DataFrame(wf_rows)
            fig_wf = go.Figure()
            for i, wh in enumerate(wfd["Warehouse"].unique()):
                s = wfd[wfd["Warehouse"]==wh]
                fig_wf.add_trace(go.Scatter(x=s["Month"], y=s["Forecast"], name=wh,
                    mode="lines+markers", line=dict(color=COLORS[i%len(COLORS)],width=2.5,dash="dot"),
                    marker=dict(size=8,line=dict(color="#080e1a",width=2))))
            fig_wf.update_layout(**CD(), height=250, xaxis=gX(), yaxis=gY(), legend=leg())
            st.plotly_chart(fig_wf, use_container_width=True)
            tbl_wf = wfd.copy()
            tbl_wf["Month"]    = tbl_wf["Month"].dt.strftime("%b %Y")
            tbl_wf["Forecast"] = tbl_wf["Forecast"].round(0).astype(int)
            tbl_wf["Upper"]    = tbl_wf["Upper"].round(0).astype(int)
            tbl_wf.columns     = ["Month","Warehouse","Forecast Units","Upper Bound"]
            st.dataframe(tbl_wf.sort_values(["Month","Warehouse"]), use_container_width=True, hide_index=True)

        sec("Top Products per Warehouse")
        wsel = st.selectbox("Warehouse", sorted(del_df["Warehouse"].unique()))
        tp = del_df[del_df["Warehouse"]==wsel].groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(10)
        fig_tp = go.Figure(go.Bar(x=tp.values, y=tp.index, orientation="h",
            marker=dict(color="#2ed8c3", line=dict(color="rgba(0,0,0,0)")),
            text=tp.values, textposition="outside", textfont=dict(color="#4a5e7a")))
        fig_tp.update_layout(**CD(), height=300, xaxis=gX(), yaxis=dict(showgrid=False,color="#8a9dc0"))
        st.plotly_chart(fig_tp, use_container_width=True)

    with t5:
        sec("Region Performance Overview")
        rs = del_df.groupby("Region").agg(
            Orders=("Order_ID","count"), Revenue=("Net_Revenue","sum"),
            Qty=("Quantity","sum"), Avg_Del=("Delivery_Days","mean"),
            Returns=("Return_Flag","mean")).reset_index().sort_values("Revenue",ascending=False)

        met = st.selectbox("Metric", ["Revenue","Orders","Qty","Avg_Del","Returns"])
        fig_r = go.Figure(go.Bar(x=rs["Region"], y=rs[met],
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(rs))], line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>"))
        fig_r.update_layout(**CD(), height=290, xaxis={**gX(),"tickangle":-25}, yaxis=gY())
        st.plotly_chart(fig_r, use_container_width=True)

        cl5, cr5 = st.columns(2, gap="large")
        with cl5:
            sec("Best Carrier per Region")
            bc = (del_df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
                  .mean().reset_index().sort_values("Delivery_Days").groupby("Region").first().reset_index())
            bc.columns = ["Region","Best Carrier","Avg Days"]
            bc["Avg Days"] = bc["Avg Days"].round(1)
            st.dataframe(bc, use_container_width=True, hide_index=True)
        with cr5:
            sec("Region Return Rate Ranking")
            rr = del_df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False)*100
            fig_ret = go.Figure(go.Bar(x=rr.values, y=rr.index, orientation="h",
                marker=dict(color=["#ff6b6b" if v>20 else "#f5a623" if v>12 else "#56e0a0" for v in rr.values], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in rr.values], textposition="outside", textfont=dict(color="#4a5e7a")))
            fig_ret.update_layout(**CD(), height=270, xaxis=gX(), yaxis=dict(showgrid=False,color="#8a9dc0"))
            st.plotly_chart(fig_ret, use_container_width=True)

        sec("Region Revenue Forecast")
        top_reg = del_df["Region"].value_counts().head(5).index.tolist()
        fig_rf  = go.Figure()
        for i, reg in enumerate(top_reg):
            s = del_df[del_df["Region"]==reg].groupby(del_df["Order_Date"].dt.to_period("M"))["Net_Revenue"].sum().rename("v")
            r = ml_forecast(s.values.astype(float), s.index, 6)
            if r is None: continue
            fig_rf.add_trace(go.Scatter(x=r["hist_ds"], y=r["hist_y"], name=reg,
                line=dict(color=COLORS[i],width=1.5,dash="solid"), opacity=0.25, showlegend=False))
            fig_rf.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=reg,
                mode="lines+markers", line=dict(color=COLORS[i],width=2.5,dash="dot"),
                marker=dict(size=8,line=dict(color="#080e1a",width=2))))
        fig_rf.update_layout(**CD(), height=270, xaxis=gX(), yaxis=gY(), legend=leg())
        st.plotly_chart(fig_rf, use_container_width=True)

# ─── SIDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("""<div style='padding:18px 0 26px'>
  <div style='font-family:DM Mono,monospace;font-size:0.58rem;letter-spacing:0.16em;
       text-transform:uppercase;color:#4a5e7a;margin-bottom:5px'>Supply Chain Platform</div>
  <div style='font-family:Outfit,sans-serif;font-size:1.75rem;font-weight:900;
       letter-spacing:-0.04em;background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow</div>
  <div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#4a5e7a;
       margin-top:2px;letter-spacing:0.05em'>D2D INTELLIGENCE</div>
</div>""", unsafe_allow_html=True)

PAGES = {
    "Overview":               page_overview,
    "Demand Forecasting":     page_demand,
    "Inventory Optimisation": page_inventory,
    "Production Planning":    page_production,
    "Logistics Optimisation": page_logistics,
    "Decision Chatbot":       page_chatbot,
}

sel = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("""<div style='border-top:1px solid rgba(255,255,255,0.06);padding-top:18px;margin-top:8px'>
  <div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#4a5e7a;line-height:2.1;letter-spacing:0.04em'>
    <span style='color:#8a9dc0'>DATA RANGE</span><br>Jan 2024 – Dec 2025<br>
    <span style='color:#8a9dc0'>DATASET</span><br>5,200 orders · 50 SKUs · 🇮🇳 India D2D<br>
    <span style='color:#8a9dc0'>MODEL</span><br>Ridge + Structural Break<br>
    <span style='color:#8a9dc0'>CHATBOT</span><br>Groq LLaMA-3.3-70B<br>
  </div>
  <div style='margin-top:14px;font-family:DM Mono,monospace;font-size:0.6rem;color:#4a5e7a'>
    <span style='color:#f5a623'>PIPELINE</span><br>
    Demand → Inventory → Production → Logistics → Chatbot
  </div>
</div>""", unsafe_allow_html=True)

PAGES[sel]()
