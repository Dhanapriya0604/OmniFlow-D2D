import streamlit as st
st.set_page_config(
    page_title="OmniFlow D2D Intelligence",page_icon="⬡",
    layout="wide",initial_sidebar_state="expanded"
)
def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg: #f8fafc;--text: #0f172a;--muted: #475569;--primary: #1e3a8a;
        --border: #e5e7eb;--accent: #e0e7ff;--panel:#ffffff;
    }
    html, body {
        background-color: var(--bg);color: var(--text);font-family: Inter, system-ui;
    }
    section.main > div {
        animation: fadeIn 0.4s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section-title {
        font-size: 28px;font-weight: 800;margin: 28px 0 14px 0;
    }
    .card {
        background: white;padding: 22px;border-radius: 16px;border: 1px solid var(--border);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);transition: all 0.25s ease;
    }
    .card:hover {
        transform: translateY(-4px);box-shadow: 0 18px 40px rgba(0,0,0,0.14);
    }
    .metric-card {
        background: linear-gradient(180deg, #eef4ff, #ffffff);padding: 18px;
        text-align: center;border-radius: 16px;
        box-shadow: 0 6px 18px rgba(30,58,138,0.18);transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px);box-shadow: 0 16px 36px rgba(30,58,138,0.28);
    }
    .metric-label {
        font-size: 14px;color: var(--muted);
    }
    .metric-value {
        font-size: 30px;font-weight: 900;color: var(--primary);
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;border-radius: 12px;padding: 10px 18px;
        font-weight: 600;color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent);color: var(--primary);
        box-shadow: 0 6px 18px rgba(30,58,138,0.25);
    }
    .page-title{
        font-size:34px;font-weight:900;margin-bottom:4px;
    }   
    .page-subtitle{
        font-size:14px;color:#64748b;margin-bottom:18px;
    }       
    .section-header{
        margin-top:20px;margin-bottom:8px;
    }   
    .section-header-line{
        height:2px;background:linear-gradient(90deg,#e5e7eb,transparent);
        margin-top:6px;
    }    
    .metric-sub{
        font-size:11px;color:#64748b;margin-top:4px;
    }   
    .badge{
        display:inline-block;padding:6px 10px;font-size:11px;
        font-weight:600;border-radius:8px;margin-right:6px;
    } 
    .badge-amber{background:#fff7ed;color:#ea580c}
    .badge-teal{background:#ecfeff;color:#0891b2}
    .badge-lav{background:#f5f3ff;color:#7c3aed}
    .badge-coral{background:#fff1f2;color:#e11d48}
    .badge-sky{background:#eff6ff;color:#2563eb}
    .badge-mint{background:#ecfdf5;color:#059669}
    .badge-purple{background:#faf5ff;color:#7c3aed}
    .info-banner{
        border-radius:12px;padding:14px 16px;margin:10px 0;font-size:13px;
    }
    .banner-teal{
        background:#f0fdfa;border:1px solid #5eead4;
    } 
    .banner-amber{
        background:#fffbeb;border:1px solid #fbbf24;
    }
    .banner-coral{
        background:#fff1f2;border:1px solid #fb7185;
    }
    .banner-mint{
       background:#ecfdf5;border:1px solid #34d399;
    }
    .banner-purple{
       background:#faf5ff;border:1px solid #a78bfa;
    }
    .chat-user-bubble{
        background:#1e3a8a;color:white;padding:10px 14px;
        border-radius:14px;max-width:70%;margin-left:auto;
    }
    .chat-ai-bubble{
        background:#f1f5f9;padding:12px 14px;border-radius:14px;max-width:80%;
    }
    .alert-item{
        border-radius:10px;padding:10px 12px;
        margin-bottom:8px;border:1px solid #e5e7eb;
    }
    .alert-critical{
        background:#fef2f2;
    } 
    .alert-warn{
        background:#fff7ed;
    }  
    .model-quality-card{
        background:white;border-radius:16px;padding:18px;
        border:1px solid #e5e7eb;box-shadow:0 6px 20px rgba(0,0,0,0.08);
    } 
    .ensemble-card{
        background:linear-gradient(135deg,#f8faff,#ffffff);border-radius:16px;padding:18px;
        border:1px solid #c7d7fd;box-shadow:0 6px 20px rgba(30,58,138,0.10);
        margin-bottom:14px;
    }
    .model-pill{
        display:inline-block;padding:4px 10px;font-size:11px;font-weight:700;
        border-radius:20px;margin-right:6px;margin-bottom:4px;
    }
    .pill-ridge{background:#eff6ff;color:#1d4ed8}
    .pill-rf{background:#f0fdf4;color:#15803d}
    .pill-gb{background:#fef9c3;color:#a16207}
    .pill-ensemble{background:#fdf4ff;color:#7e22ce}
    .about-card{
        background:white;border:1px solid #e5e7eb;border-radius:16px;
        padding:18px;margin-bottom:20px;box-shadow:0 6px 20px rgba(0,0,0,0.08);
    }
    .chat-card{
        background:#f1f5f9;border:1px solid #e5e7eb;border-radius:12px;
        padding:12px;font-size:13px;transition:all .2s;
    }   
    .chat-card:hover{
        background:#e0e7ff;transform:translateY(-2px);
    }
    .block-container{
        padding-top:2rem;padding-bottom:2rem;
    }
    .js-plotly-plot .plotly .main-svg{
        overflow:visible !important;
    }  
    .js-plotly-plot{
        overflow:visible !important;
    }
    </style>
    """, unsafe_allow_html=True)
inject_css()
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import requests as _requests

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_ecommerce_orders.csv")

COLORS   = ["#1565C0","#2E7D32","#E65100","#C62828","#6A1B9A","#00695C"]
COLORS_S = ["#1976D2","#388E3C","#F57C00","#D32F2F","#7B1FA2","#00796B"]

MODEL_COLORS = {
    "Ridge":    "#3B82F6",
    "RandomForest": "#22C55E",
    "GradBoost":    "#F59E0B",
    "Ensemble": "#8B5CF6",
}

def CD():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", family="Inter, sans-serif", size=11),
        margin=dict(l=30,r=50,t=40,b=30)
    )

def gY(): return dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False, tickcolor="#333333")
def gX(): return dict(showgrid=False, zeroline=False, tickcolor="#333333")
def leg(): return dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#E0E0E0", borderwidth=1, font=dict(color="#333333",size=10))

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

def model_quality_verdict(nrmse, r2):
    accuracy_pct = max(0, round((1 - nrmse) * 100, 1))
    if nrmse < 0.10 and r2 >= 0.95:
        grade, label, explanation, icon = "A+", "Excellent", "Ensemble captures demand patterns with very high precision. Safe to rely on forecasts for procurement and production decisions.", "✅"
    elif nrmse < 0.15 and r2 >= 0.90:
        grade, label, explanation, icon = "A", "Very Good", "Strong fit with low error. Forecasts are reliable for 1–3 month planning horizons.", "✅"
    elif nrmse < 0.20 and r2 >= 0.85:
        grade, label, explanation, icon = "B+", "Good", "Model performs well. Minor variance in hold-out period — use confidence intervals for safety stock calculations.", "🟦"
    elif nrmse < 0.25 and r2 >= 0.75:
        grade, label, explanation, icon = "B", "Acceptable", "Moderate fit. Suitable for directional planning. Add extra safety buffer (10–15%) to production targets.", "⚠️"
    elif nrmse < 0.35 and r2 >= 0.60:
        grade, label, explanation, icon = "C", "Weak", "High variance in predictions. Treat forecasts as indicative only. Consider more data or feature engineering.", "⚠️"
    else:
        grade, label, explanation, icon = "D", "Poor", "Model struggles to capture the demand pattern. Do NOT use for procurement without manual override.", "🔴"
    return grade, label, explanation, accuracy_pct, icon

def render_model_quality(res):
    grade, label, explanation, accuracy_pct, icon = model_quality_verdict(res["nrmse"], res["r2"])
    
    if "model_metrics" in res:
        st.markdown("<div class='ensemble-card'>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.75rem;font-weight:700;color:#4a5e7a;
            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px'>
            🤖 Individual Model Performance (Hold-out Evaluation)</div>""", unsafe_allow_html=True)
        mm = res["model_metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (mname, pill_cls, clr) in zip(
            [mc1, mc2, mc3, mc4],
            [("Ridge","pill-ridge","#3B82F6"),("RandomForest","pill-rf","#22C55E"),
             ("GradBoost","pill-gb","#F59E0B"),("Ensemble","pill-ensemble","#8B5CF6")]
        ):
            if mname in mm:
                m = mm[mname]
                col.markdown(f"""<div style='text-align:center;padding:10px;border-radius:12px;
                    border:1px solid #e5e7eb;background:white'>
                    <div class='model-pill {pill_cls}'>{mname}</div>
                    <div style='font-size:0.7rem;color:#64748b;margin-top:6px'>RMSE</div>
                    <div style='font-size:1.1rem;font-weight:800;color:{clr}'>{m["rmse"]:.1f}</div>
                    <div style='font-size:0.68rem;color:#94a3b8'>NRMSE {m["nrmse"]*100:.1f}% · R² {m["r2"]:.3f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        weights = res.get("weights", {})
        if weights:
            total = sum(weights.values())
            st.markdown(f"""<div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:10px;
                padding:10px 14px;font-size:0.75rem;margin:8px 0'>
                <b style='color:#1e3a8a'>Ensemble Weights (inverse-RMSE blending):</b>
                <span class='model-pill pill-ridge'>Ridge {weights.get("Ridge",0)/total*100:.0f}%</span>
                <span class='model-pill pill-rf'>RF {weights.get("RandomForest",0)/total*100:.0f}%</span>
                <span class='model-pill pill-gb'>GB {weights.get("GradBoost",0)/total*100:.0f}%</span>
            </div>""", unsafe_allow_html=True)
        sp(0.5)

    m1, m2, m3, m4, m5 = st.columns(5)
    kpi(m1, "RMSE",      f"{res['rmse']:.1f}",        "sky", "ensemble hold-out")
    kpi(m2, "NRMSE",     f"{res['nrmse']*100:.1f}%",  "sky", "normalised RMSE")
    kpi(m3, "MAE",       f"{res['mae']:.1f}",         "sky", "mean abs error")
    kpi(m4, "R² Score",  f"{res['r2']:.3f}",          "sky", "model fit (1=perfect)")
    kpi(m5, "Accuracy",  f"{accuracy_pct:.1f}%",      "sky", "1 − NRMSE")
    sp(0.5)

    st.markdown(f"""
    <div class='model-quality-card'>
      <div style='display:flex;align-items:center;gap:14px;margin-bottom:12px'>
        <div style='font-size:1.6rem'>{icon}</div>
        <div>
          <div style='font-size:0.7rem;text-transform:uppercase;
               letter-spacing:0.12em;color:#64748b;margin-bottom:4px'>
               Ensemble Quality Grade
          </div>
          <div style='font-size:1.5rem;font-weight:900'>
            {grade} <span style='font-size:0.9rem;font-weight:600;color:#334155'>
            {label}</span>
          </div>
        </div>
        <div style='margin-left:auto;text-align:right'>
          <div style='font-size:0.7rem;color:#64748b'>Forecast Accuracy</div>
          <div style='font-size:2rem;font-weight:900'>{accuracy_pct:.1f}%</div>
        </div>
      </div>
      <div style='font-size:0.85rem;color:#334155;
           border-top:1px solid rgba(0,0,0,0.06);
           padding-top:10px;line-height:1.6'>
        📋 <b>Interpretation:</b> {explanation}
      </div>
    </div>
    """, unsafe_allow_html=True)
    sp(0.5)

@st.cache_data(show_spinner="Loading & cleaning data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df["Region"]      = df["Region"].replace("Pune", "Maharashtra")
    df["YearMonth"]   = df["Order_Date"].dt.to_period("M")
    df["Year"]        = df["Order_Date"].dt.year
    df["Month_Num"]   = df["Order_Date"].dt.month
    df["Net_Revenue"] = np.where(df["Return_Flag"] == 1, 0.0, df["Revenue_INR"])
    df["Net_Qty"]     = np.where(df["Return_Flag"] == 1, 0, df["Quantity"])
    df.loc[df["Order_Status"] == "Cancelled", "Delivery_Days"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def get_ops(df):
    return df[df["Order_Status"].isin(["Delivered", "Shipped"])].copy()

@st.cache_data(show_spinner=False)
def get_delivered(df):
    return df[df["Order_Status"] == "Delivered"].copy()

def _to_timestamp_index(idx):
    if hasattr(idx, 'to_timestamp'):
        return idx.to_timestamp()
    return pd.DatetimeIndex(idx)

def build_features(n_hist, n_future, ds_hist, regime_start_idx):
    all_t       = np.arange(n_hist + n_future)
    ts          = _to_timestamp_index(ds_hist)
    hist_months = ts.month.values
    last_month  = int(hist_months[-1])
    fut_months  = np.array([(last_month + i - 1) % 12 + 1 for i in range(1, n_future + 1)])
    mn          = np.concatenate([hist_months, fut_months])
    regime      = (all_t >= regime_start_idx).astype(float)
    quarter = np.where(mn <= 3, 1, np.where(mn <= 6, 2, np.where(mn <= 9, 3, 4)))
    q1 = (quarter == 1).astype(float)
    q2 = (quarter == 2).astype(float)
    q3 = (quarter == 3).astype(float)
    X = np.column_stack([
        all_t, all_t ** 2,
        np.sin(2 * np.pi * mn / 12), np.cos(2 * np.pi * mn / 12),
        np.sin(4 * np.pi * mn / 12), np.cos(4 * np.pi * mn / 12),
        np.sin(6 * np.pi * mn / 12), np.cos(6 * np.pi * mn / 12),
        regime, all_t * regime,
        q1, q2, q3, np.log1p(all_t),
    ])
    return X

def _fit_predict_model(model, Xtr, ytr, Xte, Xfull, X_fut, sc):
    """Fit a single model, return eval_pred, fitted, forecast."""
    Xtr_s  = sc.transform(Xtr)
    Xte_s  = sc.transform(Xte)
    Xall_s = sc.transform(Xfull)
    Xfut_s = sc.transform(X_fut)
    model.fit(Xtr_s, ytr)
    eval_pred = np.maximum(model.predict(Xte_s), 0)
    fitted    = np.maximum(model.predict(Xall_s), 0)
    forecast  = np.maximum(model.predict(Xfut_s), 0)
    return eval_pred, fitted, forecast

def _detect_regime(series_values, min_idx=6):
    """Find the index where a structural break (growth acceleration) most likely occurred.
    Uses rolling mean comparison — picks the point with maximum ratio of
    post-mean to pre-mean, constrained so both windows have at least min_idx points."""
    n = len(series_values)
    best_idx, best_ratio = min_idx, 1.0
    for i in range(min_idx, n - min_idx):
        pre  = series_values[:i].mean()
        post = series_values[i:].mean()
        ratio = post / (pre + 1e-9)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx   = i
    return best_idx

def ml_forecast(series_values, ds_index, n_future=6):
    n = len(series_values)
    if n < 6:
        return None

    regime_idx = _detect_regime(series_values)
    X_all  = build_features(n, n_future, ds_index, regime_idx)
    X_hist = X_all[:n]
    X_fut  = X_all[n:]

    models = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=3,
                                              min_samples_leaf=4, random_state=42),
        "GradBoost":    GradientBoostingRegressor(n_estimators=80, max_depth=2,
                                                  learning_rate=0.08, subsample=0.9,
                                                  random_state=42),
    }

    eval_preds  = {}
    model_rmses = {}
    model_metrics = {}
    h = 4

    n_folds   = min(3, n // 6)
    fold_size = 2
    fold_rmses = {m: [] for m in models}
    for fold in range(n_folds):
        te_end   = n - fold * fold_size
        te_start = te_end - fold_size
        if te_start < 6:
            break
        Xtr_f = X_hist[:te_start]; ytr_f = series_values[:te_start]
        Xte_f = X_hist[te_start:te_end]; yte_f = series_values[te_start:te_end]
        sc_f  = StandardScaler(); sc_f.fit(Xtr_f)
        for mname, mdl_cls in [
            ("Ridge",        Ridge(alpha=1.0)),
            ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=42)),
            ("GradBoost",    GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.08, subsample=0.9, random_state=42)),
        ]:
            mdl_cls.fit(sc_f.transform(Xtr_f), ytr_f)
            ep_f = np.maximum(mdl_cls.predict(sc_f.transform(Xte_f)), 0)
            fold_rmses[mname].append(np.sqrt(mean_squared_error(yte_f, ep_f)))

    for mname in models:
        avg_rmse = np.mean(fold_rmses[mname]) if fold_rmses[mname] else 1.0
        nrmse_v  = avg_rmse / np.mean(series_values) if np.mean(series_values) > 0 else 0
        ss_rat   = max(0, 1 - (avg_rmse**2 / (np.var(series_values) + 1e-9)))
        model_rmses[mname]    = avg_rmse
        model_metrics[mname]  = {"rmse": avg_rmse, "nrmse": nrmse_v, "mae": avg_rmse * 0.8, "r2": ss_rat}

    Xtr, Xte = X_hist[:-h], X_hist[-h:]
    ytr, yte = series_values[:-h], series_values[-h:]
    sc_h = StandardScaler(); sc_h.fit(Xtr)
    for mname, mdl in models.items():
        ep, _, _ = _fit_predict_model(mdl, Xtr, ytr, Xte, X_hist, X_fut, sc_h)
        eval_preds[mname] = ep

    inv_rmse  = {m: 1.0 / (r + 1e-9) for m, r in model_rmses.items()}
    total_inv = sum(inv_rmse.values())
    weights   = {m: v / total_inv for m, v in inv_rmse.items()}

    ypred_eval = sum(weights[m] * eval_preds[m] for m in models)

    sc2 = StandardScaler()
    sc2.fit(X_hist)
    fitted_per_model   = {}
    forecast_per_model = {}

    for mname, mdl in models.items():
        _, fitted, forecast = _fit_predict_model(mdl, X_hist, series_values, Xte, X_hist, X_fut, sc2)
        fitted_per_model[mname]   = fitted
        forecast_per_model[mname] = forecast

    ensemble_fitted   = sum(weights[m] * fitted_per_model[m]   for m in models)
    ensemble_forecast = sum(weights[m] * forecast_per_model[m] for m in models)

    residuals = series_values - ensemble_fitted
    resid_std = residuals.std()
    ss_res_e  = np.sum(residuals ** 2)
    ss_tot_e  = np.sum((series_values - np.mean(series_values)) ** 2)
    r2_e      = 1 - ss_res_e / ss_tot_e if ss_tot_e > 0 else 0
    rmse_e    = np.sqrt(mean_squared_error(yte, ypred_eval))
    nrmse_e   = rmse_e / np.mean(yte) if np.mean(yte) > 0 else 0
    mae_e     = mean_absolute_error(yte, ypred_eval)

    model_metrics["Ensemble"] = {"rmse": rmse_e, "nrmse": nrmse_e, "mae": mae_e, "r2": r2_e}

    ts_index  = _to_timestamp_index(ds_index)
    last_dt   = ts_index[-1]
    fut_dates = pd.date_range(last_dt + pd.offsets.MonthBegin(1), periods=n_future, freq="MS")

    log_resid_std = np.log1p(resid_std / (np.mean(series_values) + 1e-9))
    ci_lo = np.maximum(ensemble_forecast * np.exp(-1.645 * log_resid_std * np.sqrt(np.arange(1, n_future+1))), 0)
    ci_hi = ensemble_forecast * np.exp( 1.645 * log_resid_std * np.sqrt(np.arange(1, n_future+1)))

    return {
        "hist_ds":             ts_index,
        "hist_y":              series_values,
        "fitted":              ensemble_fitted,
        "fitted_per_model":    fitted_per_model,
        "forecast_per_model":  forecast_per_model,
        "fut_ds":              fut_dates,
        "forecast":            ensemble_forecast,
        "ci_lo":               ci_lo,
        "ci_hi":               ci_hi,
        "rmse":                rmse_e,
        "nrmse":               nrmse_e,
        "mae":                 mae_e,
        "r2":                  r2_e,
        "resid_std":           resid_std,
        "eval_actual":         yte,
        "eval_pred":           ypred_eval,
        "eval_ds":             ts_index[-h:],
        "model_metrics":       model_metrics,
        "weights":             {m: weights[m] for m in models},
    }

@st.cache_data(show_spinner=False)
def compute_inventory(order_cost=500, hold_pct=0.20, lead_time=7, z=1.65):
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    sku_monthly = (ops.groupby(["SKU_ID","YM"])["Net_Qty"]
                   .sum().reset_index().sort_values(["SKU_ID","YM"]))
    sku_stats = (ops.groupby(["SKU_ID","Product_Name","Category"])
                 .agg(avg_price=("Sell_Price","mean"), total_qty=("Net_Qty","sum"))
                 .reset_index())

    # Use only delivered orders for lead time std (Shipped orders have no Delivery_Days)
    del_ops    = df[df["Order_Status"] == "Delivered"].copy()
    lt_std_map = (del_ops.groupby("Category")["Delivery_Days"]
                  .std().fillna(1.0).to_dict())

    rows = []
    for _, sk in sku_stats.iterrows():
        sku     = sk["SKU_ID"]
        skd     = sku_monthly[sku_monthly["SKU_ID"]==sku].sort_values("YM")
        demands = skd["Net_Qty"].values
        if len(demands) < 2:
            continue

        avg_d    = demands.mean()
        std_d    = demands.std() if len(demands) > 1 else avg_d * 0.2
        peak_d   = demands.max()
        econ_d   = (avg_d * 0.6 + peak_d * 0.4)   # blended avg+peak for EOQ
        daily_d  = avg_d / 30.0
        ann_d    = econ_d * 12
        uc       = max(sk["avg_price"], 1.0)

        eoq       = int(np.sqrt(2 * ann_d * order_cost / (uc * hold_pct))) if ann_d > 0 else 10
        eoq       = max(eoq, 1)

        daily_std  = std_d / np.sqrt(30)
        lt_std     = lt_std_map.get(sk["Category"], 1.0)  # lead time std in days
        ss = int(z * np.sqrt(lead_time * daily_std**2 + daily_d**2 * lt_std**2))
        ss = max(ss, 0)

        rop = int(daily_d * lead_time + ss)
        rop = max(rop, 1)

        stock   = rop + eoq
        pending = 0
        for demand in demands:
            stock   = max(stock - demand + pending, 0)
            pending = 0
            if stock < rop:
                n_orders  = max(1, int(np.ceil((rop + ss - stock) / eoq)))
                pending   = n_orders * eoq

        current_stock = max(stock, 0)

        if current_stock < ss:
            status = "🔴 Critical"
        elif current_stock < rop:
            status = "🟡 Low"
        else:
            status = "🟢 Adequate"

        margin_rate  = 0.20
        daily_margin = daily_d * uc * margin_rate
        days_exposed = max(lead_time - (current_stock / daily_d if daily_d > 0 else 0), 0)
        stockout_cost = round(daily_margin * days_exposed, 0) if status == "🔴 Critical" else 0

        rows.append({
            "SKU_ID": sku, "Product_Name": sk["Product_Name"],
            "Category": sk["Category"],
            "Monthly_Avg": round(avg_d, 1),  "Monthly_Std": round(std_d, 1),
            "Daily_Std":   round(daily_std, 2),
            "EOQ": eoq, "SS": ss, "ROP": rop,
            "Current_Stock": current_stock, "Status": status,
            "Unit_Price": round(uc, 0),    "Annual_Demand": round(ann_d, 0),
            "Forecast_6M": int(avg_d * 6 * 1.05),
            "Stockout_Cost_Day": stockout_cost,
            "Total_Revenue": round(sk["total_qty"] * uc, 0),
        })

    inv_df = pd.DataFrame(rows)
    if inv_df.empty:
        return inv_df

    inv_df = inv_df.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
    inv_df["Rev_Cum_Pct"] = inv_df["Total_Revenue"].cumsum() / inv_df["Total_Revenue"].sum() * 100
    inv_df["ABC"] = np.where(inv_df["Rev_Cum_Pct"] <= 70, "A",
                    np.where(inv_df["Rev_Cum_Pct"] <= 90, "B", "C"))
    return inv_df

@st.cache_data(show_spinner=False)
def compute_production(cap_mult=1.0, buffer_pct=0.15):
    df  = load_data()
    ops = get_ops(df)
    inv = compute_inventory()
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    cat_monthly = ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
    ds_index    = cat_monthly.index

    rows = []
    for cat in cat_monthly.columns:
        vals = cat_monthly[cat].values.astype(float)
        res  = ml_forecast(vals, ds_index)
        if res is None:
            continue

        crit_total = float(inv[(inv["Category"]==cat)&(inv["Status"]=="🔴 Critical")]["Monthly_Avg"].sum())
        low_total  = float(inv[(inv["Category"]==cat)&(inv["Status"]=="🟡 Low")]["Monthly_Avg"].sum())

        boost_schedule = {0: 1.0, 1: 0.25}

        for i, (dt, fc) in enumerate(zip(res["fut_ds"], res["forecast"])):
            boost_factor = boost_schedule.get(i, 0.0)
            crit_boost   = crit_total * 0.5 * boost_factor
            low_boost    = low_total  * 0.25 * boost_factor
            net_prod     = max(fc + crit_boost + low_boost, 0) * cap_mult
            prod         = net_prod * (1 + buffer_pct)
            rows.append({
                "Month_dt": dt, "Month": dt.strftime("%b %Y"),
                "Category": cat,
                "Demand_Forecast": round(fc, 0),
                "Crit_Boost":      round(crit_boost, 0),
                "Low_Boost":       round(low_boost, 0),
                "Buffer":          round(prod - net_prod, 0),
                "Production":      round(prod, 0),
                "CI_Lo": round(res["ci_lo"][i], 0),
                "CI_Hi": round(res["ci_hi"][i], 0),
            })

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_logistics(w_speed=0.40, w_cost=0.35, w_returns=0.25):
    df     = load_data()
    del_df = get_delivered(df)
    plan   = compute_production()   # feed from production module

    carr = del_df.groupby("Courier_Partner").agg(
        Orders=("Order_ID","count"), Avg_Days=("Delivery_Days","mean"),
        Avg_Cost=("Shipping_Cost_INR","mean"), Total_Cost=("Shipping_Cost_INR","sum"),
        Return_Rate=("Return_Flag","mean"),
    ).reset_index()

    carr["Norm_Days"]    = 1 - (carr["Avg_Days"]    - carr["Avg_Days"].min())    / (carr["Avg_Days"].max()    - carr["Avg_Days"].min()    + 1e-9)
    carr["Norm_Cost"]    = 1 - (carr["Avg_Cost"]    - carr["Avg_Cost"].min())    / (carr["Avg_Cost"].max()    - carr["Avg_Cost"].min()    + 1e-9)
    carr["Norm_Returns"] = 1 - (carr["Return_Rate"] - carr["Return_Rate"].min()) / (carr["Return_Rate"].max() - carr["Return_Rate"].min() + 1e-9)
    carr["Perf_Score"]   = (w_speed*carr["Norm_Days"] + w_cost*carr["Norm_Cost"] + w_returns*carr["Norm_Returns"]).round(3)
    carr["Delay_Index"]  = (carr["Avg_Days"] / carr["Avg_Days"].min()).round(2)
    carr["Cost_Score"]   = (carr["Avg_Cost"] / carr["Avg_Cost"].min()).round(2)

    region_carr_score = (
        del_df.groupby(["Region","Courier_Partner"]).agg(
            Avg_Days=("Delivery_Days","mean"),
            Avg_Cost=("Shipping_Cost_INR","mean"),
            Return_Rate=("Return_Flag","mean"),
            Orders=("Order_ID","count"),
        ).reset_index()
    )
    for col, wt in [("Avg_Days", w_speed), ("Avg_Cost", w_cost), ("Return_Rate", w_returns)]:
        mn = region_carr_score[col].min(); mx = region_carr_score[col].max()
        region_carr_score[f"Norm_{col}"] = 1 - (region_carr_score[col] - mn) / (mx - mn + 1e-9)
    region_carr_score["Score"] = (
        w_speed   * region_carr_score["Norm_Avg_Days"] +
        w_cost    * region_carr_score["Norm_Avg_Cost"] +
        w_returns * region_carr_score["Norm_Return_Rate"]
    )
    best = (region_carr_score.sort_values("Score", ascending=False)
            .groupby("Region").first().reset_index()
            [["Region","Courier_Partner","Avg_Days","Avg_Cost","Score"]])

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
    opt["Potential_Saving"] = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) * opt["Orders"]).round(0)
    opt["Saving_Pct"]       = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) / opt["Current_Avg_Cost"] * 100).round(1)

    # --- Production-driven forward shipment plan ---
    # avg shipping cost per unit from historical data
    avg_ship_cost_per_unit = (del_df["Shipping_Cost_INR"].sum() /
                               del_df["Quantity"].replace(0, np.nan).sum())
    avg_ship_cost_per_unit = max(avg_ship_cost_per_unit, 1.0)

    # avg units per order (for order volume projection)
    avg_units_per_order = max(del_df["Quantity"].mean(), 1.0)

    fwd_rows = []
    if not plan.empty:
        for _, row in plan.iterrows():
            proj_units  = row["Production"]
            proj_orders = round(proj_units / avg_units_per_order)
            proj_cost   = round(proj_units * avg_ship_cost_per_unit, 0)
            fwd_rows.append({
                "Month_dt":    row["Month_dt"],
                "Month":       row["Month"],
                "Category":    row["Category"],
                "Prod_Units":  int(proj_units),
                "Proj_Orders": int(proj_orders),
                "Proj_Ship_Cost": int(proj_cost),
                "CI_Lo_Units": int(row["CI_Lo"]),
                "CI_Hi_Units": int(row["CI_Hi"]),
            })
    fwd_plan = pd.DataFrame(fwd_rows)

    return carr, best, opt, current, fwd_plan

def build_context():
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    m_orders = ops.groupby("YM")["Order_ID"].count().rename("v")
    m_qty    = ops.groupby("YM")["Net_Qty"].sum().rename("v")
    m_rev    = ops.groupby("YM")["Net_Revenue"].sum().rename("v")

    r_ord = ml_forecast(m_orders.values.astype(float), m_orders.index, 6)
    r_rev = ml_forecast(m_rev.values.astype(float),    m_rev.index,    6)
    r_qty = ml_forecast(m_qty.values.astype(float),    m_qty.index,    6)

    def fc_str(r, fmt):
        if r is None: return "N/A"
        return "; ".join([f"{d.strftime('%b%Y')}:{fmt(v)}" for d,v in zip(r["fut_ds"], r["forecast"])])

    inv  = compute_inventory()
    plan = compute_production()
    carr, best_carr, opt, _, fwd_plan = compute_logistics()

    n_crit     = (inv["Status"]=="🔴 Critical").sum()
    n_low      = (inv["Status"]=="🟡 Low").sum()
    crit_prods = ", ".join(inv[inv["Status"]=="🔴 Critical"]["Product_Name"].head(5).tolist())
    total_stockout_cost = inv["Stockout_Cost_Day"].sum()
    abc_counts = inv["ABC"].value_counts().to_dict() if "ABC" in inv.columns else {}
    abc_str    = ", ".join([f"{k}:{v} SKUs" for k,v in sorted(abc_counts.items())])

    prod_sum   = plan.groupby("Category")["Production"].sum().to_dict() if not plan.empty else {}
    prod_str   = ", ".join([f"{k}:{v:.0f}u" for k,v in prod_sum.items()])
    peak_mo    = plan.groupby("Month_dt")["Production"].sum().idxmax().strftime("%b %Y") if not plan.empty else "N/A"
    carr_str   = "; ".join([f"{r['Courier_Partner']}: {r['Orders']}ord, {r['Avg_Days']:.1f}d, ₹{r['Avg_Cost']:.0f}/ship, score:{r['Perf_Score']:.3f}" for _,r in carr.iterrows()])
    best_str   = ", ".join([f"{r['Region']}→{r['Courier_Partner']}(score:{r['Score']:.2f})" for _,r in best_carr.iterrows()])
    saving_total = opt["Potential_Saving"].sum()
    saving_str   = "; ".join([f"{r['Region']}: save ₹{r['Potential_Saving']:,.0f} with {r['Optimal_Carrier']}" for _,r in opt.iterrows() if r['Potential_Saving']>0])
    fwd_str = ""
    if not fwd_plan.empty:
        fwd_agg = fwd_plan.groupby("Month").agg(Units=("Prod_Units","sum"), Cost=("Proj_Ship_Cost","sum")).reset_index()
        fwd_str = "; ".join([f"{r['Month']}:{r['Units']:.0f}u/₹{r['Cost']:,.0f}" for _,r in fwd_agg.iterrows()])
    cat_rev  = ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
    cat_str  = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in cat_rev.items()])
    top_reg  = ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False).head(5)
    reg_str  = ", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in top_reg.items()])
    top_sku  = ops.groupby("Product_Name")["Net_Revenue"].sum().sort_values(ascending=False).head(8)
    sku_str  = ", ".join(top_sku.index.tolist())

    if r_ord:
        mm = r_ord.get("model_metrics", {})
        ens = mm.get("Ensemble", {}); ridge = mm.get("Ridge", {})
        rf  = mm.get("RandomForest", {}); gb  = mm.get("GradBoost", {})
        metric_str = (f"Ensemble RMSE:{ens.get('rmse',0):.1f}, NRMSE:{ens.get('nrmse',0)*100:.1f}%, R²:{ens.get('r2',0):.2f} | "
                      f"Ridge R²:{ridge.get('r2',0):.2f} | RF R²:{rf.get('r2',0):.2f} | GB R²:{gb.get('r2',0):.2f}")
    else:
        metric_str = ""

    return f"""=== OmniFlow D2D Intelligence ===
DATASET: 5,200 orders | Jan 2024–Dec 2025 | India D2D (Amazon, Flipkart, B2B)
SUMMARY: Net Revenue ₹{ops['Net_Revenue'].sum()/1e7:.2f}Cr | Active Orders {len(ops):,} |
         Return Rate {df[df['Order_Status']=='Returned'].shape[0]/len(ops)*100:.1f}% |
         Avg Delivery {ops['Delivery_Days'].mean():.1f}d

[MODULE 1 — DEMAND FORECAST (Ensemble: Ridge + RF + GradBoost)]
{metric_str}
Order Forecast: {fc_str(r_ord, lambda v: f"{v:.0f}")}
Qty Forecast:   {fc_str(r_qty, lambda v: f"{v:.0f}u")}
Revenue Forecast: {fc_str(r_rev, lambda v: f"₹{v/1e6:.1f}M")}

[MODULE 2 — INVENTORY (EOQ + Full SS + ROP + ABC | fed by Demand forecast)]
Critical: {n_crit} | Low: {n_low} | Adequate: {inv['Status'].eq('🟢 Adequate').sum()}
ABC: {abc_str}
Reorder IMMEDIATELY: {crit_prods}
Est. Stockout Loss/Day: ₹{total_stockout_cost:,.0f}

[MODULE 3 — PRODUCTION (6-month | fed by Demand + Inventory)]
By Category: {prod_str}
Peak Month: {peak_mo}

[MODULE 4 — LOGISTICS (fed by Production Plan | Score: Speed 40%, Cost 35%, Returns 25%)]
Carriers: {carr_str}
Best per Region: {best_str}
Cost Saving: ₹{saving_total:,.0f} | {saving_str}
Forward Shipment Plan (production-driven): {fwd_str if fwd_str else 'N/A'}

CATEGORIES: {cat_str}
TOP REGIONS: {reg_str}
TOP PRODUCTS: {sku_str}"""

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
    return f"""You are OmniFlow, an expert AI supply chain analyst for an India D2D e-commerce business.

=== YOUR EXPERTISE ===
• Demand forecasting — 3-Model Ensemble (Ridge + Random Forest + Gradient Boosting), inverse-RMSE weighted, asymmetric log-normal CI
• Inventory — Wilson EOQ (peak-blended demand), Full Safety Stock SS=z×√(LT×σ_d²+D²×σ_LT²), ROP, ABC classification
• Production — 6-month plan with decay-corrected replenishment boost (month 1 full, month 2 residual only)
• Logistics — weighted composite score (Speed 40%, Cost 35%, Returns 25%), configurable per analyst
• Indian e-commerce — Amazon.in, Flipkart, Shiprocket, INCREFF B2B

=== RESPONSE RULES ===
1. Lead with one precise, data-backed insight
2. Use bullet points (▸) with exact numbers (₹, %, units, days)
3. 4–8 bullets per answer
4. No generic closings
5. If not in context, say so

=== LIVE CONTEXT ===
{ctx}"""

def draw_ensemble_chart(res, chart_key, height=320, title="", show_models=True):
    """Draw forecast chart with optional per-model visibility."""
    fig = go.Figure()

    x_ci = list(res["fut_ds"]) + list(res["fut_ds"])[::-1]
    y_ci = list(res["ci_hi"])  + list(res["ci_lo"])[::-1]
    fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
        fillcolor="rgba(139,92,246,0.07)", line=dict(color="rgba(0,0,0,0)"),
        name="90% CI", showlegend=True))

    fig.add_trace(go.Scatter(x=res["hist_ds"], y=res["hist_y"], name="Actual",
        line=dict(color="#4a5e7a", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>"))

    if show_models and "fitted_per_model" in res:
        model_styles = [
            ("Ridge",        MODEL_COLORS["Ridge"],       "dot"),
            ("RandomForest", MODEL_COLORS["RandomForest"],"dashdot"),
            ("GradBoost",    MODEL_COLORS["GradBoost"],   "longdash"),
        ]
        for mname, clr, dash in model_styles:
            if mname in res["fitted_per_model"]:
                fig.add_trace(go.Scatter(
                    x=res["hist_ds"], y=res["fitted_per_model"][mname],
                    name=f"{mname} fit", line=dict(color=clr, width=1.2, dash=dash),
                    opacity=0.55, visible="legendonly",
                    hovertemplate=f"<b>{mname}</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}}<extra></extra>"))

    fig.add_trace(go.Scatter(x=res["hist_ds"], y=res["fitted"], name="Ensemble fit",
        line=dict(color=MODEL_COLORS["Ensemble"], width=1.5, dash="dot"), opacity=0.6))

    if show_models and "forecast_per_model" in res:
        model_styles = [
            ("Ridge",        MODEL_COLORS["Ridge"],       "dot"),
            ("RandomForest", MODEL_COLORS["RandomForest"],"dashdot"),
            ("GradBoost",    MODEL_COLORS["GradBoost"],   "longdash"),
        ]
        for mname, clr, dash in model_styles:
            if mname in res["forecast_per_model"]:
                fig.add_trace(go.Scatter(
                    x=res["fut_ds"], y=res["forecast_per_model"][mname],
                    name=f"{mname} fc", line=dict(color=clr, width=1.8, dash=dash),
                    mode="lines+markers", marker=dict(size=5, color=clr),
                    visible="legendonly",
                    hovertemplate=f"<b>{mname} Forecast</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}}<extra></extra>"))

    fig.add_trace(go.Scatter(x=res["fut_ds"], y=res["forecast"], name="Ensemble Forecast",
        line=dict(color=MODEL_COLORS["Ensemble"], width=2.8, dash="dot"),
        mode="lines+markers",
        marker=dict(size=8, color=MODEL_COLORS["Ensemble"], line=dict(color="#FFFFFF", width=2)),
        hovertemplate="<b>Ensemble Forecast</b><br>%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"))

    fig.add_trace(go.Scatter(x=res["eval_ds"], y=res["eval_pred"], name="Eval (ensemble)",
        mode="markers",
        marker=dict(size=10, color="#EF4444", symbol="x", line=dict(color="#FFFFFF", width=2))))

    fig.update_layout(**CD(), height=height, xaxis=gX(), yaxis=gY(), legend=leg(),
        title=dict(text=title, font=dict(color="#4a5e7a", size=11)))
    return fig

def page_chatbot():
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#000000'>Decision Intelligence Chatbot</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:16px'>
      <span class='badge badge-amber'>Demand</span>
      <span class='badge badge-teal'>Inventory</span>
      <span class='badge badge-lav'>Production</span>
      <span class='badge badge-coral'>Logistics</span>
      <span class='badge badge-sky'>Decision Alert</span>
      <span class='badge badge-purple'>3-Model Ensemble</span>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""<div style='margin-top:18px;border-top:1px solid rgba(255,255,255,0.06);
            padding-top:16px;font-family:DM Mono,monospace;font-size:0.65rem;
            color:#4a5e7a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>
            AI Config</div>""", unsafe_allow_html=True)
        api_key = st.text_input(
            "Groq API Key", type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxx", help="Paste your Groq API key"
        )
        if api_key and len(api_key.strip()) > 10:
            if api_key.strip().startswith("gsk_"):
                st.markdown("<div style='font-size:0.62rem;color:#56e0a0;font-family:DM Mono,monospace;margin-top:4px'>✅ Key looks valid</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.62rem;color:#ff6b6b;font-family:DM Mono,monospace;margin-top:4px'>⚠️ Key should start with gsk_</div>", unsafe_allow_html=True)

    ctx    = build_context()
    system = build_system_prompt(ctx)

    with st.expander("Live Context fed to AI", expanded=False):
        st.code(ctx, language="text")

    key_ok = bool(api_key and len(api_key.strip()) > 10)
    if not key_ok:
        st.markdown("""<div class='info-banner banner-amber'>
          <b style='color:#000000'>⚠️ API Key Required — Enter Groq key in the sidebar</b>
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
        "Which model (Ridge/RF/GradBoost) has the best R² score?",
        "If I increase lead time to 14 days, how does ROP change?",
        "What are the top 5 revenue-generating products?",
    ]

    if not st.session_state.chat_msgs:
        sec("Quick Queries — click any to get started", "⚡")
        cols = st.columns(4)
        for i, s in enumerate(CHAT_SUGGESTIONS):
            with cols[i % 4]:
                clicked = st.button(s, key=f"sug_{i}", use_container_width=True)
                if clicked:
                    if not key_ok:
                        st.warning("⚠️ Enter your API key in the sidebar first.")
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
            safe = _re.sub(r'\*\*(.+?)\*\*', r'<span style="color:#000000;font-weight:700">\1</span>', safe)
            safe = _re.sub(r'\*(.+?)\*',     r'<span style="color:#334155;font-style:italic">\1</span>', safe)
            html_parts = []
            for line in safe.split("\n"):
                line = line.strip()
                if not line:
                    html_parts.append("<div style='height:5px'></div>")
                elif _re.match(r"^[▸\-•] ", line):
                    body = line[2:].strip()
                    html_parts.append(f"<div style='display:flex;gap:8px;margin:5px 0'><span style='color:#000000;flex-shrink:0;margin-top:2px'>▸</span><span style='color:#334155;line-height:1.65'>{body}</span></div>")
                else:
                    html_parts.append(f"<div style='color:#334155;line-height:1.65;margin:3px 0'>{line}</div>")
            st.markdown(f"<div style='margin:10px 0'><div class='chat-ai-bubble'>{chr(10).join(html_parts)}</div></div>", unsafe_allow_html=True)

    sp()
    ci, cb, cc = st.columns([5,1,1])
    with ci:
        user_in = st.text_input("Ask anything…", key="user_input",
            placeholder="e.g. Which model had the best accuracy on the hold-out period?",
            label_visibility="collapsed")
    with cb:
        if st.button("Send", use_container_width=True):
            if not key_ok:
                st.warning("⚠️ Enter your API key in the sidebar first.")
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
                st.markdown(f"<div class='alert-item alert-critical'><b style='color:#000000'>{r['Product_Name']}</b> <span style='color:#333333'>[{r['Category']}]</span><br><span style='color:#4a5e7a;font-size:0.71rem'>Stock: {r['Current_Stock']} · ROP: {r['ROP']}</span></div>", unsafe_allow_html=True)
        with al2:
            st.markdown("""<div style='font-size:0.74rem;font-weight:700;color:#f5a623;
                letter-spacing:0.06em;text-transform:uppercase;font-family:DM Mono,monospace;margin-bottom:10px'>
                💰 Cost Saving Opportunities</div>""", unsafe_allow_html=True)
            _, _, opt, _, _ = compute_logistics()
            for _, r in opt.sort_values("Potential_Saving", ascending=False).head(5).iterrows():
                if r["Potential_Saving"] > 0:
                    st.markdown(f"<div class='alert-item alert-warn'><b style='color:#000000'>{r['Region']}</b> → <b style='color:#000000'>{r['Optimal_Carrier']}</b><br><span style='color:#4a5e7a;font-size:0.71rem'>Save ₹{r['Potential_Saving']:,.0f} ({r['Saving_Pct']:.1f}%)</span></div>", unsafe_allow_html=True)

        sp()
        sec("Revenue Forecast — Next 3 Months (Ensemble)", "📈")
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

def page_overview():
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("""<div class='page-title' style='background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow D2D</div>
    <div class='page-subtitle'>Predictive Logistics & AI-Powered Demand-to-Delivery Intelligence · 3-Model Ensemble Forecasting</div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class='about-card'>
      <div style='font-size:1.0rem;font-weight:700;color:#334155;margin-bottom:10px'>About This Platform</div>
      <p style='color:#334155;line-height:1.85;margin:0;font-size:0.86rem'>
        <b style='color:#000000'>OmniFlow</b> is an end-to-end intelligence platform built on
        <b style='color:#000000'>D2D orders</b> across India.
        Modules are causally connected: Demand signals drive Inventory EOQ/SS,
        which drives Production targets, which informs Logistics optimisation.
        All forecasting uses a <b style='color:#000000'>3-Model Ensemble</b>:
        <span class='model-pill pill-ridge'>Ridge Regression</span>
        <span class='model-pill pill-rf'>Random Forest</span>
        <span class='model-pill pill-gb'>Gradient Boosting</span>
        blended with <b style='color:#000000'>inverse-RMSE weights</b> for maximum accuracy.
        Revenue KPIs are net of returns.
      </p>
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
        sec("Monthly Net Revenue + Ensemble Forecast")
        m_rev_s = ops.groupby(ops["Order_Date"].dt.to_period("M"))["Net_Revenue"].sum().rename("v")
        r_ov = ml_forecast(m_rev_s.values.astype(float), m_rev_s.index, n_future=6)
        fig = go.Figure()
        if r_ov is not None:
            x_ci = list(r_ov["fut_ds"]) + list(r_ov["fut_ds"])[::-1]
            y_ci = list(r_ov["ci_hi"])  + list(r_ov["ci_lo"])[::-1]
            fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
                fillcolor="rgba(139,92,246,0.06)", line=dict(color="rgba(0,0,0,0)"),
                name="90% CI", showlegend=True))
            fig.add_trace(go.Scatter(x=r_ov["hist_ds"], y=r_ov["hist_y"], name="Actual",
                fill="tozeroy", fillcolor="rgba(245,166,35,0.04)",
                line=dict(color="#F59E0B", width=2.5),
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=r_ov["fut_ds"], y=r_ov["forecast"], name="Ensemble Forecast",
                mode="lines+markers",
                line=dict(color="#8B5CF6", width=2.5, dash="dot"),
                marker=dict(size=7, color="#8B5CF6", line=dict(color="#FFFFFF", width=2)),
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>"))
        fig.update_layout(**CD(), height=260, xaxis=gX(),
            yaxis={**gY(), "tickformat":",.0f"}, legend=leg())
        st.plotly_chart(fig, use_container_width=True, key="chart_1")

    with c_r:
        sec("Net Revenue by Category")
        cat = ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
        fig2 = go.Figure(go.Pie(labels=cat.index, values=cat.values, hole=.58,
            marker=dict(colors=COLORS, line=dict(color="#FFFFFF",width=3)),
            textinfo="label+percent", textfont=dict(size=10,color="#333333")))
        fig2.update_layout(**CD(), height=260, showlegend=False,
            annotations=[dict(text="Net Rev",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#4a5e7a",family="DM Mono"))])
        st.plotly_chart(fig2, use_container_width=True, key="chart_2")

    sp(0.5)
    c3a, c3b, c3c = st.columns(3, gap="large")
    with c3a:
        sec("Orders by Channel")
        ch = ops["Sales_Channel"].value_counts()
        fig3 = go.Figure(go.Bar(x=ch.values, y=ch.index, orientation="h",
            marker=dict(color=COLORS[:len(ch)], line=dict(color="rgba(0,0,0,0)")),
            text=ch.values, textposition="outside", textfont=dict(color="#333333",size=10),cliponaxis=False))
        fig3.update_layout(**CD(), height=240, xaxis=gX(), yaxis=dict(showgrid=False,color="#64748B"))
        st.plotly_chart(fig3, use_container_width=True, key="chart_3")

    with c3b:
        sec("Top Regions by Net Revenue")
        reg = ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False)
        fig4 = go.Figure(go.Bar(x=reg.index, y=reg.values,
            marker=dict(color=COLORS_S*2, line=dict(color="rgba(0,0,0,0)"))))
        fig4.update_layout(**CD(), height=240, xaxis={**gX(),"tickangle":-30}, yaxis=gY())
        st.plotly_chart(fig4, use_container_width=True, key="chart_4")

    with c3c:
        sec("Order Status Split")
        sc2 = df["Order_Status"].value_counts()
        fig5 = go.Figure(go.Bar(x=sc2.index, y=sc2.values,
            marker=dict(color=["#22C55E","#3B82F6","#EF4444","#F59E0B"][:len(sc2)], line=dict(color="rgba(0,0,0,0)"))))
        fig5.update_layout(**CD(), height=240, xaxis=gX(), yaxis=gY())
        st.plotly_chart(fig5, use_container_width=True, key="chart_5")

    sp(0.5)
    sec("Module Dependency Pipeline")
    st.markdown("""<div style='background:linear-gradient(135deg,var(--panel),var(--surface));
        border:1px solid var(--border);border-radius:16px;padding:22px;
        display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:0'>
      <div style='background:#f8fafc;color:#0f172a;border:1px solid #e5e7eb;border-radius:12px;padding:11px 17px;
           font-weight:700;font-size:0.78rem;text-align:center;min-width:110px;font-family:DM Mono,monospace'>
           Demand<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>Ridge+RF+GB</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:#f8fafc;border-radius:12px;padding:11px 17px;
           font-weight:700;font-size:0.78rem;text-align:center;min-width:110px;
           border:1px solid #e5e7eb;color:#0f172a;font-family:DM Mono,monospace'>
           Inventory<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>EOQ + ROP</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:#f8fafc;color:#0f172a;border:1px solid #e5e7eb;border-radius:12px;padding:11px 17px;
           font-weight:700;font-size:0.78rem;text-align:center;min-width:110px;font-family:DM Mono,monospace'>
           Production<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>6-Month Plan</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:#f8fafc;color:#0f172a;border:1px solid #e5e7eb;border-radius:12px;padding:11px 17px;
           font-weight:700;font-size:0.78rem;text-align:center;min-width:110px;font-family:DM Mono,monospace'>
           Logistics<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>+ Cost Opt</span></div>
      <div style='color:#4a5e7a;font-size:1.2rem;padding:0 8px;opacity:0.5'>→</div>
      <div style='background:#f8fafc;color:#0f172a;border:1px solid #e5e7eb;border-radius:12px;padding:11px 17px;
           font-weight:700;font-size:0.78rem;text-align:center;min-width:110px;font-family:DM Mono,monospace'>
           AI Chatbot<span style='display:block;font-size:0.58rem;font-weight:400;color:#4a5e7a;margin-top:3px'>Groq LLaMA</span></div>
    </div>""", unsafe_allow_html=True)

def page_demand():
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#000000'>Demand Forecasting</div>", unsafe_allow_html=True)
    
    banner("""<b style='color:#000000'>🤖 3-Model Ensemble:</b>
    <span class='model-pill pill-ridge'>Ridge Regression</span>
    <span class='model-pill pill-rf'>Random Forest</span>
    <span class='model-pill pill-gb'>Gradient Boosting</span>
    — blended via <b>inverse-RMSE weights</b> from rolling walk-forward CV (3 folds).
    Quantity demand uses <b>Net_Qty</b> (returns excluded — loop-back correction applied).
    Asymmetric log-normal 90% CI. Features: trend, Fourier seasonality, structural-break, quarter dummies.""", "purple")

    sec("Overall Ensemble Model Performance")
    m_orders = ops.groupby("YM")["Order_ID"].count().rename("v")
    res_overall = ml_forecast(m_orders.values.astype(float), m_orders.index, n_future=6)
    if res_overall is not None:
        render_model_quality(res_overall)
    sp()

    if res_overall and "model_metrics" in res_overall:
        sec("Model Accuracy Comparison")
        mm = res_overall["model_metrics"]
        mnames = ["Ridge", "RandomForest", "GradBoost", "Ensemble"]
        r2_vals    = [mm[m]["r2"]    for m in mnames if m in mm]
        nrmse_vals = [mm[m]["nrmse"] for m in mnames if m in mm]
        labels     = [m for m in mnames if m in mm]
        colors_bar = [MODEL_COLORS.get(m, "#888") for m in labels]

        bca, bcb = st.columns(2, gap="large")
        with bca:
            fig_r2 = go.Figure(go.Bar(
                x=labels, y=r2_vals,
                marker=dict(color=colors_bar, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
                textfont=dict(color="#334155")))
            fig_r2.add_hline(y=0.9, line_dash="dash", line_color="#22C55E",
                annotation_text=" Target R²=0.90", annotation_font=dict(color="#22C55E", size=10))
            fig_r2.update_layout(**CD(), height=240, xaxis=gX(),
                yaxis={**gY(), "title":"R² Score", "range":[0,1]},
                title=dict(text="R² Score by Model (higher = better)", font=dict(size=11,color="#4a5e7a")))
            st.plotly_chart(fig_r2, use_container_width=True, key="model_r2_bar")
        with bcb:
            fig_nrmse = go.Figure(go.Bar(
                x=labels, y=[v*100 for v in nrmse_vals],
                marker=dict(color=colors_bar, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v*100:.1f}%" for v in nrmse_vals], textposition="outside",
                textfont=dict(color="#334155")))
            fig_nrmse.add_hline(y=15, line_dash="dash", line_color="#22C55E",
                annotation_text=" Target NRMSE<15%", annotation_font=dict(color="#22C55E", size=10))
            fig_nrmse.update_layout(**CD(), height=240, xaxis=gX(),
                yaxis={**gY(), "title":"NRMSE %"},
                title=dict(text="NRMSE % by Model (lower = better)", font=dict(size=11,color="#4a5e7a")))
            st.plotly_chart(fig_nrmse, use_container_width=True, key="model_nrmse_bar")
        sp()

    c1, c2, c3 = st.columns([2,2,1])
    metric_opt = c1.selectbox("Metric",    ["Orders (#)","Quantity (Units)","Net Revenue (₹)"])
    level_opt  = c2.selectbox("Breakdown", ["Overall","Category","Region","Sales Channel"])
    horizon    = c3.slider("Months ahead", 3, 12, 6)

    col_map = {"Orders (#)":"Order_ID","Quantity (Units)":"Net_Qty","Net Revenue (₹)":"Net_Revenue"}
    col = col_map[metric_opt]

    def get_series(sub):
        if col == "Order_ID":
            return sub.groupby("YM")["Order_ID"].count().rename("v")
        return sub.groupby("YM")[col].sum().rename("v")

    def draw(series, color="#8B5CF6", title="", chart_key="demand_main"):
        res = ml_forecast(series.values.astype(float), series.index, n_future=horizon)
        if res is None:
            st.info("Insufficient data."); return None
        fig = draw_ensemble_chart(res, chart_key=chart_key, height=320, title=title)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
        return res

    if level_opt == "Overall":
        series = get_series(ops)
        res = draw(series, chart_key="demand_overall")
        if res is not None:
            sec("Forecast Table (Ensemble)")
            tbl = pd.DataFrame({
                "Month":     [d.strftime("%b %Y") for d in res["fut_ds"]],
                "Ensemble":  res["forecast"].round(0).astype(int),
                "Ridge":     np.maximum(res["forecast_per_model"]["Ridge"], 0).round(0).astype(int),
                "RandomForest": np.maximum(res["forecast_per_model"]["RandomForest"], 0).round(0).astype(int),
                "GradBoost": np.maximum(res["forecast_per_model"]["GradBoost"], 0).round(0).astype(int),
                "Lower 90%": res["ci_lo"].round(0).astype(int),
                "Upper 90%": res["ci_hi"].round(0).astype(int),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        grp_map = {"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}
        grp  = grp_map[level_opt]
        top  = ops[grp].value_counts().head(5).index.tolist()
        tabs = st.tabs(top)
        for _ti, (tab, val, color) in enumerate(zip(tabs, top, COLORS)):
            with tab:
                draw(get_series(ops[ops[grp]==val]), color=color, title=val, chart_key=f"demand_bd_{_ti}")

    sp()
    sec("YoY Revenue Growth")
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
            r24    = yr_rev.loc[2024, cat]
            r25    = yr_rev.loc[2025, cat]
            r_proj = proj_next.get(cat, 0)
            g25    = (r25-r24)/r24*100 if r24>0 else 0
            g_proj = (r_proj-r25)/r25*100 if r25>0 else 0
            rows_yoy.append({"Category":cat,
                "2024 (₹M)":round(r24/1e6,1), "2025 (₹M)":round(r25/1e6,1),
                "YoY 24→25":f"{g25:+.1f}%",
                "Projected (₹M)":round(r_proj/1e6,1), "YoY Growth (Proj)":f"{g_proj:+.1f}%"})
        st.dataframe(pd.DataFrame(rows_yoy).sort_values("Projected (₹M)", ascending=False),
                     use_container_width=True, hide_index=True)

    sp()
    sec("Category-Level Demand Forecast (Ensemble)")
    cat_monthly2 = ops.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
    tabs2 = st.tabs(list(cat_monthly2.columns))
    for _ci, (tab, cat, col2) in enumerate(zip(tabs2, cat_monthly2.columns, COLORS)):
        with tab:
            vals = cat_monthly2[cat].rename("v")
            res2 = ml_forecast(vals.values.astype(float), vals.index, n_future=6)
            if res2 is None:
                st.info("Insufficient data."); continue
            fig = draw_ensemble_chart(res2, chart_key=f"demand_cat_{_ci}", height=300, title=cat)
            st.plotly_chart(fig, use_container_width=True, key=f"demand_cat_{_ci}")
            tbl2 = pd.DataFrame({
                "Month":      [d.strftime("%b %Y") for d in res2["fut_ds"]],
                "Ensemble":   res2["forecast"].round(0).astype(int),
                "Ridge":      np.maximum(res2["forecast_per_model"]["Ridge"],0).round(0).astype(int),
                "RandomForest":np.maximum(res2["forecast_per_model"]["RandomForest"],0).round(0).astype(int),
                "GradBoost":  np.maximum(res2["forecast_per_model"]["GradBoost"],0).round(0).astype(int),
                "CI Lo":      res2["ci_lo"].round(0).astype(int),
                "CI Hi":      res2["ci_hi"].round(0).astype(int),
            })
            st.dataframe(tbl2, use_container_width=True, hide_index=True)

def page_inventory():
    df  = load_data()
    ops = get_ops(df)
    ops = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title' style='color:#000000'>Inventory Optimisation</div>", unsafe_allow_html=True)

    with st.expander("Inventory Parameters", expanded=False):
        p1,p2,p3,p4 = st.columns(4)
        order_cost = p1.number_input("Order Cost ₹", 100, 5000, 500, 50)
        hold_pct   = p2.slider("Holding Cost %", 5, 40, 20) / 100
        lead_time  = p3.slider("Lead Time days", 1, 30, 7)
        svc        = p4.selectbox("Service Level", ["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"])
        z_map      = {"90% (z=1.28)":1.28,"95% (z=1.65)":1.65,"99% (z=2.33)":2.33}
        z          = z_map[svc]

    banner("SS formula: <b>z × √(LT × σ_demand² + D_avg² × σ_LT²)</b> — accounts for both demand AND lead time variability. EOQ uses peak-blended demand (60% avg + 40% peak) to handle seasonality.", "purple")

    inv    = compute_inventory(order_cost, hold_pct, lead_time, z)
    if inv.empty:
        st.warning("No inventory data."); return

    n_crit = (inv["Status"]=="🔴 Critical").sum()
    n_low  = (inv["Status"]=="🟡 Low").sum()
    n_ok   = (inv["Status"]=="🟢 Adequate").sum()
    total_stockout = inv["Stockout_Cost_Day"].sum()
    n_a = (inv["ABC"]=="A").sum(); n_b = (inv["ABC"]=="B").sum(); n_c = (inv["ABC"]=="C").sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, "Total SKUs",       len(inv),          "mint")
    kpi(c2, "🔴 Critical",      n_crit,            "mint", "immediate reorder")
    kpi(c3, "🟡 Low",           n_low,             "mint", "approaching ROP")
    kpi(c4, "🟢 Adequate",      n_ok,              "mint", "well-stocked")
    kpi(c5, "A-Class SKUs",     n_a,               "sky",  "top 70% revenue")
    kpi(c6, "Stockout Cost/Day",f"₹{total_stockout:,.0f}", "coral", "critical SKUs only")
    sp()

    cl, cr = st.columns([1,2], gap="large")
    with cl:
        sec("Stock Status Distribution")
        sc_colors = {"🔴 Critical":"#EF4444","🟡 Low":"#F59E0B","🟢 Adequate":"#22C55E"}
        sc = inv["Status"].value_counts()
        fig = go.Figure(go.Pie(labels=sc.index, values=sc.values, hole=.6,
            marker=dict(colors=[sc_colors.get(s,"#4a5e7a") for s in sc.index], line=dict(color="#FFFFFF",width=3)),
            textinfo="label+value", textfont=dict(size=10,color="#333333")))
        fig.update_layout(**CD(), height=270, showlegend=False,
            annotations=[dict(text="SKUs",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#4a5e7a",family="DM Mono"))])
        st.plotly_chart(fig, use_container_width=True, key="chart_7")

    with cr:
        sec("EOQ / Safety Stock / ROP by Category")
        ci2 = inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2 = go.Figure()
        for i,(m2,lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(name=lbl, x=ci2["Category"], y=ci2[m2].round(1),
                marker=dict(color=["#F59E0B","#06B6D4","#8B5CF6"][i], line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(), height=270, barmode="group",
            xaxis={**gX(),"tickangle":-10}, yaxis=gY(), legend=leg())
        st.plotly_chart(fig2, use_container_width=True, key="chart_8")

    sp()
    sec("EOQ Cost Trade-off: Annual Ordering vs Holding Cost by Category")
    eoq_tbl = inv.groupby("Category").agg(
        Avg_EOQ=("EOQ","mean"), Avg_Ann_Demand=("Annual_Demand","mean"),
        Avg_Price=("Unit_Price","mean")
    ).reset_index()
    eoq_tbl["Ann_Order_Cost"]   = ((eoq_tbl["Avg_Ann_Demand"] / eoq_tbl["Avg_EOQ"].replace(0,1)) * order_cost).round(0)
    eoq_tbl["Ann_Holding_Cost"] = ((eoq_tbl["Avg_EOQ"] / 2) * eoq_tbl["Avg_Price"] * hold_pct).round(0)
    eoq_tbl["Total_Cost"]       = eoq_tbl["Ann_Order_Cost"] + eoq_tbl["Ann_Holding_Cost"]
    fig_eoq = go.Figure()
    fig_eoq.add_trace(go.Bar(name="Annual Ordering Cost", x=eoq_tbl["Category"], y=eoq_tbl["Ann_Order_Cost"],
        marker=dict(color="#3B82F6", line=dict(color="rgba(0,0,0,0)"))))
    fig_eoq.add_trace(go.Bar(name="Annual Holding Cost", x=eoq_tbl["Category"], y=eoq_tbl["Ann_Holding_Cost"],
        marker=dict(color="#F59E0B", line=dict(color="rgba(0,0,0,0)"))))
    fig_eoq.add_trace(go.Scatter(name="Total Cost", x=eoq_tbl["Category"], y=eoq_tbl["Total_Cost"],
        mode="markers+text", marker=dict(size=12, color="#EF4444", symbol="diamond"),
        text=[f"₹{v:,.0f}" for v in eoq_tbl["Total_Cost"]], textposition="top center",
        textfont=dict(color="#334155", size=9)))
    fig_eoq.update_layout(**CD(), height=260, barmode="group",
        xaxis={**gX(),"tickangle":-10}, yaxis={**gY(),"title":"₹ / Year"}, legend=leg())
    st.plotly_chart(fig_eoq, use_container_width=True, key="eoq_cost_chart")

    sp()
    abc_l, abc_r = st.columns(2, gap="large")
    with abc_l:
        sec("ABC Classification (Revenue Pareto)")
        abc_grp = inv.groupby("ABC").agg(SKUs=("SKU_ID","count"), Revenue=("Total_Revenue","sum")).reset_index()
        abc_grp["Rev_Pct"] = (abc_grp["Revenue"] / abc_grp["Revenue"].sum() * 100).round(1)
        fig_abc = go.Figure(go.Bar(
            x=abc_grp["ABC"], y=abc_grp["Rev_Pct"],
            marker=dict(color=["#1565C0","#2E7D32","#E65100"], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{r['SKUs']} SKUs<br>{r['Rev_Pct']:.1f}%" for _,r in abc_grp.iterrows()],
            textposition="outside", textfont=dict(color="#334155")))
        fig_abc.update_layout(**CD(), height=250,
            xaxis={**gX(),"title":"ABC Class"},
            yaxis={**gY(),"title":"Revenue %"})
        st.plotly_chart(fig_abc, use_container_width=True, key="abc_chart")

    with abc_r:
        sec("Stockout Cost by Category (Critical SKUs)")
        so = inv[inv["Status"]=="🔴 Critical"].groupby("Category")["Stockout_Cost_Day"].sum().reset_index()
        if so.empty:
            st.info("No critical SKUs.")
        else:
            fig_so = go.Figure(go.Bar(
                x=so["Category"], y=so["Stockout_Cost_Day"],
                marker=dict(color="#EF4444", line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:,.0f}/day" for v in so["Stockout_Cost_Day"]],
                textposition="outside", textfont=dict(color="#334155")))
            fig_so.update_layout(**CD(), height=250,
                xaxis=gX(), yaxis={**gY(),"title":"₹ Lost / Day"})
            st.plotly_chart(fig_so, use_container_width=True, key="stockout_chart")

    sec("Critical SKU Alerts — Reorder Immediately")
    crit_df = inv[inv["Status"]=="🔴 Critical"][
        ["SKU_ID","Product_Name","Category","ABC","Current_Stock","SS","ROP","EOQ","Monthly_Avg","Unit_Price","Stockout_Cost_Day"]
    ].copy()
    crit_df.columns = ["SKU","Product","Category","ABC","Current Stock","Safety Stock","ROP","Order Qty (EOQ)","Avg/Month","Unit Price ₹","Stockout ₹/Day"]
    for c in ["Current Stock","Safety Stock","ROP","Order Qty (EOQ)"]:
        crit_df[c] = crit_df[c].astype(int)
    st.dataframe(crit_df.sort_values("Stockout ₹/Day", ascending=False), use_container_width=True, hide_index=True)

    sp()
    sec("Inventory Demand Forecast (Ensemble)")
    cat_monthly = ops.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
    if cat_monthly.empty:
        st.info("No category demand data available for forecasting.")
        return
    tabs = st.tabs(list(cat_monthly.columns))
    for tab,cat in zip(tabs,cat_monthly.columns):
        with tab:
            series = cat_monthly[cat]
            res = ml_forecast(series.values.astype(float), series.index, 6)
            if res is None:
                st.info("Not enough data for forecast.")
                continue
            fig = draw_ensemble_chart(res, chart_key=f"inv_demand_{cat}", height=280)
            st.plotly_chart(fig, use_container_width=True, key=f"inv_demand_{cat}")

    sp()
    sec("SKU-Level Inventory Table")
    abc_f  = st.multiselect("Filter ABC", ["A","B","C"], default=["A","B","C"])
    cat_f  = st.multiselect("Filter Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    stat_f = st.multiselect("Filter Status", ["🔴 Critical","🟡 Low","🟢 Adequate"], default=["🔴 Critical","🟡 Low","🟢 Adequate"])
    disp   = inv[
        (inv["ABC"].isin(abc_f)) &
        (inv["Category"].isin(cat_f)) &
        (inv["Status"].isin(stat_f))
    ][["SKU_ID","Product_Name","Category","ABC","Monthly_Avg","Current_Stock","EOQ","SS","ROP","Unit_Price","Stockout_Cost_Day","Status"]].copy()
    disp.columns = ["SKU","Product","Category","ABC","Avg/Month","Current Stock","EOQ","Safety Stock","ROP","Price ₹","Stockout ₹/Day","Status"]
    for c in ["Avg/Month","Current Stock","EOQ","Safety Stock","ROP"]:
        disp[c] = disp[c].astype(int)
    st.dataframe(disp.sort_values(["ABC","Status"]), use_container_width=True, hide_index=True)

    sp()
    sec("Stock Level Forecast — Depletion & Replenishment Simulation")
    plan_for_inv = compute_production()   # feed from production module
    cat_monthly_qty = ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
    cats = sorted(inv["Category"].unique())
    tabs_inv = st.tabs(cats)
    for tab, cat in zip(tabs_inv, cats):
        with tab:
            cat_inv = inv[inv["Category"] == cat]
            if cat_inv.empty or cat not in cat_monthly_qty.columns:
                st.info("No data."); continue

            avg_eoq    = max(int(cat_inv["EOQ"].mean()), 1)
            avg_rop    = max(int(cat_inv["ROP"].mean()), 1)
            avg_ss     = max(int(cat_inv["SS"].mean()), 0)
            # Use SUM of current stock across all SKUs in category — not avg
            total_stock = max(int(cat_inv["Current_Stock"].sum()), 0)
            n_crit_cat  = (cat_inv["Status"] == "🔴 Critical").sum()
            n_low_cat   = (cat_inv["Status"] == "🟡 Low").sum()

            # Use production plan forecast as demand driver (production = what needs to ship)
            cat_plan = plan_for_inv[plan_for_inv["Category"] == cat].sort_values("Month_dt") if not plan_for_inv.empty else pd.DataFrame()

            vals = cat_monthly_qty[cat].values.astype(float)
            res  = ml_forecast(vals, cat_monthly_qty.index, 6)
            if res is None:
                st.info("Insufficient data for forecast."); continue

            # Use production forecast if available, else fall back to demand forecast
            if not cat_plan.empty and len(cat_plan) == len(res["fut_ds"]):
                sim_demand = cat_plan["Demand_Forecast"].values
                sim_label  = "Production-driven demand"
            else:
                sim_demand = res["forecast"]
                sim_label  = "ML demand forecast"

            stock = total_stock
            stock_levels=[]; reorder_months=[]; reorder_qty=[]
            months_labels = [d.strftime("%b %Y") for d in res["fut_ds"]]
            for i, fc_demand in enumerate(sim_demand):
                stock -= fc_demand
                if stock <= avg_rop:
                    n_orders  = max(1, int(np.ceil((avg_rop + avg_ss - stock) / avg_eoq)) + 1)
                    order_qty = n_orders * avg_eoq
                    stock    += order_qty
                    reorder_months.append(i); reorder_qty.append(order_qty)
                stock = max(stock, avg_ss)
                stock_levels.append(round(stock))

            fig = go.Figure()
            resid = res["resid_std"]
            ci_upper = [max(s + resid * (i+1) * 0.5, avg_ss) for i, s in enumerate(stock_levels)]
            ci_lower = [max(s - resid * (i+1) * 0.5, 0)       for i, s in enumerate(stock_levels)]
            fig.add_trace(go.Scatter(x=months_labels+months_labels[::-1], y=ci_upper+ci_lower[::-1],
                fill="toself", fillcolor="rgba(46,216,195,0.06)", line=dict(color="rgba(0,0,0,0)"),
                name="Stock Uncertainty Band"))
            fig.add_trace(go.Scatter(x=months_labels, y=stock_levels, name="Projected Stock",
                mode="lines+markers", line=dict(color="#06B6D4", width=3),
                marker=dict(size=10, color=stock_levels,
                    colorscale=[[0,"#EF4444"],[0.4,"#F59E0B"],[1,"#22C55E"]],
                    cmin=avg_ss, cmax=max(stock_levels) if stock_levels else 1,
                    line=dict(color="#FFFFFF", width=2), showscale=False),
                hovertemplate="<b>%{x}</b><br>Stock: %{y} units<extra></extra>"))
            # Overlay production plan demand as a reference line
            if not cat_plan.empty:
                fig.add_trace(go.Scatter(
                    x=months_labels, y=list(sim_demand), name="Production Demand",
                    mode="lines", line=dict(color="#F59E0B", width=1.5, dash="dot"),
                    opacity=0.7))
            fig.add_hline(y=avg_rop, line_dash="dash", line_color="#F59E0B", line_width=2,
                annotation_text=f"  ROP: {avg_rop}", annotation_font=dict(color="#F59E0B",size=11,family="DM Mono"))
            fig.add_hline(y=avg_ss, line_dash="dot", line_color="#EF4444", line_width=2,
                annotation_text=f"  SS: {avg_ss}", annotation_font=dict(color="#EF4444",size=11,family="DM Mono"))
            for ri, rqty in zip(reorder_months, reorder_qty):
                fig.add_vline(x=ri, line_dash="dot", line_color="rgba(155,135,212,0.6)", line_width=1.5)
                fig.add_annotation(x=months_labels[ri], y=max(stock_levels)*1.08 if stock_levels else avg_rop*2,
                    text=f"📦 +{rqty}u", showarrow=False,
                    font=dict(color="#8B5CF6",size=10,family="DM Mono"),
                    bgcolor="rgba(22,34,54,0.8)",bordercolor="rgba(155,135,212,0.3)",
                    borderwidth=1, borderpad=4)
            fig.update_layout(**CD(), height=320,
                xaxis={**gX(),"title":"Month"}, yaxis={**gY(),"title":"Units in Stock"},
                legend={**leg(),"orientation":"h","y":-0.25},
                title=dict(text=f"{cat} — {sim_label} · Starting stock: {total_stock:,} units (category total)",
                           font=dict(color="#333333",size=11)))
            st.plotly_chart(fig, use_container_width=True, key=f"inv_stock_{cat}")

            ka,kb,kc,kd,ke = st.columns(5)
            kpi(ka,"Starting Stock", avg_stock, "mint","current avg")
            kpi(kb,"ROP (avg)",      avg_rop,   "mint","trigger level")
            kpi(kc,"Safety Stock",   avg_ss,    "mint","min buffer")
            kpi(kd,"EOQ (avg)",      avg_eoq,   "mint","order batch size")
            kpi(ke,"Reorders",       len(reorder_months),"mint",
                "over 6 months" if len(reorder_months)==0 else f"across {len(reorder_months)} month(s)")
            sp(0.5)
            if n_crit_cat > 0:
                banner(f"🔴 <b style='color:#000000'>{n_crit_cat} Critical SKUs</b> in {cat} are below Safety Stock. Immediate replenishment required.", "coral")
            elif n_low_cat > 0:
                banner(f"🟡 <b style='color:#000000'>{n_low_cat} SKUs</b> approaching ROP — place orders this week.", "amber")
            else:
                banner(f"✅ All SKUs in <b style='color:#000000'>{cat}</b> are Adequate. Next reorder at ROP={avg_rop}.", "mint")

    sp()
    sec("Category Demand Forecast Comparison")
    fig3 = go.Figure()
    for i, cat in enumerate(cat_monthly_qty.columns):
        r = ml_forecast(cat_monthly_qty[cat].values.astype(float), cat_monthly_qty.index, 6)
        if r is None: continue
        clr = COLORS[i % len(COLORS)]
        fig3.add_trace(go.Scatter(x=r["hist_ds"], y=r["hist_y"], line=dict(color=clr,width=1,dash="dot"), opacity=0.6, showlegend=False))
        fig3.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=cat,
            mode="lines+markers", line=dict(color=clr,width=2.5), marker=dict(size=7,color=clr,line=dict(color="#FFFFFF",width=2))))
    fig3.update_layout(**CD(), height=280, xaxis=gX(), yaxis={**gY(),"title":"Forecast Units"}, legend=leg())
    st.plotly_chart(fig3, use_container_width=True, key="inv_cat_demand")

def page_production():
    st.markdown("<div class='page-title' style='color:#000000'>Production Planning</div>", unsafe_allow_html=True)
    p1,p2 = st.columns(2)
    cap = p1.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1)
    buf = p2.slider("Safety Buffer %", 5, 40, 15) / 100

    plan   = compute_production(cap, buf)
    inv    = compute_inventory()

    if plan.empty:
        st.warning("Insufficient data for production plan."); return

    agg = plan.groupby("Month_dt")[["Production","Demand_Forecast","Crit_Boost","Low_Boost"]].sum().reset_index()

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Total Production", f"{plan['Production'].sum():,.0f}","amber","units · 6 months")
    kpi(c2,"Total Demand",     f"{plan['Demand_Forecast'].sum():,.0f}","amber","forecast units")
    kpi(c3,"Avg / Month",      f"{agg['Production'].mean():,.0f}","amber","units/month")
    peak = agg.loc[agg["Production"].idxmax(),"Month_dt"]
    kpi(c4,"Peak Month",       peak.strftime("%b %Y"),"amber","highest demand")
    sp()

    sec("Production Target vs Ensemble Demand Forecast")
    df_prod  = load_data()
    ops_prod = get_ops(df_prod).copy()
    ops_prod["YM"] = ops_prod["Order_Date"].dt.to_period("M")
    hist_qty = ops_prod.groupby("YM")["Quantity"].sum().rename("v")
    hist_ts  = _to_timestamp_index(hist_qty.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_ts, y=hist_qty.values, name="Historical Demand",
        fill="tozeroy", fillcolor="rgba(74,94,122,0.12)",
        line=dict(color="#4a5e7a", width=1.8),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>"))
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Production"], name="Production Target",
        marker=dict(color="#8B5CF6", line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Bar(x=agg["Month_dt"], y=agg["Crit_Boost"]+agg["Low_Boost"],
        name="Inv. Replenishment", marker=dict(color="rgba(255,107,107,0.7)", line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Scatter(x=agg["Month_dt"], y=agg["Demand_Forecast"], name="Ensemble Forecast",
        mode="lines+markers", line=dict(color="#F59E0B",width=2.5),
        marker=dict(size=8,color="#F59E0B",line=dict(color="#FFFFFF",width=2))))
    fig.update_layout(**CD(), height=320, barmode="stack", xaxis=gX(), yaxis=gY(), legend=leg())
    st.plotly_chart(fig, use_container_width=True, key="chart_11")

    cl, cr = st.columns(2, gap="large")
    with cl:
        sec("Production by Category")
        cat_hist = ops_prod.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
        cat_hist_ts = _to_timestamp_index(cat_hist.index)
        fig2 = go.Figure()
        for i, cat in enumerate(plan["Category"].unique()):
            clr = COLORS[i%len(COLORS)]
            if cat in cat_hist.columns:
                fig2.add_trace(go.Scatter(x=cat_hist_ts, y=cat_hist[cat].values,
                    name=f"{cat} (hist)", line=dict(color=clr,width=1.5,dash="dot"),
                    opacity=0.6, showlegend=False))
            s = plan[plan["Category"]==cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(x=s["Month_dt"], y=s["Production"], name=cat,
                marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(), height=280, barmode="stack",
            xaxis=gX(), yaxis=gY(), legend={**leg(),"orientation":"h","y":-0.3})
        st.plotly_chart(fig2, use_container_width=True, key="chart_12")
    with cr:
        sec("Production Demand Gap")
        agg["Gap"] = agg["Production"] - agg["Demand_Forecast"]
        fig3 = go.Figure(go.Bar(x=agg["Month_dt"], y=agg["Gap"],
            marker=dict(color=["#22C55E" if g>=0 else "#EF4444" for g in agg["Gap"]], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside", textfont=dict(color="#333333")))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
        fig3.update_layout(**CD(), height=280, xaxis=gX(), yaxis={**gY(),"title":"Units Surplus / Deficit"})
        st.plotly_chart(fig3, use_container_width=True, key="chart_13")

    sec("Detailed Production Schedule")
    cat_f = st.selectbox("Filter Category", ["All"] + list(plan["Category"].unique()))
    d2 = plan if cat_f=="All" else plan[plan["Category"]==cat_f]
    d3 = d2[["Month","Category","Demand_Forecast","Crit_Boost","Low_Boost","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d3.columns = ["Month","Category","Demand Fc","Crit Boost","Low Boost","Buffer","Production","Demand Lo","Demand Hi"]
    st.dataframe(d3.sort_values("Month"), use_container_width=True, hide_index=True)

def page_logistics():
    df     = load_data()
    ops    = get_ops(df)
    ops    = ops.copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    del_df = get_delivered(df)

    st.markdown("<div class='page-title' style='color:#000000'>Logistics Optimisation</div>", unsafe_allow_html=True)

    with st.expander("Carrier Scoring Weights", expanded=False):
        wc1, wc2, wc3 = st.columns(3)
        w_speed   = wc1.slider("Speed weight %",   10, 70, 40) / 100
        w_cost    = wc2.slider("Cost weight %",    10, 70, 35) / 100
        w_returns = wc3.slider("Returns weight %", 10, 70, 25) / 100
        total_w   = w_speed + w_cost + w_returns
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w*100:.0f}% — they will be normalised automatically.")
        w_speed /= total_w; w_cost /= total_w; w_returns /= total_w

    carr, best_carr, opt, _, fwd_plan = compute_logistics(w_speed, w_cost, w_returns)
    t1,t2,t3,t4,t5 = st.tabs(["Carrier Scorecard","Cost Optimisation","Delay Intel","Warehouse Forecast","Regions"])

    with t1:
        sec("Carrier Performance Scorecard")
        banner(f"Score weights: <b>Speed {w_speed*100:.0f}%</b> · <b>Cost {w_cost*100:.0f}%</b> · <b>Returns {w_returns*100:.0f}%</b> — normalised composite 0–1 (higher = better)", "teal")
        fig = go.Figure()
        for i, (_, r) in enumerate(carr.iterrows()):
            fig.add_trace(go.Scatter(x=[r["Avg_Days"]], y=[r["Avg_Cost"]], mode="markers+text",
                marker=dict(size=max(r["Orders"]/35,16), color=COLORS[i], opacity=0.9, line=dict(color="#FFFFFF",width=2)),
                text=[r["Courier_Partner"]], textposition="top center", name=r["Courier_Partner"],
                hovertemplate=f"<b>{r['Courier_Partner']}</b><br>Orders:{r['Orders']}<br>Avg Del:{r['Avg_Days']:.1f}d<br>Avg Cost:₹{r['Avg_Cost']:.0f}<br>Score:{r['Perf_Score']:.3f}<extra></extra>"))
        fig.update_layout(**CD(), height=320, showlegend=False,
            xaxis={**gY(),"title":"Avg Delivery Days"}, yaxis={**gY(),"title":"Avg Shipping Cost"})
        st.plotly_chart(fig, use_container_width=True, key="chart_14")

        d2 = carr[["Courier_Partner","Orders","Avg_Days","Avg_Cost","Return_Rate","Delay_Index","Cost_Score","Perf_Score"]].copy()
        d2["Avg_Days"]    = d2["Avg_Days"].round(1)
        d2["Avg_Cost"]    = d2["Avg_Cost"].round(1)
        d2["Return_Rate"] = (d2["Return_Rate"]*100).round(1).astype(str)+"%"
        d2["Perf_Score"]  = d2["Perf_Score"].round(3)
        d2.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Delay Index","Cost Index","Perf Score"]
        st.dataframe(d2.sort_values("Perf Score", ascending=False), use_container_width=True, hide_index=True)

        sp()
        sec("Carrier Order Volume + Ensemble Forecast")
        cm = del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        fig_carr_combo = go.Figure()
        for i, c in enumerate(cm.columns):
            clr = COLORS[i%len(COLORS)]
            r = ml_forecast(cm[c].values.astype(float), cm.index, 6)
            if r is None:
                fig_carr_combo.add_trace(go.Scatter(x=cm.index.to_timestamp(), y=cm[c], name=c, line=dict(color=clr,width=2)))
                continue
            x_ci = list(r["fut_ds"]) + list(r["fut_ds"])[::-1]
            y_ci = list(r["ci_hi"])  + list(r["ci_lo"])[::-1]
            fig_carr_combo.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
                fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.06)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            fig_carr_combo.add_trace(go.Scatter(x=r["hist_ds"], y=r["hist_y"], name=c,
                line=dict(color=clr, width=2.5)))
            fig_carr_combo.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"],
                name=f"{c} (fcst)", line=dict(color=clr, width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=6,line=dict(color="#FFFFFF",width=1.5)), showlegend=False))
        fig_carr_combo.update_layout(**CD(), height=300, xaxis=gX(), yaxis={**gY(),"title":"Orders"}, legend=leg())
        st.plotly_chart(fig_carr_combo, use_container_width=True, key="chart_15")

        sec("Recommended Carrier per Category")
        plan = compute_production()
        if not plan.empty:
            prod_by_cat = plan.groupby("Category")["Production"].sum().reset_index()
            cat_carr = del_df.groupby(["Category","Courier_Partner"]).agg(
                Avg_Days=("Delivery_Days","mean"),
                Avg_Cost=("Shipping_Cost_INR","mean"),
                Return_Rate=("Return_Flag","mean"),
            ).reset_index()
            for col_c, wt_c in [("Avg_Days", w_speed), ("Avg_Cost", w_cost), ("Return_Rate", w_returns)]:
                mn_c = cat_carr[col_c].min(); mx_c = cat_carr[col_c].max()
                cat_carr[f"N_{col_c}"] = 1 - (cat_carr[col_c] - mn_c) / (mx_c - mn_c + 1e-9)
            cat_carr["Score"] = w_speed*cat_carr["N_Avg_Days"] + w_cost*cat_carr["N_Avg_Cost"] + w_returns*cat_carr["N_Return_Rate"]
            best_cat = cat_carr.sort_values("Score", ascending=False).groupby("Category").first().reset_index()
            best_cat = best_cat.merge(prod_by_cat.rename(columns={"Production":"Planned Units 6M"}), on="Category", how="left")
            best_cat["Avg_Days"]        = best_cat["Avg_Days"].round(1)
            best_cat["Avg_Cost"]        = best_cat["Avg_Cost"].round(1)
            best_cat["Return_Rate"]     = (best_cat["Return_Rate"]*100).round(1)
            best_cat["Score"]           = best_cat["Score"].round(3)
            best_cat["Planned Units 6M"]= best_cat["Planned Units 6M"].fillna(0).astype(int)
            best_cat = best_cat[["Category","Courier_Partner","Avg_Days","Avg_Cost","Return_Rate","Score","Planned Units 6M"]]
            best_cat.columns = ["Category","Recommended Carrier","Avg Days","Avg Cost ₹","Return Rate %","Score","Planned Units"]
            st.dataframe(best_cat.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)

    with t2:
        sec("Logistics Cost Optimisation Analysis")
        total_current = del_df["Shipping_Cost_INR"].sum()
        total_saving  = opt["Potential_Saving"].sum()
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1,"Current Spend",         f"₹{total_current:,.0f}","sky","all deliveries")
        kpi(c2,"Optimised Spend",       f"₹{total_current-total_saving:,.0f}","sky","with best carriers")
        kpi(c3,"Total Saving Potential",f"₹{total_saving:,.0f}","sky","by region switch")
        kpi(c4,"Saving %",              f"{total_saving/total_current*100:.1f}%","sky","of total spend")
        sp()
        sec("Region-Level Cost Comparison")
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(name="Current Avg ₹", x=opt["Region"], y=opt["Current_Avg_Cost"], marker=dict(color="#EF4444",line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.add_trace(go.Bar(name="Optimal Avg ₹", x=opt["Region"], y=opt["Min_Avg_Cost"],     marker=dict(color="#22C55E",line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.update_layout(**CD(), height=280, barmode="group", xaxis={**gX(),"tickangle":-25}, yaxis=gY(), legend=leg())
        st.plotly_chart(fig_cost, use_container_width=True, key="chart_16")
        sec("Saving by Region")
        s_sorted = opt.sort_values("Potential_Saving", ascending=False)
        fig_sav = go.Figure(go.Bar(x=s_sorted["Region"],y=s_sorted["Potential_Saving"],
            marker=dict(color="#F59E0B",line=dict(color="rgba(0,0,0,0)")),
            text=[f"₹{v:,.0f}" for v in s_sorted["Potential_Saving"]],
            textposition="outside",textfont=dict(color="#333333")))
        fig_sav.update_layout(**CD(), height=250, xaxis={**gX(),"tickangle":-25}, yaxis=gY())
        st.plotly_chart(fig_sav, use_container_width=True, key="chart_17")
        sp()
        sec("Logistics Order Forecast (Ensemble)")
        series = del_df.groupby(del_df["Order_Date"].dt.to_period("M"))["Order_ID"].count()
        res = ml_forecast(series.values.astype(float), series.index, 6)
        if res:
            fig = draw_ensemble_chart(res, chart_key="log_ord_fc", height=300)
            st.plotly_chart(fig, use_container_width=True, key="log_ord_fc")
        sec("Optimisation Recommendation Table")
        opt_disp = opt.copy()
        opt_disp["Current_Avg_Cost"] = opt_disp["Current_Avg_Cost"].round(1)
        opt_disp["Min_Avg_Cost"]     = opt_disp["Min_Avg_Cost"].round(1)
        opt_disp["Potential_Saving"] = opt_disp["Potential_Saving"].astype(int)
        opt_disp = opt_disp[["Region","Optimal_Carrier","Current_Avg_Cost","Min_Avg_Cost","Potential_Saving","Saving_Pct","Orders"]]
        opt_disp.columns = ["Region","Switch To","Current Avg ₹","Optimal Avg ₹","Saving ₹","Saving %","Orders"]
        st.dataframe(opt_disp.sort_values("Saving ₹", ascending=False), use_container_width=True, hide_index=True)

        sp()
        sec("Forward Shipment Plan (Production-Driven)")
        if not fwd_plan.empty:
            banner("Projected shipment volumes and costs are derived from the <b>Production Plan</b> — not historical data. Uses avg shipping cost/unit from delivered orders.", "teal")
            # Monthly total across all categories
            fwd_agg = fwd_plan.groupby("Month_dt").agg(
                Month=("Month","first"),
                Total_Units=("Prod_Units","sum"),
                Total_Orders=("Proj_Orders","sum"),
                Total_Ship_Cost=("Proj_Ship_Cost","sum"),
                CI_Lo=("CI_Lo_Units","sum"),
                CI_Hi=("CI_Hi_Units","sum"),
            ).reset_index().sort_values("Month_dt")

            fa1, fa2, fa3 = st.columns(3)
            kpi(fa1, "6M Planned Units",   f"{fwd_agg['Total_Units'].sum():,}",         "sky", "from production plan")
            kpi(fa2, "6M Projected Orders",f"{fwd_agg['Total_Orders'].sum():,}",         "sky", "estimated shipments")
            kpi(fa3, "6M Proj Ship Cost",  f"₹{fwd_agg['Total_Ship_Cost'].sum():,.0f}", "sky", "at current avg rate")
            sp()

            fig_fwd = go.Figure()
            x_ci = list(fwd_agg["Month_dt"]) + list(fwd_agg["Month_dt"])[::-1]
            y_ci = list(fwd_agg["CI_Hi"])    + list(fwd_agg["CI_Lo"])[::-1]
            fig_fwd.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
                fillcolor="rgba(59,130,246,0.07)", line=dict(color="rgba(0,0,0,0)"), name="Demand CI"))
            fig_fwd.add_trace(go.Bar(
                x=fwd_agg["Month_dt"], y=fwd_agg["Total_Units"],
                name="Planned Shipment Units", marker=dict(color="#3B82F6", opacity=0.8, line=dict(color="rgba(0,0,0,0)")),
                hovertemplate="<b>%{x|%b %Y}</b><br>Units: %{y:,}<extra></extra>"))
            fig_fwd.update_layout(**CD(), height=270, barmode="overlay",
                xaxis=gX(), yaxis={**gY(),"title":"Planned Units"},
                legend={**leg(),"orientation":"h","y":-0.25})
            st.plotly_chart(fig_fwd, use_container_width=True, key="fwd_ship_chart")

            # Projected cost line
            fig_cost_fwd = go.Figure(go.Scatter(
                x=fwd_agg["Month_dt"], y=fwd_agg["Total_Ship_Cost"],
                mode="lines+markers", line=dict(color="#8B5CF6", width=2.5),
                marker=dict(size=8, color="#8B5CF6", line=dict(color="#FFFFFF", width=2)),
                fill="toself", fillcolor="rgba(139,92,246,0.06)",
                hovertemplate="<b>%{x|%b %Y}</b><br>Proj Cost: ₹%{y:,.0f}<extra></extra>",
                name="Projected Shipping Cost"))
            fig_cost_fwd.update_layout(**CD(), height=230,
                xaxis=gX(), yaxis={**gY(),"title":"₹ Shipping Cost"})
            st.plotly_chart(fig_cost_fwd, use_container_width=True, key="fwd_cost_chart")

            # Per-category breakdown table
            sec("Category Breakdown — 6-Month Shipment Plan")
            cat_fwd = fwd_plan.groupby("Category").agg(
                Units=("Prod_Units","sum"),
                Orders=("Proj_Orders","sum"),
                Ship_Cost=("Proj_Ship_Cost","sum"),
            ).reset_index().sort_values("Units", ascending=False)
            cat_fwd.columns = ["Category","Planned Units","Est. Orders","Proj. Ship Cost ₹"]
            st.dataframe(cat_fwd, use_container_width=True, hide_index=True)
        else:
            st.info("Production plan not available — run production module first.")

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
                text=[f"{v}%" for v in rd_s["Rate"]], textposition="outside", textfont=dict(color="#333333")))
            fig_r.update_layout(**CD(), height=300, xaxis={**gX(),"title":"Delay %"}, yaxis=dict(showgrid=False,color="#64748B"))
            st.plotly_chart(fig_r, use_container_width=True, key="chart_18")
        with cr3:
            sec("Delay Rate by Carrier")
            cd = del_df2.groupby("Courier_Partner").agg(T=("Order_ID","count"), D=("Delayed","sum")).reset_index()
            cd["Rate"] = (cd["D"]/cd["T"]*100).round(1)
            fig_c = go.Figure(go.Bar(x=cd["Courier_Partner"], y=cd["Rate"],
                marker=dict(color=["#EF4444" if v>35 else "#F59E0B" if v>20 else "#22C55E" for v in cd["Rate"]], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v}%" for v in cd["Rate"]], textposition="outside", textfont=dict(color="#333333")))
            fig_c.update_layout(**CD(), height=300, xaxis=gX(), yaxis={**gY(),"title":"Delay %"})
            st.plotly_chart(fig_c, use_container_width=True, key="chart_19")
        sec("Carrier × Region Delay Heatmap")
        pv = del_df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0)*100
        fig_h = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index),
            colorscale=[[0,"#0d1829"],[0.4,"#7c4fd0"],[0.7,"#e87adb"],[1,"#EF4444"]],
            text=np.round(pv.values,1), texttemplate="%{text}%", textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color="#8a9dc0",size=10))))
        fig_h.update_layout(**CD(), height=260,
            xaxis=dict(showgrid=False,tickangle=-25,color="#64748B"),
            yaxis=dict(showgrid=False,color="#64748B"))
        st.plotly_chart(fig_h, use_container_width=True, key="chart_20")
        sec("Avg Delivery Days Forecast (Ensemble)")
        delay_m = del_df.groupby(del_df["Order_Date"].dt.to_period("M"))["Delivery_Days"].mean().rename("v")
        r_del   = ml_forecast(delay_m.values.astype(float), delay_m.index, 6)
        if r_del:
            fig_d = draw_ensemble_chart(r_del, chart_key="delay_fc", height=250)
            st.plotly_chart(fig_d, use_container_width=True, key="delay_fc")

    with t4:
        sec("Warehouse Shipment Volume + Ensemble Forecast")
        banner("Historical bars show delivered volumes. Dotted lines = ML ensemble forecast. Production-driven inbound plan (from production module) shown below.", "purple")
        wm = del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh = go.Figure()
        wf_rows = []
        for i, wh in enumerate(wm.columns):
            clr = COLORS[i%len(COLORS)]
            r = ml_forecast(wm[wh].values.astype(float), wm.index, 6)
            if r is None:
                fig_wh.add_trace(go.Bar(x=wm.index.to_timestamp(), y=wm[wh], name=wh,
                    marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)")))); continue
            fig_wh.add_trace(go.Bar(x=r["hist_ds"], y=r["hist_y"], name=wh,
                marker=dict(color=clr, opacity=0.85, line=dict(color="rgba(0,0,0,0)"))))
            fig_wh.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=f"{wh} (fcst)",
                mode="lines+markers", line=dict(color=clr, width=2.5, dash="dot"),
                marker=dict(size=8,line=dict(color="#FFFFFF",width=2)), showlegend=False))
            for dt, fc, hi in zip(r["fut_ds"], r["forecast"], r["ci_hi"]):
                wf_rows.append({"Month":dt,"Warehouse":wh,"Forecast":fc,"Upper":hi})
        fig_wh.update_layout(**CD(), height=310, barmode="stack", xaxis=gX(), yaxis=gY(), legend=leg())
        st.plotly_chart(fig_wh, use_container_width=True, key="chart_22")
        if wf_rows:
            tbl_wf = pd.DataFrame(wf_rows)
            tbl_wf["Month"]    = tbl_wf["Month"].dt.strftime("%b %Y")
            tbl_wf["Forecast"] = tbl_wf["Forecast"].round(0).astype(int)
            tbl_wf["Upper"]    = tbl_wf["Upper"].round(0).astype(int)
            tbl_wf.columns     = ["Month","Warehouse","Forecast Units","Upper Bound"]
            st.dataframe(tbl_wf.sort_values(["Month","Warehouse"]), use_container_width=True, hide_index=True)

        # Production-driven inbound plan per warehouse
        sp()
        sec("Production-Driven Inbound Plan per Warehouse")
        if not fwd_plan.empty:
            # Distribute production units to warehouses by their historical share
            wh_share = (del_df.groupby("Warehouse")["Quantity"].sum() /
                        del_df["Quantity"].sum()).to_dict()
            inbound_rows = []
            for _, row in fwd_plan.iterrows():
                for wh, share in wh_share.items():
                    inbound_rows.append({
                        "Month":    row["Month"],
                        "Month_dt": row["Month_dt"],
                        "Warehouse": wh,
                        "Category":  row["Category"],
                        "Inbound_Units": round(row["Prod_Units"] * share),
                        "Proj_Ship_Cost": round(row["Proj_Ship_Cost"] * share),
                    })
            inbound_df = pd.DataFrame(inbound_rows)
            inbound_agg = inbound_df.groupby(["Month_dt","Month","Warehouse"]).agg(
                Inbound_Units=("Inbound_Units","sum"),
                Proj_Ship_Cost=("Proj_Ship_Cost","sum"),
            ).reset_index().sort_values(["Month_dt","Warehouse"])

            fig_inb = go.Figure()
            for i, wh in enumerate(sorted(inbound_agg["Warehouse"].unique())):
                wdf = inbound_agg[inbound_agg["Warehouse"]==wh]
                fig_inb.add_trace(go.Bar(
                    x=wdf["Month"], y=wdf["Inbound_Units"],
                    name=wh, marker=dict(color=COLORS[i%len(COLORS)], line=dict(color="rgba(0,0,0,0)"))))
            fig_inb.update_layout(**CD(), height=280, barmode="group",
                xaxis={**gX(),"tickangle":-25}, yaxis={**gY(),"title":"Planned Inbound Units"}, legend=leg())
            st.plotly_chart(fig_inb, use_container_width=True, key="wh_inbound_chart")
            disp_inb = inbound_agg[["Month","Warehouse","Inbound_Units","Proj_Ship_Cost"]].copy()
            disp_inb.columns = ["Month","Warehouse","Planned Inbound Units","Proj. Ship Cost ₹"]
            st.dataframe(disp_inb, use_container_width=True, hide_index=True)
        else:
            st.info("Production plan not available.")
        sec("Top Products per Warehouse")
        wsel = st.selectbox("Warehouse", sorted(del_df["Warehouse"].unique()))
        tp = del_df[del_df["Warehouse"]==wsel].groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(10)
        fig_tp = go.Figure(go.Bar(x=tp.values, y=tp.index, orientation="h",
            marker=dict(color="#06B6D4", line=dict(color="rgba(0,0,0,0)")),
            text=tp.values, textposition="outside", textfont=dict(color="#333333")))
        fig_tp.update_layout(**CD(), height=300, xaxis=gX(), yaxis=dict(showgrid=False,color="#64748B"))
        st.plotly_chart(fig_tp, use_container_width=True, key="chart_23")

        sp()
        sec("Warehouse Revenue & Cost Efficiency")
        wh_stats = del_df.groupby("Warehouse").agg(
            Revenue=("Net_Revenue","sum"),
            Units=("Quantity","sum"),
            Shipping_Cost=("Shipping_Cost_INR","sum"),
            Orders=("Order_ID","count"),
        ).reset_index()
        wh_stats["Revenue_Per_Unit"]       = (wh_stats["Revenue"]       / wh_stats["Units"].replace(0,1)).round(1)
        wh_stats["Shipping_Cost_Per_Unit"] = (wh_stats["Shipping_Cost"] / wh_stats["Units"].replace(0,1)).round(2)
        wh_stats["Net_Margin_Per_Unit"]    = (wh_stats["Revenue_Per_Unit"] - wh_stats["Shipping_Cost_Per_Unit"]).round(1)
        wh_l, wh_r = st.columns(2, gap="large")
        with wh_l:
            fig_wrev = go.Figure(go.Bar(
                x=wh_stats["Warehouse"], y=wh_stats["Revenue"]/1e6,
                marker=dict(color=COLORS[:len(wh_stats)], line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:.1f}M" for v in wh_stats["Revenue"]/1e6],
                textposition="outside", textfont=dict(color="#333333")))
            fig_wrev.update_layout(**CD(), height=240, xaxis=gX(),
                yaxis={**gY(),"title":"Net Revenue (₹M)"},
                title=dict(text="Revenue by Warehouse", font=dict(size=11,color="#4a5e7a")))
            st.plotly_chart(fig_wrev, use_container_width=True, key="wh_rev_chart")
        with wh_r:
            fig_wcpu = go.Figure(go.Bar(
                x=wh_stats["Warehouse"], y=wh_stats["Shipping_Cost_Per_Unit"],
                marker=dict(color=["#EF4444" if v > wh_stats["Shipping_Cost_Per_Unit"].mean() else "#22C55E"
                                   for v in wh_stats["Shipping_Cost_Per_Unit"]], line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:.1f}" for v in wh_stats["Shipping_Cost_Per_Unit"]],
                textposition="outside", textfont=dict(color="#333333")))
            fig_wcpu.update_layout(**CD(), height=240, xaxis=gX(),
                yaxis={**gY(),"title":"Shipping Cost / Unit (₹)"},
                title=dict(text="Cost per Unit Shipped (red = above avg)", font=dict(size=11,color="#4a5e7a")))
            st.plotly_chart(fig_wcpu, use_container_width=True, key="wh_cpu_chart")
        wh_disp = wh_stats[["Warehouse","Orders","Units","Revenue","Revenue_Per_Unit","Shipping_Cost_Per_Unit","Net_Margin_Per_Unit"]].copy()
        wh_disp["Revenue"] = wh_disp["Revenue"].round(0).astype(int)
        wh_disp.columns = ["Warehouse","Orders","Units","Revenue ₹","Rev/Unit ₹","Ship Cost/Unit ₹","Net Margin/Unit ₹"]
        st.dataframe(wh_disp.sort_values("Revenue ₹", ascending=False), use_container_width=True, hide_index=True)

    with t5:
        sec("Region Performance Overview")
        rs = del_df.groupby("Region").agg(
            Orders=("Order_ID","count"), Revenue=("Net_Revenue","sum"),
            Qty=("Quantity","sum"), Avg_Del=("Delivery_Days","mean"),
            Returns=("Return_Flag","mean")).reset_index().sort_values("Revenue",ascending=False)
        rs["Returns_Pct"] = (rs["Returns"] * 100).round(1)

        met = st.selectbox("Metric", ["Revenue","Orders","Qty","Avg_Del","Return Rate (%)"])
        met_col_map = {"Revenue":"Revenue","Orders":"Orders","Qty":"Qty","Avg_Del":"Avg_Del","Return Rate (%)":"Returns_Pct"}
        met_label_map = {"Revenue":"Revenue (₹)","Orders":"Orders","Qty":"Units Sold","Avg_Del":"Avg Delivery Days","Return Rate (%)":"Return Rate (%)"}
        plot_col = met_col_map[met]
        y_vals = rs[plot_col]
        fig_r = go.Figure(go.Bar(
            x=rs["Region"], y=y_vals,
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(rs))], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.1f}%" if met=="Return Rate (%)" else f"{v:,.0f}" for v in y_vals],
            textposition="outside", textfont=dict(color="#333333")))
        fig_r.update_layout(**CD(), height=290,
            xaxis={**gX(),"tickangle":-25},
            yaxis={**gY(),"title":met_label_map[met]})
        st.plotly_chart(fig_r, use_container_width=True, key="chart_24")
        cl5, cr5 = st.columns(2, gap="large")
        with cl5:
            sec("Best Carrier per Region (Multi-factor Score)")
            bc = best_carr[["Region","Courier_Partner","Avg_Days","Avg_Cost","Score"]].copy()
            bc["Avg_Days"]  = bc["Avg_Days"].round(1)
            bc["Avg_Cost"]  = bc["Avg_Cost"].round(1)
            bc["Score"]     = bc["Score"].round(3)
            bc.columns = ["Region","Best Carrier","Avg Days","Avg Cost ₹","Score (0–1)"]
            st.dataframe(bc.sort_values("Score (0–1)", ascending=False), use_container_width=True, hide_index=True)
        with cr5:
            sec("Region Return Rate Ranking")
            rr = del_df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False)*100
            fig_ret = go.Figure(go.Bar(x=rr.values, y=rr.index, orientation="h",
                marker=dict(color=["#EF4444" if v>20 else "#F59E0B" if v>12 else "#22C55E" for v in rr.values], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in rr.values], textposition="outside", textfont=dict(color="#333333")))
            fig_ret.update_layout(**CD(), height=270, xaxis=gX(), yaxis=dict(showgrid=False,color="#64748B"))
            st.plotly_chart(fig_ret, use_container_width=True, key="chart_25")
        sec("Region Revenue Forecast (Ensemble)")
        top_reg = del_df["Region"].value_counts().head(5).index.tolist()
        fig_rf  = go.Figure()
        for i, reg in enumerate(top_reg):
            s = del_df[del_df["Region"]==reg].groupby(del_df["Order_Date"].dt.to_period("M"))["Net_Revenue"].sum().rename("v")
            r = ml_forecast(s.values.astype(float), s.index, 6)
            if r is None: continue
            fig_rf.add_trace(go.Scatter(x=r["hist_ds"], y=r["hist_y"], name=reg,
                line=dict(color=COLORS[i],width=1.5,dash="solid"), opacity=0.65, showlegend=False))
            fig_rf.add_trace(go.Scatter(x=r["fut_ds"], y=r["forecast"], name=reg,
                mode="lines+markers", line=dict(color=COLORS[i],width=2.5,dash="dot"),
                marker=dict(size=8,line=dict(color="#FFFFFF",width=2))))
        fig_rf.update_layout(**CD(), height=270, xaxis=gX(), yaxis=gY(), legend=leg())
        st.plotly_chart(fig_rf, use_container_width=True, key="chart_26")

st.sidebar.markdown("""<div style='padding:18px 0 26px'>
  <div style='font-family:DM Mono,monospace;font-size:0.58rem;letter-spacing:0.16em;
       text-transform:uppercase;color:#4a5e7a;margin-bottom:5px'>Supply Chain Platform</div>
  <div style='font-family:Outfit,sans-serif;font-size:1.75rem;font-weight:900;
       letter-spacing:-0.04em;background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow</div>
  <div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#4a5e7a;
       margin-top:2px;letter-spacing:0.05em'>D2D INTELLIGENCE · 3-MODEL ENSEMBLE</div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style='font-family:DM Mono,monospace;font-size:0.6rem;
    color:#4a5e7a;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em'>
    Forecast Engine</div>
    <div style='font-size:0.72rem;color:#334155;background:#f8faff;border:1px solid #c7d7fd;
    border-radius:10px;padding:10px 12px;margin-bottom:16px;line-height:1.8'>
    <span style='color:#3B82F6;font-weight:700'>① Ridge</span> +
    <span style='color:#22C55E;font-weight:700'>② Random Forest</span> +
    <span style='color:#F59E0B;font-weight:700'>③ Grad Boost</span>
    <br><span style='color:#8B5CF6;font-weight:700'>④ Ensemble</span>
    via inverse-RMSE blend
</div>""", unsafe_allow_html=True)

PAGES = {
    "Overview":               page_overview,
    "Demand Forecasting":     page_demand,
    "Inventory Optimisation": page_inventory,
    "Production Planning":    page_production,
    "Logistics Optimisation": page_logistics,
    "Decision Chatbot":       page_chatbot,
}

sel = st.sidebar.radio("", list(PAGES.keys()))
PAGES[sel]()
