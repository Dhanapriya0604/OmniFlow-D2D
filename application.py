import streamlit as st
st.set_page_config(
    page_title="OmniFlow D2D Intelligence", page_icon="⬡",
    layout="wide", initial_sidebar_state="expanded"
)

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap');
    :root {
        --bg:#f8fafc;--text:#0f172a;--muted:#475569;--primary:#1e3a8a;
        --border:#e5e7eb;--accent:#e0e7ff;--panel:#ffffff;
    }
    html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
    section.main>div{animation:fadeIn 0.35s ease-in-out;}
    @keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
    .page-title{font-size:32px;font-weight:900;margin-bottom:4px;color:#0f172a;}
    .page-subtitle{font-size:13px;color:#64748b;margin-bottom:20px;}
    .section-title{font-size:18px;font-weight:800;margin:24px 0 10px;color:#0f172a;}
    .section-line{height:2px;background:linear-gradient(90deg,#e5e7eb,transparent);margin-bottom:14px;}
    .metric-card{background:linear-gradient(160deg,#eef4ff,#ffffff);padding:16px 14px;
        text-align:center;border-radius:14px;border:1px solid #c7d7fd;
        box-shadow:0 4px 14px rgba(30,58,138,0.10);transition:all .22s ease;}
    .metric-card:hover{transform:translateY(-4px);box-shadow:0 12px 28px rgba(30,58,138,0.18);}
    .metric-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;font-family:'DM Mono',monospace;}
    .metric-value{font-size:26px;font-weight:900;color:#1e3a8a;line-height:1.2;margin-top:4px;}
    .metric-sub{font-size:10px;color:#94a3b8;margin-top:3px;}
    .card{background:white;padding:20px;border-radius:14px;border:1px solid #e5e7eb;
        box-shadow:0 6px 20px rgba(0,0,0,0.07);transition:all .22s;}
    .card:hover{transform:translateY(-3px);box-shadow:0 14px 32px rgba(0,0,0,0.13);}
    .info-banner{border-radius:10px;padding:12px 14px;margin:8px 0;font-size:12.5px;line-height:1.6;}
    .banner-teal{background:#f0fdfa;border:1px solid #5eead4;}
    .banner-amber{background:#fffbeb;border:1px solid #fbbf24;}
    .banner-coral{background:#fff1f2;border:1px solid #fb7185;}
    .banner-mint{background:#ecfdf5;border:1px solid #34d399;}
    .banner-purple{background:#faf5ff;border:1px solid #a78bfa;}
    .banner-sky{background:#eff6ff;border:1px solid #93c5fd;}
    .badge{display:inline-block;padding:4px 9px;font-size:10.5px;font-weight:600;
        border-radius:7px;margin-right:5px;margin-bottom:3px;}
    .badge-amber{background:#fff7ed;color:#ea580c}
    .badge-teal{background:#ecfeff;color:#0891b2}
    .badge-lav{background:#f5f3ff;color:#7c3aed}
    .badge-coral{background:#fff1f2;color:#e11d48}
    .badge-sky{background:#eff6ff;color:#2563eb}
    .badge-mint{background:#ecfdf5;color:#059669}
    .badge-purple{background:#faf5ff;color:#7c3aed}
    .model-pill{display:inline-block;padding:3px 9px;font-size:10px;font-weight:700;
        border-radius:20px;margin-right:5px;margin-bottom:3px;}
    .pill-ridge{background:#eff6ff;color:#1d4ed8}
    .pill-rf{background:#f0fdf4;color:#15803d}
    .pill-gb{background:#fef9c3;color:#a16207}
    .pill-ensemble{background:#fdf4ff;color:#7e22ce}
    .about-section{background:white;border:1px solid #e5e7eb;border-radius:16px;
        padding:22px 26px;margin-bottom:18px;box-shadow:0 6px 20px rgba(0,0,0,0.06);}
    .pipeline-box{background:white;border:1px solid #c7d7fd;border-radius:14px;
        padding:18px 22px;text-align:center;min-width:105px;
        font-weight:700;font-size:12px;font-family:'DM Mono',monospace;color:#0f172a;}
    .pipeline-sub{font-size:9.5px;font-weight:400;color:#64748b;margin-top:3px;display:block;}
    .chat-user-bubble{background:#1e3a8a;color:white;padding:10px 14px;border-radius:14px;
        max-width:72%;margin-left:auto;font-size:13.5px;}
    .chat-ai-bubble{background:#f1f5f9;padding:12px 15px;border-radius:14px;
        max-width:82%;font-size:13px;border:1px solid #e5e7eb;}
    .alert-item{border-radius:9px;padding:9px 12px;margin-bottom:7px;border:1px solid #e5e7eb;}
    .alert-critical{background:#fef2f2;}
    .alert-warn{background:#fff7ed;}
    .model-quality-card{background:white;border-radius:14px;padding:18px;
        border:1px solid #e5e7eb;box-shadow:0 4px 16px rgba(0,0,0,0.07);}
    .ensemble-card{background:linear-gradient(135deg,#f8faff,#ffffff);border-radius:14px;
        padding:16px;border:1px solid #c7d7fd;box-shadow:0 4px 16px rgba(30,58,138,0.08);margin-bottom:12px;}
    .stTabs [data-baseweb="tab"]{background:#f1f5f9;border-radius:10px;padding:9px 16px;
        font-weight:600;color:#475569;font-size:13px;}
    .stTabs [aria-selected="true"]{background:#e0e7ff;color:#1e3a8a;
        box-shadow:0 4px 14px rgba(30,58,138,0.18);}
    .block-container{padding-top:1.8rem;padding-bottom:2rem;}
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
import os, requests as _requests

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_ecommerce_orders.csv")
COLORS    = ["#1565C0","#2E7D32","#E65100","#C62828","#6A1B9A","#00695C"]
MODEL_COLORS = {"Ridge":"#3B82F6","RandomForest":"#22C55E","GradBoost":"#F59E0B","Ensemble":"#8B5CF6"}

# ─── Layout helpers ──────────────────────────────────────────────────────────

def CD():
    return dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155",family="Inter,sans-serif",size=11),
        margin=dict(l=30,r=50,t=42,b=30))
def gY(): return dict(showgrid=True,gridcolor="rgba(0,0,0,0.06)",zeroline=False,tickcolor="#64748b")
def gX(): return dict(showgrid=False,zeroline=False,tickcolor="#64748b")
def leg(): return dict(bgcolor="rgba(255,255,255,0.95)",bordercolor="#E0E0E0",borderwidth=1,font=dict(color="#334155",size=10))

def kpi(col, label, value, cls="sky", sub=""):
    col.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='{"color:#dc2626" if cls=="coral" else "color:#1e3a8a" if cls=="sky" else "color:#059669" if cls=="mint" else "color:#d97706" if cls=="amber" else "color:#7c3aed"}'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(label, emoji=""):
    st.markdown(f"""<div class='section-title'>{emoji} {label}</div>
    <div class='section-line'></div>""", unsafe_allow_html=True)

def banner(html, cls="teal"):
    st.markdown(f"<div class='info-banner banner-{cls}'>{html}</div>", unsafe_allow_html=True)

def sp(n=1):
    st.markdown(f"<div style='height:{n*12}px'></div>", unsafe_allow_html=True)

# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df["Region"]      = df["Region"].replace("Pune","Maharashtra")
    df["YearMonth"]   = df["Order_Date"].dt.to_period("M")
    df["Year"]        = df["Order_Date"].dt.year
    df["Month_Num"]   = df["Order_Date"].dt.month
    df["Net_Revenue"] = np.where(df["Return_Flag"]==1, 0.0, df["Revenue_INR"])
    df["Net_Qty"]     = np.where(df["Return_Flag"]==1, 0,   df["Quantity"])
    df.loc[df["Order_Status"]=="Cancelled","Delivery_Days"] = np.nan
    return df

@st.cache_data
def get_ops(df):   return df[df["Order_Status"].isin(["Delivered","Shipped"])].copy()
@st.cache_data
def get_delivered(df): return df[df["Order_Status"]=="Delivered"].copy()

# ─── ML Forecast engine ───────────────────────────────────────────────────────

def _to_ts(idx):
    return idx.to_timestamp() if hasattr(idx,"to_timestamp") else pd.DatetimeIndex(idx)

def _build_features(n_hist, n_future, ds_hist, regime_idx):
    n = n_hist + n_future
    t = np.arange(n)
    ts = _to_ts(ds_hist)
    h_months = ts.month.values
    last_m = int(h_months[-1])
    f_months = np.array([(last_m+i-1)%12+1 for i in range(1,n_future+1)])
    mn = np.concatenate([h_months, f_months])
    regime = (t >= regime_idx).astype(float)
    q = np.where(mn<=3,1,np.where(mn<=6,2,np.where(mn<=9,3,4)))
    return np.column_stack([
        t, t**2,
        np.sin(2*np.pi*mn/12), np.cos(2*np.pi*mn/12),
        np.sin(4*np.pi*mn/12), np.cos(4*np.pi*mn/12),
        np.sin(6*np.pi*mn/12), np.cos(6*np.pi*mn/12),
        regime, t*regime,
        (q==1).astype(float),(q==2).astype(float),(q==3).astype(float),
        np.log1p(t),
    ])

def _detect_regime(vals, min_idx=6):
    best_idx,best_ratio = min_idx,1.0
    for i in range(min_idx,len(vals)-min_idx):
        r = vals[i:].mean()/(vals[:i].mean()+1e-9)
        if r>best_ratio: best_ratio=r;best_idx=i
    return best_idx

def ml_forecast(vals, ds_idx, n_future=6):
    n = len(vals)
    if n < 6: return None
    regime_idx = _detect_regime(vals)
    X_all  = _build_features(n, n_future, ds_idx, regime_idx)
    X_hist = X_all[:n]; X_fut = X_all[n:]

    models = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100,max_depth=3,min_samples_leaf=4,random_state=42),
        "GradBoost":    GradientBoostingRegressor(n_estimators=80,max_depth=2,learning_rate=0.08,subsample=0.9,random_state=42),
    }
    h = 4
    n_folds = min(3, n//6); fold_size=2
    fold_rmses = {m:[] for m in models}
    for fold in range(n_folds):
        te_end=n-fold*fold_size; te_start=te_end-fold_size
        if te_start<6: break
        Xtr=X_hist[:te_start]; ytr=vals[:te_start]
        Xte=X_hist[te_start:te_end]; yte=vals[te_start:te_end]
        sc=StandardScaler(); sc.fit(Xtr)
        for mname, mdl in [("Ridge",Ridge(alpha=1.0)),
            ("RandomForest",RandomForestRegressor(n_estimators=100,max_depth=3,min_samples_leaf=4,random_state=42)),
            ("GradBoost",GradientBoostingRegressor(n_estimators=80,max_depth=2,learning_rate=0.08,subsample=0.9,random_state=42))]:
            mdl.fit(sc.transform(Xtr),ytr)
            ep=np.maximum(mdl.predict(sc.transform(Xte)),0)
            fold_rmses[mname].append(np.sqrt(mean_squared_error(yte,ep)))

    model_rmses={}; model_metrics={}
    for mname in models:
        avg_rmse = np.mean(fold_rmses[mname]) if fold_rmses[mname] else 1.0
        nrmse_v  = avg_rmse/np.mean(vals) if np.mean(vals)>0 else 0
        r2_v     = max(0,1-(avg_rmse**2/(np.var(vals)+1e-9)))
        model_rmses[mname]=avg_rmse
        model_metrics[mname]={"rmse":avg_rmse,"nrmse":nrmse_v,"mae":avg_rmse*0.8,"r2":r2_v}

    Xtr=X_hist[:-h]; ytr=vals[:-h]; Xte=X_hist[-h:]; yte=vals[-h:]
    sc_h=StandardScaler(); sc_h.fit(Xtr)
    eval_preds={}
    for mname,mdl in models.items():
        mdl.fit(sc_h.transform(Xtr),ytr)
        eval_preds[mname]=np.maximum(mdl.predict(sc_h.transform(Xte)),0)

    inv_rmse={m:1.0/(r+1e-9) for m,r in model_rmses.items()}
    tot=sum(inv_rmse.values()); weights={m:v/tot for m,v in inv_rmse.items()}
    ypred_eval=sum(weights[m]*eval_preds[m] for m in models)

    sc2=StandardScaler(); sc2.fit(X_hist)
    fitted_pm={}; forecast_pm={}
    for mname,mdl in models.items():
        mdl.fit(sc2.transform(X_hist),vals)
        fitted_pm[mname]=np.maximum(mdl.predict(sc2.transform(X_hist)),0)
        forecast_pm[mname]=np.maximum(mdl.predict(sc2.transform(X_fut)),0)

    ens_fitted   = sum(weights[m]*fitted_pm[m]   for m in models)
    ens_forecast = sum(weights[m]*forecast_pm[m] for m in models)

    residuals=vals-ens_fitted; resid_std=residuals.std()
    ss_res=np.sum(residuals**2); ss_tot=np.sum((vals-np.mean(vals))**2)
    r2_e=1-ss_res/ss_tot if ss_tot>0 else 0
    rmse_e=np.sqrt(mean_squared_error(yte,ypred_eval))
    nrmse_e=rmse_e/np.mean(yte) if np.mean(yte)>0 else 0
    mae_e=mean_absolute_error(yte,ypred_eval)
    model_metrics["Ensemble"]={"rmse":rmse_e,"nrmse":nrmse_e,"mae":mae_e,"r2":r2_e}

    ts_idx=_to_ts(ds_idx)
    last_dt=ts_idx[-1]
    fut_dates=pd.date_range(last_dt+pd.offsets.MonthBegin(1),periods=n_future,freq="MS")
    log_std=np.log1p(resid_std/(np.mean(vals)+1e-9))
    ci_lo=np.maximum(ens_forecast*np.exp(-1.645*log_std*np.sqrt(np.arange(1,n_future+1))),0)
    ci_hi=ens_forecast*np.exp(1.645*log_std*np.sqrt(np.arange(1,n_future+1)))

    return dict(hist_ds=ts_idx,hist_y=vals,fitted=ens_fitted,
        fitted_per_model=fitted_pm,forecast_per_model=forecast_pm,
        fut_ds=fut_dates,forecast=ens_forecast,ci_lo=ci_lo,ci_hi=ci_hi,
        rmse=rmse_e,nrmse=nrmse_e,mae=mae_e,r2=r2_e,resid_std=resid_std,
        eval_actual=yte,eval_pred=ypred_eval,eval_ds=ts_idx[-h:],
        model_metrics=model_metrics,weights={m:weights[m] for m in models})

# ─── Chart helper ─────────────────────────────────────────────────────────────

def ensemble_chart(res, chart_key, height=300, title="", show_models=True):
    fig = go.Figure()
    fig.add_vrect(x0=res["fut_ds"][0], x1=res["fut_ds"][-1],
        fillcolor="rgba(139,92,246,0.04)", layer="below", line_width=0)
    fig.add_vline(x=res["fut_ds"][0], line_dash="dash",
        line_color="rgba(139,92,246,0.4)", line_width=1.5)
    fig.add_annotation(x=res["fut_ds"][0], y=1, yref="paper", yanchor="top", xanchor="left",
        text=" Forecast →", showarrow=False,
        font=dict(color="#8B5CF6",size=9,family="DM Mono"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(139,92,246,0.4)", borderwidth=1, borderpad=3)

    x_ci=list(res["fut_ds"])+list(res["fut_ds"])[::-1]
    y_ci=list(res["ci_hi"])+list(res["ci_lo"])[::-1]
    fig.add_trace(go.Scatter(x=x_ci,y=y_ci,fill="toself",
        fillcolor="rgba(139,92,246,0.07)",line=dict(color="rgba(0,0,0,0)"),name="90% CI"))
    fig.add_trace(go.Scatter(x=res["hist_ds"],y=res["hist_y"],name="Actual",
        line=dict(color="#4a5e7a",width=2.2),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"))
    if show_models and "fitted_per_model" in res:
        for mname,clr,dash in [("Ridge","#3B82F6","dot"),("RandomForest","#22C55E","dashdot"),("GradBoost","#F59E0B","longdash")]:
            if mname in res["fitted_per_model"]:
                fig.add_trace(go.Scatter(x=res["hist_ds"],y=res["fitted_per_model"][mname],
                    name=f"{mname} fit",line=dict(color=clr,width=1.2,dash=dash),
                    opacity=0.5,visible="legendonly"))
    fig.add_trace(go.Scatter(x=res["hist_ds"],y=res["fitted"],name="Ensemble fit",
        line=dict(color="#8B5CF6",width=1.5,dash="dot"),opacity=0.55))
    if show_models and "forecast_per_model" in res:
        for mname,clr,dash in [("Ridge","#3B82F6","dot"),("RandomForest","#22C55E","dashdot"),("GradBoost","#F59E0B","longdash")]:
            if mname in res["forecast_per_model"]:
                fig.add_trace(go.Scatter(x=res["fut_ds"],y=res["forecast_per_model"][mname],
                    name=f"{mname} fc",line=dict(color=clr,width=1.8,dash=dash),
                    mode="lines+markers",marker=dict(size=5,color=clr),visible="legendonly"))
    fig.add_trace(go.Scatter(x=res["fut_ds"],y=res["forecast"],name="Ensemble Forecast",
        line=dict(color="#8B5CF6",width=2.8,dash="dot"),mode="lines+markers",
        marker=dict(size=8,color="#8B5CF6",line=dict(color="#FFFFFF",width=2)),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=res["eval_ds"],y=res["eval_pred"],name="Eval (ensemble)",
        mode="markers",marker=dict(size=9,color="#EF4444",symbol="x",line=dict(color="#FFFFFF",width=2))))
    fig.update_layout(**CD(),height=height,xaxis=gX(),yaxis=gY(),legend=leg(),
        title=dict(text=title,font=dict(color="#64748b",size=11)))
    return fig

def model_grade(nrmse,r2):
    acc=max(0,round((1-nrmse)*100,1))
    if   nrmse<0.10 and r2>=0.95: g,l,icon="A+","Excellent","✅"
    elif nrmse<0.15 and r2>=0.90: g,l,icon="A","Very Good","✅"
    elif nrmse<0.20 and r2>=0.85: g,l,icon="B+","Good","🟦"
    elif nrmse<0.25 and r2>=0.75: g,l,icon="B","Acceptable","⚠️"
    elif nrmse<0.35 and r2>=0.60: g,l,icon="C","Weak","⚠️"
    else:                          g,l,icon="D","Poor","🔴"
    return g,l,icon,acc

def render_model_quality(res):
    g,l,icon,acc = model_grade(res["nrmse"],res["r2"])
    if "model_metrics" in res:
        st.markdown("<div class='ensemble-card'>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:11px;font-weight:700;color:#4a5e7a;
            letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px'>
            🤖 Individual Model Performance (Walk-Forward CV)</div>""", unsafe_allow_html=True)
        mm=res["model_metrics"]
        cols=st.columns(4)
        for col,(mname,pcls,clr) in zip(cols,[("Ridge","pill-ridge","#3B82F6"),("RandomForest","pill-rf","#22C55E"),
                                              ("GradBoost","pill-gb","#F59E0B"),("Ensemble","pill-ensemble","#8B5CF6")]):
            if mname in mm:
                m=mm[mname]
                col.markdown(f"""<div style='text-align:center;padding:10px;border-radius:10px;
                    border:1px solid #e5e7eb;background:white'>
                    <div class='model-pill {pcls}'>{mname}</div>
                    <div style='font-size:10px;color:#64748b;margin-top:5px'>RMSE</div>
                    <div style='font-size:18px;font-weight:800;color:{clr}'>{m["rmse"]:.1f}</div>
                    <div style='font-size:10px;color:#94a3b8'>NRMSE {m["nrmse"]*100:.1f}% · R² {m["r2"]:.3f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        w=res.get("weights",{})
        if w:
            tot=sum(w.values())
            st.markdown(f"""<div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:8px;
                padding:8px 12px;font-size:11px;margin:6px 0'>
                <b style='color:#1e3a8a'>Ensemble blend (inverse-RMSE):</b>
                <span class='model-pill pill-ridge'>Ridge {w.get("Ridge",0)/tot*100:.0f}%</span>
                <span class='model-pill pill-rf'>RF {w.get("RandomForest",0)/tot*100:.0f}%</span>
                <span class='model-pill pill-gb'>GB {w.get("GradBoost",0)/tot*100:.0f}%</span>
            </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5=st.columns(5)
    kpi(c1,"RMSE",f"{res['rmse']:.1f}","sky","hold-out")
    kpi(c2,"NRMSE",f"{res['nrmse']*100:.1f}%","sky","normalised")
    kpi(c3,"MAE",f"{res['mae']:.1f}","sky","mean abs err")
    kpi(c4,"R² Score",f"{res['r2']:.3f}","sky","fit quality")
    kpi(c5,"Accuracy",f"{acc:.1f}%","mint","1 − NRMSE")
    sp(0.5)
    st.markdown(f"""<div class='model-quality-card'>
      <div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>
        <div style='font-size:22px'>{icon}</div>
        <div>
          <div style='font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#64748b;margin-bottom:3px'>Ensemble Quality Grade</div>
          <div style='font-size:22px;font-weight:900'>{g} <span style='font-size:14px;font-weight:600;color:#475569'>{l}</span></div>
        </div>
        <div style='margin-left:auto;text-align:right'>
          <div style='font-size:10px;color:#64748b'>Forecast Accuracy</div>
          <div style='font-size:28px;font-weight:900;color:#1e3a8a'>{acc:.1f}%</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    sp(0.5)

# ─── Module computations ──────────────────────────────────────────────────────

@st.cache_data
def compute_inventory(order_cost=500, hold_pct=0.20, lead_time=7, z=1.65):
    """
    Inventory optimisation that:
    1. Uses ACTUAL Current_Stock_Units from dataset (real stock snapshot per SKU)
    2. Uses dataset Reorder_Point as the ROP baseline (already set by the business)
    3. Computes EOQ from ML demand forecast (category-level forward demand scaled to SKU)
    4. Computes Safety Stock from demand variability and lead time variability
    5. Status = Actual Stock vs ROP and SS thresholds
    """
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")
    del_ops=df[df["Order_Status"]=="Delivered"].copy()

    # Lead time variability by category from actual delivery data
    lt_std_map=del_ops.groupby("Category")["Delivery_Days"].std().fillna(1.0).to_dict()

    # Monthly demand per SKU from ops history
    sku_monthly=(ops.groupby(["SKU_ID","YM"])["Net_Qty"].sum().reset_index().sort_values(["SKU_ID","YM"]))

    # Actual stock snapshot: latest values per SKU
    df_sorted=df.sort_values("Order_Date")
    sku_snapshot=df_sorted.groupby("SKU_ID").agg(
        actual_stock=("Current_Stock_Units","last"),
        dataset_rop=("Reorder_Point","last"),        # business-set ROP from dataset
        dataset_status=("Stock_Status","last"),
        Product_Name=("Product_Name","first"),
        Category=("Category","first"),
        avg_price=("Sell_Price","mean"),
        total_qty=("Net_Qty","sum")
    ).reset_index()

    # Category-level demand forecast (6 months forward) for forward-looking EOQ
    cat_monthly=ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
    cat_forecast={}
    cat_hist_avg={}
    for cat in cat_monthly.columns:
        cat_hist_avg[cat]=float(cat_monthly[cat].mean())
        res=ml_forecast(cat_monthly[cat].values.astype(float),cat_monthly.index,6)
        if res is not None:
            cat_forecast[cat]=float(np.mean(res["forecast"]))  # avg forecast monthly demand

    rows=[]
    for _,sk in sku_snapshot.iterrows():
        sku=sk["SKU_ID"]; cat=sk["Category"]
        skd=sku_monthly[sku_monthly["SKU_ID"]==sku].sort_values("YM")
        demands=skd["Net_Qty"].values
        if len(demands)<1: continue

        # Historical demand stats
        avg_d=float(np.mean(demands))
        std_d=float(np.std(demands)) if len(demands)>1 else avg_d*0.25
        peak_d=float(np.max(demands))

        # Forward-looking demand: blend historical avg with forecast-scaled estimate
        if cat in cat_forecast and cat in cat_hist_avg and cat_hist_avg[cat]>0:
            sku_share=avg_d/cat_hist_avg[cat]
            fc_monthly=cat_forecast[cat]*sku_share
            # Conservative blend: 50% historical, 30% peak, 20% forecast
            econ_d=avg_d*0.50+peak_d*0.30+fc_monthly*0.20
        else:
            econ_d=avg_d*0.60+peak_d*0.40

        daily_d=avg_d/30.0
        ann_d=econ_d*12
        uc=max(float(sk["avg_price"]),1.0)

        # EOQ (Wilson formula with blended annual demand)
        eoq=max(int(np.sqrt(2*ann_d*order_cost/(uc*hold_pct))) if ann_d>0 else 10, 1)

        # Safety Stock: z * sqrt(LT*sigma_d^2 + D^2*sigma_LT^2) using user-set lead_time
        daily_std=std_d/np.sqrt(30)
        lt_std=lt_std_map.get(cat, 1.0)
        ss=max(int(z*np.sqrt(lead_time*daily_std**2+daily_d**2*lt_std**2)), 0)

        # Use dataset ROP as the primary trigger (reflects actual business policy)
        # Augment with computed ROP if it's higher (conservative)
        computed_rop=max(int(daily_d*lead_time+ss), 1)
        rop=max(int(sk["dataset_rop"]), computed_rop)

        # ACTUAL current stock from dataset
        current_stock=int(sk["actual_stock"])

        # Status: compare actual stock to ROP and SS thresholds
        if current_stock<=ss:
            status="🔴 Critical"
        elif current_stock<rop:
            status="🟡 Low"
        elif current_stock>rop+2*eoq:
            status="🟢 Overstocked"
        else:
            status="🟢 Adequate"

        # Days of stock remaining at historical avg demand
        days_stock=round(current_stock/daily_d, 1) if daily_d>0 else 999

        # Stockout cost for critical SKUs
        margin_rate=0.20
        daily_margin=daily_d*uc*margin_rate
        days_exposed=max(lead_time-(current_stock/daily_d if daily_d>0 else 0), 0)
        stockout_cost=round(daily_margin*days_exposed, 0) if status=="🔴 Critical" else 0

        rows.append({"SKU_ID":sku,"Product_Name":sk["Product_Name"],"Category":cat,
            "Monthly_Avg":round(avg_d,1),
            "Monthly_Std":round(std_d,1),
            "Forecast_Avg":round(econ_d,1),
            "EOQ":eoq,"SS":ss,"ROP":rop,
            "Current_Stock":current_stock,
            "Days_of_Stock":days_stock,
            "Status":status,
            "Dataset_Status":sk["dataset_status"],
            "Unit_Price":round(uc,0),
            "Annual_Demand":round(ann_d,0),
            "Stockout_Cost_Day":stockout_cost,
            "Total_Revenue":round(float(sk["total_qty"])*uc,0)})

    inv_df=pd.DataFrame(rows)
    if inv_df.empty: return inv_df
    inv_df=inv_df.sort_values("Total_Revenue",ascending=False).reset_index(drop=True)
    cum_pct=inv_df["Total_Revenue"].cumsum()/inv_df["Total_Revenue"].sum()*100
    inv_df["ABC"]=np.where(cum_pct<=70,"A",np.where(cum_pct<=90,"B","C"))
    return inv_df

@st.cache_data
def compute_production(cap_mult=1.0, buffer_pct=0.15):
    """
    Production plan fed by:
    - ML demand forecast per category (6 months forward)
    - Inventory replenishment needs (critical/low SKUs from actual stock vs ROP)
    - Safety buffer on top
    """
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")
    inv=compute_inventory()
    cat_monthly=ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
    ds_index=cat_monthly.index; rows=[]
    for cat in cat_monthly.columns:
        vals=cat_monthly[cat].values.astype(float)
        res=ml_forecast(vals,ds_index)
        if res is None: continue

        # Replenishment needed: sum of (ROP - Current_Stock) for critical/low SKUs
        cat_inv=inv[inv["Category"]==cat]
        crit_skus=cat_inv[cat_inv["Status"]=="🔴 Critical"]
        low_skus=cat_inv[cat_inv["Status"]=="🟡 Low"]

        # Replenishment gap = how many units below ROP per SKU (× EOQ to order)
        crit_gap=float((crit_skus["ROP"]-crit_skus["Current_Stock"]).clip(lower=0).sum())
        low_gap=float((low_skus["ROP"]-low_skus["Current_Stock"]).clip(lower=0).sum())

        # Distribute replenishment boost: 60% in month 1, 40% in month 2
        boost_schedule={0:0.60, 1:0.40}
        for i,(dt,fc) in enumerate(zip(res["fut_ds"],res["forecast"])):
            bf=boost_schedule.get(i,0.0)
            crit_boost=crit_gap*bf
            low_boost=low_gap*bf*0.5  # low priority gets half weight
            net_prod=max(fc+crit_boost+low_boost,0)*cap_mult
            prod=net_prod*(1+buffer_pct)
            rows.append({"Month_dt":dt,"Month":dt.strftime("%b %Y"),"Category":cat,
                "Demand_Forecast":round(fc,0),
                "Crit_Boost":round(crit_boost,0),
                "Low_Boost":round(low_boost,0),
                "Buffer":round(prod-net_prod,0),
                "Production":round(prod,0),
                "CI_Lo":round(res["ci_lo"][i],0),
                "CI_Hi":round(res["ci_hi"][i],0)})
    return pd.DataFrame(rows)

@st.cache_data
def compute_logistics(w_speed=0.40,w_cost=0.35,w_returns=0.25):
    df=load_data(); del_df=get_delivered(df); plan=compute_production()
    carr=del_df.groupby("Courier_Partner").agg(
        Orders=("Order_ID","count"),Avg_Days=("Delivery_Days","mean"),
        Avg_Cost=("Shipping_Cost_INR","mean"),Total_Cost=("Shipping_Cost_INR","sum"),
        Return_Rate=("Return_Flag","mean")).reset_index()
    for col,wt in [("Avg_Days",w_speed),("Avg_Cost",w_cost),("Return_Rate",w_returns)]:
        mn=carr[col].min(); mx=carr[col].max()
        carr[f"Norm_{col}"]=1-(carr[col]-mn)/(mx-mn+1e-9)
    carr["Perf_Score"]=(w_speed*carr["Norm_Avg_Days"]+w_cost*carr["Norm_Avg_Cost"]+w_returns*carr["Norm_Return_Rate"]).round(3)
    carr["Delay_Index"]=(carr["Avg_Days"]/carr["Avg_Days"].min()).round(2)
    region_carr=(del_df.groupby(["Region","Courier_Partner"]).agg(
        Avg_Days=("Delivery_Days","mean"),Avg_Cost=("Shipping_Cost_INR","mean"),
        Return_Rate=("Return_Flag","mean"),Orders=("Order_ID","count")).reset_index())
    for col,wt in [("Avg_Days",w_speed),("Avg_Cost",w_cost),("Return_Rate",w_returns)]:
        mn=region_carr[col].min(); mx=region_carr[col].max()
        region_carr[f"Norm_{col}"]=1-(region_carr[col]-mn)/(mx-mn+1e-9)
    region_carr["Score"]=w_speed*region_carr["Norm_Avg_Days"]+w_cost*region_carr["Norm_Avg_Cost"]+w_returns*region_carr["Norm_Return_Rate"]
    best=(region_carr.sort_values("Score",ascending=False).groupby("Region").first().reset_index()
          [["Region","Courier_Partner","Avg_Days","Avg_Cost","Score"]])
    cheapest=(del_df.groupby(["Region","Courier_Partner"])
              .agg(avg_cost=("Shipping_Cost_INR","mean"),orders=("Order_ID","count"))
              .reset_index().sort_values("avg_cost").groupby("Region").first().reset_index()
              .rename(columns={"Courier_Partner":"Optimal_Carrier","avg_cost":"Min_Avg_Cost"}))
    region_costs=(del_df.groupby("Region").agg(
        Current_Avg_Cost=("Shipping_Cost_INR","mean"),
        Orders=("Order_ID","count"),Total_Spend=("Shipping_Cost_INR","sum")).reset_index())
    opt=region_costs.merge(cheapest[["Region","Optimal_Carrier","Min_Avg_Cost"]],on="Region")
    opt["Potential_Saving"]=((opt["Current_Avg_Cost"]-opt["Min_Avg_Cost"])*opt["Orders"]).round(0)
    opt["Saving_Pct"]=((opt["Current_Avg_Cost"]-opt["Min_Avg_Cost"])/opt["Current_Avg_Cost"]*100).round(1)
    avg_ship_unit=max(del_df["Shipping_Cost_INR"].sum()/del_df["Quantity"].replace(0,np.nan).sum(),1.0)
    avg_units_ord=max(del_df["Quantity"].mean(),1.0)
    fwd_rows=[]
    if not plan.empty:
        for _,row in plan.iterrows():
            fwd_rows.append({"Month_dt":row["Month_dt"],"Month":row["Month"],"Category":row["Category"],
                "Prod_Units":int(row["Production"]),"Proj_Orders":int(round(row["Production"]/avg_units_ord)),
                "Proj_Ship_Cost":int(round(row["Production"]*avg_ship_unit,0)),
                "CI_Lo_Units":int(row["CI_Lo"]),"CI_Hi_Units":int(row["CI_Hi"])})
    return carr, best, opt, pd.DataFrame(fwd_rows)

# ─── Context builder for chatbot ──────────────────────────────────────────────

def build_context():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")
    m_orders=ops.groupby("YM")["Order_ID"].count().rename("v")
    m_qty=ops.groupby("YM")["Net_Qty"].sum().rename("v")
    m_rev=ops.groupby("YM")["Net_Revenue"].sum().rename("v")
    r_ord=ml_forecast(m_orders.values.astype(float),m_orders.index,6)
    r_rev=ml_forecast(m_rev.values.astype(float),m_rev.index,6)
    r_qty=ml_forecast(m_qty.values.astype(float),m_qty.index,6)
    def fc_str(r,fmt):
        if r is None: return "N/A"
        return "; ".join([f"{d.strftime('%b%Y')}:{fmt(v)}" for d,v in zip(r["fut_ds"],r["forecast"])])
    inv=compute_inventory(); plan=compute_production()
    carr,best_carr,opt,fwd_plan=compute_logistics()
    n_crit=(inv["Status"]=="🔴 Critical").sum(); n_low=(inv["Status"]=="🟡 Low").sum()
    crit_prods=", ".join(inv[inv["Status"]=="🔴 Critical"]["Product_Name"].head(5).tolist())
    total_stockout=inv["Stockout_Cost_Day"].sum()
    abc_str=", ".join([f"{k}:{v} SKUs" for k,v in sorted(inv["ABC"].value_counts().to_dict().items())])
    prod_sum=plan.groupby("Category")["Production"].sum().to_dict() if not plan.empty else {}
    prod_str=", ".join([f"{k}:{v:.0f}u" for k,v in prod_sum.items()])
    peak_mo=plan.groupby("Month_dt")["Production"].sum().idxmax().strftime("%b %Y") if not plan.empty else "N/A"
    carr_str="; ".join([f"{r['Courier_Partner']}: {r['Orders']}ord, {r['Avg_Days']:.1f}d, ₹{r['Avg_Cost']:.0f}/ship, score:{r['Perf_Score']:.3f}" for _,r in carr.iterrows()])
    saving_total=opt["Potential_Saving"].sum()
    saving_str="; ".join([f"{r['Region']}: save ₹{r['Potential_Saving']:,.0f} with {r['Optimal_Carrier']}" for _,r in opt.iterrows() if r['Potential_Saving']>0])
    fwd_str=""
    if not fwd_plan.empty:
        fwd_agg=fwd_plan.groupby("Month").agg(Units=("Prod_Units","sum"),Cost=("Proj_Ship_Cost","sum")).reset_index()
        fwd_str="; ".join([f"{r['Month']}:{r['Units']:.0f}u/₹{r['Cost']:,.0f}" for _,r in fwd_agg.iterrows()])
    cat_rev=ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
    cat_str=", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in cat_rev.items()])
    top_reg=ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False).head(5)
    reg_str=", ".join([f"{k}:₹{v/1e6:.1f}M" for k,v in top_reg.items()])
    top_sku=ops.groupby("Product_Name")["Net_Revenue"].sum().sort_values(ascending=False).head(8)
    sku_str=", ".join(top_sku.index.tolist())
    if r_ord:
        mm=r_ord.get("model_metrics",{}); ens=mm.get("Ensemble",{})
        metric_str=f"Ensemble RMSE:{ens.get('rmse',0):.1f}, NRMSE:{ens.get('nrmse',0)*100:.1f}%, R²:{ens.get('r2',0):.2f}"
    else: metric_str=""
    return f"""=== OmniFlow D2D Intelligence ===
DATASET: 5,010 orders | Jan 2024–Dec 2025 | India D2D (Amazon, Flipkart, B2B)
SUMMARY: Net Revenue ₹{ops['Net_Revenue'].sum()/1e7:.2f}Cr | Active Orders {len(ops):,} | Return Rate {df[df['Order_Status']=='Returned'].shape[0]/len(ops)*100:.1f}% | Avg Delivery {ops['Delivery_Days'].mean():.1f}d
[DEMAND FORECAST] {metric_str}
Order Forecast: {fc_str(r_ord,lambda v:f"{v:.0f}")}
Qty Forecast: {fc_str(r_qty,lambda v:f"{v:.0f}u")}
Revenue Forecast: {fc_str(r_rev,lambda v:f"₹{v/1e6:.1f}M")}
[INVENTORY] Critical:{n_crit} Low:{n_low} Adequate:{inv['Status'].eq('🟢 Adequate').sum()} | ABC:{abc_str}
Reorder NOW: {crit_prods} | Stockout Loss/Day: ₹{total_stockout:,.0f}
[PRODUCTION] {prod_str} | Peak: {peak_mo}
[LOGISTICS] {carr_str}
Best per Region: {", ".join([f"{r['Region']}→{r['Courier_Partner']}" for _,r in best_carr.iterrows()])}
Savings: ₹{saving_total:,.0f} | {saving_str}
Forward Plan: {fwd_str if fwd_str else "N/A"}
CATEGORIES: {cat_str} | TOP REGIONS: {reg_str} | TOP PRODUCTS: {sku_str}"""

def call_llm(messages, system, api_key):
    hdrs={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    body={"model":"llama-3.3-70b-versatile","max_tokens":1800,"temperature":0.4,
        "messages":[{"role":"system","content":system}]+messages}
    try:
        r=_requests.post("https://api.groq.com/openai/v1/chat/completions",headers=hdrs,json=body,timeout=50)
        if r.status_code==401: return "❌ Invalid Groq API key."
        if r.status_code==429: return "⚠️ Rate limit reached. Wait a moment."
        if r.status_code!=200: return f"⚠️ Groq error ({r.status_code}): {r.text[:300]}"
        return r.json()["choices"][0]["message"]["content"]
    except _requests.exceptions.Timeout: return "⚠️ Request timed out."
    except Exception as e: return f"⚠️ Error: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")
    delivered=df[df["Order_Status"]=="Delivered"]
    net_rev=ops["Net_Revenue"].sum()
    ret_rate=df[df["Order_Status"]=="Returned"].shape[0]/len(ops)*100
    avg_del=delivered["Delivery_Days"].mean()

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);border-radius:18px;
    padding:30px 32px;margin-bottom:24px;'>
      <div style='font-size:11px;font-family:DM Mono,monospace;color:#93c5fd;letter-spacing:.14em;
      text-transform:uppercase;margin-bottom:6px'>Supply Chain Intelligence Platform</div>
      <div style='font-size:38px;font-weight:900;color:white;letter-spacing:-.02em;line-height:1.1'>
        OmniFlow D2D</div>
      <div style='font-size:13px;color:#93c5fd;margin-top:6px'>
        Demand-to-Delivery · 3-Model Ensemble Forecasting · India E-Commerce</div>
    </div>""", unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4,c5,c6=st.columns(6)
    kpi(c1,"Net Revenue",f"₹{net_rev/1e7:.1f}Cr","amber","excl. returns")
    kpi(c2,"Active Orders",f"{len(ops):,}","sky","Del + Shipped")
    kpi(c3,"Units Sold",f"{ops['Quantity'].sum():,}","sky","all products")
    kpi(c4,"Return Rate",f"{ret_rate:.1f}%","coral","of active")
    kpi(c5,"Avg Delivery",f"{avg_del:.1f}d","mint","delivered only")
    kpi(c6,"SKU Categories",f"{df['Category'].nunique()}","sky","product types")
    sp()

    # About section
    st.markdown("""<div class='about-section'>
    <div style='font-size:16px;font-weight:800;color:#0f172a;margin-bottom:14px'>📋 About This Platform</div>
    <p style='color:#334155;line-height:1.9;font-size:13.5px;margin:0 0 14px'>
    <b style='color:#0f172a'>OmniFlow</b> is a fully integrated, end-to-end supply chain intelligence platform
    built on <b>5,010 D2D e-commerce orders</b> across India (Jan 2024 – Dec 2025), spanning
    <b>Amazon.in, Flipkart / Shiprocket, and INCREFF B2B</b> channels.
    </p>
    <p style='color:#334155;line-height:1.9;font-size:13.5px;margin:0 0 14px'>
    The platform is structured as a <b>closed-loop causal pipeline</b>:
    demand signals feed inventory optimisation, which informs production scheduling, which drives logistics planning.
    Every module shows both <b>historical actuals and a 6-month forward forecast</b>.
    </p>
    <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:16px'>
      <div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:10px;padding:14px'>
        <div style='font-size:11px;font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px'>📊 Dataset</div>
        <div style='font-size:12.5px;color:#334155;line-height:1.8'>
          • <b>5,010 orders</b> across Jan 2024 – Dec 2025<br>
          • <b>4 categories</b>: Electronics & Mobiles, Fashion & Apparel, Home & Kitchen, Health & Personal Care<br>
          • <b>9 regions</b> across India (Karnataka, Tamil Nadu, Delhi, Maharashtra…)<br>
          • <b>5 courier partners</b>: BlueDart, Delhivery, DTDC, Ecom Express, XpressBees<br>
          • <b>4 warehouses</b>: Bengaluru, Delhi, Hyderabad, Mumbai
        </div>
      </div>
      <div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:10px;padding:14px'>
        <div style='font-size:11px;font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px'>🤖 Forecast Engine</div>
        <div style='font-size:12.5px;color:#334155;line-height:1.8'>
          • <span class='model-pill pill-ridge'>Ridge Regression</span> trend + Fourier seasonality<br>
          • <span class='model-pill pill-rf'>Random Forest</span> non-linear demand patterns<br>
          • <span class='model-pill pill-gb'>Gradient Boosting</span> boosted residual correction<br>
          • <span class='model-pill pill-ensemble'>Ensemble</span> inverse-RMSE weighted blend<br>
          • Walk-forward CV (3 folds) · Asymmetric log-normal 90% CI
        </div>
      </div>
      <div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:10px;padding:14px'>
        <div style='font-size:11px;font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px'>📦 Inventory Module</div>
        <div style='font-size:12.5px;color:#334155;line-height:1.8'>
          • Wilson <b>EOQ</b> with peak-blended demand (60% avg + 40% peak)<br>
          • Full Safety Stock: <b>z·√(LT·σ_d² + D²·σ_LT²)</b><br>
          • Reorder Point = Daily Demand × Lead Time + SS<br>
          • <b>ABC classification</b> (Pareto: A=70%, B=90%, C=100%)<br>
          • Stockout cost estimation for critical SKUs
        </div>
      </div>
      <div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:10px;padding:14px'>
        <div style='font-size:11px;font-weight:700;color:#1e3a8a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px'>🚚 Logistics Module</div>
        <div style='font-size:12.5px;color:#334155;line-height:1.8'>
          • Carrier scoring: <b>Speed 40% + Cost 35% + Returns 25%</b> (configurable)<br>
          • Region × carrier delay heatmap<br>
          • Cost saving analysis with optimal carrier recommendations<br>
          • Production-driven forward shipment plan (6 months)<br>
          • Warehouse inbound volume forecast
        </div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)

    # Pipeline
    sec("Closed-Loop Module Pipeline","🔄")
    st.markdown("""<div style='background:white;border:1px solid #e5e7eb;border-radius:14px;
    padding:22px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:0;margin-bottom:8px'>
      <div class='pipeline-box'>Demand Forecast<span class='pipeline-sub'>Ridge + RF + GB<br>3-Model Ensemble</span></div>
      <div style='color:#8B5CF6;font-size:22px;padding:0 10px;font-weight:300'>→</div>
      <div class='pipeline-box' style='border-color:#34d399'>Inventory<span class='pipeline-sub'>EOQ + SS + ROP<br>ABC Pareto</span></div>
      <div style='color:#8B5CF6;font-size:22px;padding:0 10px;font-weight:300'>→</div>
      <div class='pipeline-box' style='border-color:#fbbf24'>Production Plan<span class='pipeline-sub'>6-Month Targets<br>Replenishment Boost</span></div>
      <div style='color:#8B5CF6;font-size:22px;padding:0 10px;font-weight:300'>→</div>
      <div class='pipeline-box' style='border-color:#f87171'>Logistics Opt.<span class='pipeline-sub'>Carrier Scoring<br>Cost Saving</span></div>
      <div style='color:#8B5CF6;font-size:22px;padding:0 10px;font-weight:300'>→</div>
      <div class='pipeline-box' style='border-color:#a78bfa'>AI Chatbot<span class='pipeline-sub'>Groq LLaMA 3.3<br>Full Context</span></div>
    </div>""", unsafe_allow_html=True)
    banner("Each module is causally connected — Demand signals drive Inventory EOQ/SS, which sets Production targets, which feeds Logistics shipment planning. The AI Chatbot receives the full computed context from all modules.", "purple")
    sp()

    # Quick stats
    sec("Key Business Metrics","📊")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        cat=ops.groupby("Category")["Net_Revenue"].sum().sort_values(ascending=False)
        fig=go.Figure(go.Bar(x=cat.index,y=cat.values/1e6,
            marker=dict(color=COLORS[:len(cat)],line=dict(color="rgba(0,0,0,0)")),
            text=[f"₹{v/1e6:.1f}M" for v in cat.values],textposition="outside",textfont=dict(color="#334155")))
        fig.update_layout(**CD(),height=250,title=dict(text="Net Revenue by Category (₹M)",font=dict(size=11,color="#64748b")),
            xaxis=gX(),yaxis={**gY(),"title":"₹M"})
        st.plotly_chart(fig,use_container_width=True,key="ov_cat_rev")
    with col_b:
        sc=df["Order_Status"].value_counts()
        fig2=go.Figure(go.Pie(labels=sc.index,values=sc.values,hole=.58,
            marker=dict(colors=["#22C55E","#3B82F6","#EF4444","#F59E0B"],line=dict(color="#FFFFFF",width=3)),
            textinfo="label+percent",textfont=dict(size=11,color="#333333")))
        fig2.update_layout(**CD(),height=250,showlegend=False,
            title=dict(text="Order Status Split",font=dict(size=11,color="#64748b")),
            annotations=[dict(text="Orders",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#64748b",family="DM Mono"))])
        st.plotly_chart(fig2,use_container_width=True,key="ov_status")

    sp()
    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        top_reg=ops.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False)
        fig3=go.Figure(go.Bar(x=top_reg.index,y=top_reg.values/1e6,
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(top_reg))],line=dict(color="rgba(0,0,0,0)")),
            text=[f"₹{v/1e6:.1f}M" for v in top_reg.values],textposition="outside",textfont=dict(color="#334155")))
        fig3.update_layout(**CD(),height=240,title=dict(text="Revenue by Region (₹M)",font=dict(size=11,color="#64748b")),
            xaxis={**gX(),"tickangle":-25},yaxis={**gY(),"title":"₹M"})
        st.plotly_chart(fig3,use_container_width=True,key="ov_region")
    with col_d:
        ch=ops.groupby("Sales_Channel")["Net_Revenue"].sum().sort_values(ascending=False)
        fig4=go.Figure(go.Pie(labels=ch.index,values=ch.values,hole=.55,
            marker=dict(colors=["#1565C0","#2E7D32","#E65100"],line=dict(color="#FFFFFF",width=3)),
            textinfo="label+percent",textfont=dict(size=11)))
        fig4.update_layout(**CD(),height=240,showlegend=False,
            title=dict(text="Revenue by Sales Channel",font=dict(size=11,color="#64748b")),
            annotations=[dict(text="Channel",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#64748b",family="DM Mono"))])
        st.plotly_chart(fig4,use_container_width=True,key="ov_channel")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Demand Forecasting
# ═══════════════════════════════════════════════════════════════════════════════

def page_demand():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title'>📈 Demand Forecasting</div>", unsafe_allow_html=True)
    st.markdown("""<div class='page-subtitle'>3-Model Ensemble · Ridge + Random Forest + Gradient Boosting · Inverse-RMSE Weighted Blend</div>""", unsafe_allow_html=True)
    banner("""<b>Methodology:</b> Walk-forward cross-validation (3 folds). Features include trend, quadratic trend,
    Fourier seasonality (3 harmonics), structural-break regime, quarter dummies, log-time.
    Net_Qty used (returns excluded). Asymmetric log-normal 90% CI widens with forecast horizon.""","purple")

    # Overall model quality
    sec("Ensemble Model Quality (Overall Orders)","🤖")
    m_orders=ops.groupby("YM")["Order_ID"].count().rename("v")
    res_ov=ml_forecast(m_orders.values.astype(float),m_orders.index,6)
    if res_ov: render_model_quality(res_ov)

    sp()
    # Model comparison bars
    if res_ov and "model_metrics" in res_ov:
        sec("Model Accuracy Comparison","📊")
        mm=res_ov["model_metrics"]
        labels=[m for m in ["Ridge","RandomForest","GradBoost","Ensemble"] if m in mm]
        r2_vals=[mm[m]["r2"] for m in labels]
        nrmse_vals=[mm[m]["nrmse"]*100 for m in labels]
        clrs=[MODEL_COLORS.get(m,"#888") for m in labels]
        bc1,bc2=st.columns(2,gap="large")
        with bc1:
            fig=go.Figure(go.Bar(x=labels,y=r2_vals,marker=dict(color=clrs,line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.3f}" for v in r2_vals],textposition="outside",textfont=dict(color="#334155")))
            fig.add_hline(y=0.9,line_dash="dash",line_color="#22C55E",
                annotation_text=" Target R²=0.90",annotation_font=dict(color="#22C55E",size=10))
            fig.update_layout(**CD(),height=240,xaxis=gX(),yaxis={**gY(),"title":"R² Score","range":[0,1.1]},
                title=dict(text="R² Score (higher = better)",font=dict(size=11,color="#64748b")))
            st.plotly_chart(fig,use_container_width=True,key="d_r2")
        with bc2:
            fig2=go.Figure(go.Bar(x=labels,y=nrmse_vals,marker=dict(color=clrs,line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in nrmse_vals],textposition="outside",textfont=dict(color="#334155")))
            fig2.add_hline(y=15,line_dash="dash",line_color="#22C55E",
                annotation_text=" Target <15%",annotation_font=dict(color="#22C55E",size=10))
            fig2.update_layout(**CD(),height=240,xaxis=gX(),yaxis={**gY(),"title":"NRMSE %"},
                title=dict(text="NRMSE % (lower = better)",font=dict(size=11,color="#64748b")))
            st.plotly_chart(fig2,use_container_width=True,key="d_nrmse")
    sp()

    # Controls
    c1,c2,c3=st.columns([2,2,1])
    metric_opt=c1.selectbox("Metric",["Orders (#)","Quantity (Units)","Net Revenue (₹)"],key="d_metric")
    level_opt=c2.selectbox("Breakdown",["Overall","Category","Region","Sales Channel"],key="d_level")
    horizon=c3.slider("Forecast months",3,12,6,key="d_horizon")
    col_map={"Orders (#)":"Order_ID","Quantity (Units)":"Net_Qty","Net Revenue (₹)":"Net_Revenue"}
    col=col_map[metric_opt]

    def get_series(sub):
        if col=="Order_ID": return sub.groupby("YM")["Order_ID"].count().rename("v")
        return sub.groupby("YM")[col].sum().rename("v")

    def draw_with_table(series, title="", chart_key="d_main"):
        res=ml_forecast(series.values.astype(float),series.index,n_future=horizon)
        if res is None: st.info("Insufficient data."); return
        fig=ensemble_chart(res,chart_key=chart_key,height=310,title=title)
        st.plotly_chart(fig,use_container_width=True,key=chart_key)
        tbl=pd.DataFrame({"Month":[d.strftime("%b %Y") for d in res["fut_ds"]],
            "Ensemble":res["forecast"].round(0).astype(int),
            "Ridge":np.maximum(res["forecast_per_model"]["Ridge"],0).round(0).astype(int),
            "RandomForest":np.maximum(res["forecast_per_model"]["RandomForest"],0).round(0).astype(int),
            "GradBoost":np.maximum(res["forecast_per_model"]["GradBoost"],0).round(0).astype(int),
            "Lower 90%":res["ci_lo"].round(0).astype(int),"Upper 90%":res["ci_hi"].round(0).astype(int)})
        st.dataframe(tbl,use_container_width=True,hide_index=True)

    sec("Forecast Chart + Table","📉")
    if level_opt=="Overall":
        draw_with_table(get_series(ops), chart_key="d_overall")
    else:
        grp_map={"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}
        grp=grp_map[level_opt]
        top=ops[grp].value_counts().head(5).index.tolist()
        tabs=st.tabs(top)
        for i,(tab,val) in enumerate(zip(tabs,top)):
            with tab:
                draw_with_table(get_series(ops[ops[grp]==val]),title=val,chart_key=f"d_bd_{i}")

    sp()
    sec("YoY Revenue Growth by Category","📅")
    yr_rev=ops.groupby(["Year","Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    cat_monthly=ops.groupby(["YM","Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    proj_next={}
    for cat in cat_monthly.columns:
        r=ml_forecast(cat_monthly[cat].values.astype(float),cat_monthly.index,12)
        if r: proj_next[cat]=sum(v for d,v in zip(r["fut_ds"],r["forecast"]) if d.year==r["fut_ds"][0].year)
    if 2024 in yr_rev.index and 2025 in yr_rev.index:
        rows=[]
        for cat in yr_rev.columns:
            r24=yr_rev.loc[2024,cat]; r25=yr_rev.loc[2025,cat]; rp=proj_next.get(cat,0)
            rows.append({"Category":cat,"2024 ₹M":round(r24/1e6,1),"2025 ₹M":round(r25/1e6,1),
                "YoY 24→25":f"{(r25-r24)/r24*100:+.1f}%" if r24>0 else "N/A",
                "Projected ₹M":round(rp/1e6,1),"Projected Growth":f"{(rp-r25)/r25*100:+.1f}%" if r25>0 else "N/A"})
        st.dataframe(pd.DataFrame(rows).sort_values("Projected ₹M",ascending=False),use_container_width=True,hide_index=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Inventory
# ═══════════════════════════════════════════════════════════════════════════════

def page_inventory():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title'>📦 Inventory Optimisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Fed by Demand Forecast · EOQ + Full Safety Stock + ROP + ABC Classification</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Inventory Parameters", expanded=False):
        p1,p2,p3,p4=st.columns(4)
        order_cost=p1.number_input("Order Cost ₹",100,5000,500,50)
        hold_pct=p2.slider("Holding Cost %",5,40,20)/100
        lead_time=p3.slider("Lead Time (days)",1,30,7)
        svc=p4.selectbox("Service Level",["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"])
        z={"90% (z=1.28)":1.28,"95% (z=1.65)":1.65,"99% (z=2.33)":2.33}[svc]

    banner("""<b>Data pipeline:</b> Current stock comes from the <b>actual dataset snapshot</b> (Current_Stock_Units per SKU).
    EOQ and Safety Stock are computed using <b>ML demand forecast</b> (blended historical + category-level 6-month forward forecast per SKU share).
    SS formula: <b>z × √(LT × σ_d² + D_avg² × σ_LT²)</b> · ROP = Daily_Demand × Lead_Time + SS · Status = Actual Stock vs computed ROP/SS.""","purple")

    inv=compute_inventory(order_cost,hold_pct,lead_time,z)
    if inv.empty: st.warning("No inventory data."); return

    n_crit=(inv["Status"]=="🔴 Critical").sum()
    n_low=(inv["Status"]=="🟡 Low").sum()
    n_ok=inv["Status"].str.startswith("🟢").sum()
    total_stockout=inv["Stockout_Cost_Day"].sum()
    n_a=(inv["ABC"]=="A").sum()
    avg_days_stock=inv["Days_of_Stock"].replace(999,np.nan).median()
    c1,c2,c3,c4,c5,c6=st.columns(6)
    kpi(c1,"Total SKUs",len(inv),"sky")
    kpi(c2,"🔴 Critical",n_crit,"coral","below safety stock")
    kpi(c3,"🟡 Low",n_low,"amber","below ROP")
    kpi(c4,"🟢 Adequate/Over",n_ok,"mint","above ROP")
    kpi(c5,"A-Class SKUs",n_a,"sky","top 70% revenue")
    kpi(c6,"Stockout ₹/Day",f"₹{total_stockout:,.0f}","coral","critical SKUs")
    sp()

    cl,cr=st.columns([1,2],gap="large")
    with cl:
        sec("Stock Status Distribution")
        sc=inv["Status"].value_counts()
        sc_clrs={"🔴 Critical":"#EF4444","🟡 Low":"#F59E0B","🟢 Adequate":"#22C55E","🟢 Overstocked":"#06B6D4"}
        fig=go.Figure(go.Pie(labels=sc.index,values=sc.values,hole=.6,
            marker=dict(colors=[sc_clrs.get(s,"#888") for s in sc.index],line=dict(color="#FFFFFF",width=3)),
            textinfo="label+value",textfont=dict(size=10)))
        fig.update_layout(**CD(),height=260,showlegend=False,
            annotations=[dict(text="SKUs",x=.5,y=.5,showarrow=False,font=dict(size=10,color="#64748b",family="DM Mono"))])
        st.plotly_chart(fig,use_container_width=True,key="inv_status_pie")
    with cr:
        sec("EOQ / Safety Stock / ROP by Category")
        ci2=inv.groupby("Category")[["EOQ","SS","ROP"]].mean().reset_index()
        fig2=go.Figure()
        for i,(m2,lbl) in enumerate([("EOQ","EOQ"),("SS","Safety Stock"),("ROP","Reorder Point")]):
            fig2.add_trace(go.Bar(name=lbl,x=ci2["Category"],y=ci2[m2].round(1),
                marker=dict(color=["#F59E0B","#06B6D4","#8B5CF6"][i],line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(),height=260,barmode="group",xaxis={**gX(),"tickangle":-10},yaxis=gY(),legend=leg())
        st.plotly_chart(fig2,use_container_width=True,key="inv_eoq_bar")

    sp()
    sec("EOQ Cost Trade-off by Category")
    eoq_tbl=inv.groupby("Category").agg(Avg_EOQ=("EOQ","mean"),Avg_Ann_Demand=("Annual_Demand","mean"),Avg_Price=("Unit_Price","mean")).reset_index()
    eoq_tbl["Ann_Order_Cost"]=(eoq_tbl["Avg_Ann_Demand"]/eoq_tbl["Avg_EOQ"].replace(0,1)*order_cost).round(0)
    eoq_tbl["Ann_Holding_Cost"]=(eoq_tbl["Avg_EOQ"]/2*eoq_tbl["Avg_Price"]*hold_pct).round(0)
    eoq_tbl["Total_Cost"]=eoq_tbl["Ann_Order_Cost"]+eoq_tbl["Ann_Holding_Cost"]
    fig_eoq=go.Figure()
    fig_eoq.add_trace(go.Bar(name="Annual Ordering Cost",x=eoq_tbl["Category"],y=eoq_tbl["Ann_Order_Cost"],marker=dict(color="#3B82F6",line=dict(color="rgba(0,0,0,0)"))))
    fig_eoq.add_trace(go.Bar(name="Annual Holding Cost",x=eoq_tbl["Category"],y=eoq_tbl["Ann_Holding_Cost"],marker=dict(color="#F59E0B",line=dict(color="rgba(0,0,0,0)"))))
    fig_eoq.add_trace(go.Scatter(name="Total Cost",x=eoq_tbl["Category"],y=eoq_tbl["Total_Cost"],mode="markers+text",
        marker=dict(size=12,color="#EF4444",symbol="diamond"),
        text=[f"₹{v:,.0f}" for v in eoq_tbl["Total_Cost"]],textposition="top center",textfont=dict(color="#334155",size=9)))
    fig_eoq.update_layout(**CD(),height=250,barmode="group",xaxis={**gX(),"tickangle":-10},yaxis={**gY(),"title":"₹/Year"},legend=leg())
    st.plotly_chart(fig_eoq,use_container_width=True,key="eoq_cost")

    sp()
    abc_l,abc_r=st.columns(2,gap="large")
    with abc_l:
        sec("ABC Pareto Classification")
        abc_grp=inv.groupby("ABC").agg(SKUs=("SKU_ID","count"),Revenue=("Total_Revenue","sum")).reset_index()
        abc_grp["Rev_Pct"]=(abc_grp["Revenue"]/abc_grp["Revenue"].sum()*100).round(1)
        fig_abc=go.Figure(go.Bar(x=abc_grp["ABC"],y=abc_grp["Rev_Pct"],
            marker=dict(color=["#1565C0","#2E7D32","#E65100"],line=dict(color="rgba(0,0,0,0)")),
            text=[f"{r['SKUs']} SKUs · {r['Rev_Pct']:.1f}%" for _,r in abc_grp.iterrows()],
            textposition="outside",textfont=dict(color="#334155")))
        fig_abc.update_layout(**CD(),height=240,xaxis={**gX(),"title":"ABC Class"},yaxis={**gY(),"title":"Revenue %"})
        st.plotly_chart(fig_abc,use_container_width=True,key="abc_chart")
    with abc_r:
        sec("Stockout Cost by Category (Critical)")
        so=inv[inv["Status"]=="🔴 Critical"].groupby("Category")["Stockout_Cost_Day"].sum().reset_index()
        if so.empty: st.info("No critical SKUs.")
        else:
            fig_so=go.Figure(go.Bar(x=so["Category"],y=so["Stockout_Cost_Day"],
                marker=dict(color="#EF4444",line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:,.0f}/day" for v in so["Stockout_Cost_Day"]],
                textposition="outside",textfont=dict(color="#334155")))
            fig_so.update_layout(**CD(),height=240,xaxis=gX(),yaxis={**gY(),"title":"₹ Lost / Day"})
            st.plotly_chart(fig_so,use_container_width=True,key="stockout_chart")

    sec("🔴 Critical SKU Alerts — Reorder Immediately")
    crit_df=inv[inv["Status"]=="🔴 Critical"][["SKU_ID","Product_Name","Category","ABC","Current_Stock","SS","ROP","EOQ","Monthly_Avg","Forecast_Avg","Days_of_Stock","Unit_Price","Stockout_Cost_Day"]].copy()
    crit_df.columns=["SKU","Product","Category","ABC","Current Stock","Safety Stock","ROP","Order Qty (EOQ)","Hist Avg/Mo","Forecast Avg/Mo","Days of Stock","Unit Price ₹","Stockout ₹/Day"]
    for c in ["Current Stock","Safety Stock","ROP","Order Qty (EOQ)"]: crit_df[c]=crit_df[c].astype(int)
    st.dataframe(crit_df.sort_values("Stockout ₹/Day",ascending=False),use_container_width=True,hide_index=True)

    sp()
    sec("Demand Forecast per Category (Feeds into Production)","📈")
    cat_qty=ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
    tabs=st.tabs(list(cat_qty.columns))
    for i,(tab,cat) in enumerate(zip(tabs,cat_qty.columns)):
        with tab:
            res=ml_forecast(cat_qty[cat].values.astype(float),cat_qty.index,6)
            if res is None: st.info("Insufficient data."); continue
            fig=ensemble_chart(res,chart_key=f"inv_fc_{i}",height=270)
            st.plotly_chart(fig,use_container_width=True,key=f"inv_fc_{i}")
            st.dataframe(pd.DataFrame({"Month":[d.strftime("%b %Y") for d in res["fut_ds"]],
                "Forecast":res["forecast"].round(0).astype(int),
                "CI Lo":res["ci_lo"].round(0).astype(int),"CI Hi":res["ci_hi"].round(0).astype(int)}),
                use_container_width=True,hide_index=True)

    sp()
    sec("Stock Depletion & Replenishment Simulation","🔁")
    banner("Starting stock = <b>actual Current_Stock_Units</b> from dataset (latest per SKU, summed by category). Demand driven by ML ensemble forecast. ROP and SS computed from forecast-blended EOQ model. 📦 = replenishment trigger.","teal")
    plan_for_inv=compute_production()
    cats=sorted(inv["Category"].unique())
    tabs2=st.tabs(cats)
    for tab,cat in zip(tabs2,cats):
        with tab:
            cat_inv=inv[inv["Category"]==cat]
            if cat_inv.empty or cat not in cat_qty.columns: st.info("No data."); continue
            avg_eoq=max(int(cat_inv["EOQ"].mean()),1); avg_rop=max(int(cat_inv["ROP"].mean()),1)
            avg_ss=max(int(cat_inv["SS"].mean()),0); total_stock=max(int(cat_inv["Current_Stock"].sum()),0)
            cat_plan=plan_for_inv[plan_for_inv["Category"]==cat].sort_values("Month_dt") if not plan_for_inv.empty else pd.DataFrame()
            res=ml_forecast(cat_qty[cat].values.astype(float),cat_qty.index,6)
            if res is None: st.info("Insufficient data."); continue
            sim_demand=cat_plan["Demand_Forecast"].values if not cat_plan.empty and len(cat_plan)==len(res["fut_ds"]) else res["forecast"]
            months_lbl=[d.strftime("%b %Y") for d in res["fut_ds"]]
            stock=total_stock; stock_levels=[]; reorder_months=[]; reorder_qty=[]
            for i,fc in enumerate(sim_demand):
                stock-=fc
                if stock<=avg_rop:
                    n_ord=max(1,int(np.ceil((avg_rop+avg_ss-stock)/avg_eoq))+1)
                    order_qty=n_ord*avg_eoq; stock+=order_qty
                    reorder_months.append(i); reorder_qty.append(order_qty)
                stock=max(stock,avg_ss); stock_levels.append(round(stock))
            fig=go.Figure()
            ci_u=[max(s+res["resid_std"]*(i+1)*0.5,avg_ss) for i,s in enumerate(stock_levels)]
            ci_l=[max(s-res["resid_std"]*(i+1)*0.5,0) for i,s in enumerate(stock_levels)]
            fig.add_trace(go.Scatter(x=months_lbl+months_lbl[::-1],y=ci_u+ci_l[::-1],fill="toself",
                fillcolor="rgba(46,216,195,0.06)",line=dict(color="rgba(0,0,0,0)"),name="Uncertainty"))
            fig.add_trace(go.Scatter(x=months_lbl,y=stock_levels,name="Projected Stock",
                mode="lines+markers",line=dict(color="#06B6D4",width=3),
                marker=dict(size=10,color=stock_levels,colorscale=[[0,"#EF4444"],[0.4,"#F59E0B"],[1,"#22C55E"]],
                    cmin=avg_ss,cmax=max(stock_levels) if stock_levels else 1,line=dict(color="#FFFFFF",width=2))))
            if not cat_plan.empty:
                fig.add_trace(go.Scatter(x=months_lbl,y=list(sim_demand),name="Production Demand",
                    mode="lines",line=dict(color="#F59E0B",width=1.5,dash="dot"),opacity=0.7))
            fig.add_hline(y=avg_rop,line_dash="dash",line_color="#F59E0B",line_width=2,
                annotation_text=f"  ROP: {avg_rop}",annotation_font=dict(color="#F59E0B",size=10,family="DM Mono"))
            fig.add_hline(y=avg_ss,line_dash="dot",line_color="#EF4444",line_width=2,
                annotation_text=f"  SS: {avg_ss}",annotation_font=dict(color="#EF4444",size=10,family="DM Mono"))
            for ri,rqty in zip(reorder_months,reorder_qty):
                fig.add_vline(x=ri,line_dash="dot",line_color="rgba(139,92,246,0.5)",line_width=1.5)
                fig.add_annotation(x=months_lbl[ri],y=max(stock_levels)*1.1 if stock_levels else avg_rop*2,
                    text=f"📦 +{rqty}u",showarrow=False,
                    font=dict(color="#8B5CF6",size=10,family="DM Mono"),
                    bgcolor="rgba(22,34,54,0.8)",bordercolor="rgba(139,92,246,0.3)",borderwidth=1,borderpad=4)
            fig.update_layout(**CD(),height=300,xaxis={**gX(),"title":"Month"},
                yaxis={**gY(),"title":"Units in Stock"},legend={**leg(),"orientation":"h","y":-0.28},
                title=dict(text=f"{cat} · Starting stock: {total_stock:,} units (category total)",font=dict(color="#64748b",size=11)))
            st.plotly_chart(fig,use_container_width=True,key=f"sim_{cat}")
            ka,kb,kc,kd,ke=st.columns(5)
            kpi(ka,"Starting Stock",total_stock,"mint","category total")
            kpi(kb,"ROP (avg)",avg_rop,"sky","trigger level")
            kpi(kc,"Safety Stock",avg_ss,"sky","min buffer")
            kpi(kd,"EOQ (avg)",avg_eoq,"sky","batch size")
            kpi(ke,"Reorders",len(reorder_months),"mint","over 6 months")
            sp(0.5)
            n_crit_cat=(cat_inv["Status"]=="🔴 Critical").sum(); n_low_cat=(cat_inv["Status"]=="🟡 Low").sum()
            if n_crit_cat>0: banner(f"🔴 <b>{n_crit_cat} Critical SKUs</b> in {cat} — immediate replenishment required.","coral")
            elif n_low_cat>0: banner(f"🟡 <b>{n_low_cat} SKUs</b> approaching ROP — place orders this week.","amber")
            else: banner(f"✅ All {cat} SKUs adequate. Next reorder at ROP={avg_rop}.","mint")

    sp()
    sec("Full SKU-Level Inventory Table","🗃️")
    abc_f=st.multiselect("Filter ABC",["A","B","C"],default=["A","B","C"])
    cat_f=st.multiselect("Filter Category",sorted(df["Category"].unique()),default=sorted(df["Category"].unique()))
    stat_f=st.multiselect("Filter Status",sorted(inv["Status"].unique()),default=sorted(inv["Status"].unique()))
    disp=inv[(inv["ABC"].isin(abc_f))&(inv["Category"].isin(cat_f))&(inv["Status"].isin(stat_f))][
        ["SKU_ID","Product_Name","Category","ABC","Monthly_Avg","Forecast_Avg","Current_Stock","Days_of_Stock","EOQ","SS","ROP","Unit_Price","Stockout_Cost_Day","Status"]].copy()
    disp.columns=["SKU","Product","Category","ABC","Hist Avg/Mo","Forecast Avg/Mo","Current Stock","Days of Stock","EOQ","Safety Stock","ROP","Price ₹","Stockout ₹/Day","Status"]
    for c in ["Current Stock","EOQ","Safety Stock","ROP"]: disp[c]=disp[c].astype(int)
    st.dataframe(disp.sort_values(["ABC","Status"]),use_container_width=True,hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Production
# ═══════════════════════════════════════════════════════════════════════════════

def page_production():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title'>🏭 Production Planning</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Fed by Demand Forecast + Inventory Status · 6-Month Forward Plan with Replenishment Boost</div>", unsafe_allow_html=True)
    banner("Production = ML Demand Forecast + Inventory Replenishment Gap (ROP − Current Stock for critical/low SKUs, distributed 60% month 1 / 40% month 2) + Safety Buffer. All values flow from the Inventory module which uses actual stock data.","teal")

    p1,p2=st.columns(2)
    cap=p1.slider("Capacity Multiplier",0.5,2.0,1.0,0.1)
    buf=p2.slider("Safety Buffer %",5,40,15)/100

    plan=compute_production(cap,buf)
    inv=compute_inventory()
    if plan.empty: st.warning("Insufficient data."); return

    agg=plan.groupby("Month_dt")[["Production","Demand_Forecast","Crit_Boost","Low_Boost"]].sum().reset_index()
    c1,c2,c3,c4=st.columns(4)
    kpi(c1,"Total Production",f"{plan['Production'].sum():,.0f}","amber","units · 6 months")
    kpi(c2,"Total Demand Fc",f"{plan['Demand_Forecast'].sum():,.0f}","sky","forecast units")
    kpi(c3,"Avg / Month",f"{agg['Production'].mean():,.0f}","sky","units/month")
    peak=agg.loc[agg["Production"].idxmax(),"Month_dt"]
    kpi(c4,"Peak Month",peak.strftime("%b %Y"),"amber","highest volume")
    sp()

    sec("Production Target vs Ensemble Demand Forecast (History + Forward)","📊")
    hist_qty=ops.groupby("YM")["Net_Qty"].sum().rename("v")
    hist_ts=_to_ts(hist_qty.index)
    forecast_start=agg["Month_dt"].min()

    # Historical ML fit
    res_hist=ml_forecast(hist_qty.values.astype(float),hist_qty.index,6)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist_ts,y=hist_qty.values,name="Historical Demand",
        fill="tozeroy",fillcolor="rgba(74,94,122,0.10)",line=dict(color="#4a5e7a",width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>"))
    if res_hist:
        fig.add_trace(go.Scatter(x=res_hist["hist_ds"],y=res_hist["fitted"],name="Ensemble Fit",
            line=dict(color="#8B5CF6",width=1.5,dash="dot"),opacity=0.6))
    fig.add_trace(go.Bar(x=agg["Month_dt"],y=agg["Production"],name="Production Target",
        marker=dict(color="#8B5CF6",opacity=0.85,line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Bar(x=agg["Month_dt"],y=agg["Crit_Boost"]+agg["Low_Boost"],name="Inv. Replenishment Boost",
        marker=dict(color="rgba(239,68,68,0.7)",line=dict(color="rgba(0,0,0,0)"))))
    fig.add_trace(go.Scatter(x=agg["Month_dt"],y=agg["Demand_Forecast"],name="Ensemble Demand Forecast",
        mode="lines+markers",line=dict(color="#F59E0B",width=2.5),
        marker=dict(size=8,color="#F59E0B",line=dict(color="#FFFFFF",width=2))))
    if res_hist:
        x_ci=list(res_hist["fut_ds"])+list(res_hist["fut_ds"])[::-1]
        y_ci=list(res_hist["ci_hi"])+list(res_hist["ci_lo"])[::-1]
        fig.add_trace(go.Scatter(x=x_ci,y=y_ci,fill="toself",
            fillcolor="rgba(139,92,246,0.07)",line=dict(color="rgba(0,0,0,0)"),name="90% CI"))
    fig.add_vline(x=forecast_start,line_dash="dash",line_color="rgba(139,92,246,0.5)",line_width=2)
    fig.add_annotation(x=forecast_start,y=1,yref="paper",yanchor="top",
        text="◀ History  |  Forecast ▶",showarrow=False,
        font=dict(color="#8B5CF6",size=10,family="DM Mono"),
        bgcolor="rgba(255,255,255,0.85)",bordercolor="#8B5CF6",borderwidth=1,borderpad=4)
    fig.update_layout(**CD(),height=320,barmode="stack",xaxis=gX(),yaxis=gY(),legend=leg())
    st.plotly_chart(fig,use_container_width=True,key="prod_main")

    cl,cr=st.columns(2,gap="large")
    with cl:
        sec("Production by Category (History + Forecast)")
        cat_hist=ops.groupby(["YM","Category"])["Quantity"].sum().unstack(fill_value=0)
        cat_hist_ts=_to_ts(cat_hist.index)
        fig2=go.Figure()
        fig2.add_vrect(x0=plan["Month_dt"].min(),x1=plan["Month_dt"].max(),
            fillcolor="rgba(139,92,246,0.04)",layer="below",line_width=0)
        fig2.add_vline(x=plan["Month_dt"].min(),line_dash="dash",line_color="rgba(139,92,246,0.4)",line_width=1.5)
        for i,cat in enumerate(plan["Category"].unique()):
            clr=COLORS[i%len(COLORS)]
            if cat in cat_hist.columns:
                fig2.add_trace(go.Scatter(x=cat_hist_ts,y=cat_hist[cat].values,name=f"{cat} hist",
                    line=dict(color=clr,width=1.5,dash="dot"),opacity=0.55,showlegend=False))
            s=plan[plan["Category"]==cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(x=s["Month_dt"],y=s["Production"],name=cat,
                marker=dict(color=clr,line=dict(color="rgba(0,0,0,0)"))))
        fig2.update_layout(**CD(),height=270,barmode="stack",xaxis=gX(),yaxis=gY(),
            legend={**leg(),"orientation":"h","y":-0.32})
        st.plotly_chart(fig2,use_container_width=True,key="prod_cat")
    with cr:
        sec("Production vs Demand Gap")
        agg["Gap"]=agg["Production"]-agg["Demand_Forecast"]
        fig3=go.Figure(go.Bar(x=agg["Month_dt"],y=agg["Gap"],
            marker=dict(color=["#22C55E" if g>=0 else "#EF4444" for g in agg["Gap"]],line=dict(color="rgba(0,0,0,0)")),
            text=[f"{g:+.0f}" for g in agg["Gap"]],textposition="outside",textfont=dict(color="#334155")))
        fig3.add_hline(y=0,line_dash="dash",line_color="rgba(0,0,0,0.2)")
        fig3.update_layout(**CD(),height=270,xaxis=gX(),yaxis={**gY(),"title":"Units Surplus / Deficit"})
        st.plotly_chart(fig3,use_container_width=True,key="prod_gap")

    sec("Detailed 6-Month Production Schedule","📋")
    cat_f=st.selectbox("Filter Category",["All"]+list(plan["Category"].unique()))
    d2=plan if cat_f=="All" else plan[plan["Category"]==cat_f]
    d3=d2[["Month","Category","Demand_Forecast","Crit_Boost","Low_Boost","Buffer","Production","CI_Lo","CI_Hi"]].copy()
    d3.columns=["Month","Category","Demand Fc","Crit Boost","Low Boost","Buffer","Production","Demand Lo","Demand Hi"]
    st.dataframe(d3.sort_values("Month"),use_container_width=True,hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Logistics
# ═══════════════════════════════════════════════════════════════════════════════

def page_logistics():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")
    del_df=get_delivered(df)

    st.markdown("<div class='page-title'>🚚 Logistics Optimisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Fed by Production Plan · Carrier Scoring · Cost Optimisation · Forward Shipment Planning</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Carrier Scoring Weights", expanded=False):
        wc1,wc2,wc3=st.columns(3)
        w_speed=wc1.slider("Speed weight %",10,70,40)/100
        w_cost=wc2.slider("Cost weight %",10,70,35)/100
        w_returns=wc3.slider("Returns weight %",10,70,25)/100
        tot=w_speed+w_cost+w_returns; w_speed/=tot; w_cost/=tot; w_returns/=tot

    carr,best_carr,opt,fwd_plan=compute_logistics(w_speed,w_cost,w_returns)
    plan=compute_production()

    t1,t2,t3,t4,t5=st.tabs(["📊 Carrier Scorecard","💰 Cost Optimisation","⏱ Delay Analysis","🏭 Forward Plan","🗺 Regions"])

    with t1:
        sec("Carrier Performance Scorecard")
        banner(f"Composite score: Speed {w_speed*100:.0f}% · Cost {w_cost*100:.0f}% · Returns {w_returns*100:.0f}% — normalised 0–1 (higher = better)","teal")
        # Bubble chart
        fig=go.Figure()
        for i,(_,r) in enumerate(carr.iterrows()):
            fig.add_trace(go.Scatter(x=[r["Avg_Days"]],y=[r["Avg_Cost"]],mode="markers+text",
                marker=dict(size=max(r["Orders"]/35,16),color=COLORS[i],opacity=0.88,line=dict(color="#FFFFFF",width=2)),
                text=[r["Courier_Partner"]],textposition="top center",name=r["Courier_Partner"],
                hovertemplate=f"<b>{r['Courier_Partner']}</b><br>Orders:{r['Orders']}<br>Avg Days:{r['Avg_Days']:.1f}<br>Avg Cost:₹{r['Avg_Cost']:.0f}<br>Score:{r['Perf_Score']:.3f}<extra></extra>"))
        fig.update_layout(**CD(),height=300,showlegend=False,
            xaxis={**gX(),"title":"Avg Delivery Days"},yaxis={**gY(),"title":"Avg Shipping Cost ₹"})
        st.plotly_chart(fig,use_container_width=True,key="log_bubble")
        d2=carr[["Courier_Partner","Orders","Avg_Days","Avg_Cost","Return_Rate","Delay_Index","Perf_Score"]].copy()
        d2["Avg_Days"]=d2["Avg_Days"].round(1); d2["Avg_Cost"]=d2["Avg_Cost"].round(1)
        d2["Return_Rate"]=(d2["Return_Rate"]*100).round(1).astype(str)+"%"; d2["Perf_Score"]=d2["Perf_Score"].round(3)
        d2.columns=["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Delay Index","Perf Score"]
        st.dataframe(d2.sort_values("Perf Score",ascending=False),use_container_width=True,hide_index=True)

        sp()
        sec("Carrier Order Volume — History + Ensemble Forecast","📈")
        cm=del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        fig_c=go.Figure()
        for i,c in enumerate(cm.columns):
            clr=COLORS[i%len(COLORS)]; r=ml_forecast(cm[c].values.astype(float),cm.index,6)
            if r is None:
                fig_c.add_trace(go.Scatter(x=cm.index.to_timestamp(),y=cm[c],name=c,line=dict(color=clr,width=2))); continue
            x_ci=list(r["fut_ds"])+list(r["fut_ds"])[::-1]
            y_ci=list(r["ci_hi"])+list(r["ci_lo"])[::-1]
            fig_c.add_trace(go.Scatter(x=x_ci,y=y_ci,fill="toself",fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.07)",line=dict(color="rgba(0,0,0,0)"),showlegend=False))
            fig_c.add_trace(go.Scatter(x=r["hist_ds"],y=r["hist_y"],name=c,line=dict(color=clr,width=2.2)))
            fig_c.add_trace(go.Scatter(x=r["fut_ds"],y=r["forecast"],name=f"{c} (fcst)",line=dict(color=clr,width=2.2,dash="dot"),mode="lines+markers",marker=dict(size=6,line=dict(color="#FFFFFF",width=1.5)),showlegend=False))
        fig_c.update_layout(**CD(),height=290,xaxis=gX(),yaxis={**gY(),"title":"Orders"},legend=leg())
        st.plotly_chart(fig_c,use_container_width=True,key="log_carr_fc")

        if not plan.empty:
            sec("Recommended Carrier per Category (Based on Score)")
            cat_carr=del_df.groupby(["Category","Courier_Partner"]).agg(
                Avg_Days=("Delivery_Days","mean"),Avg_Cost=("Shipping_Cost_INR","mean"),Return_Rate=("Return_Flag","mean")).reset_index()
            for col_c,wt_c in [("Avg_Days",w_speed),("Avg_Cost",w_cost),("Return_Rate",w_returns)]:
                mn_c=cat_carr[col_c].min(); mx_c=cat_carr[col_c].max()
                cat_carr[f"N_{col_c}"]=1-(cat_carr[col_c]-mn_c)/(mx_c-mn_c+1e-9)
            cat_carr["Score"]=w_speed*cat_carr["N_Avg_Days"]+w_cost*cat_carr["N_Avg_Cost"]+w_returns*cat_carr["N_Return_Rate"]
            best_cat=cat_carr.sort_values("Score",ascending=False).groupby("Category").first().reset_index()
            prod_by_cat=plan.groupby("Category")["Production"].sum().reset_index()
            best_cat=best_cat.merge(prod_by_cat.rename(columns={"Production":"Planned Units 6M"}),on="Category",how="left")
            best_cat["Avg_Days"]=best_cat["Avg_Days"].round(1); best_cat["Avg_Cost"]=best_cat["Avg_Cost"].round(1)
            best_cat["Return_Rate"]=(best_cat["Return_Rate"]*100).round(1); best_cat["Score"]=best_cat["Score"].round(3)
            best_cat["Planned Units 6M"]=best_cat["Planned Units 6M"].fillna(0).astype(int)
            best_cat=best_cat[["Category","Courier_Partner","Avg_Days","Avg_Cost","Return_Rate","Score","Planned Units 6M"]]
            best_cat.columns=["Category","Recommended Carrier","Avg Days","Avg Cost ₹","Return Rate %","Score","Planned Units"]
            st.dataframe(best_cat.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

    with t2:
        sec("Logistics Cost Optimisation")
        total_curr=del_df["Shipping_Cost_INR"].sum(); total_sav=opt["Potential_Saving"].sum()
        c1,c2,c3,c4=st.columns(4)
        kpi(c1,"Current Spend",f"₹{total_curr:,.0f}","sky","all deliveries")
        kpi(c2,"Optimised Spend",f"₹{total_curr-total_sav:,.0f}","mint","best carriers")
        kpi(c3,"Total Saving",f"₹{total_sav:,.0f}","mint","carrier switch")
        kpi(c4,"Saving %",f"{total_sav/total_curr*100:.1f}%","mint","of total spend")
        sp()
        sec("Region-Level Cost Comparison")
        fig_cost=go.Figure()
        fig_cost.add_trace(go.Bar(name="Current Avg ₹",x=opt["Region"],y=opt["Current_Avg_Cost"],marker=dict(color="#EF4444",line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.add_trace(go.Bar(name="Optimal Avg ₹",x=opt["Region"],y=opt["Min_Avg_Cost"],marker=dict(color="#22C55E",line=dict(color="rgba(0,0,0,0)"))))
        fig_cost.update_layout(**CD(),height=270,barmode="group",xaxis={**gX(),"tickangle":-25},yaxis=gY(),legend=leg())
        st.plotly_chart(fig_cost,use_container_width=True,key="log_cost")
        sec("Savings by Region")
        s_s=opt.sort_values("Potential_Saving",ascending=False)
        fig_sav=go.Figure(go.Bar(x=s_s["Region"],y=s_s["Potential_Saving"],
            marker=dict(color="#F59E0B",line=dict(color="rgba(0,0,0,0)")),
            text=[f"₹{v:,.0f}" for v in s_s["Potential_Saving"]],textposition="outside",textfont=dict(color="#334155")))
        fig_sav.update_layout(**CD(),height=240,xaxis={**gX(),"tickangle":-25},yaxis=gY())
        st.plotly_chart(fig_sav,use_container_width=True,key="log_saving")
        sp()
        sec("Logistics Order Forecast — History + Ensemble","📈")
        series=del_df.groupby(del_df["Order_Date"].dt.to_period("M"))["Order_ID"].count()
        res=ml_forecast(series.values.astype(float),series.index,6)
        if res:
            fig=ensemble_chart(res,chart_key="log_ord_fc",height=290)
            st.plotly_chart(fig,use_container_width=True,key="log_ord_fc")
        sec("Optimisation Recommendation Table")
        od=opt.copy(); od["Current_Avg_Cost"]=od["Current_Avg_Cost"].round(1); od["Min_Avg_Cost"]=od["Min_Avg_Cost"].round(1)
        od["Potential_Saving"]=od["Potential_Saving"].astype(int)
        od=od[["Region","Optimal_Carrier","Current_Avg_Cost","Min_Avg_Cost","Potential_Saving","Saving_Pct","Orders"]]
        od.columns=["Region","Switch To","Current Avg ₹","Optimal Avg ₹","Saving ₹","Saving %","Orders"]
        st.dataframe(od.sort_values("Saving ₹",ascending=False),use_container_width=True,hide_index=True)

    with t3:
        sec("Delay Hotspot Analysis","⚠️")
        thr=st.slider("Delay Threshold (days)",3,10,7,key="log_thr")
        del_df2=del_df.copy(); del_df2["Delayed"]=del_df2["Delivery_Days"]>thr
        cl3,cr3=st.columns(2,gap="large")
        with cl3:
            sec("Delay Rate by Region")
            rd=del_df2.groupby("Region").agg(T=("Order_ID","count"),D=("Delayed","sum")).reset_index()
            rd["Rate"]=(rd["D"]/rd["T"]*100).round(1)
            rd_s=rd.sort_values("Rate",ascending=True)
            fig_r=go.Figure(go.Bar(x=rd_s["Rate"],y=rd_s["Region"],orientation="h",
                marker=dict(color=[f"rgba(255,107,107,{min(v/60+0.25,0.9):.2f})" for v in rd_s["Rate"]],line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v}%" for v in rd_s["Rate"]],textposition="outside",textfont=dict(color="#334155")))
            fig_r.update_layout(**CD(),height=290,xaxis={**gX(),"title":"Delay %"},yaxis=dict(showgrid=False,color="#64748b"))
            st.plotly_chart(fig_r,use_container_width=True,key="log_delay_region")
        with cr3:
            sec("Delay Rate by Carrier")
            cd=del_df2.groupby("Courier_Partner").agg(T=("Order_ID","count"),D=("Delayed","sum")).reset_index()
            cd["Rate"]=(cd["D"]/cd["T"]*100).round(1)
            fig_cd=go.Figure(go.Bar(x=cd["Courier_Partner"],y=cd["Rate"],
                marker=dict(color=["#EF4444" if v>35 else "#F59E0B" if v>20 else "#22C55E" for v in cd["Rate"]],line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v}%" for v in cd["Rate"]],textposition="outside",textfont=dict(color="#334155")))
            fig_cd.update_layout(**CD(),height=290,xaxis=gX(),yaxis={**gY(),"title":"Delay %"})
            st.plotly_chart(fig_cd,use_container_width=True,key="log_delay_carrier")
        sec("Carrier × Region Delay Heatmap")
        pv=del_df2.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0)*100
        fig_h=go.Figure(go.Heatmap(z=pv.values,x=list(pv.columns),y=list(pv.index),
            colorscale=[[0,"#0d1829"],[0.4,"#7c4fd0"],[0.7,"#e87adb"],[1,"#EF4444"]],
            text=np.round(pv.values,1),texttemplate="%{text}%",textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color="#8a9dc0",size=10))))
        fig_h.update_layout(**CD(),height=255,xaxis=dict(showgrid=False,tickangle=-25,color="#64748b"),
            yaxis=dict(showgrid=False,color="#64748b"))
        st.plotly_chart(fig_h,use_container_width=True,key="log_heat")
        sec("Avg Delivery Days Forecast — History + Ensemble","📈")
        delay_m=del_df.groupby(del_df["Order_Date"].dt.to_period("M"))["Delivery_Days"].mean().rename("v")
        r_del=ml_forecast(delay_m.values.astype(float),delay_m.index,6)
        if r_del:
            fig_d=ensemble_chart(r_del,chart_key="delay_fc",height=260)
            st.plotly_chart(fig_d,use_container_width=True,key="delay_fc")

    with t4:
        sec("Production-Driven Forward Shipment Plan","🏭")
        banner("Projected volumes and costs derived from the Production Plan. Uses historical avg shipping cost/unit. Confirms logistics capacity needed for each forecast month.","purple")
        if not fwd_plan.empty:
            fwd_agg=fwd_plan.groupby("Month_dt").agg(Month=("Month","first"),
                Total_Units=("Prod_Units","sum"),Total_Orders=("Proj_Orders","sum"),
                Total_Ship_Cost=("Proj_Ship_Cost","sum"),
                CI_Lo=("CI_Lo_Units","sum"),CI_Hi=("CI_Hi_Units","sum")).reset_index().sort_values("Month_dt")
            fa1,fa2,fa3=st.columns(3)
            kpi(fa1,"6M Planned Units",f"{fwd_agg['Total_Units'].sum():,}","sky","from production plan")
            kpi(fa2,"6M Est. Orders",f"{fwd_agg['Total_Orders'].sum():,}","sky","projected shipments")
            kpi(fa3,"6M Ship Cost",f"₹{fwd_agg['Total_Ship_Cost'].sum():,.0f}","amber","at current avg rate")
            sp()
            fig_fwd=go.Figure()
            x_ci=list(fwd_agg["Month_dt"])+list(fwd_agg["Month_dt"])[::-1]
            y_ci=list(fwd_agg["CI_Hi"])+list(fwd_agg["CI_Lo"])[::-1]
            fig_fwd.add_trace(go.Scatter(x=x_ci,y=y_ci,fill="toself",fillcolor="rgba(59,130,246,0.08)",line=dict(color="rgba(0,0,0,0)"),name="Demand CI"))
            fig_fwd.add_trace(go.Bar(x=fwd_agg["Month_dt"],y=fwd_agg["Total_Units"],name="Planned Shipment Units",
                marker=dict(color="#3B82F6",opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
                hovertemplate="<b>%{x|%b %Y}</b><br>Units: %{y:,}<extra></extra>"))
            fig_fwd.update_layout(**CD(),height=260,barmode="overlay",xaxis=gX(),yaxis={**gY(),"title":"Planned Units"},
                legend={**leg(),"orientation":"h","y":-0.28})
            st.plotly_chart(fig_fwd,use_container_width=True,key="fwd_units")
            fig_cost=go.Figure(go.Scatter(x=fwd_agg["Month_dt"],y=fwd_agg["Total_Ship_Cost"],
                mode="lines+markers",line=dict(color="#8B5CF6",width=2.5),
                marker=dict(size=8,color="#8B5CF6",line=dict(color="#FFFFFF",width=2)),
                fill="tozeroy",fillcolor="rgba(139,92,246,0.07)",
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>",name="Projected Cost"))
            fig_cost.update_layout(**CD(),height=220,xaxis=gX(),yaxis={**gY(),"title":"₹ Shipping Cost"})
            st.plotly_chart(fig_cost,use_container_width=True,key="fwd_cost")
            sec("Category Breakdown — 6-Month Shipment Plan")
            cat_fwd=fwd_plan.groupby("Category").agg(Units=("Prod_Units","sum"),Orders=("Proj_Orders","sum"),
                Ship_Cost=("Proj_Ship_Cost","sum")).reset_index().sort_values("Units",ascending=False)
            cat_fwd.columns=["Category","Planned Units","Est. Orders","Proj. Ship Cost ₹"]
            st.dataframe(cat_fwd,use_container_width=True,hide_index=True)

        sec("Warehouse Shipment Volume — History + Forecast","🏢")
        wm=del_df.groupby([del_df["Order_Date"].dt.to_period("M"),"Warehouse"])["Quantity"].sum().unstack(fill_value=0)
        fig_wh=go.Figure()
        for i,wh in enumerate(wm.columns):
            clr=COLORS[i%len(COLORS)]; r=ml_forecast(wm[wh].values.astype(float),wm.index,6)
            if r is None:
                fig_wh.add_trace(go.Bar(x=wm.index.to_timestamp(),y=wm[wh],name=wh,marker=dict(color=clr,line=dict(color="rgba(0,0,0,0)")))); continue
            fig_wh.add_trace(go.Bar(x=r["hist_ds"],y=r["hist_y"],name=wh,marker=dict(color=clr,opacity=0.85,line=dict(color="rgba(0,0,0,0)"))))
            fig_wh.add_trace(go.Scatter(x=r["fut_ds"],y=r["forecast"],name=f"{wh} (fcst)",
                mode="lines+markers",line=dict(color=clr,width=2.5,dash="dot"),
                marker=dict(size=7,line=dict(color="#FFFFFF",width=2)),showlegend=False))
        fig_wh.update_layout(**CD(),height=290,barmode="stack",xaxis=gX(),yaxis=gY(),legend=leg())
        st.plotly_chart(fig_wh,use_container_width=True,key="wh_vol")

        if not fwd_plan.empty:
            sec("Production-Driven Inbound Plan per Warehouse")
            wh_share=(del_df.groupby("Warehouse")["Quantity"].sum()/del_df["Quantity"].sum()).to_dict()
            inb_rows=[]
            for _,row in fwd_plan.iterrows():
                for wh,sh in wh_share.items():
                    inb_rows.append({"Month":row["Month"],"Month_dt":row["Month_dt"],
                        "Warehouse":wh,"Inbound_Units":round(row["Prod_Units"]*sh),"Proj_Ship_Cost":round(row["Proj_Ship_Cost"]*sh)})
            inb=pd.DataFrame(inb_rows)
            inb_agg=inb.groupby(["Month_dt","Month","Warehouse"]).agg(
                Inbound_Units=("Inbound_Units","sum"),Proj_Ship_Cost=("Proj_Ship_Cost","sum")).reset_index().sort_values(["Month_dt","Warehouse"])
            fig_inb=go.Figure()
            for i,wh in enumerate(sorted(inb_agg["Warehouse"].unique())):
                wdf=inb_agg[inb_agg["Warehouse"]==wh]
                fig_inb.add_trace(go.Bar(x=wdf["Month"],y=wdf["Inbound_Units"],name=wh,
                    marker=dict(color=COLORS[i%len(COLORS)],line=dict(color="rgba(0,0,0,0)"))))
            fig_inb.update_layout(**CD(),height=260,barmode="group",xaxis={**gX(),"tickangle":-25},yaxis={**gY(),"title":"Planned Inbound Units"},legend=leg())
            st.plotly_chart(fig_inb,use_container_width=True,key="wh_inbound")
            disp_inb=inb_agg[["Month","Warehouse","Inbound_Units","Proj_Ship_Cost"]].copy()
            disp_inb.columns=["Month","Warehouse","Planned Inbound Units","Proj. Ship Cost ₹"]
            st.dataframe(disp_inb,use_container_width=True,hide_index=True)

    with t5:
        sec("Region Performance Overview")
        rs=del_df.groupby("Region").agg(Orders=("Order_ID","count"),Revenue=("Net_Revenue","sum"),
            Qty=("Quantity","sum"),Avg_Del=("Delivery_Days","mean"),Returns=("Return_Flag","mean")).reset_index()
        rs["Returns_Pct"]=(rs["Returns"]*100).round(1)
        met=st.selectbox("Metric",["Revenue","Orders","Qty","Avg_Del","Return Rate (%)"])
        met_col={"Revenue":"Revenue","Orders":"Orders","Qty":"Qty","Avg_Del":"Avg_Del","Return Rate (%)":"Returns_Pct"}[met]
        met_lbl={"Revenue":"Revenue ₹","Orders":"Orders","Qty":"Units","Avg_Del":"Avg Delivery Days","Return Rate (%)":"Return Rate %"}[met]
        y=rs[met_col]
        fig_r=go.Figure(go.Bar(x=rs["Region"],y=y,
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(rs))],line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.1f}%" if met=="Return Rate (%)" else f"{v:,.0f}" for v in y],
            textposition="outside",textfont=dict(color="#334155")))
        fig_r.update_layout(**CD(),height=280,xaxis={**gX(),"tickangle":-25},yaxis={**gY(),"title":met_lbl})
        st.plotly_chart(fig_r,use_container_width=True,key="log_region")
        cl5,cr5=st.columns(2,gap="large")
        with cl5:
            sec("Best Carrier per Region")
            bc=best_carr[["Region","Courier_Partner","Avg_Days","Avg_Cost","Score"]].copy()
            bc["Avg_Days"]=bc["Avg_Days"].round(1); bc["Avg_Cost"]=bc["Avg_Cost"].round(1); bc["Score"]=bc["Score"].round(3)
            bc.columns=["Region","Best Carrier","Avg Days","Avg Cost ₹","Score (0–1)"]
            st.dataframe(bc.sort_values("Score (0–1)",ascending=False),use_container_width=True,hide_index=True)
        with cr5:
            sec("Region Return Rate Ranking")
            rr=del_df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False)*100
            fig_ret=go.Figure(go.Bar(x=rr.values,y=rr.index,orientation="h",
                marker=dict(color=["#EF4444" if v>20 else "#F59E0B" if v>12 else "#22C55E" for v in rr.values],line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in rr.values],textposition="outside",textfont=dict(color="#334155")))
            fig_ret.update_layout(**CD(),height=270,xaxis=gX(),yaxis=dict(showgrid=False,color="#64748b"))
            st.plotly_chart(fig_ret,use_container_width=True,key="log_ret_rank")
        sec("Region Revenue Forecast — History + Ensemble","📈")
        top_reg=del_df["Region"].value_counts().head(5).index.tolist()
        fig_rf=go.Figure()
        for i,reg in enumerate(top_reg):
            s=del_df[del_df["Region"]==reg].groupby(del_df["Order_Date"].dt.to_period("M"))["Net_Revenue"].sum().rename("v")
            r=ml_forecast(s.values.astype(float),s.index,6)
            if r is None: continue
            clr=COLORS[i%len(COLORS)]
            fig_rf.add_trace(go.Scatter(x=r["hist_ds"],y=r["hist_y"],name=reg,
                line=dict(color=clr,width=1.5,dash="solid"),opacity=0.6,showlegend=False))
            fig_rf.add_trace(go.Scatter(x=r["fut_ds"],y=r["forecast"],name=reg,
                mode="lines+markers",line=dict(color=clr,width=2.5,dash="dot"),
                marker=dict(size=8,line=dict(color="#FFFFFF",width=2))))
        fig_rf.update_layout(**CD(),height=260,xaxis=gX(),yaxis=gY(),legend=leg())
        st.plotly_chart(fig_rf,use_container_width=True,key="log_reg_fc")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Decision Chatbot
# ═══════════════════════════════════════════════════════════════════════════════

def page_chatbot():
    df=load_data(); ops=get_ops(df).copy()
    ops["YM"]=ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-title'>🤖 Decision Intelligence Chatbot</div>", unsafe_allow_html=True)
    st.markdown("""<div style='margin-bottom:14px'>
      <span class='badge badge-amber'>Demand</span>
      <span class='badge badge-teal'>Inventory</span>
      <span class='badge badge-lav'>Production</span>
      <span class='badge badge-coral'>Logistics</span>
      <span class='badge badge-sky'>Decision Alert</span>
      <span class='badge badge-purple'>3-Model Ensemble</span>
      <span class='badge badge-mint'>Groq LLaMA 3.3-70B</span>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""<div style='margin-top:14px;border-top:1px solid rgba(255,255,255,0.08);
            padding-top:14px;font-family:DM Mono,monospace;font-size:10px;
            color:#4a5e7a;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px'>AI Config</div>""", unsafe_allow_html=True)
        api_key=st.text_input("Groq API Key",type="password",placeholder="gsk_xxxxxxxxxxxxxxxxx",help="Get free key at console.groq.com")
        if api_key and len(api_key.strip())>10:
            if api_key.strip().startswith("gsk_"):
                st.markdown("<div style='font-size:10px;color:#56e0a0;font-family:DM Mono;margin-top:3px'>✅ Key looks valid</div>",unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:10px;color:#ff6b6b;font-family:DM Mono;margin-top:3px'>⚠️ Should start with gsk_</div>",unsafe_allow_html=True)

    ctx=build_context()
    system=f"""You are OmniFlow, an expert AI supply chain analyst for an India D2D e-commerce business.

EXPERTISE: Demand forecasting (Ridge+RF+GradBoost ensemble), Inventory (Wilson EOQ, full safety stock, ROP, ABC), Production planning (6-month, decay-boosted replenishment), Logistics (weighted composite score), Indian e-commerce (Amazon.in, Flipkart, Shiprocket, INCREFF B2B).

RESPONSE RULES:
1. Lead with one precise data-backed insight
2. Use bullet points (▸) with exact numbers (₹, %, units, days)
3. 4-8 bullets per answer
4. No generic closings
5. If not in context, say so clearly

LIVE CONTEXT:
{ctx}"""

    with st.expander("📋 Live Context fed to AI", expanded=False):
        st.code(ctx,language="text")

    key_ok=bool(api_key and len(api_key.strip())>10)
    if not key_ok:
        banner("⚠️ <b>API Key Required</b> — Enter your Groq key in the sidebar to enable the chatbot. Get a free key at <a href='https://console.groq.com' target='_blank'>console.groq.com</a>","amber")

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs=[]

    SUGGESTIONS=[
        "Which product will have the highest demand next month?",
        "Which SKUs are at critical inventory risk right now?",
        "How should I adjust production targets for next quarter?",
        "Which courier should I use for Maharashtra to minimise cost?",
        "Compare all carriers across speed, cost and returns",
        "Give me a complete 6-month production plan summary",
        "Which regions need the most logistics support?",
        "What is the EOQ for Fashion & Apparel?",
        "Calculate total logistics cost saving if I switch carriers",
        "Which model (Ridge/RF/GradBoost) has the best R² score?",
        "What are the top 5 revenue-generating products?",
        "How can I reduce shipping costs across all regions?",
        "Explain the safety stock formula for Electronics",
        "What is the reorder point for Home & Kitchen SKUs?",
        "Which warehouse handles the most volume?",
        "If I increase lead time to 14 days, how does ROP change?",
    ]

    if not st.session_state.chat_msgs:
        sec("⚡ Quick Queries — click any to get started")
        cols=st.columns(4)
        for i,s in enumerate(SUGGESTIONS):
            with cols[i%4]:
                if st.button(s,key=f"sug_{i}",use_container_width=True):
                    if not key_ok:
                        st.warning("⚠️ Enter your API key first.")
                    else:
                        st.session_state.chat_msgs.append({"role":"user","content":s})
                        with st.spinner("OmniFlow analysing…"):
                            reply=call_llm([{"role":"user","content":s}],system,api_key.strip())
                        st.session_state.chat_msgs.append({"role":"assistant","content":reply})
                        st.rerun()

    import re as _re
    for msg in st.session_state.chat_msgs:
        role,content=msg["role"],msg["content"]
        if role=="user":
            st.markdown(f"<div style='margin:10px 0'><div class='chat-user-bubble'>{content}</div></div>",unsafe_allow_html=True)
        else:
            safe=content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            safe=_re.sub(r'\*\*(.+?)\*\*',r'<span style="color:#0f172a;font-weight:700">\1</span>',safe)
            safe=_re.sub(r'\*(.+?)\*',r'<span style="color:#334155;font-style:italic">\1</span>',safe)
            parts=[]
            for line in safe.split("\n"):
                line=line.strip()
                if not line: parts.append("<div style='height:4px'></div>")
                elif _re.match(r'^[▸\-•] ',line):
                    body=line[2:].strip()
                    parts.append(f"<div style='display:flex;gap:7px;margin:4px 0'><span style='color:#1e3a8a;flex-shrink:0;margin-top:2px'>▸</span><span style='color:#334155;line-height:1.6'>{body}</span></div>")
                else:
                    parts.append(f"<div style='color:#334155;line-height:1.6;margin:2px 0'>{line}</div>")
            st.markdown(f"<div style='margin:10px 0'><div class='chat-ai-bubble'>{''.join(parts)}</div></div>",unsafe_allow_html=True)

    sp()
    ci,cb,cc=st.columns([5,1,1])
    with ci:
        user_in=st.text_input("Ask anything…",key="user_input",
            placeholder="e.g. Which model had the best accuracy on the hold-out period?",label_visibility="collapsed")
    with cb:
        if st.button("Send",use_container_width=True):
            if not key_ok: st.warning("⚠️ Enter your API key first.")
            elif user_in.strip():
                st.session_state.chat_msgs.append({"role":"user","content":user_in.strip()})
                with st.spinner("OmniFlow thinking…"):
                    reply=call_llm(st.session_state.chat_msgs[-20:],system,api_key.strip())
                st.session_state.chat_msgs.append({"role":"assistant","content":reply})
                st.rerun()
    with cc:
        if st.button("Clear",use_container_width=True):
            st.session_state.chat_msgs=[]; st.rerun()

    if not st.session_state.chat_msgs:
        sp()
        sec("⚡ Live Decision Alerts")
        al1,al2=st.columns(2,gap="large")
        with al1:
            st.markdown("""<div style='font-size:11px;font-weight:700;color:#EF4444;
                letter-spacing:.06em;text-transform:uppercase;font-family:DM Mono;margin-bottom:8px'>
                🔴 Critical SKUs — Reorder NOW</div>""",unsafe_allow_html=True)
            inv=compute_inventory()
            for _,r in inv[inv["Status"]=="🔴 Critical"][["Product_Name","Category","Current_Stock","ROP"]].head(5).iterrows():
                st.markdown(f"<div class='alert-item alert-critical'><b style='color:#0f172a'>{r['Product_Name']}</b> <span style='color:#64748b;font-size:11px'>[{r['Category']}]</span><br><span style='color:#64748b;font-size:11px'>Stock: {r['Current_Stock']} · ROP: {r['ROP']}</span></div>",unsafe_allow_html=True)
        with al2:
            st.markdown("""<div style='font-size:11px;font-weight:700;color:#d97706;
                letter-spacing:.06em;text-transform:uppercase;font-family:DM Mono;margin-bottom:8px'>
                💰 Cost Saving Opportunities</div>""",unsafe_allow_html=True)
            _,_,opt,_=compute_logistics()
            for _,r in opt.sort_values("Potential_Saving",ascending=False).head(5).iterrows():
                if r["Potential_Saving"]>0:
                    st.markdown(f"<div class='alert-item alert-warn'><b style='color:#0f172a'>{r['Region']}</b> → <b style='color:#0f172a'>{r['Optimal_Carrier']}</b><br><span style='color:#64748b;font-size:11px'>Save ₹{r['Potential_Saving']:,.0f} ({r['Saving_Pct']:.1f}%)</span></div>",unsafe_allow_html=True)

        sp()
        sec("📈 Revenue Forecast — Next 3 Months (Ensemble)")
        m_rev=ops.groupby("YM")["Net_Revenue"].sum().rename("v")
        r_rev=ml_forecast(m_rev.values.astype(float),m_rev.index,3)
        if r_rev is not None:
            last=float(m_rev.iloc[-1]); rc=st.columns(3)
            for i,(dt,fc,lo,hi) in enumerate(zip(r_rev["fut_ds"],r_rev["forecast"],r_rev["ci_lo"],r_rev["ci_hi"])):
                chg=(fc-last)/last*100 if last>0 else 0
                kpi(rc[i],f"{'📈' if chg>=0 else '📉'} {dt.strftime('%b %Y')}",f"₹{fc/1e6:.1f}M","mint" if chg>=0 else "coral",f"{chg:+.1f}% | CI ₹{lo/1e6:.1f}M–₹{hi/1e6:.1f}M")
                last=fc


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar + Navigation
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""<div style='padding:16px 0 22px'>
  <div style='font-family:DM Mono,monospace;font-size:9px;letter-spacing:.16em;
       text-transform:uppercase;color:#4a5e7a;margin-bottom:4px'>Supply Chain Platform</div>
  <div style='font-size:28px;font-weight:900;letter-spacing:-.03em;
       background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow</div>
  <div style='font-family:DM Mono,monospace;font-size:10px;color:#4a5e7a;
       margin-top:2px;letter-spacing:.05em'>D2D INTELLIGENCE · INDIA</div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style='font-family:DM Mono,monospace;font-size:9px;color:#4a5e7a;
    margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em'>Closed-Loop Pipeline</div>
    <div style='font-size:11px;color:#334155;background:#f8faff;border:1px solid #c7d7fd;
    border-radius:8px;padding:9px 11px;margin-bottom:14px;line-height:1.9'>
    <span style='color:#3B82F6;font-weight:700'>① Ridge</span> +
    <span style='color:#22C55E;font-weight:700'>② RF</span> +
    <span style='color:#F59E0B;font-weight:700'>③ GradBoost</span><br>
    <span style='color:#8B5CF6;font-weight:700'>④ Ensemble</span> via inverse-RMSE<br>
    <span style='color:#64748b;font-size:10px'>Demand → Inventory → Production → Logistics</span>
</div>""", unsafe_allow_html=True)

PAGES={
    "🏠 Overview":                page_overview,
    "📈 Demand Forecasting":      page_demand,
    "📦 Inventory Optimisation":  page_inventory,
    "🏭 Production Planning":     page_production,
    "🚚 Logistics Optimisation":  page_logistics,
    "🤖 Decision Chatbot":        page_chatbot,
}
sel=st.sidebar.radio("",list(PAGES.keys()))
PAGES[sel]()
