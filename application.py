import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="OmniFlow D2D Intelligence", page_icon="⬡",
    layout="wide", initial_sidebar_state="expanded",
)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_ecommerce_orders.csv")
COLORS = ["#1565C0", "#2E7D32", "#E65100", "#C62828", "#6A1B9A", "#00695C"]
MODEL_COLORS = {
    "Ridge": "#3B82F6",
    "RandomForest": "#22C55E",
    "GradBoost": "#F59E0B",
    "Ensemble": "#8B5CF6",
}
DEFAULT_ORDER_COST = 500
DEFAULT_HOLD_PCT   = 0.20
DEFAULT_LEAD_TIME  = 7
DEFAULT_SERVICE_Z  = 1.65
N_FUTURE_MONTHS    = 6
MIN_HISTORY_MONTHS = 6
N_ESTIMATORS_RF    = 300      # 300 trees — stable variance
MAX_DEPTH_RF       = 3        # depth 3 — sweet spot for 20 training points
MIN_SAMPLES_LEAF   = 3        # prevents overfitting on small leaves
N_ESTIMATORS_GB    = 150      # enough rounds without memorising
MAX_DEPTH_GB       = 2        # shallow GB avoids R²=1.0 overfit
LEARNING_RATE_GB   = 0.05     # balanced learning rate
SUBSAMPLE_GB       = 0.85     # row sampling regularisation
RIDGE_ALPHA        = 0.1      # stable regularisation for seasonal Fourier fit
CI_Z               = 1.645
MIN_REGIME_IDX     = 6
MARGIN_RATE        = 0.20
DEMAND_PEAK_WEIGHT = 0.30
BOOST_SCHEDULE     = {0: 0.60, 1: 0.40}
DEFAULT_W_SPEED   = 0.40
DEFAULT_W_COST    = 0.35
DEFAULT_W_RETURNS = 0.25

def get_horizon() -> int:
    return st.session_state.get("global_horizon", N_FUTURE_MONTHS)

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap');
    :root{--bg:#f8fafc;--text:#0f172a;--muted:#475569;--primary:#1e3a8a;--border:#e5e7eb;--accent:#e0e7ff;--panel:#ffffff;}
    html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
    section.main>div{animation:fadeIn 0.35s ease-in-out;}
    @keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
    .page-title{font-size:32px;font-weight:900;margin-bottom:4px;color:#0f172a;}
    .section-title{font-size:18px;font-weight:800;margin:24px 0 10px;color:#0f172a;}
    .section-line{height:2px;background:linear-gradient(90deg,#e5e7eb,transparent);margin-bottom:14px;}
    .metric-card{background:linear-gradient(160deg,#eef4ff,#ffffff);padding:12px 14px;margin-bottom:8px;min-height:80px;display:flex;flex-direction:column;justify-content:center;align-items:center;}
    .metric-card:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(30,58,138,0.12);}
    .metric-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;font-family:'DM Mono',monospace;}
    .metric-value{font-size:26px;font-weight:900;color:#1e3a8a;line-height:1.2;margin-top:4px;}
    .metric-sub{font-size:10px;color:#94a3b8;margin-top:3px;}
    .card{background:white;padding:20px;border-radius:14px;border:1px solid #e5e7eb;box-shadow:0 6px 20px rgba(0,0,0,0.07);transition:all .22s;}
    .card:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(0,0,0,0.10);}
    .info-banner{border-radius:10px;padding:12px 14px;margin:8px 0;font-size:12.5px;line-height:1.6;}
    .banner-teal{background:#f0fdfa;border:1px solid #5eead4;}
    .banner-amber{background:#fffbeb;border:1px solid #fbbf24;}
    .banner-coral{background:linear-gradient(135deg,#fff1f2,#ffffff);border:1px solid #fecaca;border-left:4px solid #ef4444;font-weight:600;}
    .banner-mint{background:#ecfdf5;border:1px solid #34d399;}
    .banner-sky{background:#eff6ff;border:1px solid #93c5fd;}
    .sku-alert-card{border-radius:12px;padding:12px 14px;margin-bottom:8px;border:1px solid #e5e7eb;transition:all .2s;cursor:default;}
    .sku-alert-card:hover{transform:translateX(3px);}
    .sku-critical{background:linear-gradient(135deg,#fef2f2,#fff);border-left:4px solid #ef4444;}
    .sku-low{background:linear-gradient(135deg,#fffbeb,#fff);border-left:4px solid #f59e0b;}
    .sku-ok{background:linear-gradient(135deg,#f0fdf4,#fff);border-left:4px solid #22c55e;}
    .model-pill{display:inline-block;padding:3px 9px;font-size:10px;font-weight:700;border-radius:20px;margin-right:5px;margin-bottom:3px;}
    .pill-ridge{background:#eff6ff;color:#1d4ed8}
    .pill-rf{background:#f0fdf4;color:#15803d}
    .pill-gb{background:#fef9c3;color:#a16207}
    .pill-ensemble{background:#fdf4ff;color:#7e22ce}
    .about-section{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:22px 26px;margin-bottom:18px;box-shadow:0 6px 20px rgba(0,0,0,0.06);}
    .pipeline-box{background:white;border:1px solid #c7d7fd;border-radius:14px;padding:18px 22px;text-align:center;min-width:105px;font-weight:700;font-size:12px;font-family:'DM Mono',monospace;color:#0f172a;}
    .pipeline-sub{font-size:9.5px;font-weight:400;color:#64748b;margin-top:3px;display:block;}
    .alert-item{border-radius:9px;padding:9px 12px;margin-bottom:7px;border:1px solid #e5e7eb;}
    .alert-critical{background:#fef2f2;}
    .alert-warn{background:#fff7ed;}
    .model-quality-card{background:white;border-radius:14px;padding:18px;border:1px solid #e5e7eb;box-shadow:0 4px 16px rgba(0,0,0,0.07);}
    .ensemble-card{background:linear-gradient(135deg,#f8faff,#ffffff);border-radius:14px;padding:16px;border:1px solid #c7d7fd;box-shadow:0 4px 16px rgba(30,58,138,0.08);margin-bottom:12px;}
    .stTabs [data-baseweb="tab"]{background:#f1f5f9;border-radius:10px;padding:9px 16px;font-weight:600;color:#475569;font-size:13px;}
    .stTabs [aria-selected="true"]{background:#e0e7ff;color:#1e3a8a;box-shadow:0 4px 14px rgba(30,58,138,0.18);}
    .block-container{padding-top:1.8rem;padding-bottom:2rem;}
    .stMultiSelect div[data-baseweb="tag"]{background:#eef2ff !important;color:#1e3a8a !important;border-radius:8px !important;border:1px solid #c7d7fd !important;font-weight:600;}
    .stMultiSelect div[data-baseweb="tag"] svg{color:#64748b !important;}
    .stMultiSelect{background:white;padding:6px;border-radius:10px;border:1px solid #e5e7eb;}
    .horizon-badge{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#e0e7ff,#f0f4ff);
        border:1px solid #c7d7fd;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;
        color:#1e3a8a;font-family:'DM Mono',monospace;margin-bottom:6px;}
    </style>
    """, unsafe_allow_html=True)

def CD() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter,sans-serif", size=11),
        margin=dict(l=30, r=50, t=42, b=30),
    )

def gY() -> dict:
    return dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False, tickcolor="#64748b")

def gX() -> dict:
    return dict(showgrid=False, zeroline=False, tickcolor="#64748b")

def leg() -> dict:
    return dict(
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#E0E0E0",
        borderwidth=1, font=dict(color="#334155", size=10),
    )

def kpi(col, label: str, value, cls: str = "sky", sub: str = "") -> None:
    color_map = {
        "coral": "#dc2626",
        "sky":   "#1e3a8a",
        "mint":  "#059669",
        "amber": "#d97706",
    }
    color = color_map.get(cls, "#7c3aed")
    col.markdown(
        f"""<div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='color:{color}'>{value}</div>
          <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True,
    )

def sec(label: str, emoji: str = "") -> None:
    st.markdown(
        f"""<div class='section-title'>{emoji} {label}</div>
        <div class='section-line'></div>""", unsafe_allow_html=True,
    )

def banner(html: str, cls: str = "teal") -> None:
    st.markdown(f"<div class='info-banner banner-{cls}'>{html}</div>", unsafe_allow_html=True)

def sp(n: float = 1) -> None:
    st.markdown(f"<div style='height:{n * 12}px'></div>", unsafe_allow_html=True)

def horizon_badge(n_months: int) -> None:
    st.markdown(
        f"<div class='horizon-badge'>📅 Forecast Horizon: {n_months} months</div>",
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
    df["Year"]        = df["Order_Date"].dt.year
    df["Net_Revenue"] = np.where(df["Return_Flag"] == 1, 0.0, df["Revenue_INR"])
    df["Net_Qty"]     = np.where(df["Return_Flag"] == 1, 0,   df["Quantity"])
    return df

@st.cache_data
def get_ops(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Order_Status"].isin(["Delivered", "Shipped"])].copy()

@st.cache_data
def get_delivered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Order_Status"] == "Delivered"].copy()

def _to_ts(idx) -> pd.DatetimeIndex:
    return idx.to_timestamp() if hasattr(idx, "to_timestamp") else pd.DatetimeIndex(idx)

def _make_models() -> dict:
    return {
        "Ridge": Ridge(alpha=RIDGE_ALPHA, fit_intercept=True),
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_depth=MAX_DEPTH_RF,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=0.75,
            bootstrap=True,
            random_state=42,
        ),
        "GradBoost": GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS_GB,
            max_depth=MAX_DEPTH_GB,
            learning_rate=LEARNING_RATE_GB,
            subsample=SUBSAMPLE_GB,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=42,
        )
    }

def _build_features(n_hist: int, n_future: int, ds_hist, regime_idx: int) -> np.ndarray:
    n  = n_hist + n_future
    t  = np.arange(n)
    ts = _to_ts(ds_hist)
    h_months = ts.month.values
    last_m   = int(h_months[-1])
    f_months = np.array([(last_m + i - 1) % 12 + 1 for i in range(1, n_future + 1)])
    mn       = np.concatenate([h_months, f_months])
    regime   = (t >= regime_idx).astype(float)
    q        = np.where(mn <= 3, 1, np.where(mn <= 6, 2, np.where(mn <= 9, 3, 4)))
    return np.column_stack([
        t, t ** 2,
        np.sin(2 * np.pi * mn / 12), np.cos(2 * np.pi * mn / 12),
        np.sin(4 * np.pi * mn / 12), np.cos(4 * np.pi * mn / 12),
        np.sin(6 * np.pi * mn / 12), np.cos(6 * np.pi * mn / 12),
        regime, t * regime,
        (q == 1).astype(float), (q == 2).astype(float), (q == 3).astype(float),
        np.log1p(t),
    ])

def _detect_regime(vals: np.ndarray, min_idx: int = MIN_REGIME_IDX) -> int:
    best_idx, best_ratio = min_idx, 1.0
    for i in range(min_idx, len(vals) - min_idx):
        r = vals[i:].mean() / (vals[:i].mean() + 1e-9)
        if r > best_ratio:
            best_ratio = r
            best_idx   = i
    return best_idx

def ml_forecast(vals: np.ndarray, ds_idx, n_future: int = N_FUTURE_MONTHS) -> dict | None:
    n = len(vals)
    if n < MIN_HISTORY_MONTHS:
        return None
    regime_idx = _detect_regime(vals)
    X_all  = _build_features(n, n_future, ds_idx, regime_idx)
    X_hist = X_all[:n]
    X_fut  = X_all[n:]
    mean_vals = np.mean(vals)
    ss_tot    = np.sum((vals - mean_vals) ** 2)

    # ── Step 1: Hold-out split — train on first (n-4), test on last 4 ──
    h             = 4
    Xtr_h, ytr_h  = X_hist[:-h], vals[:-h]
    Xte_h, yte_h  = X_hist[-h:], vals[-h:]
    mean_holdout  = float(np.mean(yte_h)) if np.mean(yte_h) > 0 else 1.0

    # ── Step 2: Train each model on (n-4) pts → predict hold-out 4 pts ──
    holdout_preds: dict[str, np.ndarray] = {}
    for mname, mdl in _make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(Xtr_h, ytr_h)
        holdout_preds[mname] = np.maximum(pipe.predict(Xte_h), 0)

    # ── Step 3: CV folds on (n-4) data → inverse-RMSE weights ──
    n_tr      = len(ytr_h)
    n_folds   = min(3, n_tr // 6)
    fold_size = 2
    fold_rmses: dict[str, list] = {m: [] for m in _make_models()}
    for fold in range(n_folds):
        te_end   = n_tr - fold * fold_size
        te_start = te_end - fold_size
        if te_start < MIN_HISTORY_MONTHS:
            break
        Xtr, ytr = Xtr_h[:te_start], ytr_h[:te_start]
        Xte, yte = Xtr_h[te_start:te_end], ytr_h[te_start:te_end]
        for mname, mdl in _make_models().items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
            pipe.fit(Xtr, ytr)
            ep = np.maximum(pipe.predict(Xte), 0)
            fold_rmses[mname].append(np.sqrt(mean_squared_error(yte, ep)))
    model_rmses: dict[str, float] = {}
    for mname in fold_rmses:
        model_rmses[mname] = np.mean(fold_rmses[mname]) if fold_rmses[mname] else 1.0
    inv_rmse = {m: 1.0 / (r + 1e-9) for m, r in model_rmses.items()}
    tot      = sum(inv_rmse.values())
    weights  = {m: v / tot for m, v in inv_rmse.items()}

    # ── Step 4: Individual model metrics — all from same hold-out ──
    model_metrics: dict[str, dict] = {}
    for mname in _make_models():
        hp       = holdout_preds[mname]
        rmse_m   = float(np.sqrt(mean_squared_error(yte_h, hp)))
        nrmse_m  = rmse_m / mean_holdout
        mae_m    = float(mean_absolute_error(yte_h, hp))
        model_metrics[mname] = {"rmse": rmse_m, "nrmse": nrmse_m, "mae": mae_m, "r2": 0.0}

    # Full-fit R² for each model (train on all n, predict all n)
    for mname, mdl in _make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(X_hist, vals)
        fp       = np.maximum(pipe.predict(X_hist), 0)
        ss_res_m = np.sum((vals - fp) ** 2)
        model_metrics[mname]["r2"] = max(0.0, 1 - ss_res_m / (ss_tot + 1e-9))

    # ── Step 5: Ensemble hold-out prediction ──
    ypred_eval = sum(weights[m] * holdout_preds[m] for m in _make_models())
    rmse_e     = float(np.sqrt(mean_squared_error(yte_h, ypred_eval)))
    nrmse_e    = rmse_e / mean_holdout
    mae_e      = float(mean_absolute_error(yte_h, ypred_eval))

    # ── Step 6: Full retrain on ALL n points → final fitted + forecast ──
    fitted_pm:   dict[str, np.ndarray] = {}
    forecast_pm: dict[str, np.ndarray] = {}
    for mname, mdl in _make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(X_hist, vals)
        fitted_pm[mname]   = np.maximum(pipe.predict(X_hist), 0)
        forecast_pm[mname] = np.maximum(pipe.predict(X_fut),  0)
    ens_fitted   = sum(weights[m] * fitted_pm[m]   for m in _make_models())
    ens_forecast = sum(weights[m] * forecast_pm[m] for m in _make_models())
    residuals    = vals - ens_fitted
    resid_std    = residuals.std()
    ss_res_e     = np.sum(residuals ** 2)
    r2_e         = max(0.0, 1 - ss_res_e / (ss_tot + 1e-9))
    model_metrics["Ensemble"] = {"rmse": rmse_e, "nrmse": nrmse_e, "mae": mae_e, "r2": r2_e}
    ts_idx   = _to_ts(ds_idx)
    last_dt  = ts_idx[-1]
    fut_dates = pd.date_range(last_dt + pd.offsets.MonthBegin(1), periods=n_future, freq="MS")
    log_std = np.log1p(resid_std / (mean_vals + 1e-9))
    steps   = np.arange(1, n_future + 1)
    ci_lo   = np.maximum(ens_forecast * np.exp(-CI_Z * log_std * np.sqrt(steps)), 0)
    ci_hi   = ens_forecast * np.exp(CI_Z * log_std * np.sqrt(steps))
    return dict(
        hist_ds=ts_idx, hist_y=vals, fitted=ens_fitted,
        fitted_per_model=fitted_pm, forecast_per_model=forecast_pm,
        fut_ds=fut_dates, forecast=ens_forecast, ci_lo=ci_lo, ci_hi=ci_hi,
        rmse=rmse_e, nrmse=nrmse_e, mae=mae_e, r2=r2_e, resid_std=resid_std,
        eval_actual=yte_h, eval_pred=ypred_eval, eval_ds=ts_idx[-h:],
        model_metrics=model_metrics, weights={m: weights[m] for m in _make_models()},
    )

@st.cache_data
def compute_category_forecasts(n_future: int = N_FUTURE_MONTHS) -> dict:
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    cat_monthly = ops.groupby(["YM", "Category"])["Net_Qty"].sum().unstack(fill_value=0)
    results: dict[str, dict] = {}
    for cat in cat_monthly.columns:
        res = ml_forecast(cat_monthly[cat].values.astype(float), cat_monthly.index, n_future)
        if res is not None:
            results[cat] = {
                "mean":      float(np.mean(res["forecast"])),
                "monthly":   res["forecast"].tolist(),
                "ci_lo":     res["ci_lo"].tolist(),
                "ci_hi":     res["ci_hi"].tolist(),
                "fut_ds":    res["fut_ds"],
                "resid_std": res["resid_std"],
                "hist_avg":  float(cat_monthly[cat].mean()),
            }
    return results

def ensemble_chart(res: dict, chart_key: str, height: int = 300, title: str = "", show_models: bool = True) -> go.Figure:
    fig = go.Figure()
    # Forecast zone shading
    fig.add_vrect(x0=res["fut_ds"][0], x1=res["fut_ds"][-1],
        fillcolor="rgba(139,92,246,0.04)", layer="below", line_width=0,
    )
    fig.add_vline(x=res["fut_ds"][0], line_dash="dash",
                  line_color="rgba(139,92,246,0.4)", line_width=1.5)
    # 90% CI ribbon (no legend entry, no hover)
    x_ci = list(res["fut_ds"]) + list(res["fut_ds"])[::-1]
    y_ci = list(res["ci_hi"]) + list(res["ci_lo"])[::-1]
    fig.add_trace(go.Scatter(
        x=x_ci, y=y_ci, fill="toself",
        fillcolor="rgba(139,92,246,0.10)", line=dict(color="rgba(0,0,0,0)"),
        name="90% CI", hoverinfo="skip", showlegend=True,
    ))
    # Actual historical line
    fig.add_trace(go.Scatter(
        x=res["hist_ds"], y=res["hist_y"], name="Actual",
        line=dict(color="#1e3a8a", width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
    ))
    # Per-model fits — hidden by default, toggleable
    model_styles = [
        ("Ridge",        "#3B82F6", "dot"),
        ("RandomForest", "#22C55E", "dashdot"),
        ("GradBoost",    "#F59E0B", "longdash"),
    ]
    if show_models and "fitted_per_model" in res:
        for mname, clr, dash in model_styles:
            if mname in res["fitted_per_model"]:
                fig.add_trace(go.Scatter(
                    x=res["hist_ds"], y=res["fitted_per_model"][mname],
                    name=f"{mname} fit", line=dict(color=clr, width=1.2, dash=dash),
                    opacity=0.7, visible="legendonly",
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{mname}: %{{y:,.0f}}<extra></extra>",
                ))
    # Ensemble fitted line — hidden by default
    fig.add_trace(go.Scatter(
        x=res["hist_ds"], y=res["fitted"], name="Ensemble fit",
        line=dict(color="#8B5CF6", width=1.5, dash="dot"), opacity=0.7,
        visible="legendonly",
        hovertemplate="<b>%{x|%b %Y}</b><br>Ensemble fit: %{y:,.0f}<extra></extra>",
    ))
    # Per-model forecasts — hidden by default, toggleable
    if show_models and "forecast_per_model" in res:
        for mname, clr, dash in model_styles:
            if mname in res["forecast_per_model"]:
                fig.add_trace(go.Scatter(
                    x=res["fut_ds"], y=res["forecast_per_model"][mname],
                    name=f"{mname} forecast", line=dict(color=clr, width=1.6, dash=dash),
                    mode="lines+markers", marker=dict(size=5, color=clr),
                    visible="legendonly",
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{mname}: %{{y:,.0f}}<extra></extra>",
                ))
    # Ensemble Forecast — always visible, bold
    fig.add_trace(go.Scatter(
        x=res["fut_ds"], y=res["forecast"], name="Ensemble Forecast",
        line=dict(color="#8B5CF6", width=3.0), mode="lines+markers",
        marker=dict(size=9, color="#8B5CF6", line=dict(color="#FFFFFF", width=2)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
    ))
    # Eval markers — hold-out predictions vs actuals
    fig.add_trace(go.Scatter(
        x=res["eval_ds"], y=res["eval_pred"], name="Hold-out eval",
        mode="markers",
        marker=dict(size=10, color="#EF4444", symbol="x-thin",
                    line=dict(color="#EF4444", width=2.5)),
        hovertemplate="<b>%{x|%b %Y}</b><br>Eval pred: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CD(), height=height, xaxis=gX(), yaxis=gY(),
        legend={**leg(), "traceorder": "normal"},
        title=dict(text=title, font=dict(color="#64748b", size=11)),
    )
    return fig

def model_grade(nrmse: float, r2: float) -> tuple:
    acc = max(0.0, round((1 - nrmse) * 100, 1))
    if   nrmse < 0.10 and r2 >= 0.95: g, l, icon = "A+", "Excellent", "✅"
    elif nrmse < 0.15 and r2 >= 0.90: g, l, icon = "A",  "Very Good", "✅"
    elif nrmse < 0.20 and r2 >= 0.85: g, l, icon = "B+", "Good",      "🟦"
    elif nrmse < 0.25 and r2 >= 0.75: g, l, icon = "B",  "Acceptable","⚠️"
    elif nrmse < 0.35 and r2 >= 0.60: g, l, icon = "C",  "Weak",      "⚠️"
    else:                              g, l, icon = "D",  "Poor",      "🔴"
    return g, l, icon, acc

def render_model_quality(res: dict) -> None:
    g, l, icon, acc = model_grade(res["nrmse"], res["r2"])
    if "model_metrics" in res:
        st.markdown("""<div style='font-size:11px;font-weight:700;color:#4a5e7a;
            letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px'>
            Individual Model Performance</div>""", unsafe_allow_html=True)
        mm   = res["model_metrics"]
        cols = st.columns(4)
        model_display = [
            ("Ridge",        "pill-ridge",    "#3B82F6"),
            ("RandomForest", "pill-rf",       "#22C55E"),
            ("GradBoost",    "pill-gb",       "#F59E0B"),
            ("Ensemble",     "pill-ensemble", "#8B5CF6"),
        ]
        for col, (mname, pcls, clr) in zip(cols, model_display):
            if mname in mm:
                m    = mm[mname]
                _acc = max(0.0, round((1 - m["nrmse"]) * 100, 1))
                _g, _l, _ico, _ = model_grade(m["nrmse"], m["r2"])
                col.markdown(
                    f"""<div style='text-align:center;padding:10px;border-radius:10px;
                        border:1px solid #e5e7eb;background:white'>
                        <div class='model-pill {pcls}'>{mname}</div>
                        <div style='font-size:10px;color:#64748b;margin-top:5px'>RMSE (hold-out)</div>
                        <div style='font-size:18px;font-weight:800;color:{clr}'>{m["rmse"]:.1f}</div>
                        <div style='font-size:10px;color:#94a3b8'>NRMSE {m["nrmse"]*100:.1f}% · R² {m["r2"]:.3f}</div>
                        <div style='font-size:12px;font-weight:700;color:{clr};margin-top:4px'>Accuracy {_acc:.1f}%</div>
                        <div style='font-size:10px;color:#94a3b8'>{_ico} {_g} {_l}</div>
                    </div>""", unsafe_allow_html=True,
                )
        w = res.get("weights", {})
        if w:
            tot = sum(w.values())
            st.markdown(
                f"""<div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:8px;
                    padding:8px 12px;font-size:11px;margin:6px 0'>
                    <b style='color:#1e3a8a'>Ensemble blend (inverse-RMSE):</b>
                    <span class='model-pill pill-ridge'>Ridge {w.get("Ridge",0)/tot*100:.0f}%</span>
                    <span class='model-pill pill-rf'>RF {w.get("RandomForest",0)/tot*100:.0f}%</span>
                    <span class='model-pill pill-gb'>GB {w.get("GradBoost",0)/tot*100:.0f}%</span>
                </div>""", unsafe_allow_html=True,
            )
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "RMSE",     f"{res['rmse']:.1f}",         "sky",  "hold-out")
    kpi(c2, "NRMSE",    f"{res['nrmse']*100:.1f}%",   "sky",  "normalised")
    kpi(c3, "MAE",      f"{res['mae']:.1f}",           "sky",  "mean abs err")
    kpi(c4, "R² Score", f"{res['r2']:.3f}",            "sky",  "fit quality")
    kpi(c5, "Accuracy", f"{acc:.1f}%",                 "mint", "1 − NRMSE")
    sp(0.5)
    st.markdown(
        f"""<div class='model-quality-card'>
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
        </div>""", unsafe_allow_html=True,
    )
    sp(0.5)

@st.cache_data
def compute_inventory(
    order_cost: float = DEFAULT_ORDER_COST,
    hold_pct:   float = DEFAULT_HOLD_PCT,
    lead_time:  int   = DEFAULT_LEAD_TIME,
    z:          float = DEFAULT_SERVICE_Z,
    n_future:   int   = N_FUTURE_MONTHS,
) -> pd.DataFrame:
    df          = load_data()
    ops         = get_ops(df).copy()
    ops["YM"]   = ops["Order_Date"].dt.to_period("M")
    del_ops     = get_delivered(df)
    cat_fcs     = compute_category_forecasts(n_future)
    lt_std_map  = del_ops.groupby("Category")["Delivery_Days"].std().fillna(1.0).to_dict()
    sku_monthly = (
        ops.groupby(["SKU_ID", "YM"])["Net_Qty"].sum().reset_index().sort_values(["SKU_ID", "YM"])
    )
    df_sorted    = df.sort_values("Order_Date")
    sku_snapshot = df_sorted.groupby("SKU_ID").agg(
        actual_stock = ("Current_Stock_Units", "last"),
        dataset_rop  = ("Reorder_Point",       "last"),
        Product_Name = ("Product_Name",        "first"),
        Category     = ("Category",            "first"),
        avg_price    = ("Sell_Price",          "mean"),
        total_qty    = ("Net_Qty",             "sum"),
    ).reset_index()
    sku_stats = (
        sku_monthly.groupby("SKU_ID")["Net_Qty"]
        .agg(hist_avg="mean", hist_std="std", peak_d="max")
        .reset_index()
    )
    sku_stats["hist_std"] = sku_stats["hist_std"].fillna(sku_stats["hist_avg"] * 0.25)
    cat_hist_avg: dict[str, float] = {
        cat: info["hist_avg"] for cat, info in cat_fcs.items()
    }
    sku_snapshot = sku_snapshot.merge(sku_stats, on="SKU_ID", how="left")
    sku_snapshot["hist_avg"]  = sku_snapshot["hist_avg"].fillna(0)
    sku_snapshot["hist_std"]  = sku_snapshot["hist_std"].fillna(0)
    sku_snapshot["peak_d"]    = sku_snapshot["peak_d"].fillna(0)
    def _sku_forecast(row):
        cat = row["Category"]
        h_avg = row["hist_avg"]
        if cat in cat_fcs and cat_hist_avg.get(cat, 0) > 0:
            share = h_avg / cat_hist_avg[cat]
            fc    = cat_fcs[cat]
            return (
                fc["mean"] * share,
                [v * share for v in fc["monthly"]],
                fc["mean"] * share * 0.70 + row["peak_d"] * DEMAND_PEAK_WEIGHT,
            )
        return h_avg, [h_avg] * n_future, h_avg * 0.60 + row["peak_d"] * 0.40
    demand_cols = sku_snapshot.apply(_sku_forecast, axis=1, result_type="expand")
    demand_cols.columns = ["avg_d", "fc_next6", "econ_d"]
    sku_snapshot = pd.concat([sku_snapshot, demand_cols], axis=1)
    uc        = sku_snapshot["avg_price"].clip(lower=1.0)
    ann_d     = sku_snapshot["econ_d"] * 12
    eoq = np.maximum(np.where(ann_d > 0, np.sqrt(2 * ann_d * order_cost / (uc * hold_pct)), 0), 1).astype(int)
    daily_d   = sku_snapshot["avg_d"] / 30.0
    daily_std = sku_snapshot["hist_std"] / np.sqrt(30)
    lt_std    = sku_snapshot["Category"].map(lt_std_map).fillna(1.0)
    ss = np.maximum((z * np.sqrt(lead_time * daily_std ** 2 + daily_d ** 2 * lt_std ** 2)).astype(int), 0)
    computed_rop = np.maximum((daily_d * lead_time + ss).astype(int), 1)
    rop          = np.maximum(sku_snapshot["dataset_rop"].astype(int), computed_rop)
    current_stock = sku_snapshot["actual_stock"].astype(int)
    demand_6m = sku_snapshot["fc_next6"].apply(
        lambda lst: int(round(sum(lst))) if isinstance(lst, list) else int(round(float(lst) * n_future))
    )
    demand_driven_need = np.maximum(demand_6m.values + ss - current_stock, 0)
    replenishment_need = np.maximum(rop + eoq - current_stock, 0)
    prod_need = np.maximum(demand_driven_need, replenishment_need)
    demand_cover_pct = np.where(
        demand_6m > 0, np.minimum(current_stock / demand_6m.values * 100, 100).round(1), 100.0,
    )
    status = np.where(current_stock <= ss, "🔴 Critical",
        np.where(current_stock < rop, "🟡 Low", "🟢 Adequate"),
    )
    days_stock   = np.where(daily_d > 0, (current_stock / daily_d).round(1), 999.0)
    units_below  = np.maximum(ss - current_stock, 0)
    daily_margin  = daily_d * uc * MARGIN_RATE
    stockout_cost = np.where(status == "🔴 Critical",
        np.round(units_below * uc * MARGIN_RATE + daily_margin * lead_time, 0), 0.0,
    )
    inv_df = pd.DataFrame({
        "SKU_ID":          sku_snapshot["SKU_ID"],
        "Product_Name":    sku_snapshot["Product_Name"],
        "Category":        sku_snapshot["Category"],
        "Monthly_Avg":     sku_snapshot["hist_avg"].round(1),
        "Forecast_Next6":  sku_snapshot["fc_next6"],
        "Demand_6M":       demand_6m,
        "Demand_Cover_Pct":demand_cover_pct,
        "EOQ":             eoq,
        "SS":              ss,
        "ROP":             rop,
        "Current_Stock":   current_stock,
        "Days_of_Stock":   days_stock,
        "Status":          status,
        "Unit_Price":      uc.round(0),
        "Stockout_Cost":   stockout_cost,
        "Prod_Need":       prod_need,
        "Total_Revenue":   (sku_snapshot["total_qty"] * uc).round(0),
    })
    inv_df = inv_df[inv_df["Monthly_Avg"] > 0].reset_index(drop=True)
    if inv_df.empty:
        return inv_df
    inv_df  = inv_df.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
    cum_pct = inv_df["Total_Revenue"].cumsum() / inv_df["Total_Revenue"].sum() * 100
    inv_df["ABC"] = np.where(cum_pct <= 70, "A", np.where(cum_pct <= 90, "B", "C"))
    return inv_df

@st.cache_data
def compute_production(cap_mult: float = 1.0, n_future: int = N_FUTURE_MONTHS) -> pd.DataFrame:
    inv     = compute_inventory(n_future=n_future)
    cat_fcs = compute_category_forecasts(n_future)
    rows = []
    for cat, fc_info in cat_fcs.items():
        fc_arr = np.array(fc_info["monthly"])
        ci_lo  = np.array(fc_info["ci_lo"])
        ci_hi  = np.array(fc_info["ci_hi"])
        fut_ds = fc_info["fut_ds"]
        cat_inv = inv[inv["Category"] == cat]
        if cat_inv.empty:
            continue
        prod_need_cat = float(cat_inv["Prod_Need"].sum())
        crit_skus = cat_inv[cat_inv["Status"] == "🔴 Critical"]
        low_skus  = cat_inv[cat_inv["Status"] == "🟡 Low"]
        crit_gap  = float((crit_skus["ROP"] - crit_skus["Current_Stock"]).clip(lower=0).sum())
        low_gap   = float((low_skus["ROP"]  - low_skus["Current_Stock"]).clip(lower=0).sum())
        current_stock_cat = float(cat_inv["Current_Stock"].sum())
        demand_6m_cat     = float(cat_inv["Demand_6M"].sum())
        forecast_total = float(fc_arr.sum())
        for i, (dt, fc) in enumerate(zip(fut_ds, fc_arr)):
            demand_share = fc / forecast_total if forecast_total > 0 else 1.0 / len(fc_arr)
            prod = prod_need_cat * demand_share * cap_mult
            bf         = BOOST_SCHEDULE.get(i, 0.0)
            crit_boost = crit_gap * bf
            low_boost  = low_gap  * bf * 0.5
            rows.append({
                "Month_dt":        dt,
                "Month":           dt.strftime("%b %Y"),
                "Category":        cat,
                "Demand_Forecast": round(fc, 0),
                "Current_Stock":   round(current_stock_cat, 0),
                "Demand_6M_Cat":   round(demand_6m_cat, 0),
                "Prod_Need_Cat":   round(prod_need_cat, 0),
                "Crit_Boost":      round(crit_boost, 0),
                "Low_Boost":       round(low_boost, 0),
                "Production":      round(prod, 0),
                "CI_Lo":           round(ci_lo[i], 0),
                "CI_Hi":           round(ci_hi[i], 0),
            })
    return pd.DataFrame(rows)

@st.cache_data
def build_sku_production_plan(n_future: int = N_FUTURE_MONTHS) -> pd.DataFrame:
    df     = load_data()
    del_df = get_delivered(df)
    inv    = compute_inventory(n_future=n_future)
    wh_cat = (del_df.groupby(["Category", "Warehouse"])["Quantity"].sum().reset_index())
    wh_cat["wh_share"] = wh_cat.groupby("Category")["Quantity"].transform(
        lambda x: x / x.sum()
    )
    avg_ship = (del_df.groupby(["Category", "Warehouse"]).agg(avg_cost=("Shipping_Cost_INR", "mean"))
        .reset_index().rename(columns={"Warehouse": "Target_Warehouse"})
    )
    needs = inv[inv["Prod_Need"] > 0].copy()
    abc_weight = {"A": 3, "B": 2, "C": 1}
    needs["ABC_Priority"] = needs["ABC"].map(abc_weight)
    needs["Daily_Demand"] = (needs["Monthly_Avg"] / 30).clip(lower=0.01)
    needs["Days_Left"]    = (needs["Current_Stock"] / needs["Daily_Demand"]).round(1).clip(upper=999)
    needs["Priority_Score"] = (
        needs["ABC_Priority"] * 3
        + needs["Stockout_Cost"] / 1000
        + (needs["ROP"] - needs["Current_Stock"]).clip(lower=0)
    )
    def _urgency(row):
        if row["Status"] == "🔴 Critical":   return "🔴 Urgent"
        if row["Days_Left"] <= 14:            return "🟠 High"
        if row["Days_Left"] <= 30:            return "🟡 Medium"
        return "🟢 Normal"
    needs["Urgency"] = needs.apply(_urgency, axis=1)
    wh_assignments = []
    for cat, grp in needs.groupby("Category"):
        cat_wh = (
            wh_cat[wh_cat["Category"] == cat].sort_values("wh_share", ascending=False).reset_index(drop=True)
        )
        warehouses = cat_wh["Warehouse"].tolist()
        shares     = cat_wh["wh_share"].values
        skus_sorted = grp.sort_values(["ABC_Priority", "Priority_Score"], ascending=[False, False])
        n = len(skus_sorted)
        cumulative = np.round(np.cumsum(shares) * n).astype(int).clip(0, n)
        prev = 0
        slot_assignments: list[str] = []
        for cut, wh in zip(cumulative, warehouses):
            count = int(cut) - prev
            slot_assignments.extend([wh] * count)
            prev = int(cut)
        while len(slot_assignments) < n:
            slot_assignments.append(warehouses[-1])
        for (idx, _), wh in zip(skus_sorted.iterrows(), slot_assignments):
            cat_share = float(
                cat_wh.loc[cat_wh["Warehouse"] == wh, "wh_share"].values[0]
                if wh in cat_wh["Warehouse"].values else shares[0]
            )
            wh_assignments.append({
                "idx":              idx,
                "Target_Warehouse": wh,
                "WH_Share_Pct":     round(cat_share * 100, 1),
            })
    wh_df = pd.DataFrame(wh_assignments).set_index("idx")
    needs = needs.join(wh_df[["Target_Warehouse", "WH_Share_Pct"]])
    needs["Target_Warehouse"] = needs["Target_Warehouse"].fillna("Central WH")
    needs["WH_Share_Pct"]     = needs["WH_Share_Pct"].fillna(100.0)
    needs = needs.merge(avg_ship, on=["Category", "Target_Warehouse"], how="left")
    needs["avg_cost"]      = needs["avg_cost"].fillna(del_df["Shipping_Cost_INR"].mean())
    needs["Est_Ship_Cost"] = (needs["Prod_Need"] * needs["avg_cost"]).round(0)
    wh_total = needs.groupby("Target_Warehouse")["Prod_Need"].transform("sum")
    needs["WH_Share_Pct"] = (needs["Prod_Need"] / wh_total.clip(lower=1) * 100).round(1)
    needs = needs.sort_values(["Priority_Score", "Days_Left"], ascending=[False, True]).reset_index(drop=True)
    return needs[[
        "SKU_ID", "Product_Name", "Category", "ABC", "Urgency", "Prod_Need",
        "Current_Stock", "Demand_6M", "Demand_Cover_Pct", "Days_Left",
        "Stockout_Cost", "Target_Warehouse", "WH_Share_Pct",
        "Est_Ship_Cost", "Status",
    ]]

@st.cache_data
def compute_logistics(
    w_speed:   float = DEFAULT_W_SPEED,
    w_cost:    float = DEFAULT_W_COST,
    w_returns: float = DEFAULT_W_RETURNS,
    n_future:  int   = N_FUTURE_MONTHS,
):
    df     = load_data()
    del_df = get_delivered(df)
    plan   = compute_production(n_future=n_future)
    carrier_returns = df.groupby("Courier_Partner")["Return_Flag"].mean().reset_index()
    carrier_returns.columns = ["Courier_Partner", "Return_Rate"]
    carr = del_df.groupby("Courier_Partner").agg(
        Orders   = ("Order_ID",          "count"),
        Avg_Days = ("Delivery_Days",     "mean"),
        Avg_Cost = ("Shipping_Cost_INR", "mean"),
    ).reset_index()
    carr = carr.merge(carrier_returns, on="Courier_Partner", how="left")
    carr["Return_Rate"] = carr["Return_Rate"].fillna(0)
    for col in ["Avg_Days", "Avg_Cost", "Return_Rate"]:
        mn = carr[col].min(); mx = carr[col].max()
        carr[f"Norm_{col}"] = 1 - (carr[col] - mn) / (mx - mn + 1e-9)
    carr["Perf_Score"] = (
        w_speed   * carr["Norm_Avg_Days"]
        + w_cost  * carr["Norm_Avg_Cost"]
        + w_returns * carr["Norm_Return_Rate"]
    ).round(3)
    cheapest = (del_df.groupby(["Region", "Courier_Partner"])
        .agg(avg_cost=("Shipping_Cost_INR", "mean"), orders=("Order_ID", "count"))
        .reset_index().sort_values("avg_cost")
        .groupby("Region").first().reset_index()
        .rename(columns={"Courier_Partner": "Optimal_Carrier", "avg_cost": "Min_Avg_Cost"})
    )
    region_costs = del_df.groupby("Region").agg(
        Current_Avg_Cost = ("Shipping_Cost_INR", "mean"),
        Orders           = ("Order_ID",          "count"),
        Total_Spend      = ("Shipping_Cost_INR", "sum"),
    ).reset_index()
    opt = region_costs.merge(cheapest[["Region", "Optimal_Carrier", "Min_Avg_Cost"]], on="Region")
    opt["Potential_Saving"] = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) * opt["Orders"]).round(0)
    opt["Saving_Pct"] = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) / opt["Current_Avg_Cost"] * 100).round(1)
    avg_ship_unit = max(del_df["Shipping_Cost_INR"].sum() / max(del_df["Quantity"].replace(0, np.nan).sum(), 1), 1.0)
    hist_orders = max(len(del_df), 1)
    avg_units_ord = max(del_df["Quantity"].sum() / hist_orders, 1.0)
    fwd_rows = []
    if not plan.empty:
        for _, row in plan.iterrows():
            fc_units = row["Demand_Forecast"]
            fwd_rows.append({
                "Month_dt":      row["Month_dt"],
                "Month":         row["Month"],
                "Category":      row["Category"],
                "Prod_Units":    int(row["Production"]),
                "Proj_Orders":   int(round(fc_units / avg_units_ord)),
                "Proj_Ship_Cost":int(round(fc_units * avg_ship_unit, 0)),
                "CI_Lo_Units":   int(row["CI_Lo"]),
                "CI_Hi_Units":   int(row["CI_Hi"]),
            })
    return carr, opt, pd.DataFrame(fwd_rows)

def page_overview() -> None:
    df  = load_data()
    ops = get_ops(df).copy()
    del_df = get_delivered(df)
    total_orders  = len(df)
    total_rev     = ops["Net_Revenue"].sum()
    avg_ov        = ops["Net_Revenue"].mean()
    ret_rate      = df["Return_Flag"].mean() * 100
    on_time       = (del_df["Delivery_Days"] <= 3).mean() * 100
    n_skus        = df["SKU_ID"].nunique()
    st.markdown("""
     <div style='background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);border-radius:18px;
         padding:30px 32px;margin-bottom:24px;'>
      <div style='font-size:38px;font-weight:900;color:white;letter-spacing:-.02em;
           text-transform:uppercase;line-height:1.1'>OmniFlow D2D Intelligence</div>
      <div style='font-size:11px;font-family:DM Mono,monospace;color:#93c5fd;letter-spacing:.14em;
           text-transform:uppercase;margin-top:6px;margin-bottom:4px'>
        AI Driven Demand to Delivery Optimization System
      </div>
     </div>""", unsafe_allow_html=True)
    sec("Dataset at a Glance")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, "Total Orders", f"{total_orders:,}", "sky",   "Jan 2024 – Dec 2025")
    kpi(k2, "Net Revenue", f"₹{total_rev/1e7:.2f}Cr", "mint", "delivered + shipped")
    kpi(k3, "Avg Order Value", f"₹{avg_ov:,.0f}",         "sky", "per active order")
    kpi(k4, "Return Rate", f"{ret_rate:.1f}%", "coral", f"{df['Return_Flag'].sum()} orders")
    kpi(k5, "On-Time Delivery", f"{on_time:.1f}%", "mint", "delivered ≤ 3 days")
    kpi(k6, "Unique SKUs", str(n_skus), "sky", "across 4 categories")
    sp(0.5)
    st.markdown("""
    <div class='about-section'>
    <div style='font-size:16px;font-weight:900;margin-bottom:14px'>Dataset Scope</div>
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px'>
      <div class='card'>
        <div style='font-size:12px;font-weight:800;color:#1e3a8a;margin-bottom:8px'>Orders & Revenue</div>
        <div style='font-size:12px;line-height:1.8;color:#475569'>
          <b>5,010 orders</b> · Jan 2024 – Dec 2025<br>
          73.9% Delivered · 12.3% Shipped<br>
          9.3% Returned · 4.5% Cancelled<br>
          Avg order value: <b>₹8,159</b>
        </div>
      </div>
      <div class='card'>
        <div style='font-size:12px;font-weight:800;color:#1e3a8a;margin-bottom:8px'>Sales Channels</div>
        <div style='font-size:12px;line-height:1.8;color:#475569'>
          <b>Amazon.in</b> — 2,099 orders (41.9%)<br>
          <b>Shiprocket</b> — 1,761 orders (35.1%)<br>
          <b>INCREFF B2B</b> — 1,150 orders (23.0%)<br>
          Multi-channel D2C + B2B mix
        </div>
      </div>
      <div class='card'>
        <div style='font-size:12px;font-weight:800;color:#1e3a8a;margin-bottom:8px'>Geography</div>
        <div style='font-size:12px;line-height:1.8;color:#475569'>
          <b>9 Indian regions:</b><br>
          Maharashtra · Delhi · Uttar Pradesh<br>
          Karnataka · Gujarat · Tamil Nadu<br>
          Telangana · West Bengal · Rajasthan
        </div>
      </div>
      <div class='card'>
        <div style='font-size:12px;font-weight:800;color:#1e3a8a;margin-bottom:8px'>Operations</div>
        <div style='font-size:12px;line-height:1.8;color:#475569'>
          <b>4 Warehouses:</b> Delhi · Mumbai · Bengaluru · Hyderabad<br>
          <b>5 Carriers:</b> BlueDart · Delhivery · DTDC · Ecom Express · XpressBees<br>
          Avg delivery: <b>2.2 days</b>
        </div>
      </div>
      <div class='card'>
        <div style='font-size:12px;font-weight:800;color:#1e3a8a;margin-bottom:8px'>Products</div>
        <div style='font-size:12px;line-height:1.8;color:#475569'>
          <b>50 SKUs</b> across 4 categories:<br>
          Electronics & Mobiles (dominant)<br>
          Fashion & Apparel · Home & Kitchen<br>
          Health & Personal Care
        </div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class='about-section'>
    <div style='font-size:16px;font-weight:900;margin-bottom:6px'>Analytics Pipeline — What Each Module Decides</div>
    <div style='font-size:12px;color:#64748b;margin-bottom:14px'>
      Modules are chained: demand forecast drives inventory, inventory drives production, production drives logistics routing.
    </div>
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px'>
      <div class='card' style='border-top:3px solid #3b82f6'>
        <div style='font-size:11px;font-weight:800;color:#3b82f6;letter-spacing:.06em;text-transform:uppercase'>1 · Demand Forecasting</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>How much will sell?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Ridge + Random Forest + Gradient Boosting <b>ensemble</b> forecasts orders, quantity and revenue
          for the selected horizon — by overall, category, region and sales channel. Outputs a 90% confidence interval.
        </div>
      </div>
      <div class='card' style='border-top:3px solid #f59e0b'>
        <div style='font-size:11px;font-weight:800;color:#f59e0b;letter-spacing:.06em;text-transform:uppercase'>2 · Inventory Optimisation</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>Which SKUs need restocking?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Wilson EOQ formula computes optimal order batch. Safety stock protects against demand variance.
          ROP triggers reorder. SKUs are ABC-classified. Demand window matches selected forecast horizon.
        </div>
      </div>
      <div class='card' style='border-top:3px solid #8b5cf6'>
        <div style='font-size:11px;font-weight:800;color:#8b5cf6;letter-spacing:.06em;text-transform:uppercase'>3 · Production Planning</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>How many units to make, when?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Monthly production targets derived from forecasted demand. Horizon matches global selection.
          Critical/Low SKUs get urgency boosts into Month 1–2. SKUs are routed to warehouses proportionally.
        </div>
      </div>
      <div class='card' style='border-top:3px solid #059669'>
        <div style='font-size:11px;font-weight:800;color:#059669;letter-spacing:.06em;text-transform:uppercase'>4 · Logistics Optimisation</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>Which carrier, at what cost?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Carrier composite score = weighted(speed + cost + return rate). Identifies cheapest carrier
          per region and projects forward shipping cost based on the production plan horizon.
        </div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)

def page_demand() -> None:
    n_future = get_horizon()
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    st.markdown("<div class='page-title'>Demand Forecasting</div>", unsafe_allow_html=True)
    horizon_badge(n_future)
    sec("Ensemble Model Quality")
    m_orders = ops.groupby("YM")["Order_ID"].count().rename("v")
    res_ov   = ml_forecast(m_orders.values.astype(float), m_orders.index, n_future)
    if res_ov:
        render_model_quality(res_ov)
    sp()
    if res_ov and "model_metrics" in res_ov:
        sec("Model Accuracy Comparison")
        mm     = res_ov["model_metrics"]
        labels = [m for m in ["Ridge", "RandomForest", "GradBoost", "Ensemble"] if m in mm]
        r2_vals    = [mm[m]["r2"]          for m in labels]
        nrmse_vals = [mm[m]["nrmse"] * 100 for m in labels]
        clrs       = [MODEL_COLORS.get(m, "#888") for m in labels]
        bc1, bc2 = st.columns(2, gap="large")
        with bc1:
            fig = go.Figure(go.Bar(
                x=labels, y=r2_vals,
                marker=dict(color=clrs, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
                textfont=dict(color="#334155"),
            ))
            fig.add_hline(y=0.9, line_dash="dash", line_color="#22C55E",
                          annotation_text=" Target R²=0.90", annotation_font=dict(color="#22C55E", size=10))
            fig.update_layout(**CD(), height=240, xaxis=gX(),
                              yaxis={**gY(), "title": "R² Score", "range": [0, 1.25]},
                              title=dict(text="R² Score (higher = better)", font=dict(size=11, color="#64748b")))
            st.plotly_chart(fig, use_container_width=True, key="d_r2")
        with bc2:
            fig2 = go.Figure(go.Bar(
                x=labels, y=nrmse_vals,
                marker=dict(color=clrs, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in nrmse_vals], textposition="outside",
                textfont=dict(color="#334155"),
            ))
            fig2.add_hline(y=15, line_dash="dash", line_color="#22C55E",
                           annotation_text=" Target <15%", annotation_font=dict(color="#22C55E", size=10))
            fig2.update_layout(**CD(), height=240, xaxis=gX(),
                               yaxis={**gY(), "title": "NRMSE (%)", "autorange": True},
                               title=dict(text="NRMSE % (lower = better)", font=dict(size=11, color="#64748b")))
            st.plotly_chart(fig2, use_container_width=True, key="d_nrmse")
    sp()
    c1, c2 = st.columns([2, 2])
    metric_opt = c1.selectbox("Metric", ["Orders", "Quantity", "Net Revenue"], key="d_metric")
    level_opt  = c2.selectbox("Breakdown", ["Overall", "Category", "Region", "Sales Channel"], key="d_level")
    col_map = {"Orders": "Order_ID", "Quantity": "Net_Qty", "Net Revenue": "Net_Revenue"}
    col     = col_map[metric_opt]
    def get_series(sub):
        if col == "Order_ID":
            return sub.groupby("YM")["Order_ID"].count().rename("v")
        return sub.groupby("YM")[col].sum().rename("v")
    def draw_with_table(series, title: str = "", chart_key: str = "d_main") -> None:
        res = ml_forecast(series.values.astype(float), series.index, n_future=n_future)
        if res is None:
            st.info("Insufficient data.")
            return
        fig = ensemble_chart(res, chart_key=chart_key, height=310, title=title)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
        tbl = pd.DataFrame({
            "Month":        [d.strftime("%b %Y") for d in res["fut_ds"]],
            "Ensemble":     res["forecast"].round(0).astype(int),
            "Ridge":        np.maximum(res["forecast_per_model"]["Ridge"],        0).round(0).astype(int),
            "RandomForest": np.maximum(res["forecast_per_model"]["RandomForest"], 0).round(0).astype(int),
            "GradBoost":    np.maximum(res["forecast_per_model"]["GradBoost"],    0).round(0).astype(int),
            "Lower 90%":    res["ci_lo"].round(0).astype(int),
            "Upper 90%":    res["ci_hi"].round(0).astype(int),
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    sec(f"Forecast Chart — {n_future}-Month Horizon")
    if level_opt == "Overall":
        draw_with_table(get_series(ops), chart_key="d_overall")
    else:
        grp_map = {"Category": "Category", "Region": "Region", "Sales Channel": "Sales_Channel"}
        grp     = grp_map[level_opt]
        top     = ops[grp].value_counts().index.tolist()
        tabs    = st.tabs(top)
        for i, (tab, val) in enumerate(zip(tabs, top)):
            with tab:
                draw_with_table(get_series(ops[ops[grp] == val]), title=val, chart_key=f"d_bd_{i}")
    sp()
    sec("YoY Revenue Growth by Category")
    yr_rev     = ops.groupby(["Year", "Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    cat_monthly = ops.groupby(["YM", "Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    proj_next: dict[str, float] = {}
    for cat in cat_monthly.columns:
        r = ml_forecast(cat_monthly[cat].values.astype(float), cat_monthly.index, n_future)
        if r:
            proj_next[cat] = float(r["forecast"].sum())
    if 2024 in yr_rev.index and 2025 in yr_rev.index:
        rows = []
        for cat in yr_rev.columns:
            r24 = yr_rev.loc[2024, cat]; r25 = yr_rev.loc[2025, cat]; rp = proj_next.get(cat, 0)
            rows.append({
                "Category":              cat,
                "2024 ₹M":               round(r24 / 1e6, 1),
                "2025 ₹M":               round(r25 / 1e6, 1),
                "YoY 24→25":             f"{(r25-r24)/r24*100:+.1f}%" if r24 > 0 else "N/A",
                f"Next {n_future}M Proj ₹M": round(rp / 1e6, 1),
                "Projected Growth":      f"{(rp-r25)/r25*100:+.1f}%" if r25 > 0 else "N/A",
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values(f"Next {n_future}M Proj ₹M", ascending=False),
            use_container_width=True, hide_index=True,
        )

def page_inventory() -> None:
    n_future = get_horizon()
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    st.markdown("<div class='page-title'>Inventory Optimization</div>", unsafe_allow_html=True)
    horizon_badge(n_future)
    with st.expander("Parameters", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        order_cost = p1.number_input("Order Cost", 100, 5000, DEFAULT_ORDER_COST, 50)
        hold_pct   = p2.slider("Holding Cost %", 5, 40, int(DEFAULT_HOLD_PCT * 100)) / 100
        lead_time  = p3.slider("Lead Time days", 1, 30, DEFAULT_LEAD_TIME)
        svc        = p4.selectbox("Service Level", ["90% (z=1.28)", "95% (z=1.65)", "99% (z=2.33)"], index=1)
        z          = {"90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65, "99% (z=2.33)": 2.33}[svc]
    inv = compute_inventory(order_cost, hold_pct, lead_time, z, n_future)
    if inv.empty:
        st.warning("No inventory data.")
        return
    n_crit          = (inv["Status"] == "🔴 Critical").sum()
    n_low           = (inv["Status"] == "🟡 Low").sum()
    total_prod_need = int(inv["Prod_Need"].sum())
    total_demand_6m = int(inv["Demand_6M"].sum())
    ops_ym   = ops["YM"].max()
    fc_start = (ops_ym + 1).to_timestamp().strftime("%b %Y")
    fc_end   = (ops_ym + n_future).to_timestamp().strftime("%b %Y")
    fc_range = f"{fc_start} – {fc_end}"
    crit_days = inv[inv["Status"] == "🔴 Critical"]["Days_of_Stock"]
    crit_days = crit_days[crit_days < 999]
    avg_crit_days = f"{crit_days.mean():.0f}d avg" if len(crit_days) > 0 else "—"
    low_days  = inv[inv["Status"] == "🟡 Low"]["Days_of_Stock"]
    low_days  = low_days[low_days < 999]
    avg_low_days = f"{low_days.mean():.0f}d avg" if len(low_days) > 0 else "—"
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Total SKUs",          len(inv),                  "sky",   "active SKUs")
    kpi(c2, "🔴 Critical SKUs",    n_crit,                    "coral", f"stock ≤ safety stock · {avg_crit_days}")
    kpi(c3, "🟡 Low Stock",        n_low,                     "amber", f"stock < reorder point · {avg_low_days}")
    kpi(c4, f"{n_future}M Forecast Demand", f"{total_demand_6m:,}", "sky", f"units · {fc_range}")
    kpi(c5, "Units to Produce",    f"{total_prod_need:,}",    "mint",  f"to meet demand by {fc_end}")
    sp()
    sec("Stock Position")
    sc1, sc2, sc3 = st.columns([2, 2, 1])
    cat_f  = sc1.multiselect("Category", sorted(inv["Category"].unique()),
                             default=sorted(inv["Category"].unique()), key="al_cat")
    stat_f = sc2.multiselect("Status", ["🔴 Critical", "🟡 Low", "🟢 Adequate"],
                             default=["🔴 Critical", "🟡 Low", "🟢 Adequate"], key="al_stat")
    abc_f = sc3.multiselect("ABC", ["A", "B", "C"], default=["A", "B", "C"], key="al_abc")
    sv = inv[ inv["Category"].isin(cat_f) & inv["Status"].isin(stat_f) & inv["ABC"].isin(abc_f)].copy()
    if sv.empty:
        banner("✅ No SKUs match selected filters.", "mint")
    else:
        STATUS_CLR = {
            "🔴 Critical":    "#ef4444",
            "🟡 Low":         "#f59e0b",
            "🟢 Adequate":    "#22c55e",
            "🟢 Overstocked": "#06b6d4",
        }
        fig_sc = go.Figure()
        ax_max = max(sv["Current_Stock"].max(), sv["ROP"].max()) * 1.1
        fig_sc.add_trace(go.Scatter(
            x=[0, ax_max], y=[0, ax_max], mode="lines",
            line=dict(color="rgba(100,116,139,0.25)", width=1.5, dash="dash"),
            name="Stock = ROP", hoverinfo="skip",
        ))
        fig_sc.add_vrect(x0=0, x1=sv["ROP"].mean(), fillcolor="rgba(239,68,68,0.04)", layer="below", line_width=0)
        for status, clr in STATUS_CLR.items():
            grp = sv[sv["Status"] == status]
            if grp.empty:
                continue
            bubble_sz = np.clip(grp["Prod_Need"].values, 8, 60)
            fig_sc.add_trace(go.Scatter(
                x=grp["Current_Stock"], y=grp["ROP"],
                mode="markers", name=status,
                marker=dict(
                    size=bubble_sz, color=clr, opacity=0.82,
                    line=dict(color="#FFFFFF", width=1.5),
                    sizemode="area", sizeref=2.0 * 60 / (40.0 ** 2), sizemin=6,
                ),
                customdata=grp[["Product_Name", "SKU_ID", "Prod_Need", "Demand_6M", "Demand_Cover_Pct"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "SKU: %{customdata[1]}<br>"
                    "Stock: %{x}<br>"
                    "ROP: %{y}<br>"
                    f"{n_future}M Demand: %{{customdata[3]:,}} units<br>"
                    "Stock Covers: %{customdata[4]:.0f}% of demand<br>"
                    "Produce: <b>%{customdata[2]} units</b>"
                ),
            ))
        fig_sc.update_layout(
            **CD(), height=400,
            xaxis={**gX(), "title": "Current Stock (units)"},
            yaxis={**gY(), "title": "Reorder Point (units)"},
            legend={**leg(), "orientation": "h", "y": -0.18},
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="scatter_stock")
        action = sv.sort_values(["Status", "Prod_Need"], ascending=[True, False])
        if not action.empty:
            sp(0.5)
            sec("SKU Inventory Table — Action Queue")
            tbl = action[[
                "SKU_ID", "Product_Name", "Category", "ABC", "Status",
                "Current_Stock", "Demand_6M", "Demand_Cover_Pct",
                "ROP", "EOQ", "SS", "Prod_Need",
            ]].copy()
            tbl.columns = [
                "SKU", "Product", "Category", "ABC", "Status",
                "Stock", f"{n_future}M Demand", "Covers %",
                "ROP", "EOQ", "Safety Stock", "Units to Produce",
            ]
            for c in ["Stock", f"{n_future}M Demand", "ROP", "EOQ", "Safety Stock", "Units to Produce"]:
                tbl[c] = tbl[c].astype(int)
            tbl["Covers %"] = tbl["Covers %"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=340)

def page_production() -> None:
    n_future = get_horizon()
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    st.markdown("<div class='page-title'>Production Planning</div>", unsafe_allow_html=True)
    horizon_badge(n_future)
    cap = st.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1)
    plan = compute_production(cap, n_future)
    if plan.empty:
        st.warning("Insufficient data.")
        return
    agg = plan.groupby("Month_dt")[["Production", "Demand_Forecast", "Crit_Boost", "Low_Boost"]].sum().reset_index()
    inv_for_kpi = compute_inventory(n_future=n_future)
    total_prod_need_inv = int(inv_for_kpi["Prod_Need"].sum())
    total_demand_6m_inv = int(inv_for_kpi["Demand_6M"].sum())
    total_stock_inv     = int(inv_for_kpi["Current_Stock"].sum())
    # Both Production Required and Prod_Need must come from the same source (inv_for_kpi)
    # to avoid rounding drift from compute_production's monthly demand-share allocation.
    peak = agg.loc[agg["Production"].idxmax(), "Month_dt"]
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Units to Produce",          f"{total_prod_need_inv:,}",    "amber", f"gap: demand + SS − stock")
    kpi(c2, f"{n_future}M Forecast Demand", f"{total_demand_6m_inv:,}", "sky",   "what customers will order")
    kpi(c3, "Current Stock Total",       f"{total_stock_inv:,}",        "sky",   "across all SKUs")
    kpi(c4, "Scheduled Production",      f"{plan['Production'].sum():,.0f}", "amber", f"cap x{cap:.1f} · {n_future} months")
    kpi(c5, "Peak Month", peak.strftime("%b %Y"), "coral", "highest scheduled volume")
    sp()
    sec(f"Production Target vs Ensemble Demand Forecast — {n_future}-Month Horizon")
    hist_qty       = ops.groupby("YM")["Net_Qty"].sum().rename("v")
    hist_ts        = _to_ts(hist_qty.index)
    forecast_start = agg["Month_dt"].min()
    res_hist       = ml_forecast(hist_qty.values.astype(float), hist_qty.index, n_future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_ts, y=hist_qty.values, name="Historical Demand",
        fill="tozeroy", fillcolor="rgba(74,94,122,0.10)", line=dict(color="#4a5e7a", width=2),
    ))
    if res_hist:
        fig.add_trace(go.Scatter(
            x=res_hist["hist_ds"], y=res_hist["fitted"], name="Ensemble Fit",
            line=dict(color="#8B5CF6", width=1.5, dash="dot"), opacity=0.6,
        ))
    fig.add_trace(go.Bar(
        x=agg["Month_dt"], y=agg["Production"], name="Production Target",
        marker=dict(color="#8B5CF6", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{int(v):,}" for v in agg["Production"]],
        textposition="inside", textfont=dict(color="white", size=9),
    ))
    fig.add_trace(go.Scatter(
        x=agg["Month_dt"], y=agg["Demand_Forecast"], name="Ensemble Demand Forecast",
        mode="lines+markers", line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=8, color="#F59E0B", line=dict(color="#FFFFFF", width=2)),
    ))
    if res_hist:
        x_ci = list(res_hist["fut_ds"]) + list(res_hist["fut_ds"])[::-1]
        y_ci = list(res_hist["ci_hi"])  + list(res_hist["ci_lo"])[::-1]
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci, fill="toself",
            fillcolor="rgba(139,92,246,0.07)", line=dict(color="rgba(0,0,0,0)"), name="90% CI",
        ))
    fig.add_vline(x=forecast_start, line_dash="dash", line_color="rgba(139,92,246,0.5)", line_width=2)
    fig.update_layout(**CD(), height=320, xaxis=gX(), yaxis=gY(), legend=leg())
    st.plotly_chart(fig, use_container_width=True, key="prod_main")

    cl, cr = st.columns(2, gap="large")
    with cl:
        sec("Production by Category")
        cat_hist    = ops.groupby(["YM", "Category"])["Net_Qty"].sum().unstack(fill_value=0)
        cat_hist_ts = _to_ts(cat_hist.index)
        fig2 = go.Figure()
        fig2.add_vrect(x0=plan["Month_dt"].min(), x1=plan["Month_dt"].max(),
                       fillcolor="rgba(139,92,246,0.04)", layer="below", line_width=0)
        fig2.add_vline(x=plan["Month_dt"].min(), line_dash="dash",
                       line_color="rgba(139,92,246,0.4)", line_width=1.5)
        for i, cat in enumerate(plan["Category"].unique()):
            clr = COLORS[i % len(COLORS)]
            if cat in cat_hist.columns:
                fig2.add_trace(go.Scatter(
                    x=cat_hist_ts, y=cat_hist[cat].values, name=f"{cat} hist",
                    line=dict(color=clr, width=1.5, dash="dot"), opacity=0.55, showlegend=False,
                ))
            s = plan[plan["Category"] == cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(
                x=s["Month_dt"], y=s["Production"], name=cat,
                marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{int(v):,}" for v in s["Production"]],
                textposition="inside", textfont=dict(color="white", size=8),
            ))
        fig2.update_layout(**CD(), height=270, barmode="stack", xaxis=gX(), yaxis=gY(),
                           legend={**leg(), "orientation": "h", "y": -0.32})
        st.plotly_chart(fig2, use_container_width=True, key="prod_cat")
    with cr:
        sec("Production vs Demand Gap")
        agg["Gap"] = agg["Production"] - agg["Demand_Forecast"]
        fig3 = go.Figure(go.Bar(
            x=agg["Month_dt"], y=agg["Gap"],
            marker=dict(color=["#22C55E" if g >= 0 else "#EF4444" for g in agg["Gap"]],
                        line=dict(color="rgba(0,0,0,0)")),
            text=[f"{g:+.0f}" for g in agg["Gap"]], textposition="outside",
            textfont=dict(color="#334155"),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
        fig3.update_layout(**CD(), height=270, xaxis=gX(),
                           yaxis={**gY(), "title": "Units Surplus / Deficit"})
        st.plotly_chart(fig3, use_container_width=True, key="prod_gap")
    sec("Production Schedule")
    cat_f = st.selectbox("Filter Category", ["All"] + list(plan["Category"].unique()))
    d2    = plan if cat_f == "All" else plan[plan["Category"] == cat_f]
    d3    = d2[[
        "Month", "Category",
        "Current_Stock", "Demand_6M_Cat", "Prod_Need_Cat",
        "Demand_Forecast", "Crit_Boost", "Low_Boost",
        "Production", "CI_Lo", "CI_Hi",
    ]].copy()
    d3.columns = [
        "Month", "Category",
        "Cat Stock", f"{n_future}M Demand", "Inv Prod Need",
        "Demand Fc", "Crit Boost", "Low Boost",
        "Production", "Demand Lo", "Demand Hi",
    ]
    st.dataframe(d3.sort_values("Month"), use_container_width=True, hide_index=True)
    if cat_f != "All":
        filtered_prod   = int(d2["Production"].sum())
        filtered_demand = int(d2["Demand_Forecast"].sum())
        filtered_need   = int(d2["Prod_Need_Cat"].iloc[0]) if not d2.empty else 0
        banner(
            f"<b>Filtered subtotal — {cat_f}:</b> &nbsp;"
            f"Inv Prod Need = <b>{filtered_need:,} units</b> &nbsp;|&nbsp; "
            f"Production ({n_future} mo) = <b>{filtered_prod:,} units</b> &nbsp;|&nbsp; "
            f"Demand Forecast = <b>{filtered_demand:,} units</b> &nbsp;|&nbsp; "
            f"Overall (all categories) = <b>{int(plan['Production'].sum()):,} units</b>",
            "sky",
        )
    sp()
    st.markdown("<div style='font-size:22px;font-weight:900;color:black;letter-spacing:-.02em'>Fulfillment & Routing Plan</div>",
                unsafe_allow_html=True)
    sku_plan = build_sku_production_plan(n_future)
    if sku_plan.empty:
        banner("✅ All SKUs are adequately stocked — no production orders needed.", "mint")
        return
    n_urgent      = (sku_plan["Urgency"] == "🔴 Urgent").sum()
    n_high        = (sku_plan["Urgency"] == "🟠 High").sum()
    total_units   = int(sku_plan["Prod_Need"].sum())
    total_ship    = sku_plan["Est_Ship_Cost"].sum()
    stockout_risk = sku_plan["Stockout_Cost"].sum()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, "SKUs Needing Stock",  len(sku_plan),           "sky",   "Prod_Need > 0")
    kpi(k2, "🔴 Urgent",           n_urgent,                "coral", "stock ≤ safety stock")
    kpi(k3, "🟠 High",             n_high,                  "amber", "≤14 days stock left")
    kpi(k4, "Gap Units Total",     f"{total_units:,}",      "sky",   "demand-driven prod need")
    kpi(k5, "Est. Ship Cost",      f"₹{total_ship:,.0f}",  "amber", "to target warehouses")
    kpi(k6, "Stockout Risk",       f"₹{stockout_risk:,.0f}","coral", "if not restocked")
    sp(0.5)
    pt2, pt3 = st.tabs(["Warehouse Routing", "Visual Analysis"])
    with pt2:
        sec("Warehouse Stock Needs & Routing Plan")
        wh_dist = (sku_plan.groupby("Target_Warehouse")
            .agg(SKUs=("SKU_ID", "count"), Units=("Prod_Need", "sum"))
            .reset_index().sort_values("Units", ascending=False)
        )
        n_warehouses = len(wh_dist)
        wh_colors = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
                     "#059669", "#10b981", "#34d399"][:n_warehouses]
        fig_wh_dist = go.Figure()
        fig_wh_dist.add_trace(go.Bar(
            x=wh_dist["Target_Warehouse"], y=wh_dist["Units"],
            name="Units", marker=dict(color=wh_colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{int(v):,}" for v in wh_dist["Units"]], textposition="outside",
            textfont=dict(color="#334155"),
        ))
        fig_wh_dist.add_trace(go.Scatter(
            x=wh_dist["Target_Warehouse"], y=wh_dist["SKUs"],
            name="SKU count", yaxis="y2", mode="markers+text",
            marker=dict(size=14, color="#f59e0b", line=dict(color="#fff", width=2)),
            text=[f"{v} SKUs" for v in wh_dist["SKUs"]],
            textposition="top center", textfont=dict(size=9, color="#d97706"),
        ))
        fig_wh_dist.update_layout(
            **CD(), height=240, barmode="relative",
            xaxis=gX(),
            yaxis={**gY(), "title": "Units to Receive"},
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title="SKU Count", tickcolor="#d97706", range=[0, wh_dist["SKUs"].max() * 3]),
            legend={**leg(), "orientation": "h", "y": -0.28},
            title=dict(text=f"Inbound units split across {n_warehouses} warehouse(s)", font=dict(size=11, color="#64748b")),
        )
        st.plotly_chart(fig_wh_dist, use_container_width=True, key="wh_dist_bar")
        sp(0.5)
        wh_agg = (
            sku_plan.groupby("Target_Warehouse")
            .agg(
                SKUs           = ("SKU_ID",        "count"),
                Total_Units    = ("Prod_Need",      "sum"),
                Urgent_SKUs    = ("Urgency",        lambda x: (x == "🔴 Urgent").sum()),
                High_SKUs      = ("Urgency",        lambda x: (x == "🟠 High").sum()),
                Total_Ship_Cost= ("Est_Ship_Cost",  "sum"),
                Categories     = ("Category",       lambda x: ", ".join(sorted(x.unique()))),
                Avg_Days_Left  = ("Days_Left",      lambda x: x[x < 999].mean()),
            )
            .reset_index().sort_values("Urgent_SKUs", ascending=False)
        )
        wh_cols = st.columns(min(len(wh_agg), 4), gap="medium")
        for col, (_, wh) in zip(wh_cols, wh_agg.iterrows()):
            if wh["Urgent_SKUs"] > 0:
                urgency_badge = f"<span style='background:#fee2e2;color:#dc2626;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>🔴 {int(wh['Urgent_SKUs'])} urgent</span>"
            elif wh["High_SKUs"] > 0:
                urgency_badge = f"<span style='background:#fff7ed;color:#d97706;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>🟠 {int(wh['High_SKUs'])} high</span>"
            else:
                urgency_badge = "<span style='background:#f0fdf4;color:#15803d;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>🟢 Scheduled</span>"
            col.markdown(f"""
            <div style='background:white;border:1px solid #e5e7eb;border-radius:14px;
                 padding:18px;box-shadow:0 4px 16px rgba(0,0,0,0.07);height:100%'>
              <div style='font-size:16px;font-weight:900;color:#0f172a;margin-bottom:4px'>{wh["Target_Warehouse"]}</div>
              <div style='margin-bottom:10px'>{urgency_badge}</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                <div style='background:#f8fafc;border-radius:8px;padding:8px;text-align:center'>
                  <div style='font-size:9px;color:#94a3b8;text-transform:uppercase;font-family:DM Mono'>SKUs Inbound</div>
                  <div style='font-size:24px;font-weight:900;color:#1e3a8a'>{int(wh["SKUs"])}</div>
                </div>
                <div style='background:#f8fafc;border-radius:8px;padding:8px;text-align:center'>
                  <div style='font-size:9px;color:#94a3b8;text-transform:uppercase;font-family:DM Mono'>Units Needed</div>
                  <div style='font-size:24px;font-weight:900;color:#059669'>{int(wh["Total_Units"]):,}</div>
                </div>
              </div>
              <div style='margin-top:10px;font-size:10px;color:#64748b;font-family:DM Mono'>
                <div>Ship Cost: <b style='color:#0f172a'>₹{int(wh["Total_Ship_Cost"]):,}</b></div>
                <div style='margin-top:3px'>Categories: <b style='color:#0f172a'>{wh["Categories"]}</b></div>
                <div style='margin-top:3px'>Avg days left: <b style='color:#d97706'>{wh["Avg_Days_Left"]:.1f}d</b></div>
              </div>
            </div>""", unsafe_allow_html=True)
        sp()
        sec("Detailed Shipment Routing Plan")
        routing_tbl = sku_plan[[
            "Target_Warehouse", "SKU_ID", "Product_Name", "Category", "ABC", "Urgency",
            "Prod_Need", "Days_Left", "Est_Ship_Cost", "WH_Share_Pct",
        ]].copy()
        routing_tbl["Days_Left"]     = routing_tbl["Days_Left"].apply(lambda x: f"{int(x)}d" if x < 999 else "∞")
        routing_tbl["Est_Ship_Cost"] = routing_tbl["Est_Ship_Cost"].apply(lambda x: f"₹{int(x):,}")
        routing_tbl["WH_Share_Pct"]  = routing_tbl["WH_Share_Pct"].apply(lambda x: f"{x:.0f}%")
        routing_tbl.columns = ["Warehouse", "SKU", "Product", "Category", "ABC", "Urgency",
                               "Units", "Days Left", "Ship Cost", "% of WH Inbound"]
        st.dataframe(routing_tbl.sort_values(["Warehouse", "Urgency"]),
                     use_container_width=True, hide_index=True, height=380)
    with pt3:
        sec("Production Urgency Distribution")
        va1, va2 = st.columns(2, gap="large")
        urg_color_map = {
            "🔴 Urgent": "#ef4444", "🟠 High": "#f97316",
            "🟡 Medium": "#eab308", "🟢 Normal": "#22c55e",
        }
        with va1:
            urg_counts = sku_plan["Urgency"].value_counts().reset_index()
            urg_counts.columns = ["Urgency", "Count"]
            fig_d = go.Figure(go.Pie(
                labels=urg_counts["Urgency"], values=urg_counts["Count"], hole=0.55,
                marker=dict(colors=[urg_color_map.get(u, "#888") for u in urg_counts["Urgency"]],
                            line=dict(color="#ffffff", width=2)),
                textinfo="label+value", textfont=dict(size=11),
            ))
            fig_d.update_layout(**CD(), height=260, showlegend=False,
                                title=dict(text="SKUs by Urgency Tier", font=dict(size=11, color="#64748b")))
            st.plotly_chart(fig_d, use_container_width=True, key="pq_donut")
        with va2:
            cat_units = sku_plan.groupby(["Category", "Urgency"])["Prod_Need"].sum().reset_index()
            fig_bu = go.Figure()
            for urg, clr in urg_color_map.items():
                sub = cat_units[cat_units["Urgency"] == urg]
                if sub.empty:
                    continue
                fig_bu.add_trace(go.Bar(
                    name=urg, x=sub["Category"], y=sub["Prod_Need"],
                    marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)")),
                    text=sub["Prod_Need"].astype(int),
                    textposition="inside", textfont=dict(color="white", size=9),
                ))
            fig_bu.update_layout(**CD(), height=260, barmode="stack",
                                 xaxis={**gX(), "tickangle": -10},
                                 yaxis={**gY(), "title": "Units to Produce"},
                                 legend={**leg(), "orientation": "h", "y": -0.32},
                                 title=dict(text="Units Needed by Category & Urgency", font=dict(size=11, color="#64748b")))
            st.plotly_chart(fig_bu, use_container_width=True, key="pq_cat_bar")
        sp()
        sec("Days of Stock Remaining — Most Critical SKUs")
        top20 = sku_plan.sort_values("Days_Left", ascending=True).head(20).copy()
        top20["Label"]     = top20["Product_Name"].str[:22] + " [" + top20["SKU_ID"] + "]"
        top20["Bar_Color"] = top20["Days_Left"].apply(
            lambda x: "#ef4444" if x <= 7 else "#f97316" if x <= 14 else "#eab308" if x <= 30 else "#22c55e"
        )
        top20_s = top20.sort_values("Days_Left", ascending=True)
        fig_hl = go.Figure(go.Bar(
            x=top20_s["Days_Left"].clip(upper=60), y=top20_s["Label"],
            orientation="h",
            marker=dict(color=top20_s["Bar_Color"].tolist(), line=dict(color="rgba(0,0,0,0)")),
            text=[f"{int(v)}d · {int(u):,} units" for v, u in zip(top20_s["Days_Left"], top20_s["Prod_Need"])],
            textposition="outside", textfont=dict(color="#334155", size=9),
            customdata=top20_s[["Category", "Target_Warehouse", "Urgency"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Days left: %{x:.0f}d<br>"
                "Category: %{customdata[0]}<br>"
                "Warehouse: %{customdata[1]}<br>"
                "Urgency: %{customdata[2]}<extra></extra>"
            ),
        ))
        for xv, clr, lbl in [(7, "#ef4444", " 7d"), (14, "#f97316", " 14d"), (30, "#eab308", " 30d")]:
            fig_hl.add_vline(x=xv, line_dash="dash", line_color=clr, line_width=1.5,
                             annotation_text=lbl, annotation_font=dict(color=clr, size=9))
        _x_max = max(float(top20_s["Days_Left"].clip(upper=60).max()) * 1.35, 75)
        fig_hl.update_layout(**CD(), height=max(300, len(top20_s) * 22),
                             xaxis={**gX(), "title": "Days of Stock Remaining", "range": [0, _x_max]},
                             yaxis=dict(showgrid=False, color="#64748b", automargin=True),
                             title=dict(text="Top 20 Most Urgent SKUs — Days of Stock Left",
                                        font=dict(size=11, color="#64748b")))
        st.plotly_chart(fig_hl, use_container_width=True, key="pq_days_bar")

def page_logistics() -> None:
    n_future = get_horizon()
    df     = load_data()
    ops    = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    del_df = get_delivered(df)
    st.markdown("<div class='page-title'>Logistics Optimization</div>", unsafe_allow_html=True)
    horizon_badge(n_future)
    total_spend  = del_df["Shipping_Cost_INR"].sum()
    avg_days     = del_df["Delivery_Days"].mean()
    on_time_pct  = (del_df["Delivery_Days"] <= 3).mean() * 100
    avg_cost_ord = del_df["Shipping_Cost_INR"].mean()
    ret_rate     = df["Return_Flag"].mean() * 100
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Total Shipping Spend", f"₹{total_spend:,.0f}",  "sky",   "all delivered orders")
    kpi(k2, "Avg Delivery Days",    f"{avg_days:.1f}d",       "mint",  "across all carriers")
    kpi(k3, "On-Time Rate",         f"{on_time_pct:.1f}%",   "mint",  "delivered ≤ 3 days")
    kpi(k4, "Avg Cost / Order",     f"₹{avg_cost_ord:.0f}",  "sky",   "per shipment")
    kpi(k5, "Return Rate",          f"{ret_rate:.1f}%",       "coral", "of all orders")
    sp(0.5)
    with st.expander("Carrier Scoring Weights", expanded=False):
        wc1, wc2, wc3 = st.columns(3)
        w_speed   = wc1.slider("Speed weight %",   10, 70, int(DEFAULT_W_SPEED   * 100)) / 100
        w_cost    = wc2.slider("Cost weight %",    10, 70, int(DEFAULT_W_COST    * 100)) / 100
        w_returns = wc3.slider("Returns weight %", 10, 70, int(DEFAULT_W_RETURNS * 100)) / 100
        tot = w_speed + w_cost + w_returns
        w_speed /= tot; w_cost /= tot; w_returns /= tot
    carr, opt, fwd_plan = compute_logistics(w_speed, w_cost, w_returns, n_future)
    plan = compute_production(n_future=n_future)
    t1, t2, t3 = st.tabs(["Carrier Performance", "Cost & Delay", "Forward Plan"])
    with t1:
        sec("Speed vs Cost — Carrier Scorecard")
        fig = go.Figure()
        for i, (_, r) in enumerate(carr.iterrows()):
            fig.add_trace(go.Scatter(
                x=[r["Avg_Days"]], y=[r["Avg_Cost"]], mode="markers+text",
                marker=dict(size=max(r["Orders"] / 50, 14), color=COLORS[i],
                            opacity=0.88, line=dict(color="#FFFFFF", width=2)),
                text=[r["Courier_Partner"]], textposition="top center",
                name=r["Courier_Partner"],
                hovertemplate=(
                    f"<b>{r['Courier_Partner']}</b><br>Orders: {r['Orders']}<br>"
                    f"Avg Days: {r['Avg_Days']:.1f}<br>Avg Cost: ₹{r['Avg_Cost']:.0f}<br>"
                    f"Score: {r['Perf_Score']:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(**CD(), height=270, showlegend=False,
                          xaxis={**gX(), "title": "Avg Delivery Days  (lower = faster)", "rangemode": "normal"},
                          yaxis={**gY(), "title": "Avg Shipping Cost INR  (lower = cheaper)", "rangemode": "normal"})
        st.plotly_chart(fig, use_container_width=True, key="log_bubble")
        ta1, ta2 = st.columns(2, gap="large")
        with ta1:
            sec("Carrier Metrics Table")
            d2 = carr[["Courier_Partner", "Orders", "Avg_Days", "Avg_Cost", "Return_Rate", "Perf_Score"]].copy()
            d2["Avg_Days"]    = d2["Avg_Days"].round(1)
            d2["Avg_Cost"]    = d2["Avg_Cost"].round(1)
            d2["Return_Rate"] = (d2["Return_Rate"] * 100).round(1).astype(str) + "%"
            d2["Perf_Score"]  = d2["Perf_Score"].round(3)
            d2.columns = ["Carrier", "Orders", "Avg Days", "Avg Cost ₹", "Return Rate", "Score"]
            st.dataframe(d2.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
        with ta2:
            sec("Best Carrier per Category")
            if not plan.empty:
                cat_carr = del_df.groupby(["Category", "Courier_Partner"]).agg(
                    Avg_Days=("Delivery_Days", "mean"),
                    Avg_Cost=("Shipping_Cost_INR", "mean"),
                ).reset_index()
                cat_carr_ret = df.groupby(["Category", "Courier_Partner"])["Return_Flag"].mean().reset_index()
                cat_carr_ret.columns = ["Category", "Courier_Partner", "Return_Rate"]
                cat_carr = cat_carr.merge(cat_carr_ret, on=["Category", "Courier_Partner"], how="left")
                cat_carr["Return_Rate"] = cat_carr["Return_Rate"].fillna(0)
                for col_c in ["Avg_Days", "Avg_Cost", "Return_Rate"]:
                    mn_c = cat_carr[col_c].min(); mx_c = cat_carr[col_c].max()
                    cat_carr[f"N_{col_c}"] = 1 - (cat_carr[col_c] - mn_c) / (mx_c - mn_c + 1e-9)
                cat_carr["Score"] = (
                    w_speed * cat_carr["N_Avg_Days"]
                    + w_cost * cat_carr["N_Avg_Cost"]
                    + w_returns * cat_carr["N_Return_Rate"]
                )
                best_cat = cat_carr.sort_values("Score", ascending=False).groupby("Category").first().reset_index()
                prod_by_cat = plan.groupby("Category")["Production"].sum().reset_index()
                best_cat = best_cat.merge(prod_by_cat.rename(columns={"Production": "Planned Units"}), on="Category", how="left")
                best_cat["Avg_Days"]      = best_cat["Avg_Days"].round(1)
                best_cat["Avg_Cost"]      = best_cat["Avg_Cost"].round(1)
                best_cat["Score"]         = best_cat["Score"].round(3)
                best_cat["Planned Units"] = best_cat["Planned Units"].fillna(0).astype(int)
                best_cat = best_cat[["Category", "Courier_Partner", "Avg_Days", "Avg_Cost", "Score", "Planned Units"]]
                best_cat.columns = ["Category", "Best Carrier", "Avg Days", "Avg Cost ₹", "Score", "Planned Units"]
                st.dataframe(best_cat.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("Production plan not available.")
        sp(0.5)
        sec("Carrier x Region Heatmap")

        hc1, hc2, hc3 = st.columns([2, 2, 2])
        delay_thr   = hc1.slider("Delay threshold (days)", 3, 10, DEFAULT_LEAD_TIME, key="log_thr")
        heat_metric = hc2.selectbox("Metric", ["Delay Rate %", "Avg Delivery Days", "Avg Shipping Cost"], key="heat_metric")
        show_annot  = hc3.toggle("Show cell values", value=True, key="heat_annot")

        del_df_delayed = del_df.copy()
        del_df_delayed["Delayed"] = del_df_delayed["Delivery_Days"] > delay_thr

        if heat_metric == "Delay Rate %":
            pv = (del_df_delayed.groupby(["Courier_Partner", "Region"])["Delayed"]
                  .mean().unstack(fill_value=0) * 100)
            def fmt(v): return f"{v:.1f}%"
            colorscale = [
                [0.00, "#166534"], [0.25, "#16a34a"], [0.50, "#eab308"],
                [0.75, "#f97316"], [1.00, "#7f1d1d"],
            ]
            zmid = 25.0
        elif heat_metric == "Avg Delivery Days":
            pv = (del_df_delayed.groupby(["Courier_Partner", "Region"])["Delivery_Days"]
                  .mean().unstack(fill_value=0))
            def fmt(v): return f"{v:.1f}d"
            colorscale = [
                [0.00, "#166534"], [0.25, "#16a34a"], [0.50, "#eab308"],
                [0.75, "#f97316"], [1.00, "#7f1d1d"],
            ]
            zmid = float(pv.values[pv.values > 0].mean()) if pv.values.any() else 3.0
        else:
            pv = (del_df_delayed.groupby(["Courier_Partner", "Region"])["Shipping_Cost_INR"]
                  .mean().unstack(fill_value=0))
            def fmt(v): return f"Rs{v:.0f}"
            colorscale = [
                [0.00, "#166534"], [0.25, "#16a34a"], [0.50, "#eab308"],
                [0.75, "#f97316"], [1.00, "#7f1d1d"],
            ]
            zmid = float(pv.values[pv.values > 0].mean()) if pv.values.any() else 100.0

        carriers = list(pv.index)
        regions  = list(pv.columns)
        z_vals   = pv.values.copy()

        # cell annotation text (plain strings, no HTML tags)
        if show_annot:
            cell_text = [[fmt(z_vals[r][c]) for c in range(len(regions))]
                         for r in range(len(carriers))]
        else:
            cell_text = None

        # order count customdata (2-D int array, same shape as z_vals)
        order_pv = (del_df_delayed.groupby(["Courier_Partner", "Region"])["Order_ID"]
                    .count().unstack(fill_value=0)
                    .reindex(index=pv.index, columns=pv.columns, fill_value=0))
        customdata = order_pv.values.astype(int)

        fig_h = go.Figure(go.Heatmap(
            z=z_vals,
            x=regions,
            y=carriers,
            colorscale=colorscale,
            text=cell_text,
            texttemplate="%{text}" if show_annot else "",
            textfont=dict(size=10, color="white"),
            customdata=customdata,
            hovertemplate="<b>%{y} to %{x}</b><br>Value: %{z:.1f}<br>Orders: %{customdata}<extra></extra>",
            colorbar=dict(
                tickfont=dict(color="#64748b", size=9),
                thickness=12,
                len=0.85,
            ),
            xgap=3,
            ygap=3,
        ))

        # outline worst cell per row in red, best in green
        worst_cols = np.argmax(z_vals, axis=1)
        best_cols  = np.argmin(z_vals, axis=1)
        for r, c in enumerate(worst_cols):
            fig_h.add_shape(type="rect",
                x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                line=dict(color="#ef4444", width=2.5), fillcolor="rgba(0,0,0,0)")
        for r, c in enumerate(best_cols):
            fig_h.add_shape(type="rect",
                x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                line=dict(color="#22c55e", width=2.5), fillcolor="rgba(0,0,0,0)")

        # row-average annotations on the right
        for r in range(len(carriers)):
            fig_h.add_annotation(
                x=len(regions) - 0.5 + 0.7, y=r,
                text=f"avg {fmt(float(np.mean(z_vals[r])))}",
                showarrow=False, xanchor="left",
                font=dict(size=9, color="#64748b"),
            )

        cell_h = max(55, 300 // max(len(carriers), 1))
        layout = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter,sans-serif", size=11),
            height=cell_h * len(carriers) + 80,
            margin=dict(l=30, r=120, t=44, b=40),
            title=dict(
                text=f"{heat_metric} by Carrier x Region  |  threshold >{delay_thr}d",
                font=dict(size=11, color="#64748b"),
            ),
            xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=11, color="#334155")),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#334155")),
        )
        fig_h.update_layout(**layout)

        st.plotly_chart(fig_h, use_container_width=True, key="log_heat")

        # summary banner
        flat      = z_vals.flatten()
        worst_idx = np.unravel_index(int(np.argmax(flat)), z_vals.shape)
        best_idx  = np.unravel_index(int(np.argmin(flat)), z_vals.shape)
        banner(
            f"Worst lane: <b>{carriers[worst_idx[0]]} to {regions[worst_idx[1]]}</b>"
            f" ({fmt(float(flat[np.argmax(flat)]))})"
            f" &nbsp;|&nbsp; "
            f"Best lane: <b>{carriers[best_idx[0]]} to {regions[best_idx[1]]}</b>"
            f" ({fmt(float(flat[np.argmin(flat)]))})"
            f" &nbsp;|&nbsp; "
            f"Overall avg: <b>{fmt(float(np.mean(flat)))}</b>",
            "sky",
        )
    with t2:
        total_sav = opt["Potential_Saving"].sum()
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Current Spend",    f"₹{total_spend:,.0f}",                   "sky",  "all deliveries")
        kpi(c2, "Optimised Spend",  f"₹{total_spend - total_sav:,.0f}",       "mint", "with best carriers")
        kpi(c3, "Potential Saving", f"₹{total_sav:,.0f}",                     "mint", "by switching carrier")
        kpi(c4, "Saving %",         f"{total_sav/total_spend*100:.1f}%",      "mint", "of total spend")
        sp(0.5)
        del_df_t2 = del_df.copy()
        del_df_t2["Delayed"] = del_df_t2["Delivery_Days"] > delay_thr
        tb1, tb2 = st.columns(2, gap="large")
        with tb1:
            sec("Region Cost — Current vs Optimal")
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Bar(
                name="Current ₹/order", x=opt["Region"], y=opt["Current_Avg_Cost"],
                marker=dict(color="#EF4444", line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:.0f}" for v in opt["Current_Avg_Cost"]],
                textposition="outside", textfont=dict(color="#334155"),
            ))
            fig_cost.add_trace(go.Bar(
                name="Optimal ₹/order", x=opt["Region"], y=opt["Min_Avg_Cost"],
                marker=dict(color="#22C55E", line=dict(color="rgba(0,0,0,0)")),
                text=[f"₹{v:.0f}" for v in opt["Min_Avg_Cost"]],
                textposition="outside", textfont=dict(color="#334155"),
            ))
            fig_cost.update_layout(
                **CD(), height=270, barmode="group",
                xaxis={**gX(), "tickangle": -20, "automargin": True},
                yaxis={**gY(), "title": "Avg Cost per Order ₹"},
                legend={**leg(), "orientation": "h", "y": -0.3},
            )
            st.plotly_chart(fig_cost, use_container_width=True, key="log_cost")
        with tb2:
            sec("Delay Rate by Region")
            rd = del_df_t2.groupby("Region").agg(T=("Order_ID", "count"), D=("Delayed", "sum")).reset_index()
            rd["Rate"] = (rd["D"] / rd["T"] * 100).round(1)
            rd_s = rd.sort_values("Rate", ascending=True)
            fig_r = go.Figure(go.Bar(
                x=rd_s["Rate"], y=rd_s["Region"], orientation="h",
                marker=dict(
                    color=[f"rgba(239,68,68,{min(v/50+0.2,0.9):.2f})" for v in rd_s["Rate"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v}%" for v in rd_s["Rate"]], textposition="outside",
                textfont=dict(color="#334155"),
            ))
            fig_r.update_layout(**CD(), height=270,
                xaxis={**gX(), "title": "Delay Rate %"}, yaxis=dict(showgrid=False, color="#64748b"))
            st.plotly_chart(fig_r, use_container_width=True, key="log_delay_region")
        sp(0.5)
        tb3, tb4 = st.columns(2, gap="large")
        with tb3:
            sec("Carrier Switch Recommendations")
            od = opt[["Region", "Optimal_Carrier", "Current_Avg_Cost", "Min_Avg_Cost", "Potential_Saving", "Saving_Pct", "Orders"]].copy()
            od["Current_Avg_Cost"] = od["Current_Avg_Cost"].round(1)
            od["Min_Avg_Cost"]     = od["Min_Avg_Cost"].round(1)
            od["Potential_Saving"] = od["Potential_Saving"].astype(int)
            od.columns = ["Region", "Switch To", "Current Avg ₹", "Optimal Avg ₹", "Saving ₹", "Saving %", "Orders"]
            st.dataframe(od.sort_values("Saving ₹", ascending=False), use_container_width=True, hide_index=True)
        with tb4:
            sec("Delay Rate by Carrier")
            cd = del_df_t2.groupby("Courier_Partner").agg(T=("Order_ID", "count"), D=("Delayed", "sum")).reset_index()
            cd["Rate"] = (cd["D"] / cd["T"] * 100).round(1)
            fig_cd = go.Figure(go.Bar(
                x=cd["Courier_Partner"], y=cd["Rate"],
                marker=dict(
                    color=["#EF4444" if v > 35 else "#F59E0B" if v > 20 else "#22C55E" for v in cd["Rate"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v}%" for v in cd["Rate"]], textposition="outside",
                textfont=dict(color="#334155"),
            ))
            fig_cd.update_layout(**CD(), height=240, xaxis=gX(), yaxis={**gY(), "title": "Delay Rate %"})
            st.plotly_chart(fig_cd, use_container_width=True, key="log_delay_carrier")
    with t3:
        if fwd_plan.empty:
            st.info("No forward plan available — production plan has no actionable SKUs.")
        else:
            fwd_agg = (
                fwd_plan.groupby("Month_dt")
                .agg(
                    Month           = ("Month",         "first"),
                    Total_Units     = ("Prod_Units",     "sum"),
                    Total_Orders    = ("Proj_Orders",    "sum"),
                    Total_Ship_Cost = ("Proj_Ship_Cost", "sum"),
                    CI_Lo           = ("CI_Lo_Units",    "sum"),
                    CI_Hi           = ("CI_Hi_Units",    "sum"),
                ).reset_index().sort_values("Month_dt")
            )
            fc1, fc2, fc3 = st.columns(3)
            kpi(fc1, f"{n_future}M Planned Units", f"{fwd_agg['Total_Units'].sum():,}",         "sky",   "from production plan")
            kpi(fc2, f"{n_future}M Est. Orders",   f"{fwd_agg['Total_Orders'].sum():,}",        "sky",   "projected shipments")
            kpi(fc3, f"{n_future}M Ship Cost",     f"₹{fwd_agg['Total_Ship_Cost'].sum():,.0f}", "amber", "at current avg rate")
            sp(0.5)
            tc1, tc2 = st.columns([3, 2], gap="large")
            with tc1:
                sec(f"Production → Shipment Plan — {n_future} Months")
                fig_fwd = go.Figure()
                x_ci = list(fwd_agg["Month_dt"]) + list(fwd_agg["Month_dt"])[::-1]
                y_ci = list(fwd_agg["CI_Hi"])     + list(fwd_agg["CI_Lo"])[::-1]
                fig_fwd.add_trace(go.Scatter(
                    x=x_ci, y=y_ci, fill="toself",
                    fillcolor="rgba(59,130,246,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="Demand 90% CI", hoverinfo="skip",
                ))
                fig_fwd.add_trace(go.Bar(
                    x=fwd_agg["Month_dt"], y=fwd_agg["Total_Units"],
                    name="Planned Units",
                    marker=dict(color="#3B82F6", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
                    text=[f"{int(v):,}" for v in fwd_agg["Total_Units"]],
                    textposition="inside", textfont=dict(color="white", size=9),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Units: %{y:,}<extra></extra>",
                ))
                fig_fwd.update_layout(
                    **CD(), height=260, barmode="overlay", xaxis=gX(),
                    yaxis={**gY(), "title": "Units"},
                    legend={**leg(), "orientation": "h", "y": -0.28},
                )
                st.plotly_chart(fig_fwd, use_container_width=True, key="fwd_units")
            with tc2:
                sec("Category Breakdown")
                cat_fwd = (
                    fwd_plan.groupby("Category")
                    .agg(Units=("Prod_Units", "sum"), Orders=("Proj_Orders", "sum"), Cost=("Proj_Ship_Cost", "sum"))
                    .reset_index().sort_values("Units", ascending=False)
                )
                cat_fwd.columns = ["Category", "Units", "Est. Orders", "Ship Cost ₹"]
                st.dataframe(cat_fwd, use_container_width=True, hide_index=True, height=210)
            sec("Projected Shipping Cost")
            fig_cost2 = go.Figure(go.Scatter(
                x=fwd_agg["Month_dt"], y=fwd_agg["Total_Ship_Cost"],
                mode="lines+markers", line=dict(color="#8B5CF6", width=2.5),
                marker=dict(size=8, color="#8B5CF6", line=dict(color="#FFFFFF", width=2)),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.07)",
                hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
            ))
            fig_cost2.update_layout(
                **CD(), height=200,
                xaxis={**gX(), "tickangle": 0},
                yaxis={**gY(), "title": "₹"},
            )
            st.plotly_chart(fig_cost2, use_container_width=True, key="fwd_cost")
            sp(0.5)
            sec("Inbound Plan per Warehouse")
            wh_share = (del_df.groupby("Warehouse")["Quantity"].sum() / del_df["Quantity"].sum()).to_dict()
            inb_rows = [
                {"Month": row["Month"], "Month_dt": row["Month_dt"],
                 "Warehouse": wh, "Inbound_Units": round(row["Prod_Units"] * sh),
                 "Proj_Ship_Cost": round(row["Proj_Ship_Cost"] * sh)}
                for _, row in fwd_plan.iterrows() for wh, sh in wh_share.items()
            ]
            inb_agg = (
                pd.DataFrame(inb_rows)
                .groupby(["Month_dt", "Month", "Warehouse"])
                .agg(Inbound_Units=("Inbound_Units", "sum"), Proj_Ship_Cost=("Proj_Ship_Cost", "sum"))
                .reset_index().sort_values(["Month_dt", "Warehouse"])
            )
            fig_inb = go.Figure()
            for i, wh in enumerate(sorted(inb_agg["Warehouse"].unique())):
                wdf = inb_agg[inb_agg["Warehouse"] == wh]
                fig_inb.add_trace(go.Bar(
                    x=wdf["Month"], y=wdf["Inbound_Units"], name=wh,
                    marker=dict(color=COLORS[i % len(COLORS)], line=dict(color="rgba(0,0,0,0)")),
                    text=[f"{int(v):,}" for v in wdf["Inbound_Units"]],
                    textposition="outside", textfont=dict(color="#334155", size=8),
                ))
            fig_inb.update_layout(**CD(), height=250, barmode="group",
                                  xaxis={**gX(), "tickangle": -25},
                                  yaxis={**gY(), "title": "Planned Inbound Units"}, legend=leg())
            st.plotly_chart(fig_inb, use_container_width=True, key="wh_inbound")
            disp_inb = inb_agg[["Month", "Warehouse", "Inbound_Units", "Proj_Ship_Cost"]].copy()
            disp_inb.columns = ["Month", "Warehouse", "Planned Units", "Proj. Ship Cost ₹"]
            st.dataframe(disp_inb, use_container_width=True, hide_index=True)

def main() -> None:
    inject_css()
    st.sidebar.markdown("""<div style='padding:16px 0 10px'>
      <div style='font-size:28px;font-weight:900;letter-spacing:-.03em;text-transform:uppercase;
           background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>OmniFlow D2D</div>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:10px;font-weight:700;color:#4a5e7a;letter-spacing:.1em;"
        "text-transform:uppercase;font-family:DM Mono;margin-bottom:4px'>📅 Forecast Horizon</div>",
        unsafe_allow_html=True,
    )
    horizon_val = st.sidebar.select_slider(
        "Forecast horizon (months)",
        options=[3, 6, 9, 12],
        value=st.session_state.get("global_horizon", 6),
        key="global_horizon",
        label_visibility="collapsed",
        help="Sets the forecast window for ALL modules: Demand, Inventory, Production & Logistics",
    )
    st.sidebar.markdown(
        f"<div style='font-size:10px;color:#64748b;font-family:DM Mono;margin-top:2px;"
        f"margin-bottom:12px'>→ All modules use <b style='color:#1e3a8a'>{horizon_val}M</b> window</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='height:1px;background:rgba(100,116,139,0.15);margin-bottom:10px'></div>",
        unsafe_allow_html=True,
    )
    PAGES = {
        "Overview":               page_overview,
        "Demand Forecasting":     page_demand,
        "Inventory Optimization": page_inventory,
        "Production Planning":    page_production,
        "Logistics Optimization": page_logistics,
    }
    sel = st.sidebar.radio("Navigation", list(PAGES.keys()))
    PAGES[sel]()

if __name__ == "__main__":
    main()
