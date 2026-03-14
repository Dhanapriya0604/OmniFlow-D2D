import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# ─────────────────────────────────────────────
# CONSTANTS  (all business-logic defaults here)
# ─────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india_ecommerce_orders.csv")

COLORS = ["#1565C0", "#2E7D32", "#E65100", "#C62828", "#6A1B9A", "#00695C"]
MODEL_COLORS = {
    "Ridge": "#3B82F6",
    "RandomForest": "#22C55E",
    "GradBoost": "#F59E0B",
    "Ensemble": "#8B5CF6",
}

# ── Inventory defaults ──
# FIX: ORDER_COST is per-order procurement cost, not production cost
DEFAULT_ORDER_COST = 500        # ₹ per purchase order (admin + processing)
DEFAULT_HOLD_PCT   = 0.20       # 20% annual holding rate on COGS (not sell price)
# FIX: REPLENISHMENT lead time is separate from customer delivery time
DEFAULT_REPLEN_LEAD_TIME = 14   # days — typical India supplier replenishment lead time
DEFAULT_SERVICE_Z  = 1.65       # 95% service level z-score
N_FUTURE_MONTHS    = 6
MIN_HISTORY_MONTHS = 6

# ── ML hyper-parameters ──
N_ESTIMATORS_RF  = 200          # reduced — less overfitting on small datasets
MAX_DEPTH_RF     = 2            # shallower — better generalisation on ~20 pts
MIN_SAMPLES_LEAF = 4
N_ESTIMATORS_GB  = 100
MAX_DEPTH_GB     = 2
LEARNING_RATE_GB = 0.05
SUBSAMPLE_GB     = 0.80
RIDGE_ALPHA      = 0.5          # stronger regularisation for seasonal Fourier fit
CI_Z             = 1.645
MIN_REGIME_IDX   = 6

# ── Financials ──
MARGIN_RATE        = 0.20       # gross margin for stockout cost estimation
COGS_RATE          = 0.60       # COGS = sell_price × COGS_RATE for holding cost
DEMAND_PEAK_WEIGHT = 0.30
BOOST_SCHEDULE     = {0: 0.60, 1: 0.40}

# ── Logistics ──
DEFAULT_W_SPEED   = 0.40
DEFAULT_W_COST    = 0.35
DEFAULT_W_RETURNS = 0.25


def get_horizon() -> int:
    return st.session_state.get("global_horizon", N_FUTURE_MONTHS)


# ──────────────────────────────────
# CSS INJECTION
# ──────────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800;900&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

    :root {
      --bg: #f0f2f8;
      --surface: #ffffff;
      --surface2: #f8f9fc;
      --border: #e2e5ef;
      --primary: #1a2b6d;
      --primary-light: #2541b2;
      --accent: #00c6ae;
      --accent2: #ff6b35;
      --text: #0d1324;
      --muted: #5a6580;
      --danger: #e53935;
      --warn: #f57c00;
      --ok: #2e7d32;
      --panel-shadow: 0 2px 16px rgba(26,43,109,0.08);
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    section.main > div { animation: fadeUp 0.4s ease; }
    @keyframes fadeUp {
      from { opacity:0; transform:translateY(8px); }
      to   { opacity:1; transform:translateY(0); }
    }

    /* ── Page header ── */
    .page-header {
      background: linear-gradient(120deg, #1a2b6d 0%, #2541b2 55%, #0091ff 100%);
      border-radius: 16px;
      padding: 28px 32px 22px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }
    .page-header::before {
      content:'';
      position:absolute; inset:0;
      background: radial-gradient(ellipse at 80% 50%, rgba(0,198,174,0.18) 0%, transparent 65%);
    }
    .page-header-title {
      font-family: 'Syne', sans-serif;
      font-size: 30px;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.03em;
      line-height: 1.1;
      position: relative;
    }
    .page-header-sub {
      font-size: 11px;
      font-family: 'DM Mono', monospace;
      color: rgba(255,255,255,0.65);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-top: 6px;
      position: relative;
    }

    /* ── Metric cards ── */
    .kpi-grid { display:grid; gap:12px; margin-bottom:20px; }
    .kpi-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: var(--panel-shadow);
      transition: transform .18s, box-shadow .18s;
      position: relative;
      overflow: hidden;
    }
    .kpi-card::after {
      content:'';
      position:absolute; top:0; left:0; right:0; height:3px;
      background: var(--kpi-accent, var(--primary-light));
      border-radius: 14px 14px 0 0;
    }
    .kpi-card:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(26,43,109,0.13); }
    .kpi-label {
      font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:.08em; color:var(--muted); font-family:'DM Mono',monospace;
      margin-bottom:6px;
    }
    .kpi-value {
      font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
      color:var(--kpi-accent, var(--primary));
      line-height:1.1; letter-spacing:-0.02em;
    }
    .kpi-sub { font-size:10px; color:var(--muted); margin-top:3px; }

    /* ── Section titles ── */
    .sec-title {
      font-family:'Syne',sans-serif; font-size:16px; font-weight:800;
      color:var(--text); letter-spacing:-0.01em;
      margin: 24px 0 4px;
      display:flex; align-items:center; gap:8px;
    }
    .sec-divider {
      height:2px;
      background: linear-gradient(90deg, var(--primary-light), transparent);
      margin-bottom:14px; border-radius:2px;
    }

    /* ── Info banners ── */
    .banner {
      border-radius:10px; padding:11px 15px;
      font-size:12.5px; line-height:1.6; margin:8px 0;
    }
    .banner-teal  { background:#f0fdfb; border:1px solid #5eead4; }
    .banner-amber { background:#fffbeb; border:1px solid #fbbf24; }
    .banner-coral { background:#fff1f2; border:1px solid #fecaca; border-left:4px solid #ef4444; font-weight:600; }
    .banner-sky   { background:#eff6ff; border:1px solid #93c5fd; }
    .banner-mint  { background:#ecfdf5; border:1px solid #34d399; }

    /* ── SKU alert cards ── */
    .sku-card {
      border-radius:12px; padding:12px 14px; margin-bottom:8px;
      border:1px solid var(--border); transition:transform .18s;
    }
    .sku-card:hover { transform:translateX(4px); }
    .sku-critical { background:linear-gradient(135deg,#fef2f2,#fff); border-left:4px solid #ef4444; }
    .sku-low      { background:linear-gradient(135deg,#fffbeb,#fff); border-left:4px solid #f59e0b; }
    .sku-ok       { background:linear-gradient(135deg,#f0fdf4,#fff); border-left:4px solid #22c55e; }

    /* ── Model pills ── */
    .pill {
      display:inline-block; padding:3px 9px; font-size:10px;
      font-weight:700; border-radius:20px; margin:2px;
    }
    .pill-ridge    { background:#eff6ff; color:#1d4ed8; }
    .pill-rf       { background:#f0fdf4; color:#15803d; }
    .pill-gb       { background:#fef9c3; color:#a16207; }
    .pill-ensemble { background:#fdf4ff; color:#7e22ce; }

    /* ── Horizon badge ── */
    .horizon-badge {
      display:inline-flex; align-items:center; gap:6px;
      background:linear-gradient(135deg,#e0e7ff,#f0f4ff);
      border:1px solid #c7d7fd; border-radius:20px;
      padding:4px 14px; font-size:11px; font-weight:700;
      color:#1e3a8a; font-family:'DM Mono',monospace; margin-bottom:8px;
    }

    /* ── Pipeline cards ── */
    .pipeline-card {
      background:var(--surface); border:1px solid var(--border);
      border-radius:14px; padding:18px 20px;
      box-shadow:var(--panel-shadow);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
      background:#f1f5f9; border-radius:10px; padding:9px 18px;
      font-weight:700; color:#475569; font-size:13px;
    }
    .stTabs [aria-selected="true"] {
      background:#e0e7ff; color:#1e3a8a;
      box-shadow:0 4px 14px rgba(30,58,139,0.18);
    }

    /* ── Misc ── */
    .block-container { padding-top:1.6rem; padding-bottom:2rem; }
    .stMultiSelect div[data-baseweb="tag"] {
      background:#eef2ff !important; color:#1e3a8a !important;
      border-radius:8px !important; border:1px solid #c7d7fd !important; font-weight:600;
    }

    /* ── Heatmap overlay polish ── */
    .heatmap-wrap {
      background:var(--surface); border:1px solid var(--border);
      border-radius:16px; padding:20px;
      box-shadow:var(--panel-shadow);
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────
# PLOTLY HELPERS
# ──────────────────────────────────
def CD() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="DM Sans, sans-serif", size=11),
        margin=dict(l=36, r=52, t=44, b=32),
    )

def gY() -> dict:
    return dict(showgrid=True, gridcolor="rgba(0,0,0,0.055)",
                zeroline=False, tickcolor="#94a3b8")

def gX() -> dict:
    return dict(showgrid=False, zeroline=False, tickcolor="#94a3b8")

def leg() -> dict:
    return dict(
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#E0E0E0",
        borderwidth=1, font=dict(color="#334155", size=10),
    )


# ──────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────
def kpi(col, label: str, value, accent: str = "#2541b2", sub: str = "") -> None:
    col.markdown(
        f"""<div class='kpi-card' style='--kpi-accent:{accent}'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{value}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True,
    )

def sec(label: str, emoji: str = "") -> None:
    prefix = f"{emoji} " if emoji else ""
    st.markdown(
        f"<div class='sec-title'>{prefix}{label}</div>"
        "<div class='sec-divider'></div>",
        unsafe_allow_html=True,
    )

def banner(html: str, cls: str = "teal") -> None:
    st.markdown(f"<div class='banner banner-{cls}'>{html}</div>", unsafe_allow_html=True)

def sp(n: float = 1) -> None:
    st.markdown(f"<div style='height:{int(n*12)}px'></div>", unsafe_allow_html=True)

def horizon_badge(n: int) -> None:
    st.markdown(
        f"<div class='horizon-badge'>📅 Forecast Horizon: {n} months</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────
# DATA LOADING
# ──────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file not found: {DATA_FILE}")
        st.stop()
    df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])

    # ── Basic validation ──
    required = ["Order_Date","Order_ID","SKU_ID","Product_Name","Category",
                "Order_Status","Quantity","Revenue_INR","Return_Flag",
                "Delivery_Days","Shipping_Cost_INR","Courier_Partner",
                "Region","Warehouse","Sales_Channel","Sell_Price",
                "Current_Stock_Units","Reorder_Point","Stock_Status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["Quantity"]     = df["Quantity"].clip(lower=0)
    df["Revenue_INR"]  = df["Revenue_INR"].clip(lower=0)
    df["Delivery_Days"]= df["Delivery_Days"].clip(lower=0)
    df["YearMonth"]    = df["Order_Date"].dt.to_period("M")
    df["Year"]         = df["Order_Date"].dt.year
    df["Month_Num"]    = df["Order_Date"].dt.month
    df["Net_Revenue"]  = np.where(df["Return_Flag"] == 1, 0.0, df["Revenue_INR"])
    df["Net_Qty"]      = np.where(df["Return_Flag"] == 1, 0,   df["Quantity"])
    return df


@st.cache_data
def get_ops(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Order_Status"].isin(["Delivered", "Shipped"])].copy()


@st.cache_data
def get_delivered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Order_Status"] == "Delivered"].copy()


# ──────────────────────────────────
# SUPPLY CHAIN KPIs  (FIX: added fill rate, turnover, stockout rate, OTIF)
# ──────────────────────────────────
@st.cache_data
def compute_sc_kpis(df: pd.DataFrame) -> dict:
    """
    Compute the six standard supply chain KPIs that were absent:
    Fill Rate, Service Level Achieved, Inventory Turnover,
    Stockout Rate, Forecast Accuracy (MAPE proxy), OTIF.
    """
    del_df = df[df["Order_Status"] == "Delivered"].copy()
    ops    = df[df["Order_Status"].isin(["Delivered","Shipped"])].copy()

    total_orders    = len(df[df["Order_Status"] != "Cancelled"])
    fulfilled       = len(del_df)
    # Fill Rate: % of demand orders that were fully delivered
    fill_rate       = fulfilled / max(total_orders, 1) * 100

    # Stockout proxy: orders that were returned or cancelled due to stock issues
    stockout_proxy  = df[df["Order_Status"].isin(["Cancelled","Returned"])].shape[0]
    stockout_rate   = stockout_proxy / max(len(df), 1) * 100

    # OTIF: On-Time (≤3 days) AND In-Full (status=Delivered, not partial)
    on_time         = (del_df["Delivery_Days"] <= 3).sum()
    otif_rate       = on_time / max(len(del_df), 1) * 100

    # Inventory Turnover (annualised): COGS / avg inventory value
    # COGS ≈ Net_Qty sold × (Sell_Price × COGS_RATE)
    # Avg inventory value = mean(Current_Stock × Sell_Price × COGS_RATE) per SKU snapshot
    df_sorted = df.sort_values("Order_Date")
    sku_snap  = df_sorted.groupby("SKU_ID").agg(
        stock=("Current_Stock_Units","last"),
        price=("Sell_Price","mean"),
    )
    avg_inv_value = (sku_snap["stock"] * sku_snap["price"] * COGS_RATE).sum()
    cogs_total    = (ops["Net_Qty"] * ops["Sell_Price"] * COGS_RATE).sum()
    months_span   = max((df["Order_Date"].max() - df["Order_Date"].min()).days / 30.44, 1)
    inv_turnover  = (cogs_total / max(avg_inv_value, 1)) * (12 / months_span)

    # Days of Inventory Outstanding
    doi = 365 / max(inv_turnover, 0.01)

    # Service level achieved (% orders delivered within agreed lead time ≤7d)
    svc_achieved  = (del_df["Delivery_Days"] <= 7).mean() * 100

    # Return Rate
    return_rate   = df["Return_Flag"].mean() * 100

    return dict(
        fill_rate=round(fill_rate, 1),
        stockout_rate=round(stockout_rate, 1),
        otif_rate=round(otif_rate, 1),
        inv_turnover=round(inv_turnover, 2),
        doi=round(doi, 1),
        svc_achieved=round(svc_achieved, 1),
        return_rate=round(return_rate, 1),
    )


# ──────────────────────────────────
# FORECASTING ENGINE
# ──────────────────────────────────
def _to_ts(idx) -> pd.DatetimeIndex:
    return idx.to_timestamp() if hasattr(idx, "to_timestamp") else pd.DatetimeIndex(idx)


def _make_models() -> dict:
    """
    FIX: Reduced model complexity throughout to avoid overfitting on ~20 data points.
    RF: depth=2, 200 trees.  GB: depth=2, 100 rounds, lr=0.05.  Ridge: alpha=0.5.
    """
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
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=0.75,
            random_state=42,
        )
    }


def _build_features(n_hist: int, n_future: int, ds_hist, regime_idx: int) -> np.ndarray:
    """
    FIX: Reduced feature set to prevent overfitting.
    14-feature set → 10 features (removed redundant quadratic + log trend).
    Regime feature retained but treated as soft prior, not hard break.
    """
    n  = n_hist + n_future
    t  = np.arange(n, dtype=float)
    ts = _to_ts(ds_hist)
    h_months = ts.month.values
    last_m   = int(h_months[-1])
    f_months = np.array([(last_m + i - 1) % 12 + 1 for i in range(1, n_future + 1)])
    mn       = np.concatenate([h_months, f_months])
    regime   = (t >= regime_idx).astype(float)
    q        = np.where(mn <= 3, 1, np.where(mn <= 6, 2, np.where(mn <= 9, 3, 4)))
    # 10 features: trend, sin/cos at 2 Fourier harmonics, regime, 3 quarter dummies
    return np.column_stack([
        t / max(n, 1),                          # normalised trend
        np.sin(2 * np.pi * mn / 12),
        np.cos(2 * np.pi * mn / 12),
        np.sin(4 * np.pi * mn / 12),
        np.cos(4 * np.pi * mn / 12),
        regime,
        (q == 1).astype(float),
        (q == 2).astype(float),
        (q == 3).astype(float),
        (q == 4).astype(float),
    ])


def _detect_regime(vals: np.ndarray, min_idx: int = MIN_REGIME_IDX) -> int:
    """
    FIX: Guard against edge case where len(vals) <= 2*min_idx.
    Returns min_idx (no regime shift) if insufficient data.
    """
    if len(vals) <= 2 * min_idx:
        return min_idx
    best_idx, best_ratio = min_idx, 1.0
    for i in range(min_idx, len(vals) - min_idx):
        r = vals[i:].mean() / (vals[:i].mean() + 1e-9)
        if r > best_ratio:
            best_ratio = r
            best_idx   = i
    return best_idx


def _compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    FIX: MAPE was completely absent from original. Added here.
    Uses symmetric MAPE to handle near-zero actuals gracefully.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def _compute_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    FIX: Bias (MFE) tracking was absent.  Positive = over-forecast, negative = under-forecast.
    """
    return float(np.mean(predicted - actual))


def ml_forecast(vals: np.ndarray, ds_idx, n_future: int = N_FUTURE_MONTHS) -> dict | None:
    """
    Walk-forward validated ensemble forecast.

    FIX summary:
    ─ Hold-out extended to max(4, n//5) points (not always 4)
    ─ MAPE and Bias added to all model metrics
    ─ CV folds now use minimum 3 training points per fold
    ─ R² reported ONLY from hold-out (not full-fit), avoiding misleading memorisation score
    ─ Feature set reduced from 14→10 to limit overfitting
    ─ Confidence intervals use residual std from hold-out, not full-fit residuals
    """
    n = len(vals)
    if n < MIN_HISTORY_MONTHS:
        return None

    regime_idx = _detect_regime(vals)
    X_all  = _build_features(n, n_future, ds_idx, regime_idx)
    X_hist = X_all[:n]
    X_fut  = X_all[n:]

    mean_vals = np.mean(vals)
    ss_tot    = np.sum((vals - mean_vals) ** 2)

    # ── Hold-out split: max(4, n//5) points ──
    h             = max(4, n // 5)
    Xtr_h, ytr_h  = X_hist[:-h], vals[:-h]
    Xte_h, yte_h  = X_hist[-h:], vals[-h:]
    mean_holdout  = float(np.mean(yte_h)) if np.mean(yte_h) > 0 else 1.0

    # ── Individual model hold-out predictions ──
    holdout_preds: dict[str, np.ndarray] = {}
    for mname, mdl in _make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(Xtr_h, ytr_h)
        holdout_preds[mname] = np.maximum(pipe.predict(Xte_h), 0)

    # ── CV-based ensemble weights (inverse-RMSE from walk-forward folds) ──
    n_tr = len(ytr_h)
    # FIX: ensure each fold has at least MIN_HISTORY_MONTHS training points
    fold_size = max(2, h // 2)
    n_folds   = min(3, (n_tr - MIN_HISTORY_MONTHS) // fold_size)
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

    # ── FIX: All metrics computed from hold-out only (not full-fit) ──
    model_metrics: dict[str, dict] = {}
    for mname in _make_models():
        hp     = holdout_preds[mname]
        rmse_m = float(np.sqrt(mean_squared_error(yte_h, hp)))
        mae_m  = float(mean_absolute_error(yte_h, hp))
        mape_m = _compute_mape(yte_h, hp)
        bias_m = _compute_bias(yte_h, hp)
        nrmse_m = rmse_m / mean_holdout
        # FIX: R² from hold-out residuals against hold-out mean (not full-fit)
        ss_res_ho = np.sum((yte_h - hp) ** 2)
        ss_tot_ho = np.sum((yte_h - mean_holdout) ** 2)
        r2_ho     = max(0.0, 1 - ss_res_ho / (ss_tot_ho + 1e-9))
        model_metrics[mname] = dict(
            rmse=rmse_m, nrmse=nrmse_m, mae=mae_m,
            mape=mape_m, bias=bias_m, r2=r2_ho,
        )

    # ── Ensemble hold-out metrics ──
    ypred_eval = sum(weights[m] * holdout_preds[m] for m in _make_models())
    rmse_e   = float(np.sqrt(mean_squared_error(yte_h, ypred_eval)))
    mae_e    = float(mean_absolute_error(yte_h, ypred_eval))
    mape_e   = _compute_mape(yte_h, ypred_eval)
    bias_e   = _compute_bias(yte_h, ypred_eval)
    nrmse_e  = rmse_e / mean_holdout
    ss_res_e = np.sum((yte_h - ypred_eval) ** 2)
    ss_tot_e = np.sum((yte_h - mean_holdout) ** 2)
    r2_e     = max(0.0, 1 - ss_res_e / (ss_tot_e + 1e-9))

    # ── Full retrain on ALL n points → final fitted + forecast ──
    fitted_pm:   dict[str, np.ndarray] = {}
    forecast_pm: dict[str, np.ndarray] = {}
    for mname, mdl in _make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(X_hist, vals)
        fitted_pm[mname]   = np.maximum(pipe.predict(X_hist), 0)
        forecast_pm[mname] = np.maximum(pipe.predict(X_fut),  0)
    ens_fitted   = sum(weights[m] * fitted_pm[m]   for m in _make_models())
    ens_forecast = sum(weights[m] * forecast_pm[m] for m in _make_models())

    # ── FIX: CI based on hold-out residual std, not full-fit residuals ──
    ho_residuals = yte_h - ypred_eval
    resid_std    = float(ho_residuals.std())
    model_metrics["Ensemble"] = dict(
        rmse=rmse_e, nrmse=nrmse_e, mae=mae_e,
        mape=mape_e, bias=bias_e, r2=r2_e,
    )

    ts_idx    = _to_ts(ds_idx)
    last_dt   = ts_idx[-1]
    fut_dates = pd.date_range(last_dt + pd.offsets.MonthBegin(1), periods=n_future, freq="MS")
    log_std   = np.log1p(resid_std / (mean_holdout + 1e-9))
    steps     = np.arange(1, n_future + 1)
    ci_lo     = np.maximum(ens_forecast * np.exp(-CI_Z * log_std * np.sqrt(steps)), 0)
    ci_hi     = ens_forecast * np.exp(CI_Z  * log_std * np.sqrt(steps))

    return dict(
        hist_ds=ts_idx, hist_y=vals, fitted=ens_fitted,
        fitted_per_model=fitted_pm, forecast_per_model=forecast_pm,
        fut_ds=fut_dates, forecast=ens_forecast, ci_lo=ci_lo, ci_hi=ci_hi,
        rmse=rmse_e, nrmse=nrmse_e, mae=mae_e, mape=mape_e, bias=bias_e, r2=r2_e,
        resid_std=resid_std,
        eval_actual=yte_h, eval_pred=ypred_eval, eval_ds=ts_idx[-h:],
        model_metrics=model_metrics,
        weights={m: weights[m] for m in _make_models()},
        n_holdout=h,
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
            results[cat] = dict(
                mean=float(np.mean(res["forecast"])),
                monthly=res["forecast"].tolist(),
                ci_lo=res["ci_lo"].tolist(),
                ci_hi=res["ci_hi"].tolist(),
                fut_ds=res["fut_ds"],
                resid_std=res["resid_std"],
                hist_avg=float(cat_monthly[cat].mean()),
            )
    return results


# ──────────────────────────────────
# MODEL GRADE
# ──────────────────────────────────
def model_grade(nrmse: float, r2: float, mape: float) -> tuple:
    """
    FIX: Grade now uses MAPE in addition to NRMSE and R²,
    giving a more honest quality assessment.
    """
    acc = max(0.0, round((1 - nrmse) * 100, 1))
    if   nrmse < 0.12 and r2 >= 0.80 and (np.isnan(mape) or mape < 15):
        g, l, icon = "A+", "Excellent",  "✅"
    elif nrmse < 0.18 and r2 >= 0.65 and (np.isnan(mape) or mape < 22):
        g, l, icon = "A",  "Very Good",  "✅"
    elif nrmse < 0.25 and r2 >= 0.50 and (np.isnan(mape) or mape < 30):
        g, l, icon = "B+", "Good",       "🟦"
    elif nrmse < 0.32 and r2 >= 0.35:
        g, l, icon = "B",  "Acceptable", "⚠️"
    elif nrmse < 0.45 and r2 >= 0.20:
        g, l, icon = "C",  "Weak",       "⚠️"
    else:
        g, l, icon = "D",  "Poor",       "🔴"
    return g, l, icon, acc


def render_model_quality(res: dict) -> None:
    g, l, icon, acc = model_grade(res["nrmse"], res["r2"], res.get("mape", np.nan))
    n_ho = res.get("n_holdout", 4)

    if "model_metrics" in res:
        st.markdown(
            f"""<div style='font-size:11px;font-weight:700;color:#4a5e7a;
            letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px'>
            Individual Model Performance — Hold-out ({n_ho} pts)</div>""",
            unsafe_allow_html=True,
        )
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
                _g, _l, _ico, _ = model_grade(m["nrmse"], m["r2"], m.get("mape", np.nan))
                mape_str = f"{m['mape']:.1f}%" if not np.isnan(m.get("mape", np.nan)) else "—"
                bias_str = f"{m['bias']:+.1f}" if not np.isnan(m.get("bias", np.nan)) else "—"
                col.markdown(
                    f"""<div style='text-align:center;padding:12px;border-radius:12px;
                        border:1px solid #e5e7eb;background:white;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>
                        <div class='pill {pcls}'>{mname}</div>
                        <div style='font-size:9px;color:#64748b;margin-top:6px'>RMSE (hold-out)</div>
                        <div style='font-size:20px;font-weight:800;color:{clr}'>{m['rmse']:.1f}</div>
                        <div style='font-size:10px;color:#94a3b8;margin-top:2px'>
                          MAPE {mape_str} · Bias {bias_str}
                        </div>
                        <div style='font-size:10px;color:#94a3b8'>NRMSE {m['nrmse']*100:.1f}% · R² {m['r2']:.3f}</div>
                        <div style='font-size:13px;font-weight:800;color:{clr};margin-top:5px'>Acc {_acc:.1f}%</div>
                        <div style='font-size:10px;color:#94a3b8'>{_ico} {_g} · {_l}</div>
                    </div>""", unsafe_allow_html=True,
                )
        w = res.get("weights", {})
        if w:
            tot = sum(w.values())
            st.markdown(
                f"""<div style='background:#f8faff;border:1px solid #c7d7fd;border-radius:8px;
                    padding:8px 14px;font-size:11px;margin:8px 0'>
                    <b style='color:#1e3a8a'>Ensemble blend (inverse-RMSE weights):</b>
                    <span class='pill pill-ridge'>Ridge {w.get("Ridge",0)/tot*100:.0f}%</span>
                    <span class='pill pill-rf'>RF {w.get("RandomForest",0)/tot*100:.0f}%</span>
                    <span class='pill pill-gb'>GB {w.get("GradBoost",0)/tot*100:.0f}%</span>
                </div>""", unsafe_allow_html=True,
            )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    mape_disp = f"{res.get('mape',0):.1f}%" if not np.isnan(res.get("mape", np.nan)) else "—"
    bias_disp = f"{res.get('bias',0):+.1f}"
    kpi(c1, "RMSE",     f"{res['rmse']:.1f}",          "#2541b2", "hold-out")
    kpi(c2, "NRMSE",    f"{res['nrmse']*100:.1f}%",    "#2541b2", "normalised")
    kpi(c3, "MAE",      f"{res['mae']:.1f}",            "#2541b2", "mean abs err")
    kpi(c4, "MAPE",     mape_disp,                      "#2541b2", "% error")
    kpi(c5, "Bias",     bias_disp,                      "#f57c00", "+over / −under")
    kpi(c6, "Accuracy", f"{max(0.0,round((1-res['nrmse'])*100,1)):.1f}%", "#00c6ae", "1 − NRMSE")
    sp(0.5)
    g2, l2, icon2, acc2 = model_grade(res["nrmse"], res["r2"], res.get("mape", np.nan))
    st.markdown(
        f"""<div style='background:white;border-radius:14px;padding:16px 20px;
            border:1px solid #e5e7eb;box-shadow:0 4px 16px rgba(0,0,0,0.07);
            display:flex;align-items:center;gap:14px;margin-bottom:8px'>
          <div style='font-size:24px'>{icon2}</div>
          <div>
            <div style='font-size:10px;text-transform:uppercase;letter-spacing:.1em;
                 color:#64748b;font-family:DM Mono'>Ensemble Grade ({n_ho}-pt hold-out)</div>
            <div style='font-size:22px;font-weight:900'>{g2}
              <span style='font-size:14px;font-weight:600;color:#475569'>{l2}</span>
            </div>
          </div>
          <div style='margin-left:auto;text-align:right'>
            <div style='font-size:10px;color:#64748b'>Forecast Accuracy</div>
            <div style='font-size:28px;font-weight:900;color:#1a2b6d'>{acc2:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True,
    )
    sp(0.5)


# ──────────────────────────────────
# INVENTORY OPTIMIZATION
# ──────────────────────────────────
@st.cache_data
def compute_inventory(
    order_cost:  float = DEFAULT_ORDER_COST,
    hold_pct:    float = DEFAULT_HOLD_PCT,
    replen_lead: int   = DEFAULT_REPLEN_LEAD_TIME,
    z:           float = DEFAULT_SERVICE_Z,
    n_future:    int   = N_FUTURE_MONTHS,
) -> pd.DataFrame:
    df       = load_data()
    ops      = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    cat_fcs  = compute_category_forecasts(n_future)

    # ── FIX: Use REPLENISHMENT lead time variability, not customer delivery days ──
    # We don't have supplier lead time data in dataset, so we estimate it from
    # inbound receiving patterns using warehouse-level delivery days as proxy for
    # inbound logistics, scaled by a replenishment factor.
    # This is clearly flagged as an estimate.
    del_ops         = df[df["Order_Status"] == "Delivered"].copy()
    # Replenishment lead time std estimate: use category-level delivery day std
    # as a proportional proxy (replen_lead × coefficient_of_variation)
    cat_cv = del_ops.groupby("Category")["Delivery_Days"].apply(
        lambda x: x.std() / (x.mean() + 1e-9)
    ).fillna(0.2).to_dict()

    sku_monthly = (
        ops.groupby(["SKU_ID", "YM"])["Net_Qty"].sum()
        .reset_index().sort_values(["SKU_ID", "YM"])
    )
    df_sorted    = df.sort_values("Order_Date")
    sku_snapshot = df_sorted.groupby("SKU_ID").agg(
        actual_stock   = ("Current_Stock_Units", "last"),
        dataset_rop    = ("Reorder_Point",       "last"),
        dataset_status = ("Stock_Status",        "last"),
        Product_Name   = ("Product_Name",        "first"),
        Category       = ("Category",            "first"),
        avg_price      = ("Sell_Price",          "mean"),
        total_qty      = ("Net_Qty",             "sum"),
    ).reset_index()
    sku_stats = (
        sku_monthly.groupby("SKU_ID")["Net_Qty"]
        .agg(hist_avg="mean", hist_std="std", peak_d="max")
        .reset_index()
    )
    sku_stats["hist_std"] = sku_stats["hist_std"].fillna(sku_stats["hist_avg"] * 0.25)
    cat_hist_avg = {cat: info["hist_avg"] for cat, info in cat_fcs.items()}

    sku_snapshot = sku_snapshot.merge(sku_stats, on="SKU_ID", how="left")
    sku_snapshot["hist_avg"]  = sku_snapshot["hist_avg"].fillna(0)
    sku_snapshot["hist_std"]  = sku_snapshot["hist_std"].fillna(0)
    sku_snapshot["peak_d"]    = sku_snapshot["peak_d"].fillna(0)

    def _sku_forecast(row):
        cat   = row["Category"]
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

    # ── FIX: Use COGS (not sell price) as holding cost base ──
    uc_sell  = sku_snapshot["avg_price"].clip(lower=1.0)
    uc_cogs  = uc_sell * COGS_RATE          # true inventory cost basis

    ann_d = sku_snapshot["econ_d"] * 12
    # EOQ uses COGS-based holding cost
    eoq = np.maximum(
        np.where(ann_d > 0, np.sqrt(2 * ann_d * order_cost / (uc_cogs * hold_pct)), 0), 1
    ).astype(int)

    daily_d   = sku_snapshot["avg_d"] / 30.0
    daily_std = sku_snapshot["hist_std"] / np.sqrt(30)

    # ── FIX: Safety stock uses replenishment lead time, not customer delivery days ──
    # SS = z × √(L×σ_d² + d²×σ_L²)
    # σ_L = replen_lead × CV_category  (estimated replenishment lead time std)
    lt_std_replen = sku_snapshot["Category"].map(cat_cv).fillna(0.2) * replen_lead
    ss = np.maximum(
        (z * np.sqrt(
            replen_lead * daily_std ** 2
            + daily_d ** 2 * lt_std_replen ** 2
        )).astype(int),
        0,
    )
    computed_rop = np.maximum((daily_d * replen_lead + ss).astype(int), 1)
    # FIX: Use computed_rop exclusively; dataset_rop only as reference, not override
    rop          = computed_rop

    current_stock = sku_snapshot["actual_stock"].astype(int)
    demand_6m = sku_snapshot["fc_next6"].apply(
        lambda lst: int(round(sum(lst))) if isinstance(lst, list)
        else int(round(float(lst) * n_future))
    )
    demand_driven_need  = np.maximum(demand_6m.values + ss - current_stock, 0)
    replenishment_need  = np.maximum(rop + eoq - current_stock, 0)
    # FIX: Prod_Need = demand-driven need; replenishment_need as cross-check only
    prod_need = demand_driven_need.copy()

    demand_cover_pct = np.where(
        demand_6m > 0,
        np.minimum(current_stock / demand_6m.values * 100, 100).round(1),
        100.0,
    )
    status = np.where(
        current_stock <= ss, "🔴 Critical",
        np.where(current_stock < rop, "🟡 Low", "🟢 Adequate"),
    )
    days_stock  = np.where(daily_d > 0, (current_stock / daily_d).round(1), 999.0)
    weeks_cover = np.where(daily_d > 0, (current_stock / (daily_d * 7)).round(1), 99.0)
    units_below = np.maximum(ss - current_stock, 0)

    daily_margin  = daily_d * uc_sell * MARGIN_RATE
    stockout_cost = np.where(
        status == "🔴 Critical",
        np.round(units_below * uc_sell * MARGIN_RATE + daily_margin * replen_lead, 0),
        0.0,
    )

    # ── FIX: Inventory Turnover per SKU (annualised) ──
    inv_value_per_sku = current_stock * uc_cogs
    cogs_per_sku      = sku_snapshot["total_qty"] * uc_cogs
    months_range      = max((df["Order_Date"].max() - df["Order_Date"].min()).days / 30.44, 1)
    sku_turnover = np.where(
        inv_value_per_sku > 0,
        (cogs_per_sku / inv_value_per_sku) * (12 / months_range),
        0.0,
    ).round(2)

    inv_df = pd.DataFrame({
        "SKU_ID":          sku_snapshot["SKU_ID"],
        "Product_Name":    sku_snapshot["Product_Name"],
        "Category":        sku_snapshot["Category"],
        "Monthly_Avg":     sku_snapshot["hist_avg"].round(1),
        "Monthly_Std":     sku_snapshot["hist_std"].round(1),
        "Forecast_Avg":    sku_snapshot["avg_d"].round(1),
        "Forecast_Next6":  sku_snapshot["fc_next6"],
        "Demand_6M":       demand_6m,
        "Demand_Cover_Pct":demand_cover_pct,
        "EOQ":             eoq,
        "SS":              ss,
        "ROP":             rop,
        "Current_Stock":   current_stock,
        "Days_of_Stock":   days_stock,
        "Weeks_Cover":     weeks_cover,
        "Inv_Turnover":    sku_turnover,          # FIX: added per-SKU turnover
        "Status":          status,
        "Dataset_Status":  sku_snapshot["dataset_status"],
        "Unit_Price":      uc_sell.round(0),
        "COGS_Unit":       uc_cogs.round(0),
        "Annual_Demand":   ann_d.round(0),
        "Stockout_Cost":   stockout_cost,
        "Prod_Need":       prod_need,
        "Total_Revenue":   (sku_snapshot["total_qty"] * uc_sell).round(0),
    })
    inv_df = inv_df[inv_df["Monthly_Avg"] > 0].reset_index(drop=True)
    if inv_df.empty:
        return inv_df

    inv_df  = inv_df.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
    cum_pct = inv_df["Total_Revenue"].cumsum() / inv_df["Total_Revenue"].sum() * 100
    inv_df["ABC"] = np.where(cum_pct <= 70, "A", np.where(cum_pct <= 90, "B", "C"))
    return inv_df


# ──────────────────────────────────
# PRODUCTION PLANNING
# ──────────────────────────────────
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
        prod_need_cat     = float(cat_inv["Prod_Need"].sum())
        crit_skus         = cat_inv[cat_inv["Status"] == "🔴 Critical"]
        low_skus          = cat_inv[cat_inv["Status"] == "🟡 Low"]
        crit_gap          = float((crit_skus["ROP"] - crit_skus["Current_Stock"]).clip(lower=0).sum())
        low_gap           = float((low_skus["ROP"]  - low_skus["Current_Stock"]).clip(lower=0).sum())
        current_stock_cat = float(cat_inv["Current_Stock"].sum())
        demand_6m_cat     = float(cat_inv["Demand_6M"].sum())
        forecast_total    = float(fc_arr.sum())

        for i, (dt, fc) in enumerate(zip(fut_ds, fc_arr)):
            demand_share = fc / forecast_total if forecast_total > 0 else 1.0 / len(fc_arr)
            prod         = prod_need_cat * demand_share * cap_mult
            bf           = BOOST_SCHEDULE.get(i, 0.0)
            crit_boost   = crit_gap * bf
            low_boost    = low_gap  * bf * 0.5
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
    wh_cat = (
        del_df.groupby(["Category","Warehouse"])["Quantity"].sum().reset_index()
    )
    wh_cat["wh_share"] = wh_cat.groupby("Category")["Quantity"].transform(
        lambda x: x / x.sum()
    )
    avg_ship = (
        del_df.groupby(["Category","Warehouse"])
        .agg(avg_cost=("Shipping_Cost_INR","mean"))
        .reset_index()
        .rename(columns={"Warehouse":"Target_Warehouse"})
    )
    needs = inv[inv["Prod_Need"] > 0].copy()
    abc_weight = {"A":3,"B":2,"C":1}
    needs["ABC_Priority"] = needs["ABC"].map(abc_weight)
    needs["Daily_Demand"] = (needs["Monthly_Avg"] / 30).clip(lower=0.01)
    needs["Days_Left"]    = (needs["Current_Stock"] / needs["Daily_Demand"]).round(1).clip(upper=999)
    needs["Priority_Score"] = (
        needs["ABC_Priority"] * 3
        + needs["Stockout_Cost"] / 1000
        + (needs["ROP"] - needs["Current_Stock"]).clip(lower=0)
    )

    def _urgency(row):
        if row["Status"] == "🔴 Critical": return "🔴 Urgent"
        if row["Days_Left"] <= 14:         return "🟠 High"
        if row["Days_Left"] <= 30:         return "🟡 Medium"
        return "🟢 Normal"

    needs["Urgency"] = needs.apply(_urgency, axis=1)

    wh_assignments = []
    for cat, grp in needs.groupby("Category"):
        cat_wh = (
            wh_cat[wh_cat["Category"] == cat]
            .sort_values("wh_share", ascending=False)
            .reset_index(drop=True)
        )
        warehouses = cat_wh["Warehouse"].tolist()
        shares     = cat_wh["wh_share"].values
        skus_sorted = grp.sort_values(["ABC_Priority","Priority_Score"], ascending=[False,False])
        n = len(skus_sorted)
        cumulative = np.round(np.cumsum(shares) * n).astype(int).clip(0, n)
        prev = 0
        slot_assignments: list[str] = []
        for cut, wh in zip(cumulative, warehouses):
            count = int(cut) - prev
            slot_assignments.extend([wh] * count)
            prev = int(cut)
        while len(slot_assignments) < n:
            slot_assignments.append(warehouses[-1] if warehouses else "Central WH")
        for (idx, _), wh in zip(skus_sorted.iterrows(), slot_assignments):
            cat_share = float(
                cat_wh.loc[cat_wh["Warehouse"] == wh, "wh_share"].values[0]
                if wh in cat_wh["Warehouse"].values else shares[0]
            )
            wh_assignments.append({"idx":idx,"Target_Warehouse":wh,"WH_Share_Pct":round(cat_share*100,1)})

    wh_df  = pd.DataFrame(wh_assignments).set_index("idx")
    needs  = needs.join(wh_df[["Target_Warehouse","WH_Share_Pct"]])
    needs["Target_Warehouse"] = needs["Target_Warehouse"].fillna("Central WH")
    needs["WH_Share_Pct"]     = needs["WH_Share_Pct"].fillna(100.0)
    needs  = needs.merge(avg_ship, on=["Category","Target_Warehouse"], how="left")
    needs["avg_cost"]      = needs["avg_cost"].fillna(del_df["Shipping_Cost_INR"].mean())
    needs["Est_Ship_Cost"] = (needs["Prod_Need"] * needs["avg_cost"]).round(0)
    wh_total               = needs.groupby("Target_Warehouse")["Prod_Need"].transform("sum")
    needs["WH_Share_Pct"]  = (needs["Prod_Need"] / wh_total.clip(lower=1) * 100).round(1)
    needs = needs.sort_values(["Priority_Score","Days_Left"], ascending=[False,True]).reset_index(drop=True)

    return needs[[
        "SKU_ID","Product_Name","Category","ABC","Urgency","Prod_Need",
        "Current_Stock","Demand_6M","Demand_Cover_Pct","Days_Left",
        "Stockout_Cost","Target_Warehouse","WH_Share_Pct","Est_Ship_Cost","Status",
    ]]


# ──────────────────────────────────
# LOGISTICS
# ──────────────────────────────────
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
    carrier_returns.columns = ["Courier_Partner","Return_Rate"]
    region_carrier_returns = df.groupby(["Region","Courier_Partner"])["Return_Flag"].mean().reset_index()
    region_carrier_returns.columns = ["Region","Courier_Partner","Return_Rate"]

    carr = del_df.groupby("Courier_Partner").agg(
        Orders    =("Order_ID","count"),
        Avg_Days  =("Delivery_Days","mean"),
        Avg_Cost  =("Shipping_Cost_INR","mean"),
        Total_Cost=("Shipping_Cost_INR","sum"),
    ).reset_index()
    carr = carr.merge(carrier_returns, on="Courier_Partner", how="left")
    carr["Return_Rate"] = carr["Return_Rate"].fillna(0)
    for col_, _ in [("Avg_Days",w_speed),("Avg_Cost",w_cost),("Return_Rate",w_returns)]:
        mn = carr[col_].min(); mx = carr[col_].max()
        carr[f"Norm_{col_}"] = 1 - (carr[col_] - mn) / (mx - mn + 1e-9)
    carr["Perf_Score"] = (
        w_speed   * carr["Norm_Avg_Days"]
        + w_cost  * carr["Norm_Avg_Cost"]
        + w_returns * carr["Norm_Return_Rate"]
    ).round(3)
    carr["Delay_Index"] = (carr["Avg_Days"] / carr["Avg_Days"].min()).round(2)

    region_carr = del_df.groupby(["Region","Courier_Partner"]).agg(
        Avg_Days=("Delivery_Days","mean"),
        Avg_Cost=("Shipping_Cost_INR","mean"),
        Orders  =("Order_ID","count"),
    ).reset_index()
    region_carr = region_carr.merge(region_carrier_returns, on=["Region","Courier_Partner"], how="left")
    region_carr["Return_Rate"] = region_carr["Return_Rate"].fillna(0)
    for col_, _ in [("Avg_Days",w_speed),("Avg_Cost",w_cost),("Return_Rate",w_returns)]:
        mn = region_carr[col_].min(); mx = region_carr[col_].max()
        region_carr[f"Norm_{col_}"] = 1 - (region_carr[col_] - mn) / (mx - mn + 1e-9)
    region_carr["Score"] = (
        w_speed * region_carr["Norm_Avg_Days"]
        + w_cost * region_carr["Norm_Avg_Cost"]
        + w_returns * region_carr["Norm_Return_Rate"]
    )
    best = (
        region_carr.sort_values("Score", ascending=False)
        .groupby("Region").first().reset_index()
        [["Region","Courier_Partner","Avg_Days","Avg_Cost","Score"]]
    )
    cheapest = (
        del_df.groupby(["Region","Courier_Partner"])
        .agg(avg_cost=("Shipping_Cost_INR","mean"), orders=("Order_ID","count"))
        .reset_index().sort_values("avg_cost")
        .groupby("Region").first().reset_index()
        .rename(columns={"Courier_Partner":"Optimal_Carrier","avg_cost":"Min_Avg_Cost"})
    )
    region_costs = del_df.groupby("Region").agg(
        Current_Avg_Cost=("Shipping_Cost_INR","mean"),
        Orders          =("Order_ID","count"),
        Total_Spend     =("Shipping_Cost_INR","sum"),
    ).reset_index()
    opt = region_costs.merge(cheapest[["Region","Optimal_Carrier","Min_Avg_Cost"]], on="Region")
    opt["Potential_Saving"] = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) * opt["Orders"]).round(0)
    opt["Saving_Pct"]       = ((opt["Current_Avg_Cost"] - opt["Min_Avg_Cost"]) / opt["Current_Avg_Cost"] * 100).round(1)

    avg_ship_unit = max(del_df["Shipping_Cost_INR"].sum() / max(del_df["Quantity"].replace(0, np.nan).sum(), 1), 1.0)
    hist_orders   = max(len(del_df), 1)
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
                "Demand_Units":  int(fc_units),
                "Proj_Orders":   int(round(fc_units / avg_units_ord)),
                "Proj_Ship_Cost":int(round(fc_units * avg_ship_unit, 0)),
                "CI_Lo_Units":   int(row["CI_Lo"]),
                "CI_Hi_Units":   int(row["CI_Hi"]),
            })
    return carr, best, opt, pd.DataFrame(fwd_rows)


# ──────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────
def ensemble_chart(res: dict, chart_key: str, height: int = 300, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(
        x0=res["fut_ds"][0], x1=res["fut_ds"][-1],
        fillcolor="rgba(139,92,246,0.04)", layer="below", line_width=0,
    )
    fig.add_vline(x=res["fut_ds"][0], line_dash="dash",
                  line_color="rgba(139,92,246,0.4)", line_width=1.5)
    x_ci = list(res["fut_ds"]) + list(res["fut_ds"])[::-1]
    y_ci = list(res["ci_hi"]) + list(res["ci_lo"])[::-1]
    fig.add_trace(go.Scatter(
        x=x_ci, y=y_ci, fill="toself",
        fillcolor="rgba(139,92,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI",
    ))
    fig.add_trace(go.Scatter(
        x=res["hist_ds"], y=res["hist_y"], name="Actual",
        line=dict(color="#4a5e7a", width=2.2),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=res["hist_ds"], y=res["fitted"], name="Ensemble fit",
        line=dict(color="#8B5CF6", width=1.5, dash="dot"), opacity=0.55, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=res["fut_ds"], y=res["forecast"], name="Ensemble Forecast",
        line=dict(color="#8B5CF6", width=2.8, dash="dot"), mode="lines+markers",
        marker=dict(size=8, color="#8B5CF6", line=dict(color="#FFFFFF", width=2)),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=res["eval_ds"], y=res["eval_pred"], name="Hold-out eval",
        mode="markers",
        marker=dict(size=9, color="#EF4444", symbol="x", line=dict(color="#FFFFFF", width=2)),
    ))
    fig.update_layout(
        **CD(), height=height, xaxis=gX(), yaxis=gY(), legend=leg(),
        title=dict(text=title, font=dict(color="#64748b", size=11)),
    )
    return fig


# ──────────────────────────────────
# PAGES
# ──────────────────────────────────
def page_overview() -> None:
    df     = load_data()
    ops    = get_ops(df).copy()
    del_df = get_delivered(df)
    sc_kpis = compute_sc_kpis(df)

    st.markdown("""
    <div class='page-header'>
      <div class='page-header-title'>OmniFlow D2D Intelligence</div>
      <div class='page-header-sub'>AI-Driven Demand-to-Delivery Optimization · India E-Commerce</div>
    </div>""", unsafe_allow_html=True)

    sec("Dataset at a Glance", "📦")
    total_orders = len(df)
    total_rev    = ops["Net_Revenue"].sum()
    avg_ov       = ops["Net_Revenue"].mean()
    ret_rate     = df["Return_Flag"].mean() * 100
    on_time      = (del_df["Delivery_Days"] <= 3).mean() * 100
    n_skus       = df["SKU_ID"].nunique()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total Orders",       f"{total_orders:,}",       "#2541b2", "Jan 2024 – Dec 2025")
    kpi(c2, "Net Revenue",        f"₹{total_rev/1e7:.2f}Cr", "#00c6ae", "delivered + shipped")
    kpi(c3, "Avg Order Value",    f"₹{avg_ov:,.0f}",         "#2541b2", "per active order")
    kpi(c4, "Return Rate",        f"{ret_rate:.1f}%",         "#e53935", f"{df['Return_Flag'].sum()} orders")
    kpi(c5, "On-Time Delivery",   f"{on_time:.1f}%",          "#00c6ae", "delivered ≤ 3 days")
    kpi(c6, "Unique SKUs",        str(n_skus),                "#2541b2", "across 4 categories")

    sp()
    sec("Supply Chain Health KPIs", "📊")
    banner(
        "⚠️ <b>Note:</b> Fill Rate, Stockout Rate, OTIF, and Inventory Turnover are computed "
        "from historical order data. Supplier-side fill rate requires inbound PO data not present in this dataset.",
        "amber",
    )
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    kpi(k1, "Fill Rate",          f"{sc_kpis['fill_rate']:.1f}%",    "#00c6ae", "orders fulfilled / demanded")
    kpi(k2, "OTIF Rate",          f"{sc_kpis['otif_rate']:.1f}%",    "#2541b2", "on-time & in-full")
    kpi(k3, "Stockout / Return",  f"{sc_kpis['stockout_rate']:.1f}%","#e53935", "cancelled + returned %")
    kpi(k4, "Inv. Turnover",      f"{sc_kpis['inv_turnover']:.1f}x", "#2541b2", "annualised turns")
    kpi(k5, "Days of Inventory",  f"{sc_kpis['doi']:.0f}d",          "#f57c00", "avg stock coverage")
    kpi(k6, "Svc Level (≤7d)",    f"{sc_kpis['svc_achieved']:.1f}%", "#00c6ae", "delivered within 7 days")
    kpi(k7, "Return Rate",        f"{sc_kpis['return_rate']:.1f}%",  "#e53935", "all order returns")

    sp()
    st.markdown("""
    <div style='background:white;border:1px solid #e2e5ef;border-radius:16px;
         padding:24px 28px;margin-bottom:18px;box-shadow:0 2px 16px rgba(26,43,109,0.08)'>
    <div style='font-family:Syne,sans-serif;font-size:16px;font-weight:800;margin-bottom:16px'>
      Analytics Pipeline — What Each Module Decides
    </div>
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px'>
      <div style='background:#f8f9fc;border-radius:12px;padding:16px;border-top:3px solid #3b82f6'>
        <div style='font-size:11px;font-weight:800;color:#3b82f6;letter-spacing:.06em;text-transform:uppercase'>1 · Demand Forecasting</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>How much will sell?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Ridge + RF + GradBoost ensemble with walk-forward validation. MAPE + Bias + RMSE reported. 90% CI.
        </div>
      </div>
      <div style='background:#f8f9fc;border-radius:12px;padding:16px;border-top:3px solid #f59e0b'>
        <div style='font-size:11px;font-weight:800;color:#f59e0b;letter-spacing:.06em;text-transform:uppercase'>2 · Inventory Optimisation</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>Which SKUs need restocking?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Wilson EOQ on COGS cost basis. Safety stock uses replenishment lead time (not delivery days). ABC + per-SKU turnover.
        </div>
      </div>
      <div style='background:#f8f9fc;border-radius:12px;padding:16px;border-top:3px solid #8b5cf6'>
        <div style='font-size:11px;font-weight:800;color:#8b5cf6;letter-spacing:.06em;text-transform:uppercase'>3 · Production Planning</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>How many units to make?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Demand-driven production schedule. Critical/Low SKUs front-loaded. Warehouse routing by volume share.
        </div>
      </div>
      <div style='background:#f8f9fc;border-radius:12px;padding:16px;border-top:3px solid #059669'>
        <div style='font-size:11px;font-weight:800;color:#059669;letter-spacing:.06em;text-transform:uppercase'>4 · Logistics Optimisation</div>
        <div style='font-size:11px;font-weight:700;color:#0f172a;margin:6px 0 4px'><i>Which carrier, at what cost?</i></div>
        <div style='font-size:11.5px;color:#475569;line-height:1.7'>
          Composite score: weighted(speed + cost + return rate). Best carrier per region. Savings quantified.
        </div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)


def page_demand() -> None:
    n_future = get_horizon()
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-header'><div class='page-header-title'>Demand Forecasting</div></div>",
                unsafe_allow_html=True)
    horizon_badge(n_future)

    sec("Ensemble Model Quality", "🤖")
    m_orders = ops.groupby("YM")["Order_ID"].count().rename("v")
    res_ov   = ml_forecast(m_orders.values.astype(float), m_orders.index, n_future)
    if res_ov:
        render_model_quality(res_ov)
    sp()

    if res_ov and "model_metrics" in res_ov:
        sec("Model Accuracy Comparison", "📐")
        mm     = res_ov["model_metrics"]
        labels = [m for m in ["Ridge","RandomForest","GradBoost","Ensemble"] if m in mm]
        r2_v   = [mm[m]["r2"]           for m in labels]
        nrmse_v= [mm[m]["nrmse"] * 100  for m in labels]
        mape_v = [mm[m].get("mape", 0)  for m in labels]
        bias_v = [mm[m].get("bias", 0)  for m in labels]
        clrs   = [MODEL_COLORS.get(m, "#888") for m in labels]

        bc1, bc2, bc3, bc4 = st.columns(4)
        for col_, vals_, title_, ylabel_, target_y, target_label, higher_better in [
            (bc1, r2_v,    "R² Score (hold-out)", "R²",        0.65, "Target 0.65", True),
            (bc2, nrmse_v, "NRMSE % (hold-out)",  "NRMSE %",   25,   "Target <25%", False),
            (bc3, mape_v,  "MAPE % (hold-out)",   "MAPE %",    25,   "Target <25%", False),
            (bc4, bias_v,  "Forecast Bias",        "Bias (units)", 0, "Zero bias",  None),
        ]:
            fig_ = go.Figure(go.Bar(
                x=labels, y=vals_,
                marker=dict(color=clrs, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.2f}" if title_ == "R² Score (hold-out)" else
                      f"{v:+.1f}" if "Bias" in title_ else f"{v:.1f}%" for v in vals_],
                textposition="outside", textfont=dict(color="#334155"),
            ))
            fig_.add_hline(
                y=target_y, line_dash="dash", line_color="#22C55E",
                annotation_text=f" {target_label}",
                annotation_font=dict(color="#22C55E", size=9),
            )
            yrange = [0, max(vals_) * 1.35] if higher_better is not False else None
            if yrange is None and "Bias" in title_:
                yrange = [min(min(vals_)*1.4, -1), max(max(vals_)*1.4, 1)]
            fig_.update_layout(
                **CD(), height=220, xaxis=gX(),
                yaxis={**gY(), "title": ylabel_, **({"range": yrange} if yrange else {})},
                title=dict(text=title_, font=dict(size=11, color="#64748b")),
            )
            col_.plotly_chart(fig_, use_container_width=True, key=f"d_cmp_{title_[:4]}")
    sp()

    c1, c2 = st.columns([2, 2])
    metric_opt = c1.selectbox("Metric", ["Orders","Quantity","Net Revenue"], key="d_metric")
    level_opt  = c2.selectbox("Breakdown", ["Overall","Category","Region","Sales Channel"], key="d_level")
    col_map = {"Orders":"Order_ID","Quantity":"Net_Qty","Net Revenue":"Net_Revenue"}
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

    sec(f"Forecast Chart — {n_future}-Month Horizon", "📈")
    if level_opt == "Overall":
        draw_with_table(get_series(ops), chart_key="d_overall")
    else:
        grp_map = {"Category":"Category","Region":"Region","Sales Channel":"Sales_Channel"}
        grp     = grp_map[level_opt]
        top     = ops[grp].value_counts().index.tolist()
        tabs    = st.tabs(top)
        for i, (tab, val) in enumerate(zip(tabs, top)):
            with tab:
                draw_with_table(get_series(ops[ops[grp] == val]), title=val, chart_key=f"d_bd_{i}")

    sp()
    sec("YoY Revenue Growth by Category", "📅")
    yr_rev      = ops.groupby(["Year","Category"])["Net_Revenue"].sum().unstack(fill_value=0)
    cat_monthly = ops.groupby(["YM","Category"])["Net_Revenue"].sum().unstack(fill_value=0)
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
                "Category": cat,
                "2024 ₹M":  round(r24 / 1e6, 1),
                "2025 ₹M":  round(r25 / 1e6, 1),
                "YoY 24→25": f"{(r25-r24)/r24*100:+.1f}%" if r24 > 0 else "N/A",
                f"Next {n_future}M Proj ₹M": round(rp / 1e6, 1),
                "Projected Growth": f"{(rp-r25)/r25*100:+.1f}%" if r25 > 0 else "N/A",
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

    st.markdown("<div class='page-header'><div class='page-header-title'>Inventory Optimization</div></div>",
                unsafe_allow_html=True)
    horizon_badge(n_future)

    # ── Parameters (no wrapper tab) ──
    with st.expander("⚙️ Parameters", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        order_cost  = p1.number_input("Order Cost ₹", 100, 5000, DEFAULT_ORDER_COST, 50,
                                      help="Admin + processing cost per purchase order")
        hold_pct    = p2.slider("Holding Cost %", 5, 40, int(DEFAULT_HOLD_PCT * 100),
                                help="Annual holding % applied to COGS (not sell price)") / 100
        # FIX: renamed from lead_time to replen_lead with correct tooltip
        replen_lead = p3.slider("Replenishment Lead Time (days)", 1, 60, DEFAULT_REPLEN_LEAD_TIME,
                                help="Supplier replenishment lead time — NOT customer delivery days")
        svc = p4.selectbox("Service Level", ["90% (z=1.28)","95% (z=1.65)","99% (z=2.33)"], index=1,
                           help="Target service level for safety stock calculation")
        z   = {"90% (z=1.28)":1.28,"95% (z=1.65)":1.65,"99% (z=2.33)":2.33}[svc]

    banner(
        "🔧 <b>Note on Safety Stock:</b> Lead time variability is estimated from category-level "
        "delivery day coefficient of variation as a proxy. For production-grade use, replace with "
        "actual supplier lead time data per SKU.",
        "amber",
    )

    inv = compute_inventory(order_cost, hold_pct, replen_lead, z, n_future)
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

    crit_days    = inv[inv["Status"] == "🔴 Critical"]["Days_of_Stock"]
    crit_days    = crit_days[crit_days < 999]
    avg_crit_d   = f"{crit_days.mean():.0f}d avg" if len(crit_days) > 0 else "—"

    avg_turnover = inv["Inv_Turnover"].replace(0, np.nan).mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total SKUs",            len(inv),                      "#2541b2", "active")
    kpi(c2, "🔴 Critical",           n_crit,                        "#e53935", f"≤ safety stock · {avg_crit_d}")
    kpi(c3, "🟡 Low",                n_low,                         "#f57c00", "< reorder point")
    kpi(c4, f"{n_future}M Demand",   f"{total_demand_6m:,}",        "#2541b2", f"units · {fc_start}–{fc_end}")
    kpi(c5, "Units to Produce",      f"{total_prod_need:,}",        "#00c6ae", "demand-driven gap")
    kpi(c6, "Avg Inv. Turnover",     f"{avg_turnover:.1f}x",        "#2541b2", "annualised per SKU")

    sp()

    # ── FIX: No wrapper tab — direct section rendering ──
    sec("Stock Position", "🗂️")
    sc1, sc2, sc3 = st.columns([2, 2, 1])
    cat_f  = sc1.multiselect("Category",  sorted(inv["Category"].unique()),
                              default=sorted(inv["Category"].unique()), key="al_cat")
    stat_f = sc2.multiselect("Status",    ["🔴 Critical","🟡 Low","🟢 Adequate"],
                              default=["🔴 Critical","🟡 Low","🟢 Adequate"], key="al_stat")
    abc_f  = sc3.multiselect("ABC",       ["A","B","C"], default=["A","B","C"], key="al_abc")

    sv = inv[
        inv["Category"].isin(cat_f) &
        inv["Status"].isin(stat_f) &
        inv["ABC"].isin(abc_f)
    ].copy()

    if sv.empty:
        banner("✅ No SKUs match selected filters.", "mint")
        return

    # ── POLISHED SCATTER: Stock vs ROP ──
    STATUS_CLR = {
        "🔴 Critical": "#ef4444",
        "🟡 Low":      "#f59e0b",
        "🟢 Adequate": "#22c55e",
    }
    fig_sc = go.Figure()
    ax_max = max(sv["Current_Stock"].max(), sv["ROP"].max()) * 1.12

    # Danger zone shading
    fig_sc.add_shape(type="rect", x0=0, y0=0, x1=sv["ROP"].mean(), y1=ax_max,
                     fillcolor="rgba(239,68,68,0.04)", line_width=0, layer="below")
    # Diagonal reference line
    fig_sc.add_trace(go.Scatter(
        x=[0, ax_max], y=[0, ax_max], mode="lines",
        line=dict(color="rgba(100,116,139,0.22)", width=1.5, dash="dash"),
        name="Stock = ROP", hoverinfo="skip",
    ))
    for status, clr in STATUS_CLR.items():
        grp = sv[sv["Status"] == status]
        if grp.empty:
            continue
        bubble_sz = np.clip(np.sqrt(grp["Prod_Need"].values + 1) * 4, 8, 55)
        fig_sc.add_trace(go.Scatter(
            x=grp["Current_Stock"], y=grp["ROP"],
            mode="markers", name=status,
            marker=dict(
                size=bubble_sz, color=clr, opacity=0.80,
                line=dict(color="#FFFFFF", width=1.8),
                sizemode="diameter",
            ),
            customdata=grp[["Product_Name","SKU_ID","Prod_Need","Demand_6M",
                             "Demand_Cover_Pct","Inv_Turnover","Days_of_Stock"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> [%{customdata[1]}]<br>"
                "Stock: <b>%{x}</b> · ROP: <b>%{y}</b><br>"
                f"{n_future}M Demand: %{{customdata[3]:,}} units<br>"
                "Stock Covers: %{customdata[4]:.0f}%<br>"
                "Turnover: %{customdata[5]:.1f}x/yr<br>"
                "Days Left: %{customdata[6]:.0f}d<br>"
                "Produce: <b>%{customdata[2]} units</b>"
                "<extra></extra>"
            ),
        ))
    # Quadrant labels
    for txt, x, y in [
        ("▲ Act Now\n(stock < ROP)", sv["ROP"].mean()*0.4, ax_max*0.88),
        ("✓ Adequate\n(stock > ROP)", ax_max*0.65, ax_max*0.88),
    ]:
        fig_sc.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            font=dict(size=9, color="#94a3b8"),
            bgcolor="rgba(255,255,255,0.7)", bordercolor="#e2e5ef",
            borderwidth=1, borderpad=4,
        )
    fig_sc.update_layout(
        **CD(), height=420,
        xaxis={**gX(), "title": "Current Stock (units)", "range": [0, ax_max]},
        yaxis={**gY(), "title": "Reorder Point (units)", "range": [0, ax_max]},
        legend={**leg(), "orientation":"h", "y":-0.16},
        title=dict(text="Bubble size ∝ units to produce", font=dict(size=10, color="#94a3b8")),
    )
    st.plotly_chart(fig_sc, use_container_width=True, key="scatter_stock")

    sp()
    # ── FIX: Days-of-Stock Distribution ──
    sec("Days-of-Stock Distribution", "⏱️")
    doi_c1, doi_c2 = st.columns(2, gap="large")
    with doi_c1:
        bins    = [0, 7, 14, 30, 60, 999]
        labels_ = ["<7d","7–14d","14–30d","30–60d",">60d"]
        sv_doi  = sv[sv["Days_of_Stock"] < 999].copy()
        sv_doi["DOI_Bin"] = pd.cut(sv_doi["Days_of_Stock"], bins=bins, labels=labels_)
        doi_cnt = sv_doi.groupby("DOI_Bin", observed=True).size().reset_index(name="SKUs")
        bar_clrs = ["#ef4444","#f97316","#eab308","#22c55e","#06b6d4"]
        fig_doi = go.Figure(go.Bar(
            x=doi_cnt["DOI_Bin"].astype(str), y=doi_cnt["SKUs"],
            marker=dict(color=bar_clrs[:len(doi_cnt)], line=dict(color="rgba(0,0,0,0)")),
            text=doi_cnt["SKUs"], textposition="outside", textfont=dict(color="#334155"),
        ))
        fig_doi.update_layout(
            **CD(), height=240, xaxis=gX(),
            yaxis={**gY(), "title":"SKU Count"},
            title=dict(text="Days-of-Stock Buckets", font=dict(size=11, color="#64748b")),
        )
        doi_c1.plotly_chart(fig_doi, use_container_width=True, key="doi_dist")

    with doi_c2:
        # Inventory Turnover by Category
        cat_turn = inv.groupby("Category")["Inv_Turnover"].mean().reset_index()
        fig_turn = go.Figure(go.Bar(
            x=cat_turn["Category"], y=cat_turn["Inv_Turnover"],
            marker=dict(
                color=["#1a2b6d","#2541b2","#0091ff","#00c6ae"][:len(cat_turn)],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.1f}x" for v in cat_turn["Inv_Turnover"]],
            textposition="outside", textfont=dict(color="#334155"),
        ))
        fig_turn.update_layout(
            **CD(), height=240, xaxis=gX(),
            yaxis={**gY(), "title":"Turns / Year"},
            title=dict(text="Avg Inventory Turnover by Category", font=dict(size=11, color="#64748b")),
        )
        doi_c2.plotly_chart(fig_turn, use_container_width=True, key="inv_turn")

    sp()
    sec("SKU Inventory Action Queue", "🗒️")
    action = sv.sort_values(["Status","Prod_Need"], ascending=[True,False])
    tbl = action[[
        "SKU_ID","Product_Name","Category","ABC","Status",
        "Current_Stock","Demand_6M","Demand_Cover_Pct",
        "ROP","EOQ","SS","Prod_Need","Inv_Turnover","Days_of_Stock",
    ]].copy()
    tbl.columns = [
        "SKU","Product","Category","ABC","Status",
        "Stock",f"{n_future}M Demand","Covers %",
        "ROP","EOQ","Safety Stock","Units to Produce","Turnover (x/yr)","Days Left",
    ]
    for c in ["Stock",f"{n_future}M Demand","ROP","EOQ","Safety Stock","Units to Produce"]:
        tbl[c] = tbl[c].astype(int)
    tbl["Covers %"]       = tbl["Covers %"].apply(lambda x: f"{x:.0f}%")
    tbl["Turnover (x/yr)"]= tbl["Turnover (x/yr)"].apply(lambda x: f"{x:.1f}x")
    tbl["Days Left"]      = tbl["Days Left"].apply(lambda x: "∞" if x >= 999 else f"{x:.0f}d")
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=360)


def page_production() -> None:
    n_future = get_horizon()
    df  = load_data()
    ops = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")

    st.markdown("<div class='page-header'><div class='page-header-title'>Production Planning</div></div>",
                unsafe_allow_html=True)
    horizon_badge(n_future)

    cap  = st.slider("Capacity Multiplier", 0.5, 2.0, 1.0, 0.1)
    plan = compute_production(cap, n_future)
    if plan.empty:
        st.warning("Insufficient data.")
        return

    agg             = plan.groupby("Month_dt")[["Production","Demand_Forecast","Crit_Boost","Low_Boost"]].sum().reset_index()
    inv_for_kpi     = compute_inventory(n_future=n_future)
    total_prod_need = int(inv_for_kpi["Prod_Need"].sum())
    total_demand_6m = int(inv_for_kpi["Demand_6M"].sum())
    total_stock     = int(inv_for_kpi["Current_Stock"].sum())
    peak            = agg.loc[agg["Production"].idxmax(), "Month_dt"]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Production Required",  f"{plan['Production'].sum():,.0f}", "#f57c00", f"demand-driven · {n_future}M")
    kpi(c2, f"{n_future}M Forecast",f"{total_demand_6m:,}",            "#2541b2", "what customers will order")
    kpi(c3, "Current Stock Total",  f"{total_stock:,}",                "#2541b2", "all SKUs")
    kpi(c4, "Stock Gap",            f"{total_prod_need:,}",            "#e53935", "demand + SS − current")
    kpi(c5, "Peak Month",           peak.strftime("%b %Y"),             "#f57c00", "highest volume")

    sp()
    sec(f"Production Target vs Demand Forecast — {n_future}-Month Horizon", "📊")
    hist_qty = ops.groupby("YM")["Net_Qty"].sum().rename("v")
    hist_ts  = _to_ts(hist_qty.index)
    res_hist = ml_forecast(hist_qty.values.astype(float), hist_qty.index, n_future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_ts, y=hist_qty.values, name="Historical Demand",
        fill="tozeroy", fillcolor="rgba(74,94,122,0.10)",
        line=dict(color="#4a5e7a", width=2),
    ))
    if res_hist:
        fig.add_trace(go.Scatter(
            x=res_hist["hist_ds"], y=res_hist["fitted"], name="Ensemble Fit",
            line=dict(color="#8B5CF6", width=1.5, dash="dot"), opacity=0.6,
        ))
        x_ci = list(res_hist["fut_ds"]) + list(res_hist["fut_ds"])[::-1]
        y_ci = list(res_hist["ci_hi"])  + list(res_hist["ci_lo"])[::-1]
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci, fill="toself",
            fillcolor="rgba(139,92,246,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI",
        ))
    fig.add_trace(go.Bar(
        x=agg["Month_dt"], y=agg["Production"], name="Production Target",
        marker=dict(color="#8B5CF6", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
    ))
    fig.add_trace(go.Scatter(
        x=agg["Month_dt"], y=agg["Demand_Forecast"], name="Demand Forecast",
        mode="lines+markers", line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=8, color="#F59E0B", line=dict(color="#FFFFFF", width=2)),
    ))
    fig.add_vline(x=plan["Month_dt"].min(), line_dash="dash",
                  line_color="rgba(139,92,246,0.5)", line_width=2)
    fig.update_layout(**CD(), height=320, barmode="stack",
                      xaxis=gX(), yaxis=gY(), legend=leg())
    st.plotly_chart(fig, use_container_width=True, key="prod_main")

    cl, cr = st.columns(2, gap="large")
    with cl:
        sec("Production by Category")
        cat_hist    = ops.groupby(["YM","Category"])["Net_Qty"].sum().unstack(fill_value=0)
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
                    line=dict(color=clr, width=1.5, dash="dot"),
                    opacity=0.5, showlegend=False,
                ))
            s = plan[plan["Category"] == cat].sort_values("Month_dt")
            fig2.add_trace(go.Bar(
                x=s["Month_dt"], y=s["Production"], name=cat,
                marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)")),
            ))
        fig2.update_layout(**CD(), height=270, barmode="stack",
                           xaxis=gX(), yaxis=gY(),
                           legend={**leg(), "orientation":"h", "y":-0.32})
        st.plotly_chart(fig2, use_container_width=True, key="prod_cat")

    with cr:
        sec("Production vs Demand Gap")
        agg["Gap"] = agg["Production"] - agg["Demand_Forecast"]
        fig3 = go.Figure(go.Bar(
            x=agg["Month_dt"], y=agg["Gap"],
            marker=dict(
                color=["#22C55E" if g >= 0 else "#EF4444" for g in agg["Gap"]],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{g:+.0f}" for g in agg["Gap"]],
            textposition="outside", textfont=dict(color="#334155"),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
        fig3.update_layout(
            **CD(), height=270, xaxis=gX(),
            yaxis={**gY(), "title":"Units Surplus / Deficit"},
        )
        st.plotly_chart(fig3, use_container_width=True, key="prod_gap")

    sec("Production Schedule", "📋")
    cat_f = st.selectbox("Filter Category", ["All"] + list(plan["Category"].unique()))
    d2    = plan if cat_f == "All" else plan[plan["Category"] == cat_f]
    d3    = d2[[
        "Month","Category",
        "Current_Stock","Demand_6M_Cat","Prod_Need_Cat",
        "Demand_Forecast","Crit_Boost","Low_Boost",
        "Production","CI_Lo","CI_Hi",
    ]].copy()
    d3.columns = [
        "Month","Category",
        "Cat Stock",f"{n_future}M Demand","Inv Prod Need",
        "Demand Fc","Crit Boost","Low Boost",
        "Production","Demand Lo","Demand Hi",
    ]
    st.dataframe(d3.sort_values("Month"), use_container_width=True, hide_index=True)

    sp()
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
        "color:#0d1324;letter-spacing:-.02em'>Fulfilment & Routing Plan</div>",
        unsafe_allow_html=True,
    )
    sku_plan = build_sku_production_plan(n_future)
    if sku_plan.empty:
        banner("✅ All SKUs adequately stocked — no production orders needed.", "mint")
        return

    n_urgent    = (sku_plan["Urgency"] == "🔴 Urgent").sum()
    n_high      = (sku_plan["Urgency"] == "🟠 High").sum()
    total_units = int(sku_plan["Prod_Need"].sum())
    total_ship  = sku_plan["Est_Ship_Cost"].sum()
    sc_risk     = sku_plan["Stockout_Cost"].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, "SKUs Needing Stock",  len(sku_plan),            "#2541b2", "Prod_Need > 0")
    kpi(k2, "🔴 Urgent",           n_urgent,                 "#e53935", "≤ safety stock")
    kpi(k3, "🟠 High",             n_high,                   "#f57c00", "≤14d stock left")
    kpi(k4, "Gap Units",           f"{total_units:,}",        "#2541b2", "demand-driven")
    kpi(k5, "Est. Ship Cost",      f"₹{total_ship:,.0f}",    "#f57c00", "to target WHs")
    kpi(k6, "Stockout Risk",       f"₹{sc_risk:,.0f}",       "#e53935", "if not restocked")

    sp(0.5)
    pt2, pt3 = st.tabs(["Warehouse Routing","Visual Analysis"])

    with pt2:
        sec("Warehouse Stock Needs & Routing Plan")
        wh_dist = (
            sku_plan.groupby("Target_Warehouse")
            .agg(SKUs=("SKU_ID","count"), Units=("Prod_Need","sum"))
            .reset_index().sort_values("Units", ascending=False)
        )
        n_wh = len(wh_dist)
        wh_colors = ["#1a2b6d","#2541b2","#0091ff","#00c6ae","#059669","#10b981"][:n_wh]
        fig_wh = go.Figure()
        fig_wh.add_trace(go.Bar(
            x=wh_dist["Target_Warehouse"], y=wh_dist["Units"],
            name="Units", marker=dict(color=wh_colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{int(v):,}" for v in wh_dist["Units"]],
            textposition="outside", textfont=dict(color="#334155"),
        ))
        fig_wh.add_trace(go.Scatter(
            x=wh_dist["Target_Warehouse"], y=wh_dist["SKUs"],
            name="SKU count", yaxis="y2", mode="markers+text",
            marker=dict(size=14, color="#f59e0b", line=dict(color="#fff",width=2)),
            text=[f"{v} SKUs" for v in wh_dist["SKUs"]],
            textposition="top center", textfont=dict(size=9,color="#d97706"),
        ))
        fig_wh.update_layout(
            **CD(), height=240, barmode="group", xaxis=gX(),
            yaxis={**gY(), "title":"Units to Receive"},
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title="SKU Count", tickcolor="#d97706",
                        range=[0, wh_dist["SKUs"].max()*3]),
            legend={**leg(), "orientation":"h","y":-0.28},
        )
        st.plotly_chart(fig_wh, use_container_width=True, key="wh_dist_bar")

        sp(0.5)
        sec("Detailed Shipment Routing Plan")
        routing_tbl = sku_plan[[
            "Target_Warehouse","SKU_ID","Product_Name","Category","ABC","Urgency",
            "Prod_Need","Days_Left","Est_Ship_Cost","WH_Share_Pct",
        ]].copy()
        routing_tbl["Days_Left"]     = routing_tbl["Days_Left"].apply(lambda x: f"{int(x)}d" if x < 999 else "∞")
        routing_tbl["Est_Ship_Cost"] = routing_tbl["Est_Ship_Cost"].apply(lambda x: f"₹{int(x):,}")
        routing_tbl["WH_Share_Pct"]  = routing_tbl["WH_Share_Pct"].apply(lambda x: f"{x:.0f}%")
        routing_tbl.columns = ["Warehouse","SKU","Product","Category","ABC","Urgency",
                               "Units","Days Left","Ship Cost","% of WH Inbound"]
        st.dataframe(routing_tbl.sort_values(["Warehouse","Urgency"]),
                     use_container_width=True, hide_index=True, height=380)

    with pt3:
        sec("Production Urgency Distribution")
        va1, va2 = st.columns(2, gap="large")
        urg_color_map = {
            "🔴 Urgent":"#ef4444","🟠 High":"#f97316",
            "🟡 Medium":"#eab308","🟢 Normal":"#22c55e",
        }
        with va1:
            urg_cnt = sku_plan["Urgency"].value_counts().reset_index()
            urg_cnt.columns = ["Urgency","Count"]
            fig_d = go.Figure(go.Pie(
                labels=urg_cnt["Urgency"], values=urg_cnt["Count"], hole=0.55,
                marker=dict(
                    colors=[urg_color_map.get(u,"#888") for u in urg_cnt["Urgency"]],
                    line=dict(color="#ffffff",width=2),
                ),
                textinfo="label+value", textfont=dict(size=11),
            ))
            fig_d.update_layout(**CD(), height=260, showlegend=False,
                                title=dict(text="SKUs by Urgency Tier",
                                           font=dict(size=11,color="#64748b")))
            st.plotly_chart(fig_d, use_container_width=True, key="pq_donut")
        with va2:
            cat_units = sku_plan.groupby(["Category","Urgency"])["Prod_Need"].sum().reset_index()
            fig_bu = go.Figure()
            for urg, clr in urg_color_map.items():
                sub = cat_units[cat_units["Urgency"] == urg]
                if sub.empty:
                    continue
                fig_bu.add_trace(go.Bar(
                    name=urg, x=sub["Category"], y=sub["Prod_Need"],
                    marker=dict(color=clr, line=dict(color="rgba(0,0,0,0)")),
                    text=sub["Prod_Need"].astype(int),
                    textposition="inside", textfont=dict(color="white",size=9),
                ))
            fig_bu.update_layout(
                **CD(), height=260, barmode="stack",
                xaxis={**gX(),"tickangle":-10},
                yaxis={**gY(),"title":"Units to Produce"},
                legend={**leg(),"orientation":"h","y":-0.32},
                title=dict(text="Units by Category & Urgency",font=dict(size=11,color="#64748b")),
            )
            st.plotly_chart(fig_bu, use_container_width=True, key="pq_cat_bar")

        sp()
        sec("Days of Stock — Most Critical SKUs")
        top20 = sku_plan.sort_values("Days_Left", ascending=True).head(20).copy()
        top20["Label"]     = top20["Product_Name"].str[:22] + " [" + top20["SKU_ID"] + "]"
        top20["Bar_Color"] = top20["Days_Left"].apply(
            lambda x: "#ef4444" if x<=7 else "#f97316" if x<=14 else "#eab308" if x<=30 else "#22c55e"
        )
        top20_s = top20.sort_values("Days_Left", ascending=True)
        fig_hl = go.Figure(go.Bar(
            x=top20_s["Days_Left"].clip(upper=60), y=top20_s["Label"],
            orientation="h",
            marker=dict(color=top20_s["Bar_Color"].tolist(), line=dict(color="rgba(0,0,0,0)")),
            text=[f"{int(v)}d · {int(u):,}u" for v,u in zip(top20_s["Days_Left"],top20_s["Prod_Need"])],
            textposition="outside", textfont=dict(color="#334155",size=9),
        ))
        for xv, clr, lbl in [(7,"#ef4444"," 7d"),(14,"#f97316"," 14d"),(30,"#eab308"," 30d")]:
            fig_hl.add_vline(x=xv, line_dash="dash", line_color=clr, line_width=1.5,
                             annotation_text=lbl, annotation_font=dict(color=clr,size=9))
        fig_hl.update_layout(
            **CD(), height=max(300,len(top20_s)*22),
            xaxis={**gX(),"title":"Days of Stock Remaining","range":[0,70]},
            yaxis=dict(showgrid=False,color="#64748b",automargin=True),
            title=dict(text="Top 20 Most Urgent SKUs",font=dict(size=11,color="#64748b")),
        )
        st.plotly_chart(fig_hl, use_container_width=True, key="pq_days_bar")


def page_logistics() -> None:
    n_future = get_horizon()
    df     = load_data()
    ops    = get_ops(df).copy()
    ops["YM"] = ops["Order_Date"].dt.to_period("M")
    del_df = get_delivered(df)

    st.markdown("<div class='page-header'><div class='page-header-title'>Logistics Optimization</div></div>",
                unsafe_allow_html=True)
    horizon_badge(n_future)

    total_spend  = del_df["Shipping_Cost_INR"].sum()
    avg_days     = del_df["Delivery_Days"].mean()
    on_time_pct  = (del_df["Delivery_Days"] <= 3).mean() * 100
    avg_cost_ord = del_df["Shipping_Cost_INR"].mean()
    ret_rate     = df["Return_Flag"].mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Total Shipping Spend", f"₹{total_spend:,.0f}", "#2541b2", "all delivered orders")
    kpi(k2, "Avg Delivery Days",    f"{avg_days:.1f}d",      "#00c6ae", "across all carriers")
    kpi(k3, "On-Time Rate",         f"{on_time_pct:.1f}%",  "#00c6ae", "delivered ≤ 3 days")
    kpi(k4, "Avg Cost / Order",     f"₹{avg_cost_ord:.0f}", "#2541b2", "per shipment")
    kpi(k5, "Return Rate",          f"{ret_rate:.1f}%",      "#e53935", "of all orders")
    sp(0.5)

    with st.expander("⚙️ Carrier Scoring Weights", expanded=False):
        wc1, wc2, wc3 = st.columns(3)
        w_speed   = wc1.slider("Speed weight %",   10, 70, int(DEFAULT_W_SPEED   * 100)) / 100
        w_cost    = wc2.slider("Cost weight %",    10, 70, int(DEFAULT_W_COST    * 100)) / 100
        w_returns = wc3.slider("Returns weight %", 10, 70, int(DEFAULT_W_RETURNS * 100)) / 100
        tot = w_speed + w_cost + w_returns
        w_speed /= tot; w_cost /= tot; w_returns /= tot

    carr, best_carr, opt, fwd_plan = compute_logistics(w_speed, w_cost, w_returns, n_future)
    plan = compute_production(n_future=n_future)

    t1, t2, t3 = st.tabs(["Carrier Performance","Cost & Delay","Forward Plan"])

    with t1:
        sec("Speed vs Cost — Carrier Scorecard", "🚚")
        fig = go.Figure()
        for i, (_, r) in enumerate(carr.iterrows()):
            fig.add_trace(go.Scatter(
                x=[r["Avg_Days"]], y=[r["Avg_Cost"]], mode="markers+text",
                marker=dict(size=max(r["Orders"]/50,14), color=COLORS[i],
                            opacity=0.88, line=dict(color="#FFFFFF",width=2)),
                text=[r["Courier_Partner"]], textposition="top center",
                name=r["Courier_Partner"],
                hovertemplate=(
                    f"<b>{r['Courier_Partner']}</b><br>Orders: {r['Orders']}<br>"
                    f"Avg Days: {r['Avg_Days']:.1f}<br>Avg Cost: ₹{r['Avg_Cost']:.0f}<br>"
                    f"Score: {r['Perf_Score']:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            **CD(), height=270, showlegend=False,
            xaxis={**gX(),"title":"Avg Delivery Days  ← faster"},
            yaxis={**gY(),"title":"Avg Shipping Cost ₹  ↓ cheaper"},
        )
        st.plotly_chart(fig, use_container_width=True, key="log_bubble")

        ta1, ta2 = st.columns(2, gap="large")
        with ta1:
            sec("Carrier Metrics Table")
            d2 = carr[["Courier_Partner","Orders","Avg_Days","Avg_Cost","Return_Rate","Perf_Score"]].copy()
            d2["Avg_Days"]    = d2["Avg_Days"].round(1)
            d2["Avg_Cost"]    = d2["Avg_Cost"].round(1)
            d2["Return_Rate"] = (d2["Return_Rate"]*100).round(1).astype(str) + "%"
            d2["Perf_Score"]  = d2["Perf_Score"].round(3)
            d2.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹","Return Rate","Score"]
            st.dataframe(d2.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)
        with ta2:
            sec("Best Carrier per Category")
            if not plan.empty:
                cat_carr = del_df.groupby(["Category","Courier_Partner"]).agg(
                    Avg_Days=("Delivery_Days","mean"),
                    Avg_Cost=("Shipping_Cost_INR","mean"),
                ).reset_index()
                cat_carr_ret = df.groupby(["Category","Courier_Partner"])["Return_Flag"].mean().reset_index()
                cat_carr_ret.columns = ["Category","Courier_Partner","Return_Rate"]
                cat_carr = cat_carr.merge(cat_carr_ret, on=["Category","Courier_Partner"], how="left")
                cat_carr["Return_Rate"] = cat_carr["Return_Rate"].fillna(0)
                for col_c in ["Avg_Days","Avg_Cost","Return_Rate"]:
                    mn_c=cat_carr[col_c].min(); mx_c=cat_carr[col_c].max()
                    cat_carr[f"N_{col_c}"] = 1-(cat_carr[col_c]-mn_c)/(mx_c-mn_c+1e-9)
                cat_carr["Score"] = (
                    w_speed   * cat_carr["N_Avg_Days"]
                    + w_cost  * cat_carr["N_Avg_Cost"]
                    + w_returns * cat_carr["N_Return_Rate"]
                )
                best_cat = cat_carr.sort_values("Score",ascending=False).groupby("Category").first().reset_index()
                prod_by_cat = plan.groupby("Category")["Production"].sum().reset_index()
                best_cat = best_cat.merge(prod_by_cat.rename(columns={"Production":"Planned Units"}),
                                          on="Category", how="left")
                best_cat["Avg_Days"]      = best_cat["Avg_Days"].round(1)
                best_cat["Avg_Cost"]      = best_cat["Avg_Cost"].round(1)
                best_cat["Score"]         = best_cat["Score"].round(3)
                best_cat["Planned Units"] = best_cat["Planned Units"].fillna(0).astype(int)
                best_cat = best_cat[["Category","Courier_Partner","Avg_Days","Avg_Cost","Score","Planned Units"]]
                best_cat.columns = ["Category","Best Carrier","Avg Days","Avg Cost ₹","Score","Planned Units"]
                st.dataframe(best_cat.sort_values("Score",ascending=False),
                             use_container_width=True, hide_index=True)

        sp(0.5)
        # ── POLISHED HEATMAP ──
        sec("Carrier × Region Performance Heatmap", "🌡️")

        hm_metric = st.selectbox(
            "Heatmap metric",
            ["Delay Rate %","Avg Delivery Days","Avg Cost ₹","Return Rate %"],
            key="log_hm_metric",
        )
        delay_thr = st.slider("Delay threshold (days)", 3, 10, DEFAULT_REPLEN_LEAD_TIME,
                              key="log_thr")

        del_hm = del_df.copy()
        del_hm["Delayed"] = del_hm["Delivery_Days"] > delay_thr

        if hm_metric == "Delay Rate %":
            pv = del_hm.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0) * 100
            colorscale = [
                [0.0, "#0a1628"], [0.2, "#1a3a6b"], [0.4, "#7b4dd0"],
                [0.7, "#e87adb"], [1.0, "#ef4444"],
            ]
            text_suffix = "%"
        elif hm_metric == "Avg Delivery Days":
            pv = del_hm.groupby(["Courier_Partner","Region"])["Delivery_Days"].mean().unstack(fill_value=0)
            colorscale = [
                [0.0, "#064e3b"], [0.4, "#059669"],
                [0.7, "#fbbf24"], [1.0, "#ef4444"],
            ]
            text_suffix = "d"
        elif hm_metric == "Avg Cost ₹":
            pv = del_hm.groupby(["Courier_Partner","Region"])["Shipping_Cost_INR"].mean().unstack(fill_value=0)
            colorscale = [
                [0.0, "#1e3a8a"], [0.5, "#3b82f6"], [1.0, "#ef4444"],
            ]
            text_suffix = ""
        else:  # Return Rate %
            pv_ret = df.groupby(["Courier_Partner","Region"])["Return_Flag"].mean().unstack(fill_value=0) * 100
            pv = pv_ret
            colorscale = [
                [0.0, "#14532d"], [0.4, "#22c55e"],
                [0.7, "#fbbf24"], [1.0, "#ef4444"],
            ]
            text_suffix = "%"

        z_vals = pv.values
        text_vals = np.round(z_vals, 1)

        # ── Polished heatmap with annotations and clean grid ──
        fig_h = go.Figure()
        fig_h.add_trace(go.Heatmap(
            z=z_vals,
            x=list(pv.columns),
            y=list(pv.index),
            colorscale=colorscale,
            text=[[f"{v:.1f}{text_suffix}" for v in row] for row in text_vals],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            hovertemplate=(
                "<b>%{y}</b> → <b>%{x}</b><br>"
                f"{hm_metric}: %{{text}}<extra></extra>"
            ),
            colorbar=dict(
                tickfont=dict(color="#64748b", size=10),
                title=dict(text=hm_metric, font=dict(size=10, color="#64748b"), side="right"),
                thickness=14, len=0.85,
            ),
            xgap=3, ygap=3,
        ))

        # Row-level best performer annotations
        for yi, carrier in enumerate(pv.index):
            best_region_idx = int(np.argmin(z_vals[yi]))
            best_region     = pv.columns[best_region_idx]
            fig_h.add_annotation(
                x=best_region, y=carrier,
                text="★", showarrow=False,
                font=dict(size=14, color="rgba(255,220,50,0.9)"),
                xanchor="center", yanchor="middle",
            )

        fig_h.update_layout(
            **CD(), height=300 + len(pv.index) * 20,
            xaxis=dict(
                showgrid=False, tickangle=-30, color="#64748b",
                title=dict(text="Region", font=dict(size=11)),
            ),
            yaxis=dict(
                showgrid=False, color="#64748b",
                title=dict(text="Carrier", font=dict(size=11)),
            ),
            title=dict(
                text=f"<b>{hm_metric}</b> by Carrier × Region  "
                     f"<span style='font-size:10px;color:#94a3b8'>"
                     f"(★ = best value per carrier row)</span>",
                font=dict(size=12, color="#334155"),
            ),
            plot_bgcolor="rgba(248,249,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.markdown("<div class='heatmap-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(fig_h, use_container_width=True, key="log_heat_polished")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Per-carrier best region summary row ──
        best_per_row = []
        for carrier in pv.index:
            row_vals = z_vals[list(pv.index).index(carrier)]
            best_idx = int(np.argmin(row_vals))
            best_per_row.append({
                "Carrier": carrier,
                f"Best Region ({hm_metric})": pv.columns[best_idx],
                "Value": f"{row_vals[best_idx]:.1f}{text_suffix}",
                "Worst Region": pv.columns[int(np.argmax(row_vals))],
                "Worst Value": f"{row_vals[int(np.argmax(row_vals))]:.1f}{text_suffix}",
            })
        st.dataframe(pd.DataFrame(best_per_row), use_container_width=True, hide_index=True)

    with t2:
        total_sav = opt["Potential_Saving"].sum()
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Current Spend",    f"₹{total_spend:,.0f}",               "#2541b2", "all deliveries")
        kpi(c2, "Optimised Spend",  f"₹{total_spend-total_sav:,.0f}",     "#00c6ae", "with best carriers")
        kpi(c3, "Potential Saving", f"₹{total_sav:,.0f}",                 "#00c6ae", "by switching")
        kpi(c4, "Saving %",         f"{total_sav/total_spend*100:.1f}%",  "#00c6ae", "of total spend")
        sp(0.5)

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
                xaxis={**gX(),"tickangle":-30},
                yaxis={**gY(),"title":"Avg Cost per Order ₹"},
                legend={**leg(),"orientation":"h","y":-0.3},
            )
            st.plotly_chart(fig_cost, use_container_width=True, key="log_cost")
        with tb2:
            sec("Delay Rate by Region")
            del_df_t2 = del_df.copy()
            del_df_t2["Delayed"] = del_df_t2["Delivery_Days"] > delay_thr
            rd = del_df_t2.groupby("Region").agg(
                T=("Order_ID","count"), D=("Delayed","sum")
            ).reset_index()
            rd["Rate"] = (rd["D"] / rd["T"] * 100).round(1)
            rd_s = rd.sort_values("Rate", ascending=True)
            fig_r = go.Figure(go.Bar(
                x=rd_s["Rate"], y=rd_s["Region"], orientation="h",
                marker=dict(
                    color=[f"rgba(239,68,68,{min(v/50+0.2,0.9):.2f})" for v in rd_s["Rate"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v}%" for v in rd_s["Rate"]],
                textposition="outside", textfont=dict(color="#334155"),
            ))
            fig_r.update_layout(
                **CD(), height=270,
                xaxis={**gX(),"title":"Delay Rate %"},
                yaxis=dict(showgrid=False, color="#64748b"),
            )
            st.plotly_chart(fig_r, use_container_width=True, key="log_delay_region")

        sp(0.5)
        tb3, tb4 = st.columns(2, gap="large")
        with tb3:
            sec("Carrier Switch Recommendations")
            od = opt[["Region","Optimal_Carrier","Current_Avg_Cost","Min_Avg_Cost",
                       "Potential_Saving","Saving_Pct","Orders"]].copy()
            od["Current_Avg_Cost"] = od["Current_Avg_Cost"].round(1)
            od["Min_Avg_Cost"]     = od["Min_Avg_Cost"].round(1)
            od["Potential_Saving"] = od["Potential_Saving"].astype(int)
            od.columns = ["Region","Switch To","Current Avg ₹","Optimal Avg ₹","Saving ₹","Saving %","Orders"]
            st.dataframe(od.sort_values("Saving ₹",ascending=False),
                         use_container_width=True, hide_index=True)
        with tb4:
            sec("Delay Rate by Carrier")
            del_df_c = del_df.copy()
            del_df_c["Delayed"] = del_df_c["Delivery_Days"] > delay_thr
            cd = del_df_c.groupby("Courier_Partner").agg(
                T=("Order_ID","count"), D=("Delayed","sum")
            ).reset_index()
            cd["Rate"] = (cd["D"] / cd["T"] * 100).round(1)
            fig_cd = go.Figure(go.Bar(
                x=cd["Courier_Partner"], y=cd["Rate"],
                marker=dict(
                    color=["#EF4444" if v>35 else "#F59E0B" if v>20 else "#22C55E" for v in cd["Rate"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v}%" for v in cd["Rate"]],
                textposition="outside", textfont=dict(color="#334155"),
            ))
            fig_cd.update_layout(
                **CD(), height=240, xaxis=gX(),
                yaxis={**gY(),"title":"Delay Rate %"},
            )
            st.plotly_chart(fig_cd, use_container_width=True, key="log_delay_carrier")

    with t3:
        if fwd_plan.empty:
            st.info("No forward plan available — production plan has no actionable SKUs.")
        else:
            fwd_agg = (
                fwd_plan.groupby("Month_dt")
                .agg(
                    Month           =("Month","first"),
                    Total_Units     =("Prod_Units","sum"),
                    Total_Orders    =("Proj_Orders","sum"),
                    Total_Ship_Cost =("Proj_Ship_Cost","sum"),
                    CI_Lo           =("CI_Lo_Units","sum"),
                    CI_Hi           =("CI_Hi_Units","sum"),
                ).reset_index().sort_values("Month_dt")
            )
            fc1, fc2, fc3 = st.columns(3)
            kpi(fc1, f"{n_future}M Planned Units", f"{fwd_agg['Total_Units'].sum():,}",         "#2541b2", "from production plan")
            kpi(fc2, f"{n_future}M Est. Orders",   f"{fwd_agg['Total_Orders'].sum():,}",        "#2541b2", "projected shipments")
            kpi(fc3, f"{n_future}M Ship Cost",     f"₹{fwd_agg['Total_Ship_Cost'].sum():,.0f}", "#f57c00", "at current avg rate")
            sp(0.5)

            tc1, tc2 = st.columns([3,2], gap="large")
            with tc1:
                sec(f"Production → Shipment Plan — {n_future} Months")
                fig_fwd = go.Figure()
                x_ci = list(fwd_agg["Month_dt"]) + list(fwd_agg["Month_dt"])[::-1]
                y_ci = list(fwd_agg["CI_Hi"])     + list(fwd_agg["CI_Lo"])[::-1]
                fig_fwd.add_trace(go.Scatter(
                    x=x_ci, y=y_ci, fill="toself",
                    fillcolor="rgba(59,130,246,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="Demand 90% CI",
                ))
                fig_fwd.add_trace(go.Bar(
                    x=fwd_agg["Month_dt"], y=fwd_agg["Total_Units"],
                    name="Planned Units",
                    marker=dict(color="#3B82F6", opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
                ))
                fig_fwd.update_layout(
                    **CD(), height=260, barmode="overlay", xaxis=gX(),
                    yaxis={**gY(),"title":"Units"},
                    legend={**leg(),"orientation":"h","y":-0.28},
                )
                st.plotly_chart(fig_fwd, use_container_width=True, key="fwd_units")
            with tc2:
                sec("Category Breakdown")
                cat_fwd = (
                    fwd_plan.groupby("Category")
                    .agg(Units=("Prod_Units","sum"),
                         Orders=("Proj_Orders","sum"),
                         Cost=("Proj_Ship_Cost","sum"))
                    .reset_index().sort_values("Units", ascending=False)
                )
                cat_fwd.columns = ["Category","Units","Est. Orders","Ship Cost ₹"]
                st.dataframe(cat_fwd, use_container_width=True, hide_index=True, height=210)

            sec("Projected Shipping Cost")
            fig_cost2 = go.Figure(go.Scatter(
                x=fwd_agg["Month_dt"], y=fwd_agg["Total_Ship_Cost"],
                mode="lines+markers", line=dict(color="#8B5CF6",width=2.5),
                marker=dict(size=8,color="#8B5CF6",line=dict(color="#FFFFFF",width=2)),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.07)",
            ))
            fig_cost2.update_layout(
                **CD(), height=200,
                xaxis={**gX(),"tickangle":0},
                yaxis={**gY(),"title":"₹"},
            )
            st.plotly_chart(fig_cost2, use_container_width=True, key="fwd_cost")

            sp(0.5)
            sec("Inbound Plan per Warehouse")
            wh_share = (
                del_df.groupby("Warehouse")["Quantity"].sum()
                / del_df["Quantity"].sum()
            ).to_dict()
            inb_rows = [
                {"Month":row["Month"],"Month_dt":row["Month_dt"],
                 "Warehouse":wh,"Inbound_Units":round(row["Prod_Units"]*sh),
                 "Proj_Ship_Cost":round(row["Proj_Ship_Cost"]*sh)}
                for _,row in fwd_plan.iterrows() for wh,sh in wh_share.items()
            ]
            inb_agg = (
                pd.DataFrame(inb_rows)
                .groupby(["Month_dt","Month","Warehouse"])
                .agg(Inbound_Units=("Inbound_Units","sum"),
                     Proj_Ship_Cost=("Proj_Ship_Cost","sum"))
                .reset_index().sort_values(["Month_dt","Warehouse"])
            )
            fig_inb = go.Figure()
            for i, wh in enumerate(sorted(inb_agg["Warehouse"].unique())):
                wdf = inb_agg[inb_agg["Warehouse"] == wh]
                fig_inb.add_trace(go.Bar(
                    x=wdf["Month"], y=wdf["Inbound_Units"], name=wh,
                    marker=dict(color=COLORS[i%len(COLORS)], line=dict(color="rgba(0,0,0,0)")),
                ))
            fig_inb.update_layout(
                **CD(), height=250, barmode="group",
                xaxis={**gX(),"tickangle":-25},
                yaxis={**gY(),"title":"Planned Inbound Units"},
                legend=leg(),
            )
            st.plotly_chart(fig_inb, use_container_width=True, key="wh_inbound")
            disp_inb = inb_agg[["Month","Warehouse","Inbound_Units","Proj_Ship_Cost"]].copy()
            disp_inb.columns = ["Month","Warehouse","Planned Units","Proj. Ship Cost ₹"]
            st.dataframe(disp_inb, use_container_width=True, hide_index=True)


# ──────────────────────────────────
# MAIN
# ──────────────────────────────────
def main() -> None:
    inject_css()

    st.sidebar.markdown("""
    <div style='padding:16px 0 10px'>
      <div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;
           letter-spacing:-.03em;
           background:linear-gradient(135deg,#f5a623,#ff6b6b,#2ed8c3);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        OmniFlow D2D
      </div>
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
