# ======================================================================================
# OmniFlow-D2D : Demand Forecasting Intelligence Module
# ======================================================================================
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import holidays
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings("ignore")
india_holidays = holidays.India()
if "demand_products" not in st.session_state:
    st.session_state["demand_products"] = []
if "selected_product" not in st.session_state:
    st.session_state["selected_product"] = None
if "forecast_range" not in st.session_state:
    st.session_state["forecast_range"] = None
st.set_page_config(page_title="Demand Forecasting Intelligence", layout="wide")
def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg: #f8fafc;
        --text: #0f172a;
        --muted: #475569;
        --primary: #1e3a8a;
        --border: #e5e7eb;
        --accent: #e0e7ff;
    }
    html, body {
        background-color: var(--bg);
        color: var(--text);
        font-family: Inter, system-ui;
    }
    section.main > div {
        animation: fadeIn 0.4s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 28px 0 14px 0;
    }
    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transition: all 0.25s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.14);
    }
    .metric-card {
        background: linear-gradient(180deg, #eef4ff, #ffffff);
        padding: 18px;
        text-align: center;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(30,58,138,0.18);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 36px rgba(30,58,138,0.28);
    }
    .metric-label {
        font-size: 14px;
        color: var(--muted);
    }
    .metric-value {
        font-size: 30px;
        font-weight: 900;
        color: var(--primary);
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 600;
        color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent);
        color: var(--primary);
        box-shadow: 0 6px 18px rgba(30,58,138,0.25);
    }
    </style>
    """, unsafe_allow_html=True)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SALES_PATH = os.path.join(DATA_DIR, "retail_sales_transactions.csv")
PRODUCT_PATH = os.path.join(DATA_DIR, "fmcg_product_master.csv")
INVENTORY_PATH = os.path.join(DATA_DIR, "retail_inventory_snapshot.csv")
LOGISTICS_PATH = os.path.join(DATA_DIR, "supply_chain_logistics_shipments.csv")
FORECAST_PATH = os.path.join(DATA_DIR, "forecast_output.csv")
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "date","product_id","region","daily_sales","price",
        "promotion","holiday_flag","avg_stock","avg_delay_rate",
        "dayofweek","month","dayofyear","weekofyear",
        "lag_1","lag_7","rolling_7","rolling_14","rolling_30",
        "lag_365","forecast","lower_ci","upper_ci"
    ],
    "Description": [
        "Transaction date","Unique product identifier","Sales region",
        "Units sold per day (target)","Unit selling price","Promotion flag",
        "Holiday flag","Average stock level","Average logistics delay rate",
        "Day of week feature","Month feature","Day of year feature",
        "ISO week number","Sales lag 1 day","Sales lag 7 days",
        "7-day rolling mean","14-day rolling mean","30-day rolling mean",
        "Yearly seasonal lag","Forecasted demand",
        "Lower confidence bound","Upper confidence bound"
    ]
})
@st.cache_data
def load_tables():
    for f in [SALES_PATH, PRODUCT_PATH, INVENTORY_PATH, LOGISTICS_PATH]:
        if not os.path.exists(f):
            st.error(f"Missing required file: {os.path.basename(f)}")
            st.stop()
    sales = pd.read_csv(SALES_PATH)
    products = pd.read_csv(PRODUCT_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    logistics = pd.read_csv(LOGISTICS_PATH)
    for df in [sales, products, inventory, logistics]:
        df.columns = df.columns.str.lower().str.strip()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    return sales, products, inventory, logistics
def build_demand_dataset():
    sales, products, inventory, logistics = load_tables()
    df = sales.merge(
        products[[
         "product_id","category","brand","unit_weight_kg","shelf_life_days","mrp"
        ]],on="product_id", how="left"
    )
    inv_agg = (
        inventory.groupby("product_id", as_index=False).agg(avg_stock=("on_hand_qty","mean"))
    )
    df = df.merge(inv_agg, on="product_id", how="left")
    log_agg = (logistics.groupby("destination_region", as_index=False)
        .agg(avg_delay_rate=("delay_flag","mean"))
        .rename(columns={"destination_region":"region"})
    )
    df = df.merge(log_agg, on="region", how="left")
    return df
def prepare_features(df):
    df = df.rename(columns={
        "units_sold":"daily_sales",
        "selling_price":"price"
    })   
    df = df[(df.daily_sales >= 0) & (df.price > 0)]
    q_low  = df["daily_sales"].quantile(0.01)
    q_high = df["daily_sales"].quantile(0.99)
    df["daily_sales"] = df["daily_sales"].clip(q_low, q_high)
    df["promotion"] = df["promotion_flag"].fillna(0)
    df["holiday_flag"] = df["holiday_flag"].fillna(0)
    df["avg_delay_rate"] = df["avg_delay_rate"].fillna(0)
    df["avg_stock"] = df["avg_stock"].fillna(df["avg_stock"].median())
    df["region"] = df["region"].astype(str).str.upper().str.strip()
    df = df.sort_values(["product_id","date"]).reset_index(drop=True)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["sin_dow"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["cos_dow"] = np.cos(2*np.pi*df["dayofweek"]/7)    
    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)
    df["lag_1"]  = df.groupby("product_id")["daily_sales"].shift(1)
    df["lag_7"]  = df.groupby("product_id")["daily_sales"].shift(7)
    df["rolling_7"] = (
        df.groupby("product_id")["daily_sales"]
        .rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["rolling_14"] = (
       df.groupby("product_id")["daily_sales"]
       .rolling(14).mean().reset_index(level=0, drop=True)
    )   
    df["rolling_30"] = (
       df.groupby("product_id")["daily_sales"]
       .rolling(30).mean().reset_index(level=0, drop=True)
    )
    df["lag_365"] = df.groupby("product_id")["daily_sales"].shift(365)    
    df["lag_365"] = df["lag_365"].fillna(df["rolling_30"])
    df["promo_effect"] = df["rolling_7"] * df["promotion"]
    df["lag_diff_1"] = df["daily_sales"] - df["lag_1"]
    df["lag_ratio_1"] = df["daily_sales"] / (df["lag_1"] + 1)
    df["trend_7"] = df["rolling_7"] - df["lag_7"]
    df["price_change"] = df.groupby("product_id")["price"].pct_change().fillna(0)
    df["promo_rolling_7"] = df.groupby("product_id")["promotion"].rolling(7).mean().reset_index(0,drop=True)
    lag_cols = ["lag_1","lag_7","rolling_7","rolling_14","rolling_30"]  
    for col in lag_cols:
        df[col] = df.groupby("product_id")[col]\
        .transform(lambda x: x.fillna(x.mean()))   
    return df
def data_profiling(df):
    return {
        "Total Records": len(df),
        "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "Unique Products": df["product_id"].nunique(),
        "Average Daily Sales": round(df["daily_sales"].mean(), 2),
        "Sales Volatility (Std)": round(df["daily_sales"].std(), 2),
        "Promotion Share (%)": round(df["promotion"].mean() * 100, 2),
        "Date Sorted": df["date"].is_monotonic_increasing,
        "Missing Values": int(df.isnull().sum().sum())
    }
def train_models(X_train, y_train_log, X_test, y_test_log):
    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
             n_estimators=600,learning_rate=0.03,max_depth=5,
             min_samples_leaf=3,subsample=0.8,random_state=42
         )
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    params = {
        "n_estimators": [300, 500, 700],
        "max_depth": [12, 16, 20, None],
        "min_samples_leaf": [1, 2, 3]
    }
    search = RandomizedSearchCV(
        rf,params,n_iter=5,cv=3,scoring="neg_mean_squared_error",n_jobs=-1
    )
    search.fit(X_train, y_train_log)
    models["Random Forest"] = search.best_estimator_
    results = []
    forecasts = {}
    y_true = np.expm1(y_test_log)
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        forecasts[name] = preds
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        range_y = y_true.max() - y_true.min()
        nrmse = rmse / (range_y + 1e-6)
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "R2": r2
        })
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model = results_df.iloc[0]["Model"]
    return results_df, forecasts, best_model, search.best_estimator_
def demand_nlp(df, results_df, best_model, q, top_product_global=None):
    q = q.lower().strip()
    if any(x in q for x in ["average demand", "avg demand", "mean demand"]):
        return f"Average forecasted demand is {df['forecast'].mean():.2f} units."    
    if any(x in q for x in [
        "maximum demand product", "highest demand product",
        "top demand product", "most demanded product"
    ]):
        prod_demand = (df.groupby("product_id")["forecast"].mean().sort_values(ascending=False))
        top_product = prod_demand.index[0]
        top_value = prod_demand.iloc[0]   
        return (
            f"The product with the highest average forecasted demand is "
            f"**{top_product}**, with approximately **{top_value:.0f} units**."
        )    
    if any(x in q for x in [
        "minimum demand product", "lowest demand product","least demanded product"
    ]):
        prod_demand = (df.groupby("product_id")["forecast"].mean().sort_values())
        low_product = prod_demand.index[0]
        low_value = prod_demand.iloc[0]    
        return (
            f"The product with the lowest average forecasted demand is "
            f"**{low_product}**, with approximately **{low_value:.0f} units**."
        ) 
    if any(x in q for x in ["maximum demand", "max demand"]) and "product" not in q:
        return f"Maximum forecasted demand value is {df['forecast'].max():.0f} units."
    if any(x in q for x in ["minimum demand", "min demand", "lowest demand"]):
        return f"Minimum forecasted demand is {df['forecast'].min():.0f} units."
    if any(x in q for x in ["volatility", "variation", "unstable"]):
        vol = df["forecast"].std()
        return (
            f"Demand volatility (standard deviation) is {vol:.2f}. "
            "High volatility indicates unstable demand; higher safety stock is recommended."
        )
    if any(x in q for x in ["trend", "direction", "growth", "decline"]):
        slope = np.polyfit(range(len(df)), df["forecast"], 1)[0]
        if slope > 0:
            return "Demand shows an upward trend over the forecast horizon."
        elif slope < 0:
            return "Demand shows a downward trend over the forecast horizon."
        else:
            return "Demand appears relatively stable with no strong trend."
    if any(x in q for x in ["best model", "which model", "chosen model"]):
        return f"The best performing model is **{best_model}**, selected based on lowest RMSE."
    if "rmse" in q:
        return f"The RMSE of the best model is {results_df.iloc[0]['RMSE']:.2f}."
    if "mae" in q:
        return f"The MAE of the best model is {results_df.iloc[0]['MAE']:.2f}."
    if any(x in q for x in ["compare models", "model comparison"]):
        comparison = results_df[["Model", "RMSE"]].to_string(index=False)
        return f"Model comparison based on RMSE:\n{comparison}"
    if any(x in q for x in ["why this model", "why best model"]):
        return (
            f"{best_model} was selected because it achieved the lowest RMSE, "
            "indicating better accuracy in predicting demand compared to other models."
        )
    if any(x in q for x in ["stockout", "inventory risk", "risk"]):
        if df["forecast"].std() > df["forecast"].mean() * 0.3:
            return (
                "There is a high stockout risk due to volatile demand. "
                "Increasing safety stock or improving replenishment frequency is recommended."
            )
        return "Inventory risk appears manageable under current demand patterns."
    if any(x in q for x in ["safety stock"]):
        return (
            "Safety stock should be increased when demand volatility is high "
            "or when logistics delays are frequent."
        )
    if any(x in q for x in ["delay", "logistics", "delivery"]):
        if "avg_delay_rate" in df.columns and df["avg_delay_rate"].mean() > 0.3:
            return (
                "High logistics delay rates detected. Delays may amplify demand uncertainty "
                "and should be factored into inventory planning."
            )
        return "Logistics performance appears stable with limited impact on demand."
    if any(x in q for x in ["summary", "overall", "insight"]):
        return (
            f"Demand is forecasted using {best_model} with moderate volatility. "
            "Inventory planning should account for variability, "
            "and logistics delays should be monitored to avoid service disruptions."
        )
    if any(x in q for x in ["recommend", "suggest", "action"]):
        return (
            "Recommended actions:\n"
            "- Use forecast output for dynamic inventory planning\n"
            "- Increase safety stock for high-volatility products\n"
            "- Monitor logistics delays in high-risk regions\n"
            "- Re-train models periodically with new data"
        )
    if any(x in q for x in ["top 3 products", "top products", "top 5 products"]):
        top_n = 3 if "3" in q else 5
        top_products = (df.groupby("product_id")["forecast"].mean().sort_values(ascending=False).head(top_n))
        return (
            f"Top {top_n} products by average forecasted demand:\n" + top_products.to_string()
        )
    if "last month" in q:
        last_month = df["date"].max() - pd.DateOffset(months=1)
        avg_last_month = df[df["date"] >= last_month]["forecast"].mean()
        return f"Average demand in the last month was {avg_last_month:.2f} units."   
    if "next week" in q:
        return "This forecast operates on daily granularity. Weekly aggregation can be added in the planning module."
    if "most demand product" in q and top_product_global:
        return (
            f"The most demanded product is **{top_product_global}**, "
            "based on highest average sales."
        )   
    if "kpi" in q:
        return (
            f"Key KPIs:\n"
            f"- Best Model: {best_model}\n"
            f"- Avg Forecast: {df['forecast'].mean():.2f}\n"
            f"- Volatility: {df['forecast'].std():.2f}"
        )
    return (
        "I can help with:\n"
        "- average / max / min demand\n"
        "- demand trend or volatility\n"
        "- best model and RMSE\n"
        "- model comparison\n"
        "- inventory risk or safety stock\n"
        "- logistics delay impact\n"
        "- executive summary or recommendations"
    )
def fit_final_model(model, X, y):
    model.fit(X.astype(float), y.astype(float))
    return model
def demand_forecasting_page():
    inject_css()
    tab1, tab2 = st.tabs(["📘 Overview", "📊 Application"])
    with tab1:
        st.markdown('<div class="section-title">Demand Forecasting Module – Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        This module predicts future product demand using historical sales data enriched with
        inventory and logistics context. It serves as the intelligence backbone of OmniFlow-D2D.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-title">Objectives</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul>
            <li>Accurately forecast short-term and medium-term demand</li>
            <li>Reduce overstocking and stockout risks</li>
            <li>Enable proactive inventory and production planning</li>
            <li>Support data-driven managerial decisions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul>
            <li>Sales data enriched with product, inventory, and logistics features</li>
            <li>Lag and rolling-window features capture temporal behavior</li>
            <li>Multiple ML models trained and evaluated</li>
            <li>Best model selected using RMSE</li>
            <li>Confidence intervals quantify forecast uncertainty</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Outputs</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul>
            <li>Product-level demand forecasts</li>
            <li>Forecast confidence intervals</li>
            <li>Model performance metrics</li>
            <li>Executive KPIs</li>
            <li>AI-assisted insights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        raw_df = build_demand_dataset()
        raw_df = raw_df.sort_values(["product_id","date"])
        product_mode = st.radio(
            "Product View Mode",["Single Product", "Multiple Products"],
            horizontal=True, key="product_mode"
        )     
        product_list = sorted(raw_df["product_id"].unique())
        if product_mode == "Single Product":   
            if ("selected_product" not in st.session_state
                or st.session_state.selected_product not in product_list
            ):
                st.session_state.selected_product = product_list[0]
            product = st.selectbox("Select Product", product_list,
                index=product_list.index(st.session_state.selected_product),
            ) 
            st.session_state.selected_product = product
            df_selected = raw_df[raw_df["product_id"] == product].copy() 
        else:
            selected_products = st.multiselect("Select Products",product_list,
                default=st.session_state.get("demand_products", [])
            )
            st.session_state["demand_products"] = selected_products         
            if len(selected_products) == 0:
                df_selected = raw_df.copy()
            else:
                df_selected = raw_df[raw_df["product_id"].isin(selected_products)].copy()
        if st.button("🔄 Reset Product Selection"):
            st.session_state["demand_products"] = []
            if len(product_list) > 0:
                st.session_state["selected_product"] = product_list[0]
            st.rerun()
        if len(df_selected) < 15:
            st.warning("⚠️ This product has limited history. Forecast accuracy may be lower.")
        df = prepare_features(df_selected)
        vol = df.groupby("product_id")["daily_sales"].std()
        mean_sales = df.groupby("product_id")["daily_sales"].mean()     
        difficulty = (vol / (mean_sales + 1e-6)).rename("sku_difficulty")       
        df = df.merge(difficulty,left_on="product_id",right_index=True,how="left")      
        df["sku_difficulty"] = df["sku_difficulty"].fillna(0)
        df = df.sort_values("date").reset_index(drop=True)
        le = LabelEncoder()
        df["region_encoded"] = le.fit_transform(df["region"].astype(str))
        df["demand_regime"] = np.where(df["rolling_30"] > df["rolling_7"] * 1.2,"Growing",
            np.where(df["rolling_30"] < df["rolling_7"] * 0.8,"Declining","Stable")
        )  
        min_date = df["date"].min()
        max_date = df["date"].max()       
        FEATURES = [
            "price","promotion","holiday_flag",
            "region_encoded","avg_stock","avg_delay_rate",
            "dayofweek","is_weekend","month","dayofyear","weekofyear",
            "lag_1","lag_7","rolling_7","rolling_14",
            "rolling_30","lag_365","lag_diff_1","lag_ratio_1","trend_7",
            "price_change","promo_rolling_7","promo_effect",
            "sin_dow","cos_dow","sin_month","cos_month","sku_difficulty"
        ]       
        df = df.sort_values("date").reset_index(drop=True)
        split_idx = max(int(len(df) * 0.8), len(df) - 14)
        if len(df) - split_idx < 5:
            split_idx = len(df) - 5        
        train_df = df.iloc[:split_idx].copy()
        val_df   = df.iloc[split_idx:].copy()
        X_train = train_df[FEATURES].copy()
        X_val   = val_df[FEATURES].copy()        
        y_train = train_df["daily_sales"].copy()
        y_val   = val_df["daily_sales"].copy()   
        X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)        
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0)
        y_val   = pd.to_numeric(y_val, errors="coerce").fillna(0)
        y_train = y_train.clip(lower=0)
        y_val   = y_val.clip(lower=0)
        y_train_log = np.log1p(y_train)
        y_val_log   = np.log1p(y_val)
        st.markdown(
            '<div class="section-title">Seasonality Heatmap</div>',unsafe_allow_html=True
        ) 
        train_df = train_df.dropna(subset=["date"])
        season_pivot = train_df.pivot_table(values="daily_sales", index=train_df["date"].dt.day_name(),
            columns=train_df["date"].dt.month, aggfunc="mean"
        )        
        st.plotly_chart(px.imshow(season_pivot, aspect="auto"), use_container_width=True)
        for col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")      
        for col in X_val.columns:
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")       
        X_train = X_train.fillna(0)
        X_val   = X_val.fillna(0)        
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0)
        y_val   = pd.to_numeric(y_val, errors="coerce").fillna(0)
        results_df, forecasts, best_model, tuned_rf = train_models(X_train, y_train_log, X_val, y_val_log)
        model_map = {
            "Linear Regression": LinearRegression(),
            "Random Forest": tuned_rf,
            "Gradient Boosting": GradientBoostingRegressor(
                 n_estimators=600, learning_rate=0.03, max_depth=5,
                 min_samples_leaf=3, subsample=0.8, random_state=42
            )
        }      
        final_model = model_map[best_model]
        X_full = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_full = np.log1p(df["daily_sales"].clip(lower=0))
        final_model.fit(X_full.astype(float), y_full.astype(float))
        val_preds = forecasts[best_model]
        y_true = np.expm1(y_val_log)
        residuals = y_true - val_preds
        val_df["residual"] = residuals        
        product_sigma = (val_df.groupby("product_id")["residual"].std().fillna(residuals.std()))
        residual_model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        )        
        residual_model.fit(X_val, residuals)
        err_df = pd.DataFrame({
            "date": val_df["date"], "abs_error": abs(residuals)
        }).sort_values("date")        
        err_df["rolling_error"] = err_df["abs_error"].rolling(14).mean()  
        st.markdown(
            '<div class="section-title">Select Future Forecast Range</div>', unsafe_allow_html=True
        )       
        max_allowed_date = pd.Timestamp("2026-06-30")
        default_start = df["date"].max() + pd.Timedelta(days=1)
        default_end = min(default_start + pd.Timedelta(days=90), max_allowed_date)      
        default_start_d = default_start.date()
        default_end_d = default_end.date()
        max_allowed_d = max_allowed_date.date()
        stored_range = st.session_state.get("forecast_range", None)    
        if (stored_range is None or len(stored_range) != 2):
            st.session_state.forecast_range = (default_start_d, default_end_d)
        else:
            start = pd.to_datetime(stored_range[0]).date()
            end = pd.to_datetime(stored_range[1]).date()
            start = max(start, default_start_d)
            end = min(end, max_allowed_d)    
            st.session_state.forecast_range = (start, end)    
        future_range = st.date_input(
            "Select Future Forecast Dates",
            value=st.session_state.forecast_range,
            min_value=default_start_d, max_value=max_allowed_d,
        )    
        st.session_state.forecast_range = future_range     
        future_start = pd.to_datetime(future_range[0])
        future_end = pd.to_datetime(future_range[1])
        future_end = min(future_end, max_allowed_date)       
        future_dates = pd.date_range(future_start, future_end, freq="D")
        FORECAST_DAYS = len(future_dates)
        if FORECAST_DAYS == 0:
            st.warning("No future dates selected.")
            st.stop()
        future_df = pd.DataFrame({"date": future_dates})
        if default_start > max_allowed_date:
            st.warning("No future horizon available beyond June 2026 based on current data.")
            st.stop()         
        last_row = df.sort_values("date").iloc[-1]        
        future_df["price"] = last_row["price"]
        future_df["dayofweek"] = future_df["date"].dt.dayofweek     
        if "dayofweek" not in df.columns:
            df["dayofweek"] = df["date"].dt.dayofweek        
        promo_pattern = (df.groupby("dayofweek")["promotion"].mean())   
        future_df["promotion"] = (future_df["dayofweek"].map(promo_pattern).fillna(0))
        future_df["holiday_flag"] = (future_df["date"].isin(india_holidays).astype(int))
        future_df["region"] = last_row["region"]
        future_df["avg_stock"] = last_row["avg_stock"]
        future_df["avg_delay_rate"] = last_row["avg_delay_rate"]
        future_df["sin_dow"] = np.sin(2*np.pi*future_df["dayofweek"]/7)
        future_df["cos_dow"] = np.cos(2*np.pi*future_df["dayofweek"]/7)
        future_df["month"] = future_df["date"].dt.month
        future_df["sin_month"] = np.sin(2*np.pi*future_df["month"]/12)
        future_df["cos_month"] = np.cos(2*np.pi*future_df["month"]/12)
        future_df["dayofyear"] = future_df["date"].dt.dayofyear
        future_df["weekofyear"] = future_df["date"].dt.isocalendar().week.astype(int)       
        all_fc = []
        if df.empty:
            st.warning("No data available for forecasting.")
            st.stop()
        for pid in df["product_id"].unique():            
            prod_df = df[df["product_id"] == pid].copy()
            history = prod_df.copy()
            regime = history["demand_regime"].iloc[-1]
            future_preds = []
            if prod_df.empty:
                continue
            for i in range(FORECAST_DAYS):     
                row = future_df.iloc[i].copy()
                row["lag_1"] = history.iloc[-1]["daily_sales"]
                if len(history) >= 7:
                    row["lag_7"] = history.iloc[-7]["daily_sales"]
                    row["rolling_7"] = history["daily_sales"].tail(7).mean()
                else:
                    m = history["daily_sales"].mean()
                    row["lag_7"] = m
                    row["rolling_7"] = m                 
                row["rolling_14"] = history["daily_sales"].tail(14).mean()
                row["rolling_30"] = history["daily_sales"].tail(30).mean()
                row["lag_diff_1"] = row["lag_1"] - row["lag_7"]
                row["lag_ratio_1"] = row["lag_1"] / (row["lag_7"] + 1)
                row["trend_7"] = row["rolling_7"] - row["lag_7"]
                row["sku_difficulty"] = history.get("sku_difficulty", pd.Series([0])).iloc[-1]
                row["price_change"] = history["price_change"].tail(7).mean()
                row["promo_rolling_7"] = history["promotion"].tail(7).mean()       
                X_future = pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0)
                X_future = X_future.astype(float)       
                pred_log = final_model.predict(X_future)[0]
                pred = np.expm1(pred_log)
                if row["holiday_flag"] == 1:
                    pred *= 1.15
                res_corr = residual_model.predict(X_future)[0]
                pred = pred + 0.3 * res_corr
                hist_mean = history["daily_sales"].mean()          
                pred = max(hist_mean * 0.2, pred)
                pred = np.clip(pred,hist_mean * 0.3,hist_mean * 3)         
                hist_std = history["daily_sales"].std()         
                if np.isnan(hist_std):
                    hist_std = 0             
                upper = hist_mean + 3 * hist_std
                lower = max(0, hist_mean - 3 * hist_std)             
                pred = np.clip(pred, lower, upper)   
                recent_mean = history["daily_sales"].tail(14).mean()
                pred = 0.85 * pred + 0.15 * recent_mean
                difficulty = row.get("sku_difficulty", 0)
                alpha = 0.6 if difficulty > 1 else 0.8
                pred = alpha * pred + (1 - alpha) * recent_mean
                pred = np.clip(pred,recent_mean * 0.5,recent_mean * 1.8)
                global_mean = history["daily_sales"].mean()
                pred = 0.9 * pred + 0.1 * global_mean
                if regime == "Growing":
                   pred *= 1.05
                elif regime == "Declining":
                    pred *= 0.95
                pred = min(pred, history["daily_sales"].max() * 2)
                future_preds.append(pred)            
                new_hist = row.copy()
                new_hist["daily_sales"] = pred               
                history = pd.concat([history, new_hist.to_frame().T], ignore_index=True)            
            tmp = future_df.copy()
            tmp["product_id"] = pid
            tmp["forecast"] = future_preds
            product_region = df[df["product_id"] == pid]["region"].iloc[-1]
            tmp["region"] = product_region
            all_fc.append(tmp)      
        if len(all_fc) == 0:
            st.warning("No forecasts generated.")
            df_fc = pd.DataFrame(columns=["date","product_id","forecast"])
        else:
            df_fc = pd.concat(all_fc, ignore_index=True)   
        for col in ["date","product_id","forecast"]:
            if col not in df_fc.columns:
                df_fc[col] = np.nan
        trend = df_fc.groupby("product_id")["forecast"].pct_change()
        df_fc["forecast"] = df_fc["forecast"] * (1 - trend.fillna(0)*0.3)     
        df_fc["sigma"] = df_fc["product_id"].map(product_sigma)
        df_fc["sigma"] = df_fc["sigma"].fillna(residuals.std())
        df_fc["sigma"] = df_fc["sigma"].clip(lower=1)
        df_fc["sigma"] = np.minimum(df_fc["sigma"],df_fc["forecast"] * 0.5)             
        df_fc["lower_ci"] = np.maximum(
            df_fc["forecast"] * 0.3,df_fc["forecast"] - 1.96 * df_fc["sigma"]
        )        
        df_fc["upper_ci"] = (df_fc["forecast"] + 1.96 * df_fc["sigma"])
        total_future_demand = df_fc["forecast"].sum()
        peak_day = df_fc.loc[df_fc["forecast"].idxmax(),"date"]
        with st.expander("📘 Data Dictionary "):
            st.dataframe( DATA_DICTIONARY[DATA_DICTIONARY["Column"].isin(df_fc.columns)],
                use_container_width=True
            ) 
        with st.expander("🔍 Data Profiling "):
            profile_fc = {
                "Total Forecast Records": len(df_fc),
                "Future Date Range": f"{df_fc['date'].min().date()} → {df_fc['date'].max().date()}",
                "Products": df_fc["product_id"].nunique(),
                "Avg Forecast": round(df_fc["forecast"].mean(), 2),
                "Forecast Volatility": round(df_fc["forecast"].std(), 2)
            }
            for k, v in profile_fc.items():
                st.write(f"**{k}:** {v}")
        if "all_forecasts" not in st.session_state:
            st.session_state["all_forecasts"] = pd.DataFrame()        
        if not st.session_state["all_forecasts"].empty and "product" in locals():
            st.session_state["all_forecasts"] = (
                st.session_state["all_forecasts"]
                [st.session_state["all_forecasts"]["product_id"] != product]
            )        
        st.session_state["all_forecasts"] = pd.concat(
            [st.session_state["all_forecasts"], df_fc], ignore_index=True
        )
        st.session_state["all_forecasts"].to_csv( FORECAST_PATH, index=False)  
        top_product_global = (
            df.groupby("product_id")["daily_sales"].mean().sort_values(ascending=False).index[0]
        )    
        st.markdown(
            '<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True
        )     
        st.markdown(
            "<div class='card'>"
            "Comparison of all trained models using MAE, RMSE, and R² metrics. "
            "The model with the lowest RMSE is selected as the production model."
            "</div>", unsafe_allow_html=True
        )     
        st.dataframe(results_df, use_container_width=True)
        st.markdown('<div class="section-title">Executive KPIs</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        metrics = [
            ("Best Model", best_model),
            ("Product", df_fc["product_id"].iloc[0] if not df_fc.empty else "N/A"),
            ("Avg Forecast", int(df_fc.forecast.mean())),
            ("RMSE", round(results_df.iloc[0]["RMSE"],2)),
            ("NRMSE", round(results_df.iloc[0]["NRMSE"],3))
        ]
        for col,(k,v) in zip([c1,c2,c3,c4,c5],metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{k}</div>
                    <div class="metric-value">{v}</div>
                </div>
                """, unsafe_allow_html=True)
        st.success(
            f"✅ **{best_model}** selected as the production model "
            f"with lowest RMSE ({results_df.iloc[0]['RMSE']:.2f})."
        )   
        st.markdown(
            '<div class="section-title">RMSE Comparison Across Models</div>',unsafe_allow_html=True
        )        
        rmse_fig = px.bar(
            results_df, x="Model", y="RMSE",
            text="RMSE", template="plotly_white"
        )        
        rmse_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')        
        rmse_fig.update_layout(
            yaxis_title="Root Mean Squared Error", xaxis_title="Model",
            uniformtext_minsize=10, uniformtext_mode='hide'
        )       
        st.plotly_chart(rmse_fig, use_container_width=True)
        st.markdown(
            '<div class="section-title">Forecast with Confidence Intervals</div>', unsafe_allow_html=True
        )       
        fig = go.Figure()       
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["forecast"], name="Forecast", line=dict(width=3)
        ))        
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["upper_ci"], name="Upper CI", line=dict(dash="dot")
        ))      
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["lower_ci"], name="Lower CI", fill="tonexty", opacity=0.25
        ))      
        st.plotly_chart(fig, use_container_width=True)     
        st.write(f"Total Future Demand: {int(total_future_demand)}")
        st.write(f"Peak Demand Day: {peak_day.date()}")
        st.markdown('<div class="section-title">💬 Demand Analytics Assistant</div>', unsafe_allow_html=True)
        q = st.text_input("Ask anything about demand forecasting")
        if q:
            st.markdown(
                f"<div class='card'>{demand_nlp(df_fc, results_df, best_model, q, top_product_global)}</div>",
                unsafe_allow_html=True
            )
        st.markdown(
            '<div class="section-title">Forecast Output Preview</div>', unsafe_allow_html=True
        )       
        preview_cols = ["date","product_id","forecast","lower_ci","upper_ci"]       
        today = pd.Timestamp.today().normalize()
        future_limit = today + pd.Timedelta(days=90)     
        preview_df = df_fc[(df_fc["date"] >= today) & (df_fc["date"] <= future_limit)]      
        st.dataframe(preview_df[preview_cols], use_container_width=True)
        st.download_button("⬇ Download Forecast Output",
            df_fc.to_csv(index=False), file_name="forecast_demand.csv"
        )
