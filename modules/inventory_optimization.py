# ======================================================================================
# OmniFlow-D2D : Inventory Optimization Module
# ======================================================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
st.set_page_config(page_title="Inventory Optimization", layout="wide")
def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg: #f8fafc;
        --text: #0f172a;
        --muted: #475569;
        --primary: #065f46;
        --border: #e5e7eb;
        --accent: #dcfce7;
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
        background: linear-gradient(180deg, #ecfdf5, #ffffff);
        padding: 18px;
        text-align: center;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(6,95,70,0.25);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 36px rgba(6,95,70,0.35);
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
        box-shadow: 0 6px 18px rgba(6,95,70,0.25);
    }
    </style>
    """, unsafe_allow_html=True)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INVENTORY_PATH = os.path.join(DATA_DIR, "retail_inventory_snapshot.csv")
DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "product_id","current_stock","avg_daily_demand","annual_demand",
        "EOQ","safety_stock","reorder_point","stock_status"
    ],
    "Description": [
        "Unique product identifier","Current available inventory",
        "Average daily forecasted demand","Projected annual demand",
        "Economic Order Quantity","Buffer stock to handle demand uncertainty",
        "Stock level at which reorder is triggered","Inventory health indicator"
    ]
})
@st.cache_data
def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["warehouse_id"] = (
        df["warehouse_id"].astype(str).str.upper()
        .str.replace("-", "", regex=False).str.strip()
    )
    df["product_id"] = df["product_id"].astype(str).str.upper().str.strip()
    df["on_hand_qty"] = pd.to_numeric(df["on_hand_qty"], errors="coerce")
    df["on_hand_qty"] = df["on_hand_qty"].fillna(0)
    df["on_hand_qty"] = df["on_hand_qty"].clip(lower=0)
    return df
def data_profiling(df):
    return {
        "Total Products": df["product_id"].nunique(),
        "Average Stock Level": round(df["current_stock"].mean(), 2),
        "Products at Risk": int(
            df["stock_status"].isin(["🔴 Critical", "🟠 Reorder Required"]).sum()
        ),
        "Average EOQ": round(df["EOQ"].mean(), 2),
        "Average Safety Stock": round(df["safety_stock"].mean(), 2)
    }
def inventory_optimization(forecast_df, inventory_df):
    forecast_df["forecast"] = pd.to_numeric(forecast_df["forecast"], errors="coerce").fillna(0)
    forecast_df["product_id"] = (forecast_df["product_id"].astype(str).str.upper().str.strip())
    demand = (forecast_df.groupby("product_id").agg(avg_daily_demand=("forecast", "mean"),
        demand_std=("forecast", "std")).reset_index()
    )   
    demand["demand_std"] = demand["demand_std"].fillna(0)
    demand["annual_demand"] = demand["avg_daily_demand"] * 365
    df = demand.merge(
        inventory_df.groupby("product_id", as_index=False)
        .agg(current_stock=("on_hand_qty", "sum")),
        on="product_id",how="left"
    )
    df["current_stock"] = df["current_stock"].fillna(0)
    ordering_cost = 500
    holding_cost_rate = 0.25
    lead_time_days = 14
    service_level_z = 1.65
    df["holding_cost"] = np.maximum(holding_cost_rate * (df["avg_daily_demand"] + df["demand_std"]),1)
    df["EOQ"] = np.sqrt((2 * df["annual_demand"] * ordering_cost) / (df["holding_cost"] + 1)) 
    df["EOQ"] = df["EOQ"].clip(lower=df["avg_daily_demand"] * 14,upper=df["avg_daily_demand"] * 60)
    df["safety_stock"] = (service_level_z * df["demand_std"] * np.sqrt(lead_time_days)).clip(lower=0)
    df["safety_stock"] = np.maximum(df["safety_stock"],df["avg_daily_demand"] * 2)
    planning_days = 14
    df["future_demand_14"] = df["avg_daily_demand"] * planning_days
    df["reorder_point"] = np.ceil(df["future_demand_14"] + df["safety_stock"])
    df["stockout_risk"] = (df["reorder_point"] - df["current_stock"]) / (df["reorder_point"] + 1e-6)
    df["stockout_risk"] = df["stockout_risk"].clip(0, 1)
    df["stock_cover_days"] = (df["current_stock"] / df["avg_daily_demand"].replace(0,1))
    df["stock_status"] = np.where(
        df["current_stock"] < df["future_demand_14"] * 0.5,"🔴 Critical",
        np.where(df["current_stock"] < df["future_demand_14"],
            "🟠 Reorder Required","🟢 Stock OK")
    )
    return df
def inventory_nlp(df, q):
    q = q.lower()
    if "products at risk" in q or "reorder" in q:
        return f"{df['stock_status'].isin(['🔴 Critical', '🟠 Reorder Required']).sum()} products require reorder."
    if "average eoq" in q:
        return f"Average EOQ is {df['EOQ'].mean():.0f} units."
    if "safety stock" in q:
        return f"Average safety stock is {df['safety_stock'].mean():.0f} units."
    if "summary" in q:
        return (
            "Inventory optimization indicates reorder requirements for high-risk products. "
            "EOQ and safety stock levels are optimized to balance cost and service level."
        )
    return (
        "You can ask about:\n"
        "- products at risk\n"
        "- average EOQ\n"
        "- safety stock\n"
        "- inventory summary"
    )
def inventory_optimization_page():
    inject_css()
    tab1, tab2 = st.tabs(["📘 Overview", "📦 Application"])
    with tab1:
        st.markdown('<div class="section-title">Inventory Optimization – Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        This module transforms forecasted demand into actionable inventory decisions.
        It minimizes stockouts and holding costs using EOQ, safety stock, and reorder point logic.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-title">Objectives</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul>
            <li>Determine optimal order quantities</li>
            <li>Identify products at stockout risk</li>
            <li>Optimize inventory holding cost</li>
            <li>Support proactive replenishment planning</li>
        </ul>
        </div>    
        """, unsafe_allow_html=True)
    with tab2:  
        FORECAST_PATH = os.path.join(DATA_DIR, "forecast_output.csv")    
        if "all_forecasts" in st.session_state and not st.session_state["all_forecasts"].empty:
            forecast_df = st.session_state["all_forecasts"]    
        elif os.path.exists(FORECAST_PATH):
            forecast_df = pd.read_csv(FORECAST_PATH)
            st.session_state["all_forecasts"] = forecast_df        
        else:
            st.info(
                "ℹ️ Inventory Optimization depends on Demand Forecasting.\n\n"
                "Please run the Demand Forecasting module once to generate forecasts."
            )
            return
        inventory_df = load_inventory()
        opt_df = inventory_optimization(forecast_df, inventory_df)
        opt_df["planning_demand"] = opt_df["avg_daily_demand"] * 14 
        output_path = os.path.join(DATA_DIR, "inventory_optimization.csv")
        opt_df.to_csv(output_path, index=False)
        st.session_state["inventory_optimized"] = opt_df.copy()
        if "inventory_view_mode" not in st.session_state:
            st.session_state.inventory_view_mode = "Overall Inventory"     
        view_mode = st.radio("Inventory Analysis Mode",
            ["Overall Inventory", "Single Product"],horizontal=True,
            index=["Overall Inventory", "Single Product"].index(
                st.session_state.inventory_view_mode
            )
        )     
        st.session_state.inventory_view_mode = view_mode
        if view_mode == "Single Product":
            product_list = sorted(opt_df["product_id"].unique())
            if "inventory_selected_product" not in st.session_state:
                st.session_state.inventory_selected_product = product_list[0]          
            product = st.selectbox("Select Product",product_list,
                index=product_list.index(st.session_state.inventory_selected_product)
            )         
            st.session_state.inventory_selected_product = product
            view_df = opt_df[opt_df["product_id"] == product]
        else:
            view_df = opt_df.copy()
        with st.expander("📘 Data Dictionary"):
            st.dataframe(DATA_DICTIONARY, use_container_width=True)
        with st.expander("🔍 Data Profiling "):
            profile = data_profiling(view_df)
            for k, v in profile.items():
                st.write(f"**{k}:** {v}")
        st.markdown('<div class="section-title">Executive KPIs</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        kpis = [
            ("Products at Risk",
             (opt_df["stock_status"].isin(["🔴 Critical", "🟠 Reorder Required"])).sum()),
            ("Average EOQ", int(opt_df["EOQ"].mean())),
            ("Average Safety Stock", int(opt_df["safety_stock"].mean()))
        ]
        for col, (t, v) in zip([c1, c2, c3], kpis):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{t}</div>
                    <div class="metric-value">{v}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Stock Level vs Reorder Point</div>',
            unsafe_allow_html=True
        )      
        stock_fig = px.scatter(
            view_df,x="reorder_point",y="current_stock",
            color="stock_status",
            color_discrete_map={
                "🔴 Critical": "red",
                "🟠 Reorder Required": "orange",
                "🟢 Stock OK": "green"
            },
            hover_data=["product_id"],
            labels={
                "reorder_point": "Reorder Point",
                "current_stock": "Current Stock"
            },template="plotly_white"
        )      
        stock_fig.add_shape(
            type="line",x0=0, y0=0,
            x1=view_df["reorder_point"].max(),
            y1=view_df["reorder_point"].max(),
            line=dict(dash="dot", color="red")
        )     
        st.plotly_chart(stock_fig, use_container_width=True)
        st.markdown('<div class="section-title">Inventory Plan Preview</div>', unsafe_allow_html=True)
        st.dataframe(
            view_df[[
                "product_id","current_stock","EOQ","safety_stock",
                "planning_demand","reorder_point","stock_status"
            ]], use_container_width=True
        )
        st.markdown('<div class="section-title">💬 Inventory Assistant</div>', unsafe_allow_html=True)
        q = st.text_input("Ask inventory-related questions")
        if q:
            st.markdown(
                f"<div class='card'>{inventory_nlp(view_df, q)}</div>",unsafe_allow_html=True
            )
        c1, c2, c3 = st.columns(3)
        with c2:
            st.download_button(
                "⬇ Download Inventory Output",
                view_df.to_csv(index=False),
                "inventory_optimization.csv"
            )
