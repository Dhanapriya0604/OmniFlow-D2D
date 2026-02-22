# ======================================================================================
# OmniFlow-D2D : Logistics Optimization Module
# ======================================================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
st.set_page_config(page_title="Logistics Optimization", layout="wide")
def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg:#f8fafc;
        --text:#0f172a;
        --muted:#475569;
        --primary:#1d4ed8;
        --accent:#dbeafe;
        --border:#e5e7eb;
    }
    html, body {
        background:var(--bg);
        color:var(--text);
        font-family:Inter;
    }
    section.main > div {
        animation: fadeIn .4s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity:0; transform:translateY(6px);}
        to {opacity:1; transform:translateY(0);}
    }
    .section-title {
        font-size:28px;
        font-weight:800;
        margin:28px 0 14px;
    }
    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transition: 0.25s;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.14);
    }
    .metric-card {
        background:linear-gradient(180deg,#dbeafe,#ffffff);
        padding:18px;
        border-radius:16px;
        text-align:center;
        box-shadow:0 6px 18px rgba(29,78,216,0.25);
    }
    .metric-value {
        font-size:30px;
        font-weight:900;
        color:var(--primary);
    }
    .metric-card:hover {
        transform: translateY(-6px);
        transition: 0.3s ease;
        box-shadow: 0 18px 40px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent);
        color: var(--primary);
        box-shadow: 0 6px 18px rgba(180,83,9,0.25);
    }
    </style>
    """, unsafe_allow_html=True)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH  = os.path.join(DATA_DIR, "retail_inventory_snapshot.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")
LOGISTICS_PATH  = os.path.join(DATA_DIR, "supply_chain_logistics_shipments.csv")
def clean_text_column(df, col, remove_dash=False):
    if col in df.columns:
        df[col] = (df[col].astype(str).str.strip().str.upper())
        if remove_dash:
            df[col] = df[col].str.replace("-", "", regex=False)
    return df
@st.cache_data
def load_forecasts():
    if "all_forecasts" in st.session_state:
        df = st.session_state["all_forecasts"].copy()
    else:
        df = pd.read_csv(FORECAST_PATH)    
    df.columns = df.columns.str.lower()
    df = clean_text_column(df, "product_id")
    if "region" not in df.columns or df["region"].isnull().all():
        df["region"] = "UNKNOWN"
    else:
        df["region"] = df["region"].astype(str).str.upper().str.strip() 
    return df
@st.cache_data
def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    df.columns = df.columns.str.lower()
    df = clean_text_column(df, "warehouse_id", remove_dash=True)
    df = clean_text_column(df, "product_id")
    df["on_hand_qty"] = pd.to_numeric(df["on_hand_qty"], errors="coerce")
    df["on_hand_qty"] = df["on_hand_qty"].fillna(0)
    df["on_hand_qty"] = df["on_hand_qty"].clip(lower=0)
    return df
@st.cache_data
def load_production():
    if os.path.exists(PRODUCTION_PATH):
        df = pd.read_csv(PRODUCTION_PATH)
        df.columns = df.columns.str.lower()
        df = clean_text_column(df, "product_id")
        return df
    return pd.DataFrame()
@st.cache_data
def load_logistics():
    df = pd.read_csv(LOGISTICS_PATH)
    df.columns = (
        df.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")
    )
    if "warehouse_id" in df.columns and "source_warehouse" not in df.columns:
        df["source_warehouse"] = df["warehouse_id"]
    df = clean_text_column(df, "source_warehouse", remove_dash=True)
    df = clean_text_column(df, "destination_region")
    df = clean_text_column(df, "carrier")  
    df = clean_text_column(df, "product_id")
    required_cols = ["source_warehouse","destination_region","carrier",
        "actual_delivery_days","delay_flag","logistics_cost"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN" if col in ["carrier", "destination_region"] else 0
    df["actual_delivery_days"] = pd.to_numeric(df["actual_delivery_days"], errors="coerce").fillna(5)
    df["logistics_cost"] = pd.to_numeric(df["logistics_cost"], errors="coerce").fillna(0)
    df["delay_flag"] = pd.to_numeric(df["delay_flag"], errors="coerce").fillna(0)
    df["destination_region"] = df["destination_region"].astype(str).str.upper().str.strip()
    return df
def logistics_optimization(forecast_df, inventory_df, production_df, logistics_df):
    if "date" in forecast_df.columns:
        forecast_df = forecast_df.sort_values(["product_id","date"])
    forecast_df["rank"] = forecast_df.groupby("product_id").cumcount()   
    demand = (forecast_df[forecast_df["rank"] < 14]
        .groupby("product_id", as_index=False)["forecast"]
        .mean().rename(columns={"forecast": "avg_daily_demand"})
    )
    planning_days = 14
    demand["planning_demand"] = demand["avg_daily_demand"] * planning_days
    stock = (
        inventory_df.sort_values("on_hand_qty", ascending=False)
        .groupby("product_id", as_index=False)
        .first()[["product_id","warehouse_id","on_hand_qty"]]
        .rename(columns={"on_hand_qty":"current_stock"})
    )
    df = demand.merge(stock, on="product_id", how="left")
    df["current_stock"] = df["current_stock"].fillna(0)
    df["warehouse_id"] = df["warehouse_id"].fillna("WH_UNKNOWN")
    df["stock_cover_days"] = (df["current_stock"] / df["avg_daily_demand"].replace(0, 1))
    df["shipping_need_14d"] = np.where(df["stock_cover_days"] > 28, 0,
        np.where(df["stock_cover_days"] > 14,df["planning_demand"] * 0.5,df["planning_demand"])
    )
    df["shipping_need_14d"] = df["shipping_need_14d"].round().astype(int)
    if not production_df.empty:
        df = df.merge(production_df[["product_id","production_required"]],
            on="product_id", how="left"
        )
    df["production_required"] = df["production_required"].fillna(0)
    logistics_df["delay_flag"] = logistics_df.get("delay_flag", 0)
    forecast_df["region"] = forecast_df["region"].astype(str).str.strip()    
    if {"product_id","destination_region"}.issubset(logistics_df.columns): 
        region_dist = (logistics_df.groupby(["product_id","destination_region"])
            .size().reset_index(name="shipments")
        )  
    else:
        region_dist = pd.DataFrame(columns=["product_id","destination_region","shipments"])
    region_dist["share"] = (
        region_dist.groupby("product_id")["shipments"].transform(lambda x: x / x.sum())
    ) 
    region_dist["share"] = region_dist["share"].clip(upper=0.5)
    df = df.merge(region_dist, on="product_id", how="left")    
    df["shipping_need_14d"] = (df["shipping_need_14d"] * df["share"]).fillna(df["shipping_need_14d"])
    fallback_region = (forecast_df.groupby("product_id")["region"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "UNKNOWN").reset_index()
    )   
    df = df.merge(fallback_region, on="product_id", how="left")
    df["destination_region"] = df["destination_region"].fillna(df["region"])
    df["destination_region"] = df["destination_region"].astype(str)
    df["destination_region"] = df["destination_region"].replace(
        {"1": "NORTH", "2": "SOUTH", "3": "WEST", "4": "EAST"}
    )
    warehouse_region_map = {
        "WH01": "NORTH",
        "WH02": "SOUTH",
        "WH03": "WEST",
        "WH04": "EAST"
    }    
    df["destination_region"] = df["destination_region"].fillna(
        df["warehouse_id"].map(warehouse_region_map)
    )
    df.drop(columns=["region"], errors="ignore", inplace=True)
    logistics_df["destination_region"] = (
        logistics_df["destination_region"].astype(str).str.strip().str.upper()
    )    
    if logistics_df.empty:
        region_stats = pd.DataFrame(columns=["destination_region","total_shipments",
            "delayed_shipments","avg_transit_days","avg_shipping_cost","avg_delay_rate"
        ])
    else:
        region_stats = (logistics_df.groupby("destination_region", as_index=False)
            .agg(
                total_shipments=("delay_flag", "count"),delayed_shipments=("delay_flag", "sum"),
                avg_transit_days=("actual_delivery_days", "mean"),avg_shipping_cost=("logistics_cost", "mean"),
            )
        )
        region_stats["avg_delay_rate"] = (
            region_stats["delayed_shipments"] / region_stats["total_shipments"].replace(0, 1)
        )
    df = df.merge(region_stats, on="destination_region", how="left")
    df["avg_delay_rate"] = df["avg_delay_rate"].fillna(0.05)
    median_days = logistics_df["actual_delivery_days"].median()
    if pd.isna(median_days):
        median_days = 5      
    df["avg_transit_days"] = df["avg_transit_days"].fillna(median_days)  
    cost_med = logistics_df["logistics_cost"].median()
    if pd.isna(cost_med):
        cost_med = 0
    df["avg_shipping_cost"] = df["avg_shipping_cost"].fillna(cost_med)
    carrier_perf = (logistics_df
        .dropna(subset=["source_warehouse","carrier"])
        .groupby(["source_warehouse","carrier"], as_index=False)
        .agg(carrier_delay=("delay_flag","mean"))
    )
    best_carrier = (
        carrier_perf.sort_values("carrier_delay").groupby("source_warehouse").first().reset_index()
    )
    df = df.merge(best_carrier,left_on="warehouse_id",
        right_on="source_warehouse",how="left"
    ).drop(columns=["source_warehouse"])
    df.rename(columns={
        "carrier":"recommended_carrier","carrier_delay":"carrier_delay_rate"
    }, inplace=True)
    df["recommended_carrier"] = df["recommended_carrier"].fillna("STANDARD")
    df["carrier_delay_rate"].fillna(df["avg_delay_rate"], inplace=True)
    df["logistics_risk"] = np.where(df["avg_delay_rate"] > 0.25,"High Delay Risk","Logistics Stable")
    df["shipping_priority"] = (df["shipping_need_14d"] * (1 + df["avg_delay_rate"]))
    df["cost_score"] = df["avg_shipping_cost"] / df["avg_shipping_cost"].max()
    df["delay_score"] = df["avg_delay_rate"]
    df["optimization_score"] = (0.6 * df["cost_score"] +  0.4 * df["delay_score"])
    df["shipping_need_14d"] = df["shipping_need_14d"].clip(lower=0)
    df = df.sort_values("optimization_score")
    df["warehouse_id"] = df["warehouse_id"].fillna("WH_UNKNOWN")
    df["destination_region"] = (df["destination_region"].fillna("UNKNOWN").astype(str).str.upper())
    df["recommended_carrier"] = df["recommended_carrier"].fillna("STANDARD")
    df["avg_delay_rate"] = df["avg_delay_rate"].fillna(0)
    df["avg_transit_days"] = df["avg_transit_days"].fillna(
        logistics_df["actual_delivery_days"].median()
    ) 
    df["avg_shipping_cost"] = df["avg_shipping_cost"].fillna(logistics_df["logistics_cost"].median()) 
    df["production_required"] = df["production_required"].fillna(0)   
    df["avg_daily_demand"] = pd.to_numeric(df["avg_daily_demand"], errors="coerce").fillna(0).round().astype(int)
    df["shipping_need_14d"] = pd.to_numeric(df["shipping_need_14d"], errors="coerce").fillna(0).round().astype(int)   
    df["avg_shipping_cost"] = df["avg_shipping_cost"].clip(100, 500)
    df["avg_transit_days"] = pd.to_numeric(df["avg_transit_days"], errors="coerce").fillna(0).round().astype(int)
    df = df.replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)
    return df   
def logistics_optimization_page():
    inject_css()
    tab1, tab2 = st.tabs(["📘 Overview", "🚚 Application"])
    with tab1:
        st.markdown(
            '<div class="section-title">Logistics Optimization Overview</div>',unsafe_allow_html=True
        )
        st.markdown("""<div class="card">
        Logistics optimization ensures products move efficiently from
        warehouses to destination regions while minimizing delays and costs.
        This module recommends carriers, estimates transit times,
        and highlights shipment risks.
        </div>""", unsafe_allow_html=True)
    with tab2:
        forecast_df   = load_forecasts()
        inventory_df  = load_inventory()
        production_df = load_production()
        logistics_df  = load_logistics()
        opt_df = logistics_optimization(forecast_df,inventory_df,production_df,logistics_df)
        log_path = os.path.join(DATA_DIR, "logistics_plan.csv")
        opt_df.to_csv(log_path, index=False)
        st.markdown('<div class="section-title">Logistics KPIs</div>',unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)    
        opt_df["shipment_size"] = np.where(
            opt_df["destination_region"] == "WEST", 100,
            np.where(opt_df["destination_region"] == "NORTH", 80, 60)
        )     
        opt_df["shipments_required"] = np.ceil(opt_df["shipping_need_14d"] / opt_df["shipment_size"])
        metrics = [
            ("Avg Delay Rate", round(opt_df["avg_delay_rate"].mean(),2)),
            ("Avg Transit Days", round(opt_df["avg_transit_days"].mean(),1)),
            ("Planning Shipments", int(opt_df["shipping_need_14d"].sum())),   
            ("Optimized Cost", int(
                (opt_df["shipments_required"] * opt_df["avg_shipping_cost"] * (1 - opt_df["delay_score"])).sum()
            ))
        ]
        for col, (k, v) in zip([c1, c2, c3, c4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div>{k}</div>
                    <div class="metric-value">{v}</div>
                </div>
                """, unsafe_allow_html=True)         
        st.markdown(
            '<div class="section-title">Shipping Need by Product</div>', unsafe_allow_html=True
        )
        st.plotly_chart(
            px.bar(
            opt_df, x="product_id", y="shipping_need_14d", color="destination_region",
            color_discrete_map={
                "NORTH": "#1f77b4","SOUTH": "#2ca02c","EAST": "#ff7f0e",
                "WEST": "#d62728","UNKNOWN": "#7f7f7f"
            },hover_data=["avg_delay_rate","avg_transit_days"]),use_container_width=True
        )
        st.markdown(
            '<div class="section-title">Shipping Demand by Region</div>', unsafe_allow_html=True
        )
        region_ship = (
            opt_df.groupby("destination_region")["shipping_need_14d"].sum().reset_index()
        )
        st.plotly_chart(
            px.bar(region_ship,x="destination_region",y="shipping_need_14d"),use_container_width=True
        )   
        region_delay = (
            opt_df.groupby("destination_region")["avg_delay_rate"].mean().reset_index()
        )
        st.plotly_chart(
            px.bar(region_delay,x="destination_region",y="avg_delay_rate",
                title="Delay Rate by Region"),use_container_width=True
        )
        st.markdown(
            '<div class="section-title">Delay Risk Split</div>',unsafe_allow_html=True
        )
        st.plotly_chart(
            px.pie(opt_df,names="logistics_risk",hole=0.4),use_container_width=True
        )
        risk_df = opt_df[opt_df["logistics_risk"] == "High Delay Risk"]
        st.dataframe(
            risk_df[["product_id","destination_region","shipping_need_14d","avg_delay_rate"]]
        )
        st.markdown(
            '<div class="section-title">Logistics Output Preview</div>',unsafe_allow_html=True
        )
        st.dataframe(opt_df, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        with c2:
            st.download_button(
                "⬇ Download Logistics Plan",opt_df.to_csv(index=False),"logistics_plan.csv"
            )
