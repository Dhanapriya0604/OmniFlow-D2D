# ======================================================================================
# OmniFlow-D2D : Logistics Optimization Module
# ======================================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Logistics Optimization", layout="wide")

# ======================================================================================
# CSS UI
# ======================================================================================
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

    </style>
    """, unsafe_allow_html=True)

# ======================================================================================
# PATHS
# ======================================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FORECAST_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH  = os.path.join(DATA_DIR, "retail_inventory_snapshot.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")
LOGISTICS_PATH  = os.path.join(DATA_DIR, "supply_chain_logistics_shipments.csv")

def clean_text_column(df, col, remove_dash=False):
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if remove_dash:
            df[col] = df[col].str.replace("-", "", regex=False)

    return df

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_forecasts():
    if "all_forecasts" in st.session_state:
        df = st.session_state["all_forecasts"].copy()
    else:
        df = pd.read_csv(FORECAST_PATH)

    df.columns = df.columns.str.lower()
    df = clean_text_column(df, "product_id")
    return df

@st.cache_data
def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    df.columns = df.columns.str.lower()
    df = clean_text_column(df, "warehouse_id", remove_dash=True)
    df = clean_text_column(df, "product_id")
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
    # clean column names
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    df = clean_text_column(df, "source_warehouse", remove_dash=True)
    df = clean_text_column(df, "destination_region")
    df = clean_text_column(df, "carrier")
    df = clean_text_column(df, "product_id")

    # ensure required columns exist
    required_cols = [
        "source_warehouse",
        "destination_region",
        "carrier",
        "actual_delivery_days",
        "delay_flag",
        "logistics_cost"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    return df


# ======================================================================================
# LOGISTICS OPTIMIZATION
# ======================================================================================
def logistics_optimization(forecast_df, inventory_df, production_df, logistics_df):

    forecast_df = forecast_df.sort_values("date")

    demand = (
        forecast_df.groupby("product_id")
        .head(14)
        .groupby("product_id")["forecast"]
        .mean()
        .reset_index(name="avg_daily_demand")
    )

    demand["weekly_demand"] = demand["avg_daily_demand"] * 7

    # Inventory with warehouse
    stock = (
        inventory_df.groupby("product_id", as_index=False)
        .agg(
            current_stock=("on_hand_qty", "sum"),
            warehouse_id=("warehouse_id", "first")
        )
    )

    df = demand.merge(stock, on="product_id", how="left")

    # Shipping need
    df["weekly_shipping_need"] = np.maximum(
        df["weekly_demand"] * 0.2,
        df["weekly_demand"] - df["current_stock"] * 0.25
    )

    # Production link
    if not production_df.empty:
        df = df.merge(
            production_df[["product_id","production_required"]],
            on="product_id",
            how="left"
        )
    else:
        df["production_required"] = 0

    logistics_df["delay_flag"] = logistics_df.get("delay_flag", 0)

    # Warehouse → Region
    region_map = (
        logistics_df
        .groupby("product_id")["destination_region"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    
    df = df.merge(region_map, on="product_id", how="left")

    # Region performance
    region_stats = (
        logistics_df.groupby("destination_region", as_index=False)
        .agg(
            avg_delay_rate=("delay_flag","mean"),
            avg_transit_days=("actual_delivery_days","mean"),
            avg_shipping_cost=("logistics_cost","mean")
        )
    )

    df = df.merge(region_stats, on="destination_region", how="left")

    # Fill missing safely
    df["avg_delay_rate"].fillna(region_stats["avg_delay_rate"].mean(), inplace=True)
    median_days = logistics_df["actual_delivery_days"].median()
    if pd.isna(median_days):
        median_days = 5
    
    df["avg_transit_days"] = df["avg_transit_days"].fillna(median_days)
    
    df["avg_shipping_cost"] = df["avg_shipping_cost"].fillna(
        logistics_df["logistics_cost"].median()
    )

    # Carrier recommendation
    carrier_perf = (
        logistics_df
        .dropna(subset=["source_warehouse","carrier"])
        .groupby(["source_warehouse","carrier"], as_index=False)
        .agg(carrier_delay=("delay_flag","mean"))
    )

    best_carrier = (
        carrier_perf.sort_values("carrier_delay")
        .groupby("source_warehouse")
        .first()
        .reset_index()
    )

    df = df.merge(
        best_carrier,
        left_on="warehouse_id",
        right_on="source_warehouse",
        how="left"
    ).drop(columns=["source_warehouse"])

    df.rename(columns={
        "carrier":"recommended_carrier",
        "carrier_delay":"carrier_delay_rate"
    }, inplace=True)

    df["recommended_carrier"] = df["recommended_carrier"].fillna("Standard Carrier")
    df["carrier_delay_rate"].fillna(df["avg_delay_rate"], inplace=True)

    # Risk
    threshold = df["avg_delay_rate"].quantile(0.7)

    df["logistics_risk"] = np.where(
        df["avg_delay_rate"] > threshold,
        "High Delay Risk",
        "Logistics Stable"
    )

    # Priority
    df["shipping_priority"] = (
        df["weekly_shipping_need"] * (1 + df["avg_delay_rate"])
    )
    df["weekly_shipping_need"] = df["weekly_shipping_need"].clip(lower=0)
    df = df.sort_values("shipping_priority", ascending=False)
   
    # ================= FINAL CLEANUP =================
    df["warehouse_id"] = df["warehouse_id"].fillna("UNKNOWN")
    df["destination_region"] = (
        df["destination_region"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    df["recommended_carrier"] = df["recommended_carrier"].fillna("STANDARD")
    
    df["avg_delay_rate"] = df["avg_delay_rate"].fillna(0)
    df["avg_transit_days"] = df["avg_transit_days"].fillna(
        logistics_df["actual_delivery_days"].median()
    ) 
    df["avg_shipping_cost"] = df["avg_shipping_cost"].fillna(
        logistics_df["logistics_cost"].median()
    ) 
    df["production_required"] = df["production_required"].fillna(0)   
    df = df.replace([np.inf, -np.inf], 0)
    df["avg_daily_demand"] = df["avg_daily_demand"].round().astype(int)
    df["weekly_shipping_need"] = df["weekly_shipping_need"].round().astype(int)
    df["avg_shipping_cost"] = df["avg_shipping_cost"].round(2)

    return df
    
def logistics_optimization_page():

    inject_css()

    tab1, tab2 = st.tabs(["📘 Overview", "🚚 Application"])

    # ================= OVERVIEW =================
    with tab1:
        st.markdown(
            '<div class="section-title">Logistics Optimization Overview</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        Logistics optimization ensures products move efficiently from
        warehouses to destination regions while minimizing delays and costs.
        This module recommends carriers, estimates transit times,
        and highlights shipment risks.
        """)

    # ================= APPLICATION =================
    with tab2:

        forecast_df   = load_forecasts()
        inventory_df  = load_inventory()
        production_df = load_production()
        logistics_df  = load_logistics()

        opt_df = logistics_optimization(
            forecast_df,
            inventory_df,
            production_df,
            logistics_df
        )
        # ---------------- EDA ----------------
        st.markdown(
            '<div class="section-title">Logistics Data Overview</div>',
            unsafe_allow_html=True
        )
        
        with st.expander("Dataset Summary"):
        
            c1, c2, c3 = st.columns(3)
        
            c1.metric("Total Shipments", len(logistics_df))
            c2.metric("Warehouses",
                      logistics_df["source_warehouse"].nunique())
            c3.metric("Regions",
                      logistics_df["destination_region"].nunique())
        
            st.write("### Missing Values")
            st.dataframe(
                logistics_df.isna().sum().reset_index(
                ).rename(columns={"index": "Column", 0: "Missing Count"}),
                use_container_width=True
            )

        # -------- KPIs --------
        st.markdown('<div class="section-title">Logistics KPIs</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        metrics = [
            ("Avg Delay Rate",
             round(opt_df["avg_delay_rate"].mean(),2)),

            ("Avg Transit Days",
             round(opt_df["avg_transit_days"].mean(),1)),

            ("Weekly Shipments",
             int(opt_df["weekly_shipping_need"].sum()))
        ]

        for col, (k, v) in zip([c1, c2, c3], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div>{k}</div>
                    <div class="metric-value">{v}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write(logistics_df["destination_region"].value_counts())
        # -------- Charts --------
        st.markdown(
            '<div class="section-title">Shipping Need by Product</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            px.bar(
                opt_df,
                x="product_id",
                y="weekly_shipping_need",
                color="destination_region",
                hover_data=["avg_delay_rate","avg_transit_days"]
            ),
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">Shipping Demand by Region</div>',
            unsafe_allow_html=True
        )

        region_ship = (
            opt_df.groupby("destination_region")["weekly_shipping_need"]
            .sum()
            .reset_index()
        )

        st.plotly_chart(
            px.bar(region_ship,
                   x="destination_region",
                   y="weekly_shipping_need"),
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">Delay Risk Split</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            px.pie(opt_df,
                   names="logistics_risk",
                   hole=0.4),
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">Transit Time vs Delay Risk</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            px.scatter(
                opt_df,
                x="avg_transit_days",
                y="avg_delay_rate",
                color="logistics_risk",
                color_discrete_map={
                    "High Delay Risk": "red",
                    "Logistics Stable": "green"
                },
                hover_data=["product_id"]
            ),
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">Top Shipping Risks</div>',
            unsafe_allow_html=True
        )
        
        st.dataframe(
            opt_df.sort_values("avg_delay_rate",
                               ascending=False).head(5),
            use_container_width=True
        )

        # -------- Output Preview --------
        st.markdown(
            '<div class="section-title">Logistics Output Preview</div>',
            unsafe_allow_html=True
        )

        st.dataframe(opt_df, use_container_width=True)

        st.download_button(
            "Download Logistics Plan",
            opt_df.to_csv(index=False),
            "logistics_plan.csv"
        )
