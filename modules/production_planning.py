# ======================================================================================
# OmniFlow-D2D : Production Planning Module
# ======================================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Production Planning", layout="wide")

# ======================================================================================
# CSS UI
# ======================================================================================
def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg: #f8fafc;
        --text: #0f172a;
        --muted: #475569;
        --primary: #b45309;
        --accent: #fef3c7;
        --border: #e5e7eb;
    }

    html, body {
        background: var(--bg);
        color: var(--text);
        font-family: Inter;
    }

    section.main > div {
        animation: fadeIn 0.4s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity:0; transform:translateY(6px);}
        to {opacity:1; transform:translateY(0);}
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 28px 0 12px 0;
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
        background: linear-gradient(180deg,#fef3c7,#ffffff);
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(180,83,9,0.25);
        transition: 0.25s;
    }

    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 36px rgba(180,83,9,0.35);
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
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent);
        color: var(--primary);
        box-shadow: 0 6px 18px rgba(180,83,9,0.25);
    }
    </style>
    """, unsafe_allow_html=True)

# ======================================================================================
# PATHS
# ======================================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FORECAST_PATH = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH = os.path.join(DATA_DIR, "retail_inventory_snapshot.csv")
MANUFACTURING_PATH = os.path.join(DATA_DIR, "manufacturing_production_orders.csv")

# ======================================================================================
# LOAD DATA
# ======================================================================================
@st.cache_data
def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data
def load_manufacturing():
    df = pd.read_csv(MANUFACTURING_PATH)
    df.columns = df.columns.str.lower()
    return df


def load_forecasts():
    if "all_forecasts" in st.session_state:
        return st.session_state["all_forecasts"]
    return pd.read_csv(FORECAST_PATH)

# ======================================================================================
# PRODUCTION LOGIC
# ======================================================================================
def production_planning(forecast_df, inventory_df, manufacturing_df):
    # Use next 14 days demand only
    forecast_df = forecast_df.sort_values(["product_id", "date"])
    
    demand = (
        forecast_df
        .groupby("product_id")
        .apply(lambda x: x.head(14))
        .reset_index(drop=True)
        .groupby("product_id")["forecast"]
        .mean()
        .reset_index(name="avg_daily_demand")
    )

    demand["monthly_demand"] = demand["avg_daily_demand"] * 30

    inv = (
        inventory_df.groupby("product_id", as_index=False)
        .agg(current_stock=("on_hand_qty", "sum"))
    )

    df = demand.merge(inv, on="product_id", how="left")     
    monthly_need = df["avg_daily_demand"] * 30

    df["production_required"] = np.maximum(
        0,
        monthly_need - df["current_stock"]
    )
    mfg_agg = (
        manufacturing_df
        .groupby("product_id", as_index=False)
        .agg(
            batch_size=("planned_qty", "mean")
        )
    )    
    df = df.merge(mfg_agg, on="product_id", how="left")    
    df["batch_size"] = df["batch_size"].fillna(100)
    
    df["daily_capacity"] = df["batch_size"]    
    df["max_possible_production"] = df["daily_capacity"] * 30

    df["production_required"] = np.minimum(
        df["production_required"],
        df["max_possible_production"]
    )
    # keep 10% capacity buffer
    df["production_required"] *= 0.9

    df["production_required"] = np.where(
        df["production_required"] < df["batch_size"] * 0.6,
        0,
        df["production_required"]
    )

    df["production_required"] = np.ceil(df["production_required"])

    df["production_batches"] = np.ceil(
        df["production_required"] / df["batch_size"]
    )
    df["backlog"] = np.maximum(
        0,
        df["monthly_demand"] - df["current_stock"] - df["max_possible_production"]
    )
    df["days_required"] = np.ceil(df["production_required"] / df["daily_capacity"])
    
    df["production_status"] = np.where(
        df["production_required"] > 0,
        "⚠ Production Needed",
        "✅ Stock Sufficient"
    )
    # Production priority score
    df["production_priority"] = (
        df["production_required"] +
        df["backlog"] * 1.5
    )
    
    df = df.sort_values("production_priority", ascending=False)

    df["current_stock"] = df["current_stock"].fillna(0)

    return df
PRODUCTION_LINES = [
    {"line": "Line-1", "capacity": 800},
    {"line": "Line-2", "capacity": 700},
    {"line": "Line-3", "capacity": 600},
]

def auto_production_schedule(prod_df):

    today = pd.Timestamp.today().normalize()

    work_df = prod_df.copy()
    work_df["remaining"] = work_df["production_required"]

    schedule_rows = []
    day = 0

    while work_df["remaining"].sum() > 0:

        day_capacity = work_df["daily_capacity"].sum()
        active = work_df[work_df["remaining"] > 0]

        if len(active) == 0:
            break

        share = day_capacity / len(active)

        for idx, row in active.iterrows():

            produce_today = min(share, row["remaining"])

            schedule_rows.append({
                "date": today + pd.Timedelta(days=day),
                "product_id": row["product_id"],
                "production_qty": round(produce_today)
            })

            work_df.loc[idx, "remaining"] -= produce_today

        day += 1

    return pd.DataFrame(schedule_rows)
def allocate_production_lines(schedule_df):

    schedule_rows = []

    for date, day_df in schedule_df.groupby("date"):

        remaining = day_df.copy()

        for line_info in PRODUCTION_LINES:

            line_name = line_info["line"]
            capacity = line_info["capacity"]

            cap_left = capacity

            for idx, row in remaining.iterrows():

                if cap_left <= 0:
                    break

                qty = min(row["production_qty"], cap_left)

                schedule_rows.append({
                    "date": date,
                    "line": line_name,
                    "product_id": row["product_id"],
                    "production_qty": qty
                })

                remaining.loc[idx, "production_qty"] -= qty
                cap_left -= qty

        remaining = remaining[remaining["production_qty"] > 0]

    return pd.DataFrame(schedule_rows)

# ======================================================================================
# DATA DICTIONARY
# ======================================================================================
PRODUCTION_DATA_DICTIONARY = pd.DataFrame({
    "Column": [
        "product_id",
        "avg_daily_demand",
        "monthly_demand",
        "current_stock",
        "production_required",
        "batch_size",
        "production_batches",
        "daily_capacity",
        "days_required",
        "production_status"
    ],
    "Description": [
        "Unique product identifier",
        "Average forecasted daily demand",
        "Monthly demand estimate",
        "Current inventory level",
        "Units required to meet demand",
        "Production batch size",
        "Number of batches needed",
        "Daily manufacturing capacity",
        "Days required to complete production",
        "Production decision status"
    ]
})
# ======================================================================================
# DATA PROFILING
# ======================================================================================
def production_profiling(df):
    return {
        "Total Products": df["product_id"].nunique(),
        "Total Production Required": int(df["production_required"].sum()),
        "Avg Production Days": int(df["days_required"].mean()),
        "Products Below Reorder":
            int((df["production_required"] > 0).sum())
    }

# ======================================================================================
# MAIN PAGE
# ======================================================================================
def production_planning_page():

    inject_css()

    tab1, tab2 = st.tabs(["📘 Overview", "🏭 Application"])

    # -------- OVERVIEW --------
    with tab1:
        st.markdown('<div class="section-title">Production Planning Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        Production planning ensures manufacturing capacity meets future demand
        while minimizing idle capacity and stock shortages.
        </div>
        """, unsafe_allow_html=True)

    # -------- APPLICATION --------
    with tab2:

        forecast_df = load_forecasts()
        inventory_df = load_inventory()
        manufacturing_df = load_manufacturing()        

        prod_df = production_planning(
            forecast_df,
            inventory_df,
            manufacturing_df
        )
        prod_df = prod_df.sort_values("production_required", ascending=False)

        schedule_df = auto_production_schedule(prod_df)
        line_schedule_df = allocate_production_lines(schedule_df)
        with st.expander("📘 Data Dictionary"):
            st.dataframe(PRODUCTION_DATA_DICTIONARY, use_container_width=True)

        with st.expander("🔍 Data Profiling Summary"):
            profile = production_profiling(prod_df)
            for k, v in profile.items():
                st.write(f"**{k}:** {v}")
        if prod_df["backlog"].sum() == 0:
            st.info(
                "Production capacity is sufficient. No backlog detected."
            )
        else:
            st.warning(
                "Backlog detected. Production capacity insufficient for demand."
            )

        # KPIs
        st.markdown('<div class="section-title">Production KPIs</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        metrics = [
            ("Total Production Required",
             int(prod_df["production_required"].sum())),
            ("Avg Batches",
             int(prod_df["production_batches"].mean())),
            ("Products Needing Production",
             int((prod_df["production_required"] > 0).sum()))
        ]

        for col, (k, v) in zip([c1, c2, c3], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{k}</div>
                    <div class="metric-value">{v}</div>
                </div>
                """, unsafe_allow_html=True)

        # Charts
        st.markdown(
            '<div class="section-title">Demand vs Current Stock</div>',
            unsafe_allow_html=True
        )
        
        fig_ds = px.bar(
            prod_df,
            x="product_id",
            y=["monthly_demand", "current_stock"],
            barmode="group",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_ds, use_container_width=True)
        
        st.markdown(
            '<div class="section-title">Production Days Required per Product</div>',
            unsafe_allow_html=True
        )
        
        fig_days = px.bar(
            prod_df,
            x="product_id",
            y="days_required",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_days, use_container_width=True)

        st.markdown(
            '<div class="section-title">Production Status Split</div>',
            unsafe_allow_html=True
        )
        
        fig_status = px.pie(
            prod_df,
            names="production_status",
            hole=0.4,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_status, use_container_width=True)

        st.markdown('<div class="section-title">Production Requirement</div>', unsafe_allow_html=True)
        st.plotly_chart(
            px.bar(prod_df,
                   x="product_id",
                   y="production_required",
                   color="production_status",
                   template="plotly_white"),
            use_container_width=True
        )

        st.markdown('<div class="section-title">Capacity Usage</div>', unsafe_allow_html=True)
        st.plotly_chart(
            px.scatter(
                prod_df,
                x="daily_capacity",
                y="production_required",
                color="production_status",
                color_discrete_map={
                    "⚠ Production Needed": "red",
                    "✅ Stock Sufficient": "green"
                },
                hover_data=["product_id"],
                template="plotly_white"
            ),
            use_container_width=True
        )

        st.markdown('<div class="section-title">Production Plan Output</div>', unsafe_allow_html=True)
        st.dataframe(
            prod_df[[
                "product_id",
                "production_required",
                "daily_capacity",
                "days_required",
                "production_status"
            ]],
            use_container_width=True
        )
        st.markdown(
            '<div class="section-title">Auto Production Schedule</div>',
            unsafe_allow_html=True
        )
        st.dataframe(schedule_df, use_container_width=True)
        st.markdown(
            '<div class="section-title">Production Line Allocation</div>',
            unsafe_allow_html=True
        )
        
        st.dataframe(line_schedule_df, use_container_width=True)
        # Output
        st.download_button(
            "Download Production Plan",
            prod_df.to_csv(index=False),
            "production_plan.csv"
        )
        st.download_button(
            "Download Production Schedule",
            schedule_df.to_csv(index=False),
            "production_schedule.csv"
        )
        st.download_button(
            "Download Line Allocation",
            line_schedule_df.to_csv(index=False),
            "line_allocation.csv"
        )


