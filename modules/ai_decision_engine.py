# ======================================================================================
# OmniFlow-D2D : Decision Intelligence Module
# ======================================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Decision Intelligence", layout="wide")

# ======================================================================================
# CSS UI
# ======================================================================================
def inject_css():
    st.markdown("""
    <style>
    .section-title {
        font-size:28px;
        font-weight:800;
        margin:20px 0;
    }

    .insight-card {
        background:linear-gradient(180deg,#eef2ff,#ffffff);
        padding:20px;
        border-radius:14px;
        box-shadow:0 8px 20px rgba(0,0,0,0.08);
        margin-bottom:12px;
        transition:0.2s;
    }

    .insight-card:hover {
        transform:translateY(-3px);
        box-shadow:0 14px 30px rgba(0,0,0,0.15);
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
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")


# ======================================================================================
# LOADERS
# ======================================================================================
def load_forecasts():
    if "all_forecasts" in st.session_state:
        return st.session_state["all_forecasts"]
    return pd.read_csv(FORECAST_PATH)


def load_inventory():
    df = pd.read_csv(INVENTORY_PATH)
    df.columns = df.columns.str.lower()
    return df


def load_production():
    if os.path.exists(PRODUCTION_PATH):
        df = pd.read_csv(PRODUCTION_PATH)
        df.columns = df.columns.str.lower()
        return df
    return pd.DataFrame()


# ======================================================================================
# DECISION ENGINE
# ======================================================================================
def decision_engine(forecast_df, inv_df, prod_df):

    # ---- Product name mapping ----
    if "product_name" in inv_df.columns:
        product_map = inv_df[
            ["product_id", "product_name"]
        ].drop_duplicates()
    else:
        product_map = forecast_df[
            ["product_id"]
        ].drop_duplicates()
        product_map["product_name"] = product_map["product_id"]

    # ---- Demand analysis ----
    demand_avg = (
        forecast_df.groupby("product_id")["forecast"]
        .mean()
        .reset_index(name="avg_forecast")
    )

    demand_avg = demand_avg.merge(
        product_map,
        on="product_id",
        how="left"
    )

    high = demand_avg.sort_values(
        "avg_forecast", ascending=False
    ).iloc[0]

    low = demand_avg.sort_values(
        "avg_forecast"
    ).iloc[0]

    # Demand category
    demand_avg["category"] = pd.qcut(
        demand_avg["avg_forecast"],
        3,
        labels=["Low", "Medium", "High"]
    )

    # ---- Inventory risk ----
    if "stock_status" in inv_df.columns:
        risk_products = inv_df[
            inv_df["stock_status"].isin(
                ["🔴 Critical", "🟠 Reorder Required"]
            )
        ][["product_id"]].drop_duplicates()

        risk_products = risk_products.merge(
            product_map,
            on="product_id",
            how="left"
        )

        risk_products = (
            risk_products["product_name"]
            + " (" + risk_products["product_id"] + ")"
        ).tolist()
    else:
        risk_products = []

    # ---- Production need ----
    if not prod_df.empty:
        prod_needed = prod_df[
            prod_df["production_required"] > 0
        ][["product_id"]]

        prod_needed = prod_needed.merge(
            product_map,
            on="product_id",
            how="left"
        )

        prod_needed = (
            prod_needed["product_name"]
            + " (" + prod_needed["product_id"] + ")"
        ).tolist()
    else:
        prod_needed = []

    # ---- Model metrics ----
    if {"model", "rmse"}.issubset(forecast_df.columns):
        model_perf = (
            forecast_df.groupby("model")["rmse"]
            .mean()
            .reset_index()
        )
        best_model = model_perf.sort_values(
            "rmse"
        ).iloc[0]["model"]
    else:
        best_model = "Random Forest"

    insights = [
        f"📈 Highest demand expected for {high['product_name']} ({high['product_id']})",
        f"📉 Lowest demand observed for {low['product_name']} ({low['product_id']})",
        f"⚠ Stock risk products: {', '.join(risk_products[:5]) if risk_products else 'None'}",
        f"🏭 Production required for: {', '.join(prod_needed[:5]) if prod_needed else 'None'}",
        f"🤖 Best forecasting model: {best_model}",
        "🚚 Recommendation: Increase supply for high-demand and risky stock items."
    ]
    return insights

# ======================================================================================
# PAGE
# ======================================================================================
def decision_intelligence_page():

    inject_css()

    st.markdown(
        '<div class="section-title">Decision Intelligence</div>',
        unsafe_allow_html=True
    )

    forecast_df = load_forecasts()
    inv_df = load_inventory()
    prod_df = load_production()

    insights = decision_engine(
        forecast_df,
        inv_df,
        prod_df
    )

    for text in insights:
        st.markdown(
            f"<div class='insight-card'>{text}</div>",
            unsafe_allow_html=True
        )
