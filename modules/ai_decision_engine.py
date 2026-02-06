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
    :root {
        --bg:#f8fafc;
        --text:#0f172a;
        --primary:#0ea5e9;
        --border:#e5e7eb;
    }

    section.main > div {
        animation: fadeIn .4s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity:0; transform:translateY(6px);}
        to {opacity:1; transform:translateY(0);}
    }

    .metric-card {
        background:white;
        border-radius:16px;
        padding:18px;
        text-align:center;
        border:1px solid var(--border);
        box-shadow:0 6px 18px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size:28px;
        font-weight:900;
        color:var(--primary);
    }

    .section-title {
        font-size:26px;
        font-weight:800;
        margin:24px 0 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================================================================================
# PATHS
# ======================================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FORECAST_PATH = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory_optimization.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")
LOGISTICS_PATH = os.path.join(DATA_DIR, "logistics_plan.csv")

# ======================================================================================
# LOADERS
# ======================================================================================
@st.cache_data
def load_data():
    forecast = pd.read_csv(FORECAST_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    production = pd.read_csv(PRODUCTION_PATH)
    logistics = pd.read_csv(LOGISTICS_PATH)

    forecast.columns = forecast.columns.str.lower()
    inventory.columns = inventory.columns.str.lower()
    production.columns = production.columns.str.lower()
    logistics.columns = logistics.columns.str.lower()

    return forecast, inventory, production, logistics

# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
def compute_insights(forecast, inventory, production, logistics):

    avg_forecast = forecast["forecast"].mean()

    high_demand_products = (
        forecast.groupby("product_id")["forecast"]
        .mean()
        .nlargest(3)
        .index.tolist()
    )

    risk_products = inventory[
        inventory["stock_status"].isin(
            ["🔴 Critical", "🟠 Reorder Required"]
        )
    ]["product_id"].tolist()

    production_needed = production[
        production["production_required"] > 0
    ]["product_id"].tolist()

    high_delay_regions = logistics[
        logistics["logistics_risk"] == "High Delay Risk"
    ]["destination_region"].unique().tolist()

    return {
        "avg_forecast": avg_forecast,
        "high_demand_products": high_demand_products,
        "risk_products": risk_products,
        "production_needed": production_needed,
        "delay_regions": high_delay_regions,
    }

# ======================================================================================
# NLP ASSISTANT
# ======================================================================================
def decision_nlp(insights, q):

    q = q.lower()

    if "high demand" in q:
        return f"High demand products: {', '.join(insights['high_demand_products'])}"

    if "risk" in q or "stock" in q:
        return f"Products at stock risk: {', '.join(insights['risk_products'])}"

    if "production" in q:
        return f"Products needing production: {', '.join(insights['production_needed'])}"

    if "delay" in q or "logistics" in q:
        return f"High delay risk regions: {', '.join(insights['delay_regions'])}"

    if "average forecast" in q:
        return f"Average demand forecast is {insights['avg_forecast']:.0f} units."

    return (
        "Ask questions like:\n"
        "- high demand products\n"
        "- stock risk products\n"
        "- production needed\n"
        "- logistics delay regions\n"
        "- average forecast"
    )

# ======================================================================================
# PAGE
# ======================================================================================
def decision_intelligence_page():

    inject_css()

    st.markdown(
        '<div class="section-title">Decision Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    forecast, inventory, production, logistics = load_data()
    insights = compute_insights(
        forecast, inventory, production, logistics
    )

    # ================= KPIs =================
    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Avg Forecast",
         int(insights["avg_forecast"])),

        ("Products at Risk",
         len(insights["risk_products"])),

        ("Production Needed",
         len(insights["production_needed"])),

        ("Delay Regions",
         len(insights["delay_regions"]))
    ]

    for col, (k, v) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div>{k}</div>
                <div class="metric-value">{v}</div>
            </div>
            """, unsafe_allow_html=True)

    # ================= INSIGHTS =================
    st.markdown(
        '<div class="section-title">System Insights</div>',
        unsafe_allow_html=True
    )

    st.write("### High Demand Products")
    st.write(insights["high_demand_products"])

    st.write("### Inventory Risk Products")
    st.write(insights["risk_products"])

    st.write("### Production Required")
    st.write(insights["production_needed"])

    st.write("### Logistics Delay Regions")
    st.write(insights["delay_regions"])

    # ================= NLP =================
    st.markdown(
        '<div class="section-title">AI Decision Assistant</div>',
        unsafe_allow_html=True
    )

    q = st.text_input("Ask supply-chain questions")

    if q:
        st.success(decision_nlp(insights, q))
