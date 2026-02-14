# ======================================================================================
# OmniFlow-D2D : AI Decision Intelligence Engine (Corrected)
# ======================================================================================

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Decision Engine", layout="wide")

# ======================================================================================
# PATHS
# ======================================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

FORECAST_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH  = os.path.join(DATA_DIR, "inventory_optimization.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")
LOGISTICS_PATH  = os.path.join(DATA_DIR, "logistics_plan.csv")

# ======================================================================================
# UI CSS
# ======================================================================================
def inject_css():
    st.markdown("""
    <style>
    section.main > div {
        animation: fadeIn .5s ease;
    }

    @keyframes fadeIn {
        from {opacity:0; transform:translateY(10px);}
        to {opacity:1; transform:translateY(0);}
    }

    .title {
        font-size:32px;
        font-weight:800;
        margin-bottom:20px;
    }

    .kpi-card {
        background:white;
        padding:20px;
        border-radius:16px;
        text-align:center;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        transition:0.25s;
    }

    .kpi-card:hover {
        transform:translateY(-5px);
    }

    .kpi-value {
        font-size:30px;
        font-weight:900;
        color:#0284c7;
    }

    .section-title {
        font-size:22px;
        font-weight:700;
        margin-top:20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================================================================================
# DATA LOADING
# ======================================================================================
def safe_read(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        return df
    return pd.DataFrame()

def load_data():
    return (
        safe_read(FORECAST_PATH),
        safe_read(INVENTORY_PATH),
        safe_read(PRODUCTION_PATH),
        safe_read(LOGISTICS_PATH)
    )

# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
@st.cache_data
def compute_insights(forecast, inventory, production, logistics):

    # ---- Forecast ----
    if "forecast" in forecast.columns:
        avg_forecast = forecast["forecast"].mean()
    else:
        avg_forecast = 0

    if {"product_id", "forecast"}.issubset(forecast.columns):
        high_demand = (
            forecast.groupby("product_id")["forecast"]
            .mean()
            .nlargest(3)
            .index.tolist()
        )
        total_products = forecast["product_id"].nunique()
    else:
        high_demand = []
        total_products = 0

    # ---- Inventory Risk ----
    if {"stock_status", "product_id"}.issubset(inventory.columns):
        risk_products = inventory.loc[
            inventory["stock_status"].str.contains("critical|reorder", case=False, na=False),
            "product_id"
        ].tolist()
    else:
        risk_products = []

    # ---- Production Need ----
    if {"production_required", "product_id"}.issubset(production.columns):
        production_needed = production.loc[
            production["production_required"] > 0,
            "product_id"
        ].tolist()
    else:
        production_needed = []

    # ---- Logistics Delay ----
    if {"logistics_risk", "destination_region"}.issubset(logistics.columns):
        delay_regions = logistics.loc[
            logistics["logistics_risk"].str.contains("delay", case=False, na=False),
            "destination_region"
        ].unique().tolist()
    else:
        delay_regions = []

    # ---- Health score ----
    risk_ratio = len(risk_products) / total_products if total_products else 0

    health_score = max(
        0,
        100 - risk_ratio * 60 - len(delay_regions) * 5
    )

    # ---- Bottleneck ----
    if risk_ratio > 0.2:
        bottleneck = "Inventory"
    elif production_needed:
        bottleneck = "Production"
    elif delay_regions:
        bottleneck = "Logistics"
    else:
        bottleneck = "None"

    return {
        "avg_forecast": round(avg_forecast, 2),
        "high_demand": high_demand,
        "risk_products": risk_products,
        "production_needed": production_needed,
        "delay_regions": delay_regions,
        "health_score": round(health_score, 1),
        "bottleneck": bottleneck,
        "total_products": total_products
    }

# ======================================================================================
# NLP ASSISTANT
# ======================================================================================
def decision_nlp(insights, q):
    q = q.lower()

    if "risk" in q:
        return f"Products at risk: {insights['risk_products'] or 'None'}"

    if "production" in q:
        return f"Production needed: {insights['production_needed'] or 'None'}"

    if "delay" in q:
        return f"Delay regions: {insights['delay_regions'] or 'None'}"

    if "summary" in q:
        return (
            f"Health Score: {insights['health_score']}. "
            f"Bottleneck: {insights['bottleneck']}."
        )

    if "recommend" in q:
        actions = []
        if insights["risk_products"]:
            actions.append("Replenish critical inventory.")
        if insights["production_needed"]:
            actions.append("Increase production output.")
        if insights["delay_regions"]:
            actions.append("Optimize logistics routes.")
        if not actions:
            actions.append("Operations stable.")

        return "\n".join(actions)

    return "Ask: risk, production, delay, summary, recommend"

# ======================================================================================
# PAGE
# ======================================================================================
def decision_intelligence_page():

    inject_css()

    st.markdown('<div class="title">🧠 AI Decision Intelligence</div>', unsafe_allow_html=True)

    forecast, inventory, production, logistics = load_data()
    insights = compute_insights(forecast, inventory, production, logistics)

    # KPI Cards
    cols = st.columns(4)
    metrics = [
        ("Avg Forecast", int(insights["avg_forecast"])),
        ("Products at Risk", len(insights["risk_products"])),
        ("Production Needed", len(insights["production_needed"])),
        ("Delay Regions", len(insights["delay_regions"]))
    ]

    for col, (name, val) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div>{name}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Health Status
    score = insights["health_score"]

    if score > 80:
        st.success("🟢 Supply chain stable")
    elif score > 60:
        st.warning("🟠 Monitoring required")
    else:
        st.error("🔴 Immediate intervention needed")

    st.markdown(f"### System Health Score: {score}/100")
    st.warning(f"Bottleneck: {insights['bottleneck']}")

    # Insights
    st.markdown("### High Demand Products")
    st.write(insights["high_demand"] or "No demand spikes")

    st.markdown("### Inventory Risk Products")
    st.write(insights["risk_products"] or "No risk")

    st.markdown("### Production Needed")
    st.write(insights["production_needed"] or "None")

    st.markdown("### Logistics Delay Regions")
    st.write(insights["delay_regions"] or "None")

    # NLP Assistant
    st.markdown("### AI Assistant")
    q = st.text_input("Ask supply-chain questions")

    if q:
        st.success(decision_nlp(insights, q))

    # API Output
    st.markdown("### API Output")
    st.json(insights)
