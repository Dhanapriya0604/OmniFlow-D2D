# ======================================================================================
# OmniFlow-D2D : AI Decision Intelligence Engine (NEW)
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
# MODERN UI CSS
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
        transform:translateY(-6px);
        box-shadow:0 16px 30px rgba(0,0,0,0.15);
    }

    .kpi-value {
        font-size:30px;
        font-weight:900;
        color:#0284c7;
    }

    .health-good {color:#059669;}
    .health-warn {color:#d97706;}
    .health-bad  {color:#dc2626;}

    .section-title {
        font-size:24px;
        font-weight:700;
        margin-top:25px;
    }

    </style>
    """, unsafe_allow_html=True)

# ======================================================================================
# DATA LOADER
# ======================================================================================
def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_data():
    forecast   = safe_read(FORECAST_PATH)
    inventory  = safe_read(INVENTORY_PATH)
    production = safe_read(PRODUCTION_PATH)
    logistics  = safe_read(LOGISTICS_PATH)
    return forecast, inventory, production, logistics

# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
def compute_insights(forecast, inventory, production, logistics):

    avg_forecast = forecast["forecast"].mean() if not forecast.empty else 0

    high_demand = (
        forecast.groupby("product_id")["forecast"]
        .mean()
        .nlargest(3)
        .index.tolist()
        if not forecast.empty else []
    )

    risk_products = (
        inventory.loc[
            inventory["stock_status"].isin(
                ["🔴 Critical", "🟠 Reorder Required", "Critical"]
            ),
            "product_id"
        ].tolist()
        if "stock_status" in inventory.columns else []
    )

    production_needed = (
        production.loc[
            production["production_required"] > 0,
            "product_id"
        ].tolist()
        if "production_required" in production.columns else []
    )

    delay_regions = (
        logistics.loc[
            logistics["logistics_risk"] == "High Delay Risk",
            "destination_region"
        ].unique().tolist()
        if "logistics_risk" in logistics.columns else []
    )

    total_products = forecast["product_id"].nunique() if not forecast.empty else 0

    risk_ratio = len(risk_products) / total_products if total_products else 0

    health_score = max(0, 100 - risk_ratio*60 - len(delay_regions)*5)

    bottleneck = "None"
    if risk_ratio > 0.2:
        bottleneck = "Inventory"
    elif len(production_needed) > 0:
        bottleneck = "Production"
    elif len(delay_regions) > 0:
        bottleneck = "Logistics"

    return {
        "avg_forecast": avg_forecast,
        "high_demand": high_demand,
        "risk_products": risk_products,
        "production_needed": production_needed,
        "delay_regions": delay_regions,
        "health_score": health_score,
        "bottleneck": bottleneck,
        "total_products": total_products
    }

# ======================================================================================
# NLP ASSISTANT
# ======================================================================================
def decision_nlp(insights, q):

    q = q.lower()

    if "risk" in q:
        return f"Products at risk: {insights['risk_products']}"

    if "production" in q:
        return f"Production needed for: {insights['production_needed']}"

    if "delay" in q:
        return f"Delay regions: {insights['delay_regions']}"

    if "summary" in q:
        return (
            f"Health Score: {insights['health_score']}. "
            f"{len(insights['risk_products'])} products at risk."
        )

    if "recommend" in q:
        actions = []
        if insights["risk_products"]:
            actions.append("Replenish stock immediately.")
        if insights["production_needed"]:
            actions.append("Increase production capacity.")
        if insights["delay_regions"]:
            actions.append("Change logistics routes.")
        if not actions:
            actions.append("Operations stable.")

        return "\n".join(actions)

    return "Try: risk, production, delay, summary, recommend"

# ======================================================================================
# MAIN PAGE
# ======================================================================================
def decision_intelligence_page():

    inject_css()

    st.markdown('<div class="title">🧠 AI Decision Intelligence</div>', unsafe_allow_html=True)

    forecast, inventory, production, logistics = load_data()
    insights = compute_insights(forecast, inventory, production, logistics)

    # KPI CARDS
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

    # HEALTH STATUS
    score = insights["health_score"]

    color = "health-good" if score > 80 else \
            "health-warn" if score > 60 else \
            "health-bad"

    st.markdown(f"""
    <div class="section-title">System Health</div>
    <h2 class="{color}">{score}/100</h2>
    """, unsafe_allow_html=True)

    st.warning(f"Bottleneck: {insights['bottleneck']}")

    # INSIGHTS
    st.markdown("### High Demand Products")
    st.write(insights["high_demand"])

    st.markdown("### Inventory Risk Products")
    st.write(insights["risk_products"])

    st.markdown("### Production Needed")
    st.write(insights["production_needed"])

    st.markdown("### Delay Regions")
    st.write(insights["delay_regions"])

    # NLP
    st.markdown("### AI Assistant")
    q = st.text_input("Ask supply chain questions")

    if q:
        st.success(decision_nlp(insights, q))

    # API Output
    st.markdown("### API Output")
    st.json(insights)
