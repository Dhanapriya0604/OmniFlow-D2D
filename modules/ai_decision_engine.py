# ======================================================================================
# OmniFlow-D2D : Decision Intelligence Module
# ======================================================================================

import os
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

    def safe_read(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    forecast = st.session_state.get(
        "all_forecasts",
        safe_read(FORECAST_PATH)
    )

    inventory = safe_read(INVENTORY_PATH)
    production = safe_read(PRODUCTION_PATH)
    logistics = safe_read(LOGISTICS_PATH)

    return forecast, inventory, production, logistics

# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
def compute_insights(forecast, inventory, production, logistics):

    # ---------- Forecast ----------
    if not forecast.empty and "forecast" in forecast.columns:
        avg_forecast = forecast["forecast"].mean()

        high_demand_products = (
            forecast.groupby("product_id")["forecast"]
            .mean()
            .nlargest(3)
            .index.tolist()
        )
    else:
        avg_forecast = 0
        high_demand_products = []

    # ---------- Inventory Risk ----------
    if (
        not inventory.empty and
        "stock_status" in inventory.columns and
        "product_id" in inventory.columns
    ):
        risk_products = inventory.loc[
            inventory["stock_status"].isin(
                ["🔴 Critical", "🟠 Reorder Required"]
            ),
            "product_id"
        ].tolist()
    else:
        risk_products = []

    # ---------- Production ----------
    if (
        not production.empty and
        "production_required" in production.columns and
        "product_id" in production.columns
    ):
        production_needed = production[
            production["production_required"] > 0
        ]["product_id"].tolist()

        production_load = production["production_required"].sum()
    else:
        production_needed = []
        production_load = 0

    # ---------- Logistics ----------
    if (
        not logistics.empty and
        "logistics_risk" in logistics.columns and
        "destination_region" in logistics.columns
    ):
        high_delay_regions = logistics[
            logistics["logistics_risk"] == "High Delay Risk"
        ]["destination_region"].unique().tolist()
    else:
        high_delay_regions = []

    if (
        not logistics.empty and
        "weekly_shipping_need" in logistics.columns
    ):
        shipping_load = logistics["weekly_shipping_need"].sum()
    
        if "avg_shipping_cost" in logistics.columns:
            shipping_cost = (
                logistics["weekly_shipping_need"] *
                logistics["avg_shipping_cost"]
            ).sum()
        else:
            shipping_cost = 0
    else:
        shipping_load = 0
        shipping_cost = 0

    # ---------- Product count ----------
    if "product_id" in forecast.columns:
        total_products = forecast["product_id"].nunique()
    else:
        total_products = 0

    risk_ratio = (
        len(risk_products) / total_products
        if total_products else 0
    )

    # ---------- Health Score ----------
    health_score = max(
        0,
        100 - (risk_ratio * 60 + len(high_delay_regions) * 5)
    )

    # ---------- Bottleneck ----------
    if risk_ratio > 0.2:
        bottleneck = "Inventory"
    elif production_load > shipping_load:
        bottleneck = "Production"
    elif shipping_load > production_load:
        bottleneck = "Logistics"
    else:
        bottleneck = "None"

    return {
        "avg_forecast": avg_forecast,
        "high_demand_products": high_demand_products,
        "risk_products": risk_products,
        "production_needed": production_needed,
        "delay_regions": high_delay_regions,
        "bottleneck": bottleneck,
        "total_products": total_products,
        "risk_ratio": risk_ratio,
        "production_load": production_load,
        "shipping_load": shipping_load,
        "shipping_cost": shipping_cost,
        "health_score": health_score
    }

def predict_future_risk(insights):
    future_risk = []
    if insights["risk_ratio"] > 0.2:
        future_risk.append("Inventory risk likely to increase")
    if insights["production_load"] > 50000:
        future_risk.append("Production overload risk")
    if insights["shipping_load"] > 20000:
        future_risk.append("Logistics congestion risk")
    if not future_risk:
        future_risk.append("Supply chain stable")

    return future_risk

# ======================================================================================
# NLP ASSISTANT
# ======================================================================================
def decision_nlp(insights, q):
    q = q.lower()
    # ---- demand ----
    if "high demand" in q:
        return f"High demand products: {', '.join(insights['high_demand_products'])}"
    # ---- inventory ----
    if "risk" in q or "stock" in q:
        if len(insights["risk_products"]) == 0:
            return "No products currently at stock risk."
        return f"Products at stock risk: {', '.join(insights['risk_products'])}"
    # ---- production ----
    if "production" in q:
        if len(insights["production_needed"]) == 0:
            return "Production is sufficient. No products require production."
        return f"Production required for: {', '.join(insights['production_needed'])}"
    # ---- logistics ----
    if "delay" in q or "logistics" in q:
        if len(insights["delay_regions"]) == 0:
            return "No logistics delay risks detected."
        return f"High delay risk regions: {', '.join(insights['delay_regions'])}"   
    if "improve" in q:
        return (
            "Improvement Areas:\n"
            "- Optimize safety stock levels\n"
            "- Improve forecast accuracy\n"
            "- Reduce carrier delay risks\n"
            "- Balance warehouse stock\n"
        )
    # ---- action recommendation ----
    if "action" in q or "recommend" in q:
        actions = []    
        if insights["risk_products"]:
            actions.append("Replenish stock for critical items.")    
        if insights["production_needed"]:
            actions.append("Increase production capacity.")    
        if insights["delay_regions"]:
            actions.append("Switch carriers or routes in delay regions.")   
        if insights["health_score"] < 70:
            actions.append("System health below safe threshold.")  
        if not actions:
            actions.append("Operations running smoothly.")
    
        return "Recommended Actions:\n- " + "\n- ".join(actions)
  
    # ---- summary ----
    if "summary" in q:
        return (
            f"Avg forecast demand: {insights['avg_forecast']:.0f}. "
            f"{len(insights['risk_products'])} products at risk, "
            f"{len(insights['production_needed'])} need production."
        )
    for product in insights["risk_products"]:
        if product.lower() in q:
            return f"{product} is at stock risk and requires action."
    
    for product in insights["production_needed"]:
        if product.lower() in q:
            return f"{product} requires production scheduling."
    if "executive summary" in q:
        return (
            f"System health score is {insights['health_score']}/100. "
            f"{len(insights['risk_products'])} products at risk. "
            f"{len(insights['production_needed'])} need production. "
            f"Logistics delays detected in {len(insights['delay_regions'])} regions. "
            f"Primary bottleneck: {insights['bottleneck']}."
        )

    return (
        "Try asking:\n"
        "- high demand products\n"
        "- stock risk\n"
        "- production needed\n"
        "- logistics delay\n"
        "- recommended action\n"
        "- summary"
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
    future_risk = predict_future_risk(insights)

    st.markdown("### Future Risk Prediction")
    
    st.write(future_risk)
    st.markdown("### Recommended Management Actions")
    actions = decision_nlp(insights, "recommend")
    st.info(actions)

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
    st.warning(f"System Bottleneck: {insights['bottleneck']}")
    st.write(f"### System Health Score: {insights['health_score']} / 100")
    
    if insights["health_score"] > 80:
        st.success("System performing well")
    elif insights["health_score"] > 60:
        st.warning("System needs monitoring")
    else:
        st.error("System requires intervention")
    
    st.markdown("### Executive Summary")
    st.info(decision_nlp(insights, "executive summary"))
    
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
    st.markdown("### API Response Output")

    api_response = {
        "health_score": insights["health_score"],
        "avg_forecast": insights["avg_forecast"],
        "total_products": insights["total_products"],
        "risk_ratio": insights["risk_ratio"],
        "production_load": insights["production_load"],
        "shipping_load": insights["shipping_load"],
        "risk_products": insights["risk_products"],
        "production_needed": insights["production_needed"],
        "delay_regions": insights["delay_regions"],
        "shipping_cost": insights["shipping_cost"],
        "future_risk": future_risk
    }
    
    st.json(api_response)
