# ======================================================================================
# OmniFlow-D2D : AI Decision Intelligence Engine (Modern Dashboard)
# ======================================================================================

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Decision Intelligence", layout="wide")

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
# MODERN DASHBOARD CSS
# ======================================================================================
def inject_css():
    st.markdown("""
    <style>
    section.main > div {
        animation: fadeIn .4s ease;
    }
    @keyframes fadeIn {
        from {opacity:0; transform:translateY(10px);}
        to {opacity:1; transform:translateY(0);}
    }
    .kpi-card {
        background:white;
        padding:22px;
        border-radius:18px;
        text-align:center;
        box-shadow:0 12px 32px rgba(0,0,0,0.08);
        transition:0.3s;
    }
    .kpi-card:hover {
        transform:translateY(-6px);
        box-shadow:0 20px 50px rgba(0,0,0,0.15);
    }
    .kpi-value {
        font-size:34px;
        font-weight:900;
        color:#0284c7;
    }
    .panel {
        background:white;
        padding:20px;
        border-radius:16px;
        box-shadow:0 10px 30px rgba(0,0,0,0.08);
        margin-bottom:20px;
    }
    .section-title {
        font-size:22px;
        font-weight:700;
        margin-bottom:10px;
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
    return (
        safe_read(FORECAST_PATH),
        safe_read(INVENTORY_PATH),
        safe_read(PRODUCTION_PATH),
        safe_read(LOGISTICS_PATH),
    )

# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
@st.cache_data
def compute_insights(forecast, inventory, production, logistics):
    avg_forecast = forecast["forecast"].mean() if not forecast.empty else 0
    high_demand = (
        forecast.groupby("product_id")["forecast"].mean()
        .sort_values(ascending=False).head(5)
        if not forecast.empty else []
    )
    risk_products = (
        inventory.loc[
            inventory["stock_status"].str.contains("Critical|Reorder", na=False),"product_id"
        ].tolist()
        if "stock_status" in inventory.columns else []
    )
    production_needed = (
        production.loc[
            production["production_required"] > 0,"product_id"
        ].tolist()
        if "production_required" in production.columns else []
    )
    delay_regions = (
        logistics.loc[
            logistics["logistics_risk"] == "High Delay Risk","destination_region"
        ].unique().tolist()
        if "logistics_risk" in logistics.columns else []
    )
    total_products = forecast["product_id"].nunique() if not forecast.empty else 0
    risk_ratio = len(risk_products) / total_products if total_products else 0
    health_score = max(0, 100 - risk_ratio * 60 - len(delay_regions) * 5)
    bottleneck = "None"
    if risk_products:
        bottleneck = "Inventory"
    elif production_needed:
        bottleneck = "Production"
    elif delay_regions:
        bottleneck = "Logistics"
    return {
        "avg_forecast": avg_forecast,
        "high_demand": high_demand,
        "risk_products": risk_products,
        "production_needed": production_needed,
        "delay_regions": delay_regions,
        "health_score": health_score,
        "bottleneck": bottleneck,
        "total_products": total_products,
    }

# ======================================================================================
# NLP ASSISTANT
# ======================================================================================
def decision_nlp(insights, q):
    q = q.lower()
    if "risk" in q:
        return f"Risk products: {insights['risk_products']}"
    if "production" in q:
        return f"Production required: {insights['production_needed']}"
    if "delay" in q:
        return f"Delay regions: {insights['delay_regions']}"
    if "summary" in q:
        return (
            f"Health Score {insights['health_score']}. "
            f"{len(insights['risk_products'])} products at risk."
        )
    if "recommend" in q:
        actions = []
        if insights["risk_products"]:
            actions.append("Replenish risky inventory.")
        if insights["production_needed"]:
            actions.append("Increase production.")
        if insights["delay_regions"]:
            actions.append("Optimize logistics.")
        if not actions:
            actions.append("Operations stable.")
        return "\n".join(actions)
    return "Ask about risk, production, delay, summary or recommendations."

# ======================================================================================
# MAIN DASHBOARD PAGE
# ======================================================================================
def decision_intelligence_page():
    inject_css()
    forecast, inventory, production, logistics = load_data()
    insights = compute_insights(forecast, inventory, production, logistics)
    st.title("🧠 AI Decision Intelligence")

    # ================= KPI ROW =================
    cols = st.columns(4)
    metrics = [
        ("Avg Forecast", int(insights["avg_forecast"])),
        ("Products at Risk", len(insights["risk_products"])),
        ("Production Needed", len(insights["production_needed"])),
        ("Delay Regions", len(insights["delay_regions"])),
    ]
    for col, (name, val) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div>{name}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ================= MAIN PANELS =================
    left, right = st.columns([2,1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Demand Leaders")
        if isinstance(insights["high_demand"], pd.Series):
            st.bar_chart(insights["high_demand"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Production Pressure")
        if not production.empty:
            st.bar_chart(
                production.set_index("product_id")["production_required"]
            )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### System Health")
        st.metric("Health Score", f"{insights['health_score']}/100")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Bottleneck")
        st.info(insights["bottleneck"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= RISK OVERVIEW =================
    st.markdown("### Risk Overview")
    r1, r2, r3 = st.columns(3)
    r1.markdown(f"**Inventory Risk:** {insights['risk_products'] or 'None'}")
    r2.markdown(f"**Production Needed:** {insights['production_needed'] or 'None'}")
    r3.markdown(f"**Delay Regions:** {insights['delay_regions'] or 'None'}")

    # ================= AI ASSISTANT =================
    st.markdown("### AI Assistant")
    q = st.text_input("Ask supply-chain questions")
    if q:
        st.success(decision_nlp(insights, q))

    # ================= API OUTPUT =================
    st.markdown("### API Output")
    st.json(insights)
