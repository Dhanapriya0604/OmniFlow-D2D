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
    forecast = safe_read(FORECAST_PATH)
    inventory = safe_read(INVENTORY_PATH)
    production = safe_read(PRODUCTION_PATH)
    logistics = safe_read(LOGISTICS_PATH)
    for df in [forecast, inventory, production, logistics]:
        if not df.empty:
            df.columns = df.columns.astype(str)
            df.columns = df.columns.str.lower().str.strip()
    return forecast, inventory, production, logistics
# ======================================================================================
# INSIGHT ENGINE
# ======================================================================================
@st.cache_data
def compute_insights(forecast, inventory, production, logistics):
    avg_forecast = (
        forecast["forecast"].mean()
        if not forecast.empty and "forecast" in forecast.columns else 0
    )
    if (
        not forecast.empty and
        "product_id" in forecast.columns and"forecast" in forecast.columns
    ):
        high_demand = (forecast.groupby("product_id")["forecast"].mean()
            .sort_values(ascending=False).head(5)
        )
    else:
        high_demand = pd.Series(dtype=float)
    # ================= INVENTORY RISK =================
    risk_products = []  
    if not inventory.empty:  
        inventory.columns = inventory.columns.str.lower()   
        if {"product_id", "current_stock", "reorder_point"}.issubset(inventory.columns):
            risk_df = inventory[
                inventory["current_stock"]<= inventory["reorder_point"]
            ]
            risk_products = risk_df["product_id"].tolist()
        elif "stock_status" in inventory.columns:  
            risk_df = inventory[
                inventory["stock_status"].astype(str)
                .str.contains("critical|reorder", case=False, na=False)
            ]
            risk_products = risk_df["product_id"].tolist()

    # ================= PRODUCTION NEED =================
    production_needed = [] 
    if (not production.empty and
        {"product_id","production_required"}.issubset(production.columns)
    ):   
        production_needed = production[production["production_required"] > 0]["product_id"].tolist()
    if not production_needed:
        production_needed = risk_products.copy()
        
   # ================= LOGISTICS RISK =================
    delay_regions = []   
    if not logistics.empty:  
        logistics.columns = logistics.columns.str.lower()   
        if "avg_delay_rate" in logistics.columns:   
            delay_regions = logistics[logistics["avg_delay_rate"] > 0.15]["destination_region"].unique().tolist() 
        elif "delay_flag" in logistics.columns:
            region_delay = (logistics.groupby("destination_region")["delay_flag"].mean().reset_index())
            delay_regions = region_delay[region_delay["delay_flag"] > 0.2]["destination_region"].tolist()

    total_products = forecast["product_id"].nunique() if not forecast.empty else 0
    risk_ratio = len(risk_products) / total_products if total_products else 0
    health_score = max(0,100- len(risk_products) * 12 - len(production_needed) * 6 - len(delay_regions) * 5)
    
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
# MANAGEMENT ACTION ENGINE
# ======================================================================================
def generate_management_actions(insights):
    actions = []
    if insights["risk_products"]:
        for p in insights["risk_products"]:
            actions.append(f"Replenish inventory immediately for product {p}.")
    if insights["production_needed"]:
        for p in insights["production_needed"]:
            actions.append(f"Schedule urgent production for product {p}.")
    if insights["delay_regions"]:
        for r in insights["delay_regions"]:
            actions.append(f"Optimize logistics or change carrier in {r} region.")
    if not actions:
        actions.append("Supply chain operating normally. Continue monitoring.")
    return actions
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
        if "health" in q:
            return f"System health score is {insights['health_score']}."        
        if "bottleneck" in q:
            return f"Current bottleneck is {insights['bottleneck']}."            
        if "high demand" in q:
            return f"Top demand products: {list(insights['high_demand'].index)}"            
        if "inventory status" in q:
            return f"{len(insights['risk_products'])} products below reorder level."            
        if "production load" in q:
            return f"{len(insights['production_needed'])} products need production."           
        if "logistics status" in q:
            return f"{len(insights['delay_regions'])} regions have logistics delay risk."

    return "Ask about risk, production, delay, summary or recommendations."

# ======================================================================================
# MAIN DASHBOARD PAGE
# ======================================================================================
def decision_intelligence_page():
    inject_css()
    forecast, inventory, production, logistics = load_data()
    insights = compute_insights(forecast, inventory, production, logistics)
    actions = generate_management_actions(insights)
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
        if (not production.empty and
            "product_id" in production.columns and"production_required" in production.columns
        ):
            st.bar_chart(production.set_index("product_id")["production_required"])
        else:
            st.info("No production pressure data available.")
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
    st.markdown("### Recommended Management Actions")
    for act in actions:
        st.info(act)
    # ================= RISK OVERVIEW =================
    st.markdown("### Risk Overview")
    r1, r2, r3 = st.columns(3)
    r1.markdown("Inventory Risk:" +
        (", ".join(insights["risk_products"]) if insights["risk_products"] else "None")
    )
    r2.markdown("Production Needed:" +
        (", ".join(insights["production_needed"]) if insights["production_needed"] else "None")
    )
    r3.markdown("Delay Regions:" +
        (", ".join(insights["delay_regions"]) if insights["delay_regions"] else "None")
    )

    # ================= AI ASSISTANT =================
    st.markdown("### AI Assistant")
    q = st.text_input("Ask supply-chain questions")
    if q:
        st.success(decision_nlp(insights, q))

    # ================= API OUTPUT =================
    st.markdown("### API Output")
    st.json(insights)
