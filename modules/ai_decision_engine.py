# ======================================================================================
# OmniFlow-D2D : AI Decision Intelligence Engine (Modern Dashboard)
# ======================================================================================
import os
import pandas as pd
import streamlit as st
import plotly.express as px
st.set_page_config(page_title="AI Decision Intelligence", layout="wide")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH  = os.path.join(DATA_DIR, "inventory_optimization.csv")
PRODUCTION_PATH = os.path.join(DATA_DIR, "production_plan.csv")
LOGISTICS_PATH  = os.path.join(DATA_DIR, "logistics_plan.csv")
def inject_css():
    st.markdown("""
    <style>   
    :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --text: #1f2937;
        --primary: #2563eb;
        --success: #16a34a;
        --danger: #dc2626;
        --warning: #f59e0b;
        --border: #e5e7eb;
    }
    body {
        background: linear-gradient(180deg, #f8fafc, #eef2ff);
    }
    .kpi-card {
        background: var(--card);
        padding:22px;
        border-radius:18px;
        text-align:center;
        box-shadow:0 6px 18px rgba(0,0,0,0.06);
    }
    .kpi-value {
        font-size:32px;
        font-weight:900;
    }
    .floating-card {
        background: #ffffff;
        padding:22px;
        border-radius:18px;
        border-left: 6px solid var(--primary);
        box-shadow:0 8px 24px rgba(0,0,0,0.08);
        text-align:center;
    }
    .section-title {
        font-size:22px;
        font-weight:700;
        margin-bottom:10px;
    }
    .action-item {
        background: #f1f5f9;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 6px 0;
        font-size: 14px;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 8px;
        border-left: 4px solid #2563eb;
    }
    </style>
    """, unsafe_allow_html=True)
def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
@st.cache_data
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
@st.cache_data
def compute_insights(forecast, inventory, production, logistics):
    avg_forecast = (forecast["forecast"].mean()
        if not forecast.empty and "forecast" in forecast.columns else 0
    )
    if (
        not forecast.empty and
        "product_id" in forecast.columns and"forecast" in forecast.columns
    ):
        high_demand = (forecast.groupby("product_id")["forecast"].mean()
            .sort_values(ascending=False).head(5))
    else:
        high_demand = pd.Series(dtype=float)
    risk_products = []  
    if not inventory.empty:  
        inventory.columns = inventory.columns.str.lower()   
        if {"product_id", "current_stock", "reorder_point"}.issubset(inventory.columns):
            risk_df = inventory[inventory["current_stock"]<= inventory["reorder_point"]]
            risk_products = risk_df["product_id"].tolist()
        elif "stock_status" in inventory.columns:  
            risk_df = inventory[inventory["stock_status"].astype(str).str.contains("critical|reorder", case=False, na=False)]
            risk_products = risk_df["product_id"].tolist()
    production_needed = [] 
    if (not production.empty and
        {"product_id","production_required"}.issubset(production.columns)
    ):   
        production_needed = production[production["production_required"] > 0]["product_id"].tolist()
    if not production_needed:
        production_needed = risk_products.copy()
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
    health_score = max(0,100-len(risk_products)* 8-len(production_needed)* 5-len(delay_regions)* 4)  
    bottleneck = "None"
    if len(risk_products) >= len(production_needed) and risk_products:
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
def product_decisions(forecast, inventory, production):
    decisions = []
    if forecast.empty:
        return pd.DataFrame()
    demand = (forecast.groupby("product_id")["forecast"].mean().reset_index(name="avg_demand"))
    stock = inventory.groupby("product_id")["current_stock"].sum().reset_index() \
        if "current_stock" in inventory.columns else pd.DataFrame()
    prod = production[["product_id","production_required"]] \
        if "production_required" in production.columns else pd.DataFrame()
    df = demand.merge(stock, on="product_id", how="left")
    df = df.merge(prod, on="product_id", how="left")
    df.fillna(0, inplace=True)
    df["decision"] = "Healthy"
    df.loc[df["current_stock"] < df["avg_demand"] * 3,"decision"] = "Replenish Inventory"
    df.loc[df["production_required"] > 0,"decision"] = "Increase Production"
    df.loc[
        (df["current_stock"] < df["avg_demand"]) & (df["production_required"] > 0),"decision"
    ] = "Urgent Production"
    return df.sort_values("avg_demand", ascending=False)
def decision_nlp(insights, q):
    q = q.lower()
    if "risk" in q:
        return f"Risk products: {insights['risk_products']}"
    if "production" in q:
        return f"Production required: {insights['production_needed']}"
    if "delay" in q:
        return f"Delay regions: {insights['delay_regions']}"
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
    if "summary" in q:
        return (
            f"Health Score {insights['health_score']}. "
            f"{len(insights['risk_products'])} products at risk."
        )
    for p in insights["risk_products"]:
        if p.lower() in q:
            return f"{p} is below reorder level and needs replenishment."
    if "recommend" in q:
        actions = []
        if insights["risk_products"]:
            actions.append("Replenish risky inventory.")
        if insights["production_needed"]:
            actions.append("Increase production capacity.")
        if insights["delay_regions"]:
            actions.append("Optimize logistics routes.")
        if not actions:
            actions.append("Operations stable.")
        return "\n".join(actions)
    return "Ask about risk, production, health, bottleneck, logistics or recommendations."
def generate_decision_summary(insights):
    actions = []
    if insights["risk_products"]:
        actions.append("Replenish critical inventory immediately.")
    if insights["production_needed"]:
        actions.append("Schedule urgent production.")
    if insights["delay_regions"]:
        actions.append("Investigate logistics delays.")
    if not actions:
        actions.append("Operations stable across supply chain.")
    return {
        "top_risk_product": insights["risk_products"][0]
            if insights["risk_products"] else "None",
        "recommended_actions": actions
    }
def decision_intelligence_page():
    inject_css()
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = load_data()   
    forecast, inventory, production, logistics = st.session_state["data_loaded"]
    insights = compute_insights(forecast, inventory, production, logistics)
    summary = generate_decision_summary(insights)
    st.title("🧠 AI Decision Intelligence")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    cols = st.columns(4)
    metrics = [
        ("Avg Forecast", int(insights["avg_forecast"])),
        ("Products at Risk", len(insights["risk_products"])),
        ("Production Needed", len(insights["production_needed"])),
        ("Delay Regions", len(insights["delay_regions"])),
    ]
    kpi_colors = {
        "Avg Forecast": "#2563eb",     
        "Products at Risk": "#dc2626", 
        "Production Needed": "#f59e0b",
        "Delay Regions": "#7c3aed"      
    }
    for col, (name, val) in zip(cols, metrics):
        with col:
            color = kpi_colors.get(name, "#2563eb")
            st.markdown(f"""
            <div class="kpi-card">
                <div>{name}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Product Decisions</div>', unsafe_allow_html=True)
    decision_df = product_decisions(forecast, inventory, production)  
    decision_df["color"] = decision_df["decision"].map({
        "Healthy": "#00CC96",
        "Replenish Inventory": "#FFA15A",
        "Increase Production": "#AB63FA",
        "Urgent Production": "#EF553B"
    })
    if not decision_df.empty:
        st.dataframe(decision_df, use_container_width=True)   
    st.markdown('<div class="section-title">Executive Decision Panel</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="floating-card">
        <h3>Top Risk Product</h3>
        <h2>{summary["top_risk_product"]}</h2>
        <p><b>Recommended Actions</b></p>
        <div style="text-align:left; margin-top:10px;">
            {''.join(f'<div class="action-item">✔ {a}</div>' for a in summary["recommended_actions"])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    health = insights["health_score"]
    if health > 75:
        color = "#16a34a"
    elif health > 50:
        color = "#f59e0b"
    else:
        color = "#dc2626"
    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)  
    hcol, bcol = st.columns(2)   
    with hcol:
        st.markdown(f"""
        <div class="floating-card">
            <h3>System Health</h3>
            <h1 style="color:{color}">{health}/100</h1>
        </div>
        """, unsafe_allow_html=True) 
    with bcol:
        st.markdown(f"""
        <div class="floating-card">
            <h3>Bottleneck</h3>
            <h1>{insights['bottleneck']}</h1>
        </div>
        """, unsafe_allow_html=True)        
    st.markdown('<div class="section-title">Performance Analytics</div>', unsafe_allow_html=True)    
    dcol, pcol = st.columns(2)    
    with dcol:
        st.markdown('<div class="section-title">Demand Leaders</div>', unsafe_allow_html=True)
        if isinstance(insights["high_demand"], pd.Series) and not insights["high_demand"].empty:
            fig1 = px.bar(
                insights["high_demand"].reset_index(), x="product_id", y="forecast",
                color_discrete_sequence=["#636EFA"] 
            )     
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No demand data available.")
        st.markdown('</div>', unsafe_allow_html=True)    
    with pcol:
        st.markdown('<div class="section-title">Production Pressure</div>', unsafe_allow_html=True)
        if (not production.empty and
            "product_id" in production.columns and "production_required" in production.columns
        ):
            fig2 = px.bar(
                production, x="product_id", y="production_required",
                color_discrete_sequence=["#EF553B"] 
            )   
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No production pressure detected.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Assistant</div>', unsafe_allow_html=True)
    q = st.text_input("Ask supply-chain questions")
    if q:
        st.success(decision_nlp(insights, q))
    st.markdown('<div class="section-title">API Output</div>', unsafe_allow_html=True)
    st.json(insights)
