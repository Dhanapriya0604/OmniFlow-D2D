# application.py
# OmniFlow-D2D : Streamlit Application (MODULE-BASED)
import os
import sys
import streamlit as st
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from modules.demand_forecasting import demand_forecasting_page
from modules.inventory_optimization import inventory_optimization_page
from modules.production_planning import production_planning_page
from modules.logistics_prediction import logistics_optimization_page
from modules.ai_decision_engine import decision_intelligence_page
def inject_global_css():
    st.markdown("""
    <style>
    :root {
        --bg: #f8fafc;
        --text: #0f172a;
        --muted: #475569;
        --primary: #1e3a8a;
        --border: #e5e7eb;
        --card-bg: #ffffff;
    }
    html, body {
        background-color: var(--bg);
        color: var(--text);
        font-family: Inter, system-ui, -apple-system;
    }
    section.main > div {
        animation: fadeIn 0.45s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 28px 0 14px 0;
    }
    .of-card {
        background: var(--card-bg);
        padding: 22px;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 26px rgba(0,0,0,0.08);
        transition: all 0.25s ease;
        margin-bottom: 18px;
    }
    .of-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 38px rgba(0,0,0,0.14);
    }
    .of-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .of-subtitle {
        color: var(--muted);
        font-size: 15px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9, #ffffff);
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 18px;
    }
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)
inject_global_css()

def show_overview():
    st.title("OmniFlow D2D")
    st.subheader("Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System")
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    OmniFlow D2D is an End-to-End AI-Powered Supply Chain Optimization System that connects demand forecasting,
    inventory optimization, production planning, and logistics into one unified flow.
    It transforms raw data into actionable decisions for faster and smarter operations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">End-to-End Flow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    📊 Demand Forecasting → 📦 Inventory Optimization → ⚙️ Production Planning → 🚚 Logistics Optimization → 🤖 AI Decision Engine
    <br><br>
    Each module feeds the next, forming a closed-loop intelligent system.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Core Modules</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    <ul>
        <li><b>Demand Intelligence:</b> ML-based forecasting with trends & seasonality</li>
        <li><b>Inventory Optimization:</b> Prevents stockouts and optimizes stock levels</li>
        <li><b>Production Planning:</b> Converts demand into production plans</li>
        <li><b>Predictive Logistics:</b> Optimizes shipments and detects delays</li>
        <li><b>AI Decision Engine:</b> Identifies risks and recommends actions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    <ul>
        <li>Feature engineering (lags, trends, seasonality)</li>
        <li>Model selection using RMSE</li>
        <li>Automated production scheduling</li>
        <li>Logistics risk detection</li>
        <li>AI-based decision recommendations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    <ul>
        <li>Reduce stockouts & overstock</li>
        <li>Align production with demand</li>
        <li>Optimize logistics cost & delivery</li>
        <li>Improve supply chain visibility</li>
        <li>Enable faster decision-making</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="of-card">
    Python • Pandas • NumPy • Scikit-learn<br><br>
    Machine Learning Models • Optimization Logic<br><br>
    Streamlit • Plotly • Data Pipelines
    </div>
    """, unsafe_allow_html=True)
    st.success("🚀 Fully Integrated Demand-to-Delivery Intelligence System")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Demand Intelligence",
        "Inventory Optimization",
        "Production Planning",
        "Predictive Logistics",
        "AI Insights"
    ]
)
if menu == "Overview":
    show_overview()
elif menu == "Demand Intelligence":
    st.header("📊 Demand Forecasting Intelligence")
    demand_forecasting_page()
elif menu == "Inventory Optimization":
    st.header("📦 Inventory Optimization")
    inventory_optimization_page()
elif menu == "Production Planning":
    st.header("⚙️ Production Planning")
    production_planning_page()
elif menu == "Predictive Logistics":
    st.header("🚚 Predictive Logistics")
    logistics_optimization_page()
elif menu == "AI Insights":
    st.header("🤖 AI Decision Engine")
    decision_intelligence_page()
