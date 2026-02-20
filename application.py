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
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)
inject_global_css()
st.title("OmniFlow D2D")
st.subheader("Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System")
def show_overview():

    st.markdown("""<div class="of-card">
        <div class="of-title">📦 OmniFlow D2D</div>
        <div class="of-subtitle">
            End-to-End AI-Powered Supply Chain Optimization System
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="of-card">
    <b>What this system does</b><br><br>
    OmniFlow D2D transforms raw business data into actionable supply chain decisions.
    It predicts demand, optimizes inventory, plans production, improves logistics,
    and finally generates AI-driven recommendations — all in one unified platform.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="of-card">
    <b>End-to-End Flow</b><br><br>

    📊 Demand Forecasting → 📦 Inventory Optimization → ⚙️ Production Planning → 🚚 Logistics Optimization → 🤖 AI Decision Engine

    <br>
    Each module feeds the next, forming a closed-loop intelligent system.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="of-card">
    <b>Core Modules</b><br><br>

    📊 <b>Demand Intelligence</b><br>
    ML-based forecasting using lag features, rolling averages, seasonality, and model selection (RF, GBM, Linear)

    <br><br>

    📦 <b>Inventory Optimization</b><br>
    Uses demand + stock signals to prevent stockouts and maintain optimal inventory levels

    <br><br>

    ⚙️ <b>Production Planning</b><br>
    Converts demand into production requirements with capacity, batch sizing, and scheduling

    <br><br>

    🚚 <b>Predictive Logistics</b><br>
    Optimizes shipments, assigns regions, estimates delays, and calculates shipping cost

    <br><br>

    🤖 <b>AI Decision Intelligence</b><br>
    Identifies risks, bottlenecks, and recommends business actions dynamically
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="of-card">
    <b>What makes it intelligent</b><br><br>

    ✔ Feature engineering (lags, rolling trends, seasonality)  
    ✔ Model selection using RMSE  
    ✔ Forecast confidence intervals  
    ✔ Automated production scheduling  
    ✔ Logistics risk detection  
    ✔ AI-based decision recommendations  
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="of-card">
    <b>Business Impact</b><br><br>

    ✔ Reduce stockouts & overstock  
    ✔ Align production with real demand  
    ✔ Optimize logistics cost & delivery  
    ✔ Improve supply chain visibility  
    ✔ Enable faster data-driven decisions  
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="of-card">
    <b>Technology Stack</b><br><br>

    Python • Pandas • NumPy • Scikit-learn  
    Machine Learning Models • Optimization Logic  
    Streamlit • Plotly • Data Pipelines  
    </div>""", unsafe_allow_html=True)
    st.success("🚀 From Forecast → Plan → Execute → Decide — Fully Connected Intelligence System")
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
