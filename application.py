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
    st.markdown("""
    <div class="of-card">
        <div class="of-title">📦 OmniFlow D2D</div>
        <div class="of-subtitle">
            Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="of-card">
    <b>🔄 End-to-End AI-Powered Supply Chain Optimization System</b><br><br>
    📊 Demand → 📦 Inventory → ⚙️ Production → 🚚 Logistics → 🤖 AI Decisions
    <br><br>
    <span style="color:#475569;">Closed-loop intelligent system</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="grid">
        <div class="of-card">
        <b>📊 Demand Intelligence</b><br><br>
        ML forecasting using lag features, trends & seasonality
        </div>
        <div class="of-card">
        <b>📦 Inventory Optimization</b><br><br>
        Prevents stockouts & maintains optimal stock levels
        </div>
        <div class="of-card">
        <b>⚙️ Production Planning</b><br><br>
        Converts demand into production with capacity planning
        </div>
        <div class="of-card">
        <b>🚚 Logistics Optimization</b><br><br>
        Shipment planning, delay prediction & cost estimation
        </div>
        <div class="of-card">
        <b>🤖 AI Decision Engine</b><br><br>
        Detects risks & recommends business actions
        </div>
        <div class="of-card">
        <b>⚡ Key Capabilities</b><br><br>
        ✔ Forecasting<br>
        ✔ Optimization<br>
        ✔ Scheduling<br>
        ✔ Risk Detection<br>
        ✔ AI Decisions
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="grid">
        <div class="of-card">
        <b>🎯 Business Impact</b><br><br>
        ✔ Reduce stockouts & overstock<br>
        ✔ Align production with demand<br>
        ✔ Optimize logistics cost<br>
        ✔ Improve visibility
        </div>
        <div class="of-card">
        <b>🧠 Tech Stack</b><br><br>
        Python • Pandas • NumPy<br>
        Scikit-learn • ML Models<br>
        Streamlit • Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.success("🚀 Forecast → Plan → Execute → Decide — Unified Intelligence System")
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
