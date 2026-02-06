# application.py
# OmniFlow-D2D : Streamlit Application (MODULE-BASED)

import os
import sys
import streamlit as st

# --------------------------------------------------
# ENSURE PROJECT ROOT IS IN PYTHON PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --------------------------------------------------
# CORRECT MODULE IMPORTS
# --------------------------------------------------
from modules.demand_forecasting import demand_forecasting_page
from modules.inventory_optimization import inventory_optimization_page
from modules.production_planning import production_planning_page
from modules.logistics_prediction import logistics_optimization_page
from modules.ai_decision_engine import decision_intelligence_page

# --------------------------------------------------
# GLOBAL UI THEME (CONSISTENT WITH MODULES)
# --------------------------------------------------
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

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="OmniFlow D2D",
    page_icon="📦",
    layout="wide"
)

inject_global_css()

st.title("OmniFlow D2D")
st.subheader("Predictive Logistics & AI-Powered Demand-to-Delivery Optimization System")

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------
def show_overview():

    st.markdown("""
    <div class="of-card">
        <div class="of-title">📘 OmniFlow D2D – System Overview</div>
        <div class="of-subtitle">
            Predictive Logistics & AI-Powered Demand-to-Delivery Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1️⃣ ABSTRACT
    with st.expander("1️⃣ Abstract", expanded=True):
        st.markdown("""
        <div class="of-card">
        <b>OmniFlow D2D</b> is an AI-driven, end-to-end supply chain optimization platform
        designed to bridge the gap between market demand and final product delivery.

        The system integrates <b>demand forecasting, inventory optimization,
        production planning, predictive logistics, and AI-based decision intelligence</b>
        into a single unified application.

        Traditional supply chain systems operate in silos.
        OmniFlow D2D eliminates these silos by ensuring real-time demand signals
        drive procurement, production, inventory, and transportation decisions.

        The platform operates as a <b>closed-loop intelligence system</b>.
        </div>
        """, unsafe_allow_html=True)

    # 2️⃣ TOOLS
    with st.expander("2️⃣ Tools & Technologies Used"):
        st.markdown("""
        <div class="of-card">
        <b>Programming & Analytics</b>
        <ul>
            <li>Python</li>
            <li>Pandas & NumPy</li>
        </ul>

        <b>Machine Learning</b>
        <ul>
            <li>Scikit-Learn</li>
            <li>Prophet</li>
            <li>SARIMA</li>
        </ul>

        <b>Optimization & Deployment</b>
        <ul>
            <li>EOQ, Safety Stock, Reorder Point</li>
            <li>Streamlit</li>
            <li>GitHub & Streamlit Cloud</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # 3️⃣ DATA SOURCES
    with st.expander("3️⃣ Data Sources Used"):
        st.markdown("""
        <div class="of-card">
        The system operates on enterprise-style operational data that reflects
        real-world business processes across the supply chain lifecycle.
    
        <ul>
            <li>Historical demand and sales behavior</li>
            <li>Inventory levels and stock movement information</li>
            <li>Manufacturing and production performance records</li>
            <li>Transportation and delivery operations data</li>
        </ul>
    
        These data inputs enable analytical modeling, forecasting, optimization,
        and decision intelligence across multiple functional areas.
        </div>
        """, unsafe_allow_html=True)

    # 4️⃣ MODULES
    with st.expander("4️⃣ System Modules & Functionality"):
        st.markdown("""
        <div class="of-card">
        <ul>
            <li>Demand Forecasting</li>
            <li>Inventory Optimization</li>
            <li>Production Planning</li>
            <li>Predictive Logistics</li>
            <li>AI Decision Intelligence</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # 5️⃣ BENEFITS
    with st.expander("5️⃣ What Can Be Done Using OmniFlow D2D"):
        st.markdown("""
        <div class="of-card">
        <ul>
            <li>Accurate demand forecasting</li>
            <li>Reduced stockouts & wastage</li>
            <li>Optimized production schedules</li>
            <li>Lower logistics costs</li>
            <li>AI-driven decision support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # 6️⃣ CHALLENGES
    with st.expander("6️⃣ Challenges & Solutions"):
        st.table({
            "Challenge": [
                "Stockouts & Overstocks",
                "Logistics Delays",
                "High Costs",
                "Poor Visibility"
            ],
            "Solution": [
                "Demand Forecasting & Inventory Optimization",
                "Predictive Logistics",
                "Optimization Models",
                "Real-time Dashboards"
            ]
        })

    # 7️⃣ USERS
    with st.expander("7️⃣ Who This Project Is Useful For"):
        st.markdown("""
        <div class="of-card">
        <b>Enterprise Roles</b>
        <ul>
            <li>Supply Chain Managers</li>
            <li>Inventory Managers</li>
            <li>Production Planners</li>
            <li>Logistics Coordinators</li>
            <li>Data Scientists</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.success("🔁 Demand → Inventory → Production → Logistics → AI Decisions")

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
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

# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------
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
    ai_insights()
