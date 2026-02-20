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
        min-height: 120px;
    }
    .of-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 38px rgba(0,0,0,0.14);
    }
    .of-card span {
        font-size: 13px;
        font-weight: 500;
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
    st.markdown("""<div class="of-card">
    OmniFlow D2D is an <b>end-to-end AI-powered supply chain intelligence system</b> that connects
    demand forecasting, inventory optimization, production planning, and logistics into one unified pipeline.
    It transforms raw operational data into <b>real-time, actionable decisions</b>.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="of-card">
            <div class="of-title">🔄 End-to-End Flow</div>
            <div style="font-weight:600; margin-top:10px;">
                📊 Demand → 📦 Inventory → ⚙️ Production → 🚚 Logistics → 🤖 AI
            </div>
            <div class="of-subtitle" style="margin-top:12px;">
                Closed-loop system where each module continuously feeds the next.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="of-card">
            <div class="of-title">⚙️ Technology Stack</div>
            <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;">
                <span style="background:#eef2ff; padding:6px 12px; border-radius:10px;">Python</span>
                <span style="background:#eef2ff; padding:6px 12px; border-radius:10px;">Pandas</span>
                <span style="background:#eef2ff; padding:6px 12px; border-radius:10px;">NumPy</span>
                <span style="background:#eef2ff; padding:6px 12px; border-radius:10px;">Scikit-learn</span>
                <span style="background:#f0fdf4; padding:6px 12px; border-radius:10px;">ML Models</span>
                <span style="background:#f0fdf4; padding:6px 12px; border-radius:10px;">Optimization</span>
                <span style="background:#ecfeff; padding:6px 12px; border-radius:10px;">Streamlit</span>
                <span style="background:#ecfeff; padding:6px 12px; border-radius:10px;">Plotly</span>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Core Modules</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class="of-card">
        <b>📊 Demand Intelligence</b><br><br>
        ML-based forecasting using trends, seasonality, and feature engineering.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="of-card">
        <b>⚙️ Production Planning</b><br><br>
        Converts demand signals into optimized production schedules.
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="of-card">
        <b>📦 Inventory Optimization</b><br><br>
        Maintains optimal stock levels and prevents stockouts.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="of-card">
        <b>🚚 Predictive Logistics</b><br><br>
        Optimizes shipments, routes, and delay risks.
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="of-card">
        <b>🤖 AI Decision Engine</b><br><br>
        Detects risks, bottlenecks, and recommends actions.
        </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="of-card">
            <div class="of-title">🧠 Key Capabilities</div>
            <ul style="padding-left:18px; line-height:1.8;">
                <li>Advanced feature engineering</li>
                <li>Model selection using RMSE</li>
                <li>Forecast confidence intervals</li>
                <li>Automated production planning</li>
                <li>Logistics risk detection</li>
                <li>AI-driven recommendations</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="of-card">
            <div class="of-title">📈 Business Impact</div>
            <ul style="padding-left:18px; line-height:1.8;">
                <li>Reduce stockouts & overstock</li>
                <li>Align production with demand</li>
                <li>Optimize logistics cost</li>
                <li>Improve visibility</li>
                <li>Enable faster decisions</li>
            </ul>
        </div>""", unsafe_allow_html=True)
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
