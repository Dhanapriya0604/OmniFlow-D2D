import streamlit as st

st.set_page_config(
    page_title="OmniFlow D2D Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:       #0a0e1a;
  --surface:  #111827;
  --border:   #1e2d45;
  --accent:   #00e5ff;
  --accent2:  #ff6b35;
  --accent3:  #7c3aed;
  --text:     #e2e8f0;
  --muted:    #64748b;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

.stApp { background-color: var(--bg) !important; }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent3), var(--accent)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
}
.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent);
}
.metric-label {
  font-size: .8rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
}

div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}

.tag {
  display: inline-block;
  background: var(--accent3);
  color: white;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .06em;
}
.tag-green  { background: #059669; }
.tag-orange { background: #d97706; }
.tag-red    { background: #dc2626; }
.tag-blue   { background: #2563eb; }

.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
.stSelectbox label, .stSlider label { color: var(--text) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from modules import overview, demand, inventory, production, logistics, chatbot

PAGES = {
    "🏠 Overview":             overview,
    "📈 Demand Forecasting":   demand,
    "📦 Inventory Optimization": inventory,
    "🏭 Production Planning":  production,
    "🚚 Logistics Intelligence": logistics,
    "🤖 Decision Chatbot":     chatbot,
}

# Sidebar nav
st.sidebar.markdown("""
<div style='padding:16px 0 24px'>
  <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#00e5ff;'>OmniFlow</div>
  <div style='font-size:.75rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;'>D2D Supply Intelligence</div>
</div>
""", unsafe_allow_html=True)

selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:.72rem;color:#64748b;line-height:1.7'>
📅 Data: Jan 2024 – Dec 2025<br>
🔮 Forecast: to Jun 2026<br>
📊 5,200 orders<br>
🇮🇳 India D2D Supply Chain
</div>
""", unsafe_allow_html=True)

# Route
PAGES[selection].render()
