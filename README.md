# OmniFlow D2D Intelligence Platform

AI-powered supply chain analytics for India D2D e-commerce.

## 📁 Project Structure
```
omniflow_app/
├── app.py                          # Main entry point
├── requirements.txt
├── OmniFlow_D2D_India_Unified_5200.csv   # ← place CSV here
└── modules/
    ├── __init__.py
    ├── data_loader.py              # Shared cached data loader
    ├── overview.py                 # Overview + KPI dashboard
    ├── demand.py                   # Demand forecasting to Jun 2026
    ├── inventory.py                # EOQ + Safety Stock optimisation
    ├── production.py               # Production planning
    ├── logistics.py                # Carrier, delay, warehouse analytics
    └── chatbot.py                  # AI decision chatbot (Claude API)
```

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place data file
Copy `OmniFlow_D2D_India_Unified_5200.csv` into the `omniflow_app/` folder.

### 3. Run the app
```bash
cd omniflow_app
streamlit run app.py
```

The app will open at http://localhost:8501

## 📊 Module Overview

| Module | Description | Depends On |
|--------|-------------|------------|
| Overview | Project KPIs, revenue trends, module flow | Raw data |
| Demand Forecasting | Linear trend + seasonal forecast to Jun 2026 | Raw data |
| Inventory Optimization | EOQ, Safety Stock, Reorder Points | Demand forecast |
| Production Planning | Monthly production targets + gap analysis | Demand + Inventory |
| Logistics Intelligence | Carrier perf, delay hotspots, warehouse forecast | Raw data + Demand |
| Decision Chatbot | Claude-powered Q&A with live SC context | All modules |

## 🤖 Chatbot Notes
The chatbot uses the Anthropic Claude API via the claude.ai environment proxy.
It automatically builds a live context snapshot from all 5,200 orders + forecasts.

## 🎨 Design
- Dark theme: deep navy (#0a0e1a) with cyan (#00e5ff) and violet (#7c3aed) accents
- Fonts: Syne (headers) + DM Sans (body)
- All charts: Plotly with transparent backgrounds
