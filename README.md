# OmniFlow D2D Intelligence

**AI-Driven Demand to Delivery Intelligence System**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧭 Project Overview

OmniFlow D2D Intelligence is an end-to-end supply chain analytics platform that simulates a real-world India e-commerce operation — from customer demand signals all the way through to last-mile delivery. It is designed as a **decision-support system** for supply chain managers, analysts, and business stakeholders.

The platform helps users:

- **Forecast demand** at the SKU level using Ridge Regression, Random Forest, Gradient Boosting, and a custom Ensemble model with consistent hold-out evaluation
- **Optimise inventory** using dynamic EOQ with per-SKU holding cost tiers, seasonal safety stock buffers, and forward Reorder Point (ROP) alerts to prevent stockouts
- **Plan production** using a pull-forward strategy that shifts capacity to meet demand peaks while preserving total monthly production integrity
- **Monitor logistics** by scoring carrier performance, identifying regional delay hotspots, and forecasting warehouse utilisation

Built on a synthetic dataset of **5,200+ orders, 50 SKUs, 9 regions, and 4 warehouses**, deployed on Streamlit Community Cloud.

---

## 📁 Project Structure

```
OmniFlow-D2D/
├── application.py                            # Main Streamlit entry point
├── requirements.txt                          # Python dependencies
├── OmniFlow_D2D_India_Unified_5200.csv       # Primary dataset (5,200+ orders)
└── india_ecommerce_orders.csv                # Supporting dataset
```

---

## 📊 Module Overview

| Module | Description | Depends On |
|--------|-------------|------------|
| Overview | Project KPIs, revenue trends, module flow | Raw data |
| Demand Forecasting | Ridge, Random Forest, Gradient Boosting & Ensemble forecasting | Raw data |
| Inventory Optimization | EOQ, Safety Stock, Reorder Points, ABC classification | Demand forecast |
| Production Planning | Pull-forward planning, monthly targets, gap analysis | Demand + Inventory |
| Logistics Intelligence | Carrier performance, delay hotspots, warehouse forecast | Raw data + Demand |

---

## ✨ Key Features

- **ML Forecasting Pipeline** — Ridge Regression, Random Forest, Gradient Boosting, and custom Ensemble with consistent hold-out evaluation and a global forecast horizon selector
- **Universal SCM Schema Mapping** — Compatible with any standard supply chain dataset via configurable column mapping
- **Dynamic EOQ** — Per-SKU holding cost tiers for accurate inventory cost modelling
- **Seasonal Safety Stock** — Demand variability-aware buffer stock calculations
- **Forward ROP Alerts** — Proactive reorder point notifications before stockouts occur
- **Decision Recommendations Engine** — Auto-generated SKU-level action items (reorder, expedite, hold)
- **Pull-Forward Production Planning** — Preserves total production integrity while shifting capacity to meet demand peaks

---

## 🗂️ Dataset

Synthetic India e-commerce supply chain dataset:
- **5,200+ orders**
- **50 SKUs**
- **9 regions**
- **4 warehouses**

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| ML Models | Scikit-learn (Ridge, Random Forest, Gradient Boosting, Ensemble) |
| Visualisation | Plotly, Matplotlib |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |

---

## 📈 Model Performance

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Ridge Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| Ensemble | 0.984 | 41.8 | 32.1 |

*(Fill in individual model hold-out results)*

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/Dhanapriya0604/OmniFlow-D2D.git
cd OmniFlow-D2D
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run application.py
```

The app will open at **http://localhost:8501**

---

