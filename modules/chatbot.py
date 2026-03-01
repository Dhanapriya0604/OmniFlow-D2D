"""Decision Chatbot — AI-powered supply chain Q&A using Anthropic API."""
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from .data_loader import load_data
from .demand import forecast_series

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def build_context(df: pd.DataFrame) -> str:
    """Build a concise supply chain context string for the LLM."""
    # Key stats
    total_rev    = df["Revenue_INR"].sum()
    total_orders = len(df)
    return_rate  = df["Return_Flag"].mean() * 100
    avg_del      = df["Delivery_Days"].mean()

    # Category breakdown
    cat_rev = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
    cat_str = ", ".join([f"{k}: ₹{v/1e6:.1f}M" for k,v in cat_rev.items()])

    # Carrier performance
    carrier_stats = df.groupby("Courier_Partner").agg(
        orders=("Order_ID","count"),
        avg_del=("Delivery_Days","mean"),
        return_rate=("Return_Flag","mean")
    )
    carr_str = "; ".join([
        f"{r}: {d['orders']} orders, {d['avg_del']:.1f}d avg, {d['return_rate']*100:.1f}% returns"
        for r, d in carrier_stats.iterrows()
    ])

    # Region top 5
    top_reg = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    reg_str = ", ".join([f"{k}: ₹{v/1e6:.1f}M" for k,v in top_reg.items()])

    # Demand forecast
    m_qty = df.groupby("YearMonth")["Quantity"].sum().rename("value")
    fore  = forecast_series(m_qty, 6)
    future = fore[fore["type"]=="forecast"]
    fore_str = "; ".join([
        f"{row['ds'].strftime('%b %Y')}: {row['y']:.0f} units"
        for _, row in future.iterrows()
    ])

    # Top SKUs
    top_sku = df.groupby("Product_Name")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
    sku_str = ", ".join(top_sku.index.tolist())

    # Warehouse
    wh_rev = df.groupby("Warehouse")["Revenue_INR"].sum().sort_values(ascending=False)
    wh_str = ", ".join([f"{k}: ₹{v/1e6:.1f}M" for k,v in wh_rev.items()])

    return f"""
=== OmniFlow D2D India Supply Chain Intelligence Context ===
Dataset: 5,200 orders, Jan 2024 – Dec 2025, India D2D

KEY METRICS:
- Total Revenue: ₹{total_rev/1e7:.2f} Crore
- Total Orders: {total_orders:,}
- Return Rate: {return_rate:.1f}%
- Avg Delivery: {avg_del:.1f} days

CATEGORY REVENUE: {cat_str}

TOP REGIONS: {reg_str}

CARRIER PERFORMANCE: {carr_str}

WAREHOUSE REVENUE: {wh_str}

TOP PRODUCTS: {sku_str}

DEMAND FORECAST (next 6 months): {fore_str}
"""


def call_claude(messages: list, system: str) -> str:
    """Call Anthropic API and return text response."""
    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 1000,
                "system": system,
                "messages": messages,
            }
        )
        data = resp.json()
        if "content" in data:
            return "".join(block.get("text","") for block in data["content"] if block.get("type")=="text")
        elif "error" in data:
            return f"⚠️ API Error: {data['error'].get('message', 'Unknown error')}"
        return "⚠️ Unexpected response format."
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"


SUGGESTIONS = [
    "Which carrier should I use for Maharashtra orders?",
    "What products will have highest demand in April 2026?",
    "Which regions have critical stock risk?",
    "How should I adjust production for June 2026?",
    "What's the best warehouse for Electronics shipments?",
    "Which category is growing fastest and why?",
    "How can we reduce our return rate?",
    "What's the optimal reorder strategy for Home & Kitchen?",
]


def render():
    df = load_data()

    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f59e0b'>
        🤖 Decision Intelligence Chatbot
      </div>
      <div style='color:#64748b;font-size:.9rem'>
        Ask anything about your supply chain · powered by Claude AI · context-aware decisions
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Context bar ────────────────────────────────────────────────────────────
    context_str = build_context(df)

    with st.expander("📊 Live Data Context (fed to AI)", expanded=False):
        st.code(context_str, language="text")

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # System prompt
    system_prompt = f"""You are OmniFlow, an expert supply chain decision intelligence assistant for an India D2D e-commerce operation. 
You have deep expertise in demand forecasting, inventory management, production planning, and logistics optimization.
Always give specific, actionable recommendations backed by the data provided.
Be concise but comprehensive. Use bullet points for lists. Use ₹ for Indian Rupees.
Reference specific numbers from the context when relevant.

Here is the current supply chain data context:
{context_str}

When answering:
- Reference specific metrics and numbers from the context
- Give concrete, actionable recommendations
- If forecasting, mention confidence levels
- Consider interdependencies between modules (demand → inventory → production → logistics)
- Flag risks and opportunities
"""

    # ── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("<div class='section-title'>💡 Suggested Questions</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, suggestion in enumerate(SUGGESTIONS):
            with cols[i % 4]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":suggestion})
                    with st.spinner("Thinking..."):
                        reply = call_claude(
                            [{"role":"user","content":suggestion}],
                            system_prompt
                        )
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.rerun()

    # ── Chat display ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='display:flex;justify-content:flex-end;margin:12px 0'>
              <div style='background:#1e2d45;border:1px solid #2d3f55;border-radius:12px 12px 2px 12px;
                          padding:12px 18px;max-width:75%;color:#e2e8f0;font-size:.9rem'>
                {msg['content']}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:flex;justify-content:flex-start;margin:12px 0;gap:10px'>
              <div style='width:32px;height:32px;border-radius:50%;
                          background:linear-gradient(135deg,#7c3aed,#00e5ff);
                          display:flex;align-items:center;justify-content:center;
                          font-size:.9rem;flex-shrink:0'>🤖</div>
              <div style='background:#111827;border:1px solid #1e2d45;
                          border-radius:2px 12px 12px 12px;
                          padding:14px 18px;max-width:78%;color:#e2e8f0;font-size:.9rem;
                          line-height:1.7'>
                {msg['content'].replace(chr(10), '<br>')}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_inp, col_btn, col_clr = st.columns([5,1,1])
    with col_inp:
        user_input = st.text_input(
            "Ask a supply chain question…",
            key="chat_input",
            placeholder="e.g., What production adjustments do I need for Q2 2026?",
            label_visibility="collapsed"
        )
    with col_btn:
        send = st.button("Send 🚀", use_container_width=True)
    with col_clr:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role":"user","content":user_input.strip()})
        # Build messages for multi-turn context (last 10 turns)
        api_msgs = st.session_state.chat_history[-10:]
        with st.spinner("OmniFlow AI is thinking…"):
            reply = call_claude(api_msgs, system_prompt)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

    # ── Quick stats panel ─────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Quick Decision Metrics</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Action Items**")
            # High return regions
            ret_reg = df.groupby("Region")["Return_Flag"].mean().sort_values(ascending=False).head(3)
            for reg, rate in ret_reg.items():
                color = "#dc2626" if rate > 0.2 else "#f59e0b"
                st.markdown(f"""<div style='background:#111827;border-left:3px solid {color};
                    padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0;font-size:.85rem'>
                    ⚠️ <b>{reg}</b> — {rate*100:.1f}% return rate — review carrier/product quality
                    </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("**Forecast Alerts**")
            m_rev = df.groupby("YearMonth")["Revenue_INR"].sum().rename("value")
            fore = forecast_series(m_rev, 3)
            future = fore[fore["type"]=="forecast"]
            last_hist = float(m_rev.iloc[-1])
            for _, row in future.iterrows():
                chg = (row["y"] - last_hist) / last_hist * 100
                clr = "#10b981" if chg >= 0 else "#dc2626"
                icon = "📈" if chg >= 0 else "📉"
                st.markdown(f"""<div style='background:#111827;border-left:3px solid {clr};
                    padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0;font-size:.85rem'>
                    {icon} <b>{row['ds'].strftime("%b %Y")}</b> — ₹{row['y']/1e6:.1f}M forecast
                    ({chg:+.1f}% vs last month)
                    </div>""", unsafe_allow_html=True)
                last_hist = row["y"]
