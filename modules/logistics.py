"""Logistics Intelligence — carrier performance, delay analysis, warehouse shipment forecast."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .data_loader import load_data
from .demand import forecast_series

COLORS = ["#00e5ff","#ff6b35","#7c3aed","#10b981","#f59e0b","#3b82f6","#ec4899"]


def render():
    df = load_data()

    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#ff6b35'>
        🚚 Logistics Intelligence
      </div>
      <div style='color:#64748b;font-size:.9rem'>
        Carrier performance · delay hotspots · warehouse demand · future logistics forecast
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🚛 Carrier Analysis", "⚠️ Delay Intelligence",
                     "🏪 Warehouse Forecast", "📍 Region Heatmap"])

    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:  # CARRIER ANALYSIS
        st.markdown("<div class='section-title'>Carrier Performance Scorecard</div>",
                    unsafe_allow_html=True)

        carrier_stats = df.groupby("Courier_Partner").agg(
            Orders         = ("Order_ID", "count"),
            Avg_Delivery   = ("Delivery_Days", "mean"),
            Avg_Cost       = ("Shipping_Cost_INR", "mean"),
            Return_Rate    = ("Return_Flag", "mean"),
            Revenue_Served = ("Revenue_INR", "sum"),
        ).reset_index()
        carrier_stats["Delay_Index"] = (
            (carrier_stats["Avg_Delivery"] / carrier_stats["Avg_Delivery"].min()) *
            (1 + carrier_stats["Return_Rate"])
        ).round(2)
        carrier_stats["Best_For"] = carrier_stats.apply(
            lambda r: "Speed" if r["Avg_Delivery"] == carrier_stats["Avg_Delivery"].min()
                      else ("Cost" if r["Avg_Cost"] == carrier_stats["Avg_Cost"].min()
                            else "Volume"), axis=1
        )

        # Bubble chart: Orders vs Avg Delivery vs Cost
        fig = go.Figure()
        for i, row in carrier_stats.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Avg_Delivery"]], y=[row["Avg_Cost"]],
                mode="markers+text",
                marker=dict(size=row["Orders"]/50, color=COLORS[i%len(COLORS)], opacity=0.75),
                text=[row["Courier_Partner"]],
                textposition="top center",
                name=row["Courier_Partner"],
                hovertemplate=(
                    f"<b>{row['Courier_Partner']}</b><br>"
                    f"Orders: {row['Orders']}<br>"
                    f"Avg Delivery: {row['Avg_Delivery']:.1f}d<br>"
                    f"Avg Cost: ₹{row['Avg_Cost']:.1f}<br>"
                    f"Return Rate: {row['Return_Rate']*100:.1f}%"
                )
            ))
        fig.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=340, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title="Avg Delivery Days", showgrid=True, gridcolor="#1e2d45"),
            yaxis=dict(title="Avg Shipping Cost (₹)", showgrid=True, gridcolor="#1e2d45"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Carrier scorecard table
        disp = carrier_stats[["Courier_Partner","Orders","Avg_Delivery","Avg_Cost",
                               "Return_Rate","Delay_Index","Best_For"]].copy()
        disp["Avg_Delivery"]   = disp["Avg_Delivery"].round(1)
        disp["Avg_Cost"]       = disp["Avg_Cost"].round(1)
        disp["Return_Rate"]    = (disp["Return_Rate"]*100).round(1).astype(str) + "%"
        disp.columns = ["Carrier","Orders","Avg Days","Avg Cost ₹",
                        "Return Rate","Delay Index","Best For"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Carrier monthly trend
        st.markdown("<div class='section-title'>Carrier Orders Trend</div>", unsafe_allow_html=True)
        carr_monthly = df.groupby([df["Order_Date"].dt.to_period("M"),"Courier_Partner"])["Order_ID"].count().unstack(fill_value=0)
        fig2 = go.Figure()
        for i, carrier in enumerate(carr_monthly.columns):
            fig2.add_trace(go.Scatter(
                x=carr_monthly.index.to_timestamp(), y=carr_monthly[carrier],
                name=carrier, line=dict(color=COLORS[i%len(COLORS)], width=2)
            ))
        fig2.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=260, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:  # DELAY INTELLIGENCE
        st.markdown("<div class='section-title'>Delay Hotspot Analysis</div>", unsafe_allow_html=True)

        # Define "delayed" as > 7 days (avg + 1sd could be used but 7d is standard)
        delay_threshold = st.slider("Define Delay Threshold (days)", 3, 15, 7)
        df["Delayed"] = df["Delivery_Days"] > delay_threshold

        # Delay by region
        reg_delay = df.groupby("Region").agg(
            Total  = ("Order_ID","count"),
            Delayed= ("Delayed","sum")
        ).reset_index()
        reg_delay["Delay_Rate"] = (reg_delay["Delayed"] / reg_delay["Total"] * 100).round(1)
        reg_delay = reg_delay.sort_values("Delay_Rate", ascending=False)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Delay Rate by Region**")
            fig_r = go.Figure(go.Bar(
                x=reg_delay["Delay_Rate"], y=reg_delay["Region"],
                orientation="h",
                marker_color=[f"rgba(220,38,38,{v/100})" for v in reg_delay["Delay_Rate"]],
                text=[f"{v}%" for v in reg_delay["Delay_Rate"]], textposition="outside",
                textfont=dict(color="#94a3b8")
            ))
            fig_r.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                font_color="#94a3b8", height=320, margin=dict(l=0,r=50,t=0,b=0),
                xaxis=dict(showgrid=False, title="Delay Rate %"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col_r:
            st.markdown("**Delay Rate by Carrier**")
            carr_delay = df.groupby("Courier_Partner").agg(
                Total=("Order_ID","count"), Delayed=("Delayed","sum")
            ).reset_index()
            carr_delay["Delay_Rate"] = (carr_delay["Delayed"] / carr_delay["Total"] * 100).round(1)
            fig_c = go.Figure(go.Bar(
                x=carr_delay["Courier_Partner"], y=carr_delay["Delay_Rate"],
                marker_color=["#dc2626" if v > 30 else "#f59e0b" if v > 15 else "#10b981"
                               for v in carr_delay["Delay_Rate"]],
                text=[f"{v}%" for v in carr_delay["Delay_Rate"]], textposition="outside",
                textfont=dict(color="#94a3b8")
            ))
            fig_c.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                font_color="#94a3b8", height=320, margin=dict(l=0,r=0,t=0,b=40),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45",
                                                        title="Delay Rate %")
            )
            st.plotly_chart(fig_c, use_container_width=True)

        # Carrier-Region heatmap
        st.markdown("<div class='section-title'>Carrier × Region Delay Heatmap</div>",
                    unsafe_allow_html=True)
        pivot = df.groupby(["Courier_Partner","Region"])["Delayed"].mean().unstack(fill_value=0) * 100
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0,"#0f172a"],[0.5,"#7c3aed"],[1,"#dc2626"]],
            text=np.round(pivot.values,1), texttemplate="%{text}%",
            colorbar=dict(tickfont=dict(color="#94a3b8"))
        ))
        fig_h.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=280, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:  # WAREHOUSE FORECAST
        st.markdown("<div class='section-title'>Warehouse Shipment Analysis & Forecast</div>",
                    unsafe_allow_html=True)

        wh_monthly = df.groupby([df["Order_Date"].dt.to_period("M"),"Warehouse"])["Quantity"].sum().unstack(fill_value=0)

        fig_wh = go.Figure()
        for i, wh in enumerate(wh_monthly.columns):
            fig_wh.add_trace(go.Bar(
                x=wh_monthly.index.to_timestamp(), y=wh_monthly[wh],
                name=wh, marker_color=COLORS[i%len(COLORS)]
            ))
        fig_wh.update_layout(
            barmode="stack",
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=300, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_wh, use_container_width=True)

        # Forecast per warehouse
        st.markdown("<div class='section-title'>Warehouse Demand Forecast (to Jun 2026)</div>",
                    unsafe_allow_html=True)

        wh_fore_all = []
        for i, wh in enumerate(wh_monthly.columns):
            series = wh_monthly[wh].rename("value")
            if len(series.dropna()) < 4:
                continue
            fore = forecast_series(series, periods=6)
            future = fore[fore["type"]=="forecast"]
            for _, row in future.iterrows():
                wh_fore_all.append({"Month": row["ds"], "Warehouse": wh, "Forecast": row["y"]})

        wh_fore_df = pd.DataFrame(wh_fore_all)
        if not wh_fore_df.empty:
            fig_wf = go.Figure()
            for i, wh in enumerate(wh_fore_df["Warehouse"].unique()):
                sub = wh_fore_df[wh_fore_df["Warehouse"]==wh]
                fig_wf.add_trace(go.Scatter(
                    x=sub["Month"], y=sub["Forecast"], name=wh,
                    mode="lines+markers",
                    line=dict(color=COLORS[i%len(COLORS)], width=2.5, dash="dot"),
                    marker=dict(size=8)
                ))
            fig_wf.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                font_color="#94a3b8", height=280, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # Warehouse product affinity
        st.markdown("<div class='section-title'>Top Products per Warehouse</div>", unsafe_allow_html=True)
        wh_sel = st.selectbox("Select Warehouse", df["Warehouse"].unique())
        top_prods = (df[df["Warehouse"]==wh_sel]
                     .groupby("Product_Name")["Quantity"].sum()
                     .sort_values(ascending=False).head(10))
        fig_tp = go.Figure(go.Bar(
            x=top_prods.values, y=top_prods.index, orientation="h",
            marker_color="#00e5ff",
            text=top_prods.values, textposition="outside",
            textfont=dict(color="#94a3b8")
        ))
        fig_tp.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=320, margin=dict(l=0,r=60,t=10,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:  # REGION HEATMAP
        st.markdown("<div class='section-title'>Region Performance Overview</div>", unsafe_allow_html=True)

        region_stats = df.groupby("Region").agg(
            Orders        = ("Order_ID","count"),
            Revenue       = ("Revenue_INR","sum"),
            Avg_Del_Days  = ("Delivery_Days","mean"),
            Return_Rate   = ("Return_Flag","mean"),
            Qty           = ("Quantity","sum"),
        ).reset_index().sort_values("Revenue", ascending=False)
        region_stats["Avg_Del_Days"]  = region_stats["Avg_Del_Days"].round(1)
        region_stats["Return_Rate"]   = (region_stats["Return_Rate"]*100).round(1)

        metric_sel = st.selectbox("Metric to display", ["Revenue","Orders","Qty","Avg_Del_Days","Return_Rate"])
        fig_map = go.Figure(go.Bar(
            x=region_stats["Region"], y=region_stats[metric_sel],
            marker=dict(
                color=region_stats[metric_sel],
                colorscale=[[0,"#1e2d45"],[1,"#00e5ff"]]
            ),
            text=region_stats[metric_sel].round(0), textposition="outside",
            textfont=dict(color="#94a3b8")
        ))
        fig_map.update_layout(
            paper_bgcolor="transparent", plot_bgcolor="transparent",
            font_color="#94a3b8", height=320, margin=dict(l=0,r=0,t=10,b=60),
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Best carrier per region
        st.markdown("<div class='section-title'>Best Carrier per Region (lowest avg delay)</div>",
                    unsafe_allow_html=True)
        best_carrier = (df.groupby(["Region","Courier_Partner"])["Delivery_Days"]
                        .mean().reset_index()
                        .sort_values("Delivery_Days")
                        .groupby("Region").first().reset_index())
        best_carrier.columns = ["Region","Best Carrier","Avg Delivery Days"]
        best_carrier["Avg Delivery Days"] = best_carrier["Avg Delivery Days"].round(1)
        st.dataframe(best_carrier, use_container_width=True, hide_index=True)
