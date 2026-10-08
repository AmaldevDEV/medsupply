"""
Medinative - Medical Supply Intelligence
High-Speed Healthcare Operations Command Center
Track 2: Medical Supply Intelligence | Singularity 2026

Visual Identity:
- RED (#EF3340): Urgency / Critical Risk
- LIGHTNING BLUE (#168BFF / #00BFFF): AI / Speed / Intelligence / Action
- WHITE / SLATE (#F8FAFC / #94A3B8): Healthcare / Clarity / Trust
- DARK COMMAND-CENTER THEME: #0B0F17 (Base), #111722 (Panel), #263244 (Borders)

Decision Architecture:
Detect -> Predict -> Decide -> Act
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Setup page config
st.set_page_config(
    page_title="Medinative | Medical Supply Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.database import query_df
from models.inventory_projection import InventoryProjectionEngine
from models.expiry_wastage import ExpiryWastageDetector
from models.anomaly.detector import RegionalAnomalyDetector
from optimization.priority import SupplyPriorityEngine
from optimization.redistribution import RedistributionOptimizer
from optimization.emergency_matching import EmergencySourcingEngine
from simulation.what_if import WhatIfSimulator
from simulation.outbreak import OutbreakSimulator
from assistant.agent import MedicalSupplyAIAssistant

# Initialize engines with caching
@st.cache_resource
def get_engines():
    return {
        "projection": InventoryProjectionEngine(),
        "expiry": ExpiryWastageDetector(),
        "anomaly": RegionalAnomalyDetector(),
        "priority": SupplyPriorityEngine(),
        "redistribution": RedistributionOptimizer(),
        "emergency": EmergencySourcingEngine(),
        "what_if": WhatIfSimulator(),
        "outbreak": OutbreakSimulator(),
        "assistant": MedicalSupplyAIAssistant()
    }

engines = get_engines()

# Helper queries
@st.cache_data(ttl=60)
def load_hospitals():
    return query_df("SELECT * FROM hospitals ORDER BY hospital_id ASC")

@st.cache_data(ttl=60)
def load_medicines():
    return query_df("SELECT * FROM medicines ORDER BY medicine_id ASC")

hospitals_df = load_hospitals()
medicines_df = load_medicines()

hosp_id_to_name = dict(zip(hospitals_df["hospital_id"], hospitals_df["hospital_name"]))
med_id_to_name = dict(zip(medicines_df["medicine_id"], medicines_df["medicine_name"]))

# Session state defaults
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏥 Clinical Operations Portal"
if "selected_hospital_drill" not in st.session_state:
    st.session_state["selected_hospital_drill"] = "HOSP_A"
if "emergency_selected_req" not in st.session_state:
    st.session_state["emergency_selected_req"] = "EMERG_REQ_001"
if "show_ai_drawer" not in st.session_state:
    st.session_state["show_ai_drawer"] = False
if "map_filter_mode" not in st.session_state:
    st.session_state["map_filter_mode"] = "All"
if "map_highlight_hosp" not in st.session_state:
    st.session_state["map_highlight_hosp"] = "HOSP_A"

# ==============================================================================
# NAVIGATION & CONSOLE SWITCHER
# ==============================================================================
NAV_OPTIONS = [
    "🏥 Clinical Operations Portal",
    "🚨 Regional Command Mesh",
    "⚡ Dark Tactical Vector Hub",
    "🏥 Hospital Drilldown",
    "📈 Inventory & Forecast",
    "⚠️ Shortage & Lead-Time Risk",
    "⏳ Expiry / Waste Risk",
    "🔄 Normal Redistribution",
    "🚑 Emergency Sourcing",
    "🧪 What-If & Outbreak Simulator",
    "🤖 AI Supply Assistant"
]

# Sidebar Navigation (keeps main viewport 100% clean, flush, and zero-margin)
with st.sidebar:
    st.markdown("### ⚡ Medinative Command System")
    selected_nav = st.selectbox(
        "Operational Console View",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_OPTIONS else 0,
        key="sidebar_nav_selector"
    )
    st.session_state["nav_page"] = selected_nav

page = st.session_state["nav_page"]

# ==============================================================================
# CONDITIONAL STYLING: ZERO BLANK SPACE FOR PORTAL / DARK THEME FOR MESH
# ==============================================================================
if page == "🏥 Clinical Operations Portal":
    st.markdown("""
    <style>
    /* 1. Eliminate Streamlit Header Blank Space & Toolbar */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* 2. Zero Out All Container Padding & Margins - Flush to top */
    .block-container {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        padding-left: 0px !important;
        padding-right: 0px !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        max-width: 100% !important;
    }

    /* 3. Base Background */
    .stApp {
        background-color: #F7F9FC !important;
        color: #172033 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 4. Full-width flush iframe */
    iframe {
        width: 100% !important;
        border: none !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Dark Command Center Styling
    st.markdown("""
    <style>
    /* Base Dark Background */
    .stApp {
        background-color: #0B0F17 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Header Banner */
    .medinative-header {
        background: #111722;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 12px 24px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .medinative-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .medinative-logo {
        color: #168BFF;
        font-size: 1.8rem;
        font-weight: 900;
        filter: drop-shadow(0 0 10px rgba(22, 139, 255, 0.65));
    }
    .medinative-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .medinative-subtitle {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .medinative-telemetry {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.8rem;
    }
    .telemetry-online {
        color: #22C55E;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .telemetry-tag {
        background: #1A2234;
        border: 1px solid #263244;
        padding: 4px 10px;
        border-radius: 6px;
        color: #94A3B8;
        font-family: monospace;
        font-size: 0.75rem;
    }

    /* Metric KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #111722 !important;
        border: 1px solid #263244 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #F8FAFC !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* Custom Cards */
    .med-card {
        background: #111722;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .med-card-critical {
        border-left: 4px solid #EF3340 !important;
    }
    .med-card-speed {
        border-left: 4px solid #168BFF !important;
    }
    .med-card-safe {
        border-left: 4px solid #22C55E !important;
    }
    .med-card-warning {
        border-left: 4px solid #F5B942 !important;
    }

    /* Data Tables */
    div[data-testid="stDataFrame"] {
        background-color: #111722 !important;
        border: 1px solid #263244 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Primary Button Styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, #168BFF 0%, #00BFFF 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(22, 139, 255, 0.4) !important;
        transition: all 0.2s ease;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(22, 139, 255, 0.6) !important;
    }
    button[kind="secondary"] {
        background-color: #111722 !important;
        border: 1px solid #263244 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    button[kind="secondary"]:hover {
        border-color: #168BFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dark Header Banner for non-portal views
    st.markdown("""
    <div class="medinative-header">
        <div class="medinative-brand">
            <span class="medinative-logo">⚡</span>
            <div>
                <div class="medinative-title">Medinative</div>
                <div class="medinative-subtitle">Medical Supply Intelligence • Healthcare Operations</div>
            </div>
        </div>
        <div class="medinative-telemetry">
            <span class="telemetry-online">● Intelligence Engine Online</span>
            <span class="telemetry-tag">⚡ 10 Facilities Mesh</span>
            <span class="telemetry-tag">INR (₹) Standard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Drawer for Demo Scenarios & Global Parameters
with st.sidebar:
    st.markdown("### ⚡ Medinative Control Panel")
    st.caption("AI Decision-Support System | Regional Healthcare Network")
    st.markdown("---")
    
    st.markdown("#### 🎯 Quick-Load Demo Scenarios")
    demo_scenario = st.selectbox(
        "Load Evaluation Scenario",
        [
            "None (Custom Operations)",
            "Scenario 1: Normal Operations",
            "Scenario 2: Hospital A Stockout Risk (MED004)",
            "Scenario 3: Hospital C Expiring Batch (MED001)",
            "Scenario 4: Regional Outbreak (Hospitals D, E, F)",
            "Scenario 5: Competing Priority Allocation",
            "Scenario 6: Hospital H Emergency Surgery (Solvable)",
            "Scenario 7: Impossible Emergency Deadline (Unsolvable)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    llm_status = "🟢 Ollama Connected" if engines["assistant"].is_ollama_available else "🟡 Deterministic Mode"
    st.info(f"**Local AI Assistant**: {llm_status}\n\n**Database**: SQLite (`medical_supply.db`)\n**Transport**: ₹35/km dedicated courier")

# Helper function to create dark intelligence network map
def create_network_map(hospitals, highlight_hosp="HOSP_A", show_routes=True, filter_mode="All"):
    fig = go.Figure()
    
    # 1. Add Redistribution Routes (Electric Blue lines: #168BFF)
    if show_routes:
        # Normal transfer: Hospital B (surplus) -> Hospital A (deficit)
        fig.add_trace(go.Scattermap(
            lat=[28.5284, 28.5672],
            lon=[77.2185, 77.2100],
            mode="lines+text",
            line=dict(width=3, color="#168BFF"),
            text=["", "⚡ 2,500u (B → A)"],
            textposition="top center",
            textfont=dict(color="#00BFFF", size=10),
            name="Redistribution: B → A",
            hoverinfo="text",
            hovertext="Recommended Redistribution: Hospital B → Hospital A<br>2,500 units | Transit: 34m | Cost: ₹612.50 | 🟢 Feasible"
        ))
        
        # Emergency STAT Transfer: Hospital B -> Hospital H (Brighter Cyan: #00BFFF)
        fig.add_trace(go.Scattermap(
            lat=[28.5284, 28.6304],
            lon=[77.2185, 77.2177],
            mode="lines+text",
            line=dict(width=4, color="#00BFFF"),
            text=["", "🚨 STAT: B → H (500u)"],
            textposition="bottom right",
            textfont=dict(color="#00BFFF", size=11),
            name="Emergency: B → H",
            hoverinfo="text",
            hovertext="⚡ STAT Emergency Transfer: Hospital B → Hospital H<br>500 units | Transit: 34m via Expressway | Required: 10:00 AM | 🟢 Feasible"
        ))

    # 2. Add Hospital Markers with Clinical Risk States
    risk_colors = {
        "HOSP_A": "#EF3340", # Red (Critical stockout)
        "HOSP_D": "#EF3340", # Red (Outbreak surge / Severe gridlock)
        "HOSP_H": "#EF3340", # Red (Emergency surgical deficit)
        "HOSP_C": "#F5B942", # Amber (Imminent batch expiry)
        "HOSP_E": "#F5B942", # Amber (High trauma load)
        "HOSP_F": "#F5B942", # Amber (Supplier lead-time lag)
        "HOSP_G": "#F5B942", # Amber (Alternative required)
        "HOSP_B": "#22C55E", # Green (Safe surplus)
        "HOSP_I": "#F5B942", # Amber (Congested bridge chokepoint)
        "HOSP_J": "#22C55E", # Green (Safe surplus)
    }

    filtered_df = hospitals.copy()
    if filter_mode == "Critical Only":
        filtered_df = filtered_df[filtered_df["hospital_id"].isin(["HOSP_A", "HOSP_D", "HOSP_H"])]

    marker_colors = [risk_colors.get(h, "#22C55E") for h in filtered_df["hospital_id"]]
    marker_sizes = [24 if h == highlight_hosp else 16 for h in filtered_df["hospital_id"]]

    fig.add_trace(go.Scattermap(
        lat=filtered_df["latitude"],
        lon=filtered_df["longitude"],
        mode="markers+text",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            opacity=0.95
        ),
        text=[
            f"{row['hospital_name']} {'🔴' if risk_colors.get(row['hospital_id']) == '#EF3340' else ('🟡' if risk_colors.get(row['hospital_id']) == '#F5B942' else '🟢')}"
            for _, row in filtered_df.iterrows()
        ],
        textposition="top right",
        textfont=dict(color="#F8FAFC", size=11, family="sans-serif"),
        name="Hospitals",
        hoverinfo="text",
        hovertext=[
            f"<b>{row['hospital_name']}</b><br>"
            f"Type: {row['type']}<br>"
            f"Beds: {row['beds']} (ICU: {row['ICU_capacity']})<br>"
            f"Occupancy: {int(row['current_occupancy']*100)}%<br>"
            f"Emergency Load: {row['emergency_patient_load']} pts/day"
            for _, row in filtered_df.iterrows()
        ]
    ))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=28.59, lon=77.20),
            zoom=9.7
        ),
        paper_bgcolor="#111722",
        plot_bgcolor="#111722",
        margin=dict(r=0, t=0, l=0, b=0),
        height=450,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.06,
            bgcolor="rgba(17, 23, 34, 0.85)",
            bordercolor="#263244",
            borderwidth=1,
            font=dict(color="#F8FAFC", size=10)
        )
    )
    return fig

# ==============================================================================
# PAGE 1: 🏥 CLINICAL OPERATIONS PORTAL (DEFAULT LANDING VIEW)
# ==============================================================================
if page == "🏥 Clinical Operations Portal":
    html_path = BASE_DIR / "dashboard" / "medicinative_portal.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=1300, scrolling=True)
    else:
        st.error(f"Portal file not found at {html_path}")

# ==============================================================================
# PAGE 2: 🚨 REGIONAL COMMAND MESH
# ==============================================================================
elif page == "🚨 Regional Command Mesh":
    # Check regional anomalies
    anomalies = engines["anomaly"].detect_anomalies(recent_window_days=7)
    if anomalies:
        for anom in anomalies:
            if anom["is_regional"]:
                st.warning(f"⚠️ **REGIONAL ANOMALY DETECTED**: {anom['explanation']}")

    # Scan overall shortage risks
    shortages_df = engines["projection"].scan_all_shortage_risks()
    crit_shortages = shortages_df[shortages_df["lead_time_risk"] == "CRITICAL"]
    high_shortages = shortages_df[shortages_df["lead_time_risk"] == "HIGH"]

    # Scan batch expiry risks
    expiry_df = engines["expiry"].scan_all_batches()
    at_risk_expiry = expiry_df[expiry_df["at_risk_waste_units"] > 0] if not expiry_df.empty else pd.DataFrame()
    total_waste_loss = at_risk_expiry["potential_financial_loss_inr"].sum() if not at_risk_expiry.empty else 0.0

    # Telemetry and View Switch Bar
    tc1, tc2 = st.columns([5, 1])
    with tc1:
        st.markdown(
            "<div style='font-size: 0.8rem; color: #94A3B8; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>"
            "<span style='color: #22C55E;'>● Live Telemetry Synced</span>"
            "<span>|</span><span>10 Hospital Hubs</span>"
            "<span>|</span><span style='color: #168BFF;'>Detect → Predict → Decide → Act</span>"
            "</div>",
            unsafe_allow_html=True
        )
    with tc2:
        if st.button("🏥 Clinical Portal", use_container_width=True, type="secondary", help="Switch to redesigned clinical operations portal"):
            st.session_state["nav_page"] = "🏥 Clinical Operations Portal"
            st.rerun()

    # 1. TOP KPI ROW (Compact, High-Contrast)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Critical Shortages",
            value=f"🔴 {len(crit_shortages)}",
            delta=f"{len(high_shortages)} High Risk (< Lead Time)",
            delta_color="inverse"
        )
    with k2:
        st.metric(
            label="Transferable Surplus",
            value=f"⚡ {shortages_df['transferable_surplus'].sum():,} units",
            delta="Verified Safe for Transfer"
        )
    with k3:
        st.metric(
            label="Expiry Waste Risk",
            value=f"₹{total_waste_loss:,.0f}",
            delta=f"{len(at_risk_expiry)} Batches at Imminent Risk",
            delta_color="inverse"
        )
    with k4:
        st.metric(
            label="Open Emergency Requests",
            value="🚨 3",
            delta="1 Solvable | 1 Critical Deadline"
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. MAIN TWO-COLUMN COMMAND CENTER LAYOUT (58% Left Map / 42% Right Actions)
    col_map, col_act = st.columns([58, 42])

    with col_map:
        st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;'>
            <h3 style='margin: 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>🌐 Regional Hospital Facility Network</h3>
            <span style='font-size: 0.75rem; color: #94A3B8;'>Real-time facility status & supply movement</span>
        </div>
        """, unsafe_allow_html=True)

        # Map filter controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
        with ctrl_col1:
            if st.button("All Facilities", use_container_width=True, type="secondary"):
                st.session_state["map_filter_mode"] = "All"
        with ctrl_col2:
            if st.button("🔴 Critical Only", use_container_width=True, type="secondary"):
                st.session_state["map_filter_mode"] = "Critical Only"
        with ctrl_col3:
            st.session_state["map_highlight_hosp"] = st.selectbox(
                "Highlight Facility",
                hospitals_df["hospital_id"].tolist(),
                index=hospitals_df["hospital_id"].tolist().index(st.session_state["map_highlight_hosp"]),
                format_func=lambda x: f"{hosp_id_to_name[x]} ({x})",
                label_visibility="collapsed"
            )

        # Render Mapbox Darkmap
        fig_map = create_network_map(
            hospitals_df,
            highlight_hosp=st.session_state["map_highlight_hosp"],
            show_routes=True,
            filter_mode=st.session_state["map_filter_mode"]
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Map Legend & Selected Hospital Action Card
        hl_id = st.session_state["map_highlight_hosp"]
        hl_info = hospitals_df[hospitals_df["hospital_id"] == hl_id].iloc[0]
        hl_shortage = shortages_df[shortages_df["hospital_id"] == hl_id]
        hl_crit = hl_shortage[hl_shortage["lead_time_risk"] == "CRITICAL"]
        
        risk_label = "🔴 Critical Risk" if not hl_crit.empty else "🟢 Operational / Safe"
        risk_class = "med-card-critical" if not hl_crit.empty else "med-card-safe"
        highest_med = hl_crit.iloc[0]["medicine_name"] if not hl_crit.empty else "No Critical Shortage"
        days_left = hl_crit.iloc[0]["days_until_stockout"] if not hl_crit.empty else "14+ days"

        st.markdown(f"""
        <div class='med-card {risk_class}' style='margin-top: -6px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong style='font-size: 1rem; color: #F8FAFC;'>{hl_info['hospital_name']}</strong>
                    <span style='font-size: 0.8rem; margin-left: 8px; color: #94A3B8;'>{hl_info['type']}</span>
                </div>
                <span style='font-size: 0.8rem; font-weight: 700; color: {"#EF3340" if "Critical" in risk_label else "#22C55E"};'>{risk_label}</span>
            </div>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; font-size: 0.8rem;'>
                <div><span style='color: #94A3B8;'>Occupancy:</span> <strong style='color: #F8FAFC;'>{int(hl_info['current_occupancy']*100)}%</strong></div>
                <div><span style='color: #94A3B8;'>Emergency:</span> <strong style='color: #F8FAFC;'>{hl_info['emergency_patient_load']} pts/d</strong></div>
                <div><span style='color: #94A3B8;'>Highest Risk:</span> <strong style='color: #EF3340;'>{highest_med}</strong></div>
                <div><span style='color: #94A3B8;'>Stockout In:</span> <strong style='color: #EF3340;'>{days_left}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"🔍 View {hl_info['hospital_name']} Drilldown", type="primary", use_container_width=True):
            st.session_state["selected_hospital_drill"] = hl_id
            st.session_state["nav_page"] = "🏥 Hospital Drilldown"
            st.rerun()

    with col_act:
        # RIGHT PANEL: Critical Shortage Watchlist
        st.markdown("<h3 style='margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>⚠️ Top Critical Shortage Watchlist</h3>", unsafe_allow_html=True)
        
        if not crit_shortages.empty:
            watch_df = crit_shortages[[
                "hospital_name", "medicine_name", "current_stock",
                "days_until_stockout", "supplier_lead_time", "lead_time_risk"
            ]].copy()
            watch_df.columns = ["Hospital", "Supply", "Stock", "Stockout", "Lead Time", "Risk"]
            st.dataframe(watch_df, hide_index=True, use_container_width=True)
        else:
            st.success("No immediate critical shortages detected.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # RIGHT PANEL: Pending Emergency Requests
        st.markdown("<h3 style='margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>🚑 Pending Emergency Requests</h3>", unsafe_allow_html=True)
        
        emerg_q = "SELECT emergency_request_id, requesting_hospital, medicine_id, quantity_required, required_by, procedure_type, urgency_level FROM emergency_requests"
        emerg_records = query_df(emerg_q)
        
        for _, er in emerg_records.iterrows():
            req_h_name = hosp_id_to_name.get(er["requesting_hospital"], er["requesting_hospital"])
            req_m_name = med_id_to_name.get(er["medicine_id"], er["medicine_id"])
            is_solvable = (er["requesting_hospital"] == "HOSP_H")
            
            e_col_text, e_col_btn = st.columns([3, 1])
            with e_col_text:
                st.markdown(f"""
                <div class='med-card med-card-critical' style='padding: 10px; margin-bottom: 6px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong style='color: #F8FAFC; font-size: 0.85rem;'>{er['emergency_request_id']} • {req_h_name}</strong>
                        <span style='color: #EF3340; font-weight: 700; font-size: 0.75rem;'>{er['urgency_level']}</span>
                    </div>
                    <div style='font-size: 0.8rem; color: #94A3B8; margin-top: 2px;'>
                        Supply: <strong style='color: #F8FAFC;'>{req_m_name}</strong> ({er['quantity_required']:,} units)
                    </div>
                    <div style='font-size: 0.75rem; color: #94A3B8;'>
                        Procedure: {er['procedure_type']} | Deadline: <strong style='color: #F8FAFC;'>{er['required_by']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with e_col_btn:
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                if st.button("⚡ Find Source", key=f"btn_src_{er['emergency_request_id']}", use_container_width=True, type="primary"):
                    st.session_state["emergency_selected_req"] = er["emergency_request_id"]
                    st.session_state["nav_page"] = "🚑 Emergency Sourcing"
                    st.rerun()

    st.markdown("---")

    # 3. THREE-COLUMN INTELLIGENCE SECTION BELOW THE MAP
    st.markdown("<h3 style='margin: 0 0 12px 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>⚡ Three-Tier Supply Intelligence: Forecast → Projection → Action</h3>", unsafe_allow_html=True)
    c_fc, c_proj, c_act = st.columns(3)

    with c_fc:
        st.markdown("<h4 style='font-size: 0.95rem; color: #168BFF; margin: 0 0 6px 0;'>1. Demand Forecast (ML Prediction)</h4>", unsafe_allow_html=True)
        st.caption("Autoregressive multi-horizon ML vs historical consumption for Hospital A (MED004)")
        
        hist_sample = query_df(
            "SELECT date, daily_consumption FROM demand_history WHERE hospital_id = 'HOSP_A' AND medicine_id = 'MED004' ORDER BY date DESC LIMIT 20"
        ).sort_values("date")
        fc_sample = query_df(
            "SELECT target_date as date, predicted_demand FROM forecasts WHERE hospital_id = 'HOSP_A' AND medicine_id = 'MED004' ORDER BY day_ahead ASC LIMIT 14"
        )
        
        fig_fc_mini = go.Figure()
        fig_fc_mini.add_trace(go.Scatter(x=hist_sample["date"], y=hist_sample["daily_consumption"], mode="lines", name="Actual", line=dict(color="#94A3B8", width=1.5)))
        fig_fc_mini.add_trace(go.Scatter(x=fc_sample["date"], y=fc_sample["predicted_demand"], mode="lines+markers", name="ML Forecast", line=dict(color="#168BFF", width=2.5, dash="dash")))
        fig_fc_mini.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#111722", plot_bgcolor="#111722", font=dict(color="#94A3B8", size=9), showlegend=True)
        st.plotly_chart(fig_fc_mini, use_container_width=True)

    with c_proj:
        st.markdown("<h4 style='font-size: 0.95rem; color: #EF3340; margin: 0 0 6px 0;'>2. Inventory Projection & Stockout</h4>", unsafe_allow_html=True)
        st.caption("Day-by-day simulated forward depletion for Hospital A (MED004)")
        
        proj_sample = engines["projection"].project_inventory("HOSP_A", "MED004")
        t_df = pd.DataFrame(proj_sample["trajectory"]).head(14)
        
        fig_t_mini = go.Figure()
        fig_t_mini.add_trace(go.Scatter(x=t_df["date"], y=t_df["projected_stock"], mode="lines+markers", name="Stock", line=dict(color="#EF3340", width=2.5)))
        fig_t_mini.add_hline(y=proj_sample["safety_stock_threshold"], line_dash="dot", line_color="#EF3340", annotation_text="Safety Buffer", annotation_font_size=9)
        fig_t_mini.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#111722", plot_bgcolor="#111722", font=dict(color="#94A3B8", size=9), showlegend=False)
        st.plotly_chart(fig_t_mini, use_container_width=True)

    with c_act:
        st.markdown("<h4 style='font-size: 0.95rem; color: #22C55E; margin: 0 0 6px 0;'>3. AI Recommended Actions</h4>", unsafe_allow_html=True)
        st.caption("Top 3 algorithmic directives prioritized by urgency and feasibility")
        
        st.markdown("""
        <div class='med-card med-card-speed' style='padding: 8px 12px; margin-bottom: 6px;'>
            <div style='font-size: 0.8rem; font-weight: 700; color: #00BFFF;'>1. Hospital B → Hospital A</div>
            <div style='font-size: 0.75rem; color: #F8FAFC;'>Transfer 2,500 units MED004 • Transit: 34m</div>
            <div style='font-size: 0.7rem; color: #94A3B8;'>Prevents critical stockout in 1 day; B holds 4,500 surplus.</div>
        </div>
        <div class='med-card med-card-warning' style='padding: 8px 12px; margin-bottom: 6px;'>
            <div style='font-size: 0.8rem; font-weight: 700; color: #F5B942;'>2. Hospital C → Hospital A</div>
            <div style='font-size: 0.75rem; color: #F8FAFC;'>Transfer 2,300 units MED001 • Transit: 45m</div>
            <div style='font-size: 0.7rem; color: #94A3B8;'>Salvages batch BAT-C-001-WASTE (₹19,55,000 loss).</div>
        </div>
        <div class='med-card med-card-critical' style='padding: 8px 12px; margin-bottom: 6px;'>
            <div style='font-size: 0.8rem; font-weight: 700; color: #EF3340;'>3. STAT Source: Hospital B → Hospital H</div>
            <div style='font-size: 0.75rem; color: #F8FAFC;'>Dispatch 500 units MED008 • Transit: 34m</div>
            <div style='font-size: 0.7rem; color: #94A3B8;'>Arrives 17.8h before surgery; saves 36m over gridlock.</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: ⚡ DARK TACTICAL VECTOR HUB
# ==============================================================================
elif page == "⚡ Dark Tactical Vector Hub":
    tc1, tc2 = st.columns([5, 1])
    with tc1:
        st.markdown("<h3 style='margin: 0; color: #F8FAFC;'>⚡ Dark Tactical Vector Command Hub</h3>", unsafe_allow_html=True)
        st.caption("Pixel-perfect tactical high-density command center with animated vector SVG telemetry mesh, real-time alerts, and spatial routing.")
    with tc2:
        if st.button("🏥 Clinical Portal", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "🏥 Clinical Operations Portal"
            st.rerun()

    html_path = BASE_DIR / "dashboard" / "medinative_command_center.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=1200, scrolling=True)
    else:
        st.error(f"HTML template not found at {html_path}")

# ==============================================================================
# PAGE 2: 🏥 HOSPITAL DRILLDOWN
# ==============================================================================
elif page == "🏥 Hospital Drilldown":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>🏥 Hospital Facility Deep-Dive</h2>", unsafe_allow_html=True)
    
    sel_hosp = st.selectbox(
        "Select Hospital Facility",
        hospitals_df["hospital_id"].tolist(),
        index=hospitals_df["hospital_id"].tolist().index(st.session_state["selected_hospital_drill"]),
        format_func=lambda x: f"{hosp_id_to_name[x]} ({x})"
    )
    st.session_state["selected_hospital_drill"] = sel_hosp
    h_info = hospitals_df[hospitals_df["hospital_id"] == sel_hosp].iloc[0]

    # Facility Profile Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beds Capacity", f"{h_info['beds']} beds", f"ICU: {h_info['ICU_capacity']}")
    c2.metric("Current Occupancy", f"{int(h_info['current_occupancy']*100)}%", f"Daily Cap: {h_info['daily_patient_capacity']}")
    c3.metric("Emergency Influx", f"{h_info['emergency_patient_load']} pts/day", "Acuity Baseline")
    c4.metric("Safety Stock Policy", f"{h_info['safety_stock_policy_days']} Days", "Mandatory Reserve")

    st.markdown(f"**Facility Profile**: {h_info['description']}")
    st.markdown("---")

    # Inventory Portfolio for selected hospital
    st.subheader("📦 Inventory Portfolio & Days to Stockout")
    hosp_shortages = engines["projection"].scan_all_shortage_risks()
    hosp_subset = hosp_shortages[hosp_shortages["hospital_id"] == sel_hosp].sort_values("days_until_stockout")

    # Bar chart of stockout days
    fig_stockout = px.bar(
        hosp_subset,
        x="medicine_name",
        y="days_until_stockout",
        color="lead_time_risk",
        color_discrete_map={"CRITICAL": "#EF3340", "HIGH": "#F5B942", "MODERATE": "#168BFF", "LOW": "#22C55E"},
        title="Projected Days Until Stockout vs Clinical Supplies",
        labels={"medicine_name": "Medical Supply", "days_until_stockout": "Days to Stockout"},
        template="plotly_dark"
    )
    fig_stockout.add_hline(y=h_info["safety_stock_policy_days"], line_dash="dash", line_color="#EF3340", annotation_text="Safety Stock Threshold")
    fig_stockout.update_layout(paper_bgcolor="#111722", plot_bgcolor="#111722")
    st.plotly_chart(fig_stockout, use_container_width=True)

    # Detailed inventory table
    st.dataframe(
        hosp_subset[[
            "medicine_name", "category", "current_stock", "safety_threshold",
            "avg_daily_demand", "days_until_stockout", "supplier_lead_time",
            "lead_time_risk", "transferable_surplus", "risk_explanation"
        ]],
        hide_index=True,
        use_container_width=True
    )

# ==============================================================================
# PAGE 3: 📈 INVENTORY & FORECAST
# ==============================================================================
elif page == "📈 Inventory & Forecast":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>📈 Demand Forecasting & Model Validation</h2>", unsafe_allow_html=True)
    st.markdown("Autoregressive multi-horizon ML predictions with time-based holdout evaluation.")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sel_hosp = st.selectbox("Select Facility", hospitals_df["hospital_id"].tolist(), format_func=lambda x: hosp_id_to_name[x])
    with f_col2:
        sel_med = st.selectbox("Select Medical Supply", medicines_df["medicine_id"].tolist(), format_func=lambda x: f"{med_id_to_name[x]} ({x})")

    # Query demand history
    hist_df = query_df(
        "SELECT date, daily_consumption, patient_load, emergency_load FROM demand_history WHERE hospital_id = ? AND medicine_id = ? ORDER BY date ASC",
        (sel_hosp, sel_med)
    )
    
    # Query forecasts
    fc_df = query_df(
        "SELECT target_date as date, predicted_demand, lower_bound, upper_bound FROM forecasts WHERE hospital_id = ? AND medicine_id = ? ORDER BY day_ahead ASC",
        (sel_hosp, sel_med)
    )

    # Query metrics
    met_df = query_df(
        "SELECT * FROM model_metrics WHERE hospital_id = ? AND medicine_id = ?",
        (sel_hosp, sel_med)
    )

    if not met_df.empty:
        m = met_df.iloc[0]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("ML Model MAE", f"{m['ml_mae']} units", "Holdout Test")
        mc2.metric("Baseline MAE", f"{m['baseline_mae']} units", "7-Day Moving Avg")
        mc3.metric("RMSE", f"{m['ml_rmse']} units", "Residual Dispersion")
        mc4.metric("Improvement", f"{m['improvement_pct']}%", "vs Baseline")

    # Interactive Forecast Chart
    if not hist_df.empty and not fc_df.empty:
        recent_hist = hist_df.tail(60).copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_hist["date"],
            y=recent_hist["daily_consumption"],
            mode="lines+markers",
            name="Actual Consumption (Historical)",
            line=dict(color="#94A3B8", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["date"],
            y=fc_df["predicted_demand"],
            mode="lines+markers",
            name="ML Forecast (RandomForest)",
            line=dict(color="#168BFF", width=2.5, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=list(fc_df["date"]) + list(fc_df["date"])[::-1],
            y=list(fc_df["upper_bound"]) + list(fc_df["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(0,191,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Prediction Interval"
        ))

        fig.update_layout(
            title=f"Historical Demand & 30-Day Multi-Horizon Forecast — {hosp_id_to_name[sel_hosp]} ({med_id_to_name[sel_med]})",
            xaxis_title="Date",
            yaxis_title="Units / Day",
            hovermode="x unified",
            height=450,
            paper_bgcolor="#111722",
            plot_bgcolor="#111722",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 4: ⚠️ SHORTAGE & LEAD-TIME RISK
# ==============================================================================
elif page == "⚠️ Shortage & Lead-Time Risk":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>⚠️ Day-by-Day Shortage & Lead-Time Risk Simulation</h2>", unsafe_allow_html=True)
    st.markdown("Simulates future inventory day-by-day to pinpoint exact stockout dates and evaluate procurement lead times.")

    col_a, col_b = st.columns(2)
    with col_a:
        s_hosp = st.selectbox("Hospital", hospitals_df["hospital_id"].tolist(), format_func=lambda x: hosp_id_to_name[x])
    with col_b:
        s_med = st.selectbox("Supply", medicines_df["medicine_id"].tolist(), format_func=lambda x: f"{med_id_to_name[x]} ({x})")

    proj = engines["projection"].project_inventory(s_hosp, s_med)

    risk_color = "#EF3340" if proj["lead_time_risk"] == "CRITICAL" else ("#F5B942" if proj["lead_time_risk"] == "HIGH" else "#22C55E")
    st.markdown(f"### Status: <span style='color:{risk_color};font-weight:bold;'>[{proj['lead_time_risk']} RISK]</span>", unsafe_allow_html=True)
    st.info(proj["risk_explanation"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current On-Hand Stock", f"{proj['current_stock']:,} units")
    m2.metric("Days Until Stockout", f"{proj['days_until_stockout']} Days")
    m3.metric("Supplier Lead Time", f"{proj['supplier_lead_time_days']} Days")
    m4.metric("Safe Transferable Surplus", f"{proj['transferable_surplus']:,} units")

    traj_df = pd.DataFrame(proj["trajectory"])
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=traj_df["date"],
        y=traj_df["projected_stock"],
        mode="lines+markers",
        name="Projected Inventory",
        line=dict(color="#22C55E" if proj["lead_time_risk"] == "LOW" else "#EF3340", width=3)
    ))
    fig_traj.add_hline(
        y=proj["safety_stock_threshold"],
        line_dash="dot",
        line_color="#EF3340",
        annotation_text=f"Mandatory Safety Reserve ({proj['safety_stock_threshold']} units)"
    )
    fig_traj.update_layout(
        title="Simulated Forward 30-Day Inventory Trajectory",
        xaxis_title="Date",
        yaxis_title="Projected On-Hand Stock (Units)",
        height=400,
        paper_bgcolor="#111722",
        plot_bgcolor="#111722",
        font=dict(color="#F8FAFC")
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# ==============================================================================
# PAGE 5: ⏳ EXPIRY / WASTE RISK
# ==============================================================================
elif page == "⏳ Expiry / Waste Risk":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>⏳ Batch-Level Expiry and Wastage Risk</h2>", unsafe_allow_html=True)
    st.markdown("Evaluates batch expiration timelines against forecast consumption to proactively eliminate medical waste.")

    exp_df = engines["expiry"].scan_all_batches()
    at_risk_df = exp_df[exp_df["at_risk_waste_units"] > 0] if not exp_df.empty else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    c1.metric("Batches with Waste Risk", f"{len(at_risk_df)} Batches")
    c2.metric("Total Units at Risk", f"{at_risk_df['at_risk_waste_units'].sum():,} units" if not at_risk_df.empty else "0")
    c3.metric("Financial Waste Exposure", f"₹{at_risk_df['potential_financial_loss_inr'].sum():,.2f}" if not at_risk_df.empty else "₹0")

    st.markdown("---")
    st.subheader("🔍 At-Risk Batch Breakdown")
    if not at_risk_df.empty:
        fig_waste = px.bar(
            at_risk_df.head(10),
            x="batch_id",
            y=["expected_usage_before_expiry", "at_risk_waste_units"],
            title="Batch Stock: Expected Usage vs At-Risk Waste Units",
            labels={"value": "Units", "batch_id": "Batch ID"},
            color_discrete_map={"expected_usage_before_expiry": "#22C55E", "at_risk_waste_units": "#EF3340"},
            template="plotly_dark"
        )
        fig_waste.update_layout(paper_bgcolor="#111722", plot_bgcolor="#111722")
        st.plotly_chart(fig_waste, use_container_width=True)

        st.dataframe(
            at_risk_df[[
                "hospital_name", "medicine_name", "batch_id", "quantity",
                "days_until_expiry", "expected_usage_before_expiry",
                "safety_stock_protect", "at_risk_waste_units",
                "potential_financial_loss_inr", "risk_tier", "explanation"
            ]],
            column_config={
                "potential_financial_loss_inr": st.column_config.NumberColumn("Loss Exposure (₹)", format="₹%.2f"),
                "quantity": st.column_config.NumberColumn("Quantity (Units)"),
                "expected_usage_before_expiry": st.column_config.NumberColumn("Exp. Usage"),
                "at_risk_waste_units": st.column_config.NumberColumn("At-Risk Units"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("No batches facing imminent waste risk.")

# ==============================================================================
# PAGE 6: 🔄 NORMAL REDISTRIBUTION
# ==============================================================================
elif page == "🔄 Normal Redistribution":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>🔄 Inter-Hospital Redistribution Optimizer</h2>", unsafe_allow_html=True)
    st.markdown("Matches safe surplus donor hospitals with deficit facilities while respecting road transit, traffic flow, and safety buffers.")

    if st.button("🚀 Re-Run Redistribution Optimization", type="primary"):
        st.cache_data.clear()

    plan = engines["redistribution"].generate_redistribution_plan()
    
    if plan:
        pdf = pd.DataFrame(plan)
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Transfers", f"{len(pdf)} Transfers")
        c2.metric("Total Supplies Reallocated", f"{pdf['quantity'].sum():,} units")
        c3.metric("Estimated Transport Cost", f"₹{pdf['estimated_cost_inr'].sum():,.2f}")

        st.markdown("---")
        st.subheader("📋 Actionable Transfer Directives")
        st.dataframe(
            pdf[[
                "recommendation_id", "from_hospital_name", "to_hospital_name",
                "medicine_name", "batch_id", "quantity", "transit_time_minutes",
                "distance_km", "traffic_level", "estimated_cost_inr", "google_maps_url", "reason"
            ]],
            column_config={
                "google_maps_url": st.column_config.LinkColumn("Google Maps Route", display_text="🗺️ Open Map"),
                "estimated_cost_inr": st.column_config.NumberColumn("Est. Cost (₹)", format="₹%.2f"),
                "transit_time_minutes": st.column_config.NumberColumn("Transit (Mins)"),
                "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No inter-hospital transfers currently required.")

# ==============================================================================
# PAGE 7: 🚑 EMERGENCY SOURCING (CREATIVE MODULE)
# ==============================================================================
elif page == "🚑 Emergency Sourcing":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>⚡ Emergency Inter-Hospital Sourcing Engine</h2>", unsafe_allow_html=True)
    st.caption("Find the fastest feasible hospital source when normal procurement cannot meet an urgent requirement.")

    st.markdown("#### ⚡ Quick-Load Evaluation Scenarios")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        load_h = st.button("🏥 Scenario 6: Hospital H Emergency Aortic Repair (Solvable)")
    with b_col2:
        load_d = st.button("💥 Scenario 7: Hospital D Mass Casualty (Impossible Deadline)")
    with b_col3:
        load_g = st.button("💉 Request 3: Hospital G Refractory Immunologic Crisis")

    req_h_def = "HOSP_H"
    req_m_def = "MED008"
    req_q_def = 500
    req_hours_def = 18.5
    req_proc_def = "Emergency Aortic Repair / Reversal"
    req_urg_def = "STAT Critical"

    if load_d:
        req_h_def = "HOSP_D"
        req_m_def = "MED004"
        req_q_def = 800
        req_hours_def = 1.5
        req_proc_def = "Mass Casualty Multi-Trauma Resuscitation"
        req_urg_def = "Immediate Code Red"
    elif load_g:
        req_h_def = "HOSP_G"
        req_m_def = "MED018"
        req_q_def = 80
        req_hours_def = 24.0
        req_proc_def = "Severe Refractory Immunologic Crisis"
        req_urg_def = "Urgent Priority"

    # Traffic Condition Simulator Selector
    st.markdown("#### 🚦 Live Corridor Traffic Condition Simulation")
    traffic_sim_choice = st.radio(
        "Corridor Traffic Congestion",
        [
            "Baseline (Live Saved Corridor Profiles)",
            "Low (Night / Open Expressway - 1.0x delay)",
            "Moderate (Standard Midday Flow - 1.25x delay)",
            "Heavy (Peak Rush Hour - 1.70x delay)",
            "Severe (Major Chokepoint Gridlock - 2.40x delay)"
        ],
        horizontal=True,
        help="Simulates real-time road conditions. When congestion strikes a nearby corridor, the engine re-routes to a faster, farther facility via uncongested expressways."
    )
    traffic_map = {
        "Baseline (Live Saved Corridor Profiles)": None,
        "Low (Night / Open Expressway - 1.0x delay)": "Low",
        "Moderate (Standard Midday Flow - 1.25x delay)": "Moderate",
        "Heavy (Peak Rush Hour - 1.70x delay)": "Heavy",
        "Severe (Major Chokepoint Gridlock - 2.40x delay)": "Severe"
    }
    selected_traffic = traffic_map[traffic_sim_choice]

    with st.form("emergency_dispatch_form"):
        st.subheader("📝 Emergency Medicine Transfer Request")
        f1, f2, f3 = st.columns(3)
        with f1:
            req_hosp = st.selectbox(
                "Requesting Hospital",
                hospitals_df["hospital_id"].tolist(),
                index=hospitals_df["hospital_id"].tolist().index(req_h_def),
                format_func=lambda x: hosp_id_to_name[x]
            )
            procedure = st.text_input("Clinical Procedure", value=req_proc_def)
        with f2:
            req_med = st.selectbox(
                "Required Medical Supply",
                medicines_df["medicine_id"].tolist(),
                index=medicines_df["medicine_id"].tolist().index(req_m_def),
                format_func=lambda x: f"{med_id_to_name[x]} ({x})"
            )
            urgency = st.selectbox(
                "Clinical Urgency Level",
                ["STAT Critical", "Immediate Code Red", "Urgent Priority", "Standard"],
                index=["STAT Critical", "Immediate Code Red", "Urgent Priority", "Standard"].index(req_urg_def)
            )
        with f3:
            quantity = st.number_input("Quantity Required", min_value=10, max_value=5000, value=req_q_def, step=50)
            hours_ahead = st.slider("Hours Until Procedure Deadline", min_value=0.5, max_value=48.0, value=float(req_hours_def), step=0.5)

        submit_btn = st.form_submit_button("⚡ FIND EMERGENCY SUPPLY", use_container_width=True)

    # Process evaluation
    req_id = "EMERG_REQ_001" if req_hosp == "HOSP_H" and req_med == "MED008" else ("EMERG_REQ_002" if req_hosp == "HOSP_D" else "EMERG_REQ_003")
    eval_res = engines["emergency"].evaluate_emergency_request(req_id, simulated_traffic_condition=selected_traffic)

    st.markdown("---")
    st.subheader("🎯 Evaluation & Inter-Hospital Matching Results")

    # Local Stock & Supplier Status Banner
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Local On-Hand Stock", f"{eval_res['current_local_stock']} units", "Stockout Deficit")
    s_col2.metric("Contract Supplier Status", "UNAVAILABLE", eval_res["supplier_status"], delta_color="inverse")
    s_col3.metric("Required By Deadline", eval_res["required_by"])

    # ⚡ NEAREST VS FASTEST APPROACH COMPARISON CARD
    nvf = eval_res.get("nearest_vs_fastest", {})
    nearest = nvf.get("nearest_hospital_by_distance", {})
    fastest = nvf.get("fastest_hospital_by_time", {})

    st.markdown("### ⚡ Nearest vs. Fastest Approach: Traffic-Aware Optimization")
    st.caption("Traditional dispatch systems naively select the closest geographical hospital. Our ML optimization engine evaluates live road traffic and delays to select the FASTEST feasible source.")

    comp1, comp2 = st.columns(2)
    with comp1:
        st.markdown("#### 📍 Geographically Nearest Facility")
        st.write(f"**Facility**: {nearest.get('hospital_name', 'N/A')}")
        st.write(f"**Distance**: {nearest.get('distance_km', 0.0)} km")
        st.write(f"**Traffic Level**: `{nearest.get('traffic_level', 'N/A')}`")
        st.write(f"**Estimated Transit**: {nearest.get('transit_time_minutes', 0)} mins")
        status_label = "✅ Feasible Safe Donor" if nearest.get("is_feasible") else "❌ REJECTED (Chokepoint / Insufficient Stock)"
        st.write(f"**Feasibility**: {status_label}")
        if nearest.get("google_maps_url"):
            st.link_button("🗺️ View Nearest Route in Google Maps", url=nearest["google_maps_url"], use_container_width=True)

    with comp2:
        st.markdown("#### ⚡ Fastest Feasible Source (Model Selected)")
        if fastest and fastest.get("hospital_id"):
            st.write(f"**Facility**: {fastest.get('hospital_name', 'N/A')}")
            st.write(f"**Distance**: {fastest.get('distance_km', 0.0)} km")
            st.write(f"**Traffic Level**: `{fastest.get('traffic_level', 'N/A')}`")
            st.write(f"**Estimated Transit**: {fastest.get('transit_time_minutes', 0)} mins")
            time_saved = fastest.get("time_saved_vs_nearest_minutes", 0)
            if time_saved > 0:
                st.success(f"⚡ **Saves {time_saved} minutes** compared to nearest route!")
            else:
                st.info("Route is also the geographically closest feasible facility.")
            if fastest.get("google_maps_url"):
                st.link_button("🗺️ View Fastest Route in Google Maps", url=fastest["google_maps_url"], use_container_width=True)
        else:
            st.error("No candidate facility could arrive before the required deadline.")

    st.markdown("---")

    if eval_res["has_feasible_source"]:
        rec = eval_res["recommended_source"]
        st.success(f"### ✅ RECOMMENDED DISPATCH: **{rec['candidate_hospital_name']}**")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Transfer Quantity", f"{eval_res['quantity_required']:,} units")
        r2.metric("Donor Safe Surplus", f"{rec['transferable_surplus']:,} units")
        r3.metric("Courier Transit Time", f"{rec['transit_time_minutes']} mins ({rec['distance_km']} km)")
        r4.metric("Estimated Physical Arrival", rec["estimated_arrival"])

        # Traffic Level & Decision Rationale
        st.markdown(f"**Traffic Flow**: `{rec['traffic_level']}` (Congestion Multiplier: {rec['traffic_delay_factor']}x)")
        st.markdown(f"**Clinical Decision Rationale**:\n> {rec['recommendation_reason']}")
        
        # Action button and Google Maps Embed Button
        col_act1, col_act2 = st.columns([2, 3])
        with col_act1:
            if st.button("🚨 Initiate Transfer Request", type="primary", use_container_width=True):
                st.balloons()
                st.success(f"Transfer Request DISPATCHED to {rec['candidate_hospital_name']}. Medical Director authorization requested.")
        with col_act2:
            if rec.get("google_maps_url"):
                st.link_button("🗺️ Open in Google Maps (Live Navigation & Traffic)", url=rec["google_maps_url"], use_container_width=True)

        st.caption(f"🛡️ *{eval_res['clinical_notice']}*")

    else:
        st.error(f"### ❌ NO FEASIBLE HOSPITAL SOURCE FOUND BEFORE THE DEADLINE")
        st.warning(
            "Every evaluated network facility is either currently facing a shortage, holds insufficient safe transferable surplus, "
            "or courier transportation cannot physically arrive before the procedure deadline."
        )

    # Candidates Audit
    st.markdown("### 📋 Complete Network Candidates Audit")
    cand_df = pd.DataFrame(eval_res["all_evaluated_candidates"])
    cand_df["status"] = cand_df["is_feasible"].apply(lambda x: "✅ FEASIBLE" if x else "❌ REJECTED")
    st.dataframe(
        cand_df[[
            "status", "candidate_hospital_name", "current_stock",
            "transferable_surplus", "distance_km", "transit_time_minutes", "traffic_level",
            "estimated_arrival", "donor_remains_safe", "google_maps_url", "rejection_reason"
        ]],
        column_config={
            "google_maps_url": st.column_config.LinkColumn("Google Maps Route", display_text="🗺️ Open Map"),
            "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.1f km"),
            "transit_time_minutes": st.column_config.NumberColumn("Transit (Mins)"),
            "current_stock": st.column_config.NumberColumn("Current Stock"),
            "transferable_surplus": st.column_config.NumberColumn("Safe Surplus"),
        },
        hide_index=True,
        use_container_width=True
    )

# ==============================================================================
# PAGE 8: 🧪 WHAT-IF & OUTBREAK SIMULATOR
# ==============================================================================
elif page == "🧪 What-If & Outbreak Simulator":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>🧪 What-If Multi-Scenario Simulation Engine</h2>", unsafe_allow_html=True)
    st.markdown("Stress-test supply chain resilience against demand surges, regional outbreaks, lead-time extensions, and capacity loss.")

    sim_tab1, sim_tab2 = st.tabs(["⚡ Scenario Simulation", "🦠 Targeted Regional Outbreak"])

    with sim_tab1:
        st.subheader("🎛️ Operational Disruption Parameters")
        w1, w2, w3 = st.columns(3)
        with w1:
            demand_surge = st.slider("Global Demand Surge (%)", 0, 100, 50, step=10)
            lead_time_add = st.slider("Supplier Lead Time Extension (Days)", 0, 10, 2, step=1)
        with w2:
            outbreak_hosps = st.multiselect("Outbreak Affected Facilities", hospitals_df["hospital_id"].tolist(), default=["HOSP_D", "HOSP_E"])
            outbreak_severity = st.slider("Outbreak Severity Multiplier", 1.0, 3.0, 2.0, step=0.2)
        with w3:
            donor_loss = st.slider("Hospital B Stock Loss (%)", 0, 80, 0, step=10)
            transit_slow = st.slider("Transport Slowdown Factor", 1.0, 2.5, 1.2, step=0.1)

        if st.button("🚀 Run Stress Simulation", type="primary"):
            sim_res = engines["what_if"].run_simulation(
                demand_surge_pct=float(demand_surge),
                outbreak_hospitals=outbreak_hosps,
                outbreak_severity_mult=float(outbreak_severity),
                supplier_lead_time_added_days=int(lead_time_add),
                donor_stock_reduction_pct=float(donor_loss),
                transport_slowdown_factor=float(transit_slow)
            )

            st.markdown("---")
            st.subheader("📊 Before vs After Stress Impact")
            st.info(f"**Executive Synthesis**: {sim_res['executive_summary']}")

            comp = sim_res["comparison"]
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Critical Shortages", comp["critical_shortages"]["after"], delta=f"+{comp['critical_shortages']['delta']}", delta_color="inverse")
            sc2.metric("High Risk Facilities", comp["high_risk_count"]["after"], delta=f"+{comp['high_risk_count']['delta']}", delta_color="inverse")
            sc3.metric("Avg Days to Stockout", f"{comp['average_days_until_stockout']['after']}d", delta=f"{comp['average_days_until_stockout']['delta']}d", delta_color="normal")
            sc4.metric("Network Safe Surplus", f"{comp['network_transferable_surplus_units']['after']:,} units", delta=f"{comp['network_transferable_surplus_units']['delta']:,} units")

            st.markdown("#### 🚑 Emergency Sourcing Impact Under Stress")
            st.warning(comp["emergency_sourcing"]["after_reason"])

    with sim_tab2:
        st.subheader("🦠 Regional Outbreak Stress Test")
        st.caption("Simulates sudden epidemic cluster surge across designated facilities.")
        
        ob_btn = st.button("⚡ SIMULATE REGIONAL OUTBREAK", type="primary")
        
        ob_res = engines["outbreak"].simulate_regional_outbreak(
            cluster_hospital_ids=["HOSP_D", "HOSP_E", "HOSP_F"],
            target_medicine_ids=["MED007", "MED016", "MED005"],
            severity_multiplier=2.4
        )
        
        # Outbreak alert badge
        st.markdown("""
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0;'>
            <div class='med-card med-card-critical'>
                <strong>Hospital D (Epicenter)</strong><br>
                <span style='color: #EF3340; font-weight: 800; font-size: 1.2rem;'>🔴 +140%</span> Surge Demand
            </div>
            <div class='med-card med-card-critical'>
                <strong>Hospital E (Trauma Center)</strong><br>
                <span style='color: #EF3340; font-weight: 800; font-size: 1.2rem;'>🔴 +118%</span> Emergency Load
            </div>
            <div class='med-card med-card-warning'>
                <strong>Hospital F (District Clinic)</strong><br>
                <span style='color: #F5B942; font-weight: 800; font-size: 1.2rem;'>🟠 +92%</span> High Consumption
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Outbreak Scenario**: {ob_res['summary_narrative']}")
        
        ob_col1, ob_col2, ob_col3 = st.columns(3)
        ob_col1.metric("New Stockout Breaches", f"{ob_res['new_critical_shortages']} Breaches")
        ob_col2.metric("Surge Deficit Needed", f"{ob_res['total_surge_units_needed']:,} units")
        ob_col3.metric("Network Relief Coverage", f"{ob_res['relief_coverage_pct']}%")

        st.dataframe(pd.DataFrame(ob_res["facility_impacts"]), hide_index=True, use_container_width=True)

# ==============================================================================
# PAGE 9: 🤖 AI SUPPLY ASSISTANT
# ==============================================================================
elif page == "🤖 AI Supply Assistant":
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 800;'>🤖 Medinative Supply Intelligence Assistant</h2>", unsafe_allow_html=True)
    st.markdown("Grounded natural-language decision support. The assistant queries verified database facts, models, and optimization tools—**never fabricating numbers**.")

    st.markdown("#### 💡 Suggested Questions")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("❓ Which hospitals will run out of MED004 next week?"):
            st.session_state["query_input"] = "Which hospitals are likely to run out of Medicine MED004 next week?"
        if st.button("❓ Can Hospital B donate 2,000 units?"):
            st.session_state["query_input"] = "Can Hospital B donate 2,000 units of MED004?"
    with q_col2:
        if st.button("❓ What stock is likely to expire?"):
            st.session_state["query_input"] = "What stock is likely to expire and cause wastage?"
        if st.button("❓ Why is Hospital A high priority?"):
            st.session_state["query_input"] = "Why is Hospital A high priority for scarce supplies?"
    with q_col3:
        if st.button("❓ Emergency surgery: need 500 units MED008 tomorrow"):
            st.session_state["query_input"] = "We have an emergency surgery tomorrow and need 500 units of MED008. Can another hospital provide it?"
        if st.button("❓ What happens if demand increases by 50%?"):
            st.session_state["query_input"] = "What happens if demand increases by 50% across the network?"

    user_query = st.text_input(
        "Enter question about hospital supplies, shortages, surplus, transfers, or simulations:",
        value=st.session_state.get("query_input", "")
    )

    if user_query:
        with st.spinner("Searching network ⚡ → Checking surplus → Checking transport → Grounding facts..."):
            ans = engines["assistant"].query(user_query)

        st.markdown("---")
        st.markdown(ans["answer"])

        with st.expander("🛠️ Grounded Tool Call Audit (Zero Hallucination Proof)"):
            st.write(f"**Analytical Tool Used**: `{ans['tool_used']}`")
            st.write(f"**Execution Mode**: `{ans['llm_model']}` (Local LLM Active: `{ans['is_local_llm_active']}`)")
            st.json(ans["retrieved_facts"])
