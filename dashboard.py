import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import os

# Page Configuration
st.set_page_config(
    page_title="NuclearSentinel RTDAS", 
    page_icon="☢️",
    layout="wide"
)

API_URL = "http://localhost:8081/api/v1/predict"

st.title("☢️ NuclearSentinel: Real-Time Telemetry & Diagnostic System (RTDAS)")
st.markdown("**Simulated IGCAR/MAPS PHT (Primary Heat Transport) Sensor Telemetry & Autonomous Triage**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Sensor Feed (PHT Vibration, Temp & Pressure)")
    chart_placeholder = st.empty()
    
    st.markdown("### 🖼️ Computer Vision Anomaly Analysis")
    image_placeholder = st.empty()

with col2:
    st.subheader("🛡️ System Status")
    status_placeholder = st.empty()
    prob_placeholder = st.empty()
    
    st.subheader("🛠️ AI Agent Triage Report")
    report_placeholder = st.empty()

# Diagnostics Stream Controller
if st.button("🚀 Start Diagnostics Stream", type="primary"):
    for i in range(100):
        # Simulate rolling telemetry stream matching shape (15, 21)
        noise = np.random.normal(0, 0.1, (15, 21))
        trend = (i / 100.0) 
        sequence = (np.ones((15, 21)) * 0.5) + noise + trend
        
        # Plot top 3 primary PHT sensors
        df_chart = pd.DataFrame(sequence[:, :3], columns=["PHT-VIB-01", "PHT-TEMP-04", "PHT-PRESS-02"])
        chart_placeholder.line_chart(df_chart)

        payload = {"sequence": sequence.tolist()}
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                res_data = response.json()
                
                tier = res_data.get("alert_tier", "UNKNOWN")
                prob = res_data.get("anomaly_probability", 0.0)
                report = res_data.get("maintenance_report", None)
                chart_path = res_data.get("annotated_chart", None)
                
                # Update Status Tier
                if "CRITICAL" in tier:
                    status_placeholder.error(f"🚨 **{tier}**")
                elif "WARNING" in tier:
                    status_placeholder.warning(f"⚠️ **{tier}**")
                else:
                    status_placeholder.success(f"✅ **{tier}** (Nominal Operations)")
                    
                # Update Probability Metric
                prob_placeholder.metric(label="Degradation / Failure Probability", value=f"{prob:.2%}")
                
                # Display the LangGraph agent's report if generated
                if report:
                    report_placeholder.code(report, language="markdown")
                else:
                    report_placeholder.info("System operating within normal envelopes. Triage standing by.")
                
                # Display OpenCV Visual Asset if an anomaly was caught and saved locally
                if chart_path and os.path.exists(chart_path):
                    image_placeholder.image(chart_path, caption=f"OpenCV Annotated Bounding-Box Alert [Cycle {i}]")
                else:
                    image_placeholder.empty()

            else:
                status_placeholder.error(f"API Error: {response.text}")
                break
            
        except requests.exceptions.ConnectionError:
            status_placeholder.error("❌ API Offline. Check if FastAPI server (main.py) is running on port 8081.")
            break
            
        time.sleep(0.5)