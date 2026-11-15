import streamlit as st
import requests
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="NuclearSentinel RTDAS", layout="wide")
API_URL = "http://localhost:8080/api/v1/predict" 

st.title("☢️ NuclearSentinel: Predictive Maintenance")
st.markdown("**Simulated IGCAR/MAPS PHT (Primary Heat Transport) Sensor Telemetry**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Feed (Vibration & Temp)")
    chart_placeholder = st.empty()

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    prob_placeholder = st.empty()
    
    st.subheader("AI Agent Triage")
    report_placeholder = st.empty() # Placeholder for the LangGraph report

if st.button("Start Diagnostics Stream"):
    for i in range(100):
        noise = np.random.normal(0, 0.1, (15, 21))
        trend = (i / 100.0) 
        sequence = (np.ones((15, 21)) * 0.5) + noise + trend
        
        df_chart = pd.DataFrame(sequence[:, :3], columns=["PHT-VIB-01", "PHT-TEMP-04", "PHT-PRESS-02"])
        chart_placeholder.line_chart(df_chart)

        payload = {"sequence": sequence.tolist()}
        try:
            response = requests.post(API_URL, json=payload).json()
            
            tier = response.get("alert_tier", "UNKNOWN")
            prob = response.get("anomaly_probability", 0.0)
            report = response.get("maintenance_report", None)
            
            if "CRITICAL" in tier:
                status_placeholder.error(f"🚨 **{tier}**")
            elif "WARNING" in tier:
                status_placeholder.warning(f"⚠️ **{tier}**")
            else:
                status_placeholder.success(f"✅ **{tier}**")
                
            prob_placeholder.metric(label="Failure Probability", value=f"{prob:.2%}")
            
            # Display the LangGraph agent's report if one was generated
            if report:
                report_placeholder.code(report, language="markdown")
            else:
                report_placeholder.info("No active maintenance alerts.")
            
        except requests.exceptions.ConnectionError:
            status_placeholder.error("API Offline. Check FastAPI server.")
            break
            
        time.sleep(0.5)