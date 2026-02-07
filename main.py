import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SESSION STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kidney Viability Command Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'decision' not in st.session_state:
    st.session_state.decision = None
if 'decision_time' not in st.session_state:
    st.session_state.decision_time = None
if 'start_time' not in st.session_state:
    # Simulate an offer coming in 3 hours ago
    st.session_state.start_time = datetime.now() - timedelta(hours=3, minutes=12)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & CALCULATIONS
# -----------------------------------------------------------------------------

@st.cache_data
def calculate_kdri_kdpi(age, height_cm, weight_kg, ethnicity, hypertension, 
                       diabetes, cause_of_death, creatinine, hcv_status, dcd_status):
    """
    Calculates Estimated KDRI and maps to an approximate KDPI.
    Uses simplified coefficients based on OPTN logic for the BioHack demo.
    """
    try:
        # Convert units
        height_m = height_cm / 100.0
        weight_kg = float(weight_kg)
        
        # KDRI Coefficients (Simplified for hackathon demo)
        # Reference: Age 40, White, No HTN/DM, Creatinine 1.0
        
        # Age Factor
        if age < 18:
            x_age = (age - 50) * 0.0194 
        elif age > 50:
            x_age = (age - 50) * 0.0166
        else:
            x_age = 0 # Baseline

        # Ethnicity Factor (Removed in 2024 update, but kept as 0 for legacy logic if needed)
        x_eth = 0 
        if ethnicity == "Black/African American":
            # NOTE: OPTN removed race coefficient in 2024. 
            # We keep variable for UI but set coefficient to 0 or very low for demo accuracy.
            x_eth = 0.0 

        x_htn = 0.126 if hypertension else 0
        x_dm = 0.130 if diabetes else 0
        x_cva = 0.088 if cause_of_death == "CVA (Stroke)" else 0
        
        # Creatinine Factor
        x_cr = 0.22 * (creatinine - 1) if creatinine > 1 else 0 
        
        x_hcv = 0.24 if hcv_status == "Positive" else 0
        x_dcd = 0.13 if dcd_status else 0

        # Calculate raw KDRI (Log-scale)
        # 1.0 is the reference donor risk
        log_kdri = x_age + x_eth + x_htn + x_dm + x_cva + x_cr + x_hcv + x_dcd
        kdri_raw = np.exp(log_kdri)
        
        # Approximate mapping of KDRI to KDPI (Sigmoidal curve approximation)
        # In real clinical practice, this looks up a yearly OPTN table.
        # Logic: High KDRI (>1.5) -> High KDPI (>85%)
        kdpi_pred = 100 / (1 + np.exp(-2.5 * (kdri_raw - 1.25))) 
        kdpi_pred = max(0, min(99, kdpi_pred * 100)) # Clamp to 0-99%

        return round(kdri_raw, 2), round(kdpi_pred, 1)

    except Exception as e:
        return 0.0, 0.0

def calculate_viability_window(kdpi, fmn_level, resistance, tubular_injury_score):
    """
    Calculates how much time is LEFT before the organ is non-viable.
    Base Rule: A perfect kidney lasts 30 hours on pump.
    Penalties: High KDPI, High FMN, High Resistance reduce this window.
    """
    base_window_hours = 30.0 # Maximum theoretical limit on pump
    
    # Penalty 1: KDPI (Age/History)
    # A 99% KDPI kidney loses 12 hours of viability window immediately
    kdpi_penalty = (kdpi / 100.0) * 12.0
    
    # Penalty 2: Current FMN Level (Metabolic Distress)
    # FMN > 300 is bad.
    fmn_penalty = 0
    if fmn_level > 300:
        fmn_penalty = (fmn_level - 300) / 100.0 # lose 1 hour for every 100 units over 300
        
    # Penalty 3: Vascular Resistance
    # Resistance > 0.4 is bad
    res_penalty = 0
    if resistance > 0.4:
        res_penalty = (resistance - 0.4) * 10 # significant penalty for resistance
        
    # Penalty 4: Biopsy Score (TIS)
    tis_penalty = tubular_injury_score * 0.5 # 0.5 hours lost per point
    
    total_viability_hours = base_window_hours - kdpi_penalty - fmn_penalty - res_penalty - tis_penalty
    
    # Ensure min of 2 hours for UI stability
    return max(2.0, total_viability_hours)

def generate_perfusion_data(hours_simulated=4):
    """Generates synthetic time-series data for machine perfusion (BioHack feature)"""
    hours = np.linspace(0, hours_simulated, 40)
    
    # FMN (Flavin Mononucleotide) - Biomarker of mitochondrial damage
    fmn = 800 * np.exp(-0.8 * hours) + 150 + np.random.normal(0, 15, 40)
    
    # Renal Resistance - Measure of vascular health
    resistance = 0.5 - 0.2 * (1 - np.exp(-hours)) + np.random.normal(0, 0.01, 40)
    
    return pd.DataFrame({
        "Time (Hours)": hours,
        "FMN (au)": fmn,
        "Resistance (mmHg/mL/min)": resistance
    })

# -----------------------------------------------------------------------------
# 3. SIDEBAR: DATA INPUTS (REACTIVE)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Kidney_cross_section.jpg/640px-Kidney_cross_section.jpg", use_column_width=True)
    st.header("1. Donor Demographics")
    st.caption("Change these to see KDPI update instantly:")
    
    # Key inputs that drive KDPI
    d_age = st.slider("Age", 0, 90, 48)
    d_creat = st.slider("Creatinine (mg/dL)", 0.5, 8.0, 1.1, step=0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        d_hgt = st.number_input("Height (cm)", 50, 250, 175)
    with col2:
        d_wgt = st.number_input("Weight (kg)", 20, 200, 82)
        
    d_eth = st.selectbox("Ethnicity", ["White/Other", "Black/African American", "Asian", "Hispanic"])
    d_cod = st.selectbox("Cause of Death", ["Trauma", "CVA (Stroke)", "Anoxia", "Other"])
    
    st.subheader("Clinical History")
    d_htn = st.checkbox("Hypertension", value=True)
    d_dm = st.checkbox("Diabetes", value=False)
    d_hcv = st.selectbox("HCV Status", ["Negative", "Positive"])
    d_dcd = st.checkbox("DCD (Donation after Circulatory Death)")

    st.divider()
    
    st.header("2. BioHack Advanced Tools")
    run_ai_analysis = st.toggle("Enable AI Biopsy Analysis", value=True)
    connect_pump = st.toggle("Connect to Perfusion Pump", value=True)
    
    # Hidden Inputs for simulation logic
    # In a real app, these would come from the CSV/API
    sim_fmn = 350 # Simulated current FMN reading
    sim_res = 0.38 # Simulated current Resistance
    sim_tis = 4 # Simulated Tubular Injury Score

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD LOGIC
# -----------------------------------------------------------------------------

# --- HEADER SECTION: DECISION TIMER & SCORE ---
col_head1, col_head2, col_head3 = st.columns([2, 1, 1])

# Calculate KDPI live based on sidebar inputs
kdri_val, kdpi_val = calculate_kdri_kdpi(d_age, d_hgt, d_wgt, d_eth, d_htn, d_dm, d_cod, d_creat, d_hcv, d_dcd)

# Calculate Dynamic Time Remaining
total_viability_window = calculate_viability_window(kdpi_val, sim_fmn, sim_res, sim_tis)
elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds() / 3600
time_left_hours = total_viability_window - elapsed_time

# Formatting Time Left
if time_left_hours <= 0:
    time_display = "EXPIRED"
    time_color = "#000000"
    time_delta_color = "off"
elif time_left_hours < 2:
    time_display = f"{int(time_left_hours)}h {int((time_left_hours%1)*60)}m"
    time_color = "#dc3545" # Red - CRITICAL
    time_delta_color = "inverse"
elif time_left_hours < 6:
    time_display = f"{int(time_left_hours)}h {int((time_left_hours%1)*60)}m"
    time_color = "#ffc107" # Yellow - Warning
    time_delta_color = "normal"
else:
    time_display = f"{int(time_left_hours)}h {int((time_left_hours%1)*60)}m"
    time_color = "#28a745" # Green - Safe
    time_delta_color = "normal"


with col_head1:
    st.title("Kidney Viability Command Dashboard")
    st.caption(f"Offer ID: #UNOS-{int(time.time())} | Center: OSU Wexner Medical Center")

with col_head2:
    # THE NEW DYNAMIC COUNTER
    st.markdown(f"""
        <div style="text-align:center; border: 2px solid {time_color}; border-radius: 10px; padding: 5px; background-color: #f8f9fa;">
            <div style="font-size:12px; font-weight:bold; color:gray;">VIABILITY WINDOW REMAINING</div>
            <div style="font-size:32px; font-weight:bold; color:{time_color};">{time_display}</div>
            <div style="font-size:10px;">Based on KDPI {int(kdpi_val)}% & Perfusion Data</div>
        </div>
    """, unsafe_allow_html=True)

with col_head3:
    # Traffic Light Logic based on KDPI
    if kdpi_val < 35:
        color = "#28a745" # Green
        status = "HIGH QUALITY"
    elif kdpi_val < 85:
        color = "#ffc107" # Amber
        status = "MARGINAL"
    else:
        color = "#dc3545" # Red
        status = "HIGH RISK"
        
    st.markdown(f"""
        <div style="background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white; font-weight:bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size:14px; opacity:0.9;">VIABILITY STATUS</div>
            <div style="font-size:24px;">{status}</div>
            <div style="font-size:14px;">KDPI: {int(kdpi_val)}%</div>
        </div>
    """, unsafe_allow_html=True)

# --- DECISION BANNER (If already made) ---
if st.session_state.decision:
    if st.session_state.decision == "Accepted":
        st.success(f"✅ ORGAN ACCEPTED at {st.session_state.decision_time}. Logistics team has been notified.")
    else:
        st.error(f"❌ ORGAN DECLINED at {st.session_state.decision_time}. Offer passed to next center.")

# --- TABS FOR DETAILED DATA ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 KDPI & Labs", "🧬 Machine Perfusion", "🔬 AI Biopsy & Imaging", "🩸 Liquid Biopsy"])

# --- TAB 1: KDPI & CLINICAL SUMMARY ---
with tab1:
    col_k1, col_k2 = st.columns([1, 2])
    
    with col_k1:
        st.subheader("KDPI Score")
        # Gauge Chart for KDPI using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = kdpi_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "KDPI %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "#d4f5d4"}, # Best
                    {'range': [20, 85], 'color': "#fff4d4"}, # Standard
                    {'range': [85, 100], 'color': "#fad4d4"}], # High Risk
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85}}))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"A KDPI of {int(kdpi_val)}% means this kidney has higher risk of failure than {int(kdpi_val)}% of donors.")
    
    with col_k2:
        st.subheader("Clinical Summary")
        
        # Key Risks Highlight
        risks = []
        if d_age > 60: risks.append("Advanced Age")
        if d_creat > 1.5: risks.append("Elevated Creatinine")
        if d_htn: risks.append("History of Hypertension")
        if d_dm: risks.append("Diabetes")
        
        if risks:
            st.warning(f"⚠️ **Key Risk Factors Identified:** {', '.join(risks)}")
        else:
            st.success("✅ No major donor risk factors identified.")

        # Data Table
        summary_data = {
            "Metric": ["Age", "Creatinine", "Hypertension", "Diabetes", "Cause of Death", "Ethnicity"],
            "Value": [d_age, f"{d_creat} mg/dL", "Yes" if d_htn else "No", "Yes" if d_dm else "No", d_cod, d_eth]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# --- TAB 2: MACHINE PERFUSION (BIOHACK FEATURE) ---
with tab2:
    if connect_pump:
        st.subheader("Live Perfusion Telemetry (Hypothermic)")
        st.caption("Data streaming from organ transport device...")
        
        # Generate synthetic data
        perf_data = generate_perfusion_data()
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("#### 📉 Metabolic Decay (FMN)")
            # FMN Graph
            fig_fmn = go.Figure()
            fig_fmn.add_trace(go.Scatter(x=perf_data["Time (Hours)"], y=perf_data["FMN (au)"], 
                                       mode='lines', name='FMN', line=dict(color='#ff4b4b', width=3)))
            fig_fmn.update_layout(title="Flavin Mononucleotide Levels", xaxis_title="Hours on Pump", yaxis_title="FMN (au)", height=300)
            st.plotly_chart(fig_fmn, use_container_width=True)
            st.caption("⬇️ **Interpretation:** Decreasing FMN indicates mitochondria are recovering. Rising FMN indicates necrosis.")
            
        with col_p2:
            st.markdown("#### 🩸 Vascular Resistance")
            # Resistance Graph
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=perf_data["Time (Hours)"], y=perf_data["Resistance (mmHg/mL/min)"], 
                                       mode='lines', name='Resistance', line=dict(color='#007bff', width=3)))
            fig_res.update_layout(title="Renal Resistance Trend", xaxis_title="Hours on Pump", yaxis_title="Resistance", height=300)
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("⬇️ **Interpretation:** Stable/Lower resistance (< 0.4) indicates healthy vasculature.")
            
    else:
        st.warning("⚠️ Perfusion Pump not connected. Toggle 'Connect to Perfusion Pump' in the sidebar to view live data.")

# --- TAB 3: AI BIOPSY & IMAGING ---
with tab3:
    st.subheader("Radiology & Pathology Hub")
    
    col_img1, col_img2 = st.columns([1, 1])
    
    with col_img1:
        st.markdown("#### 1. Imaging Upload (CT/MRI)")
        uploaded_img = st.file_uploader("Upload DICOM/PNG", type=['png', 'jpg', 'dcm'])
        radiology_comments = st.text_area("Radiologist Impression", "Kidneys appear normal in size (11cm). No masses, cysts, or stones. Single renal artery bilaterally.")
        
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded Scan", use_column_width=True)
        else:
            # Placeholder if no image
            st.info("Awaiting image upload...")
            
    with col_img2:
        st.markdown("#### 2. AI Biopsy Analysis")
        st.caption("Automated Tubular Injury Score (TIS) & Glomerulosclerosis calculation")
        
        uploaded_slide = st.file_uploader("Upload Biopsy Slide", type=['png', 'jpg'])
        
        if uploaded_slide and run_ai_analysis:
            with st.spinner("Analyzing Glomeruli & Tubules (Running MATLAB Algorithm)..."):
                time.sleep(2.5) # Simulate processing time
                
                # Mock AI Results
                score_glom = np.random.randint(2, 18)
                score_tis = np.random.randint(1, 4)
                
                st.success("Analysis Complete")
                
                m1, m2 = st.columns(2)
                m1.metric("Glomerulosclerosis", f"{score_glom}%", "Normal < 10%")
                m2.metric("Tubular Injury Score", f"{score_tis}/10", "Low Risk")
                
                st.progress(score_glom / 100, text=f"Fibrosis Density: {score_glom}%")
                
                if score_tis > 5:
                    st.error("AI Alert: Significant tubular dilation detected.")
                else:
                    st.info("AI Note: Tubular structure appears intact.")
                    
        elif not uploaded_slide:
            st.info("Upload a slide to run the AI viability check.")

# --- TAB 4: LIQUID BIOPSY ---
with tab4:
    st.subheader("Liquid Biopsy & Molecular Biomarkers")
    st.markdown("Advanced non-invasive testing for rejection risk.")
    
    c_bio1, c_bio2 = st.columns(2)
    with c_bio1:
        dd_cfdna = st.number_input("Donor-Derived Cell-Free DNA (%)", 0.0, 5.0, 0.45, step=0.01)
        if dd_cfdna > 1.0:
            st.error("⚠️ **High Risk:** dd-cfDNA > 1% indicates potential injury.")
        else:
            st.success("✅ **Low Risk:** dd-cfDNA is within normal range (< 1%).")
            
    with c_bio2:
        biomarker_cxcl10 = st.slider("CXCL10 Levels (pg/mL)", 0, 500, 45)
        if biomarker_cxcl10 > 100:
             st.warning("⚠️ Elevated CXCL10 correlates with inflammation.")
        else:
             st.caption("CXCL10 levels are baseline.")

# -----------------------------------------------------------------------------
# 5. FOOTER / ACTION BUTTONS
# -----------------------------------------------------------------------------
st.divider()

# Only show buttons if a decision hasn't been made yet
if st.session_state.decision is None:
    col_act1, col_act2, col_act3 = st.columns([1, 1, 3])

    with col_act1:
        if st.button("✅ ACCEPT OFFER", type="primary", use_container_width=True):
            st.session_state.decision = "Accepted"
            st.session_state.decision_time = datetime.now().strftime("%H:%M:%S")
            st.rerun()
            
    with col_act2:
        if st.button("❌ DECLINE OFFER", type="secondary", use_container_width=True):
            st.session_state.decision = "Declined"
            st.session_state.decision_time = datetime.now().strftime("%H:%M:%S")
            st.rerun()

    with col_act3:
        st.caption("Authorized Surgeon: Dr. Alex Mian | Session ID: 9482-ADF | Encrypted Connection")
else:
    # If decision is made, show reset button
    if st.button("Reset Dashboard (Demo Mode)"):
        st.session_state.decision = None
        st.session_state.decision_time = None
        st.rerun()