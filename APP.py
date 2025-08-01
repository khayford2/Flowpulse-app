import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="FlowPulse Predictor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .well-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-offline { background-color: #dc3545; }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Define the directory where model and scaler files are stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# Function to load models and scalers
@st.cache_resource
def load_models():
    """Load all models and P2/P1 scalers with error handling"""
    models = {}
    scalers = {}
    
    model_files = {
        'p2_pt': 'P2_PT.pkl',
        'p2_tt': 'P2_TT.pkl',
        'p1_pt': 'P1_PT.pkl',
        'p1_tt': 'P1_TT.pkl',
        'rb_pt': 'RB_PT.pkl',
        'rb_tt': 'RB_TT.pkl'
    }
    
    scaler_files = {
        'p2': 'scaler_p2.pkl',
        'p2t':'scaler_p2t.pkl',
        'p1': 'scaler_p1.pkl',
        'p1t':'scaler_p1t.pkl',
        'rb':'scaler_RB.pkl',
        'rbt':'scaler_RBt.pkl'
    }
    
    # Display current working directory for debugging
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Model directory: {MODEL_DIR}")
    
    # Load models
    for key, filename in model_files.items():
        file_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(file_path):
            st.error(f"Model file not found: {file_path}")
            return None, None
        try:
            models[key] = joblib.load(file_path)
        except Exception as e:
            st.error(f"Error loading model {file_path}: {str(e)}")
            return None, None
    
    # Load scalers for P2 and P1
    for key, filename in scaler_files.items():
        file_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(file_path):
            st.error(f"Scaler file not found: {file_path}. Please train models with scalers.")
            return None, None
        try:
            scalers[key] = joblib.load(file_path)
        except Exception as e:
            st.error(f"Error loading scaler {file_path}: {str(e)}")
            return None, None
    
    return models, scalers

# Header
st.markdown("""
<div class="main-header">
    <h1>🛢️ FlowPulse Predictor</h1>
    <p>Real-time Pressure & Temperature Prediction for Wells JX-23, JX-53, and JX-71</p>
</div>
""", unsafe_allow_html=True)

# Load models and scalers
models, scalers = load_models()

# Sidebar for system status and controls
with st.sidebar:
    st.markdown("## 🔧 System Control Panel")
    
    # System status
    st.markdown("### System Status")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Check if models and scalers are loaded
    models_status = "online" if models is not None and scalers is not None else "offline"
    models_indicator = "status-online" if models is not None and scalers is not None else "status-offline"
    
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <div><span class="status-indicator status-online"></span>System Online</div>
        <div><span class="status-indicator {models_indicator}"></span>Models & Scalers {models_status.title()}</div>
        <div><span class="status-indicator status-online"></span>Data Connection</div>
        <div style="margin-top: 0.5rem; color: #666;">Last Update: {current_time}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### Quick Actions")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.button("📊 Generate Report"):
        st.info("Report generation feature coming soon!")
    
    if st.button("⚙️ System Settings"):
        st.info("Settings panel coming soon!")
    
    # Model information
    st.markdown("### 🤖 Model Information")
    st.info("""
    **Active Models:**
    - P2: Random Forest
    - P1: Random Forest
    - RB: Decision Tree
    
    **Prediction Pipeline:**
    1. P2 uses JX-23 data (scaled)
    2. P1 uses all wells + P2 predictions (scaled)
    3. RB uses cumulative data + P1 predictions (unscaled)
    """)

# Well input sections
st.markdown("## 📊 Well Input Parameters")

tab1, tab2, tab3 = st.tabs(["🏭 JX-23 Well", "🏭 JX-53 Well", "🏭 JX-71 Well"])

with tab1:
    st.markdown('<div class="well-section">', unsafe_allow_html=True)
    st.markdown("### JX-23 Well Parameters")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Flow Rates**")
        jx23_oil = st.number_input("Oil Flowrate (sm3/hr)", min_value=0.0, value=50.0, step=1.0, key="jx23_oil")
        jx23_water = st.number_input("Water Flowrate (sm3/hr)", min_value=0.0, value=50.0, step=1.0, key="jx23_water")
        jx23_gas = st.number_input("Gas Flowrate (sm3/hr)", min_value=0.0, value=1000.0, step=10.0, key="jx23_gas")
    
    with col_b:
        st.markdown("**Sensor Readings**")
        jx23_pt1 = st.number_input("PT1-JX23 Pressure (bar)", min_value=0.0, value=2500.0, step=10.0, key="jx23_pt1")
        jx23_pt2 = st.number_input("PT2-JX23 Pressure (bar)", min_value=0.0, value=2400.0, step=10.0, key="jx23_pt2")
        jx23_tt1 = st.number_input("TT1-JX23 Temperature (°C)", min_value=0.0, value=180.0, step=1.0, key="jx23_tt1")
        jx23_tt2 = st.number_input("TT2-JX23 Temperature (°C)", min_value=0.0, value=175.0, step=1.0, key="jx23_tt2")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="well-section">', unsafe_allow_html=True)
    st.markdown("### JX-53 Well Parameters")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Flow Rates**")
        jx53_oil = st.number_input("Oil Flowrate (sm3/hr)", min_value=0.0, value=200.0, step=1.0, key="jx53_oil")
        jx53_water = st.number_input("Water Flowrate (sm3/hr)", min_value=0.0, value=75.0, step=1.0, key="jx53_water")
        jx53_gas = st.number_input("Gas Flowrate (sm3/hr)", min_value=0.0, value=1200.0, step=10.0, key="jx53_gas")
    
    with col_b:
        st.markdown("**Sensor Readings**")
        jx53_pt1 = st.number_input("PT1-JX53 Pressure (bar)", min_value=0.0, value=2600.0, step=10.0, key="jx53_pt1")
        jx53_pt2 = st.number_input("PT2-JX53 Pressure (bar)", min_value=0.0, value=2500.0, step=10.0, key="jx53_pt2")
        jx53_tt1 = st.number_input("TT1-JX53 Temperature (°C)", min_value=0.0, value=185.0, step=1.0, key="jx53_tt1")
        jx53_tt2 = st.number_input("TT2-JX53 Temperature (°C)", min_value=0.0, value=180.0, step=1.0, key="jx53_tt2")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="well-section">', unsafe_allow_html=True)
    st.markdown("### JX-71 Well Parameters")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Flow Rates**")
        jx71_oil = st.number_input("Oil Flowrate (sm3/hr)", min_value=0.0, value=180.0, step=1.0, key="jx71_oil")
        jx71_water = st.number_input("Water Flowrate (sm3/hr)", min_value=0.0, value=60.0, step=1.0, key="jx71_water")
        jx71_gas = st.number_input("Gas Flowrate (sm3/hr)", min_value=0.0, value=1100.0, step=10.0, key="jx71_gas")
    
    with col_b:
        st.markdown("**Sensor Readings**")
        jx71_pt1 = st.number_input("PT1-JX71 Pressure (bar)", min_value=0.0, value=2550.0, step=10.0, key="jx71_pt1")
        jx71_pt2 = st.number_input("PT2-JX71 Pressure (bar)", min_value=0.0, value=2450.0, step=10.0, key="jx71_pt2")
        jx71_tt1 = st.number_input("TT1-JX71 Temperature (°C)", min_value=0.0, value=400.0, step=1.0, key="jx71_tt1")
        jx71_tt2 = st.number_input("TT2-JX71 Temperature (°C)", min_value=0.0, value=400.0, step=1.0, key="jx71_tt2")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Cumulative flowrates display
st.markdown("## 📈 Cumulative Production Summary")

col1, col2 = st.columns([2, 1])

with col1:
    # Calculate cumulative flowrates
    cumulative_oil = jx23_oil + jx53_oil + jx71_oil
    cumulative_water = jx23_water + jx53_water + jx71_water
    cumulative_gas = jx23_gas + jx53_gas + jx71_gas
    
    # Display cumulative values
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Oil Production", f"{cumulative_oil:.1f} sm3/hr", 
                 delta=f"{cumulative_oil/3:.1f} avg per well")
    with col_b:
        st.metric("Total Water Production", f"{cumulative_water:.1f} sm3/hr", 
                 delta=f"{cumulative_water/3:.1f} avg per well")
    with col_c:
        st.metric("Total Gas Production", f"{cumulative_gas:.1f} sm3/hr", 
                 delta=f"{cumulative_gas/3:.1f} avg per well")

with col2:
    # Create a pie chart for production breakdown
    fig = go.Figure(data=[go.Pie(
        labels=['JX-23', 'JX-53', 'JX-71'],
        values=[jx23_oil, jx53_oil, jx71_oil],
        hole=0.3,
        title="Oil Production by Well"
    )])
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Prediction button and results
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Disable prediction button if models or scalers are not loaded
    if models is None or scalers is None:
        st.button("🔮 Generate Predictions", key="predict_button", disabled=True)
        st.error("❌ Prediction disabled: Models or scalers not loaded. Please check file paths.")
    else:
        if st.button("🔮 Generate Predictions", key="predict_button"):
            with st.spinner("Generating predictions..."):
                try:
                    # Prepare P2 input (6 features)
                    p2_input = np.array([[jx23_oil, jx23_gas, jx23_pt1, jx23_pt2, jx23_tt1, jx23_tt2]])
                    t2_input=np.array([[jx23_oil, jx23_gas, jx23_pt1, jx23_pt2, jx23_tt1, jx23_tt2]])
                    if p2_input.shape[1] != 6:
                        raise ValueError(f"P2 model expects 6 features, but got {p2_input.shape[1]}")
                    
                    # Scale P2 input
                    p2_input_scaled = scalers['p2'].transform(p2_input)
                    t2_input_scaled= scalers['p2t'].transform(t2_input)
                    
                    # Predict P2
                    p2_pressure = models['p2_pt'].predict(p2_input_scaled)[0]
                    p2_temperature = models['p2_tt'].predict(p2_input_scaled)[0] 
                    
                    # Prepare P1 input (8 features)
                    p1_input = np.array([[cumulative_oil, cumulative_water, cumulative_gas, 
                                        jx53_pt2, jx53_tt1, jx71_pt2, jx71_tt1, p2_pressure]])
                    p1t_input = np.array([[cumulative_oil, cumulative_water, cumulative_gas, 
                                         jx53_pt2, jx53_tt1, jx71_pt2, jx71_tt1, p2_temperature]])
                    if p1_input.shape[1] != 8:
                        raise ValueError(f"P1 model expects 8 features, but got {p1_input.shape[1]}")
                    
                    # Scale P1 input
                    p1_input_scaled = scalers['p1'].transform(p1_input)
                    p1t_input_scaled = scalers['p1t'].transform(p1t_input)
                    
                    # Predict P1
                    p1_pressure = models['p1_pt'].predict(p1_input_scaled)[0] 
                    p1_temperature = models['p1_tt'].predict(p1t_input_scaled)[0] 
                    
                    # Prepare RB input (5 features)
                    rb_input = np.array([[cumulative_oil, cumulative_water, cumulative_gas, 
                                        p1_pressure]])
                    rb_inputt = np.array([[cumulative_oil, cumulative_water, cumulative_gas, 
                                        p1_temperature]])
                    if rb_input.shape[1] != 4:
                        raise ValueError(f"RB model expects 4 features, but got {rb_input.shape[1]}")
                    rb_input_scaled = scalers['rb'].transform(rb_input)
                    rbt_input_scaled = scalers['rbt'].transform(rb_inputt)
                    
                    # Predict RB 
                    rb_pressure = models['rb_pt'].predict(rb_input_scaled)[0]
                    rb_temperature = models['rb_tt'].predict(rbt_input_scaled)[0]
                    
                    # Display predictions with enhanced styling
                    st.markdown("## 🎯 Prediction Results")
                    
                    # Create columns for prediction results
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🔧 Manifold 1 (P2)</h3>
                            <div style="font-size: 1.2em; margin: 10px 0;">
                                <strong>Pressure:</strong> {p2_pressure:.2f} bar
                            </div>
                            <div style="font-size: 1.2em;">
                                <strong>Temperature:</strong> {p2_temperature:.2f} °C
                            </div>
                            <div style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                                Model: Random Forest (Scaled Input)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🔧 Manifold 2 (P1)</h3>
                            <div style="font-size: 1.2em; margin: 10px 0;">
                                <strong>Pressure:</strong> {p1_pressure:.2f} bar
                            </div>
                            <div style="font-size: 1.2em;">
                                <strong>Temperature:</strong> {p1_temperature:.2f} °C
                            </div>
                            <div style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                                Model: Decision Tree (Scaled Input)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col3:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🔧 Riser Base (RB)</h3>
                            <div style="font-size: 1.2em; margin: 10px 0;">
                                <strong>Pressure:</strong> {rb_pressure:.2f} bar
                            </div>
                            <div style="font-size: 1.2em;">
                                <strong>Temperature:</strong> {rb_temperature:.2f} °C
                            </div>
                            <div style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                                Model: Decision Tree (Unscaled Input)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create visualization of predictions
                    st.markdown("### 📈 Prediction Visualization")
                    
                    locations = ['Manifold 1 (P2)', 'Manifold 2 (P1)', 'Riser Base (RB)']
                    pressures = [p2_pressure, p1_pressure, rb_pressure]
                    temperatures = [p2_temperature, p1_temperature, rb_temperature]
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Pressure Profile", "Temperature Profile"),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=locations, y=pressures,
                            mode='lines+markers',
                            name="Pressure (bar)",
                            line=dict(color='rgba(30, 60, 114, 0.8)', width=3),
                            marker=dict(size=10)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=locations, y=temperatures,
                            mode='lines+markers',
                            name="Temperature (°C)",
                            line=dict(color='rgba(255, 99, 132, 0.8)', width=3),
                            marker=dict(size=10)
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        title_text="Predicted Values Through System",
                        title_x=0.5
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction summary table
                    st.markdown("### 📋 Prediction Summary")
                    
                    summary_data = {
                        'Location': ['Manifold 1 (P2)', 'Manifold 2 (P1)', 'Riser Base (RB)'],
                        'Pressure (bar)': [f"{p2_pressure:.2f}", f"{p1_pressure:.2f}", f"{rb_pressure:.2f}"],
                        'Temperature (°C)': [f"{p2_temperature:.2f}", f"{p1_temperature:.2f}", f"{rb_temperature:.2f}"],
                        'Model Type': ['Esembled', 'Random Forest', 'Esembled'],
                        
                    }
                    
                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Success message
                    st.success("✅ Predictions generated successfully!")
                    
                except ValueError as ve:
                    st.error(f"❌ Input validation error: {str(ve)}")
                    st.info("💡 Please ensure input data matches the expected features for each model.")
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.info("💡 Please check that all model and scaler files are compatible with the input data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🛢️ Oil Field Sensor Monitoring System | Predictive Analytics for Wells JX-23, JX-53, JX-71</p>
    <p>THIS PREDICTIVE APP CAN MAKE MISTAKES SO BE CAREFUL WHEN USING IT!</p>
    <p><strong>Prediction Pipeline:</strong> Dev Kelvin Hayford</p>
</div>
""", unsafe_allow_html=True)
