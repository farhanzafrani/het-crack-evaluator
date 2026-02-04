import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from database import fetch_het_data, get_db_connection
from statistics_logic import filter_data, calculate_quantiles, convert_threshold
import os

# Page Config
st.set_page_config(
    page_title="HET EdgeCrackEvaluator",
    page_icon="🏎️",
    layout="wide"
)

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Session State Initialization
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'db_credentials' not in st.session_state:
    st.session_state.db_credentials = {}
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = pd.DataFrame()

# DB Credential Popup/View
if not st.session_state.db_connected:
    st.title("Database Connection")
    with st.container():
        st.info("Please enter your PostgreSQL credentials to access the HET EdgeCrackEvaluator.")
        col1, col2 = st.columns(2)
        with col1:
            db_host = st.text_input("Host", value="localhost")
            db_name = st.text_input("Database Name", value="het_db")
        with col2:
            db_user = st.text_input("User", value="postgres")
            db_pass = st.text_input("Password", type="password")
        
        if st.button("Connect to Database"):
            try:
                # conn = get_db_connection(db_host, db_name, db_user, db_pass)
                conn = None
                st.session_state.db_connected = True
                # Store minimal info if needed, or just relying on connected state
                st.session_state.db_credentials = {
                    'host': db_host,
                    'database': db_name,
                    'user': db_user,
                    'password': db_pass
                }
                
                with st.spinner("Fetching data..."):
                    st.session_state.raw_data = fetch_het_data(conn)
                    # conn.close()
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")
    st.stop()

# --- MAIN APP UI ---

st.sidebar.title("Configuration")

# Sidebar - Specification Filters
st.sidebar.header("Data Selection")
df = st.session_state.raw_data

# Dynamic filter options
materials = sorted(df['material'].unique())
suppliers = ['Keine Auswahl'] + sorted(df['supplier'].unique().tolist())
clearances = ['Keine Auswahl'] + sorted(df['clearance'].unique().tolist())
timestamps = ['Keine Auswahl'] + sorted(df['timeStampMeas'].unique().tolist())

sel_material = st.sidebar.selectbox("Material", materials)
sel_supplier = st.sidebar.multiselect("Supplier", suppliers, default=['Keine Auswahl'])
sel_clearance = st.sidebar.multiselect("Test Clearance", clearances, default=['Keine Auswahl'])
sel_timestamp = st.sidebar.multiselect("Timestamp Testing", timestamps, default=['Keine Auswahl'])

col_t1, col_t2 = st.sidebar.columns(2)
min_thick = col_t1.number_input("Min Thick [mm]", value=0.0, step=0.1)
max_thick = col_t2.number_input("Max Thick [mm]", value=5.0, step=0.1)

# Sidebar - Evaluation Settings
st.sidebar.divider()
st.sidebar.header("Evaluation Settings")
eval_type = st.sidebar.radio("Evaluation Type", ["Hole Expansion Coefficient", "Major Strain"])
method = st.sidebar.radio("Method", ["Direct Global Quantile", "Global Quantile Prediction (Gauss)"])

# Quantiles
q1_level = st.sidebar.number_input("Global Quantile 1", value=0.05, min_value=0.0, max_value=1.0)
q2_level = st.sidebar.number_input("Global Quantile 2", value=0.50, min_value=0.0, max_value=1.0)
charge_q_level = st.sidebar.number_input("Exp. Set Quantile", value=0.50, min_value=0.0, max_value=1.0)

# Threshold
add_threshold = st.sidebar.checkbox("Add Evaluation Threshold", value=False)
threshold_val = st.sidebar.number_input("Threshold Value", value=40.0)

# threshold_val = convert_threshold(threshold_val, eval_type)

# Display Options
st.sidebar.divider()
show_delimiting = st.sidebar.checkbox("Show Delimiting Lines", value=True)
show_exp_quantile = st.sidebar.checkbox("Show Quantile of Exp. Set", value=True)
show_q1 = st.sidebar.checkbox("Show Global Quantile 1", value=True)
show_q2 = st.sidebar.checkbox("Show Global Quantile 2", value=False)
text_size = st.sidebar.slider("Text Description Size (%)", 0, 100, 80)

# Filter Data
specs = {
    'material': sel_material,
    'supplier': sel_supplier,
    'clearance': sel_clearance,
    'timeStampMeas': sel_timestamp,
    'minthick': min_thick,
    'maxthick': max_thick,
    'min_n_spec': 5 # Constant for now
}
filtered_df = filter_data(df, specs)

# --- MAIN AREA ---

tabs = st.tabs(["Material Selection", "Evaluation & Plot", "Raw Data"])

with tabs[0]:
    st.header(f"Selected Material: {sel_material}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Evaluated Experiments", len(filtered_df))
    n_specs = filtered_df['measdata_HET'].apply(len).sum() if not filtered_df.empty else 0
    col2.metric("Total Specimens", n_specs)
    
    st.subheader("Experiment Selection")
    all_labs = filtered_df['LabProt'].tolist()
    selected_labs = st.multiselect("Select Experiments show in Plot", all_labs, default=all_labs)
    
    # Update filter with chosen labs
    specs['selected_labs'] = selected_labs
    final_df = filter_data(filtered_df, specs)

with tabs[1]:
    if final_df.empty:
        st.error("Zero remaining data sets. Please check filters.")
    else:
        # Statistical Analysis
        stats_q1 = calculate_quantiles(final_df, eval_type, q1_level)
        stats_q2 = calculate_quantiles(final_df, eval_type, q2_level)
        stats_charge = calculate_quantiles(final_df, eval_type, charge_q_level)
        
        # PLOTLY REPLICATION
        fig = go.Figure()
        
        # Data consolidation for plotting
        x_indices = []
        y_values = []
        delimiting_lines = [0.5]
        
        data_col = 'measdata_HET' if eval_type == "Hole Expansion Coefficient" else 'measdata_Strain'
        
        current_idx = 1
        for i, (_, row) in enumerate(final_df.iterrows()):
            meas = row[data_col]
            x_range = list(range(current_idx, current_idx + len(meas)))
            x_indices.extend(x_range)
            y_values.extend(meas)
            
            # Scatter for measurements
            fig.add_trace(go.Scatter(
                x=x_range, y=meas,
                mode='markers',
                marker=dict(color='#009900', size=6, symbol='diamond'),
                name='Measurement Data' if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo='text',
                text=[f"Lab: {row['LabProt']}<br>Value: {m:.2f}" for m in meas]
            ))
            
            # Exp Quantile Line
            if show_exp_quantile:
                q_val = stats_charge['per_experiment'][i]['q_gauss' if method.startswith("Global") else 'q_direct']
                fig.add_trace(go.Scatter(
                    x=[current_idx-0.5, current_idx + len(meas) - 0.5],
                    y=[q_val, q_val],
                    mode='lines',
                    line=dict(color='red', dash='dot', width=2),
                    name='Exp. Set Quantile' if i == 0 else None,
                    showlegend=(i == 0)
                ))
                fig.add_annotation(
                    x=current_idx, y=q_val,
                    text=f"{q_val:.2f}",
                    showarrow=False, font=dict(color='red', size=10),
                    yshift=10
                )

            current_idx += len(meas)
            delimiting_lines.append(current_idx - 0.5)

        # Global Lines
        total_len = current_idx - 1
        x_span = [0.5, total_len + 0.5]
        
        if show_q1:
            q1_val = stats_q1['global']['q_gauss' if method.startswith("Global") else 'q_direct']
            fig.add_trace(go.Scatter(
                x=x_span, y=[q1_val, q1_val],
                mode='lines',
                line=dict(color='blue', dash='dot', width=2),
                name=f"Global Q1 ({q1_level})"
            ))
            fig.add_annotation(
                x=total_len*0.05, y=q1_val,
                text=f"Global Q1: {q1_val:.2f}",
                showarrow=False, font=dict(color='blue', size=12, weight='bold'),
                yshift=-15
            )

        if show_q2:
            q2_val = stats_q2['global']['q_gauss' if method.startswith("Global") else 'q_direct']
            fig.add_trace(go.Scatter(
                x=x_span, y=[q2_val, q2_val],
                mode='lines',
                line=dict(color='black', dash='dot', width=2),
                name=f"Global Q2 ({q2_level})"
            ))

        if add_threshold:
            fig.add_trace(go.Scatter(
                x=x_span, y=[threshold_val, threshold_val],
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                name="Threshold"
            ))

        # Delimiting Lines
        if show_delimiting:
            for dl in delimiting_lines[1:-1]:
                fig.add_vline(x=dl, line_width=1, line_dash="dash", line_color="black")

        # Layout
        fig.update_layout(
            title=f"HEC Evaluation - {sel_material}<br><sup>Method: {method}</sup>",
            xaxis_title="Measurement Index [-]",
            yaxis_title="Hole Expansion Coefficient [%]" if eval_type.startswith("Hole") else "Major Strain [-]",
            template="plotly_white",
            height=600,
            margin=dict(l=20, r=20, t=60, b=100),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Info Block at bottom (Simulated with annotations for now)
        if text_size > 0:
            for i, (_, row) in enumerate(final_df.iterrows()):
                x_pos = (delimiting_lines[i] + delimiting_lines[i+1]) / 2
                info_text = f"Lab: {row['LabProt']}<br>{row['supplier']}<br>{row['thick']:.2f}mm"
                fig.add_annotation(
                    x=x_pos, y=0,
                    text=info_text,
                    showarrow=False,
                    yshift=-60,
                    font=dict(size=12 * (text_size/100)),
                    align='center'
                )

        st.plotly_chart(fig, width='stretch')

with tabs[2]:
    st.subheader("Raw Data View")
    st.dataframe(final_df if not final_df.empty else df)

# Footer
st.divider()
st.caption("HET EdgeCrackEvaluator - Python3 v1.0")
