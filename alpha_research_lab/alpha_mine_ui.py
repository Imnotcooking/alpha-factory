import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Alpha Mine Lab", layout="wide", page_icon="🔬")
DB_PATH = "research_memory.db"

# ==========================================
# DATABASE HELPER
# ==========================================
def fetch_query(query, params=()):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# THE LEDGER (SIDEBAR)
# ==========================================
st.sidebar.markdown("## 📜 Run Ledger")
st.sidebar.caption("Historical backtest iterations.")

runs_df = fetch_query("""
    SELECT r.run_id, f.name, r.round_number, r.holdout_ic, r.timestamp, d.failure_code
    FROM backtest_runs r
    JOIN factors f ON r.factor_id = f.factor_id
    LEFT JOIN diagnostics d ON r.run_id = d.run_id
    ORDER BY r.timestamp DESC
""")

if runs_df.empty:
    st.warning("Database empty or not found. Run `python evaluator.py` first!")
    st.stop()

# Populate the sidebar with historical runs, coloring them by success/failure
for _, row in runs_df.iterrows():
    status_icon = "🔴" if pd.notna(row['failure_code']) else "🟢"
    # Format: 🔴 Range_Repair (v1) | IC: -0.012
    label = f"{status_icon} {row['name']} (v{row['round_number']})\nIC: {row['holdout_ic']:.4f}"
    st.sidebar.button(label, key=row['run_id'], use_container_width=True)

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.markdown("## 🔬 Factor Research & Diagnostics")

# --- TOP METRICS ---
c1, c2, c3, c4 = st.columns(4)
best_ic = runs_df['holdout_ic'].max()
failure_rate = (runs_df['failure_code'].notna().sum() / len(runs_df)) * 100

c1.metric("Total Experiments", len(runs_df))
c2.metric("Best Holdout IC", f"{best_ic:.4f}")
c3.metric("Candidate Factors", len(runs_df['name'].unique()))
c4.metric("Failure Rate", f"{failure_rate:.1f}%")

st.markdown("---")

# --- THE CANDIDATE MATRIX ---
st.markdown("### 🗄️ Candidate Matrix")
matrix_df = fetch_query("""
    SELECT f.name as Factor, r.round_number as Round, r.validation_ic as Val_IC, 
           r.holdout_ic as Holdout_IC, d.failure_code as Diagnostics, d.suggested_action as Next_Step
    FROM backtest_runs r
    JOIN factors f ON r.factor_id = f.factor_id
    LEFT JOIN diagnostics d ON r.run_id = d.run_id
    ORDER BY r.timestamp DESC
""")

# Styling function to highlight failed runs in red and passed runs in green
def style_matrix(row):
    color = '#D32F2F' if pd.notna(row['Diagnostics']) else '#00C853' # Red vs Green
    return [f'color: {color}; font-weight: bold' if col == 'Diagnostics' else '' for col in row.index]

st.dataframe(
    matrix_df.style.apply(style_matrix, axis=1), 
    use_container_width=True, 
    hide_index=True
)

# --- EQUITY CURVE VISUALIZATION ---
st.markdown("---")
st.markdown("### 📈 Equity Curve (Holdout Period Simulation)")
st.caption("Simulating top-decile vs bottom-decile portfolio returns against the benchmark.")

# We generate a dummy cumulative return chart for visual structure 
# (In the future, your evaluator will pass the real return array here)
days = pd.date_range(start="2023-01-01", periods=252, freq="B")
strategy = np.cumsum(np.random.normal(0.0015, 0.01, len(days))) # Simulated winning factor
benchmark = np.cumsum(np.random.normal(0.0005, 0.01, len(days))) # Simulated benchmark

fig = go.Figure()
fig.add_trace(go.Scatter(x=days, y=strategy, mode='lines', name='Factor Portfolio', line=dict(color='#00E676', width=2)))
fig.add_trace(go.Scatter(x=days, y=benchmark, mode='lines', name='Benchmark (SPY)', line=dict(color='#B0BEC5', width=1, dash='dash')))

fig.update_layout(
    template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
st.plotly_chart(fig, use_container_width=True)