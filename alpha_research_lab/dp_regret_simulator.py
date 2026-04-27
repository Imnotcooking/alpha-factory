# alpha_research_lab/dp_regret_simulator.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="DP Regret Simulator", layout="wide")

st.title("⚖️ DP Regret Simulator (God Mode vs. AI)")
st.markdown("Calculate the 'Regret Gap' between the optimal perfect-hindsight trades and our RL Agent's actual trades.")

# Mock Data Generation for the visual (Replace with your DB logic later)
dates = pd.date_range(start='2024-01-01', periods=100)
god_mode_pnl = np.cumsum(np.random.normal(0.005, 0.01, 100)) # Perfect foresight
ai_agent_pnl = np.cumsum(np.random.normal(0.002, 0.015, 100)) # Reality

df = pd.DataFrame({
    'Date': dates,
    'God Mode (Perfect Hindsight)': god_mode_pnl,
    'RL Agent (Actual)': ai_agent_pnl
})
df['Regret Gap'] = df['God Mode (Perfect Hindsight)'] - df['RL Agent (Actual)']

col1, col2, col3 = st.columns(3)
col1.metric("Total God Mode Return", f"{(god_mode_pnl[-1] * 100):.2f}%")
col2.metric("Total AI Return", f"{(ai_agent_pnl[-1] * 100):.2f}%")
col3.metric("Current Regret Gap", f"{(df['Regret Gap'].iloc[-1] * 100):.2f}%", delta="Smaller is better", delta_color="inverse")

st.subheader("Performance Trajectory")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['God Mode (Perfect Hindsight)'], name="God Mode", line=dict(color='gold', width=2)))
fig.add_trace(go.Scatter(x=df['Date'], y=df['RL Agent (Actual)'], name="RL Agent", line=dict(color='cyan', width=2)))
fig.update_layout(template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("The Regret Gap (Cost of Imperfection)")
st.area_chart(df.set_index('Date')['Regret Gap'], color="#ff4b4b")

st.info("💡 **Weekend Objective:** Re-train the PPO Agent. If the red Regret Gap shrinks compared to last week, the algorithm is learning. If it widens, revert to the previous model weights.")