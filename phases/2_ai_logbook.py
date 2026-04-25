import streamlit as st
import pandas as pd
import sqlite3
import os

st.markdown("## 📖 AI Execution Ledger")
st.caption("Immutable record of all autonomous trades executed by the Cron Engine.")

DB_PATH = "portfolio_memory.db"

if not os.path.exists(DB_PATH):
    st.warning("Database not found. Waiting for Cron Engine to execute first trade...")
    st.stop()

conn = sqlite3.connect(DB_PATH)
trades_df = pd.read_sql_query("SELECT * FROM trade_history ORDER BY timestamp DESC", conn)
conn.close()

if trades_df.empty:
    st.info("No trades executed yet.")
else:
    # Formatting for mobile readability
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    trades_df['execution_price'] = trades_df['execution_price'].apply(lambda x: f"${x:,.2f}")
    trades_df['Value'] = (trades_df['quantity'] * trades_df['execution_price'].replace('[\$,]', '', regex=True).astype(float)).apply(lambda x: f"${x:,.2f}")
    
    display_df = trades_df[['timestamp', 'action', 'quantity', 'ticker', 'execution_price', 'strategy']]
    display_df.columns = ['Time', 'Side', 'Qty', 'Asset', 'Price', 'Agent']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)