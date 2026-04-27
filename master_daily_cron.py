import os
import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import yfinance as yf
import torch
from datetime import datetime
from stable_baselines3 import PPO
from dotenv import load_dotenv
from ib_insync import IB, Stock, MarketOrder

# Import your V2.0 Strategy Desk PMs and Optimizer
from strategy_agents.agent_001_macro_trend import MacroTrendAgent
from strategy_agents.agent_002_fast_reversion import FastReversionAgent
from portfolio_optimizer import ConvexOptimizer

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
TRADING_MODE = os.getenv("TRADING_MODE", "PAPER")
UNIVERSE = os.getenv("OPERATING_UNIVERSE", "MACRO")
DB_PATH = "portfolio_memory.db"

print(f"============== ALPHA FACTORY CRON JOB ==============")
print(f"STATUS: Waking up at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"MODE: {TRADING_MODE} TRADING")
print(f"UNIVERSE: {UNIVERSE}")
print(f"====================================================")

# 2. CONFIGURE THE CIO BRAIN
if UNIVERSE == "MACRO":
    codebook_path = "models/vq_codebook_institutional_macro.pt"
    policy_path = "models/horizon_policy_macro.zip"
    data_lake_path = "offline_quant_lab/data/institutional_macro.parquet"
    ticker = "SPY"
else:
    codebook_path = "models/vq_codebook_highbeta_meme.pt"
    policy_path = "models/horizon_policy_meme.zip"
    data_lake_path = "offline_quant_lab/data/crypto_meme.parquet"
    ticker = "NVDA"

# 3. LIVE STATE GENERATION (The Weather Report)
print(f"[1/5] Downloading live market state...")
raw_data = yf.download(ticker, period="1y", progress=False)['Close']
df_pd = raw_data.reset_index()
df_pd.columns = ['Date', 'Close']
lf = pl.LazyFrame(df_pd)

lf = lf.with_columns([
    (pl.col("Close") / pl.col("Close").shift(1)).log().alias("Ret_1D"),
    (pl.col("Close") / pl.col("Close").shift(20)).log().alias("Ret_20D"),
    ((pl.col("Close") / pl.col("Close").shift(1)).log().rolling_std(window_size=20) * (252 ** 0.5)).alias("Vol_20D"),
]).drop_nulls()

latest_state = lf.collect().tail(1)

# 4. CIO INFERENCE (PPO Archetype & Beta Mapping)
print("[2/5] CIO processing State Space matrix...")
payload = torch.load(codebook_path, map_location=torch.device('cpu'), weights_only=False)
expected_features = payload.get('features', [c for c in latest_state.columns if c not in ['Date', 'Close']])

raw_vector = latest_state.select(expected_features).to_numpy()[0]
scaled_vector = (raw_vector - payload['scaler_mean']) / payload['scaler_scale']

ai_commander = PPO.load(policy_path, device="cpu")
action, _ = ai_commander.predict(scaled_vector, deterministic=True)
action_idx = int(action)

# V2.0 Beta Mapping based on Archetype
target_beta = 1.0
regime_name = "Unknown"
regime_color = "normal"
active_agent_name = "Standby"

if action_idx in [0, 1, 2]:
    regime_name = f"Volatility Shock (Archetype {action_idx})"
    regime_color = "error" 
    target_beta = 0.3      
    print(f"      -> Regime: {regime_name}. Target Beta: {target_beta}")
    
    # In a panic, we activate the Fast Reversion desk to buy the dips
    active_agent = FastReversionAgent() 
    
elif action_idx in [5, 6, 7]:
    regime_name = f"Bull Momentum (Archetype {action_idx})"
    regime_color = "success" 
    target_beta = 1.0        
    print(f"      -> Regime: {regime_name}. Target Beta: {target_beta}")
    
    # In a bull market, we activate the Macro Trend desk to ride the wave
    active_agent = MacroTrendAgent()
else:
    regime_name = f"Neutral/Choppy (Archetype {action_idx})"
    regime_color = "warning" 
    target_beta = 0.0
    print(f"      -> Regime: {regime_name}. CIO commands 100% Cash.")
    active_agent = None

# 5. STRATEGY ROUTING & CONVEX OPTIMIZATION
print("[3/5] Agent Scoring & SciPy Optimization...")
final_orders = pd.DataFrame()

# Get total capital from DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS portfolio_metrics (date TEXT UNIQUE, available_cash REAL, total_nav REAL)")
cursor.execute("SELECT available_cash FROM portfolio_metrics ORDER BY date DESC LIMIT 1")
res = cursor.fetchone()
total_capital = res[0] if res else 100000.0

if active_agent and target_beta > 0.0:
    df_lake = pl.read_parquet(data_lake_path)
    
    # Get the raw fractional weights from the Agent
    agent_signals = active_agent.generate_signals(df_lake)
    active_agent_name = active_agent.agent_name
    
    if not agent_signals.empty:
        # Create a mock identity covariance matrix for today (You will upgrade this later)
        n_assets = len(agent_signals)
        cov_matrix = np.eye(n_assets) * 0.0004 
        
        # Route to the SciPy Phase 3 Optimizer
        optimizer = ConvexOptimizer(target_beta=target_beta, total_capital=total_capital)
        final_orders = optimizer.optimize(agent_signals, cov_matrix)

# 6. IBKR EXECUTION & STREAMLIT DB LOGGING
print("[4/5] Engaging IBKR Execution Protocol...")
total_spent = 0.0
daily_pnl = 0.0 # Will calculate from IBKR later

if not final_orders.empty:
    ib_port = 7497 if TRADING_MODE == "PAPER" else 7496
    try:
        ib = IB()
        ib.connect('172.17.0.1', ib_port, clientId=99)
        
        for index, row in final_orders.iterrows():
            target_ticker = row['symbol']
            shares = int(row['Target_Shares'])
            
            if shares <= 0: continue
            print(f"      -> Transmitting Market BUY for {shares} shares of {target_ticker}...")
            contract = Stock(target_ticker, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            order = MarketOrder('BUY', shares)
            trade = ib.placeOrder(contract, order)
            
            while not trade.isDone():
                ib.sleep(0.1)
                
            fill_price = trade.orderStatus.avgFillPrice
            total_spent += (fill_price * shares)
            print(f"         [FILLED] {shares} @ ${fill_price:.2f}")
            
        ib.disconnect()
    except Exception as e:
        print(f"❌ CRITICAL EXECUTION FAILURE: {e}")

print("[5/5] Writing Live State to Streamlit Database...")
# Update Cash
new_cash = total_capital - total_spent

# Update the exact table your app.py is looking for
cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_state (
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        current_regime TEXT,
        regime_color TEXT,
        total_nav REAL,
        daily_pnl REAL,
        active_agent TEXT,
        target_beta REAL
    )
''')

# Insert today's data so the Streamlit Executive Console updates instantly
cursor.execute('''
    INSERT INTO daily_state (current_regime, regime_color, total_nav, daily_pnl, active_agent, target_beta)
    VALUES (?, ?, ?, ?, ?, ?)
''', (regime_name, regime_color, new_cash, daily_pnl, active_agent_name, target_beta))

cursor.execute('''INSERT INTO portfolio_metrics (date, available_cash, total_nav) VALUES (DATE('now'), ?, ?) ON CONFLICT(date) DO UPDATE SET available_cash=excluded.available_cash, total_nav=excluded.total_nav''', (new_cash, new_cash))

conn.commit()
conn.close()
print("✅ Alpha Factory Sequence Complete. Streamlit Dashboard Updated.")