import os
import sqlite3
import pandas as pd
import polars as pl
import yfinance as yf
import torch
from stable_baselines3 import PPO
from dotenv import load_dotenv
from ib_insync import IB, Stock, MarketOrder

# Import your Strategy Desk PMs
import strategy_agents.agent_001_momentum as momentum_agent
import strategy_agents.agent_002_mean_reversion as mean_reversion_agent

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
TRADING_MODE = os.getenv("TRADING_MODE", "PAPER")
UNIVERSE = os.getenv("OPERATING_UNIVERSE", "MACRO")

print(f"============== ALPHA FACTORY CRON JOB ==============")
print(f"STATUS: Waking up...")
print(f"MODE: {TRADING_MODE} TRADING")
print(f"UNIVERSE: {UNIVERSE}")
print(f"====================================================")

# 2. CONFIGURE THE CIO BRAIN
if UNIVERSE == "MACRO":
    codebook_path = "models/vq_codebook_institutional_macro.pt"
    policy_path = "models/horizon_policy_macro.zip"
    ticker = "SPY"
else:
    codebook_path = "models/vq_codebook_highbeta_meme.pt"
    policy_path = "models/horizon_policy_meme.zip"
    ticker = "NVDA"

# 3. LIVE STATE GENERATION
print(f"[1/4] Downloading live market data for {ticker}...")
raw_data = yf.download(ticker, period="1y", progress=False)['Close']
df_pd = raw_data.reset_index()
df_pd.columns = ['Date', 'Close']
lf = pl.LazyFrame(df_pd)

lf = lf.with_columns([
    (pl.col("Close") / pl.col("Close").shift(1)).log().alias("Ret_1D"),
    (pl.col("Close") / pl.col("Close").shift(5)).log().alias("Ret_5D"),
    (pl.col("Close") / pl.col("Close").shift(20)).log().alias("Ret_20D"),
    (pl.col("Close") / pl.col("Close").shift(60)).log().alias("Ret_60D"),
    ((pl.col("Close") / pl.col("Close").shift(1)).log().rolling_std(window_size=20) * (252 ** 0.5)).alias("Vol_20D"),
    ((pl.col("Close") / pl.col("Close").shift(1)).log().rolling_std(window_size=60) * (252 ** 0.5)).alias("Vol_60D"),
    ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=20)) - 1.0).alias("Dist_SMA_20"),
    ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=50)) - 1.0).alias("Dist_SMA_50"),
    ((pl.col("Close") / pl.col("Close").rolling_mean(window_size=200)) - 1.0).alias("Dist_SMA_200"),
]).drop_nulls()

latest_state = lf.collect().tail(1)

# 4. CIO INFERENCE
print("[2/4] CIO processing State Space matrix...")
payload = torch.load(codebook_path, map_location=torch.device('cpu'), weights_only=False)
expected_features = payload.get('features', [c for c in latest_state.columns if c not in ['Date', 'Close']])

raw_vector = latest_state.select(expected_features).to_numpy()[0]
scaled_vector = (raw_vector - payload['scaler_mean']) / payload['scaler_scale']

agent = PPO.load(policy_path, device="cpu")
action, _ = agent.predict(scaled_vector, deterministic=True)
action_idx = int(action)
print(f"      -> CIO classified market as Archetype {action_idx}.")

# 5. STRATEGY ROUTING
print("[3/4] Routing to Desk PMs...")
target_orders = pd.DataFrame()

if action_idx in [0, 1, 2]:
    print("      -> Regime: Bear/Panic. Activating Mean-Reversion Desk.")
    target_orders = mean_reversion_agent.run_strategy(state_space_path="offline_quant_lab/data/master_state_space.parquet")
elif action_idx in [5, 6, 7]:
    print("      -> Regime: Bull/Breakout. Activating Momentum Desk.")
    target_orders = momentum_agent.run_strategy(state_space_path="offline_quant_lab/data/master_state_space.parquet")
else:
    print("      -> Regime: Neutral/Choppy. CIO commands 100% Cash.")

# 6. IBKR EXECUTION ENGINE
print("[4/4] Engaging IBKR Execution Protocol...")
if target_orders.empty:
    print("✅ No target orders generated today. System returning to sleep.")
else:
    ib_port = 7497 if TRADING_MODE == "PAPER" else 7496
    
    try:
        ib = IB()
        ib.connect('172.17.0.1', ib_port, clientId=99) # clientId=99 so it doesn't clash with your Streamlit app
        
        conn = sqlite3.connect("portfolio_memory.db")
        cursor = conn.cursor()
        total_spent = 0.0
        
        for index, row in target_orders.iterrows():
            target_ticker = row['Ticker']
            shares = int(row['Target_Shares'])
            strategy = row['Strategy']
            
            if shares <= 0: continue
            
            # Fire the Order
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
            
            # Write to Ledger
            cursor.execute('''INSERT INTO trade_history (ticker, action, quantity, execution_price, strategy) VALUES (?, ?, ?, ?, ?)''', (target_ticker, 'BUY', shares, fill_price, strategy))
            
            cursor.execute('''
                INSERT INTO open_positions (ticker, strategy, quantity, average_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET 
                    quantity=quantity + excluded.quantity,
                    average_price=((average_price * quantity) + (excluded.average_price * excluded.quantity)) / (quantity + excluded.quantity),
                    last_updated=CURRENT_TIMESTAMP
            ''', (target_ticker, strategy, shares, fill_price))
            
        # Update Cash
        cursor.execute("SELECT available_cash FROM portfolio_metrics ORDER BY date DESC LIMIT 1")
        res = cursor.fetchone()
        new_cash = (res[0] if res else 100000.0) - total_spent
        cursor.execute('''INSERT INTO portfolio_metrics (date, available_cash) VALUES (DATE('now'), ?) ON CONFLICT(date) DO UPDATE SET available_cash=excluded.available_cash''', (new_cash,))
        
        conn.commit()
        conn.close()
        ib.disconnect()
        print("✅ All orders executed and logged to SQLite Memory Bank. Goodnight.")
        
    except Exception as e:
        print(f"❌ CRITICAL EXECUTION FAILURE: {e}")