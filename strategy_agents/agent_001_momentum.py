import polars as pl
import sqlite3
import os

print("Booting Desk PM: Momentum/Breakout Agent...")

def query_available_cash(db_path="portfolio_memory.db"):
    """Ask the Commander (Database) how much cash we are allowed to use."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # In a fresh database, this might be empty, so we default to 100k for the lab
        cursor.execute("SELECT available_cash FROM portfolio_metrics ORDER BY date DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 100000.0
    except Exception as e:
        print(f"⚠️ DB Warning: {e}. Defaulting to $100,000 Paper Capital.")
        return 100000.0

def run_strategy(state_space_path="../offline_quant_lab/data/master_state_space.parquet", max_positions=3):
    """
    The Core Momentum Logic.
    Scans the cross-sectional state space, ranks the highest velocity stocks, 
    and generates an execution target list.
    """
    
    if not os.path.exists(state_space_path):
        raise FileNotFoundError(f"Cannot find {state_space_path}. Has the Data Lake compiled?")
        
    print(f"🌊 Surfer Agent scanning the cross-section...")
    
    # 1. Load the latest market data (Only get the most recent date available)
    df = pl.read_parquet(state_space_path)
    latest_date = df["Date"].max()
    today_market = df.filter(pl.col("Date") == latest_date)
    
    # 2. THE FACTOR FILTER (The Momentum Math)
    # We want stocks that are up over the last month (20D), quarter (60D), 
    # and are physically trading above their 20-day moving average.
    breakout_candidates = today_market.filter(
        (pl.col("Ret_20D") > 0.05) &  # Up at least 5% this month
        (pl.col("Ret_60D") > 0.0) &   # Positive 3-month trend
        (pl.col("Dist_SMA_20") > 0.0) # Price is > 20-day moving average
    )
    
    # 3. THE RANKER
    # Sort the survivors by pure 20-day momentum (highest velocity at the top)
    ranked_candidates = breakout_candidates.sort("Ret_20D", descending=True)
    
    # Take the Top N targets
    target_list = ranked_candidates.head(max_positions).to_pandas()
    
    # 4. CAPITAL ALLOCATION
    available_cash = query_available_cash()
    
    if len(target_list) == 0:
        print("🛑 No momentum breakouts detected today. Recommending 100% Cash.")
        return pd.DataFrame() # Return empty targets
        
    # Equal weight allocation among the winners
    cash_per_stock = available_cash / len(target_list)
    
    # Calculate exactly how many fractional shares we should buy
    target_list['Target_Allocation_$'] = cash_per_stock
    target_list['Target_Shares'] = target_list['Target_Allocation_$'] / target_list['Close']
    target_list['Strategy'] = "Momentum_Breakout"
    
    print(f"🎯 Strategy Selected {len(target_list)} Targets. Total Capital Deployed: ${available_cash:,.2f}")
    
    # Return a clean dataframe containing just the instructions for the Execution Engine
    return target_list[['Ticker', 'Close', 'Target_Shares', 'Target_Allocation_$', 'Strategy']]

if __name__ == "__main__":
    # Test the agent locally
    # Note: Make sure your pathing matches where your parquet file actually is!
    # If running from inside strategy_agents/, the path is ../offline_quant_lab/...
    targets = run_strategy(state_space_path="../offline_quant_lab/data/master_state_space.parquet")
    print("\n--- TODAY'S EXECUTION ORDERS ---")
    print(targets)