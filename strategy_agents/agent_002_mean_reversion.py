import polars as pl
import sqlite3
import os

print("Booting Desk PM: Mean-Reversion Agent...")

def query_available_cash(db_path="portfolio_memory.db"):
    """Ask the Commander (Database) how much cash we are allowed to use."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT available_cash FROM portfolio_metrics ORDER BY date DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 100000.0
    except Exception as e:
        print(f"⚠️ DB Warning: {e}. Defaulting to $100,000 Paper Capital.")
        return 100000.0

def run_strategy(state_space_path="../offline_quant_lab/data/master_state_space.parquet", max_positions=3):
    """
    The Core Mean-Reversion Logic.
    Scans for stocks that have deviated significantly below their 20-day 
    and 50-day moving averages, betting on a short-term snap-back rally.
    """
    
    if not os.path.exists(state_space_path):
        raise FileNotFoundError(f"Cannot find {state_space_path}.")
        
    print(f"📉 Rubber Band Agent scanning for oversold deviations...")
    
    # 1. Load the latest market data
    df = pl.read_parquet(state_space_path)
    latest_date = df["Date"].max()
    today_market = df.filter(pl.col("Date") == latest_date)
    
    # 2. THE FACTOR FILTER (The Mean-Reversion Math)
    # We want stocks that are:
    # A) Trading at least 5% BELOW their 20-day moving average
    # B) Trading BELOW their 50-day moving average
    # C) Have high recent volatility (panic selling creates the best snap-backs)
    bargain_candidates = today_market.filter(
        (pl.col("Dist_SMA_20") < -0.05) & 
        (pl.col("Dist_SMA_50") < 0.0) &
        (pl.col("Vol_20D") > 0.25) # Requires annualized vol > 25% to ensure it's a volatile drop, not a slow bleed
    )
    
    # 3. THE RANKER
    # Sort the survivors by Dist_SMA_20 ascending (the MOST negative number goes to the top)
    # e.g., -0.15 is better than -0.06
    ranked_candidates = bargain_candidates.sort("Dist_SMA_20", descending=False)
    
    target_list = ranked_candidates.head(max_positions).to_pandas()
    
    # 4. CAPITAL ALLOCATION
    available_cash = query_available_cash()
    
    if len(target_list) == 0:
        print("🛑 No extreme oversold deviations detected today. Recommending 100% Cash.")
        return pd.DataFrame()
        
    cash_per_stock = available_cash / len(target_list)
    
    target_list['Target_Allocation_$'] = cash_per_stock
    target_list['Target_Shares'] = target_list['Target_Allocation_$'] / target_list['Close']
    target_list['Strategy'] = "Mean_Reversion"
    
    print(f"🎯 Strategy Selected {len(target_list)} Targets. Total Capital Deployed: ${available_cash:,.2f}")
    
    return target_list[['Ticker', 'Close', 'Target_Shares', 'Target_Allocation_$', 'Strategy']]

if __name__ == "__main__":
    targets = run_strategy(state_space_path="../offline_quant_lab/data/master_state_space.parquet")
    print("\n--- TODAY'S EXECUTION ORDERS ---")
    print(targets)