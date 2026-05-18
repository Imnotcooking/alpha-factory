import os
import requests
from dotenv import load_dotenv, find_dotenv
import argparse
import importlib
import pandas as pd
import numpy as np
import sqlite3
import itertools
from evaluator import AlphaEvaluator

def main():
    # 1. Setup the Terminal Command Arguments
    parser = argparse.ArgumentParser(description="Alpha Mine Backtest Engine")
    parser.add_argument("--factor", type=str, required=True, help="Name of the factor file (e.g., fac_001_asym_price)")
    args = parser.parse_args()

    factor_module_name = args.factor
    print(f"🚀 Booting Backtest Engine for: {factor_module_name}...")

    # 2. Dynamically Import the Factor Code
    try:
        # This tells Python to look inside the 'factors' folder for the file
        factor_module = importlib.import_module(f"factors.{factor_module_name}")
    except ModuleNotFoundError:
        print(f"❌ ERROR: Could not find factors/{factor_module_name}.py")
        return

    # 3. Register the Factor Metadata into the Database (if it's new)
    conn = sqlite3.connect("research_memory.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO factors (factor_id, name, category, complexity_score)
        VALUES (?, ?, ?, ?)
    ''', (factor_module.FACTOR_ID, factor_module.FACTOR_NAME, factor_module.CATEGORY, factor_module.COMPLEXITY))
    conn.commit()
    conn.close()

    # 4. Load the Data Lake (Real API Data Integration)
    print("   -> Connecting to Financial Modeling Prep API...")
    
    # Securely load the API key from the parent folder's .env file
    load_dotenv(find_dotenv())
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("❌ CRITICAL ERROR: Could not find FMP_API_KEY in .env file.")
        return

    tickers = ["AAPL", "MSFT", "GOOGL"]
    all_data = []

    for ticker in tickers:
        print(f"      -> Downloading true market state for {ticker}...")
        
        # --- DEFENSIVE API CALLS (Using FMP 'Stable' Endpoints) ---
        
        # A smart parser to handle FMP's new JSON wrapping formats
        def parse_fmp_json(response):
            data = response.json()
            if isinstance(data, dict):
                return data.get('results', data.get('historical', []))
            return data

        # 1. Fetch Daily Prices
        price_url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&apikey={api_key}"
        price_res = requests.get(price_url)
        
        if price_res.status_code != 200:
            print(f"❌ API ERROR: FMP Price Endpoint failed for {ticker}. Status: {price_res.status_code}")
            continue 
            
        prices = parse_fmp_json(price_res)
        df_price = pd.DataFrame(prices)[['date', 'close', 'volume']]
        df_price['date'] = pd.to_datetime(df_price['date'])
        
        # Calculate Forward Return
        df_price = df_price.sort_values('date').reset_index(drop=True)
        df_price['forward_return'] = df_price['close'].pct_change().shift(-1)

        # 2. Fetch Fundamentals (Income, Cash Flow, Balance Sheet)
        inc_url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period=quarter&limit=20&apikey={api_key}"
        cf_url = f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period=quarter&limit=20&apikey={api_key}"
        bs_url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period=quarter&limit=20&apikey={api_key}"
        
        df_inc = pd.DataFrame(parse_fmp_json(requests.get(inc_url)))[['date', 'netIncome']]
        df_cf = pd.DataFrame(parse_fmp_json(requests.get(cf_url)))[['date', 'operatingCashFlow']]
        df_bs = pd.DataFrame(parse_fmp_json(requests.get(bs_url)))[['date', 'totalAssets']]
        
        # Merge the quarterly fundamentals together
        df_funds = df_inc.merge(df_cf, on='date', how='outer').merge(df_bs, on='date', how='outer')
        df_funds['date'] = pd.to_datetime(df_funds['date'])
        
        df_funds = df_funds.rename(columns={
            'netIncome': 'net_income',
            'operatingCashFlow': 'operating_cash_flow',
            'totalAssets': 'total_assets'
        })

        # --- THE ALIGNMENT MATRIX ---
        df_ticker = pd.merge(df_price, df_funds, on='date', how='left')
        df_ticker = df_ticker.ffill().dropna()
        df_ticker['ticker'] = ticker
        
        all_data.append(df_ticker)

    # Combine all individual stock matrices into the final Data Lake
    df = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Data Lake built successfully. Shape: {df.shape}")

    # 5. Apply the Math
    print("   -> Computing Factor Scores...")
    df = factor_module.compute(df)

    # 6. Pass to the Evaluator (The strict judge from Phase 2)
    evaluator = AlphaEvaluator()
    evaluator.run_evaluation(factor_module.FACTOR_ID, df)

if __name__ == "__main__":
    main()