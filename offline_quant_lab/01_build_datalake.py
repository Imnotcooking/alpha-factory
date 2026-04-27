import os
import requests
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

print("Starting Institutional Data Ingestion (FMP Premium Stable Engine)...")

# 1. LOAD API KEY
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY: 
    raise ValueError("❌ FMP_API_KEY is missing in your local .env file!")

DATA_DIR = "offline_quant_lab/data"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. DEFINE UNIVERSES
# Top 50 S&P 500 components for ultra-liquid Institutional Macro testing
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CRM", "CVX", "NFLX", "AMD", "PEP", "KO", "ADBE", "WMT", "TMO",
    "ACN", "MCD", "DIS", "CSCO", "ABT", "INTC", "QCOM", "INTU", "VZ", "CMCSA",
    "IBM", "PFE", "NOW", "AMAT", "TXN", "GE", "ISRG", "BA", "SPY", "QQQ"
]

# Note: FMP usually expects crypto without hyphens (e.g., BTCUSD)
MEME_CRYPTO_TICKERS = ["BTCUSD", "ETHUSD", "DOGEUSD", "CIFR", "HIMS", "RKLB", "OKLO", "MSTR", "CRCL"]

def fetch_single_ticker(ticker: str) -> pl.DataFrame:
    """Pulls historical data using FMP's Premium Stable endpoint."""
    # The new August 2025+ FMP Stable Endpoint
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&apikey={FMP_API_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        
        # If FMP blocks us, print the exact reason
        if response.status_code != 200:
            print(f"⚠️ FMP Error {response.status_code} for {ticker}: {response.text}")
            return None
            
        data = response.json()
        
        # FMP Stable API returns a direct list of dicts OR a dict containing 'historical'
        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'historical' in data:
            records = data['historical']
        elif isinstance(data, dict) and "Error Message" in data:
            print(f"⚠️ FMP Rejected {ticker}: {data['Error Message']}")
            return None
            
        if len(records) > 0:
            df = pl.DataFrame(records)
            
            # Select strictly capitalized columns to prevent Agent crashes
            df = df.select([
                pl.col("date").alias("Date"),
                pl.col("close").alias("Close"),
                pl.col("volume").alias("Volume")
            ])
            # Add Ticker column
            return df.with_columns(pl.lit(ticker).alias("Ticker"))
            
    except Exception as e:
        print(f"⚠️ Failed to fetch {ticker}: {e}")
        
    return None

def build_lake(tickers: list, filename: str):
    print(f"🌊 Downloading {len(tickers)} assets for {filename}...")
    all_data = []
    
    # Multi-threading to hit FMP Premium rate limits safely
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            df = future.result()
            if df is not None:
                all_data.append(df)
                
    if all_data:
        master_df = pl.concat(all_data)
        master_df = master_df.sort(["Ticker", "Date"])
        file_path = os.path.join(DATA_DIR, filename)
        master_df.write_parquet(file_path)
        print(f"✅ Success! Saved {filename}. Total rows: {master_df.height:,}")
    else:
        print(f"❌ Critical Failure: No data downloaded for {filename}")

if __name__ == "__main__":
    build_lake(SP500_TICKERS, "institutional_macro.parquet")
    build_lake(MEME_CRYPTO_TICKERS, "crypto_meme.parquet")