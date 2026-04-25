import yfinance as yf
import polars as pl
import os

print("Starting Institutional Data Ingestion...")

# 1. Define the Universe
# We start with highly liquid macro proxies and mega-caps to build our State Space.
# Once you verify this works, you can expand this list to 500 stocks.
universe = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", # Macro ETFs
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", # Mega-cap Tech
    "JPM", "XOM", "UNH", "JNJ", "V" # Diversified Sectors
]

# 2. Fetch Data
print(f"Downloading 20 years of data for {len(universe)} assets...")
# We use auto_adjust=True to adjust for stock splits and dividends automatically
raw_data = yf.download(tickers=universe, period="20y", interval="1d", auto_adjust=True, progress=True)

# 3. Clean and Transform (Pandas -> Polars)
print("Transforming matrix to Polars columnar format...")

# yfinance returns a multi-index dataframe if multiple tickers are requested. 
# We isolate the 'Close' and 'Volume' prices, stack them, and reset the index.
closes = raw_data['Close'].stack().reset_index()
closes.columns = ['Date', 'Ticker', 'Close']

volumes = raw_data['Volume'].stack().reset_index()
volumes.columns = ['Date', 'Ticker', 'Volume']

# Merge them together using Pandas before converting to Polars
df_merged = closes.merge(volumes, on=['Date', 'Ticker'])

# Convert to Polars DataFrame for high-performance saving
df_pl = pl.from_pandas(df_merged)

# Sort the data chronologically and by ticker
df_pl = df_pl.sort(["Ticker", "Date"])

# 4. Save to Parquet
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "master_price_history.parquet")

print(f"Compressing and saving to {file_path}...")
df_pl.write_parquet(file_path)

print(f"✅ Success! Data Lake built. Total rows: {df_pl.height:,}")