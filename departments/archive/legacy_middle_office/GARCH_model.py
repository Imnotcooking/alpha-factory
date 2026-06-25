import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
import datetime
import warnings

# Suppress harmless warnings from the arch library
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. DATA INGESTION
# ==========================================
def fetch_data(ticker, lookback_years=3):
    """
    Fetches daily data. GARCH needs a good history (3+ years) to
    establish a reliable long-term variance baseline.
    """
    start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    print(f"--- Fetching data for {ticker} ---")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=1)
        except KeyError:
            data.columns = data.columns.droplevel(1)

    return data['Close'].dropna()

# ==========================================
# 2. GARCH ENGINE (The Forecaster)
# ==========================================
def calculate_garch_volatility(ticker):
    """
    Fits a GARCH(1,1) model to forecast tomorrow's volatility.
    """
    # 1. Get Data
    prices = fetch_data(ticker)

    # 2. Calculate Returns (Scaled)
    # GARCH optimizers work best when data is scaled by 100 (e.g. 1.0 instead of 0.01)
    returns = 100 * prices.pct_change().dropna()

    # 3. Define Model: GARCH(1,1)
    # p=1 (Lag of Error), q=1 (Lag of Variance)
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')

    # 4. Fit Model
    # disp='off' suppresses the iteration log
    res = model.fit(disp='off')

    # 5. Forecast
    # We want the variance for the very next time step
    forecast = res.forecast(horizon=1)
    next_day_variance = forecast.variance.iloc[-1, 0]

    # 6. Convert to Annualized Volatility
    # The variance is in "Percent Squared".
    # Square root gives us Daily Volatility in Percent (e.g., 2.5 means 2.5%)
    daily_vol_percent = np.sqrt(next_day_variance)

    # Annualize: Daily Vol * Sqrt(252)
    annual_vol_percent = daily_vol_percent * np.sqrt(252)

    return annual_vol_percent, res

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PROJECT 2A: GARCH VOLATILITY FORECASTER")
    print("="*50)

    ticker = input("Enter Ticker (default NVDA): ").upper() or "NVDA"

    try:
        # Run Engine
        forecast_vol, model_res = calculate_garch_volatility(ticker)

        print("\n" + "-"*40)
        print(f"GARCH REPORT: {ticker}")
        print("-" * 40)
        print(f"Forecasted Annual Volatility: {forecast_vol:.2f}%")
        print("-" * 40)

        print("\n[Interpretation]")
        print("This is the 'Theoretical Baseline' volatility.")
        print("Compare this with Implied Volatility (IV) to find value.")
        print("If IV > GARCH: Options might be Expensive.")
        print("If IV < GARCH: Options might be Cheap.")
        print("="*50)

        # Optional: Plot the Conditional Volatility over time
        # This shows how the "risk" has changed historically according to the model
        print("\n>> Plotting Volatility History...")
        fig = model_res.plot(annualize='D')
        plt.title(f"{ticker} GARCH(1,1) Conditional Volatility")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        print("Ensure 'arch' library is installed: pip install arch")