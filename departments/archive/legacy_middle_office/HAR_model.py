import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import warnings

# Suppress harmless warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. DATA INGESTION
# ==========================================
def fetch_data(ticker, lookback_years=3):
    """
    Fetches daily OHLC data. HAR needs High/Low for the Parkinson Estimator.
    """
    start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    print(f"--- Fetching data for {ticker} ---")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Handle MultiIndex (yfinance v0.2+ fix)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=1)
        except KeyError:
            data.columns = data.columns.droplevel(1)

    return data.dropna()

# ==========================================
# 2. HAR-RV ENGINE (The Empirical Forecaster)
# ==========================================
def calculate_har_volatility(ticker):
    """
    Calculates the HAR-RV forecast using the Parkinson Estimator.
    Returns: Annualized Volatility (float), Model Results (for plotting)
    """
    # 1. Get Data
    df = fetch_data(ticker)

    # 2. Feature Engineering (Parkinson Volatility)
    # Formula: 1 / (4 * ln(2)) * (ln(High/Low))^2
    const = 1 / (4 * np.log(2))
    df['Log_HL'] = np.log(df['High'] / df['Low'])
    df['RV_daily'] = const * (df['Log_HL'] ** 2)

    # 3. Create Lags (The "Heterogeneous" components)
    # Daily (1d), Weekly (5d), Monthly (22d)
    df['RV_weekly'] = df['RV_daily'].rolling(window=5).mean()
    df['RV_monthly'] = df['RV_daily'].rolling(window=22).mean()

    # 4. Define Target (Tomorrow's Volatility)
    df['Target'] = df['RV_daily'].shift(-1)

    # Clean data for regression
    model_data = df.dropna()

    # 5. Train Model (OLS Regression)
    X = model_data[['RV_daily', 'RV_weekly', 'RV_monthly']]
    X = sm.add_constant(X) # Add Intercept
    Y = model_data['Target']

    model = sm.OLS(Y, X).fit()

    # 6. Forecast Tomorrow
    # We take the *very last row* of known data to predict the unknown tomorrow
    latest_features = df.iloc[[-1]][['RV_daily', 'RV_weekly', 'RV_monthly']]
    latest_features = sm.add_constant(latest_features, has_constant='add')

    pred_variance = model.predict(latest_features).values[0]

    # 7. Annualize
    # Sqrt(Variance) = Daily Vol -> * Sqrt(252) = Annual Vol
    pred_daily_vol = np.sqrt(pred_variance)
    pred_annual_vol = pred_daily_vol * np.sqrt(252)

    # Return forecast and the dataframe (for plotting if needed)
    return pred_annual_vol, model, df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PROJECT 2B: HAR-RV VOLATILITY FORECASTER")
    print("="*50)

    ticker = input("Enter Ticker (default TSLA): ").upper() or "TSLA"

    try:
        # Run Engine
        forecast_vol, model, data = calculate_har_volatility(ticker)

        # Extract Coefficients to understand the market regime
        beta_d = model.params['RV_daily']
        beta_w = model.params['RV_weekly']
        beta_m = model.params['RV_monthly']

        print("\n" + "-"*40)
        print(f"HAR-RV REPORT: {ticker}")
        print("-" * 40)
        print(f"Forecasted Annual Volatility: {forecast_vol*100:.2f}%")
        print("-" * 40)

        print(f"\n[Market Regime Analysis]")
        print(f"Daily Impact (Panic): {beta_d:.2f}")
        print(f"Weekly Impact (Trend): {beta_w:.2f}")
        print(f"Monthly Impact (Structure): {beta_m:.2f}")

        if beta_d > beta_m:
            print(">> REGIME: Jittery. Short-term panic is driving volatility.")
        elif beta_m > 0.3:
            print(">> REGIME: Sticky. Long-term volatility is elevated and persisting.")
        else:
            print(">> REGIME: Mean Reverting / Normal.")

        print("="*50)

        # Plotting (Actual vs Forecast on training data)
        # We need to predict on the whole set to plot the line
        print("\n>> Plotting HAR Fit...")
        X_all = sm.add_constant(data[['RV_daily', 'RV_weekly', 'RV_monthly']].dropna())
        predictions = model.predict(X_all)

        # Annualize for plot
        actual_vol = np.sqrt(data['RV_daily']) * np.sqrt(252)
        pred_vol = np.sqrt(predictions) * np.sqrt(252)

        plt.figure(figsize=(12, 6))
        plt.plot(actual_vol.index[-252:], actual_vol.tail(252), color='gray', alpha=0.4, label='Actual Vol (Parkinson)')
        plt.plot(pred_vol.index[-252:], pred_vol.tail(252), color='blue', linewidth=1.5, label='HAR Forecast')
        plt.title(f"{ticker} HAR-RV Model (1 Year View)")
        plt.ylabel("Annualized Volatility")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")