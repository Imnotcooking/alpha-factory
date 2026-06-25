import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_historical_returns(ticker, days_to_hold=30, years_back=10):
    """
    Fetches a decade of data and calculates the rolling returns for your specific hold time.
    """
    print(f"[*] Fetching {years_back} years of historical data for {ticker}...")

    # 1. Fetch Data
    df = yf.download(ticker, period=f"{years_back}y", interval="1d", progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1)
        except: df.columns = df.columns.droplevel(1)

    if df.empty:
        print("❌ Error: Could not fetch data.")
        return np.array([])

    # 2. Convert Calendar Days to Trading Days
    # 30 calendar days is roughly 21 trading days (5/7 ratio)
    trading_days = int(days_to_hold * (252 / 365))
    if trading_days < 1: trading_days = 1

    # 3. Calculate Rolling Returns
    # We use overlapping periods. i.e., What was the return from Day 1 to Day 22? Day 2 to Day 23?
    df['Rolling_Return'] = df['Close'].pct_change(periods=trading_days)

    # Drop NaNs (the first 'trading_days' rows will be NaN)
    valid_returns = df['Rolling_Return'].dropna().values

    print(f"    - Analyzed {len(valid_returns)} historical {days_to_hold}-day periods.")
    return valid_returns

def check_historical_odds(historical_returns, target_pct, direction='up'):
    """
    Queries the distribution: "How often did it move past this target?"
    target_pct: The percentage move required (e.g., 0.05 for 5%)
    """
    if len(historical_returns) == 0: return 0.0

    total_periods = len(historical_returns)

    if direction == 'up':
        # E.g., How many times did it return >= +5%?
        hits = np.sum(historical_returns >= target_pct)
    elif direction == 'down':
        # E.g., How many times did it return <= -5%?
        # Note: target_pct should be negative for downside checks
        hits = np.sum(historical_returns <= target_pct)
    elif direction == 'inside':
        # E.g., How many times did it stay BETWEEN -5% and +5%?
        # target_pct should be passed as a positive absolute bound
        hits = np.sum((historical_returns >= -target_pct) & (historical_returns <= target_pct))
    else:
        return 0.0

    probability = hits / total_periods
    return probability

def generate_distribution_report(ticker, days_to_hold=30):
    """
    Visualizes and reports the historical realities of the stock.
    """
    print(f"\n📊 HISTORICAL DISTRIBUTION ANALYZER: {ticker}")
    print("="*60)

    returns = fetch_historical_returns(ticker, days_to_hold)
    if len(returns) == 0: return

    # Calculate some key statistics
    mean_ret = np.mean(returns)
    win_rate_raw = np.sum(returns > 0) / len(returns)

    # Calculate Percentiles (What is a "normal" move vs an "extreme" move?)
    p10 = np.percentile(returns, 10) # Bottom 10% (Crashing)
    p90 = np.percentile(returns, 90) # Top 10% (Mooning)

    print(f"\n📈 OVERALL BEHAVIOR ({days_to_hold}-Day Window):")
    print(f"    - Average Return:    {mean_ret*100:+.2f}%")
    print(f"    - Historical Drift:  Up {win_rate_raw*100:.1f}% of the time")
    print(f"    - Bottom 10% Move:   Worse than {p10*100:+.2f}%")
    print(f"    - Top 10% Move:      Better than {p90*100:+.2f}%")

    print("\n🎯 SPECIFIC SCENARIO PROBABILITIES:")
    # We test some standard moves: +/- 5%, 10%, 15%
    test_moves = [0.05, 0.10, 0.15]

    print(f"{'MOVE':<15} | {'UPSIDE FREQUENCY':<18} | {'DOWNSIDE FREQUENCY'}")
    print("-" * 60)

    for move in test_moves:
        prob_up = check_historical_odds(returns, move, 'up')
        prob_down = check_historical_odds(returns, -move, 'down')

        move_str = f"+/- {move*100:.0f}%"
        print(f"{move_str:<15} | {prob_up*100:>6.1f}% of the time | {prob_down*100:>6.1f}% of the time")

    print("="*60 + "\n")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 5))

    # Plot empirical histogram
    plt.hist(returns, bins=100, color='royalblue', edgecolor='black', alpha=0.7, density=True, label='Actual Returns')

    # Plot standard normal curve for comparison
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mean_ret) / np.std(returns))**2) / (np.std(returns) * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2, linestyle='--', label='Normal Distribution (Theory)')

    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title(f"{ticker}: Historical {days_to_hold}-Day Return Distribution (10 Years)")
    plt.xlabel(f"{days_to_hold}-Day Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_ticker = input("Enter Ticker to Analyze (e.g. QQQ): ").upper() or "QQQ"
    days = int(input("Target Hold Time (Days): ") or 30)
    generate_distribution_report(test_ticker, days_to_hold=days)