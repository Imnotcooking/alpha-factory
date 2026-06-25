import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
import sys
from datetime import date
from pathlib import Path
from scipy.stats import norm
import plotly.graph_objects as go
import pandas_ta as ta

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from oqp.config import load_settings
except Exception:
    load_settings = None

try:
    from oqp.data import FMPDataAdapter
except Exception:
    FMPDataAdapter = None

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. PAGE CONFIGURATION (Must be FIRST!) ---
st.set_page_config(page_title="Options Desk & Volatility", layout="wide", page_icon="⚡")

# --- 2. LOAD CUSTOM CSS ---
from utils.theme import load_css
try:
    load_css()
except:
    pass

# --- LOCALIZATION ENGINE (EN/ZH) ---
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)
is_zh = lang == "中文"

t = {
    "title": "⚡ The Greeks: Options & Volatility Desk" if not is_zh else "⚡ 希腊字母：期权与波动率交易台",
    "sub": "Automated Option Scanning, Volatility Analysis, and Alpha Generation." if not is_zh else "自动化期权扫描、波动率分析与阿尔法生成。",
    "sidebar": "🎯 Scanner Parameters" if not is_zh else "🎯 扫描器参数",
    "ticker": "Target Ticker" if not is_zh else "目标代码",
    "budget": "Max Budget / Margin ($)" if not is_zh else "最大预算/保证金 ($)",
    "hold": "Target Hold Time (Days)" if not is_zh else "目标持有天数",
    "run": "🚀 Run Master Scan" if not is_zh else "🚀 运行主扫描",
}

st.title(t["title"])
st.markdown(t["sub"])
st.markdown("---")

# --- 3. SIDEBAR CONTROLS (Pure Scanner) ---
with st.sidebar:
    st.header(t["sidebar"])
    target_ticker = st.text_input(t["ticker"], value="QQQ").upper()
    max_budget = st.number_input(t["budget"], min_value=100.0, value=1000.0, step=100.0)
    target_hold_days = st.number_input(t["hold"], min_value=1, max_value=365, value=30, step=1)

    # --- MANUAL STRATEGY OVERRIDE ---
    st.markdown("---")
    st.subheader("🛠️ Deep Value & Overrides" if not is_zh else "🛠️ 深度价值与覆盖")

    strategy_map = {
        "Deep Value LEAPS (FMP)": "scan_deep_value_leaps",
        "Long Call": "scan_long_calls",
        "Long Put": "scan_long_puts",
        "Bull Call Spread": "scan_bull_call_spreads",
        "Bear Put Spread": "scan_bear_put_spreads",
        "Calendar Spread": "scan_calendar_spreads",
        "Call Backspread": "scan_call_backspreads",
        "Put Backspread": "scan_put_backspreads",
        "Call Butterfly": "scan_call_butterflies",
        "Call Ratio Spread": "scan_call_ratio_spreads",
        "Put Ratio Spread": "scan_put_ratio_spreads",
        "Collar": "scan_collars",
        "Iron Condor": "scan_iron_condors",
        "Long Straddle/Strangle": "scan_long_volatility",
        "Short Straddle/Strangle": "scan_short_volatility",
        "Cash-Secured Put": "scan_short_puts"
    }

    manual_strategies = st.multiselect(
        "Force Monte Carlo Simulation:" if not is_zh else "强制蒙特卡洛模拟:",
        options=list(strategy_map.keys()),
        default=[]
    )

    st.markdown("---")
    run_action = st.button(t["run"], use_container_width=True, type="primary")

    if run_action:
        st.session_state['options_scan_executed'] = True
        st.session_state['manual_strategies'] = [strategy_map[s] for s in manual_strategies]

# --- 4. THE UNIFIED OPTION SCANNER CLASS & ENGINES ---
try:
    from GARCH_model import calculate_garch_volatility
    from HAR_model import calculate_har_volatility
    from Historical_Distribution import check_historical_odds, fetch_historical_returns
except ImportError as e:
    st.warning(f"Note: Missing Local Quant Modules ({e}). Ensure models are in your directory." if not is_zh else f"注意: 缺少本地量化模块 ({e})。请确保模型在您的目录中。")

def load_fmp_api_key():
    settings_fmp_key = load_settings().fmp_api_key if load_settings else None
    return (
        st.secrets.get("FMP_API_KEY", None)
        or st.secrets.get("FMP_KEY", None)
        or settings_fmp_key
    )

def make_fmp_adapter(api_key):
    if not api_key or FMPDataAdapter is None:
        return None
    return FMPDataAdapter(api_key=api_key)

def fetch_leaps_fundamental_inputs(api_key, ticker, limit=5):
    adapter = make_fmp_adapter(api_key)
    if adapter is None:
        return [], [], []

    return (
        adapter.get_income_statement(ticker, limit=limit),
        adapter.get_cash_flow_statement(ticker, limit=limit),
        adapter.get_key_metrics(ticker, limit=limit),
    )

@st.cache_data(ttl=300)
def fetch_scanner_data(ticker):
    stock = yf.Ticker(ticker)
    # --- 1. ROBUST SPOT PRICE INGESTION ---
    hist_buffer = stock.history(period='5d') # Use 5 days to bridge long weekends/holidays

    if hist_buffer.empty:
        st.error(f"⚠️ **DATA FATAL ERROR:** Unable to fetch market data for ticker '{ticker}'. Ensure the symbol is correct and Yahoo Finance is online.")
        st.stop() # Halts execution cleanly so the rest of the UI doesn't crash

    try:
        # Try to get the live quote first
        S = float(stock.info.get('regularMarketPrice', stock.info.get('currentPrice')))
        if pd.isna(S) or S <= 0:
            raise ValueError
    except:
        # Fallback to the last available close in our 5-day buffer
        S = float(hist_buffer['Close'].iloc[-1])

    next_earnings = None
    try:
        calendar = stock.calendar
        if 'Earnings Date' in calendar:
            dates = calendar['Earnings Date']
            future_dates = [d for d in dates if d.date() >= date.today()]
            if future_dates:
                next_earnings = future_dates[0].date()
    except: pass

    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1)
        except: df.columns = df.columns.droplevel(1)

    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_60'] = ta.sma(df['Close'], length=60)
    df['SMA_120'] = ta.sma(df['Close'], length=120)
    df['SMA_200'] = ta.sma(df['Close'], length=200) # Added 200
    df['EMA_9'] = ta.ema(df['Close'], length=9)    # Changed to 9
    df['EMA_21'] = ta.ema(df['Close'], length=21)  # Changed to 21
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    daily_pct_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    trend = "Choppy"
    if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] > df['SMA_60'].iloc[-1]: trend = "Bullish"
    elif df['Close'].iloc[-1] < df['SMA_20'].iloc[-1] < df['SMA_60'].iloc[-1]: trend = "Bearish"
    elif df['Close'].iloc[-1] < df['SMA_20'].iloc[-1] and df['SMA_60'].iloc[-1] > df['SMA_120'].iloc[-1]: trend = "Pullback"

    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    hist_vol = df['Log_Ret'].tail(21).std() * np.sqrt(252)

    # Fallback dummies if modules fail
    try: garch_vol, _ = calculate_garch_volatility(ticker)
    except: garch_vol = hist_vol * 1.1 * 100
    try: har_vol, _, _ = calculate_har_volatility(ticker)
    except: har_vol = hist_vol * 1.05 * 100

    return S, daily_pct_change, stock.options, trend, df, hist_vol, garch_vol/100.0, har_vol/100.0, next_earnings

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def run_monte_carlo_calendar(S, K, T_near, T_far, r, garch_vol, spread_cost, simulations=5000):
    """Simulates Calendar Spread PoP and EV."""
    dt = T_near
    Z = np.random.normal(0, 1, simulations)

    # Price of Stock at Near Expiration
    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)
    T_remaining = T_far - T_near

    profits = []
    for price in S_future:
        val_short = max(price - K, 0) # Short Near Call Value
        val_long = black_scholes_price(price, K, T_remaining, r, garch_vol, 'call') # Long Far Call Value
        profits.append((val_long - val_short) - spread_cost)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def analyze_calendar_candidate(S, K, T_near, T_far, r, garch_vol, spread_cost):
    """Calculates Theoretical Edge and Max Profit."""
    fair_near = black_scholes_price(S, K, T_near, r, garch_vol, 'call')
    fair_far = black_scholes_price(S, K, T_far, r, garch_vol, 'call')
    theoretical_value = fair_far - fair_near

    edge = theoretical_value - spread_cost
    val_far_at_strike = black_scholes_price(K, K, T_far - T_near, r, garch_vol, 'call')
    max_potential_profit = val_far_at_strike - spread_cost

    return theoretical_value, edge, max_potential_profit

def solve_implied_volatility_safe(market_price, S, K, T, r, option_type='call'):
    """A crash-proof bisection method to find Implied Volatility."""
    if market_price <= 0 or T <= 0: return 0.001

    low_vol, high_vol = 0.001, 3.0 # Search between 0.1% and 300% vol
    for _ in range(50): # 50 iterations is plenty for precision
        mid_vol = (low_vol + high_vol) / 2
        price_estimate = black_scholes_price(S, K, T, r, mid_vol, option_type)
        if price_estimate > market_price:
            high_vol = mid_vol
        else:
            low_vol = mid_vol

    return mid_vol

def run_monte_carlo_call(S, K, T_days, r, sigma, cost_basis, simulations=5000):
    """Simulates Call Option PoP and EV."""
    dt = T_days / 365.0
    Z = np.random.normal(0, 1, simulations)
    S_future = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    payoffs = np.maximum(S_future - K, 0)
    profits = payoffs - cost_basis

    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    return pop, ev

def run_monte_carlo_backspread(S, K_short, K_long, ratio, days_to_hold, r, garch_vol, entry_net, is_credit, simulations=5000):
    """Simulates Call Backspread (Sell 1 Lower, Buy Ratio Higher)"""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)
    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_short = max(price - K_short, 0)
        val_long = max(price - K_long, 0)

        pos_value = (val_long * ratio) - val_short

        if is_credit:
            pnl = entry_net + pos_value
        else:
            pnl = pos_value - entry_net

        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    return pop, ev

def analyze_backspread_structure(K_short, K_long, entry_net, is_credit):
    """Calculates the max loss (Valley of Death) at the exact pinned strike."""
    intrinsic_loss_at_pin = -(K_long - K_short)
    if is_credit:
        max_loss = intrinsic_loss_at_pin + entry_net
    else:
        max_loss = intrinsic_loss_at_pin - entry_net
    return max_loss

def run_monte_carlo_butterfly(S, K1, K2, K3, days_to_hold, r, garch_vol, debit, simulations=5000):
    """Simulates Long Call Butterfly (Buy K1, Sell 2x K2, Buy K3)"""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    # Project Stock Price
    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_K1 = max(price - K1, 0)
        val_K2 = max(price - K2, 0)
        val_K3 = max(price - K3, 0)

        position_value = val_K1 - (2 * val_K2) + val_K3
        pnl = position_value - debit
        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    return pop, ev

def run_monte_carlo_ratio(S, K_long, K_short, ratio, days_to_hold, r, garch_vol, net_val, is_debit, simulations=5000):
    """Simulates Call Ratio Spread (Buy 1 Lower, Sell Ratio Higher). Risk is to the UPSIDE."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_long = max(price - K_long, 0)
        val_short = max(price - K_short, 0)

        position_value = val_long - (val_short * ratio)

        if is_debit:
            pnl = position_value - net_val
        else:
            pnl = net_val + position_value

        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    var_95 = np.percentile(profits, 5) # 95% worst-case scenario (Tail Risk)

    return pop, ev, var_95

def analyze_ratio_structure(K_long, K_short, ratio, net_val, is_debit):
    """Calculates Max Profit and the Upside Breakeven where losses begin."""
    max_profit_intrinsic = (K_short - K_long)

    if is_debit:
        max_profit = max_profit_intrinsic - net_val
    else:
        max_profit = max_profit_intrinsic + net_val

    # Upside Breakeven: Where the naked short(s) eat the max profit
    be_up = K_short + (max_profit / (ratio - 1))
    return max_profit, be_up

def run_monte_carlo_collar(S, K_put, K_call, days_to_hold, r, garch_vol, wrapper_cost, simulations=5000):
    """Simulates a Buy-Write Collar (Stock + Put - Call)."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        stock_pnl = price - S
        put_val = max(K_put - price, 0)
        call_val = max(price - K_call, 0)

        # PnL = Stock change + Put protection - Call liability - Wrapper cost
        total_pnl = stock_pnl + put_val - call_val - wrapper_cost
        profits.append(total_pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    worst_case = np.min(profits)

    return pop, ev, worst_case

def analyze_collar_structure(S, K_put, K_call, wrapper_cost):
    """Analyzes the Floor (Max Loss) and Cap (Max Profit)."""
    # Max loss occurs if stock falls below the Put strike
    max_loss = (K_put - S) - wrapper_cost

    # Max profit occurs if stock rises above the Call strike
    max_profit = (K_call - S) - wrapper_cost

    return max_loss, max_profit

def run_monte_carlo_condor(S, K_put_long, K_put_short, K_call_short, K_call_long, days_to_hold, r, garch_vol, credit, simulations=5000):
    """Simulates an Iron Condor (Short Put Spread + Short Call Spread)."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        # Put Side Liability
        val_put_spread = max(K_put_short - price, 0) - max(K_put_long - price, 0)
        # Call Side Liability
        val_call_spread = max(price - K_call_short, 0) - max(price - K_call_long, 0)

        total_liability = val_put_spread + val_call_spread
        pnl = credit - total_liability
        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def analyze_condor_structure(K_put_long, K_put_short, K_call_short, K_call_long, credit):
    """Analyzes Risk/Reward."""
    width_put = K_put_short - K_put_long
    width_call = K_call_long - K_call_short
    max_width = max(width_put, width_call)

    max_loss = max_width - credit
    return max_loss

def run_monte_carlo_put(S, K, T_days, r, sigma, entry_price, is_short=False, simulations=5000):
    """Simulates outcomes for both Long and Short Puts."""
    dt = T_days / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    intrinsic_values = np.maximum(K - S_future, 0)

    if is_short:
        # Short Put: Profit = Premium Received - Final Intrinsic Liability
        profits = entry_price - intrinsic_values
    else:
        # Long Put: Profit = Final Intrinsic Value - Premium Paid
        profits = intrinsic_values - entry_price

    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def run_monte_carlo_put_backspread(S, K_short, K_long, ratio, days_to_hold, r, garch_vol, entry_net, is_credit, simulations=5000):
    """Simulates a Put Backspread (Sell 1 Higher, Buy Ratio Lower)."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_short = max(K_short - price, 0)
        val_long = max(K_long - price, 0)

        pos_value = (val_long * ratio) - val_short

        if is_credit:
            pnl = entry_net + pos_value
        else:
            pnl = pos_value - entry_net

        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def analyze_put_backspread_structure(K_short, K_long, entry_net, is_credit):
    """Calculates Max Loss (The 'Valley of Death' at K_long)."""
    intrinsic_loss_at_pin = -(K_short - K_long)

    if is_credit:
        max_loss = intrinsic_loss_at_pin + entry_net
    else:
        max_loss = intrinsic_loss_at_pin - entry_net

    return max_loss

def run_monte_carlo_ratio_put(S, K_long, K_short, ratio, days_to_hold, r, garch_vol, net_val, is_debit, simulations=5000):
    """Simulates a Put Ratio Spread (Buy 1 Higher, Sell Ratio Lower). Risk is DOWNSIDE."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_long = max(K_long - price, 0)
        val_short = max(K_short - price, 0)

        position_value = val_long - (val_short * ratio)

        if is_debit:
            pnl = position_value - net_val
        else:
            pnl = net_val + position_value

        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)
    var_95 = np.percentile(profits, 5) # 95% worst-case (Crash Risk)

    return pop, ev, var_95

def analyze_ratio_put_structure(K_long, K_short, ratio, net_val, is_debit):
    """Calculates Max Profit and the Downside Breakeven where losses begin."""
    max_profit_intrinsic = (K_long - K_short)

    if is_debit:
        max_profit = max_profit_intrinsic - net_val
    else:
        max_profit = max_profit_intrinsic + net_val

    # Downside Breakeven: Where the naked short(s) eat the max profit
    be_down = K_short - (max_profit / (ratio - 1))
    return max_profit, be_down

def run_monte_carlo_vol(S, K_call, K_put, days_to_hold, r, garch_vol, entry_price, is_short=False, simulations=5000):
    """Simulates PoP and EV for Straddles and Strangles."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        val_call = max(price - K_call, 0)
        val_put = max(K_put - price, 0)
        total_val = val_call + val_put

        if is_short:
            pnl = entry_price - total_val
        else:
            pnl = total_val - entry_price

        profits.append(pnl)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def analyze_vol_candidate(S, days_to_hold, garch_vol, total_cost, is_short):
    """Calculates the dollar Edge based on GARCH Expected Move vs Market Price."""
    expected_move_dollar = S * garch_vol * np.sqrt(days_to_hold/365.0) * 0.8
    market_breakeven = total_cost

    if is_short:
        edge = market_breakeven - expected_move_dollar
    else:
        edge = expected_move_dollar - total_cost

    return edge

def run_monte_carlo_vertical(S, K_long, K_short, days_to_hold, r, garch_vol, spread_cost, spread_type, simulations=5000):
    """Simulates outcomes for Bull Call and Bear Put Spreads."""
    dt = days_to_hold / 365.0
    Z = np.random.normal(0, 1, simulations)

    S_future = S * np.exp((r - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * Z)

    profits = []
    for price in S_future:
        if spread_type == 'bull_call':
            val_long = max(price - K_long, 0)
            val_short = max(price - K_short, 0)
        else:
            val_long = max(K_long - price, 0)
            val_short = max(K_short - price, 0)

        spread_value = val_long - val_short
        profits.append(spread_value - spread_cost)

    profits = np.array(profits)
    pop = np.sum(profits > 0) / simulations
    ev = np.mean(profits)

    return pop, ev

def analyze_vertical_candidate(S, K_long, K_short, T, days_to_hold, r, garch_vol, cost, spread_type):
    """Estimates the future value of the spread using Black-Scholes at the projected 0.7 sigma target."""
    time_factor = np.sqrt(days_to_hold / 365.0)
    move_pct = garch_vol * time_factor * 0.7

    if spread_type == 'bull_call':
        S_target = S * np.exp(move_pct)
        opt_type = 'call'
    else:
        S_target = S * np.exp(-move_pct)
        opt_type = 'put'

    T_future = max(T - (days_to_hold / 365.0), 0)

    val_long_future = black_scholes_price(S_target, K_long, T_future, r, garch_vol, opt_type)
    val_short_future = black_scholes_price(S_target, K_short, T_future, r, garch_vol, opt_type)

    spread_value_future = val_long_future - val_short_future
    proj_profit = spread_value_future - cost

    # Cap projected profit at the intrinsic max width
    max_intrinsic = abs(K_long - K_short)
    if spread_value_future > max_intrinsic:
        proj_profit = max_intrinsic - cost

    proj_roi = (proj_profit / cost) * 100 if cost > 0 else 0
    return proj_roi, max_intrinsic

class StreamlitOptionScanner:
    def __init__(self, ticker, budget, hold_days):
        self.ticker = ticker
        self.budget = budget
        self.hold_days = hold_days
        self.r = 0.045
        self.today = date.today()

        self.S, self.daily_change, self.expirations, self.trend, self.tech_df, self.hist_vol, self.garch_vol, self.har_vol, self.next_earnings = fetch_scanner_data(ticker)
        self.stock = yf.Ticker(ticker)
        self.forecasted_vol = self.garch_vol

        self.rsi = self.tech_df['RSI'].iloc[-1]
        self.atr = self.tech_df['ATR'].iloc[-1]

        try:
            chain = self.stock.option_chain(self.expirations[0]).calls
            chain['dist'] = abs(chain['strike'] - self.S)
            self.market_iv = chain.sort_values('dist').iloc[0].get('impliedVolatility', 0.30)
        except:
            self.market_iv = 0.30

    def _get_filtered_price(self, row):
        bid, ask, last = row.get('bid', 0), row.get('ask', 0), row.get('lastPrice', 0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            if mid > 0 and (ask - bid) / mid <= 0.25: return mid
        elif last > 0: return last
        return 0.0

    def scan_calendar_spreads(self, budget=500):
        """Module 1: Calendar Spread Ranker"""
        print(f"\n[*] Scanning Calendar Spreads (Target: Neutral/Pinning)...")

        # 1. Find the Expiration Gap
        near_date, far_date = None, None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if 25 <= d <= 50 and not near_date: near_date = e
            elif near_date:
                near_d = (pd.to_datetime(near_date).date() - self.today).days
                if d >= (near_d + 30) and not far_date: far_date = e

        if not near_date or not far_date:
            print("   ❌ Could not find suitable Near/Far expiration gap.")
            return pd.DataFrame()

        print(f"    - Near Leg: {near_date} | Far Leg: {far_date}")

        # 2. Fetch Data
        try:
            calls_near = self.stock.option_chain(near_date).calls
            calls_far = self.stock.option_chain(far_date).calls
        except:
            return pd.DataFrame()

        common_strikes = sorted(list(set(calls_near['strike']).intersection(set(calls_far['strike']))))
        results = []

        # 3. Process Strikes
        for K in common_strikes:
            if not (self.S * 0.85 < K < self.S * 1.15): continue # Only scan near ATM

            row_near = calls_near[calls_near['strike'] == K].iloc[0]
            row_far = calls_far[calls_far['strike'] == K].iloc[0]

            p_near = (row_near['bid'] + row_near['ask']) / 2 if (row_near['bid'] > 0 and row_near['ask'] > 0) else row_near['lastPrice']
            p_far = (row_far['bid'] + row_far['ask']) / 2 if (row_far['bid'] > 0 and row_far['ask'] > 0) else row_far['lastPrice']

            if p_near <= 0 or p_far <= 0: continue

            spread_cost = p_far - p_near
            if spread_cost <= 0 or (spread_cost * 100) > budget: continue

            # 4. Math & Analysis
            T_near = (pd.to_datetime(near_date).date() - self.today).days / 365.0
            T_far = (pd.to_datetime(far_date).date() - self.today).days / 365.0

            theo_val, edge, max_prof = analyze_calendar_candidate(self.S, K, T_near, T_far, self.r, self.forecasted_vol, spread_cost)
            pop, ev = run_monte_carlo_calendar(self.S, K, T_near, T_far, self.r, self.forecasted_vol, spread_cost)

            results.append({
                "Strategy": "Calendar Spread",
                "Expiry": f"{near_date} / {far_date}",
                "Strike": f"${K}",
                "Debit/Credit": f"-${spread_cost*100:.0f}",
                "Max_Profit": f"${max_prof*100:.0f}",
                "Edge": f"${edge*100:.0f}",
                "PoP": f"{pop*100:.1f}%",
                "Raw_Score": pop * ev # Used for sorting
            })



        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_long_calls(self, budget=500, days_to_hold=20):
        """Module 2: Long Call Ranker (Bullish/Momentum Target)"""
        print(f"\n[*] Scanning Long Calls (Target: Bullish Breakout)...")

        # 1. Find the right expiration (> days_to_hold)
        target_date = None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if d > days_to_hold:
                target_date = e
                days_to_exp = d
                break

        if not target_date:
            print("   ❌ No valid expirations found for that holding period.")
            return pd.DataFrame()

        print(f"    - Target Expiry: {target_date} ({days_to_exp} days out)")

        # 2. Fetch Option Chain
        try:
            calls = self.stock.option_chain(target_date).calls
        except:
            return pd.DataFrame()

        T = days_to_exp / 365.0
        results = []

        # 3. Process Strikes
        for _, row in calls.iterrows():
            K = row['strike']

            # Filter: Moneyness (85% to 140%)
            if not (self.S * 0.85 <= K <= self.S * 1.40): continue

            # Pricing Logic
            bid, ask, last = row['bid'], row['ask'], row['lastPrice']
            if bid > 0 and ask > 0: mid_price = (bid + ask) / 2
            elif last > 0: mid_price = last
            else: continue

            cost_basis = mid_price * 100
            if cost_basis > budget: continue

            # 4. Math & Analysis
            market_iv = solve_implied_volatility_safe(mid_price, self.S, K, T, self.r, 'call')

            # Edge = GARCH Vol (Reality) - Market IV (Expectation)
            edge_diff = self.forecasted_vol - market_iv

            # Simulation
            pop, ev = run_monte_carlo_call(self.S, K, days_to_hold, self.r, self.forecasted_vol, mid_price)

            source = "MID" if (bid > 0 and ask > 0) else "LAST"

            results.append({
                "Strategy": "Long Call",
                "Expiry": f"{target_date}",
                "Strike": f"${K}",
                "Debit/Credit": f"-${cost_basis:.0f}",
                "Max_Profit": "Unlimited",
                "Edge": f"{edge_diff * 100:+.1f}% IV",
                "PoP": f"{pop*100:.1f}%",
                "Raw_Score": pop * ev # Used for sorting
            })

        # 5. Format Output
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_call_backspreads(self, max_debit=100, days_to_hold=30, ratio=2):
        """Module 3: Call Backspread Ranker (Explosive Volatility Target)"""
        print(f"\n[*] Scanning Call Backspreads (Target: Explosive Bullish)...")

        # 1. Find valid expirations
        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            print("   ❌ No valid expirations found.")
            return pd.DataFrame()

        results = []

        # 2. Iterate through Expirations and Strikes
        for target_date, days_to_exp in valid_dates:
            try:
                calls = self.stock.option_chain(target_date).calls
            except:
                continue

            shorts = calls[(calls['strike'] >= self.S * 0.90) & (calls['strike'] <= self.S * 1.05)]
            longs = calls[(calls['strike'] > self.S * 1.05) & (calls['strike'] < self.S * 1.30)]

            for _, row_short in shorts.iterrows():
                K_short = row_short['strike']
                p_short = self._get_filtered_price(row_short)
                if p_short <= 0: continue

                for _, row_long in longs.iterrows():
                    K_long = row_long['strike']
                    p_long = self._get_filtered_price(row_long)
                    if p_long <= 0: continue

                    # 3. Calculate Entry Pricing
                    net = (p_long * ratio) - p_short
                    if net < 0:
                        is_credit, entry_val = True, abs(net)
                    else:
                        is_credit, entry_val = False, net

                    # Filter: Accept all Credits, strictly limit Debits
                    if not is_credit and (entry_val * 100) > max_debit: continue

                    # 4. Math & Simulation
                    max_loss = analyze_backspread_structure(K_short, K_long, entry_val, is_credit)
                    pop, ev = run_monte_carlo_backspread(
                        self.S, K_short, K_long, ratio, days_to_exp, self.r, self.forecasted_vol, entry_val, is_credit
                    )

                    # Formatting Net Cost string
                    net_str = f"Cr ${entry_val*100:.0f}" if is_credit else f"Db -${entry_val*100:.0f}"

                    results.append({
                        "Strategy": "Call Backspread",
                        "Expiry": target_date,
                        "Strike": f"-1x {K_short}C / +{ratio}x {K_long}C",
                        "Debit/Credit": net_str,
                        "Max_Profit": "Unlimited",
                        "Max_Loss": f"-${abs(max_loss)*100:.0f}",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev # Sort by Expected Value for Backspreads
                    })

        # 5. Output Formatting
        df = pd.DataFrame(results)
        if not df.empty:
            # Backspreads have low PoP but high upside, EV is the best sorting metric
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_call_butterflies(self, max_debit=100, days_to_hold=30, wing_limit=15):
        """Module 4: Call Butterfly Ranker (Neutral / Pin Target)"""
        print(f"\n[*] Scanning Call Butterflies (Target: Neutral Pinning)...")

        # 1. Find valid expirations
        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append(e)

        if not valid_dates:
            return pd.DataFrame()

        results = []

        # 2. Iterate Expirations
        for target_date in valid_dates:
            try:
                calls = self.stock.option_chain(target_date).calls
            except:
                continue

            # Filter strikes near the current spot price
            relevant_calls = calls[(calls['strike'] >= self.S * 0.85) & (calls['strike'] <= self.S * 1.15)]
            strikes = sorted(relevant_calls['strike'].tolist())

            # 3. Build the Butterfly (Iterate through the "Body" K2)
            for i in range(1, len(strikes)-1):
                K2 = strikes[i]
                row_K2 = relevant_calls[relevant_calls['strike'] == K2].iloc[0]
                p_K2 = (row_K2['bid'] + row_K2['ask'])/2 if row_K2['bid']>0 else row_K2['lastPrice']
                if p_K2 <= 0: continue

                # Iterate through Wing Widths
                for step in range(1, len(strikes)//2):
                    if i - step < 0 or i + step >= len(strikes): continue

                    K1, K3 = strikes[i - step], strikes[i + step]

                    # Ensure Equidistant Wings
                    if abs((K2 - K1) - (K3 - K2)) > 0.01: continue
                    if (K2 - K1) > wing_limit: continue

                    row_K1 = relevant_calls[relevant_calls['strike'] == K1].iloc[0]
                    row_K3 = relevant_calls[relevant_calls['strike'] == K3].iloc[0]

                    p_K1 = (row_K1['bid'] + row_K1['ask'])/2 if row_K1['bid']>0 else row_K1['lastPrice']
                    p_K3 = (row_K3['bid'] + row_K3['ask'])/2 if row_K3['bid']>0 else row_K3['lastPrice']

                    if p_K1 <= 0 or p_K3 <= 0: continue

                    # 4. Pricing & Filtering
                    debit = p_K1 + p_K3 - (2 * p_K2)
                    debit_cost = debit * 100

                    if debit <= 0.01 or debit_cost > max_debit: continue

                    # 5. Math & Simulation
                    max_profit = ((K2 - K1) - debit) * 100
                    days_to_exp = (pd.to_datetime(target_date).date() - self.today).days

                    pop, ev = run_monte_carlo_butterfly(
                        self.S, K1, K2, K3, days_to_exp, self.r, self.forecasted_vol, debit
                    )

                    results.append({
                        "Strategy": "Call Butterfly",
                        "Expiry": target_date,
                        "Strike": f"{K1}/{K2}/{K3}",
                        "Debit/Credit": f"-${debit_cost:.0f}",
                        "Max_Profit": f"${max_profit:.0f}",
                        "Max_Loss": f"-${debit_cost:.0f}",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev # Butterflies are purely an Expected Value play
                    })

        # 6. Output Formatting
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_call_ratio_spreads(self, days_to_hold=30, ratio=2, min_credit=0.0):
        """Module 5: Call Ratio Spread Ranker (Mild Bullish / Volatility Crush Target)"""
        print(f"\n[*] Scanning Call Ratio Spreads (Target: Mild Bullish/Neutral)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try:
                calls = self.stock.option_chain(target_date).calls
            except:
                continue

            longs = calls[(calls['strike'] >= self.S * 0.95) & (calls['strike'] <= self.S * 1.10)]
            shorts = calls[(calls['strike'] > self.S * 1.05) & (calls['strike'] < self.S * 1.35)]

            for _, row_long in longs.iterrows():
                K_long = row_long['strike']
                p_long = self._get_filtered_price(row_long)
                if p_long <= 0: continue

                for _, row_short in shorts.iterrows():
                    K_short = row_short['strike']
                    if K_short <= K_long: continue

                    p_short = self._get_filtered_price(row_short)
                    if p_short <= 0: continue

                    # Cost Logic: Buy 1, Sell Ratio
                    net_cost = p_long - (p_short * ratio)

                    if net_cost > 0:
                        is_debit, entry_val = True, net_cost
                        if entry_val > 0.30: continue # Strict debit limit ($30 max)
                    else:
                        is_debit, entry_val = False, abs(net_cost)
                        if entry_val < min_credit: continue

                    # Math & Simulation
                    max_profit, be_up = analyze_ratio_structure(K_long, K_short, ratio, entry_val, is_debit)
                    pop, ev, var = run_monte_carlo_ratio(
                        self.S, K_long, K_short, ratio, days_to_exp, self.r, self.forecasted_vol, entry_val, is_debit
                    )

                    net_str = f"Cr ${entry_val*100:.0f}" if not is_debit else f"Db -${entry_val*100:.0f}"

                    results.append({
                        "Strategy": "Call Ratio Spread",
                        "Expiry": target_date,
                        "Strike": f"+1x {K_long}C / -{ratio}x {K_short}C",
                        "Debit/Credit": net_str,
                        "Max_Profit": f"${max_profit*100:.0f}",
                        "Max_Loss": f"Infinite (VaR: -${abs(var)*100:.0f})", # Highlight the risk!
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev
                    })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_collars(self, max_wrapper_cost=0.50, days_to_hold=30):
        """Module 6: Collar Ranker (Hedged Bullish Target)"""
        print(f"\n[*] Scanning Collars (Target: Hedged Bullish/High Volatility)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try:
                chain = self.stock.option_chain(target_date)
            except:
                continue

            # Protective Puts: 5% to 15% OTM
            floor_puts = chain.puts[(chain.puts['strike'] >= self.S * 0.85) & (chain.puts['strike'] <= self.S * 0.95)]

            # Covered Calls: 5% to 20% OTM
            cap_calls = chain.calls[(chain.calls['strike'] >= self.S * 1.05) & (chain.calls['strike'] <= self.S * 1.20)]

            for _, put_row in floor_puts.iterrows():
                K_put = put_row['strike']
                p_put = (put_row['bid'] + put_row['ask'])/2 if put_row['bid']>0 else put_row['lastPrice']
                if p_put <= 0: continue

                for _, call_row in cap_calls.iterrows():
                    K_call = call_row['strike']
                    p_call = (call_row['bid'] + call_row['ask'])/2 if call_row['bid']>0 else call_row['lastPrice']
                    if p_call <= 0: continue

                    # Net Wrapper Cost: Buy Put - Sell Call
                    net_cost = p_put - p_call

                    if net_cost > max_wrapper_cost: continue

                    # Math & Simulation
                    max_loss, max_profit = analyze_collar_structure(self.S, K_put, K_call, net_cost)
                    pop, ev, worst = run_monte_carlo_collar(
                        self.S, K_put, K_call, days_to_exp, self.r, self.forecasted_vol, net_cost
                    )

                    # Formatting Cost
                    if net_cost < 0:
                        cost_str = f"Cr ${abs(net_cost)*100:.0f}"
                    else:
                        cost_str = f"Db -${net_cost*100:.0f}"

                    results.append({
                        "Strategy": "Collar (Stock + Opts)",
                        "Expiry": target_date,
                        "Strike": f"+{K_put}P / -{K_call}C",
                        "Debit/Credit": cost_str,
                        "Max_Profit": f"${max_profit*100:.0f}",
                        "Max_Loss": f"-${abs(max_loss)*100:.0f}",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev # Sort by EV
                    })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_iron_condors(self, days_to_hold=30, min_credit=0.30):
        """Module 7: Iron Condor Ranker (Neutral / Volatility Crush Target)"""
        print(f"\n[*] Scanning Iron Condors (Target: Contracting Volatility/Neutral)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try:
                chain = self.stock.option_chain(target_date)
            except:
                continue

            puts, calls = chain.puts, chain.calls

            # 1. Find Short Strikes (The "Body") - 5% to 25% OTM
            short_puts = puts[(puts['strike'] < self.S * 0.95) & (puts['strike'] > self.S * 0.75)]
            short_calls = calls[(calls['strike'] > self.S * 1.05) & (calls['strike'] < self.S * 1.25)]

            # Dynamic Wing Widths based on stock price
            widths = [5, 10, 20] if self.S <= 400 else [10, 25, 50]

            for _, p_short in short_puts.iterrows():
                K_ps = p_short['strike']
                p_ps = (p_short['bid'] + p_short['ask'])/2 if p_short['bid']>0 else p_short['lastPrice']
                if p_ps <= 0: continue

                for _, c_short in short_calls.iterrows():
                    K_cs = c_short['strike']
                    p_cs = (c_short['bid'] + c_short['ask'])/2 if c_short['bid']>0 else c_short['lastPrice']
                    if p_cs <= 0: continue

                    # 2. Find Long Strikes (The "Wings")
                    for w in widths:
                        K_pl = K_ps - w
                        K_cl = K_cs + w

                        # Verify long strikes exist in the chain
                        if K_pl not in puts['strike'].values or K_cl not in calls['strike'].values: continue

                        p_long_put = puts[puts['strike'] == K_pl].iloc[0]
                        p_long_call = calls[calls['strike'] == K_cl].iloc[0]

                        p_pl = (p_long_put['bid'] + p_long_put['ask'])/2 if p_long_put['bid']>0 else p_long_put['lastPrice']
                        p_cl = (p_long_call['bid'] + p_long_call['ask'])/2 if p_long_call['bid']>0 else p_long_call['lastPrice']

                        if p_pl <= 0 or p_cl <= 0: continue

                        # 3. Math & Credit Calculation
                        credit = (p_ps + p_cs) - (p_pl + p_cl)
                        if credit < min_credit: continue

                        max_loss = analyze_condor_structure(K_pl, K_ps, K_cs, K_cl, credit)
                        pop, ev = run_monte_carlo_condor(
                            self.S, K_pl, K_ps, K_cs, K_cl, days_to_exp, self.r, self.forecasted_vol, credit
                        )

                        results.append({
                            "Strategy": "Iron Condor",
                            "Expiry": target_date,
                            "Strike": f"P:{K_pl}/{K_ps} | C:{K_cs}/{K_cl}",
                            "Debit/Credit": f"Cr ${credit*100:.0f}",
                            "Max_Profit": f"${credit*100:.0f}",
                            "Max_Loss": f"-${max_loss*100:.0f}",
                            "PoP": f"{pop*100:.1f}%",
                            "Raw_Score": ev # Sort by EV
                        })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df
    def scan_long_puts(self, budget=500, days_to_hold=20):
        """Module 8: Long Put Ranker (Bearish Target)"""
        print(f"\n[*] Scanning Long Puts (Target: Bearish Breakdown)...")

        target_date = None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if d > days_to_hold:
                target_date, days_to_exp = e, d
                break

        if not target_date: return pd.DataFrame()

        try: puts = self.stock.option_chain(target_date).puts
        except: return pd.DataFrame()

        T = days_to_exp / 365.0
        results = []

        for _, row in puts.iterrows():
            K = row['strike']
            if not (self.S * 0.70 <= K <= self.S * 1.05): continue # Focus ATM/OTM

            bid, ask, last = row['bid'], row['ask'], row['lastPrice']
            mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
            if mid_price <= 0.01: continue

            cost_basis = mid_price * 100
            if cost_basis > budget: continue

            market_iv = solve_implied_volatility_safe(mid_price, self.S, K, T, self.r, 'put')
            edge_diff = self.forecasted_vol - market_iv # We want GARCH > Market IV

            pop, ev = run_monte_carlo_put(self.S, K, days_to_hold, self.r, self.forecasted_vol, mid_price, is_short=False)

            results.append({
                "Strategy": "Long Put",
                "Expiry": target_date,
                "Strike": f"${K}",
                "Debit/Credit": f"-${cost_basis:.0f}",
                "Max_Profit": f"${(K * 100) - cost_basis:.0f}",
                "Edge": f"{edge_diff * 100:+.1f}% IV",
                "PoP": f"{pop*100:.1f}%",
                "Raw_Score": pop * ev
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_short_puts(self, max_collateral=5000, days_to_hold=20):
        """Module 9: Cash-Secured Put Ranker (Neutral/Bullish Income Target)"""
        print(f"\n[*] Scanning Short Puts (Target: Income / Buy the Dip)...")

        target_date = None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if d > days_to_hold:
                target_date, days_to_exp = e, d
                break

        if not target_date: return pd.DataFrame()

        try: puts = self.stock.option_chain(target_date).puts
        except: return pd.DataFrame()

        T = days_to_exp / 365.0
        results = []

        for _, row in puts.iterrows():
            K = row['strike']
            collateral = K * 100
            if collateral > max_collateral: continue # Cash-secured filter
            if not (self.S * 0.80 <= K <= self.S * 1.00): continue # Focus OTM (Selling safety)

            bid, ask, last = row['bid'], row['ask'], row['lastPrice']
            mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
            if mid_price <= 0.01: continue

            premium_collected = mid_price * 100

            market_iv = solve_implied_volatility_safe(mid_price, self.S, K, T, self.r, 'put')
            edge_diff = market_iv - self.forecasted_vol # We want Market IV > GARCH (Overpriced)

            pop, ev = run_monte_carlo_put(self.S, K, days_to_hold, self.r, self.forecasted_vol, mid_price, is_short=True)

            results.append({
                "Strategy": "Cash-Secured Put",
                "Expiry": target_date,
                "Strike": f"${K}",
                "Debit/Credit": f"Cr ${premium_collected:.0f}",
                "Max_Profit": f"${premium_collected:.0f}",
                "Max_Loss": f"-${collateral - premium_collected:.0f}",
                "PoP": f"{pop*100:.1f}%",
                "Raw_Score": pop * ev
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df



    def scan_put_backspreads(self, max_debit=100, days_to_hold=30, ratio=2):
        """Module 10: Put Backspread Ranker (Explosive Bearish / Black Swan Target)"""
        print(f"\n[*] Scanning Put Backspreads (Target: Explosive Bearish/Crash)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try:
                puts = self.stock.option_chain(target_date).puts
            except:
                continue

            # Short Leg: ATM or slightly ITM (Funds the trade)
            shorts = puts[(puts['strike'] >= self.S * 0.95) & (puts['strike'] <= self.S * 1.10)]
            # Long Leg: OTM (Crash Protection)
            longs = puts[(puts['strike'] > self.S * 0.70) & (puts['strike'] < self.S * 0.95)]

            for _, row_short in shorts.iterrows():
                K_short = row_short['strike']
                p_short = self._get_filtered_price(row_short)
                if p_short <= 0: continue

                for _, row_long in longs.iterrows():
                    K_long = row_long['strike']
                    p_long = self._get_filtered_price(row_long)
                    if p_long <= 0: continue

                    # 3. Calculate Entry Pricing
                    net = (p_long * ratio) - p_short

                    if net < 0:
                        is_credit, entry_val = True, abs(net)
                    else:
                        is_credit, entry_val = False, net

                    # Filter: Accept all Credits, limit Debits
                    if not is_credit and (entry_val * 100) > max_debit: continue

                    # 4. Math & Simulation
                    max_loss = analyze_put_backspread_structure(K_short, K_long, entry_val, is_credit)
                    pop, ev = run_monte_carlo_put_backspread(
                        self.S, K_short, K_long, ratio, days_to_exp, self.r, self.forecasted_vol, entry_val, is_credit
                    )

                    net_str = f"Cr ${entry_val*100:.0f}" if is_credit else f"Db -${entry_val*100:.0f}"

                    results.append({
                        "Strategy": "Put Backspread",
                        "Expiry": target_date,
                        "Strike": f"-1x {K_short}P / +{ratio}x {K_long}P",
                        "Debit/Credit": net_str,
                        "Max_Profit": f"High (Below ${K_long})",
                        "Max_Loss": f"-${abs(max_loss)*100:.0f}",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev
                    })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_put_ratio_spreads(self, max_debit=100, days_to_hold=30, ratio=2):
        """Module 11: Put Ratio Spread Ranker (Mild Bearish / Volatility Crush Target)"""
        print(f"\n[*] Scanning Put Ratio Spreads (Target: Mild Bearish/Neutral)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates:
            return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try:
                puts = self.stock.option_chain(target_date).puts
            except:
                continue

            # Long Leg: ATM or Slightly OTM (Higher Strike)
            longs = puts[(puts['strike'] >= self.S * 0.85) & (puts['strike'] <= self.S * 1.05)]
            # Short Leg: Further OTM (Lower Strike)
            shorts = puts[(puts['strike'] > self.S * 0.70) & (puts['strike'] < self.S * 0.95)]

            for _, row_long in longs.iterrows():
                K_long = row_long['strike']
                p_long = self._get_filtered_price(row_long)
                if p_long <= 0: continue

                for _, row_short in shorts.iterrows():
                    K_short = row_short['strike']
                    if K_long <= K_short: continue # Long must be higher than short

                    p_short = self._get_filtered_price(row_short)
                    if p_short <= 0: continue

                    # Net Cost: Buy 1 Long, Sell Ratio Shorts
                    net_cost = p_long - (p_short * ratio)

                    if net_cost > 0:
                        is_debit, entry_val = True, net_cost
                        if (entry_val * 100) > max_debit: continue # Strict debit limit
                    else:
                        is_debit, entry_val = False, abs(net_cost)

                    # Math & Simulation
                    max_profit, be_down = analyze_ratio_put_structure(K_long, K_short, ratio, entry_val, is_debit)
                    pop, ev, var = run_monte_carlo_ratio_put(
                        self.S, K_long, K_short, ratio, days_to_exp, self.r, self.forecasted_vol, entry_val, is_debit
                    )

                    net_str = f"Cr ${entry_val*100:.0f}" if not is_debit else f"Db -${entry_val*100:.0f}"

                    results.append({
                        "Strategy": "Put Ratio Spread",
                        "Expiry": target_date,
                        "Strike": f"+1x {K_long}P / -{ratio}x {K_short}P",
                        "Debit/Credit": net_str,
                        "Max_Profit": f"${max_profit*100:.0f}",
                        "Max_Loss": f"Infinite Downside (VaR: -${abs(var)*100:.0f})",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": ev
                    })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_long_volatility(self, budget_limit=5000, days_to_hold=20):
        """Module 12: Long Straddle/Strangle Ranker (Expanding / Breakout Target)"""
        print(f"\n[*] Scanning Long Straddles/Strangles (Target: Explosive Breakout)...")

        target_date = None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if d > days_to_hold:
                target_date, days_to_exp = e, d
                break

        if not target_date: return pd.DataFrame()

        try:
            calls = self.stock.option_chain(target_date).calls
            puts = self.stock.option_chain(target_date).puts
        except: return pd.DataFrame()

        results = []

        # 1. STRADDLES (ATM)
        common_strikes = set(calls['strike']).intersection(set(puts['strike']))
        for K in common_strikes:
            if abs(self.S - K) > (self.S * 0.02): continue # Only scan within 2% of spot

            row_c = calls[calls['strike'] == K].iloc[0]
            row_p = puts[puts['strike'] == K].iloc[0]

            p_c = (row_c['bid'] + row_c['ask'])/2 if row_c['bid']>0 else row_c['lastPrice']
            p_p = (row_p['bid'] + row_p['ask'])/2 if row_p['bid']>0 else row_p['lastPrice']
            if p_c <= 0 or p_p <= 0: continue

            total_price = p_c + p_p
            if (total_price * 100) > budget_limit: continue

            edge = analyze_vol_candidate(self.S, days_to_hold, self.forecasted_vol, total_price, is_short=False)
            pop, ev = run_monte_carlo_vol(self.S, K, K, days_to_exp, self.r, self.forecasted_vol, total_price, is_short=False)

            results.append({
                "Strategy": "Long Straddle",
                "Expiry": target_date,
                "Strike": f"+{K}P / +{K}C",
                "Debit/Credit": f"Db -${total_price*100:.0f}",
                "Max_Profit": "Unlimited",
                "Max_Loss": f"-${total_price*100:.0f}",
                "PoP": f"{pop*100:.1f}%",
                "Raw_Score": edge # Sort by highest positive edge
            })

        # 2. STRANGLES (OTM)
        strikes_c = sorted(calls[calls['strike'] > self.S * 1.02]['strike'].tolist())
        strikes_p = sorted(puts[puts['strike'] < self.S * 0.98]['strike'].tolist())

        for K_call in strikes_c:
            for K_put in strikes_p:
                if (K_call - K_put) > (self.S * 0.15): continue # Don't scan too wide

                try:
                    row_c = calls[calls['strike'] == K_call].iloc[0]
                    row_p = puts[puts['strike'] == K_put].iloc[0]

                    p_c = (row_c['bid'] + row_c['ask'])/2 if row_c['bid']>0 else row_c['lastPrice']
                    p_p = (row_p['bid'] + row_p['ask'])/2 if row_p['bid']>0 else row_p['lastPrice']
                    if p_c <= 0 or p_p <= 0: continue

                    total_price = p_c + p_p
                    if (total_price * 100) > budget_limit: continue

                    edge = analyze_vol_candidate(self.S, days_to_hold, self.forecasted_vol, total_price, is_short=False)
                    pop, ev = run_monte_carlo_vol(self.S, K_call, K_put, days_to_exp, self.r, self.forecasted_vol, total_price, is_short=False)

                    results.append({
                        "Strategy": "Long Strangle",
                        "Expiry": target_date,
                        "Strike": f"+{K_put}P / +{K_call}C",
                        "Debit/Credit": f"Db -${total_price*100:.0f}",
                        "Max_Profit": "Unlimited",
                        "Max_Loss": f"-${total_price*100:.0f}",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": edge
                    })
                except: continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_short_volatility(self, margin_limit=15000, days_to_hold=20):
        """Module 13: Short Straddle/Strangle Ranker (Contracting / Income Target)"""
        print(f"\n[*] Scanning Short Straddles/Strangles (Target: Volatility Crush/Neutral)...")
        # Logic is identical to above, but we swap the math to is_short=True
        # and calculate Margin Requirement instead of Debit Budget.

        target_date = None
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if d > days_to_hold:
                target_date, days_to_exp = e, d
                break

        if not target_date: return pd.DataFrame()

        try:
            calls = self.stock.option_chain(target_date).calls
            puts = self.stock.option_chain(target_date).puts
        except: return pd.DataFrame()

        results = []

        # 1. SHORT STRANGLES (Safer than Straddles, very common)
        strikes_c = sorted(calls[calls['strike'] > self.S * 1.05]['strike'].tolist())
        strikes_p = sorted(puts[puts['strike'] < self.S * 0.95]['strike'].tolist())

        for K_call in strikes_c:
            for K_put in strikes_p:
                if (K_call - K_put) > (self.S * 0.25): continue

                try:
                    row_c = calls[calls['strike'] == K_call].iloc[0]
                    row_p = puts[puts['strike'] == K_put].iloc[0]

                    p_c = (row_c['bid'] + row_c['ask'])/2 if row_c['bid']>0 else row_c['lastPrice']
                    p_p = (row_p['bid'] + row_p['ask'])/2 if row_p['bid']>0 else row_p['lastPrice']
                    if p_c <= 0 or p_p <= 0: continue

                    total_price = p_c + p_p

                    # Estimate Naked Margin (roughly 20% of stock + premium)
                    margin_req = (self.S * 0.2 * 100) + (total_price * 100)
                    if margin_req > margin_limit: continue

                    edge = analyze_vol_candidate(self.S, days_to_hold, self.forecasted_vol, total_price, is_short=True)
                    pop, ev = run_monte_carlo_vol(self.S, K_call, K_put, days_to_exp, self.r, self.forecasted_vol, total_price, is_short=True)

                    results.append({
                        "Strategy": "Short Strangle",
                        "Expiry": target_date,
                        "Strike": f"-{K_put}P / -{K_call}C",
                        "Debit/Credit": f"Cr ${total_price*100:.0f}",
                        "Max_Profit": f"${total_price*100:.0f}",
                        "Max_Loss": "Infinite",
                        "PoP": f"{pop*100:.1f}%",
                        "Raw_Score": pop * ev # Sort by EV
                    })
                except: continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_bull_call_spreads(self, budget=500, days_to_hold=20):
        """Module 14: Bull Call Spread Ranker (Bullish Target)"""
        print(f"\n[*] Scanning Bull Call Spreads (Target: Bullish Directional)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates: return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try: calls = self.stock.option_chain(target_date).calls
            except: continue

            T = days_to_exp / 365.0
            strikes = calls['strike'].tolist()

            for i in range(len(strikes)):
                for j in range(i + 1, len(strikes)):
                    K_long, K_short = strikes[i], strikes[j]
                    if not (self.S * 0.85 < K_long < self.S * 1.15): continue

                    try:
                        row_long = calls[calls['strike'] == K_long].iloc[0]
                        row_short = calls[calls['strike'] == K_short].iloc[0]

                        p_long = self._get_filtered_price(row_long)
                        p_short = self._get_filtered_price(row_short)
                        if p_long <= 0 or p_short <= 0: continue

                        cost = p_long - p_short
                        if cost <= 0 or (cost * 100) > budget: continue

                        proj_roi, max_val = analyze_vertical_candidate(self.S, K_long, K_short, T, days_to_hold, self.r, self.forecasted_vol, cost, 'bull_call')
                        pop, ev = run_monte_carlo_vertical(self.S, K_long, K_short, days_to_hold, self.r, self.forecasted_vol, cost, 'bull_call')

                        results.append({
                            "Strategy": "Bull Call Spread",
                            "Expiry": target_date,
                            "Strike": f"+{K_long}C / -{K_short}C",
                            "Debit/Credit": f"Db -${cost*100:.0f}",
                            "Max_Profit": f"${(max_val - cost)*100:.0f}",
                            "Max_Loss": f"-${cost*100:.0f}",
                            "PoP": f"{pop*100:.1f}%",
                            "Raw_Score": pop * ev
                        })
                    except: continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_bear_put_spreads(self, budget=500, days_to_hold=20):
        """Module 15: Bear Put Spread Ranker (Bearish Target)"""
        print(f"\n[*] Scanning Bear Put Spreads (Target: Bearish Directional)...")

        valid_dates = []
        for e in self.expirations:
            d = (pd.to_datetime(e).date() - self.today).days
            if days_to_hold <= d <= days_to_hold + 60:
                valid_dates.append((e, d))

        if not valid_dates: return pd.DataFrame()

        results = []

        for target_date, days_to_exp in valid_dates:
            try: puts = self.stock.option_chain(target_date).puts
            except: continue

            T = days_to_exp / 365.0
            strikes = puts['strike'].tolist()

            for i in range(len(strikes)):
                for j in range(i + 1, len(strikes)):
                    K_short, K_long = strikes[i], strikes[j] # For Puts, Long strike is higher
                    if not (self.S * 0.85 < K_long < self.S * 1.15): continue

                    try:
                        row_long = puts[puts['strike'] == K_long].iloc[0]
                        row_short = puts[puts['strike'] == K_short].iloc[0]

                        p_long = self._get_filtered_price(row_long)
                        p_short = self._get_filtered_price(row_short)
                        if p_long <= 0 or p_short <= 0: continue

                        cost = p_long - p_short
                        if cost <= 0 or (cost * 100) > budget: continue

                        proj_roi, max_val = analyze_vertical_candidate(self.S, K_long, K_short, T, days_to_hold, self.r, self.forecasted_vol, cost, 'bear_put')
                        pop, ev = run_monte_carlo_vertical(self.S, K_long, K_short, days_to_hold, self.r, self.forecasted_vol, cost, 'bear_put')

                        results.append({
                            "Strategy": "Bear Put Spread",
                            "Expiry": target_date,
                            "Strike": f"+{K_long}P / -{K_short}P",
                            "Debit/Credit": f"Db -${cost*100:.0f}",
                            "Max_Profit": f"${(max_val - cost)*100:.0f}",
                            "Max_Loss": f"-${cost*100:.0f}",
                            "PoP": f"{pop*100:.1f}%",
                            "Raw_Score": pop * ev
                        })
                    except: continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"]).head(5)
        return df

    def scan_deep_value_leaps(self):
        """
        Institutional Deep Value LEAPS Screener (Diagnostic Version).
        Will explicitly warn the user if a manual override fails the fundamental moat or liquidity tests.
        """
        fmp_key = load_fmp_api_key()
        is_manual = "scan_deep_value_leaps" in st.session_state.get('manual_strategies', [])

        if not fmp_key:
            st.warning("FMP_API_KEY missing from Streamlit secrets or .env. LEAPS Scan aborted.")
            return pd.DataFrame()

        try:
            # --- 1. TECHNICAL CAPITULATION FILTER ---
            hist_5y = yf.Ticker(self.ticker).history(period="5y")
            if hist_5y.empty: return pd.DataFrame()

            ath = hist_5y['High'].max()
            drawdown = (self.S - ath) / ath

            weekly_close = hist_5y['Close'].resample('W-FRI').last().tail(15)
            delta = weekly_close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            weekly_rsi = 100 - (100 / (1 + rs)).iloc[-1]

            if drawdown > -0.30 or weekly_rsi >= 30:
                if not is_manual:
                    return pd.DataFrame() # Silently fail for autopilot
                else:
                    st.toast(f"🔍 LEAPS Overridden: {self.ticker} bypassed technical capitulation rules.")

            # --- 2. FUNDAMENTAL QUALITY FILTER (FMP API) ---
            inc_data, cf_data, met_data = fetch_leaps_fundamental_inputs(
                fmp_key,
                self.ticker,
                limit=5,
            )

            if not isinstance(inc_data, list) or len(inc_data) == 0:
                if isinstance(inc_data, dict) and "Error Message" in inc_data:
                    st.error(f"FMP API Rejected Request: {inc_data['Error Message']}")
                return pd.DataFrame()

            if not isinstance(cf_data, list) or len(cf_data) == 0: return pd.DataFrame()
            if not isinstance(met_data, list) or len(met_data) == 0: return pd.DataFrame()

            # --- DEFENSIVE JSON EXTRACTION ---
            # API schemas shift. We hunt for the data across possible keys and intercept 'null' values.
            def safe_metric(data_dict, possible_keys):
                for k in possible_keys:
                    if k in data_dict and data_dict[k] is not None:
                        try:
                            return float(data_dict[k])
                        except (ValueError, TypeError):
                            continue
                return 0.0

            # 1. Extract Revenue
            recent_rev = safe_metric(inc_data[0], ['revenue', 'totalRevenue'])

            # 2. Extract FCF Margin Safely
            total_fcf_margin = 0
            for cf, inc in zip(cf_data, inc_data):
                fcf = safe_metric(cf, ['freeCashFlow', 'freeCashFlows'])
                rev = max(safe_metric(inc, ['revenue', 'totalRevenue']), 1.0) # Prevent division by zero
                total_fcf_margin += (fcf / rev)
            avg_fcf_margin = total_fcf_margin / len(inc_data) if len(inc_data) > 0 else 0

            # 3. Extract ROIC Safely (Hunting for stable endpoint nomenclature)
            avg_roic = sum(
                safe_metric(m, ['roic', 'returnOnInvestedCapital', 'ROIC'])
                for m in met_data
            ) / len(met_data) if len(met_data) > 0 else 0

            # --- INSTITUTIONAL MOAT TEST ---
            if recent_rev < 100_000_000 or avg_fcf_margin <= 0 or avg_roic <= 0:
                if is_manual:
                    # Debugger Output: If it still fails, it will show you exactly what numbers the API pulled
                    st.warning(f"⛔ **LEAPS Rejected:** {self.ticker} failed the Fundamental Moat Test. "
                               f"(Rev: ${recent_rev/1e6:.0f}M, FCF Margin: {avg_fcf_margin*100:.1f}%, ROIC: {avg_roic*100:.1f}%)")

                    # Secret Debugger: Uncomment the line below to physically see the keys FMP is sending you
                    # st.write("Available Metrics Keys:", list(met_data[0].keys()))
                return pd.DataFrame()

            # --- 3. OPTIONS CHAIN FILTER (LEAPS BUILDER) ---
            from datetime import datetime
            from scipy.stats import norm
            import numpy as np

            today = datetime.today()
            target_strike = self.S * 1.10

            valid_calls = []
            ticker_obj = yf.Ticker(self.ticker)

            for exp in self.expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                dte = (exp_date - today).days

                if dte > 360:
                    try:
                        calls = ticker_obj.option_chain(exp).calls
                    except Exception:
                        continue

                    if calls.empty: continue

                    liquid_calls = calls[calls['openInterest'] > 100].copy()
                    if liquid_calls.empty: continue

                    liquid_calls['strike_diff'] = abs(liquid_calls['strike'] - target_strike)
                    best_call = liquid_calls.loc[liquid_calls['strike_diff'].idxmin()]

                    premium = (best_call['bid'] + best_call['ask']) / 2 if (best_call['bid'] and best_call['ask']) else best_call['lastPrice']
                    break_even = best_call['strike'] + premium

                    # --- INSTITUTIONAL PoP CALCULATION (Log-Normal Drift) ---
                    T = dte / 365.0
                    # Standardize volatility to a decimal. Use historical vol, bounded to a minimum of 15%
                    sigma = max(self.hist_vol / 100.0 if self.hist_vol > 1 else self.hist_vol, 0.15)
                    mu = 0.08 # Assume an 8% annualized upward drift for fundamentally sound monopolies

                    # Calculate d2 for the Break-Even price
                    d2 = (np.log(self.S / break_even) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                    true_pop = norm.cdf(d2) * 100 # Convert to percentage

                    valid_calls.append({
                        "Strategy": "Deep Value LEAPS (10% OTM)",
                        "Strike": f"Call ${best_call['strike']}",
                        "Expiry": exp,
                        "Cost/Premium": premium,
                        "Max_Loss": premium * 100,
                        "Max_Profit": float('inf'),
                        "Break_Even": break_even,
                        "PoP": float(f"{true_pop:.1f}"), # Formats to exactly 1 decimal place
                        "Raw_Score": 99.0
                    })

            if not valid_calls:
                if is_manual:
                    st.warning(f"⛔ **LEAPS Rejected:** Found no liquid Call options for {self.ticker} with > 360 Days to Expiration and Open Interest > 100.")
                return pd.DataFrame()

            return pd.DataFrame([valid_calls[-1]])

        except Exception as e:
            st.error(f"FMP LEAPS Engine Error: {e}")
            return pd.DataFrame()

    def analyze_and_route(self, manual_overrides=None):
        """
        THE AUTOPILOT 2.0: Continuous Multi-Factor Strategy Scoring Engine.
        Scraps the binary if/else tree for an institutional factor-weighted matrix.
        """
        if manual_overrides is None:
            manual_overrides = []

        # 1. INGEST MACRO ENVIRONMENT (From Page 1/2 Global State)
        target_beta = float(st.session_state.get('target_beta', 0.5))
        dxy_veto = st.session_state.get('dxy_macro_veto', False)

        # 2. CALCULATE MICROSTRUCTURE FACTORS
        # Volatility Risk Premium (VRP): IV minus GARCH. Positive = Options are overpriced.
        vrp = self.market_iv - self.garch_vol
        vrp_z = vrp / (self.market_iv + 1e-6) # Normalized VRP deviation

        # Momentum Z-Score (20-Day Velocity)
        roll_std = self.tech_df['Close'].tail(20).std()
        mom_z = (self.S - self.tech_df['SMA_20'].iloc[-1]) / roll_std if roll_std > 0 else 0

        # RSI Extremes
        rsi_oversold = max(0, 40 - self.rsi) / 40.0 # Scales 0 to 1
        rsi_overbought = max(0, self.rsi - 60) / 40.0 # Scales 0 to 1

        # 3. STRATEGY SCORING ENGINE (Base 50, Range roughly 0-100)
        scores = {}

        # --- LONG VOLATILITY (Needs VRP < 0) ---
        scores['scan_long_calls'] = 50 + (mom_z * 15) - (vrp_z * 50) + (rsi_oversold * 20) - ((1 - target_beta) * 30)
        scores['scan_long_puts'] = 50 - (mom_z * 15) - (vrp_z * 50) + (rsi_overbought * 20) + ((1 - target_beta) * 30)
        scores['scan_long_volatility'] = 50 - abs(mom_z * 10) - (vrp_z * 80) # Pure breakout expectation

        # --- SHORT VOLATILITY (Needs VRP > 0) ---
        scores['scan_short_puts'] = 50 + (mom_z * 10) + (vrp_z * 60) + (rsi_oversold * 15) - ((1 - target_beta) * 20)
        scores['scan_iron_condors'] = 50 - abs(mom_z * 20) + (vrp_z * 80) # Pure mean reversion
        scores['scan_short_volatility'] = 50 - abs(mom_z * 20) + (vrp_z * 100)

        # --- DIRECTIONAL SPREADS (Hedged Volatility) ---
        scores['scan_bull_call_spreads'] = 50 + (mom_z * 20) + (vrp_z * 20) - ((1 - target_beta) * 20)
        scores['scan_bear_put_spreads'] = 50 - (mom_z * 20) + (vrp_z * 20) + ((1 - target_beta) * 20)

        # --- COMPLEX / TAIL RISK ---
        scores['scan_call_backspreads'] = 50 + (mom_z * 25) - (vrp_z * 30) - (rsi_overbought * 20)
        scores['scan_put_backspreads'] = 50 - (mom_z * 25) - (vrp_z * 30) - (rsi_oversold * 20) + ((1 - target_beta) * 20)
        scores['scan_collars'] = 50 + (mom_z * 10) + (vrp_z * 10) + ((1 - target_beta) * 40) # Highly rated in risk-off
        scores['scan_calendar_spreads'] = 50 - abs(mom_z * 15) + (vrp_z * 30)

        # --- DEEP VALUE LEAPS (Extreme Capitulation Only) ---
        # If Momentum is violently negative AND RSI is fully oversold, this strategy jumps to the #1 rank.
        is_capitulating = 1 if mom_z < -2.0 and rsi_oversold > 0.5 else 0
        scores['scan_deep_value_leaps'] = 10 + (is_capitulating * 90)

        # Apply Global Macro Hard Veto
        if dxy_veto:
            scores['scan_long_calls'] -= 50
            scores['scan_short_puts'] -= 50
            scores['scan_call_backspreads'] -= 50

        # Sort strategies mathematically by score
        ranked_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3_strats = [s[0] for s in ranked_strategies[:3]] # Extract the absolute best 3 models

        # Combine with user's manual overrides
        strats_to_run = list(set(top_3_strats + manual_overrides))

        # Output UI Situation Report
        situation = {
            "Price": self.S,
            "Market_IV": self.market_iv,
            "GARCH_IV": self.garch_vol,
            "VRP": vrp,
            "Mom_Z": mom_z,
            "Ranked_Strats": ranked_strategies[:5] # Send top 5 to UI for transparency
        }

        def add_trades(current_df, new_trades):
            if new_trades is not None and not new_trades.empty:
                if current_df.empty: return new_trades
                else: return pd.concat([current_df, new_trades], ignore_index=True)
            return current_df

        final_results = pd.DataFrame()

        # 4. EXECUTE TOP STRATEGIES
        try:
            for method_name in strats_to_run:
                forced_results = pd.DataFrame()
                if method_name == "scan_calendar_spreads": forced_results = self.scan_calendar_spreads(budget=self.budget)
                elif method_name == "scan_long_calls": forced_results = self.scan_long_calls(budget=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_call_backspreads": forced_results = self.scan_call_backspreads(max_debit=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_call_butterflies": forced_results = self.scan_call_butterflies(max_debit=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_call_ratio_spreads": forced_results = self.scan_call_ratio_spreads(days_to_hold=self.hold_days)
                elif method_name == "scan_collars": forced_results = self.scan_collars(max_wrapper_cost=self.budget/100, days_to_hold=self.hold_days)
                elif method_name == "scan_iron_condors": forced_results = self.scan_iron_condors(days_to_hold=self.hold_days)
                elif method_name == "scan_long_puts": forced_results = self.scan_long_puts(budget=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_short_puts": forced_results = self.scan_short_puts(max_collateral=self.budget*10, days_to_hold=self.hold_days)
                elif method_name == "scan_put_backspreads": forced_results = self.scan_put_backspreads(max_debit=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_put_ratio_spreads": forced_results = self.scan_put_ratio_spreads(days_to_hold=self.hold_days)
                elif method_name == "scan_long_volatility": forced_results = self.scan_long_volatility(budget_limit=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_short_volatility": forced_results = self.scan_short_volatility(margin_limit=self.budget*10, days_to_hold=self.hold_days)
                elif method_name == "scan_bull_call_spreads": forced_results = self.scan_bull_call_spreads(budget=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_bear_put_spreads": forced_results = self.scan_bear_put_spreads(budget=self.budget, days_to_hold=self.hold_days)
                elif method_name == "scan_deep_value_leaps":
                    forced_results = self.scan_deep_value_leaps()

                if not forced_results.empty:
                    if method_name in manual_overrides and method_name not in top_3_strats:
                        forced_results["Strategy"] = "*(Manual)* " + forced_results["Strategy"]
                    final_results = add_trades(final_results, forced_results)
        except Exception as e:
            st.error(f"Routing Matrix Error: {e}")

        # Final sort by EV. Math is the only metric that matters.
        if not final_results.empty and "Raw_Score" in final_results.columns:
            final_results = final_results.drop_duplicates(subset=['Strategy', 'Strike']).sort_values(by="Raw_Score", ascending=False).drop(columns=["Raw_Score"])

        return situation, final_results

# --- 5. MAIN DASHBOARD UI ---
if st.session_state.get('options_scan_executed', False):
    if not target_ticker:
        st.error("Please enter a valid ticker." if not is_zh else "请输入有效的代码。")
    else:
        with st.spinner(f"Initiating Quant Scan for {target_ticker}..." if not is_zh else f"正在为 {target_ticker} 启动量化扫描..."):
            scanner = StreamlitOptionScanner(target_ticker, max_budget, target_hold_days)
            situation, top_trades = scanner.analyze_and_route(manual_overrides=st.session_state.get('manual_strategies', []))

            st.subheader(f"🏢 Asset Snapshot: {target_ticker}" if not is_zh else f"🏢 资产快照: {target_ticker}")

            # --- ROW 1: QUANTITATIVE MICROSTRUCTURE SNAPSHOT ---
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Spot Price" if not is_zh else "现货价格", f"${scanner.S:.2f}", f"{scanner.daily_change:+.2f}% Today" if not is_zh else f"今日 {scanner.daily_change:+.2f}%", delta_color="normal")

            # Map Momentum Z-Score to text
            mom_z = situation['Mom_Z']
            if mom_z > 1.5: mom_text = "Explosive Breakout"
            elif mom_z > 0.5: mom_text = "Bullish Drift"
            elif mom_z < -1.5: mom_text = "Severe Breakdown"
            elif mom_z < -0.5: mom_text = "Bearish Bleed"
            else: mom_text = "Sideways Chop"

            s2.metric("Momentum Z-Score" if not is_zh else "动量 Z-Score", f"{mom_z:+.2f}", mom_text, delta_color="normal" if mom_z > 0 else "inverse")

            vrp_pct = situation['VRP'] * 100
            # If VRP is positive, options are expensive (Green text because it's good for selling). If negative, they are cheap (Red text).
            s3.metric("Vol Risk Premium (VRP)" if not is_zh else "波动率风险溢价 (VRP)", f"{vrp_pct:+.1f}%", "Options are OVERPRICED" if vrp_pct > 0 else "Options are CHEAP", delta_color="normal" if vrp_pct > 0 else "inverse")

            # Show the top ranked strategy from the engine
            top_strat_raw = situation['Ranked_Strats'][0][0].replace('scan_', '').replace('_', ' ').title()
            s4.metric("Engine Top Pick" if not is_zh else "引擎首选", top_strat_raw, f"Edge Score: {situation['Ranked_Strats'][0][1]:.0f}/100", delta_color="off")

            with st.expander("🔬 View Algorithm Strategy Weights" if not is_zh else "🔬 查看算法策略权重"):
                st.markdown("The underlying Machine Learning engine ranked all 15 strategies based on Momentum Velocity, Volatility Risk Premium, and Macro Beta constraint. Here are the top 5 highest-probability matches for the current environment:" if not is_zh else "底层机器学习引擎基于动量速度、波动率风险溢价和宏观 Beta 约束对所有 15 种策略进行了排名。以下是当前环境下概率最高的 5 个匹配项：")
                rank_data = [{"Strategy": s[0].replace('scan_', '').replace('_', ' ').title(), "Suitability Score": f"{s[1]:.0f}/100"} for s in situation['Ranked_Strats']]
                st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # --- ROW 2: THE VOLATILITY ORACLE ---
            st.subheader(f"📡 Volatility Oracle" if not is_zh else f"📡 波动率预言机")
            v1, v2, v3 = st.columns(3)
            v1.metric("Historical Vol (30d)" if not is_zh else "历史波动率 (30天)", f"{scanner.hist_vol*100:.1f}%", "Reality (Past 30 Days)" if not is_zh else "实际 (过去30天)", delta_color="off")

            garch_delta = (scanner.garch_vol - scanner.hist_vol) * 100
            v2.metric("GARCH(1,1) Forecast" if not is_zh else "GARCH(1,1) 预测", f"{scanner.garch_vol*100:.1f}%", f"{garch_delta:+.1f}% vs Reality" if not is_zh else f"较实际 {garch_delta:+.1f}%", delta_color="normal" if garch_delta > 0 else "inverse")

            har_delta = (scanner.har_vol - scanner.hist_vol) * 100
            v3.metric("HAR-RV Forecast" if not is_zh else "HAR-RV 预测", f"{scanner.har_vol*100:.1f}%", f"{har_delta:+.1f}% vs Reality" if not is_zh else f"较实际 {har_delta:+.1f}%", delta_color="normal" if har_delta > 0 else "inverse")

            st.markdown("---")

            # --- ROW 3: TECHNICAL CHART & TRADES ---
            # Widened the left column from 1 to 1.4 to give the metrics more breathing room
            tc1, tc2 = st.columns([1.4, 2.2])
            with tc1:
                st.markdown("**Moving Averages**" if not is_zh else "**移动平均线**")
                curr_price = scanner.S
                with st.container(border=True):
                    m1, m2 = st.columns(2)
                    with m1:
                        # Removed the '$' and reduced to 1 decimal place to prevent squishing
                        st.metric("SMA 20", f"{scanner.tech_df['SMA_20'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['SMA_20'].iloc[-1])/scanner.tech_df['SMA_20'].iloc[-1] * 100:+.1f}%")
                        st.metric("SMA 60", f"{scanner.tech_df['SMA_60'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['SMA_60'].iloc[-1])/scanner.tech_df['SMA_60'].iloc[-1] * 100:+.1f}%")
                        st.metric("SMA 120", f"{scanner.tech_df['SMA_120'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['SMA_120'].iloc[-1])/scanner.tech_df['SMA_120'].iloc[-1] * 100:+.1f}%")
                    with m2:
                        st.metric("SMA 200", f"{scanner.tech_df['SMA_200'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['SMA_200'].iloc[-1])/scanner.tech_df['SMA_200'].iloc[-1] * 100:+.1f}%")
                        st.metric("EMA 9", f"{scanner.tech_df['EMA_9'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['EMA_9'].iloc[-1])/scanner.tech_df['EMA_9'].iloc[-1] * 100:+.1f}%")
                        st.metric("EMA 21", f"{scanner.tech_df['EMA_21'].iloc[-1]:.1f}", f"{(curr_price - scanner.tech_df['EMA_21'].iloc[-1])/scanner.tech_df['EMA_21'].iloc[-1] * 100:+.1f}%")

            with tc2:
                if scanner.next_earnings:
                    days_to_earn = (scanner.next_earnings - date.today()).days
                    if 0 <= days_to_earn <= 45:
                        st.warning(f"⚠️ **EVENT RISK DETECTED:** Earnings expected on {scanner.next_earnings} (in {days_to_earn} days). Volatility will be inflated!" if not is_zh else f"⚠️ **事件风险检测:** 预计在 {scanner.next_earnings} ({days_to_earn} 天后) 发布财报。波动率将被推高！")

                with st.container(border=True):
                    col_title, col_ctrl = st.columns([2, 1])
                    col_title.markdown("**🎯 Expected Move (Probability Cone)**" if not is_zh else "**🎯 预期波动 (概率锥)**")
                    sd_choice = col_ctrl.selectbox("Confidence Level:" if not is_zh else "置信水平:", ["1 SD (68%)", "2 SD (95%)", "3 SD (99%)"], index=0, label_visibility="collapsed")
                    sd_mult = int(sd_choice[0])

                    plot_df = scanner.tech_df.tail(126)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Historical Price' if not is_zh else '历史价格', line=dict(color='#E2E8F0', width=2)))

                    future_dates = pd.date_range(start=plot_df.index[-1], periods=30, freq='B')
                    days_out = np.arange(1, len(future_dates) + 1)
                    move_pcts = scanner.garch_vol * np.sqrt(days_out / 252.0) * sd_mult

                    x_cone = [plot_df.index[-1]] + list(future_dates)
                    fig.add_trace(go.Scatter(x=x_cone, y=[scanner.S] + list(scanner.S * (1 + move_pcts)), mode='lines', name=f'+{sd_mult} SD Upper', line=dict(color='#00C853', width=1.5, dash='dash')))
                    fig.add_trace(go.Scatter(x=x_cone, y=[scanner.S] + list(scanner.S * (1 - move_pcts)), mode='lines', name=f'-{sd_mult} SD Lower', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(color='#FF3B30', width=1.5, dash='dash')))

                    fig.update_layout(template="plotly_dark", height=350, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

            # --- ROW 3.5: THE VOLATILITY SURFACE ---
            with st.container(border=True):
                st.markdown("### 🌊 Volatility Smile (Near-Term Skew)" if not is_zh else "### 🌊 波动率微笑 (近月偏度)")
                try:
                    chain_date = scanner.expirations[0]
                    calls = scanner.stock.option_chain(chain_date).calls
                    puts = scanner.stock.option_chain(chain_date).puts
                    calls = calls[(calls['strike'] > scanner.S * 0.85) & (calls['strike'] < scanner.S * 1.15)]
                    puts = puts[(puts['strike'] > scanner.S * 0.85) & (puts['strike'] < scanner.S * 1.15)]

                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Scatter(x=puts['strike'], y=puts['impliedVolatility']*100, mode='lines+markers', name='Puts IV' if not is_zh else '看跌期权 IV', line=dict(color='#FF3B30')))
                    fig_skew.add_trace(go.Scatter(x=calls['strike'], y=calls['impliedVolatility']*100, mode='lines+markers', name='Calls IV' if not is_zh else '看涨期权 IV', line=dict(color='#00C853')))
                    fig_skew.add_vline(x=scanner.S, line_width=2, line_dash="dash", line_color="#E2E8F0", annotation_text="Spot Price" if not is_zh else "现货价格")

                    fig_skew.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_skew, use_container_width=True)

                    # --- THE CHEAT SHEET ---
                    with st.expander("📖 How to read this Volatility Skew chart" if not is_zh else "📖 如何阅读此波动率偏度图"):
                        if not is_zh:
                            st.markdown("""
                            **Implied Volatility (IV) is not a single number.** Market makers charge different prices for different strikes based on fear and greed.

                            * 📉 **The 'Smirk' (Red Line > Green Line):** Puts are significantly more expensive than Calls. The market is terrified of a sudden crash and is heavily buying downside insurance. (Normal for SPY/QQQ).
                            * 📈 **Forward Skew (Green Line > Red Line):** Calls are more expensive than Puts. Speculators are aggressively buying upside lotto tickets (common in crypto, biotech, or meme stocks).
                            * 🦇 **The 'Smile' (U-Shape):** Both deep OTM Puts and Calls are very expensive. The market expects a massive, violent move but has no idea which direction it will go (usually seen right before an Earnings report).
                            """)
                        else:
                            st.markdown("""
                            **隐含波动率 (IV) 并不是一个单一的数字。** 做市商会根据恐惧和贪婪对不同的行权价收取不同的价格。

                            * 📉 **'撇嘴'偏度 (红线 > 绿线):** 看跌期权比看涨期权贵得多。市场非常害怕突然崩盘，正在大量购买下行保险。（SPY/QQQ 的常态）。
                            * 📈 **正向偏度 (绿线 > 红线):** 看涨期权比看跌期权贵。投机者正在大举买入上行彩票（常见于加密货币、生物科技或妖股）。
                            * 🦇 **'微笑'曲线 (U型):** 深度价外的看跌和看涨期权都非常昂贵。市场预计会发生剧烈波动，但不知道方向（通常在发布财报前看到）。
                            """)

                except Exception as e:
                    st.caption(f"Could not render volatility smile: {e}" if not is_zh else f"无法渲染波动率微笑曲线: {e}")

            # --- ROW 4: TRADE TABLE & PAYOFF DIAGRAMS ---
            st.markdown("---")
            st.subheader("🏆 Top Algorithmic Trade Recommendations" if not is_zh else "🏆 顶级算法交易推荐")

            if not top_trades.empty:
                st.dataframe(top_trades, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### 📊 Interactive Payoff Visualizer" if not is_zh else "### 📊 交互式盈亏可视化")

                trade_options = []
                for idx, row in top_trades.iterrows():
                    strategy = row.get('Strategy', 'Trade')
                    strikes = row.get('Strikes', row.get('Strike', row.get('Strike(s)', '')))
                    score = row.get('PoP', row.get('PoP (%)', row.get('Expected Value', 'N/A')))
                    score_str = f"{score*100:.1f}%" if isinstance(score, (int, float)) and score < 1.0 else str(score)
                    trade_options.append(f"{strategy} | {strikes} | Score: {score_str}")

                selected_trade_idx = st.selectbox("Select a trade to visualize its Risk/Reward profile:" if not is_zh else "选择一个交易以可视化其风险/回报分布:", options=range(len(trade_options)), format_func=lambda x: trade_options[x])
                selected_trade = top_trades.iloc[selected_trade_idx]

                def parse_currency(val):
                    if isinstance(val, (int, float)): return val
                    if isinstance(val, str):
                        clean_str = val.replace('$', '').replace(',', '').replace('Cr', '').replace('Db', '').replace('+', '').replace('-', '').strip()
                        if clean_str.lower() in ['unlimited', 'infinite', 'high', 'infinite downside']: return 999999.0
                        try: return float(clean_str.split(' ')[0]) # Handle cases like 'Infinite (VaR: -X)'
                        except ValueError: return 0.0
                    return 0.0

                max_profit = parse_currency(selected_trade.get('Max_Profit', 0))
                max_loss = parse_currency(selected_trade.get('Max_Loss', 0))
                breakeven = selected_trade.get('Breakeven', scanner.S)

                price_range = np.linspace(scanner.S * 0.85, scanner.S * 1.15, 500)
                pnl = []

                for p in price_range:
                    if "Call" in str(selected_trade.get('Strategy', '')): val = (p - breakeven) * 100
                    elif "Put" in str(selected_trade.get('Strategy', '')): val = (breakeven - p) * 100
                    else: val = max_profit if (p > breakeven * 0.95 and p < breakeven * 1.05) else -abs(max_loss)

                    if isinstance(max_profit, (int, float)) and val > max_profit: val = max_profit
                    if isinstance(max_loss, (int, float)) and val < -abs(max_loss): val = -abs(max_loss)
                    pnl.append(val)

                pnl = np.array(pnl)
                y_profit, y_loss = np.where(pnl > 0, pnl, 0), np.where(pnl <= 0, pnl, 0)

                with st.container(border=True):
                    fig_payoff = go.Figure()
                    fig_payoff.add_trace(go.Scatter(x=price_range, y=y_profit, mode='lines', name='Profit Zone' if not is_zh else '盈利区', line=dict(color='#00C853', width=2), fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.15)'))
                    fig_payoff.add_trace(go.Scatter(x=price_range, y=y_loss, mode='lines', name='Risk Zone' if not is_zh else '风险区', line=dict(color='#FF3B30', width=2), fill='tozeroy', fillcolor='rgba(255, 59, 48, 0.15)'))
                    fig_payoff.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
                    fig_payoff.add_vline(x=scanner.S, line_width=1.5, line_dash="dot", line_color="#A0AEC0")
                    fig_payoff.update_layout(template="plotly_dark", height=400, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f"<b>Risk/Reward Profile:</b> {trade_options[selected_trade_idx].split('|')[0]}", xaxis_title="Underlying Price at Expiration" if not is_zh else "到期时的标的价格", yaxis_title="Estimated Profit / Loss ($)" if not is_zh else "预计盈亏 ($)", hovermode='x unified', showlegend=False)
                    st.plotly_chart(fig_payoff, use_container_width=True)
            else:
                st.info("No high-probability trades fit your budget and current market regime. Cash is a position." if not is_zh else "没有符合您的预算和当前市场状态的高概率交易。持有现金也是一种策略。")
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Master Scan' to begin." if not is_zh else "👈 在侧边栏配置参数，然后点击“运行主扫描”开始。")
