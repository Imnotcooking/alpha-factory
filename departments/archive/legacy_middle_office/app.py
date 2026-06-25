import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
import json
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from oqp.portfolio import (
        DEFAULT_IBKR_METRICS_PATH,
        ManualPortfolioInputs,
        compute_nav_drawdowns,
        default_portfolio_ledger_path,
        load_historical_nav,
        load_latest_live_positions,
        value_portfolio_snapshot,
        write_historical_nav,
    )
except Exception:
    DEFAULT_IBKR_METRICS_PATH = None
    ManualPortfolioInputs = None
    compute_nav_drawdowns = None
    default_portfolio_ledger_path = None
    load_historical_nav = None
    load_latest_live_positions = None
    value_portfolio_snapshot = None
    write_historical_nav = None

# --- 1. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Macro-Terminal Command Center", layout="wide", page_icon="🌍")

from utils.theme import load_css
try:
    load_css()
except:
    pass

# SURGICAL PATCH: Prevent Material Icon ligature text leak in expanders
st.markdown("""
    <style>
    /* Force Streamlit's internal icon spans to retain their native material font */
    span.material-symbols-rounded,
    div[data-testid="stExpander"] summary span {
        font-family: 'Material Symbols Rounded' !important;
    }
    </style>
""", unsafe_allow_html=True)

# SURGICAL PATCH: Prevent Metric Truncation
st.markdown("""
    <style>
    /* Force Streamlit metrics to dynamically resize and prevent '...' truncation */
    div[data-testid="metric-container"] > div > div {
        font-size: 1.6rem !important; /* Shrunk from default 2rem */
        white-space: nowrap !important;
    }
    </style>
""", unsafe_allow_html=True)

# SURGICAL PATCH: Fix Input Field Text Contrast
st.markdown("""
    <style>
    /* Force text color inside all input boxes to be pure black and slightly bold */
    div[data-baseweb="input"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1.5 GLOBAL STATE INITIALIZATION ---
if 'scan_executed' not in st.session_state: st.session_state['scan_executed'] = False
if 'last_mode' not in st.session_state: st.session_state['last_mode'] = "Option Scanner"
if 'portfolio_value' not in st.session_state: st.session_state['portfolio_value'] = 0.0
if 'portfolio_beta' not in st.session_state: st.session_state['portfolio_beta'] = 1.0

# --- LOCALIZATION ENGINE (EN/ZH) ---
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)
is_zh = lang == "中文"

# Translation Dictionary
t = {
    "title": "🌍 Macro-Terminal Command Center" if not is_zh else "🌍 宏观终端指挥中心",
    "analysis_win": "⚙️ Analysis Window" if not is_zh else "⚙️ 分析窗口",
    "lookback": "Historical Lookback" if not is_zh else "历史回溯期",
    "bench": "Benchmark" if not is_zh else "基准指数",
    "manual_inputs": "⚙️ Global Settings & Manual Inputs" if not is_zh else "⚙️ 全局设置与手动输入",
    "cash": "💵 Liquid Cash (Uninvested)" if not is_zh else "💵 流动现金 (未投资)",
    "cny_assets": "🇨🇳 Manual Asian Assets" if not is_zh else "🇨🇳 亚洲资产手动输入",
    "overview": "### 🏦 Portfolio Overview" if not is_zh else "### 🏦 投资组合概览",
    "net_worth": "Total Net Worth (USD)" if not is_zh else "净资产总值 (USD)",
    "unrealized_pl": "Total Unrealized P/L ($)" if not is_zh else "未实现盈亏总额 ($)",
    "overall_pl_pct": "Overall P/L (%)" if not is_zh else "总盈亏 (%)",
    "reserves": "Liquid Reserves" if not is_zh else "流动储备金",
    "alloc_tracker": "### 📋 Interactive Allocation Tracker" if not is_zh else "### 📋 交互式资产配置追踪",
    "alloc_pie": "Allocation by Category" if not is_zh else "按类别分类的资产配置",
    "perf_engine": "### 📈 Performance & Risk Engine" if not is_zh else "### 📈 绩效与风险引擎",
    "macro_pulse": "### 🌐 Global Macro Pulse (Live)" if not is_zh else "### 🌐 全球宏观脉搏 (实时)",
    "save_btn": "💾 SAVE ALL DASHBOARD CHANGES" if not is_zh else "💾 保存所有仪表板更改",
    "export_btn": "📄 EXPORT PORTFOLIO LEDGER (CSV)" if not is_zh else "📄 导出投资组合账本 (CSV)",
    "fetching": "Fetching market data..." if not is_zh else "正在获取市场数据..."
}

st.title(t["title"])

# --- MEMORY SYSTEM: LOAD/SAVE DEFAULTS ---
DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_defaults.json")

def load_defaults():
    if os.path.exists(DEFAULTS_FILE):
        with open(DEFAULTS_FILE, "r") as f:
            return json.load(f)
    return {
        "t212_cash_eur": 0.0, "futu_cash_usd": 0.0, "cny_mutual_fund": 94000.0,
        "cny_mutual_fund_pnl": 0.0, "cny_gold_grams": 8.5, "cny_gold_cost": 900.0,
        "asset_preferences": {}, "custom_headers": {}
    }
defaults = load_defaults()

# --- 2. GLOBAL CONTROLS & MANUAL INPUTS (NOW IN AN EXPANDER) ---
with st.sidebar:
    st.markdown(f"**{t['analysis_win']}**")
    time_window = st.selectbox(t['lookback'], ["1mo", "3mo", "6mo", "1y", "ytd"], index=2)
    benchmark = st.selectbox(t['bench'], ["QQQ", "SPY"])
    st.markdown("---")

with st.expander(t['manual_inputs'], expanded=False):
    col_ctrl2, col_ctrl3 = st.columns(2)
    with col_ctrl2:
        with st.container(border=True):
            st.markdown(f"**{t['cash']}**")
            t212_cash_eur = st.number_input("T212 Cash (EUR)", min_value=0.0, value=float(defaults.get("t212_cash_eur", 0.0)), step=100.0)
            futu_cash_usd = st.number_input("Futu Cash (USD)", min_value=0.0, value=float(defaults.get("futu_cash_usd", 0.0)), step=100.0)

    with col_ctrl3:
        with st.container(border=True):
            st.markdown(f"**{t['cny_assets']}**")
            cny_mutual_fund = st.number_input("Mutual Fund Cost (CNY)", min_value=0.0, value=float(defaults.get("cny_mutual_fund", 94000.0)), step=1000.0)
            cny_mf_pnl = st.number_input("Mutual Fund P/L (CNY)", value=float(defaults.get("cny_mutual_fund_pnl", 0.0)), step=100.0)
            cny_gold_grams = st.number_input("Gold Weight (Grams)", min_value=0.0, value=float(defaults.get("cny_gold_grams", 8.5)), step=0.5)
            cny_gold_cost = st.number_input("Gold Cost (CNY/g)", min_value=0.0, value=float(defaults.get("cny_gold_cost", 900.0)), step=10.0)

# We create an empty container here so the Save button renders at the TOP of the page
global_save_container = st.container()
st.markdown("---")

# --- 3. AUTOMATED DATA INGESTION ---
# Dynamically find the clean_data folder so it works on both Mac and AWS
DB_PATH = (
    str(default_portfolio_ledger_path())
    if default_portfolio_ledger_path is not None
    else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Portfolio",
        "clean_data",
        "macro_terminal.db",
    )
)
CLEAN_DIR = (
    str(DEFAULT_IBKR_METRICS_PATH.parent)
    if DEFAULT_IBKR_METRICS_PATH is not None
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Portfolio", "clean_data")
)

def load_live_positions():
    """Connects to SQLite and fetches the most recent portfolio snapshot."""
    if not os.path.exists(DB_PATH): return None
    if load_latest_live_positions is None:
        st.error("Shared portfolio ledger reader is unavailable.")
        return None
    try:
        df = load_latest_live_positions(DB_PATH)
        return None if df.empty else df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

def load_ibkr_metrics():
    """Loads the live IBKR Cash and Margin Buffer generated by ETL"""
    metrics_file = os.path.join(CLEAN_DIR, "ibkr_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}

df_portfolio = load_live_positions()
ibkr_metrics = load_ibkr_metrics()

# Extract IBKR Metrics
ibkr_cash_usd = ibkr_metrics.get("Available_Cash_USD", 0.0)
ibkr_margin_buffer = ibkr_metrics.get("Margin_Buffer_USD", 0.0)

# --- 4. MARKET DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_market_data(tickers, period):
    clean_tickers = [t for t in tickers if len(str(t)) < 10]
    yf_tickers = [t.replace('BRK.B', 'BRK-B') for t in clean_tickers]
    # NEW: Added Global Macro tickers to the fetch list
    # NEW: Expanded Global Macro tickers
    macro_tickers = ['QQQ', 'SPY', 'TLT', '^HSI', '000300.SS', 'CBON', 'VGK', 'EWJ', 'EWY', 'GC=F', 'CL=F', 'BTC-USD']
    fetch_list = list(set(yf_tickers + ['EURUSD=X', 'GBPUSD=X', 'CNYUSD=X', 'HKDUSD=X'] + macro_tickers))
    data = yf.download(fetch_list, period=period, progress=False)['Close']
    data.dropna(axis=1, how='all', inplace=True)
    data.ffill(inplace=True)
    return data

if df_portfolio is not None:
    unique_tickers = df_portfolio['ticker'].unique().tolist()

    with st.spinner(t['fetching']):
        hist_data = fetch_market_data(unique_tickers, time_window)
        live_prices = hist_data.iloc[-1]

    if value_portfolio_snapshot is None or ManualPortfolioInputs is None:
        st.error("Shared portfolio valuation engine is unavailable.")
        st.stop()

    valuation = value_portfolio_snapshot(
        df_portfolio,
        hist_data,
        benchmark=benchmark,
        manual_inputs=ManualPortfolioInputs(
            t212_cash_eur=float(t212_cash_eur),
            futu_cash_usd=float(futu_cash_usd),
            ibkr_cash_usd=float(ibkr_cash_usd),
            cny_mutual_fund=float(cny_mutual_fund),
            cny_mutual_fund_pnl=float(cny_mf_pnl),
            cny_gold_grams=float(cny_gold_grams),
            cny_gold_cost=float(cny_gold_cost),
        ),
        asset_preferences=defaults.get("asset_preferences", {}),
    )
    df_clean = valuation.position_valuation
    df_broker = valuation.broker_summary
    df_agg = valuation.asset_summary
    usd_hist = valuation.usd_history
    total_portfolio_value = valuation.total_net_worth
    total_portfolio_pnl = valuation.total_pnl
    total_cash_usd = valuation.total_cash
    portfolio_beta = valuation.portfolio_beta
    eur_usd = valuation.fx_rates.get("EUR", 1.0)

    st.session_state['portfolio_value'] = total_portfolio_value
    st.session_state['portfolio_beta'] = portfolio_beta
    st.session_state['unified_portfolio'] = df_agg

    # ==========================================
    # 📊 PORTFOLIO OVERVIEW (8 QUADRANTS)
    # ==========================================
    st.markdown("### 📊 Portfolio Overview" if not is_zh else "### 📊 投资组合概览")
    st.caption("Aggregated global exposure and live broker reconciliation." if not is_zh else "汇总的全球敞口和实时经纪商对账。")

    with st.container(border=True):
        # --- LOAD BANKED PROFITS FROM ETL ---
        banked_json_path = os.path.join(CLEAN_DIR, "banked_profits.json")
        banked_t212_usd = 0.0
        try:
            if os.path.exists(banked_json_path):
                with open(banked_json_path, "r") as f:
                    banked_data = json.load(f)
                    # Convert the scraped EUR profit into live USD using today's FX
                    banked_t212_usd = banked_data.get('Trading212_EUR', 0.0) * eur_usd
        except Exception as e:
            pass

        # Helper function to safely pull values from our df_broker dataframe
        def get_broker_metrics(name):
            if name in df_broker['Broker'].values:
                b_val = df_broker.loc[df_broker['Broker'] == name, 'Current_USD'].values[0]
                b_pnl = df_broker.loc[df_broker['Broker'] == name, 'PnL_USD'].values[0]
                return b_val, b_pnl
            return 0.0, 0.0

        ibkr_val, ibkr_pnl = get_broker_metrics('IBKR Live')
        t212_val, t212_pnl = get_broker_metrics('Trading212')
        futu_val, futu_pnl = get_broker_metrics('Futubull')
        cn_fund_val, cn_fund_pnl = get_broker_metrics('Chinese Fund')
        cn_gold_val, cn_gold_pnl = get_broker_metrics('Chinese Gold')

        # Calculate Row 1 Dynamics (Adding IBKR Cash)
        all_time_pnl = total_portfolio_pnl + banked_t212_usd

        # --- ROW 1: GLOBAL AGGREGATE ---
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        r1c1.metric("Total Net Worth" if not is_zh else "总净值", f"${total_portfolio_value:,.2f}")
        r1c2.metric("All-Time Total PnL" if not is_zh else "历史总盈亏", f"${all_time_pnl:,.2f}", f"${banked_t212_usd:,.2f} Banked" if not is_zh else f"${banked_t212_usd:,.2f} 已落袋", delta_color="normal")

        # Display Margin Buffer under Total Cash
        buffer_text = f"Margin Buffer: ${ibkr_margin_buffer:,.0f}" if not is_zh else f"保证金缓冲: ${ibkr_margin_buffer:,.0f}"
        r1c3.metric("Total Liquid Cash" if not is_zh else "总流动现金", f"${total_cash_usd:,.2f}", buffer_text, delta_color="off")

        r1c4.metric("Portfolio Beta" if not is_zh else "投资组合 Beta", f"{portfolio_beta:.2f}", "Systematic Risk" if not is_zh else "系统性风险", delta_color="inverse" if portfolio_beta > 1.2 else "off")

        nav_history = pd.DataFrame()
        if (
            write_historical_nav is not None
            and load_historical_nav is not None
            and compute_nav_drawdowns is not None
        ):
            try:
                write_historical_nav(
                    DB_PATH,
                    snapshot_date=datetime.date.today(),
                    total_net_worth=float(total_portfolio_value),
                    total_cash=float(total_cash_usd),
                    portfolio_beta=float(portfolio_beta),
                )
                nav_history = compute_nav_drawdowns(load_historical_nav(DB_PATH))
            except Exception as e:
                st.warning(f"NAV history update failed: {e}")

        if not nav_history.empty:
            latest_nav = nav_history.iloc[-1]
            max_drawdown = float(nav_history["drawdown"].min())
            max_drawdown_pct = float(nav_history["drawdown_pct"].min()) * 100
            st.markdown("#### 📈 Equity & Drawdown History" if not is_zh else "#### 📈 净值与回撤历史")
            nav_cols = st.columns(4)
            nav_cols[0].metric("Stored NAV Days" if not is_zh else "已存净值天数", str(len(nav_history)))
            nav_cols[1].metric("Latest NAV" if not is_zh else "最新净值", f"${float(latest_nav['total_net_worth']):,.2f}")
            nav_cols[2].metric("Daily PnL" if not is_zh else "单日盈亏", f"${float(latest_nav['daily_pnl']):,.2f}")
            nav_cols[3].metric("Max Drawdown" if not is_zh else "最大回撤", f"${max_drawdown:,.2f}", f"{max_drawdown_pct:.2f}%")

            chart_nav = nav_history.set_index("date")[["total_net_worth", "equity_peak"]]
            chart_dd = nav_history.set_index("date")[["drawdown"]]
            st.line_chart(chart_nav)
            st.line_chart(chart_dd)

        st.divider()

        # --- ROW 2: LIVE BROKER PnL ---
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)

        # Helper function to perfectly format the PnL string (e.g. "+$1,234" or "-$500")
        def fmt_pnl(val): return f"{'+$' if val >= 0 else '-$'}{abs(val):,.0f}"

        r2c1.metric("IBKR Live" if not is_zh else "IBKR (实时)", f"${ibkr_val:,.0f}", fmt_pnl(ibkr_pnl), delta_color="normal")
        r2c2.metric("T212" if not is_zh else "T212", f"${t212_val:,.0f}", fmt_pnl(t212_pnl), delta_color="normal")
        r2c3.metric("Futubull" if not is_zh else "富途牛牛", f"${futu_val:,.0f}", fmt_pnl(futu_pnl), delta_color="normal")
        r2c4.metric("CN Fund" if not is_zh else "招商基金", f"${cn_fund_val:,.0f}", fmt_pnl(cn_fund_pnl), delta_color="normal")
        r2c5.metric("CN Gold" if not is_zh else "中国黄金", f"${cn_gold_val:,.0f}", fmt_pnl(cn_gold_pnl), delta_color="normal")

        # --- AGGREGATE CHECKER ---
        calculated_sum = ibkr_pnl + t212_pnl + futu_pnl + cn_fund_pnl + cn_gold_pnl

        if abs(calculated_sum - total_portfolio_pnl) > 1.0:
            st.error(
                f"⚠️ **DATA MISMATCH DETECTED:** The sum of your 4 brokers' live PnL is **${calculated_sum:+.2f}**, "
                f"but the master dashboard is reporting **${total_portfolio_pnl:+.2f}**. Check your CSV FX conversion rates or missing data rows!"
                if not is_zh else
                f"⚠️ **检测到数据不匹配:** 您的 4 个经纪商的实时盈亏总和为 **${calculated_sum:+.2f}**, "
                f"但主仪表板报告为 **${total_portfolio_pnl:+.2f}**。请检查您的 CSV 外汇转换率或丢失的数据行！"
            )
    # ==========================================
    # 5.2 INSTITUTIONAL REGIME GOVERNOR (SJM MULTI-FACTOR)
    # ==========================================
    st.markdown("### 🏛️ Institutional Regime Governor (Continuous SJM)" if not is_zh else "### 🏛️ 机构级状态调节器 (连续 SJM)")
    st.caption("Multi-factor probability model utilizing Trend, Volatility Term Structure, and Credit Spreads with a Jump Penalty." if not is_zh else "利用趋势、波动率期限结构和信用利差的带有跳跃惩罚的多因子概率模型。")

    with st.container(border=True):
        with st.spinner("Compiling Macro Microstructure Data..." if not is_zh else "正在编译宏观微观结构数据..."):
            try:
                # 1. Fetch Institutional Proxies (Trend, Credit, Volatility)
                macro_tickers = ['SPY', 'JNK', 'IEF', '^VIX', '^VIX3M']
                macro_df = yf.download(macro_tickers, period="2y", progress=False)['Close'].ffill()

                if not macro_df.empty and len(macro_df.columns) >= 4:
                    # --- Pillar 1: Trend (Baseline Gravity) ---
                    spy = macro_df['SPY']
                    spy_sma200 = spy.rolling(200).mean()
                    trend_raw = (spy - spy_sma200) / spy_sma200

                    # --- Pillar 2: Credit Spreads (The Truth Serum) ---
                    # High Yield (JNK) vs Treasury (IEF)
                    credit_ratio = macro_df['JNK'] / macro_df['IEF']
                    credit_sma50 = credit_ratio.rolling(50).mean()
                    credit_raw = (credit_ratio - credit_sma50) / credit_sma50

                    # --- Pillar 3: Implied Volatility (The Catalyst) ---
                    # Term Structure: VIX (1 month) vs VIX3M (3 months)
                    # If VIX > VIX3M, curve is inverted (Backwardation = Panic)
                    vix = macro_df['^VIX']
                    vix3m = macro_df['^VIX3M'] if '^VIX3M' in macro_df.columns else macro_df['^VIX'].rolling(63).mean()
                    vol_raw = (vix3m - vix) / vix3m # Positive = Contango (Safe), Negative = Backwardation (Fear)

                    # --- THE SJM MATH ENGINE ---
                    # Standardize features (Z-Score) so they carry equal mathematical weight
                    def z_score(series): return (series - series.mean()) / series.std()

                    z_trend = z_score(trend_raw)
                    z_credit = z_score(credit_raw)
                    z_vol = z_score(vol_raw)

                    # Ensemble Weighted Score (Vol gets highest weight per Wolfe Research)
                    ensemble_z = (0.4 * z_vol) + (0.3 * z_trend) + (0.3 * z_credit)

                    # Sigmoid Transformation to output a 0.0 to 1.0 Continuous Probability
                    bull_prob = 1 / (1 + np.exp(-ensemble_z))

                    current_prob = bull_prob.iloc[-1]
                    prev_prob_5d = bull_prob.iloc[-5] # 1-week lookback for the Jump Penalty
                    prob_delta = current_prob - prev_prob_5d

                    # Global state for beta scaling
                    st.session_state['target_beta'] = current_prob

                    # --- UI ROUTING: TOP METRICS ---
                    # We changed from 4 columns to 5 columns to fit the VIX
                    col_prob, col_beta, col_credit, col_vix, col_circuit = st.columns([1, 1, 1, 1, 1.3])

                    col_prob.metric(
                        "Bull Regime Prob" if not is_zh else "牛市状态概率",
                        f"{current_prob*100:.1f}%",
                        f"{prob_delta*100:+.1f}% (5D)",
                        delta_color="normal" if prob_delta > 0 else "inverse"
                    )

                    col_beta.metric(
                        "Target US Beta" if not is_zh else "目标美国 Beta",
                        f"{current_prob:.2f}",
                        "Page 2 Input" if not is_zh else "对冲器输入",
                        delta_color="off"
                    )

                    current_credit = credit_ratio.iloc[-1]
                    credit_sma_val = credit_sma50.iloc[-1]
                    credit_pct_diff = (current_credit / credit_sma_val) - 1

                    col_credit.metric(
                        "Credit (JNK/IEF)" if not is_zh else "信用压力 (JNK/IEF)",
                        f"{current_credit:.2f}",
                        f"{credit_pct_diff*100:+.2f}% vs 50-SMA",
                        delta_color="normal" if credit_pct_diff > 0 else "inverse"
                    )

                    # --- NEW VIX WIDGET ---
                    vix_current = vix.iloc[-1]
                    vix_delta = vix_current - vix.iloc[-2]
                    col_vix.metric(
                        "VIX Index" if not is_zh else "VIX 恐慌指数",
                        f"{vix_current:.2f}",
                        f"{vix_delta:+.2f} (1D)",
                        delta_color="inverse" # VIX going up is bad (red), going down is good (green)
                    )

                    # --- THE JUMP PENALTY CIRCUIT BREAKER ---
                    jump_threshold = 0.15

                    if abs(prob_delta) < jump_threshold:
                        breaker_status = "HOLD"
                        breaker_color = "#E2E8F0" # Neutral Gray
                    elif prob_delta >= jump_threshold:
                        breaker_status = "BULL SHIFT"
                        breaker_color = "#00C853" # Neon Green
                    else:
                        breaker_status = "BEAR SHIFT"
                        breaker_color = "#FF3B30" # Neon Red

                    col_circuit.markdown(f"**Circuit Breaker:**" if not is_zh else f"**熔断器状态:**")
                    col_circuit.markdown(f"<h3 style='color: {breaker_color}; margin-top: 0px;'>{breaker_status}</h3>", unsafe_allow_html=True)

                    # --- FACTOR ROTATION ALERTS ---
                    if current_prob >= 0.70:
                        factor_text = "🟢 **High Bull Probability:** Maximize exposure to **Momentum** and **Small Cap** factors. Risk-seeking behavior is mathematically rewarded here."
                    elif current_prob >= 0.40:
                        factor_text = "🟡 **Transitioning / Neutral:** Market is unsure. Rotate capital into **Quality**, **Minimum Volatility**, and large-cap equities."
                    else:
                        factor_text = "🔴 **Deep Bear Probability:** Risk-Off. Rotate into **Defensive Assets** (Gold, TLT, Utilities). Execute volatility-based hedges."

                    st.info(f"**Factor Protocol:** {factor_text}" if not is_zh else f"**因子协议:** {factor_text}")

                    # --- REGIME PROBABILITY CHART ---
                    fig_regime = go.Figure()
                    plot_prob = bull_prob.tail(252) * 100
                    fill_color = 'rgba(0, 200, 83, 0.2)' if current_prob >= 0.5 else 'rgba(255, 59, 48, 0.2)'
                    line_color = '#00C853' if current_prob >= 0.5 else '#FF3B30'

                    fig_regime.add_trace(go.Scatter(
                        x=plot_prob.index, y=plot_prob.values, mode='lines', name='Bull Probability',
                        line=dict(color=line_color, width=2), fill='tozeroy', fillcolor=fill_color
                    ))
                    fig_regime.add_hline(y=50, line_width=1.5, line_dash="dash", line_color="#E2E8F0", opacity=0.5)
                    fig_regime.update_layout(
                        template="plotly_dark", height=200, margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(range=[0, 100], ticksuffix="%"), xaxis=dict(showgrid=False), hovermode='x unified'
                    )
                    st.plotly_chart(fig_regime, use_container_width=True, config={'displayModeBar': False})

                    # --- NEW VIX TERM STRUCTURE CHART ---
                    fig_vix = go.Figure()
                    plot_vix = vix.tail(252)
                    plot_vix3m = vix3m.tail(252)

                    fig_vix.add_trace(go.Scatter(x=plot_vix.index, y=plot_vix.values, mode='lines', name='VIX (1M)', line=dict(color='#FF3B30', width=1.5)))
                    fig_vix.add_trace(go.Scatter(x=plot_vix3m.index, y=plot_vix3m.values, mode='lines', name='VIX3M (3M)', line=dict(color='#00C853', width=1.5)))

                    fig_vix.update_layout(
                        template="plotly_dark", height=180, margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False), hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                    )
                    st.plotly_chart(fig_vix, use_container_width=True, config={'displayModeBar': False})

                    # --- EDUCATIONAL EXPANDER ---
                    with st.expander("📖 Beginner's Guide: SJM, Credit & VIX" if not is_zh else "📖 新手指南: SJM, 信用利差 与 VIX"):
                        if not is_zh:
                            st.markdown("""
                            **1. The Target US Beta Cap (How to use it)**
                            * If the Probability is **34%**, it means your **Target US Macro Beta** should be **0.34**.
                            * If your portfolio's live beta is 0.63, go to **Page 2 (Hedging)** and input 0.34 as your target.

                            **2. The JNK/IEF Credit Spread (The Truth Serum)**
                            * 🟢 **Green Delta (> 0% vs SMA):** Credit is flowing easily. Bullish for equities.
                            * 🔴 **Red Delta (< 0% vs SMA):** Liquidity is freezing. Structural bear market is beginning.

                            **3. VIX Term Structure (The Catalyst)**
                            Look at the bottom chart comparing the 1-Month VIX (Red) to the 3-Month VIX (Green).
                            * 🟢 **Contango (Normal):** The Green line is ABOVE the Red line. The market is calm and expects future risk to be higher than present risk.
                            * 🔴 **Backwardation (Panic):** The Red line spikes ABOVE the Green line. The market is in active panic mode. Do not buy the dip until the Red line drops back below the Green line.
                            """)
                        else:
                            st.markdown("""
                            **1. 目标美国 Beta 上限 (如何使用)**
                            * 如果概率为 **34%**，则意味着您的 **目标美国宏观 Beta** 应为 **0.34**。
                            * 转至 **页面 2 (对冲)** 并将 0.34 作为目标输入以压缩风险。

                            **2. JNK/IEF 信用利差 (吐真剂)**
                            * 🟢 **绿色增量 (> 均线 0%):** 信贷流动顺畅。对股市极为利好。
                            * 🔴 **红色增量 (< 均线 0%):** 流动性正在冻结。结构性熊市正在开始。

                            **3. VIX 期限结构 (催化剂)**
                            观察底部比较 1 个月 VIX (红色) 和 3 个月 VIX (绿色) 的图表。
                            * 🟢 **升水 (Contango - 正常):** 绿线在红线之上。市场平静。
                            * 🔴 **贴水 (Backwardation - 恐慌):** 红线飙升至绿线之上。市场处于极度恐慌状态。在红线回落至绿线下方之前，请勿逢低买入。
                            """)

                else:
                    st.warning("Failed to download Institutional Macro Proxies." if not is_zh else "下载机构宏观代理数据失败。")
            except Exception as e:
                st.error(f"Failed to calculate Regime Probability: {e}" if not is_zh else f"计算状态概率失败: {e}")

    # ==========================================
    # 5.3 MASTER QUANT FEATURE MATRIX & ASSET FILTER
    # ==========================================
    st.markdown("---")
    st.markdown("### 🧮 Master Quant Matrix & Asset Filter" if not is_zh else "### 🧮 主量化矩阵与资产过滤器")
    st.caption("Check/Uncheck assets here to filter them out of the Heatmap below." if not is_zh else "在此处勾选/取消勾选资产，以在下方的热力图中过滤它们。")

    with st.container(border=True):
        feature_data = []
        for _, row in df_agg.iterrows():
            ticker = row['Ticker']
            if row['Category'] == 'Cash': continue

            yf_t = 'GC=F' if ticker == 'Physical Gold' else ticker.replace('BRK.B', 'BRK-B')
            if yf_t in usd_hist.columns:
                prices = usd_hist[yf_t].dropna()
                if len(prices) > 63:
                    close_price = prices.iloc[-1]
                    ret_1d = (prices.iloc[-1] / prices.iloc[-2] - 1) * 100
                    ret_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
                    ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
                    ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100

                    # 21-day annualized historical volatility
                    daily_rets = prices.pct_change().tail(21).dropna()
                    vol_30d = daily_rets.std() * np.sqrt(252) * 100

                    feature_data.append({
                        "Track": True,
                        "Asset": ticker,
                        "Category": row['Category'],
                        "Price": close_price,
                        "1D_Ret": ret_1d,
                        "1W_Mom": ret_1w,
                        "1M_Mom": ret_1m,
                        "3M_Mom": ret_3m,
                        "30D_Vol": vol_30d
                    })

        df_features = pd.DataFrame(feature_data)
        if not df_features.empty:
            edited_features = st.data_editor(
                df_features,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Track": st.column_config.CheckboxColumn("👀", help="Uncheck to hide from Heatmap"),
                    "Asset": st.column_config.TextColumn("Ticker", disabled=True),
                    "Category": st.column_config.TextColumn("Class", disabled=True),
                    "Price": st.column_config.NumberColumn("Close", format="$%.2f", disabled=True),
                    "1D_Ret": st.column_config.NumberColumn("1D Ret", format="%+.2f%%", disabled=True),
                    "1W_Mom": st.column_config.ProgressColumn("1W Mom", format="%+.2f%%", min_value=-15, max_value=15),
                    "1M_Mom": st.column_config.ProgressColumn("1M Mom", format="%+.2f%%", min_value=-30, max_value=30),
                    "3M_Mom": st.column_config.ProgressColumn("3M Mom", format="%+.2f%%", min_value=-50, max_value=50),
                    "30D_Vol": st.column_config.NumberColumn("30D Vol (Ann.)", format="%.2f%%", disabled=True)
                }
            )
            # Extract checked tickers to feed the Heatmap
            active_tickers = edited_features[edited_features['Track'] == True]['Asset'].tolist()
        else:
            active_tickers = []
            st.info("Not enough historical data to generate the Qlib Feature Matrix.")

        # --- QLIB MATRIX EDUCATIONAL EXPANDER ---
        with st.expander("📖 Beginner's Guide: Momentum & Volatility" if not is_zh else "📖 新手指南: 动量与波动率"):
            if not is_zh:
                st.markdown("""
                **1. Momentum (The Direction)**
                Momentum simply measures how much an asset has moved over a specific timeframe (1 Week, 1 Month, etc.).
                * 🔥 **High (Hot):** Greater than **+10%**. The asset is surging rapidly.
                * 📈 **Positive (Healthy):** Between **0% and +10%**. Steady, normal growth.
                * 🩸 **Negative (Bleeding):** Less than **0%**. The asset is losing value.

                **2. Volatility (The Ride)**
                Annualized Volatility (30D) measures how violently the price swings up and down.
                * 🟢 **Low (Calm):** **< 15%**. Typical for stable index funds (like SPY) or defensive stocks. Safe for large positions.
                * 🟡 **Moderate (Normal):** **15% to 25%**. Standard behavior for individual stocks (like AAPL).
                * 🔴 **High (Wild):** **> 25%**. Typical for high-growth Tech (TSLA, NVDA) and Crypto. High risk of sudden, violent drops.

                **How to read the grid like a Pro:**
                1. 🏆 **The Holy Grail (Steady Winners):** Positive Momentum (> 0%) + Low Volatility (< 15%). These are your core investments. They grind upward quietly without giving you a heart attack.
                2. ⚠️ **The Danger Zone (Boom or Bust):** High Momentum (> 10%) + High Volatility (> 25%). These are explosive breakouts. They make money fast, but can crash just as quickly. Book profits early.
                3. 🔪 **The Falling Knife:** Negative Momentum (< 0%) + High Volatility (> 25%). The asset is crashing violently. Do not buy this hoping for a quick bounce; wait for the volatility to drop first.
                """)
            else:
                st.markdown("""
                **1. 动量 (方向)**
                动量简单地衡量了资产在特定时间（1周、1个月等）内的价格变动幅度。
                * 🔥 **高 (火热):** 大于 **+10%**。资产正在快速飙升。
                * 📈 **正向 (健康):** 在 **0% 到 +10%** 之间。稳步、正常的增长。
                * 🩸 **负向 (流血):** 小于 **0%**。资产正在贬值。

                **2. 波动率 (颠簸程度)**
                年化波动率 (30天) 衡量价格上下波动的剧烈程度。
                * 🟢 **低 (平静):** **< 15%**。常见于稳定的指数基金 (如 SPY) 或防御性股票。适合大资金安全配置。
                * 🟡 **中等 (正常):** **15% 到 25%**。个股的标准表现 (如 AAPL)。
                * 🔴 **高 (狂野):** **> 25%**。常见于高增长科技股 (TSLA, NVDA) 和加密货币。存在突然暴跌的高风险。

                **如何像专业人士一样解读网格:**
                1. 🏆 **“圣杯” (稳健赢家):** 正动量 (> 0%) + 低波动率 (< 15%)。这些是您的核心投资。它们悄然向上攀升，不会让您担惊受怕。
                2. ⚠️ **危险区 (暴涨或暴跌):** 高动量 (> 10%) + 高波动率 (> 25%)。这是爆发性的突破。赚钱快，但崩盘也一样快。应尽早锁定利润。
                3. 🔪 **掉落的飞刀:** 负动量 (< 0%) + 高波动率 (> 25%)。资产正在猛烈崩盘。不要为了博取快速反弹而买入；先等波动率降下来再说。
                """)

    # ==========================================
    # 5.5 PORTFOLIO HEATMAP
    # ==========================================
    st.markdown("---")
    st.markdown("### 🗺️ Portfolio Heatmap" if not is_zh else "### 🗺️ 投资组合热力图")
    st.caption("Visualizing absolute exposure and 1-Day momentum." if not is_zh else "可视化绝对风险敞口和 1 日动量。")

    with st.container(border=True):
        if active_tickers:
            df_heat = df_agg[df_agg['Ticker'].isin(active_tickers)].copy()

            # --- THE SURGICAL FIX: PREVENT ZERO-DIVISION ---
            # Treemaps cannot mathematically process zero or negative box sizes.
            df_heat = df_heat[df_heat['Current_USD'] > 0]

            if not df_heat.empty:
                day_returns = {}
                for tick_sym in df_heat['Ticker'].unique():
                    yf_t = 'GC=F' if tick_sym == 'Physical Gold' else tick_sym.replace('BRK.B', 'BRK-B')
                    if yf_t in usd_hist.columns and len(usd_hist[yf_t]) >= 2:
                        day_returns[tick_sym] = (usd_hist[yf_t].iloc[-1] / usd_hist[yf_t].iloc[-2]) - 1
                    else:
                        day_returns[tick_sym] = 0.0

                df_heat['1D_Ret'] = df_heat['Ticker'].map(day_returns)
                df_heat['Ret_Text'] = (df_heat['1D_Ret'] * 100).apply(lambda x: f"{x:+.2f}%")

                clean_colors = [
                    [0.0, '#FF3B30'],
                    [0.48, '#5C0000'],
                    [0.5, '#1A1A1A'],
                    [0.52, '#003300'],
                    [1.0, '#00C853']
                ]

                fig_tree = px.treemap(
                    df_heat, path=[px.Constant("Portfolio" if not is_zh else "投资组合"), 'Category', 'Ticker'],
                    values='Current_USD', color='1D_Ret',
                    custom_data=['Ret_Text', 'Current_USD'],
                    color_continuous_scale=clean_colors, color_continuous_midpoint=0
                )

                fig_tree.update_traces(
                    texttemplate="<b>%{label}</b><br>%{customdata[0]}",
                    textfont=dict(color="#FFFFFF", size=15, family="Inter, sans-serif"),
                    root_color="#0E1117",
                    marker=dict(line=dict(color="#0E1117", width=3)),
                    hovertemplate="<b>%{label}</b><br>Value: $%{customdata[1]:,.0f}<br>1D Return: %{customdata[0]}<extra></extra>"
                )

                fig_tree.update_layout(
                    template="plotly_dark", height=450, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_tree, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("The selected assets currently have $0.00 value and cannot be mapped." if not is_zh else "所选资产当前价值为 $0.00，无法映射。")
        else:
            st.warning("All assets are currently hidden. Check an asset in the Master Quant Matrix to render the Heatmap." if not is_zh else "所有资产当前已隐藏。在主量化矩阵中勾选资产以渲染热力图。")

    # --- 6. THE INTERACTIVE TABLE & DYNAMIC PIE CHART ---
    st.markdown(t["alloc_tracker"])
    ch = defaults.get("custom_headers", {})

    # We silently use English defaults for the dataframe logic, but display the UI based on language
    h_cat = ch.get("h_cat", "Category" if not is_zh else "类别")
    h_val = ch.get("h_val", "Current Value ($)" if not is_zh else "当前价值 ($)")
    h_act = ch.get("h_act", "Actual %" if not is_zh else "实际 %")
    h_tgt = ch.get("h_tgt", "Target %" if not is_zh else "目标 %")
    h_cor = ch.get("h_cor", f"Corr to {benchmark}" if not is_zh else f"与 {benchmark} 相关性")
    h_bet = ch.get("h_bet", f"Beta to {benchmark}" if not is_zh else f"贝塔 ({benchmark})")
    h_pct = ch.get("h_pct", "P/L (%)" if not is_zh else "盈亏 (%)")
    h_pnl = ch.get("h_pnl", "Unrealized P/L ($)" if not is_zh else "未实现盈亏 ($)")

    col_table, col_pie = st.columns([2.2, 1.2]) # Gives the pie chart more breathing room
    with col_table:
        with st.container(border=True):
            display_df = df_agg[['Category', 'Ticker', 'Current_USD', 'Actual_Weight_%', 'Target_Weight_%', f'Corr_to_{benchmark}', f'Beta_to_{benchmark}', 'PnL_Pct', 'PnL_USD']].copy()

            edited_df = st.data_editor(
                display_df, use_container_width=True, hide_index=True,
                column_config={
                    "Category": st.column_config.SelectboxColumn(h_cat, options=["Defensive", "Core Foundation", "Core Compounding", "Aggressive", "Cash", "Hedge"], required=True),
                    "Current_USD": st.column_config.NumberColumn(h_val, format="$ %.2f"),
                    "Actual_Weight_%": st.column_config.ProgressColumn(h_act, format="%.1f%%", min_value=0, max_value=100),
                    "Target_Weight_%": st.column_config.NumberColumn(h_tgt, format="%.1f%%", min_value=0, max_value=100),
                    f'Corr_to_{benchmark}': st.column_config.NumberColumn(h_cor, format="%.2f"),
                    f'Beta_to_{benchmark}': st.column_config.NumberColumn(h_bet, format="%.2f"),
                    "PnL_Pct": st.column_config.NumberColumn(h_pct, format="%+.2f %%"),
                    "PnL_USD": st.column_config.NumberColumn(h_pnl, format="$ %+.2f")
                }
            )

    with col_pie:
        with st.container(border=True):
            cat_df = edited_df.groupby('Category')['Current_USD'].sum().reset_index()
            # Classy Dark-Theme Palette
            classy_colors = ['#7C3AED', '#4F46E5', '#06B6D4', '#8B5CF6', '#3B82F6', '#64748B']

            fig_pie = px.pie(
                cat_df, values='Current_USD', names='Category', hole=0.55,
                color_discrete_sequence=classy_colors
            )

            # 1. TRACES: Show ONLY percentages inside, force Inter font
            fig_pie.update_traces(
                textinfo="percent",
                textposition='inside',
                insidetextorientation='horizontal',
                textfont=dict(color='#FFFFFF', size=14, family="Inter, sans-serif"),
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>"
            )

            # 2. LAYOUT: Turn on horizontal legend, push it to the bottom
            fig_pie.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(t=50, b=80, l=10, r=10), # Extra bottom margin for the legend
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Inter, sans-serif", size=12, color="#94A3B8")
                ),
                title=dict(
                    text=t["alloc_pie"],
                    x=0.5, y=0.95,
                    xanchor='center', yanchor='top',
                    font=dict(family="Inter, sans-serif", color="#FFFFFF", size=18)
                )
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    # --- 7. THE TRIPLE-LINE PERFORMANCE CHART & RISK ENGINE ---
    st.markdown("---")
    st.markdown(t["perf_engine"])

    daily_returns = usd_hist.pct_change().dropna()
    port_daily_return = pd.Series(0.0, index=daily_returns.index)
    for _, row in edited_df.iterrows():
        if row['Category'] != 'Cash':
            ticker = row['Ticker']
            weight = row['Actual_Weight_%'] / 100.0
            yf_t = 'GC=F' if ticker == 'Physical Gold' else ticker.replace('BRK.B', 'BRK-B')
            if yf_t in daily_returns.columns: port_daily_return += daily_returns[yf_t] * weight

    port_cum = (1 + port_daily_return).cumprod() - 1
    spy_cum = (1 + daily_returns.get('SPY', 0)).cumprod() - 1
    qqq_cum = (1 + daily_returns.get('QQQ', 0)).cumprod() - 1

    # --- INSTITUTIONAL RISK MATH ---
    rolling_max = (1 + port_cum).cummax()
    drawdown = (1 + port_cum) / rolling_max - 1
    max_dd = drawdown.min()

    # Standard CTA Risk-Free Rate assumption (approx 4.0% currently)
    risk_free_rate = 0.04
    ann_ret = port_daily_return.mean() * 252
    ann_vol = port_daily_return.std() * np.sqrt(252)
    downside_returns = port_daily_return[port_daily_return < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0001

    sharpe_ratio = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    sortino_ratio = (ann_ret - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    calmar_ratio = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    with st.container(border=True):
        # --- RISK METRICS WIDGETS ---
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Sharpe Ratio" if not is_zh else "夏普比率", f"{sharpe_ratio:.2f}", "Risk-Adjusted Return" if not is_zh else "风险调整后收益", delta_color="off")
        rm2.metric("Sortino Ratio" if not is_zh else "索提诺比率", f"{sortino_ratio:.2f}", "Downside Protection" if not is_zh else "下行风险保护", delta_color="off")
        rm3.metric("Calmar Ratio" if not is_zh else "卡玛比率", f"{calmar_ratio:.2f}", "Return vs Max DD" if not is_zh else "收益 vs 最大回撤", delta_color="off")
        rm4.metric("Max Drawdown" if not is_zh else "最大回撤", f"{max_dd*100:.2f}%", "Peak-to-Trough" if not is_zh else "顶至底跌幅", delta_color="inverse")

        st.divider()

        # --- Stacked Charts (Triple Equity Curve + Underwater Drawdown) ---
        fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

        # The Glowing Purple Portfolio Line
        fig_perf.add_trace(go.Scatter(
            x=port_cum.index, y=port_cum.values * 100,
            mode='lines', name='My Portfolio',
            line=dict(color='#A855F7', width=3),
            fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'
        ), row=1, col=1)

        # Muted Benchmarks
        fig_perf.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum.values * 100, mode='lines', name='SPY', line=dict(color='rgba(255, 255, 255, 0.2)', width=2, dash='dot')), row=1, col=1)
        fig_perf.add_trace(go.Scatter(x=qqq_cum.index, y=qqq_cum.values * 100, mode='lines', name='QQQ', line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot')), row=1, col=1)

        # Drawdown - Deep Red
        fig_perf.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            mode='lines', fill='tozeroy', name='Drawdown',
            line=dict(color='#FF3B30', width=1),
            fillcolor='rgba(255, 59, 48, 0.2)'
        ), row=2, col=1)

        fig_perf.update_layout(
            template="plotly_dark", height=500, margin=dict(t=10, b=10, l=10, r=10),
            hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_perf.update_yaxes(ticksuffix="%", row=1, col=1)
        fig_perf.update_yaxes(ticksuffix="%", row=2, col=1)
        st.plotly_chart(fig_perf, use_container_width=True)

        # --- EDUCATIONAL EXPANDER: RISK METRICS ---
        with st.expander("📖 Quant Guide: Equity Curve & Risk Metrics" if not is_zh else "📖 量化指南：净值曲线与风险指标"):
            if not is_zh:
                st.markdown("""
                **1. How the Equity Curve is Calculated**
                This engine reconstructs your portfolio's historical path. It takes the *current* weights of your assets and maps them backward against their daily closing prices. The daily returns are blended and cumulatively compounded. SPY and QQQ are plotted as static benchmarks.

                **2. The Risk Metrics (How to read them like a CTA)**
                * **Sharpe Ratio (Return vs. Total Volatility):** Measures excess return per unit of risk. A Sharpe of 0.30 means you are taking on high volatility for relatively little reward. Hedge funds generally target > 1.0. To fix a low Sharpe, you must either increase returns or (more easily) ruthlessly cut highly volatile assets that aren't performing.
                * **Sortino Ratio (Return vs. Bad Volatility):** Similar to Sharpe, but it only penalizes *downward* price swings. If your Sortino is much higher than your Sharpe, it means your volatility is mostly to the upside (which is good).
                * **Calmar Ratio (Return vs. Drawdown):** Annualized Return divided by Maximum Drawdown. CTAs love this. A Calmar > 1.0 means your annual return exceeds your worst historical drop.
                * **Maximum Drawdown:** The terrifying "peak-to-trough" drop. If this is -40%, you need a +66% return just to break even. Rule #1 of CTA funds: Protect the downside.
                """)
            else:
                st.markdown("""
                **1. 净值曲线是如何计算的？**
                该引擎重构了您投资组合的历史路径。它提取您当前资产的权重，并将其与历史每日收盘价向后映射。每日收益被混合并进行累积复利计算。SPY 和 QQQ 作为静态基准进行对比。

                **2. 风险指标 (如何像 CTA 基金经理一样解读)**
                * **夏普比率 Sharpe (收益 vs 总波动率):** 衡量每单位风险的超额回报。0.30 的夏普意味着您承担了高波动性，但获得的回报相对较少。对冲基金通常目标是 > 1.0。要修复较低的夏普比率，您必须提高收益，或者（更容易的做法）无情地削减表现不佳的高波动性资产。
                * **索提诺比率 Sortino (收益 vs 坏波动率):** 类似于夏普，但它只惩罚*下行*价格波动。如果您的索提诺远高于夏普，这意味着您的波动主要集中在上行（这是好事）。
                * **卡玛比率 Calmar (收益 vs 回撤):** 年化收益率除以最大回撤。CTA 基金非常看重这个指标。卡玛 > 1.0 意味着您的年收益超过了历史上最严重的下跌。
                * **最大回撤 Max Drawdown:** 令人恐惧的“顶至底”跌幅。如果回撤是 -40%，您需要 +66% 的收益才能回本。CTA 基金的第一法则：保护下行风险。
                """)

    # --- 7.5 SYSTEMIC LINKAGE (CORRELATION MATRIX) ---
    st.markdown("---")
    st.markdown("### 🕸️ Systemic Linkage Matrix" if not is_zh else "### 🕸️ 系统性联动矩阵")

    with st.container(border=True):
        # Toggle between Pearson Correlation and Annualized Covariance
        matrix_type = st.radio("Select Matrix Mode:", ["Pearson Correlation", "Annualized Covariance"], horizontal=True)

        if matrix_type == "Pearson Correlation":
            z_data = log_returns.corr()
            zmin, zmax = -1.0, 1.0
            # Stealth Neon Colorscale: Red (Inverse) -> Deep Black (Uncorrelated) -> Green (Highly Correlated)
            colorscale = [[0.0, '#FF3B30'], [0.5, '#09090B'], [1.0, '#00C853']]
            val_format = ".2f"
        else:
            z_data = log_returns.cov() * 252  # Annualized
            # Center covariance dynamically based on max/min to keep 0 at Deep Black
            max_abs = max(abs(z_data.min().min()), abs(z_data.max().max()))
            zmin, zmax = -max_abs, max_abs
            colorscale = [[0.0, '#FF3B30'], [0.5, '#09090B'], [1.0, '#00C853']]
            val_format = ".4f"

        # Build the Heatmap
        fig_matrix = go.Figure(data=go.Heatmap(
            z=z_data.values,
            x=z_data.columns,
            y=z_data.index,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            text=np.round(z_data.values, 4),
            texttemplate="%{text:" + val_format + "}",
            hoverinfo="x+y+z",
            showscale=False # Hide standard colorbar to keep UI clean
        ))

        fig_matrix.update_layout(
            template="plotly_dark",
            height=650,
            margin=dict(t=30, b=30, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_autorange='reversed' # Standard matrix view (diagonal top-left to bottom-right)
        )

        st.plotly_chart(fig_matrix, use_container_width=True)

    # --- 8. GLOBAL MACRO PULSE (NEW) ---
    st.markdown("---")
    st.markdown(t["macro_pulse"])

    # Calculate % Change for Macro Indicators
    def get_macro_delta(ticker):
        try:
            return (hist_data[ticker].iloc[-1] / hist_data[ticker].iloc[-2]) - 1
        except:
            return 0.0

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        with st.container(border=True):
            st.metric("🇺🇸 S&P 500 (SPY)", f"${live_prices.get('SPY', 0):.2f}", f"{get_macro_delta('SPY')*100:+.2f}%")
            st.metric("🇺🇸 NASDAQ (QQQ)", f"${live_prices.get('QQQ', 0):.2f}", f"{get_macro_delta('QQQ')*100:+.2f}%")
            st.metric("🇺🇸 US 20Y+ Bonds (TLT)" if not is_zh else "🇺🇸 美债 20年+ (TLT)", f"${live_prices.get('TLT', 0):.2f}", f"{get_macro_delta('TLT')*100:+.2f}%")

    with m2:
        with st.container(border=True):
            st.metric("🇭🇰 Hang Seng (^HSI)" if not is_zh else "🇭🇰 恒生指数 (^HSI)", f"{live_prices.get('^HSI', 0):,.0f}", f"{get_macro_delta('^HSI')*100:+.2f}%")
            st.metric("🇨🇳 CSI 300" if not is_zh else "🇨🇳 沪深300指数", f"{live_prices.get('000300.SS', 0):,.0f}", f"{get_macro_delta('000300.SS')*100:+.2f}%")
            st.metric("🇨🇳 China Bonds (CBON)" if not is_zh else "🇨🇳 中国债券 (CBON)", f"${live_prices.get('CBON', 0):.2f}", f"{get_macro_delta('CBON')*100:+.2f}%")

    with m3:
        with st.container(border=True):
            st.metric("🇪🇺 Europe (VGK)" if not is_zh else "🇪🇺 欧洲市场 (VGK)", f"${live_prices.get('VGK', 0):.2f}", f"{get_macro_delta('VGK')*100:+.2f}%")
            st.metric("🇯🇵 Japan (EWJ)" if not is_zh else "🇯🇵 日本市场 (EWJ)", f"${live_prices.get('EWJ', 0):.2f}", f"{get_macro_delta('EWJ')*100:+.2f}%")
            st.metric("🇰🇷 South Korea (EWY)" if not is_zh else "🇰🇷 韩国市场 (EWY)", f"${live_prices.get('EWY', 0):.2f}", f"{get_macro_delta('EWY')*100:+.2f}%")

    with m4:
        with st.container(border=True):
            st.metric("🥇 Gold (GC=F)" if not is_zh else "🥇 黄金 (GC=F)", f"${live_prices.get('GC=F', 0):,.1f}", f"{get_macro_delta('GC=F')*100:+.2f}%")
            st.metric("🛢️ WTI Crude (CL=F)" if not is_zh else "🛢️ WTI原油 (CL=F)", f"${live_prices.get('CL=F', 0):.2f}", f"{get_macro_delta('CL=F')*100:+.2f}%")
            st.metric("₿ Bitcoin (BTC-USD)" if not is_zh else "₿ 比特币 (BTC-USD)", f"${live_prices.get('BTC-USD', 0):,.0f}", f"{get_macro_delta('BTC-USD')*100:+.2f}%")

    # --- EXECUTE GLOBAL SAVE & EXPORT ---
    with global_save_container:
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button(t["save_btn"], type="primary", use_container_width=True):
                defaults["t212_cash_eur"] = t212_cash_eur
                defaults["futu_cash_usd"] = futu_cash_usd
                defaults["cny_mutual_fund"] = cny_mutual_fund
                defaults["cny_mutual_fund_pnl"] = cny_mf_pnl
                defaults["cny_gold_grams"] = cny_gold_grams
                defaults["cny_gold_cost"] = cny_gold_cost
                prefs = defaults.get("asset_preferences", {})
                for _, row in edited_df.iterrows(): prefs[row['Ticker']] = {"Category": row['Category'], "Target_Weight_%": row['Target_Weight_%']}
                defaults["asset_preferences"] = prefs
                with open(DEFAULTS_FILE, "w") as f: json.dump(defaults, f)
                st.success("✅ Saved successfully!" if not is_zh else "✅ 成功保存所有设置！")

        with b2:
            # Export clean portfolio data for external quant analysis
            csv_data = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t["export_btn"],
                data=csv_data,
                file_name=f'portfolio_ledger_{datetime.date.today()}.csv',
                mime='text/csv',
                use_container_width=True
            )

else:
    st.warning("No valid unified portfolio found.")
