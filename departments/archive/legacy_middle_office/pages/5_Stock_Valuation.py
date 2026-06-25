import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import os
import json
import sys
from pathlib import Path

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

# --- NEW: GEMINI SDK IMPORT ---
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- 2. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Fundamental Valuation", layout="wide", page_icon="🏦")

from utils.theme import load_css
try:
    load_css()
except:
    pass

# --- LOCALIZATION ENGINE (EN/ZH) ---
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)
is_zh = lang == "中文"

t = {
    "title": "🏦 The Vault: Deep Fundamental Valuation" if not is_zh else "🏦 金库：深度基本面估值",
    "subtitle": "Institutional DCF modeling, peer relative valuation, and FMP data integration." if not is_zh else "机构级现金流折现模型、同行相对估值及 FMP 数据集成。",
    "sb_ctrl": "⚙️ Workspace Controls" if not is_zh else "⚙️ 工作区控制台",
    "sb_api": "🔑 API Credentials" if not is_zh else "🔑 API 凭证",
    "sb_watch": "⭐ My Watchlist" if not is_zh else "⭐ 我的自选股",
    "tick": "Target Ticker" if not is_zh else "目标代码",
    "load_tick": "Load Saved Ticker:" if not is_zh else "加载已保存的代码：",
    "new_search": "-- New Search --" if not is_zh else "-- 新搜索 --",
    "dcf_model": "### Valuation Model" if not is_zh else "### 估值模型",
    "fcf_g": "Standard FCF Growth" if not is_zh else "标准自由现金流增长",
    "rev_m": "Revenue & Margin Expansion" if not is_zh else "收入与利润率扩张",
    "wacc": "Discount Rate (WACC) %" if not is_zh else "折现率 (WACC) %",
    "perp": "Terminal Growth Rate %" if not is_zh else "永续增长率 %",
    "g1": "FCF Growth (Y1-5) %" if not is_zh else "自由现金流增长 (1-5年) %",
    "g2": "FCF Growth (Y6-10) %" if not is_zh else "自由现金流增长 (6-10年) %",
    "rev1": "Rev Growth (Y1-5) %" if not is_zh else "收入增长 (1-5年) %",
    "rev2": "Rev Growth (Y6-10) %" if not is_zh else "收入增长 (6-10年) %",
    "t_margin": "Target FCF Margin (Y10) %" if not is_zh else "目标自由现金流利润率 (第10年) %",
    "tab1": "⚖️ Fundamentals & DCF" if not is_zh else "⚖️ 基本面与DCF估值",
    "tab2": "📈 Technicals" if not is_zh else "📈 技术分析",
    "tab3": "🎯 Analyst Targets" if not is_zh else "🎯 分析师目标价",
    "tab4": "👯 Peer Comparison" if not is_zh else "👯 同行比较",
    "tab5": "🧠 AI Analyst" if not is_zh else "🧠 AI 分析师"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==========================================
# 0. WATCHLIST & MEMORY SYSTEM
# ==========================================
WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return []

def save_to_watchlist(ticker):
    current = load_watchlist()
    if ticker not in current:
        current.append(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(current, f)
        st.toast(f"⭐ {ticker} added to Watchlist!")

def remove_from_watchlist(ticker):
    current = load_watchlist()
    if ticker in current:
        current.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(current, f)
        st.toast(f"❌ {ticker} removed from Watchlist.")

# --- API KEY MEMORY (Reusing the system from Page 2) ---
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.json")

def load_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {"fmp_key": "", "gemini_key": ""}

def save_keys(fmp_key, gemini_key):
    with open(API_KEYS_FILE, "w") as f:
        json.dump({"fmp_key": fmp_key, "gemini_key": gemini_key}, f)

def make_fmp_adapter(api_key):
    if not api_key or FMPDataAdapter is None:
        return None
    return FMPDataAdapter(api_key=api_key)

def fetch_fmp_json(api_key, endpoint, *, stable=False, params=None, suppress_error_messages=True):
    adapter = make_fmp_adapter(api_key)
    if adapter is None:
        return []

    try:
        json_data = adapter.get_json(endpoint, stable=stable, params=params)
    except Exception:
        return []

    if suppress_error_messages and isinstance(json_data, dict) and "Error Message" in json_data:
        return []

    return json_data

# ==========================================
# 1. ROBUST YFINANCE DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker_symbol, api_key):
    ticker = yf.Ticker(ticker_symbol)
    data, financials, historicals, mandates = {}, {}, {}, {}

    # Helper to safely handle FMP returning 'None' instead of 0
    def safe_num(val):
        return float(val) if val is not None else 0.0

    def calc_cagr(series):
        if len(series) < 2: return 0.0
        start, end = series.iloc[0], series.iloc[-1]
        if start <= 0 or end <= 0: return 0.0
        return (end / start) ** (1 / (len(series) - 1)) - 1

    def get_latest(df, possible_cols):
        """Hunts down financial metrics even if Yahoo changes the column names."""
        for col in possible_cols:
            if col in df.columns:
                series = df[col].dropna()
                if not series.empty:
                    return series.iloc[-1]
        return 0.0

    try:
        # Check FMP First, Fall back to Yfinance if empty
        profile_fmp = fetch_fmp_json(api_key, f"profile/{ticker_symbol}")

        info = ticker.info

        # Prefer FMP for basic quote data if possible, but YF works perfectly fine
        data['price'] = safe_num(profile_fmp[0].get('price')) if profile_fmp else info.get('currentPrice', 0)
        data['market_cap'] = safe_num(profile_fmp[0].get('mktCap')) if profile_fmp else info.get('marketCap', 0)
        data['sector'] = profile_fmp[0].get('sector', 'Unknown') if profile_fmp else info.get('sector', 'Unknown')

        data['pe'] = info.get('trailingPE', 999)
        data['peg'] = info.get('pegRatio', 999)
        data['shares'] = info.get('sharesOutstanding', 1)
        data['total_cash'] = info.get('totalCash', 0)
        data['total_debt'] = info.get('totalDebt', 0)

        # Financial Statements (Annual & Quarterly)
        inc = ticker.financials.T.sort_index() if hasattr(ticker, 'financials') else pd.DataFrame()
        bal = ticker.balance_sheet.T.sort_index() if hasattr(ticker, 'balance_sheet') else pd.DataFrame()
        cf = ticker.cashflow.T.sort_index() if hasattr(ticker, 'cashflow') else pd.DataFrame()

        q_inc = ticker.quarterly_financials.T.sort_index() if hasattr(ticker, 'quarterly_financials') else pd.DataFrame()
        q_bal = ticker.quarterly_balance_sheet.T.sort_index() if hasattr(ticker, 'quarterly_balance_sheet') else pd.DataFrame()
        q_cf = ticker.quarterly_cashflow.T.sort_index() if hasattr(ticker, 'quarterly_cashflow') else pd.DataFrame()

        # --- QUARTERLY FUNDAMENTAL QUICK CHECK ---
        data['q_rev'] = get_latest(q_inc, ['Total Revenue', 'Operating Revenue'])
        data['q_gross'] = get_latest(q_inc, ['Gross Profit'])
        data['q_op_inc'] = get_latest(q_inc, ['Operating Income', 'EBIT', 'Ebit'])
        data['q_curr_assets'] = get_latest(q_bal, ['Total Current Assets', 'Current Assets'])
        data['q_curr_liab'] = get_latest(q_bal, ['Total Current Liabilities', 'Current Liabilities'])
        data['q_inventory'] = get_latest(q_bal, ['Inventory'])
        data['q_total_debt'] = get_latest(q_bal, ['Total Debt', 'Long Term Debt'])
        data['q_equity'] = get_latest(q_bal, ['Stockholders Equity', 'Total Stockholder Equity'])

        q_ocf = get_latest(q_cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        q_capex = get_latest(q_cf, ['Capital Expenditure'])
        data['q_fcf'] = q_ocf - abs(q_capex)

        # --- ANNUAL DATA FOR DCF & ROCE ---
        rev = inc.get('Total Revenue', pd.Series(dtype='float64'))
        fcf_series = cf.get('Free Cash Flow', pd.Series(dtype='float64'))

        data['ttm_revenue'] = rev.iloc[-1] if not rev.empty else 0
        data['fcf_ttm'] = fcf_series.iloc[-1] if not fcf_series.empty else (get_latest(cf, ['Operating Cash Flow']) - abs(get_latest(cf, ['Capital Expenditure'])))

        ebit = get_latest(inc, ['Operating Income', 'EBIT', 'Ebit'])
        t_assets = get_latest(bal, ['Total Assets'])
        c_liab = get_latest(bal, ['Total Current Liabilities', 'Current Liabilities'])
        capital_employed = t_assets - c_liab
        data['roce'] = ebit / capital_employed if capital_employed > 0 else 0

        st.session_state['auto_rev_cagr'] = calc_cagr(rev.tail(4)) * 100 if not rev.empty else 12.0
        st.session_state['auto_fcf_cagr'] = calc_cagr(fcf_series.tail(4)) * 100 if not fcf_series.empty else 15.0

        # --- 10-POINT CHECKLIST ---
        net_inc = inc.get('Net Income', pd.Series(dtype='float64'))
        curr_assets = bal.get('Total Current Assets', pd.Series(dtype='float64'))
        curr_liab = bal.get('Total Current Liabilities', pd.Series(dtype='float64'))
        lt_debt = bal.get('Long Term Debt', bal.get('Total Debt', pd.Series(dtype='float64')))
        equity = bal.get('Stockholders Equity', bal.get('Total Stockholder Equity', pd.Series(dtype='float64')))
        shares_hist = inc.get('Basic Average Shares', inc.get('Diluted Average Shares', pd.Series(dtype='float64')))
        ocf_hist = cf.get('Operating Cash Flow', pd.Series(dtype='float64'))
        capex_hist = cf.get('Capital Expenditure', pd.Series(dtype='float64'))
        divs_hist = cf.get('Cash Dividends Paid', pd.Series(dtype='float64'))

        pe, peg = data['pe'], data['peg']
        mandates['#1 P/E<25 & PEG<1'] = (0 < pe < 25) and (0 < peg < 1)
        mandates['#2 Revenue growth +'] = calc_cagr(rev) > 0 if not rev.empty else False
        mandates['#3 Op. profit growth +'] = calc_cagr(inc.get('Operating Income', pd.Series(dtype='float64'))) > 0
        mandates['#4 Net profit growth +'] = calc_cagr(net_inc) > 0 if not net_inc.empty else False
        mandates['#5 Current assets > liabilities'] = (curr_assets.iloc[-1] > curr_liab.iloc[-1]) if not curr_assets.empty else False
        mandates['#6 LT Debt/Net Inc <4'] = (lt_debt.iloc[-1] / net_inc.iloc[-1] < 4) if (not lt_debt.empty and not net_inc.empty and net_inc.iloc[-1] > 0) else False
        mandates['#7 Equity growing'] = calc_cagr(equity) > 0 if not equity.empty else False
        mandates['#8 Shares decreasing'] = calc_cagr(shares_hist) < 0 if len(shares_hist) > 1 else False
        if not ocf_hist.empty and not capex_hist.empty:
            d_val = abs(divs_hist.iloc[-1]) if not divs_hist.empty else 0
            mandates['#9 OCF covers capex & payout'] = ocf_hist.iloc[-1] > (abs(capex_hist.iloc[-1]) + d_val)
        else: mandates['#9 OCF covers capex & payout'] = False
        mandates['#10 FCF growth +'] = calc_cagr(fcf_series) > 0 if not fcf_series.empty else False

        # --- TECHNICALS ---
        hist = ticker.history(period="1y")
        if not hist.empty:
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            std_20 = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['SMA_20'] + (std_20 * 2)
            hist['BB_Lower'] = hist['SMA_20'] - (std_20 * 2)
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = ema_12 - ema_26
            hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACD_Hist'] = hist['MACD'] - hist['Signal']
            data['hist_df'] = hist

    except Exception as e:
        print(f"Fetch error: {e}")

    return data, financials, historicals, mandates

# --- . SIDEBAR CONTROLS ---
with st.sidebar:
    st.header(t["sb_ctrl"])

    st.subheader(t["sb_api"])
    settings = load_settings() if load_settings else None
    default_fmp = st.secrets.get("FMP_KEY", "") if hasattr(st, "secrets") else ""
    default_fmp = default_fmp or st.secrets.get("FMP_API_KEY", "")
    default_fmp = default_fmp or (settings.fmp_api_key if settings else "")
    default_gemini = st.secrets.get("GEMINI_KEY", "") if hasattr(st, "secrets") else ""
    default_gemini = default_gemini or st.secrets.get("GEMINI_API_KEY", "")
    default_gemini = default_gemini or (settings.gemini_api_key if settings else "")

    raw_fmp = st.text_input("FMP API Key", value=default_fmp, type="password")
    raw_gemini = st.text_input("Gemini API Key", value=default_gemini, type="password")

    # SURGICAL PATCH: The String Sanitizer
    # Automatically strips invisible trailing spaces, newlines, and accidental quotes
    fmp_key = str(raw_fmp).strip().replace('"', '').replace("'", "")
    gemini_key = str(raw_gemini).strip().replace('"', '').replace("'", "")
    st.markdown("---")

    st.subheader(t["sb_watch"])
    watchlist = load_watchlist()

    if watchlist:
        selected_watch = st.selectbox(t["load_tick"], [t["new_search"]] + watchlist)
        target_ticker = st.text_input(t["tick"], value="AAPL").upper() if selected_watch == t["new_search"] else selected_watch
    else:
        target_ticker = st.text_input(t["tick"], value="AAPL").upper()

    if target_ticker:
        if target_ticker in watchlist:
            if st.button(f"❌ Remove {target_ticker}" if not is_zh else f"❌ 移除 {target_ticker}", use_container_width=True):
                remove_from_watchlist(target_ticker); st.rerun()
        else:
            if st.button(f"⭐ Save {target_ticker}" if not is_zh else f"⭐ 保存 {target_ticker}", use_container_width=True):
                save_to_watchlist(target_ticker); st.rerun()

    st.markdown("---")

    # Fetch Data & Setup Baselines (No sliders here!)
    data, financials, historicals, mandates = {}, {}, {}, {}
    if target_ticker and fmp_key:
        data, financials, historicals, mandates = fetch_fundamental_data(target_ticker, fmp_key)

    # 3. DCF Parameters (Your original logic)
    data, financials, historicals, mandates = {}, {}, {}, {}

    if target_ticker and fmp_key:
        data, financials, historicals, mandates = fetch_fundamental_data(target_ticker, fmp_key)
        default_fcf_g = st.session_state.get('auto_fcf_cagr', 15.0)
        default_rev_g = st.session_state.get('auto_rev_cagr', 12.0)
    else:
        default_fcf_g, default_rev_g = 15.0, 12.0

    st.markdown("### Valuation Model")
    model_choice = st.radio("Select Cash Flow Engine:", ["Standard FCF Growth", "Revenue & Margin Expansion"])

    st.markdown("### Cost of Capital (%)")
    wacc_pct = st.number_input("Discount Rate (WACC) %", 1.0, 50.0, 10.0, 0.5, format="%.2f")
    perp_g_pct = st.number_input("Terminal Growth Rate %", -5.0, 10.0, 2.50, 0.1, format="%.2f")

    st.markdown("### Growth Assumptions (%)")
    if model_choice == "Standard FCF Growth":
        g1_pct = st.number_input("FCF Growth (Y1-5) %", -50.0, 200.0, float(default_fcf_g), 1.0, format="%.2f")
        g2_pct = st.number_input("FCF Growth (Y6-10) %", -50.0, 200.0, float(g1_pct * 0.7), 1.0, format="%.2f")
        dcf_inputs = {"type": "Standard", "g1": g1_pct/100, "g2": g2_pct/100, "wacc": wacc_pct/100, "perp_g": perp_g_pct/100}
    else:
        rev_g1_pct = st.number_input("Rev Growth (Y1-5) %", -50.0, 200.0, float(default_rev_g), 1.0, format="%.2f")
        rev_g2_pct = st.number_input("Rev Growth (Y6-10) %", -50.0, 200.0, float(rev_g1_pct * 0.7), 1.0, format="%.2f")
        target_margin_pct = st.number_input("Target FCF Margin (Y10) %", 1.0, 100.0, 25.0, 1.0, format="%.2f")
        dcf_inputs = {"type": "Margin", "rev_g1": rev_g1_pct/100, "rev_g2": rev_g2_pct/100, "margin": target_margin_pct/100, "wacc": wacc_pct/100, "perp_g": perp_g_pct/100}

# ==========================================
# 3. DASHBOARD EXECUTION WRAPPER
# ==========================================
if target_ticker and data.get('price'):

    # ==========================================
    # 4. MAIN DASHBOARD UI & PORTFOLIO LEDGER
    # ==========================================

    # --- NEW: PORTFOLIO AUTO-FILTER LEDGER ---
    df_port = st.session_state.get('unified_portfolio')
    if df_port is not None and not df_port.empty:
        with st.expander("📂 My Long-Term Stock Holdings (Auto-Filtered from Page 1)" if not is_zh else "📂 我的长期股票持仓 (自动从页面1过滤)", expanded=False):
            # Smart Filter: Remove Options (tickers with numbers/dates) and Cash
            stocks_only = df_port[
                (~df_port['Ticker'].str.contains(r'\d', na=False)) &
                (df_port['Ticker'] != 'CASH')
            ].copy()

            if not stocks_only.empty:
                st.dataframe(stocks_only, use_container_width=True, hide_index=True)
                st.caption("Click any ticker in the sidebar 'Target Ticker' box to run a deep-dive DCF on it." if not is_zh else "在侧边栏'目标代码'框中输入上方任何代码以运行深度 DCF 分析。")
            else:
                st.info("No long-term stock holdings detected in your portfolio." if not is_zh else "在您的投资组合中未检测到长期股票持仓。")

    # --- EXECUTIVE SNAPSHOT ---
    st.subheader(f"🏢 Executive Snapshot: {target_ticker} ({data.get('sector', 'Unknown')})" if not is_zh else f"🏢 执行快照: {target_ticker} ({data.get('sector', '未知')})")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price" if not is_zh else "当前价格", f"${data.get('price', 0):.2f}")
        c2.metric("Market Cap" if not is_zh else "市值", f"${data.get('market_cap', 0)/1e9:.2f}B")
        c3.metric("P/E Ratio" if not is_zh else "市盈率 (P/E)", f"{data.get('pe', 0):.1f}")
        c4.metric("ROCE (Annual)" if not is_zh else "资本回报率 (ROCE)", f"{data.get('roce', 0)*100:.1f}%", "Capital Efficiency" if not is_zh else "资本效率")

    st.markdown("---")

    # ==========================================
    # TABBED WORKSPACE
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚖️ Fundamentals & DCF" if not is_zh else "⚖️ 基本面与DCF估值",
        "📈 Technicals" if not is_zh else "📈 技术分析",
        "🎯 Analyst Targets" if not is_zh else "🎯 分析师目标价",
        "👯 Peer Comparison" if not is_zh else "👯 同行比较",
        "🧠 Gemini AI Analyst" if not is_zh else "🧠 AI 分析师"
    ])

    # --- TAB 1: FUNDAMENTALS & VALUATION ---
    with tab1:
        # ==========================================
        # 1. QUARTERLY HEALTH CHECK (With Tooltips)
        # ==========================================
        st.markdown("### 🩺 Fundamental Quick Check (Latest Quarter)" if not is_zh else "### 🩺 基本面快速检查 (最新季度)")
        st.caption("Key liquidity and profitability metrics calculated from the most recent earnings report." if not is_zh else "从最近的财报中计算出的关键流动性和盈利指标。")

        def format_currency(value):
            if abs(value) >= 1e12: return f"${value/1e12:.2f}T"
            elif abs(value) >= 1e9: return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6: return f"${value/1e6:.2f}M"
            else: return f"${value:,.0f}"

        q_curr_assets, q_curr_liab = data.get('q_curr_assets', 0), data.get('q_curr_liab', 0)
        curr_ratio = q_curr_assets / q_curr_liab if q_curr_liab > 0 else 0
        quick_ratio = (q_curr_assets - data.get('q_inventory', 0)) / q_curr_liab if q_curr_liab > 0 else 0
        q_equity = data.get('q_equity', 0)
        de_ratio = data.get('q_total_debt', 0) / q_equity if q_equity > 0 else 0
        q_rev = data.get('q_rev', 0)
        gross_margin = data.get('q_gross', 0) / q_rev if q_rev > 0 else 0
        op_margin = data.get('q_op_inc', 0) / q_rev if q_rev > 0 else 0
        q_fcf_value = data.get('q_fcf', 0)
        formatted_fcf = format_currency(q_fcf_value)

        with st.container(border=True):
            # Row 1: Liquidity & Leverage
            h1, h2, h3 = st.columns(3)
            h1.metric("Current Ratio" if not is_zh else "流动比率", f"{curr_ratio:.2f}", "Target: > 1.5", delta_color="off", help="**Total Current Assets / Total Current Liabilities.** Measures ability to pay short-term obligations. > 1.5 is healthy." if not is_zh else "**总流动资产 / 总流动负债。** 衡量偿还短期债务的能力。> 1.5 为健康。")
            h2.metric("Quick Ratio" if not is_zh else "速动比率", f"{quick_ratio:.2f}", "Target: > 1.0", delta_color="off", help="**(Current Assets - Inventory) / Current Liabilities.** A stricter liquidity test. > 1.0 means they can pay debts without selling inventory." if not is_zh else "**(流动资产 - 存货) / 流动负债。** 更严格的流动性测试。> 1.0 意味着他们无需出售存货即可偿还债务。")
            h3.metric("Debt/Equity" if not is_zh else "债务/权益比", f"{de_ratio:.2f}", "Target: < 2.0", delta_color="inverse", help="**Total Debt / Shareholder Equity.** Measures financial leverage. High numbers indicate aggressive debt financing." if not is_zh else "**总债务 / 股东权益。** 衡量财务杠杆。高数值表明激进的债务融资。")

            # Row 2: Cash Flow & Profitability
            h4, h5, h6 = st.columns(3)
            h4.metric("Qtr FCF" if not is_zh else "季度自由现金流", formatted_fcf, "Cash Generated" if not is_zh else "产生的现金", help="**Operating Cash Flow - CapEx.** The actual cash the business generated in the last 90 days." if not is_zh else "**经营现金流 - 资本支出。** 企业在过去90天内实际产生的现金。")
            h5.metric("Gross Margin" if not is_zh else "毛利率", f"{gross_margin*100:.1f}%", help="**Gross Profit / Revenue.** Shows basic production efficiency." if not is_zh else "**毛利润 / 营业收入。** 显示基本生产效率。")
            h6.metric("Op Margin" if not is_zh else "营业利润率", f"{op_margin*100:.1f}%", help="**Operating Income / Revenue.** Shows core business profitability before taxes and interest." if not is_zh else "**营业利润 / 营业收入。** 显示扣除税息前的核心业务盈利能力。")

        st.markdown("---")

        # ==========================================
        # 2. DYNAMIC IN-PAGE DCF CONTROLS
        # ==========================================
        st.markdown("### 🎛️ Dynamic DCF Assumptions" if not is_zh else "### 🎛️ 动态 DCF 假设")

        # Restored your model choice toggle!
        model_choice = st.radio("Valuation Model:" if not is_zh else "估值模型:", ["Standard FCF Growth" if not is_zh else "标准自由现金流增长", "Revenue & Margin Expansion" if not is_zh else "收入与利润率扩张"], horizontal=True)

        with st.container(border=True):
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1: in_wacc = st.number_input("WACC (%)" if not is_zh else "折现率 WACC (%)", value=10.0, step=0.5, format="%.2f") / 100
            with dc2: in_perp = st.number_input("Terminal Growth (%)" if not is_zh else "永续增长率 (%)", value=2.5, step=0.1, format="%.2f") / 100

            if "Standard" in model_choice or "标准" in model_choice:
                with dc3: in_g1 = st.number_input("Years 1-5 Growth (%)" if not is_zh else "1-5年增长率 (%)", value=float(st.session_state.get('auto_fcf_cagr', 15.0)), step=1.0, format="%.2f") / 100
                with dc4: in_g2 = st.number_input("Years 6-10 Growth (%)" if not is_zh else "6-10年增长率 (%)", value=float(in_g1 * 100 * 0.7), step=1.0, format="%.2f") / 100
            else:
                with dc3: in_rev_g1 = st.number_input("Rev Growth Y1-5 (%)" if not is_zh else "1-5年收入增长 (%)", value=float(st.session_state.get('auto_rev_cagr', 12.0)), step=1.0, format="%.2f") / 100
                with dc4: in_margin = st.number_input("Target FCF Margin Y10 (%)" if not is_zh else "第10年目标FCF利润率 (%)", value=25.0, step=1.0, format="%.2f") / 100

        # Dynamic Math Engine (Restored both models)
        future_fcf = []
        if "Standard" in model_choice or "标准" in model_choice:
            curr_fcf = data.get('fcf_ttm', 0)
            for i in range(10):
                curr_fcf *= (1 + (in_g1 if i < 5 else in_g2))
                future_fcf.append(curr_fcf)
        else:
            ttm_rev = data.get('ttm_revenue', 0)
            current_margin = (data.get('fcf_ttm', 0) / ttm_rev) if ttm_rev > 0 else 0
            margins = np.linspace(current_margin, in_margin, 10)
            curr_rev = ttm_rev
            for i in range(10):
                curr_rev *= (1 + (in_rev_g1 if i < 5 else in_rev_g1 * 0.7))
                future_fcf.append(curr_rev * margins[i])

        pv_fcf = [f / (1 + in_wacc)**(i+1) for i, f in enumerate(future_fcf)]
        tv = (future_fcf[-1] * (1 + in_perp)) / (in_wacc - in_perp)
        pv_tv = tv / (1 + in_wacc)**10

        total_cash = data.get('total_cash', 0)
        total_debt = data.get('total_debt', 0)
        shares = data.get('shares', 1) or 1

        equity_value = sum(pv_fcf) + pv_tv + total_cash - total_debt
        fair_value = equity_value / shares
        margin_of_safety = (fair_value - data['price']) / fair_value if fair_value > 0 else 0

        st.markdown("---")

        # ==========================================
        # 3. VALUATION VERDICT & CHECKLIST
        # ==========================================
        col_check, col_val = st.columns([1, 1.5])

        with col_check:
            st.markdown("### ✅ 10-Point Quality Checklist" if not is_zh else "### ✅ 10项质量清单")
            if mandates and len(mandates) > 0:
                score = sum(mandates.values())
                st.progress(score / 10.0, text=f"Score: {score}/10" if not is_zh else f"得分: {score}/10")

                # Clean translated table formatting
                check_df = pd.DataFrame([{"Criterion" if not is_zh else "指标": k, "Status" if not is_zh else "状态": "✅ Pass" if v else "🔴 Fail"} for k, v in mandates.items()])
                st.dataframe(check_df, hide_index=True, use_container_width=True)
            else:
                st.info("Data unavailable for checklist." if not is_zh else "清单数据不可用。")

        with col_val:
            with st.container(border=True):
                st.markdown("### ⚖️ DCF Valuation Verdict" if not is_zh else "### ⚖️ DCF 估值结论")
                v1, v2 = st.columns(2)
                v1.metric("Intrinsic Value" if not is_zh else "内在价值", f"${fair_value:.2f}")
                mos_color = "normal" if margin_of_safety > 0 else "inverse"
                v2.metric("Margin of Safety" if not is_zh else "安全边际", f"{margin_of_safety*100:+.1f}%", delta_color=mos_color)

                st.markdown("**Enterprise to Equity Bridge:**" if not is_zh else "**企业价值至权益价值计算桥梁:**")
                clean_bridge = pd.DataFrame({
                    "Valuation Metric" if not is_zh else "估值指标": ["PV of 10-Yr Cash Flows", "PV of Terminal Value", "Enterprise Value", "+ Total Cash", "- Total Debt", "Equity Value"] if not is_zh else ["10年现金流现值", "终值现值", "企业价值", "+ 总现金", "- 总债务", "权益价值"],
                    "Amount (Billions)" if not is_zh else "金额 (十亿)": [f"${sum(pv_fcf)/1e9:.2f}B", f"${pv_tv/1e9:.2f}B", f"${(sum(pv_fcf)+pv_tv)/1e9:.2f}B", f"${total_cash/1e9:.2f}B", f"${total_debt/1e9:.2f}B", f"${equity_value/1e9:.2f}B"]
                })
                st.dataframe(clean_bridge, use_container_width=True, hide_index=True)

        # ==========================================
        # 3.5 AI DCF ADJUSTER (GEMINI NLP)
        # ==========================================
        st.markdown("---")
        st.markdown("### 🤖 AI DCF Adjuster" if not is_zh else "### 🤖 AI DCF 调整器")
        st.caption("Let Gemini analyze real-time news sentiment to recommend WACC and Terminal Growth adjustments." if not is_zh else "让 Gemini 分析实时新闻情绪，以推荐 WACC 和永续增长率的调整。")

        if HAS_GEMINI and gemini_key:
            if st.button("🔮 Analyze News & Adjust Model" if not is_zh else "🔮 分析新闻并调整模型", type="secondary", use_container_width=True):
                with st.spinner("Scraping live news and consulting Gemini..." if not is_zh else "正在抓取实时新闻并咨询 Gemini..."):
                    try:
                        # 1. Fetch live news using yfinance
                        yf_ticker = yf.Ticker(target_ticker)
                        news_data = yf_ticker.news[:5] # Grab the 5 latest headlines
                        if news_data:
                            headlines = "\n".join([f"- {n.get('title', '')} ({n.get('publisher', '')})" for n in news_data])
                        else:
                            headlines = "No recent news found for this ticker."

                        # 2. Build the Quantitative Prompt
                        prompt = f"""
                        You are an elite quantitative equity analyst advising a hedge fund manager.
                        Target Asset: {target_ticker}
                        Current User Assumptions: WACC = {in_wacc*100}%, Terminal Growth = {in_perp*100}%

                        Recent News Headlines:
                        {headlines}

                        Task:
                        1. Analyze the sentiment of these headlines (Bullish, Bearish, Regulatory Risk, Macro Tailwinds, etc.).
                        2. Based ONLY on this news, recommend specific numerical adjustments to the user's WACC and Terminal Growth. (e.g., "Increase WACC by 0.5% due to heightened regulatory risk", "Increase Terminal Growth by 0.2% due to AI monetization tailwinds").
                        3. Format your response in 3 brutal, concise bullet points: The Sentiment, The WACC Adjustment, and The Terminal Growth Adjustment. Do not write a long essay.
                        """

                        # 3. Ping the API
                        client = genai.Client(api_key=gemini_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=0.2)
                        )

                        # 4. Render the Insight
                        st.success("### 🧠 Gemini's Quantitative Verdict" if not is_zh else "### 🧠 Gemini 的量化结论")
                        st.markdown(response.text)
                        st.caption("Adjust your sliders above based on this intel to see the new intrinsic value." if not is_zh else "根据此情报调整上方的滑块，查看新的内在价值。")

                    except Exception as e:
                        st.error(f"Error consulting Gemini: {e}")
        elif not gemini_key:
            st.warning("Please enter your Gemini API Key in the sidebar to unlock the AI DCF Adjuster." if not is_zh else "请在侧边栏输入您的 Gemini API 密钥以解锁 AI DCF 调整器。")
        else:
            st.error("Google GenAI SDK not installed. Please run `pip install google-genai`." if not is_zh else "未安装 Google GenAI SDK。请运行 `pip install google-genai`。")

        # ==========================================
        # 4. FORECAST CHARTS & SENSITIVITY
        # ==========================================
        st.markdown("---")
        with st.container(border=True):
            st.markdown("### 📊 10-Year Free Cash Flow Projection" if not is_zh else "### 📊 10年自由现金流预测")
            fig = go.Figure(data=[
                go.Bar(x=[f"Year {i}" if not is_zh else f"第 {i} 年" for i in range(1, 11)], y=future_fcf, marker_color='#4F46E5', name="Projected FCF" if not is_zh else "预测自由现金流"),
                go.Scatter(x=[f"Year {i}" if not is_zh else f"第 {i} 年" for i in range(1, 11)], y=pv_fcf, mode='lines+markers', marker_color='#06B6D4', name="Discounted PV" if not is_zh else "折现现值")
            ])
            fig.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown("### 🌡️ Sensitivity Matrix" if not is_zh else "### 🌡️ 敏感性矩阵")
            def get_dynamic_sens(w_adj, g_mult):
                adj_fcf = [f * g_mult for f in future_fcf]
                adj_pv = sum([f / (1 + w_adj)**(i+1) for i, f in enumerate(adj_fcf)])
                adj_tv = (adj_fcf[-1] * (1 + in_perp)) / (w_adj - in_perp)
                adj_pv_tv = adj_tv / (1 + w_adj)**10
                return (adj_pv + adj_pv_tv + total_cash - total_debt) / shares

            w_range = [in_wacc-0.01, in_wacc, in_wacc+0.01]
            g_range = [0.9, 1.0, 1.1]

            matrix = [[get_dynamic_sens(w, g) for w in w_range] for g in g_range]
            row_names = ["Pessimistic (-10% CF)", "Base Case", "Optimistic (+10% CF)"] if not is_zh else ["悲观 (-10% 现金流)", "基本假设", "乐观 (+10% 现金流)"]
            df_sens = pd.DataFrame(
                matrix,
                index=row_names,
                columns=[f"{w*100:.1f}% WACC" for w in w_range]
            )
            # Use stealth dark theme colors for the heatmap dataframe
            st.dataframe(df_sens.style.format("${:.2f}").background_gradient(cmap='viridis', axis=None), use_container_width=True)

    # --- TAB 2: TECHNICALS ---
    with tab2:
        hist_df = data.get('hist_df')
        if hist_df is not None and not hist_df.empty:
            with st.container(border=True):
                st.markdown("### 🕯️ Price Action, Moving Averages & Bollinger Bands")
                fig_price = go.Figure()

                # Bollinger Bands
                fig_price.add_trace(go.Scatter(x=hist_df.index, y=hist_df['BB_Upper'], name="Upper BB", line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False))
                fig_price.add_trace(go.Scatter(x=hist_df.index, y=hist_df['BB_Lower'], name="Lower BB", line=dict(color='rgba(255,255,255,0.1)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', showlegend=False))

                # MAs and Price
                fig_price.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], name="50-Day SMA", line=dict(color='#F6AD55', width=1.5, dash='dash')))
                fig_price.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], name="200-Day SMA", line=dict(color='#0084FF', width=1.5, dash='dash')))
                fig_price.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name="Price", line=dict(color='#E2E8F0', width=2)))

                fig_price.update_layout(template="plotly_dark", height=400, margin=dict(t=10, b=10, l=10, r=10), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_price, use_container_width=True)

            with st.container(border=True):
                st.markdown("### 🌊 MACD (Momentum Shift)")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACD'], name="MACD", line=dict(color='#0084FF', width=2)))
                fig_macd.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Signal'], name="Signal", line=dict(color='#F6AD55', width=2, dash='dot')))
                colors = ['#00C853' if val >= 0 else '#FF3B30' for val in hist_df['MACD_Hist']]
                fig_macd.add_trace(go.Bar(x=hist_df.index, y=hist_df['MACD_Hist'], name="Histogram", marker_color=colors, opacity=0.7))
                fig_macd.update_layout(template="plotly_dark", height=250, margin=dict(t=10, b=10, l=10, r=10), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.warning("Historical price data not available for technical analysis.")

    # --- TAB 3: ANALYST TARGETS (FMP STABLE INTEGRATION) ---
    with tab3:
        st.markdown(f"### 🎯 {target_ticker} Price Targets vs. Historical Price" if not is_zh else f"### 🎯 {target_ticker} 目标价与历史价格")
        st.caption("Visualizing consensus estimates and analyst targets over time." if not is_zh else "可视化随着时间的推移的共识预期和分析师目标价。")

        with st.spinner("Fetching consensus data directly from FMP..." if not is_zh else "直接从 FMP 获取共识数据..."):
            try:
                # 1. Use yfinance for bulletproof historical price data
                hist_df = yf.Ticker(target_ticker).history(period="2y")

                fig_targets = go.Figure()
                if not hist_df.empty:
                    fig_targets.add_trace(go.Scatter(
                        x=hist_df.index, y=hist_df['Close'], mode='lines', name='Close Price',
                        line=dict(color='#0084FF', width=2),
                        hovertemplate="<b>Close Price</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>"
                    ))

                # 2. Use FMP's /stable/ endpoint for the Price Target Consensus
                targets_res = fetch_fmp_json(
                    fmp_key,
                    "price-target-consensus",
                    stable=True,
                    params={"symbol": target_ticker},
                    suppress_error_messages=False,
                )

                if isinstance(targets_res, list) and len(targets_res) > 0:
                    data_pt = targets_res[0]
                    target_cons = data_pt.get('targetConsensus')
                    target_high = data_pt.get('targetHigh')
                    target_low = data_pt.get('targetLow')

                    # Plot the horizontal target lines
                    if target_cons:
                        fig_targets.add_hline(y=target_cons, line_dash="dash", line_color="#00C853", annotation_text=f"Consensus Target: ${target_cons:.2f}", annotation_position="top left", annotation_font_color="#00C853")
                    if target_high:
                        fig_targets.add_hline(y=target_high, line_dash="dot", line_color="#F6AD55", annotation_text=f"High Target: ${target_high:.2f}", annotation_position="top right", annotation_font_color="#F6AD55")
                    if target_low:
                        fig_targets.add_hline(y=target_low, line_dash="dot", line_color="#FF3B30", annotation_text=f"Low Target: ${target_low:.2f}", annotation_position="bottom right", annotation_font_color="#FF3B30")

                elif isinstance(targets_res, dict) and "Error Message" in targets_res:
                    st.error(f"FMP API Error: {targets_res['Error Message']}")
                else:
                    st.warning("Consensus target data not found." if not is_zh else "未找到共识目标价数据。")

                fig_targets.update_layout(
                    template="plotly_dark", height=500, margin=dict(t=10, b=10, l=10, r=10),
                    hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )

                with st.container(border=True):
                    st.plotly_chart(fig_targets, use_container_width=True)

            except Exception as e:
                st.error(f"Error building target chart: {e}")

    # --- TAB 4: PEER COMPARISON (FMP STABLE INTEGRATION & FULL TOOLTIPS) ---
    with tab4:
        st.markdown(f"### 👯 Peer Comparison Analysis" if not is_zh else "### 👯 同行比较分析")
        st.caption(f"Comparing {target_ticker} against peers. **Hover over any column header to see the accounting formula!**" if not is_zh else f"将 {target_ticker} 与同行进行比较。**将鼠标悬停在任何列标题上以查看会计公式！**")

        with st.spinner("Fetching valuation multiples and financial ratios..." if not is_zh else "正在获取估值倍数和财务比率..."):
            try:
                # 1. Use the /stable/ endpoint for peers
                peers_res = fetch_fmp_json(
                    fmp_key,
                    "stock-peers",
                    stable=True,
                    params={"symbol": target_ticker},
                    suppress_error_messages=False,
                )

                peer_list = []
                if isinstance(peers_res, list) and len(peers_res) > 0:
                    if 'peersList' in peers_res[0]:
                        peer_list = peers_res[0]['peersList']
                    else:
                        peer_list = peers_res # FMP sometimes returns a flat list

                if peer_list:
                    combined_tickers = list(dict.fromkeys([target_ticker] + peer_list[:4]))
                    metrics_list, ratios_list = [], []

                    for tick in combined_tickers:
                        # 2. Use the /stable/ endpoints for metrics and ratios
                        m_res = fetch_fmp_json(
                            fmp_key,
                            "key-metrics-ttm",
                            stable=True,
                            params={"symbol": tick},
                            suppress_error_messages=False,
                        )
                        if isinstance(m_res, list) and len(m_res) > 0:
                            row = m_res[0]
                            row['symbol'] = tick
                            metrics_list.append(row)

                        r_res = fetch_fmp_json(
                            fmp_key,
                            "ratios-ttm",
                            stable=True,
                            params={"symbol": tick},
                            suppress_error_messages=False,
                        )
                        if isinstance(r_res, list) and len(r_res) > 0:
                            row = r_res[0]
                            row['symbol'] = tick
                            ratios_list.append(row)

                    metrics_df, ratios_df = pd.DataFrame(metrics_list), pd.DataFrame(ratios_list)
                    if not metrics_df.empty: metrics_df.set_index('symbol', inplace=True)
                    if not ratios_df.empty: ratios_df.set_index('symbol', inplace=True)

                    # --- FULL FMP JSON KEY MAPPINGS & TOOLTIPS ---
                    val_dict = {
                        "peRatioTTM": ("P/E Ratio", "Price / Earnings. Measures how much investors are willing to pay per dollar of earnings."),
                        "pfcfRatioTTM": ("P/FCF", "Price / Free Cash Flow. Similar to P/E but uses hard cash."),
                        "pbRatioTTM": ("P/B Ratio", "Price / Book Value. Compares market valuation to the accounting value of equity."),
                        "enterpriseValueMultipleTTM": ("EV/EBITDA", "Enterprise Value / EBITDA. A capital-structure-neutral valuation multiple."),
                        "evToSalesTTM": ("EV/Sales", "Enterprise Value / Revenue. Compares total firm value to sales."),
                        "evToOperatingCashFlowTTM": ("EV/OCF", "Enterprise Value / Operating Cash Flow."),
                        "evToFreeCashFlowTTM": ("EV/FCF", "Enterprise Value / Free Cash Flow."),
                        "earningsYieldTTM": ("Earnings Yield", "Earnings / Price. The inverse of P/E. Shows percentage return on earnings."),
                        "freeCashFlowYieldTTM": ("FCF Yield", "Free Cash Flow / Market Cap. Cash generation relative to valuation."),
                        "dividendYieldTTM": ("Dividend Yield", "Annual Dividends Per Share / Price.")
                    }

                    ratio_categories = {
                        "Liquidity": {
                            "currentRatioTTM": ("Current Ratio", "Current Assets / Current Liabilities. Measures ability to pay short-term obligations."),
                            "quickRatioTTM": ("Quick Ratio", "(Current Assets - Inventory) / Current Liabilities. Stricter measure of short-term liquidity."),
                            "cashRatioTTM": ("Cash Ratio", "Cash & Equivalents / Current Liabilities. Ability to pay off short-term debt with cash only.")
                        },
                        "Efficiency": {
                            "daysOfSalesOutstandingTTM": ("Days Sales Outstanding (DSO)", "Average time it takes to collect revenue after a sale."),
                            "daysOfInventoryOutstandingTTM": ("Days Inventory Outstanding (DIO)", "Average time it takes to turn inventory into sales."),
                            "daysOfPayablesOutstandingTTM": ("Days Payables Outstanding (DPO)", "Average time it takes the company to pay its bills."),
                            "cashConversionCycleTTM": ("Cash Conversion Cycle", "DIO + DSO - DPO. Time taken to convert resource inputs into cash flows."),
                            "assetTurnoverTTM": ("Asset Turnover", "Revenue / Total Assets. How efficiently assets are used to generate sales.")
                        },
                        "Profitability": {
                            "grossProfitMarginTTM": ("Gross Margin", "Gross Profit / Revenue. Percentage of revenue left after direct production costs."),
                            "operatingProfitMarginTTM": ("Operating Margin", "Operating Income / Revenue. Profitability from core operations before interest/tax."),
                            "netProfitMarginTTM": ("Net Margin", "Net Income / Revenue. The final percentage of revenue that is profit."),
                            "returnOnAssetsTTM": ("ROA", "Net Income / Total Assets. Efficiency of using assets to generate profit."),
                            "returnOnEquityTTM": ("ROE", "Net Income / Shareholder's Equity. Profitability relative to shareholder investment."),
                            "returnOnCapitalEmployedTTM": ("ROCE", "EBIT / Capital Employed. Profitability and efficiency of capital use.")
                        },
                        "Leverage": {
                            "debtRatioTTM": ("Debt Ratio", "Total Debt / Total Assets. Proportion of assets financed by debt."),
                            "debtEquityRatioTTM": ("Debt-to-Equity", "Total Debt / Shareholder's Equity. Measures financial leverage."),
                            "longTermDebtToCapitalizationTTM": ("LT Debt to Cap", "Long Term Debt / (Long Term Debt + Equity). Long-term solvency metric.")
                        },
                        "Coverage": {
                            "interestCoverageTTM": ("Interest Coverage", "EBIT / Interest Expense. How easily a company can pay interest on outstanding debt."),
                            "cashFlowCoverageRatiosTTM": ("CF to Debt", "Operating Cash Flow / Total Debt. Ability to cover total debt with operating cash.")
                        },
                        "Operating Cash Flow": {
                            "operatingCashFlowSalesRatioTTM": ("OCF Margin", "Operating Cash Flow / Revenue. Cash generated per dollar of sales."),
                            "freeCashFlowOperatingCashFlowRatioTTM": ("FCF/OCF", "Free Cash Flow / Operating Cash Flow. How much operating cash becomes free cash."),
                            "capitalExpenditureCoverageRatioTTM": ("CapEx Coverage", "Operating Cash Flow / CapEx. Ability to fund investments from core operations.")
                        }
                    }

                    # --- UI RENDERING ---
                    sub_tab1, sub_tab2 = st.tabs(["💰 Valuation Multiples", "📊 Financial Ratios"])

                    with sub_tab1:
                        if not metrics_df.empty:
                            val_cols_found = [c for c in val_dict.keys() if c in metrics_df.columns]
                            if val_cols_found:
                                display_df = metrics_df[val_cols_found]
                                cfg = {col: st.column_config.NumberColumn(val_dict[col][0], help=val_dict[col][1], format="%.2f") for col in val_cols_found}
                                st.dataframe(display_df, use_container_width=True, column_config=cfg)
                            else:
                                st.info("Standard valuation metrics currently unavailable for this ticker group.")
                        else:
                            st.warning("Could not fetch metrics.")

                    with sub_tab2:
                        if not ratios_df.empty:
                            cat_choice = st.selectbox("Select Ratio Category", list(ratio_categories.keys()))
                            current_dict = ratio_categories[cat_choice]
                            rat_cols_found = [c for c in current_dict.keys() if c in ratios_df.columns]

                            if rat_cols_found:
                                display_df = ratios_df[rat_cols_found]
                                cfg = {col: st.column_config.NumberColumn(current_dict[col][0], help=current_dict[col][1], format="%.3f") for col in rat_cols_found}
                                st.dataframe(display_df, use_container_width=True, column_config=cfg)
                            else:
                                st.info(f"Standard '{cat_choice}' ratios currently unavailable for this ticker group.")
                        else:
                            st.warning("Could not fetch ratios.")

                else:
                    if isinstance(peers_res, dict) and "Error Message" in peers_res:
                        st.error(f"FMP API Error: {peers_res['Error Message']}")
                    else:
                        st.warning(f"Could not find peer data for {target_ticker}.")

            except Exception as e:
                st.error(f"Error fetching direct FMP peer data: {e}")

    # --- TAB 5: GEMINI AI AGENTS ---
    with tab5:
        st.markdown(f"### 🧠 Institutional AI Agents")
        st.caption("Custom-engineered quantitative agents interacting with your live portfolio data.")

        if not HAS_GEMINI:
            st.error("The 'google-genai' library is not installed. Please run `pip install google-genai`.")
        elif not gemini_key:
            st.warning("Please enter your Gemini API Key in the sidebar (or secrets.toml) to activate the AI Agents.")
        else:
            # 1. Agent Selection Toggle
            agent_mode = st.radio("Select AI Agent:", ["Portfolio Planner", "Stock Researcher"], horizontal=True)
            st.markdown("---")

            # 2. Extract Live Portfolio Data for the Agent
            df_port = st.session_state.get('unified_portfolio')
            total_port_value = float(st.session_state.get('portfolio_value', 0.0))

            t212_cash = 0.0
            shares_owned = 0.0
            avg_cost = 0.0

            if df_port is not None and not df_port.empty:
                # 1. Hunt directly for Trading212 Cash (Labeled as 'EUR Cash' in app.py)
                cash_df = df_port[df_port['Ticker'] == 'EUR Cash']
                t212_cash = cash_df['Current_USD'].sum() if not cash_df.empty else 0.0

                # 2. Extract specific stock position
                pos = df_port[df_port['Ticker'] == target_ticker]
                if not pos.empty:
                    pos_current_usd = pos['Current_USD'].sum()
                    pos_cost_usd = pos['Cost_USD'].sum()
                    live_price = data.get('price', 0)

                    # Reverse-engineer shares and cost basis since app.py aggregates them away
                    if live_price > 0:
                        shares_owned = pos_current_usd / live_price
                        avg_cost = pos_cost_usd / shares_owned if shares_owned > 0 else 0

            # 3. Define the System Prompts
            PORTFOLIO_PLANNER_PROMPT = """
            Role & Persona:
            You are an elite Private Fund Manager and Quantitative Analyst advising a retail investor on their personal portfolio (often referred to as the "Uncle Fund").
            Your tone is confident, strategic, empathetic, but brutally objective. You do not panic during market crashes; you view them as opportunities. You use professional but accessible terminology like "Portfolio Litter" (tiny positions), "Cash Fortress" (cash reserves), "Averaging Down," and "Mean Reversion."

            Core Directives:
            1. Always Protect the Cash: Never recommend spending 100% of the user's available cash. Always keep a reserve.
            2. Hate Market Orders: Strongly prefer "Limit Orders" (Traps) to catch volatility, unless immediate exposure is structurally necessary.
            3. Be Specific: Never say "buy some shares." Calculate exact share quantities, specific price targets, and total capital allocation based on the user's cash balance.
            4. Calculate the Impact: Always tell the user what their *new* average cost will be if the orders fill.

            Required Output Structure:
            Whenever the user asks for a plan on a stock, you MUST use the following exact structure and Markdown formatting:

            ### 1. The Diagnosis & Reality Check
            * Start with a punchy one-liner assessing the situation.
            * Briefly explain *why* the stock is at its current price.
            * State their current profit/loss status clearly.

            ### 2. The Strategy
            * Give the overarching move a name.
            * Explain the logic behind the move.

            ### 3. The Execution Plan
            Provide exact orders using bullet points. For each order include: Action, Cost, and Why.

            ### 4. The Portfolio & Cash Impact
            * State the expected new share count and new average cost.
            * Cash Check-In: Subtract the capital required for these orders from the user's total un-reserved cash and state the remaining "Cash Fortress" balance.

            ### 5. The Next Step
            * End with a single, focused question asking the user what part of the portfolio they want to analyze next.
            """

            STOCK_RESEARCHER_PROMPT = """
            Role & Tone:
            You are an elite, highly insightful, and candid financial analyst AI embedded in a professional dashboard. Your goal is to break down complex market dynamics, stock valuations, and economic news into simple, highly scannable, and actionable intelligence. Be objective, straightforward, and engaging. Balance empathy with candor—if a stock looks like a terrible idea, say so gently but firmly.

            Formatting Directives:
            You must structure EVERY stock analysis response using the following exact layout to ensure ultimate scannability. Use exact Markdown formatting as requested.

            1. Introduction (The "Vibe" Check):
            > Start with 2-3 short sentences summarizing the current market sentiment and the reality of the stock right now. Use a relevant analogy if it helps simplify the concept.

            2. ⚖️ The Bull vs. Bear Case:
            > Use horizontal lines (---) to separate this section.
            * **The Pros (Why it's a Buy):** Provide a bulleted list detailing fundamental tailwinds, valuation metrics, or upcoming catalysts. Bold the core concept of each bullet.
            * **The Cons (The Risks):** Provide a bulleted list detailing macroeconomic headwinds, valuation traps, or competitor threats. Bold the core concept of each bullet.

            3. 🗓️ "Surprise Days" to Watch:
            > Always include a markdown table tracking upcoming catalysts. The table must have exactly three columns: | Date | Event | Why it matters |. Include upcoming earnings, dividend dates, product launches, or macroeconomic reports relevant to the stock.

            4. 💡 The Verdict:
            > Give a definitive, multi-layered conclusion formatted with blockquotes (>) to highlight the ultimate bottom line or trading strategy. Break it down into clear scenarios (e.g., "Buy if...", "Hold if...", "Pass if...").
            """

            # 4. Initialize Gemini Client
            client = genai.Client(api_key=gemini_key)

            chat_id = f"gemini_{agent_mode.replace(' ', '_')}_{target_ticker}"
            if chat_id not in st.session_state:
                if agent_mode == "Portfolio Planner":
                    welcome_msg = f"I see you hold **{shares_owned:.2f} shares** of {target_ticker} at an average cost of **${avg_cost:.2f}**, and we have **${t212_cash:,.2f}** in the Trading212 Cash Fortress. How are we maneuvering today?"
                else:
                    welcome_msg = f"I have pulled the latest fundamental data for {target_ticker}. Ask me to generate a full research report or analyze a specific metric."

                st.session_state[chat_id] = [{"role": "assistant", "content": welcome_msg}]

            # 5. Display Chat History
            for msg in st.session_state[chat_id]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 6. Chat Input Field & Execution
            if user_prompt := st.chat_input(f"Talk to the {agent_mode}..."):
                st.session_state[chat_id].append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    with st.spinner(f"The {agent_mode} is calculating..."):
                        try:
                            # Inject live contextual data dynamically
                            if agent_mode == "Portfolio Planner":
                                live_context = f"""
                                --- LIVE PORTFOLIO DATA FOR THIS QUERY ---
                                Target Asset: {target_ticker} (Current Market Price: ${data.get('price', 0):.2f})
                                Total Portfolio Value: ${total_port_value:,.2f}
                                Trading212 Available Cash: ${t212_cash:,.2f}
                                Current Share Count: {shares_owned:.4f}
                                Average Cost Basis: ${avg_cost:.2f}
                                ------------------------------------------
                                User Query: {user_prompt}
                                """
                                sys_instruct = PORTFOLIO_PLANNER_PROMPT
                            else:
                                # We feed the Stock Researcher the live math from your Python backend!
                                live_context = f"""
                                --- LIVE FUNDAMENTAL DATA FOR THIS QUERY ---
                                Target Asset: {target_ticker} (Sector: {data.get('sector', 'Unknown')})
                                Current Price: ${data.get('price', 0):.2f}
                                Market Cap: ${data.get('market_cap', 0)/1e9:.2f}B
                                P/E Ratio: {data.get('pe', 0):.2f}
                                Return on Capital Employed (ROCE): {data.get('roce', 0)*100:.1f}%
                                Latest Quarterly FCF: ${data.get('q_fcf', 0):,.0f}
                                Debt to Equity Ratio: {data.get('q_total_debt', 0) / (data.get('q_equity', 1) or 1):.2f}
                                ------------------------------------------
                                User Query: {user_prompt}
                                """
                                sys_instruct = STOCK_RESEARCHER_PROMPT

                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=live_context,
                                config=types.GenerateContentConfig(
                                    system_instruction=sys_instruct,
                                    temperature=0.4
                                )
                            )

                            ai_reply = response.text
                            st.markdown(ai_reply)
                            st.session_state[chat_id].append({"role": "assistant", "content": ai_reply})

                        except Exception as e:
                            st.error(f"Gemini API Error: {e}")

else:
    st.info("👈 Enter a ticker in the sidebar to initiate Valuation.")
