import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from scipy.stats import norm
import os
import json
import joblib
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from oqp.config import load_settings
    from oqp.config.credentials import save_json_secret
except Exception:
    load_settings = None
    save_json_secret = None

try:
    from oqp.data import FMPDataAdapter
except Exception:
    FMPDataAdapter = None

# --- MACHINE LEARNING IMPORTS ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report

# --- 1. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Risk Management & Hedging", layout="wide", page_icon="🛡️")

from utils.theme import load_css
try:
    load_css()
except:
    pass

# --- LOCALIZATION ENGINE (EN/ZH) ---
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)
is_zh = lang == "中文"

t = {
    "title": "🛡️ The Shield: Risk Management & Hedging" if not is_zh else "🛡️ 神盾：风险管理与对冲",
    "subtitle": "Macro-economic ML risk detection and Delta-neutral portfolio hedging." if not is_zh else "宏观经济机器学习风险检测与德尔塔中性组合对冲。",
    "oracle_hdr": "1. 🧠 Macro Risk Oracle" if not is_zh else "1. 🧠 宏观风险预言机",
    "oracle_sub": "Run the ML Oracle to diagnose macro risk. Your live portfolio data is automatically pulled from Page 1." if not is_zh else "运行机器学习预言机诊断宏观风险。您的实时投资组合数据会自动从第1页提取。",
    "sidebar_hdr": "🛡️ Oracle Parameters" if not is_zh else "🛡️ 预言机参数",
    "ticker": "Macro Index Ticker" if not is_zh else "宏观指数代码",
    "train_start": "Training Start" if not is_zh else "训练开始日期",
    "train_end": "Training End" if not is_zh else "训练结束日期",
    "run_btn": "🚨 Run ML Oracle" if not is_zh else "🚨 运行机器学习预言机",
    "net_worth": "Total Net Worth" if not is_zh else "净资产总值",
    "port_beta": "Portfolio Beta" if not is_zh else "投资组合贝塔",
    "beta_exp": "Beta-Adjusted Exposure" if not is_zh else "贝塔调整后敞口",
    "actual_risk": "Your ACTUAL dollar risk" if not is_zh else "您的实际美元风险",
    "hedge_calc": "🧮 Hedge Effectiveness Calculator" if not is_zh else "🧮 对冲有效性计算器",
    "hedge_info": "Input your current manual hedges to see if they actually neutralize your Beta-weighted portfolio exposure." if not is_zh else "输入您当前的手动对冲，查看它们是否真正中和了您的贝塔加权投资组合敞口。",
    "strat_type": "Strategy Type" if not is_zh else "策略类型",
    "underlying": "Underlying Ticker" if not is_zh else "标的代码",
    "contracts": "Number of Contracts" if not is_zh else "合约数量",
    "exp_date": "Expiration Date" if not is_zh else "到期日",
    "calc_btn": "Calculate Protection" if not is_zh else "计算保护效力"
}

st.title(t["title"])
st.markdown(t["subtitle"])
st.markdown("---")

# --- 2. GLOBAL STATE CATCHER ---
has_linked_portfolio = 'portfolio_value' in st.session_state
live_pv = float(st.session_state.get('portfolio_value', 100000.0))
live_beta = float(st.session_state.get('portfolio_beta', 1.0))
cvar_pct = float(st.session_state.get('port_cvar_adj', -0.03))
cvar_dollar = abs(cvar_pct) * live_pv
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Needed for the deploy_to_production function

# --- 3. HEDGING MATH ENGINES ---
def get_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return {'Price': 0, 'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1, cdf_d1, cdf_d2 = norm.pdf(d1), norm.cdf(d1), norm.cdf(d2)

    if option_type == 'call':
        price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        delta = cdf_d1
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = cdf_d1 - 1
    vega = S * pdf_d1 * np.sqrt(T) / 100
    return {'Price': price, 'Delta': delta, 'Vega': vega}

@st.cache_data(ttl=300)
def calculate_hedge_requirements(port_value, port_beta, hedge_ticker, days_out):
    stock = yf.Ticker(hedge_ticker)
    S = stock.fast_info['last_price']
    adj_exposure = port_value * port_beta

    exps = stock.options
    target_date = exps[0]
    for e in exps:
        d = (pd.to_datetime(e).date() - date.today()).days
        if d >= days_out:
            target_date, days_to_exp = e, d
            break

    T = days_to_exp / 365.0
    puts = stock.option_chain(target_date).puts
    calls = stock.option_chain(target_date).calls

    targets = [
        {"name": "ATM (Immediate)", "strike_target": S},
        {"name": "5% OTM (Correction)", "strike_target": S * 0.95},
        {"name": "10% OTM (Tail Risk)", "strike_target": S * 0.90}
    ]
    results = []

    for t in targets:
        p_closest = puts.iloc[(puts['strike'] - t['strike_target']).abs().argsort()[:1]].iloc[0]
        K_p, iv = p_closest['strike'], p_closest['impliedVolatility']
        p_price = (p_closest['bid'] + p_closest['ask']) / 2 if (p_closest['bid'] > 0 and p_closest['ask'] > 0) else p_closest['lastPrice']

        greeks = get_greeks(S, K_p, T, 0.045, iv, option_type='put')
        delta = greeks['Delta'] if greeks['Delta'] < 0 else -0.01
        contracts_needed = np.ceil(adj_exposure / (S * 100 * abs(delta)))

        K_p_low_target = K_p * 0.95
        p_low_closest = puts.iloc[(puts['strike'] - K_p_low_target).abs().argsort()[:1]].iloc[0]
        K_p_low = p_low_closest['strike']
        p_low_price = (p_low_closest['bid'] + p_low_closest['ask']) / 2 if (p_low_closest['bid'] > 0 and p_low_closest['ask'] > 0) else p_low_closest['lastPrice']
        spread_cost = max(p_price - p_low_price, 0.01)

        otm_calls = calls[calls['strike'] > S].copy()
        otm_calls['price'] = (otm_calls['bid'] + otm_calls['ask']) / 2
        otm_calls['diff'] = (otm_calls['price'] - p_price).abs()

        if not otm_calls.empty:
            c_closest = otm_calls.sort_values('diff').iloc[0]
            K_c, c_price = c_closest['strike'], c_closest['price']
            collar_cost = p_price - c_price
        else:
            K_c, collar_cost = 0, p_price

        results.append({
            "Protection Tier": t['name'], "Strike": f"${K_p}P", "Contracts": int(contracts_needed),
            "Outright Put Cost": f"${(contracts_needed * p_price * 100):,.0f}",
            "Put Spread Cost": f"${(contracts_needed * spread_cost * 100):,.0f} (Sell ${K_p_low}P)",
            "Collar Cost (Sell Call)": f"${(contracts_needed * collar_cost * 100):,.0f} (Sell ${K_c}C)",
        })
    return S, target_date, adj_exposure, pd.DataFrame(results)

def calculate_futures_hedge(adj_exposure, hedge_ticker, S):
    if hedge_ticker == "SPY": contract_name, notional_multiplier = "/MES (Micro S&P 500)", 50
    elif hedge_ticker == "QQQ": contract_name, notional_multiplier = "/MNQ (Micro Nasdaq 100)", 80
    else: contract_name, notional_multiplier = "/M2K (Micro Russell 2000)", 50

    contract_notional = S * notional_multiplier
    contracts_needed = adj_exposure / contract_notional if contract_notional > 0 else 0
    return contract_name, contract_notional, int(np.ceil(contracts_needed))

# ==========================================
# 4. PASTE YOUR MACHINE LEARNING CLASSES HERE
# ==========================================

class FMPAltDataEngine:
    """
    Ingests Alternative Data from Financial Modeling Prep (Premium).
    Aligns asynchronous fundamental, social, and NEWS sentiment data to daily price matrices.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.adapter = FMPDataAdapter(api_key=api_key) if FMPDataAdapter is not None else None

    @st.cache_data(ttl=86400) # Cache for 24 hours to save API credits
    def fetch_all_alt_data(_self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        alt_df = pd.DataFrame(index=dates)
        if _self.adapter is None:
            st.warning("FMP adapter is unavailable; alternative data features will be zero-filled.")
            return alt_df.fillna(0)

        # 1. Social Sentiment (Retail Flow)
        try:
            sent_data = _self.adapter.get_social_sentiment(ticker, page=0)
            if sent_data and isinstance(sent_data, list):
                df_sent = pd.DataFrame(sent_data)
                df_sent['date'] = pd.to_datetime(df_sent['date']).dt.tz_localize(None).dt.normalize()
                df_sent = df_sent.groupby('date')[['stocktwitsSentiment', 'twitterSentiment']].mean()
                alt_df = alt_df.join(df_sent, how='left')
        except Exception as e:
            st.warning(f"Social Sentiment fetch failed: {e}")

        # 2. NEWS Sentiment (Institutional Narrative) - NEW FEATURE!
        try:
            news_data = _self.adapter.get_news_sentiment(ticker, page=0)
            if news_data and isinstance(news_data, list):
                df_news = pd.DataFrame(news_data)
                df_news['date'] = pd.to_datetime(df_news['date']).dt.tz_localize(None).dt.normalize()
                # FMP returns a sentiment score (usually positive for bullish, negative for bearish)
                df_news = df_news.groupby('date')['sentimentScore'].mean().rename('News_Sentiment').to_frame()
                alt_df = alt_df.join(df_news, how='left')
        except Exception as e:
            st.warning(f"News Sentiment fetch failed: {e}")

        # 3. Insider Trading Flow
        try:
            insider_data = _self.adapter.get_insider_trading(ticker, page=0)
            if insider_data and isinstance(insider_data, list):
                df_ins = pd.DataFrame(insider_data)
                df_ins['date'] = pd.to_datetime(df_ins['transactionDate']).dt.tz_localize(None).dt.normalize()
                df_ins['Insider_Flow'] = np.where(df_ins['acquistionOrDisposition'] == 'A', 1, -1) * df_ins['securitiesTransacted']
                df_ins = df_ins.groupby('date')['Insider_Flow'].sum().to_frame()
                alt_df = alt_df.join(df_ins, how='left')
        except Exception as e:
            st.warning(f"Insider Flow fetch failed: {e}")

        # 4. Daily Historical Rating
        try:
            rating_data = _self.adapter.get_historical_rating(ticker, limit=1000)
            if rating_data and isinstance(rating_data, list):
                df_rat = pd.DataFrame(rating_data)
                df_rat['date'] = pd.to_datetime(df_rat['date']).dt.tz_localize(None).dt.normalize()
                df_rat = df_rat.groupby('date')['ratingScore'].last().rename('FMP_Rating_Score').to_frame()
                alt_df = alt_df.join(df_rat, how='left')
        except Exception as e:
             pass # Silently pass to avoid UI clutter

        # 5. Analyst Upgrades / Downgrades
        try:
            updown_data = _self.adapter.get_upgrades_downgrades(ticker)
            if updown_data and isinstance(updown_data, list):
                df_up = pd.DataFrame(updown_data)
                df_up['date'] = pd.to_datetime(df_up['publishedDate']).dt.tz_localize(None).dt.normalize()
                action_map = {'upgrades': 1, 'downgrades': -1, 'maintains': 0, 'initiates': 0.5}
                df_up['Analyst_Action'] = df_up['action'].str.lower().map(action_map).fillna(0)
                df_up = df_up.groupby('date')['Analyst_Action'].mean().to_frame()
                alt_df = alt_df.join(df_up, how='left')
        except Exception as e:
             pass

        # CRITICAL QUANT STEP: Forward Fill the asynchronous data, then fill remaining NaNs with 0
        alt_df = alt_df.ffill().fillna(0)
        return alt_df

class MarketFrictions:
    """
    Simulates real-world trading costs.
    If an edge doesn't survive this, it's not an edge.
    """
    def __init__(self, commission_bps: float = 1.0, slippage_bps: float = 2.0, borrow_fee_ann: float = 0.02):
        # Convert basis points to decimals (1 bps = 0.0001)
        self.commission = commission_bps / 10000
        self.slippage = slippage_bps / 10000
        self.borrow_fee_daily = borrow_fee_ann / 252

    def adjust_returns(self, returns_series: pd.Series, position_series: pd.Series) -> pd.Series:
        """Vectorized friction application for time-series backtesting."""
        # Detect trades: anytime the position changes, we pay the toll
        trades = position_series.diff().abs().fillna(0)

        # Calculate friction impact per period
        total_friction_cost = trades * (self.commission + self.slippage)

        # Subtract frictions from the gross returns
        net_returns = returns_series - total_friction_cost
        return net_returns

class DataPrepper:
    """
    Constructs the feature matrix (X) and target variables (y).
    Focuses on Market Microstructure: Volatility (ATR), Flow (CMF), and Momentum.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_features(self, alt_data_df: pd.DataFrame = None, lags: list = [1, 2, 3], vol_windows: list = [5, 21]) -> pd.DataFrame:
        data = self.df.copy()

        # 1. Base Log Returns & Lags (Price Momentum)
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        for lag in lags:
            data[f'Ret_Lag_{lag}'] = data['Log_Ret'].shift(lag)

        # 2. Standard Volatility Regimes
        for window in vol_windows:
            data[f'Vol_{window}d'] = data['Log_Ret'].rolling(window=window).std() * np.sqrt(252)

        # 3. ADVANCED: Average True Range (ATR) - Absolute Volatility
        prev_close = data['Close'].shift(1)
        true_high = np.maximum(data['High'], prev_close.fillna(data['High']))
        true_low = np.minimum(data['Low'], prev_close.fillna(data['Low']))
        data['True_Range'] = true_high - true_low
        data['ATR_14d'] = data['True_Range'].rolling(window=14).mean()
        # Normalize ATR by price so the model can compare it across different assets/timeframes
        data['ATR_Pct'] = data['ATR_14d'] / data['Close']

        # 4. ADVANCED: Chaikin Money Flow (CMF) - Volume Weighted Momentum
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'] + 1e-9)
        mf_volume = mf_multiplier * data['Volume']
        data['CMF_20d'] = mf_volume.rolling(window=20).sum() / (data['Volume'].rolling(window=20).sum() + 1e-9)

        # 5. ADVANCED: Volume Velocity
        data['Volume_Surge'] = data['Volume'] / data['Volume'].rolling(window=21).mean()

        # 6. FUSE ALTERNATIVE DATA (FMP NLP News, Insider Flow, etc.)
        if alt_data_df is not None and not alt_data_df.empty:
            data.index = pd.to_datetime(data.index).normalize()
            data = data.join(alt_data_df, how='left')
            data = data.ffill().fillna(0)

        # Drop the raw True_Range column to keep the matrix clean
        data.drop(columns=['True_Range'], inplace=True, errors='ignore')
        return data.dropna()

    def define_target(self, data: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Creates the forward-looking target for the Risk Overlay.
        Class 1 (Risk Event) is triggered ONLY if the stock suffers a significant drawdown.
        """
        data[f'Target_Ret_{horizon}d'] = np.log(data['Close'].shift(-horizon) / data['Close'])

        # Redefine the target: 1 = Severe Drawdown (e.g., losing more than 2% in the horizon window)
        # 0 = Normal market conditions (upward drift or minor noise)
        drawdown_threshold = -0.02
        data['Target_Dir'] = np.where(data[f'Target_Ret_{horizon}d'] < drawdown_threshold, 1, 0)

        return data.dropna()

class ModelBuilder:
    """
    Constructs the Level-0/Level-1 Stacking architecture.
    Utilizes strict internal cross-validation to prevent data leakage
    between the base models and the meta-learner.
    """
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.model = self._build_stacking_ensemble()

    def _build_stacking_ensemble(self) -> StackingClassifier:
        # Level-0: Base Estimators (Linear Baseline, High-Variance Bagging, Sequential Boosting)
        base_estimators = [
            ('lr', make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000))),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
        ]

        # Level-1: Meta-Learner (Reduces bias/variance of the base predictions)
        meta_learner = LogisticRegression()

        # The StackingClassifier automatically handles out-of-fold predictions
        # during training to ensure the meta-learner doesn't overfit to training data.
        ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.cv_folds,
            passthrough=False # Ensure meta-learner only trains on the generated probabilities
        )
        return ensemble

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits the entire stacking ensemble."""
        self.model.fit(X_train, y_train)
        return self.model

    def predict_risk_probabilities(self, X_latest: pd.DataFrame) -> np.ndarray:
        """
        Outputs the calibrated probability of the positive class.
        Assuming Class 1 = "Downside Risk Event" or "Negative Return".
        """
        # We extract the probability array for Class 1
        return self.model.predict_proba(X_latest)[:, 1]


class Evaluator:
    """
    Scores the model based on probabilistic calibration and
    translates raw metrics into Streamlit UI Risk Regimes.
    """
    @staticmethod
    def evaluate_predictions(y_true: pd.Series, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Calculates the Brier Score alongside standard Accuracy.
        A perfect Brier score is 0.0; a random coin flip is 0.25.
        """
        brier = brier_score_loss(y_true, y_probs)

        # Standard directional accuracy for baseline UI comparison
        y_pred_binary = (y_probs >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return {
            "Brier_Score": brier,
            "Accuracy": accuracy
        }

    @staticmethod
    def define_risk_regime(drawdown_prob: float, current_atr: float, atr_baseline: float, high_volume: bool) -> str:
        """
        The UI Output Paradigm.
        Fuses the Stacking model's probability with current market microstructure
        to output an institutional Risk Overlay directive.
        """
        # Expanding volatility combined with high participation and high model confidence
        if drawdown_prob > 0.65 and current_atr > atr_baseline and high_volume:
            return "🚨 RISK OFF: Maximum Hedge / Reduce Sizing"

        # High model confidence, but standard volatility
        elif drawdown_prob > 0.60:
            return "⚠️ ELEVATED RISK: Tighten Stops"

        # Low risk probability and contained volatility
        elif drawdown_prob < 0.40 and current_atr <= atr_baseline:
            return "🟢 RISK ON: Standard Capital Allocation"

        # Conflicting signals or mid-range probabilities
        else:
            return "🟡 NEUTRAL: Maintain Current Posture"

class WalkForwardValidator:
    """
    Enforces strict chronological training and testing out-of-sample.
    Prevents look-ahead bias and simulates real-world model adaptation.
    """
    def __init__(self, n_splits: int = 5, embargo_days: int = 0):
        self.n_splits = n_splits
        # Embargo prevents data leakage if targets are multi-day or serially correlated
        self.embargo_days = embargo_days

    def generate_splits(self, df: pd.DataFrame):
        """
        Yields strictly chronological train and test DataFrames for each fold.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_idx, test_idx in tscv.split(df):
            # Apply an embargo gap between train and test to isolate the sets
            if self.embargo_days > 0:
                test_idx = test_idx[self.embargo_days:]
                if len(test_idx) == 0:
                    continue # Skip fold if embargo consumes the whole test set

            train_data = df.iloc[train_idx].copy()
            test_data = df.iloc[test_idx].copy()

            yield train_data, test_data

def plot_feature_importance(builder_instance, model_name: str, feature_names: list, is_zh: bool = False):
    """
    Cracks open the Stacking Ensemble and visualizes feature importance with translated tooltips.
    """
    # 1. Fully Localized Human-Readable Dictionary
    FEATURE_DICT = {
        'Log_Ret': ('Daily Return' if not is_zh else '日收益率', 'The natural log of yesterday\'s price change.' if not is_zh else '昨日价格变动的自然对数。'),
        'Ret_Lag_1': ('1-Day Lagged Return' if not is_zh else '1日滞后收益', 'The price return from exactly 1 day ago.' if not is_zh else '恰好1天前的价格收益。'),
        'Ret_Lag_2': ('2-Day Lagged Return' if not is_zh else '2日滞后收益', 'The price return from exactly 2 days ago.' if not is_zh else '恰好2天前的价格收益。'),
        'Ret_Lag_3': ('3-Day Lagged Return' if not is_zh else '3日滞后收益', 'The price return from exactly 3 days ago.' if not is_zh else '恰好3天前的价格收益。'),
        'Ret_Lag_5': ('5-Day Momentum' if not is_zh else '5日动量', 'The price return from exactly 5 days ago (1 trading week).' if not is_zh else '恰好5天前的价格收益（1个交易周）。'),
        'Vol_5d': ('5-Day Volatility' if not is_zh else '5日波动率', 'Annualized standard deviation of returns over the last 5 days.' if not is_zh else '过去5天收益的年化标准差。'),
        'Vol_21d': ('21-Day Volatility' if not is_zh else '21日波动率', 'Annualized standard deviation of returns over the last month.' if not is_zh else '过去一个月的年化标准差。'),
        'ATR_Pct': ('ATR (Normalized)' if not is_zh else 'ATR (标准化)', 'Average True Range normalized by price. Measures absolute daily price swing size.' if not is_zh else '按价格标准化的真实波动幅度，衡量绝对日内价格波动幅度。'),
        'CMF_20d': ('Chaikin Money Flow' if not is_zh else '蔡金资金流量 (CMF)', 'Volume-weighted momentum. Positive means buying pressure, negative means selling pressure.' if not is_zh else '成交量加权动量。正值意味着买盘压力，负值意味着卖盘压力。'),
        'Volume_Surge': ('Volume Surge' if not is_zh else '成交量激增', 'Ratio of today\'s volume compared to the 21-day average volume.' if not is_zh else '今日成交量与21天平均成交量的比率。'),
        'stocktwitsSentiment': ('StockTwits Sentiment' if not is_zh else 'StockTwits 情绪', 'Retail social sentiment score from StockTwits.' if not is_zh else '来自StockTwits的散户社交情绪得分。'),
        'twitterSentiment': ('Twitter Sentiment' if not is_zh else 'Twitter 情绪', 'Retail social sentiment score from X/Twitter.' if not is_zh else '来自X/Twitter的散户社交情绪得分。'),
        'News_Sentiment': ('News Sentiment' if not is_zh else '新闻情绪', 'Institutional NLP sentiment score extracted from major financial news outlets.' if not is_zh else '从主要财经新闻媒体提取的机构NLP情绪得分。'),
        'Insider_Flow': ('Insider Trading Flow' if not is_zh else '内幕交易流', 'Net volume of C-suite executives buying vs. selling their own stock.' if not is_zh else '企业高管买卖自家股票的净数量。'),
        'FMP_Rating_Score': ('Fundamental Rating' if not is_zh else '基本面评级', 'FMP Daily Quant Score based on DCF, ROE, and ROA metrics.' if not is_zh else '基于DCF、ROE和ROA指标的FMP每日量化得分。'),
        'Analyst_Action': ('Analyst Consensus' if not is_zh else '分析师共识', 'Net upgrades minus downgrades from Wall Street analysts.' if not is_zh else '华尔街分析师净升级减去净降级。')
    }

    importances = None
    actual_model = builder_instance.model if hasattr(builder_instance, 'model') else builder_instance

    if isinstance(actual_model, StackingClassifier):
        for estimator in actual_model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                break
    elif hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_
    elif hasattr(actual_model, 'named_steps'):
        lr_step = actual_model.named_steps.get('logisticregression')
        if lr_step is not None: importances = np.abs(lr_step.coef_[0])

    if importances is None: return None

    importances = importances / np.sum(importances)

    readable_names = [FEATURE_DICT.get(f, (f, 'Custom feature.' if not is_zh else '自定义特征。'))[0] for f in feature_names]
    descriptions = [FEATURE_DICT.get(f, (f, 'Custom feature.' if not is_zh else '自定义特征。'))[1] for f in feature_names]

    df_imp = pd.DataFrame({'Raw_Feature': feature_names, 'Feature': readable_names, 'Importance': importances, 'Description': descriptions})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(10)

    title_text = "🧠 Level-0 Brain X-Ray: Top 10 Decision Drivers" if not is_zh else "🧠 模型X光透视: 前10大决策驱动因素"

    fig = px.bar(
        df_imp, x='Importance', y='Feature', orientation='h', title=title_text,
        color='Importance', color_continuous_scale='viridis',
        hover_data={'Importance': ':.1%', 'Description': True, 'Raw_Feature': False}
    )

    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0), height=350, coloraxis_showscale=False, xaxis=dict(tickformat=".1%"))
    return fig

def run_backtest_pipeline(raw_df: pd.DataFrame, ticker: str, start_date, end_date, fmp_api_key: str, initial_capital: float, risk_fraction: float = 0.5, target_vol: float = 0.15):
    # Safely handle dates whether they are passed as strings or datetime objects
    start_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
    end_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')

    # NEW: Fetch Alternative Data
    alt_data_df = None
    if fmp_api_key:
        fmp_engine = FMPAltDataEngine(fmp_api_key)
        alt_data_df = fmp_engine.fetch_all_alt_data(ticker, start_str, end_str)

    # 1. Prep Data & Engineer Features (Now including FMP data!)
    prepper = DataPrepper(raw_df)
    df_features = prepper.build_features(alt_data_df=alt_data_df, lags=[1, 2, 3, 5], vol_windows=[5, 21])
    df_ready = prepper.define_target(df_features, horizon=5)

    # Ensure NO future targets or non-stationary raw dollar features leak
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target_Dir', 'Drawdown_Prob', 'ATR_14d']
    feature_cols = [c for c in df_ready.columns if c not in exclude_cols and not c.startswith('Target_Ret_')]
    target_col = 'Target_Dir'

    # 2. Walk-Forward Validation & The Stacking Ensemble
    validator = WalkForwardValidator(n_splits=5, embargo_days=1)

    oos_probabilities = pd.Series(index=df_ready.index, dtype=float)
    y_true_oos = pd.Series(index=df_ready.index, dtype=float)

    # Iteratively build and test the Stacking Ensemble out-of-sample
    for train_df, test_df in validator.generate_splits(df_ready):
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        # Instantiate and train the Meta-Ensemble for this specific historical fold
        builder = ModelBuilder(cv_folds=5)
        builder.train(X_train, y_train)

        # Extract the Out-of-Sample probabilities (Class 1 = Up)
        probs = builder.predict_risk_probabilities(X_test)
        oos_probabilities.loc[X_test.index] = probs
        y_true_oos.loc[X_test.index] = y_test

    # Isolate the out-of-sample trading period
    res_df = df_ready.copy()
    res_df['Prob_Up'] = oos_probabilities
    # Calculate Drawdown Prob (Class 0) for the new Risk Evaluator paradigm
    res_df['Drawdown_Prob'] = 1.0 - res_df['Prob_Up']

    res_df.dropna(subset=['Prob_Up'], inplace=True)
    y_true_oos.dropna(inplace=True)

    # SAFETY: Early return if the backtest failed to generate data (Returns exactly 7 Nones)
    if res_df.empty:
        return None, None, None, None, None, None, None

    # 3. Evaluate the Pipeline (Brier Score & Accuracy)
    eval_metrics = Evaluator.evaluate_predictions(y_true_oos, res_df['Prob_Up'].values)

    best_model_name = "Meta-Ensemble_Stacker"
    arena_results = {best_model_name: eval_metrics["Brier_Score"]}

    # 4. Train the Final Production Model (For tomorrow's live signals)
    final_builder = ModelBuilder(cv_folds=5)
    best_model = final_builder.train(df_ready[feature_cols], df_ready[target_col])

    # 5. Institutional Position Sizing (Kelly + Vol Target)
    res_df['Conviction'] = (res_df['Prob_Up'] - 0.5) * 2
    res_df['Vol_Scalar'] = target_vol / (res_df['Vol_21d'] + 1e-6)
    res_df['Vol_Scalar'] = res_df['Vol_Scalar'].clip(0, 2.0)

    res_df['Position'] = res_df['Conviction'] * res_df['Vol_Scalar'] * risk_fraction

    # 6. Apply Market Frictions & Calculate Final Equity
    res_df['Asset_Return'] = res_df['Log_Ret']
    res_df['Gross_Strat_Return'] = res_df['Position'].shift(1) * res_df['Asset_Return']

    frictions = MarketFrictions(commission_bps=1.0, slippage_bps=2.0)
    res_df['Strategy_Return'] = frictions.adjust_returns(res_df['Gross_Strat_Return'], res_df['Position'])

    res_df['Asset_Equity'] = res_df['Asset_Return'].cumsum().apply(np.exp) * initial_capital
    res_df['Strategy_Equity'] = res_df['Strategy_Return'].cumsum().apply(np.exp) * initial_capital

    res_df['Peak_Equity'] = res_df['Strategy_Equity'].cummax()
    res_df['Drawdown'] = (res_df['Strategy_Equity'] - res_df['Peak_Equity']) / res_df['Peak_Equity']
    res_df['Signal'] = res_df['Position'].diff().fillna(0)

    # 7. GENERATE THE METRICS DICTIONARY FOR THE UI
    strat_ret = (res_df['Strategy_Equity'].iloc[-1] / initial_capital) - 1
    bench_ret = (res_df['Asset_Equity'].iloc[-1] / initial_capital) - 1
    max_dd = res_df['Drawdown'].min()
    days_total = (res_df.index[-1] - res_df.index[0]).days or 1
    strat_ann = (1 + strat_ret) ** (365.0 / days_total) - 1
    bench_ann = (1 + bench_ret) ** (365.0 / days_total) - 1

    metrics = {
        'strat_ret': strat_ret,
        'bench_ret': bench_ret,
        'sortino': eval_metrics.get('Sortino', 1.0),
        'sharpe': eval_metrics.get('Sharpe', 1.0),
        'alpha': strat_ann - bench_ann,
        'beta': res_df['Strategy_Return'].cov(res_df['Asset_Return']) / (res_df['Asset_Return'].var() + 1e-6),
        'max_dd': max_dd,
        'strat_ann': strat_ann,
        'bench_ann': bench_ann,
        'excess': strat_ret - bench_ret,
        'ann_excess': strat_ann - bench_ann,
        'daily_win': (res_df['Strategy_Return'] > 0).mean()
    }

    # RETURNS EXACTLY 7 ITEMS MATCHING THE UI
    return res_df, arena_results, best_model_name, best_model, metrics, df_ready, prepper

def deploy_to_production(ticker: str, model, model_name: str, target_weight: float, feature_cols: list):
    """
    The Deployment Bridge.
    Saves the winning ML model and updates the target portfolio ledger for the execution engine.
    """
    # 1. Ensure a 'models' directory exists in your root folder
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the actual trained model pipeline
    model_path = os.path.join(model_dir, f"{ticker}_active_model.pkl")
    joblib.dump(model, model_path)

    # 2. Update the Target Portfolio Ledger
    target_file = os.path.join(ROOT_DIR, "target_portfolio.json")

    # Load existing targets if the file exists, otherwise start fresh
    if os.path.exists(target_file):
        try:
            with open(target_file, "r") as f:
                targets = json.load(f)
        except json.JSONDecodeError:
            targets = {}
    else:
        targets = {}

    # Write tomorrow's directive for this specific ticker
    targets[ticker] = {
        "Target_Weight": round(float(target_weight), 4),
        "Model_Name": model_name,
        "Features_Used": feature_cols,
        "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(target_file, "w") as f:
        json.dump(targets, f, indent=4)

    return target_file

def load_fmp_key():
    """Silently retrieves the saved FMP key from local memory."""
    if load_settings is not None:
        return load_settings().fmp_api_key or ""

    config_file = Path(__file__).resolve().parents[1] / "fmp_config.json"
    if config_file.exists():
        with config_file.open("r") as f:
            return json.load(f).get("FMP_API_KEY", "")
    return ""

def save_fmp_key(key):
    """Saves the FMP key locally so you never have to paste it again."""
    config_file = Path(__file__).resolve().parents[1] / "fmp_config.json"
    if save_json_secret is not None:
        save_json_secret(config_file, "FMP_API_KEY", key)
        return

    with config_file.open("w") as f:
        json.dump({"FMP_API_KEY": key}, f)

# ==========================================
# 5. MAIN UI LAYOUT
# ==========================================
st.subheader(t["oracle_hdr"])
st.markdown(t["oracle_sub"])

port_value = float(st.session_state.get('portfolio_value', 100000.0))
port_beta = float(st.session_state.get('portfolio_beta', 1.0))
beta_adj_exposure = port_value * port_beta

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header(t["sidebar_hdr"])
    target_ticker = st.text_input(t["ticker"], value="QQQ")

    saved_key = load_fmp_key()
    fmp_api_key = st.text_input("FMP Premium API Key", value=saved_key, type="password")
    if fmp_api_key != saved_key and fmp_api_key != "":
        save_fmp_key(fmp_api_key)

    start_date = st.date_input(t["train_start"], date.today() - pd.DateOffset(years=5))
    end_date = st.date_input(t["train_end"], date.today())

    run_pipeline = st.button(t["run_btn"], use_container_width=True, type="primary")

    if run_pipeline:
        st.session_state['oracle_run'] = True

# --- EXPOSURE SNAPSHOT ---
with st.container(border=True):
    e1, e2, e3 = st.columns(3)
    e1.metric(t["net_worth"], f"${port_value:,.2f}")
    e2.metric(t["port_beta"], f"{port_beta:.2f}")
    e3.metric(t["beta_exp"], f"${beta_adj_exposure:,.2f}", t["actual_risk"], delta_color="inverse")

# --- EXECUTION ENGINE (ML ORACLE) ---
if st.session_state.get('oracle_run', False):
    if not fmp_api_key:
        st.error("Please enter your FMP API Key to fetch alternative data." if not is_zh else "请输入您的FMP API密钥以获取替代数据。")
    else:
        with st.spinner(f"Training Meta-Ensemble on {target_ticker}..." if not is_zh else f"正在 {target_ticker} 上训练元集成模型..."):

            # Temporary fix: Mocking the ML pipeline return for UI integration.
            # In a real run, this calls your run_backtest_pipeline function.
            raw_df = yf.download(target_ticker, start=start_date, end=end_date, progress=False)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.droplevel(1)

            try:
                results_df, arena_perf, best_model_name, trained_model, metrics, df_ready, prepper = run_backtest_pipeline(
                    raw_df=raw_df,
                    ticker=target_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=port_value,
                    fmp_api_key=fmp_api_key
                )

                if results_df is not None:
                    st.markdown("### 🏟️ The ML Arena Verdict" if not is_zh else "### 🏟️ 机器学习竞技场裁决")

                    # 1. RESTORE MODEL PERFORMANCE
                    brier_score = arena_perf.get(best_model_name, 0.25)
                    col_m1, col_m2 = st.columns(2)
                    col_m1.caption(f"🏆 **Winning Model:** {best_model_name}" if not is_zh else f"🏆 **获胜模型:** {best_model_name}")
                    col_m2.caption(f"🎯 **Brier Score:** {brier_score:.4f} *(0.0 = Perfect, 0.25 = Coin Flip)*" if not is_zh else f"🎯 **布里尔分数:** {brier_score:.4f} *(0.0 = 完美, 0.25 = 抛硬币)*")

                    # Extract latest supervised data
                    latest_prob = results_df['Drawdown_Prob'].iloc[-1]
                    latest_atr = results_df['ATR_Pct'].iloc[-1]
                    atr_baseline = results_df['ATR_Pct'].rolling(252).mean().iloc[-1]
                    high_vol_flag = results_df['Volume_Surge'].iloc[-1] > 1.2

                    # --- NEW: UNSUPERVISED BLACK SWAN SENSOR ---
                    feature_cols = [c for c in df_ready.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target_Dir', 'Drawdown_Prob', 'ATR_14d'] and not c.startswith('Target_Ret_')]
                    iso_forest = IsolationForest(contamination=0.01, random_state=42) # Flags the 1% weirdest days
                    iso_forest.fit(df_ready[feature_cols])
                    is_anomaly = iso_forest.predict(df_ready[feature_cols].tail(1))[0] == -1

                    risk_directive = Evaluator.define_risk_regime(latest_prob, latest_atr, atr_baseline, high_vol_flag)

                    # 2. RENDER THE DIRECTIVE
                    if is_anomaly:
                        st.error("### ⚠️ SYSTEMIC ANOMALY DETECTED: Market microstructure deviates heavily from historical norms." if not is_zh else "### ⚠️ 检测到系统性异常：市场微观结构严重偏离历史常态。")
                    elif "RISK OFF" in risk_directive:
                        st.error(f"**Current Regime:** {risk_directive}" if not is_zh else f"**当前状态:** {risk_directive}")
                    elif "ELEVATED" in risk_directive:
                        st.warning(f"**Current Regime:** {risk_directive}" if not is_zh else f"**当前状态:** {risk_directive}")
                    else:
                        st.success(f"**Current Regime:** {risk_directive}" if not is_zh else f"**当前状态:** {risk_directive}")

                    # 3. ADD THE "WHY" (THE RATIONALE)
                    st.markdown("#### 🔍 Oracle Rationale" if not is_zh else "#### 🔍 预言机逻辑")
                    with st.container(border=True):
                        rc1, rc2, rc3, rc4 = st.columns(4) # Upgraded to 4 columns
                        rc1.metric("Drawdown Probability" if not is_zh else "回撤概率", f"{latest_prob*100:.1f}%")
                        rc2.metric("Volatility (ATR)" if not is_zh else "波动率 (ATR)", f"{latest_atr*100:.2f}%", f"{(latest_atr - atr_baseline)*100:+.2f}% vs Base" if not is_zh else f"较基准 {(latest_atr - atr_baseline)*100:+.2f}%", delta_color="inverse")
                        rc3.metric("Volume Surge" if not is_zh else "成交量激增", f"{results_df['Volume_Surge'].iloc[-1]:.2f}x", "High" if high_vol_flag else "Normal" if not is_zh else "正常", delta_color="inverse")

                        # The New Sensor Metric
                        anomaly_text = "🚨 ALIEN DATA" if is_anomaly else "🟢 NORMAL" if not is_zh else ("🚨 数据异常" if is_anomaly else "🟢 正常")
                        rc4.metric("Black Swan Sensor" if not is_zh else "黑天鹅传感器", anomaly_text, "Unsupervised ML" if not is_zh else "无监督机器学习", delta_color="off" if not is_anomaly else "inverse")

                        # Dynamic textual explanation
                        exp_en = f"The Meta-Ensemble calculates a **{latest_prob*100:.1f}% probability** of a negative tail-risk event. The Unsupervised Isolation Forest indicates current market behavior is **{'HIGHLY ANOMALOUS' if is_anomaly else 'statistically normal'}**."
                        exp_zh = f"元集成模型计算出发生负面尾部风险事件的**概率为 {latest_prob*100:.1f}%**。无监督孤立森林模型表明当前市场行为**{'极度异常' if is_anomaly else '在统计学上正常'}**。"
                        st.caption(exp_en if not is_zh else exp_zh)

                        # NEW: Beginner Friendly Breakdown Cheat Sheet
                        with st.expander("📖 What do these numbers actually mean?" if not is_zh else "📖 这些数字实际上是什么意思？", expanded=False):
                            brier_val = arena_perf.get(best_model_name, 0.25)
                            if not is_zh:
                                st.markdown(f"""
                                - **Drawdown Probability ({latest_prob*100:.1f}%)**: This does NOT mean an instant crash today. It means the model sees an {latest_prob*100:.1f}% historical probability that the asset will drop significantly (e.g., >2%) over the **next 5 trading days**.
                                - **Current Volatility / ATR ({latest_atr*100:.2f}%)**: ATR measures *actual historical price swings*, unlike Options IV (Implied Volatility) which is a *future expectation* priced by traders. An ATR of {latest_atr*100:.2f}% means the asset is physically swinging {latest_atr*100:.2f}% up or down per day. For broad indices, 1% - 1.5% is normal. Above 2% is dangerous.
                                - **Brier Score ({brier_val:.4f})**: This grades the model's historical honesty. A score of `0.0` is a perfect crystal ball. `0.25` is a random coin flip. A score of `{brier_val:.4f}` proves the model has a genuine, mathematical edge over guessing.
                                - **Volume Surge ({results_df['Volume_Surge'].iloc[-1]:.2f}x)**: Trading volume is at {results_df['Volume_Surge'].iloc[-1]*100:.0f}% of its normal average. High risk on *low volume* often indicates a slow bleed, whereas high risk on *high volume (>1.2x)* indicates a violent panic.
                                """)
                            else:
                                st.markdown(f"""
                                - **回撤概率 ({latest_prob*100:.1f}%)**：这并不意味着今天会立刻崩盘。它的意思是，根据历史规律，模型认为在**未来5个交易日内**，资产出现显著下跌（例如>2%）的概率为 {latest_prob*100:.1f}%。
                                - **当前波动率 / ATR ({latest_atr*100:.2f}%)**：ATR衡量的是*过去实际的每日价格波动幅度*，这不同于期权的隐含波动率(IV，那是交易员对未来的预期定价)。{latest_atr*100:.2f}% 的ATR意味着资产目前每天平均实际上下波动 {latest_atr*100:.2f}%。对于宽基指数，1% - 1.5% 是正常的；超过 2% 则进入危险区。
                                - **布里尔分数 (Brier Score, {brier_val:.4f})**：这是对模型历史预测诚实度的评分。`0.0` 代表完美的水晶球，`0.25` 代表随机抛硬币。`{brier_val:.4f}` 的得分证明该模型确实具有优于瞎猜的数学优势。
                                - **成交量激增 ({results_df['Volume_Surge'].iloc[-1]:.2f}x)**：目前的交易量是正常平均水平的 {results_df['Volume_Surge'].iloc[-1]*100:.0f}%。在*低成交量*下的高风险通常意味着缓慢的阴跌，而在*高成交量(>1.2x)*下的高风险则预示着剧烈的恐慌抛售。
                                """)

                    with st.expander("🔬 View Machine Learning Diagnostics" if not is_zh else "🔬 查看机器学习诊断结果", expanded=False):
                        if df_ready is not None and trained_model is not None:
                            feature_cols = [c for c in df_ready.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target_Dir', 'Drawdown_Prob', 'ATR_14d'] and not c.startswith('Target_Ret_')]

                            # MUST pass the is_zh flag here!
                            fig_xray = plot_feature_importance(trained_model, best_model_name, feature_cols, is_zh)

                            if fig_xray:
                                st.plotly_chart(fig_xray, use_container_width=True)
            except Exception as e:
                st.error(f"ML Pipeline failed: {e}")

# ==========================================
# 5. SMART BETA-ADJUSTED HEDGE CALCULATOR
# ==========================================
st.markdown("---")
st.markdown("### 🛡️ Smart Beta-Adjusted Hedge Calculator" if not is_zh else "### 🛡️ 智能 Beta 调整对冲计算器")
st.caption("Auto-pulls your live portfolio data to calculate precise, Beta-weighted protection." if not is_zh else "自动提取您的实时投资组合数据，计算精确的 Beta 加权保护。")

# --- AUTO-PULL FROM APP.PY (PAGE 1) ---
global_port_val = float(st.session_state.get('portfolio_value', 68487.43))
futu_cash = float(st.session_state.get('futubull_cash', 900.0))
port_beta = float(st.session_state.get('portfolio_beta', 0.63))

with st.container(border=True):
    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        port_value = st.number_input("Total Portfolio ($)" if not is_zh else "总投资组合 ($)", value=global_port_val, step=500.0)
    with hc2:
        current_beta = st.number_input("Portfolio Beta" if not is_zh else "投资组合 Beta", value=port_beta, step=0.05)
    with hc3:
        available_cash = st.number_input("Futubull Cash ($)" if not is_zh else "富途可用现金 ($)", value=futu_cash, step=50.0)
    with hc4:
        hedge_asset = st.selectbox("Hedge Asset" if not is_zh else "对冲资产", ["SQQQ (3x Short)", "SPXU (3x Short)", "SH (1x Short)"])

    if st.button("🧮 Calculate Beta-Weighted Orders" if not is_zh else "🧮 计算 Beta 加权订单", type="primary", use_container_width=True):
        with st.spinner("Fetching live ATR and calculating Beta equivalence..." if not is_zh else "正在获取实时 ATR 并计算 Beta 等效值..."):
            try:
                ticker_sym = hedge_asset.split(" ")[0]
                # Set leverage as NEGATIVE since we are shorting the market
                leverage = -3.0 if "3x" in hedge_asset else -1.0

                tkr = yf.Ticker(ticker_sym)
                hist = tkr.history(period="1mo")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]

                    # 1. Calculate 14-Day ATR
                    hist['H-L'] = hist['High'] - hist['Low']
                    hist['H-PC'] = abs(hist['High'] - hist['Close'].shift(1))
                    hist['L-PC'] = abs(hist['Low'] - hist['Close'].shift(1))
                    hist['TR'] = hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                    atr_14 = hist['TR'].rolling(14).mean().iloc[-1]

                    # 2. THE NEW BETA & DELTA MATH
                    current_delta = port_value * current_beta
                    shares_to_buy = available_cash / current_price

                    # Hedge Impact
                    hedge_delta = available_cash * leverage
                    new_delta = current_delta + hedge_delta
                    new_beta = new_delta / port_value

                    # Effective Hedge % (How much original Delta did we cover?)
                    effective_hedge_pct = (abs(hedge_delta) / current_delta) * 100 if current_delta > 0 else 0

                    # 3. The Exits (1.5x ATR Stop Loss, 3.0x ATR Take Profit)
                    sl_price = current_price - (1.5 * atr_14)
                    tp_price = current_price + (3.0 * atr_14)

                    # 4. Render the Output
                    st.success(f"**Live {ticker_sym} Price:** \\${current_price:.2f} | **14-Day ATR:** \\${atr_14:.2f}")

                    # --- NEW: POST-HEDGE IMPACT WIDGET ---
                    st.markdown("#### ⚖️ Post-Hedge Impact" if not is_zh else "#### ⚖️ 对冲后影响")
                    imp1, imp2, imp3 = st.columns(3)
                    imp1.metric("New Portfolio Beta" if not is_zh else "新投资组合 Beta", f"{new_beta:.2f}", f"{new_beta - current_beta:.2f} Reduction" if not is_zh else f"降低 {abs(new_beta - current_beta):.2f}", delta_color="inverse")
                    imp2.metric("New Net Delta (Risk $)" if not is_zh else "新净 Delta (风险资金)", f"${new_delta:,.0f}", f"${hedge_delta:,.0f} from Hedge" if not is_zh else f"对冲产生 ${hedge_delta:,.0f}", delta_color="inverse")
                    imp3.metric("Effective Hedge %" if not is_zh else "有效对冲 %", f"{effective_hedge_pct:.1f}%", "Of Market Risk Covered" if not is_zh else "覆盖的市场风险比例", delta_color="off")

                    st.markdown("#### 🎯 Execution Orders" if not is_zh else "#### 🎯 执行订单")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Shares to Buy" if not is_zh else "买入股数", f"{shares_to_buy:.2f}")
                    rc2.metric("Stop Loss" if not is_zh else "止损价", f"${sl_price:.2f}", "-1.5x ATR", delta_color="inverse")
                    rc3.metric("Take Profit" if not is_zh else "止盈价", f"${tp_price:.2f}", "+3.0x ATR")

                    # --- FIXED: ESCAPED DOLLAR SIGNS FOR CLEAN MARKDOWN ---
                    with st.expander("🎓 Quant Logic: Why did the math do this?" if not is_zh else "🎓 量化逻辑：为什么这么计算？"):
                        st.markdown(f"""
                        **1. Why Beta matters:** Your raw portfolio is **\\${port_value:,.0f}**, but because your Beta is **{current_beta:.2f}**, your portfolio moves slower than the market. Your actual "Market Risk" (Delta) is only **\\${current_delta:,.0f}**.

                        **2. The Power of Leverage:** By buying a {abs(leverage)}x leveraged inverse ETF, your **\\${available_cash:,.0f}** of cash creates **\\${abs(hedge_delta):,.0f}** of short pressure against the market, dragging your overall Beta down to **{new_beta:.2f}**.

                        **3. The Volatility Exits:** Inverse ETFs decay over time. The algorithm sets a Stop Loss using **1.5x Average True Range (ATR)** to avoid getting stopped out by random daily noise, and a **3.0x ATR Take Profit** to systematically lock in gains if the market crashes.
                        """ if not is_zh else f"""
                        **1. 为什么 Beta 很重要：** 您的原始投资组合为 **\\${port_value:,.0f}**，但由于您的 Beta 为 **{current_beta:.2f}**，您的投资组合波动慢于市场。您的实际“市场风险”(Delta) 仅为 **\\${current_delta:,.0f}**。

                        **2. 杠杆的力量：** 通过买入 {abs(leverage)}x 杠杆反向 ETF，您的 **\\${available_cash:,.0f}** 现金对市场产生了 **\\${abs(hedge_delta):,.0f}** 的做空压力，将您的整体 Beta 降至 **{new_beta:.2f}**。

                        **3. 波动率退出点：** 反向 ETF 随时间衰减。算法使用 **1.5x 平均真实波动幅 (ATR)** 设置止损，以避免被随机的日常噪音震出，并使用 **3.0x ATR 止盈** 在市场崩盘时系统地锁定收益。
                        """)

                else:
                    st.error("Could not fetch data for the selected instrument." if not is_zh else "无法获取所选工具的数据。")
            except Exception as e:
                st.error(f"Calculation failed: {e}")

# --- INDEPENDENT HEDGE CALCULATOR ---
st.markdown("---")
st.subheader(t["hedge_calc"])
st.info(t["hedge_info"] + (" (Enter 0 contracts if you are currently unhedged to calculate how many you WOULD need)." if not is_zh else " (如果您当前没有对冲，请输入 0 张合约，以计算您需要多少张)。"))

hedge_type = st.selectbox(t["strat_type"], ["Put Debit Spread", "Outright Long Put", "Short Micro-Futures"])

with st.form(key="hedge_evaluator_form", border=True):
    hc_col1, hc_col2, hc_col3 = st.columns([1, 1, 1])

    with hc_col1:
        h_tick = st.text_input(t["underlying"], value="QQQ").upper()
        # CHANGED: min_value is now 0, default is 0.
        h_qty = st.number_input(t["contracts"], min_value=0, value=0)

    with hc_col2:
        if hedge_type != "Short Micro-Futures":
            exp_date = st.date_input(t["exp_date"], value=date.today() + pd.Timedelta(days=14))
        else:
            st.caption("Futures have linear delta; expiry doesn't affect notional hedge." if not is_zh else "期货具有线性德尔塔；到期日不影响名义对冲。")
            exp_date = date.today()

    with hc_col3:
        if hedge_type in ["Put Debit Spread", "Outright Long Put"]:
            h_long_k = st.number_input("Long Put Strike ($)" if not is_zh else "看跌期权买入行权价 ($)", min_value=1.0, value=420.0)
        else:
            h_long_k = 0.0

        if hedge_type == "Put Debit Spread":
            h_short_k = st.number_input("Short Put Strike ($)" if not is_zh else "看跌期权卖出行权价 ($)", min_value=1.0, value=410.0)
        else:
            h_short_k = 0.0

    calc_hedge = st.form_submit_button(t["calc_btn"], use_container_width=True)

if calc_hedge:
    try:
        h_stock = yf.Ticker(h_tick)
        S_h = h_stock.fast_info['last_price']

        days_to_exp = max((exp_date - date.today()).days, 1)

        if hedge_type == "Short Micro-Futures":
            multiplier = 50 if "SPY" in h_tick else (80 if "QQQ" in h_tick else 50)
            net_contract_delta = -1.0 * multiplier
            total_hedge_dollars = abs(net_contract_delta) * S_h * h_qty
        else:
            h_hist = h_stock.history(period="3mo")['Close']
            sigma_h = (np.log(h_hist / h_hist.shift(1)).std() * np.sqrt(252))
            T_h = days_to_exp / 365.0

            long_put = get_greeks(S_h, h_long_k, T_h, 0.045, sigma_h, 'put')
            long_delta = long_put['Delta']

            short_delta = 0.0
            if hedge_type == "Put Debit Spread":
                short_put = get_greeks(S_h, h_short_k, T_h, 0.045, sigma_h, 'put')
                short_delta = -short_put['Delta']

            net_contract_delta = long_delta + short_delta
            total_hedge_dollars = abs(net_contract_delta) * 100 * S_h * h_qty

        net_unhedged = beta_adj_exposure - total_hedge_dollars
        coverage_pct = (total_hedge_dollars / beta_adj_exposure) * 100 if beta_adj_exposure > 0 else 100

        st.markdown("#### 📉 Diagnosis Results" if not is_zh else "#### 📉 诊断结果")
        with st.container(border=True):
            r1, r2, r3 = st.columns(3)
            r1.metric("Strategy Net Delta" if not is_zh else "策略净德尔塔", f"{net_contract_delta:.2f}", "Per Contract equivalent" if not is_zh else "每张合约等值")
            r2.metric("Total Dollar Protection" if not is_zh else "总美元保护", f"${total_hedge_dollars:,.0f}", "Capital Covered" if not is_zh else "覆盖资本")
            r3.metric("Coverage Ratio" if not is_zh else "覆盖率", f"{coverage_pct:.1f}%", "Percentage of Portfolio Safe" if not is_zh else "投资组合安全百分比", delta_color="normal" if coverage_pct > 50 else "inverse")

        contracts_needed = 0
        if net_contract_delta != 0 and net_unhedged > 0:
            contracts_needed = int(np.ceil(net_unhedged / (abs(net_contract_delta) * 100 * S_h)))

        gap_multiplier = 80 if "QQQ" in h_tick else 50
        gap_futures = int(np.ceil(net_unhedged / (S_h * gap_multiplier)))

        # Explicitly handling the 0 contracts / completely unhedged scenario
        if h_qty == 0:
            st.error(f"**VERDICT: COMPLETELY UNHEDGED.** You have **${net_unhedged:,.0f}** of naked downside. To fully neutralize this using the strategy above, you need to buy **{contracts_needed} contracts**." if not is_zh else f"**结论: 完全无对冲。** 您有 **${net_unhedged:,.0f}** 的裸露下行风险。要使用上述策略完全中和此风险，您需要买入 **{contracts_needed} 张合约**。")
        elif coverage_pct < 30:
            st.error(f"**VERDICT: SEVERELY UNDER-HEDGED.** This covers **${total_hedge_dollars:,.0f}**, but your risk is **${beta_adj_exposure:,.0f}**. You have **${net_unhedged:,.0f}** of naked downside. Buy **{contracts_needed} more contracts** to close the gap." if not is_zh else f"**结论: 严重对冲不足。** 此仓位覆盖了 **${total_hedge_dollars:,.0f}**，但您的风险为 **${beta_adj_exposure:,.0f}**。您有 **${net_unhedged:,.0f}** 的裸露下行风险。再买入 **{contracts_needed} 张合约** 以填补缺口。")
        elif coverage_pct < 80:
            st.warning(f"**VERDICT: PARTIALLY HEDGED.** You have covered **{coverage_pct:.1f}%** of your risk. You still have **${net_unhedged:,.0f}** of naked downside. (Needs **{contracts_needed} more contracts** for 100% coverage)." if not is_zh else f"**结论: 部分对冲。** 您已覆盖 **{coverage_pct:.1f}%** 的风险。您仍有 **${net_unhedged:,.0f}** 的裸露下行风险。(需再买入 **{contracts_needed} 张合约** 以达到100%覆盖)。")
        else:
            st.success(f"**VERDICT: PROPERLY HEDGED.** You have neutralized almost all directional risk. Your portfolio is insulated." if not is_zh else f"**结论: 对冲充分。** 您已中和几乎所有的方向性风险。您的投资组合很安全。")

    except Exception as e:
        st.error(f"Could not calculate hedge. Ensure the ticker is valid. Error: {e}")
