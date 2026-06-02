import os
import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar

st.set_page_config(page_title="Regime Characterization", layout="wide")

init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = "English" if st.session_state.lang == "EN" else "中文"

text = {
    "title": {"English": "🏛️ Market Regime Characterization", "中文": "🏛️ 市场状态刻画与分析"},
    "subtitle": {"English": "Visualizing HMM Continuous Probabilities over Asset Prices.", "中文": "基于隐马尔可夫模型 (HMM) 的连续状态概率与资产价格叠加视图。"},
    "select_asset": {"English": "Select Asset to Overlay:", "中文": "选择要叠加的资产："},
    "price": {"English": "Close Price", "中文": "收盘价"},
    "regime_0": {"English": "High-Beta Bull (Regime 0)", "中文": "高波动牛市 (状态 0)"},
    "regime_1": {"English": "Panic / Chop (Regime 1)", "中文": "恐慌 / 剧烈震荡 (状态 1)"},
    "regime_2": {"English": "Quiet / Normal (Regime 2)", "中文": "平稳 / 正常常规 (状态 2)"},
    "y_price": {"English": "Price", "中文": "价格"},
    "y_prob": {"English": "Probability", "中文": "概率"},
}

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    regimes = pl.scan_parquet(os.path.join(base, "Macro_Regimes.parquet")).collect().to_pandas()
    prices = pl.scan_parquet(os.path.join(base, "ML_Stacked_Matrix.parquet")).select(["date", "ticker", "close"]).collect().to_pandas()
    regimes["date"] = pd.to_datetime(regimes["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    return regimes, prices

st.title(text["title"][lang])
st.markdown(text["subtitle"][lang])

regimes, prices = load_data()
selected_ticker = st.selectbox(text["select_asset"][lang], prices["ticker"].unique())
asset_prices = prices[prices["ticker"] == selected_ticker].sort_values("date")
merged_data = pd.merge(asset_prices, regimes, on="date", how="inner")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Scatter(x=merged_data["date"], y=merged_data["close"], mode="lines", name=text["price"][lang], line=dict(color="#2C3E50", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=merged_data["date"], y=merged_data["prob_regime_2"], mode="lines", stackgroup="one", name=text["regime_2"][lang], fillcolor="rgba(0, 200, 100, 0.5)", line=dict(width=0)), row=2, col=1)
fig.add_trace(go.Scatter(x=merged_data["date"], y=merged_data["prob_regime_0"], mode="lines", stackgroup="one", name=text["regime_0"][lang], fillcolor="rgba(50, 150, 255, 0.5)", line=dict(width=0)), row=2, col=1)
fig.add_trace(go.Scatter(x=merged_data["date"], y=merged_data["prob_regime_1"], mode="lines", stackgroup="one", name=text["regime_1"][lang], fillcolor="rgba(255, 50, 50, 0.5)", line=dict(width=0)), row=2, col=1)
fig.update_layout(height=700, hovermode="x unified", title_text=f"{selected_ticker} - {text['title'][lang]}")
fig.update_yaxes(title_text=text["y_price"][lang], row=1, col=1)
fig.update_yaxes(title_text=text["y_prob"][lang], range=[0, 1], tickformat=".0%", hoverformat=".2%", row=2, col=1)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
