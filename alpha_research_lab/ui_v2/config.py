import os

# ==========================================
# GLOBAL SETTINGS
# ==========================================
APP_TITLE = "Alpha Mine Lab"
APP_ICON = "🔬"
LAYOUT = "wide"

# Custom CSS to shrink metric font sizes so they fit neatly in columns
CUSTOM_CSS = """
<style>
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; 
}
</style>
"""

# Automatically find the database one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "research_memory.db")
LOGS_DIR = os.path.join(BASE_DIR, "execution_logs")


def get_theme_css(theme_mode: str) -> str:
    """Shared UI theme CSS for all Streamlit pages."""
    if theme_mode == "DARK":
        return """
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
[data-testid="stSidebar"] { background-color: #151A24; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #FAFAFA !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #FAFAFA !important; }
.stDataFrame, .stTable { color: #FAFAFA !important; }
</style>
"""
    return """
<style>
.stApp { background-color: #FFFFFF; color: #111827; }
[data-testid="stSidebar"] { background-color: #F8FAFC; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111827 !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #111827 !important; }
.stDataFrame, .stTable { color: #111827 !important; }
</style>
"""


def get_plotly_template(theme_mode: str) -> str:
    return "plotly_dark" if theme_mode == "DARK" else "plotly_white"

# ==========================================
# TRANSLATION DICTIONARY (Fully Restored)
# ==========================================
TEXT = {
    "EN": {
        "title": "🔬 Factor Research & Diagnostics",
        "ledger": "## 📜 Run Ledger",
        "ledger_desc": "Historical backtest iterations.",
        "db_empty": "Database empty or not found. Run the Evaluator first!",
        "metrics": ["Total Experiments", "Best Holdout IC", "Candidate Factors", "Failure Rate"],
        "matrix_title": "### 🗄️ Candidate Matrix",
        "cols": {"Factor": "Factor", "Round": "Round", "Val_IC": "Val_IC", "Holdout_IC": "Holdout_IC", "Diagnostics": "Diagnostics", "Next_Step": "Next_Step"},
        "chart_title": "### 📈 Institutional Risk Analytics (Tear Sheet)",
        "chart_desc": "Out-of-sample portfolio simulation with drawdown constraints.",
        "trace_strat": "Factor Portfolio",
        "trace_bench": "Benchmark (SPY)",
        "risk_metrics": ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"],
        "glossary_title": "📖 Metric Definitions & Institutional Bounds",
        "glossary": {
            "Information Coefficient (IC)": "Measures the correlation between factor scores and future returns.<br>• <b>< 2%:</b> Noise<br>• <b>2% - 5%:</b> Tradeable Alpha<br>• <b>> 10%:</b> Warning (Possible Lookahead Bias)",
            "Sharpe Ratio": "Return per unit of volatility.<br>• <b>< 1.0:</b> Poor<br>• <b>1.0 - 2.0:</b> Good<br>• <b>> 3.0:</b> Warning (Check transaction costs)",
            "Max Drawdown": "Worst peak-to-trough loss. CTAs strictly monitor this due to leverage.<br>• <b>< 10%:</b> Excellent<br>• <b>> 20%:</b> Un-tradeable with high leverage",
            "Calmar Ratio": "Annualized Return divided by Max Drawdown (Return per unit of pain).<br>• <b>< 1.0:</b> Too risky<br>• <b>> 2.0:</b> Exceptional"
        }, 
        "tab_tearsheet": "📊 Tear Sheet",
        "tab_correlation": "🔗 Correlation Matrix",
        "corr_title": "Factor Orthogonalization (Return Correlation)",
        "corr_desc": "Checking for multicollinearity across historical factor returns. High correlation (>0.7) indicates duplicate risk.", 
        "heatmap_title": "🗓️ Monthly Return Matrix",
        "vol_title": "📉 Rolling 30-Day Volatility",
        "heatmap_help_title": "💡 How to read the Heatmap",
        "heatmap_help": "Look for **Consistency** over **Magnitude**. A robust factor should have a balanced mix of green months. If all profits are clustered in a single year or a specific month (e.g., only making money in March 2020), the model is likely overfit to a specific macro event rather than capturing true alpha.",
        "vol_help_title": "💡 How to read Rolling Volatility",
        "vol_help": "This tracks how much the strategy's risk fluctuates over time. Because our Execution Desk uses **Volatility Scaling**, this line should ideally remain relatively flat. Massive spikes indicate that the strategy lost control of its risk sizing during a market shock.",
        "dna_pnl_title": "💡 How to read PnL & Entry/Exit",
        "dna_pnl": "**PnL Distribution (Top Left):** Look at the tails. If the green (Loss) bars stretch much further to the left than the red (Win) bars stretch to the right, your losers are bigger than your winners (Negative Skew).<br><br>**Entry vs Exit (Bottom Left):** The dashed line is the break-even point. If you see massive clusters far away from the line, it means your holding periods or stop-losses are too loose.",
        "dna_time_title": "💡 How to read Holding Time",
        "dna_time": "**Holding Time (Top Right):** StatArb models should have tight, consistent holding times (e.g., 24-48 hours). If you have a 'fat tail' of trades held for hundreds of hours, the model is trapped in a non-reverting spread.<br><br>**PnL vs Time (Bottom Right):** The holy grail is 'Cut losses early, let profits run.' If your largest red bubbles (losses) are clustered on the far right (long holding times), your model is stubbornly holding onto bad trades.",
        "tab_dna": "🧬 Strategy DNA",
        "tab_pareto": "📈 Pareto Frontier",
        "tab_ml": "🧠 ML Feature Importance",
        "tab_conditional_perf": "🏛️ Conditional Strategy Performance",
        "exec_dd_title": "### 📉 Drawdown Savings from Regime Model",
        "exec_dd_desc": "Compares baseline SMA/Bollinger max drawdown versus the final Layer 4 Router.",
        "exec_to_title": "### 🔁 Turnover Evolution (Binary → Continuous → Discretized)",
        "exec_to_desc": "Quantifies execution drag reduction through continuous math and discretization.",
        "exec_missing": "Insufficient run history for one or more required factors. Please run the baseline and Layer 4 backtests first.",
        "select_run_hint": "👈 Please select a completed backtest run from the Ledger to view detailed analytics.",
        "ml_caption": "Visualizing the internal decision-making weights of the Machine Learning model.",
        "ml_missing": "ℹ️ No Feature Importance data found. This tab will populate when you select an ML-generated run.",
        "ts_no_returns": "⚠️ No return data found for this run. It may have failed execution.",
        "strategy_net": "**🟢 Strategy (Net of Fees)**",
        "benchmark_eq": "**🟡 Benchmark (Equal-Weight)**",
        "avg_turnover": "Avg Turnover",
        "holdout_ic": "Holdout IC",
        "total_trades": "Total Trades",
        "copilot_title": "### 🤖 Alpha Architect Co-Pilot (Prompt Generator)",
        "factor_passed": "🟢 Factor Passed! Use the prompt below to ask your agent for risk-adjusted optimization.",
        "factor_failed": "🔴 Factor Failed",
        "leaderboard_title": "### 🏆 Alpha Leaderboard",
        "corr_select_factors": "🔍 Select Factors to Compare",
        "corr_select_min": "👆 Please select at least two factors from the dropdown above to view the correlation matrix.",
        "corr_waiting": "⏳ Waiting for more factors... You need at least 2 successful backtests to generate a correlation matrix.",
        "dna_trade_caption": "Trade-Level Analytics (StatArb / Pairs)",
        "dna_portfolio_caption": "Cross-Sectional Daily Net PnL Frequency (Portfolio-Level)",
        "dna_error": "⚠️ Error rendering trade analytics",
        "dna_no_returns": "⚠️ No return data found for this run.",
        "pareto_caption": "Visualizing the execution boundary of this factor across all historical iterations.",
        "pareto_help": "💡 How to read this: The optimal iterations are in the Top-Right corner (High Return, Low Drawdown). Color indicates Turnover. Bubble size indicates Out-of-Sample IC.",
        "pareto_error": "⚠️ Could not render Pareto Frontier due to data inconsistency",
        "pareto_missing": "⚠️ Not enough data. Run this factor at least two times to generate a Pareto Frontier.",
        "asset_zoo_title": "🦁 The Asset Zoo: Market Microstructure Clustering",
        "asset_zoo_subtitle": "Gaussian Mixture Model (GMM) clustering of asset micro-personalities to determine strategy targeting.",
        "asset_zoo_missing": "❌ Profile data not found. Please run `data_engine/asset_profiler.py` first.",
        "asset_cluster_dist": "### 🧬 Cluster Distribution",
        "asset_tier1": "Tier 1 (Trend / Breakout)",
        "asset_tier2": "Tier 2 (Mean-Reverting)",
        "asset_tier3": "Tier 3 (Toxic / Illiquid)",
        "asset_radar": "### 🔭 Market Radar (Risk vs. Trendiness)",
        "asset_target_board": "### 🎯 Strategy Targeting Board",
        "asset_select_target": "Select Strategy Target:",
        "asset_copy_list": "**Copy-Paste Target Universe (Python List):**",
        "exec_page_title": "🏛️ Executive Dashboard",
        "exec_page_subtitle": "Conditional Strategy Performance",
        "regime_page_title": "🏛️ Market Regime Characterization",
        "regime_page_subtitle": "Visualizing HMM Continuous Probabilities over Asset Prices.",
        "regime_select_asset": "Select Asset to Overlay:",
    },
    "ZH": {
        "title": "🔬 因子投研与诊断系统",
        "ledger": "## 📜 运行记录 (Ledger)",
        "ledger_desc": "历史回测迭代记录。",
        "db_empty": "未找到数据库。请先运行评估器 (Evaluator)！",
        "metrics": ["实验总数", "最佳样本外 IC", "候选因子数量", "失败率"],
        "matrix_title": "### 🗄️ 候选因子矩阵",
        "cols": {"Factor": "因子名称", "Round": "迭代轮次", "Val_IC": "样本内 IC", "Holdout_IC": "样本外 IC", "Diagnostics": "诊断结果", "Next_Step": "下一步建议"},
        "chart_title": "### 📈 机构级风控分析 (Tear Sheet)",
        "chart_desc": "受回撤约束的样本外投资组合模拟。",
        "trace_strat": "因子投资组合",
        "trace_bench": "基准 (沪深300 / SPY)",
        "risk_metrics": ["年化收益率", "年化波动率", "夏普比率", "最大回撤", "卡玛比率"], 
        "glossary_title": "📖 指标定义与机构级标准",
        "glossary": {
            "信息系数 (IC)": "衡量因子得分与未来收益的相关性。<br>• <b>< 2%:</b> 噪音<br>• <b>2% - 5%:</b> 优质可交易 Alpha<br>• <b>> 10%:</b> 警告 (极可能存在未来函数)",
            "夏普比率 (Sharpe)": "每单位波动率带来的超额收益。<br>• <b>< 1.0:</b> 较差<br>• <b>1.0 - 2.0:</b> 优秀<br>• <b>> 3.0:</b> 警告 (需检查是否遗漏手续费/滑点)",
            "最大回撤 (MDD)": "资产从最高点到最低点的最大跌幅。CTA 基金因杠杆原因对此要求极严。<br>• <b>< 10%:</b> 极佳<br>• <b>> 20%:</b> 无法在高杠杆下交易",
            "卡玛比率 (Calmar)": "年化收益率除以最大回撤 (每承担一单位回撤带来的收益)。<br>• <b>< 1.0:</b> 风险过高<br>• <b>> 2.0:</b> 表现优异"
        }, 
        "tab_tearsheet": "📊 风控分析 (Tear Sheet)",
        "tab_correlation": "🔗 相关性矩阵 (Correlation Matrix)",
        "corr_title": "因子正交化 (收益率相关性检测)",
        "corr_desc": "检测历史因子收益率之间的共线性。高相关性 (>0.7) 意味着重叠的风险敞口。",
        "heatmap_title": "🗓️ 月度收益率热力图",
        "vol_title": "📉 滚动 30 天年化波动率",
        "heatmap_help_title": "💡 如何解读热力图",
        "heatmap_help": "寻找**一致性**而非**高收益**。一个稳健的因子应该在不同年份均匀地分布着绿色的盈利月份。如果所有的利润都集中在某一年或某个月（例如只在2020年3月赚钱），模型很可能是对特定宏观事件的过度拟合，而不是捕捉到了真正的 Alpha。",
        "vol_help_title": "💡 如何解读滚动波动率",
        "vol_help": "这追踪了策略风险随时间的波动情况。因为我们的执行模块使用了**波动率缩放 (Volatility Scaling)**，这条线理想情况下应该保持相对平稳。巨大的尖峰表明策略在市场冲击期间失去了对风险敞口的控制。",
        "dna_pnl_title": "💡 如何解读盈亏分布与价位散点",
        "dna_pnl": "**盈亏分布 (左上):** 关注尾部特征。如果绿色（亏损）柱形向左延伸的距离，远大于红色（盈利）柱形向右延伸的距离，说明单笔亏损远大于单笔盈利（负偏度）。<br><br>**开平仓价位 (左下):** 虚线是盈亏平衡点。如果你看到大量散点远离这条虚线，说明你的持仓周期或止损设置过于宽松。",
        "dna_time_title": "💡 如何解读持仓时间分布",
        "dna_time": "**持仓时间 (右上):** 统计套利模型应该具有紧凑、一致的持仓时间（例如 24-48 小时）。如果你发现有一个长尾分布，包含持仓数百小时的交易，说明模型被困在了一个不回归的价差中。<br><br>**盈亏 vs 时间 (右下):** 交易的圣杯是“截断亏损，让利润奔跑”。如果你最大的红色气泡（亏损）集中在图表的最右侧（长持仓时间），说明你的模型在顽固地死扛亏损单。",
        "tab_dna": "🧬 策略 DNA",
        "tab_pareto": "📈 帕累托前沿 (Pareto)",
        "tab_ml": "🧠 机器学习特征重要性",
        "tab_conditional_perf": "🏛️ 条件策略绩效",
        "exec_dd_title": "### 📉 状态模型带来的回撤节省",
        "exec_dd_desc": "对比基线 SMA/Bollinger 与最终 Layer 4 Router 的最大回撤。",
        "exec_to_title": "### 🔁 换手率演化（二值 → 连续 → 离散）",
        "exec_to_desc": "量化连续化与离散化对执行摩擦损耗的改善。",
        "exec_missing": "缺少关键回测记录。请先运行基线与 Layer 4 回测。",
        "select_run_hint": "👈 请先从左侧运行记录中选择一个已完成的回测以查看详细分析。",
        "ml_caption": "可视化机器学习模型内部决策权重。",
        "ml_missing": "ℹ️ 未找到特征重要性数据。选择由机器学习模型生成的回测后将自动显示。",
        "ts_no_returns": "⚠️ 未找到该次运行的收益数据，可能执行失败。",
        "strategy_net": "**🟢 策略（净值，含费用）**",
        "benchmark_eq": "**🟡 基准（等权）**",
        "avg_turnover": "平均换手率",
        "holdout_ic": "样本外 IC",
        "total_trades": "总交易笔数",
        "copilot_title": "### 🤖 Alpha 架构师副驾驶（Prompt 生成器）",
        "factor_passed": "🟢 因子已通过！可使用下方提示词让智能体继续做风险调整优化。",
        "factor_failed": "🔴 因子未通过",
        "leaderboard_title": "### 🏆 Alpha 排行榜",
        "corr_select_factors": "🔍 选择要对比的因子",
        "corr_select_min": "👆 请至少选择两个因子以查看相关性矩阵。",
        "corr_waiting": "⏳ 等待更多因子... 至少需要 2 个成功回测才能生成相关性矩阵。",
        "dna_trade_caption": "交易级别分析（统计套利 / 配对）",
        "dna_portfolio_caption": "横截面日度净值盈亏频率（组合级）",
        "dna_error": "⚠️ 交易分析渲染失败",
        "dna_no_returns": "⚠️ 未找到该次运行的收益数据。",
        "pareto_caption": "可视化该因子在历史迭代中的执行边界。",
        "pareto_help": "💡 解读：最优迭代位于右上角（高收益、低回撤）。颜色代表换手率，气泡大小代表样本外 IC。",
        "pareto_error": "⚠️ 帕累托前沿渲染失败，数据可能不一致",
        "pareto_missing": "⚠️ 数据不足。请至少运行该因子两次以生成帕累托前沿。",
        "asset_zoo_title": "🦁 资产动物园：市场微观结构聚类",
        "asset_zoo_subtitle": "使用高斯混合模型（GMM）对资产微观特征聚类，用于策略定向。",
        "asset_zoo_missing": "❌ 未找到画像数据。请先运行 `data_engine/asset_profiler.py`。",
        "asset_cluster_dist": "### 🧬 聚类分布",
        "asset_tier1": "第一层（趋势 / 突破）",
        "asset_tier2": "第二层（均值回归）",
        "asset_tier3": "第三层（高毒性 / 低流动性）",
        "asset_radar": "### 🔭 市场雷达（风险 vs 趋势效率）",
        "asset_target_board": "### 🎯 策略定向看板",
        "asset_select_target": "选择策略目标：",
        "asset_copy_list": "**可复制目标池（Python 列表）：**",
        "exec_page_title": "🏛️ 高管看板",
        "exec_page_subtitle": "条件策略绩效",
        "regime_page_title": "🏛️ 市场状态刻画与分析",
        "regime_page_subtitle": "可视化 HMM 连续状态概率与价格叠加。",
        "regime_select_asset": "选择要叠加的资产：",
    }
}