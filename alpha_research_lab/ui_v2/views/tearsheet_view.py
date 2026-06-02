import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import TEXT

class TearSheetView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(self, run_id: str, run_metadata: pd.Series, lang: str = "EN"):
        t = TEXT[lang]
        
        # 1. Fetch Cleaned Data
        df = self.dm.get_run_returns(run_id)
        if df.empty:
            st.warning(t["ts_no_returns"])
            return

        days = df['date']
        daily_returns = df['net_return'].values
        daily_returns_bench = df['benchmark_return'].values
        daily_leverage = df['portfolio_leverage'].values
        avg_turnover = df['daily_turnover'].mean() * 100

        # 2. Risk Math (Strategy)
        cum_returns = (1 + daily_returns).cumprod()
        ann_return = (cum_returns[-1]) ** (252 / len(days)) - 1 if len(days) > 0 else 0
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # 3. Risk Math (Benchmark)
        cum_returns_bench = (1 + daily_returns_bench).cumprod()
        ann_ret_bench = (cum_returns_bench[-1]) ** (252 / len(days)) - 1 if len(days) > 0 else 0
        ann_vol_bench = np.std(daily_returns_bench) * np.sqrt(252)
        sharpe_bench = ann_ret_bench / ann_vol_bench if ann_vol_bench != 0 else 0
        
        rolling_max_bench = np.maximum.accumulate(cum_returns_bench)
        drawdown_bench = (cum_returns_bench - rolling_max_bench) / rolling_max_bench
        max_dd_bench = np.min(drawdown_bench) if len(drawdown_bench) > 0 else 0
        calmar_bench = ann_ret_bench / abs(max_dd_bench) if max_dd_bench != 0 else 0

        # --- EXTRACT NEW METRICS ---
        holdout_ic = run_metadata.get('holdout_ic', 0.0)
        total_trades = run_metadata.get('total_trades', 'N/A')

        # --- RENDER TOP DASHBOARD ---
        st.markdown(t["strategy_net"])
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(t["risk_metrics"][0], f"{ann_return * 100:.2f}%")
        r2.metric(t["risk_metrics"][1], f"{ann_vol * 100:.2f}%")
        r3.metric(t["risk_metrics"][2], f"{sharpe:.2f}")
        r4.metric(t["risk_metrics"][3], f"{max_dd * 100:.2f}%")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric(t["risk_metrics"][4], f"{calmar:.2f}")
        r6.metric(t["avg_turnover"], f"{avg_turnover:.1f}%")
        r7.metric(t["holdout_ic"], f"{holdout_ic:.4f}")
        r8.metric(t["total_trades"], f"{total_trades}")

        st.markdown(t["benchmark_eq"])
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(t["risk_metrics"][0], f"{ann_ret_bench * 100:.2f}%")
        b2.metric(t["risk_metrics"][1], f"{ann_vol_bench * 100:.2f}%")
        b3.metric(t["risk_metrics"][2], f"{sharpe_bench:.2f}")
        b4.metric(t["risk_metrics"][3], f"{max_dd_bench * 100:.2f}%")

        b5, b6, b7, b8 = st.columns(4)
        b5.metric(t["risk_metrics"][4], f"{calmar_bench:.2f}")
        b6.metric(t["avg_turnover"], "Passive")
        b7.metric(t["holdout_ic"], "N/A")
        b8.metric(t["total_trades"], "Passive")

        st.markdown("---")

        # --- RENDER AI CO-PILOT PROMPT ---
        self._render_copilot(run_metadata, ann_return, ann_vol, sharpe, max_dd, calmar, t)
        st.markdown("---")

        # --- RENDER MAIN PLOTLY CHARTS ---
        self._render_charts(days, cum_returns, cum_returns_bench, drawdown, daily_leverage, daily_returns, ann_vol, t)

    def _render_copilot(self, meta, ann_ret, ann_vol, sharpe, max_dd, calmar, t):
        st.markdown(t["copilot_title"])
        
        failure_code = meta.get('failure_code', 'NONE')
        suggested_action = meta.get('suggested_action', 'N/A')
        
        metrics_text = f"""Current Portfolio Metrics:
- Validation IC (In-Sample): {meta['validation_ic']:.4f}
- Holdout IC (Out-of-Sample): {meta['holdout_ic']:.4f}
- Annualized Return: {ann_ret * 100:.2f}%
- Annualized Volatility: {ann_vol * 100:.2f}%
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd * 100:.2f}%
- Calmar Ratio: {calmar:.2f}"""

        if pd.isna(failure_code) or failure_code == "NONE":
            st.success(t["factor_passed"])
            prompt = f"I am optimizing the institutional CTA factor '{meta['name']}'.\n\nIt successfully passed the Rank IC tests. Here are the exact simulation results:\n\n{metrics_text}\n\nBased purely on the quantitative metrics above, please rewrite the Polars `compute(df)` function to optimize its risk-adjusted performance. Diagnose its weaknesses and introduce mathematical controls to improve execution without destroying the Holdout IC."
        else:
            st.error(f"{t['factor_failed']}: [{failure_code}]")
            prompt = f"I am backtesting the institutional CTA factor '{meta['name']}'.\n\nThe Alpha Pipeline rejected the latest iteration with the following diagnostic code: `{failure_code}`.\nThe automated engine's suggested action is: \"{suggested_action}\"\n\n{metrics_text}\n\nPlease rewrite the Polars `compute(df)` function to address this exact mathematical failure. Fix the execution flaw to improve the Out-of-Sample performance."

        st.code(prompt, language="markdown")

    def _render_charts(self, days, cum_returns, cum_returns_bench, drawdown, daily_leverage, daily_returns, ann_vol, t):
        # 1. Main Tear Sheet Grid
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Scatter(x=days, y=(cum_returns - 1) * 100, name="Net Strategy", line=dict(color='#00E676', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=days, y=(cum_returns_bench - 1) * 100, name="Benchmark", line=dict(color='#FFCA28', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=days, y=drawdown * 100, name="Drawdown", fill='tozeroy', line=dict(color='#D32F2F', width=1), fillcolor='rgba(211, 47, 47, 0.3)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=days, y=daily_leverage, name="Gross Leverage", fill='tozeroy', line=dict(color='#2196F3', width=1), fillcolor='rgba(33, 150, 243, 0.2)'), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

        # 2. Monthly Heatmap & Volatility Matrix
        df_monthly = pd.DataFrame({'date': days, 'return': daily_returns})
        df_monthly['year'] = df_monthly['date'].dt.year
        df_monthly['month'] = df_monthly['date'].dt.month
        monthly_pivot = df_monthly.groupby(['year', 'month'])['return'].apply(lambda x: (1 + x).prod() - 1).reset_index().pivot(index='year', columns='month', values='return').fillna(0)
        for m in range(1, 13):
            if m not in monthly_pivot.columns: monthly_pivot[m] = 0.0
        monthly_pivot = monthly_pivot[range(1, 13)]

        df_vol = pd.DataFrame({'date': days, 'rolling_vol': df_monthly['return'].rolling(30).std() * np.sqrt(252)})

        c_heat, c_vol = st.columns([1.5, 1])
        with c_heat:
            st.markdown(f"#### {t.get('heatmap_title', '🗓️ Monthly Return Matrix')}")
            fig_hm = go.Figure(data=go.Heatmap(z=monthly_pivot.values * 100, x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], y=monthly_pivot.index, colorscale='RdYlGn', zmid=0, text=np.round(monthly_pivot.values * 100, 1), texttemplate="%{text}%", hoverinfo="z"))
            fig_hm.update_layout(template="plotly_dark", height=350, yaxis=dict(autorange="reversed", type='category'), margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hm, use_container_width=True)
            with st.expander(t.get('heatmap_help_title', '💡 Insight')): st.markdown(t.get('heatmap_help', ''))
            
        with c_vol:
            st.markdown(f"#### {t.get('vol_title', '📉 Rolling 30-Day Volatility')}")
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=df_vol['date'], y=df_vol['rolling_vol'] * 100, mode='lines', line=dict(color='#FF9800', width=2), fill='tozeroy', fillcolor='rgba(255, 152, 0, 0.2)'))
            fig_v.add_hline(y=ann_vol*100, line_dash="dash", line_color="gray")
            fig_v.update_layout(template="plotly_dark", height=350, yaxis_title="Ann. Volatility (%)", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_v, use_container_width=True)
            with st.expander(t.get('vol_help_title', '💡 Insight')): st.markdown(t.get('vol_help', ''))