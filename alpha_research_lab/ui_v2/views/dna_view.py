import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import TEXT

class DNAView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(self, run_id: str, lang: str = "EN"):
        t = TEXT[lang]
        
        st.markdown(f"### {t.get('tab_dna', '🧬 Strategy DNA')}")
        
        # 1. Fetch the Trade Ledger from the DataManager
        df_trades = self.dm.get_trade_ledger(run_id)
        
        # 2. Smart Routing: Trade Analytics vs. Portfolio Fallback
        if not df_trades.empty:
            st.caption(t["dna_trade_caption"])
            self._render_trade_analytics(df_trades, t)
        else:
            st.caption(t["dna_portfolio_caption"])
            self._render_portfolio_fallback(run_id, t)

    def _render_trade_analytics(self, df_trades: pd.DataFrame, t: dict):
        """Renders the 4-chart grid for models that produce discrete trades."""
        try:
            df_trades['trade_pnl_pct'] = df_trades['trade_pnl'] * 100
            
            # Chinese-standard colors: Red = Profit, Green = Loss
            color_map = {'Win': '#F44336', 'Loss': '#4CAF50'}
            
            c1, c2 = st.columns(2)
            
            with c1:
                # Chart 1: PnL Distribution
                fig1 = px.histogram(
                    df_trades, x="trade_pnl_pct", color="win_loss_flag",
                    color_discrete_map=color_map,
                    title="💰 PnL Distribution / 盈亏金额分布",
                    labels={"trade_pnl_pct": "Trade PnL (%)", "win_loss_flag": "Result"},
                    barmode="overlay"
                )
                fig1.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 3: Entry vs Exit Scatter
                fig3 = px.scatter(
                    df_trades, x="entry_price", y="exit_price", color="win_loss_flag",
                    color_discrete_map=color_map,
                    title="🎯 Entry vs Exit Price / 开平仓价位散点",
                    labels={"entry_price": "Entry Price", "exit_price": "Exit Price", "win_loss_flag": "Result"}
                )
                min_val = min(df_trades['entry_price'].min(), df_trades['exit_price'].min())
                max_val = max(df_trades['entry_price'].max(), df_trades['exit_price'].max())
                fig3.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
                fig3.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig3, use_container_width=True)
                
                # Educational Expander
                with st.expander(t.get('dna_pnl_title', '💡 Insight')):
                    st.markdown(t.get('dna_pnl', ''), unsafe_allow_html=True)
                    
            with c2:
                # Chart 2: Holding Time Distribution
                fig2 = px.histogram(
                    df_trades, x="holding_period_hours", color="win_loss_flag",
                    color_discrete_map=color_map,
                    title="⏳ Holding Time (Hours) / 持仓时间分布",
                    labels={"holding_period_hours": "Hours Held", "win_loss_flag": "Result"},
                    barmode="overlay"
                )
                fig2.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Chart 4: PnL vs Holding Time Scatter
                fig4 = px.scatter(
                    df_trades, x="holding_period_hours", y="trade_pnl_pct", 
                    color="trade_pnl_pct", size=df_trades["trade_pnl_pct"].abs(),
                    color_continuous_scale="RdYlGn", 
                    title="⏱️ PnL vs Holding Time / 盈亏 vs 持仓时间",
                    labels={"holding_period_hours": "Hours Held", "trade_pnl_pct": "PnL (%)"}
                )
                fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig4.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig4, use_container_width=True)
                
                # Educational Expander
                with st.expander(t.get('dna_time_title', '💡 Insight')):
                    st.markdown(t.get('dna_time', ''), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"{t['dna_error']}: {e}")

    def _render_portfolio_fallback(self, run_id: str, t: dict):
        """Renders the standard Daily PnL histogram for models without discrete trades."""
        df_returns = self.dm.get_run_returns(run_id)
        
        if df_returns.empty:
            st.warning(t["dna_no_returns"])
            return
            
        daily_returns = df_returns['net_return'].fillna(0).values
        winning_days = daily_returns[daily_returns > 0] * 100
        losing_days = daily_returns[daily_returns <= 0] * 100
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=winning_days, name='Profitable Days (盈利)', marker_color='#F44336'))
        fig_hist.add_trace(go.Histogram(x=losing_days, name='Losing Days (亏损)', marker_color='#4CAF50'))
        
        fig_hist.update_layout(
            barmode='overlay', template="plotly_dark", height=500,
            margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Daily Net Return (%)", yaxis_title="Frequency (Days)"
        )
        fig_hist.update_traces(opacity=0.8)
        st.plotly_chart(fig_hist, use_container_width=True)