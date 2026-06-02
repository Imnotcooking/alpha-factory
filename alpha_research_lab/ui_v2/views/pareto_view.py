import streamlit as st
import plotly.express as px
import pandas as pd
from config import TEXT

class ParetoView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(self, run_metadata: pd.Series, lang: str = "EN"):
        t = TEXT[lang]
        
        st.markdown(f"### {t.get('tab_pareto', '📈 Pareto Frontier')}")
        st.caption(t["pareto_caption"])
        
        # 1. Fetch the specific factor name from the UI table
        current_factor_name = run_metadata['name']
        
        # 2. Ask the DataManager for the clean data
        frontier_df = self.dm.get_pareto_data(current_factor_name)
        
        # 3. Render Logic
        if len(frontier_df) > 1:
            try:
                fig_pareto = px.scatter(
                    frontier_df, 
                    x='Drawdown (%)', 
                    y='Return (%)', 
                    color='turnover_rate', 
                    size='bubble_size',
                    hover_data=['round_number', 'holdout_ic'],
                    color_continuous_scale='Turbo',
                    labels={
                        'turnover_rate': 'Turnover (%) / 换手率', 
                        'bubble_size': 'Abs(IC)'
                    }
                )
                
                fig_pareto.update_layout(
                    template="plotly_dark", 
                    height=500, 
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_autorange='reversed' # Reverses X so the best drawdowns are on the right
                )
                
                # Baseline for 0% return
                fig_pareto.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                # Educational Expander
                st.info(t["pareto_help"])
            
            except Exception as e:
                st.error(f"{t['pareto_error']}: {e}")
        else:
            st.warning(t["pareto_missing"])