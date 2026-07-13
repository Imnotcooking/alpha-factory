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
        frontier_df = self._prepare_frontier_frame(self.dm.get_pareto_data(current_factor_name))

        # 3. Render Logic
        if len(frontier_df) > 1:
            try:
                fig_pareto = px.scatter_3d(
                    frontier_df,
                    x='Drawdown Magnitude (%)',
                    y='Return (%)',
                    z='Turnover (%)',
                    color='holdout_ic',
                    size='bubble_size',
                    hover_data={
                        'round_number': True,
                        'Return (%)': ':.2f',
                        'Drawdown Magnitude (%)': ':.2f',
                        'Turnover (%)': ':.2f',
                        'holdout_ic': ':.4f',
                        'bubble_size': False,
                    },
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    size_max=20,
                    labels={
                        'round_number': t.get('pareto_round', 'Round'),
                        'Drawdown Magnitude (%)': t.get('pareto_axis_drawdown', 'Drawdown Magnitude (%)'),
                        'Return (%)': t.get('pareto_axis_return', 'Return (%)'),
                        'Turnover (%)': t.get('pareto_axis_turnover', 'Turnover (%)'),
                        'holdout_ic': t.get('pareto_color_ic', 'Holdout IC'),
                        'bubble_size': t.get('pareto_size_abs_ic', 'Abs(IC)'),
                    },
                )

                fig_pareto.update_layout(
                    template="plotly_dark",
                    height=560,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    scene=dict(
                        xaxis=dict(
                            title=t.get('pareto_axis_drawdown', 'Drawdown Magnitude (%)'),
                            autorange='reversed',
                        ),
                        yaxis=dict(title=t.get('pareto_axis_return', 'Return (%)')),
                        zaxis=dict(title=t.get('pareto_axis_turnover', 'Turnover (%)')),
                        camera=dict(eye=dict(x=1.6, y=-1.7, z=1.15)),
                    ),
                )
                fig_pareto.update_traces(
                    marker=dict(
                        opacity=0.86,
                        line=dict(width=0.8, color="rgba(15, 23, 42, 0.65)"),
                    )
                )

                st.plotly_chart(fig_pareto, width="stretch")

                # Educational Expander
                st.info(t["pareto_help"])

            except Exception as e:
                st.error(f"{t['pareto_error']}: {e}")
        else:
            st.warning(t["pareto_missing"])

    @staticmethod
    def _prepare_frontier_frame(frontier_df: pd.DataFrame) -> pd.DataFrame:
        out = frontier_df.copy()
        if out.empty:
            return out

        out["Return (%)"] = pd.to_numeric(out.get("Return (%)"), errors="coerce")
        if "Drawdown Magnitude (%)" not in out.columns:
            drawdown = pd.to_numeric(out.get("Drawdown (%)"), errors="coerce")
            out["Drawdown Magnitude (%)"] = drawdown.abs()
        out["Turnover (%)"] = pd.to_numeric(out.get("turnover_rate"), errors="coerce").fillna(0.0) * 100.0
        out["holdout_ic"] = pd.to_numeric(out.get("holdout_ic"), errors="coerce").fillna(0.0)
        out["bubble_size"] = out["holdout_ic"].abs().clip(lower=0.01)
        return out.dropna(subset=["Return (%)", "Drawdown Magnitude (%)", "Turnover (%)"])
