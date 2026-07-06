import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import TEXT

class MatrixView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render_leaderboard(self, lang: str = "EN"):
        """Renders the top Candidate Matrix and Alpha Leaderboard on the main page."""
        t = TEXT[lang]
        st.markdown(t["matrix_title"])

        runs_df = self.dm.get_all_runs()
        if runs_df.empty:
            st.info(t.get("db_empty", "No data available."))
            return

        # Prepare Matrix DataFrame
        matrix_df = runs_df[['name', 'round_number', 'validation_ic', 'holdout_ic']].copy()
        
        # Rename columns dynamically based on language
        factor_col = t["cols"]["Factor"]
        round_col = t["cols"]["Round"]
        val_col = t["cols"]["Val_IC"]
        hold_col = t["cols"]["Holdout_IC"]
        
        matrix_df = matrix_df.rename(columns={
            'name': factor_col,
            'round_number': round_col,
            'validation_ic': val_col,
            'holdout_ic': hold_col
        })

        # --- THE CHAMPION PODIUM ---
        top_idx = matrix_df[hold_col].idxmax()
        top_factor = matrix_df.loc[top_idx]
        
        st.markdown(t["leaderboard_title"])
        st.success(
            f"**🥇 Top Model:** `{top_factor[factor_col]}` (Round {top_factor[round_col]})  \n"
            f"**Holdout IC:** `{top_factor[hold_col]:.4f}` &nbsp;|&nbsp; "
            f"**Validation IC:** `{top_factor[val_col]:.4f}`"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # --- THE FULL MATRIX ---
        styled_df = matrix_df.style.format({
            val_col: "{:.4f}",
            hold_col: "{:.4f}"
        })
        st.dataframe(styled_df, width="stretch", hide_index=True, height=300)

    def render_correlation(self, lang: str = "EN"):
        """Renders the dynamic Factor Orthogonalization Heatmap in Tab 2."""
        t = TEXT[lang]
        st.markdown(f"### {t['corr_title']}")
        st.caption(t['corr_desc'])

        runs_df = self.dm.get_all_runs()
        if runs_df.empty:
            return

        # Extract a dictionary of the LATEST run for each unique factor
        latest_runs = runs_df.drop_duplicates(subset=['name'], keep='first')
        run_dict = dict(zip(latest_runs['name'], latest_runs['run_id']))

        # Fetch all overlapping returns from the DataManager
        df_corr = self.dm.get_correlation_returns(run_dict)

        if not df_corr.empty and len(df_corr.columns) > 1:
            available_factors = df_corr.columns.tolist()
            
            # Default to the 5 most recent factors to keep the initial load clean
            default_selections = available_factors[-5:] if len(available_factors) >= 5 else available_factors
            
            selected_factors = st.multiselect(
                t["corr_select_factors"],
                options=available_factors,
                default=default_selections
            )
            
            if len(selected_factors) >= 2:
                filtered_df = df_corr[selected_factors]
                corr_matrix = filtered_df.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r', 
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    hoverinfo="z"
                ))

                fig_corr.update_layout(
                    template="plotly_dark",
                    height=max(400, len(selected_factors) * 60), # Dynamically scale height
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_corr, width="stretch")
            else:
                st.info(t["corr_select_min"])
        else:
            st.warning(t["corr_waiting"])