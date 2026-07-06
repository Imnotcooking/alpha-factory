import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, TEXT, get_plotly_template

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from diagnostics.ic_decay import DEFAULT_FEATURE, compute_ic_decay, list_feature_columns
except Exception:
    DEFAULT_FEATURE = "f_oi_growth_10d"
    compute_ic_decay = None
    list_feature_columns = None


@st.cache_data(show_spinner=False)
def _cached_ic_decay(matrix_path: str, matrix_mtime: float, feature: str, tickers: tuple[str, ...]):
    return compute_ic_decay(
        feature=feature,
        matrix_path=matrix_path,
        tickers=tickers or None,
    )


@st.cache_data(show_spinner=False)
def _cached_feature_columns(matrix_path: str, matrix_mtime: float):
    return list_feature_columns(matrix_path)


class MLView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(
        self,
        run_id: str,
        run_metadata: pd.Series,
        lang: str = "EN",
        theme_mode: str = "DARK",
    ):
        t = TEXT[lang] if lang in TEXT else TEXT["EN"]

        self._render_feature_importance(run_id, run_metadata, theme_mode, t)
        st.markdown("---")
        self._render_ic_decay_panel(run_id, run_metadata, theme_mode, t)
        st.markdown("---")
        self._render_shap_dna(theme_mode)

    def _render_feature_importance(
        self,
        run_id: str,
        run_metadata: pd.Series,
        theme_mode: str,
        t: dict,
    ):
        st.markdown(f"### {t.get('tab_ml', 'ML Feature Importance')}")
        st.caption(t.get("ml_caption", "Visualizing the internal decision-making weights of the Machine Learning model."))

        factor_id = run_metadata.get("factor_id", None)
        importance_df = self.dm.get_feature_importance(run_id, factor_id=factor_id)
        if importance_df.empty or not {"feature", "importance"}.issubset(importance_df.columns):
            st.info(t.get("ml_missing", "No Feature Importance data found."))
            return

        plot_df = importance_df.copy()
        plot_df["importance"] = pd.to_numeric(plot_df["importance"], errors="coerce")
        plot_df = plot_df.dropna(subset=["feature", "importance"]).sort_values("importance", ascending=True)
        if plot_df.empty:
            st.info(t.get("ml_missing", "No Feature Importance data found."))
            return

        fig = px.bar(
            plot_df.tail(20),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            template=get_plotly_template(theme_mode),
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")

    def _render_ic_decay_panel(
        self,
        run_id: str,
        run_metadata: pd.Series,
        theme_mode: str,
        t: dict,
    ):
        if compute_ic_decay is None or list_feature_columns is None:
            return

        matrix_path = os.path.join(ALPHA_RUNTIME_DATA_ROOT, "feature_store", "ML_Feature_Matrix.parquet")
        if not os.path.exists(matrix_path):
            return

        matrix_mtime = os.path.getmtime(matrix_path)
        try:
            available_features = _cached_feature_columns(matrix_path, matrix_mtime)
        except Exception as exc:
            st.warning(f"{t.get('ic_decay_missing', 'IC decay unavailable')}: {exc}")
            return

        if not available_features:
            st.info(t.get("ic_decay_no_features", "No engineered feature columns found for IC decay."))
            return

        factor_id = run_metadata.get("factor_id", None)
        importance_df = self.dm.get_feature_importance(run_id, factor_id=factor_id)
        if not importance_df.empty and "feature" in importance_df.columns:
            ranked_features = [
                feature
                for feature in importance_df["feature"].dropna().astype(str).tolist()
                if feature in available_features
            ]
        else:
            ranked_features = []

        if not ranked_features:
            ranked_features = [DEFAULT_FEATURE] if DEFAULT_FEATURE in available_features else [available_features[0]]

        st.markdown(f"#### {t.get('ic_decay_title', 'IC Decay Curve')}")
        st.caption(
            t.get(
                "ic_decay_desc",
                "Tests whether a feature's cross-sectional signal persists across forward return horizons.",
            )
        )

        selected_feature = st.selectbox(
            t.get("ic_decay_select_feature", "Feature to test"),
            ranked_features,
            index=0,
            key=f"ml_ic_decay_feature_{run_id}",
        )

        tickers = self._parse_ticker_scope(run_metadata.get("traded_tickers", ""))
        try:
            ic_df = _cached_ic_decay(
                matrix_path,
                matrix_mtime,
                selected_feature,
                tuple(sorted(tickers)),
            )
        except Exception as exc:
            st.warning(f"{t.get('ic_decay_missing', 'IC decay unavailable')}: {exc}")
            return

        if ic_df.empty or ic_df["ic"].dropna().empty:
            st.info(t.get("ic_decay_no_data", "No valid IC decay observations for this feature."))
            return

        one_day_ic = ic_df.loc[ic_df["horizon"] == 1, "ic"]
        one_day_ic = one_day_ic.iloc[0] if not one_day_ic.empty else np.nan
        peak_row = ic_df.loc[ic_df["ic"].abs().idxmax()]
        valid_days = int(ic_df["valid_days"].max())

        m1, m2, m3 = st.columns(3)
        m1.metric(t.get("ic_decay_1d", "1D IC"), "N/A" if pd.isna(one_day_ic) else f"{one_day_ic:.4f}")
        m2.metric(
            t.get("ic_decay_peak", "Peak |IC|"),
            f"{peak_row['ic']:.4f} @ {int(peak_row['horizon'])}d",
        )
        m3.metric(t.get("ic_decay_days", "Valid Days"), f"{valid_days:,}")

        marker_colors = np.where(ic_df["ic"] >= 0, "#00E676", "#FF5252")
        fig_ic = go.Figure()
        fig_ic.add_trace(
            go.Scatter(
                x=ic_df["horizon"],
                y=ic_df["ic"],
                mode="lines+markers",
                name=selected_feature,
                line=dict(color="#40C4FF", width=2.5),
                marker=dict(size=8, color=marker_colors),
                customdata=np.stack(
                    [
                        ic_df["ic_ir"].fillna(0.0),
                        ic_df["positive_day_rate"].fillna(0.0),
                        ic_df["sample_count"].fillna(0).astype(int),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Horizon=%{x}d<br>"
                    "Mean IC=%{y:.4f}<br>"
                    "ICIR=%{customdata[0]:.3f}<br>"
                    "Positive day rate=%{customdata[1]:.1%}<br>"
                    "Samples=%{customdata[2]:,}<extra></extra>"
                ),
            )
        )
        fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_ic.update_layout(
            template=get_plotly_template(theme_mode),
            height=330,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=t.get("ic_decay_xaxis", "Forward horizon (days)"),
            yaxis_title=t.get("ic_decay_yaxis", "Mean daily Spearman IC"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ic, width="stretch")

        with st.expander(t.get("ic_decay_help_title", "How to read IC decay")):
            st.markdown(t.get("ic_decay_help", ""))

    def _render_shap_dna(self, theme_mode: str):
        st.markdown("### AI DNA & Economic Rationale (SHAP Autopsy)")
        st.markdown(
            """
            This module cracks open the XGBoost black box using SHAP values.
            By mapping feature importance to the GMM-HMM market regimes, we prove that the AI physically alters its decision-making DNA to survive market crashes.
            """
        )

        shap_df = self.dm.get_shap_dna()
        if shap_df.empty:
            st.warning("SHAP DNA Matrix not found. Please run `python diagnostics/shap_engine.py` first.")
            return

        st.markdown("#### Interactive Regime Sandbox")
        regimes = shap_df["regime_name"].unique()
        selected_regime = st.selectbox("Select Market Regime to Analyze:", regimes)
        regime_data = shap_df[shap_df["regime_name"] == selected_regime].copy()

        feature_cols = [c for c in regime_data.columns if c not in ["regime", "regime_name"]]
        plot_data = regime_data.melt(
            id_vars=["regime_name"],
            value_vars=feature_cols,
            var_name="Feature",
            value_name="SHAP Value",
        )
        plot_data = plot_data.sort_values(by="SHAP Value", ascending=True)

        fig_imp = px.bar(
            plot_data,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(
            title=f"XGBoost Feature Dominance: {selected_regime}",
            template=get_plotly_template(theme_mode),
            height=550,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_imp, width="stretch")

        st.markdown("#### Institutional Interpretation")
        top_features = plot_data.tail(2)["Feature"].tolist()
        if len(top_features) < 2:
            st.info("Not enough SHAP features to generate commentary.")
            return

        top_1 = top_features[1]
        top_2 = top_features[0]
        if "Panic" in selected_regime or "Rough" in selected_regime:
            st.error(
                f"**CRISIS MODE DETECTED:** In this highly volatile state, the AI's decision tree shifts its DNA to heavily weight `{top_1}` and `{top_2}`. It abandons standard linear assumptions to focus on survival, liquidity, and fractal roughness."
            )
        elif "Trend" in selected_regime:
            st.success(
                f"**TRENDING REGIME DETECTED:** The AI is focusing heavily on `{top_1}` and `{top_2}`. It is capturing persistent directional momentum while giving lower priority to mean-reverting signals."
            )
        else:
            st.info(
                f"**CHOP / MEAN-REVERSION DETECTED:** The AI balances `{top_1}` and `{top_2}` to navigate sideways structural whipsaws. It expects directional breakouts to fail and hunts for relative value mispricings."
            )

    @staticmethod
    def _parse_ticker_scope(raw_tickers) -> list[str]:
        raw = str(raw_tickers).strip()
        if not raw or raw.upper() in {"ALL", "NONE", "NAN", "<NA>"}:
            return []
        return [ticker.strip() for ticker in raw.split(",") if ticker.strip()]
