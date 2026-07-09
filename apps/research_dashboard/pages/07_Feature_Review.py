from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, get_plotly_template
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)

ROOT_DIR = os.path.dirname(UI_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from oqp.research.ml import (
    FeatureGovernanceConfig,
    compute_feature_governance,
    list_matrix_files,
    load_matrix,
)
from oqp.research.ml import PurgedMDAConfig, compute_oos_mda
from oqp.research.latent import (
    attach_gmm_diagnostics,
    load_saved_latents,
    save_latent_artifacts,
    train_temporal_vqvae_latents,
)


from oqp.ui.translations import research_page_legacy_catalog


PAGE_TEXT = research_page_legacy_catalog("alpha_feature_governance")


st.set_page_config(page_title="Feature Review", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang if st.session_state.lang in PAGE_TEXT else "EN"
t = PAGE_TEXT[lang]
template = get_plotly_template(st.session_state.theme_mode)


@st.cache_data(show_spinner=False)
def _cached_matrix(matrix_path: str, matrix_mtime: float) -> pd.DataFrame:
    return load_matrix(matrix_path)


@st.cache_data(show_spinner=False)
def _cached_governance(
    matrix_path: str,
    matrix_mtime: float,
    target_col: str,
    include_prob_features: bool,
    corr_threshold: float,
    min_assets_per_day: int,
) -> dict:
    matrix = load_matrix(matrix_path)
    config = FeatureGovernanceConfig(
        target_col=target_col,
        include_prob_features=include_prob_features,
        corr_threshold=corr_threshold,
        min_assets_per_day=min_assets_per_day,
    )
    return compute_feature_governance(matrix, config)


@st.cache_data(show_spinner=False)
def _cached_oos_mda(
    matrix_path: str,
    matrix_mtime: float,
    target_col: str,
    feature_cols: tuple[str, ...],
    n_splits: int,
    embargo_days: int,
    max_rows: int,
    max_features: int,
    permutation_repeats: int,
    min_assets_per_day: int,
) -> dict:
    matrix = load_matrix(matrix_path)
    config = PurgedMDAConfig(
        n_splits=n_splits,
        embargo_days=embargo_days,
        max_rows=max_rows,
        max_features=max_features,
        permutation_repeats=permutation_repeats,
        min_assets_per_day=min_assets_per_day,
    )
    return compute_oos_mda(
        matrix,
        feature_cols=list(feature_cols),
        target_col=target_col,
        config=config,
    )


def _format_percent_cols(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    format_map = {
        "missing_pct": "{:.2%}",
        "coverage_pct": "{:.2%}",
        "zero_pct": "{:.2%}",
        "mean_ic": "{:.4f}",
        "abs_mean_ic": "{:.4f}",
        "ic_ir": "{:.3f}",
        "positive_day_rate": "{:.1%}",
        "turnover_proxy": "{:.3f}",
        "quality_score": "{:.3f}",
        "max_abs_corr_to_peer": "{:.3f}",
    }
    existing = {col: fmt for col, fmt in format_map.items() if col in df.columns}
    return df.style.format(existing)


def _top_features(summary: pd.DataFrame, limit: int = 25) -> list[str]:
    if summary.empty:
        return []
    return summary.head(limit)["feature"].tolist()


def _format_mda_cols(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    format_map = {
        "mda_mean": "{:.5f}",
        "mda_std": "{:.5f}",
        "mda_median": "{:.5f}",
        "positive_fold_rate": "{:.1%}",
        "baseline_score_mean": "{:.5f}",
        "permuted_score_mean": "{:.5f}",
        "gain_importance_mean": "{:.5f}",
        "mda_rank": "{:.0f}",
        "gain_rank": "{:.0f}",
        "gain_minus_mda_rank": "{:+.0f}",
    }
    existing = {col: fmt for col, fmt in format_map.items() if col in df.columns}
    return df.style.format(existing)


def _render_metric_row(health: dict, code_ic: pd.DataFrame, t: dict):
    cols = st.columns(4)
    active = f"{health.get('active_codes', 0)} / {health.get('total_codes', 0)}"
    cols[0].metric(t["active_codes"], active)
    perplexity = health.get("usage_perplexity", np.nan)
    cols[1].metric(
        t["perplexity"], "N/A" if pd.isna(perplexity) else f"{perplexity:.2f}"
    )
    largest = health.get("largest_code_pct", np.nan)
    cols[2].metric(t["largest_code"], "N/A" if pd.isna(largest) else f"{largest:.1%}")
    if code_ic.empty or "mean_code_ic" not in code_ic.columns:
        ic_label = "N/A"
    else:
        ic_label = f"{code_ic['mean_code_ic'].iloc[0]:.4f}"
    cols[3].metric(t["code_ic"], ic_label)


matrix_files = list_matrix_files(BASE_DIR)
if not matrix_files:
    st.error(t["no_file"])
    st.stop()

st.title(t["page_title"])
st.caption(t["subtitle"])

control_cols = st.columns([0.36, 0.2, 0.18, 0.14, 0.12])
with control_cols[0]:
    labels = [path.name for path in matrix_files]
    selected_label = st.selectbox(t["matrix"], labels, index=0)
    selected_path = str(matrix_files[labels.index(selected_label)])

matrix_mtime = os.path.getmtime(selected_path)
try:
    preview_df = _cached_matrix(selected_path, matrix_mtime)
except Exception as exc:
    st.error(f"{t['load_error']}: {exc}")
    st.stop()

target_candidates = [col for col in preview_df.columns if col.startswith("target_")]
if not target_candidates:
    target_candidates = [col for col in preview_df.columns if "target" in col.lower()]
if not target_candidates:
    target_candidates = ["target_1d_rank"]

with control_cols[1]:
    default_target_idx = (
        target_candidates.index("target_1d_rank")
        if "target_1d_rank" in target_candidates
        else 0
    )
    target_col = st.selectbox(t["target"], target_candidates, index=default_target_idx)

with control_cols[2]:
    has_prob = any(col.startswith("prob_") for col in preview_df.columns)
    include_prob = st.checkbox(t["include_prob"], value=has_prob)

with control_cols[3]:
    corr_threshold = st.slider(t["corr_threshold"], 0.60, 0.98, 0.85, 0.01)

with control_cols[4]:
    min_assets = st.slider(t["min_assets"], 3, 30, 5, 1)

try:
    result = _cached_governance(
        selected_path,
        matrix_mtime,
        target_col,
        include_prob,
        corr_threshold,
        min_assets,
    )
except Exception as exc:
    st.error(f"{t['load_error']}: {exc}")
    st.stop()

metadata = result["metadata"]
summary = result["summary"]
daily_ic = result["daily_ic"]
corr_matrix = result["corr_matrix"]
corr_pairs = result["corr_pairs"]
clusters = result["clusters"]
family_counts = result["family_counts"]
pca_variance = result["pca_variance"]
pca_loadings = result["pca_loadings"]
keeper_features = result["keeper_features"]

metric_cols = st.columns(4)
metric_cols[0].metric(t["rows"], f"{metadata['rows']:,}")
metric_cols[1].metric(t["features"], f"{metadata['features']:,}")
metric_cols[2].metric(t["assets"], f"{metadata['assets']:,}")
if metadata["date_min"] is not None and metadata["date_max"] is not None:
    date_label = f"{pd.to_datetime(metadata['date_min']).date()} -> {pd.to_datetime(metadata['date_max']).date()}"
else:
    date_label = "N/A"
metric_cols[3].metric(t["dates"], date_label)

with st.expander(t["manual_title"], expanded=False):
    st.markdown(t["manual"])

tabs = st.tabs(
    [
        t["overview"],
        t["corr"],
        t["stability"],
        t["mda"],
        t["missing"],
        t["pca"],
        t["latent"],
        t["protocol"],
    ]
)

with tabs[0]:
    left, right = st.columns([0.44, 0.56])
    with left:
        st.markdown(f"### {t['family_title']}")
        if not family_counts.empty:
            fig_family = px.bar(
                family_counts.sort_values("feature_count"),
                x="feature_count",
                y="family",
                orientation="h",
                color="avg_abs_ic",
                color_continuous_scale="Viridis",
                labels={
                    "feature_count": "Features",
                    "family": "Family",
                    "avg_abs_ic": "|IC|",
                },
                template=template,
            )
            fig_family.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_family, use_container_width=True)
    with right:
        st.markdown(f"### {t['summary_title']}")
        show_cols = [
            "feature",
            "family",
            "quality_score",
            "mean_ic",
            "ic_ir",
            "positive_day_rate",
            "missing_pct",
            "turnover_proxy",
        ]
        st.dataframe(
            _format_percent_cols(summary[show_cols]),
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:
    st.markdown(f"### {t['heatmap_title']}")
    display_features = _top_features(summary, 40)
    heat = corr_matrix.loc[display_features, display_features]
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="rho"),
        )
    )
    fig_corr.update_layout(
        template=template,
        height=max(480, min(900, 28 * len(display_features))),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    pair_col, cluster_col = st.columns([0.45, 0.55])
    with pair_col:
        st.markdown(f"#### {t['pairs_title']}")
        if corr_pairs.empty:
            st.info(t["no_pairs"])
        else:
            st.dataframe(
                _format_percent_cols(corr_pairs.head(50)),
                use_container_width=True,
                hide_index=True,
            )
    with cluster_col:
        st.markdown(f"#### {t['clusters_title']}")
        cluster_cols = [
            "cluster_id",
            "cluster_size",
            "representative",
            "feature",
            "family",
            "abs_mean_ic",
            "turnover_proxy",
            "max_abs_corr_to_peer",
            "keep_candidate",
        ]
        st.dataframe(
            _format_percent_cols(clusters[cluster_cols]),
            use_container_width=True,
            hide_index=True,
        )

with tabs[2]:
    left, right = st.columns([0.48, 0.52])
    with left:
        st.markdown(f"### {t['ic_bar']}")
        ic_plot = summary.sort_values("abs_mean_ic", ascending=True).tail(25).copy()
        fig_ic = px.bar(
            ic_plot,
            x="mean_ic",
            y="feature",
            orientation="h",
            color="family",
            hover_data=["ic_ir", "positive_day_rate", "valid_ic_days"],
            template=template,
        )
        fig_ic.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_ic.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_ic, use_container_width=True)

    with right:
        st.markdown(f"### {t['turnover_scatter']}")
        scatter_df = summary.dropna(subset=["turnover_proxy", "mean_ic"]).copy()
        fig_turnover = px.scatter(
            scatter_df,
            x="turnover_proxy",
            y="mean_ic",
            color="family",
            size="coverage_pct",
            hover_name="feature",
            hover_data=["ic_ir", "positive_day_rate", "missing_pct"],
            template=template,
        )
        fig_turnover.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_turnover.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_turnover, use_container_width=True)

    st.markdown(f"### {t['daily_ic']}")
    if daily_ic.empty:
        st.info("No daily IC observations available.")
    else:
        default_features = _top_features(summary, 5)
        selected_features = st.multiselect(
            "Features",
            summary["feature"].tolist(),
            default=default_features,
            key="feature_governance_daily_ic_features",
        )
        plot_daily = daily_ic[daily_ic["feature"].isin(selected_features)].copy()
        if plot_daily.empty:
            st.info("Select at least one feature.")
        else:
            pivot = plot_daily.pivot(
                index="date", columns="feature", values="ic"
            ).sort_index()
            rolled = pivot.rolling(60, min_periods=10).mean().reset_index()
            melted = rolled.melt(
                "date", var_name="feature", value_name="rolling_ic"
            ).dropna()
            fig_daily = px.line(
                melted,
                x="date",
                y="rolling_ic",
                color="feature",
                template=template,
            )
            fig_daily.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_daily.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_daily, use_container_width=True)

with tabs[3]:
    st.markdown(f"### {t['mda_title']}")
    with st.expander(t["manual_title"], expanded=False):
        st.markdown(t["mda_help"])

    mda_controls = st.columns([0.16, 0.14, 0.16, 0.18, 0.18, 0.18])
    with mda_controls[0]:
        mda_max_features = st.select_slider(
            t["mda_features"], options=[10, 15, 25, 40], value=25
        )
    with mda_controls[1]:
        mda_folds = st.select_slider(t["mda_folds"], options=[3, 4, 5, 7], value=5)
    with mda_controls[2]:
        mda_embargo = st.select_slider(t["mda_embargo"], options=[0, 4, 8, 20], value=8)
    with mda_controls[3]:
        mda_max_rows = st.select_slider(
            t["mda_rows"], options=[10_000, 25_000, 50_000, 100_000], value=50_000
        )
    with mda_controls[4]:
        mda_repeats = st.select_slider(t["mda_repeats"], options=[1, 2, 3], value=1)
    with mda_controls[5]:
        st.write("")
        run_mda = st.button(t["mda_run"], use_container_width=True)

    mda_feature_cols = tuple(_top_features(summary, int(mda_max_features)))
    mda_key = (
        selected_path,
        matrix_mtime,
        target_col,
        mda_feature_cols,
        int(mda_folds),
        int(mda_embargo),
        int(mda_max_rows),
        int(mda_max_features),
        int(mda_repeats),
        int(min_assets),
    )
    if run_mda:
        try:
            with st.spinner(t["mda_running"]):
                mda_result = _cached_oos_mda(*mda_key)
            st.session_state["feature_governance_mda_key"] = mda_key
            st.session_state["feature_governance_mda_result"] = mda_result
        except Exception as exc:
            st.error(f"{t['load_error']}: {exc}")

    mda_result = None
    if st.session_state.get("feature_governance_mda_key") == mda_key:
        mda_result = st.session_state.get("feature_governance_mda_result")

    if not mda_result:
        st.info(t["mda_not_run"])
    else:
        mda_summary = mda_result.get("summary", pd.DataFrame())
        mda_fold_scores = mda_result.get("fold_scores", pd.DataFrame())
        mda_metadata = mda_result.get("metadata", {})
        mda_metric_cols = st.columns(4)
        mda_metric_cols[0].metric(t["rows"], f"{mda_metadata.get('rows', 0):,}")
        mda_metric_cols[1].metric(t["features"], f"{mda_metadata.get('features', 0):,}")
        mda_metric_cols[2].metric(t["mda_folds"], f"{mda_metadata.get('folds', 0):,}")
        baseline_score = mda_metadata.get("baseline_score_mean", np.nan)
        mda_metric_cols[3].metric(
            t["mda_score"],
            "N/A" if pd.isna(baseline_score) else f"{baseline_score:.4f}",
        )

        if not mda_summary.empty:
            left, right = st.columns([0.50, 0.50])
            with left:
                st.markdown(f"#### {t['mda_plot']}")
                plot_mda = mda_summary.sort_values("mda_mean", ascending=True).tail(25)
                fig_mda = px.bar(
                    plot_mda,
                    x="mda_mean",
                    y="feature",
                    orientation="h",
                    color="diagnosis",
                    hover_data=[
                        "positive_fold_rate",
                        "mda_std",
                        "gain_importance_mean",
                    ],
                    template=template,
                )
                fig_mda.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_mda.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_mda, use_container_width=True)

            with right:
                st.markdown(f"#### {t['mda_gain_scatter']}")
                scatter_mda = mda_summary.dropna(
                    subset=["gain_importance_mean", "mda_mean"]
                ).copy()
                if scatter_mda.empty:
                    st.info(t["mda_gain_unavailable"])
                else:
                    fig_gain = px.scatter(
                        scatter_mda,
                        x="gain_importance_mean",
                        y="mda_mean",
                        color="diagnosis",
                        size="positive_fold_rate",
                        hover_name="feature",
                        hover_data=["mda_rank", "gain_rank", "gain_minus_mda_rank"],
                        template=template,
                    )
                    fig_gain.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_gain.update_layout(
                        height=520, margin=dict(l=10, r=10, t=20, b=10)
                    )
                    st.plotly_chart(fig_gain, use_container_width=True)

            st.markdown(f"#### {t['mda_table']}")
            show_mda_cols = [
                "feature",
                "diagnosis",
                "mda_mean",
                "mda_std",
                "positive_fold_rate",
                "baseline_score_mean",
                "permuted_score_mean",
                "gain_importance_mean",
                "mda_rank",
                "gain_rank",
                "gain_minus_mda_rank",
                "observations",
            ]
            st.dataframe(
                _format_mda_cols(
                    mda_summary[
                        [col for col in show_mda_cols if col in mda_summary.columns]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not mda_fold_scores.empty:
            with st.expander(t["mda_fold_scores"], expanded=False):
                st.dataframe(mda_fold_scores, use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown(f"### {t['missing_bar']}")
    miss = summary.sort_values("missing_pct", ascending=True).tail(30)
    fig_missing = px.bar(
        miss,
        x="missing_pct",
        y="feature",
        orientation="h",
        color="family",
        template=template,
    )
    fig_missing.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    miss_cols = [
        "feature",
        "family",
        "missing_pct",
        "coverage_pct",
        "zero_pct",
        "unique_count",
        "std",
        "min",
        "max",
    ]
    st.dataframe(
        _format_percent_cols(
            summary[miss_cols].sort_values("missing_pct", ascending=False)
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[5]:
    if pca_variance.empty or pca_loadings.empty:
        st.info("PCA baseline unavailable for this matrix.")
    else:
        st.info(t["pca_market_note"])
        left, right = st.columns([0.48, 0.52])
        with left:
            st.markdown(f"### {t['pca_variance']}")
            fig_pca = go.Figure()
            fig_pca.add_trace(
                go.Bar(
                    x=pca_variance["component"],
                    y=pca_variance["explained_variance_ratio"],
                    name="Individual",
                    marker_color="#40C4FF",
                )
            )
            fig_pca.add_trace(
                go.Scatter(
                    x=pca_variance["component"],
                    y=pca_variance["cumulative_variance"],
                    name="Cumulative",
                    mode="lines+markers",
                    line=dict(color="#00C853", width=3),
                )
            )
            fig_pca.update_layout(
                template=template,
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with right:
            st.markdown(f"### {t['pca_loadings']}")
            component_options = [
                col for col in pca_loadings.columns if col.startswith("PC")
            ]
            component = st.selectbox(t["component"], component_options)
            loading_df = pca_loadings[["feature", component]].copy()
            loading_df["abs_loading"] = loading_df[component].abs()
            loading_df = loading_df.sort_values("abs_loading", ascending=False).head(20)
            loading_df = loading_df.sort_values(component)
            fig_load = px.bar(
                loading_df,
                x=component,
                y="feature",
                orientation="h",
                color=component,
                color_continuous_scale="RdBu",
                template=template,
            )
            fig_load.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_load, use_container_width=True)

        st.dataframe(
            pca_variance.style.format(
                {
                    "explained_variance_ratio": "{:.2%}",
                    "cumulative_variance": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tabs[6]:
    st.markdown(f"### {t['latent_title']}")
    with st.expander(t["manual_title"], expanded=False):
        st.markdown(t["latent_help"])

    control_row = st.columns([0.16, 0.18, 0.14, 0.14, 0.14, 0.12, 0.12])
    with control_row[0]:
        latent_window = st.slider(t["window"], 5, 60, 20, 5)
    with control_row[1]:
        latent_samples = st.select_slider(
            t["samples"],
            options=[2_000, 5_000, 10_000, 20_000, 50_000],
            value=10_000,
        )
    with control_row[2]:
        latent_codes = st.select_slider(
            t["codes"], options=[8, 12, 16, 24, 32], value=16
        )
    with control_row[3]:
        latent_dim = st.select_slider(
            t["latent_dim"], options=[2, 4, 8, 12, 16], value=8
        )
    with control_row[4]:
        latent_epochs = st.slider(t["epochs"], 1, 50, 8, 1)
    with control_row[5]:
        train_clicked = st.button(t["train_latent"], use_container_width=True)
    with control_row[6]:
        load_clicked = st.button(t["load_latent"], use_container_width=True)

    latent_result = None
    if train_clicked:
        feature_cols = [
            feature
            for feature in summary["feature"].tolist()
            if feature in preview_df.columns
            and pd.api.types.is_numeric_dtype(preview_df[feature])
        ]
        try:
            with st.spinner(t["latent_training"]):
                latent_result = train_temporal_vqvae_latents(
                    preview_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    window_size=latent_window,
                    max_samples=latent_samples,
                    num_codes=latent_codes,
                    latent_dim=latent_dim,
                    epochs=latent_epochs,
                )
                paths = save_latent_artifacts(latent_result)
                st.session_state["feature_governance_latent_result"] = {
                    key: value
                    for key, value in latent_result.items()
                    if key != "encoder"
                }
            st.success(f"{t['latent_saved']}: {paths['latent']}")
        except Exception as exc:
            st.error(f"{t['load_error']}: {exc}")

    if latent_result is None:
        latent_result = st.session_state.get("feature_governance_latent_result")
    if latent_result is None and load_clicked:
        latent_result = load_saved_latents()
        if latent_result:
            st.session_state["feature_governance_latent_result"] = latent_result
    if latent_result is None:
        latent_result = load_saved_latents()

    if not latent_result:
        st.info(t["latent_unavailable"])
    else:
        latent_df = latent_result.get("latent", pd.DataFrame())
        usage_df = latent_result.get("usage", pd.DataFrame())
        loss_df = latent_result.get("loss_history", pd.DataFrame())
        health = latent_result.get("health", {})
        code_ic = latent_result.get("code_ic", pd.DataFrame())
        if code_ic is None or code_ic.empty:
            from oqp.research.latent import compute_code_target_ic

            code_ic = compute_code_target_ic(latent_df, target_col=target_col)

        _render_metric_row(health, code_ic, t)
        largest_code_pct = health.get("largest_code_pct", np.nan)
        if pd.notna(largest_code_pct) and largest_code_pct > 0.80:
            st.warning(t["collapse_warning"])

        if not usage_df.empty:
            st.markdown(f"#### {t['usage_title']}")
            usage_plot = usage_df.copy()
            fig_usage = px.bar(
                usage_plot,
                x="vq_code",
                y="usage_pct",
                color="usage_pct",
                color_continuous_scale="Viridis",
                template=template,
            )
            fig_usage.update_layout(
                height=320, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_usage, use_container_width=True)

        if not loss_df.empty and {"epoch", "loss"}.issubset(loss_df.columns):
            st.markdown(f"#### {t['loss_title']}")
            loss_cols = [
                col
                for col in [
                    "loss",
                    "reconstruction_loss",
                    "vq_loss",
                    "orthogonality_loss",
                    "usage_loss",
                ]
                if col in loss_df.columns
            ]
            loss_plot = loss_df.melt(
                "epoch", value_vars=loss_cols, var_name="loss_type", value_name="value"
            )
            fig_loss = px.line(
                loss_plot,
                x="epoch",
                y="value",
                color="loss_type",
                template=template,
            )
            fig_loss.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_loss, use_container_width=True)

        feature_cols = [
            feature
            for feature in summary["feature"].tolist()
            if feature in latent_df.columns
        ]
        if feature_cols and not latent_df.empty:
            from oqp.research.latent import compute_manual_feature_profile

            profile = compute_manual_feature_profile(latent_df, feature_cols)
            st.markdown(f"#### {t['manual_profile']}")
            if not profile.empty:
                heat_cols = [col for col in profile.columns if col.endswith("_mean")]
                heat_profile = profile[["vq_code", *heat_cols]].set_index("vq_code")
                heat_profile.columns = [
                    col.removesuffix("_mean") for col in heat_profile.columns
                ]
                fig_profile = go.Figure(
                    data=go.Heatmap(
                        z=heat_profile.values,
                        x=heat_profile.columns,
                        y=heat_profile.index,
                        colorscale="RdBu",
                        reversescale=True,
                        colorbar=dict(title="mean"),
                    )
                )
                fig_profile.update_layout(
                    template=template, height=420, margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_profile, use_container_width=True)

        gmm_path = os.path.join(
            ALPHA_RUNTIME_DATA_ROOT, "regime", "GMM_Rolling_Probabilities.parquet"
        )
        gmm_diag = attach_gmm_diagnostics(latent_df, gmm_path)
        gmm_row_pct = gmm_diag.get("gmm_row_pct", pd.DataFrame())
        st.markdown(f"#### {t['gmm_overlap']}")
        if gmm_row_pct.empty:
            st.info(t["gmm_missing"])
        else:
            gmm_heat = gmm_row_pct.set_index("vq_code")
            fig_gmm = go.Figure(
                data=go.Heatmap(
                    z=gmm_heat.values,
                    x=[f"GMM {col}" for col in gmm_heat.columns],
                    y=gmm_heat.index,
                    colorscale="Blues",
                    colorbar=dict(title="row %"),
                    zmin=0,
                    zmax=1,
                )
            )
            fig_gmm.update_layout(
                template=template, height=420, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_gmm, use_container_width=True)

        st.markdown(f"#### {t['latent_table']}")
        show_latent_cols = [
            col
            for col in [
                "date",
                "ticker",
                "sector",
                "vq_code",
                "vq_distance",
                target_col,
                "z_vq_01",
                "z_vq_02",
                "z_vq_03",
                "z_vq_04",
            ]
            if col in latent_df.columns
        ]
        st.dataframe(
            latent_df[show_latent_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )

with tabs[7]:
    st.markdown(f"### {t['shortlist']}")
    if keeper_features.empty:
        st.info("No representative feature shortlist available.")
    else:
        keep_cols = [
            "cluster_id",
            "cluster_size",
            "feature",
            "family",
            "quality_score",
            "mean_ic",
            "ic_ir",
            "positive_day_rate",
            "missing_pct",
            "turnover_proxy",
        ]
        st.dataframe(
            _format_percent_cols(keeper_features[keep_cols]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(f"### {t['comparison']}")
    st.markdown(t["comparison_text"])

    protocol_df = pd.DataFrame(
        [
            {
                "feature_set": "Raw engineered features",
                "status": "Available",
                "purpose": "Transparent baseline used by current ML models",
            },
            {
                "feature_set": "Correlation-cluster representatives",
                "status": "Available",
                "purpose": "Reduce duplicate risk while preserving interpretation",
            },
            {
                "feature_set": "PCA components",
                "status": "Diagnostic baseline",
                "purpose": "Linear compression comparison",
            },
            {
                "feature_set": "VQ-VAE latent variables",
                "status": "Scaffolded",
                "purpose": "Discrete nonlinear market-state cross-check",
            },
        ]
    )
    st.dataframe(protocol_df, use_container_width=True, hide_index=True)
