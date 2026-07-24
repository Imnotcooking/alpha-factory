from __future__ import annotations

import json
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

from config import ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, DB_PATH, get_plotly_template
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)
from views.model_catalog_panel import render_model_catalog_panel

ROOT_DIR = os.path.dirname(UI_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from oqp.research.ml import (
    FeatureGovernanceConfig,
    compute_feature_governance,
    infer_feature_matrix_asset_class,
    list_ml_experiments,
    list_matrix_files,
    load_matrix,
    observed_feature_matrix_asset_classes,
    prepare_feature_matrix_taxonomy,
    probe_model_runtime,
    scope_feature_matrix,
)
from oqp.research.ml import PurgedMDAConfig, compute_oos_mda
from oqp.research.model_registry import list_model_artifacts
from apps.research_dashboard.services.latent_artifacts import (
    attach_gmm_diagnostics,
    load_saved_latents,
    save_latent_artifacts,
)
from departments.research.workflows.latent.temporal_vqvae import train_temporal_vqvae_latents


from oqp.ui.translations import research_page_legacy_catalog
from oqp.ui.asset_taxonomy import lane_label


PAGE_TEXT = research_page_legacy_catalog("alpha_feature_governance")


st.set_page_config(page_title="ML Hub", layout="wide")
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
    asset_class: str,
    default_asset_class: str,
    target_col: str,
    corr_threshold: float,
    min_assets_per_day: int,
) -> dict:
    matrix = scope_feature_matrix(
        load_matrix(matrix_path),
        asset_class,
        default_asset_class=default_asset_class,
    )
    config = FeatureGovernanceConfig(
        target_col=target_col,
        include_prob_features=False,
        corr_threshold=corr_threshold,
        min_assets_per_day=min_assets_per_day,
    )
    return compute_feature_governance(matrix, config)


@st.cache_data(show_spinner=False)
def _cached_oos_mda(
    matrix_path: str,
    matrix_mtime: float,
    asset_class: str,
    default_asset_class: str,
    target_col: str,
    feature_cols: tuple[str, ...],
    n_splits: int,
    embargo_days: int,
    max_rows: int,
    max_features: int,
    permutation_repeats: int,
    min_assets_per_day: int,
) -> dict:
    matrix = scope_feature_matrix(
        load_matrix(matrix_path),
        asset_class,
        default_asset_class=default_asset_class,
    )
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


def _json_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return {}
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _json_dict(value: object) -> dict:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, dict) else {}


def _experiment_table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        validation = _json_dict(row.get("validation_policy_json"))
        metrics = _json_dict(row.get("metrics_json"))
        records.append(
            {
                "experiment_id": row.get("experiment_id"),
                "created_at": row.get("created_at"),
                "status": row.get("status"),
                "model": row.get("model_type"),
                "factor_id": row.get("factor_id"),
                "asset_class": row.get("asset_class"),
                "target": row.get("target_col"),
                "validation": validation.get("mode"),
                "folds": metrics.get("fold_count"),
                "oos_rank_ic": metrics.get("oos_mean_daily_spearman_ic"),
                "features": row.get("feature_count"),
                "oos_rows": row.get("prediction_rows"),
            }
        )
    frame = pd.DataFrame(records)
    if "oos_rank_ic" in frame.columns:
        frame["oos_rank_ic"] = pd.to_numeric(frame["oos_rank_ic"], errors="coerce")
    return frame


def _registry_table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        metrics = _json_dict(row.get("metrics_json"))
        split_policy = _json_dict(row.get("split_policy_json"))
        artifact_path = _workspace_path(row.get("artifact_path"))
        records.append(
            {
                "artifact_id": row.get("artifact_id"),
                "created_at": row.get("created_at"),
                "model_name": row.get("model_name"),
                "model_type": row.get("model_type"),
                "factor_id": row.get("factor_id"),
                "target": row.get("target_col"),
                "validation": split_policy.get("mode"),
                "features": row.get("feature_count"),
                "oos_rank_ic": metrics.get("oos_mean_daily_spearman_ic"),
                "artifact_format": row.get("artifact_format"),
                "artifact_available": bool(artifact_path and artifact_path.is_file()),
                "artifact_sha256": str(row.get("artifact_sha256") or "")[:12],
                "data_sha256": str(row.get("data_sha256") or "")[:12],
            }
        )
    frame = pd.DataFrame(records)
    if "oos_rank_ic" in frame.columns:
        frame["oos_rank_ic"] = pd.to_numeric(frame["oos_rank_ic"], errors="coerce")
    return frame


def _workspace_path(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(BASE_DIR) / path


@st.cache_data(show_spinner=False, ttl=300)
def _cached_runtime_probe(model_type: str) -> dict:
    status = probe_model_runtime(model_type)
    return {
        "model": status.model_type,
        "available": status.available,
        "detail": status.detail,
    }


matrix_files = list_matrix_files(BASE_DIR)
if not matrix_files:
    st.error(t["no_file"])
    st.stop()

st.title(t["page_title"])
st.caption(t["subtitle"])
st.caption(t["pipeline"])

source_cols = st.columns([0.38, 0.32, 0.30], vertical_alignment="bottom")
with source_cols[0]:
    feature_store_root = Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store"
    labels = []
    for path in matrix_files:
        try:
            labels.append(path.relative_to(feature_store_root).as_posix())
        except ValueError:
            labels.append(path.name)
    selected_label = st.selectbox(t["matrix"], labels, index=0)
    selected_path = str(matrix_files[labels.index(selected_label)])

matrix_mtime = os.path.getmtime(selected_path)
try:
    raw_preview_df = _cached_matrix(selected_path, matrix_mtime)
except Exception as exc:
    st.error(f"{t['load_error']}: {exc}")
    st.stop()

default_asset_class = infer_feature_matrix_asset_class(selected_path)
has_declared_taxonomy = any(
    column in raw_preview_df.columns for column in ("asset_class", "market_vertical")
)
taxonomy_preview = prepare_feature_matrix_taxonomy(
    raw_preview_df,
    default_asset_class=default_asset_class,
)
observed_asset_classes = observed_feature_matrix_asset_classes(
    taxonomy_preview,
    default_asset_class=default_asset_class,
)
if not observed_asset_classes:
    st.error(t["taxonomy_empty"])
    st.stop()

with source_cols[1]:
    selected_asset_class = st.selectbox(
        t["asset_class"],
        observed_asset_classes,
        index=0,
        format_func=lambda value: (
            f"{value} - {lane_label(value, language=lang)}"
        ),
        key=f"feature_review_asset_class_{lang.lower()}",
    )

preview_df = scope_feature_matrix(
    taxonomy_preview,
    selected_asset_class,
    default_asset_class=default_asset_class,
)
if preview_df.empty:
    st.error(t["taxonomy_empty"])
    st.stop()

target_candidates = [col for col in preview_df.columns if col.startswith("target_")]
if not target_candidates:
    target_candidates = [col for col in preview_df.columns if "target" in col.lower()]
if not target_candidates:
    target_candidates = ["target_1d_rank"]

with source_cols[2]:
    default_target_idx = (
        target_candidates.index("target_1d_rank")
        if "target_1d_rank" in target_candidates
        else 0
    )
    target_col = st.selectbox(t["target"], target_candidates, index=default_target_idx)

analysis_cols = st.columns(2, vertical_alignment="bottom")
with analysis_cols[0]:
    corr_threshold = st.slider(t["corr_threshold"], 0.60, 0.98, 0.85, 0.01)

with analysis_cols[1]:
    min_assets = st.slider(t["min_assets"], 3, 30, 5, 1)

if not has_declared_taxonomy:
    st.caption(
        t["taxonomy_inferred"].format(asset_class=selected_asset_class)
    )

try:
    result = _cached_governance(
        selected_path,
        matrix_mtime,
        selected_asset_class,
        default_asset_class,
        target_col,
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

metric_cols = st.columns([0.18, 0.18, 0.16, 0.48])
metric_cols[0].metric(t["rows"], f"{metadata['rows']:,}")
metric_cols[1].metric(t["features"], f"{metadata['features']:,}")
metric_cols[2].metric(t["assets"], f"{metadata['assets']:,}")
if metadata["date_min"] is not None and metadata["date_max"] is not None:
    date_label = (
        f"{pd.to_datetime(metadata['date_min']).date()}"
        f" -> {pd.to_datetime(metadata['date_max']).date()}"
    )
else:
    date_label = "N/A"
metric_cols[3].metric(t["dates"], date_label)

with st.expander(t["manual_title"], expanded=False):
    st.markdown(t["manual"])

try:
    experiment_rows = list_ml_experiments(Path(DB_PATH), limit=250)
    registry_rows = list_model_artifacts(Path(DB_PATH), limit=250)
    registry_error = None
except Exception as exc:
    experiment_rows = []
    registry_rows = []
    registry_error = str(exc)
selected_experiment = None

tabs = st.tabs(
    [
        t["tab_data"],
        t["tab_models"],
        t["tab_evidence"],
        t["tab_explainability"],
        t["tab_registry"],
    ]
)

with tabs[0]:
    st.markdown(f"### {t['data_stage_title']}")
    st.caption(t["data_stage_caption"])
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
            st.plotly_chart(fig_family, width="stretch")
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
            width="stretch",
            hide_index=True,
        )

with tabs[2]:
    st.markdown(f"### {t['evidence_stage_title']}")
    st.caption(t["evidence_stage_caption"])

    st.markdown(f"#### {t['evidence_scope_title']}")
    st.caption(
        t["evidence_scope_caption"].format(
            feature_count=len(summary),
            matrix_name=Path(selected_path).name,
        )
    )
    scope_controls = st.columns(3)
    family_options = [
        t["evidence_all_families"],
        *sorted(summary["family"].dropna().astype(str).unique()),
    ]
    selected_evidence_family = scope_controls[0].selectbox(
        t["evidence_family"],
        family_options,
        key="ml_hub_evidence_family",
    )
    selected_evidence_set = scope_controls[1].selectbox(
        t["evidence_set"],
        [
            t["evidence_all_features"],
            t["evidence_representatives"],
        ],
        key="ml_hub_evidence_set",
    )
    feature_focus_options = [
        t["evidence_all_features"],
        *summary["feature"].astype(str).tolist(),
    ]
    selected_evidence_feature = scope_controls[2].selectbox(
        t["evidence_feature"],
        feature_focus_options,
        key="ml_hub_evidence_feature",
    )

    evidence_summary = summary.copy()
    if selected_evidence_family != t["evidence_all_families"]:
        evidence_summary = evidence_summary[
            evidence_summary["family"].astype(str) == selected_evidence_family
        ]
    if selected_evidence_set == t["evidence_representatives"]:
        representative_features = set(
            clusters.loc[
                clusters["keep_candidate"].fillna(False).astype(bool),
                "feature",
            ].astype(str)
        )
        evidence_summary = evidence_summary[
            evidence_summary["feature"].astype(str).isin(representative_features)
        ]
    if selected_evidence_feature != t["evidence_all_features"]:
        evidence_summary = evidence_summary[
            evidence_summary["feature"].astype(str) == selected_evidence_feature
        ]

    visible_features = evidence_summary["feature"].astype(str).tolist()
    visible_feature_set = set(visible_features)
    scope_metrics = st.columns(4)
    scope_metrics[0].metric(
        t["visible_features"],
        f"{len(visible_features)} / {len(summary)}",
    )
    scope_metrics[1].metric(
        t["visible_families"],
        f"{evidence_summary['family'].nunique():,}",
    )
    scope_metrics[2].metric(
        t["mean_abs_rank_ic"],
        "N/A"
        if evidence_summary.empty
        else f"{evidence_summary['abs_mean_ic'].mean():.4f}",
    )
    scope_metrics[3].metric(
        t["median_coverage"],
        "N/A"
        if evidence_summary.empty
        else f"{evidence_summary['coverage_pct'].median():.1%}",
    )

    st.markdown(f"##### {t['evidence_inventory_title']}")
    if evidence_summary.empty:
        st.warning(t["evidence_empty_scope"])
    else:
        cluster_scope = clusters[
            [
                "feature",
                "cluster_id",
                "cluster_size",
                "keep_candidate",
            ]
        ].copy()
        evidence_inventory = evidence_summary.merge(
            cluster_scope,
            on="feature",
            how="left",
        )
        evidence_inventory_cols = [
            "feature",
            "family",
            "mean_ic",
            "ic_ir",
            "positive_day_rate",
            "valid_ic_days",
            "coverage_pct",
            "turnover_proxy",
            "cluster_id",
            "cluster_size",
            "keep_candidate",
        ]
        st.dataframe(
            _format_percent_cols(evidence_inventory[evidence_inventory_cols]),
            width="stretch",
            hide_index=True,
        )

        st.markdown(f"### {t['predictiveness_title']}")
        st.caption(t["predictiveness_caption"])
        ic_plot = evidence_summary.sort_values("mean_ic", ascending=True).copy()
        fig_ic = px.bar(
            ic_plot,
            x="mean_ic",
            y="feature",
            orientation="h",
            color="family",
            hover_data=["ic_ir", "positive_day_rate", "valid_ic_days"],
            labels={
                "mean_ic": t["mean_rank_ic_axis"],
                "feature": t["feature_axis"],
                "family": t["evidence_family"],
            },
            template=template,
        )
        fig_ic.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_ic.update_layout(
            height=max(320, min(680, 42 * len(ic_plot) + 120)),
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text=None,
        )
        st.plotly_chart(fig_ic, width="stretch")

        with st.expander(t["implementation_friction_title"], expanded=False):
            st.caption(t["implementation_friction_caption"])
            scatter_df = evidence_summary.dropna(
                subset=["turnover_proxy", "mean_ic"]
            ).copy()
            if scatter_df.empty:
                st.info(t["implementation_friction_empty"])
            else:
                fig_turnover = px.scatter(
                    scatter_df,
                    x="turnover_proxy",
                    y="mean_ic",
                    color="family",
                    size="coverage_pct",
                    hover_name="feature",
                    hover_data=["ic_ir", "positive_day_rate", "missing_pct"],
                    labels={
                        "turnover_proxy": t["rank_turnover_axis"],
                        "mean_ic": t["mean_rank_ic_axis"],
                        "family": t["evidence_family"],
                    },
                    template=template,
                )
                fig_turnover.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_turnover.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend_title_text=None,
                )
                st.plotly_chart(fig_turnover, width="stretch")

        st.markdown(f"### {t['stability_title']}")
        st.caption(t["stability_caption"])
        if daily_ic.empty:
            st.info(t["daily_ic_empty"])
        else:
            default_features = _top_features(evidence_summary, min(3, len(evidence_summary)))
            stability_key = (
                f"ml_hub_evidence_stability_{selected_evidence_family}_"
                f"{selected_evidence_set}_{selected_evidence_feature}"
            )
            selected_features = st.multiselect(
                t["stability_features"],
                visible_features,
                default=default_features,
                key=stability_key,
            )
            plot_daily = daily_ic[
                daily_ic["feature"].astype(str).isin(selected_features)
            ].copy()
            if plot_daily.empty:
                st.info(t["stability_select_feature"])
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
                    labels={
                        "date": t["date_axis"],
                        "rolling_ic": t["rolling_rank_ic_axis"],
                        "feature": t["feature_axis"],
                    },
                    template=template,
                )
                fig_daily.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_daily.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend_title_text=None,
                )
                st.plotly_chart(fig_daily, width="stretch")

        st.markdown(f"### {t['distinctiveness_title']}")
        st.caption(t["distinctiveness_caption"])
        if len(visible_features) < 2:
            st.info(t["distinctiveness_single_feature"])
        else:
            heat = corr_matrix.loc[visible_features, visible_features]
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
                height=max(360, min(680, 38 * len(visible_features) + 140)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_corr, width="stretch")

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
        visible_clusters = clusters[
            clusters["feature"].astype(str).isin(visible_feature_set)
        ]
        st.dataframe(
            _format_percent_cols(visible_clusters[cluster_cols]),
            width="stretch",
            hide_index=True,
        )

        visible_pairs = corr_pairs[
            corr_pairs["feature_a"].astype(str).isin(visible_feature_set)
            & corr_pairs["feature_b"].astype(str).isin(visible_feature_set)
        ]
        with st.expander(t["pairs_title"], expanded=False):
            if visible_pairs.empty:
                st.caption(t["no_pairs"])
            else:
                st.dataframe(
                    _format_percent_cols(visible_pairs.head(50)),
                    width="stretch",
                    hide_index=True,
                )

with tabs[2]:
    st.markdown(f"### {t['mda_title']}")
    st.caption(t["mda_section_caption"])
    with st.expander(t["mda_setup"], expanded=False):
        st.markdown(t["mda_help"])

        mda_controls = st.columns([0.16, 0.14, 0.16, 0.18, 0.18, 0.18])
        with mda_controls[0]:
            mda_max_features = st.select_slider(
                t["mda_features"], options=[10, 15, 25, 40], value=25
            )
        with mda_controls[1]:
            mda_folds = st.select_slider(
                t["mda_folds"], options=[3, 4, 5, 7], value=5
            )
        with mda_controls[2]:
            mda_embargo = st.select_slider(
                t["mda_embargo"], options=[0, 4, 8, 20], value=8
            )
        with mda_controls[3]:
            mda_max_rows = st.select_slider(
                t["mda_rows"],
                options=[10_000, 25_000, 50_000, 100_000],
                value=50_000,
            )
        with mda_controls[4]:
            mda_repeats = st.select_slider(
                t["mda_repeats"], options=[1, 2, 3], value=1
            )
        with mda_controls[5]:
            st.write("")
            run_mda = st.button(
                t["mda_run"],
                width="stretch",
                disabled=evidence_summary.empty,
            )

    mda_feature_cols = tuple(
        _top_features(evidence_summary, int(mda_max_features))
    )
    mda_key = (
        selected_path,
        matrix_mtime,
        selected_asset_class,
        default_asset_class,
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
        st.caption(t["mda_not_run"])
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
                st.plotly_chart(fig_mda, width="stretch")

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
                    st.plotly_chart(fig_gain, width="stretch")

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
                width="stretch",
                hide_index=True,
            )

        if not mda_fold_scores.empty:
            with st.expander(t["mda_fold_scores"], expanded=False):
                st.dataframe(mda_fold_scores, width="stretch", hide_index=True)

with tabs[0]:
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
    st.plotly_chart(fig_missing, width="stretch")

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
        width="stretch",
        hide_index=True,
    )

with tabs[1]:
    st.markdown(f"### {t['models_stage_title']}")
    st.caption(t["models_stage_caption"])

    render_model_catalog_panel(t, language=lang)

    runtime_header = st.columns([0.76, 0.24], vertical_alignment="center")
    runtime_header[0].markdown(f"#### {t['runtime_readiness']}")
    if runtime_header[1].button(
        t["check_runtimes"],
        width="stretch",
        key="ml_hub_check_runtimes",
    ):
        with st.spinner(t["runtime_checking"]):
            st.session_state["ml_hub_runtime_status"] = [
                _cached_runtime_probe(model_type)
                for model_type in ("lightgbm", "xgboost")
            ]

    runtime_rows = st.session_state.get("ml_hub_runtime_status")
    if runtime_rows:
        runtime_frame = pd.DataFrame(runtime_rows)
        runtime_frame["status"] = runtime_frame["available"].map(
            {True: t["ready"], False: t["unavailable"]}
        )
        st.dataframe(
            runtime_frame[["model", "status", "detail"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(t["runtime_not_checked"])

    experiment_frame = _experiment_table(experiment_rows)
    completed_count = sum(row.get("status") == "completed" for row in experiment_rows)
    failed_count = sum(row.get("status") == "failed" for row in experiment_rows)
    latest_ic = np.nan
    if not experiment_frame.empty and "oos_rank_ic" in experiment_frame:
        numeric_ic = pd.to_numeric(experiment_frame["oos_rank_ic"], errors="coerce").dropna()
        if not numeric_ic.empty:
            latest_ic = float(numeric_ic.iloc[0])

    experiment_metrics = st.columns(4)
    experiment_metrics[0].metric(t["experiments"], f"{len(experiment_rows):,}")
    experiment_metrics[1].metric(t["completed"], f"{completed_count:,}")
    experiment_metrics[2].metric(t["failed"], f"{failed_count:,}")
    experiment_metrics[3].metric(
        t["latest_oos_ic"],
        "N/A" if pd.isna(latest_ic) else f"{latest_ic:.4f}",
    )

    st.markdown(f"#### {t['experiment_ledger']}")
    if registry_error:
        st.warning(f"{t['registry_error']}: {registry_error}")
    elif experiment_frame.empty:
        st.info(t["no_experiments"])
    else:
        filter_cols = st.columns(2)
        model_options = [t["all_models"], *sorted(experiment_frame["model"].dropna().unique())]
        selected_model_filter = filter_cols[0].selectbox(
            t["model_filter"],
            model_options,
            key="ml_hub_model_filter",
        )
        asset_options = [t["all_assets"], *sorted(experiment_frame["asset_class"].dropna().unique())]
        selected_asset_filter = filter_cols[1].selectbox(
            t["asset_filter"],
            asset_options,
            key="ml_hub_asset_filter",
        )
        filtered_experiments = experiment_frame.copy()
        if selected_model_filter != t["all_models"]:
            filtered_experiments = filtered_experiments[
                filtered_experiments["model"] == selected_model_filter
            ]
        if selected_asset_filter != t["all_assets"]:
            filtered_experiments = filtered_experiments[
                filtered_experiments["asset_class"] == selected_asset_filter
            ]
        st.dataframe(
            filtered_experiments.style.format({"oos_rank_ic": "{:.4f}"}, na_rep="N/A"),
            width="stretch",
            hide_index=True,
        )

        experiment_ids = filtered_experiments["experiment_id"].astype(str).tolist()
        if experiment_ids:
            selected_experiment_id = st.selectbox(
                t["inspect_experiment"],
                experiment_ids,
                key="ml_hub_experiment_id",
            )
            selected_experiment = next(
                row
                for row in experiment_rows
                if str(row.get("experiment_id")) == selected_experiment_id
            )
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.markdown(f"##### {t['validation_policy']}")
                st.json(_json_dict(selected_experiment.get("validation_policy_json")))
            with detail_cols[1]:
                st.markdown(f"##### {t['experiment_metrics']}")
                st.json(_json_dict(selected_experiment.get("metrics_json")))
            with st.expander(t["reproducibility_record"], expanded=False):
                st.json(
                    {
                        "data_path": selected_experiment.get("data_path"),
                        "data_sha256": selected_experiment.get("data_sha256"),
                        "artifact_id": selected_experiment.get("artifact_id"),
                        "artifact_path": selected_experiment.get("artifact_path"),
                        "importance_path": selected_experiment.get("importance_path"),
                        "predictions_path": selected_experiment.get("predictions_path"),
                        "hyperparameters": _json_dict(
                            selected_experiment.get("hyperparams_json")
                        ),
                    }
                )

    st.markdown(f"#### {t['training_entrypoint']}")
    st.caption(t["training_entrypoint_caption"])
    command_cols = st.columns(3)
    command_model = command_cols[0].selectbox(
        t["model_adapter"],
        ["lightgbm", "xgboost"],
        key="ml_hub_command_model",
    )
    command_validation = command_cols[1].selectbox(
        t["validation_mode"],
        ["walk_forward", "fixed_date"],
        key="ml_hub_command_validation",
    )
    command_factor = command_cols[2].text_input(
        t["factor_module"],
        value="fac_054_XGBoost_Alpha",
        key="ml_hub_command_factor",
    )
    selected_cli_path = Path(selected_path)
    try:
        selected_cli_path = selected_cli_path.resolve().relative_to(Path(BASE_DIR).resolve())
    except ValueError:
        pass
    command = (
        "python scripts/research/run_ml_backtest.py "
        f"--asset {selected_asset_class} --factor {command_factor} "
        f"--model {command_model} --feature-matrix {selected_cli_path.as_posix()} "
        f"--target-column {target_col} --validation-mode {command_validation} --retrain"
    )
    st.code(command, language="bash")

with tabs[3]:
    st.markdown(f"### {t['explain_stage_title']}")
    st.caption(t["explain_stage_caption"])

    explain_candidates = [
        row for row in experiment_rows if row.get("status") == "completed"
    ]
    selected_explain_experiment = None
    if not explain_candidates:
        st.info(t["no_explainable_experiments"])
    else:
        explain_labels = {
            str(row.get("experiment_id")): (
                f"{row.get('experiment_id')} | {row.get('model_type')} | "
                f"{row.get('factor_id') or 'no factor'}"
            )
            for row in explain_candidates
        }
        selected_explain_id = st.selectbox(
            t["explain_experiment"],
            list(explain_labels),
            format_func=explain_labels.get,
            key="ml_hub_explain_experiment",
        )
        selected_explain_experiment = next(
            row
            for row in explain_candidates
            if str(row.get("experiment_id")) == selected_explain_id
        )
        explain_metrics = _json_dict(selected_explain_experiment.get("metrics_json"))
        explain_metric_cols = st.columns(4)
        explain_metric_cols[0].metric(
            t["model_adapter"],
            str(selected_explain_experiment.get("model_type") or "N/A"),
        )
        explain_metric_cols[1].metric(
            t["target"],
            str(selected_explain_experiment.get("target_col") or "N/A"),
        )
        explain_rank_ic = pd.to_numeric(
            explain_metrics.get("oos_mean_daily_spearman_ic"),
            errors="coerce",
        )
        explain_metric_cols[2].metric(
            t["oos_rank_ic"],
            "N/A" if pd.isna(explain_rank_ic) else f"{float(explain_rank_ic):.4f}",
        )
        explain_metric_cols[3].metric(
            t["features"],
            f"{int(selected_explain_experiment.get('feature_count') or 0):,}",
        )

    importance_path = (
        _workspace_path(selected_explain_experiment.get("importance_path"))
        if selected_explain_experiment
        else None
    )
    if importance_path and importance_path.is_file():
        try:
            importance_df = pd.read_csv(importance_path)
        except Exception as exc:
            st.warning(f"{t['importance_error']}: {exc}")
        else:
            numeric_candidates = [
                column
                for column in importance_df.columns
                if column != "feature"
                and pd.api.types.is_numeric_dtype(importance_df[column])
            ]
            if "feature" in importance_df.columns and numeric_candidates:
                importance_col = (
                    "importance_mean"
                    if "importance_mean" in numeric_candidates
                    else numeric_candidates[0]
                )
                importance_values = (
                    importance_df[["feature", importance_col]]
                    .dropna()
                    .assign(
                        absolute_importance=lambda frame: frame[importance_col].abs()
                    )
                    .sort_values("absolute_importance", ascending=False)
                )
                total_importance = float(
                    importance_values["absolute_importance"].sum()
                )
                concentration_cols = st.columns(3)
                concentration_cols[0].metric(
                    t["explained_features"],
                    f"{len(importance_values):,}",
                )
                concentration_cols[1].metric(
                    t["top_feature_share"],
                    "N/A"
                    if total_importance <= 0
                    else f"{importance_values['absolute_importance'].iloc[0] / total_importance:.1%}",
                )
                concentration_cols[2].metric(
                    t["top_five_share"],
                    "N/A"
                    if total_importance <= 0
                    else f"{importance_values['absolute_importance'].head(5).sum() / total_importance:.1%}",
                )
                importance_plot = (
                    importance_values
                    .sort_values(importance_col, ascending=True)
                    .tail(25)
                )
                st.markdown(f"#### {t['saved_importance']}")
                fig_importance = px.bar(
                    importance_plot,
                    x=importance_col,
                    y="feature",
                    orientation="h",
                    template=template,
                )
                fig_importance.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_importance, width="stretch")
                st.dataframe(
                    importance_values[
                        ["feature", importance_col, "absolute_importance"]
                    ].head(50),
                    width="stretch",
                    hide_index=True,
                )
    elif selected_explain_experiment:
        st.info(t["importance_unavailable"])

with tabs[0]:
    st.markdown(f"#### {t['pca_diagnostic']}")
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
            st.plotly_chart(fig_pca, width="stretch")

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
            st.plotly_chart(fig_load, width="stretch")

        st.dataframe(
            pca_variance.style.format(
                {
                    "explained_variance_ratio": "{:.2%}",
                    "cumulative_variance": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with tabs[1]:
    st.markdown(f"### {t['latent_title']}")
    st.warning(t["latent_legacy_warning"])
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
        train_clicked = st.button(t["train_latent"], width="stretch")
    with control_row[6]:
        load_clicked = st.button(t["load_latent"], width="stretch")

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
            from oqp.research.ml.latent.diagnostics import compute_code_target_ic

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
            st.plotly_chart(fig_usage, width="stretch")

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
            st.plotly_chart(fig_loss, width="stretch")

        feature_cols = [
            feature
            for feature in summary["feature"].tolist()
            if feature in latent_df.columns
        ]
        if feature_cols and not latent_df.empty:
            from oqp.research.ml.latent.diagnostics import compute_manual_feature_profile

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
                st.plotly_chart(fig_profile, width="stretch")

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
            st.plotly_chart(fig_gmm, width="stretch")

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
            width="stretch",
            hide_index=True,
        )

with tabs[4]:
    st.markdown(f"### {t['registry_stage_title']}")
    st.caption(t["registry_stage_caption"])

    registry_frame = _registry_table(registry_rows)
    if registry_error:
        st.warning(f"{t['registry_error']}: {registry_error}")
    elif registry_frame.empty:
        st.info(t["no_registered_models"])
    else:
        registry_metrics = st.columns(4)
        registry_metrics[0].metric(t["registered_artifacts"], f"{len(registry_frame):,}")
        registry_metrics[1].metric(
            t["registered_models"],
            f"{registry_frame['model_name'].nunique():,}",
        )
        registry_metrics[2].metric(
            t["available_artifacts"],
            f"{int(registry_frame['artifact_available'].sum()):,}",
        )
        latest_registry_date = pd.to_datetime(
            registry_frame["created_at"], errors="coerce"
        ).max()
        registry_metrics[3].metric(
            t["latest_registration"],
            "N/A"
            if pd.isna(latest_registry_date)
            else latest_registry_date.strftime("%Y-%m-%d"),
        )

        registry_filters = st.columns(2)
        registry_model_types = [
            t["all_models"],
            *sorted(registry_frame["model_type"].dropna().unique()),
        ]
        selected_registry_model = registry_filters[0].selectbox(
            t["model_filter"],
            registry_model_types,
            key="ml_hub_registry_model_filter",
        )
        registry_factors = [
            t["all_factors"],
            *sorted(registry_frame["factor_id"].dropna().unique()),
        ]
        selected_registry_factor = registry_filters[1].selectbox(
            t["factor_filter"],
            registry_factors,
            key="ml_hub_registry_factor_filter",
        )
        filtered_registry = registry_frame.copy()
        if selected_registry_model != t["all_models"]:
            filtered_registry = filtered_registry[
                filtered_registry["model_type"] == selected_registry_model
            ]
        if selected_registry_factor != t["all_factors"]:
            filtered_registry = filtered_registry[
                filtered_registry["factor_id"] == selected_registry_factor
            ]

        st.markdown(f"#### {t['model_registry']}")
        st.dataframe(
            filtered_registry.style.format(
                {"oos_rank_ic": "{:.4f}"},
                na_rep="N/A",
            ),
            width="stretch",
            hide_index=True,
        )

        visible_artifact_ids = filtered_registry["artifact_id"].astype(str).tolist()
        if visible_artifact_ids:
            selected_artifact_id = st.selectbox(
                t["inspect_artifact"],
                visible_artifact_ids,
                key="ml_hub_registry_artifact_id",
            )
            selected_artifact = next(
                row
                for row in registry_rows
                if str(row.get("artifact_id")) == selected_artifact_id
            )
            registry_detail_cols = st.columns(2)
            with registry_detail_cols[0]:
                st.markdown(f"##### {t['validation_contract']}")
                st.json(_json_dict(selected_artifact.get("split_policy_json")))
            with registry_detail_cols[1]:
                st.markdown(f"##### {t['artifact_fingerprints']}")
                st.json(
                    {
                        "artifact_path": selected_artifact.get("artifact_path"),
                        "artifact_sha256": selected_artifact.get("artifact_sha256"),
                        "data_path": selected_artifact.get("data_path"),
                        "data_sha256": selected_artifact.get("data_sha256"),
                    }
                )
            with st.expander(t["artifact_configuration"], expanded=False):
                st.json(
                    {
                        "features": _json_value(
                            selected_artifact.get("feature_cols_json")
                        ),
                        "hyperparameters": _json_dict(
                            selected_artifact.get("hyperparams_json")
                        ),
                        "metrics": _json_dict(selected_artifact.get("metrics_json")),
                        "metadata": _json_dict(selected_artifact.get("metadata_json")),
                    }
                )
