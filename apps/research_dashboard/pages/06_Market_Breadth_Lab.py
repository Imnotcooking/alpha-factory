from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(UI_DIR)
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, get_plotly_template
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)
from oqp.data.asset_taxonomy import (
    LANE_METADATA,
    is_options_asset_class,
    is_vectorizable_asset_class,
    load_asset_taxonomy,
)
from oqp.data.runtime_paths import (
    default_futures_cn_index_daily_file,
    discover_asset_class_files,
    discover_futures_cn_daily_files,
)
from oqp.risk.factor_breadth import (
    RiskBreadthConfig,
    classify_breadth_regimes,
    compute_risk_factor_breadth,
    summarize_breadth_regime_periods,
)


DEFAULT_SOURCE = default_futures_cn_index_daily_file()
CACHE_VERSION = "market_breadth_v6_regime_periods"


from oqp.ui.translations import research_page_legacy_catalog


PAGE_TEXT = research_page_legacy_catalog("risk_factor_breadth_lab")


st.set_page_config(page_title="Market Breadth Lab", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang if st.session_state.lang in PAGE_TEXT else "EN"
t = PAGE_TEXT[lang]
template = get_plotly_template(st.session_state.theme_mode)


def _asset_class_options() -> list[str]:
    taxonomy = load_asset_taxonomy(BASE_DIR)
    ordered = ["FUTURES_CN", *sorted(key for key in taxonomy if key != "FUTURES_CN")]
    return ordered


def _asset_label(asset_class: str) -> str:
    lane = LANE_METADATA.get(asset_class, {})
    label = lane.get("label", asset_class)
    label_zh = lane.get("label_zh", label)
    if lang == "ZH" and label_zh:
        return f"{asset_class} - {label_zh}"
    return f"{asset_class} - {label}"


def _list_daily_files(asset_class: str) -> list[Path]:
    runtime_feature_store = Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store"
    seen: dict[Path, None] = {}
    for path in discover_asset_class_files(asset_class, timeframe="daily"):
        seen[path.resolve()] = None
    if asset_class == "FUTURES_CN":
        for path in discover_futures_cn_daily_files():
            seen[path.resolve()] = None
        for pattern in ("ML_Feature_Matrix.parquet",):
            for path in sorted(runtime_feature_store.glob(pattern) if runtime_feature_store.exists() else []):
                if path.is_file():
                    seen[path.resolve()] = None
        root_matrix = (
            Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store" / "ML_Feature_Matrix.parquet"
        )
        if root_matrix.exists():
            seen[root_matrix.resolve()] = None
    files = list(seen.keys())
    if asset_class == "FUTURES_CN" and DEFAULT_SOURCE.exists():
        default_resolved = DEFAULT_SOURCE.resolve()
        files = [
            default_resolved,
            *[path for path in files if path != default_resolved],
        ]
    return files


def _source_labels(files: list[Path]) -> list[str]:
    base_dir = Path(BASE_DIR).resolve()
    labels = []
    for path in files:
        label = path.name
        if label in labels:
            try:
                label = str(path.relative_to(base_dir))
            except ValueError:
                label = str(path)
        labels.append(label)
    return labels


@st.cache_data(show_spinner=False)
def _cached_risk_breadth(
    source_path: str,
    source_mtime: float,
    asset_class: str,
    variance_threshold: float,
    rolling_window: int,
    max_assets: int,
    risk_imputation: str,
    cache_version: str,
) -> dict:
    config = RiskBreadthConfig(
        variance_threshold=variance_threshold,
        rolling_window=rolling_window,
        rolling_step=21,
        asset_class=asset_class,
        max_assets=max_assets,
        risk_imputation=risk_imputation,
    )
    return compute_risk_factor_breadth(source_path, config=config)


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def _format_num(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}"


def _fallback_component_labels(
    sector_abs: pd.DataFrame, asset_loadings: pd.DataFrame
) -> pd.DataFrame:
    if sector_abs.empty or asset_loadings.empty:
        return pd.DataFrame()

    rows = []
    components = (
        sector_abs[["component", "component_idx"]]
        .drop_duplicates()
        .sort_values("component_idx")
        .head(20)
    )
    industrial = {"化工", "黑色", "能源", "有色", "建材"}
    agriculture = {"油脂油料", "软商品", "生鲜"}

    for _, component_row in components.iterrows():
        component = component_row["component"]
        sector_slice = sector_abs[sector_abs["component"] == component].sort_values(
            "abs_loading_share", ascending=False
        )
        asset_slice = asset_loadings[
            asset_loadings["component"] == component
        ].sort_values("abs_loading", ascending=False)
        top_sectors = sector_slice.head(3)["sector"].tolist()
        top_assets = asset_slice.head(5)["base_symbol"].tolist()
        top_set = set(top_sectors)

        if len(top_set & industrial) >= 2:
            label_en = "Cyclical industrial commodity beta"
            label_zh = "工业品周期共同因子"
            interpretation_en = "Dominated by industrial commodity sectors."
            interpretation_zh = "主要由工业品板块驱动。"
        elif len(top_set & agriculture) >= 2:
            label_en = "Agricultural and soft-commodity beta"
            label_zh = "农产品与软商品共同因子"
            interpretation_en = "Dominated by agricultural or soft-commodity sectors."
            interpretation_zh = "主要由农产品或软商品板块驱动。"
        elif "国债" in top_set:
            label_en = "Rates duration factor"
            label_zh = "利率久期因子"
            interpretation_en = "Dominated by Chinese government bond futures."
            interpretation_zh = "主要由国债期货驱动。"
        elif "股指" in top_set:
            label_en = "Equity index beta"
            label_zh = "股指风险因子"
            interpretation_en = "Dominated by equity-index futures."
            interpretation_zh = "主要由股指期货驱动。"
        elif "贵金属" in top_set:
            label_en = "Precious-metals factor"
            label_zh = "贵金属因子"
            interpretation_en = "Dominated by precious metals."
            interpretation_zh = "主要由贵金属驱动。"
        else:
            label_en = "Mixed sector factor"
            label_zh = "混合板块因子"
            interpretation_en = "No single economic family dominates cleanly."
            interpretation_zh = "没有单一经济板块能清晰解释该主成分。"

        rows.append(
            {
                "component": component,
                "component_idx": int(component_row["component_idx"]),
                "label_en": label_en,
                "label_zh": label_zh,
                "top_sectors": ", ".join(top_sectors),
                "top_assets": ", ".join(top_assets),
                "positive_basket_en": ", ".join(top_assets[:3]) or "N/A",
                "positive_basket_zh": ", ".join(top_assets[:3]) or "N/A",
                "negative_basket_en": "N/A",
                "negative_basket_zh": "N/A",
                "top3_sector_share": float(
                    sector_slice.head(3)["abs_loading_share"].sum()
                ),
                "label_confidence": float(
                    min(1.0, max(0.0, sector_slice.head(3)["abs_loading_share"].sum()))
                ),
                "interpretation_en": interpretation_en,
                "interpretation_zh": interpretation_zh,
            }
        )

    return pd.DataFrame(rows)


st.title(t["title"])
st.caption(t["subtitle"])

asset_options = _asset_class_options()
asset_default = asset_options.index("FUTURES_CN") if "FUTURES_CN" in asset_options else 0
asset_class = st.selectbox(
    t.get("asset_class", "Asset class"),
    asset_options,
    index=asset_default,
    format_func=_asset_label,
)

if is_options_asset_class(asset_class):
    st.warning(
        t.get(
            "options_note",
            "Options market breadth needs a separate options risk engine using underlying returns, IV moves, Greeks, expiry buckets, and liquidity. Plain option price PCA is deliberately disabled here.",
        )
    )
    st.stop()

if not is_vectorizable_asset_class(asset_class):
    st.warning(
        t.get(
            "unsupported_asset",
            "This asset class is not vectorizable in the taxonomy yet, so covariance PCA is disabled.",
        )
    )
    st.stop()

files = _list_daily_files(asset_class)
if not files:
    st.error(
        f"{t['error']}: "
        f"{t.get('no_source', 'no daily parquet files found for this asset class.')}"
    )
    st.stop()

control_cols = st.columns([0.42, 0.16, 0.14, 0.14, 0.14])
with control_cols[0]:
    labels = _source_labels(files)
    selected_label = st.selectbox(t["source"], labels, index=0)
    selected_path = files[labels.index(selected_label)]
with control_cols[1]:
    variance_threshold = st.slider(t["variance"], 0.80, 0.99, 0.95, 0.01)
with control_cols[2]:
    rolling_window = st.select_slider(
        t["window"], options=[252, 504, 756, 1008], value=504
    )
with control_cols[3]:
    default_max_assets = 75 if asset_class == "FUTURES_CN" else 300
    max_assets = st.select_slider(
        t.get("max_assets", "Max assets"),
        options=[50, 75, 150, 300, 500, 1000],
        value=default_max_assets,
    )
with control_cols[4]:
    risk_view_label = st.selectbox(
        t.get("risk_view", "Risk view"),
        [t.get("risk_ffill", "Forward-fill"), t.get("risk_bridge", "Brownian Bridge")],
        index=0,
    )
risk_imputation = "brownian_bridge" if risk_view_label == t.get("risk_bridge", "Brownian Bridge") else "ffill"

try:
    result = _cached_risk_breadth(
        str(selected_path),
        selected_path.stat().st_mtime,
        asset_class,
        float(variance_threshold),
        int(rolling_window),
        int(max_assets),
        risk_imputation,
        CACHE_VERSION,
    )
except Exception as exc:
    st.error(f"{t['error']}: {exc}")
    st.stop()

metrics = result["metrics"]
spectrum = result["spectrum"]
asset_loadings = result["asset_loadings"]
sector_signed = result["sector_signed"]
sector_abs = result["sector_abs"]
component_labels = result.get("component_labels")
if component_labels is None:
    component_labels = _fallback_component_labels(sector_abs, asset_loadings)
rolling = result["rolling_breadth"]
component_stability = result.get("component_stability", pd.DataFrame())
rolling_regimes = classify_breadth_regimes(rolling) if not rolling.empty else pd.DataFrame()
regime_periods = summarize_breadth_regime_periods(rolling_regimes)

st.markdown(f"### {t['cards']}")
if metrics["valid_assets"] < 20:
    st.warning(t["low_assets"])

card_cols = st.columns(6)
card_cols[0].metric(t["naive"], f"{metrics['naive_breadth']:,}")
card_cols[1].metric(t["valid"], f"{metrics['valid_assets']:,}")
card_cols[2].metric(t["br95"], f"{metrics['br_threshold']:,}")
card_cols[3].metric(t["eff_rank"], _format_num(metrics["effective_rank"]))
card_cols[4].metric(t["participation"], _format_num(metrics["participation_ratio"]))
card_cols[5].metric(t["haircut"], _format_pct(metrics["breadth_haircut"]))

st.caption(
    f"{metrics['date_min'].date()} -> {metrics['date_max'].date()} | "
    f"observations={metrics['observations']:,} | "
    f"source_assets={result['source_assets']:,} | selected_assets={result.get('selected_assets', metrics['valid_assets']):,} | "
    f"source_rows={result['source_rows']:,}"
)

if not component_labels.empty:
    pc1 = component_labels[component_labels["component"] == "PC1"].iloc[0]
    pc1_label = pc1["label_zh"] if lang == "ZH" else pc1["label_en"]
    pc1_share = spectrum.loc[
        spectrum["component"] == "PC1", "explained_variance_ratio"
    ].iloc[0]
    st.caption(
        f"{t['pc_label_prefix']}: {pc1_label} | PC1 {pc1_share:.1%} | "
        f"{t['pc1_top_sectors']}: {pc1['top_sectors']}"
    )

st.markdown("---")
st.markdown(f"### {t['spectrum']}")
plot_spectrum = spectrum.head(min(40, len(spectrum))).copy()
fig_spectrum = go.Figure()
fig_spectrum.add_trace(
    go.Bar(
        x=plot_spectrum["component"],
        y=plot_spectrum["explained_variance_ratio"],
        name="Individual",
        marker_color="#40C4FF",
    )
)
fig_spectrum.add_trace(
    go.Scatter(
        x=plot_spectrum["component"],
        y=plot_spectrum["cumulative_variance"],
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="#00C853", width=3),
    )
)
fig_spectrum.add_hline(y=variance_threshold, line_dash="dash", line_color="#FF5252")
fig_spectrum.update_layout(
    template=template,
    height=430,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis_tickformat=".0%",
    yaxis_title="Variance",
)
st.plotly_chart(fig_spectrum, width="stretch")

if not component_labels.empty:
    st.markdown(f"#### {t['component_interpreter']}")
    label_name = "label_zh" if lang == "ZH" else "label_en"
    positive_name = "positive_basket_zh" if lang == "ZH" else "positive_basket_en"
    negative_name = "negative_basket_zh" if lang == "ZH" else "negative_basket_en"
    interp_name = "interpretation_zh" if lang == "ZH" else "interpretation_en"
    material_components = spectrum.loc[
        (spectrum["explained_variance_ratio"] >= 0.02)
        | (spectrum["cumulative_variance"] <= 0.70),
        "component",
    ].head(10).tolist()
    if not material_components:
        material_components = spectrum["component"].head(min(5, len(spectrum))).tolist()
    diagnostics = component_labels[
        component_labels["component"].isin(material_components)
    ].merge(
        spectrum[
            [
                "component",
                "explained_variance_ratio",
                "cumulative_variance",
            ]
        ],
        on="component",
        how="left",
    )
    display_labels = diagnostics[
        [
            "component",
            "explained_variance_ratio",
            "cumulative_variance",
            label_name,
            positive_name,
            negative_name,
            "label_confidence",
            interp_name,
        ]
    ].copy()
    display_labels = display_labels.rename(
        columns={
            "component": t.get("component_col", "PC"),
            "explained_variance_ratio": t.get("variance_col", "Variance"),
            "cumulative_variance": t.get("cumulative_col", "Cumulative"),
            label_name: "label",
            positive_name: t.get("positive_basket", "Positive basket"),
            negative_name: t.get("negative_basket", "Negative basket"),
            "label_confidence": t.get("label_confidence", "Label confidence"),
            interp_name: "interpretation",
        }
    )
    display_labels = display_labels.rename(
        columns={
            "label": t.get("label_col", "Label"),
            "interpretation": t.get("interpretation_col", "Interpretation"),
        }
    )
    st.dataframe(
        display_labels.style.format(
            {
                t.get("variance_col", "Variance"): "{:.1%}",
                t.get("cumulative_col", "Cumulative"): "{:.1%}",
                t.get("label_confidence", "Label confidence"): "{:.0%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(t.get("confidence_note", "Label confidence is heuristic, not a statistical confidence interval."))

if not component_stability.empty:
    st.markdown(f"#### {t.get('component_stability', 'Component Stability')}")
    st.caption(
        t.get(
            "component_stability_note",
            "Rolling PCA stability compares each recent window's component loadings with the full-sample component. It is robustness evidence, not a confidence interval.",
        )
    )
    stability_label_name = "window_label_zh" if lang == "ZH" else "window_label_en"
    stability_sector_name = "window_dominant_sector_zh" if lang == "ZH" else "window_dominant_sector_en"
    if stability_sector_name not in component_stability.columns:
        stability_sector_name = "window_dominant_sector"
    plot_stability = component_stability.dropna(subset=["loading_similarity"]).copy()
    fig_stability = px.line(
        plot_stability,
        x="date",
        y="loading_similarity",
        color="component",
        markers=True,
        template=template,
        labels={
            "date": t.get("date_col", "Date"),
            "loading_similarity": t.get("loading_similarity", "Loading Similarity"),
            "component": t.get("component_col", "PC"),
        },
        hover_data={
            stability_label_name: True,
            stability_sector_name: True,
            "label_confidence": ":.0%",
            "explained_variance_ratio": ":.1%",
        },
    )
    fig_stability.add_hline(y=0.70, line_dash="dash", line_color="#FFAB40")
    fig_stability.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_stability, width="stretch")

    summary = (
        component_stability.groupby("component", as_index=False)
        .agg(
            windows=("date", "nunique"),
            avg_similarity=("loading_similarity", "mean"),
            min_similarity=("loading_similarity", "min"),
            label_match_rate=("label_match", "mean"),
            sector_match_rate=("dominant_sector_match", "mean"),
            avg_label_confidence=("label_confidence", "mean"),
        )
        .sort_values("component")
    )
    latest_labels = (
        component_stability.sort_values("date")
        .groupby("component", as_index=False)
        .tail(1)[["component", stability_label_name, stability_sector_name]]
    )
    summary = summary.merge(latest_labels, on="component", how="left")
    display_stability = summary.rename(
        columns={
            "component": t.get("component_col", "PC"),
            "windows": t.get("windows_col", "Windows"),
            "avg_similarity": t.get("avg_similarity", "Avg Similarity"),
            "min_similarity": t.get("min_similarity", "Min Similarity"),
            "label_match_rate": t.get("label_match_rate", "Label Match"),
            "sector_match_rate": t.get("sector_match_rate", "Sector Match"),
            "avg_label_confidence": t.get("avg_label_confidence", "Avg Label Confidence"),
            stability_label_name: t.get("latest_label", "Latest Label"),
            stability_sector_name: t.get("latest_sector", "Latest Sector"),
        }
    )
    st.dataframe(
        display_stability.style.format(
            {
                t.get("avg_similarity", "Avg Similarity"): "{:.0%}",
                t.get("min_similarity", "Min Similarity"): "{:.0%}",
                t.get("label_match_rate", "Label Match"): "{:.0%}",
                t.get("sector_match_rate", "Sector Match"): "{:.0%}",
                t.get("avg_label_confidence", "Avg Label Confidence"): "{:.0%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
elif int(result.get("component_stability_skipped_windows", 0)) > 0:
    st.info(t.get("component_stability_empty", "Component stability is unavailable for this source/window."))

st.markdown("---")
st.markdown(f"### {t['sector']}")
with st.expander(t["sector_help_title"], expanded=False):
    st.markdown(t["sector_help"])

map_cols = st.columns([0.24, 0.76])
with map_cols[0]:
    map_mode = st.radio(
        t["sector"],
        [t["absolute"], t["signed"]],
        horizontal=False,
        label_visibility="collapsed",
    )
    component_options = spectrum["component"].head(min(20, len(spectrum))).tolist()
    selected_component = st.selectbox(t["component"], component_options)

with map_cols[1]:
    if map_mode == t["absolute"]:
        heat_source = sector_abs[sector_abs["component"].isin(component_options)]
        heat = heat_source.pivot(
            index="sector", columns="component", values="abs_loading_share"
        ).fillna(0.0)
        heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
        fig_sector = px.imshow(
            heat,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="Abs share"),
            template=template,
        )
        fig_sector.update_coloraxes(colorbar_tickformat=".0%")
    else:
        heat_source = sector_signed[sector_signed["component"].isin(component_options)]
        heat = heat_source.pivot(
            index="sector", columns="component", values="signed_mean_loading"
        ).fillna(0.0)
        heat = heat.loc[heat.abs().sum(axis=1).sort_values(ascending=False).index]
        fig_sector = px.imshow(
            heat,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Signed loading"),
            template=template,
        )
    fig_sector.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_sector, width="stretch")

st.markdown(f"#### {t['top_loadings']}: {selected_component}")
top_loadings = asset_loadings[asset_loadings["component"] == selected_component].copy()
top_loadings = top_loadings.sort_values("abs_loading", ascending=False).head(20)
top_loadings["loading"] = top_loadings["loading"].round(4)
top_loadings["abs_loading"] = top_loadings["abs_loading"].round(4)
st.dataframe(
    top_loadings[
        [
            "ticker",
            "base_symbol",
            "sector",
            "loading",
            "abs_loading",
            "explained_variance_ratio",
        ]
    ].style.format({"explained_variance_ratio": "{:.1%}"}),
    width="stretch",
    hide_index=True,
)

st.markdown("---")
st.markdown(f"### {t['rolling']}")
st.caption(f"{t['skipped']}: {result['rolling_skipped_windows']}")
if rolling.empty:
    st.info(t["rolling_empty"])
else:
    rolling_plot = rolling.melt(
        id_vars=["date"],
        value_vars=["br95", "effective_rank", "participation_ratio"],
        var_name="metric",
        value_name="value",
    )
    fig_roll = px.line(
        rolling_plot,
        x="date",
        y="value",
        color="metric",
        template=template,
    )
    fig_roll.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Effective dimensions",
    )
    st.plotly_chart(fig_roll, width="stretch")

    fig_haircut = px.area(
        rolling,
        x="date",
        y="breadth_haircut",
        template=template,
    )
    fig_haircut.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_tickformat=".0%",
        yaxis_title=t["haircut"],
    )
    st.plotly_chart(fig_haircut, width="stretch")

    if not regime_periods.empty:
        st.markdown(f"#### {t.get('regime_periods', 'Breadth Regime Periods')}")
        st.caption(
            t.get(
                "regime_periods_caption",
                "Low/Normal/High breadth blocks are adaptive quantiles of the rolling breadth haircut.",
            )
        )
        regime_label = {
            "Low": t.get("regime_low", "Low"),
            "Normal": t.get("regime_normal", "Normal"),
            "High": t.get("regime_high", "High"),
        }
        use_col = "research_use_zh" if lang == "ZH" else "research_use_en"
        display_periods = regime_periods[
            [
                "start",
                "end",
                "breadth_regime",
                "windows",
                "avg_br95",
                "avg_effective_rank",
                "avg_breadth_haircut",
                use_col,
            ]
        ].copy()
        display_periods["start"] = pd.to_datetime(display_periods["start"]).dt.date.astype(str)
        display_periods["end"] = pd.to_datetime(display_periods["end"]).dt.date.astype(str)
        display_periods["breadth_regime_label"] = display_periods["breadth_regime"].map(regime_label)
        display_periods = display_periods.rename(
            columns={
                "start": t.get("regime_start", "Start"),
                "end": t.get("regime_end", "End"),
                "breadth_regime_label": t.get("regime_col", "Regime"),
                "windows": t.get("windows_col", "Windows"),
                "avg_br95": t.get("avg_br95", "Avg BR95"),
                "avg_effective_rank": t.get("avg_effective_rank", "Avg Effective Rank"),
                "avg_breadth_haircut": t.get("avg_haircut", "Avg Haircut"),
                use_col: t.get("research_use", "Research Use"),
            }
        )
        display_periods = display_periods.drop(columns=["breadth_regime"])
        display_periods = display_periods[
            [
                t.get("regime_start", "Start"),
                t.get("regime_end", "End"),
                t.get("regime_col", "Regime"),
                t.get("windows_col", "Windows"),
                t.get("avg_br95", "Avg BR95"),
                t.get("avg_effective_rank", "Avg Effective Rank"),
                t.get("avg_haircut", "Avg Haircut"),
                t.get("research_use", "Research Use"),
            ]
        ]

        def _regime_period_style(row: pd.Series) -> list[str]:
            regime_value = str(row.get(t.get("regime_col", "Regime"), ""))
            if regime_value == t.get("regime_low", "Low"):
                color = "background-color: rgba(239, 68, 68, 0.16)"
            elif regime_value == t.get("regime_high", "High"):
                color = "background-color: rgba(34, 197, 94, 0.14)"
            else:
                color = "background-color: rgba(245, 158, 11, 0.13)"
            return [color] * len(row)

        st.dataframe(
            display_periods.style.apply(_regime_period_style, axis=1).format(
                {
                    t.get("avg_br95", "Avg BR95"): "{:.1f}",
                    t.get("avg_effective_rank", "Avg Effective Rank"): "{:.1f}",
                    t.get("avg_haircut", "Avg Haircut"): "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
