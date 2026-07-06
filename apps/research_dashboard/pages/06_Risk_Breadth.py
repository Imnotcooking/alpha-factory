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
from oqp.risk.factor_breadth import (
    RiskBreadthConfig,
    compute_risk_factor_breadth,
)


DEFAULT_SOURCE = (
    Path(ALPHA_RUNTIME_DATA_ROOT)
    / "market_data"
    / "daily"
    / "全市场_1d_index_20180101_20260602.parquet"
)
CACHE_VERSION = "risk_breadth_v2_component_labels"


from oqp.ui.translations import research_page_legacy_catalog


PAGE_TEXT = research_page_legacy_catalog("risk_factor_breadth_lab")


st.set_page_config(page_title="Risk Breadth", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang if st.session_state.lang in PAGE_TEXT else "EN"
t = PAGE_TEXT[lang]
template = get_plotly_template(st.session_state.theme_mode)


def _list_daily_files() -> list[Path]:
    runtime_daily = Path(ALPHA_RUNTIME_DATA_ROOT) / "market_data" / "daily"
    runtime_feature_store = Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store"
    patterns = ["*1d*index*.parquet", "*1d*main*.parquet", "ML_Feature_Matrix.parquet"]
    seen: dict[Path, None] = {}
    for root in (runtime_daily, runtime_feature_store):
        for pattern in patterns:
            for path in sorted(root.glob(pattern) if root.exists() else []):
                if path.is_file():
                    seen[path.resolve()] = None
    root_matrix = (
        Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store" / "ML_Feature_Matrix.parquet"
    )
    if root_matrix.exists():
        seen[root_matrix.resolve()] = None
    files = list(seen.keys())
    if DEFAULT_SOURCE.exists():
        default_resolved = DEFAULT_SOURCE.resolve()
        files = [
            default_resolved,
            *[path for path in files if path != default_resolved],
        ]
    return files


@st.cache_data(show_spinner=False)
def _cached_risk_breadth(
    source_path: str,
    source_mtime: float,
    variance_threshold: float,
    rolling_window: int,
    cache_version: str,
) -> dict:
    config = RiskBreadthConfig(
        variance_threshold=variance_threshold,
        rolling_window=rolling_window,
        rolling_step=21,
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
                "top3_sector_share": float(
                    sector_slice.head(3)["abs_loading_share"].sum()
                ),
                "interpretation_en": interpretation_en,
                "interpretation_zh": interpretation_zh,
            }
        )

    return pd.DataFrame(rows)


st.title(t["title"])
st.caption(t["subtitle"])

files = _list_daily_files()
if not files:
    st.error(f"{t['error']}: no candidate parquet files found.")
    st.stop()

control_cols = st.columns([0.58, 0.2, 0.22])
with control_cols[0]:
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
    selected_label = st.selectbox(t["source"], labels, index=0)
    selected_path = files[labels.index(selected_label)]
with control_cols[1]:
    variance_threshold = st.slider(t["variance"], 0.80, 0.99, 0.95, 0.01)
with control_cols[2]:
    rolling_window = st.select_slider(
        t["window"], options=[252, 504, 756, 1008], value=504
    )

try:
    result = _cached_risk_breadth(
        str(selected_path),
        selected_path.stat().st_mtime,
        float(variance_threshold),
        int(rolling_window),
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
    f"observations={metrics['observations']:,} | source_rows={result['source_rows']:,}"
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
    interp_name = "interpretation_zh" if lang == "ZH" else "interpretation_en"
    display_labels = component_labels.head(10)[
        [
            "component",
            label_name,
            "top_sectors",
            "top_assets",
            "top3_sector_share",
            interp_name,
        ]
    ].copy()
    display_labels = display_labels.rename(
        columns={
            label_name: "label",
            interp_name: "interpretation",
        }
    )
    st.dataframe(
        display_labels.style.format({"top3_sector_share": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )

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
