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
    load_daily_market_data,
    select_assets_for_breadth,
    summarize_breadth_regime_periods,
)
from oqp.risk.market_breadth import (
    MarketBreadthConfig,
    build_research_window_table,
    compute_market_structure,
)


DEFAULT_SOURCE = default_futures_cn_index_daily_file()
CACHE_VERSION = "market_breadth_v8_instrument_master_sectors"


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


@st.cache_data(show_spinner=False)
def _cached_market_structure(
    source_path: str,
    source_mtime: float,
    asset_class: str,
    volatility_lookback: int,
    max_assets: int,
    cache_version: str,
) -> dict:
    raw = load_daily_market_data(source_path, asset_class=asset_class)
    selected = select_assets_for_breadth(
        raw,
        RiskBreadthConfig(asset_class=asset_class, max_assets=max_assets),
    )
    sector_map: dict[str, str] = {}
    if asset_class == "FUTURES_CN":
        from oqp.data.instruments import InstrumentMaster

        sector_map = InstrumentMaster("FUTURES_CN").get_sector_map()
    elif "sector" in selected.columns:
        sector_rows = selected.dropna(subset=["ticker", "sector"])
        if not sector_rows.empty:
            sector_map = (
                sector_rows.groupby(sector_rows["ticker"].astype(str))["sector"]
                .agg(lambda values: str(values.mode().iat[0]) if not values.mode().empty else "Unknown")
                .to_dict()
            )
    return compute_market_structure(
        selected,
        asset_class=asset_class,
        sector_map=sector_map,
        config=MarketBreadthConfig(
            volatility_lookback=volatility_lookback,
            minimum_observations=max(10, min(60, volatility_lookback // 3)),
        ),
    )


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


def _latest_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _weight_source_label(source: str) -> str:
    labels = {
        "market_cap": t.get("weight_market_cap", "Market capitalization"),
        "open_interest_notional_proxy": t.get(
            "weight_open_interest", "Open-interest notional proxy"
        ),
        "traded_value": t.get("weight_turnover", "Traded value"),
        "price_x_volume_proxy": t.get("weight_volume", "Price x volume proxy"),
        "equal_weight_fallback": t.get("weight_equal", "Equal-weight fallback"),
        "unavailable": t.get("unavailable", "Unavailable"),
    }
    return labels.get(str(source), str(source))


def _render_overview(
    structure: dict,
    risk_metrics: dict,
    rolling_risk: pd.DataFrame,
) -> None:
    directional = structure["directional"]["daily"]
    concentration = structure["concentration"]["daily"]
    volatility_assets = structure["volatility"]["by_asset"]

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        t.get("directional_now", "Directional breadth"),
        _format_pct(_latest_value(directional, "directional_breadth")),
    )
    metric_cols[1].metric(
        t.get("concentration_now", "Effective assets"),
        _format_num(_latest_value(concentration, "effective_assets")),
    )
    metric_cols[2].metric(
        t.get("risk_now", "Independent risk dimensions"),
        _format_num(float(risk_metrics.get("effective_rank", float("nan")))),
    )
    median_vol = (
        float(volatility_assets["annualized_vol"].median())
        if not volatility_assets.empty
        else float("nan")
    )
    metric_cols[3].metric(t.get("volatility_now", "Median realized volatility"), _format_pct(median_vol))

    st.caption(t.get("overview_note", "Each card answers a different market-structure question; they are complementary, not interchangeable."))
    lens_rows = [
        {
            t.get("lens_col", "Lens"): t.get("tab_directional", "Directional Breadth"),
            t.get("question_col", "Question answered"): t.get(
                "directional_question", "How widely are assets advancing or declining?"
            ),
            t.get("method_col", "Method"): "(N+ - N-) / N",
        },
        {
            t.get("lens_col", "Lens"): t.get("tab_concentration", "Concentration Breadth"),
            t.get("question_col", "Question answered"): t.get(
                "concentration_question", "How many economically weighted assets are present?"
            ),
            t.get("method_col", "Method"): "1 / sum(w^2)",
        },
        {
            t.get("lens_col", "Lens"): t.get("tab_risk", "Risk Breadth"),
            t.get("question_col", "Question answered"): t.get(
                "risk_question", "How many statistically independent covariance dimensions exist?"
            ),
            t.get("method_col", "Method"): "PCA / eigen spectrum",
        },
        {
            t.get("lens_col", "Lens"): t.get("tab_volatility", "Volatility Map"),
            t.get("question_col", "Question answered"): t.get(
                "volatility_question", "Where is realized movement concentrated by asset and industry?"
            ),
            t.get("method_col", "Method"): t.get("realized_vol_method", "Rolling realized volatility"),
        },
    ]
    st.dataframe(pd.DataFrame(lens_rows), use_container_width=True, hide_index=True)

    percentile_parts: list[pd.DataFrame] = []
    series_specs = [
        (directional, "directional_breadth", t.get("tab_directional", "Directional Breadth")),
        (concentration, "effective_assets", t.get("tab_concentration", "Concentration Breadth")),
        (structure["volatility"]["market_timeline"], "median_vol", t.get("tab_volatility", "Volatility Map")),
        (rolling_risk, "effective_rank", t.get("tab_risk", "Risk Breadth")),
    ]
    for source, value_col, label in series_specs:
        if source.empty or value_col not in source.columns:
            continue
        piece = source[["date", value_col]].dropna().copy()
        piece["date"] = pd.to_datetime(piece["date"], errors="coerce")
        piece = piece.dropna(subset=["date"]).set_index("date").resample("ME").mean().reset_index()
        piece["percentile"] = piece[value_col].rank(pct=True)
        piece["lens"] = label
        percentile_parts.append(piece[["date", "lens", "percentile"]])
    if percentile_parts:
        history = pd.concat(percentile_parts, ignore_index=True)
        fig = px.line(
            history,
            x="date",
            y="percentile",
            color="lens",
            template=template,
            labels={
                "date": t.get("date_col", "Date"),
                "percentile": t.get("history_percentile", "Historical percentile"),
                "lens": t.get("lens_col", "Lens"),
            },
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def _render_directional(structure: dict) -> None:
    daily = structure["directional"]["daily"].copy()
    sectors = structure["directional"]["by_sector"].copy()
    if daily.empty:
        st.info(t.get("directional_empty", "Directional breadth is unavailable for this source."))
        return

    daily["breadth_21d"] = daily["directional_breadth"].rolling(21, min_periods=5).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["directional_breadth"],
            name=t.get("daily_breadth", "Daily breadth"),
            mode="lines",
            line=dict(width=1, color="#64B5F6"),
            customdata=daily[["advancers", "decliners", "active_assets"]],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                + t.get("directional_now", "Directional breadth")
                + ": %{y:.1%}<br>"
                + t.get("advancers", "Advancers")
                + ": %{customdata[0]:,.0f}<br>"
                + t.get("decliners", "Decliners")
                + ": %{customdata[1]:,.0f}<br>"
                + t.get("active_assets", "Active assets")
                + ": %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["breadth_21d"],
            name=t.get("rolling_21d", "21-day average"),
            mode="lines",
            line=dict(width=3, color="#FF7043"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9E9E9E")
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        template=template,
        yaxis_tickformat=".0%",
        yaxis_range=[-1, 1],
        yaxis_title=t.get("directional_axis", "Advance / decline breadth"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not sectors.empty:
        sectors["month"] = pd.to_datetime(sectors["date"]).dt.to_period("M").astype(str)
        monthly = sectors.groupby(["month", "sector"], as_index=False)["directional_breadth"].mean()
        months = sorted(monthly["month"].unique())[-36:]
        monthly = monthly[monthly["month"].isin(months)]
        heat = monthly.pivot(index="sector", columns="month", values="directional_breadth").fillna(0.0)
        heat = heat.loc[heat.abs().mean(axis=1).sort_values(ascending=False).index]
        fig_heat = px.imshow(
            heat,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
            template=template,
            labels={"x": t.get("month_col", "Month"), "y": t.get("industry_col", "Industry"), "color": t.get("directional_axis", "Breadth")},
        )
        fig_heat.update_layout(height=max(360, min(680, 90 + 30 * len(heat))), margin=dict(l=10, r=10, t=30, b=10))
        fig_heat.update_coloraxes(colorbar_tickformat=".0%")
        st.markdown(f"#### {t.get('industry_directional', 'Directional Breadth by Industry')}")
        st.plotly_chart(fig_heat, use_container_width=True)


def _render_volatility(structure: dict) -> None:
    by_asset = structure["volatility"]["by_asset"].copy()
    by_sector = structure["volatility"]["by_sector"].copy()
    timeline = structure["volatility"]["sector_timeline"].copy()
    if by_asset.empty:
        st.info(t.get("volatility_empty", "Volatility diagnostics are unavailable for this source/lookback."))
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        top_assets = by_asset.head(25).sort_values("annualized_vol")
        fig_asset = px.bar(
            top_assets,
            x="annualized_vol",
            y="ticker",
            orientation="h",
            color="vol_percentile",
            color_continuous_scale="Turbo",
            template=template,
            custom_data=["name", "sector", "observations", "last_date"],
            labels={"annualized_vol": t.get("annualized_vol", "Annualized volatility"), "ticker": t.get("asset_col", "Asset"), "vol_percentile": t.get("vol_percentile", "Vol percentile")},
        )
        fig_asset.update_traces(
            hovertemplate=(
                t.get("asset_col", "Asset") + ": %{y}<br>"
                + t.get("name_col", "Name") + ": %{customdata[0]}<br>"
                + t.get("industry_col", "Industry") + ": %{customdata[1]}<br>"
                + t.get("annualized_vol", "Annualized volatility") + ": %{x:.1%}<br>"
                + t.get("observations_col", "Observations") + ": %{customdata[2]:,.0f}<extra></extra>"
            )
        )
        fig_asset.update_layout(height=610, margin=dict(l=10, r=10, t=35, b=10), xaxis_tickformat=".0%", title=t.get("asset_volatility", "Highest-Volatility Assets"))
        st.plotly_chart(fig_asset, use_container_width=True)
    with chart_right:
        sector_plot = by_sector.sort_values("median_vol").tail(25)
        fig_sector_vol = px.bar(
            sector_plot,
            x="median_vol",
            y="sector",
            orientation="h",
            color="high_vol_share",
            color_continuous_scale="YlOrRd",
            template=template,
            hover_data={"assets": True, "mean_vol": ":.1%", "high_vol_share": ":.1%"},
            labels={"median_vol": t.get("median_vol", "Median volatility"), "sector": t.get("industry_col", "Industry"), "high_vol_share": t.get("high_vol_share", "High-vol share")},
        )
        fig_sector_vol.update_layout(height=610, margin=dict(l=10, r=10, t=35, b=10), xaxis_tickformat=".0%", title=t.get("industry_volatility", "Volatility by Industry"))
        st.plotly_chart(fig_sector_vol, use_container_width=True)

    if not timeline.empty and not by_sector.empty:
        top_sectors = by_sector.head(8)["sector"].tolist()
        plot_timeline = timeline[timeline["sector"].isin(top_sectors)]
        fig_timeline = px.line(
            plot_timeline,
            x="date",
            y="median_vol",
            color="sector",
            template=template,
            labels={"date": t.get("date_col", "Date"), "median_vol": t.get("median_vol", "Median volatility"), "sector": t.get("industry_col", "Industry")},
        )
        fig_timeline.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10), yaxis_tickformat=".0%")
        st.markdown(f"#### {t.get('industry_volatility_history', 'Industry Volatility Through Time')}")
        st.plotly_chart(fig_timeline, use_container_width=True)


def _render_concentration(structure: dict) -> None:
    concentration = structure["concentration"]
    daily = concentration["daily"].copy()
    sector_weights = concentration["sector_weights"].copy()
    latest_assets = concentration["latest_assets"].copy()
    source_label = _weight_source_label(concentration.get("weight_source", "unavailable"))
    st.caption(f"{t.get('weight_source', 'Weight source')}: {source_label}. {t.get('weight_source_note', 'The source is shown explicitly because HHI changes meaning with the selected weights.')}")
    if daily.empty:
        st.info(t.get("concentration_empty", "Concentration breadth is unavailable for this source."))
        return

    cols = st.columns(4)
    cols[0].metric(t.get("effective_assets", "Effective assets"), _format_num(_latest_value(daily, "effective_assets")))
    cols[1].metric(t.get("effective_industries", "Effective industries"), _format_num(_latest_value(daily, "effective_sectors")))
    cols[2].metric(t.get("top5_share", "Top 5 share"), _format_pct(_latest_value(daily, "top_5_share")))
    cols[3].metric(t.get("largest_weight", "Largest weight"), _format_pct(_latest_value(daily, "largest_weight")))

    plot = daily.melt(
        id_vars=["date"],
        value_vars=["assets", "effective_assets", "effective_sectors"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "assets": t.get("active_assets", "Active assets"),
        "effective_assets": t.get("effective_assets", "Effective assets"),
        "effective_sectors": t.get("effective_industries", "Effective industries"),
    }
    plot["metric"] = plot["metric"].map(metric_labels)
    fig = px.line(plot, x="date", y="value", color="metric", template=template, labels={"date": t.get("date_col", "Date"), "value": t.get("effective_count", "Effective count"), "metric": t.get("metric_col", "Metric")})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)

    detail_left, detail_right = st.columns([0.48, 0.52])
    with detail_left:
        if not sector_weights.empty:
            latest_date = sector_weights["date"].max()
            latest_sector = sector_weights[sector_weights["date"].eq(latest_date)].sort_values("weight")
            fig_sector = px.bar(latest_sector, x="weight", y="sector", orientation="h", template=template, hover_data={"assets": True}, labels={"weight": t.get("weight_col", "Weight"), "sector": t.get("industry_col", "Industry")})
            fig_sector.update_layout(height=max(360, min(620, 100 + 30 * len(latest_sector))), margin=dict(l=10, r=10, t=35, b=10), xaxis_tickformat=".0%", title=t.get("industry_weight", "Latest Industry Weight"))
            st.plotly_chart(fig_sector, use_container_width=True)
    with detail_right:
        if not latest_assets.empty:
            display = latest_assets.head(20).rename(columns={"ticker": t.get("asset_col", "Asset"), "name": t.get("name_col", "Name"), "sector": t.get("industry_col", "Industry"), "weight": t.get("weight_col", "Weight")})
            st.markdown(f"#### {t.get('largest_assets', 'Largest Weighted Assets')}")
            st.dataframe(display[[t.get("asset_col", "Asset"), t.get("name_col", "Name"), t.get("industry_col", "Industry"), t.get("weight_col", "Weight")]].style.format({t.get("weight_col", "Weight"): "{:.1%}"}), use_container_width=True, hide_index=True)


def _render_research_windows(structure: dict, rolling_risk: pd.DataFrame) -> None:
    windows = build_research_window_table(
        structure["directional"]["daily"],
        structure["concentration"]["daily"],
        structure["volatility"]["market_timeline"],
        rolling_risk,
    )
    if windows.empty:
        st.info(t.get("windows_empty", "Research-window guidance is unavailable for this source."))
        return
    st.caption(t.get("windows_note", "Use these monthly structural states to cover different market conditions across train, validation, and out-of-sample periods. They diagnose dates; they do not prescribe one universal window length."))
    display = windows.tail(36).copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m")
    state_map = {
        "Broad advance": t.get("state_broad_advance", "Broad advance"),
        "Broad decline": t.get("state_broad_decline", "Broad decline"),
        "Mixed": t.get("state_mixed", "Mixed"),
        "High concentration": t.get("state_high_concentration", "High concentration"),
        "Normal concentration": t.get("state_normal_concentration", "Normal concentration"),
        "Low concentration": t.get("state_low_concentration", "Low concentration"),
        "High volatility": t.get("state_high_volatility", "High volatility"),
        "Normal volatility": t.get("state_normal_volatility", "Normal volatility"),
        "Low volatility": t.get("state_low_volatility", "Low volatility"),
        "Low": t.get("regime_low", "Low / Compressed"),
        "Normal": t.get("regime_normal", "Normal"),
        "High": t.get("regime_high", "High / Expanded"),
        "Unavailable": t.get("unavailable", "Unavailable"),
    }
    for column in ["direction_state", "concentration_state", "volatility_state", "risk_state"]:
        display[column] = display[column].map(state_map).fillna(display[column])
    use_col = "research_use_zh" if lang == "ZH" else "research_use_en"
    display = display[["date", "direction_state", "concentration_state", "volatility_state", "risk_state", use_col]].rename(
        columns={
            "date": t.get("month_col", "Month"),
            "direction_state": t.get("tab_directional", "Directional Breadth"),
            "concentration_state": t.get("tab_concentration", "Concentration Breadth"),
            "volatility_state": t.get("tab_volatility", "Volatility Map"),
            "risk_state": t.get("tab_risk", "Risk Breadth"),
            use_col: t.get("research_use", "Research Use"),
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


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

control_cols = st.columns([0.34, 0.13, 0.13, 0.13, 0.13, 0.14])
with control_cols[0]:
    labels = _source_labels(files)
    selected_label = st.selectbox(t["source"], labels, index=0)
    selected_path = files[labels.index(selected_label)]
with control_cols[1]:
    volatility_lookback = st.select_slider(
        t.get("volatility_lookback", "Volatility lookback"),
        options=[21, 63, 126, 252, 504],
        value=252,
    )
with control_cols[2]:
    variance_threshold = st.slider(t["variance"], 0.80, 0.99, 0.95, 0.01)
with control_cols[3]:
    rolling_window = st.select_slider(
        t["window"], options=[252, 504, 756, 1008], value=504
    )
with control_cols[4]:
    default_max_assets = 75 if asset_class == "FUTURES_CN" else 300
    max_assets = st.select_slider(
        t.get("max_assets", "Max assets"),
        options=[50, 75, 150, 300, 500, 1000],
        value=default_max_assets,
    )
with control_cols[5]:
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
    structure = _cached_market_structure(
        str(selected_path),
        selected_path.stat().st_mtime,
        asset_class,
        int(volatility_lookback),
        int(max_assets),
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

(
    overview_tab,
    directional_tab,
    volatility_tab,
    concentration_tab,
    risk_tab,
    windows_tab,
) = st.tabs(
    [
        t.get("tab_overview", "Overview"),
        t.get("tab_directional", "Directional Breadth"),
        t.get("tab_volatility", "Volatility Map"),
        t.get("tab_concentration", "Concentration Breadth"),
        t.get("tab_risk", "Risk Breadth"),
        t.get("tab_windows", "Research Windows"),
    ]
)

with overview_tab:
    _render_overview(structure, metrics, rolling)

with directional_tab:
    _render_directional(structure)

with volatility_tab:
    _render_volatility(structure)

with concentration_tab:
    _render_concentration(structure)

with windows_tab:
    _render_research_windows(structure, rolling_regimes)

with risk_tab:
    # RISK_TAB_CONTENT_START
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
    st.plotly_chart(fig_spectrum, use_container_width=True)

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
            use_container_width=True,
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
        st.plotly_chart(fig_stability, use_container_width=True)

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
            use_container_width=True,
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
        st.plotly_chart(fig_sector, use_container_width=True)

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
        use_container_width=True,
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
        st.plotly_chart(fig_roll, use_container_width=True)

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
        st.plotly_chart(fig_haircut, use_container_width=True)

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
                use_container_width=True,
                hide_index=True,
            )
