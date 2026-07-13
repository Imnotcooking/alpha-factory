from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib.util

import pandas as pd
import streamlit as st


UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(UI_DIR)
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ui_state import apply_global_style, init_global_ui_state, render_global_controls_in_sidebar
from arbitrage_lab import charts
from arbitrage_lab.components import (
    candidate_display_table,
    fmt_int,
    fmt_num,
    render_explainer,
    render_interpretation,
)
from arbitrage_lab.text import PAGE_TEXT
from oqp.research.state_space import (
    ARBITRAGE_CALENDAR,
    ARBITRAGE_CROSS_PRODUCT,
    ARBITRAGE_STATISTICAL,
    DataAuditConfig,
    OpportunityScanConfig,
    compute_data_audit,
    construct_spread_for_candidate,
    normalize_daily_market_frame,
    run_opportunity_scan,
)
from oqp.research.state_space import (
    RelationshipLabConfig,
    list_daily_price_files,
    run_relationship_dkf,
)
from oqp.research.state_space import (
    SPREAD_CONTRACT_VALUE,
    SPREAD_LINEAR_PRICE,
    SPREAD_PRICE_RATIO,
    SPREAD_RETURN_RESIDUAL,
    latest_spread_summary,
    simple_spread_backtest,
)


CACHE_VERSION = "adaptive_arbitrage_lab_v1"
SPREAD_METHOD_LABELS = {
    "Return residual": SPREAD_RETURN_RESIDUAL,
    "Price ratio": SPREAD_PRICE_RATIO,
    "Linear price spread": SPREAD_LINEAR_PRICE,
    "Contract-value spread": SPREAD_CONTRACT_VALUE,
}


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("_ui_v2_config", _CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError(f"Could not load UI config from {_CONFIG_PATH}")
_UI_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_UI_CONFIG)
BASE_DIR = _UI_CONFIG.BASE_DIR
get_plotly_template = _UI_CONFIG.get_plotly_template


if os.environ.get("OQP_EMBEDDED_STREAMLIT_PAGE") != "1":
    st.set_page_config(page_title="Arbitrage Lab", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang if st.session_state.lang in PAGE_TEXT else "EN"
t = PAGE_TEXT[lang]
template = get_plotly_template(st.session_state.theme_mode)


@st.cache_data(show_spinner=False)
def _load_market(path: str, mtime: float, cache_version: str) -> pd.DataFrame:
    return normalize_daily_market_frame(pd.read_parquet(path))


@st.cache_data(show_spinner=False)
def _scan_market(
    path: str,
    mtime: float,
    min_observations: int,
    lookback: int,
    zscore_window: int,
    max_assets: int,
    min_abs_correlation: float,
    cache_version: str,
) -> dict:
    market = _load_market(path, mtime, cache_version)
    cfg = OpportunityScanConfig(
        min_observations=int(min_observations),
        lookback=int(lookback),
        zscore_window=int(zscore_window),
        max_assets=int(max_assets),
        min_abs_correlation=float(min_abs_correlation),
    )
    return run_opportunity_scan(market, cfg)


@st.cache_data(show_spinner=False)
def _run_dkf_cached(
    path: str,
    mtime: float,
    y_ticker: str,
    x_ticker: str,
    process_noise: float,
    obs_noise: float,
    initial_cov: float,
    cache_version: str,
) -> dict:
    market = _load_market(path, mtime, cache_version)
    cfg = RelationshipLabConfig(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        y_label=y_ticker,
        x_label=x_ticker,
        process_noise=process_noise,
        observation_noise=obs_noise,
        initial_state_covariance=initial_cov,
    )
    return run_relationship_dkf(market, cfg)


def _select_candidate(candidates: pd.DataFrame, key: str) -> pd.Series | None:
    if candidates.empty:
        return None
    options = candidates["candidate_id"].astype(str).tolist()
    selected = st.selectbox(t["selected"], options, index=0, key=key)
    return candidates[candidates["candidate_id"].astype(str) == selected].iloc[0]


def _spread_from_candidate_controls(
    market: pd.DataFrame,
    candidate: pd.Series,
    key_prefix: str,
    copy: dict,
) -> pd.DataFrame:
    cols = st.columns(4)
    method_labels = copy.get("spread_method_labels", {})
    hedge_labels = copy.get("hedge_method_labels", {})
    method_label = cols[0].selectbox(
        copy["spread_method"],
        list(SPREAD_METHOD_LABELS.keys()),
        index=0,
        key=f"{key_prefix}_method",
        format_func=lambda value: method_labels.get(value, value),
    )
    hedge_method = cols[1].selectbox(
        copy["hedge_method"],
        ["ols", "fixed"],
        index=0,
        key=f"{key_prefix}_hedge",
        format_func=lambda value: hedge_labels.get(value, value),
    )
    hedge_lookback = cols[2].number_input(
        copy["hedge_lookback"],
        min_value=60,
        max_value=2000,
        value=504,
        step=21,
        key=f"{key_prefix}_lookback",
    )
    z_window = cols[3].number_input(
        copy["spread_z_window"],
        min_value=30,
        max_value=1000,
        value=126,
        step=21,
        key=f"{key_prefix}_z",
    )
    return construct_spread_for_candidate(
        market,
        candidate,
        method=SPREAD_METHOD_LABELS[method_label],
        hedge_method=hedge_method,
        hedge_lookback=int(hedge_lookback),
        zscore_window=int(z_window),
    )


def _candidate_from_tickers(metadata: pd.DataFrame, y_ticker: str, x_ticker: str) -> pd.Series:
    meta = metadata.set_index("ticker")
    y = meta.loc[y_ticker] if y_ticker in meta.index else pd.Series(dtype=object)
    x = meta.loc[x_ticker] if x_ticker in meta.index else pd.Series(dtype=object)
    return pd.Series(
        {
            "candidate_id": f"{y_ticker} ~ {x_ticker}",
            "y_ticker": y_ticker,
            "x_ticker": x_ticker,
            "y_multiplier": float(y.get("multiplier", 1.0) or 1.0),
            "x_multiplier": float(x.get("multiplier", 1.0) or 1.0),
        }
    )


st.title(t["title"])
st.caption(t["subtitle"])
with st.expander(t["manual_title"], expanded=False):
    st.markdown(t["manual"])

files = list_daily_price_files(BASE_DIR)
if not files:
    st.warning(t["empty"])
    st.stop()

labels = [path.name for path in files]
source_label = st.selectbox(t["source"], labels, index=0)
source_path = files[labels.index(source_label)]
source_mtime = source_path.stat().st_mtime

control_cols = st.columns([1, 1, 1, 1, 1])
min_obs = control_cols[0].number_input(t["min_obs"], min_value=60, max_value=2_000, value=252, step=21)
lookback = control_cols[1].number_input(t["lookback"], min_value=120, max_value=2_500, value=504, step=21)
z_window = control_cols[2].number_input(t["z_window"], min_value=30, max_value=1_000, value=126, step=21)
max_assets = control_cols[3].number_input(t["max_assets"], min_value=5, max_value=100, value=40, step=5)
min_corr = control_cols[4].slider(t["min_corr"], 0.0, 0.95, 0.15, 0.05)
render_explainer(t["controls_help_title"], t["controls_help"])

try:
    market = _load_market(str(source_path), source_mtime, CACHE_VERSION)
    scan = _scan_market(
        str(source_path),
        source_mtime,
        int(min_obs),
        int(lookback),
        int(z_window),
        int(max_assets),
        float(min_corr),
        CACHE_VERSION,
    )
except Exception as exc:
    st.error(f"{t['run_error']}: {exc}")
    st.stop()

candidates = scan["candidates"]
metadata = scan["metadata"]
if candidates.empty:
    st.warning(t["no_candidates"])
    render_explainer(t["metric_help_title"], t["metric_help"], expanded=True)
    st.stop()

tabs = st.tabs(t["tabs"])

with tabs[0]:
    st.caption(t["run_note"])
    render_explainer(t["radar_help_title"], t["radar_help"])
    filtered = candidates.copy().reset_index(drop=True)
    st.caption(t["full_universe_note"].format(count=len(filtered)))
    if filtered.empty:
        st.warning(t["no_candidates"])
    else:
        top = filtered.iloc[0]
        metric_cols = st.columns(4)
        metric_cols[0].metric(t["score"], fmt_num(top["opportunity_score"], 1))
        metric_cols[1].metric(t["latest_z"], fmt_num(top["latest_z"], 2))
        metric_cols[2].metric(t["half_life"], fmt_num(top["half_life"], 1))
        metric_cols[3].metric(t["cost"], f"{fmt_num(top['round_turn_cost_bps'], 1)} bps")
        render_interpretation(str(top["interpretation"]), lang=lang)

        left, right = st.columns([1.2, 0.8])
        with left:
            st.plotly_chart(charts.opportunity_scatter(filtered, template), width="stretch")
        with right:
            st.plotly_chart(charts.top_dislocation_bar(filtered, template), width="stretch")
        st.plotly_chart(charts.sector_heatmap(filtered, template), width="stretch")
        render_explainer(t["metric_help_title"], t["metric_help"])
        st.dataframe(candidate_display_table(filtered, limit=80, lang=lang), width="stretch", hide_index=True)

with tabs[1]:
    render_explainer(t["workspace_help_title"], t["workspace_help"], expanded=True)
    workspace_options = ["scanner", "manual"]
    workspace_option_labels = t["workspace_mode_options"]
    workspace_mode = st.radio(
        t["workspace_mode"],
        workspace_options,
        format_func=lambda value: workspace_option_labels.get(value, value),
        horizontal=True,
        key="workspace_mode",
    )
    if workspace_mode == "manual":
        tickers = metadata["ticker"].dropna().astype(str).tolist()
        default_y = str(candidates.iloc[0]["y_ticker"])
        default_x = str(candidates.iloc[0]["x_ticker"])
        pair_cols = st.columns(2)
        y_ticker = pair_cols[0].selectbox(t["y_asset"], tickers, index=tickers.index(default_y) if default_y in tickers else 0, key="workspace_manual_y")
        x_options = [ticker for ticker in tickers if ticker != y_ticker]
        x_ticker = pair_cols[1].selectbox(t["x_asset"], x_options, index=x_options.index(default_x) if default_x in x_options else 0, key="workspace_manual_x")
        selected = _candidate_from_tickers(metadata, y_ticker, x_ticker)
    else:
        selected = _select_candidate(candidates, "workspace_candidate")

    if selected is not None:
        interpretation = str(selected.get("interpretation", "") or "")
        if interpretation and interpretation != "nan":
            render_interpretation(interpretation, lang=lang)

        st.markdown(f"### {t['dynamic_relationship']}")
        render_explainer(t["drill_help_title"], t["drill_help"], expanded=False)
        noise_cols = st.columns(3)
        process_noise = noise_cols[0].number_input(t["process_noise"], min_value=1e-8, max_value=1e-1, value=1e-4, format="%.8f")
        obs_noise = noise_cols[1].number_input(t["observation_noise"], min_value=1e-8, max_value=1e-1, value=1e-4, format="%.8f")
        initial_cov = noise_cols[2].number_input(t["initial_uncertainty"], min_value=0.01, max_value=100.0, value=10.0, format="%.2f")
        try:
            result = _run_dkf_cached(
                str(source_path),
                source_mtime,
                str(selected["y_ticker"]),
                str(selected["x_ticker"]),
                float(process_noise),
                float(obs_noise),
                float(initial_cov),
                CACHE_VERSION,
            )
            pair = result["pair"]
            summary = result["summary"]
            metric_cols = st.columns(4)
            metric_cols[0].metric(t["rows"], fmt_int(summary.get("rows")))
            metric_cols[1].metric(t["beta"], fmt_num(summary.get("beta_latest"), 3))
            metric_cols[2].metric(t["beta_change"], fmt_num(summary.get("beta_change"), 3))
            metric_cols[3].metric(t["extreme_rate"], f"{fmt_num(100 * summary.get('extreme_z_rate', 0.0), 1)}%")
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.dkf_beta(pair, template), width="stretch")
            c2.plotly_chart(charts.dkf_residual(pair, template), width="stretch")
            c3, c4 = st.columns(2)
            c3.plotly_chart(charts.dkf_uncertainty(pair, template), width="stretch")
            c4.plotly_chart(charts.residual_distribution(pair, template), width="stretch")
            with st.expander(t["recent_relationship_rows"], expanded=False):
                cols = ["date", "y_return", "x_return", "dynamic_alpha", "dynamic_beta", "residual_z", "state_uncertainty", "beta_l1_change"]
                st.dataframe(pair[[col for col in cols if col in pair.columns]].tail(250), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"{t['run_error']}: {exc}")

        st.markdown(f"### {t['spread_construction']}")
        render_explainer(t["builder_help_title"], t["builder_help"], expanded=False)
        try:
            spread = _spread_from_candidate_controls(market, selected, "workspace_builder", t)
            spread_summary = latest_spread_summary(spread)
            metric_cols = st.columns(4)
            metric_cols[0].metric(t["latest_z"], fmt_num(spread_summary.get("latest_z"), 2))
            metric_cols[1].metric(t["beta"], fmt_num(spread_summary.get("beta"), 3))
            metric_cols[2].metric(t["half_life"], fmt_num(spread_summary.get("half_life"), 1))
            metric_cols[3].metric(t["rows"], fmt_int(spread_summary.get("rows")))
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.price_legs(spread, template), width="stretch")
            c2.plotly_chart(charts.spread_zscore(spread, template), width="stretch")
            st.plotly_chart(charts.spread_level(spread, template), width="stretch")

            with st.expander(t["backtest_preview"], expanded=False):
                st.markdown(f"**{t['backtest_help_title']}**")
                st.markdown(t["backtest_help"])
                cols = st.columns(4)
                entry_z = cols[0].number_input(t["entry_z"], min_value=0.5, max_value=5.0, value=2.0, step=0.25)
                exit_z = cols[1].number_input(t["exit_z"], min_value=0.0, max_value=2.0, value=0.5, step=0.25)
                stop_z = cols[2].number_input(t["stop_z"], min_value=1.0, max_value=8.0, value=3.5, step=0.25)
                raw_cost = selected.get("round_turn_cost_bps", 0.0)
                cost_default = float(raw_cost) if pd.notna(raw_cost) else 0.0
                cost_bps = cols[3].number_input(t["cost_bps"], min_value=0.0, max_value=100.0, value=cost_default, step=0.5)
                preview = simple_spread_backtest(spread, entry_z=float(entry_z), exit_z=float(exit_z), stop_z=float(stop_z), cost_bps=float(cost_bps))
                curve = preview["curve"]
                preview_summary = preview["summary"]
                preview_cols = st.columns(4)
                preview_cols[0].metric(t["trades"], fmt_int(preview_summary.get("trades")))
                preview_cols[1].metric(t["win_rate"], "N/A" if pd.isna(preview_summary.get("win_rate")) else f"{100 * preview_summary.get('win_rate'):.1f}%")
                preview_cols[2].metric(t["net_pnl"], fmt_num(preview_summary.get("net_pnl"), 4))
                preview_cols[3].metric(t["max_dd"], fmt_num(preview_summary.get("max_drawdown"), 4))
                st.plotly_chart(charts.backtest_equity(curve, template), width="stretch")
                st.plotly_chart(charts.backtest_drawdown(curve, template), width="stretch")
                st.markdown(f"##### {t['trades']}")
                st.dataframe(preview["trades"], width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"{t['run_error']}: {exc}")

with tabs[2]:
    render_explainer(t["maps_help_title"], t["maps_help"], expanded=True)
    map_options = ["calendar", "cross_product", "statistical"]
    map_option_labels = t["map_view_options"]
    map_view = st.radio(
        t["map_view"],
        map_options,
        format_func=lambda value: map_option_labels.get(value, value),
        horizontal=True,
        key="map_view",
    )
    if map_view == "calendar":
        render_explainer(t["calendar_help_title"], t["calendar_help"], expanded=False)
        calendar = candidates[candidates["arbitrage_type"] == ARBITRAGE_CALENDAR].copy()
        if calendar.empty:
            st.info(t["calendar_empty"])
            base_counts = metadata.groupby("base_symbol", as_index=False).agg(assets=("ticker", "nunique"))
            st.dataframe(base_counts.sort_values("assets", ascending=False).head(30), width="stretch", hide_index=True)
        else:
            st.plotly_chart(charts.top_dislocation_bar(calendar, template), width="stretch")
            st.dataframe(candidate_display_table(calendar, limit=80, lang=lang), width="stretch", hide_index=True)
    elif map_view == "cross_product":
        render_explainer(t["cross_help_title"], t["cross_help"], expanded=False)
        cross = candidates[candidates["arbitrage_type"] == ARBITRAGE_CROSS_PRODUCT].copy()
        if cross.empty:
            st.info(t["cross_empty"])
        else:
            st.plotly_chart(charts.sector_heatmap(cross, template), width="stretch")
            st.dataframe(candidate_display_table(cross, limit=100, lang=lang), width="stretch", hide_index=True)
    else:
        render_explainer(t["stat_help_title"], t["stat_help"], expanded=False)
        stat = candidates[candidates["arbitrage_type"] == ARBITRAGE_STATISTICAL].copy()
        if stat.empty:
            st.info(t["stat_empty"])
        else:
            st.plotly_chart(charts.opportunity_scatter(stat, template), width="stretch")
            st.dataframe(candidate_display_table(stat, limit=100, lang=lang), width="stretch", hide_index=True)

with tabs[3]:
    render_explainer(t["audit_help_title"], t["audit_help"], expanded=True)
    audit = compute_data_audit(market, DataAuditConfig(min_observations=int(min_obs)))
    summary = audit["summary"]
    metric_cols = st.columns(4)
    metric_cols[0].metric(t["rows"], fmt_int(summary.get("rows")))
    metric_cols[1].metric(t["assets"], fmt_int(summary.get("assets")))
    metric_cols[2].metric(t["eligible_assets"], fmt_int(summary.get("eligible_assets")))
    metric_cols[3].metric(
        t["contract_level"],
        t["yes"] if summary.get("has_contract_level_duplicates") else t["no"],
    )
    st.caption(t["date_range"].format(file=source_path.name, start=summary.get("date_min"), end=summary.get("date_max")))
    st.markdown(f"#### {t['schema']}")
    st.dataframe(audit["schema"], width="stretch", hide_index=True)
    st.markdown(f"#### {t['asset_coverage']}")
    st.dataframe(audit["assets"].sort_values(["eligible", "observations"], ascending=[False, False]), width="stretch", hide_index=True)
