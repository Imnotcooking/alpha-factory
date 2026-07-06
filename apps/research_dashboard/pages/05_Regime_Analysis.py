from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from plotly.subplots import make_subplots

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

_CONFIG_PATH = Path(UI_DIR) / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("_ui_v2_config", _CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError(f"Unable to load UI config from {_CONFIG_PATH}")
_UI_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_UI_CONFIG)

BASE_DIR = _UI_CONFIG.BASE_DIR
ALPHA_RUNTIME_DATA_ROOT = _UI_CONFIG.ALPHA_RUNTIME_DATA_ROOT
ALPHA_RUNTIME_ARTIFACT_ROOT = _UI_CONFIG.ALPHA_RUNTIME_ARTIFACT_ROOT
TEXT = _UI_CONFIG.TEXT
get_plotly_template = _UI_CONFIG.get_plotly_template
from asset_taxonomy import (
    DEFAULT_ASSET_CLASS,
    attach_asset_class,
    load_asset_taxonomy,
    taxonomy_options,
    taxonomy_row,
    ticker_asset_class_map,
)
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)

ASSET_TAXONOMY = load_asset_taxonomy(BASE_DIR)

ROOT_DIR = os.path.dirname(UI_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from oqp.research.latent import (
    codebook_health_summary,
    compute_gmm_overlap,
    merge_gmm_probabilities,
)
from oqp.research.latent import load_saved_latents


from oqp.ui.translations import research_page_legacy_catalog


LAB_TEXT = research_page_legacy_catalog("regime_characterisation_lab")


STATE_KEYS = ["Quiet", "Chop", "Panic"]
STATE_COLORS = {"Quiet": "#2ca02c", "Chop": "#ff7f0e", "Panic": "#d62728"}
QUAD_COLORS = {
    "Trend Highway": "#2ecc71",
    "Toxic Whipsaw": "#e74c3c",
    "Slow Grind": "#27ae60",
    "Sleepy Chop": "#95a5a6",
}


if os.environ.get("OQP_EMBEDDED_STREAMLIT_PAGE") != "1":
    st.set_page_config(page_title="Regime Analysis", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang if st.session_state.lang in LAB_TEXT else "EN"
t = TEXT[lang] if lang in TEXT else TEXT["EN"]
lt = LAB_TEXT[lang]
tpl = get_plotly_template(st.session_state.theme_mode)


def _file_mtime(path: str | Path) -> float:
    path = Path(path)
    return path.stat().st_mtime if path.exists() else 0.0


def _rolling_z(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


@st.cache_data(show_spinner=False)
def load_data(
    matrix_path: str, regimes_path: str, matrix_mtime: float, regimes_mtime: float
):
    feature_cols = [
        "date",
        "ticker",
        "close",
        "gk_vol_z",
        "amihud_z",
        "ker_20d",
        "f_macro_hurst",
        "target_1d_rank",
        "target_4d_rank",
    ]
    optional_cols = ["asset_class", "market_vertical", "dataset_id", "universe_id"]
    try:
        features_df = pd.read_parquet(matrix_path)
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.nan
        kept_cols = [
            *feature_cols,
            *[col for col in optional_cols if col in features_df.columns],
        ]
        features_df = features_df[kept_cols].copy()
        features_df["date"] = pd.to_datetime(features_df["date"])
        features_df["ticker"] = features_df["ticker"].astype(str)
    except Exception:
        features_df = pd.DataFrame(columns=feature_cols)

    try:
        regimes = pd.read_parquet(regimes_path)
        regimes["date"] = pd.to_datetime(regimes["date"])
        regimes["ticker"] = regimes["ticker"].astype(str)
    except Exception:
        regimes = pd.DataFrame(
            columns=["date", "ticker", "p_state_0", "p_state_1", "p_state_2"]
        )

    return regimes, features_df


@st.cache_data(show_spinner=False)
def load_vq_artifacts(artifact_dir: str, artifact_mtime: float):
    return load_saved_latents(artifact_dir=artifact_dir)


def prepare_asset_data(
    features_df: pd.DataFrame, regimes: pd.DataFrame, ticker: str
) -> tuple[pd.DataFrame, dict]:
    asset_features = features_df[features_df["ticker"] == ticker].copy()
    asset_regimes = regimes[regimes["ticker"] == ticker].copy()
    merged = asset_features.merge(asset_regimes, on=["date", "ticker"], how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        return merged, {}

    for idx in range(3):
        prob_col = f"p_state_{idx}"
        smooth_col = f"p_{idx}_smooth"
        merged[prob_col] = pd.to_numeric(
            merged.get(prob_col, 0.0), errors="coerce"
        ).fillna(0.0)
        merged[smooth_col] = merged[prob_col].ewm(span=7, adjust=False).mean()

    merged["gk_z"] = pd.to_numeric(merged["gk_vol_z"], errors="coerce").fillna(0.0)
    merged["amihud_z"] = pd.to_numeric(merged["amihud_z"], errors="coerce").fillna(0.0)
    merged["ker_z"] = _rolling_z(pd.to_numeric(merged["ker_20d"], errors="coerce"))
    merged["hurst"] = pd.to_numeric(
        merged.get("f_macro_hurst", 0.5), errors="coerce"
    ).fillna(0.5)

    merged["raw_dom"] = (
        merged[[f"p_state_{idx}" for idx in range(3)]].to_numpy().argmax(axis=1)
    )
    merged["stress_metric"] = merged["amihud_z"] + merged["gk_z"]
    stress_by_state = merged.groupby("raw_dom")["stress_metric"].mean().sort_values()
    ordered = [int(idx) for idx in stress_by_state.index]
    ordered.extend([idx for idx in range(3) if idx not in ordered])
    state_map = {"Quiet": ordered[0], "Chop": ordered[1], "Panic": ordered[2]}

    for state_name, raw_idx in state_map.items():
        merged[f"prob_{state_name}"] = merged[f"p_{raw_idx}_smooth"]

    merged["Aligned_State_Idx"] = (
        merged[[f"prob_{state}" for state in STATE_KEYS]].to_numpy().argmax(axis=1)
    )
    merged["State"] = merged["Aligned_State_Idx"].map(
        {0: "Quiet", 1: "Chop", 2: "Panic"}
    )
    merged["dynamic_threshold"] = merged[["prob_Quiet", "prob_Chop"]].max(axis=1)
    merged["is_panic"] = (merged["prob_Panic"] > merged["dynamic_threshold"]) & (
        merged["hurst"] < 0.5
    )
    merged["daily_return"] = merged["close"].pct_change()

    return merged, state_map


def merge_asset_vq(
    merged_data: pd.DataFrame, latent_result: dict, ticker: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not latent_result or "latent" not in latent_result:
        return pd.DataFrame(), pd.DataFrame(), {}

    latent_df = latent_result.get("latent", pd.DataFrame()).copy()
    usage_df = latent_result.get("usage", pd.DataFrame()).copy()
    if latent_df.empty or "vq_code" not in latent_df.columns:
        return pd.DataFrame(), usage_df, latent_result.get("health", {})

    latent_df["date"] = pd.to_datetime(latent_df["date"])
    latent_df["ticker"] = latent_df["ticker"].astype(str)
    asset_latent = latent_df[latent_df["ticker"] == ticker].copy()
    if asset_latent.empty:
        return pd.DataFrame(), usage_df, latent_result.get("health", {})

    vq_cols = [
        "date",
        "ticker",
        "vq_code",
        "vq_distance",
        *[col for col in asset_latent.columns if col.startswith("z_vq_")],
    ]
    vq_cols = [col for col in vq_cols if col in asset_latent.columns]
    asset_latent = asset_latent[vq_cols].copy()
    out = merged_data.merge(asset_latent, on=["date", "ticker"], how="left")
    health = latent_result.get("health") or codebook_health_summary(usage_df)
    return out, usage_df, health


def compute_meta_labels(
    merged_data: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = merged_data.sort_values("date").copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    future_prices = pd.concat(
        [close.shift(-step) for step in range(1, horizon + 1)], axis=1
    )
    future_returns = future_prices.divide(close, axis=0) - 1.0

    df["future_return"] = close.shift(-horizon) / close - 1.0
    df["future_downside_move"] = future_returns.min(axis=1)
    df["future_abs_move"] = future_returns.abs().max(axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["future_return", "future_downside_move", "future_abs_move"]
    )
    if df.empty:
        return df, {}, pd.DataFrame()

    downside_threshold = df["future_downside_move"].quantile(0.20)
    turbulence_threshold = df["future_abs_move"].quantile(0.80)
    df["downside_stress"] = df["future_downside_move"] <= downside_threshold
    df["turbulence_stress"] = df["future_abs_move"] >= turbulence_threshold
    df["stress_event"] = df["downside_stress"] | df["turbulence_stress"]
    df["gmm_panic_call"] = df["State"].eq("Panic")

    panic_calls = df["gmm_panic_call"].sum()
    stress_events = df["stress_event"].sum()
    true_positive = (df["gmm_panic_call"] & df["stress_event"]).sum()
    metrics = {
        "stress_base_rate": float(df["stress_event"].mean()),
        "panic_precision": (
            float(true_positive / panic_calls) if panic_calls else np.nan
        ),
        "stress_recall": (
            float(true_positive / stress_events) if stress_events else np.nan
        ),
        "false_alarm_rate": (
            float((df["gmm_panic_call"] & ~df["stress_event"]).sum() / panic_calls)
            if panic_calls
            else np.nan
        ),
        "downside_threshold": float(downside_threshold),
        "turbulence_threshold": float(turbulence_threshold),
    }
    by_state = (
        df.groupby("State")
        .agg(
            observations=("State", "count"),
            stress_rate=("stress_event", "mean"),
            avg_future_return=("future_return", "mean"),
            avg_future_downside=("future_downside_move", "mean"),
            avg_future_abs_move=("future_abs_move", "mean"),
        )
        .reset_index()
    )
    return df, metrics, by_state


def pct(value) -> str:
    return "N/A" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def num(value, digits: int = 2) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"


def _fmt_date(value) -> str:
    if pd.isna(value):
        return "N/A"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _latest_age_days(value) -> int | None:
    if pd.isna(value):
        return None
    latest = pd.to_datetime(value).tz_localize(None).normalize()
    today = pd.Timestamp.today().normalize()
    return max(int((today - latest).days), 0)


def prepare_taxonomy_frames(
    features_df: pd.DataFrame, regimes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = attach_asset_class(features_df, default=DEFAULT_ASSET_CLASS)
    mapping = ticker_asset_class_map(features, default=DEFAULT_ASSET_CLASS)
    regime_frame = attach_asset_class(
        regimes, default=DEFAULT_ASSET_CLASS, ticker_to_asset_class=mapping
    )
    return features, regime_frame


def _lane_counts(
    features_df: pd.DataFrame, regimes: pd.DataFrame, asset_class: str
) -> tuple[int, int]:
    if "asset_class" not in features_df.columns or "asset_class" not in regimes.columns:
        return 0, 0
    feature_lane = features_df[features_df["asset_class"].eq(asset_class)]
    regime_lane = regimes[regimes["asset_class"].eq(asset_class)]
    if feature_lane.empty or regime_lane.empty:
        return int(len(regime_lane)), 0
    feature_tickers = set(feature_lane["ticker"].dropna().astype(str))
    regime_tickers = set(regime_lane["ticker"].dropna().astype(str))
    return int(len(regime_lane)), len(feature_tickers & regime_tickers)


def _taxonomy_context(
    asset_class: str, features_df: pd.DataFrame, regimes: pd.DataFrame
) -> dict:
    local_rows, local_assets = _lane_counts(features_df, regimes, asset_class)
    return taxonomy_row(
        asset_class, ASSET_TAXONOMY, local_rows=local_rows, local_assets=local_assets
    )


def render_lane_selector(features_df: pd.DataFrame, regimes: pd.DataFrame) -> str:
    observed = set(
        features_df.get("asset_class", pd.Series(dtype=str)).dropna().astype(str)
    )
    observed.update(
        regimes.get("asset_class", pd.Series(dtype=str)).dropna().astype(str)
    )
    options = taxonomy_options(
        ASSET_TAXONOMY, observed=observed, default=DEFAULT_ASSET_CLASS
    )
    default_index = (
        options.index(DEFAULT_ASSET_CLASS) if DEFAULT_ASSET_CLASS in options else 0
    )
    return st.selectbox(
        lt["asset_class"],
        options,
        index=default_index,
        format_func=lambda value: value,
        key="regime_asset_lane",
    )


def filter_asset_lane(
    features_df: pd.DataFrame, regimes: pd.DataFrame, asset_class: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        features_df[features_df["asset_class"].eq(asset_class)].copy(),
        regimes[regimes["asset_class"].eq(asset_class)].copy(),
    )


def render_signal_quality(merged_data: pd.DataFrame) -> None:
    latest = merged_data.iloc[-1]
    confidence = float(latest[[f"prob_{state}" for state in STATE_KEYS]].max())
    panic_prob = float(latest["prob_Panic"])
    current_state = str(latest["State"])
    age_days = _latest_age_days(latest["date"])
    rows = len(merged_data)

    if age_days is not None and age_days > 30:
        st.warning(f"**{lt['quality_title']}** - {lt['quality_stale']}")
    elif rows < 252:
        st.warning(f"**{lt['quality_title']}** - {lt['quality_sparse']}")
    elif confidence < 0.50:
        st.warning(f"**{lt['quality_title']}** - {lt['quality_unclear']}")
    elif current_state == "Panic" and panic_prob >= 0.60 and confidence >= 0.60:
        st.warning(f"**{lt['quality_title']}** - {lt['quality_risk']}")
    elif confidence >= 0.65 and rows >= 500:
        st.success(f"**{lt['quality_title']}** - {lt['quality_promising']}")
    else:
        st.info(f"**{lt['quality_title']}** - {lt['quality_ok']}")


def render_current_cards(
    merged_data: pd.DataFrame, vq_asset: pd.DataFrame, health: dict
):
    latest = merged_data.iloc[-1]
    current_state = str(latest["State"])
    confidence = latest[[f"prob_{state}" for state in STATE_KEYS]].max()
    current_vq = "N/A"
    if not vq_asset.empty and "vq_code" in vq_asset.columns:
        valid_vq = vq_asset.dropna(subset=["vq_code"])
        if not valid_vq.empty:
            current_vq = str(int(valid_vq.iloc[-1]["vq_code"]))

    cols = st.columns(6)
    cols[0].metric(lt["current_state"], current_state)
    cols[1].metric(lt["panic_prob"], pct(latest["prob_Panic"]))
    cols[2].metric(lt["confidence"], pct(confidence))
    cols[3].metric(lt["hurst"], num(latest["hurst"], 3))
    cols[4].metric(lt["vq_code"], current_vq)
    cols[5].metric(lt["latest_date"], _fmt_date(latest["date"]))
    st.caption(f"{lt['sample_rows']}: {len(merged_data):,}")
    render_signal_quality(merged_data)
    with st.expander(lt["current_help_title"], expanded=False):
        st.markdown(lt["current_help"])

    largest_code = health.get("largest_code_pct") if health else np.nan
    if pd.notna(largest_code) and largest_code > 0.80:
        st.warning(lt["vq_collapse"])


def render_timeline(merged_data: pd.DataFrame):
    st.markdown(f"**{lt['asset']}: {selected_ticker} | Walk-Forward Time Series**")
    fig_ts = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
    )
    fig_ts.add_trace(
        go.Scatter(
            x=merged_data["date"],
            y=merged_data["close"],
            mode="lines",
            name=t.get("regime_close_price", "Close Price"),
            line=dict(color="#E0E0E0", width=1.5),
        ),
        row=1,
        col=1,
    )
    panic_zones = merged_data[merged_data["is_panic"]]
    if not panic_zones.empty:
        fig_ts.add_trace(
            go.Scatter(
                x=panic_zones["date"],
                y=panic_zones["close"],
                mode="markers",
                name=lt["panic_flag"],
                marker=dict(color="red", size=5, symbol="circle"),
            ),
            row=1,
            col=1,
        )

    probability_fills = {
        "Quiet": "rgba(44, 160, 44, 0.55)",
        "Chop": "rgba(255, 127, 14, 0.55)",
        "Panic": "rgba(214, 39, 40, 0.65)",
    }
    for state_name in STATE_KEYS:
        fig_ts.add_trace(
            go.Scatter(
                x=merged_data["date"],
                y=merged_data[f"prob_{state_name}"],
                mode="lines",
                stackgroup="one",
                name=state_name,
                fillcolor=probability_fills[state_name],
                line=dict(width=0),
            ),
            row=2,
            col=1,
        )
    fig_ts.add_trace(
        go.Scatter(
            x=merged_data["date"],
            y=merged_data["dynamic_threshold"],
            mode="lines",
            name=lt["dynamic_threshold"],
            line=dict(color="black", width=2, dash="dot"),
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    for feature, label, color in [
        ("gk_z", t.get("regime_feat_gk", "GK Volatility"), "#f1c40f"),
        ("amihud_z", t.get("regime_feat_amihud", "Amihud Illiquidity"), "#9b59b6"),
        ("ker_z", t.get("regime_feat_ker", "Trend Efficiency"), "#3498db"),
    ]:
        fig_ts.add_trace(
            go.Scatter(
                x=merged_data["date"],
                y=merged_data[feature],
                mode="lines",
                name=label,
                line=dict(color=color, width=1),
            ),
            row=3,
            col=1,
        )
    fig_ts.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5, row=3, col=1)
    fig_ts.update_layout(
        height=720,
        hovermode="x unified",
        template=tpl,
        margin=dict(t=10, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_ts.update_yaxes(title_text=t.get("regime_y_price", "Price"), row=1, col=1)
    fig_ts.update_yaxes(
        title_text=t.get("regime_y_prob", "Probability"),
        range=[0, 1],
        tickformat=".0%",
        row=2,
        col=1,
    )
    fig_ts.update_yaxes(
        title_text=t.get("regime_stress_title", "Z-Scores"), row=3, col=1
    )
    st.plotly_chart(fig_ts, width="stretch")

    with st.expander(lt["timeline_help_title"]):
        st.markdown(lt["timeline_help"])


def render_phase_space(merged_data: pd.DataFrame):
    st.markdown(f"**{t.get('regime_phase_title', 'Covariance Phase Space')}**")
    fig_phase = px.scatter_3d(
        merged_data,
        x="amihud_z",
        y="gk_z",
        z="ker_z",
        color="State",
        color_discrete_map=STATE_COLORS,
        hover_data={
            "date": "|%Y-%m-%d",
            "amihud_z": ":.2f",
            "gk_z": ":.2f",
            "ker_z": ":.2f",
            "State": False,
        },
        opacity=0.62,
    )
    fig_phase.update_layout(
        height=680,
        template=tpl,
        margin=dict(t=0, l=0, r=0, b=0),
        scene=dict(
            xaxis_title="Amihud (Illiquidity)",
            yaxis_title="Volatility (GK)",
            zaxis_title="Trend Efficiency (KER)",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_phase, width="stretch")
    with st.expander(lt["phase_help_title"]):
        st.markdown(lt["phase_help"])


def classify_quadrant(row):
    vol = row["gk_z"]
    h = row["hurst"]
    if vol > 0 and h > 0.5:
        return "Trend Highway"
    if vol > 0 and h <= 0.5:
        return "Toxic Whipsaw"
    if vol <= 0 and h > 0.5:
        return "Slow Grind"
    return "Sleepy Chop"


def render_radar(merged_data: pd.DataFrame):
    radar_df = merged_data.copy()
    radar_df["Regime_Quadrant"] = radar_df.apply(classify_quadrant, axis=1)
    fig = px.scatter(
        radar_df,
        x="hurst",
        y="gk_z",
        color="Regime_Quadrant",
        color_discrete_map=QUAD_COLORS,
        hover_data={
            "date": "|%Y-%m-%d",
            "close": ":.2f",
            "gk_z": ":.2f",
            "hurst": ":.2f",
        },
        opacity=0.7,
    )
    fig.add_vline(
        x=0.5, line_width=2, line_dash="dash", line_color="white", opacity=0.5
    )
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(
        height=560,
        template=tpl,
        xaxis_title="Hurst Exponent: Rough/Mean-Reverting to Smooth/Trending",
        yaxis_title="GK Volatility Z-Score",
        xaxis=dict(range=[0, 1]),
        legend=dict(
            title="", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
    )
    st.plotly_chart(fig, width="stretch")


def render_vq_cross_check(
    merged_data: pd.DataFrame,
    regimes: pd.DataFrame,
    latent_result: dict,
    usage_df: pd.DataFrame,
    vq_asset: pd.DataFrame,
):
    st.markdown(f"### {lt['vq_title']}")
    with st.expander("How to read / 使用说明", expanded=False):
        st.markdown(lt["vq_help"])

    if not latent_result:
        st.info(lt["no_vq"])
        return

    left, right = st.columns([0.44, 0.56])
    with left:
        st.markdown(f"#### {lt['vq_usage']}")
        if usage_df.empty:
            st.info(lt["no_vq"])
        else:
            fig_usage = px.bar(
                usage_df,
                x="vq_code",
                y="usage_pct",
                color="usage_pct",
                color_continuous_scale="Viridis",
                template=tpl,
            )
            fig_usage.update_layout(
                height=330, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_usage, width="stretch")

    with right:
        st.markdown(f"#### {lt['vq_gmm']}")
        latent_df = latent_result.get("latent", pd.DataFrame())
        if latent_df.empty:
            st.info(lt["no_vq"])
        else:
            merged_latent = merge_gmm_probabilities(latent_df, regimes)
            _, row_pct = compute_gmm_overlap(merged_latent)
            if row_pct.empty:
                st.info(lt["no_vq"])
            else:
                heat = row_pct.set_index("vq_code")
                fig_gmm = go.Figure(
                    data=go.Heatmap(
                        z=heat.values,
                        x=[f"GMM {col}" for col in heat.columns],
                        y=heat.index,
                        colorscale="Blues",
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="row %"),
                    )
                )
                fig_gmm.update_layout(
                    template=tpl, height=330, margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_gmm, width="stretch")

    st.markdown(f"#### {lt['vq_timeline']}")
    if vq_asset.empty or "vq_code" not in vq_asset.columns:
        st.info(lt["vq_missing_asset"])
        return
    fig_vq = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
    )
    fig_vq.add_trace(
        go.Scatter(
            x=merged_data["date"], y=merged_data["close"], mode="lines", name="Close"
        ),
        row=1,
        col=1,
    )
    fig_vq.add_trace(
        go.Scatter(
            x=vq_asset["date"],
            y=vq_asset["vq_code"],
            mode="markers+lines",
            name="VQ Code",
            line=dict(color="#40C4FF", width=1.5),
            marker=dict(size=5, color=vq_asset["vq_code"], colorscale="Viridis"),
        ),
        row=2,
        col=1,
    )
    fig_vq.update_layout(template=tpl, height=430, margin=dict(l=10, r=10, t=20, b=10))
    fig_vq.update_yaxes(title_text="Price", row=1, col=1)
    fig_vq.update_yaxes(title_text="VQ Code", row=2, col=1)
    st.plotly_chart(fig_vq, width="stretch")


def render_meta_diagnostics(merged_data: pd.DataFrame):
    st.markdown(f"### {lt['meta_title']}")
    st.caption(lt["meta_caption"])
    with st.expander("How to read / 使用说明", expanded=False):
        st.markdown(lt["meta_help"])

    horizon = st.selectbox(
        lt["horizon"], [1, 5, 10, 20], index=1, format_func=lambda value: f"{value}d"
    )
    meta_df, metrics, by_state = compute_meta_labels(merged_data, int(horizon))
    if meta_df.empty:
        st.info("Not enough future observations for this horizon.")
        return

    cols = st.columns(4)
    cols[0].metric(lt["stress_base"], pct(metrics.get("stress_base_rate")))
    cols[1].metric(lt["panic_precision"], pct(metrics.get("panic_precision")))
    cols[2].metric(lt["stress_recall"], pct(metrics.get("stress_recall")))
    cols[3].metric(lt["false_alarm"], pct(metrics.get("false_alarm_rate")))
    with st.expander(lt["meta_metric_help_title"], expanded=False):
        st.markdown(lt["meta_metric_help"])

    left, right = st.columns([0.55, 0.45])
    with left:
        st.markdown(f"#### {lt['meta_box']}")
        plot_df = meta_df.copy()
        plot_df["future_return_bps"] = plot_df["future_return"] * 10_000.0
        fig_box = px.box(
            plot_df,
            x="State",
            y="future_return_bps",
            color="State",
            color_discrete_map=STATE_COLORS,
            template=tpl,
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_box.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Future Return (bps)",
        )
        st.plotly_chart(fig_box, width="stretch")

    with right:
        st.markdown(f"#### {lt['meta_rates']}")
        if by_state.empty:
            st.info("No state-level metrics available.")
        else:
            fig_rates = px.bar(
                by_state,
                x="State",
                y="stress_rate",
                color="State",
                color_discrete_map=STATE_COLORS,
                template=tpl,
            )
            fig_rates.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_tickformat=".0%",
                yaxis_title="Stress Rate",
                showlegend=False,
            )
            st.plotly_chart(fig_rates, width="stretch")
    st.dataframe(
        by_state.style.format(
            {
                "stress_rate": "{:.1%}",
                "avg_future_return": "{:.2%}",
                "avg_future_downside": "{:.2%}",
                "avg_future_abs_move": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_pipeline_audit(merged_data: pd.DataFrame):
    st.markdown(f"### {lt['pipeline']}")
    st.markdown(
        """
The regime input pipeline uses winsorized and smoothed microstructure features, rolling z-scores, small numerical jitter in the model training path, and EWMA-smoothed probabilities to reduce regime flicker.
        """
    )
    st.markdown(f"#### {lt['hist']}")
    cols = st.columns(3)
    for col, feature, title, color in [
        (cols[0], "gk_z", "GK Volatility", "#f1c40f"),
        (cols[1], "amihud_z", "Amihud Illiquidity", "#9b59b6"),
        (cols[2], "ker_z", "Trend Efficiency", "#3498db"),
    ]:
        fig = px.histogram(
            merged_data,
            x=feature,
            nbins=50,
            title=title,
            color_discrete_sequence=[color],
        )
        fig.update_layout(
            template=tpl,
            height=250,
            margin=dict(t=30, l=10, r=10, b=10),
            showlegend=False,
        )
        col.plotly_chart(fig, width="stretch")


def render_gmm_density(merged_data: pd.DataFrame):
    st.markdown(f"### {lt['density']}")
    x_min, x_max = merged_data["amihud_z"].min(), merged_data["amihud_z"].max()
    y_min, y_max = merged_data["gk_z"].min(), merged_data["gk_z"].max()
    x_pad = max((x_max - x_min) * 0.1, 0.1)
    y_pad = max((y_max - y_min) * 0.1, 0.1)
    x_range = np.linspace(x_min - x_pad, x_max + x_pad, 80)
    y_range = np.linspace(y_min - y_pad, y_max + y_pad, 80)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    pos = np.dstack((x_grid, y_grid))

    fig = go.Figure()
    surface_colors = {"Quiet": "Greens", "Chop": "Oranges", "Panic": "Reds"}
    rendered = 0
    for state_name in STATE_KEYS:
        state_data = merged_data[merged_data["State"] == state_name][
            ["amihud_z", "gk_z"]
        ].dropna()
        if len(state_data) <= 10:
            continue
        mu = state_data.mean().to_numpy()
        cov = state_data.cov().to_numpy() + (np.eye(2) * 1e-4)
        try:
            z = stats.multivariate_normal(mu, cov).pdf(pos)
        except Exception:
            continue
        if z.max() > 0:
            z = z / z.max()
        fig.add_trace(
            go.Surface(
                z=z,
                x=x_grid,
                y=y_grid,
                name=state_name,
                colorscale=surface_colors[state_name],
                showscale=False,
                opacity=0.85,
            )
        )
        rendered += 1

    if not rendered:
        st.info(lt["density_empty"])
        return
    fig.update_layout(
        height=580,
        template=tpl,
        margin=dict(t=0, l=0, r=0, b=0),
        scene=dict(
            xaxis_title="Illiquidity (Amihud Z)",
            yaxis_title="Volatility (GK Z)",
            zaxis_title="Normalized Density",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.6)),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander(lt["density_help_title"]):
        st.markdown(lt["density_help"])


def render_state_profiler(merged_data: pd.DataFrame):
    st.markdown(f"### {lt['profiler']}")
    profiler = (
        merged_data.groupby("State")
        .agg(
            Frequency=("State", "count"),
            Avg_Daily_Return=("daily_return", "mean"),
            Volatility_Z=("gk_z", "mean"),
            Illiquidity_Z=("amihud_z", "mean"),
            Trend_Efficiency=("ker_20d", "mean"),
        )
        .reset_index()
    )
    if profiler.empty:
        st.info("No state profile available.")
        return
    total_days = max(profiler["Frequency"].sum(), 1)
    display = profiler.copy()
    display["% of Time"] = display["Frequency"] / total_days
    table = display[
        [
            "State",
            "% of Time",
            "Avg_Daily_Return",
            "Volatility_Z",
            "Illiquidity_Z",
            "Trend_Efficiency",
        ]
    ]
    left, right = st.columns([1, 1.5])
    with left:
        st.dataframe(
            table.style.format(
                {
                    "% of Time": "{:.1%}",
                    "Avg_Daily_Return": "{:.2%}",
                    "Volatility_Z": "{:.2f}",
                    "Illiquidity_Z": "{:.2f}",
                    "Trend_Efficiency": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        melted = profiler.melt(
            id_vars=["State"],
            value_vars=["Volatility_Z", "Illiquidity_Z", "Trend_Efficiency"],
            var_name="Metric",
            value_name="Value",
        )
        fig = px.bar(
            melted,
            x="State",
            y="Value",
            color="Metric",
            barmode="group",
            template=tpl,
            color_discrete_map={
                "Volatility_Z": "#f1c40f",
                "Illiquidity_Z": "#9b59b6",
                "Trend_Efficiency": "#3498db",
            },
        )
        fig.update_layout(height=310, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch")


matrix_path = os.path.join(
    ALPHA_RUNTIME_DATA_ROOT, "feature_store", "ML_Feature_Matrix.parquet"
)
regimes_path = os.path.join(
    ALPHA_RUNTIME_DATA_ROOT, "regime", "GMM_Rolling_Probabilities.parquet"
)
regimes, features_df = load_data(
    matrix_path,
    regimes_path,
    _file_mtime(matrix_path),
    _file_mtime(regimes_path),
)

st.title(lt["title"])
st.caption(lt["subtitle"])

if features_df.empty:
    st.warning("Feature matrix is missing or empty. Run feature_engineering.py first.")
    st.stop()
if regimes.empty:
    st.warning(
        "GMM probability file is missing or empty. Run train_rolling_gmm.py first."
    )
    st.stop()

features_df, regimes = prepare_taxonomy_frames(features_df, regimes)

selector_left, selector_right = st.columns(2, vertical_alignment="bottom")
with selector_left:
    selected_asset_class = render_lane_selector(features_df, regimes)
lane_row = _taxonomy_context(selected_asset_class, features_df, regimes)

features_df, regimes = filter_asset_lane(features_df, regimes, selected_asset_class)
if features_df.empty or regimes.empty:
    if not lane_row.get("has_local_regime_data", False):
        st.warning(lt["taxonomy_status_empty"])
    elif not lane_row.get("vectorizable", True):
        st.warning(lt["taxonomy_status_nonvector"])
    st.stop()

feature_tickers = set(features_df["ticker"].dropna().astype(str))
regime_tickers = set(regimes["ticker"].dropna().astype(str))
asset_options = sorted(feature_tickers & regime_tickers)
if not asset_options:
    st.warning("No assets found in feature matrix.")
    st.stop()

with selector_right:
    selected_ticker = st.selectbox(lt["asset"], asset_options, key="regime_asset")

merged_data, state_map = prepare_asset_data(features_df, regimes, selected_ticker)
if merged_data.empty:
    st.warning("No overlapping feature/regime rows for the selected asset.")
    st.stop()

artifact_dir = Path(ALPHA_RUNTIME_ARTIFACT_ROOT) / "latent_factors"
latent_artifact_mtime = max(
    (_file_mtime(path) for path in artifact_dir.glob("*") if path.is_file()),
    default=0.0,
)
latent_result = load_vq_artifacts(str(artifact_dir), latent_artifact_mtime)
vq_asset, usage_df, vq_health = merge_asset_vq(
    merged_data, latent_result, selected_ticker
)

render_current_cards(merged_data, vq_asset, vq_health)

tabs = st.tabs(
    [lt["tab_timeline"], lt["tab_geometry"], lt["tab_diag"], lt["tab_cross"]]
)

with tabs[0]:
    st.info(lt["current_context"])
    render_timeline(merged_data)

with tabs[1]:
    st.info(lt["geometry_context"])
    col_phase, col_radar = st.columns([1.05, 0.95])
    with col_phase:
        render_phase_space(merged_data)
    with col_radar:
        render_radar(merged_data)

with tabs[2]:
    st.info(lt["diagnostic_context"])
    render_pipeline_audit(merged_data)
    st.markdown("---")
    render_gmm_density(merged_data)
    st.markdown("---")
    render_state_profiler(merged_data)
    with st.expander(lt["profiler_help_title"]):
        st.markdown(lt["profiler_help"])

with tabs[3]:
    st.info(lt["validation_context"])
    render_vq_cross_check(merged_data, regimes, latent_result, usage_df, vq_asset)
    st.markdown("---")
    render_meta_diagnostics(merged_data)
