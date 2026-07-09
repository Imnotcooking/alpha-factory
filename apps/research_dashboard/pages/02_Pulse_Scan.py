from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

PROJECT_ROOT = os.path.dirname(UI_DIR)
REPO_ROOT = str(Path(PROJECT_ROOT).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if os.environ.get("OQP_EMBEDDED_STREAMLIT_PAGE") != "1":
    st.set_page_config(page_title="Pulse Scan", layout="wide")

_CONFIG_PATH = Path(UI_DIR) / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("_ui_v2_config", _CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise ImportError(f"Unable to load UI config from {_CONFIG_PATH}")
_UI_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_UI_CONFIG)

BASE_DIR = _UI_CONFIG.BASE_DIR
DB_PATH = _UI_CONFIG.DB_PATH
LOGS_DIR = _UI_CONFIG.LOGS_DIR
ALPHA_RUNTIME_DATA_ROOT = _UI_CONFIG.ALPHA_RUNTIME_DATA_ROOT
get_plotly_template = _UI_CONFIG.get_plotly_template
from tick_pulse_lab.asset_ranker_view import render_asset_download_ranker
from tick_pulse_lab.pulse_discovery import (
    PULSE_CACHE_TYPE,
    build_pulse_zone_summary,
    build_directionless_pulse_frame,
    detect_directionless_pulse_events,
    summarize_pulse_file,
    summarize_pulse_zones_file,
)
from tick_pulse_lab.hypothesis_seeds import save_hypothesis_seed
from tick_pulse_lab.research_cache import (
    CACHE_SCHEMA_VERSION,
    get_or_compute_dataframe,
    load_cached_dataframe,
    make_cache_key,
)
from tick_pulse_lab.text import PAGE_TEXT as TICK_LAB_TEXT
from tick_pulse_lab.views import _render_contract_health, _render_raw_row_viewer
from oqp.data.runtime_paths import discover_futures_cn_tick_files
from oqp.research.tick_pulse import (
    DEFAULT_TICK_FILE,
    contract_summary,
    load_tick_scope,
    load_ticks,
)
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)


from oqp.ui.translations import research_page_legacy_catalog


TEXT = research_page_legacy_catalog("pulse_discovery_lab")


def _rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO_ROOT)


def _resolve(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    repo_candidate = Path(REPO_ROOT) / path_obj
    if repo_candidate.exists():
        return str(repo_candidate)
    return str(Path(PROJECT_ROOT) / path_obj)


def _ensure_option_state(key: str, options: list, default_value):
    if not options:
        return None
    if default_value not in options:
        default_value = options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default_value
    return st.session_state[key]


def _render_nav(options: dict[str, str], t: dict) -> str:
    option_keys = list(options.keys())
    _ensure_option_state("pulse_discovery_nav", option_keys, option_keys[0])
    selected = st.pills(
        t["nav_label"],
        option_keys,
        selection_mode="single",
        format_func=lambda key: options[key],
        key="pulse_discovery_nav",
        label_visibility="collapsed",
    )
    return selected or option_keys[0]


def _render_page_workflow(t: dict) -> None:
    with st.expander(t["workflow_title"], expanded=False):
        st.markdown(t["workflow_text"])


def _render_section_context(section: str, t: dict) -> None:
    context_key = {
        "ranker": "ranker_context",
        "universe": "universe_context",
        "single": "workspace_context",
        "cross": "cross_context",
    }.get(section)
    if context_key:
        st.info(t[context_key])


def _render_scope_story(
    *,
    selected_file: str,
    selected_symbol: str,
    window_seconds: float,
    percentile_pct: float,
    t: dict,
) -> None:
    st.caption(
        t["scope_story"].format(
            file=selected_file,
            symbol=selected_symbol,
            window=f"{window_seconds:.0f}",
            percentile=f"{percentile_pct:.1f}",
        )
    )


def _render_pulse_validity_callout(
    *,
    frame: pd.DataFrame,
    events: pd.DataFrame,
    event_threshold: float,
    t: dict,
) -> None:
    frame_rows = int(len(frame))
    event_count = int(len(events))
    if event_count == 0:
        st.warning(t["validity_no_events"])
    elif frame_rows < 1_000:
        st.warning(t["validity_sparse"])
    elif event_count < 20:
        st.info(t["validity_few_events"])
    elif np.isfinite(event_threshold) and event_threshold < 1.0:
        st.warning(t["validity_tiny_threshold"])
    elif event_count >= 30 and np.isfinite(event_threshold) and event_threshold >= 2.0:
        st.success(t["validity_promising"])
    else:
        st.info(t["validity_ok"])


def _product_hint(path: str) -> str:
    match = re.match(r"^\d+contract_([A-Za-z]+)_raw_", os.path.basename(path))
    return match.group(1) if match else ""


PRODUCT_NAMES_ZH = {
    "a": "黄大豆1号",
    "ag": "白银",
    "al": "铝",
    "ao": "氧化铝",
    "au": "黄金",
    "bu": "沥青",
    "c": "玉米",
    "cf": "棉花",
    "cu": "铜",
    "eb": "苯乙烯",
    "ec": "集运欧线",
    "eg": "乙二醇",
    "fg": "玻璃",
    "fu": "燃料油",
    "hc": "热卷",
    "i": "铁矿石",
    "j": "焦炭",
    "jm": "焦煤",
    "lc": "碳酸锂",
    "lh": "生猪",
    "lu": "低硫燃料油",
    "m": "豆粕",
    "ma": "甲醇",
    "ni": "镍",
    "p": "棕榈油",
    "pb": "铅",
    "pf": "短纤",
    "pg": "液化石油气",
    "pp": "聚丙烯",
    "ps": "多晶硅",
    "rb": "螺纹钢",
    "rm": "菜粕",
    "ru": "橡胶",
    "sa": "纯碱",
    "sc": "原油",
    "sf": "硅铁",
    "si": "工业硅",
    "sm": "锰硅",
    "sn": "锡",
    "sp": "纸浆",
    "sr": "白糖",
    "ss": "不锈钢",
    "ta": "PTA",
    "v": "PVC",
    "y": "豆油",
    "zn": "锌",
}

PRODUCT_NAMES_EN = {
    "ag": "Silver",
    "au": "Gold",
    "cu": "Copper",
    "fg": "Glass",
    "fu": "Fuel oil",
    "jm": "Coking coal",
    "lc": "Lithium carbonate",
    "ma": "Methanol",
    "ps": "Polysilicon",
    "sa": "Soda ash",
    "ta": "PTA",
}


def _product_name(asset: str, t: dict) -> str:
    asset_text = str(asset)
    key = asset_text.lower()
    if t.get("_lang") == "ZH":
        return PRODUCT_NAMES_ZH.get(key, asset_text)
    return PRODUCT_NAMES_EN.get(key, asset_text)


def _product_label(asset: str, t: dict) -> str:
    name = _product_name(asset, t)
    asset_text = str(asset)
    return asset_text if name == asset_text else f"{asset_text} ({name})"


def _discover_tick_files() -> list[dict]:
    rows = []
    for path in sorted(
        discover_futures_cn_tick_files(patterns=("*_tick_all_data.parquet",)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        product = _product_hint(path.name)
        rel = os.path.relpath(path, REPO_ROOT)
        size_mb = path.stat().st_size / (1024 * 1024)
        label = f"{path.name} ({size_mb:.1f} MB)"
        rows.append(
            {
                "path": rel,
                "label": label,
                "mtime": path.stat().st_mtime,
                "product_hint": product,
                "size_mb": size_mb,
            }
        )
    return rows


def _dedupe_product_files(files: list[dict]) -> list[dict]:
    selected = {}
    for item in files:
        product = str(item.get("product_hint") or item["path"]).lower()
        current = selected.get(product)
        if current is None or (item["size_mb"], item["mtime"]) > (
            current["size_mb"],
            current["mtime"],
        ):
            selected[product] = item
    return sorted(
        selected.values(),
        key=lambda item: str(item.get("product_hint") or item["path"]).lower(),
    )


@st.cache_data(show_spinner=False)
def _contract_summary(path: str, mtime: float) -> pd.DataFrame:
    source_file = _rel(path)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "pulse_discovery_contract_summary",
        "source_file": source_file,
        "source_mtime": float(mtime),
    }
    cache_key = make_cache_key(payload)

    return get_or_compute_dataframe(
        db_path=DB_PATH,
        logs_dir=LOGS_DIR,
        base_dir=BASE_DIR,
        cache_key=cache_key,
        metadata={
            "cache_type": "pulse_discovery_contract_summary",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": "ALL",
            "symbol": "ALL",
            "hypothesis": "",
            "threshold_mode": "",
            "window": 0,
            "min_success_ticks": 0.0,
            "horizon_set": "",
            "backend": "python",
        },
        compute_fn=lambda: contract_summary(load_ticks(path)),
    ).data


@st.cache_data(show_spinner=False)
def _tick_scope(path: str, mtime: float, product: str, symbol: str) -> pd.DataFrame:
    return load_tick_scope(path, product=product, symbol=symbol)


@st.cache_data(show_spinner=False)
def _pulse_frame(
    path: str,
    mtime: float,
    product: str,
    symbol: str,
    window_seconds: float,
    percentile: float,
    collapse_gap: float,
) -> pd.DataFrame:
    source_file = _rel(path)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": f"{PULSE_CACHE_TYPE}_frame",
        "source_file": source_file,
        "source_mtime": float(mtime),
        "product": product,
        "symbol": symbol,
        "window_seconds": float(window_seconds),
        "percentile": float(percentile),
        "collapse_gap_seconds": float(collapse_gap),
    }
    cache_key = make_cache_key(payload)

    result = get_or_compute_dataframe(
        db_path=DB_PATH,
        logs_dir=LOGS_DIR,
        base_dir=BASE_DIR,
        cache_key=cache_key,
        metadata={
            "cache_type": f"{PULSE_CACHE_TYPE}_frame",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": product,
            "symbol": symbol,
            "hypothesis": "",
            "threshold_mode": "directionless_percentile",
            "window": int(round(window_seconds * 1000)),
            "min_success_ticks": float(percentile),
            "horizon_set": str(collapse_gap),
            "backend": "python_numpy",
        },
        compute_fn=lambda: build_directionless_pulse_frame(
            load_tick_scope(path, product=product, symbol=symbol),
            window_seconds=window_seconds,
        ),
    )
    frame = result.data
    frame.attrs["pulse_window_seconds_config"] = float(window_seconds)
    frame.attrs["pulse_cache_hit"] = result.cache_hit
    frame.attrs["pulse_cache_elapsed"] = result.elapsed_seconds
    return frame


@st.cache_data(show_spinner=False)
def _pulse_events(
    path: str,
    mtime: float,
    product: str,
    symbol: str,
    window_seconds: float,
    percentile: float,
    collapse_gap: float,
) -> pd.DataFrame:
    source_file = _rel(path)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": f"{PULSE_CACHE_TYPE}_events",
        "source_file": source_file,
        "source_mtime": float(mtime),
        "product": product,
        "symbol": symbol,
        "window_seconds": float(window_seconds),
        "percentile": float(percentile),
        "collapse_gap_seconds": float(collapse_gap),
    }
    cache_key = make_cache_key(payload)
    cached = load_cached_dataframe(DB_PATH, cache_key, BASE_DIR)
    if cached is not None:
        events, metadata = cached
        events.attrs["pulse_cache_hit"] = True
        events.attrs["pulse_cache_elapsed"] = float(
            metadata.get("elapsed_seconds") or 0.0
        )
        return events

    frame = _pulse_frame(
        path, mtime, product, symbol, window_seconds, percentile, collapse_gap
    )
    frame.attrs["pulse_window_seconds_config"] = float(window_seconds)
    result = get_or_compute_dataframe(
        db_path=DB_PATH,
        logs_dir=LOGS_DIR,
        base_dir=BASE_DIR,
        cache_key=cache_key,
        metadata={
            "cache_type": f"{PULSE_CACHE_TYPE}_events",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": product,
            "symbol": symbol,
            "hypothesis": "",
            "threshold_mode": "directionless_percentile",
            "window": int(round(window_seconds * 1000)),
            "min_success_ticks": float(percentile),
            "horizon_set": str(collapse_gap),
            "backend": "python_numpy",
        },
        compute_fn=lambda: detect_directionless_pulse_events(
            frame,
            percentile=percentile,
            collapse_gap_seconds=collapse_gap,
        ),
    )
    events = result.data
    events.attrs["pulse_cache_hit"] = result.cache_hit
    events.attrs["pulse_cache_elapsed"] = result.elapsed_seconds
    return events


def _severity_ladder_figure(
    frame: pd.DataFrame, selected_source_index: int | None, tpl: str, t: dict
) -> go.Figure:
    values = (
        pd.to_numeric(frame["pulse_abs_move_ticks"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    values = values[values > 0]
    if values.empty:
        return go.Figure()
    percentile_grid = np.array(
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5, 99.9, 100.0]
    )
    quantiles = [float(values.quantile(p / 100.0)) for p in percentile_grid]
    fig = go.Figure()
    zones = [
        (0, 90, "rgba(56, 189, 248, 0.12)", t["zone_calm"]),
        (90, 95, "rgba(34, 197, 94, 0.12)", t["zone_active"]),
        (95, 99, "rgba(245, 158, 11, 0.14)", t["zone_watch"]),
        (99, 99.5, "rgba(239, 68, 68, 0.18)", t["zone_pulse"]),
        (99.5, 100, "rgba(124, 58, 237, 0.16)", t["zone_extreme"]),
    ]
    for x0, x1, color, label in zones:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )
    fig.add_trace(
        go.Scatter(
            x=percentile_grid,
            y=quantiles,
            mode="lines+markers",
            name=t["abs_move"],
            line=dict(color="#0f172a", width=2.5),
            marker=dict(size=7, color="#0f172a"),
            hovertemplate=f"{t['hover_percentile']}=%{{x:.1f}}%<br>{t['hover_move']}=%{{y:.2f}} ticks<extra></extra>",
        )
    )
    for p, color in [(95, "#f59e0b"), (99, "#ef4444"), (99.5, "#7c3aed")]:
        q = float(values.quantile(p / 100.0))
        fig.add_trace(
            go.Scatter(
                x=[p],
                y=[q],
                mode="markers+text",
                marker=dict(size=12, color=color),
                text=[f"p{p:g}: {q:.1f}"],
                textposition="top center",
                name=f"p{p:g}",
                hovertemplate=f"p{p:g}<br>{t['hover_move']}={q:.2f} ticks<extra></extra>",
            )
        )
    if selected_source_index is not None and selected_source_index in frame.index:
        selected_value = float(frame.loc[selected_source_index, "pulse_abs_move_ticks"])
        selected_percentile = float((values <= selected_value).mean() * 100.0)
        selected_percentile = min(max(selected_percentile, 0.0), 100.0)
        fig.add_trace(
            go.Scatter(
                x=[selected_percentile],
                y=[selected_value],
                mode="markers+text",
                marker=dict(color="#16a34a", size=15, symbol="diamond"),
                text=[t["selected_pulse"]],
                textposition="bottom center",
                name=t["selected_pulse"],
                hovertemplate=f"{t['hover_selected_pulse']}<br>{t['hover_percentile']}=%{{x:.2f}}%<br>{t['hover_move']}=%{{y:.2f}} ticks<extra></extra>",
            )
        )
    fig.update_layout(
        template=tpl,
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title=t["axis_percentile"],
        yaxis_title=t["abs_move"],
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    return fig


def _zone_summary(frame: pd.DataFrame, t: dict) -> pd.DataFrame:
    raw = build_pulse_zone_summary(frame)
    if raw.empty:
        return pd.DataFrame()
    zone_labels = _zone_labels(t)
    return pd.DataFrame(
        {
            t["zone"]: raw["zone_code"].map(zone_labels),
            t["zone_range"]: raw.apply(
                lambda row: f"p{row['percentile_start'] * 100:.1f} - p{row['percentile_end'] * 100:.1f}",
                axis=1,
            ),
            t["zone_windows"]: raw["windows"],
            t["zone_share"]: raw["share"],
            t["zone_avg"]: raw["avg_net_distance_ticks"],
            t["zone_max"]: raw["max_net_distance_ticks"],
        }
    )


def _zone_labels(t: dict) -> dict[str, str]:
    return {
        "normal": t["zone_calm"],
        "active": t["zone_active"],
        "watch": t["zone_watch"],
        "pulse": t["zone_pulse"],
        "extreme": t["zone_extreme"],
    }


def _format_cross_zone_summary(raw: pd.DataFrame, t: dict) -> pd.DataFrame:
    if raw.empty:
        return raw
    zone_labels = _zone_labels(t)
    display = raw.copy()
    display[t["asset"]] = display["asset"]
    display[t["asset_name"]] = display["asset"].map(
        lambda asset: _product_name(asset, t)
    )
    display[t["main_contract"]] = display["main_contract"]
    display[t["zone"]] = display["zone_code"].map(zone_labels)
    display[t["zone_range"]] = display.apply(
        lambda row: f"p{row['percentile_start'] * 100:.1f} - p{row['percentile_end'] * 100:.1f}",
        axis=1,
    )
    display[t["zone_windows"]] = display["windows"]
    display[t["zone_share"]] = display["share"]
    display[t["zone_avg"]] = display["avg_net_distance_ticks"]
    display[t["zone_max"]] = display["max_net_distance_ticks"]
    return display[
        [
            t["asset"],
            t["asset_name"],
            t["main_contract"],
            t["zone"],
            t["zone_range"],
            t["zone_windows"],
            t["zone_share"],
            t["zone_avg"],
            t["zone_max"],
        ]
    ]


def _style_cross_zone_summary(display: pd.DataFrame, t: dict):
    if display.empty:
        return display

    def row_style(row):
        zone = row[t["zone"]]
        if zone == t["zone_extreme"]:
            return ["background-color: rgba(124, 58, 237, 0.18)"] * len(row)
        if zone == t["zone_pulse"]:
            return ["background-color: rgba(239, 68, 68, 0.16)"] * len(row)
        if zone == t["zone_watch"]:
            return ["background-color: rgba(245, 158, 11, 0.12)"] * len(row)
        return [""] * len(row)

    return display.style.apply(row_style, axis=1).format(
        {
            t["zone_share"]: "{:.2%}",
            t["zone_avg"]: "{:.2f}",
            t["zone_max"]: "{:.2f}",
        }
    )


def _format_cross_asset_summary(raw: pd.DataFrame, t: dict) -> pd.DataFrame:
    if raw.empty:
        return raw
    display = raw.copy()
    display[t["asset"]] = display["asset"]
    display[t["asset_name"]] = display["asset"].map(
        lambda asset: _product_name(asset, t)
    )
    display[t["main_contract"]] = display.get("main_contract", "")
    display[t["source"]] = display.get("source_file", "")
    display[t["events"]] = display.get("pulse_events", np.nan)
    display[t["pulses_hour_short"]] = display.get("pulses_per_trading_hour", np.nan)
    display["p95"] = display.get("p95_move_ticks", np.nan)
    display["p99"] = display.get("p99_move_ticks", np.nan)
    display["p99.5"] = display.get("p995_move_ticks", np.nan)
    display[t["threshold"]] = display.get("pulse_threshold_ticks", np.nan)
    display[t["top20_avg"]] = display.get("avg_top20_pulse_ticks", np.nan)
    display[t["rows"]] = display.get("median_snapshots_per_window", np.nan)
    columns = [
        t["asset"],
        t["asset_name"],
        t["main_contract"],
        t["events"],
        t["pulses_hour_short"],
        "p95",
        "p99",
        "p99.5",
        t["threshold"],
        t["top20_avg"],
        t["rows"],
        t["source"],
    ]
    if "error" in display.columns:
        display["Error"] = display["error"].fillna("")
        columns.append("Error")
    return display[columns]


def _event_table(events: pd.DataFrame, top_n: int, t: dict) -> pd.DataFrame:
    if events.empty:
        return events
    display = events.head(top_n).copy()
    display[t["time"]] = (
        pd.to_datetime(display["event_time"])
        .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        .str[:-3]
    )
    display[t["direction"]] = display["pulse_direction"]
    display[t["abs_move"]] = display["pulse_abs_move_ticks"]
    display[t["net_move"]] = display["pulse_net_move_ticks"]
    display[t["velocity"]] = display["pulse_velocity_ticks_per_sec"]
    display[t["volume"]] = display["pulse_volume_delta"]
    display[t["spread"]] = display["pulse_spread_mean"]
    display[t["book"]] = display["pulse_book_imbalance_mean"]
    display[t["flow"]] = display["pulse_flow_imbalance"]
    display[t["range"]] = display["pulse_path_range_ticks"]
    display[t["cluster"]] = display["event_cluster_size"]
    return display[
        [
            "event_rank",
            t["time"],
            t["direction"],
            t["abs_move"],
            t["net_move"],
            t["velocity"],
            t["volume"],
            t["spread"],
            t["book"],
            t["flow"],
            t["range"],
            t["cluster"],
        ]
    ]


def _event_inspector(
    frame: pd.DataFrame, event: pd.Series, window_seconds: float, tpl: str, t: dict
) -> go.Figure:
    event_time = pd.to_datetime(event["event_time"])
    start_time = pd.to_datetime(event["pulse_start_time"])
    buffer = pd.Timedelta(seconds=max(window_seconds * 2.0, 20.0))
    zoom = frame.loc[
        (frame["datetime"] >= start_time - buffer)
        & (frame["datetime"] <= event_time + buffer)
    ].copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    fig.add_trace(
        go.Scatter(
            x=zoom["datetime"],
            y=zoom["last_price"],
            mode="lines",
            name=t["last_price"],
            line=dict(color="#334155", width=1.4),
            customdata=np.stack(
                [
                    zoom["pulse_abs_move_ticks"].fillna(0),
                    zoom["pulse_net_move_ticks"].fillna(0),
                    zoom["pulse_volume_delta"].fillna(0),
                    zoom["pulse_flow_imbalance"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                f"%{{x}}<br>{t['hover_price']}=%{{y:.3f}}<br>"
                f"{t['abs_move']}=%{{customdata[0]:.2f}}<br>"
                f"{t['net_move']}=%{{customdata[1]:.2f}}<br>"
                f"{t['volume']}=%{{customdata[2]:.0f}}<br>"
                f"{t['flow']}=%{{customdata[3]:.2f}}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[start_time],
            y=[event["pulse_start_price"]],
            mode="markers",
            name=t["window_start"],
            marker=dict(color="#f59e0b", size=10),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[event_time],
            y=[event["last_price"]],
            mode="markers",
            name=t["pulse_peak"],
            marker=dict(color="#ef4444", size=11, symbol="x"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=zoom["datetime"],
            y=zoom["volume_delta"],
            name=t["volume"],
            marker_color="#38bdf8",
            opacity=0.65,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=zoom["datetime"],
            y=zoom["spread"],
            mode="lines",
            name=t["spread"],
            line=dict(color="#a855f7", width=1.2),
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=zoom["datetime"],
            y=zoom["book_imbalance"],
            mode="lines",
            name=t["book"],
            line=dict(color="#22c55e", width=1.2),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_vrect(
        x0=start_time,
        x1=event_time,
        fillcolor="#facc15",
        opacity=0.18,
        line_width=0,
        row="all",
        col=1,
    )
    fig.update_layout(
        template=tpl,
        height=620,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text=t["hover_price"], row=1, col=1)
    fig.update_yaxes(title_text=t["volume"], row=2, col=1)
    fig.update_yaxes(title_text=t["spread"], row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text=t["book"], row=3, col=1, secondary_y=True)
    return fig


def _fingerprint(frame: pd.DataFrame, event: pd.Series, t: dict):
    metrics = [
        (
            t["metric_move"],
            "pulse_abs_move_ticks",
            abs(float(event["pulse_abs_move_ticks"])),
        ),
        (
            t["metric_velocity"],
            "pulse_velocity_ticks_per_sec",
            abs(float(event["pulse_velocity_ticks_per_sec"])),
        ),
        (t["metric_volume"], "pulse_volume_delta", float(event["pulse_volume_delta"])),
        (t["metric_spread"], "pulse_spread_mean", float(event["pulse_spread_mean"])),
        (
            t["metric_book"],
            "pulse_book_imbalance_mean",
            float(event["pulse_book_imbalance_mean"]),
        ),
        (
            t["metric_flow"],
            "pulse_flow_imbalance",
            float(event["pulse_flow_imbalance"]),
        ),
    ]
    cols = st.columns(3)
    for idx, (label, column, value) in enumerate(metrics):
        background = (
            pd.to_numeric(frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if column in {"pulse_velocity_ticks_per_sec"}:
            background = background.abs()
        std = float(background.std(ddof=0)) if len(background) > 1 else np.nan
        mean = float(background.mean()) if len(background) else np.nan
        z = (value - mean) / std if np.isfinite(std) and std > 1e-12 else np.nan
        cols[idx % 3].metric(
            label,
            _format_fingerprint_value(value),
            "N/A" if not np.isfinite(z) else f"{z:+.2f}σ",
        )

    elapsed = float(event.get("pulse_window_seconds", np.nan))
    snapshots = int(event.get("pulse_window_snapshots", 0) or 0)
    if np.isfinite(elapsed):
        caption_key = "speed_caution" if elapsed < 1.0 else "speed_context"
        st.caption(t[caption_key].format(elapsed=elapsed, snapshots=snapshots))


def _render_pulse_quality_badge(frame: pd.DataFrame, event: pd.Series, t: dict) -> dict:
    diagnostics = _pulse_quality_diagnostics(frame, event, t)
    status = diagnostics["status"]
    label = {
        "good": t["quality_good"],
        "review": t["quality_review"],
        "danger": t["quality_danger"],
    }[status]
    help_text = {
        "good": t["quality_good_help"],
        "review": t["quality_review_help"],
        "danger": t["quality_danger_help"],
    }[status]
    reason_text = (
        "; ".join(diagnostics["reasons"]) if diagnostics["reasons"] else help_text
    )

    message = f"**{label}.** {help_text}  \n{t['quality_reason_prefix']}: {reason_text}"
    if status == "good":
        st.success(message)
    elif status == "danger":
        st.error(message)
    else:
        st.warning(message)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t["quality_spread"], f"{diagnostics['spread_ticks']:.2f} ticks")
    c2.metric(t["quality_volume"], _format_sigma(diagnostics["volume_z"]))
    c3.metric(t["quality_elapsed"], f"{diagnostics['elapsed']:.6f}s")
    c4.metric(t["quality_session"], f"{diagnostics['session_age']:.1f}s")
    return diagnostics


def _infer_pulse_family(diagnostics: dict) -> str:
    if diagnostics.get("session_age", 0.0) < 60.0:
        return "opening_pulse"
    if diagnostics.get("spread_ticks", 0.0) > 3.0:
        return "liquidity_shock"
    if diagnostics.get("status") == "good":
        return "clean_intraday_pulse"
    return "mixed_pulse"


def _post_pulse_outcome_frame(
    frame: pd.DataFrame,
    event: pd.Series,
    horizons_seconds: tuple[int, ...] = (5, 10, 30, 60),
) -> pd.DataFrame:
    event_time = pd.to_datetime(event.get("event_time"))
    if pd.isna(event_time):
        return pd.DataFrame()

    scoped = frame.copy()
    if "symbol" in scoped.columns and "symbol" in event.index:
        scoped = scoped.loc[scoped["symbol"].astype(str).eq(str(event.get("symbol")))]
    if "_session_id" in scoped.columns and "_session_id" in event.index:
        scoped = scoped.loc[scoped["_session_id"].eq(event.get("_session_id"))]
    scoped = scoped.sort_values("datetime").reset_index(drop=True)
    if scoped.empty:
        return pd.DataFrame()

    times = pd.to_datetime(scoped["datetime"]).to_numpy()
    event_price = float(event.get("last_price", np.nan))
    tick_size = float(event.get("tick_size_est", np.nan))
    direction = str(event.get("pulse_direction") or "")
    if not np.isfinite(event_price) or not np.isfinite(tick_size) or tick_size <= 0:
        return pd.DataFrame()

    rows = []
    for horizon in horizons_seconds:
        target_time = event_time + pd.Timedelta(seconds=int(horizon))
        loc = int(np.searchsorted(times, np.datetime64(target_time), side="left"))
        if loc >= len(scoped):
            rows.append(
                {
                    "horizon_seconds": int(horizon),
                    "future_time": pd.NaT,
                    "future_price": np.nan,
                    "future_move_ticks": np.nan,
                    "directional_move_ticks": np.nan,
                    "behavior": "missing",
                }
            )
            continue

        future = scoped.iloc[loc]
        future_price = float(future.get("last_price", np.nan))
        future_move = (
            (future_price - event_price) / tick_size
            if np.isfinite(future_price)
            else np.nan
        )
        if direction == "Down":
            directional_move = -future_move
        elif direction == "Up":
            directional_move = future_move
        else:
            directional_move = np.nan

        if not np.isfinite(directional_move):
            behavior = "missing"
        elif directional_move > 0:
            behavior = "continuation"
        elif directional_move < 0:
            behavior = "fade"
        else:
            behavior = "flat"
        rows.append(
            {
                "horizon_seconds": int(horizon),
                "future_time": future["datetime"],
                "future_price": future_price,
                "future_move_ticks": future_move,
                "directional_move_ticks": directional_move,
                "behavior": behavior,
            }
        )
    return pd.DataFrame(rows)


def _post_pulse_outcomes_for_events(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    horizons_seconds: tuple[int, ...] = (5, 10, 30, 60),
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    event_work = events.copy()
    event_work["event_time"] = pd.to_datetime(event_work["event_time"], errors="coerce")
    event_work = event_work.dropna(subset=["event_time"])
    if event_work.empty:
        return pd.DataFrame()

    frame_work = frame.sort_values("datetime").copy()
    group_keys = (
        ["symbol", "_session_id"]
        if {"symbol", "_session_id"}.issubset(frame_work.columns)
        else ["symbol"]
    )
    rows = []

    for key_value, event_group in event_work.groupby(group_keys, sort=False):
        if not isinstance(key_value, tuple):
            key_value = (key_value,)
        mask = pd.Series(True, index=frame_work.index)
        for key, value in zip(group_keys, key_value):
            mask &= frame_work[key].eq(value)
        scoped = frame_work.loc[mask].sort_values("datetime").reset_index(drop=True)
        if scoped.empty:
            continue

        times = pd.to_datetime(scoped["datetime"]).to_numpy()
        event_times = pd.to_datetime(event_group["event_time"]).to_numpy()
        event_prices = pd.to_numeric(
            event_group["last_price"], errors="coerce"
        ).to_numpy(dtype=float)
        tick_sizes = (
            pd.to_numeric(event_group["tick_size_est"], errors="coerce")
            .replace(0, np.nan)
            .to_numpy(dtype=float)
        )
        directions = event_group["pulse_direction"].astype(str).to_numpy()

        for horizon in horizons_seconds:
            target_times = event_times + np.timedelta64(int(horizon), "s")
            locs = np.searchsorted(times, target_times, side="left")
            valid = locs < len(scoped)
            future_prices = np.full(len(event_group), np.nan)
            if valid.any():
                future_prices[valid] = pd.to_numeric(
                    scoped.iloc[locs[valid]]["last_price"], errors="coerce"
                ).to_numpy(dtype=float)
            future_move = (future_prices - event_prices) / tick_sizes
            directional_move = np.where(
                directions == "Down",
                -future_move,
                np.where(directions == "Up", future_move, np.nan),
            )
            behavior = np.select(
                [directional_move > 0, directional_move < 0, directional_move == 0],
                ["continuation", "fade", "flat"],
                default="missing",
            )
            rows.append(
                pd.DataFrame(
                    {
                        "source_index": event_group.get(
                            "source_index",
                            pd.Series(index=event_group.index, dtype=float),
                        ).to_numpy(),
                        "event_rank": event_group.get(
                            "event_rank",
                            pd.Series(index=event_group.index, dtype=float),
                        ).to_numpy(),
                        "horizon_seconds": int(horizon),
                        "future_move_ticks": future_move,
                        "directional_move_ticks": directional_move,
                        "behavior": behavior,
                    }
                )
            )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _behavior_recommendation(
    outcomes: pd.DataFrame, *, require_population_edge: bool = False
) -> dict:
    valid = (
        outcomes["directional_move_ticks"].replace([np.inf, -np.inf], np.nan).dropna()
        if not outcomes.empty
        else pd.Series(dtype=float)
    )
    if valid.empty:
        return {
            "behavior": "unclear",
            "score": np.nan,
            "continuation_share": np.nan,
            "fade_share": np.nan,
            "samples": 0,
        }
    score = float(valid.mean())
    continuation_share = float((valid > 0).mean())
    fade_share = float((valid < 0).mean())
    continuation_ok = score >= 0.5 and (
        continuation_share >= 0.52 or not require_population_edge
    )
    fade_ok = score <= -0.5 and (fade_share >= 0.52 or not require_population_edge)
    if continuation_ok:
        behavior = "continuation"
    elif fade_ok:
        behavior = "fade"
    else:
        behavior = "unclear"
    return {
        "behavior": behavior,
        "score": score,
        "continuation_share": continuation_share,
        "fade_share": fade_share,
        "samples": int(len(valid)),
    }


def _render_behavior_preview(
    frame: pd.DataFrame, events: pd.DataFrame, event: pd.Series, t: dict
) -> dict:
    outcomes = _post_pulse_outcome_frame(frame, event)
    selected_recommendation = _behavior_recommendation(outcomes)
    asset_outcomes = _post_pulse_outcomes_for_events(frame, events)
    asset_recommendation = _behavior_recommendation(
        asset_outcomes, require_population_edge=True
    )
    st.markdown(f"### {t['behavior_preview_title']}")
    st.caption(t["behavior_preview_caption"])

    label = t["behavior_recommendation_labels"].get(
        asset_recommendation["behavior"],
        asset_recommendation["behavior"],
    )
    score = asset_recommendation["score"]
    if asset_recommendation["behavior"] == "continuation":
        st.success(
            t["behavior_asset_recommendation"].format(
                label=label, score=f"{score:+.2f}"
            )
        )
    elif asset_recommendation["behavior"] == "fade":
        st.warning(
            t["behavior_asset_recommendation"].format(
                label=label, score=f"{score:+.2f}"
            )
        )
    else:
        st.info(t["behavior_asset_recommendation_unclear"])

    if not asset_outcomes.empty:
        summary = _behavior_horizon_summary(asset_outcomes, t)
        if not summary.empty:
            st.dataframe(
                summary.style.format(
                    {
                        t["behavior_continuation_rate"]: "{:.1%}",
                        t["behavior_fade_rate"]: "{:.1%}",
                        t["behavior_directional_move"]: "{:+.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    if outcomes.empty:
        return (
            asset_recommendation
            if asset_recommendation["behavior"] in {"continuation", "fade"}
            else selected_recommendation
        )

    st.caption(t["behavior_selected_caption"])
    display = outcomes.copy()
    display[t["behavior_horizon"]] = display["horizon_seconds"].map(
        lambda value: f"+{int(value)}s"
    )
    display[t["behavior_future_time"]] = (
        pd.to_datetime(display["future_time"]).dt.strftime("%H:%M:%S.%f").str[:-3]
    )
    display[t["behavior_raw_move"]] = display["future_move_ticks"]
    display[t["behavior_directional_move"]] = display["directional_move_ticks"]
    display[t["behavior_label"]] = display["behavior"].map(
        lambda value: t["behavior_outcome_labels"].get(value, value)
    )
    table = display[
        [
            t["behavior_horizon"],
            t["behavior_future_time"],
            t["behavior_raw_move"],
            t["behavior_directional_move"],
            t["behavior_label"],
        ]
    ]

    def row_style(row):
        value = row[t["behavior_directional_move"]]
        if pd.isna(value):
            return [""] * len(row)
        if value > 0:
            return ["background-color: rgba(34, 197, 94, 0.12)"] * len(row)
        if value < 0:
            return ["background-color: rgba(239, 68, 68, 0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        table.style.apply(row_style, axis=1).format(
            {
                t["behavior_raw_move"]: "{:+.2f}",
                t["behavior_directional_move"]: "{:+.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    return (
        asset_recommendation
        if asset_recommendation["behavior"] in {"continuation", "fade"}
        else selected_recommendation
    )


def _behavior_horizon_summary(outcomes: pd.DataFrame, t: dict) -> pd.DataFrame:
    valid = (
        outcomes.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["directional_move_ticks"])
        .copy()
    )
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby("horizon_seconds", sort=True)
    rows = []
    for horizon, group in grouped:
        moves = group["directional_move_ticks"]
        rows.append(
            {
                t["behavior_horizon"]: f"+{int(horizon)}s",
                t["behavior_events"]: int(len(group)),
                t["behavior_continuation_rate"]: float((moves > 0).mean()),
                t["behavior_fade_rate"]: float((moves < 0).mean()),
                t["behavior_directional_move"]: float(moves.mean()),
            }
        )
    return pd.DataFrame(rows)


def _default_seed_rule(event: pd.Series, diagnostics: dict, behavior: str) -> dict:
    status = str(diagnostics.get("status", "review"))
    pulse_family = _infer_pulse_family(diagnostics)
    spread_ticks = float(diagnostics.get("spread_ticks", 0.0) or 0.0)
    min_abs_move = abs(float(event.get("pulse_abs_move_ticks", 1.0) or 1.0))

    if status == "good":
        spread_mode = "controlled"
        max_spread_ticks = max(2.0, min(3.0, spread_ticks + 0.5))
        min_spread_ticks = 0.0
    elif spread_ticks > 3.0:
        spread_mode = "wide"
        max_spread_ticks = 999.0
        min_spread_ticks = max(3.0, spread_ticks * 0.8)
    else:
        spread_mode = "any"
        max_spread_ticks = 999.0
        min_spread_ticks = 0.0

    return {
        "rule_version": "pulse_seed_v1",
        "direction_filter": str(event.get("pulse_direction") or "Any"),
        "behavior": behavior,
        "min_abs_move_ticks": round(max(1.0, min_abs_move), 4),
        "spread_filter_mode": spread_mode,
        "max_spread_ticks": round(float(max_spread_ticks), 4),
        "min_spread_ticks": round(float(min_spread_ticks), 4),
        "min_volume_intensity": 1.0,
        "min_session_age_seconds": 0.0 if pulse_family == "opening_pulse" else 60.0,
    }


def _render_seed_promotion(
    *,
    selected_file: str,
    selected_product: str,
    selected_symbol: str,
    event: pd.Series,
    diagnostics: dict,
    behavior_recommendation: dict,
    t: dict,
) -> None:
    with st.expander(t["seed_promote_title"], expanded=False):
        st.markdown(t["seed_promote_help"])
        if os.environ.get("OQP_MANAGER_DEMO") == "1":
            st.info(t["manager_demo_read_only"])
            return
        behavior_labels = {
            "continuation": t["seed_behavior_continuation"],
            "fade": t["seed_behavior_fade"],
        }
        recommended_behavior = behavior_recommendation.get("behavior")
        default_behavior = (
            recommended_behavior
            if recommended_behavior in {"continuation", "fade"}
            else ("fade" if diagnostics.get("status") == "danger" else "continuation")
        )
        behavior = st.segmented_control(
            t["seed_behavior"],
            ["continuation", "fade"],
            default=default_behavior,
            format_func=lambda key: behavior_labels.get(key, key),
            key=f"pulse_seed_behavior_{int(event.get('source_index', 0))}",
        )
        if behavior is None:
            behavior = "continuation"

        pulse_family = _infer_pulse_family(diagnostics)
        default_name = (
            f"{selected_symbol} {pulse_family} "
            f"{event.get('pulse_direction', 'Any')} {behavior}"
        )
        seed_name = st.text_input(
            t["seed_name"],
            value=default_name,
            key=f"pulse_seed_name_{int(event.get('source_index', 0))}",
        )
        rule = _default_seed_rule(event, diagnostics, behavior)
        st.json(rule, expanded=False)

        if st.button(
            t["seed_save"], key=f"pulse_seed_save_{int(event.get('source_index', 0))}"
        ):
            payload = {
                "name": seed_name,
                "source_file": selected_file,
                "product": selected_product,
                "symbol": selected_symbol,
                "event_time": str(
                    pd.to_datetime(event.get("event_time", event.get("datetime", "")))
                ),
                "pulse_direction": str(event.get("pulse_direction") or ""),
                "behavior": behavior,
                "quality_status": str(diagnostics.get("status") or ""),
                "pulse_family": pulse_family,
                "rule": rule,
                "example": _event_example_payload(event, diagnostics),
            }
            seed_id = save_hypothesis_seed(DB_PATH, payload)
            st.success(t["seed_saved"].format(seed_id=seed_id))


def _event_example_payload(event: pd.Series, diagnostics: dict) -> dict:
    keys = [
        "source_index",
        "event_rank",
        "event_time",
        "pulse_direction",
        "pulse_abs_move_ticks",
        "pulse_net_move_ticks",
        "pulse_velocity_ticks_per_sec",
        "pulse_volume_delta",
        "pulse_spread_mean",
        "pulse_book_imbalance_mean",
        "pulse_flow_imbalance",
        "tick_size_est",
    ]
    example = {key: _json_safe(event.get(key)) for key in keys if key in event.index}
    example["diagnostics"] = {
        key: _json_safe(value) for key, value in diagnostics.items() if key != "reasons"
    }
    example["reasons"] = diagnostics.get("reasons", [])
    return example


def _json_safe(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def _pulse_quality_diagnostics(frame: pd.DataFrame, event: pd.Series, t: dict) -> dict:
    tick_size = float(event.get("tick_size_est", np.nan))
    spread_raw = float(event.get("pulse_spread_mean", np.nan))
    spread_ticks = (
        spread_raw / tick_size
        if np.isfinite(spread_raw) and np.isfinite(tick_size) and tick_size > 0
        else np.nan
    )
    move_value = abs(float(event.get("pulse_abs_move_ticks", np.nan)))
    volume_value = float(event.get("pulse_volume_delta", np.nan))
    spread_value = float(event.get("pulse_spread_mean", np.nan))
    elapsed = float(event.get("pulse_window_seconds", np.nan))
    session_age = _event_session_age_seconds(frame, event)

    move_z = _z_score(frame, "pulse_abs_move_ticks", move_value, absolute=True)
    volume_z = _z_score(frame, "pulse_volume_delta", volume_value)
    spread_z = _z_score(frame, "pulse_spread_mean", spread_value)

    dangerous = []
    supportive = []
    caution = []
    if np.isfinite(spread_ticks):
        if spread_ticks > 3.0:
            dangerous.append(("wide_spread", spread_ticks))
        elif spread_ticks <= 2.0:
            supportive.append("good_spread")
        else:
            caution.append("spread")
    if np.isfinite(spread_z) and spread_z > 2.0:
        dangerous.append(("extreme_spread", spread_z))
    if np.isfinite(elapsed) and elapsed < 0.01:
        dangerous.append(("tiny_elapsed", elapsed))
    if np.isfinite(session_age) and session_age < 60.0:
        dangerous.append(("open", session_age))
    if np.isfinite(volume_z):
        if volume_z >= 1.0:
            supportive.append("good_volume")
        elif volume_z < 0.0:
            caution.append("low_volume")
    if np.isfinite(move_z) and move_z >= 2.0:
        supportive.append("good_move")

    if dangerous:
        status = "danger"
    elif {"good_move", "good_volume", "good_spread"}.issubset(set(supportive)):
        status = "good"
    else:
        status = "review"

    return {
        "status": status,
        "spread_ticks": spread_ticks if np.isfinite(spread_ticks) else 0.0,
        "volume_z": volume_z,
        "move_z": move_z,
        "spread_z": spread_z,
        "elapsed": elapsed if np.isfinite(elapsed) else 0.0,
        "session_age": session_age if np.isfinite(session_age) else 0.0,
        "reasons": _quality_reason_text(
            dangerous=dangerous,
            supportive=supportive,
            caution=caution,
            spread_ticks=spread_ticks,
            move_z=move_z,
            volume_z=volume_z,
            spread_z=spread_z,
            elapsed=elapsed,
            session_age=session_age,
            t=t,
        ),
    }


def _quality_reason_text(
    *,
    dangerous: list,
    supportive: list[str],
    caution: list[str],
    spread_ticks: float,
    move_z: float,
    volume_z: float,
    spread_z: float,
    elapsed: float,
    session_age: float,
    t: dict,
) -> list[str]:
    reasons = []
    for kind, _ in dangerous:
        if kind == "wide_spread":
            reasons.append(
                t["quality_reason_wide_spread"].format(spread_ticks=spread_ticks)
            )
        elif kind == "extreme_spread":
            reasons.append(t["quality_reason_extreme_spread"].format(spread_z=spread_z))
        elif kind == "tiny_elapsed":
            reasons.append(t["quality_reason_tiny_elapsed"].format(elapsed=elapsed))
        elif kind == "open":
            reasons.append(t["quality_reason_open"].format(session_age=session_age))
    if not reasons:
        if "good_move" in supportive:
            reasons.append(t["quality_reason_good_move"].format(move_z=move_z))
        if "good_volume" in supportive:
            reasons.append(t["quality_reason_good_volume"].format(volume_z=volume_z))
        if "good_spread" in supportive:
            reasons.append(
                t["quality_reason_good_spread"].format(spread_ticks=spread_ticks)
            )
        if "low_volume" in caution:
            reasons.append(t["quality_reason_low_volume"].format(volume_z=volume_z))
    return reasons


def _event_session_age_seconds(frame: pd.DataFrame, event: pd.Series) -> float:
    try:
        symbol = event.get("symbol")
        session_id = event.get("_session_id")
        event_time = pd.to_datetime(event.get("event_time"))
        scoped = frame.loc[
            frame["symbol"].astype(str).eq(str(symbol))
            & frame["_session_id"].eq(session_id)
        ]
        if scoped.empty:
            return np.nan
        return float(
            (event_time - pd.to_datetime(scoped["datetime"].min())).total_seconds()
        )
    except Exception:
        return np.nan


def _z_score(
    frame: pd.DataFrame, column: str, value: float, *, absolute: bool = False
) -> float:
    background = (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if absolute:
        background = background.abs()
    std = float(background.std(ddof=0)) if len(background) > 1 else np.nan
    mean = float(background.mean()) if len(background) else np.nan
    return (value - mean) / std if np.isfinite(std) and std > 1e-12 else np.nan


def _format_sigma(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:+.2f}σ"


def _format_fingerprint_value(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def _render_cross_asset(
    files: list[dict], window_seconds: float, percentile: float, tpl: str, t: dict
):
    st.markdown(f"### {t['cross_asset']}")
    st.caption(t["cross_asset_help"])
    product_files = _dedupe_product_files(
        [item for item in files if item.get("product_hint")]
    )
    if not product_files:
        st.info(t["cross_asset_empty"])
        return

    manifest = "|".join(
        f"{item['path']}:{item['mtime']}:{item['size_mb']}" for item in product_files
    )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": f"{PULSE_CACHE_TYPE}_cross_asset",
        "source_file": manifest,
        "window_seconds": float(window_seconds),
        "percentile": float(percentile),
    }
    cache_key = make_cache_key(payload)
    cached = load_cached_dataframe(DB_PATH, cache_key, BASE_DIR)
    if cached is not None:
        result, metadata = cached
        st.caption(
            f"{t['cache']}: {t['cache_hit']} | {t['rows_label']}={len(result):,} | "
            f"{t['elapsed']}={float(metadata.get('elapsed_seconds') or 0.0):.2f}s"
        )
    else:
        result = None

    if st.button(t["cross_asset_run"], width="stretch"):
        progress = st.progress(0, text=f"0% - {t['cross_asset_loading']}")

        def compute() -> pd.DataFrame:
            rows = []
            total = len(product_files)
            for idx, item in enumerate(product_files, start=1):
                pct = int((idx - 1) / total * 100)
                progress.progress(
                    pct,
                    text=f"{pct}% - {t['cross_asset_loading']} {item['product_hint']}",
                )
                try:
                    rows.append(
                        summarize_pulse_file(
                            _resolve(item["path"]),
                            product=item["product_hint"],
                            window_seconds=window_seconds,
                            percentile=percentile,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "asset": item["product_hint"],
                            "source_file": item["path"],
                            "error": str(exc),
                        }
                    )
            out = pd.DataFrame(rows)
            for column in [
                "pulses_per_trading_hour",
                "p99_move_ticks",
                "avg_top20_pulse_ticks",
            ]:
                if column not in out.columns:
                    out[column] = np.nan
            return out.sort_values(
                ["pulses_per_trading_hour", "p99_move_ticks", "avg_top20_pulse_ticks"],
                ascending=[False, False, False],
                na_position="last",
            )

        cache_result = get_or_compute_dataframe(
            db_path=DB_PATH,
            logs_dir=LOGS_DIR,
            base_dir=BASE_DIR,
            cache_key=cache_key,
            metadata={
                "cache_type": f"{PULSE_CACHE_TYPE}_cross_asset",
                "source_file": manifest,
                "source_mtime": max(float(item["mtime"]) for item in product_files),
                "product": "ALL",
                "symbol": "MAIN",
                "hypothesis": "",
                "threshold_mode": "directionless_percentile",
                "window": int(round(window_seconds * 1000)),
                "min_success_ticks": float(percentile),
                "horizon_set": "",
                "backend": "python_numpy",
            },
            compute_fn=compute,
        )
        result = cache_result.data
        progress.progress(100, text=f"100% - {t['loading_done']}")
        st.success(
            f"{t['cache']}: {t['cache_saved']} | {t['rows_label']}={len(result):,} | "
            f"{t['elapsed']}={cache_result.elapsed_seconds:.2f}s"
        )

    if result is None or result.empty:
        return

    chart_df = result.dropna(subset=["pulses_per_trading_hour"]).sort_values(
        "pulses_per_trading_hour", ascending=True
    )
    if not chart_df.empty:
        chart_df = chart_df.copy()
        chart_df["asset_label"] = chart_df["asset"].map(
            lambda asset: _product_label(asset, t)
        )
        chart_df["asset_name"] = chart_df["asset"].map(
            lambda asset: _product_name(asset, t)
        )
        fig = go.Figure(
            go.Bar(
                x=chart_df["pulses_per_trading_hour"],
                y=chart_df["asset_label"],
                orientation="h",
                marker=dict(
                    color=chart_df["p99_move_ticks"],
                    colorscale="Blues",
                    colorbar=dict(title=t["p99_ticks"]),
                ),
                customdata=np.stack(
                    [
                        chart_df["asset"],
                        chart_df["asset_name"],
                        chart_df["main_contract"],
                        chart_df["p99_move_ticks"],
                        chart_df["avg_top20_pulse_ticks"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"{t['asset']}=%{{customdata[0]}}<br>{t['asset_name']}=%{{customdata[1]}}<br>"
                    f"{t['main_short']}=%{{customdata[2]}}<br>{t['pulses_hour_short']}=%{{x:.2f}}<br>"
                    f"p99=%{{customdata[3]:.2f}}<br>{t['top20_avg']}=%{{customdata[4]:.2f}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template=tpl,
            height=max(360, 30 * len(chart_df)),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=t["axis_pulses_per_hour"],
            yaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")
    display_result = _format_cross_asset_summary(result, t)
    st.dataframe(
        display_result.style.format(
            {
                t["events"]: "{:.0f}",
                t["pulses_hour_short"]: "{:.2f}",
                "p95": "{:.2f}",
                "p99": "{:.2f}",
                "p99.5": "{:.2f}",
                t["threshold"]: "{:.2f}",
                t["top20_avg"]: "{:.2f}",
                t["rows"]: "{:.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander(t["cross_asset_columns_help_title"], expanded=False):
        st.markdown(t["cross_asset_columns_help"])


def _render_cross_asset_zone_summary(files: list[dict], window_seconds: float, t: dict):
    st.markdown(f"### {t['cross_zone_title']}")
    st.caption(t["cross_zone_help"])
    product_files = _dedupe_product_files(
        [item for item in files if item.get("product_hint")]
    )
    if not product_files:
        st.info(t["cross_asset_empty"])
        return

    manifest = "|".join(
        f"{item['path']}:{item['mtime']}:{item['size_mb']}" for item in product_files
    )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": f"{PULSE_CACHE_TYPE}_cross_asset_zones",
        "source_file": manifest,
        "window_seconds": float(window_seconds),
    }
    cache_key = make_cache_key(payload)
    result = None
    cached = load_cached_dataframe(DB_PATH, cache_key, BASE_DIR)
    if cached is not None:
        result, metadata = cached
        st.caption(
            f"{t['cache']}: {t['cache_hit']} | {t['rows_label']}={len(result):,} | "
            f"{t['elapsed']}={float(metadata.get('elapsed_seconds') or 0.0):.2f}s"
        )

    if st.button(
        t["cross_zone_run"], width="stretch", key="pulse_cross_zone_run"
    ):
        progress = st.progress(0, text=f"0% - {t['cross_zone_loading']}")

        def compute() -> pd.DataFrame:
            frames = []
            total = len(product_files)
            for idx, item in enumerate(product_files, start=1):
                pct = int((idx - 1) / total * 100)
                progress.progress(
                    pct,
                    text=f"{pct}% - {t['cross_zone_loading']} {item['product_hint']}",
                )
                try:
                    zones = summarize_pulse_zones_file(
                        _resolve(item["path"]),
                        product=item["product_hint"],
                        window_seconds=window_seconds,
                    )
                    if not zones.empty:
                        frames.append(zones)
                except Exception as exc:
                    st.warning(f"{item['product_hint']}: {exc}")
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        cache_result = get_or_compute_dataframe(
            db_path=DB_PATH,
            logs_dir=LOGS_DIR,
            base_dir=BASE_DIR,
            cache_key=cache_key,
            metadata={
                "cache_type": f"{PULSE_CACHE_TYPE}_cross_asset_zones",
                "source_file": manifest,
                "source_mtime": max(float(item["mtime"]) for item in product_files),
                "product": "ALL",
                "symbol": "MAIN",
                "hypothesis": "",
                "threshold_mode": "directionless_percentile_zones",
                "window": int(round(window_seconds * 1000)),
                "min_success_ticks": 0.0,
                "horizon_set": "",
                "backend": "python_numpy",
            },
            compute_fn=compute,
        )
        result = cache_result.data
        progress.progress(100, text=f"100% - {t['loading_done']}")
        st.success(
            f"{t['cache']}: {t['cache_saved']} | {t['rows_label']}={len(result):,} | "
            f"{t['elapsed']}={cache_result.elapsed_seconds:.2f}s"
        )

    if result is None or result.empty:
        st.info(t["cross_zone_empty"])
        return

    display = _format_cross_zone_summary(result, t)
    st.dataframe(
        _style_cross_zone_summary(display, t),
        width="stretch",
        hide_index=True,
    )
    st.caption(t["zone_note"])
    with st.expander(t["cross_zone_columns_help_title"], expanded=False):
        st.markdown(t["cross_zone_columns_help"])


def main():
    init_global_ui_state()
    apply_global_style()
    render_global_controls_in_sidebar()
    lang = st.session_state.get("lang", "EN")
    if lang == "CN":
        lang = "ZH"
    t = TEXT.get(lang, TEXT["EN"])
    lab_t = TICK_LAB_TEXT.get(lang, TICK_LAB_TEXT["EN"])
    tpl = get_plotly_template(st.session_state.get("theme_mode", "LIGHT"))

    st.title(t["title"])
    st.caption(t["subtitle"])
    _render_page_workflow(t)

    section_labels = {
        "ranker": t["section_asset_ranker"],
        "universe": t["section_contract_universe"],
        "single": t["section_pulse_discovery"],
        "cross": t["section_cross_summary"],
    }
    selected_section = _render_nav(section_labels, t)
    _render_section_context(selected_section, t)

    if selected_section == "ranker":
        render_asset_download_ranker(
            project_root=REPO_ROOT,
            t=lab_t,
            tpl=tpl,
            key_prefix="pulse_asset_rank",
        )
        return

    tick_files = _discover_tick_files()
    if not tick_files:
        st.warning(t["no_tick_files"])
    file_options = [item["path"] for item in tick_files] or [DEFAULT_TICK_FILE]
    labels = {item["path"]: item["label"] for item in tick_files}
    selected_file = st.selectbox(
        t["file"],
        file_options,
        format_func=lambda path: labels.get(path, path),
        key="pulse_file",
    )
    path = _resolve(selected_file)
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    mtime = os.path.getmtime(path)
    progress = st.progress(0, text=f"0% - {t['loading_summary']}")
    summary = _contract_summary(path, mtime)
    progress.progress(25, text=f"25% - {t['loading_summary']}")
    symbols = sorted(summary["symbol"].astype(str).dropna().unique())
    if summary.empty or not symbols:
        st.error(t["cross_asset_empty"])
        st.stop()
    default_symbol = (
        summary.sort_values("positive_volume_delta", ascending=False)["symbol"]
        .astype(str)
        .iloc[0]
    )

    scope_cols = st.columns(
        [1.2, 0.8] if selected_section == "universe" else [1.2, 0.7, 0.7, 0.7]
    )
    with scope_cols[0]:
        selected_symbol = st.selectbox(
            t["symbol"],
            symbols,
            index=symbols.index(default_symbol) if default_symbol in symbols else 0,
            key="pulse_symbol",
        )

    symbol_row = summary.loc[summary["symbol"].astype(str).eq(selected_symbol)]
    selected_product = (
        str(symbol_row["product"].iloc[0])
        if not symbol_row.empty
        else _product_hint(selected_file)
    )

    if selected_section == "universe":
        raw_scope = _tick_scope(path, mtime, selected_product, selected_symbol)
        progress.progress(100, text=f"100% - {t['loading_done']}")
        st.markdown(f"### {t['contract_universe_title']}")
        st.caption(t["contract_universe_help"])
        schema_df = pd.DataFrame(
            {
                "column": raw_scope.columns,
                "dtype": [str(raw_scope[col].dtype) for col in raw_scope.columns],
            }
        )
        st.markdown(f"#### {lab_t['schema']}")
        st.dataframe(schema_df, width="stretch", hide_index=True)
        _render_raw_row_viewer(raw_scope, selected_symbol, lab_t)
        _render_contract_health(summary, tpl, lab_t)
        st.dataframe(summary, width="stretch", hide_index=True)
        return

    with scope_cols[1]:
        window_seconds = st.slider(
            t["window"],
            2,
            60,
            10,
            1,
            format=f"%d {t['seconds']}",
            key="pulse_window_seconds",
        )
    with scope_cols[2]:
        percentile_pct = st.slider(
            t["percentile"], 90.0, 99.9, 99.0, 0.1, key="pulse_percentile_pct"
        )
    with scope_cols[3]:
        top_n = st.slider(t["top"], 10, 200, 50, 10, key="pulse_top_n")

    percentile = float(percentile_pct) / 100.0
    collapse_gap = float(window_seconds)
    _render_scope_story(
        selected_file=selected_file,
        selected_symbol=selected_symbol,
        window_seconds=float(window_seconds),
        percentile_pct=float(percentile_pct),
        t=t,
    )

    if selected_section == "cross":
        progress.progress(100, text=f"100% - {t['loading_done']}")
        _render_cross_asset_zone_summary(tick_files, float(window_seconds), t)
        st.divider()
        _render_cross_asset(tick_files, float(window_seconds), percentile, tpl, t)
        return

    progress.progress(45, text=f"45% - {t['loading_frame']}")
    frame = _pulse_frame(
        path,
        mtime,
        selected_product,
        selected_symbol,
        float(window_seconds),
        percentile,
        collapse_gap,
    )
    progress.progress(75, text=f"75% - {t['loading_events']}")
    events = _pulse_events(
        path,
        mtime,
        selected_product,
        selected_symbol,
        float(window_seconds),
        percentile,
        collapse_gap,
    )
    progress.progress(100, text=f"100% - {t['loading_done']}")
    st.success(f"100% - {t['loading_done']}")

    event_threshold = (
        float(
            pd.to_numeric(
                events.get("event_threshold_ticks", pd.Series(dtype=float)),
                errors="coerce",
            )
            .dropna()
            .iloc[0]
        )
        if not events.empty and "event_threshold_ticks" in events.columns
        else float(frame["pulse_abs_move_ticks"].quantile(percentile))
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["rows"], f"{len(frame):,}")
    m2.metric(t["events"], f"{len(events):,}")
    m3.metric(t["threshold"], f"{event_threshold:.2f} ticks")
    m4.metric(t["p99"], f"{frame['pulse_abs_move_ticks'].quantile(0.99):.2f} ticks")
    _render_pulse_validity_callout(
        frame=frame, events=events, event_threshold=event_threshold, t=t
    )
    with st.expander(t["metric_help_title"], expanded=False):
        st.markdown(t["metric_help"])
    st.caption(
        f"{t['cache']}: "
        f"{t['cache_frame']}={t['cache_hit'] if frame.attrs.get('pulse_cache_hit') else t['cache_computed']}, "
        f"{t['cache_events']}={t['cache_hit'] if events.attrs.get('pulse_cache_hit') else t['cache_computed']}"
    )

    selected_source_index = (
        int(events.iloc[0]["source_index"]) if not events.empty else None
    )
    event_options: list[int] = []
    labels_by_index: dict[int, str] = {}
    if not events.empty:
        event_options = events.head(top_n)["source_index"].astype(int).tolist()
        labels_by_index = {
            int(row.source_index): (
                f"#{int(row.event_rank)} | {pd.to_datetime(row.event_time).strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{row.pulse_direction} | {row.pulse_abs_move_ticks:.2f} ticks"
            )
            for row in events.head(top_n).itertuples()
        }
        if st.session_state.get("pulse_selected_event") not in event_options:
            st.session_state["pulse_selected_event"] = event_options[0]
        selected_source_index = int(st.session_state["pulse_selected_event"])

    st.markdown(f"### {t['distribution']}")
    st.caption(t["distribution_help"])
    st.plotly_chart(
        _severity_ladder_figure(frame, selected_source_index, tpl, t),
        width="stretch",
    )
    zone_summary = _zone_summary(frame, t)
    if not zone_summary.empty:
        st.dataframe(
            zone_summary.style.format(
                {
                    t["zone_share"]: "{:.2%}",
                    t["zone_avg"]: "{:.2f}",
                    t["zone_max"]: "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(t["zone_note"])
        with st.expander(t["severity_help_title"], expanded=False):
            st.markdown(t["severity_help"])

    st.markdown(f"### {t['event_table']}")
    if events.empty:
        st.info(t["no_events"])
    else:
        st.dataframe(
            _event_table(events, top_n, t).style.format(
                {
                    t["abs_move"]: "{:.2f}",
                    t["net_move"]: "{:.2f}",
                    t["velocity"]: "{:.3f}",
                    t["volume"]: "{:,.0f}",
                    t["spread"]: "{:.3f}",
                    t["book"]: "{:.3f}",
                    t["flow"]: "{:.3f}",
                    t["range"]: "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        with st.expander(t["event_columns_help_title"], expanded=False):
            st.markdown(t["event_columns_help"])

        st.markdown(f"### {t['inspector']}")
        selected_source_index = st.selectbox(
            t["event_select"],
            event_options,
            format_func=lambda idx: labels_by_index.get(int(idx), str(idx)),
            key="pulse_selected_event",
        )
        event = events.loc[
            events["source_index"].astype(int).eq(int(selected_source_index))
        ].iloc[0]
        st.markdown(f"### {t['fingerprint']}")
        _fingerprint(frame, event, t)
        diagnostics = _render_pulse_quality_badge(frame, event, t)
        with st.expander(t["event_reading_help_title"], expanded=False):
            st.markdown(f"**{t['fingerprint_help_title']}**")
            st.markdown(t["fingerprint_help"])
            st.markdown(f"**{t['inspector_help_title']}**")
            st.markdown(t["inspector_help"])
        st.plotly_chart(
            _event_inspector(frame, event, float(window_seconds), tpl, t),
            width="stretch",
        )
        behavior_recommendation = _render_behavior_preview(frame, events, event, t)
        _render_seed_promotion(
            selected_file=selected_file,
            selected_product=selected_product,
            selected_symbol=selected_symbol,
            event=event,
            diagnostics=diagnostics,
            behavior_recommendation=behavior_recommendation,
            t=t,
        )


if __name__ == "__main__":
    main()
