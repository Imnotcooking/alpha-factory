from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from oqp.data.runtime_paths import discover_futures_cn_tick_files

from .asset_ranker import (
    discover_daily_universe_files,
    discover_intraday_universe_files,
    filter_ranked_assets,
    load_daily_universe,
    rank_daily_asset_volatility,
)


RANK_CACHE_VERSION = "asset_download_ranker_v2_runtime_root"
LENS_CONFIG = {
    "daily": {
        "discover": discover_daily_universe_files,
        "annualization": 252,
        "lookback_options": [63, 126, 252, 504, 756],
        "default_lookback": 252,
        "min_obs_min": 20,
        "default_min_obs": 120,
        "min_obs_step": 10,
    },
    "minute": {
        "discover": discover_intraday_universe_files,
        "annualization": 252 * 240,
        "lookback_options": [1_200, 5_000, 20_000, 60_000, 120_000],
        "default_lookback": 20_000,
        "min_obs_min": 100,
        "default_min_obs": 5_000,
        "min_obs_step": 100,
    },
}


@st.cache_data(show_spinner=False)
def compute_asset_rankings(
    path: str,
    mtime: float,
    lookback_days: int,
    min_observations: int,
    annualization: int,
    cache_version: str,
) -> tuple[pd.DataFrame, dict]:
    bar_df = load_daily_universe(path)
    ranked = rank_daily_asset_volatility(
        bar_df,
        lookback_days=lookback_days,
        min_observations=min_observations,
        annualization=annualization,
    )
    return ranked, _dataset_profile(bar_df)


def _dataset_profile(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "rows": 0,
            "assets": 0,
            "date_min": "",
            "date_max": "",
            "timeframe": "Unknown",
        }

    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    assets = frame["ticker"].astype(str).nunique() if "ticker" in frame.columns else 0
    if dates.empty:
        return {
            "rows": int(len(frame)),
            "assets": int(assets),
            "date_min": "",
            "date_max": "",
            "timeframe": "Unknown",
        }

    unique_dates = dates.drop_duplicates().sort_values()
    deltas = unique_dates.diff().dropna()
    positive_deltas = deltas[deltas > pd.Timedelta(0)]
    median_delta = positive_deltas.median() if not positive_deltas.empty else pd.NaT

    return {
        "rows": int(len(frame)),
        "assets": int(assets),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "timeframe": _format_timeframe(median_delta),
    }


def _format_timeframe(delta: pd.Timedelta | pd.NaT) -> str:
    if pd.isna(delta):
        return "Unknown"
    seconds = int(round(delta.total_seconds()))
    if seconds <= 0:
        return "Unknown"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    if seconds < 86_400 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 86_400 and seconds % 86_400 == 0:
        return f"{seconds // 86_400}d"
    return str(delta)


def _format_date_range(profile: dict) -> str:
    start = profile.get("date_min")
    end = profile.get("date_max")
    if pd.isna(start) or pd.isna(end):
        return "-"
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.time() == pd.Timestamp("00:00").time() and end_ts.time() == pd.Timestamp("00:00").time():
        return f"{start_ts.date()} -> {end_ts.date()}"
    return f"{start_ts:%Y-%m-%d %H:%M} -> {end_ts:%Y-%m-%d %H:%M}"


def _format_profile_timestamp(value) -> str:
    if pd.isna(value):
        return "-"
    timestamp = pd.Timestamp(value)
    if timestamp.time() == pd.Timestamp("00:00").time():
        return f"{timestamp:%Y-%m-%d}"
    return f"{timestamp:%Y-%m-%d %H:%M}"


def render_asset_download_ranker(
    *,
    project_root: str,
    t: dict,
    tpl: str,
    key_prefix: str = "asset_rank",
) -> None:
    lens_keys = ["daily", "minute", "tick"]
    lens_labels = t["pattern_lens_options"]
    lens_cols = st.columns([0.8, 2.2])
    with lens_cols[0]:
        selected_lens = st.selectbox(
            t["pattern_lens"],
            lens_keys,
            format_func=lambda key: lens_labels[key],
            key=f"{key_prefix}_pattern_lens",
        )
    with lens_cols[1]:
        st.info(t["pattern_lens_purpose"][selected_lens])

    if selected_lens == "tick":
        _render_tick_lens(project_root=project_root, t=t)
        return

    lens_config = LENS_CONFIG[selected_lens]
    source_files = lens_config["discover"](project_root)
    if not source_files:
        st.warning(t["asset_rank_empty"])
        return

    file_options = [item["path"] for item in source_files]
    label_by_path = {item["path"]: item["label"] for item in source_files}
    control_cols = st.columns([1.6, 0.8, 0.8, 0.8])
    with control_cols[0]:
        selected_source_file = st.selectbox(
            t["asset_source_file"][selected_lens],
            file_options,
            index=0,
            format_func=lambda path: label_by_path.get(path, path),
            key=f"{key_prefix}_{selected_lens}_file",
        )
    with control_cols[1]:
        lookback_days = st.select_slider(
            t["asset_lookback_by_lens"][selected_lens],
            options=lens_config["lookback_options"],
            value=lens_config["default_lookback"],
            key=f"{key_prefix}_{selected_lens}_lookback",
        )
    with control_cols[2]:
        min_observations = st.slider(
            t["asset_min_obs_by_lens"][selected_lens],
            min_value=lens_config["min_obs_min"],
            max_value=max(lens_config["min_obs_min"], lookback_days),
            value=min(lens_config["default_min_obs"], lookback_days),
            step=lens_config["min_obs_step"],
            key=f"{key_prefix}_{selected_lens}_min_obs",
        )
    with control_cols[3]:
        top_n = st.slider(
            t["asset_top_n"],
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            key=f"{key_prefix}_top_n",
        )

    st.caption(t["pattern_lens_data_note"][selected_lens])

    source_path = _resolve_path(project_root, selected_source_file)
    if not os.path.exists(source_path):
        st.error(f"File not found: {source_path}")
        return

    try:
        ranked, profile = compute_asset_rankings(
            source_path,
            os.path.getmtime(source_path),
            int(lookback_days),
            int(min_observations),
            int(lens_config["annualization"]),
            f"{RANK_CACHE_VERSION}_{selected_lens}",
        )
    except Exception as exc:
        st.error(f"{t['asset_rank_empty']}: {exc}")
        return

    if ranked.empty:
        st.info(t["asset_rank_empty"])
        return

    filtered = filter_ranked_assets(
        ranked,
        top_n=int(top_n),
    )

    metric_labels = t["asset_rank_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(metric_labels["ranked_assets"], f"{len(ranked):,}")
    m2.metric(metric_labels["top_vol"], f"{ranked['recent_ann_vol'].max():.1%}")
    m3.metric(metric_labels["median_vol"], f"{ranked['recent_ann_vol'].median():.1%}")
    top_name = ranked.sort_values("download_priority_score", ascending=False).iloc[0]["base_symbol"]
    m4.metric(metric_labels["top_priority"], str(top_name))

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(t["dataset_timeframe"], str(profile["timeframe"]))
    p2.metric(t["dataset_start"], _format_profile_timestamp(profile["date_min"]))
    p3.metric(t["dataset_end"], _format_profile_timestamp(profile["date_max"]))
    p4.metric(t["dataset_assets"], f"{int(profile['assets']):,}")
    p5.metric(t["dataset_rows"], f"{int(profile['rows']):,}")

    if filtered.empty:
        st.info(t["asset_rank_empty"])
        return

    st.markdown(f"#### {t['asset_rank_chart_by_lens'][selected_lens]}")
    chart_df = filtered.sort_values("recent_ann_vol", ascending=True).copy()
    duplicated_labels = chart_df["base_symbol"].duplicated(keep=False)
    chart_df["asset_label"] = chart_df["base_symbol"].astype(str)
    chart_df.loc[duplicated_labels, "asset_label"] = (
        chart_df.loc[duplicated_labels, "base_symbol"].astype(str)
        + " | "
        + chart_df.loc[duplicated_labels, "ticker"].astype(str)
    )
    fig = px.bar(
        chart_df,
        x="recent_ann_vol",
        y="asset_label",
        color="sector",
        orientation="h",
        hover_data={
            "ticker": True,
            "base_symbol": True,
            "download_priority_score": ":.2f",
            "avg_daily_volume": ":,.0f",
            "avg_oi": ":,.0f",
            "avg_intraday_range_pct": ":.2%",
            "coverage": ":.0%",
        },
        template=tpl,
    )
    fig.update_layout(
        height=max(420, min(900, 28 * len(chart_df))),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_tickformat=".0%",
        xaxis_title=t["asset_sort_options"]["recent_ann_vol"],
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "download_priority_rank",
        "vol_rank",
        "ticker",
        "base_symbol",
        "sector",
        "recent_ann_vol",
        "full_ann_vol",
        "avg_intraday_range_pct",
        "avg_abs_return_bps",
        "avg_daily_volume",
        "avg_oi",
        "coverage",
        "valid_return_days",
        "last_date",
        "download_symbol_hint",
    ]
    display = filtered[display_cols].copy()
    display["last_date"] = pd.to_datetime(display["last_date"]).dt.date
    st.markdown(f"#### {t['asset_rank_table_by_lens'][selected_lens]}")
    st.dataframe(
        display.style.format(
            {
                "recent_ann_vol": "{:.1%}",
                "full_ann_vol": "{:.1%}",
                "avg_intraday_range_pct": "{:.2%}",
                "avg_abs_return_bps": "{:.1f}",
                "avg_daily_volume": "{:,.0f}",
                "avg_oi": "{:,.0f}",
                "coverage": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    if os.environ.get("OQP_MANAGER_DEMO") == "1":
        st.caption(t.get("manager_demo_download_hidden", "Download disabled in manager demo mode."))
    else:
        st.download_button(
            t["asset_rank_download"],
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"pattern_lab_{selected_lens}_rank.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_tick_lens(*, project_root: str, t: dict) -> None:
    tick_files = discover_futures_cn_tick_files()
    st.markdown(f"#### {t['tick_lens_title']}")
    st.caption(t["tick_lens_caption"])
    if not tick_files:
        st.warning(t["asset_rank_empty"])
        return

    rows = []
    for path in tick_files:
        stat = path.stat()
        try:
            display_path = str(path.relative_to(project_root))
        except ValueError:
            display_path = str(path)
        rows.append(
            {
                t["tick_file"]: display_path,
                t["tick_size_mb"]: stat.st_size / 1024 / 1024,
                t["tick_modified"]: pd.to_datetime(stat.st_mtime, unit="s").strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )

    m1, m2 = st.columns(2)
    m1.metric(t["tick_file_count"], f"{len(rows):,}")
    m2.metric(t["pattern_lens"], t["pattern_lens_options"]["tick"])
    st.info(t["tick_lens_route"])
    st.dataframe(
        pd.DataFrame(rows).style.format({t["tick_size_mb"]: "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )


def _resolve_path(project_root: str, path_text: str) -> str:
    if os.path.isabs(path_text):
        return path_text
    return os.path.join(project_root, path_text)
