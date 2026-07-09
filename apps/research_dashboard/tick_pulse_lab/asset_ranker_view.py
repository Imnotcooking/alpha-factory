from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from .asset_ranker import (
    discover_daily_universe_files,
    filter_ranked_assets,
    load_daily_universe,
    rank_daily_asset_volatility,
)


RANK_CACHE_VERSION = "asset_download_ranker_v2_runtime_root"


@st.cache_data(show_spinner=False)
def compute_asset_rankings(
    path: str,
    mtime: float,
    lookback_days: int,
    min_observations: int,
    cache_version: str,
) -> pd.DataFrame:
    daily_df = load_daily_universe(path)
    return rank_daily_asset_volatility(
        daily_df,
        lookback_days=lookback_days,
        min_observations=min_observations,
    )


def render_asset_download_ranker(
    *,
    project_root: str,
    t: dict,
    tpl: str,
    key_prefix: str = "asset_rank",
) -> None:
    daily_files = discover_daily_universe_files(project_root)
    if not daily_files:
        st.warning(t["asset_rank_empty"])
        return

    file_options = [item["path"] for item in daily_files]
    label_by_path = {item["path"]: item["label"] for item in daily_files}
    control_cols = st.columns([1.6, 0.8, 0.8, 0.8])
    with control_cols[0]:
        selected_daily_file = st.selectbox(
            t["asset_daily_file"],
            file_options,
            index=0,
            format_func=lambda path: label_by_path.get(path, path),
            key=f"{key_prefix}_daily_file",
        )
    with control_cols[1]:
        lookback_days = st.select_slider(
            t["asset_lookback"],
            options=[63, 126, 252, 504, 756],
            value=252,
            key=f"{key_prefix}_lookback",
        )
    with control_cols[2]:
        min_observations = st.slider(
            t["asset_min_obs"],
            min_value=20,
            max_value=max(20, lookback_days),
            value=min(120, lookback_days),
            step=10,
            key=f"{key_prefix}_min_obs",
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

    daily_path = _resolve_path(project_root, selected_daily_file)
    if not os.path.exists(daily_path):
        st.error(f"File not found: {daily_path}")
        return

    try:
        ranked = compute_asset_rankings(
            daily_path,
            os.path.getmtime(daily_path),
            int(lookback_days),
            int(min_observations),
            RANK_CACHE_VERSION,
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

    if filtered.empty:
        st.info(t["asset_rank_empty"])
        return

    st.markdown(f"#### {t['asset_rank_chart']}")
    chart_df = filtered.sort_values("recent_ann_vol", ascending=True).copy()
    fig = px.bar(
        chart_df,
        x="recent_ann_vol",
        y="base_symbol",
        color="sector",
        orientation="h",
        hover_data={
            "ticker": True,
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
    st.markdown(f"#### {t['asset_rank_table']}")
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
            file_name="tick_pulse_asset_download_rank.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _resolve_path(project_root: str, path_text: str) -> str:
    if os.path.isabs(path_text):
        return path_text
    return os.path.join(project_root, path_text)
