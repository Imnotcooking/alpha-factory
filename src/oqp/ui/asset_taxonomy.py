"""Dashboard helpers for the shared market-lane taxonomy."""

from __future__ import annotations

from html import escape
from typing import Iterable

import pandas as pd
import streamlit as st

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical
from oqp.ui.streamlit_theme import render_dark_table


DEFAULT_DASHBOARD_LANES = (
    "EQUITY_US",
    "OPTIONS_US",
    "EQUITY_CN",
    "OPTIONS_CN",
    "FUTURES_CN",
)

UI_LANE_METADATA: dict[str, dict[str, str]] = {
    "EQUITY_US": {
        "label": "US Equities",
        "label_zh": "美国股票",
        "provider": "FMP + Yahoo; Massive where available",
        "broker": "IBKR",
        "execution": "IBKR paper/live guardrails",
        "status": "active",
    },
    "OPTIONS_US": {
        "label": "US Options",
        "label_zh": "美国期权",
        "provider": "Massive primary; Yahoo fallback",
        "broker": "IBKR",
        "execution": "IBKR paper/live guardrails",
        "status": "active",
    },
    "EQUITY_CN": {
        "label": "Chinese Equities",
        "label_zh": "中国股票",
        "provider": "Wind/QMT planned",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "status": "planned_qmt",
    },
    "OPTIONS_CN": {
        "label": "Chinese Options",
        "label_zh": "中国期权",
        "provider": "Wind/QMT planned",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "status": "planned_qmt",
    },
    "FUTURES_CN": {
        "label": "Chinese Futures",
        "label_zh": "中国期货",
        "provider": "Local runtime files; Wind/QMT later",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "status": "active_data_planned_execution",
    },
}


def _display_status(value: object) -> str:
    status = str(value or "configured").strip().lower()
    labels = {
        "active": "Active",
        "planned": "Planned",
        "bridge": "Bridge",
        "planned_qmt": "Planned QMT",
        "active_data_planned_execution": "Active Data / Planned Execution",
    }
    return labels.get(status, status.replace("_", " ").title())


def _fallback_taxonomy_frame(asset_classes: Iterable[str]) -> pd.DataFrame:
    rows = []
    for value in asset_classes:
        asset_class = normalize_market_vertical(value)
        meta = ASSET_TAXONOMY.get(asset_class, {})
        lane = UI_LANE_METADATA.get(asset_class, {})
        rows.append(
            {
                "asset_class": asset_class,
                "region": meta.get("region", ""),
                "instrument_family": meta.get("instrument_family", ""),
                "label": lane.get("label", asset_class),
                "label_zh": lane.get("label_zh", lane.get("label", asset_class)),
                "provider": lane.get("provider", "TBD"),
                "broker": lane.get("broker", "TBD"),
                "execution": lane.get("execution", "TBD"),
                "lane_status": lane.get("status", "configured"),
            }
        )
    return pd.DataFrame(rows)


def _taxonomy_frame(asset_classes: Iterable[str]) -> pd.DataFrame:
    """Return taxonomy rows, tolerating stale Streamlit module caches."""

    classes = list(asset_classes)
    try:
        from oqp.data import asset_taxonomy as data_asset_taxonomy

        builder = getattr(data_asset_taxonomy, "taxonomy_frame", None)
        if callable(builder):
            return builder(asset_classes=classes)
    except Exception:
        pass
    return _fallback_taxonomy_frame(classes)


def lane_label(asset_class: str, *, language: str = "en") -> str:
    """Return a human label for a market lane."""

    frame = _taxonomy_frame([asset_class])
    if frame.empty:
        return normalize_market_vertical(asset_class)
    row = frame.iloc[0]
    if str(language).lower().startswith("zh"):
        return str(row.get("label_zh") or row.get("label") or row.get("asset_class"))
    return str(row.get("label") or row.get("asset_class"))


def dashboard_taxonomy_frame(
    lanes: Iterable[str] | None = None,
    *,
    language: str = "en",
) -> pd.DataFrame:
    """Return the compact lane table used by Ops and Research dashboards."""

    raw = _taxonomy_frame(lanes or DEFAULT_DASHBOARD_LANES)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "Lane",
                "Family",
                "Region",
                "Broker / Route",
                "Data Stack",
                "Execution",
                "Status",
            ]
        )
    label_col = "label_zh" if str(language).lower().startswith("zh") else "label"
    out = pd.DataFrame(
        {
            "Lane": raw[label_col].fillna(raw["asset_class"]),
            "Family": raw["instrument_family"].fillna("").astype(str).str.title(),
            "Region": raw["region"].fillna("").astype(str),
            "Broker / Route": raw["broker"].fillna("TBD").astype(str),
            "Data Stack": raw["provider"].fillna("TBD").astype(str),
            "Execution": raw["execution"].fillna("TBD").astype(str),
            "Status": raw["lane_status"].map(_display_status),
        }
    )
    return out


def render_market_lane_overview(
    *,
    language: str = "en",
    lanes: Iterable[str] | None = None,
    title: str = "Asset Taxonomy",
    title_zh: str = "资产分类",
    subtitle: str = (
        "The same market lanes are shared by research, discretionary tools, "
        "paper/live operations, and future QMT execution."
    ),
    subtitle_zh: str = "研究、主观工具、模拟/实盘运营与未来 QMT 执行共用同一套资产线。",
    expanded: bool = False,
) -> None:
    """Render a compact taxonomy panel."""

    active_zh = str(language).lower().startswith("zh")
    heading = title_zh if active_zh else title
    copy = subtitle_zh if active_zh else subtitle
    with st.expander(heading, expanded=expanded):
        st.caption(copy)
        render_dark_table(
            dashboard_taxonomy_frame(lanes, language=language),
            empty_message="No asset taxonomy lanes are configured.",
            max_height_px=260,
        )


def render_market_lane_chips(
    *,
    language: str = "en",
    lanes: Iterable[str] | None = None,
    caption: str | None = None,
) -> None:
    """Render lane chips for pages where a full table would be too heavy."""

    frame = _taxonomy_frame(lanes or DEFAULT_DASHBOARD_LANES)
    if frame.empty:
        return
    label_col = "label_zh" if str(language).lower().startswith("zh") else "label"
    chips = []
    for _, row in frame.iterrows():
        status = str(row.get("lane_status") or "configured").replace("_", " ")
        chips.append(
            f'<span class="oqp-lane-chip" title="{escape(status)}">'
            f'{escape(str(row.get(label_col) or row.get("asset_class")))}'
            f'<small>{escape(str(row.get("broker") or ""))}</small>'
            "</span>"
        )
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(
        f"""
        <style>
        .oqp-lane-chip-wrap {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: stretch;
            margin: 0.45rem 0 1.0rem 0;
        }}
        .oqp-lane-chip {{
            display: inline-flex;
            flex-direction: column;
            justify-content: center;
            min-width: 8.5rem;
            border-radius: 10px;
            padding: 0.54rem 0.72rem;
            color: #f8fafc;
            background:
                linear-gradient(135deg, rgba(45, 212, 191, 0.16), rgba(96, 165, 250, 0.10)),
                rgba(9, 16, 27, 0.92);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.028),
                0 0 0 1px rgba(96, 165, 250, 0.10);
            font-weight: 800;
            line-height: 1.1;
        }}
        .oqp-lane-chip small {{
            margin-top: 0.28rem;
            color: #91a5bf;
            font-size: 0.68rem;
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .oqp-lane-chip-note {{
            color: #94a3b8;
            margin: -0.2rem 0 0.55rem 0;
            font-size: 0.88rem;
        }}
        </style>
        <div class="oqp-lane-chip-note">{caption_html}</div>
        <div class="oqp-lane-chip-wrap">{''.join(chips)}</div>
        """,
        unsafe_allow_html=True,
    )
