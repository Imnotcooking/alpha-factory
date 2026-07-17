"""Shared Streamlit theme and localization helpers for Ops dashboards."""

from __future__ import annotations

from html import escape
from itertools import count
from re import sub
from typing import Any

import streamlit as st

from oqp.ui.translations import normalize_language, ops_text


LANGUAGE_LABELS = ("English", "中文")
_CHART_COUNTER = count()


def tr(language: str, english: str, chinese: str) -> str:
    """Return localized text for the active dashboard language."""

    return chinese if normalize_language(language) == "zh" else english


def language_selector(*, key: str = "oqp_language_label") -> str:
    """Render the shared language selector and return ``en`` or ``zh``."""

    label = st.sidebar.selectbox(
        ops_text("en", "language_label"),
        LANGUAGE_LABELS,
        index=0,
        key=key,
    )
    active_language = "zh" if label == "中文" else "en"
    st.sidebar.caption(ops_text(active_language, "language_caption"))
    return active_language


def page_header(
    *,
    title: str,
    subtitle: str,
    language: str,
    title_zh: str | None = None,
    subtitle_zh: str | None = None,
    kicker: str = "Alpha Factory",
    kicker_zh: str = "Alpha Factory",
) -> None:
    """Render a compact styled page header."""

    active_title = tr(language, title, title_zh or title)
    active_subtitle = tr(language, subtitle, subtitle_zh or subtitle)
    active_kicker = tr(language, kicker, kicker_zh)
    st.markdown(
        f"""
        <section class="oqp-page-header">
            <div class="oqp-kicker">{escape(active_kicker)}</div>
            <h1>{escape(active_title)}</h1>
            <p>{escape(active_subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    *,
    subtitle: str = "",
    accent: str = "teal",
) -> None:
    """Render a compact section heading inside a bento panel."""

    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="oqp-section-heading oqp-accent-{escape(accent)}">
            <div>
                <h2>{escape(title)}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_tiles(items: list[tuple[str, str, str]]) -> None:
    """Render compact navigation tiles for Streamlit multipage routes."""

    tiles = []
    for label, href, meta in items:
        tiles.append(
            '<a class="oqp-nav-tile" '
            f'href="{escape(href)}" target="_self">'
            f'<span>{escape(label)}</span>'
            f'<small>{escape(meta)}</small>'
            "</a>"
        )
    st.markdown(
        f'<div class="oqp-nav-grid">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


def _token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "empty"


def render_dark_table(
    frame: Any,
    *,
    empty_message: str = "No rows available.",
    max_height_px: int | None = None,
) -> None:
    """Render a compact dark table for bento-style dashboard summaries."""

    import pandas as pd

    frame = pd.DataFrame() if frame is None else pd.DataFrame(frame).copy()
    if frame.empty:
        st.info(empty_message)
        return

    columns = list(frame.columns)
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    rows = []
    chip_columns = {"priority", "state", "status", "mode", "gate", "fit", "conviction"}
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            column_key = str(column).strip().lower()
            if column_key in chip_columns:
                chip = _token(text)
                cells.append(
                    f'<td><span class="oqp-chip oqp-chip-{chip}">{escape(text)}</span></td>'
                )
            else:
                cell_class = "oqp-long-cell" if len(text) > 72 else ""
                cells.append(f'<td class="{cell_class}">{escape(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    height_style = (
        f"max-height: {int(max_height_px)}px; overflow: auto;"
        if max_height_px
        else ""
    )
    st.markdown(
        f"""
        <div class="oqp-dark-table-wrap" style="{height_style}">
            <table class="oqp-dark-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_frame(data: Any):
    import pandas as pd

    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data).copy()


def _chart_x(frame: Any) -> tuple[Any, str]:
    index_name = frame.index.name or "index"
    return frame.index, str(index_name).replace("_", " ").title()


def _ensure_plotly_template() -> bool:
    """Register the reusable Ops Plotly template if Plotly is available."""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return False

    if "oqp_dark" not in pio.templates:
        pio.templates["oqp_dark"] = go.layout.Template(
            layout={
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(8,12,18,0.82)",
                "font": {"color": "#DDE7F3", "family": "Inter, sans-serif"},
                "colorway": [
                    "#2DD4BF",
                    "#60A5FA",
                    "#F59E0B",
                    "#FB7185",
                    "#A78BFA",
                    "#22C55E",
                ],
                "xaxis": {
                    "gridcolor": "rgba(89,111,139,0.16)",
                    "zerolinecolor": "rgba(89,111,139,0.24)",
                    "linecolor": "rgba(89,111,139,0.18)",
                },
                "yaxis": {
                    "gridcolor": "rgba(89,111,139,0.16)",
                    "zerolinecolor": "rgba(89,111,139,0.24)",
                    "linecolor": "rgba(89,111,139,0.18)",
                },
                "legend": {
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(89,111,139,0.14)",
                },
                "margin": {"l": 20, "r": 20, "t": 36, "b": 24},
            }
        )
    pio.templates.default = "oqp_dark"
    return True


def style_dark_plotly(
    fig: Any,
    *,
    height: int | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    barmode: str | None = None,
    hovermode: str | None = "x unified",
    margin: dict[str, int] | None = None,
) -> Any:
    """Apply the Ops dark visual system to a Plotly figure."""

    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(8,12,18,0.92)",
        "font": {"color": "#DDE7F3", "family": "Inter, sans-serif"},
        "margin": margin or dict(t=18, b=28, l=30, r=24),
        "xaxis": {
            "title": xaxis_title,
            "gridcolor": "rgba(89,111,139,0.14)",
            "zerolinecolor": "rgba(89,111,139,0.20)",
            "linecolor": "rgba(89,111,139,0.20)",
            "tickfont": {"color": "#94A3B8"},
            "title_font": {"color": "#94A3B8"},
        },
        "yaxis": {
            "title": yaxis_title,
            "gridcolor": "rgba(89,111,139,0.14)",
            "zerolinecolor": "rgba(89,111,139,0.20)",
            "linecolor": "rgba(89,111,139,0.20)",
            "tickfont": {"color": "#94A3B8"},
            "title_font": {"color": "#94A3B8"},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(89,111,139,0.10)",
            "font": {"color": "#DDE7F3"},
        },
    }
    if _ensure_plotly_template():
        layout["template"] = "oqp_dark"
    if height is not None:
        layout["height"] = height
    if barmode:
        layout["barmode"] = barmode
    if hovermode:
        layout["hovermode"] = hovermode

    fig.update_layout(**layout)
    try:
        fig.update_xaxes(showline=False, ticks="", automargin=True)
        fig.update_yaxes(showline=False, ticks="", automargin=True)
        fig.update_scenes(
            xaxis={
                "backgroundcolor": "rgba(8,12,18,0.92)",
                "gridcolor": "rgba(89,111,139,0.18)",
                "zerolinecolor": "rgba(89,111,139,0.22)",
                "color": "#DDE7F3",
            },
            yaxis={
                "backgroundcolor": "rgba(8,12,18,0.92)",
                "gridcolor": "rgba(89,111,139,0.18)",
                "zerolinecolor": "rgba(89,111,139,0.22)",
                "color": "#DDE7F3",
            },
            zaxis={
                "backgroundcolor": "rgba(8,12,18,0.92)",
                "gridcolor": "rgba(89,111,139,0.18)",
                "zerolinecolor": "rgba(89,111,139,0.22)",
                "color": "#DDE7F3",
            },
        )
    except Exception:
        pass
    return fig


def render_dark_line_chart(
    data: Any,
    *,
    empty_message: str = "No chart data available.",
    height: int = 320,
    yaxis_title: str | None = None,
) -> None:
    """Render a dark Plotly line chart from a dataframe-like object."""

    frame = _chart_frame(data)
    if frame.empty:
        st.info(empty_message)
        return

    import plotly.graph_objects as go

    x_values, x_title = _chart_x(frame)
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame[column],
                mode="lines",
                name=str(column).replace("_", " ").title(),
                line={"width": 2.2},
            )
        )
    style_dark_plotly(
        fig,
        height=height,
        xaxis_title=x_title if x_title != "Index" else None,
        yaxis_title=yaxis_title,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
        key=f"oqp_dark_line_{next(_CHART_COUNTER)}",
    )


def render_dark_bar_chart(
    data: Any,
    *,
    empty_message: str = "No chart data available.",
    height: int = 320,
    barmode: str = "group",
    yaxis_title: str | None = None,
) -> None:
    """Render a dark Plotly bar chart from a dataframe-like object."""

    frame = _chart_frame(data)
    if frame.empty:
        st.info(empty_message)
        return

    import plotly.graph_objects as go

    x_values, x_title = _chart_x(frame)
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=frame[column],
                name=str(column).replace("_", " ").title(),
            )
        )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    style_dark_plotly(
        fig,
        height=height,
        barmode=barmode,
        xaxis_title=x_title if x_title != "Index" else None,
        yaxis_title=yaxis_title,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
        key=f"oqp_dark_bar_{next(_CHART_COUNTER)}",
    )


def render_dark_pie_chart(
    data: Any,
    *,
    names: str,
    values: str,
    empty_message: str = "No chart data available.",
    height: int = 320,
    hole: float = 0.56,
) -> None:
    """Render a dark Plotly donut chart from a dataframe-like object."""

    frame = _chart_frame(data)
    if frame.empty or names not in frame.columns or values not in frame.columns:
        st.info(empty_message)
        return

    import pandas as pd
    import plotly.graph_objects as go

    chart = frame.copy()
    chart[values] = pd.to_numeric(chart[values], errors="coerce").fillna(0.0)
    chart = chart.loc[chart[values].abs() > 0].copy()
    if chart.empty:
        st.info(empty_message)
        return

    fig = go.Figure(
        go.Pie(
            labels=chart[names],
            values=chart[values],
            hole=hole,
            sort=False,
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            marker={
                "line": {"color": "rgba(8,12,18,0.92)", "width": 2},
            },
            hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )
    style_dark_plotly(
        fig,
        height=height,
        margin=dict(t=18, b=40, l=10, r=10),
        hovermode=None,
    )
    fig.update_traces(textfont={"color": "#F8FAFC", "size": 13})
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11, "color": "#94A3B8"},
        },
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
        key=f"oqp_dark_pie_{next(_CHART_COUNTER)}",
    )


def render_dark_area_chart(
    data: Any,
    *,
    empty_message: str = "No chart data available.",
    height: int = 320,
    yaxis_title: str | None = None,
) -> None:
    """Render a dark stacked area chart from a dataframe-like object."""

    frame = _chart_frame(data)
    if frame.empty:
        st.info(empty_message)
        return

    import plotly.graph_objects as go

    x_values, x_title = _chart_x(frame)
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame[column],
                mode="lines",
                name=str(column).replace("_", " ").title(),
                stackgroup="one",
                line={"width": 1.8},
            )
        )
    style_dark_plotly(
        fig,
        height=height,
        xaxis_title=x_title if x_title != "Index" else None,
        yaxis_title=yaxis_title,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
        key=f"oqp_dark_area_{next(_CHART_COUNTER)}",
    )


def apply_ops_theme() -> None:
    """Apply the Ops dashboard visual system.

    This adapts the legacy Middle Office dark terminal look into one reusable
    style layer, with less brittle selectors and a quieter operations palette.
    """

    _ensure_plotly_template()

    st.markdown(
        """
        <style>
            :root {
                --oqp-bg: #080A0F;
                --oqp-panel: #101722;
                --oqp-panel-2: #0D131D;
                --oqp-panel-soft: rgba(13, 19, 29, 0.82);
                --oqp-panel-softer: rgba(255, 255, 255, 0.025);
                --oqp-border: rgba(89, 111, 139, 0.045);
                --oqp-border-strong: rgba(45, 212, 191, 0.10);
                --oqp-text: #E5E7EB;
                --oqp-muted: #94A3B8;
                --oqp-subtle: #64748B;
                --oqp-teal: #2DD4BF;
                --oqp-blue: #60A5FA;
                --oqp-amber: #F59E0B;
                --oqp-rose: #FB7185;
                --oqp-green: #22C55E;
            }

            html, body, [class*="st-"] {
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
                letter-spacing: 0 !important;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% -10%, rgba(45, 212, 191, 0.10), transparent 22rem),
                    radial-gradient(circle at 94% 0%, rgba(59, 130, 246, 0.11), transparent 26rem),
                    linear-gradient(180deg, #070A0F 0%, #0A1018 52%, #070A0F 100%) !important;
                color: var(--oqp-text) !important;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"] {
                background-color: transparent !important;
            }

            [data-testid="stMainBlockContainer"] {
                padding-top: 1.6rem !important;
                padding-bottom: 3rem !important;
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(17, 23, 34, 0.98), rgba(8, 10, 15, 0.98)) !important;
                border-right: 1px solid rgba(89, 111, 139, 0.06) !important;
            }

            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div {
                color: var(--oqp-muted) !important;
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
                border-radius: 8px !important;
                padding: 0.18rem !important;
                background: rgba(7, 12, 19, 0.88) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.018),
                    0 10px 22px rgba(0, 0, 0, 0.10);
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] *,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] *::before,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] *::after,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {
                border: 0 !important;
                border-color: transparent !important;
                box-shadow: none !important;
                outline: 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radiogroup"] {
                gap: 0.2rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radio"] {
                min-height: 1.85rem !important;
                border-radius: 7px !important;
                border: 0 !important;
                color: #AFC0D4 !important;
                background: transparent !important;
                font-weight: 700 !important;
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button:focus,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button:focus-visible,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label:focus-within,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radio"]:focus,
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radio"]:focus-visible {
                outline: 0 !important;
                box-shadow:
                    inset 0 0 0 1px rgba(45, 212, 191, 0.18) !important;
                border-color: transparent !important;
            }

            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-pressed="true"],
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-selected="true"],
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label:has(input:checked),
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [aria-checked="true"],
            [data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
                color: #E8FFF9 !important;
                background:
                    linear-gradient(135deg, rgba(45, 212, 191, 0.28), rgba(96, 165, 250, 0.18)) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.035),
                    0 8px 18px rgba(45, 212, 191, 0.10) !important;
            }

            .oqp-page-header {
                border-radius: 8px;
                background:
                    linear-gradient(90deg, rgba(45, 212, 191, 0.10), transparent 22rem),
                    linear-gradient(135deg, rgba(13, 22, 31, 0.98), rgba(8, 12, 18, 0.92));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.018),
                    0 18px 48px rgba(0, 0, 0, 0.28);
                padding: 1.18rem 1.2rem 1.05rem 1.2rem;
                margin-bottom: 1.25rem;
            }

            .oqp-page-header .oqp-kicker {
                color: var(--oqp-teal);
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em !important;
                margin-bottom: 0.4rem;
            }

            .oqp-page-header h1 {
                color: #F8FAFC !important;
                font-size: 2rem !important;
                line-height: 1.15 !important;
                margin: 0 !important;
                font-weight: 750 !important;
            }

            .oqp-page-header p {
                color: var(--oqp-muted) !important;
                max-width: 72rem;
                margin: 0.5rem 0 0 0 !important;
                font-size: 0.96rem !important;
            }

            h1, h2, h3, h4 {
                color: #F8FAFC !important;
                font-weight: 700 !important;
                letter-spacing: 0 !important;
            }

            h2, h3 {
                margin-top: 0.7rem !important;
            }

            p, li, label, span {
                color: var(--oqp-text);
            }

            .oqp-section-heading {
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                margin: 0 0 0.95rem 0;
            }

            .oqp-section-heading::before {
                content: "";
                display: block;
                width: 0.28rem;
                min-width: 0.28rem;
                height: 2.15rem;
                border-radius: 999px;
                background: linear-gradient(180deg, #2DD4BF, #60A5FA);
                box-shadow: 0 0 20px rgba(45, 212, 191, 0.18);
            }

            .oqp-section-heading h2 {
                margin: 0 !important;
                font-size: 1.22rem !important;
                line-height: 1.1 !important;
            }

            .oqp-section-heading p {
                color: var(--oqp-muted) !important;
                margin: 0.3rem 0 0 0 !important;
                font-size: 0.86rem !important;
            }

            .oqp-accent-amber::before {
                background: linear-gradient(180deg, #F59E0B, #FBBF24);
                box-shadow: 0 0 20px rgba(245, 158, 11, 0.17);
            }

            .oqp-accent-rose::before {
                background: linear-gradient(180deg, #FB7185, #F59E0B);
                box-shadow: 0 0 20px rgba(251, 113, 133, 0.14);
            }

            .oqp-accent-blue::before {
                background: linear-gradient(180deg, #60A5FA, #A78BFA);
                box-shadow: 0 0 20px rgba(96, 165, 250, 0.16);
            }

            .oqp-accent-green::before {
                background: linear-gradient(180deg, #22C55E, #2DD4BF);
                box-shadow: 0 0 20px rgba(34, 197, 94, 0.13);
            }

            div[data-testid="stMetric"] {
                border-radius: 8px;
                background:
                    linear-gradient(180deg, rgba(18, 27, 40, 0.52), rgba(10, 15, 23, 0.36));
                padding: 0.7rem 0.8rem;
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.010),
                    0 8px 18px rgba(0, 0, 0, 0.08);
                min-height: 5.2rem;
                min-width: 0 !important;
                overflow: hidden !important;
            }

            [data-testid="stMetricLabel"] {
                color: var(--oqp-subtle) !important;
                font-size: 0.76rem !important;
                font-weight: 700 !important;
                text-transform: uppercase;
                letter-spacing: 0.06em !important;
            }

            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
                color: #F8FAFC !important;
                font-size: clamp(0.92rem, 1.12vw, 1.62rem) !important;
                font-weight: 760 !important;
                line-height: 1.12 !important;
                white-space: normal !important;
                overflow: hidden !important;
                text-overflow: clip !important;
                word-break: normal !important;
                overflow-wrap: anywhere !important;
            }

            [data-testid="stMetricValue"] {
                display: block !important;
                max-width: 100% !important;
            }

            [data-testid="stMetricDelta"] {
                color: var(--oqp-teal) !important;
                font-weight: 650 !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                border-radius: 8px;
                overflow: hidden;
                background: rgba(15, 23, 42, 0.42);
            }

            div[data-testid="stPlotlyChart"],
            div[data-testid="stVegaLiteChart"],
            div[data-testid="stDeckGlJsonChart"] {
                border-radius: 8px;
                background:
                    linear-gradient(180deg, rgba(13, 19, 29, 0.82), rgba(8, 12, 18, 0.76));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.014),
                    0 14px 34px rgba(0, 0, 0, 0.18);
                padding: 0.35rem;
                overflow: hidden;
            }

            div[data-testid="stPlotlyChart"] .js-plotly-plot,
            div[data-testid="stPlotlyChart"] .plot-container,
            div[data-testid="stPlotlyChart"] .svg-container,
            div[data-testid="stPlotlyChart"] svg.main-svg {
                background: transparent !important;
            }

            div[data-testid="stAlert"] {
                border-radius: 8px !important;
                border: 0 !important;
                background: rgba(15, 23, 42, 0.68) !important;
            }

            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div {
                color: #DDE7F3 !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid rgba(89, 111, 139, 0.04);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
                color: var(--oqp-muted) !important;
                background: rgba(15, 23, 42, 0.40);
                border: 1px solid transparent;
                height: 2.55rem;
                padding: 0 0.85rem;
            }

            .stTabs [aria-selected="true"] {
                color: #F8FAFC !important;
                border-color: rgba(89, 111, 139, 0.06);
                border-bottom-color: transparent;
                background: rgba(17, 23, 34, 0.92);
            }

            .stButton > button,
            .stFormSubmitButton > button {
                border-radius: 8px !important;
                border: 1px solid rgba(45, 212, 191, 0.24) !important;
                background: linear-gradient(90deg, rgba(20, 184, 166, 0.72), rgba(37, 99, 235, 0.74)) !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                border-color: rgba(245, 158, 11, 0.58) !important;
                box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.12), 0 14px 30px rgba(0, 0, 0, 0.25) !important;
                transform: translateY(-1px);
            }

            .stDownloadButton > button {
                border-radius: 8px !important;
                border: 0 !important;
                background: rgba(255, 255, 255, 0.035) !important;
                color: var(--oqp-text) !important;
            }

            .stTextInput input,
            .stTextArea textarea,
            .stNumberInput input,
            div[data-baseweb="select"] {
                color: #F8FAFC !important;
                -webkit-text-fill-color: #F8FAFC !important;
                background-color: rgba(15, 23, 42, 0.76) !important;
                border-color: var(--oqp-border) !important;
                border-radius: 8px !important;
            }

            div[data-baseweb="select"] span {
                color: #F8FAFC !important;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"],
            [data-testid="stSidebar"] div[data-baseweb="select"] > div {
                color: #DDE7F3 !important;
                -webkit-text-fill-color: #DDE7F3 !important;
                background:
                    linear-gradient(180deg, rgba(14, 21, 32, 0.94), rgba(8, 12, 18, 0.90)) !important;
                border: 0 !important;
                border-radius: 8px !important;
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.018),
                    0 10px 22px rgba(0, 0, 0, 0.10) !important;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] *,
            [data-testid="stSidebar"] div[data-baseweb="select"] input {
                color: #DDE7F3 !important;
                -webkit-text-fill-color: #DDE7F3 !important;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] svg {
                color: #94A3B8 !important;
                fill: #94A3B8 !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border: 0 !important;
                border-radius: 8px !important;
                background:
                    linear-gradient(180deg, rgba(15, 22, 33, 0.56), rgba(8, 12, 18, 0.42)) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.010),
                    0 14px 32px rgba(0, 0, 0, 0.10) !important;
                padding: 0.95rem !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.oqp-accent-teal) {
                background:
                    linear-gradient(135deg, rgba(45, 212, 191, 0.045), transparent 34rem),
                    linear-gradient(180deg, rgba(13, 22, 31, 0.68), rgba(8, 12, 18, 0.52)) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.oqp-accent-blue) {
                background:
                    linear-gradient(135deg, rgba(96, 165, 250, 0.050), transparent 34rem),
                    linear-gradient(180deg, rgba(13, 21, 34, 0.68), rgba(8, 12, 18, 0.52)) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.oqp-accent-amber) {
                background:
                    linear-gradient(135deg, rgba(245, 158, 11, 0.044), transparent 32rem),
                    linear-gradient(180deg, rgba(19, 20, 17, 0.68), rgba(8, 12, 18, 0.52)) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.oqp-accent-rose) {
                background:
                    linear-gradient(135deg, rgba(251, 113, 133, 0.040), transparent 32rem),
                    linear-gradient(180deg, rgba(21, 18, 25, 0.68), rgba(8, 12, 18, 0.52)) !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.oqp-accent-green) {
                background:
                    linear-gradient(135deg, rgba(34, 197, 94, 0.038), transparent 32rem),
                    linear-gradient(180deg, rgba(13, 23, 21, 0.68), rgba(8, 12, 18, 0.52)) !important;
            }

            a[data-testid="stPageLink-NavLink"] {
                display: inline-flex !important;
                width: fit-content !important;
                max-width: 100% !important;
                border: 0 !important;
                border-radius: 8px !important;
                background:
                    linear-gradient(180deg, rgba(17, 24, 36, 0.64), rgba(10, 15, 23, 0.48)) !important;
                color: #DDE7F3 !important;
                font-weight: 700 !important;
                min-height: 2.15rem !important;
                padding: 0.25rem 0.55rem !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.012);
            }

            a[data-testid="stPageLink-NavLink"]:hover {
                border-color: rgba(45, 212, 191, 0.38) !important;
                background:
                    linear-gradient(180deg, rgba(20, 184, 166, 0.16), rgba(17, 24, 39, 0.86)) !important;
                color: #F8FAFC !important;
            }

            .oqp-nav-grid {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 0.55rem;
                margin: 0.2rem 0 0.1rem 0;
            }

            .oqp-nav-tile {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 4.8rem;
                border-radius: 8px;
                padding: 0.7rem 0.76rem;
                background:
                    linear-gradient(150deg, rgba(17, 24, 36, 0.60), rgba(8, 12, 18, 0.42));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.010),
                    0 8px 18px rgba(0, 0, 0, 0.08);
                transition: transform 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
            }

            .oqp-nav-tile span {
                color: #F8FAFC !important;
                font-size: 0.9rem;
                font-weight: 800;
                line-height: 1.2;
            }

            .oqp-nav-tile small {
                color: var(--oqp-muted);
                font-size: 0.72rem;
                line-height: 1.25;
                margin-top: 0.65rem;
            }

            .oqp-nav-tile:hover {
                transform: translateY(-1px);
                background:
                    linear-gradient(150deg, rgba(45, 212, 191, 0.10), rgba(17, 24, 39, 0.60));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.018),
                    0 12px 26px rgba(0, 0, 0, 0.14);
            }

            @media (max-width: 1100px) {
                .oqp-nav-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            .oqp-dark-table-wrap {
                width: 100%;
                border-radius: 8px;
                overflow: hidden;
                background:
                    linear-gradient(180deg, rgba(14, 20, 31, 0.96), rgba(8, 12, 18, 0.92));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.012),
                    0 16px 34px rgba(0, 0, 0, 0.12);
                margin: 0.35rem 0 1rem 0;
            }

            .oqp-dark-table {
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                font-size: 0.82rem;
            }

            .oqp-dark-table,
            .oqp-dark-table th,
            .oqp-dark-table td {
                border-left: 0 !important;
                border-right: 0 !important;
                border-top: 0 !important;
            }

            .oqp-dark-table thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: rgba(15, 23, 35, 0.98);
                color: var(--oqp-subtle);
                font-size: 0.72rem;
                font-weight: 800;
                text-align: left;
                text-transform: uppercase;
                letter-spacing: 0.045em !important;
                padding: 0.7rem 0.75rem;
                border-bottom: 1px solid rgba(89, 111, 139, 0.055) !important;
            }

            .oqp-dark-table tbody tr {
                border-bottom: 1px solid rgba(89, 111, 139, 0.035) !important;
            }

            .oqp-dark-table tbody tr:nth-child(even) {
                background: rgba(255, 255, 255, 0.018);
            }

            .oqp-dark-table tbody tr:hover {
                background: rgba(45, 212, 191, 0.055);
            }

            .oqp-dark-table td {
                color: #DDE7F3;
                padding: 0.64rem 0.75rem;
                vertical-align: top;
                word-break: break-word;
            }

            .oqp-dark-table .oqp-long-cell {
                color: #B6C4D8;
            }

            .oqp-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 3.5rem;
                border-radius: 999px;
                padding: 0.12rem 0.5rem;
                color: #DDE7F3;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: rgba(148, 163, 184, 0.08);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em !important;
            }

            .oqp-chip-fail,
            .oqp-chip-high,
            .oqp-chip-locked,
            .oqp-chip-blocked {
                color: #FFD4DC;
                border-color: rgba(251, 113, 133, 0.28);
                background: rgba(251, 113, 133, 0.12);
            }

            .oqp-chip-warn,
            .oqp-chip-medium,
            .oqp-chip-waiting,
            .oqp-chip-armed {
                color: #FFE9B0;
                border-color: rgba(245, 158, 11, 0.30);
                background: rgba(245, 158, 11, 0.12);
            }

            .oqp-chip-pass,
            .oqp-chip-low,
            .oqp-chip-ready,
            .oqp-chip-active,
            .oqp-chip-open,
            .oqp-chip-clear {
                color: #B9F8E8;
                border-color: rgba(45, 212, 191, 0.28);
                background: rgba(45, 212, 191, 0.11);
            }

            a {
                color: var(--oqp-blue) !important;
                text-decoration: none !important;
            }

            code {
                color: var(--oqp-teal) !important;
                background: rgba(45, 212, 191, 0.08) !important;
                border: 1px solid rgba(45, 212, 191, 0.13) !important;
                border-radius: 6px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
