from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from oqp.research.tick_pulse import valid_tick_mask

from .constants import (
    BREAKDOWN_MAX_BOOK_IMBALANCE,
    BREAKDOWN_MIN_VOLUME_INTENSITY,
    MAX_PLOT_POINTS,
    MIN_BOOK_IMBALANCE,
    MIN_FLOW_IMBALANCE,
    MIN_VOLUME_INTENSITY,
    OUTCOME_COLORS,
    RESEARCH_SWEEP_HORIZONS,
    RTV_MIN_FAST_MOVE_TICKS,
)
from .engine import (
    _build_research_sweep,
    _feature_group_keys,
    _is_bearish_hypothesis,
    _is_relative_velocity_hypothesis,
    _pct_text,
    format_research_sweep_display,
)
from .research_copy import _research_readout_text

def _outcome_label(value: str, t: dict) -> str:
    return t.get("outcome_labels", {}).get(str(value), str(value))


def _direction_label(value: str, t: dict) -> str:
    return t.get("direction_labels", {}).get(str(value), str(value))


def _is_seed_hypothesis(hypothesis: str) -> bool:
    return str(hypothesis).startswith("seed:")


def _format_row_value(value) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):,.6f}".rstrip("0").rstrip(".")
    return str(value)


def _plot_series_with_session_breaks(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column].copy()
    if "_session_break" in df.columns:
        values = values.mask(df["_session_break"].fillna(False))
    return values


def _active_trading_rangebreaks() -> list[dict]:
    return [
        dict(bounds=["sat", "mon"]),
        dict(bounds=[2.0, 9.5], pattern="hour"),
        dict(bounds=[10.25, 10.5], pattern="hour"),
        dict(bounds=[11.5, 13.5], pattern="hour"),
        dict(bounds=[15.0, 21.0], pattern="hour"),
    ]


def _downsample_for_plot(df: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    positions = np.linspace(0, len(df) - 1, max_points)
    positions = np.unique(np.round(positions).astype(int))
    return df.iloc[positions].copy()


def _select_visible_candidates(candidates: pd.DataFrame, max_events: int) -> pd.DataFrame:
    ordered = candidates.sort_values("datetime").reset_index(drop=True)
    if len(ordered) <= max_events:
        return ordered

    positions = np.linspace(0, len(ordered) - 1, max_events)
    positions = np.unique(np.round(positions).astype(int))
    return ordered.iloc[positions].reset_index(drop=True)


def _build_event_distribution(candidates: pd.DataFrame, t: dict) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    grouped = candidates.assign(event_date=candidates["datetime"].dt.date).groupby("event_date", sort=True)
    rows = []
    for event_date, group in grouped:
        correct = int(group["is_correct"].sum())
        total = int(len(group))
        rows.append(
            {
                t["audit_date"]: str(event_date),
                t["events"]: total,
                t["correct_moves"]: correct,
                t["accuracy"]: f"{correct / total:.1%}" if total else "N/A",
                t["avg_move"]: f"{group['future_move_ticks'].mean():.2f}",
                t["audit_first_event"]: group["datetime"].min().strftime("%H:%M:%S"),
                t["audit_last_event"]: group["datetime"].max().strftime("%H:%M:%S"),
            }
        )
    return pd.DataFrame(rows)


def _render_calculation_audit(
    raw_scope: pd.DataFrame,
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    visible_candidates: pd.DataFrame,
    t: dict,
):
    st.markdown(f"#### {t['calculation_audit']}")
    st.caption(t["audit_note"])

    raw_rows = int(len(raw_scope))
    invalid_rows = int((~valid_tick_mask(raw_scope)).sum()) if raw_rows else 0
    if "_session_id" in features.columns and not features.empty:
        session_count = int(features.groupby(_feature_group_keys(features), sort=False).ngroups)
    else:
        session_count = 0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric(t["raw_rows"], f"{raw_rows:,}")
    a2.metric(t["valid_feature_rows"], f"{len(features):,}")
    a3.metric(t["dropped_rows"], f"{invalid_rows:,}")
    a4.metric(t["session_count"], f"{session_count:,}")

    st.caption(
        t["marker_caption"].format(
            shown=len(visible_candidates),
            total=len(candidates),
        )
    )
    st.caption(t["event_episode_note"])

    distribution = _build_event_distribution(candidates, t)
    if not distribution.empty:
        st.markdown(f"##### {t['event_distribution']}")
        st.dataframe(distribution, width="stretch", hide_index=True)


def _render_raw_row_viewer(raw_df: pd.DataFrame, selected_symbol: str, t: dict):
    st.markdown(f"#### {t['row_viewer']}")
    symbol_rows = raw_df[raw_df["symbol"] == selected_symbol].sort_values("datetime").copy()
    if symbol_rows.empty:
        st.info(t["row_empty"].format(symbol=selected_symbol))
        return

    symbol_rows.insert(0, "_global_row", symbol_rows.index)
    symbol_rows = symbol_rows.reset_index(drop=True)
    modes = t["row_modes"]
    cols = st.columns([1, 1])
    with cols[0]:
        row_mode = st.selectbox(t["row_mode"], modes, index=0, key=f"raw_row_mode_{selected_symbol}")

    if row_mode == modes[0]:
        row_idx = 0
    elif row_mode == modes[1]:
        row_idx = len(symbol_rows) // 2
    elif row_mode == modes[2]:
        row_idx = len(symbol_rows) - 1
    else:
        with cols[1]:
            row_idx = st.number_input(
                t["row_index"],
                min_value=0,
                max_value=len(symbol_rows) - 1,
                value=0,
                step=1,
                key=f"raw_row_index_{selected_symbol}",
            )
    row_idx = int(row_idx)
    row = symbol_rows.iloc[row_idx]
    global_row = int(row["_global_row"])

    st.caption(
        t["row_position"].format(
            symbol=selected_symbol,
            row=row_idx,
            last=len(symbol_rows) - 1,
            source=global_row,
        )
    )

    original_columns = [col for col in raw_df.columns if col in symbol_rows.columns]
    one_row = row[original_columns].to_frame().T
    st.dataframe(one_row, width="stretch", hide_index=True)

    meanings = t.get("row_meanings", {})
    detail_rows = [
        {
            t["field"]: col,
            t["value"]: _format_row_value(row[col]),
            t["dtype"]: str(raw_df[col].dtype),
            t["meaning"]: meanings.get(col, ""),
        }
        for col in original_columns
    ]
    st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)


def _render_research_readout(
    features: pd.DataFrame,
    min_success_ticks: float,
    lang: str,
    t: dict,
    thresholds_by_hypothesis: dict[str, dict[str, float]] | None = None,
    sweep_raw: pd.DataFrame | None = None,
):
    st.markdown(f"#### {t['research_title']}")
    st.markdown(_research_readout_text(lang, min_success_ticks))
    st.markdown(f"#### {t['sweep_title']}")
    if sweep_raw is None:
        _, display = _build_research_sweep(features, min_success_ticks, t, thresholds_by_hypothesis)
    else:
        if isinstance(sweep_raw, tuple) and sweep_raw and isinstance(sweep_raw[0], pd.DataFrame):
            sweep_raw = sweep_raw[0]
        display = format_research_sweep_display(sweep_raw, t)
    st.dataframe(display, width="stretch", hide_index=True)


def _render_contract_health(summary: pd.DataFrame, tpl: str, t: dict):
    st.markdown(f"#### {t['contract_health']}")
    plot_df = summary.sort_values("positive_volume_delta", ascending=True)
    fig = px.bar(
        plot_df,
        x="positive_volume_delta",
        y="symbol",
        orientation="h",
        color="median_spread",
        color_continuous_scale="Viridis_r",
        hover_data=["rows", "oi_first", "oi_last", "tick_size_est", "max_spread"],
        labels={
            "positive_volume_delta": t["chart_positive_volume_delta"],
            "median_spread": t["chart_median_spread"],
            "symbol": t["symbol"],
            "rows": t["rows"],
        },
    )
    fig.update_layout(
        template=tpl,
        height=330,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=t["chart_positive_volume_delta"],
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def _render_event_map(features: pd.DataFrame, candidates: pd.DataFrame, hypothesis: str, tpl: str, t: dict):
    st.markdown(f"#### {t['event_map']}")
    st.caption(t["trading_hours_compressed_note"])
    plot_features = _downsample_for_plot(features)
    if len(plot_features) < len(features):
        st.caption(t["plot_downsample_note"].format(points=len(plot_features)))
    flow_threshold = -MIN_FLOW_IMBALANCE
    volume_threshold = BREAKDOWN_MIN_VOLUME_INTENSITY if hypothesis == "bearish_breakdown" else MIN_VOLUME_INTENSITY
    if hypothesis == "bearish_breakdown":
        book_threshold = BREAKDOWN_MAX_BOOK_IMBALANCE
    elif hypothesis == "bearish":
        book_threshold = -MIN_BOOK_IMBALANCE
    else:
        book_threshold = MIN_BOOK_IMBALANCE
    correct_symbol = "triangle-down" if _is_bearish_hypothesis(hypothesis) else "triangle-up"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.52, 0.24, 0.24],
    )
    fig.add_trace(
        go.Scatter(
            x=plot_features["datetime"],
            y=_plot_series_with_session_breaks(plot_features, "last_price"),
            mode="lines",
            name=t["chart_last_price"],
            line=dict(color="#90A4AE", width=1.2),
        ),
        row=1,
        col=1,
    )

    for outcome, group in candidates.groupby("outcome"):
        if _is_relative_velocity_hypothesis(hypothesis):
            marker_symbol = np.where(
                group["outcome"].eq("Correct") & group["expected_direction"].eq("Down"),
                "triangle-down",
                np.where(
                    group["outcome"].eq("Correct") & group["expected_direction"].eq("Up"),
                    "triangle-up",
                    "x",
                ),
            )
            custom_cols = [
                "future_move_ticks",
                "rtv_fast_move_ticks",
                "rtv_threshold_ticks",
                "rtv_threshold_ratio",
                "rtv_direction",
                "expected_direction",
            ]
            hovertemplate = (
                f"{t['chart_time']}=%{{x}}<br>"
                f"{t['chart_price']}=%{{y:.2f}}<br>"
                f"{t['chart_future_move']}=%{{customdata[0]:.1f}} {t['ticks']}<br>"
                f"{t['chart_rtv_fast']}=%{{customdata[1]:.1f}} {t['ticks']}<br>"
                f"{t['chart_rtv_threshold']}=%{{customdata[2]:.1f}} {t['ticks']}<br>"
                f"{t['chart_rtv_percentile']}=%{{customdata[3]:.2f}}<br>"
                f"{t['chart_rtv_direction']}=%{{customdata[4]}}<br>"
                f"{t['expected']}=%{{customdata[5]}}<extra></extra>"
            )
        elif _is_seed_hypothesis(hypothesis):
            marker_symbol = np.where(
                group["outcome"].eq("Correct") & group["expected_direction"].eq("Down"),
                "triangle-down",
                np.where(
                    group["outcome"].eq("Correct") & group["expected_direction"].eq("Up"),
                    "triangle-up",
                    "x",
                ),
            )
            custom_cols = [
                "future_move_ticks",
                "seed_abs_move_ticks",
                "seed_spread_ticks",
                "volume_intensity",
                "seed_direction",
                "expected_direction",
            ]
            hovertemplate = (
                f"{t['chart_time']}=%{{x}}<br>"
                f"{t['chart_price']}=%{{y:.2f}}<br>"
                f"{t['chart_future_move']}=%{{customdata[0]:.1f}} {t['ticks']}<br>"
                f"seed move=%{{customdata[1]:.1f}} {t['ticks']}<br>"
                f"spread=%{{customdata[2]:.1f}} {t['ticks']}<br>"
                f"{t['chart_volume_intensity']}=%{{customdata[3]:.1f}}<br>"
                f"seed direction=%{{customdata[4]}}<br>"
                f"{t['expected']}=%{{customdata[5]}}<extra></extra>"
            )
        else:
            marker_symbol = correct_symbol if outcome == "Correct" else "x"
            custom_cols = [
                "future_move_ticks",
                "flow_imbalance",
                "book_imbalance",
                "volume_intensity",
                "rolling_mid_move_ticks",
                "price_shock",
            ]
            hovertemplate = (
                f"{t['chart_time']}=%{{x}}<br>"
                f"{t['chart_price']}=%{{y:.2f}}<br>"
                f"{t['chart_future_move']}=%{{customdata[0]:.1f}} {t['ticks']}<br>"
                f"{t['chart_flow']}=%{{customdata[1]:.2f}}<br>"
                f"{t['chart_book']}=%{{customdata[2]:.2f}}<br>"
                f"{t['chart_volume_intensity']}=%{{customdata[3]:.1f}}<br>"
                f"{t['chart_rolling_mid_move']}=%{{customdata[4]:.1f}} {t['ticks']}<br>"
                f"price_shock=%{{customdata[5]:.1f}}<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=group["datetime"],
                y=group["last_price"],
                mode="markers",
                name=_outcome_label(outcome, t),
                marker=dict(
                    color=OUTCOME_COLORS.get(outcome, "#B0BEC5"),
                    size=12,
                    symbol=marker_symbol,
                    line=dict(width=0.5, color="#263238"),
                ),
                customdata=group[custom_cols],
                hovertemplate=hovertemplate,
            ),
            row=1,
            col=1,
        )

    if _is_relative_velocity_hypothesis(hypothesis):
        fig.add_trace(
            go.Scatter(
                x=plot_features["datetime"],
                y=_plot_series_with_session_breaks(plot_features, "rtv_fast_move_ticks"),
                mode="lines",
                name=t["chart_rtv_fast"],
                line=dict(color="#FF5252", width=1),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_features["datetime"],
                y=_plot_series_with_session_breaks(plot_features, "rtv_threshold_ticks"),
                mode="lines",
                name=t["chart_rtv_threshold"],
                line=dict(color="#00C853", width=1, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_features["datetime"],
                y=-_plot_series_with_session_breaks(plot_features, "rtv_threshold_ticks"),
                mode="lines",
                name=f"-{t['chart_rtv_threshold']}",
                line=dict(color="#00C853", width=1, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_features["datetime"],
                y=_plot_series_with_session_breaks(plot_features, "rtv_abs_move_ticks"),
                mode="lines",
                name=t["chart_rtv_abs"],
                line=dict(color="#40C4FF", width=1),
                fill="tozeroy",
                fillcolor="rgba(64,196,255,0.12)",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_features["datetime"],
                y=_plot_series_with_session_breaks(plot_features, "rtv_threshold_ticks"),
                mode="lines",
                name=t["chart_rtv_threshold"],
                line=dict(color="#00C853", width=1, dash="dash"),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=RTV_MIN_FAST_MOVE_TICKS, line_color="#FFAB00", line_dash="dot", row=3, col=1)
        fig.update_layout(
            template=tpl,
            height=680,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            legend_title_text=t["outcome"],
        )
        fig.update_yaxes(title_text=t["chart_price"], row=1, col=1)
        fig.update_yaxes(title_text=t["chart_rtv_fast"], row=2, col=1)
        fig.update_yaxes(title_text=t["chart_rtv_abs"], row=3, col=1)
        fig.update_xaxes(rangebreaks=_active_trading_rangebreaks())
        st.plotly_chart(fig, width="stretch")
        return

    fig.add_trace(
        go.Scatter(
            x=plot_features["datetime"],
            y=_plot_series_with_session_breaks(plot_features, "flow_imbalance"),
            mode="lines",
            name=t["chart_flow_imbalance"],
            line=dict(color="#FF5252", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_features["datetime"],
            y=_plot_series_with_session_breaks(plot_features, "book_imbalance"),
            mode="lines",
            name=t["chart_book_imbalance"],
            line=dict(color="#00C853", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=flow_threshold, line_color="#FF5252", line_dash="dash", row=2, col=1)
    fig.add_hline(y=book_threshold, line_color="#00C853", line_dash="dash", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=plot_features["datetime"],
            y=_plot_series_with_session_breaks(plot_features, "volume_intensity"),
            mode="lines",
            name=t["chart_volume_intensity"],
            line=dict(color="#40C4FF", width=1),
            fill="tozeroy",
            fillcolor="rgba(64,196,255,0.12)",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=volume_threshold, line_color="#40C4FF", line_dash="dash", row=3, col=1)
    fig.update_layout(
        template=tpl,
        height=680,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        legend_title_text=t["outcome"],
    )
    fig.update_yaxes(title_text=t["chart_price"], row=1, col=1)
    fig.update_yaxes(title_text=t["chart_flow_book"], row=2, col=1)
    fig.update_yaxes(title_text=t["chart_volume_x"], row=3, col=1)
    fig.update_xaxes(rangebreaks=_active_trading_rangebreaks())
    st.plotly_chart(fig, width="stretch")


def _event_option_label(row: pd.Series, position: int, t: dict) -> str:
    outcome = _outcome_label(row.get("outcome", ""), t)
    expected = _direction_label(row.get("expected_direction", ""), t)
    timestamp = row["datetime"].strftime("%Y-%m-%d %H:%M:%S")
    return f"#{position + 1} | {timestamp} | {outcome} | {expected}"


def _event_horizon_frame(
    features: pd.DataFrame,
    event: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    if event.name not in features.index:
        return pd.DataFrame()

    mask = features["symbol"].eq(event["symbol"])
    if "_session_id" in features.columns and "_session_id" in event.index:
        mask &= features["_session_id"].eq(event["_session_id"])
    group = features.loc[mask].sort_values("datetime")
    matches = np.flatnonzero(group.index.to_numpy() == event.name)
    if len(matches) == 0:
        return pd.DataFrame()

    row_loc = int(matches[0])
    current_price = float(event.get("mid_price", event.get("last_price", np.nan)))
    tick_size = float(event.get("tick_size_est", np.nan))
    expected = str(event.get("expected_direction", ""))
    rows = []
    for horizon in horizons:
        future_loc = row_loc + int(horizon)
        if future_loc >= len(group) or pd.isna(tick_size) or tick_size == 0:
            rows.append(
                {
                    "horizon": horizon,
                    "horizon_label": f"t+{horizon}",
                    "future_datetime": pd.NaT,
                    "future_price": np.nan,
                    "future_move_ticks": np.nan,
                    "expected_move_ticks": np.nan,
                    "direction_result": "Missing",
                }
            )
            continue

        future = group.iloc[future_loc]
        future_price = float(future["mid_price"])
        raw_move = (future_price - current_price) / tick_size
        if expected == "Down":
            expected_move = -raw_move
        elif expected == "Up":
            expected_move = raw_move
        else:
            expected_move = np.nan
        if pd.isna(expected_move):
            direction_result = "Missing"
        elif expected_move > 0:
            direction_result = "Profitable"
        elif expected_move < 0:
            direction_result = "Loss"
        else:
            direction_result = "Flat"
        rows.append(
            {
                "horizon": horizon,
                "horizon_label": f"t+{horizon}",
                "future_datetime": future["datetime"],
                "future_price": future_price,
                "future_move_ticks": raw_move,
                "expected_move_ticks": expected_move,
                "direction_result": direction_result,
            }
        )
    return pd.DataFrame(rows)


def _expected_move_in_hypothesis_direction(candidates: pd.DataFrame) -> pd.Series:
    future_move = pd.to_numeric(candidates["future_move_ticks"], errors="coerce")
    expected = candidates.get("expected_direction", pd.Series("", index=candidates.index)).astype(str)
    return pd.Series(
        np.select(
            [expected.eq("Down"), expected.eq("Up")],
            [-future_move, future_move],
            default=np.nan,
        ),
        index=candidates.index,
        dtype=float,
    )


def _render_payoff_asymmetry_chart(
    candidates: pd.DataFrame,
    min_success_ticks: float,
    tpl: str,
    t: dict,
):
    st.markdown(f"##### {t['payoff_chart_title']}")
    st.caption(t["payoff_chart_caption"])

    expected_move = _expected_move_in_hypothesis_direction(candidates).dropna()
    if expected_move.empty:
        st.info(t["payoff_empty"])
        return

    positive = expected_move[expected_move > 0]
    negative = expected_move[expected_move < 0]
    avg_favorable = float(positive.mean()) if len(positive) else np.nan
    avg_adverse_mag = float((-negative).mean()) if len(negative) else np.nan
    avg_adverse_plot = -avg_adverse_mag if pd.notna(avg_adverse_mag) else np.nan
    accuracy = float(candidates["is_correct"].mean()) if "is_correct" in candidates else np.nan
    payoff_ratio = (
        avg_favorable / avg_adverse_mag
        if pd.notna(avg_favorable) and pd.notna(avg_adverse_mag) and avg_adverse_mag > 0
        else np.nan
    )

    metric_cols = st.columns(2)
    metric_cols[0].metric(t["accuracy"], "N/A" if pd.isna(accuracy) else f"{accuracy:.1%}")
    metric_cols[1].metric(t["payoff_ratio"], "N/A" if pd.isna(payoff_ratio) else f"{payoff_ratio:.2f}x")
    st.caption(t["payoff_health_note"])

    plot_df = pd.DataFrame(
        {
            "label": [
                t["payoff_avg_favorable"],
                t["payoff_avg_adverse"],
            ],
            "value": [avg_favorable, avg_adverse_plot],
            "kind": ["favorable", "adverse"],
        }
    )
    color_map = {
        "favorable": "#00C853",
        "adverse": "#FF5252",
    }
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["value"],
            marker_color=[color_map[kind] for kind in plot_df["kind"]],
            customdata=np.column_stack([plot_df["value"], plot_df["kind"]]),
            hovertemplate=t["payoff_hover"],
        )
    )
    fig.add_hline(y=0, line_color="#78909C", line_dash="dash")
    fig.add_hline(
        y=float(min_success_ticks),
        line_color="#00C853",
        line_dash="dot",
        annotation_text=t["event_horizon_success_line"].format(success=f"{min_success_ticks:g}"),
    )
    fig.update_layout(
        template=tpl,
        height=330,
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis_title=t["payoff_yaxis"],
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def _render_event_horizon_bars(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    min_success_ticks: float,
    tpl: str,
    t: dict,
):
    st.markdown(f"#### {t['event_horizon_section_title']}")
    if candidates.empty:
        st.info(t["no_events"])
        return

    payoff_col, event_col = st.columns([1, 1])
    with payoff_col:
        _render_payoff_asymmetry_chart(candidates, min_success_ticks, tpl, t)

    with event_col:
        st.markdown(f"##### {t['event_horizon_title']}")
        st.caption(t["event_horizon_caption"])
        ordered = candidates.sort_values("datetime").copy()
        options = list(range(len(ordered)))
        key_symbol = str(ordered["symbol"].iloc[0]) if "symbol" in ordered.columns and not ordered.empty else "events"
        selected_pos = st.selectbox(
            t["select_event"],
            options,
            index=0,
            format_func=lambda pos: _event_option_label(ordered.iloc[int(pos)], int(pos), t),
            key=f"event_horizon_select_{key_symbol}_{len(ordered)}",
        )
        event = ordered.iloc[int(selected_pos)]
        horizon_df = _event_horizon_frame(features, event, RESEARCH_SWEEP_HORIZONS)
        if horizon_df.empty:
            st.info(t["event_horizon_empty"])
            return

        expected_label = _direction_label(event.get("expected_direction", ""), t)
        color_map = {
            "Profitable": "#00C853",
            "Loss": "#FF5252",
            "Flat": "#90A4AE",
            "Missing": "#CFD8DC",
        }
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=horizon_df["horizon_label"],
                y=horizon_df["expected_move_ticks"],
                marker_color=[color_map.get(value, "#CFD8DC") for value in horizon_df["direction_result"]],
                customdata=np.column_stack(
                    [
                        horizon_df["future_datetime"].astype(str),
                        horizon_df["future_price"],
                        horizon_df["future_move_ticks"],
                        horizon_df["expected_move_ticks"],
                        horizon_df["direction_result"].map(lambda value: t["event_horizon_result_labels"].get(value, value)),
                    ]
                ),
                hovertemplate=t["event_horizon_hover"],
            )
        )
        fig.add_hline(y=0, line_color="#78909C", line_dash="dash")
        fig.add_hline(
            y=float(min_success_ticks),
            line_color="#00C853",
            line_dash="dot",
            annotation_text=t["event_horizon_success_line"].format(success=f"{min_success_ticks:g}"),
        )
        fig.update_layout(
            template=tpl,
            height=330,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title=t["event_horizon_yaxis"].format(expected=expected_label),
            xaxis_title=t["sweep_horizon"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")


def _render_outcome_overview(candidates: pd.DataFrame, tpl: str, t: dict):
    st.markdown(f"#### {t['outcome_overview']}")
    plot_candidates = candidates.copy()
    plot_candidates["event_date"] = plot_candidates["datetime"].dt.date.astype(str)
    plot_candidates["outcome_label"] = plot_candidates["outcome"].map(lambda value: _outcome_label(value, t))
    color_map = {
        _outcome_label(key, t): value
        for key, value in OUTCOME_COLORS.items()
    }

    cols = st.columns([1.1, 1])
    with cols[0]:
        by_day = (
            plot_candidates.groupby(["event_date", "outcome_label"], sort=True)
            .size()
            .reset_index(name=t["ledger_count"])
        )
        fig = px.bar(
            by_day,
            x="event_date",
            y=t["ledger_count"],
            color="outcome_label",
            color_discrete_map=color_map,
            labels={
                "event_date": t["ledger_date"],
                "outcome_label": t["outcome"],
                t["ledger_count"]: t["ledger_count"],
            },
            title=t["events_by_day_chart"],
        )
        fig.update_layout(
            template=tpl,
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            legend_title_text=t["outcome"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with cols[1]:
        fig = px.histogram(
            plot_candidates,
            x="future_move_ticks",
            color="outcome_label",
            nbins=36,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map=color_map,
            labels={
                "future_move_ticks": f"{t['ledger_future_move']} ({t['ticks']})",
                "outcome_label": t["outcome"],
                "count": t["ledger_count"],
            },
            title=t["future_move_chart"],
        )
        fig.add_vline(x=0, line_color="#90A4AE", line_dash="dash")
        fig.update_layout(
            template=tpl,
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            legend_title_text=t["outcome"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")


def _build_event_ledger(candidates: pd.DataFrame, t: dict) -> pd.DataFrame:
    ordered = candidates.sort_values("datetime").reset_index(drop=True)
    ledger = pd.DataFrame(
        {
            t["ledger_date"]: ordered["datetime"].dt.date.astype(str),
            t["ledger_time"]: ordered["datetime"].dt.strftime("%H:%M:%S.%f").str.rstrip("0").str.rstrip("."),
            t["symbol"]: ordered["symbol"],
            t["ledger_session"]: ordered["_session_id"].add(1).astype(int) if "_session_id" in ordered else "",
            t["ledger_outcome"]: ordered["outcome"].map(lambda value: _outcome_label(value, t)),
            t["ledger_expected"]: ordered["expected_direction"].map(lambda value: _direction_label(value, t)),
            t["ledger_last_price"]: ordered["last_price"],
            t["ledger_future_time"]: ordered["future_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str.rstrip("0").str.rstrip("."),
            t["ledger_future_move"]: ordered["future_move_ticks"],
            t["ledger_flow"]: ordered["flow_imbalance"],
            t["ledger_book"]: ordered["book_imbalance"],
            t["ledger_volume_x"]: ordered["volume_intensity"],
            t["ledger_rolling_mid"]: ordered["rolling_mid_move_ticks"],
            t["ledger_price_shock"]: ordered["price_shock"],
        }
    )
    if "rtv_fast_move_ticks" in ordered.columns:
        ledger[t["ledger_rtv_fast"]] = ordered["rtv_fast_move_ticks"]
        ledger[t["ledger_rtv_threshold"]] = ordered["rtv_threshold_ticks"]
        ledger[t["ledger_rtv_pct"]] = ordered["rtv_threshold_ratio"]
    return ledger


def _style_event_ledger(ledger: pd.DataFrame, t: dict):
    correct_label = _outcome_label("Correct", t)
    failed_label = _outcome_label("Failed", t)
    outcome_col = t["ledger_outcome"]

    def color_outcome(column: pd.Series) -> list[str]:
        styles = []
        for value in column:
            if value == correct_label:
                styles.append("background-color: #E8F5E9; color: #1B5E20; font-weight: 700;")
            elif value == failed_label:
                styles.append("background-color: #FFEBEE; color: #B71C1C; font-weight: 700;")
            else:
                styles.append("")
        return styles

    format_map = {
        t["ledger_last_price"]: "{:.2f}",
        t["ledger_future_move"]: "{:+.2f}",
        t["ledger_flow"]: "{:.3f}",
        t["ledger_book"]: "{:.3f}",
        t["ledger_volume_x"]: "{:.2f}",
        t["ledger_rolling_mid"]: "{:+.2f}",
        t["ledger_price_shock"]: "{:.2f}",
        t["ledger_rtv_fast"]: "{:+.2f}",
        t["ledger_rtv_threshold"]: "{:.2f}",
        t["ledger_rtv_pct"]: "{:.2f}",
    }
    format_map = {col: fmt for col, fmt in format_map.items() if col in ledger.columns}
    return ledger.style.format(format_map).apply(color_outcome, subset=[outcome_col])


def _render_event_ledger(candidates: pd.DataFrame, t: dict):
    st.markdown(f"#### {t['event_ledger']}")
    st.caption(t["event_ledger_note"])
    ledger = _build_event_ledger(candidates, t)
    st.download_button(
        t["event_ledger_download"],
        ledger.to_csv(index=False).encode("utf-8-sig"),
        file_name="tick_pulse_event_ledger.csv",
        mime="text/csv",
        width="stretch",
    )
    st.dataframe(
        _style_event_ledger(ledger, t),
        width="stretch",
        hide_index=True,
        height=560,
    )
