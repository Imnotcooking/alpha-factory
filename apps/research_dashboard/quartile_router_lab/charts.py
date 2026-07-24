"""Plotly figures for the quartile router lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STATE_COLORS = {
    "Q1": "#2A9D78",
    "Q2": "#2E6F95",
    "Q3": "#D58A27",
    "Q4": "#BC4639",
}
STRATEGY_COLORS = {
    "router": "#111827",
    "paper_router": "#111827",
    "ema_q4_router": "#2A9D78",
    "ema_20_60_reversal_router": "#8B5A83",
    "dual_ema_router": "#E07A5F",
    "macd_reversal_router": "#6B7280",
    "macd_ema_router": "#007C83",
    "momentum": "#2E6F95",
    "reversal": "#BC4639",
    "ema_5_10": "#D58A27",
    "ema_20_60": "#4C956C",
    "macd_12_26_9": "#7B61A8",
    "static_50_50": "#7C6F64",
}


def _layout(fig: go.Figure, template: str, *, height: int, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        template=template,
        height=height,
        margin=dict(l=20, r=20, t=78, b=24),
        hovermode=hovermode,
        title=dict(y=0.97, x=0.0, xanchor="left", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1),
        font=dict(size=12),
    )
    return fig


def regime_timeline(monthly: pd.DataFrame, template: str, labels: dict[str, str]) -> go.Figure:
    frame = monthly.sort_values("date")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.82, 0.18],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["current_volatility"],
            name=labels["volatility"],
            mode="lines+markers",
            line=dict(color="#222222", width=2),
            marker=dict(
                color=frame["volatility_state"].map(STATE_COLORS), size=7
            ),
            customdata=np.stack(
                [frame["signal_month"], frame["volatility_state"], frame["selected_strategy"]],
                axis=-1,
            ),
            hovertemplate=(
                labels["holding_month"]
                + ": %{x|%Y-%m}<br>"
                + labels["signal_month"]
                + ": %{customdata[0]}<br>"
                + labels["state"]
                + ": %{customdata[1]}<br>"
                + labels["strategy"]
                + ": %{customdata[2]}<br>"
                + labels["volatility"]
                + ": %{y:.2%}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    for column, name, color in [
        ("q25", "Q1 / Q2", STATE_COLORS["Q1"]),
        ("q50", "Q2 / Q3", STATE_COLORS["Q2"]),
        ("q75", "Q3 / Q4", STATE_COLORS["Q4"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                name=name,
                mode="lines",
                line=dict(color=color, width=1.3, dash="dot"),
                hovertemplate="%{x|%Y-%m}: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    state_number = frame["volatility_state"].map({state: idx for idx, state in enumerate(STATE_COLORS)})
    colorscale = []
    for idx, state in enumerate(STATE_COLORS):
        start = idx / 4.0
        end = (idx + 1) / 4.0
        colorscale.extend([(start, STATE_COLORS[state]), (end, STATE_COLORS[state])])
    fig.add_trace(
        go.Heatmap(
            x=frame["date"],
            y=[labels["holding_state"]],
            z=[state_number],
            zmin=0,
            zmax=3,
            colorscale=colorscale,
            showscale=False,
            customdata=[[state for state in frame["volatility_state"]]],
            hovertemplate="%{x|%Y-%m}: %{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=labels["volatility_axis"], tickformat=".1%", row=1, col=1)
    fig.update_yaxes(showticklabels=True, row=2, col=1)
    fig.update_xaxes(title_text=labels["holding_month"], row=2, col=1)
    fig.update_layout(title=labels["regime_title"])
    return _layout(fig, template, height=490)


def diagnostic_timeline(
    monthly: pd.DataFrame,
    column: str,
    metric_label: str,
    template: str,
    title: str,
    percent: bool,
    state_label: str,
    signal_label: str,
) -> go.Figure:
    frame = monthly.dropna(subset=[column]).sort_values("date")
    fig = go.Figure(
        go.Bar(
            x=frame["date"],
            y=frame[column],
            marker_color=frame["volatility_state"].map(STATE_COLORS),
            customdata=np.stack([frame["volatility_state"], frame["signal_month"]], axis=-1),
            hovertemplate=(
                "%{x|%Y-%m}<br>"
                + metric_label
                + ": %{y:"
                + (".2%" if percent else ".2f")
                + "}<br>"
                + state_label
                + ": %{customdata[0]}<br>"
                + signal_label
                + ": %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title=title, showlegend=False)
    fig.update_yaxes(title=metric_label, tickformat=".1%" if percent else None)
    fig.update_xaxes(title=None)
    return _layout(fig, template, height=340, hovermode="x")


def product_market_percentile_timeline(
    detail: pd.DataFrame,
    template: str,
    title: str,
    product_label: str,
    market_label: str,
    percentile_label: str,
    state_label: str,
) -> go.Figure:
    frame = detail.sort_values("holding_month").copy()
    frame["date"] = pd.PeriodIndex(frame["holding_month"], freq="M").to_timestamp("M")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["product_volatility_percentile"],
            name=product_label,
            mode="lines+markers",
            line=dict(color="#2E6F95", width=2),
            marker=dict(
                color=frame["product_volatility_state"].map(STATE_COLORS), size=7
            ),
            customdata=frame[["product_volatility_state"]].to_numpy(),
            hovertemplate=(
                "%{x|%Y-%m}<br>"
                + product_label
                + ": %{y:.1%}<br>"
                + state_label
                + ": %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["market_volatility_percentile"],
            name=market_label,
            mode="lines",
            line=dict(color="#111827", width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#BC4639", opacity=0.06, line_width=0)
    fig.update_layout(title=title)
    fig.update_yaxes(title=percentile_label, tickformat=".0%", range=[0.0, 1.0])
    return _layout(fig, template, height=420)


def product_q4_breadth_timeline(
    breadth: pd.DataFrame,
    template: str,
    title: str,
    equal_label: str,
    oi_label: str,
    y_title: str,
) -> go.Figure:
    frame = breadth.sort_values("holding_month").copy()
    frame["date"] = pd.PeriodIndex(frame["holding_month"], freq="M").to_timestamp("M")
    fig = go.Figure()
    for column, name, color in [
        ("product_q4_share", equal_label, "#2E6F95"),
        ("oi_weighted_product_q4_share", oi_label, "#D58A27"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(
                    color=frame["market_volatility_state"].map(STATE_COLORS), size=6
                ),
                customdata=frame[["market_volatility_state"]].to_numpy(),
                hovertemplate=(
                    "%{x|%Y-%m}: %{y:.1%}<br>Market: %{customdata[0]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(title=title)
    fig.update_yaxes(title=y_title, tickformat=".0%", range=[0.0, 1.0])
    return _layout(fig, template, height=380)


def comparison_equity_drawdown(
    comparison: pd.DataFrame,
    basis: str,
    template: str,
    title: str,
    strategy_labels: dict[str, str],
    equity_label: str,
    drawdown_label: str,
) -> go.Figure:
    frame = comparison.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.07,
    )
    for strategy, group in frame.groupby("strategy", sort=False):
        group = group.sort_values("date")
        returns = group[f"{basis}_return"].fillna(0.0)
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        primary = strategy == "ema_q4_router"
        color = STRATEGY_COLORS.get(strategy, "#667085")
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=wealth - 1.0,
                name=strategy_labels.get(strategy, strategy),
                mode="lines",
                line=dict(color=color, width=3 if primary else 1.5),
                hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=drawdown,
                name=strategy_labels.get(strategy, strategy),
                mode="lines",
                line=dict(color=color, width=2 if primary else 1),
                showlegend=False,
                hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(title=title)
    fig.update_yaxes(title_text=equity_label, tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text=drawdown_label, tickformat=".0%", row=2, col=1)
    fig = _layout(fig, template, height=620)
    fig.update_layout(
        margin=dict(l=20, r=20, t=82, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.13,
            xanchor="left",
            x=0.0,
        ),
    )
    return fig


def transition_heatmap(
    matrix: pd.DataFrame,
    template: str,
    title: str,
    next_state_label: str,
    current_state_label: str,
    probability_label: str,
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale=[[0.0, "#F4F5F7"], [1.0, "#2E6F95"]],
            zmin=0.0,
            zmax=max(0.5, float(matrix.to_numpy().max())),
            text=np.vectorize(lambda value: f"{value:.1%}")(matrix.to_numpy()),
            texttemplate="%{text}",
            hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>",
            colorbar=dict(title=probability_label),
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=next_state_label)
    fig.update_yaxes(title=current_state_label, autorange="reversed")
    return _layout(fig, template, height=360, hovermode="closest")


def equity_curve(
    monthly: pd.DataFrame,
    basis: str,
    template: str,
    title: str,
    strategy_labels: dict[str, str],
    y_title: str,
) -> go.Figure:
    suffix = "return" if basis == "gross" else "net_return"
    columns = {
        "router": f"router_{basis}_return",
        "momentum": f"momentum_{suffix}",
        "reversal": f"reversal_{suffix}",
        "static_50_50": f"static_50_50_{suffix}",
    }
    fig = go.Figure()
    for strategy, column in columns.items():
        if column not in monthly.columns:
            continue
        curve = (1.0 + monthly[column].fillna(0.0)).cumprod() - 1.0
        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=curve,
                name=strategy_labels[strategy],
                mode="lines",
                line=dict(
                    color=STRATEGY_COLORS[strategy],
                    width=3 if strategy == "router" else 1.8,
                    dash="solid" if strategy == "router" else "dot",
                ),
                hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(title=title)
    fig.update_yaxes(title=y_title, tickformat=".0%")
    return _layout(fig, template, height=460)


def monthly_return_bars(
    monthly: pd.DataFrame, basis: str, template: str, title: str, y_title: str,
    state_label: str, strategy_label: str,
) -> go.Figure:
    column = f"router_{basis}_return"
    fig = go.Figure(
        go.Bar(
            x=monthly["date"],
            y=monthly[column],
            marker_color=monthly["volatility_state"].map(STATE_COLORS),
            customdata=np.stack(
                [monthly["volatility_state"], monthly["selected_strategy"]], axis=-1
            ),
            hovertemplate=(
                "%{x|%Y-%m}: %{y:.2%}<br>"
                + state_label
                + ": %{customdata[0]}<br>"
                + strategy_label
                + ": %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.0, line_color="#444444", line_width=1)
    fig.update_layout(title=title, showlegend=False)
    fig.update_yaxes(title=y_title, tickformat=".0%")
    return _layout(fig, template, height=360, hovermode="x")


def state_contribution(
    state_summary: pd.DataFrame, basis: str, template: str, title: str,
    y_title: str, months_label: str, strategy_label: str,
) -> go.Figure:
    column = f"total_{basis}_contribution"
    fig = go.Figure(
        go.Bar(
            x=state_summary["state"],
            y=state_summary[column],
            marker_color=state_summary["state"].map(STATE_COLORS),
            text=state_summary[column].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            customdata=np.stack(
                [state_summary["months"], state_summary["selected_strategy"]], axis=-1
            ),
            hovertemplate=(
                "%{x}: %{y:.2%}<br>"
                + months_label
                + ": %{customdata[0]}<br>"
                + strategy_label
                + ": %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.0, line_color="#444444", line_width=1)
    fig.update_layout(title=title, showlegend=False)
    fig.update_yaxes(title=y_title, tickformat=".0%")
    return _layout(fig, template, height=360, hovermode="x")


def holdings_bar(
    positions: pd.DataFrame,
    template: str,
    title: str,
    labels: dict[str, str],
) -> go.Figure:
    frame = positions.loc[positions["target_weight"].ne(0.0)].sort_values("target_weight")
    colors = np.where(frame["target_weight"].gt(0.0), "#2A9D78", "#BC4639")
    fig = go.Figure(
        go.Bar(
            x=frame["target_weight"],
            y=frame["root"],
            orientation="h",
            marker_color=colors,
            customdata=frame[
                ["sector", "holding_return", "gross_contribution"]
            ].to_numpy(),
            hovertemplate=(
                "%{y}<br>"
                + labels["weight"]
                + ": %{x:.2%}<br>"
                + labels["sector"]
                + ": %{customdata[0]}<br>"
                + labels["return"]
                + ": %{customdata[1]:.2%}<br>"
                + labels["gross_contribution"]
                + ": %{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.0, line_color="#333333", line_width=1)
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title=labels["weight"], tickformat=".0%")
    return _layout(fig, template, height=max(380, min(720, 28 * len(frame))), hovermode="closest")


def holdings_heatmap(
    holdings: pd.DataFrame,
    state: str,
    template: str,
    title: str,
    holding_month_label: str,
    weight_label: str,
    max_roots: int = 28,
) -> go.Figure:
    frame = holdings.loc[
        holdings["volatility_state"].eq(state) & holdings["target_weight"].ne(0.0)
    ].copy()
    activity = (
        frame.groupby("root", observed=True)["target_weight"]
        .apply(lambda values: values.abs().mean())
        .nlargest(max_roots)
    )
    frame = frame.loc[frame["root"].isin(activity.index)]
    pivot = frame.pivot(index="root", columns="month", values="target_weight").fillna(0.0)
    pivot = pivot.reindex(activity.index)
    limit = max(0.01, float(np.nanmax(np.abs(pivot.to_numpy()))) if not pivot.empty else 0.01)
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=pivot.index,
            zmin=-limit,
            zmax=limit,
            zmid=0.0,
            colorscale=[[0.0, "#BC4639"], [0.5, "#F6F7F8"], [1.0, "#2A9D78"]],
            colorbar=dict(title=weight_label, tickformat=".0%"),
            hovertemplate="%{y}<br>%{x}<br>Weight: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=holding_month_label, type="category")
    fig.update_yaxes(title=None, autorange="reversed")
    return _layout(fig, template, height=max(440, 22 * len(pivot)), hovermode="closest")
