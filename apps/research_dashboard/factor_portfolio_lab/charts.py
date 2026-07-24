"""Plotly figures for factor-portfolio construction and diagnostics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def factor_weight_chart(weights: pd.DataFrame, *, template: str, labels: dict) -> go.Figure:
    frame = weights.sort_values("weight", ascending=True)
    fig = px.bar(
        frame,
        x="weight",
        y="factor_id",
        orientation="h",
        color="category",
        text=frame["weight"].map(lambda value: f"{value:.1%}"),
        labels={"weight": labels["weight"], "factor_id": labels["factor"]},
        template=template,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=max(300, 54 * len(frame)),
        legend_title_text=labels["category"],
        xaxis_tickformat=".0%",
        margin=dict(l=20, r=80, t=20, b=30),
    )
    return fig


def correlation_heatmap(correlation: pd.DataFrame, *, template: str, title: str) -> go.Figure:
    fig = px.imshow(
        correlation,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
        aspect="auto",
        template=template,
    )
    fig.update_layout(title=title, height=max(380, 68 * len(correlation)))
    return fig


def coverage_chart(coverage: pd.DataFrame, *, template: str, labels: dict) -> go.Figure:
    frame = coverage.sort_values("coverage", ascending=True)
    fig = px.bar(
        frame,
        x="coverage",
        y="factor_id",
        orientation="h",
        text=frame["coverage"].map(lambda value: f"{value:.1%}"),
        color="coverage",
        color_continuous_scale="Teal",
        labels={"coverage": labels["coverage"], "factor_id": labels["factor"]},
        template=template,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=max(300, 54 * len(frame)),
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
        margin=dict(l=20, r=80, t=20, b=30),
    )
    return fig


def contribution_chart(
    contribution: pd.DataFrame,
    *,
    template: str,
    labels: dict,
) -> go.Figure:
    frame = contribution.sort_values("mean_abs_contribution", ascending=True)
    fig = px.bar(
        frame,
        x="mean_abs_contribution",
        y="factor_id",
        orientation="h",
        color="correlation_to_composite",
        color_continuous_scale="Viridis",
        labels={
            "mean_abs_contribution": labels["mean_abs_contribution"],
            "factor_id": labels["factor"],
            "correlation_to_composite": labels["correlation_to_composite"],
        },
        template=template,
    )
    fig.update_layout(
        height=max(300, 54 * len(frame)),
        margin=dict(l=20, r=30, t=20, b=30),
    )
    return fig


def contribution_timeline(
    frame: pd.DataFrame,
    contribution_columns: dict[str, str],
    *,
    template: str,
    labels: dict,
) -> go.Figure:
    daily = frame[["date", *contribution_columns.values()]].copy()
    daily = daily.groupby("date", as_index=False).mean(numeric_only=True)
    daily = daily.rename(
        columns={value: factor_id for factor_id, value in contribution_columns.items()}
    )
    long = daily.melt(id_vars="date", var_name="factor_id", value_name="contribution")
    fig = px.area(
        long,
        x="date",
        y="contribution",
        color="factor_id",
        labels={
            "date": labels["date"],
            "contribution": labels["contribution"],
            "factor_id": labels["factor"],
        },
        template=template,
    )
    fig.update_layout(height=430, hovermode="x unified")
    return fig


def leave_one_out_chart(summary: pd.DataFrame, *, template: str, labels: dict) -> go.Figure:
    frame = summary.sort_values("mean_abs_signal_change", ascending=True)
    fig = px.bar(
        frame,
        x="mean_abs_signal_change",
        y="omitted_factor",
        orientation="h",
        color="correlation_to_full",
        color_continuous_scale="Plasma",
        labels={
            "mean_abs_signal_change": labels["signal_change"],
            "omitted_factor": labels["omitted_factor"],
            "correlation_to_full": labels["correlation_to_full"],
        },
        template=template,
    )
    fig.update_layout(
        height=max(300, 54 * len(frame)),
        margin=dict(l=20, r=30, t=20, b=30),
    )
    return fig


__all__ = [
    "contribution_chart",
    "contribution_timeline",
    "correlation_heatmap",
    "coverage_chart",
    "factor_weight_chart",
    "leave_one_out_chart",
]
