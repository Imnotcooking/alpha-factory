from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def opportunity_scatter(candidates: pd.DataFrame, template: str) -> go.Figure:
    fig = px.scatter(
        candidates,
        x="stability_score",
        y="dislocation_score",
        size="avg_pair_volume",
        color="arbitrage_type",
        hover_name="candidate_id",
        hover_data=["opportunity_score", "latest_z", "correlation", "half_life"],
        template=template,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Stability score",
        yaxis_title="Dislocation score",
    )
    return fig


def sector_heatmap(candidates: pd.DataFrame, template: str) -> go.Figure:
    heat = (
        candidates.groupby("sector_pair", as_index=False)
        .agg(score=("opportunity_score", "mean"), count=("candidate_id", "size"))
        .sort_values("score", ascending=False)
        .head(20)
    )
    fig = px.bar(
        heat,
        x="score",
        y="sector_pair",
        color="count",
        orientation="h",
        template=template,
        labels={"score": "Avg opportunity score", "sector_pair": "Sector pair", "count": "Pairs"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis={"categoryorder": "total ascending"})
    return fig


def top_dislocation_bar(candidates: pd.DataFrame, template: str) -> go.Figure:
    top = candidates.sort_values("abs_latest_z", ascending=False).head(12)
    fig = px.bar(
        top,
        x="abs_latest_z",
        y="candidate_id",
        color="opportunity_score",
        orientation="h",
        template=template,
        labels={"abs_latest_z": "|Latest z|", "candidate_id": "Candidate"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis={"categoryorder": "total ascending"})
    return fig


def price_legs(spread: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    if not spread.empty:
        y_norm = spread["y_price"] / spread["y_price"].dropna().iloc[0] * 100.0
        x_norm = spread["x_price"] / spread["x_price"].dropna().iloc[0] * 100.0
        fig.add_trace(go.Scatter(x=spread["date"], y=y_norm, mode="lines", name="Y normalized"))
        fig.add_trace(go.Scatter(x=spread["date"], y=x_norm, mode="lines", name="X normalized"))
    fig.update_layout(template=template, height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Indexed to 100")
    return fig


def spread_zscore(spread: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread["date"], y=spread["spread_z"], mode="lines", name="Spread z"))
    for y, color in [(2.0, "#dc2626"), (-2.0, "#16a34a"), (0.0, "#64748b")]:
        fig.add_hline(y=y, line_dash="dash", line_color=color)
    fig.update_layout(template=template, height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Z-score")
    return fig


def spread_level(spread: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread["date"], y=spread["spread"], mode="lines", name="Spread"))
    fig.update_layout(template=template, height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Spread")
    return fig


def dkf_beta(pair: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pair["date"], y=pair["dynamic_beta"], mode="lines", name="Dynamic beta"))
    fig.update_layout(template=template, height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Beta")
    return fig


def dkf_residual(pair: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pair["date"], y=pair["residual_z"], mode="lines", name="Residual z"))
    fig.add_hline(y=2.0, line_dash="dash", line_color="#dc2626")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="#16a34a")
    fig.update_layout(template=template, height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Residual z")
    return fig


def dkf_uncertainty(pair: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pair["date"], y=pair["state_uncertainty"], mode="lines", name="State uncertainty"))
    fig.update_layout(template=template, height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Trace(P)")
    return fig


def residual_distribution(pair: pd.DataFrame, template: str) -> go.Figure:
    fig = px.histogram(pair.dropna(subset=["residual_z"]), x="residual_z", nbins=60, template=template)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Residual z")
    return fig


def backtest_equity(curve: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], mode="lines", name="Net preview equity"))
    fig.update_layout(template=template, height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Spread PnL units")
    return fig


def backtest_drawdown(curve: pd.DataFrame, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["drawdown"], mode="lines", fill="tozeroy", name="Drawdown"))
    fig.update_layout(template=template, height=280, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Drawdown")
    return fig
