#!/usr/bin/env python3
"""Build public figures from the sanitized aggregate evidence allowlist only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence/public_evidence.json"
FIGURES = PROJECT / "paper/figures"
STATES = ["Q1", "Q2", "Q3", "Q4"]
BLUE = "#1F5A75"
RED = "#B64235"
GREEN = "#13866F"
GOLD = "#9A6B20"
GRAY = "#8A9198"


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})

    performance = evidence["gross_performance"]
    labels = [row["strategy"] for row in performance]
    returns = [row["annual_return_pct"] for row in performance]
    drawdowns = [row["maximum_drawdown_pct"] for row in performance]
    colors = [BLUE, RED, GRAY, GREEN, GOLD]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(labels, returns, color=colors)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Annualized arithmetic return (%)")
    axes[0].set_title("Gross return")
    axes[0].tick_params(axis="x", rotation=30)
    style(axes[0])
    axes[1].bar(labels, drawdowns, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Maximum drawdown (%)")
    axes[1].set_title("Path risk")
    axes[1].tick_params(axis="x", rotation=30)
    style(axes[1])
    fig.suptitle("Endpoint outperformance does not establish a robust router")
    fig.tight_layout()
    save(fig, "fig01_public_performance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for ax, key, title in [
        (axes[0], "primary_proxy", "Lagged open-interest notional"),
        (axes[1], "equal_product_proxy", "Equal product"),
    ]:
        rows = evidence["conditional_returns_pct"][key]
        momentum = [row["momentum"] for row in rows]
        reversal = [row["reversal"] for row in rows]
        positions = np.arange(4)
        ax.bar(positions - 0.18, momentum, 0.36, color=BLUE, label="Momentum")
        ax.bar(positions + 0.18, reversal, 0.36, color=RED, label="Reversal")
        ax.set_xticks(positions, STATES)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel("Mean monthly return (%)")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Conditional sleeve ranking is non-monotonic and proxy-sensitive")
    fig.tight_layout()
    save(fig, "fig02_public_conditional_returns.png")

    stability = evidence["proxy_stability"]
    metrics = ["Volatility\ncorrelation", "Quartile-state\nagreement", "Q4 set\noverlap"]
    values = [
        stability["volatility_correlation"],
        stability["quartile_state_agreement"],
        stability["q4_jaccard_overlap"],
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bars = ax.bar(metrics, values, color=[BLUE, RED, GOLD], width=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coefficient / share")
    ax.set_title("Similar volatility series do not imply stable routing states")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")
    style(ax)
    fig.tight_layout()
    save(fig, "fig03_public_proxy_stability.png")

    anatomy = evidence["cross_sectional_anatomy"]
    measures = [
        ("mean_pairwise_correlation", "Pairwise correlation"),
        ("dispersion_pct", "Cross-sectional dispersion (%)"),
        ("directional_coherence", "Directional coherence"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    state_colors = ["#A9C3CF", "#75A1B4", "#447F98", RED]
    for ax, (key, title) in zip(axes, measures, strict=True):
        ax.bar(STATES, [row[key] for row in anatomy], color=state_colors)
        ax.set_title(title)
        style(ax)
    fig.suptitle("High aggregate volatility is more systemic, but not directionally sufficient")
    fig.tight_layout()
    save(fig, "fig04_public_market_anatomy.png")

    influence = evidence["q4_influence"]
    labels = ["Q4 mean", "Q4 median", "Mean excluding\nlargest event"]
    values = [
        influence["mean_reversal_minus_momentum_pct"],
        influence["median_reversal_minus_momentum_pct"],
        influence["mean_excluding_largest_event_pct"],
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(labels, values, color=[GREEN, GRAY, RED], width=0.62)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Reversal minus momentum (% per month)")
    ax.set_title("The apparent Q4 advantage is not representative of the state")
    style(ax)
    fig.tight_layout()
    save(fig, "fig05_public_event_influence.png")


if __name__ == "__main__":
    main()
