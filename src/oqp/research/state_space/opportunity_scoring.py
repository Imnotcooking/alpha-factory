from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "bounded_score",
    "cost_score",
    "dislocation_score",
    "interpret_candidate",
    "liquidity_score_from_rank",
    "mean_reversion_score",
    "score_opportunity",
    "stability_score",
]


def bounded_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def dislocation_score(latest_z: float) -> float:
    if not np.isfinite(latest_z):
        return 0.0
    return float(np.clip(abs(latest_z) / 3.0, 0.0, 1.0))


def mean_reversion_score(half_life: float) -> float:
    if not np.isfinite(half_life) or half_life <= 0:
        return 0.0
    if 2.0 <= half_life <= 60.0:
        return 1.0
    if half_life < 2.0:
        return bounded_score(half_life, 0.0, 2.0)
    return float(np.clip(1.0 - ((half_life - 60.0) / 120.0), 0.0, 1.0))


def stability_score(correlation: float, beta_drift: float) -> float:
    corr_component = bounded_score(abs(correlation), 0.2, 0.85)
    drift_component = 1.0 / (1.0 + max(0.0, float(beta_drift))) if np.isfinite(beta_drift) else 0.0
    return float(np.clip(0.65 * corr_component + 0.35 * drift_component, 0.0, 1.0))


def cost_score(round_turn_cost_bps: float) -> float:
    if not np.isfinite(round_turn_cost_bps):
        return 0.5
    return float(np.clip(1.0 - max(0.0, round_turn_cost_bps) / 25.0, 0.0, 1.0))


def liquidity_score_from_rank(log_liquidity: pd.Series) -> pd.Series:
    values = pd.to_numeric(log_liquidity, errors="coerce").fillna(0.0)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=values.index)
    return values.rank(pct=True).astype(float)


def score_opportunity(row: pd.Series) -> dict[str, float | str]:
    dislocation = dislocation_score(float(row.get("latest_z", np.nan)))
    mean_rev = mean_reversion_score(float(row.get("half_life", np.nan)))
    stable = stability_score(float(row.get("correlation", np.nan)), float(row.get("beta_drift", np.nan)))
    liquid = float(row.get("liquidity_score", 0.5))
    cheap = cost_score(float(row.get("round_turn_cost_bps", np.nan)))
    score = 100.0 * (
        0.32 * dislocation
        + 0.24 * stable
        + 0.20 * mean_rev
        + 0.14 * liquid
        + 0.10 * cheap
    )
    return {
        "dislocation_score": dislocation,
        "stability_score": stable,
        "mean_reversion_score": mean_rev,
        "cost_score": cheap,
        "opportunity_score": float(np.clip(score, 0.0, 100.0)),
        "interpretation": interpret_candidate(
            opportunity_score=score,
            latest_z=float(row.get("latest_z", np.nan)),
            half_life=float(row.get("half_life", np.nan)),
            beta_drift=float(row.get("beta_drift", np.nan)),
        ),
    }


def interpret_candidate(
    *,
    opportunity_score: float,
    latest_z: float,
    half_life: float,
    beta_drift: float,
) -> str:
    if not np.isfinite(opportunity_score):
        return "Insufficient data"
    if abs(latest_z) < 1.5:
        return "Watchlist: relationship may be valid, but current dislocation is mild"
    if not np.isfinite(half_life) or half_life > 120:
        return "Fragile: dislocated, but mean reversion is weak or unproven"
    if np.isfinite(beta_drift) and beta_drift > 0.75:
        return "Fragile: hedge ratio is moving too much"
    if opportunity_score >= 70:
        return "Promising for research: strong dislocation with acceptable stability"
    if opportunity_score >= 50:
        return "Worth reviewing: some evidence, needs drilldown and cost checks"
    return "Low priority: weak combined evidence"
