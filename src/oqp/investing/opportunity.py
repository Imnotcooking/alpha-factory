"""Decision summaries for the discretionary opportunity hub."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_opportunity_lens_frame(
    *,
    spot: float,
    target_consensus: float,
    direction: str,
    direction_score: float,
    news_label: str,
    news_score: float,
    forecast_vol: float,
    market_iv: float,
    rsi_14: float,
    expiration_count: int,
) -> pd.DataFrame:
    """Build normalized, dashboard-ready lenses for one discretionary ticker."""

    target_upside = (target_consensus / spot) - 1.0 if spot > 0 and target_consensus > 0 else None
    vol_edge = forecast_vol - market_iv
    timing_score = _timing_score(rsi_14)
    rows = [
        {
            "Lens": "Valuation",
            "Score": _clamp((target_upside or 0.0) / 0.25),
            "Bias": _valuation_bias(target_upside),
            "Detail": f"Consensus upside {target_upside:.1%}" if target_upside is not None else "No target consensus.",
        },
        {
            "Lens": "Direction",
            "Score": _clamp(direction_score),
            "Bias": direction or "Neutral",
            "Detail": f"Cross-check score {direction_score:.2f}.",
        },
        {
            "Lens": "News NLP",
            "Score": _clamp(news_score),
            "Bias": news_label or "No data",
            "Detail": f"Article tone score {news_score:.2f}.",
        },
        {
            "Lens": "Vol Pricing",
            "Score": _clamp(vol_edge / 0.15),
            "Bias": _vol_bias(vol_edge),
            "Detail": f"Forecast vol {forecast_vol:.1%}; ATM IV {market_iv:.1%}.",
        },
        {
            "Lens": "Timing",
            "Score": timing_score,
            "Bias": _timing_bias(rsi_14),
            "Detail": f"RSI 14 is {rsi_14:.1f}.",
        },
        {
            "Lens": "Options Depth",
            "Score": _clamp(expiration_count / 20.0),
            "Bias": "available" if expiration_count > 0 else "missing",
            "Detail": f"{expiration_count} expirations available.",
        },
    ]
    return pd.DataFrame(rows)


def build_vehicle_route_frame(
    *,
    target_upside: float | None,
    direction: str,
    direction_score: float,
    news_score: float,
    forecast_vol: float,
    market_iv: float,
) -> pd.DataFrame:
    """Rank discretionary route choices without approving any trade."""

    abs_direction = abs(direction_score)
    vol_edge = forecast_vol - market_iv
    direction_text = direction or "Neutral"
    rows = [
        {
            "Vehicle": "Shares / staged entry",
            "Fit": _fit_label(
                target_upside is not None and target_upside > 0.15 and direction_text != "Bearish" and news_score > -0.15,
                target_upside is not None and target_upside > 0.05 and direction_text != "Bearish",
            ),
            "Current Read": _target_detail(target_upside),
            "Risk Check": "Needs valuation margin of safety and invalidation level.",
        },
        {
            "Vehicle": "Long options",
            "Fit": _fit_label(abs_direction >= 0.30 and vol_edge > 0.03, abs_direction >= 0.20 and vol_edge >= -0.02),
            "Current Read": f"{direction_text} direction; forecast vol minus IV {vol_edge:.1%}.",
            "Risk Check": "Use only when convexity is worth the decay.",
        },
        {
            "Vehicle": "Defined-risk spread",
            "Fit": _fit_label(abs_direction >= 0.30, abs_direction >= 0.18),
            "Current Read": f"{direction_text} direction with controlled max loss.",
            "Risk Check": "Check spread width, liquidity, and max loss.",
        },
        {
            "Vehicle": "Income / short vol",
            "Fit": _fit_label(abs_direction < 0.20 and vol_edge < -0.04, abs_direction < 0.30 and vol_edge < 0.0),
            "Current Read": f"Market IV minus forecast vol {(market_iv - forecast_vol):.1%}.",
            "Risk Check": "Avoid if catalyst/news risk is high.",
        },
        {
            "Vehicle": "Watch / wait",
            "Fit": _fit_label(abs_direction < 0.18 and (target_upside is None or target_upside < 0.08), True),
            "Current Read": "Use when signals are mixed or incomplete.",
            "Risk Check": "Set alert trigger instead of forcing a trade.",
        },
    ]
    return pd.DataFrame(rows)


def build_options_playbook_frame(
    *,
    route_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    forecast_vol: float,
    market_iv: float,
    reference_expiry: str,
    chain_source: str,
) -> pd.DataFrame:
    """Combine route fit and contract candidates into a compact options playbook."""

    route_lookup = {}
    if route_frame is not None and not route_frame.empty:
        route_lookup = {
            str(row.get("Vehicle") or ""): str(row.get("Fit") or "Low")
            for row in route_frame.to_dict("records")
        }
    rows = [
        _playbook_row(
            theme="Long options",
            fit=route_lookup.get("Long options", "Low"),
            candidates=candidates,
            matcher=("Long Call", "Long Put"),
            forecast_vol=forecast_vol,
            market_iv=market_iv,
            reference_expiry=reference_expiry,
            chain_source=chain_source,
        ),
        _playbook_row(
            theme="Defined-risk spread",
            fit=route_lookup.get("Defined-risk spread", "Low"),
            candidates=candidates,
            matcher=("Bull Call", "Bear Put", "Butterfly", "Backspread", "Ratio"),
            forecast_vol=forecast_vol,
            market_iv=market_iv,
            reference_expiry=reference_expiry,
            chain_source=chain_source,
        ),
        _playbook_row(
            theme="Income / short vol",
            fit=route_lookup.get("Income / short vol", "Low"),
            candidates=candidates,
            matcher=("Cash-Secured Put", "Iron Condor", "Calendar"),
            forecast_vol=forecast_vol,
            market_iv=market_iv,
            reference_expiry=reference_expiry,
            chain_source=chain_source,
        ),
        {
            "Theme": "Shares / staged entry",
            "Fit": route_lookup.get("Shares / staged entry", "Low"),
            "Best Candidate": "equity entry plan",
            "PoP": None,
            "EV": None,
            "Max Loss": None,
            "Why": "Use valuation, target upside, and invalidation instead of option Greeks.",
            "Reference": reference_expiry or "n/a",
            "Source": chain_source,
        },
        {
            "Theme": "Watch / wait",
            "Fit": route_lookup.get("Watch / wait", "Medium"),
            "Best Candidate": "alert / no trade",
            "PoP": None,
            "EV": None,
            "Max Loss": None,
            "Why": "Keep this when the cockpit has mixed or weak pressure.",
            "Reference": reference_expiry or "n/a",
            "Source": chain_source,
        },
    ]
    order = {"High": 0, "Medium": 1, "Low": 2}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["Fit"].map(order).fillna(3)
    return frame.sort_values(["_order", "Theme"]).drop(columns=["_order"]).reset_index(drop=True)


def build_thesis_draft(
    *,
    symbol: str,
    action_bucket: str,
    action_reason: str,
    route_frame: pd.DataFrame,
    direction: str,
    direction_score: float,
    news_label: str,
    news_score: float,
    top_topics: str,
    top_keywords: str,
    catalyst_evidence: str = "",
) -> str:
    """Create a structured, editable thesis draft for the human decision pad."""

    route = primary_route(route_frame)
    evidence = []
    if top_topics:
        evidence.append(f"News topics: {top_topics}.")
    if top_keywords:
        evidence.append(f"Repeated keywords: {top_keywords}.")
    if catalyst_evidence:
        evidence.append(f"Catalyst board: {catalyst_evidence}.")
    evidence_text = "\n".join(f"- {item}" for item in evidence) or "- No article/topic evidence loaded yet."
    return (
        f"{symbol.upper()} thesis draft\n\n"
        f"1. Core read\n"
        f"- Action bucket: {action_bucket}\n"
        f"- Preferred route: {route}\n"
        f"- Reason: {action_reason}\n\n"
        f"2. Direction and sentiment\n"
        f"- Directional read: {direction} ({direction_score:.2f})\n"
        f"- News tone: {news_label} ({news_score:.2f})\n"
        f"{evidence_text}\n\n"
        f"3. Trade thesis\n"
        f"- I would consider this if the market confirms the setup instead of chasing the first print.\n\n"
        f"4. Invalidation\n"
        f"- Thesis is wrong if price action, news tone, or volatility pricing contradicts the route above.\n\n"
        f"5. Next check\n"
        f"- Review liquidity, spread width, earnings/catalyst dates, and max loss before any proposal."
    )


def build_decision_checklist_frame(
    *,
    route_frame: pd.DataFrame,
    article_count: int,
    target_upside: float | None,
    expiration_count: int,
    market_iv: float,
    forecast_vol: float,
) -> pd.DataFrame:
    """Build a compact pre-trade checklist for the discretionary hub."""

    route = primary_route(route_frame)
    vol_edge = forecast_vol - market_iv
    rows = [
        {
            "Check": "Primary route selected",
            "Status": "ready" if route != "Watch / wait" else "watch",
            "Detail": route,
        },
        {
            "Check": "News context loaded",
            "Status": "ready" if article_count > 0 else "missing",
            "Detail": f"{article_count} scored articles.",
        },
        {
            "Check": "Valuation context",
            "Status": "ready" if target_upside is not None else "missing",
            "Detail": f"Target upside {target_upside:.1%}." if target_upside is not None else "No target consensus.",
        },
        {
            "Check": "Options chain depth",
            "Status": "ready" if expiration_count > 0 else "missing",
            "Detail": f"{expiration_count} expirations available.",
        },
        {
            "Check": "Vol pricing read",
            "Status": "ready" if abs(vol_edge) >= 0.02 else "mixed",
            "Detail": f"Forecast vol minus ATM IV {vol_edge:.1%}.",
        },
    ]
    return pd.DataFrame(rows)


def primary_route(route_frame: pd.DataFrame) -> str:
    """Return the best non-low route from a vehicle route frame."""

    if route_frame is None or route_frame.empty:
        return "Watch / wait"
    order = {"High": 0, "Medium": 1, "Low": 2}
    frame = route_frame.copy()
    frame["_order"] = frame["Fit"].map(order).fillna(3)
    frame = frame.sort_values("_order")
    row = frame.iloc[0]
    vehicle = str(row.get("Vehicle") or "Watch / wait")
    fit = str(row.get("Fit") or "Low")
    if fit == "Low":
        return "Watch / wait"
    return vehicle


def _target_detail(target_upside: float | None) -> str:
    if target_upside is None:
        return "No analyst target consensus."
    return f"Consensus upside {target_upside:.1%}."


def _playbook_row(
    *,
    theme: str,
    fit: str,
    candidates: pd.DataFrame,
    matcher: tuple[str, ...],
    forecast_vol: float,
    market_iv: float,
    reference_expiry: str,
    chain_source: str,
) -> dict[str, Any]:
    candidate = _best_candidate(candidates, matcher)
    if candidate is None:
        best = "no candidate"
        pop = ev = max_loss = None
    else:
        best = str(candidate.get("Strategy") or theme)
        structure = str(candidate.get("Structure") or "")
        if structure:
            best = f"{best}: {structure}"
        pop = candidate.get("PoP")
        ev = candidate.get("EV")
        max_loss = candidate.get("Max Loss")
    return {
        "Theme": theme,
        "Fit": fit,
        "Best Candidate": best,
        "PoP": pop,
        "EV": ev,
        "Max Loss": max_loss,
        "Why": _playbook_why(theme, forecast_vol, market_iv),
        "Reference": reference_expiry or "n/a",
        "Source": chain_source,
    }


def _best_candidate(candidates: pd.DataFrame, matcher: tuple[str, ...]) -> dict[str, Any] | None:
    if candidates is None or candidates.empty or "Strategy" not in candidates:
        return None
    strategy = candidates["Strategy"].astype(str)
    mask = strategy.apply(lambda value: any(token.lower() in value.lower() for token in matcher))
    subset = candidates.loc[mask].copy()
    if subset.empty:
        return None
    if "EV" in subset:
        subset["_ev"] = pd.to_numeric(subset["EV"], errors="coerce").fillna(float("-inf"))
        subset = subset.sort_values("_ev", ascending=False)
    return subset.iloc[0].to_dict()


def _playbook_why(theme: str, forecast_vol: float, market_iv: float) -> str:
    vol_edge = forecast_vol - market_iv
    if theme == "Long options":
        return f"Best when direction is strong and forecast vol exceeds IV by enough; current edge {vol_edge:.1%}."
    if theme == "Defined-risk spread":
        return "Useful when direction exists but premium/risk must be controlled."
    if theme == "Income / short vol":
        return f"Best when market IV is rich vs forecast vol; current richness {(market_iv - forecast_vol):.1%}."
    return "Review against the cockpit before forcing a structure."


def _valuation_bias(target_upside: float | None) -> str:
    if target_upside is None:
        return "missing"
    if target_upside >= 0.15:
        return "constructive"
    if target_upside <= -0.10:
        return "expensive"
    return "mixed"


def _vol_bias(vol_edge: float) -> str:
    if vol_edge >= 0.03:
        return "long-vol friendly"
    if vol_edge <= -0.04:
        return "income/short-vol friendly"
    return "neutral"


def _timing_bias(rsi_14: float) -> str:
    if pd.isna(rsi_14):
        return "missing"
    if rsi_14 <= 35:
        return "oversold"
    if rsi_14 >= 70:
        return "extended"
    return "neutral"


def _timing_score(rsi_14: float) -> float:
    if pd.isna(rsi_14):
        return 0.0
    if rsi_14 <= 30:
        return 0.55
    if rsi_14 <= 40:
        return 0.25
    if rsi_14 >= 75:
        return -0.55
    if rsi_14 >= 65:
        return -0.25
    return 0.0


def _fit_label(high: bool, medium: bool) -> str:
    if high:
        return "High"
    if medium:
        return "Medium"
    return "Low"


def _clamp(value: Any, lower: float = -1.0, upper: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return max(lower, min(upper, number))
