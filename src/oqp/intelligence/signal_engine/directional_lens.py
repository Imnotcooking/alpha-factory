"""Directional cross-checks for discretionary options workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from oqp.data import FMPDataAdapter
from oqp.data.base import DataAdapterError


HORIZON_DEFINITIONS: tuple[tuple[str, int], ...] = (("1D", 1), ("1W", 5), ("1M", 21))
HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "1D": {"trend": 0.30, "rsi": 0.25, "sentiment": 0.25, "analyst": 0.10, "options": 0.10},
    "1W": {"trend": 0.35, "rsi": 0.15, "sentiment": 0.25, "analyst": 0.15, "options": 0.10},
    "1M": {"trend": 0.35, "rsi": 0.10, "sentiment": 0.15, "analyst": 0.30, "options": 0.10},
}

STRATEGY_DIRECTION_MAP = {
    "Long Call": "Bullish",
    "Long Calls": "Bullish",
    "Long Put": "Bearish",
    "Long Puts": "Bearish",
    "Cash-Secured Put": "Bullish / neutral",
    "Cash-Secured Puts": "Bullish / neutral",
    "Bull Call Spread": "Bullish",
    "Bull Call Spreads": "Bullish",
    "Bear Put Spread": "Bearish",
    "Bear Put Spreads": "Bearish",
    "Iron Condor": "Range-bound",
    "Iron Condors": "Range-bound",
    "Call Butterfly": "Pin / neutral",
    "Calendar Spread": "Pin / neutral",
    "Calendar Spreads": "Pin / neutral",
    "Call Ratio Spread": "Bullish / convex",
    "Put Ratio Spread": "Bearish / convex",
    "Call Backspread": "Explosive bullish",
    "Call Backspreads": "Explosive bullish",
    "Put Backspread": "Explosive bearish",
    "Put Backspreads": "Explosive bearish",
    "Long Straddle / Strangle": "Breakout",
    "Short Straddle / Strangle": "Range-bound",
    "Deep Value LEAPS": "Capitulation rebound",
    "Collars": "Defensive",
}


@dataclass(frozen=True, slots=True)
class DirectionalLensResult:
    """Dashboard-ready directional signal bundle."""

    symbol: str
    as_of: str
    spot: float
    horizon_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    contribution_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    sentiment_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def fetch_directional_sentiment(
    api_key: str | None,
    symbol: str,
    *,
    adapter_factory: Callable[[str], FMPDataAdapter] = FMPDataAdapter,
) -> dict[str, Any]:
    """Fetch optional FMP sentiment/rating payloads for a symbol.

    The caller should cache this. Missing access is treated as empty data so the
    directional lens can still run from local price/option signals.
    """

    if not api_key or not symbol:
        return {}
    try:
        adapter = adapter_factory(api_key)
    except Exception:
        return {}

    payload: dict[str, Any] = {}
    calls = {
        "news": lambda: adapter.get_news_sentiment(symbol),
        "social": lambda: adapter.get_social_sentiment(symbol),
        "upgrades": lambda: adapter.get_upgrades_downgrades(symbol),
        "rating": lambda: adapter.get_historical_rating(symbol, limit=30),
    }
    for name, call in calls.items():
        try:
            payload[name] = call()
        except (DataAdapterError, OSError, ValueError, TypeError):
            payload[name] = []
    return payload


def build_directional_lens(
    symbol: str,
    history: pd.DataFrame,
    *,
    spot: float | None = None,
    price_targets: dict[str, Any] | None = None,
    sentiment_payload: dict[str, Any] | None = None,
    calls: pd.DataFrame | None = None,
    puts: pd.DataFrame | None = None,
) -> DirectionalLensResult:
    """Build a multi-horizon directional cross-check for an option ticker."""

    clean = _normalize_history(history)
    symbol_key = str(symbol or "").upper().strip()
    resolved_spot = _resolve_spot(clean, spot)
    as_of = _as_of(clean)
    close = pd.to_numeric(clean.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()

    sentiment_score, sentiment_rows = _sentiment_score(sentiment_payload or {})
    analyst_score, analyst_detail = _analyst_score(price_targets or {}, resolved_spot)
    options_score, options_detail = _options_skew_score(calls, puts, resolved_spot)
    rsi_value = _rsi(close)
    rsi_score = _rsi_score(rsi_value)
    realized_vol = _realized_vol(close, 21) or _realized_vol(close, 63) or 0.25

    horizon_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for label, days in HORIZON_DEFINITIONS:
        trend_score, trend_detail = _trend_score(close, resolved_spot, days, realized_vol)
        signal_values = {
            "trend": trend_score,
            "rsi": rsi_score,
            "sentiment": sentiment_score,
            "analyst": analyst_score,
            "options": options_score,
        }
        details = {
            "trend": trend_detail,
            "rsi": f"RSI 14={rsi_value:.1f}" if rsi_value is not None else "RSI unavailable",
            "sentiment": "FMP news/social/rating sentiment" if not sentiment_rows.empty else "No FMP sentiment payload",
            "analyst": analyst_detail,
            "options": options_detail,
        }
        weights = HORIZON_WEIGHTS[label]
        weighted_sum = 0.0
        active_weight = 0.0
        for signal, weight in weights.items():
            score = signal_values.get(signal)
            if score is None or pd.isna(score):
                continue
            weighted_sum += float(score) * weight
            active_weight += weight
            contribution_rows.append(
                {
                    "Horizon": label,
                    "Signal": signal.title(),
                    "Score": float(score),
                    "Weight": weight,
                    "Contribution": float(score) * weight,
                    "Detail": details[signal],
                }
            )
        final_score = weighted_sum / active_weight if active_weight else 0.0
        confidence = min(1.0, abs(final_score) * active_weight)
        horizon_rows.append(
            {
                "Horizon": label,
                "Days": days,
                "Direction": direction_label(final_score),
                "Score": final_score,
                "Confidence": confidence,
                "Expected Move": realized_vol * math.sqrt(days / 252),
                "Trend Score": trend_score,
                "RSI Score": rsi_score,
                "Sentiment Score": sentiment_score,
                "Analyst Score": analyst_score,
                "Options Score": options_score,
                "Data Coverage": active_weight,
                "Detail": _horizon_detail(final_score, active_weight, days),
            }
        )

    horizon_frame = pd.DataFrame(horizon_rows)
    contribution_frame = pd.DataFrame(contribution_rows)
    primary = _primary_direction(horizon_frame, preferred="1W")
    return DirectionalLensResult(
        symbol=symbol_key,
        as_of=as_of,
        spot=resolved_spot,
        horizon_frame=horizon_frame,
        contribution_frame=contribution_frame,
        sentiment_frame=sentiment_rows,
        summary={
            "primary_horizon": "1W",
            "primary_direction": primary.get("Direction", "Neutral"),
            "primary_score": float(primary.get("Score", 0.0) or 0.0),
            "primary_confidence": float(primary.get("Confidence", 0.0) or 0.0),
        },
    )


def add_strategy_direction_columns(
    frame: pd.DataFrame,
    lens: DirectionalLensResult | pd.DataFrame | None,
    *,
    horizon: str = "1W",
) -> pd.DataFrame:
    """Add payoff direction, model direction, and alignment columns."""

    if frame.empty:
        return frame
    out = frame.copy()
    horizon_frame = lens.horizon_frame if isinstance(lens, DirectionalLensResult) else lens
    model_row = _primary_direction(horizon_frame if isinstance(horizon_frame, pd.DataFrame) else pd.DataFrame(), preferred=horizon)
    model_direction = str(model_row.get("Direction") or "Neutral")
    model_score = float(model_row.get("Score") or 0.0)
    out["Payoff Direction"] = out["Strategy"].map(strategy_payoff_direction).fillna("Mixed")
    out["Model Direction"] = model_direction
    out["Signal Horizon"] = horizon
    out["Alignment"] = out["Payoff Direction"].map(lambda value: strategy_alignment(value, model_direction, model_score))
    return out


def strategy_payoff_direction(strategy: object) -> str:
    text = str(strategy or "").strip()
    if text in STRATEGY_DIRECTION_MAP:
        return STRATEGY_DIRECTION_MAP[text]
    lower = text.lower()
    if "bull" in lower or "call" in lower and "short" not in lower:
        return "Bullish"
    if "bear" in lower or "put" in lower and "short" not in lower:
        return "Bearish"
    if "condor" in lower or "butterfly" in lower or "calendar" in lower:
        return "Range-bound"
    return "Mixed"


def strategy_alignment(payoff_direction: object, model_direction: str, model_score: float = 0.0) -> str:
    payoff = str(payoff_direction or "").lower()
    model = str(model_direction or "Neutral").lower()
    strong_move = abs(float(model_score or 0.0)) >= 0.35
    if "bull" in payoff or "rebound" in payoff:
        return "Agree" if model == "bullish" else "Conflict" if model == "bearish" else "Mixed"
    if "bear" in payoff or "defensive" in payoff:
        return "Agree" if model == "bearish" else "Conflict" if model == "bullish" else "Mixed"
    if "range" in payoff or "pin" in payoff or "neutral" in payoff:
        return "Agree" if model == "neutral" else "Conflict" if strong_move else "Mixed"
    if "breakout" in payoff or "explosive" in payoff or "convex" in payoff:
        return "Agree" if strong_move else "Mixed"
    return "Mixed"


def direction_label(score: float) -> str:
    if score >= 0.20:
        return "Bullish"
    if score <= -0.20:
        return "Bearish"
    return "Neutral"


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy()
    if "Date" not in frame.columns:
        frame = frame.reset_index()
    columns = {str(column).lower().replace("_", " ").strip(): column for column in frame.columns}
    date_col = _first_existing(columns, ("date", "datetime", "timestamp", "index"))
    close_col = _first_existing(columns, ("close", "adj close", "adj_close", "price"))
    if date_col is None or close_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    out["Close"] = pd.to_numeric(frame[close_col], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    return out.set_index("Date").sort_index()


def _resolve_spot(history: pd.DataFrame, spot: float | None) -> float:
    parsed = _optional_float(spot)
    if parsed is not None and parsed > 0:
        return parsed
    close = pd.to_numeric(history.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else 0.0


def _trend_score(close: pd.Series, spot: float, days: int, realized_vol: float) -> tuple[float, str]:
    if close.empty or spot <= 0:
        return 0.0, "No price history"
    period = min(max(days, 1), max(len(close) - 1, 1))
    prior = float(close.iloc[-period - 1]) if len(close) > period else float(close.iloc[0])
    ret = (spot / prior) - 1 if prior > 0 else 0.0
    expected = max(realized_vol * math.sqrt(max(days, 1) / 252), 0.01)
    ma_window = min(20 if days <= 5 else 50, len(close))
    ma = float(close.tail(ma_window).mean()) if ma_window else spot
    ma_gap = (spot / ma) - 1 if ma > 0 else 0.0
    score = (0.70 * math.tanh(ret / expected)) + (0.30 * math.tanh(ma_gap / 0.05))
    return _clip(score), f"{days}D return={ret:.2%}; MA gap={ma_gap:.2%}"


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) <= window:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest = rsi.dropna()
    return float(latest.iloc[-1]) if not latest.empty else None


def _rsi_score(rsi: float | None) -> float:
    if rsi is None:
        return 0.0
    if rsi < 30:
        return _clip((35 - rsi) / 20)
    if rsi > 70:
        return _clip((65 - rsi) / 20)
    return _clip((50 - rsi) / 60)


def _realized_vol(close: pd.Series, window: int) -> float | None:
    returns = np.log(close / close.shift(1)).dropna().tail(window)
    if len(returns) < max(5, window // 3):
        return None
    value = float(returns.std() * math.sqrt(252))
    return value if math.isfinite(value) and value > 0 else None


def _sentiment_score(payload: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for source, raw in payload.items():
        for item in _iter_records(raw)[:50]:
            score = _record_sentiment_score(item)
            if score is None:
                continue
            rows.append(
                {
                    "Source": source.title(),
                    "Date": _first_text(item, ("date", "publishedDate", "publishedAt", "timestamp")),
                    "Title": _first_text(item, ("title", "headline", "newsTitle", "companyName", "symbol")),
                    "Sentiment": _first_text(item, ("sentiment", "stocktwitsSentiment", "twitterSentiment", "newGrade", "grade")),
                    "Score": score,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return 0.0, frame
    return _clip(float(pd.to_numeric(frame["Score"], errors="coerce").dropna().mean())), frame


def _record_sentiment_score(row: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for key, value in row.items():
        key_text = str(key).lower()
        if isinstance(value, str):
            text_score = _text_sentiment_score(value)
            if text_score is not None and any(token in key_text for token in ("sentiment", "grade", "rating", "action")):
                candidates.append(text_score)
            continue
        numeric = _optional_float(value)
        if numeric is None:
            continue
        if not any(token in key_text for token in ("sentiment", "score", "rating", "grade")):
            continue
        candidates.append(_normalize_sentiment_number(numeric))
    return _clip(float(np.nanmean(candidates))) if candidates else None


def _text_sentiment_score(value: str) -> float | None:
    text = value.lower().strip()
    if any(word in text for word in ("strong buy", "outperform", "overweight", "upgrade", "positive", "bullish", "buy")):
        return 0.70
    if any(word in text for word in ("strong sell", "underperform", "underweight", "downgrade", "negative", "bearish", "sell")):
        return -0.70
    if any(word in text for word in ("hold", "neutral", "market perform", "equal weight")):
        return 0.0
    return None


def _normalize_sentiment_number(value: float) -> float:
    if -1.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 5.0:
        return (value - 2.5) / 2.5
    if 0.0 <= value <= 100.0:
        return (value - 50.0) / 50.0
    return math.tanh(value)


def _analyst_score(price_targets: dict[str, Any], spot: float) -> tuple[float, str]:
    scores: list[float] = []
    consensus = _optional_float(price_targets.get("targetConsensus"))
    if consensus is not None and spot > 0:
        upside = (consensus / spot) - 1
        scores.append(_clip(upside / 0.25))
    recommendation_key = price_targets.get("recommendationKey")
    if recommendation_key:
        text_score = _text_sentiment_score(str(recommendation_key))
        if text_score is not None:
            scores.append(text_score)
    recommendation_mean = _optional_float(price_targets.get("recommendationMean"))
    if recommendation_mean is not None:
        scores.append(_clip((3.0 - recommendation_mean) / 2.0))
    if not scores:
        return 0.0, "No analyst target or recommendation data"
    detail = f"Consensus target={consensus:.2f}" if consensus is not None else "Recommendation data"
    return _clip(float(np.nanmean(scores))), detail


def _options_skew_score(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float) -> tuple[float, str]:
    call_iv = _nearest_iv(calls, spot * 1.05)
    put_iv = _nearest_iv(puts, spot * 0.95)
    if call_iv is None or put_iv is None:
        call_iv = _nearest_iv(calls, spot)
        put_iv = _nearest_iv(puts, spot)
    if call_iv is None or put_iv is None:
        return 0.0, "No option skew data"
    skew = put_iv - call_iv
    score = _clip(-skew / 0.10)
    return score, f"Put IV - call IV={skew:.1%}"


def _nearest_iv(chain: pd.DataFrame | None, target_strike: float) -> float | None:
    if chain is None or chain.empty or "strike" not in chain or "impliedVolatility" not in chain:
        return None
    frame = chain.copy()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["impliedVolatility"] = pd.to_numeric(frame["impliedVolatility"], errors="coerce")
    frame = frame.dropna(subset=["strike", "impliedVolatility"])
    frame = frame[frame["impliedVolatility"] > 0]
    if frame.empty:
        return None
    row = frame.assign(distance=(frame["strike"] - target_strike).abs()).sort_values("distance").iloc[0]
    return float(row["impliedVolatility"])


def _iter_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [dict(row) for row in value.to_dict("records")]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "historical", "content", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _primary_direction(frame: pd.DataFrame, *, preferred: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"Direction": "Neutral", "Score": 0.0, "Confidence": 0.0}
    subset = frame[frame["Horizon"].eq(preferred)] if "Horizon" in frame else pd.DataFrame()
    row = subset.iloc[0] if not subset.empty else frame.iloc[0]
    return dict(row)


def _horizon_detail(score: float, coverage: float, days: int) -> str:
    return f"{direction_label(score)} {days}D read; coverage={coverage:.0%}; score={score:.2f}"


def _as_of(history: pd.DataFrame) -> str:
    if history.empty:
        return ""
    index = pd.to_datetime(history.index, errors="coerce").dropna()
    if index.empty:
        return ""
    return index.max().date().isoformat()


def _first_existing(columns: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or not math.isfinite(parsed):
        return None
    return parsed


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))
