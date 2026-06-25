"""Options analytics, volatility, and scanner helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd


OptionRight = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class VolatilitySnapshot:
    historical_vol_21d: float
    historical_vol_63d: float
    ewma_vol: float
    parkinson_vol: float
    forecast_vol: float
    daily_return: float
    trend: str
    rsi_14: float
    atr_14: float


@dataclass(frozen=True, slots=True)
class OptionCandidate:
    strategy: str
    expiry: str
    strike: str
    debit_credit: float
    max_profit: float | None
    max_loss: float | None
    probability_of_profit: float
    expected_value: float
    edge: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OptionLeg:
    option_type: OptionRight
    strike: float
    premium: float
    quantity: int
    expiry: str | None = None


@dataclass(frozen=True, slots=True)
class StrategyScore:
    scanner: str
    strategy: str
    category: str
    directional_bias: str
    volatility_bias: str
    complexity: str
    score: float
    raw_score: float
    reason: str


@dataclass(frozen=True, slots=True)
class StrategySimulation:
    probability_of_profit: float
    expected_value: float
    value_at_risk_95: float
    worst_case: float
    best_case: float
    terminal_prices: np.ndarray
    profits: np.ndarray


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionRight = "call",
) -> float:
    if spot <= 0 or strike <= 0 or volatility <= 0:
        return 0.0
    if time_to_expiry <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    if option_type == "call":
        return spot * normal_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * normal_cdf(d2)
    return strike * math.exp(-rate * time_to_expiry) * normal_cdf(-d2) - spot * normal_cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionRight = "call",
) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    price = black_scholes_price(spot, strike, time_to_expiry, rate, volatility, option_type)
    delta = normal_cdf(d1) if option_type == "call" else normal_cdf(d1) - 1
    gamma = normal_pdf(d1) / (spot * volatility * sqrt_t)
    theta = -((spot * normal_pdf(d1) * volatility) / (2 * sqrt_t)) / 365
    vega = spot * normal_pdf(d1) * sqrt_t / 100
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }


def solve_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionRight = "call",
) -> float:
    if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return 0.0

    low, high = 0.001, 3.0
    mid = 0.30
    for _ in range(60):
        mid = (low + high) / 2
        estimate = black_scholes_price(spot, strike, time_to_expiry, rate, mid, option_type)
        if estimate > market_price:
            high = mid
        else:
            low = mid
    return float(mid)


def mid_price(row: pd.Series | dict[str, Any]) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        if mid > 0:
            return mid
    return last if last > 0 else 0.0


def normalize_option_chain(chain: pd.DataFrame, option_type: OptionRight) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    out = chain.copy()
    out["option_type"] = option_type
    out["mid"] = out.apply(mid_price, axis=1)
    out["spread"] = pd.to_numeric(out.get("ask", 0), errors="coerce").fillna(0) - pd.to_numeric(
        out.get("bid", 0), errors="coerce"
    ).fillna(0)
    out = out[pd.to_numeric(out["strike"], errors="coerce").notna()]
    out = out[out["mid"] > 0]
    return out


def choose_expiration(expirations: list[str] | tuple[str, ...], min_days: int, today: date | None = None) -> str | None:
    anchor = today or date.today()
    for expiry in expirations:
        days = (pd.to_datetime(expiry).date() - anchor).days
        if days >= min_days:
            return expiry
    return expirations[-1] if expirations else None


def time_to_expiry(expiry: str, today: date | None = None) -> float:
    anchor = today or date.today()
    days = max((pd.to_datetime(expiry).date() - anchor).days, 1)
    return days / 365.0


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace({0.0: pd.NA})
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi.dropna().iloc[-1], 50.0) if not rsi.dropna().empty else 50.0


def _atr(history: pd.DataFrame, window: int = 14) -> float:
    if history.empty or not {"High", "Low", "Close"}.issubset(history.columns):
        return 0.0
    high_low = history["High"] - history["Low"]
    high_close = (history["High"] - history["Close"].shift(1)).abs()
    low_close = (history["Low"] - history["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean().dropna()
    return safe_float(atr.iloc[-1]) if not atr.empty else 0.0


def volatility_snapshot(history: pd.DataFrame) -> VolatilitySnapshot:
    if history.empty or "Close" not in history.columns:
        return VolatilitySnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unknown", 50.0, 0.0)

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    hist_21 = safe_float(log_returns.tail(21).std()) * math.sqrt(252)
    hist_63 = safe_float(log_returns.tail(63).std()) * math.sqrt(252)
    ewma = safe_float(log_returns.ewm(alpha=1 - 0.94).std().iloc[-1]) * math.sqrt(252)

    if {"High", "Low"}.issubset(history.columns):
        high = pd.to_numeric(history["High"], errors="coerce")
        low = pd.to_numeric(history["Low"], errors="coerce")
        parkinson_daily = ((np.log(high / low)) ** 2) / (4 * math.log(2))
        parkinson_vol = safe_float(np.sqrt(parkinson_daily.tail(63).mean())) * math.sqrt(252)
    else:
        parkinson_vol = hist_63

    forecast = np.nanmean([value for value in (hist_21, hist_63, ewma, parkinson_vol) if value > 0])
    sma_20 = close.rolling(20).mean()
    sma_60 = close.rolling(60).mean()
    if len(close) >= 60 and close.iloc[-1] > sma_20.iloc[-1] > sma_60.iloc[-1]:
        trend = "bullish"
    elif len(close) >= 60 and close.iloc[-1] < sma_20.iloc[-1] < sma_60.iloc[-1]:
        trend = "bearish"
    elif len(close) >= 20 and close.iloc[-1] < sma_20.iloc[-1]:
        trend = "pullback"
    else:
        trend = "choppy"

    daily_return = (close.iloc[-1] / close.iloc[-2]) - 1 if len(close) > 1 else 0.0
    return VolatilitySnapshot(
        historical_vol_21d=float(hist_21),
        historical_vol_63d=float(hist_63),
        ewma_vol=float(ewma),
        parkinson_vol=float(parkinson_vol),
        forecast_vol=float(0.0 if pd.isna(forecast) else forecast),
        daily_return=float(daily_return),
        trend=trend,
        rsi_14=_rsi(close),
        atr_14=_atr(history),
    )


def simulate_terminal_prices(
    spot: float,
    days_to_hold: int,
    volatility: float,
    *,
    rate: float = 0.045,
    simulations: int = 5000,
    seed: int | None = 42,
) -> np.ndarray:
    if simulations <= 0 or spot <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    dt = max(days_to_hold, 1) / 365.0
    shocks = rng.normal(0, 1, simulations)
    return spot * np.exp((rate - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * shocks)


def option_payoff(terminal_prices: np.ndarray, strike: float, option_type: OptionRight) -> np.ndarray:
    if option_type == "call":
        return np.maximum(terminal_prices - strike, 0)
    return np.maximum(strike - terminal_prices, 0)


def simulate_single_option(
    *,
    spot: float,
    strike: float,
    premium: float,
    option_type: OptionRight,
    side: Literal["long", "short"],
    days_to_hold: int,
    volatility: float,
    simulations: int = 5000,
) -> StrategySimulation:
    terminals = simulate_terminal_prices(spot, days_to_hold, volatility, simulations=simulations)
    intrinsic = option_payoff(terminals, strike, option_type)
    profits = intrinsic - premium if side == "long" else premium - intrinsic
    return summarize_simulation(terminals, profits)


def simulate_multi_leg_options(
    *,
    spot: float,
    legs: list[OptionLeg],
    days_to_hold: int,
    volatility: float,
    rate: float = 0.045,
    simulations: int = 3000,
) -> StrategySimulation:
    terminals = simulate_terminal_prices(
        spot,
        days_to_hold,
        volatility,
        rate=rate,
        simulations=simulations,
    )
    entry_cost = sum(leg.quantity * leg.premium for leg in legs)
    terminal_value = np.zeros_like(terminals)
    for leg in legs:
        terminal_value += leg.quantity * option_payoff(terminals, leg.strike, leg.option_type)
    profits = terminal_value - entry_cost
    return summarize_simulation(terminals, profits)


def summarize_simulation(terminal_prices: np.ndarray, profits: np.ndarray) -> StrategySimulation:
    if len(profits) == 0:
        return StrategySimulation(0.0, 0.0, 0.0, 0.0, 0.0, terminal_prices, profits)
    return StrategySimulation(
        probability_of_profit=float(np.mean(profits > 0)),
        expected_value=float(np.mean(profits)),
        value_at_risk_95=float(np.percentile(profits, 5)),
        worst_case=float(np.min(profits)),
        best_case=float(np.max(profits)),
        terminal_prices=terminal_prices,
        profits=profits,
    )


def _normalized_sorted_chain(chain: pd.DataFrame, option_type: OptionRight) -> pd.DataFrame:
    normalized = normalize_option_chain(chain, option_type)
    if normalized.empty:
        return normalized
    normalized = normalized.copy()
    normalized["strike"] = pd.to_numeric(normalized["strike"], errors="coerce")
    normalized = normalized.dropna(subset=["strike", "mid"])
    return normalized.sort_values("strike").reset_index(drop=True)


def _entry_cost(legs: list[OptionLeg]) -> float:
    return float(sum(leg.quantity * leg.premium for leg in legs))


def _legs_text(legs: list[OptionLeg]) -> str:
    chunks = []
    for leg in legs:
        side = "+" if leg.quantity > 0 else "-"
        right = "C" if leg.option_type == "call" else "P"
        chunks.append(f"{side}{abs(leg.quantity)}x {leg.strike:g}{right}")
    return " / ".join(chunks)


def _leg_payloads(legs: list[OptionLeg], *, default_expiry: str) -> list[dict[str, Any]]:
    return [
        {
            "option_type": leg.option_type,
            "strike": leg.strike,
            "premium": leg.premium,
            "quantity": leg.quantity,
            "expiry": leg.expiry or default_expiry,
        }
        for leg in legs
    ]


def _candidate_record(
    *,
    strategy: str,
    expiry: str,
    legs: list[OptionLeg],
    simulation: StrategySimulation,
    max_profit: float | None,
    max_loss: float | None,
    edge: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = _entry_cost(legs)
    row: dict[str, Any] = {
        "Strategy": strategy,
        "Expiry": expiry,
        "Structure": _legs_text(legs),
        "Debit/Credit": -entry * 100,
        "Max Profit": None if max_profit is None else max_profit * 100,
        "Max Loss": None if max_loss is None else max_loss * 100,
        "PoP": simulation.probability_of_profit,
        "EV": simulation.expected_value * 100,
        "VaR 95": simulation.value_at_risk_95 * 100,
        "Edge": edge * 100,
        "Legs": _leg_payloads(legs, default_expiry=expiry),
    }
    if extra:
        row.update(extra)
    return row


def _best_rows(rows: list[dict[str, Any]], limit: int = 10) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["EV", "PoP"], ascending=False).head(limit).reset_index(drop=True)


def scan_long_options(
    chain: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    option_type: OptionRight,
    budget: float,
    days_to_hold: int,
    forecast_vol: float,
    rate: float = 0.045,
    today: date | None = None,
) -> pd.DataFrame:
    normalized = normalize_option_chain(chain, option_type)
    if normalized.empty:
        return pd.DataFrame()

    t = time_to_expiry(expiry, today)
    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        strike = safe_float(row["strike"])
        premium = safe_float(row["mid"])
        cost = premium * 100
        if cost <= 0 or cost > budget:
            continue
        if option_type == "call" and not (spot * 0.85 <= strike <= spot * 1.40):
            continue
        if option_type == "put" and not (spot * 0.70 <= strike <= spot * 1.05):
            continue
        market_iv = solve_implied_volatility(premium, spot, strike, t, rate, option_type)
        sim = simulate_single_option(
            spot=spot,
            strike=strike,
            premium=premium,
            option_type=option_type,
            side="long",
            days_to_hold=days_to_hold,
            volatility=forecast_vol,
        )
        rows.append(
            {
                "Strategy": f"Long {option_type.title()}",
                "Expiry": expiry,
                "Strike": strike,
                "Structure": _legs_text([OptionLeg(option_type, strike, premium, 1, expiry)]),
                "Debit/Credit": -cost,
                "Max Profit": None,
                "Max Loss": -cost,
                "PoP": sim.probability_of_profit,
                "EV": sim.expected_value * 100,
                "IV Edge": forecast_vol - market_iv,
                "Mid": premium,
                "Market IV": market_iv,
                "Legs": _leg_payloads(
                    [OptionLeg(option_type, strike, premium, 1, expiry)],
                    default_expiry=expiry,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["EV", "PoP"], ascending=False).head(10) if rows else pd.DataFrame()


def scan_cash_secured_puts(
    puts: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    max_collateral: float,
    days_to_hold: int,
    forecast_vol: float,
    rate: float = 0.045,
    today: date | None = None,
) -> pd.DataFrame:
    normalized = normalize_option_chain(puts, "put")
    if normalized.empty:
        return pd.DataFrame()

    t = time_to_expiry(expiry, today)
    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        strike = safe_float(row["strike"])
        premium = safe_float(row["mid"])
        collateral = strike * 100
        if collateral <= 0 or collateral > max_collateral:
            continue
        if not (spot * 0.80 <= strike <= spot * 1.00):
            continue
        market_iv = solve_implied_volatility(premium, spot, strike, t, rate, "put")
        sim = simulate_single_option(
            spot=spot,
            strike=strike,
            premium=premium,
            option_type="put",
            side="short",
            days_to_hold=days_to_hold,
            volatility=forecast_vol,
        )
        rows.append(
            {
                "Strategy": "Cash-Secured Put",
                "Expiry": expiry,
                "Strike": strike,
                "Structure": _legs_text([OptionLeg("put", strike, premium, -1, expiry)]),
                "Debit/Credit": premium * 100,
                "Max Profit": premium * 100,
                "Max Loss": -(collateral - premium * 100),
                "PoP": sim.probability_of_profit,
                "EV": sim.expected_value * 100,
                "IV Edge": market_iv - forecast_vol,
                "Mid": premium,
                "Market IV": market_iv,
                "Legs": _leg_payloads(
                    [OptionLeg("put", strike, premium, -1, expiry)],
                    default_expiry=expiry,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["EV", "PoP"], ascending=False).head(10) if rows else pd.DataFrame()


def scan_vertical_spreads(
    chain: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    spread_type: Literal["bull_call", "bear_put"],
    budget: float,
    days_to_hold: int,
    forecast_vol: float,
    rate: float = 0.045,
    simulations: int = 3000,
) -> pd.DataFrame:
    option_type: OptionRight = "call" if spread_type == "bull_call" else "put"
    normalized = _normalized_sorted_chain(chain, option_type)
    if normalized.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, long_row in normalized.iterrows():
        long_strike = safe_float(long_row["strike"])
        long_mid = safe_float(long_row["mid"])
        if spread_type == "bull_call":
            if not (spot * 0.85 <= long_strike <= spot * 1.10):
                continue
            short_candidates = normalized[
                (normalized["strike"] > long_strike) & (normalized["strike"] <= spot * 1.35)
            ].head(8)
            strategy = "Bull Call Spread"
        else:
            if not (spot * 0.90 <= long_strike <= spot * 1.15):
                continue
            short_candidates = normalized[
                (normalized["strike"] < long_strike) & (normalized["strike"] >= spot * 0.65)
            ].tail(8)
            strategy = "Bear Put Spread"

        for _, short_row in short_candidates.iterrows():
            short_strike = safe_float(short_row["strike"])
            short_mid = safe_float(short_row["mid"])
            debit = long_mid - short_mid
            width = abs(short_strike - long_strike)
            if debit <= 0 or width <= 0 or debit * 100 > budget:
                continue

            legs = [
                OptionLeg(option_type, long_strike, long_mid, 1),
                OptionLeg(option_type, short_strike, short_mid, -1),
            ]
            simulation = simulate_multi_leg_options(
                spot=spot,
                legs=legs,
                days_to_hold=days_to_hold,
                volatility=forecast_vol,
                rate=rate,
                simulations=simulations,
            )
            rows.append(
                _candidate_record(
                    strategy=strategy,
                    expiry=expiry,
                    legs=legs,
                    simulation=simulation,
                    max_profit=max(width - debit, 0.0),
                    max_loss=-debit,
                    extra={"Width": width},
                )
            )
    return _best_rows(rows)


def scan_calendar_spreads(
    near_calls: pd.DataFrame,
    far_calls: pd.DataFrame,
    *,
    spot: float,
    near_expiry: str,
    far_expiry: str,
    budget: float,
    forecast_vol: float,
    rate: float = 0.045,
    today: date | None = None,
    simulations: int = 3000,
) -> pd.DataFrame:
    near = _normalized_sorted_chain(near_calls, "call")
    far = _normalized_sorted_chain(far_calls, "call")
    if near.empty or far.empty:
        return pd.DataFrame()

    anchor = today or date.today()
    near_days = max((pd.to_datetime(near_expiry).date() - anchor).days, 1)
    t_near = near_days / 365.0
    t_far = time_to_expiry(far_expiry, anchor)
    remaining = max(t_far - t_near, 1 / 365)
    near_prices = near.set_index("strike")["mid"]
    far_prices = far.set_index("strike")["mid"]
    common_strikes = sorted(set(near_prices.index).intersection(set(far_prices.index)))
    terminals = simulate_terminal_prices(
        spot,
        near_days,
        forecast_vol,
        rate=rate,
        simulations=simulations,
    )

    rows: list[dict[str, Any]] = []
    for strike in common_strikes:
        strike = safe_float(strike)
        if not (spot * 0.85 <= strike <= spot * 1.15):
            continue
        near_mid = safe_float(near_prices.loc[strike])
        far_mid = safe_float(far_prices.loc[strike])
        debit = far_mid - near_mid
        if debit <= 0 or debit * 100 > budget:
            continue

        long_far_value = np.array(
            [black_scholes_price(price, strike, remaining, rate, forecast_vol, "call") for price in terminals]
        )
        short_near_value = option_payoff(terminals, strike, "call")
        simulation = summarize_simulation(terminals, long_far_value - short_near_value - debit)
        fair_near = black_scholes_price(spot, strike, t_near, rate, forecast_vol, "call")
        fair_far = black_scholes_price(spot, strike, t_far, rate, forecast_vol, "call")
        edge = (fair_far - fair_near) - debit
        legs = [
            OptionLeg("call", strike, far_mid, 1, far_expiry),
            OptionLeg("call", strike, near_mid, -1, near_expiry),
        ]
        rows.append(
            _candidate_record(
                strategy="Calendar Spread",
                expiry=f"{near_expiry} / {far_expiry}",
                legs=legs,
                simulation=simulation,
                max_profit=None,
                max_loss=-debit,
                edge=edge,
                extra={"Near Expiry": near_expiry, "Far Expiry": far_expiry},
            )
        )
    return _best_rows(rows)


def scan_iron_condors(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    max_risk: float,
    days_to_hold: int,
    forecast_vol: float,
    rate: float = 0.045,
    simulations: int = 2500,
) -> pd.DataFrame:
    call_chain = _normalized_sorted_chain(calls, "call")
    put_chain = _normalized_sorted_chain(puts, "put")
    if call_chain.empty or put_chain.empty:
        return pd.DataFrame()

    short_puts = put_chain[(put_chain["strike"] >= spot * 0.80) & (put_chain["strike"] < spot)].tail(8)
    short_calls = call_chain[(call_chain["strike"] > spot) & (call_chain["strike"] <= spot * 1.20)].head(8)

    rows: list[dict[str, Any]] = []
    for _, short_put in short_puts.iterrows():
        sp_strike = safe_float(short_put["strike"])
        long_puts = put_chain[put_chain["strike"] < sp_strike].tail(4)
        for _, short_call in short_calls.iterrows():
            sc_strike = safe_float(short_call["strike"])
            long_calls = call_chain[call_chain["strike"] > sc_strike].head(4)
            for _, long_put in long_puts.iterrows():
                lp_strike = safe_float(long_put["strike"])
                for _, long_call in long_calls.iterrows():
                    lc_strike = safe_float(long_call["strike"])
                    credit = (
                        safe_float(short_put["mid"])
                        + safe_float(short_call["mid"])
                        - safe_float(long_put["mid"])
                        - safe_float(long_call["mid"])
                    )
                    width = max(sp_strike - lp_strike, lc_strike - sc_strike)
                    if credit <= 0 or width <= 0:
                        continue
                    max_loss = -(width - credit)
                    if abs(max_loss) * 100 > max_risk:
                        continue
                    legs = [
                        OptionLeg("put", lp_strike, safe_float(long_put["mid"]), 1),
                        OptionLeg("put", sp_strike, safe_float(short_put["mid"]), -1),
                        OptionLeg("call", sc_strike, safe_float(short_call["mid"]), -1),
                        OptionLeg("call", lc_strike, safe_float(long_call["mid"]), 1),
                    ]
                    simulation = simulate_multi_leg_options(
                        spot=spot,
                        legs=legs,
                        days_to_hold=days_to_hold,
                        volatility=forecast_vol,
                        rate=rate,
                        simulations=simulations,
                    )
                    rows.append(
                        _candidate_record(
                            strategy="Iron Condor",
                            expiry=expiry,
                            legs=legs,
                            simulation=simulation,
                            max_profit=credit,
                            max_loss=max_loss,
                            extra={"Width": width},
                        )
                    )
    return _best_rows(rows)


def scan_call_butterflies(
    calls: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    budget: float,
    days_to_hold: int,
    forecast_vol: float,
    rate: float = 0.045,
    simulations: int = 2500,
) -> pd.DataFrame:
    call_chain = _normalized_sorted_chain(calls, "call")
    if call_chain.empty or len(call_chain) < 3:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for middle_idx in range(1, len(call_chain) - 1):
        middle = call_chain.iloc[middle_idx]
        middle_strike = safe_float(middle["strike"])
        if not (spot * 0.85 <= middle_strike <= spot * 1.15):
            continue
        max_wing = min(5, middle_idx, len(call_chain) - middle_idx - 1)
        for wing in range(1, max_wing + 1):
            lower = call_chain.iloc[middle_idx - wing]
            upper = call_chain.iloc[middle_idx + wing]
            lower_strike = safe_float(lower["strike"])
            upper_strike = safe_float(upper["strike"])
            wing_width = min(middle_strike - lower_strike, upper_strike - middle_strike)
            debit = safe_float(lower["mid"]) - (2 * safe_float(middle["mid"])) + safe_float(upper["mid"])
            if debit <= 0 or wing_width <= 0 or debit * 100 > budget:
                continue
            legs = [
                OptionLeg("call", lower_strike, safe_float(lower["mid"]), 1),
                OptionLeg("call", middle_strike, safe_float(middle["mid"]), -2),
                OptionLeg("call", upper_strike, safe_float(upper["mid"]), 1),
            ]
            simulation = simulate_multi_leg_options(
                spot=spot,
                legs=legs,
                days_to_hold=days_to_hold,
                volatility=forecast_vol,
                rate=rate,
                simulations=simulations,
            )
            rows.append(
                _candidate_record(
                    strategy="Call Butterfly",
                    expiry=expiry,
                    legs=legs,
                    simulation=simulation,
                    max_profit=max(wing_width - debit, 0.0),
                    max_loss=-debit,
                    extra={"Width": wing_width},
                )
            )
    return _best_rows(rows)


def scan_ratio_spreads(
    chain: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    option_type: OptionRight,
    budget: float,
    days_to_hold: int,
    forecast_vol: float,
    ratio: int = 2,
    rate: float = 0.045,
    simulations: int = 2500,
) -> pd.DataFrame:
    normalized = _normalized_sorted_chain(chain, option_type)
    if normalized.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, long_row in normalized.iterrows():
        long_strike = safe_float(long_row["strike"])
        if option_type == "call":
            if not (spot * 0.80 <= long_strike <= spot * 1.05):
                continue
            short_candidates = normalized[
                (normalized["strike"] > long_strike) & (normalized["strike"] <= spot * 1.30)
            ].head(8)
            strategy = "Call Ratio Spread"
        else:
            if not (spot * 0.95 <= long_strike <= spot * 1.20):
                continue
            short_candidates = normalized[
                (normalized["strike"] < long_strike) & (normalized["strike"] >= spot * 0.65)
            ].tail(8)
            strategy = "Put Ratio Spread"

        for _, short_row in short_candidates.iterrows():
            short_strike = safe_float(short_row["strike"])
            entry = safe_float(long_row["mid"]) - (ratio * safe_float(short_row["mid"]))
            if entry > 0 and entry * 100 > budget:
                continue
            width = abs(short_strike - long_strike)
            if width <= 0:
                continue
            legs = [
                OptionLeg(option_type, long_strike, safe_float(long_row["mid"]), 1),
                OptionLeg(option_type, short_strike, safe_float(short_row["mid"]), -ratio),
            ]
            simulation = simulate_multi_leg_options(
                spot=spot,
                legs=legs,
                days_to_hold=days_to_hold,
                volatility=forecast_vol,
                rate=rate,
                simulations=simulations,
            )
            rows.append(
                _candidate_record(
                    strategy=strategy,
                    expiry=expiry,
                    legs=legs,
                    simulation=simulation,
                    max_profit=max(width - entry, 0.0),
                    max_loss=simulation.worst_case,
                    extra={"Ratio": ratio, "Width": width},
                )
            )
    return _best_rows(rows)


def scan_backspreads(
    chain: pd.DataFrame,
    *,
    spot: float,
    expiry: str,
    option_type: OptionRight,
    budget: float,
    days_to_hold: int,
    forecast_vol: float,
    ratio: int = 2,
    rate: float = 0.045,
    simulations: int = 2500,
) -> pd.DataFrame:
    normalized = _normalized_sorted_chain(chain, option_type)
    if normalized.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, short_row in normalized.iterrows():
        short_strike = safe_float(short_row["strike"])
        if option_type == "call":
            if not (spot * 0.90 <= short_strike <= spot * 1.05):
                continue
            long_candidates = normalized[
                (normalized["strike"] > short_strike) & (normalized["strike"] <= spot * 1.35)
            ].head(8)
            strategy = "Call Backspread"
        else:
            if not (spot * 0.95 <= short_strike <= spot * 1.15):
                continue
            long_candidates = normalized[
                (normalized["strike"] < short_strike) & (normalized["strike"] >= spot * 0.65)
            ].tail(8)
            strategy = "Put Backspread"

        for _, long_row in long_candidates.iterrows():
            long_strike = safe_float(long_row["strike"])
            entry = (ratio * safe_float(long_row["mid"])) - safe_float(short_row["mid"])
            if entry > 0 and entry * 100 > budget:
                continue
            width = abs(long_strike - short_strike)
            if width <= 0:
                continue
            legs = [
                OptionLeg(option_type, short_strike, safe_float(short_row["mid"]), -1),
                OptionLeg(option_type, long_strike, safe_float(long_row["mid"]), ratio),
            ]
            simulation = simulate_multi_leg_options(
                spot=spot,
                legs=legs,
                days_to_hold=days_to_hold,
                volatility=forecast_vol,
                rate=rate,
                simulations=simulations,
            )
            max_loss_at_pin = -(width + entry)
            rows.append(
                _candidate_record(
                    strategy=strategy,
                    expiry=expiry,
                    legs=legs,
                    simulation=simulation,
                    max_profit=None,
                    max_loss=min(max_loss_at_pin, 0.0),
                    extra={"Ratio": ratio, "Width": width},
                )
            )
    return _best_rows(rows)


def historical_holding_returns(history: pd.DataFrame, days_to_hold: int) -> pd.Series:
    if history.empty or "Close" not in history.columns:
        return pd.Series(dtype=float)
    trading_days = max(int(days_to_hold * (252 / 365)), 1)
    return history["Close"].pct_change(periods=trading_days).dropna()


def historical_odds(returns: pd.Series, target_pct: float, direction: Literal["up", "down", "inside"] = "up") -> float:
    if returns.empty:
        return 0.0
    if direction == "up":
        hits = returns >= target_pct
    elif direction == "down":
        hits = returns <= target_pct
    else:
        hits = (returns >= -abs(target_pct)) & (returns <= abs(target_pct))
    return float(hits.mean())


STRATEGY_CATALOG: dict[str, dict[str, str]] = {
    "scan_long_calls": {
        "strategy": "Long Calls",
        "category": "Directional",
        "directional_bias": "Bullish",
        "volatility_bias": "Long vol",
        "complexity": "Simple",
    },
    "scan_long_puts": {
        "strategy": "Long Puts",
        "category": "Directional hedge",
        "directional_bias": "Bearish",
        "volatility_bias": "Long vol",
        "complexity": "Simple",
    },
    "scan_long_volatility": {
        "strategy": "Long Straddle / Strangle",
        "category": "Volatility",
        "directional_bias": "Breakout",
        "volatility_bias": "Long vol",
        "complexity": "Intermediate",
    },
    "scan_short_puts": {
        "strategy": "Cash-Secured Puts",
        "category": "Income",
        "directional_bias": "Bullish / neutral",
        "volatility_bias": "Short vol",
        "complexity": "Simple",
    },
    "scan_iron_condors": {
        "strategy": "Iron Condors",
        "category": "Income",
        "directional_bias": "Range-bound",
        "volatility_bias": "Short vol",
        "complexity": "Intermediate",
    },
    "scan_short_volatility": {
        "strategy": "Short Straddle / Strangle",
        "category": "Volatility",
        "directional_bias": "Range-bound",
        "volatility_bias": "Short vol",
        "complexity": "Advanced",
    },
    "scan_bull_call_spreads": {
        "strategy": "Bull Call Spreads",
        "category": "Defined-risk directional",
        "directional_bias": "Bullish",
        "volatility_bias": "Neutral / short vol",
        "complexity": "Intermediate",
    },
    "scan_bear_put_spreads": {
        "strategy": "Bear Put Spreads",
        "category": "Defined-risk directional",
        "directional_bias": "Bearish",
        "volatility_bias": "Neutral / short vol",
        "complexity": "Intermediate",
    },
    "scan_call_backspreads": {
        "strategy": "Call Backspreads",
        "category": "Tail / convexity",
        "directional_bias": "Explosive bullish",
        "volatility_bias": "Long vol",
        "complexity": "Advanced",
    },
    "scan_put_backspreads": {
        "strategy": "Put Backspreads",
        "category": "Tail hedge",
        "directional_bias": "Explosive bearish",
        "volatility_bias": "Long vol",
        "complexity": "Advanced",
    },
    "scan_collars": {
        "strategy": "Collars",
        "category": "Portfolio hedge",
        "directional_bias": "Defensive",
        "volatility_bias": "Mixed",
        "complexity": "Intermediate",
    },
    "scan_calendar_spreads": {
        "strategy": "Calendar Spreads",
        "category": "Term structure",
        "directional_bias": "Pin / neutral",
        "volatility_bias": "Term long vol",
        "complexity": "Intermediate",
    },
    "scan_deep_value_leaps": {
        "strategy": "Deep Value LEAPS",
        "category": "Long-term convexity",
        "directional_bias": "Capitulation rebound",
        "volatility_bias": "Long vol",
        "complexity": "Advanced",
    },
}


def _bounded_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _strategy_reason(
    *,
    scanner: str,
    momentum_z: float,
    vrp_z: float,
    rsi_oversold: float,
    rsi_overbought: float,
    dxy_macro_veto: bool,
) -> str:
    notes: list[str] = []
    if vrp_z > 0.10:
        notes.append("market IV is rich versus forecast")
    elif vrp_z < -0.10:
        notes.append("forecast vol is above market IV")
    else:
        notes.append("vol premium is near neutral")

    if momentum_z > 1.0:
        notes.append("upside momentum is strong")
    elif momentum_z < -1.0:
        notes.append("downside momentum is strong")
    else:
        notes.append("momentum is range-like")

    if rsi_oversold > 0:
        notes.append("RSI is oversold")
    elif rsi_overbought > 0:
        notes.append("RSI is overbought")

    if dxy_macro_veto and scanner in {"scan_long_calls", "scan_short_puts", "scan_call_backspreads"}:
        notes.append("macro veto penalized bullish risk")

    return "; ".join(notes)


def score_option_strategies(
    *,
    spot: float,
    moving_average_20: float,
    rolling_std_20: float,
    rsi_14: float,
    market_iv: float,
    forecast_vol: float,
    target_beta: float = 0.5,
    dxy_macro_veto: bool = False,
) -> pd.DataFrame:
    """Rank option strategy families using the old desk's routing logic.

    This does not select contracts or place trades. It answers the prior question:
    which strategy family fits the current volatility, momentum, and risk posture?
    """

    if spot <= 0:
        return pd.DataFrame()

    market_iv = max(float(market_iv or 0.0), 0.001)
    forecast_vol = max(float(forecast_vol or 0.0), 0.001)
    rolling_std_20 = max(float(rolling_std_20 or 0.0), 0.0)
    target_beta = max(0.0, min(1.5, float(target_beta)))

    vrp = market_iv - forecast_vol
    vrp_z = vrp / market_iv
    momentum_z = (spot - moving_average_20) / rolling_std_20 if rolling_std_20 > 0 else 0.0
    rsi_oversold = max(0.0, 40.0 - float(rsi_14 or 50.0)) / 40.0
    rsi_overbought = max(0.0, float(rsi_14 or 50.0) - 60.0) / 40.0

    is_capitulating = momentum_z < -2.0 and rsi_oversold > 0.5
    capitulation_score = 125 + (abs(momentum_z) * 5) + (rsi_oversold * 10) if is_capitulating else 10

    scores = {
        "scan_long_calls": 50 + (momentum_z * 15) - (vrp_z * 50) + (rsi_oversold * 20) - ((1 - target_beta) * 30),
        "scan_long_puts": 50 - (momentum_z * 15) - (vrp_z * 50) + (rsi_overbought * 20) + ((1 - target_beta) * 30),
        "scan_long_volatility": 50 - abs(momentum_z * 10) - (vrp_z * 80),
        "scan_short_puts": 50 + (momentum_z * 10) + (vrp_z * 60) + (rsi_oversold * 15) - ((1 - target_beta) * 20),
        "scan_iron_condors": 50 - abs(momentum_z * 20) + (vrp_z * 80),
        "scan_short_volatility": 50 - abs(momentum_z * 20) + (vrp_z * 100),
        "scan_bull_call_spreads": 50 + (momentum_z * 20) + (vrp_z * 20) - ((1 - target_beta) * 20),
        "scan_bear_put_spreads": 50 - (momentum_z * 20) + (vrp_z * 20) + ((1 - target_beta) * 20),
        "scan_call_backspreads": 50 + (momentum_z * 25) - (vrp_z * 30) - (rsi_overbought * 20),
        "scan_put_backspreads": 50 - (momentum_z * 25) - (vrp_z * 30) - (rsi_oversold * 20) + ((1 - target_beta) * 20),
        "scan_collars": 50 + (momentum_z * 10) + (vrp_z * 10) + ((1 - target_beta) * 40),
        "scan_calendar_spreads": 50 - abs(momentum_z * 15) + (vrp_z * 30),
        "scan_deep_value_leaps": capitulation_score,
    }

    if dxy_macro_veto:
        for scanner in ("scan_long_calls", "scan_short_puts", "scan_call_backspreads"):
            scores[scanner] -= 50

    rows: list[dict[str, Any]] = []
    for scanner, raw_score in scores.items():
        catalog = STRATEGY_CATALOG[scanner]
        rows.append(
            {
                "Scanner": scanner,
                "Strategy": catalog["strategy"],
                "Category": catalog["category"],
                "Directional Bias": catalog["directional_bias"],
                "Volatility Bias": catalog["volatility_bias"],
                "Complexity": catalog["complexity"],
                "Score": _bounded_score(raw_score),
                "Raw Score": float(raw_score),
                "Reason": _strategy_reason(
                    scanner=scanner,
                    momentum_z=momentum_z,
                    vrp_z=vrp_z,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought,
                    dxy_macro_veto=dxy_macro_veto,
                ),
                "VRP": float(vrp),
                "Momentum Z": float(momentum_z),
            }
        )

    return pd.DataFrame(rows).sort_values(["Raw Score", "Strategy"], ascending=[False, True]).reset_index(drop=True)


def format_scanner_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("PoP", "IV Edge", "Market IV"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("Debit/Credit", "Max Profit", "Max Loss", "EV", "Mid", "Strike", "VaR 95", "Edge", "Width"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
