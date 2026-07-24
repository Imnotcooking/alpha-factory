"""Causal, asset-aware liquidity eligibility for research datasets.

Liquidity eligibility is a hard execution gate, not a universe-ranking signal.
The gate preserves the full row grid, records why each row is or is not
eligible, and uses only completed sessions so intraday research cannot see the
rest of the current day's volume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from oqp.contracts.market_vertical import (
    ASSET_TAXONOMY,
    MarketVertical,
    normalize_market_vertical,
)
from oqp.data import InstrumentMaster
LIQUIDITY_POLICY_VERSION = "1.0"
DEFAULT_LIQUIDITY_LOOKBACK_SESSIONS = 20
DEFAULT_LIQUIDITY_MIN_OBSERVATIONS = 15

ELIGIBLE = "eligible"
WARMUP = "warmup_insufficient_history"
MISSING_MARKET_DATA = "missing_required_market_data"
MISSING_MULTIPLIER = "missing_contract_multiplier"
BELOW_TRADED_NOTIONAL = "below_traded_notional_floor"
MISSING_OPEN_INTEREST = "missing_open_interest_history"
BELOW_OPEN_INTEREST_NOTIONAL = "below_open_interest_notional_floor"
BELOW_OPTION_VOLUME = "below_option_volume_floor"
BELOW_OPTION_OPEN_INTEREST = "below_option_open_interest_floor"
MISSING_OPTION_MARK = "missing_option_mark"
MISSING_OPTION_QUOTE = "missing_option_quote"
OPTION_SPREAD_TOO_WIDE = "option_spread_too_wide"


@dataclass(frozen=True, slots=True)
class LiquidityEligibilityPolicy:
    """Frozen assumptions used to decide whether a row is investable."""

    policy_id: str
    market_vertical: str
    initial_capital: float
    capital_currency: str
    lookback_sessions: int = DEFAULT_LIQUIDITY_LOOKBACK_SESSIONS
    min_observations: int = DEFAULT_LIQUIDITY_MIN_OBSERVATIONS
    decision_lag_sessions: int = 1
    max_position_weight: float = 0.05
    max_daily_volume_participation: float = 0.01
    static_min_daily_traded_notional: float = 0.0
    require_open_interest: bool = False
    max_open_interest_share: float = 0.02
    min_option_volume: float = 1.0
    min_option_open_interest: float = 1.0
    max_option_spread_pct: float | None = 0.25
    allow_option_settlement_proxy: bool = True
    intraday_volume_aggregation: str = "sum"
    version: str = LIQUIDITY_POLICY_VERSION

    def __post_init__(self) -> None:
        vertical = normalize_market_vertical(self.market_vertical)
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be at least 1")
        if not 1 <= self.min_observations <= self.lookback_sessions:
            raise ValueError("min_observations must be within the lookback window")
        if self.decision_lag_sessions < 0:
            raise ValueError("decision_lag_sessions cannot be negative")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 < self.max_daily_volume_participation <= 1:
            raise ValueError("max_daily_volume_participation must be in (0, 1]")
        if not 0 < self.max_open_interest_share <= 1:
            raise ValueError("max_open_interest_share must be in (0, 1]")
        if self.intraday_volume_aggregation not in {"sum", "last"}:
            raise ValueError("intraday_volume_aggregation must be 'sum' or 'last'")
        object.__setattr__(self, "market_vertical", vertical)

    @property
    def instrument_family(self) -> str:
        return str(
            ASSET_TAXONOMY.get(self.market_vertical, {}).get(
                "instrument_family",
                "asset",
            )
        )

    @property
    def capacity_min_daily_traded_notional(self) -> float:
        return (
            self.initial_capital
            * self.max_position_weight
            / self.max_daily_volume_participation
        )

    @property
    def required_daily_traded_notional(self) -> float:
        return max(
            float(self.static_min_daily_traded_notional),
            float(self.capacity_min_daily_traded_notional),
        )

    @property
    def required_open_interest_notional(self) -> float:
        if not self.require_open_interest:
            return 0.0
        return (
            self.initial_capital
            * self.max_position_weight
            / self.max_open_interest_share
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "instrument_family": self.instrument_family,
                "capacity_min_daily_traded_notional": (
                    self.capacity_min_daily_traded_notional
                ),
                "required_daily_traded_notional": (
                    self.required_daily_traded_notional
                ),
                "required_open_interest_notional": (
                    self.required_open_interest_notional
                ),
            }
        )
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LiquidityEligibilitySummary:
    total_rows: int
    eligible_rows: int
    ineligible_rows: int
    eligible_rate: float
    reason_counts: dict[str, int]
    policy_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_liquidity_policy(
    market_vertical: str,
    *,
    initial_capital: float | None = None,
    capital_currency: str | None = None,
    max_position_weight: float | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LiquidityEligibilityPolicy:
    """Resolve one explicit policy from taxonomy, capital, and overrides."""

    # Import lazily because the backtesting package also exposes evaluators that
    # consume this liquidity module.
    from oqp.research.backtesting.capital_policy import (
        resolve_execution_capital,
    )

    vertical = normalize_market_vertical(market_vertical)
    taxonomy = ASSET_TAXONOMY.get(vertical, {})
    capital = resolve_execution_capital(
        asset_class=vertical,
        initial_capital=initial_capital,
        capital_currency=capital_currency,
    )
    family = str(taxonomy.get("instrument_family") or "asset")
    require_open_interest = family == "future"
    if family == "option":
        lookback_sessions = 1
        min_observations = 1
        decision_lag_sessions = 0
    else:
        lookback_sessions = DEFAULT_LIQUIDITY_LOOKBACK_SESSIONS
        min_observations = DEFAULT_LIQUIDITY_MIN_OBSERVATIONS
        decision_lag_sessions = 1

    policy = LiquidityEligibilityPolicy(
        policy_id=f"liq_001_{vertical.lower()}_capacity",
        market_vertical=vertical,
        initial_capital=float(capital.initial_capital),
        capital_currency=str(capital.currency),
        lookback_sessions=lookback_sessions,
        min_observations=min_observations,
        decision_lag_sessions=decision_lag_sessions,
        max_position_weight=float(max_position_weight or 0.05),
        static_min_daily_traded_notional=float(
            taxonomy.get("min_daily_traded_value") or 0.0
        ),
        require_open_interest=require_open_interest,
    )
    if not overrides:
        return policy

    allowed = set(asdict(policy))
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(
            "Unknown liquidity policy override(s): " + ", ".join(unknown)
        )
    return replace(policy, **dict(overrides))


def assess_liquidity_eligibility(
    frame: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.DataFrame:
    """Attach causal eligibility diagnostics without deleting any rows."""

    if frame.empty:
        out = frame.copy()
        out["liquidity_eligible"] = pd.Series(dtype=bool)
        out["liquidity_reason_code"] = pd.Series(dtype=str)
        return _attach_policy_attrs(out, policy)
    if policy.instrument_family == "option":
        return _assess_option_rows(frame, policy)
    return _assess_session_rows(frame, policy)


def ensure_liquidity_eligibility(
    frame: pd.DataFrame,
    *,
    market_vertical: str,
    initial_capital: float | None = None,
    capital_currency: str | None = None,
    max_position_weight: float | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Resolve and apply the shared policy, reusing a matching assessment."""

    policy = resolve_liquidity_policy(
        market_vertical,
        initial_capital=initial_capital or _optional_float(frame.attrs.get("initial_capital")),
        capital_currency=capital_currency or frame.attrs.get("capital_currency"),
        max_position_weight=max_position_weight,
        overrides=overrides or frame.attrs.get("liquidity_policy_overrides"),
    )
    if (
        "liquidity_eligible" in frame.columns
        and frame.attrs.get("liquidity_policy_fingerprint") == policy.fingerprint
    ):
        return frame
    return assess_liquidity_eligibility(frame, policy)


def liquidity_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an attributed view for factor metrics and cross-sectional ranks."""

    if "liquidity_eligible" not in frame.columns:
        raise ValueError("frame has not been assessed for liquidity eligibility")
    out = frame.loc[_coerce_eligibility_mask(frame["liquidity_eligible"])].copy()
    out.attrs.update(frame.attrs)
    return out


def _coerce_eligibility_mask(values: pd.Series) -> pd.Series:
    """Return a strict boolean mask from nullable or serialized eligibility."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return (
            pd.to_numeric(values, errors="coerce")
            .fillna(0.0)
            .ne(0.0)
        )
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def apply_liquidity_gate(
    frame: pd.DataFrame,
    *,
    weight_columns: Sequence[str] = (
        "routed_target_weight",
        "final_target_weight",
        "target_weight",
        "signal",
    ),
) -> pd.DataFrame:
    """Force ineligible targets flat while retaining rows needed for exits."""

    if "liquidity_eligible" not in frame.columns:
        raise ValueError("frame has not been assessed for liquidity eligibility")
    out = frame.copy()
    out.attrs.update(frame.attrs)
    blocked = ~_coerce_eligibility_mask(out["liquidity_eligible"])
    forced_flat = pd.Series(False, index=out.index)
    gated_columns: list[str] = []
    for column in weight_columns:
        if column not in out.columns:
            continue
        backup = f"pre_liquidity_{column}"
        if backup not in out.columns:
            out[backup] = out[column]
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        forced_flat |= blocked & values.abs().gt(0.0)
        out.loc[blocked, column] = 0.0
        gated_columns.append(column)
    out["liquidity_forced_flat"] = forced_flat
    out.attrs["liquidity_gated_weight_columns"] = gated_columns
    out.attrs["liquidity_forced_flat_rows"] = int(forced_flat.sum())
    return out


def _assess_session_rows(
    frame: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.DataFrame:
    out = frame.copy()
    attrs = dict(frame.attrs)
    if "ticker" not in out.columns and "symbol" in out.columns:
        out["ticker"] = out["symbol"].astype(str)
    if "ticker" not in out.columns:
        raise ValueError("liquidity eligibility requires ticker or symbol")
    session_col = _session_source_column(out)
    out["_liquidity_session"] = _normalize_session(out[session_col])
    if out["_liquidity_session"].isna().any():
        raise ValueError("liquidity eligibility found invalid session dates")
    out["_liquidity_row_order"] = np.arange(len(out))
    out = out.sort_values(["ticker", "_liquidity_session", session_col])

    price_col = _first_column(out, ("close", "settlement", "last_price", "last"))
    volume_col = _first_column(out, ("volume", "vol"))
    oi_col = _first_column(out, ("open_interest", "oi"))
    out["_liquidity_price"] = (
        pd.to_numeric(out[price_col], errors="coerce")
        if price_col
        else np.nan
    )
    out["_liquidity_volume"] = (
        pd.to_numeric(out[volume_col], errors="coerce")
        if volume_col
        else np.nan
    )
    out["_liquidity_open_interest"] = (
        pd.to_numeric(out[oi_col], errors="coerce")
        if oi_col
        else np.nan
    )
    out["_liquidity_multiplier"] = _resolve_multiplier(out, policy)

    keys = ["ticker", "_liquidity_session"]
    bars_per_session = out.groupby(keys, sort=False).size()
    is_intraday = bool((bars_per_session > 1).any())
    volume_aggregation = (
        policy.intraday_volume_aggregation if is_intraday else "last"
    )
    grouped = out.groupby(keys, sort=True)
    sessions = grouped.agg(
        liquidity_price=("_liquidity_price", _last_valid),
        liquidity_volume=(
            "_liquidity_volume",
            _sum_valid if volume_aggregation == "sum" else _last_valid,
        ),
        liquidity_open_interest=("_liquidity_open_interest", _last_valid),
        liquidity_contract_multiplier=("_liquidity_multiplier", _last_valid),
    ).reset_index()
    sessions["liquidity_traded_notional"] = (
        sessions["liquidity_price"]
        * sessions["liquidity_volume"]
        * sessions["liquidity_contract_multiplier"]
    )
    sessions["liquidity_open_interest_notional"] = (
        sessions["liquidity_price"]
        * sessions["liquidity_open_interest"]
        * sessions["liquidity_contract_multiplier"]
    )
    sessions = sessions.sort_values(keys).reset_index(drop=True)
    sessions["liquidity_history_observations"] = _causal_rolling(
        sessions,
        "liquidity_traded_notional",
        policy,
        operation="count",
    )
    sessions["liquidity_trailing_median_notional"] = _causal_rolling(
        sessions,
        "liquidity_traded_notional",
        policy,
        operation="median",
    )
    sessions["liquidity_oi_history_observations"] = _causal_rolling(
        sessions,
        "liquidity_open_interest_notional",
        policy,
        operation="count",
    )
    sessions["liquidity_trailing_median_oi_notional"] = _causal_rolling(
        sessions,
        "liquidity_open_interest_notional",
        policy,
        operation="median",
    )
    sessions["liquidity_required_daily_notional"] = (
        policy.required_daily_traded_notional
    )
    sessions["liquidity_required_oi_notional"] = (
        policy.required_open_interest_notional
    )
    sessions["liquidity_reason_code"] = _session_reason_codes(sessions, policy)
    sessions["liquidity_eligible"] = sessions["liquidity_reason_code"].eq(ELIGIBLE)

    diagnostic_columns = [
        column
        for column in sessions.columns
        if column.startswith("liquidity_")
    ]
    out = out.merge(
        sessions[[*keys, *diagnostic_columns]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    out = out.sort_values("_liquidity_row_order").drop(
        columns=[
            "_liquidity_session",
            "_liquidity_row_order",
            "_liquidity_price",
            "_liquidity_volume",
            "_liquidity_open_interest",
            "_liquidity_multiplier",
        ]
    )
    out.index = frame.index
    out.attrs.update(attrs)
    out.attrs["liquidity_intraday_detected"] = is_intraday
    out.attrs["liquidity_intraday_volume_aggregation"] = volume_aggregation
    out.attrs["liquidity_session_source_column"] = session_col
    return _attach_policy_attrs(out, policy)


def _assess_option_rows(
    frame: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.DataFrame:
    from oqp.options.liquidity import mark_price, spread_pct

    out = frame.copy()
    out["liquidity_option_volume"] = _numeric_column(out, "volume")
    out["liquidity_option_open_interest"] = _numeric_column(
        out,
        "open_interest",
    )
    out["liquidity_option_spread_pct"] = out.apply(spread_pct, axis=1)
    out["liquidity_option_mark"] = out.apply(mark_price, axis=1)
    has_quote = _numeric_column(out, "bid").gt(0) & _numeric_column(
        out,
        "ask",
    ).gt(0)
    reason = pd.Series(ELIGIBLE, index=out.index, dtype=object)
    reason = reason.mask(
        out["liquidity_option_mark"].isna(),
        MISSING_OPTION_MARK,
    )
    reason = reason.mask(
        reason.eq(ELIGIBLE)
        & out["liquidity_option_volume"].fillna(0.0).lt(policy.min_option_volume),
        BELOW_OPTION_VOLUME,
    )
    reason = reason.mask(
        reason.eq(ELIGIBLE)
        & out["liquidity_option_open_interest"]
        .fillna(0.0)
        .lt(policy.min_option_open_interest),
        BELOW_OPTION_OPEN_INTEREST,
    )
    if not policy.allow_option_settlement_proxy:
        reason = reason.mask(reason.eq(ELIGIBLE) & ~has_quote, MISSING_OPTION_QUOTE)
    if policy.max_option_spread_pct is not None:
        reason = reason.mask(
            reason.eq(ELIGIBLE)
            & has_quote
            & out["liquidity_option_spread_pct"].gt(
                policy.max_option_spread_pct
            ),
            OPTION_SPREAD_TOO_WIDE,
        )
    out["liquidity_reason_code"] = reason
    out["liquidity_eligible"] = reason.eq(ELIGIBLE)
    return _attach_policy_attrs(out, policy)


def _session_reason_codes(
    sessions: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.Series:
    reason = pd.Series(ELIGIBLE, index=sessions.index, dtype=object)
    missing_multiplier = sessions["liquidity_contract_multiplier"].isna()
    missing_market_data = (
        sessions["liquidity_price"].isna()
        | sessions["liquidity_volume"].isna()
    )
    insufficient_history = sessions["liquidity_history_observations"].lt(
        policy.min_observations
    )
    below_notional = sessions["liquidity_trailing_median_notional"].lt(
        policy.required_daily_traded_notional
    )
    reason = reason.mask(missing_multiplier, MISSING_MULTIPLIER)
    reason = reason.mask(reason.eq(ELIGIBLE) & missing_market_data, MISSING_MARKET_DATA)
    reason = reason.mask(reason.eq(ELIGIBLE) & insufficient_history, WARMUP)
    reason = reason.mask(reason.eq(ELIGIBLE) & below_notional, BELOW_TRADED_NOTIONAL)
    if policy.require_open_interest:
        missing_oi = sessions["liquidity_oi_history_observations"].lt(
            policy.min_observations
        )
        below_oi = sessions["liquidity_trailing_median_oi_notional"].lt(
            policy.required_open_interest_notional
        )
        reason = reason.mask(reason.eq(ELIGIBLE) & missing_oi, MISSING_OPEN_INTEREST)
        reason = reason.mask(
            reason.eq(ELIGIBLE) & below_oi,
            BELOW_OPEN_INTEREST_NOTIONAL,
        )
    return reason


def _causal_rolling(
    sessions: pd.DataFrame,
    value_col: str,
    policy: LiquidityEligibilityPolicy,
    *,
    operation: str,
) -> pd.Series:
    def calculate(values: pd.Series) -> pd.Series:
        lagged = values.shift(policy.decision_lag_sessions)
        rolling = lagged.rolling(policy.lookback_sessions, min_periods=1)
        if operation == "count":
            return rolling.count()
        if operation == "median":
            return rolling.median()
        raise ValueError(f"Unsupported rolling operation: {operation}")

    return sessions.groupby("ticker", sort=False, group_keys=False)[value_col].apply(
        calculate
    )


def _resolve_multiplier(
    frame: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.Series:
    if policy.instrument_family != "future":
        return pd.Series(1.0, index=frame.index)
    if "multiplier" in frame.columns:
        supplied = pd.to_numeric(frame["multiplier"], errors="coerce")
    elif "contract_multiplier" in frame.columns:
        supplied = pd.to_numeric(frame["contract_multiplier"], errors="coerce")
    else:
        supplied = pd.Series(np.nan, index=frame.index)
    if policy.market_vertical != MarketVertical.FUTURES_CN.value:
        return supplied.where(supplied.gt(0.0))

    master = InstrumentMaster(policy.market_vertical)
    by_ticker: dict[str, float] = {}
    for ticker in frame["ticker"].astype(str).unique():
        profile = master.get_profile(ticker)
        by_ticker[ticker] = (
            float(profile.multiplier)
            if profile.exchange != "UNKNOWN" and profile.multiplier > 0
            else np.nan
        )
    resolved = frame["ticker"].astype(str).map(by_ticker)
    return supplied.where(supplied.gt(0.0), resolved)


def _attach_policy_attrs(
    frame: pd.DataFrame,
    policy: LiquidityEligibilityPolicy,
) -> pd.DataFrame:
    reason_counts = {
        str(reason): int(count)
        for reason, count in frame.get(
            "liquidity_reason_code",
            pd.Series(dtype=str),
        ).value_counts(dropna=False).items()
    }
    eligible_rows = int(
        frame.get("liquidity_eligible", pd.Series(dtype=bool))
        .fillna(False)
        .sum()
    )
    total_rows = int(len(frame))
    summary = LiquidityEligibilitySummary(
        total_rows=total_rows,
        eligible_rows=eligible_rows,
        ineligible_rows=total_rows - eligible_rows,
        eligible_rate=(eligible_rows / total_rows) if total_rows else 0.0,
        reason_counts=reason_counts,
        policy_fingerprint=policy.fingerprint,
    )
    frame.attrs["liquidity_policy"] = policy.to_dict()
    frame.attrs["liquidity_policy_id"] = policy.policy_id
    frame.attrs["liquidity_policy_fingerprint"] = policy.fingerprint
    frame.attrs["liquidity_eligibility_summary"] = summary.to_dict()
    frame.attrs["liquidity_policy_status"] = "assessed"
    return frame


def _session_source_column(frame: pd.DataFrame) -> str:
    for column in ("trading_day", "economic_day", "session_date", "date", "datetime"):
        if column in frame.columns:
            return column
    raise ValueError(
        "liquidity eligibility requires trading_day, economic_day, session_date, date, or datetime"
    )


def _normalize_session(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        parsed = parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return parsed.dt.normalize()


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _last_valid(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan


def _sum_valid(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.sum()) if not valid.empty else np.nan


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


__all__ = [
    "BELOW_OPEN_INTEREST_NOTIONAL",
    "BELOW_TRADED_NOTIONAL",
    "DEFAULT_LIQUIDITY_LOOKBACK_SESSIONS",
    "DEFAULT_LIQUIDITY_MIN_OBSERVATIONS",
    "ELIGIBLE",
    "LIQUIDITY_POLICY_VERSION",
    "LiquidityEligibilityPolicy",
    "LiquidityEligibilitySummary",
    "apply_liquidity_gate",
    "assess_liquidity_eligibility",
    "ensure_liquidity_eligibility",
    "liquidity_eligible_rows",
    "resolve_liquidity_policy",
]
