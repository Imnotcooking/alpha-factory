"""Return-horizon utilities for daily research backtests."""

from __future__ import annotations

import pandas as pd
import numpy as np


RETURN_HORIZON_AUTO = "auto"
RETURN_BAR_SIGNAL_NEXT_BAR = "bar_signal_next_bar"
RETURN_NEXT_OPEN_TO_NEXT_CLOSE = "close_signal_next_open_to_close"
RETURN_NEXT_OPEN_TO_NEXT_OPEN = "close_signal_next_open_to_next_open"
RETURN_CLOSE_TO_NEXT_CLOSE = "close_signal_close_to_next_close"
RETURN_CLOSE_TO_NEXT_OPEN = "close_signal_close_to_next_open"
RETURN_CLOSE_TO_CLOSE_FALLBACK = "close_to_close_fallback"

VALID_RETURN_HORIZONS = {
    RETURN_HORIZON_AUTO,
    RETURN_BAR_SIGNAL_NEXT_BAR,
    RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    RETURN_CLOSE_TO_NEXT_CLOSE,
    RETURN_CLOSE_TO_NEXT_OPEN,
    RETURN_CLOSE_TO_CLOSE_FALLBACK,
}

RETURN_HORIZON_ALIASES = {
    "": RETURN_HORIZON_AUTO,
    "auto": RETURN_HORIZON_AUTO,
    "intraday": RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    "open_to_close": RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    "next_open_to_next_close": RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    "next_open_next_close": RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    "close_signal_next_open_to_close": RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    "swing": RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    "open_to_open": RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    "next_open_to_next_open": RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    "next_open_to_following_open": RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    "close_signal_next_open_to_next_open": RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    "close_to_close": RETURN_CLOSE_TO_NEXT_CLOSE,
    "close_to_next_close": RETURN_CLOSE_TO_NEXT_CLOSE,
    "signal_close_to_next_close": RETURN_CLOSE_TO_NEXT_CLOSE,
    "close_signal_close_to_next_close": RETURN_CLOSE_TO_NEXT_CLOSE,
    "overnight": RETURN_CLOSE_TO_NEXT_OPEN,
    "close_to_open": RETURN_CLOSE_TO_NEXT_OPEN,
    "close_to_next_open": RETURN_CLOSE_TO_NEXT_OPEN,
    "signal_close_to_next_open": RETURN_CLOSE_TO_NEXT_OPEN,
    "close_signal_close_to_next_open": RETURN_CLOSE_TO_NEXT_OPEN,
    "bar": RETURN_BAR_SIGNAL_NEXT_BAR,
    "next_bar": RETURN_BAR_SIGNAL_NEXT_BAR,
    "bar_signal_next_bar": RETURN_BAR_SIGNAL_NEXT_BAR,
    "fallback": RETURN_CLOSE_TO_CLOSE_FALLBACK,
    "close_to_close_fallback": RETURN_CLOSE_TO_CLOSE_FALLBACK,
}

RETURN_HORIZON_DESCRIPTIONS = {
    RETURN_BAR_SIGNAL_NEXT_BAR: "signal at current bar, mark next bar close/price",
    RETURN_NEXT_OPEN_TO_NEXT_CLOSE: "signal after today's close, enter next open, mark next close",
    RETURN_NEXT_OPEN_TO_NEXT_OPEN: "signal after today's close, enter next open, mark following open",
    RETURN_CLOSE_TO_NEXT_CLOSE: "signal at today's close, mark close-to-next-close diagnostic return",
    RETURN_CLOSE_TO_NEXT_OPEN: "signal at today's close, mark close-to-next-open overnight gap",
    RETURN_CLOSE_TO_CLOSE_FALLBACK: "fallback close-to-close return when open is unavailable",
}


def normalize_return_horizon(value: str | None) -> str:
    key = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = RETURN_HORIZON_ALIASES.get(key, key)
    if normalized not in VALID_RETURN_HORIZONS:
        raise ValueError(
            f"Invalid return_horizon={value!r}. Expected one of "
            f"{sorted(VALID_RETURN_HORIZONS)} or aliases {sorted(RETURN_HORIZON_ALIASES)}."
        )
    return normalized


def infer_default_return_horizon(
    *,
    data_frequency: str,
    has_open: bool,
) -> str:
    if str(data_frequency).strip().lower() in {"intraday", "tick"}:
        return RETURN_BAR_SIGNAL_NEXT_BAR
    return RETURN_NEXT_OPEN_TO_NEXT_CLOSE if has_open else RETURN_CLOSE_TO_CLOSE_FALLBACK


def attach_return_horizon(
    df: pd.DataFrame,
    *,
    return_horizon: str,
    data_frequency: str,
) -> pd.DataFrame:
    """Attach forward/execution returns for a named signal-to-mark horizon."""

    horizon = normalize_return_horizon(return_horizon)
    if horizon == RETURN_HORIZON_AUTO:
        horizon = infer_default_return_horizon(
            data_frequency=data_frequency,
            has_open="open" in df.columns,
        )

    required = {"date", "ticker", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot build return horizon without columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    grouped = out.groupby("ticker", sort=False)
    next_close = grouped["close"].shift(-1)

    if horizon == RETURN_BAR_SIGNAL_NEXT_BAR:
        out["forward_return"] = _price_return(next_close, close)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = close
    elif horizon == RETURN_NEXT_OPEN_TO_NEXT_CLOSE:
        _require_open(out, horizon)
        next_open = grouped["open"].shift(-1)
        out["forward_return"] = _price_return(next_close, next_open)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = pd.to_numeric(next_open, errors="coerce")
    elif horizon == RETURN_NEXT_OPEN_TO_NEXT_OPEN:
        _require_open(out, horizon)
        next_open = grouped["open"].shift(-1)
        following_open = grouped["open"].shift(-2)
        out["forward_return"] = _price_return(following_open, next_open)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = pd.to_numeric(next_open, errors="coerce")
    elif horizon == RETURN_CLOSE_TO_NEXT_CLOSE:
        out["forward_return"] = _price_return(next_close, close)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = close
    elif horizon == RETURN_CLOSE_TO_NEXT_OPEN:
        _require_open(out, horizon)
        next_open = grouped["open"].shift(-1)
        out["forward_return"] = _price_return(next_open, close)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = close
    elif horizon == RETURN_CLOSE_TO_CLOSE_FALLBACK:
        out["forward_return"] = _price_return(next_close, close)
        out["execution_period_return"] = out["forward_return"]
        out["execution_price"] = close
    else:
        raise ValueError(f"Unsupported return horizon: {horizon}")

    out["forward_return"] = pd.to_numeric(out["forward_return"], errors="coerce").replace([np.inf, -np.inf], pd.NA)
    out["execution_period_return"] = pd.to_numeric(
        out["execution_period_return"],
        errors="coerce",
    ).replace([np.inf, -np.inf], pd.NA)
    out["execution_price"] = pd.to_numeric(out["execution_price"], errors="coerce").fillna(close)
    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["return_horizon"] = horizon
    out.attrs["execution_assumption"] = horizon
    out.attrs["return_horizon_description"] = RETURN_HORIZON_DESCRIPTIONS.get(horizon, horizon)
    out.attrs["benchmark_return_col"] = "execution_period_return"
    return out


def _price_return(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numerator / denominator - 1, ignoring non-positive price placeholders."""

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid = numerator.gt(0.0) & denominator.gt(0.0)
    out = pd.Series(np.nan, index=denominator.index, dtype=float)
    out.loc[valid] = numerator.loc[valid] / denominator.loc[valid] - 1.0
    return out


def _require_open(df: pd.DataFrame, horizon: str) -> None:
    if "open" not in df.columns:
        raise ValueError(f"return_horizon={horizon!r} requires an open column.")
