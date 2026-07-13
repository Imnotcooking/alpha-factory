"""Deterministic adversarial fixtures for point-in-time roll construction.

These rows are software-test inputs only.  They are intentionally artificial,
contain a large cross-contract basis, and are never eligible as paper evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd


STAGE_OWNER = 3


@dataclass(frozen=True)
class Stage3AdversarialFixture:
    contract_rows: pd.DataFrame
    expected_selection: Mapping[date, str | None]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.contract_rows, pd.DataFrame) or self.contract_rows.empty:
            raise ValueError("Stage 3 fixture requires non-empty contract rows.")
        object.__setattr__(
            self,
            "expected_selection",
            MappingProxyType(dict(self.expected_selection)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def make_stage3_adversarial_fixture() -> Stage3AdversarialFixture:
    """Return a small fixture with rolls, gaps, ties, expiry, and bad liquidity."""

    dates = tuple(pd.bdate_range("2020-01-02", periods=9))
    product = "SYN_RB"
    specifications = {
        "SYN_RB2001": {
            "listing_date": pd.Timestamp("2019-10-01"),
            "last_trade_date": pd.Timestamp("2020-01-08"),
            "base_close": 100.0,
            "open_interest": (1000, 900, 800, 700, 600, None, None, None, None),
            "volume": (800, 750, 700, 650, 600, None, None, None, None),
        },
        "SYN_RB2002": {
            "listing_date": pd.Timestamp("2019-11-01"),
            "last_trade_date": pd.Timestamp("2020-02-20"),
            "base_close": 150.0,
            # The second date is intentionally absent.  Later dates retain the
            # contract's own close history needed for a roll-safe return.
            "open_interest": (500, None, 1200, 1300, 1000, 800, 0, 1900, 1800),
            "volume": (500, None, 900, 950, 700, 650, 0, 1500, 1400),
        },
        "SYN_RB2003": {
            "listing_date": pd.Timestamp("2020-01-08"),
            "last_trade_date": pd.Timestamp("2020-03-20"),
            "base_close": 220.0,
            "open_interest": (None, None, None, None, 1000, 1400, 0, 900, 1000),
            "volume": (None, None, None, None, 700, 1000, 0, 800, 850),
        },
    }

    rows: list[dict[str, Any]] = []
    for contract_index, (contract, specification) in enumerate(specifications.items()):
        for date_index, trading_date in enumerate(dates):
            open_interest = specification["open_interest"][date_index]
            volume = specification["volume"][date_index]
            if open_interest is None or volume is None:
                continue
            close = float(specification["base_close"]) + 0.5 * date_index
            open_price = close - 0.2
            high = close + 0.8
            low = close - 1.0
            limit_locked = contract == "SYN_RB2002" and date_index == 7
            stale_bar = contract == "SYN_RB2001" and date_index == 3
            multiplier = 10.0
            rows.append(
                {
                    "product": product,
                    "contract": contract,
                    "exchange": "SYN",
                    "trading_date": trading_date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "settlement": (open_price + high + low + close) / 4.0,
                    "volume": int(volume),
                    "turnover": close * float(volume) * multiplier,
                    "open_interest": int(open_interest),
                    "multiplier": multiplier,
                    "tick_size": 0.1,
                    "limit_lock_flag": limit_locked,
                    "stale_bar_flag": stale_bar,
                    "source_row_id": (
                        f"{product}:{trading_date:%Y%m%d}:{contract}:{contract_index}"
                    ),
                    "listing_date": specification["listing_date"],
                    "last_trade_date": specification["last_trade_date"],
                }
            )

    contract_rows = (
        pd.DataFrame(rows)
        .sort_values(["product", "contract", "trading_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_selection = {
        dates[1].date(): "SYN_RB2001",
        dates[2].date(): "SYN_RB2001",
        dates[3].date(): "SYN_RB2002",
        dates[4].date(): "SYN_RB2002",
        dates[5].date(): "SYN_RB2002",
        dates[6].date(): "SYN_RB2003",
        dates[7].date(): None,
        dates[8].date(): "SYN_RB2003",
    }
    return Stage3AdversarialFixture(
        contract_rows=contract_rows,
        expected_selection=expected_selection,
        metadata={
            "fixture_id": "stage3_adversarial_roll_fixture_v1",
            "synthetic": True,
            "scientific_evidence": False,
            "paper_eligible": False,
            "contains_cross_contract_basis": True,
            "contains_missing_contract_row": True,
            "contains_zero_liquidity": True,
            "contains_limit_lock": True,
            "contains_stale_bar": True,
            "contains_tie": True,
            "contains_new_listing_and_expiry": True,
        },
    )


def make_invalid_stage3_frames() -> dict[str, pd.DataFrame]:
    """Return independently invalid copies for validator failure tests."""

    valid = make_stage3_adversarial_fixture().contract_rows
    invalid_ohlc = valid.copy(deep=True)
    invalid_ohlc.loc[0, "high"] = invalid_ohlc.loc[0, "close"] - 1.0

    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)

    negative_liquidity = valid.copy(deep=True)
    negative_liquidity.loc[0, "open_interest"] = -1

    invalid_listing = valid.copy(deep=True)
    invalid_listing.loc[0, "listing_date"] = (
        pd.Timestamp(invalid_listing.loc[0, "trading_date"]) + pd.Timedelta(days=1)
    )

    intraday_timestamp = valid.copy(deep=True)
    intraday_timestamp.loc[0, "trading_date"] = (
        pd.Timestamp(intraday_timestamp.loc[0, "trading_date"])
        + pd.Timedelta(hours=12)
    )

    return {
        "invalid_ohlc": invalid_ohlc,
        "duplicate_key": duplicate,
        "negative_liquidity": negative_liquidity,
        "invalid_listing_interval": invalid_listing,
        "intraday_timestamp": intraday_timestamp,
    }


__all__ = [
    "STAGE_OWNER",
    "Stage3AdversarialFixture",
    "make_invalid_stage3_frames",
    "make_stage3_adversarial_fixture",
]
