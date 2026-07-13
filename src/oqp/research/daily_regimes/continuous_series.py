"""Point-in-time continuous futures construction for Paper 1.

The contract effective on product date ``t`` is ranked using only rows from the
prior available product date.  Returns always compare the selected contract to
that same contract's prior close, and the chained index never rewrites history
or imports a cross-contract basis jump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isclose, log
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from oqp.research.daily_regimes.contracts import RawDailyBarContract


STAGE_OWNER = 3
BUILDER_ID = "lagged_liquidity_continuous_series_v1"
REQUIRED_PANEL_COLUMNS = (
    "product",
    "trading_date",
    "decision_date",
    "contract",
    "selected_contract",
    "source_row_id",
    "exchange",
    "previous_contract",
    "roll_flag",
    "roll_reason",
    "sequence_id",
    "chain_reset_flag",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume",
    "turnover",
    "open_interest",
    "multiplier",
    "tick_size",
    "limit_lock_flag",
    "stale_bar_flag",
    "previous_same_contract_close",
    "same_contract_log_return",
    "diagnostic_cross_contract_log_return",
    "continuous_index",
    "selection_open_interest",
    "selection_volume",
)


@dataclass(frozen=True)
class ContinuousSeriesConfig:
    """Frozen causal contract-selection and chaining policy."""

    decision_lag_periods: int = 1
    primary_metric: str = "open_interest"
    secondary_metric: str = "volume"
    minimum_volume: int = 1
    minimum_open_interest: int = 1
    tie_breakers: tuple[str, ...] = (
        "earliest_last_trade_date",
        "contract",
    )
    exclude_limit_locked: bool = True
    exclude_stale_bars: bool = True
    return_convention: str = "selected_contract_same_contract_close"
    adjustment_convention: str = (
        "chained_same_contract_returns_no_history_rewrite"
    )
    continuous_index_base: float = 100.0
    missing_policy: str = "flag_no_backfill"

    def __post_init__(self) -> None:
        if self.decision_lag_periods != 1:
            raise ValueError("Stage 3 requires exactly one lagged decision period.")
        if (self.primary_metric, self.secondary_metric) != (
            "open_interest",
            "volume",
        ):
            raise ValueError("Stage 3 ranks open_interest first and volume second.")
        if self.minimum_volume < 1 or self.minimum_open_interest < 1:
            raise ValueError("Minimum volume and open interest must be positive.")
        if self.tie_breakers != ("earliest_last_trade_date", "contract"):
            raise ValueError("Stage 3 tie breakers are frozen by preregistration.")
        if not isinstance(self.exclude_limit_locked, bool) or not isinstance(
            self.exclude_stale_bars, bool
        ):
            raise TypeError("Exclusion settings must be booleans.")
        if self.return_convention != "selected_contract_same_contract_close":
            raise ValueError("Cross-contract returns are prohibited.")
        if self.adjustment_convention != (
            "chained_same_contract_returns_no_history_rewrite"
        ):
            raise ValueError("The non-revising chained-index convention is required.")
        if not np.isfinite(self.continuous_index_base) or self.continuous_index_base <= 0:
            raise ValueError("continuous_index_base must be finite and positive.")
        if self.missing_policy != "flag_no_backfill":
            raise ValueError("Stage 3 forbids filling missing selections.")


@dataclass(frozen=True)
class ContinuousSeriesResult:
    """Product-day panel and a complete selection ledger, including failures."""

    panel: pd.DataFrame
    lineage: pd.DataFrame
    builder_id: str = BUILDER_ID
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame.")
        if not isinstance(self.lineage, pd.DataFrame):
            raise TypeError("lineage must be a pandas DataFrame.")
        if not self.builder_id.strip():
            raise ValueError("builder_id must be non-empty.")
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def roll_ledger(self) -> pd.DataFrame:
        return self.lineage


@runtime_checkable
class ContinuousSeriesBuilder(Protocol):
    @property
    def builder_id(self) -> str:
        """Stable implementation identifier recorded in manifests."""

    def build(
        self,
        contract_rows: pd.DataFrame,
        *,
        config: ContinuousSeriesConfig,
    ) -> ContinuousSeriesResult:
        """Build a causal product panel without mutating source rows."""


class LaggedLiquidityContinuousSeriesBuilder:
    """Select the next product row from the prior date's liquidity only."""

    builder_id = BUILDER_ID

    def build(
        self,
        contract_rows: pd.DataFrame,
        *,
        config: ContinuousSeriesConfig,
    ) -> ContinuousSeriesResult:
        RawDailyBarContract(require_sorted=False).validate(contract_rows)
        rows = contract_rows.copy(deep=True)
        rows["trading_date"] = pd.to_datetime(rows["trading_date"]).dt.normalize()
        for optional_date in ("listing_date", "last_trade_date"):
            if optional_date in rows.columns:
                rows[optional_date] = pd.to_datetime(rows[optional_date]).dt.normalize()
        rows = rows.sort_values(
            ["product", "trading_date", "contract"], kind="mergesort"
        ).reset_index(drop=True)

        panel_rows: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        for product, product_rows in rows.groupby("product", sort=True):
            product_panel, product_ledger = _build_product(
                str(product), product_rows.reset_index(drop=True), config
            )
            panel_rows.extend(product_panel)
            ledger_rows.extend(product_ledger)

        panel = pd.DataFrame(panel_rows, columns=REQUIRED_PANEL_COLUMNS)
        ledger_columns = (
            "product",
            "trading_date",
            "decision_date",
            "selection_status",
            "selected_contract",
            "previous_contract",
            "roll_flag",
            "roll_reason",
            "selection_open_interest",
            "selection_volume",
            "selected_source_row_id",
        )
        lineage = pd.DataFrame(ledger_rows, columns=ledger_columns)
        if not panel.empty:
            panel = panel.sort_values(
                ["product", "trading_date"], kind="mergesort"
            ).reset_index(drop=True)
        if not lineage.empty:
            lineage = lineage.sort_values(
                ["product", "trading_date"], kind="mergesort"
            ).reset_index(drop=True)
        result = ContinuousSeriesResult(
            panel=panel,
            lineage=lineage,
            diagnostics={
                "input_rows": int(len(rows)),
                "output_rows": int(len(panel)),
                "ledger_rows": int(len(lineage)),
                "roll_count": int(panel["roll_flag"].sum()) if not panel.empty else 0,
                "unselected_dates": int(
                    lineage["selection_status"].ne("selected").sum()
                )
                if not lineage.empty
                else 0,
                "chain_reset_count": int(panel["chain_reset_flag"].sum())
                if not panel.empty
                else 0,
                "scientific_evidence": False,
            },
        )
        validate_continuous_series_result(result)
        return result


def build_continuous_series(
    contract_rows: pd.DataFrame,
    *,
    config: ContinuousSeriesConfig | None = None,
) -> ContinuousSeriesResult:
    """Build the preregistered point-in-time continuous product panel."""

    resolved = config or ContinuousSeriesConfig()
    return LaggedLiquidityContinuousSeriesBuilder().build(
        contract_rows, config=resolved
    )


def validate_continuous_series_result(
    result: ContinuousSeriesResult,
    *,
    required_columns: tuple[str, ...] = REQUIRED_PANEL_COLUMNS,
) -> None:
    """Validate timing, lineage, return, and non-revising-index invariants."""

    missing = [column for column in required_columns if column not in result.panel]
    if missing:
        raise ValueError(f"Continuous-series result is missing columns: {missing}")
    if result.panel.duplicated(["product", "trading_date"]).any():
        raise ValueError("Continuous panel contains duplicate product-date rows.")
    if result.lineage.duplicated(["product", "trading_date"]).any():
        raise ValueError("Roll ledger contains duplicate product-date rows.")
    if result.panel.empty:
        return

    trading_dates = pd.to_datetime(result.panel["trading_date"])
    decision_dates = pd.to_datetime(result.panel["decision_date"])
    if (decision_dates >= trading_dates).any():
        raise ValueError("Every selection decision must precede its effective date.")
    if not result.panel["contract"].equals(result.panel["selected_contract"]):
        raise ValueError("Panel contract must equal the selected contract.")
    if result.panel["source_row_id"].isna().any():
        raise ValueError("Every selected panel row requires raw source lineage.")
    expected_return = np.log(
        result.panel["close"].to_numpy(dtype=float)
        / result.panel["previous_same_contract_close"].to_numpy(dtype=float)
    )
    actual_return = result.panel["same_contract_log_return"].to_numpy(dtype=float)
    if not np.allclose(expected_return, actual_return, atol=1e-12, rtol=0.0):
        raise ValueError("same_contract_log_return does not match its raw closes.")

    for _, group in result.panel.groupby("product", sort=False):
        previous_row: pd.Series | None = None
        for _, row in group.sort_values("trading_date").iterrows():
            if bool(row["chain_reset_flag"]):
                if not isclose(
                    float(row["continuous_index"]),
                    float(group.iloc[0]["continuous_index"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("A reset sequence must restart at the index base.")
            elif previous_row is not None:
                expected_index = float(previous_row["continuous_index"]) * exp(
                    float(row["same_contract_log_return"])
                )
                if not isclose(
                    float(row["continuous_index"]),
                    expected_index,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError("continuous_index violates recursive chaining.")
            previous_row = row


def _build_product(
    product: str,
    rows: pd.DataFrame,
    config: ContinuousSeriesConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = tuple(sorted(pd.Timestamp(value) for value in rows["trading_date"].unique()))
    panel_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    previous_contract: str | None = None
    previous_output_date: pd.Timestamp | None = None
    previous_output_close: float | None = None
    previous_index: float | None = None
    sequence_number = 0

    for date_index in range(config.decision_lag_periods, len(dates)):
        decision_date = dates[date_index - config.decision_lag_periods]
        effective_date = dates[date_index]
        decision_rows = rows.loc[rows["trading_date"].eq(decision_date)].copy()
        eligible = _eligible_candidates(decision_rows, effective_date, config)
        if eligible.empty:
            ledger_rows.append(
                _ledger_row(
                    product=product,
                    trading_date=effective_date,
                    decision_date=decision_date,
                    status="no_eligible_contract",
                    selected_contract=None,
                    previous_contract=previous_contract,
                    roll_flag=False,
                    roll_reason="no_eligible_lagged_contract",
                )
            )
            previous_output_date = previous_output_date
            continue

        selected_decision_row = _rank_candidates(eligible).iloc[0]
        selected_contract = str(selected_decision_row["contract"])
        effective_rows = rows.loc[
            rows["trading_date"].eq(effective_date)
            & rows["contract"].astype(str).eq(selected_contract)
        ]
        if effective_rows.empty:
            ledger_rows.append(
                _ledger_row(
                    product=product,
                    trading_date=effective_date,
                    decision_date=decision_date,
                    status="missing_effective_bar",
                    selected_contract=selected_contract,
                    previous_contract=previous_contract,
                    roll_flag=False,
                    roll_reason="selected_contract_missing_effective_bar",
                    selection_open_interest=float(
                        selected_decision_row["open_interest"]
                    ),
                    selection_volume=float(selected_decision_row["volume"]),
                )
            )
            continue

        effective_row = effective_rows.iloc[0]
        selected_previous_close = float(selected_decision_row["close"])
        selected_close = float(effective_row["close"])
        same_contract_return = log(selected_close / selected_previous_close)
        roll_flag = previous_contract is not None and selected_contract != previous_contract
        if previous_contract is None:
            roll_reason = "initial_selection"
        elif roll_flag:
            roll_reason = "lagged_liquidity_rank_change"
        else:
            roll_reason = "contract_retained"

        contiguous = previous_output_date is not None and previous_output_date == decision_date
        chain_reset = previous_index is None or not contiguous
        if chain_reset:
            if previous_index is not None:
                sequence_number += 1
            continuous_index = float(config.continuous_index_base)
        else:
            continuous_index = float(previous_index) * exp(same_contract_return)
        cross_contract_return = (
            np.nan
            if previous_output_close is None
            else log(selected_close / previous_output_close)
        )
        sequence_id = f"{product}:{sequence_number:03d}"
        panel_row = {
            "product": product,
            "trading_date": effective_date,
            "decision_date": decision_date,
            "contract": selected_contract,
            "selected_contract": selected_contract,
            "source_row_id": str(effective_row["source_row_id"]),
            "exchange": str(effective_row["exchange"]),
            "previous_contract": previous_contract,
            "roll_flag": roll_flag,
            "roll_reason": roll_reason,
            "sequence_id": sequence_id,
            "chain_reset_flag": chain_reset,
            "open": float(effective_row["open"]),
            "high": float(effective_row["high"]),
            "low": float(effective_row["low"]),
            "close": selected_close,
            "settlement": float(
                effective_row["settlement"]
                if "settlement" in effective_row
                else selected_close
            ),
            "volume": float(effective_row["volume"]),
            "turnover": float(effective_row["turnover"]),
            "open_interest": float(effective_row["open_interest"]),
            "multiplier": float(effective_row["multiplier"]),
            "tick_size": float(effective_row["tick_size"]),
            "limit_lock_flag": bool(effective_row["limit_lock_flag"]),
            "stale_bar_flag": bool(effective_row["stale_bar_flag"]),
            "previous_same_contract_close": selected_previous_close,
            "same_contract_log_return": same_contract_return,
            "diagnostic_cross_contract_log_return": cross_contract_return,
            "continuous_index": continuous_index,
            "selection_open_interest": float(selected_decision_row["open_interest"]),
            "selection_volume": float(selected_decision_row["volume"]),
        }
        panel_rows.append(panel_row)
        ledger_rows.append(
            _ledger_row(
                product=product,
                trading_date=effective_date,
                decision_date=decision_date,
                status="selected",
                selected_contract=selected_contract,
                previous_contract=previous_contract,
                roll_flag=roll_flag,
                roll_reason=roll_reason,
                selection_open_interest=float(selected_decision_row["open_interest"]),
                selection_volume=float(selected_decision_row["volume"]),
                selected_source_row_id=str(effective_row["source_row_id"]),
            )
        )
        previous_contract = selected_contract
        previous_output_date = effective_date
        previous_output_close = selected_close
        previous_index = continuous_index

    return panel_rows, ledger_rows


def _eligible_candidates(
    rows: pd.DataFrame,
    effective_date: pd.Timestamp,
    config: ContinuousSeriesConfig,
) -> pd.DataFrame:
    eligible = rows.loc[
        rows["volume"].ge(config.minimum_volume)
        & rows["open_interest"].ge(config.minimum_open_interest)
    ].copy()
    if config.exclude_limit_locked:
        eligible = eligible.loc[~eligible["limit_lock_flag"].astype(bool)]
    if config.exclude_stale_bars:
        eligible = eligible.loc[~eligible["stale_bar_flag"].astype(bool)]
    if "listing_date" in eligible:
        eligible = eligible.loc[eligible["listing_date"].le(effective_date)]
    if "last_trade_date" in eligible:
        eligible = eligible.loc[eligible["last_trade_date"].ge(effective_date)]
    return eligible


def _rank_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.copy()
    if "last_trade_date" not in ranked:
        ranked["last_trade_date"] = pd.Timestamp.max.normalize()
    return ranked.sort_values(
        ["open_interest", "volume", "last_trade_date", "contract"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )


def _ledger_row(
    *,
    product: str,
    trading_date: pd.Timestamp,
    decision_date: pd.Timestamp,
    status: str,
    selected_contract: str | None,
    previous_contract: str | None,
    roll_flag: bool,
    roll_reason: str,
    selection_open_interest: float | None = None,
    selection_volume: float | None = None,
    selected_source_row_id: str | None = None,
) -> dict[str, Any]:
    return {
        "product": product,
        "trading_date": trading_date,
        "decision_date": decision_date,
        "selection_status": status,
        "selected_contract": selected_contract,
        "previous_contract": previous_contract,
        "roll_flag": roll_flag,
        "roll_reason": roll_reason,
        "selection_open_interest": selection_open_interest,
        "selection_volume": selection_volume,
        "selected_source_row_id": selected_source_row_id,
    }


__all__ = [
    "BUILDER_ID",
    "ContinuousSeriesBuilder",
    "ContinuousSeriesConfig",
    "ContinuousSeriesResult",
    "LaggedLiquidityContinuousSeriesBuilder",
    "REQUIRED_PANEL_COLUMNS",
    "STAGE_OWNER",
    "build_continuous_series",
    "validate_continuous_series_result",
]
