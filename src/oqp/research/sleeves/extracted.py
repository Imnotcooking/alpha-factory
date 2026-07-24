"""Contracts for sleeve rules extracted from legacy hybrid factor recipes.

The normal sleeve engines cover the preferred reusable portfolio geometries.
This module owns the smaller set of stateful lifecycle rules that existed
inside historical ``fac_*`` implementations.  Keeping their contracts here
allows the factor files to expose only causal forecasts while preserving the
old construction choices as independently identifiable research components.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


EXTRACTED_SLEEVE_SCHEMA_VERSION = 1
VALID_EXTRACTED_RULE_FAMILIES = {
    "proportional_score",
    "opposite_event_state",
    "cross_sectional_z_tail",
    "decision_refresh",
    "top_bottom_n",
    "fixed_event_hold",
    "residual_event_ttl",
    "atr_donchian_exit_state",
    "atr_chandelier_exit_state",
    "rank_hysteresis_long_only",
    "intraday_session_state",
    "intraday_fixed_weight_state",
    "ewm_event_decay",
    "indicator_specific_exit_state",
}
VALID_SIGNAL_ORIENTATIONS = {"higher_is_bullish", "higher_is_bearish"}


@dataclass(frozen=True, slots=True)
class ExtractedSleeveConfig:
    """Immutable definition of one extracted factor-to-position rule."""

    sleeve_id: str
    factor_id: str
    market_vertical: str
    rule_family: str
    source_factor_ids: tuple[str, ...]
    signal_orientation: str = "higher_is_bullish"
    signal_col: str = "factor_score"
    date_col: str = "date"
    product_col: str = "ticker"
    required_signal_columns: tuple[str, ...] = ("factor_score",)
    parameters: Mapping[str, Any] | None = None
    output_col: str = "target_weight"
    execution_supported: bool = False
    schema_version: int = EXTRACTED_SLEEVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "sleeve_id",
            "factor_id",
            "market_vertical",
            "signal_col",
            "date_col",
            "product_col",
            "output_col",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)

        family = str(self.rule_family).strip().lower()
        if family not in VALID_EXTRACTED_RULE_FAMILIES:
            raise ValueError(f"unknown extracted sleeve rule family: {family}")
        orientation = str(self.signal_orientation).strip().lower()
        if orientation not in VALID_SIGNAL_ORIENTATIONS:
            raise ValueError(f"unknown signal orientation: {orientation}")
        sources = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in self.source_factor_ids
                if str(value).strip()
            )
        )
        if not sources:
            raise ValueError("source_factor_ids must contain at least one factor")
        required = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in self.required_signal_columns
                if str(value).strip()
            )
        )
        if self.signal_col not in required:
            required = (self.signal_col, *required)
        parameters = dict(self.parameters or {})
        object.__setattr__(self, "rule_family", family)
        object.__setattr__(self, "signal_orientation", orientation)
        object.__setattr__(self, "source_factor_ids", sources)
        object.__setattr__(self, "required_signal_columns", required)
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sleeve_id": self.sleeve_id,
            "factor_id": self.factor_id,
            "market_vertical": self.market_vertical,
            "rule_family": self.rule_family,
            "source_factor_ids": list(self.source_factor_ids),
            "signal_orientation": self.signal_orientation,
            "signal_col": self.signal_col,
            "date_col": self.date_col,
            "product_col": self.product_col,
            "required_signal_columns": list(self.required_signal_columns),
            "parameters": dict(self.parameters or {}),
            "output_col": self.output_col,
            "execution_supported": self.execution_supported,
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EXTRACTED_SLEEVE_SCHEMA_VERSION",
    "ExtractedSleeveConfig",
    "VALID_EXTRACTED_RULE_FAMILIES",
]
