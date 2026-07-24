"""Versioned contracts for translating one pure factor into target positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


SLEEVE_CONSTRUCTION_SCHEMA_VERSION = 1
VALID_GEOMETRIES = {"cross_sectional", "time_series"}
VALID_EXPRESSIONS = {"long_short", "long_only", "short_only", "directional"}
VALID_CONSTRUCTIONS = {
    "top_bottom_quantile",
    "continuous_rank",
    "continuous_zscore",
    "proportional_score",
    "time_series_sign",
}
VALID_NORMALIZATIONS = {
    "absolute_score_to_gross",
    "equal_weight",
    "rank_weight",
    "zscore_weight",
}
VALID_MISSING_POLICIES = {"neutral", "exclude"}
VALID_ZERO_SIGNAL_POLICIES = {"eligible", "neutral"}
VALID_SIGNAL_TIMINGS = {"already_lagged", "decision_time"}
VALID_SIGNAL_ORIENTATIONS = {"higher_is_bullish", "higher_is_bearish"}
VALID_RETURN_ASSUMPTIONS = {
    "close_signal_next_open_to_close",
    "close_signal_next_open_to_next_open",
}


@dataclass(frozen=True, slots=True)
class SleeveConstructionConfig:
    sleeve_id: str
    factor_id: str
    market_vertical: str
    construction_geometry: str = "cross_sectional"
    expression: str = "long_short"
    construction: str = "top_bottom_quantile"
    normalization: str = "equal_weight"
    signal_col: str = "alpha_score"
    date_col: str = "date"
    product_col: str = "ticker"
    sector_col: str = "sector"
    split_col: str = "research_split"
    return_col: str = "forward_return"
    return_assumption: str = "close_signal_next_open_to_close"
    signal_orientation: str = "higher_is_bullish"
    winsor_lower_quantile: float | None = 0.01
    winsor_upper_quantile: float | None = 0.99
    long_fraction: float = 0.20
    short_fraction: float = 0.20
    rebalance_every_n_periods: int = 1
    holding_periods: int = 1
    max_weight_per_contract: float | None = 0.05
    max_sector_gross: float | None = None
    sector_cap_reason: str = ""
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 0.0
    missing_signal_policy: str = "neutral"
    zero_signal_policy: str = "eligible"
    signal_timing: str = "already_lagged"
    execution_delay_periods: int = 0
    minimum_cross_section: int = 10
    minimum_distinct_signals: int = 2
    optimization_permitted: bool = False
    schema_version: int = SLEEVE_CONSTRUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "sleeve_id",
            "factor_id",
            "market_vertical",
            "signal_col",
            "date_col",
            "product_col",
            "sector_col",
            "split_col",
            "return_col",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)

        geometry = str(self.construction_geometry).strip().lower()
        expression = str(self.expression).strip().lower()
        construction = str(self.construction).strip().lower()
        normalization = str(self.normalization).strip().lower()
        missing_policy = str(self.missing_signal_policy).strip().lower()
        zero_signal_policy = str(self.zero_signal_policy).strip().lower()
        signal_timing = str(self.signal_timing).strip().lower()
        return_assumption = str(self.return_assumption).strip().lower()
        signal_orientation = str(self.signal_orientation).strip().lower()
        if geometry not in VALID_GEOMETRIES:
            raise ValueError(f"unknown construction geometry: {geometry}")
        if expression not in VALID_EXPRESSIONS:
            raise ValueError(f"unknown sleeve expression: {expression}")
        if construction not in VALID_CONSTRUCTIONS:
            raise ValueError(f"unknown sleeve construction: {construction}")
        if normalization not in VALID_NORMALIZATIONS:
            raise ValueError(f"unknown sleeve normalization: {normalization}")
        if missing_policy not in VALID_MISSING_POLICIES:
            raise ValueError(f"unknown missing signal policy: {missing_policy}")
        if zero_signal_policy not in VALID_ZERO_SIGNAL_POLICIES:
            raise ValueError(f"unknown zero signal policy: {zero_signal_policy}")
        if signal_timing not in VALID_SIGNAL_TIMINGS:
            raise ValueError(f"unknown signal timing: {signal_timing}")
        if return_assumption not in VALID_RETURN_ASSUMPTIONS:
            raise ValueError(f"unknown return assumption: {return_assumption}")
        if signal_orientation not in VALID_SIGNAL_ORIENTATIONS:
            raise ValueError(f"unknown signal orientation: {signal_orientation}")
        if geometry == "cross_sectional" and construction == "time_series_sign":
            raise ValueError("time_series_sign requires time_series geometry")
        if geometry == "time_series" and construction not in {
            "proportional_score",
            "time_series_sign",
        }:
            raise ValueError(
                "time_series geometry requires time_series_sign or "
                "proportional_score"
            )
        if construction == "top_bottom_quantile" and expression == "directional":
            raise ValueError("top_bottom_quantile does not support directional expression")
        if normalization == "rank_weight" and construction == "time_series_sign":
            raise ValueError("rank_weight is not defined for time_series_sign")
        if construction == "top_bottom_quantile" and normalization != "equal_weight":
            raise ValueError("top_bottom_quantile currently requires equal_weight")
        if construction == "continuous_rank" and normalization != "rank_weight":
            raise ValueError("continuous_rank currently requires rank_weight")
        if construction == "continuous_zscore" and normalization != "zscore_weight":
            raise ValueError("continuous_zscore requires zscore_weight")
        if normalization == "zscore_weight" and construction != "continuous_zscore":
            raise ValueError("zscore_weight is only defined for continuous_zscore")
        if (
            construction == "proportional_score"
            and normalization != "absolute_score_to_gross"
        ):
            raise ValueError(
                "proportional_score requires absolute_score_to_gross"
            )
        if (
            normalization == "absolute_score_to_gross"
            and construction != "proportional_score"
        ):
            raise ValueError(
                "absolute_score_to_gross is only defined for proportional_score"
            )
        if construction == "proportional_score" and expression != "directional":
            raise ValueError(
                "proportional_score requires directional expression"
            )
        if not 0.0 < float(self.long_fraction) < 0.5:
            raise ValueError("long_fraction must be in (0, 0.5)")
        if not 0.0 < float(self.short_fraction) < 0.5:
            raise ValueError("short_fraction must be in (0, 0.5)")
        if self.winsor_lower_quantile is not None:
            if not 0.0 <= float(self.winsor_lower_quantile) < 0.5:
                raise ValueError("winsor_lower_quantile must be in [0, 0.5)")
        if self.winsor_upper_quantile is not None:
            if not 0.5 < float(self.winsor_upper_quantile) <= 1.0:
                raise ValueError("winsor_upper_quantile must be in (0.5, 1]")
        if (
            self.winsor_lower_quantile is not None
            and self.winsor_upper_quantile is not None
            and self.winsor_lower_quantile >= self.winsor_upper_quantile
        ):
            raise ValueError("winsor quantiles must be ordered")
        if int(self.rebalance_every_n_periods) < 1:
            raise ValueError("rebalance_every_n_periods must be positive")
        if int(self.holding_periods) < 1:
            raise ValueError("holding_periods must be positive")
        if int(self.execution_delay_periods) < 0:
            raise ValueError("execution_delay_periods cannot be negative")
        if int(self.minimum_cross_section) < 3:
            raise ValueError("minimum_cross_section must be at least 3")
        minimum_distinct_signals = int(self.minimum_distinct_signals)
        if construction == "time_series_sign":
            if minimum_distinct_signals < 1:
                raise ValueError(
                    "time_series_sign minimum_distinct_signals must be at least 1"
                )
        elif minimum_distinct_signals < 2:
            raise ValueError(
                "cross-sectional minimum_distinct_signals must be at least 2"
            )
        if self.max_weight_per_contract is not None and not (
            0.0 < float(self.max_weight_per_contract) <= 1.0
        ):
            raise ValueError("max_weight_per_contract must be in (0, 1]")
        if self.max_sector_gross is not None and not (
            0.0 < float(self.max_sector_gross) <= float(self.target_gross_exposure)
        ):
            raise ValueError("max_sector_gross must be positive and no larger than gross")
        gross = float(self.target_gross_exposure)
        net = float(self.target_net_exposure)
        if not 0.0 < gross <= 2.0:
            raise ValueError("target_gross_exposure must be in (0, 2]")
        if abs(net) > gross:
            raise ValueError("absolute net exposure cannot exceed gross exposure")
        if expression == "long_short" and not (-gross < net < gross):
            raise ValueError("long_short requires both long and short exposure")
        if expression == "long_only" and net != gross:
            raise ValueError("long_only requires target_net_exposure equal to gross")
        if expression == "short_only" and net != -gross:
            raise ValueError("short_only requires target_net_exposure equal to negative gross")
        if signal_timing == "decision_time" and self.execution_delay_periods < 1:
            raise ValueError("decision_time signals require at least one execution delay")
        if (
            return_assumption == "close_signal_next_open_to_close"
            and int(self.holding_periods) != 1
        ):
            raise ValueError("session-flat next-open-to-close returns require holding_periods=1")
        if bool(self.optimization_permitted):
            raise ValueError("Phase 3 fixed-default sleeves cannot permit optimization")

        object.__setattr__(self, "construction_geometry", geometry)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "construction", construction)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "missing_signal_policy", missing_policy)
        object.__setattr__(self, "zero_signal_policy", zero_signal_policy)
        object.__setattr__(self, "signal_timing", signal_timing)
        object.__setattr__(self, "return_assumption", return_assumption)
        object.__setattr__(self, "signal_orientation", signal_orientation)
        object.__setattr__(self, "long_fraction", float(self.long_fraction))
        object.__setattr__(self, "short_fraction", float(self.short_fraction))
        object.__setattr__(
            self, "rebalance_every_n_periods", int(self.rebalance_every_n_periods)
        )
        object.__setattr__(self, "holding_periods", int(self.holding_periods))
        object.__setattr__(
            self, "execution_delay_periods", int(self.execution_delay_periods)
        )
        object.__setattr__(self, "minimum_cross_section", int(self.minimum_cross_section))
        object.__setattr__(
            self, "minimum_distinct_signals", minimum_distinct_signals
        )
        object.__setattr__(self, "target_gross_exposure", gross)
        object.__setattr__(self, "target_net_exposure", net)

    @property
    def long_gross_budget(self) -> float:
        if self.expression == "short_only":
            return 0.0
        if self.expression == "long_only":
            return self.target_gross_exposure
        if self.expression == "directional":
            return self.target_gross_exposure
        return (self.target_gross_exposure + self.target_net_exposure) / 2.0

    @property
    def short_gross_budget(self) -> float:
        if self.expression == "long_only":
            return 0.0
        if self.expression == "short_only":
            return self.target_gross_exposure
        if self.expression == "directional":
            return self.target_gross_exposure
        return (self.target_gross_exposure - self.target_net_exposure) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SLEEVE_CONSTRUCTION_SCHEMA_VERSION",
    "SleeveConstructionConfig",
]
