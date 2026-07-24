"""Persistent, staggered cross-sectional sleeve construction.

This module translates an already-causal daily score panel into target weights.
It deliberately does not inspect realised or forward returns.  Selection is
performed before the tradability mask so a selected-but-untradable contract
becomes cash rather than causing the next-ranked contract to be substituted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


PERSISTENT_SLEEVE_SCHEMA_VERSION = 1
VALID_PERSISTENT_CONSTRUCTIONS = {
    "decile_equal",
    "quintile_equal",
    "quintile_inverse_vol",
    "continuous_rank_inverse_vol",
    "top_decile_broad_hedge",
    "asymmetric_inverse_vol",
    "rank_hysteresis_inverse_vol",
    "sparse_positive_event",
    "signed_score_inverse_vol",
    "signed_equal",
    "signed_power",
    "sector_matched_signed_inverse_vol",
    "sector_matched_signed_power",
    "global_sector_blend_signed_equal",
}
VALID_SIGNAL_ORIENTATIONS = {"higher_is_bullish", "higher_is_bearish"}
VALID_ZERO_SIGNAL_POLICIES = {"neutral"}
VALID_INCUMBENT_INELIGIBILITY_POLICIES = {
    "exit_at_next_executable_open",
}
INVERSE_VOL_CONSTRUCTIONS = {
    "quintile_inverse_vol",
    "continuous_rank_inverse_vol",
    "asymmetric_inverse_vol",
    "rank_hysteresis_inverse_vol",
    "signed_score_inverse_vol",
    "sector_matched_signed_inverse_vol",
}
POWER_CONSTRUCTIONS = {
    "signed_power",
    "sector_matched_signed_power",
}
SECTOR_CONSTRUCTIONS = {
    "sector_matched_signed_inverse_vol",
    "sector_matched_signed_power",
    "global_sector_blend_signed_equal",
}


class PersistentSleeveAlignmentError(ValueError):
    """Raised when an input score panel lacks a causal-alignment attestation."""


@dataclass(frozen=True, slots=True)
class PersistentSleeveConfig:
    """Frozen recipe for one persistent sleeve."""

    sleeve_id: str
    factor_id: str
    market_vertical: str
    construction: str
    signal_col: str = "alpha_score"
    date_col: str = "date"
    product_col: str = "ticker"
    rank_eligible_col: str = "rank_eligible"
    tradable_col: str = "tradable"
    volatility_col: str = "trailing_volatility"
    sector_col: str = "execution_sector"
    signal_orientation: str = "higher_is_bullish"
    zero_signal_policy: str = "neutral"
    incumbent_ineligibility_policy: str = "exit_at_next_executable_open"
    long_fraction: float = 0.10
    short_fraction: float = 0.10
    holding_periods: int = 1
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 0.0
    max_weight_per_contract: float | None = 0.10
    minimum_cross_section: int = 20
    minimum_distinct_signals: int = 2
    long_entry_rank: float = 0.90
    long_exit_rank: float = 0.65
    short_entry_rank: float = 0.10
    short_exit_rank: float = 0.35
    score_power: float = 1.0
    inverse_vol_power: float = 1.0
    sector_blend_weight: float = 0.50
    cohort_age_weights: tuple[float, ...] | None = None
    terminal_cash: bool = True
    preserve_untradable_as_cash: bool = True
    replacement_permitted: bool = False
    rescale_after_caps: bool = False
    future_return_eligibility_permitted: bool = False
    optimization_permitted: bool = False
    schema_version: int = PERSISTENT_SLEEVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "sleeve_id",
            "factor_id",
            "market_vertical",
            "signal_col",
            "date_col",
            "product_col",
            "rank_eligible_col",
            "tradable_col",
            "volatility_col",
            "sector_col",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)

        construction = str(self.construction).strip().lower()
        orientation = str(self.signal_orientation).strip().lower()
        zero_signal_policy = str(self.zero_signal_policy).strip().lower()
        incumbent_policy = str(
            self.incumbent_ineligibility_policy
        ).strip().lower()
        if construction not in VALID_PERSISTENT_CONSTRUCTIONS:
            raise ValueError(f"unknown persistent construction: {construction}")
        if orientation not in VALID_SIGNAL_ORIENTATIONS:
            raise ValueError(f"unknown signal orientation: {orientation}")
        if zero_signal_policy not in VALID_ZERO_SIGNAL_POLICIES:
            raise ValueError(
                f"unknown persistent zero-signal policy: {zero_signal_policy}"
            )
        if incumbent_policy not in VALID_INCUMBENT_INELIGIBILITY_POLICIES:
            raise ValueError(
                "unknown persistent incumbent-ineligibility policy: "
                f"{incumbent_policy}"
            )
        if int(self.holding_periods) < 1:
            raise ValueError("holding_periods must be positive")
        if int(self.minimum_cross_section) < 1:
            raise ValueError("minimum_cross_section must be positive")
        if int(self.minimum_distinct_signals) < 1:
            raise ValueError("minimum_distinct_signals must be positive")
        for field in ("long_fraction", "short_fraction"):
            value = float(getattr(self, field))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{field} must be in (0, 1)")
            object.__setattr__(self, field, value)
        gross = float(self.target_gross_exposure)
        net = float(self.target_net_exposure)
        if not 0.0 < gross <= 2.0:
            raise ValueError("target_gross_exposure must be in (0, 2]")
        if abs(net) > gross:
            raise ValueError("absolute net exposure cannot exceed gross exposure")
        if construction == "sparse_positive_event" and not math.isclose(net, gross):
            raise ValueError("sparse_positive_event must be long-only")
        if construction != "sparse_positive_event" and not (-gross < net < gross):
            raise ValueError("long-short constructions require both leg budgets")
        if self.max_weight_per_contract is not None and not (
            0.0 < float(self.max_weight_per_contract) <= 1.0
        ):
            raise ValueError("max_weight_per_contract must be in (0, 1]")
        if construction in {
            "decile_equal",
            "quintile_equal",
            "quintile_inverse_vol",
            "asymmetric_inverse_vol",
            "top_decile_broad_hedge",
        } and self.long_fraction + self.short_fraction > 1.0:
            raise ValueError("long and short selection fractions cannot exceed one")
        score_power = float(self.score_power)
        inverse_vol_power = float(self.inverse_vol_power)
        if score_power not in {0.0, 1.0}:
            raise ValueError("score_power must be frozen at 0 or 1")
        if inverse_vol_power not in {0.0, 1.0}:
            raise ValueError("inverse_vol_power must be frozen at 0 or 1")
        sector_blend_weight = float(self.sector_blend_weight)
        if not 0.0 <= sector_blend_weight <= 1.0:
            raise ValueError("sector_blend_weight must be in [0, 1]")
        cohort_age_weights = self.cohort_age_weights
        if cohort_age_weights is not None:
            try:
                cohort_age_weights = tuple(
                    float(weight) for weight in cohort_age_weights
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cohort_age_weights must be a finite numeric sequence"
                ) from exc
            if len(cohort_age_weights) != int(self.holding_periods):
                raise ValueError(
                    "cohort_age_weights must have one weight per holding period"
                )
            if any(
                not np.isfinite(weight) or weight <= 0.0
                for weight in cohort_age_weights
            ):
                raise ValueError(
                    "cohort_age_weights must contain finite positive weights"
                )
            if not math.isclose(
                sum(cohort_age_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("cohort_age_weights must sum to one")
        if not (
            0.5 <= float(self.long_exit_rank) < float(self.long_entry_rank) <= 1.0
        ):
            raise ValueError("long hysteresis ranks are not ordered")
        if not (
            0.0 <= float(self.short_entry_rank) < float(self.short_exit_rank) <= 0.5
        ):
            raise ValueError("short hysteresis ranks are not ordered")
        if not bool(self.preserve_untradable_as_cash):
            raise ValueError("persistent sleeves must preserve untradable selections as cash")
        if bool(self.replacement_permitted):
            raise ValueError("persistent sleeves cannot replace unavailable selections")
        if bool(self.rescale_after_caps):
            raise ValueError("persistent sleeves cannot rescale after contract caps")
        if bool(self.future_return_eligibility_permitted):
            raise ValueError("future returns cannot determine sleeve eligibility")
        if bool(self.optimization_permitted):
            raise ValueError("registered persistent sleeves cannot permit optimization")

        object.__setattr__(self, "construction", construction)
        object.__setattr__(self, "signal_orientation", orientation)
        object.__setattr__(self, "zero_signal_policy", zero_signal_policy)
        object.__setattr__(
            self,
            "incumbent_ineligibility_policy",
            incumbent_policy,
        )
        object.__setattr__(self, "holding_periods", int(self.holding_periods))
        object.__setattr__(
            self, "minimum_cross_section", int(self.minimum_cross_section)
        )
        object.__setattr__(
            self, "minimum_distinct_signals", int(self.minimum_distinct_signals)
        )
        object.__setattr__(self, "target_gross_exposure", gross)
        object.__setattr__(self, "target_net_exposure", net)
        object.__setattr__(self, "long_entry_rank", float(self.long_entry_rank))
        object.__setattr__(self, "long_exit_rank", float(self.long_exit_rank))
        object.__setattr__(self, "short_entry_rank", float(self.short_entry_rank))
        object.__setattr__(self, "short_exit_rank", float(self.short_exit_rank))
        object.__setattr__(self, "score_power", score_power)
        object.__setattr__(self, "inverse_vol_power", inverse_vol_power)
        object.__setattr__(
            self,
            "sector_blend_weight",
            sector_blend_weight,
        )
        object.__setattr__(
            self,
            "cohort_age_weights",
            cohort_age_weights,
        )

    @property
    def long_gross_budget(self) -> float:
        return (self.target_gross_exposure + self.target_net_exposure) / 2.0

    @property
    def short_gross_budget(self) -> float:
        return (self.target_gross_exposure - self.target_net_exposure) / 2.0

    @property
    def uses_inverse_volatility(self) -> bool:
        return (
            self.construction in INVERSE_VOL_CONSTRUCTIONS
            or (
                self.construction in POWER_CONSTRUCTIONS
                and self.inverse_vol_power > 0.0
            )
        )

    @property
    def uses_sector_matching(self) -> bool:
        return self.construction in SECTOR_CONSTRUCTIONS

    @property
    def resolved_cohort_age_weights(self) -> tuple[float, ...]:
        """Return frozen age-zero-first weights for active cohorts."""

        if self.cohort_age_weights is not None:
            return self.cohort_age_weights
        equal_weight = 1.0 / float(self.holding_periods)
        return (equal_weight,) * self.holding_periods

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve fingerprints for existing equal-age configs.  The optional
        # field is material only when a sleeve explicitly freezes a custom
        # cohort-age profile.
        if payload["cohort_age_weights"] is None:
            payload.pop("cohort_age_weights")
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PersistentSleeveResult:
    config: PersistentSleeveConfig
    positions: pd.DataFrame
    daily_summary: pd.DataFrame


def build_persistent_sleeve_targets(
    frame: pd.DataFrame,
    config: PersistentSleeveConfig,
) -> PersistentSleeveResult:
    """Build deterministic staggered targets without consulting future returns."""

    if not (
        bool(frame.attrs.get("causal_signal_alignment_verified"))
        or bool(frame.attrs.get("causal_return_alignment_verified"))
    ):
        raise PersistentSleeveAlignmentError(
            "persistent sleeve construction requires an upstream causal signal "
            "attestation"
        )
    required = {
        config.date_col,
        config.product_col,
        config.signal_col,
        config.rank_eligible_col,
        config.tradable_col,
    }
    if config.uses_inverse_volatility:
        required.add(config.volatility_col)
    if config.uses_sector_matching:
        required.add(config.sector_col)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"persistent sleeve input is missing columns: {missing}")

    source_attrs = dict(frame.attrs)
    source = frame.copy()
    source[config.date_col] = pd.to_datetime(
        source[config.date_col], errors="coerce"
    ).dt.normalize()
    source[config.product_col] = (
        source[config.product_col].astype("string").str.strip()
    )
    invalid_key = source[config.date_col].isna() | source[config.product_col].isna()
    invalid_key |= source[config.product_col].eq("")
    if bool(invalid_key.any()):
        raise ValueError("persistent sleeve input contains invalid date/product keys")
    if source.duplicated([config.date_col, config.product_col]).any():
        raise ValueError("persistent sleeve input requires unique date/product rows")
    source[config.signal_col] = _numeric(source[config.signal_col])
    source[config.rank_eligible_col] = _boolean(source[config.rank_eligible_col])
    source[config.tradable_col] = _boolean(source[config.tradable_col])
    if config.uses_inverse_volatility:
        source[config.volatility_col] = _numeric(source[config.volatility_col])
    if config.uses_sector_matching:
        source[config.sector_col] = (
            source[config.sector_col].astype("string").str.strip()
        )

    dates = pd.Index(source[config.date_col].drop_duplicates().sort_values())
    products = pd.Index(
        sorted(source[config.product_col].astype(str).drop_duplicates())
    )
    grid = pd.MultiIndex.from_product(
        [dates, products], names=[config.date_col, config.product_col]
    ).to_frame(index=False)
    out = grid.merge(
        source,
        on=[config.date_col, config.product_col],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    out[config.rank_eligible_col] = _boolean(out[config.rank_eligible_col])
    out[config.tradable_col] = _boolean(out[config.tradable_col])
    out[config.signal_col] = _numeric(out[config.signal_col])
    if config.uses_inverse_volatility:
        out[config.volatility_col] = _numeric(out[config.volatility_col])
        out["risk_eligible"] = (
            out[config.volatility_col].notna()
            & out[config.volatility_col].gt(0.0)
        )
    else:
        out["risk_eligible"] = True
    if config.uses_sector_matching:
        out[config.sector_col] = (
            out[config.sector_col].astype("string").str.strip()
        )

    out["oriented_signal"] = (
        out[config.signal_col]
        if config.signal_orientation == "higher_is_bullish"
        else -out[config.signal_col]
    )
    out["incumbent_eligible"] = (
        out[config.rank_eligible_col].astype(bool)
        & out["risk_eligible"].astype(bool)
        & out["oriented_signal"].notna()
    )
    out["rank_percentile"] = np.nan
    out["selection_side"] = "flat"
    out["selected"] = False
    out["selected_untradable"] = False
    out["pre_cap_weight"] = 0.0
    out["contract_cap_bound"] = False
    out["cohort_weight"] = 0.0
    out["hysteresis_state"] = "flat"

    allowed_dates = _formation_dates(dates, config)
    out["formation_allowed"] = out[config.date_col].isin(allowed_dates)
    state = {product: "flat" for product in products}

    for date, index in out.groupby(config.date_col, sort=True).groups.items():
        group = out.loc[index]
        eligible = (
            group[config.rank_eligible_col]
            & group["oriented_signal"].notna()
        )
        ordered_index, ranks = _deterministic_ranks(group, eligible, config)
        if not ranks.empty:
            out.loc[ranks.index, "rank_percentile"] = ranks
        if (
            len(ordered_index) < config.minimum_cross_section
            or group.loc[ordered_index, "oriented_signal"].nunique()
            < config.minimum_distinct_signals
        ):
            if config.construction == "rank_hysteresis_inverse_vol":
                for row_index, product in zip(
                    group.index, group[config.product_col], strict=True
                ):
                    state[str(product)] = "flat"
                    out.loc[row_index, "hysteresis_state"] = "flat"
            continue

        pre_weight, selected, side, states = _formation_weights(
            group,
            ordered_index,
            ranks,
            config,
            state,
        )
        out.loc[group.index, "hysteresis_state"] = states
        if date not in allowed_dates:
            continue

        effective_tradable = (
            group[config.tradable_col].astype(bool)
            & group["risk_eligible"].astype(bool)
        )
        cap_bound = pd.Series(False, index=group.index, dtype=bool)
        cohort = pre_weight.copy()
        if config.max_weight_per_contract is not None:
            cap = float(config.max_weight_per_contract)
            cap_bound = pre_weight.abs().gt(cap + 1e-15)
            cohort = cohort.clip(-cap, cap)
        cohort = cohort.where(effective_tradable, 0.0)

        out.loc[group.index, "pre_cap_weight"] = pre_weight
        out.loc[group.index, "selected"] = selected
        out.loc[group.index, "selection_side"] = side
        out.loc[group.index, "selected_untradable"] = (
            selected & ~effective_tradable
        )
        out.loc[group.index, "contract_cap_bound"] = cap_bound
        out.loc[group.index, "cohort_weight"] = cohort

    out = out.sort_values(
        [config.product_col, config.date_col], kind="mergesort"
    )
    out["eligibility_segment"] = (
        ~out["incumbent_eligible"]
    ).groupby(out[config.product_col], sort=False).cumsum()
    cohort_age_weights = config.resolved_cohort_age_weights
    out["desired_target_weight"] = out.groupby(
        [config.product_col, "eligibility_segment"],
        sort=False,
    )["cohort_weight"].transform(
        lambda values: _aggregate_cohort_age_weights(
            values,
            cohort_age_weights,
        )
    )
    out["desired_target_weight"] = out["desired_target_weight"].where(
        out["incumbent_eligible"],
        0.0,
    )
    out["target_weight"] = out["desired_target_weight"]
    out["prior_target_weight"] = out.groupby(config.product_col, sort=False)[
        "target_weight"
    ].shift(1).fillna(0.0)
    out["target_turnover"] = (
        out["target_weight"] - out["prior_target_weight"]
    ).abs()
    out = out.sort_values(
        [config.date_col, config.product_col], kind="mergesort"
    ).reset_index(drop=True)

    if config.terminal_cash and len(dates):
        terminal = out[config.date_col].eq(dates[-1])
        if not bool(out.loc[terminal, "target_weight"].abs().le(1e-12).all()):
            raise RuntimeError("terminal-cash construction failed to close final targets")

    daily = _daily_summary(out, config)
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["persistent_sleeve_config"] = config.to_dict()
    out.attrs["persistent_sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["future_return_eligibility_used"] = False
    return PersistentSleeveResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _formation_dates(
    dates: pd.Index,
    config: PersistentSleeveConfig,
) -> set[pd.Timestamp]:
    if not config.terminal_cash:
        return set(dates)
    if len(dates) <= config.holding_periods:
        return set()
    return set(dates[: -config.holding_periods])


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _aggregate_cohort_age_weights(
    values: pd.Series,
    age_weights: tuple[float, ...],
) -> pd.Series:
    """Aggregate current and lagged cohorts with age zero first."""

    target = pd.Series(0.0, index=values.index, dtype=float)
    for age, weight in enumerate(age_weights):
        target += values.shift(age).fillna(0.0) * weight
    return target


def _boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return _numeric(values).fillna(0.0).ne(0.0)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _deterministic_ranks(
    group: pd.DataFrame,
    eligible: pd.Series,
    config: PersistentSleeveConfig,
) -> tuple[pd.Index, pd.Series]:
    ordered = (
        group.loc[eligible, ["oriented_signal", config.product_col]]
        .assign(_product=lambda value: value[config.product_col].astype(str))
        .sort_values(["oriented_signal", "_product"], kind="mergesort")
    )
    ranks = group.loc[eligible, "oriented_signal"].rank(
        method="average",
        pct=True,
    )
    return ordered.index, ranks


def _formation_weights(
    group: pd.DataFrame,
    ordered_index: pd.Index,
    ranks: pd.Series,
    config: PersistentSleeveConfig,
    state: dict[str, str],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    weights = pd.Series(0.0, index=group.index, dtype=float)
    selected = pd.Series(False, index=group.index, dtype=bool)
    side = pd.Series("flat", index=group.index, dtype="string")
    states = pd.Series("flat", index=group.index, dtype="string")
    if config.construction == "decile_equal":
        long_index, short_index = _signed_tail_indices(
            group,
            ranks,
            config.long_fraction,
            config.short_fraction,
        )
        selected.loc[long_index.union(short_index)] = True
        if len(long_index):
            weights.loc[long_index] = (
                config.long_gross_budget / len(long_index)
            )
        if len(short_index):
            weights.loc[short_index] = (
                -config.short_gross_budget / len(short_index)
            )
    elif config.construction == "quintile_equal":
        long_index, short_index = _signed_tail_indices(
            group,
            ranks,
            config.long_fraction,
            config.short_fraction,
        )
        selected.loc[long_index.union(short_index)] = True
        if len(long_index):
            weights.loc[long_index] = (
                config.long_gross_budget / len(long_index)
            )
        if len(short_index):
            weights.loc[short_index] = (
                -config.short_gross_budget / len(short_index)
            )
    elif config.construction == "quintile_inverse_vol":
        long_index, short_index = _signed_tail_indices(
            group,
            ranks,
            config.long_fraction,
            config.short_fraction,
        )
        selected.loc[long_index.union(short_index)] = True
        weights += _inverse_vol_leg(
            group, long_index, config.long_gross_budget, 1.0, config
        )
        weights += _inverse_vol_leg(
            group, short_index, config.short_gross_budget, -1.0, config
        )
    elif config.construction == "continuous_rank_inverse_vol":
        long_index = ranks.index[
            ranks.gt(0.5)
            & group.loc[ranks.index, "oriented_signal"].gt(0.0)
        ]
        short_index = ranks.index[
            ranks.lt(0.5)
            & group.loc[ranks.index, "oriented_signal"].lt(0.0)
        ]
        selected.loc[long_index.union(short_index)] = True
        weights += _inverse_vol_leg(
            group,
            long_index,
            config.long_gross_budget,
            1.0,
            config,
            intensity=(ranks.loc[long_index] - 0.5),
        )
        weights += _inverse_vol_leg(
            group,
            short_index,
            config.short_gross_budget,
            -1.0,
            config,
            intensity=(0.5 - ranks.loc[short_index]),
        )
    elif config.construction == "top_decile_broad_hedge":
        long_index, _ = _signed_tail_indices(
            group,
            ranks,
            config.long_fraction,
            config.short_fraction,
        )
        selected.loc[ordered_index] = True
        if len(long_index):
            weights.loc[long_index] += (
                config.long_gross_budget / len(long_index)
            )
        if len(ordered_index):
            weights.loc[ordered_index] -= (
                config.short_gross_budget / len(ordered_index)
            )
    elif config.construction == "asymmetric_inverse_vol":
        long_index, short_index = _signed_tail_indices(
            group,
            ranks,
            config.long_fraction,
            config.short_fraction,
        )
        selected.loc[long_index.union(short_index)] = True
        weights += _inverse_vol_leg(
            group, long_index, config.long_gross_budget, 1.0, config
        )
        weights += _inverse_vol_leg(
            group, short_index, config.short_gross_budget, -1.0, config
        )
    elif config.construction == "rank_hysteresis_inverse_vol":
        long_members: list[int] = []
        short_members: list[int] = []
        valid_ranks = ranks.to_dict()
        for row_index, product in zip(
            group.index, group[config.product_col], strict=True
        ):
            product_key = str(product)
            rank = valid_ranks.get(row_index)
            signal = group.loc[row_index, "oriented_signal"]
            previous = state.get(product_key, "flat")
            current = _next_hysteresis_state(
                previous,
                rank,
                signal,
                config,
            )
            state[product_key] = current
            states.loc[row_index] = current
            if current == "long":
                long_members.append(row_index)
            elif current == "short":
                short_members.append(row_index)
        long_index = pd.Index(long_members)
        short_index = pd.Index(short_members)
        selected.loc[long_index.union(short_index)] = True
        weights += _inverse_vol_leg(
            group, long_index, config.long_gross_budget, 1.0, config
        )
        weights += _inverse_vol_leg(
            group, short_index, config.short_gross_budget, -1.0, config
        )
    elif config.construction == "sparse_positive_event":
        event_index = group.index[
            group.index.isin(ordered_index) & group["oriented_signal"].gt(0.0)
        ]
        selected.loc[event_index] = True
        if len(event_index):
            weights.loc[event_index] = config.long_gross_budget / len(event_index)
    elif config.construction == "signed_score_inverse_vol":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        selected.loc[long_index.union(short_index)] = True
        strength = signed_signal.abs()
        weights += _inverse_vol_leg(
            group,
            long_index,
            config.long_gross_budget,
            1.0,
            config,
            intensity=strength.loc[long_index],
        )
        weights += _inverse_vol_leg(
            group,
            short_index,
            config.short_gross_budget,
            -1.0,
            config,
            intensity=strength.loc[short_index],
        )
    elif config.construction == "signed_equal":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        selected.loc[long_index.union(short_index)] = True
        if len(long_index):
            weights.loc[long_index] = (
                config.long_gross_budget / len(long_index)
            )
        if len(short_index):
            weights.loc[short_index] = (
                -config.short_gross_budget / len(short_index)
            )
    elif config.construction == "signed_power":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        selected.loc[long_index.union(short_index)] = True
        weights += _power_weight_leg(
            group,
            long_index,
            config.long_gross_budget,
            1.0,
            config,
            intensity=signed_signal.abs(),
        )
        weights += _power_weight_leg(
            group,
            short_index,
            config.short_gross_budget,
            -1.0,
            config,
            intensity=signed_signal.abs(),
        )
    elif config.construction == "sector_matched_signed_inverse_vol":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        sector_weights, matched = _sector_matched_inverse_vol_weights(
            group,
            long_index,
            short_index,
            config,
            intensity=signed_signal.abs(),
        )
        selected.loc[matched] = True
        weights += sector_weights
    elif config.construction == "sector_matched_signed_power":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        sector_weights, matched = _sector_matched_power_weights(
            group,
            long_index,
            short_index,
            config,
            intensity=signed_signal.abs(),
        )
        selected.loc[matched] = True
        weights += sector_weights
    elif config.construction == "global_sector_blend_signed_equal":
        signed_signal = group.loc[ordered_index, "oriented_signal"]
        long_index = signed_signal.index[signed_signal.gt(0.0)]
        short_index = signed_signal.index[signed_signal.lt(0.0)]
        global_weights = _signed_equal_weights(
            group,
            long_index,
            short_index,
            config,
        )
        sector_weights, matched = _sector_matched_power_weights(
            group,
            long_index,
            short_index,
            config,
            intensity=signed_signal.abs(),
            score_power=0.0,
            inverse_vol_power=0.0,
        )
        selected.loc[long_index.union(short_index).union(matched)] = True
        sector_share = config.sector_blend_weight
        weights += (1.0 - sector_share) * global_weights
        weights += sector_share * sector_weights
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported persistent construction: {config.construction}")

    side.loc[weights.gt(0.0)] = "long"
    side.loc[weights.lt(0.0)] = "short"
    return weights, selected, side, states


def _signed_tail_indices(
    group: pd.DataFrame,
    ranks: pd.Series,
    long_fraction: float,
    short_fraction: float,
) -> tuple[pd.Index, pd.Index]:
    signal = group.loc[ranks.index, "oriented_signal"]
    long_index = ranks.index[
        ranks.gt(1.0 - float(long_fraction)) & signal.gt(0.0)
    ]
    short_index = ranks.index[
        ranks.le(float(short_fraction)) & signal.lt(0.0)
    ]
    return long_index, short_index


def _inverse_vol_leg(
    group: pd.DataFrame,
    members: pd.Index,
    budget: float,
    sign: float,
    config: PersistentSleeveConfig,
    *,
    intensity: pd.Series | None = None,
) -> pd.Series:
    result = pd.Series(0.0, index=group.index, dtype=float)
    if not len(members) or budget <= 0.0:
        return result
    volatility = group.loc[members, config.volatility_col]
    raw = pd.Series(np.nan, index=members, dtype=float)
    valid = volatility.notna() & volatility.gt(0.0)
    raw.loc[valid] = 1.0 / volatility.loc[valid]
    if intensity is not None:
        raw *= intensity.reindex(members)
    finite_positive = raw.replace([np.inf, -np.inf], np.nan).dropna()
    finite_positive = finite_positive[finite_positive.gt(0.0)]
    fallback = float(finite_positive.median()) if len(finite_positive) else 1.0
    raw = raw.fillna(fallback).clip(lower=0.0)
    denominator = float(raw.sum())
    if denominator > 0.0 and np.isfinite(denominator):
        result.loc[members] = sign * budget * raw / denominator
    return result


def _power_weight_leg(
    group: pd.DataFrame,
    members: pd.Index,
    budget: float,
    sign: float,
    config: PersistentSleeveConfig,
    *,
    intensity: pd.Series,
    score_power: float | None = None,
    inverse_vol_power: float | None = None,
) -> pd.Series:
    """Allocate a leg under frozen score- and inverse-volatility exponents."""

    result = pd.Series(0.0, index=group.index, dtype=float)
    if not len(members) or budget <= 0.0:
        return result
    score_exponent = (
        config.score_power if score_power is None else float(score_power)
    )
    volatility_exponent = (
        config.inverse_vol_power
        if inverse_vol_power is None
        else float(inverse_vol_power)
    )
    raw = pd.Series(1.0, index=members, dtype=float)
    if score_exponent > 0.0:
        strength = pd.to_numeric(
            intensity.reindex(members), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        raw *= strength.abs().pow(score_exponent)
    if volatility_exponent > 0.0:
        volatility = pd.to_numeric(
            group.loc[members, config.volatility_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        raw *= volatility.where(volatility.gt(0.0)).pow(
            -volatility_exponent
        )
    finite_positive = raw.replace([np.inf, -np.inf], np.nan).dropna()
    finite_positive = finite_positive[finite_positive.gt(0.0)]
    fallback = float(finite_positive.median()) if len(finite_positive) else 1.0
    raw = raw.fillna(fallback).clip(lower=0.0)
    denominator = float(raw.sum())
    if denominator > 0.0 and np.isfinite(denominator):
        result.loc[members] = sign * budget * raw / denominator
    return result


def _signed_equal_weights(
    group: pd.DataFrame,
    long_index: pd.Index,
    short_index: pd.Index,
    config: PersistentSleeveConfig,
) -> pd.Series:
    """Return a sign-preserving equal-weight book with independent leg cash."""

    result = pd.Series(0.0, index=group.index, dtype=float)
    if len(long_index):
        result.loc[long_index] = config.long_gross_budget / len(long_index)
    if len(short_index):
        result.loc[short_index] = (
            -config.short_gross_budget / len(short_index)
        )
    return result


def _next_hysteresis_state(
    previous: str,
    rank: float | None,
    signal: float | None,
    config: PersistentSleeveConfig,
) -> str:
    if (
        rank is None
        or signal is None
        or not np.isfinite(rank)
        or not np.isfinite(signal)
        or signal == 0.0
    ):
        return "flat"
    if (
        previous == "long"
        and signal > 0.0
        and rank >= config.long_exit_rank
    ):
        return "long"
    if (
        previous == "short"
        and signal < 0.0
        and rank <= config.short_exit_rank
    ):
        return "short"
    if signal > 0.0 and rank >= config.long_entry_rank:
        return "long"
    if signal < 0.0 and rank <= config.short_entry_rank:
        return "short"
    return "flat"


def _sector_matched_inverse_vol_weights(
    group: pd.DataFrame,
    long_index: pd.Index,
    short_index: pd.Index,
    config: PersistentSleeveConfig,
    *,
    intensity: pd.Series,
) -> tuple[pd.Series, pd.Index]:
    """Allocate equal long/short risk budgets only in two-sided sectors."""

    result = pd.Series(0.0, index=group.index, dtype=float)
    if not len(long_index) or not len(short_index):
        return result, pd.Index([])
    long_sector = group.loc[long_index, config.sector_col]
    short_sector = group.loc[short_index, config.sector_col]
    matched_sectors = sorted(
        set(long_sector.dropna().astype(str))
        & set(short_sector.dropna().astype(str))
    )
    if not matched_sectors:
        return result, pd.Index([])
    long_budget = config.long_gross_budget / len(matched_sectors)
    short_budget = config.short_gross_budget / len(matched_sectors)
    matched_members = pd.Index([])
    for sector in matched_sectors:
        sector_long = long_sector.index[long_sector.astype(str).eq(sector)]
        sector_short = short_sector.index[short_sector.astype(str).eq(sector)]
        matched_members = matched_members.union(sector_long).union(sector_short)
        result += _inverse_vol_leg(
            group,
            sector_long,
            long_budget,
            1.0,
            config,
            intensity=intensity.reindex(sector_long),
        )
        result += _inverse_vol_leg(
            group,
            sector_short,
            short_budget,
            -1.0,
            config,
            intensity=intensity.reindex(sector_short),
        )
    return result, matched_members


def _sector_matched_power_weights(
    group: pd.DataFrame,
    long_index: pd.Index,
    short_index: pd.Index,
    config: PersistentSleeveConfig,
    *,
    intensity: pd.Series,
    score_power: float | None = None,
    inverse_vol_power: float | None = None,
) -> tuple[pd.Series, pd.Index]:
    """Allocate equal budgets to sectors that contain both forecast signs."""

    result = pd.Series(0.0, index=group.index, dtype=float)
    if not len(long_index) or not len(short_index):
        return result, pd.Index([])
    long_sector = group.loc[long_index, config.sector_col]
    short_sector = group.loc[short_index, config.sector_col]
    matched_sectors = sorted(
        set(long_sector.dropna().astype(str))
        & set(short_sector.dropna().astype(str))
    )
    if not matched_sectors:
        return result, pd.Index([])
    long_budget = config.long_gross_budget / len(matched_sectors)
    short_budget = config.short_gross_budget / len(matched_sectors)
    matched_members = pd.Index([])
    for sector in matched_sectors:
        sector_long = long_sector.index[long_sector.astype(str).eq(sector)]
        sector_short = short_sector.index[short_sector.astype(str).eq(sector)]
        matched_members = matched_members.union(sector_long).union(sector_short)
        result += _power_weight_leg(
            group,
            sector_long,
            long_budget,
            1.0,
            config,
            intensity=intensity,
            score_power=score_power,
            inverse_vol_power=inverse_vol_power,
        )
        result += _power_weight_leg(
            group,
            sector_short,
            short_budget,
            -1.0,
            config,
            intensity=intensity,
            score_power=score_power,
            inverse_vol_power=inverse_vol_power,
        )
    return result, matched_members


def _daily_summary(
    positions: pd.DataFrame,
    config: PersistentSleeveConfig,
) -> pd.DataFrame:
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=("target_weight", lambda value: float(value.abs().sum())),
        net_exposure=("target_weight", "sum"),
        long_exposure=(
            "target_weight",
            lambda value: float(value.clip(lower=0.0).sum()),
        ),
        short_exposure=(
            "target_weight",
            lambda value: float((-value.clip(upper=0.0)).sum()),
        ),
        cohort_gross=("cohort_weight", lambda value: float(value.abs().sum())),
        target_turnover=("target_turnover", "sum"),
        active_products=("target_weight", lambda value: int(value.ne(0.0).sum())),
        selected_products=("selected", "sum"),
        selected_untradable_products=("selected_untradable", "sum"),
        contract_cap_count=("contract_cap_bound", "sum"),
        formation_allowed=("formation_allowed", "max"),
    ).reset_index()
    daily["gross_realization"] = (
        daily["gross_exposure"] / config.target_gross_exposure
    )
    return daily


__all__ = [
    "INVERSE_VOL_CONSTRUCTIONS",
    "SECTOR_CONSTRUCTIONS",
    "PERSISTENT_SLEEVE_SCHEMA_VERSION",
    "VALID_INCUMBENT_INELIGIBILITY_POLICIES",
    "VALID_PERSISTENT_CONSTRUCTIONS",
    "VALID_ZERO_SIGNAL_POLICIES",
    "PersistentSleeveAlignmentError",
    "PersistentSleeveConfig",
    "PersistentSleeveResult",
    "build_persistent_sleeve_targets",
]
