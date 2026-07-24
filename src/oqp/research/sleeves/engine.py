"""Deterministic factor-score to target-position construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from oqp.research.sleeves.contracts import SleeveConstructionConfig


class SleeveAlignmentError(ValueError):
    """Raised when the input score is not proven executable at the target date."""


@dataclass(frozen=True, slots=True)
class SleeveConstructionResult:
    config: SleeveConstructionConfig
    positions: pd.DataFrame
    daily_summary: pd.DataFrame


def build_sleeve_targets(
    frame: pd.DataFrame,
    config: SleeveConstructionConfig,
) -> SleeveConstructionResult:
    required = {
        config.date_col,
        config.product_col,
        config.signal_col,
        config.sector_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"sleeve input is missing columns: {missing}")
    if not bool(frame.attrs.get("causal_return_alignment_verified")):
        raise SleeveAlignmentError(
            "sleeve construction requires an upstream causal signal attestation"
        )

    out = frame.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="coerce").dt.normalize()
    out[config.product_col] = out[config.product_col].astype("string").str.strip()
    out[config.sector_col] = (
        out[config.sector_col].astype("string").str.strip().fillna("Unknown")
    )
    out[config.signal_col] = pd.to_numeric(
        out[config.signal_col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[config.date_col, config.product_col])
    if out.duplicated([config.date_col, config.product_col]).any():
        raise ValueError("sleeve input requires unique date/product rows")
    out = out.sort_values(
        [config.date_col, config.product_col], kind="mergesort"
    ).reset_index(drop=True)

    dates = pd.Index(out[config.date_col].drop_duplicates().sort_values())
    rebalance_dates = set(dates[:: config.rebalance_every_n_periods])
    out["rebalance_decision"] = out[config.date_col].isin(rebalance_dates)
    out["signal_missing"] = out[config.signal_col].isna()
    out["signal_neutral"] = out[config.signal_col].eq(0.0)
    out["winsorized_signal"] = np.nan
    out["cross_sectional_rank"] = np.nan
    out["cross_sectional_zscore"] = np.nan
    out["selection_side"] = "flat"
    out["decision_target_weight"] = np.nan
    out["contract_cap_bound"] = False
    out["sector_cap_bound"] = False

    for date, index in out.groupby(config.date_col, sort=True).groups.items():
        if date not in rebalance_dates:
            continue
        group = out.loc[index]
        valid = group[config.signal_col].notna()
        if config.zero_signal_policy == "neutral":
            valid &= group[config.signal_col].ne(0.0)
        if int(valid.sum()) < config.minimum_cross_section:
            out.loc[index, "decision_target_weight"] = 0.0
            continue
        valid_index = group.index[valid]
        signal = group.loc[valid_index, config.signal_col]
        if config.signal_orientation == "higher_is_bearish":
            signal = -signal
        if int(signal.nunique(dropna=True)) < config.minimum_distinct_signals:
            out.loc[index, "decision_target_weight"] = 0.0
            continue
        winsorized = _winsorize(signal, config)
        out.loc[valid_index, "winsorized_signal"] = winsorized
        ranks = winsorized.rank(method="average", pct=True)
        out.loc[valid_index, "cross_sectional_rank"] = ranks
        standard_deviation = float(winsorized.std(ddof=0))
        if standard_deviation > 0.0 and np.isfinite(standard_deviation):
            zscores = (winsorized - float(winsorized.mean())) / standard_deviation
        else:
            zscores = pd.Series(0.0, index=winsorized.index, dtype=float)
        out.loc[valid_index, "cross_sectional_zscore"] = zscores

        if config.construction == "top_bottom_quantile":
            weights, side = _top_bottom_weights(group.loc[valid_index], winsorized, config)
        elif config.construction == "continuous_rank":
            weights, side = _continuous_rank_weights(
                group.loc[valid_index], ranks, config
            )
        elif config.construction == "continuous_zscore":
            weights, side = _continuous_zscore_weights(
                group.loc[valid_index], zscores, config
            )
        elif config.construction == "proportional_score":
            weights, side = _proportional_score_weights(
                group.loc[valid_index], winsorized, config
            )
        else:
            weights, side = _time_series_sign_weights(
                group.loc[valid_index], winsorized, config
            )
        capped, contract_bound, sector_bound = _apply_caps(
            group.loc[valid_index], weights, config
        )
        out.loc[index, "decision_target_weight"] = 0.0
        out.loc[valid_index, "decision_target_weight"] = capped
        out.loc[valid_index, "selection_side"] = side
        out.loc[valid_index, "contract_cap_bound"] = contract_bound
        out.loc[valid_index, "sector_cap_bound"] = sector_bound

    out = out.sort_values(
        [config.product_col, config.date_col], kind="mergesort"
    )
    held = out["decision_target_weight"].where(out["rebalance_decision"])
    if config.holding_periods == 1:
        out["held_target_weight"] = held
    else:
        out["held_target_weight"] = held.groupby(
            out[config.product_col], sort=False
        ).ffill(limit=config.holding_periods - 1)
    out["held_target_weight"] = out["held_target_weight"].fillna(0.0)
    if config.execution_delay_periods:
        out["target_weight"] = out.groupby(config.product_col, sort=False)[
            "held_target_weight"
        ].shift(config.execution_delay_periods).fillna(0.0)
    else:
        out["target_weight"] = out["held_target_weight"]
    out["prior_target_weight"] = out.groupby(config.product_col, sort=False)[
        "target_weight"
    ].shift(1).fillna(0.0)
    out["target_turnover"] = (
        out["target_weight"] - out["prior_target_weight"]
    ).abs()
    out = out.sort_values(
        [config.date_col, config.product_col], kind="mergesort"
    ).reset_index(drop=True)
    daily = _daily_summary(out, config)
    out.attrs.update(frame.attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    return SleeveConstructionResult(config=config, positions=out, daily_summary=daily)


def _winsorize(
    signal: pd.Series,
    config: SleeveConstructionConfig,
) -> pd.Series:
    lower = (
        float(signal.quantile(config.winsor_lower_quantile))
        if config.winsor_lower_quantile is not None
        else -math.inf
    )
    upper = (
        float(signal.quantile(config.winsor_upper_quantile))
        if config.winsor_upper_quantile is not None
        else math.inf
    )
    return signal.clip(lower=lower, upper=upper)


def _top_bottom_weights(
    group: pd.DataFrame,
    signal: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    count = len(group)
    long_count = max(int(math.floor(count * config.long_fraction)), 1)
    short_count = max(int(math.floor(count * config.short_fraction)), 1)
    ordered = pd.DataFrame(
        {
            "signal": signal,
            "product": group[config.product_col].astype(str),
        },
        index=group.index,
    ).sort_values(["signal", "product"], kind="mergesort")
    short_index = ordered.index[:short_count]
    long_index = ordered.index[-long_count:]
    weights = pd.Series(0.0, index=group.index, dtype=float)
    side = pd.Series("flat", index=group.index, dtype="string")
    if config.expression in {"long_short", "long_only"}:
        weights.loc[long_index] = config.long_gross_budget / len(long_index)
        side.loc[long_index] = "long"
    if config.expression in {"long_short", "short_only"}:
        weights.loc[short_index] = -config.short_gross_budget / len(short_index)
        side.loc[short_index] = "short"
    return weights, side


def _continuous_rank_weights(
    group: pd.DataFrame,
    ranks: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    centered = ranks - 0.5
    weights = pd.Series(0.0, index=group.index, dtype=float)
    side = pd.Series("flat", index=group.index, dtype="string")
    positive = centered.clip(lower=0.0)
    negative = -centered.clip(upper=0.0)
    if config.expression == "directional":
        denominator = float(centered.abs().sum())
        if denominator > 0:
            weights = centered / denominator * config.target_gross_exposure
            side.loc[weights.gt(0.0)] = "long"
            side.loc[weights.lt(0.0)] = "short"
        return weights, side
    if config.expression in {"long_short", "long_only", "directional"}:
        denominator = float(positive.sum())
        if denominator > 0:
            weights.loc[positive.index] += (
                positive / denominator * config.long_gross_budget
            )
            side.loc[positive.gt(0.0)] = "long"
    if config.expression in {"long_short", "short_only", "directional"}:
        denominator = float(negative.sum())
        if denominator > 0:
            weights.loc[negative.index] -= (
                negative / denominator * config.short_gross_budget
            )
            side.loc[negative.gt(0.0)] = "short"
    return weights, side


def _continuous_zscore_weights(
    group: pd.DataFrame,
    zscores: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    """Allocate each leg in proportion to cross-sectional z-score magnitude."""

    weights = pd.Series(0.0, index=group.index, dtype=float)
    side = pd.Series("flat", index=group.index, dtype="string")
    positive = zscores.clip(lower=0.0)
    negative = -zscores.clip(upper=0.0)
    if config.expression == "directional":
        denominator = float(zscores.abs().sum())
        if denominator > 0.0:
            weights = zscores / denominator * config.target_gross_exposure
            side.loc[weights.gt(0.0)] = "long"
            side.loc[weights.lt(0.0)] = "short"
        return weights, side
    if config.expression in {"long_short", "long_only"}:
        denominator = float(positive.sum())
        if denominator > 0.0:
            weights.loc[positive.index] += (
                positive / denominator * config.long_gross_budget
            )
            side.loc[positive.gt(0.0)] = "long"
    if config.expression in {"long_short", "short_only"}:
        denominator = float(negative.sum())
        if denominator > 0.0:
            weights.loc[negative.index] -= (
                negative / denominator * config.short_gross_budget
            )
            side.loc[negative.gt(0.0)] = "short"
    return weights, side


def _proportional_score_weights(
    group: pd.DataFrame,
    signal: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    """Preserve signed score magnitude and normalize absolute exposure.

    Unlike ``continuous_zscore``, this construction does not re-centre or
    re-standardize an upstream score. Contract and sector caps are applied by
    the shared cap stage afterwards, without redistributing any clipped gross.
    """

    weights = pd.Series(0.0, index=group.index, dtype=float)
    side = pd.Series("flat", index=group.index, dtype="string")
    denominator = float(signal.abs().sum())
    if denominator > 0.0 and np.isfinite(denominator):
        weights = signal / denominator * config.target_gross_exposure
        side.loc[weights.gt(0.0)] = "long"
        side.loc[weights.lt(0.0)] = "short"
    return weights, side


def _time_series_sign_weights(
    group: pd.DataFrame,
    signal: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    weights = pd.Series(0.0, index=group.index, dtype=float)
    side = pd.Series("flat", index=group.index, dtype="string")
    positive = signal.gt(0.0)
    negative = signal.lt(0.0)
    if config.expression == "directional":
        active = positive | negative
        if active.any():
            weights.loc[active] = (
                np.sign(signal.loc[active])
                * config.target_gross_exposure
                / int(active.sum())
            )
            side.loc[positive] = "long"
            side.loc[negative] = "short"
        return weights, side
    if config.expression in {"long_short", "long_only", "directional"} and positive.any():
        weights.loc[positive] = config.long_gross_budget / int(positive.sum())
        side.loc[positive] = "long"
    if config.expression in {"long_short", "short_only", "directional"} and negative.any():
        weights.loc[negative] = -config.short_gross_budget / int(negative.sum())
        side.loc[negative] = "short"
    return weights, side


def _apply_caps(
    group: pd.DataFrame,
    weights: pd.Series,
    config: SleeveConstructionConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    capped = weights.astype(float).copy()
    contract_bound = pd.Series(False, index=group.index)
    sector_bound = pd.Series(False, index=group.index)
    if config.max_weight_per_contract is not None:
        limit = float(config.max_weight_per_contract)
        contract_bound = capped.abs().gt(limit + 1e-15)
        capped = capped.clip(-limit, limit)
    if config.max_sector_gross is not None:
        side_sector_limit = float(config.max_sector_gross)
        if config.expression == "long_short":
            side_sector_limit /= 2.0
        for sign_mask in (capped.gt(0.0), capped.lt(0.0)):
            side = group.loc[sign_mask]
            if side.empty:
                continue
            gross_by_sector = capped.loc[side.index].abs().groupby(
                side[config.sector_col], sort=False
            ).sum()
            for sector, gross in gross_by_sector.items():
                if gross <= side_sector_limit + 1e-15:
                    continue
                members = side.index[side[config.sector_col].eq(sector)]
                capped.loc[members] *= side_sector_limit / gross
                sector_bound.loc[members] = True
    if config.expression == "long_short":
        long_gross = float(capped.clip(lower=0.0).sum())
        short_gross = float((-capped.clip(upper=0.0)).sum())
        common = min(long_gross, short_gross)
        if long_gross > 0:
            capped.loc[capped.gt(0.0)] *= common / long_gross
        if short_gross > 0:
            capped.loc[capped.lt(0.0)] *= common / short_gross
    return capped, contract_bound, sector_bound


def _daily_summary(
    positions: pd.DataFrame,
    config: SleeveConstructionConfig,
) -> pd.DataFrame:
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=("target_weight", lambda values: float(values.abs().sum())),
        net_exposure=("target_weight", "sum"),
        long_exposure=("target_weight", lambda values: float(values.clip(lower=0.0).sum())),
        short_exposure=("target_weight", lambda values: float((-values.clip(upper=0.0)).sum())),
        target_turnover=("target_turnover", "sum"),
        active_products=("target_weight", lambda values: int(values.abs().gt(1e-12).sum())),
        long_products=("target_weight", lambda values: int(values.gt(1e-12).sum())),
        short_products=("target_weight", lambda values: int(values.lt(-1e-12).sum())),
        signal_coverage=(config.signal_col, lambda values: float(values.notna().mean())),
        active_signal_coverage=(config.signal_col, lambda values: float(values.fillna(0.0).ne(0.0).mean())),
        contract_cap_count=("contract_cap_bound", "sum"),
        sector_cap_count=("sector_cap_bound", "sum"),
        rebalance_decision=("rebalance_decision", "max"),
    ).reset_index()
    sector_gross = (
        positions.assign(_absolute_weight=positions["target_weight"].abs())
        .groupby([config.date_col, config.sector_col], sort=True)["_absolute_weight"]
        .sum()
        .groupby(level=0)
        .max()
    )
    daily["largest_sector_gross"] = daily[config.date_col].map(sector_gross).fillna(0.0)
    daily["gross_realization"] = (
        daily["gross_exposure"] / config.target_gross_exposure
    )
    return daily


__all__ = [
    "SleeveAlignmentError",
    "SleeveConstructionResult",
    "build_sleeve_targets",
]
