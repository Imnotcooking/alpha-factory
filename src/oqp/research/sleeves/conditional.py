"""Transparent conditional-behaviour diagnostics for frozen sleeves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.sleeves.evidence import ANNUALIZATION_DAYS, SleeveEvidenceBundle
from oqp.research.sleeves.standalone import StandaloneSleeveTestBundle


CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ConditionalBehaviourConfig:
    date_col: str = "date"
    product_col: str = "ticker"
    close_col: str = "close"
    volume_col: str = "volume"
    open_interest_col: str = "open_interest"
    symbol_col: str = "symbol"
    volatility_window: int = 60
    volatility_min_periods: int = 40
    percentile_window: int = 756
    percentile_min_history: int = 126
    volume_percentile_window: int = 252
    volume_percentile_min_history: int = 60
    high_volatility_percentile: float = 0.75
    shock_window: int = 756
    shock_min_history: int = 252
    shock_quantile: float = 0.99
    minimum_cross_section: int = 10
    hac_max_lag: int = 5
    confidence_level: float = 0.95
    optimization_permitted: bool = False
    router_backtest_permitted: bool = False
    schema_version: int = CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        positive = {
            "volatility_window": self.volatility_window,
            "volatility_min_periods": self.volatility_min_periods,
            "percentile_window": self.percentile_window,
            "percentile_min_history": self.percentile_min_history,
            "volume_percentile_window": self.volume_percentile_window,
            "volume_percentile_min_history": self.volume_percentile_min_history,
            "shock_window": self.shock_window,
            "shock_min_history": self.shock_min_history,
            "minimum_cross_section": self.minimum_cross_section,
        }
        if any(int(value) < 1 for value in positive.values()):
            raise ValueError("all Phase 5 window and coverage settings must be positive")
        if self.volatility_min_periods > self.volatility_window:
            raise ValueError("volatility_min_periods cannot exceed volatility_window")
        if self.percentile_min_history > self.percentile_window:
            raise ValueError("percentile_min_history cannot exceed percentile_window")
        if self.volume_percentile_min_history > self.volume_percentile_window:
            raise ValueError(
                "volume_percentile_min_history cannot exceed volume_percentile_window"
            )
        if self.shock_min_history > self.shock_window:
            raise ValueError("shock_min_history cannot exceed shock_window")
        if not 0.5 < float(self.high_volatility_percentile) < 1.0:
            raise ValueError("high_volatility_percentile must be in (0.5, 1)")
        if not 0.5 < float(self.shock_quantile) < 1.0:
            raise ValueError("shock_quantile must be in (0.5, 1)")
        if int(self.hac_max_lag) < 0:
            raise ValueError("hac_max_lag cannot be negative")
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if self.optimization_permitted or self.router_backtest_permitted:
            raise ValueError("Phase 5 permits neither optimization nor router backtests")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class ObservableConditionsBundle:
    config: ConditionalBehaviourConfig
    product_conditions: pd.DataFrame
    market_conditions: pd.DataFrame
    definitions: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConditionalBehaviourBundle:
    config: ConditionalBehaviourConfig
    summary: dict[str, Any]
    bucket_metrics: pd.DataFrame
    condition_daily: pd.DataFrame
    definitions: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


QUARTILE_LABELS = (
    "Q1: lowest",
    "Q2: normal-low",
    "Q3: elevated",
    "Q4: highest",
)


CONDITION_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "condition_id": "market_trailing_volatility_percentile",
        "display_name": "Market trailing-volatility percentile",
        "scope": "market",
        "value_col": "market_volatility_percentile",
        "bucket_col": "market_volatility_bucket",
        "bucket_order": QUARTILE_LABELS,
        "formula": (
            "60-session volatility of the equal-weight roll-clean market return, "
            "ranked against the preceding 756 market observations"
        ),
    },
    {
        "condition_id": "contract_volatility_percentile",
        "display_name": "Each contract's own volatility percentile",
        "scope": "contract",
        "value_col": "contract_volatility_percentile",
        "bucket_col": "contract_volatility_bucket",
        "bucket_order": QUARTILE_LABELS,
        "formula": (
            "60-session roll-clean volatility for each product, ranked only "
            "against that product's preceding 756 observations"
        ),
    },
    {
        "condition_id": "high_volatility_fraction",
        "display_name": "Fraction of contracts experiencing high volatility",
        "scope": "market",
        "value_col": "high_volatility_fraction",
        "bucket_col": "high_volatility_fraction_bucket",
        "bucket_order": (
            "Contained: <10%",
            "Limited: 10-25%",
            "Broad: 25-50%",
            "Systemic: >=50%",
        ),
        "formula": (
            "Fraction of products whose own causal volatility percentile is at "
            "or above 75%"
        ),
    },
    {
        "condition_id": "cross_sectional_return_dispersion",
        "display_name": "Cross-sectional return dispersion",
        "scope": "market",
        "value_col": "dispersion_percentile",
        "bucket_col": "dispersion_bucket",
        "bucket_order": QUARTILE_LABELS,
        "formula": (
            "Cross-sectional standard deviation of current roll-clean product "
            "returns, ranked against the preceding 756 market observations"
        ),
    },
    {
        "condition_id": "directional_coherence",
        "display_name": "Directional coherence",
        "scope": "market",
        "value_col": "directional_coherence",
        "bucket_col": "directional_coherence_bucket",
        "bucket_order": (
            "Balanced: <25%",
            "Mixed: 25-50%",
            "Broad: 50-75%",
            "Near-unanimous: >=75%",
        ),
        "formula": "Absolute mean sign of current cross-sectional product returns",
    },
    {
        "condition_id": "volume_participation_percentile",
        "display_name": "Liquidity: market volume participation percentile",
        "scope": "market",
        "value_col": "median_volume_percentile",
        "bucket_col": "volume_participation_bucket",
        "bucket_order": QUARTILE_LABELS,
        "formula": (
            "Median product volume percentile, where each product is ranked "
            "against its own preceding 252 observations"
        ),
    },
    {
        "condition_id": "open_interest_participation",
        "display_name": "Open-interest participation breadth",
        "scope": "market",
        "value_col": "open_interest_increase_fraction",
        "bucket_col": "open_interest_participation_bucket",
        "bucket_order": (
            "Contracting: <40%",
            "Balanced-low: 40-50%",
            "Balanced-high: 50-60%",
            "Expanding: >=60%",
        ),
        "formula": (
            "Fraction of products with positive same-contract open-interest "
            "change; observations crossing a main-contract switch are excluded"
        ),
    },
    {
        "condition_id": "shock_age",
        "display_name": "Shock age",
        "scope": "market",
        "value_col": "shock_age",
        "bucket_col": "shock_age_bucket",
        "bucket_order": (
            "Shock day: 0",
            "Early: 1-2",
            "Digesting: 3-5",
            "Mature: 6-20",
            "Distant: >20",
        ),
        "formula": (
            "Sessions since median absolute product return exceeded the 99th "
            "percentile of its preceding 252-756 observations"
        ),
    },
)


def build_observable_conditions(
    market_panel: pd.DataFrame,
    config: ConditionalBehaviourConfig | None = None,
    *,
    source_fingerprint: str = "",
) -> ObservableConditionsBundle:
    """Build product and market conditions using only information known at date t."""

    config = config or ConditionalBehaviourConfig()
    required = {
        config.date_col,
        config.product_col,
        config.close_col,
        config.volume_col,
        config.open_interest_col,
        config.symbol_col,
    }
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError(f"condition source is missing columns: {missing}")
    product = market_panel.loc[:, sorted(required)].copy()
    product[config.date_col] = pd.to_datetime(
        product[config.date_col], errors="raise"
    ).dt.normalize()
    product[config.product_col] = product[config.product_col].astype(str)
    product[config.symbol_col] = product[config.symbol_col].astype("string")
    product = product.sort_values(
        [config.product_col, config.date_col], kind="mergesort"
    ).reset_index(drop=True)
    if product.duplicated([config.date_col, config.product_col]).any():
        raise ValueError("condition source must be unique by date and product")

    for column in (config.close_col, config.volume_col, config.open_interest_col):
        product[column] = pd.to_numeric(product[column], errors="coerce")
    grouped = product.groupby(config.product_col, sort=False, group_keys=False)
    previous_close = grouped[config.close_col].shift(1)
    product["contract_return"] = (
        product[config.close_col] / previous_close - 1.0
    ).where(previous_close.gt(0.0))
    product["contract_volatility"] = grouped["contract_return"].transform(
        lambda series: series.rolling(
            config.volatility_window,
            min_periods=config.volatility_min_periods,
        ).std(ddof=1)
        * math.sqrt(ANNUALIZATION_DAYS)
    )
    product["contract_volatility_percentile"] = grouped[
        "contract_volatility"
    ].transform(
        lambda series: _trailing_percentile(
            series,
            config.percentile_window,
            config.percentile_min_history,
        )
    )
    product["volume_percentile"] = grouped[config.volume_col].transform(
        lambda series: _trailing_percentile(
            series,
            config.volume_percentile_window,
            config.volume_percentile_min_history,
        )
    )
    previous_oi = grouped[config.open_interest_col].shift(1)
    previous_symbol = grouped[config.symbol_col].shift(1)
    same_contract = product[config.symbol_col].eq(previous_symbol)
    product["open_interest_change"] = (
        product[config.open_interest_col] / previous_oi - 1.0
    ).where(same_contract & previous_oi.gt(0.0))
    product["high_volatility"] = product[
        "contract_volatility_percentile"
    ].ge(config.high_volatility_percentile).where(
        product["contract_volatility_percentile"].notna()
    )
    product["contract_volatility_bucket"] = _quartile_bucket(
        product["contract_volatility_percentile"]
    )

    eligible = product.loc[product["contract_return"].notna()].copy()
    market = eligible.groupby(config.date_col, sort=True).agg(
        market_product_count=("contract_return", "count"),
        market_return=("contract_return", "mean"),
        cross_sectional_dispersion=("contract_return", "std"),
        directional_coherence=("contract_return", _directional_coherence),
        market_shock_score=("contract_return", lambda values: float(values.abs().median())),
        high_volatility_fraction=("high_volatility", "mean"),
        median_volume_percentile=("volume_percentile", "median"),
        open_interest_increase_fraction=(
            "open_interest_change",
            _positive_fraction,
        ),
    ).reset_index()
    insufficient = market["market_product_count"].lt(config.minimum_cross_section)
    diagnostic_columns = [
        "market_return",
        "cross_sectional_dispersion",
        "directional_coherence",
        "market_shock_score",
        "high_volatility_fraction",
        "median_volume_percentile",
        "open_interest_increase_fraction",
    ]
    market.loc[insufficient, diagnostic_columns] = np.nan
    market["market_trailing_volatility"] = market["market_return"].rolling(
        config.volatility_window,
        min_periods=config.volatility_min_periods,
    ).std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)
    market["market_volatility_percentile"] = _trailing_percentile(
        market["market_trailing_volatility"],
        config.percentile_window,
        config.percentile_min_history,
    )
    market["dispersion_percentile"] = _trailing_percentile(
        market["cross_sectional_dispersion"],
        config.percentile_window,
        config.percentile_min_history,
    )
    market["shock_threshold"] = _trailing_quantile(
        market["market_shock_score"],
        config.shock_window,
        config.shock_min_history,
        config.shock_quantile,
    )
    market["broad_market_shock"] = market["market_shock_score"].ge(
        market["shock_threshold"]
    ).where(market["shock_threshold"].notna())
    market["shock_age"] = _shock_age(market["broad_market_shock"])

    market["market_volatility_bucket"] = _quartile_bucket(
        market["market_volatility_percentile"]
    )
    market["high_volatility_fraction_bucket"] = _bucket(
        market["high_volatility_fraction"],
        (-np.inf, 0.10, 0.25, 0.50, np.inf),
        CONDITION_DEFINITIONS[2]["bucket_order"],
    )
    market["dispersion_bucket"] = _quartile_bucket(
        market["dispersion_percentile"]
    )
    market["directional_coherence_bucket"] = _bucket(
        market["directional_coherence"],
        (-np.inf, 0.25, 0.50, 0.75, np.inf),
        CONDITION_DEFINITIONS[4]["bucket_order"],
    )
    market["volume_participation_bucket"] = _quartile_bucket(
        market["median_volume_percentile"]
    )
    market["open_interest_participation_bucket"] = _bucket(
        market["open_interest_increase_fraction"],
        (-np.inf, 0.40, 0.50, 0.60, np.inf),
        CONDITION_DEFINITIONS[6]["bucket_order"],
    )
    market["shock_age_bucket"] = _bucket(
        market["shock_age"],
        (-np.inf, 0.5, 2.5, 5.5, 20.5, np.inf),
        CONDITION_DEFINITIONS[7]["bucket_order"],
    )

    product_output_columns = [
        config.date_col,
        config.product_col,
        "contract_return",
        "contract_volatility",
        "contract_volatility_percentile",
        "contract_volatility_bucket",
        "volume_percentile",
        "open_interest_change",
        "high_volatility",
    ]
    manifest = {
        "schema_version": CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint,
        "input_rows": int(len(product)),
        "input_products": int(product[config.product_col].nunique()),
        "input_start": product[config.date_col].min().date().isoformat(),
        "input_end": product[config.date_col].max().date().isoformat(),
        "causal_timing": "conditions_known_after_close_t_before_open_t_plus_1",
        "optimization_permitted": False,
        "router_backtest_permitted": False,
    }
    return ObservableConditionsBundle(
        config=config,
        product_conditions=product.loc[:, product_output_columns].copy(),
        market_conditions=market,
        definitions=CONDITION_DEFINITIONS,
        manifest=manifest,
    )


def build_conditional_behaviour(
    phase3: SleeveEvidenceBundle,
    phase4: StandaloneSleeveTestBundle,
    observables: ObservableConditionsBundle,
) -> ConditionalBehaviourBundle:
    """Describe a frozen sleeve inside observable buckets without routing it."""

    config = observables.config
    if phase3.config.factor_id != phase4.summary.get("factor_id"):
        raise ValueError("Phase 3 and Phase 4 factor identities do not match")
    if phase3.config.sleeve_id != phase4.summary.get("sleeve_id"):
        raise ValueError("Phase 3 and Phase 4 sleeve identities do not match")
    expected_phase3 = phase4.manifest.get("phase3_config_fingerprint")
    if expected_phase3 != phase3.manifest.get("config_fingerprint"):
        raise ValueError("Phase 4 does not attest the supplied Phase 3 configuration")
    if phase3.config.date_col != config.date_col:
        raise ValueError("condition date column does not match the sleeve")
    if phase3.config.product_col != config.product_col:
        raise ValueError("condition product column does not match the sleeve")

    market_daily = phase4.daily_diagnostics.copy()
    market_daily[config.date_col] = pd.to_datetime(
        market_daily[config.date_col], errors="raise"
    ).dt.normalize()
    market_daily = market_daily.merge(
        observables.market_conditions,
        on=config.date_col,
        how="left",
        validate="one_to_one",
    )
    position = phase3.positions.copy()
    position[config.date_col] = pd.to_datetime(
        position[config.date_col], errors="raise"
    ).dt.normalize()
    position = position.merge(
        observables.product_conditions,
        on=[config.date_col, config.product_col],
        how="left",
        validate="many_to_one",
    )

    metric_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    for definition in observables.definitions:
        if definition["scope"] == "market":
            daily = _market_condition_daily(market_daily, definition, config)
        else:
            daily = _contract_condition_daily(position, definition, config)
        daily_frames.append(daily)
        metric_frames.append(_condition_metrics(daily, definition, config))
    condition_daily = pd.concat(daily_frames, ignore_index=True)
    bucket_metrics = pd.concat(metric_frames, ignore_index=True)

    summary = {
        "schema_version": CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION,
        "factor_id": phase3.config.factor_id,
        "sleeve_id": phase3.config.sleeve_id,
        "market_vertical": phase3.config.market_vertical,
        "standalone_status": phase4.summary.get("standalone_status"),
        "router_eligible_before_phase5": bool(
            phase4.summary.get("router_eligible", False)
        ),
        "condition_count": len(observables.definitions),
        "bucket_metric_rows": int(len(bucket_metrics)),
        "interpretation": (
            "Exploratory conditional evidence only. Bucket differences do not "
            "constitute a router rule or a router backtest."
        ),
    }
    manifest = {
        "schema_version": CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "observable_manifest": observables.manifest,
        "phase3_config_fingerprint": phase3.manifest.get("config_fingerprint"),
        "phase4_config_fingerprint": phase4.manifest.get("config_fingerprint"),
        "phase4_criterion_fingerprint": phase4.manifest.get(
            "criterion_fingerprint"
        ),
        "factor_id": phase3.config.factor_id,
        "sleeve_id": phase3.config.sleeve_id,
        "causal_alignment_verified": True,
        "optimization_permitted": False,
        "router_backtest_permitted": False,
        "confidence_interval": {
            "method": "newey_west_hac_mean",
            "level": config.confidence_level,
            "max_lag": config.hac_max_lag,
            "multiple_comparison_adjusted": False,
        },
    }
    return ConditionalBehaviourBundle(
        config=config,
        summary=summary,
        bucket_metrics=bucket_metrics,
        condition_daily=condition_daily,
        definitions=observables.definitions,
        manifest=manifest,
    )


def write_observable_conditions_bundle(
    bundle: ObservableConditionsBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundle.product_conditions.to_parquet(
        destination / "product_conditions.parquet", index=False
    )
    bundle.market_conditions.to_parquet(
        destination / "market_conditions.parquet", index=False
    )
    _write_json(destination / "definitions.json", list(bundle.definitions))
    _write_json(destination / "manifest.json", bundle.manifest)
    return destination


def write_conditional_behaviour_bundle(
    bundle: ConditionalBehaviourBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundle.bucket_metrics.to_csv(destination / "bucket_metrics.csv", index=False)
    bundle.condition_daily.to_parquet(
        destination / "condition_daily.parquet", index=False
    )
    _write_json(destination / "definitions.json", list(bundle.definitions))
    _write_json(destination / "summary.json", bundle.summary)
    _write_json(destination / "manifest.json", bundle.manifest)
    return destination


def load_observable_conditions_bundle(
    output_dir: str | Path,
) -> ObservableConditionsBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = ConditionalBehaviourConfig(**manifest["config"])
    definitions = tuple(
        json.loads((source / "definitions.json").read_text(encoding="utf-8"))
    )
    return ObservableConditionsBundle(
        config=config,
        product_conditions=pd.read_parquet(source / "product_conditions.parquet"),
        market_conditions=pd.read_parquet(source / "market_conditions.parquet"),
        definitions=definitions,
        manifest=manifest,
    )


def load_conditional_behaviour_bundle(
    output_dir: str | Path,
) -> ConditionalBehaviourBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = ConditionalBehaviourConfig(**manifest["config"])
    definitions = tuple(
        json.loads((source / "definitions.json").read_text(encoding="utf-8"))
    )
    return ConditionalBehaviourBundle(
        config=config,
        summary=json.loads((source / "summary.json").read_text(encoding="utf-8")),
        bucket_metrics=pd.read_csv(source / "bucket_metrics.csv"),
        condition_daily=pd.read_parquet(source / "condition_daily.parquet"),
        definitions=definitions,
        manifest=manifest,
    )


def _market_condition_daily(
    market_daily: pd.DataFrame,
    definition: dict[str, Any],
    config: ConditionalBehaviourConfig,
) -> pd.DataFrame:
    bucket_col = definition["bucket_col"]
    value_col = definition["value_col"]
    required = [
        config.date_col,
        "research_split",
        bucket_col,
        value_col,
        "gross_return",
        "net_return",
        "exchange_fee_return",
        "slippage_return",
        "cost_return",
        "turnover",
        "executed_gross",
        "active_products",
    ]
    daily = market_daily.loc[:, required].dropna(subset=[bucket_col]).copy()
    daily = daily.rename(columns={bucket_col: "bucket", value_col: "condition_value"})
    daily["condition_id"] = definition["condition_id"]
    daily["condition_scope"] = "market"
    daily["position_count"] = daily["active_products"]
    daily["active_position_count"] = daily["active_products"]
    return daily


def _contract_condition_daily(
    positions: pd.DataFrame,
    definition: dict[str, Any],
    config: ConditionalBehaviourConfig,
) -> pd.DataFrame:
    bucket_col = definition["bucket_col"]
    value_col = definition["value_col"]
    eligible = positions.loc[positions[bucket_col].notna()].copy()
    eligible["absolute_executed_weight"] = eligible["executed_weight"].abs()
    eligible["active_position"] = eligible["contracts"].ne(0.0)
    daily = eligible.groupby(
        [config.date_col, "research_split", bucket_col],
        observed=True,
        sort=True,
        as_index=False,
    ).agg(
        condition_value=(value_col, "mean"),
        gross_return=("gross_contribution", "sum"),
        net_return=("net_contribution", "sum"),
        exchange_fee_return=("exchange_fee_return", "sum"),
        slippage_return=("slippage_return", "sum"),
        cost_return=("cost_return", "sum"),
        turnover=("turnover", "sum"),
        executed_gross=("absolute_executed_weight", "sum"),
        active_products=("active_position", "sum"),
        position_count=(config.product_col, "count"),
        active_position_count=("active_position", "sum"),
    )
    daily = daily.rename(columns={bucket_col: "bucket"})
    daily["condition_id"] = definition["condition_id"]
    daily["condition_scope"] = "contract"
    return daily


def _condition_metrics(
    daily: pd.DataFrame,
    definition: dict[str, Any],
    config: ConditionalBehaviourConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bucket_order = tuple(definition["bucket_order"])
    for split in ("validation", "holdout", "full"):
        split_sample = daily if split == "full" else daily.loc[
            daily["research_split"].eq(split)
        ]
        for bucket_index, bucket in enumerate(bucket_order, start=1):
            sample = split_sample.loc[
                split_sample["bucket"].astype(str).eq(str(bucket))
            ].sort_values(config.date_col)
            if sample.empty:
                continue
            rows.append(
                _conditional_metric_row(
                    sample,
                    definition,
                    split,
                    str(bucket),
                    bucket_index,
                    config,
                )
            )
    return pd.DataFrame(rows)


def _conditional_metric_row(
    sample: pd.DataFrame,
    definition: dict[str, Any],
    split: str,
    bucket: str,
    bucket_index: int,
    config: ConditionalBehaviourConfig,
) -> dict[str, Any]:
    net = pd.to_numeric(sample["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(sample["gross_return"], errors="coerce").dropna()
    active = pd.to_numeric(sample["executed_gross"], errors="coerce").gt(0.0)
    net_mean = float(net.mean()) if len(net) else math.nan
    hac_se = _newey_west_mean_se(net, config.hac_max_lag)
    critical = NormalDist().inv_cdf(0.5 + config.confidence_level / 2.0)
    annualized_net = net_mean * ANNUALIZATION_DAYS
    annualized_se = hac_se * ANNUALIZATION_DAYS
    return {
        "condition_id": definition["condition_id"],
        "condition_name": definition["display_name"],
        "condition_scope": definition["scope"],
        "research_split": split,
        "bucket": bucket,
        "bucket_order": bucket_index,
        "mean_condition_value": _mean(sample["condition_value"]),
        "date_count": int(len(sample)),
        "active_date_count": int(active.sum()),
        "position_count": int(
            pd.to_numeric(sample["position_count"], errors="coerce").sum()
        ),
        "active_position_count": int(
            pd.to_numeric(sample["active_position_count"], errors="coerce").sum()
        ),
        "gross_total_contribution": float(gross.sum()),
        "net_total_contribution": float(net.sum()),
        "gross_annualized_mean": _annualized_mean(gross),
        "net_annualized_mean": annualized_net,
        "net_annualized_volatility": _annualized_volatility(net),
        "net_sharpe": _sharpe(net),
        "annualized_turnover": _annualized_mean(sample["turnover"]),
        "annualized_exchange_fees": _annualized_mean(
            sample["exchange_fee_return"]
        ),
        "annualized_slippage": _annualized_mean(sample["slippage_return"]),
        "annualized_cost": _annualized_mean(sample["cost_return"]),
        "mean_executed_gross": _mean(sample["executed_gross"]),
        "active_net_hit_rate": float(
            sample.loc[active, "net_return"].gt(0.0).mean()
        )
        if active.any()
        else math.nan,
        "net_daily_mean_bps": net_mean * 10_000.0,
        "net_daily_mean_hac_se_bps": hac_se * 10_000.0,
        "net_annualized_mean_ci_lower": annualized_net - critical * annualized_se,
        "net_annualized_mean_ci_upper": annualized_net + critical * annualized_se,
        "net_mean_hac_t": net_mean / hac_se if hac_se > 0.0 else math.nan,
        "confidence_level": config.confidence_level,
        "hac_max_lag": config.hac_max_lag,
    }


def _trailing_percentile(
    series: pd.Series,
    window: int,
    min_history: int,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for index, current in enumerate(values):
        if not np.isfinite(current):
            continue
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) >= min_history:
            result[index] = float(np.mean(history <= current))
    return pd.Series(result, index=series.index, dtype=float)


def _trailing_quantile(
    series: pd.Series,
    window: int,
    min_history: int,
    quantile: float,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        history = values[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if len(history) >= min_history:
            result[index] = float(np.quantile(history, quantile))
    return pd.Series(result, index=series.index, dtype=float)


def _shock_age(shocks: pd.Series) -> pd.Series:
    age = np.full(len(shocks), np.nan, dtype=float)
    last_shock: int | None = None
    for index, value in enumerate(shocks.array):
        if pd.isna(value):
            continue
        if bool(value):
            last_shock = index
            age[index] = 0.0
        elif last_shock is not None:
            age[index] = float(index - last_shock)
    return pd.Series(age, index=shocks.index, dtype=float)


def _quartile_bucket(series: pd.Series) -> pd.Series:
    return _bucket(
        series,
        (-np.inf, 0.25, 0.50, 0.75, np.inf),
        QUARTILE_LABELS,
    )


def _bucket(
    series: pd.Series,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        numeric,
        bins=bins,
        labels=labels,
        right=False,
        ordered=True,
    )


def _directional_coherence(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(abs(np.sign(numeric).mean()))


def _positive_fraction(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.gt(0.0).mean()) if len(numeric) else math.nan


def _newey_west_mean_se(series: pd.Series, max_lag: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    count = len(values)
    if count < 2:
        return math.nan
    demeaned = values - values.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / count)
    usable_lag = min(int(max_lag), count - 1)
    for lag in range(1, usable_lag + 1):
        covariance = float(
            np.dot(demeaned[lag:], demeaned[:-lag]) / count
        )
        long_run_variance += 2.0 * (1.0 - lag / (usable_lag + 1.0)) * covariance
    return math.sqrt(max(long_run_variance, 0.0) / count)


def _annualized_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean() * ANNUALIZATION_DAYS) if len(numeric) else math.nan


def _annualized_volatility(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return math.nan
    return float(numeric.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS))


def _sharpe(series: pd.Series) -> float:
    annualized_volatility = _annualized_volatility(series)
    if not np.isfinite(annualized_volatility) or annualized_volatility <= 0.0:
        return math.nan
    return _annualized_mean(series) / annualized_volatility


def _mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else math.nan


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION",
    "CONDITION_DEFINITIONS",
    "ConditionalBehaviourBundle",
    "ConditionalBehaviourConfig",
    "ObservableConditionsBundle",
    "build_conditional_behaviour",
    "build_observable_conditions",
    "load_conditional_behaviour_bundle",
    "load_observable_conditions_bundle",
    "write_conditional_behaviour_bundle",
    "write_observable_conditions_bundle",
]
