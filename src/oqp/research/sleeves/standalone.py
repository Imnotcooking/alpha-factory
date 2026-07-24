"""Standalone economic validation for frozen sleeves before routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.sleeves.evidence import ANNUALIZATION_DAYS, SleeveEvidenceBundle
from oqp.research.success_criteria import (
    SuccessCriterionRegistry,
    SuccessCriterionResult,
    SuccessCriterionSpec,
    evaluate_success_criterion,
)


STANDALONE_SLEEVE_TEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StandaloneSleeveTestConfig:
    criterion_profile_id: str = "sleeve_daily_standalone_net_value_v1"
    validation_label: str = "validation"
    holdout_label: str = "holdout"
    extreme_event_quantile: float = 0.99
    extreme_event_pre_periods: int = 5
    extreme_event_post_periods: int = 5
    minimum_market_products: int = 10
    reconciliation_tolerance: float = 1e-12
    optimization_permitted: bool = False
    schema_version: int = STANDALONE_SLEEVE_TEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.criterion_profile_id).strip():
            raise ValueError("criterion_profile_id cannot be empty")
        if not 0.5 < float(self.extreme_event_quantile) < 1.0:
            raise ValueError("extreme_event_quantile must be in (0.5, 1)")
        if int(self.extreme_event_pre_periods) < 1:
            raise ValueError("extreme_event_pre_periods must be positive")
        if int(self.extreme_event_post_periods) < 1:
            raise ValueError("extreme_event_post_periods must be positive")
        if int(self.minimum_market_products) < 3:
            raise ValueError("minimum_market_products must be at least 3")
        if float(self.reconciliation_tolerance) <= 0:
            raise ValueError("reconciliation_tolerance must be positive")
        if bool(self.optimization_permitted):
            raise ValueError("Phase 4 standalone tests cannot permit optimization")
        object.__setattr__(self, "criterion_profile_id", str(self.criterion_profile_id).strip())
        object.__setattr__(self, "validation_label", str(self.validation_label).strip())
        object.__setattr__(self, "holdout_label", str(self.holdout_label).strip())
        object.__setattr__(self, "extreme_event_quantile", float(self.extreme_event_quantile))
        object.__setattr__(self, "extreme_event_pre_periods", int(self.extreme_event_pre_periods))
        object.__setattr__(self, "extreme_event_post_periods", int(self.extreme_event_post_periods))
        object.__setattr__(self, "minimum_market_products", int(self.minimum_market_products))
        object.__setattr__(self, "reconciliation_tolerance", float(self.reconciliation_tolerance))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class StandaloneSleeveTestBundle:
    config: StandaloneSleeveTestConfig
    criterion: SuccessCriterionSpec
    criterion_result: SuccessCriterionResult
    summary: dict[str, Any]
    split_metrics: pd.DataFrame
    daily_diagnostics: pd.DataFrame
    product_contribution: pd.DataFrame
    sector_contribution: pd.DataFrame
    yearly_contribution: pd.DataFrame
    extreme_events: pd.DataFrame
    extreme_event_study: pd.DataFrame
    extreme_window_summary: pd.DataFrame
    gate_evaluation: pd.DataFrame
    manifest: dict[str, Any]


def build_standalone_sleeve_test(
    phase3: SleeveEvidenceBundle,
    config: StandaloneSleeveTestConfig | None = None,
    *,
    criterion: SuccessCriterionSpec | None = None,
) -> StandaloneSleeveTestBundle:
    """Evaluate a saved sleeve on its own, without routing or optimization."""

    config = config or StandaloneSleeveTestConfig()
    criterion = criterion or SuccessCriterionRegistry.load().resolve(
        config.criterion_profile_id
    )
    if criterion.research_object != "sleeve":
        raise ValueError("standalone sleeve testing requires a sleeve criterion")
    if bool(phase3.config.optimization_permitted):
        raise ValueError("optimized Phase 3 input is not eligible for the fixed Phase 4 test")
    if not bool(phase3.manifest.get("causal_alignment_verified")):
        raise ValueError("Phase 4 requires verified causal alignment from Phase 3")

    positions = phase3.positions.copy()
    daily = phase3.daily_returns.copy()
    date_col = phase3.config.date_col
    split_col = phase3.config.split_col
    return_col = phase3.config.return_col
    required_positions = {
        date_col,
        split_col,
        phase3.config.product_col,
        phase3.config.sector_col,
        return_col,
        "contracts",
        "executed_weight",
        "gross_contribution",
        "net_contribution",
        "exchange_fee_return",
        "slippage_return",
        "cost_return",
        "turnover",
    }
    missing = sorted(required_positions.difference(positions.columns))
    if missing:
        raise ValueError(f"Phase 4 positions are missing columns: {missing}")
    required_daily = {
        date_col,
        "research_split",
        "gross_return",
        "net_return",
        "exchange_fee_return",
        "slippage_return",
        "cost_return",
        "turnover",
        "executed_gross",
        "executed_net",
        "active_products",
    }
    missing = sorted(required_daily.difference(daily.columns))
    if missing:
        raise ValueError(f"Phase 4 daily evidence is missing columns: {missing}")

    positions[date_col] = pd.to_datetime(positions[date_col], errors="raise").dt.normalize()
    daily[date_col] = pd.to_datetime(daily[date_col], errors="raise").dt.normalize()
    _reconcile_phase3(positions, daily, phase3, config)
    observed_splits = set(daily["research_split"].dropna().astype(str))
    required_splits = {config.validation_label, config.holdout_label}
    if not required_splits.issubset(observed_splits):
        raise ValueError(
            "Phase 4 requires validation and holdout splits; missing "
            + ", ".join(sorted(required_splits.difference(observed_splits)))
        )

    daily = _attach_exposure_and_market_diagnostics(
        positions, daily, phase3, config
    )
    validation_scores = daily.loc[
        daily["research_split"].eq(config.validation_label)
        & daily["market_shock_score"].notna(),
        "market_shock_score",
    ]
    if validation_scores.empty:
        raise ValueError("validation has no eligible market-shock observations")
    extreme_threshold = float(validation_scores.quantile(config.extreme_event_quantile))
    daily["extreme_market_event"] = daily["market_shock_score"].ge(extreme_threshold)

    split_metrics = _build_split_metrics(daily, positions, phase3, config)
    product_contribution = _contribution_summary(
        positions,
        phase3.config.product_col,
        split_col,
    )
    sector_contribution = _contribution_summary(
        positions,
        phase3.config.sector_col,
        split_col,
    )
    yearly_contribution = _yearly_contribution(positions, date_col, split_col)
    extreme_events = _extreme_events(daily, date_col)
    extreme_event_study, extreme_window_summary = _extreme_event_analysis(
        daily,
        date_col,
        config,
    )

    validation = split_metrics.set_index("research_split").loc[config.validation_label]
    holdout = split_metrics.set_index("research_split").loc[config.holdout_label]
    criterion_metrics = {
        "validation_net_sharpe": validation["net_sharpe"],
        "cash_validation_net_sharpe": 0.0,
        "validation_annualized_net_return": validation["net_annualized_mean"],
        "validation_break_even_cost_multiple": validation["break_even_cost_multiple"],
        "validation_trading_days": validation["active_date_count"],
    }
    criterion_result = evaluate_success_criterion(criterion, criterion_metrics)
    holdout_confirmation = bool(
        pd.notna(holdout["net_sharpe"])
        and pd.notna(holdout["net_annualized_mean"])
        and float(holdout["net_sharpe"]) > 0.0
        and float(holdout["net_annualized_mean"]) > 0.0
    )
    router_eligible = bool(criterion_result.passed and holdout_confirmation)
    if not criterion_result.passed:
        status = "blocked_validation"
    elif not holdout_confirmation:
        status = "blocked_holdout_confirmation"
    else:
        status = "eligible_for_router_research"
    gate_evaluation = _gate_table(criterion_result)

    concentration = _concentration_summary(
        product_contribution,
        yearly_contribution,
        sector_contribution,
    )
    summary = {
        "schema_version": STANDALONE_SLEEVE_TEST_SCHEMA_VERSION,
        "factor_id": phase3.config.factor_id,
        "sleeve_id": phase3.config.sleeve_id,
        "validation_decision": criterion_result.decision.value,
        "holdout_confirmation_passed": holdout_confirmation,
        "router_eligible": router_eligible,
        "standalone_status": status,
        "validation": _json_safe(validation.to_dict()),
        "holdout": _json_safe(holdout.to_dict()),
        "success_criterion": _json_safe(criterion_result.to_dict()),
        "concentration": _json_safe(concentration),
        "extreme_event": {
            "definition": (
                "Cross-sectional median absolute product return at or above the "
                "validation-frozen quantile threshold."
            ),
            "quantile": config.extreme_event_quantile,
            "validation_threshold": extreme_threshold,
            "full_event_count": int(daily["extreme_market_event"].sum()),
            "validation_event_count": int(
                daily.loc[daily["research_split"].eq(config.validation_label), "extreme_market_event"].sum()
            ),
            "holdout_event_count": int(
                daily.loc[daily["research_split"].eq(config.holdout_label), "extreme_market_event"].sum()
            ),
        },
        "interpretation_boundary": (
            "Concentration and extreme-event diagnostics are descriptive in this "
            "version because no numerical gates for them were predeclared."
        ),
    }
    manifest = {
        "schema_version": STANDALONE_SLEEVE_TEST_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "criterion": criterion.to_dict(),
        "criterion_fingerprint": criterion.fingerprint,
        "criterion_result": criterion_result.to_dict(),
        "factor_id": phase3.config.factor_id,
        "sleeve_id": phase3.config.sleeve_id,
        "market_vertical": phase3.config.market_vertical,
        "phase3_config_fingerprint": phase3.manifest.get("config_fingerprint"),
        "input_data_fingerprint": phase3.manifest.get("input_data_fingerprint"),
        "factor_definition_fingerprint": phase3.manifest.get(
            "factor_definition_fingerprint"
        ),
        "factor_implementation_fingerprint": phase3.manifest.get(
            "factor_implementation_fingerprint"
        ),
        "sleeve_implementation_fingerprint": phase3.manifest.get(
            "sleeve_implementation_fingerprint"
        ),
        "sleeve_definition_fingerprint": phase3.manifest.get(
            "sleeve_definition_fingerprint"
        ),
        "predictive_evidence_config_fingerprint": phase3.manifest.get(
            "predictive_evidence_config_fingerprint"
        ),
        "causal_alignment_verified": True,
        "optimization_permitted": False,
        "extreme_event_protocol": {
            "score": "cross_sectional_median_absolute_product_forward_return",
            "threshold_sample": config.validation_label,
            "quantile": config.extreme_event_quantile,
            "frozen_threshold": extreme_threshold,
            "minimum_market_products": config.minimum_market_products,
            "use": "diagnostic_only_not_router_input",
        },
        "reconciliation": {
            "phase3_daily_returns_rebuilt_from_positions": True,
            "tolerance": config.reconciliation_tolerance,
        },
    }
    return StandaloneSleeveTestBundle(
        config=config,
        criterion=criterion,
        criterion_result=criterion_result,
        summary=summary,
        split_metrics=split_metrics,
        daily_diagnostics=daily,
        product_contribution=product_contribution,
        sector_contribution=sector_contribution,
        yearly_contribution=yearly_contribution,
        extreme_events=extreme_events,
        extreme_event_study=extreme_event_study,
        extreme_window_summary=extreme_window_summary,
        gate_evaluation=gate_evaluation,
        manifest=manifest,
    )


def write_standalone_sleeve_test_bundle(
    bundle: StandaloneSleeveTestBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(_json_safe(bundle.summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(_json_safe(bundle.manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    bundle.split_metrics.to_csv(destination / "split_metrics.csv", index=False)
    bundle.daily_diagnostics.to_parquet(destination / "daily_diagnostics.parquet", index=False)
    bundle.product_contribution.to_csv(destination / "product_contribution.csv", index=False)
    bundle.sector_contribution.to_csv(destination / "sector_contribution.csv", index=False)
    bundle.yearly_contribution.to_csv(destination / "yearly_contribution.csv", index=False)
    bundle.extreme_events.to_csv(destination / "extreme_events.csv", index=False)
    bundle.extreme_event_study.to_csv(destination / "extreme_event_study.csv", index=False)
    bundle.extreme_window_summary.to_csv(destination / "extreme_window_summary.csv", index=False)
    bundle.gate_evaluation.to_csv(destination / "gate_evaluation.csv", index=False)
    return destination


def load_standalone_sleeve_test_bundle(
    output_dir: str | Path,
) -> StandaloneSleeveTestBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config = StandaloneSleeveTestConfig(**manifest["config"])
    criterion = SuccessCriterionSpec.from_mapping(
        manifest["criterion"]["profile_id"], manifest["criterion"]
    )
    criterion_result = evaluate_success_criterion(
        criterion,
        _criterion_metrics_from_summary(summary),
    )
    return StandaloneSleeveTestBundle(
        config=config,
        criterion=criterion,
        criterion_result=criterion_result,
        summary=summary,
        split_metrics=pd.read_csv(source / "split_metrics.csv"),
        daily_diagnostics=pd.read_parquet(source / "daily_diagnostics.parquet"),
        product_contribution=pd.read_csv(source / "product_contribution.csv"),
        sector_contribution=pd.read_csv(source / "sector_contribution.csv"),
        yearly_contribution=pd.read_csv(source / "yearly_contribution.csv"),
        extreme_events=pd.read_csv(source / "extreme_events.csv"),
        extreme_event_study=pd.read_csv(source / "extreme_event_study.csv"),
        extreme_window_summary=pd.read_csv(source / "extreme_window_summary.csv"),
        gate_evaluation=pd.read_csv(source / "gate_evaluation.csv"),
        manifest=manifest,
    )


def _reconcile_phase3(
    positions: pd.DataFrame,
    daily: pd.DataFrame,
    phase3: SleeveEvidenceBundle,
    config: StandaloneSleeveTestConfig,
) -> None:
    date_col = phase3.config.date_col
    rebuilt = (
        positions.groupby(date_col, as_index=False, sort=True)
        .agg(
            gross_return=("gross_contribution", "sum"),
            cost_return=("cost_return", "sum"),
            net_return=("net_contribution", "sum"),
            turnover=("turnover", "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    observed = daily[[date_col, "gross_return", "cost_return", "net_return", "turnover"]].sort_values(date_col).reset_index(drop=True)
    if not rebuilt[date_col].equals(observed[date_col]):
        raise ValueError("Phase 3 position dates do not reconcile with daily evidence")
    for column in ("gross_return", "cost_return", "net_return", "turnover"):
        difference = (rebuilt[column] - observed[column]).abs().max()
        if pd.isna(difference) or float(difference) > config.reconciliation_tolerance:
            raise ValueError(f"Phase 3 {column} does not reconcile from positions")


def _attach_exposure_and_market_diagnostics(
    positions: pd.DataFrame,
    daily: pd.DataFrame,
    phase3: SleeveEvidenceBundle,
    config: StandaloneSleeveTestConfig,
) -> pd.DataFrame:
    date_col = phase3.config.date_col
    sector_col = phase3.config.sector_col
    return_col = phase3.config.return_col
    work = positions.copy()
    work["absolute_executed_weight"] = work["executed_weight"].abs()
    gross = work.groupby(date_col)["absolute_executed_weight"].transform("sum")
    work["position_gross_share"] = np.where(
        gross.gt(0.0), work["absolute_executed_weight"] / gross, 0.0
    )
    position = work.groupby(date_col, sort=True).agg(
        position_hhi=("position_gross_share", lambda x: float(np.square(x).sum())),
        largest_position_share=("position_gross_share", "max"),
        long_gross=("executed_weight", lambda x: float(x.clip(lower=0.0).sum())),
        short_gross=("executed_weight", lambda x: float((-x.clip(upper=0.0)).sum())),
        position_gross_hit_rate=("gross_contribution", lambda x: _active_hit_rate(x, work.loc[x.index, "contracts"])),
        position_net_hit_rate=("net_contribution", lambda x: _active_hit_rate(x, work.loc[x.index, "contracts"])),
    )
    position["effective_products"] = np.where(
        position["position_hhi"].gt(0.0), 1.0 / position["position_hhi"], 0.0
    )

    sector = (
        work.groupby([date_col, sector_col], sort=True)["absolute_executed_weight"]
        .sum()
        .rename("sector_gross")
        .reset_index()
    )
    sector_total = sector.groupby(date_col)["sector_gross"].transform("sum")
    sector["sector_share"] = np.where(
        sector_total.gt(0.0), sector["sector_gross"] / sector_total, 0.0
    )
    sector_daily = sector.groupby(date_col, sort=True).agg(
        sector_hhi=("sector_share", lambda x: float(np.square(x).sum())),
        largest_sector_share=("sector_share", "max"),
        active_sectors=("sector_gross", lambda x: int(x.gt(0.0).sum())),
    )
    sector_daily["effective_sectors"] = np.where(
        sector_daily["sector_hhi"].gt(0.0),
        1.0 / sector_daily["sector_hhi"],
        0.0,
    )

    market = work.groupby(date_col, sort=True).agg(
        market_product_count=(return_col, "count"),
        market_shock_score=(return_col, lambda x: float(pd.to_numeric(x, errors="coerce").abs().median())),
        market_equal_weight_return=(return_col, "mean"),
        market_directional_coherence=(return_col, _directional_coherence),
    )
    market.loc[
        market["market_product_count"].lt(config.minimum_market_products),
        "market_shock_score",
    ] = np.nan

    out = daily.merge(position.reset_index(), on=date_col, how="left", validate="one_to_one")
    out = out.merge(sector_daily.reset_index(), on=date_col, how="left", validate="one_to_one")
    out = out.merge(market.reset_index(), on=date_col, how="left", validate="one_to_one")
    out["active_net_hit"] = out["net_return"].gt(0.0).where(out["active_products"].gt(0))
    out["active_gross_hit"] = out["gross_return"].gt(0.0).where(out["active_products"].gt(0))
    return out


def _build_split_metrics(
    daily: pd.DataFrame,
    positions: pd.DataFrame,
    phase3: SleeveEvidenceBundle,
    config: StandaloneSleeveTestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = ["full", config.validation_label, config.holdout_label]
    for label in labels:
        sample = daily if label == "full" else daily.loc[daily["research_split"].eq(label)]
        dates = set(sample[phase3.config.date_col])
        position_sample = positions.loc[positions[phase3.config.date_col].isin(dates)]
        row = _performance_metrics(sample, label)
        row.update(
            {
                "position_gross_hit_rate": _active_hit_rate(
                    position_sample["gross_contribution"], position_sample["contracts"]
                ),
                "position_net_hit_rate": _active_hit_rate(
                    position_sample["net_contribution"], position_sample["contracts"]
                ),
                "mean_position_hhi": _mean(sample.loc[sample["active_products"].gt(0), "position_hhi"]),
                "mean_effective_products": _mean(sample.loc[sample["active_products"].gt(0), "effective_products"]),
                "mean_largest_position_share": _mean(sample.loc[sample["active_products"].gt(0), "largest_position_share"]),
                "mean_sector_hhi": _mean(sample.loc[sample["active_products"].gt(0), "sector_hhi"]),
                "mean_effective_sectors": _mean(sample.loc[sample["active_products"].gt(0), "effective_sectors"]),
                "mean_largest_sector_share": _mean(sample.loc[sample["active_products"].gt(0), "largest_sector_share"]),
                "mean_long_gross": _mean(sample["long_gross"]),
                "mean_short_gross": _mean(sample["short_gross"]),
                "extreme_event_count": int(sample["extreme_market_event"].sum()),
            }
        )
        non_extreme = sample.loc[~sample["extreme_market_event"]]
        row["net_annualized_mean_ex_extremes"] = _annualized_mean(non_extreme["net_return"])
        row["net_sharpe_ex_extremes"] = _sharpe(non_extreme["net_return"])
        event_net = float(sample.loc[sample["extreme_market_event"], "net_return"].sum())
        total_abs_net = float(sample["net_return"].abs().sum())
        row["extreme_net_contribution_share"] = (
            event_net / total_abs_net if total_abs_net > 0 else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_metrics(sample: pd.DataFrame, label: str) -> dict[str, Any]:
    net = pd.to_numeric(sample["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(sample["gross_return"], errors="coerce").dropna()
    wealth = (1.0 + net).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    annual_cost = _annualized_mean(sample["cost_return"])
    annual_gross = _annualized_mean(gross)
    active = sample["active_products"].gt(0)
    return {
        "research_split": label,
        "date_count": int(len(sample)),
        "active_date_count": int(active.sum()),
        "gross_total_return": _compound(gross),
        "net_total_return": _compound(net),
        "gross_annualized_mean": annual_gross,
        "net_annualized_mean": _annualized_mean(net),
        "gross_annualized_volatility": _annualized_volatility(gross),
        "net_annualized_volatility": _annualized_volatility(net),
        "net_sharpe": _sharpe(net),
        "maximum_drawdown": float(drawdown.min()) if len(drawdown) else math.nan,
        "annualized_turnover": _annualized_mean(sample["turnover"]),
        "annualized_exchange_fees": _annualized_mean(sample["exchange_fee_return"]),
        "annualized_slippage": _annualized_mean(sample["slippage_return"]),
        "annualized_cost": annual_cost,
        "break_even_cost_multiple": annual_gross / annual_cost if annual_cost > 0 else math.nan,
        "active_gross_hit_rate": float(sample.loc[active, "gross_return"].gt(0.0).mean()) if active.any() else math.nan,
        "active_net_hit_rate": float(sample.loc[active, "net_return"].gt(0.0).mean()) if active.any() else math.nan,
        "mean_executed_gross": _mean(sample["executed_gross"]),
        "mean_executed_net": _mean(sample["executed_net"]),
        "mean_active_products": _mean(sample["active_products"]),
    }


def _contribution_summary(
    positions: pd.DataFrame,
    member_col: str,
    split_col: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, sample in [("full", positions), *positions.groupby(split_col, sort=False)]:
        grouped = sample.groupby(member_col, as_index=False, sort=True).agg(
            gross_contribution=("gross_contribution", "sum"),
            exchange_fee_return=("exchange_fee_return", "sum"),
            slippage_return=("slippage_return", "sum"),
            cost_return=("cost_return", "sum"),
            net_contribution=("net_contribution", "sum"),
            turnover=("turnover", "sum"),
            mean_absolute_weight=("executed_weight", lambda x: float(x.abs().mean())),
            active_positions=("contracts", lambda x: int(x.ne(0.0).sum())),
            gross_hit_rate=("gross_contribution", lambda x: float(x[sample.loc[x.index, "contracts"].ne(0.0)].gt(0.0).mean()) if sample.loc[x.index, "contracts"].ne(0.0).any() else math.nan),
            net_hit_rate=("net_contribution", lambda x: float(x[sample.loc[x.index, "contracts"].ne(0.0)].gt(0.0).mean()) if sample.loc[x.index, "contracts"].ne(0.0).any() else math.nan),
        )
        grouped.insert(0, "research_split", str(label))
        total = float(grouped["net_contribution"].abs().sum())
        grouped["absolute_net_contribution_share"] = np.where(
            total > 0.0, grouped["net_contribution"].abs() / total, np.nan
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _yearly_contribution(
    positions: pd.DataFrame,
    date_col: str,
    split_col: str,
) -> pd.DataFrame:
    work = positions.copy()
    work["year"] = work[date_col].dt.year
    grouped = work.groupby(["year", split_col], as_index=False, sort=True).agg(
        gross_contribution=("gross_contribution", "sum"),
        exchange_fee_return=("exchange_fee_return", "sum"),
        slippage_return=("slippage_return", "sum"),
        cost_return=("cost_return", "sum"),
        net_contribution=("net_contribution", "sum"),
        turnover=("turnover", "sum"),
        active_positions=("contracts", lambda x: int(x.ne(0.0).sum())),
    )
    total = float(grouped["net_contribution"].abs().sum())
    grouped["absolute_net_contribution_share"] = np.where(
        total > 0.0, grouped["net_contribution"].abs() / total, np.nan
    )
    return grouped


def _extreme_events(daily: pd.DataFrame, date_col: str) -> pd.DataFrame:
    columns = [
        date_col,
        "research_split",
        "market_shock_score",
        "market_product_count",
        "market_equal_weight_return",
        "market_directional_coherence",
        "gross_return",
        "net_return",
        "cost_return",
        "turnover",
        "executed_gross",
    ]
    return daily.loc[daily["extreme_market_event"], columns].reset_index(drop=True)


def _extreme_event_analysis(
    daily: pd.DataFrame,
    date_col: str,
    config: StandaloneSleeveTestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = daily.sort_values(date_col).reset_index(drop=True)
    event_indices = ordered.index[ordered["extreme_market_event"]].tolist()
    window_rows: list[dict[str, Any]] = []
    pre_indices: set[int] = set()
    post_indices: set[int] = set()
    for event_index in event_indices:
        event_date = ordered.loc[event_index, date_col]
        for relative_day in range(
            -config.extreme_event_pre_periods,
            config.extreme_event_post_periods + 1,
        ):
            index = event_index + relative_day
            if index < 0 or index >= len(ordered):
                continue
            row = ordered.loc[index]
            window_rows.append(
                {
                    "event_date": event_date,
                    "relative_day": relative_day,
                    "observation_date": row[date_col],
                    "event_split": ordered.loc[event_index, "research_split"],
                    "observation_split": row["research_split"],
                    "gross_return": row["gross_return"],
                    "net_return": row["net_return"],
                    "cost_return": row["cost_return"],
                    "turnover": row["turnover"],
                    "executed_gross": row["executed_gross"],
                    "active_products": row["active_products"],
                }
            )
            if relative_day < 0:
                pre_indices.add(index)
            elif relative_day > 0:
                post_indices.add(index)
    windows = pd.DataFrame(window_rows)
    if windows.empty:
        study = pd.DataFrame(
            columns=["relative_day", "observations", "mean_gross_return", "mean_net_return", "median_net_return", "mean_cost_return", "net_hit_rate", "mean_turnover", "mean_executed_gross", "cumulative_mean_net_return"]
        )
    else:
        study = windows.groupby("relative_day", as_index=False, sort=True).agg(
            observations=("net_return", "count"),
            mean_gross_return=("gross_return", "mean"),
            mean_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            mean_cost_return=("cost_return", "mean"),
            net_hit_rate=("net_return", lambda x: float(x.gt(0.0).mean())),
            mean_turnover=("turnover", "mean"),
            mean_executed_gross=("executed_gross", "mean"),
        )
        study["cumulative_mean_net_return"] = (
            1.0 + study["mean_net_return"]
        ).cumprod() - 1.0

    event_set = set(event_indices)
    categories = {
        "pre_event": sorted(pre_indices.difference(event_set)),
        "event": sorted(event_set),
        "post_event": sorted(post_indices.difference(event_set)),
        "non_event": sorted(set(ordered.index).difference(event_set)),
    }
    summary_rows: list[dict[str, Any]] = []
    for split in ("full", config.validation_label, config.holdout_label):
        split_mask = pd.Series(True, index=ordered.index) if split == "full" else ordered["research_split"].eq(split)
        for window, indices in categories.items():
            index = [item for item in indices if bool(split_mask.loc[item])]
            sample = ordered.loc[index]
            metrics = _performance_metrics(sample, split)
            summary_rows.append(
                {
                    "research_split": split,
                    "window": window,
                    "date_count": metrics["date_count"],
                    "net_annualized_mean": metrics["net_annualized_mean"],
                    "net_annualized_volatility": metrics["net_annualized_volatility"],
                    "net_sharpe": metrics["net_sharpe"],
                    "net_total_return": metrics["net_total_return"],
                    "active_net_hit_rate": metrics["active_net_hit_rate"],
                    "annualized_cost": metrics["annualized_cost"],
                    "annualized_turnover": metrics["annualized_turnover"],
                }
            )
    return study, pd.DataFrame(summary_rows)


def _concentration_summary(
    products: pd.DataFrame,
    years: pd.DataFrame,
    sectors: pd.DataFrame,
) -> dict[str, Any]:
    product_full = products.loc[products["research_split"].eq("full")]
    sector_full = sectors.loc[sectors["research_split"].eq("full")]
    return {
        "top_product_absolute_net_contribution_share": _top_share(product_full, 1),
        "top_five_product_absolute_net_contribution_share": _top_share(product_full, 5),
        "product_net_contribution_hhi": _share_hhi(product_full),
        "top_sector_absolute_net_contribution_share": _top_share(sector_full, 1),
        "sector_net_contribution_hhi": _share_hhi(sector_full),
        "top_year_absolute_net_contribution_share": _top_share(years, 1),
        "year_net_contribution_hhi": _share_hhi(years),
        "positive_product_fraction": float(product_full["net_contribution"].gt(0.0).mean()) if len(product_full) else math.nan,
        "positive_year_fraction": float(years["net_contribution"].gt(0.0).mean()) if len(years) else math.nan,
    }


def _gate_table(result: SuccessCriterionResult) -> pd.DataFrame:
    rows = [
        {
            "gate": "primary_absolute_floor",
            "metric": "validation_net_sharpe",
            "value": result.primary_value,
            "operator": ">=",
            "threshold": 0.0,
            "passed": result.primary_floor_passed,
        },
        {
            "gate": "cash_comparator",
            "metric": "validation_net_sharpe_minus_cash",
            "value": result.improvement,
            "operator": ">=",
            "threshold": 0.0,
            "passed": result.comparator_passed,
        },
    ]
    rows.extend(
        {
            "gate": gate.name,
            "metric": gate.metric,
            "value": gate.value,
            "operator": gate.operator,
            "threshold": gate.threshold,
            "passed": gate.passed,
        }
        for gate in result.gates
    )
    return pd.DataFrame(rows)


def _criterion_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    validation = summary["validation"]
    return {
        "validation_net_sharpe": validation["net_sharpe"],
        "cash_validation_net_sharpe": 0.0,
        "validation_annualized_net_return": validation["net_annualized_mean"],
        "validation_break_even_cost_multiple": validation["break_even_cost_multiple"],
        "validation_trading_days": validation["active_date_count"],
    }


def _active_hit_rate(contribution: pd.Series, contracts: pd.Series) -> float:
    active = pd.to_numeric(contracts, errors="coerce").fillna(0.0).ne(0.0)
    values = pd.to_numeric(contribution, errors="coerce").loc[active].dropna()
    return float(values.gt(0.0).mean()) if len(values) else math.nan


def _directional_coherence(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.gt(0.0).mean() - clean.lt(0.0).mean()) if len(clean) else math.nan


def _top_share(frame: pd.DataFrame, count: int) -> float:
    values = pd.to_numeric(
        frame.get("absolute_net_contribution_share"), errors="coerce"
    ).dropna()
    return float(values.nlargest(count).sum()) if len(values) else math.nan


def _share_hhi(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(
        frame.get("absolute_net_contribution_share"), errors="coerce"
    ).dropna()
    return float(np.square(values).sum()) if len(values) else math.nan


def _compound(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float((1.0 + clean).prod() - 1.0) if len(clean) else math.nan


def _annualized_mean(values: pd.Series) -> float:
    return _mean(values) * ANNUALIZATION_DAYS


def _annualized_volatility(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)) if len(clean) > 1 else math.nan


def _sharpe(values: pd.Series) -> float:
    volatility = _annualized_volatility(values)
    return _annualized_mean(values) / volatility if pd.notna(volatility) and volatility > 1e-15 else math.nan


def _mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


__all__ = [
    "STANDALONE_SLEEVE_TEST_SCHEMA_VERSION",
    "StandaloneSleeveTestBundle",
    "StandaloneSleeveTestConfig",
    "build_standalone_sleeve_test",
    "load_standalone_sleeve_test_bundle",
    "write_standalone_sleeve_test_bundle",
]
