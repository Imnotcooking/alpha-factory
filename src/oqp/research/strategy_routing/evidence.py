"""Causal evidence for one frozen two-sleeve router hypothesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oqp.research.sleeves.evidence import (
    ANNUALIZATION_DAYS,
    SleeveEvidenceBundle,
    execute_intraday_session_targets,
    summarize_executed_positions,
)
from oqp.research.sleeves.standalone import StandaloneSleeveTestBundle
from oqp.research.success_criteria import (
    SuccessCriterionRegistry,
    SuccessCriterionResult,
    SuccessCriterionSpec,
    evaluate_success_criterion,
)


ROUTER_HYPOTHESIS_SCHEMA_VERSION = 1
VALID_SCORE_ORIENTATIONS = {"higher_favors_a", "higher_favors_b"}
ATTAINABLE_STRATEGIES = (
    "sleeve_a",
    "sleeve_b",
    "static_blend",
    "exposure_scaled_blend",
    "router",
)
ORACLE_STRATEGY = "oracle_upper_bound"


@dataclass(frozen=True, slots=True)
class RouterHypothesisConfig:
    """Predeclared economic and statistical contract for one router test."""

    hypothesis_id: str
    router_id: str
    economic_claim: str
    economic_mechanism: str
    market_vertical: str
    sleeve_a_factor_id: str
    sleeve_a_id: str
    sleeve_b_factor_id: str
    sleeve_b_id: str
    score_name: str
    score_source_fingerprint: str
    hypothesis_frozen_on: str
    score_col: str = "router_score"
    score_orientation: str = "higher_favors_a"
    threshold: float = 0.0
    date_col: str = "date"
    validation_label: str = "validation"
    holdout_label: str = "holdout"
    static_blend_weight_a: float = 0.50
    exposure_target_volatility: float = 0.10
    exposure_lookback_periods: int = 60
    exposure_min_periods: int = 40
    exposure_min_scale: float = 0.0
    exposure_max_scale: float = 1.0
    monthly_hac_max_lag: int = 3
    confidence_level: float = 0.95
    criterion_profile_id: str = "router_incremental_net_value_v1"
    optimization_permitted: bool = False
    oracle_is_upper_bound_only: bool = True
    schema_version: int = ROUTER_HYPOTHESIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required_text = (
            "hypothesis_id",
            "router_id",
            "economic_claim",
            "economic_mechanism",
            "market_vertical",
            "sleeve_a_factor_id",
            "sleeve_a_id",
            "sleeve_b_factor_id",
            "sleeve_b_id",
            "score_name",
            "score_source_fingerprint",
            "hypothesis_frozen_on",
            "score_col",
            "date_col",
        )
        for field in required_text:
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)
        if (
            self.sleeve_a_factor_id,
            self.sleeve_a_id,
        ) == (
            self.sleeve_b_factor_id,
            self.sleeve_b_id,
        ):
            raise ValueError("router sleeves A and B must be distinct")
        orientation = str(self.score_orientation).strip().lower()
        if orientation not in VALID_SCORE_ORIENTATIONS:
            raise ValueError(f"unknown score orientation: {orientation}")
        if not 0.0 <= float(self.static_blend_weight_a) <= 1.0:
            raise ValueError("static_blend_weight_a must be in [0, 1]")
        if float(self.exposure_target_volatility) <= 0.0:
            raise ValueError("exposure_target_volatility must be positive")
        if int(self.exposure_lookback_periods) < 2:
            raise ValueError("exposure_lookback_periods must be at least 2")
        if not 2 <= int(self.exposure_min_periods) <= int(
            self.exposure_lookback_periods
        ):
            raise ValueError("exposure_min_periods must be within the lookback")
        if not 0.0 <= float(self.exposure_min_scale) <= float(
            self.exposure_max_scale
        ):
            raise ValueError("exposure scale bounds are invalid")
        if int(self.monthly_hac_max_lag) < 0:
            raise ValueError("monthly_hac_max_lag cannot be negative")
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        try:
            frozen = pd.Timestamp(self.hypothesis_frozen_on).normalize()
        except Exception as exc:
            raise ValueError("hypothesis_frozen_on must be a valid date") from exc
        if self.optimization_permitted:
            raise ValueError("Phase 6 does not permit router optimization")
        if not self.oracle_is_upper_bound_only:
            raise ValueError("the oracle may only be an unattainable upper bound")
        object.__setattr__(self, "score_orientation", orientation)
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self, "static_blend_weight_a", float(self.static_blend_weight_a)
        )
        object.__setattr__(
            self,
            "exposure_target_volatility",
            float(self.exposure_target_volatility),
        )
        object.__setattr__(
            self, "exposure_lookback_periods", int(self.exposure_lookback_periods)
        )
        object.__setattr__(
            self, "exposure_min_periods", int(self.exposure_min_periods)
        )
        object.__setattr__(self, "exposure_min_scale", float(self.exposure_min_scale))
        object.__setattr__(self, "exposure_max_scale", float(self.exposure_max_scale))
        object.__setattr__(self, "monthly_hac_max_lag", int(self.monthly_hac_max_lag))
        object.__setattr__(self, "confidence_level", float(self.confidence_level))
        object.__setattr__(self, "hypothesis_frozen_on", frozen.date().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class RouterHypothesisEvidenceBundle:
    config: RouterHypothesisConfig
    criterion: SuccessCriterionSpec
    criterion_result: SuccessCriterionResult
    summary: dict[str, Any]
    decision_log: pd.DataFrame
    relative_advantage: pd.DataFrame
    execution_positions: pd.DataFrame
    daily_comparison: pd.DataFrame
    strategy_metrics: pd.DataFrame
    routing_metrics: pd.DataFrame
    monthly_paired_tests: pd.DataFrame
    subperiod_stability: pd.DataFrame
    manifest: dict[str, Any]


def build_router_hypothesis_evidence(
    phase3_a: SleeveEvidenceBundle,
    phase4_a: StandaloneSleeveTestBundle,
    phase3_b: SleeveEvidenceBundle,
    phase4_b: StandaloneSleeveTestBundle,
    router_scores: pd.DataFrame,
    config: RouterHypothesisConfig,
    *,
    criterion: SuccessCriterionSpec | None = None,
) -> RouterHypothesisEvidenceBundle:
    """Test a frozen score against the next-period relative sleeve advantage."""

    _validate_inputs(phase3_a, phase4_a, phase3_b, phase4_b, config)
    criterion = criterion or SuccessCriterionRegistry.load().resolve(
        config.criterion_profile_id
    )
    if criterion.research_object != "router":
        raise ValueError("Phase 6 requires a router success criterion")

    score = _prepare_scores(router_scores, config)
    base = _align_sleeve_positions(phase3_a, phase3_b)
    available_dates = pd.Index(base[config.date_col].drop_duplicates())
    score = score.loc[score[config.date_col].isin(available_dates)].copy()
    if score.empty:
        raise ValueError("router score has no dates in common with the sleeves")
    holdout_start = _holdout_start(base, config)
    if pd.Timestamp(config.hypothesis_frozen_on) >= holdout_start:
        raise ValueError(
            "the router hypothesis must be frozen before the untouched holdout begins"
        )

    capital = float(phase3_a.manifest["execution"]["capital"])
    slippage = float(
        phase3_a.manifest["execution"]["slippage_ticks_per_side"]
    )
    daily_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []

    sleeve_a = _execute_combination(
        base,
        score[[config.date_col]],
        phase3_a,
        strategy_id="sleeve_a",
        weight_a=1.0,
        weight_b=0.0,
        capital=capital,
        slippage=slippage,
    )
    sleeve_b = _execute_combination(
        base,
        score[[config.date_col]],
        phase3_a,
        strategy_id="sleeve_b",
        weight_a=0.0,
        weight_b=1.0,
        capital=capital,
        slippage=slippage,
    )
    static_weight = config.static_blend_weight_a
    static_blend = _execute_combination(
        base,
        score[[config.date_col]],
        phase3_a,
        strategy_id="static_blend",
        weight_a=static_weight,
        weight_b=1.0 - static_weight,
        capital=capital,
        slippage=slippage,
    )
    for positions, daily in (sleeve_a, sleeve_b, static_blend):
        position_frames.append(positions)
        daily_frames.append(daily)

    static_daily = static_blend[1]
    exposure_schedule = _causal_exposure_schedule(static_daily, config)
    exposure_scaled = _execute_combination(
        base,
        score[[config.date_col]].merge(
            exposure_schedule, on=config.date_col, how="left", validate="one_to_one"
        ),
        phase3_a,
        strategy_id="exposure_scaled_blend",
        weight_a=static_weight,
        weight_b=1.0 - static_weight,
        capital=capital,
        slippage=slippage,
        scale_col="exposure_scale",
    )
    position_frames.append(exposure_scaled[0])
    daily_frames.append(exposure_scaled[1])

    decision_log = _build_decision_log(score, base, config)
    router_schedule = decision_log[
        [config.date_col, "allocation_a", "allocation_b"]
    ]
    routed = _execute_combination(
        base,
        router_schedule,
        phase3_a,
        strategy_id="router",
        weight_a="allocation_a",
        weight_b="allocation_b",
        capital=capital,
        slippage=slippage,
    )
    position_frames.append(routed[0])
    daily_frames.append(routed[1])

    attainable_daily = pd.concat(daily_frames, ignore_index=True)
    relative = _relative_advantage(sleeve_a[1], sleeve_b[1], decision_log, config)
    oracle_schedule = relative[[config.date_col, "better_sleeve"]].copy()
    oracle_schedule["allocation_a"] = oracle_schedule["better_sleeve"].eq(
        "sleeve_a"
    ).astype(float)
    oracle_schedule["allocation_b"] = 1.0 - oracle_schedule["allocation_a"]
    oracle = _execute_combination(
        base,
        oracle_schedule[[config.date_col, "allocation_a", "allocation_b"]],
        phase3_a,
        strategy_id=ORACLE_STRATEGY,
        weight_a="allocation_a",
        weight_b="allocation_b",
        capital=capital,
        slippage=slippage,
    )
    position_frames.append(oracle[0])
    daily_comparison = pd.concat(
        [attainable_daily, oracle[1]], ignore_index=True
    ).sort_values(["strategy_id", config.date_col])
    execution_positions = pd.concat(position_frames, ignore_index=True).sort_values(
        ["strategy_id", config.date_col, phase3_a.config.product_col]
    )

    strategy_metrics = _strategy_metrics(daily_comparison, config)
    routing_metrics = _routing_metrics(relative, daily_comparison, config)
    monthly_tests = _monthly_paired_tests(daily_comparison, config)
    best_alternative = _best_validation_alternative(strategy_metrics, config)
    stability = _subperiod_stability(
        relative, daily_comparison, best_alternative, config
    )
    criterion_metrics = _criterion_metrics(
        strategy_metrics, monthly_tests, best_alternative, config
    )
    criterion_result = evaluate_success_criterion(criterion, criterion_metrics)
    holdout_confirmation = _holdout_confirmation(
        monthly_tests, best_alternative, config
    )
    if not criterion_result.passed:
        status = "blocked_validation"
    elif not holdout_confirmation:
        status = "blocked_holdout_confirmation"
    else:
        status = "eligible_for_strategy_review"

    summary = {
        "schema_version": ROUTER_HYPOTHESIS_SCHEMA_VERSION,
        "hypothesis_id": config.hypothesis_id,
        "router_id": config.router_id,
        "economic_claim": config.economic_claim,
        "economic_mechanism": config.economic_mechanism,
        "score_target": "next_holding_period_net_return_sleeve_a_minus_sleeve_b",
        "best_validation_alternative": best_alternative,
        "validation_decision": criterion_result.decision.value,
        "holdout_confirmation_passed": holdout_confirmation,
        "router_status": status,
        "oracle_role": "unattainable_upper_bound_excluded_from_all_gates",
        "validation": _split_summary_payload(
            strategy_metrics, routing_metrics, monthly_tests, config.validation_label
        ),
        "holdout": _split_summary_payload(
            strategy_metrics, routing_metrics, monthly_tests, config.holdout_label
        ),
        "success_criterion": _json_safe(criterion_result.to_dict()),
    }
    manifest = {
        "schema_version": ROUTER_HYPOTHESIS_SCHEMA_VERSION,
        "phase": "Phase 6: Router Hypothesis",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "criterion": criterion.to_dict(),
        "criterion_fingerprint": criterion.fingerprint,
        "criterion_result": criterion_result.to_dict(),
        "sleeve_a": _sleeve_manifest_reference(phase3_a, phase4_a),
        "sleeve_b": _sleeve_manifest_reference(phase3_b, phase4_b),
        "causal_alignment_verified": True,
        "score_timing": "known_after_close_t_before_next_open_execution",
        "relative_target": "net_return_a_t_plus_1_minus_net_return_b_t_plus_1",
        "final_positions_reexecuted": True,
        "costs_recomputed_after_combining_targets": True,
        "optimization_permitted": False,
        "oracle_permitted_as_strategy": False,
        "oracle_excluded_from_selection_and_gates": True,
        "multiple_hypothesis_search_permitted": False,
    }
    return RouterHypothesisEvidenceBundle(
        config=config,
        criterion=criterion,
        criterion_result=criterion_result,
        summary=summary,
        decision_log=decision_log,
        relative_advantage=relative,
        execution_positions=execution_positions,
        daily_comparison=daily_comparison.reset_index(drop=True),
        strategy_metrics=strategy_metrics,
        routing_metrics=routing_metrics,
        monthly_paired_tests=monthly_tests,
        subperiod_stability=stability,
        manifest=manifest,
    )


def write_router_hypothesis_evidence_bundle(
    bundle: RouterHypothesisEvidenceBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", bundle.summary)
    _write_json(destination / "manifest.json", bundle.manifest)
    bundle.decision_log.to_parquet(destination / "decision_log.parquet", index=False)
    bundle.relative_advantage.to_parquet(
        destination / "relative_advantage.parquet", index=False
    )
    bundle.execution_positions.to_parquet(
        destination / "execution_positions.parquet", index=False
    )
    bundle.daily_comparison.to_parquet(
        destination / "daily_comparison.parquet", index=False
    )
    bundle.strategy_metrics.to_csv(destination / "strategy_metrics.csv", index=False)
    bundle.routing_metrics.to_csv(destination / "routing_metrics.csv", index=False)
    bundle.monthly_paired_tests.to_csv(
        destination / "monthly_paired_tests.csv", index=False
    )
    bundle.subperiod_stability.to_csv(
        destination / "subperiod_stability.csv", index=False
    )
    return destination


def load_router_hypothesis_evidence_bundle(
    output_dir: str | Path,
) -> RouterHypothesisEvidenceBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config = RouterHypothesisConfig(**manifest["config"])
    criterion = SuccessCriterionSpec.from_mapping(
        manifest["criterion"]["profile_id"], manifest["criterion"]
    )
    criterion_result = evaluate_success_criterion(
        criterion,
        _criterion_metrics_from_saved(
            pd.read_csv(source / "strategy_metrics.csv"),
            pd.read_csv(source / "monthly_paired_tests.csv"),
            summary["best_validation_alternative"],
            config,
        ),
    )
    return RouterHypothesisEvidenceBundle(
        config=config,
        criterion=criterion,
        criterion_result=criterion_result,
        summary=summary,
        decision_log=pd.read_parquet(source / "decision_log.parquet"),
        relative_advantage=pd.read_parquet(source / "relative_advantage.parquet"),
        execution_positions=pd.read_parquet(source / "execution_positions.parquet"),
        daily_comparison=pd.read_parquet(source / "daily_comparison.parquet"),
        strategy_metrics=pd.read_csv(source / "strategy_metrics.csv"),
        routing_metrics=pd.read_csv(source / "routing_metrics.csv"),
        monthly_paired_tests=pd.read_csv(source / "monthly_paired_tests.csv"),
        subperiod_stability=pd.read_csv(source / "subperiod_stability.csv"),
        manifest=manifest,
    )


def audit_router_readiness(
    phase4_root: str | Path,
    *,
    frozen_hypothesis_count: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize whether the library can form an admissible Phase 6 pair."""

    root = Path(phase4_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "factor_id": payload.get("factor_id", ""),
                "sleeve_id": payload.get("sleeve_id", ""),
                "standalone_status": payload.get("standalone_status", "unknown"),
                "router_eligible": bool(payload.get("router_eligible", False)),
                "validation_net_sharpe": _nested_value(
                    payload, "validation", "net_sharpe"
                ),
                "holdout_net_sharpe": _nested_value(
                    payload, "holdout", "net_sharpe"
                ),
                "artifact_path": str(path.parent),
            }
        )
    sleeves = pd.DataFrame(rows)
    eligible = int(sleeves["router_eligible"].sum()) if not sleeves.empty else 0
    pair_count = eligible * (eligible - 1) // 2
    hypotheses = int(frozen_hypothesis_count)
    ready = pair_count > 0 and hypotheses > 0
    summary = {
        "schema_version": ROUTER_HYPOTHESIS_SCHEMA_VERSION,
        "phase": "Phase 6: Router Hypothesis",
        "standalone_sleeves": int(len(sleeves)),
        "eligible_sleeves": eligible,
        "eligible_pairs": pair_count,
        "frozen_hypotheses": hypotheses,
        "ready_for_empirical_router_test": ready,
        "status": "ready" if ready else "blocked",
        "blockers": [
            message
            for condition, message in (
                (
                    eligible < 2,
                    "At least two independently eligible Phase 4 sleeves are required.",
                ),
                (
                    hypotheses < 1,
                    "A dated economic hypothesis must be frozen before untouched data.",
                ),
            )
            if condition
        ],
        "oracle_role": "unattainable_upper_bound_only",
    }
    return summary, sleeves


def write_router_readiness_snapshot(
    summary: dict[str, Any],
    sleeves: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "readiness.json", summary)
    sleeves.to_csv(destination / "sleeves.csv", index=False)
    return destination


def _validate_inputs(
    phase3_a: SleeveEvidenceBundle,
    phase4_a: StandaloneSleeveTestBundle,
    phase3_b: SleeveEvidenceBundle,
    phase4_b: StandaloneSleeveTestBundle,
    config: RouterHypothesisConfig,
) -> None:
    pairs = ((phase3_a, phase4_a, "A"), (phase3_b, phase4_b, "B"))
    expected = {
        "A": (config.sleeve_a_factor_id, config.sleeve_a_id),
        "B": (config.sleeve_b_factor_id, config.sleeve_b_id),
    }
    for phase3, phase4, label in pairs:
        observed = (phase3.config.factor_id, phase3.config.sleeve_id)
        if observed != expected[label]:
            raise ValueError(f"sleeve {label} identity does not match the hypothesis")
        if not bool(phase4.summary.get("router_eligible", False)):
            raise ValueError(f"sleeve {label} did not pass standalone Phase 4 gates")
        if phase4.manifest.get("phase3_config_fingerprint") != phase3.manifest.get(
            "config_fingerprint"
        ):
            raise ValueError(f"sleeve {label} Phase 4 attestation does not match Phase 3")
        if phase3.config.market_vertical != config.market_vertical:
            raise ValueError(f"sleeve {label} market does not match the hypothesis")
        if bool(phase3.config.optimization_permitted):
            raise ValueError("optimized sleeves cannot enter the frozen Phase 6 test")
        if not bool(phase3.manifest.get("causal_alignment_verified")):
            raise ValueError(f"sleeve {label} lacks verified causal alignment")
    comparable = (
        "return_assumption",
        "holding_periods",
        "date_col",
        "product_col",
        "split_col",
        "return_col",
    )
    for field in comparable:
        if getattr(phase3_a.config, field) != getattr(phase3_b.config, field):
            raise ValueError(f"router sleeves use different {field}")
    if phase3_a.manifest.get("input_data_fingerprint") != phase3_b.manifest.get(
        "input_data_fingerprint"
    ):
        raise ValueError("router sleeves must use the same frozen dataset")
    if phase3_a.manifest.get("execution") != phase3_b.manifest.get("execution"):
        raise ValueError("router sleeves must use the same execution and cost profile")


def _prepare_scores(
    scores: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    required = {config.date_col, config.score_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"router score frame is missing columns: {missing}")
    out = scores[[config.date_col, config.score_col]].copy()
    out[config.date_col] = pd.to_datetime(
        out[config.date_col], errors="raise"
    ).dt.normalize()
    if out[config.date_col].duplicated().any():
        raise ValueError("router score must be unique by decision date")
    out[config.score_col] = pd.to_numeric(out[config.score_col], errors="coerce")
    out = out.dropna(subset=[config.score_col]).sort_values(config.date_col)
    if config.score_orientation == "higher_favors_a":
        out["predicted_relative_advantage_score"] = (
            out[config.score_col] - config.threshold
        )
    else:
        out["predicted_relative_advantage_score"] = (
            config.threshold - out[config.score_col]
        )
    return out.reset_index(drop=True)


def _align_sleeve_positions(
    phase3_a: SleeveEvidenceBundle,
    phase3_b: SleeveEvidenceBundle,
) -> pd.DataFrame:
    config = phase3_a.config
    keys = [config.date_col, config.product_col]
    metadata = [
        config.sector_col,
        config.split_col,
        config.return_col,
        "next_symbol",
        "next_actual_open",
        "next_multiplier",
        "next_tick_size",
        "next_fee_type",
        "next_fee_open",
        "next_fee_close_today",
    ]
    required = set(keys + metadata + ["target_weight"])
    for frame, label in ((phase3_a.positions, "A"), (phase3_b.positions, "B")):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"sleeve {label} positions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"sleeve {label} positions are not unique by date/product")
    left = phase3_a.positions[keys + metadata + ["target_weight"]].rename(
        columns={"target_weight": "target_weight_a"}
    )
    right = phase3_b.positions[keys + metadata + ["target_weight"]].rename(
        columns={
            **{column: f"{column}_b" for column in metadata},
            "target_weight": "target_weight_b",
        }
    )
    out = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(out) != len(left) or len(out) != len(right):
        raise ValueError("router sleeves do not share the same date/product panel")
    for column in metadata:
        other = f"{column}_b"
        if pd.api.types.is_numeric_dtype(out[column]) or pd.api.types.is_numeric_dtype(
            out[other]
        ):
            a = pd.to_numeric(out[column], errors="coerce")
            b = pd.to_numeric(out[other], errors="coerce")
            equal = np.isclose(a, b, equal_nan=True)
        else:
            equal = out[column].astype("string").fillna("<NA>").eq(
                out[other].astype("string").fillna("<NA>")
            )
        if not bool(np.all(equal)):
            raise ValueError(f"router sleeves disagree on shared field {column}")
        out = out.drop(columns=other)
    out[config.date_col] = pd.to_datetime(out[config.date_col]).dt.normalize()
    return out.sort_values(keys).reset_index(drop=True)


def _execute_combination(
    base: pd.DataFrame,
    schedule: pd.DataFrame,
    phase3_template: SleeveEvidenceBundle,
    *,
    strategy_id: str,
    weight_a: float | str,
    weight_b: float | str,
    capital: float,
    slippage: float,
    scale_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = phase3_template.config
    dynamic = [value for value in (weight_a, weight_b, scale_col) if isinstance(value, str)]
    schedule_columns = [config.date_col, *dict.fromkeys(dynamic)]
    allocation = schedule[schedule_columns].copy()
    out = base.merge(allocation, on=config.date_col, how="inner", validate="many_to_one")
    a = pd.to_numeric(out[weight_a], errors="coerce") if isinstance(weight_a, str) else float(weight_a)
    b = pd.to_numeric(out[weight_b], errors="coerce") if isinstance(weight_b, str) else float(weight_b)
    scale: float | pd.Series
    scale = (
        pd.to_numeric(out[scale_col], errors="coerce").fillna(1.0)
        if scale_col
        else 1.0
    )
    out["target_weight"] = (
        a * pd.to_numeric(out["target_weight_a"], errors="coerce").fillna(0.0)
        + b * pd.to_numeric(out["target_weight_b"], errors="coerce").fillna(0.0)
    ) * scale
    out["contract_cap_bound"] = False
    out["sector_cap_bound"] = False
    executed = execute_intraday_session_targets(
        out,
        config,
        capital=capital,
        slippage_ticks_per_side=slippage,
    )
    executed["strategy_id"] = strategy_id
    executed["attainable"] = strategy_id != ORACLE_STRATEGY
    daily = summarize_executed_positions(executed, config)
    daily["strategy_id"] = strategy_id
    daily["attainable"] = strategy_id != ORACLE_STRATEGY
    return executed, daily


def _causal_exposure_schedule(
    static_daily: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    ordered = static_daily.sort_values(config.date_col).copy()
    trailing = ordered["net_return"].shift(1).rolling(
        config.exposure_lookback_periods,
        min_periods=config.exposure_min_periods,
    ).std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)
    raw = config.exposure_target_volatility / trailing.replace(0.0, np.nan)
    ordered["trailing_blend_volatility"] = trailing
    ordered["exposure_scale"] = raw.clip(
        lower=config.exposure_min_scale,
        upper=config.exposure_max_scale,
    ).fillna(config.exposure_max_scale)
    return ordered[
        [config.date_col, "trailing_blend_volatility", "exposure_scale"]
    ]


def _build_decision_log(
    score: pd.DataFrame,
    base: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    split = base[[config.date_col, "research_split"]].drop_duplicates()
    out = score.merge(split, on=config.date_col, how="left", validate="one_to_one")
    out["selected_sleeve"] = np.where(
        out["predicted_relative_advantage_score"].ge(0.0),
        "sleeve_a",
        "sleeve_b",
    )
    out["allocation_a"] = out["selected_sleeve"].eq("sleeve_a").astype(float)
    out["allocation_b"] = 1.0 - out["allocation_a"]
    out["switch"] = out["selected_sleeve"].ne(out["selected_sleeve"].shift(1))
    if len(out):
        out.loc[out.index[0], "switch"] = False
    out["hypothesis_id"] = config.hypothesis_id
    out["router_id"] = config.router_id
    out["execution_start"] = "next_actual_open_after_decision"
    return out.reset_index(drop=True)


def _relative_advantage(
    sleeve_a_daily: pd.DataFrame,
    sleeve_b_daily: pd.DataFrame,
    decision_log: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    columns = [config.date_col, "gross_return", "net_return", "cost_return", "turnover"]
    a = sleeve_a_daily[columns].rename(
        columns={column: f"{column}_a" for column in columns if column != config.date_col}
    )
    b = sleeve_b_daily[columns].rename(
        columns={column: f"{column}_b" for column in columns if column != config.date_col}
    )
    out = decision_log.merge(a, on=config.date_col, how="inner", validate="one_to_one")
    out = out.merge(b, on=config.date_col, how="inner", validate="one_to_one")
    out["relative_net_advantage_a_minus_b"] = out["net_return_a"] - out["net_return_b"]
    out["relative_gross_advantage_a_minus_b"] = out["gross_return_a"] - out["gross_return_b"]
    out["better_sleeve"] = np.select(
        [out["relative_net_advantage_a_minus_b"].gt(0.0), out["relative_net_advantage_a_minus_b"].lt(0.0)],
        ["sleeve_a", "sleeve_b"],
        default="tie",
    )
    out["selection_correct"] = out["selected_sleeve"].eq(out["better_sleeve"]).where(
        out["better_sleeve"].ne("tie")
    )
    selected_a = out["selected_sleeve"].eq("sleeve_a")
    out["selected_minus_unselected_net"] = np.where(
        selected_a,
        out["relative_net_advantage_a_minus_b"],
        -out["relative_net_advantage_a_minus_b"],
    )
    return out


def _strategy_metrics(
    daily: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in (config.validation_label, config.holdout_label, "full"):
        split_frame = daily if split == "full" else daily.loc[
            daily["research_split"].eq(split)
        ]
        for strategy_id, sample in split_frame.groupby("strategy_id", sort=False):
            net = pd.to_numeric(sample["net_return"], errors="coerce").dropna()
            gross = pd.to_numeric(sample["gross_return"], errors="coerce").dropna()
            wealth = (1.0 + net).cumprod()
            drawdown = wealth / wealth.cummax() - 1.0
            rows.append(
                {
                    "research_split": split,
                    "strategy_id": strategy_id,
                    "attainable": bool(strategy_id != ORACLE_STRATEGY),
                    "date_count": int(len(sample)),
                    "active_date_count": int(sample["active_products"].gt(0).sum()),
                    "gross_total_return": _compound(gross),
                    "net_total_return": _compound(net),
                    "gross_annualized_mean": _annualized_mean(gross),
                    "net_annualized_mean": _annualized_mean(net),
                    "net_annualized_volatility": _annualized_volatility(net),
                    "net_sharpe": _sharpe(net),
                    "maximum_drawdown": float(drawdown.min()) if len(drawdown) else math.nan,
                    "annualized_turnover": _annualized_mean(sample["turnover"]),
                    "annualized_cost": _annualized_mean(sample["cost_return"]),
                    "positive_net_day_fraction": float(net.gt(0.0).mean()) if len(net) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _routing_metrics(
    relative: pd.DataFrame,
    daily: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in (config.validation_label, config.holdout_label, "full"):
        sample = relative if split == "full" else relative.loc[
            relative["research_split"].eq(split)
        ]
        router = _daily_strategy(daily, "router", split, config)
        static = _daily_strategy(daily, "static_blend", split, config)
        monthly_ic = _monthly_correlations(sample, config, method="pearson")
        monthly_rank = _monthly_correlations(sample, config, method="spearman")
        non_ties = sample.loc[sample["selection_correct"].notna()]
        correct = non_ties.loc[non_ties["selection_correct"]]
        wrong = non_ties.loc[~non_ties["selection_correct"]]
        rows.append(
            {
                "research_split": split,
                "routing_ic": _corr(
                    sample["predicted_relative_advantage_score"],
                    sample["relative_net_advantage_a_minus_b"],
                    "pearson",
                ),
                "routing_rank_ic": _corr(
                    sample["predicted_relative_advantage_score"],
                    sample["relative_net_advantage_a_minus_b"],
                    "spearman",
                ),
                "routing_icir": _icir(monthly_ic),
                "routing_rank_icir": _icir(monthly_rank),
                "valid_routing_dates": int(len(sample)),
                "valid_routing_months": int(
                    pd.to_datetime(sample[config.date_col]).dt.to_period("M").nunique()
                ),
                "better_sleeve_hit_rate": float(non_ties["selection_correct"].mean()) if len(non_ties) else math.nan,
                "correct_dates": int(len(correct)),
                "wrong_dates": int(len(wrong)),
                "tie_dates": int(sample["better_sleeve"].eq("tie").sum()),
                "mean_selected_advantage_correct_bps": _mean(correct["selected_minus_unselected_net"]) * 10_000.0,
                "mean_selected_advantage_wrong_bps": _mean(wrong["selected_minus_unselected_net"]) * 10_000.0,
                "switch_count": int(sample["switch"].sum()),
                "switch_rate": float(sample["switch"].mean()) if len(sample) else math.nan,
                "router_minus_static_annualized_turnover": _annualized_mean(router["turnover"]) - _annualized_mean(static["turnover"]),
                "router_minus_static_annualized_cost": _annualized_mean(router["cost_return"]) - _annualized_mean(static["cost_return"]),
            }
        )
    return pd.DataFrame(rows)


def _monthly_paired_tests(
    daily: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    work = daily.copy()
    work["month"] = pd.to_datetime(work[config.date_col]).dt.to_period("M").astype(str)
    monthly = (
        work.groupby(["research_split", "strategy_id", "month"], sort=True)["net_return"]
        .apply(_compound)
        .rename("monthly_net_return")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for split in (config.validation_label, config.holdout_label, "full"):
        sample = monthly if split == "full" else monthly.loc[
            monthly["research_split"].eq(split)
        ]
        router = sample.loc[sample["strategy_id"].eq("router"), ["month", "monthly_net_return"]].rename(
            columns={"monthly_net_return": "router_monthly_return"}
        )
        for comparator in ATTAINABLE_STRATEGIES[:-1]:
            other = sample.loc[sample["strategy_id"].eq(comparator), ["month", "monthly_net_return"]].rename(
                columns={"monthly_net_return": "comparator_monthly_return"}
            )
            paired = router.merge(other, on="month", how="inner", validate="one_to_one")
            increment = paired["router_monthly_return"] - paired["comparator_monthly_return"]
            mean = _mean(increment)
            se = _newey_west_mean_se(increment, config.monthly_hac_max_lag)
            critical = NormalDist().inv_cdf(0.5 + config.confidence_level / 2.0)
            rows.append(
                {
                    "research_split": split,
                    "comparator": comparator,
                    "month_count": int(len(paired)),
                    "mean_monthly_increment": mean,
                    "annualized_increment": mean * 12.0,
                    "monthly_increment_hac_se": se,
                    "monthly_increment_hac_t": mean / se if se > 0.0 else math.nan,
                    "monthly_increment_ci_lower": mean - critical * se if np.isfinite(se) else math.nan,
                    "monthly_increment_ci_upper": mean + critical * se if np.isfinite(se) else math.nan,
                    "positive_increment_month_fraction": float(increment.gt(0.0).mean()) if len(increment) else math.nan,
                    "hac_max_lag": config.monthly_hac_max_lag,
                }
            )
    return pd.DataFrame(rows)


def _subperiod_stability(
    relative: pd.DataFrame,
    daily: pd.DataFrame,
    best_alternative: str,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    work = relative.copy()
    work["year"] = pd.to_datetime(work[config.date_col]).dt.year
    routed = daily.loc[daily["strategy_id"].eq("router"), [config.date_col, "net_return"]].rename(columns={"net_return": "router_net_return"})
    comparator = daily.loc[daily["strategy_id"].eq(best_alternative), [config.date_col, "net_return"]].rename(columns={"net_return": "alternative_net_return"})
    work = work.merge(routed, on=config.date_col, how="left", validate="one_to_one")
    work = work.merge(comparator, on=config.date_col, how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for year, sample in work.groupby("year", sort=True):
        non_ties = sample.loc[sample["selection_correct"].notna()]
        increment = sample["router_net_return"] - sample["alternative_net_return"]
        rows.append(
            {
                "year": int(year),
                "research_split": str(sample["research_split"].mode().iloc[0]),
                "date_count": int(len(sample)),
                "routing_ic": _corr(sample["predicted_relative_advantage_score"], sample["relative_net_advantage_a_minus_b"], "pearson"),
                "routing_rank_ic": _corr(sample["predicted_relative_advantage_score"], sample["relative_net_advantage_a_minus_b"], "spearman"),
                "better_sleeve_hit_rate": float(non_ties["selection_correct"].mean()) if len(non_ties) else math.nan,
                "router_minus_best_alternative_annualized_mean": _annualized_mean(increment),
            }
        )
    return pd.DataFrame(rows)


def _best_validation_alternative(
    strategy_metrics: pd.DataFrame,
    config: RouterHypothesisConfig,
) -> str:
    eligible = strategy_metrics.loc[
        strategy_metrics["research_split"].eq(config.validation_label)
        & strategy_metrics["strategy_id"].isin(ATTAINABLE_STRATEGIES[:-1])
    ].dropna(subset=["net_annualized_mean"])
    if eligible.empty:
        raise ValueError("no attainable validation comparator is available")
    return str(eligible.sort_values("net_annualized_mean", ascending=False).iloc[0]["strategy_id"])


def _criterion_metrics(
    strategy_metrics: pd.DataFrame,
    monthly_tests: pd.DataFrame,
    best_alternative: str,
    config: RouterHypothesisConfig,
) -> dict[str, float]:
    router = _strategy_metric_row(strategy_metrics, "router", config.validation_label)
    comparator = _strategy_metric_row(
        strategy_metrics, best_alternative, config.validation_label
    )
    paired = monthly_tests.loc[
        monthly_tests["research_split"].eq(config.validation_label)
        & monthly_tests["comparator"].eq(best_alternative)
    ].iloc[0]
    return {
        "validation_net_mean_return": float(router["net_annualized_mean"]),
        "best_alternative_validation_net_mean_return": float(
            comparator["net_annualized_mean"]
        ),
        "validation_increment_hac_t": float(paired["monthly_increment_hac_t"]),
        "validation_net_sharpe": float(router["net_sharpe"]),
        "validation_valid_routing_periods": float(paired["month_count"]),
    }


def _criterion_metrics_from_saved(
    strategy_metrics: pd.DataFrame,
    monthly_tests: pd.DataFrame,
    best_alternative: str,
    config: RouterHypothesisConfig,
) -> dict[str, float]:
    return _criterion_metrics(strategy_metrics, monthly_tests, best_alternative, config)


def _holdout_confirmation(
    monthly_tests: pd.DataFrame,
    best_alternative: str,
    config: RouterHypothesisConfig,
) -> bool:
    row = monthly_tests.loc[
        monthly_tests["research_split"].eq(config.holdout_label)
        & monthly_tests["comparator"].eq(best_alternative)
    ]
    return bool(
        not row.empty
        and pd.notna(row.iloc[0]["mean_monthly_increment"])
        and float(row.iloc[0]["mean_monthly_increment"]) > 0.0
    )


def _split_summary_payload(
    strategy_metrics: pd.DataFrame,
    routing_metrics: pd.DataFrame,
    monthly_tests: pd.DataFrame,
    split: str,
) -> dict[str, Any]:
    strategy = strategy_metrics.loc[strategy_metrics["research_split"].eq(split)]
    routing = routing_metrics.loc[routing_metrics["research_split"].eq(split)]
    paired = monthly_tests.loc[monthly_tests["research_split"].eq(split)]
    return _json_safe(
        {
            "strategies": strategy.to_dict(orient="records"),
            "routing": routing.iloc[0].to_dict() if not routing.empty else {},
            "paired_monthly_hac": paired.to_dict(orient="records"),
        }
    )


def _sleeve_manifest_reference(
    phase3: SleeveEvidenceBundle,
    phase4: StandaloneSleeveTestBundle,
) -> dict[str, Any]:
    return {
        "factor_id": phase3.config.factor_id,
        "sleeve_id": phase3.config.sleeve_id,
        "phase3_config_fingerprint": phase3.manifest.get("config_fingerprint"),
        "phase4_config_fingerprint": phase4.manifest.get("config_fingerprint"),
        "phase4_router_eligible": bool(phase4.summary.get("router_eligible")),
        "standalone_status": phase4.summary.get("standalone_status"),
    }


def _holdout_start(base: pd.DataFrame, config: RouterHypothesisConfig) -> pd.Timestamp:
    values = base.loc[
        base["research_split"].eq(config.holdout_label), config.date_col
    ]
    if values.empty:
        raise ValueError("Phase 6 requires an untouched holdout split")
    return pd.Timestamp(values.min()).normalize()


def _daily_strategy(
    daily: pd.DataFrame,
    strategy_id: str,
    split: str,
    config: RouterHypothesisConfig,
) -> pd.DataFrame:
    out = daily.loc[daily["strategy_id"].eq(strategy_id)]
    if split != "full":
        out = out.loc[out["research_split"].eq(split)]
    return out


def _strategy_metric_row(
    metrics: pd.DataFrame,
    strategy_id: str,
    split: str,
) -> pd.Series:
    row = metrics.loc[
        metrics["strategy_id"].eq(strategy_id)
        & metrics["research_split"].eq(split)
    ]
    if row.empty:
        raise ValueError(f"missing {split} metrics for {strategy_id}")
    return row.iloc[0]


def _monthly_correlations(
    sample: pd.DataFrame,
    config: RouterHypothesisConfig,
    *,
    method: str,
) -> pd.Series:
    if sample.empty:
        return pd.Series(dtype=float)
    work = sample.copy()
    work["month"] = pd.to_datetime(work[config.date_col]).dt.to_period("M")
    return work.groupby("month", sort=True).apply(
        lambda frame: _corr(
            frame["predicted_relative_advantage_score"],
            frame["relative_net_advantage_a_minus_b"],
            method,
        ),
        include_groups=False,
    ).dropna()


def _corr(left: pd.Series, right: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return math.nan
    return float(frame["left"].corr(frame["right"], method=method))


def _icir(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return math.nan
    std = float(clean.std(ddof=1))
    return float(clean.mean() / std) if std > 0.0 else math.nan


def _newey_west_mean_se(values: Iterable[float], max_lag: int) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    count = len(clean)
    if count < 2:
        return math.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / count)
    usable_lag = min(int(max_lag), count - 1)
    for lag in range(1, usable_lag + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / count)
        long_run_variance += 2.0 * (1.0 - lag / (usable_lag + 1.0)) * covariance
    return math.sqrt(max(long_run_variance, 0.0) / count)


def _compound(values: Iterable[float]) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float((1.0 + clean).prod() - 1.0) if len(clean) else math.nan


def _mean(values: Iterable[float]) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def _annualized_mean(values: Iterable[float]) -> float:
    return _mean(values) * ANNUALIZATION_DAYS


def _annualized_volatility(values: Iterable[float]) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(clean.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)) if len(clean) > 1 else math.nan


def _sharpe(values: Iterable[float]) -> float:
    mean = _annualized_mean(values)
    volatility = _annualized_volatility(values)
    return mean / volatility if volatility and volatility > 0.0 else math.nan


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ATTAINABLE_STRATEGIES",
    "ORACLE_STRATEGY",
    "ROUTER_HYPOTHESIS_SCHEMA_VERSION",
    "RouterHypothesisConfig",
    "RouterHypothesisEvidenceBundle",
    "audit_router_readiness",
    "build_router_hypothesis_evidence",
    "load_router_hypothesis_evidence_bundle",
    "write_router_hypothesis_evidence_bundle",
    "write_router_readiness_snapshot",
]
