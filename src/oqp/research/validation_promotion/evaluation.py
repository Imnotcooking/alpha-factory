"""Phase 10 router validation and promotion decisions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.strategy_routing import RouterHypothesisEvidenceBundle
from oqp.research.validation_promotion.contracts import (
    PHASE10_SCHEMA_VERSION,
    PaperTradingEvidence,
    PromotionDecision,
    PromotionGateResult,
    PromotionGateStatus,
    PromotionPolicyRegistry,
    RouterPromotionPolicy,
    RouterPromotionReviewBundle,
    RouterPromotionReviewConfig,
    stable_promotion_hash,
)


ANNUALIZATION_DAYS = 252.0


def router_evidence_fingerprint(
    evidence: RouterHypothesisEvidenceBundle,
) -> str:
    payload = {
        "manifest": evidence.manifest,
        "summary": evidence.summary,
        "frames": {
            name: _frame_fingerprint(getattr(evidence, name))
            for name in (
                "decision_log",
                "execution_positions",
                "daily_comparison",
                "strategy_metrics",
                "routing_metrics",
                "monthly_paired_tests",
                "subperiod_stability",
            )
        },
    }
    return stable_promotion_hash(payload)


def build_router_promotion_review(
    evidence: RouterHypothesisEvidenceBundle,
    perturbations: pd.DataFrame,
    config: RouterPromotionReviewConfig,
    *,
    policy: RouterPromotionPolicy | None = None,
    paper_evidence: PaperTradingEvidence | None = None,
) -> RouterPromotionReviewBundle:
    policy = policy or PromotionPolicyRegistry.load().resolve(
        config.policy_profile_id
    )
    if policy.profile_id != config.policy_profile_id:
        raise ValueError("promotion review resolved the wrong policy profile")
    if str(evidence.config.router_id) != config.router_id:
        raise ValueError("promotion review router ID does not match its evidence")

    observed_evidence_fingerprint = router_evidence_fingerprint(evidence)
    best_alternative = str(evidence.summary["best_validation_alternative"])
    month_concentration = _month_concentration(
        evidence.daily_comparison,
        best_alternative,
        policy,
        evidence.config.validation_label,
        evidence.config.holdout_label,
    )
    product_concentration = _product_concentration(
        evidence.execution_positions,
        best_alternative,
        policy,
        evidence.config.validation_label,
        evidence.config.holdout_label,
    )
    validation_periods = evidence.subperiod_stability.copy()
    perturbation_results = _prepare_perturbations(
        perturbations, config.perturbation_plan_fingerprint
    )

    gates: list[PromotionGateResult] = []
    _append_governance_gates(
        gates,
        evidence,
        config,
        observed_evidence_fingerprint,
        perturbation_results,
        policy,
    )
    _append_core_economic_gates(
        gates,
        evidence,
        month_concentration,
        product_concentration,
        validation_periods,
        perturbation_results,
        best_alternative,
        policy,
    )
    if paper_evidence is not None:
        _append_paper_trading_gates(
            gates, paper_evidence, evidence, config, policy
        )

    decision = _promotion_decision(gates, paper_evidence is not None)
    current_stage = "paper_trading" if paper_evidence is not None else "frozen_holdout"
    next_stage = {
        PromotionDecision.ELIGIBLE_FOR_PAPER_TRADING: "paper_trading",
        PromotionDecision.ELIGIBLE_FOR_PRODUCTION_REVIEW: "production_review",
    }.get(decision)
    gate_frame = pd.DataFrame([gate.to_dict() for gate in gates])
    failed_gate_ids = gate_frame.loc[
        gate_frame["status"].ne(PromotionGateStatus.PASS.value), "gate_id"
    ].astype(str).tolist()
    summary = {
        "schema_version": PHASE10_SCHEMA_VERSION,
        "review_id": config.review_id,
        "router_id": config.router_id,
        "policy_profile_id": policy.profile_id,
        "decision": decision.value,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "best_frozen_comparator": best_alternative,
        "gate_count": len(gates),
        "passed_gate_count": int(
            gate_frame["status"].eq(PromotionGateStatus.PASS.value).sum()
        ),
        "failed_gate_ids": failed_gate_ids,
        "paper_evidence_supplied": paper_evidence is not None,
        "failure_is_valid_research_result": (
            decision is PromotionDecision.FAILED_RESEARCH_RESULT
        ),
        "full_sample_sharpe_used_for_promotion": False,
        "conclusion": _decision_explanation(decision),
    }
    manifest = {
        "schema_version": PHASE10_SCHEMA_VERSION,
        "phase": "Phase 10: Validation and Promotion",
        "lifecycle": [
            "discovery",
            "chronological_validation",
            "frozen_holdout",
            "paper_trading",
            "production_review",
        ],
        "review_config": config.to_dict(),
        "review_config_fingerprint": config.fingerprint,
        "policy": policy.to_dict(),
        "policy_fingerprint": policy.fingerprint,
        "source_router_config_fingerprint": evidence.config.fingerprint,
        "source_evidence_fingerprint": observed_evidence_fingerprint,
        "paper_evidence": asdict(paper_evidence) if paper_evidence else None,
        "paper_evidence_fingerprint": (
            paper_evidence.fingerprint if paper_evidence else None
        ),
        "promotion_uses_full_sample_sharpe": False,
        "baseline_frozen_from_validation": best_alternative,
        "holdout_reselection_permitted": False,
        "failed_results_retained": True,
    }
    return RouterPromotionReviewBundle(
        config=config,
        policy=policy,
        decision=decision,
        current_stage=current_stage,
        next_stage=next_stage,
        summary=summary,
        gate_results=gate_frame,
        month_concentration=month_concentration,
        product_concentration=product_concentration,
        validation_periods=validation_periods,
        perturbations=perturbation_results,
        manifest=manifest,
    )


def write_router_promotion_review(
    bundle: RouterPromotionReviewBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", bundle.summary)
    _write_json(destination / "manifest.json", bundle.manifest)
    bundle.gate_results.to_csv(destination / "gate_results.csv", index=False)
    bundle.month_concentration.to_csv(
        destination / "month_concentration.csv", index=False
    )
    bundle.product_concentration.to_csv(
        destination / "product_concentration.csv", index=False
    )
    bundle.validation_periods.to_csv(
        destination / "validation_periods.csv", index=False
    )
    bundle.perturbations.to_csv(destination / "perturbations.csv", index=False)
    return destination


def _append_governance_gates(
    gates: list[PromotionGateResult],
    evidence: RouterHypothesisEvidenceBundle,
    config: RouterPromotionReviewConfig,
    observed_evidence_fingerprint: str,
    perturbations: pd.DataFrame,
    policy: RouterPromotionPolicy,
) -> None:
    _gate(
        gates,
        "router_config_frozen",
        "reproducibility",
        evidence.config.fingerprint == config.router_config_fingerprint,
        evidence.config.fingerprint,
        "==",
        config.router_config_fingerprint,
        "governance",
        "The router configuration must match the fingerprint frozen for review.",
    )
    _gate(
        gates,
        "source_evidence_frozen",
        "reproducibility",
        observed_evidence_fingerprint == config.source_evidence_fingerprint,
        observed_evidence_fingerprint,
        "==",
        config.source_evidence_fingerprint,
        "governance",
        "The exact Phase 6 evidence bundle must be reproducible.",
    )
    plan_matches = bool(
        not len(perturbations)
        or perturbations["plan_fingerprint"]
        .astype(str)
        .eq(config.perturbation_plan_fingerprint)
        .all()
    )
    _gate(
        gates,
        "perturbation_plan_frozen",
        "reproducibility",
        plan_matches,
        bool(plan_matches),
        "==",
        True,
        "governance",
        "Every perturbation must belong to the frozen robustness plan.",
    )
    manifest_checks = {
        "causal_alignment_verified": True,
        "costs_recomputed_after_combining_targets": True,
        "oracle_excluded_from_selection_and_gates": True,
    }
    for key, expected in manifest_checks.items():
        observed = bool(evidence.manifest.get(key, False))
        _gate(
            gates,
            key,
            "reproducibility",
            observed is expected,
            observed,
            "==",
            expected,
            "governance",
            f"Phase 6 manifest must confirm {key}.",
        )
    holdout_dates = pd.to_datetime(
        evidence.daily_comparison.loc[
            evidence.daily_comparison[policy.split_col].eq(
                evidence.config.holdout_label
            ),
            policy.date_col,
        ],
        errors="coerce",
    ).dropna()
    holdout_end = holdout_dates.max() if len(holdout_dates) else pd.NaT
    review_after_holdout = bool(
        pd.notna(holdout_end)
        and pd.Timestamp(config.reviewed_on) >= pd.Timestamp(holdout_end).normalize()
    )
    _gate(
        gates,
        "review_after_frozen_holdout",
        "reproducibility",
        review_after_holdout,
        config.reviewed_on,
        ">=",
        str(pd.Timestamp(holdout_end).date()) if pd.notna(holdout_end) else "missing",
        "governance",
        "Promotion review must occur after the frozen holdout window is complete.",
    )


def _append_core_economic_gates(
    gates: list[PromotionGateResult],
    evidence: RouterHypothesisEvidenceBundle,
    months: pd.DataFrame,
    products: pd.DataFrame,
    periods: pd.DataFrame,
    perturbations: pd.DataFrame,
    comparator: str,
    policy: RouterPromotionPolicy,
) -> None:
    validation_label = evidence.config.validation_label
    holdout_label = evidence.config.holdout_label
    validation_months = months.loc[months["research_split"].eq(validation_label)]
    holdout_months = months.loc[months["research_split"].eq(holdout_label)]
    _sufficiency_gate(
        gates,
        "validation_month_count",
        "sample",
        len(validation_months),
        policy.minimum_validation_months,
        "The chronological validation sample needs enough monthly comparisons.",
    )
    _sufficiency_gate(
        gates,
        "holdout_month_count",
        "sample",
        len(holdout_months),
        policy.minimum_holdout_months,
        "The frozen holdout needs enough monthly comparisons.",
    )
    _positive_gate(
        gates,
        "validation_net_increment",
        "incremental_economics",
        _mean(validation_months["net_increment"]),
        "The router must beat the strongest validation-selected attainable baseline after costs.",
    )
    _positive_gate(
        gates,
        "holdout_net_increment",
        "incremental_economics",
        _mean(holdout_months["net_increment"]),
        "The same frozen comparator and router must retain a positive holdout increment.",
    )

    criterion_passed = bool(evidence.criterion_result.passed)
    _gate(
        gates,
        "phase6_validation_criterion",
        "incremental_economics",
        criterion_passed,
        criterion_passed,
        "==",
        True,
        "economic",
        "The router must pass its predeclared Phase 6 validation criterion.",
    )

    top_month_share = _max_or_nan(validation_months["positive_increment_share"])
    _maximum_gate(
        gates,
        "top_month_positive_increment_share",
        "concentration",
        top_month_share,
        policy.maximum_top_month_positive_increment_share,
        "No single validation month may supply an excessive share of positive increment.",
    )
    leave_month = _leave_top_out_mean(validation_months, "net_increment")
    _positive_gate(
        gates,
        "leave_best_month_out_increment",
        "concentration",
        leave_month,
        "The validation increment must remain positive after removing its best month.",
    )

    validation_products = products.loc[
        products["research_split"].eq(validation_label)
    ]
    top_product_share = _max_or_nan(
        validation_products["positive_increment_share"]
    )
    _maximum_gate(
        gates,
        "top_product_positive_increment_share",
        "concentration",
        top_product_share,
        policy.maximum_top_product_positive_increment_share,
        "No single product may supply an excessive share of positive validation increment.",
    )
    leave_product = _leave_top_out_sum(validation_products, "net_increment")
    _positive_gate(
        gates,
        "leave_best_product_out_increment",
        "concentration",
        leave_product,
        "The validation increment must remain positive after removing its best product.",
    )

    validation_periods = periods.loc[
        periods["research_split"].eq(validation_label)
    ]
    _sufficiency_gate(
        gates,
        "validation_subperiod_count",
        "stability",
        len(validation_periods),
        policy.minimum_validation_subperiods,
        "At least two chronological validation subperiods are required.",
    )
    period_metric = "router_minus_best_alternative_annualized_mean"
    positive_period_fraction = float(
        pd.to_numeric(validation_periods[period_metric], errors="coerce")
        .dropna()
        .gt(0.0)
        .mean()
    ) if len(validation_periods) else math.nan
    if len(validation_periods) < policy.minimum_validation_subperiods:
        _insufficient_metric_gate(
            gates,
            "positive_validation_subperiod_fraction",
            "stability",
            positive_period_fraction,
            policy.required_positive_validation_subperiod_fraction,
            "The router's incremental sign needs enough validation subperiods.",
        )
    else:
        _minimum_gate(
            gates,
            "positive_validation_subperiod_fraction",
            "stability",
            positive_period_fraction,
            policy.required_positive_validation_subperiod_fraction,
            "The router's incremental sign must persist across validation subperiods.",
        )

    decision_log = evidence.decision_log.loc[
        evidence.decision_log[policy.split_col].eq(validation_label)
    ]
    switch_count = int(decision_log["switch"].fillna(False).sum())
    _sufficiency_gate(
        gates,
        "validation_switch_count",
        "routing_events",
        switch_count,
        policy.minimum_validation_switches,
        "The router needs enough actual allocation switches to evaluate switching economics.",
    )
    selected_counts = decision_log["selected_sleeve"].value_counts()
    minimum_selection_count = int(
        min(selected_counts.get("sleeve_a", 0), selected_counts.get("sleeve_b", 0))
    )
    _sufficiency_gate(
        gates,
        "minimum_selections_per_sleeve",
        "routing_events",
        minimum_selection_count,
        policy.minimum_selections_per_sleeve,
        "Both sleeves must receive enough selections; an almost-static router is not informative.",
    )

    candidate_perturbations = perturbations.loc[
        ~perturbations["is_baseline"].fillna(False).astype(bool)
    ]
    _sufficiency_gate(
        gates,
        "parameter_perturbation_count",
        "robustness",
        len(candidate_perturbations),
        policy.minimum_perturbation_count,
        "The frozen neighbourhood must contain enough reasonable parameter perturbations.",
    )
    positive_perturbation_fraction = float(
        candidate_perturbations["validation_net_increment"].gt(0.0).mean()
    ) if len(candidate_perturbations) else math.nan
    if len(candidate_perturbations) < policy.minimum_perturbation_count:
        _insufficient_metric_gate(
            gates,
            "positive_parameter_perturbation_fraction",
            "robustness",
            positive_perturbation_fraction,
            policy.minimum_positive_perturbation_fraction,
            "The perturbation pass fraction needs the frozen minimum number of variants.",
        )
    else:
        _minimum_gate(
            gates,
            "positive_parameter_perturbation_fraction",
            "robustness",
            positive_perturbation_fraction,
            policy.minimum_positive_perturbation_fraction,
            "Nearby parameter choices should preserve positive validation increment.",
        )
    reproducible_perturbations = bool(
        not len(candidate_perturbations)
        or candidate_perturbations["reproducible"].astype(bool).all()
    )
    _gate(
        gates,
        "perturbations_reproducible",
        "robustness",
        reproducible_perturbations,
        reproducible_perturbations,
        "==",
        True,
        "governance",
        "Every perturbation result must be reproducible from its declared configuration.",
    )

    gross_benefit, incremental_cost, ratio = _switching_economics(
        evidence.daily_comparison, comparator, validation_label, policy.date_col
    )
    _positive_gate(
        gates,
        "gross_routing_selection_benefit",
        "switching_economics",
        gross_benefit,
        "Routing must add gross selection value before receiving credit for costs.",
    )
    _minimum_gate(
        gates,
        "switching_benefit_cost_ratio",
        "switching_economics",
        ratio,
        policy.minimum_switching_benefit_cost_ratio,
        "Gross routing benefit must exceed the incremental cost of switching.",
    )
    if math.isfinite(incremental_cost):
        gates[-1] = PromotionGateResult(
            **{
                **gates[-1].to_dict(),
                "explanation": (
                    f"Gross routing benefit must exceed incremental switching cost; "
                    f"annualized incremental cost was {incremental_cost:.6f}."
                ),
            }
        )


def _append_paper_trading_gates(
    gates: list[PromotionGateResult],
    paper: PaperTradingEvidence,
    evidence: RouterHypothesisEvidenceBundle,
    config: RouterPromotionReviewConfig,
    policy: RouterPromotionPolicy,
) -> None:
    _gate(
        gates,
        "paper_router_identity",
        "paper_trading",
        paper.router_id == config.router_id,
        paper.router_id,
        "==",
        config.router_id,
        "governance",
        "Paper trading must run the reviewed router.",
    )
    _gate(
        gates,
        "paper_config_fingerprint",
        "paper_trading",
        paper.router_config_fingerprint == config.router_config_fingerprint,
        paper.router_config_fingerprint,
        "==",
        config.router_config_fingerprint,
        "governance",
        "Paper trading must use the frozen router configuration.",
    )
    holdout_end = pd.to_datetime(
        evidence.daily_comparison.loc[
            evidence.daily_comparison[policy.split_col].eq(
                evidence.config.holdout_label
            ),
            policy.date_col,
        ],
        errors="coerce",
    ).max()
    paper_after_holdout = bool(
        pd.notna(holdout_end)
        and pd.Timestamp(paper.start_date) > pd.Timestamp(holdout_end).normalize()
    )
    _gate(
        gates,
        "paper_starts_after_holdout",
        "paper_trading",
        paper_after_holdout,
        paper.start_date,
        ">",
        str(pd.Timestamp(holdout_end).date()) if pd.notna(holdout_end) else "missing",
        "governance",
        "Paper trading must begin strictly after the frozen holdout ends.",
    )
    _sufficiency_gate(
        gates,
        "paper_observation_count",
        "paper_trading",
        paper.observation_count,
        policy.minimum_paper_observations,
        "Paper trading needs enough out-of-research observations.",
    )
    _sufficiency_gate(
        gates,
        "paper_switch_count",
        "paper_trading",
        paper.switch_count,
        policy.minimum_paper_switches,
        "Paper trading must observe enough real router switches.",
    )
    _positive_gate(
        gates,
        "paper_net_increment",
        "paper_trading",
        paper.router_net_return - paper.comparator_net_return,
        "The paper router must beat its frozen comparator after observed costs.",
    )
    _sufficiency_gate(
        gates,
        "paper_reproducible_runs",
        "paper_trading",
        paper.reproducible_run_count,
        2,
        "At least two independent paper snapshots must reproduce the same configuration.",
        failure_kind="governance",
    )
    cost_ratio = (
        paper.realized_cost_return / paper.modeled_cost_return
        if paper.modeled_cost_return > 0.0
        else (0.0 if paper.realized_cost_return <= 0.0 else math.inf)
    )
    _maximum_gate(
        gates,
        "paper_realized_to_modeled_cost_ratio",
        "paper_trading",
        cost_ratio,
        policy.maximum_realized_to_modeled_cost_ratio,
        "Observed paper costs must remain reasonably close to the frozen cost model.",
    )


def _month_concentration(
    daily: pd.DataFrame,
    comparator: str,
    policy: RouterPromotionPolicy,
    validation_label: str,
    holdout_label: str,
) -> pd.DataFrame:
    required = {
        policy.date_col,
        policy.split_col,
        "strategy_id",
        "net_return",
        "gross_return",
        "cost_return",
    }
    _require_columns(daily, required, "daily comparison")
    rows: list[dict[str, Any]] = []
    for split in (validation_label, holdout_label):
        sample = daily.loc[daily[policy.split_col].eq(split)]
        router = sample.loc[sample["strategy_id"].eq("router")]
        baseline = sample.loc[sample["strategy_id"].eq(comparator)]
        merged = router[
            [policy.date_col, "net_return"]
        ].merge(
            baseline[[policy.date_col, "net_return"]],
            on=policy.date_col,
            how="inner",
            suffixes=("_router", "_comparator"),
            validate="one_to_one",
        )
        merged["month"] = pd.to_datetime(merged[policy.date_col]).dt.to_period("M")
        for month, month_sample in merged.groupby("month", sort=True):
            router_return = _compound(month_sample["net_return_router"])
            comparator_return = _compound(month_sample["net_return_comparator"])
            rows.append(
                {
                    "research_split": split,
                    "month": str(month),
                    "router_net_return": router_return,
                    "comparator_net_return": comparator_return,
                    "net_increment": router_return - comparator_return,
                }
            )
    out = pd.DataFrame(rows)
    out["positive_increment"] = out["net_increment"].clip(lower=0.0)
    denominator = out.groupby("research_split")["positive_increment"].transform("sum")
    out["positive_increment_share"] = np.where(
        denominator.gt(0.0), out["positive_increment"] / denominator, np.nan
    )
    return out


def _product_concentration(
    positions: pd.DataFrame,
    comparator: str,
    policy: RouterPromotionPolicy,
    validation_label: str,
    holdout_label: str,
) -> pd.DataFrame:
    required = {
        policy.product_col,
        policy.split_col,
        "strategy_id",
        "net_contribution",
    }
    _require_columns(positions, required, "execution positions")
    sample = positions.loc[
        positions[policy.split_col].isin([validation_label, holdout_label])
        & positions["strategy_id"].isin(["router", comparator])
    ].copy()
    grouped = (
        sample.groupby(
            [policy.split_col, "strategy_id", policy.product_col],
            as_index=False,
            sort=True,
        )["net_contribution"]
        .sum()
    )
    router = grouped.loc[grouped["strategy_id"].eq("router")].rename(
        columns={"net_contribution": "router_net_contribution"}
    )
    baseline = grouped.loc[grouped["strategy_id"].eq(comparator)].rename(
        columns={"net_contribution": "comparator_net_contribution"}
    )
    keys = [policy.split_col, policy.product_col]
    out = router[keys + ["router_net_contribution"]].merge(
        baseline[keys + ["comparator_net_contribution"]],
        on=keys,
        how="outer",
    ).fillna(0.0)
    out["net_increment"] = (
        out["router_net_contribution"] - out["comparator_net_contribution"]
    )
    out["positive_increment"] = out["net_increment"].clip(lower=0.0)
    denominator = out.groupby(policy.split_col)["positive_increment"].transform("sum")
    out["positive_increment_share"] = np.where(
        denominator.gt(0.0), out["positive_increment"] / denominator, np.nan
    )
    return out.rename(columns={policy.split_col: "research_split"})


def _prepare_perturbations(
    perturbations: pd.DataFrame,
    expected_plan_fingerprint: str,
) -> pd.DataFrame:
    required = {
        "perturbation_id",
        "plan_fingerprint",
        "validation_net_increment",
        "reproducible",
    }
    _require_columns(perturbations, required, "parameter perturbations")
    out = perturbations.copy()
    if out["perturbation_id"].astype(str).duplicated().any():
        raise ValueError("perturbation IDs must be unique")
    out["validation_net_increment"] = pd.to_numeric(
        out["validation_net_increment"], errors="raise"
    )
    if "is_baseline" not in out:
        out["is_baseline"] = False
    out["plan_matches_review"] = out["plan_fingerprint"].astype(str).eq(
        expected_plan_fingerprint
    )
    return out.sort_values("perturbation_id").reset_index(drop=True)


def _switching_economics(
    daily: pd.DataFrame,
    comparator: str,
    validation_label: str,
    date_col: str,
) -> tuple[float, float, float]:
    sample = daily.loc[daily["research_split"].eq(validation_label)]
    router = sample.loc[sample["strategy_id"].eq("router")]
    baseline = sample.loc[sample["strategy_id"].eq(comparator)]
    merged = router[[date_col, "gross_return", "cost_return"]].merge(
        baseline[[date_col, "gross_return", "cost_return"]],
        on=date_col,
        how="inner",
        suffixes=("_router", "_comparator"),
        validate="one_to_one",
    )
    gross_benefit = _mean(
        merged["gross_return_router"] - merged["gross_return_comparator"]
    ) * ANNUALIZATION_DAYS
    incremental_cost = _mean(
        merged["cost_return_router"] - merged["cost_return_comparator"]
    ) * ANNUALIZATION_DAYS
    if gross_benefit <= 0.0:
        ratio = -math.inf
    elif incremental_cost <= 0.0:
        ratio = math.inf
    else:
        ratio = gross_benefit / incremental_cost
    return gross_benefit, incremental_cost, ratio


def _promotion_decision(
    gates: list[PromotionGateResult], paper_supplied: bool
) -> PromotionDecision:
    failed = [gate for gate in gates if gate.status is not PromotionGateStatus.PASS]
    if any(gate.failure_kind == "governance" for gate in failed):
        return PromotionDecision.BLOCKED_GOVERNANCE
    if any(gate.status is PromotionGateStatus.INSUFFICIENT for gate in failed):
        return PromotionDecision.HOLD_FOR_MORE_EVIDENCE
    if any(
        gate.failure_kind == "economic"
        and gate.status is PromotionGateStatus.FAIL
        for gate in failed
    ):
        return PromotionDecision.FAILED_RESEARCH_RESULT
    if failed:
        return PromotionDecision.FAILED_RESEARCH_RESULT
    if paper_supplied:
        return PromotionDecision.ELIGIBLE_FOR_PRODUCTION_REVIEW
    return PromotionDecision.ELIGIBLE_FOR_PAPER_TRADING


def _decision_explanation(decision: PromotionDecision) -> str:
    return {
        PromotionDecision.ELIGIBLE_FOR_PAPER_TRADING: (
            "Chronological validation and the frozen holdout passed every "
            "promotion gate. The router may enter paper trading, not production."
        ),
        PromotionDecision.ELIGIBLE_FOR_PRODUCTION_REVIEW: (
            "Frozen research and paper-trading gates passed. Human production "
            "review is required before any deployment decision."
        ),
        PromotionDecision.HOLD_FOR_MORE_EVIDENCE: (
            "The current evidence is insufficient for promotion; the router is held "
            "without interpreting missing observations as economic failure."
        ),
        PromotionDecision.BLOCKED_GOVERNANCE: (
            "A frozen fingerprint, causal lineage, or reproducibility requirement "
            "failed. Economic promotion cannot be evaluated safely."
        ),
        PromotionDecision.FAILED_RESEARCH_RESULT: (
            "A sufficiently observed economic promotion gate failed. The negative "
            "result is retained as a valid research outcome."
        ),
    }[decision]


def _gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    passed: bool,
    observed: Any,
    operator: str,
    threshold: Any,
    failure_kind: str,
    explanation: str,
) -> None:
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=(
                PromotionGateStatus.PASS if passed else PromotionGateStatus.FAIL
            ),
            observed=observed,
            operator=operator,
            threshold=threshold,
            failure_kind=failure_kind,
            explanation=explanation,
        )
    )


def _sufficiency_gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    observed: int,
    threshold: int,
    explanation: str,
    *,
    failure_kind: str = "evidence",
) -> None:
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=(
                PromotionGateStatus.PASS
                if int(observed) >= int(threshold)
                else PromotionGateStatus.INSUFFICIENT
            ),
            observed=int(observed),
            operator=">=",
            threshold=int(threshold),
            failure_kind=failure_kind,
            explanation=explanation,
        )
    )


def _positive_gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    observed: float,
    explanation: str,
) -> None:
    finite = math.isfinite(float(observed))
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=(
                PromotionGateStatus.PASS
                if finite and float(observed) > 0.0
                else PromotionGateStatus.FAIL
            ),
            observed=float(observed),
            operator=">",
            threshold=0.0,
            failure_kind="economic",
            explanation=explanation,
        )
    )


def _minimum_gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    observed: float,
    threshold: float,
    explanation: str,
) -> None:
    finite = math.isfinite(float(observed))
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=(
                PromotionGateStatus.PASS
                if finite and float(observed) >= float(threshold)
                else PromotionGateStatus.FAIL
            ),
            observed=float(observed),
            operator=">=",
            threshold=float(threshold),
            failure_kind="economic",
            explanation=explanation,
        )
    )


def _insufficient_metric_gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    observed: float,
    threshold: float,
    explanation: str,
) -> None:
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=PromotionGateStatus.INSUFFICIENT,
            observed=float(observed),
            operator=">=",
            threshold=float(threshold),
            failure_kind="evidence",
            explanation=explanation,
        )
    )


def _maximum_gate(
    gates: list[PromotionGateResult],
    gate_id: str,
    category: str,
    observed: float,
    threshold: float,
    explanation: str,
) -> None:
    finite = math.isfinite(float(observed))
    gates.append(
        PromotionGateResult(
            gate_id=gate_id,
            category=category,
            status=(
                PromotionGateStatus.PASS
                if finite and float(observed) <= float(threshold)
                else PromotionGateStatus.FAIL
            ),
            observed=float(observed),
            operator="<=",
            threshold=float(threshold),
            failure_kind="economic",
            explanation=explanation,
        )
    )


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    digest.update(json.dumps([str(value) for value in frame.dtypes]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _leave_top_out_mean(frame: pd.DataFrame, metric: str) -> float:
    values = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(values.drop(index=values.idxmax()).mean())


def _leave_top_out_sum(frame: pd.DataFrame, metric: str) -> float:
    values = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(values.drop(index=values.idxmax()).sum())


def _compound(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float((1.0 + clean).prod() - 1.0) if len(clean) else math.nan


def _mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def _max_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.max()) if len(clean) else math.nan


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


__all__ = [
    "build_router_promotion_review",
    "router_evidence_fingerprint",
    "write_router_promotion_review",
]
