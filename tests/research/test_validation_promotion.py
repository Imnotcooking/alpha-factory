from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from oqp.research.strategy_routing import (
    RouterHypothesisConfig,
    RouterHypothesisEvidenceBundle,
)
from oqp.research.validation_promotion import (
    PaperTradingEvidence,
    PromotionDecision,
    PromotionPolicyRegistry,
    RouterPromotionReviewConfig,
    audit_validation_promotion,
    build_router_promotion_review,
    router_evidence_fingerprint,
    write_router_promotion_review,
    write_validation_promotion_readiness,
)


def _evidence() -> RouterHypothesisEvidenceBundle:
    months = pd.period_range("2021-01", "2024-06", freq="M")
    dates = pd.DatetimeIndex(month.start_time + pd.Timedelta(days=4) for month in months)
    splits = ["validation" if date < pd.Timestamp("2024-01-01") else "holdout" for date in dates]
    config = RouterHypothesisConfig(
        hypothesis_id="rth_phase10_test",
        router_id="rtr_phase10_test",
        economic_claim="The score predicts which frozen sleeve has higher next-period net return.",
        economic_mechanism="The observable condition separates the two sleeve payoff states.",
        market_vertical="FUTURES_CN",
        sleeve_a_factor_id="fac_a",
        sleeve_a_id="slv_a",
        sleeve_b_factor_id="fac_b",
        sleeve_b_id="slv_b",
        score_name="frozen test score",
        score_source_fingerprint="score-v1",
        hypothesis_frozen_on="2020-12-31",
    )
    daily_rows = []
    decision_rows = []
    position_rows = []
    products = ["A", "B", "C", "D"]
    for index, (date, split) in enumerate(zip(dates, splits, strict=True)):
        selected = "sleeve_a" if index % 2 == 0 else "sleeve_b"
        decision_rows.append(
            {
                "date": date,
                "research_split": split,
                "selected_sleeve": selected,
                "switch": index > 0,
            }
        )
        router_net = 0.006 if split == "validation" else 0.005
        strategy_values = {
            "router": (router_net + 0.001, 0.001, router_net),
            "static_blend": (0.0025, 0.0005, 0.002),
            "sleeve_a": (0.0020, 0.0005, 0.0015),
            "sleeve_b": (0.0018, 0.0005, 0.0013),
            "exposure_scaled_blend": (0.0023, 0.0005, 0.0018),
        }
        for strategy_id, (gross, cost, net) in strategy_values.items():
            daily_rows.append(
                {
                    "date": date,
                    "research_split": split,
                    "strategy_id": strategy_id,
                    "gross_return": gross,
                    "cost_return": cost,
                    "net_return": net,
                }
            )
        for product in products:
            position_rows.extend(
                [
                    {
                        "date": date,
                        "research_split": split,
                        "strategy_id": "router",
                        "ticker": product,
                        "net_contribution": router_net / len(products),
                    },
                    {
                        "date": date,
                        "research_split": split,
                        "strategy_id": "static_blend",
                        "ticker": product,
                        "net_contribution": 0.002 / len(products),
                    },
                ]
            )
    subperiods = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024],
            "research_split": ["validation", "validation", "validation", "holdout"],
            "router_minus_best_alternative_annualized_mean": [0.10, 0.08, 0.09, 0.05],
        }
    )
    return RouterHypothesisEvidenceBundle(
        config=config,
        criterion=SimpleNamespace(),
        criterion_result=SimpleNamespace(passed=True),
        summary={
            "best_validation_alternative": "static_blend",
            "router_status": "eligible_for_strategy_review",
        },
        decision_log=pd.DataFrame(decision_rows),
        relative_advantage=pd.DataFrame(),
        execution_positions=pd.DataFrame(position_rows),
        daily_comparison=pd.DataFrame(daily_rows),
        strategy_metrics=pd.DataFrame(),
        routing_metrics=pd.DataFrame(),
        monthly_paired_tests=pd.DataFrame(),
        subperiod_stability=subperiods,
        manifest={
            "config_fingerprint": config.fingerprint,
            "causal_alignment_verified": True,
            "costs_recomputed_after_combining_targets": True,
            "oracle_excluded_from_selection_and_gates": True,
        },
    )


def _perturbations(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation_id": [f"nearby_{index}" for index in range(count)],
            "plan_fingerprint": "perturbation-plan-v1",
            "validation_net_increment": [0.003 - index * 0.0001 for index in range(count)],
            "reproducible": True,
            "is_baseline": False,
        }
    )


def _review_config(evidence: RouterHypothesisEvidenceBundle) -> RouterPromotionReviewConfig:
    return RouterPromotionReviewConfig(
        review_id="prm_phase10_test",
        router_id=evidence.config.router_id,
        router_config_fingerprint=evidence.config.fingerprint,
        source_evidence_fingerprint=router_evidence_fingerprint(evidence),
        perturbation_plan_fingerprint="perturbation-plan-v1",
        reviewed_on="2024-07-01",
    )


def test_policy_is_versioned_and_does_not_use_full_sample_sharpe() -> None:
    profile = PromotionPolicyRegistry.load().resolve(
        "router_validation_promotion_v1"
    )
    assert profile.minimum_validation_months == 24
    assert profile.minimum_holdout_months == 6
    assert "sharpe" not in profile.to_dict()
    assert len(profile.fingerprint) == 64


def test_complete_frozen_evidence_is_eligible_for_paper_trading() -> None:
    evidence = _evidence()
    review = build_router_promotion_review(
        evidence, _perturbations(), _review_config(evidence)
    )
    assert review.decision is PromotionDecision.ELIGIBLE_FOR_PAPER_TRADING
    assert review.next_stage == "paper_trading"
    assert review.summary["full_sample_sharpe_used_for_promotion"] is False
    assert review.gate_results["status"].eq("pass").all()


def test_too_few_parameter_neighbours_holds_without_declaring_failure() -> None:
    evidence = _evidence()
    review = build_router_promotion_review(
        evidence, _perturbations(2), _review_config(evidence)
    )
    assert review.decision is PromotionDecision.HOLD_FOR_MORE_EVIDENCE
    assert review.summary["failure_is_valid_research_result"] is False
    assert "parameter_perturbation_count" in review.summary["failed_gate_ids"]


def test_one_month_dependency_is_recorded_as_failed_research_result() -> None:
    evidence = _evidence()
    changed = evidence.daily_comparison.copy()
    first_validation_date = changed.loc[
        changed["research_split"].eq("validation"), "date"
    ].min()
    changed.loc[
        changed["date"].eq(first_validation_date)
        & changed["strategy_id"].eq("router"),
        ["gross_return", "net_return"],
    ] = [0.80, 0.799]
    concentrated = replace(evidence, daily_comparison=changed)
    review = build_router_promotion_review(
        concentrated, _perturbations(), _review_config(concentrated)
    )
    assert review.decision is PromotionDecision.FAILED_RESEARCH_RESULT
    assert review.summary["failure_is_valid_research_result"] is True
    assert "top_month_positive_increment_share" in review.summary["failed_gate_ids"]


def test_stale_evidence_fingerprint_blocks_governance() -> None:
    evidence = _evidence()
    config = replace(_review_config(evidence), source_evidence_fingerprint="stale")
    review = build_router_promotion_review(evidence, _perturbations(), config)
    assert review.decision is PromotionDecision.BLOCKED_GOVERNANCE
    assert "source_evidence_frozen" in review.summary["failed_gate_ids"]


def test_review_before_holdout_completion_is_governance_blocked() -> None:
    evidence = _evidence()
    config = replace(_review_config(evidence), reviewed_on="2024-03-01")
    review = build_router_promotion_review(evidence, _perturbations(), config)
    assert review.decision is PromotionDecision.BLOCKED_GOVERNANCE
    assert "review_after_frozen_holdout" in review.summary["failed_gate_ids"]


def test_paper_evidence_can_advance_only_to_production_review() -> None:
    evidence = _evidence()
    paper = PaperTradingEvidence(
        router_id=evidence.config.router_id,
        router_config_fingerprint=evidence.config.fingerprint,
        start_date="2024-07-01",
        end_date="2024-10-31",
        observation_count=88,
        switch_count=9,
        router_net_return=0.06,
        comparator_net_return=0.03,
        modeled_cost_return=0.01,
        realized_cost_return=0.012,
        reproducible_run_count=3,
    )
    review = build_router_promotion_review(
        evidence,
        _perturbations(),
        _review_config(evidence),
        paper_evidence=paper,
    )
    assert review.decision is PromotionDecision.ELIGIBLE_FOR_PRODUCTION_REVIEW
    assert review.next_stage == "production_review"


def test_failed_review_is_written_and_retained_in_readiness_ledger(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    changed = evidence.subperiod_stability.copy()
    changed.loc[changed["year"].eq(2022), "router_minus_best_alternative_annualized_mean"] = -0.02
    unstable = replace(evidence, subperiod_stability=changed)
    review = build_router_promotion_review(
        unstable, _perturbations(), _review_config(unstable)
    )
    write_router_promotion_review(review, tmp_path / review.config.review_id)
    summary, ledger, policies = audit_validation_promotion(tmp_path)
    assert summary["failed_research_result_count"] == 1
    assert ledger.iloc[0]["decision"] == "failed_research_result"
    destination = write_validation_promotion_readiness(
        summary, ledger, policies, tmp_path
    )
    assert (destination / "promotion_ledger.csv").exists()
