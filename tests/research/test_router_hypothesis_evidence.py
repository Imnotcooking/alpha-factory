from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oqp.research.sleeves import SleeveConstructionConfig, SleeveEvidenceBundle
from oqp.research.strategy_routing import (
    ORACLE_STRATEGY,
    RouterHypothesisConfig,
    audit_router_readiness,
    build_router_hypothesis_evidence,
    load_router_hypothesis_evidence_bundle,
    write_router_hypothesis_evidence_bundle,
)


def _fixture() -> tuple[
    SleeveEvidenceBundle,
    SimpleNamespace,
    SleeveEvidenceBundle,
    SimpleNamespace,
    pd.DataFrame,
    RouterHypothesisConfig,
]:
    months = pd.period_range("2021-01", "2024-06", freq="M")
    dates: list[pd.Timestamp] = []
    scores: list[float] = []
    pattern = (-1.0, -0.35, 0.40, 0.90)
    for month_index, month in enumerate(months):
        amplitude = 1.0 + 0.20 * np.sin(month_index / 3.0)
        for day, value in zip((3, 10, 17, 24), pattern, strict=True):
            dates.append(pd.Timestamp(month.start_time.year, month.start_time.month, day))
            scores.append(float(value * amplitude))
    date_index = pd.DatetimeIndex(dates)
    split = np.where(date_index < pd.Timestamp("2024-01-01"), "validation", "holdout")
    forward = np.asarray(scores) * 0.01
    common = pd.DataFrame(
        {
            "date": date_index,
            "ticker": "TEST",
            "sector": "Synthetic",
            "research_split": split,
            "forward_return": forward,
            "next_symbol": "TEST2501",
            "next_actual_open": 100.0,
            "next_multiplier": 10.0,
            "next_tick_size": 1.0,
            "next_fee_type": "fixed",
            "next_fee_open": 1.0,
            "next_fee_close_today": 1.0,
        }
    )
    execution = {
        "capital": 1_000_000.0,
        "capital_currency": "CNY",
        "integer_contracts": True,
        "slippage_ticks_per_side": 0.5,
        "return_assumption": "close_signal_next_open_to_close",
        "entry": "next_actual_open",
        "exit": "next_actual_close",
        "fee_fields": ["next_fee_type", "next_fee_open", "next_fee_close_today"],
    }

    def phase3(factor_id: str, sleeve_id: str, target: float) -> SleeveEvidenceBundle:
        config = SleeveConstructionConfig(
            sleeve_id=sleeve_id,
            factor_id=factor_id,
            market_vertical="FUTURES_CN",
            max_weight_per_contract=None,
        )
        positions = common.assign(target_weight=target)
        return SleeveEvidenceBundle(
            config=config,
            summary={},
            positions=positions,
            daily_returns=pd.DataFrame(),
            split_summary=pd.DataFrame(),
            yearly_summary=pd.DataFrame(),
            product_summary=pd.DataFrame(),
            sector_summary=pd.DataFrame(),
            manifest={
                "config_fingerprint": config.fingerprint,
                "input_data_fingerprint": "f" * 64,
                "causal_alignment_verified": True,
                "execution": execution,
            },
        )

    def phase4(bundle: SleeveEvidenceBundle) -> SimpleNamespace:
        return SimpleNamespace(
            summary={
                "factor_id": bundle.config.factor_id,
                "sleeve_id": bundle.config.sleeve_id,
                "router_eligible": True,
                "standalone_status": "eligible_for_router_research",
            },
            manifest={
                "phase3_config_fingerprint": bundle.config.fingerprint,
                "config_fingerprint": f"phase4-{bundle.config.factor_id}",
            },
        )

    a = phase3("fac_test_a", "slv_test_a", 0.40)
    b = phase3("fac_test_b", "slv_test_b", -0.40)
    score_frame = pd.DataFrame({"date": date_index, "router_score": scores})
    config = RouterHypothesisConfig(
        hypothesis_id="rth_test_relative_advantage",
        router_id="rtr_test",
        economic_claim=(
            "A positive observable score predicts that sleeve A will outperform "
            "sleeve B over the next executable session."
        ),
        economic_mechanism="The score identifies which side of the synthetic payoff is active.",
        market_vertical="FUTURES_CN",
        sleeve_a_factor_id=a.config.factor_id,
        sleeve_a_id=a.config.sleeve_id,
        sleeve_b_factor_id=b.config.factor_id,
        sleeve_b_id=b.config.sleeve_id,
        score_name="Synthetic signed condition",
        score_source_fingerprint="s" * 64,
        hypothesis_frozen_on="2023-12-31",
        exposure_lookback_periods=10,
        exposure_min_periods=5,
    )
    return a, phase4(a), b, phase4(b), score_frame, config


def test_router_contract_requires_distinct_sleeves_and_economic_claim() -> None:
    _, _, _, _, _, config = _fixture()
    with pytest.raises(ValueError, match="economic_claim"):
        replace(config, economic_claim="")
    with pytest.raises(ValueError, match="must be distinct"):
        replace(
            config,
            sleeve_b_factor_id=config.sleeve_a_factor_id,
            sleeve_b_id=config.sleeve_a_id,
        )


def test_ineligible_standalone_sleeve_blocks_phase6() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    phase4_b.summary["router_eligible"] = False
    with pytest.raises(ValueError, match="sleeve B did not pass"):
        build_router_hypothesis_evidence(
            a, phase4_a, b, phase4_b, scores, config
        )


def test_hypothesis_must_predate_the_untouched_holdout() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    late = replace(config, hypothesis_frozen_on="2024-01-10")
    with pytest.raises(ValueError, match="frozen before"):
        build_router_hypothesis_evidence(a, phase4_a, b, phase4_b, scores, late)


def test_router_score_predicts_relative_advantage_and_selects_better_sleeve() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    bundle = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    validation = bundle.routing_metrics.set_index("research_split").loc["validation"]
    assert validation["routing_ic"] > 0.99
    assert validation["routing_rank_ic"] > 0.99
    assert validation["better_sleeve_hit_rate"] == pytest.approx(1.0)
    assert validation["mean_selected_advantage_correct_bps"] > 0.0
    assert validation["wrong_dates"] == 0


def test_comparator_costs_are_recomputed_after_target_netting() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    bundle = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    validation = bundle.strategy_metrics.loc[
        bundle.strategy_metrics["research_split"].eq("validation")
    ].set_index("strategy_id")
    assert validation.loc["sleeve_a", "annualized_cost"] > 0.0
    assert validation.loc["sleeve_b", "annualized_cost"] > 0.0
    assert validation.loc["static_blend", "annualized_cost"] == pytest.approx(0.0)
    assert validation.loc["static_blend", "annualized_turnover"] == pytest.approx(0.0)


def test_all_required_comparators_and_unattainable_oracle_are_recorded() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    bundle = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    observed = set(bundle.daily_comparison["strategy_id"])
    assert observed == {
        "sleeve_a",
        "sleeve_b",
        "static_blend",
        "exposure_scaled_blend",
        "router",
        ORACLE_STRATEGY,
    }
    oracle = bundle.strategy_metrics.loc[
        bundle.strategy_metrics["strategy_id"].eq(ORACLE_STRATEGY)
    ]
    assert not oracle["attainable"].any()
    assert bundle.manifest["oracle_excluded_from_selection_and_gates"] is True
    assert bundle.summary["best_validation_alternative"] != ORACLE_STRATEGY


def test_paired_monthly_hac_switching_and_subperiod_stability_are_reported() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    bundle = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    validation = bundle.monthly_paired_tests.loc[
        bundle.monthly_paired_tests["research_split"].eq("validation")
    ]
    assert set(validation["comparator"]) == {
        "sleeve_a",
        "sleeve_b",
        "static_blend",
        "exposure_scaled_blend",
    }
    assert validation["month_count"].min() >= 24
    assert bundle.routing_metrics["switch_count"].max() > 0
    assert bundle.subperiod_stability["year"].nunique() == 4


def test_future_payoff_change_does_not_rewrite_earlier_router_decisions() -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    first = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    changed_positions_a = a.positions.copy()
    changed_positions_b = b.positions.copy()
    cutoff = pd.Timestamp("2024-03-01")
    changed_positions_a.loc[
        changed_positions_a["date"].ge(cutoff), "forward_return"
    ] *= -10.0
    changed_positions_b.loc[
        changed_positions_b["date"].ge(cutoff), "forward_return"
    ] *= -10.0
    changed_a = replace(a, positions=changed_positions_a)
    changed_b = replace(b, positions=changed_positions_b)
    second = build_router_hypothesis_evidence(
        changed_a, phase4_a, changed_b, phase4_b, scores, config
    )
    columns = ["date", "selected_sleeve", "allocation_a", "allocation_b"]
    pd.testing.assert_frame_equal(
        first.decision_log.loc[first.decision_log["date"].lt(cutoff), columns].reset_index(drop=True),
        second.decision_log.loc[second.decision_log["date"].lt(cutoff), columns].reset_index(drop=True),
    )


def test_router_evidence_artifact_roundtrip(tmp_path) -> None:
    a, phase4_a, b, phase4_b, scores, config = _fixture()
    bundle = build_router_hypothesis_evidence(
        a, phase4_a, b, phase4_b, scores, config
    )
    output = write_router_hypothesis_evidence_bundle(bundle, tmp_path / "router")
    restored = load_router_hypothesis_evidence_bundle(output)
    assert restored.config.fingerprint == bundle.config.fingerprint
    assert restored.summary["oracle_role"] == "unattainable_upper_bound_excluded_from_all_gates"
    assert len(restored.decision_log) == len(bundle.decision_log)


def test_readiness_audit_requires_two_eligible_sleeves_and_a_frozen_hypothesis(
    tmp_path,
) -> None:
    for factor, eligible in (("fac_a", True), ("fac_b", False)):
        destination = tmp_path / factor / "FUTURES_CN" / "slv_default"
        destination.mkdir(parents=True)
        (destination / "summary.json").write_text(
            pd.Series(
                {
                    "factor_id": factor,
                    "sleeve_id": "slv_default",
                    "standalone_status": "eligible" if eligible else "blocked",
                    "router_eligible": eligible,
                    "validation": {"net_sharpe": 0.5},
                    "holdout": {"net_sharpe": 0.4},
                }
            ).to_json(),
            encoding="utf-8",
        )
    summary, sleeves = audit_router_readiness(
        tmp_path, frozen_hypothesis_count=0
    )
    assert len(sleeves) == 2
    assert summary["eligible_sleeves"] == 1
    assert summary["ready_for_empirical_router_test"] is False
    assert len(summary["blockers"]) == 2
