from __future__ import annotations

from oqp.research.factor_deduplication import (
    FactorQualityProfile,
    is_near_duplicate_edge,
    select_cluster_representative,
)


def _profile(factor_id: str, **overrides) -> FactorQualityProfile:
    values = {
        "factor_id": factor_id,
        "predictiveness": 0.020,
        "icir": 0.50,
        "positive_period_share": 0.58,
        "coverage": 0.90,
        "simplicity": 0.80,
    }
    values.update(overrides)
    return FactorQualityProfile(**values)


def test_duplicate_edge_requires_every_condition() -> None:
    assert is_near_duplicate_edge(
        common_observations=12_000,
        score_spearman=-0.85,
        target_pearson=-0.93,
        active_overlap=0.70,
    )
    assert not is_near_duplicate_edge(
        common_observations=12_000,
        score_spearman=0.85,
        target_pearson=0.93,
        active_overlap=0.40,
    )


def test_clear_epsilon_dominator_becomes_representative() -> None:
    result = select_cluster_representative(
        [
            _profile("fac_simple"),
            _profile(
                "fac_weaker",
                predictiveness=0.015,
                icir=0.30,
                coverage=0.80,
                simplicity=0.40,
            ),
        ]
    )

    assert result.representative == "fac_simple"
    assert result.dominated_factors == ("fac_weaker",)


def test_conflicting_quality_tradeoffs_require_manual_review() -> None:
    result = select_cluster_representative(
        [
            _profile(
                "fac_predictive",
                predictiveness=0.030,
                icir=0.80,
                coverage=0.65,
                simplicity=0.30,
            ),
            _profile(
                "fac_simple",
                predictiveness=0.020,
                icir=0.50,
                coverage=0.95,
                simplicity=0.95,
            ),
        ]
    )

    assert result.decision == "manual_review_multiple_pareto_survivors"
    assert result.representative is None
    assert set(result.pareto_frontier) == {"fac_predictive", "fac_simple"}


def test_tiny_predictive_difference_does_not_defeat_simplicity() -> None:
    result = select_cluster_representative(
        [
            _profile(
                "fac_complex",
                predictiveness=0.021,
                simplicity=0.20,
            ),
            _profile(
                "fac_simple",
                predictiveness=0.020,
                simplicity=0.90,
            ),
        ]
    )

    assert result.representative == "fac_simple"
    assert result.dominated_factors == ("fac_complex",)
