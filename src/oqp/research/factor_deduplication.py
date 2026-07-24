"""Transparent near-duplicate grouping and representative selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DuplicateThresholds:
    """Evidence required before two contract-compatible factors form an edge."""

    minimum_common_observations: int = 10_000
    minimum_abs_score_spearman: float = 0.80
    minimum_abs_target_pearson: float = 0.90
    minimum_active_overlap: float = 0.60


@dataclass(frozen=True, slots=True)
class FactorQualityProfile:
    """Common-data evidence used only after factors enter one duplicate cluster."""

    factor_id: str
    predictiveness: float
    icir: float
    positive_period_share: float
    coverage: float
    simplicity: float

    def __post_init__(self) -> None:
        values = (
            self.predictiveness,
            self.icir,
            self.positive_period_share,
            self.coverage,
            self.simplicity,
        )
        if not self.factor_id.strip() or not all(math.isfinite(value) for value in values):
            raise ValueError("factor quality profiles require an ID and finite metrics")
        for name, value in (
            ("positive_period_share", self.positive_period_share),
            ("coverage", self.coverage),
            ("simplicity", self.simplicity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualityTolerances:
    """Differences smaller than these values are treated as research noise."""

    predictiveness: float = 0.002
    icir: float = 0.10
    positive_period_share: float = 0.05
    coverage: float = 0.02
    simplicity: float = 0.00

    def __post_init__(self) -> None:
        if any(value < 0 or not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("quality tolerances must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RepresentativeSelection:
    decision: str
    representative: str | None
    pareto_frontier: tuple[str, ...]
    dominated_factors: tuple[str, ...]
    rationale: str


def is_near_duplicate_edge(
    *,
    common_observations: int,
    score_spearman: float,
    target_pearson: float,
    active_overlap: float,
    thresholds: DuplicateThresholds = DuplicateThresholds(),
) -> bool:
    """Return true only when every frozen redundancy condition is satisfied."""

    values = (score_spearman, target_pearson, active_overlap)
    if not all(math.isfinite(value) for value in values):
        return False
    return bool(
        int(common_observations) >= thresholds.minimum_common_observations
        and abs(score_spearman) >= thresholds.minimum_abs_score_spearman
        and abs(target_pearson) >= thresholds.minimum_abs_target_pearson
        and active_overlap >= thresholds.minimum_active_overlap
    )


def select_cluster_representative(
    profiles: Iterable[FactorQualityProfile],
    *,
    tolerances: QualityTolerances = QualityTolerances(),
) -> RepresentativeSelection:
    """Select only when one factor epsilon-dominates every alternative.

    Higher values are better for every quality field. Differences inside the
    declared tolerances do not let a slightly better backtest defeat a simpler
    or better-covered implementation. Multiple non-dominated factors produce a
    manual-review result instead of an arbitrary weighted score.
    """

    candidates = tuple(profiles)
    if not candidates:
        raise ValueError("representative selection requires at least one factor")
    ids = [profile.factor_id for profile in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("factor quality profile IDs must be unique")
    if len(candidates) == 1:
        return RepresentativeSelection(
            decision="retain_unique",
            representative=candidates[0].factor_id,
            pareto_frontier=(candidates[0].factor_id,),
            dominated_factors=(),
            rationale="No duplicate alternative entered the cluster.",
        )

    dominated: set[str] = set()
    for candidate in candidates:
        for alternative in candidates:
            if candidate.factor_id == alternative.factor_id:
                continue
            if _epsilon_dominates(alternative, candidate, tolerances):
                dominated.add(candidate.factor_id)
                break

    frontier = tuple(sorted(set(ids).difference(dominated)))
    if len(frontier) == 1:
        representative = frontier[0]
        return RepresentativeSelection(
            decision="retain_representative_pending_dependency_review",
            representative=representative,
            pareto_frontier=frontier,
            dominated_factors=tuple(sorted(dominated)),
            rationale=(
                "One implementation is no worse outside the frozen tolerances "
                "and materially better on at least one declared quality dimension."
            ),
        )
    return RepresentativeSelection(
        decision="manual_review_multiple_pareto_survivors",
        representative=None,
        pareto_frontier=frontier,
        dominated_factors=tuple(sorted(dominated)),
        rationale=(
            "The surviving implementations make different quality trade-offs; "
            "no weighted backtest score is allowed to choose between them."
        ),
    )


def _epsilon_dominates(
    left: FactorQualityProfile,
    right: FactorQualityProfile,
    tolerances: QualityTolerances,
) -> bool:
    fields = (
        "predictiveness",
        "icir",
        "positive_period_share",
        "coverage",
        "simplicity",
    )
    no_worse = all(
        getattr(left, field) >= getattr(right, field) - getattr(tolerances, field)
        for field in fields
    )
    materially_better = any(
        getattr(left, field) > getattr(right, field) + getattr(tolerances, field)
        for field in fields
    )
    return no_worse and materially_better


__all__ = [
    "DuplicateThresholds",
    "FactorQualityProfile",
    "QualityTolerances",
    "RepresentativeSelection",
    "is_near_duplicate_edge",
    "select_cluster_representative",
]
