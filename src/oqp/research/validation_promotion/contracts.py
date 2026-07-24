"""Frozen contracts for Phase 10 validation and promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


PHASE10_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROMOTION_POLICY_REGISTRY = REPO_ROOT / "config/research/promotion_policy.yaml"


class PromotionDecision(str, Enum):
    ELIGIBLE_FOR_PAPER_TRADING = "eligible_for_paper_trading"
    ELIGIBLE_FOR_PRODUCTION_REVIEW = "eligible_for_production_review"
    HOLD_FOR_MORE_EVIDENCE = "hold_for_more_evidence"
    BLOCKED_GOVERNANCE = "blocked_governance"
    FAILED_RESEARCH_RESULT = "failed_research_result"


class PromotionGateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class RouterPromotionPolicy:
    profile_id: str
    research_object: str
    minimum_validation_months: int
    minimum_holdout_months: int
    minimum_validation_switches: int
    minimum_selections_per_sleeve: int
    minimum_validation_subperiods: int
    required_positive_validation_subperiod_fraction: float
    maximum_top_month_positive_increment_share: float
    maximum_top_product_positive_increment_share: float
    minimum_perturbation_count: int
    minimum_positive_perturbation_fraction: float
    minimum_switching_benefit_cost_ratio: float
    minimum_paper_observations: int
    minimum_paper_switches: int
    maximum_realized_to_modeled_cost_ratio: float
    date_col: str = "date"
    product_col: str = "ticker"
    split_col: str = "research_split"
    status: str = "active"

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id",
            "research_object",
            "date_col",
            "product_col",
            "split_col",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        if self.research_object != "router":
            raise ValueError("Phase 10 version 1 supports router promotion only")
        integer_fields = (
            "minimum_validation_months",
            "minimum_holdout_months",
            "minimum_validation_switches",
            "minimum_selections_per_sleeve",
            "minimum_validation_subperiods",
            "minimum_perturbation_count",
            "minimum_paper_observations",
            "minimum_paper_switches",
        )
        for field_name in integer_fields:
            value = int(getattr(self, field_name))
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        fractions = (
            "required_positive_validation_subperiod_fraction",
            "maximum_top_month_positive_increment_share",
            "maximum_top_product_positive_increment_share",
            "minimum_positive_perturbation_fraction",
        )
        for field_name in fractions:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "minimum_switching_benefit_cost_ratio",
            "maximum_realized_to_modeled_cost_ratio",
        ):
            value = float(getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "status", str(self.status).strip().lower())

    @property
    def fingerprint(self) -> str:
        return stable_promotion_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, profile_id: str, payload: Mapping[str, Any]
    ) -> "RouterPromotionPolicy":
        return cls(profile_id=profile_id, **dict(payload))


class PromotionPolicyRegistry:
    def __init__(
        self,
        profiles: dict[str, RouterPromotionPolicy],
        *,
        registry_id: str,
        schema_version: int,
        source_path: Path,
    ) -> None:
        self.profiles = dict(profiles)
        self.registry_id = str(registry_id)
        self.schema_version = int(schema_version)
        self.source_path = source_path

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_PROMOTION_POLICY_REGISTRY
    ) -> "PromotionPolicyRegistry":
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        raw_profiles = payload.get("profiles") or {}
        if not isinstance(raw_profiles, Mapping) or not raw_profiles:
            raise ValueError("promotion policy registry cannot be empty")
        profiles = {
            str(profile_id): RouterPromotionPolicy.from_mapping(
                str(profile_id), value
            )
            for profile_id, value in raw_profiles.items()
        }
        return cls(
            profiles,
            registry_id=str(payload.get("registry_id") or ""),
            schema_version=int(payload.get("schema_version", 1)),
            source_path=source,
        )

    def resolve(self, profile_id: str) -> RouterPromotionPolicy:
        key = str(profile_id).strip()
        try:
            profile = self.profiles[key]
        except KeyError as exc:
            raise KeyError(f"unknown promotion policy profile: {key}") from exc
        if profile.status != "active":
            raise ValueError(f"promotion policy profile {key} is not active")
        return profile


@dataclass(frozen=True, slots=True)
class RouterPromotionReviewConfig:
    review_id: str
    router_id: str
    router_config_fingerprint: str
    source_evidence_fingerprint: str
    perturbation_plan_fingerprint: str
    reviewed_on: str
    policy_profile_id: str = "router_validation_promotion_v1"
    optimization_candidate_fingerprint: str = ""
    schema_version: int = PHASE10_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "review_id",
            "router_id",
            "router_config_fingerprint",
            "source_evidence_fingerprint",
            "perturbation_plan_fingerprint",
            "reviewed_on",
            "policy_profile_id",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        reviewed = pd.Timestamp(self.reviewed_on).normalize()
        object.__setattr__(self, "reviewed_on", reviewed.date().isoformat())
        object.__setattr__(
            self,
            "optimization_candidate_fingerprint",
            str(self.optimization_candidate_fingerprint).strip(),
        )

    @property
    def fingerprint(self) -> str:
        return stable_promotion_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperTradingEvidence:
    router_id: str
    router_config_fingerprint: str
    start_date: str
    end_date: str
    observation_count: int
    switch_count: int
    router_net_return: float
    comparator_net_return: float
    modeled_cost_return: float
    realized_cost_return: float
    reproducible_run_count: int

    def __post_init__(self) -> None:
        if not str(self.router_id).strip() or not str(
            self.router_config_fingerprint
        ).strip():
            raise ValueError("paper evidence requires router identity and fingerprint")
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        if end < start:
            raise ValueError("paper evidence end_date cannot precede start_date")
        for field_name in ("observation_count", "switch_count", "reproducible_run_count"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "start_date", start.date().isoformat())
        object.__setattr__(self, "end_date", end.date().isoformat())

    @property
    def fingerprint(self) -> str:
        return stable_promotion_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    gate_id: str
    category: str
    status: PromotionGateStatus | str
    observed: float | int | str | bool | None
    operator: str
    threshold: float | int | str | bool | None
    failure_kind: str
    explanation: str

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, PromotionGateStatus)
            else PromotionGateStatus(str(self.status))
        )
        if self.failure_kind not in {"economic", "evidence", "governance"}:
            raise ValueError("unknown promotion failure kind")
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value}


@dataclass(frozen=True, slots=True)
class RouterPromotionReviewBundle:
    config: RouterPromotionReviewConfig
    policy: RouterPromotionPolicy
    decision: PromotionDecision
    current_stage: str
    next_stage: str | None
    summary: dict[str, Any]
    gate_results: pd.DataFrame
    month_concentration: pd.DataFrame
    product_concentration: pd.DataFrame
    validation_periods: pd.DataFrame
    perturbations: pd.DataFrame
    manifest: dict[str, Any]


def stable_promotion_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_PROMOTION_POLICY_REGISTRY",
    "PHASE10_SCHEMA_VERSION",
    "PaperTradingEvidence",
    "PromotionDecision",
    "PromotionGateResult",
    "PromotionGateStatus",
    "PromotionPolicyRegistry",
    "RouterPromotionPolicy",
    "RouterPromotionReviewBundle",
    "RouterPromotionReviewConfig",
    "stable_promotion_hash",
]
