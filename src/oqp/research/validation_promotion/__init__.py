"""Phase 10 validation and promotion gates."""

from oqp.research.validation_promotion.contracts import (
    DEFAULT_PROMOTION_POLICY_REGISTRY,
    PHASE10_SCHEMA_VERSION,
    PaperTradingEvidence,
    PromotionDecision,
    PromotionGateResult,
    PromotionGateStatus,
    PromotionPolicyRegistry,
    RouterPromotionPolicy,
    RouterPromotionReviewBundle,
    RouterPromotionReviewConfig,
)
from oqp.research.validation_promotion.evaluation import (
    build_router_promotion_review,
    router_evidence_fingerprint,
    write_router_promotion_review,
)
from oqp.research.validation_promotion.readiness import (
    audit_validation_promotion,
    write_validation_promotion_readiness,
)

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
    "build_router_promotion_review",
    "audit_validation_promotion",
    "router_evidence_fingerprint",
    "write_router_promotion_review",
    "write_validation_promotion_readiness",
]
