"""Causal routing of frozen strategy sleeves."""

from oqp.research.strategy_routing.contracts import (
    RouterContract,
    resolve_router_contract,
)
from oqp.research.strategy_routing.engine import (
    RoutedSleeveResult,
    build_discrete_state_allocations,
    route_sleeve_targets,
    validate_router_allocations,
)
from oqp.research.strategy_routing.evidence import (
    ATTAINABLE_STRATEGIES,
    ORACLE_STRATEGY,
    ROUTER_HYPOTHESIS_SCHEMA_VERSION,
    RouterHypothesisConfig,
    RouterHypothesisEvidenceBundle,
    audit_router_readiness,
    build_router_hypothesis_evidence,
    load_router_hypothesis_evidence_bundle,
    write_router_hypothesis_evidence_bundle,
    write_router_readiness_snapshot,
)
from oqp.research.strategy_routing.registry import (
    PRIVATE_ROUTER_ROOT,
    iter_router_files,
    load_router_module,
    resolve_router_path,
)

__all__ = [
    "ATTAINABLE_STRATEGIES",
    "ORACLE_STRATEGY",
    "PRIVATE_ROUTER_ROOT",
    "ROUTER_HYPOTHESIS_SCHEMA_VERSION",
    "RoutedSleeveResult",
    "RouterContract",
    "RouterHypothesisConfig",
    "RouterHypothesisEvidenceBundle",
    "audit_router_readiness",
    "build_discrete_state_allocations",
    "build_router_hypothesis_evidence",
    "iter_router_files",
    "load_router_module",
    "load_router_hypothesis_evidence_bundle",
    "resolve_router_contract",
    "resolve_router_path",
    "route_sleeve_targets",
    "validate_router_allocations",
    "write_router_hypothesis_evidence_bundle",
    "write_router_readiness_snapshot",
]
