"""Shared contracts that connect research, paper trading, and money views."""

from importlib import import_module
from typing import Any

from oqp.contracts.artifact_io import (
    LoadedStrategyCandidate,
    StrategyCandidateArtifactError,
    StrategyCandidateArtifactIssue,
    StrategyCandidateLoadResult,
    load_strategy_candidate_artifacts,
    parse_strategy_candidate,
    strategy_candidate_directory,
    strategy_candidate_to_dict,
    write_strategy_candidate_artifact,
)
from oqp.contracts.market_vertical import MarketVertical, normalize_market_vertical
from oqp.contracts.strategy_candidate import (
    CandidateIntakeState,
    CandidateMetrics,
    CandidateSafetyLimits,
    CandidateStatus,
    StrategyCandidate,
)

_ALPHA_EXPORTS = {
    "AlphaCandidateExportError",
    "candidate_from_backtest_row",
    "load_latest_candidate_from_research_db",
    "write_candidate_from_research_db",
}


def __getattr__(name: str) -> Any:
    if name in _ALPHA_EXPORTS:
        module = import_module("oqp.contracts.alpha_export")
        return getattr(module, name)
    raise AttributeError(f"module 'oqp.contracts' has no attribute {name!r}")


__all__ = [
    "AlphaCandidateExportError",
    "CandidateIntakeState",
    "CandidateMetrics",
    "CandidateSafetyLimits",
    "CandidateStatus",
    "LoadedStrategyCandidate",
    "MarketVertical",
    "StrategyCandidate",
    "StrategyCandidateArtifactError",
    "StrategyCandidateArtifactIssue",
    "StrategyCandidateLoadResult",
    "candidate_from_backtest_row",
    "load_latest_candidate_from_research_db",
    "load_strategy_candidate_artifacts",
    "normalize_market_vertical",
    "parse_strategy_candidate",
    "strategy_candidate_directory",
    "strategy_candidate_to_dict",
    "write_candidate_from_research_db",
    "write_strategy_candidate_artifact",
]
