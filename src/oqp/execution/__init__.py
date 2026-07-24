"""Execution engines and order-routing workflows."""

from importlib import import_module
from typing import Any

from oqp.execution.artifacts import (
    LoadedTradeProposal,
    ProposalArtifactError,
    ProposalArtifactIssue,
    ProposalLoadResult,
    load_trade_proposal_artifacts,
    order_intent_to_dict,
    parse_instrument,
    parse_order_intent,
    parse_trade_proposal,
    trade_proposal_directory,
    trade_proposal_to_dict,
    write_trade_proposal_artifact,
)
from oqp.execution.models import OrderIntent, ProposalStatus, TradeProposal
from oqp.execution.transaction_costs import (
    CostUseCase,
    OrderCostEstimate,
    TransactionCostConfigurationError,
    TransactionCostProfile,
    TransactionCostReadinessError,
    TransactionCostRegistry,
    attach_transaction_cost_policy,
    ensure_transaction_cost_policy,
)


_RESEARCH_BRIDGE_EXPORTS = {
    "ResearchSignalBridgeError",
    "ResearchSignalIssue",
    "SignalProposalConfig",
    "SignalProposalResult",
    "build_trade_proposal_from_signal_rows",
    "load_research_signal_rows",
    "order_intent_from_signal_row",
    "signal_from_row",
    "write_research_signal_proposal",
}

_OPTIONS_BRIDGE_EXPORTS = {
    "OptionsProposalBridgeError",
    "OptionsProposalResult",
    "build_option_trade_proposal_from_candidate",
    "write_option_trade_proposal_from_candidate",
}

_GUARDRAIL_EXPORTS = {
    "GuardrailCheck",
    "GuardrailReport",
    "GuardrailSeverity",
    "evaluate_trade_proposal",
}


def __getattr__(name: str) -> Any:
    if name in _GUARDRAIL_EXPORTS:
        module = import_module("oqp.execution.guardrails")
        return getattr(module, name)
    if name in _RESEARCH_BRIDGE_EXPORTS:
        module = import_module("oqp.execution.research_bridge")
        return getattr(module, name)
    if name in _OPTIONS_BRIDGE_EXPORTS:
        module = import_module("oqp.execution.options_bridge")
        return getattr(module, name)
    raise AttributeError(f"module 'oqp.execution' has no attribute {name!r}")

__all__ = [
    "evaluate_trade_proposal",
    "attach_transaction_cost_policy",
    "CostUseCase",
    "ensure_transaction_cost_policy",
    "GuardrailCheck",
    "GuardrailReport",
    "GuardrailSeverity",
    "LoadedTradeProposal",
    "OrderIntent",
    "OptionsProposalBridgeError",
    "OptionsProposalResult",
    "order_intent_to_dict",
    "parse_instrument",
    "parse_order_intent",
    "parse_trade_proposal",
    "ProposalArtifactError",
    "ProposalArtifactIssue",
    "ProposalLoadResult",
    "ProposalStatus",
    "ResearchSignalBridgeError",
    "ResearchSignalIssue",
    "SignalProposalConfig",
    "SignalProposalResult",
    "build_trade_proposal_from_signal_rows",
    "build_option_trade_proposal_from_candidate",
    "load_research_signal_rows",
    "order_intent_from_signal_row",
    "signal_from_row",
    "load_trade_proposal_artifacts",
    "TradeProposal",
    "OrderCostEstimate",
    "TransactionCostConfigurationError",
    "TransactionCostProfile",
    "TransactionCostReadinessError",
    "TransactionCostRegistry",
    "trade_proposal_directory",
    "trade_proposal_to_dict",
    "write_research_signal_proposal",
    "write_option_trade_proposal_from_candidate",
    "write_trade_proposal_artifact",
]
