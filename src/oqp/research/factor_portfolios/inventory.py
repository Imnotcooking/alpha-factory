"""Read-only inventory of factor recipes available to portfolio construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.contracts import resolve_factor_supported_markets
from oqp.research.factor_governance import inspect_factor_governance
from oqp.research.factors import iter_factor_files, load_factor_module
from oqp.research.strategy_routing import (
    iter_router_files,
    load_router_module,
    resolve_router_contract,
)
from oqp.research.strategy_risk_overlays import (
    iter_strategy_risk_overlay_files,
    load_strategy_risk_overlay_module,
    resolve_strategy_risk_overlay_contract,
)


def factor_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_factor_files(include_public_examples=True):
        try:
            module = load_factor_module(path.stem)
            contract = getattr(module, "FACTOR_CONTRACT", {}) or {}
            metadata = getattr(module, "FACTOR_METADATA", {}) or {}
            if not isinstance(contract, dict):
                contract = {}
            if not isinstance(metadata, dict):
                metadata = {}
            factor_id = path.stem
            name = str(
                getattr(module, "FACTOR_NAME", None)
                or metadata.get("name")
                or getattr(module, "FACTOR_ID", None)
                or factor_id
            )
            category = str(
                getattr(module, "CATEGORY", None)
                or metadata.get("category")
                or "Unclassified"
            )
            component_type = _component_type(module, metadata, category, path)
            governance = inspect_factor_governance(path, module)
            rows.append(
                {
                    "factor_id": factor_id,
                    "declared_id": str(getattr(module, "FACTOR_ID", factor_id)),
                    "name": name,
                    "category": category,
                    "component_type": component_type,
                    "factor_family": governance.factor_family,
                    "factor_subfamily": governance.factor_subfamily,
                    "data_frequency": governance.data_frequency,
                    "portfolio_layer": governance.portfolio_layer,
                    "deduplication_cohort": governance.deduplication_cohort,
                    "governance_status": "normalized" if governance.valid else "legacy",
                    "governance_issues": "; ".join(governance.issues),
                    "supported_markets": ", ".join(
                        resolve_factor_supported_markets(module)
                    ),
                    "evaluation_geometry": contract.get("evaluation_geometry", "legacy"),
                    "execution_mode": contract.get("execution_mode", "legacy"),
                    "execution_lag": contract.get("execution_lag", "legacy"),
                    "return_assumption": contract.get("return_assumption", "legacy"),
                    "source": str(path),
                    "load_error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "factor_id": path.stem,
                    "declared_id": path.stem,
                    "name": path.stem,
                    "category": "Unavailable",
                    "component_type": "unavailable",
                    "factor_family": "",
                    "factor_subfamily": "",
                    "data_frequency": "",
                    "portfolio_layer": "",
                    "deduplication_cohort": "",
                    "governance_status": "unavailable",
                    "governance_issues": "",
                    "supported_markets": "",
                    "evaluation_geometry": "",
                    "execution_mode": "",
                    "execution_lag": "",
                    "return_assumption": "",
                    "source": str(path),
                    "load_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def compatible_factor_inventory(
    inventory: pd.DataFrame,
    market_vertical: str,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    supported = inventory["supported_markets"].fillna("").astype(str)
    mask = supported.str.contains(r"(?:^|,\s*)\*(?:,|$)", regex=True)
    mask |= supported.str.split(",").map(
        lambda values: market_vertical in {value.strip() for value in values}
    )
    mask &= inventory["load_error"].fillna("").eq("")
    return inventory.loc[mask].reset_index(drop=True)


def router_inventory() -> pd.DataFrame:
    """Return routers separately so they cannot enter raw factor blends."""

    rows: list[dict[str, Any]] = []
    for path in iter_router_files():
        try:
            module = load_router_module(path.stem)
            metadata = getattr(module, "ROUTER_METADATA", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            contract = resolve_router_contract(module, router_id=path.stem)
            rows.append(
                {
                    "router_id": path.stem,
                    "name": str(metadata.get("name") or path.stem),
                    "status": str(metadata.get("status") or "unclassified"),
                    "frequency": str(metadata.get("frequency") or "unclassified"),
                    "supported_markets": ", ".join(contract.supported_markets),
                    "state_col": contract.state_col,
                    "decision_date_col": contract.decision_date_col,
                    "effective_date_col": contract.effective_date_col,
                    "decision_lag_periods": contract.decision_lag_periods,
                    "source": str(path),
                    "load_error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "router_id": path.stem,
                    "name": path.stem,
                    "status": "unavailable",
                    "frequency": "",
                    "supported_markets": "",
                    "state_col": "",
                    "decision_date_col": "",
                    "effective_date_col": "",
                    "decision_lag_periods": "",
                    "source": str(path),
                    "load_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def compatible_router_inventory(
    inventory: pd.DataFrame,
    market_vertical: str,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    supported = inventory["supported_markets"].fillna("").astype(str)
    mask = supported.str.contains(r"(?:^|,\s*)\*(?:,|$)", regex=True)
    mask |= supported.str.split(",").map(
        lambda values: market_vertical in {value.strip() for value in values}
    )
    mask &= inventory["load_error"].fillna("").eq("")
    return inventory.loc[mask].reset_index(drop=True)


def strategy_risk_overlay_inventory() -> pd.DataFrame:
    """Return exposure overlays separately from factors and sleeve routers."""

    rows: list[dict[str, Any]] = []
    for path in iter_strategy_risk_overlay_files():
        try:
            module = load_strategy_risk_overlay_module(path.stem)
            metadata = getattr(module, "OVERLAY_METADATA", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            contract = resolve_strategy_risk_overlay_contract(
                module,
                overlay_id=path.stem,
            )
            parameters = getattr(module, "DEFAULT_PARAMETERS", {}) or {}
            if not isinstance(parameters, dict):
                parameters = {}
            rows.append(
                {
                    "overlay_id": path.stem,
                    "name": str(metadata.get("name") or path.stem),
                    "status": str(metadata.get("status") or "unclassified"),
                    "frequency": str(
                        metadata.get("frequency") or "unclassified"
                    ),
                    "supported_markets": ", ".join(
                        contract.supported_markets
                    ),
                    "scope": contract.scope,
                    "decision_time": contract.decision_time,
                    "effective_time": contract.effective_time,
                    "allow_sign_flip": contract.allow_sign_flip,
                    "allow_gross_increase": contract.allow_gross_increase,
                    "economic_rationale": str(
                        metadata.get("economic_rationale") or ""
                    ),
                    "known_limitations": str(
                        metadata.get("known_limitations") or ""
                    ),
                    "contract": contract.to_dict(),
                    "parameters": parameters,
                    "source": str(path),
                    "load_error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "overlay_id": path.stem,
                    "name": path.stem,
                    "status": "unavailable",
                    "frequency": "",
                    "supported_markets": "",
                    "scope": "",
                    "decision_time": "",
                    "effective_time": "",
                    "allow_sign_flip": "",
                    "allow_gross_increase": "",
                    "economic_rationale": "",
                    "known_limitations": "",
                    "contract": {},
                    "parameters": {},
                    "source": str(path),
                    "load_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def compatible_strategy_risk_overlay_inventory(
    inventory: pd.DataFrame,
    market_vertical: str,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    supported = inventory["supported_markets"].fillna("").astype(str)
    mask = supported.str.contains(r"(?:^|,\s*)\*(?:,|$)", regex=True)
    mask |= supported.str.split(",").map(
        lambda values: market_vertical in {value.strip() for value in values}
    )
    mask &= inventory["load_error"].fillna("").eq("")
    return inventory.loc[mask].reset_index(drop=True)


def _component_type(module, metadata: dict[str, Any], category: str, path: Path) -> str:
    explicit = (
        getattr(module, "COMPONENT_TYPE", None)
        or metadata.get("component_type")
        or metadata.get("type")
    )
    if explicit:
        return str(explicit).strip().lower()
    label = " ".join(
        [
            str(getattr(module, "FACTOR_NAME", "")),
            category,
            path.stem,
        ]
    ).lower()
    if "router" in label:
        return "router"
    if "state" in label and "state space" not in label:
        return "state"
    return "factor"


__all__ = [
    "compatible_factor_inventory",
    "compatible_router_inventory",
    "compatible_strategy_risk_overlay_inventory",
    "factor_inventory",
    "router_inventory",
    "strategy_risk_overlay_inventory",
]
