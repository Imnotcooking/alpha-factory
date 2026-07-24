"""Validation and application of strategy-level risk overlays."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from oqp.research.strategy_risk_overlays.contracts import (
    StrategyRiskOverlayContract,
    resolve_strategy_risk_overlay_contract,
)


@dataclass(frozen=True)
class StrategyRiskOverlayResult:
    frame: pd.DataFrame
    contract: StrategyRiskOverlayContract


def apply_strategy_risk_overlay(
    targets: pd.DataFrame,
    module: ModuleType,
    *,
    overlay_id: str,
    parameters: Mapping[str, Any] | None,
    market_vertical: str,
) -> StrategyRiskOverlayResult:
    """Apply one overlay while preserving the strategy's position grid."""

    contract = resolve_strategy_risk_overlay_contract(
        module,
        overlay_id=overlay_id,
    )
    supported = set(contract.supported_markets)
    if "*" not in supported and market_vertical not in supported:
        raise ValueError(
            f"{contract.overlay_id} does not support {market_vertical}"
        )
    required = {
        contract.date_col,
        contract.ticker_col,
        contract.price_col,
        contract.source_weight_col,
    }
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(
            f"{contract.overlay_id} input is missing: {', '.join(missing)}"
        )
    keys = [contract.date_col, contract.ticker_col]
    if targets.duplicated(keys).any():
        raise ValueError(
            f"{contract.overlay_id} requires one row per date and ticker"
        )

    apply_function = getattr(module, "apply", None)
    if not callable(apply_function):
        raise ValueError(f"{contract.overlay_id} must expose apply()")
    original_attrs = dict(targets.attrs)
    source = pd.to_numeric(
        targets[contract.source_weight_col], errors="coerce"
    ).fillna(0.0)
    out = apply_function(targets.copy(), parameters=dict(parameters or {}))
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"{contract.overlay_id} apply() must return a DataFrame")
    if contract.output_weight_col not in out.columns:
        raise ValueError(
            f"{contract.overlay_id} output is missing "
            f"{contract.output_weight_col!r}"
        )
    if out.duplicated(keys).any() or len(out) != len(targets):
        raise ValueError(
            f"{contract.overlay_id} must preserve the strategy position grid"
        )
    expected_keys = pd.MultiIndex.from_frame(targets[keys])
    actual_keys = pd.MultiIndex.from_frame(out[keys])
    if not expected_keys.equals(actual_keys):
        raise ValueError(
            f"{contract.overlay_id} must preserve row order and position keys"
        )

    output = pd.to_numeric(
        out[contract.output_weight_col], errors="coerce"
    )
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError(f"{contract.overlay_id} produced non-finite weights")
    tolerance = 1e-12
    if not contract.allow_sign_flip:
        sign_flip = (source.abs() > tolerance) & (
            np.sign(source) != np.sign(output)
        )
        if sign_flip.any():
            raise ValueError(f"{contract.overlay_id} flipped a position sign")
    if not contract.allow_gross_increase:
        gross_increase = output.abs() > source.abs() + tolerance
        if gross_increase.any():
            raise ValueError(
                f"{contract.overlay_id} increased gross exposure"
            )

    out.attrs.update(original_attrs)
    out.attrs["strategy_risk_overlay_id"] = contract.overlay_id
    out.attrs["strategy_risk_overlay_contract"] = contract.to_dict()
    out.attrs["strategy_risk_overlay_parameters"] = dict(parameters or {})
    return StrategyRiskOverlayResult(frame=out, contract=contract)


__all__ = ["StrategyRiskOverlayResult", "apply_strategy_risk_overlay"]
