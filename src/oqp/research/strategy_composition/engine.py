"""Phase 7 assembly of frozen sleeves, routers, overlays, and execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from oqp.research.sleeves.evidence import SleeveEvidenceBundle
from oqp.research.sleeves.standalone import StandaloneSleeveTestBundle
from oqp.research.backtesting.margin_policy import apply_margin_utilization_cap
from oqp.research.strategy_composition.contracts import StrategyCompositionConfig
from oqp.research.strategy_risk_overlays import (
    apply_strategy_risk_overlay,
    load_strategy_risk_overlay_module,
)
from oqp.research.strategy_routing import (
    RouterContract,
    RouterHypothesisEvidenceBundle,
    route_sleeve_targets,
)


ANNUALIZATION_DAYS = 252.0
SESSION_FLAT_RETURN_ASSUMPTION = "close_signal_next_open_to_close"
REPO_ROOT = Path(__file__).resolve().parents[4]
TRANSACTION_COST_REGISTRY = REPO_ROOT / "config/execution/transaction_costs.yaml"


@dataclass(frozen=True, slots=True)
class FrozenSleeveComponent:
    sleeve_id: str
    factor_id: str
    market_vertical: str
    return_assumption: str
    config_fingerprint: str
    standalone_status: str
    router_eligible: bool
    target_positions: pd.DataFrame

    @classmethod
    def from_evidence(
        cls,
        phase3: SleeveEvidenceBundle,
        phase4: StandaloneSleeveTestBundle,
        *,
        sleeve_id: str | None = None,
    ) -> "FrozenSleeveComponent":
        component_id = str(sleeve_id or phase3.config.sleeve_id).strip()
        return cls(
            sleeve_id=component_id,
            factor_id=phase3.config.factor_id,
            market_vertical=phase3.config.market_vertical,
            return_assumption=phase3.config.return_assumption,
            config_fingerprint=str(
                phase3.manifest.get("config_fingerprint") or phase3.config.fingerprint
            ),
            standalone_status=str(
                phase4.summary.get("standalone_status") or "unknown"
            ),
            router_eligible=bool(phase4.summary.get("router_eligible", False)),
            target_positions=phase3.positions.copy(deep=True),
        )


@dataclass(frozen=True, slots=True)
class FrozenRouterComponent:
    router_id: str
    market_vertical: str
    sleeve_ids: tuple[str, ...]
    evidence_fingerprint: str
    router_status: str
    contract: RouterContract
    allocations: pd.DataFrame

    @classmethod
    def from_evidence(
        cls, evidence: RouterHypothesisEvidenceBundle
    ) -> "FrozenRouterComponent":
        config = evidence.config
        sleeve_ids = (config.sleeve_a_id, config.sleeve_b_id)
        if len(set(sleeve_ids)) != 2:
            raise ValueError(
                "Phase 7 requires distinct stable sleeve IDs; a generic sleeve "
                "template cannot identify both routed components"
            )
        date_col = config.date_col
        required = {date_col, "allocation_a", "allocation_b"}
        missing = sorted(required.difference(evidence.decision_log.columns))
        if missing:
            raise ValueError(f"router decision log is missing columns: {missing}")
        rows: list[pd.DataFrame] = []
        for sleeve_id, allocation_col in zip(
            sleeve_ids, ("allocation_a", "allocation_b"), strict=True
        ):
            part = evidence.decision_log[[date_col, allocation_col]].copy()
            part["decision_date"] = pd.to_datetime(part[date_col], errors="raise")
            part["effective_date"] = part["decision_date"]
            part["sleeve_id"] = sleeve_id
            part["allocation"] = pd.to_numeric(
                part[allocation_col], errors="raise"
            )
            rows.append(
                part[["decision_date", "effective_date", "sleeve_id", "allocation"]]
            )
        return cls(
            router_id=config.router_id,
            market_vertical=config.market_vertical,
            sleeve_ids=sleeve_ids,
            evidence_fingerprint=str(
                evidence.manifest.get("config_fingerprint") or config.fingerprint
            ),
            router_status=str(
                evidence.summary.get("router_status") or "unknown"
            ),
            contract=RouterContract(
                router_id=config.router_id,
                decision_lag_periods=0,
                supported_markets=(config.market_vertical,),
            ),
            allocations=pd.concat(rows, ignore_index=True),
        )


@dataclass(frozen=True, slots=True)
class StrategyCompositionBundle:
    config: StrategyCompositionConfig
    summary: dict[str, Any]
    positions: pd.DataFrame
    daily_returns: pd.DataFrame
    component_audit: pd.DataFrame
    transformation_audit: pd.DataFrame
    router_allocations: pd.DataFrame
    manifest: dict[str, Any]


def compose_strategy(
    config: StrategyCompositionConfig,
    sleeves: Mapping[str, FrozenSleeveComponent],
    *,
    router: FrozenRouterComponent | None = None,
    overlay_modules: Mapping[str, ModuleType] | None = None,
) -> StrategyCompositionBundle:
    """Compose frozen components and charge costs once on final positions."""

    _validate_cost_profile(config)
    ordered_sleeves = _validate_components(config, sleeves, router)
    base, sleeve_targets = _align_sleeve_targets(ordered_sleeves)
    transformations: list[dict[str, Any]] = []

    if router is None:
        frame = base.copy()
        frame["routed_target_weight"] = sleeve_targets[config.sleeves[0]][
            "target_weight"
        ].to_numpy()
        router_allocations = pd.DataFrame(
            columns=["decision_date", "effective_date", "sleeve_id", "allocation"]
        )
    else:
        routed = route_sleeve_targets(
            base, sleeve_targets, router.allocations, router.contract
        )
        frame = routed.frame
        router_allocations = routed.allocations
    frame["final_target_weight"] = frame["routed_target_weight"]
    transformations.append(_audit_step(frame, "router_allocation", "final_target_weight"))

    modules = dict(overlay_modules or {})
    overlay_fingerprints: dict[str, str] = {}
    for overlay_id in config.risk_overlays:
        module = modules.get(overlay_id) or load_strategy_risk_overlay_module(
            overlay_id
        )
        default_parameters = dict(getattr(module, "DEFAULT_PARAMETERS", {}))
        before = frame["final_target_weight"].copy()
        result = apply_strategy_risk_overlay(
            frame,
            module,
            overlay_id=overlay_id,
            parameters=default_parameters,
            market_vertical=config.market_vertical,
        )
        frame = result.frame
        frame[f"{overlay_id}_input_weight"] = before.to_numpy()
        frame[f"{overlay_id}_output_weight"] = frame[
            "final_target_weight"
        ].to_numpy()
        overlay_fingerprints[overlay_id] = _module_fingerprint(
            module, default_parameters
        )
        transformations.append(
            _audit_step(frame, f"risk_overlay:{overlay_id}", "final_target_weight")
        )

    frame = _apply_allocator(frame, config)
    transformations.append(
        _audit_step(frame, "allocator", "allocated_target_weight")
    )
    positions = _execute_final_session_targets(frame, config)
    daily = _summarize_daily(positions)
    metrics = _performance_metrics(daily)
    component_audit = _component_audit(ordered_sleeves, router, overlay_fingerprints)
    transformation_audit = pd.DataFrame(transformations)
    summary = {
        "schema_version": config.schema_version,
        "strategy_id": config.strategy_id,
        "status": "composed",
        "operation_order": [
            "sleeve_targets",
            "router_allocation",
            "risk_overlays",
            "allocator",
            "final_position_execution",
            "transaction_costs",
        ],
        "sleeve_count": len(config.sleeves),
        "router_id": config.router,
        "risk_overlay_count": len(config.risk_overlays),
        "performance": metrics,
        "costs_computed_from_final_positions": True,
        "sleeve_hypothetical_costs_consumed": False,
    }
    manifest = {
        "schema_version": config.schema_version,
        "phase": "Phase 7: Strategy Composition",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "component_fingerprints": {
            component.sleeve_id: component.config_fingerprint
            for component in ordered_sleeves
        }
        | ({router.router_id: router.evidence_fingerprint} if router else {})
        | overlay_fingerprints,
        "component_order": summary["operation_order"],
        "components_immutable": True,
        "optimization_permitted": False,
        "sleeve_hypothetical_cost_columns_ignored": True,
        "costs_computed_once_after_final_position_changes": True,
        "transaction_cost_profile": config.execution.transaction_cost_profile,
        "return_assumption": ordered_sleeves[0].return_assumption,
    }
    return StrategyCompositionBundle(
        config=config,
        summary=summary,
        positions=positions,
        daily_returns=daily,
        component_audit=component_audit,
        transformation_audit=transformation_audit,
        router_allocations=router_allocations,
        manifest=manifest,
    )


def write_strategy_composition_bundle(
    bundle: StrategyCompositionBundle, output_dir: str | Path
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", bundle.summary)
    _write_json(destination / "manifest.json", bundle.manifest)
    bundle.positions.to_parquet(destination / "positions.parquet", index=False)
    bundle.daily_returns.to_parquet(destination / "daily_returns.parquet", index=False)
    bundle.component_audit.to_csv(destination / "component_audit.csv", index=False)
    bundle.transformation_audit.to_csv(
        destination / "transformation_audit.csv", index=False
    )
    bundle.router_allocations.to_parquet(
        destination / "router_allocations.parquet", index=False
    )
    return destination


def _validate_components(
    config: StrategyCompositionConfig,
    sleeves: Mapping[str, FrozenSleeveComponent],
    router: FrozenRouterComponent | None,
) -> tuple[FrozenSleeveComponent, ...]:
    if set(sleeves) != set(config.sleeves):
        raise ValueError("provided sleeve components must exactly match strategy references")
    ordered = tuple(sleeves[sleeve_id] for sleeve_id in config.sleeves)
    for reference, component in zip(config.sleeves, ordered, strict=True):
        if component.sleeve_id != reference:
            raise ValueError(f"sleeve component mismatch for {reference}")
        if component.market_vertical != config.market_vertical:
            raise ValueError(f"{reference} does not support {config.market_vertical}")
        if not component.router_eligible:
            raise ValueError(f"{reference} did not pass the Phase 4 standalone gate")
    assumptions = {component.return_assumption for component in ordered}
    if len(assumptions) != 1:
        raise ValueError("all sleeves must share one return assumption")
    if assumptions != {SESSION_FLAT_RETURN_ASSUMPTION}:
        raise NotImplementedError(
            "Phase 7 final execution currently supports the frozen next-open-to-close "
            "one-session sleeve contract"
        )
    if config.router:
        if router is None or router.router_id != config.router:
            raise ValueError("the configured frozen router component is missing")
        if router.router_status != "eligible_for_strategy_review":
            raise ValueError("router did not pass the Phase 6 validation and holdout gates")
        if router.market_vertical != config.market_vertical:
            raise ValueError("router market does not match the strategy")
        if set(router.sleeve_ids) != set(config.sleeves):
            raise ValueError("router sleeve references do not match the strategy")
    elif router is not None:
        raise ValueError("an undeclared router component was supplied")
    return ordered


def _align_sleeve_targets(
    sleeves: tuple[FrozenSleeveComponent, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    execution_columns = [
        "date",
        "ticker",
        "close",
        "research_split",
        "forward_return",
        "next_symbol",
        "next_actual_open",
        "next_multiplier",
        "next_tick_size",
        "next_fee_type",
        "next_fee_open",
        "next_fee_close_today",
    ]
    targets: dict[str, pd.DataFrame] = {}
    base: pd.DataFrame | None = None
    keys: pd.MultiIndex | None = None
    for component in sleeves:
        missing = sorted(
            set(execution_columns + ["target_weight"]).difference(
                component.target_positions.columns
            )
        )
        if missing:
            raise ValueError(f"{component.sleeve_id} targets are missing: {missing}")
        positions = component.target_positions.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)
        if positions.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{component.sleeve_id} has duplicate position keys")
        observed_keys = pd.MultiIndex.from_frame(positions[["date", "ticker"]])
        if keys is None:
            keys = observed_keys
            base = positions[execution_columns].copy(deep=True)
        elif not keys.equals(observed_keys):
            raise ValueError("all sleeves must use the same date-product grid")
        else:
            assert base is not None
            for column in execution_columns[2:]:
                left = base[column].reset_index(drop=True)
                right = positions[column].reset_index(drop=True)
                if not left.equals(right):
                    raise ValueError(
                        f"sleeves disagree on shared execution field {column}"
                    )
        targets[component.sleeve_id] = positions[
            ["date", "ticker", "target_weight"]
        ].copy(deep=True)
    assert base is not None
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    return base, targets


def _apply_allocator(
    frame: pd.DataFrame, config: StrategyCompositionConfig
) -> pd.DataFrame:
    out = frame.copy()
    source = pd.to_numeric(out["final_target_weight"], errors="coerce").fillna(0.0)
    cap = config.allocator.max_contract_weight
    out["contract_cap_bound"] = source.abs().gt(cap + 1e-12)
    out["contract_capped_weight"] = source.clip(lower=-cap, upper=cap)
    max_gross = config.allocator.max_gross_leverage
    gross = out.groupby("date")["contract_capped_weight"].transform(
        lambda values: float(values.abs().sum())
    )
    if max_gross is None:
        out["gross_scale"] = 1.0
    else:
        out["gross_scale"] = np.minimum(
            1.0,
            max_gross / gross.where(gross.gt(0.0), np.nan),
        ).fillna(1.0)
    out["gross_cap_bound"] = out["gross_scale"].lt(1.0 - 1e-12)
    out["allocated_target_weight"] = (
        out["contract_capped_weight"] * out["gross_scale"]
    )
    return apply_margin_utilization_cap(
        out,
        market_vertical=config.market_vertical,
        source_weight_col="allocated_target_weight",
        max_margin_utilization=config.allocator.max_margin_utilization,
        target_columns=("allocated_target_weight",),
    )


def _execute_final_session_targets(
    frame: pd.DataFrame, config: StrategyCompositionConfig
) -> pd.DataFrame:
    out = frame.copy()
    capital = config.execution.capital
    target = pd.to_numeric(
        out["allocated_target_weight"], errors="coerce"
    ).fillna(0.0)
    price = pd.to_numeric(out["next_actual_open"], errors="coerce")
    multiplier = pd.to_numeric(out["next_multiplier"], errors="coerce")
    notional = price * multiplier
    executable = notional.gt(0.0) & out["next_symbol"].notna()
    desired = target * capital / notional.where(executable)
    out["contracts"] = np.trunc(desired).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    out["execution_eligible"] = executable
    out["executed_weight"] = out["contracts"] * notional.fillna(0.0) / capital
    if "margin_rate" in out.columns:
        out["executed_margin_contribution"] = (
            out["executed_weight"].abs()
            * pd.to_numeric(out["margin_rate"], errors="coerce").fillna(0.0)
        )
    contracts = out["contracts"].abs()
    open_fee = pd.to_numeric(out["next_fee_open"], errors="coerce").fillna(0.0)
    close_fee = pd.to_numeric(
        out["next_fee_close_today"], errors="coerce"
    ).fillna(0.0)
    fee_sum = open_fee + close_fee
    out["exchange_fee_cny"] = np.where(
        out["next_fee_type"].astype(str).str.lower().eq("fixed"),
        contracts * fee_sum,
        contracts * notional.fillna(0.0) * fee_sum,
    )
    out["slippage_cny"] = (
        2.0
        * contracts
        * config.execution.slippage_ticks_per_side
        * pd.to_numeric(out["next_tick_size"], errors="coerce").fillna(0.0)
        * multiplier.fillna(0.0)
    )
    out["exchange_fee_return"] = out["exchange_fee_cny"] / capital
    out["slippage_return"] = out["slippage_cny"] / capital
    out["cost_return"] = out["exchange_fee_return"] + out["slippage_return"]
    out["turnover"] = 2.0 * contracts * notional.fillna(0.0) / capital
    forward = pd.to_numeric(out["forward_return"], errors="coerce")
    out["gross_contribution"] = out["executed_weight"] * forward.fillna(0.0)
    out["net_contribution"] = out["gross_contribution"] - out["cost_return"]
    return out


def _summarize_daily(positions: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, Any]] = {
        "research_split": ("research_split", "first"),
        "gross_return": ("gross_contribution", "sum"),
        "exchange_fee_return": ("exchange_fee_return", "sum"),
        "slippage_return": ("slippage_return", "sum"),
        "cost_return": ("cost_return", "sum"),
        "turnover": ("turnover", "sum"),
        "target_gross": (
            "allocated_target_weight",
            lambda x: float(x.abs().sum()),
        ),
        "target_net": ("allocated_target_weight", "sum"),
        "executed_gross": ("executed_weight", lambda x: float(x.abs().sum())),
        "executed_net": ("executed_weight", "sum"),
        "active_products": ("contracts", lambda x: int(x.ne(0.0).sum())),
    }
    if "margin_contribution" in positions.columns:
        aggregations["target_margin_utilization"] = (
            "margin_contribution",
            "sum",
        )
    if "executed_margin_contribution" in positions.columns:
        aggregations["executed_margin_utilization"] = (
            "executed_margin_contribution",
            "sum",
        )
    daily = (
        positions.groupby("date", as_index=False, sort=True)
        .agg(**aggregations)
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["net_return"] = daily["gross_return"] - daily["cost_return"]
    daily["cumulative_gross_return"] = (1.0 + daily["gross_return"]).cumprod() - 1.0
    daily["cumulative_net_return"] = (1.0 + daily["net_return"]).cumprod() - 1.0
    return daily


def _performance_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(daily["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(daily["gross_return"], errors="coerce").dropna()
    wealth = (1.0 + net).cumprod()
    volatility = float(net.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)) if len(net) > 1 else math.nan
    annualized = float(net.mean() * ANNUALIZATION_DAYS) if len(net) else math.nan
    return _json_safe(
        {
            "trading_days": len(net),
            "gross_total_return": float((1.0 + gross).prod() - 1.0) if len(gross) else math.nan,
            "net_total_return": float(wealth.iloc[-1] - 1.0) if len(wealth) else math.nan,
            "net_annualized_mean": annualized,
            "net_annualized_volatility": volatility,
            "net_sharpe": annualized / volatility if volatility > 0.0 else math.nan,
            "maximum_drawdown": float((wealth / wealth.cummax() - 1.0).min()) if len(wealth) else math.nan,
            "annualized_cost": float(daily["cost_return"].mean() * ANNUALIZATION_DAYS),
            "annualized_turnover": float(daily["turnover"].mean() * ANNUALIZATION_DAYS),
        }
    )


def _component_audit(
    sleeves: tuple[FrozenSleeveComponent, ...],
    router: FrozenRouterComponent | None,
    overlay_fingerprints: Mapping[str, str],
) -> pd.DataFrame:
    rows = [
        {
            "component_type": "sleeve",
            "component_id": sleeve.sleeve_id,
            "fingerprint": sleeve.config_fingerprint,
            "status": sleeve.standalone_status,
            "eligible": sleeve.router_eligible,
        }
        for sleeve in sleeves
    ]
    if router:
        rows.append(
            {
                "component_type": "router",
                "component_id": router.router_id,
                "fingerprint": router.evidence_fingerprint,
                "status": router.router_status,
                "eligible": router.router_status == "eligible_for_strategy_review",
            }
        )
    rows.extend(
        {
            "component_type": "risk_overlay",
            "component_id": overlay_id,
            "fingerprint": fingerprint,
            "status": "frozen_default_parameters",
            "eligible": True,
        }
        for overlay_id, fingerprint in overlay_fingerprints.items()
    )
    return pd.DataFrame(rows)


def _audit_step(frame: pd.DataFrame, step: str, weight_col: str) -> dict[str, Any]:
    weight = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    daily_gross = weight.abs().groupby(frame["date"]).sum()
    return {
        "step": step,
        "weight_column": weight_col,
        "rows": len(frame),
        "mean_daily_gross": float(daily_gross.mean()) if len(daily_gross) else 0.0,
        "max_position_weight": float(weight.abs().max()) if len(weight) else 0.0,
    }


def _validate_cost_profile(config: StrategyCompositionConfig) -> None:
    payload = yaml.safe_load(TRANSACTION_COST_REGISTRY.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles") or {}
    profile = profiles.get(config.execution.transaction_cost_profile)
    if not isinstance(profile, Mapping):
        raise ValueError(
            f"unknown transaction cost profile: {config.execution.transaction_cost_profile}"
        )
    if str(profile.get("market_vertical")) != config.market_vertical:
        raise ValueError("transaction cost profile does not match the strategy market")
    if not bool(profile.get("research_net_ready", False)):
        raise ValueError(
            f"transaction cost profile {config.execution.transaction_cost_profile} "
            "is not ready for a net research backtest"
        )


def _module_fingerprint(module: ModuleType, parameters: Mapping[str, Any]) -> str:
    path = getattr(module, "__file__", None)
    source_hash = ""
    if path and Path(path).exists():
        source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = {
        "component_id": str(getattr(module, "OVERLAY_ID", "")),
        "source_hash": source_hash,
        "parameters": dict(parameters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FrozenRouterComponent",
    "FrozenSleeveComponent",
    "StrategyCompositionBundle",
    "compose_strategy",
    "write_strategy_composition_bundle",
]
