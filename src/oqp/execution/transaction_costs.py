"""Versioned transaction-cost profiles and readiness gates."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from oqp.contracts.market_vertical import normalize_market_vertical


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRANSACTION_COST_CONFIG = REPO_ROOT / "config" / "execution" / "transaction_costs.yaml"


class TransactionCostConfigurationError(ValueError):
    """Raised when the central cost registry is malformed."""


class TransactionCostReadinessError(RuntimeError):
    """Raised when a cost profile is too incomplete for the requested claim."""


class CostUseCase(str, Enum):
    EXPLORATORY_GROSS = "exploratory_gross"
    RESEARCH_NET = "research_net"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class TransactionCostProfile:
    profile_id: str
    market_vertical: str
    status: str
    completeness: str
    currency: str
    research_net_ready: bool
    production_ready: bool
    engine_support: str
    effective_from: str | None = None
    commission: Mapping[str, Any] = field(default_factory=dict)
    third_party_fees: Mapping[str, Any] = field(default_factory=dict)
    slippage: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, profile_id: str, payload: Mapping[str, Any]) -> "TransactionCostProfile":
        market_vertical = normalize_market_vertical(payload.get("market_vertical"))
        status = str(payload.get("status") or "").strip().lower()
        if status not in {"verified", "placeholder"}:
            raise TransactionCostConfigurationError(
                f"Cost profile {profile_id!r} has unsupported status {status!r}."
            )
        return cls(
            profile_id=str(profile_id),
            market_vertical=market_vertical,
            status=status,
            completeness=str(payload.get("completeness") or "unknown"),
            currency=str(payload.get("currency") or "USD").upper(),
            research_net_ready=bool(payload.get("research_net_ready", False)),
            production_ready=bool(payload.get("production_ready", False)),
            engine_support=str(payload.get("engine_support") or "gross_only"),
            effective_from=(
                str(payload["effective_from"])
                if payload.get("effective_from") is not None
                else None
            ),
            commission=dict(payload.get("commission") or {}),
            third_party_fees=dict(payload.get("third_party_fees") or {}),
            slippage=dict(payload.get("slippage") or {}),
            sources=tuple(dict(item) for item in payload.get("sources") or ()),
            limitations=tuple(str(item) for item in payload.get("limitations") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "market_vertical": self.market_vertical,
            "status": self.status,
            "completeness": self.completeness,
            "currency": self.currency,
            "effective_from": self.effective_from,
            "research_net_ready": self.research_net_ready,
            "production_ready": self.production_ready,
            "engine_support": self.engine_support,
            "commission": dict(self.commission),
            "third_party_fees": dict(self.third_party_fees),
            "slippage": dict(self.slippage),
            "sources": [dict(item) for item in self.sources],
            "limitations": list(self.limitations),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def assert_ready(self, use_case: str | CostUseCase) -> None:
        resolved = CostUseCase(str(getattr(use_case, "value", use_case)).strip().lower())
        if resolved == CostUseCase.EXPLORATORY_GROSS:
            return
        if self.status != "verified":
            raise TransactionCostReadinessError(
                f"{self.profile_id} is a placeholder. Only exploratory_gross analysis is allowed."
            )
        if resolved == CostUseCase.RESEARCH_NET and not self.research_net_ready:
            raise TransactionCostReadinessError(
                f"{self.profile_id} is not wired accurately enough for research_net results. "
                f"Engine support is {self.engine_support!r}."
            )
        if resolved == CostUseCase.PRODUCTION and not self.production_ready:
            raise TransactionCostReadinessError(
                f"{self.profile_id} cannot support a production-ready claim."
            )

    def readiness_actions(self) -> tuple[str, ...]:
        """Return concrete work required before this profile can be promoted."""

        actions = list(self.limitations)
        if self.status == "placeholder":
            actions.append(
                "Populate broker commissions, third-party fees, and slippage in the central registry."
            )
        if self.engine_support == "estimator_only":
            actions.append(
                "Wire the profile's exact order logic into the shared backtest engine and add reconciliation tests."
            )
        if self.engine_support == "gross_only":
            actions.append(
                "Add a market-specific execution adapter before reporting net performance."
            )
        return tuple(dict.fromkeys(str(action) for action in actions if str(action).strip()))


@dataclass(frozen=True, slots=True)
class OrderCostEstimate:
    profile_id: str
    currency: str
    broker_commission: float
    regulatory_fees: float
    clearing_fees: float
    slippage_cost: float
    total_cost: float
    complete: bool
    omissions: tuple[str, ...] = ()


class TransactionCostRegistry:
    """Load, validate, route, and fingerprint transaction-cost profiles."""

    def __init__(
        self,
        *,
        registry_id: str,
        schema_version: int,
        as_of: str,
        default_profiles: Mapping[str, str],
        profiles: Mapping[str, TransactionCostProfile],
        source_path: Path,
    ) -> None:
        self.registry_id = registry_id
        self.schema_version = int(schema_version)
        self.as_of = as_of
        self.default_profiles = {
            normalize_market_vertical(key): str(value)
            for key, value in default_profiles.items()
        }
        self.profiles = dict(profiles)
        self.source_path = source_path
        self._validate()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TransactionCostRegistry":
        source_path = Path(path or DEFAULT_TRANSACTION_COST_CONFIG).expanduser().resolve()
        if not source_path.exists():
            raise TransactionCostConfigurationError(
                f"Transaction-cost registry does not exist: {source_path}"
            )
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        raw_profiles = payload.get("profiles") or {}
        profiles = {
            str(profile_id): TransactionCostProfile.from_mapping(str(profile_id), profile)
            for profile_id, profile in raw_profiles.items()
        }
        return cls(
            registry_id=str(payload.get("registry_id") or source_path.stem),
            schema_version=int(payload.get("schema_version") or 0),
            as_of=str(payload.get("as_of") or "unknown"),
            default_profiles=payload.get("default_profiles") or {},
            profiles=profiles,
            source_path=source_path,
        )

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise TransactionCostConfigurationError(
                f"Unsupported transaction-cost schema version {self.schema_version}."
            )
        if not self.profiles:
            raise TransactionCostConfigurationError("Transaction-cost registry has no profiles.")
        for market_vertical, profile_id in self.default_profiles.items():
            profile = self.profiles.get(profile_id)
            if profile is None:
                raise TransactionCostConfigurationError(
                    f"Default profile {profile_id!r} for {market_vertical} does not exist."
                )
            if profile.market_vertical != market_vertical:
                raise TransactionCostConfigurationError(
                    f"Default profile {profile_id!r} belongs to {profile.market_vertical}, "
                    f"not {market_vertical}."
                )

    def resolve(
        self,
        market_vertical: str,
        profile_id: str | None = None,
    ) -> TransactionCostProfile:
        vertical = normalize_market_vertical(market_vertical)
        resolved_id = str(profile_id or self.default_profiles.get(vertical) or "")
        if not resolved_id:
            raise TransactionCostConfigurationError(
                f"No default transaction-cost profile is registered for {vertical}."
            )
        profile = self.profiles.get(resolved_id)
        if profile is None:
            raise TransactionCostConfigurationError(
                f"Unknown transaction-cost profile {resolved_id!r}."
            )
        if profile.market_vertical != vertical:
            raise TransactionCostConfigurationError(
                f"Cost profile {resolved_id!r} is for {profile.market_vertical}, not {vertical}."
            )
        return profile

    def estimate_order_cost(
        self,
        profile: TransactionCostProfile | str,
        *,
        quantity: float,
        price: float,
        side: str = "buy",
        option_premium: float | None = None,
        monthly_volume: float = 0.0,
        instrument_profile: Any | None = None,
        close_kind: str = "open",
        half_spread: float | None = None,
        external_slippage_cost: float | None = None,
    ) -> OrderCostEstimate:
        if isinstance(profile, str):
            try:
                resolved = self.profiles[profile]
            except KeyError as exc:
                raise TransactionCostConfigurationError(
                    f"Unknown transaction-cost profile {profile!r}."
                ) from exc
        else:
            resolved = profile

        quantity = abs(float(quantity))
        price = max(float(price), 0.0)
        side = str(side).strip().lower()
        model = str(resolved.commission.get("model") or "")
        broker_commission = 0.0
        regulatory_fees = 0.0
        clearing_fees = 0.0
        slippage_cost = 0.0
        omissions: list[str] = []

        if model == "instrument_master_schedule":
            if instrument_profile is None:
                raise ValueError("instrument_profile is required for instrument_master_schedule.")
            fee_field = {
                "open": "fee_open",
                "close_history": "fee_close_history",
                "close_today": "fee_close_today",
            }.get(close_kind)
            if fee_field is None:
                raise ValueError(f"Unsupported close_kind {close_kind!r}.")
            fee = float(getattr(instrument_profile, fee_field))
            if str(getattr(instrument_profile, "fee_type", "ratio")).lower() == "fixed":
                broker_commission = quantity * fee
            else:
                broker_commission = (
                    quantity
                    * price
                    * float(getattr(instrument_profile, "multiplier", 1.0))
                    * fee
                )
        elif model == "per_share_order_bounds":
            base = quantity * float(resolved.commission["rate_per_share"])
            minimum = float(resolved.commission["minimum_per_order"])
            cap = quantity * price * float(resolved.commission["maximum_trade_value_rate"])
            broker_commission = min(max(base, minimum), cap) if cap > 0.0 else 0.0
            regulatory_fees = self._us_equity_regulatory_fees(
                resolved,
                quantity=quantity,
                price=price,
                side=side,
            )
        elif model == "option_premium_tiered_per_contract":
            if option_premium is None:
                raise ValueError("option_premium is required for the US option schedule.")
            maximum = float(resolved.commission.get("supported_monthly_contracts_max") or 0.0)
            if maximum and float(monthly_volume) > maximum:
                omissions.append("monthly volume exceeds the implemented commission tier")
            rate = self._option_rate(resolved, float(option_premium))
            broker_commission = max(
                quantity * rate,
                float(resolved.commission.get("minimum_per_order") or 0.0),
            )
            fees = resolved.third_party_fees
            clearing_fees = quantity * float(fees.get("occ_clearing_per_contract") or 0.0)
            regulatory_fees = quantity * float(fees.get("finra_cat_per_contract") or 0.0)
            if side in {"sell", "short", "close_sell"}:
                contract_multiplier = float(fees.get("contract_multiplier") or 100.0)
                sale_value = quantity * float(option_premium) * contract_multiplier
                regulatory_fees += sale_value * float(fees.get("sec_sale_value_rate") or 0.0)
                regulatory_fees += quantity * float(fees.get("finra_taf_per_contract_sold") or 0.0)
            if fees.get("exchange_and_orf_model") == "pending_venue_route":
                omissions.append("exchange and exchange-specific ORF fees")
        elif resolved.status == "placeholder":
            omissions.append("placeholder profile has no net-cost model")
        else:
            omissions.append(f"unsupported commission model: {model or 'missing'}")

        slippage_model = str(resolved.slippage.get("model") or "")
        if slippage_model == "fixed_ticks":
            if instrument_profile is None:
                raise ValueError("instrument_profile is required for fixed-tick slippage.")
            slippage_cost = (
                quantity
                * float(getattr(instrument_profile, "tick_size", 0.0))
                * float(getattr(instrument_profile, "multiplier", 1.0))
                * float(resolved.slippage.get("ticks_per_side") or 0.0)
            )
        elif slippage_model == "bid_ask_half_spread":
            if half_spread is None:
                omissions.append("bid/ask quote required for option slippage")
            else:
                multiplier = float(resolved.third_party_fees.get("contract_multiplier") or 1.0)
                slippage_cost = quantity * abs(float(half_spread)) * multiplier
        elif slippage_model == "square_root_tca":
            if external_slippage_cost is None:
                omissions.append("portfolio TCA inputs required for equity slippage")
            else:
                slippage_cost = max(float(external_slippage_cost), 0.0)
        elif slippage_model not in {"", "pending"}:
            omissions.append(f"unsupported slippage model: {slippage_model}")

        total = broker_commission + regulatory_fees + clearing_fees + slippage_cost
        return OrderCostEstimate(
            profile_id=resolved.profile_id,
            currency=resolved.currency,
            broker_commission=float(broker_commission),
            regulatory_fees=float(regulatory_fees),
            clearing_fees=float(clearing_fees),
            slippage_cost=float(slippage_cost),
            total_cost=float(total),
            complete=not omissions and resolved.completeness == "complete",
            omissions=tuple(omissions),
        )

    @staticmethod
    def _us_equity_regulatory_fees(
        profile: TransactionCostProfile,
        *,
        quantity: float,
        price: float,
        side: str,
    ) -> float:
        fees = profile.third_party_fees
        total = quantity * float(fees.get("finra_cat_per_share") or 0.0)
        if side in {"sell", "short", "close_sell"}:
            total += quantity * price * float(fees.get("sec_sale_value_rate") or 0.0)
            taf = quantity * float(fees.get("finra_taf_per_share_sold") or 0.0)
            taf_cap = float(fees.get("finra_taf_max_per_trade") or taf)
            total += min(taf, taf_cap)
        return total

    @staticmethod
    def _option_rate(profile: TransactionCostProfile, premium: float) -> float:
        tiers = list(profile.commission.get("premium_tiers") or ())
        for tier in tiers:
            if "premium_lt" in tier and premium < float(tier["premium_lt"]):
                return float(tier["rate_per_contract"])
            if "premium_gte" in tier and premium >= float(tier["premium_gte"]):
                return float(tier["rate_per_contract"])
        raise TransactionCostConfigurationError(
            f"No option premium tier covers premium={premium:g} in {profile.profile_id}."
        )


def attach_transaction_cost_policy(
    frame: pd.DataFrame,
    *,
    market_vertical: str,
    profile_id: str | None = None,
    use_case: str | CostUseCase = CostUseCase.RESEARCH_NET,
    registry: TransactionCostRegistry | None = None,
) -> pd.DataFrame:
    """Attach a frozen cost profile and enforce its allowed claim level."""

    resolved_registry = registry or TransactionCostRegistry.load()
    resolved_use_case = CostUseCase(str(getattr(use_case, "value", use_case)).strip().lower())
    profile = resolved_registry.resolve(market_vertical, profile_id)
    try:
        profile.assert_ready(resolved_use_case)
    except TransactionCostReadinessError as exc:
        message = format_transaction_cost_readiness_error(
            profile,
            use_case=resolved_use_case,
            registry_path=resolved_registry.source_path,
            reason=str(exc),
        )
        print(message, file=sys.stderr)
        raise TransactionCostReadinessError(message) from exc
    out = frame.copy()
    out.attrs.update(frame.attrs)
    out.attrs.update(
        {
            "transaction_cost_registry_id": resolved_registry.registry_id,
            "transaction_cost_registry_as_of": resolved_registry.as_of,
            "transaction_cost_registry_path": str(resolved_registry.source_path),
            "transaction_cost_profile_id": profile.profile_id,
            "transaction_cost_profile_fingerprint": profile.fingerprint,
            "transaction_cost_profile_status": profile.status,
            "transaction_cost_completeness": profile.completeness,
            "transaction_cost_use_case": resolved_use_case.value,
            "transaction_cost_research_net_ready": profile.research_net_ready,
            "transaction_cost_production_ready": profile.production_ready,
            "transaction_cost_engine_support": profile.engine_support,
            "transaction_cost_gross_only": resolved_use_case == CostUseCase.EXPLORATORY_GROSS,
            "transaction_cost_assumptions": profile.to_dict(),
        }
    )
    return out


ensure_transaction_cost_policy = attach_transaction_cost_policy


def format_transaction_cost_readiness_error(
    profile: TransactionCostProfile,
    *,
    use_case: str | CostUseCase,
    registry_path: str | Path,
    reason: str,
) -> str:
    """Build the same actionable readiness explanation used by CLIs and logs."""

    resolved_use_case = CostUseCase(
        str(getattr(use_case, "value", use_case)).strip().lower()
    )
    lines = [
        "[TRANSACTION COST READINESS BLOCK]",
        f"Market: {profile.market_vertical}",
        f"Profile: {profile.profile_id}",
        f"Requested claim: {resolved_use_case.value}",
        f"Reason: {reason}",
        "What to add or finish:",
    ]
    actions = profile.readiness_actions()
    if actions:
        lines.extend(f"  {index}. {action}" for index, action in enumerate(actions, start=1))
    else:
        lines.append("  1. Complete and verify the missing execution-cost inputs.")
    lines.extend(
        [
            f"Registry: {Path(registry_path)}",
            (
                "Gross-only fallback: set transaction_cost_use_case='exploratory_gross' "
                "only when the result will not be described as net or production-ready."
            ),
        ]
    )
    return "\n".join(lines)
