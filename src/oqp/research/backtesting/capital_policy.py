"""Research execution capital profiles for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ExecutionCapitalProfile:
    initial_capital: float
    currency: str
    profile: str
    source: str

    def to_attrs(self) -> dict[str, str | float]:
        return {
            "initial_capital": float(self.initial_capital),
            "capital_currency": self.currency,
            "capital_profile": self.profile,
            "capital_source": self.source,
        }


DEFAULT_CAPITAL_PROFILES: dict[str, dict[str, tuple[float, str]]] = {
    "FUTURES_CN": {
        "default": (1_000_000.0, "CNY"),
        "small_personal_futures_cn": (200_000.0, "CNY"),
        "medium_personal_futures_cn": (500_000.0, "CNY"),
        "institutional_futures_cn": (10_000_000.0, "CNY"),
    },
    "FUTURES_US": {
        "default": (1_000_000.0, "USD"),
        "small_personal_futures_us": (100_000.0, "USD"),
        "institutional_futures_us": (10_000_000.0, "USD"),
    },
    "EQUITY_US": {
        "default": (1_000_000.0, "USD"),
        "small_personal_equity_us": (100_000.0, "USD"),
        "institutional_equity_us": (10_000_000.0, "USD"),
    },
    "EQUITY_CN": {
        "default": (1_000_000.0, "CNY"),
        "small_personal_equity_cn": (200_000.0, "CNY"),
        "institutional_equity_cn": (10_000_000.0, "CNY"),
    },
    "EQUITY_HK": {
        "default": (1_000_000.0, "HKD"),
        "small_personal_equity_hk": (200_000.0, "HKD"),
        "institutional_equity_hk": (10_000_000.0, "HKD"),
    },
    "CRYPTO_PERP": {
        "default": (100_000.0, "USD"),
        "small_personal_crypto": (25_000.0, "USD"),
        "institutional_crypto": (1_000_000.0, "USD"),
    },
    "FX_SPOT": {
        "default": (1_000_000.0, "USD"),
    },
}


def resolve_execution_capital(
    *,
    asset_class: str,
    factor_module: ModuleType | None = None,
    initial_capital: float | None = None,
    capital_currency: str | None = None,
    capital_profile: str | None = None,
) -> ExecutionCapitalProfile:
    asset_key = str(asset_class or "").upper()
    profile_map = DEFAULT_CAPITAL_PROFILES.get(
        asset_key,
        {"default": (1_000_000.0, "USD")},
    )
    factor_contract = _factor_contract(factor_module)

    if initial_capital is not None:
        _, default_currency = _profile_values(
            profile_map,
            capital_profile or factor_contract.get("capital_profile"),
        )
        return ExecutionCapitalProfile(
            initial_capital=_positive_capital(initial_capital),
            currency=_currency(
                capital_currency
                or factor_contract.get("capital_currency")
                or default_currency
            ),
            profile=str(
                capital_profile
                or factor_contract.get("capital_profile")
                or "cli_override"
            ),
            source="cli_initial_capital",
        )

    if capital_profile:
        amount, currency = _profile_values(profile_map, capital_profile)
        return ExecutionCapitalProfile(
            initial_capital=amount,
            currency=_currency(capital_currency or currency),
            profile=str(capital_profile),
            source="cli_capital_profile",
        )

    factor_capital = factor_contract.get("initial_capital")
    if factor_capital not in (None, ""):
        profile = str(factor_contract.get("capital_profile") or "factor_contract")
        _, default_currency = _profile_values(profile_map, profile)
        return ExecutionCapitalProfile(
            initial_capital=_positive_capital(factor_capital),
            currency=_currency(
                capital_currency
                or factor_contract.get("capital_currency")
                or default_currency
            ),
            profile=profile,
            source="factor_contract_initial_capital",
        )

    factor_profile = factor_contract.get("capital_profile")
    if factor_profile:
        amount, currency = _profile_values(profile_map, str(factor_profile))
        return ExecutionCapitalProfile(
            initial_capital=amount,
            currency=_currency(
                capital_currency or factor_contract.get("capital_currency") or currency
            ),
            profile=str(factor_profile),
            source="factor_contract_capital_profile",
        )

    amount, currency = _profile_values(profile_map, "default")
    return ExecutionCapitalProfile(
        initial_capital=amount,
        currency=_currency(capital_currency or currency),
        profile="default",
        source=f"asset_default:{asset_key or 'UNKNOWN'}",
    )


def attach_capital_attrs(df, profile: ExecutionCapitalProfile):
    for key, value in profile.to_attrs().items():
        df.attrs[key] = value
    return df


def _factor_contract(factor_module: ModuleType | None) -> dict[str, Any]:
    if factor_module is None:
        return {}
    raw = getattr(factor_module, "FACTOR_CONTRACT", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _profile_values(
    profile_map: dict[str, tuple[float, str]],
    profile: str | None,
) -> tuple[float, str]:
    key = str(profile or "default").strip()
    if key in profile_map:
        return profile_map[key]
    return profile_map.get("default", (1_000_000.0, "USD"))


def _positive_capital(value: Any) -> float:
    amount = float(value)
    if amount <= 0:
        raise ValueError("initial capital must be positive.")
    return amount


def _currency(value: Any) -> str:
    return str(value or "USD").strip().upper() or "USD"


__all__ = [
    "DEFAULT_CAPITAL_PROFILES",
    "ExecutionCapitalProfile",
    "attach_capital_attrs",
    "resolve_execution_capital",
]
