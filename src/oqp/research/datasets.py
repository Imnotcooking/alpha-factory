from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from oqp.contracts.market_vertical import normalize_market_vertical


@dataclass(frozen=True)
class DatasetTradabilityProfile:
    dataset_role: str
    tradability: str
    price_source: str
    roll_model: str
    liquidity_model: str
    execution_reality: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_attrs(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def infer_dataset_tradability(
    df: pd.DataFrame,
    *,
    source_path: str = "",
    asset_class: str = "",
    data_frequency: str = "",
) -> DatasetTradabilityProfile:
    """Classify whether a dataset is executable contract data or a research proxy.

    This deliberately does not block research. It labels the evidentiary strength
    of the backtest so promotion/paper-trading gates can require contract data.
    """

    path_text = _normalize_text(os.path.basename(str(source_path or "")))
    asset = normalize_market_vertical(
        asset_class or df.attrs.get("market_vertical") or df.attrs.get("asset_class")
    )
    frequency = str(data_frequency or df.attrs.get("data_frequency") or "").strip().lower()
    tick_like = frequency == "tick" or _has_tick_columns(df) or "_tick_" in path_text
    index_like = _looks_like_index(df, path_text)
    main_like = _looks_like_main_contract(df, path_text)
    continuous_like = _looks_like_continuous(df, path_text)

    warnings: list[str] = []
    if tick_like:
        return DatasetTradabilityProfile(
            dataset_role="contract_tick",
            tradability="executable",
            price_source="contract_l1_tick",
            roll_model="explicit_contract",
            liquidity_model="contract_specific_l1",
            execution_reality="live_contract_proxy",
        )

    if "FUTURES" in asset and index_like:
        warnings.append(
            "Dataset appears to be an index/continuous research series; roll mechanics and contract liquidity are hidden."
        )
        return DatasetTradabilityProfile(
            dataset_role="research_index",
            tradability="research_proxy",
            price_source="daily_index",
            roll_model="hidden_index_roll",
            liquidity_model="proxy_volume_or_unknown",
            execution_reality="not_directly_executable",
            warnings=tuple(warnings),
        )

    if "FUTURES" in asset and main_like:
        warnings.append(
            "Dataset appears to be main-contract data; executable direction is closer to reality, but roll selection/liquidity must still be audited."
        )
        return DatasetTradabilityProfile(
            dataset_role="main_contract_daily",
            tradability="executable_proxy",
            price_source="daily_main_contract",
            roll_model="main_contract_roll",
            liquidity_model="contract_volume",
            execution_reality="contract_proxy_needs_roll_audit",
            warnings=tuple(warnings),
        )

    if "FUTURES" in asset and continuous_like:
        warnings.append(
            "Dataset appears to be a continuous/adjusted futures series; useful for signal research, but execution must be validated on actual contracts."
        )
        return DatasetTradabilityProfile(
            dataset_role="continuous_contract_research",
            tradability="research_proxy",
            price_source="continuous_adjusted_daily",
            roll_model="adjusted_or_unknown_roll",
            liquidity_model="proxy_volume_or_unknown",
            execution_reality="not_directly_executable",
            warnings=tuple(warnings),
        )

    if "FUTURES" in asset:
        warnings.append(
            "Futures dataset tradability is ambiguous; require a contract-data rerun before promotion."
        )
        return DatasetTradabilityProfile(
            dataset_role="unknown_futures_daily",
            tradability="unknown",
            price_source="unknown_daily",
            roll_model="unknown",
            liquidity_model="unknown",
            execution_reality="requires_data_audit",
            warnings=tuple(warnings),
        )

    return DatasetTradabilityProfile(
        dataset_role="asset_bars",
        tradability="executable_proxy",
        price_source="daily_bars",
        roll_model="not_applicable",
        liquidity_model="bar_volume",
        execution_reality="asset_bar_proxy",
    )


def attach_dataset_tradability_attrs(df: pd.DataFrame, profile: DatasetTradabilityProfile) -> pd.DataFrame:
    attrs = profile.to_attrs()
    df.attrs["dataset_tradability_profile"] = attrs
    df.attrs["dataset_role"] = profile.dataset_role
    df.attrs["data_tradability"] = profile.tradability
    df.attrs["data_price_source"] = profile.price_source
    df.attrs["data_roll_model"] = profile.roll_model
    df.attrs["data_liquidity_model"] = profile.liquidity_model
    df.attrs["data_execution_reality"] = profile.execution_reality
    df.attrs["data_tradability_warnings"] = list(profile.warnings)
    return df


def base_symbol_from_ticker(ticker: str) -> str:
    value = str(ticker or "").strip()
    paren_match = re.search(r"[\(（]([A-Za-z]+)[\)）]", value)
    if paren_match:
        return paren_match.group(1)
    return re.sub(r"\d+", "", value)


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _has_tick_columns(df: pd.DataFrame) -> bool:
    required = {"bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1"}
    return bool(required.issubset(df.columns) or {"datetime", "last_price", "symbol"}.issubset(df.columns))


def _looks_like_index(df: pd.DataFrame, path_text: str) -> bool:
    if "index" in path_text or "指数" in path_text:
        return True
    if "ticker" in df.columns:
        sample = df["ticker"].dropna().astype(str).head(500)
        if not sample.empty and sample.str.contains(r"\[指数\]|指数|\([A-Za-z]+\)", regex=True).mean() > 0.5:
            return True
    return False


def _looks_like_main_contract(df: pd.DataFrame, path_text: str) -> bool:
    if "_main_" in path_text or "raw_main" in path_text or "main_contract" in path_text:
        return True
    if "ticker" in df.columns:
        sample = df["ticker"].dropna().astype(str).head(500)
        if not sample.empty and sample.str.match(r"^[A-Za-z]+[0-9]{3,4}$").mean() > 0.5:
            return True
    if "symbol" in df.columns:
        sample = df["symbol"].dropna().astype(str).head(500)
        if not sample.empty and sample.str.match(r"^[A-Za-z]+[0-9]{3,4}$").mean() > 0.5:
            return True
    return False


def _looks_like_continuous(df: pd.DataFrame, path_text: str) -> bool:
    if any(marker in path_text for marker in ("adjusted", "continuous", "universe")):
        return True
    if "ticker" in df.columns:
        sample = df["ticker"].dropna().astype(str).head(500)
        if not sample.empty and sample.map(lambda item: bool(base_symbol_from_ticker(item))).all():
            contract_rate = sample.str.match(r"^[A-Za-z]+[0-9]{3,4}$").mean()
            return contract_rate < 0.5
    return False
