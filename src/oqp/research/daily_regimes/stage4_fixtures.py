"""Deterministic Stage 4 fixtures for formula and leakage tests only."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from oqp.research.daily_regimes.continuous_series import build_continuous_series
from oqp.research.daily_regimes.synthetic import (
    SyntheticFixtureConfig,
    make_clean_synthetic_fixture,
)


STAGE_OWNER = 4


@dataclass(frozen=True)
class Stage4SyntheticFixture:
    """Long point-in-time panel with an explicit two-sector taxonomy."""

    panel: pd.DataFrame
    source_contract_rows: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.panel, pd.DataFrame) or self.panel.empty:
            raise ValueError("Stage 4 fixture requires a non-empty panel.")
        if not isinstance(self.source_contract_rows, pd.DataFrame):
            raise TypeError("source_contract_rows must be a pandas DataFrame.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def make_stage4_synthetic_fixture(seed: int = 4204) -> Stage4SyntheticFixture:
    """Build 96 dates for four products without reading market data."""

    products = ("SYN_A", "SYN_B", "SYN_C", "SYN_D")
    source = make_clean_synthetic_fixture(
        SyntheticFixtureConfig(
            seed=seed,
            products=products,
            periods=96,
            regime_block_periods=12,
        )
    )
    continuous = build_continuous_series(source.contract_rows)
    panel = continuous.panel.copy(deep=True)
    sector_map = {
        "SYN_A": "synthetic_metals",
        "SYN_B": "synthetic_metals",
        "SYN_C": "synthetic_agriculture",
        "SYN_D": "synthetic_agriculture",
    }
    panel["sector"] = panel["product"].map(sector_map)
    panel = panel.sort_values(
        ["product", "trading_date"], kind="mergesort"
    ).reset_index(drop=True)
    return Stage4SyntheticFixture(
        panel=panel,
        source_contract_rows=source.contract_rows.copy(deep=True),
        metadata={
            "fixture_id": f"stage4_long_feature_fixture_s{seed}_v1",
            "synthetic": True,
            "scientific_evidence": False,
            "paper_eligible": False,
            "products": list(products),
            "periods_per_source_product": 96,
            "sector_taxonomy_is_synthetic": True,
            "seed": seed,
        },
    )


__all__ = [
    "STAGE_OWNER",
    "Stage4SyntheticFixture",
    "make_stage4_synthetic_fixture",
]
