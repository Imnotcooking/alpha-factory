"""Deterministic clean fixtures for the Stage 2 package smoke run.

These fixtures test schemas and orchestration only.  They are not simulations
of Chinese futures and must never be reported as empirical or model evidence.
Stage 3 adds adversarial fixtures for holidays, missing rows, zero volume,
price limits, timestamps, and point-in-time rolls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd


STAGE_OWNER = 2
SCENARIO = "clean_happy_path"
SMOKE_FEATURE_COLUMNS = ("smoke_f1", "smoke_f2", "smoke_f3")
SMOKE_OUTCOME_COLUMNS = ("smoke_risk_outcome", "smoke_tail_outcome")


@dataclass(frozen=True)
class SyntheticFixtureConfig:
    """Small deterministic fixture geometry suitable for local and CI smoke runs."""

    seed: int = 42
    products: tuple[str, ...] = ("SYN_A", "SYN_B")
    periods: int = 120
    start_date: date = date(2000, 1, 3)
    state_count: int = 3
    regime_block_periods: int = 15
    scenario: str = SCENARIO

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if not isinstance(self.products, tuple):
            raise TypeError("products must be a tuple for immutable configuration.")
        if not self.products:
            raise ValueError("At least one synthetic product is required.")
        if any(not isinstance(product, str) for product in self.products):
            raise TypeError("Synthetic product identifiers must be strings.")
        if len(self.products) != len(set(self.products)):
            raise ValueError("Synthetic product identifiers must be unique.")
        if any(not product.startswith("SYN_") for product in self.products):
            raise ValueError("Synthetic product identifiers must begin with 'SYN_'.")
        if self.periods < 2:
            raise ValueError("periods must be at least two.")
        if self.state_count < 2:
            raise ValueError("state_count must be at least two.")
        if self.regime_block_periods < 1:
            raise ValueError("regime_block_periods must be positive.")
        if not isinstance(self.start_date, date):
            raise TypeError("start_date must be a datetime.date.")
        if self.scenario != SCENARIO:
            raise ValueError(
                "Stage 2 supports only the clean_happy_path fixture; adversarial "
                "scenarios belong to Stage 3."
            )


@dataclass(frozen=True)
class SyntheticFixture:
    """In-memory synthetic inputs and separately quarantined latent truth."""

    contract_rows: pd.DataFrame
    dominant_mapping: pd.DataFrame
    smoke_panel: pd.DataFrame
    truth: pd.DataFrame
    fixture_id: str
    feature_columns: tuple[str, ...] = SMOKE_FEATURE_COLUMNS
    outcome_columns: tuple[str, ...] = SMOKE_OUTCOME_COLUMNS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, frame in (
            ("contract_rows", self.contract_rows),
            ("dominant_mapping", self.dominant_mapping),
            ("smoke_panel", self.smoke_panel),
            ("truth", self.truth),
        ):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
        if not self.fixture_id.strip():
            raise ValueError("fixture_id must be non-empty.")
        if set(self.feature_columns) & set(self.truth.columns):
            raise ValueError("Fixture feature names cannot also identify truth columns.")
        if "hidden_state" in self.smoke_panel.columns:
            raise ValueError("hidden_state must remain quarantined from smoke inputs.")


@runtime_checkable
class SyntheticFixtureFactory(Protocol):
    """Protocol for deterministic, in-memory Stage 2 fixture factories."""

    @property
    def factory_id(self) -> str:
        """Stable generator identifier included in the smoke manifest."""

    def generate(self, config: SyntheticFixtureConfig) -> SyntheticFixture:
        """Generate a new fixture without reading files, databases, or the network."""


class CleanSyntheticFixtureFactory:
    """Generate the clean Stage 2 fixture; no market edge cases are represented."""

    factory_id = "clean_synthetic_fixture_v1"

    def generate(self, config: SyntheticFixtureConfig) -> SyntheticFixture:
        dates = pd.bdate_range(start=config.start_date, periods=config.periods)
        contract_frames: list[pd.DataFrame] = []
        mapping_frames: list[pd.DataFrame] = []
        smoke_frames: list[pd.DataFrame] = []
        truth_frames: list[pd.DataFrame] = []

        for product_index, product in enumerate(config.products):
            product_seed = _named_seed(config.seed, product)
            rng = np.random.default_rng(product_seed)
            state = (
                np.arange(config.periods, dtype=int) // config.regime_block_periods
                + product_index
            ) % config.state_count
            state_scale = state.astype(float) / float(config.state_count - 1)

            return_scale = 0.0025 + 0.0035 * state_scale
            close_return = rng.normal(0.0, return_scale)
            opening_gap = rng.normal(0.0, 0.35 * return_scale)
            range_padding = np.clip(
                np.abs(rng.normal(0.003 + 0.004 * state_scale, 0.0015)),
                0.0005,
                0.04,
            )

            initial_close = 100.0 + 25.0 * product_index
            close = initial_close * np.exp(np.cumsum(close_return))
            previous_close = np.concatenate(([initial_close], close[:-1]))
            open_price = previous_close * np.exp(opening_gap)
            high = np.maximum(open_price, close) * (1.0 + range_padding)
            low = np.minimum(open_price, close) * (1.0 - range_padding)
            settlement = (open_price + high + low + close) / 4.0

            volume_level = 2200.0 - 700.0 * state_scale
            volume = np.maximum(
                1,
                np.rint(volume_level + rng.normal(0.0, 80.0, config.periods)),
            ).astype(np.int64)
            open_interest = np.maximum(
                1,
                np.rint(
                    8000.0
                    + 250.0 * product_index
                    + np.cumsum(rng.normal(0.0, 12.0, config.periods))
                ),
            ).astype(np.int64)
            multiplier = float(10 + 5 * product_index)
            turnover = close * volume.astype(float) * multiplier
            contract = f"{product}0001"

            contract_frame = pd.DataFrame(
                {
                    "product": product,
                    "contract": contract,
                    "exchange": "SYN",
                    "trading_date": dates,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "settlement": settlement,
                    "volume": volume,
                    "turnover": turnover,
                    "open_interest": open_interest,
                    "multiplier": multiplier,
                    "tick_size": 0.01,
                    "roll_flag": False,
                    "limit_lock_flag": False,
                    "stale_bar_flag": False,
                    "source_row_id": [
                        f"{product}:{stamp:%Y%m%d}:{contract}" for stamp in dates
                    ],
                    "listing_date": dates[0] - pd.Timedelta(days=365),
                    "last_trade_date": dates[-1] + pd.Timedelta(days=365),
                }
            )
            contract_frames.append(contract_frame)

            mapping_frames.append(
                pd.DataFrame(
                    {
                        "product": product,
                        "trading_date": dates,
                        "decision_date": dates - pd.Timedelta(days=1),
                        "selected_contract": contract,
                        "roll_flag": False,
                    }
                )
            )

            smoke_frames.append(
                pd.DataFrame(
                    {
                        "product": product,
                        "trading_date": dates,
                        "information_date": dates,
                        "contract": contract,
                        "smoke_f1": state_scale
                        + rng.normal(0.0, 0.08, config.periods),
                        "smoke_f2": 1.0 - state_scale
                        + rng.normal(0.0, 0.08, config.periods),
                        "smoke_f3": np.sin(np.arange(config.periods) / 9.0)
                        + 0.25 * state_scale
                        + rng.normal(0.0, 0.06, config.periods),
                        "smoke_risk_outcome": np.square(close_return)
                        + np.square(opening_gap)
                        + 1e-10,
                        "smoke_tail_outcome": (
                            close_return < -(0.0035 + 0.0015 * state_scale)
                        ).astype(np.int8),
                    }
                )
            )
            truth_frames.append(
                pd.DataFrame(
                    {
                        "product": product,
                        "trading_date": dates,
                        "hidden_state": state,
                    }
                )
            )

        fixture = SyntheticFixture(
            contract_rows=_ordered_concat(contract_frames, ["product", "trading_date"]),
            dominant_mapping=_ordered_concat(mapping_frames, ["product", "trading_date"]),
            smoke_panel=_ordered_concat(smoke_frames, ["product", "trading_date"]),
            truth=_ordered_concat(truth_frames, ["product", "trading_date"]),
            fixture_id=f"synthetic_smoke_{_config_fingerprint(config)}",
            metadata={
                "synthetic": True,
                "scientific_evidence": False,
                "scenario": config.scenario,
                "contains_market_data": False,
                "contains_edge_cases": False,
                "factory_id": self.factory_id,
                "seed": config.seed,
            },
        )
        validate_synthetic_fixture(fixture, config=config)
        return fixture


def make_clean_synthetic_fixture(
    config: SyntheticFixtureConfig | None = None,
) -> SyntheticFixture:
    """Convenience wrapper for the deterministic in-memory fixture factory."""

    return CleanSyntheticFixtureFactory().generate(config or SyntheticFixtureConfig())


def validate_synthetic_fixture(
    fixture: SyntheticFixture,
    *,
    config: SyntheticFixtureConfig,
) -> None:
    """Check properties promised by the clean generator itself.

    This is intentionally narrower than the Stage 3 market-data validator.
    """

    expected_rows = len(config.products) * config.periods
    for name, frame in (
        ("contract_rows", fixture.contract_rows),
        ("dominant_mapping", fixture.dominant_mapping),
        ("smoke_panel", fixture.smoke_panel),
        ("truth", fixture.truth),
    ):
        if len(frame) != expected_rows:
            raise ValueError(f"{name} has {len(frame)} rows; expected {expected_rows}.")
        if frame.duplicated(["product", "trading_date"]).any():
            raise ValueError(f"{name} contains duplicate product-date rows.")

    bars = fixture.contract_rows
    numeric_bar_columns = ["open", "high", "low", "close", "settlement"]
    numeric_bars = bars.loc[:, numeric_bar_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_bars).all() or (numeric_bars <= 0.0).any():
        raise ValueError("Synthetic OHLC and settlement values must be finite and positive.")
    if (bars["high"] < bars[["open", "close"]].max(axis=1)).any():
        raise ValueError("Synthetic high must cover open and close.")
    if (bars["low"] > bars[["open", "close"]].min(axis=1)).any():
        raise ValueError("Synthetic low must cover open and close.")
    if (bars[["volume", "open_interest"]] <= 0).any(axis=None):
        raise ValueError("The clean fixture requires positive volume and open interest.")

    mapping = fixture.dominant_mapping
    if (mapping["decision_date"] >= mapping["trading_date"]).any():
        raise ValueError("Synthetic mapping decisions must precede their trading date.")
    selected = mapping.merge(
        bars[["product", "trading_date", "contract"]],
        on=["product", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    if not (selected["selected_contract"] == selected["contract"]).all():
        raise ValueError("Synthetic selected contracts must exist in contract rows.")

    smoke_values = fixture.smoke_panel.loc[
        :, fixture.feature_columns + fixture.outcome_columns
    ].to_numpy(dtype=float)
    if not np.isfinite(smoke_values).all():
        raise ValueError("Smoke inputs and outcomes must be finite.")
    if "hidden_state" in fixture.smoke_panel.columns:
        raise ValueError("Hidden fixture state leaked into the smoke panel.")


def _named_seed(seed: int, name: str) -> int:
    payload = f"{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _config_fingerprint(config: SyntheticFixtureConfig) -> str:
    payload = "|".join(
        (
            str(config.seed),
            ",".join(config.products),
            str(config.periods),
            config.start_date.isoformat(),
            str(config.state_count),
            str(config.regime_block_periods),
            config.scenario,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _ordered_concat(frames: list[pd.DataFrame], sort_columns: list[str]) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = [
    "CleanSyntheticFixtureFactory",
    "SCENARIO",
    "SMOKE_FEATURE_COLUMNS",
    "SMOKE_OUTCOME_COLUMNS",
    "STAGE_OWNER",
    "SyntheticFixture",
    "SyntheticFixtureConfig",
    "SyntheticFixtureFactory",
    "make_clean_synthetic_fixture",
    "validate_synthetic_fixture",
]
