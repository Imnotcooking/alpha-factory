from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from oqp.data.vendors import MassiveOptionsDataAdapter
from oqp.research.api_datasets import (
    ApiDatasetQualityError,
    ApiDatasetRequest,
    HistoricalBacktestEligibilityError,
    load_materialized_api_dataset,
    materialize_fmp_us_equity_daily,
    materialize_massive_us_option_snapshot,
)
from oqp.research.dataset_fingerprints import DatasetFingerprintError
from oqp.research.factor_portfolios.data import load_factor_portfolio_data


RETRIEVED_AT = datetime(2026, 7, 21, 2, 30, tzinfo=UTC)


class FakeFmpAdapter:
    def __init__(self, *, empty: bool = False):
        self.empty = empty
        self.calls: list[tuple[str, bool, dict]] = []

    def get_json(self, endpoint: str, *, stable: bool, params: dict):
        self.calls.append((endpoint, stable, dict(params)))
        if self.empty:
            return []
        base = 100.0 if params["symbol"] == "AAPL" else 200.0
        return [
            {
                "date": day,
                "open": base + offset,
                "high": base + offset + 2.0,
                "low": base + offset - 1.0,
                "close": base + offset + 1.0,
                "adjClose": base + offset + 0.5,
                "volume": 1_000_000 + offset,
                "vwap": base + offset + 0.4,
            }
            for offset, day in enumerate(
                ("2026-07-16", "2026-07-17", "2026-07-20")
            )
        ]


class FakeMassiveAdapter(MassiveOptionsDataAdapter):
    def __init__(self):
        super().__init__(api_key="test-key")

    def get_option_snapshot_rows(self, underlying: str, **kwargs):
        timestamp_ns = int(
            datetime(2026, 7, 21, 2, 0, tzinfo=UTC).timestamp() * 1_000_000_000
        )
        return [
            {
                "details": {
                    "contract_type": "call",
                    "expiration_date": "2026-08-21",
                    "strike_price": 200.0,
                    "ticker": f"O:{underlying}260821C00200000",
                    "shares_per_contract": 100,
                    "primary_exchange": "XNAS",
                },
                "last_quote": {
                    "bid": 5.0,
                    "ask": 5.2,
                    "last_updated": timestamp_ns,
                },
                "last_trade": {"price": 5.1, "sip_timestamp": timestamp_ns},
                "day": {
                    "open": 4.8,
                    "high": 5.3,
                    "low": 4.7,
                    "close": 5.1,
                    "volume": 250,
                    "last_updated": timestamp_ns,
                },
                "greeks": {
                    "delta": 0.51,
                    "gamma": 0.04,
                    "theta": -0.08,
                    "vega": 0.12,
                },
                "implied_volatility": 0.31,
                "open_interest": 1_200,
            }
        ]


def _roots(tmp_path: Path) -> dict[str, Path]:
    return {
        "storage_root": tmp_path / "data",
        "manifest_root": tmp_path / "manifests",
        "workspace_root": tmp_path,
    }


def test_request_spec_redacts_credentials() -> None:
    request = ApiDatasetRequest(
        provider="fmp",
        dataset_id="us_equity_test",
        market_vertical="EQUITY_US",
        data_frequency="daily",
        endpoint="historical-price-eod/full",
        symbols=("aapl",),
        params={
            "apiKey": "do-not-write-this",
            "nested": {"access_token": "also-secret", "limit": 10},
        },
    )
    text = json.dumps(request.to_dict())

    assert "do-not-write-this" not in text
    assert "also-secret" not in text
    assert request.to_dict()["params"]["apiKey"] == "<redacted>"
    assert request.symbols == ("AAPL",)


def test_fmp_materialization_is_immutable_verified_and_backtest_loadable(
    tmp_path: Path,
) -> None:
    adapter = FakeFmpAdapter()
    bundle = materialize_fmp_us_equity_daily(
        ["MSFT", "AAPL", "AAPL"],
        start_date="2026-07-16",
        end_date="2026-07-20",
        adjustment_method="dividend_adjusted",
        adapter=adapter,
        retrieved_at=RETRIEVED_AT,
        **_roots(tmp_path),
    )

    assert bundle.historical_backtest_eligible
    assert bundle.dataset_version.startswith("sha256:")
    assert bundle.data_path.exists()
    assert bundle.raw_response_path.exists()
    assert bundle.manifest_path.exists()
    assert [call[2]["symbol"] for call in adapter.calls] == ["AAPL", "MSFT"]

    frame = load_materialized_api_dataset(bundle.root, workspace_root=tmp_path)
    assert len(frame) == 6
    assert frame.attrs["dataset_verified"] is True
    assert frame.attrs["dataset_fingerprint"] == bundle.dataset_fingerprint
    assert frame.attrs["data_vendor"] == "fmp"

    portfolio_data = load_factor_portfolio_data(
        bundle.root,
        market_vertical="EQUITY_US",
        return_horizon="close_to_close",
        workspace_root=tmp_path,
    )
    assert portfolio_data.frame.attrs["dataset_fingerprint"] == bundle.dataset_fingerprint
    assert portfolio_data.frame.attrs["dataset_verified"] is True


def test_mutating_a_saved_api_response_invalidates_bundle(tmp_path: Path) -> None:
    bundle = materialize_fmp_us_equity_daily(
        ["AAPL"],
        adapter=FakeFmpAdapter(),
        retrieved_at=RETRIEVED_AT,
        **_roots(tmp_path),
    )
    with bundle.raw_response_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(DatasetFingerprintError, match="size_changed"):
        load_materialized_api_dataset(bundle.root, workspace_root=tmp_path)


def test_empty_fmp_response_is_not_published(tmp_path: Path) -> None:
    with pytest.raises(ApiDatasetQualityError, match="quality checks"):
        materialize_fmp_us_equity_daily(
            ["MISSING"],
            adapter=FakeFmpAdapter(empty=True),
            retrieved_at=RETRIEVED_AT,
            **_roots(tmp_path),
        )
    assert not list((tmp_path / "data").rglob("materialization.json"))


def test_massive_current_snapshot_is_frozen_but_rejected_by_historical_loader(
    tmp_path: Path,
) -> None:
    bundle = materialize_massive_us_option_snapshot(
        ["AAPL"],
        adapter=FakeMassiveAdapter(),
        retrieved_at=RETRIEVED_AT,
        **_roots(tmp_path),
    )

    assert not bundle.historical_backtest_eligible
    with pytest.raises(HistoricalBacktestEligibilityError, match="current-only"):
        load_materialized_api_dataset(bundle.root, workspace_root=tmp_path)

    frame = load_materialized_api_dataset(
        bundle.root,
        require_historical_backtest_eligible=False,
        workspace_root=tmp_path,
    )
    assert len(frame) == 1
    assert frame.loc[0, "underlying_symbol"] == "AAPL"
    assert frame.attrs["dataset_verified"] is True
    assert frame.attrs["historical_backtest_eligible"] is False
