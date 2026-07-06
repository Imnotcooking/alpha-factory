"""Massive/legacy Polygon options snapshot adapter for Middle Office greeks."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from oqp.data.base import AdapterHealth, OptionsDataAdapter
from oqp.data.models import OptionChainRequest, OptionQuote


class PolygonOptionsSnapshotAdapter(OptionsDataAdapter):
    """Small adapter for the existing options snapshot greeks workflow.

    Polygon.io has been renamed to Massive. The class name stays as a legacy
    import surface, while new configuration should prefer Massive env vars.
    """

    name = "massive_options_snapshot"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("MASSIVE_API_KEY")
            or os.getenv("OPTIONS_API_KEY")
            or os.getenv("POLYGON_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("MASSIVE_API_BASE_URL")
            or os.getenv("POLYGON_API_BASE_URL")
            or "https://api.massive.com"
        ).rstrip("/")

    def healthcheck(self) -> AdapterHealth:
        has_key = bool(self.api_key)
        return AdapterHealth(
            name=self.name,
            ok=has_key,
            message=(
                "Massive API key configured for options snapshots."
                if has_key
                else "Missing MASSIVE_API_KEY, OPTIONS_API_KEY, or POLYGON_API_KEY."
            ),
            metadata={"base_url": self.base_url, "implemented": True},
        )

    def build_option_snapshot_url(self, underlying: str, occ_ticker: str) -> str:
        return (
            f"{self.base_url}/v3/snapshot/options/{underlying}/{occ_ticker}"
            f"?apiKey={self.api_key}"
        )

    def get_option_snapshot(
        self,
        underlying: str,
        occ_ticker: str,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {}

        try:
            import requests

            response = requests.get(
                self.build_option_snapshot_url(underlying, occ_ticker),
                timeout=timeout,
            )
            payload = response.json()
        except Exception:
            return {}

        if payload.get("status") != "OK":
            return {}
        results = payload.get("results")
        return results if isinstance(results, dict) else {}

    def get_option_greeks(
        self,
        underlying: str,
        occ_ticker: str,
        timeout: float = 3.0,
    ) -> tuple[float, float]:
        snapshot = self.get_option_snapshot(underlying, occ_ticker, timeout=timeout)
        greeks = snapshot.get("greeks", {}) if snapshot else {}
        delta = greeks.get("delta") or 0.0
        gamma = greeks.get("gamma") or 0.0
        return float(delta), float(gamma)

    def get_option_chain(self, request: OptionChainRequest) -> Sequence[OptionQuote]:
        raise NotImplementedError("Massive option chain integration is not wired yet.")
