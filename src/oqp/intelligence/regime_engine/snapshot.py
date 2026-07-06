"""Dashboard-friendly regime snapshot engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine, EngineStatus
from oqp.intelligence.context import EngineContext


class RegimeSnapshotEngine(BaseEngine):
    """Advisory regime reader using HMM-compatible return/vol emissions."""

    engine_id = "regime_snapshot"
    engine_name = "Regime Snapshot"
    category = "regime_engine"
    version = "0.1.0"

    def run(self, context: EngineContext):
        rows = []
        for account in ("live", "paper"):
            emissions = _nav_emissions(context.nav_history(account))
            rows.append(_regime_row(account.title(), emissions))

        frame = pd.DataFrame(rows)
        missing = int(frame["State"].eq("insufficient_history").sum()) if not frame.empty else 2
        status = EngineStatus.SKIPPED if missing == len(frame) else EngineStatus.WARN if missing else EngineStatus.PASS
        summary = (
            "Regime snapshot waiting for more NAV history."
            if status is EngineStatus.SKIPPED
            else "Regime snapshot available from return/volatility emissions."
        )
        return self.result(
            status=status,
            summary=summary,
            frames={"regimes": frame},
            metrics={"accounts": len(frame), "insufficient_history": missing},
            signals={
                row["Account"].lower(): row["State"]
                for row in rows
                if row["State"] != "insufficient_history"
            },
            metadata={"emissions": ["returns", "volatility"], "state_order": "quiet/chop/panic"},
        )


def _nav_emissions(nav_history: pd.DataFrame) -> pd.DataFrame:
    if nav_history.empty or "net_liquidation" not in nav_history:
        return pd.DataFrame(columns=["returns", "volatility"])

    out = nav_history.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    nav = pd.to_numeric(out["net_liquidation"], errors="coerce")
    returns = (
        pd.to_numeric(out["daily_return"], errors="coerce")
        if "daily_return" in out
        else nav.pct_change()
    )
    emissions = pd.DataFrame(
        {
            "date": out["date"],
            "returns": returns.fillna(0.0),
            "volatility": returns.rolling(20, min_periods=3).std().fillna(0.0),
        }
    )
    return emissions.dropna(subset=["date"]).reset_index(drop=True)


def _regime_row(account: str, emissions: pd.DataFrame) -> dict[str, Any]:
    base = {
        "Account": account,
        "State": "insufficient_history",
        "Aligned State": None,
        "Latest Return": None,
        "Latest Volatility": None,
        "Vol Percentile": None,
        "As Of": "missing",
    }
    if len(emissions) < 5:
        return base

    latest = emissions.iloc[-1]
    vol = float(latest["volatility"])
    vol_series = pd.to_numeric(emissions["volatility"], errors="coerce").dropna()
    if vol_series.empty:
        return base
    percentile = float((vol_series <= vol).mean())
    if percentile < 1 / 3:
        state = "quiet"
        aligned = 0
    elif percentile < 2 / 3:
        state = "chop"
        aligned = 1
    else:
        state = "panic"
        aligned = 2

    return {
        "Account": account,
        "State": state,
        "Aligned State": aligned,
        "Latest Return": float(latest["returns"]),
        "Latest Volatility": vol,
        "Vol Percentile": percentile,
        "As Of": str(latest["date"]),
    }
