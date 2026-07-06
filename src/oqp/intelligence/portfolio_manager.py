"""Portfolio-manager command engine for approved strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from oqp.intelligence.base import BaseEngine, EngineStatus
from oqp.intelligence.context import EngineContext


class PortfolioManagerEngine(BaseEngine):
    """Decide runtime posture for strategies that are already approved."""

    engine_id = "portfolio_manager"
    engine_name = "Portfolio Manager"
    category = "portfolio_manager"
    version = "0.1.0"

    def run(self, context: EngineContext):
        roster = _strategy_roster(context)
        signals = _strategy_signals(context)
        if roster.empty:
            requirements = pd.DataFrame(
                [
                    {
                        "Input": "approved_strategies",
                        "Status": "missing",
                        "Detail": "Approved strategy roster from research/paper governance.",
                    },
                    {
                        "Input": "strategy_signals",
                        "Status": "optional",
                        "Detail": "Runtime trigger signals from approved strategies.",
                    },
                    {
                        "Input": "account_state",
                        "Status": "available",
                        "Detail": "Live and paper account context is already in EngineContext.",
                    },
                ]
            )
            return self.result(
                status=EngineStatus.SKIPPED,
                summary="Portfolio manager is waiting for approved strategies.",
                frames={"requirements": requirements},
                metrics={"approved_strategies": 0},
                signals={"runtime_role": "post_approval_command_center"},
            )

        decisions = _runtime_decisions(roster, signals, context)
        status = _status_from_decisions(decisions)
        return self.result(
            status=status,
            summary=_summary(status, decisions),
            frames={"runtime_decisions": decisions},
            metrics={
                "approved_strategies": int(len(decisions)),
                "triggerable": int(decisions["Runtime Decision"].eq("triggerable").sum()),
                "paused": int(decisions["Runtime Decision"].str.contains("paused|locked", regex=True).sum()),
            },
            signals={
                "runtime_role": "post_approval_command_center",
                "triggerable_strategy_ids": decisions.loc[
                    decisions["Runtime Decision"].eq("triggerable"),
                    "Strategy ID",
                ].tolist(),
            },
        )


def _strategy_roster(context: EngineContext) -> pd.DataFrame:
    if not context.approved_strategies.empty:
        return context.approved_strategies.copy()
    value = context.metadata.get("approved_strategies")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _strategy_signals(context: EngineContext) -> pd.DataFrame:
    if not context.strategy_signals.empty:
        return context.strategy_signals.copy()
    value = context.metadata.get("strategy_signals")
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _runtime_decisions(
    roster: pd.DataFrame,
    signals: pd.DataFrame,
    context: EngineContext,
) -> pd.DataFrame:
    signal_lookup = _signal_lookup(signals)
    allow_paper = bool(context.settings.get("allow_paper_trading", False))
    allow_live = bool(context.settings.get("allow_live_trading", False))
    rows: list[dict[str, Any]] = []

    for _, raw in roster.iterrows():
        row = raw.to_dict()
        strategy_id = _first_text(row, "strategy_id", "factor_id", "id", default="unknown")
        environment = _first_text(
            row,
            "target_environment",
            "environment",
            "account_environment",
            default="paper",
        ).lower()
        approval_state = _first_text(row, "approval_state", "status", default="approved")
        vertical = _first_text(row, "market_vertical", "vertical", "asset_universe", default="unspecified")
        signal = signal_lookup.get(strategy_id, {})
        signal_active = bool(signal.get("active", False))
        signal_strength = signal.get("strength")

        if environment == "live" and not allow_live:
            decision = "live_locked"
            next_action = "Keep strategy out of live execution until live trading is explicitly enabled."
        elif environment == "paper" and not allow_paper:
            decision = "paper_paused"
            next_action = "Enable paper trading gate before runtime triggers can create proposals."
        elif signal_active:
            decision = "triggerable"
            next_action = "Strategy may create a paper proposal subject to sizing and safety review."
        else:
            decision = "waiting_for_signal"
            next_action = "No runtime trigger yet; keep monitoring regime, cash, risk, and signal state."

        rows.append(
            {
                "Strategy ID": strategy_id,
                "Market Vertical": vertical,
                "Target Environment": environment,
                "Approval State": approval_state,
                "Signal Active": signal_active,
                "Signal Strength": signal_strength,
                "Runtime Decision": decision,
                "Next Action": next_action,
            }
        )

    return pd.DataFrame(rows)


def _signal_lookup(signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if signals.empty:
        return {}
    out = signals.copy()
    if "strategy_id" not in out.columns and "factor_id" in out.columns:
        out["strategy_id"] = out["factor_id"]
    if "strategy_id" not in out.columns:
        return {}
    return {
        str(row["strategy_id"]): row.to_dict()
        for _, row in out.iterrows()
    }


def _status_from_decisions(decisions: pd.DataFrame) -> EngineStatus:
    if decisions.empty:
        return EngineStatus.SKIPPED
    if decisions["Runtime Decision"].eq("triggerable").any():
        return EngineStatus.PASS
    return EngineStatus.WARN


def _summary(status: EngineStatus, decisions: pd.DataFrame) -> str:
    if status is EngineStatus.PASS:
        count = int(decisions["Runtime Decision"].eq("triggerable").sum())
        return f"{count} approved strategy runtime trigger(s) are active."
    return "Approved strategies are present, but none are triggerable right now."


def _first_text(row: dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return default
