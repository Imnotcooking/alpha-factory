"""Risk control room engine for live and paper account monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oqp.accounts import account_top_positions
from oqp.intelligence.base import BaseEngine, EngineResult, EngineStatus
from oqp.intelligence.context import EngineContext


@dataclass(frozen=True, slots=True)
class RiskControlRoomConfig:
    max_gross_nav: float = 1.25
    min_cash_weight: float = 0.03
    max_position_weight: float = 0.25
    max_drawdown_pct: float = -0.15


class RiskControlRoomEngine(BaseEngine):
    """Read-only account risk monitor for the Ops dashboard."""

    engine_id = "risk_control_room"
    engine_name = "Risk Control Room"
    category = "risk_engine"
    version = "0.1.0"

    def __init__(self, config: RiskControlRoomConfig | None = None) -> None:
        self.config = config or RiskControlRoomConfig()

    def run(self, context: EngineContext) -> EngineResult:
        summary = self._summary_frame(context)
        concentration = self._concentration_frame(context)
        risk_flags = self._risk_flags_frame(summary, concentration)
        status = self._status_from_flags(risk_flags)
        message = self._message(status, risk_flags)

        return self.result(
            status=status,
            summary=message,
            metrics={
                "accounts": int(len(summary)),
                "flag_count": int(len(risk_flags)),
                "warning_count": int(
                    risk_flags["Severity"].eq("warn").sum()
                    if not risk_flags.empty
                    else 0
                ),
                "block_count": int(
                    risk_flags["Severity"].eq("block").sum()
                    if not risk_flags.empty
                    else 0
                ),
            },
            frames={
                "summary": summary,
                "concentration": concentration,
                "risk_flags": risk_flags,
            },
            signals={
                "risk_state": status.value,
                "can_escalate_to_live": status is EngineStatus.PASS,
            },
            metadata={"thresholds": self._thresholds()},
        )

    def _summary_frame(self, context: EngineContext) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for account in ("live", "paper"):
            account_summary = dict(context.account_summary(account))
            performance = dict(account_summary.get("performance") or {})
            nav = _first_number(account_summary.get("nav"), performance.get("latest_nav"))
            cash = _first_number(account_summary.get("cash"), performance.get("latest_cash"))
            gross = _first_number(performance.get("gross_exposure"), 0.0)
            gross_pct = _ratio(gross, nav)
            cash_pct = _ratio(cash, nav)
            rows.append(
                {
                    "Account": account.title(),
                    "Source": account_summary.get("source", "missing"),
                    "NAV": nav,
                    "Cash": cash,
                    "Cash %": cash_pct,
                    "Daily P&L": _first_number(
                        account_summary.get("daily_pnl"),
                        performance.get("daily_pnl"),
                    ),
                    "Gross Exposure": gross,
                    "Gross / NAV": _first_number(
                        performance.get("gross_exposure_pct"),
                        gross_pct,
                    ),
                    "Max Drawdown": _first_number(
                        performance.get("max_drawdown_pct"),
                    ),
                    "Positions": int(account_summary.get("position_count") or 0),
                    "As Of": account_summary.get("as_of") or "missing",
                }
            )
        return pd.DataFrame(rows)

    def _concentration_frame(self, context: EngineContext) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for account in ("live", "paper"):
            summary = dict(context.account_summary(account))
            performance = dict(summary.get("performance") or {})
            nav = _first_number(summary.get("nav"), performance.get("latest_nav"))
            top = account_top_positions(context.positions(account), limit=10)
            for _, row in top.iterrows():
                market_value = _first_number(row.get("market_value"), 0.0) or 0.0
                rows.append(
                    {
                        "Account": account.title(),
                        "Symbol": row.get("symbol"),
                        "Asset Class": row.get("asset_class"),
                        "Market Value": market_value,
                        "Weight": _ratio(abs(market_value), nav),
                        "Unrealized P&L": _first_number(row.get("unrealized_pnl")),
                        "Realized P&L": _first_number(row.get("realized_pnl")),
                    }
                )
        return pd.DataFrame(rows)

    def _risk_flags_frame(
        self,
        summary: pd.DataFrame,
        concentration: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in summary.iterrows():
            account = str(row["Account"])
            nav = _first_number(row.get("NAV"))
            if nav is None or nav <= 0:
                rows.append(_flag(account, "NAV missing", "warn", "No positive NAV is available."))

            gross_nav = _first_number(row.get("Gross / NAV"))
            if gross_nav is not None and gross_nav > self.config.max_gross_nav:
                rows.append(
                    _flag(
                        account,
                        "Gross exposure",
                        "warn",
                        f"{gross_nav:.2f} is above {self.config.max_gross_nav:.2f}.",
                    )
                )

            cash_pct = _first_number(row.get("Cash %"))
            if cash_pct is not None and cash_pct < self.config.min_cash_weight:
                rows.append(
                    _flag(
                        account,
                        "Cash buffer",
                        "warn",
                        f"{cash_pct:.2%} is below {self.config.min_cash_weight:.2%}.",
                    )
                )

            max_drawdown = _first_number(row.get("Max Drawdown"))
            if max_drawdown is not None and max_drawdown < self.config.max_drawdown_pct:
                rows.append(
                    _flag(
                        account,
                        "Drawdown",
                        "warn",
                        f"{max_drawdown:.2%} is below {self.config.max_drawdown_pct:.2%}.",
                    )
                )

        if not concentration.empty and "Weight" in concentration:
            for _, row in concentration.iterrows():
                weight = _first_number(row.get("Weight"))
                if weight is not None and weight > self.config.max_position_weight:
                    rows.append(
                        _flag(
                            str(row["Account"]),
                            "Position concentration",
                            "warn",
                            f"{row.get('Symbol')} is {weight:.2%} of NAV.",
                        )
                    )

        return pd.DataFrame(rows, columns=["Account", "Check", "Severity", "Detail"])

    def _status_from_flags(self, risk_flags: pd.DataFrame) -> EngineStatus:
        if risk_flags.empty:
            return EngineStatus.PASS
        if risk_flags["Severity"].eq("block").any():
            return EngineStatus.FAIL
        return EngineStatus.WARN

    @staticmethod
    def _message(status: EngineStatus, risk_flags: pd.DataFrame) -> str:
        if status is EngineStatus.PASS:
            return "No risk control room flags."
        return f"{len(risk_flags)} risk control room flag(s) need review."

    def _thresholds(self) -> dict[str, float]:
        return {
            "max_gross_nav": self.config.max_gross_nav,
            "min_cash_weight": self.config.min_cash_weight,
            "max_position_weight": self.config.max_position_weight,
            "max_drawdown_pct": self.config.max_drawdown_pct,
        }


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not pd.isna(parsed):
            return parsed
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _flag(account: str, check: str, severity: str, detail: str) -> dict[str, str]:
    return {
        "Account": account,
        "Check": check,
        "Severity": severity,
        "Detail": detail,
    }
