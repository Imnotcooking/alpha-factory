"""Research execution-mode policies for transforming signals into weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oqp.research.backtesting.portfolio_optimizer import PortfolioOptimizer


@dataclass(frozen=True)
class ExecutionModeResult:
    df: pd.DataFrame
    mode: str
    signal_col: str
    source_col: str
    detail: str


@dataclass(frozen=True)
class ExecutionModeConfig:
    max_gross_leverage: float = 1.0
    max_weight_per_asset: float | None = None
    source_col: str | None = None
    neutralize: bool = False
    source_candidates: tuple[str, ...] = field(
        default_factory=lambda: (
            "target_weight",
            "final_target_weight",
            "desired_weight",
            "signal",
            "factor_score",
            "raw_signal",
        )
    )


class BaseExecutionMode:
    name = "base"

    def __init__(self, config: ExecutionModeConfig | None = None):
        self.config = config or ExecutionModeConfig()

    def apply(self, df: pd.DataFrame) -> ExecutionModeResult:
        raise NotImplementedError

    def _select_source_col(
        self,
        df: pd.DataFrame,
        candidates: Iterable[str] | None = None,
    ) -> str:
        if self.config.source_col:
            if self.config.source_col not in df.columns:
                raise ValueError(
                    f"Execution mode {self.name!r} requires "
                    f"source_col={self.config.source_col!r}."
                )
            return self.config.source_col
        for col in candidates or self.config.source_candidates:
            if col in df.columns:
                return col
        raise ValueError(
            f"Execution mode {self.name!r} requires one of: "
            f"{', '.join(self.config.source_candidates)}."
        )


class RiskDeskExecutionMode(BaseExecutionMode):
    """Kelly + HRP + portfolio-cap path, preserved as the default mode."""

    name = "risk_desk"

    def __init__(
        self,
        config: ExecutionModeConfig | None = None,
        *,
        kelly_fraction: float = 0.5,
        max_weight: float = 0.05,
    ):
        super().__init__(config)
        self.optimizer = PortfolioOptimizer(
            kelly_fraction=kelly_fraction,
            max_weight=max_weight,
        )

    def apply(self, df: pd.DataFrame) -> ExecutionModeResult:
        out = df.copy()
        alpha_col = self._select_source_col(
            out,
            candidates=("signal", "factor_score", "raw_signal"),
        )
        out["risk_signal"] = out[alpha_col].fillna(0.0)
        out = self.optimizer.kelly.compute_weights(out, signal_col="risk_signal")

        if "ret_1d" not in out.columns:
            out["ret_1d"] = out.groupby("ticker")["close"].pct_change()
        returns_wide = (
            out.pivot(index="date", columns="ticker", values="ret_1d")
            .fillna(0.0)
            .astype(float)
        )
        global_hrp_budgets = self.optimizer.hrp.compute_weights(returns_wide)
        out["hrp_budget"] = out["ticker"].map(global_hrp_budgets).fillna(0.0)

        out["synthesized_weight"] = out["kelly_weight"] * out["hrp_budget"]
        out = self.optimizer.cap.enforce(out, "synthesized_weight")
        out["signal"] = out["final_target_weight"]
        out["execution_mode"] = self.name
        out.attrs["execution_mode"] = self.name
        return ExecutionModeResult(
            df=out,
            mode=self.name,
            signal_col="signal",
            source_col=alpha_col,
            detail="Kelly sizing + HRP budgets + portfolio cap.",
        )


class DirectExecutionMode(BaseExecutionMode):
    """Trust factor-owned target weights; apply optional clipping/gross cap."""

    name = "direct"

    def apply(self, df: pd.DataFrame) -> ExecutionModeResult:
        out = df.copy()
        source_col = self._select_source_col(out)
        out["pre_execution_weight"] = pd.to_numeric(
            out[source_col],
            errors="coerce",
        ).fillna(0.0)
        if self.config.neutralize:
            out["pre_execution_weight"] = out.groupby("date")[
                "pre_execution_weight"
            ].transform(lambda series: series - series.mean())
        out["final_target_weight"] = _cap_and_scale_weights(
            out,
            "pre_execution_weight",
            max_gross_leverage=self.config.max_gross_leverage,
            max_weight_per_asset=self.config.max_weight_per_asset,
        )
        out["signal"] = out["final_target_weight"]
        out["target_weight"] = out["final_target_weight"]
        out["execution_mode"] = self.name
        out.attrs["execution_mode"] = self.name
        return ExecutionModeResult(
            df=out,
            mode=self.name,
            signal_col="signal",
            source_col=source_col,
            detail="Factor-owned weights with optional neutralization and gross scaling.",
        )


class StatArbExecutionMode(DirectExecutionMode):
    """Preserve factor-owned stat-arb spread or basket ratios."""

    name = "statarb"

    def apply(self, df: pd.DataFrame) -> ExecutionModeResult:
        result = super().apply(df)
        result.df["execution_mode"] = self.name
        result.df.attrs["execution_mode"] = self.name
        return ExecutionModeResult(
            df=result.df,
            mode=self.name,
            signal_col=result.signal_col,
            source_col=result.source_col,
            detail="StatArb mode: preserved factor-owned leg ratios; no Kelly/HRP.",
        )


class ExecutionModeFactory:
    @staticmethod
    def create(
        mode: str,
        config: ExecutionModeConfig | None = None,
    ) -> BaseExecutionMode:
        value = (mode or "risk_desk").strip().lower().replace("-", "_")
        if value in {"risk", "riskdesk", "risk_desk"}:
            return RiskDeskExecutionMode(config=config)
        if value in {"direct", "factor_owned"}:
            return DirectExecutionMode(config=config)
        if value in {"statarb", "stat_arb", "pairs", "pair"}:
            return StatArbExecutionMode(config=config)
        raise ValueError(f"Unknown execution mode: {mode!r}")


def _cap_and_scale_weights(
    df: pd.DataFrame,
    weight_col: str,
    *,
    max_gross_leverage: float = 1.0,
    max_weight_per_asset: float | None = None,
) -> pd.Series:
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).astype(float)
    if max_weight_per_asset is not None and max_weight_per_asset > 0:
        weights = weights.clip(
            -float(max_weight_per_asset),
            float(max_weight_per_asset),
        )

    max_gross = max(float(max_gross_leverage), 0.0)
    if max_gross <= 0:
        return pd.Series(0.0, index=df.index)

    gross = weights.abs().groupby(df["date"]).transform("sum")
    safe_gross = gross.replace(0.0, np.nan)
    shrink = np.where(gross > max_gross, max_gross / safe_gross, 1.0)
    shrink = (
        pd.Series(shrink, index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return weights * shrink


__all__ = [
    "BaseExecutionMode",
    "DirectExecutionMode",
    "ExecutionModeConfig",
    "ExecutionModeFactory",
    "ExecutionModeResult",
    "RiskDeskExecutionMode",
    "StatArbExecutionMode",
]
