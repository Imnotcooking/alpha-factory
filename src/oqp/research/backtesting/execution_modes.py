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
    sizing_modules: tuple[str, ...] | str = ()
    kelly_fraction: float = 0.5
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sizing_modules",
            _normalize_sizing_modules(self.sizing_modules),
        )
        object.__setattr__(
            self,
            "max_gross_leverage",
            max(float(self.max_gross_leverage), 0.0),
        )
        if self.max_weight_per_asset is not None:
            object.__setattr__(
                self,
                "max_weight_per_asset",
                max(float(self.max_weight_per_asset), 0.0),
            )
        object.__setattr__(self, "kelly_fraction", max(float(self.kelly_fraction), 0.0))


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
    """Modular risk allocation stage after factor signal computation."""

    name = "risk_desk"

    def __init__(
        self,
        config: ExecutionModeConfig | None = None,
        *,
        kelly_fraction: float | None = None,
        max_weight: float | None = None,
    ):
        super().__init__(config)
        resolved_kelly_fraction = (
            self.config.kelly_fraction if kelly_fraction is None else float(kelly_fraction)
        )
        resolved_max_weight = (
            self.config.max_weight_per_asset
            if self.config.max_weight_per_asset is not None
            else 0.05
        )
        if max_weight is not None:
            resolved_max_weight = float(max_weight)
        self.optimizer = PortfolioOptimizer(
            kelly_fraction=resolved_kelly_fraction,
            max_weight=resolved_max_weight,
            max_gross_leverage=self.config.max_gross_leverage,
        )

    def apply(self, df: pd.DataFrame) -> ExecutionModeResult:
        out = df.copy()
        alpha_col = self._select_source_col(
            out,
            candidates=("signal", "factor_score", "raw_signal"),
        )
        modules = set(self.config.sizing_modules)
        out["risk_signal"] = pd.to_numeric(out[alpha_col], errors="coerce").fillna(0.0)
        current_weight_col = "risk_signal"

        if "kelly" in modules:
            out = self.optimizer.kelly.compute_weights(out, signal_col=current_weight_col)
            current_weight_col = "kelly_weight"

        if "hrp" in modules:
            if "ret_1d" not in out.columns:
                out["ret_1d"] = out.groupby("ticker")["close"].pct_change()
            returns_wide = (
                out.pivot(index="date", columns="ticker", values="ret_1d")
                .fillna(0.0)
                .astype(float)
            )
            global_hrp_budgets = self.optimizer.hrp.compute_weights(returns_wide)
            out["hrp_budget"] = out["ticker"].map(global_hrp_budgets).fillna(0.0)
            out["allocated_weight"] = out[current_weight_col] * out["hrp_budget"]
            current_weight_col = "allocated_weight"

        out["synthesized_weight"] = pd.to_numeric(
            out[current_weight_col],
            errors="coerce",
        ).fillna(0.0)
        out = self.optimizer.cap.enforce(out, "synthesized_weight")
        out["signal"] = out["final_target_weight"]
        out["execution_mode"] = self.name
        out.attrs["execution_mode"] = self.name
        detail = _risk_desk_detail(self.config.sizing_modules)
        return ExecutionModeResult(
            df=out,
            mode=self.name,
            signal_col="signal",
            source_col=alpha_col,
            detail=detail,
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


def _normalize_sizing_modules(value: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized_text = value.replace("+", ",").replace("|", ",").replace(";", ",")
        raw_items = [item.strip() for item in normalized_text.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]

    aliases = {
        "": "",
        "none": "",
        "off": "",
        "raw": "",
        "kelly": "kelly",
        "kelly_sizer": "kelly",
        "hrp": "hrp",
        "risk_parity": "hrp",
        "hierarchical_risk_parity": "hrp",
    }
    modules: list[str] = []
    for raw in raw_items:
        key = raw.lower().replace("-", "_").replace(" ", "_")
        if key not in aliases:
            raise ValueError(
                f"Unknown risk_desk sizing module: {raw!r}. "
                "Expected kelly, hrp, or none."
            )
        module = aliases[key]
        if module and module not in modules:
            modules.append(module)
    return tuple(modules)


def _risk_desk_detail(modules: tuple[str, ...]) -> str:
    if modules == ("kelly", "hrp"):
        return "Kelly sizing + HRP budgets + portfolio cap."
    if modules == ("kelly",):
        return "Kelly sizing + portfolio cap."
    if modules == ("hrp",):
        return "HRP risk budgets + portfolio cap."
    if not modules:
        return "Raw factor signal + portfolio cap."
    return f"{' + '.join(modules)} allocation modules + portfolio cap."


__all__ = [
    "BaseExecutionMode",
    "DirectExecutionMode",
    "ExecutionModeConfig",
    "ExecutionModeFactory",
    "ExecutionModeResult",
    "RiskDeskExecutionMode",
    "StatArbExecutionMode",
]
