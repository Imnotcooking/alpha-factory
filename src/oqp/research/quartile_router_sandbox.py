"""Analysis engine for the monthly Chinese-futures quartile router sandbox.

The engine does not define alpha signals. It recombines frozen, precomputed sleeve
weights so dashboard experiments cannot silently change the underlying factors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd


STATE_ORDER = ("Q1", "Q2", "Q3", "Q4")
STRATEGY_ORDER = ("momentum", "reversal", "static_50_50", "flat")
PAPER_ASSIGNMENTS = {
    "Q1": "momentum",
    "Q2": "momentum",
    "Q3": "momentum",
    "Q4": "reversal",
}
RETURN_COLUMNS = {
    "momentum": "momentum_return",
    "reversal": "reversal_return",
    "static_50_50": "static_50_50_return",
}
WEIGHT_COLUMNS = {
    "momentum": "mom_weight",
    "reversal": "rev_weight",
    "static_50_50": "static_50_50_weight",
}


@dataclass(frozen=True)
class QuartileRouterConfig:
    """One causal strategy assignment for each volatility quartile."""

    proxy: str
    assignments: Mapping[str, str]
    source_id: str = "frozen_07_04"

    def normalized_assignments(self) -> dict[str, str]:
        assignments = {str(key): str(value) for key, value in self.assignments.items()}
        missing = sorted(set(STATE_ORDER).difference(assignments))
        extra = sorted(set(assignments).difference(STATE_ORDER))
        invalid = sorted(
            strategy
            for strategy in assignments.values()
            if strategy not in STRATEGY_ORDER
        )
        if missing or extra or invalid:
            raise ValueError(
                "invalid quartile assignments: "
                f"missing={missing}, extra={extra}, invalid_strategies={invalid}"
            )
        return {state: assignments[state] for state in STATE_ORDER}

    def snapshot(self) -> dict[str, object]:
        return {
            "proxy": self.proxy,
            "assignments": self.normalized_assignments(),
            "source_id": self.source_id,
            "timing": "signal_month_t_routes_holding_month_t_plus_1",
        }

    def run_id(self) -> str:
        payload = json.dumps(
            self.snapshot(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass
class QuartileRouterResult:
    """Frames used by the Streamlit page and its verification tests."""

    monthly: pd.DataFrame
    holdings: pd.DataFrame
    state_summary: pd.DataFrame
    transition_matrix: pd.DataFrame
    performance: pd.DataFrame
    run_id: str
    config_snapshot: dict[str, object]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _month_key(values: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Period):
            return str(value.asfreq("M"))
        return str(pd.Period(value, freq="M"))

    return values.map(normalize)


def _validate_causal_states(states: pd.DataFrame) -> None:
    signal = states["signal_month"].map(lambda value: pd.Period(value, freq="M"))
    holding = states["holding_month"].map(lambda value: pd.Period(value, freq="M"))
    if not (signal + 1).equals(holding):
        raise ValueError("state timing must route signal month t into holding month t+1")


def _cost_map(cost_ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "root", "one_way_cost_ratio", "roll_cost_ratio"}
    _require_columns(cost_ledger, required, "cost ledger")
    costs = cost_ledger[list(required)].copy()
    costs["month"] = _month_key(costs["month"])
    consistency = costs.groupby(["month", "root"], observed=True)[
        ["one_way_cost_ratio", "roll_cost_ratio"]
    ].nunique(dropna=False)
    if not consistency.empty and consistency.to_numpy().max() > 1:
        raise ValueError("cost ratios differ across frozen strategy ledger rows")
    return costs.drop_duplicates(["month", "root"]).reset_index(drop=True)


def _select_strategy_values(
    frame: pd.DataFrame,
    strategy_column: str,
    value_columns: Mapping[str, str],
) -> pd.Series:
    selected = pd.Series(0.0, index=frame.index, dtype=float)
    for strategy, column in value_columns.items():
        selected.loc[frame[strategy_column].eq(strategy)] = pd.to_numeric(
            frame.loc[frame[strategy_column].eq(strategy), column], errors="coerce"
        ).fillna(0.0)
    return selected


def _performance_row(strategy: str, basis: str, returns: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "strategy": strategy,
            "basis": basis,
            "months": 0,
            "total_return": np.nan,
            "annualized_arithmetic_return": np.nan,
            "annualized_volatility": np.nan,
            "annualized_sharpe": np.nan,
            "maximum_drawdown": np.nan,
            "hit_rate": np.nan,
        }
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    annualized_volatility = float(clean.std(ddof=1) * np.sqrt(12.0))
    annualized_return = float(clean.mean() * 12.0)
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0.0
        else np.nan
    )
    return {
        "strategy": strategy,
        "basis": basis,
        "months": int(len(clean)),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "annualized_arithmetic_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "hit_rate": float(clean.gt(0.0).mean()),
    }


def summarize_performance(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build gross and net metrics for the router and frozen benchmarks."""

    columns = {
        ("router", "gross"): "router_gross_return",
        ("router", "net"): "router_net_return",
        ("momentum", "gross"): "momentum_return",
        ("momentum", "net"): "momentum_net_return",
        ("reversal", "gross"): "reversal_return",
        ("reversal", "net"): "reversal_net_return",
        ("static_50_50", "gross"): "static_50_50_return",
        ("static_50_50", "net"): "static_50_50_net_return",
    }
    rows = [
        _performance_row(strategy, basis, monthly[column])
        for (strategy, basis), column in columns.items()
        if column in monthly.columns
    ]
    return pd.DataFrame(rows)


def _build_transition_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    current = monthly["volatility_state"].astype(str)
    previous = current.shift(1)
    matrix = pd.crosstab(previous, current, normalize="index")
    return matrix.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0.0)


def _build_state_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    summary = (
        monthly.groupby("volatility_state", observed=True)
        .agg(
            months=("month", "size"),
            selected_strategy=("selected_strategy", "first"),
            mean_gross_return=("router_gross_return", "mean"),
            mean_net_return=("router_net_return", "mean"),
            total_gross_contribution=("router_gross_return", "sum"),
            total_net_contribution=("router_net_return", "sum"),
            momentum_mean=("momentum_return", "mean"),
            reversal_mean=("reversal_return", "mean"),
            static_mean=("static_50_50_return", "mean"),
            mean_turnover=("router_target_turnover", "mean"),
            total_cost=("router_cost_return", "sum"),
        )
        .reindex(STATE_ORDER)
        .reset_index()
        .rename(columns={"volatility_state": "state"})
    )
    summary["reversal_minus_momentum"] = (
        summary["reversal_mean"] - summary["momentum_mean"]
    )
    return summary


def build_quartile_router(
    monthly: pd.DataFrame,
    targets: pd.DataFrame,
    states: pd.DataFrame,
    cost_ledger: pd.DataFrame,
    config: QuartileRouterConfig,
    *,
    diagnostics: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
) -> QuartileRouterResult:
    """Recombine frozen sleeve weights according to a Q1-Q4 assignment."""

    assignments = config.normalized_assignments()
    _require_columns(
        monthly,
        {
            "month",
            "momentum_return",
            "reversal_return",
            "static_50_50_return",
        },
        "monthly returns",
    )
    _require_columns(
        targets,
        {
            "month",
            "root",
            "holding_return",
            "momentum_score",
            "reversal_score",
            "mom_weight",
            "rev_weight",
            "static_50_50_weight",
        },
        "targets",
    )
    _require_columns(
        states,
        {
            "signal_month",
            "holding_month",
            "proxy",
            "current_volatility",
            "q25",
            "q50",
            "q75",
            "volatility_state",
        },
        "volatility states",
    )

    state_frame = states.loc[states["proxy"].astype(str).eq(config.proxy)].copy()
    if state_frame.empty:
        raise ValueError(f"no volatility states found for proxy {config.proxy!r}")
    state_frame["signal_month"] = _month_key(state_frame["signal_month"])
    state_frame["holding_month"] = _month_key(state_frame["holding_month"])
    state_frame = state_frame.loc[
        state_frame["volatility_state"].isin(STATE_ORDER)
    ].copy()
    _validate_causal_states(state_frame)
    if state_frame["holding_month"].duplicated().any():
        raise ValueError("volatility states contain duplicate holding months")
    state_frame["selected_strategy"] = state_frame["volatility_state"].map(assignments)

    monthly_frame = monthly.copy()
    monthly_frame["month"] = _month_key(monthly_frame["month"])
    state_columns = [
        "signal_month",
        "holding_month",
        "current_volatility",
        "q25",
        "q50",
        "q75",
        "volatility_state",
        "selected_strategy",
    ]
    monthly_frame = monthly_frame.merge(
        state_frame[state_columns].rename(columns={"holding_month": "month"}),
        on="month",
        how="inner",
        validate="one_to_one",
    ).sort_values("month", kind="mergesort")
    monthly_frame["router_gross_return"] = _select_strategy_values(
        monthly_frame, "selected_strategy", RETURN_COLUMNS
    )

    if diagnostics is not None and not diagnostics.empty:
        diagnostic_frame = diagnostics.copy()
        key = "signal_month" if "signal_month" in diagnostic_frame.columns else "month"
        diagnostic_frame[key] = _month_key(diagnostic_frame[key])
        if diagnostic_frame[key].duplicated().any():
            raise ValueError("diagnostics contain duplicate signal months")
        if key != "signal_month":
            diagnostic_frame = diagnostic_frame.rename(columns={key: "signal_month"})
        monthly_frame = monthly_frame.merge(
            diagnostic_frame,
            on="signal_month",
            how="left",
            validate="one_to_one",
        )

    target_frame = targets.copy()
    target_frame["month"] = _month_key(target_frame["month"])
    target_frame["root"] = target_frame["root"].astype(str)
    cost_frame = _cost_map(cost_ledger)
    cost_frame["root"] = cost_frame["root"].astype(str)
    roots = sorted(set(target_frame["root"]).union(cost_frame["root"]))
    grid = pd.MultiIndex.from_product(
        [monthly_frame["month"].tolist(), roots], names=["month", "root"]
    ).to_frame(index=False)
    holdings = grid.merge(
        target_frame,
        on=["month", "root"],
        how="left",
        validate="one_to_one",
    )
    holdings = holdings.merge(
        monthly_frame[
            ["month", "signal_month", "volatility_state", "selected_strategy"]
        ],
        on="month",
        how="left",
        validate="many_to_one",
    )
    for column in WEIGHT_COLUMNS.values():
        holdings[column] = pd.to_numeric(holdings[column], errors="coerce").fillna(0.0)
    holdings["target_weight"] = _select_strategy_values(
        holdings, "selected_strategy", WEIGHT_COLUMNS
    )
    holdings = holdings.sort_values(["root", "month"], kind="mergesort")
    holdings["prior_target_weight"] = (
        holdings.groupby("root", sort=False)["target_weight"].shift(1).fillna(0.0)
    )
    holdings["trade_weight"] = holdings["target_weight"] - holdings["prior_target_weight"]
    holdings["target_turnover"] = holdings["trade_weight"].abs()
    holdings = holdings.merge(
        cost_frame,
        on=["month", "root"],
        how="left",
        validate="one_to_one",
    )
    holdings[["one_way_cost_ratio", "roll_cost_ratio"]] = holdings[
        ["one_way_cost_ratio", "roll_cost_ratio"]
    ].fillna(0.0)
    holdings["rebalance_cost_return"] = (
        holdings["target_turnover"] * holdings["one_way_cost_ratio"]
    )
    holdings["roll_cost_return"] = (
        holdings["target_weight"].abs() * holdings["roll_cost_ratio"]
    )
    holdings["total_cost_return"] = (
        holdings["rebalance_cost_return"] + holdings["roll_cost_return"]
    )
    holdings["gross_contribution"] = (
        holdings["target_weight"]
        * pd.to_numeric(holdings["holding_return"], errors="coerce").fillna(0.0)
    )
    holdings["net_contribution"] = (
        holdings["gross_contribution"] - holdings["total_cost_return"]
    )
    holdings["position"] = np.select(
        [holdings["target_weight"].gt(0.0), holdings["target_weight"].lt(0.0)],
        ["long", "short"],
        default="flat",
    )

    if sectors is not None and not sectors.empty:
        sector_frame = sectors[["root", "sector"]].copy()
        sector_frame["root"] = sector_frame["root"].astype(str)
        sector_frame = sector_frame.drop_duplicates("root")
        holdings = holdings.merge(
            sector_frame, on="root", how="left", validate="many_to_one"
        )
    if "sector" not in holdings.columns:
        holdings["sector"] = "Unclassified"
    holdings["sector"] = holdings["sector"].fillna("Unclassified")

    monthly_cost = (
        holdings.groupby("month", observed=True)
        .agg(
            router_target_turnover=("target_turnover", "sum"),
            router_cost_return=("total_cost_return", "sum"),
            holdings_gross_return=("gross_contribution", "sum"),
        )
        .reset_index()
    )
    monthly_frame = monthly_frame.merge(
        monthly_cost, on="month", how="left", validate="one_to_one"
    )
    if not np.allclose(
        monthly_frame["router_gross_return"],
        monthly_frame["holdings_gross_return"],
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("router gross returns do not reconcile to asset holdings")
    monthly_frame["router_net_return"] = (
        monthly_frame["router_gross_return"] - monthly_frame["router_cost_return"]
    )
    monthly_frame["router_minus_momentum_gross"] = (
        monthly_frame["router_gross_return"] - monthly_frame["momentum_return"]
    )
    monthly_frame["date"] = pd.PeriodIndex(monthly_frame["month"], freq="M").to_timestamp("M")

    holdings = holdings.sort_values(["month", "root"], kind="mergesort").reset_index(drop=True)
    monthly_frame = monthly_frame.reset_index(drop=True)
    return QuartileRouterResult(
        monthly=monthly_frame,
        holdings=holdings,
        state_summary=_build_state_summary(monthly_frame),
        transition_matrix=_build_transition_matrix(monthly_frame),
        performance=summarize_performance(monthly_frame),
        run_id=config.run_id(),
        config_snapshot=config.snapshot(),
    )
