"""Cost-aware evidence for one frozen factor sleeve."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.research.sleeves.contracts import SleeveConstructionConfig
from oqp.research.sleeves.engine import SleeveConstructionResult


SLEEVE_EVIDENCE_SCHEMA_VERSION = 1
ANNUALIZATION_DAYS = 252.0


@dataclass(frozen=True, slots=True)
class SleeveEvidenceBundle:
    config: SleeveConstructionConfig
    summary: dict[str, Any]
    positions: pd.DataFrame
    daily_returns: pd.DataFrame
    split_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    product_summary: pd.DataFrame
    sector_summary: pd.DataFrame
    manifest: dict[str, Any]


def build_sleeve_evidence(
    construction: SleeveConstructionResult,
    *,
    capital: float,
    slippage_ticks_per_side: float,
) -> SleeveEvidenceBundle:
    """Translate ideal targets into whole-contract, after-cost evidence."""

    config = construction.config
    if capital <= 0:
        raise ValueError("capital must be positive")
    if slippage_ticks_per_side < 0:
        raise ValueError("slippage_ticks_per_side cannot be negative")
    if config.return_assumption != "close_signal_next_open_to_close":
        raise NotImplementedError(
            "Phase 3 evidence currently supports the frozen next-open-to-close "
            "one-session horizon only"
        )
    required = {
        config.date_col,
        config.product_col,
        config.sector_col,
        config.split_col,
        config.return_col,
        "target_weight",
        "next_symbol",
        "next_actual_open",
        "next_multiplier",
        "next_tick_size",
        "next_fee_type",
        "next_fee_open",
        "next_fee_close_today",
    }
    missing = sorted(required.difference(construction.positions.columns))
    if missing:
        raise ValueError(f"sleeve evidence is missing columns: {missing}")

    positions = execute_intraday_session_targets(
        construction.positions,
        config,
        capital=float(capital),
        slippage_ticks_per_side=float(slippage_ticks_per_side),
    )
    daily = summarize_executed_positions(positions, config)
    split_summary = _split_summary(daily, config)
    yearly_summary = _yearly_summary(daily)
    product_summary = _member_summary(positions, config.product_col)
    sector_summary = _member_summary(positions, config.sector_col)
    full = split_summary.loc[split_summary["research_split"].eq("full")].iloc[0]
    summary = {
        "schema_version": SLEEVE_EVIDENCE_SCHEMA_VERSION,
        "sleeve_id": config.sleeve_id,
        "factor_id": config.factor_id,
        "construction": config.construction,
        "capital": float(capital),
        "slippage_ticks_per_side": float(slippage_ticks_per_side),
        "return_assumption": config.return_assumption,
        "full_sample": _json_safe(full.to_dict()),
        "cost_model": (
            "Whole contracts at next actual open; instrument-master opening and "
            "same-day-close fees; configured tick slippage charged on both sides."
        ),
        "interpretation_boundary": (
            "This evaluates one fixed sleeve construction. It is neither a router "
            "test nor evidence that alternative sleeve parameters were optimized."
        ),
    }
    attrs = construction.positions.attrs
    manifest = {
        "schema_version": SLEEVE_EVIDENCE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "factor_id": config.factor_id,
        "sleeve_id": config.sleeve_id,
        "input_data_fingerprint": str(attrs.get("input_data_fingerprint") or ""),
        "factor_definition_fingerprint": str(
            attrs.get("factor_definition_fingerprint") or ""
        ),
        "factor_implementation_fingerprint": str(
            attrs.get("factor_implementation_fingerprint") or ""
        ),
        "sleeve_implementation_fingerprint": str(
            attrs.get("sleeve_implementation_fingerprint") or ""
        ),
        "sleeve_definition_fingerprint": str(
            attrs.get("sleeve_definition_fingerprint") or ""
        ),
        "predictive_evidence_config_fingerprint": str(
            attrs.get("predictive_evidence_config_fingerprint") or ""
        ),
        "causal_alignment_verified": bool(
            attrs.get("causal_return_alignment_verified")
        ),
        "execution": {
            "capital": float(capital),
            "capital_currency": "CNY",
            "integer_contracts": True,
            "slippage_ticks_per_side": float(slippage_ticks_per_side),
            "return_assumption": config.return_assumption,
            "entry": "next_actual_open",
            "exit": "next_actual_close",
            "fee_fields": [
                "next_fee_type",
                "next_fee_open",
                "next_fee_close_today",
            ],
        },
        "formulas": {
            "gross_contribution": "executed_weight * forward_return",
            "net_return": "gross_return - exchange_fee_return - slippage_return",
            "turnover": "entry_notional_plus_exit_notional / capital",
            "sharpe": "mean(daily_return) / sample_std(daily_return) * sqrt(252)",
            "maximum_drawdown": "minimum compounded wealth / prior peak - 1",
        },
        "input": {
            "rows": int(len(positions)),
            "dates": int(positions[config.date_col].nunique()),
            "products": int(positions[config.product_col].nunique()),
            "start": _date_text(positions[config.date_col].min()),
            "end": _date_text(positions[config.date_col].max()),
        },
    }
    return SleeveEvidenceBundle(
        config=config,
        summary=summary,
        positions=positions,
        daily_returns=daily,
        split_summary=split_summary,
        yearly_summary=yearly_summary,
        product_summary=product_summary,
        sector_summary=sector_summary,
        manifest=manifest,
    )


def write_sleeve_evidence_bundle(
    bundle: SleeveEvidenceBundle,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(_json_safe(bundle.summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(_json_safe(bundle.manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    bundle.positions.to_parquet(destination / "positions.parquet", index=False)
    bundle.daily_returns.to_parquet(destination / "daily_returns.parquet", index=False)
    bundle.split_summary.to_csv(destination / "split_summary.csv", index=False)
    bundle.yearly_summary.to_csv(destination / "yearly_summary.csv", index=False)
    bundle.product_summary.to_csv(destination / "product_summary.csv", index=False)
    bundle.sector_summary.to_csv(destination / "sector_summary.csv", index=False)
    return destination


def load_sleeve_evidence_bundle(output_dir: str | Path) -> SleeveEvidenceBundle:
    source = Path(output_dir).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config = SleeveConstructionConfig(**manifest["config"])
    return SleeveEvidenceBundle(
        config=config,
        summary=summary,
        positions=pd.read_parquet(source / "positions.parquet"),
        daily_returns=pd.read_parquet(source / "daily_returns.parquet"),
        split_summary=pd.read_csv(source / "split_summary.csv"),
        yearly_summary=pd.read_csv(source / "yearly_summary.csv"),
        product_summary=pd.read_csv(source / "product_summary.csv"),
        sector_summary=pd.read_csv(source / "sector_summary.csv"),
        manifest=manifest,
    )


def execute_intraday_session_targets(
    frame: pd.DataFrame,
    config: SleeveConstructionConfig,
    *,
    capital: float,
    slippage_ticks_per_side: float,
) -> pd.DataFrame:
    out = frame.copy()
    target = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(out["next_actual_open"], errors="coerce")
    multiplier = pd.to_numeric(out["next_multiplier"], errors="coerce")
    notional = price * multiplier
    executable = notional.gt(0.0) & out["next_symbol"].notna()
    desired = target * capital / notional.where(executable)
    out["contracts"] = np.trunc(desired).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["execution_eligible"] = executable
    out["executed_weight"] = out["contracts"] * notional.fillna(0.0) / capital
    out["weight_rounding_shortfall"] = target.abs() - out["executed_weight"].abs()

    # This frozen horizon is flat by the close: every active contract is opened
    # at next open and closed the same session, irrespective of tomorrow's signal.
    contracts = out["contracts"].abs()
    fee_open = pd.to_numeric(out["next_fee_open"], errors="coerce").fillna(0.0)
    fee_close = pd.to_numeric(
        out["next_fee_close_today"], errors="coerce"
    ).fillna(0.0)
    fixed_fee = contracts * (fee_open + fee_close)
    ratio_fee = contracts * notional.fillna(0.0) * (fee_open + fee_close)
    out["exchange_fee_cny"] = np.where(
        out["next_fee_type"].astype(str).str.lower().eq("fixed"),
        fixed_fee,
        ratio_fee,
    )
    out["slippage_cny"] = (
        2.0
        * contracts
        * slippage_ticks_per_side
        * pd.to_numeric(out["next_tick_size"], errors="coerce").fillna(0.0)
        * multiplier.fillna(0.0)
    )
    out["exchange_fee_return"] = out["exchange_fee_cny"] / capital
    out["slippage_return"] = out["slippage_cny"] / capital
    out["cost_return"] = out["exchange_fee_return"] + out["slippage_return"]
    out["turnover"] = 2.0 * contracts * notional.fillna(0.0) / capital
    forward_return = pd.to_numeric(out[config.return_col], errors="coerce")
    out["return_available"] = forward_return.notna()
    out["unpriced_active_position"] = out["contracts"].ne(0.0) & forward_return.isna()
    out["gross_contribution"] = out["executed_weight"] * forward_return.fillna(0.0)
    out["net_contribution"] = out["gross_contribution"] - out["cost_return"]
    return out


def summarize_executed_positions(
    positions: pd.DataFrame,
    config: SleeveConstructionConfig,
) -> pd.DataFrame:
    split_count = positions.groupby(config.date_col)[config.split_col].nunique()
    if split_count.gt(1).any():
        raise ValueError("a decision date cannot belong to more than one research split")
    daily = (
        positions.groupby(config.date_col, as_index=False, sort=True)
        .agg(
            research_split=(config.split_col, "first"),
            gross_return=("gross_contribution", "sum"),
            exchange_fee_return=("exchange_fee_return", "sum"),
            slippage_return=("slippage_return", "sum"),
            cost_return=("cost_return", "sum"),
            turnover=("turnover", "sum"),
            target_gross=("target_weight", lambda x: float(x.abs().sum())),
            target_net=("target_weight", "sum"),
            executed_gross=("executed_weight", lambda x: float(x.abs().sum())),
            executed_net=("executed_weight", "sum"),
            active_products=("contracts", lambda x: int(x.ne(0.0).sum())),
            unpriced_active_positions=("unpriced_active_position", "sum"),
            contract_cap_count=("contract_cap_bound", "sum"),
            sector_cap_count=("sector_cap_bound", "sum"),
        )
        .sort_values(config.date_col)
        .reset_index(drop=True)
    )
    daily["net_return"] = daily["gross_return"] - daily["cost_return"]
    daily["gross_realization"] = np.where(
        daily["target_gross"].gt(0.0),
        daily["executed_gross"] / daily["target_gross"],
        np.nan,
    )
    daily["cumulative_gross_return"] = (1.0 + daily["gross_return"]).cumprod() - 1.0
    daily["cumulative_net_return"] = (1.0 + daily["net_return"]).cumprod() - 1.0
    return daily


def _split_summary(
    daily: pd.DataFrame,
    config: SleeveConstructionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = ["full"] + list(dict.fromkeys(daily["research_split"].astype(str)))
    for label in labels:
        sample = daily if label == "full" else daily.loc[daily["research_split"].eq(label)]
        rows.append(_performance_metrics(sample, label, config))
    return pd.DataFrame(rows)


def _yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    dated = daily.copy()
    dated["year"] = pd.to_datetime(dated["date"]).dt.year
    return pd.DataFrame(
        [_performance_metrics(group, str(int(year)), None) | {"year": int(year)}
         for year, group in dated.groupby("year", sort=True)]
    )


def _performance_metrics(
    daily: pd.DataFrame,
    label: str,
    config: SleeveConstructionConfig | None,
) -> dict[str, Any]:
    gross = pd.to_numeric(daily["gross_return"], errors="coerce").dropna()
    net = pd.to_numeric(daily["net_return"], errors="coerce").dropna()
    wealth = (1.0 + net).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "research_split": label,
        "date_count": int(len(daily)),
        "gross_total_return": _compound(gross),
        "net_total_return": _compound(net),
        "gross_annualized_mean": _annualized_mean(gross),
        "net_annualized_mean": _annualized_mean(net),
        "gross_annualized_volatility": _annualized_volatility(gross),
        "net_annualized_volatility": _annualized_volatility(net),
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "maximum_drawdown": float(drawdown.min()) if not drawdown.empty else math.nan,
        "mean_daily_turnover": _mean(daily["turnover"]),
        "annualized_turnover": _mean(daily["turnover"]) * ANNUALIZATION_DAYS,
        "annualized_cost": _mean(daily["cost_return"]) * ANNUALIZATION_DAYS,
        "mean_target_gross": _mean(daily["target_gross"]),
        "mean_executed_gross": _mean(daily["executed_gross"]),
        "mean_executed_net": _mean(daily["executed_net"]),
        "mean_gross_realization": _mean(daily["gross_realization"]),
        "active_day_fraction": float(daily["active_products"].gt(0).mean()) if len(daily) else math.nan,
        "mean_active_products": _mean(daily["active_products"]),
        "positive_net_day_fraction": float(net.gt(0.0).mean()) if len(net) else math.nan,
        "unpriced_active_positions": int(daily["unpriced_active_positions"].sum()),
        "target_gross_exposure": config.target_gross_exposure if config else math.nan,
    }


def _member_summary(positions: pd.DataFrame, member_col: str) -> pd.DataFrame:
    return (
        positions.groupby(member_col, as_index=False, sort=True)
        .agg(
            gross_contribution=("gross_contribution", "sum"),
            cost_return=("cost_return", "sum"),
            net_contribution=("net_contribution", "sum"),
            turnover=("turnover", "sum"),
            mean_absolute_weight=("executed_weight", lambda x: float(x.abs().mean())),
            active_days=("contracts", lambda x: int(x.ne(0.0).sum())),
            long_days=("contracts", lambda x: int(x.gt(0.0).sum())),
            short_days=("contracts", lambda x: int(x.lt(0.0).sum())),
            unpriced_active_positions=("unpriced_active_position", "sum"),
        )
        .sort_values("net_contribution", ascending=False)
        .reset_index(drop=True)
    )


def _compound(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float((1.0 + clean).prod() - 1.0) if len(clean) else math.nan


def _annualized_mean(values: pd.Series) -> float:
    return _mean(values) * ANNUALIZATION_DAYS


def _annualized_volatility(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)) if len(clean) > 1 else math.nan


def _sharpe(values: pd.Series) -> float:
    volatility = _annualized_volatility(values)
    return _annualized_mean(values) / volatility if volatility and volatility > 0 else math.nan


def _mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat() if pd.notna(value) else ""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


__all__ = [
    "ANNUALIZATION_DAYS",
    "SLEEVE_EVIDENCE_SCHEMA_VERSION",
    "SleeveEvidenceBundle",
    "build_sleeve_evidence",
    "execute_intraday_session_targets",
    "load_sleeve_evidence_bundle",
    "summarize_executed_positions",
    "write_sleeve_evidence_bundle",
]
