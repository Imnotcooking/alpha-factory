from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any

import pandas as pd


VALID_EVALUATION_GEOMETRIES = {"cross_sectional", "time_series"}
VALID_EXECUTION_MODES = {"risk_desk", "direct", "statarb"}
VALID_EXECUTION_LAGS = {"same_bar", "next_bar", "next_open", "already_lagged", "custom"}
VALID_RETURN_ASSUMPTIONS = {
    "bar_signal_next_bar",
    "close_signal_next_open_to_close",
    "close_to_close_fallback",
    "custom_forward_return",
}


@dataclass(frozen=True)
class FactorContract:
    factor_id: str
    evaluation_geometry: str
    execution_mode: str
    alpha_signal_col: str
    execution_weight_col: str
    execution_lag: str
    return_assumption: str
    contract_source: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_attrs(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def resolve_factor_contract(
    factor_module: ModuleType,
    df: pd.DataFrame,
    *,
    factor_id: str,
    requested_execution_mode: str = "auto",
    default_return_assumption: str = "custom_forward_return",
    strict: bool = False,
) -> FactorContract:
    raw_contract = getattr(factor_module, "FACTOR_CONTRACT", None)
    metadata = getattr(factor_module, "FACTOR_METADATA", {}) or {}
    if raw_contract is None:
        raw_contract = {}
        contract_source = "inferred_legacy"
    elif not isinstance(raw_contract, dict):
        raise ValueError("FACTOR_CONTRACT must be a dict.")
    else:
        contract_source = "explicit"

    warnings: list[str] = []
    if strict and contract_source != "explicit":
        raise ValueError(
            f"{factor_id} must declare FACTOR_CONTRACT when strict factor contracts are enabled."
        )

    def pick(name: str, *fallbacks: Any) -> Any:
        if name in raw_contract and raw_contract[name] not in (None, ""):
            return raw_contract[name]
        if strict:
            raise ValueError(f"{factor_id} FACTOR_CONTRACT missing required field: {name}")
        for value in fallbacks:
            if value not in (None, ""):
                if contract_source != "explicit":
                    warnings.append(f"{name} inferred as {value!r}.")
                return value
        warnings.append(f"{name} inferred by fallback.")
        return None

    evaluation_geometry = _normalize_choice(
        "evaluation_geometry",
        pick(
            "evaluation_geometry",
            getattr(factor_module, "EVALUATION_GEOMETRY", None),
            df.attrs.get("evaluation_geometry"),
            _infer_geometry(df),
        ),
        VALID_EVALUATION_GEOMETRIES,
    )
    declared_execution_mode = _normalize_choice(
        "execution_mode",
        pick(
            "execution_mode",
            getattr(factor_module, "EXECUTION_MODE", None),
            df.attrs.get("execution_mode"),
            metadata.get("execution_mode"),
            "risk_desk",
        ),
        VALID_EXECUTION_MODES,
    )
    if requested_execution_mode and requested_execution_mode != "auto":
        execution_mode = _normalize_choice("execution_mode", requested_execution_mode, VALID_EXECUTION_MODES)
        if execution_mode != declared_execution_mode:
            warnings.append(
                f"execution_mode overridden by CLI: {declared_execution_mode!r} -> {execution_mode!r}."
            )
    else:
        execution_mode = declared_execution_mode
    alpha_signal_col = str(
        pick(
            "alpha_signal_col",
            getattr(factor_module, "ALPHA_SIGNAL_COL", None),
            df.attrs.get("alpha_signal_col"),
            _first_existing(df, ("factor_score", "raw_signal", "signal", "target_weight", "final_target_weight")),
        )
    )
    execution_weight_col = str(
        pick(
            "execution_weight_col",
            getattr(factor_module, "EXECUTION_WEIGHT_COL", None),
            df.attrs.get("execution_weight_col"),
            _default_execution_weight_col(df, execution_mode, alpha_signal_col),
        )
    )
    execution_lag = _normalize_choice(
        "execution_lag",
        pick(
            "execution_lag",
            getattr(factor_module, "EXECUTION_LAG", None),
            df.attrs.get("execution_lag"),
            _infer_execution_lag(default_return_assumption),
        ),
        VALID_EXECUTION_LAGS,
    )
    return_assumption = _normalize_choice(
        "return_assumption",
        pick(
            "return_assumption",
            getattr(factor_module, "RETURN_ASSUMPTION", None),
            df.attrs.get("return_assumption"),
            df.attrs.get("execution_assumption"),
            default_return_assumption,
        ),
        VALID_RETURN_ASSUMPTIONS,
    )

    _require_column(df, alpha_signal_col, "alpha_signal_col")
    _require_column(df, execution_weight_col, "execution_weight_col")

    return FactorContract(
        factor_id=factor_id,
        evaluation_geometry=evaluation_geometry,
        execution_mode=execution_mode,
        alpha_signal_col=alpha_signal_col,
        execution_weight_col=execution_weight_col,
        execution_lag=execution_lag,
        return_assumption=return_assumption,
        contract_source=contract_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def attach_factor_contract_attrs(df: pd.DataFrame, contract: FactorContract) -> pd.DataFrame:
    df.attrs["factor_contract"] = contract.to_attrs()
    df.attrs["evaluation_geometry"] = contract.evaluation_geometry
    df.attrs["execution_mode"] = contract.execution_mode
    df.attrs["alpha_signal_col"] = contract.alpha_signal_col
    df.attrs["execution_weight_col"] = contract.execution_weight_col
    df.attrs["execution_lag"] = contract.execution_lag
    df.attrs["return_assumption"] = contract.return_assumption
    df.attrs["execution_assumption"] = contract.return_assumption
    return df


def _normalize_choice(name: str, value: Any, valid: set[str]) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cs": "cross_sectional",
        "rank_ic": "cross_sectional",
        "ts": "time_series",
        "timeseries": "time_series",
        "pearson": "time_series",
        "risk": "risk_desk",
        "riskdesk": "risk_desk",
        "stat_arb": "statarb",
        "pairs": "statarb",
        "pair": "statarb",
        "next_session_open": "next_open",
        "open_to_close": "close_signal_next_open_to_close",
        "next_snapshot": "bar_signal_next_bar",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in valid:
        raise ValueError(f"Invalid {name}: {value!r}. Expected one of {sorted(valid)}.")
    return normalized


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _default_execution_weight_col(df: pd.DataFrame, execution_mode: str, alpha_signal_col: str) -> str | None:
    if execution_mode == "risk_desk":
        return alpha_signal_col
    return _first_existing(df, ("target_weight", "final_target_weight", "desired_weight", "signal", alpha_signal_col))


def _infer_geometry(df: pd.DataFrame) -> str:
    ticker_count = df["ticker"].nunique() if "ticker" in df.columns else 0
    if ticker_count <= 1:
        return "time_series"
    counts = df.groupby("date")["ticker"].nunique() if "date" in df.columns else pd.Series(dtype=int)
    return "cross_sectional" if not counts.empty and (counts >= 3).mean() >= 0.5 else "time_series"


def _infer_execution_lag(return_assumption: str) -> str:
    if return_assumption == "close_signal_next_open_to_close":
        return "next_open"
    if return_assumption == "bar_signal_next_bar":
        return "next_bar"
    return "custom"


def _require_column(df: pd.DataFrame, column: str, role: str) -> None:
    if not column or column not in df.columns:
        raise ValueError(f"Factor contract {role}={column!r} is not present in factor output columns.")
