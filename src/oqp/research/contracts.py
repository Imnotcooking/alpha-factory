from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any

import pandas as pd

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


VALID_EVALUATION_GEOMETRIES = {"cross_sectional", "time_series"}
VALID_EXECUTION_MODES = {"risk_desk", "direct", "statarb"}
VALID_EXECUTION_LAGS = {"same_bar", "next_bar", "next_open", "already_lagged", "custom"}
VALID_RETURN_ASSUMPTIONS = {
    "bar_signal_next_bar",
    "close_signal_next_open_to_close",
    "close_signal_next_open_to_next_open",
    "close_signal_close_to_next_close",
    "close_signal_close_to_next_open",
    "close_to_close_fallback",
    "custom_forward_return",
}
FACTOR_MARKET_WILDCARDS = {"*", "ALL", "ANY"}


@dataclass(frozen=True)
class FactorContract:
    factor_id: str
    evaluation_geometry: str
    execution_mode: str
    alpha_signal_col: str
    execution_weight_col: str
    execution_lag: str
    return_assumption: str
    supported_markets: tuple[str, ...]
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
    requested_return_assumption: str | None = None,
    default_return_assumption: str = "custom_forward_return",
    market_vertical: str | None = None,
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
    declared_return_assumption = _normalize_choice(
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
    if requested_return_assumption:
        return_assumption = _normalize_choice(
            "return_assumption",
            requested_return_assumption,
            VALID_RETURN_ASSUMPTIONS,
        )
        if return_assumption != declared_return_assumption:
            warnings.append(
                "return_assumption overridden by CLI/data horizon: "
                f"{declared_return_assumption!r} -> {return_assumption!r}."
            )
    else:
        return_assumption = declared_return_assumption
    supported_markets = validate_factor_market_compatibility(
        factor_module,
        market_vertical,
        factor_id=factor_id,
        df=df,
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
        supported_markets=supported_markets,
        contract_source=contract_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def resolve_factor_supported_markets(
    factor_module: ModuleType,
    df: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    """Return normalized market verticals declared by a factor recipe."""

    raw_contract = getattr(factor_module, "FACTOR_CONTRACT", None)
    if raw_contract is not None and not isinstance(raw_contract, dict):
        raise ValueError("FACTOR_CONTRACT must be a dict.")
    raw_contract = raw_contract or {}

    metadata = getattr(factor_module, "FACTOR_METADATA", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    raw_supported = (
        raw_contract.get("supported_markets")
        or metadata.get("supported_markets")
        or metadata.get("native_market")
        or (df.attrs.get("supported_markets") if df is not None else None)
    )
    return _normalize_market_list(raw_supported, default=("*",), field_name="supported_markets")


def validate_factor_market_compatibility(
    factor_module: ModuleType,
    market_vertical: str | None,
    *,
    factor_id: str = "",
    df: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    """Validate that a factor recipe is allowed to run on a market vertical."""

    supported = resolve_factor_supported_markets(factor_module, df=df)
    if market_vertical is None:
        return supported

    normalized_market = normalize_market_vertical(market_vertical)
    if normalized_market not in ASSET_TAXONOMY:
        raise ValueError(
            f"Invalid market_vertical={market_vertical!r}. Expected one of {sorted(ASSET_TAXONOMY)}."
        )
    label = factor_id or getattr(factor_module, "FACTOR_ID", "factor")
    if "*" not in supported and normalized_market not in supported:
        raise ValueError(
            f"{label} is not declared for {normalized_market}. "
            f"Supported markets: {', '.join(supported)}."
        )
    return supported


def attach_factor_contract_attrs(df: pd.DataFrame, contract: FactorContract) -> pd.DataFrame:
    df.attrs["factor_contract"] = contract.to_attrs()
    df.attrs["evaluation_geometry"] = contract.evaluation_geometry
    df.attrs["execution_mode"] = contract.execution_mode
    df.attrs["alpha_signal_col"] = contract.alpha_signal_col
    df.attrs["execution_weight_col"] = contract.execution_weight_col
    df.attrs["execution_lag"] = contract.execution_lag
    df.attrs["return_assumption"] = contract.return_assumption
    df.attrs["supported_markets"] = list(contract.supported_markets)
    df.attrs["execution_assumption"] = contract.return_assumption
    return df


def _normalize_market_list(
    value: Any,
    *,
    default: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        raw = str(item or "").strip()
        if not raw:
            continue
        marker = raw.upper().replace("-", "_").replace(" ", "_")
        if marker in FACTOR_MARKET_WILDCARDS:
            normalized.append("*")
            continue
        market = normalize_market_vertical(raw)
        if market not in ASSET_TAXONOMY:
            raise ValueError(
                f"Invalid factor {field_name} entry {item!r}. "
                f"Expected one of {sorted(ASSET_TAXONOMY)} or '*'."
            )
        normalized.append(market)
    return tuple(dict.fromkeys(normalized)) or default


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
        "next_open_to_next_close": "close_signal_next_open_to_close",
        "next_open_to_next_open": "close_signal_next_open_to_next_open",
        "open_to_open": "close_signal_next_open_to_next_open",
        "close_to_next_close": "close_signal_close_to_next_close",
        "close_to_close": "close_signal_close_to_next_close",
        "close_to_next_open": "close_signal_close_to_next_open",
        "overnight": "close_signal_close_to_next_open",
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
    if return_assumption in {"close_signal_next_open_to_close", "close_signal_next_open_to_next_open"}:
        return "next_open"
    if return_assumption == "bar_signal_next_bar":
        return "next_bar"
    if return_assumption in {"close_signal_close_to_next_close", "close_signal_close_to_next_open"}:
        return "same_bar"
    return "custom"


def _require_column(df: pd.DataFrame, column: str, role: str) -> None:
    if not column or column not in df.columns:
        raise ValueError(f"Factor contract {role}={column!r} is not present in factor output columns.")
