"""Volatility forecasting models and persistence."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_VOL_FORECAST_DB_PATH = REPO_ROOT / "runtime" / "db" / "market" / "vol_forecasts.db"
DEFAULT_VOL_FORECAST_HORIZONS = (1, 5, 21)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class VolatilityForecast:
    """One annualized volatility forecast for a symbol/horizon/model."""

    symbol: str
    as_of: str
    horizon_days: int
    model: str
    forecast_vol: float | None
    status: str = "ok"
    detail: str = ""
    components: dict[str, Any] | None = None
    created_at: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "horizon_days": self.horizon_days,
            "model": self.model,
            "forecast_vol": self.forecast_vol,
            "status": self.status,
            "detail": self.detail,
            "components": self.components or {},
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def ensure_vol_forecast_schema(path: str | Path = DEFAULT_VOL_FORECAST_DB_PATH) -> Path:
    """Create the volatility forecast schema if needed."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS volatility_forecasts (
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                model TEXT NOT NULL,
                forecast_vol REAL,
                status TEXT NOT NULL,
                detail TEXT,
                components_json TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (symbol, as_of, horizon_days, model)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_volatility_forecasts_symbol_created
            ON volatility_forecasts (symbol, created_at)
            """
        )
        conn.commit()
    return db_path


def write_volatility_forecasts(
    forecasts: Iterable[VolatilityForecast],
    *,
    path: str | Path = DEFAULT_VOL_FORECAST_DB_PATH,
) -> int:
    """Persist volatility forecasts and return rows written."""

    rows = [forecast.to_row() for forecast in forecasts]
    if not rows:
        return 0

    db_path = ensure_vol_forecast_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO volatility_forecasts (
                symbol, as_of, horizon_days, model, forecast_vol, status,
                detail, components_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, as_of, horizon_days, model) DO UPDATE SET
                forecast_vol = excluded.forecast_vol,
                status = excluded.status,
                detail = excluded.detail,
                components_json = excluded.components_json,
                created_at = excluded.created_at
            """,
            [
                (
                    row["symbol"],
                    row["as_of"],
                    int(row["horizon_days"]),
                    row["model"],
                    _optional_float(row["forecast_vol"]),
                    row["status"],
                    row["detail"],
                    json.dumps(row["components"], sort_keys=True),
                    row["created_at"],
                )
                for row in rows
            ],
        )
        conn.commit()
    return len(rows)


def load_latest_volatility_forecasts(
    symbol: str,
    *,
    path: str | Path = DEFAULT_VOL_FORECAST_DB_PATH,
) -> pd.DataFrame:
    """Load the latest forecast set for a symbol."""

    symbol_key = normalize_symbol(symbol)
    if not symbol_key:
        return forecast_frame([])

    db_path = ensure_vol_forecast_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        latest = conn.execute(
            """
            SELECT as_of
            FROM volatility_forecasts
            WHERE symbol = ?
            ORDER BY as_of DESC, created_at DESC
            LIMIT 1
            """,
            (symbol_key,),
        ).fetchone()
        if latest is None:
            return forecast_frame([])
        frame = pd.read_sql_query(
            """
            SELECT symbol, as_of, horizon_days, model, forecast_vol, status, detail,
                   components_json, created_at
            FROM volatility_forecasts
            WHERE symbol = ? AND as_of = ?
            ORDER BY horizon_days, model
            """,
            conn,
            params=(symbol_key, latest[0]),
        )
    return frame


def forecast_volatility_models(
    symbol: str,
    history: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_VOL_FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Run baseline, HAR-HV, GARCH, and ensemble volatility forecasts."""

    symbol_key = normalize_symbol(symbol)
    clean_history = normalize_ohlc_history(history)
    horizon_values = sorted({max(1, int(horizon)) for horizon in horizons})
    if clean_history.empty or "Close" not in clean_history.columns:
        return forecast_frame(
            [
                VolatilityForecast(
                    symbol=symbol_key,
                    as_of="",
                    horizon_days=horizon,
                    model=model,
                    forecast_vol=None,
                    status="missing_data",
                    detail="No usable close history.",
                )
                for horizon in horizon_values
                for model in ("baseline_blend", "har_hv", "garch_1_1", "ensemble")
            ]
        )

    as_of = as_of_date(clean_history)
    returns = log_returns(clean_history["Close"])
    components = realized_vol_components(clean_history)

    forecasts: list[VolatilityForecast] = []
    for horizon in horizon_values:
        baseline = baseline_vol_forecast(components)
        har = har_hv_forecast(returns, horizon_days=horizon)
        garch = garch_1_1_forecast(returns, horizon_days=horizon)
        ensemble = ensemble_forecast(
            {
                "baseline_blend": baseline.forecast_vol,
                "har_hv": har.forecast_vol,
                "garch_1_1": garch.forecast_vol,
            }
        )
        forecasts.extend(
            [
                _with_common(baseline, symbol_key, as_of, horizon),
                _with_common(har, symbol_key, as_of, horizon),
                _with_common(garch, symbol_key, as_of, horizon),
                VolatilityForecast(
                    symbol=symbol_key,
                    as_of=as_of,
                    horizon_days=horizon,
                    model="ensemble",
                    forecast_vol=ensemble,
                    status="ok" if ensemble is not None else "missing",
                    detail="Weighted blend of available baseline/HAR/GARCH forecasts.",
                    components={
                        "baseline_blend": baseline.forecast_vol,
                        "har_hv": har.forecast_vol,
                        "garch_1_1": garch.forecast_vol,
                    },
                ),
            ]
        )
    return forecast_frame(forecasts)


def forecast_frame(forecasts: Iterable[VolatilityForecast]) -> pd.DataFrame:
    rows = [forecast.to_row() for forecast in forecasts]
    columns = [
        "symbol",
        "as_of",
        "horizon_days",
        "model",
        "forecast_vol",
        "status",
        "detail",
        "components",
        "created_at",
    ]
    return pd.DataFrame(rows, columns=columns)


def select_forecast_vol(
    forecasts: pd.DataFrame,
    *,
    horizon_days: int,
    preferred_model: str = "ensemble",
    fallback: float | None = None,
) -> float | None:
    """Pick a forecast vol from a forecast frame."""

    if forecasts.empty:
        return fallback
    frame = forecasts.copy()
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    target = int(horizon_days)
    valid = frame.dropna(subset=["horizon_days"]).copy()
    if valid.empty:
        return fallback
    valid["horizon_distance"] = (valid["horizon_days"].astype(int) - target).abs()
    selected_horizon = int(valid.sort_values(["horizon_distance", "horizon_days"]).iloc[0]["horizon_days"])
    subset = valid[valid["horizon_days"].eq(selected_horizon)]
    preferred = subset[subset["model"].eq(preferred_model)]
    row = preferred.iloc[0] if not preferred.empty else subset.iloc[0]
    value = _optional_float(row.get("forecast_vol"))
    return fallback if value is None else value


def realized_vol_components(history: pd.DataFrame) -> dict[str, float | None]:
    """Compute baseline realized-volatility components."""

    clean = normalize_ohlc_history(history)
    close = pd.to_numeric(clean["Close"], errors="coerce").dropna()
    returns = log_returns(close)
    hist_21 = annualize_daily_std(returns.tail(21))
    hist_63 = annualize_daily_std(returns.tail(63))
    ewma = annualize_daily_std(returns.ewm(alpha=1 - 0.94).std().dropna().tail(1))
    parkinson = parkinson_vol(clean)
    return {
        "historical_vol_21d": hist_21,
        "historical_vol_63d": hist_63,
        "ewma_vol": ewma,
        "parkinson_vol": parkinson,
    }


def baseline_vol_forecast(components: dict[str, float | None]) -> VolatilityForecast:
    values = [_optional_float(value) for value in components.values()]
    values = [value for value in values if value is not None and value > 0]
    forecast = float(np.nanmean(values)) if values else None
    return VolatilityForecast(
        symbol="",
        as_of="",
        horizon_days=1,
        model="baseline_blend",
        forecast_vol=forecast,
        status="ok" if forecast is not None else "missing",
        detail="Mean of 21D HV, 63D HV, EWMA vol, and Parkinson high-low vol.",
        components=components,
    )


def har_hv_forecast(returns: pd.Series, *, horizon_days: int) -> VolatilityForecast:
    """Fit a daily-data HAR-HV model and forecast annualized vol."""

    daily_var = pd.to_numeric(returns, errors="coerce").dropna().pow(2)
    horizon = max(1, int(horizon_days))
    min_obs = max(80, horizon + 35)
    if len(daily_var) < min_obs:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="har_hv",
            forecast_vol=None,
            status="insufficient_data",
            detail=f"Need at least {min_obs} returns for HAR-HV; found {len(daily_var)}.",
        )

    features = pd.DataFrame(index=daily_var.index)
    features["daily"] = daily_var.shift(1)
    features["weekly"] = daily_var.shift(1).rolling(5).mean()
    features["monthly"] = daily_var.shift(1).rolling(22).mean()
    target = daily_var.shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    design = pd.concat([target.rename("target"), features], axis=1).dropna()
    if len(design) < 40:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="har_hv",
            forecast_vol=None,
            status="insufficient_data",
            detail="Not enough complete HAR-HV design rows.",
        )

    x = np.column_stack(
        [
            np.ones(len(design)),
            design["daily"].to_numpy(dtype=float),
            design["weekly"].to_numpy(dtype=float),
            design["monthly"].to_numpy(dtype=float),
        ]
    )
    y = design["target"].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="har_hv",
            forecast_vol=None,
            status="fit_failed",
            detail=str(exc),
        )

    latest_features = features.dropna().iloc[-1]
    prediction = float(
        beta[0]
        + beta[1] * latest_features["daily"]
        + beta[2] * latest_features["weekly"]
        + beta[3] * latest_features["monthly"]
    )
    prediction = max(prediction, 0.0)
    forecast = math.sqrt(prediction * TRADING_DAYS_PER_YEAR)
    return VolatilityForecast(
        symbol="",
        as_of="",
        horizon_days=horizon,
        model="har_hv",
        forecast_vol=float(forecast),
        detail="HAR-HV using lagged daily, weekly, and monthly realized variance.",
        components={
            "beta_const": float(beta[0]),
            "beta_daily": float(beta[1]),
            "beta_weekly": float(beta[2]),
            "beta_monthly": float(beta[3]),
            "design_rows": int(len(design)),
        },
    )


def garch_1_1_forecast(returns: pd.Series, *, horizon_days: int) -> VolatilityForecast:
    """Fit GARCH(1,1) with the `arch` package and forecast annualized vol."""

    clean = pd.to_numeric(returns, errors="coerce").dropna()
    horizon = max(1, int(horizon_days))
    if len(clean) < 80:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="garch_1_1",
            forecast_vol=None,
            status="insufficient_data",
            detail=f"Need at least 80 returns for GARCH; found {len(clean)}.",
        )

    try:
        from arch import arch_model
    except Exception as exc:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="garch_1_1",
            forecast_vol=None,
            status="unavailable",
            detail=f"arch package unavailable: {exc}",
        )

    try:
        model = arch_model(clean * 100, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
        fitted = model.fit(disp="off", show_warning=False)
        forecast = fitted.forecast(horizon=horizon, reindex=False)
        variance = forecast.variance.iloc[-1].to_numpy(dtype=float)
    except Exception as exc:
        return VolatilityForecast(
            symbol="",
            as_of="",
            horizon_days=horizon,
            model="garch_1_1",
            forecast_vol=None,
            status="fit_failed",
            detail=str(exc),
        )

    daily_variance_decimal = float(np.nanmean(variance)) / 10_000
    daily_variance_decimal = max(daily_variance_decimal, 0.0)
    vol = math.sqrt(daily_variance_decimal * TRADING_DAYS_PER_YEAR)
    params = {key: float(value) for key, value in fitted.params.items()}
    return VolatilityForecast(
        symbol="",
        as_of="",
        horizon_days=horizon,
        model="garch_1_1",
        forecast_vol=float(vol),
        detail="GARCH(1,1) forecast from daily log returns.",
        components={"params": params, "aic": float(fitted.aic), "bic": float(fitted.bic)},
    )


def ensemble_forecast(values: dict[str, float | None]) -> float | None:
    weights = {"baseline_blend": 0.30, "har_hv": 0.40, "garch_1_1": 0.30}
    usable = {
        model: _optional_float(value)
        for model, value in values.items()
        if _optional_float(value) is not None and _optional_float(value) > 0
    }
    if not usable:
        return None
    total_weight = sum(weights.get(model, 1.0) for model in usable)
    return float(sum(usable[model] * weights.get(model, 1.0) for model in usable) / total_weight)


def normalize_ohlc_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy()
    if "Date" not in frame.columns:
        frame = frame.reset_index()
    columns = {str(column).lower().replace("_", " ").strip(): column for column in frame.columns}
    date_col = _first_existing(columns, ("date", "datetime", "timestamp", "index"))
    close_col = _first_existing(columns, ("close", "adj close", "adj_close", "price"))
    if date_col is None or close_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    out["Close"] = pd.to_numeric(frame[close_col], errors="coerce")
    for target, candidates in {
        "Open": ("open",),
        "High": ("high",),
        "Low": ("low",),
        "Volume": ("volume",),
    }.items():
        column = _first_existing(columns, candidates)
        out[target] = pd.to_numeric(frame[column], errors="coerce") if column is not None else pd.NA
    out = out.dropna(subset=["Date", "Close"])
    return out.set_index("Date").sort_index()


def log_returns(close: pd.Series) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce").dropna()
    return np.log(prices / prices.shift(1)).dropna()


def annualize_daily_std(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    value = float(clean.iloc[-1]) if len(clean) == 1 else float(clean.std())
    if not math.isfinite(value):
        return None
    return value * math.sqrt(TRADING_DAYS_PER_YEAR)


def parkinson_vol(history: pd.DataFrame) -> float | None:
    if not {"High", "Low"}.issubset(history.columns):
        return None
    high = pd.to_numeric(history["High"], errors="coerce")
    low = pd.to_numeric(history["Low"], errors="coerce")
    valid = (high > 0) & (low > 0)
    if not valid.any():
        return None
    daily = ((np.log(high[valid] / low[valid])) ** 2) / (4 * math.log(2))
    value = np.sqrt(daily.tail(63).mean()) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return None if pd.isna(value) else float(value)


def as_of_date(history: pd.DataFrame) -> str:
    if history.empty:
        return ""
    index = pd.to_datetime(history.index, errors="coerce").dropna()
    if index.empty:
        return ""
    return index.max().date().isoformat()


def normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _with_common(
    forecast: VolatilityForecast,
    symbol: str,
    as_of: str,
    horizon_days: int,
) -> VolatilityForecast:
    return VolatilityForecast(
        symbol=symbol,
        as_of=as_of,
        horizon_days=horizon_days,
        model=forecast.model,
        forecast_vol=forecast.forecast_vol,
        status=forecast.status,
        detail=forecast.detail,
        components=forecast.components,
    )


def _first_existing(columns: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or not math.isfinite(parsed):
        return None
    return parsed
