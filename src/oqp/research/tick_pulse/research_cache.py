from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from oqp.research.trials import record_research_trial


CACHE_SCHEMA_VERSION = "tick_pulse_research_cache_v5"


@dataclass(frozen=True)
class TickPulseCacheResult:
    data: pd.DataFrame
    cache_hit: bool
    cache_key: str
    artifact_path: str
    elapsed_seconds: float
    backend: str


def make_cache_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def ensure_tick_pulse_cache_tables(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_pulse_research_cache (
                cache_key TEXT PRIMARY KEY,
                cache_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_mtime REAL NOT NULL,
                product TEXT,
                symbol TEXT,
                hypothesis TEXT,
                threshold_mode TEXT,
                window INTEGER,
                min_success_ticks REAL,
                horizon_set TEXT,
                backend TEXT,
                artifact_path TEXT NOT NULL,
                row_count INTEGER,
                elapsed_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def get_tick_pulse_cache_metadata(db_path: str, cache_key: str) -> dict[str, Any] | None:
    ensure_tick_pulse_cache_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tick_pulse_research_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    return dict(row) if row else None


def load_cached_dataframe(db_path: str, cache_key: str, base_dir: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    metadata = get_tick_pulse_cache_metadata(db_path, cache_key)
    if not metadata:
        return None

    artifact_path = _resolve_artifact_path(metadata["artifact_path"], base_dir)
    if not artifact_path.exists():
        return None

    return _coerce_dataframe(pd.read_parquet(artifact_path)), metadata


def get_or_compute_dataframe(
    *,
    db_path: str,
    logs_dir: str,
    base_dir: str,
    cache_key: str,
    metadata: dict[str, Any],
    compute_fn: Callable[[], pd.DataFrame],
) -> TickPulseCacheResult:
    cached = load_cached_dataframe(db_path, cache_key, base_dir)
    if cached is not None:
        data, row = cached
        _record_research_trials_for_cache(db_path, cache_key, row, data)
        return TickPulseCacheResult(
            data=data,
            cache_hit=True,
            cache_key=cache_key,
            artifact_path=str(row["artifact_path"]),
            elapsed_seconds=float(row["elapsed_seconds"] or 0.0),
            backend=str(row["backend"] or "sqlite_cache"),
        )

    started = time.perf_counter()
    data = _coerce_dataframe(compute_fn())
    elapsed = time.perf_counter() - started

    artifact_dir = Path(logs_dir) / "tick_pulse_cache"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_abs = artifact_dir / f"{cache_key}.parquet"
    data.to_parquet(artifact_abs, index=False)
    artifact_rel = os.path.relpath(artifact_abs, base_dir)

    ensure_tick_pulse_cache_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tick_pulse_research_cache (
                cache_key, cache_type, schema_version, source_file, source_mtime,
                product, symbol, hypothesis, threshold_mode, window,
                min_success_ticks, horizon_set, backend, artifact_path,
                row_count, elapsed_seconds, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                cache_key,
                metadata.get("cache_type", "horizon_sweep"),
                CACHE_SCHEMA_VERSION,
                metadata.get("source_file", ""),
                float(metadata.get("source_mtime", 0.0)),
                metadata.get("product", ""),
                metadata.get("symbol", ""),
                metadata.get("hypothesis", ""),
                metadata.get("threshold_mode", ""),
                int(metadata.get("window", 0)),
                float(metadata.get("min_success_ticks", 0.0)),
                metadata.get("horizon_set", ""),
                metadata.get("backend", "python"),
                artifact_rel,
                int(len(data)),
                float(elapsed),
            ),
        )
        conn.commit()

    _record_research_trials_for_cache(db_path, cache_key, metadata, data)
    return TickPulseCacheResult(
        data=data,
        cache_hit=False,
        cache_key=cache_key,
        artifact_path=artifact_rel,
        elapsed_seconds=elapsed,
        backend=metadata.get("backend", "python"),
    )


def _resolve_artifact_path(path_text: str, base_dir: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, tuple) and value and isinstance(value[0], pd.DataFrame):
        value = value[0]
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Tick pulse cache expected a pandas DataFrame, got {type(value)!r}")
    return value.copy()


def _record_research_trials_for_cache(
    db_path: str,
    cache_key: str,
    metadata: dict[str, Any],
    data: pd.DataFrame,
) -> None:
    cache_type = str(metadata.get("cache_type", ""))
    if cache_type not in {"math_horizon_sweep", "cross_asset_main_contract_sweep"}:
        return
    if data.empty or "hypothesis" not in data.columns or "horizon" not in data.columns:
        return

    for idx, row in data.reset_index(drop=True).iterrows():
        events = _safe_int(row.get("events"))
        if events <= 0:
            continue
        successes = _safe_int(row.get("successes"))
        accuracy = _safe_float(row.get("accuracy"))
        base_rate = _safe_float(row.get("base_rate"))
        if successes < 0 and np.isfinite(accuracy):
            successes = int(round(accuracy * events))

        p_value = _binomial_lift_p_value(successes, events, base_rate)
        hypothesis = str(row.get("hypothesis") or metadata.get("hypothesis") or "")
        source_file = str(row.get("source_file") or metadata.get("source_file") or "")
        product = str(row.get("asset") or metadata.get("product") or "")
        symbol = str(row.get("main_contract") or metadata.get("symbol") or "")
        threshold_columns = {
            key: _safe_float(row.get(key))
            for key in data.columns
            if str(key).startswith("threshold_")
        }
        signature_payload = {
            "experiment_source": "tick_pulse_ui_cache",
            "cache_type": cache_type,
            "source_file": source_file,
            "product": product,
            "symbol": symbol,
            "hypothesis": hypothesis,
            "horizon": _safe_int(row.get("horizon")),
            "window": _safe_int(metadata.get("window")),
            "min_success_ticks": _safe_float(metadata.get("min_success_ticks")),
            "threshold_mode": str(row.get("threshold_mode") or metadata.get("threshold_mode") or ""),
            "threshold_rule_code": str(row.get("threshold_rule_code") or ""),
            "horizon_set": str(metadata.get("horizon_set") or ""),
            **threshold_columns,
        }
        vertical_metadata = {
            "market_vertical": "FUTURES_CN",
            "dataset_id": source_file,
            "data_frequency": "tick",
            "dataset_role": "contract_tick" if symbol and symbol != "MAIN_CONTRACT" else "tick_research_sweep",
            "data_tradability": "executable",
            "data_price_source": "contract_l1_tick",
            "data_vendor": "local_parquet",
            "execution_assumption": "event_study_forward_horizon",
        }
        record_research_trial(
            db_path,
            factor_id=f"tick_pulse_{hypothesis}" if hypothesis else "tick_pulse_lab",
            research_family="tick_pulse_lab",
            trial_signature_payload=signature_payload,
            params=signature_payload,
            experiment_source="tick_pulse_ui_cache",
            asset_class="FUTURES_CN",
            vertical_metadata=vertical_metadata,
            evaluation_geometry="time_series_event_study",
            metric_name="event_accuracy_vs_base_rate",
            metric_value=_safe_float(row.get("lift")),
            raw_p_value=p_value,
            hit_rate_p_value=p_value,
            sample_size=events,
            metadata={
                "cache_key": cache_key,
                "cache_type": cache_type,
                "events": events,
                "successes": successes,
                "accuracy": accuracy,
                "base_rate": base_rate,
                "ci_low": _safe_float(row.get("ci_low")),
                "ci_high": _safe_float(row.get("ci_high")),
                "backend": str(row.get("backend") or metadata.get("backend") or ""),
            },
        )


def _binomial_lift_p_value(successes: int, events: int, base_rate: float) -> float:
    if events <= 0 or successes < 0 or not np.isfinite(base_rate):
        return np.nan
    if base_rate <= 0.0:
        return 0.0 if successes > 0 else 1.0
    if base_rate >= 1.0:
        return 1.0
    return float(stats.binomtest(int(successes), int(events), float(base_rate), alternative="greater").pvalue)


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1
