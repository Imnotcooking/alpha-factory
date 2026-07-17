"""Deterministic fixtures for the broker-free Alpha Factory demo profile."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oqp.accounts import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    TradeEvent,
    write_account_snapshot,
    write_account_trade_events,
)
from oqp.demo.profile import DEMO_PROFILE, PROFILE_SCHEMA_VERSION, DemoPaths, demo_paths, read_profile_marker


DEMO_SEED = 20260717
DEMO_FACTOR_IDS = ("demo_sma_trend", "demo_carry", "demo_ml_blend")


@dataclass(frozen=True, slots=True)
class DemoSeedResult:
    profile: str
    paths: DemoPaths
    as_of: str
    seed: int
    files_written: tuple[Path, ...]
    research_runs: int
    account_snapshots: int
    option_contracts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "runtime_root": str(self.paths.runtime_root),
            "as_of": self.as_of,
            "seed": self.seed,
            "files_written": [str(path) for path in self.files_written],
            "research_runs": self.research_runs,
            "account_snapshots": self.account_snapshots,
            "option_contracts": self.option_contracts,
        }


def seed_demo_profile(
    repo_root: str | Path | None = None,
    *,
    as_of: str | date | None = None,
    seed: int = DEMO_SEED,
    force: bool = False,
) -> DemoSeedResult:
    """Build a complete, isolated demo runtime without vendor or broker access."""

    paths = demo_paths(repo_root)
    existing_marker = read_profile_marker(paths.repo_root)
    existing_profile = str(existing_marker.get("profile") or "")
    if existing_profile and existing_profile != DEMO_PROFILE and not force:
        raise RuntimeError(
            f"Runtime profile is {existing_profile!r}; pass --force to select the demo profile."
        )

    as_of_date = pd.Timestamp(as_of or date.today()).date()
    # These databases are owned exclusively by the demo profile. Rebuilding
    # them avoids stale rows and makes repeated seeds byte-for-byte stable.
    for database in (
        paths.research_db,
        paths.account_ledger,
        paths.portfolio_ledger,
        paths.paper_ledger,
    ):
        if database.exists():
            database.unlink()
    for directory in (
        paths.data_root,
        paths.artifact_root,
        paths.research_db.parent,
        paths.account_ledger.parent,
        paths.portfolio_ledger.parent,
        paths.paper_ledger.parent,
        paths.profile_marker.parent,
        paths.runtime_root / "cache" / "matplotlib",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    files.extend(_seed_market_data(paths, as_of_date, seed))
    option_file, option_count = _seed_option_chain(paths, as_of_date, seed + 11)
    files.append(option_file)
    research_files, run_count = _seed_research(paths, as_of_date, seed + 23)
    files.extend(research_files)
    snapshot_count = _seed_accounts(paths, as_of_date, seed + 37)

    files = list(dict.fromkeys(path.resolve() for path in files))
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": DEMO_PROFILE,
        "seed": seed,
        "as_of": as_of_date.isoformat(),
        "research_runs": run_count,
        "account_snapshots": snapshot_count,
        "option_contracts": option_count,
        "files": [
            {
                "path": _display_path(path, paths.repo_root),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
            if path.exists()
        ],
    }
    paths.seed_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    created_at = existing_marker.get("created_at")
    if not created_at or existing_profile != DEMO_PROFILE:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    marker = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": DEMO_PROFILE,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_root": _display_path(paths.runtime_root, paths.repo_root),
        "seed_manifest": _display_path(paths.seed_manifest, paths.repo_root),
    }
    paths.profile_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    files.extend((paths.seed_manifest.resolve(), paths.profile_marker.resolve()))
    return DemoSeedResult(
        profile=DEMO_PROFILE,
        paths=paths,
        as_of=as_of_date.isoformat(),
        seed=seed,
        files_written=tuple(files),
        research_runs=run_count,
        account_snapshots=snapshot_count,
        option_contracts=option_count,
    )


def _seed_market_data(paths: DemoPaths, as_of: date, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    daily_dir = paths.data_root / "futures_cn" / "daily"
    intraday_dir = paths.data_root / "futures_cn" / "intraday"
    daily_dir.mkdir(parents=True, exist_ok=True)
    intraday_dir.mkdir(parents=True, exist_ok=True)

    specifications = (
        ("rb", "Ferrous", 3650.0, 0.013, 1.00),
        ("cu", "Base Metals", 72000.0, 0.011, 0.72),
        ("au", "Precious Metals", 560.0, 0.009, 0.35),
        ("ag", "Precious Metals", 7100.0, 0.016, 0.48),
        ("TA", "Chemicals", 5900.0, 0.014, 0.60),
        ("MA", "Chemicals", 2550.0, 0.015, 0.54),
        ("IF", "Equity Index", 4100.0, 0.010, 0.22),
        ("sc", "Energy", 610.0, 0.018, 0.82),
    )
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=504)
    market_shock = rng.normal(0.00015, 0.007, len(dates))
    rows: list[pd.DataFrame] = []
    for symbol, industry, base, vol, beta in specifications:
        idiosyncratic = rng.normal(0.00008, vol, len(dates))
        returns = beta * market_shock + math.sqrt(max(1.0 - beta**2, 0.05)) * idiosyncratic
        close = base * np.exp(np.cumsum(returns))
        overnight = rng.normal(0.0, vol * 0.22, len(dates))
        open_price = close * np.exp(overnight)
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0005, vol * 0.8, len(dates)))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0005, vol * 0.8, len(dates)))
        volume = rng.lognormal(mean=11.0, sigma=0.35, size=len(dates)).round()
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                    "open_interest": rng.lognormal(10.5, 0.28, len(dates)).round(),
                    "industry": industry,
                    "sector": industry,
                    "asset_class": "FUTURES_CN",
                    "is_fresh": True,
                }
            )
        )
    daily = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"])
    daily_path = daily_dir / "demo_futures_cn_daily.parquet"
    daily.to_parquet(daily_path, index=False)

    intraday_rows: list[pd.DataFrame] = []
    intraday_days = pd.bdate_range(end=pd.Timestamp(as_of), periods=5)
    for symbol, industry, base, vol, _ in specifications[:4]:
        price = float(base)
        for trading_day in intraday_days:
            morning = pd.date_range(
                datetime.combine(trading_day.date(), time(9, 0)),
                datetime.combine(trading_day.date(), time(11, 29)),
                freq="1min",
            )
            afternoon = pd.date_range(
                datetime.combine(trading_day.date(), time(13, 30)),
                datetime.combine(trading_day.date(), time(14, 59)),
                freq="1min",
            )
            stamps = morning.append(afternoon)
            minute_returns = rng.normal(0.0, vol / math.sqrt(240), len(stamps))
            close = price * np.exp(np.cumsum(minute_returns))
            open_price = np.r_[price, close[:-1]]
            spread = np.maximum(close * vol / 90.0, 0.01)
            volume = rng.lognormal(6.5, 0.55, len(stamps)).round()
            intraday_rows.append(
                pd.DataFrame(
                    {
                        "datetime": stamps,
                        "date": stamps.normalize(),
                        "ticker": symbol,
                        "symbol": symbol,
                        "open": open_price,
                        "high": np.maximum(open_price, close) + spread,
                        "low": np.minimum(open_price, close) - spread,
                        "close": close,
                        "last_price": close,
                        "volume": volume,
                        "turnover": volume * close,
                        "open_interest": rng.lognormal(9.7, 0.22, len(stamps)).round(),
                        "bid_price_1": close - spread / 2.0,
                        "ask_price_1": close + spread / 2.0,
                        "industry": industry,
                        "asset_class": "FUTURES_CN",
                        "is_fresh": True,
                    }
                )
            )
            price = float(close[-1])
    intraday = pd.concat(intraday_rows, ignore_index=True).sort_values(["datetime", "ticker"])
    intraday_path = intraday_dir / "demo_futures_cn_1m.parquet"
    intraday.to_parquet(intraday_path, index=False)
    return [daily_path, intraday_path]


def _seed_option_chain(paths: DemoPaths, as_of: date, seed: int) -> tuple[Path, int]:
    rng = np.random.default_rng(seed)
    output_dir = paths.data_root / "options_us" / "api_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    underlyings = {"SPY": 650.0, "QQQ": 580.0}
    for underlying, spot in underlyings.items():
        for dte in (30, 60):
            expiry = as_of + timedelta(days=dte)
            for strike_ratio in (0.92, 0.96, 1.0, 1.04, 1.08):
                strike = round(spot * strike_ratio, 2)
                for right in ("call", "put"):
                    intrinsic = max(spot - strike, 0.0) if right == "call" else max(strike - spot, 0.0)
                    time_value = spot * 0.18 * math.sqrt(dte / 365.0) * math.exp(-abs(strike_ratio - 1.0) * 5)
                    mark = max(intrinsic + time_value * 0.35, 0.05)
                    spread = max(mark * 0.035, 0.02)
                    delta = (0.52 - (strike_ratio - 1.0) * 3.2) if right == "call" else (-0.48 - (strike_ratio - 1.0) * 3.2)
                    symbol = f"{underlying}{expiry:%y%m%d}{'C' if right == 'call' else 'P'}{int(strike * 1000):08d}"
                    rows.append(
                        {
                            "date": as_of,
                            "timestamp": datetime.combine(as_of, time(20, 0), tzinfo=timezone.utc),
                            "option_symbol": symbol,
                            "vendor_symbol": f"O:{symbol}",
                            "market_vertical": "OPTIONS_US",
                            "exchange": "DEMO",
                            "underlying_symbol": underlying,
                            "underlying_type": "equity_etf",
                            "expiry": expiry,
                            "right": right,
                            "strike": strike,
                            "multiplier": 100.0,
                            "currency": "USD",
                            "exercise_style": "american",
                            "settlement_style": "physical",
                            "bid": mark - spread / 2.0,
                            "ask": mark + spread / 2.0,
                            "mid": mark,
                            "last": mark * rng.normal(1.0, 0.01),
                            "mark": mark,
                            "close": mark,
                            "volume": int(rng.integers(100, 4000)),
                            "open_interest": int(rng.integers(1000, 25000)),
                            "implied_volatility": 0.18 + abs(strike_ratio - 1.0) * 0.22,
                            "delta": float(np.clip(delta, -0.98, 0.98)),
                            "gamma": 0.008 * math.exp(-abs(strike_ratio - 1.0) * 8),
                            "theta": -mark / max(dte, 1) * 0.4,
                            "vega": spot * 0.01 * math.sqrt(dte / 365.0),
                            "quote_timestamp": datetime.combine(as_of, time(20, 0), tzinfo=timezone.utc),
                            "quote_source": "oqp_demo",
                        }
                    )
    frame = pd.DataFrame(rows)
    path = output_dir / "demo_options_us_chain.parquet"
    frame.to_parquet(path, index=False)
    return path, len(frame)


def _seed_research(paths: DemoPaths, as_of: date, seed: int) -> tuple[list[Path], int]:
    rng = np.random.default_rng(seed)
    returns_dir = paths.artifact_root / "returns"
    trades_dir = paths.artifact_root / "trades"
    importance_dir = paths.artifact_root / "feature_importance"
    assumptions_dir = paths.artifact_root / "assumptions"
    demo_dir = paths.artifact_root / "demo"
    for directory in (returns_dir, trades_dir, importance_dir, assumptions_dir, demo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=504)
    benchmark = rng.normal(0.00022, 0.0105, len(dates))
    run_specs = (
        ("demo_trend_r1", "demo_sma_trend", 1, 0.038, 0.031, 0.10, 0.074, -0.118, 0.82, "run_backtest", "heuristic"),
        ("demo_trend_r2", "demo_sma_trend", 2, 0.043, 0.036, 0.085, 0.096, -0.102, 1.01, "run_backtest", "heuristic"),
        ("demo_trend_r3", "demo_sma_trend", 3, 0.049, 0.041, 0.072, 0.119, -0.091, 1.22, "run_backtest", "heuristic"),
        ("demo_carry_r1", "demo_carry", 1, 0.031, 0.027, 0.045, 0.081, -0.073, 1.05, "run_backtest", "heuristic"),
        ("demo_ml_r1", "demo_ml_blend", 1, 0.055, 0.046, 0.115, 0.137, -0.086, 1.36, "run_ml_backtest", "xgboost"),
    )
    factors = (
        ("demo_sma_trend", "Demo CN Futures Trend", "Trend", "Medium-horizon continuation after volatility scaling.", 2, f"{as_of.isoformat()}T09:00:00"),
        ("demo_carry", "Demo Cross-Asset Carry", "Carry", "Relative carry rewards assets with favorable roll economics.", 2, f"{as_of.isoformat()}T09:05:00"),
        ("demo_ml_blend", "Demo ML Factor Blend", "Machine Learning", "A transparent synthetic blend used to demonstrate model diagnostics.", 4, f"{as_of.isoformat()}T09:10:00"),
    )
    factor_map = {row[0]: row for row in factors}
    files: list[Path] = []
    run_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(run_specs):
        run_id, factor_id, round_number, val_ic, holdout_ic, turnover, ann_ret, max_dd, sharpe, runner, model = spec
        signal_strength = 0.00035 + index * 0.000035
        noise = rng.normal(0.0, 0.0068 - index * 0.00025, len(dates))
        net_return = 0.20 * benchmark + signal_strength + noise
        gross_return = net_return + turnover / 252.0 * 0.0008
        leverage = np.clip(0.55 + np.abs(rng.normal(0.0, 0.26, len(dates))), 0.2, 1.55)
        equity = 1_000_000.0 * np.cumprod(1.0 + net_return)
        long_notional = equity * leverage * rng.uniform(0.45, 0.72, len(dates))
        short_notional = equity * leverage - long_notional
        returns = pd.DataFrame(
            {
                "date": dates,
                "net_return": net_return,
                "gross_return": gross_return,
                "benchmark_return": benchmark,
                "portfolio_leverage": leverage,
                "gross_leverage": leverage,
                "daily_turnover": np.clip(rng.normal(turnover, turnover * 0.24, len(dates)), 0.0, None),
                "equity": equity,
                "long_notional": long_notional,
                "short_notional": short_notional,
                "initial_capital": 1_000_000.0,
            }
        )
        returns_path = returns_dir / f"returns_{run_id}.csv"
        returns.to_csv(returns_path, index=False)
        files.append(returns_path)

        trades = _demo_trades(dates, rng, count=180 + index * 12)
        trades_path = trades_dir / f"trades_{run_id}.csv"
        trades.to_csv(trades_path, index=False)
        files.append(trades_path)

        assumption_path = assumptions_dir / f"assumptions_{run_id}.json"
        assumption_path.write_text(
            json.dumps(
                {
                    "profile": DEMO_PROFILE,
                    "data": {
                        "asset_class": "FUTURES_CN",
                        "data_frequency": "1d",
                        "prepared_data_start": dates.min().date().isoformat(),
                        "prepared_data_end": dates.max().date().isoformat(),
                        "prepared_data_rows": int(len(dates) * 8),
                        "vendor": "oqp_demo",
                    },
                    "benchmark": {
                        "benchmark_label": "Equal-weight CN futures demo universe",
                        "benchmark_column": "benchmark_return",
                        "return_mode": "daily",
                    },
                    "signal_and_execution_mode": {
                        "runner": runner,
                        "execution_lag": "1 bar",
                        "cost_model": "2 bps round trip",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        files.append(assumption_path)
        run_rows.append(
            {
                "run_id": run_id,
                "factor_id": factor_id,
                "round_number": round_number,
                "validation_ic": val_ic,
                "holdout_ic": holdout_ic,
                "crisis_ic": holdout_ic * 0.58,
                "turnover_rate": turnover,
                "annualized_return": ann_ret,
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe,
                "total_trades": len(trades),
                "asset_class": "FUTURES_CN",
                "market_vertical": "FUTURES_CN",
                "dataset_id": "oqp_demo_futures_cn_daily",
                "universe_id": "demo_cn_futures_8",
                "data_frequency": "1d",
                "dataset_role": "synthetic_demo",
                "data_tradability": "illustrative_only",
                "data_price_source": "synthetic_ohlc",
                "data_execution_reality": "research_simulation",
                "data_vendor": "oqp_demo",
                "execution_assumption": "next_bar_close_with_costs",
                "factor_contract_source": "oqp.demo.seed",
                "alpha_signal_col": "factor_score",
                "execution_weight_col": "target_weight",
                "execution_mode": "vectorized",
                "execution_lag": "1 bar",
                "return_assumption": "close_to_close",
                "universe_size": 8,
                "traded_tickers": "ALL",
                "returns_file_path": _display_path(returns_path, paths.repo_root),
                "evaluation_geometry": "cross_sectional_daily",
                "validation_rows": 252 * 8,
                "holdout_rows": 126 * 8,
                "crisis_rows": 42 * 8,
                "avg_daily_cost_bps": 1.2 + turnover * 4,
                "initial_capital": 1_000_000.0,
                "capital_currency": "CNY",
                "research_family": factor_id,
                "stat_research_family": factor_id,
                "backtest_engine": "event_driven_python" if model == "xgboost" else "vectorized_cpp_optional",
                "runner": runner,
                "runner_name": runner,
                "engine_type": "ml" if model == "xgboost" else "heuristic",
                "model_type": model,
                "model_family": model,
                "strategy_type": "single_factor",
                "timestamp": datetime.combine(as_of - timedelta(days=len(run_specs) - index), time(12, 0)).isoformat(),
            }
        )

    importance_path = importance_dir / "feature_importance_demo_ml_r1.csv"
    pd.DataFrame(
        {
            "feature": ["trend_20d", "carry_60d", "volatility_20d", "breadth_state", "liquidity_score"],
            "importance": [0.31, 0.24, 0.19, 0.16, 0.10],
        }
    ).to_csv(importance_path, index=False)
    files.append(importance_path)

    factor_metadata_path = demo_dir / "factor_metadata.json"
    factor_metadata_path.write_text(
        json.dumps(
            {
                factor_id: {
                    "name": factor_map[factor_id][1],
                    "category": factor_map[factor_id][2],
                    "supported_markets": ["FUTURES_CN"],
                    "data_frequency": "1d",
                    "demo_only": True,
                }
                for factor_id in DEMO_FACTOR_IDS
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files.append(factor_metadata_path)
    _write_research_database(paths.research_db, factors, run_rows)
    files.append(paths.research_db)
    return files, len(run_rows)


def _demo_trades(dates: pd.DatetimeIndex, rng: np.random.Generator, *, count: int) -> pd.DataFrame:
    tickers = np.array(["rb", "cu", "au", "ag", "TA", "MA", "IF", "sc"])
    industries = {
        "rb": "Ferrous",
        "cu": "Base Metals",
        "au": "Precious Metals",
        "ag": "Precious Metals",
        "TA": "Chemicals",
        "MA": "Chemicals",
        "IF": "Equity Index",
        "sc": "Energy",
    }
    selected = rng.choice(tickers, size=count, replace=True)
    entries = pd.to_datetime(rng.choice(dates[:-8], size=count, replace=True)).sort_values()
    hold_hours = rng.integers(24, 145, size=count)
    directions = rng.choice(["Long", "Short"], size=count, p=[0.56, 0.44])
    pnl = np.clip(rng.normal(0.0052, 0.021, size=count), -0.085, 0.11)
    entry_price = rng.lognormal(7.8, 0.55, size=count)
    signed_move = pnl * np.where(directions == "Long", 1.0, -1.0)
    return pd.DataFrame(
        {
            "trade_id": [f"demo_trade_{index:04d}" for index in range(count)],
            "ticker": selected,
            "company_name": [f"{industries[str(symbol)]} future" for symbol in selected],
            "asset_name_zh": selected,
            "entry_time": entries,
            "exit_time": entries + pd.to_timedelta(hold_hours, unit="h"),
            "direction": directions,
            "entry_price": entry_price,
            "exit_price": entry_price * (1.0 + signed_move),
            "holding_period_hours": hold_hours,
            "trade_pnl": pnl,
            "win_loss_flag": np.where(pnl >= 0, "Win", "Loss"),
        }
    )


def _write_research_database(
    db_path: Path,
    factors: tuple[tuple[Any, ...], ...],
    run_rows: list[dict[str, Any]],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_columns = {
        "run_id": "TEXT PRIMARY KEY",
        "factor_id": "TEXT",
        "round_number": "INTEGER",
        "validation_ic": "REAL",
        "holdout_ic": "REAL",
        "crisis_ic": "REAL",
        "turnover_rate": "REAL",
        "annualized_return": "REAL",
        "max_drawdown": "REAL",
        "sharpe_ratio": "REAL",
        "total_trades": "INTEGER",
        "asset_class": "TEXT",
        "market_vertical": "TEXT",
        "dataset_id": "TEXT",
        "universe_id": "TEXT",
        "data_frequency": "TEXT",
        "dataset_role": "TEXT",
        "data_tradability": "TEXT",
        "data_price_source": "TEXT",
        "data_execution_reality": "TEXT",
        "data_vendor": "TEXT",
        "execution_assumption": "TEXT",
        "factor_contract_source": "TEXT",
        "alpha_signal_col": "TEXT",
        "execution_weight_col": "TEXT",
        "execution_mode": "TEXT",
        "execution_lag": "TEXT",
        "return_assumption": "TEXT",
        "universe_size": "INTEGER",
        "traded_tickers": "TEXT",
        "returns_file_path": "TEXT",
        "evaluation_geometry": "TEXT",
        "validation_rows": "INTEGER",
        "holdout_rows": "INTEGER",
        "crisis_rows": "INTEGER",
        "raw_event_count": "INTEGER",
        "quality_event_count": "INTEGER",
        "throttled_event_count": "INTEGER",
        "active_tick_count": "INTEGER",
        "avg_daily_cost_bps": "REAL",
        "initial_capital": "REAL",
        "capital_currency": "TEXT",
        "research_family": "TEXT",
        "stat_research_family": "TEXT",
        "backtest_engine": "TEXT",
        "runner": "TEXT",
        "runner_name": "TEXT",
        "engine_type": "TEXT",
        "model_type": "TEXT",
        "model_family": "TEXT",
        "strategy_type": "TEXT",
        "timestamp": "TEXT",
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                economic_rationale TEXT,
                complexity_score INTEGER,
                status TEXT DEFAULT 'DEMO',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns_sql = ", ".join(f"{name} {kind}" for name, kind in run_columns.items())
        conn.execute(f"CREATE TABLE IF NOT EXISTS backtest_runs ({columns_sql})")
        existing = {row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)")}
        for name, kind in run_columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {name} {kind.replace(' PRIMARY KEY', '')}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                failure_code TEXT,
                suggested_action TEXT
            )
            """
        )
        conn.execute("DELETE FROM diagnostics WHERE run_id LIKE 'demo_%'")
        conn.execute("DELETE FROM backtest_runs WHERE run_id LIKE 'demo_%'")
        conn.execute("DELETE FROM factors WHERE factor_id LIKE 'demo_%'")
        conn.executemany(
            """
            INSERT INTO factors (
                factor_id, name, category, economic_rationale, complexity_score, status, created_at
            ) VALUES (?, ?, ?, ?, ?, 'DEMO', ?)
            """,
            factors,
        )
        for row in run_rows:
            columns = list(row)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO backtest_runs ({', '.join(columns)}) VALUES ({placeholders})",
                [row[column] for column in columns],
            )
        conn.commit()


def _seed_accounts(paths: DemoPaths, as_of: date, seed: int) -> int:
    rng = np.random.default_rng(seed)
    option_expiry = as_of + timedelta(days=90)
    option_symbol = f"SPY{option_expiry:%y%m%d}C{int(675 * 1000):08d}"
    snapshot_count = 0
    for environment, profile, base_nav, days in (
        (AccountEnvironment.LIVE, "demo_live_readonly", 250_000.0, 60),
        (AccountEnvironment.PAPER, "demo_paper", 100_000.0, 45),
    ):
        dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=days)
        nav_returns = rng.normal(0.00035, 0.0065, len(dates))
        nav_values = base_nav * np.cumprod(1.0 + nav_returns)
        for index, (stamp, nav) in enumerate(zip(dates, nav_values, strict=True)):
            latest = index == len(dates) - 1
            positions = _demo_positions(option_symbol, option_expiry, rng, latest=latest)
            cash = float(nav * 0.28)
            snapshot = AccountSnapshot(
                snapshot_id=f"demo_{environment.value}_{stamp:%Y%m%d}",
                as_of=datetime.combine(stamp.date(), time(20, 0), tzinfo=timezone.utc),
                account_id=f"DEMO-{environment.value.upper()}",
                broker="oqp_demo",
                profile=profile,
                environment=environment,
                currency="USD",
                net_liquidation=float(nav),
                cash=cash,
                buying_power=cash * 2.0,
                gross_position_value=sum(abs(position.market_value or 0.0) for position in positions),
                margin_buffer=cash * 1.65,
                positions=positions,
                cash_balances=(CashSnapshot(currency="USD", cash=cash, settled_cash=cash),),
                metadata={"demo": True, "read_only": True},
            )
            write_account_snapshot(paths.account_ledger, snapshot, snapshot_date=stamp.date())
            snapshot_count += 1

    trade_symbols = ("SPY", "QQQ", "TLT", "GLD", option_symbol)
    events = []
    for index in range(12):
        symbol = trade_symbols[index % len(trade_symbols)]
        is_option = symbol == option_symbol
        events.append(
            TradeEvent(
                event_id=f"demo_event_{index:03d}",
                event_type="fill",
                occurred_at=datetime.combine(
                    as_of - timedelta(days=12 - index),
                    time(15, 30),
                    tzinfo=timezone.utc,
                ),
                account_id="DEMO-PAPER",
                broker="oqp_demo",
                profile="demo_paper",
                environment=AccountEnvironment.PAPER,
                symbol=symbol,
                side="BUY" if index % 3 else "SELL",
                quantity=1.0 if is_option else float(5 + index),
                price=8.4 if is_option else float(100 + index * 13),
                commission=0.65 if is_option else 0.15,
                currency="USD",
                strategy_id="demo_sma_trend",
                order_id=f"demo_order_{index:03d}",
                metadata={"demo": True},
            )
        )
    write_account_trade_events(paths.account_ledger, events)
    return snapshot_count


def _demo_positions(
    option_symbol: str,
    option_expiry: date,
    rng: np.random.Generator,
    *,
    latest: bool,
) -> tuple[PositionSnapshot, ...]:
    prices = {"SPY": 650.0, "QQQ": 580.0, "TLT": 88.0, "GLD": 305.0}
    quantities = {"SPY": 90.0, "QQQ": 55.0, "TLT": 180.0, "GLD": 70.0}
    positions: list[PositionSnapshot] = []
    for symbol in prices:
        price = prices[symbol] * float(rng.normal(1.0, 0.012 if latest else 0.02))
        average_cost = prices[symbol] * (0.91 if symbol in {"SPY", "GLD"} else 0.96)
        quantity = quantities[symbol]
        positions.append(
            PositionSnapshot(
                symbol=symbol,
                asset_class="EQUITY_US",
                quantity=quantity,
                average_cost=average_cost,
                market_price=price,
                market_value=quantity * price,
                unrealized_pnl=quantity * (price - average_cost),
                currency="USD",
                metadata={"demo": True, "sleeve": "core"},
            )
        )
    positions.append(
        PositionSnapshot(
            symbol=option_symbol,
            asset_class="OPTIONS_US",
            quantity=3.0,
            average_cost=7.6,
            market_price=8.4,
            market_value=3.0 * 8.4 * 100.0,
            unrealized_pnl=3.0 * (8.4 - 7.6) * 100.0,
            currency="USD",
            multiplier=100.0,
            metadata={
                "demo": True,
                "underlying_symbol": "SPY",
                "expiry": option_expiry.isoformat(),
                "right": "call",
                "strike": 675.0,
                "delta": 0.43,
                "gamma": 0.008,
                "theta": -0.19,
                "vega": 0.71,
            },
        )
    )
    return tuple(positions)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(path) as conn:
            for statement in conn.iterdump():
                digest.update(statement.encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
