"""Compact dashboard-only spread summary and backtest preview."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["latest_spread_summary", "simple_spread_backtest"]


def latest_spread_summary(spread: pd.DataFrame) -> dict[str, float]:
    if spread.empty:
        return {}
    z = pd.to_numeric(spread.get("spread_z"), errors="coerce")
    values = pd.to_numeric(spread.get("spread"), errors="coerce")
    beta = pd.to_numeric(spread.get("hedge_beta"), errors="coerce")
    half_life = pd.to_numeric(spread.get("half_life"), errors="coerce")
    return {
        "rows": float(len(spread)),
        "latest_z": float(z.dropna().iloc[-1]) if z.notna().any() else np.nan,
        "spread_vol": float(values.diff().std(ddof=1)) if values.notna().sum() > 2 else np.nan,
        "beta": float(beta.dropna().iloc[-1]) if beta.notna().any() else np.nan,
        "half_life": float(half_life.dropna().iloc[-1]) if half_life.notna().any() else np.nan,
    }


def simple_spread_backtest(
    spread: pd.DataFrame,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    cost_bps: float = 0.0,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Run a compact mean-reversion preview on a spread series."""

    required = {"date", "spread", "spread_z"}
    missing = sorted(required - set(spread.columns))
    if missing:
        raise ValueError(f"Spread frame missing required columns: {missing}")

    work = spread[["date", "spread", "spread_z"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["spread"] = pd.to_numeric(work["spread"], errors="coerce")
    work["spread_z"] = pd.to_numeric(work["spread_z"], errors="coerce")
    work = work.dropna(subset=["date", "spread", "spread_z"]).reset_index(drop=True)
    if work.empty:
        return {"curve": work, "trades": pd.DataFrame(), "summary": {}}

    positions: list[int] = []
    position = 0
    for z_value in work["spread_z"]:
        z = float(z_value)
        if position == 0:
            if z >= entry_z:
                position = -1
            elif z <= -entry_z:
                position = 1
        elif position > 0:
            if z >= -exit_z or z <= -stop_z:
                position = 0
        elif position < 0:
            if z <= exit_z or z >= stop_z:
                position = 0
        positions.append(position)

    work["position"] = pd.Series(positions, dtype=float)
    work["spread_pnl"] = work["position"].shift(1).fillna(0.0) * work["spread"].diff().fillna(0.0)
    work["trade_size"] = work["position"].diff().abs().fillna(work["position"].abs())
    work["cost"] = work["trade_size"] * (float(cost_bps) / 10000.0)
    work["net_pnl"] = work["spread_pnl"] - work["cost"]
    work["equity"] = work["net_pnl"].cumsum()
    work["drawdown"] = work["equity"] - work["equity"].cummax()

    trades = _extract_trades(work)
    summary = {
        "observations": float(len(work)),
        "trades": float(len(trades)),
        "net_pnl": float(work["net_pnl"].sum()),
        "max_drawdown": float(work["drawdown"].min()) if len(work) else np.nan,
        "win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else np.nan,
        "avg_holding_days": float(trades["holding_days"].mean()) if not trades.empty else np.nan,
        "turnover_events": float(work["trade_size"].gt(0).sum()),
    }
    return {"curve": work, "trades": trades, "summary": summary}


def _extract_trades(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    entry_idx: int | None = None
    entry_position = 0.0
    for idx, row in curve.iterrows():
        position = float(row["position"])
        prev = float(curve.loc[idx - 1, "position"]) if idx > 0 else 0.0
        if prev == 0.0 and position != 0.0:
            entry_idx = int(idx)
            entry_position = position
        elif prev != 0.0 and position == 0.0 and entry_idx is not None:
            segment = curve.loc[entry_idx:idx]
            rows.append(
                {
                    "entry_date": curve.loc[entry_idx, "date"],
                    "exit_date": row["date"],
                    "side": "long_spread" if entry_position > 0 else "short_spread",
                    "holding_days": float(max(1, idx - entry_idx)),
                    "entry_z": float(curve.loc[entry_idx, "spread_z"]),
                    "exit_z": float(row["spread_z"]),
                    "pnl": float(segment["net_pnl"].sum()),
                }
            )
            entry_idx = None
            entry_position = 0.0
    return pd.DataFrame(rows)
