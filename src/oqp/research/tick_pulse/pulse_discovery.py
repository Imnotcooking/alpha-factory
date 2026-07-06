from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.data import InstrumentMaster, TickUniverseSelector
from oqp.research.tick_pulse.features import (
    add_session_ids,
    clean_tick_data,
    infer_tick_size,
    load_ticks,
    product_prefix,
)


PULSE_CACHE_TYPE = "directionless_pulse_discovery_v1"
PULSE_REQUIRED_COLUMNS = [
    "symbol",
    "datetime",
    "last_price",
    "volume",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "oi",
]
PULSE_ZONE_BANDS = [
    ("normal", 0.0, 0.90),
    ("active", 0.90, 0.95),
    ("watch", 0.95, 0.99),
    ("pulse", 0.99, 0.995),
    ("extreme", 0.995, 1.0),
]


def prepare_pulse_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_tick_data(TickUniverseSelector.normalize_schema(df))
    out = add_session_ids(out)
    out["symbol"] = out["symbol"].astype(str)
    out["product"] = out["symbol"].map(product_prefix)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "last_price"]).sort_values(["symbol", "datetime"])

    tick_size_rows = {
        symbol: _resolve_tick_size(symbol, group)
        for symbol, group in out.groupby("symbol", sort=False)
    }
    out["tick_size_est"] = out["symbol"].map(
        lambda symbol: tick_size_rows.get(symbol, {}).get("tick_size")
    ).replace(0, np.nan)
    out["tick_size_inferred"] = out["symbol"].map(
        lambda symbol: tick_size_rows.get(symbol, {}).get("inferred_tick_size")
    )
    out["tick_size_source"] = out["symbol"].map(
        lambda symbol: tick_size_rows.get(symbol, {}).get("source", "unknown")
    )
    out["mid_price"] = (out["bid_price_1"] + out["ask_price_1"]) / 2.0
    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    depth = out["bid_volume_1"] + out["ask_volume_1"]
    out["book_imbalance"] = np.where(
        depth > 0,
        (out["bid_volume_1"] - out["ask_volume_1"]) / depth,
        0.0,
    )
    grouped = out.groupby(["symbol", "_session_id"], sort=False)
    out["volume_delta_raw"] = grouped["volume"].diff().fillna(0.0)
    out["volume_delta"] = out["volume_delta_raw"].clip(lower=0.0)
    out["last_price_delta"] = grouped["last_price"].diff().fillna(0.0)

    prev_bid = grouped["bid_price_1"].shift(1)
    prev_ask = grouped["ask_price_1"].shift(1)
    tick_rule = np.sign(out["last_price_delta"]).replace(0, np.nan)
    tick_rule = tick_rule.groupby(out["symbol"], sort=False).ffill().fillna(0.0)
    out["trade_sign"] = np.select(
        [
            (out["volume_delta"] > 0) & (out["last_price"] >= prev_ask),
            (out["volume_delta"] > 0) & (out["last_price"] <= prev_bid),
            out["volume_delta"] > 0,
        ],
        [1.0, -1.0, tick_rule],
        default=0.0,
    )
    out["signed_volume"] = out["trade_sign"] * out["volume_delta"]
    return out.reset_index(drop=True)


def _resolve_tick_size(symbol: str, price_frame: pd.DataFrame) -> dict[str, float | str]:
    inferred = infer_tick_size(price_frame)
    try:
        profile = InstrumentMaster("FUTURES_CN").get_profile(symbol)
        master_tick = float(profile.tick_size)
        if profile.exchange != "UNKNOWN" and np.isfinite(master_tick) and master_tick > 0:
            source = "instrument_master"
            if np.isfinite(inferred) and abs(float(inferred) - master_tick) > max(1e-8, master_tick * 1e-4):
                source = "instrument_master_inferred_mismatch"
            return {
                "tick_size": master_tick,
                "inferred_tick_size": float(inferred) if np.isfinite(inferred) else np.nan,
                "source": source,
            }
    except Exception:
        pass

    return {
        "tick_size": float(inferred) if np.isfinite(inferred) and inferred > 0 else np.nan,
        "inferred_tick_size": float(inferred) if np.isfinite(inferred) else np.nan,
        "source": "inferred_from_prices",
    }


def build_directionless_pulse_frame(
    df: pd.DataFrame,
    *,
    window_seconds: float = 10.0,
) -> pd.DataFrame:
    out = prepare_pulse_frame(df)
    window_seconds = max(float(window_seconds), 0.001)

    n = len(out)
    pulse_start_pos = np.zeros(n, dtype=np.int64)
    pulse_window_seconds = np.zeros(n, dtype=np.float64)
    pulse_window_snapshots = np.ones(n, dtype=np.int64)
    pulse_start_price = np.full(n, np.nan, dtype=np.float64)
    pulse_net_move_ticks = np.full(n, np.nan, dtype=np.float64)
    pulse_abs_move_ticks = np.full(n, np.nan, dtype=np.float64)
    pulse_velocity_ticks_per_sec = np.full(n, np.nan, dtype=np.float64)
    pulse_volume_delta = np.full(n, np.nan, dtype=np.float64)
    pulse_spread_mean = np.full(n, np.nan, dtype=np.float64)
    pulse_book_imbalance_mean = np.full(n, np.nan, dtype=np.float64)
    pulse_flow_imbalance = np.full(n, np.nan, dtype=np.float64)
    pulse_path_range_ticks = np.full(n, np.nan, dtype=np.float64)

    for _, group in out.groupby(["symbol", "_session_id"], sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        local_n = len(group)
        if local_n == 0:
            continue

        times = group["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        window_ns = int(window_seconds * 1_000_000_000)
        start_local = np.searchsorted(times, times - window_ns, side="left")
        current_local = np.arange(local_n)

        prices = group["last_price"].to_numpy(dtype=np.float64)
        ticks = group["tick_size_est"].to_numpy(dtype=np.float64)
        safe_ticks = np.where(np.isfinite(ticks) & (ticks > 0), ticks, np.nan)

        start_prices = prices[start_local]
        elapsed_seconds = (times - times[start_local]) / 1_000_000_000.0
        snapshots = current_local - start_local + 1
        net_ticks = (prices - start_prices) / safe_ticks
        abs_ticks = np.abs(net_ticks)
        velocity = np.divide(
            net_ticks,
            elapsed_seconds,
            out=np.zeros_like(net_ticks, dtype=np.float64),
            where=elapsed_seconds > 0,
        )

        volume_sum = _window_sum(group["volume_delta"].to_numpy(dtype=np.float64), start_local)
        spread_mean = _window_mean(group["spread"].to_numpy(dtype=np.float64), start_local)
        book_mean = _window_mean(group["book_imbalance"].to_numpy(dtype=np.float64), start_local)
        signed_sum = _window_sum(group["signed_volume"].to_numpy(dtype=np.float64), start_local)
        flow = np.divide(
            signed_sum,
            volume_sum,
            out=np.zeros_like(signed_sum, dtype=np.float64),
            where=volume_sum > 0,
        )
        min_price, max_price = _window_min_max(prices, start_local)
        path_range = (max_price - min_price) / safe_ticks

        pulse_start_pos[idx] = idx[start_local]
        pulse_window_seconds[idx] = elapsed_seconds
        pulse_window_snapshots[idx] = snapshots
        pulse_start_price[idx] = start_prices
        pulse_net_move_ticks[idx] = net_ticks
        pulse_abs_move_ticks[idx] = abs_ticks
        pulse_velocity_ticks_per_sec[idx] = velocity
        pulse_volume_delta[idx] = volume_sum
        pulse_spread_mean[idx] = spread_mean
        pulse_book_imbalance_mean[idx] = book_mean
        pulse_flow_imbalance[idx] = flow
        pulse_path_range_ticks[idx] = path_range

    out["pulse_start_pos"] = pulse_start_pos
    out["pulse_start_time"] = out["datetime"].iloc[pulse_start_pos].to_numpy()
    out["pulse_start_price"] = pulse_start_price
    out["pulse_window_seconds"] = pulse_window_seconds
    out["pulse_window_snapshots"] = pulse_window_snapshots
    out["pulse_net_move_ticks"] = pulse_net_move_ticks
    out["pulse_abs_move_ticks"] = pulse_abs_move_ticks
    out["pulse_direction"] = np.select(
        [pulse_net_move_ticks > 0, pulse_net_move_ticks < 0],
        ["Up", "Down"],
        default="Flat",
    )
    out["pulse_velocity_ticks_per_sec"] = pulse_velocity_ticks_per_sec
    out["pulse_volume_delta"] = pulse_volume_delta
    out["pulse_spread_mean"] = pulse_spread_mean
    out["pulse_book_imbalance_mean"] = pulse_book_imbalance_mean
    out["pulse_flow_imbalance"] = pulse_flow_imbalance
    out["pulse_path_range_ticks"] = pulse_path_range_ticks
    out.attrs["pulse_window_seconds_config"] = window_seconds
    out.attrs["tick_pulse_backend"] = "python_numpy"
    return out


def detect_directionless_pulse_events(
    pulse_frame: pd.DataFrame,
    *,
    percentile: float = 0.99,
    collapse_gap_seconds: float | None = None,
) -> pd.DataFrame:
    if pulse_frame.empty:
        return _empty_events()

    percentile = float(np.clip(percentile, 0.0, 1.0))
    window_seconds = float(pulse_frame.attrs.get("pulse_window_seconds_config", 10.0))
    collapse_gap_seconds = float(collapse_gap_seconds if collapse_gap_seconds is not None else window_seconds)

    values = pd.to_numeric(pulse_frame["pulse_abs_move_ticks"], errors="coerce")
    threshold = float(values.dropna().quantile(percentile)) if values.notna().any() else np.nan
    if not np.isfinite(threshold):
        return _empty_events(threshold)

    flagged = pulse_frame.loc[values >= threshold].copy()
    flagged = flagged.loc[pd.to_numeric(flagged["pulse_window_seconds"], errors="coerce") > 0]
    if flagged.empty:
        return _empty_events(threshold)

    flagged = flagged.sort_values(["symbol", "_session_id", "datetime"]).copy()
    cluster_ids = []
    cluster_id = -1
    previous_symbol = None
    previous_session = None
    previous_time = None
    for _, row in flagged.iterrows():
        current_symbol = row["symbol"]
        current_session = row["_session_id"]
        current_time = row["datetime"]
        starts_new = (
            current_symbol != previous_symbol
            or current_session != previous_session
            or previous_time is None
            or (current_time - previous_time).total_seconds() > collapse_gap_seconds
        )
        if starts_new:
            cluster_id += 1
        cluster_ids.append(cluster_id)
        previous_symbol = current_symbol
        previous_session = current_session
        previous_time = current_time

    flagged["event_cluster_id"] = cluster_ids
    flagged["event_threshold_ticks"] = threshold
    cluster_stats = flagged.groupby("event_cluster_id", sort=False).agg(
        event_cluster_start=("datetime", "min"),
        event_cluster_end=("datetime", "max"),
        event_cluster_size=("datetime", "size"),
    )
    keep_idx = (
        flagged.sort_values(["event_cluster_id", "pulse_abs_move_ticks", "datetime"], ascending=[True, False, True])
        .groupby("event_cluster_id", sort=False)
        .head(1)
        .index
    )
    events = flagged.loc[keep_idx].sort_values("pulse_abs_move_ticks", ascending=False).copy()
    events = events.join(cluster_stats, on="event_cluster_id")
    events["event_rank"] = np.arange(1, len(events) + 1)
    events["event_time"] = events["datetime"]
    events["event_percentile"] = percentile
    events.attrs["event_threshold_ticks"] = threshold
    events.attrs["pulse_window_seconds_config"] = window_seconds
    events.attrs["collapse_gap_seconds"] = collapse_gap_seconds
    return events.reset_index(drop=False).rename(columns={"index": "source_index"})


def build_pulse_discovery(
    df: pd.DataFrame,
    *,
    window_seconds: float = 10.0,
    percentile: float = 0.99,
    collapse_gap_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_directionless_pulse_frame(df, window_seconds=window_seconds)
    events = detect_directionless_pulse_events(
        frame,
        percentile=percentile,
        collapse_gap_seconds=collapse_gap_seconds,
    )
    return frame, events


def build_pulse_zone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(frame["pulse_abs_move_ticks"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]
    if values.empty:
        return pd.DataFrame(
            columns=[
                "zone_code",
                "percentile_start",
                "percentile_end",
                "windows",
                "share",
                "avg_net_distance_ticks",
                "max_net_distance_ticks",
            ]
        )

    sorted_values = values.sort_values(kind="mergesort").reset_index(drop=True)
    n = len(sorted_values)
    rows = []
    for zone_code, lo, hi in PULSE_ZONE_BANDS:
        start = int(np.floor(lo * n))
        end = int(np.floor(hi * n)) if hi < 1.0 else n
        end = max(end, start + 1) if hi > lo else start
        scoped = sorted_values.iloc[start:end]
        rows.append(
            {
                "zone_code": zone_code,
                "percentile_start": lo,
                "percentile_end": hi,
                "windows": int(len(scoped)),
                "share": float(len(scoped) / n) if n else 0.0,
                "avg_net_distance_ticks": float(scoped.mean()) if not scoped.empty else np.nan,
                "max_net_distance_ticks": float(scoped.max()) if not scoped.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_pulse_file(
    path: str | Path,
    *,
    product: str = "",
    window_seconds: float = 10.0,
    percentile: float = 0.99,
    top_n: int = 20,
) -> dict:
    raw = load_ticks(str(path))
    selector = TickUniverseSelector(product=product)
    scoped = selector.select(raw)
    frame, events = build_pulse_discovery(
        scoped,
        window_seconds=window_seconds,
        percentile=percentile,
    )

    valid_abs = pd.to_numeric(frame["pulse_abs_move_ticks"], errors="coerce").dropna()
    span_hours = _trading_hours(frame)
    top_values = valid_abs.nlargest(max(int(top_n), 1))
    return {
        "asset": scoped.attrs.get("selected_product", product or _product_hint(path)),
        "main_contract": scoped.attrs.get("selected_symbol", ""),
        "source_file": str(path),
        "rows": int(len(frame)),
        "trading_hours": float(span_hours),
        "p95_move_ticks": float(valid_abs.quantile(0.95)) if not valid_abs.empty else np.nan,
        "p99_move_ticks": float(valid_abs.quantile(0.99)) if not valid_abs.empty else np.nan,
        "p995_move_ticks": float(valid_abs.quantile(0.995)) if not valid_abs.empty else np.nan,
        "pulse_threshold_ticks": float(events.attrs.get("event_threshold_ticks", np.nan)),
        "pulse_events": int(len(events)),
        "pulses_per_trading_hour": float(len(events) / span_hours) if span_hours > 0 else np.nan,
        "avg_top20_pulse_ticks": float(top_values.mean()) if not top_values.empty else np.nan,
        "median_snapshots_per_window": float(frame["pulse_window_snapshots"].median()) if not frame.empty else np.nan,
    }


def summarize_pulse_zones_file(
    path: str | Path,
    *,
    product: str = "",
    window_seconds: float = 10.0,
) -> pd.DataFrame:
    raw = load_ticks(str(path))
    selector = TickUniverseSelector(product=product)
    scoped = selector.select(raw)
    frame = build_directionless_pulse_frame(scoped, window_seconds=window_seconds)
    zones = build_pulse_zone_summary(frame)
    if zones.empty:
        return zones

    zones.insert(0, "source_file", str(path))
    zones.insert(0, "main_contract", scoped.attrs.get("selected_symbol", ""))
    zones.insert(0, "asset", scoped.attrs.get("selected_product", product or _product_hint(path)))
    zones["rows"] = int(len(frame))
    zones["trading_hours"] = _trading_hours(frame)
    return zones


def rank_cross_asset_pulses(
    paths: list[str | Path],
    *,
    product_by_path: dict[str, str] | None = None,
    window_seconds: float = 10.0,
    percentile: float = 0.99,
    top_n: int = 20,
) -> pd.DataFrame:
    rows = []
    product_by_path = product_by_path or {}
    for path in paths:
        path_text = str(path)
        product = product_by_path.get(path_text, _product_hint(path_text))
        rows.append(
            summarize_pulse_file(
                path_text,
                product=product,
                window_seconds=window_seconds,
                percentile=percentile,
                top_n=top_n,
            )
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["pulses_per_trading_hour", "p99_move_ticks", "avg_top20_pulse_ticks"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _window_sum(values: np.ndarray, start_local: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(clean)])
    end = np.arange(len(clean)) + 1
    return cumulative[end] - cumulative[start_local]


def _window_mean(values: np.ndarray, start_local: np.ndarray) -> np.ndarray:
    clean = values.astype(np.float64)
    valid = np.isfinite(clean).astype(np.float64)
    sums = _window_sum(np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0), start_local)
    counts = _window_sum(valid, start_local)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _window_min_max(values: np.ndarray, start_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_deque: deque[int] = deque()
    max_deque: deque[int] = deque()
    mins = np.full(len(values), np.nan, dtype=np.float64)
    maxs = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        while min_deque and min_deque[0] < start_local[i]:
            min_deque.popleft()
        while max_deque and max_deque[0] < start_local[i]:
            max_deque.popleft()
        while min_deque and values[min_deque[-1]] >= value:
            min_deque.pop()
        while max_deque and values[max_deque[-1]] <= value:
            max_deque.pop()
        min_deque.append(i)
        max_deque.append(i)
        mins[i] = values[min_deque[0]]
        maxs[i] = values[max_deque[0]]
    return mins, maxs


def _empty_events(threshold: float = np.nan) -> pd.DataFrame:
    columns = [
        "source_index",
        "event_rank",
        "event_time",
        "symbol",
        "pulse_direction",
        "pulse_net_move_ticks",
        "pulse_abs_move_ticks",
        "pulse_velocity_ticks_per_sec",
        "pulse_volume_delta",
        "pulse_spread_mean",
        "pulse_book_imbalance_mean",
        "pulse_flow_imbalance",
        "pulse_path_range_ticks",
        "event_cluster_id",
        "event_cluster_size",
        "event_threshold_ticks",
    ]
    out = pd.DataFrame(columns=columns)
    out.attrs["event_threshold_ticks"] = threshold
    return out


def _trading_hours(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    spans = (
        frame.groupby(["symbol", "_session_id"], sort=False)["datetime"]
        .agg(["min", "max"])
        .assign(seconds=lambda x: (x["max"] - x["min"]).dt.total_seconds())
    )
    return float(spans["seconds"].clip(lower=0).sum() / 3600.0)


def _product_hint(path: str | Path) -> str:
    name = os.path.basename(str(path))
    match = re.match(r"^\d+contract_([A-Za-z]+)_raw_", name)
    return match.group(1) if match else ""
