from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_TICK_FILE = ""
REQUIRED_COLUMNS = {
    "symbol",
    "datetime",
    "last_price",
    "volume",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "oi",
}
PRICE_COLUMNS = ("last_price", "bid_price_1", "ask_price_1")
SESSION_GAP = pd.Timedelta(minutes=30)

__all__ = [
    "DEFAULT_TICK_FILE",
    "PRICE_COLUMNS",
    "PulseThresholds",
    "REQUIRED_COLUMNS",
    "SESSION_GAP",
    "add_session_ids",
    "build_pulse_features",
    "clean_tick_data",
    "contract_summary",
    "filter_scope",
    "flag_pulses",
    "infer_tick_size",
    "load_tick_scope",
    "load_ticks",
    "product_prefix",
    "valid_tick_mask",
]


@dataclass(frozen=True)
class PulseThresholds:
    pulse_score: float = 0.68
    min_flow_imbalance: float = 0.35
    min_volume_intensity: float = 3.0
    min_price_shock: float = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Level-1 tick data and build first-pass market imbalance "
            "/ pulse diagnostics."
        )
    )
    parser.add_argument("--file", default=DEFAULT_TICK_FILE, help="Input parquet tick file.")
    parser.add_argument("--product", default="au", help="Product prefix to inspect, e.g. au.")
    parser.add_argument("--symbol", default="", help="Optional exact contract symbol, e.g. au2608.")
    parser.add_argument("--window", type=int, default=120, help="Rolling tick window for pulse features.")
    parser.add_argument("--top", type=int, default=20, help="Number of top pulse candidates to print.")
    parser.add_argument("--pulse-score", type=float, default=0.68, help="Minimum pulse score.")
    parser.add_argument("--save-csv", default="", help="Optional path to save pulse candidate CSV.")
    return parser.parse_args()


def load_ticks(
    path: str,
    *,
    filters: Any | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tick file not found: {path}")

    read_kwargs = {}
    if filters is not None:
        read_kwargs["filters"] = filters
    if columns is not None:
        read_kwargs["columns"] = columns
    try:
        df = pd.read_parquet(path, **read_kwargs)
    except (TypeError, ValueError):
        read_kwargs.pop("filters", None)
        df = pd.read_parquet(path, **read_kwargs)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Tick file missing required columns: {sorted(missing)}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_tick_scope(path: str, product: str = "", symbol: str = "") -> pd.DataFrame:
    if symbol:
        try:
            scoped = load_ticks(path, filters=[("symbol", "==", symbol)])
            scoped = scoped[scoped["symbol"].str.lower() == symbol.lower()].copy()
            if not scoped.empty:
                return scoped
        except Exception:
            pass

    df = load_ticks(path)
    return filter_scope(df, product=product, symbol=symbol)


def valid_tick_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column in PRICE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask &= values.gt(0)

    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    bid_size = pd.to_numeric(df["bid_volume_1"], errors="coerce")
    ask_size = pd.to_numeric(df["ask_volume_1"], errors="coerce")
    mask &= ask.ge(bid)
    mask &= bid_size.ge(0) & ask_size.ge(0)
    return mask.fillna(False)


def clean_tick_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[valid_tick_mask(df)].copy()


def add_session_ids(df: pd.DataFrame, session_gap: pd.Timedelta = SESSION_GAP) -> pd.DataFrame:
    out = df.sort_values(["symbol", "datetime"]).copy()
    grouped = out.groupby("symbol", sort=False)
    time_gap = grouped["datetime"].diff()
    volume_reset = grouped["volume"].diff().lt(0).fillna(False)
    symbol_start = out["symbol"].ne(out["symbol"].shift(1))

    session_break = symbol_start | time_gap.gt(session_gap).fillna(False) | volume_reset
    out["_session_break"] = session_break
    out["_session_gap_seconds"] = time_gap.dt.total_seconds().fillna(0.0)
    out["_session_id"] = session_break.groupby(out["symbol"], sort=False).cumsum().astype(int) - 1
    return out


def product_prefix(symbol: str) -> str:
    match = re.match(r"^([A-Za-z]+)", str(symbol))
    return match.group(1) if match else ""


def infer_tick_size(price_frame: pd.DataFrame) -> float:
    prices = pd.concat(
        [
            price_frame["last_price"],
            price_frame["bid_price_1"],
            price_frame["ask_price_1"],
        ],
        ignore_index=True,
    )
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices > 0]
    if prices.empty:
        return np.nan

    unique_prices = np.unique(np.round(prices.to_numpy(dtype=float), 6))
    diffs = np.diff(np.sort(unique_prices))
    diffs = diffs[diffs > 1e-8]
    if len(diffs) == 0:
        return np.nan

    # Use a low percentile rather than the absolute minimum to avoid rare quote dust.
    return float(np.round(np.quantile(diffs, 0.05), 6))


def contract_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in df.groupby("symbol", sort=True):
        group = group.sort_values("datetime")
        valid_mask = valid_tick_mask(group)
        valid_group = group.loc[valid_mask].copy()
        if valid_group.empty:
            valid_group = group.copy()
        volume_delta = valid_group["volume"].diff().fillna(0.0)
        spread = valid_group["ask_price_1"] - valid_group["bid_price_1"]
        rows.append(
            {
                "symbol": symbol,
                "product": product_prefix(symbol),
                "rows": len(group),
                "valid_rows": int(valid_mask.sum()),
                "invalid_rows": int((~valid_mask).sum()),
                "invalid_pct": float((~valid_mask).mean()),
                "start": group["datetime"].min(),
                "end": group["datetime"].max(),
                "first_price": valid_group["last_price"].iloc[0],
                "last_price": valid_group["last_price"].iloc[-1],
                "low": valid_group["last_price"].min(),
                "high": valid_group["last_price"].max(),
                "first_volume": group["volume"].iloc[0],
                "last_volume": group["volume"].iloc[-1],
                "positive_volume_delta": volume_delta.clip(lower=0).sum(),
                "negative_volume_resets": int((volume_delta < 0).sum()),
                "oi_first": valid_group["oi"].iloc[0],
                "oi_last": valid_group["oi"].iloc[-1],
                "tick_size_est": infer_tick_size(valid_group),
                "median_spread": spread.median(),
                "max_spread": spread.max(),
            }
        )
    return pd.DataFrame(rows)


def _rolling_by_symbol(
    df: pd.DataFrame,
    column: str,
    window: int,
    func: str,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = min_periods or max(10, window // 5)
    group_keys = ["symbol", "_session_id"] if "_session_id" in df.columns else ["symbol"]
    grouped = df.groupby(group_keys, sort=False)[column]
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=min_periods), func)())


def build_pulse_features(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    out = clean_tick_data(df)
    out = add_session_ids(out)
    out["product"] = out["symbol"].map(product_prefix)

    tick_size_map = {
        symbol: infer_tick_size(group)
        for symbol, group in out.groupby("symbol", sort=False)
    }
    out["tick_size_est"] = out["symbol"].map(tick_size_map).replace(0, np.nan)

    out["mid_price"] = (out["bid_price_1"] + out["ask_price_1"]) / 2.0
    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    book_depth = out["bid_volume_1"] + out["ask_volume_1"]
    out["book_imbalance"] = np.where(
        book_depth > 0,
        (out["bid_volume_1"] - out["ask_volume_1"]) / book_depth,
        0.0,
    )

    grouped = out.groupby(["symbol", "_session_id"], sort=False)
    out["volume_delta_raw"] = grouped["volume"].diff().fillna(0.0)
    out["volume_delta"] = out["volume_delta_raw"].clip(lower=0.0)
    out["oi_delta"] = grouped["oi"].diff().fillna(0.0)
    out["last_price_delta"] = grouped["last_price"].diff().fillna(0.0)
    out["mid_price_delta"] = grouped["mid_price"].diff().fillna(0.0)
    out["mid_move_ticks"] = out["mid_price_delta"] / out["tick_size_est"]

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

    out["rolling_signed_volume"] = _rolling_by_symbol(out, "signed_volume", window, "sum")
    out["rolling_total_volume"] = _rolling_by_symbol(out, "volume_delta", window, "sum")
    out["flow_imbalance"] = np.where(
        out["rolling_total_volume"] > 0,
        out["rolling_signed_volume"] / out["rolling_total_volume"],
        0.0,
    )

    positive_delta = out["volume_delta"].where(out["volume_delta"] > 0)
    out["_positive_volume_delta"] = positive_delta
    out["rolling_pos_volume_median"] = _rolling_by_symbol(
        out,
        "_positive_volume_delta",
        window,
        "median",
    )
    fallback_median = (
        out.groupby("symbol")["volume_delta"]
        .transform(lambda s: s[s > 0].median())
        .replace(0, np.nan)
    )
    out["rolling_pos_volume_median"] = (
        out["rolling_pos_volume_median"]
        .fillna(fallback_median)
        .fillna(1.0)
        .clip(lower=1.0)
    )
    out["volume_intensity"] = out["volume_delta"] / out["rolling_pos_volume_median"]

    out["rolling_mid_move_ticks"] = _rolling_by_symbol(out, "mid_move_ticks", window, "sum")
    out["_abs_mid_move_ticks"] = out["mid_move_ticks"].abs()
    out["rolling_abs_tick_median"] = _rolling_by_symbol(
        out,
        "_abs_mid_move_ticks",
        window,
        "median",
    ).fillna(0.0)
    out["price_shock"] = out["rolling_mid_move_ticks"].abs() / (out["rolling_abs_tick_median"] + 1.0)

    flow_strength = out["flow_imbalance"].abs().fillna(0.0).clip(0.0, 1.0)
    book_strength = out["book_imbalance"].abs().fillna(0.0).clip(0.0, 1.0)
    volume_strength = (np.log1p(out["volume_intensity"].clip(lower=0.0)) / np.log1p(10.0)).clip(0.0, 1.0)
    price_strength = (out["price_shock"].fillna(0.0) / 3.0).clip(0.0, 1.0)

    out["pulse_score"] = (
        0.40 * flow_strength
        + 0.25 * book_strength
        + 0.20 * volume_strength
        + 0.15 * price_strength
    )
    direction_seed = out["flow_imbalance"].where(out["flow_imbalance"].abs() >= 0.10, out["book_imbalance"])
    out["pulse_direction"] = np.sign(direction_seed).map({1.0: "buy", -1.0: "sell", 0.0: "neutral"})

    flow_dir = np.sign(out["flow_imbalance"])
    book_dir = np.sign(out["book_imbalance"])
    price_dir = np.sign(out["rolling_mid_move_ticks"])
    out["flow_price_aligned"] = (flow_dir * price_dir) > 0
    out["book_flow_aligned"] = (book_dir * flow_dir) > 0
    out["pulse_type"] = np.select(
        [
            out["flow_price_aligned"] & out["book_flow_aligned"],
            out["flow_price_aligned"] & ~out["book_flow_aligned"],
            ~out["flow_price_aligned"] & out["book_flow_aligned"],
        ],
        [
            "momentum",
            "absorption",
            "queue_pressure",
        ],
        default="mixed",
    )

    out = out.drop(columns=["_positive_volume_delta", "_abs_mid_move_ticks"])
    return out


def flag_pulses(features: pd.DataFrame, thresholds: PulseThresholds) -> pd.DataFrame:
    is_candidate = (
        (features["pulse_score"] >= thresholds.pulse_score)
        & (features["flow_imbalance"].abs() >= thresholds.min_flow_imbalance)
        & (
            (features["volume_intensity"] >= thresholds.min_volume_intensity)
            | (features["price_shock"] >= thresholds.min_price_shock)
        )
    )
    return features.loc[is_candidate].copy()


def filter_scope(df: pd.DataFrame, product: str = "", symbol: str = "") -> pd.DataFrame:
    scoped = df.copy()
    if product:
        scoped = scoped[scoped["symbol"].map(product_prefix).str.lower() == product.lower()]
    if symbol:
        scoped = scoped[scoped["symbol"].str.lower() == symbol.lower()]
    return scoped


def print_schema(df: pd.DataFrame):
    print("\n=== SCHEMA ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    for i, column in enumerate(df.columns):
        print(f"{i:>2}. {column:<16} {df[column].dtype}")


def print_definition():
    print("\n=== FIRST-PASS 脉冲 / PULSE DEFINITION ===")
    print(
        "A tick-level pulse is a short-lived microstructure burst where trade flow, "
        "top-of-book imbalance, and price response align."
    )
    print("Prototype components:")
    print("1. Trade-flow pressure: signed volume over a rolling tick window.")
    print("2. Queue pressure: (bid_volume_1 - ask_volume_1) / (bid_volume_1 + ask_volume_1).")
    print("3. Volume burst: current volume_delta versus rolling positive-volume median.")
    print("4. Price response: rolling mid-price move measured in inferred ticks.")
    print(
        "Candidate pulse = high combined pulse_score plus meaningful flow imbalance, "
        "and either abnormal volume or abnormal price response."
    )
    print("Pulse types:")
    print("- momentum: flow, visible queue, and price response point the same way.")
    print("- absorption: aggressive flow and price move align, but visible queue leans the other way.")
    print("- queue_pressure: visible queue and flow align, but price has not yet followed.")
    print("- mixed: strong burst, but signs are not clean enough for a subtype.")


def print_contracts(summary: pd.DataFrame):
    print("\n=== CONTRACTS ===")
    products = ", ".join(sorted(summary["product"].dropna().unique()))
    print(f"Detected products: {products}")
    if not summary.empty:
        liquid = summary.sort_values("positive_volume_delta", ascending=False).iloc[0]
        print(
            "Most liquid by positive volume delta: "
            f"{liquid['symbol']} ({liquid['positive_volume_delta']:,.0f} contracts)"
        )
    cols = [
        "symbol",
        "product",
        "rows",
        "start",
        "end",
        "last_volume",
        "positive_volume_delta",
        "oi_first",
        "oi_last",
        "tick_size_est",
        "median_spread",
        "max_spread",
    ]
    print(summary[cols].to_string(index=False))


def print_top_pulses(candidates: pd.DataFrame, top: int):
    print("\n=== TOP PULSE CANDIDATES ===")
    if candidates.empty:
        print("No pulse candidates under current thresholds.")
        return

    cols = [
        "symbol",
        "datetime",
        "pulse_type",
        "pulse_direction",
        "pulse_score",
        "last_price",
        "volume_delta",
        "volume_intensity",
        "flow_imbalance",
        "book_imbalance",
        "price_shock",
        "spread",
        "oi_delta",
    ]
    ranked = candidates.sort_values("pulse_score", ascending=False).head(top)
    print(ranked[cols].to_string(index=False))


def print_pulse_summary(candidates: pd.DataFrame):
    print("\n=== PULSE SUMMARY BY CONTRACT ===")
    if candidates.empty:
        print("No pulse candidates under current thresholds.")
        return
    summary = (
        candidates.groupby(["symbol", "pulse_type", "pulse_direction"])
        .agg(
            pulse_count=("pulse_score", "size"),
            max_score=("pulse_score", "max"),
            mean_score=("pulse_score", "mean"),
            max_volume_intensity=("volume_intensity", "max"),
            max_price_shock=("price_shock", "max"),
        )
        .reset_index()
        .sort_values(["symbol", "pulse_direction"])
    )
    print(summary.to_string(index=False))


def main():
    args = parse_args()
    thresholds = PulseThresholds(pulse_score=args.pulse_score)

    df = load_ticks(args.file)
    print_schema(df)

    summary = contract_summary(df)
    print_contracts(summary)
    print_definition()

    scoped = filter_scope(df, product=args.product, symbol=args.symbol)
    if scoped.empty:
        raise ValueError(f"No rows matched product={args.product!r}, symbol={args.symbol!r}.")

    print("\n=== ANALYSIS SCOPE ===")
    print(f"Product: {args.product or 'ALL'}")
    print(f"Symbol:  {args.symbol or 'ALL'}")
    print(f"Rows:    {len(scoped):,}")
    print(f"Symbols: {', '.join(sorted(scoped['symbol'].unique()))}")

    features = build_pulse_features(scoped, window=args.window)
    candidates = flag_pulses(features, thresholds)
    print_pulse_summary(candidates)
    print_top_pulses(candidates, args.top)

    if args.save_csv:
        output_dir = os.path.dirname(args.save_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        candidates.to_csv(args.save_csv, index=False)
        print(f"\nSaved pulse candidates -> {args.save_csv}")


if __name__ == "__main__":
    main()
