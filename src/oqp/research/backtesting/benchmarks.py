"""Benchmark return generators for research backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd


BENCHMARK_RETURN_COL = "benchmark_return"
BENCHMARK_ABSOLUTE = "ABSOLUTE"
BENCHMARK_BUY_AND_HOLD = "BUY_AND_HOLD"
BENCHMARK_RISK_FREE = "RISK_FREE"
BENCHMARK_INDEX = "INDEX"
BENCHMARK_SPY = "SPY"
BENCHMARK_CSI300 = "CSI300"
BENCHMARK_SECTOR_NEUTRAL = "SECTOR_NEUTRAL"
DEFAULT_SPY_TICKERS = ("SPY", "SPY.US", "SPY.N")
DEFAULT_CSI300_TICKERS = ("CSI300", "CSI_300", "000300.SH", "399300.SZ")
ACTIVE_WEIGHT_COLUMNS = (
    "target_weight",
    "final_target_weight",
    "signal",
    "weight",
    "factor_score",
)


class BaseBenchmark(ABC):
    """Interface for benchmark series generators."""

    @abstractmethod
    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with ``date`` and ``benchmark_return`` columns."""
        raise NotImplementedError


class AbsoluteReturnBenchmark(BaseBenchmark):
    """Flat daily return benchmark derived from an annual hurdle rate."""

    def __init__(self, ann_rate: float = 0.05):
        self.ann_rate = float(ann_rate)
        self.daily_rate = (1.0 + self.ann_rate) ** (1.0 / 252.0) - 1.0

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        _require_columns(strategy_df, {"date"}, "strategy_df")
        dates = (
            pd.to_datetime(strategy_df["date"], errors="coerce")
            .dt.normalize()
            .dropna()
            .unique()
        )
        out = pd.DataFrame({"date": pd.to_datetime(sorted(dates))})
        out[BENCHMARK_RETURN_COL] = self.daily_rate
        return out


class RiskFreeRateBenchmark(AbsoluteReturnBenchmark):
    """Flat daily benchmark for cash or risk-free-rate comparisons."""

    def __init__(self, ann_rate: float = 0.03):
        super().__init__(ann_rate=ann_rate)


class TickerBenchmark(BaseBenchmark):
    """Buy-and-hold benchmark for one or more named benchmark tickers."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        benchmark_tickers: str | Iterable[str],
        *,
        stale_limit: int = 3,
        strict: bool = True,
    ):
        self.raw_df = raw_df.copy()
        self.benchmark_tickers = _normalize_tickers(benchmark_tickers)
        self.stale_limit = int(stale_limit)
        self.strict = bool(strict)
        if not self.benchmark_tickers:
            raise ValueError("benchmark_tickers must contain at least one ticker.")

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        _require_columns(self.raw_df, {"date", "ticker", "close"}, "raw_df")
        raw = self.raw_df.copy()
        raw["ticker"] = raw["ticker"].astype(str)
        benchmark_raw = raw[raw["ticker"].isin(self.benchmark_tickers)].copy()
        if benchmark_raw.empty:
            if self.strict:
                raise ValueError(
                    "raw_df contains no rows for benchmark tickers: "
                    f"{list(self.benchmark_tickers)}"
                )
            return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

        benchmark = dynamic_equal_weight_benchmark(
            benchmark_raw,
            stale_limit=self.stale_limit,
        )
        dates = _strategy_dates(strategy_df)
        if not dates.empty:
            benchmark = benchmark[benchmark["date"].isin(dates)]
        return benchmark.reset_index(drop=True)


class SPYBenchmark(TickerBenchmark):
    """Convenience S&P 500 ETF benchmark using SPY-style ticker aliases."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_SPY_TICKERS,
        stale_limit: int = 3,
        strict: bool = True,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
        )


class CSI300Benchmark(TickerBenchmark):
    """Convenience CSI 300 index benchmark using common ticker aliases."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_CSI300_TICKERS,
        stale_limit: int = 3,
        strict: bool = True,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
        )


class BuyAndHoldBenchmark(BaseBenchmark):
    """
    Dynamic equal-weight benchmark over the assets the strategy actually traded.

    When no active strategy weights can be inferred, the benchmark falls back to
    the full raw universe. This preserves the legacy alpha-lab behavior while
    keeping the selection rule deterministic and testable.
    """

    def __init__(self, raw_df: pd.DataFrame, *, stale_limit: int = 3):
        self.raw_df = raw_df.copy()
        self.stale_limit = int(stale_limit)

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in self.raw_df.columns:
            return AbsoluteReturnBenchmark().generate(strategy_df)

        weight_col = self.select_executed_weight_col(strategy_df)
        active_pairs = self.build_daily_active_pairs(strategy_df, weight_col)
        raw = self.raw_df.copy()
        if "ticker" in raw.columns:
            raw["ticker"] = raw["ticker"].astype(str)

        active_matrix = None
        if not active_pairs.empty:
            raw = raw[raw["ticker"].isin(active_pairs["ticker"].unique())].copy()
            active_matrix = (
                active_pairs.assign(is_active=True)
                .pivot_table(
                    index="date",
                    columns="ticker",
                    values="is_active",
                    aggfunc="any",
                    fill_value=False,
                )
                .astype(bool)
            )

        return dynamic_equal_weight_benchmark(
            raw,
            active_matrix=active_matrix,
            stale_limit=self.stale_limit,
        )

    @staticmethod
    def select_executed_weight_col(strategy_df: pd.DataFrame) -> str | None:
        for column in ACTIVE_WEIGHT_COLUMNS:
            if column in strategy_df.columns:
                return column
        return None

    @staticmethod
    def build_daily_active_pairs(
        strategy_df: pd.DataFrame,
        weight_col: str | None,
    ) -> pd.DataFrame:
        if weight_col is None or not {"date", "ticker"}.issubset(
            strategy_df.columns
        ):
            return pd.DataFrame(columns=["date", "ticker"])

        audit_df = strategy_df[["date", "ticker", weight_col]].copy()
        audit_df["date"] = (
            pd.to_datetime(audit_df["date"], errors="coerce")
            .dt.normalize()
        )
        audit_df = audit_df.dropna(subset=["date", "ticker"])
        audit_df["ticker"] = audit_df["ticker"].astype(str)
        weights = pd.to_numeric(audit_df[weight_col], errors="coerce").fillna(0.0)
        active_mask = weights.abs() > 1e-12
        return audit_df.loc[active_mask, ["date", "ticker"]].drop_duplicates()


class SectorNeutralBenchmark(BaseBenchmark):
    """
    Equal-weight benchmark by active sector, then active ticker within sector.

    This removes the accidental bias where a sector with many traded symbols
    dominates the benchmark simply because it has more rows in the universe.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        sector_col: str = "sector",
        stale_limit: int = 3,
    ):
        self.raw_df = raw_df.copy()
        self.sector_col = sector_col
        self.stale_limit = int(stale_limit)

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "ticker", "close", self.sector_col}
        _require_columns(self.raw_df, required, "raw_df")

        raw = self.raw_df[["date", "ticker", "close", self.sector_col]].copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw = raw.dropna(subset=["date", "ticker", "close", self.sector_col])
        raw["ticker"] = raw["ticker"].astype(str)
        raw[self.sector_col] = raw[self.sector_col].astype(str)
        raw = raw[raw["close"] > 0]
        if raw.empty:
            return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

        weight_col = BuyAndHoldBenchmark.select_executed_weight_col(strategy_df)
        active_pairs = BuyAndHoldBenchmark.build_daily_active_pairs(strategy_df, weight_col)

        active_matrix = None
        if not active_pairs.empty:
            raw = raw[raw["ticker"].isin(active_pairs["ticker"].unique())].copy()
            active_matrix = (
                active_pairs.assign(is_active=True)
                .pivot_table(
                    index="date",
                    columns="ticker",
                    values="is_active",
                    aggfunc="any",
                    fill_value=False,
                )
                .astype(bool)
            )
            if raw.empty:
                return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

        returns = _daily_ticker_returns(raw, stale_limit=self.stale_limit)
        if active_matrix is not None:
            active = active_matrix.copy()
            active.index = pd.to_datetime(active.index, errors="coerce").normalize()
            active.columns = active.columns.astype(str)
            active = active.reindex(
                index=returns.index,
                columns=returns.columns,
                fill_value=False,
            )
            returns = returns.where(active)

        sector_map = (
            raw.sort_values("date")
            .drop_duplicates("ticker", keep="last")
            .set_index("ticker")[self.sector_col]
            .to_dict()
        )
        sector_returns = []
        for sector in sorted(set(sector_map.values())):
            tickers = [
                ticker
                for ticker, ticker_sector in sector_map.items()
                if ticker_sector == sector
            ]
            sector_frame = returns.reindex(columns=tickers)
            sector_returns.append(sector_frame.mean(axis=1).rename(sector))

        if not sector_returns:
            return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

        benchmark = pd.concat(sector_returns, axis=1).mean(axis=1).reset_index()
        benchmark.columns = ["date", BENCHMARK_RETURN_COL]
        return benchmark.dropna(subset=[BENCHMARK_RETURN_COL]).reset_index(drop=True)


class BenchmarkFactory:
    """Factory preserving the alpha-lab benchmark type strings."""

    @staticmethod
    def create_benchmark(
        b_type: str | None,
        raw_df: pd.DataFrame | None = None,
        **kwargs: object,
    ) -> BaseBenchmark:
        benchmark_type = _canonical_benchmark_type(b_type)
        ann_rate = float(kwargs.get("ann_rate", 0.05))
        risk_free_rate = float(
            kwargs.get("risk_free_rate", kwargs.get("ann_rate", 0.03))
        )
        stale_limit = int(kwargs.get("stale_limit", 3))
        strict = bool(kwargs.get("strict", True))
        if benchmark_type == BENCHMARK_ABSOLUTE:
            return AbsoluteReturnBenchmark(ann_rate=ann_rate)
        if benchmark_type == BENCHMARK_RISK_FREE:
            return RiskFreeRateBenchmark(ann_rate=risk_free_rate)
        if benchmark_type == BENCHMARK_BUY_AND_HOLD:
            return BuyAndHoldBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                stale_limit=stale_limit,
            )
        if benchmark_type == BENCHMARK_INDEX:
            benchmark_tickers = kwargs.get("benchmark_tickers", kwargs.get("tickers"))
            if benchmark_tickers is None:
                raise ValueError("benchmark_tickers required for INDEX benchmark.")
            return TickerBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers,  # type: ignore[arg-type]
                stale_limit=stale_limit,
                strict=strict,
            )
        if benchmark_type == BENCHMARK_SPY:
            return SPYBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_SPY_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
            )
        if benchmark_type == BENCHMARK_CSI300:
            return CSI300Benchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_CSI300_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
            )
        if benchmark_type == BENCHMARK_SECTOR_NEUTRAL:
            return SectorNeutralBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                sector_col=str(kwargs.get("sector_col", "sector")),
                stale_limit=stale_limit,
            )
        return AbsoluteReturnBenchmark(ann_rate=ann_rate)


def dynamic_equal_weight_benchmark(
    raw_df: pd.DataFrame,
    *,
    active_matrix: pd.DataFrame | None = None,
    stale_limit: int = 3,
) -> pd.DataFrame:
    """Build daily equal-weight returns from long-form OHLCV-style data."""

    _require_columns(raw_df, {"date", "ticker", "close"}, "raw_df")
    raw = raw_df[["date", "ticker", "close"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["date", "ticker", "close"])
    raw["ticker"] = raw["ticker"].astype(str)
    raw = raw[raw["close"] > 0]
    if raw.empty:
        return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

    returns = _daily_ticker_returns(raw, stale_limit=stale_limit)

    if active_matrix is not None:
        active = active_matrix.copy()
        active.index = pd.to_datetime(active.index, errors="coerce").normalize()
        active.columns = active.columns.astype(str)
        active = active.reindex(
            index=returns.index,
            columns=returns.columns,
            fill_value=False,
        )
        returns = returns.where(active)

    out = returns.mean(axis=1).reset_index()
    out.columns = ["date", BENCHMARK_RETURN_COL]
    return out.dropna(subset=[BENCHMARK_RETURN_COL]).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _require_raw_df(raw_df: pd.DataFrame | None, benchmark_type: str) -> pd.DataFrame:
    if raw_df is None:
        raise ValueError(f"raw_df required for {benchmark_type} benchmark.")
    return raw_df


def _canonical_benchmark_type(b_type: str | None) -> str:
    benchmark_type = str(b_type or BENCHMARK_ABSOLUTE).strip().upper().replace("-", "_")
    aliases = {
        "ABS": BENCHMARK_ABSOLUTE,
        "HURDLE": BENCHMARK_ABSOLUTE,
        "CASH": BENCHMARK_RISK_FREE,
        "RISKFREE": BENCHMARK_RISK_FREE,
        "RISK_FREE_RATE": BENCHMARK_RISK_FREE,
        "BUYHOLD": BENCHMARK_BUY_AND_HOLD,
        "BUY_HOLD": BENCHMARK_BUY_AND_HOLD,
        "BUY_AND_HOLD": BENCHMARK_BUY_AND_HOLD,
        "MARKET": BENCHMARK_INDEX,
        "TICKER": BENCHMARK_INDEX,
        "EXTERNAL_INDEX": BENCHMARK_INDEX,
        "S&P500": BENCHMARK_SPY,
        "SP500": BENCHMARK_SPY,
        "SNP500": BENCHMARK_SPY,
        "CSI_300": BENCHMARK_CSI300,
        "000300.SH": BENCHMARK_CSI300,
        "399300.SZ": BENCHMARK_CSI300,
        "SECTOR": BENCHMARK_SECTOR_NEUTRAL,
        "SECTOR_NEUTRAL": BENCHMARK_SECTOR_NEUTRAL,
    }
    return aliases.get(benchmark_type, benchmark_type)


def _normalize_tickers(tickers: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(tickers, str):
        tickers = (tickers,)
    normalized = []
    for ticker in tickers:
        if ticker is None:
            continue
        value = str(ticker).strip()
        if value:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def _strategy_dates(strategy_df: pd.DataFrame) -> pd.Series:
    _require_columns(strategy_df, {"date"}, "strategy_df")
    return pd.Series(
        pd.to_datetime(strategy_df["date"], errors="coerce")
        .dt.normalize()
        .dropna()
        .unique()
    )


def _daily_ticker_returns(raw: pd.DataFrame, *, stale_limit: int = 3) -> pd.DataFrame:
    daily = raw.groupby(["date", "ticker"], as_index=False)["close"].last()
    prices = (
        daily.pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill(limit=stale_limit)
    )
    return prices.pct_change(fill_method=None)


__all__ = [
    "ACTIVE_WEIGHT_COLUMNS",
    "BENCHMARK_ABSOLUTE",
    "BENCHMARK_BUY_AND_HOLD",
    "BENCHMARK_CSI300",
    "BENCHMARK_INDEX",
    "BENCHMARK_RISK_FREE",
    "BENCHMARK_RETURN_COL",
    "BENCHMARK_SECTOR_NEUTRAL",
    "BENCHMARK_SPY",
    "DEFAULT_CSI300_TICKERS",
    "DEFAULT_SPY_TICKERS",
    "AbsoluteReturnBenchmark",
    "BaseBenchmark",
    "BenchmarkFactory",
    "BuyAndHoldBenchmark",
    "CSI300Benchmark",
    "RiskFreeRateBenchmark",
    "SPYBenchmark",
    "SectorNeutralBenchmark",
    "TickerBenchmark",
    "dynamic_equal_weight_benchmark",
]
