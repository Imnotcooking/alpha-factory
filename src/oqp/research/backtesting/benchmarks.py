"""Benchmark return generators for research backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import pandas as pd

from oqp.data.views import build_market_data_views


BENCHMARK_RETURN_COL = "benchmark_return"
BENCHMARK_ABSOLUTE = "ABSOLUTE"
BENCHMARK_BUY_AND_HOLD = "BUY_AND_HOLD"
BENCHMARK_RISK_FREE = "RISK_FREE"
BENCHMARK_INDEX = "INDEX"
BENCHMARK_SPY = "SPY"
BENCHMARK_QQQ = "QQQ"
BENCHMARK_CSI300 = "CSI300"
BENCHMARK_HSI = "HSI"
BENCHMARK_NANHUA = "NANHUA"
BENCHMARK_DXY = "DXY"
BENCHMARK_SECTOR_NEUTRAL = "SECTOR_NEUTRAL"
BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT = "FULL_UNIVERSE_EQUAL_WEIGHT"
BENCHMARK_SAME_HORIZON_COL = "benchmark_return_same_horizon"
DEFAULT_SPY_TICKERS = ("SPY", "SPY.US", "SPY.N")
DEFAULT_QQQ_TICKERS = ("QQQ", "QQQ.US", "QQQ.N")
DEFAULT_CSI300_TICKERS = (
    "SSE.000300",
    "000300.SH",
    "000300.SS",
    "CSI300",
    "CSI_300",
    "399300.SZ",
)
DEFAULT_HSI_TICKERS = ("^HSI", "HSI", "HSI.HK", "HANG_SENG", "HANG SENG")
DEFAULT_NANHUA_TICKERS = (
    "NANHUA",
    "NANHUA_COMMODITY",
    "NHCI",
    "南华商品指数",
    "南华综合指数",
)
DEFAULT_DXY_TICKERS = ("DXY", "DX-Y.NYB", "DX=F", "USDX")
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
        return_col: str | None = None,
    ):
        self.raw_df = raw_df.copy()
        self.benchmark_tickers = _normalize_tickers(benchmark_tickers)
        self.stale_limit = int(stale_limit)
        self.strict = bool(strict)
        self.return_col = str(return_col) if return_col else None
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
            return_col=self.return_col,
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
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
        )


class QQQBenchmark(TickerBenchmark):
    """Convenience Nasdaq 100 ETF benchmark using QQQ-style ticker aliases."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_QQQ_TICKERS,
        stale_limit: int = 3,
        strict: bool = True,
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
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
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
        )


class HSIBenchmark(TickerBenchmark):
    """Convenience Hang Seng benchmark using common index aliases."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_HSI_TICKERS,
        stale_limit: int = 3,
        strict: bool = True,
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
        )


class NanhuaBenchmark(TickerBenchmark):
    """Convenience China futures commodity-index benchmark."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_NANHUA_TICKERS,
        stale_limit: int = 3,
        strict: bool = False,
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
        )


class DXYBenchmark(TickerBenchmark):
    """Convenience US Dollar Index benchmark for FX research."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        benchmark_tickers: str | Iterable[str] = DEFAULT_DXY_TICKERS,
        stale_limit: int = 3,
        strict: bool = False,
        return_col: str | None = None,
    ):
        super().__init__(
            raw_df,
            benchmark_tickers,
            stale_limit=stale_limit,
            strict=strict,
            return_col=return_col,
        )


class BuyAndHoldBenchmark(BaseBenchmark):
    """
    Dynamic equal-weight benchmark over the assets the strategy actually traded.

    When no active strategy weights can be inferred, the benchmark falls back to
    the full raw universe. This preserves the legacy alpha-lab behavior while
    keeping the selection rule deterministic and testable.
    """

    def __init__(self, raw_df: pd.DataFrame, *, stale_limit: int = 3, return_col: str | None = None):
        self.raw_df = raw_df.copy()
        self.stale_limit = int(stale_limit)
        self.return_col = str(return_col) if return_col else None

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
            return_col=self.return_col,
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


class FullUniverseEqualWeightBenchmark(BaseBenchmark):
    """Equal-weight benchmark over the full raw tradable universe."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        *,
        stale_limit: int = 3,
        exclude_index_rows: bool = True,
        return_col: str | None = None,
    ):
        self.raw_df = raw_df.copy()
        self.stale_limit = int(stale_limit)
        self.exclude_index_rows = bool(exclude_index_rows)
        self.return_col = str(return_col) if return_col else None

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in self.raw_df.columns:
            return AbsoluteReturnBenchmark().generate(strategy_df)

        raw = self.raw_df.copy()
        if "ticker" in raw.columns:
            raw["ticker"] = raw["ticker"].astype(str)
        if self.exclude_index_rows:
            raw = raw.loc[~_detect_index_rows(raw)].copy()
        if raw.empty:
            return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

        benchmark = dynamic_equal_weight_benchmark(
            raw,
            stale_limit=self.stale_limit,
            return_col=self.return_col,
        )
        dates = _strategy_dates(strategy_df)
        if not dates.empty:
            benchmark = benchmark[benchmark["date"].isin(dates)]
        return benchmark.reset_index(drop=True)


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
        return_col: str | None = None,
    ):
        self.raw_df = raw_df.copy()
        self.sector_col = sector_col
        self.stale_limit = int(stale_limit)
        self.return_col = str(return_col) if return_col else None

    def generate(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "ticker", self.sector_col}
        if self.return_col and self.return_col in self.raw_df.columns:
            required.add(self.return_col)
        else:
            required.add("close")
        _require_columns(self.raw_df, required, "raw_df")

        raw_cols = ["date", "ticker", self.sector_col]
        raw_cols.append(self.return_col if self.return_col and self.return_col in self.raw_df.columns else "close")
        raw = self.raw_df[raw_cols].copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        if "close" in raw.columns:
            raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
            raw = raw[raw["close"] > 0]
        if self.return_col and self.return_col in raw.columns:
            raw[self.return_col] = pd.to_numeric(raw[self.return_col], errors="coerce")
        raw = raw.dropna(subset=["date", "ticker", self.sector_col])
        raw["ticker"] = raw["ticker"].astype(str)
        raw[self.sector_col] = raw[self.sector_col].astype(str)
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

        returns = _daily_ticker_returns(raw, stale_limit=self.stale_limit, return_col=self.return_col)
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
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT:
            return FullUniverseEqualWeightBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                stale_limit=stale_limit,
                exclude_index_rows=bool(kwargs.get("exclude_index_rows", True)),
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
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
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
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
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_QQQ:
            return QQQBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_QQQ_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
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
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_HSI:
            return HSIBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_HSI_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_NANHUA:
            return NanhuaBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_NANHUA_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_DXY:
            return DXYBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                benchmark_tickers=kwargs.get(  # type: ignore[arg-type]
                    "benchmark_tickers",
                    DEFAULT_DXY_TICKERS,
                ),
                stale_limit=stale_limit,
                strict=strict,
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        if benchmark_type == BENCHMARK_SECTOR_NEUTRAL:
            return SectorNeutralBenchmark(
                _require_raw_df(raw_df, benchmark_type),
                sector_col=str(kwargs.get("sector_col", "sector")),
                stale_limit=stale_limit,
                return_col=kwargs.get("return_col"),  # type: ignore[arg-type]
            )
        return AbsoluteReturnBenchmark(ann_rate=ann_rate)


def resolve_default_benchmark_policy(
    asset_class: str | None,
    *,
    factor_metadata: dict | None = None,
    factor_id: str | None = None,
) -> dict[str, object]:
    """Return the default benchmark policy for an asset taxonomy lane."""

    asset = str(asset_class or "").strip().upper()
    metadata = factor_metadata if isinstance(factor_metadata, dict) else {}
    tech_heavy = _looks_tech_heavy(metadata, factor_id=factor_id)

    if asset == "EQUITY_US":
        primary = _benchmark_policy_item(
            BENCHMARK_QQQ if tech_heavy else BENCHMARK_SPY,
            "Nasdaq 100 ETF buy-and-hold" if tech_heavy else "S&P 500 ETF buy-and-hold",
            "asset_class_market_index",
            DEFAULT_QQQ_TICKERS if tech_heavy else DEFAULT_SPY_TICKERS,
            strict=False,
        )
        secondary = _benchmark_policy_item(
            BENCHMARK_SPY if tech_heavy else BENCHMARK_QQQ,
            "S&P 500 ETF buy-and-hold" if tech_heavy else "Nasdaq 100 ETF buy-and-hold",
            "style_context_index",
            DEFAULT_SPY_TICKERS if tech_heavy else DEFAULT_QQQ_TICKERS,
            strict=False,
            column="benchmark_return_spy" if tech_heavy else "benchmark_return_qqq",
        )
        return {**primary, "secondary_benchmarks": [secondary]}

    if asset == "EQUITY_CN":
        csi300 = _benchmark_policy_item(
            BENCHMARK_CSI300,
            "CSI 300 buy-and-hold",
            "asset_class_market_index",
            DEFAULT_CSI300_TICKERS,
            strict=True,
        )
        full_universe = _benchmark_policy_item(
            BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
            "A-share full-universe equal-weight close-to-close benchmark",
            "full_universe_equal_weight",
            (),
            strict=False,
            column="benchmark_return_ashare_equal_weight",
        )
        same_horizon_control = _benchmark_policy_item(
            BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
            "A-share full-universe equal-weight same-horizon control",
            "same_horizon_universe_control",
            (),
            strict=False,
            column=BENCHMARK_SAME_HORIZON_COL,
            return_mode="same_horizon",
        )
        if _prefers_full_universe_equal_weight(metadata, factor_id=factor_id):
            full_universe = {**full_universe, "benchmark_column": BENCHMARK_RETURN_COL}
            csi300 = {**csi300, "benchmark_column": "benchmark_return_csi300"}
            return {
                **full_universe,
                "secondary_benchmarks": [csi300],
                "same_horizon_controls": [same_horizon_control],
            }
        return {
            **csi300,
            "secondary_benchmarks": [full_universe],
            "same_horizon_controls": [same_horizon_control],
        }

    if asset == "EQUITY_HK":
        return _benchmark_policy_item(
            BENCHMARK_HSI,
            "Hang Seng Index buy-and-hold",
            "asset_class_market_index",
            DEFAULT_HSI_TICKERS,
            strict=False,
        )

    if asset == "FUTURES_CN":
        primary = _benchmark_policy_item(
            BENCHMARK_BUY_AND_HOLD,
            "Dynamic equal-weight active futures universe",
            "active_universe_futures_basket",
            (),
            strict=False,
        )
        nanhua = _benchmark_policy_item(
            BENCHMARK_NANHUA,
            "Nanhua commodity index buy-and-hold",
            "broad_commodity_index",
            DEFAULT_NANHUA_TICKERS,
            strict=False,
            column="benchmark_return_nanhua",
        )
        same_horizon_control = _benchmark_policy_item(
            BENCHMARK_BUY_AND_HOLD,
            "Futures active-universe equal-weight same-horizon control",
            "same_horizon_universe_control",
            (),
            strict=False,
            column=BENCHMARK_SAME_HORIZON_COL,
            return_mode="same_horizon",
        )
        if _prefers_same_horizon_active_universe(metadata, factor_id=factor_id):
            same_horizon_primary = {
                **same_horizon_control,
                "benchmark_column": BENCHMARK_RETURN_COL,
            }
            passive_context = {
                **primary,
                "benchmark_column": "benchmark_return_passive_active_universe",
            }
            return {
                **same_horizon_primary,
                "secondary_benchmarks": [passive_context, nanhua],
            }
        return {
            **primary,
            "secondary_benchmarks": [nanhua],
            "same_horizon_controls": [same_horizon_control],
        }

    if asset == "FUTURES_US":
        primary = _benchmark_policy_item(
            BENCHMARK_BUY_AND_HOLD,
            "Dynamic equal-weight active US futures universe",
            "active_universe_futures_basket",
            (),
            strict=False,
        )
        same_horizon_control = _benchmark_policy_item(
            BENCHMARK_BUY_AND_HOLD,
            "US futures active-universe equal-weight same-horizon control",
            "same_horizon_universe_control",
            (),
            strict=False,
            column=BENCHMARK_SAME_HORIZON_COL,
            return_mode="same_horizon",
        )
        if _prefers_same_horizon_active_universe(metadata, factor_id=factor_id):
            same_horizon_primary = {
                **same_horizon_control,
                "benchmark_column": BENCHMARK_RETURN_COL,
            }
            passive_context = {
                **primary,
                "benchmark_column": "benchmark_return_passive_active_universe",
            }
            return {**same_horizon_primary, "secondary_benchmarks": [passive_context]}
        return {**primary, "same_horizon_controls": [same_horizon_control]}

    if asset == "CRYPTO_PERP":
        return _benchmark_policy_item(
            BENCHMARK_BUY_AND_HOLD,
            "Buy-and-hold traded crypto basket",
            "active_crypto_universe",
            (),
            strict=False,
        )

    if asset == "FX_SPOT":
        primary = _benchmark_policy_item(
            BENCHMARK_ABSOLUTE,
            "Cash / zero-return FX baseline",
            "cash_baseline",
            (),
            strict=False,
            ann_rate=0.0,
        )
        dxy = _benchmark_policy_item(
            BENCHMARK_DXY,
            "US Dollar Index buy-and-hold",
            "usd_context_index",
            DEFAULT_DXY_TICKERS,
            strict=False,
            column="benchmark_return_dxy",
        )
        return {**primary, "secondary_benchmarks": [dxy]}

    return _benchmark_policy_item(
        BENCHMARK_BUY_AND_HOLD,
        "Dynamic equal-weight active universe",
        "legacy_active_universe",
        (),
        strict=False,
    )


def _benchmark_policy_item(
    benchmark_type: str,
    label: str,
    role: str,
    tickers: Iterable[str],
    *,
    strict: bool,
    column: str = BENCHMARK_RETURN_COL,
    ann_rate: float | None = None,
    return_mode: str = "passive_close_to_close",
) -> dict[str, object]:
    item: dict[str, object] = {
        "benchmark_type": benchmark_type,
        "benchmark_label": label,
        "benchmark_role": role,
        "benchmark_tickers": tuple(tickers),
        "benchmark_column": column,
        "strict": bool(strict),
        "return_mode": return_mode,
    }
    if ann_rate is not None:
        item["ann_rate"] = float(ann_rate)
    return item


def _looks_tech_heavy(metadata: dict, *, factor_id: str | None = None) -> bool:
    text = " ".join(
        str(value)
        for value in (
            factor_id,
            metadata.get("factor_id"),
            metadata.get("name"),
            metadata.get("category"),
            metadata.get("sector"),
            metadata.get("universe_id"),
            metadata.get("benchmark_style"),
            metadata.get("benchmark_preference"),
        )
        if value is not None
    ).upper()
    return any(token in text for token in ("TECH", "NASDAQ", "NDX", "QQQ", "GROWTH"))


def _prefers_full_universe_equal_weight(metadata: dict, *, factor_id: str | None = None) -> bool:
    text = " ".join(
        str(value)
        for value in (
            factor_id,
            metadata.get("factor_id"),
            metadata.get("name"),
            metadata.get("category"),
            metadata.get("universe_id"),
            metadata.get("benchmark_style"),
            metadata.get("benchmark_preference"),
            metadata.get("benchmark_hint"),
        )
        if value is not None
    ).upper()
    tokens = (
        "FULL_UNIVERSE",
        "FULL UNIVERSE",
        "EQUAL_WEIGHT",
        "EQUAL WEIGHT",
        "BROAD_A_SHARE",
        "BROAD A SHARE",
        "ALL_A_SHARE",
        "ALL A SHARE",
        "指增",
    )
    return any(token in text for token in tokens)


def _prefers_same_horizon_active_universe(metadata: dict, *, factor_id: str | None = None) -> bool:
    text = " ".join(
        str(value)
        for value in (
            factor_id,
            metadata.get("factor_id"),
            metadata.get("name"),
            metadata.get("category"),
            metadata.get("universe_id"),
            metadata.get("benchmark_style"),
            metadata.get("benchmark_preference"),
            metadata.get("benchmark_hint"),
        )
        if value is not None
    ).upper()
    tokens = (
        "SAME_HORIZON_ACTIVE_UNIVERSE",
        "SAME HORIZON ACTIVE UNIVERSE",
        "SAME_HORIZON",
        "SAME HORIZON",
        "NEXT_BAR_CONTROL",
        "NEXT BAR CONTROL",
    )
    return any(token in text for token in tokens)


def dynamic_equal_weight_benchmark(
    raw_df: pd.DataFrame,
    *,
    active_matrix: pd.DataFrame | None = None,
    stale_limit: int = 3,
    return_col: str | None = None,
) -> pd.DataFrame:
    """Build daily equal-weight returns from long-form OHLCV-style data."""

    use_return_col = bool(return_col and return_col in raw_df.columns)
    required = {"date", "ticker", str(return_col) if use_return_col else "close"}
    _require_columns(raw_df, required, "raw_df")
    raw = raw_df[["date", "ticker", str(return_col) if use_return_col else "close"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    value_col = str(return_col) if use_return_col else "close"
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=["date", "ticker", value_col])
    raw["ticker"] = raw["ticker"].astype(str)
    if not use_return_col:
        raw = raw[raw["close"] > 0]
    if raw.empty:
        return pd.DataFrame(columns=["date", BENCHMARK_RETURN_COL])

    returns = _daily_ticker_returns(raw, stale_limit=stale_limit, return_col=return_col if use_return_col else None)
    returns = returns.replace([np.inf, -np.inf], pd.NA)

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
        "FULL_UNIVERSE": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "FULL_UNIVERSE_EW": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "FULL_UNIVERSE_EQUAL_WEIGHT": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "EQUAL_WEIGHT_UNIVERSE": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "UNIVERSE_EQUAL_WEIGHT": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "ASHARE_EQUAL_WEIGHT": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "A_SHARE_EQUAL_WEIGHT": BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT,
        "MARKET": BENCHMARK_INDEX,
        "TICKER": BENCHMARK_INDEX,
        "EXTERNAL_INDEX": BENCHMARK_INDEX,
        "S&P500": BENCHMARK_SPY,
        "SP500": BENCHMARK_SPY,
        "SNP500": BENCHMARK_SPY,
        "NASDAQ100": BENCHMARK_QQQ,
        "NASDAQ_100": BENCHMARK_QQQ,
        "NDX": BENCHMARK_QQQ,
        "QQQ": BENCHMARK_QQQ,
        "CSI_300": BENCHMARK_CSI300,
        "SSE.000300": BENCHMARK_CSI300,
        "000300.SH": BENCHMARK_CSI300,
        "000300.SS": BENCHMARK_CSI300,
        "399300.SZ": BENCHMARK_CSI300,
        "HANG_SENG": BENCHMARK_HSI,
        "HANGSENG": BENCHMARK_HSI,
        "^HSI": BENCHMARK_HSI,
        "NANHUA": BENCHMARK_NANHUA,
        "NHCI": BENCHMARK_NANHUA,
        "DXY": BENCHMARK_DXY,
        "USDX": BENCHMARK_DXY,
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


def _daily_ticker_returns(raw: pd.DataFrame, *, stale_limit: int = 3, return_col: str | None = None) -> pd.DataFrame:
    if return_col and return_col in raw.columns:
        daily = raw.groupby(["date", "ticker"], as_index=False)[return_col].last()
        return daily.pivot(index="date", columns="ticker", values=return_col).sort_index()

    daily = raw.groupby(["date", "ticker"], as_index=False)["close"].last()
    views = build_market_data_views(
        daily,
        timestamp_col="date",
        asset_col="ticker",
        price_cols=("close",),
        max_stale_bars=stale_limit,
    )
    prices = views.accounting.pivot(index="date", columns="ticker", values="close").sort_index()
    return prices.pct_change(fill_method=None)


def _detect_index_rows(frame: pd.DataFrame) -> pd.Series:
    is_index = pd.Series(False, index=frame.index)
    tickers = frame.get("ticker", pd.Series("", index=frame.index)).astype(str)
    is_index |= tickers.str.startswith(("SSE.000", "SHSE.000"))

    if {"exchange", "instrument"}.issubset(frame.columns):
        exchange = frame["exchange"].astype(str).str.upper()
        instrument = frame["instrument"].astype(str)
        is_index |= exchange.eq("SSE") & instrument.isin(
            {"000001", "000016", "000300", "000852", "000905"}
        )

    if "name" in frame.columns:
        names = frame["name"].astype(str).str.upper()
        is_index |= names.str.contains("指数|沪深300|上证50|中证500|中证1000|CSI", regex=True, na=False)

    return is_index.fillna(False)


__all__ = [
    "ACTIVE_WEIGHT_COLUMNS",
    "BENCHMARK_ABSOLUTE",
    "BENCHMARK_BUY_AND_HOLD",
    "BENCHMARK_CSI300",
    "BENCHMARK_DXY",
    "BENCHMARK_FULL_UNIVERSE_EQUAL_WEIGHT",
    "BENCHMARK_HSI",
    "BENCHMARK_INDEX",
    "BENCHMARK_NANHUA",
    "BENCHMARK_QQQ",
    "BENCHMARK_RISK_FREE",
    "BENCHMARK_RETURN_COL",
    "BENCHMARK_SECTOR_NEUTRAL",
    "BENCHMARK_SAME_HORIZON_COL",
    "BENCHMARK_SPY",
    "DEFAULT_CSI300_TICKERS",
    "DEFAULT_DXY_TICKERS",
    "DEFAULT_HSI_TICKERS",
    "DEFAULT_NANHUA_TICKERS",
    "DEFAULT_QQQ_TICKERS",
    "DEFAULT_SPY_TICKERS",
    "AbsoluteReturnBenchmark",
    "BaseBenchmark",
    "BenchmarkFactory",
    "BuyAndHoldBenchmark",
    "CSI300Benchmark",
    "DXYBenchmark",
    "FullUniverseEqualWeightBenchmark",
    "HSIBenchmark",
    "NanhuaBenchmark",
    "QQQBenchmark",
    "RiskFreeRateBenchmark",
    "SPYBenchmark",
    "SectorNeutralBenchmark",
    "TickerBenchmark",
    "dynamic_equal_weight_benchmark",
    "resolve_default_benchmark_policy",
]
