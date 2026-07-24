"""Live portfolio factor and PCA diagnostics.

The functions in this module deliberately consume cached price history instead
of calling market-data vendors directly. Provider jobs can populate the cache
from FMP, Yahoo, Massive, or another source without changing the dashboard math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oqp.market import normalize_price_history
from oqp.risk.factor_breadth import compute_breadth_metrics


FACTOR_PROXY_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "Market (SPY)": ("SPY",),
    "Size (IWM-SPY)": ("IWM", "SPY"),
    "Value (IWD-IWF)": ("IWD", "IWF"),
    "Momentum (MTUM-SPY)": ("MTUM", "SPY"),
    "Quality (QUAL-SPY)": ("QUAL", "SPY"),
}


@dataclass(frozen=True)
class LiveFactorLabConfig:
    lookback_days: int = 504
    min_observations: int = 60
    max_components: int = 5
    explained_threshold: float = 0.80


@dataclass(frozen=True)
class MarketRiskConfig:
    """Settings for single-benchmark portfolio risk decomposition."""

    benchmark: str = "QQQ"
    lookback_days: int = 504
    min_observations: int = 60
    annualization: int = 252
    min_covered_weight: float = 0.80


def compute_market_risk_decomposition(
    exposure: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    symbol_col: str = "Symbol",
    weight_col: str = "Weight",
    value_col: str = "Economic Exposure",
    config: MarketRiskConfig | None = None,
) -> dict[str, Any]:
    """Estimate portfolio beta and split realized risk into market and residual risk.

    Position betas are estimated independently against the benchmark. Portfolio
    beta is their current-weighted sum. The volatility split comes from an OLS
    regression of a static-weight portfolio return series on benchmark returns.
    Missing history remains explicitly uncovered instead of being assigned beta 0.
    """

    cfg = config or MarketRiskConfig()
    benchmark = canonical_risk_symbol(cfg.benchmark)
    weights = _market_risk_weights(
        exposure,
        symbol_col=symbol_col,
        weight_col=weight_col,
        value_col=value_col,
    )
    empty_positions = pd.DataFrame(
        columns=[
            "Symbol", "Weight", "Beta", "Beta Contribution", "Correlation",
            "Annualized Volatility", "Observations", "History Start", "History End", "Status",
        ]
    )
    if weights.empty or not benchmark:
        return _empty_market_risk_result(cfg, empty_positions, weights)

    history_cfg = LiveFactorLabConfig(
        lookback_days=cfg.lookback_days,
        min_observations=2,
    )
    requested = [*weights["symbol"].tolist(), benchmark]
    returns = _return_matrix(price_history, requested, history_cfg)
    if benchmark not in returns:
        return _empty_market_risk_result(cfg, empty_positions, weights, missing_benchmark=True)

    benchmark_returns = pd.to_numeric(returns[benchmark], errors="coerce")
    benchmark_variance = float(benchmark_returns.var(ddof=1))
    rows: list[dict[str, Any]] = []
    covered_symbols: list[str] = []
    weight_lookup = weights.set_index("symbol")["weight"].to_dict()
    for row in weights.itertuples(index=False):
        symbol = str(row.symbol)
        weight = float(row.weight)
        if symbol not in returns or benchmark_variance <= 0:
            rows.append(_missing_market_risk_row(symbol, weight))
            continue
        aligned = pd.concat(
            [pd.to_numeric(returns[symbol], errors="coerce").rename("asset"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) < cfg.min_observations:
            rows.append(_missing_market_risk_row(symbol, weight, observations=len(aligned)))
            continue
        beta = float(aligned["asset"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1))
        correlation = float(aligned["asset"].corr(aligned["benchmark"]))
        annualized_volatility = float(aligned["asset"].std(ddof=1) * np.sqrt(cfg.annualization))
        covered_symbols.append(symbol)
        rows.append(
            {
                "Symbol": symbol,
                "Weight": weight,
                "Beta": beta,
                "Beta Contribution": weight * beta,
                "Correlation": correlation,
                "Annualized Volatility": annualized_volatility,
                "Observations": len(aligned),
                "History Start": aligned.index.min(),
                "History End": aligned.index.max(),
                "Status": "covered",
            }
        )

    positions = pd.DataFrame(rows, columns=empty_positions.columns)
    covered_weight = float(weights.loc[weights["symbol"].isin(covered_symbols), "weight"].abs().sum())
    total_weight = float(weights["weight"].abs().sum())
    coverage_ratio = covered_weight / total_weight if total_weight > 0 else 0.0
    portfolio_beta = float(pd.to_numeric(positions["Beta Contribution"], errors="coerce").sum(min_count=1))
    if coverage_ratio < cfg.min_covered_weight:
        portfolio_beta = np.nan

    regression = _portfolio_market_regression(
        returns,
        covered_symbols,
        weight_lookup,
        benchmark,
        cfg,
    )
    calculation_status = "live" if pd.notna(portfolio_beta) and regression["status"] == "live" else "insufficient_history"
    result = {
        "benchmark": benchmark,
        "portfolio_beta": portfolio_beta,
        "covered_weight": coverage_ratio,
        "positions": positions.sort_values("Beta Contribution", ascending=False, na_position="last").reset_index(drop=True),
        **regression,
        "status": calculation_status,
    }
    return result


def _market_risk_weights(
    exposure: pd.DataFrame,
    *,
    symbol_col: str,
    weight_col: str,
    value_col: str,
) -> pd.DataFrame:
    if exposure.empty or symbol_col not in exposure:
        return pd.DataFrame(columns=["symbol", "weight"])
    source = exposure.copy()
    source["symbol"] = source[symbol_col].map(canonical_risk_symbol)
    source = source.loc[source["symbol"].ne("") & ~source["symbol"].isin({"CASH", "USD CASH"})]
    if weight_col in source:
        source["weight"] = pd.to_numeric(source[weight_col], errors="coerce")
    else:
        values = pd.to_numeric(source.get(value_col), errors="coerce")
        denominator = float(values.abs().sum())
        source["weight"] = values / denominator if denominator > 0 else np.nan
    grouped = source.groupby("symbol", as_index=False)["weight"].sum(min_count=1).dropna(subset=["weight"])
    return grouped.loc[grouped["weight"].ne(0)].reset_index(drop=True)


def _portfolio_market_regression(
    returns: pd.DataFrame,
    symbols: list[str],
    weights: dict[str, float],
    benchmark: str,
    cfg: MarketRiskConfig,
) -> dict[str, Any]:
    columns = [symbol for symbol in symbols if symbol in returns]
    if not columns or benchmark not in returns:
        return _empty_market_regression()
    regression_columns = list(dict.fromkeys([*columns, benchmark]))
    aligned = returns[regression_columns].dropna()
    if len(aligned) < cfg.min_observations:
        return _empty_market_regression(observations=len(aligned))
    portfolio = sum(aligned[symbol] * float(weights.get(symbol, 0.0)) for symbol in columns)
    market = aligned[benchmark]
    x = np.column_stack([np.ones(len(aligned)), market.to_numpy(dtype=float)])
    y = portfolio.to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - (x @ coefficients)
    total_variance = float(np.var(y, ddof=1))
    systematic_variance = float(coefficients[1] ** 2 * np.var(market, ddof=1))
    idiosyncratic_variance = float(np.var(residual, ddof=1))
    explained_total = systematic_variance + idiosyncratic_variance
    return {
        "status": "live",
        "observations": len(aligned),
        "history_start": aligned.index.min(),
        "history_end": aligned.index.max(),
        "total_volatility": np.sqrt(max(total_variance, 0.0) * cfg.annualization),
        "systematic_volatility": np.sqrt(max(systematic_variance, 0.0) * cfg.annualization),
        "idiosyncratic_volatility": np.sqrt(max(idiosyncratic_variance, 0.0) * cfg.annualization),
        "systematic_risk_share": systematic_variance / explained_total if explained_total > 0 else np.nan,
        "idiosyncratic_risk_share": idiosyncratic_variance / explained_total if explained_total > 0 else np.nan,
        "regression_beta": float(coefficients[1]),
        "alpha_annualized": float(coefficients[0] * cfg.annualization),
    }


def _missing_market_risk_row(symbol: str, weight: float, *, observations: int = 0) -> dict[str, Any]:
    return {
        "Symbol": symbol,
        "Weight": weight,
        "Beta": np.nan,
        "Beta Contribution": np.nan,
        "Correlation": np.nan,
        "Annualized Volatility": np.nan,
        "Observations": observations,
        "History Start": pd.NaT,
        "History End": pd.NaT,
        "Status": "insufficient history",
    }


def _empty_market_regression(*, observations: int = 0) -> dict[str, Any]:
    return {
        "status": "insufficient_history",
        "observations": observations,
        "history_start": pd.NaT,
        "history_end": pd.NaT,
        "total_volatility": np.nan,
        "systematic_volatility": np.nan,
        "idiosyncratic_volatility": np.nan,
        "systematic_risk_share": np.nan,
        "idiosyncratic_risk_share": np.nan,
        "regression_beta": np.nan,
        "alpha_annualized": np.nan,
    }


def _empty_market_risk_result(
    cfg: MarketRiskConfig,
    positions: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    missing_benchmark: bool = False,
) -> dict[str, Any]:
    result = {
        "benchmark": canonical_risk_symbol(cfg.benchmark),
        "portfolio_beta": np.nan,
        "covered_weight": 0.0,
        "positions": positions,
        **_empty_market_regression(),
    }
    result["status"] = "missing_benchmark" if missing_benchmark else "insufficient_history"
    return result


def factor_proxy_symbols() -> list[str]:
    """Return all symbols required for the default factor-proxy set."""

    symbols: list[str] = []
    for legs in FACTOR_PROXY_DEFINITIONS.values():
        symbols.extend(legs)
    return list(dict.fromkeys(symbols))


def combine_price_histories(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple long-form price-history frames, keeping latest duplicates."""

    normalized = [normalize_price_history(frame) for frame in frames if frame is not None and not frame.empty]
    if not normalized:
        return pd.DataFrame(columns=["symbol", "date", "close"])
    out = pd.concat(normalized, ignore_index=True)
    out["symbol"] = out["symbol"].map(canonical_risk_symbol)
    out = out.dropna(subset=["date", "close"])
    return (
        out.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def exposure_symbols(exposure: pd.DataFrame, *, symbol_col: str = "Symbol") -> list[str]:
    """Return risk symbols from an exposure table, excluding cash placeholders."""

    if exposure.empty or symbol_col not in exposure:
        return []
    symbols = [canonical_risk_symbol(value) for value in exposure[symbol_col].tolist()]
    return [symbol for symbol in dict.fromkeys(symbols) if symbol and symbol not in {"CASH", "USD CASH"}]


def compute_factor_proxy_lab(
    exposure: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    symbol_col: str = "Symbol",
    value_col: str = "Economic Exposure",
    config: LiveFactorLabConfig | None = None,
) -> dict[str, Any]:
    """Estimate portfolio sensitivity to ETF-based style proxies."""

    cfg = config or LiveFactorLabConfig()
    weights = _exposure_weights(exposure, symbol_col=symbol_col, value_col=value_col)
    returns = _return_matrix(price_history, weights["symbol"].tolist(), cfg)
    portfolio = _portfolio_returns(returns, weights)
    factor_returns, missing_proxies = _factor_proxy_returns(price_history, cfg)
    if portfolio.empty or factor_returns.empty:
        return _empty_factor_result(weights, returns, factor_returns, missing_proxies)

    aligned = pd.concat([portfolio.rename("portfolio"), factor_returns], axis=1).dropna()
    if len(aligned) < cfg.min_observations or aligned.shape[1] < 2:
        return _empty_factor_result(weights, returns, factor_returns, missing_proxies, aligned_rows=len(aligned))

    y = aligned["portfolio"].to_numpy(dtype=float)
    x = aligned.drop(columns=["portfolio"]).to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(aligned)), x])
    coefficients, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ coefficients
    residual = y - fitted
    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(np.sum(residual**2))
    r_squared = 0.0 if total_ss <= 0 else 1.0 - residual_ss / total_ss
    portfolio_vol = float(np.std(y, ddof=1) * np.sqrt(252)) if len(y) > 1 else np.nan
    residual_vol = float(np.std(residual, ddof=1) * np.sqrt(252)) if len(residual) > 1 else np.nan

    factors = aligned.drop(columns=["portfolio"]).columns.tolist()
    beta_rows = [
        {
            "Factor": factor,
            "Beta": float(beta),
            "Proxy": _proxy_description(factor),
        }
        for factor, beta in zip(factors, coefficients[1:], strict=False)
    ]
    summary = pd.DataFrame(
        [
            {"Metric": "Observations", "Value": len(aligned), "Detail": _date_range(aligned.index)},
            {"Metric": "Portfolio assets covered", "Value": len(returns.columns), "Detail": ", ".join(returns.columns[:12])},
            {"Metric": "Factors available", "Value": len(factors), "Detail": ", ".join(factors)},
            {"Metric": "R squared", "Value": r_squared, "Detail": "Share of portfolio return variation explained by proxies."},
            {"Metric": "Portfolio vol", "Value": portfolio_vol, "Detail": "Annualized realized volatility."},
            {"Metric": "Residual vol", "Value": residual_vol, "Detail": "Annualized volatility after removing proxy factor moves."},
        ]
    )
    return {
        "status": "live",
        "summary": summary,
        "betas": pd.DataFrame(beta_rows),
        "factor_returns": factor_returns,
        "portfolio_returns": portfolio,
        "missing_proxies": missing_proxies,
        "missing_assets": [symbol for symbol in weights["symbol"] if symbol not in returns.columns],
    }


def compute_pca_crowding_lab(
    exposure: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    symbol_col: str = "Symbol",
    value_col: str = "Economic Exposure",
    config: LiveFactorLabConfig | None = None,
) -> dict[str, Any]:
    """Run correlation PCA over portfolio underlying returns."""

    cfg = config or LiveFactorLabConfig()
    weights = _exposure_weights(exposure, symbol_col=symbol_col, value_col=value_col)
    returns = _return_matrix(price_history, weights["symbol"].tolist(), cfg)
    if returns.shape[1] < 2 or len(returns) < cfg.min_observations:
        return _empty_pca_result(weights, returns)

    x = returns.dropna(how="all").copy()
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=1).replace(0, np.nan)
    z = x.sub(means, axis=1).div(stds, axis=1).dropna(axis=1, how="all").fillna(0.0)
    if z.shape[1] < 2:
        return _empty_pca_result(weights, returns)

    covariance = np.cov(z.to_numpy(dtype=float), rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(explained)
    components = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    spectrum = pd.DataFrame(
        {
            "Component": components,
            "Explained Variance": explained,
            "Cumulative": cumulative,
            "Eigenvalue": eigenvalues,
        }
    )

    loadings_rows: list[dict[str, Any]] = []
    n_components = min(cfg.max_components, eigenvectors.shape[1])
    weight_lookup = weights.set_index("symbol")["weight"].to_dict()
    for comp_idx in range(n_components):
        component = f"PC{comp_idx + 1}"
        for asset_idx, symbol in enumerate(z.columns):
            loading = float(eigenvectors[asset_idx, comp_idx])
            loadings_rows.append(
                {
                    "Symbol": symbol,
                    "Component": component,
                    "Loading": loading,
                    "Abs Loading": abs(loading),
                    "Portfolio Weight": float(weight_lookup.get(symbol, 0.0)),
                    "Explained Variance": float(explained[comp_idx]),
                }
            )
    loadings = pd.DataFrame(loadings_rows)
    top_drivers = _pca_top_drivers(loadings, spectrum)
    metrics = compute_breadth_metrics(
        eigenvalues,
        variance_threshold=cfg.explained_threshold,
        naive_breadth=len(z.columns),
    )
    summary = pd.DataFrame(
        [
            {"Metric": "Observations", "Value": len(z), "Detail": _date_range(z.index)},
            {"Metric": "Assets covered", "Value": len(z.columns), "Detail": ", ".join(z.columns[:12])},
            {"Metric": "PC1 variance", "Value": float(explained[0]) if len(explained) else np.nan, "Detail": "Higher means one common driver dominates."},
            {"Metric": "Components to 80%", "Value": metrics["br_threshold"], "Detail": "Correlation-PCA breadth estimate."},
            {"Metric": "Effective rank", "Value": metrics["effective_rank"], "Detail": "Lower means more crowded common risk."},
        ]
    )
    return {
        "status": "live",
        "summary": summary,
        "spectrum": spectrum,
        "loadings": loadings,
        "top_drivers": top_drivers,
        "returns": returns,
        "missing_assets": [symbol for symbol in weights["symbol"] if symbol not in returns.columns],
    }


def canonical_risk_symbol(value: object) -> str:
    text = str(value or "").upper().strip()
    if not text or text in {"NAN", "NONE", "MISSING"}:
        return ""
    if text in {"0700", "700", "0700.HK", "700.HK", "TCEHY", "TENCENT"}:
        return "TENCENT"
    if text in {"BRK B", "BRK.B", "BRK-B"}:
        return "BRK B"
    return text


def _empty_factor_result(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    missing_proxies: list[str],
    *,
    aligned_rows: int = 0,
) -> dict[str, Any]:
    summary = pd.DataFrame(
        [
            {"Metric": "Observations", "Value": aligned_rows, "Detail": "Need more overlapping portfolio and factor return history."},
            {"Metric": "Portfolio assets covered", "Value": len(returns.columns), "Detail": ", ".join(returns.columns[:12])},
            {"Metric": "Factors available", "Value": len(factor_returns.columns), "Detail": ", ".join(factor_returns.columns)},
        ]
    )
    return {
        "status": "insufficient_history",
        "summary": summary,
        "betas": pd.DataFrame(columns=["Factor", "Beta", "Proxy"]),
        "factor_returns": factor_returns,
        "portfolio_returns": pd.Series(dtype=float),
        "missing_proxies": missing_proxies,
        "missing_assets": [symbol for symbol in weights["symbol"] if symbol not in returns.columns],
    }


def _empty_pca_result(weights: pd.DataFrame, returns: pd.DataFrame) -> dict[str, Any]:
    summary = pd.DataFrame(
        [
            {"Metric": "Observations", "Value": len(returns), "Detail": "Need enough overlapping return history."},
            {"Metric": "Assets covered", "Value": len(returns.columns), "Detail": ", ".join(returns.columns[:12])},
        ]
    )
    return {
        "status": "insufficient_history",
        "summary": summary,
        "spectrum": pd.DataFrame(columns=["Component", "Explained Variance", "Cumulative", "Eigenvalue"]),
        "loadings": pd.DataFrame(columns=["Symbol", "Component", "Loading", "Abs Loading", "Portfolio Weight", "Explained Variance"]),
        "top_drivers": pd.DataFrame(columns=["Component", "Explained Variance", "Top Positive", "Top Negative"]),
        "returns": returns,
        "missing_assets": [symbol for symbol in weights["symbol"] if symbol not in returns.columns],
    }


def _exposure_weights(
    exposure: pd.DataFrame,
    *,
    symbol_col: str,
    value_col: str,
) -> pd.DataFrame:
    columns = ["symbol", "exposure", "weight"]
    if exposure.empty or symbol_col not in exposure:
        return pd.DataFrame(columns=columns)
    source = exposure.copy()
    source["symbol"] = source[symbol_col].map(canonical_risk_symbol)
    if value_col not in source:
        fallback = "Exposure Value" if "Exposure Value" in source else "Market Value"
        value_col = fallback if fallback in source else value_col
    source["exposure"] = pd.to_numeric(source.get(value_col, pd.Series(0.0, index=source.index)), errors="coerce").fillna(0.0).abs()
    source = source.loc[source["symbol"].ne("") & ~source["symbol"].isin({"CASH", "USD CASH"})]
    grouped = source.groupby("symbol", as_index=False)["exposure"].sum()
    total = float(grouped["exposure"].sum()) if not grouped.empty else 0.0
    grouped["weight"] = 0.0 if total <= 0 else grouped["exposure"] / total
    return grouped.sort_values("exposure", ascending=False).reset_index(drop=True)


def _return_matrix(
    price_history: pd.DataFrame,
    symbols: Iterable[str],
    config: LiveFactorLabConfig,
) -> pd.DataFrame:
    requested = [canonical_risk_symbol(symbol) for symbol in symbols]
    requested = [symbol for symbol in dict.fromkeys(requested) if symbol]
    history = combine_price_histories(price_history)
    if history.empty or not requested:
        return pd.DataFrame()
    if config.lookback_days > 0:
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=config.lookback_days)
        history = history.loc[pd.to_datetime(history["date"]).dt.tz_localize(None) >= cutoff]
    history = history.loc[history["symbol"].isin(requested)]
    prices = history.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    valid = returns.columns[returns.notna().sum() >= config.min_observations].tolist()
    return returns[valid].dropna(how="all") if valid else pd.DataFrame(index=returns.index)


def _portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    if returns.empty or weights.empty:
        return pd.Series(dtype=float)
    aligned_weights = weights.set_index("symbol")["weight"].reindex(returns.columns).fillna(0.0)
    if float(aligned_weights.sum()) <= 0:
        return pd.Series(dtype=float)
    aligned_weights = aligned_weights / aligned_weights.sum()
    return returns.fillna(0.0).dot(aligned_weights)


def _factor_proxy_returns(
    price_history: pd.DataFrame,
    config: LiveFactorLabConfig,
) -> tuple[pd.DataFrame, list[str]]:
    proxy_returns = _return_matrix(price_history, factor_proxy_symbols(), config)
    factors: dict[str, pd.Series] = {}
    missing: list[str] = []
    for factor, legs in FACTOR_PROXY_DEFINITIONS.items():
        if len(legs) == 1:
            symbol = legs[0]
            if symbol in proxy_returns:
                factors[factor] = proxy_returns[symbol]
            else:
                missing.append(symbol)
            continue
        long_leg, short_leg = legs
        if long_leg in proxy_returns and short_leg in proxy_returns:
            factors[factor] = proxy_returns[long_leg] - proxy_returns[short_leg]
        else:
            missing.extend([symbol for symbol in legs if symbol not in proxy_returns])
    return pd.DataFrame(factors).dropna(how="all"), list(dict.fromkeys(missing))


def _pca_top_drivers(loadings: pd.DataFrame, spectrum: pd.DataFrame) -> pd.DataFrame:
    if loadings.empty or spectrum.empty:
        return pd.DataFrame(columns=["Component", "Explained Variance", "Top Positive", "Top Negative"])
    rows = []
    variance = spectrum.set_index("Component")["Explained Variance"].to_dict()
    for component, group in loadings.groupby("Component"):
        positive = group.sort_values("Loading", ascending=False).head(4)
        negative = group.sort_values("Loading", ascending=True).head(4)
        rows.append(
            {
                "Component": component,
                "Explained Variance": float(variance.get(component, np.nan)),
                "Top Positive": ", ".join(positive["Symbol"].tolist()),
                "Top Negative": ", ".join(negative["Symbol"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _proxy_description(factor: str) -> str:
    legs = FACTOR_PROXY_DEFINITIONS.get(factor, ())
    return legs[0] if len(legs) == 1 else " - ".join(legs)


def _date_range(index: Any) -> str:
    values = pd.to_datetime(pd.Index(index), errors="coerce").dropna()
    if values.empty:
        return "missing"
    return f"{values.min().date()} to {values.max().date()}"
