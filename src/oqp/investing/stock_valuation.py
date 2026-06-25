"""Equity research helpers for the money dashboard stock valuation page."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from oqp.data import FMPDataAdapter
from oqp.data.base import DataAdapterError


FMPFetcher = Callable[..., Any]


VALUATION_MULTIPLE_COLUMNS: dict[str, tuple[str, str]] = {
    "peRatioTTM": (
        "P/E Ratio",
        "Price / Earnings. Measures how much investors pay per dollar of earnings.",
    ),
    "pfcfRatioTTM": (
        "P/FCF",
        "Price / Free Cash Flow. Uses hard cash rather than accounting earnings.",
    ),
    "pbRatioTTM": (
        "P/B Ratio",
        "Price / Book Value. Compares market value to accounting equity.",
    ),
    "enterpriseValueMultipleTTM": (
        "EV/EBITDA",
        "Enterprise Value / EBITDA. A capital-structure-neutral valuation multiple.",
    ),
    "evToSalesTTM": ("EV/Sales", "Enterprise Value / Revenue."),
    "evToOperatingCashFlowTTM": (
        "EV/OCF",
        "Enterprise Value / Operating Cash Flow.",
    ),
    "evToFreeCashFlowTTM": ("EV/FCF", "Enterprise Value / Free Cash Flow."),
    "earningsYieldTTM": (
        "Earnings Yield",
        "Earnings / Price. The inverse of P/E.",
    ),
    "freeCashFlowYieldTTM": (
        "FCF Yield",
        "Free Cash Flow / Market Cap.",
    ),
    "dividendYieldTTM": ("Dividend Yield", "Annual Dividends Per Share / Price."),
}


RATIO_CATEGORIES: dict[str, dict[str, tuple[str, str]]] = {
    "Liquidity": {
        "currentRatioTTM": (
            "Current Ratio",
            "Current Assets / Current Liabilities.",
        ),
        "quickRatioTTM": (
            "Quick Ratio",
            "(Current Assets - Inventory) / Current Liabilities.",
        ),
        "cashRatioTTM": (
            "Cash Ratio",
            "Cash & Equivalents / Current Liabilities.",
        ),
    },
    "Efficiency": {
        "daysOfSalesOutstandingTTM": (
            "Days Sales Outstanding",
            "Average time to collect revenue after a sale.",
        ),
        "daysOfInventoryOutstandingTTM": (
            "Days Inventory Outstanding",
            "Average time to turn inventory into sales.",
        ),
        "daysOfPayablesOutstandingTTM": (
            "Days Payables Outstanding",
            "Average time the company takes to pay suppliers.",
        ),
        "cashConversionCycleTTM": (
            "Cash Conversion Cycle",
            "DIO + DSO - DPO.",
        ),
        "assetTurnoverTTM": (
            "Asset Turnover",
            "Revenue / Total Assets.",
        ),
    },
    "Profitability": {
        "grossProfitMarginTTM": ("Gross Margin", "Gross Profit / Revenue."),
        "operatingProfitMarginTTM": (
            "Operating Margin",
            "Operating Income / Revenue.",
        ),
        "netProfitMarginTTM": ("Net Margin", "Net Income / Revenue."),
        "returnOnAssetsTTM": ("ROA", "Net Income / Total Assets."),
        "returnOnEquityTTM": ("ROE", "Net Income / Shareholder Equity."),
        "returnOnCapitalEmployedTTM": (
            "ROCE",
            "EBIT / Capital Employed.",
        ),
    },
    "Leverage": {
        "debtRatioTTM": ("Debt Ratio", "Total Debt / Total Assets."),
        "debtEquityRatioTTM": (
            "Debt-to-Equity",
            "Total Debt / Shareholder Equity.",
        ),
        "longTermDebtToCapitalizationTTM": (
            "LT Debt to Capitalization",
            "Long Term Debt / (Long Term Debt + Equity).",
        ),
    },
    "Coverage": {
        "interestCoverageTTM": (
            "Interest Coverage",
            "EBIT / Interest Expense.",
        ),
        "cashFlowCoverageRatiosTTM": (
            "Cash Flow to Debt",
            "Operating Cash Flow / Total Debt.",
        ),
    },
    "Operating Cash Flow": {
        "operatingCashFlowSalesRatioTTM": (
            "OCF Margin",
            "Operating Cash Flow / Revenue.",
        ),
        "freeCashFlowOperatingCashFlowRatioTTM": (
            "FCF/OCF",
            "Free Cash Flow / Operating Cash Flow.",
        ),
        "capitalExpenditureCoverageRatioTTM": (
            "CapEx Coverage",
            "Operating Cash Flow / CapEx.",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class DCFValuation:
    future_fcf: list[float]
    present_value_fcf: list[float]
    terminal_value: float
    present_value_terminal: float
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    margin_of_safety: float

    @property
    def bridge_rows(self) -> list[dict[str, float | str]]:
        return [
            {"Valuation Metric": "PV of 10-Yr Cash Flows", "Amount": sum(self.present_value_fcf)},
            {"Valuation Metric": "PV of Terminal Value", "Amount": self.present_value_terminal},
            {"Valuation Metric": "Enterprise Value", "Amount": self.enterprise_value},
            {"Valuation Metric": "Equity Value", "Amount": self.equity_value},
        ]


@dataclass(frozen=True, slots=True)
class PeerComparison:
    peer_symbols: list[str] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    ratios: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None


def safe_num(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def calculate_cagr(values: pd.Series | list[float]) -> float:
    series = pd.Series(values).dropna()
    if len(series) < 2:
        return 0.0
    start = safe_num(series.iloc[0])
    end = safe_num(series.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (1 / (len(series) - 1)) - 1


def latest_statement_value(df: pd.DataFrame, possible_columns: list[str]) -> float:
    if df.empty:
        return 0.0
    for column in possible_columns:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                return safe_num(series.iloc[-1])
    return 0.0


def fetch_fmp_json(
    api_key: str | None,
    endpoint: str,
    *,
    stable: bool = False,
    params: dict[str, Any] | None = None,
    suppress_error_messages: bool = True,
    adapter_factory: Callable[[str], FMPDataAdapter] = FMPDataAdapter,
) -> Any:
    if not api_key:
        return []
    try:
        json_data = adapter_factory(api_key).get_json(
            endpoint,
            stable=stable,
            params=params,
        )
    except (DataAdapterError, OSError, ValueError):
        return []

    if suppress_error_messages and isinstance(json_data, dict) and "Error Message" in json_data:
        return []
    return json_data


def _statement_frame(statement: object) -> pd.DataFrame:
    if isinstance(statement, pd.DataFrame) and not statement.empty:
        return statement.T.sort_index()
    return pd.DataFrame()


def fetch_fundamental_data(
    ticker_symbol: str,
    api_key: str | None,
    *,
    ticker_factory: Callable[[str], Any] | None = None,
    fmp_fetcher: FMPFetcher = fetch_fmp_json,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bool]]:
    """Fetch the data bundle consumed by the stock valuation dashboard."""

    symbol = ticker_symbol.upper().strip()
    if not symbol:
        return {}, {}, {}, {}

    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker

    ticker = ticker_factory(symbol)
    data: dict[str, Any] = {}
    financials: dict[str, Any] = {}
    historicals: dict[str, Any] = {}
    mandates: dict[str, bool] = {}

    profile_fmp = fmp_fetcher(api_key, f"profile/{symbol}")
    info = getattr(ticker, "info", {}) or {}

    data["price"] = (
        safe_num(profile_fmp[0].get("price")) if isinstance(profile_fmp, list) and profile_fmp else safe_num(info.get("currentPrice"))
    )
    data["market_cap"] = (
        safe_num(profile_fmp[0].get("mktCap")) if isinstance(profile_fmp, list) and profile_fmp else safe_num(info.get("marketCap"))
    )
    data["sector"] = (
        profile_fmp[0].get("sector", "Unknown") if isinstance(profile_fmp, list) and profile_fmp else info.get("sector", "Unknown")
    )
    data["pe"] = safe_num(info.get("trailingPE"), 999.0)
    data["peg"] = safe_num(info.get("pegRatio"), 999.0)
    data["shares"] = safe_num(info.get("sharesOutstanding"), 1.0) or 1.0
    data["total_cash"] = safe_num(info.get("totalCash"))
    data["total_debt"] = safe_num(info.get("totalDebt"))

    inc = _statement_frame(getattr(ticker, "financials", pd.DataFrame()))
    bal = _statement_frame(getattr(ticker, "balance_sheet", pd.DataFrame()))
    cf = _statement_frame(getattr(ticker, "cashflow", pd.DataFrame()))
    q_inc = _statement_frame(getattr(ticker, "quarterly_financials", pd.DataFrame()))
    q_bal = _statement_frame(getattr(ticker, "quarterly_balance_sheet", pd.DataFrame()))
    q_cf = _statement_frame(getattr(ticker, "quarterly_cashflow", pd.DataFrame()))

    data["q_rev"] = latest_statement_value(q_inc, ["Total Revenue", "Operating Revenue"])
    data["q_gross"] = latest_statement_value(q_inc, ["Gross Profit"])
    data["q_op_inc"] = latest_statement_value(q_inc, ["Operating Income", "EBIT", "Ebit"])
    data["q_curr_assets"] = latest_statement_value(q_bal, ["Total Current Assets", "Current Assets"])
    data["q_curr_liab"] = latest_statement_value(q_bal, ["Total Current Liabilities", "Current Liabilities"])
    data["q_inventory"] = latest_statement_value(q_bal, ["Inventory"])
    data["q_total_debt"] = latest_statement_value(q_bal, ["Total Debt", "Long Term Debt"])
    data["q_equity"] = latest_statement_value(q_bal, ["Stockholders Equity", "Total Stockholder Equity"])
    q_ocf = latest_statement_value(q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    q_capex = latest_statement_value(q_cf, ["Capital Expenditure"])
    data["q_fcf"] = q_ocf - abs(q_capex)

    rev = pd.to_numeric(inc.get("Total Revenue", pd.Series(dtype="float64")), errors="coerce")
    fcf_series = pd.to_numeric(cf.get("Free Cash Flow", pd.Series(dtype="float64")), errors="coerce")
    data["ttm_revenue"] = safe_num(rev.iloc[-1]) if not rev.empty else 0.0
    fallback_fcf = latest_statement_value(cf, ["Operating Cash Flow"]) - abs(
        latest_statement_value(cf, ["Capital Expenditure"])
    )
    data["fcf_ttm"] = safe_num(fcf_series.iloc[-1]) if not fcf_series.empty else fallback_fcf

    ebit = latest_statement_value(inc, ["Operating Income", "EBIT", "Ebit"])
    total_assets = latest_statement_value(bal, ["Total Assets"])
    current_liabilities = latest_statement_value(
        bal,
        ["Total Current Liabilities", "Current Liabilities"],
    )
    capital_employed = total_assets - current_liabilities
    data["roce"] = ebit / capital_employed if capital_employed > 0 else 0.0
    data["auto_rev_cagr"] = calculate_cagr(rev.tail(4)) * 100 if not rev.empty else 12.0
    data["auto_fcf_cagr"] = calculate_cagr(fcf_series.tail(4)) * 100 if not fcf_series.empty else 15.0

    net_income = pd.to_numeric(inc.get("Net Income", pd.Series(dtype="float64")), errors="coerce")
    curr_assets = pd.to_numeric(
        bal.get("Total Current Assets", bal.get("Current Assets", pd.Series(dtype="float64"))),
        errors="coerce",
    )
    curr_liab = pd.to_numeric(
        bal.get("Total Current Liabilities", bal.get("Current Liabilities", pd.Series(dtype="float64"))),
        errors="coerce",
    )
    lt_debt = pd.to_numeric(
        bal.get("Long Term Debt", bal.get("Total Debt", pd.Series(dtype="float64"))),
        errors="coerce",
    )
    equity = pd.to_numeric(
        bal.get("Stockholders Equity", bal.get("Total Stockholder Equity", pd.Series(dtype="float64"))),
        errors="coerce",
    )
    shares_hist = pd.to_numeric(
        inc.get("Basic Average Shares", inc.get("Diluted Average Shares", pd.Series(dtype="float64"))),
        errors="coerce",
    )
    ocf_hist = pd.to_numeric(cf.get("Operating Cash Flow", pd.Series(dtype="float64")), errors="coerce")
    capex_hist = pd.to_numeric(cf.get("Capital Expenditure", pd.Series(dtype="float64")), errors="coerce")
    divs_hist = pd.to_numeric(cf.get("Cash Dividends Paid", pd.Series(dtype="float64")), errors="coerce")

    pe = safe_num(data["pe"], 999.0)
    peg = safe_num(data["peg"], 999.0)
    mandates["#1 P/E<25 & PEG<1"] = (0 < pe < 25) and (0 < peg < 1)
    mandates["#2 Revenue growth +"] = calculate_cagr(rev) > 0 if not rev.empty else False
    mandates["#3 Op. profit growth +"] = (
        calculate_cagr(pd.to_numeric(inc.get("Operating Income", pd.Series(dtype="float64")), errors="coerce")) > 0
    )
    mandates["#4 Net profit growth +"] = calculate_cagr(net_income) > 0 if not net_income.empty else False
    mandates["#5 Current assets > liabilities"] = (
        safe_num(curr_assets.iloc[-1]) > safe_num(curr_liab.iloc[-1])
        if not curr_assets.empty and not curr_liab.empty
        else False
    )
    mandates["#6 LT Debt/Net Inc <4"] = (
        safe_num(lt_debt.iloc[-1]) / safe_num(net_income.iloc[-1]) < 4
        if not lt_debt.empty and not net_income.empty and safe_num(net_income.iloc[-1]) > 0
        else False
    )
    mandates["#7 Equity growing"] = calculate_cagr(equity) > 0 if not equity.empty else False
    mandates["#8 Shares decreasing"] = calculate_cagr(shares_hist) < 0 if len(shares_hist) > 1 else False
    if not ocf_hist.empty and not capex_hist.empty:
        dividends = abs(safe_num(divs_hist.iloc[-1])) if not divs_hist.empty else 0.0
        mandates["#9 OCF covers capex & payout"] = safe_num(ocf_hist.iloc[-1]) > (
            abs(safe_num(capex_hist.iloc[-1])) + dividends
        )
    else:
        mandates["#9 OCF covers capex & payout"] = False
    mandates["#10 FCF growth +"] = calculate_cagr(fcf_series) > 0 if not fcf_series.empty else False

    try:
        hist = ticker.history(period="1y")
    except Exception:
        hist = pd.DataFrame()
    if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
        hist = hist.copy()
        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        std_20 = hist["Close"].rolling(window=20).std()
        hist["BB_Upper"] = hist["SMA_20"] + (std_20 * 2)
        hist["BB_Lower"] = hist["SMA_20"] - (std_20 * 2)
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
        ema_12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = hist["Close"].ewm(span=26, adjust=False).mean()
        hist["MACD"] = ema_12 - ema_26
        hist["Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_Hist"] = hist["MACD"] - hist["Signal"]
        data["hist_df"] = hist
        historicals["price_history_1y"] = hist

    financials.update(
        {
            "income_statement": inc,
            "balance_sheet": bal,
            "cash_flow": cf,
            "quarterly_income_statement": q_inc,
            "quarterly_balance_sheet": q_bal,
            "quarterly_cash_flow": q_cf,
        }
    )
    return data, financials, historicals, mandates


def calculate_dcf_valuation(
    data: dict[str, Any],
    *,
    model: str,
    wacc: float,
    terminal_growth: float,
    fcf_growth_1: float | None = None,
    fcf_growth_2: float | None = None,
    revenue_growth_1: float | None = None,
    revenue_growth_2: float | None = None,
    target_fcf_margin: float | None = None,
) -> DCFValuation:
    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")

    future_fcf: list[float] = []
    if model == "standard":
        current_fcf = safe_num(data.get("fcf_ttm"))
        growth_1 = fcf_growth_1 if fcf_growth_1 is not None else 0.15
        growth_2 = fcf_growth_2 if fcf_growth_2 is not None else growth_1 * 0.7
        for year in range(10):
            current_fcf *= 1 + (growth_1 if year < 5 else growth_2)
            future_fcf.append(current_fcf)
    elif model == "margin":
        ttm_revenue = safe_num(data.get("ttm_revenue"))
        current_margin = safe_num(data.get("fcf_ttm")) / ttm_revenue if ttm_revenue > 0 else 0.0
        target_margin = target_fcf_margin if target_fcf_margin is not None else 0.25
        margins = np.linspace(current_margin, target_margin, 10)
        current_revenue = ttm_revenue
        growth_1 = revenue_growth_1 if revenue_growth_1 is not None else 0.12
        growth_2 = revenue_growth_2 if revenue_growth_2 is not None else growth_1 * 0.7
        for year in range(10):
            current_revenue *= 1 + (growth_1 if year < 5 else growth_2)
            future_fcf.append(float(current_revenue * margins[year]))
    else:
        raise ValueError(f"Unknown DCF model: {model}")

    present_value_fcf = [
        cash_flow / (1 + wacc) ** (index + 1)
        for index, cash_flow in enumerate(future_fcf)
    ]
    terminal_value = (future_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
    present_value_terminal = terminal_value / (1 + wacc) ** 10
    enterprise_value = sum(present_value_fcf) + present_value_terminal
    equity_value = enterprise_value + safe_num(data.get("total_cash")) - safe_num(data.get("total_debt"))
    shares = safe_num(data.get("shares"), 1.0) or 1.0
    fair_value = equity_value / shares
    price = safe_num(data.get("price"))
    margin_of_safety = (fair_value - price) / fair_value if fair_value > 0 else 0.0

    return DCFValuation(
        future_fcf=[float(value) for value in future_fcf],
        present_value_fcf=[float(value) for value in present_value_fcf],
        terminal_value=float(terminal_value),
        present_value_terminal=float(present_value_terminal),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        fair_value_per_share=float(fair_value),
        margin_of_safety=float(margin_of_safety),
    )


def fetch_price_target_consensus(api_key: str | None, symbol: str) -> dict[str, Any]:
    result = fetch_fmp_json(
        api_key,
        "price-target-consensus",
        stable=True,
        params={"symbol": symbol.upper()},
        suppress_error_messages=False,
    )
    if isinstance(result, list) and result:
        first = result[0]
        return first if isinstance(first, dict) else {}
    return {}


def fetch_peer_comparison(
    api_key: str | None,
    symbol: str,
    *,
    max_peers: int = 4,
) -> PeerComparison:
    peers_result = fetch_fmp_json(
        api_key,
        "stock-peers",
        stable=True,
        params={"symbol": symbol.upper()},
        suppress_error_messages=False,
    )
    if isinstance(peers_result, dict) and "Error Message" in peers_result:
        return PeerComparison(error=str(peers_result["Error Message"]))

    peer_list: list[str] = []
    if isinstance(peers_result, list) and peers_result:
        first = peers_result[0]
        if isinstance(first, dict) and isinstance(first.get("peersList"), list):
            peer_list = [str(item).upper() for item in first["peersList"]]
        else:
            peer_list = [str(item).upper() for item in peers_result]

    if not peer_list:
        return PeerComparison(error=f"Could not find peer data for {symbol.upper()}.")

    combined_symbols = list(dict.fromkeys([symbol.upper(), *peer_list[:max_peers]]))
    metrics_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for peer_symbol in combined_symbols:
        metrics_result = fetch_fmp_json(
            api_key,
            "key-metrics-ttm",
            stable=True,
            params={"symbol": peer_symbol},
            suppress_error_messages=False,
        )
        if isinstance(metrics_result, list) and metrics_result and isinstance(metrics_result[0], dict):
            row = dict(metrics_result[0])
            row["symbol"] = peer_symbol
            metrics_rows.append(row)

        ratios_result = fetch_fmp_json(
            api_key,
            "ratios-ttm",
            stable=True,
            params={"symbol": peer_symbol},
            suppress_error_messages=False,
        )
        if isinstance(ratios_result, list) and ratios_result and isinstance(ratios_result[0], dict):
            row = dict(ratios_result[0])
            row["symbol"] = peer_symbol
            ratio_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    ratios = pd.DataFrame(ratio_rows)
    if not metrics.empty:
        metrics = metrics.set_index("symbol")
    if not ratios.empty:
        ratios = ratios.set_index("symbol")

    return PeerComparison(peer_symbols=combined_symbols, metrics=metrics, ratios=ratios)


def format_compact_currency(value: float) -> str:
    amount = safe_num(value)
    if abs(amount) >= 1e12:
        return f"${amount / 1e12:.2f}T"
    if abs(amount) >= 1e9:
        return f"${amount / 1e9:.2f}B"
    if abs(amount) >= 1e6:
        return f"${amount / 1e6:.2f}M"
    return f"${amount:,.0f}"
