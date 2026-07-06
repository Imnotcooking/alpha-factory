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
    shares_outstanding: float
    fair_value_per_share: float
    margin_of_safety: float

    @property
    def bridge_rows(self) -> list[dict[str, float | str]]:
        return [
            {"Valuation Metric": "PV of 10-Yr Cash Flows", "Amount": sum(self.present_value_fcf)},
            {"Valuation Metric": "PV of Terminal Value", "Amount": self.present_value_terminal},
            {"Valuation Metric": "Enterprise Value", "Amount": self.enterprise_value},
            {"Valuation Metric": "Equity Value", "Amount": self.equity_value},
            {"Valuation Metric": "Shares Outstanding", "Amount": self.shares_outstanding},
            {"Valuation Metric": "Intrinsic Value / Share", "Amount": self.fair_value_per_share},
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
    except (DataAdapterError, OSError, ValueError) as exc:
        if not suppress_error_messages:
            return {"Error Message": str(exc)}
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
    data["company_name"] = (
        profile_fmp[0].get("companyName") if isinstance(profile_fmp, list) and profile_fmp else info.get("longName")
    ) or symbol
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
        shares_outstanding=float(shares),
        fair_value_per_share=float(fair_value),
        margin_of_safety=float(margin_of_safety),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _usable_growth_percent(value: object, *, default: float, low: float = -50.0, high: float = 150.0) -> float:
    parsed = safe_num(value, np.nan)
    if pd.isna(parsed) or parsed < low or parsed > high:
        return default
    return float(parsed)


def estimate_dcf_assumptions(data: dict[str, Any]) -> dict[str, float | str]:
    """Produce transparent starter DCF assumptions from available fundamentals.

    These are intentionally conservative suggestions, not automated valuation truth.
    The dashboard keeps the final inputs editable so the user can override them.
    """

    terminal_growth = 2.5
    fcf_cagr = _usable_growth_percent(data.get("auto_fcf_cagr"), default=np.nan)
    rev_cagr = _usable_growth_percent(data.get("auto_rev_cagr"), default=np.nan)

    market_cap = safe_num(data.get("market_cap"))
    total_cash = safe_num(data.get("total_cash"))
    total_debt = safe_num(data.get("total_debt"))
    net_debt_ratio = (total_debt - total_cash) / market_cap if market_cap > 0 else 0.0
    roce = _clamp(safe_num(data.get("roce")), 0.0, 0.50)
    positive_quality_business = safe_num(data.get("fcf_ttm")) > 0 and roce >= 0.15

    if not pd.isna(fcf_cagr) and fcf_cagr >= 0 and not pd.isna(rev_cagr) and rev_cagr >= 0:
        growth_1 = (fcf_cagr * 0.65) + (rev_cagr * 0.35)
        growth_source = "FCF/revenue CAGR blend"
    elif not pd.isna(fcf_cagr) and fcf_cagr >= 0:
        growth_1 = fcf_cagr
        growth_source = "FCF CAGR"
    elif not pd.isna(rev_cagr) and rev_cagr >= 0:
        growth_1 = rev_cagr * 0.80
        growth_source = "revenue CAGR adjusted for margin/cash-flow conversion"
    elif not pd.isna(fcf_cagr):
        growth_1 = fcf_cagr
        growth_source = "negative FCF CAGR"
    elif not pd.isna(rev_cagr):
        growth_1 = rev_cagr
        growth_source = "negative revenue CAGR"
    else:
        growth_1 = 8.0
        growth_source = "default mature growth"

    if growth_1 < terminal_growth and positive_quality_business:
        growth_1 = terminal_growth
        growth_source = f"{growth_source}; floored for positive-FCF/high-ROCE business"

    growth_1 = _clamp(float(growth_1), -10.0, 40.0)
    growth_2 = _clamp(max(growth_1 * 0.60, terminal_growth if positive_quality_business else -10.0), -10.0, 18.0)

    # A simple listed-equity WACC starter: base cost of capital, leverage premium,
    # and a modest profitability offset. This is a prompt for review, not a CAPM engine.
    wacc = 9.5 + (_clamp(net_debt_ratio, -0.10, 0.60) * 4.0) - (roce * 2.0)
    wacc = _clamp(wacc, 6.5, 16.0)

    return {
        "wacc_pct": round(wacc, 1),
        "terminal_growth_pct": terminal_growth,
        "growth_1_pct": round(growth_1, 1),
        "growth_2_pct": round(growth_2, 1),
        "growth_source": growth_source,
        "raw_fcf_cagr_pct": round(float(fcf_cagr), 1) if not pd.isna(fcf_cagr) else "n/a",
        "raw_rev_cagr_pct": round(float(rev_cagr), 1) if not pd.isna(rev_cagr) else "n/a",
        "net_debt_to_market_cap": round(net_debt_ratio, 4),
        "roce": round(roce, 4),
    }


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_rows = (
            payload.get("data")
            or payload.get("results")
            or payload.get("items")
            or payload.get("documents")
            or payload.get("transcripts")
            or []
        )
        if isinstance(raw_rows, dict):
            raw_rows = [raw_rows]
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raw_rows = []
    return [row for row in raw_rows if isinstance(row, dict)]


def _document_text_preview(row: dict[str, Any]) -> str:
    text = _first_text(
        row,
        [
            "text",
            "content",
            "transcript",
            "finalLink",
            "acceptedDate",
            "description",
            "summary",
        ],
    )
    text = " ".join(text.split())
    return text[:500]


DCF_SOURCE_COLUMNS = ["Source", "Document", "Date", "Title", "URL", "Text Preview"]


def _payload_error(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("Error Message") or payload.get("message") or payload.get("error") or "")
    return ""


def _document_rows(
    payload: Any,
    *,
    source: str,
    normalized_symbol: str,
    seen: set[tuple[str, str, str]],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if limit <= 0:
        return rows
    for raw in _payload_rows(payload):
        document_type = _first_text(raw, ["type", "formType", "form", "period", "quarter"]) or "document"
        date = _first_text(raw, ["date", "fillingDate", "filingDate", "acceptedDate", "publishedDate"])
        title = _first_text(raw, ["title", "name", "symbol"]) or f"{normalized_symbol} {document_type}"
        url = _first_text(raw, ["finalLink", "link", "url", "reportLink", "filingUrl"])
        key = (document_type, date, url or title)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "Source": source,
                "Document": document_type,
                "Date": date,
                "Title": title,
                "URL": url,
                "Text Preview": _document_text_preview(raw),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _transcript_periods(payload: Any) -> list[tuple[int, int]]:
    periods: list[tuple[int, int]] = []
    if isinstance(payload, dict):
        raw_items = payload.get("data") or payload.get("results") or payload.get("items") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        year = int(safe_num(item.get("year"), 0))
        quarter = int(safe_num(item.get("quarter"), 0))
        if year > 0 and 1 <= quarter <= 4 and (year, quarter) not in periods:
            periods.append((year, quarter))
    return periods


def _access_issue_row(source: str, detail: str) -> dict[str, str]:
    return {
        "Source": source,
        "Document": "Access issue",
        "Date": "",
        "Title": detail,
        "URL": "",
        "Text Preview": "",
    }


def fetch_dcf_source_documents(
    api_key: str | None,
    symbol: str,
    *,
    limit: int = 8,
    fmp_fetcher: FMPFetcher = fetch_fmp_json,
) -> pd.DataFrame:
    """Fetch source material that can support DCF assumptions.

    The endpoint list is deliberately best-effort because FMP plan coverage differs.
    Empty responses are normal and should not break the dashboard.
    """

    normalized_symbol = symbol.upper().strip()
    if not api_key or not normalized_symbol:
        return pd.DataFrame(columns=DCF_SOURCE_COLUMNS)

    filing_requests = [
        (
            "FMP stable SEC filings",
            "sec-filings-search/symbol",
            True,
            {"symbol": normalized_symbol, "limit": limit},
        ),
        (
            "FMP 10-K filings",
            f"sec_filings/{normalized_symbol}",
            False,
            {"type": "10-k", "page": 0},
        ),
        (
            "FMP 10-Q filings",
            f"sec_filings/{normalized_symbol}",
            False,
            {"type": "10-q", "page": 0},
        ),
    ]

    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for source, endpoint, stable, params in filing_requests:
        payload = fmp_fetcher(
            api_key,
            endpoint,
            stable=stable,
            params=params,
            suppress_error_messages=False,
        )
        error = _payload_error(payload)
        if error:
            errors.append((source, error))
            continue
        rows.extend(
            _document_rows(
                payload,
                source=source,
                normalized_symbol=normalized_symbol,
                seen=seen,
                limit=limit - len(rows),
            )
        )
        if len(rows) >= limit:
            break

    transcript_dates_source = "FMP transcript dates"
    transcript_dates = fmp_fetcher(
        api_key,
        "earning-call-transcript-dates",
        stable=True,
        params={"symbol": normalized_symbol},
        suppress_error_messages=False,
    )
    transcript_error = _payload_error(transcript_dates)
    transcript_rows_before = len(rows)
    if transcript_error:
        errors.append((transcript_dates_source, transcript_error))
    else:
        periods = _transcript_periods(transcript_dates)
        if not periods:
            current_year = pd.Timestamp.now(tz="UTC").year
            periods = [
                (year, quarter)
                for year in (current_year, current_year - 1)
                for quarter in (4, 3, 2, 1)
            ]
        for year, quarter in periods[:4]:
            payload = fmp_fetcher(
                api_key,
                "earning-call-transcript",
                stable=True,
                params={"symbol": normalized_symbol, "year": year, "quarter": quarter},
                suppress_error_messages=False,
            )
            error = _payload_error(payload)
            if error:
                errors.append((f"FMP transcript {year}Q{quarter}", error))
                if "HTTP 402" in error or "Payment Required" in error:
                    break
                continue
            rows.extend(
                _document_rows(
                    payload,
                    source=f"FMP transcript {year}Q{quarter}",
                    normalized_symbol=normalized_symbol,
                    seen=seen,
                    limit=limit - len(rows),
                )
            )
            if len(rows) >= limit:
                break

    if errors and (not rows or len(rows) == transcript_rows_before):
        prioritized_errors = [error for error in errors if "transcript" in error[0].lower()] or errors
        rows.extend(_access_issue_row(source, detail) for source, detail in prioritized_errors[:3])

    return pd.DataFrame(rows, columns=DCF_SOURCE_COLUMNS)


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def build_dcf_assumption_evidence(
    data: dict[str, Any],
    documents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact evidence board for WACC and growth inputs."""

    hints = estimate_dcf_assumptions(data)
    def _hint_percent(value: object) -> str:
        if isinstance(value, str):
            return value
        return f"{safe_num(value):.1f}%"

    raw_fcf_cagr = _hint_percent(hints.get("raw_fcf_cagr_pct"))
    raw_rev_cagr = _hint_percent(hints.get("raw_rev_cagr_pct"))
    history_text = f"FCF CAGR {raw_fcf_cagr}; revenue CAGR {raw_rev_cagr}"
    doc_text = ""
    if isinstance(documents, pd.DataFrame) and not documents.empty:
        source_columns = [column for column in ("Title", "Text Preview") if column in documents]
        doc_text = " ".join(
            str(value)
            for value in documents[source_columns].fillna("").to_numpy().ravel().tolist()
            if str(value).strip()
        )

    growth_hits = _keyword_hits(
        doc_text,
        ["guidance", "growth", "demand", "backlog", "revenue", "margin", "capacity"],
    )
    risk_hits = _keyword_hits(
        doc_text,
        ["risk", "debt", "rates", "competition", "demand", "margin", "foreign exchange", "tariff"],
    )

    rows = [
        {
            "Input": "WACC",
            "Suggested": f"{safe_num(hints.get('wacc_pct')):.1f}%",
            "Primary Evidence": (
                f"net debt / market cap {safe_num(hints.get('net_debt_to_market_cap')):.1%}; "
                f"ROCE {safe_num(hints.get('roce')):.1%}"
            ),
            "Source Material": ", ".join(risk_hits[:5]) if risk_hits else "load filings/transcripts for risk language",
            "How To Use": "Raise for leverage, fragile cash flows, or rate sensitivity; lower for durable cash generation.",
        },
        {
            "Input": "Y1-5 Growth",
            "Suggested": f"{safe_num(hints.get('growth_1_pct')):.1f}%",
            "Primary Evidence": f"{hints.get('growth_source')} ({history_text})",
            "Source Material": ", ".join(growth_hits[:5]) if growth_hits else "load filings/transcripts for management guidance",
            "How To Use": "Use this for the explicit forecast period; override with concrete guidance when available.",
        },
        {
            "Input": "Y6-10 Growth",
            "Suggested": f"{safe_num(hints.get('growth_2_pct')):.1f}%",
            "Primary Evidence": "decay of Y1-5 growth toward maturity",
            "Source Material": ", ".join(growth_hits[:5]) if growth_hits else "load filings/transcripts for long-run runway",
            "How To Use": "Usually lower than Y1-5 unless the runway is unusually long and well-supported.",
        },
    ]
    return pd.DataFrame(rows)


def _first_number(row: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        value = safe_num(row.get(key), default=np.nan)
        if not pd.isna(value) and value > 0:
            return value
    return default


def _first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_price_target_rows(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_rows = payload.get("data") or payload.get("results") or payload.get("items") or []
        if isinstance(raw_rows, dict):
            raw_rows = [raw_rows]
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raw_rows = []

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        target = _first_number(
            raw,
            [
                "priceTarget",
                "targetPrice",
                "newPriceTarget",
                "analystTargetPrice",
                "target",
                "pt",
                "targetConsensus",
                "targetMeanPrice",
            ],
        )
        if target <= 0:
            continue
        rows.append(
            {
                "Published Date": _first_text(
                    raw,
                    [
                        "publishedDate",
                        "publishedDateTime",
                        "date",
                        "pricedDate",
                        "createdAt",
                        "updatedAt",
                    ],
                ),
                "Firm": _first_text(
                    raw,
                    [
                        "analystCompany",
                        "company",
                        "firm",
                        "brokerage",
                        "publishedBy",
                        "publisher",
                        "site",
                    ],
                ),
                "Analyst": _first_text(raw, ["analystName", "analyst", "author", "analyst_name"]),
                "Rating": _first_text(raw, ["rating", "newGrade", "grade", "recommendation", "action"]),
                "Target": float(target),
                "Source": source,
                "Title": _first_text(raw, ["newsTitle", "title", "headline"]),
                "URL": _first_text(raw, ["newsURL", "url", "link"]),
            }
        )
    return rows


def _normalize_peer_symbols(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        raw_items = payload.get("peersList") or payload.get("peers") or payload.get("data") or payload.get("results") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    if isinstance(raw_items, dict):
        raw_items = [raw_items]

    symbols: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            symbol = item
        elif isinstance(item, dict):
            nested = item.get("peersList") or item.get("peers")
            if isinstance(nested, list):
                symbols.extend(_normalize_peer_symbols(nested))
                continue
            symbol = _first_text(
                item,
                [
                    "symbol",
                    "ticker",
                    "peerSymbol",
                    "companySymbol",
                    "relatedSymbol",
                ],
            )
        else:
            symbol = ""
        symbol = symbol.upper().strip()
        if symbol and "{" not in symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _fetch_yahoo_price_targets(symbol: str, *, ticker_factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker

    try:
        ticker = ticker_factory(symbol.upper())
    except Exception:
        return {}

    result: dict[str, Any] = {}
    try:
        analyst_targets = getattr(ticker, "analyst_price_targets", None)
    except Exception:
        analyst_targets = None

    if isinstance(analyst_targets, pd.Series):
        analyst_targets = analyst_targets.to_dict()
    if isinstance(analyst_targets, dict):
        result["targetConsensus"] = _first_number(
            analyst_targets,
            ["mean", "targetMeanPrice", "targetConsensus", "average"],
        )
        result["targetHigh"] = _first_number(analyst_targets, ["high", "targetHighPrice", "targetHigh"])
        result["targetLow"] = _first_number(analyst_targets, ["low", "targetLowPrice", "targetLow"])
        result["targetMedian"] = _first_number(analyst_targets, ["median", "targetMedianPrice", "targetMedian"])
        analysts = _first_number(
            analyst_targets,
            ["numberOfAnalysts", "numberOfAnalystOpinions", "analystCount"],
        )
        if analysts:
            result["numberOfAnalystOpinions"] = int(analysts)
    elif isinstance(analyst_targets, pd.DataFrame):
        rows = _normalize_price_target_rows(analyst_targets.to_dict("records"), source="Yahoo Finance")
        if rows:
            result["targetRows"] = rows
            targets = pd.Series([row["Target"] for row in rows], dtype="float64")
            result["targetConsensus"] = float(targets.mean())
            result["targetHigh"] = float(targets.max())
            result["targetLow"] = float(targets.min())
            result["numberOfAnalystOpinions"] = int(len(rows))

    try:
        info = getattr(ticker, "info", {}) or {}
    except Exception:
        info = {}
    if isinstance(info, dict):
        result["targetConsensus"] = result.get("targetConsensus") or safe_num(info.get("targetMeanPrice"))
        result["targetHigh"] = result.get("targetHigh") or safe_num(info.get("targetHighPrice"))
        result["targetLow"] = result.get("targetLow") or safe_num(info.get("targetLowPrice"))
        result["targetMedian"] = result.get("targetMedian") or safe_num(info.get("targetMedianPrice"))
        analysts = result.get("numberOfAnalystOpinions") or safe_num(info.get("numberOfAnalystOpinions"))
        if analysts:
            result["numberOfAnalystOpinions"] = int(analysts)
        if info.get("recommendationKey"):
            result["recommendationKey"] = info.get("recommendationKey")
        if safe_num(info.get("recommendationMean")):
            result["recommendationMean"] = safe_num(info.get("recommendationMean"))

    if any(safe_num(result.get(key)) > 0 for key in ("targetConsensus", "targetHigh", "targetLow")):
        result["source"] = "Yahoo Finance"
    return result


def fetch_price_target_consensus(api_key: str | None, symbol: str) -> dict[str, Any]:
    normalized_symbol = symbol.upper().strip()
    result: dict[str, Any] = {}
    sources: list[str] = []

    consensus = fetch_fmp_json(
        api_key,
        "price-target-consensus",
        stable=True,
        params={"symbol": normalized_symbol},
        suppress_error_messages=False,
    )
    if isinstance(consensus, list) and consensus and isinstance(consensus[0], dict):
        result.update(consensus[0])
        sources.append("FMP")

    fmp_rows: list[dict[str, Any]] = []
    for endpoint in ("price-target-news", "price-target-summary"):
        payload = fetch_fmp_json(
            api_key,
            endpoint,
            stable=True,
            params={"symbol": normalized_symbol},
            suppress_error_messages=False,
        )
        fmp_rows.extend(_normalize_price_target_rows(payload, source="FMP"))
    if fmp_rows:
        result["targetRows"] = fmp_rows
        targets = pd.Series([row["Target"] for row in fmp_rows], dtype="float64")
        result["targetConsensus"] = result.get("targetConsensus") or float(targets.mean())
        result["targetHigh"] = result.get("targetHigh") or float(targets.max())
        result["targetLow"] = result.get("targetLow") or float(targets.min())
        result["numberOfAnalystOpinions"] = result.get("numberOfAnalystOpinions") or int(len(fmp_rows))
        if "FMP" not in sources:
            sources.append("FMP")

    yahoo = _fetch_yahoo_price_targets(normalized_symbol)
    if yahoo:
        for key in ("targetConsensus", "targetHigh", "targetLow", "targetMedian", "numberOfAnalystOpinions"):
            if not result.get(key) and yahoo.get(key):
                result[key] = yahoo[key]
        if not result.get("targetRows") and yahoo.get("targetRows"):
            result["targetRows"] = yahoo["targetRows"]
        for key in ("recommendationKey", "recommendationMean"):
            if yahoo.get(key):
                result[key] = yahoo[key]
        sources.append("Yahoo Finance")

    if sources:
        result["source"] = " + ".join(dict.fromkeys(sources))
    return result


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

    normalized_symbol = symbol.upper()
    peer_list = [peer for peer in _normalize_peer_symbols(peers_result) if peer != normalized_symbol]

    if not peer_list:
        return PeerComparison(error=f"Could not find peer data for {normalized_symbol}.")

    combined_symbols = list(dict.fromkeys([normalized_symbol, *peer_list[:max_peers]]))
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
