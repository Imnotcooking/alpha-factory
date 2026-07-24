from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.investing import (
    build_dcf_assumption_evidence,
    calculate_cagr,
    calculate_dcf_valuation,
    estimate_dcf_assumptions,
    fetch_dcf_source_documents,
    load_stock_watchlist,
    normalize_watchlist,
    save_stock_watchlist,
)


class InvestingStockValuationTests(unittest.TestCase):
    def test_watchlist_normalization_and_runtime_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchlist.json"
            save_stock_watchlist([" aapl ", "MSFT", "aapl", "", None], path)
            loaded = load_stock_watchlist(path)

        self.assertEqual(loaded, ["AAPL", "MSFT"])

    def test_normalize_watchlist_preserves_order(self) -> None:
        self.assertEqual(normalize_watchlist(["spy", "QQQ", "SPY"]), ["SPY", "QQQ"])

    def test_calculate_cagr_handles_bad_inputs(self) -> None:
        self.assertEqual(calculate_cagr(pd.Series([0, 10, 20])), 0.0)
        self.assertGreater(calculate_cagr(pd.Series([100, 121])), 0.20)

    def test_standard_dcf_valuation_bridge(self) -> None:
        result = calculate_dcf_valuation(
            {
                "fcf_ttm": 100.0,
                "total_cash": 20.0,
                "total_debt": 10.0,
                "shares": 10.0,
                "price": 12.0,
            },
            model="standard",
            wacc=0.10,
            terminal_growth=0.025,
            fcf_growth_1=0.05,
            fcf_growth_2=0.03,
        )

        self.assertEqual(len(result.future_fcf), 10)
        self.assertGreater(result.fair_value_per_share, 0.0)
        bridge = {row["Valuation Metric"]: row["Amount"] for row in result.bridge_rows}
        self.assertEqual(bridge["Shares Outstanding"], 10.0)
        self.assertAlmostEqual(bridge["Intrinsic Value / Share"], result.fair_value_per_share)

    def test_dcf_requires_wacc_above_terminal_growth(self) -> None:
        with self.assertRaises(ValueError):
            calculate_dcf_valuation(
                {"fcf_ttm": 100.0, "shares": 10.0},
                model="standard",
                wacc=0.02,
                terminal_growth=0.025,
            )

    def test_estimate_dcf_assumptions_uses_fundamental_history(self) -> None:
        hints = estimate_dcf_assumptions(
            {
                "auto_fcf_cagr": 18.2,
                "auto_rev_cagr": 11.5,
                "market_cap": 1_000.0,
                "total_cash": 200.0,
                "total_debt": 100.0,
                "roce": 0.25,
            }
        )

        self.assertEqual(hints["growth_source"], "FCF/revenue CAGR blend")
        self.assertGreater(float(hints["growth_1_pct"]), float(hints["growth_2_pct"]))
        self.assertGreater(float(hints["wacc_pct"]), 0.0)
        self.assertEqual(hints["raw_fcf_cagr_pct"], 18.2)

    def test_estimate_dcf_assumptions_does_not_blindly_project_negative_fcf_cagr(self) -> None:
        hints = estimate_dcf_assumptions(
            {
                "auto_fcf_cagr": -3.9,
                "auto_rev_cagr": -1.0,
                "fcf_ttm": 100.0,
                "market_cap": 1_000.0,
                "total_cash": 100.0,
                "total_debt": 120.0,
                "roce": 0.50,
            }
        )

        self.assertGreaterEqual(float(hints["growth_1_pct"]), float(hints["terminal_growth_pct"]))
        self.assertIn("floored", str(hints["growth_source"]))

    def test_dcf_source_documents_normalize_best_effort_payloads(self) -> None:
        def fake_fetcher(api_key, endpoint, **kwargs):
            self.assertEqual(api_key, "key")
            if endpoint == "sec-filings-search/symbol":
                return [
                    {
                        "formType": "10-K",
                        "filingDate": "2026-02-01",
                        "title": "AAPL annual report",
                        "finalLink": "https://example.com/10k",
                    }
                ]
            return []

        docs = fetch_dcf_source_documents("key", "AAPL", fmp_fetcher=fake_fetcher)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs.iloc[0]["Document"], "10-K")
        self.assertEqual(docs.iloc[0]["URL"], "https://example.com/10k")

    def test_dcf_source_documents_fetches_transcript_by_year_and_quarter(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_fetcher(api_key, endpoint, **kwargs):
            params = kwargs.get("params") or {}
            calls.append((endpoint, params))
            if endpoint == "earning-call-transcript-dates":
                return [{"year": 2025, "quarter": 3}]
            if endpoint == "earning-call-transcript":
                self.assertEqual(params["year"], 2025)
                self.assertEqual(params["quarter"], 3)
                return [
                    {
                        "symbol": "AAPL",
                        "quarter": 3,
                        "year": 2025,
                        "date": "2025-08-01",
                        "content": "Management discussed services revenue growth and margin.",
                    }
                ]
            return []

        docs = fetch_dcf_source_documents("key", "AAPL", fmp_fetcher=fake_fetcher)

        self.assertIn(("earning-call-transcript-dates", {"symbol": "AAPL"}), calls)
        self.assertTrue((docs["Source"] == "FMP transcript 2025Q3").any())

    def test_dcf_source_documents_surfaces_access_issues(self) -> None:
        def fake_fetcher(api_key, endpoint, **kwargs):
            return {"Error Message": f"FMP request failed for endpoint: {endpoint} HTTP 402"}

        docs = fetch_dcf_source_documents("key", "AAPL", fmp_fetcher=fake_fetcher)

        self.assertFalse(docs.empty)
        self.assertTrue((docs["Document"] == "Access issue").all())

    def test_dcf_assumption_evidence_includes_core_inputs(self) -> None:
        evidence = build_dcf_assumption_evidence(
            {
                "auto_fcf_cagr": 14.0,
                "market_cap": 1_000.0,
                "total_cash": 50.0,
                "total_debt": 150.0,
                "roce": 0.12,
            },
            pd.DataFrame(
                [
                    {
                        "Title": "Management guidance discusses revenue growth and demand",
                        "Text Preview": "Debt and rates remain a risk.",
                    }
                ]
            ),
        )

        self.assertEqual(set(evidence["Input"]), {"WACC", "Y1-5 Growth", "Y6-10 Growth"})
        self.assertTrue(evidence["Suggested"].str.endswith("%").all())


if __name__ == "__main__":
    unittest.main()
