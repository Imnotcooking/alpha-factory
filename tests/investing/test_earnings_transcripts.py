from __future__ import annotations

import unittest

from oqp.investing import (
    build_management_keyword_frame,
    build_management_tone_frame,
    fetch_rapidapi_earnings_transcript_bundle,
    management_tone_summary,
)


class EarningsTranscriptTests(unittest.TestCase):
    def test_fetch_bundle_uses_latest_call_and_executive_segments(self) -> None:
        calls: list[tuple[str, dict | None]] = []

        def fetcher(path: str, params: dict | None = None):
            calls.append((path, params))
            if path == "/api/v1/companies/ticker/AAPL/latest":
                return {
                    "data": {
                        "id": 123,
                        "company_ticker": "AAPL",
                        "event_date_time": "2026-01-30T21:00:00Z",
                        "transcript_title": "Apple Q1 call",
                    }
                }
            if path == "/api/v1/transcripts/123/summary":
                return {"data": {"preview": "Management discussed strong demand and guidance."}}
            if path == "/api/v1/transcripts/123":
                return {
                    "data": {
                        "full_transcript_text": (
                            "We are confident in durable services growth. "
                            "Gross margin improved, though macro risk remains."
                        )
                    }
                }
            if path == "/api/v1/transcripts/123/components":
                return {
                    "data": {
                        "components": [
                            {
                                "speaker_name": "CEO",
                                "speaker_type": "executive",
                                "text": "Demand was strong and we are confident in growth.",
                            },
                            {
                                "speaker_name": "Analyst",
                                "speaker_type": "analyst",
                                "text": "Can you discuss demand?",
                            },
                        ]
                    }
                }
            if path == "/api/v1/speakers/123":
                self.assertEqual(params, {"speaker_type": "executive"})
                return {
                    "data": {
                        "segments": [
                            {
                                "speaker_name": "CEO",
                                "speaker_title": "Chief Executive Officer",
                                "text": "Demand was strong and we are confident in growth.",
                            },
                            {
                                "speaker_name": "CFO",
                                "speaker_title": "Chief Financial Officer",
                                "text": "Margins improved while we remain cautious on macro headwinds.",
                            },
                        ]
                    }
                }
            return {"data": {}}

        bundle = fetch_rapidapi_earnings_transcript_bundle("key", "aapl", fetcher=fetcher)

        self.assertEqual(bundle["status"], "ok")
        self.assertEqual(bundle["earnings_id"], 123)
        self.assertIn(("/api/v1/transcripts/123/components", None), calls)
        self.assertIn(("/api/v1/speakers/123", {"speaker_type": "executive"}), calls)

    def test_management_tone_outputs_dashboard_frames(self) -> None:
        bundle = {
            "status": "ok",
            "earnings_id": 123,
            "latest": {"event_date_time": "2026-01-30T21:00:00Z", "transcript_title": "Apple Q1 call"},
            "full": {"full_transcript_text": "Strong demand and improved margins, but risk from macro pressure."},
            "components": {
                "components": [
                    {"speaker_name": "CEO", "speaker_type": "executive", "text": "We are confident in strong demand and durable growth."},
                    {"speaker_name": "CFO", "speaker_type": "executive", "text": "Margin expansion improved despite macro headwinds."},
                    {"speaker_name": "Analyst", "speaker_type": "analyst", "text": "What about iPhone demand?"},
                ],
            },
        }

        tone = build_management_tone_frame(bundle)
        keywords = build_management_keyword_frame(bundle)
        summary = management_tone_summary(bundle, tone)

        self.assertEqual(set(tone["Lens"]), {"Management Overall", "Guidance / Outlook", "Margins / Profitability", "Risks / Headwinds"})
        self.assertFalse(keywords.empty)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["executive_segments"], 2)
        self.assertEqual(summary["tone_source"], "executive components")

    def test_missing_symbol_returns_error_status(self) -> None:
        bundle = fetch_rapidapi_earnings_transcript_bundle("key", "", fetcher=lambda *_args: {})

        self.assertEqual(bundle["status"], "error")


if __name__ == "__main__":
    unittest.main()
