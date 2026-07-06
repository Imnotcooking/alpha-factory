from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.config import load_settings
from oqp.investing import (
    analyze_company_outlook,
    build_company_outlook_packet,
    ensure_llm_evidence_schema,
)


class LLMEvidenceTests(unittest.TestCase):
    def test_zai_settings_default_when_key_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("ZAI_API_KEY=test-key\n", encoding="utf-8")

            settings = load_settings(env_file)

        self.assertEqual(settings.llm_provider, "zai")
        self.assertEqual(settings.llm_model, "glm-5.2")
        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertTrue(settings.llm_evidence_enabled)

    def test_packet_prefers_executive_components(self) -> None:
        packet = build_company_outlook_packet(
            symbol="aapl",
            transcript_bundle={
                "status": "ok",
                "earnings_id": 123,
                "latest": {"event_date_time": "2026-01-30T21:00:00Z"},
                "components": {
                    "components": [
                        {
                            "speaker_name": "CEO",
                            "speaker_type": "executive",
                            "text": "We see durable growth and strong services demand.",
                        },
                        {
                            "speaker_name": "Analyst",
                            "speaker_type": "analyst",
                            "text": "What about China?",
                        },
                    ]
                },
                "full": {"full_transcript_text": "Full transcript fallback."},
            },
            news_articles=pd.DataFrame(
                [{"Published": "2026-01-31", "Title": "Apple raises guidance", "Text": "Analysts cite demand."}]
            ),
            valuation_data={"price": 200, "sector": "Technology"},
        )

        self.assertEqual(packet["symbol"], "AAPL")
        self.assertEqual(len(packet["management_segments"]), 1)
        self.assertEqual(packet["transcript_excerpt"], "")
        self.assertEqual(packet["recent_news"][0]["title"], "Apple raises guidance")

    def test_analyze_company_outlook_caches_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "llm.db"
            ensure_llm_evidence_schema(db_path)
            calls = 0

            def fake_client(_messages, _packet):
                nonlocal calls
                calls += 1
                return {
                    "executive_summary": "Management sounded constructive.",
                    "short_term_outlook": {
                        "horizon": "0-3 months",
                        "label": "bullish",
                        "confidence": 0.7,
                        "summary": "Near-term demand language was strong.",
                        "evidence": ["CEO cited demand."],
                    },
                    "mid_term_outlook": {
                        "horizon": "3-18 months",
                        "label": "neutral",
                        "confidence": 0.5,
                        "summary": "Margins need confirmation.",
                        "evidence": [],
                    },
                    "long_term_outlook": {
                        "horizon": "18+ months",
                        "label": "bullish",
                        "confidence": 0.6,
                        "summary": "Durability remains plausible.",
                        "evidence": [],
                    },
                    "dcf_assumption_clues": {},
                    "options_implications": {},
                    "key_risks": [],
                    "source_limits": [],
                }

            kwargs = dict(
                symbol="AAPL",
                api_key="key",
                provider="zai",
                base_url="https://api.z.ai/api/paas/v4",
                model="glm-5.2",
                transcript_bundle={"status": "ok", "full": {"full_transcript_text": "Strong demand."}},
                news_articles=pd.DataFrame(),
                valuation_data={},
                path=db_path,
                chat_client=fake_client,
            )
            first = analyze_company_outlook(**kwargs)
            second = analyze_company_outlook(**kwargs)

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "cached")
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
