from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.investing import (
    analyze_news_articles,
    build_news_catalyst_board,
    build_news_evidence_frame,
    catalyst_evidence_text,
    fetch_fmp_news_articles,
    load_cached_news_articles,
    news_cache_status,
    nlp_provider_status,
    score_text_sentiment,
    write_news_articles,
)


class NewsNLPTests(unittest.TestCase):
    def sample_payload(self) -> list[dict[str, str]]:
        return [
            {
                "publishedDate": "2026-06-29 13:00:00",
                "title": "AAPL raises guidance after strong demand",
                "site": "Example News",
                "url": "https://example.com/aapl-strong",
                "text": "Analysts say demand improved and margin growth may accelerate.",
            },
            {
                "publishedDate": "2026-06-28 10:00:00",
                "title": "AAPL faces regulatory probe risk",
                "site": "Example News",
                "url": "https://example.com/aapl-risk",
                "text": "The company warned of pressure from a regulatory investigation.",
            },
        ]

    def test_fetch_fmp_news_articles_normalizes_v3_payload(self) -> None:
        calls: list[tuple[str, bool, dict[str, object]]] = []

        def fetcher(api_key: str, endpoint: str, *, stable: bool, params: dict[str, object], **_: object):
            calls.append((endpoint, stable, params))
            return self.sample_payload()

        frame = fetch_fmp_news_articles("key", "aapl", fmp_fetcher=fetcher)

        self.assertEqual(calls[0][0], "stock_news")
        self.assertEqual(frame.iloc[0]["symbol"], "AAPL")
        self.assertIn("raises guidance", frame.iloc[0]["title"])

    def test_cache_round_trip_and_status(self) -> None:
        frame = fetch_fmp_news_articles("key", "AAPL", fmp_fetcher=lambda *_args, **_kwargs: self.sample_payload())
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "news.sqlite3"
            written = write_news_articles("AAPL", frame, path=db_path)
            loaded = load_cached_news_articles("AAPL", path=db_path)
            status = news_cache_status("AAPL", path=db_path)

        self.assertEqual(written, 2)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(status["state"], "fresh")

    def test_analyze_articles_scores_keywords_and_topics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "published_at": "2026-06-29",
                    "title": "AAPL upgrade as earnings beat expectations",
                    "site": "Example",
                    "url": "https://example.com/beat",
                    "text": "The analyst raised the target after strong revenue growth.",
                    "source": "FMP",
                }
            ]
        )
        result = analyze_news_articles("AAPL", frame)

        self.assertEqual(result.summary["article_count"], 1)
        self.assertGreater(result.summary["sentiment_score"], 0)
        self.assertFalse(result.keyword_frame.empty)
        self.assertIn("Analyst Action", set(result.topic_frame["Topic"]))

    def test_catalyst_board_groups_article_topics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Published": "2026-06-29",
                    "Title": "AAPL upgrade after earnings beat",
                    "Site": "Example",
                    "Sentiment": "Positive",
                    "Score": 0.55,
                    "Topics": "Earnings, Analyst Action",
                    "Bullish Terms": "upgrade, beats expectations",
                    "Bearish Terms": "",
                    "URL": "https://example.com/upgrade",
                },
                {
                    "Published": "2026-06-28",
                    "Title": "AAPL regulatory probe risk",
                    "Site": "Example",
                    "Sentiment": "Negative",
                    "Score": -0.45,
                    "Topics": "Regulatory / Legal",
                    "Bullish Terms": "",
                    "Bearish Terms": "probe, risk",
                    "URL": "https://example.com/probe",
                },
            ]
        )

        catalyst = build_news_catalyst_board(frame)
        evidence = build_news_evidence_frame(frame)
        text = catalyst_evidence_text(catalyst)

        self.assertIn("Earnings", set(catalyst["Catalyst"]))
        self.assertIn("Regulatory / Legal", set(catalyst["Catalyst"]))
        self.assertEqual(evidence.iloc[0]["Catalyst"], "Earnings, Analyst Action")
        self.assertIn("articles", text)

    def test_score_text_sentiment_handles_bearish_text(self) -> None:
        score, bullish, bearish = score_text_sentiment("Shares fall after weak guidance and downgrade warning")

        self.assertLess(score, 0)
        self.assertFalse(bullish)
        self.assertTrue(bearish)

    def test_provider_status_contract(self) -> None:
        frame = nlp_provider_status(
            fmp_key="fmp",
            openai_key=None,
            x_bearer_token="x-token",
            reddit_client_id=None,
        )

        self.assertEqual(list(frame.columns), ["Provider", "Role", "Status", "Detail"])
        statuses = dict(zip(frame["Provider"], frame["Status"], strict=False))
        self.assertEqual(statuses["Local Lexicon"], "ready")
        self.assertEqual(statuses["FMP News/Articles"], "configured")
        self.assertEqual(statuses["OpenAI Extractor"], "optional")
        self.assertEqual(statuses["X / Twitter"], "configured")
        self.assertEqual(statuses["Reddit"], "not configured")


if __name__ == "__main__":
    unittest.main()
