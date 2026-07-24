from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oqp.journal import load_journal_entries, write_journal_entry


class JournalLedgerTest(unittest.TestCase):
    def test_write_and_load_journal_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "journal.db"

            result = write_journal_entry(
                db_path,
                entry_date="2026-06-30",
                category="trade_thesis",
                title="AAPL breakout thesis",
                body="Watching post-earnings drift.",
                environment="paper",
                symbols=["AAPL"],
                strategies=["manual_watch"],
                tags=["thesis", "options"],
                mistake=None,
                lesson="Wait for confirmation.",
                follow_up="Review after close.",
                metadata={"confidence": 0.6},
            )

            self.assertEqual(result.category, "trade_thesis")
            entries = load_journal_entries(db_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries.iloc[0]["title"], "AAPL breakout thesis")
            self.assertEqual(entries.iloc[0]["category"], "trade_thesis")
            self.assertIn("AAPL", entries.iloc[0]["symbols_json"])
            self.assertIn("confidence", entries.iloc[0]["metadata_json"])

    def test_filter_entries_by_date_and_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "journal.db"
            write_journal_entry(
                db_path,
                entry_date="2026-06-29",
                category="daily_note",
                title="Old note",
            )
            write_journal_entry(
                db_path,
                entry_date="2026-06-30",
                category="mistake",
                title="Chased signal",
            )

            filtered = load_journal_entries(
                db_path,
                category="mistake",
                entry_date="2026-06-30",
            )

            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered.iloc[0]["title"], "Chased signal")


if __name__ == "__main__":
    unittest.main()
