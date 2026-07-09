from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

from oqp.ui.research_dashboard_config import TEXT


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"

if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from views.dna_view import DNAView  # noqa: E402


class ResearchDashboardDNAViewTests(unittest.TestCase):
    def test_trade_stats_include_payoff_and_expectancy_metrics(self) -> None:
        trades = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "trade_pnl": [0.04, 0.02, -0.01, -0.03],
                "holding_period_hours": [24, 48, 72, 96],
            }
        )

        stats = DNAView._trade_stats(trades)

        self.assertEqual(stats["total_trades"], 4)
        self.assertAlmostEqual(stats["win_rate"], 0.5)
        self.assertAlmostEqual(stats["profit_factor"], 1.5)
        self.assertAlmostEqual(stats["payoff_ratio"], 1.5)
        self.assertAlmostEqual(stats["avg_win"], 0.03)
        self.assertAlmostEqual(stats["avg_loss"], -0.02)
        self.assertAlmostEqual(stats["expectancy"], 0.005)
        self.assertAlmostEqual(stats["median_hold"], 60.0)
        self.assertEqual(stats["profit_concentration_count"], 2)
        self.assertEqual(stats["profit_concentration_total"], 2)
        self.assertAlmostEqual(stats["profit_concentration_share"], 1.0)

    def test_dna_metric_labels_are_translated(self) -> None:
        expected_keys = {
            "dna_total_trades",
            "dna_win_rate",
            "dna_profit_factor",
            "dna_median_hold",
            "dna_payoff_ratio",
            "dna_avg_win",
            "dna_avg_loss",
            "dna_profit_concentration",
        }

        self.assertTrue(expected_keys.issubset(TEXT["EN"]))
        self.assertTrue(expected_keys.issubset(TEXT["ZH"]))
        self.assertEqual(TEXT["EN"]["dna_payoff_ratio"], "Payoff Ratio")
        self.assertEqual(TEXT["ZH"]["dna_payoff_ratio"], "盈亏比")
        self.assertEqual(TEXT["EN"]["dna_profit_concentration"], "80% Profit Tickers")
        self.assertEqual(TEXT["ZH"]["dna_profit_concentration"], "80%盈利标的")

    def test_profit_concentration_counts_tickers_for_eighty_percent_of_profit(self) -> None:
        trades = pd.DataFrame(
            {
                "ticker": ["A", "A", "B", "C", "D", "E"],
                "trade_pnl": [0.30, 0.20, 0.30, 0.20, -0.40, 0.00],
            }
        )

        concentration = DNAView._profit_concentration(trades)

        self.assertEqual(concentration["profit_concentration_count"], 2)
        self.assertEqual(concentration["profit_concentration_total"], 3)
        self.assertAlmostEqual(concentration["profit_concentration_share"], 0.80)

    def test_asset_winner_loser_frame_uses_top_five_each_side_and_names(self) -> None:
        tickers = [f"W{i}" for i in range(6)] + [f"L{i}" for i in range(6)]
        pnls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06]
        trades = pd.DataFrame(
            {
                "ticker": tickers + ["W5", "L5"],
                "trade_pnl": pnls + [0.01, -0.01],
                "name_zh": [""] * len(tickers) + ["最强赢家", ""],
            }
        )

        original_lookup = DNAView._cn_equity_name_lookup
        DNAView._cn_equity_name_lookup = staticmethod(lambda _base_dir: {"L5": "最大亏损股"})
        try:
            frame = DNAView(data_manager=None)._asset_winner_loser_frame(trades)
        finally:
            DNAView._cn_equity_name_lookup = original_lookup

        self.assertEqual(len(frame), 10)
        self.assertNotIn("W0", set(frame["ticker"]))
        self.assertNotIn("L0", set(frame["ticker"]))
        self.assertEqual(set(frame["side"]), {"Winner", "Loser"})
        self.assertEqual(frame.loc[frame["ticker"].eq("W5"), "company_name"].iloc[0], "最强赢家")
        self.assertEqual(frame.loc[frame["ticker"].eq("L5"), "company_name"].iloc[0], "最大亏损股")
        self.assertEqual(int(frame.loc[frame["ticker"].eq("W5"), "trade_count"].iloc[0]), 2)

    def test_trade_pnl_distribution_tracks_flat_trades_and_tails_separately(self) -> None:
        trades = pd.DataFrame(
            {
                "trade_pnl_pct": [
                    -50.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.5,
                    1.0,
                    2.0,
                    3.0,
                    80.0,
                ]
            }
        )

        frame, summary = DNAView._trade_pnl_distribution_frame(
            trades,
            bins=6,
            lower_q=0.10,
            upper_q=0.90,
        )

        self.assertEqual(summary["flat"], 2)
        self.assertGreater(summary["left_tail"], 0)
        self.assertGreater(summary["right_tail"], 0)
        self.assertFalse(frame.empty)
        self.assertAlmostEqual(float(frame["share_pct"].sum()), 100.0)
        self.assertTrue({"range_label", "count", "avg_pnl", "sum_pnl"}.issubset(frame.columns))

    def test_holding_pain_frame_adds_names_labels_and_hover_fields(self) -> None:
        trades = pd.DataFrame(
            {
                "ticker": ["SSE.600000", "SSE.600001", "SSE.600002"],
                "trade_pnl_pct": [12.0, -4.0, 0.0],
                "holding_period_hours": [24.5, 96.0, 12.0],
                "direction": ["Long", "short", ""],
                "entry_time": ["2026-01-02 09:30:00", "2026-01-03", None],
                "exit_time": ["2026-01-05 14:55:00", "2026-01-06", None],
                "entry_price": [10.25, 8.0, None],
                "exit_price": [11.5, 7.5, None],
                "name_zh": ["", "本地名称", ""],
            }
        )

        original_lookup = DNAView._cn_equity_name_lookup
        DNAView._cn_equity_name_lookup = staticmethod(lambda _base_dir: {"SSE.600000": "浦发银行"})
        try:
            frame = DNAView(data_manager=None)._holding_pain_frame(trades)
        finally:
            DNAView._cn_equity_name_lookup = original_lookup

        self.assertEqual(frame.loc[frame["ticker"].eq("SSE.600000"), "company_name"].iloc[0], "浦发银行")
        self.assertEqual(frame.loc[frame["ticker"].eq("SSE.600001"), "company_name"].iloc[0], "本地名称")
        self.assertEqual(list(frame["side_key"]), ["Long", "Short", "Unknown"])
        self.assertEqual(list(frame["result_key"]), ["Win", "Loss", "Flat"])
        self.assertIn("2026-01-02 09:30 @ 10.25", frame["entry_label"].iloc[0])
        self.assertIn("2026-01-05 14:55 @ 11.50", frame["exit_label"].iloc[0])
        self.assertGreaterEqual(float(frame["marker_size"].min()), 5.0)
        self.assertLessEqual(float(frame["marker_size"].max()), 18.0)

    def test_edge_stability_frame_groups_entry_period_and_holding_bucket(self) -> None:
        trades = pd.DataFrame(
            {
                "entry_time": [
                    "2026-01-05",
                    "2026-01-09",
                    "2026-02-01",
                    "2026-02-15",
                    None,
                ],
                "exit_time": [
                    "2026-01-06",
                    "2026-01-12",
                    "2026-02-10",
                    "2026-03-05",
                    "2026-03-20",
                ],
                "holding_period_hours": [12, 60, 200, 400, 20],
                "trade_pnl_pct": [2.0, -1.0, 3.0, -2.0, 4.0],
            }
        )

        frame, meta = DNAView._edge_stability_frame(trades)

        self.assertEqual(meta["period_level"], "month")
        self.assertEqual(meta["periods"], ["2026-01", "2026-02", "2026-03"])
        self.assertEqual(meta["buckets"], ["<1d", "1-3d", "3-7d", "7-14d", ">14d"])

        jan_fast = frame[
            frame["entry_period"].eq("2026-01") & frame["hold_bucket"].eq("<1d")
        ].iloc[0]
        self.assertEqual(int(jan_fast["trade_count"]), 1)
        self.assertAlmostEqual(float(jan_fast["avg_pnl"]), 2.0)
        self.assertAlmostEqual(float(jan_fast["win_rate"]), 100.0)

        feb_long = frame[
            frame["entry_period"].eq("2026-02") & frame["hold_bucket"].eq(">14d")
        ].iloc[0]
        self.assertEqual(int(feb_long["trade_count"]), 1)
        self.assertAlmostEqual(float(feb_long["sum_pnl"]), -2.0)
        self.assertAlmostEqual(float(feb_long["win_rate"]), 0.0)

        fallback_exit = frame[
            frame["entry_period"].eq("2026-03") & frame["hold_bucket"].eq("<1d")
        ].iloc[0]
        self.assertAlmostEqual(float(fallback_exit["avg_pnl"]), 4.0)


if __name__ == "__main__":
    unittest.main()
