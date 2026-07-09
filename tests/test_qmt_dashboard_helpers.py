from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from oqp.ui.qmt_dashboard import (
    qmt_account_rows,
    qmt_audit_events_frame,
    qmt_exposure_by_asset,
    qmt_overall_status,
    qmt_route_candidate_frame,
    qmt_safety_gate_frame,
    qmt_status_frame,
    qmt_strategy_route_frame,
    qmt_submit_state,
)


def settings(**overrides):
    base = {
        "qmt_connector_enabled": True,
        "qmt_connector_url": "http://127.0.0.1:58668",
        "qmt_submit_connector_url": "http://127.0.0.1:58669",
        "qmt_api_token": "secret",
        "qmt_request_signing_secret": "signing-secret",
        "qmt_audit_log_path": Path("/tmp/qmt_audit.jsonl"),
        "qmt_require_private_connector": True,
        "qmt_account_type": "STOCK",
        "qmt_session_id": 880001,
        "qmt_timeout_seconds": 5.0,
        "allow_qmt_paper_order_submit": False,
        "allow_qmt_live_trading": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class QMTDashboardHelperTests(unittest.TestCase):
    def test_status_and_account_rows_filter_qmt_only(self) -> None:
        snapshot = SimpleNamespace(
            item_rows=[
                {"Category": "Broker Heartbeat", "Check": "QMT paper connector heartbeat", "Status": "pass", "Detail": "ok"},
                {"Category": "Safety", "Check": "QMT submit token configured", "Status": "fail", "Detail": "missing"},
                {"Category": "Broker Heartbeat", "Check": "IBKR paper", "Status": "pass", "Detail": "ok"},
            ],
            account_rows=[
                {
                    "environment": "paper",
                    "broker": "qmt",
                    "profile": "qmt_paper_readonly",
                    "account_id": "PAPER123",
                    "net_liquidation": 1_000_000,
                    "cash": 800_000,
                    "daily_pnl": 1000,
                    "position_count": 2,
                    "as_of": "2026-07-07T00:00:00Z",
                    "age_hours": 1.5,
                },
                {"environment": "paper", "broker": "ibkr", "profile": "ibkr_paper_readonly"},
            ],
        )

        status = qmt_status_frame(snapshot)
        accounts = qmt_account_rows(snapshot)

        self.assertEqual(len(status), 2)
        self.assertEqual(qmt_overall_status(snapshot), "fail")
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts.iloc[0]["Broker"], "qmt")

    def test_safety_gate_frame_reflects_submit_requirements(self) -> None:
        frame = qmt_safety_gate_frame(
            settings(
                allow_qmt_paper_order_submit=True,
                qmt_request_signing_secret=None,
                qmt_submit_connector_url="http://127.0.0.1:58668",
            )
        )

        failures = frame[frame["Status"].eq("fail")]

        self.assertEqual(qmt_submit_state(settings(allow_qmt_paper_order_submit=True)), "paper armed")
        self.assertIn("HMAC signing", set(failures["Gate"]))
        self.assertIn("Submit connector isolated", set(failures["Gate"]))

    def test_position_and_strategy_route_helpers_classify_qmt_lanes(self) -> None:
        positions = pd.DataFrame(
            [
                {"environment": "paper", "broker": "qmt", "profile": "qmt_paper_readonly", "symbol": "600000.SH", "asset_class": "equity", "market_value": 1000, "unrealized_pnl": 10},
                {"environment": "paper", "broker": "ibkr", "profile": "ibkr_paper_readonly", "symbol": "SPY", "asset_class": "etf", "market_value": 500, "unrealized_pnl": 1},
            ]
        )
        registry = pd.DataFrame(
            [
                {"strategy_id": "cn_demo", "market_vertical": "EQUITY_CN", "status": "paper_running", "allowed_symbols_json": json.dumps(["600000.SH"])},
                {"strategy_id": "us_demo", "market_vertical": "EQUITY_US", "status": "paper_running", "allowed_symbols_json": json.dumps(["SPY"])},
            ]
        )

        exposure = qmt_exposure_by_asset(positions)
        routes = qmt_strategy_route_frame(registry)
        candidate = qmt_route_candidate_frame("600000.SH", settings())

        self.assertEqual(len(exposure), 1)
        self.assertIn("candidate", set(routes["QMT Route"]))
        self.assertEqual(candidate.loc[candidate["Field"].eq("Likely Lane"), "Value"].iloc[0], "EQUITY_CN")

    def test_audit_events_parse_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qmt_audit.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "ts": "2026-07-07T01:00:00+00:00",
                        "event": "oqp_submit_order",
                        "endpoint": "/submit_order",
                        "status_code": 200,
                        "account_id": "PAPER123",
                        "request": {"symbol": "600000.SH"},
                        "response": {"message": "submitted"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            frame = qmt_audit_events_frame(settings(qmt_audit_log_path=path), limit=5)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["Symbol"], "600000.SH")
        self.assertEqual(frame.iloc[0]["Event"], "oqp_submit_order")


if __name__ == "__main__":
    unittest.main()
