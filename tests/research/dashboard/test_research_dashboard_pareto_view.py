from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

from oqp.ui.research_dashboard_config import TEXT


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"

if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from views.pareto_view import ParetoView  # noqa: E402


class ResearchDashboardParetoViewTests(unittest.TestCase):
    def test_prepare_frontier_frame_uses_turnover_as_percent_axis(self) -> None:
        raw = pd.DataFrame(
            {
                "round_number": [1, 2],
                "Return (%)": [4.0, -1.5],
                "Drawdown (%)": [-10.0, -25.0],
                "turnover_rate": [0.12, 0.035],
                "holdout_ic": [0.04, -0.02],
            }
        )

        frame = ParetoView._prepare_frontier_frame(raw)

        self.assertEqual(frame["Drawdown Magnitude (%)"].tolist(), [10.0, 25.0])
        self.assertEqual(frame["Turnover (%)"].round(2).tolist(), [12.0, 3.5])
        self.assertEqual(frame["bubble_size"].tolist(), [0.04, 0.02])

    def test_pareto_3d_translation_keys_exist(self) -> None:
        expected = {
            "pareto_axis_drawdown",
            "pareto_axis_return",
            "pareto_axis_turnover",
            "pareto_color_ic",
            "pareto_size_abs_ic",
            "pareto_round",
        }

        self.assertTrue(expected.issubset(TEXT["EN"]))
        self.assertTrue(expected.issubset(TEXT["ZH"]))


if __name__ == "__main__":
    unittest.main()
