from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.config import load_settings  # noqa: E402
from oqp.data import get_options_adapter  # noqa: E402


class DataVendorAliasTests(unittest.TestCase):
    def test_massive_key_feeds_legacy_polygon_snapshot_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MASSIVE_API_KEY=massive-demo-key\n", encoding="utf-8")

            settings = load_settings(env_file)
            canonical = get_options_adapter("massive", settings=settings)
            legacy_snapshot = get_options_adapter("polygon", settings=settings)

        self.assertEqual(settings.polygon_api_key, "massive-demo-key")
        self.assertTrue(canonical.healthcheck().ok)
        self.assertTrue(legacy_snapshot.healthcheck().ok)
        self.assertEqual(legacy_snapshot.healthcheck().name, "massive_options_snapshot")


if __name__ == "__main__":
    unittest.main()
