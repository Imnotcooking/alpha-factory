from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.config import load_settings  # noqa: E402


def load_readiness_module():
    module_path = REPO_ROOT / "scripts" / "check_ibkr_server_readiness.py"
    spec = importlib.util.spec_from_file_location("check_ibkr_server_readiness", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IBKRServerReadinessTests(unittest.TestCase):
    def test_builds_live_readonly_intended_config_from_env(self) -> None:
        module = load_readiness_module()

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "IBKR_HOST=127.0.0.1",
                        "IBKR_LIVE_PORT=4001",
                        "IBKR_LIVE_CLIENT_ID=777",
                        "IBKR_LIVE_MONITOR_ENABLED=true",
                        "ALLOW_LIVE_TRADING=false",
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_settings(env_file)

        config = module.intended_config(settings, "live")

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 4001)
        self.assertEqual(config.client_id, 777)
        self.assertTrue(config.readonly)
        self.assertEqual(config.metadata["profile"], "ibkr_live_readonly")

    def test_live_profile_gate_fails_when_disabled(self) -> None:
        module = load_readiness_module()

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("IBKR_LIVE_MONITOR_ENABLED=false\n", encoding="utf-8")
            settings = load_settings(env_file)

        check = module._profile_gate_check(settings, "live")

        self.assertEqual(check.status, "fail")
        self.assertIn("IBKR_LIVE_MONITOR_ENABLED=true", check.detail)

    def test_socket_check_passes_against_local_listener(self) -> None:
        module = load_readiness_module()

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        original_create_connection = module.socket.create_connection
        module.socket.create_connection = lambda *args, **kwargs: FakeSocket()
        try:
            check = module._socket_check("127.0.0.1", 4001, timeout=1.0)
        finally:
            module.socket.create_connection = original_create_connection

        self.assertEqual(check.status, "pass")

    def test_redacts_account_id(self) -> None:
        module = load_readiness_module()

        self.assertEqual(module._redact_account("DU1234567"), "DU***67")
        self.assertEqual(module._redact_account("123"), "***")
        self.assertEqual(module._redact_account(None), "n/a")


if __name__ == "__main__":
    unittest.main()
