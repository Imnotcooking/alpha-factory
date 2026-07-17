from __future__ import annotations

from pathlib import Path

from oqp.cli import DASHBOARDS, build_parser, dashboard_command
from oqp.demo.seed import seed_demo_profile


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "oqp").mkdir(parents=True)
    (root / "apps" / "research_dashboard").mkdir(parents=True)
    (root / "apps" / "ops_dashboard").mkdir(parents=True)
    (root / "apps" / "research_dashboard" / "Homepage.py").touch()
    (root / "apps" / "ops_dashboard" / "Homepage.py").touch()
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return root


def test_parser_exposes_public_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["init", "--profile", "demo"]).command == "init"
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["dashboard", "research"]).dashboard == "research"
    assert parser.parse_args(["test", "smoke"]).suite == "smoke"


def test_dashboard_command_uses_documented_ports_and_demo_paths(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    seed_demo_profile(root, as_of="2026-07-17")

    research, research_env, cwd = dashboard_command("research", repo_root=root)
    ops, ops_env, _ = dashboard_command("ops", repo_root=root)

    assert DASHBOARDS["research"][1] == 8524
    assert DASHBOARDS["ops"][1] == 8529
    assert research[research.index("--server.port") + 1] == "8524"
    assert ops[ops.index("--server.port") + 1] == "8529"
    assert research_env["ALPHA_RESEARCH_DB_PATH"].startswith(str(root / "runtime" / "demo"))
    assert ops_env["OQP_ACCOUNT_LEDGER_PATH"].startswith(str(root / "runtime" / "demo"))
    assert cwd == root
