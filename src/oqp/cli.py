"""Public command-line front door for Oxford Quant Pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from oqp.config.paths import resolve_repo_root
from oqp.demo.doctor import doctor_exit_code, run_doctor
from oqp.demo.profile import DEMO_PROFILE, demo_environment, read_profile_marker
from oqp.demo.seed import seed_demo_profile


DASHBOARDS = {
    "research": ("apps/research_dashboard/Homepage.py", 8524),
    "ops": ("apps/ops_dashboard/Homepage.py", 8529),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oqp",
        description="Oxford Quant Pipeline onboarding and local runtime commands.",
    )
    parser.add_argument("--repo-root", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize an isolated runtime profile.")
    init_parser.add_argument("--profile", choices=(DEMO_PROFILE,), default=DEMO_PROFILE)
    init_parser.add_argument("--force", action="store_true", help="Replace the selected runtime profile marker.")
    init_parser.add_argument("--as-of", help="Fixture end date in YYYY-MM-DD form.")

    doctor_parser = subparsers.add_parser("doctor", help="Check local installation and runtime readiness.")
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch a Streamlit dashboard.")
    dashboard_parser.add_argument("dashboard", choices=tuple(DASHBOARDS))
    dashboard_parser.add_argument("--port", type=int)
    dashboard_parser.add_argument("--address", default="127.0.0.1")
    dashboard_parser.add_argument("--no-browser", action="store_true")
    dashboard_parser.add_argument("--dry-run", action="store_true")

    test_parser = subparsers.add_parser("test", help="Run a documented verification lane.")
    test_parser.add_argument("suite", choices=("smoke",), default="smoke", nargs="?")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_repo_root(configured_root=args.repo_root)
    if args.command == "init":
        return _init_command(args, root)
    if args.command == "doctor":
        return _doctor_command(args, root)
    if args.command == "dashboard":
        return _dashboard_command(args, root)
    if args.command == "test":
        return _test_command(args, root)
    return 2


def dashboard_command(
    dashboard: str,
    *,
    repo_root: str | Path | None = None,
    port: int | None = None,
    address: str = "127.0.0.1",
    no_browser: bool = False,
) -> tuple[list[str], dict[str, str], Path]:
    root = resolve_repo_root(configured_root=repo_root)
    relative_app, default_port = DASHBOARDS[dashboard]
    selected_port = int(port or default_port)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / relative_app),
        "--server.port",
        str(selected_port),
        "--server.address",
        address,
        "--server.headless",
        "true" if no_browser else "false",
    ]
    marker = read_profile_marker(root)
    if marker.get("profile") == DEMO_PROFILE:
        env = demo_environment(root)
    else:
        env = dict(os.environ)
        env["OQP_REPO_ROOT"] = str(root)
        python_path = [str(root / "src"), str(root)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)
    return command, env, root


def _init_command(args: argparse.Namespace, root: Path) -> int:
    result = seed_demo_profile(root, as_of=args.as_of, force=args.force)
    print(f"Initialized OQP profile: {result.profile}")
    print(f"Runtime: {result.paths.runtime_root}")
    print(
        f"Fixtures: {result.research_runs} research runs, "
        f"{result.account_snapshots} account snapshots, "
        f"{result.option_contracts} option contracts"
    )
    print("Next: oqp doctor")
    return 0


def _doctor_command(args: argparse.Namespace, root: Path) -> int:
    checks = run_doctor(root)
    if args.as_json:
        print(json.dumps([check.to_dict() for check in checks], indent=2, sort_keys=True))
    else:
        width = max(len(check.name) for check in checks)
        labels = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
        for check in checks:
            print(f"[{labels[check.status]}] {check.name:<{width}}  {check.detail}")
    return doctor_exit_code(checks)


def _dashboard_command(args: argparse.Namespace, root: Path) -> int:
    command, env, cwd = dashboard_command(
        args.dashboard,
        repo_root=root,
        port=args.port,
        address=args.address,
        no_browser=args.no_browser,
    )
    port = command[command.index("--server.port") + 1]
    print(f"Launching {args.dashboard} dashboard at http://{args.address}:{port}")
    if env.get("OQP_PROFILE") == DEMO_PROFILE:
        print("Runtime profile: demo (isolated under runtime/demo)")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    except KeyboardInterrupt:
        return 130
    return int(completed.returncode)


def _test_command(args: argparse.Namespace, root: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/onboarding",
    ]
    env = dict(os.environ)
    env["OQP_REPO_ROOT"] = str(root)
    python_path = [str(root / "src"), str(root)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    print(shlex.join(command))
    return int(subprocess.run(command, cwd=root, env=env, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
