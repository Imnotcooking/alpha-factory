#!/usr/bin/env python3
"""Check that public commits do not include private alpha or runtime state."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Rule:
    pattern: str
    reason: str


PRIVATE_RULES = (
    Rule("alpha_research_lab/factors/fac_*.py", "live alpha factor implementation"),
    Rule("alpha_research_lab/factors/factor_metadata_private*.py", "private factor metadata"),
    Rule("alpha_research_lab/factors/*private_metadata*.py", "private factor metadata"),
    Rule("alpha_research_lab/**/*candidate*", "research candidate artifact"),
    Rule("alpha_research_lab/**/*trial*", "research trial artifact"),
    Rule("alpha_research_lab/**/*promotion*", "research promotion artifact"),
    Rule("alpha_research_lab/data_cache/**", "cached research data"),
    Rule("alpha_research_lab/metadata/**", "private research metadata"),
    Rule("alpha_research_lab/data_engine/metadata/**", "private data-engine metadata"),
    Rule("alpha_research_lab/execution_logs/**", "execution or return logs"),
    Rule("alpha_research_lab/ui_v2/execution_logs/**", "dashboard execution logs"),
    Rule("alpha_research_lab/archive/**", "private alpha archive"),
    Rule("alpha_research_lab/ml_engine/*model*.json", "local model artifact"),
    Rule("alpha_research_lab/regime_engine/*.pkl", "trained regime artifact"),
    Rule("alpha_research_lab/run_fac_*.py", "active factor sweep script"),
    Rule("alpha_research_lab/*sweep*.py", "research sweep script"),
    Rule("alpha_research_lab/*backtest*.py", "research backtest script"),
    Rule("alpha_research_lab/*diagnostic*.py", "research diagnostic script"),
    Rule("alpha_research_lab/*probe*.py", "research probe script"),
    Rule("alpha_research_lab/ic_decay_*.png", "research performance image"),
    Rule("departments/archive/legacy_alpha_factory/**/factor_library/fac_*.py", "legacy live factor"),
    Rule("departments/archive/legacy_alpha_factory/**/models/**", "legacy model artifact"),
    Rule("departments/archive/legacy_alpha_factory/strategy_agents/agent_*.py", "legacy strategy agent"),
    Rule("runtime/**", "runtime state"),
    Rule("logs/**", "local logs"),
    Rule("data/**", "local data lake"),
    Rule("*.db", "SQLite ledger or database"),
    Rule("*.sqlite", "SQLite ledger or database"),
    Rule("*.sqlite3", "SQLite ledger or database"),
    Rule("*.duckdb", "DuckDB ledger or database"),
    Rule("*.pt", "model checkpoint"),
    Rule("*.pth", "model checkpoint"),
    Rule("*.pkl", "pickle/model artifact"),
    Rule("*.joblib", "model artifact"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if staged or dirty files include private/public-boundary paths.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all tracked and untracked dirty files instead of only staged files.",
    )
    return parser.parse_args()


def git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def candidate_paths(*, check_all: bool) -> list[str]:
    if check_all:
        tracked_dirty = git_paths("diff", "--name-only")
        staged = git_paths("diff", "--cached", "--name-only")
        untracked = git_paths("ls-files", "--others", "--exclude-standard")
        return sorted(set(tracked_dirty + staged + untracked))
    return sorted(set(git_paths("diff", "--cached", "--name-only")))


def match_rule(path: str) -> Rule | None:
    normalized = path.replace("\\", "/")
    for rule in PRIVATE_RULES:
        if fnmatch.fnmatch(normalized, rule.pattern):
            return rule
    return None


def main() -> int:
    args = parse_args()
    paths = candidate_paths(check_all=bool(args.all))
    violations = [(path, rule) for path in paths if (rule := match_rule(path))]

    scope = "dirty worktree" if args.all else "staged changes"
    if not violations:
        print(f"PASS public commit hygiene: no private paths in {scope}.")
        return 0

    print(f"FAIL public commit hygiene: private paths found in {scope}.", file=sys.stderr)
    for path, rule in violations:
        print(f"- {path}: {rule.reason}", file=sys.stderr)
    print(
        "\nUse explicit path staging and keep private alpha/runtime artifacts out of public commits.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
