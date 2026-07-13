"""Command-line entry point for the daily-regime research package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from oqp.research.daily_regimes.smoke import main as smoke_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "smoke":
        return smoke_main(arguments[1:])

    parser = argparse.ArgumentParser(prog="python -m oqp.research.daily_regimes")
    parser.add_argument("command", choices=("smoke",))
    parser.parse_args(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
