"""Operator controls for the local paid-call concurrency policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from suite_tools.paid_call_lease import (
    load_paid_call_lease_status,
    load_paid_call_policy,
    set_paid_call_policy,
)


def _snapshot(lease_dir: Path | None) -> dict:
    policy = load_paid_call_policy(lease_dir)
    status = load_paid_call_lease_status(lease_dir)
    return {
        **policy,
        "active_count": int(status.get("active_count") or 0),
        "waiting_count": int(status.get("waiting_count") or 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or change local paid-call capacity.")
    parser.add_argument("--lease-dir", type=Path, help="Override the shared lease directory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("get", help="Show the authoritative global limit and live usage.")
    set_parser = commands.add_parser("set", help="Set the authoritative global limit.")
    set_parser.add_argument("--global", dest="global_limit", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "set":
        try:
            set_paid_call_policy(args.global_limit, lease_dir=args.lease_dir)
        except ValueError as exc:
            build_parser().error(str(exc))
    snapshot = _snapshot(args.lease_dir)
    if args.json:
        print(json.dumps(snapshot, sort_keys=True))
    else:
        print(f"Global paid-call limit: {snapshot['global_limit']}")
        print(f"Active: {snapshot['active_count']}  Waiting: {snapshot['waiting_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
