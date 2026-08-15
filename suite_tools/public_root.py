"""Create a checksum-free public Git root from an audited source export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from suite_tools.source_release import SourceReleaseError, seed_public_root_from_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a new public Git root from a verified source export."
    )
    parser.add_argument("--export-root", default=".")
    parser.add_argument("--out", required=True, help="Absent or empty output directory.")
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = seed_public_root_from_export(
            export_root=Path(args.export_root),
            out_dir=Path(args.out),
            release_version=args.release_version,
        )
    except SourceReleaseError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"Public root blocked: {exc}")
        return 1

    if args.json:
        print(json.dumps({"ok": True, **receipt}, indent=2, sort_keys=True))
    else:
        print(f"Seeded {receipt['file_count']} audited files into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
