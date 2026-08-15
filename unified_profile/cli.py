"""CLI for unified cross-module benchmark profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unified_profile.adapters import load_aita_results, load_epis_results, load_sus_results
from unified_profile.export import export_bundle
from unified_profile.profile import build_all_profiles
from unified_profile.report import generate_unified_report


def _existing_paths(values: list[str] | None, flag: str) -> list[Path]:
    paths = [Path(value) for value in values or []]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{flag} path does not exist: {', '.join(missing)}")
    return paths


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sus-dir", action="append", default=[], help="SUS result file or directory. Repeatable.")
    parser.add_argument("--aita-path", action="append", default=[], help="AITA result file or directory. Repeatable.")
    parser.add_argument("--epis-dir", action="append", default=[], help="Epistemic result file or directory. Repeatable.")


def _build_report_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Generate a unified sycophancy profile report.")
    _add_input_args(parser)
    parser.add_argument("--output", required=True, help="Output directory for REPORT.md.")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified sycophancy profile tools.")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Generate a unified markdown report.")
    _add_input_args(report_parser)
    report_parser.add_argument("--output", required=True, help="Output directory for REPORT.md.")

    export_parser = subparsers.add_parser("export", help="Generate a traceable artifact bundle.")
    _add_input_args(export_parser)
    export_parser.add_argument("--output", required=True, help="Output directory for the export bundle.")
    export_parser.add_argument(
        "--include-conversations",
        action="store_true",
        help="Copy raw conversation JSONs into the bundle. Defaults to scores/reports only.",
    )

    return parser


def _run_report(args: argparse.Namespace) -> int:
    sus_paths = _existing_paths(args.sus_dir, "--sus-dir")
    aita_paths = _existing_paths(args.aita_path, "--aita-path")
    epis_paths = _existing_paths(args.epis_dir, "--epis-dir")

    sus_data = load_sus_results(sus_paths) if sus_paths else {}
    aita_data = load_aita_results(aita_paths) if aita_paths else {}
    epis_data = load_epis_results(epis_paths) if epis_paths else {}

    for label, data, paths in [
        ("SUS", sus_data, sus_paths),
        ("AITA", aita_data, aita_paths),
        ("Epistemic", epis_data, epis_paths),
    ]:
        if paths and not data:
            print(f"Warning: {label} paths existed but produced no supported data.")

    profiles = build_all_profiles(sus_data, aita_data, epis_data)
    generate_unified_report(profiles, Path(args.output))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    sus_paths = _existing_paths(args.sus_dir, "--sus-dir")
    aita_paths = _existing_paths(args.aita_path, "--aita-path")
    epis_paths = _existing_paths(args.epis_dir, "--epis-dir")
    payload = export_bundle(
        sus_paths=sus_paths,
        aita_paths=aita_paths,
        epis_paths=epis_paths,
        output_dir=Path(args.output),
        include_conversations=args.include_conversations,
    )
    print(f"Export written to: {Path(args.output)}")
    print(f"Runs: {len(payload['runs'])}")
    print(f"Copied artifacts: {len(payload['copied_artifacts'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {"report", "export", "-h", "--help"}:
        # Backward-compatible report mode from 03-02.
        args = _build_report_parser().parse_args(args_list)
        return _run_report(args)

    parser = build_parser()
    args = parser.parse_args(args_list)
    if args.command == "export":
        return _run_export(args)
    if args.command == "report":
        return _run_report(args)
    parser.print_help()
    return 0
