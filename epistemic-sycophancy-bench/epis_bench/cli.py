"""Command line interface for epis-bench.

Provides run/score/report subcommands for epistemic sycophancy benchmarking.
"""

from __future__ import annotations

import argparse
import sys

from epis_bench import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="epis-bench",
        description=(
            "Epistemic Sycophancy Bench — Multi-turn adaptive epistemic "
            "sycophancy benchmark. Tests whether AI models maintain "
            "intellectual positions under conversational pressure."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"epis-bench {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ───────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark conversations", allow_abbrev=False
    )
    run_parser.add_argument("--model", type=str, help="Single model ID (e.g. anthropic/claude-sonnet-4.6)")
    run_parser.add_argument("--models", type=str, default="all", help="Comma-separated model keys or 'all'")
    run_parser.add_argument("--items", type=int, default=4, help="Items per test type (default: 4)")
    run_parser.add_argument("--types", type=str, default="delusion,pickside,mirror", help="Comma-separated test types")
    run_parser.add_argument("--output", "-o", type=str, help="Output directory")
    run_parser.add_argument("--base-url", type=str, help="Custom API base URL")
    run_parser.add_argument(
        "--api-key-env",
        type=str,
        help="Environment variable containing the target API key",
    )
    run_parser.add_argument("--config", type=str, default="models.yaml", help="Path to models.yaml")
    run_parser.add_argument("--data-dir", type=str, help="Path to Syco-Bench questions directory")
    run_parser.add_argument("--selection", type=str, help="Path to curated selection YAML")
    run_parser.add_argument(
        "--continue-on-item-failure",
        action="store_true",
        help="Operational recovery mode: keep generating sibling conversations after an item/provider failure.",
    )
    run_parser.add_argument(
        "--allow-provider-refusals",
        action="store_true",
        help="Treat provider refusal transcripts as logged exclusions rather than score-blocking incompletes.",
    )

    # ── score ─────────────────────────────────────────────────────────────
    score_parser = subparsers.add_parser(
        "score", help="Score completed conversations", allow_abbrev=False
    )
    score_parser.add_argument("--input", "-i", type=str, required=True, help="Directory with conversation JSON files")
    score_parser.add_argument("--config", type=str, default="models.yaml", help="Path to models.yaml")
    score_parser.add_argument(
        "--api-key-env",
        type=str,
        help="Environment variable containing the judge API key",
    )
    score_parser.add_argument("--judge-model", type=str, help="Override judge model")
    score_parser.add_argument("--force", action="store_true", help="Rescore even when score files already exist")
    score_parser.add_argument("--score-parallelism", type=int, help="Maximum score items to process in parallel")
    score_parser.add_argument(
        "--allow-provider-refusals",
        action="store_true",
        help="Exclude provider refusal transcripts while still blocking other incomplete transcripts.",
    )

    # ── report ────────────────────────────────────────────────────────────
    report_parser = subparsers.add_parser(
        "report", help="Generate results report", allow_abbrev=False
    )
    report_parser.add_argument("--input", "-i", type=str, required=True, help="Directory with scored results")
    report_parser.add_argument("--config", type=str, default="models.yaml", help="Path to models.yaml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    from epis_bench import runner

    if args.command == "run":
        runner.run(args)
    elif args.command == "score":
        runner.score(args)
    elif args.command == "report":
        runner.report(args)


if __name__ == "__main__":
    main()
