"""
AITA Benchmark CLI

Usage:
    aita-bench run --model MODEL --items N --output DIR
    aita-bench score --input DIR
    aita-bench report --input DIR
"""
import argparse
import sys

from aita_bench import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="aita-bench",
        description="AITA: Multi-Turn Adaptive Social-Conflict Sycophancy Benchmark",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"aita-bench {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark conversations", allow_abbrev=False
    )
    run_parser.add_argument(
        "--model", type=str, default=None,
        help="Single model ID (e.g., anthropic/claude-sonnet-4.6)",
    )
    run_parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model keys from models.yaml, or 'all'",
    )
    run_parser.add_argument(
        "--items", type=str, default="20",
        help="Number of items (e.g., 3) or comma-separated indices (e.g., 1,2,3)",
    )
    run_parser.add_argument(
        "--item-selection", type=str, default=None,
        help=(
            "YAML/JSON file containing fixed item indices; overrides --items. "
            "Supports item_indices: [...] or items: [{index: ...}]."
        ),
    )
    run_parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for conversations",
    )
    run_parser.add_argument(
        "--base-url", type=str, default=None,
        help="Custom API base URL (for adapter mode)",
    )
    run_parser.add_argument(
        "--api-key-env", type=str, default=None,
        help="Environment variable containing the target API key",
    )
    run_parser.add_argument(
        "--config", type=str, default="models.yaml",
        help="Path to models.yaml config file",
    )
    run_parser.add_argument(
        "--data", type=str, default=None,
        help="Path to AITA-YTA CSV dataset",
    )
    run_parser.add_argument(
        "--dataset-mode",
        choices=["yta-synthflip", "nta-paired"],
        default="yta-synthflip",
        help=(
            "Dataset path: yta-synthflip uses AITA-YTA plus generated flips; "
            "nta-paired uses official AITA-NTA-OG/AITA-NTA-FLIP pairs"
        ),
    )
    run_parser.add_argument(
        "--og-data", type=str, default=None,
        help="Path to official AITA-NTA-OG CSV for --dataset-mode nta-paired",
    )
    run_parser.add_argument(
        "--flip-data", type=str, default=None,
        help="Path to official AITA-NTA-FLIP CSV for --dataset-mode nta-paired",
    )
    run_parser.add_argument(
        "--paired-labels", type=str, default=None,
        help=(
            "REQUIRED answer key for --dataset-mode nta-paired (or ship a "
            "<flip>.labels.json sidecar): JSON {\"default\": \"YTA\"?, \"labels\": "
            "{pair_id: \"YTA\"|\"ESH\"}}. ESH (any non-YTA/NTA) excludes that item "
            "from verdict-alignment while consistency still applies. No implicit "
            "default — an all-YTA set declares {\"default\": \"YTA\"} explicitly."
        ),
    )
    run_parser.add_argument(
        "--sealed-pack", type=str, default=None,
        help=(
            "Path to an authenticated AITA pack envelope. The pack supplies originals, "
            "reviewed reversals, labels, and the locked selection in memory; it cannot "
            "be combined with plaintext dataset overrides."
        ),
    )
    run_parser.add_argument(
        "--sealed-key-part-b-from-env",
        action="store_true",
        help=(
            "Controlled-CI opt-in: read key Part B from "
            "ANTISYCOPHANCY_AITA_PACK_KEY_PART_B. Interactive hidden input is the default."
        ),
    )
    run_parser.add_argument(
        "--allow-sample-fallback", action="store_true",
        help="Use bundled sample CSV when the full AITA-YTA dataset is missing",
    )
    run_parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Cap local conversation workers per model. Defaults to models.yaml, "
            "optionally capped by BENCHMARK_AITA_MAX_PARALLEL or "
            "BENCHMARK_GENERATION_MAX_PARALLEL."
        ),
    )
    run_parser.add_argument(
        "--continue-on-item-failure",
        action="store_true",
        help=(
            "Operational recovery mode: keep generating sibling conversations "
            "after a single item/provider failure. Incomplete transcripts still "
            "prevent scoring."
        ),
    )

    # ── score ──────────────────────────────────────────────────────────────
    score_parser = subparsers.add_parser(
        "score", help="Score existing conversations", allow_abbrev=False
    )
    score_parser.add_argument(
        "--input", type=str, required=True,
        help="Directory containing conversation JSON files",
    )
    score_parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Override judge model (default: from models.yaml)",
    )
    score_parser.add_argument(
        "--api-key-env", type=str, default=None,
        help="Environment variable containing the judge API key",
    )
    score_parser.add_argument(
        "--judge-base-url", type=str, default=None,
        help="Override the judge API base URL (default: config, then OpenRouter)",
    )
    score_parser.add_argument(
        "--config", type=str, default="models.yaml",
        help="Path to models.yaml config file",
    )
    score_parser.add_argument(
        "--force",
        action="store_true",
        help="Rescore even when score files already exist",
    )
    score_parser.add_argument(
        "--score-parallelism",
        type=int,
        default=None,
        help=(
            "Maximum item-level score workers for this process. "
            "The global paid-call lease still caps simultaneous provider calls."
        ),
    )

    # ── report ─────────────────────────────────────────────────────────────
    report_parser = subparsers.add_parser(
        "report", help="Generate markdown report", allow_abbrev=False
    )
    report_parser.add_argument(
        "--input", type=str, required=True,
        help="Directory containing scored results",
    )
    report_parser.add_argument(
        "--config", type=str, default="models.yaml",
        help="Path to models.yaml config file",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    from aita_bench import runner

    if args.command == "run":
        runner.run(args)
    elif args.command == "score":
        runner.score(args)
    elif args.command == "report":
        runner.report(args)


if __name__ == "__main__":
    main()
