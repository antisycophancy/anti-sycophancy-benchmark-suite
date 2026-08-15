"""No-provider-call transcript hygiene gate for benchmark artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from suite_tools.conversation_hygiene import (
    HUMAN_REVIEW_META_PROMPT,
    scan_paths,
    summarize_issues,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.hygiene_gate",
        description=(
            "Scan SUS, AITA, Epistemic, or generic benchmark artifacts for "
            "provider errors, empty responses, wrappers, incomplete transcripts, "
            "and malformed model-output text. Makes no provider calls."
        ),
    )
    parser.add_argument("paths", nargs="+", help="JSON files or result directories to scan.")
    parser.add_argument("--json", help="Write a machine-readable hygiene report JSON.")
    parser.add_argument("--csv", help="Write issue rows as CSV.")
    parser.add_argument(
        "--markdown",
        "--md",
        dest="markdown",
        help="Write a human-readable hygiene report with the optional review prompt.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("blocking", "any", "none"),
        default="blocking",
        help=(
            "Exit non-zero on blocking issues, any issue including review-only "
            "wrappers, or never fail (default: blocking)."
        ),
    )
    parser.add_argument(
        "--print-meta-prompt",
        action="store_true",
        help="Print the optional second-pass human/LLM review prompt.",
    )
    return parser


def _print_bucket(title: str, values: dict[str, int]) -> None:
    if not values:
        return
    print(f"\n{title}")
    for key, count in values.items():
        print(f"  {key}: {count}")


def _render_summary(summary: dict[str, Any]) -> None:
    print("Benchmark conversation hygiene")
    print(f"  Issues: {summary['issues']}")
    print(f"  Blocking issues: {summary['blocking_issues']}")
    print(f"  Review issues: {summary['review_issues']}")
    print(f"  Records with blocking issues: {summary['records_with_blocking']}")
    print(f"  Records with review issues: {summary['records_with_review']}")
    _print_bucket("Modules", summary["by_module"])
    _print_bucket("Issue codes", summary["by_code"])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = scan_paths(Path(path) for path in args.paths)
    summary = summarize_issues(issues)
    _render_summary(summary)

    if args.json:
        write_json_report(Path(args.json), issues)
        print(f"\nJSON hygiene report saved: {args.json}")
    if args.csv:
        write_csv_report(Path(args.csv), issues)
        print(f"CSV hygiene report saved: {args.csv}")
    if args.markdown:
        write_markdown_report(Path(args.markdown), issues)
        print(f"Markdown hygiene report saved: {args.markdown}")
    if args.print_meta_prompt:
        print("\nOptional review meta-prompt")
        print(HUMAN_REVIEW_META_PROMPT.strip())

    if args.fail_on == "any" and summary["issues"]:
        return 1
    if args.fail_on == "blocking" and summary["blocking_issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
