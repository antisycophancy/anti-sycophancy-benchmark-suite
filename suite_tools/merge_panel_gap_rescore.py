"""Merge partial SUS results with panel-gap rescore rows.

Usage (for a completed panel-gap-rescore in a sus run dir):

    python -m suite_tools.merge_panel_gap_rescore \\
        --sus-dir results/prepared/<run>/sus \\
        --rescore-dir results/prepared/<run>/sus/panel-gap-rescore

Produces FINAL_RESULTS.json + FINAL_RESULTS-conversations.json alongside
the existing FINAL_RESULTS-partial.json (which is left untouched).

Row identity key: (condition_id, scenario, run_number).  Two rows with the
same key are considered the same benchmark sample; the rescore version wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from both the benchmark/ root and from the sus-bench/ dir.
_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
_SUS_BENCH_ROOT = _BENCHMARK_ROOT / "sus-bench"
for _p in [str(_BENCHMARK_ROOT), str(_SUS_BENCH_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sus_bench.report import write_json
from sus_bench.stats import aggregate_runs


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    """Unique identity for a scored row: (condition_id, scenario, run_number)."""
    return (
        str(row.get("condition_id", "")),
        str(row.get("scenario", "")),
        int(row.get("run_number", 0)),
    )


def merge(
    sus_dir: Path,
    rescore_dir: Path,
) -> list[dict[str, Any]]:
    """Merge partial and rescore conversations, preserving original order.

    Returns the merged list in the order defined by the original generation
    conversations file (``sus-bench-*-conversations.json``).

    Raises:
        FileNotFoundError: if expected files are missing.
        ValueError: if a rescore row has num_judges != 3 or if the merged
            count does not equal the original generation count.
    """
    # Load original generation conversations (defines order and total N).
    gen_files = sorted(sus_dir.glob("sus-bench-*-conversations.json"))
    if not gen_files:
        raise FileNotFoundError(f"No sus-bench-*-conversations.json in {sus_dir}")
    gen_path = gen_files[-1]  # pick latest if multiple
    with open(gen_path) as f:
        original: list[dict[str, Any]] = json.load(f)

    # Load partial scored rows.
    partial_path = sus_dir / "FINAL_RESULTS-partial-conversations.json"
    with open(partial_path) as f:
        partial_rows: list[dict[str, Any]] = json.load(f)

    # Load rescore rows (the -conversations.json from the score run).
    rescore_conv_path = rescore_dir / "RESCORE_RESULTS-conversations.json"
    with open(rescore_conv_path) as f:
        rescore_rows: list[dict[str, Any]] = json.load(f)

    # Validate: every rescore row must have num_judges == 3.
    for row in rescore_rows:
        pa = row.get("post_analysis") or {}
        nj = pa.get("num_judges")
        if nj != 3:
            raise ValueError(
                f"Rescore row {_row_key(row)} has num_judges={nj}, expected 3"
            )

    # Build lookup: key -> scored row (rescore takes precedence over partial).
    scored: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in partial_rows:
        scored[_row_key(row)] = row
    for row in rescore_rows:
        scored[_row_key(row)] = row

    # Walk original order and substitute scored versions.
    merged: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for orig_row in original:
        key = _row_key(orig_row)
        if key in scored:
            merged.append(scored[key])
        else:
            missing.append(key)

    if missing:
        raise ValueError(
            f"Merge incomplete — {len(missing)} original rows have no scored "
            f"version: {missing}"
        )

    if len(merged) != len(original):
        raise ValueError(
            f"Merged count {len(merged)} != original count {len(original)}"
        )

    return merged


def write_merged(
    merged: list[dict[str, Any]],
    sus_dir: Path,
    rescore_dir: Path,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Aggregate and write FINAL_RESULTS.json + FINAL_RESULTS-conversations.json.

    Returns (results_path, conversations_path).
    """
    aggregated = aggregate_runs(merged)

    # Load rescore cost to report alongside (may be absent for older runs).
    rescore_cost: dict[str, Any] | None = None
    rescore_summary_path = rescore_dir / "RESCORE_RESULTS.json"
    if rescore_summary_path.exists():
        with open(rescore_summary_path) as f:
            rescore_summary = json.load(f)
        rescore_cost = rescore_summary.get("cost")

    out_path = sus_dir / "FINAL_RESULTS.json"
    write_json(
        merged,
        aggregated,
        out_path,
        run_id=run_id or f"merged-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        cost=rescore_cost,
    )

    conv_path = sus_dir / "FINAL_RESULTS-conversations.json"
    return out_path, conv_path


def update_run_status(sus_dir: Path, merged: list[dict[str, Any]]) -> None:
    """Rewrite RUN_STATUS.json to reflect completed merge."""
    status_path = sus_dir / "RUN_STATUS.json"
    with open(status_path) as f:
        status = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    status["status"] = "completed"
    status["validity"] = "score_ready"
    status["updated_at"] = now
    status["completed_at"] = now
    status["scored_results"] = len(merged)
    status["results_path"] = str(sus_dir / "FINAL_RESULTS.json")
    # Clear failure fields
    status.pop("failure_reason", None)
    status.pop("failure_stage", None)
    status.pop("failed_at", None)
    status.pop("score_failures", None)
    status.pop("partial_results_path", None)
    status.pop("rerun_recommended", None)

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sus-dir",
        type=Path,
        required=True,
        help="Path to the sus/ run directory containing FINAL_RESULTS-partial-conversations.json",
    )
    parser.add_argument(
        "--rescore-dir",
        type=Path,
        required=True,
        help="Path to the panel-gap-rescore/ subdir containing RESCORE_RESULTS-conversations.json",
    )
    args = parser.parse_args(argv)

    print(f"Merging: {args.sus_dir}")
    merged = merge(args.sus_dir, args.rescore_dir)
    print(f"  Merged {len(merged)} rows total")

    out_path, conv_path = write_merged(merged, args.sus_dir, args.rescore_dir)
    print(f"  Written: {out_path}")
    print(f"  Written: {conv_path}")

    # Verify all rows have num_judges=3.
    bad = [
        i
        for i, r in enumerate(merged)
        if (r.get("post_analysis") or {}).get("num_judges") != 3
    ]
    if bad:
        print(f"ERROR: rows at indices {bad} do not have num_judges=3 after merge")
        sys.exit(1)

    update_run_status(args.sus_dir, merged)
    print(f"  RUN_STATUS.json updated to completed/score_ready")


if __name__ == "__main__":
    main()
