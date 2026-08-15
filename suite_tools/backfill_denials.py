"""Retro-backfill of missed denials into BLOCKS.jsonl and BLOCK_REVIEWS.jsonl.

Dual discovery:
  A) Failed conversation artifacts whose failure_reason matches a typed denial
     signature (stop_reason=refusal with optional classifier=<cat>, or an OpenAI
     cyber_policy 400 body).
  B) RUN_EVENTS.jsonl lines whose event name is in ``_ITEM_FAILURE_EVENTS`` AND
     whose failure_reason, failure_status, or nested error.message matches the
     same signatures.

Append-only: transcripts and events are never modified.

Idempotence key: (module, model, unit_id, category, backfill_id).  BLOCKS and
BLOCK_REVIEWS are reconciled independently — a crash between the two writes heals
on re-apply.

Denial signatures are matched by ``_match_denial_signature``.  The rules mirror
``suite_tools.provider_signals`` (stop_reason=refusal + classifier extraction;
OpenAI cyber_policy body); when provider_signals rules change, update both
modules together.

Usage::

    # dry-run (default) — only categories in ALLOWED_CATEGORIES
    python -m suite_tools.backfill_denials <run_dir>

    # restrict to a subset of categories
    python -m suite_tools.backfill_denials <run_dir> --categories cyber,cyber_policy

    # write the blocks and reviews
    python -m suite_tools.backfill_denials <run_dir> --apply
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKFILL_ID = "retro-audit-20260721"
BLOCKS_FILENAME = "BLOCKS.jsonl"
BLOCK_REVIEWS_FILENAME = "BLOCK_REVIEWS.jsonl"
BLOCK_SCHEMA_VERSION = "benchmark-block-v1"
REVIEW_SCHEMA_VERSION = "benchmark-block-review-v1"

# 014b F1/F2 typed-signature denials that this backfill is authorised to record.
#   cyber / bio / refusal — Anthropic native stop_reason=refusal (classifier
#     extracted; "refusal" = stop_reason=refusal with no classifier present).
#   cyber_policy          — OpenAI 400 body with code="cyber_policy".
# Bare text-heuristic matches (e.g. "refused" in plain English) are never
# produced by _match_denial_signature, so they can never reach this allowlist.
ALLOWED_CATEGORIES: frozenset[str] = frozenset({"cyber", "bio", "refusal", "cyber_policy"})

# Filenames to skip when scanning for transcript artifacts.
_SKIP_FILENAMES = frozenset(
    {
        BLOCKS_FILENAME,
        BLOCK_REVIEWS_FILENAME,
        "RUN_EVENTS.jsonl",
        "RUN_STATUS.json",
        "RUN_CONTRACT.json",
        "RUN_CONTROL.json",
        "SCHEDULER_EVENTS.jsonl",
        "SCHEDULER_STATUS.json",
        "FINAL_RESULTS.json",
        "REPORT.md",
        "ATTEMPTS.jsonl",
    }
)

# Event names that indicate a per-item failure (Source B).  Only events whose
# name is in this set are considered.  This prevents stage-level failure events
# (e.g. "model_batch_failed", "stage_failed") — which may embed aggregate denial
# messages in their failure_reason — from being mistakenly backfilled as
# individual unit denials.  All real-data item-level denial events use
# "model_batch_item_failed"; "conversation_failed" covers custom runners.
_ITEM_FAILURE_EVENTS = frozenset(
    {
        "model_batch_item_failed",
        "conversation_failed",
        "item_failed",
        "unit_failed",
    }
)

# Module name normalisation → canonical unit_id prefix.
# Only these module names are accepted.  Unknown module names in events cause
# the event to be skipped with a stderr warning.
_MODULE_PREFIX: dict[str, str] = {
    "epis": "epis",
    "epistemic": "epis",
    "aita": "aita",
    "sus": "sus",
}


# ---------------------------------------------------------------------------
# Denial signature matching
# ---------------------------------------------------------------------------


def _match_denial_signature(text: str | None) -> str | None:
    """Return the category if text encodes a typed denial signature; else None.

    F1: Anthropic native refusal — contains ``stop_reason=refusal``.  The
    classifier value (e.g. ``cyber``) is extracted if present; otherwise the
    category is ``refusal`` (meaning stop_reason=refusal with no classifier —
    never a plain text-heuristic match).

    F2: OpenAI cyber_policy — body contains ``"code": "cyber_policy"`` or the
    human-readable phrase "flagged for possible cybersecurity risk".
    """
    if not text:
        return None

    # F1 — Anthropic native refusal (stop_reason=refusal)
    if "stop_reason=refusal" in text:
        m = re.search(r"classifier=(\w+)", text)
        return m.group(1) if m else "refusal"

    # F2 — OpenAI cyber_policy (400 body with code or phrase)
    if '"code": "cyber_policy"' in text or '"code":"cyber_policy"' in text:
        return "cyber_policy"
    if "flagged for possible cybersecurity risk" in text.lower():
        return "cyber_policy"

    return None


# ---------------------------------------------------------------------------
# Module resolution
# ---------------------------------------------------------------------------


def _resolve_event_module(module_raw: str) -> str | None:
    """Map an event module name to its canonical unit_id prefix.

    Returns None for unknown module names; the event scanner will emit a
    stderr warning and skip the event rather than guessing.
    """
    return _MODULE_PREFIX.get(str(module_raw).lower())


def _infer_transcript_module(*, has_test_type: bool) -> str:
    """Infer the unit_id module prefix from transcript structural fields.

    Transcripts do not carry a module field; structure is the only signal:
      - test_type present → epistemic-sycophancy (epis)
      - test_type absent  → AITA (aita)
    """
    return "epis" if has_test_type else "aita"


# ---------------------------------------------------------------------------
# unit_id construction
# ---------------------------------------------------------------------------


def _build_unit_id(prefix: str, model: str, item_idx: int, test_type: str | None, side: str) -> str:
    """Construct a canonical unit_id from its components."""
    if prefix == "epis":
        return f"epis:{model}:{test_type}:item{item_idx}:{side}"
    if prefix == "aita":
        return f"aita:{model}:item{item_idx}:{side}"
    # sus and future modules: include test_type if present, omit if not.
    if test_type:
        return f"{prefix}:{model}:{test_type}:item{item_idx}:{side}"
    return f"{prefix}:{model}:item{item_idx}:{side}"


# ---------------------------------------------------------------------------
# Internal helpers for loading existing ledger keys
# ---------------------------------------------------------------------------


def _load_block_keys(run_dir: Path) -> set[tuple]:
    """Return the set of (module, model, unit_id, category, backfill_id) keys
    already present in BLOCKS.jsonl for THIS backfill_id."""
    path = run_dir / BLOCKS_FILENAME
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("backfill_id") == BACKFILL_ID:
            keys.add(
                (r.get("module"), r.get("model"), r.get("unit_id"), r.get("category"), r.get("backfill_id"))
            )
    return keys


def _load_review_keys(run_dir: Path) -> set[tuple]:
    """Return the set of (module, model, unit_id, category, backfill_id) keys
    already present in BLOCK_REVIEWS.jsonl for THIS backfill_id."""
    path = run_dir / BLOCK_REVIEWS_FILENAME
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("backfill_id") == BACKFILL_ID:
            keys.add(
                (r.get("module"), r.get("model"), r.get("unit_id"), r.get("category"), r.get("backfill_id"))
            )
    return keys


def _idempotence_key(denial: dict[str, Any]) -> tuple:
    """Extract the five-tuple idempotence key from a denial dict."""
    return (
        denial.get("module"),
        denial.get("model"),
        denial.get("unit_id"),
        denial.get("category"),
        BACKFILL_ID,
    )


# ---------------------------------------------------------------------------
# Source A — transcript artifact scanning
# ---------------------------------------------------------------------------


def _scan_transcripts(run_dir: Path) -> list[dict[str, Any]]:
    """Scan failed transcript JSON files for denial signatures."""
    found: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in _SKIP_FILENAMES or path.name.endswith("_scores.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("completed", True):
            continue
        failure_reason = data.get("failure_reason") or ""
        category = _match_denial_signature(failure_reason)
        if category is None:
            continue
        model = data.get("model") or ""
        item_idx = data.get("item_idx")
        test_type = data.get("test_type")
        side = data.get("side") or ""
        if item_idx is None or not side:
            continue
        prefix = _infer_transcript_module(has_test_type=bool(test_type))
        unit_id = _build_unit_id(prefix, model, int(item_idx), test_type, side)
        found.append(
            {
                "module": prefix,
                "model": model,
                "unit_id": unit_id,
                "category": category,
                "evidence_class": "model_signal",
                "evidence_pointer": path.name,
                "failure_reason": failure_reason,
            }
        )
    return found


# ---------------------------------------------------------------------------
# Source B — RUN_EVENTS.jsonl scanning
# ---------------------------------------------------------------------------


def _scan_events(run_dir: Path) -> list[dict[str, Any]]:
    """Scan RUN_EVENTS.jsonl for per-item terminal failure events with denial
    signatures.

    Guards applied (both must pass):
      1. Event name is in ``_ITEM_FAILURE_EVENTS``.
      2. ``item_idx`` is present (item-level, not batch/stage-level).

    Signature is probed in ``failure_reason``, ``failure_status``, and
    nested ``error.message`` — in that order of priority.

    ``evidence_pointer`` is ``RUN_EVENTS.jsonl#sequence=N`` (1-based line
    index; the append-only ledger makes this stable).
    """
    events_path = run_dir / "RUN_EVENTS.jsonl"
    if not events_path.exists():
        return []
    found: list[dict[str, Any]] = []
    for line_idx, raw_line in enumerate(events_path.read_text().splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        # Guard 1: known item-failure event name.
        event_name = event.get("event") or ""
        if event_name not in _ITEM_FAILURE_EVENTS:
            continue

        # Guard 2: must be a per-item event.
        item_idx = event.get("item_idx")
        if item_idx is None:
            continue

        # Probe all text fields that may carry the denial signature.
        failure_reason = event.get("failure_reason") or ""
        failure_status = event.get("failure_status") or ""
        error_message = ""
        error_obj = event.get("error")
        if isinstance(error_obj, dict):
            error_message = error_obj.get("message") or ""

        category = (
            _match_denial_signature(failure_reason)
            or _match_denial_signature(failure_status)
            or _match_denial_signature(error_message)
        )
        if category is None:
            continue

        # Resolve module — unknown modules are skipped with a warning.
        module_raw = event.get("module") or ""
        prefix = _resolve_event_module(module_raw)
        if prefix is None:
            print(
                f"WARNING: backfill_denials: skipping event at line {line_idx} "
                f"with unknown module {module_raw!r}",
                file=sys.stderr,
            )
            continue

        test_type = event.get("test_type")
        side = event.get("side") or ""
        model = event.get("model") or ""
        if not side:
            continue

        # Use the first non-empty text that carried the signature as the
        # effective evidence text for the pointer label.
        evidence_text = failure_reason or failure_status or error_message
        unit_id = _build_unit_id(prefix, model, int(item_idx), test_type, side)
        found.append(
            {
                "module": prefix,
                "model": model,
                "unit_id": unit_id,
                "category": category,
                "evidence_class": "model_signal",
                "evidence_pointer": f"RUN_EVENTS.jsonl#sequence={line_idx}",
                "failure_reason": evidence_text,
            }
        )
    return found


# ---------------------------------------------------------------------------
# Core scanning (no allowlist or BLOCKS filtering)
# ---------------------------------------------------------------------------


def _scan_denials(run_dir: Path) -> list[dict[str, Any]]:
    """Scan both sources and return all potential denials, deduplicated by
    (unit_id, category) — transcript findings take priority over event findings
    for the same unit."""
    run_dir = Path(run_dir)
    from_transcripts = _scan_transcripts(run_dir)
    from_events = _scan_events(run_dir)

    # Deduplicate: track (unit_id, category) seen; transcripts first.
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for denial in from_transcripts + from_events:
        key = (denial["unit_id"], denial["category"])
        if key not in seen:
            seen.add(key)
            result.append(denial)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover(
    run_dir: "Path | str",
    *,
    categories: "frozenset[str] | None" = None,
) -> list[dict[str, Any]]:
    """Return denials not yet recorded in BLOCKS.jsonl (dry-run preview).

    Only returns denials whose ``category`` is in ``categories`` (default:
    ``ALLOWED_CATEGORIES``).  Pass an explicit set to narrow further.

    Each dict has at minimum: ``module``, ``model``, ``unit_id``, ``category``,
    ``evidence_class``, ``evidence_pointer``.
    """
    run_dir = Path(run_dir)
    allowed = ALLOWED_CATEGORIES if categories is None else categories
    all_denials = _scan_denials(run_dir)
    existing = _load_block_keys(run_dir)
    return [
        d for d in all_denials
        if d["category"] in allowed and _idempotence_key(d) not in existing
    ]


def _append_block(run_dir: "Path | str", denial: dict[str, Any]) -> None:
    """Append one entry to BLOCKS.jsonl (does NOT check idempotence — caller's
    responsibility)."""
    run_dir = Path(run_dir)
    block: dict[str, Any] = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": denial.get("module"),
        "model": denial.get("model"),
        "unit_id": denial.get("unit_id"),
        "evidence_class": denial.get("evidence_class", "model_signal"),
        "category": denial.get("category"),
        "evidence_pointer": denial.get("evidence_pointer"),
        "backfilled": True,
        "backfill_id": BACKFILL_ID,
    }
    with (run_dir / BLOCKS_FILENAME).open("a") as fh:
        fh.write(json.dumps(block, default=str) + "\n")


def _append_review(run_dir: "Path | str", denial: dict[str, Any]) -> None:
    """Append one entry to BLOCK_REVIEWS.jsonl (does NOT check idempotence —
    caller's responsibility)."""
    run_dir = Path(run_dir)
    review: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "module": denial.get("module"),
        "model": denial.get("model"),
        "unit_id": denial.get("unit_id"),
        "category": denial.get("category"),
        "backfill_id": BACKFILL_ID,
        "disposition": "safety_declination",
        "reviewer": BACKFILL_ID,
        "rationale": (
            f"Retro-backfill: denial signature detected in "
            f"{denial.get('evidence_pointer')} "
            f"(category={denial.get('category')})"
        ),
    }
    with (run_dir / BLOCK_REVIEWS_FILENAME).open("a") as fh:
        fh.write(json.dumps(review, default=str) + "\n")


def apply(
    run_dir: "Path | str",
    *,
    categories: "frozenset[str] | None" = None,
) -> list[dict[str, Any]]:
    """Write missing BLOCKS and BLOCK_REVIEWS entries, reconciled independently.

    Only processes denials whose ``category`` is in ``categories`` (default:
    ``ALLOWED_CATEGORIES``).

    Returns the list of denials that were newly written to either ledger (for
    progress reporting).  Safe to call multiple times: a second call is a no-op
    when both ledgers are already complete; a call after a crash between the two
    writes heals only the missing ledger.
    """
    run_dir = Path(run_dir)
    allowed = ALLOWED_CATEGORIES if categories is None else categories
    all_denials = [d for d in _scan_denials(run_dir) if d["category"] in allowed]
    existing_blocks = _load_block_keys(run_dir)
    existing_reviews = _load_review_keys(run_dir)

    acted: list[dict[str, Any]] = []
    for denial in all_denials:
        key = _idempotence_key(denial)
        wrote_anything = False
        if key not in existing_blocks:
            _append_block(run_dir, denial)
            existing_blocks.add(key)  # prevent double-write within this call
            wrote_anything = True
        if key not in existing_reviews:
            _append_review(run_dir, denial)
            existing_reviews.add(key)
            wrote_anything = True
        if wrote_anything:
            acted.append(denial)
    return acted


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _print_table(denials: list[dict[str, Any]]) -> None:
    if not denials:
        print("  (nothing to backfill)")
        return
    header = f"  {'unit_id':<55}  {'cat':<20}  evidence_pointer"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for d in denials:
        print(
            f"  {d['unit_id']:<55}  {d['category']:<20}  {d['evidence_pointer']}"
        )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Retro-backfill missed denials into BLOCKS.jsonl / BLOCK_REVIEWS.jsonl"
    )
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the backfill entries (default: dry-run only)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated category allowlist override "
            f"(default: {','.join(sorted(ALLOWED_CATEGORIES))})"
        ),
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    cats: frozenset[str] | None = None
    if args.categories:
        cats = frozenset(c.strip() for c in args.categories.split(",") if c.strip())

    if args.apply:
        acted = apply(run_dir, categories=cats)
        print(f"Backfill applied: {len(acted)} denial(s) written/healed in {run_dir}")
        _print_table(acted)
    else:
        found = discover(run_dir, categories=cats)
        print(f"Dry-run: {len(found)} denial(s) would be backfilled in {run_dir}")
        _print_table(found)


if __name__ == "__main__":
    main()
