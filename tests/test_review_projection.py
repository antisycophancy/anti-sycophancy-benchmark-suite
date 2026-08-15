"""Tests for suite_tools.review_projection — Task 020-T5 (D4 + D5).

RED-first contract for the review model + effective-evidence projection: the one
place BLOCK_REVIEWS judgment joins BLOCKS + attempt_failure_classified facts into
one effective state.  Covers event_ref parse/resolve (incl. hash-anchored line
refs + drift detection), supersession chains, the O_EXCL-locked append helper,
the backfill v1 grandfather resolver, and the attempt-aware unit reduction.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from suite_tools import review_projection as rp


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _contract(run: Path, module: str, units: list[dict], *, run_id: str = "run-1") -> None:
    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": run_id,
        "modules": [{"module": module, "stage": "generation", "expected_units": units}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "x",
             "route": "openrouter", "effort": "high", "profile": None}]},
    }))


def _block(unit_id, *, attempt=1, category="refusal", cls="model_signal",
           block_id=None, **extra) -> dict:
    rec = {
        "schema_version": "benchmark-block-v2",
        "timestamp": "2026-01-01T00:00:00Z",
        "module": "aita", "stage": "gen", "attempt_number": attempt, "model": "m",
        "unit_id": unit_id, "unit": {"item_idx": 0, "side": "side_a"},
        "evidence_class": cls, "category": category,
    }
    if block_id is not None:
        rec["block_id"] = block_id
    rec.update(extra)
    return rec


def _event(*, unit_id=None, attempt=1, category="refusal", cls="model_signal",
           event_id=None, action="halt", **extra) -> dict:
    rec = {
        "schema_version": "benchmark-run-monitor-v1",
        "sequence": 1, "timestamp": "2026-01-01T00:00:00Z",
        "module": "aita", "stage": "gen", "event": "attempt_failure_classified",
        "attempt_number": attempt, "action": action,
        "evidence_class": cls, "category": category, "failure_reason": "boom",
        "model": "m",
    }
    if unit_id is not None:
        rec["unit_id"] = unit_id
    if event_id is not None:
        rec["event_id"] = event_id
    rec.update(extra)
    return rec


def _review(event_ref, disposition, *, review_id=None, category="refusal",
            resolved_category=None, supersedes=None, module="aita", model="m",
            unit_id=None) -> dict:
    rec = {
        "schema_version": rp.REVIEW_SCHEMA_VERSION,
        "review_id": review_id or uuid.uuid4().hex,
        "event_ref": event_ref, "module": module, "model": model,
        "category": category, "disposition": disposition,
        "reviewer": "tester", "rationale": "because",
        "reviewed_at": "2026-01-02T00:00:00Z",
    }
    if unit_id is not None:
        rec["unit_id"] = unit_id
    if resolved_category is not None:
        rec["resolved_category"] = resolved_category
    if supersedes is not None:
        rec["supersedes_review_id"] = supersedes
    return rec


# ---------------------------------------------------------------------------
# D4 — event_ref parse / resolve / drift
# ---------------------------------------------------------------------------

def test_parse_event_ref_id_forms():
    for source in ("blocks", "events"):
        ref = rp.parse_event_ref(f"{source}-id:abc123")
        assert ref.source == source
        assert ref.kind == "id"
        assert ref.ident == "abc123"


def test_parse_event_ref_line_forms():
    ref = rp.parse_event_ref("blocks-line:7:deadbeef")
    assert ref.source == "blocks"
    assert ref.kind == "line"
    assert ref.line == 7
    assert ref.sha8 == "deadbeef"


def test_parse_event_ref_rejects_garbage():
    with pytest.raises(ValueError):
        rp.parse_event_ref("nonsense")
    with pytest.raises(ValueError):
        rp.parse_event_ref("blocks-line:notanint:abcd")


def test_resolve_block_id_ref(tmp_path):
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    resolved = rp.resolve_event_ref(tmp_path, f"blocks-id:{bid}")
    assert resolved.record["unit_id"] == "aita:m:item0:side_a"
    assert resolved.source == "blocks"


def test_resolve_event_id_ref(tmp_path):
    eid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [_event(event_id=eid)])
    resolved = rp.resolve_event_ref(tmp_path, f"events-id:{eid}")
    assert resolved.record["event"] == "attempt_failure_classified"
    assert resolved.source == "events"


def test_resolve_line_ref_hash_matches(tmp_path):
    rec = _block("aita:m:item0:side_a")  # no block_id -> line ref territory
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [rec])
    raw = (tmp_path / "BLOCKS.jsonl").read_text().splitlines()[0]
    sha8 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    resolved = rp.resolve_event_ref(tmp_path, f"blocks-line:1:{sha8}")
    assert resolved.record["unit_id"] == "aita:m:item0:side_a"


def test_resolve_line_ref_hash_mismatch_is_hard_error(tmp_path):
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a")])
    with pytest.raises(rp.EventRefDriftError) as exc:
        rp.resolve_event_ref(tmp_path, "blocks-line:1:00000000")
    # error names the drift (the file + line)
    assert "BLOCKS.jsonl" in str(exc.value) or "line 1" in str(exc.value)


def test_line_sha8_is_first8_of_sha256():
    raw = b'{"a": 1}'
    assert rp.line_sha8(raw) == hashlib.sha256(raw).hexdigest()[:8]


# ---------------------------------------------------------------------------
# D4 — locked append + supersession chain
# ---------------------------------------------------------------------------

def test_append_review_writes_and_loads(tmp_path):
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    review = _review(f"blocks-id:{bid}", "safety_declination")
    rp.append_review(tmp_path, review)
    loaded = rp.load_reviews(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["disposition"] == "safety_declination"


def test_append_duplicate_active_review_errors_without_supersede(tmp_path):
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    ref = f"blocks-id:{bid}"
    rp.append_review(tmp_path, _review(ref, "safety_declination"))
    # A second active review for the same fact without supersede is the loser.
    with pytest.raises(rp.DuplicateActiveReviewError):
        rp.append_review(tmp_path, _review(ref, "needs_escalation"))


def test_append_supersede_is_allowed_and_head_wins(tmp_path):
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    ref = f"blocks-id:{bid}"
    first = _review(ref, "safety_declination")
    rp.append_review(tmp_path, first)
    second = _review(ref, "needs_escalation", supersedes=first["review_id"])
    rp.append_review(tmp_path, second)
    result = rp.project(tmp_path)
    fv = result.events_by_ref[ref]
    # Head of the supersession chain wins.
    assert fv.active_review["review_id"] == second["review_id"]
    assert fv.disposition == "needs_escalation"


def test_append_lock_contention_errors_cleanly(tmp_path):
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    lock = tmp_path / rp.BLOCK_REVIEWS_LOCK_FILENAME
    # Manually hold the lock (simulate a concurrent invocation mid-append).
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with pytest.raises(rp.ReviewLockError):
            rp.append_review(tmp_path, _review(f"blocks-id:{bid}", "safety_declination"),
                             lock_timeout=0.3)
    finally:
        os.close(fd)
        lock.unlink()


# ---------------------------------------------------------------------------
# D4 — backfill v1 grandfathering
# ---------------------------------------------------------------------------

def test_backfill_v1_review_grandfathered_to_unique_block(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 3}])
    block = _block("aita:m:item0:side_a", cls="model_signal", category="cyber")
    block.update({"schema_version": "benchmark-block-v1", "backfilled": True,
                  "backfill_id": "retro-audit-20260721"})
    block.pop("block_id", None)
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [block])
    v1_review = {
        "schema_version": "benchmark-block-review-v1", "module": "aita", "model": "m",
        "unit_id": "aita:m:item0:side_a", "category": "cyber",
        "backfill_id": "retro-audit-20260721", "disposition": "safety_declination",
        "reviewer": "retro-audit-20260721",
    }
    _write_jsonl(tmp_path / "BLOCK_REVIEWS.jsonl", [v1_review])
    result = rp.project(tmp_path)
    uv = result.units_by_id["aita:m:item0:side_a"]
    assert uv.disposition == "safety_declination"
    # It counts as an active safety_declination review on the block's fact.
    assert uv.carrier is not None
    assert uv.carrier.active_review is not None


def test_backfill_v1_review_ambiguous_is_hard_error(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 3}])
    # Two physical backfilled blocks with the SAME composite -> ambiguous.
    block = _block("aita:m:item0:side_a", cls="model_signal", category="cyber")
    block.update({"schema_version": "benchmark-block-v1", "backfilled": True,
                  "backfill_id": "retro-audit-20260721"})
    block.pop("block_id", None)
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [dict(block), dict(block)])
    v1_review = {
        "schema_version": "benchmark-block-review-v1", "module": "aita", "model": "m",
        "unit_id": "aita:m:item0:side_a", "category": "cyber",
        "backfill_id": "retro-audit-20260721", "disposition": "safety_declination",
    }
    _write_jsonl(tmp_path / "BLOCK_REVIEWS.jsonl", [v1_review])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


# ---------------------------------------------------------------------------
# D5 — attempt-aware unit reduction: retry
# ---------------------------------------------------------------------------

def test_retry_reopens_then_later_attempt_resolves_without_inheritance(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=1, block_id=bid)])
    rp.append_review(tmp_path, _review(f"blocks-id:{bid}", "retry"))

    # Before the later attempt: pending_retry (no strictly-later completion).
    r1 = rp.project(tmp_path)
    assert r1.units_by_id["aita:m:item0:side_a"].state == "pending_retry"

    # Attempt-2 completed artifact stamped with the later attempt resolves it,
    # and the new outcome does NOT inherit the retry review.
    (tmp_path / "t.json").write_text(json.dumps(
        {"attempt_number": 2, "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
    r2 = rp.project(tmp_path)
    uv = r2.units_by_id["aita:m:item0:side_a"]
    assert uv.state == "completed"
    assert uv.disposition != "retry"  # obligation replaced, review not inherited


def test_older_attempt_completion_does_not_satisfy_newer_block_retry(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    # Block at attempt 3 with retry; a completed artifact stamped attempt 2 (older).
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=3, block_id=bid)])
    rp.append_review(tmp_path, _review(f"blocks-id:{bid}", "retry"))
    (tmp_path / "t.json").write_text(json.dumps(
        {"attempt_number": 2, "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "pending_retry"


# ---------------------------------------------------------------------------
# D5 — v1 artifact compat, BOTH paths
# ---------------------------------------------------------------------------

def test_v1_artifact_compat_zero_reviews_attemptless_artifact_wins(tmp_path):
    """Zero reviews: an attempt-less completed artifact discharges an
    attempt-less block (today's completed-artifact-wins behaviour)."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    b = _block("aita:m:item0:side_a")
    b.pop("attempt_number", None)  # attempt-less v1-shape block
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [b])
    (tmp_path / "t.json").write_text(json.dumps(
        {"turns": [{"model_response": "a"}, {"model_response": "b"}]}))  # attempt-less
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "completed"


def test_v1_artifact_compat_post_retry_attemptless_artifact_does_not_discharge(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    b = _block("aita:m:item0:side_a", block_id="fixedblockid")
    b.pop("attempt_number", None)  # attempt-less block (attempt 0)
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [b])
    rp.append_review(tmp_path, _review("blocks-id:fixedblockid", "retry"))
    (tmp_path / "t.json").write_text(json.dumps(
        {"turns": [{"model_response": "a"}, {"model_response": "b"}]}))  # attempt-less
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "pending_retry"


# ---------------------------------------------------------------------------
# D5 — same-attempt conflict vs compatible duplicates
# ---------------------------------------------------------------------------

def test_same_attempt_conflicting_facts_are_integrity_error(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=1, cls="model_signal", category="refusal")])
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [
        _event(unit_id="aita:m:item0:side_a", attempt=1, cls="instrument_defect",
               category="timeout", event_id="e1")])
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.integrity_error is True
    assert uv.state == "unresolved"


def test_compatible_duplicates_resolve_blocks_over_events_order_independent(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    block = _block("aita:m:item0:side_a", attempt=1, cls="model_signal", category="refusal")
    event = _event(unit_id="aita:m:item0:side_a", attempt=1, cls="model_signal",
                   category="refusal", event_id="e1")
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [block])
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [event])
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.integrity_error is False
    assert uv.carrier.source == "blocks"  # BLOCKS-over-events, order-independent
    assert uv.state == "terminal_model_signal"


# ---------------------------------------------------------------------------
# D5 — scope + member obligations + unmappable rejection
# ---------------------------------------------------------------------------

def test_member_scoped_retry_creates_member_obligation(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    # A member-scoped halt event (no unit_id) with a retry review.
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [_event(event_id="e1")])
    rp.append_review(tmp_path, _review("events-id:e1", "retry"))
    result = rp.project(tmp_path)
    assert len(result.member_obligations) == 1
    assert result.member_obligations[0].event_ref == "events-id:e1"


def test_member_retry_uses_stage_completion_after_status_moves_to_scoring(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    failure = _event(
        event_id="e1", attempt=2, cls="unknown", category="unclassified",
        stage="generation",
    )
    generation_completed = {
        "schema_version": "benchmark-run-ledger-v1",
        "sequence": 2,
        "timestamp": "2026-01-03T00:00:00Z",
        "module": "aita",
        "stage": "generation",
        "event": "stage_completed",
        "attempt_number": 4,
        "status": "completed",
    }
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [failure, generation_completed])
    (tmp_path / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "stage": "scoring",
        "attempt_number": 1,
        "status": "completed",
    }))
    rp.append_review(tmp_path, _review(
        "events-id:e1", "retry", category="unclassified",
    ))

    result = rp.project(tmp_path)
    fact = result.events_by_ref["events-id:e1"]
    assert rp.is_publication_blocking(fact) is None
    assert len(result.member_obligations) == 1
    assert result.member_obligations[0].fulfilled is True


def test_retry_on_unmappable_legacy_is_rejected(tmp_path):
    # A BLOCKS fact with no unit_id is unmappable-legacy (a block is inherently
    # unit-level; a missing unit_id cannot be mapped).  retry is rejected.
    b = _block("x", cls="model_signal")
    b.pop("unit_id", None)
    b.pop("block_id", None)
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [b])
    raw = (tmp_path / "BLOCKS.jsonl").read_text().splitlines()[0]
    ref = f"blocks-line:1:{rp.line_sha8(raw.encode('utf-8'))}"
    with pytest.raises(rp.ReviewValidationError):
        rp.append_review(tmp_path, _review(ref, "retry"))


def test_block_without_unit_id_scope_is_unmappable(tmp_path):
    b = _block("x", cls="model_signal")
    b.pop("unit_id", None)
    b.pop("block_id", None)
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [b])
    result = rp.project(tmp_path)
    fv = list(result.events_by_ref.values())[0]
    assert fv.scope == "unmappable_legacy"


def test_member_scoped_event_scope(tmp_path):
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [_event(event_id="e1")])  # no unit_id
    result = rp.project(tmp_path)
    fv = result.events_by_ref["events-id:e1"]
    assert fv.scope == "member"


# ---------------------------------------------------------------------------
# D5 — disposition semantics
# ---------------------------------------------------------------------------

def test_safety_declination_resolves_unknown_with_resolved_category(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="unknown", category="ambiguous_403", block_id=bid)])
    rp.append_review(tmp_path, _review(
        f"blocks-id:{bid}", "safety_declination", category="ambiguous_403",
        resolved_category="cyber"))
    result = rp.project(tmp_path)
    fv = result.events_by_ref[f"blocks-id:{bid}"]
    assert fv.effective_class == "model_signal"
    assert fv.effective_category == "cyber"
    assert fv.resolution_status == "resolved"


def test_safety_declination_on_unknown_without_resolved_category_errors(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="unknown", category="ambiguous_403", block_id=bid)])
    # No resolved_category on an unclassified underlying category -> error.
    with pytest.raises(rp.ReviewValidationError):
        rp.append_review(tmp_path, _review(
            f"blocks-id:{bid}", "safety_declination", category="ambiguous_403"))


def test_needs_escalation_is_unresolved(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id=bid)])
    rp.append_review(tmp_path, _review(f"blocks-id:{bid}", "needs_escalation"))
    fv = rp.project(tmp_path).events_by_ref[f"blocks-id:{bid}"]
    assert fv.resolution_status == "unresolved"


def test_instrument_defect_effective_class(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    bid = uuid.uuid4().hex
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id=bid)])
    rp.append_review(tmp_path, _review(f"blocks-id:{bid}", "instrument_defect"))
    result = rp.project(tmp_path)
    fv = result.events_by_ref[f"blocks-id:{bid}"]
    assert fv.effective_class == "instrument_defect"
    uv = result.units_by_id["aita:m:item0:side_a"]
    assert uv.state == "instrument_defect"


# ---------------------------------------------------------------------------
# events_by_ref covers BOTH sources
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# final-gate F3 — strict attempt ordering for V2 carriers
# ---------------------------------------------------------------------------

def test_v2_block_not_discharged_by_older_attempt_artifact(tmp_path):
    """A V2 (attempt-stamped) model-signal block must NOT be discharged by a
    completed artifact from an EARLIER attempt — that would resurrect a declined
    unit into scoring (final-gate F3, Sol's reproduction)."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=2, block_id="b1")])
    (tmp_path / "t.json").write_text(json.dumps(
        {"attempt_number": 1, "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "terminal_model_signal"
    assert uv.reason == "BLOCKS.jsonl model_signal entry"


def test_v2_block_discharged_by_strictly_later_attempt_artifact(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=2, block_id="b1")])
    (tmp_path / "t.json").write_text(json.dumps(
        {"attempt_number": 3, "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "completed"
    assert uv.reason == "completed_after_block"


def test_v2_block_not_discharged_by_attemptless_artifact(tmp_path):
    """An attempt-less artifact (attempt 0) cannot discharge a V2 attempt-1
    block (0 is not strictly later)."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", attempt=1, block_id="b1")])
    (tmp_path / "t.json").write_text(json.dumps(
        {"turns": [{"model_response": "a"}, {"model_response": "b"}]}))  # attempt-less
    uv = rp.project(tmp_path).units_by_id["aita:m:item0:side_a"]
    assert uv.state == "terminal_model_signal"


# ---------------------------------------------------------------------------
# final-gate F4 — supersession is not a gate bypass
# ---------------------------------------------------------------------------

def test_supersede_of_nonexistent_id_is_rejected(tmp_path):
    """Sol's reproduction: needs_escalation, then a supersede-of-a-NONEXISTENT-id
    safety_declination must be REJECTED, not silently become the resolver."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    rp.append_review(tmp_path, _review("blocks-id:b1", "needs_escalation", review_id="r1"))
    with pytest.raises(rp.ReviewValidationError):
        rp.append_review(tmp_path, _review(
            "blocks-id:b1", "safety_declination", review_id="r2",
            supersedes="does-not-exist"))


def test_supersede_must_name_current_active_head(tmp_path):
    """A supersede naming a stale (already-superseded) review is rejected."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    rp.append_review(tmp_path, _review("blocks-id:b1", "safety_declination", review_id="r1"))
    rp.append_review(tmp_path, _review("blocks-id:b1", "needs_escalation",
                                       review_id="r2", supersedes="r1"))
    # r1 is no longer the head; superseding it again is rejected.
    with pytest.raises(rp.ReviewValidationError):
        rp.append_review(tmp_path, _review("blocks-id:b1", "safety_declination",
                                           review_id="r3", supersedes="r1"))


def test_multiple_active_heads_is_integrity_error(tmp_path):
    """Two non-superseding v2 heads for one ref (e.g. hand-edited file) → the
    projection raises rather than silently picking by timestamp."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    _write_jsonl(tmp_path / rp.BLOCK_REVIEWS_FILENAME, [
        _review("blocks-id:b1", "needs_escalation", review_id="r1"),
        _review("blocks-id:b1", "safety_declination", review_id="r2"),
    ])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_dangling_supersession_target_rejected_at_load(tmp_path):
    """final-gate2 F4: a ledger-injected review whose supersedes_review_id names
    a NONEXISTENT review must NOT be silently accepted as the active resolver —
    the read path fails closed even when append-time validation was bypassed."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    # Hand-written (append-bypassing) record with a dangling supersede target.
    _write_jsonl(tmp_path / rp.BLOCK_REVIEWS_FILENAME, [
        _review("blocks-id:b1", "safety_declination", review_id="r1",
                supersedes="ghost-review-id")])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_duplicate_review_id_across_ledger_is_integrity_error(tmp_path):
    """final-gate2 F4: two reviews sharing a review_id make a supersede target
    ambiguous — the whole ledger is rejected."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1"),
        _block("aita:m:item1:side_a", cls="model_signal", block_id="b2")])
    _write_jsonl(tmp_path / rp.BLOCK_REVIEWS_FILENAME, [
        _review("blocks-id:b1", "safety_declination", review_id="dup"),
        _review("blocks-id:b2", "safety_declination", review_id="dup")])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_supersession_cycle_is_integrity_error(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    _write_jsonl(tmp_path / rp.BLOCK_REVIEWS_FILENAME, [
        _review("blocks-id:b1", "needs_escalation", review_id="r1", supersedes="r2"),
        _review("blocks-id:b1", "safety_declination", review_id="r2", supersedes="r1"),
    ])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_invalid_disposition_rejected_by_projection(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    _write_jsonl(tmp_path / rp.BLOCK_REVIEWS_FILENAME, [
        _review("blocks-id:b1", "bogus_disposition", review_id="r1")])
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_malformed_review_line_rejected_by_projection(tmp_path):
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    (tmp_path / rp.BLOCK_REVIEWS_FILENAME).write_text(
        json.dumps(_review("blocks-id:b1", "safety_declination", review_id="r1")) + "\n"
        + "this is not json{\n")
    with pytest.raises(rp.ReviewValidationError):
        rp.project(tmp_path)


def test_append_stamps_full_v2_schema_from_fact(tmp_path):
    """A minimal review (no reviewed_at/module/model/category) is completed from
    the target fact at append time (final-gate F4c)."""
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", category="cyber", block_id="b1")])
    written = rp.append_review(tmp_path, {
        "event_ref": "blocks-id:b1", "disposition": "safety_declination",
        "reviewer": "t", "rationale": "r"})
    assert written["schema_version"] == rp.REVIEW_SCHEMA_VERSION
    assert written["review_id"]
    assert written["reviewed_at"]
    assert written["module"] == "aita"
    assert written["model"] == "m"
    assert written["category"] == "cyber"


def test_review_lock_is_not_stolen_even_when_old(tmp_path):
    """final-gate F4: no blind time-based steal.  A lock aged far past the old
    120s threshold is still NOT stolen — append waits then errors cleanly."""
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="model_signal", block_id="b1")])
    lock = tmp_path / rp.BLOCK_REVIEWS_LOCK_FILENAME
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        old = time.time() - 10_000  # ~2.7h old, well past the removed 120s steal
        os.utime(str(lock), (old, old))
        with pytest.raises(rp.ReviewLockError):
            rp.append_review(tmp_path, _review("blocks-id:b1", "safety_declination"),
                             lock_timeout=0.3)
        assert lock.exists()  # the held lock was not stolen
    finally:
        os.close(fd)
        lock.unlink()


def test_bare_unknown_block_no_review_does_not_terminalize_unit(tmp_path):
    """A bare unknown-class block with NO review must not drive the owed view to
    terminal/unresolved (byte-identity with pre-projection owed_units, which
    ignored non-model_signal blocks).  Its unresolved status still shows on the
    fact view for the gate."""
    _contract(tmp_path, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json",
         "planned_turns": 2}])
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", cls="unknown", category="ambiguous_403",
               block_id="b1")])
    # No transcript -> owed via the predicate path (block ignored by owed view).
    result = rp.project(tmp_path)
    uv = result.units_by_id["aita:m:item0:side_a"]
    assert uv.state == "owed"
    assert uv.reason == "artifact missing"
    # But the fact itself is unresolved (gate/T7 will act on this).
    assert result.events_by_ref["blocks-id:b1"].resolution_status == "unresolved"


def test_events_by_ref_covers_blocks_and_events(tmp_path):
    _write_jsonl(tmp_path / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id="b1")])
    _write_jsonl(tmp_path / "RUN_EVENTS.jsonl", [_event(event_id="e1")])
    result = rp.project(tmp_path)
    assert "blocks-id:b1" in result.events_by_ref
    assert "events-id:e1" in result.events_by_ref
    # Non-attempt_failure_classified events are not facts.
    assert all(fv.fact.get("event") in (None, "attempt_failure_classified")
               for fv in result.events_by_ref.values())
