"""Tests for `bench review` verb (Task 020-T6, plan §3 T6 + D4/D5).

Library-level tests per test_bench_registry.py pattern (no subprocess).
Exercises both LIST mode and DISPOSITION mode via the public `bench.review()`
function.

Fixture helpers reuse the shapes from test_review_projection.py so that the
projecting primitives stay authoritative; this file only tests the CLI layer.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from suite_tools import bench
from suite_tools import review_projection as rp
from suite_tools.run_contract import provenance_hashes


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _contract(run: Path, module: str, units: list[dict], *, run_id: str | None = None) -> None:
    if run_id is None:
        run_id = run.name
    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": run_id,
        "modules": [{"module": module, "stage": "generation", "expected_units": units}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "x",
             "route": "openrouter", "effort": "high", "profile": None}]},
    }))
    (run / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1", "attempt_number": 1,
        "status": "running", "started_at": "2026-01-01T00:00:00Z",
    }))


def _block(unit_id, *, attempt=1, category="refusal", cls="model_signal",
           block_id=None, provider=None, provider_code=None,
           native_finish_reason=None, signal_source=None, billed_attempts=None,
           raw_body_excerpt=None, **extra) -> dict:
    rec = {
        "schema_version": "benchmark-block-v2",
        "timestamp": "2026-01-01T00:00:00Z",
        "module": "aita", "stage": "gen", "attempt_number": attempt, "model": "m",
        "unit_id": unit_id, "unit": {"item_idx": 0, "side": "side_a"},
        "evidence_class": cls, "category": category,
    }
    if block_id is not None:
        rec["block_id"] = block_id
    if provider is not None:
        rec["provider"] = provider
    if provider_code is not None:
        rec["provider_code"] = provider_code
    if native_finish_reason is not None:
        rec["native_finish_reason"] = native_finish_reason
    if signal_source is not None:
        rec["signal_source"] = signal_source
    if billed_attempts is not None:
        rec["billed_attempts"] = billed_attempts
    if raw_body_excerpt is not None:
        rec["raw_body_excerpt"] = raw_body_excerpt
    rec.update(extra)
    return rec


def _event(*, unit_id=None, attempt=1, category="refusal", cls="model_signal",
           event_id=None, action="halt", provider=None, provider_code=None,
           native_finish_reason=None, signal_source=None, billed_attempts=None,
           raw_body_excerpt=None, **extra) -> dict:
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
    if provider is not None:
        rec["provider"] = provider
    if provider_code is not None:
        rec["provider_code"] = provider_code
    if native_finish_reason is not None:
        rec["native_finish_reason"] = native_finish_reason
    if signal_source is not None:
        rec["signal_source"] = signal_source
    if billed_attempts is not None:
        rec["billed_attempts"] = billed_attempts
    if raw_body_excerpt is not None:
        rec["raw_body_excerpt"] = raw_body_excerpt
    rec.update(extra)
    return rec


def _review_rec(event_ref, disposition, *, review_id=None, resolved_category=None,
                supersedes=None) -> dict:
    rec = {
        "schema_version": rp.REVIEW_SCHEMA_VERSION,
        "review_id": review_id or uuid.uuid4().hex,
        "event_ref": event_ref, "module": "aita", "model": "m",
        "category": "refusal", "disposition": disposition,
        "reviewer": "tester", "rationale": "because",
        "reviewed_at": "2026-01-02T00:00:00Z",
    }
    if resolved_category is not None:
        rec["resolved_category"] = resolved_category
    if supersedes is not None:
        rec["supersedes_review_id"] = supersedes
    return rec


def _make_run(root: Path, name: str, module: str = "aita") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    _contract(d, module, [{"unit_id": f"{module}:m:item0:side_a",
                           "expected_transcript_path": "t.json",
                           "planned_turns": 2}])
    return d


# ---------------------------------------------------------------------------
# LIST mode — basic surface
# ---------------------------------------------------------------------------


def test_list_surfaces_unresolved_block_fact(tmp_path):
    """LIST mode shows a fact from BLOCKS with no active review as unreviewed."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid,
               provider="anthropic", provider_code="403",
               native_finish_reason="stop", signal_source="http_status",
               billed_attempts=1, raw_body_excerpt="error body")
    ])
    result = bench.review(roots=[tmp_path])
    assert result["schema_version"] == bench.SCHEMA_VERSION
    rows = result["rows"]
    assert len(rows) == 1
    row = rows[0]
    # run context
    assert row["run_id"] == "run1"
    assert "run1" in row["run_path"]
    # fact identity
    assert row["event_ref"] == f"blocks-id:{bid}"
    assert row["source"] == "blocks"
    assert row["scope"] == "unit"
    assert row["unit_id"] == "aita:m:item0:side_a"
    assert row["evidence_class"] == "model_signal"
    assert row["category"] == "refusal"
    assert row["attempt_number"] == 1
    # v2 snapshot fields
    assert row["provider"] == "anthropic"
    assert row["provider_code"] == "403"
    assert row["native_finish_reason"] == "stop"
    assert row["signal_source"] == "http_status"
    assert row["billed_attempts"] == 1
    assert row["raw_body_excerpt"] == "error body"
    # disposition status
    assert row["disposition_status"] == "unreviewed"
    assert row["active_review"] is None


def test_list_surfaces_unresolved_event_fact(tmp_path):
    """LIST mode shows a fact from RUN_EVENTS (member-scoped) as unreviewed."""
    run = _make_run(tmp_path, "run1")
    eid = uuid.uuid4().hex
    _write_jsonl(run / "RUN_EVENTS.jsonl", [
        _event(event_id=eid, cls="unknown", category="ambiguous_403")
    ])
    result = bench.review(roots=[tmp_path])
    rows = result["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["event_ref"] == f"events-id:{eid}"
    assert row["source"] == "events"
    assert row["scope"] == "member"
    assert row["unit_id"] is None
    assert row["evidence_class"] == "unknown"
    assert row["disposition_status"] == "unreviewed"


def test_list_excludes_resolved_by_default(tmp_path):
    """A resolved (safety_declination) fact is hidden unless --all."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)
    ])
    rp.append_review(run, _review_rec(f"blocks-id:{bid}", "safety_declination"))

    # Without --all: no rows
    result = bench.review(roots=[tmp_path])
    assert result["rows"] == []

    # With --all: one row
    result_all = bench.review(roots=[tmp_path], include_resolved=True)
    assert len(result_all["rows"]) == 1
    row = result_all["rows"][0]
    assert row["disposition_status"] == "resolved"


def test_list_needs_escalation_counts_as_unresolved(tmp_path):
    """needs_escalation review keeps the fact as unresolved (shows without --all)."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)
    ])
    rp.append_review(run, _review_rec(f"blocks-id:{bid}", "needs_escalation"))

    result = bench.review(roots=[tmp_path])
    assert len(result["rows"]) == 1
    assert result["rows"][0]["disposition_status"] == "needs_escalation"


def test_list_filter_by_class(tmp_path):
    """--class filter restricts to only facts of that evidence_class."""
    run = _make_run(tmp_path, "run1")
    bid1 = uuid.uuid4().hex
    bid2 = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid1, cls="model_signal"),
        _block("aita:m:item0:side_a", block_id=bid2, cls="unknown", category="ambiguous_403"),
    ])
    result = bench.review(roots=[tmp_path], evidence_class="unknown")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["evidence_class"] == "unknown"


def test_list_filter_by_scope(tmp_path):
    """--scope filter restricts to unit / member / unmappable."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)
    ])
    _write_jsonl(run / "RUN_EVENTS.jsonl", [
        _event(event_id=eid)  # member-scoped (no unit_id)
    ])

    unit_result = bench.review(roots=[tmp_path], scope="unit")
    assert all(r["scope"] == "unit" for r in unit_result["rows"])
    assert len(unit_result["rows"]) == 1

    member_result = bench.review(roots=[tmp_path], scope="member")
    assert all(r["scope"] == "member" for r in member_result["rows"])
    assert len(member_result["rows"]) == 1


def test_list_filter_by_run(tmp_path):
    """--run filter restricts to only that run directory."""
    root = tmp_path / "results"
    run1 = _make_run(root, "run1")
    run2 = _make_run(root, "run2")
    bid1 = uuid.uuid4().hex
    bid2 = uuid.uuid4().hex
    _write_jsonl(run1 / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid1)])
    _write_jsonl(run2 / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid2)])

    result = bench.review(roots=[root], run_dir=run1)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["run_id"] == "run1"


def test_list_both_sources_in_one_run(tmp_path):
    """Facts from BLOCKS and RUN_EVENTS both surface in the queue."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    _write_jsonl(run / "RUN_EVENTS.jsonl", [_event(event_id=eid)])
    result = bench.review(roots=[tmp_path])
    sources = {r["source"] for r in result["rows"]}
    assert sources == {"blocks", "events"}


def test_list_active_review_included_in_row(tmp_path):
    """The active review dict (not just status) is present on the row."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    rp.append_review(run, _review_rec(f"blocks-id:{bid}", "needs_escalation"))

    result = bench.review(roots=[tmp_path])
    row = result["rows"][0]
    assert row["active_review"] is not None
    assert row["active_review"]["disposition"] == "needs_escalation"


# ---------------------------------------------------------------------------
# LIST mode — legacy grandfathering
# ---------------------------------------------------------------------------


def test_list_v1_backfill_review_shows_as_resolved(tmp_path):
    """A v1 backfill safety_declination review marks the block as resolved."""
    run = _make_run(tmp_path, "run1")
    block = _block("aita:m:item0:side_a", cls="model_signal", category="cyber")
    block.update({"schema_version": "benchmark-block-v1", "backfilled": True,
                  "backfill_id": "retro-audit-20260721"})
    block.pop("block_id", None)
    _write_jsonl(run / "BLOCKS.jsonl", [block])
    v1_review = {
        "schema_version": rp.REVIEW_SCHEMA_VERSION_V1,
        "module": "aita", "model": "m",
        "unit_id": "aita:m:item0:side_a", "category": "cyber",
        "backfill_id": "retro-audit-20260721", "disposition": "safety_declination",
        "reviewer": "retro-audit-20260721",
    }
    _write_jsonl(run / "BLOCK_REVIEWS.jsonl", [v1_review])

    # Default (exclude resolved): no rows
    result = bench.review(roots=[tmp_path])
    assert result["rows"] == []

    # With --all: one row, resolved
    result_all = bench.review(roots=[tmp_path], include_resolved=True)
    assert len(result_all["rows"]) == 1
    assert result_all["rows"][0]["disposition_status"] == "resolved"


def test_list_no_unit_id_block_has_scope_unmappable(tmp_path):
    """A BLOCKS fact without a unit_id is scoped as unmappable_legacy."""
    run = tmp_path / "run1"
    run.mkdir(parents=True, exist_ok=True)
    _contract(run, "aita", [])  # no units in contract
    block = _block("aita:m:item0:side_a")
    block.pop("unit_id", None)
    block.pop("block_id", None)
    _write_jsonl(run / "BLOCKS.jsonl", [block])

    result = bench.review(roots=[tmp_path])
    assert len(result["rows"]) == 1
    assert result["rows"][0]["scope"] == "unmappable_legacy"
    assert result["rows"][0]["unit_id"] is None


def test_list_filter_scope_unmappable_alias(tmp_path):
    """--scope unmappable maps to unmappable_legacy (CLI alias)."""
    run = tmp_path / "run1"
    run.mkdir(parents=True, exist_ok=True)
    _contract(run, "aita", [])
    block = _block("aita:m:item0:side_a")
    block.pop("unit_id", None)
    block.pop("block_id", None)
    _write_jsonl(run / "BLOCKS.jsonl", [block])

    result_alias = bench.review(roots=[tmp_path], scope="unmappable")
    result_full = bench.review(roots=[tmp_path], scope="unmappable_legacy")
    assert len(result_alias["rows"]) == 1
    assert len(result_full["rows"]) == 1


# ---------------------------------------------------------------------------
# DISPOSITION mode — basic append + re-projection
# ---------------------------------------------------------------------------


def test_disposition_appends_and_echoes_new_state(tmp_path):
    """Disposition mode appends the review and returns the new projected state."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)
    ])
    ref = f"blocks-id:{bid}"
    result = bench.review(
        run_dir=run,
        event_ref=ref,
        disposition="safety_declination",
        reviewer="tester",
        reason="confirmed safety block",
    )
    assert result["schema_version"] == bench.SCHEMA_VERSION
    # written record echoed
    written = result["written_review"]
    assert written["event_ref"] == ref
    assert written["disposition"] == "safety_declination"
    assert written["reviewer"] == "tester"
    assert "review_id" in written
    # new effective state from re-projection
    new_state = result["new_effective_state"]
    assert new_state["resolution_status"] == "resolved"
    assert new_state["effective_class"] == "model_signal"
    assert new_state["disposition"] == "safety_declination"


def test_disposition_safety_declination_on_ambiguous_fact_requires_resolved_category(tmp_path):
    """safety_declination on an unknown/ambiguous_403 fact requires resolved_category."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid, cls="unknown", category="ambiguous_403")
    ])
    ref = f"blocks-id:{bid}"
    with pytest.raises(Exception) as exc_info:
        bench.review(
            run_dir=run,
            event_ref=ref,
            disposition="safety_declination",
            reviewer="tester",
            reason="confirmed",
        )
    assert "resolved_category" in str(exc_info.value).lower() or \
           "unclassified" in str(exc_info.value).lower() or \
           "ambiguous" in str(exc_info.value).lower()


def test_disposition_safety_declination_with_resolved_category_succeeds(tmp_path):
    """safety_declination with resolved_category on an ambiguous fact succeeds."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid, cls="unknown", category="ambiguous_403")
    ])
    ref = f"blocks-id:{bid}"
    result = bench.review(
        run_dir=run,
        event_ref=ref,
        disposition="safety_declination",
        reviewer="tester",
        reason="confirmed guardrail",
        resolved_category="guardrail_permission_denied",
    )
    assert result["written_review"]["resolved_category"] == "guardrail_permission_denied"
    assert result["new_effective_state"]["effective_category"] == "guardrail_permission_denied"


def test_disposition_retry_on_unmappable_rejected_with_t5_error(tmp_path):
    """retry on unmappable-legacy fact is rejected (T5's error surfaced, not re-encoded)."""
    run = tmp_path / "run1"
    run.mkdir(parents=True, exist_ok=True)
    _contract(run, "aita", [])
    block = _block("aita:m:item0:side_a")
    block.pop("unit_id", None)
    block.pop("block_id", None)
    _write_jsonl(run / "BLOCKS.jsonl", [block])
    raw = (run / "BLOCKS.jsonl").read_text().splitlines()[0]
    ref = f"blocks-line:1:{rp.line_sha8(raw.encode('utf-8'))}"
    with pytest.raises(Exception) as exc_info:
        bench.review(
            run_dir=run,
            event_ref=ref,
            disposition="retry",
            reviewer="tester",
            reason="try again",
        )
    err = str(exc_info.value)
    assert "retry" in err.lower() or "unmappable" in err.lower()


def test_disposition_duplicate_active_errors_without_supersede(tmp_path):
    """A second active review for the same fact without --supersede is an error."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    ref = f"blocks-id:{bid}"
    bench.review(run_dir=run, event_ref=ref, disposition="safety_declination",
                 reviewer="tester", reason="first")
    with pytest.raises(Exception) as exc_info:
        bench.review(run_dir=run, event_ref=ref, disposition="needs_escalation",
                     reviewer="tester", reason="second without supersede")
    assert "active" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()


def test_disposition_supersede_chain(tmp_path):
    """--supersede chain: new review supersedes the old one and becomes the head."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    ref = f"blocks-id:{bid}"

    # First review
    r1 = bench.review(run_dir=run, event_ref=ref, disposition="safety_declination",
                      reviewer="tester", reason="first")
    first_review_id = r1["written_review"]["review_id"]

    # Second review superseding the first
    r2 = bench.review(run_dir=run, event_ref=ref, disposition="needs_escalation",
                      reviewer="tester", reason="escalate",
                      supersede=first_review_id)
    assert r2["written_review"]["supersedes_review_id"] == first_review_id
    # The new effective state reflects the new (superseding) disposition
    assert r2["new_effective_state"]["disposition"] == "needs_escalation"


def test_disposition_with_issue_ref(tmp_path):
    """--issue-ref is persisted in the written review."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    result = bench.review(
        run_dir=run,
        event_ref=f"blocks-id:{bid}",
        disposition="needs_escalation",
        reviewer="tester",
        reason="escalate",
        issue_ref="ISSUE-42",
    )
    assert result["written_review"].get("issue_ref") == "ISSUE-42"


def test_disposition_re_projection_shows_consequence_immediately(tmp_path):
    """After instrument_defect, new_effective_state reflects the reclassification."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid, cls="unknown", category="ambiguous_403")
    ])
    ref = f"blocks-id:{bid}"
    result = bench.review(run_dir=run, event_ref=ref, disposition="instrument_defect",
                          reviewer="tester", reason="instrument issue")
    new_state = result["new_effective_state"]
    assert new_state["effective_class"] == "instrument_defect"
    assert new_state["resolution_status"] == "resolved"


# ---------------------------------------------------------------------------
# Schema / structure checks
# ---------------------------------------------------------------------------


def test_review_returns_schema_version(tmp_path):
    result = bench.review(roots=[tmp_path])
    assert result["schema_version"] == bench.SCHEMA_VERSION


def test_review_list_empty_when_no_runs(tmp_path):
    result = bench.review(roots=[tmp_path])
    assert result["rows"] == []
    assert "scan_warnings" in result


# ---------------------------------------------------------------------------
# gate_blocking + resolution_status on rows (review fix #1)
# ---------------------------------------------------------------------------


def test_list_row_gate_blocking_unknown_class(tmp_path):
    """A bare unknown-class fact (no review) is gate_blocking=True (unresolved)."""
    run = _make_run(tmp_path, "run1")
    eid = uuid.uuid4().hex
    _write_jsonl(run / "RUN_EVENTS.jsonl", [
        _event(event_id=eid, cls="unknown", category="ambiguous_403")
    ])
    result = bench.review(roots=[tmp_path])
    row = result["rows"][0]
    assert row["resolution_status"] == "unresolved"
    assert row["gate_blocking"] is True


def test_list_row_gate_blocking_false_for_bare_model_signal(tmp_path):
    """A bare model_signal block (no review) shows unreviewed but gate_blocking=False."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)  # cls="model_signal" default
    ])
    result = bench.review(roots=[tmp_path])
    row = result["rows"][0]
    assert row["disposition_status"] == "unreviewed"
    assert row["resolution_status"] == "resolved"   # projection: known class, no review
    assert row["gate_blocking"] is False


def test_list_row_gate_blocking_needs_escalation_non_unknown(tmp_path):
    """A needs_escalation review on a known-class fact is still gate_blocking=True."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid, cls="model_signal", category="refusal")
    ])
    rp.append_review(run, _review_rec(f"blocks-id:{bid}", "needs_escalation"))
    result = bench.review(roots=[tmp_path])
    row = result["rows"][0]
    assert row["disposition_status"] == "needs_escalation"
    assert row["gate_blocking"] is True


# ---------------------------------------------------------------------------
# Disposition-mode echo for line-form refs on v2 facts (review fix #3)
# ---------------------------------------------------------------------------


def test_disposition_line_form_ref_on_v2_fact_echoes_true_state(tmp_path):
    """Line-form ref targeting a v2 fact (with block_id) echoes the true
    new_effective_state, not resolution_status='unknown'.

    bench.review normalizes the line-form ref to id-form before storing so that:
    (a) the projection can match the review to the fact (which is keyed by id-form),
    (b) the echo lookup finds the updated FactView with the new disposition.
    The written_review carries the normalized id-form ref.
    """
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [
        _block("aita:m:item0:side_a", block_id=bid)
    ])
    # Build the line-form ref for the physical line.
    raw_line = (run / "BLOCKS.jsonl").read_text().splitlines()[0]
    line_ref = f"blocks-line:1:{rp.line_sha8(raw_line.encode('utf-8'))}"
    id_ref = f"blocks-id:{bid}"  # canonical form bench.review normalizes to

    result = bench.review(
        run_dir=run,
        event_ref=line_ref,
        disposition="needs_escalation",
        reviewer="tester",
        reason="escalate",
    )
    # Written review is stored with the canonical id-form ref.
    assert result["written_review"]["event_ref"] == id_ref, (
        "expected line-form ref to be normalized to id-form in stored review"
    )
    # The new_effective_state must reflect the actual projection state.
    new_state = result["new_effective_state"]
    assert new_state["resolution_status"] != "unknown", (
        f"line-form ref lookup missed; got resolution_status='unknown'. "
        f"canonical_ref={new_state.get('event_ref')}"
    )
    assert new_state["event_ref"] == id_ref
    assert new_state["disposition"] == "needs_escalation"
    assert new_state["resolution_status"] == "unresolved"


def test_list_row_has_evidence_pointer(tmp_path):
    """evidence_pointer field points to the physical file."""
    run = _make_run(tmp_path, "run1")
    bid = uuid.uuid4().hex
    _write_jsonl(run / "BLOCKS.jsonl", [_block("aita:m:item0:side_a", block_id=bid)])
    result = bench.review(roots=[tmp_path])
    row = result["rows"][0]
    assert "evidence_pointer" in row
    assert "BLOCKS.jsonl" in row["evidence_pointer"] or "blocks" in row["evidence_pointer"].lower()


# ---------------------------------------------------------------------------
# Three-surface consistency: instrument_defect and safety_declination
# (Finding 7 — SEV-2)
# ---------------------------------------------------------------------------
# All three surfaces (bench review gate_blocking, bench blockers, bundle gate)
# must agree: instrument_defect is publication-blocking; safety_declination clears
# all three.  One shared run fixture is exercised through each surface in turn.


def _make_gate_run(
    root: Path,
    name: str,
    *,
    evidence_class: str,
    disposition: str | None = None,
) -> Path:
    """Create a completed run with a member-scoped halt event of the given class.

    Member-scoped (no unit_id) events are always gate-relevant for the bundle,
    so this fixture exercises the gate without requiring a won-unit collision.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    # RUN_CONTRACT.json — minimal valid shape
    unit_id = "aita:m:item0:side_a"
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": name,
        "modules": [{"module": "aita", "stage": "generation", "expected_units": [
            {"unit_id": unit_id, "model_key": "m",
             "expected_transcript_path": "t.json", "planned_turns": 1}
        ]}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "gpt-4",
             "condition_hash": "sha256:" + "a" * 64,
             "route_hash": "sha256:" + "b" * 64,
             "route": "openrouter", "effort": "high", "profile": None}
        ]},
    }
    contract["provenance"] = provenance_hashes(contract)
    (d / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    # RUN_STATUS.json — completed so the bundle gate doesn't fire for "not completed"
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "attempt_number": 1,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
    }))
    # Scoring artifact so the unit state is "completed" in the projection
    score = {"verdict_alignment_a_majority": True, "verdict_alignment_a": 1.0}
    (d / "t.json").write_text(json.dumps(
        {"completed": True, "turns": [{"model_response": "ok"}], **score,
         "condition_id": "cond-m", "condition_hash": "sha256:" + "a" * 64,
         "route_hash": "sha256:" + "b" * 64,
         "conversation": [{"role": "user", "content": "hi"}]}
    ))
    (d / "FINAL_RESULTS.json").write_text(json.dumps(
        {"scores": {f"m_item0": score}}
    ))
    # RUN_EVENTS.jsonl — member-scoped halt (no unit_id → scope="member")
    eid = uuid.uuid4().hex
    (d / "RUN_EVENTS.jsonl").write_text(json.dumps({
        "schema_version": "benchmark-run-monitor-v1",
        "sequence": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "module": "aita",
        "stage": "gen",
        "event": "attempt_failure_classified",
        "attempt_number": 1,
        "action": "halt",
        "evidence_class": evidence_class,
        "category": "instrument",
        "failure_reason": "test",
        "model": "m",
        "event_id": eid,
    }) + "\n")
    # Apply a review if requested
    if disposition is not None:
        rp.append_review(d, {
            "schema_version": rp.REVIEW_SCHEMA_VERSION,
            "review_id": uuid.uuid4().hex,
            "event_ref": f"events-id:{eid}",
            "module": "aita",
            "model": "m",
            "category": "instrument",
            "disposition": disposition,
            "resolved_category": "safety_declination" if disposition == "safety_declination" else None,
            "reviewer": "tester",
            "rationale": "test",
            "reviewed_at": "2026-01-02T00:00:00Z",
        })
    return d


def _make_gate_experiment(tmp_path: Path, run: Path) -> Path:
    """Wrap a single run dir in a minimal experiment for bundle.emit()."""
    exp = tmp_path / "exp"
    exp.mkdir(exist_ok=True)
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "gate-test",
        "title": "gate test",
        "instrument": {"modules": ["aita"], "hashes": {}},
        "conditions": [],
        "target": {"n_items": 1},
        "members": [
            {"path": str((run / "RUN_CONTRACT.json").resolve()), "role": "pilot"},
        ],
    }))
    return exp


# --- instrument_defect: three surfaces all show as blocking ---


def test_three_surface_instrument_defect_gate_blocking_in_review_rows(tmp_path):
    """bench review: instrument_defect event → gate_blocking=True in list row.

    Confirms that gate_blocking derives from is_publication_blocking (not
    resolution_status, which is 'resolved' for known-class facts).
    """
    run = _make_gate_run(tmp_path / "runs", "run1", evidence_class="instrument_defect")
    result = bench.review(roots=[tmp_path / "runs"])
    assert result["rows"], "expected at least one row"
    rows_with_defect = [r for r in result["rows"] if r["evidence_class"] == "instrument_defect"]
    assert rows_with_defect, "instrument_defect event should appear in review rows"
    assert all(r["gate_blocking"] is True for r in rows_with_defect), (
        f"gate_blocking should be True for instrument_defect; got: {rows_with_defect}"
    )


def test_three_surface_instrument_defect_still_listed_in_blockers(tmp_path):
    """bench blockers: instrument_defect halt event must NOT be suppressed.

    The old _BLOCKER_RESOLVING_DISPOSITIONS included instrument_defect, which
    caused it to be suppressed from blockers. Now only safety_declination clears.
    """
    run = _make_gate_run(tmp_path / "runs", "run1", evidence_class="instrument_defect")
    result = bench.blockers(roots=[tmp_path / "runs"])
    assert result["blockers"], (
        "instrument_defect halt must appear in blockers output (not suppressed)"
    )
    classes = [b["evidence_class"] for b in result["blockers"]]
    assert "instrument_defect" in classes, (
        f"expected instrument_defect in blockers; got: {classes}"
    )


def test_three_surface_instrument_defect_gates_bundle(tmp_path):
    """bundle.emit: instrument_defect event on a member gates the bundle.

    The member-scoped instrument_defect event is always gate-relevant
    (scope='member'); is_publication_blocking must catch it and raise.
    """
    from suite_tools import bundle

    run = _make_gate_run(tmp_path / "runs", "run1", evidence_class="instrument_defect")
    exp = _make_gate_experiment(tmp_path, run)
    with pytest.raises(ValueError, match="instrument_defect"):
        bundle.emit(exp, out_dir=tmp_path / "out")


# --- safety_declination: three surfaces all show as cleared ---


def test_three_surface_safety_declination_gate_blocking_false_in_review_rows(tmp_path):
    """bench review: safety_declination review → gate_blocking=False, row hidden.

    A safety_declination review makes the fact triage-resolved; it should not
    appear in the default list view, and gate_blocking should be False.
    """
    run = _make_gate_run(
        tmp_path / "runs", "run1",
        evidence_class="unknown",
        disposition="safety_declination",
    )
    # Default (exclude resolved): row should be hidden
    result_default = bench.review(roots=[tmp_path / "runs"])
    event_rows = [r for r in result_default["rows"] if r.get("source") == "events"]
    assert not event_rows, (
        "safety_declination-reviewed row should be hidden from default list view"
    )
    # With include_resolved=True: row appears with gate_blocking=False
    result_all = bench.review(roots=[tmp_path / "runs"], include_resolved=True)
    event_rows_all = [r for r in result_all["rows"] if r.get("source") == "events"]
    assert event_rows_all, "safety_declination row should appear with include_resolved=True"
    assert all(r["gate_blocking"] is False for r in event_rows_all), (
        f"gate_blocking should be False for safety_declination; got: {event_rows_all}"
    )


def test_three_surface_safety_declination_clears_blockers(tmp_path):
    """bench blockers: safety_declination halt event is suppressed (not listed).

    Only non-publication-blocking adjudications (is_publication_blocking=None)
    drop off the blockers list; safety_declination is the canonical case.
    """
    run = _make_gate_run(
        tmp_path / "runs", "run1",
        evidence_class="unknown",
        disposition="safety_declination",
    )
    result = bench.blockers(roots=[tmp_path / "runs"])
    assert not result["blockers"], (
        "safety_declination halt should be suppressed from blockers output; "
        f"got: {result['blockers']}"
    )


def test_three_surface_safety_declination_clears_bundle_gate(tmp_path):
    """bundle.emit: safety_declination review clears the bundle gate.

    After a safety_declination review on an unknown-class fact, the bundle gate
    should not fire (is_publication_blocking returns None).
    """
    from suite_tools import bundle

    run = _make_gate_run(
        tmp_path / "runs", "run1",
        evidence_class="unknown",
        disposition="safety_declination",
    )
    exp = _make_gate_experiment(tmp_path, run)
    # Should complete without raising (gate cleared)
    result = bundle.emit(exp, out_dir=tmp_path / "out")
    assert result.get("bundle_dir"), "bundle should emit successfully after safety_declination"


# ---------------------------------------------------------------------------
# F6 (SEV-2) — gate_blocking must reflect unit and member layers, not just
# the fact-level predicate.  A retry-disposed fact with undischarged unit
# state shows gate_blocking=False from is_publication_blocking alone but
# gate_blocking=True from _compute_gate_blocking (unit layer).
# ---------------------------------------------------------------------------


def _make_retry_run(root: Path, name: str, *, discharged: bool) -> Path:
    """Create a run with a unit-scoped block under a retry review.

    discharged=False: run attempt_number=1, block attempt=1 → pending_retry.
    discharged=True:  artifact has attempt_number=2 → carrier discharged → completed.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    unit_id = "aita:m:item0:side_a"
    bid = uuid.uuid4().hex
    (d / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": name,
        "modules": [{"module": "aita", "stage": "generation", "expected_units": [
            {"unit_id": unit_id, "expected_transcript_path": "t.json", "planned_turns": 1}
        ]}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "gpt-4",
             "route": "openrouter", "effort": "high", "profile": None}
        ]},
    }))
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "attempt_number": 1,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
    }))
    # Write the block at attempt 1
    _write_jsonl(d / "BLOCKS.jsonl", [
        _block(unit_id, block_id=bid, attempt=1, cls="model_signal", category="refusal")
    ])
    # Add retry review for this block
    rp.append_review(d, _review_rec(f"blocks-id:{bid}", "retry"))
    if discharged:
        # Artifact stamped with attempt_number=2 — strictly later than the block's
        # attempt_number=1, which discharges the retry carrier.
        (d / "t.json").write_text(json.dumps({
            "completed": True,
            "attempt_number": 2,
            "turns": [{"model_response": "ok"}],
            "conversation": [{"role": "user", "content": "hi"}],
        }))
    # (no artifact when undischarged — carrier_discharged returns False)
    return d


def test_f6_undischarged_retry_gate_blocking_true_unit_layer(tmp_path):
    """Undischarged retry block → gate_blocking=True, gate_reason='unit'.

    is_publication_blocking returns None for retry facts (no retry clause at
    fact level), but the UnitView state is pending_retry — the unit layer of
    _compute_gate_blocking must catch this and set gate_blocking=True.

    Reproduction: Sol F6 — operator sees gate_blocking=False; bundle refuses.
    """
    run = _make_retry_run(tmp_path / "runs", "run1", discharged=False)
    result = bench.review(roots=[tmp_path / "runs"], include_resolved=True)
    block_rows = [r for r in result["rows"] if r.get("source") == "blocks"]
    assert block_rows, "expected at least one block row"
    row = block_rows[0]
    # Fact-level predicate returns None (retry is not in is_publication_blocking)
    # so gate_blocking MUST come from the unit layer.
    assert row["gate_blocking"] is True, (
        f"undischarged retry block must have gate_blocking=True; "
        f"got gate_blocking={row['gate_blocking']}, gate_reason={row.get('gate_reason')}"
    )
    assert row.get("gate_reason") == "unit", (
        f"expected gate_reason='unit' for undischarged retry; "
        f"got gate_reason={row.get('gate_reason')!r}"
    )


def test_f6_discharged_retry_gate_blocking_false(tmp_path):
    """Discharged retry (later-attempt artifact) → gate_blocking=False.

    After discharge the UnitView state is 'completed'; no layer fires.
    """
    run = _make_retry_run(tmp_path / "runs", "run1", discharged=True)
    result = bench.review(roots=[tmp_path / "runs"], include_resolved=True)
    block_rows = [r for r in result["rows"] if r.get("source") == "blocks"]
    assert block_rows, "expected at least one block row"
    row = block_rows[0]
    assert row["gate_blocking"] is False, (
        f"discharged retry block must have gate_blocking=False; "
        f"got gate_blocking={row['gate_blocking']}, gate_reason={row.get('gate_reason')}"
    )
    assert row.get("gate_reason") is None, (
        f"expected gate_reason=None for discharged retry; "
        f"got gate_reason={row.get('gate_reason')!r}"
    )


def test_f6_member_obligation_makes_clean_fact_gate_blocking(tmp_path):
    """Pending member obligation → otherwise-clean fact shows gate_blocking=True, gate_reason='member'.

    A member-scoped retry obligation blocks the whole member.  A model_signal
    block in the same run has no fact-level or unit-level blocking issues, but
    the member layer of _compute_gate_blocking must flag it as gate_blocking=True.
    """
    d = tmp_path / "runs" / "run1"
    d.mkdir(parents=True, exist_ok=True)
    unit_id = "aita:m:item0:side_a"
    bid = uuid.uuid4().hex
    eid = uuid.uuid4().hex

    (d / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "run1",
        "modules": [{"module": "aita", "stage": "generation", "expected_units": [
            {"unit_id": unit_id, "expected_transcript_path": "t.json", "planned_turns": 1}
        ]}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "gpt-4",
             "route": "openrouter", "effort": "high", "profile": None}
        ]},
    }))
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "attempt_number": 1,       # no later attempt → member obligation unfulfilled
        "status": "running",
        "started_at": "2026-01-01T00:00:00Z",
    }))

    # Clean unit-scoped model_signal block (no review, unit will be terminal_model_signal)
    _write_jsonl(d / "BLOCKS.jsonl", [
        _block(unit_id, block_id=bid, attempt=1, cls="model_signal", category="refusal")
    ])
    # Member-scoped halt event with retry review → unfulfilled member obligation
    # (no unit_id → scope="member")
    _write_jsonl(d / "RUN_EVENTS.jsonl", [
        _event(event_id=eid, cls="model_signal", category="refusal", action="halt", attempt=1)
    ])
    rp.append_review(d, _review_rec(f"events-id:{eid}", "retry"))

    result = bench.review(roots=[tmp_path / "runs"], include_resolved=True)
    # The model_signal block row should show gate_blocking=True from member layer
    block_rows = [r for r in result["rows"] if r.get("source") == "blocks"]
    assert block_rows, "expected block row for model_signal block"
    row = block_rows[0]
    # Fact level: is_publication_blocking returns None (model_signal, no review)
    # Unit level: terminal_model_signal is publishable → no blocking
    # Member level: unfulfilled retry obligation → gate_blocking=True
    assert row["gate_blocking"] is True, (
        f"member obligation must make clean fact gate_blocking=True; "
        f"got gate_blocking={row['gate_blocking']}, gate_reason={row.get('gate_reason')}"
    )
    assert row.get("gate_reason") == "member", (
        f"expected gate_reason='member'; got gate_reason={row.get('gate_reason')!r}"
    )
