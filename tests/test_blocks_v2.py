"""T4: BLOCKS v2 + evidence snapshot on both fact sources (plan 020 D3, D10).

RED-first: all tests written before implementation. Verifies:
- record_block with raw_error → all new v2 fields persisted.
- block_id uuid4 uniqueness across two RunMonitor instances writing the same run.
- Sanitization: fake secret redacted in raw_body_excerpt; sha256 over ORIGINAL bytes.
- Truncation: >2000-char body → excerpt capped, digest of full body.
- Mixed v1+v2 fixture file (incl. v1 record with NO unit_id) parsed without error
  by every reader in the D3 inventory (owed_units, score_rows, _blocks_union path,
  backfill).
- attempt_failure_classified events carry event_id + snapshot fields.
- Old-shape (v1) events still parse everywhere (bench.blockers).
"""
from __future__ import annotations

import hashlib
import json
import uuid

import pytest

from suite_tools.run_monitor import (
    BLOCK_SCHEMA_VERSION,
    RunMonitor,
)

MODEL_SIGNAL = "model_signal"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _monitor(tmp_path):
    return RunMonitor(tmp_path, module="aita", stage="generation")


class _FakeErrorBody:
    """Minimal exception with .raw_response dict (mimics ProviderApiError shape)."""
    def __init__(self, body: dict):
        self.raw_response = body

    def __str__(self):
        return f"FakeError: {self.raw_response}"


class _FakeBodyOnly:
    """Exception with .body dict but no .raw_response (mimics SDK PermissionDeniedError)."""
    def __init__(self, body: dict):
        self.body = body

    def __str__(self):
        return "FakeBodyOnly"


# ── Schema version bump ───────────────────────────────────────────────────────

def test_block_schema_version_is_v2():
    """Schema constant is benchmark-block-v2 after T4."""
    assert BLOCK_SCHEMA_VERSION == "benchmark-block-v2"


# ── New v2 fields persisted ───────────────────────────────────────────────────

def test_record_block_v2_new_fields_persisted(tmp_path):
    """record_block with a raw_error produces all new v2 fields in BLOCKS.jsonl."""
    monitor = _monitor(tmp_path)
    raw_body = {"finish_reason": "content_filter", "error": {"code": "content_filter"}}
    evidence = {
        "evidence_class": MODEL_SIGNAL,
        "category": "content_filter",
        "signal_source": "provider-signals-v2",
        "retry_policy": {"kind": "bounded_retry", "max_retries": 1},
    }
    monitor.record_block(
        unit={"item_idx": 1, "side": "side_a"},
        evidence=evidence,
        model="gpt-5.6",
        unit_id="aita:gpt-5.6:item1:side_a",
        raw_error=_FakeErrorBody(raw_body),
        billed_attempts=2,
    )
    lines = (tmp_path / "BLOCKS.jsonl").read_text().splitlines()
    assert len(lines) == 1
    block = json.loads(lines[0])

    assert block["schema_version"] == "benchmark-block-v2"

    # block_id: uuid4 hex (32 hex chars)
    assert "block_id" in block
    assert len(block["block_id"]) == 32
    uuid.UUID(hex=block["block_id"], version=4)  # must be valid uuid4

    # evidence class + category (pre-existing)
    assert block["evidence_class"] == MODEL_SIGNAL
    assert block["category"] == "content_filter"

    # evidence snapshot fields from evidence dict
    assert block["signal_source"] == "provider-signals-v2"
    assert block["retry_policy_kind"] == "bounded_retry"

    # billed_attempts from parameter
    assert block["billed_attempts"] == 2

    # raw body digest fields (from raw_error)
    assert "raw_body_sha256" in block
    assert "raw_body_excerpt" in block
    assert len(block["raw_body_sha256"]) == 64   # sha256 hex digest


def test_record_block_without_raw_error_omits_body_fields(tmp_path):
    """record_block without raw_error omits raw_body_* fields (they are optional)."""
    monitor = _monitor(tmp_path)
    monitor.record_block(
        unit={"item_idx": 1, "side": "side_a"},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "refusal"},
        model="m",
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())
    assert block["schema_version"] == "benchmark-block-v2"
    assert "block_id" in block  # always present
    assert "raw_body_sha256" not in block
    assert "raw_body_excerpt" not in block


def test_record_block_stochastic_flag_threaded(tmp_path):
    """stochastic=True from evidence dict is stored on the block."""
    monitor = _monitor(tmp_path)
    evidence = {
        "evidence_class": MODEL_SIGNAL,
        "category": "SAFETY",
        "signal_source": "provider-signals-v2",
        "stochastic": True,
        "retry_policy": {"kind": "stochastic_retry", "max_retries": 2},
    }
    monitor.record_block(
        unit={"item_idx": 2},
        evidence=evidence,
        model="gemini-3-flash",
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())
    assert block.get("stochastic") is True
    assert block.get("retry_policy_kind") == "stochastic_retry"


# ── block_id uniqueness ───────────────────────────────────────────────────────

def test_block_id_unique_across_two_monitor_instances(tmp_path):
    """Two RunMonitor instances write different block_ids (uuid4, no counter collision)."""
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()

    m1 = RunMonitor(run1, module="aita", stage="gen")
    m2 = RunMonitor(run2, module="aita", stage="gen")

    ev = {"evidence_class": MODEL_SIGNAL, "category": "refusal"}
    m1.record_block(unit={"item_idx": 0}, evidence=ev, model="m")
    m2.record_block(unit={"item_idx": 0}, evidence=ev, model="m")

    b1 = json.loads((run1 / "BLOCKS.jsonl").read_text().strip())
    b2 = json.loads((run2 / "BLOCKS.jsonl").read_text().strip())

    assert b1["block_id"] != b2["block_id"]
    # Both are valid uuid4
    uuid.UUID(hex=b1["block_id"], version=4)
    uuid.UUID(hex=b2["block_id"], version=4)


def test_multiple_blocks_in_same_run_have_unique_ids(tmp_path):
    """Multiple record_block calls in the same monitor produce unique block_ids."""
    monitor = _monitor(tmp_path)
    ev = {"evidence_class": MODEL_SIGNAL, "category": "refusal"}
    for i in range(5):
        monitor.record_block(unit={"item_idx": i}, evidence=ev, model="m")

    ids = [json.loads(l)["block_id"] for l in (tmp_path / "BLOCKS.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids)), "block_ids must be unique within a run"


# ── Sanitization ──────────────────────────────────────────────────────────────

def test_raw_body_excerpt_sanitizes_secret(tmp_path):
    """Fake API key in raw body → REDACTED in excerpt; sha256 of ORIGINAL bytes."""
    monitor = _monitor(tmp_path)
    secret = "sk-" + "abc123456789012345"
    raw_body = {
        "error": {"message": f"Forbidden: auth={secret}", "code": "content_filter"}
    }

    # Compute expected sha256 of the ORIGINAL (pre-sanitization) serialized body.
    # _make_evidence_snapshot serializes dicts with sort_keys=True, no spaces.
    body_text = json.dumps(raw_body, separators=(",", ":"), sort_keys=True)
    expected_sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()

    monitor.record_block(
        unit={"item_idx": 1},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "content_filter"},
        model="m",
        raw_error=_FakeErrorBody(raw_body),
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())

    # Secret must be redacted in the excerpt
    assert secret not in block["raw_body_excerpt"], (
        "raw_body_excerpt must not contain the raw secret"
    )
    assert "redacted" in block["raw_body_excerpt"].lower()

    # SHA256 must be over the ORIGINAL bytes (pre-sanitization provenance digest)
    assert block["raw_body_sha256"] == expected_sha


def test_raw_body_excerpt_sanitizes_secret_from_body_attr(tmp_path):
    """SDK-shaped exception (.body not .raw_response) also sanitizes correctly."""
    monitor = _monitor(tmp_path)
    secret = "sk-" + "or-v1-supersecretkey12345"
    raw_body = {"error": {"message": f"key={secret}"}}

    body_text = json.dumps(raw_body, separators=(",", ":"), sort_keys=True)
    expected_sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()

    monitor.record_block(
        unit={"item_idx": 1},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "content_filter"},
        model="m",
        raw_error=_FakeBodyOnly(raw_body),
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())
    assert secret not in block["raw_body_excerpt"]
    assert block["raw_body_sha256"] == expected_sha


# ── Truncation ────────────────────────────────────────────────────────────────

def test_raw_body_excerpt_capped_at_2000_chars(tmp_path):
    """Excerpt is capped at 2000 chars; sha256 is of the FULL body (pre-truncation)."""
    monitor = _monitor(tmp_path)
    long_message = "X" * 4000
    raw_body = {"message": long_message}

    body_text = json.dumps(raw_body, separators=(",", ":"), sort_keys=True)
    expected_sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()

    monitor.record_block(
        unit={"item_idx": 1},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "refusal"},
        model="m",
        raw_error=_FakeErrorBody(raw_body),
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())

    assert len(block["raw_body_excerpt"]) <= 2000
    # Digest must be of the FULL body, not the truncated excerpt
    assert block["raw_body_sha256"] == expected_sha
    assert block["raw_body_sha256"] != hashlib.sha256(
        block["raw_body_excerpt"].encode("utf-8")
    ).hexdigest(), "sha256 should be of full body, not the truncated excerpt"


# ── Mixed v1+v2 fixture readers ───────────────────────────────────────────────

@pytest.fixture()
def mixed_blocks_run(tmp_path):
    """A run directory with a mixed v1+v2 BLOCKS.jsonl (incl. v1 with no unit_id)."""
    v1_no_uid = {
        "schema_version": "benchmark-block-v1",
        "timestamp": "2026-01-01T00:00:00Z",
        "module": "aita", "stage": "gen", "attempt_number": 1, "model": "m1",
        "unit": {"item_idx": 1, "side": "side_a"},
        "evidence_class": "model_signal", "category": "refusal",
        # No unit_id — this is the v1 without-unit_id case per D3 spec.
    }
    v1_with_uid = {
        "schema_version": "benchmark-block-v1",
        "timestamp": "2026-01-01T00:00:01Z",
        "module": "aita", "stage": "gen", "attempt_number": 1, "model": "m1",
        "unit": {"item_idx": 2, "side": "side_a"},
        "evidence_class": "model_signal", "category": "refusal",
        "unit_id": "aita:m1:item2:side_a",
    }
    v2_record = {
        "schema_version": "benchmark-block-v2",
        "block_id": uuid.uuid4().hex,
        "timestamp": "2026-01-01T00:00:02Z",
        "module": "aita", "stage": "gen", "attempt_number": 2, "model": "m1",
        "unit": {"item_idx": 3, "side": "side_a"},
        "evidence_class": "model_signal", "category": "content_filter",
        "unit_id": "aita:m1:item3:side_a",
        "signal_source": "provider-signals-v2",
        "retry_policy_kind": "bounded_retry",
        "stochastic": False,
        "billed_attempts": 2,
        "raw_body_sha256": "a" * 64,
        "raw_body_excerpt": "excerpt here",
    }
    (tmp_path / "BLOCKS.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [v1_no_uid, v1_with_uid, v2_record]) + "\n"
    )
    return tmp_path


def test_mixed_fixture_owed_units_reader(mixed_blocks_run):
    """owed_units._load_blocked_unit_ids reads mixed v1+v2 without error."""
    from suite_tools.owed_units import _load_blocked_unit_ids
    blocked = _load_blocked_unit_ids(mixed_blocks_run)
    # v1 with uid and v2 should appear; v1 without uid is skipped (no unit_id)
    assert "aita:m1:item2:side_a" in blocked
    assert "aita:m1:item3:side_a" in blocked
    # No exception raised; v1-no-uid row silently skipped (no unit_id in the entry)


def test_mixed_fixture_score_rows_reader(mixed_blocks_run):
    """score_rows._block_categories reads mixed v1+v2 without error."""
    from suite_tools.score_rows import _block_categories
    cats = _block_categories(mixed_blocks_run)
    assert "aita:m1:item2:side_a" in cats
    assert "aita:m1:item3:side_a" in cats
    assert cats["aita:m1:item3:side_a"] == "content_filter"


def test_mixed_fixture_backfill_reader(mixed_blocks_run):
    """backfill_denials._load_block_keys reads mixed v1+v2 without error."""
    from suite_tools.backfill_denials import _load_block_keys
    keys = _load_block_keys(mixed_blocks_run)
    # Returns set of 5-tuples; may be empty (no backfill_id in fixture)
    assert isinstance(keys, set)


def test_mixed_fixture_blocks_union_reader(mixed_blocks_run):
    """The bundle blocks projection (now via review_projection + the D2/D7
    allowlist) tolerates mixed v1+v2 BLOCKS and drops non-allowlisted fields
    (e.g. raw_body_excerpt) without error — the T7 replacement for _blocks_union."""
    from suite_tools.bundle import _capture_run_snapshot, _published_facts_for_member
    snap = _capture_run_snapshot(mixed_blocks_run, "m1")
    won = {"aita:m1:item2:side_a", "aita:m1:item3:side_a"}
    blocks, _evidence, _refs = _published_facts_for_member(snap, won)
    unit_ids = [r.get("unit_id") for r in blocks]
    assert "aita:m1:item2:side_a" in unit_ids
    assert "aita:m1:item3:side_a" in unit_ids
    # allowlist: the v2 raw_body_excerpt is dropped; the provenance digest is kept.
    v2 = next(r for r in blocks if r["unit_id"] == "aita:m1:item3:side_a")
    assert "raw_body_excerpt" not in v2
    assert v2["raw_body_sha256"] == "a" * 64
    assert v2["member_id"] == "m1"


# ── attempt_failure_classified event_id ──────────────────────────────────────

def test_attempt_failure_classified_event_has_event_id(tmp_path):
    """attempt_failure_classified events automatically carry event_id (uuid4 hex)."""
    monitor = _monitor(tmp_path)
    monitor.record(
        "attempt_failure_classified",
        model="test-model",
        evidence_class="model_signal",
        category="guardrail_permission_denied",
        action="halt",
        failure_reason="blocked by guardrail",
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    halt_events = [e for e in events if e.get("event") == "attempt_failure_classified"]
    assert len(halt_events) == 1
    ev = halt_events[0]
    assert "event_id" in ev, "event_id must be present in attempt_failure_classified events"
    uuid.UUID(hex=ev["event_id"], version=4)  # must be valid uuid4


def test_attempt_failure_classified_event_id_unique_across_events(tmp_path):
    """Multiple attempt_failure_classified events get unique event_ids."""
    monitor = _monitor(tmp_path)
    for i in range(3):
        monitor.record(
            "attempt_failure_classified",
            model="m",
            evidence_class="unknown",
            category="unclassified",
            action="halt",
            failure_reason=f"failure {i}",
        )
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "attempt_failure_classified"
    ]
    assert len(events) == 3
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids)), "event_ids must be unique across events"


def test_other_events_do_not_get_event_id(tmp_path):
    """Non-attempt_failure_classified events do not get an event_id stamped."""
    monitor = _monitor(tmp_path)
    monitor.record("turn_saved", model="m", item_idx=1, side="side_a", turn=1)
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    non_start_events = [e for e in events if e.get("event") == "turn_saved"]
    for ev in non_start_events:
        assert "event_id" not in ev


# ── Old-shape events still parse everywhere ───────────────────────────────────

def test_old_shape_attempt_failure_classified_parses_in_blockers(tmp_path):
    """bench.blockers works with v1-shape attempt_failure_classified events (no event_id)."""
    root = tmp_path / "results"
    run_dir = root / "stuck_run"
    run_dir.mkdir(parents=True)

    # Minimal run fixtures expected by bench._scan_runs
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "stuck_run",
        "modules": [{"module": "aita", "expected_units": []}],
        "identity": {"sample_spec": {"item_indices": [0]}, "model_conditions": []},
    }))
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "attempt_number": 1,
        "started_at": "2026-01-01T00:00:00Z",
    }))
    # Old-shape event: no event_id, no snapshot fields
    old_event = {
        "schema_version": "benchmark-run-ledger-v1",
        "sequence": 2,
        "timestamp": "2026-01-01T00:00:01Z",
        "module": "aita",
        "stage": "gen",
        "event": "attempt_failure_classified",
        "attempt_number": 1,
        "action": "halt",
        "category": "unclassified",
        "evidence_class": "unknown",
        "failure_reason": "some legacy error",
        "model": "old-model",
    }
    (run_dir / "RUN_EVENTS.jsonl").write_text(json.dumps(old_event) + "\n")

    from suite_tools.bench import blockers
    result = blockers(roots=[root])
    assert isinstance(result, dict)
    assert "blockers" in result
    blocker_ids = {b["run_id"] for b in result["blockers"]}
    assert "stuck_run" in blocker_ids


# ── Issue 1: secret straddling the 2000-char boundary ────────────────────────

class _FakeBodyStr:
    """Exception whose raw_response is a plain string (not a dict).

    Used to test the boundary-straddling secret case without JSON
    serialization distorting the character positions.
    """
    def __init__(self, s: str):
        self.raw_response = s

    def __str__(self):
        return self.raw_response


def test_secret_straddling_2000_char_boundary_is_redacted(tmp_path):
    """Secret that straddles the 2000-char truncation boundary must be redacted.

    BUG (Issue 1): sanitize_error_message(body_text[:2000]) truncates FIRST so a
    secret whose suffix is clipped falls below the regex min-length floor
    (sk-[...]{16,}) and survives un-redacted in the excerpt.

    FIX: sanitize_error_message(body_text)[:2000] — sanitize THEN truncate.

    Construction:
    - Padding 1985 'X's brings us to position 1985.
    - Secret prefix plus 17 suffix characters = 20 chars total.
    - With truncate-first: excerpt has "sk-" + 12 chars = 15 < 16-char minimum → NOT matched.
    - With sanitize-first: full 17-char suffix is visible → matched and redacted.
    """
    monitor = _monitor(tmp_path)
    secret = "sk-" + "abcdefghijklmnopq"  # 3 + 17 chars; regex needs 16+ after the prefix
    padding = "X" * 1985
    body_text = padding + secret        # secret starts at position 1985; total = 2005

    monitor.record_block(
        unit={"item_idx": 1},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "refusal"},
        model="m",
        raw_error=_FakeBodyStr(body_text),
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())

    assert secret not in block.get("raw_body_excerpt", ""), (
        "Secret straddling the 2000-char boundary must be redacted in excerpt — "
        "fix: sanitize_error_message(body_text)[:2000], not body_text[:2000]"
    )
    assert "redacted" in block.get("raw_body_excerpt", "").lower(), (
        "raw_body_excerpt must contain '<redacted>' after the fix"
    )


# ── Issue 2: billed_attempts from ContentBlockPolicyExecutor ─────────────────

def test_executor_billed_attempt_count_bounded_retry_1():
    """ContentBlockPolicyExecutor.billed_attempt_count() returns 2 after bounded_retry(1).

    bounded_retry(1): first decide() returns 'continue' (1 paid retry allowed),
    second decide() returns 'terminalize'.  billed_attempt_count() = _signal_attempts + 1 = 2.
    The runner must stamp exc._billed_attempts = executor.billed_attempt_count() BEFORE
    raising so record_block receives the correct count via getattr(exc, '_billed_attempts', 1).
    """
    from suite_tools.content_block_policy import ContentBlockPolicyExecutor

    policy = {"kind": "bounded_retry", "max_retries": 1}
    executor = ContentBlockPolicyExecutor()
    ev = {"evidence_class": MODEL_SIGNAL, "category": "content_filter",
          "retry_policy": policy}

    assert executor.decide(ev) == "continue"     # first attempt → retry allowed
    assert executor.decide(ev) == "terminalize"  # second attempt → exhausted
    assert executor.billed_attempt_count() == 2, (
        "bounded_retry(1) burns 2 paid calls: initial + 1 retry"
    )


def test_billed_attempts_2_flows_to_block_record(tmp_path):
    """_billed_attempts=2 on exception → block record billed_attempts=2.

    Simulates the runner's terminalize path after the fix:
      exc._billed_attempts = executor.billed_attempt_count()  # = 2
      raise exc
      ...
      record_block(..., billed_attempts=getattr(exc, '_billed_attempts', 1))

    RED before fix: runners don't stamp _billed_attempts, so getattr returns 1.
    GREEN after fix: _billed_attempts=2 is stamped; block shows billed_attempts=2.
    """
    monitor = _monitor(tmp_path)

    class FakeRefusal(Exception):
        raw_response = {"error": {"code": "content_policy_violation"}}

    exc = FakeRefusal("content block: content_filter")
    exc._billed_attempts = 2   # what the fixed runner stamps via executor.billed_attempt_count()

    monitor.record_block(
        unit={"item_idx": 1},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "content_filter"},
        model="gpt-5.6",
        raw_error=exc,
        billed_attempts=getattr(exc, "_billed_attempts", 1),
    )
    block = json.loads((tmp_path / "BLOCKS.jsonl").read_text().strip())
    assert block.get("billed_attempts") == 2, (
        "billed_attempts must be 2 when executor.billed_attempt_count()=2 is stamped on exc"
    )


def test_new_shape_events_also_parse_in_blockers(tmp_path):
    """bench.blockers works with v2-shape events that carry event_id and snapshot fields."""
    root = tmp_path / "results"
    run_dir = root / "new_run"
    run_dir.mkdir(parents=True)

    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "new_run",
        "modules": [{"module": "aita", "expected_units": []}],
        "identity": {"sample_spec": {"item_indices": [0]}, "model_conditions": []},
    }))
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1",
        "attempt_number": 1,
        "started_at": "2026-01-01T00:00:00Z",
    }))
    # New-shape event: has event_id + snapshot fields
    new_event = {
        "schema_version": "benchmark-run-ledger-v1",
        "sequence": 2,
        "timestamp": "2026-01-01T00:00:01Z",
        "module": "aita",
        "stage": "gen",
        "event": "attempt_failure_classified",
        "attempt_number": 1,
        "action": "halt",
        "category": "guardrail_permission_denied",
        "evidence_class": "model_signal",
        "failure_reason": "blocked by guardrail",
        "model": "gpt-5.6",
        # v2 new fields (additive)
        "event_id": uuid.uuid4().hex,
        "signal_source": "provider-signals-v2",
        "retry_policy_kind": "terminal",
        "stochastic": False,
        "raw_body_sha256": "a" * 64,
        "raw_body_excerpt": "...",
        "billed_attempts": 1,
    }
    (run_dir / "RUN_EVENTS.jsonl").write_text(json.dumps(new_event) + "\n")

    from suite_tools.bench import blockers
    result = blockers(roots=[root])
    blocker_ids = {b["run_id"] for b in result["blockers"]}
    assert "new_run" in blocker_ids
