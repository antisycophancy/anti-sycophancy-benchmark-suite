"""Tests for suite_tools.backfill_denials (Task 4, Phase B)."""

from __future__ import annotations

import json
from pathlib import Path

from suite_tools import backfill_denials


def _write(p: Path, obj):
    p.write_text(json.dumps(obj))


def _mk_run(tmp_path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    # (a) F1 failed transcript with typed refusal signature
    _write(run / "fable_item1_delusion_side_a.json", {
        "model": "fable", "item_idx": 1, "test_type": "delusion", "side": "side_a",
        "completed": False,
        "failure_reason": "HTTP 200: Anthropic native provider refusal; stop_reason=refusal; classifier=cyber",
    })
    # (c) plain timeout — ignored
    _write(run / "fable_item2_delusion_side_a.json", {
        "model": "fable", "item_idx": 2, "test_type": "delusion", "side": "side_a",
        "completed": False, "failure_reason": "request timed out after 120s",
    })
    # (d) already-blocked refusal — ignored (block already present)
    _write(run / "fable_item3_delusion_side_a.json", {
        "model": "fable", "item_idx": 3, "test_type": "delusion", "side": "side_a",
        "completed": False, "provider_refusal": True,
        "failure_reason": "stop_reason=refusal; classifier=cyber",
    })
    (run / "BLOCKS.jsonl").write_text(json.dumps({
        "schema_version": "benchmark-block-v1", "module": "epis", "model": "fable",
        "unit_id": "epis:fable:delusion:item3:side_a", "category": "cyber",
        "backfilled": True, "backfill_id": "retro-audit-20260721",
    }) + "\n")
    (run / "BLOCK_REVIEWS.jsonl").write_text(json.dumps({
        "schema_version": "benchmark-block-review-v1", "module": "epis", "model": "fable",
        "unit_id": "epis:fable:delusion:item3:side_a", "category": "cyber",
        "backfill_id": "retro-audit-20260721", "disposition": "safety_declination",
    }) + "\n")
    # (b) F2 evidence only in RUN_EVENTS
    events = [{"event": "conversation_failed", "module": "epis", "model": "gpt56",
               "item_idx": 4, "test_type": "delusion", "side": "side_a",
               "failure_reason": "HTTP 400: flagged for possible cybersecurity risk"}]
    (run / "RUN_EVENTS.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run


def test_dry_run_reports_f1_and_f2_only(tmp_path):
    run = _mk_run(tmp_path)
    found = backfill_denials.discover(run)
    got = sorted((d["unit_id"], d["category"], d["evidence_pointer"]) for d in found)
    assert got == [
        ("epis:fable:delusion:item1:side_a", "cyber", "fable_item1_delusion_side_a.json"),
        ("epis:gpt56:delusion:item4:side_a", "cyber_policy", "RUN_EVENTS.jsonl#sequence=1"),
    ]


def test_apply_writes_full_key_reviews_then_idempotent(tmp_path):
    run = _mk_run(tmp_path)
    backfill_denials.apply(run)
    blocks = [json.loads(x) for x in (run / "BLOCKS.jsonl").read_text().splitlines()]
    reviews = [json.loads(x) for x in (run / "BLOCK_REVIEWS.jsonl").read_text().splitlines()]
    added_blocks = [b for b in blocks if b.get("backfilled") and b["unit_id"] != "epis:fable:delusion:item3:side_a"]
    assert {b["category"] for b in added_blocks} == {"cyber", "cyber_policy"}
    assert all(b["evidence_class"] == "model_signal" and b.get("unit_id") for b in added_blocks)
    added_reviews = [r for r in reviews if r["unit_id"] != "epis:fable:delusion:item3:side_a"]
    assert len(added_reviews) == 2
    for r in added_reviews:
        assert {"module", "model", "unit_id", "category", "backfill_id"} <= set(r)
    before_blocks = (run / "BLOCKS.jsonl").read_bytes()
    before_reviews = (run / "BLOCK_REVIEWS.jsonl").read_bytes()
    backfill_denials.apply(run)  # second apply is a no-op
    assert (run / "BLOCKS.jsonl").read_bytes() == before_blocks
    assert (run / "BLOCK_REVIEWS.jsonl").read_bytes() == before_reviews


def test_reviews_heal_after_partial_write(tmp_path):
    run = _mk_run(tmp_path)
    for d in backfill_denials.discover(run):
        backfill_denials._append_block(run, d)  # blocks only (simulated crash before reviews)
    backfill_denials.apply(run)  # re-apply reconciles missing reviews independently
    reviews = [json.loads(x) for x in (run / "BLOCK_REVIEWS.jsonl").read_text().splitlines()]
    added_reviews = [r for r in reviews if r["unit_id"] != "epis:fable:delusion:item3:side_a"]
    assert len(added_reviews) == 2
    added_blocks = [json.loads(x) for x in (run / "BLOCKS.jsonl").read_text().splitlines()
                    if '"backfilled"' in x and "item3" not in x]
    assert len(added_blocks) == 2  # blocks not duplicated by the heal


def test_transcripts_never_modified(tmp_path):
    run = _mk_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.glob("*_item*.json")}
    backfill_denials.apply(run)
    after = {p.name: p.read_bytes() for p in run.glob("*_item*.json")}
    assert before == after


# ---------------------------------------------------------------------------
# Review-round fixes (declared allowlist, failure_status probe, module guard)
# ---------------------------------------------------------------------------


def test_category_outside_allowlist_not_emitted(tmp_path):
    """A stop_reason=refusal with an out-of-scope classifier is silently dropped.
    The signature is valid (typed, not text-heuristic), but the category
    'violence' is not in ALLOWED_CATEGORIES so discover() must not return it."""
    run = tmp_path / "run_cat"
    run.mkdir()
    _write(run / "model_item5_test_side_a.json", {
        "model": "model", "item_idx": 5, "test_type": "test", "side": "side_a",
        "completed": False,
        "failure_reason": "stop_reason=refusal; classifier=violence",
    })
    (run / "RUN_EVENTS.jsonl").write_text("")
    found = backfill_denials.discover(run)
    assert found == [], f"Expected empty but got {found}"


def test_event_failure_status_probe(tmp_path):
    """Denial signature present only in failure_status (not failure_reason) is
    discovered via Source B.  Verifies the failure_status probe added by the
    review fix."""
    run = tmp_path / "run_fs"
    run.mkdir()
    events = [
        {
            "event": "model_batch_item_failed",
            "module": "epis",
            "model": "testmodel",
            "item_idx": 1,
            "test_type": "delusion",
            "side": "side_a",
            # No failure_reason; denial only in failure_status.
            "failure_status": "stop_reason=refusal; classifier=cyber",
        }
    ]
    (run / "RUN_EVENTS.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    found = backfill_denials.discover(run)
    assert len(found) == 1, f"Expected 1 but got {found}"
    assert found[0]["category"] == "cyber"
    assert found[0]["evidence_pointer"] == "RUN_EVENTS.jsonl#sequence=1"


def test_unknown_module_event_skipped(tmp_path, capsys):
    """Events with a module name not in _MODULE_PREFIX are skipped and a
    stderr warning mentioning the unknown name is emitted."""
    run = tmp_path / "run_mod"
    run.mkdir()
    events = [
        {
            "event": "model_batch_item_failed",
            "module": "unknown_bench",
            "model": "testmodel",
            "item_idx": 1,
            "test_type": "delusion",
            "side": "side_a",
            "failure_reason": "stop_reason=refusal; classifier=cyber",
        }
    ]
    (run / "RUN_EVENTS.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    found = backfill_denials.discover(run)
    assert found == [], f"Expected empty but got {found}"
    captured = capsys.readouterr()
    assert "unknown_bench" in captured.err, f"Expected warning for 'unknown_bench' in stderr: {captured.err!r}"
