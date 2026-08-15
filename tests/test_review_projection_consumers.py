"""Consumer wiring: dispositions drive owed_units / score_rows / blockers
through the projection (plan 020 T5, acceptance 3).

The zero-review byte-identity guard lives in test_review_projection_identity.py;
these tests assert the *positive* direction — that a real v2 review changes what
the consumers report.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from suite_tools import bench
from suite_tools import owed_units as _owed
from suite_tools import review_projection as rp
from suite_tools import score_rows as _sr


def _contract(run: Path, units: list[dict], *, run_id="run-1") -> None:
    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": run_id,
        "modules": [{"module": "aita", "stage": "gen", "expected_units": units}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "c", "canonical_model": "x",
             "route": "openrouter", "effort": "high", "profile": None}]},
    }))


def _block(run: Path, unit_id, *, block_id, category="refusal", cls="model_signal"):
    (run / "BLOCKS.jsonl").write_text(json.dumps({
        "schema_version": "benchmark-block-v2", "block_id": block_id,
        "timestamp": "2026-01-01T00:00:00Z", "module": "aita", "stage": "gen",
        "attempt_number": 1, "model": "m", "unit_id": unit_id,
        "unit": {"item_idx": 0, "side": "side_a"},
        "evidence_class": cls, "category": category,
    }) + "\n")


def test_retry_review_flips_owed_units_from_terminal_to_owed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    uid = "aita:m:item0:side_a"
    _contract(run, [{"unit_id": uid, "expected_transcript_path": "t.json",
                     "planned_turns": 2}])  # artifact absent -> would be terminal
    bid = uuid.uuid4().hex
    _block(run, uid, block_id=bid)

    before = _owed.owed_units(run, module="aita")
    assert before["units"][0]["state"] == "terminal_model_signal"

    rp.append_review(run, {"schema_version": rp.REVIEW_SCHEMA_VERSION,
                           "review_id": uuid.uuid4().hex, "event_ref": f"blocks-id:{bid}",
                           "module": "aita", "model": "m", "category": "refusal",
                           "disposition": "retry", "reviewer": "t", "rationale": "r"})
    after = _owed.owed_units(run, module="aita")
    assert after["units"][0]["state"] == "owed"
    assert after["units"][0]["reason"] == "pending_retry"


def test_safety_declination_resolved_category_flows_into_score_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    uid = "aita:m:item0:a"
    _contract(run, [{"unit_id": uid, "side": "a", "item_idx": 0,
                     "expected_score_path": "m_item0_scores.json", "planned_turns": 1}])
    (run / "FINAL_RESULTS.json").write_text(json.dumps({"scores": {}}))
    bid = uuid.uuid4().hex
    _block(run, uid, block_id=bid, category="ambiguous_403", cls="unknown")
    rp.append_review(run, {"schema_version": rp.REVIEW_SCHEMA_VERSION,
                           "review_id": uuid.uuid4().hex, "event_ref": f"blocks-id:{bid}",
                           "module": "aita", "model": "m", "category": "ambiguous_403",
                           "resolved_category": "cyber", "disposition": "safety_declination",
                           "reviewer": "t", "rationale": "r"})
    out = _sr.score_rows(run, module="aita")
    entry = next(u for u in out["units"] if u["unit_id"] == uid)
    assert entry["outcome_class"] == "terminal_model_signal"
    assert entry["category"] == "cyber"  # resolved_category, not the raw ambiguous_403


def _blockers_run(root: Path, *, run_id: str) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1", "run_id": run_id,
        "modules": [{"module": "aita", "expected_units": []}],
        "identity": {"sample_spec": {"item_indices": [0]}, "model_conditions": []},
    }))
    (run / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1", "attempt_number": 1,
        "started_at": "2026-01-01T00:00:00Z"}))
    (run / "RUN_EVENTS.jsonl").write_text(json.dumps({
        "schema_version": "benchmark-run-ledger-v1", "sequence": 2,
        "timestamp": "2026-01-01T00:00:01Z", "module": "aita", "stage": "gen",
        "event": "attempt_failure_classified", "attempt_number": 1, "action": "halt",
        "category": "ambiguous_403", "evidence_class": "unknown",
        "failure_reason": "bare 403", "model": "m", "event_id": "halt-1"}) + "\n")
    return run


def test_safety_declination_drops_member_halt_from_blockers(tmp_path):
    root = tmp_path / "results"
    run = _blockers_run(root, run_id="stuck")
    assert {b["run_id"] for b in bench.blockers(roots=[root])["blockers"]} == {"stuck"}

    rp.append_review(run, {"schema_version": rp.REVIEW_SCHEMA_VERSION,
                           "review_id": uuid.uuid4().hex, "event_ref": "events-id:halt-1",
                           "module": "aita", "model": "m", "category": "ambiguous_403",
                           "resolved_category": "cyber", "disposition": "safety_declination",
                           "reviewer": "t", "rationale": "r"})
    assert bench.blockers(roots=[root])["blockers"] == []


def test_needs_escalation_keeps_member_halt_as_blocker(tmp_path):
    root = tmp_path / "results"
    run = _blockers_run(root, run_id="stuck")
    rp.append_review(run, {"schema_version": rp.REVIEW_SCHEMA_VERSION,
                           "review_id": uuid.uuid4().hex, "event_ref": "events-id:halt-1",
                           "module": "aita", "model": "m", "category": "ambiguous_403",
                           "disposition": "needs_escalation", "reviewer": "t", "rationale": "r"})
    assert {b["run_id"] for b in bench.blockers(roots=[root])["blockers"]} == {"stuck"}
