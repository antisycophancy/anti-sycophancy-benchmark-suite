"""Tests for suite_tools.bundle — Task 020-T7 (D2 + D6 + D7).

The publication-safety core of Phase D:

* **Allowlist projection** of BLOCKS records into ``data/blocks.jsonl`` — the v2
  ``raw_body_excerpt`` (and every unknown/future key) is dropped by construction,
  not by denylist.  The regression guard here MUST fail before the allowlist
  lands: pre-phase ``_blocks_union`` spreads whole v2 records so a raw body would
  reach the bundle.
* **NEW** ``data/evidence.jsonl`` — projected ``attempt_failure_classified`` facts,
  member/winner scoped, so every fact a review may reference exists in the bundle.
* **NEW** ``data/block_reviews.jsonl`` — active review + full supersession chain,
  ``rationale`` projected out by default; ``--include-review-rationale`` includes
  it AND stamps ``contains_review_rationale`` in the manifest (mirror of the
  transcripts flag).
* **Three-clause hard gate** (D6) — no bypass — fails before the promote, listing
  every offender as ``(member_id, unit_id?, event_ref, reason)``.
* **RunSnapshot** — per-member capture (RUN_STATUS + ledger/review bytes +
  fingerprints); the projection + gate consume only the snapshot; a pre-promote
  fingerprint recheck aborts on drift with staging cleanup.

Distinctive sentinel strings are planted so the tree-wide leak assertions can
grep the entire emitted bundle.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from suite_tools import bundle


# distinctive, greppable sentinels -------------------------------------------
RAW_BODY_SENTINEL = "RAWBODYSENTINEL_deadbeef_prompt_echo"
RATIONALE_SENTINEL = "RATIONALESENTINEL_cafef00d_reviewer_note"
CALL_DIAGNOSTIC_SENTINEL = "CALLDIAGNOSTICSENTINEL_private_sidecar"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _completed_conv(*, attempt: int = 1, turns: int = 1) -> dict:
    return {"completed": True, "attempt_number": attempt,
            "condition_id": "cond-m",
            "condition_hash": "sha256:cond-m",
            "route_hash": "sha256:route-m",
            "turns": [{"model_response": "x"} for _ in range(turns)]}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _mk_run(
    run_dir: Path,
    *,
    module: str = "aita",
    run_id: str,
    units: list[dict],
    status: dict | None,
    artifacts: dict[str, dict] | None = None,
    blocks: list[dict] | None = None,
    events: list[dict] | None = None,
    reviews: list[dict] | None = None,
    started_at: str = "2026-07-20T12:00:00Z",
) -> Path:
    """Materialise one run directory and return its RUN_CONTRACT.json path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": run_id,
        "modules": [{"module": module, "stage": "generation",
                     "expected_units": units}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m",
             "condition_hash": "sha256:cond-m",
             "route_hash": "sha256:route-m", "canonical_model": "x",
             "route": "openrouter", "effort": "high", "profile": None}]},
    }
    contract["provenance"] = bundle.provenance_hashes(contract)
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    status_doc = {"attempt_number": 1, "started_at": started_at}
    if status is not None:
        status_doc.update(status)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps(status_doc))
    for rel, data in (artifacts or {}).items():
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
    if blocks:
        _write_jsonl(run_dir / "BLOCKS.jsonl", blocks)
    if events:
        _write_jsonl(run_dir / "RUN_EVENTS.jsonl", events)
    if reviews:
        _write_jsonl(run_dir / "BLOCK_REVIEWS.jsonl", reviews)
    return run_dir / "RUN_CONTRACT.json"


def _experiment(tmp_path: Path, members: list[tuple[Path, str]], *,
                experiment_id: str = "gate-exp", modules=("aita",)) -> Path:
    exp = tmp_path / "exp"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": experiment_id,
        "title": "t",
        "instrument": {"modules": list(modules), "hashes": {}},
        "conditions": [],
        "target": {"n_items": 1},
        "members": [{"path": str(cp.resolve()), "role": role}
                    for cp, role in members],
    }))
    return exp


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


def _event(*, unit_id=None, attempt=1, category="ambiguous_403", cls="unknown",
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


def test_contract_projection_rejects_stale_source_provenance():
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "stale-source",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": {},
            "model_conditions": [],
            "execution": {},
        },
    }
    contract["provenance"] = bundle.provenance_hashes(contract)
    contract["identity"]["judge_panel"] = {"rubric_version": "changed-after-freeze"}

    with pytest.raises(ValueError, match="source contract provenance drift"):
        bundle._project_contract_for_bundle(contract)


def test_contract_projection_rejects_missing_source_provenance():
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "missing-source-provenance",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": {},
            "model_conditions": [],
            "execution": {},
        },
    }

    with pytest.raises(ValueError, match="source contract provenance missing"):
        bundle._project_contract_for_bundle(contract)


def test_bundle_provenance_audit_recomputes_projected_contract(tmp_path):
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir()
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "bundle-stale",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": {},
            "model_conditions": [],
        },
    }
    contract["provenance"] = bundle.provenance_hashes(contract)
    contract["identity"]["judge_panel"] = {"rubric_version": "mutated"}
    (provenance_dir / "RUN_CONTRACT-m1.json").write_text(json.dumps(contract))

    issues = bundle.audit_bundle_provenance(tmp_path)

    assert len(issues) == 1
    assert issues[0].path == "provenance/RUN_CONTRACT-m1.json"
    assert issues[0].reason == "projected contract provenance drift"


def test_bundle_provenance_audit_requires_contract_for_every_manifest_member(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "outcomes.jsonl").write_text(json.dumps({"unit_id": "u1"}) + "\n")
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": bundle.BUNDLE_SCHEMA_VERSION,
        "members": [{"member_id": "m1"}],
        "union": {"units": [{"unit_id": "u1"}]},
        "payload_files": [{"path": "data/outcomes.jsonl"}],
    }))

    issues = bundle.audit_bundle_provenance(tmp_path)

    assert [(issue.path, issue.reason) for issue in issues] == [
        ("provenance/RUN_CONTRACT-m1.json", "projected contract missing")
    ]


def test_bundle_provenance_audit_refuses_empty_member_manifest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "outcomes.jsonl").write_text(json.dumps({"unit_id": "u1"}) + "\n")
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": bundle.BUNDLE_SCHEMA_VERSION,
        "members": [],
        "union": {"units": [{"unit_id": "u1"}]},
        "payload_files": [{"path": "data/outcomes.jsonl"}],
    }))

    issues = bundle.audit_bundle_provenance(tmp_path)

    assert [(issue.path, issue.reason) for issue in issues] == [
        ("BUNDLE_MANIFEST.json", "bundle member provenance missing")
    ]


def test_bundle_provenance_audit_refuses_empty_payload_inventory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "outcomes.jsonl").write_text(json.dumps({"unit_id": "u1"}) + "\n")
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": bundle.BUNDLE_SCHEMA_VERSION,
        "members": [{"member_id": "m1"}],
        "union": {"units": [{"unit_id": "u1"}]},
        "payload_files": [],
    }))

    issues = bundle.audit_bundle_provenance(tmp_path)

    assert any(issue.reason == "bundle payload inventory empty" for issue in issues)


def _review(event_ref, disposition, *, review_id="rv1", category="refusal",
            resolved_category=None, supersedes=None, rationale="because",
            issue_ref=None, unit_id=None) -> dict:
    rec = {
        "schema_version": "benchmark-block-review-v2",
        "review_id": review_id, "event_ref": event_ref,
        "module": "aita", "model": "m",
        "category": category, "disposition": disposition,
        "reviewer": "tester", "rationale": rationale,
        "reviewed_at": "2026-01-02T00:00:00Z",
    }
    if unit_id is not None:
        rec["unit_id"] = unit_id
    if resolved_category is not None:
        rec["resolved_category"] = resolved_category
    if supersedes is not None:
        rec["supersedes_review_id"] = supersedes
    if issue_ref is not None:
        rec["issue_ref"] = issue_ref
    return rec


def _unit(unit_id, *, planned=1, score=True, transcript=True) -> dict:
    u = {"unit_id": unit_id, "model_key": "m", "planned_turns": planned}
    if transcript:
        u["expected_transcript_path"] = f"{unit_id.replace(':', '_')}_t.json"
    if score:
        u["expected_score_path"] = f"{unit_id.replace(':', '_')}_s.json"
    return u


def _emit(exp: Path, tmp_path: Path, **kwargs) -> Path:
    return Path(bundle.emit(exp, out_dir=tmp_path / "out", **kwargs)["bundle_dir"])


def test_package_refuses_experiment_without_members(tmp_path):
    exp = _experiment(tmp_path, [])

    with pytest.raises(ValueError, match="no active members"):
        _emit(exp, tmp_path)

    assert not any((tmp_path / "out").glob("bundle-*"))


@pytest.mark.parametrize(
    "member_path",
    [None, "", "   ", ".", 42],
    ids=["null", "empty", "whitespace", "dot", "non-string"],
)
def test_package_refuses_ambiguous_active_member_paths(tmp_path, member_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "bad-member-path",
        "members": [{"path": member_path, "role": "pilot"}],
    }))

    with pytest.raises(ValueError, match="active member.*RUN_CONTRACT.json"):
        _emit(exp, tmp_path)

    assert not any((tmp_path / "out").glob("bundle-*"))


@pytest.mark.parametrize("path_kind", ["directory", "wrong-name", "missing"])
def test_package_requires_active_member_path_to_name_regular_contract(
    tmp_path, path_kind,
):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    if path_kind == "directory":
        member_path = run_dir
    elif path_kind == "wrong-name":
        member_path = run_dir / "contract.json"
        member_path.write_text("{}")
    else:
        member_path = run_dir / "RUN_CONTRACT.json"

    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "bad-contract-path",
        "members": [{"path": str(member_path.resolve()), "role": "pilot"}],
    }))

    with pytest.raises(ValueError, match="existing regular RUN_CONTRACT.json"):
        _emit(exp, tmp_path)

    assert not any((tmp_path / "out").glob("bundle-*"))


def test_package_refuses_empty_union_before_payload_writes(tmp_path, monkeypatch):
    cp = _mk_run(
        tmp_path / "runs" / "empty", run_id="empty", units=[],
        status={"status": "completed"},
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("bundle attempted a payload write before rejecting empty union")

    monkeypatch.setattr(bundle, "_write_json", unexpected_write)
    monkeypatch.setattr(bundle, "_write_jsonl", unexpected_write)
    monkeypatch.setattr(bundle, "_write_scores_csv", unexpected_write)

    with pytest.raises(ValueError, match="union contains no units"):
        _emit(exp, tmp_path)

    assert not any((tmp_path / "out").glob("bundle-*"))


# ---------------------------------------------------------------------------
# 1. Allowlist projection — raw_body_excerpt must never reach a bundle (RED)
# ---------------------------------------------------------------------------


def test_raw_body_excerpt_never_reaches_bundle(tmp_path):
    """REGRESSION: a winning unit's block carries raw_body_excerpt with a
    distinctive sentinel; the allowlist projection must drop it entirely."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1", raw_body_excerpt=RAW_BODY_SENTINEL,
                       raw_body_sha256="a" * 64)],
    )
    run_dir = cp.parent
    (run_dir / "CALL_DIAGNOSTICS.jsonl").write_text(
        json.dumps({"private": CALL_DIAGNOSTIC_SENTINEL}) + "\n"
    )
    assert "CALL_DIAGNOSTICS.jsonl" not in bundle._member_input_files(run_dir)
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)

    blocks_path = bundle_dir / "data" / "blocks.jsonl"
    assert blocks_path.exists()
    records = [json.loads(l) for l in blocks_path.read_text().splitlines() if l]
    assert records, "winning unit block should be published"
    for rec in records:
        assert "raw_body_excerpt" not in rec
        assert rec.get("raw_body_sha256") == "a" * 64      # digest kept

    # tree-wide: the raw body sentinel appears nowhere in the emitted bundle
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*")
                     if p.is_file())
    assert RAW_BODY_SENTINEL not in blob
    assert CALL_DIAGNOSTIC_SENTINEL not in blob


def test_blocks_allowlist_drops_unknown_future_keys(tmp_path):
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1",
                       some_future_field="LEAKY_FUTURE_VALUE_zzz",
                       raw_body_sha256="a" * 64)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    rec = json.loads((bundle_dir / "data" / "blocks.jsonl").read_text().splitlines()[0])
    assert "some_future_field" not in rec
    assert "LEAKY_FUTURE_VALUE_zzz" not in (bundle_dir / "data" / "blocks.jsonl").read_text()
    # allowlisted fields survive
    assert rec["block_id"] == "b1"
    assert rec["evidence_class"] == "model_signal"
    assert rec["member_id"] == "m1"
    assert rec["event_ref"] == "blocks-id:b1"


# ---------------------------------------------------------------------------
# 2. Three-clause hard gate (D6)
# ---------------------------------------------------------------------------


def test_gate_events_source_unknown_names_member_and_ref(tmp_path):
    """Clause a: a member-scoped unknown-class event with no review gates the
    contributing member, naming (member_id, event_ref)."""
    clean = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(clean, score=False)],
        status={"status": "completed"},
        artifacts={f"{clean.replace(':', '_')}_t.json": _completed_conv()},
        events=[_event(cls="unknown", category="ambiguous_403", event_id="e1")],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value)
    assert "m1" in msg
    assert "events-id:e1" in msg
    assert "unknown" in msg.lower()
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


def test_gate_retry_review_alone_still_gates(tmp_path):
    """Clause c: a retry review leaves the unit pending_retry (owed) until a
    strictly-later attempt discharges it."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1", attempt=1)],
        reviews=[_review("blocks-id:b1", "retry", unit_id=uid)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "pending_retry" in str(exc.value) or "owed" in str(exc.value)
    assert "m1" in str(exc.value)


def test_gate_retry_discharged_by_later_attempt_emits(tmp_path):
    """A completed artifact from a strictly-later attempt discharges the retry;
    the bundle then emits."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False)],
        status={"status": "completed"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv(attempt=2)},
        blocks=[_block(uid, block_id="b1", attempt=1)],
        reviews=[_review("blocks-id:b1", "retry", unit_id=uid)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    assert (bundle_dir / "data" / "blocks.jsonl").exists()


def test_gate_member_obligation_pending_gates_clean_units(tmp_path):
    """Clause c: an unfulfilled member-level retry obligation gates a member
    whose units all look clean."""
    clean = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(clean, score=False)],
        status={"status": "completed"},   # attempt 1 completed, not > 1
        artifacts={f"{clean.replace(':', '_')}_t.json": _completed_conv()},
        events=[_event(cls="model_signal", category="SAFETY", event_id="e1",
                       attempt=1)],           # member-scoped (no unit_id)
        reviews=[_review("events-id:e1", "retry")],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value)
    assert "m1" in msg and "events-id:e1" in msg
    assert "obligation" in msg.lower() or "retry" in msg.lower()


def test_gate_needs_escalation_on_typed_block_gates(tmp_path):
    """Clause b: an active needs_escalation review gates even a typed
    (model_signal) block."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1", cls="model_signal")],
        reviews=[_review("blocks-id:b1", "needs_escalation", unit_id=uid)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "needs_escalation" in str(exc.value)
    assert "blocks-id:b1" in str(exc.value)


def test_gate_instrument_defect_on_contributing_member_gates(tmp_path):
    """Clause c: any instrument_defect fact on a contributing member blocks."""
    clean = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(clean, score=False)],
        status={"status": "completed"},
        artifacts={f"{clean.replace(':', '_')}_t.json": _completed_conv()},
        events=[_event(cls="environment", category="timeout_read", event_id="e1")],
        reviews=[_review("events-id:e1", "instrument_defect")],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "instrument_defect" in str(exc.value)
    assert "m1" in str(exc.value)


def test_gate_has_no_bypass_flag(tmp_path):
    """There is no keyword that lets the gate be skipped."""
    import inspect
    sig = inspect.signature(bundle.emit)
    lowered = " ".join(sig.parameters).lower()
    assert "bypass" not in lowered
    assert "skip_gate" not in lowered and "force" not in lowered


# ---------------------------------------------------------------------------
# 3. RunSnapshot — non-terminal refusal + pre-promote fingerprint recheck
# ---------------------------------------------------------------------------


def test_non_terminal_run_status_member_refused(tmp_path):
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False)],
        status={"status": "running"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()},
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "terminal" in str(exc.value).lower() or "running" in str(exc.value).lower()
    assert not any((tmp_path / "out").glob("bundle-*"))


def test_statusless_member_refused_fail_closed(tmp_path):
    """Fail-closed: a contributing member with no RUN_STATUS.status (absent or
    unreadable) is refused — a real run always records its terminal status, so a
    missing one means the run never completed."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False)],
        status=None,                       # RUN_STATUS.json without a status field
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()},
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value).lower()
    assert "terminal" in msg and ("absent" in msg or "unreadable" in msg)
    assert "m1" in str(exc.value)
    assert not any((tmp_path / "out").glob("bundle-*"))


def test_race_ledger_mutation_between_capture_and_promote_aborts(tmp_path, monkeypatch):
    """A ledger line appended after snapshot capture but before the promote
    fails the fingerprint recheck: abort + staging cleanup, no bundle."""
    uid = "aita:m:item0:a"
    run_dir = tmp_path / "runs" / "r1"
    cp = _mk_run(
        run_dir, run_id="r1",
        units=[_unit(uid, score=False)],
        status={"status": "completed"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()},
        events=[_event(cls="model_signal", category="SAFETY", unit_id=uid,
                       event_id="e1", action="halt")],
        reviews=[_review("events-id:e1", "safety_declination",
                         resolved_category="SAFETY", unit_id=uid)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])

    original_audit = bundle.audit_bundle_tree

    def mutate_then_audit(staging):
        # simulate a new attempt starting mid-bundle
        with (run_dir / "RUN_EVENTS.jsonl").open("a") as fh:
            fh.write(json.dumps(_event(cls="unknown", event_id="e2")) + "\n")
        return original_audit(staging)

    monkeypatch.setattr(bundle, "audit_bundle_tree", mutate_then_audit)

    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "drift" in str(exc.value).lower() or "changed" in str(exc.value).lower()
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


# ---------------------------------------------------------------------------
# 4. Resolved-everything → emits with all three data files
# ---------------------------------------------------------------------------


def _resolved_experiment(tmp_path: Path) -> Path:
    """One member, everything resolved: a scored unit, a block resolved to a
    declination, and an events-source unknown resolved by review."""
    scored = "aita:m:item0:a"
    declined = "aita:m:item1:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(scored, score=False), _unit(declined, score=False, transcript=False)],
        status={"status": "completed"},
        artifacts={f"{scored.replace(':', '_')}_t.json": _completed_conv()},
        blocks=[_block(declined, block_id="b1", cls="unknown",
                       category="ambiguous_403", raw_body_excerpt=RAW_BODY_SENTINEL,
                       raw_body_sha256="a" * 64)],
        events=[_event(cls="unknown", category="ambiguous_403", event_id="e1",
                       unit_id=scored, attempt=1)],
        reviews=[
            _review("blocks-id:b1", "safety_declination", review_id="rvB",
                    resolved_category="hard_refusal", rationale=RATIONALE_SENTINEL,
                    unit_id=declined, issue_ref="ISSUE-123"),
            _review("events-id:e1", "safety_declination", review_id="rvE",
                    resolved_category="hard_refusal", rationale=RATIONALE_SENTINEL,
                    unit_id=scored),
        ],
    )
    return _experiment(tmp_path, [(cp, "pilot")])


def test_resolved_everything_emits_all_three_data_files(tmp_path):
    exp = _resolved_experiment(tmp_path)
    bundle_dir = _emit(exp, tmp_path)
    data = bundle_dir / "data"
    assert (data / "blocks.jsonl").exists()
    assert (data / "evidence.jsonl").exists()
    assert (data / "block_reviews.jsonl").exists()

    blocks = [json.loads(l) for l in (data / "blocks.jsonl").read_text().splitlines() if l]
    evidence = [json.loads(l) for l in (data / "evidence.jsonl").read_text().splitlines() if l]
    reviews = [json.loads(l) for l in (data / "block_reviews.jsonl").read_text().splitlines() if l]
    assert blocks and evidence and reviews
    # winner-scoped: every published fact belongs to member m1
    assert all(r["member_id"] == "m1" for r in blocks + evidence + reviews)
    # reviews carry event_ref + issue_ref, join to published facts
    published_refs = {r["event_ref"] for r in blocks + evidence}
    assert all(r["event_ref"] in published_refs for r in reviews)
    assert any(r.get("issue_ref") == "ISSUE-123" for r in reviews)


def test_resolved_bundle_hides_rawbody_and_rationale_by_default(tmp_path):
    exp = _resolved_experiment(tmp_path)
    bundle_dir = _emit(exp, tmp_path)
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RAW_BODY_SENTINEL not in blob
    assert RATIONALE_SENTINEL not in blob
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    assert manifest["contains_review_rationale"] is False
    # block_reviews records carry no rationale key by default
    reviews = [json.loads(l) for l in
               (bundle_dir / "data" / "block_reviews.jsonl").read_text().splitlines() if l]
    assert all("rationale" not in r for r in reviews)


def test_include_review_rationale_flag_publishes_and_stamps_manifest(tmp_path):
    exp = _resolved_experiment(tmp_path)
    bundle_dir = _emit(exp, tmp_path, include_review_rationale=True)
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RATIONALE_SENTINEL in blob                       # rationale now present
    assert RAW_BODY_SENTINEL not in blob                    # raw body STILL never
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    assert manifest["contains_review_rationale"] is True
    reviews = [json.loads(l) for l in
               (bundle_dir / "data" / "block_reviews.jsonl").read_text().splitlines() if l]
    assert any(r.get("rationale") == RATIONALE_SENTINEL for r in reviews)


def test_new_data_files_pass_privacy_audit(tmp_path):
    exp = _resolved_experiment(tmp_path)
    bundle_dir = _emit(exp, tmp_path)
    issues = bundle.audit_bundle_tree(bundle_dir)
    offending = [i for i in issues
                 if "evidence.jsonl" in i.path or "block_reviews.jsonl" in i.path
                 or "blocks.jsonl" in i.path]
    assert offending == []


# ---------------------------------------------------------------------------
# 5. v1-only regression — grandfathered backfill reviews resolve their blocks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Finding 5 — projection failure must BLOCK, never silently shrink the bundle
# ---------------------------------------------------------------------------


def test_broken_member_review_aborts_instead_of_shrinking(tmp_path):
    """A member whose projection raises (here: two active review heads on one
    fact) is dropped by union() — emitting the remainder would ship a bundle
    that silently omits it (Sol repro: ZERO-unit emit). The gate aborts instead,
    naming the member."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1")],
        reviews=[
            _review("blocks-id:b1", "safety_declination", review_id="rv1",
                    resolved_category="hard_refusal", unit_id=uid),
            _review("blocks-id:b1", "safety_declination", review_id="rv2",
                    resolved_category="hard_refusal", unit_id=uid),  # 2nd active head
        ],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value).lower()
    assert "active member" in msg or "shrunk" in msg or "union error" in msg
    assert "m1" in str(exc.value)
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


def test_dangling_supersession_in_ledger_aborts_emit(tmp_path):
    """final-gate2 F4: a member whose review ledger has a dangling supersession
    target makes the projection fail closed; emit must abort (a member
    validation failure), never ship a bundle that silently omits the run."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1")],
        # Hand-written record superseding a review that does not exist.
        reviews=[_review("blocks-id:b1", "safety_declination", review_id="rv1",
                         supersedes="ghost-review-id", unit_id=uid)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError):
        _emit(exp, tmp_path)
    # Fail-closed: nothing was promoted (no silent drop, no partial bundle).
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


def test_missing_unitview_winning_unit_gates():
    """A winning unit with no UnitView in the projection is NOT presumed
    publishable — the gate fails closed and names it (finding 5b)."""
    from suite_tools import review_projection as rp

    proj = rp.ProjectionResult(member_id="r1")   # empty events/units/obligations
    snap = bundle.RunSnapshot(
        member_id="m1", run_dir=Path("/nonexistent"),
        run_status={"status": "completed"}, fingerprints={},
        projection=proj, reviews_by_ref={})
    with pytest.raises(ValueError) as exc:
        bundle._run_publication_gate([snap], {"m1": {"aita:m:item0:a"}})
    msg = str(exc.value)
    assert "aita:m:item0:a" in msg
    assert "no projected state" in msg
    assert "m1" in msg


def test_request_conformance_failure_gates_publication():
    from suite_tools import review_projection as rp

    proj = rp.ProjectionResult(member_id="r1")
    snap = bundle.RunSnapshot(
        member_id="m1",
        run_dir=Path("/nonexistent"),
        run_status={"status": "completed"},
        fingerprints={},
        projection=proj,
        reviews_by_ref={},
        request_conformance={
            "conformant": False,
            "issues": [{"kind": "request_mismatch"}],
        },
    )

    with pytest.raises(ValueError, match="effective requests do not conform"):
        bundle._run_publication_gate([snap], {"m1": {"aita:m:item0:a"}})


def test_artifact_identity_failure_gates_publication():
    from suite_tools import review_projection as rp

    proj = rp.ProjectionResult(member_id="r1")
    snap = bundle.RunSnapshot(
        member_id="m1",
        run_dir=Path("/nonexistent"),
        run_status={"status": "completed"},
        fingerprints={},
        projection=proj,
        reviews_by_ref={},
        artifact_identity={
            "conformant": False,
            "issues": [{"kind": "missing_condition_hash"}],
        },
    )

    with pytest.raises(ValueError, match="saved transcript identities do not conform"):
        bundle._run_publication_gate([snap], {"m1": {"aita:m:item0:a"}})


def test_contract_mutation_between_capture_and_promote_aborts(tmp_path, monkeypatch):
    """RUN_CONTRACT.json is now fingerprinted; a mutation after capture but
    before the promote fails the recheck and aborts (finding 5c)."""
    uid = "aita:m:item0:a"
    run_dir = tmp_path / "runs" / "r1"
    cp = _mk_run(
        run_dir, run_id="r1", units=[_unit(uid, score=False)],
        status={"status": "completed"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()},
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    original_audit = bundle.audit_bundle_tree

    def mutate_then_audit(staging):
        contract = json.loads((run_dir / "RUN_CONTRACT.json").read_text())
        contract["_tamper"] = "mid-bundle"
        (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
        return original_audit(staging)

    monkeypatch.setattr(bundle, "audit_bundle_tree", mutate_then_audit)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "drift" in str(exc.value).lower() or "changed" in str(exc.value).lower()
    assert "RUN_CONTRACT.json" in str(exc.value)
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


def test_artifact_mutation_between_capture_and_promote_aborts(tmp_path, monkeypatch):
    """Per-unit artifact files are fingerprinted too; mutating one mid-bundle
    aborts (finding 5c)."""
    uid = "aita:m:item0:a"
    run_dir = tmp_path / "runs" / "r1"
    art_rel = f"{uid.replace(':', '_')}_t.json"
    cp = _mk_run(
        run_dir, run_id="r1", units=[_unit(uid, score=False)],
        status={"status": "completed"},
        artifacts={art_rel: _completed_conv()},
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    original_audit = bundle.audit_bundle_tree

    def mutate_then_audit(staging):
        (run_dir / art_rel).write_text(json.dumps(_completed_conv(turns=9)))
        return original_audit(staging)

    monkeypatch.setattr(bundle, "audit_bundle_tree", mutate_then_audit)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "drift" in str(exc.value).lower() or "changed" in str(exc.value).lower()
    assert not any((tmp_path / "out").glob("bundle-*"))


# ---------------------------------------------------------------------------
# 6b. Finding 2 — winner-selection race BEFORE snapshot capture
# ---------------------------------------------------------------------------


def test_winner_selection_race_before_capture_aborts(tmp_path, monkeypatch):
    """Every ACTIVE member is fingerprinted BEFORE union() and re-verified
    after: a member changing during winner selection — even a losing one whose
    change would flip the winner — aborts (finding 2)."""
    uid = "aita:m:item0:a"
    tr = f"{uid.replace(':', '_')}_t.json"
    m1 = _mk_run(
        tmp_path / "runs" / "pilot", run_id="pilot",
        units=[_unit(uid, score=False)], status={"status": "completed"},
        artifacts={tr: _completed_conv()}, started_at="2026-07-20T12:00:00Z")  # wins
    m2 = _mk_run(
        tmp_path / "runs" / "expansion", run_id="expansion",
        units=[_unit(uid, score=False)], status={"status": "completed"},
        artifacts={tr: _completed_conv()}, started_at="2026-07-19T09:00:00Z")  # loses
    exp = _experiment(tmp_path, [(m1, "pilot"), (m2, "expansion")])

    real_union = bundle._experiment_mod.union

    def racing_union(exp_dir):
        result = real_union(exp_dir)
        # a member's winner-selection input changes after union has read it
        status_path = tmp_path / "runs" / "expansion" / "RUN_STATUS.json"
        st = json.loads(status_path.read_text())
        st["started_at"] = "2026-07-21T23:59:59Z"   # would now beat pilot
        status_path.write_text(json.dumps(st))
        return result

    monkeypatch.setattr(bundle._experiment_mod, "union", racing_union)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value).lower()
    assert "winner selection" in msg or "drift" in msg
    assert "m2" in str(exc.value)                    # the mutated (losing) member
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


# ---------------------------------------------------------------------------
# 6b'. Manifest race — EXPERIMENT.json is inside the fingerprint boundary
# ---------------------------------------------------------------------------


def test_manifest_mutation_before_union_aborts(tmp_path, monkeypatch):
    """A member added to EXPERIMENT.json between the captured read and union()
    (Sol's repro: a newer winner carrying unresolved evidence) aborts — the
    manifest is fingerprinted and re-verified, so the bundle never emits a
    silently-shrunk 'members=1 winner=external' result."""
    uid = "aita:m:item0:a"
    tr = f"{uid.replace(':', '_')}_t.json"
    m1 = _mk_run(
        tmp_path / "runs" / "m1", run_id="m1",
        units=[_unit(uid, score=False)], status={"status": "completed"},
        artifacts={tr: _completed_conv()}, started_at="2026-07-19T09:00:00Z")
    newer = _mk_run(
        tmp_path / "runs" / "newer", run_id="newer",
        units=[_unit(uid, score=False)], status={"status": "completed"},
        artifacts={tr: _completed_conv()},
        events=[_event(cls="unknown", event_id="e_new", unit_id=uid)],
        started_at="2026-07-21T23:59:59Z")            # would become the winner
    exp = _experiment(tmp_path, [(m1, "pilot")])       # initially ONLY m1
    exp_json = exp / "EXPERIMENT.json"

    real_union = bundle._experiment_mod.union

    def racing_union(exp_dir):
        doc = json.loads(exp_json.read_text())
        doc["members"].append({"path": str(newer.resolve()), "role": "expansion"})
        exp_json.write_text(json.dumps(doc))           # member added before read
        return real_union(exp_dir)

    monkeypatch.setattr(bundle._experiment_mod, "union", racing_union)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    msg = str(exc.value).lower()
    assert "manifest" in msg or "winner" in msg or "drift" in msg
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


def test_unmapped_union_winner_aborts(tmp_path, monkeypatch):
    """A union winner whose chosen_member is not in the captured manifest is a
    fail-closed abort naming the member — never a silent skip."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False)], status={"status": "completed"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()})
    exp = _experiment(tmp_path, [(cp, "pilot")])

    real_union = bundle._experiment_mod.union
    external = "/nonexistent/external/RUN_CONTRACT.json"

    def external_winner_union(exp_dir):
        result = real_union(exp_dir)
        for unit in result.get("units") or []:
            unit["chosen_member"] = external      # winner not in the manifest
        return result

    monkeypatch.setattr(bundle._experiment_mod, "union", external_winner_union)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "not in the captured manifest" in str(exc.value)
    assert external in str(exc.value)
    assert not any((tmp_path / "out").glob("bundle-*"))


def test_contract_mutation_during_union_aborts(tmp_path, monkeypatch):
    """RUN_CONTRACT.json is fingerprinted BEFORE union and projected contracts
    are loaded only AFTER; a contract mutation around that window is caught by
    the post-union verify, so stale provenance is never published."""
    uid = "aita:m:item0:a"
    run_dir = tmp_path / "runs" / "r1"
    cp = _mk_run(
        run_dir, run_id="r1", units=[_unit(uid, score=False)],
        status={"status": "completed"},
        artifacts={f"{uid.replace(':', '_')}_t.json": _completed_conv()})
    exp = _experiment(tmp_path, [(cp, "pilot")])

    real_union = bundle._experiment_mod.union

    def mutating_union(exp_dir):
        result = real_union(exp_dir)
        contract = json.loads((run_dir / "RUN_CONTRACT.json").read_text())
        contract["_tamper"] = "during-union"
        (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
        return result

    monkeypatch.setattr(bundle._experiment_mod, "union", mutating_union)
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "RUN_CONTRACT.json" in str(exc.value)
    assert not any((tmp_path / "out").glob("bundle-*"))


# ---------------------------------------------------------------------------
# 6c. Finding 3 — instrument defect on a LOST unit gates a contributing member
# ---------------------------------------------------------------------------


def test_instrument_defect_on_lost_unit_gates_contributing_member(tmp_path):
    """Member A wins unit1 and has an adjudicated instrument defect on unit2,
    which member B won. The defect taints the whole contributing member A and
    blocks publication regardless of winner scoping (finding 3, D6 c3)."""
    u1 = "aita:m:item0:a"     # only A has it -> A wins
    u2 = "aita:m:item1:a"     # both have it -> B (later) wins
    a = _mk_run(
        tmp_path / "runs" / "A", run_id="A",
        units=[_unit(u1, score=False), _unit(u2, score=False, transcript=False)],
        status={"status": "completed"},
        artifacts={f"{u1.replace(':', '_')}_t.json": _completed_conv()},
        blocks=[_block(u2, block_id="bd", cls="environment", category="timeout_read")],
        reviews=[_review("blocks-id:bd", "instrument_defect", unit_id=u2)],
        started_at="2026-07-19T09:00:00Z")
    b = _mk_run(
        tmp_path / "runs" / "B", run_id="B",
        units=[_unit(u2, score=False)], status={"status": "completed"},
        artifacts={f"{u2.replace(':', '_')}_t.json": _completed_conv()},
        started_at="2026-07-20T12:00:00Z")
    exp = _experiment(tmp_path, [(a, "pilot"), (b, "expansion")])
    with pytest.raises(ValueError) as exc:
        _emit(exp, tmp_path)
    assert "instrument_defect" in str(exc.value)
    assert "m1" in str(exc.value)                     # member A is gated
    assert not any((tmp_path / "out").glob("bundle-*"))


# ---------------------------------------------------------------------------
# 6d. Finding 5 — scalar/malformed values in nominally-scalar public fields
# ---------------------------------------------------------------------------


def test_scalar_unit_value_is_dropped(tmp_path):
    """A non-dict `unit` (Sol planted a raw-body string as the unit value) is
    dropped wholesale — it never reaches a public bundle (finding 5)."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1", unit=RAW_BODY_SENTINEL)],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    rec = json.loads((bundle_dir / "data" / "blocks.jsonl").read_text().splitlines()[0])
    assert "unit" not in rec
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RAW_BODY_SENTINEL not in blob


def test_dict_in_scalar_field_is_dropped(tmp_path):
    """A dict smuggled into a nominally-scalar allowlisted field (`provider`) is
    dropped entirely by the allowlist rather than emitted (finding 5)."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1",
                       provider={"raw_body_excerpt": RAW_BODY_SENTINEL})],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    rec = json.loads((bundle_dir / "data" / "blocks.jsonl").read_text().splitlines()[0])
    assert "provider" not in rec
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RAW_BODY_SENTINEL not in blob


def test_dict_in_projection_hashed_field_refuses(tmp_path):
    """A dict in a field the projection hashes (`category`) can't even be
    projected — the bundle is REFUSED fail-closed, never emitting the content
    (finding 5, the 'refuse' branch)."""
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1",
                       category={"raw_body_excerpt": RAW_BODY_SENTINEL})],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    with pytest.raises(ValueError):
        _emit(exp, tmp_path)
    assert not any((tmp_path / "out").glob("bundle-*"))
    assert not any((tmp_path / "out").glob(".*.tmp"))


# ---------------------------------------------------------------------------
# 7. Finding 6 — nested raw-body leak through the `unit` allowlist
# ---------------------------------------------------------------------------


def test_nested_rawbody_in_unit_is_projected_away(tmp_path):
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1",
                       unit={"item_idx": 0, "side": "side_a",
                             "raw_body_excerpt": RAW_BODY_SENTINEL})],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    rec = json.loads((bundle_dir / "data" / "blocks.jsonl").read_text().splitlines()[0])
    assert "raw_body_excerpt" not in rec["unit"]
    assert rec["unit"] == {"item_idx": 0, "side": "side_a"}   # identity only
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RAW_BODY_SENTINEL not in blob


def test_nested_rawbody_in_list_and_deep_dict_is_projected_away(tmp_path):
    uid = "aita:m:item0:a"
    cp = _mk_run(
        tmp_path / "runs" / "r1", run_id="r1",
        units=[_unit(uid, score=False, transcript=False)],
        status={"status": "completed"},
        blocks=[_block(uid, block_id="b1",
                       unit={"item_idx": 0,
                             "notes": [RAW_BODY_SENTINEL,
                                       {"raw_body_excerpt": RAW_BODY_SENTINEL}],
                             "deep": {"a": {"raw_body_excerpt": RAW_BODY_SENTINEL}}})],
    )
    exp = _experiment(tmp_path, [(cp, "pilot")])
    bundle_dir = _emit(exp, tmp_path)
    rec = json.loads((bundle_dir / "data" / "blocks.jsonl").read_text().splitlines()[0])
    assert rec["unit"] == {"item_idx": 0}      # list + nested dict both dropped
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert RAW_BODY_SENTINEL not in blob


def test_auditor_catches_forceplanted_nested_rawbody(tmp_path):
    """Defense in depth: even a raw-body field force-planted into public JSON
    AFTER projection (nested in a dict, and in a list) is caught by
    audit_bundle_tree via the forbidden-field-name scan (finding 6b)."""
    exp = _resolved_experiment(tmp_path)
    bundle_dir = _emit(exp, tmp_path)
    (bundle_dir / "data" / "planted_dict.json").write_text(
        json.dumps({"outer": {"raw_body_excerpt": RAW_BODY_SENTINEL}}))
    (bundle_dir / "data" / "planted_list.json").write_text(
        json.dumps({"outer": [{"raw_response": RAW_BODY_SENTINEL}]}))
    issues = bundle.audit_bundle_tree(bundle_dir)
    assert any("raw_body_excerpt" in i.path and i.reason == "private field name"
               for i in issues)
    assert any("raw_response" in i.path and i.reason == "private field name"
               for i in issues)


# ---------------------------------------------------------------------------
# 8. v1-only regression
# ---------------------------------------------------------------------------


def test_v1_backfilled_run_bundles_and_hides_rationale(tmp_path):
    """A v1 backfilled EPIS run (v1 blocks + v1 safety_declination reviews)
    bundles as today; the backfill rationale (which echoes an evidence_pointer)
    is projected out of block_reviews.jsonl by default."""
    from _review_projection_fixture import build_backfilled_epis_run

    run = build_backfilled_epis_run(tmp_path / "runs" / "epis")
    contract_path = run / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    from suite_tools.run_contract import legacy_v1_provenance_hashes

    contract["provenance"] = legacy_v1_provenance_hashes(contract)
    contract_path.write_text(json.dumps(contract))
    # mark the run terminal-completed so the snapshot accepts it
    status = json.loads((run / "RUN_STATUS.json").read_text()) if (run / "RUN_STATUS.json").exists() else {"attempt_number": 1, "started_at": "2026-07-20T00:00:00Z"}
    status["status"] = "completed"
    (run / "RUN_STATUS.json").write_text(json.dumps(status))

    exp = _experiment(tmp_path, [(run / "RUN_CONTRACT.json", "pilot")],
                      experiment_id="epis-v1", modules=("epis",))
    bundle_dir = _emit(exp, tmp_path)

    reviews = [json.loads(l) for l in
               (bundle_dir / "data" / "block_reviews.jsonl").read_text().splitlines() if l]
    assert reviews, "backfill reviews should be published (winner-scoped)"
    assert all("rationale" not in r for r in reviews)
    # the backfill rationale text ("Retro-backfill:") must appear nowhere
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert "Retro-backfill" not in blob
