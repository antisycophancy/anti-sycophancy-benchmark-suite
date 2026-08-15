import json

from suite_tools.progress_dedupe import completed_unit_keys


def _ev(name, unit_id, attempt=1):
    return {"event": name, "unit_id": unit_id, "attempt_number": attempt}


def test_disjoint_units_across_attempts_count_eight():
    events = [_ev("conversation_completed", f"u{i}", 1) for i in range(3)]
    events += [_ev("conversation_completed", f"u{i}", 2) for i in range(3, 8)]
    assert len(completed_unit_keys(events)) == 8


def test_same_unit_completed_twice_counts_one():
    assert len(completed_unit_keys([_ev("conversation_completed", "u0", 1),
                                    _ev("conversation_completed", "u0", 2)])) == 1


def test_reused_units_count_as_completed():
    events = [_ev("conversation_completed", f"u{i}", 1) for i in range(3)]
    events += [_ev("conversation_reused", f"u{i}", 2) for i in range(3)]
    events += [_ev("conversation_completed", f"u{i}", 2) for i in range(3, 8)]
    assert len(completed_unit_keys(events)) == 8


def test_sus_runs_do_not_collapse_when_unit_id_present():
    events = [
        {"event": "sus_run_completed", "unit_id": "sus:m:bridge:run1"},
        {"event": "sus_run_reused", "unit_id": "sus:m:bridge:run2"},
        {"event": "block_recorded", "unit_id": "sus:m:bridge:run3"},
    ]
    assert len(completed_unit_keys(events)) == 3


def test_environment_failure_not_counted():
    assert len(completed_unit_keys([{"event": "conversation_failed", "unit_id": "u0",
                                     "failure_status": "failed_billing"}])) == 0


# ---------------------------------------------------------------------------
# (a) Legacy-stream fallback key tests
# ---------------------------------------------------------------------------

def test_legacy_fallback_same_unit_twice_counts_one():
    # Events without unit_id but with module-specific identity fields
    # (model + item_idx + side) fall back to a tuple key. Two events for the
    # same logical unit must collapse to 1.
    e1 = {"event": "conversation_completed", "model": "m/foo", "item_idx": 3, "side": "side_a"}
    e2 = {"event": "conversation_completed", "model": "m/foo", "item_idx": 3, "side": "side_a"}
    assert len(completed_unit_keys([e1, e2])) == 1


def test_legacy_fallback_distinct_identity_fields_count_separately():
    # Different item_idx → different fallback key → different unit.
    e1 = {"event": "conversation_completed", "model": "m/foo", "item_idx": 3, "side": "side_a"}
    e2 = {"event": "conversation_completed", "model": "m/foo", "item_idx": 4, "side": "side_a"}
    assert len(completed_unit_keys([e1, e2])) == 2


def test_identity_less_events_use_sentinel_and_count_individually():
    # Events with no unit_id AND no fallback fields (model/scenario/item_idx/…)
    # all map to the all-None tuple.  Without the sentinel they would collapse
    # to 1; the ("_seq", n) sentinel gives each a unique key so bare legacy
    # streams still count correctly.
    events = [
        {"event": "conversation_completed"},
        {"event": "conversation_completed"},
    ]
    # With sentinel: ("_seq", 0) and ("_seq", 1) → 2 distinct keys.
    assert len(completed_unit_keys(events)) == 2


# ---------------------------------------------------------------------------
# (b) Cost path remains UNFILTERED by progress dedupe
# ---------------------------------------------------------------------------

def test_cost_path_unfiltered_by_progress_dedupe(tmp_path):
    # Cost accumulation lives in suite_tools/run_monitor.py::record_usage
    # (lines ~469-595). It is entirely separate from progress_dedupe and
    # accumulates on EVERY call regardless of unit identity.
    # This test confirms two same-unit attempts → cost doubles, NOT halved.
    from suite_tools.run_monitor import RunMonitor
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    # Simulate attempt 1 and attempt 2 of the same unit, each costing $0.01.
    monitor.record_usage("m", {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 5}, role="test")
    monitor.record_usage("m", {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 5}, role="test")
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    # Progress dedupe would count 1 unit; cost MUST accumulate both calls → 0.02.
    assert status["cost"]["total_cost_usd"] == 0.02
    assert status["cost"]["total_calls"] == 2
    # Confirm progress dedupe counts the same events as 1 (deduplicated).
    dup_events = [
        {"event": "conversation_completed", "unit_id": "u0", "attempt_number": 1},
        {"event": "conversation_completed", "unit_id": "u0", "attempt_number": 2},
    ]
    assert len(completed_unit_keys(dup_events)) == 1  # dedupe: 1 distinct unit
    assert len(dup_events) == 2  # raw event list has 2 entries (cost iterates all)
