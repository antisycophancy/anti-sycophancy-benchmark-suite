"""Tests for suite_tools.merge_panel_gap_rescore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from suite_tools.merge_panel_gap_rescore import _row_key, merge, update_run_status, write_merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    condition_id: str,
    scenario: str,
    run_number: int,
    num_judges: int = 3,
    sus: float = 50.0,
    model: str = "test-model",
) -> dict:
    """Minimal synthetic scored row."""
    return {
        "model": model,
        "label": f"{model} label",
        "provider_api": "openai_compatible",
        "condition_id": condition_id,
        "condition_hash": f"hash-{condition_id}",
        "condition_metadata": {},
        "request_options": {},
        "scenario": scenario,
        "scenario_name": scenario,
        "run_number": run_number,
        "score": {"sus": sus, "irq": 9.0, "pr": 9.0, "er": 9.0, "ca": 9.0},
        "score_state": "scored",
        "post_analysis": {
            "num_judges": num_judges,
            "judge_panel_complete": num_judges == 3,
            "judge_scores": [],
            "irq": 9,
            "pr": 9,
            "er": 9,
            "ca": 9,
        },
        "conversation": [{"role": "user", "content": "test"}],
        "phases": {},
        "escalation_mode": "progressive",
        "capitulated": False,
        "capitulation_phase_key": None,
        "capitulation_strategy_id": None,
        "capitulation_strategy": None,
        "phase_failure_mode": None,
        "failure_mode": "no_capitulation",
        "failure_class": "structural_safe",
        "failure_mode_source": "classifier",
        "classification_conflict": False,
        "classification_conflict_reason": None,
    }


def _write_conversations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def _make_run_status(path: Path, scored: int, expected: int) -> None:
    status = {
        "schema_version": "benchmark-run-ledger-v1",
        "module": "sus",
        "stage": "scoring",
        "status": "failed_scoring",
        "validity": "not_score_ready",
        "output_dir": str(path.parent),
        "started_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "scored_results": scored,
        "expected_results": expected,
        "score_failures": [{"index": 5, "rerun_recommended": True}],
        "failure_reason": "SUS scoring incomplete",
        "failure_stage": "judge_panel",
        "partial_results_path": str(path.parent / "FINAL_RESULTS-partial.json"),
        "rerun_recommended": True,
        "counters": {},
        "metadata": {"judge_panel": ["a", "b", "c"]},
    }
    path.write_text(json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_row_key():
    row = {"condition_id": "cond-1", "scenario": "bridge", "run_number": 5}
    assert _row_key(row) == ("cond-1", "bridge", 5)


def test_row_key_defaults():
    assert _row_key({}) == ("", "", 0)


def test_merge_inserts_rescore_row_at_correct_position(tmp_path):
    """Rescore row replaces the missing partial row; order follows original gen file."""
    sus_dir = tmp_path / "sus"
    rescore_dir = sus_dir / "panel-gap-rescore"

    # 5 original rows; row (cond-a, bridge, 3) is the failing one.
    original = [
        _make_row("cond-a", "bridge", 1),
        _make_row("cond-a", "bridge", 2),
        _make_row("cond-a", "bridge", 3),  # this one failed
        _make_row("cond-a", "bridge", 4),
        _make_row("cond-a", "bridge", 5),
    ]
    _write_conversations(
        sus_dir / "sus-bench-20260101-000000-conversations.json", original
    )

    # Partial has 4 rows (missing run_number=3).
    partial = [r for r in original if r["run_number"] != 3]
    _write_conversations(sus_dir / "FINAL_RESULTS-partial-conversations.json", partial)

    # Rescore has only the failed row with num_judges=3.
    rescore_row = _make_row("cond-a", "bridge", 3, num_judges=3, sus=42.0)
    _write_conversations(
        rescore_dir / "RESCORE_RESULTS-conversations.json", [rescore_row]
    )

    merged = merge(sus_dir, rescore_dir)

    assert len(merged) == 5
    # Order must match original.
    assert [r["run_number"] for r in merged] == [1, 2, 3, 4, 5]
    # The rescored row replaces the slot at position 2.
    assert merged[2]["score"]["sus"] == 42.0
    assert merged[2]["post_analysis"]["num_judges"] == 3


def test_merge_raises_if_rescore_row_has_incomplete_panel(tmp_path):
    sus_dir = tmp_path / "sus"
    rescore_dir = sus_dir / "panel-gap-rescore"

    original = [_make_row("cond-x", "bridge", 1)]
    _write_conversations(
        sus_dir / "sus-bench-20260101-000000-conversations.json", original
    )
    _write_conversations(sus_dir / "FINAL_RESULTS-partial-conversations.json", [])

    bad_rescore = _make_row("cond-x", "bridge", 1, num_judges=2)
    _write_conversations(
        rescore_dir / "RESCORE_RESULTS-conversations.json", [bad_rescore]
    )

    with pytest.raises(ValueError, match="num_judges=2"):
        merge(sus_dir, rescore_dir)


def test_merge_raises_if_row_missing_from_both(tmp_path):
    sus_dir = tmp_path / "sus"
    rescore_dir = sus_dir / "panel-gap-rescore"

    original = [
        _make_row("cond-y", "bridge", 1),
        _make_row("cond-y", "bridge", 2),  # not in partial or rescore
    ]
    _write_conversations(
        sus_dir / "sus-bench-20260101-000000-conversations.json", original
    )
    # Only row 1 in partial, nothing in rescore.
    _write_conversations(
        sus_dir / "FINAL_RESULTS-partial-conversations.json",
        [_make_row("cond-y", "bridge", 1)],
    )
    _write_conversations(rescore_dir / "RESCORE_RESULTS-conversations.json", [])

    with pytest.raises(ValueError, match="Merge incomplete"):
        merge(sus_dir, rescore_dir)


def test_write_merged_produces_both_files(tmp_path):
    sus_dir = tmp_path / "sus"
    rescore_dir = sus_dir / "panel-gap-rescore"

    original = [_make_row("cond-z", "bridge", i) for i in range(1, 4)]
    _write_conversations(
        sus_dir / "sus-bench-20260101-000000-conversations.json", original
    )
    _write_conversations(
        sus_dir / "FINAL_RESULTS-partial-conversations.json", original
    )
    rescore_row = _make_row("cond-z", "bridge", 1)
    _write_conversations(
        rescore_dir / "RESCORE_RESULTS-conversations.json", [rescore_row]
    )
    # Write a minimal RESCORE_RESULTS.json so write_merged can load cost.
    (rescore_dir / "RESCORE_RESULTS.json").write_text(
        json.dumps({"cost": {"total_cost_usd": 0.01}})
    )

    merged = merge(sus_dir, rescore_dir)
    out_path, conv_path = write_merged(merged, sus_dir, rescore_dir, run_id="test-run")

    assert out_path.exists()
    assert conv_path.exists()

    summary = json.loads(out_path.read_text())
    assert summary["run_id"] == "test-run"
    assert "aggregated" in summary

    convs = json.loads(conv_path.read_text())
    assert len(convs) == 3


def test_update_run_status_sets_completed(tmp_path):
    sus_dir = tmp_path / "sus"
    sus_dir.mkdir()
    status_path = sus_dir / "RUN_STATUS.json"
    _make_run_status(status_path, scored=99, expected=100)

    merged = [_make_row("c", "s", i) for i in range(100)]
    update_run_status(sus_dir, merged)

    status = json.loads(status_path.read_text())
    assert status["status"] == "completed"
    assert status["validity"] == "score_ready"
    assert status["scored_results"] == 100
    assert "score_failures" not in status
    assert "failure_reason" not in status
    assert "rerun_recommended" not in status
    assert "results_path" in status
