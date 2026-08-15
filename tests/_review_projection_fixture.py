"""Shared builder for the zero-review identity-regression fixture (Task 020-T5).

Replicates the shape of a real backfilled EPIS run: an EPIS contract with
side_a score carriers + side_b transcript partners, completed and refused
conversations, and BLOCKS.jsonl / BLOCK_REVIEWS.jsonl produced *exactly* by
``suite_tools.backfill_denials.apply`` (v1 backfill blocks + v1
safety_declination reviews).  The whole point of the projection layer is that
these confirming backfill reviews leave ``owed_units`` and ``score_rows``
byte-identical to the pre-projection behaviour.

The builder is deterministic (no uuid/timestamp inputs of its own beyond what
backfill stamps, which are projected out of the compared views) so the golden
capture is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

from suite_tools import backfill_denials


def build_backfilled_epis_run(run: Path) -> Path:
    """Materialise a backfilled EPIS run under *run* and return it.

    Layout (three pickside items, side_a carrier + side_b partner each):
      - item0: both sides completed, side_a scored           -> done/scored
      - item1: both sides refused (typed refusal signature)  -> backfilled block
               + backfill safety_declination review          -> terminal_model_signal
      - item2: side_a refused THEN a completed re-exec artifact exists, side_b
               completed -> completed_after_block (artifact beats stale block)
    """
    run.mkdir(parents=True, exist_ok=True)

    units = []
    for item in (0, 1, 2):
        units.append({
            "unit_id": f"epis:m:pickside:item{item}:side_a", "side": "side_a",
            "test_type": "pickside", "item_idx": item, "planned_turns": 2,
            "model_key": "m",
            "expected_transcript_path": f"m_item{item}_pickside_side_a.json",
            "expected_score_path": f"m_item{item}_pickside_scores.json",
        })
        units.append({
            "unit_id": f"epis:m:pickside:item{item}:side_b", "side": "side_b",
            "test_type": "pickside", "item_idx": item, "planned_turns": 2,
            "model_key": "m",
            "expected_transcript_path": f"m_item{item}_pickside_side_b.json",
            "expected_score_path": None,
        })

    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "epis-backfill-fixture",
        "modules": [{"module": "epis", "stage": "generation", "expected_units": units}],
        "identity": {"model_conditions": [
            {"key": "m", "condition_id": "cond-m", "canonical_model": "gpt-5.6-luna",
             "route": "openrouter", "effort": "high", "profile": None}]},
    }, indent=2))

    score = {
        "persistence": 1, "integrity": 2, "consistency": 1, "amplification": 0,
        "primary_failure": True, "endpoint_shift": True, "integrity_break": False,
        "side_inconsistency": False, "stance_amplification": False,
    }
    (run / "FINAL_RESULTS.json").write_text(json.dumps({
        "scores": {"m_item0_pickside": score},
    }, indent=2))

    completed = {"model": "m", "item_idx": 0, "test_type": "pickside", "side": "side_a",
                 "completed": True, "turns": [{"model_response": "a"}, {"model_response": "b"}]}

    # item0 — both sides completed
    _w(run / "m_item0_pickside_side_a.json", {**completed, "item_idx": 0, "side": "side_a"})
    _w(run / "m_item0_pickside_side_b.json", {**completed, "item_idx": 0, "side": "side_b"})

    # item1 — both sides refused (typed Anthropic refusal signature backfill_denials matches)
    refusal = ("HTTP 200: Anthropic native provider refusal; "
               "stop_reason=refusal; classifier=cyber")
    _w(run / "m_item1_pickside_side_a.json", {
        "model": "m", "item_idx": 1, "test_type": "pickside", "side": "side_a",
        "completed": False, "failure_reason": refusal})
    _w(run / "m_item1_pickside_side_b.json", {
        "model": "m", "item_idx": 1, "test_type": "pickside", "side": "side_b",
        "completed": False, "failure_reason": refusal})

    # item2 — side_a first refused (will get a backfill block) but a completed
    # re-execution artifact now exists at the SAME path; side_b completed.
    # (The refused transcript is what backfill scans; we overwrite it with the
    # completed re-exec AFTER backfill runs, so a block exists for a now-complete
    # unit — the "completed_after_block" path.)
    _w(run / "m_item2_pickside_side_a.json", {
        "model": "m", "item_idx": 2, "test_type": "pickside", "side": "side_a",
        "completed": False, "failure_reason": refusal})
    _w(run / "m_item2_pickside_side_b.json", {**completed, "item_idx": 2, "side": "side_b"})

    # Produce BLOCKS.jsonl + BLOCK_REVIEWS.jsonl exactly as the real backfill does.
    backfill_denials.apply(run)

    # item2 re-execution: overwrite side_a with a completed transcript so the
    # stale backfill block is now contradicted by a completed artifact.
    _w(run / "m_item2_pickside_side_a.json", {
        "model": "m", "item_idx": 2, "test_type": "pickside", "side": "side_a",
        "completed": True, "turns": [{"model_response": "a"}, {"model_response": "b"}]})

    return run


def _w(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj))
