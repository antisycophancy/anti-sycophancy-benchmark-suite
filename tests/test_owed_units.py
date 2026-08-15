"""Tests for suite_tools.owed_units — TDD RED phase."""
import json
from pathlib import Path
import pytest
from suite_tools import owed_units


def _contract(run: Path, module: str, units: list):
    (run / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "modules": [{"module": module, "stage": "generation", "expected_units": units}],
    }))


def test_aita_three_states(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "m_item0_side_a.json", "planned_turns": 5},
        {"unit_id": "aita:m:item1:side_a", "expected_transcript_path": "m_item1_side_a.json", "planned_turns": 5},
        {"unit_id": "aita:m:item2:side_a", "expected_transcript_path": "m_item2_side_a.json", "planned_turns": 5},
    ])
    (run / "m_item0_side_a.json").write_text(json.dumps({"turns": [1, 2, 3, 4, 5], "completed": True}))
    (run / "m_item1_side_a.json").write_text(json.dumps({"turns": [], "provider_refusal": True, "completed": False}))
    out = owed_units.owed_units(run, module="aita")
    assert out["counts"] == {"done": 1, "terminal_model_signal": 1, "owed": 1}


def test_block_only_terminal_wins_by_unit_id(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "sus", [
        {"unit_id": "sus:m:bridge:run1", "expected_transcript_path": "transcripts/m_bridge_run1.json",
         "planned_escalations": 2, "scenario": "bridge", "run_number": 1},
    ])
    (run / "transcripts").mkdir()
    (run / "transcripts" / "m_bridge_run1.json").write_text(json.dumps({"score_state": "needs_scoring", "phases": {}}))
    (run / "BLOCKS.jsonl").write_text(json.dumps({
        "unit_id": "sus:m:bridge:run1", "model": "m", "evidence_class": "model_signal", "category": "refusal",
    }) + "\n")
    out = owed_units.owed_units(run, module="sus")
    assert out["counts"]["terminal_model_signal"] == 1


def test_missing_contract_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        owed_units.owed_units(tmp_path, module="aita")


def test_epis_side_b_owed_when_transcript_missing(tmp_path):
    """EPIS pickside: side_a done, side_b transcript absent → side_b is owed."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "epis", [
        {"unit_id": "epis:m:pickside:item0:side_a",
         "expected_transcript_path": "m_item0_pickside_side_a.json",
         "planned_turns": 3},
        {"unit_id": "epis:m:pickside:item0:side_b",
         "expected_transcript_path": "m_item0_pickside_side_b.json",
         "planned_turns": 3},
    ])
    # side_a fully completed (3 turns)
    (run / "m_item0_pickside_side_a.json").write_text(
        json.dumps({"turns": [1, 2, 3]})
    )
    # side_b transcript intentionally absent
    out = owed_units.owed_units(run, module="epis")
    assert out["counts"]["done"] == 1
    assert out["counts"]["owed"] == 1


def test_schema_version_and_structure(tmp_path):
    """Return dict must have the schema-documented keys."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "aita", [
        {"unit_id": "aita:m:item0:side_a", "expected_transcript_path": "t.json", "planned_turns": 5},
    ])
    out = owed_units.owed_units(run, module="aita")
    assert out["schema_version"] == "benchmark-owed-units-v1"
    assert out["module"] == "aita"
    assert "counts" in out
    assert "units" in out
    assert isinstance(out["units"], list)
    # Each unit entry must have the four spec'd keys
    entry = out["units"][0]
    assert "unit_id" in entry
    assert "state" in entry
    assert "artifact" in entry
    assert "reason" in entry


def test_completed_artifact_beats_stale_block_entry(tmp_path):
    """A completed transcript produced by re-execution overrides a stale BLOCKS entry."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "aita", [
        {"unit_id": "aita:m:item0:side_a",
         "expected_transcript_path": "m_item0_side_a.json",
         "planned_turns": 3},
    ])
    # Completed transcript (re-execution produced enough turns)
    (run / "m_item0_side_a.json").write_text(json.dumps({"turns": [1, 2, 3]}))
    # Stale BLOCKS entry for the same unit from a prior execution
    (run / "BLOCKS.jsonl").write_text(json.dumps({
        "unit_id": "aita:m:item0:side_a",
        "model": "m",
        "evidence_class": "model_signal",
        "category": "refusal",
    }) + "\n")
    out = owed_units.owed_units(run, module="aita")
    assert out["counts"]["done"] == 1
    assert out["counts"]["terminal_model_signal"] == 0
    entry = out["units"][0]
    assert entry["state"] == "done"
    assert entry["reason"] == "completed_after_block"


def test_unknown_module_raises_value_error(tmp_path):
    """An unrecognised module name must raise ValueError naming the module."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "xray", [
        {"unit_id": "xray:m:item0", "expected_transcript_path": "t.json", "planned_turns": 3},
    ])
    (run / "t.json").write_text(json.dumps({"turns": [1, 2, 3]}))
    with pytest.raises(ValueError, match="xray"):
        owed_units.owed_units(run, module="xray")


def test_epistemic_alias_routes_to_epis_predicate(tmp_path):
    """A contract with module="epistemic" must route to the epis predicate.

    "epistemic" is the alias used by real run contracts; it must produce the
    same classification as "epis" and must NOT raise ValueError.
    """
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "epistemic", [
        {"unit_id": "epis:m:pickside:item0:side_a",
         "expected_transcript_path": "t_side_a.json",
         "planned_turns": 2},
        {"unit_id": "epis:m:pickside:item1:side_a",
         "expected_transcript_path": "t_side_a_1.json",
         "planned_turns": 2},
    ])
    # First unit complete; second absent (owed)
    (run / "t_side_a.json").write_text(json.dumps({"turns": [1, 2]}))
    out = owed_units.owed_units(run)  # module=None → all modules
    assert out["counts"]["done"] == 1
    assert out["counts"]["owed"] == 1
    assert out["counts"]["terminal_model_signal"] == 0


def test_epistemic_alias_with_explicit_filter(tmp_path):
    """owed_units(module='epistemic') and owed_units(module='epis') must both work."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "epistemic", [
        {"unit_id": "epis:m:delusion:item0:side_a",
         "expected_transcript_path": "t.json",
         "planned_turns": 1},
    ])
    (run / "t.json").write_text(json.dumps({"turns": [1]}))
    # Both aliases must resolve the same module
    out_raw = owed_units.owed_units(run, module="epistemic")
    out_epis = owed_units.owed_units(run, module="epis")
    assert out_raw["counts"]["done"] == out_epis["counts"]["done"] == 1


def test_truly_unknown_module_still_raises(tmp_path):
    """normalize_module_name must raise ValueError for unknown module names."""
    from suite_tools.suite_registry import normalize_module_name
    with pytest.raises(ValueError, match="Unknown module"):
        normalize_module_name("xray")


def test_sus_three_states_no_blocks(tmp_path):
    """SUS without BLOCKS: completed / terminal / owed classification."""
    run = tmp_path / "run"
    run.mkdir()
    _contract(run, "sus", [
        # done: elicit present + 2 escalations
        {"unit_id": "sus:m:bridge:run1",
         "expected_transcript_path": "bridge_done.json",
         "planned_escalations": 2},
        # terminal: excluded_provider_refusal score_state
        {"unit_id": "sus:m:tunnel:run1",
         "expected_transcript_path": "tunnel_term.json",
         "planned_escalations": 2},
        # owed: file missing
        {"unit_id": "sus:m:cliff:run1",
         "expected_transcript_path": "cliff_owed.json",
         "planned_escalations": 2},
    ])
    (run / "bridge_done.json").write_text(json.dumps({
        "score_state": "needs_scoring",
        "phases": {"elicit": {}, "escalate_1": {}, "escalate_2": {}},
    }))
    (run / "tunnel_term.json").write_text(json.dumps({
        "score_state": "excluded_provider_refusal",
        "phases": {"elicit": {}},
    }))
    out = owed_units.owed_units(run, module="sus")
    assert out["counts"] == {"done": 1, "terminal_model_signal": 1, "owed": 1}


def test_sus_legacy_expected_path_resolves_effort_qualified_transcript(tmp_path):
    """Legacy prepared contracts omitted request options from SUS filenames."""
    run = tmp_path / "run"
    transcripts = run / "transcripts"
    transcripts.mkdir(parents=True)
    unit_id = "sus:claude-sonnet-5-native-low-128k:bridge_heights:run1"
    _contract(run, "sus", [
        {
            "unit_id": unit_id,
            "expected_transcript_path": (
                "transcripts/claude-sonnet-5_bridge_heights_run1.json"
            ),
            "planned_escalations": 1,
        },
    ])
    actual = transcripts / (
        "claude-sonnet-5_bridge_heights_run1_"
        "optionsmax_tokens-128000-output_config-effort-low.json"
    )
    actual.write_text(json.dumps({
        "unit_id": unit_id,
        "score_state": "needs_scoring",
        "phases": {"elicit": {}, "escalate_1": {}},
    }))

    out = owed_units.owed_units(run, module="sus")

    assert out["counts"] == {"done": 1, "terminal_model_signal": 0, "owed": 0}
    assert out["units"][0]["artifact"] == str(actual)


def test_sus_legacy_expected_path_rejects_wrong_condition_transcript(tmp_path):
    """A similarly named effort artifact must match the contract unit_id."""
    run = tmp_path / "run"
    transcripts = run / "transcripts"
    transcripts.mkdir(parents=True)
    unit_id = "sus:claude-sonnet-5-native-low-128k:bridge_heights:run1"
    _contract(run, "sus", [
        {
            "unit_id": unit_id,
            "expected_transcript_path": (
                "transcripts/claude-sonnet-5_bridge_heights_run1.json"
            ),
            "planned_escalations": 1,
        },
    ])
    wrong_condition = (
        transcripts / "claude-sonnet-5_bridge_heights_run1_options-effort-high.json"
    )
    wrong_condition.write_text(json.dumps({
        "unit_id": "sus:claude-sonnet-5-native-high-128k:bridge_heights:run1",
        "score_state": "needs_scoring",
        "phases": {"elicit": {}, "escalate_1": {}},
    }))

    out = owed_units.owed_units(run, module="sus")

    assert out["counts"] == {"done": 0, "terminal_model_signal": 0, "owed": 1}
