import json
from suite_tools.unit_state import (
    aita_unit_state, epis_unit_state, sus_unit_state,
    is_terminal_model_signal, terminal_reuse_event_name,
)


def test_aita_complete_is_completed():
    assert aita_unit_state({"turns": [1, 2, 3, 4, 5], "completed": True}, 5) == "completed"


def test_aita_partial_is_owed():
    assert aita_unit_state({"turns": [1, 2]}, 5) == "owed"


def test_aita_refusal_is_terminal_even_if_partial():
    assert aita_unit_state({"turns": [1], "completed": False, "provider_refusal": True}, 5) == "terminal_model_signal"


def test_epis_refusal_via_failure_reason_string():
    conv = {"turns": [], "completed": False, "failure_reason": "HTTP 200: stop_reason=refusal; classifier=cyber"}
    assert epis_unit_state(conv, 3) == "terminal_model_signal"


def test_helpers():
    assert is_terminal_model_signal("terminal_model_signal") is True
    assert is_terminal_model_signal("completed") is False
    assert terminal_reuse_event_name({"provider_refusal": True}) == "conversation_reused_provider_refusal"
    assert terminal_reuse_event_name({"output_budget_exhausted": True}) == "conversation_reused_output_budget_exhausted"


def _write_sus(tmp_path, result):
    from sus_bench import runner as sus_runner
    return sus_runner._write_live_transcript_artifact(
        tmp_path,
        model={"id": "m", "label": "M"},
        scenario={"id": "bridge", "name": "Bridge"},
        result=result,
        run_number=1,
    )


def test_sus_completed_via_real_writer(tmp_path):
    result = {"conversation": [], "turn_outcomes": [], "score_state": "needs_scoring",
              "phases": {"elicit": {}, "extract": {}, "follow": {}, "escalate_1": {}, "escalate_2": {}}}
    persisted = json.loads(_write_sus(tmp_path, result).read_text())
    assert sus_unit_state(persisted, 2) == "completed"


def test_sus_terminal_via_real_writer(tmp_path):
    result = {"conversation": [], "turn_outcomes": [], "score_state": "excluded_provider_refusal",
              "exclusion_reason": "provider_refusal", "phases": {"elicit": {}}}
    persisted = json.loads(_write_sus(tmp_path, result).read_text())
    assert sus_unit_state(persisted, 2) == "terminal_model_signal"


def test_sus_incomplete_phases_via_real_writer_is_owed(tmp_path):
    result = {"conversation": [], "turn_outcomes": [], "score_state": "needs_scoring", "phases": {"elicit": {}}}
    persisted = json.loads(_write_sus(tmp_path, result).read_text())
    assert sus_unit_state(persisted, 2) == "owed"
