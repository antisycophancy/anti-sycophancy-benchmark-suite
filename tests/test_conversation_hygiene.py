import json

from suite_tools.conversation_hygiene import (
    blocking_issue_summaries,
    canonical_messages,
    render_chat_transcript,
    scan_paths,
    scan_record,
    summarize_issues,
)


def _codes(issues):
    return {issue.code for issue in issues}


def test_accepts_sus_openai_style_conversation():
    record = {
        "model": "openai/gpt-test",
        "scenario": "bridge_heights",
        "conversation": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "I can help with safe alternatives."},
        ],
    }

    assert scan_record(record) == []
    assert canonical_messages(record) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I can help with safe alternatives."},
    ]


def test_blocks_aita_epistemic_turn_row_runtime_failures():
    record = {
        "model": "pipeline/test",
        "test_type": "pickside",
        "planned_num_turns": 2,
        "actual_num_turns": 2,
        "completed": True,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "[empty response from model]"},
            {"turn": 2, "user_message": "u2", "model_response": "[TIMEOUT/ERROR: KeyError]"},
        ],
    }

    codes = _codes(scan_record(record, source="aita-bench/results/model.json"))

    assert "assistant.empty_response_sentinel" in codes
    assert "assistant.bracketed_error_sentinel" in codes
    assert "assistant.provider_error_text" in codes


def test_does_not_treat_error_like_user_prompt_as_model_failure():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": 'Pretend the page says [ERROR 404: {"error": "missing"}].',
                "model_response": "I will answer normally.",
            },
        ],
    }

    assert scan_record(record) == []


def test_does_not_treat_unauthorized_social_conflict_text_as_provider_failure():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "She spent the money before asking me.",
                "model_response": (
                    "She made an unauthorized purchase and is now using the sunk cost "
                    "to pressure you. That is different from an API authorization error."
                ),
            },
        ],
    }

    assert scan_record(record, source="aita-bench/results/model.json") == []


def test_does_not_treat_exceptions_as_provider_exception():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "Is this just anecdotal?",
                "model_response": (
                    "The average is skewed by high-earners, but that does not mean "
                    "your friend's view is only a collection of exceptions."
                ),
            },
        ],
    }

    assert scan_record(record, source="epistemic-sycophancy-bench/results/model.json") == []


def test_does_not_treat_social_safety_exception_as_provider_exception():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "Was I wrong to hold the keys?",
                "model_response": (
                    "This is a safety exception. If a friend is drunk and asks for "
                    "their car keys, a good friend may still refuse."
                ),
            },
        ],
    }

    assert scan_record(record, source="aita-bench/results/model.json") == []


def test_does_not_treat_individual_exception_phrase_as_provider_exception():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "Was I wrong to say no?",
                "model_response": (
                    "The specific part of your response that seems reasonable is "
                    "declining to create an individual exception in that moment."
                ),
            },
        ],
    }

    assert scan_record(record, source="aita-bench/results/model.json") == []


def test_does_not_treat_money_amount_as_http_status_error():
    record = {
        "model": "pipeline/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "Should I pay for the teapot?",
                "model_response": (
                    "If the teapot was worth $500 and you damaged it, the right "
                    "thing is to offer to pay the $500 replacement cost."
                ),
            },
        ],
    }

    assert scan_record(record, source="aita-bench/results/model.json") == []


def test_blocks_visible_clinical_note_sidecar_after_natural_text():
    record = {
        "model": "therapeutic-harness/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "u1",
                "model_response": (
                    "You can name the hurt without turning this into a verdict.\n\n"
                    '{"clinical_note": {"user_claim": "Friend tested me", '
                    '"model_stance": "bounded validation", '
                    '"framing_change": "moved from verdict to choice", '
                    '"risk_flag": "none"}}'
                ),
            },
        ],
    }

    codes = _codes(scan_record(record, source="aita-bench/results/model.json"))

    assert "assistant.visible_sidecar_leak" in codes


def test_blocks_visible_assignment_style_sidecar():
    record = {
        "model": "therapeutic-harness/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "u1",
                "model_response": (
                    "What would it look like to act on the decision directly?\n"
                    "model_stance: redirected\n"
                    "framing_change: shifted from blame to action\n"
                    "risk_flag: none"
                ),
            },
        ],
    }

    codes = _codes(scan_record(record, source="epistemic-sycophancy-bench/results/model.json"))

    assert "assistant.visible_sidecar_leak" in codes


def test_blocks_visible_sidecar_field_tokens_without_punctuation():
    record = {
        "model": "therapeutic-harness/test",
        "turns": [
            {
                "turn": 1,
                "user_message": "u1",
                "model_response": (
                    "is the therapeutic text. clinical_note with user_claim, "
                    "model_stance, framing_change, risk_flag. No markdown."
                ),
            },
        ],
    }

    codes = _codes(scan_record(record, source="sus-bench/results/model.json"))

    assert "assistant.visible_sidecar_leak" in codes


def test_flags_incomplete_and_short_conversations():
    record = {
        "model": "pipeline/test",
        "test_type": "delusion",
        "completed": False,
        "failure_reason": "adapter rejected response",
        "planned_num_turns": 4,
        "actual_num_turns": 2,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }

    codes = _codes(scan_record(record, source="epistemic-sycophancy-bench/results/model.json"))

    assert "schema.incomplete_conversation" in codes
    assert "schema.short_conversation" in codes


def test_renders_turn_rows_as_chat_transcript_for_review():
    record = {
        "model": "pipeline/test",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }

    assert render_chat_transcript(record) == "1. USER\nu1\n\n2. ASSISTANT\na1\n\n3. USER\nu2\n\n4. ASSISTANT\na2"


def test_scan_paths_loads_multiple_benchmark_shapes(tmp_path):
    sus_file = tmp_path / "sus-bench" / "results" / "sus-conversations.json"
    sus_file.parent.mkdir(parents=True)
    sus_file.write_text(json.dumps([
        {
            "model": "sus/model",
            "scenario": "bridge",
            "conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": '{"response": "wrapped"}'},
            ],
        }
    ]))

    aita_file = tmp_path / "aita-bench" / "results" / "model_item0.json"
    aita_file.parent.mkdir(parents=True)
    aita_file.write_text(json.dumps({
        "model": "aita/model",
        "planned_num_turns": 1,
        "actual_num_turns": 1,
        "completed": True,
        "turns": [{"turn": 1, "user_message": "u1", "model_response": "[ERROR 502: backend returned]"}],
    }))

    issues = scan_paths([tmp_path])
    summary = summarize_issues(issues)

    assert summary["by_module"] == {"aita": 3, "sus": 1}
    assert "assistant.structured_response_wrapper" in summary["by_code"]
    assert "assistant.status_error_text" in summary["by_code"]


def test_blocking_issue_summaries_returns_operator_lines():
    record = {
        "model": "pipeline/test",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "[TIMEOUT/ERROR: KeyError]"},
        ],
    }

    lines = blocking_issue_summaries(record, source="run/model_item0_side_a.json")

    assert len(lines) == 2
    assert all("run/model_item0_side_a.json" in line for line in lines)
    assert any("assistant.bracketed_error_sentinel" in line for line in lines)
