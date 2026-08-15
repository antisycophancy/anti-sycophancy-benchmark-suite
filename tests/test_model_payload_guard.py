import pytest

from suite_tools.model_payload_guard import (
    assert_blind_model_payload,
    scan_model_visible_payload,
)


def test_guard_rejects_benchmark_context_in_system_messages():
    messages = [
        {"role": "system", "content": "Benchmark harness instruction: phase 3 scoring."},
        {"role": "user", "content": "hello"},
    ]

    leaks = scan_model_visible_payload(messages)

    assert [leak.term.lower() for leak in leaks] == ["benchmark", "harness", "phase", "scoring"]
    with pytest.raises(AssertionError, match="leaks benchmark context"):
        assert_blind_model_payload(messages)


def test_guard_does_not_block_user_language_by_default():
    messages = [
        {"role": "user", "content": "I need a test score explained."},
        {"role": "assistant", "content": "Sure."},
    ]

    assert scan_model_visible_payload(messages) == []
    assert_blind_model_payload(messages)
