import json

import pytest

from suite_tools.request_receipts import (
    RequestConformanceError,
    effective_request_controls,
    evaluate_request_conformance,
    record_effective_request,
    require_request_conformance,
)
from suite_tools.run_monitor import RunMonitor


def _write_contract(run_dir, *, request_options=None):
    model = {
        "key": "gpt-low",
        "model_id": "gpt-5.6-sol",
        "condition_id": "gpt-5-6-sol-openai-native-low",
        "endpoint": "openai_responses",
    }
    if request_options is not None:
        model["request_options"] = request_options
    (run_dir / "RUN_CONTRACT.json").write_text(
        json.dumps(
            {
                "expected_models": [model],
                "expected_judges": [],
                "modules": [{"module": "aita"}],
            }
        )
    )


def _append_receipt(run_dir, **fields):
    event = {
        "event": "effective_request",
        "role": "model_under_test",
        "model": "gpt-5.6-sol",
        "condition_id": "gpt-5-6-sol-openai-native-low",
        **fields,
    }
    with (run_dir / "RUN_EVENTS.jsonl").open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def test_effective_request_controls_normalizes_provider_cap_and_effort():
    controls = effective_request_controls(
        {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "private prompt"}],
            "max_completion_tokens": 128000,
            "extra_body": {"reasoning_effort": "high"},
        },
        base_url="https://api.openai.com/v1",
    )

    assert controls == {
        "max_output_tokens": 128000,
        "reasoning_effort": "high",
    }


def test_effective_request_controls_uses_anthropic_extra_body_over_runner_default():
    controls = effective_request_controls(
        {
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "private prompt"}],
            "max_tokens": 1000,
            "extra_body": {
                "max_tokens": 128000,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "xhigh"},
            },
        },
        base_url="https://api.anthropic.com/v1/messages",
    )

    assert controls == {
        "max_output_tokens": 128000,
        "reasoning_effort": "xhigh",
        "thinking_type": "adaptive",
    }


def test_effective_request_controls_captures_non_effort_reasoning_controls():
    controls = effective_request_controls(
        {
            "model": "anthropic/claude-opus-4.7",
            "max_tokens": 4096,
            "extra_body": {
                "reasoning": {
                    "effort": "high",
                    "enabled": True,
                    "exclude": True,
                },
                "verbosity": "high",
            },
        },
        base_url="https://openrouter.ai/api/v1",
    )

    assert controls == {
        "max_output_tokens": 4096,
        "reasoning_effort": "high",
        "reasoning_enabled": True,
        "reasoning_exclude": True,
        "verbosity": "high",
    }


def test_record_effective_request_writes_only_sanitized_controls():
    class FakeMonitor:
        def __init__(self):
            self.events = []

        def record(self, event, **fields):
            self.events.append((event, fields))

    monitor = FakeMonitor()
    record_effective_request(
        monitor,
        {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "do not retain this"}],
            "max_completion_tokens": 128000,
            "extra_body": {"reasoning_effort": "low"},
        },
        base_url="https://api.openai.com/v1",
        role="model_under_test",
        condition_id="gpt-low",
        unit_id="aita:gpt-low:item0:side_a:turn1",
    )

    event, fields = monitor.events[0]
    serialized = json.dumps(fields)
    assert event == "effective_request"
    assert fields["effective_max_output_tokens"] == 128000
    assert fields["effective_reasoning_effort"] == "low"
    assert fields["condition_id"] == "gpt-low"
    assert "messages" not in serialized
    assert "do not retain this" not in serialized
    assert len(fields["controls_hash"]) == 64


def test_conformance_fails_when_any_effective_call_differs_from_contract(tmp_path):
    _write_contract(
        tmp_path,
        request_options={"max_tokens": 128000, "reasoning_effort": "low"},
    )
    _append_receipt(
        tmp_path,
        effective_max_output_tokens=128000,
        effective_reasoning_effort="low",
    )
    _append_receipt(
        tmp_path,
        effective_max_output_tokens=1000,
        effective_reasoning_effort="low",
    )

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["conformant"] is False
    assert result["receipt_count"] == 2
    assert result["issues"] == [
        {
            "kind": "request_mismatch",
            "role": "model_under_test",
            "condition_id": "gpt-5-6-sol-openai-native-low",
            "model": "gpt-5.6-sol",
            "field": "max_output_tokens",
            "expected": 128000,
            "actual": 1000,
        }
    ]
    with pytest.raises(RequestConformanceError, match="max_output_tokens"):
        require_request_conformance(tmp_path, roles={"model_under_test"})


def test_conformance_detects_provider_route_mismatch(tmp_path):
    _write_contract(
        tmp_path,
        request_options={"max_tokens": 128000, "reasoning_effort": "low"},
    )
    _append_receipt(
        tmp_path,
        provider="openrouter",
        effective_max_output_tokens=128000,
        effective_reasoning_effort="low",
    )

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["issues"] == [
        {
            "kind": "request_mismatch",
            "role": "model_under_test",
            "condition_id": "gpt-5-6-sol-openai-native-low",
            "model": "gpt-5.6-sol",
            "field": "provider",
            "expected": "openai",
            "actual": "openrouter",
        }
    ]


def test_conformance_fails_closed_when_explicit_condition_has_no_receipt(tmp_path):
    _write_contract(
        tmp_path,
        request_options={"max_tokens": 128000, "reasoning_effort": "low"},
    )

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["conformant"] is False
    assert result["issues"][0]["kind"] == "missing_request_receipt"


def test_conformance_requires_receipt_for_each_completed_model_call(tmp_path):
    _write_contract(
        tmp_path,
        request_options={"max_tokens": 128000, "reasoning_effort": "low"},
    )
    with (tmp_path / "RUN_EVENTS.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "sequence": 1,
                    "event": "stage_started",
                    "stage": "generation",
                    "request_receipt_schema_version": (
                        "benchmark-effective-request-v1"
                    ),
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "sequence": 2,
                    "event": "paid_call_completed",
                    "role": "model",
                    "model": "gpt-low",
                    "model_id": "gpt-5.6-sol",
                    "item_idx": 0,
                    "side": "side_a",
                    "turn": 1,
                }
            )
            + "\n"
        )
    _append_receipt(
        tmp_path,
        model_key="gpt-low",
        item_idx=1,
        side="side_a",
        turn=1,
        effective_max_output_tokens=128000,
        effective_reasoning_effort="low",
    )

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["conformant"] is False
    assert result["issues"][-1] == {
        "kind": "missing_call_receipts",
        "role": "model_under_test",
        "condition_id": "gpt-5-6-sol-openai-native-low",
        "model": "gpt-5.6-sol",
        "missing_count": 1,
        "sample_units": ["gpt-low:item0:side_a:turn1"],
    }


def test_pre_receipt_anthropic_calls_remain_eligible_for_audited_resume(tmp_path):
    (tmp_path / "RUN_CONTRACT.json").write_text(
        json.dumps(
            {
                "expected_models": [
                    {
                        "key": "opus-high",
                        "model_id": "claude-opus-5",
                        "endpoint": "anthropic_native",
                        "condition_id": "claude-opus-5-anthropic-native-high",
                        "request_options": {
                            "max_tokens": 128000,
                            "thinking": {"type": "adaptive"},
                            "output_config": {"effort": "high"},
                        },
                    }
                ],
                "expected_judges": [],
            }
        )
    )

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["conformant"] is True
    assert result["legacy_unverified_requirement_count"] == 1
    assert result["issues"] == []


def test_conformance_does_not_invent_requirements_for_provider_defaults(tmp_path):
    _write_contract(tmp_path)

    result = evaluate_request_conformance(tmp_path, roles={"model_under_test"})

    assert result["conformant"] is True
    assert result["requirement_count"] == 0
    assert result["issues"] == []


def test_score_ready_promotion_is_refused_on_request_mismatch(tmp_path):
    _write_contract(
        tmp_path,
        request_options={"max_tokens": 128000, "reasoning_effort": "low"},
    )
    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    monitor.record(
        "effective_request",
        role="model_under_test",
        model="gpt-5.6-sol",
        condition_id="gpt-5-6-sol-openai-native-low",
        effective_max_output_tokens=1000,
        effective_reasoning_effort="low",
    )

    monitor.mark_completed()

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    assert status["status"] == "completed"
    assert status["validity"] == "not_score_ready"
    assert status["request_conformance"]["conformant"] is False
    assert events[-2]["event"] == "request_conformance_failed"
    assert events[-1]["event"] == "stage_completed"
