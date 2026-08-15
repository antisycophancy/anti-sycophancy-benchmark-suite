import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "adapter" / "openai_contract.py"
SPEC = importlib.util.spec_from_file_location("adapter_openai_contract", MODULE_PATH)
openai_contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = openai_contract
assert SPEC.loader is not None
SPEC.loader.exec_module(openai_contract)

ROUTING_PATH = Path(__file__).resolve().parents[1] / "adapter" / "model_routing.py"
ROUTING_SPEC = importlib.util.spec_from_file_location("adapter_model_routing", ROUTING_PATH)
model_routing = importlib.util.module_from_spec(ROUTING_SPEC)
assert ROUTING_SPEC.loader is not None
ROUTING_SPEC.loader.exec_module(model_routing)


def test_chat_completion_response_uses_openai_compatible_shape():
    response = openai_contract.chat_completion_response(
        model="local/example-model",
        content="Hello from the reference adapter.",
    )

    assert response["object"] == "chat.completion"
    assert response["model"] == "local/example-model"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello from the reference adapter.",
    }
    assert response["usage"]["total_tokens"] == 0


def test_chat_completion_response_filters_custom_backend_usage():
    response = openai_contract.chat_completion_response(
        model="local/example-model",
        content="Public answer",
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "private_trace": "must not cross the boundary",
        },
    )

    assert response["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}


def test_last_user_message_reads_standard_messages():
    assert openai_contract.last_user_message(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
    ) == "Second"


def test_reference_adapter_model_routing_is_provider_neutral():
    assert model_routing.resolve_upstream_model("benchmark/requested", "") == "benchmark/requested"
    assert model_routing.resolve_upstream_model("benchmark/requested", "upstream/fixed") == "upstream/fixed"
    assert model_routing.list_adapter_model_ids("local/example-model") == ["local/example-model"]


@pytest.mark.parametrize(
    ("request_body", "expected_code"),
    [
        (None, "invalid_request"),
        ({"messages": []}, "invalid_messages"),
        ({"messages": [{"role": "invalid", "content": "Hello"}]}, "invalid_message_role"),
        ({"messages": [{"role": "user", "content": ["Hello"]}]}, "invalid_message_content"),
        ({"messages": [{"role": "user", "content": "   "}]}, "missing_user_message"),
    ],
)
def test_request_validation_rejects_malformed_benchmark_inputs(request_body, expected_code):
    with pytest.raises(openai_contract.OpenAIContractError) as error:
        openai_contract.validate_chat_completion_request(request_body)

    assert error.value.code == expected_code


def test_response_parser_keeps_only_public_usage_and_signals():
    parsed = openai_contract.parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Public answer",
                        "refusal": "provider refusal label",
                    },
                    "finish_reason": "stop",
                }
            ],
            "native_finish_reason": "end_turn",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cost": 0.012,
                "private_trace": "must not cross the boundary",
            },
            "private_debug": {"route": "secret"},
        }
    )

    assert parsed.content == "Public answer"
    assert parsed.finish_reason == "stop"
    assert parsed.native_finish_reason == "end_turn"
    assert parsed.refusal == "provider refusal label"
    assert parsed.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "cost": 0.012,
    }


def test_empty_success_response_preserves_finish_context_for_triage():
    with pytest.raises(openai_contract.OpenAIContractError) as error:
        openai_contract.parse_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "content_filter",
                    }
                ],
                "native_finish_reason": "safety",
            }
        )

    assert error.value.code == "empty_upstream_content"
    assert error.value.context == {
        "finish_reason": "content_filter",
        "native_finish_reason": "safety",
        "refusal": None,
    }


def test_refusal_only_response_is_preserved_as_model_behavior():
    parsed = openai_contract.parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I cannot help with that request.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert parsed.content == "I cannot help with that request."
    assert parsed.refusal == "I cannot help with that request."
