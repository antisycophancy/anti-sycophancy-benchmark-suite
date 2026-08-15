import importlib.util
import sys
from pathlib import Path

import pytest


ADAPTER_DIR = Path(__file__).resolve().parents[1] / "adapter"
EXAMPLE_PATH = ADAPTER_DIR / "examples" / "proprietary_json_backend.py"


def load_example():
    sys.path.insert(0, str(ADAPTER_DIR))
    try:
        for module_name in ["backend", "config", "model_routing", "openai_contract"]:
            sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location("adapter_customization_example", EXAMPLE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ADAPTER_DIR))


def test_example_translates_request_history_and_output_limit():
    example = load_example()

    payload = example.build_upstream_payload(
        {
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up"},
            ],
            "max_tokens": 512,
        }
    )

    assert payload == {
        "prompt": "Follow-up",
        "history": [
            {"speaker": "system", "text": "Be precise."},
            {"speaker": "user", "text": "First question"},
            {"speaker": "assistant", "text": "First answer"},
        ],
        "output_limit": 512,
    }


def test_example_requires_backend_key_and_builds_private_header(monkeypatch):
    example = load_example()
    monkeypatch.delenv("EXAMPLE_BACKEND_API_KEY", raising=False)

    with pytest.raises(example.AdapterBackendError) as error:
        example.build_upstream_headers()
    assert error.value.code == "missing_upstream_api_key"

    monkeypatch.setenv("EXAMPLE_BACKEND_API_KEY", "example-secret")
    assert example.build_upstream_headers() == {
        "Content-Type": "application/json",
        "X-API-Key": "example-secret",
    }


def test_example_parses_answer_and_explicit_refusal():
    example = load_example()

    answer = example.parse_upstream_response(
        {
            "answer": "Backend answer",
            "stop_reason": "end_turn",
            "input_tokens": 8,
            "output_tokens": 3,
            "private_trace": "ignored",
        }
    )
    refusal = example.parse_upstream_response(
        {
            "answer": None,
            "refusal": "I cannot answer that.",
            "stop_reason": "safety",
        }
    )

    assert answer.content == "Backend answer"
    assert answer.native_finish_reason == "end_turn"
    assert answer.usage == {"prompt_tokens": 8, "completion_tokens": 3}
    assert refusal.content == "I cannot answer that."
    assert refusal.refusal == "I cannot answer that."
