import json

import httpx
import pytest

from suite_tools.evidence import (
    ENVIRONMENT,
    INSTRUMENT_DEFECT,
    MODEL_SIGNAL,
    UNKNOWN,
    action_for,
    classify_evidence,
)
from suite_tools.provider_client import (
    ProviderMalformedResponseError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
)
from suite_tools.run_monitor import RunMonitor


def test_refusal_is_model_signal():
    error = ProviderRefusalError("declined", raw_response={"stop_reason": "refusal"})
    assert classify_evidence(error) == {"evidence_class": MODEL_SIGNAL, "category": "refusal"}


def test_content_policy_400_preserves_provider_code_end_to_end(monkeypatch):
    """Through the REAL client path (round-2 major: testing the helper alone
    proves nothing if the client never calls it). The conversion site is
    OpenAIResponsesClient's completions path (suite_tools/provider_client.py
    ~line 923: `if response.status_code == 400 and
    _is_openai_content_policy_400(response)`), which today builds the error
    from response.text WITHOUT raw_response — this test forces the fix."""
    from suite_tools import provider_client as pc

    body = {"error": {"code": "cyber_policy",
                      "message": "flagged for possible cybersecurity risk"}}

    class StubResponse:
        status_code = 400
        headers = {}
        text = json.dumps(body)

        @staticmethod
        def json():
            return body

    monkeypatch.setattr(pc.httpx, "post", lambda *args, **kwargs: StubResponse())
    client = pc.OpenAIResponsesClient(
        base_url="https://api.example.test/v1/responses",
        api_key="test-key",
    )
    with pytest.raises(ProviderRefusalError) as excinfo:
        client.chat.completions.create(
            model="gpt-5.6-test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    error = excinfo.value
    assert error.raw_response.get("error", {}).get("code") == "cyber_policy"
    assert classify_evidence(error) == {
        "evidence_class": MODEL_SIGNAL, "category": "cyber_policy",
    }


def test_output_budget_exhaustion_is_model_signal():
    error = ProviderOutputBudgetExhaustedError("burned budget", raw_response={})
    assert classify_evidence(error) == {
        "evidence_class": MODEL_SIGNAL, "category": "output_budget_exhausted",
    }


def test_typed_exception_precedence_over_status_code():
    """A ProviderRefusalError whose status_code is 200 must classify as
    model_signal even though heuristics would say nothing useful."""
    error = ProviderRefusalError("declined", raw_response={})
    assert error.status_code == 200
    assert classify_evidence(error)["evidence_class"] == MODEL_SIGNAL


def test_billing_is_environment_terminal_owed():
    class Billing(Exception):
        status_code = 402
    evidence = classify_evidence(Billing("Insufficient credits"))
    assert evidence == {"evidence_class": ENVIRONMENT, "category": "billing"}
    assert action_for(evidence) == "terminal_owed"


def test_connect_timeout_retries_read_timeout_owed():
    connect = classify_evidence(httpx.ConnectTimeout("connect timed out"))
    read = classify_evidence(httpx.ReadTimeout("read timed out"))
    assert connect == {"evidence_class": ENVIRONMENT, "category": "timeout_connect"}
    assert read == {"evidence_class": ENVIRONMENT, "category": "timeout_read"}
    assert action_for(connect) == "retry_bounded"
    assert action_for(read) == "terminal_owed"


def test_rate_limit_and_5xx_retry_bounded():
    class RateLimited(Exception):
        status_code = 429
    class ServerError(Exception):
        status_code = 502
    assert action_for(classify_evidence(RateLimited("429"))) == "retry_bounded"
    assert action_for(classify_evidence(ServerError("bad gateway"))) == "retry_bounded"


def test_malformed_success_response_is_retryable_environment_failure():
    error = ProviderMalformedResponseError(
        "choices_null",
        "Provider returned no usable content",
        raw_response={"choices": None},
    )

    evidence = classify_evidence(error)

    assert evidence == {
        "evidence_class": ENVIRONMENT,
        "category": "malformed_response",
    }
    assert action_for(evidence) == "retry_bounded"


def test_payload_bug_is_instrument_defect_halt():
    class BadRequest(Exception):
        status_code = 400
    error = BadRequest("Unsupported parameter: 'max_tokens' is not supported with this model.")
    evidence = classify_evidence(error)
    assert evidence == {"evidence_class": INSTRUMENT_DEFECT, "category": "payload"}
    assert action_for(evidence) == "halt"


def test_unclassifiable_error_is_unknown_halt():
    evidence = classify_evidence(RuntimeError("something inscrutable happened"))
    assert evidence == {"evidence_class": UNKNOWN, "category": "unclassified"}
    assert action_for(evidence) == "halt"


def test_record_block_appends_ledger_with_pointer_and_attempt(tmp_path):
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    monitor.record_block(
        unit={"item_idx": 3, "side": "side_a"},
        evidence={"evidence_class": MODEL_SIGNAL, "category": "refusal"},
        model="gpt-5.6-luna",
        evidence_pointer="gpt-5-6-luna_item3_side_a.json",
    )
    lines = (tmp_path / "BLOCKS.jsonl").read_text().splitlines()
    assert len(lines) == 1
    block = json.loads(lines[0])
    assert block["schema_version"] == "benchmark-block-v2"
    assert block["evidence_class"] == MODEL_SIGNAL
    assert block["category"] == "refusal"
    assert block["unit"] == {"item_idx": 3, "side": "side_a"}
    assert block["evidence_pointer"] == "gpt-5-6-luna_item3_side_a.json"
    assert block["attempt_number"] == monitor.attempt_number
    assert monitor.status["counters"]["events.block_recorded"] == 1


def test_payload_stage_beats_status_heuristics_for_openrouter_403():
    class GuardrailBlock(Exception):
        status_code = 403
        raw_response = {"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]},
                                  "message": "blocked"}}
    result = classify_evidence(GuardrailBlock("HTTP 403"))
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "guardrail_permission_denied"
    assert result["signal_source"] == "provider-signals-v2"


def test_google_dunning_403_stays_environment_billing():
    class Dunning(Exception):
        status_code = 403
        raw_response = {"error": {"code": 403, "status": "PERMISSION_DENIED",
                                  "message": "Lightning dunning decision is deny for project: projects/955415727012"}}
    result = classify_evidence(Dunning("HTTP 403"))
    assert result["evidence_class"] == ENVIRONMENT
    assert result["category"] == "billing"


def test_bare_google_permission_denied_is_unknown_not_auth():
    # Now resolved by the payload stage's explicit UNKNOWN — legacy 403→auth never runs.
    class Ambiguous(Exception):
        status_code = 403
        raw_response = {"error": {"code": 403, "status": "PERMISSION_DENIED",
                                  "message": "caller does not have permission"}}
    result = classify_evidence(Ambiguous("HTTP 403"))
    assert result["evidence_class"] == UNKNOWN
    assert result["category"] == "ambiguous_permission_denied"


def test_typed_refusal_with_content_filter_finish_reason_is_content_filter():
    e = ProviderRefusalError("blocked", raw_response={"finish_reason": "content_filter"})
    result = classify_evidence(e)
    assert result["evidence_class"] == MODEL_SIGNAL
    assert result["category"] == "content_filter"


def test_action_for_stochastic_model_signal_is_retry_bounded():
    assert action_for({"evidence_class": MODEL_SIGNAL, "category": "SAFETY", "stochastic": True}) == "retry_bounded"
    assert action_for({"evidence_class": MODEL_SIGNAL, "category": "refusal"}) == "record_outcome"
