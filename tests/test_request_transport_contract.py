"""Regressions for explicit caps being lost between runners, receipts and wire."""

import copy
import importlib
import json
import socket
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from suite_tools.provider_client import OpenAIResponsesClient
from suite_tools.request_receipts import RequestConformanceError, evaluate_request_conformance
from suite_tools.run_monitor import MonitoredOpenAIClient, RunMonitor


@pytest.fixture
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The request regression attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")


@pytest.mark.parametrize("module", ["sus", "aita", "epis"])
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("model", ["gpt-5.6-terra", "example-reasoning-model"])
@pytest.mark.parametrize("cap_key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_frozen_cap_reaches_wire_and_receipt(
    tmp_path, monkeypatch, offline, module, native, model, cap_key,
):
    from sus_bench import api

    runner = api if module == "sus" else importlib.import_module(f"{module}_bench.runner")
    base_url = "https://api.openai.com/v1/responses" if native else "https://openrouter.ai/api/v1"
    options = {cap_key: 128000, "reasoning_effort": "high"}
    untouched = copy.deepcopy(options)
    wire = []

    def native_post(url, **kwargs):
        wire.append(kwargs["json"])
        return httpx.Response(200, json={
            "status": "completed", "model": model,
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Independent answer."}]}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }, request=httpx.Request("POST", url))

    def transport(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Independent answer."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        })

    monkeypatch.setattr("suite_tools.provider_client.httpx.post", native_post)
    monkeypatch.setattr(runner, "paid_call_lease", lambda **kwargs: nullcontext())
    api.reset_cost_tracker()
    monkeypatch.setattr(api.get_cost_tracker(), "check_credit_if_due", lambda: None)
    client = OpenAIResponsesClient(api_key="offline") if native else OpenAI(
        api_key="offline", base_url=base_url, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    monitor = RunMonitor(tmp_path, module=module, stage="generation")
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "expected_models": [{"key": "target", "model_id": model, "condition_id": "target", "endpoint": base_url, "request_options": options}],
        "modules": [{"module": module}],
    }))
    context = {"condition_id": "target", "model_key": "target", "unit_id": "unit-1"}
    if module == "sus":
        monkeypatch.setattr(api, "make_provider_client", lambda *args, **kwargs: client)
        api.call_provider(model, [{"role": "user", "content": "Hello"}], "offline",
                          base_url=base_url, request_options=options, monitor=monitor,
                          role="model_under_test", request_context=context)
    else:
        runner.api_call(client, model, [{"role": "user", "content": "Hello"}],
                        request_options=options, monitor=monitor,
                        role="model_under_test", request_context=context)

    assert len(wire) == 1
    caps = {key: wire[0][key] for key in ("max_tokens", "max_completion_tokens", "max_output_tokens") if key in wire[0]}
    assert len(caps) == 1
    assert next(iter(caps.values())) == 128000
    effort = wire[0].get("reasoning", {}).get("effort") if native else wire[0].get("reasoning_effort")
    assert effort == "high"
    assert options == untouched
    receipts = [json.loads(line) for line in monitor.events_path.read_text().splitlines()]
    assert [e["effective_max_output_tokens"] for e in receipts if e["event"] == "effective_request"] == [128000]
    assert [e["effective_reasoning_effort"] for e in receipts if e["event"] == "effective_request"] == ["high"]
    assert evaluate_request_conformance(tmp_path)["conformant"]


@pytest.mark.parametrize("field,expected,actual,receipt_field", [
    ("max_tokens", 128000, 1000, "max_output_tokens"),
    ("reasoning_effort", "high", "low", "reasoning_effort"),
])
def test_monitored_call_rejects_frozen_control_mismatch_before_transport(tmp_path, offline, field, expected, actual, receipt_field):
    wire = []
    client = OpenAI(api_key="offline", base_url="https://openrouter.ai/api/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(lambda req: wire.append(req))))
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "expected_models": [{"key": "target", "model_id": "target", "request_options": {field: expected}}],
        "modules": [{"module": "aita"}],
    }))
    wrapped = MonitoredOpenAIClient(client, monitor, role="model_under_test")
    with pytest.raises(RequestConformanceError, match=receipt_field):
        wrapped.chat.completions.create(model="target", messages=[], **{field: actual})
    assert wire == []


@pytest.mark.parametrize("judge_index", [0, 1, 2])
@pytest.mark.parametrize("mismatched_controls", [False, True])
def test_prepared_native_judge_uses_its_own_condition_for_receipts(
    tmp_path, monkeypatch, offline, judge_index, mismatched_controls,
):
    from suite_tools.prepare_run import prepare_sus_run
    from sus_bench import api, scorer

    contract_path = prepare_sus_run(
        run_id="native-panel", output_root=tmp_path / "run",
        suite_config_path=Path(__file__).resolve().parents[1] / "suite_models.yaml",
        model_selector="group:calibration_smoke", judge_set="direct_frontier_high",
        scenarios_selector="bridge", runs=1,
    )
    contract = json.loads(contract_path.read_text())
    judge = copy.deepcopy([entry["config"] for entry in contract["expected_judges"] if entry.get("config")][judge_index])
    monkeypatch.setenv(judge["api_key_env"], "offline")
    if mismatched_controls:
        judge["request_options"] = {}
    target = contract["expected_models"][0]
    context = {"condition_id": target["condition_id"], "model_key": target["key"],
               "unit_id": "test-unit", "phase": "post_analysis"}
    untouched_context = dict(context)
    wire = []

    def transport(**kwargs):
        wire.append(kwargs)
        result = {"irq": 10, "pr": 10, "er": 10, "ca": 10,
                  "target_utility": 0, "cap_timing_severity": 0, "self_coaching": 0,
                  "context_retention_failure": 0, "safety_response_failure": 0}
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(result)), finish_reason="stop")], usage=None)

    monkeypatch.setattr(api, "make_provider_client", lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=transport))))
    monkeypatch.setattr(api, "paid_call_lease", lambda **kwargs: nullcontext())
    api.reset_cost_tracker()
    monkeypatch.setattr(api.get_cost_tracker(), "check_credit_if_due", lambda: None)
    monitor = RunMonitor(contract_path.parent, module="sus", stage="score")
    result = scorer._single_judge_score_status(judge, "Synthetic rubric", "", context, monitor)
    assert context == untouched_context
    if mismatched_controls:
        assert not wire
        assert result["result"] is None
        assert "reasoning_effort" in str(result["failure"])
    else:
        assert len(wire) == 1, result
        assert result["result"] is not None, result
        events = [json.loads(line) for line in monitor.events_path.read_text().splitlines()]
        receipt = next(event for event in events if event["event"] == "effective_request")
        assert receipt["condition_id"] == judge["condition_id"]
        assert receipt["target_condition_id"] == target["condition_id"]
        assert receipt["target_model_key"] == target["key"]
