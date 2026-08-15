import json

import httpx

import suite_tools.throughput_probe as throughput_probe
import pytest
from suite_tools.fake_provider import FakeOpenAIProvider
from suite_tools.throughput_probe import (
    DEFAULT_MAX_TOKENS,
    build_parser,
    _failure_kind,
    _rate_limit_kind,
    _steady_state_utilization,
    parse_models,
    parse_steps,
    run_one_call,
    run_probe,
    summarize_step,
    main,
)


def test_parse_steps_dedupes_positive_values():
    assert parse_steps("1,2,2,4") == [1, 2, 4]


def test_parse_models_requires_at_least_one_model():
    assert parse_models("google/gemini-3-flash-preview, anthropic/claude-3-haiku") == [
        "google/gemini-3-flash-preview",
        "anthropic/claude-3-haiku",
    ]


def test_rate_limit_kind_classifies_request_limit_headers():
    assert _rate_limit_kind(
        429,
        "Rate limit exceeded: @ratelimit/too-many-requests.",
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "123"},
    ) == "request_rate_limit"


def test_insufficient_quota_429_is_billing_not_rate_limit():
    text = '{"error":{"code":"insufficient_quota","message":"You exceeded your quota"}}'
    assert _rate_limit_kind(429, text, {}) is None
    assert _failure_kind(429, text, {}) == "billing"


def test_failure_kind_keeps_token_parameter_error_separate_from_rate_limit():
    assert _failure_kind(
        400,
        "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16.",
        {},
    ) == "token_parameter_minimum"


def test_summarize_step_reports_safe_metrics():
    summary = summarize_step(
        2,
        [
            {"ok": True, "latency_seconds": 0.3, "cost_usd": 0.001},
            {"ok": False, "rate_limited": True, "latency_seconds": 0.5, "cost_usd": 0},
            {
                "ok": False,
                "rate_limited": False,
                "failure_kind": "token_parameter_minimum",
                "latency_seconds": 0.4,
                "cost_usd": 0,
            },
        ],
    )

    assert summary["concurrency"] == 2
    assert summary["successes"] == 1
    assert summary["failures"] == 2
    assert summary["rate_limits"] == 1
    assert summary["total_cost_usd"] == 0.001
    assert summary["failure_examples"][0]["failure_kind"] == "token_parameter_minimum"


def test_steady_state_utilization_excludes_warmup_and_final_drain():
    intervals = [
        (0.0, 10.0),
        (0.1, 9.9),
        (0.2, 9.8),
        (0.3, 9.7),
    ]

    assert _steady_state_utilization(intervals, capacity=4) == 1.0


def test_parser_default_max_tokens_satisfies_openai_minimum():
    assert build_parser().parse_args([]).max_tokens == DEFAULT_MAX_TOKENS


def test_run_one_call_rejects_malformed_http_200_response(tmp_path, monkeypatch):
    class MalformedResponse:
        status_code = 200
        text = "not-json"
        headers = {}

        def json(self):
            raise json.JSONDecodeError("bad", self.text, 0)

    monkeypatch.setattr(throughput_probe.httpx, "post", lambda *args, **kwargs: MalformedResponse())

    result = run_one_call(
        model="google/gemini-flash-lite",
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="fake",
        prompt="OK",
        max_tokens=16,
        timeout_seconds=1,
        lease_dir=tmp_path / "leases",
        step_concurrency=1,
        request_index=0,
    )

    assert result["ok"] is False
    assert result["failure_kind"] == "malformed_response"


def test_run_one_call_classifies_timeout(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("provider timed out")

    monkeypatch.setattr(throughput_probe.httpx, "post", timeout)

    result = run_one_call(
        model="google/gemini-flash-lite",
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="fake",
        prompt="OK",
        max_tokens=16,
        timeout_seconds=0.01,
        lease_dir=tmp_path / "leases",
        step_concurrency=1,
        request_index=0,
    )

    assert result["ok"] is False
    assert result["failure_kind"] == "timeout"


def test_run_one_call_reports_queue_and_lease_timing(tmp_path):
    with FakeOpenAIProvider(latency_seconds=0.02) as provider:
        result = run_one_call(
            model="fake/flash-lite",
            url=provider.chat_url,
            api_key="fake",
            prompt="OK",
            max_tokens=16,
            timeout_seconds=1,
            lease_dir=tmp_path / "leases",
            step_concurrency=1,
            request_index=0,
        )

    assert result["ok"] is True
    assert result["queue_wait_seconds"] >= 0
    assert result["lease_hold_seconds"] >= 0.02


def test_run_probe_applies_each_concurrency_step_to_isolated_policy(tmp_path, monkeypatch):
    policies = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr(throughput_probe, "load_repo_env_files", lambda: None)
    monkeypatch.setattr(throughput_probe, "fetch_key_info", lambda timeout: {"limit_remaining": 5})
    monkeypatch.setattr(
        throughput_probe,
        "set_paid_call_policy",
        lambda limit, *, lease_dir, updated_by: policies.append((limit, lease_dir, updated_by)),
    )
    monkeypatch.setattr(
        throughput_probe,
        "paid_call_capacity_report",
        lambda lease_dir: {
            "effective_limit": policies[-1][0],
            "effective_limit_source": "policy:throughput_probe",
            "policy_limit": policies[-1][0],
            "policy_updated_by": "throughput_probe",
            "environment_limit": None,
            "environment_variable": None,
        },
    )
    monkeypatch.setattr(
        throughput_probe,
        "run_one_call",
        lambda **kwargs: {"ok": True, "latency_seconds": 0.01, "cost_usd": 0},
    )
    args = build_parser().parse_args([
        "--confirm-paid-calls",
        "--models", "google/gemini-flash-lite",
        "--steps", "1,4,8",
        "--requests-per-step", "1",
        "--pause-seconds", "0",
        "--output-dir", str(tmp_path / "probe"),
    ])

    report = run_probe(args)

    assert [item[0] for item in policies] == [1, 4, 8]
    assert all(item[1] == tmp_path / "probe" / "_runtime" / "paid_call_leases" for item in policies)
    assert report["results"][0]["safe_concurrency"] == 8


def test_paid_probe_requires_explicit_confirmation_before_credentials_or_network(
    monkeypatch,
):
    monkeypatch.setattr(
        throughput_probe,
        "load_repo_env_files",
        lambda: (_ for _ in ()).throw(AssertionError("dotenv must not load")),
    )
    monkeypatch.setattr(
        throughput_probe,
        "fetch_key_info",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network must not run")),
    )
    args = build_parser().parse_args([])

    with pytest.raises(SystemExit, match="--confirm-paid-calls"):
        run_probe(args)


def test_paid_probe_never_sends_openrouter_key_to_custom_chat_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-key")
    monkeypatch.setattr(throughput_probe, "load_repo_env_files", lambda: None)
    monkeypatch.setattr(
        throughput_probe,
        "fetch_key_info",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network must not run")),
    )
    args = build_parser().parse_args([
        "--confirm-paid-calls",
        "--chat-url", "https://attacker.example/v1/chat/completions",
    ])

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY.*refusing"):
        run_probe(args)


def test_probe_prints_and_persists_effective_environment_floor(tmp_path, capsys, monkeypatch):
    output_dir = tmp_path / "probe"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "3")

    status = main([
        "--fake",
        "--models", "fake/flash-lite",
        "--steps", "8",
        "--requests-per-step", "12",
        "--fake-latency", "0.01",
        "--pause-seconds", "0",
        "--output-dir", str(output_dir),
    ])

    report = json.loads((output_dir / "THROUGHPUT_PROBE.json").read_text())
    step = report["results"][0]["steps"][0]
    output = capsys.readouterr().out

    assert status == 0
    assert step["concurrency"] == 8
    assert step["effective_paid_call_limit"] == 3
    assert step["effective_paid_call_limit_source"] == (
        "environment:BENCHMARK_PAID_CALL_MAX_ACTIVE"
    )
    assert step["provider_max_active"] <= 3
    assert report["results"][0]["safe_concurrency"] == 3
    assert "requested=8 effective=3" in output
    assert "source=environment:BENCHMARK_PAID_CALL_MAX_ACTIVE" in output


def test_run_probe_can_measure_real_http_concurrency_without_paid_calls(tmp_path):
    args = build_parser().parse_args([
        "--fake",
        "--models", "fake/flash-lite",
        "--steps", "4",
        "--requests-per-step", "12",
        "--fake-latency", "0.02",
        "--pause-seconds", "0",
        "--output-dir", str(tmp_path / "probe"),
    ])

    report = run_probe(args)
    step = report["results"][0]["steps"][0]

    assert report["mode"] == "local_fake"
    assert report["total_cost_usd"] == 0
    assert step["successes"] == 12
    assert 2 <= step["provider_max_active"] <= 4
    assert 0 < step["slot_utilization"] <= 1
    assert 0 < step["provider_slot_utilization"] <= 1
    assert step["queue_wait_seconds"]["p95"] >= 0
    assert step["lease_hold_seconds"]["p95"] >= 0.02
    assert step["calls_per_minute"] > 0
    assert step["tokens_per_minute"] > 0


def test_fake_probe_reports_large_local_capacity_without_overshoot(tmp_path):
    args = build_parser().parse_args([
        "--fake",
        "--models", "fake/flash-lite",
        "--steps", "64",
        "--requests-per-step", "192",
        "--fake-latency", "0.2",
        "--pause-seconds", "0",
        "--output-dir", str(tmp_path / "probe"),
    ])

    step = run_probe(args)["results"][0]["steps"][0]

    assert step["failures"] == 0
    assert step["effective_paid_call_limit"] == 64
    assert 1 < step["provider_max_active"] <= 64


def test_one_hundred_slot_fake_probe_has_no_overshoot(tmp_path):
    args = build_parser().parse_args([
        "--fake",
        "--models", "fake/flash-lite",
        "--steps", "100",
        "--requests-per-step", "500",
        "--fake-latency", "1",
        "--pause-seconds", "0",
        "--output-dir", str(tmp_path / "probe"),
    ])

    step = run_probe(args)["results"][0]["steps"][0]

    assert step["successes"] == 500
    assert step["failures"] == 0
    assert step["effective_paid_call_limit"] == 100
    assert 1 < step["provider_max_active"] <= 100
    assert 0 < step["steady_state_slot_utilization"] <= 1
