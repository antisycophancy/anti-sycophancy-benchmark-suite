import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from suite_tools import preflight_conditions as pf
from suite_tools.prepare_run import prepare_sus_run


def _poster(responses):
    """Return a fake httpx.post that yields queued responses by call order."""
    calls = []

    def post(url, *, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        spec = responses.pop(0)
        return _FakeResp(spec)

    post.calls = calls
    return post


class _FakeResp:
    def __init__(self, spec):
        self.status_code = spec["status_code"]
        self._json = spec.get("json", {})
        self.text = spec.get("text", json.dumps(self._json))
        self.headers = {}

    def json(self):
        return self._json


def test_collect_targets_dedups_by_model_effort_endpoint():
    from suite_tools.model_config import load_suite_config

    config = load_suite_config()
    targets = pf.collect_targets_from_groups(
        config, ["gpt_5_6_sol_native_effort", "gpt_5_6_sus_none"]
    )
    # sol effort grid = low/medium/high/xhigh/max (5) + none (1) = 6 unique sol cells
    sol = [t for t in targets if t.model_id == "gpt-5.6-sol"]
    assert {t.effort for t in sol} == {"low", "medium", "high", "xhigh", "max", "none"}
    assert all(t.provider_api == "openai_responses" for t in sol)
    assert all(t.base_url == "https://api.openai.com/v1/responses" for t in sol)


def test_collect_targets_from_current_prepared_run_layout(tmp_path: Path):
    config_dir = tmp_path / "_configs" / "frontier"
    config_dir.mkdir(parents=True)
    (config_dir / "sus-models.yaml").write_text(
        """models:
  - id: therapeutic-harness/example-high
    condition_id: example-high
    provider_api: openai_compatible
    base_url: http://127.0.0.1:9999/v1/chat/completions
    api_key_env: PRIVATE_ADAPTER_API_KEY
    condition_metadata:
      effort: high
"""
    )

    targets = pf.collect_targets_from_run_dir(tmp_path)

    assert len(targets) == 1
    assert targets[0].model_id == "therapeutic-harness/example-high"
    assert targets[0].effort == "high"
    assert targets[0].condition_ids == ("example-high",)


def test_collect_targets_includes_every_rendered_paid_role(tmp_path: Path):
    (tmp_path / "all-models.yaml").write_text(
        """models:
  - id: shared/model
    condition_id: evaluated-cell
    provider_api: openai_compatible
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
judge:
  configs:
    - model_id: shared/model
      condition_id: judge-cell
      provider_api: openai_compatible
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
seeker:
  model_id: shared/model
flip_generator:
  model_id: shared/model
analyzer: shared/model
"""
    )

    targets = pf.collect_targets_from_run_dir(tmp_path)

    assert {target.role for target in targets} == {
        "model_under_test",
        "judge",
        "analyzer",
        "seeker",
        "flip_generator",
    }
    assert len(targets) == 5
    assert all(target.model_id == "shared/model" for target in targets)


def test_target_dedup_is_role_aware():
    common = {
        "model_id": "same/model",
        "provider_api": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    }

    targets = pf._targets_from_entries([
        {**common, "condition_id": "judge-a", "_preflight_role": "judge"},
        {**common, "condition_id": "judge-b", "_preflight_role": "judge"},
        {**common, "condition_id": "seeker", "_preflight_role": "seeker"},
    ])

    assert len(targets) == 2
    by_role = {target.role: target for target in targets}
    assert by_role["judge"].condition_ids == ("judge-a", "judge-b")
    assert by_role["seeker"].condition_ids == ("seeker",)


def test_collect_targets_from_module_run_dir_follows_rendered_config_artifact(tmp_path: Path):
    run_dir = tmp_path / "sus"
    config_path = tmp_path / "_configs" / "calibration" / "sus-models.yaml"
    run_dir.mkdir()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """models:
  - id: google/gemini-3-flash-preview
    key: gemini-flash
    provider_api: openai_compatible
    base_url: https://openrouter.ai/api/v1/chat/completions
    api_key_env: OPENROUTER_API_KEY
"""
    )
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "modules": [{
            "module": "sus",
            "expected_artifacts": [{
                "kind": "rendered_models",
                "path": str(config_path),
                "required_for": "diagnostic",
            }],
        }],
    }))

    targets = pf.collect_targets_from_run_dir(run_dir)

    assert len(targets) == 1
    assert targets[0].model_id == "google/gemini-3-flash-preview"


def test_collect_targets_rejects_contract_config_outside_run_group(tmp_path: Path):
    run_dir = tmp_path / "run" / "sus"
    outside = tmp_path / "attacker-models.yaml"
    run_dir.mkdir(parents=True)
    outside.write_text("models: []\n")
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "modules": [{
            "module": "sus",
            "expected_artifacts": [{
                "kind": "rendered_models",
                "path": str(outside),
                "required_for": "diagnostic",
            }],
        }],
    }))

    with pytest.raises(ValueError, match="outside the prepared run group"):
        pf.collect_targets_from_run_dir(run_dir)


def test_collect_targets_rejects_missing_contract_config_even_with_decoy(tmp_path: Path):
    run_dir = tmp_path / "run" / "sus"
    run_dir.mkdir(parents=True)
    (run_dir / "decoy-models.yaml").write_text(
        "models:\n  - id: decoy/model\n    base_url: http://127.0.0.1:9999/v1\n"
    )
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "modules": [{
            "module": "sus",
            "expected_artifacts": [{
                "kind": "rendered_models",
                "path": "../_configs/missing-models.yaml",
            }],
        }],
    }))

    with pytest.raises(ValueError, match="missing or ambiguous"):
        pf.collect_targets_from_run_dir(run_dir)


def test_collect_targets_rejects_malformed_contract_instead_of_scanning_decoy(
    tmp_path: Path,
):
    run_dir = tmp_path / "run" / "sus"
    run_dir.mkdir(parents=True)
    (run_dir / "decoy-models.yaml").write_text("models: []\n")
    (run_dir / "RUN_CONTRACT.json").write_text("{not-json")

    with pytest.raises(ValueError, match="malformed RUN_CONTRACT"):
        pf.collect_targets_from_run_dir(run_dir)


def test_preflight_main_rejects_mutated_prepared_config_before_env_or_probe(
    tmp_path,
    monkeypatch,
):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="preflight-config-drift",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[1] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    config_path = run_group / "_configs" / "calibration" / "sus-models.yaml"
    config_path.write_text(config_path.read_text() + "\n# changed\n")
    env_reads = []
    monkeypatch.setattr(pf, "load_repo_env_files", lambda: env_reads.append(True))
    monkeypatch.setattr(
        pf,
        "run_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("probe must not run")
        ),
    )

    assert pf.main(["--run-dir", str(contract_path.parent)]) == 2
    assert env_reads == []


def test_build_probe_payload_responses_sets_effort_and_small_cap():
    target = pf.ProbeTarget(
        model_id="gpt-5.6-sol",
        effort="max",
        provider_api="openai_responses",
        base_url="https://api.openai.com/v1/responses",
        api_key_env="OPENAI_API_KEY",
        request_options={
            "max_tokens": 128000,
            "reasoning_effort": "max",
            "verbosity": "high",
        },
    )
    url, payload, headers = pf.build_probe_request(target, api_key="k")
    assert url == "https://api.openai.com/v1/responses"
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "max"}
    assert payload["verbosity"] == "high"
    assert payload["max_output_tokens"] <= 16
    assert "max_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert headers["Authorization"] == "Bearer k"


def test_target_dedup_keeps_distinct_credential_envs():
    targets = pf._targets_from_entries([
        {
            "id": "one",
            "model_id": "same/model",
            "provider_api": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        {
            "id": "two",
            "model_id": "same/model",
            "provider_api": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "SECOND_OPENROUTER_KEY",
        },
    ])

    assert {target.api_key_env for target in targets} == {
        "OPENROUTER_API_KEY",
        "SECOND_OPENROUTER_KEY",
    }


def test_build_probe_payload_openai_compatible_accepts_api_root():
    target = pf.ProbeTarget(
        model_id="therapeutic-harness/example-high",
        effort="high",
        provider_api="openai_compatible",
        base_url="http://127.0.0.1:9999/v1",
        api_key_env="PRIVATE_ADAPTER_API_KEY",
    )

    url, payload, headers = pf.build_probe_request(target, api_key="k")

    assert url == "http://127.0.0.1:9999/v1/chat/completions"
    assert payload["model"] == "therapeutic-harness/example-high"
    assert "reasoning" not in payload
    assert payload["max_tokens"] <= 16
    assert headers["Authorization"] == "Bearer k"


def test_build_probe_payload_openrouter_preserves_full_request_controls():
    target = pf.ProbeTarget(
        model_id="anthropic/claude-opus-4.8",
        effort="high",
        provider_api="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        request_options={
            "max_tokens": 8192,
            "reasoning": {"enabled": True, "exclude": True},
            "verbosity": "high",
        },
    )

    url, payload, _headers = pf.build_probe_request(target, api_key="k")

    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["reasoning"] == {"enabled": True, "exclude": True}
    assert payload["verbosity"] == "high"
    assert payload["max_tokens"] == 16


def test_build_probe_payload_anthropic_preserves_adaptive_thinking_controls():
    target = pf.ProbeTarget(
        model_id="claude-opus-4-8",
        effort="xhigh",
        provider_api="anthropic_messages",
        base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        request_options={
            "max_tokens": 8192,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        },
    )

    url, payload, _headers = pf.build_probe_request(target, api_key="k")

    assert url == "https://api.anthropic.com/v1/messages"
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "xhigh"}
    assert payload["max_tokens"] == 16


def test_build_probe_payload_gemini_preserves_thinking_projection_and_small_cap():
    target = pf.ProbeTarget(
        model_id="gemini-3.1-pro-preview",
        effort="high",
        provider_api="gemini_generate_content",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        request_options={
            "generationConfig": {
                "maxOutputTokens": 8192,
                "thinkingConfig": {
                    "includeThoughts": False,
                    "thinkingLevel": "high",
                },
            }
        },
    )

    url, payload, _headers = pf.build_probe_request(target, api_key="k")

    assert url.endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert payload["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingLevel": "high",
    }
    assert payload["generationConfig"]["maxOutputTokens"] == 16


def test_target_dedup_keeps_distinct_request_control_sets():
    targets = pf._targets_from_entries([
        {
            "id": "one",
            "model_id": "same/model",
            "provider_api": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "request_options": {"verbosity": "high"},
        },
        {
            "id": "two",
            "model_id": "same/model",
            "provider_api": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "request_options": {"verbosity": "xhigh"},
        },
    ])

    assert len(targets) == 2
    assert len({pf._safe_target_identity(target)["request_controls_hash"] for target in targets}) == 2


def test_probe_target_accepts_200_as_pass():
    target = pf.ProbeTarget(
        model_id="gpt-5.6-sol",
        effort="max",
        provider_api="openai_responses",
        base_url="https://api.openai.com/v1/responses",
        api_key_env="OPENAI_API_KEY",
    )
    post = _poster([{"status_code": 200, "json": {"status": "completed"}}])
    result = pf.probe_target(target, api_key="k", poster=post)
    assert result.status == "PASS"
    assert result.http_status == 200


def test_probe_target_rejects_malformed_success_response():
    target = pf.ProbeTarget(
        model_id="openai/gpt-5.5",
        effort=None,
        provider_api="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    post = _poster([{"status_code": 200, "json": {}}])

    result = pf.probe_target(target, api_key="k", poster=post)

    assert result.status == "ERROR"
    assert result.http_status == 200
    assert result.reason_code == "malformed_response"


def test_run_preflight_rejects_missing_endpoint_origin_without_calling_provider():
    target = pf.ProbeTarget(
        "model", None, "openai_compatible", "", "CUSTOM_KEY"
    )
    post = _poster([])

    results, exit_code = pf.run_preflight(
        [target], poster=post, env={"CUSTOM_KEY": "secret"}
    )

    assert exit_code != 0
    assert results[0].reason_code == "missing_endpoint_origin"
    assert not post.calls


def test_probe_target_flags_400_invalid_param_as_fail():
    target = pf.ProbeTarget(
        model_id="gpt-5.6-sol",
        effort="max",
        provider_api="openai_responses",
        base_url="https://api.openai.com/v1/responses",
        api_key_env="OPENAI_API_KEY",
    )
    post = _poster(
        [
            {
                "status_code": 400,
                "json": {"error": {"message": "Unsupported value: 'max'"}},
                "text": "Unsupported value: 'max' for reasoning.effort",
            }
        ]
    )
    result = pf.probe_target(target, api_key="k", poster=post)
    assert result.status == "FAIL"
    assert result.http_status == 400
    assert "max" in result.reason


def test_run_preflight_returns_nonzero_on_any_fail():
    targets = [
        pf.ProbeTarget("gpt-5.6-sol", "high", "openai_responses",
                       "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
        pf.ProbeTarget("gpt-5.6-sol", "max", "openai_responses",
                       "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
    ]
    post = _poster(
        [
            {"status_code": 200, "json": {"status": "completed"}},
            {"status_code": 400, "text": "Unsupported value: 'max'"},
        ]
    )
    results, exit_code = pf.run_preflight(
        targets, poster=post, env={"OPENAI_API_KEY": "k"}
    )
    assert exit_code != 0
    statuses = {(r.target.model_id, r.target.effort): r.status for r in results}
    assert statuses[("gpt-5.6-sol", "high")] == "PASS"
    assert statuses[("gpt-5.6-sol", "max")] == "FAIL"


def test_run_preflight_all_pass_exits_zero():
    targets = [
        pf.ProbeTarget("gpt-5.6-luna", "none", "openai_responses",
                       "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
    ]
    post = _poster([{"status_code": 200, "json": {"status": "completed"}}])
    results, exit_code = pf.run_preflight(
        targets, poster=post, env={"OPENAI_API_KEY": "k"}
    )
    assert exit_code == 0
    assert results[0].status == "PASS"


def test_run_preflight_never_sends_official_key_to_untrusted_host():
    target = pf.ProbeTarget(
        "attacker/model",
        None,
        "openai_compatible",
        "https://attacker.example/v1",
        "OPENROUTER_API_KEY",
    )
    post = _poster([])

    results, exit_code = pf.run_preflight(
        [target], poster=post, env={"OPENROUTER_API_KEY": "secret"}
    )

    assert exit_code != 0
    assert results[0].status == "ERROR"
    assert "OPENROUTER_API_KEY" in results[0].reason
    assert not post.calls


def test_run_preflight_requires_explicit_https_host_allowlist_for_custom_remote_key():
    target = pf.ProbeTarget(
        "custom/model",
        None,
        "openai_compatible",
        "https://models.example/v1",
        "CUSTOM_MODEL_API_KEY",
    )
    denied_post = _poster([])

    denied, denied_code = pf.run_preflight(
        [target], poster=denied_post, env={"CUSTOM_MODEL_API_KEY": "secret"}
    )
    allowed_post = _poster([{
        "status_code": 200,
        "json": {"choices": [{"message": {"content": "ok"}}]},
    }])
    allowed, allowed_code = pf.run_preflight(
        [target],
        poster=allowed_post,
        env={"CUSTOM_MODEL_API_KEY": "secret"},
        allowed_endpoint_hosts={"models.example"},
    )

    assert denied_code != 0
    assert denied[0].status == "ERROR"
    assert not denied_post.calls
    assert allowed_code == 0
    assert allowed[0].status == "PASS"
    assert len(allowed_post.calls) == 1


def test_run_preflight_allows_custom_keys_only_on_literal_loopback_by_default():
    target = pf.ProbeTarget(
        "local/model", None, "openai_compatible", "http://[::1]:9999/v1", "LOCAL_KEY"
    )
    post = _poster([{
        "status_code": 200,
        "json": {"choices": [{"message": {"content": "ok"}}]},
    }])

    results, exit_code = pf.run_preflight(
        [target], poster=post, env={"LOCAL_KEY": "secret"}
    )

    assert exit_code == 0
    assert results[0].status == "PASS"
    assert len(post.calls) == 1


def test_run_preflight_missing_api_key_is_error_not_silent_pass():
    targets = [
        pf.ProbeTarget("gpt-5.6-luna", "max", "openai_responses",
                       "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
    ]
    post = _poster([])  # never called
    results, exit_code = pf.run_preflight(targets, poster=post, env={})
    assert exit_code != 0
    assert results[0].status in {"ERROR", "FAIL"}
    assert not post.calls


def test_main_loads_repo_env_before_the_api_key_check(monkeypatch):
    """main() must load the repo-local .env before probing.

    Regression guard. preflight_conditions did not import load_repo_env_files —
    unlike scheduler, openrouter_preflight and throughput_probe, which all do —
    so run_preflight read a bare os.environ and every DIRECT-provider cell
    (anthropic_native, google_gemini_native, openai_native) reported
    "missing API key; cannot probe" without ever reaching the provider, even
    with the key present in the suite-root .env.

    Found 2026-07-28 preflighting Claude Opus 5: 6/6 cells ERROR'd. The RUNBOOK
    mandates this preflight before any paid spend, so the one guard that catches
    provider enum/param mismatches was silently unavailable for exactly the
    models that route directly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    target = pf.ProbeTarget(
        "claude-opus-5", "high", "anthropic_messages",
        "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY",
    )
    monkeypatch.setattr(pf, "collect_targets", lambda args: [target])

    # The key exists ONLY via the repo .env loader, never in the ambient env.
    def fake_loader():
        os.environ["ANTHROPIC_API_KEY"] = "from-dotenv"

    monkeypatch.setattr(pf, "load_repo_env_files", fake_loader)

    probed = []

    def fake_probe(t, *, api_key, poster=None, allowed_endpoint_hosts=()):
        probed.append(api_key)
        return pf.ProbeResult(t, "PASS", 200, "accepted")

    monkeypatch.setattr(pf, "probe_target", fake_probe)

    exit_code = pf.main([])

    assert probed == ["from-dotenv"], (
        "probe_target was never reached with the .env key — main() failed to "
        "load the repo .env before the API-key check"
    )
    assert exit_code == 0


def test_preflight_receipt_binds_prepared_config_contract_and_role_results(
    tmp_path,
):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="durable-preflight",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[1] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    context = pf.collect_prepared_run_context(contract_path.parent)
    response = {
        "status_code": 200,
        "json": {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.00001},
        },
    }
    post = _poster([dict(response) for _ in context.targets])
    env = {target.api_key_env: "receipt-test-key" for target in context.targets}
    results, exit_code = pf.run_preflight(context.targets, poster=post, env=env)

    receipt_path = pf.write_preflight_receipt(context, results)
    receipt = json.loads(receipt_path.read_text())

    assert exit_code == 0
    assert receipt_path == contract_path.parent / "PREFLIGHT_RECEIPT.json"
    assert receipt["prepared_config"]["verified"] is True
    assert len(receipt["prepared_config"]["sha256"]) == 64
    assert len(receipt["contract_provenance_fingerprint"]) == 64
    assert receipt["target_set_hash"]
    assert receipt["receipt_fingerprint"] == pf.preflight_receipt_fingerprint(receipt)
    assert {row["role"] for row in receipt["results"]} >= {
        "model_under_test", "judge", "analyzer"
    }
    assert all(row["http_status"] == 200 for row in receipt["results"])
    assert all(row["usage"]["source"] == "response.usage" for row in receipt["results"])
    assert all(row["cost"] == {
        "state": "reported",
        "usd": 0.00001,
        "source": "response.usage.cost",
    } for row in receipt["results"])


def test_preflight_receipt_keeps_unknown_usage_unknown_and_leaks_no_body_or_key(
    tmp_path,
):
    run_group = tmp_path / "prepared"
    contract_path = prepare_sus_run(
        run_id="safe-preflight",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[1] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    context = pf.collect_prepared_run_context(contract_path.parent)
    credential = "fixture-key"
    body_sentinel = "PROVIDER_BODY_MUST_NOT_PERSIST"
    post = _poster([
        {
            "status_code": 200,
            "json": {
                "choices": [{"message": {"content": body_sentinel}}],
            },
            "text": body_sentinel,
        }
        for _ in context.targets
    ])
    results, exit_code = pf.run_preflight(
        context.targets,
        poster=post,
        env={target.api_key_env: credential for target in context.targets},
    )

    receipt_path = pf.write_preflight_receipt(context, results)
    raw = receipt_path.read_text()
    receipt = json.loads(raw)

    assert exit_code == 0
    assert credential not in raw
    assert body_sentinel not in raw
    assert all(row["cost"] == {
        "state": "unknown", "usd": None, "source": "unknown"
    } for row in receipt["results"])
    assert all(row["usage"]["state"] == "unknown" for row in receipt["results"])


def test_main_json_output_is_machine_readable_and_prompt_free(monkeypatch, capsys):
    target = pf.ProbeTarget(
        "model", None, "openai_compatible",
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", role="judge",
    )
    monkeypatch.setattr(pf, "collect_targets", lambda args: [target])
    monkeypatch.setattr(pf, "load_repo_env_files", lambda: None)
    monkeypatch.setattr(
        pf,
        "run_preflight",
        lambda *_args, **_kwargs: ([
            pf.ProbeResult(target, "FAIL", 400, "SECRET PROVIDER BODY")
        ], 1),
    )

    exit_code = pf.main(["--json"])
    output = capsys.readouterr().out
    report = json.loads(output)

    assert exit_code == 1
    assert report["results"][0]["role"] == "judge"
    assert report["results"][0]["http_status"] == 400
    assert "SECRET PROVIDER BODY" not in output


def _passing_receipt(tmp_path: Path):
    run_group = tmp_path / "prepared-admission"
    contract_path = prepare_sus_run(
        run_id="preflight-admission",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[1] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )
    context = pf.collect_prepared_run_context(contract_path.parent)
    results = [
        pf.ProbeResult(
            target,
            "PASS",
            200,
            "accepted",
            reason_code="accepted",
        )
        for target in context.targets
    ]
    receipt_path = pf.write_preflight_receipt(context, results)
    return contract_path, receipt_path


def test_validate_preflight_receipt_accepts_current_exact_role_target_passes(tmp_path):
    contract_path, receipt_path = _passing_receipt(tmp_path)

    admission = pf.validate_preflight_receipt_before_spend(contract_path.parent)

    assert admission["verified"] is True
    assert admission["path"] == str(receipt_path)
    assert admission["target_count"] >= 3
    assert 0 <= admission["age_seconds"] <= pf.PREFLIGHT_RECEIPT_TTL_SECONDS


def test_validate_preflight_receipt_rejects_missing_receipt(tmp_path):
    contract_path, receipt_path = _passing_receipt(tmp_path)
    receipt_path.unlink()

    with pytest.raises(pf.PreflightReceiptValidationError, match="missing"):
        pf.validate_preflight_receipt_before_spend(contract_path.parent)


@pytest.mark.parametrize("mutation, expected", [
    ("stale", "stale"),
    ("failed", "PASS"),
    ("duplicate", "exactly one"),
    ("target_hash", "target-set"),
])
def test_validate_preflight_receipt_rejects_rehashed_stale_failed_or_mismatched_evidence(
    tmp_path,
    mutation,
    expected,
):
    contract_path, receipt_path = _passing_receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    if mutation == "stale":
        receipt["generated_at"] = (
            datetime.now(timezone.utc)
            - timedelta(seconds=pf.PREFLIGHT_RECEIPT_TTL_SECONDS + 1)
        ).isoformat()
    elif mutation == "failed":
        receipt["results"][0]["status"] = "FAIL"
        receipt["summary"] = {
            "pass": len(receipt["results"]) - 1,
            "fail": 1,
            "error": 0,
            "total": len(receipt["results"]),
        }
    elif mutation == "duplicate":
        receipt["results"].append(dict(receipt["results"][0]))
        receipt["summary"]["pass"] += 1
        receipt["summary"]["total"] += 1
    else:
        receipt["target_set_hash"] = "0" * 64
    receipt["receipt_fingerprint"] = pf.preflight_receipt_fingerprint(receipt)
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(pf.PreflightReceiptValidationError, match=expected):
        pf.validate_preflight_receipt_before_spend(contract_path.parent)
