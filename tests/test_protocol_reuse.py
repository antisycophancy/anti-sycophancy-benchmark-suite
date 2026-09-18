import importlib
import json

import pytest

from suite_tools.artifact_identity import ArtifactIdentityError, evaluate_run_artifact_identity


MODEL = {"id": "target", "model_id": "target", "key": "target", "label": "Target",
         "condition_id": "target", "condition_hash": "sha256:model", "max_parallel": 1}
SCENARIO = {"id": "test", "name": "Test", "escalation": [{}, {}]}


def _contract(tmp_path, path):
    contract = {
        "artifact_protocol_version": "benchmark-artifact-protocol-v1",
        "provenance": {"benchmark_condition_hash": "current-protocol"},
        "expected_models": [MODEL],
        "modules": [{"module": "sus", "expected_units": [{
            "unit_id": "test-unit", "model_key": "target", "escalation_mode": "static",
            "expected_transcript_path": path.name,
        }]}],
    }
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    return contract


@pytest.mark.parametrize("module", ["sus", "aita", "epis"])
@pytest.mark.parametrize("saved_protocol", [None, "different-protocol"])
@pytest.mark.parametrize("terminal", [True, False])
def test_resume_rejects_unbound_or_different_benchmark_protocol(tmp_path, monkeypatch, module, saved_protocol, terminal):
    runner = importlib.import_module(f"{module}_bench.runner")
    artifact = {"condition_id": "target", "condition_hash": "sha256:model", "escalation_mode": "adaptive",
                "provider_refusal": terminal, "score_state": "excluded_provider_refusal" if terminal else "needs_scoring", "turns": []}
    if saved_protocol is not None:
        artifact["benchmark_condition_hash"] = saved_protocol
    if module == "sus":
        path = tmp_path / "transcripts" / runner.sus_transcript_filename(MODEL, SCENARIO, 1)
    elif module == "aita":
        path = tmp_path / "target_item0_side_a.json"
    else:
        path = tmp_path / "target_item0_delusion_side_a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))
    _contract(tmp_path, path)
    before = path.read_bytes()
    monkeypatch.setattr(runner, "api_call" if module != "sus" else "run_scenario",
                        lambda *a, **k: pytest.fail("Incompatible reuse must stop before generation"))
    with pytest.raises(ArtifactIdentityError, match="benchmark_condition_hash"):
        if module == "sus":
            runner._run_model_batch(MODEL, [SCENARIO], "offline", "judge", runs=1, temps=[None],
                                    reasoning_efforts=[None], delay=0, judge_panel=None,
                                    control_dir=tmp_path, escalation_mode="static")
        elif module == "aita":
            runner.run_conversation("target", "Synthetic post", 0, "side_a", tmp_path, None, {"target": MODEL})
        else:
            runner.run_conversation("target", {}, 0, "delusion", "side_a", tmp_path, None, {"target": MODEL})
    assert path.read_bytes() == before


@pytest.mark.parametrize("saved_protocol", [None, "different-protocol"])
def test_verifier_rejects_missing_or_conflicting_protocol_binding(tmp_path, saved_protocol):
    path = tmp_path / "unit.json"
    artifact = {"condition_id": "target", "condition_hash": "sha256:model", "escalation_mode": "static"}
    if saved_protocol is not None:
        artifact["benchmark_condition_hash"] = saved_protocol
    path.write_text(json.dumps(artifact))
    contract = _contract(tmp_path, path)
    report = evaluate_run_artifact_identity(tmp_path, contract=contract)
    assert not report["conformant"]
    assert any("benchmark_condition_hash" in issue["kind"] for issue in report["issues"])


def test_sus_live_artifact_preserves_protocol_binding(tmp_path):
    from sus_bench.runner import _write_live_transcript_artifact
    _contract(tmp_path, tmp_path / "unit.json")
    result = {"escalation_mode": "static", "benchmark_condition_hash": "current-protocol"}
    path = _write_live_transcript_artifact(tmp_path, model=MODEL, scenario=SCENARIO, result=result, run_number=1)
    artifact = json.loads(path.read_text())
    assert artifact["escalation_mode"] == "static"
    assert artifact["benchmark_condition_hash"] == "current-protocol"


def test_sus_result_serialization_preserves_protocol_binding(tmp_path):
    from sus_bench.report import write_json
    result = {"model": "target", "escalation_mode": "static", "benchmark_condition_hash": "current-protocol"}
    write_json([result], [], tmp_path / "summary.json")
    saved = json.loads((tmp_path / "summary-conversations.json").read_text())
    assert saved[0]["benchmark_condition_hash"] == "current-protocol"


@pytest.mark.parametrize("module", ["sus", "aita", "epis"])
def test_new_generation_stamps_protocol_and_can_resume_without_generation(tmp_path, monkeypatch, module):
    from suite_tools.provider_client import ProviderRefusalError
    runner = importlib.import_module(f"{module}_bench.runner")
    _contract(tmp_path, tmp_path / "unit.json")
    calls = []

    def refusal(*args, **kwargs):
        calls.append(True)
        raise ProviderRefusalError("offline refusal")

    if module == "sus":
        from suite_tools.prepare_run import _load_sus_scenarios
        from sus_bench.api import BenchmarkProviderRefusal
        def sus_refusal(*args, **kwargs):
            calls.append(True)
            raise BenchmarkProviderRefusal("offline refusal", model="target", role="model_under_test", latency_ms=0)
        monkeypatch.setattr(runner, "call_openrouter", sus_refusal)
        def run():
            return runner._run_model_batch(MODEL, _load_sus_scenarios("bridge_heights"), "offline", "judge",
                                           runs=1, temps=[None], reasoning_efforts=[None], delay=0,
                                           judge_panel=None, control_dir=tmp_path, escalation_mode="static")[0]
    else:
        monkeypatch.setattr(runner, "make_client", lambda *a, **k: None)
        monkeypatch.setattr(runner, "api_call", refusal)
        def run():
            if module == "aita":
                return runner.run_conversation("target", "Synthetic post", 0, "side_a", tmp_path, None, {"target": MODEL})
            return runner.run_conversation("target", {"statement": "Synthetic belief"}, 0, "delusion", "side_a", tmp_path, None, {"target": MODEL})

    try:
        generated = run()
    except ProviderRefusalError:
        filename = "target_item0_side_a.json" if module == "aita" else "target_item0_delusion_side_a.json"
        generated = json.loads((tmp_path / filename).read_text())
    assert generated["benchmark_condition_hash"] == "current-protocol"
    generated_calls = len(calls)
    assert generated_calls > 0
    reused = run()
    assert reused["benchmark_condition_hash"] == "current-protocol"
    assert len(calls) == generated_calls
