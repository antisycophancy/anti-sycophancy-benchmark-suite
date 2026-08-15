import json
import hashlib
import math
from pathlib import Path

import pytest

from suite_tools.run_contract import (
    CONTROL_SCHEMA_VERSION,
    CONTRACT_SCHEMA_VERSION,
    IDENTITY_PROJECTION_VERSION,
    JUDGE_PANEL_IDENTITY_KEYS,
    MODEL_CONDITION_IDENTITY_KEYS,
    PROVENANCE_IDENTITY_SCHEMA_VERSION,
    STOP_BEFORE_NEXT_PAID_CALL,
    _judge_panel_for_comparison,
    _model_condition_for_comparison,
    build_provenance_identity,
    load_run_contract,
    load_run_control,
    legacy_v3_provenance_hashes,
    provenance_hashes,
    redact_source_command,
    require_no_control_stop,
    should_stop_before_paid_call,
    stable_json_hash,
    summarize_contract,
    summarize_control,
    validate_judge_provenance_before_spend,
    validate_run_judge_provenance_before_spend,
    write_runtime_run_contract,
    write_run_contract,
    write_run_plan,
    write_run_control,
    RunControlStopRequested,
    JudgeProvenanceError,
    PreparedConfigProvenanceError,
    PreparedPricingProvenanceError,
    legacy_v1_provenance_hashes,
    validate_run_pricing_before_spend,
    validate_run_prepared_config_before_spend,
)
from suite_tools.prepare_run import main as prepare_run_main

from _reference_contracts import require_reference_contract

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())

_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "prepared"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_write_contract_redacts_secret_command_values(tmp_path):
    secret_flag_value = "target-secret-value"
    secret_equals_value = "support-secret-value"
    contract_path = write_run_contract(
        tmp_path,
        {
            "run_id": "smoke",
            "source_command": (
                "OPENROUTER_API_KEY=sk-live python run.py "
                "--api-key secret-value --models gemini-flash"
            ),
            "execute_command": (
                f"python -m aita_bench run --api-key {secret_flag_value} "
                f"--openrouter-api-key={secret_equals_value}"
            ),
            "modules": [],
        },
    )

    contract = load_run_contract(contract_path)
    serialized = contract_path.read_text()

    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert "sk-live" not in contract["source_command"]
    assert "secret-value" not in contract["source_command"]
    assert "OPENROUTER_API_KEY=<redacted>" in contract["source_command"]
    assert "--api-key '<redacted>'" in contract["source_command"]
    assert secret_flag_value not in serialized
    assert secret_equals_value not in serialized
    assert "--api-key '<redacted>'" in contract["execute_command"]
    assert "--openrouter-api-key=<redacted>" in contract["execute_command"]


def test_write_run_plan_redacts_nested_module_commands(tmp_path):
    credential_sentinel = "nested-credential-value"
    plan_path = write_run_plan(tmp_path, {
        "run_id": "redaction",
        "modules": [{
            "module": "aita",
            "execute_command": f"python -m aita_bench run --api-key={credential_sentinel}",
            "score_command": f"python -m aita_bench score --token {credential_sentinel}",
        }],
    })

    serialized = plan_path.read_text()
    plan = json.loads(serialized)
    assert credential_sentinel not in serialized
    assert "--api-key=<redacted>" in plan["modules"][0]["execute_command"]
    assert "--token '<redacted>'" in plan["modules"][0]["score_command"]


@pytest.mark.parametrize(
    "prepared_marker",
    [
        {"lifecycle_state": "prepared"},
        {"prepared": True},
    ],
)
def test_runtime_contract_writer_preserves_prepared_contract_bytes(tmp_path, prepared_marker):
    contract_path = write_run_contract(
        tmp_path,
        {
            "run_id": "prepared-run",
            **prepared_marker,
            "execute_argv": ["python", "-m", "benchmark", "run"],
            "score_argv": ["python", "-m", "benchmark", "score"],
            "modules": [],
        },
    )
    prepared_bytes = contract_path.read_bytes()

    returned_path = write_runtime_run_contract(
        tmp_path,
        {"run_id": "runtime-run", "modules": []},
    )

    assert returned_path == contract_path
    assert contract_path.read_bytes() == prepared_bytes


def test_runtime_contract_writer_replaces_unmarked_standalone_contract(tmp_path):
    write_run_contract(tmp_path, {"run_id": "old-runtime", "modules": []})

    write_runtime_run_contract(tmp_path, {"run_id": "new-runtime", "modules": []})

    assert load_run_contract(tmp_path)["run_id"] == "new-runtime"


def _write_prepared_config_contract(tmp_path):
    run_group = tmp_path / "prepared-run"
    run_dir = run_group / "sus"
    config_path = run_group / "_configs" / "public" / "sus-models.yaml"
    config_path.parent.mkdir(parents=True)
    config_bytes = b"models:\n  - id: model-a\n    base_url: https://example.test/v1\n"
    config_path.write_bytes(config_bytes)
    binding = {
        "path": "_configs/public/sus-models.yaml",
        "sha256": __import__("hashlib").sha256(config_bytes).hexdigest(),
        "bytes": len(config_bytes),
    }
    identity = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={"module": "sus"},
        sample_spec={"scenario_ids": ["s1"]},
        judge_panel={},
        model_conditions=[{"key": "model-a", "model_id": "model-a"}],
        execution={"prepared": True, "prepared_config": binding},
    )
    write_run_contract(
        run_dir,
        {
            "run_id": "prepared-run",
            "lifecycle_state": "prepared",
            "identity": identity,
            "modules": [{
                "module": "sus",
                "expected_artifacts": [{"kind": "rendered_models", **binding}],
            }],
        },
    )
    return run_dir, config_path


def test_prepared_config_validator_accepts_exact_frozen_file(tmp_path):
    run_dir, config_path = _write_prepared_config_contract(tmp_path)

    receipt = validate_run_prepared_config_before_spend(run_dir, config_path)

    assert receipt["verified"] is True
    assert receipt["sha256"] == __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [
    b"models:\n  - id: model-b\n",
    b"models:\n  - id: model-a\n    base_url: https://attacker.test/v1\n",
    b"models:\n  - id: model-a\n    request_options: {temperature: 1}\n",
])
def test_prepared_config_validator_rejects_mutated_config_bytes(tmp_path, mutation):
    run_dir, config_path = _write_prepared_config_contract(tmp_path)
    config_path.write_bytes(mutation)

    with pytest.raises(PreparedConfigProvenanceError, match="digest|byte count"):
        validate_run_prepared_config_before_spend(run_dir, config_path)


def test_prepared_config_validator_rejects_alternate_identical_file(tmp_path):
    run_dir, config_path = _write_prepared_config_contract(tmp_path)
    alternate = tmp_path / "same-looking.yaml"
    alternate.write_bytes(config_path.read_bytes())

    with pytest.raises(PreparedConfigProvenanceError, match="runtime path"):
        validate_run_prepared_config_before_spend(run_dir, alternate)


def test_prepared_config_validator_rejects_symlink_escape(tmp_path):
    run_dir, config_path = _write_prepared_config_contract(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(config_path.read_bytes())
    config_path.unlink()
    config_path.symlink_to(outside)

    with pytest.raises(PreparedConfigProvenanceError, match="outside"):
        validate_run_prepared_config_before_spend(run_dir, config_path)


def test_prepared_config_validator_rejects_missing_binding_in_v4_contract(tmp_path):
    run_dir = tmp_path / "run" / "sus"
    config_path = tmp_path / "run" / "_configs" / "models.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: []\n")
    write_run_contract(
        run_dir,
        {"run_id": "missing-binding", "lifecycle_state": "prepared", "modules": []},
    )

    with pytest.raises(PreparedConfigProvenanceError, match="binding"):
        validate_run_prepared_config_before_spend(run_dir, config_path)


def test_prepared_config_validator_preserves_legacy_unverified_contract(tmp_path):
    run_dir = tmp_path / "run" / "sus"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "run" / "models.yaml"
    config_path.write_text("models: []\n")
    contract = {
        "run_id": "legacy",
        "lifecycle_state": "prepared",
        "identity": build_provenance_identity(
            benchmark_family_id="sus",
            benchmark_spec={"module": "sus"},
            sample_spec={},
            judge_panel={},
            model_conditions=[],
            execution={"prepared": True},
        ),
        "modules": [],
    }
    contract["provenance"] = legacy_v1_provenance_hashes(contract)
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract_path.write_text(json.dumps(contract))
    before = contract_path.read_bytes()

    assert validate_run_prepared_config_before_spend(run_dir, config_path) is False
    assert contract_path.read_bytes() == before


def _write_prepared_pricing_contract(tmp_path, *, warning_threshold=None):
    pricing_path = tmp_path / "input-pricing.json"
    pricing_path.write_text(json.dumps({
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "provider": "test",
        "models": {
            "google/gemini-3-flash-preview": {
                "prompt": "0.0000005",
                "completion": "0.000003",
            },
            "google/gemini-3.1-pro-preview": {
                "prompt": "0.000002",
                "completion": "0.000012",
            },
        },
    }))
    output = tmp_path / "prepared-pricing"
    argv = [
        "--module", "sus",
        "--run-id", "prepared-pricing",
        "--output", str(output),
        "--models", "group:calibration_smoke",
        "--judge-set", "calibration",
        "--scenarios", "bridge_heights",
        "--runs", "1",
        "--pricing-snapshot", str(pricing_path),
        "--output-json",
    ]
    if warning_threshold is not None:
        argv.extend(["--warn-above-usd", str(warning_threshold)])
    status = prepare_run_main(argv)
    assert status in {0, 2}
    return output / "sus" / "RUN_CONTRACT.json"


def test_prepared_pricing_validator_accepts_exact_frozen_snapshot(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path, warning_threshold=10)

    receipt = validate_run_pricing_before_spend(contract_path)

    assert receipt["verified"] is True
    assert receipt["state"] == "estimated"
    assert receipt["path"] == "PRICING_SNAPSHOT.json"
    assert receipt["warning_state"] == "within"


def test_prepared_pricing_validator_rejects_mutated_snapshot_bytes(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path)
    snapshot_path = contract_path.parent / "PRICING_SNAPSHOT.json"
    snapshot_path.write_text(snapshot_path.read_text() + "\n")

    with pytest.raises(PreparedPricingProvenanceError, match="byte count|digest"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_recomputes_estimate_and_warning(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path, warning_threshold=10)
    contract = load_run_contract(contract_path)
    contract["cost_estimate"]["total_cost_usd"]["high"] += 1
    contract["cost_warning"]["estimated_high_usd"] += 1
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="cost estimate"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_recomputes_warning_classification(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path, warning_threshold=10)
    contract = load_run_contract(contract_path)
    contract["cost_warning"]["state"] = "exceeded"
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="cost warning"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_recomputes_authenticated_call_plan(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path)
    contract = load_run_contract(contract_path)
    contract["call_plan"]["total_calls"]["expected"] += 1
    contract["identity"]["execution"]["prepared_pricing"]["call_plan"] = contract[
        "call_plan"
    ]
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="call plan"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_rejects_authenticated_path_escape(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path)
    snapshot_path = contract_path.parent / "PRICING_SNAPSHOT.json"
    outside_path = contract_path.parent.parent / "outside-pricing.json"
    outside_path.write_bytes(snapshot_path.read_bytes())
    contract = load_run_contract(contract_path)
    for target in (
        contract["pricing_snapshot"],
        contract["identity"]["execution"]["prepared_pricing"]["pricing_snapshot"],
    ):
        target["path"] = "../outside-pricing.json"
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="path"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_rejects_authenticated_nonfinite_price(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path)
    snapshot_path = contract_path.parent / "PRICING_SNAPSHOT.json"
    pricing = json.loads(snapshot_path.read_text())
    pricing["models"]["google/gemini-3-flash-preview"]["prompt"] = math.nan
    snapshot_path.write_text(json.dumps(pricing))
    raw = snapshot_path.read_bytes()
    contract = load_run_contract(contract_path)
    binding = contract["identity"]["execution"]["prepared_pricing"][
        "pricing_snapshot"
    ]
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["bytes"] = len(raw)
    contract["pricing_snapshot"].update(binding)
    contract["pricing_snapshot"]["pricing_hash"] = stable_json_hash(pricing)
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="finite"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_rejects_authenticated_nonfinite_threshold(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path, warning_threshold=10)
    contract = load_run_contract(contract_path)
    contract["identity"]["execution"]["prepared_pricing"][
        "warning_threshold_usd"
    ] = math.nan
    contract["cost_warning"]["warning_threshold_usd"] = math.nan
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="finite"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_rejects_missing_binding_in_v4_contract(tmp_path):
    contract_path = _write_prepared_pricing_contract(tmp_path)
    contract = load_run_contract(contract_path)
    del contract["identity"]["execution"]["prepared_pricing"]
    contract.pop("provenance", None)
    write_run_contract(contract_path.parent, contract)

    with pytest.raises(PreparedPricingProvenanceError, match="binding"):
        validate_run_pricing_before_spend(contract_path)


def test_prepared_pricing_validator_preserves_legacy_unverified_contract(tmp_path):
    run_dir = tmp_path / "legacy" / "sus"
    run_dir.mkdir(parents=True)
    contract = {
        "run_id": "legacy-pricing",
        "lifecycle_state": "prepared",
        "identity": build_provenance_identity(
            benchmark_family_id="sus",
            benchmark_spec={"module": "sus"},
            sample_spec={},
            judge_panel={},
            model_conditions=[],
            execution={"prepared": True},
        ),
        "modules": [],
    }
    contract["provenance"] = legacy_v3_provenance_hashes(contract)
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract_path.write_text(json.dumps(contract))
    before = contract_path.read_bytes()

    assert validate_run_pricing_before_spend(contract_path) is False
    assert contract_path.read_bytes() == before


def test_summarize_contract_compares_expected_paths_and_model_ids(tmp_path):
    run_dir = tmp_path / "run-1"
    module_dir = run_dir / "aita"
    _write_json(module_dir / "RUN_STATUS.json", {"status": "running"})
    _write_json(module_dir / "sample.json", {"turns": []})
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": "run-1",
        "expected_models": [
            {"key": "gemini-flash", "model_id": "google/gemini-3.5-flash-preview"}
        ],
        "modules": [
            {
                "module": "aita",
                "stage": "generation",
                "output_dir": "aita",
                "expected_units": [
                    {
                        "unit_id": "aita:gemini-flash:item0:side_a",
                        "model_key": "gemini-flash",
                        "model_id": "google/gemini-3.0-flash-preview",
                        "item_idx": 0,
                        "side": "side_a",
                        "planned_turns": 5,
                        "expected_transcript_path": "sample.json",
                        "expected_score_path": "sample_scores.json",
                    }
                ],
                "expected_artifacts": [
                    {
                        "kind": "run_status",
                        "path": "RUN_STATUS.json",
                        "required_for": "diagnostic",
                    },
                    {
                        "kind": "final_results",
                        "path": "FINAL_RESULTS.json",
                        "required_for": "promotion",
                    },
                    {
                        "kind": "review_html",
                        "path": "review.html",
                        "required_for": "optional",
                    },
                ],
            }
        ],
    }

    summary = summarize_contract(
        contract,
        contract_path=run_dir / "RUN_CONTRACT.json",
        results_root=run_dir,
    )

    assert summary["run_id"] == "run-1"
    assert summary["expected_units"] == 1
    assert summary["complete_units"] == 0
    assert summary["missing_units"] == 1
    assert summary["expected_artifacts"] == 3
    assert summary["present_artifacts"] == 1
    assert summary["attention"] is True
    assert summary["modules"][0]["present_artifacts"] == 1
    assert summary["modules"][0]["missing_artifacts"][0]["kind"] == "final_results"
    assert summary["model_mismatches"] == [
        {
            "unit_id": "aita:gemini-flash:item0:side_a",
            "model_key": "gemini-flash",
            "expected_model_id": "google/gemini-3.5-flash-preview",
            "unit_model_id": "google/gemini-3.0-flash-preview",
        }
    ]


def test_control_summary_and_stop_guard(tmp_path):
    path = write_run_control(
        tmp_path,
        action=STOP_BEFORE_NEXT_PAID_CALL,
        reason="cost guard tripped",
        requested_by="dashboard",
    )
    control = load_run_control(path)
    summary = summarize_control(control, control_path=path)

    assert control["schema_version"] == CONTROL_SCHEMA_VERSION
    assert should_stop_before_paid_call(control) is True
    assert summary["active"] is True
    assert summary["label"] == "Stop before next paid call"
    assert summary["reason"] == "cost guard tripped"


def test_require_no_control_stop_records_event_and_raises(tmp_path):
    class Monitor:
        def __init__(self):
            self.events = []

        def record(self, event, **fields):
            self.events.append((event, fields))

    monitor = Monitor()
    write_run_control(tmp_path, action=STOP_BEFORE_NEXT_PAID_CALL, reason="operator stop")

    with pytest.raises(RunControlStopRequested):
        require_no_control_stop(tmp_path, monitor=monitor, context={"model": "gemini-flash"})

    assert len(monitor.events) == 1
    event, fields = monitor.events[0]
    assert event == "control_stop_requested"
    assert fields.pop("control_path").endswith("RUN_CONTROL.json")
    assert fields == {
        "action": STOP_BEFORE_NEXT_PAID_CALL,
        "state": "requested",
        "reason": "operator stop",
        "requested_by": "operator",
        "model": "gemini-flash",
    }


def test_stable_json_hash_ignores_key_order():
    assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash({"a": 1, "b": 2})


def test_redact_source_command_handles_flag_equals_form():
    redacted = redact_source_command("python run.py --token=abc --models gemini")

    assert "abc" not in redacted
    assert "--token=<redacted>" in redacted


def test_provenance_hashes_separate_comparison_model_and_execution():
    base_identity = {
        "benchmark_family_id": "aita",
        "benchmark_spec": {
            "module_version": "0.1.0",
            "prompt_hashes": {"seeker": "prompt-a"},
            "score_dimensions": ["outcome_a", "resistance_a"],
        },
        "sample_spec": {"dataset_mode": "nta-paired", "items": [0, 1], "sides": ["side_a", "side_b"]},
        "judge_panel": {
            "primary": "google/gemini-3.1-pro-preview",
            "rubric_version": "aita-judge-rubric-2026-05-11",
        },
    }
    chatgpt_identity = build_provenance_identity(
        **base_identity,
        model_conditions=[
            {"key": "gpt-5-5", "model_id": "openai/gpt-5.5", "endpoint": "openrouter"}
        ],
        execution={"run_id": "run-chatgpt", "created_at": "2026-05-24T00:00:00Z"},
    )
    claude_identity = build_provenance_identity(
        **base_identity,
        model_conditions=[
            {"key": "claude-opus", "model_id": "anthropic/claude-opus-4.7", "endpoint": "openrouter"}
        ],
        execution={"run_id": "run-claude", "created_at": "2026-08-24T00:00:00Z"},
    )

    chatgpt = provenance_hashes(chatgpt_identity)
    claude = provenance_hashes(claude_identity)

    assert chatgpt_identity["schema_version"] == PROVENANCE_IDENTITY_SCHEMA_VERSION
    assert chatgpt["comparison_spec_hash"] == claude["comparison_spec_hash"]
    assert chatgpt["benchmark_spec_hash"] == claude["benchmark_spec_hash"]
    assert chatgpt["sample_hash"] == claude["sample_hash"]
    assert chatgpt["judge_panel_hash"] == claude["judge_panel_hash"]
    assert chatgpt["model_conditions_hash"] != claude["model_conditions_hash"]
    assert chatgpt["run_execution_hash"] != claude["run_execution_hash"]


def test_provenance_hashes_change_when_sample_or_judge_changes():
    base = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "a"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1]},
        judge_panel={"primary": "judge-a", "rubric_version": "epis-judge-rubric-1"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-1"},
    )
    changed_sample = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "a"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1, 2]},
        judge_panel={"primary": "judge-a", "rubric_version": "epis-judge-rubric-1"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-2"},
    )
    changed_judge = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "a"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1]},
        judge_panel={"primary": "judge-b", "rubric_version": "epis-judge-rubric-1"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-3"},
    )

    base_hashes = provenance_hashes(base)
    assert provenance_hashes(changed_sample)["sample_hash"] != base_hashes["sample_hash"]
    assert provenance_hashes(changed_sample)["comparison_spec_hash"] != base_hashes["comparison_spec_hash"]
    assert provenance_hashes(changed_judge)["judge_panel_hash"] != base_hashes["judge_panel_hash"]
    assert provenance_hashes(changed_judge)["comparison_spec_hash"] != base_hashes["comparison_spec_hash"]


def test_provenance_hashes_ignore_judge_set_label_when_actual_panel_matches():
    labeled = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={"phase_prompts": {"post_analysis": "sus-post-v1"}, "score_dimensions": ["sus_score"]},
        sample_spec={"scenario_ids": ["bridge_heights"], "runs": 1},
        judge_panel={
            "judge_set": "calibration",
            "analyzer": "google/gemini-3-flash-preview",
            "panel": ["google/gemini-3.1-pro-preview"],
            "rubric_version": "sus-rubric-v1",
        },
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "prepared"},
    )
    runtime = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={"phase_prompts": {"post_analysis": "sus-post-v1"}, "score_dimensions": ["sus_score"]},
        sample_spec={"scenario_ids": ["bridge_heights"], "runs": 1},
        judge_panel={
            "analyzer": "google/gemini-3-flash-preview",
            "panel": ["google/gemini-3.1-pro-preview"],
            "rubric_version": "sus-rubric-v1",
        },
        model_conditions=[{"key": "model-b", "model_id": "provider/model-b"}],
        execution={"run_id": "runtime"},
    )

    labeled_hashes = provenance_hashes(labeled)
    runtime_hashes = provenance_hashes(runtime)

    assert labeled_hashes["judge_panel_hash"] == runtime_hashes["judge_panel_hash"]
    assert labeled_hashes["comparison_spec_hash"] == runtime_hashes["comparison_spec_hash"]
    assert labeled_hashes["benchmark_condition_hash"] == runtime_hashes["benchmark_condition_hash"]
    assert labeled_hashes["model_conditions_hash"] != runtime_hashes["model_conditions_hash"]


def test_benchmark_condition_hash_ignores_replicate_count_but_exact_runset_does_not():
    base = {
        "benchmark_family_id": "sus",
        "benchmark_spec": {
            "phase_prompts": {"post_analysis": "sus-post-v1"},
            "score_dimensions": ["sus_score"],
        },
        "judge_panel": {
            "analyzer": "google/gemini-3-flash-preview",
            "rubric_version": "sus-rubric-v1",
        },
        "model_conditions": [{"key": "gemini-flash", "model_id": "google/gemini-flash"}],
    }
    pilot = build_provenance_identity(
        **base,
        sample_spec={"scenario_ids": ["bridge_heights"], "runs": 3},
        execution={"run_id": "bridge-r3"},
    )
    expansion = build_provenance_identity(
        **base,
        sample_spec={"scenario_ids": ["bridge_heights"], "runs": 17},
        execution={"run_id": "bridge-r17"},
    )

    pilot_hashes = provenance_hashes(pilot)
    expansion_hashes = provenance_hashes(expansion)

    assert pilot_hashes["sample_condition_hash"] == expansion_hashes["sample_condition_hash"]
    assert pilot_hashes["benchmark_condition_hash"] == expansion_hashes["benchmark_condition_hash"]
    assert pilot_hashes["model_condition_hashes"][0]["benchmark_model_condition_hash"] == (
        expansion_hashes["model_condition_hashes"][0]["benchmark_model_condition_hash"]
    )
    assert pilot_hashes["sample_hash"] != expansion_hashes["sample_hash"]
    assert pilot_hashes["comparison_spec_hash"] != expansion_hashes["comparison_spec_hash"]
    assert pilot_hashes["run_execution_hash"] != expansion_hashes["run_execution_hash"]


def test_benchmark_condition_hash_groups_additive_model_batches():
    benchmark = {
        "benchmark_family_id": "sus",
        "benchmark_spec": {"phase_prompts": {"post_analysis": "sus-post-v1"}},
        "sample_spec": {"scenario_ids": ["bridge_heights"], "runs": 1},
        "judge_panel": {"analyzer": "google/gemini-3-flash-preview"},
    }
    first_models = build_provenance_identity(
        **benchmark,
        model_conditions=[
            {"key": "gpt-5-5", "model_id": "openai/gpt-5.5"},
            {"key": "claude-opus", "model_id": "anthropic/claude-opus-4.7"},
        ],
        execution={"run_id": "frontier-first"},
    )
    later_models = build_provenance_identity(
        **benchmark,
        model_conditions=[
            {"key": "gemini-pro", "model_id": "google/gemini-3.1-pro"},
        ],
        execution={"run_id": "frontier-later"},
    )

    first_hashes = provenance_hashes(first_models)
    later_hashes = provenance_hashes(later_models)

    assert first_hashes["benchmark_condition_hash"] == later_hashes["benchmark_condition_hash"]
    assert first_hashes["comparison_spec_hash"] == later_hashes["comparison_spec_hash"]
    assert first_hashes["model_conditions_hash"] != later_hashes["model_conditions_hash"]
    assert first_hashes["batch_condition_hash"] != later_hashes["batch_condition_hash"]


def test_model_condition_hash_ignores_operator_labels_and_local_config_paths():
    base = {
        "benchmark_family_id": "sus",
        "benchmark_spec": {"phase_prompts": {"post_analysis": "sus-post-v1"}},
        "sample_spec": {"scenario_ids": ["bridge_heights"], "runs": 1},
        "judge_panel": {"analyzer": "google/gemini-3-flash-preview"},
        "execution": {"run_id": "run-1"},
    }
    prepared = build_provenance_identity(
        **base,
        model_conditions=[
            {
                "key": "gemini-flash",
                "label": "Gemini Flash",
                "model_id": "google/gemini-flash",
                "endpoint": "openrouter",
                "max_parallel": 1,
                "source": "suite_models.yaml",
            }
        ],
    )
    runtime = build_provenance_identity(
        **base,
        model_conditions=[
            {
                "key": "calibration-gemini",
                "label": "Calibration Copy",
                "model_id": "google/gemini-flash",
                "endpoint": "openrouter",
                "max_parallel": 4,
                "source": "results/run/_configs/calibration/sus-models.yaml",
            }
        ],
    )

    prepared_hashes = provenance_hashes(prepared)
    runtime_hashes = provenance_hashes(runtime)

    assert prepared_hashes["model_conditions_hash"] == runtime_hashes["model_conditions_hash"]
    assert prepared_hashes["model_condition_hashes"][0]["hash"] == (
        runtime_hashes["model_condition_hashes"][0]["hash"]
    )
    assert prepared_hashes["batch_condition_hash"] == runtime_hashes["batch_condition_hash"]


def test_model_condition_hash_includes_provider_declared_profile_hash():
    base = {
        "benchmark_family_id": "sus",
        "benchmark_spec": {"phase_prompts": {"post_analysis": "sus-post-v1"}},
        "sample_spec": {"scenario_ids": ["bridge_heights"], "runs": 1},
        "judge_panel": {"analyzer": "google/gemini-3-flash-preview"},
        "execution": {"run_id": "run-1"},
    }
    first_profile = build_provenance_identity(
        **base,
        model_conditions=[
            {
                "model_id": "provider/served-endpoint",
                "endpoint": "openai_compatible",
                "served_profile_hash": "sha256:first",
            }
        ],
    )
    second_profile = build_provenance_identity(
        **base,
        model_conditions=[
            {
                "model_id": "provider/served-endpoint",
                "endpoint": "openai_compatible",
                "served_profile_hash": "sha256:second",
            }
        ],
    )

    assert provenance_hashes(first_profile)["model_conditions_hash"] != (
        provenance_hashes(second_profile)["model_conditions_hash"]
    )


def test_summarize_contract_derives_model_independent_comparison_hash_for_legacy_contracts(tmp_path):
    def contract(model_key, model_id, run_id):
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "run_id": run_id,
            "source_command": f"python run.py --models {model_key}",
            "expected_models": [{"key": model_key, "model_id": model_id}],
            "expected_judges": [{"role": "primary", "model_id": "judge/model"}],
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "dataset_mode": "nta-paired",
                    "output_dir": "aita",
                    "expected_units": [
                        {
                            "unit_id": f"aita:{model_key}:item0:side_a",
                            "model_key": model_key,
                            "model_id": model_id,
                            "item_idx": 0,
                            "side": "side_a",
                            "planned_turns": 5,
                            "expected_transcript_path": f"{model_key}_item0_side_a.json",
                        }
                    ],
                    "expected_artifacts": [],
                }
            ],
        }

    first = summarize_contract(contract("gpt-5-5", "openai/gpt-5.5", "run-1"), contract_path=tmp_path / "RUN_CONTRACT.json")
    second = summarize_contract(
        contract("claude-opus", "anthropic/claude-opus-4.7", "run-2"),
        contract_path=tmp_path / "RUN_CONTRACT.json",
    )

    assert first["provenance"]["comparison_spec_hash"] == second["provenance"]["comparison_spec_hash"]
    assert first["provenance"]["model_conditions_hash"] != second["provenance"]["model_conditions_hash"]
    assert first["provenance"]["run_execution_hash"] != second["provenance"]["run_execution_hash"]
    assert first["contract_fingerprint"] != second["contract_fingerprint"]


# ---------------------------------------------------------------------------
# Transport-config key exclusion: primary_config must not affect the hash
# ---------------------------------------------------------------------------

_SHARED_JUDGE_PANEL = {
    "panel": ["openai/gpt-5.5", "anthropic/claude-opus-4.7", "google/gemini-3.1-pro-preview"],
    "primary": "openai/gpt-5.5",
    "judge_prompt_hashes": {
        "consistency": "842d4d7c55541c104ace01a5f86a0380e723a084eed6fc5f6385bdafeee0e607",
        "outcome": "271c3c3b9d10523aa9314c1de78421b9953c30526bc384aa9b39e151bd55944f",
    },
    "rubric_version": "aita-judge-rubric-2026-05-30",
    "rubric_source_ids": ["MI-AUTONOMY-2012", "NICE-CG178"],
    "rubric_source_registry": "benchmark/RUBRIC_SOURCES.md",
    "seeker": "google/gemini-3-flash-preview",
}

_DUMMY_PRIMARY_CONFIG = {
    "api_key_env": "OPENROUTER_API_KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "model_id": "openai/gpt-5.5",
    "provider_api": "openai_compatible",
}

def test_judge_panel_hash_ignores_primary_config():
    """primary_config (transport metadata added in GPT-5.6 contracts) must not
    affect judge_panel_hash; adding or removing it must yield the same hash."""
    without_config = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"rubric": "aita-rubric-v1"},
        sample_spec={"scenario_ids": ["item0"], "runs": 1},
        judge_panel=_SHARED_JUDGE_PANEL,
        model_conditions=[{"key": "model-a", "model_id": "openai/gpt-5.6-sol"}],
        execution={"run_id": "run-without-config"},
    )
    with_config = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"rubric": "aita-rubric-v1"},
        sample_spec={"scenario_ids": ["item0"], "runs": 1},
        judge_panel={**_SHARED_JUDGE_PANEL, "primary_config": _DUMMY_PRIMARY_CONFIG},
        model_conditions=[{"key": "model-b", "model_id": "openai/gpt-5.6-sol"}],
        execution={"run_id": "run-with-config"},
    )

    h_without = provenance_hashes(without_config)["judge_panel_hash"]
    h_with = provenance_hashes(with_config)["judge_panel_hash"]
    assert h_without == h_with, (
        f"judge_panel_hash differs when primary_config is present:\n"
        f"  without primary_config: {h_without}\n"
        f"  with    primary_config: {h_with}"
    )


# ---------------------------------------------------------------------------
# Regression: frozen hashes from Fable frontier contracts
# (re-blessed 2026-07-24 for identity projection v3)
# ---------------------------------------------------------------------------
# HISTORY — pre-3392139 values, retained so the transition is traceable:
#
#     _FROZEN_FABLE_AITA_JUDGE_PANEL_HASH  73b62d298053d804a476ec024966e1101d0718ed0bbd2b48b28424c21e6e3ce9
#     _FROZEN_FABLE_EPIS_JUDGE_PANEL_HASH  9ecc2251c3fbcebcc0d1da187f4bef3081b68b565d497b7f7f205e67e6bc1df6
#
# Commit 3392139 (2026-07-18) narrowed the nested per-judge identity projection,
# dropping api_key_env/base_url/label.  The hashed content (panel, rubric_version,
# judge_prompt_hashes) is unchanged.  These mirror the canonical constants in
# suite_tools/assert_hash_panel.py — see the fuller HISTORY note and the
# projection-version guard there, plus PREREG_FREEZE_GPT56_20260716.json
# amendment P1.
_FROZEN_FABLE_AITA_JUDGE_PANEL_HASH = (
    "23617dd5aa1aecb4dcb2ec7b5f5946892b3e349d4ddc1ea37027c568f2161b67"
)
_FROZEN_FABLE_EPIS_JUDGE_PANEL_HASH = (
    "dfb77df3169963bf2c38c73ba030ef81c7b9cc694e2b5bf22624b4465d0b9c58"
)
_FABLE_FRONTIER_RUN = "fable-5-native-suite-n20-frontier-20260702-142711-frontier"


def test_frozen_fable_constants_mirror_assert_hash_panel():
    """These duplicates must never drift from the canonical constants.

    Two copies of the same frozen value is exactly how a re-blessing goes half
    done, so pin them to the single source of truth.
    """
    from suite_tools.assert_hash_panel import (
        CONSTANTS_BLESSED_FOR_PROJECTION_VERSION,
        FROZEN_AITA_JUDGE_PANEL_HASH,
        FROZEN_EPIS_JUDGE_PANEL_HASH,
    )

    assert _FROZEN_FABLE_AITA_JUDGE_PANEL_HASH == FROZEN_AITA_JUDGE_PANEL_HASH
    assert _FROZEN_FABLE_EPIS_JUDGE_PANEL_HASH == FROZEN_EPIS_JUDGE_PANEL_HASH
    assert CONSTANTS_BLESSED_FOR_PROJECTION_VERSION == "benchmark-identity-projection-v3"


def _load_prepared_contract(subpath: str) -> dict:
    path = _RESULTS_ROOT / subpath
    return json.loads(path.read_text())


def test_judge_panel_hash_regression_fable_frontier_aita():
    """Recomputing judge_panel_hash from the Fable frontier AITA identity must
    reproduce the frozen value exactly."""
    require_reference_contract(
        _RESULTS_ROOT,
        _RESULTS_ROOT / _FABLE_FRONTIER_RUN / "aita" / "RUN_CONTRACT.json",
    )
    contract = _load_prepared_contract(
        "fable-5-native-suite-n20-frontier-20260702-142711-frontier/aita/RUN_CONTRACT.json"
    )
    hashes = legacy_v3_provenance_hashes(contract["identity"])
    actual = hashes["judge_panel_hash"]
    assert actual == _FROZEN_FABLE_AITA_JUDGE_PANEL_HASH, (
        f"Fable AITA judge_panel_hash regression failed:\n"
        f"  expected: {_FROZEN_FABLE_AITA_JUDGE_PANEL_HASH}\n"
        f"  actual:   {actual}"
    )


def test_judge_panel_hash_regression_fable_frontier_epis():
    """Recomputing judge_panel_hash from the Fable frontier EPIS identity must
    reproduce the frozen value exactly."""
    require_reference_contract(
        _RESULTS_ROOT,
        _RESULTS_ROOT / _FABLE_FRONTIER_RUN / "epis" / "RUN_CONTRACT.json",
    )
    contract = _load_prepared_contract(
        "fable-5-native-suite-n20-frontier-20260702-142711-frontier/epis/RUN_CONTRACT.json"
    )
    hashes = legacy_v3_provenance_hashes(contract["identity"])
    actual = hashes["judge_panel_hash"]
    assert actual == _FROZEN_FABLE_EPIS_JUDGE_PANEL_HASH, (
        f"Fable EPIS judge_panel_hash regression failed:\n"
        f"  expected: {_FROZEN_FABLE_EPIS_JUDGE_PANEL_HASH}\n"
        f"  actual:   {actual}"
    )


def test_judge_panel_hash_invariant_after_primary_config_injection_aita():
    """Injecting a dummy primary_config into the Fable AITA identity must not
    change the judge_panel_hash from the frozen value."""
    require_reference_contract(
        _RESULTS_ROOT,
        _RESULTS_ROOT / _FABLE_FRONTIER_RUN / "aita" / "RUN_CONTRACT.json",
    )
    contract = _load_prepared_contract(
        "fable-5-native-suite-n20-frontier-20260702-142711-frontier/aita/RUN_CONTRACT.json"
    )
    identity = contract["identity"]
    # Inject a dummy primary_config that was not present in the Fable era
    identity_with_config = {
        **identity,
        "judge_panel": {
            **identity["judge_panel"],
            "primary_config": _DUMMY_PRIMARY_CONFIG,
        },
    }
    actual = legacy_v3_provenance_hashes(identity_with_config)["judge_panel_hash"]
    assert actual == _FROZEN_FABLE_AITA_JUDGE_PANEL_HASH, (
        f"Injecting primary_config changed Fable AITA judge_panel_hash:\n"
        f"  expected: {_FROZEN_FABLE_AITA_JUDGE_PANEL_HASH}\n"
        f"  actual:   {actual}"
    )


# ---------------------------------------------------------------------------
# Task 1: Whitelist identity projections + projection version
# ---------------------------------------------------------------------------


def test_judge_panel_projection_keeps_only_identity_keys():
    panel = _fixture("identity_judge_panel.json")
    projected = _judge_panel_for_comparison(panel)
    assert set(projected) <= JUDGE_PANEL_IDENTITY_KEYS
    for absent in ("judge_set", "judge_selector", "primary_config"):
        assert absent not in projected
    assert projected["panel"] == ["judge/model-a", "judge/model-b"]
    # nested projection: per-config transport keys dropped, identity kept
    assert projected["configs"] == [{
        "model_id": "judge/model-a",
        "provider_api": "openai_compatible",
        "condition_metadata": {"effort": "legacy_default", "provider_route": "openrouter", "role": "judge"},
    }]


def test_judge_panel_hash_insensitive_to_transport_fields():
    panel = _fixture("identity_judge_panel.json")
    moved = json.loads(json.dumps(panel))
    moved["configs"][0]["base_url"] = "https://other-proxy.example.test/v1"
    moved["configs"][0]["api_key_env"] = "OTHER_KEY"
    moved["judge_set"] = "renamed-selector"
    assert stable_json_hash(_judge_panel_for_comparison(panel)) == stable_json_hash(
        _judge_panel_for_comparison(moved)
    )


def test_judge_panel_hash_sensitive_to_identity_fields():
    panel = _fixture("identity_judge_panel.json")
    for mutate in (
        lambda p: p.__setitem__("rubric_version", "rubric-v4"),
        lambda p: p["judge_prompt_hashes"].__setitem__("outcome", "changed"),
        lambda p: p["configs"][0].__setitem__("model_id", "judge/model-z"),
        lambda p: p["configs"][0]["condition_metadata"].__setitem__("effort", "high"),
    ):
        changed = json.loads(json.dumps(panel))
        mutate(changed)
        assert stable_json_hash(_judge_panel_for_comparison(panel)) != stable_json_hash(
            _judge_panel_for_comparison(changed)
        ), mutate


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition_id", "judge-condition-b"),
        ("request_options", {"max_output_tokens": 4096, "temperature": 0}),
        ("served_profile_hash", "sha256:served-profile-b"),
        ("served_model_version", "judge-build-2026-08-13"),
        ("served_weights_fingerprint", "sha256:weights-b"),
    ],
)
def test_current_judge_panel_hash_covers_execution_identity(field, value):
    panel = _fixture("identity_judge_panel.json")
    panel["configs"][0].update(
        {
            "condition_id": "judge-condition-a",
            "request_options": {"max_output_tokens": 2048},
            "served_profile_hash": "sha256:served-profile-a",
            "served_model_version": "judge-build-a",
            "served_weights_fingerprint": "sha256:weights-a",
        }
    )
    before = stable_json_hash(_judge_panel_for_comparison(panel))
    changed = json.loads(json.dumps(panel))
    changed["configs"][0][field] = value

    assert stable_json_hash(_judge_panel_for_comparison(changed)) != before


def test_current_judge_panel_hash_covers_opaque_route_identity():
    panel = _fixture("identity_judge_panel.json")
    panel["configs"][0]["route_hash"] = "sha256:route-a"
    before = stable_json_hash(_judge_panel_for_comparison(panel))
    panel["configs"][0]["route_hash"] = "sha256:route-b"

    assert stable_json_hash(_judge_panel_for_comparison(panel)) != before


def test_current_judge_panel_hash_still_ignores_transport_secrets_and_urls():
    panel = _fixture("identity_judge_panel.json")
    before = stable_json_hash(_judge_panel_for_comparison(panel))
    changed = json.loads(json.dumps(panel))
    changed["configs"][0]["base_url"] = "https://other.example.invalid/v1"
    changed["configs"][0]["api_key_env"] = "OTHER_PRIVATE_KEY"
    changed["configs"][0]["label"] = "Operator label only"

    assert stable_json_hash(_judge_panel_for_comparison(changed)) == before


def test_model_condition_projection_drops_operator_and_derived_fields():
    condition = _fixture("identity_native_condition.json")
    projected = _model_condition_for_comparison(condition)
    assert set(projected) <= MODEL_CONDITION_IDENTITY_KEYS
    for absent in ("key", "label", "max_parallel", "source",
                   "condition_hash", "provider_condition_hash"):
        assert absent not in projected
    assert projected["model_id"] == "gpt-5.6-luna"
    assert projected["condition_metadata"]["effort"] == "high"


def test_model_condition_hash_sensitive_to_effort_and_profile():
    condition = _fixture("identity_openrouter_condition.json")
    base = stable_json_hash(_model_condition_for_comparison(condition))
    effort_changed = dict(condition, condition_metadata=dict(condition["condition_metadata"], effort="low"))
    profile_changed = dict(condition, profile_hash="different")
    assert stable_json_hash(_model_condition_for_comparison(effort_changed)) != base
    assert stable_json_hash(_model_condition_for_comparison(profile_changed)) != base


def test_provenance_hashes_stamp_projection_version_and_artifact():
    hashes = provenance_hashes({"modules": []})
    assert hashes["projection_version"] == IDENTITY_PROJECTION_VERSION
    assert hashes["artifact"] == "provenance_hashes"


def test_write_run_contract_rejects_supplied_stale_provenance(tmp_path):
    contract = {
        "run_id": "stale-provenance",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": {},
            "model_conditions": [],
            "execution": {},
        },
        "provenance": {
            "artifact": "provenance_hashes",
            "projection_version": IDENTITY_PROJECTION_VERSION,
            "comparison_spec_hash": "0" * 64,
        },
    }

    with pytest.raises(ValueError, match="stale provenance"):
        write_run_contract(tmp_path, contract)


def test_judge_provenance_validator_rejects_prompt_and_condition_drift():
    frozen = _fixture("identity_judge_panel.json")
    frozen["configs"][0].update(
        {
            "condition_id": "judge-condition-a",
            "request_options": {"max_output_tokens": 2048},
        }
    )
    resolved = json.loads(json.dumps(frozen))
    resolved["judge_prompt_hashes"]["outcome"] = "changed"
    resolved["configs"][0]["condition_id"] = "judge-condition-b"

    with pytest.raises(JudgeProvenanceError) as exc_info:
        validate_judge_provenance_before_spend(
            frozen,
            resolved,
            projection_version=IDENTITY_PROJECTION_VERSION,
        )

    assert set(exc_info.value.drift_fields) == {"configs", "judge_prompt_hashes"}


def test_judge_provenance_validator_uses_v3_compatibility_projection():
    frozen = _fixture("identity_judge_panel.json")
    resolved = json.loads(json.dumps(frozen))
    resolved["configs"][0]["condition_id"] = "newly-recorded-condition"
    resolved["configs"][0]["request_options"] = {"max_output_tokens": 4096}

    validate_judge_provenance_before_spend(
        frozen,
        resolved,
        projection_version="benchmark-identity-projection-v3",
    )


def test_current_run_judge_validator_rejects_missing_panel(tmp_path):
    contract = {
        "run_id": "missing-panel",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": {},
            "model_conditions": [],
            "execution": {},
        },
    }
    write_run_contract(tmp_path, contract)

    with pytest.raises(JudgeProvenanceError) as exc_info:
        validate_run_judge_provenance_before_spend(tmp_path, {})

    assert "judge_panel" in exc_info.value.drift_fields


def test_current_run_judge_validator_authenticates_stored_provenance(tmp_path):
    frozen = _fixture("identity_judge_panel.json")
    contract = {
        "run_id": "tampered-panel",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": frozen,
            "model_conditions": [],
            "execution": {},
        },
    }
    path = write_run_contract(tmp_path, contract)
    payload = json.loads(path.read_text())
    payload["identity"]["judge_panel"]["rubric_version"] = "tampered"
    path.write_text(json.dumps(payload))

    with pytest.raises(JudgeProvenanceError) as exc_info:
        validate_run_judge_provenance_before_spend(
            tmp_path,
            payload["identity"]["judge_panel"],
        )

    assert "stored_provenance" in exc_info.value.drift_fields


def test_prepared_contract_cannot_downgrade_by_deleting_panel_and_provenance(tmp_path):
    contract = {
        "run_id": "downgrade-attempt",
        "lifecycle_state": "prepared",
        "modules": [],
        "identity": {
            "benchmark_family_id": "test",
            "benchmark_spec": {},
            "sample_spec": {},
            "judge_panel": _fixture("identity_judge_panel.json"),
            "model_conditions": [],
            "execution": {},
        },
    }
    path = write_run_contract(tmp_path, contract)
    payload = json.loads(path.read_text())
    payload.pop("provenance")
    payload["identity"].pop("judge_panel")
    path.write_text(json.dumps(payload))

    with pytest.raises(JudgeProvenanceError) as exc_info:
        validate_run_judge_provenance_before_spend(tmp_path, {})

    assert "stored_provenance" in exc_info.value.drift_fields


# ---------------------------------------------------------------------------
# Task 2: compare_provenance and hash-panel rejection tests
# ---------------------------------------------------------------------------

import pytest
from suite_tools.run_contract import compare_provenance


def _contract(selection):
    return {
        "modules": [{"module": "aita", "expected_units": [], "selection": selection}],
        "expected_judges": [{"model_id": "judge-1"}],
        "expected_models": [{"model_id": "m", "condition_id": "m-high"}],
    }


def test_compare_provenance_same_instrument_is_comparable():
    result = compare_provenance(_contract("n20"), _contract("n20"))
    assert result["comparable"] is True
    assert result["match"]["judge_panel_hash"] is True
    assert result["projection_version"] == "benchmark-identity-projection-v4"


def test_compare_provenance_flags_instrument_difference():
    result = compare_provenance(_contract("n20"), _contract("n21-other"))
    assert result["comparable"] is False
    assert result["match"]["judge_panel_hash"] is True


def test_stale_stored_hashes_are_ignored_in_favor_of_recompute():
    """A contract carrying WRONG stored hash strings still compares by
    recomputed content — stored hashes are display metadata, never inputs."""
    a = _contract("n20")
    b = _contract("n20")
    b["provenance"] = {"judge_panel_hash": "0000stale0000", "comparison_spec_hash": "0000stale0000"}
    assert compare_provenance(a, b)["comparable"] is True


def test_unknown_identity_schema_version_is_an_error():
    bad = {"schema_version": "benchmark-provenance-identity-v99", "benchmark_spec": {}}
    with pytest.raises(ValueError, match="v99"):
        compare_provenance(bad, bad)


def test_hash_panel_is_rejected_as_identity_input():
    from suite_tools.run_contract import provenance_hashes
    panel = provenance_hashes(_contract("n20"))
    with pytest.raises(ValueError, match="hash panel"):
        compare_provenance(panel, _contract("n20"))


# Task 3: identity-direction tests for item_hash and expected_transcript_path


def test_epis_item_hash_differentiates_sample_identity():
    """Two EPIS units identical except for item_hash must hash to different
    sample-axis values, proving item_hash enters sample identity."""
    from suite_tools.run_contract import _unit_sample_axis
    base_unit = {
        "unit_id": "epis:gemini-flash:delusion:item0:side_a",
        "model_key": "gemini-flash",
        "model_id": "google/gemini-3-flash-preview",
        "item_idx": 0,
        "item_hash": "aaaa1111bbbb2222",
        "test_type": "delusion",
        "side": "side_a",
        "planned_turns": 3,
    }
    alt_unit = {**base_unit, "item_hash": "cccc3333dddd4444"}
    axis_base = _unit_sample_axis(base_unit, module_name="epis")
    axis_alt = _unit_sample_axis(alt_unit, module_name="epis")
    assert stable_json_hash(axis_base) != stable_json_hash(axis_alt)


def test_expected_transcript_path_excluded_from_sample_identity():
    """Adding expected_transcript_path to a unit must NOT change the
    sample-axis hash — it is an execution detail, not sample identity."""
    from suite_tools.run_contract import _unit_sample_axis
    unit = {
        "unit_id": "epis:gemini-flash:delusion:item0:side_a",
        "model_key": "gemini-flash",
        "model_id": "google/gemini-3-flash-preview",
        "item_idx": 0,
        "item_hash": "aaaa1111bbbb2222",
        "test_type": "delusion",
        "side": "side_a",
        "planned_turns": 3,
    }
    unit_with_path = {**unit, "expected_transcript_path": "gemini-flash_item0_delusion_side_a.json"}
    axis_without = _unit_sample_axis(unit, module_name="epis")
    axis_with = _unit_sample_axis(unit_with_path, module_name="epis")
    assert stable_json_hash(axis_without) == stable_json_hash(axis_with)
