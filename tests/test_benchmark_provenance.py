import json

from suite_tools.live_dashboard import DashboardOptions, build_dashboard_data
from suite_tools.run_contract import (
    CONTRACT_SCHEMA_VERSION,
    build_provenance_identity,
    provenance_hashes,
    write_run_contract,
)


def test_different_models_can_share_one_comparison_spec():
    shared = {
        "benchmark_family_id": "aita",
        "benchmark_spec": {
            "module_version": "0.1.0",
            "prompt_hashes": {"seeker": "seek-v1", "flip": "flip-v1"},
            "score_dimensions": ["outcome_a", "resistance_a"],
        },
        "sample_spec": {"dataset_mode": "nta-paired", "items": [0, 1], "sides": ["side_a", "side_b"]},
        "judge_panel": {"primary": "google/gemini-3.1-pro-preview", "rubric_version": "aita-v1"},
    }

    first = provenance_hashes(
        build_provenance_identity(
            **shared,
            model_conditions=[{"key": "gpt-5-5", "model_id": "openai/gpt-5.5"}],
            execution={"run_id": "chatgpt-first", "created_at": "2026-05-24T00:00:00Z"},
        )
    )
    second = provenance_hashes(
        build_provenance_identity(
            **shared,
            model_conditions=[{"key": "claude-opus", "model_id": "anthropic/claude-opus-4.7"}],
            execution={"run_id": "claude-later", "created_at": "2026-08-24T00:00:00Z"},
        )
    )

    assert first["comparison_spec_hash"] == second["comparison_spec_hash"]
    assert first["model_conditions_hash"] != second["model_conditions_hash"]
    assert first["run_execution_hash"] != second["run_execution_hash"]


def test_benchmark_or_judge_changes_break_comparison_spec():
    base = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "prompt-v1"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1]},
        judge_panel={"primary": "judge-a", "rubric_version": "rubric-v1"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-a"},
    )
    changed_prompt = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "prompt-v2"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1]},
        judge_panel={"primary": "judge-a", "rubric_version": "rubric-v1"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-b"},
    )
    changed_judge = build_provenance_identity(
        benchmark_family_id="epis",
        benchmark_spec={"prompt_hashes": {"delusion": "prompt-v1"}, "score_dimensions": ["persistence"]},
        sample_spec={"items": [0, 1]},
        judge_panel={"primary": "judge-a", "rubric_version": "rubric-v2"},
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-c"},
    )

    base_hash = provenance_hashes(base)["comparison_spec_hash"]
    assert provenance_hashes(changed_prompt)["comparison_spec_hash"] != base_hash
    assert provenance_hashes(changed_judge)["comparison_spec_hash"] != base_hash


def test_judge_prompt_hash_changes_break_judge_panel_hash():
    base = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"prompt_hashes": {"seeker": "prompt-v1"}, "score_dimensions": ["verdict_alignment_a"]},
        sample_spec={"items": [0]},
        judge_panel={
            "primary": "judge-a",
            "panel": ["judge-a", "judge-b", "judge-c"],
            "rubric_version": "rubric-v1",
            "judge_prompt_hashes": {"verdict_alignment": "judge-prompt-v1"},
        },
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-a"},
    )
    changed_judge_prompt = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"prompt_hashes": {"seeker": "prompt-v1"}, "score_dimensions": ["verdict_alignment_a"]},
        sample_spec={"items": [0]},
        judge_panel={
            "primary": "judge-a",
            "panel": ["judge-a", "judge-b", "judge-c"],
            "rubric_version": "rubric-v1",
            "judge_prompt_hashes": {"verdict_alignment": "judge-prompt-v2"},
        },
        model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
        execution={"run_id": "run-b"},
    )

    base_hashes = provenance_hashes(base)
    changed_hashes = provenance_hashes(changed_judge_prompt)
    assert changed_hashes["judge_panel_hash"] != base_hashes["judge_panel_hash"]
    assert changed_hashes["comparison_spec_hash"] != base_hashes["comparison_spec_hash"]


def test_mocked_full_suite_golden_contracts_share_identity_shape(tmp_path):
    run_id = "golden-suite"
    module_specs = {
        "aita": {
            "benchmark_spec": {"prompt_hashes": {"seeker": "aita-seeker-v1"}, "score_dimensions": ["outcome_a"]},
            "sample_spec": {"dataset_mode": "nta-paired", "items": [0], "sides": ["side_a", "side_b"]},
            "judge_panel": {"primary": "judge/model", "rubric_version": "aita-rubric-v1"},
        },
        "epis": {
            "benchmark_spec": {"prompt_hashes": {"delusion": "epis-delusion-v1"}, "score_dimensions": ["persistence"]},
            "sample_spec": {"test_types": ["delusion"], "items": {"delusion": [{"position": 0, "item_hash": "h"}]}},
            "judge_panel": {"primary": "judge/model", "rubric_version": "epis-rubric-v1"},
        },
        "sus": {
            "benchmark_spec": {"phase_prompts": {"post_analysis": "sus-post-v1"}, "score_dimensions": ["sus_score"]},
            "sample_spec": {"scenario_ids": ["bridge_heights"], "runs": 1},
            "judge_panel": {"panel": ["judge/model"], "rubric_version": "sus-rubric-v1"},
        },
    }

    for module, spec in module_specs.items():
        module_dir = tmp_path / run_id / module
        write_run_contract(
            module_dir,
            {
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "run_id": run_id,
                "contract_scope": "module",
                "results_root": str(module_dir),
                "identity": build_provenance_identity(
                    benchmark_family_id=module,
                    benchmark_spec=spec["benchmark_spec"],
                    sample_spec=spec["sample_spec"],
                    judge_panel=spec["judge_panel"],
                    model_conditions=[
                        {"key": "gpt-5-5", "model_id": "openai/gpt-5.5", "endpoint": "openrouter"}
                    ],
                    execution={"run_id": run_id, "module": module},
                ),
                "expected_models": [{"key": "gpt-5-5", "model_id": "openai/gpt-5.5"}],
                "expected_judges": [{"role": "primary", "model_id": "judge/model"}],
                "modules": [
                    {
                        "module": module,
                        "stage": "generation" if module != "sus" else "run",
                        "output_dir": str(module_dir),
                        "expected_units": [],
                        "expected_artifacts": [],
                    }
                ],
            },
        )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["contract_count"] == 3
    contracts = {contract["identity"]["benchmark_family_id"]: contract for contract in data["contracts"]}
    assert set(contracts) == {"aita", "epis", "sus"}
    for module, contract in contracts.items():
        provenance = contract["provenance"]
        assert provenance["benchmark_family_id"] == module
        assert provenance["comparison_spec_hash"]
        assert provenance["model_conditions_hash"]
        assert provenance["run_execution_hash"]

    group = next(group for group in data["groups"] if group["run_id"] == run_id)
    assert group["contract_only"] is True
    assert len(group["contracts"]) == 3


def test_dashboard_golden_payload_is_json_serializable(tmp_path):
    write_run_contract(
        tmp_path / "run-1" / "aita",
        {
            "run_id": "run-1",
            "identity": build_provenance_identity(
                benchmark_family_id="aita",
                benchmark_spec={"prompt_hashes": {"seeker": "v1"}},
                sample_spec={"items": [0]},
                judge_panel={"primary": "judge/model"},
                model_conditions=[{"key": "model-a", "model_id": "provider/model-a"}],
                execution={"run_id": "run-1"},
            ),
            "modules": [],
        },
    )

    payload = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    json.dumps(payload)
