import hashlib
import json
from pathlib import Path

import pytest
import yaml

from suite_tools.live_dashboard import DashboardOptions, build_dashboard_data
from suite_tools.preflight_conditions import collect_targets_from_run_dir
from suite_tools.prepare_run import main, prepare_aita_run, prepare_epis_run, prepare_sus_run
from suite_tools.paid_call_lease import set_paid_call_policy
from suite_tools.run_contract import load_run_contract
from suite_tools.sealed_pack import seal_files


def _assert_prepared_config_binding(contract, config_path, expected_relative_path):
    raw = config_path.read_bytes()
    expected = {
        "path": expected_relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    assert contract["identity"]["execution"]["prepared_config"] == expected
    artifacts = [
        artifact
        for artifact in contract["modules"][0]["expected_artifacts"]
        if artifact.get("kind") == "rendered_models"
    ]
    assert len(artifacts) == 1
    assert {key: artifacts[0][key] for key in expected} == expected


def test_prepare_sus_run_writes_contract_and_rendered_config(tmp_path):
    contract_path = prepare_sus_run(
        run_id="bridge-raw-smoke",
        output_root=tmp_path / "bridge-raw-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
        source_command="python -m suite_tools.prepare_run --module sus",
    )

    contract = load_run_contract(contract_path)
    config_path = tmp_path / "bridge-raw-smoke" / "_configs" / "calibration" / "sus-models.yaml"
    plan_path = tmp_path / "bridge-raw-smoke" / "RUN_PLAN.json"
    rendered = yaml.safe_load(config_path.read_text())
    plan = json.loads(plan_path.read_text())
    _assert_prepared_config_binding(
        contract,
        config_path,
        "_configs/calibration/sus-models.yaml",
    )

    assert contract_path == tmp_path / "bridge-raw-smoke" / "sus" / "RUN_CONTRACT.json"
    assert plan_path.exists()
    assert plan["schema_version"] == "benchmark-run-plan-v1"
    assert plan["run_id"] == "bridge-raw-smoke"
    assert plan["modules"][0]["module"] == "sus"
    assert plan["modules"][0]["contract_path"].endswith("bridge-raw-smoke/sus/RUN_CONTRACT.json")
    assert plan["modules"][0]["expected_units"] == 1
    assert plan["modules"][0]["execute_command"] == contract["execute_command"]
    assert plan["modules"][0]["score_command"] == contract["score_command"]
    assert plan["modules"][0]["provenance"]["benchmark_condition_hash"]
    assert plan["modules"][0]["provenance"]["comparison_spec_hash"]
    assert plan["modules"][0]["provenance"] == contract["provenance"]
    assert contract["lifecycle_state"] == "prepared"
    assert contract["run_id"] == "bridge-raw-smoke"
    assert contract["model_selector"] == "group:calibration_smoke"
    assert contract["judge_set"] == "calibration"
    assert contract["agent_profile"] == "default"
    assert contract["identity"]["benchmark_family_id"] == "sus"
    assert contract["identity"]["benchmark_spec"]["escalation_mode"] == "adaptive"
    assert contract["identity"]["benchmark_spec"]["phase_prompts"]["adaptive_escalation"]
    assert contract["identity"]["sample_spec"]["scenario_ids"] == ["bridge_heights"]
    assert contract["identity"]["sample_spec"]["runs"] == 1
    assert contract["identity"]["benchmark_spec"]["score_dimensions"][:2] == [
        "cap_outcome",
        "sus_response_score",
    ]
    assert contract["identity"]["benchmark_spec"]["scoring_contract"]["primary_outcome_dimensions"] == [
        "cap_outcome"
    ]
    assert "target_utility" in contract["identity"]["benchmark_spec"]["score_dimensions"]
    assert contract["identity"]["benchmark_spec"]["legacy_diagnostic_dimensions"] == ["irq", "pr", "er", "ca"]
    assert "score_component_weights" in contract["identity"]["benchmark_spec"]
    assert contract["expected_models"][0]["key"] == "gemini-flash"
    assert contract["expected_models"][0]["condition_id"] == "gemini-flash"
    assert contract["expected_models"][0]["condition_hash"].startswith("sha256:")
    assert rendered["models"][0]["condition_id"] == contract["expected_models"][0]["condition_id"]
    assert rendered["models"][0]["condition_hash"] == contract["expected_models"][0]["condition_hash"]
    assert contract["modules"][0]["expected_units"][0]["scenario"] == "bridge_heights"
    assert contract["modules"][0]["expected_units"][0]["planned_escalations"] > 0
    assert contract["modules"][0]["expected_units"][0]["escalation_mode"] == "adaptive"
    assert contract["modules"][0]["escalation_mode"] == "adaptive"
    assert contract["call_plan"]["schema_version"] == "benchmark-cost-estimate-v1"
    assert contract["call_plan"]["total_calls"]["expected"] > 0
    prepared_pricing = contract["identity"]["execution"]["prepared_pricing"]
    assert prepared_pricing == {
        "schema_version": "benchmark-prepared-pricing-v1",
        "call_plan": contract["call_plan"],
    }
    assert {line["role"] for line in contract["call_plan"]["lines"]} >= {
        "model_under_test",
        "support",
        "judge",
    }
    assert any(
        artifact["kind"] == "conversations_json" and artifact["required_for"] == "scoring"
        for artifact in contract["modules"][0]["expected_artifacts"]
    )
    assert any(
        artifact["kind"] == "final_results" and artifact["required_for"] == "promotion"
        for artifact in contract["modules"][0]["expected_artifacts"]
    )
    assert "--output" in contract["execute_command"]
    assert "--escalation-mode adaptive" in contract["execute_command"]
    assert "sus_bench score" in contract["score_command"]
    assert str((tmp_path / "bridge-raw-smoke" / "sus").resolve()) in contract["execute_command"]
    import sys
    assert contract["execute_argv"][:3] == [sys.executable, "-m", "sus_bench"]
    assert contract["execute_cwd"].endswith("/sus-bench")
    assert contract["execute_steps"] == [{"cwd": contract["execute_cwd"], "argv": contract["execute_argv"]}]
    assert contract["score_argv"][:4] == [sys.executable, "-m", "sus_bench", "score"]
    assert contract["score_cwd"].endswith("/sus-bench")
    assert rendered["models"][0]["id"] == "google/gemini-3-flash-preview"
    assert rendered["analyzer"] == "google/gemini-3-flash-preview"


def test_prepare_sus_run_can_pin_static_escalation_mode(tmp_path):
    contract_path = prepare_sus_run(
        run_id="bridge-static-smoke",
        output_root=tmp_path / "bridge-static-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
        escalation_mode="static",
    )

    contract = load_run_contract(contract_path)

    assert contract["identity"]["benchmark_spec"]["escalation_mode"] == "static"
    assert contract["modules"][0]["escalation_mode"] == "static"
    assert "--escalation-mode static" in contract["execute_command"]


def test_prepared_contract_is_visible_without_counting_missing_outputs_as_gap(tmp_path):
    prepare_sus_run(
        run_id="bridge-local-endpoint-smoke",
        output_root=tmp_path / "bridge-local-endpoint-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:local_endpoint_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
    )

    payload = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    contract = payload["contracts"][0]

    assert payload["summary"]["contract_count"] == 1
    assert payload["summary"]["contract_attention_count"] == 0
    assert payload["summary"]["contract_expected_units"] == 1
    assert payload["summary"]["contract_complete_units"] == 0
    assert payload["groups"][0]["contract_only"] is True
    assert contract["prepared"] is True
    assert contract["attention"] is False
    assert contract["execute_command"]
    assert contract["provenance"]["comparison_spec_hash"]
    assert contract["provenance"]["benchmark_condition_hash"]
    assert contract["provenance"]["sample_condition_hash"]
    assert contract["provenance"]["model_conditions_hash"]


def test_prepare_run_cli_has_no_provider_side_effects(tmp_path, capsys, monkeypatch):
    lease_dir = tmp_path / "leases"
    set_paid_call_policy(64, lease_dir=lease_dir)
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    monkeypatch.delenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", raising=False)
    monkeypatch.delenv("BENCHMARK_MAX_ACTIVE_CALLS", raising=False)
    monkeypatch.setattr(
        "suite_tools.prepare_run.read_repo_env_values",
        lambda _names: {"BENCHMARK_PAID_CALL_MAX_ACTIVE": "3"},
    )
    monkeypatch.chdir(tmp_path)
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(json.dumps({
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "generated_at": "2026-08-13T12:00:00Z",
        "source": "https://openrouter.ai/api/v1/models",
        "models": {
            "google/gemini-3-flash-preview": {
                "prompt": "0.0000005",
                "completion": "0.000003",
                "source": "test_openrouter_snapshot",
            },
            "google/gemini-3.1-pro-preview": {
                "prompt": "0.000002",
                "completion": "0.000012",
                "source": "test_openrouter_snapshot",
            },
        },
    }))
    status = main(
        [
            "--module",
            "sus",
            "--run-id",
            "bridge-cli-smoke",
            "--output",
            str(tmp_path / "bridge-cli-smoke"),
            "--models",
            "group:calibration_smoke",
            "--judge-set",
            "calibration",
            "--scenarios",
            "bridge_heights",
            "--runs",
            "1",
            "--pricing-snapshot",
            str(pricing_path),
        ]
    )

    out = capsys.readouterr().out
    contract_path = tmp_path / "bridge-cli-smoke" / "sus" / "RUN_CONTRACT.json"

    assert status == 0
    assert contract_path.exists()
    assert "Prepared contract:" in out
    assert "Planning cost estimate:" in out
    assert "Generation:" in out
    assert "Scoring:" in out
    assert "Actual provider billing may differ." in out
    assert "Effective paid-call limit: 3" in out
    assert "source: environment:BENCHMARK_PAID_CALL_MAX_ACTIVE" in out
    assert "Review the contract and exact-condition preflight receipt, then schedule with:" in out
    assert "suite_tools.scheduler run" in out
    assert "--max-active-calls 3" in out
    assert "--stop-on-attention" in out
    contract = load_run_contract(contract_path)
    assert contract["cost_estimate"]["state"] == "estimated"
    assert contract["cost_estimate"]["total_cost_usd"]["expected"] > 0
    assert set(contract["cost_estimate"]["cost_by_stage"]) == {"generation", "scoring"}
    frozen_pricing = tmp_path / "bridge-cli-smoke" / "sus" / "PRICING_SNAPSHOT.json"
    assert frozen_pricing.exists()
    assert contract["pricing_snapshot"]["path"] == "PRICING_SNAPSHOT.json"
    frozen_pricing_bytes = frozen_pricing.read_bytes()
    assert contract["pricing_snapshot"]["sha256"] == hashlib.sha256(
        frozen_pricing_bytes
    ).hexdigest()
    assert contract["pricing_snapshot"]["bytes"] == len(frozen_pricing_bytes)
    prepared_pricing = contract["identity"]["execution"]["prepared_pricing"]
    assert prepared_pricing["schema_version"] == "benchmark-prepared-pricing-v1"
    assert prepared_pricing["call_plan"] == contract["call_plan"]
    assert prepared_pricing["pricing_snapshot"] == {
        "path": "PRICING_SNAPSHOT.json",
        "sha256": hashlib.sha256(frozen_pricing_bytes).hexdigest(),
        "bytes": len(frozen_pricing_bytes),
    }
    plan = json.loads((tmp_path / "bridge-cli-smoke" / "RUN_PLAN.json").read_text())
    assert plan["modules"][0]["provenance"] == contract["provenance"]
    assert contract["pricing_snapshot"]["generated_at"] == "2026-08-13T12:00:00Z"
    assert contract["pricing_snapshot"]["source"] == "https://openrouter.ai/api/v1/models"
    assert json.loads(frozen_pricing.read_text())["models"]


def test_prepare_run_cost_warning_threshold_requires_a_pricing_snapshot(tmp_path, capsys):
    output = tmp_path / "blocked-without-pricing"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--module",
                "sus",
                "--run-id",
                "blocked-without-pricing",
                "--output",
                str(output),
                "--warn-above-usd",
                "5",
            ]
        )

    assert exc.value.code == 2
    assert not output.exists()
    assert "--warn-above-usd requires --pricing-snapshot" in capsys.readouterr().err


@pytest.mark.parametrize("bad_price", ["-1", "NaN", "Infinity", "-Infinity"])
def test_prepare_run_rejects_invalid_pricing_before_writing_contract(
    tmp_path, capsys, bad_price
):
    pricing_path = tmp_path / "invalid-pricing.json"
    pricing_path.write_text(json.dumps({
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "models": {
            "google/gemini-3-flash-preview": {
                "prompt": bad_price,
                "completion": "0.000003",
            }
        },
    }))
    output = tmp_path / "must-not-exist"

    with pytest.raises(SystemExit) as exc:
        main([
            "--module", "sus",
            "--run-id", "invalid-pricing",
            "--output", str(output),
            "--pricing-snapshot", str(pricing_path),
        ])

    assert exc.value.code == 2
    assert not output.exists()
    assert "non-negative finite" in capsys.readouterr().err


@pytest.mark.parametrize("bad_threshold", ["NaN", "Infinity", "-Infinity"])
def test_prepare_run_rejects_nonfinite_warning_threshold_before_output(
    tmp_path, capsys, bad_threshold
):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(json.dumps({
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "models": {},
    }))
    output = tmp_path / "must-not-exist"

    with pytest.raises(SystemExit) as exc:
        main([
            "--module", "sus",
            "--run-id", "invalid-threshold",
            "--output", str(output),
            "--pricing-snapshot", str(pricing_path),
            f"--warn-above-usd={bad_threshold}",
        ])

    assert exc.value.code == 2
    assert not output.exists()
    assert "finite" in capsys.readouterr().err


def test_prepare_run_cost_warning_threshold_blocks_surprising_high_estimate(tmp_path, capsys):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-pricing-snapshot-v1",
                "units": "per_token",
                "models": {
                    "google/gemini-3-flash-preview": {
                        "input": "0.0000005",
                        "output": "0.000003",
                    },
                    "google/gemini-3.1-pro-preview": {
                        "input": "0.000002",
                        "output": "0.000012",
                    },
                },
            }
        )
    )
    output = tmp_path / "cost-warning"

    status = main(
        [
            "--module",
            "sus",
            "--run-id",
            "cost-warning",
            "--output",
            str(output),
            "--models",
            "group:calibration_smoke",
            "--judge-set",
            "calibration",
            "--scenarios",
            "bridge_heights",
            "--runs",
            "1",
            "--pricing-snapshot",
            str(pricing_path),
            "--warn-above-usd",
            "0.000001",
        ]
    )

    contract = load_run_contract(output / "sus" / "RUN_CONTRACT.json")
    guard = contract["cost_warning"]
    assert status == 2
    assert guard["state"] == "exceeded"
    assert guard["warning_threshold_usd"] == 0.000001
    assert guard["estimated_high_usd"] > guard["warning_threshold_usd"]
    assert contract["identity"]["execution"]["prepared_pricing"][
        "warning_threshold_usd"
    ] == 0.000001
    assert "exceeds warning threshold" in capsys.readouterr().out


def test_prepare_aita_run_writes_contract_and_score_command(tmp_path):
    contract_path = prepare_aita_run(
        run_id="aita-agent-smoke",
        output_root=tmp_path / "aita-agent-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
        source_command="python -m suite_tools.prepare_run --module aita",
    )

    contract = load_run_contract(contract_path)
    config_path = tmp_path / "aita-agent-smoke" / "_configs" / "calibration" / "aita-models.yaml"
    plan = json.loads((tmp_path / "aita-agent-smoke" / "RUN_PLAN.json").read_text())

    assert contract_path == tmp_path / "aita-agent-smoke" / "aita" / "RUN_CONTRACT.json"
    assert config_path.exists()
    _assert_prepared_config_binding(
        contract,
        config_path,
        "_configs/calibration/aita-models.yaml",
    )
    assert contract["lifecycle_state"] == "prepared"
    assert contract["run_id"] == "aita-agent-smoke"
    assert contract["identity"]["benchmark_family_id"] == "aita"
    assert contract["identity"]["benchmark_spec"]["scoring_contract"]["primary_outcome_dimensions"] == [
        "verdict_alignment_a",
        "verdict_alignment_b",
        "paired_verdict_alignment",
    ]
    assert "therapeutic_a" in contract["identity"]["benchmark_spec"]["scoring_contract"]["diagnostic_dimensions"]
    assert contract["model_selector"] == "group:calibration_smoke"
    assert contract["agent_profile"] == "default"
    assert contract["identity"]["judge_panel"]["seeker"] == "google/gemini-3-flash-preview"
    assert contract["identity"]["judge_panel"]["flip_generator"] == "google/gemini-3-flash-preview"
    assert {"role": "seeker", "model_id": "google/gemini-3-flash-preview"} in contract["expected_judges"]
    assert {
        "role": "flip_generator",
        "model_id": "google/gemini-3-flash-preview",
    } in contract["expected_judges"]
    assert contract["modules"][0]["module"] == "aita"
    assert contract["modules"][0]["dataset_mode"] == "yta-synthflip"
    assert contract["modules"][0]["dataset_manifest"]["dataset_mode"] == "yta-synthflip"
    assert contract["modules"][0]["dataset_manifest"]["manifest_hash"]
    assert any(
        artifact["kind"] == "rendered_models"
        for artifact in contract["modules"][0]["expected_artifacts"]
    )
    assert [target.model_id for target in collect_targets_from_run_dir(contract_path.parent)]
    assert "manifest_hash" not in contract["identity"]["sample_spec"]["dataset_manifest"]
    assert len(contract["modules"][0]["expected_units"]) == 2
    assert {unit["side"] for unit in contract["modules"][0]["expected_units"]} == {"side_a", "side_b"}
    assert {
        unit["side"]: unit["ground_truth"]
        for unit in contract["modules"][0]["expected_units"]
    } == {"side_a": "YTA", "side_b": "synthetic_reversal"}
    aita_judge_line = next(
        line
        for line in contract["call_plan"]["lines"]
        if line["operation"] == "aita_dimensions"
    )
    assert aita_judge_line["calls"] == {"low": 8, "expected": 8, "high": 8}
    assert contract["modules"][0]["expected_units"][0]["planned_turns"] == 5
    assert contract["modules"][0]["dataset_manifest"]["selected_items"][0]["sides"] == ["side_a", "side_b"]
    assert contract["identity"]["sample_spec"]["items"][0]["sides"] == ["side_a", "side_b"]
    assert "aita_bench run" in contract["execute_command"]
    assert "aita_bench score" in contract["score_command"]
    import sys
    assert contract["execute_argv"][:3] == [sys.executable, "-m", "aita_bench"]
    assert contract["score_argv"][:4] == [sys.executable, "-m", "aita_bench", "score"]
    assert plan["modules"][0]["module"] == "aita"
    assert plan["modules"][0]["score_command"] == contract["score_command"]
    assert plan["modules"][0]["provenance"]["benchmark_condition_hash"]


def test_prepare_aita_model_partitions_preserve_comparable_identity_and_units(tmp_path):
    def prepare(run_id: str, model_selector: str) -> dict:
        path = prepare_aita_run(
            run_id=run_id,
            output_root=tmp_path / run_id,
            suite_config_path=Path("suite_models.yaml"),
            model_selector=model_selector,
            judge_set="frontier",
            items="1",
            dataset_mode="yta-synthflip",
            allow_sample_fallback=True,
        )
        return load_run_contract(path)

    parent = prepare(
        "aita-parent",
        "claude-sonnet-5-native-high,claude-opus-5-native-high",
    )
    sonnet = prepare("aita-sonnet", "claude-sonnet-5-native-high")
    opus = prepare("aita-opus", "claude-opus-5-native-high")

    comparable_hashes = (
        "benchmark_spec_hash",
        "sample_hash",
        "judge_panel_hash",
        "benchmark_condition_hash",
        "comparison_spec_hash",
    )
    for key in comparable_hashes:
        assert parent["provenance"][key] == sonnet["provenance"][key]
        assert parent["provenance"][key] == opus["provenance"][key]

    def unit_ids(contract: dict) -> set[str]:
        return {
            unit["unit_id"]
            for unit in contract["modules"][0]["expected_units"]
        }

    parent_units = unit_ids(parent)
    sonnet_units = unit_ids(sonnet)
    opus_units = unit_ids(opus)
    assert sonnet_units.isdisjoint(opus_units)
    assert sonnet_units | opus_units == parent_units

    parent_conditions = {
        condition["key"]: condition
        for condition in parent["provenance"]["model_condition_hashes"]
    }
    for child in (sonnet, opus):
        condition = child["provenance"]["model_condition_hashes"][0]
        assert condition == parent_conditions[condition["key"]]

    assert len({
        parent["provenance"]["model_conditions_hash"],
        sonnet["provenance"]["model_conditions_hash"],
        opus["provenance"]["model_conditions_hash"],
    }) == 3


def test_prepare_aita_run_records_official_dataset_manifest(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    og_path.write_text(
        "id,original_post\n"
        "alpha,original alpha\n"
        "bad,original bad\n"
        "beta,original beta\n"
    )
    flip_path.write_text(
        "id,flipped_story\n"
        "alpha,flipped alpha\n"
        "bad,ERROR\n"
        "beta,flipped beta\n"
    )
    flip_path.with_suffix(".labels.json").write_text('{"default": "YTA", "labels": {}}\n')

    contract_path = prepare_aita_run(
        run_id="aita-official-manifest",
        output_root=tmp_path / "aita-official-manifest",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="2",
        dataset_mode="nta-paired",
        og_data=str(og_path),
        flip_data=str(flip_path),
        source_command="python -m suite_tools.prepare_run --module aita",
    )

    contract = load_run_contract(contract_path)
    manifest = contract["modules"][0]["dataset_manifest"]

    assert len(contract["modules"][0]["expected_units"]) == 4
    assert manifest["dataset_mode"] == "nta-paired"
    assert manifest["flip_source"] == "official_aita_nta_flip"
    assert manifest["official_pair_count"] == 3
    assert manifest["valid_pair_count"] == 2
    assert manifest["malformed_official_rows"] == [
        {"index": 1, "id": "bad", "fields": ["flipped_story"]}
    ]
    assert [pair["pair_id"] for pair in manifest["selected_pairs"]] == ["alpha", "beta"]
    assert manifest["manifest_hash"]
    assert "manifest_hash" not in contract["identity"]["sample_spec"]["dataset_manifest"]


def test_prepare_aita_run_binds_sealed_pack_without_persisting_unlock_fragment(tmp_path):
    sealed = seal_files(
        {
            "flip.csv": b"id,flipped_story\nsynthetic-pair,synthetic reversal\n",
            "flip.labels.json": b'{"labels":{"synthetic-pair":"YTA"}}\n',
            "og.csv": b"id,original_post\nsynthetic-pair,synthetic original\n",
            "selection.yaml": b"items:\n  - index: 0\n    pair_id: synthetic-pair\n",
        },
        pack_id="synthetic-aita-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    envelope = dict(sealed.envelope, ciphertext_file="synthetic.sealed")
    envelope_path = tmp_path / "synthetic.envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    (tmp_path / "synthetic.sealed").write_bytes(sealed.ciphertext)

    contract_path = prepare_aita_run(
        run_id="aita-sealed-pack",
        output_root=tmp_path / "aita-sealed-pack",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="99",
        dataset_mode="nta-paired",
        sealed_pack=str(envelope_path),
        sealed_pack_key_part_b=sealed.key_part_b,
        source_command="python -m suite_tools.prepare_run --module aita --sealed-pack PACK",
    )

    contract = load_run_contract(contract_path)
    manifest = contract["modules"][0]["dataset_manifest"]
    serialized = contract_path.read_text()
    assert manifest["sealed_pack"]["ciphertext_sha256"] == sealed.envelope["ciphertext_sha256"]
    assert contract["identity"]["sample_spec"]["dataset_manifest"]["sealed_pack"][
        "plaintext_identity_sha256"
    ] == sealed.envelope["plaintext_identity_sha256"]
    assert "--sealed-pack" in contract["execute_command"]
    assert sealed.key_part_b not in serialized
    assert "synthetic original" not in serialized


def test_prepare_aita_run_accepts_fixed_item_selection(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    selection_path = tmp_path / "selection.yaml"
    og_path.write_text(
        "id,original_post\n"
        "alpha,original alpha\n"
        "beta,original beta\n"
        "gamma,original gamma\n"
    )
    flip_path.write_text(
        "id,flipped_story\n"
        "alpha,flipped alpha\n"
        "beta,flipped beta\n"
        "gamma,flipped gamma\n"
    )
    flip_path.with_suffix(".labels.json").write_text('{"default": "YTA", "labels": {}}\n')
    selection_path.write_text(
        "name: aita-test-selection\n"
        "sample_seed: 20260526\n"
        "items:\n"
        "  - index: 2\n"
        "  - index: 0\n"
    )

    contract_path = prepare_aita_run(
        run_id="aita-selection-manifest",
        output_root=tmp_path / "aita-selection-manifest",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="nta-paired",
        og_data=str(og_path),
        flip_data=str(flip_path),
        item_selection=str(selection_path),
        source_command="python -m suite_tools.prepare_run --module aita",
    )

    contract = load_run_contract(contract_path)
    manifest = contract["modules"][0]["dataset_manifest"]
    units = contract["modules"][0]["expected_units"]

    assert contract["modules"][0]["expected_units"][0]["source_pair_hash"]
    assert contract["modules"][0]["expected_units"][0]["side_prompt_hash"]
    assert manifest["selection"]["item_indices"] == [0, 2]
    assert manifest["selection"]["item_selection"]["name"] == "aita-test-selection"
    assert [pair["pair_id"] for pair in manifest["selected_pairs"]] == ["alpha", "gamma"]
    assert contract["execute_command"].count("--item-selection") == 1
    assert {unit["item_idx"] for unit in units} == {0, 2}


def test_prepare_aita_run_rejects_stale_selection_hashes(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    selection_path = tmp_path / "selection.yaml"
    og_path.write_text("id,original_post\nalpha,original alpha\n")
    flip_path.write_text("id,flipped_story\nalpha,flipped alpha\n")
    flip_path.with_suffix(".labels.json").write_text('{"default": "YTA", "labels": {}}\n')
    selection_path.write_text(
        "name: stale-selection\n"
        "items:\n"
        "  - index: 0\n"
        "    pair_id: alpha\n"
        "    source_pair_hash: definitely-not-the-real-hash\n"
    )

    with pytest.raises(ValueError, match="hash metadata does not match"):
        prepare_aita_run(
            run_id="aita-stale-selection",
            output_root=tmp_path / "aita-stale-selection",
            suite_config_path=Path("suite_models.yaml"),
            model_selector="group:calibration_smoke",
            judge_set="calibration",
            items="1",
            dataset_mode="nta-paired",
            og_data=str(og_path),
            flip_data=str(flip_path),
            item_selection=str(selection_path),
            source_command="python -m suite_tools.prepare_run --module aita",
        )


def test_prepare_aita_same_questions_keep_sample_hash_when_selection_file_metadata_changes(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    selection_a = tmp_path / "selection-a.yaml"
    selection_b = tmp_path / "nested" / "selection-b.yaml"
    selection_b.parent.mkdir()
    og_path.write_text(
        "id,original_post\n"
        "alpha,original alpha\n"
        "beta,original beta\n"
    )
    flip_path.write_text(
        "id,flipped_story\n"
        "alpha,flipped alpha\n"
        "beta,flipped beta\n"
    )
    flip_path.with_suffix(".labels.json").write_text('{"default": "YTA", "labels": {}}\n')
    selection_a.write_text(
        "name: first-selection-name\n"
        "sample_seed: 1\n"
        "items:\n"
        "  - index: 0\n"
    )
    selection_b.write_text(
        "name: copied-selection-with-different-metadata\n"
        "sample_seed: 999\n"
        "items:\n"
        "  - index: 0\n"
    )

    path_a = prepare_aita_run(
        run_id="aita-selection-hash-a",
        output_root=tmp_path / "aita-selection-hash-a",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="nta-paired",
        og_data=str(og_path),
        flip_data=str(flip_path),
        item_selection=str(selection_a),
        source_command="python -m suite_tools.prepare_run --module aita",
    )
    path_b = prepare_aita_run(
        run_id="aita-selection-hash-b",
        output_root=tmp_path / "aita-selection-hash-b",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="nta-paired",
        og_data=str(og_path),
        flip_data=str(flip_path),
        item_selection=str(selection_b),
        source_command="python -m suite_tools.prepare_run --module aita",
    )

    plan_a = json.loads((path_a.parents[1] / "RUN_PLAN.json").read_text())
    plan_b = json.loads((path_b.parents[1] / "RUN_PLAN.json").read_text())
    sample_a = plan_a["modules"][0]["provenance"]["sample_hash"]
    sample_b = plan_b["modules"][0]["provenance"]["sample_hash"]
    unit_a = load_run_contract(path_a)["modules"][0]["expected_units"][0]
    unit_b = load_run_contract(path_b)["modules"][0]["expected_units"][0]

    assert sample_a == sample_b
    assert unit_a["source_pair_hash"] == unit_b["source_pair_hash"]


def test_prepare_epis_run_writes_contract_and_score_command(tmp_path):
    selection = Path("epistemic-sycophancy-bench/data/calibration-selection.yaml")
    contract_path = prepare_epis_run(
        run_id="epis-agent-smoke",
        output_root=tmp_path / "epis-agent-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items=1,
        types="delusion,pickside,mirror",
        selection=str(selection.resolve()),
        source_command="python -m suite_tools.prepare_run --module epis",
    )

    contract = load_run_contract(contract_path)
    config_path = tmp_path / "epis-agent-smoke" / "_configs" / "calibration" / "epis-models.yaml"
    plan = json.loads((tmp_path / "epis-agent-smoke" / "RUN_PLAN.json").read_text())
    expected_units = contract["modules"][0]["expected_units"]

    assert contract_path == tmp_path / "epis-agent-smoke" / "epis" / "RUN_CONTRACT.json"
    assert config_path.exists()
    _assert_prepared_config_binding(
        contract,
        config_path,
        "_configs/calibration/epis-models.yaml",
    )
    assert contract["lifecycle_state"] == "prepared"
    assert contract["run_id"] == "epis-agent-smoke"
    # Family id must match the module key so the explicit identity and the
    # fallback derivation (_fallback_benchmark_family_id) can never fork the
    # benchmark_spec_hash. "epis" remains an accepted input alias.
    assert contract["identity"]["benchmark_family_id"] == "epistemic"
    assert contract["identity"]["benchmark_spec"]["scoring_contract"]["primary_outcome_dimensions"] == [
        "primary_failure"
    ]
    assert "stance_amplification" in contract["identity"]["benchmark_spec"]["scoring_contract"]["mechanism_dimensions"]
    assert contract["modules"][0]["module"] == "epistemic"
    assert contract["agent_profile"] == "default"
    assert contract["identity"]["judge_panel"]["seeker"] == "google/gemini-3-flash-preview"
    assert {"role": "seeker", "model_id": "google/gemini-3-flash-preview"} in contract["expected_judges"]
    assert any(
        artifact["kind"] == "rendered_models"
        for artifact in contract["modules"][0]["expected_artifacts"]
    )
    assert [target.model_id for target in collect_targets_from_run_dir(contract_path.parent)]
    assert len(expected_units) == 5
    assert {unit["test_type"] for unit in expected_units} == {"delusion", "pickside", "mirror"}
    assert "epis_bench run" in contract["execute_command"]
    assert "epis_bench score" in contract["score_command"]
    assert "epis_bench report" in contract["score_command"]
    import sys
    assert contract["execute_argv"][:3] == [sys.executable, "-m", "epis_bench"]
    assert contract["score_argv"][:4] == [sys.executable, "-m", "epis_bench", "score"]
    assert len(contract["score_steps"]) == 2
    assert contract["score_steps"][1]["argv"][:4] == [sys.executable, "-m", "epis_bench", "report"]
    assert plan["modules"][0]["module"] == "epis"
    assert plan["modules"][0]["expected_units"] == 5
    assert plan["modules"][0]["provenance"]["comparison_spec_hash"]


def test_prepare_run_cli_output_json_for_agents(tmp_path, capsys, monkeypatch):
    lease_dir = tmp_path / "leases"
    set_paid_call_policy(64, lease_dir=lease_dir)
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "3")
    status = main(
        [
            "--module",
            "epis",
            "--run-id",
            "epis-json-smoke",
            "--output",
            str(tmp_path / "epis-json-smoke"),
            "--models",
            "group:calibration_smoke",
            "--judge-set",
            "calibration",
            "--agent-profile",
            "gemini_35_flash",
            "--items",
            "1",
            "--types",
            "delusion",
            "--selection",
            str(Path("epistemic-sycophancy-bench/data/calibration-selection.yaml").resolve()),
            "--output-json",
            "--non-interactive",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["module"] == "epistemic"
    assert payload["run_id"] == "epis-json-smoke"
    assert payload["agent_profile"] == "gemini_35_flash"
    assert payload["expected_units"] == 1
    assert payload["paid_call_capacity"]["effective_limit"] == 3
    assert payload["paid_call_capacity"]["effective_limit_source"] == (
        "environment:BENCHMARK_PAID_CALL_MAX_ACTIVE"
    )
    assert payload["paid_call_capacity"]["policy_limit"] == 64
    assert payload["contract_path"].endswith("epis-json-smoke/epis/RUN_CONTRACT.json")
    assert "execute_command" in payload
    assert "score_command" in payload


def test_epis_sample_spec_uses_repo_relative_selection(tmp_path, monkeypatch):
    from suite_tools.prepare_run import _repo_relative_or_name
    from suite_tools.suite_registry import REPO_ROOT

    inside = REPO_ROOT / "epistemic-sycophancy-bench" / "data" / "selection.yaml"
    assert _repo_relative_or_name(str(inside)) == "epistemic-sycophancy-bench/data/selection.yaml"

    outside = tmp_path / "selection.yaml"
    outside.write_text("{}")
    assert _repo_relative_or_name(str(outside)) == "selection.yaml"
    assert _repo_relative_or_name(None) is None


def test_sus_expected_units_carry_expected_transcript_path(tmp_path):
    contract_path = prepare_sus_run(
        run_id="sus-enrich",
        output_root=tmp_path / "sus-enrich",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
        source_command="python -m suite_tools.prepare_run --module sus",
    )
    contract = load_run_contract(contract_path)
    units = contract["modules"][0]["expected_units"]
    assert units
    for unit in units:
        assert unit["expected_transcript_path"].startswith("transcripts/")
        assert unit["expected_transcript_path"].endswith(".json")
        assert f"run{unit['run_number']}" in unit["expected_transcript_path"]


def test_sus_expected_transcript_path_includes_condition_request_options(tmp_path):
    contract_path = prepare_sus_run(
        run_id="sus-effort-path",
        output_root=tmp_path / "sus-effort-path",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="claude-sonnet-5-native-low-128k",
        judge_set="frontier",
        scenarios_selector="bridge_heights",
        runs=1,
    )

    contract = load_run_contract(contract_path)
    unit = contract["modules"][0]["expected_units"][0]

    assert "options" in unit["expected_transcript_path"]
    assert "effort-low" in unit["expected_transcript_path"]


def test_epis_expected_units_carry_stable_item_hash(tmp_path):
    selection = Path("epistemic-sycophancy-bench/data/calibration-selection.yaml")
    contract_path = prepare_epis_run(
        run_id="epis-enrich",
        output_root=tmp_path / "epis-enrich",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items=1,
        types="delusion,pickside,mirror",
        selection=str(selection.resolve()),
        source_command="python -m suite_tools.prepare_run --module epis",
    )
    contract = load_run_contract(contract_path)
    units = contract["modules"][0]["expected_units"]
    assert units
    for unit in units:
        assert isinstance(unit["item_hash"], str) and len(unit["item_hash"]) >= 16
    by_item = {}
    for unit in units:
        by_item.setdefault((unit["test_type"], unit["item_idx"]), set()).add(unit["item_hash"])
    for hashes in by_item.values():
        assert len(hashes) == 1  # same sample-axis item → one content hash across model_keys
