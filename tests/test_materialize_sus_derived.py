import json
from pathlib import Path

from suite_tools.materialize_sus_derived import materialize
from suite_tools.owed_units import owed_units
from suite_tools.run_contract import (
    build_provenance_identity,
    provenance_hashes,
    write_run_contract,
)
from suite_tools.score_rows import score_rows


def _contract(path: Path, *, panel: list[str]) -> Path:
    model_condition = {
        "key": "model-low",
        "label": "Model low",
        "model_id": "model",
        "condition_id": "model-low",
        "condition_hash": "sha256:model-low",
        "condition_metadata": {"effort": "low"},
    }
    identity = build_provenance_identity(
        benchmark_family_id="sus",
        benchmark_spec={"module": "sus", "version": "v1"},
        sample_spec={"scenario_ids": ["bridge"], "runs": 2},
        judge_panel={"analyzer": "analyzer", "panel": panel},
        model_conditions=[model_condition],
        execution={"run_id": path.parent.name},
    )
    return write_run_contract(path.parent, {
        "run_id": path.parent.name,
        "identity": identity,
        "expected_models": [model_condition],
        "expected_judges": [
            {"role": "analyzer", "model_id": "analyzer"},
            *[{"role": "panel", "model_id": model} for model in panel],
        ],
        "modules": [{
            "module": "sus",
            "scenarios": ["bridge"],
            "runs": 2,
            "expected_units": [
                {"unit_id": f"sus:model-low:bridge:run{run}", "model_key": "model-low",
                 "scenario": "bridge", "run_number": run, "planned_escalations": 0}
                for run in (1, 2)
            ],
        }],
    })


def _scored(run_number: int) -> dict:
    return {
        "model": "model",
        "label": "Model low",
        "condition_id": "model-low",
        "condition_metadata": {"effort": "low"},
        "scenario": "bridge",
        "run_number": run_number,
        "phases": {"elicit": {"response": "answer"}},
        "score": {"sus": 10, "target_utility": 1},
        "rescore_metadata": {
            "analyzer_model": "analyzer",
            "judge_panel": ["judge-a", "judge-b"],
            "source_files": ["/Users/operator/private/source.json"],
        },
    }


def _terminal(run_number: int) -> dict:
    return {
        "model": "model",
        "label": "Model low",
        "condition_id": "model-low",
        "scenario": "bridge",
        "run_number": run_number,
        "score_state": "excluded_provider_refusal",
        "failure_reason": "provider refusal",
    }


def test_materialize_builds_score_ready_derived_run_without_mutating_sources(tmp_path):
    source_contract = _contract(tmp_path / "source" / "RUN_CONTRACT.json", panel=["old-judge"])
    template_contract = _contract(
        tmp_path / "template" / "RUN_CONTRACT.json", panel=["judge-a", "judge-b"]
    )
    score_summary = tmp_path / "score-summary.json"
    score_sidecar = tmp_path / "score-sidecar.json"
    source_conversations = tmp_path / "source-conversations.json"
    score_summary.write_text(json.dumps({"version": "v1", "cost": {"total": 1}}))
    score_sidecar.write_text(json.dumps([_scored(1)]))
    source_conversations.write_text(json.dumps([_scored(1), _terminal(2)]))
    original_source = source_contract.read_bytes()

    output = materialize(
        model_source_contract=source_contract,
        benchmark_template_contract=template_contract,
        score_summary=score_summary,
        score_sidecar=score_sidecar,
        source_conversations=source_conversations,
        output_dir=tmp_path / "derived",
        run_id="derived",
    )

    assert source_contract.read_bytes() == original_source
    contract = json.loads((output / "RUN_CONTRACT.json").read_text())
    status = json.loads((output / "RUN_STATUS.json").read_text())
    assert contract["lifecycle_state"] == "derived_complete"
    assert contract["derived_from"]["source_hashes"]["score_sidecar"]
    assert provenance_hashes(contract)["judge_panel_hash"] == provenance_hashes(
        json.loads(template_contract.read_text())
    )["judge_panel_hash"]
    assert status["status"] == "completed" and status["validity"] == "score_ready"

    owed = owed_units(output)
    assert owed["counts"] == {"done": 1, "terminal_model_signal": 1, "owed": 0}
    rows = score_rows(output)
    assert rows["units"][0]["outcome_class"] == "scored"
    assert rows["units"][1]["outcome_class"] == "terminal_model_signal"
    assert any(row["dimension"] == "cap_outcome" for row in rows["rows"])

    public_score_text = (output / "FINAL_RESULTS-conversations.json").read_text()
    assert "/Users/" not in public_score_text
    assert "source_artifact_sha256" in public_score_text
    selected = json.loads(public_score_text)
    assert all(row["condition_hash"] == "sha256:model-low" for row in selected)
    assert contract["derived_from"]["identity_normalization"]["restored_row_count"] == 2
    assert contract["derived_from"]["identity_normalization"]["restored_field_counts"][
        "condition_hash"
    ] == 2


def test_materialize_accepts_legacy_condition_without_condition_id(tmp_path):
    source_contract = _contract(tmp_path / "source" / "RUN_CONTRACT.json", panel=["judge-a"])
    template_contract = _contract(tmp_path / "template" / "RUN_CONTRACT.json", panel=["judge-a"])
    for contract_path in (source_contract, template_contract):
        contract = json.loads(contract_path.read_text())
        for condition in contract["identity"]["model_conditions"]:
            condition.pop("condition_id", None)
            condition["key"] = "Legacy Model"
            condition["label"] = "Legacy Model"
        for condition in contract["expected_models"]:
            condition.pop("condition_id", None)
            condition["key"] = "Legacy Model"
            condition["label"] = "Legacy Model"
        for unit in contract["modules"][0]["expected_units"]:
            unit["model_key"] = "Legacy Model"
            unit["unit_id"] = unit["unit_id"].replace("model-low", "deadbeef1234")
        contract_path.write_text(json.dumps(contract))

    rows = []
    for run_number in (1, 2):
        row = _scored(run_number)
        row.pop("condition_id")
        row["label"] = "Legacy Model"
        row["rescore_metadata"]["judge_panel"] = ["judge-a"]
        rows.append(row)
    score_summary = tmp_path / "score-summary.json"
    score_sidecar = tmp_path / "score-sidecar.json"
    score_summary.write_text(json.dumps({"version": "v1"}))
    score_sidecar.write_text(json.dumps(rows))

    output = materialize(
        model_source_contract=source_contract,
        benchmark_template_contract=template_contract,
        score_summary=score_summary,
        score_sidecar=score_sidecar,
        output_dir=tmp_path / "derived",
        run_id="derived",
    )

    assert owed_units(output)["counts"] == {
        "done": 2,
        "terminal_model_signal": 0,
        "owed": 0,
    }
