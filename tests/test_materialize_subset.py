import json

from suite_tools.materialize_subset import materialize
from suite_tools.owed_units import owed_units
from suite_tools.run_contract import provenance_hashes


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_materialize_filters_models_and_copies_selected_artifacts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    models = [
        {"key": "keep", "model_id": "provider/keep", "condition_id": "keep"},
        {"key": "drop", "model_id": "provider/drop", "condition_id": "drop"},
    ]
    units = []
    for model in models:
        key = model["key"]
        for side in ("side_a", "side_b"):
            transcript = f"{key}_item0_{side}.json"
            score = f"{key}_item0_scores.json"
            units.append(
                {
                    "unit_id": f"aita:{key}:item0:{side}",
                    "model_key": key,
                    "model_id": model["model_id"],
                    "item_idx": 0,
                    "side": side,
                    "planned_turns": 1,
                    "expected_transcript_path": transcript,
                    "expected_score_path": score,
                }
            )
            _write_json(
                source / transcript,
                {"completed": True, "turns": [{"model_response": "ok"}]},
            )
        _write_json(
            source / f"{key}_item0_scores.json",
            {"model": key, "item_idx": 0, "verdict_alignment_a": 1, "verdict_alignment_b": 1},
        )

    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "source",
        "expected_models": models,
        "expected_judges": [],
        "identity": {
            "benchmark_family_id": "aita",
            "benchmark_spec": {"module": "aita"},
            "sample_spec": {"n_items": 1},
            "judge_panel": {},
            "model_conditions": models,
            "execution": {},
        },
        "modules": [{"module": "aita", "expected_units": units}],
    }
    contract["provenance"] = provenance_hashes(contract)
    _write_json(source / "RUN_CONTRACT.json", contract)
    _write_json(
        source / "RUN_STATUS.json",
        {"status": "completed", "validity": "score_ready"},
    )
    _write_json(
        source / "FINAL_RESULTS.json",
        {
            "metadata": {"models": ["keep", "drop"], "missing_scores": []},
            "scores": {
                "keep_item0": {"model": "keep", "item_idx": 0},
                "drop_item0": {"model": "drop", "item_idx": 0},
            },
        },
    )
    source_contract_before = (source / "RUN_CONTRACT.json").read_bytes()

    output = materialize(
        source_run_dir=source,
        output_dir=tmp_path / "derived",
        run_id="derived",
        excluded_model_keys={"drop"},
        reason="publication exclusion",
    )

    derived_contract = json.loads((output / "RUN_CONTRACT.json").read_text())
    assert [model["key"] for model in derived_contract["expected_models"]] == ["keep"]
    assert {unit["model_key"] for unit in derived_contract["modules"][0]["expected_units"]} == {"keep"}
    assert derived_contract["derived_from"]["excluded_model_keys"] == ["drop"]
    assert derived_contract["provenance"] == provenance_hashes(derived_contract)
    assert (output / "keep_item0_side_a.json").read_bytes() == (
        source / "keep_item0_side_a.json"
    ).read_bytes()
    assert not (output / "drop_item0_side_a.json").exists()
    final = json.loads((output / "FINAL_RESULTS.json").read_text())
    assert final["metadata"]["models"] == ["keep"]
    assert set(final["scores"]) == {"keep_item0"}
    states = owed_units(output, module="aita")
    assert states["counts"]["done"] == 2
    assert states["counts"]["owed"] == 0
    assert (source / "RUN_CONTRACT.json").read_bytes() == source_contract_before
