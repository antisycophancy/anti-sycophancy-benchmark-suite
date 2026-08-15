import json

from suite_tools.capacity_intent import (
    CAPACITY_INTENT_FILENAME,
    CAPACITY_INTENT_SCHEMA_VERSION,
    build_capacity_intent,
    estimate_contract_capacity,
    main,
)


def _contract():
    return {
        "run_id": "capacity-smoke",
        "expected_models": [
            {
                "key": "private-fast",
                "model_id": "private-endpoint/fast",
                "endpoint": "local_openai_compatible",
                "condition_hash": "sha256:condition",
                "served_profile_hash": "sha256:served",
                "provider_condition_hash": "sha256:provider-condition",
                "condition_metadata": {"adapter_family": "private"},
            },
            {
                "key": "public-flash",
                "model_id": "google/gemini-3-flash-preview",
                "endpoint": "openrouter",
            },
        ],
        "modules": [
            {
                "module": "sus",
                "expected_units": [
                    {
                        "unit_id": "sus:private-fast:bridge:0",
                        "model_key": "private-fast",
                        "model_id": "private-endpoint/fast",
                        "planned_turns": 5,
                    },
                    {
                        "unit_id": "sus:public-flash:bridge:0",
                        "model_key": "public-flash",
                        "model_id": "google/gemini-3-flash-preview",
                        "planned_turns": 5,
                    },
                ],
            },
            {
                "module": "epis",
                "expected_units": [
                    {
                        "unit_id": "epis:private-fast:delusion:0",
                        "model_key": "private-fast",
                        "model_id": "private-endpoint/fast",
                    }
                ],
            },
        ],
    }


def test_capacity_intent_matches_private_endpoint_and_preserves_hash_summaries():
    intent = build_capacity_intent(
        _contract(),
        contract_path="/tmp/RUN_CONTRACT.json",
        profile={
            "name": "private-api",
            "model_id_prefixes": ["private-endpoint/"],
            "default_turns_per_unit": 4,
            "provider_calls_per_turn": 3,
            "default_max_active_calls": 20,
            "calls_per_capacity_unit": 2,
            "min_capacity_units": 1,
            "max_capacity_units": 3,
        },
    )

    assert intent["schema_version"] == CAPACITY_INTENT_SCHEMA_VERSION
    assert intent["side_effects"] == "none"
    assert intent["contract_invariance"]["modifies_prompts"] is False
    assert intent["contract_invariance"]["modifies_scoring"] is False
    assert intent["estimate"]["matching_units"] == 2
    assert intent["estimate"]["planned_turns"] == 9
    assert intent["estimate"]["estimated_provider_calls"] == 27
    assert intent["capacity"]["max_active_calls"] == 2
    assert intent["capacity"]["target_capacity_units"] == 1
    assert intent["estimate"]["modules"] == [
        {"module": "sus", "total_units": 2, "matching_units": 1, "planned_turns": 5},
        {"module": "epis", "total_units": 1, "matching_units": 1, "planned_turns": 4},
    ]
    assert intent["estimate"]["matching_model_conditions"] == [
        {
            "key": "private-fast",
            "model_id": "private-endpoint/fast",
            "endpoint": "local_openai_compatible",
            "condition_hash": "sha256:condition",
            "served_profile_hash": "sha256:served",
            "provider_condition_hash": "sha256:provider-condition",
            "condition_metadata": {"adapter_family": "private"},
        }
    ]


def test_capacity_estimate_can_match_by_endpoint_name_or_substring():
    by_name = estimate_contract_capacity(
        _contract(),
        {"endpoint_names": ["local_openai_compatible"], "provider_calls_per_turn": 2},
    )
    by_substring = estimate_contract_capacity(
        _contract(),
        {"endpoint_contains": ["openrouter"], "provider_calls_per_turn": 2},
    )

    assert by_name["matching_model_keys"] == ["private-fast"]
    assert by_name["matching_units"] == 2
    assert by_substring["matching_model_keys"] == ["public-flash"]
    assert by_substring["matching_units"] == 1


def test_capacity_intent_defaults_to_all_models_when_no_match_rules_are_configured():
    intent = build_capacity_intent(
        _contract(),
        profile={"default_max_active_calls": 2, "calls_per_capacity_unit": 2},
    )

    assert intent["match"]["default_is_all_models"] is True
    assert intent["estimate"]["matching_units"] == 3
    assert intent["capacity"]["max_active_calls"] == 2
    assert intent["capacity"]["target_capacity_units"] == 1


def test_capacity_intent_cli_writes_side_effect_free_intent(tmp_path, capsys):
    run_dir = tmp_path / "prepared" / "sus"
    run_dir.mkdir(parents=True)
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(_contract()))

    status = main(
        [
            "--contract",
            str(run_dir),
            "--match-model-prefix",
            "private-endpoint/",
            "--provider-calls-per-turn",
            "3",
            "--calls-per-capacity-unit",
            "2",
            "--json",
        ]
    )

    out = capsys.readouterr().out
    intent_path = run_dir / CAPACITY_INTENT_FILENAME
    written = json.loads(intent_path.read_text())

    assert status == 0
    assert intent_path.exists()
    assert json.loads(out)["estimate"]["matching_units"] == 2
    assert written["contract_path"] == str((run_dir / "RUN_CONTRACT.json").resolve())
    assert written["side_effects"] == "none"
    assert written["contract_invariance"]["modifies_run_contract"] is False
