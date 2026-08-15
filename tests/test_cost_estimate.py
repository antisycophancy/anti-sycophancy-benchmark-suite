import json

import pytest

from suite_tools.cost_estimate import build_contract_call_plan, estimate_call_plan, main


def test_aita_call_plan_splits_generation_support_and_judging():
    contract = {
        "expected_models": [
            {
                "key": "flash-lite",
                "model_id": "google/gemini-3.1-flash-lite",
                "condition_metadata": {"provider_route": "google_direct"},
            }
        ],
        "expected_judges": [
            {"role": "panel", "model_id": "judge/one"},
            {"role": "panel", "model_id": "judge/two"},
            {"role": "seeker", "model_id": "support/seeker"},
            {"role": "flip_generator", "model_id": "support/flip"},
        ],
        "modules": [
            {
                "module": "aita",
                "dataset_mode": "yta-synthflip",
                "expected_units": [
                    {"model_key": "flash-lite", "model_id": "google/gemini-3.1-flash-lite", "item_idx": 1, "side": "side_a", "planned_turns": 5, "ground_truth": "YTA"},
                    {"model_key": "flash-lite", "model_id": "google/gemini-3.1-flash-lite", "item_idx": 1, "side": "side_b", "planned_turns": 5, "ground_truth": "NTA"},
                ],
            }
        ],
    }

    plan = build_contract_call_plan(contract)
    by_role_model = {(line["role"], line["model"]): line for line in plan["lines"]}

    assert by_role_model[("model_under_test", "google/gemini-3.1-flash-lite")]["calls"]["expected"] == 10
    assert by_role_model[("support", "support/seeker")]["calls"]["expected"] == 8
    assert by_role_model[("judge", "judge/one")]["calls"]["expected"] == 9
    assert by_role_model[("judge", "judge/two")]["calls"]["expected"] == 9
    assert by_role_model[("support", "support/flip")]["calls"]["expected"] == 1
    assert by_role_model[("model_under_test", "google/gemini-3.1-flash-lite")]["provider"] == "google_direct"


def test_aita_judge_calls_follow_selected_sides_and_available_labels():
    contract = {
        "expected_models": [{"key": "m", "model_id": "model/a"}],
        "expected_judges": [{"role": "panel", "model_id": "judge/one"}],
        "modules": [
            {
                "module": "aita",
                "expected_units": [
                    {
                        "model_key": "m",
                        "model_id": "model/a",
                        "item_idx": 1,
                        "side": "side_a",
                        "planned_turns": 5,
                        "ground_truth": "YTA",
                    },
                    {
                        "model_key": "m",
                        "model_id": "model/a",
                        "item_idx": 2,
                        "side": "side_a",
                        "planned_turns": 1,
                    },
                ],
            }
        ],
    }

    plan = build_contract_call_plan(contract)
    judge_line = next(line for line in plan["lines"] if line["role"] == "judge")

    # Item 1: outcome + therapeutic + resistance + verdict alignment.
    # Item 2: outcome + therapeutic only. No two-sided consistency calls.
    assert judge_line["calls"] == {"low": 6, "expected": 6, "high": 6}


def test_sus_call_plan_includes_inline_compliance_judging():
    contract = {
        "expected_models": [{"key": "m", "model_id": "model/a"}],
        "expected_judges": [
            {"role": "analyzer", "model_id": "judge/compliance"},
            {"role": "panel", "model_id": "judge/panel"},
        ],
        "modules": [
            {
                "module": "sus",
                "expected_units": [
                    {"model_key": "m", "model_id": "model/a", "scenario": "s1", "planned_escalations": 2, "escalation_mode": "adaptive"},
                    {"model_key": "m", "model_id": "model/a", "scenario": "s2", "planned_escalations": 2, "escalation_mode": "adaptive"},
                ],
            }
        ],
    }

    plan = build_contract_call_plan(contract)
    by_operation = {line["operation"]: line for line in plan["lines"]}

    assert by_operation["sus_conversation"]["calls"] == {
        "low": 2,
        "expected": 6,
        "high": 16,
    }
    assert by_operation["sus_conversation"]["retry_contingency"] == {
        "scope": "high",
        "retries_per_turn": 1,
    }
    assert by_operation["sus_compliance"]["model"] == "judge/compliance"
    assert by_operation["sus_compliance"]["calls"] == {
        "low": 2,
        "expected": 6,
        "high": 8,
    }
    assert by_operation["sus_analysis"]["calls"] == {
        "low": 0,
        "expected": 6,
        "high": 8,
    }
    assert by_operation["sus_post_analysis"]["calls"]["expected"] == 2


def test_multiturn_call_plan_accounts_for_recursive_input_growth():
    contract = {
        "expected_models": [{"key": "m", "model_id": "model/a"}],
        "expected_judges": [],
        "modules": [
            {
                "module": "aita",
                "expected_units": [
                    {
                        "model_key": "m",
                        "model_id": "model/a",
                        "item_idx": 1,
                        "side": "side_a",
                        "planned_turns": 3,
                    }
                ],
            }
        ],
    }
    profiles = {
        "model_under_test": {
            "input": 100,
            "output": 20,
            "input_growth": 50,
        }
    }

    plan = build_contract_call_plan(contract, token_profiles=profiles)
    line = next(line for line in plan["lines"] if line["role"] == "model_under_test")

    assert line["calls"]["expected"] == 3
    assert line["input_tokens_total"]["expected"] == 450
    assert line["output_tokens_total"]["expected"] == 60
    estimate = estimate_call_plan(
        plan,
        {
            "units": "per_token",
            "models": {"model/a": {"input": "0.001", "output": "0"}},
        },
    )
    assert estimate["total_cost_usd"]["expected"] == 0.45


def test_estimate_call_plan_reports_range_and_unknown_pricing():
    call_plan = {
        "lines": [
            {
                "stage": "generation",
                "role": "model_under_test",
                "model": "model/a",
                "provider": "openrouter",
                "calls": {"low": 2, "expected": 2, "high": 2},
                "input_tokens_per_call": {"low": 100, "expected": 100, "high": 100},
                "output_tokens_per_call": {"low": 50, "expected": 50, "high": 50},
            },
            {
                "stage": "scoring",
                "role": "judge",
                "model": "judge/a",
                "provider": "google_direct",
                "calls": {"low": 1, "expected": 1, "high": 1},
                "input_tokens_per_call": {"low": 200, "expected": 200, "high": 200},
                "output_tokens_per_call": {"low": 100, "expected": 100, "high": 100},
            },
            {
                "stage": "generation",
                "role": "support",
                "model": "unknown/model",
                "provider": "openrouter",
                "calls": {"low": 1, "expected": 1, "high": 1},
                "input_tokens_per_call": {"low": 10, "expected": 10, "high": 10},
                "output_tokens_per_call": {"low": 10, "expected": 10, "high": 10},
            },
        ]
    }
    pricing = {
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "models": {
            "model/a": {"prompt": "0.000001", "completion": "0.000002", "source": "openrouter_catalog"},
            "judge/a": {"input": 0.000002, "output": 0.000004, "source": "google_direct_snapshot"},
        },
    }

    estimate = estimate_call_plan(call_plan, pricing)

    assert estimate["state"] == "partial"
    assert estimate["total_cost_usd"]["expected"] == 0.0012
    assert estimate["cost_by_stage"]["generation"]["expected"] == 0.0004
    assert estimate["cost_by_stage"]["scoring"]["expected"] == 0.0008
    assert estimate["unknown_pricing"] == ["unknown/model"]


def test_estimate_call_plan_requires_explicit_supported_price_units():
    call_plan = {"lines": []}

    with pytest.raises(ValueError, match="pricing snapshot units"):
        estimate_call_plan(call_plan, {"models": {}})

    with pytest.raises(ValueError, match="pricing snapshot units"):
        estimate_call_plan(call_plan, {"units": "per_request", "models": {}})


@pytest.mark.parametrize("bad_price", ["-1", "NaN", "Infinity", "-Infinity"])
def test_estimate_call_plan_rejects_negative_or_nonfinite_pricing(bad_price):
    call_plan = {
        "lines": [{
            "stage": "generation",
            "role": "model_under_test",
            "model": "model/a",
            "provider": "openrouter",
            "calls": {"low": 1, "expected": 1, "high": 1},
            "input_tokens_per_call": {"low": 1, "expected": 1, "high": 1},
            "output_tokens_per_call": {"low": 1, "expected": 1, "high": 1},
        }]
    }

    with pytest.raises(ValueError, match="non-negative finite"):
        estimate_call_plan(
            call_plan,
            {
                "units": "per_token",
                "models": {"model/a": {"prompt": bad_price, "completion": "0.1"}},
            },
        )


def test_estimate_call_plan_converts_per_million_token_prices():
    call_plan = {
        "lines": [
            {
                "stage": "generation",
                "role": "model_under_test",
                "model": "model/a",
                "provider": "google_direct",
                "calls": {"low": 1, "expected": 1, "high": 1},
                "input_tokens_per_call": {
                    "low": 1_000_000,
                    "expected": 1_000_000,
                    "high": 1_000_000,
                },
                "output_tokens_per_call": {"low": 0, "expected": 0, "high": 0},
            }
        ]
    }
    snapshot = {
        "units": "per_million_tokens",
        "models": {"model/a": {"input": 2, "output": 4}},
    }

    estimate = estimate_call_plan(call_plan, snapshot)

    assert estimate["pricing_units"] == "per_million_tokens"
    assert estimate["total_cost_usd"]["expected"] == 2.0


def test_cost_estimate_cli_reads_contract_and_pricing_snapshot(tmp_path, capsys):
    contract_path = tmp_path / "RUN_CONTRACT.json"
    pricing_path = tmp_path / "PRICING.json"
    contract_path.write_text(json.dumps({
        "expected_models": [{"key": "m", "model_id": "model/a"}],
        "expected_judges": [],
        "modules": [{"module": "aita", "expected_units": [{"model_key": "m", "planned_turns": 1}]}],
    }))
    pricing_path.write_text(json.dumps({
        "schema_version": "benchmark-pricing-snapshot-v1",
        "units": "per_token",
        "models": {
            "model/a": {"prompt": "0.000001", "completion": "0.000002"}
        },
    }))

    assert main(["--contract", str(contract_path), "--pricing-snapshot", str(pricing_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["call_plan"]["total_calls"]["expected"] == 1
    assert payload["estimate"]["state"] == "estimated"
