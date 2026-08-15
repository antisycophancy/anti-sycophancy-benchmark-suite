import json

import pytest

from sus_bench.report import write_html, write_json
from sus_bench.stats import aggregate_runs


def test_write_json_preserves_provider_refusal_metadata(tmp_path):
    out = tmp_path / "results.json"
    write_json(
        [
            {
                "model": "claude-fable-5",
                "label": "Claude Fable 5",
                "scenario": "bridge_heights",
                "score": None,
                "post_analysis": None,
                "conversation": [
                    {"role": "user", "content": "first prompt"},
                ],
                "turn_outcomes": [
                    {
                        "type": "provider_refusal",
                        "stop_reason": "refusal",
                        "timestamp": "2026-07-14T12:00:00+00:00",
                    }
                ],
                "phases": {
                    "elicit": {
                        "prompt": "first prompt",
                        "provider_refusal": True,
                        "stop_reason": "refusal",
                    }
                },
                "score_state": "excluded_provider_refusal",
                "exclusion_reason": "provider_refusal",
                "provider_refusal": {
                    "phase": "elicit",
                    "stop_reason": "refusal",
                    "stop_details": {"category": "reasoning_extraction"},
                },
            }
        ],
        [],
        out,
    )

    conversations = json.loads(out.with_name("results-conversations.json").read_text())

    assert conversations[0]["score_state"] == "excluded_provider_refusal"
    assert conversations[0]["exclusion_reason"] == "provider_refusal"
    assert conversations[0]["provider_refusal"]["phase"] == "elicit"
    assert conversations[0]["turn_outcomes"][0]["type"] == "provider_refusal"


def test_write_json_preserves_complete_condition_identity(tmp_path):
    out = tmp_path / "results.json"
    identity = {
        "provider_api": "openai_compatible",
        "condition_id": "condition-1",
        "condition_hash": "sha256:condition-1",
        "profile_id": "profile-1",
        "served_profile_id": "served-1",
        "served_profile_hash": "sha256:served-1",
        "provider_condition_id": "provider-condition-1",
        "provider_condition_hash": "sha256:provider-condition-1",
        "provider_version": "provider-v1",
        "condition_metadata": {"effort": "high"},
        "request_options": {"max_tokens": 128000},
    }
    write_json(
        [{
            "model": "test/model",
            "label": "Test Model",
            "scenario": "bridge_heights",
            "score": None,
            "post_analysis": None,
            "conversation": [],
            "turn_outcomes": [],
            "phases": {},
            **identity,
        }],
        [],
        out,
    )

    saved = json.loads(out.with_name("results-conversations.json").read_text())[0]
    for field, value in identity.items():
        assert saved[field] == value


@pytest.mark.parametrize("prefix", ["[TIMEOUT", "[ERROR", "[Provider refusal"])
def test_write_json_rejects_synthetic_assistant_content_in_scored_transcripts(tmp_path, prefix):
    with pytest.raises(ValueError, match="synthetic assistant content"):
        write_json(
            [
                {
                    "model": "test/model",
                    "label": "Test",
                    "scenario": "bridge_heights",
                    "score_state": "scored",
                    "score": {"sus": 50},
                    "conversation": [
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": f"{prefix}: harness text]"},
                    ],
                    "phases": {},
                }
            ],
            [],
            tmp_path / "results.json",
        )


def test_report_excludes_provider_refusals_from_cap_rate():
    results = [
        {
            "model": "claude-fable-5",
            "label": "Claude Fable 5",
            "scenario": "bridge_heights",
            "scenario_name": "Bridge Heights",
            "score": {"sus": 10, "irq": 0, "pr": 0, "er": 0, "ca": 0},
            "phases": {"elicit": {"gave_compliance_info": True}},
        },
        {
            "model": "claude-fable-5",
            "label": "Claude Fable 5",
            "scenario": "bridge_heights",
            "scenario_name": "Bridge Heights",
            "score": {"sus": 20, "irq": 0, "pr": 0, "er": 0, "ca": 0},
            "phases": {"escalate_1": {"gave_compliance_info": True}},
        },
        {
            "model": "claude-fable-5",
            "label": "Claude Fable 5",
            "scenario": "bridge_heights",
            "scenario_name": "Bridge Heights",
            "score": {"sus": 0, "irq": 0, "pr": 0, "er": 0, "ca": 0},
            "phases": {"elicit": {"gave_compliance_info": False}},
        },
        {
            "model": "claude-fable-5",
            "label": "Claude Fable 5",
            "scenario": "bridge_heights",
            "scenario_name": "Bridge Heights",
            "score_state": "excluded_provider_refusal",
            "exclusion_reason": "provider_refusal",
            "phases": {"elicit": {"gave_compliance_info": True}},
        },
    ]

    block = aggregate_runs(results)[0]

    assert block["runs"] == 3
    assert block["cap_rate"] == round(2 / 3, 3)
    assert block["excluded_provider_refusal_count"] == 1


def test_write_html_escapes_untrusted_result_content_and_sets_csp(tmp_path):
    payload = '<script>alert(1)</script><img src=x onerror="alert(2)">'
    aggregated = [
        {
            "label": payload,
            "scenario_name": payload,
            "runs": 1,
            "sus_mean": 10.0,
            "cap_rate": 0.0,
            "irq_mean": 0.0,
            "pr_mean": 0.0,
            "er_mean": 0.0,
            "ca_mean": 0.0,
            "cap_rate_wilson_95_ci_low": 0.0,
            "cap_rate_wilson_95_ci_high": 0.0,
            "capitulation_phase": payload,
            "individual_results": [
                {
                    "run_number": 1,
                    "capitulated": False,
                    "score": {"sus": 10},
                    "post_analysis": {
                        "compliance_analysis": payload,
                        "safety_theater_indicators": [payload],
                        "num_judges": 1,
                        "judge_scores": [
                            {
                                "judge": payload,
                                "irq": 0,
                                "pr": 0,
                                "er": 0,
                                "ca": 0,
                                "irq_explanation": payload,
                            }
                        ],
                    },
                    "conversation": [
                        {"role": payload, "content": payload},
                    ],
                }
            ],
        }
    ]
    out = tmp_path / "dashboard.html"

    write_html(aggregated, out)

    html = out.read_text()
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'http-equiv="Content-Security-Policy"' in html
    assert "default-src 'none'" in html
