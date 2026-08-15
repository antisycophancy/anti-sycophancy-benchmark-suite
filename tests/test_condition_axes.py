import json
from pathlib import Path

from suite_tools.condition_axes import condition_axes, load_model_aliases

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_axes_from_native_condition():
    axes = condition_axes(_fixture("identity_native_condition.json"), aliases={})
    assert axes == {
        "canonical_model": "gpt-5.6-luna",
        "route": "openai_responses",
        "effort": "high",
        "profile": None,
    }


def test_openrouter_alias_resolves_to_same_canonical_model():
    axes = condition_axes(
        _fixture("identity_openrouter_condition.json"),
        aliases={"openai/gpt-5.6-luna": "gpt-5.6-luna"},
    )
    assert axes["canonical_model"] == "gpt-5.6-luna"
    assert axes["route"] == "openrouter"


def test_full_profile_axis_with_lineage():
    axes = condition_axes(_fixture("identity_openrouter_condition.json"), aliases={})
    assert axes["profile"] == {
        "profile_id": "th-prompts-v3",
        "profile_hash": "f00f00",
        "parent_profile_id": "th-prompts-v2",
    }


def test_load_model_aliases_reads_yaml(tmp_path):
    p = tmp_path / "suite_models.yaml"
    p.write_text("model_aliases:\n  'openai/gpt-5.6-luna': gpt-5.6-luna\n")
    assert load_model_aliases(p) == {"openai/gpt-5.6-luna": "gpt-5.6-luna"}


def test_missing_aliases_section_yields_empty_map(tmp_path):
    p = tmp_path / "suite_models.yaml"
    p.write_text("models: {}\n")
    assert load_model_aliases(p) == {}
