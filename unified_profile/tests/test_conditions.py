"""Adversarial condition and observation accounting for read-only reports."""

import json

import pytest

from unified_profile.adapters import load_aita_results, load_epis_results, load_sus_results
from unified_profile.cli import main
from unified_profile.export import _coverage_markdown
from unified_profile.profile import build_all_profiles
from unified_profile.report import generate_unified_report


LOADERS = {"sus": load_sus_results, "aita": load_aita_results, "epis": load_epis_results}


def _condition(effort="high"):
    return {
        "condition_id": f"test-{effort}",
        "condition_hash": f"hash-{effort}",
        "route_hash": "route-one",
        "provider_api": "openai_compatible",
        "request_options": {"reasoning": {"effort": effort}},
    }


def _write(root, module, idx=0, condition=None, alias="m", name=None):
    root.mkdir(parents=True, exist_ok=True)
    row = {"model": alias, "model_id": "m", **(condition or {})}
    if module == "sus":
        row.update(run_number=idx + 1, scenario="test", score={"sus": idx * 100}, phases={})
        payload = [row]
        default_name = f"m-{idx}-conversations.json"
    elif module == "aita":
        row.update(item_idx=idx, outcome_a=2, outcome_b=2, consistency=1)
        payload = row
        default_name = f"m_item{idx}_scores.json"
    else:
        row.update(item_idx=idx, test_type="mirror", persistence=1, integrity=2, consistency=1, amplification=0)
        payload = row
        default_name = f"m_item{idx}_mirror_scores.json"
    path = root / (name or default_name)
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("module", LOADERS)
@pytest.mark.parametrize("difference", ["effort", "route", "unknown", "sampling", "protocol"])
def test_incompatible_conditions_are_not_averaged(tmp_path, module, difference):
    first = _condition()
    second = _condition()
    if difference == "effort":
        second = _condition("low")
    elif difference == "route":
        second["route_hash"] = "route-two"
    elif difference == "unknown":
        second = {}
    elif difference == "sampling":
        second["request_options"] = {**second["request_options"], "temperature": 0.5}
    else:
        first["benchmark_condition_hash"] = "protocol-one"
        second["benchmark_condition_hash"] = "protocol-two"
    _write(tmp_path / "one", module, condition=first, alias="first-alias")
    _write(tmp_path / "two", module, condition=second, alias="second-alias")
    with pytest.raises(ValueError, match="[Ii]ncompatible.*condition"):
        LOADERS[module](tmp_path)


@pytest.mark.parametrize("module", LOADERS)
def test_overlapping_inputs_do_not_inflate_denominators(tmp_path, module):
    path = _write(tmp_path, module, condition=_condition())
    _write(tmp_path, module, idx=1, condition=_condition())
    before = path.read_bytes()
    normal = LOADERS[module](tmp_path)
    overlapped = LOADERS[module]([tmp_path, path, tmp_path])
    assert overlapped == normal
    assert normal["m"]["n_items"] == 2
    assert path.read_bytes() == before


def test_sus_final_excludes_partial_and_per_unit_copies(tmp_path):
    source = _write(tmp_path, "sus", condition=_condition())
    _write(tmp_path, "sus", condition=_condition(), name="FINAL_RESULTS-partial-conversations.json")
    final = _write(tmp_path, "sus", condition=_condition(), name="FINAL_RESULTS-conversations.json")
    profile = load_sus_results([tmp_path, source])["m"]
    assert profile["n_items"] == 1
    assert profile["metadata"]["source_paths"] == [str(final)]


@pytest.mark.parametrize("module", LOADERS)
def test_duplicate_logical_observations_fail_but_independent_runs_survive(tmp_path, module):
    first = _write(tmp_path / "one", module, condition=_condition())
    duplicate = first.with_name("copy-" + first.name)
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="[Dd]uplicate.*observation"):
        LOADERS[module](tmp_path)
    duplicate.unlink()
    _write(tmp_path / "two", module, condition=_condition())
    assert LOADERS[module](tmp_path)["m"]["n_items"] == 2


@pytest.mark.parametrize("options,effort", [
    ({"reasoning_effort": "none"}, "none"),
    ({"reasoning": {"effort": "high"}}, "high"),
    ({"output_config": {"effort": "max"}}, "max"),
    ({"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}, "HIGH"),
    ({}, "unknown (not recorded)"),
])
def test_report_and_coverage_show_configured_effort(tmp_path, options, effort):
    condition = _condition()
    condition["request_options"] = options
    _write(tmp_path / "input", "sus", condition=condition)
    profiles = build_all_profiles(load_sus_results(tmp_path / "input"), {}, {})
    report = generate_unified_report(profiles, tmp_path / "report")
    assert "Configured effort" in report
    assert f"| {effort} |" in report
    assert "hash-high" in report
    assert "route-one" in report
    assert f"| {effort} |" in _coverage_markdown(profiles)


def test_metadata_effort_cannot_override_request_effort(tmp_path):
    condition = {**_condition(), "condition_metadata": {"effort": "low"}}
    _write(tmp_path, "sus", condition=condition)
    with pytest.raises(ValueError, match="[Ii]ncompatible.*effort"):
        load_sus_results(tmp_path)


def test_cross_module_conditions_must_match_before_composite(tmp_path):
    _write(tmp_path / "sus", "sus", condition=_condition())
    _write(tmp_path / "aita", "aita", condition=_condition("low"))
    sus = load_sus_results(tmp_path / "sus")
    aita = load_aita_results(tmp_path / "aita")
    with pytest.raises(ValueError, match="[Ii]ncompatible.*condition"):
        build_all_profiles(sus, aita, {})


def test_module_protocols_can_differ_with_same_model_condition(tmp_path):
    _write(tmp_path / "sus", "sus", condition={**_condition(), "benchmark_condition_hash": "sus-protocol"})
    _write(tmp_path / "aita", "aita", condition={**_condition(), "benchmark_condition_hash": "aita-protocol"})
    profiles = build_all_profiles(load_sus_results(tmp_path / "sus"), load_aita_results(tmp_path / "aita"), {})
    assert len(profiles) == 1
    assert profiles[0].condition["controls"]["reasoning_effort"] == "high"


def test_cli_refuses_mixed_conditions_without_writing_report(tmp_path, capsys):
    _write(tmp_path / "input", "sus", condition=_condition())
    _write(tmp_path / "input", "sus", idx=1, condition=_condition("low"))
    assert main(["report", "--sus-dir", str(tmp_path / "input"), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out" / "REPORT.md").exists()
    assert "condition" in capsys.readouterr().err


@pytest.mark.parametrize("field,values", [
    ("reasoning_effort", ("low", "high")),
    ("temperature", (0.1, 0.9)),
])
def test_sus_writer_preserves_cli_sweep_controls_before_aggregation(tmp_path, field, values):
    from sus_bench.report import write_json
    from suite_tools.model_config import ensure_model_condition_identity

    identity = ensure_model_condition_identity({"model_id": "m"}, key="m")
    for index, value in enumerate(values):
        row = {**identity, "model": "m", "run_number": 1, "scenario": "test",
               "score": {"sus": 100 * index}, field: value}
        write_json([row], [], tmp_path / str(index) / "results.json")
    with pytest.raises(ValueError, match="[Ii]ncompatible.*condition"):
        load_sus_results(tmp_path)


def test_sus_writer_preserves_single_cli_effort_label(tmp_path):
    from sus_bench.report import write_json

    row = {"model": "m", "run_number": 1, "score": {"sus": 10}, "reasoning_effort": "high"}
    write_json([row], [], tmp_path / "input" / "results.json")
    profiles = build_all_profiles(load_sus_results(tmp_path / "input"), {}, {})
    assert profiles[0].condition["controls"]["reasoning_effort"] == "high"
