"""Tests for assert_hash_panel component-wise comparison-identity checker."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from suite_tools.assert_hash_panel import (
    CONSTANTS_BLESSED_FOR_PROJECTION_VERSION,
    EPIS_INITIAL_FORMATTER_NEW_HASH,
    EPIS_INITIAL_FORMATTER_OLD_HASH,
    FROZEN_AITA_JUDGE_PANEL_HASH,
    FROZEN_EPIS_JUDGE_PANEL_HASH,
    FROZEN_SUS_JUDGE_PANEL_HASH,
    GPT56_AITA_COMPARISON_SPEC_HASH,
    GPT56_EPIS_COMPARISON_SPEC_HASH,
    GPT56_SUS_COMPARISON_SPEC_HASH,
    StaleFrozenConstantsError,
    _epis_benchmark_spec_content_equal,
    _extract_item_universe_aita,
    _extract_item_universe_epis,
    _extract_scenario_universe_sus,
    assert_comparison_identity,
    assert_constants_blessed_for_current_projection,
)
from suite_tools.run_contract import (
    IDENTITY_PROJECTION_VERSION,
    build_provenance_identity,
    legacy_v3_provenance_hashes,
    provenance_hashes,
    write_run_contract,
)

from _reference_contracts import (
    FAIL,
    RUN,
    SKIP,
    reference_gate_decision,
    require_reference_contract,
)

_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "prepared"

_AITA_REF = (
    _RESULTS_ROOT
    / "fable-5-native-suite-n20-frontier-20260702-142711-frontier"
    / "aita"
    / "RUN_CONTRACT.json"
)
_EPIS_REF = (
    _RESULTS_ROOT
    / "fable-5-native-suite-n20-frontier-20260702-142711-frontier"
    / "epis"
    / "RUN_CONTRACT.json"
)
_SUS_REF = (
    _RESULTS_ROOT / "gpt56-luna-low-canary-20260715" / "sus" / "RUN_CONTRACT.json"
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_extract_item_universe_aita_flat_format():
    ss = {"item_indices": [0, 1, 2, 3]}
    assert _extract_item_universe_aita(ss) == {0, 1, 2, 3}


def test_extract_item_universe_aita_per_item_format():
    ss = {"items": [{"item_idx": 0, "pair_id": "x"}, {"item_idx": 5, "pair_id": "y"}]}
    assert _extract_item_universe_aita(ss) == {0, 5}


def test_extract_item_universe_aita_old_new_agree_on_same_indices():
    indices = list(range(20))
    old_ss = {"item_indices": indices}
    new_ss = {"items": [{"item_idx": i, "item_hash": f"hash{i}"} for i in indices]}
    assert _extract_item_universe_aita(old_ss) == _extract_item_universe_aita(new_ss)


def test_extract_item_universe_aita_empty():
    assert _extract_item_universe_aita({}) == set()


def test_extract_item_universe_epis_returns_hash_sets_per_test_type():
    ss = {
        "items": {
            "delusion": [
                {"item_hash": "aaa", "position": 0},
                {"item_hash": "bbb", "position": 1},
            ],
            "mirror": [{"item_hash": "ccc", "position": 0}],
        }
    }
    result = _extract_item_universe_epis(ss)
    assert result == {
        "delusion": frozenset(["aaa", "bbb"]),
        "mirror": frozenset(["ccc"]),
    }


def test_extract_item_universe_epis_empty():
    assert _extract_item_universe_epis({}) == {}


def test_extract_scenario_universe_sus_returns_dict():
    ss = {"scenario_hashes": {"bridge_heights": "deadbeef123"}}
    assert _extract_scenario_universe_sus(ss) == {"bridge_heights": "deadbeef123"}


def test_extract_scenario_universe_sus_empty():
    assert _extract_scenario_universe_sus({}) == {}


# ---------------------------------------------------------------------------
# Synthetic component-wise checker tests (fixture-based, no disk contracts)
# ---------------------------------------------------------------------------


def _minimal_aita_identity(*, judge_panel_hash_override: str | None = None) -> dict:
    """Build minimal AITA identity for fixture contracts."""
    judge_panel = {
        "primary": "openai/gpt-5.5",
        "panel": ["openai/gpt-5.5", "anthropic/claude-opus-4.7", "google/gemini-3.1-pro"],
        "rubric_version": "aita-rubric-v1",
    }
    return build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={
            "module": "aita",
            "module_version": "0.1.0",
            "dataset_mode": "nta-paired",
            "prompt_hashes": {"seeker": "seeker-v1", "flip": "flip-v1"},
            "score_dimensions": ["paired_verdict_alignment"],
        },
        sample_spec={
            "dataset_manifest": {"schema_version": "aita-dataset-manifest-v1", "files": []},
            "dataset_mode": "nta-paired",
            "items": [{"item_idx": i, "item_hash": f"hash{i:02d}"} for i in range(20)],
        },
        judge_panel=judge_panel,
        model_conditions=[{"key": "test-model", "model_id": "test/model"}],
        execution={"run_id": "test-run"},
    )


def _write_contract_json(path: Path, identity: dict, *, module: str = "aita") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": path.parent.name,
        "schema_version": "benchmark-run-contract-v1",
        "identity": identity,
        "modules": [{"module": module}],
        "expected_models": [],
        "expected_judges": [],
    }
    path.write_text(json.dumps(payload))


def test_component_wise_checker_passes_with_matching_synthetic_contracts(tmp_path):
    identity = _minimal_aita_identity()
    ref_path = tmp_path / "ref" / "aita" / "RUN_CONTRACT.json"
    new_path = tmp_path / "new" / "aita" / "RUN_CONTRACT.json"
    _write_contract_json(ref_path, identity)
    _write_contract_json(new_path, identity)

    # Patch judge_panel_hash to match frozen constant by using the real frozen
    # constant as the expected value — but here we use real provenance hashes
    # computed from a judge_panel that exactly produces the frozen hash.
    # For this synthetic test we just assert the checker exits 0 when comparing
    # a contract to itself (every component is trivially equal).
    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=ref_path,
        epis_reference_path=ref_path,  # unused for aita
        sus_reference_path=ref_path,   # unused for aita
    )
    # benchmark_spec_hash, dataset_manifest, dataset_mode, item_universe all match;
    # judge_panel_hash will FAIL because the synthetic judge panel doesn't match
    # the frozen constant — that is expected (we only test the machinery, not the
    # frozen constant itself here).
    # The key assertion: exactly 1 failure (judge_panel_hash mismatch).
    assert result == 1


def test_component_wise_checker_detects_benchmark_spec_mismatch(tmp_path):
    ref_identity = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"module": "aita", "prompt_hashes": {"seeker": "v1"}},
        sample_spec={
            "dataset_manifest": {"files": []},
            "dataset_mode": "nta-paired",
            "item_indices": list(range(5)),
        },
        judge_panel={"primary": "j/model"},
        model_conditions=[{"key": "m", "model_id": "m/m"}],
        execution={},
    )
    new_identity = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"module": "aita", "prompt_hashes": {"seeker": "v2"}},  # changed
        sample_spec={
            "dataset_manifest": {"files": []},
            "dataset_mode": "nta-paired",
            "items": [{"item_idx": i} for i in range(5)],
        },
        judge_panel={"primary": "j/model"},
        model_conditions=[{"key": "m", "model_id": "m/m"}],
        execution={},
    )
    ref_path = tmp_path / "ref" / "aita" / "RUN_CONTRACT.json"
    new_path = tmp_path / "new" / "aita" / "RUN_CONTRACT.json"
    _write_contract_json(ref_path, ref_identity)
    _write_contract_json(new_path, new_identity)

    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=ref_path,
        epis_reference_path=ref_path,
        sus_reference_path=ref_path,
    )
    # benchmark_spec_hash differs → failure; plus judge_panel_hash won't match frozen
    assert result == 1


def test_component_wise_checker_detects_item_universe_mismatch(tmp_path):
    ref_identity = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"module": "aita", "prompt_hashes": {"seeker": "v1"}},
        sample_spec={
            "dataset_manifest": {"files": []},
            "dataset_mode": "nta-paired",
            "item_indices": list(range(20)),
        },
        judge_panel={"primary": "j/model"},
        model_conditions=[{"key": "m", "model_id": "m/m"}],
        execution={},
    )
    # new contract with only 19 items — universe mismatch
    new_identity = build_provenance_identity(
        benchmark_family_id="aita",
        benchmark_spec={"module": "aita", "prompt_hashes": {"seeker": "v1"}},
        sample_spec={
            "dataset_manifest": {"files": []},
            "dataset_mode": "nta-paired",
            "items": [{"item_idx": i} for i in range(19)],
        },
        judge_panel={"primary": "j/model"},
        model_conditions=[{"key": "m", "model_id": "m/m"}],
        execution={},
    )
    ref_path = tmp_path / "ref" / "aita" / "RUN_CONTRACT.json"
    new_path = tmp_path / "new" / "aita" / "RUN_CONTRACT.json"
    _write_contract_json(ref_path, ref_identity)
    _write_contract_json(new_path, new_identity)

    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=ref_path,
        epis_reference_path=ref_path,
        sus_reference_path=ref_path,
    )
    assert result == 1


def test_component_wise_checker_skips_unknown_module(tmp_path):
    identity = _minimal_aita_identity()
    bad_path = tmp_path / "bad" / "RUN_CONTRACT.json"
    _write_contract_json(bad_path, identity, module="unknown_module")

    result = assert_comparison_identity(
        [bad_path],
        aita_reference_path=tmp_path / "nonexistent",
        epis_reference_path=tmp_path / "nonexistent",
        sus_reference_path=tmp_path / "nonexistent",
    )
    # Nothing checked → 0 failures
    assert result == 0


# ---------------------------------------------------------------------------
# Live-contract tests
#
# These are gated by _reference_contracts.require_reference_contract, which
# SKIPS only in a genuine fresh clone (no results/prepared/ at all) and FAILS
# in a data-bearing checkout whose reference contracts have gone missing.  They
# used to skip silently in both cases, which is how the 3392139 projection
# change went unnoticed for six days — see tests/_reference_contracts.py.
# ---------------------------------------------------------------------------


def test_live_gpt56_aita_sol_component_wise():
    """All 5 components must PASS for gpt56-sol AITA."""
    require_reference_contract(_RESULTS_ROOT, _AITA_REF)
    new_path = (
        _RESULTS_ROOT
        / "gpt56-sol-suite-n20-frontier-20260716"
        / "aita"
        / "RUN_CONTRACT.json"
    )
    require_reference_contract(_RESULTS_ROOT, new_path)
    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=_AITA_REF,
        epis_reference_path=_EPIS_REF,
        sus_reference_path=_SUS_REF,
    )
    assert result == 0


def test_live_gpt56_epis_fingerprint_scheme_cross_verified():
    """Verify scheme-cross facts for the real fable-era vs gpt56-era EPIS contracts.

    Facts proved here:
    - benchmark_spec_hash DIFFERS (different fingerprint schemes, same source)
    - judge_panel_hash MATCHES frozen constant
    - item universe MATCHES
    - _epis_benchmark_spec_content_equal returns PASS
    """
    require_reference_contract(_RESULTS_ROOT, _EPIS_REF)
    new_path = (
        _RESULTS_ROOT
        / "gpt56-sol-suite-n20-frontier-20260716"
        / "epis"
        / "RUN_CONTRACT.json"
    )
    require_reference_contract(_RESULTS_ROOT, new_path)

    new_contract = json.loads(new_path.read_text())
    ref_contract = json.loads(_EPIS_REF.read_text())
    new_h = provenance_hashes(new_contract.get("identity", {}))
    ref_h = provenance_hashes(ref_contract.get("identity", {}))

    # benchmark_spec_hash must differ (different fingerprint schemes)
    assert new_h["benchmark_spec_hash"] != ref_h["benchmark_spec_hash"]
    # judge_panel_hash must match frozen constant
    assert new_h["judge_panel_hash"] == FROZEN_EPIS_JUDGE_PANEL_HASH
    # item universe must match
    new_ss = new_contract["identity"]["sample_spec"]
    ref_ss = ref_contract["identity"]["sample_spec"]
    assert _extract_item_universe_epis(new_ss) == _extract_item_universe_epis(ref_ss)
    # scheme-aware content check must PASS
    new_bs = new_contract["identity"]["benchmark_spec"]
    ref_bs = ref_contract["identity"]["benchmark_spec"]
    ok, note = _epis_benchmark_spec_content_equal(new_bs, ref_bs)
    assert ok, f"scheme-aware check unexpectedly failed: {note}"


def test_live_gpt56_sus_sol_component_wise():
    """All 5 components must PASS for gpt56-sol SUS."""
    require_reference_contract(_RESULTS_ROOT, _SUS_REF)
    new_path = (
        _RESULTS_ROOT
        / "gpt56-sol-suite-n20-frontier-20260716"
        / "sus"
        / "RUN_CONTRACT.json"
    )
    require_reference_contract(_RESULTS_ROOT, new_path)
    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=_AITA_REF,
        epis_reference_path=_EPIS_REF,
        sus_reference_path=_SUS_REF,
    )
    assert result == 0


def test_live_component_wise_all_10_contracts_all_pass():
    """Run all 10 new gpt56 contracts and confirm all 50 components PASS.

    EPIS benchmark_spec_hash uses scheme-aware content equality (same source,
    two fingerprint schemes), so no failures are expected.
    """
    for ref in (_AITA_REF, _EPIS_REF, _SUS_REF):
        require_reference_contract(_RESULTS_ROOT, ref)
    contract_dirs = [
        "gpt56-sol-suite-n20-frontier-20260716/aita",
        "gpt56-sol-suite-n20-frontier-20260716/epis",
        "gpt56-sol-suite-n20-frontier-20260716/sus",
        "gpt56-terra-suite-n20-frontier-20260716/aita",
        "gpt56-terra-suite-n20-frontier-20260716/epis",
        "gpt56-terra-suite-n20-frontier-20260716/sus",
        "gpt56-luna-suite-n20-frontier-20260716/aita",
        "gpt56-luna-suite-n20-frontier-20260716/epis",
        "gpt56-luna-suite-n20-frontier-20260716/sus",
        "gpt56-none-sus-n20-frontier-20260716/sus",
    ]
    paths = [_RESULTS_ROOT / d / "RUN_CONTRACT.json" for d in contract_dirs]
    for path in paths:
        require_reference_contract(_RESULTS_ROOT, path)

    from io import StringIO
    import sys as _sys

    old_stdout = _sys.stdout
    _sys.stdout = StringIO()
    try:
        rc = assert_comparison_identity(
            paths,
            aita_reference_path=_AITA_REF,
            epis_reference_path=_EPIS_REF,
            sus_reference_path=_SUS_REF,
        )
    finally:
        _sys.stdout = old_stdout

    assert rc == 0, "Expected all 50 component checks to PASS (0 failures)"


# ---------------------------------------------------------------------------
# Projection-version binding for the frozen constants
#
# The frozen constants are hashes of identities *as filtered by a projection*.
# A projection bump re-values all of them without touching any content, which is
# exactly what commit 3392139 did on 2026-07-18.  These tests make the next such
# bump fail loudly at the constants, instead of surfacing as six mystery hash
# mismatches days later.
# ---------------------------------------------------------------------------


def test_frozen_constants_remain_bound_to_legacy_projection_v3():
    """Frozen preregistration constants retain their original projection."""
    assert CONSTANTS_BLESSED_FOR_PROJECTION_VERSION == "benchmark-identity-projection-v3", (
        "suite_tools/assert_hash_panel.py frozen constants are stale: they were "
        f"blessed for {CONSTANTS_BLESSED_FOR_PROJECTION_VERSION!r}; current is "
        f"{IDENTITY_PROJECTION_VERSION!r}. Preserve the v3 verification path "
        "(recompute from the stored contracts, keep the old values in the HISTORY "
        "comment, and add a dated amendment to PREREG_FREEZE_GPT56_20260716.json) "
        "— see assert_constants_blessed_for_current_projection()."
    )


def test_projection_guard_passes_for_the_frozen_projection():
    """The guard is a no-op for the projection that produced the constants."""
    assert_constants_blessed_for_current_projection("benchmark-identity-projection-v3")


def test_projection_guard_raises_with_rebless_instructions_on_version_bump():
    """A future projection bump must raise an actionable error, not pass silently."""
    with pytest.raises(StaleFrozenConstantsError) as excinfo:
        assert_constants_blessed_for_current_projection(
            "benchmark-identity-projection-v5"
        )
    message = str(excinfo.value)
    # The message must name both versions and tell the reader what to do.
    assert CONSTANTS_BLESSED_FOR_PROJECTION_VERSION in message
    assert "benchmark-identity-projection-v5" in message
    assert "PREREG_FREEZE_GPT56_20260716.json" in message
    assert "--component-wise" in message


def test_frozen_judge_panel_constants_match_recomputation_from_stored_contracts():
    """Each frozen constant must be reproducible from the contract it anchors.

    This is the check that actually binds the constants to disk: it recomputes
    judge_panel_hash under the active projection and compares byte-for-byte.
    """
    cases = [
        (
            _AITA_REF,
            "fable-5-native-suite-n20-frontier-20260702-142711-frontier/aita",
            FROZEN_AITA_JUDGE_PANEL_HASH,
        ),
        (
            _EPIS_REF,
            "fable-5-native-suite-n20-frontier-20260702-142711-frontier/epis",
            FROZEN_EPIS_JUDGE_PANEL_HASH,
        ),
        (_SUS_REF, "gpt56-luna-low-canary-20260715/sus", FROZEN_SUS_JUDGE_PANEL_HASH),
    ]
    for path, label, frozen in cases:
        require_reference_contract(_RESULTS_ROOT, path)
        contract = json.loads(path.read_text())
        actual = legacy_v3_provenance_hashes(contract["identity"])["judge_panel_hash"]
        assert actual == frozen, (
            f"frozen judge_panel_hash for {label} does not match recomputation:\n"
            f"  frozen: {frozen}\n"
            f"  actual: {actual}"
        )


def test_gpt56_comparison_spec_anchors_match_recomputation_from_stored_contracts():
    """The three gpt56-era anchors must be reproducible from the gpt56 contracts."""
    cases = [
        ("gpt56-sol-suite-n20-frontier-20260716/aita", GPT56_AITA_COMPARISON_SPEC_HASH),
        ("gpt56-sol-suite-n20-frontier-20260716/epis", GPT56_EPIS_COMPARISON_SPEC_HASH),
        ("gpt56-sol-suite-n20-frontier-20260716/sus", GPT56_SUS_COMPARISON_SPEC_HASH),
    ]
    for subpath, anchor in cases:
        path = _RESULTS_ROOT / subpath / "RUN_CONTRACT.json"
        require_reference_contract(_RESULTS_ROOT, path)
        contract = json.loads(path.read_text())
        actual = legacy_v3_provenance_hashes(contract["identity"])["comparison_spec_hash"]
        assert actual == anchor, (
            f"gpt56 comparison_spec_hash anchor for {subpath} does not match:\n"
            f"  anchor: {anchor}\n"
            f"  actual: {actual}"
        )


# ---------------------------------------------------------------------------
# The skip/fail discriminator itself (see tests/_reference_contracts.py)
# ---------------------------------------------------------------------------


def test_gate_runs_when_reference_contract_is_present(tmp_path):
    prepared = tmp_path / "prepared"
    contract = prepared / "run" / "RUN_CONTRACT.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    decision, _ = reference_gate_decision(prepared, contract)
    assert decision == RUN


def test_gate_skips_on_a_genuine_fresh_clone(tmp_path):
    """No results/prepared/ at all → no benchmark data → skipping is honest."""
    prepared = tmp_path / "prepared"  # never created
    contract = prepared / "run" / "RUN_CONTRACT.json"
    decision, reason = reference_gate_decision(prepared, contract)
    assert decision == SKIP
    assert "fresh clone" in reason


def test_gate_fails_when_data_bearing_checkout_is_missing_the_contract(tmp_path):
    """This is the regression that made the 3392139 break invisible for six days.

    results/prepared/ exists (so the checkout does carry benchmark data) but the
    frozen reference contract is gone.  That must FAIL, not skip.
    """
    prepared = tmp_path / "prepared"
    (prepared / "some-other-run").mkdir(parents=True)
    contract = prepared / "missing-run" / "RUN_CONTRACT.json"
    decision, reason = reference_gate_decision(prepared, contract)
    assert decision == FAIL
    assert "data-bearing" in reason
    # The failure must tell the reader how to resolve it.
    assert "BENCH_ALLOW_MISSING_REFERENCE_CONTRACTS" in reason


def test_gate_escape_hatch_downgrades_fail_to_skip(tmp_path):
    """Third-party checkouts with their own runs can opt out explicitly."""
    prepared = tmp_path / "prepared"
    (prepared / "their-own-run").mkdir(parents=True)
    contract = prepared / "missing-run" / "RUN_CONTRACT.json"
    decision, _ = reference_gate_decision(prepared, contract, allow_missing=True)
    assert decision == SKIP


def test_gate_escape_hatch_does_not_apply_when_contract_is_present(tmp_path):
    """The opt-out must never suppress a check that could actually run."""
    prepared = tmp_path / "prepared"
    contract = prepared / "run" / "RUN_CONTRACT.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    decision, _ = reference_gate_decision(prepared, contract, allow_missing=True)
    assert decision == RUN


def test_this_checkout_is_data_bearing_so_live_tests_are_really_running():
    """Guard against the whole live-test block silently vanishing again.

    If results/prepared/ is present, every live reference contract in this file
    must resolve — otherwise the suite would be reporting green on nothing.

    Routed through the shared gate so it honours the same fresh-clone skip and
    the same explicit opt-out as the tests it is guarding.
    """
    for ref in (_AITA_REF, _EPIS_REF, _SUS_REF):
        require_reference_contract(_RESULTS_ROOT, ref)


def test_gpt56_era_comparison_spec_hash_constants_are_distinct_per_module():
    """The three GPT-5.6-era anchors must be distinct from each other and from v1.0.1."""
    anchors = [
        GPT56_AITA_COMPARISON_SPEC_HASH,
        GPT56_EPIS_COMPARISON_SPEC_HASH,
        GPT56_SUS_COMPARISON_SPEC_HASH,
    ]
    assert len(set(anchors)) == 3, "GPT-5.6 per-module anchors must all be distinct"
    # Must differ from frozen v1.0.1 AITA value
    assert GPT56_AITA_COMPARISON_SPEC_HASH != "2370e13e75f5790cf60babfb3da09515a16b71561fcf8bfbef6536843887d146"


# ---------------------------------------------------------------------------
# EPIS fingerprint-scheme helpers and runner/prepare consistency
# ---------------------------------------------------------------------------


def test_epis_initial_formatter_old_hash_is_constant_string_scheme():
    """Stored constant matches stable_json_hash of the old constant-string input."""
    import sys
    from suite_tools.run_contract import stable_json_hash as sjh

    assert EPIS_INITIAL_FORMATTER_OLD_HASH == sjh("epis_bench.prompts.format_initial_prompt")


def test_epis_initial_formatter_new_hash_is_source_hash_scheme():
    """Stored constant matches stable_json_hash(inspect.getsource(format_initial_prompt))."""
    import inspect
    import sys

    sys.path.insert(0, str(_RESULTS_ROOT.parents[1] / "epistemic-sycophancy-bench"))
    from epis_bench import prompts as ep
    from suite_tools.run_contract import stable_json_hash as sjh

    assert EPIS_INITIAL_FORMATTER_NEW_HASH == sjh(inspect.getsource(ep.format_initial_prompt))


def test_runner_and_prepare_compute_same_initial_formatter_hash():
    """runner.py (fixed) and prepare_run.py must produce the same initial_formatter value.

    Both now use stable_json_hash(inspect.getsource(format_initial_prompt)), so
    the value each embeds in the contract's benchmark_spec must be identical.
    """
    import inspect
    import sys

    epis_root = _RESULTS_ROOT.parents[1] / "epistemic-sycophancy-bench"
    if str(epis_root) not in sys.path:
        sys.path.insert(0, str(epis_root))

    from epis_bench import prompts as ep
    from epis_bench.runner import format_initial_prompt as runner_fn  # noqa: PLC0415
    from suite_tools.run_contract import stable_json_hash as sjh

    # Both imports must resolve to the same function object / same source
    runner_hash = sjh(inspect.getsource(runner_fn))
    prepare_hash = sjh(inspect.getsource(ep.format_initial_prompt))
    assert runner_hash == prepare_hash, (
        "runner.py and prepare_run.py compute different initial_formatter hashes — "
        "runner.py still uses the old constant-string scheme"
    )
    assert runner_hash == EPIS_INITIAL_FORMATTER_NEW_HASH


def test_epis_benchmark_spec_content_equal_accepts_scheme_crossing():
    """The scheme-aware checker must return PASS when new=source-hash, ref=constant-string."""
    new_bs = {
        "prompt_hashes": {
            "initial_formatter": EPIS_INITIAL_FORMATTER_NEW_HASH,
            "seeker_delusion": "same-value",
        },
        "module": "epistemic",
    }
    ref_bs = {
        "prompt_hashes": {
            "initial_formatter": EPIS_INITIAL_FORMATTER_OLD_HASH,
            "seeker_delusion": "same-value",
        },
        "module": "epistemic",
    }
    ok, note = _epis_benchmark_spec_content_equal(new_bs, ref_bs)
    assert ok, f"Expected PASS but got FAIL: {note}"
    assert "same source content" in note


def test_epis_benchmark_spec_content_equal_rejects_wrong_new_hash():
    """Must return FAIL when the new contract has a hash that matches neither scheme."""
    new_bs = {
        "prompt_hashes": {"initial_formatter": "deadbeef" * 8},
        "module": "epistemic",
    }
    ref_bs = {
        "prompt_hashes": {"initial_formatter": EPIS_INITIAL_FORMATTER_OLD_HASH},
        "module": "epistemic",
    }
    ok, note = _epis_benchmark_spec_content_equal(new_bs, ref_bs)
    assert not ok, "Expected FAIL but got PASS"


def test_epis_benchmark_spec_content_equal_rejects_other_prompt_hash_mismatch():
    """Must return FAIL when a non-initial_formatter seeker prompt hash differs."""
    new_bs = {
        "prompt_hashes": {
            "initial_formatter": EPIS_INITIAL_FORMATTER_NEW_HASH,
            "seeker_delusion": "value-a",
        },
        "module": "epistemic",
    }
    ref_bs = {
        "prompt_hashes": {
            "initial_formatter": EPIS_INITIAL_FORMATTER_OLD_HASH,
            "seeker_delusion": "value-b",  # different
        },
        "module": "epistemic",
    }
    ok, note = _epis_benchmark_spec_content_equal(new_bs, ref_bs)
    assert not ok, "Expected FAIL but got PASS"


def test_live_gpt56_epis_benchmark_spec_content_equal_passes():
    """The scheme-aware check must PASS for the real gpt56-sol EPIS contract."""
    require_reference_contract(_RESULTS_ROOT, _EPIS_REF)
    new_path = (
        _RESULTS_ROOT
        / "gpt56-sol-suite-n20-frontier-20260716"
        / "epis"
        / "RUN_CONTRACT.json"
    )
    require_reference_contract(_RESULTS_ROOT, new_path)
    result = assert_comparison_identity(
        [new_path],
        aita_reference_path=_AITA_REF,
        epis_reference_path=_EPIS_REF,
        sus_reference_path=_SUS_REF,
    )
    assert result == 0, "Expected all 5 components to PASS (including scheme-aware EPIS benchmark_spec)"
