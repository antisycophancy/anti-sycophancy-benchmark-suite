"""Tests for suite_tools.experiment — TDD RED → GREEN phase.

Fixtures build minimal RUN_CONTRACT.json run directories (either via the real
prepare_aita_run/prepare_sus_run helpers or hand-crafted minimal contracts)
and return the run dir (the directory containing RUN_CONTRACT.json).

Fix-review additions (collision listing, fingerprint moved-member, non-degenerate fixtures):
- test_status_lists_duplicate_unit_collisions
- test_status_flags_absent_member_contract
- test_status_flags_fingerprint_mismatch_member
"""
import json
from pathlib import Path

import pytest

from suite_tools import experiment
from suite_tools.prepare_run import prepare_aita_run, prepare_sus_run
from suite_tools.run_contract import summarize_contract


# ---------------------------------------------------------------------------
# Shared non-trivial specs used by the two-condition fixtures.
# Both cond_a and cond_b share these → same benchmark_condition_hash.
# Different model_conditions → different condition_ids.
# ---------------------------------------------------------------------------

_SHARED_AITA_BENCHMARK_SPEC = {
    "scoring_contract": {
        "primary_outcome_dimensions": ["verdict_alignment_a", "verdict_alignment_b"],
        "diagnostic_dimensions": ["therapeutic_a", "therapeutic_b"],
    },
    "rubric_version": "aita-rubric-v2",
}

_SHARED_JUDGE_PANEL = {
    "seeker": "google/gemini-3-flash-preview",
    "flip_generator": "google/gemini-3-flash-preview",
}

_SHARED_SAMPLE_SPEC = {
    "items": [{"id": "item-0", "sides": ["side_a"]}],
    "n_items": 1,
}


# ---------------------------------------------------------------------------
# Fixtures: real prepare_* runs
# ---------------------------------------------------------------------------


@pytest.fixture
def aita_run(tmp_path):
    """Real aita run built with prepare_aita_run; returns run dir."""
    contract_path = prepare_aita_run(
        run_id="aita-exp-smoke",
        output_root=tmp_path / "aita-exp-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
        source_command="test",
    )
    return contract_path.parent  # e.g. tmp_path/aita-exp-smoke/aita


@pytest.fixture
def sus_run(tmp_path):
    """Real sus run built with prepare_sus_run; returns run dir."""
    contract_path = prepare_sus_run(
        run_id="sus-exp-smoke",
        output_root=tmp_path / "sus-exp-smoke",
        suite_config_path=Path("suite_models.yaml"),
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        scenarios_selector="bridge_heights",
        runs=1,
        source_command="test",
    )
    return contract_path.parent  # e.g. tmp_path/sus-exp-smoke/sus


# ---------------------------------------------------------------------------
# Fixture: aita run with different benchmark_condition_hash (other panel)
# ---------------------------------------------------------------------------


@pytest.fixture
def aita_run_other_panel(tmp_path):
    """Aita run whose benchmark_condition_hash differs from the standard aita_run.

    Uses a different judge_panel and benchmark_spec content so the recomputed
    benchmark_condition_hash is guaranteed to differ from the real prepare_aita_run
    output used in aita_run.
    """
    run_dir = tmp_path / "aita-other-panel"
    run_dir.mkdir()
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "aita-other-panel",
        "lifecycle_state": "prepared",
        "identity": {
            "benchmark_family_id": "aita",
            "benchmark_spec": {"scoring_contract": {"primary_outcome_dimensions": ["verdict_DIFFERENT"]}},
            "sample_spec": {"items": [], "n_items": 0},
            "judge_panel": {"panel": "different-panel-id"},
            "model_conditions": [{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
        },
        "expected_models": [{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
        "expected_judges": [],
        "modules": [
            {
                "module": "aita",
                "stage": "generation",
                "expected_units": [],
            }
        ],
    }
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    return run_dir


# ---------------------------------------------------------------------------
# Fixtures: two-condition aita runs (non-degenerate specs, 1 unit each)
# ---------------------------------------------------------------------------


def _minimal_aita_contract(model_key: str, model_id: str, transcript_path: str) -> dict:
    """Build a minimal but non-trivial aita RUN_CONTRACT.json with exactly 1 unit.

    Both cond_a and cond_b share the same benchmark_spec / judge_panel / sample_spec
    content so their benchmark_condition_hash values are equal (required for both
    to be adoptable into the same experiment).  They differ only in model_conditions
    (different keys/model_ids), giving distinct condition_ids.
    """
    return {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": f"aita-cond-{model_key}",
        "lifecycle_state": "prepared",
        "identity": {
            "benchmark_family_id": "aita",
            "benchmark_spec": _SHARED_AITA_BENCHMARK_SPEC,
            "sample_spec": _SHARED_SAMPLE_SPEC,
            "judge_panel": _SHARED_JUDGE_PANEL,
            "model_conditions": [{"key": model_key, "model_id": model_id}],
        },
        "expected_models": [{"key": model_key, "model_id": model_id}],
        "expected_judges": [
            {"role": "seeker", "model_id": "google/gemini-3-flash-preview"},
        ],
        "modules": [
            {
                "module": "aita",
                "stage": "generation",
                "expected_units": [
                    {
                        "unit_id": f"aita:{model_key}:item0:side_a",
                        "model_key": model_key,
                        "model_id": model_id,
                        "item_idx": 0,
                        "side": "side_a",
                        "planned_turns": 5,
                        "expected_transcript_path": transcript_path,
                    }
                ],
            }
        ],
    }


@pytest.fixture
def aita_run_cond_a_done(tmp_path):
    """Aita run for condition 'model-a' with its single unit's transcript complete."""
    run_dir = tmp_path / "aita-cond-a"
    run_dir.mkdir()
    transcript_path = "model-a_item0_side_a.json"
    contract = _minimal_aita_contract("model-a", "provider/model-a", transcript_path)
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    # Write a completed transcript (5 turns >= planned_turns=5 → done)
    (run_dir / transcript_path).write_text(json.dumps({"turns": [1, 2, 3, 4, 5]}))
    return run_dir


@pytest.fixture
def aita_run_cond_b_owed(tmp_path):
    """Aita run for condition 'model-b' with its single unit's transcript absent (owed)."""
    run_dir = tmp_path / "aita-cond-b"
    run_dir.mkdir()
    transcript_path = "model-b_item0_side_a.json"
    contract = _minimal_aita_contract("model-b", "provider/model-b", transcript_path)
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    # No transcript written → unit is owed
    return run_dir


# ---------------------------------------------------------------------------
# Original 5 tests from the brief
# ---------------------------------------------------------------------------


def test_init_from_run_seeds_per_module_instrument(tmp_path, aita_run, sus_run):
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1", from_runs=[aita_run, sus_run])
    manifest = json.loads((exp / "EXPERIMENT.json").read_text())
    assert set(manifest["instrument"]["hashes"]) == {"aita", "sus"}
    assert manifest["instrument"]["hashes"]["aita"] != manifest["instrument"]["hashes"]["sus"]


@pytest.mark.parametrize(
    "experiment_id",
    ["", ".", "..", "../escape", "/tmp/escape", "a/b", "a\\b", "line\nbreak"],
)
def test_init_rejects_nonportable_experiment_ids_before_writing(tmp_path, experiment_id):
    exp = tmp_path / "exp"

    with pytest.raises(ValueError, match="experiment_id must be"):
        experiment.init(exp, experiment_id=experiment_id, title="unsafe")

    assert not exp.exists()


def test_target_items_from_flag(tmp_path, aita_run):
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1", from_runs=[aita_run], target_items=20)
    assert json.loads((exp / "EXPERIMENT.json").read_text())["target"]["n_items"] == 20


def test_adopt_maintains_conditions_and_canonical_fingerprint(tmp_path, aita_run):
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1", from_runs=[aita_run])
    experiment.adopt(exp, aita_run, role="pilot")
    manifest = json.loads((exp / "EXPERIMENT.json").read_text())
    member = manifest["members"][0]
    assert member["contract_fingerprint"] == summarize_contract(aita_run / "RUN_CONTRACT.json")["contract_fingerprint"]
    assert "projection_version" in member
    assert manifest["conditions"]
    assert all({"key", "condition_id", "condition_hash"} <= set(c) for c in manifest["conditions"])


def test_adopt_refuses_instrument_mismatch(tmp_path, aita_run, aita_run_other_panel):
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1", from_runs=[aita_run])
    with pytest.raises(experiment.InstrumentMismatch) as ei:
        experiment.adopt(exp, aita_run_other_panel, role="expansion")
    assert "benchmark_condition_hash" in str(ei.value)


def test_status_reports_distinct_counts_per_condition(tmp_path, aita_run_cond_a_done, aita_run_cond_b_owed):
    # R3-4: two conditions, one fully done and one owed → status reports DIFFERENT
    # done/owed counts per condition, not just two condition labels.
    # Build aita_run_cond_a_done with its single unit's transcript complete, and
    # aita_run_cond_b_owed with its unit missing (owed), via prepare_aita_run + a
    # written/absent transcript respectively (target_items=1 each).
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")
    experiment.adopt(exp, aita_run_cond_b_owed, role="expansion")
    st = experiment.status(exp)
    by_condition = {row["condition_id"]: row for row in st["completeness"]}
    assert len(by_condition) == 2
    counts = sorted((row["done"], row["owed"]) for row in by_condition.values())
    assert counts == [(0, 1), (1, 0)]  # distinct per-condition completeness, not just labels


def test_status_maps_legacy_sus_hash_unit_id_via_expected_unit_model_key(tmp_path):
    run_dir = tmp_path / "legacy-sus"
    run_dir.mkdir()
    transcript_path = "unit.json"
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "legacy-sus",
        "identity": {
            "benchmark_family_id": "sus",
            "benchmark_spec": {"module": "sus", "version": "v1"},
            "sample_spec": {"scenario_ids": ["bridge"], "runs": 1},
            "judge_panel": {"panel": ["judge"]},
            "model_conditions": [{
                "key": "model-key",
                "model_id": "provider/model",
                "condition_id": "scientific-condition-id",
            }],
        },
        "expected_models": [{"key": "model-key", "model_id": "provider/model"}],
        "modules": [{
            "module": "sus",
            "expected_units": [{
                "unit_id": "sus:deadbeef1234:bridge:run1",
                "model_key": "model-key",
                "scenario": "bridge",
                "run_number": 1,
                "planned_escalations": 0,
                "expected_transcript_path": transcript_path,
            }],
        }],
    }
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    (run_dir / transcript_path).write_text(json.dumps({"phases": {"elicit": {}}}))

    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1", from_runs=[run_dir])
    experiment.adopt(exp, run_dir, role="pilot")

    assert experiment.status(exp)["completeness"] == [{
        "condition_id": "scientific-condition-id",
        "module": "sus",
        "done": 1,
        "owed": 0,
        "terminal": 0,
    }]


# ---------------------------------------------------------------------------
# Fix-review: collision listing
# ---------------------------------------------------------------------------


def test_status_lists_duplicate_unit_collisions(tmp_path, aita_run_cond_a_done):
    """Adopting the same run twice → same unit appears in both members → collision listed."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")
    experiment.adopt(exp, aita_run_cond_a_done, role="expansion")  # same run → duplicate unit_ids
    st = experiment.status(exp)
    assert st["collisions"], "expected at least one collision"
    collision = st["collisions"][0]
    assert collision["unit_id"] == "aita:model-a:item0:side_a"
    assert collision["kept_member"]
    assert collision["dropped_members"]
    # Counts still correct — newest-wins means done=1, not done=2
    by_condition = {row["condition_id"]: row for row in st["completeness"]}
    assert by_condition["model-a"]["done"] == 1
    assert by_condition["model-a"]["owed"] == 0


# ---------------------------------------------------------------------------
# Fix-review: moved-member detection
# ---------------------------------------------------------------------------


def test_status_flags_absent_member_contract(tmp_path, aita_run_cond_a_done):
    """Member whose contract file has been deleted gets a 'file_missing' warning
    and its units are excluded from completeness."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")
    # Delete the contract file after adoption
    (aita_run_cond_a_done / "RUN_CONTRACT.json").unlink()
    st = experiment.status(exp)
    assert any(w.get("reason") == "file_missing" for w in st.get("warnings") or [])
    # Unit excluded → no completeness rows
    assert st["completeness"] == []


def test_status_flags_fingerprint_mismatch_member(tmp_path, aita_run_cond_a_done):
    """Member whose contract file exists but fingerprint changed gets a
    'fingerprint_mismatch' warning and its units are excluded from completeness."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")
    # Mutate a fingerprinted field (schema_version is hashed into contract_fingerprint)
    contract_path = aita_run_cond_a_done / "RUN_CONTRACT.json"
    data = json.loads(contract_path.read_text())
    data["schema_version"] = "benchmark-run-contract-MUTATED"
    contract_path.write_text(json.dumps(data))
    st = experiment.status(exp)
    assert any(w.get("reason") == "fingerprint_mismatch" for w in st.get("warnings") or [])
    # Unit excluded from completeness
    assert st["completeness"] == []


# ---------------------------------------------------------------------------
# Terminal bucket + module dimension (final-review items)
# ---------------------------------------------------------------------------


@pytest.fixture
def aita_run_cond_terminal(tmp_path):
    """Aita run for condition 'model-t' with its single unit in terminal state (output_budget_exhausted)."""
    run_dir = tmp_path / "aita-cond-t"
    run_dir.mkdir()
    transcript_path = "model-t_item0_side_a.json"
    contract = _minimal_aita_contract("model-t", "provider/model-t", transcript_path)
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    # Write a terminal transcript (output_budget_exhausted=True → terminal_model_signal)
    (run_dir / transcript_path).write_text(json.dumps({
        "turns": [],
        "completed": False,
        "output_budget_exhausted": True,
    }))
    return run_dir


def test_status_terminal_unit_not_counted_as_owed(tmp_path, aita_run_cond_terminal):
    """A terminal_model_signal unit must appear in terminal=1 and owed=0, not owed=1."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_terminal], target_items=1)
    experiment.adopt(exp, aita_run_cond_terminal, role="pilot")
    st = experiment.status(exp)
    rows = st["completeness"]
    assert len(rows) == 1
    row = rows[0]
    assert row["terminal"] == 1
    assert row["owed"] == 0
    assert row["done"] == 0
    assert row["module"] == "aita"
    assert row["condition_id"] == "model-t"


def test_status_completeness_has_module_dimension(tmp_path, aita_run_cond_a_done, aita_run_cond_b_owed):
    """status() completeness rows include the module field (per condition × module)."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")
    experiment.adopt(exp, aita_run_cond_b_owed, role="expansion")
    st = experiment.status(exp)
    for row in st["completeness"]:
        assert "module" in row, "completeness row missing 'module' field"
        assert row["module"] == "aita"
    # terminal field present on all rows
    for row in st["completeness"]:
        assert "terminal" in row


def test_status_captures_member_errors_for_corrupt_contract(tmp_path, aita_run_cond_a_done):
    """A member whose owed_units call errors is captured in member_errors, not warnings."""
    exp = tmp_path / "exp"
    experiment.init(exp, experiment_id="e1", title="E1",
                    from_runs=[aita_run_cond_a_done], target_items=1)
    experiment.adopt(exp, aita_run_cond_a_done, role="pilot")

    # Corrupt the contract by replacing the modules list with an unrecognised module name
    contract_path = aita_run_cond_a_done / "RUN_CONTRACT.json"
    data = json.loads(contract_path.read_text())
    # Point to an unknown module; owed_units raises ValueError for it.
    # We also need to keep the fingerprint stable, so we re-adopt after mutation.
    # Instead: monkeypatch _compute_owed_units to raise for this test.
    from unittest.mock import patch
    with patch("suite_tools.experiment._compute_owed_units", side_effect=ValueError("unknown module 'bogus'")):
        st = experiment.status(exp)

    assert st.get("member_errors"), "expected at least one member_error"
    err = st["member_errors"][0]
    assert "error" in err
    assert "unknown module" in err["error"]


# ---------------------------------------------------------------------------
# BUG 2: module="epistemic" alias in contracts
# ---------------------------------------------------------------------------


def _minimal_epistemic_contract(started_at: str) -> tuple[Path, str]:
    """Build a run dir with module='epistemic' (real-contract alias) and one unit."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    unit = "epis:m:pickside:item0:side_a"
    (d / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "epis-run",
        "modules": [{
            "module": "epistemic",          # the real-contract alias
            "expected_units": [{
                "unit_id": unit,
                "side": "side_a",
                "test_type": "pickside",
                "item_idx": 0,
                "planned_turns": 2,
                "expected_transcript_path": "t.json",
            }],
        }],
        "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]},
    }))
    (d / "RUN_STATUS.json").write_text(json.dumps(
        {"attempt_number": 1, "started_at": started_at}))
    (d / "t.json").write_text(json.dumps({"turns": [1, 2], "completed": True}))
    return d, unit


def test_union_accepts_epistemic_alias(tmp_path):
    """experiment.union() must process a member with module='epistemic' without raising.

    The unit must appear in the union output (not in member_errors), and the
    module field in the union result must be the canonical "epis" name.
    """
    run_dir, unit_id = _minimal_epistemic_contract("2026-07-20T12:00:00Z")
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "epis-alias-test",
        "title": "t",
        "instrument": {"modules": ["epistemic"], "hashes": {}},
        "conditions": [],
        "target": {"n_items": 1},
        "members": [{"path": str((run_dir / "RUN_CONTRACT.json").resolve()), "role": "pilot"}],
    }))
    result = experiment.union(exp)
    assert not result["member_errors"], (
        f"union() raised for module='epistemic': {result['member_errors']}"
    )
    assert result["units"], "no units in union result — epistemic alias not processed"
    # Module name in union output must be canonical "epis"
    unit_record = next((u for u in result["units"] if u["unit_id"] == unit_id), None)
    assert unit_record is not None, f"unit {unit_id!r} missing from union"
    assert unit_record["module"] == "epis", (
        f"expected module='epis', got {unit_record['module']!r}"
    )


def test_union_truly_unknown_module_captured_in_member_errors(tmp_path):
    """A contract with a genuinely unknown module produces a member_error, not a crash."""
    run_dir = tmp_path / "unknown_mod"
    run_dir.mkdir()
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "xray-run",
        "modules": [{"module": "xray", "expected_units": [
            {"unit_id": "xray:m:item0", "expected_transcript_path": "t.json", "planned_turns": 1}
        ]}],
        "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]},
    }))
    (run_dir / "t.json").write_text(json.dumps({"turns": [1], "completed": True}))
    (run_dir / "RUN_STATUS.json").write_text(json.dumps(
        {"attempt_number": 1, "started_at": "2026-07-20T12:00:00Z"}))
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "xray-test",
        "title": "t",
        "instrument": {"modules": ["xray"], "hashes": {}},
        "conditions": [],
        "target": {"n_items": 1},
        "members": [{"path": str((run_dir / "RUN_CONTRACT.json").resolve()), "role": "pilot"}],
    }))
    result = experiment.union(exp)
    assert result["member_errors"], "expected member_errors for truly unknown module 'xray'"
