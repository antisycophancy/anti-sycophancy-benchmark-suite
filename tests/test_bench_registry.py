import json
from pathlib import Path

import pytest

from suite_tools import bench
from suite_tools import assert_hash_panel as ahp
from suite_tools.live_dashboard import DISPOSITION_SCHEMA_VERSION
from suite_tools.run_contract import legacy_v1_provenance_hashes, legacy_v3_provenance_hashes


def _run(root: Path, name: str, module: str, *, attempt: int = 1,
         halt_on: int | None = None) -> Path:
    """halt_on = the attempt_number that emitted an action=halt event, or None."""
    d = root / name
    d.mkdir(parents=True)
    (d / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1", "run_id": name,
        "modules": [{"module": module, "expected_units": []}],
        "identity": {"sample_spec": {"item_indices": [0, 1, 2]}, "model_conditions": []},
    }))
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "schema_version": "benchmark-run-status-v1", "attempt_number": attempt,
        "started_at": "2026-07-20T10:00:00Z"}))
    if halt_on is not None:
        # real runner event shape (aita runner.py:615 / epis :523 / sus :560)
        (d / "RUN_EVENTS.jsonl").write_text(json.dumps({
            "event": "attempt_failure_classified", "attempt_number": halt_on,
            "action": "halt", "evidence_class": "instrument_defect",
            "category": "adapter_integrity", "failure_reason": "malformed judge output"}) + "\n")
    return d


def _reject(run_dir: Path) -> None:
    # canonical disposition shape (live_dashboard.py:104-105,169-179)
    (run_dir / "RUN_DISPOSITION.json").write_text(json.dumps({
        "schema_version": DISPOSITION_SCHEMA_VERSION,
        "disposition": "rejected_from_analysis", "reason": "operator"}))


def test_runs_scan_prunes_archive_symlink_and_rejected(tmp_path):
    root = tmp_path / "results"
    _run(root, "good", "aita")
    _run(root, "_archive_old", "aita")
    _reject(_run(root, "rejected_one", "sus"))
    (root / "_archive_link").symlink_to(root / "good", target_is_directory=True)
    out = bench.runs(roots=[root])
    assert {r["run_id"] for r in out["runs"]} == {"good"}
    assert out["schema_version"] == "benchmark-registry-view-v1"


def test_wrong_schema_disposition_does_not_reject(tmp_path):
    root = tmp_path / "results"
    r = _run(root, "keep", "aita")
    (r / "RUN_DISPOSITION.json").write_text(json.dumps({
        "schema_version": "something-else", "disposition": "rejected_from_analysis"}))
    out = bench.runs(roots=[root])
    assert "keep" in {row["run_id"] for row in out["runs"]}   # schema mismatch -> kept


def test_default_roots_include_suite_roots_and_top_level():
    roots = bench.default_roots()
    # Identify the top-level root by path, not by a substring check on the
    # parent directory name: the repo directory itself contains "bench"
    # (benchmark, the suite checkout), which made the old
    # heuristic pass only in checkouts whose directory happened not to.
    assert (bench.REPO_ROOT / "results").resolve() in {p.resolve() for p in roots}
    for suite_dir in ("sus-bench", "aita-bench", "epistemic-sycophancy-bench"):
        assert any(suite_dir in str(p) and p.name == "results" for p in roots)
    assert len(roots) == len({p.resolve() for p in roots})   # de-duplicated


def test_runs_scan_captures_malformed_json_as_warning(tmp_path):
    root = tmp_path / "results"
    bad = root / "broken"
    bad.mkdir(parents=True)
    (bad / "RUN_CONTRACT.json").write_text("{ not json")
    out = bench.runs(roots=[root])
    assert any("broken" in w["path"] for w in out["scan_warnings"])
    assert "runs" in out                             # scan did not abort


def test_blockers_use_real_event_and_latest_attempt(tmp_path):
    root = tmp_path / "results"
    _run(root, "ok", "aita")
    _run(root, "stuck", "epis", attempt=1, halt_on=1)     # latest attempt halted
    _run(root, "resolved", "sus", attempt=2, halt_on=1)   # halt only on superseded attempt 1
    out = bench.blockers(roots=[root])
    assert {b["run_id"] for b in out["blockers"]} == {"stuck"}
    b = out["blockers"][0]
    assert b["action"] == "halt" and b["category"] == "adapter_integrity"
    assert b["failure_reason"]


def test_blockers_ignore_halt_from_completed_earlier_stage_with_reused_attempt_number(tmp_path):
    root = tmp_path / "results"
    run = _run(root, "completed", "aita", attempt=1, halt_on=1)
    status = json.loads((run / "RUN_STATUS.json").read_text())
    status.update({"stage": "scoring", "status": "completed", "validity": "score_ready"})
    (run / "RUN_STATUS.json").write_text(json.dumps(status))

    events_path = run / "RUN_EVENTS.jsonl"
    event = json.loads(events_path.read_text())
    event["stage"] = "generation"
    events_path.write_text(json.dumps(event) + "\n")

    out = bench.blockers(roots=[root])
    assert out["blockers"] == []


def test_status_alias_delegates_to_owed_units(tmp_path):
    d = _run(tmp_path, "r", "aita")
    out = bench.status(d)
    assert out["schema_version"] == "benchmark-owed-units-v1"


def test_verify_two_dirs_reports_hash_and_item_universe(tmp_path):
    a = _run(tmp_path / "results", "a", "aita")
    b = _run(tmp_path / "results", "b", "aita")
    out = bench.verify([a, b])
    assert set(out["hash_certificate"]["match"]) >= {
        "benchmark_spec_hash", "sample_condition_hash", "judge_panel_hash"}
    assert out["item_universe"]["match"] is True     # identical item_indices


def test_verify_reports_pre_receipt_direct_openai_conformance_failure(tmp_path):
    run_dir = _run(tmp_path / "results", "gpt", "aita")
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["expected_models"] = [
        {
            "key": "gpt-low",
            "model_id": "gpt-5.6-sol",
            "condition_id": "gpt-5-6-sol-openai-native-low",
            "endpoint": "openai_responses",
            "request_options": {
                "max_tokens": 128000,
                "reasoning_effort": "low",
            },
        }
    ]
    contract_path.write_text(json.dumps(contract))

    out = bench.verify([run_dir])

    assert out["clean"] is False
    assert out["request_conformance"]["conformant"] is False
    assert out["request_conformance"]["issues"][0]["kind"] == "missing_request_receipt"


def test_verify_reports_saved_transcript_identity_failure(tmp_path):
    run_dir = _run(tmp_path / "results", "identity-gap", "sus")
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["identity"]["model_conditions"] = [{
        "key": "m-high",
        "condition_id": "m-high",
        "condition_hash": "sha256:m-high",
    }]
    contract["modules"][0]["expected_units"] = [{
        "unit_id": "sus:m-high:bridge:run1",
        "model_key": "m-high",
        "expected_transcript_path": "transcripts/unit.json",
    }]
    contract_path.write_text(json.dumps(contract))
    transcript = run_dir / "transcripts" / "unit.json"
    transcript.parent.mkdir()
    transcript.write_text(json.dumps({"completed": True}))

    out = bench.verify([run_dir])

    assert out["clean"] is False
    assert out["artifact_identity"]["conformant"] is False
    assert {issue["kind"] for issue in out["artifact_identity"]["issues"]} == {
        "missing_condition_id",
        "missing_condition_hash",
    }


def test_verify_recomputes_unversioned_legacy_panel_without_rewriting_contract(tmp_path):
    run_dir = _run(tmp_path / "results", "legacy", "aita")
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["provenance"] = legacy_v1_provenance_hashes(contract)
    contract_path.write_text(json.dumps(contract))

    before = contract_path.read_bytes()
    out = bench.verify([run_dir])

    assert out["stored_projection_version"] == "benchmark-identity-projection-v1"
    assert out["verification_projection_supported"] is True
    assert out["drift"] == {}
    assert out["current_projection"]["projection_version"] == (
        "benchmark-identity-projection-v4"
    )
    assert out["clean"] is True
    assert contract_path.read_bytes() == before


def test_verify_recomputes_projection_v3_panel_without_rewriting_contract(tmp_path):
    run_dir = _run(tmp_path / "results", "legacy-v3", "aita")
    contract_path = run_dir / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["provenance"] = legacy_v3_provenance_hashes(contract)
    contract_path.write_text(json.dumps(contract))

    before = contract_path.read_bytes()
    out = bench.verify([run_dir])

    assert out["stored_projection_version"] == "benchmark-identity-projection-v3"
    assert out["verification_projection_supported"] is True
    assert out["drift"] == {}
    assert out["current_projection"]["projection_version"] == "benchmark-identity-projection-v4"
    assert out["clean"] is True
    assert contract_path.read_bytes() == before


def test_item_universe_report_is_structured_and_silent(capsys):
    ca = {"modules": [{"module": "aita"}],
          "identity": {"sample_spec": {"item_indices": [0, 1, 2]}}}
    cb = {"modules": [{"module": "aita"}],
          "identity": {"sample_spec": {"items": [{"item_idx": 0}, {"item_idx": 1},
                                                 {"item_idx": 2}]}}}
    rep = ahp.item_universe_report(ca, cb)
    assert rep["match"] is True and rep["module"] == "aita"
    assert capsys.readouterr().out == ""             # no printing


def test_verify_never_calls_assert_comparison_identity(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(ahp, "assert_comparison_identity",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    a = _run(tmp_path / "results", "a", "aita")
    b = _run(tmp_path / "results", "b", "aita")
    bench.verify([a, b])
    assert called["n"] == 0                          # compare_provenance + item_universe only


def _exp(root: Path, name: str) -> Path:
    """Create a minimal valid EXPERIMENT.json directory."""
    d = root / name
    d.mkdir(parents=True)
    import json as _json
    (d / "EXPERIMENT.json").write_text(_json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": name,
        "members": [],
    }))
    return d


def test_experiments_prunes_archive_and_symlink(tmp_path):
    root = tmp_path / "experiments"
    _exp(root, "alpha")
    _exp(root, "beta")
    _exp(root, "_archive_old")
    # symlink to alpha — should be skipped
    (root / "_link_alpha").symlink_to(root / "alpha", target_is_directory=True)
    out = bench.experiments(roots=[root])
    found_ids = {e["experiment_id"] for e in out["experiments"]}
    assert found_ids == {"alpha", "beta"}
    assert out["schema_version"] == "benchmark-registry-view-v1"


def test_experiments_result_is_sorted(tmp_path):
    root = tmp_path / "experiments"
    _exp(root, "z_exp")
    _exp(root, "a_exp")
    _exp(root, "m_exp")
    out = bench.experiments(roots=[root])
    ids = [e["experiment_id"] for e in out["experiments"]]
    assert ids == sorted(ids)
