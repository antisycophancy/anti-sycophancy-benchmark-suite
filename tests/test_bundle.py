import json
from pathlib import Path

import pytest

from suite_tools import bundle
from suite_tools import artifact_privacy as ap

CONDITION = {
    "key": "m",
    "condition_id": "cond-m",
    "condition_hash": "sha256:cond-m",
    "route_hash": "sha256:route-m",
}
ARTIFACT_IDENTITY = {
    "condition_id": CONDITION["condition_id"],
    "condition_hash": CONDITION["condition_hash"],
}


def _write_contract(path: Path, contract: dict) -> None:
    identity = contract.setdefault("identity", {})
    conditions = identity.setdefault("model_conditions", [dict(CONDITION)])
    for condition in conditions:
        if isinstance(condition, dict):
            condition.setdefault("condition_id", CONDITION["condition_id"])
            condition.setdefault("condition_hash", CONDITION["condition_hash"])
            condition.setdefault("route_hash", CONDITION["route_hash"])
    for module in contract.get("modules") or []:
        for unit in module.get("expected_units") or []:
            if isinstance(unit, dict):
                unit.setdefault("model_key", "m")
    contract["provenance"] = bundle.provenance_hashes(contract)
    path.write_text(json.dumps(contract))


def _experiment(tmp_path) -> Path:
    # reuse the union fixture helpers pattern: pilot + expansion, one collision,
    # one terminal unit, plus a sentinel transcript string that must NOT leak.
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    def mk(name, units, started_at, sentinel=None):
        d = runs / name
        d.mkdir()
        _write_contract(d / "RUN_CONTRACT.json", {
            "schema_version": "benchmark-run-contract-v1", "run_id": name,
            "source_command": "python -m x --api-key SEKRET",   # must be dropped
            "modules": [{"module": "aita", "expected_units":
                         [{"unit_id": u, "expected_score_path": f"{u}.json",
                           "planned_turns": 1} for u in units]}],
            "identity": {"execution": {"host_path": "/Users/me/secret"},  # dropped
                         "sample_spec": {"item_indices": [0]},
                         "model_conditions": [{"key": "m", "condition_id": "cond-m"}]},
        })
        (d / "RUN_STATUS.json").write_text(json.dumps(
            {"attempt_number": 1, "started_at": started_at, "status": "completed"}))
        item_score = {"verdict_alignment_a_majority": True, "verdict_alignment_a": 1.0}
        (d / "FINAL_RESULTS.json").write_text(json.dumps({"scores": {"m_item0": item_score}}))
        for u in units:                              # complete transcript + matched record -> scored
            (d / f"{u}.json").write_text(json.dumps({
                "completed": True, "turns": [{"model_response": "x"}], **item_score,
                **ARTIFACT_IDENTITY,
                "conversation": [{"role": "user", "content": sentinel or "hi"}]}))
        return d
    pilot = mk("pilot", ["aita:m:item0:a"], "2026-07-20T12:00:00Z",
               sentinel="XYZZY_SENTINEL_USER_MSG")
    expa = mk("expansion", ["aita:m:item0:a"], "2026-07-19T09:00:00Z")
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1", "experiment_id": "exp1",
        "title": "t", "instrument": {"modules": ["aita"], "hashes": {}},
        "conditions": [], "target": {"n_items": 1},
        "members": [
            {"path": str((pilot / "RUN_CONTRACT.json").resolve()), "role": "pilot"},
            {"path": str((expa / "RUN_CONTRACT.json").resolve()), "role": "expansion"},
        ]}))
    return exp


def test_bundle_has_no_absolute_paths_or_sentinel(tmp_path):
    exp = _experiment(tmp_path)
    out_dir = tmp_path / "out"
    result = bundle.emit(exp, out_dir=out_dir)
    bundle_dir = Path(result["bundle_dir"])
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert "/Users/" not in blob and "/home/" not in blob
    assert "XYZZY_SENTINEL_USER_MSG" not in blob          # default: no transcripts
    assert "SEKRET" not in blob                           # source_command dropped
    # members reduced to bundle-local ids
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    assert [m["member_id"] for m in manifest["members"]] == ["m1", "m2"]
    assert all("path" not in m for m in manifest["members"])
    assert manifest["payload_files"]
    assert bundle.audit_bundle_integrity(bundle_dir) == []
    assert bundle.audit_bundle_provenance(bundle_dir) == []


def test_bundle_revalidates_manifest_id_before_touching_outside_path(tmp_path):
    exp = _experiment(tmp_path)
    manifest_path = exp / "EXPERIMENT.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["experiment_id"] = "x/../../victim"
    manifest_path.write_text(json.dumps(manifest))
    outside_staging = tmp_path / "victim-v1.tmp"
    outside_staging.mkdir()
    sentinel = outside_staging / "keep.txt"
    sentinel.write_text("do not remove")

    with pytest.raises(ValueError, match="experiment_id must be"):
        bundle.emit(exp, out_dir=tmp_path / "out")

    assert sentinel.read_text() == "do not remove"


def test_bundle_integrity_detects_payload_mutation_and_unlisted_file(tmp_path):
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    scores_path = bundle_dir / "data" / "scores.jsonl"
    scores_path.write_text(scores_path.read_text() + "\n")
    (bundle_dir / "data" / "extra.jsonl").write_text("{}\n")

    issues = bundle.audit_bundle_integrity(bundle_dir)
    assert any(issue.path == "data/scores.jsonl" and "mismatch" in issue.reason for issue in issues)
    assert any(issue.path == "data/extra.jsonl" and issue.reason == "unlisted payload file" for issue in issues)


def test_rehashed_empty_bundle_fails_provenance_audit(tmp_path):
    """A hand-edited bundle cannot turn into valid evidence by rehashing it."""
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    outcomes_path = bundle_dir / "data" / "outcomes.jsonl"

    manifest = json.loads(manifest_path.read_text())
    manifest["union"]["units"] = []
    outcomes_path.write_text("")
    manifest["payload_files"] = bundle._payload_hash_entries(bundle_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    assert bundle.audit_bundle_integrity(bundle_dir) == []
    issues = bundle.audit_bundle_provenance(bundle_dir)
    assert any(issue.reason == "bundle union units missing" for issue in issues)
    assert any(issue.reason == "bundle outcomes empty" for issue in issues)


def test_absolute_home_path_is_detected_by_privacy_scan():
    issues = ap.scan_public_artifact_payload({"x": "/Users/me/secret/run"})
    assert any(i.reason == "absolute home path" for i in issues)


def _experiment_with_path_fields(tmp_path) -> Path:
    """Experiment whose contract carries absolute output_dir + selection paths.

    Mirrors a real epis/epistemic contract that has:
    - modules[*].output_dir = /Users/.../epis
    - modules[*].selection  = /Users/.../data/selection.yaml
    - identity.sample_spec.selection = /Users/.../data/selection.yaml
    The bundle projection must sanitize all three without mutating the source.
    """
    d = tmp_path / "runs" / "epis_run"
    d.mkdir(parents=True)
    abs_output_dir = "/home/ci-runner/benchmark/results/epis"
    abs_selection  = "/home/ci-runner/benchmark/epistemic-sycophancy-bench/data/selection.yaml"
    unit = "epis:m:pickside:item0:side_a"
    _write_contract(d / "RUN_CONTRACT.json", {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "epis_run",
        "source_command": "python -m x",
        "modules": [{
            "module": "epis",
            "output_dir": abs_output_dir,          # must be dropped
            "selection": abs_selection,             # must be basename-only
            "expected_units": [{
                "unit_id": unit,
                "side": "side_a",
                "test_type": "pickside",
                "item_idx": 0,
                "planned_turns": 2,
                "expected_transcript_path": "t_side_a.json",
                "expected_score_path": "s.json",
            }],
        }],
        "identity": {
            "execution": {"host_path": abs_output_dir},
            "sample_spec": {
                "selection": abs_selection,         # must be basename-only
                "test_types": ["pickside"],
            },
            "model_conditions": [{"key": "m", "condition_id": "cond-m"}],
        },
    })
    (d / "RUN_STATUS.json").write_text(json.dumps(
        {"attempt_number": 1, "started_at": "2026-07-20T12:00:00Z", "status": "completed"}))
    score = {"persistence": 1, "integrity": 2, "consistency": 1, "amplification": 0}
    (d / "FINAL_RESULTS.json").write_text(json.dumps({"scores": {"m_item0_pickside": score}}))
    (d / "t_side_a.json").write_text(json.dumps(
        {"completed": True, **ARTIFACT_IDENTITY,
         "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
    (d / "s.json").write_text(json.dumps({"completed": True, **score}))
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1",
        "experiment_id": "epis_path_test",
        "title": "t",
        "instrument": {"modules": ["epis"], "hashes": {}},
        "conditions": [],
        "target": {"n_items": 1},
        "members": [{"path": str((d / "RUN_CONTRACT.json").resolve()), "role": "pilot"}],
    }))
    return exp


def test_projected_contract_sanitizes_absolute_path_fields(tmp_path):
    """output_dir must be absent; selection paths must be basename only.

    Asserts:
    - The emitted RUN_CONTRACT-m1.json has no /Users/ substring.
    - output_dir is not present in any projected module.
    - selection field is reduced to basename (no leading '/').
    - The privacy gate passes (bundle.emit completes without raising).
    """
    exp = _experiment_with_path_fields(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    prov_text = (bundle_dir / "provenance" / "RUN_CONTRACT-m1.json").read_text()
    prov = json.loads(prov_text)

    # Privacy: no absolute home paths anywhere in the emitted contract
    assert "/Users/" not in prov_text, "absolute home path leaked into projected contract"

    # output_dir must be absent from every projected module entry
    for mod in prov.get("modules", []):
        assert "output_dir" not in mod, f"output_dir leaked into projected module: {mod}"

    # selection must be present but basename-only (no leading '/')
    for mod in prov.get("modules", []):
        sel = mod.get("selection")
        if sel is not None:
            assert not str(sel).startswith("/"), f"absolute selection in module: {sel}"
    # Also check identity.sample_spec.selection
    ss_sel = prov.get("identity", {}).get("sample_spec", {}).get("selection")
    if ss_sel is not None:
        assert not str(ss_sel).startswith("/"), f"absolute selection in sample_spec: {ss_sel}"

    # Full bundle tree must have zero /Users/ occurrences
    blob = "\n".join(p.read_text() for p in bundle_dir.rglob("*") if p.is_file())
    assert "/Users/" not in blob, "absolute home path leaked into bundle tree"


def test_projected_contract_drops_execution_and_source_command(tmp_path):
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    prov = json.loads((bundle_dir / "provenance" / "RUN_CONTRACT-m1.json").read_text())
    assert "source_command" not in prov
    assert "execution" not in prov.get("identity", {})
    assert "modules" in prov and "identity" in prov          # kept for recompute


def test_projected_contract_removes_private_model_routing_metadata():
    contract = {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "private-route",
        "modules": [{
            "module": "sus",
            "expected_units": [{
                "unit_id": "sus:th-model:bridge:run1",
                "model_key": "th-model",
                "model_id": "therapeutic-harness/th-model",
                "scenario": "bridge",
                "service_ref": "customer-unit-service-17",
            }],
            "deployment_alias": "customer-module-prod-17",
            "expected_artifacts": [{
                "kind": "final_results",
                "path": "FINAL_RESULTS.json",
                "required_for": "promotion",
                "service_ref": "customer-artifact-service-17",
            }],
            "dataset_manifest": {
                "files": [{
                    "path": "/Volumes/Research/customer/private-source.csv",
                    "sha256": "source-sha256",
                }],
            },
        }],
        "identity": {
            "benchmark_family_id": "sus",
            "benchmark_spec": {"module": "sus"},
            "sample_spec": {"scenario_ids": ["bridge"]},
            "judge_panel": {
                "panel": ["judge/model"],
                "primary_config": {
                    "base_url": "https://customer.example.com/v1",
                    "api_key_env": "CUSTOMER_JUDGE_API_KEY",
                    "label": "customer_judge_prod_a17",
                },
                "configs": [{
                    "model_id": "judge/model",
                    "provider_api": "openai_compatible",
                    "route_hash": "sha256:judge-route",
                    "profile_id": "judge-profile-private-id",
                    "profile_hash": "sha256:judge-profile-a",
                    "served_profile_id": "judge-served-private-id",
                    "served_profile_hash": "sha256:judge-served-a",
                    "base_url": "https://customer.example.com/v1",
                    "api_key_env": "CUSTOMER_JUDGE_API_KEY",
                    "label": "customer_judge_prod_a17",
                    "condition_metadata": {
                        "effort": "high",
                        "deployment_uuid": "judge-deployment-private-17",
                    },
                }],
            },
            "model_conditions": [{
                "key": "th-model",
                "model_id": "therapeutic-harness/th-model",
                "condition_id": "th-model",
                "condition_hash": "condition-sha256:abc",
                "route_hash": "route-sha256:def",
                "provider_api": "openai_compatible",
                "endpoint": "customer_prod_a17",
                "base_url": "https://model.customer.example.com/v1",
                "api_key_env": "CUSTOMER_MODEL_API_KEY",
                "deployment_uuid": "model-deployment-private-17",
                "service_id": "customer-model-service-private-17",
                "profile_id": "model-profile-private-id",
                "unknown_extension": "opaque-private-extension",
                "source": "/Users/operator/benchmark/generated/frontier/sus-models.yaml",
                "provider_version": "private-backend-build-123",
                "served_profile_hash": "sha256:abc",
                "condition_metadata": {
                    "adapter_profile": "private_served_endpoint",
                    "route_mapping": "exact_backend_config",
                    "deployment_uuid": "deployment-17-private",
                    "source_official_model_id": "openai/model",
                },
            }],
            "execution": {"cwd": "/Users/operator/benchmark"},
        },
    }
    contract["provenance"] = bundle.provenance_hashes(contract)

    projected = bundle._project_contract_for_bundle(contract)
    text = json.dumps(projected)
    condition = projected["identity"]["model_conditions"][0]
    assert "localhost" not in text
    assert "/Users/" not in text
    assert "private-backend" not in text
    assert "customer_prod_a17" not in text
    assert "deployment-17-private" not in text
    assert "private-source.csv" not in text
    assert "customer-module-prod-17" not in text
    assert "customer-artifact-service-17" not in text
    assert "customer-unit-service-17" not in text
    assert "customer.example.com" not in text
    assert "CUSTOMER_JUDGE_API_KEY" not in text
    assert "customer_judge_prod_a17" not in text
    assert "judge-deployment-private-17" not in text
    assert "judge-profile-private-id" not in text
    assert "judge-served-private-id" not in text
    assert "CUSTOMER_MODEL_API_KEY" not in text
    assert "model-deployment-private-17" not in text
    assert "customer-model-service-private-17" not in text
    assert "model-profile-private-id" not in text
    assert "opaque-private-extension" not in text
    assert "endpoint" not in condition
    assert condition["key"] == "th-model"
    assert condition["model_id"] == "therapeutic-harness/th-model"
    assert condition["route_hash"] == "route-sha256:def"
    assert "route_mapping" not in condition["condition_metadata"]
    assert condition["condition_metadata"]["source_official_model_id"] == "openai/model"
    assert condition["served_profile_hash"] == "sha256:abc"
    judge_config = projected["identity"]["judge_panel"]["configs"][0]
    assert judge_config["model_id"] == "judge/model"
    assert judge_config["route_hash"] == "sha256:judge-route"
    assert judge_config["profile_hash"] == "sha256:judge-profile-a"
    assert judge_config["served_profile_hash"] == "sha256:judge-served-a"
    assert judge_config["condition_metadata"] == {"effort": "high"}
    assert "dataset_manifest" not in projected["modules"][0]

    other_contract = json.loads(json.dumps(contract))
    other_contract["identity"]["judge_panel"]["configs"][0][
        "profile_hash"
    ] = "sha256:judge-profile-b"
    other_contract["provenance"] = bundle.provenance_hashes(other_contract)
    other_projected = bundle._project_contract_for_bundle(other_contract)
    assert (
        other_projected["provenance"]["judge_panel_hash"]
        != projected["provenance"]["judge_panel_hash"]
    )


def test_bundle_projects_private_score_route_to_public_adapter_label(tmp_path):
    score_rows = bundle._scores_union(
        [{"unit_id": "aita:m:item0:side_a", "chosen_member": "source"}],
        {"source": {"rows": [{
            "unit_id": "aita:m:item0:side_a",
            "condition": {
                "route": "http://localhost:9999/v1",
                "profile": {"profile_id": "therapeutic-harness"},
            },
        }]}},
        {"source": "m1"},
    )

    assert score_rows[0]["condition"]["route"] == "profile_adapter"
    assert "localhost" not in json.dumps(score_rows)


def test_bundle_projects_opaque_score_route_to_public_category(tmp_path):
    score_rows = bundle._scores_union(
        [{"unit_id": "aita:m:item0:side_a", "chosen_member": "source"}],
        {"source": {"rows": [{
            "unit_id": "aita:m:item0:side_a",
            "condition": {
                "route": "customer_prod_a17",
                "profile": {"profile_id": "deployment-profile-private"},
            },
        }]}},
        {"source": "m1"},
    )

    assert score_rows[0]["condition"]["route"] == "profile_adapter"
    assert score_rows[0]["condition"]["profile"] is None
    assert "customer_prod_a17" not in json.dumps(score_rows)
    assert "deployment-profile-private" not in json.dumps(score_rows)


def test_member_score_rows_fails_closed_when_score_projection_breaks(tmp_path, monkeypatch):
    def broken_score_rows(_run_dir):
        raise RuntimeError("broken score artifact")

    monkeypatch.setattr(bundle, "compute_score_rows", broken_score_rows)

    with pytest.raises(ValueError, match="could not derive score rows for bundle member m1"):
        bundle._member_score_rows({"member-path": tmp_path}, {"member-path": "m1"})


def test_certificate_uses_compare_provenance_and_item_universe(tmp_path):
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    cert = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())["certificate"]
    assert "match" in cert["pairwise"][0]                    # compare_provenance shape
    assert "item_universe" in cert["pairwise"][0]


def test_union_winner_drives_outcomes(tmp_path):
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    outcomes = [json.loads(l) for l in
                (bundle_dir / "data" / "outcomes.jsonl").read_text().splitlines() if l]
    dup = next(o for o in outcomes if o["unit_id"] == "aita:m:item0:a")
    assert dup["member_id"] == "m1"                          # pilot started later, wins


def test_derived_aggregates_use_only_union_winners(tmp_path):
    # EPIS pilot (started later -> wins) and expansion (loses) provide the SAME
    # unit with DIFFERENT sycophancy inputs; the experiment aggregate must reflect
    # ONLY the winner — collision losers cannot contaminate it (Sol round-2 B3).
    runs = tmp_path / "runs"
    runs.mkdir()

    def mk(name, started_at, persistence):
        d = runs / name
        d.mkdir()
        unit = "epis:m:pickside:item0:side_a"
        _write_contract(d / "RUN_CONTRACT.json", {
            "schema_version": "benchmark-run-contract-v1", "run_id": name,
            "modules": [{"module": "epis", "expected_units": [
                {"unit_id": unit, "side": "side_a", "test_type": "pickside",
                 "item_idx": 0, "planned_turns": 2,
                 "expected_transcript_path": "t_side_a.json",
                 "expected_score_path": "s.json"}]}],
            "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]}})
        (d / "RUN_STATUS.json").write_text(json.dumps(
            {"attempt_number": 1, "started_at": started_at, "status": "completed"}))
        score = {"persistence": persistence, "integrity": 2, "consistency": 1,
                 "amplification": 0}
        (d / "FINAL_RESULTS.json").write_text(
            json.dumps({"scores": {"m_item0_pickside": score}}))
        (d / "t_side_a.json").write_text(json.dumps(
            {"completed": True, **ARTIFACT_IDENTITY,
             "turns": [{"model_response": "a"}, {"model_response": "b"}]}))
        (d / "s.json").write_text(json.dumps({"completed": True, **score}))
        return d

    winner = mk("pilot", "2026-07-20T12:00:00Z", persistence=1)
    mk("expansion", "2026-07-19T09:00:00Z", persistence=0)   # loser, different input
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1", "experiment_id": "e", "title": "t",
        "instrument": {"modules": ["epis"], "hashes": {}}, "conditions": [],
        "target": {"n_items": 1},
        "members": [
            {"path": str((winner / "RUN_CONTRACT.json").resolve()), "role": "pilot"},
            {"path": str((runs / "expansion" / "RUN_CONTRACT.json").resolve()),
             "role": "expansion"}]}))
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    aggs = [json.loads(l) for l in
            (bundle_dir / "data" / "derived_aggregates.jsonl").read_text().splitlines() if l]
    syc = next(a for a in aggs if a["dimension"] == "epistemic_sycophancy_score")
    assert syc["n"] == 1                              # winner only, not 2 members pooled
    assert syc["helper"] == "epis_bench.report.compute_epistemic_sycophancy_score"


def test_poisoned_payload_aborts_atomically(tmp_path):
    exp = _experiment(tmp_path)
    # poison: inject an api_key field into a member score artifact
    runs = tmp_path / "runs"
    art = runs / "pilot" / "aita:m:item0:a.json"
    data = json.loads(art.read_text())
    # Construct the secret-looking value at runtime so the literal does not trip
    # release_audit's secret scan on this source file (repo convention).
    data["api_key"] = "sk-" + "a" * 24
    art.write_text(json.dumps(data))
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="privacy check failed"):
        bundle.emit(exp, out_dir=out_dir)
    # atomic staging: no partial bundle and no leftover tmp dir
    assert not any(out_dir.glob("bundle-*"))
    assert not any(out_dir.glob(".*.tmp"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_score_payload_aborts_bundle_atomically(tmp_path, value):
    exp = _experiment(tmp_path)
    artifact = tmp_path / "runs" / "pilot" / "aita:m:item0:a.json"
    payload = json.loads(artifact.read_text())
    payload["verdict_alignment_a"] = value
    artifact.write_text(json.dumps(payload))
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="non-finite numeric value"):
        bundle.emit(exp, out_dir=out_dir)

    assert not any(out_dir.glob("bundle-*"))
    assert not any(out_dir.glob(".*.tmp"))


def test_bundle_tree_audit_flags_nonfinite_json_and_csv(tmp_path):
    bundle_dir = tmp_path / "bundle"
    data_dir = bundle_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "scores.jsonl").write_text('{"score": NaN}\n')
    (data_dir / "scores.csv").write_text("score\nInfinity\n")

    issues = bundle.audit_bundle_tree(bundle_dir)

    assert any(
        issue.path.startswith("data/scores.jsonl")
        and issue.reason == "non-finite numeric value"
        for issue in issues
    )
    assert any(
        issue.path.startswith("data/scores.csv")
        and issue.reason == "non-finite numeric value"
        for issue in issues
    )


def test_version_auto_increments(tmp_path):
    exp = _experiment(tmp_path)
    out_dir = tmp_path / "out"
    b1 = Path(bundle.emit(exp, out_dir=out_dir)["bundle_dir"])
    b2 = Path(bundle.emit(exp, out_dir=out_dir)["bundle_dir"])
    assert b1.name.endswith("-v1") and b2.name.endswith("-v2")


def test_bundle_tree_audit_scans_every_file(tmp_path):
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    # plant a leak into an already-emitted bundle and confirm the tree audit sees it
    (bundle_dir / "data" / "leak.txt").write_text("visit https://box.internal/x")
    issues = bundle.audit_bundle_tree(bundle_dir)
    assert any("leak.txt" in i.path for i in issues)


def test_block_absolute_evidence_pointer_aborts_bundle(tmp_path):
    # Producers must write relative pointers only. A block on the WINNER (pilot/m1)
    # carrying an absolute run path in evidence_pointer must abort the bundle —
    # the pre-write gate + boundary-anchored home-path regex catch it.
    exp = _experiment(tmp_path)
    (tmp_path / "runs" / "pilot" / "BLOCKS.jsonl").write_text(json.dumps({
        "unit_id": "aita:m:item0:a", "evidence_class": "model_signal", "category": "X",
        "evidence_pointer": "/Users/me/runs/pilot/RUN_CONTRACT.json"}) + "\n")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="privacy check failed"):
        bundle.emit(exp, out_dir=out_dir)
    assert not any(out_dir.glob("bundle-*"))
    assert not any(out_dir.glob(".*.tmp"))


def test_bundle_default_emit_writes_html_report(tmp_path):
    # bundle.emit() with write_report=True (default) must produce report/index.html
    # inside the bundle directory as part of the atomic emission.
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    assert (bundle_dir / "report" / "index.html").exists()


def test_bundle_no_report_flag_skips_html(tmp_path):
    # write_report=False must suppress the HTML report while still producing
    # REPORT.md and all data files.
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out", write_report=False)["bundle_dir"])
    assert not (bundle_dir / "report" / "index.html").exists()


def test_blocks_scoped_to_union_winners_only(tmp_path):
    # _experiment: pilot (m1, started later -> winner) + expansion (m2, loser) both
    # provide aita:m:item0:a. Plant a stale block on the LOSER for that unit; it must
    # not reach the bundle because pilot won the unit (Sol round-3 finding 5).
    exp = _experiment(tmp_path)
    (tmp_path / "runs" / "expansion" / "BLOCKS.jsonl").write_text(json.dumps({
        "unit_id": "aita:m:item0:a", "evidence_class": "model_signal",
        "category": "LOSER_BLOCK"}) + "\n")
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    blocks_text = (bundle_dir / "data" / "blocks.jsonl").read_text()
    assert "LOSER_BLOCK" not in blocks_text          # loser's block filtered out


def _multi_suite_experiment(tmp_path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    exp = tmp_path / "exp"
    exp.mkdir()
    members = []
    for module, unit, artifact in [
        ("aita", "aita:m:item0:a", "aita:m:item0:a.json"),
        ("epis", "epis:m:pickside:item0:side_a", "m_item0_pickside_side_a.json"),
        ("sus", "sus:m:scen1:run1", "transcripts/scen1_run1.json"),
    ]:
        d = runs / module
        d.mkdir()
        _write_contract(d / "RUN_CONTRACT.json", {
            "schema_version": "benchmark-run-contract-v1", "run_id": module,
            "modules": [{"module": module, "expected_units":
                         [{"unit_id": unit, "expected_transcript_path": artifact}]}],
            "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]}})
        (d / "RUN_STATUS.json").write_text(json.dumps(
            {"attempt_number": 1, "started_at": "2026-07-20T00:00:00Z", "status": "completed"}))
        art_path = d / artifact
        art_path.parent.mkdir(parents=True, exist_ok=True)
        # A genuinely-completed artifact for each module's unit_state predicate so
        # the T7 publication gate (owed winning units block) does not refuse the
        # bundle: aita/epis need >=planned_turns (0 here); sus needs an "elicit"
        # phase.  The conversation sentinel is preserved for the transcript tests.
        art_path.write_text(json.dumps({
            "completed": True,
            **ARTIFACT_IDENTITY,
            "phases": {"elicit": {}},
            "conversation": [{"role": "user", "content": f"SENTINEL_{module}"}]}))
        members.append({"path": str((d / "RUN_CONTRACT.json").resolve()), "role": "pilot"})
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1", "experiment_id": "multi",
        "title": "t", "instrument": {"modules": ["aita", "epis", "sus"], "hashes": {}},
        "conditions": [], "target": {"n_items": 3}, "members": members}))
    return exp


def test_transcripts_only_appear_with_flag_all_three_suites(tmp_path):
    exp = _multi_suite_experiment(tmp_path)
    # default: no review.html, no sentinels
    b_off = Path(bundle.emit(exp, out_dir=tmp_path / "off")["bundle_dir"])
    assert not (b_off / "report" / "review.html").exists()
    off_blob = "\n".join(p.read_text() for p in b_off.rglob("*") if p.is_file())
    assert "SENTINEL_" not in off_blob
    m_off = json.loads((b_off / "BUNDLE_MANIFEST.json").read_text())
    assert m_off["contains_transcripts"] is False
    # with flag: review.html holds all three suites' sentinels, ONLY under review.html
    b_on = Path(bundle.emit(exp, out_dir=tmp_path / "on",
                            include_transcripts=True)["bundle_dir"])
    review = b_on / "report" / "review.html"
    assert review.exists()
    review_text = review.read_text()
    for module in ("aita", "epis", "sus"):
        assert f"SENTINEL_{module}" in review_text          # directories passed, not FINAL_RESULTS
    # sentinels appear nowhere else in the tree
    other = "\n".join(p.read_text() for p in b_on.rglob("*")
                      if p.is_file() and p != review)
    assert "SENTINEL_" not in other
    m_on = json.loads((b_on / "BUNDLE_MANIFEST.json").read_text())
    assert m_on["contains_transcripts"] is True
    assert "review.html" in (b_on / "report" / "index.html").read_text()


def test_sealed_aita_pack_forbids_raw_transcript_bundle(tmp_path):
    exp = _experiment(tmp_path)
    experiment = json.loads((exp / "EXPERIMENT.json").read_text())
    for member in experiment["members"]:
        contract_path = Path(member["path"])
        contract = json.loads(contract_path.read_text())
        contract["modules"][0]["dataset_manifest"] = {
            "schema_version": "aita-dataset-manifest-v1",
            "dataset_mode": "nta-paired",
            "distribution_mode": "sealed_public_pack",
            "sealed_pack": {
                "pack_id": "synthetic-sealed-pack",
                "plaintext_identity_sha256": "a" * 64,
            },
        }
        _write_contract(contract_path, contract)

    with pytest.raises(ValueError, match="sealed AITA.*raw transcripts"):
        bundle.emit(
            exp,
            out_dir=tmp_path / "blocked",
            include_transcripts=True,
        )

    assert not any((tmp_path / "blocked").glob("bundle-*"))

    with pytest.raises(ValueError, match="sealed AITA.*free-text review rationale"):
        bundle.emit(
            exp,
            out_dir=tmp_path / "blocked-rationale",
            include_review_rationale=True,
        )

    safe = Path(bundle.emit(exp, out_dir=tmp_path / "safe")["bundle_dir"])
    safe_payload = b"".join(path.read_bytes() for path in safe.rglob("*") if path.is_file())
    assert b"XYZZY_SENTINEL_USER_MSG" not in safe_payload


def test_transcript_bundle_still_blocks_secret_in_conversation(tmp_path):
    exp = _multi_suite_experiment(tmp_path)
    d = tmp_path / "runs" / "aita"
    art = d / "aita:m:item0:a.json"
    data = json.loads(art.read_text())
    # Runtime-constructed secret (broken literal keeps release_audit clean here).
    data["conversation"][0]["content"] = "here is " + "sk-" + "a" * 24
    art.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="secret|privacy"):
        bundle.emit(exp, out_dir=tmp_path / "leak", include_transcripts=True)
    assert not any((tmp_path / "leak").glob("bundle-*"))    # atomic: nothing left


def test_transcript_review_paths_are_member_relative(tmp_path):
    """review.html must use member-relative labels; the run-dir absolute prefix must be absent."""
    exp = _multi_suite_experiment(tmp_path)
    b = Path(bundle.emit(exp, out_dir=tmp_path / "out", include_transcripts=True)["bundle_dir"])
    review_text = (b / "report" / "review.html").read_text()
    # At least one member-relative label (aita member = m1) appears in the HTML.
    assert "m1/aita/" in review_text
    # The absolute tmp-path prefix must be absent from review.html.
    assert str(tmp_path) not in review_text


def test_transcript_review_blocks_absolute_home_path_in_field(tmp_path):
    """A '/Users/..' path injected into a metadata field must abort the pre-render scan."""
    exp = _multi_suite_experiment(tmp_path)
    d = tmp_path / "runs" / "aita"
    art = d / "aita:m:item0:a.json"
    data = json.loads(art.read_text())
    data["run_note"] = "/Users/someone/secret-run/"   # not a path field → not normalized away
    art.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="secret|privacy|absolute|home"):
        bundle.emit(exp, out_dir=tmp_path / "leak", include_transcripts=True)
    assert not any((tmp_path / "leak").glob("bundle-*"))    # atomic: nothing left


def _epis_refusal_experiment(tmp_path) -> Path:
    """EPIS experiment: side_a scored, side_b a terminal refusal (model_signal block)."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    d = runs / "epis"
    d.mkdir()
    units = [
        {"unit_id": "epis:m:mirror:item0:side_a", "side": "side_a",
         "test_type": "mirror", "item_idx": 0, "planned_turns": 2,
         "expected_transcript_path": "t_side_a.json", "expected_score_path": "s.json"},
        {"unit_id": "epis:m:mirror:item0:side_b", "side": "side_b",
         "test_type": "mirror", "item_idx": 0, "planned_turns": 2,
         "expected_transcript_path": "t_side_b.json", "expected_score_path": None},
    ]
    _write_contract(d / "RUN_CONTRACT.json", {
        "schema_version": "benchmark-run-contract-v1", "run_id": "epis",
        "modules": [{"module": "epis", "expected_units": units}],
        "identity": {"model_conditions": [{"key": "m", "condition_id": "cond-m"}]}})
    (d / "RUN_STATUS.json").write_text(json.dumps(
        {"attempt_number": 1, "started_at": "2026-07-20T00:00:00Z", "status": "completed"}))
    score = {"persistence": 1, "integrity": 2, "consistency": 1, "amplification": 0,
             "primary_failure": True, "endpoint_shift": True, "integrity_break": False,
             "side_inconsistency": False, "stance_amplification": False}
    (d / "FINAL_RESULTS.json").write_text(json.dumps({"scores": {"m_item0_mirror": score}}))
    conv = {
        "completed": True,
        **ARTIFACT_IDENTITY,
        "turns": [{"model_response": "a"}, {"model_response": "b"}],
    }
    (d / "t_side_a.json").write_text(json.dumps(conv))
    (d / "s.json").write_text(json.dumps({"completed": True, **score}))
    # side_b: no transcript + a model_signal block -> terminal refusal
    (d / "BLOCKS.jsonl").write_text(json.dumps({
        "unit_id": "epis:m:mirror:item0:side_b",
        "evidence_class": "model_signal", "category": "safety_refusal"}) + "\n")
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "EXPERIMENT.json").write_text(json.dumps({
        "schema_version": "benchmark-experiment-v1", "experiment_id": "epis-ref",
        "title": "t", "instrument": {"modules": ["epis"], "hashes": {}},
        "conditions": [], "target": {"n_items": 1},
        "members": [{"path": str((d / "RUN_CONTRACT.json").resolve()), "role": "pilot"}]}))
    return exp


def test_outcomes_agree_with_manifest_union_section(tmp_path):
    # Task 1: side_b's own terminal refusal must not be erased by mirroring
    # side_a's scored class, and the bundle outcomes must agree with the union.
    exp = _epis_refusal_experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    union_state = {u["unit_id"]: u["state"] for u in manifest["union"]["units"]}
    outcomes = [json.loads(l) for l in
                (bundle_dir / "data" / "outcomes.jsonl").read_text().splitlines() if l]
    outcome_by_uid = {o["unit_id"]: o["outcome_class"] for o in outcomes}

    sb = "epis:m:mirror:item0:side_b"
    assert outcome_by_uid[sb] == "terminal_model_signal"     # refusal survives to outcomes
    assert union_state[sb] == "terminal_model_signal"        # ...and to the union section
    assert outcome_by_uid["epis:m:mirror:item0:side_a"] == "scored"

    # General agreement: union state and outcome_class are consistent for every unit.
    agree = {"terminal_model_signal": {"terminal_model_signal"},
             "owed": {"missing"}, "done": {"scored", "unscored"}}
    for uid, state in union_state.items():
        assert outcome_by_uid[uid] in agree.get(state, {outcome_by_uid[uid]}), (
            f"union state {state} disagrees with outcome {outcome_by_uid[uid]} for {uid}")


def test_manifest_exclusion_policy_is_truthful_object(tmp_path):
    # Task 4: exclusion_policy is a self-describing object whose definition states
    # the implemented behavior exactly (no phantom id).
    exp = _experiment(tmp_path)
    bundle_dir = Path(bundle.emit(exp, out_dir=tmp_path / "out")["bundle_dir"])
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    policy = manifest["exclusion_policy"]
    assert policy["id"] == "responsive-subset-v1"
    definition = policy["definition"]
    assert "outcome_class=scored units only" in definition
    assert "declination rates" in definition
    assert "pending-scoring and excluded from both" in definition
