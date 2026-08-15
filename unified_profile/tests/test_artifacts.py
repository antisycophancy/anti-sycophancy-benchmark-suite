import hashlib
from pathlib import Path

from unified_profile.artifacts import discover_aita_runs, discover_all_runs, discover_epis_runs, discover_sus_runs
from unified_profile.manifest import artifact_ref, repo_relative_path, sha256_file


FIXTURES = Path(__file__).parent / "fixtures"


def test_artifact_ref_uses_relative_path_and_checksum():
    path = FIXTURES / "sus_conversations.json"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    ref = artifact_ref(path, "conversation")

    assert ref.path == repo_relative_path(path)
    assert ref.kind == "conversation"
    assert ref.sha256 == expected
    assert sha256_file(path) == expected


def test_discover_sus_runs_fixture():
    manifests = discover_sus_runs(FIXTURES / "sus_conversations.json")

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.module == "sus"
    assert manifest.quality == "test"
    assert manifest.models == ["anthropic/claude-opus-4.6"]
    assert manifest.n_conversations == 2
    assert manifest.n_scores == 2
    assert manifest.artifacts[0].sha256


def test_discover_aita_run_fixture():
    manifests = discover_aita_runs(FIXTURES / "aita_run")

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.module == "aita"
    assert manifest.quality == "test"
    assert manifest.models == ["anthropic/claude-opus-4.6"]
    assert manifest.n_conversations == 1
    assert manifest.n_scores == 1


def test_discover_epis_smoke_quality(tmp_path):
    run_dir = tmp_path / "smoke-fixture" / "delusion"
    run_dir.mkdir(parents=True)
    (run_dir / "model_item0_delusion_scores.json").write_text(
        '{"model":"google/gemini-3-flash-preview","item_idx":0,"judge_model":"judge"}'
    )
    (run_dir / "model_item0_delusion_side_a.json").write_text(
        '{"model":"google/gemini-3-flash-preview","item_idx":0,"seeker_model":"seeker","turns":[]}'
    )

    manifests = discover_epis_runs(tmp_path / "smoke-fixture")

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.module == "epistemic"
    assert manifest.quality == "smoke"
    assert manifest.n_conversations == 1
    assert manifest.n_scores == 1


def test_discover_research_aita_run_with_multiple_scores(tmp_path):
    path = tmp_path / "research-aita-run"
    path.mkdir()
    for idx in range(10):
        (path / f"model_item{idx}_scores.json").write_text('{"outcome": 1}\n')

    manifests = discover_aita_runs(path)

    assert len(manifests) == 1
    assert manifests[0].quality == "research"
    assert manifests[0].n_scores >= 10


def test_discover_all_runs_combines_modules():
    manifests = discover_all_runs(
        {
            "sus": [FIXTURES / "sus_conversations.json"],
            "aita": [FIXTURES / "aita_run"],
            "epistemic": [],
        }
    )

    assert [manifest.module for manifest in manifests] == ["aita", "sus"]
