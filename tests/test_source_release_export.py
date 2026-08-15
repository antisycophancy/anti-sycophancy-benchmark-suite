import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from suite_tools.source_release import (
    SourceReleaseError,
    export_source_release,
    seed_public_root_from_export,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_release_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "release@example.invalid")
    _git(repo, "config", "user.name", "Release Test")
    (repo / ".gitignore").write_text("private/\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "antisycophancy-suite-tools"\nversion = "0.1.0"\n'
    )
    (repo / "README.md").write_text("public source\n")
    (repo / "CITATION.cff").write_text('version: "0.1.0"\n')
    package_specs = {
        "sus-bench": ("sus_bench", "sus-bench"),
        "aita-bench": ("aita_bench", "aita-bench"),
        "epistemic-sycophancy-bench": ("epis_bench", "epis-bench"),
    }
    for project, (package, cli_name) in package_specs.items():
        package_dir = repo / project / package
        package_dir.mkdir(parents=True)
        (repo / project / "pyproject.toml").write_text(
            f'[project]\nname = "{package}"\nversion = "0.1.0"\n'
        )
        (package_dir / "__init__.py").write_text('__version__ = "0.1.0"\n')
        (package_dir / "cli.py").write_text(
            f"from {package} import __version__\n"
            f'VERSION = f"{cli_name} {{__version__}}"\n'
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "release source")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_export_is_exact_audited_tree_with_checksums(tmp_path):
    repo, sha = _init_release_repo(tmp_path)
    private = repo / "private"
    private.mkdir()
    (private / "secret.txt").write_text("not exported")
    out = tmp_path / "export"

    receipt = export_source_release(
        repo_root=repo,
        revision=sha,
        out_dir=out,
        release_version="0.1.0",
    )

    files = sorted(
        path.relative_to(out).as_posix()
        for path in out.rglob("*")
        if path.is_file()
    )
    assert files == [
        ".gitignore",
        "CITATION.cff",
        "README.md",
        "SHA256SUMS",
        "aita-bench/aita_bench/__init__.py",
        "aita-bench/aita_bench/cli.py",
        "aita-bench/pyproject.toml",
        "epistemic-sycophancy-bench/epis_bench/__init__.py",
        "epistemic-sycophancy-bench/epis_bench/cli.py",
        "epistemic-sycophancy-bench/pyproject.toml",
        "pyproject.toml",
        "sus-bench/pyproject.toml",
        "sus-bench/sus_bench/__init__.py",
        "sus-bench/sus_bench/cli.py",
    ]
    assert not (out / ".git").exists()
    assert not (out / "private").exists()
    assert receipt["revision"] == sha
    assert receipt["release_version"] == "0.1.0"
    assert receipt["audit_issue_count"] == 0

    checksums = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        checksums[path] = digest
    assert "SHA256SUMS" not in checksums
    assert checksums["README.md"] == hashlib.sha256(
        (out / "README.md").read_bytes()
    ).hexdigest()


def test_audited_export_seeds_checksum_free_public_root_that_reexports(tmp_path):
    private_repo, private_sha = _init_release_repo(tmp_path)
    first_export = tmp_path / "first-export"
    public_root = tmp_path / "public-root"

    export_source_release(
        repo_root=private_repo,
        revision=private_sha,
        out_dir=first_export,
        release_version="0.1.0",
    )
    receipt = seed_public_root_from_export(
        export_root=first_export,
        out_dir=public_root,
        release_version="0.1.0",
    )

    assert receipt["checksum_included"] is False
    assert not (public_root / "SHA256SUMS").exists()
    assert (public_root / "README.md").read_text() == "public source\n"

    _git(public_root, "init", "-q")
    _git(public_root, "config", "user.email", "release@example.invalid")
    _git(public_root, "config", "user.name", "Release Test")
    _git(public_root, "add", ".")
    _git(public_root, "commit", "-qm", "public root")
    public_sha = _git(public_root, "rev-parse", "HEAD")

    second_export = tmp_path / "second-export"
    second_receipt = export_source_release(
        repo_root=public_root,
        revision=public_sha,
        out_dir=second_export,
        release_version="0.1.0",
    )
    assert second_receipt["revision"] == public_sha
    assert (second_export / "SHA256SUMS").is_file()


def test_public_root_seed_refuses_tampered_or_nonempty_inputs(tmp_path):
    private_repo, private_sha = _init_release_repo(tmp_path)
    release_export = tmp_path / "release-export"
    export_source_release(
        repo_root=private_repo,
        revision=private_sha,
        out_dir=release_export,
        release_version="0.1.0",
    )
    (release_export / "README.md").write_text("tampered\n")

    with pytest.raises(SourceReleaseError, match="audit failed"):
        seed_public_root_from_export(
            export_root=release_export,
            out_dir=tmp_path / "public-root",
            release_version="0.1.0",
        )

    (release_export / "README.md").write_text("public source\n")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n")
    with pytest.raises(SourceReleaseError, match="empty"):
        seed_public_root_from_export(
            export_root=release_export,
            out_dir=occupied,
            release_version="0.1.0",
        )
    assert (occupied / "keep.txt").read_text() == "keep\n"


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_export_refuses_dirty_selected_tree(tmp_path, dirty_kind):
    repo, sha = _init_release_repo(tmp_path)
    if dirty_kind == "tracked":
        (repo / "README.md").write_text("changed\n")
    else:
        (repo / "notes.txt").write_text("untracked\n")

    with pytest.raises(SourceReleaseError, match="working tree must be clean"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_export_refuses_nonempty_destination(tmp_path):
    repo, sha = _init_release_repo(tmp_path)
    out = tmp_path / "export"
    out.mkdir()
    (out / "existing.txt").write_text("keep")

    with pytest.raises(SourceReleaseError, match="empty"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=out,
            release_version="0.1.0",
        )
    assert (out / "existing.txt").read_text() == "keep"


def test_export_refuses_symlinks_and_submodule_modes(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    (repo / "linked").symlink_to("README.md")
    _git(repo, "add", "linked")
    _git(repo, "commit", "-qm", "add symlink")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="symlinks or submodules"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_export_requires_full_revision_and_matching_release_version(tmp_path):
    repo, sha = _init_release_repo(tmp_path)
    with pytest.raises(SourceReleaseError, match="full immutable commit"):
        export_source_release(
            repo_root=repo,
            revision=sha[:12],
            out_dir=tmp_path / "short",
            release_version="0.1.0",
        )
    with pytest.raises(SourceReleaseError, match="release version mismatch"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "mismatch",
            release_version="1.0.0",
        )


def test_export_fails_closed_when_extracted_tree_fails_release_audit(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    internal = repo / "internal"
    internal.mkdir()
    (internal / "operator.txt").write_text("private")
    _git(repo, "add", "internal/operator.txt")
    _git(repo, "commit", "-qm", "bad source")
    sha = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "export"

    with pytest.raises(SourceReleaseError, match="release audit failed"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=out,
            release_version="0.1.0",
        )
    assert not out.exists()


def test_export_refuses_tracked_checksum_manifest_name(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    (repo / "SHA256SUMS").write_text("tracked source bytes\n")
    _git(repo, "add", "SHA256SUMS")
    _git(repo, "commit", "-qm", "checksum collision")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="reserved SHA256SUMS"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_export_refuses_newline_bearing_paths(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    (repo / "line\nbreak.txt").write_text("text")
    _git(repo, "add", "line\nbreak.txt")
    _git(repo, "commit", "-qm", "ambiguous checksum path")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="newline-bearing"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_suite_export_requires_every_version_identity_file(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    (repo / "aita-bench" / "aita_bench" / "__init__.py").unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-qm", "looks like suite but incomplete")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="release identity file missing"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_suite_export_refuses_wrong_root_project_name(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "renamed-to-bypass"\nversion = "0.1.0"\n'
    )
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "wrong root identity")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="root project name mismatch"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_suite_export_refuses_hardcoded_cli_version(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    cli = repo / "epistemic-sycophancy-bench" / "epis_bench" / "cli.py"
    cli.write_text('VERSION = "epis-bench 9.9.9"\n')
    _git(repo, "add", str(cli.relative_to(repo)))
    _git(repo, "commit", "-qm", "drifted CLI")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="CLI release version"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )


def test_export_refuses_uncleared_aita_source_material(tmp_path):
    repo, _sha = _init_release_repo(tmp_path)
    data = repo / "aita-bench" / "data" / "curated" / "items.csv"
    data.parent.mkdir(parents=True)
    data.write_text("post\n")
    manifests = repo / "manifests"
    manifests.mkdir()
    (manifests / "aita-data-clearance.json").write_text(json.dumps({
        "schema_version": "aita-data-clearance-v1",
        "status": "not_cleared",
        "reviewer_identity": "pending-human-review",
        "decision_date": None,
        "covered_path_patterns": ["aita-bench/data/curated/**"],
        "source_provenance": "source",
        "privacy_review": "pending",
        "terms_policy_basis": "pending",
        "transformation_redaction_notes": "notes",
        "evidence_references": ["reference"],
    }))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "uncleared data")
    sha = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SourceReleaseError, match="not cleared"):
        export_source_release(
            repo_root=repo,
            revision=sha,
            out_dir=tmp_path / "export",
            release_version="0.1.0",
        )
