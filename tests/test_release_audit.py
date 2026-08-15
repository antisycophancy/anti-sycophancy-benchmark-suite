import hashlib
import subprocess

import pytest

from suite_tools.release_audit import FIXTURE_NOQA_MARKER, audit_tracked_files, run_audit
from suite_tools.suite_registry import REPO_ROOT


def _issues_by_path(issues):
    return {(issue.path, issue.severity, issue.reason) for issue in issues}


def test_release_audit_blocks_private_paths_and_secret_text():
    issues = audit_tracked_files(
        [".orca/run.json", "internal/runbook.md", "suite_tools/example.py"],
        text_by_path={
            ".orca/run.json": "{}",
            "internal/runbook.md": "private ops",
            "suite_tools/example.py": "OPENROUTER_API_KEY=sk-" + ("x" * 24),
        },
    )

    reasons = _issues_by_path(issues)

    assert any(path == "internal/runbook.md" and severity == "blocker" for path, severity, _ in reasons)
    assert any(path == ".orca/run.json" and severity == "blocker" for path, severity, _ in reasons)
    assert any(path == "suite_tools/example.py" and "secret-looking" in reason for path, _, reason in reasons)


def test_release_audit_allows_only_canonical_agent_skill_discovery_wrappers():
    allowed = [
        ".claude/skills/antisycophancy/SKILL.md",
        ".agents/skills/antisycophancy/SKILL.md",
    ]
    blocked = [
        ".claude/settings.local.json",
        ".claude/skills/unrelated/SKILL.md",
        ".agents/local-state.json",
        ".agents/skills/unrelated/SKILL.md",
    ]
    text_by_path = {path: "public skill wrapper\n" for path in [*allowed, *blocked]}

    issues = audit_tracked_files([*allowed, *blocked], text_by_path=text_by_path)
    blocked_paths = {issue.path for issue in issues if issue.severity == "blocker"}

    assert not blocked_paths.intersection(allowed)
    assert blocked_paths.issuperset(blocked)


def test_release_audit_blocks_plaintext_aita_source_urls():
    source_url = "https://www.reddit.com/r/" + "AmItheAsshole/comments/example/post/"
    issues = audit_tracked_files(
        ["docs/source-index.md"],
        text_by_path={"docs/source-index.md": source_url},
    )

    assert any(
        issue.path == "docs/source-index.md"
        and issue.reason == "plaintext AITA source URL"
        for issue in issues
    )


def test_git_free_export_uses_and_verifies_checksum_manifest(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("public source\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  README.md\n")

    assert run_audit(repo_root=tmp_path) == []

    source.write_text("modified after export\n")
    issues = run_audit(repo_root=tmp_path)

    assert any(issue.path == "README.md" and "digest mismatch" in issue.reason for issue in issues)


def test_git_free_export_fails_closed_without_checksum_manifest(tmp_path):
    (tmp_path / "README.md").write_text("unverified source\n")

    issues = run_audit(repo_root=tmp_path)

    assert any(issue.path == "SHA256SUMS" and issue.severity == "blocker" for issue in issues)


def test_git_free_export_rejects_unlisted_file(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("public source\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  README.md\n")
    (tmp_path / "sitecustomize.py").write_text("raise SystemExit('unexpected execution')\n")

    issues = run_audit(repo_root=tmp_path)

    assert any(
        issue.path == "sitecustomize.py" and issue.reason == "file is not listed in checksum manifest"
        for issue in issues
    )


def test_git_free_export_rejects_generated_directory_symlink(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("public source\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  README.md\n")
    (tmp_path / "venv").symlink_to(tmp_path / "outside", target_is_directory=True)

    issues = run_audit(repo_root=tmp_path)

    assert any(
        issue.path == "venv" and issue.reason == "unexpected symlink in exported source tree"
        for issue in issues
    )


def test_release_audit_blocks_paper_evidence_manifests():
    issues = audit_tracked_files(
        [
            "manifests/paper-edition-2026-08-04.json",
            "manifests/paper-writing-handoff-2026-08-04.md",
            "manifests/aita-prospective-dataset-protocol-v1.md",
        ],
        text_by_path={
            "manifests/paper-edition-2026-08-04.json": "{}",
            "manifests/paper-writing-handoff-2026-08-04.md": "handoff",
            "manifests/aita-prospective-dataset-protocol-v1.md": "public methodology",
        },
    )

    blockers = {issue.path for issue in issues if issue.severity == "blocker"}
    assert "manifests/paper-edition-2026-08-04.json" in blockers
    assert "manifests/paper-writing-handoff-2026-08-04.md" in blockers
    assert "manifests/aita-prospective-dataset-protocol-v1.md" not in blockers


def test_release_audit_blocks_legacy_names_and_warns_on_result_artifacts():
    legacy_aita_root = "am-i-the-" + "elephant"
    legacy_sus_root = "self-undermining-" + "sycophancy-bench"
    issues = audit_tracked_files(
        [
            "docs/methods.md",
            f"{legacy_aita_root}/README.md",
            "sus-bench/results/.gitkeep",
            "sus-bench/results/run.json",
            "aita-bench/results/sample/conversations/sample.json",
        ],
        text_by_path={
            "docs/methods.md": "Old name: " + legacy_sus_root + " and " + "AI" + "TE.",
            "sus-bench/results/run.json": '{"ok": true}',
            "aita-bench/results/sample/conversations/sample.json": '{"ok": true}',
        },
    )

    blockers = [issue for issue in issues if issue.severity == "blocker"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    assert any(issue.path == "docs/methods.md" for issue in blockers)
    assert any(issue.path == f"{legacy_aita_root}/README.md" for issue in blockers)
    assert any(issue.path == "sus-bench/results/run.json" for issue in warnings)
    assert not any(issue.path == "aita-bench/results/sample/conversations/sample.json" for issue in warnings)


def test_release_audit_scans_json_result_payloads_for_private_fields():
    issues = audit_tracked_files(
        ["sus-bench/results/run.json"],
        text_by_path={"sus-bench/results/run.json": '{"raw_prompt": "private"}'},
    )

    assert any(issue.severity == "blocker" and "public artifact privacy" in issue.reason for issue in issues)


def test_release_audit_blocks_malformed_public_result_json():
    path = "sus-bench/results/run.json"

    issues = audit_tracked_files([path], text_by_path={path: "{"})

    assert any(
        issue.path == path
        and issue.severity == "blocker"
        and issue.reason == "malformed public JSON artifact"
        for issue in issues
    )


def test_fixture_noqa_marker_suppresses_secret_on_marked_line():
    """A line ending with FIXTURE_NOQA_MARKER is exempted; other lines are not."""
    secret_fixture = "sk-" + "abc123456789012345"
    marked_text = f'secret = "{secret_fixture}"  {FIXTURE_NOQA_MARKER}'
    issues = audit_tracked_files(
        ["tests/test_example.py"],
        text_by_path={"tests/test_example.py": marked_text},
    )
    assert not any("secret-looking" in issue.reason for issue in issues), (
        "Marked fixture line should not trigger a secret blocker"
    )


def test_unmarked_secret_in_test_file_still_blocks():
    """An unmarked secret-looking string in a test file must still be a blocker."""
    secret_fixture = "sk-" + "realkey12345678901234567890"
    unmarked_text = "api_key = " + f'"{secret_fixture}"'
    issues = audit_tracked_files(
        ["tests/test_example.py"],
        text_by_path={"tests/test_example.py": unmarked_text},
    )
    assert any(
        issue.severity == "blocker" and "secret-looking" in issue.reason
        for issue in issues
    ), "Unmarked secret in a test file must still trigger a blocker"


@pytest.mark.parametrize(
    "private_identifier",
    [
        "user_" + ("A" * 32),
        "https://openrouter.ai/workspaces/default/keys/" + ("0" * 40),
    ],
)
def test_release_audit_blocks_private_provider_identifiers(private_identifier):
    issues = audit_tracked_files(
        ["tests/provider-fixture.txt"],
        text_by_path={"tests/provider-fixture.txt": private_identifier},
    )

    assert any(
        issue.severity == "blocker" and issue.reason == "secret-looking text"
        for issue in issues
    )


def _is_git_worktree(path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path):
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "release-audit@example.invalid")
    _git(path, "config", "user.name", "Release Audit Test")


def test_run_audit_includes_nonignored_untracked_files_and_spaces(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("safe")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "base")
    secret = "sk-" + ("u" * 24)
    (repo / "untracked note.txt").write_text(secret)

    issues = run_audit(repo_root=repo)

    assert any(
        issue.path == "untracked note.txt" and issue.reason == "secret-looking text"
        for issue in issues
    )


def test_run_audit_excludes_ignored_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored.env\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "base")
    (repo / "ignored.env").write_text("sk-" + ("i" * 24))

    issues = run_audit(repo_root=repo)

    assert not any(issue.path == "ignored.env" for issue in issues)


@pytest.mark.parametrize("name", ["config.toml", "run.sh", "key.pem", "app.js", "CITATION.cff"])
def test_release_audit_scans_text_regardless_of_extension(tmp_path, name):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / name).write_text("api_key = " + "sk-" + ("x" * 24))
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", "add text")

    issues = run_audit(repo_root=repo)

    assert any(issue.path == name and issue.reason == "secret-looking text" for issue in issues)


def test_history_mode_finds_deleted_secret_without_echoing_it(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    secret = "sk-" + ("h" * 24)
    path = repo / "deleted.txt"
    path.write_text(secret)
    _git(repo, "add", "deleted.txt")
    _git(repo, "commit", "-qm", "add secret")
    path.unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-qm", "delete secret")

    issues = run_audit(repo_root=repo, history=True)
    rendered = "\n".join(str(issue.as_dict()) for issue in issues)

    assert any(
        issue.path.startswith("deleted.txt@")
        and issue.reason == "secret-looking text in reachable history"
        for issue in issues
    )
    assert secret not in rendered


def test_history_mode_checks_every_path_alias_for_reused_blob(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "safe.txt").write_text("ordinary text")
    _git(repo, "add", "safe.txt")
    _git(repo, "commit", "-qm", "safe alias")
    internal = repo / "internal"
    internal.mkdir()
    (internal / "operator.txt").write_text("ordinary text")
    _git(repo, "add", "internal/operator.txt")
    _git(repo, "commit", "-qm", "private alias")

    issues = run_audit(repo_root=repo, history=True)

    assert any(
        issue.path.startswith("internal/operator.txt@")
        and issue.reason == "private/local path in reachable history"
        for issue in issues
    )


def test_history_mode_handles_newline_filename_without_quoting_bypass(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    path = repo / "internal\noperator.toml"
    path.write_text("api_key = " + "sk-" + ("n" * 24))
    _git(repo, "add", "internal\noperator.toml")
    _git(repo, "commit", "-qm", "quoted path")

    issues = run_audit(repo_root=repo, history=True)

    assert any("internal\noperator.toml@" in issue.path for issue in issues)


def test_history_mode_scans_blob_reachable_only_from_tag(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "safe.txt").write_text("safe")
    _git(repo, "add", "safe.txt")
    _git(repo, "commit", "-qm", "base")
    secret = "sk-" + ("t" * 24)
    blob_id = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=secret,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _git(repo, "tag", "blob-only", blob_id)

    issues = run_audit(repo_root=repo, history=True)

    assert any(
        issue.path.startswith("<unmapped-reachable-blob>@")
        and issue.reason == "secret-looking text in reachable history"
        for issue in issues
    )


def test_history_mode_blocks_oversized_reachable_blob(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    internal = repo / "internal"
    internal.mkdir()
    (internal / "large.bin").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    _git(repo, "add", "internal/large.bin")
    _git(repo, "commit", "-qm", "large private blob")

    issues = run_audit(repo_root=repo, history=True)

    assert any(
        issue.path.startswith("internal/large.bin@")
        and issue.reason == "reachable history blob exceeds scan limit"
        for issue in issues
    )


@pytest.mark.skipif(
    not _is_git_worktree(REPO_ROOT),
    reason="requires a git worktree",
)
def test_real_repo_passes_release_audit():
    """The actual tracked tree must produce zero blockers.

    This test catches regressions where a new tracked file introduces a secret
    or blocked path without the audit being updated first.  It is intentionally
    fast: git ls-files is the only I/O beyond reading those files.
    """
    issues = run_audit(repo_root=REPO_ROOT)
    blockers = [issue for issue in issues if issue.severity == "blocker"]
    assert not blockers, (
        f"release_audit found {len(blockers)} blocker(s) in the real tracked tree:\n"
        + "\n".join(f"  [{i.severity}] {i.path}: {i.reason}" for i in blockers)
    )


def test_public_suite_excludes_legacy_unaccounted_paid_calibration_tools():
    for relative in (
        "suite_tools/aita_pressure_curve.py",
        "suite_tools/aita_tail_probe.py",
        "suite_tools/aita_trajectory_compare.py",
    ):
        assert not (REPO_ROOT / relative).exists()
