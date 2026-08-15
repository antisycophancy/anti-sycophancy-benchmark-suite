"""Release-surface audit for benchmark source and public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from suite_tools.artifact_privacy import scan_public_artifact_payload
from suite_tools.suite_registry import REPO_ROOT


BLOCKING_PATH_PREFIXES = (
    ".agents/",
    ".benchmark-companion/",
    ".claude/",
    ".orca/",
    ".planning/",
    ".superpowers/",
    "docs/goals/",
    "docs/superpowers/",
    "internal/",
    "paper-writer/",
    "plans/",
    "private_question_bank/",
)
PUBLIC_AGENT_DISCOVERY_PATHS = {
    ".agents/skills/antisycophancy/SKILL.md",
    ".claude/skills/antisycophancy/SKILL.md",
}
PAPER_EVIDENCE_PATH_RE = re.compile(
    r"^manifests/(?:"
    r"paper-(?:analysis-selection|edition|model-inventory|writing-handoff)|"
    r"aita-harness-recovered"
    r")[^/]*\.(?:json|md)$"
)
RESULT_PATH_RE = re.compile(r"(^|/)results/")
ALLOWED_RESULT_PATHS = (
    "aita-bench/results/sample/",
    "epistemic-sycophancy-bench/results/.gitkeep",
    "sus-bench/results/.gitkeep",
)
LEGACY_AITE_NAME = r"\bAI" + r"TE\b"
LEGACY_AITA_ROOT_NAME = "am-i-the-" + "elephant"
LEGACY_SUS_ROOT_NAME = "self-undermining-" + "sycophancy-bench"
LEGACY_SUS_PHRASE = "self-undermining " + "sycophancy"
LEGACY_TEXT_PATTERNS = (
    re.compile(LEGACY_SUS_ROOT_NAME, re.IGNORECASE),
    re.compile(LEGACY_SUS_PHRASE, re.IGNORECASE),
    re.compile(LEGACY_AITA_ROOT_NAME, re.IGNORECASE),
    re.compile(LEGACY_AITE_NAME),
)
SECRET_TEXT_RE = re.compile(
    r"(?i)("
    r"sk-[a-z0-9_-]{20,}|"
    r"sk-or-v1-[a-z0-9_-]{20,}|"
    r"user_[a-z0-9]{20,}|"
    r"openrouter\.ai/workspaces/[a-z0-9_-]+/keys/[a-z0-9_-]{20,}|"
    r"authorization:[ \t]*bearer[ \t]+[a-z0-9._-]{20,}|"
    r"(api[_-]?key|token|secret|password)[ \t]*[:=][ \t]*['\"]?[a-z0-9._-]{20,}"
    r")"
)
AITA_SOURCE_URL_RE = re.compile(
    r"https?://(?:www\.)?reddit\.com/r/AmItheAsshole/comments/",
    re.IGNORECASE,
)
# Lines ending with this comment are deliberate test fixtures that exercise the
# secret scrubber.  Any UNMARKED secret-looking string still triggers a blocker.
FIXTURE_NOQA_MARKER = "# noqa: release-audit-fixture"
MAX_HISTORY_BLOB_BYTES = 2 * 1024 * 1024
CHECKSUM_FILE = "SHA256SUMS"
GENERATED_EXPORT_PARTS = {"__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class ReleaseAuditIssue:
    """One release-surface issue."""

    path: str
    severity: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "severity": self.severity, "reason": self.reason}


def _is_allowed_result_path(path: str) -> bool:
    return path.endswith("/.gitkeep") or any(path.startswith(prefix) for prefix in ALLOWED_RESULT_PATHS)


def _is_public_agent_discovery_path(path: str) -> bool:
    return path in PUBLIC_AGENT_DISCOVERY_PATHS


def _tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _is_git_worktree(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == repo_root.resolve()
    except OSError:
        return False


def _checksum_manifest_files(repo_root: Path) -> tuple[list[str], list[ReleaseAuditIssue]]:
    """Discover and verify files in a Git-free exported source tree."""
    manifest = repo_root / CHECKSUM_FILE
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], [
            ReleaseAuditIssue(
                CHECKSUM_FILE,
                "blocker",
                "Git-free source tree is missing its checksum manifest",
            )
        ]

    files: list[str] = []
    issues: list[ReleaseAuditIssue] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        digest, separator, relative = line.partition("  ")
        path = PurePosixPath(relative)
        unsafe = (
            separator != "  "
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not relative
            or "\\" in relative
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative == CHECKSUM_FILE
            or relative in seen
        )
        if unsafe:
            issues.append(
                ReleaseAuditIssue(
                    f"{CHECKSUM_FILE}:{line_number}",
                    "blocker",
                    "malformed or unsafe checksum manifest entry",
                )
            )
            continue

        seen.add(relative)
        files.append(relative)
        candidate = repo_root / relative
        if not candidate.is_file() or candidate.is_symlink():
            issues.append(
                ReleaseAuditIssue(relative, "blocker", "checksum manifest member is missing or not a regular file")
            )
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != digest:
            issues.append(ReleaseAuditIssue(relative, "blocker", "checksum manifest digest mismatch"))

    if not files:
        issues.append(ReleaseAuditIssue(CHECKSUM_FILE, "blocker", "checksum manifest has no source members"))

    actual_files: set[str] = set()
    for root, dirnames, filenames in os.walk(repo_root, followlinks=False):
        directory = Path(root)
        relative_dir = directory.relative_to(repo_root)
        for name in dirnames:
            relative = (relative_dir / name).as_posix().removeprefix("./")
            if (repo_root / relative).is_symlink():
                issues.append(ReleaseAuditIssue(relative, "blocker", "unexpected symlink in exported source tree"))
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_generated_export_path((relative_dir / name).as_posix())
            and not (directory / name).is_symlink()
        ]
        for name in filenames:
            relative = (relative_dir / name).as_posix()
            candidate = repo_root / relative
            if candidate.is_symlink():
                issues.append(ReleaseAuditIssue(relative, "blocker", "unexpected symlink in exported source tree"))
            elif relative in {"./" + CHECKSUM_FILE, CHECKSUM_FILE} or _is_generated_export_path(relative):
                continue
            elif candidate.is_file():
                actual_files.add(relative.removeprefix("./"))
    for relative in sorted(actual_files - seen):
        issues.append(ReleaseAuditIssue(relative, "blocker", "file is not listed in checksum manifest"))
    return files, issues


def _is_generated_export_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return (
        (parts and parts[0] == "venv")
        or any(part in GENERATED_EXPORT_PARTS or part.endswith(".egg-info") for part in parts)
        or relative == ".coverage"
    )


def discover_release_files(repo_root: Path) -> tuple[list[str], list[ReleaseAuditIssue]]:
    """Return the authoritative public-source inventory for a checkout or export."""
    if _is_git_worktree(repo_root):
        return _tracked_files(repo_root), []
    return _checksum_manifest_files(repo_root)


def _read_tracked_text(repo_root: Path, path: str) -> str | None:
    try:
        raw = (repo_root / path).read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="ignore")


def _scan_json_artifact(path: str, text: str) -> list[ReleaseAuditIssue]:
    if not path.endswith(".json") or not RESULT_PATH_RE.search(path):
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [ReleaseAuditIssue(path, "blocker", "malformed public JSON artifact")]
    return [
        ReleaseAuditIssue(path, "blocker", f"public artifact privacy: {issue.path} {issue.reason}")
        for issue in scan_public_artifact_payload(payload)
    ]


def audit_tracked_files(
    files: list[str],
    *,
    repo_root: Path = REPO_ROOT,
    text_by_path: Mapping[str, str] | None = None,
) -> list[ReleaseAuditIssue]:
    """Audit tracked paths and optional file text for release-surface problems."""
    issues: list[ReleaseAuditIssue] = []
    text_by_path = text_by_path or {}

    for path in sorted(files):
        normalized = path.replace("\\", "/")
        for prefix in BLOCKING_PATH_PREFIXES:
            if normalized.startswith(prefix) and not _is_public_agent_discovery_path(normalized):
                issues.append(ReleaseAuditIssue(normalized, "blocker", f"tracked private/local path: {prefix}"))
        for pattern in LEGACY_TEXT_PATTERNS:
            if pattern.search(normalized):
                issues.append(ReleaseAuditIssue(normalized, "blocker", f"legacy public path/name: {pattern.pattern}"))

        if PAPER_EVIDENCE_PATH_RE.search(normalized):
            issues.append(
                ReleaseAuditIssue(
                    normalized,
                    "blocker",
                    "tracked paper evidence manifest; archive outside the source repository",
                )
            )

        if RESULT_PATH_RE.search(normalized) and not _is_allowed_result_path(normalized):
            issues.append(
                ReleaseAuditIssue(
                    normalized,
                    "warning",
                    "tracked result artifact; keep only fixtures/samples or publish from an archive",
                )
            )

        text = text_by_path.get(normalized)
        if text is None:
            text = _read_tracked_text(repo_root, normalized)
        if text is None:
            continue

        if any(
            SECRET_TEXT_RE.search(line)
            for line in text.splitlines()
            if FIXTURE_NOQA_MARKER not in line
        ):
            issues.append(ReleaseAuditIssue(normalized, "blocker", "secret-looking text"))
        if AITA_SOURCE_URL_RE.search(text):
            issues.append(
                ReleaseAuditIssue(normalized, "blocker", "plaintext AITA source URL")
            )
        for pattern in LEGACY_TEXT_PATTERNS:
            if pattern.search(text):
                issues.append(ReleaseAuditIssue(normalized, "blocker", f"legacy public terminology: {pattern.pattern}"))
        issues.extend(_scan_json_artifact(normalized, text))

    return issues


def _history_object_paths(repo_root: Path) -> dict[str, set[str]]:
    tree_result = subprocess.run(
        ["git", "log", "--all", "--format=%T"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    objects: dict[str, set[str]] = {}
    for tree_id in dict.fromkeys(tree_result.stdout.splitlines()):
        if not tree_id:
            continue
        result = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--full-tree", tree_id],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            metadata, separator, raw_path = record.partition(b"\t")
            if not separator:
                raise RuntimeError("unexpected git ls-tree response")
            fields = metadata.split(b" ")
            if len(fields) != 3 or fields[1] != b"blob":
                continue
            object_id = fields[2].decode("ascii")
            path = raw_path.decode("utf-8", errors="surrogateescape")
            objects.setdefault(object_id, set()).add(path)
    return objects


def _all_reachable_object_ids(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "rev-list", "--objects", "--all", "--no-object-names"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    object_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", item) for item in object_ids):
        raise RuntimeError("unexpected git rev-list object id")
    return list(dict.fromkeys(object_ids))


def _history_blob_metadata(repo_root: Path, object_ids: list[str]) -> dict[str, int]:
    if not object_ids:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=repo_root,
        check=True,
        input="\n".join(object_ids) + "\n",
        capture_output=True,
        text=True,
    )
    blobs: dict[str, int] = {}
    for line in result.stdout.splitlines():
        object_id, object_type, size_text = line.split(" ", 2)
        size = int(size_text)
        if object_type == "blob":
            blobs[object_id] = size
    return blobs


def _read_history_blobs(repo_root: Path, blobs: Mapping[str, int]) -> dict[str, bytes]:
    if not blobs:
        return {}
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("git cat-file pipes were not created")
    process.stdin.write(("\n".join(blobs) + "\n").encode("ascii"))
    process.stdin.close()

    payloads: dict[str, bytes] = {}
    for requested_id in blobs:
        header = process.stdout.readline().decode("ascii").strip().split(" ")
        if len(header) != 3 or header[1] != "blob":
            process.kill()
            raise RuntimeError(f"unexpected git cat-file response for {requested_id}")
        object_id, _object_type, size_text = header
        size = int(size_text)
        payloads[object_id] = process.stdout.read(size)
        if process.stdout.read(1) != b"\n":
            process.kill()
            raise RuntimeError(f"unterminated git cat-file response for {requested_id}")

    if process.wait() != 0:
        raise RuntimeError("git cat-file failed while scanning reachable history")
    return payloads


def audit_reachable_history(repo_root: Path) -> list[ReleaseAuditIssue]:
    """Audit bounded textual blobs in reachable Git history without echoing matches."""
    object_paths = _history_object_paths(repo_root)
    reachable_ids = _all_reachable_object_ids(repo_root)
    blob_sizes = _history_blob_metadata(repo_root, list(dict.fromkeys([*object_paths, *reachable_ids])))
    bounded_sizes = {
        object_id: size
        for object_id, size in blob_sizes.items()
        if size <= MAX_HISTORY_BLOB_BYTES
    }
    blob_payloads = _read_history_blobs(repo_root, bounded_sizes)
    issues: list[ReleaseAuditIssue] = []

    for object_id, size in blob_sizes.items():
        paths = sorted(object_paths.get(object_id) or {"<unmapped-reachable-blob>"})
        for stored_path in paths:
            path = stored_path.replace("\\", "/")
            issue_path = f"{path}@{object_id[:12]}"
            if size > MAX_HISTORY_BLOB_BYTES:
                issues.append(
                    ReleaseAuditIssue(issue_path, "blocker", "reachable history blob exceeds scan limit")
                )
            for prefix in BLOCKING_PATH_PREFIXES:
                if path.startswith(prefix) and not _is_public_agent_discovery_path(path):
                    issues.append(
                        ReleaseAuditIssue(issue_path, "blocker", "private/local path in reachable history")
                    )
                    break
            if PAPER_EVIDENCE_PATH_RE.search(path):
                issues.append(
                    ReleaseAuditIssue(issue_path, "blocker", "paper evidence path in reachable history")
                )
            if any(pattern.search(path) for pattern in LEGACY_TEXT_PATTERNS):
                issues.append(
                    ReleaseAuditIssue(issue_path, "blocker", "legacy public path/name in reachable history")
                )

    for object_id, raw in blob_payloads.items():
        text = None if b"\0" in raw else raw.decode("utf-8", errors="ignore")
        for stored_path in sorted(object_paths.get(object_id) or {"<unmapped-reachable-blob>"}):
            normalized_path = stored_path.replace("\\", "/")
            issue_path = f"{normalized_path}@{object_id[:12]}"
            if text is not None and any(
                SECRET_TEXT_RE.search(line)
                for line in text.splitlines()
                if FIXTURE_NOQA_MARKER not in line
            ):
                issues.append(
                    ReleaseAuditIssue(issue_path, "blocker", "secret-looking text in reachable history")
                )
            if text is not None and AITA_SOURCE_URL_RE.search(text):
                issues.append(
                    ReleaseAuditIssue(
                        issue_path,
                        "blocker",
                        "plaintext AITA source URL in reachable history",
                    )
                )

    return issues


def run_audit(*, repo_root: Path = REPO_ROOT, history: bool = False) -> list[ReleaseAuditIssue]:
    """Audit the releasable working tree and, optionally, reachable history."""
    files, discovery_issues = discover_release_files(repo_root)
    issues = [*discovery_issues, *audit_tracked_files(files, repo_root=repo_root)]
    if history:
        if _is_git_worktree(repo_root):
            issues.extend(audit_reachable_history(repo_root))
        else:
            issues.append(
                ReleaseAuditIssue(
                    ".git",
                    "blocker",
                    "reachable-history audit requires a Git worktree",
                )
            )
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the benchmark release surface.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Benchmark repo root.")
    parser.add_argument("--json", action="store_true", help="Print JSON issue list.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero for warnings as well as blockers.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan bounded textual blobs in reachable Git history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = run_audit(repo_root=Path(args.repo_root), history=args.history)
    blockers = [issue for issue in issues if issue.severity == "blocker"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    if args.json:
        print(json.dumps([issue.as_dict() for issue in issues], indent=2, sort_keys=True))
    else:
        if not issues:
            print("Release audit passed: no release-surface issues found.")
        else:
            print(f"Release audit found {len(blockers)} blocker(s), {len(warnings)} warning(s).")
            for issue in issues:
                print(f"[{issue.severity}] {issue.path}: {issue.reason}")

    if blockers or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
