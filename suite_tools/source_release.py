"""Deterministic, fail-closed source release export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

from suite_tools.release_audit import audit_tracked_files, run_audit
from suite_tools.data_clearance import (
    DataClearanceError,
    tracked_aita_source_paths,
    validate_public_data_clearance,
)
from suite_tools.suite_registry import REPO_ROOT


FULL_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.IGNORECASE)
RELEASE_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.-]*)\Z")
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)
CITATION_VERSION_RE = re.compile(r'^version:\s*["\']?([^"\'\s]+)["\']?\s*$', re.MULTILINE)
SUITE_PROJECT_NAME = "antisycophancy-suite-tools"
SUITE_VERSION_IDENTITY_PATHS = {
    "pyproject.toml",
    "sus-bench/pyproject.toml",
    "aita-bench/pyproject.toml",
    "epistemic-sycophancy-bench/pyproject.toml",
    "sus-bench/sus_bench/__init__.py",
    "aita-bench/aita_bench/__init__.py",
    "epistemic-sycophancy-bench/epis_bench/__init__.py",
    "CITATION.cff",
}
SUITE_CLI_VERSION_PATHS = {
    "sus-bench/sus_bench/cli.py": ("from sus_bench import __version__", "sus-bench {__version__}"),
    "aita-bench/aita_bench/cli.py": ("from aita_bench import __version__", "aita-bench {__version__}"),
    "epistemic-sycophancy-bench/epis_bench/cli.py": ("from epis_bench import __version__", "epis-bench {__version__}"),
}


class SourceReleaseError(RuntimeError):
    """Raised when an immutable source export cannot be proven safe."""


def _git(repo_root: Path, *args: str, text: bool = False) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceReleaseError("git could not inspect the selected source tree") from exc


def _resolve_full_commit(repo_root: Path, revision: str) -> str:
    if not FULL_OBJECT_ID_RE.fullmatch(revision):
        raise SourceReleaseError("--sha must name a full immutable commit object id")
    resolved = _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}", text=True).stdout.strip()
    if resolved.lower() != revision.lower():
        raise SourceReleaseError("--sha must name a full immutable commit object id")
    return resolved


def _require_clean_tree(repo_root: Path) -> None:
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        text=True,
    ).stdout
    if status:
        raise SourceReleaseError("working tree must be clean before source export")


def _tree_files(repo_root: Path, revision: str) -> list[str]:
    output = _git(repo_root, "ls-tree", "-r", "-z", "--full-tree", revision).stdout
    files: list[str] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise SourceReleaseError("git returned an invalid tree entry")
        fields = metadata.split(b" ")
        if len(fields) != 3:
            raise SourceReleaseError("git returned an invalid tree entry")
        mode, object_type, _object_id = fields
        if mode in {b"120000", b"160000"} or object_type == b"commit":
            raise SourceReleaseError("release trees must not contain symlinks or submodules")
        path = raw_path.decode("utf-8", errors="strict")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise SourceReleaseError("release tree contains an unsafe path")
        if "\n" in path or "\r" in path:
            raise SourceReleaseError("release tree contains a newline-bearing path")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path):
            raise SourceReleaseError("release tree contains a path unsafe for portable checksum verification")
        files.append(path)
    files = sorted(files)
    if "SHA256SUMS" in files:
        raise SourceReleaseError("release tree contains reserved SHA256SUMS path")
    return files


def _extract_archive(repo_root: Path, revision: str, destination: Path, expected: list[str]) -> None:
    archive_path = destination.parent / "source.tar"
    with archive_path.open("wb") as handle:
        try:
            subprocess.run(
                ["git", "archive", "--format=tar", revision],
                cwd=repo_root,
                check=True,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SourceReleaseError("git archive failed") from exc

    extracted: list[str] = []
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise SourceReleaseError("source archive contains an unsafe path")
            target = destination.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise SourceReleaseError("source archive contains a non-file entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SourceReleaseError("source archive contains an unreadable file")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            extracted.append(member.name)
    archive_path.unlink()
    if sorted(extracted) != expected:
        raise SourceReleaseError("source archive did not match the selected Git tree")


def _release_versions(root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in sorted(root.rglob("pyproject.toml")):
        try:
            project = tomllib.loads(path.read_text()).get("project", {})
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SourceReleaseError("release contains an invalid pyproject.toml") from exc
        version = project.get("version") if isinstance(project, dict) else None
        if isinstance(version, str):
            versions[path.relative_to(root).as_posix()] = version
    for path in sorted(root.rglob("__init__.py")):
        match = INIT_VERSION_RE.search(path.read_text())
        if match:
            versions[path.relative_to(root).as_posix()] = match.group(1)
    citation = root / "CITATION.cff"
    if citation.exists():
        match = CITATION_VERSION_RE.search(citation.read_text())
        if not match:
            raise SourceReleaseError("CITATION.cff must declare a release version")
        versions["CITATION.cff"] = match.group(1)
    return versions


def _validate_release_version(root: Path, release_version: str) -> None:
    if not RELEASE_VERSION_RE.fullmatch(release_version):
        raise SourceReleaseError("--release-version must be an explicit version")
    versions = _release_versions(root)
    missing = sorted(SUITE_VERSION_IDENTITY_PATHS - set(versions))
    missing.extend(
        path for path in sorted(SUITE_CLI_VERSION_PATHS) if not (root / path).is_file()
    )
    if missing:
        raise SourceReleaseError(
            "release identity file missing: " + ", ".join(missing)
        )
    root_project = tomllib.loads((root / "pyproject.toml").read_text()).get("project", {})
    if not isinstance(root_project, dict) or root_project.get("name") != SUITE_PROJECT_NAME:
        raise SourceReleaseError("release identity root project name mismatch")
    for path, required_fragments in SUITE_CLI_VERSION_PATHS.items():
        text = (root / path).read_text()
        if any(fragment not in text for fragment in required_fragments):
            raise SourceReleaseError(f"CLI release version is not package-derived: {path}")
    if not versions or any(version != release_version for version in versions.values()):
        raise SourceReleaseError("release version mismatch across source identity files")


def _write_checksums(root: Path, files: list[str]) -> None:
    lines = []
    for path in sorted(files):
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
        lines.append(f"{digest}  {path}\n")
    (root / "SHA256SUMS").write_text("".join(lines))


def _verified_export_members(export_root: Path) -> list[str]:
    issues = run_audit(repo_root=export_root)
    if issues:
        raise SourceReleaseError(
            f"export audit failed with {len(issues)} issue(s); do not seed a public root"
        )

    members: list[str] = []
    manifest = export_root / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        _digest, separator, relative = line.partition("  ")
        if separator != "  ":
            raise SourceReleaseError("export checksum manifest is malformed")
        members.append(relative)
    return members


def seed_public_root_from_export(
    *,
    export_root: Path,
    out_dir: Path,
    release_version: str,
) -> dict[str, Any]:
    """Copy a verified export into a checksum-free, Git-ready public root."""
    export_root = export_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir == export_root or export_root in out_dir.parents:
        raise SourceReleaseError("--out must be outside the audited export")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SourceReleaseError("--out must be absent or an empty directory")

    members = _verified_export_members(export_root)
    if "SHA256SUMS" in members:
        raise SourceReleaseError("export manifest must not list SHA256SUMS")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="public-root-", dir=out_dir.parent) as temp:
        staging = Path(temp) / "tree"
        staging.mkdir()
        for relative in members:
            source = export_root / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)

        _validate_release_version(staging, release_version)
        _validate_data_clearance(staging, members)
        issues = audit_tracked_files(members, repo_root=staging)
        if issues:
            raise SourceReleaseError(
                f"public-root audit failed with {len(issues)} issue(s)"
            )
        if (staging / "SHA256SUMS").exists():
            raise SourceReleaseError("public root must not contain generated SHA256SUMS")
        if out_dir.exists():
            out_dir.rmdir()
        os.replace(staging, out_dir)

    return {
        "schema_version": "benchmark-public-root-seed-receipt-v1",
        "release_version": release_version,
        "file_count": len(members),
        "checksum_included": False,
        "source_manifest_sha256": hashlib.sha256(
            (export_root / "SHA256SUMS").read_bytes()
        ).hexdigest(),
    }


def _validate_data_clearance(root: Path, files: list[str]) -> None:
    aita_paths = tracked_aita_source_paths(files)
    if not aita_paths:
        return
    clearance_path = root / "manifests" / "aita-data-clearance.json"
    try:
        record = json.loads(clearance_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceReleaseError("AITA data clearance record is missing or unreadable") from exc
    try:
        validate_public_data_clearance(record, aita_paths)
    except DataClearanceError as exc:
        raise SourceReleaseError(str(exc)) from exc


def export_source_release(
    *,
    repo_root: Path,
    revision: str,
    out_dir: Path,
    release_version: str,
) -> dict[str, Any]:
    """Export one clean commit without creating or rewriting Git history."""
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SourceReleaseError("--out must be absent or an empty directory")
    revision = _resolve_full_commit(repo_root, revision)
    _require_clean_tree(repo_root)
    files = _tree_files(repo_root, revision)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-release-", dir=out_dir.parent) as temp:
        staging = Path(temp) / "tree"
        staging.mkdir()
        _extract_archive(repo_root, revision, staging, files)
        _validate_release_version(staging, release_version)
        _validate_data_clearance(staging, files)
        issues = audit_tracked_files(files, repo_root=staging)
        if issues:
            raise SourceReleaseError(
                f"release audit failed with {len(issues)} issue(s); run release_audit for details"
            )
        _write_checksums(staging, files)
        if out_dir.exists():
            out_dir.rmdir()
        os.replace(staging, out_dir)

    return {
        "schema_version": "benchmark-source-release-receipt-v1",
        "revision": revision,
        "release_version": release_version,
        "file_count": len(files),
        "audit_issue_count": 0,
        "checksum_file": "SHA256SUMS",
        "git_history_included": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an audited immutable source release.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--sha", required=True, help="Full immutable commit object id.")
    parser.add_argument("--out", required=True, help="Absent or empty output directory.")
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = export_source_release(
            repo_root=Path(args.repo_root),
            revision=args.sha,
            out_dir=Path(args.out),
            release_version=args.release_version,
        )
    except SourceReleaseError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"Source release blocked: {exc}")
        return 1
    if args.json:
        print(json.dumps({"ok": True, **receipt}, indent=2, sort_keys=True))
    else:
        print(f"Exported {receipt['file_count']} files from {receipt['revision']} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
