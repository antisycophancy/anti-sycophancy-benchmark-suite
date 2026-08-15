"""Manifest schema for traceable benchmark artifact bundles."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

RunQuality = Literal["smoke", "test", "research", "paper_candidate", "paper_final"]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    kind: str
    sha256: str


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    module: str
    quality: RunQuality
    source_root: str
    models: list[str]
    source_model_keys: list[str]
    judge_model: str | None
    seeker_model: str | None
    item_ids: list[str]
    n_conversations: int
    n_scores: int
    artifacts: list[ArtifactRef]
    notes: list[str]


def repo_relative_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    """Return a POSIX path relative to the benchmark repo root."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path, kind: str, repo_root: Path = REPO_ROOT) -> ArtifactRef:
    """Create an artifact reference with a repo-relative path and checksum."""
    return ArtifactRef(
        path=repo_relative_path(path, repo_root),
        kind=kind,
        sha256=sha256_file(path),
    )


def manifest_to_dict(manifest: RunManifest) -> dict:
    return asdict(manifest)


def manifests_to_dict(manifests: list[RunManifest]) -> list[dict]:
    return [manifest_to_dict(manifest) for manifest in manifests]
