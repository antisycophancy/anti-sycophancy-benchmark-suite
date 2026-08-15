"""Consent-gated acquisition and verification for the separate AITA data pack.

This module downloads only the public envelope and ciphertext named by the
registry. It never accepts an unlock fragment and never opens plaintext.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

import httpx


REGISTRY_SCHEMA = "aita-sealed-pack-registry-v1"
STATUS_SCHEMA = "aita-data-pack-status-v1"
RECEIPT_SCHEMA = "aita-data-pack-download-receipt-v1"
VERIFY_SCHEMA = "aita-data-pack-verification-receipt-v1"
RECEIPT_FILE = "AITA_DATA_PACK_RECEIPT.json"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "manifests" / "aita-sealed-pack-v1.json"
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_ENVELOPE_BYTES = 1024 * 1024
MAX_CIPHERTEXT_BYTES = 64 * 1024 * 1024
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
    }
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DOWNLOAD_HEADERS = {
    "accept": "application/octet-stream",
    "accept-encoding": "identity",
    "user-agent": "antisycophancy-data-pack/1",
}
_MAX_REDIRECTS = 5


class DataPackError(ValueError):
    """Raised when data-pack acquisition or verification must fail closed."""


def _read_registry(path: str | Path = DEFAULT_REGISTRY) -> tuple[dict[str, Any], str]:
    registry_path = Path(path)
    try:
        if registry_path.is_symlink() or not registry_path.is_file():
            raise DataPackError("AITA data-pack registry is missing or unsafe")
        if registry_path.stat().st_size > MAX_REGISTRY_BYTES:
            raise DataPackError("AITA data-pack registry is too large")
        raw = registry_path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except DataPackError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataPackError("AITA data-pack registry is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != REGISTRY_SCHEMA:
        raise DataPackError("AITA data-pack registry schema is unsupported")
    for field in ("pack_id", "pack_version"):
        if not isinstance(value.get(field), str) or not _IDENTITY.fullmatch(value[field]):
            raise DataPackError(f"AITA data-pack registry {field} is invalid")
    pair_count = value.get("pair_count")
    if not isinstance(pair_count, int) or isinstance(pair_count, bool) or pair_count <= 0:
        raise DataPackError("AITA data-pack registry pair_count is invalid")
    distribution = value.get("distribution")
    identity = value.get("identity")
    unlock = value.get("unlock")
    if (
        not isinstance(distribution, dict)
        or not isinstance(identity, dict)
        or not isinstance(unlock, dict)
    ):
        raise DataPackError("AITA data-pack registry metadata is incomplete")
    if distribution.get("mode") != "separate-signed-release":
        raise DataPackError("AITA data-pack registry distribution mode is unsupported")
    expected_identity = {
        "canonicalization_version": "canonical-json-exact-bytes-v1",
        "cipher_suite": "AES-256-GCM",
        "key_scheme": "public-base64url-split-22-21-v1",
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise DataPackError(f"AITA data-pack registry identity.{field} is unsupported")
    plaintext_digest = identity.get("plaintext_identity_sha256")
    if not isinstance(plaintext_digest, str) or not _SHA256.fullmatch(plaintext_digest):
        raise DataPackError(
            "AITA data-pack registry identity.plaintext_identity_sha256 is invalid"
        )
    part_b_url = unlock.get("part_b_url")
    if part_b_url is not None and not isinstance(part_b_url, str):
        raise DataPackError("AITA data-pack registry unlock.part_b_url is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _artifact(registry: Mapping[str, Any], kind: str) -> dict[str, Any]:
    distribution = registry["distribution"]
    name = distribution.get(f"{kind}_file")
    size = distribution.get(f"{kind}_bytes")
    digest = distribution.get(f"{kind}_sha256")
    limit = MAX_ENVELOPE_BYTES if kind == "envelope" else MAX_CIPHERTEXT_BYTES
    if not isinstance(name, str) or not _IDENTITY.fullmatch(name) or name in {".", ".."}:
        raise DataPackError(f"AITA data-pack registry {kind}_file is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size > limit:
        raise DataPackError(f"AITA data-pack registry {kind}_bytes is invalid")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise DataPackError(f"AITA data-pack registry {kind}_sha256 is invalid")
    url = distribution.get(f"{kind}_url")
    if url is not None and not isinstance(url, str):
        raise DataPackError(f"AITA data-pack registry {kind}_url is invalid")
    return {"file": name, "bytes": size, "sha256": digest, "url": url}


def registry_status(registry_path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Describe the public pack without making a network request."""
    registry, _ = _read_registry(registry_path)
    envelope = _artifact(registry, "envelope")
    ciphertext = _artifact(registry, "ciphertext")
    distribution = registry["distribution"]
    unlock = registry["unlock"]
    part_b_url = unlock.get("part_b_url")
    for item in (envelope, ciphertext):
        if item["url"]:
            _validated_url(item["url"], source=item["file"])
    if part_b_url:
        _validated_reference_url(part_b_url, source="Part B")
    missing_assets = sorted(
        f"distribution.{kind}_url"
        for kind, artifact in (("envelope", envelope), ("ciphertext", ciphertext))
        if not artifact["url"]
    )
    missing = [*missing_assets]
    if not part_b_url:
        missing.append("unlock.part_b_url")
    missing.sort()
    return {
        "schema_version": STATUS_SCHEMA,
        "pack_id": registry["pack_id"],
        "pack_version": registry["pack_version"],
        "pair_count": registry["pair_count"],
        "download_available": not missing_assets,
        "unlock_available": bool(part_b_url),
        "run_available": not missing,
        "confirmation_required": True,
        "missing_manifest_fields": missing,
        "repository_url": distribution.get("repository_url"),
        "release_url": distribution.get("release_url"),
        "part_b_url": part_b_url,
        "artifacts": [
            {field: envelope[field] for field in ("file", "bytes", "sha256", "url")},
            {field: ciphertext[field] for field in ("file", "bytes", "sha256", "url")},
        ],
    }


def _validated_url(value: str, *, source: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise DataPackError(f"AITA data-pack {source} URL is invalid") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or host not in _ALLOWED_DOWNLOAD_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise DataPackError(
            f"AITA data-pack {source} URL must be credential-free HTTPS on an allowed GitHub release host"
        )
    if host == "github.com" and "/releases/download/" not in parsed.path:
        raise DataPackError(
            f"AITA data-pack {source} URL must identify a GitHub release download"
        )
    return value


def _validated_reference_url(value: str, *, source: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise DataPackError(f"AITA data-pack {source} URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise DataPackError(
            f"AITA data-pack {source} URL must be credential-free HTTPS"
        )
    return value


def _download_verified(
    *,
    client: httpx.Client,
    artifact: Mapping[str, Any],
    output: Path,
) -> None:
    current_url = _validated_url(artifact["url"], source=artifact["file"])
    for redirect_count in range(_MAX_REDIRECTS + 1):
        try:
            request = client.build_request("GET", current_url, headers=_DOWNLOAD_HEADERS)
            for sensitive_header in ("authorization", "cookie", "proxy-authorization"):
                request.headers.pop(sensitive_header, None)
            response = client.send(
                request,
                stream=True,
                follow_redirects=False,
                auth=None,
            )
        except httpx.HTTPError as exc:
            raise DataPackError(f"AITA data-pack download failed for {artifact['file']}") from exc
        try:
            if response.status_code in _REDIRECT_STATUSES:
                if redirect_count >= _MAX_REDIRECTS:
                    raise DataPackError("AITA data-pack download exceeded the redirect limit")
                location = response.headers.get("location")
                if not location:
                    raise DataPackError("AITA data-pack redirect did not provide a location")
                current_url = _validated_url(
                    urljoin(current_url, location),
                    source=f"redirect for {artifact['file']}",
                )
                continue
            if response.status_code != 200:
                raise DataPackError(
                    f"AITA data-pack download returned HTTP {response.status_code} for {artifact['file']}"
                )
            content_encoding = response.headers.get("content-encoding")
            if content_encoding and content_encoding.lower() != "identity":
                raise DataPackError("AITA data-pack download used an unsupported content encoding")
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise DataPackError("AITA data-pack response Content-Length is invalid") from exc
                if declared_size != artifact["bytes"]:
                    raise DataPackError(
                        f"AITA data-pack byte count mismatch for {artifact['file']}"
                    )

            digest = hashlib.sha256()
            size = 0
            try:
                with output.open("xb") as handle:
                    for chunk in response.iter_raw():
                        size += len(chunk)
                        if size > artifact["bytes"]:
                            raise DataPackError(
                                f"AITA data-pack byte count mismatch for {artifact['file']}"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            except httpx.HTTPError as exc:
                raise DataPackError(f"AITA data-pack download failed for {artifact['file']}") from exc
            except OSError as exc:
                raise DataPackError(f"AITA data-pack could not write {artifact['file']}") from exc
            if size != artifact["bytes"]:
                raise DataPackError(f"AITA data-pack byte count mismatch for {artifact['file']}")
            if digest.hexdigest() != artifact["sha256"]:
                raise DataPackError(f"AITA data-pack SHA-256 mismatch for {artifact['file']}")
            try:
                output.chmod(0o644)
            except OSError as exc:
                raise DataPackError(f"AITA data-pack could not secure {artifact['file']}") from exc
            return
        finally:
            response.close()
    raise DataPackError("AITA data-pack download exceeded the redirect limit")


def _verify_envelope_identity(
    registry: Mapping[str, Any],
    envelope_path: Path,
    ciphertext: Mapping[str, Any],
) -> None:
    try:
        envelope = json.loads(
            envelope_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataPackError("AITA data-pack envelope is not valid JSON") from exc
    if not isinstance(envelope, dict):
        raise DataPackError("AITA data-pack envelope must be a JSON object")
    identity = registry["identity"]
    expected = {
        "schema": "antisycophancy-sealed-pack-v1",
        "pack_id": registry["pack_id"],
        "pack_version": registry["pack_version"],
        "pair_count": registry["pair_count"],
        "cipher_suite": identity.get("cipher_suite"),
        "canonicalization_version": identity.get("canonicalization_version"),
        "plaintext_identity_sha256": identity.get("plaintext_identity_sha256"),
        "key_scheme": identity.get("key_scheme"),
        "ciphertext_file": ciphertext["file"],
        "ciphertext_bytes": ciphertext["bytes"],
        "ciphertext_sha256": ciphertext["sha256"],
    }
    mismatched = sorted(field for field, value in expected.items() if envelope.get(field) != value)
    if mismatched:
        raise DataPackError(
            "AITA data-pack envelope does not match the registry: " + ", ".join(mismatched)
        )


def _verified_artifacts(
    destination: Path,
    *,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if destination.is_symlink() or not destination.is_dir():
        raise DataPackError("AITA data-pack directory is missing or unsafe")
    envelope = _artifact(registry, "envelope")
    ciphertext = _artifact(registry, "ciphertext")
    verified: list[dict[str, Any]] = []
    for item in (envelope, ciphertext):
        path = destination / item["file"]
        try:
            if path.is_symlink() or not path.is_file():
                raise DataPackError(f"AITA data-pack artifact is missing or unsafe: {item['file']}")
            size = path.stat().st_size
            if size != item["bytes"]:
                raise DataPackError(f"AITA data-pack byte count mismatch for {item['file']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except DataPackError:
            raise
        except OSError as exc:
            raise DataPackError(f"AITA data-pack artifact is unreadable: {item['file']}") from exc
        if digest != item["sha256"]:
            raise DataPackError(f"AITA data-pack SHA-256 mismatch for {item['file']}")
        verified.append({field: item[field] for field in ("file", "bytes", "sha256")})
    _verify_envelope_identity(registry, destination / envelope["file"], ciphertext)
    return verified


def _receipt(
    *,
    schema_version: str,
    registry: Mapping[str, Any],
    registry_sha256: str,
    artifacts: list[dict[str, Any]],
    include_timestamp: bool,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": schema_version,
        "verified": True,
        "pack_id": registry["pack_id"],
        "pack_version": registry["pack_version"],
        "pair_count": registry["pair_count"],
        "registry_sha256": registry_sha256,
        "artifacts": artifacts,
        "part_b_included": False,
        "plaintext_materialized": False,
    }
    if include_timestamp:
        receipt["downloaded_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return receipt


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o644)
    except OSError as exc:
        raise DataPackError("AITA data-pack receipt could not be written") from exc


def fetch_data_pack(
    destination: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    confirmed: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Download a public envelope and ciphertext after explicit consent."""
    if confirmed is not True:
        raise DataPackError("AITA data-pack download requires explicit confirmation")
    registry, registry_sha256 = _read_registry(registry_path)
    envelope = _artifact(registry, "envelope")
    ciphertext = _artifact(registry, "ciphertext")
    missing = [kind for kind, item in (("envelope", envelope), ("ciphertext", ciphertext)) if not item["url"]]
    if missing:
        names = ", ".join(f"distribution.{kind}_url" for kind in missing)
        raise DataPackError(
            f"AITA data-pack registry does not publish direct asset URLs yet: {names}"
        )
    destination_path = Path(destination).absolute()
    if destination_path.exists() or destination_path.is_symlink():
        raise DataPackError("AITA data-pack destination must not already exist")
    parent = destination_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise DataPackError("AITA data-pack destination parent is missing or unsafe")

    owned_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(30.0),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        staging: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=parent)
        )
    except OSError as exc:
        if owned_client:
            active_client.close()
        raise DataPackError("AITA data-pack staging directory could not be created") from exc
    try:
        _download_verified(client=active_client, artifact=envelope, output=staging / envelope["file"])
        _download_verified(client=active_client, artifact=ciphertext, output=staging / ciphertext["file"])
        artifacts = _verified_artifacts(staging, registry=registry)
        receipt = _receipt(
            schema_version=RECEIPT_SCHEMA,
            registry=registry,
            registry_sha256=registry_sha256,
            artifacts=artifacts,
            include_timestamp=True,
        )
        _write_receipt(staging / RECEIPT_FILE, receipt)
        if destination_path.exists() or destination_path.is_symlink():
            raise DataPackError("AITA data-pack destination appeared during download")
        try:
            os.rename(staging, destination_path)
        except OSError as exc:
            raise DataPackError("AITA data-pack directory could not be published atomically") from exc
        staging = None
        return receipt
    finally:
        if owned_client:
            active_client.close()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def verify_data_pack(
    destination: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Recheck an existing envelope and ciphertext against the local registry."""
    registry, registry_sha256 = _read_registry(registry_path)
    artifacts = _verified_artifacts(Path(destination).absolute(), registry=registry)
    return _receipt(
        schema_version=VERIFY_SCHEMA,
        registry=registry,
        registry_sha256=registry_sha256,
        artifacts=artifacts,
        include_timestamp=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, fetch, or verify the separate AITA sealed data pack.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser(
        "status",
        help="Show registry availability without networking",
        allow_abbrev=False,
    )
    status.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    status.add_argument("--json", action="store_true")

    fetch = commands.add_parser(
        "fetch",
        help="Download and verify the public envelope and ciphertext",
        allow_abbrev=False,
    )
    fetch.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    fetch.add_argument("--destination", required=True)
    fetch.add_argument(
        "--confirm-download",
        action="store_true",
        help="Confirm the exact registry-listed network download",
    )
    fetch.add_argument("--json", action="store_true")

    verify = commands.add_parser(
        "verify",
        help="Recheck an existing pack against the registry",
        allow_abbrev=False,
    )
    verify.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    verify.add_argument("--destination", required=True)
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = registry_status(args.registry)
        elif args.command == "fetch":
            result = fetch_data_pack(
                args.destination,
                registry_path=args.registry,
                confirmed=args.confirm_download,
            )
        elif args.command == "verify":
            result = verify_data_pack(args.destination, registry_path=args.registry)
        else:  # pragma: no cover - argparse constrains this value.
            raise DataPackError("unsupported AITA data-pack command")
    except DataPackError as exc:
        payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload, sort_keys=True) if args.json else f"AITA data pack blocked: {exc}")
        return 1

    payload = {"ok": True, **result}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    elif args.command == "status":
        state = "available" if result["download_available"] else "not yet available"
        print(f"AITA data pack {result['pack_id']} {result['pack_version']}: {state}")
    elif args.command == "fetch":
        print(f"Downloaded and verified AITA data pack at {args.destination}")
    else:
        print(f"Verified AITA data pack at {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
