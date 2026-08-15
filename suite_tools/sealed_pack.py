"""Authenticated, deliberately low-friction distribution for public data packs.

The key is reconstructable from two public fragments. This keeps plaintext out
of ordinary source indexing; it is not a confidentiality or DRM mechanism.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ENVELOPE_SCHEMA = "antisycophancy-sealed-pack-v1"
PLAINTEXT_SCHEMA = "antisycophancy-sealed-plaintext-v1"
CIPHER_SUITE = "AES-256-GCM"
CANONICALIZATION_VERSION = "canonical-json-exact-bytes-v1"
KEY_SCHEME = "public-base64url-split-22-21-v1"
KEY_PART_A_LENGTH = 22
KEY_PART_B_LENGTH = 21
KEY_CHECK_HEX_LENGTH = 16
MAX_CIPHERTEXT_BYTES = 64 * 1024 * 1024
_PORTABLE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class SealedPackError(ValueError):
    """Raised when a sealed pack cannot be authenticated or decoded."""


@dataclass(frozen=True)
class SealedPackBuild:
    envelope: dict
    ciphertext: bytes
    key_part_a: str = field(repr=False)
    key_part_b: str = field(repr=False)


@dataclass(frozen=True)
class OpenedSealedPack:
    envelope: dict
    files: dict[str, bytes]

    def read_bytes(self, path: str) -> bytes:
        try:
            return self.files[path]
        except KeyError as exc:
            raise SealedPackError(f"sealed pack does not contain required file: {path}") from exc

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        try:
            return self.read_bytes(path).decode(encoding)
        except UnicodeDecodeError as exc:
            raise SealedPackError(f"sealed pack file is not valid {encoding}: {path}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SealedPackError("sealed pack metadata is not canonical JSON") from exc
    return text.encode("utf-8")


def _portable_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise SealedPackError("sealed pack paths must be nonempty strings")
    if "\\" in path or "\n" in path or "\r" in path or "\x00" in path:
        raise SealedPackError("sealed pack path is not portable")
    pure = PurePosixPath(path)
    if pure.is_absolute() or path != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SealedPackError("sealed pack path must be a normalized relative path")
    if not all(_PORTABLE_SEGMENT.fullmatch(part) for part in pure.parts):
        raise SealedPackError("sealed pack path contains unsupported characters")
    return path


def canonical_plaintext_bytes(files: Mapping[str, bytes]) -> bytes:
    """Encode exact source bytes into the versioned deterministic container."""
    if not isinstance(files, Mapping) or not files:
        raise SealedPackError("sealed pack requires at least one file")
    records = []
    seen: set[str] = set()
    for raw_path, raw_bytes in sorted(files.items(), key=lambda item: str(item[0])):
        path = _portable_path(raw_path)
        if path in seen:
            raise SealedPackError(f"sealed pack contains duplicate path: {path}")
        seen.add(path)
        if not isinstance(raw_bytes, bytes):
            raise SealedPackError(f"sealed pack file must be bytes: {path}")
        records.append(
            {
                "bytes_b64": base64.b64encode(raw_bytes).decode("ascii"),
                "path": path,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        )
    return _canonical_json_bytes({"files": records, "schema": PLAINTEXT_SCHEMA})


def _decode_plaintext(payload: bytes) -> dict[str, bytes]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SealedPackError("sealed pack plaintext container is invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != PLAINTEXT_SCHEMA:
        raise SealedPackError("sealed pack plaintext schema is unsupported")
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise SealedPackError("sealed pack plaintext file inventory is invalid")
    files: dict[str, bytes] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes_b64", "path", "sha256"}:
            raise SealedPackError("sealed pack plaintext file record is invalid")
        path = _portable_path(record.get("path"))
        if path in files:
            raise SealedPackError(f"sealed pack contains duplicate path: {path}")
        try:
            raw = base64.b64decode(record.get("bytes_b64"), validate=True)
        except (TypeError, ValueError) as exc:
            raise SealedPackError(f"sealed pack file encoding is invalid: {path}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if not isinstance(record.get("sha256"), str) or not hmac.compare_digest(digest, record["sha256"]):
            raise SealedPackError(f"sealed pack file digest mismatch: {path}")
        files[path] = raw
    if canonical_plaintext_bytes(files) != payload:
        raise SealedPackError("sealed pack plaintext is not canonically encoded")
    return files


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str, *, expected_bytes: int, field: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise SealedPackError(f"sealed pack {field} is invalid")
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise SealedPackError(f"sealed pack {field} is invalid") from exc
    if len(raw) != expected_bytes or _b64url_encode(raw) != value:
        raise SealedPackError(f"sealed pack {field} has the wrong length")
    return raw


def _key_check(key: bytes) -> str:
    return hashlib.sha256(b"antisycophancy-sealed-pack-key-check-v1\x00" + key).hexdigest()[
        :KEY_CHECK_HEX_LENGTH
    ]


def _aad_fields(envelope: Mapping[str, object]) -> dict[str, object]:
    names = (
        "schema",
        "pack_id",
        "pack_version",
        "pair_count",
        "cipher_suite",
        "canonicalization_version",
        "plaintext_identity_sha256",
        "key_scheme",
        "key_check",
    )
    try:
        return {name: envelope[name] for name in names}
    except KeyError as exc:
        raise SealedPackError(f"sealed pack envelope is missing field: {exc.args[0]}") from exc


def _validate_identity_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise SealedPackError(f"sealed pack {field} is invalid")
    return value


def seal_files(
    files: Mapping[str, bytes],
    *,
    pack_id: str,
    pack_version: str,
    pair_count: int,
    key: bytes,
    nonce: bytes,
) -> SealedPackBuild:
    """Seal one immutable pack. Maintainer callers generate key and nonce once."""
    pack_id = _validate_identity_text(pack_id, field="pack_id")
    pack_version = _validate_identity_text(pack_version, field="pack_version")
    if not isinstance(pair_count, int) or isinstance(pair_count, bool) or pair_count <= 0:
        raise SealedPackError("sealed pack pair_count must be a positive integer")
    if not isinstance(key, bytes) or len(key) != 32:
        raise SealedPackError("sealed pack key must be exactly 32 bytes")
    if not isinstance(nonce, bytes) or len(nonce) != 12:
        raise SealedPackError("sealed pack nonce must be exactly 12 bytes")

    plaintext = canonical_plaintext_bytes(files)
    encoded_key = _b64url_encode(key)
    key_part_a = encoded_key[:KEY_PART_A_LENGTH]
    key_part_b = encoded_key[KEY_PART_A_LENGTH:]
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "pack_id": pack_id,
        "pack_version": pack_version,
        "pair_count": pair_count,
        "cipher_suite": CIPHER_SUITE,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "plaintext_identity_sha256": hashlib.sha256(plaintext).hexdigest(),
        "key_scheme": KEY_SCHEME,
        "key_part_a": key_part_a,
        "key_part_b_length": KEY_PART_B_LENGTH,
        "key_check": _key_check(key),
        "nonce_b64url": _b64url_encode(nonce),
    }
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _canonical_json_bytes(_aad_fields(envelope)))
    envelope["ciphertext_sha256"] = hashlib.sha256(ciphertext).hexdigest()
    envelope["ciphertext_bytes"] = len(ciphertext)
    return SealedPackBuild(
        envelope=envelope,
        ciphertext=ciphertext,
        key_part_a=key_part_a,
        key_part_b=key_part_b,
    )


def open_sealed_pack(
    envelope: Mapping[str, object],
    ciphertext: bytes,
    *,
    key_part_b: str,
) -> OpenedSealedPack:
    """Authenticate and decrypt a pack without materializing plaintext files."""
    if not isinstance(envelope, Mapping) or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise SealedPackError("sealed pack envelope schema is unsupported")
    if envelope.get("cipher_suite") != CIPHER_SUITE:
        raise SealedPackError("sealed pack cipher suite is unsupported")
    if envelope.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise SealedPackError("sealed pack canonicalization version is unsupported")
    if envelope.get("key_scheme") != KEY_SCHEME:
        raise SealedPackError("sealed pack key scheme is unsupported")
    _validate_identity_text(envelope.get("pack_id"), field="pack_id")
    _validate_identity_text(envelope.get("pack_version"), field="pack_version")
    if not isinstance(envelope.get("pair_count"), int) or envelope["pair_count"] <= 0:
        raise SealedPackError("sealed pack pair_count is invalid")
    if not isinstance(ciphertext, bytes):
        raise SealedPackError("sealed pack ciphertext must be bytes")
    expected_bytes = envelope.get("ciphertext_bytes")
    if not isinstance(expected_bytes, int) or expected_bytes != len(ciphertext):
        raise SealedPackError("sealed pack ciphertext length mismatch")
    digest = hashlib.sha256(ciphertext).hexdigest()
    expected_digest = envelope.get("ciphertext_sha256")
    if not isinstance(expected_digest, str) or not hmac.compare_digest(digest, expected_digest):
        raise SealedPackError("sealed pack ciphertext digest mismatch")

    part_a = envelope.get("key_part_a")
    if not isinstance(part_a, str) or len(part_a) != KEY_PART_A_LENGTH:
        raise SealedPackError("sealed pack key part A is invalid")
    if not isinstance(key_part_b, str) or len(key_part_b) != KEY_PART_B_LENGTH:
        raise SealedPackError("sealed pack key part B has the wrong length")
    key = _b64url_decode(part_a + key_part_b, expected_bytes=32, field="reconstructed key")
    expected_key_check = envelope.get("key_check")
    if not isinstance(expected_key_check, str) or not hmac.compare_digest(_key_check(key), expected_key_check):
        raise SealedPackError("sealed pack key fragments do not match this pack")
    nonce = _b64url_decode(envelope.get("nonce_b64url"), expected_bytes=12, field="nonce")

    try:
        plaintext = AESGCM(key).decrypt(
            nonce,
            ciphertext,
            _canonical_json_bytes(_aad_fields(envelope)),
        )
    except InvalidTag as exc:
        raise SealedPackError("sealed pack authentication failed") from exc
    plaintext_digest = hashlib.sha256(plaintext).hexdigest()
    expected_plaintext_digest = envelope.get("plaintext_identity_sha256")
    if not isinstance(expected_plaintext_digest, str) or not hmac.compare_digest(
        plaintext_digest,
        expected_plaintext_digest,
    ):
        raise SealedPackError("sealed pack plaintext identity mismatch")
    files = _decode_plaintext(plaintext)
    return OpenedSealedPack(envelope=dict(envelope), files=files)


def open_sealed_pack_path(envelope_path: str | Path, *, key_part_b: str) -> OpenedSealedPack:
    """Open an envelope and its direct-sibling ciphertext without path escape."""
    path = Path(envelope_path)
    try:
        envelope = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SealedPackError("sealed pack envelope is unreadable") from exc
    if not isinstance(envelope, dict):
        raise SealedPackError("sealed pack envelope must be a JSON object")
    ciphertext_name = envelope.get("ciphertext_file")
    if (
        not isinstance(ciphertext_name, str)
        or not _PORTABLE_SEGMENT.fullmatch(ciphertext_name)
        or ciphertext_name in {".", ".."}
    ):
        raise SealedPackError("sealed pack ciphertext_file must name a direct sibling file")
    ciphertext_path = path.parent / ciphertext_name
    try:
        if path.is_symlink() or ciphertext_path.is_symlink():
            raise SealedPackError("sealed pack envelope and ciphertext must not be symlinks")
        size = ciphertext_path.stat().st_size
        if not ciphertext_path.is_file() or size > MAX_CIPHERTEXT_BYTES:
            raise SealedPackError("sealed pack ciphertext file is missing or too large")
        ciphertext = ciphertext_path.read_bytes()
    except OSError as exc:
        raise SealedPackError("sealed pack ciphertext file is unreadable") from exc
    return open_sealed_pack(envelope, ciphertext, key_part_b=key_part_b)


def write_sealed_pack_release(
    files: Mapping[str, bytes],
    *,
    output_dir: str | Path,
    key_part_b_path: str | Path,
    pack_id: str,
    pack_version: str,
    pair_count: int,
    key_part_b_locator: str,
    correction_contact: str,
) -> dict[str, object]:
    """Build one immutable public pack and a separately held Part-B file."""
    output = Path(output_dir).resolve()
    part_b_path = Path(key_part_b_path).resolve()
    if output.exists():
        raise SealedPackError("sealed-pack output directory must not already exist")
    if part_b_path.exists():
        raise SealedPackError("sealed-pack Part-B output must not already exist")
    if output == part_b_path or output in part_b_path.parents:
        raise SealedPackError("sealed-pack Part B must be outside the public pack directory")
    for field, value in (
        ("key_part_b_locator", key_part_b_locator),
        ("correction_contact", correction_contact),
    ):
        if not isinstance(value, str) or not value.strip() or any(ch in value for ch in "\r\n\x00"):
            raise SealedPackError(f"sealed-pack {field} is invalid")

    built = seal_files(
        files,
        pack_id=pack_id,
        pack_version=pack_version,
        pair_count=pair_count,
        key=secrets.token_bytes(32),
        nonce=secrets.token_bytes(12),
    )
    stem = f"{pack_id}-{pack_version}"
    ciphertext_file = f"{stem}.sealed"
    envelope_file = f"{stem}.envelope.json"
    public_envelope = dict(
        built.envelope,
        ciphertext_file=ciphertext_file,
        correction_contact=correction_contact,
        key_part_b_locator=key_part_b_locator,
        disclosure=(
            "Public reconstructable-key anti-indexing friction only; not confidentiality, "
            "DRM, rights clearance, or effective retraction."
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    part_b_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{stem}.", dir=output.parent) as temp_dir:
        staging = Path(temp_dir) / "pack"
        staging.mkdir(mode=0o755)
        ciphertext_path = staging / ciphertext_file
        envelope_path = staging / envelope_file
        ciphertext_path.write_bytes(built.ciphertext)
        envelope_path.write_text(
            json.dumps(public_envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        ciphertext_path.chmod(0o644)
        envelope_path.chmod(0o644)
        try:
            descriptor = os.open(part_b_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(built.key_part_b + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            part_b_path.chmod(0o600)
        except Exception:
            part_b_path.unlink(missing_ok=True)
            raise
        try:
            os.replace(staging, output)
        except Exception:
            part_b_path.unlink(missing_ok=True)
            raise

    return {
        "schema": "antisycophancy-sealed-pack-build-receipt-v1",
        "pack_id": pack_id,
        "pack_version": pack_version,
        "pair_count": pair_count,
        "envelope_file": envelope_file,
        "ciphertext_file": ciphertext_file,
        "ciphertext_sha256": built.envelope["ciphertext_sha256"],
        "plaintext_identity_sha256": built.envelope["plaintext_identity_sha256"],
        "key_part_b_locator": key_part_b_locator,
    }


def _read_source_files(source_dir: str | Path, members: str) -> dict[str, bytes]:
    root = Path(source_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise SealedPackError("sealed-pack source directory is missing or unsafe")
    requested = [member.strip() for member in members.split(",") if member.strip()]
    if not requested:
        raise SealedPackError("sealed-pack build requires at least one --files member")
    files: dict[str, bytes] = {}
    total = 0
    for raw_member in requested:
        member = _portable_path(raw_member)
        source = root.joinpath(*PurePosixPath(member).parts)
        try:
            if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
                raise SealedPackError(f"sealed-pack source member is missing or unsafe: {member}")
            raw = source.read_bytes()
        except OSError as exc:
            raise SealedPackError(f"sealed-pack source member is unreadable: {member}") from exc
        total += len(raw)
        if total > MAX_CIPHERTEXT_BYTES:
            raise SealedPackError("sealed-pack source inventory is too large")
        files[member] = raw
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a public reconstructable-key sealed data pack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Create one immutable sealed-pack release")
    build.add_argument("--source-dir", required=True)
    build.add_argument("--files", required=True, help="Comma-separated relative source members")
    build.add_argument("--output", required=True, help="New public pack directory")
    build.add_argument("--part-b-output", required=True, help="New private handoff file outside --output")
    build.add_argument("--pack-id", required=True)
    build.add_argument("--pack-version", required=True)
    build.add_argument("--pair-count", type=int, required=True)
    build.add_argument("--part-b-locator", required=True)
    build.add_argument("--correction-contact", required=True)
    build.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = _read_source_files(args.source_dir, args.files)
        receipt = write_sealed_pack_release(
            files,
            output_dir=args.output,
            key_part_b_path=args.part_b_output,
            pack_id=args.pack_id,
            pack_version=args.pack_version,
            pair_count=args.pair_count,
            key_part_b_locator=args.part_b_locator,
            correction_contact=args.correction_contact,
        )
    except SealedPackError as exc:
        payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload, sort_keys=True) if args.json else f"Sealed pack blocked: {exc}")
        return 1
    payload = {"ok": True, **receipt}
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
