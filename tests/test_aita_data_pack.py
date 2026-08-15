from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from suite_tools.aita_data_pack import (
    DataPackError,
    RECEIPT_FILE,
    build_parser,
    fetch_data_pack,
    main,
    registry_status,
    verify_data_pack,
)


def _envelope(ciphertext: bytes) -> bytes:
    value = {
        "schema": "antisycophancy-sealed-pack-v1",
        "pack_id": "aita-reversed-n20",
        "pack_version": "v1",
        "pair_count": 20,
        "cipher_suite": "AES-256-GCM",
        "canonicalization_version": "canonical-json-exact-bytes-v1",
        "plaintext_identity_sha256": "a" * 64,
        "key_scheme": "public-base64url-split-22-21-v1",
        "ciphertext_file": "pack.sealed",
        "ciphertext_bytes": len(ciphertext),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _registry(
    tmp_path: Path,
    *,
    envelope: bytes | None = None,
    ciphertext: bytes = b"sealed-fixture",
    envelope_url: str | None = "https://github.com/example/data/releases/download/v1/pack.envelope.json",
    ciphertext_url: str | None = "https://github.com/example/data/releases/download/v1/pack.sealed",
    part_b_url: str | None = "https://example.org/paper/aita-part-b",
) -> Path:
    if envelope is None:
        envelope = _envelope(ciphertext)
    distribution: dict[str, object] = {
        "mode": "separate-signed-release",
        "ciphertext_file": "pack.sealed",
        "ciphertext_bytes": len(ciphertext),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "envelope_file": "pack.envelope.json",
        "envelope_bytes": len(envelope),
        "envelope_sha256": hashlib.sha256(envelope).hexdigest(),
    }
    if envelope_url is not None:
        distribution["envelope_url"] = envelope_url
    if ciphertext_url is not None:
        distribution["ciphertext_url"] = ciphertext_url
    manifest = {
        "schema_version": "aita-sealed-pack-registry-v1",
        "pack_id": "aita-reversed-n20",
        "pack_version": "v1",
        "pair_count": 20,
        "distribution": distribution,
        "identity": {
            "plaintext_identity_sha256": "a" * 64,
            "canonicalization_version": "canonical-json-exact-bytes-v1",
            "cipher_suite": "AES-256-GCM",
            "key_scheme": "public-base64url-split-22-21-v1",
        },
        "unlock": {"part_b_url": part_b_url},
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_status_reports_that_canonical_preview_has_no_download_or_unlock_urls(tmp_path: Path):
    registry = _registry(
        tmp_path,
        envelope_url=None,
        ciphertext_url=None,
        part_b_url=None,
    )

    status = registry_status(registry)

    assert status == {
        "schema_version": "aita-data-pack-status-v1",
        "pack_id": "aita-reversed-n20",
        "pack_version": "v1",
        "pair_count": 20,
        "download_available": False,
        "unlock_available": False,
        "run_available": False,
        "confirmation_required": True,
        "missing_manifest_fields": [
            "distribution.ciphertext_url",
            "distribution.envelope_url",
            "unlock.part_b_url",
        ],
        "repository_url": None,
        "release_url": None,
        "part_b_url": None,
        "artifacts": [
            {
                "file": "pack.envelope.json",
                "bytes": len(_envelope(b"sealed-fixture")),
                "sha256": hashlib.sha256(_envelope(b"sealed-fixture")).hexdigest(),
                "url": None,
            },
            {
                "file": "pack.sealed",
                "bytes": len(b"sealed-fixture"),
                "sha256": hashlib.sha256(b"sealed-fixture").hexdigest(),
                "url": None,
            },
        ],
    }


def test_status_reports_full_run_availability_with_paper_locator(tmp_path: Path):
    status = registry_status(_registry(tmp_path))

    assert status["download_available"] is True
    assert status["unlock_available"] is True
    assert status["run_available"] is True
    assert status["missing_manifest_fields"] == []
    assert status["part_b_url"] == "https://example.org/paper/aita-part-b"


def test_status_rejects_an_unsafe_part_b_locator(tmp_path: Path):
    registry = _registry(tmp_path, part_b_url="http://example.org/paper/aita-part-b")

    with pytest.raises(DataPackError, match="Part B URL must be credential-free HTTPS"):
        registry_status(registry)


def test_fetch_confirmation_flag_does_not_accept_an_abbreviation(tmp_path: Path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "fetch",
                "--destination",
                str(tmp_path / "pack"),
                "--confirm",
            ]
        )


def test_fetch_fails_clearly_when_registry_has_no_direct_asset_urls(tmp_path: Path):
    registry = _registry(tmp_path, envelope_url=None, ciphertext_url=None)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match="does not publish direct asset URLs"):
            fetch_data_pack(
                tmp_path / "pack",
                registry_path=registry,
                confirmed=True,
                client=client,
            )

    assert requests == []
    assert not (tmp_path / "pack").exists()


def test_fetch_publishes_only_verified_assets_and_prompt_free_receipt(tmp_path: Path):
    ciphertext = b"sealed-fixture"
    envelope = _envelope(ciphertext)
    registry = _registry(tmp_path, envelope=envelope, ciphertext=ciphertext)
    responses = {
        "/envelope": envelope,
        "/ciphertext": ciphertext,
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.scheme == "https"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        if request.url.host == "github.com":
            suffix = "envelope" if request.url.path.endswith(".json") else "ciphertext"
            return httpx.Response(
                302,
                headers={"location": f"https://release-assets.githubusercontent.com/{suffix}"},
            )
        body = responses[request.url.path]
        return httpx.Response(
            200,
            headers={"content-length": str(len(body))},
            stream=httpx.ByteStream(body),
        )

    destination = tmp_path / "downloaded-pack"
    with httpx.Client(
        auth=("must-not", "leak"),
        cookies={"must-not": "leak"},
        transport=httpx.MockTransport(handler),
    ) as client:
        receipt = fetch_data_pack(
            destination,
            registry_path=registry,
            confirmed=True,
            client=client,
        )

    assert [request.url.host for request in requests] == [
        "github.com",
        "release-assets.githubusercontent.com",
        "github.com",
        "release-assets.githubusercontent.com",
    ]
    assert (destination / "pack.envelope.json").read_bytes() == envelope
    assert (destination / "pack.sealed").read_bytes() == ciphertext
    assert json.loads((destination / RECEIPT_FILE).read_text()) == receipt
    assert receipt["schema_version"] == "aita-data-pack-download-receipt-v1"
    assert receipt["verified"] is True
    assert receipt["part_b_included"] is False
    assert "prompt" not in json.dumps(receipt).lower()
    assert "key_part_b" not in json.dumps(receipt).lower()

    verified = verify_data_pack(destination, registry_path=registry)
    assert verified["verified"] is True
    assert verified["artifacts"] == receipt["artifacts"]


@pytest.mark.parametrize(
    ("bad_ciphertext", "message"),
    [
        (b"short", "byte count mismatch"),
        (b"sealed-fixture!", "byte count mismatch"),
        (b"tamper-fixture", "SHA-256 mismatch"),
    ],
)
def test_fetch_rejects_wrong_size_or_hash_and_removes_staging(
    tmp_path: Path,
    bad_ciphertext: bytes,
    message: str,
):
    ciphertext = b"sealed-fixture"
    envelope = _envelope(ciphertext)
    registry = _registry(tmp_path, envelope=envelope, ciphertext=ciphertext)

    def handler(request: httpx.Request) -> httpx.Response:
        body = envelope if request.url.path.endswith(".json") else bad_ciphertext
        return httpx.Response(200, stream=httpx.ByteStream(body))

    destination = tmp_path / "downloaded-pack"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match=message):
            fetch_data_pack(
                destination,
                registry_path=registry,
                confirmed=True,
                client=client,
            )

    assert not destination.exists()
    assert list(tmp_path.glob(".downloaded-pack.*")) == []


def test_fetch_rejects_non_https_registry_url_before_network(tmp_path: Path):
    registry = _registry(
        tmp_path,
        envelope_url="http://github.com/example/data/releases/download/v1/pack.envelope.json",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match="credential-free HTTPS"):
            fetch_data_pack(
                tmp_path / "downloaded-pack",
                registry_path=registry,
                confirmed=True,
                client=client,
            )

    assert requests == []


def test_fetch_revalidates_redirect_hosts(tmp_path: Path):
    registry = _registry(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://example.com/pack"})

    destination = tmp_path / "downloaded-pack"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match="allowed GitHub release host"):
            fetch_data_pack(
                destination,
                registry_path=registry,
                confirmed=True,
                client=client,
            )

    assert len(requests) == 1
    assert not destination.exists()
    assert list(tmp_path.glob(".downloaded-pack.*")) == []


@pytest.mark.parametrize("as_symlink", [False, True])
def test_fetch_refuses_existing_or_symlink_destination_before_network(
    tmp_path: Path,
    as_symlink: bool,
):
    registry = _registry(tmp_path)
    destination = tmp_path / "downloaded-pack"
    if as_symlink:
        destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    else:
        destination.mkdir()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match="must not already exist"):
            fetch_data_pack(
                destination,
                registry_path=registry,
                confirmed=True,
                client=client,
            )

    assert requests == []


def test_verify_rechecks_existing_artifact_bytes(tmp_path: Path):
    ciphertext = b"sealed-fixture"
    envelope = _envelope(ciphertext)
    registry = _registry(tmp_path, envelope=envelope, ciphertext=ciphertext)
    destination = tmp_path / "downloaded-pack"
    destination.mkdir()
    (destination / "pack.envelope.json").write_bytes(envelope)
    (destination / "pack.sealed").write_bytes(ciphertext)

    assert verify_data_pack(destination, registry_path=registry)["verified"] is True

    (destination / "pack.sealed").write_bytes(b"tamper-fixture")
    with pytest.raises(DataPackError, match="SHA-256 mismatch"):
        verify_data_pack(destination, registry_path=registry)


def test_cli_status_is_json_and_fetch_without_confirmation_is_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    registry = _registry(tmp_path)

    assert main(["status", "--registry", str(registry), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ok"] is True
    assert status["confirmation_required"] is True

    assert (
        main(
            [
                "fetch",
                "--registry",
                str(registry),
                "--destination",
                str(tmp_path / "pack"),
                "--json",
            ]
        )
        == 1
    )
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["ok"] is False
    assert "explicit confirmation" in blocked["error"]


def test_fetch_requires_explicit_confirmation_before_any_network(tmp_path: Path):
    registry = _registry(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataPackError, match="explicit confirmation"):
            fetch_data_pack(
                tmp_path / "pack",
                registry_path=registry,
                confirmed=False,
                client=client,
            )

    assert requests == []
    assert not (tmp_path / "pack").exists()
