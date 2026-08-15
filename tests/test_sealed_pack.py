from __future__ import annotations

import hashlib
import json

import pytest

from suite_tools.sealed_pack import (
    SealedPackError,
    canonical_plaintext_bytes,
    main,
    open_sealed_pack,
    open_sealed_pack_path,
    seal_files,
    write_sealed_pack_release,
)


def test_sealed_pack_round_trip_preserves_exact_file_bytes():
    files = {
        "flip.labels.json": b'{"labels":{"synthetic-pair":"YTA"}}\n',
        "flip.csv": b"id,flipped_story\nsynthetic-pair,synthetic reversal\n",
        "og.csv": b"id,original_post\nsynthetic-pair,synthetic original\r\n",
        "selection.yaml": b"items:\n  - index: 0\n",
    }

    sealed = seal_files(
        files,
        pack_id="synthetic-aita-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    opened = open_sealed_pack(
        sealed.envelope,
        sealed.ciphertext,
        key_part_b=sealed.key_part_b,
    )

    assert opened.files == files
    assert sealed.envelope["plaintext_identity_sha256"] == hashlib.sha256(
        canonical_plaintext_bytes(files)
    ).hexdigest()
    assert sealed.envelope["ciphertext_sha256"] == hashlib.sha256(
        sealed.ciphertext
    ).hexdigest()
    assert len(sealed.key_part_a) == 22
    assert len(sealed.key_part_b) == 21
    assert sealed.key_part_b not in repr(sealed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda envelope: envelope.__setitem__("pack_id", "other-pack"), "authentication"),
        (lambda envelope: envelope.__setitem__("pair_count", 2), "authentication"),
        (
            lambda envelope: envelope.__setitem__("plaintext_identity_sha256", "0" * 64),
            "authentication",
        ),
    ],
)
def test_sealed_pack_rejects_authenticated_metadata_tampering(mutation, message):
    sealed = seal_files(
        {"synthetic.txt": b"SYNTHETIC"},
        pack_id="synthetic-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    envelope = dict(sealed.envelope)
    mutation(envelope)

    with pytest.raises(SealedPackError, match=message):
        open_sealed_pack(envelope, sealed.ciphertext, key_part_b=sealed.key_part_b)


def test_sealed_pack_rejects_wrong_key_and_ciphertext_tampering():
    sealed = seal_files(
        {"synthetic.txt": b"SYNTHETIC"},
        pack_id="synthetic-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )

    with pytest.raises(SealedPackError, match="key fragments"):
        open_sealed_pack(sealed.envelope, sealed.ciphertext, key_part_b="A" * 21)

    tampered = bytearray(sealed.ciphertext)
    tampered[-1] ^= 1
    with pytest.raises(SealedPackError, match="ciphertext digest"):
        open_sealed_pack(sealed.envelope, bytes(tampered), key_part_b=sealed.key_part_b)


def test_sealed_pack_path_refuses_ciphertext_symlink_escape(tmp_path):
    sealed = seal_files(
        {"synthetic.txt": b"SYNTHETIC"},
        pack_id="synthetic-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    external = tmp_path.parent / f"{tmp_path.name}-external.sealed"
    external.write_bytes(sealed.ciphertext)
    (tmp_path / "payload.sealed").symlink_to(external)
    envelope = dict(sealed.envelope, ciphertext_file="payload.sealed")
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps(envelope))

    with pytest.raises(SealedPackError, match="ciphertext"):
        open_sealed_pack_path(envelope_path, key_part_b=sealed.key_part_b)


def test_release_builder_keeps_part_b_outside_public_artifacts(tmp_path):
    release_dir = tmp_path / "public-pack"
    part_b_path = tmp_path / "paper-supplement-part-b.txt"
    files = {"synthetic.txt": b"SYNTHETIC PACK CONTENT"}

    receipt = write_sealed_pack_release(
        files,
        output_dir=release_dir,
        key_part_b_path=part_b_path,
        pack_id="synthetic-pack",
        pack_version="v1",
        pair_count=1,
        key_part_b_locator="paper supplement, sealed-pack appendix",
        correction_contact="research@example.invalid",
    )

    part_b = part_b_path.read_text().strip()
    envelope_path = release_dir / receipt["envelope_file"]
    ciphertext_path = release_dir / receipt["ciphertext_file"]
    public_bytes = envelope_path.read_bytes() + ciphertext_path.read_bytes()
    assert len(part_b) == 21
    assert part_b.encode() not in public_bytes
    assert part_b not in json.dumps(receipt)
    assert part_b_path.stat().st_mode & 0o077 == 0
    assert open_sealed_pack_path(envelope_path, key_part_b=part_b).files == files


def test_sealed_pack_build_cli_emits_prompt_free_receipt(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "synthetic.txt").write_text("SYNTHETIC CLI PACK\n")
    output = tmp_path / "public"
    part_b = tmp_path / "paper" / "part-b.txt"

    result = main(
        [
            "build",
            "--source-dir",
            str(source),
            "--files",
            "synthetic.txt",
            "--output",
            str(output),
            "--part-b-output",
            str(part_b),
            "--pack-id",
            "synthetic-cli-pack",
            "--pack-version",
            "v1",
            "--pair-count",
            "1",
            "--part-b-locator",
            "paper supplement",
            "--correction-contact",
            "research@example.invalid",
            "--json",
        ]
    )

    printed = capsys.readouterr().out
    assert result == 0
    assert json.loads(printed)["ok"] is True
    assert part_b.read_text().strip() not in printed


def test_release_builder_does_not_publish_pack_when_part_b_write_fails(tmp_path, monkeypatch):
    release_dir = tmp_path / "public-pack"
    part_b_path = tmp_path / "paper" / "part-b.txt"
    real_open = __import__("os").open

    def fail_part_b_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None and str(path) == str(part_b_path.resolve()):
            raise OSError("synthetic Part-B write failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("suite_tools.sealed_pack.os.open", fail_part_b_open)

    with pytest.raises(OSError, match="synthetic Part-B write failure"):
        write_sealed_pack_release(
            {"synthetic.txt": b"SYNTHETIC PACK CONTENT"},
            output_dir=release_dir,
            key_part_b_path=part_b_path,
            pack_id="synthetic-pack",
            pack_version="v1",
            pair_count=1,
            key_part_b_locator="paper supplement",
            correction_contact="research@example.invalid",
        )

    assert not release_dir.exists()
    assert not part_b_path.exists()
