import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from suite_tools.sealed_pack import seal_files
import pytest

from suite_tools.run_contract import stable_json_hash


REPO_ROOT = Path(__file__).resolve().parents[1]
AITA_ROOT = REPO_ROOT / "aita-bench"
PUBLIC_CURATED_ROOT = AITA_ROOT / "data" / "curated" / "aita_nta_paired_curated_v1"
REVERSED_N20V1_ROOT = AITA_ROOT / "data" / "curated" / "aita_reversed_n20_v1"
PRIVATE_CURATED_ROOT = (
    REPO_ROOT / "private_question_bank" / "aita" / "curated" / "aita_nta_paired_curated_v1"
)


def test_aita_runner_loads_locked_selection_from_sealed_pack_without_plaintext_files(tmp_path):
    runner = _load_aita_runner()
    original = "SYNTHETIC ORIGINAL SENTINEL"
    reversal = "SYNTHETIC REVERSAL SENTINEL"
    sealed = seal_files(
        {
            "flip.csv": f"id,flipped_story\nsynthetic-pair,{reversal}\n".encode(),
            "flip.labels.json": b'{"labels":{"synthetic-pair":"YTA"}}\n',
            "og.csv": f"id,original_post\nsynthetic-pair,{original}\n".encode(),
            "selection.yaml": b"items:\n  - index: 0\n    pair_id: synthetic-pair\n",
        },
        pack_id="synthetic-aita-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    envelope = dict(sealed.envelope)
    envelope["ciphertext_file"] = "synthetic-aita-pack.sealed"
    envelope_path = tmp_path / "synthetic-aita-pack.envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    (tmp_path / envelope["ciphertext_file"]).write_bytes(sealed.ciphertext)

    args = SimpleNamespace(
        items="99",
        item_selection=None,
        og_data=None,
        flip_data=None,
        paired_labels=None,
        sealed_pack=str(envelope_path),
        sealed_pack_key_part_b=sealed.key_part_b,
    )

    item_indices, items, flips = runner.load_nta_paired_items(args)

    assert item_indices == [0]
    assert items[0]["original"] == original
    assert flips[0] == reversal
    assert items[0]["pair_id"] == "synthetic-pair"
    assert args._sealed_pack_context["pack_id"] == "synthetic-aita-pack"
    assert original not in envelope_path.read_text()
    assert original.encode() not in (tmp_path / envelope["ciphertext_file"]).read_bytes()

    manifest = runner.build_dataset_manifest(args, "nta-paired", item_indices, items, flips)
    assert manifest["sealed_pack"] == {
        "pack_id": "synthetic-aita-pack",
        "pack_version": "v1",
        "pair_count": 1,
        "ciphertext_sha256": sealed.envelope["ciphertext_sha256"],
        "plaintext_identity_sha256": sealed.envelope["plaintext_identity_sha256"],
        "key_scheme": sealed.envelope["key_scheme"],
        "file_hashes": args._sealed_pack_context["file_hashes"],
    }
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    assert "path" not in manifest["files"][0]
    assert original not in serialized_manifest
    assert sealed.key_part_b not in serialized_manifest


def test_sealed_pack_key_environment_requires_explicit_noninteractive_opt_in(monkeypatch):
    runner = _load_aita_runner()
    monkeypatch.setenv("ANTISYCOPHANCY_AITA_PACK_KEY_PART_B", "E" * 21)
    prompted = []

    value = runner.acquire_sealed_pack_key_part_b(
        SimpleNamespace(sealed_key_part_b_from_env=False),
        prompt=lambda _message: prompted.append(True) or "P" * 21,
    )

    assert value == "P" * 21
    assert prompted == [True]
    assert os.environ["ANTISYCOPHANCY_AITA_PACK_KEY_PART_B"] == "E" * 21


def _curated_v1_selection_or_skip() -> dict:
    """curated_v1 was retired to the maintainer-private tree (superseded by the
    shipped aita_reversed_n20_v1 set). Its validation tests skip when its
    selection is not present in this checkout."""
    sel = PUBLIC_CURATED_ROOT / "N20.seed-20260526.selection.yaml"
    if not sel.exists():
        pytest.skip("curated_v1 retired to private; selection not present")
    return yaml.safe_load(sel.read_text())


def _private_curated_root() -> Path:
    required = [
        "POOL_150.jsonl",
        "EXCLUSIONS.jsonl",
        "SEMANTIC_REVIEW.jsonl",
        "POOL_150.manifest.json",
        "N50.seed-20260526.selection.yaml",
        "N50_EXTENSION.rows21-50.seed-20260526.selection.yaml",
    ]
    if not all((PRIVATE_CURATED_ROOT / name).exists() for name in required):
        pytest.skip("private AITA curated pool is not present in this checkout")
    return PRIVATE_CURATED_ROOT


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_csv_rows(path: Path) -> list[dict]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


_SPEAKER_VERDICT_RE = re.compile(
    r"\b("
    r"AITA|WIBTA|AITAH|WIBTAH|"
    r"am\s+i\s+(?:the\s+)?(?:asshole|ah|wrong|in\s+the\s+wrong|bad\s+(?:guy|one)|at\s+fault|unreasonable|being\s+unreasonable|mistaken|overreacting)|"
    r"was\s+i\s+(?:the\s+)?(?:asshole|ah|wrong|in\s+the\s+wrong|bad\s+(?:guy|one)|at\s+fault|unreasonable|mistaken|overreacting)|"
    r"would\s+i\s+be\s+(?:the\s+)?(?:asshole|ah|wrong)|"
    r"would\s+i\s+be\s+in\s+the\s+wrong|"
    r"did\s+i\s+do\s+something\s+wrong|do\s+i\s+do\s+something\s+wrong|"
    r"make\s+(?:me|us)\s+the\s+bad\s+(?:guy|one)|"
    r"are\s+we\s+(?:the\s+)?(?:assholes|ahs|wrong)|"
    r"were\s+we\s+(?:the\s+)?(?:assholes|ahs|wrong)|"
    r"is\s+it\s+wrong\s+(?:of|for)\s+(?:me|us)|"
    r"was\s+it\s+wrong\s+(?:of|for)\s+(?:me|us)|"
    r"does\s+it\s+seem\s+(?:unreasonable|mistaken)\s+for\s+(?:me|us)|"
    r"am\s+i\s+justified|was\s+i\s+justified|are\s+we\s+justified"
    r")\b",
    re.IGNORECASE,
)

_OPPOSING_PARTY_VERDICT_RE = re.compile(
    r"\b(?:would|are|were)\s+(?:they|he|she)\s+(?:be\s+)?(?:the\s+)?"
    r"(?:asshole|assholes|ah|wrong|at\s+fault)\b",
    re.IGNORECASE,
)


def _asks_for_speaker_verdict(text: str) -> bool:
    if not _SPEAKER_VERDICT_RE.search(text):
        return False
    # The row-1073 failure mode was a flipped prompt asking whether the other
    # party would be the asshole. Keep that out of the accepted pool.
    return not _OPPOSING_PARTY_VERDICT_RE.search(text[:260])


def _load_aita_runner():
    sys.path.insert(0, str(AITA_ROOT))
    from aita_bench import runner

    return runner


def test_aita_curated_pool_manifest_is_internally_consistent():
    private_root = _private_curated_root()
    pool = _load_jsonl(private_root / "POOL_150.jsonl")
    exclusions = _load_jsonl(private_root / "EXCLUSIONS.jsonl")
    semantic_review = _load_jsonl(private_root / "SEMANTIC_REVIEW.jsonl")
    manifest = json.loads((private_root / "POOL_150.manifest.json").read_text())
    selection = _curated_v1_selection_or_skip()
    n50_selection = yaml.safe_load((private_root / "N50.seed-20260526.selection.yaml").read_text())
    n50_extension = yaml.safe_load(
        (private_root / "N50_EXTENSION.rows21-50.seed-20260526.selection.yaml").read_text()
    )

    pool_indices = {row["index"] for row in pool}
    pool_pair_ids = {row["pair_id"] for row in pool}
    exclusion_indices = {row["index"] for row in exclusions}
    selected_indices = {row["index"] for row in selection["items"]}
    accepted_review_indices = {
        row["index"] for row in semantic_review if row["decision"] == "accepted"
    }
    excluded_review_indices = {
        row["index"] for row in semantic_review if row["decision"] == "excluded"
    }

    assert len(pool) == 150
    assert len(pool_indices) == 150
    assert len(pool_pair_ids) == 150
    assert len(selection["items"]) == 20
    assert selected_indices <= pool_indices
    assert not (pool_indices & exclusion_indices)
    assert manifest["status"] == "locked_semantic_review_pool"
    assert manifest["curation_policy"]["semantic_review_status"] == "complete"
    assert manifest["semantic_review"]["hash"] == stable_json_hash(semantic_review)
    assert manifest["semantic_review"]["rows"] == len(semantic_review)
    assert manifest["semantic_review"]["accepted"] == len(pool)
    assert manifest["semantic_review"]["excluded"] == len(excluded_review_indices)
    assert pool_indices == accepted_review_indices
    assert not (pool_indices & excluded_review_indices)
    assert not (selected_indices & excluded_review_indices)
    assert all(row["review_status"] == "semantic_review_passed" for row in pool)
    assert manifest["pool"]["hash"] == stable_json_hash(pool)
    assert manifest["pilot_sample"]["hash"] == stable_json_hash(selection["items"])
    assert manifest["pool"]["rows"] == len(pool)
    assert manifest["pilot_sample"]["rows"] == len(selection["items"])
    assert manifest["exclusions"]["rows"] == len(exclusions)
    assert selection["status"] == "locked_pilot_sample_semantic_reviewed"
    assert selection["pool_hash"] == manifest["pool"]["hash"]
    assert selection["sample_hash"] == manifest["pilot_sample"]["hash"]
    assert n50_selection["status"] == "locked_pilot_sample_semantic_reviewed"
    assert n50_extension["status"] == "locked_extension_sample_semantic_reviewed"
    assert n50_selection["pool_hash"] == manifest["pool"]["hash"]
    assert n50_extension["pool_hash"] == manifest["pool"]["hash"]
    assert n50_selection["sample_hash"] == stable_json_hash(n50_selection["items"])
    assert n50_extension["sample_hash"] == stable_json_hash(n50_extension["items"])
    assert n50_selection["parent_sample"]["sample_hash"] == selection["sample_hash"]
    assert n50_extension["parent_sample"]["sample_hash"] == selection["sample_hash"]
    assert n50_extension["combined_sample"]["sample_hash"] == n50_selection["sample_hash"]
    assert n50_selection["items"][: len(selection["items"])] == selection["items"]
    assert n50_selection["items"][len(selection["items"]) :] == n50_extension["items"]
    assert len(n50_selection["items"]) == 50
    assert len(n50_extension["items"]) == 30
    assert {row["index"] for row in n50_selection["items"]} <= pool_indices
    assert not (selected_indices & {row["index"] for row in n50_extension["items"]})


def test_aita_curated_pool_rejects_opposing_party_verdict_prompts():
    private_root = _private_curated_root()
    pool = _load_jsonl(private_root / "POOL_150.jsonl")
    flips = _load_csv_rows(AITA_ROOT / "data" / "AITA-NTA-FLIP.csv")

    assert 1073 not in {row["index"] for row in pool}

    misses = []
    for row in pool:
        idx = row["index"]
        if _OPPOSING_PARTY_VERDICT_RE.search(flips[idx]["flipped_story"][:260]):
            misses.append((idx, "side_b"))

    assert misses == []


def test_aita_n20_pilot_prompts_ask_for_speaker_verdicts():
    n20_selection = _curated_v1_selection_or_skip()
    selections = [n20_selection]
    private_n50_path = PRIVATE_CURATED_ROOT / "N50.seed-20260526.selection.yaml"
    if private_n50_path.exists():
        selections.append(yaml.safe_load(private_n50_path.read_text()))
    originals = _load_csv_rows(AITA_ROOT / "data" / "AITA-NTA-OG.csv")
    flips = _load_csv_rows(AITA_ROOT / "data" / "AITA-NTA-FLIP.csv")

    misses = []
    for selection in selections:
        for row in selection["items"]:
            idx = row["index"]
            if not _asks_for_speaker_verdict(originals[idx]["original_post"]):
                misses.append((idx, "side_a"))
            if not _asks_for_speaker_verdict(flips[idx]["flipped_story"]):
                misses.append((idx, "side_b"))

    assert misses == []


def test_aita_curated_hashes_match_runner_canonical_prompt_identity(tmp_path):
    runner = _load_aita_runner()
    pool_path = PRIVATE_CURATED_ROOT / "POOL_150.jsonl"
    pool = _load_jsonl(pool_path) if pool_path.exists() else []
    selection = _curated_v1_selection_or_skip()
    selection_path = PUBLIC_CURATED_ROOT / "N20.seed-20260526.selection.yaml"
    paired_labels = tmp_path / "AITA-NTA-FLIP.labels.json"
    paired_labels.write_text('{"default": "YTA", "labels": {}}\n')

    args = SimpleNamespace(
        items="1",
        dataset_mode="nta-paired",
        data=None,
        og_data=str(AITA_ROOT / "data" / "AITA-NTA-OG.csv"),
        flip_data=str(AITA_ROOT / "data" / "AITA-NTA-FLIP.csv"),
        paired_labels=str(paired_labels),
        item_selection=str(selection_path),
        allow_sample_fallback=False,
    )
    item_indices, items_by_idx, flips = runner.load_nta_paired_items(args)
    dataset_manifest = runner.build_dataset_manifest(
        args,
        "nta-paired",
        item_indices,
        items_by_idx,
        flips,
    )
    selected_by_idx = {row["item_idx"]: row for row in dataset_manifest["selected_pairs"]}
    pool_by_idx = {row["index"]: row for row in pool}

    for selected in selection["items"]:
        idx = selected["index"]
        canonical = selected_by_idx[idx]
        for key in ("pair_id", "source_pair_hash", "side_a_prompt_hash", "side_b_prompt_hash"):
            assert selected[key] == canonical[key]
            if pool_by_idx:
                assert pool_by_idx[idx][key] == canonical[key]


def test_aita_reversed_n20_v1_public_tree_contains_only_pack_metadata():
    assert {path.name for path in REVERSED_N20V1_ROOT.iterdir()} == {
        "PACK.md",
        "SCORING.md",
    }

    forbidden_plaintext = {
        "DATASET_CARD.md",
        "MANIFEST.json",
        "flip.csv",
        "flip.labels.json",
        "og.csv",
        "selection.yaml",
    }
    assert all(not (REVERSED_N20V1_ROOT / name).exists() for name in forbidden_plaintext)


def test_aita_reversed_n20_v1_registry_preserves_migration_identity():
    registry = json.loads((REPO_ROOT / "manifests" / "aita-sealed-pack-v1.json").read_text())

    assert registry["pack_id"] == "aita-reversed-n20"
    assert registry["pack_version"] == "v1"
    assert registry["pair_count"] == 20
    assert registry["distribution"]["mode"] == "separate-signed-release"
    assert registry["distribution"]["ciphertext_sha256"] == (
        "aad2080f05f0ffbb6169b8fae82838211ec9a75c51562da942351d72941a6e54"
    )
    assert registry["distribution"]["envelope_sha256"] == (
        "f5a57efa032ab0f754598a7ceda1168e625e3d3c95e37e13c359e6698a8d0bf1"
    )
    assert registry["identity"]["plaintext_identity_sha256"] == (
        "b88b4fcf2ba528f689020b8074f426b815def2ed31f4d020df73ad3d58fd2c1a"
    )
    assert registry["migration_evidence"]["byte_parity_verified"] is True
    assert set(registry["migration_evidence"]["files"]) == {
        "DATASET_CARD.md",
        "MANIFEST.json",
        "flip.csv",
        "flip.labels.json",
        "og.csv",
        "selection.yaml",
    }


def test_public_aita_data_contains_no_reddit_source_urls_or_posts():
    source_url_marker = b"reddit.com/r/" + b"AmItheAsshole/comments/"
    for path in (AITA_ROOT / "data").rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert source_url_marker not in payload

    sample = (AITA_ROOT / "data" / "AITA-YTA_sample.csv").read_text()
    assert sample.count("SYNTHETIC:") == 6
