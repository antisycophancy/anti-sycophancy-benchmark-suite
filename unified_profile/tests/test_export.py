import json
from pathlib import Path

from unified_profile.export import export_bundle


FIXTURES = Path(__file__).parent / "fixtures"


def test_export_writes_manifest_coverage_and_report(tmp_path):
    output = tmp_path / "bundle"

    payload = export_bundle(
        sus_paths=[FIXTURES / "sus_conversations.json"],
        aita_paths=[FIXTURES / "aita_run"],
        epis_paths=[FIXTURES / "epis_scores"],
        output_dir=output,
    )

    assert (output / "manifest.json").exists()
    assert (output / "coverage.md").exists()
    assert (output / "unified_report.md").exists()
    assert payload["include_conversations"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runs"][0]["artifacts"][0]["sha256"]
    assert "higher = more sycophancy detected (worse)" in (output / "coverage.md").read_text()


def test_export_default_does_not_copy_conversations(tmp_path):
    output = tmp_path / "bundle"

    export_bundle(
        sus_paths=[FIXTURES / "sus_conversations.json"],
        aita_paths=[FIXTURES / "aita_run"],
        epis_paths=[],
        output_dir=output,
    )

    copied = [path.name for path in (output / "artifacts").rglob("*") if path.is_file()]
    assert "opus-4-6_item1_scores.json" in copied
    assert "opus-4-6_item1_side_a.json" not in copied


def test_export_include_conversations_copies_conversations(tmp_path):
    output = tmp_path / "bundle"

    export_bundle(
        sus_paths=[],
        aita_paths=[FIXTURES / "aita_run"],
        epis_paths=[],
        output_dir=output,
        include_conversations=True,
    )

    copied = [path.name for path in (output / "artifacts").rglob("*") if path.is_file()]
    assert "opus-4-6_item1_side_a.json" in copied
