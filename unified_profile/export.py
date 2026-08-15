"""Export traceable unified benchmark data bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from unified_profile.adapters import load_aita_results, load_epis_results, load_sus_results
from unified_profile.artifacts import discover_all_runs
from unified_profile.manifest import REPO_ROOT, manifests_to_dict
from unified_profile.models import model_label
from unified_profile.profile import UnifiedModelProfile, build_all_profiles
from unified_profile.report import generate_unified_report

TARGET_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "google/gemini-3-flash-preview",
    "therapeutic-harness/th-opus-4-6",
]


def _score(value: float | None) -> str:
    return "[gap]" if value is None else f"{value:.1f}"


def _coverage_markdown(profiles: list[UnifiedModelProfile]) -> str:
    by_model = {profile.model_id: profile for profile in profiles}
    model_ids = sorted(set(by_model) | set(TARGET_MODELS))

    lines = [
        "# Unified Benchmark Coverage",
        "",
        "All scores are 0-100, higher = more sycophancy detected (worse).",
        "Gaps are explicit and are not treated as zero.",
        "",
        "| Model | Safety N | Moral N | Epistemic N | Safety | Moral | Epistemic | Status |",
        "|-------|----------|---------|-------------|--------|-------|-----------|--------|",
    ]
    for model_id in model_ids:
        profile = by_model.get(model_id)
        label = profile.label if profile else model_label(model_id)
        sus_n = profile.sus_n if profile else 0
        aita_n = profile.aita_n if profile else 0
        epis_n = profile.epis_n if profile else 0
        safety = profile.safety_score if profile else None
        moral = profile.moral_score if profile else None
        epis = profile.epistemic_score if profile else None
        gaps = []
        if sus_n == 0:
            gaps.append("missing SUS")
        if aita_n == 0:
            gaps.append("missing AITA")
        if epis_n == 0:
            gaps.append("missing epistemic")
        status = "complete" if not gaps else "; ".join(gaps)
        lines.append(
            f"| {label} | {sus_n} | {aita_n} | {epis_n} | {_score(safety)} | "
            f"{_score(moral)} | {_score(epis)} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- SUS has the strongest current coverage.",
            "- AITA selected adaptive coverage currently includes a private served endpoint, Opus 4.6, and Sonnet 4.6.",
            "- Epistemic data is currently smoke-only and should not be treated as paper-quality.",
            "- Live benchmark collection remains deferred until artifact packaging is reviewed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_export_artifacts(manifest_dicts: list[dict], output_dir: Path, include_conversations: bool) -> list[str]:
    copied = []
    artifact_root = output_dir / "artifacts"
    for manifest in manifest_dicts:
        module = manifest["module"]
        for artifact in manifest["artifacts"]:
            if artifact["kind"] == "conversation" and not include_conversations:
                continue
            source = Path(artifact["path"])
            if not source.is_absolute():
                source = REPO_ROOT / source
            if not source.exists():
                continue
            dest = artifact_root / module / artifact["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(dest.relative_to(output_dir).as_posix())
    return sorted(copied)


def export_bundle(
    *,
    sus_paths: list[Path],
    aita_paths: list[Path],
    epis_paths: list[Path],
    output_dir: Path,
    include_conversations: bool = False,
) -> dict:
    """Write a traceable bundle and return its manifest payload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = discover_all_runs({"sus": sus_paths, "aita": aita_paths, "epistemic": epis_paths})
    manifest_dicts = manifests_to_dict(manifests)

    sus_data = load_sus_results(sus_paths) if sus_paths else {}
    aita_data = load_aita_results(aita_paths) if aita_paths else {}
    epis_data = load_epis_results(epis_paths) if epis_paths else {}
    profiles = build_all_profiles(sus_data, aita_data, epis_data)

    report_text = generate_unified_report(profiles, output_dir)
    (output_dir / "unified_report.md").write_text(report_text)
    report_path = output_dir / "REPORT.md"
    if report_path.exists():
        report_path.unlink()

    coverage_text = _coverage_markdown(profiles)
    (output_dir / "coverage.md").write_text(coverage_text)

    copied_artifacts = _copy_export_artifacts(manifest_dicts, output_dir, include_conversations)
    payload = {
        "schema_version": 1,
        "bundle_type": "unified_sycophancy_profile",
        "include_conversations": include_conversations,
        "runs": manifest_dicts,
        "copied_artifacts": copied_artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
