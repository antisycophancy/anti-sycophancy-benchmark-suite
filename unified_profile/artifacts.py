"""Read-only discovery of benchmark result artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from unified_profile.manifest import REPO_ROOT, RunManifest, RunQuality, artifact_ref, repo_relative_path
from unified_profile.models import canonicalize_model_id


def _as_paths(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    return [Path(p) for p in path_or_paths]


def _load_json(path: Path) -> object | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _kind(path: Path) -> str:
    name = path.name
    if name.endswith("_scores.json") or name in {"FINAL_RESULTS.json", "mt_elephant_results.json", "n20_results.json"}:
        return "score"
    if name.endswith("_side_a.json") or name.endswith("_side_b.json") or name.endswith("-conversations.json"):
        return "conversation"
    if name == "REPORT.md":
        return "report"
    if name.endswith("SUMMARY.md") or name == "SUMMARY.md":
        return "summary"
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "config"
    return "artifact"


def _quality(path: Path) -> RunQuality:
    rel = repo_relative_path(path).lower()
    if "smoke-" in rel:
        return "smoke"
    if "/tests/" in rel or "/fixtures/" in rel:
        return "test"
    if "paper-final" in rel or "paper_final" in rel:
        return "paper_final"
    if "paper-candidate" in rel or "paper_candidate" in rel:
        return "paper_candidate"
    if "mt_elephant_20260415_042010" in rel:
        return "research"
    return "research"


def _sorted_strings(values: Iterable[object]) -> list[str]:
    return sorted({str(v) for v in values if v is not None and str(v) != ""})


def _sorted_models(values: Iterable[object]) -> list[str]:
    return sorted({canonicalize_model_id(str(v)) for v in values if v is not None and str(v) != ""})


def _item_from_name(name: str) -> str | None:
    match = re.search(r"_item(\d+)", name)
    if match:
        return match.group(1)
    return None


def _score_id_from_name(name: str) -> str:
    return name.removesuffix(".json").removesuffix("_scores")


def _model_from_name(name: str) -> str | None:
    match = re.match(r"^(.+?)_item\d+", name)
    if match:
        return match.group(1)
    return None


def _artifact_refs(files: Iterable[Path]) -> list:
    return [artifact_ref(path, _kind(path)) for path in sorted(set(files))]


def _source_root(path: Path) -> str:
    return repo_relative_path(path)


def discover_sus_runs(path: Path | str | Iterable[Path | str]) -> list[RunManifest]:
    """Discover SUS conversation/result files as run manifests."""
    manifests: list[RunManifest] = []
    for root in _as_paths(path):
        files = [root] if root.is_file() else sorted(root.rglob("*-conversations.json")) if root.exists() else []
        for conv_path in files:
            data = _load_json(conv_path)
            if not isinstance(data, list):
                continue

            summary_path = conv_path.with_name(conv_path.name.replace("-conversations.json", ".json"))
            artifacts = [conv_path]
            if summary_path.exists():
                artifacts.append(summary_path)

            models = []
            source_keys = []
            item_ids = []
            judges = []
            n_scores = 0
            for idx, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                source_model = row.get("model") or row.get("model_id")
                source_keys.append(source_model)
                models.append(source_model)
                item_ids.append(row.get("scenario") or f"row-{idx}")
                if isinstance(row.get("score"), dict):
                    n_scores += 1
                post_analysis = row.get("post_analysis")
                if isinstance(post_analysis, dict):
                    panel = post_analysis.get("judge_panel")
                    if isinstance(panel, list):
                        judges.extend(panel)

            notes = []
            if n_scores == 0:
                notes.append("conversation_only")
            if n_scores < len(data):
                notes.append("missing_scores")

            manifests.append(
                RunManifest(
                    run_id=conv_path.stem.replace("-conversations", ""),
                    module="sus",
                    quality=_quality(conv_path),
                    source_root=_source_root(conv_path.parent),
                    models=_sorted_models(models),
                    source_model_keys=_sorted_strings(source_keys),
                    judge_model=", ".join(_sorted_strings(judges)) or None,
                    seeker_model=None,
                    item_ids=_sorted_strings(item_ids),
                    n_conversations=len(data),
                    n_scores=n_scores,
                    artifacts=_artifact_refs(artifacts),
                    notes=notes,
                )
            )
    return manifests


def _aita_run_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent]
    if not root.exists():
        return []
    if _aita_files(root):
        return [root]
    return sorted({path.parent for path in root.rglob("*_scores.json")} | {path.parent for path in root.rglob("FINAL_RESULTS.json")} | {path.parent for path in root.rglob("mt_elephant_results.json")})


def _aita_files(run_dir: Path) -> list[Path]:
    patterns = ["*_side_a.json", "*_side_b.json", "*_scores.json", "FINAL_RESULTS.json", "mt_elephant_results.json", "n20_results.json", "REPORT.md", "SUMMARY.md"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(run_dir.glob(pattern)))
    return sorted(set(files))


def _row_has_number(row: dict) -> bool:
    return any(isinstance(value, (int, float)) for value in row.values())


def _aita_file_has_usable_scores(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        return False
    scores = data.get("scores")
    if isinstance(scores, dict):
        return any(_row_has_number(row) for row in scores.values() if isinstance(row, dict))
    if all(isinstance(v, dict) for v in data.values()):
        return any(
            isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float))
            for metrics in data.values()
            for metric in metrics.values()
        )
    return False


def _preferred_aita_score_index(files: list[Path]) -> Path | None:
    by_name = {path.name: path for path in files}
    for name in ["mt_elephant_results.json", "FINAL_RESULTS.json", "n20_results.json"]:
        candidate = by_name.get(name)
        if candidate and _aita_file_has_usable_scores(candidate):
            return candidate
    return None


def discover_aita_runs(path: Path | str | Iterable[Path | str]) -> list[RunManifest]:
    """Discover AITA run directories as run manifests."""
    manifests: list[RunManifest] = []
    for root in _as_paths(path):
        for run_dir in _aita_run_dirs(root):
            files = _aita_files(run_dir)
            if not files:
                continue
            preferred_score_index = _preferred_aita_score_index(files)

            models = []
            source_keys = []
            item_ids = []
            judges = []
            seekers = []
            n_conversations = 0
            score_items = set()
            for file_path in files:
                data = _load_json(file_path)
                if file_path.name.endswith(("_side_a.json", "_side_b.json")):
                    n_conversations += 1
                    if isinstance(data, dict):
                        source = data.get("model") or data.get("model_id") or _model_from_name(file_path.name)
                        source_keys.append(source)
                        models.append(data.get("model_id") or source)
                        item_ids.append(data.get("item_idx") or _item_from_name(file_path.name))
                        seekers.append(data.get("seeker_model"))
                elif file_path.name.endswith("_scores.json"):
                    if preferred_score_index is None:
                        score_items.add(_score_id_from_name(file_path.name))
                        source = _model_from_name(file_path.name)
                        source_keys.append(source)
                        models.append(source)
                        item_ids.append(_item_from_name(file_path.name))
                elif file_path == preferred_score_index and isinstance(data, dict):
                    metadata = data.get("metadata")
                    if isinstance(metadata, dict):
                        judges.append(metadata.get("judge"))
                        seekers.append(metadata.get("seeker"))
                        item_ids.extend(metadata.get("items") or [])
                        source_keys.extend(metadata.get("models") or [])
                        models.extend(metadata.get("models") or [])
                    scores = data.get("scores")
                    if isinstance(scores, dict):
                        for key, row in scores.items():
                            if isinstance(row, dict) and any(v is not None for v in row.values()):
                                score_items.add(_score_id_from_name(key))
                                source = key.rsplit("_item", 1)[0]
                                source_keys.append(source)
                                models.append(source)
                                item_ids.append(_item_from_name(key))

            notes = []
            if n_conversations == 0:
                notes.append("score_only")
            if not score_items:
                notes.append("conversation_only")
            if n_conversations and not score_items:
                notes.append("missing_scores")
            if preferred_score_index and any(path.name.endswith("_scores.json") for path in files):
                notes.append(f"preferred_score_index={preferred_score_index.name}")
                notes.append("per_item_scores_attached")

            manifests.append(
                RunManifest(
                    run_id=run_dir.name,
                    module="aita",
                    quality=_quality(run_dir),
                    source_root=_source_root(run_dir),
                    models=_sorted_models(models),
                    source_model_keys=_sorted_strings(source_keys),
                    judge_model=", ".join(_sorted_strings(judges)) or None,
                    seeker_model=", ".join(_sorted_strings(seekers)) or None,
                    item_ids=_sorted_strings(item_ids),
                    n_conversations=n_conversations,
                    n_scores=len(score_items),
                    artifacts=_artifact_refs(files),
                    notes=notes,
                )
            )
    return manifests


def _epis_run_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent]
    if not root.exists():
        return []
    if any(root.glob("*_scores.json")) or any(root.glob("*_side_a.json")) or any(root.glob("*_side_b.json")):
        return [root]
    if root.name.startswith("smoke-") and any(root.rglob("*_scores.json")):
        return [root]
    dirs = {path.parent for path in root.rglob("*_scores.json")}
    dirs.update(path.parent for path in root.rglob("*_side_a.json"))
    dirs.update(path.parent for path in root.rglob("*_side_b.json"))
    run_roots = set()
    for directory in dirs:
        smoke_parent = next((parent for parent in [directory, *directory.parents] if parent.name.startswith("smoke-")), None)
        run_roots.add(smoke_parent or directory)
    return sorted(run_roots)


def _epis_files(run_dir: Path) -> list[Path]:
    patterns = ["*_side_a.json", "*_side_b.json", "*_scores.json", "REPORT.md"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(run_dir.rglob(pattern)))
    return sorted(set(files))


def discover_epis_runs(path: Path | str | Iterable[Path | str]) -> list[RunManifest]:
    """Discover epistemic run directories as run manifests."""
    manifests: list[RunManifest] = []
    for root in _as_paths(path):
        for run_dir in _epis_run_dirs(root):
            files = _epis_files(run_dir)
            if not files:
                continue

            models = []
            source_keys = []
            item_ids = []
            judges = []
            seekers = []
            n_conversations = 0
            score_files = 0
            for file_path in files:
                data = _load_json(file_path)
                if file_path.name.endswith(("_side_a.json", "_side_b.json")):
                    n_conversations += 1
                    if isinstance(data, dict):
                        source = data.get("model") or data.get("model_id") or _model_from_name(file_path.name)
                        source_keys.append(source)
                        models.append(data.get("model_id") or source)
                        item_ids.append(data.get("item_idx") or _item_from_name(file_path.name))
                        seekers.append(data.get("seeker_model"))
                elif file_path.name.endswith("_scores.json"):
                    score_files += 1
                    if isinstance(data, dict):
                        source = data.get("model") or data.get("model_id") or _model_from_name(file_path.name)
                        source_keys.append(source)
                        models.append(data.get("model_id") or source)
                        item_ids.append(data.get("item_idx") or _item_from_name(file_path.name))
                        judges.append(data.get("judge_model"))

            notes = []
            if n_conversations == 0:
                notes.append("score_only")
            if score_files == 0:
                notes.append("conversation_only")
            if n_conversations and score_files == 0:
                notes.append("missing_scores")

            manifests.append(
                RunManifest(
                    run_id=run_dir.name,
                    module="epistemic",
                    quality=_quality(run_dir),
                    source_root=_source_root(run_dir),
                    models=_sorted_models(models),
                    source_model_keys=_sorted_strings(source_keys),
                    judge_model=", ".join(_sorted_strings(judges)) or None,
                    seeker_model=", ".join(_sorted_strings(seekers)) or None,
                    item_ids=_sorted_strings(item_ids),
                    n_conversations=n_conversations,
                    n_scores=score_files,
                    artifacts=_artifact_refs(files),
                    notes=notes,
                )
            )
    return manifests


def discover_all_runs(paths: dict[str, list[Path]]) -> list[RunManifest]:
    """Discover manifests for all modules from explicit input paths."""
    manifests: list[RunManifest] = []
    manifests.extend(discover_sus_runs(paths.get("sus", [])))
    manifests.extend(discover_aita_runs(paths.get("aita", [])))
    manifests.extend(discover_epis_runs(paths.get("epistemic", [])))
    return sorted(manifests, key=lambda m: (m.module, m.source_root, m.run_id))
