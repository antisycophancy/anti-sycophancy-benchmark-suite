"""Per-judge score breakdowns for judge self-preference visibility.

The model under test is already blinded in everything judges read
(``_blind_text`` / ``assert_blind_model_payload``), so the residual risk is
*latent* self-preference: a judge favoring its own family's style even
unlabeled (Panickssery et al. 2024). Score files keep every individual
judge's scores alongside the panel aggregate; this tool surfaces them as
per-judge means and paired deltas per model so any family tilt is visible
instead of hidden inside panel means.

Usage:
    python -m suite_tools.judge_breakdown <results_dir> [...] \
        [--output-dir DIR]

Reads any module's ``*_scores.json`` files (AITA, Epistemic, SUS — judge
entries may carry ``judge_model`` or ``judge``).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from suite_tools.panel_compare import _model_aliases
from suite_tools.suite_registry import module_key_for_record

# Judge-entry keys that are never score dimensions.
_METADATA_KEYS = {
    "item_idx",
    "judge_attempt",
    "num_judges",
    "expected_num_judges",
}
_METADATA_SUFFIXES = ("_idx", "_count", "_seconds", "_tokens", "_ms")
# Turn positions (capitulation_turn, deterministic_verdict_turn_a, ...) are
# locations in the transcript, not scores.
_METADATA_SUBSTRINGS = ("turn",)

_FAMILY_MARKERS = (
    ("anthropic", ("anthropic", "claude")),
    ("openai", ("openai", "gpt")),
    ("google", ("google", "gemini")),
    ("xai", ("x-ai", "grok")),
    ("meta", ("meta", "llama")),
    ("mistral", ("mistral",)),
    ("deepseek", ("deepseek",)),
    ("qwen", ("qwen", "alibaba")),
)


def model_family(model_id: str | None) -> str:
    """Return a coarse vendor family for a model id or slug."""
    if not model_id:
        return "unknown"
    text = str(model_id).strip().lower()
    if not text:
        return "unknown"
    for family, markers in _FAMILY_MARKERS:
        if any(marker in text for marker in markers):
            return family
    if "/" in text:
        return text.split("/", 1)[0]
    return "unknown"


def _judge_id(entry: dict[str, Any]) -> str | None:
    for key in ("judge_model", "judge"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _dimension_values(entry: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in entry.items():
        if key in _METADATA_KEYS or key.endswith(_METADATA_SUFFIXES):
            continue
        if any(marker in key for marker in _METADATA_SUBSTRINGS):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values[key] = float(value)
    return values


def _same_family(judge_id: str, target_id: str) -> bool:
    judge_family = model_family(judge_id)
    target_family = model_family(target_id)
    if judge_family != "unknown" and judge_family == target_family:
        return True
    return bool(_model_aliases(judge_id) & _model_aliases(target_id))


def _iter_score_files(inputs: list[Path]) -> Iterator[Path]:
    for root in inputs:
        root = Path(root)
        if root.is_file():
            yield root
            continue
        yield from sorted(root.rglob("*_scores.json"))


def build_judge_breakdown(inputs: list[Path | str]) -> dict[str, Any]:
    """Aggregate per-judge scores from panel score files.

    Returns per model: per judge: per dimension mean/n plus a paired
    ``delta_vs_other_judges`` (mean over items of this judge's score minus
    the same item's mean across the other judges).
    """
    paths = [Path(p) for p in inputs]
    # (module, model, judge, dim) -> list of values / paired deltas.
    # Grouped per module: same-named dimensions (e.g. "consistency") have
    # different semantics across modules and must never be pooled.
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    deltas: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    scanned = 0

    for score_file in _iter_score_files(paths):
        try:
            data = json.loads(score_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        judge_entries = data.get("judge_scores")
        if not isinstance(judge_entries, list) or not judge_entries:
            # Single-judge score files (e.g. aita's single-judge path) carry
            # the judge's scores at the top level with no judge_scores list.
            # A top-level judge_model joining several ids ("a, b") is a panel
            # aggregate, not attributable to one judge — skip those.
            judge_model = data.get("judge_model")
            if not judge_model or "," in str(judge_model):
                continue
            judge_entries = [data]
        module = module_key_for_record(data, source=score_file)
        model_id = str(
            data.get("model_id") or data.get("model") or data.get("filename_model_key") or "unknown"
        )
        labels.setdefault(model_id, str(data.get("label") or model_id))
        scanned += 1

        per_judge = {}
        for entry in judge_entries:
            if not isinstance(entry, dict):
                continue
            judge = _judge_id(entry)
            if not judge:
                continue
            per_judge[judge] = _dimension_values(entry)

        dimensions = {dim for dims in per_judge.values() for dim in dims}
        for dim in dimensions:
            scored = {judge: dims[dim] for judge, dims in per_judge.items() if dim in dims}
            for judge, value in scored.items():
                values[(module, model_id, judge, dim)].append(value)
                others = [v for other, v in scored.items() if other != judge]
                if others:
                    deltas[(module, model_id, judge, dim)].append(value - statistics.mean(others))

    modules: dict[str, Any] = {}
    for (module, model_id, judge, dim), dim_values in sorted(values.items()):
        models = modules.setdefault(module, {"models": {}})["models"]
        model = models.setdefault(
            model_id,
            {"label": labels.get(model_id, model_id), "family": model_family(model_id), "judges": {}},
        )
        judge_data = model["judges"].setdefault(
            judge,
            {
                "family": model_family(judge),
                "same_family": _same_family(judge, model_id),
                "dimensions": {},
            },
        )
        paired = deltas.get((module, model_id, judge, dim))
        judge_data["dimensions"][dim] = {
            "mean": round(statistics.mean(dim_values), 4),
            "n": len(dim_values),
            "delta_vs_other_judges": round(statistics.mean(paired), 4) if paired else None,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in paths],
        "score_files_scanned": scanned,
        "modules": modules,
    }


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.2f}"


def render_markdown(breakdown: dict[str, Any]) -> str:
    lines = [
        "# Per-Judge Score Breakdown",
        "",
        f"**Generated:** {breakdown['generated_at']}  ",
        f"**Score files:** {breakdown.get('score_files_scanned', 'n/a')}",
        "",
        "Per-judge means and paired deltas (judge minus the mean of the other",
        "judges on the same items). Model identity is blinded in judge inputs;",
        "this table makes any residual same-family tilt visible. A large",
        "positive delta on a same-family row is the self-preference signature.",
        "",
    ]

    flagged: list[str] = []
    for module, module_data in breakdown["modules"].items():
        lines.append(f"## {module}")
        lines.append("")
        for model_id, model in module_data["models"].items():
            lines.append(f"### {model['label']} (`{model_id}`)")
            lines.append("")
            lines.append("| Judge | Family | Dimension | Mean | n | Δ vs other judges |")
            lines.append("|-------|--------|-----------|------|---|-------------------|")
            for judge, judge_data in model["judges"].items():
                family = judge_data["family"]
                if judge_data["same_family"]:
                    family = f"{family} (same-family)"
                for dim, stats in judge_data["dimensions"].items():
                    delta = stats["delta_vs_other_judges"]
                    lines.append(
                        f"| `{judge}` | {family} | `{dim}` | {stats['mean']} | "
                        f"{stats['n']} | {_format_delta(delta)} |"
                    )
                    if judge_data["same_family"] and delta is not None and abs(delta) >= 0.25:
                        flagged.append(
                            f"- {module}: `{model_id}` / `{judge}` / `{dim}`: {_format_delta(delta)}"
                        )
            lines.append("")

    if flagged:
        lines.extend(["## Same-family deltas ≥ 0.25 (review)", "", *flagged, ""])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-judge score breakdowns from panel score files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Results directories (or score files) containing *_scores.json with judge_scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write judge_breakdown.json and JUDGE_BREAKDOWN.md here.",
    )
    args = parser.parse_args(argv)

    breakdown = build_judge_breakdown(args.inputs)
    markdown = render_markdown(breakdown)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "judge_breakdown.json").write_text(
            json.dumps(breakdown, indent=2) + "\n"
        )
        (args.output_dir / "JUDGE_BREAKDOWN.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
