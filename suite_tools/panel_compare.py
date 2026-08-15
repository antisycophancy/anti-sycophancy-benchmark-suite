"""Compare AITA and Epistemic score files across judge-panel directories."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PANEL_DIMENSIONS = {
    "aita": (
        "outcome_a",
        "resistance_a",
        "therapeutic_a",
        "outcome_b",
        "resistance_b",
        "therapeutic_b",
        "consistency",
    ),
    "epis": (
        "persistence",
        "integrity",
        "consistency",
        "amplification",
    ),
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _median(values: list[int | float]) -> int | float | None:
    if not values:
        return None
    value = statistics.median(values)
    if float(value).is_integer():
        return int(value)
    return value


def _majority(values: list[int | float]) -> int | float | None:
    if not values:
        return None
    [(value, count), *_] = Counter(values).most_common()
    return value if count > len(values) / 2 else None


def _display_value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _model_aliases(value: str | None) -> set[str]:
    """Return conservative aliases for matching target models to judges."""
    if not value:
        return set()
    text = str(value).strip().lower()
    if not text:
        return set()
    aliases = {text, text.replace(".", "-")}
    if "/" in text:
        tail = text.rsplit("/", 1)[-1]
        aliases.add(tail)
        aliases.add(tail.replace(".", "-"))
    for alias in list(aliases):
        if alias.endswith("-preview"):
            aliases.add(alias.removesuffix("-preview"))
    for alias in list(aliases):
        for prefix in (
            "therapeutic-harness/",
            "th-",
            "harness-",
            "alpha-",
        ):
            if alias.startswith(prefix):
                aliases.add(alias.removeprefix(prefix))
        for vendor_prefix in ("claude-", "anthropic-", "google-", "openai-"):
            if alias.startswith(vendor_prefix):
                aliases.add(alias.removeprefix(vendor_prefix))
    return aliases


def _is_self_judge(judge_model: str | None, target_ids: set[str]) -> bool:
    judge_aliases = _model_aliases(judge_model)
    target_aliases = set()
    for target in target_ids:
        target_aliases.update(_model_aliases(target))
    return bool(judge_aliases & target_aliases)


def _case_id(score_file: Path) -> str:
    return score_file.stem.removesuffix("_scores")


def _score_files(module_root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if not module_root.exists():
        return files
    for judge_dir in sorted(path for path in module_root.iterdir() if path.is_dir()):
        for score_file in sorted(judge_dir.glob("*_scores.json")):
            files.append((judge_dir.name, score_file))
    return files


def compare_panel(root: Path) -> dict[str, Any]:
    """Load per-judge score directories and return comparison data."""
    modules: dict[str, Any] = {}
    for module, dimensions in PANEL_DIMENSIONS.items():
        rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        judge_models: dict[str, str] = {}
        case_targets: dict[str, set[str]] = defaultdict(set)

        for judge_key, score_file in _score_files(root / module):
            data = _load_json(score_file)
            if data is None:
                continue
            case_id = _case_id(score_file)
            judge_models[judge_key] = str(data.get("judge_model") or judge_key)
            for target_key in ("model_id", "model", "filename_model_key"):
                target = data.get(target_key)
                if target:
                    case_targets[case_id].add(str(target))
            for dimension in dimensions:
                if dimension in data:
                    rows[case_id][dimension][judge_key] = data.get(dimension)

        cases = []
        totals = {
            "dimensions": 0,
            "unanimous": 0,
            "disagreements": 0,
            "missing": 0,
        }

        for case_id in sorted(rows):
            dimensions_out = {}
            target_ids = case_targets.get(case_id, set())
            self_judge_keys = {
                key for key, judge_model in judge_models.items()
                if _is_self_judge(judge_model, target_ids)
            }
            for dimension in dimensions:
                by_judge = rows[case_id].get(dimension, {})
                if not by_judge:
                    continue
                valid_by_judge = {
                    judge: value
                    for judge, value in by_judge.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                values = list(valid_by_judge.values())
                totals["dimensions"] += 1
                missing = sorted(set(judge_models) - set(valid_by_judge))
                invalid = sorted(set(by_judge) - set(valid_by_judge))
                if missing:
                    totals["missing"] += 1
                unique_values = sorted(set(values))
                unanimous = len(unique_values) == 1 and len(valid_by_judge) == len(judge_models)
                self_excluded_values = [
                    value for judge, value in valid_by_judge.items()
                    if judge not in self_judge_keys
                ]
                self_excluded_median = (
                    _median(self_excluded_values) if self_judge_keys else None
                )
                self_excluded_majority = (
                    _majority(self_excluded_values) if self_judge_keys else None
                )
                self_exclusion_changes = (
                    bool(self_judge_keys)
                    and bool(self_excluded_values)
                    and (
                        self_excluded_median != _median(values)
                        or self_excluded_majority != _majority(values)
                    )
                )
                if unanimous:
                    totals["unanimous"] += 1
                else:
                    totals["disagreements"] += 1

                dimensions_out[dimension] = {
                    "by_judge": {judge_models.get(key, key): by_judge[key] for key in sorted(by_judge)},
                    "valid_by_judge": {
                        judge_models.get(key, key): valid_by_judge[key]
                        for key in sorted(valid_by_judge)
                    },
                    "median": _median(values),
                    "majority": _majority(values),
                    "self_excluded_median": self_excluded_median,
                    "self_excluded_majority": self_excluded_majority,
                    "self_excluded_n": len(self_excluded_values) if self_judge_keys else None,
                    "self_judges": [
                        judge_models.get(key, key) for key in sorted(self_judge_keys)
                        if key in valid_by_judge
                    ],
                    "self_exclusion_changes": self_exclusion_changes,
                    "unique_values": unique_values,
                    "unanimous": unanimous,
                    "missing_judges": [judge_models.get(key, key) for key in missing],
                    "invalid_judges": [judge_models.get(key, key) for key in invalid],
                    "needs_review": not unanimous or self_exclusion_changes,
                }
            cases.append({
                "case_id": case_id,
                "target_model_ids": sorted(target_ids),
                "dimensions": dimensions_out,
            })

        modules[module] = {
            "judge_models": [judge_models[key] for key in sorted(judge_models)],
            "totals": totals,
            "cases": cases,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "modules": modules,
        "adjudication_policy": (
            "For archived separate-judge comparison directories, inspect "
            "median/majority and self-excluded medians as calibration signals. "
            "Current release scoring is performed by the benchmark scorers, "
            "which preserve per-judge scores and panel-majority release fields."
        ),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Judge Panel Comparison",
        "",
        f"**Generated:** {comparison['generated_at']}  ",
        f"**Root:** `{comparison['root']}`",
        "",
        comparison["adjudication_policy"],
        "",
        "## Summary",
        "",
        "| Module | Judges | Dimensions | Unanimous | Disagreement | Missing |",
        "|--------|--------|------------|-----------|--------------|---------|",
    ]

    for module, data in comparison["modules"].items():
        totals = data["totals"]
        lines.append(
            f"| {module} | {len(data['judge_models'])} | {totals['dimensions']} | "
            f"{totals['unanimous']} | {totals['disagreements']} | {totals['missing']} |"
        )

    for module, data in comparison["modules"].items():
        lines.extend(["", f"## {module.upper()}", ""])
        if data["judge_models"]:
            lines.append("Judges: " + ", ".join(f"`{judge}`" for judge in data["judge_models"]))
            lines.append("")
        lines.append("| Case | Dimension | Median | Majority | Self-Excl Median | Judge Scores | Review |")
        lines.append("|------|-----------|--------|----------|------------------|--------------|--------|")
        for case in data["cases"]:
            for dimension, dim_data in case["dimensions"].items():
                scores = ", ".join(f"{judge}={value}" for judge, value in dim_data["by_judge"].items())
                self_excl = dim_data.get("self_excluded_median")
                review = "yes" if dim_data["needs_review"] else ""
                lines.append(
                    f"| `{case['case_id']}` | `{dimension}` | {_display_value(dim_data['median'])} | "
                    f"{_display_value(dim_data['majority'])} | {_display_value(self_excl)} | {scores} | {review} |"
                )

    return "\n".join(lines) + "\n"


def write_panel_comparison(root: Path, output_dir: Path) -> dict[str, Any]:
    comparison = compare_panel(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panel_comparison.json").write_text(json.dumps(comparison, indent=2))
    (output_dir / "PANEL_REPORT.md").write_text(render_markdown(comparison))
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare score files across AITA/Epistemic judge panel directories.")
    parser.add_argument("root", type=Path, help="Root with aita/<judge>/ and epis/<judge>/ score directories.")
    parser.add_argument("--output-dir", type=Path, help="Directory for PANEL_REPORT.md and panel_comparison.json.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.root
    comparison = write_panel_comparison(args.root, output_dir)
    print(render_markdown(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
