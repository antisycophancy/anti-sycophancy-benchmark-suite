"""Output formatting: JSON, terminal table, and HTML dashboard."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sus_bench import __version__
from suite_tools.model_config import MODEL_CONDITION_METADATA_FIELDS

console = Console()
SYNTHETIC_ASSISTANT_PREFIX = re.compile(r"^\[(?:TIMEOUT|ERROR|Provider refusal)", re.IGNORECASE)


def _assert_no_synthetic_scored_assistant_content(results: list[dict]) -> None:
    for result in results:
        score_state = result.get("score_state")
        if score_state is not None and score_state != "scored":
            continue
        if score_state is None and result.get("score") is None:
            continue
        for message in result.get("conversation") or []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if SYNTHETIC_ASSISTANT_PREFIX.match(str(message.get("content") or "")):
                raise ValueError("scored transcript contains synthetic assistant content")


def write_json(
    results: list[dict],
    aggregated: list[dict],
    path: Path,
    *,
    run_id: str | None = None,
    cost: dict | None = None,
) -> None:
    """Write benchmark results to JSON.

    Saves two files:
      - {path}: Summary with scores and metadata (no conversation text)
      - {path.stem}-conversations.json: Full conversation threads

    Args:
        results: Raw per-run results.
        aggregated: Aggregated results from stats.aggregate_runs.
        path: Output path for the summary JSON.
        run_id: Optional run identifier.
        cost: Optional cost summary from CostTracker.
    """
    _assert_no_synthetic_scored_assistant_content(results)

    # Summary (without conversation text or individual run details)
    summary_agg = []
    for a in aggregated:
        entry = {k: v for k, v in a.items() if k != "individual_results"}
        summary_agg.append(entry)

    summary = {
        "run_id": run_id or f"sus-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "aggregated": summary_agg,
    }
    if cost:
        summary["cost"] = cost

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Conversations (separate file, can be large)
    conversations = []
    for r in results:
        entry = {
            "model": r["model"],
            "label": r.get("label"),
            "scenario": r.get("scenario", "unknown"),
            "scenario_name": r.get("scenario_name"),
            "run_number": r.get("run_number", 1),
            "score": r.get("score"),
            "post_analysis": r.get("post_analysis"),
            "conversation": r.get("conversation", []),
            "turn_outcomes": r.get("turn_outcomes", []),
            "phases": r.get("phases", {}),
        }
        for key in MODEL_CONDITION_METADATA_FIELDS:
            if key in r:
                entry[key] = r[key]
        for key in (
            "escalation_mode",
            "score_state",
            "exclusion_reason",
            "provider_refusal",
            "score_inline",
            "capitulated",
            "capitulation_phase_key",
            "capitulation_strategy_id",
            "capitulation_strategy",
            "phase_failure_mode",
            "failure_mode",
            "failure_class",
            "failure_mode_source",
            "classification_conflict",
            "classification_conflict_reason",
            "rescore_metadata",
            "identity_normalization",
        ):
            if key in r:
                entry[key] = r[key]
        conversations.append(entry)

    conv_path = path.with_name(f"{path.stem}-conversations.json")
    with open(conv_path, "w") as f:
        json.dump(conversations, f, indent=2, default=str)


def print_table(aggregated: list[dict]) -> None:
    """Print a rich terminal table of aggregated results.

    Args:
        aggregated: List of aggregated result dicts.
    """
    table = Table(title="sus-bench Results", show_lines=True)

    table.add_column("Model", style="bold")
    table.add_column("Scenario", style="dim")
    table.add_column("N", justify="right")
    table.add_column("SUS Resp", justify="right")
    table.add_column("Cap Rate", justify="right")
    table.add_column("IRQ", justify="right")
    table.add_column("PR", justify="right")
    table.add_column("ER", justify="right")
    table.add_column("CA", justify="right")
    table.add_column("SUS 95% CI", justify="center")
    table.add_column("Cap 95% CI", justify="center")
    table.add_column("Cap Phase")

    for a in aggregated:
        sus = a["sus_mean"]

        # Color-code SUS (lower = better, so invert colors)
        if sus <= 30:
            sus_str = f"[green]{sus:.1f}[/green]"
        elif sus <= 60:
            sus_str = f"[yellow]{sus:.1f}[/yellow]"
        else:
            sus_str = f"[red]{sus:.1f}[/red]"

        # CI range
        if a["runs"] > 1:
            ci_str = f"[{a['sus_ci_lower']:.1f}, {a['sus_ci_upper']:.1f}]"
        else:
            ci_str = "N/A"
        cap_rate = a.get("cap_rate", a.get("capitulation_rate", 0.0))
        cap_ci_str = (
            f"[{a.get('cap_rate_wilson_95_ci_low', 0.0):.2f}, "
            f"{a.get('cap_rate_wilson_95_ci_high', 0.0):.2f}]"
        )
        excluded_count = int(a.get("excluded_provider_refusal_count", 0) or 0)
        n_str = str(a["runs"])
        if excluded_count:
            n_str = f"{n_str} (+{excluded_count} excluded)"

        table.add_row(
            a["label"],
            a.get("scenario_name", a.get("scenario", "")),
            n_str,
            sus_str,
            f"{cap_rate:.1%}",
            f"{a.get('irq_mean', 0.0):.1f}",
            f"{a.get('pr_mean', 0.0):.1f}",
            f"{a.get('er_mean', 0.0):.1f}",
            f"{a.get('ca_mean', 0.0):.1f}",
            ci_str,
            cap_ci_str,
            a.get("capitulation_phase", ""),
        )

    console.print()
    console.print(table)
    console.print()


def print_segmented_table(results: list[dict]) -> None:
    """Print capitulation mode breakdown per model.

    Classifies each result on-the-fly from its phases dict (retroactive
    classification), groups by model label, and displays counts + percentages
    for each failure mode.

    Args:
        results: List of individual result dicts, each with at least
                 'phases' and 'model'/'label' keys.
    """
    from collections import Counter

    from sus_bench.classifier import FailureClass, FailureMode, classify_result, normalize_failure_mode
    from sus_bench.scoring_contract import is_score_excluded_result

    # Group results by model label
    by_model: dict[str, list[dict]] = {}
    excluded_by_model: Counter = Counter()
    for r in results:
        label = r.get("label", r.get("model", "unknown"))
        by_model.setdefault(label, [])
        if is_score_excluded_result(r):
            excluded_by_model[label] += 1
            continue
        by_model[label].append(classify_result(r))

    if not by_model:
        console.print("[dim]No results to segment.[/dim]")
        return

    # Column order is safe-first for readability.
    mode_columns = [
        FailureMode.NO_CAP,
        FailureMode.REFRAME,
        FailureMode.EMOTIONAL,
        FailureMode.INJECTION,
        FailureMode.AUTHORITY,
        FailureMode.PASSPHRASE,
        FailureMode.IMMEDIATE,
    ]

    table = Table(title="Capitulation Mode Breakdown", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Conflicts", justify="right", style="magenta")
    table.add_column("No Cap", justify="right", style="green")
    table.add_column("Reframe", justify="right", style="yellow")
    table.add_column("Emotional", justify="right", style="yellow")
    table.add_column("Injection", justify="right")
    table.add_column("Authority", justify="right")
    table.add_column("Passphrase", justify="right", style="red")
    table.add_column("Immediate", justify="right", style="red")

    for label, classifications in sorted(by_model.items()):
        n = len(classifications)
        modes = [normalize_failure_mode(c["failure_mode"]) for c in classifications]
        counts = Counter(modes)
        conflict_count = sum(1 for c in classifications if c["classification_conflict"])
        conflict_pct = round(100 * conflict_count / n) if n > 0 else 0
        excluded_count = excluded_by_model.get(label, 0)
        n_str = str(n)
        if excluded_count:
            n_str = f"{n_str} (+{excluded_count} excluded)"

        cells = [
            label,
            n_str,
            f"{conflict_count} ({conflict_pct}%)" if conflict_count else "-",
        ]
        for mode in mode_columns:
            count = counts.get(mode, 0)
            pct = round(100 * count / n) if n > 0 else 0
            cells.append(f"{count} ({pct}%)" if count > 0 else "-")

        table.add_row(*cells)

    console.print()
    console.print(table)

    # Class summary
    class_table = Table(title="Failure Class Summary", show_lines=True)
    class_table.add_column("Model", style="bold")
    class_table.add_column("N", justify="right")
    class_table.add_column("Conflicts", justify="right", style="magenta")
    class_table.add_column("Structurally Safe", justify="right", style="green")
    class_table.add_column("Deference Collapse", justify="right", style="yellow")
    class_table.add_column("Empathic Override", justify="right", style="yellow")
    class_table.add_column("Self-Coaching", justify="right", style="red")
    class_table.add_column("No Safety", justify="right", style="red")

    class_order = [
        FailureClass.STRUCTURALLY_SAFE,
        FailureClass.DEFERENCE_COLLAPSE,
        FailureClass.EMPATHIC_OVERRIDE,
        FailureClass.SELF_COACHING,
        FailureClass.NO_SAFETY,
    ]

    for label, classifications in sorted(by_model.items()):
        n = len(classifications)
        class_counts: Counter = Counter()
        conflict_count = sum(1 for c in classifications if c["classification_conflict"])
        for classification in classifications:
            class_counts[FailureClass(classification["failure_class"])] += 1

        conflict_pct = round(100 * conflict_count / n) if n > 0 else 0
        excluded_count = excluded_by_model.get(label, 0)
        n_str = str(n)
        if excluded_count:
            n_str = f"{n_str} (+{excluded_count} excluded)"
        cells = [
            label,
            n_str,
            f"{conflict_count} ({conflict_pct}%)" if conflict_count else "-",
        ]
        for fc in class_order:
            count = class_counts.get(fc, 0)
            pct = round(100 * count / n) if n > 0 else 0
            cells.append(f"{count} ({pct}%)" if count > 0 else "-")

        class_table.add_row(*cells)

    console.print()
    console.print(class_table)
    console.print()


def write_html(aggregated: list[dict], path: Path) -> None:
    """Render an HTML dashboard from aggregated results.

    Args:
        aggregated: List of aggregated result dicts.
        path: Output path for the HTML file.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "xml"),
            default_for_string=True,
        ),
    )
    template = env.get_template("dashboard.html")

    html = template.render(
        results=aggregated,
        version=__version__,
        timestamp=datetime.now().isoformat(),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
