"""Generated HTML report for a benchmark bundle.

``render_bundle_report(bundle_dir)`` reads BUNDLE_MANIFEST.json and the data/
JSONL files to produce a self-contained HTML artifact (no external references)
with:

1. **Leaderboard** per module: conditions ranked by primary_outcome + release-
   facing dimensions, direction-aware (cap=worse → ascending cap rate).
   Columns: n_expected, n_scored, n_unscored (pending count — never enters any
   denominator), per-dimension mean with CI (wilson / bootstrap / n-only), and
   declination rate (terminal_model_signal / n_expected) beside each behavioral
   column.  Cells with n < 5 carry the ``greyed low-n`` CSS class.

2. **Effort curves**: inline SVG polyline per canonical_model × release
   dimension across effort tiers, with declination-rate overlay.

3. **Refusal/block breakdown**: per condition × module table from blocks.jsonl,
   evidence pointers listed as text (not links when transcripts are absent).

4. **Run health snapshot**: members table from the manifest.

5. **Footer**: exclusion_policy, projection_version, tool_version,
   contains_transcripts, generation timestamp.

``leaderboard_order(bundle_dir, *, module) -> list[str]`` returns the ranked
condition ids (canonical_model values) for a module, testable without HTML
parsing.

CLI: ``python -m suite_tools.bundle_report BUNDLE_DIR`` writes
``report/index.html`` inside the bundle.
"""

from __future__ import annotations

import html as _html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from suite_tools.statistics import bootstrap_ci, wilson_interval

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_N_THRESHOLD = 5

# Canonical ordering for effort tier x-axis; unknown values appended sorted.
_EFFORT_ORDER = ["low", "standard", "medium", "high", "xhigh", "extended", "max"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank/invalid lines."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            pass
    return rows


def _condition_id(row: dict[str, Any]) -> str:
    """Return the canonical_model from a score row's condition field."""
    cond = row.get("condition")
    if isinstance(cond, dict):
        return str(cond.get("canonical_model") or "")
    return str(cond or "")


def _effort_value(row: dict[str, Any]) -> str:
    """Return the effort tier from a score row's condition field."""
    cond = row.get("condition")
    if isinstance(cond, dict):
        return str(cond.get("effort") or "")
    return ""


def _parse_module(unit_id: str) -> str:
    """Return the module prefix of a unit_id (e.g. 'aita' from 'aita:m:i0:a')."""
    return str(unit_id or "").split(":")[0]


def _parse_model_key(unit_id: str) -> str:
    """Return the model_key segment of a unit_id."""
    parts = str(unit_id or "").split(":")
    return parts[1] if len(parts) > 1 else ""


def _numeric(v: Any) -> float | None:
    """Coerce a value to float, treating bool as 1.0/0.0."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _direction_adjusted(mean: float, direction: str) -> float:
    """Return a score where higher always means better, for ranking."""
    d = str(direction or "").lower()
    # "cap=worse" and "higher=worse" both mean lower measured value = better
    if "worse" in d:
        return -mean
    return mean


def _effort_sort_key(effort: str) -> int:
    e = effort.lower()
    try:
        return _EFFORT_ORDER.index(e)
    except ValueError:
        return len(_EFFORT_ORDER)


def _h(text: Any) -> str:
    """HTML-escape a value for safe inline use."""
    return _html.escape(str(text or ""), quote=True)


# ---------------------------------------------------------------------------
# Public: leaderboard_order
# ---------------------------------------------------------------------------


def leaderboard_order(bundle_dir: Path | str, *, module: str) -> list[str]:
    """Return direction-aware ranked condition ids (canonical_model) for a module.

    Conditions are ranked by the mean of primary_outcome + release_facing
    dimensions.  For directions containing "worse" (cap=worse, higher=worse) a
    lower measured value ranks higher (sort ascending by raw value = descending
    by adjusted score).  Returned list is best-first.
    """
    bundle_dir = Path(bundle_dir)
    scores = _load_jsonl(bundle_dir / "data" / "scores.jsonl")

    # Filter to primary_outcome, release_facing, scored rows for this module
    primary_rows = [
        r for r in scores
        if r.get("module") == module
        and r.get("role") == "primary_outcome"
        and r.get("release_facing") is True
        and r.get("outcome_class") == "scored"
    ]

    if not primary_rows:
        return []

    # Group values by (condition_id, dimension); collect direction per dimension
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    directions: dict[str, str] = {}

    for row in primary_rows:
        cid = _condition_id(row)
        dim = str(row.get("dimension") or "")
        v = _numeric(row.get("value"))
        if v is None or not cid:
            continue
        groups[(cid, dim)].append(v)
        if row.get("direction"):
            directions[dim] = str(row["direction"])

    # Compute direction-adjusted composite score per condition
    condition_scores: dict[str, float] = defaultdict(float)
    for (cid, dim), vals in groups.items():
        if not vals:
            continue
        mean = statistics.mean(vals)
        direction = directions.get(dim, "higher=better")
        condition_scores[cid] += _direction_adjusted(mean, direction)

    # Best-first (highest adjusted score first)
    return sorted(condition_scores, key=lambda cid: condition_scores[cid], reverse=True)


# ---------------------------------------------------------------------------
# Internal: leaderboard data assembly
# ---------------------------------------------------------------------------


def _build_outcome_counts(
    outcomes: list[dict[str, Any]],
    unit_condition: dict[str, str],
    model_key_condition: dict[tuple[str, str], str],
) -> dict[tuple[str, str], dict[str, int]]:
    """Return {(module, condition_id): {outcome_class: count}}.

    For units not present in scores.jsonl (terminal / unscored), the condition
    is inferred from (module, model_key) → canonical_model via scored rows.
    """
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in outcomes:
        uid = str(outcome.get("unit_id") or "")
        oc = str(outcome.get("outcome_class") or "unknown")
        module = _parse_module(uid)
        mk = _parse_model_key(uid)
        cid = unit_condition.get(uid) or model_key_condition.get((module, mk), "")
        if module and cid:
            counts[(module, cid)][oc] += 1
    return counts  # type: ignore[return-value]


def _build_dim_groups(
    scores: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], list[float]],   # (module, cid, dim) → values
    dict[tuple[str, str, str], dict[str, Any]], # (module, cid, dim) → meta
]:
    """Build per-dimension value groups and metadata from scores.jsonl."""
    dim_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    dim_meta: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in scores:
        if row.get("outcome_class") != "scored":
            continue
        module = str(row.get("module") or "")
        cid = _condition_id(row)
        dim = str(row.get("dimension") or "")
        v = _numeric(row.get("value"))
        if not module or not dim or v is None:
            continue
        key = (module, cid, dim)
        dim_values[key].append(v)
        if key not in dim_meta:
            dim_meta[key] = {
                "statistic_kind": row.get("statistic_kind"),
                "direction": row.get("direction", ""),
                "role": row.get("role", ""),
                "release_facing": row.get("release_facing", False),
            }
    return dim_values, dim_meta


# ---------------------------------------------------------------------------
# Internal: cell rendering
# ---------------------------------------------------------------------------


def _dim_cell(
    values: list[float],
    statistic_kind: str | None,
    *,
    low_n: bool = False,
) -> tuple[str, str]:
    """Return (cell_text, css_classes) for a dimension value cell.

    For wilson: show mean% with 95% CI bounds.
    For bootstrap: show mean with 95% CI bounds.
    For None: show 'n only (no CI)'.
    """
    n = len(values)
    classes = "greyed low-n" if (n < LOW_N_THRESHOLD or low_n) else ""

    if n == 0:
        return "—", classes

    mean = statistics.mean(values)

    if statistic_kind == "wilson":
        successes = int(round(sum(values)))  # values are 0.0/1.0
        lo, hi = wilson_interval(successes, n)
        text = f"{mean:.1%} [{lo:.1%}–{hi:.1%} 95% CI] (n={n})"
    elif statistic_kind == "bootstrap":
        point, lo, hi = bootstrap_ci(values)
        text = f"{point:.2f} [{lo:.2f}–{hi:.2f} 95% CI] (n={n})"
    else:
        # statistic_kind is None → n only, no CI
        text = f"{mean:.3f} (n only, no CI) (n={n})"

    return text, classes


# ---------------------------------------------------------------------------
# Internal: SVG effort curve
# ---------------------------------------------------------------------------


_DECL_COLOR = "#f59e0b"  # amber — visually distinct from the score-line palette


def _effort_svg(
    module: str,
    dim: str,
    model_effort_vals: dict[str, dict[str, list[float]]],
    decl_map: dict[tuple[str, str, str], tuple[int, int]],
    effort_tiers: list[str],
) -> str:
    """Render an inline SVG polyline chart for one (module, dimension) pair.

    Draws one solid polyline per condition (canonical_model) for the dimension
    score, then overlays a dashed amber polyline for the declination rate
    (terminal_model_signal / n_expected) per condition × effort tier from
    *decl_map*.  A legend is included whenever at least one declination point
    is present.
    """
    w, h = 420, 200
    ml, mt, mr, mb = 45, 20, 20, 40
    iw = w - ml - mr
    ih = h - mt - mb

    if not effort_tiers:
        return ""

    # Collect all mean values for y-scale
    all_vals: list[float] = []
    for m_dict in model_effort_vals.values():
        for vals in m_dict.values():
            if vals:
                all_vals.append(statistics.mean(vals))
    if not all_vals:
        return ""

    y_min = min(0.0, min(all_vals))
    y_max = max(y_min + 1.0, max(all_vals))
    y_range = y_max - y_min or 1.0

    n_tiers = len(effort_tiers)

    def px(i: int) -> float:
        if n_tiers == 1:
            return ml + iw / 2
        return ml + (i / (n_tiers - 1)) * iw

    def py(val: float) -> float:
        return mt + ih - ((val - y_min) / y_range) * ih

    _COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#db2777"]

    lines = [
        f'<svg viewBox="0 0 {w} {h}"'
        f' role="img" aria-label="Effort curve {_h(module)} {_h(dim)}"'
        f' style="max-width:{w}px;border:1px solid #ddd;display:block">',
        f'<title>Effort curve: {_h(module)} / {_h(dim)}</title>',
        # Axes
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#aaa"/>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#aaa"/>',
    ]

    # X-axis labels
    for i, tier in enumerate(effort_tiers):
        lines.append(
            f'<text x="{px(i):.1f}" y="{mt+ih+16}" text-anchor="middle"'
            f' font-size="10" fill="#666">{_h(tier)}</text>'
        )

    # Polyline per model (score series)
    for idx, model in enumerate(sorted(model_effort_vals)):
        color = _COLORS[idx % len(_COLORS)]
        pts_parts: list[str] = []
        for i, tier in enumerate(effort_tiers):
            vals = model_effort_vals[model].get(tier, [])
            if not vals:
                continue
            mean = statistics.mean(vals)
            pts_parts.append(f"{px(i):.1f},{py(mean):.1f}")

        if not pts_parts:
            continue

        lines.append(
            f'<polyline points="{" ".join(pts_parts)}"'
            f' fill="none" stroke="{color}" stroke-width="2"/>'
        )

        # Dots + n annotation at each point
        for i, tier in enumerate(effort_tiers):
            vals = model_effort_vals[model].get(tier, [])
            if not vals:
                continue
            mean = statistics.mean(vals)
            n = len(vals)
            cx, cy = px(i), py(mean)
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>'
            )
            if n < LOW_N_THRESHOLD:
                lines.append(
                    f'<text x="{cx:.1f}" y="{cy-7:.1f}" text-anchor="middle"'
                    f' font-size="9" fill="#999">n={n}</text>'
                )

    # Dedicated declination scale: always 0–1, independent of score y-range.
    # Maps declination rate to SVG y-coordinate using the same plot area as
    # the score axis but a fixed [0, 1] domain (right-hand axis).
    def py_decl(rate: float) -> float:
        return mt + ih - rate * ih

    # --- Declination overlay: dashed amber series per condition ---
    has_decl = False
    for cid in sorted(model_effort_vals):
        decl_pts: list[str] = []
        diamond_pts: list[tuple[float, float]] = []
        for i, tier in enumerate(effort_tiers):
            key = (module, cid, tier)
            if key not in decl_map:
                continue
            n_terminal, n_expected = decl_map[key]
            if n_expected > 0:
                rate = n_terminal / n_expected
                decl_pts.append(f"{px(i):.1f},{py_decl(rate):.1f}")
                diamond_pts.append((px(i), py_decl(rate)))

        if not decl_pts:
            continue

        has_decl = True
        lines.append(
            f'<polyline points="{" ".join(decl_pts)}" fill="none"'
            f' stroke="{_DECL_COLOR}" stroke-width="1.5" stroke-dasharray="4,2"'
            f' class="declination-overlay"/>'
        )
        # Diamond markers at each declination point
        for cx_d, cy_d in diamond_pts:
            d = 4
            lines.append(
                f'<polygon'
                f' points="{cx_d:.1f},{cy_d - d:.1f}'
                f' {cx_d + d:.1f},{cy_d:.1f}'
                f' {cx_d:.1f},{cy_d + d:.1f}'
                f' {cx_d - d:.1f},{cy_d:.1f}"'
                f' fill="{_DECL_COLOR}" class="declination-marker"/>'
            )

    # Right-hand axis for declination (rendered when at least one decl series exists)
    if has_decl:
        rx = ml + iw  # right edge of plot area
        # Axis line
        lines.append(
            f'<line x1="{rx}" y1="{mt}" x2="{rx}" y2="{mt+ih}"'
            f' stroke="{_DECL_COLOR}" stroke-width="0.75" stroke-dasharray="2,2"/>'
        )
        # Tick marks and labels at 0, 0.5, 1
        for tick_val, tick_label in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            ty = py_decl(tick_val)
            lines.append(
                f'<line x1="{rx}" y1="{ty:.1f}" x2="{rx+4}" y2="{ty:.1f}"'
                f' stroke="{_DECL_COLOR}" stroke-width="0.75"/>'
            )
            lines.append(
                f'<text x="{rx+6}" y="{ty+3:.1f}" font-size="8"'
                f' fill="{_DECL_COLOR}">{tick_label}</text>'
            )

    # Legend (rendered only when there is at least one declination series)
    if has_decl:
        lx = ml + iw - 130
        ly = mt + 4
        lines.append(
            f'<line x1="{lx}" y1="{ly+3}" x2="{lx+12}" y2="{ly+3}"'
            f' stroke="{_COLORS[0]}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{lx+15}" y="{ly+7}" font-size="9" fill="#555">score</text>'
        )
        lines.append(
            f'<line x1="{lx}" y1="{ly+16}" x2="{lx+12}" y2="{ly+16}"'
            f' stroke="{_DECL_COLOR}" stroke-width="1.5" stroke-dasharray="4,2"/>'
        )
        lines.append(
            f'<text x="{lx+15}" y="{ly+20}" font-size="9" fill="#555"'
            f'>declination rate (right axis)</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: HTML sections
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #fafafa; color: #222; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.15rem; border-bottom: 1px solid #ddd; padding-bottom: .25rem; margin-top: 2rem; }
h3 { font-size: 1rem; margin-top: 1.25rem; }
table { border-collapse: collapse; margin-bottom: 1rem; font-size: .85rem; }
th { background: #f0f4f8; text-align: left; }
th, td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
.greyed { color: #999; font-style: italic; }
.low-n { color: #999; }
.declination { background: #fff8f0; }
.pending { color: #c08000; }
.unscored { color: #c08000; }
.health-ok { color: #16a34a; }
footer { margin-top: 2rem; border-top: 1px solid #ddd; padding-top: .75rem; font-size: .8rem; color: #666; }
.effort-grid { display: flex; flex-wrap: wrap; gap: 1rem; }
.effort-cell { }
"""


def _section_leaderboard(
    modules: list[str],
    dim_values: dict[tuple[str, str, str], list[float]],
    dim_meta: dict[tuple[str, str, str], dict[str, Any]],
    outcome_counts: dict[tuple[str, str], dict[str, int]],
) -> str:
    """Build the leaderboard section HTML."""
    parts = ["<section id='leaderboard'>", "<h2>Leaderboard</h2>"]

    for module in sorted(modules):
        # Collect all conditions and dimensions for this module
        cids: set[str] = set()
        dims: set[str] = set()
        for (m, cid, dim) in dim_values:
            if m == module:
                cids.add(cid)
                dims.add(dim)

        # Also add conditions that have outcomes but zero score rows (e.g. all terminal)
        for (m, cid) in outcome_counts:
            if m == module:
                cids.add(cid)

        if not cids:
            continue

        # Order dims: primary_outcome release_facing first, then others
        def dim_sort_key(dim: str) -> tuple[int, int, str]:
            # Find any key for this dim in this module
            for cid in cids:
                k = (module, cid, dim)
                if k in dim_meta:
                    meta = dim_meta[k]
                    pri = 0 if meta.get("role") == "primary_outcome" else 1
                    rel = 0 if meta.get("release_facing") else 1
                    return (pri, rel, dim)
            return (2, 2, dim)

        ordered_dims = sorted(dims, key=dim_sort_key)

        parts.append(f"<h3>Module: {_h(module)}</h3>")
        parts.append("<div style='overflow-x:auto'>")
        parts.append("<table>")

        # Header row
        header_cells = [
            "<th>Condition (canonical_model)</th>",
            "<th>n_expected</th>",
            "<th>n_scored</th>",
            "<th>n_unscored (pending)</th>",
            "<th class='declination'>Declination<br>(terminal / n_expected)</th>",
        ]
        for dim in ordered_dims:
            # Find statistic_kind for this dim
            sk = None
            for cid in cids:
                k = (module, cid, dim)
                if k in dim_meta:
                    sk = dim_meta[k].get("statistic_kind")
                    break
            if sk == "wilson":
                ci_label = "[95% CI]"
            elif sk == "bootstrap":
                ci_label = "[95% CI]"
            else:
                ci_label = "[n only, no CI]"
            header_cells.append(f"<th>{_h(dim)}<br><small>{ci_label}</small></th>")

        parts.append("<thead><tr>" + "".join(header_cells) + "</tr></thead>")
        parts.append("<tbody>")

        for cid in sorted(cids):
            oc = outcome_counts.get((module, cid), {})
            n_expected = sum(oc.values())
            n_scored = oc.get("scored", 0)
            n_unscored = oc.get("unscored", 0)
            n_terminal = oc.get("terminal_model_signal", 0)
            decl_rate = f"{n_terminal}/{n_expected}" if n_expected else "—"

            row_cells = [
                f"<td>{_h(cid)}</td>",
                f"<td>{n_expected}</td>",
                f"<td>{n_scored}</td>",
                f"<td class='pending'>{n_unscored} (pending)</td>",
                f"<td class='declination'>{decl_rate}</td>",
            ]

            for dim in ordered_dims:
                key = (module, cid, dim)
                vals = dim_values.get(key, [])
                meta = dim_meta.get(key, {})
                sk = meta.get("statistic_kind")
                cell_text, css = _dim_cell(vals, sk)
                td_class = f' class="{_h(css)}"' if css else ""
                row_cells.append(f"<td{td_class}>{_h(cell_text)}</td>")

            parts.append("<tr>" + "".join(row_cells) + "</tr>")

        parts.append("</tbody></table></div>")

    parts.append("</section>")
    return "\n".join(parts)


def _section_effort_curves(
    modules: list[str],
    scores: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> str:
    """Build the effort curves section with inline SVGs.

    Populates both the score-series data and the declination-rate overlay data
    (terminal_model_signal / n_expected per condition × effort tier).  The
    condition/effort for terminal and unscored outcome rows is inferred from the
    (module, model_key) → (cid, effort) mapping built from scored rows.
    """
    parts = ["<section id='effort-curves'>", "<h2>Effort Curves</h2>"]

    # --- Score series ---
    Curve = dict[str, dict[str, list[float]]]  # cid → effort → values
    curve_by_key: dict[tuple[str, str], Curve] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Also track unit_id and (module, model_key) → (cid, effort) for outcome inference
    unit_mce: dict[str, tuple[str, str, str]] = {}
    mk_mce: dict[tuple[str, str], tuple[str, str]] = {}  # (module, mk) → (cid, effort)

    for row in scores:
        if row.get("outcome_class") != "scored":
            continue
        module = str(row.get("module") or "")
        dim = str(row.get("dimension") or "")
        cid = _condition_id(row)
        effort = _effort_value(row)
        uid = str(row.get("unit_id") or "")
        mk = str(row.get("model_key") or _parse_model_key(uid))
        v = _numeric(row.get("value"))

        if module and cid and effort:
            if uid:
                unit_mce[uid] = (module, cid, effort)
            if mk:
                mk_mce[(module, mk)] = (cid, effort)

        if not row.get("release_facing"):
            continue
        if not module or not dim or not cid or not effort or v is None:
            continue
        curve_by_key[(module, dim)][cid][effort].append(v)

    # --- Declination map: (module, cid, effort) → (n_terminal, n_expected) ---
    mce_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for outcome in outcomes:
        uid = str(outcome.get("unit_id") or "")
        oc = str(outcome.get("outcome_class") or "unknown")
        module = _parse_module(uid)
        mk = _parse_model_key(uid)
        mce = unit_mce.get(uid)
        if mce is None:
            ce = mk_mce.get((module, mk))
            if ce:
                mce = (module, ce[0], ce[1])
        if mce:
            mce_counts[mce][oc] += 1

    decl_map: dict[tuple[str, str, str], tuple[int, int]] = {}
    for mce, counts in mce_counts.items():
        n_terminal = counts.get("terminal_model_signal", 0)
        n_expected = sum(counts.values())
        if n_expected > 0:
            decl_map[mce] = (n_terminal, n_expected)

    if not curve_by_key:
        parts.append("<p>No release-facing score rows found.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    for (module, dim) in sorted(curve_by_key):
        model_effort_vals = curve_by_key[(module, dim)]
        # Collect effort tiers present in this (module, dim)
        effort_set: set[str] = set()
        for m_dict in model_effort_vals.values():
            effort_set.update(m_dict.keys())
        effort_tiers = sorted(effort_set, key=_effort_sort_key)

        svg = _effort_svg(module, dim, model_effort_vals, decl_map, effort_tiers)
        if not svg:
            continue
        parts.append(f"<h3>{_h(module)} / {_h(dim)}</h3>")
        parts.append("<div class='effort-cell'>")
        parts.append(svg)
        parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


def _section_blocks(blocks: list[dict[str, Any]]) -> str:
    """Build the refusal/block breakdown section."""
    parts = ["<section id='blocks'>", "<h2>Refusal / Block Breakdown</h2>"]

    if not blocks:
        parts.append("<p>No blocks recorded.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    # Group by (module/unit_id prefix, category)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        cat = str(block.get("category") or "unknown")
        by_cat[cat].append(block)

    parts.append("<table>")
    parts.append(
        "<thead><tr>"
        "<th>category</th><th>unit_id</th><th>member_id</th><th>evidence</th>"
        "</tr></thead><tbody>"
    )
    for cat in sorted(by_cat):
        for block in by_cat[cat]:
            uid = _h(block.get("unit_id", ""))
            mid = _h(block.get("member_id", ""))
            # Evidence is text, not a hyperlink (no transcripts by default)
            ev = _h(block.get("evidence", ""))
            parts.append(
                f"<tr>"
                f"<td>{_h(cat)}</td>"
                f"<td>{uid}</td>"
                f"<td>{mid}</td>"
                f"<td>{ev}</td>"
                f"</tr>"
            )
    parts.append("</tbody></table>")
    parts.append("</section>")
    return "\n".join(parts)


def _section_run_health(
    manifest: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> str:
    """Build the run health snapshot section.

    Adds per-member state columns — done (scored), terminal
    (terminal_model_signal), and owed (missing) — derived from outcomes.jsonl
    grouped by the ``member_id`` field on each outcome row.
    """
    parts = ["<section id='run-health'>", "<h2>Run Health</h2>"]

    members = manifest.get("members") or []
    union = manifest.get("union") or {}
    collisions = union.get("collisions") or []
    member_errors = union.get("member_errors") or []
    warnings = union.get("warnings") or []

    # Build quick lookup for collision/error counts by member_id
    collision_by_member: dict[str, int] = defaultdict(int)
    for coll in collisions:
        kept = coll.get("kept_member", "")
        collision_by_member[kept] += 1

    error_by_member: dict[str, int] = defaultdict(int)
    for err in member_errors:
        m = err.get("member") or err.get("member_id") or err.get("path") or ""
        error_by_member[str(m)] += 1

    # Per-member outcome state counts from outcomes.jsonl
    member_state: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in outcomes:
        mid = str(outcome.get("member_id") or "")
        oc = str(outcome.get("outcome_class") or "unknown")
        if mid:
            member_state[mid][oc] += 1

    parts.append("<table>")
    parts.append(
        "<thead><tr>"
        "<th>member_id</th><th>role</th><th>fingerprint</th>"
        "<th>done</th><th>unscored</th><th>terminal</th><th>owed</th>"
        "<th>collisions</th><th>member_errors</th>"
        "</tr></thead><tbody>"
    )

    for member in members:
        mid = str(member.get("member_id") or "")
        role = str(member.get("role") or "")
        fp = str(member.get("contract_fingerprint") or member.get("fingerprint") or "—")
        n_coll = collision_by_member.get(mid, 0)
        n_err = error_by_member.get(mid, 0)
        state = member_state.get(mid, {})
        n_done = state.get("scored", 0)
        n_unscored = state.get("unscored", 0)
        n_terminal = state.get("terminal_model_signal", 0)
        n_owed = state.get("missing", 0)
        fp_class = "health-ok" if fp and fp != "—" else ""
        fp_cell = f'<td class="{fp_class}">{_h(fp[:16])}…</td>' if len(fp) > 16 else f"<td>{_h(fp)}</td>"
        parts.append(
            f"<tr>"
            f"<td>{_h(mid)}</td>"
            f"<td>{_h(role)}</td>"
            f"{fp_cell}"
            f"<td>{n_done}</td>"
            f"<td>{n_unscored}</td>"
            f"<td>{n_terminal}</td>"
            f"<td>{n_owed}</td>"
            f"<td>{n_coll}</td>"
            f"<td>{n_err}</td>"
            f"</tr>"
        )

    parts.append("</tbody></table>")

    # Summary counts
    n_units = len(union.get("units") or [])
    parts.append(
        f"<p>units: {n_units} | collisions: {len(collisions)} | "
        f"member_errors: {len(member_errors)} | warnings: {len(warnings)}</p>"
    )

    parts.append("</section>")
    return "\n".join(parts)


def _section_footer(manifest: dict[str, Any]) -> str:
    """Build the footer with provenance fields."""
    excl_raw = manifest.get("exclusion_policy") or ""
    if isinstance(excl_raw, dict):
        excl_id = _h(excl_raw.get("id") or "")
        excl_def = _h(excl_raw.get("definition") or "")
        excl = f"{excl_id} — {excl_def}" if excl_def else excl_id
    else:
        excl = _h(excl_raw)
    proj_ver = _h(manifest.get("projection_version") or "")
    tool_ver = _h(manifest.get("tool_version") or "")
    contains_transcripts = manifest.get("contains_transcripts", False)
    cert = manifest.get("certificate") or {}
    pairwise = cert.get("pairwise") or []
    hashes: list[str] = []
    for pair in pairwise:
        for k, v in (pair or {}).items():
            if "hash" in k.lower() and v:
                hashes.append(f"{_h(k)}={_h(str(v)[:12])}")
    hash_text = ", ".join(hashes) if hashes else "—"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "<footer>",
        f"<p><strong>Provenance</strong></p>",
        f"<p>exclusion_policy: {excl}</p>",
        f"<p>projection_version: {proj_ver}</p>",
        f"<p>tool_version: {tool_ver}</p>",
        f"<p>contains_transcripts: {str(contains_transcripts).lower()}</p>",
        f"<p>certificate hashes: {hash_text}</p>",
        f"<p>generated: {ts}</p>",
        "</footer>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public: render_bundle_report
# ---------------------------------------------------------------------------


def render_bundle_report(bundle_dir: Path | str) -> str:
    """Render a self-contained HTML report for a bundle.

    Returns an HTML string with no external references (no http:// or https://).
    Cells with n < 5 carry the 'greyed low-n' CSS class.  Wilson dims show 95%
    CI; bootstrap dims show 95% CI; None-statistic_kind dims show 'n only, no CI'.
    Unscored units appear as a pending count, never entering any rate denominator.
    """
    bundle_dir = Path(bundle_dir)

    # Load data
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    scores = _load_jsonl(bundle_dir / "data" / "scores.jsonl")
    outcomes = _load_jsonl(bundle_dir / "data" / "outcomes.jsonl")
    blocks = _load_jsonl(bundle_dir / "data" / "blocks.jsonl")

    # --- Build lookup maps ---
    # unit_id → condition_id (from scored rows)
    unit_condition: dict[str, str] = {}
    # (module, model_key) → condition_id (to infer condition for terminal/unscored)
    model_key_condition: dict[tuple[str, str], str] = {}

    for row in scores:
        uid = str(row.get("unit_id") or "")
        cid = _condition_id(row)
        module = str(row.get("module") or "")
        mk = str(row.get("model_key") or _parse_model_key(uid))
        if cid:
            if uid:
                unit_condition[uid] = cid
            if module and mk:
                model_key_condition[(module, mk)] = cid

    # --- Outcome counts per (module, condition_id) ---
    outcome_counts = _build_outcome_counts(outcomes, unit_condition, model_key_condition)

    # --- Dimension value groups ---
    dim_values, dim_meta = _build_dim_groups(scores)

    # --- Collect modules ---
    modules: set[str] = set()
    for (m, cid, dim) in dim_values:
        modules.add(m)
    for (m, cid) in outcome_counts:
        modules.add(m)

    # --- Build HTML ---
    exp_id = _h(manifest.get("experiment_id") or bundle_dir.name)
    title = _h(manifest.get("title") or exp_id)
    version = _h(manifest.get("version") or "")

    page_title = f"Bundle Report: {exp_id}"
    if version:
        page_title += f" v{version}"

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        f'<meta charset="utf-8">',
        f'<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_h(page_title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{_h(page_title)}</h1>",
    ]
    if title and title != exp_id:
        html_parts.append(f"<p>{title}</p>")

    html_parts.append(
        _section_leaderboard(sorted(modules), dim_values, dim_meta, outcome_counts)
    )
    html_parts.append(
        _section_effort_curves(sorted(modules), scores, outcomes)
    )
    html_parts.append(_section_blocks(blocks))
    html_parts.append(_section_run_health(manifest, outcomes))
    html_parts.append(_section_footer(manifest))

    html_parts += ["</body>", "</html>"]

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Public: write_bundle_report  (hook for bundle.py)
# ---------------------------------------------------------------------------


def write_bundle_report(bundle_dir: Path | str) -> Path:
    """Write report/index.html inside the bundle; return the output path.

    Applies the shared public-safety text gate from ``artifact_privacy`` before
    writing so a privacy-violating string aborts rather than landing on disk.
    This is the same gate used by the REPORT.md emitter in bundle.py — single
    source of truth.
    """
    from suite_tools.artifact_privacy import assert_text_public_safe

    bundle_dir = Path(bundle_dir)
    html_text = render_bundle_report(bundle_dir)

    # Defense-in-depth: data is already gated at bundle emit time, but a check
    # before writing the rendered HTML catches any renderer bug that might
    # inadvertently surface a private string.
    assert_text_public_safe(html_text)

    report_dir = bundle_dir / "report"
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m suite_tools.bundle_report",
        description="Render an HTML report from a bundle directory.",
    )
    parser.add_argument("bundle_dir", type=Path, help="Bundle directory.")
    args = parser.parse_args(argv)

    out = write_bundle_report(args.bundle_dir)
    print(f"Wrote report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
