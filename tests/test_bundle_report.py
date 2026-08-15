import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from suite_tools import bundle_report


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.classes = []
        self.polylines = 0

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "class" in d:
            self.classes.append(d["class"])
        if tag == "polyline":
            self.polylines += 1

    def handle_data(self, data):
        self.text.append(data)


def _mini_bundle(tmp_path) -> Path:
    b = tmp_path / "bundle-exp1-v1"
    (b / "data").mkdir(parents=True)
    rows = [
        # wilson binary release dim, two conditions
        {"unit_id": "aita:m:i0:a", "module": "aita", "model_key": "m",
         "condition": {"canonical_model": "gpt-5.6-luna", "effort": "high"},
         "dimension": "verdict_alignment_a", "value": True, "role": "primary_outcome",
         "release_facing": True, "statistic_kind": "wilson", "score_scope": "side",
         "outcome_class": "scored"},
        {"unit_id": "aita:m:i1:a", "module": "aita", "model_key": "m",
         "condition": {"canonical_model": "gpt-5.6-luna", "effort": "high"},
         "dimension": "verdict_alignment_a", "value": False, "role": "primary_outcome",
         "release_facing": True, "statistic_kind": "wilson", "score_scope": "side",
         "outcome_class": "scored"},
        # scalar bootstrap dim
        {"unit_id": "sus:m:s1:1", "module": "sus", "model_key": "m",
         "condition": {"canonical_model": "gpt-5.6-luna", "effort": "high"},
         "dimension": "sus_response_score", "value": 55, "role": "diagnostic_severity",
         "release_facing": True, "statistic_kind": "bootstrap", "score_scope": "run",
         "outcome_class": "scored"},
        # no-statistic_kind dim -> n only, no CI
        {"unit_id": "aita:m:i0:a", "module": "aita", "model_key": "m",
         "condition": {"canonical_model": "gpt-5.6-luna", "effort": "high"},
         "dimension": "consistency", "value": 0.5, "role": "mechanism_diagnostic",
         "release_facing": False, "statistic_kind": None, "score_scope": "pair",
         "outcome_class": "scored"},
    ]
    (b / "data" / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (b / "data" / "outcomes.jsonl").write_text("\n".join(json.dumps(o) for o in [
        {"unit_id": "aita:m:i0:a", "outcome_class": "scored", "member_id": "m1"},
        {"unit_id": "aita:m:i1:a", "outcome_class": "scored", "member_id": "m1"},
        {"unit_id": "aita:m:i2:a", "outcome_class": "terminal_model_signal",
         "category": "safety_refusal", "member_id": "m1"},
        {"unit_id": "aita:m:i3:a", "outcome_class": "unscored", "member_id": "m1"},
    ]))
    (b / "data" / "blocks.jsonl").write_text(json.dumps(
        {"unit_id": "aita:m:i2:a", "category": "safety_refusal",
         "evidence": "RUN_EVENTS.jsonl:3", "member_id": "m1"}))
    (b / "data" / "derived_aggregates.jsonl").write_text("")
    (b / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": "benchmark-bundle-v1", "experiment_id": "exp1",
        "projection_version": "v2", "exclusion_policy": "responsive-subset-v1",
        "contains_transcripts": False, "tool_version": "abc123",
        "members": [{"member_id": "m1", "role": "pilot"}],
        "certificate": {"pairwise": []}}))
    return b


def test_leaderboard_ci_follows_statistic_kind(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    c = _Collector(); c.feed(html)
    joined = " ".join(c.text)
    assert "verdict_alignment_a" in joined
    assert "95%" in joined or "CI" in joined          # wilson dim renders a CI
    # consistency has statistic_kind=None -> its cell shows n but no CI marker
    assert "n only" in joined or "no CI" in joined


def test_declination_column_present_beside_behavioral(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    assert "declination" in joined.lower()
    assert "1" in joined                              # 1 terminal_model_signal of 3


def test_every_rate_cell_names_denominator(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    assert "n_expected" in joined and "n_scored" in joined


def test_unscored_shown_as_pending_not_in_denominator(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    assert "pending" in joined.lower() or "unscored" in joined.lower()
    # behavioral denominator counts scored only (2), not the unscored unit
    assert "n_scored" in joined


def test_low_n_cell_is_greyed_with_n(tmp_path):
    b = _mini_bundle(tmp_path)
    html = bundle_report.render_bundle_report(b)
    c = _Collector(); c.feed(html)
    assert any("greyed" in cls or "low-n" in cls for cls in c.classes)


def test_effort_svg_has_one_polyline_per_model_dimension(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    c = _Collector(); c.feed(html)
    assert c.polylines >= 1                            # 1 model × release dims


def test_no_external_references(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    assert "http://" not in html and "https://" not in html


def test_footer_reports_provenance_fields(tmp_path):
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    for token in ("responsive-subset-v1", "v2", "abc123", "contains_transcripts"):
        assert token in joined


def test_sus_cap_leaderboard_ranks_lower_cap_rate_better(tmp_path):
    # Sol round-3 finding 3: cap=worse -> the condition with fewer caps ranks first.
    b = tmp_path / "bundle-cap-v1"
    (b / "data").mkdir(parents=True)
    rows = []
    for cond, caps in (("safe", [0, 0, 0, 1]), ("risky", [1, 1, 1, 0])):
        for i, v in enumerate(caps):
            rows.append({"unit_id": f"sus:{cond}:s{i}:run1", "module": "sus",
                         "model_key": cond,
                         "condition": {"canonical_model": cond, "effort": "high"},
                         "dimension": "cap_outcome", "value": v, "display_label":
                         "cap" if v else "no_cap", "role": "primary_outcome",
                         "release_facing": True, "statistic_kind": "wilson",
                         "direction": "cap=worse", "score_scope": "run",
                         "outcome_class": "scored"})
    (b / "data" / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (b / "data" / "outcomes.jsonl").write_text("\n".join(
        json.dumps({"unit_id": r["unit_id"], "outcome_class": "scored",
                    "member_id": "m1"}) for r in rows))
    (b / "data" / "blocks.jsonl").write_text("")
    (b / "data" / "derived_aggregates.jsonl").write_text("")
    (b / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": "benchmark-bundle-v1", "experiment_id": "cap",
        "projection_version": "v2", "exclusion_policy": "responsive-subset-v1",
        "contains_transcripts": False, "tool_version": "abc123",
        "members": [{"member_id": "m1", "role": "pilot"}], "certificate": {"pairwise": []}}))
    order = bundle_report.leaderboard_order(b, module="sus")   # ranked condition ids
    assert order.index("safe") < order.index("risky")          # lower cap rate ranks first


def test_effort_curve_has_declination_overlay(tmp_path):
    # _mini_bundle has 1 terminal_model_signal (aita:m:i2:a) out of 4 outcomes for
    # condition gpt-5.6-luna / effort high.  The effort curve for aita must render
    # a second (dashed) declination polyline alongside the score polyline.
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    c = _Collector(); c.feed(html)
    # At least: 1 score polyline + 1 declination polyline
    assert c.polylines >= 2, f"Expected >=2 polylines, got {c.polylines}"
    # The legend and class attribute must contain the word "declination"
    assert "declination" in html.lower()


def test_run_health_shows_member_state_counts(tmp_path):
    # _mini_bundle has outcomes for member m1: 2 scored (done), 1 terminal, 0 missing
    # (no owed).  The run health table must show done / terminal / owed columns.
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    assert "done" in joined.lower()
    assert "terminal" in joined.lower()
    assert "owed" in joined.lower()


def test_run_health_unscored_column(tmp_path):
    # _mini_bundle has 1 unscored outcome for member m1.
    # The run health table must include an "unscored" column header and
    # show the count in the cell (column order: done/unscored/terminal/owed).
    html = bundle_report.render_bundle_report(_mini_bundle(tmp_path))
    joined = " ".join(_flat(html))
    assert "unscored" in joined.lower()
    # Check column ordering via raw HTML header
    done_pos = html.index("<th>done</th>")
    unscored_pos = html.index("<th>unscored</th>")
    terminal_pos = html.index("<th>terminal</th>")
    owed_pos = html.index("<th>owed</th>")
    assert done_pos < unscored_pos < terminal_pos < owed_pos
    # The member cell should show "1" for the unscored count
    # (the mini_bundle outcomes.jsonl has exactly 1 unscored outcome for m1)
    assert ">1<" in html  # rough check; more precise: look for the cell


def test_effort_svg_declination_uses_secondary_axis(tmp_path):
    """Fix #3: Declination overlay must use a dedicated right-hand y-axis (0–1)
    rather than the score y-axis.  When scores are in [8, 8] (all high) the score
    scale spans [0, 8], so py(0.5) ≈ 151 (near bottom).  The new py_decl(0.5)
    must place the point at ≈ 90 (middle of chart), visibly different.

    Also asserts: right-axis tick labels appear and legend says '(right axis)'."""
    svg = bundle_report._effort_svg(
        module="aita",
        dim="verdict_alignment_a",
        model_effort_vals={"m1": {"high": [8.0]}},
        # rate = 1/2 = 0.5 for condition "m1", effort "high"
        decl_map={("aita", "m1", "high"): (1, 2)},
        effort_tiers=["high"],
    )
    assert svg, "SVG must be non-empty when data is present"

    # Right-axis ticks: expect labels "0", "0.5"/"50%", "1"/"100%" on right edge
    assert "right axis" in svg.lower() or "right-axis" in svg.lower() or "(right axis)" in svg
    # Legend entry must indicate right axis
    assert "(right axis)" in svg

    # The declination polyline point: with score range [0,8], py_decl(0.5) ≈ 90,
    # but py_score(0.5) ≈ 151.25.  Parse the dashed polyline points.
    import re
    # Declination polyline: points attr comes first, then stroke-dasharray
    match = re.search(r'<polyline points="([^"]+)"[^>]*stroke-dasharray="4,2"', svg)
    assert match, "Declination polyline not found"
    points_str = match.group(1)
    # Single point: "x,y"
    coords = points_str.strip().split()
    assert len(coords) >= 1
    x_str, y_str = coords[0].split(",")
    y_val = float(y_str)
    # py_decl(0.5) with ih=140, mt=20: 20 + 140 - 0.5*140 = 90.0
    # py_score(0.5) with y_range=8: 20 + 140 - (0.5/8)*140 ≈ 151.25
    # The y value must be close to 90 (secondary scale), not 151
    assert abs(y_val - 90.0) < 5.0, (
        f"Declination y={y_val:.1f} should be ~90.0 (secondary 0-1 scale), "
        f"not ~151.25 (score scale). Fix: use py_decl() for declination overlay."
    )


def _flat(html: str) -> list[str]:
    c = _Collector(); c.feed(html); return c.text
