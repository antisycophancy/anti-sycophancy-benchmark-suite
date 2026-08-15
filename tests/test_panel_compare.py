import json

from suite_tools.panel_compare import compare_panel, render_markdown, write_panel_comparison


def _write_score(path, *, judge_model, **scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"judge_model": judge_model, **scores}))


def test_compare_panel_flags_disagreement_and_invalid_values(tmp_path):
    _write_score(
        tmp_path / "aita" / "judge-a" / "gemini-flash_item0_scores.json",
        judge_model="judge/a",
        outcome_a=2,
        resistance_a=1,
        therapeutic_a=3,
        outcome_b=0,
        resistance_b=0,
        therapeutic_b=None,
        consistency=0,
    )
    _write_score(
        tmp_path / "aita" / "judge-b" / "gemini-flash_item0_scores.json",
        judge_model="judge/b",
        outcome_a=2,
        resistance_a=1,
        therapeutic_a=3,
        outcome_b=0,
        resistance_b=0,
        therapeutic_b=2,
        consistency=0,
    )

    comparison = compare_panel(tmp_path)
    case = comparison["modules"]["aita"]["cases"][0]

    assert case["dimensions"]["outcome_a"]["unanimous"] is True
    assert case["dimensions"]["outcome_a"]["needs_review"] is False
    assert case["dimensions"]["therapeutic_b"]["needs_review"] is True
    assert case["dimensions"]["therapeutic_b"]["median"] == 2
    assert case["dimensions"]["therapeutic_b"]["missing_judges"] == ["judge/a"]
    assert case["dimensions"]["therapeutic_b"]["invalid_judges"] == ["judge/a"]


def test_compare_panel_tracks_epistemic_majority(tmp_path):
    for judge_key, judge_model, consistency in [
        ("gpt", "openai/gpt-5.5", 1),
        ("opus", "anthropic/claude-opus-4.7", 0),
        ("gemini", "google/gemini-3.1-pro-preview", 1),
    ]:
        _write_score(
            tmp_path / "epis" / judge_key / "gemini-flash_item0_mirror_scores.json",
            judge_model=judge_model,
            persistence=1,
            integrity=2,
            consistency=consistency,
            amplification=1,
        )

    comparison = compare_panel(tmp_path)
    case = comparison["modules"]["epis"]["cases"][0]

    assert case["dimensions"]["consistency"]["unanimous"] is False
    assert case["dimensions"]["consistency"]["majority"] == 1
    assert case["dimensions"]["consistency"]["median"] == 1


def test_compare_panel_reports_self_excluded_scores_for_self_judge(tmp_path):
    for judge_key, judge_model, persistence in [
        ("gpt", "openai/gpt-5.5", 1),
        ("opus", "anthropic/claude-opus-4.7", 0),
        ("gemini", "google/gemini-3.1-pro-preview", 0),
    ]:
        _write_score(
            tmp_path / "epis" / judge_key / "gpt-5-5_item0_pickside_scores.json",
            judge_model=judge_model,
            model_id="openai/gpt-5.5",
            persistence=persistence,
            integrity=1,
            consistency=1,
            amplification=0,
        )

    comparison = compare_panel(tmp_path)
    case = comparison["modules"]["epis"]["cases"][0]
    persistence = case["dimensions"]["persistence"]

    assert case["target_model_ids"] == ["openai/gpt-5.5"]
    assert persistence["median"] == 0
    assert persistence["self_judges"] == ["openai/gpt-5.5"]
    assert persistence["self_excluded_median"] == 0
    assert persistence["self_excluded_majority"] == 0
    assert persistence["self_excluded_n"] == 2


def test_compare_panel_matches_preview_judge_to_short_target_key(tmp_path):
    for judge_key, judge_model, integrity in [
        ("gemini", "google/gemini-3.1-pro-preview", 2),
        ("gpt", "openai/gpt-5.5", 1),
        ("opus", "anthropic/claude-opus-4.7", 1),
    ]:
        _write_score(
            tmp_path / "epis" / judge_key / "gemini-3-1-pro_item0_delusion_scores.json",
            judge_model=judge_model,
            model="gemini-3-1-pro",
            persistence=1,
            integrity=integrity,
        )

    comparison = compare_panel(tmp_path)
    case = comparison["modules"]["epis"]["cases"][0]
    integrity = case["dimensions"]["integrity"]

    assert integrity["self_judges"] == ["google/gemini-3.1-pro-preview"]
    assert integrity["self_excluded_median"] == 1


def test_compare_panel_matches_therapeutic_harness_targets_to_underlying_judges(tmp_path):
    for judge_key, judge_model, outcome_a in [
        ("opus", "anthropic/claude-opus-4.7", 2),
        ("gpt", "openai/gpt-5.5", 1),
        ("gemini", "google/gemini-3.1-pro-preview", 1),
    ]:
        _write_score(
            tmp_path / "aita" / judge_key / "th-opus-4-7_item0_scores.json",
            judge_model=judge_model,
            model_id="therapeutic-harness/th-opus-4-7",
            model="th-opus-4-7",
            outcome_a=outcome_a,
        )

    comparison = compare_panel(tmp_path)
    case = comparison["modules"]["aita"]["cases"][0]
    outcome_a = case["dimensions"]["outcome_a"]

    assert outcome_a["self_judges"] == ["anthropic/claude-opus-4.7"]
    assert outcome_a["self_excluded_median"] == 1
    assert outcome_a["self_excluded_majority"] == 1


def test_write_panel_comparison_outputs_json_and_report(tmp_path):
    _write_score(
        tmp_path / "aita" / "judge-a" / "gemini-flash_item0_scores.json",
        judge_model="judge/a",
        outcome_a=2,
    )

    comparison = write_panel_comparison(tmp_path, tmp_path / "panel")
    report = (tmp_path / "panel" / "PANEL_REPORT.md").read_text()
    data = json.loads((tmp_path / "panel" / "panel_comparison.json").read_text())

    assert data["generated_at"] == comparison["generated_at"]
    assert "Judge Panel Comparison" in report
    assert "self-excluded medians" in report
    assert "`outcome_a`" in report


def test_render_markdown_includes_policy_and_scores(tmp_path):
    _write_score(
        tmp_path / "epis" / "judge-a" / "gemini-flash_item0_delusion_scores.json",
        judge_model="judge/a",
        persistence=1,
        integrity=2,
    )

    markdown = render_markdown(compare_panel(tmp_path))

    assert "calibration signal" in markdown
    assert "judge/a=1" in markdown
    assert "`persistence`" in markdown


def test_render_markdown_uses_na_for_no_majority(tmp_path):
    for judge_key, score in [("judge-a", 0), ("judge-b", 1), ("judge-c", 2)]:
        _write_score(
            tmp_path / "aita" / judge_key / "gemini-flash_item0_scores.json",
            judge_model=judge_key,
            outcome_a=score,
        )

    markdown = render_markdown(compare_panel(tmp_path))

    assert "| `gemini-flash_item0` | `outcome_a` | 1 | n/a | n/a |" in markdown
