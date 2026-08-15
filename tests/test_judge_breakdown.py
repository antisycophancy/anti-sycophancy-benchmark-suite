import json

from suite_tools.judge_breakdown import (
    build_judge_breakdown,
    main,
    model_family,
    render_markdown,
)


def _write_score(path, *, model_id, label, judge_entries, extra=None):
    payload = {
        "model": label.lower(),
        "model_id": model_id,
        "label": label,
        "judge_scores": judge_entries,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload))


def _aita_style_dir(tmp_path):
    """Two models x two items, 3-judge panel, under an aita/ module dir.

    The openai judge scores the openai model's consistency 1.0 on every item
    while the other judges score it 0.0 — a planted same-family tilt.
    """
    results = tmp_path / "results" / "aita"
    results.mkdir(parents=True)
    judges = [
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.7",
        "google/gemini-3.1-pro-preview",
    ]
    for item in (0, 1):
        _write_score(
            results / f"gpt_item{item}_scores.json",
            model_id="openai/gpt-5.5",
            label="GPT",
            judge_entries=[
                {
                    "judge_model": judges[0],
                    "consistency": 1.0,
                    "outcome_a": 2,
                    # turn numbers are positions, not scores
                    "deterministic_verdict_turn_a": 4,
                },
                {"judge_model": judges[1], "consistency": 0.0, "outcome_a": 2},
                {"judge_model": judges[2], "consistency": 0.0, "outcome_a": 2},
            ],
        )
        _write_score(
            results / f"opus_item{item}_scores.json",
            model_id="anthropic/claude-opus-4.7",
            label="Opus",
            judge_entries=[
                {"judge_model": judges[0], "consistency": 1.0, "outcome_a": 1},
                {"judge_model": judges[1], "consistency": 1.0, "outcome_a": 1},
                {"judge_model": judges[2], "consistency": 1.0, "outcome_a": 1},
            ],
        )
    return results


def test_model_family_extraction():
    assert model_family("openai/gpt-5.5") == "openai"
    assert model_family("anthropic/claude-opus-4.7") == "anthropic"
    assert model_family("google/gemini-3.1-pro-preview") == "google"
    assert model_family("claude-sonnet-4.6") == "anthropic"
    assert model_family("gpt-5.4") == "openai"
    assert model_family("gemini-3-flash") == "google"
    assert model_family("my-pipeline/v1") == "my-pipeline"
    assert model_family(None) == "unknown"


def test_breakdown_aggregates_per_judge_means_grouped_by_module(tmp_path):
    results = _aita_style_dir(tmp_path)

    breakdown = build_judge_breakdown([results])

    assert list(breakdown["modules"]) == ["aita"]
    gpt = breakdown["modules"]["aita"]["models"]["openai/gpt-5.5"]
    by_judge = gpt["judges"]
    assert by_judge["openai/gpt-5.5"]["dimensions"]["consistency"]["mean"] == 1.0
    assert by_judge["openai/gpt-5.5"]["dimensions"]["consistency"]["n"] == 2
    assert by_judge["anthropic/claude-opus-4.7"]["dimensions"]["consistency"]["mean"] == 0.0
    # Metadata and turn positions must not leak in as dimensions.
    assert "judge_model" not in by_judge["openai/gpt-5.5"]["dimensions"]
    assert "deterministic_verdict_turn_a" not in by_judge["openai/gpt-5.5"]["dimensions"]


def test_same_dimension_name_does_not_merge_across_modules(tmp_path):
    _aita_style_dir(tmp_path)
    epis = tmp_path / "results" / "epis"
    epis.mkdir(parents=True)
    _write_score(
        epis / "gpt_item0_delusion_scores.json",
        model_id="openai/gpt-5.5",
        label="GPT",
        judge_entries=[
            {"judge_model": "openai/gpt-5.5", "consistency": 0.0, "persistence": 2},
        ],
        extra={"test_type": "delusion"},
    )

    breakdown = build_judge_breakdown([tmp_path / "results"])

    aita_consistency = breakdown["modules"]["aita"]["models"]["openai/gpt-5.5"][
        "judges"
    ]["openai/gpt-5.5"]["dimensions"]["consistency"]
    epis_consistency = breakdown["modules"]["epistemic"]["models"]["openai/gpt-5.5"][
        "judges"
    ]["openai/gpt-5.5"]["dimensions"]["consistency"]
    assert aita_consistency["n"] == 2
    assert epis_consistency["n"] == 1
    assert epis_consistency["mean"] == 0.0


def test_breakdown_computes_paired_delta_vs_other_judges(tmp_path):
    results = _aita_style_dir(tmp_path)

    breakdown = build_judge_breakdown([results])

    models = breakdown["modules"]["aita"]["models"]
    openai_judge = models["openai/gpt-5.5"]["judges"]["openai/gpt-5.5"]
    # openai judge: 1.0 each item; others' mean per item: 0.0 -> delta +1.0
    assert openai_judge["dimensions"]["consistency"]["delta_vs_other_judges"] == 1.0
    # On a dimension where all judges agree the delta is 0.
    assert openai_judge["dimensions"]["outcome_a"]["delta_vs_other_judges"] == 0.0
    # For the opus model all judges agree -> all deltas 0.
    for judge_data in models["anthropic/claude-opus-4.7"]["judges"].values():
        assert judge_data["dimensions"]["consistency"]["delta_vs_other_judges"] == 0.0


def test_same_family_pairs_flagged(tmp_path):
    results = _aita_style_dir(tmp_path)

    breakdown = build_judge_breakdown([results])

    models = breakdown["modules"]["aita"]["models"]
    gpt = models["openai/gpt-5.5"]
    assert gpt["judges"]["openai/gpt-5.5"]["same_family"] is True
    assert gpt["judges"]["anthropic/claude-opus-4.7"]["same_family"] is False
    opus = models["anthropic/claude-opus-4.7"]
    assert opus["judges"]["anthropic/claude-opus-4.7"]["same_family"] is True
    assert opus["judges"]["openai/gpt-5.5"]["same_family"] is False


def test_handles_sus_style_judge_key_and_skips_non_panel_files(tmp_path):
    results = tmp_path / "sus-run"
    results.mkdir()
    _write_score(
        results / "model_item0_scores.json",
        model_id="anthropic/claude-sonnet-4.6",
        label="Sonnet",
        judge_entries=[
            # sus scorer writes "judge" instead of "judge_model"
            {"judge": "google/gemini-3.1-pro-preview", "irq": 3, "pr_explanation": "text"},
            {"judge": "openai/gpt-5.5", "irq": 2, "pr_explanation": "text"},
        ],
    )
    # A file with no judge_scores and no single judge_model must be ignored.
    (results / "FINAL_RESULTS.json").write_text(json.dumps({"metadata": {}}))

    breakdown = build_judge_breakdown([results])

    sonnet = breakdown["modules"]["sus"]["models"]["anthropic/claude-sonnet-4.6"]
    assert sonnet["judges"]["google/gemini-3.1-pro-preview"]["dimensions"]["irq"]["mean"] == 3
    # Explanation strings are not dimensions.
    assert "pr_explanation" not in sonnet["judges"]["openai/gpt-5.5"]["dimensions"]


def test_single_judge_files_without_judge_scores_use_top_level_fields(tmp_path):
    """aita's single-judge path writes scores at the top level with no
    judge_scores list; the breakdown must still attribute them to the judge."""
    results = tmp_path / "results" / "aita"
    results.mkdir(parents=True)
    (results / "gemini-flash_item0_scores.json").write_text(json.dumps({
        "model": "gemini-flash",
        "model_id": "google/gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "judge_model": "google/gemini-3.1-pro-preview",
        "consistency": 1,
        "outcome_a": 2,
        "item_idx": 0,
    }))
    # Panel files join judge ids with ", " at top level — no fallback there.
    (results / "other_item0_scores.json").write_text(json.dumps({
        "model_id": "openai/gpt-5.5",
        "judge_model": "judge-a, judge-b",
        "consistency": 1,
    }))

    breakdown = build_judge_breakdown([results])

    models = breakdown["modules"]["aita"]["models"]
    judge = models["google/gemini-3-flash-preview"]["judges"]["google/gemini-3.1-pro-preview"]
    assert judge["dimensions"]["consistency"]["mean"] == 1
    assert judge["dimensions"]["outcome_a"]["mean"] == 2
    assert judge["same_family"] is True
    assert "item_idx" not in judge["dimensions"]
    assert "openai/gpt-5.5" not in models


def test_markdown_flags_same_family_tilt(tmp_path):
    results = _aita_style_dir(tmp_path)

    breakdown = build_judge_breakdown([results])
    text = render_markdown(breakdown)

    assert "aita" in text
    assert "openai/gpt-5.5" in text
    assert "same-family" in text
    assert "+1.00" in text


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    results = _aita_style_dir(tmp_path)
    out_dir = tmp_path / "out"

    code = main([str(results), "--output-dir", str(out_dir)])

    assert code == 0
    data = json.loads((out_dir / "judge_breakdown.json").read_text())
    assert "modules" in data
    assert (out_dir / "JUDGE_BREAKDOWN.md").exists()
