import json
from types import SimpleNamespace

import pytest

from sus_bench import cli
from sus_bench import scorer
from suite_tools.paid_call_lease import set_paid_call_policy


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_merge_cost_summaries_rejects_nonfinite_or_negative_legacy_money(bad_value):
    merged = cli._merge_cost_summaries(
        {
            "total_cost_usd": bad_value,
            "cost_by_model": {"bad/model": bad_value},
            "reported_cost_usd": bad_value,
        },
        {},
    )

    assert merged["total_cost_usd"] == 0
    assert merged["cost_by_model"]["bad/model"] == 0
    assert merged["reported_cost_usd"] == 0
    assert merged["usage_anomaly_count"] >= 3
    assert merged["invalid_usage_fields"]["legacy_cost"] >= 3
    json.dumps(merged, allow_nan=False)


def test_sus_score_parallelism_can_be_configured_by_arg_or_env(monkeypatch):
    monkeypatch.setenv("BENCHMARK_SCORE_MAX_PARALLEL", "5")
    monkeypatch.setenv("BENCHMARK_SUS_SCORE_MAX_PARALLEL", "7")

    assert cli._configured_score_parallelism(None) == 7
    assert cli._configured_score_parallelism(3) == 3
    assert cli._configured_score_parallelism("bad") == 2


def test_sus_score_parallelism_cannot_exceed_global_policy(tmp_path, monkeypatch):
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    set_paid_call_policy(1, lease_dir=lease_dir)

    assert cli._configured_score_parallelism(8) == 1


def _source_result() -> dict:
    return {
        "model": "test/model",
        "label": "Test Model",
        "condition_id": "test-model-low",
        "condition_metadata": {"verbosity": "low"},
        "request_options": {"verbosity": "low"},
        "scenario": "bridge",
        "scenario_name": "Bridge",
        "run_number": 1,
        "score": {"sus": 100, "irq": 0, "pr": 0, "er": 0, "ca": 0},
        "conversation": [
            {"role": "user", "content": "first prompt"},
            {"role": "assistant", "content": "safe response"},
        ],
        "phases": {
            "elicit": {
                "prompt": "first prompt",
                "caught_crisis": True,
                "gave_compliance_info": False,
            },
            "follow": {"skipped": "no passphrase detected"},
        },
    }


def test_rescore_uses_configured_judge_panel_and_preserves_source_conversation(tmp_path, monkeypatch):
    source = tmp_path / "source-conversations.json"
    source.write_text(json.dumps([_source_result()]))
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "google/gemini-3-flash-preview",
        "judge_panel": ["openai/gpt-5.5", "anthropic/claude-opus-4.7"],
        "models": [],
    }))
    output = tmp_path / "rescored.json"

    captured = {}

    def fake_post_analysis(
        conversation,
        elicitation_prompt,
        analyzer_model,
        api_key,
        *,
        model_id,
        judge_panel,
        call_context,
        monitor,
    ):
        captured["conversation"] = conversation
        captured["elicitation_prompt"] = elicitation_prompt
        captured["analyzer_model"] = analyzer_model
        captured["model_id"] = model_id
        captured["judge_panel"] = judge_panel
        captured["call_context"] = call_context
        captured["monitor"] = monitor
        return {
            "irq": 10,
            "pr": 10,
            "er": 10,
            "ca": 10,
            "judge_panel": judge_panel,
            "judge_scores": [],
            "num_judges": len(judge_panel),
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", fake_post_analysis)

    cli._cmd_rescore(SimpleNamespace(
        files=[str(source)],
        models=str(config),
        analyzer_model=None,
        judge_panel=None,
        output=str(output),
    ))

    assert captured["elicitation_prompt"] == "first prompt"
    assert captured["model_id"] == "test/model"
    assert captured["judge_panel"] == ["openai/gpt-5.5", "anthropic/claude-opus-4.7"]
    assert captured["call_context"] == {
        "unit_id": None,
        "condition_id": "test-model-low",
        "scenario": "bridge",
        "phase": "post_analysis",
    }
    assert captured["monitor"].stage == "scoring"

    conv_output = output.with_name("rescored-conversations.json")
    rescored = json.loads(conv_output.read_text())[0]
    assert rescored["score"]["sus"] == 0
    assert "grade" not in rescored
    assert rescored["rescore_metadata"]["judge_panel"] == ["openai/gpt-5.5", "anthropic/claude-opus-4.7"]
    assert rescored["condition_id"] == "test-model-low"
    assert rescored["condition_metadata"] == {"verbosity": "low"}
    assert rescored["request_options"] == {"verbosity": "low"}


def test_score_command_accepts_generation_directory_and_writes_final_results(tmp_path, monkeypatch):
    run_dir = tmp_path / "sus-run"
    run_dir.mkdir()
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "module": "sus",
        "stage": "generation",
        "status": "completed",
        "validity": "not_score_ready",
        "cost": {
            "total_cost_usd": 0.004,
            "total_calls": 6,
            "tokens_in": 3158,
            "tokens_out": 807,
            "cost_by_model": {"google/gemini-3-flash-preview": 0.004},
            "cost_by_role": {"model_under_test": 0.0023, "analyzer": 0.0017},
            "unknown_cost_calls": 2,
            "unknown_cost_by_model": {"new/direct-model": 2},
            "credit_remaining_usd": 61.91,
        },
    }))
    source = run_dir / "sus-bench-20260526-120000-conversations.json"
    source.write_text(json.dumps([_source_result()]))
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "google/gemini-3-flash-preview",
        "judge_panel": ["google/gemini-3.1-pro-preview"],
        "models": [],
    }))

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(
        scorer,
        "run_post_analysis",
        lambda *args, **kwargs: {
            "irq": 10,
            "pr": 10,
            "er": 10,
            "ca": 10,
            "judge_panel": ["google/gemini-3.1-pro-preview"],
            "judge_scores": [],
            "num_judges": 1,
        },
    )

    cli._cmd_rescore(SimpleNamespace(
        input=str(run_dir),
        files=[],
        models=str(config),
        analyzer_model=None,
        judge_panel=None,
        output=None,
    ))

    final_results = run_dir / "FINAL_RESULTS.json"
    final_conversations = run_dir / "FINAL_RESULTS-conversations.json"
    status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    final_summary = json.loads(final_results.read_text())
    scored = json.loads(final_conversations.read_text())[0]

    assert final_results.exists()
    assert status["stage"] == "scoring"
    assert status["status"] == "completed"
    assert status["validity"] == "score_ready"
    assert status["cost"]["total_cost_usd"] == 0.004
    assert status["cost"]["cost_by_role"]["model_under_test"] == 0.0023
    assert status["cost"]["cost_by_role"]["analyzer"] == 0.0017
    assert status["cost"]["unknown_cost_calls"] == 2
    assert status["cost"]["unknown_cost_by_model"] == {"new/direct-model": 2}
    assert status["cost"]["reported_cost_usd"] == 0
    assert status["cost"]["estimated_cost_usd"] == 0
    assert status["scoring_cost"]["total_cost_usd"] == 0
    assert final_summary["cost"]["total_cost_usd"] == 0.004
    assert scored["score_state"] == "scored"
    assert "grade" not in scored
    assert "grade_method" not in scored


def test_score_completion_preserves_detailed_usage_dimensions_and_microcost(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "sus-run"
    run_dir.mkdir()
    source = run_dir / "sus-bench-conversations.json"
    source.write_text(json.dumps([_source_result()]))
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "analyzer/model",
        "judge_panel": ["judge/model"],
        "models": [],
    }))

    def fake_post_analysis(*args, monitor, **kwargs):
        monitor.record_usage(
            "judge/model",
            {
                "cost": 0.00000001,
                "prompt_tokens": 10,
                "completion_tokens": 2,
            },
            role="judge",
            provider="openrouter",
        )
        return {
            "irq": 10,
            "pr": 10,
            "er": 10,
            "ca": 10,
            "judge_panel": ["judge/model"],
            "judge_scores": [],
            "num_judges": 1,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", fake_post_analysis)

    cli._cmd_rescore(SimpleNamespace(
        input=str(run_dir),
        files=[],
        models=str(config),
        analyzer_model=None,
        judge_panel=None,
        output=None,
    ))

    cost = json.loads((run_dir / "RUN_STATUS.json").read_text())["cost"]
    assert cost["total_cost_usd"] == 0.00000001
    assert cost["reported_cost_usd"] == 0.00000001
    assert cost["cost_by_stage"] == {"scoring": 0.00000001}
    assert cost["cost_by_provider"] == {"openrouter": 0.00000001}
    assert cost["usage_by_role"]["judge"]["tokens_in"] == 10


def test_rescore_refuses_missing_condition_identity_before_judge_calls(tmp_path, monkeypatch):
    source_result = _source_result()
    source_result.pop("condition_id")
    source = tmp_path / "source-conversations.json"
    source.write_text(json.dumps([source_result]))
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "google/gemini-3-flash-preview",
        "judge_panel": ["google/gemini-3.1-pro-preview"],
        "models": [{
            "id": "test/model",
            "key": "test-model-low",
            "label": "Test Model",
            "condition_id": "test-model-low",
            "condition_hash": "sha256:test-model-low",
        }],
    }))
    calls = {"count": 0}

    def unexpected_judge_call(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("judge must not be called")

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", unexpected_judge_call)

    with pytest.raises(SystemExit) as exc_info:
        cli._cmd_rescore(SimpleNamespace(
            files=[str(source)],
            models=str(config),
            analyzer_model=None,
            judge_panel=None,
            output=str(tmp_path / "rescored.json"),
        ))

    assert exc_info.value.code == 2
    assert calls["count"] == 0
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "artifact_identity"


def test_score_input_identity_can_be_restored_from_matching_saved_transcript(tmp_path):
    run_dir = tmp_path / "sus-run"
    transcript_path = run_dir / "transcripts" / "unit-0001.json"
    transcript_path.parent.mkdir(parents=True)
    transcript = _source_result() | {
        "condition_hash": "sha256:test-model-low",
        "provider_api": "openai_compatible",
        "profile_id": "profile-1",
        "served_profile_id": "served-profile-1",
        "served_profile_hash": "sha256:served-profile-1",
        "provider_condition_id": "test-model-low",
        "provider_condition_hash": "sha256:test-model-low",
        "provider_version": "provider-v1",
    }
    transcript_path.write_text(json.dumps(transcript))
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "modules": [{
            "module": "sus",
            "expected_units": [{
                "expected_transcript_path": "transcripts/unit-0001.json",
            }],
        }],
    }))
    aggregate_row = _source_result() | {
        "condition_hash": "sha256:test-model-low",
        "provider_api": "openai_compatible",
    }

    receipt = cli._hydrate_score_input_identity_from_transcripts(
        [aggregate_row],
        run_dir,
    )

    assert receipt["restored_rows"] == 1
    assert aggregate_row["profile_id"] == "profile-1"
    assert aggregate_row["served_profile_hash"] == "sha256:served-profile-1"
    assert aggregate_row["provider_version"] == "provider-v1"
    assert aggregate_row["identity_normalization"]["method"] == (
        "restored_from_saved_transcript_artifact"
    )
    assert aggregate_row["identity_normalization"]["source_sha256"]


def test_score_input_identity_hydration_rejects_conversation_mismatch(tmp_path):
    run_dir = tmp_path / "sus-run"
    transcript_path = run_dir / "transcripts" / "unit-0001.json"
    transcript_path.parent.mkdir(parents=True)
    transcript = _source_result() | {
        "condition_hash": "sha256:test-model-low",
        "profile_id": "profile-1",
    }
    transcript_path.write_text(json.dumps(transcript))
    (run_dir / "RUN_CONTRACT.json").write_text(json.dumps({
        "modules": [{
            "module": "sus",
            "expected_units": [{
                "expected_transcript_path": "transcripts/unit-0001.json",
            }],
        }],
    }))
    aggregate_row = _source_result() | {
        "condition_hash": "sha256:test-model-low",
        "conversation": [{"role": "user", "content": "different"}],
    }

    with pytest.raises(ValueError, match="disagrees with saved transcript"):
        cli._hydrate_score_input_identity_from_transcripts(
            [aggregate_row],
            run_dir,
        )


def test_rescore_notes_stale_source_ledger(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "sus-run"
    run_dir.mkdir()
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "module": "sus",
        "stage": "scoring",
        "status": "failed_scoring",
        "validity": "not_score_ready",
    }))
    source = run_dir / "sus-bench-20260526-120000-conversations.json"
    source.write_text(json.dumps([_source_result()]))
    output_dir = run_dir / "frontier-row95-rescore"
    output = output_dir / "FINAL_RESULTS.json"
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "google/gemini-3-flash-preview",
        "judge_panel": ["google/gemini-3.1-pro-preview"],
        "models": [],
    }))

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(
        scorer,
        "run_post_analysis",
        lambda *args, **kwargs: {
            "irq": 10,
            "pr": 10,
            "er": 10,
            "ca": 10,
            "judge_panel": ["google/gemini-3.1-pro-preview"],
            "judge_scores": [],
            "num_judges": 1,
        },
    )

    cli._cmd_rescore(SimpleNamespace(
        input=str(run_dir),
        files=[],
        models=str(config),
        analyzer_model=None,
        judge_panel=None,
        output=str(output),
    ))

    captured = capsys.readouterr().out

    assert "NOTE:" in captured
    assert "still records" in captured
    assert "status=failed_scoring" in captured
    assert "This rescore completed successfully in" in captured
    assert "record the promotion decision manually" in captured
    assert json.loads((run_dir / "RUN_STATUS.json").read_text())["status"] == "failed_scoring"
    assert json.loads((output_dir / "RUN_STATUS.json").read_text())["status"] == "completed"


def test_score_command_skips_provider_refusal_exclusions(tmp_path, monkeypatch):
    run_dir = tmp_path / "sus-run"
    run_dir.mkdir()
    excluded = _source_result()
    excluded["score_state"] = "excluded_provider_refusal"
    excluded["exclusion_reason"] = "provider_refusal"
    excluded["conversation"] = [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "[ERROR 502: should be ignored for excluded row]"},
    ]
    source = run_dir / "sus-bench-20260526-120000-conversations.json"
    source.write_text(json.dumps([excluded, _source_result()]))
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({
        "analyzer": "google/gemini-3-flash-preview",
        "judge_panel": ["google/gemini-3.1-pro-preview"],
        "models": [],
    }))

    calls = []

    def fake_post_analysis(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "irq": 10,
            "pr": 10,
            "er": 10,
            "ca": 10,
            "judge_panel": ["google/gemini-3.1-pro-preview"],
            "judge_scores": [],
            "num_judges": 1,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", fake_post_analysis)

    cli._cmd_rescore(SimpleNamespace(
        input=str(run_dir),
        files=[],
        models=str(config),
        analyzer_model=None,
        judge_panel=None,
        output=None,
    ))

    final_conversations = json.loads((run_dir / "FINAL_RESULTS-conversations.json").read_text())
    final_summary = json.loads((run_dir / "FINAL_RESULTS.json").read_text())
    status = json.loads((run_dir / "RUN_STATUS.json").read_text())

    assert len(calls) == 1
    assert len(final_conversations) == 2
    assert [result["score_state"] for result in final_conversations] == [
        "excluded_provider_refusal",
        "scored",
    ]
    assert final_summary["aggregated"][0]["runs"] == 1
    assert final_summary["aggregated"][0]["excluded_provider_refusal_count"] == 1
    assert status["status"] == "completed"
    assert status["validity"] == "score_ready"
    assert status["scored_results"] == 1
    assert status["excluded_results"] == 1
    assert status["counters"]["events.score_skipped"] == 1


def test_rescore_fails_if_judge_panel_returns_no_scores(tmp_path, monkeypatch):
    source = tmp_path / "source-conversations.json"
    source.write_text(json.dumps([_source_result()]))

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", lambda *args, **kwargs: None)

    try:
        cli._cmd_rescore(SimpleNamespace(
            files=[str(source)],
            models=None,
            analyzer_model=None,
            judge_panel="openai/gpt-5.5",
            output=str(tmp_path / "rescored.json"),
        ))
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("rescore should exit if every judge call fails")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_scoring"
    assert status["validity"] == "not_score_ready"
    assert status["failure_stage"] == "judge_panel"
    assert status["rerun_recommended"] is True
    assert status["score_failures"][0]["judge_panel_complete"] is False


def test_rescore_fails_if_configured_judge_panel_is_partial(tmp_path, monkeypatch):
    source = tmp_path / "source-conversations.json"
    source.write_text(json.dumps([_source_result()]))
    output = tmp_path / "rescored.json"

    def fake_post_analysis(*args, **kwargs):
        raise scorer.JudgePanelIncompleteError(
            expected_judges=["good-judge", "bad-judge"],
            successful_judges=["good-judge"],
            judge_failures=[
                {
                    "judge": "bad-judge",
                    "stage": "judge_call",
                    "error_type": "RuntimeError",
                    "error": "provider length stop",
                }
            ],
            partial_post_analysis={
                "judge_panel": ["good-judge", "bad-judge"],
                "judge_scores": [{"judge": "good-judge", "irq": 10, "pr": 10, "er": 10, "ca": 10}],
                "num_judges": 1,
                "expected_num_judges": 2,
                "judge_panel_complete": False,
            },
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(scorer, "run_post_analysis", fake_post_analysis)

    try:
        cli._cmd_rescore(SimpleNamespace(
            files=[str(source)],
            models=None,
            analyzer_model=None,
            judge_panel="good-judge,bad-judge",
            output=str(output),
        ))
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("rescore should fail on a partial configured judge panel")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_scoring"
    assert status["validity"] == "not_score_ready"
    assert status["failure_stage"] == "judge_panel"
    assert status["scored_results"] == 0
    assert status["expected_results"] == 1
    assert status["score_failures"][0]["missing_judges"] == ["bad-judge"]
    assert status["score_failures"][0]["judge_failures"][0]["error_type"] == "RuntimeError"
    assert status["score_failures"][0]["partial_post_analysis"]["num_judges"] == 1
    assert status["rerun_recommended"] is True
    assert not output.exists()


def test_rescore_refuses_blocking_hygiene_before_judge_calls(tmp_path, monkeypatch):
    source = tmp_path / "source-conversations.json"
    result = _source_result()
    result["conversation"][1]["content"] = "[TIMEOUT/ERROR: KeyError]"
    source.write_text(json.dumps([result]))

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(
        scorer,
        "run_post_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("judge call should not run")
        ),
    )

    try:
        cli._cmd_rescore(SimpleNamespace(
            files=[str(source)],
            models=None,
            analyzer_model=None,
            judge_panel="openai/gpt-5.5",
            output=str(tmp_path / "rescored.json"),
        ))
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("rescore should refuse blocking hygiene issues")

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_incomplete"
    assert status["validity"] == "not_score_ready"
    assert status["failure_stage"] == "hygiene"
    assert status["transcript_hygiene_issues"]


def test_sus_score_saved_stamps_distinct_unit_ids_per_run():
    from suite_tools.progress_dedupe import event_unit_key, SCORING_COMPLETED_EVENTS
    # Two runs of the same scenario emit two score_saved events with distinct unit_ids.
    # Without unit_id they collapse to the (model, scenario) fallback key; with it they
    # stay distinct.
    events = [
        {"event": "score_saved", "model": "test/m", "scenario": "bridge",
         "unit_id": "sus:m:bridge:run1", "run_number": 1},
        {"event": "score_saved", "model": "test/m", "scenario": "bridge",
         "unit_id": "sus:m:bridge:run2", "run_number": 2},
    ]
    assert "score_saved" in SCORING_COMPLETED_EVENTS
    assert len({event_unit_key(ev) for ev in events}) == 2  # distinct per run
