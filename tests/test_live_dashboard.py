import json
import socket
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import suite_tools.live_dashboard as live_dashboard
from suite_tools.dashboard_store import DashboardStore
from suite_tools.live_dashboard import (
    run_server,
    DashboardHandler,
    DashboardOptions,
    _judge_summary,
    _model_condition_summary,
    _tail_lines,
    build_dashboard_data,
    render_html,
)
from suite_tools.paid_call_lease import (
    LEASE_EVENTS_FILENAME,
    paid_call_lease,
    record_rate_limit_cooldown,
    set_paid_call_policy,
)
from suite_tools.run_contract import build_provenance_identity


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _append_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def _handler_for_results_root(results_root):
    handler = object.__new__(DashboardHandler)
    handler.options = DashboardOptions(results_root=results_root)
    return handler


def test_tail_lines_returns_last_n_lines(tmp_path):
    long_path = tmp_path / "long.log"
    long_path.write_text("".join(f"line{index}\n" for index in range(10000)))

    assert _tail_lines(long_path, 5) == [
        "line9995",
        "line9996",
        "line9997",
        "line9998",
        "line9999",
    ]

    short_path = tmp_path / "short.log"
    short_path.write_text("alpha\nbeta\ngamma\n")

    assert _tail_lines(short_path, 10) == ["alpha", "beta", "gamma"]

    unterminated_path = tmp_path / "unterminated.log"
    unterminated_path.write_text("first\nsecond\nlast")

    assert _tail_lines(unterminated_path, 2) == ["second", "last"]

    empty_path = tmp_path / "empty.log"
    empty_path.write_text("")

    assert _tail_lines(empty_path, 5) == []


def test_runs_cache_serves_cached_payload_within_ttl(tmp_path, monkeypatch):
    DashboardHandler._runs_cache.clear()
    if hasattr(DashboardHandler, "_runs_builds_in_progress"):
        DashboardHandler._runs_builds_in_progress.clear()
    calls = []

    def fake_build(options):
        calls.append(options.results_root)
        return {"schema_version": "test", "call_count": len(calls)}

    monkeypatch.setattr(live_dashboard, "build_dashboard_data", fake_build)
    handler = _handler_for_results_root(tmp_path)

    first = DashboardHandler._cached_runs_payload(handler)
    second = DashboardHandler._cached_runs_payload(handler)

    assert first == second
    assert json.loads(first)["call_count"] == 1
    assert calls == [tmp_path]


def test_runs_cache_does_not_store_global_build_errors(tmp_path, monkeypatch):
    DashboardHandler._runs_cache.clear()
    DashboardHandler._runs_builds_in_progress.clear()
    calls = []

    def broken_build(options):
        calls.append(options.results_root)
        raise ValueError("malformed ledger")

    monkeypatch.setattr(live_dashboard, "build_dashboard_data", broken_build)
    handler = _handler_for_results_root(tmp_path)

    first = json.loads(DashboardHandler._cached_runs_payload(handler))
    second = json.loads(DashboardHandler._cached_runs_payload(handler))

    assert first["error"] == "malformed ledger"
    assert second["error"] == "malformed ledger"
    assert calls == [tmp_path, tmp_path]


def test_dashboard_source_revision_tracks_json_ledgers_not_unrelated_files(tmp_path):
    status_path = tmp_path / "run-1" / "RUN_STATUS.json"
    _write_json(status_path, {"status": "prepared"})
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("operator note")
    first = live_dashboard._dashboard_source_revision(tmp_path)

    notes_path.write_text("revised operator note")
    assert live_dashboard._dashboard_source_revision(tmp_path) == first

    _write_json(status_path, {"status": "running", "active_units": 1})
    assert live_dashboard._dashboard_source_revision(tmp_path) != first


def test_dashboard_source_revision_advances_for_time_based_watchdogs(tmp_path, monkeypatch):
    _write_json(tmp_path / "run-1" / "RUN_STATUS.json", {"status": "running"})
    monkeypatch.setattr(live_dashboard.time, "time", lambda: 100.0)
    first = live_dashboard._dashboard_source_revision(tmp_path)

    monkeypatch.setattr(
        live_dashboard.time,
        "time",
        lambda: 100.0 + live_dashboard.DASHBOARD_WATCHDOG_REVISION_SECONDS,
    )

    assert live_dashboard._dashboard_source_revision(tmp_path) != first


def test_cost_summary_preserves_reported_estimated_and_legacy_breakdown():
    summary = live_dashboard._cost_summary({
        "cost": {
            "total_cost_usd": 12.345678,
            "reported_cost_usd": 2.1,
            "estimated_cost_usd": 3.2,
            "total_calls": 5,
            "tokens_in": 100,
            "tokens_out": 50,
            "thinking_tokens_out": 25,
            "billable_tokens_out": 75,
        }
    })

    assert summary == {
        "total_cost_usd": 12.3457,
        "reported_cost_usd": 2.1,
        "estimated_cost_usd": 3.2,
        "unclassified_cost_usd": 7.0457,
        "total_calls": 5,
        "tokens": 150,
        "tokens_in": 100,
        "tokens_out": 50,
        "thinking_tokens_out": 25,
        "billable_tokens_out": 75,
        "billable_tokens": 175,
        "credit_remaining_usd": None,
    }


def test_dashboard_revision_ignores_lease_event_churn(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    lease_dir = tmp_path / "leases"
    _write_json(results_root / "run-1" / "RUN_STATUS.json", {"status": "running"})
    _write_json(lease_dir / live_dashboard.LEASE_STATUS_FILENAME, {"active_count": 0})
    monkeypatch.setattr(live_dashboard, "default_lease_dir", lambda: lease_dir)
    events_path = lease_dir / LEASE_EVENTS_FILENAME
    events_path.write_text('{"event":"lease_acquired"}\n')
    first = live_dashboard._dashboard_source_revision(results_root)

    events_path.write_text('{"event":"lease_released"}\n')

    assert live_dashboard._dashboard_source_revision(results_root) == first


def test_dashboard_isolates_a_malformed_status_ledger(tmp_path):
    _write_json(
        tmp_path / "valid-run" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "score_ready",
        },
    )
    malformed_path = tmp_path / "broken-run" / "sus" / "RUN_STATUS.json"
    _write_json(
        malformed_path,
        {
            "module": "sus",
            "stage": "generation",
            "status": ["running"],
            "validity": "not_score_ready",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 1
    assert data["groups"][0]["run_id"] == "valid-run"
    assert data["ledger_warnings"] == [
        {
            "kind": "run_status",
            "path": live_dashboard._relative(malformed_path, live_dashboard.REPO_ROOT),
            "error": "status must be a string",
        }
    ]


def test_dashboard_isolates_a_malformed_run_plan(tmp_path):
    _write_json(
        tmp_path / "valid-run" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "score_ready",
        },
    )
    malformed_path = tmp_path / "broken-run" / "RUN_PLAN.json"
    _write_json(
        malformed_path,
        {
            "schema_version": "benchmark-run-plan-v1",
            "run_id": "broken-run",
            "modules": [{"module": "sus", "expected_units": "twenty"}],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 1
    assert data["plans"] == []
    assert data["ledger_warnings"] == [
        {
            "kind": "run_plan",
            "path": live_dashboard._relative(malformed_path, live_dashboard.REPO_ROOT),
            "error": "expected_units must be a non-negative integer or collection",
        }
    ]


def test_dashboard_isolates_a_malformed_run_contract(tmp_path):
    _write_json(
        tmp_path / "valid-run" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "score_ready",
        },
    )
    malformed_path = tmp_path / "broken-run" / "sus" / "RUN_CONTRACT.json"
    _write_json(
        malformed_path,
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "broken-run",
            "modules": [{"module": "sus", "expected_units": 5}],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 1
    assert data["contracts"] == []
    assert data["ledger_warnings"] == [
        {
            "kind": "run_contract",
            "path": live_dashboard._relative(malformed_path, live_dashboard.REPO_ROOT),
            "error": "expected_units must be a list",
        }
    ]


def test_dashboard_normalizes_numeric_cost_fields_from_ledgers(tmp_path):
    _write_json(
        tmp_path / "run-1" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "score_ready",
            "cost": {
                "total_cost_usd": "1.25",
                "total_calls": "2",
                "tokens_in": "10",
                "tokens_out": "20",
                "credit_remaining_usd": "8.75",
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    cost = data["groups"][0]["modules"][0]["cost"]
    assert cost == {
        "total_cost_usd": 1.25,
        "reported_cost_usd": 0,
        "estimated_cost_usd": 0,
        "unclassified_cost_usd": 1.25,
        "total_calls": 2,
        "tokens": 30,
        "tokens_in": 10,
        "tokens_out": 20,
        "thinking_tokens_out": 0,
        "billable_tokens_out": 20,
        "billable_tokens": 30,
        "credit_remaining_usd": 8.75,
    }


def test_contract_summaries_reuse_unchanged_files(tmp_path, monkeypatch):
    contract_path = tmp_path / "run-1" / "aita" / "RUN_CONTRACT.json"
    _write_json(contract_path, {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "run-1",
        "modules": [{
            "module": "aita",
            "output_dir": ".",
            "expected_units": [{"unit_id": "u1", "expected_transcript_path": "u1.json"}],
        }],
    })
    original = live_dashboard.summarize_contract
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0].get("run_id"))
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "summarize_contract", counted)

    first = live_dashboard._load_contract_summaries(tmp_path)
    second = live_dashboard._load_contract_summaries(tmp_path)

    assert first == second
    assert calls == ["run-1"]


def test_summary_build_reuses_compact_contract_headers(tmp_path, monkeypatch):
    contract_path = tmp_path / "run-1" / "aita" / "RUN_CONTRACT.json"
    _write_json(
        contract_path,
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "identity": {
                "sample_spec": {
                    "items": {str(index): {"private": "x" * 100} for index in range(20)},
                    "scenario_ids": ["scenario-a"],
                    "runs": 3,
                    "dataset_manifest": {"private": "not needed by dashboard"},
                }
            },
            "modules": [
                {
                    "module": "aita",
                    "expected_units": [{"unit_id": "u1"}],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "run-1" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "prepared",
            "validity": "not_score_ready",
        },
    )
    live_dashboard._CONTRACT_HEADER_CACHE.clear()
    original = live_dashboard.load_run_contract
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(live_dashboard, "load_run_contract", counted)

    first = build_dashboard_data(DashboardOptions(results_root=tmp_path, summary_only=True))
    second = build_dashboard_data(DashboardOptions(results_root=tmp_path, summary_only=True))

    assert first["contracts"] == second["contracts"]
    assert calls == [contract_path]
    assert first["contracts"][0]["identity"]["sample_spec"] == {
        "scenario_ids": ["scenario-a"],
        "runs": 3,
        "item_count": 20,
    }


def test_suite_inventory_reuses_unchanged_model_config(tmp_path, monkeypatch):
    config_path = tmp_path / "suite_models.yaml"
    config_path.write_text("schema_version: test\n")
    config = {
        "schema_version": "test",
        "models": {},
        "model_groups": {},
        "judge_sets": {},
    }
    calls = []

    def counted(_path):
        calls.append(_path)
        return config

    monkeypatch.setattr(live_dashboard, "DEFAULT_SUITE_CONFIG", config_path)
    monkeypatch.setattr(live_dashboard, "load_suite_config", counted)
    monkeypatch.setattr(live_dashboard, "validate_suite_config", lambda _config: [])
    live_dashboard._SUITE_INVENTORY_CACHE.clear()

    first = live_dashboard._suite_inventory()
    second = live_dashboard._suite_inventory()
    config_path.write_text("schema_version: revised\n")
    third = live_dashboard._suite_inventory()

    assert first == second == third
    assert calls == [config_path, config_path]


def test_contract_summary_cache_tracks_nested_artifact_directories(tmp_path, monkeypatch):
    contract_path = tmp_path / "run-1" / "aita" / "RUN_CONTRACT.json"
    artifact_path = contract_path.parent / "nested" / "u1.json"
    artifact_path.parent.mkdir(parents=True)
    _write_json(
        contract_path,
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "modules": [
                {
                    "module": "aita",
                    "output_dir": str(contract_path.parent),
                    "expected_units": [
                        {
                            "unit_id": "u1",
                            "expected_transcript_path": "nested/u1.json",
                        }
                    ],
                }
            ],
        },
    )
    original = live_dashboard.summarize_contract
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0].get("run_id"))
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "summarize_contract", counted)
    first = live_dashboard._load_contract_summaries(tmp_path)
    _write_json(artifact_path, {"completed": True})
    second = live_dashboard._load_contract_summaries(tmp_path)

    assert first[0]["complete_units"] == 0
    assert second[0]["complete_units"] == 1
    assert calls == ["run-1", "run-1"]


def test_event_loaders_reuse_unchanged_ledger(tmp_path, monkeypatch):
    events_path = tmp_path / "RUN_EVENTS.jsonl"
    _append_events(events_path, [
        {"event": "turn_saved", "turn": 1, "error": "Bearer abcdefghijklmnop"},
        {"event": "score_saved", "dimension": "integrity"},
    ])
    original = live_dashboard.json.loads
    calls = []

    def counted(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(live_dashboard.json, "loads", counted)

    first = live_dashboard._load_events_filtered(events_path, {"turn_saved", "score_saved"})
    second = live_dashboard._load_events_filtered(events_path, {"turn_saved", "score_saved"})

    assert first == second
    assert len(calls) == 2
    assert first[0]["error"] == "Bearer <redacted>"


def test_transcript_evidence_reuses_unchanged_artifact(tmp_path, monkeypatch):
    transcript_path = tmp_path / "conversation.json"
    _write_json(transcript_path, {
        "model": "model/a",
        "turns": [{"turn": 1, "user_message": "hello", "model_response": "hi"}],
    })
    original = live_dashboard.json.loads
    calls = []

    def counted(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(live_dashboard.json, "loads", counted)
    kwargs = {
        "group": "run-1",
        "module_path": "aita",
        "status": {"module": "aita", "stage": "generation", "status": "completed", "validity": "not_score_ready"},
        "fallback_timestamp": "2026-01-01T00:00:00+00:00",
    }

    first = live_dashboard._evidence_items_from_transcript_file(transcript_path, **kwargs)
    second = live_dashboard._evidence_items_from_transcript_file(transcript_path, **kwargs)

    assert first == second
    assert len(calls) == 1


def test_live_sus_transcript_renders_provider_refusal_outcome_event(tmp_path):
    transcript_path = tmp_path / "sus-refusal.json"
    _write_json(
        transcript_path,
        {
            "module": "sus",
            "model": "test/model",
            "turns": [],
            "turn_outcomes": [
                {
                    "type": "provider_refusal",
                    "stop_reason": "refusal",
                    "timestamp": "2026-07-14T12:00:00+00:00",
                    "turn": 1,
                }
            ],
        },
    )

    items = live_dashboard._build_evidence_items_from_transcript_file(
        transcript_path,
        group="sus-refusal",
        module_path="sus",
        status={"module": "sus", "status": "completed", "validity": "not_score_ready"},
        fallback_timestamp="2026-07-14T12:00:01+00:00",
        limit=4,
    )

    assert len(items) == 1
    assert items[0]["kind"] == "turn_outcome"
    assert items[0]["problem"] == "Provider refusal (stop reason: refusal)"


def test_transcript_evidence_cache_replaces_stale_file_versions(tmp_path):
    live_dashboard._EVIDENCE_ARTIFACT_CACHE.clear()
    transcript_path = tmp_path / "conversation.json"
    kwargs = {
        "group": "run-1",
        "module_path": "aita",
        "status": {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
        },
        "fallback_timestamp": "2026-01-01T00:00:00+00:00",
    }
    _write_json(
        transcript_path,
        {"turns": [{"turn": 1, "user_message": "hello", "model_response": "short"}]},
    )
    live_dashboard._evidence_items_from_transcript_file(transcript_path, **kwargs)

    _write_json(
        transcript_path,
        {"turns": [{"turn": 1, "user_message": "hello", "model_response": "a longer revised response"}]},
    )
    live_dashboard._evidence_items_from_transcript_file(transcript_path, **kwargs)

    resolved = str(transcript_path.resolve())
    matching_keys = [
        key
        for key in live_dashboard._EVIDENCE_ARTIFACT_CACHE
        if key[0] == "transcript" and key[1] == resolved
    ]
    assert len(matching_keys) == 1


def test_module_evidence_skips_unchanged_directory_rescan(tmp_path, monkeypatch):
    transcript_path = tmp_path / "conversation.json"
    _write_json(transcript_path, {
        "model": "model/a",
        "turns": [{"turn": 1, "user_message": "hello", "model_response": "hi"}],
    })
    original = live_dashboard._conversation_path_candidates
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "_conversation_path_candidates", counted)
    kwargs = {
        "group": "run-1",
        "module_path": "aita",
        "status": {"module": "aita", "stage": "generation", "status": "completed", "validity": "not_score_ready", "updated_at": "2026-01-01T00:00:00+00:00"},
        "events": [{"event": "stage_completed", "timestamp": "2026-01-01T00:00:00+00:00"}],
        "output_dir": tmp_path,
    }

    first = live_dashboard._evidence_items_from_module(**kwargs)
    second = live_dashboard._evidence_items_from_module(**kwargs)

    assert first == second
    assert calls == [tmp_path]


def test_transcript_preview_skips_unchanged_directory_rescan(tmp_path, monkeypatch):
    _write_json(tmp_path / "conversation.json", {
        "turns": [{"turn": 1, "user_message": "hello", "model_response": "hi"}],
    })
    original = live_dashboard._conversation_path_candidates
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "_conversation_path_candidates", counted)
    status = {"module": "aita", "status": "completed", "updated_at": "2026-01-01T00:00:00+00:00"}

    first = live_dashboard._transcript_preview([], tmp_path, status)
    second = live_dashboard._transcript_preview([], tmp_path, status)

    assert first == second
    assert calls == [tmp_path]


def test_summary_only_build_does_not_walk_contract_units_or_transcripts(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-1" / "aita"
    _write_json(run_dir / "RUN_CONTRACT.json", {
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "run-1",
        "expected_models": [{"key": "m", "model_id": "model/a"}],
        "modules": [{
            "module": "aita",
            "stage": "generation",
            "expected_units": [{"unit_id": "u1", "expected_transcript_path": "u1.json"}],
        }],
    })
    _write_json(run_dir / "RUN_STATUS.json", {
        "module": "aita", "stage": "generation", "status": "completed",
        "validity": "not_score_ready", "updated_at": "2026-01-01T00:00:00+00:00",
    })
    monkeypatch.setattr(
        live_dashboard,
        "summarize_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full contract walk")),
    )
    monkeypatch.setattr(
        live_dashboard,
        "_evidence_items_from_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("transcript walk")),
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path, summary_only=True))

    assert data["summary"]["contract_count"] == 1
    assert data["groups"][0]["run_id"] == "run-1"
    assert data["evidence_feed"] == []


def test_scoped_evidence_detail_only_walks_selected_run(tmp_path, monkeypatch):
    for run_id in ("run-1", "run-2"):
        run_dir = tmp_path / run_id / "aita"
        _write_json(run_dir / "RUN_STATUS.json", {
            "module": "aita", "stage": "generation", "status": "completed",
            "validity": "not_score_ready", "updated_at": "2026-01-01T00:00:00+00:00",
        })
        _append_events(run_dir / "RUN_EVENTS.jsonl", [{
            "timestamp": "2026-01-01T00:00:01+00:00", "event": "stage_completed",
        }])
    visited = []
    original = live_dashboard._evidence_items_from_module

    def counted(*args, **kwargs):
        visited.append(kwargs["output_dir"])
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "_evidence_items_from_module", counted)
    payload = live_dashboard._build_dashboard_evidence_detail(
        {"summary": {"latest_run_id": "run-2"}},
        results_root=tmp_path,
        scope="latest",
        stage="all",
        content="all",
        window="100",
        module="",
    )

    assert visited == [tmp_path / "run-2" / "aita"]
    assert payload["resolved_scope"] == "run-2"
    assert {item["group"] for item in payload["items"]} == {"run-2"}


def test_scoped_contract_detail_only_resolves_selected_run(tmp_path, monkeypatch):
    for run_id in ("run-1", "run-2"):
        _write_json(tmp_path / run_id / "aita" / "RUN_CONTRACT.json", {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": run_id,
            "modules": [],
        })
    visited = []
    original = live_dashboard.summarize_contract

    def counted(contract, **kwargs):
        visited.append(kwargs["contract_path"])
        return original(contract, **kwargs)

    monkeypatch.setattr(live_dashboard, "summarize_contract", counted)
    payload = live_dashboard._build_dashboard_contract_detail(
        {"summary": {"latest_run_id": "run-2"}},
        results_root=tmp_path,
        scope="latest",
    )

    assert visited == [tmp_path / "run-2" / "aita" / "RUN_CONTRACT.json"]
    assert payload["resolved_scope"] == "run-2"
    assert len(payload["contracts"]) == 1
    assert payload["contracts"][0]["path_group_id"] == "run-2"


def test_runs_cache_rebuilds_when_source_revision_changes(tmp_path, monkeypatch):
    DashboardHandler._runs_cache.clear()
    DashboardHandler._runs_builds_in_progress.clear()
    calls = []

    def fake_build(options):
        calls.append(options.results_root)
        return {"schema_version": "test", "call_count": len(calls)}

    monkeypatch.setattr(live_dashboard, "build_dashboard_data", fake_build)
    handler = _handler_for_results_root(tmp_path)
    first = DashboardHandler._cached_runs_payload(handler)
    _write_json(tmp_path / "run-1" / "RUN_STATUS.json", {"status": "running"})
    second = DashboardHandler._cached_runs_payload(handler)

    assert json.loads(first)["call_count"] == 1
    assert json.loads(second)["call_count"] == 2


def test_dashboard_summary_payload_strips_heavy_detail_and_stays_bounded():
    large_text = "saved transcript content " * 50000
    contract = {
        "run_id": "run-1",
        "path_group_id": "run-1",
        "expected_units": 4,
        "complete_units": 3,
        "attention": True,
        "control": {"active": True},
        "identity": {"private-sized-record": large_text},
    }
    data = {
        "schema_version": "test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "groups": [
            {
                "run_id": "run-1",
                "contracts": [contract],
                "plans": [{"large": large_text}],
                "schedulers": [{"large": large_text}],
                "modules": [
                    {
                        "module": "aita",
                        "recent_events": [{"content": large_text}],
                        "event_model_progress": {"large": large_text},
                        "latest_transcript": {"model_response": "bounded preview"},
                    }
                ],
            }
        ],
        "contracts": [contract],
        "plans": [{"large": large_text}],
        "schedulers": [{"large": large_text}],
        "evidence_feed": [{"model_response": large_text}],
        "flow": {
            "lanes": [
                {
                    "id": "generating",
                    "items": [
                        {
                            "run_id": "run-1",
                            "latest_transcript": {"model_response": large_text},
                            "latest_model_response": large_text,
                            "latest_user_message": large_text,
                            "execute_command": large_text,
                        }
                    ],
                }
            ]
        },
        "operational_queue": {"stages": []},
        "summary": {"latest_run_id": "run-1"},
    }

    summary = live_dashboard._dashboard_summary_data(data)
    payload = json.dumps(summary, separators=(",", ":")).encode()

    assert len(payload) < 1_000_000
    assert summary["contracts"] == []
    assert summary["evidence_feed"] == []
    assert summary["groups"][0]["contract_summary"] == {
        "count": 1,
        "attention_count": 1,
        "expected_units": 4,
        "complete_units": 3,
        "active_control_count": 1,
    }
    assert "recent_events" not in summary["groups"][0]["modules"][0]
    assert "latest_transcript" not in summary["flow"]["lanes"][0]["items"][0]


def test_dashboard_summary_defaults_to_latest_and_keeps_a_small_run_index():
    data = {
        "groups": [
            {"run_id": "run-new", "updated_at": "2", "modules": [{"module": "aita"}]},
            {"run_id": "run-old", "updated_at": "1", "modules": [{"module": "sus"}]},
        ],
        "contracts": [],
        "plans": [],
        "schedulers": [],
        "evidence_feed": [],
        "latest_events": [
            {"group": "run-new", "event": "new"},
            {"group": "run-old", "event": "old"},
        ],
        "flow": {
            "lanes": [
                {
                    "id": "generating",
                    "items": [
                        {"run_id": "run-new", "title": "new"},
                        {"run_id": "run-old", "title": "old"},
                    ],
                }
            ]
        },
        "operational_queue": {
            "stages": [
                {
                    "id": "score_ready",
                    "items": [
                        {
                            "run_id": "run-new",
                            "title": "AITA",
                            "expected_units": 4,
                            "completed_units": 4,
                            "generation_expected_units": 8,
                            "generated_units": 8,
                            "expected_score_units": 4,
                            "scored_units": 3,
                            "excluded_score_units": 1,
                            "judge_calls_expected": 12,
                            "judge_calls_completed": 9,
                            "units": 4,
                        },
                        {
                            "run_id": "run-old",
                            "title": "SUS",
                            "expected_units": 20,
                            "completed_units": 20,
                            "units": 20,
                        },
                    ],
                }
            ],
            "total_units": 24,
            "generated_units": 24,
            "active_units": 0,
            "attention_units": 0,
        },
        "summary": {"latest_run_id": "run-new"},
    }

    summary = live_dashboard._dashboard_summary_data(data, scope="latest")

    assert [group["run_id"] for group in summary["groups"]] == ["run-new"]
    assert [group["run_id"] for group in summary["run_index"]] == ["run-new", "run-old"]
    assert [item["run_id"] for item in summary["flow"]["lanes"][0]["items"]] == ["run-new"]
    assert [event["group"] for event in summary["latest_events"]] == ["run-new"]
    assert summary["operational_queue"]["total_units"] == 8
    assert summary["operational_queue"]["generated_units"] == 8
    scoped_stage = summary["operational_queue"]["stages"][0]
    assert scoped_stage["generation_expected_units"] == 8
    assert scoped_stage["generated_units"] == 8
    assert scoped_stage["expected_score_units"] == 4
    assert scoped_stage["scored_units"] == 3
    assert scoped_stage["excluded_score_units"] == 1
    assert scoped_stage["judge_calls_expected"] == 12
    assert scoped_stage["judge_calls_completed"] == 9
    assert summary["operational_queue"]["generation_expected_units"] == 8
    assert summary["operational_queue"]["generation_completed_units"] == 8
    assert summary["operational_queue"]["score_expected_units"] == 4
    assert summary["operational_queue"]["score_completed_units"] == 4
    assert summary["operational_queue"]["judge_calls_expected"] == 12
    assert summary["operational_queue"]["judge_calls_completed"] == 9
    assert summary["operational_queue"]["stages"][0]["count"] == 1
    assert summary["operational_queue"]["stages"][0]["group_count"] == 1


def test_dashboard_detail_payloads_honor_latest_explicit_and_all_scopes():
    contracts = [
        {"run_id": "runtime-a", "path_group_id": "run-a", "expected_units": 2, "complete_units": 2},
        {"run_id": "runtime-b", "path_group_id": "run-b", "expected_units": 3, "complete_units": 1},
    ]
    evidence = [
        {"kind": "turn_pair", "group": "run-a", "module_path": "aita", "stage": "generation", "timestamp": "1"},
        {"kind": "event", "group": "run-b", "module_path": "sus", "stage": "scoring", "timestamp": "2"},
        {"kind": "turn_pair", "group": "run-b", "module_path": "sus", "stage": "generation", "timestamp": "3"},
    ]
    data = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"latest_run_id": "run-b"},
        "contracts": contracts,
        "evidence_feed": evidence,
    }

    latest = live_dashboard._dashboard_contract_payload(data, scope="latest")
    explicit = live_dashboard._dashboard_contract_payload(data, scope="run-a")
    all_runs = live_dashboard._dashboard_contract_payload(data, scope="all")

    assert [item["path_group_id"] for item in latest["contracts"]] == ["run-b"]
    assert [item["path_group_id"] for item in explicit["contracts"]] == ["run-a"]
    assert len(all_runs["contracts"]) == 2
    assert latest["summary"]["expected_units"] == 3

    evidence_latest = live_dashboard._dashboard_evidence_payload(
        data,
        scope="latest",
        stage="generating",
        content="text",
        window="25",
        module="sus",
    )
    evidence_all = live_dashboard._dashboard_evidence_payload(data, scope="all", window="all")

    assert evidence_latest["resolved_scope"] == "run-b"
    assert evidence_latest["items"] == [evidence[2]]
    assert evidence_latest["total_count"] == 1
    assert len(evidence_all["items"]) == 3


def test_dashboard_contract_progress_uses_completed_module_ledgers():
    contracts = [
        {
            "run_id": "run-1",
            "complete_units": 1,
            "expected_units": 4,
            "missing_units": 3,
            "progress_percent": 25,
            "modules": [
                {"module": "epistemic", "complete_units": 1, "expected_units": 4}
            ],
        }
    ]
    modules = [
        {
            "module": "epis",
            "module_path": "epis",
            "status": "completed",
            "validity": "not_score_ready",
            "progress": {"conversations_completed": 4},
        }
    ]

    reconciled = live_dashboard._reconcile_contract_progress_from_modules(contracts, modules)

    assert reconciled[0]["complete_units"] == 4
    assert reconciled[0]["artifact_complete_units"] == 1
    assert reconciled[0]["missing_units"] == 0
    assert reconciled[0]["progress_percent"] == 100
    assert reconciled[0]["ledger_progress_reconciled"] is True
    assert reconciled[0]["modules"][0]["artifact_complete_units"] == 1


def test_disposition_write_invalidates_runs_cache(tmp_path):
    DashboardHandler._runs_cache.clear()
    if hasattr(DashboardHandler, "_runs_builds_in_progress"):
        DashboardHandler._runs_builds_in_progress.clear()
    run_dir = tmp_path / "run-1" / "aita"
    status_path = run_dir / "RUN_STATUS.json"
    _write_json(
        status_path,
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
        },
    )
    cache_key = live_dashboard._runs_cache_key(tmp_path)
    DashboardHandler._runs_cache[cache_key] = (
        live_dashboard.time.monotonic(),
        live_dashboard._dashboard_source_revision(tmp_path),
        {},
        b"cached",
    )
    handler = _handler_for_results_root(tmp_path)
    sent = []
    handler._read_json_body = lambda: {
        "status_path": str(status_path),
        "disposition": "rejected_from_analysis",
        "reason": "operator rejected diagnostic run",
    }
    handler._send_json = lambda payload, status=200: sent.append((payload, status))

    DashboardHandler._write_disposition(handler)

    assert sent[0][0]["ok"] is True
    assert cache_key not in DashboardHandler._runs_cache


def _write_running_group_with_newer_prepared_contract(tmp_path):
    running_dir = tmp_path / "older-running" / "sus"
    transcript_path = running_dir / "gemini-flash_bridge_heights_run1.json"
    _write_json(
        transcript_path,
        {
            "model": "gemini-flash",
            "scenario": "bridge_heights",
            "conversation": [
                {"role": "user", "content": "initial prompt"},
                {"role": "assistant", "content": "running response"},
            ],
        },
    )
    _write_json(
        running_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-ledger-v1",
            "module": "sus",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": "2099-01-01T00:00:00+00:00",
            "updated_at": "2099-01-01T00:00:00+00:00",
            "metadata": {"models": ["gemini-flash"]},
        },
    )
    _append_events(
        running_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2099-01-01T00:00:10+00:00",
                "event": "conversation_started",
                "planned_turns": 5,
                "transcript_path": str(transcript_path),
            },
            {
                "timestamp": "2099-01-01T00:00:20+00:00",
                "event": "turn_saved",
                "turn": 1,
                "transcript_path": str(transcript_path),
            },
        ],
    )
    prepared_dir = tmp_path / "newer-prepared"
    _write_json(
        prepared_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "newer-prepared",
            "created_at": "2099-02-01T00:00:00+00:00",
            "contract_scope": "run_group",
            "modules": [
                {
                    "module": "sus",
                    "stage": "run",
                    "output_dir": "sus",
                    "expected_units": [
                        {"unit_id": "sus:gemini-flash:bridge_heights:run1"}
                    ],
                }
            ],
        },
    )


def test_model_summary_treats_openrouter_base_url_as_raw_openrouter():
    summary = _model_condition_summary({
        "expected_models": [
            {
                "key": "gemini-flash",
                "model_id": "google/gemini-3-flash-preview",
                "endpoint": "https://openrouter.ai/api/v1",
            }
        ]
    })

    assert summary["label"] == "Raw OpenRouter · 1 model"


def test_model_summary_reports_native_provider_and_effort_conditions():
    summary = _model_condition_summary({
        "expected_models": [
            {
                "key": "claude-sonnet-5-native-low-128k",
                "model_id": "claude-sonnet-5",
                "endpoint": "anthropic_native",
            },
            {
                "key": "claude-sonnet-5-native-high-128k",
                "model_id": "claude-sonnet-5",
                "endpoint": "anthropic_native",
            },
        ]
    })

    assert summary["label"] == "Anthropic native · 2 conditions"


def test_model_summary_reports_openai_responses_conditions():
    summary = _model_condition_summary({
        "expected_models": [
            {
                "key": "gpt-5-6-sol-native-low",
                "model_id": "gpt-5.6-sol",
                "endpoint": "openai_responses",
            },
            {
                "key": "gpt-5-6-sol-native-high",
                "model_id": "gpt-5.6-sol",
                "endpoint": "openai_responses",
            },
        ]
    })

    assert summary["label"] == "OpenAI Responses · 2 conditions"


def test_judge_summary_reports_primary_judge():
    assert _judge_summary({
        "expected_judges": [{"role": "primary", "model_id": "google/gemini-3.1-pro-preview"}]
    }) == "primary judge"


def test_render_html_uses_waiting_state_for_fetch_failures():
    html = render_html(csrf_token="test-csrf-token")
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()

    assert "Anti-sycophancy Benchmark Suite" in html
    assert "Live Benchmark Dashboard" in html
    assert 'data-csrf-token="test-csrf-token"' in html
    assert "Dashboard refresh failed" not in html
    assert "Dashboard refresh failed" not in dashboard_js
    assert "Waiting for dashboard data" in dashboard_js


def test_event_progress_treats_failed_rate_limited_as_terminal():
    progress = live_dashboard._event_progress([], {"status": "failed_rate_limited"})

    assert progress["percent"] == 100


def test_event_progress_treats_failed_timeout_as_terminal():
    progress = live_dashboard._event_progress([], {"status": "failed_timeout"})

    assert progress["percent"] == 100


def test_build_dashboard_data_surfaces_paid_call_leases(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.delenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", raising=False)
    monkeypatch.delenv("BENCHMARK_MAX_ACTIVE_CALLS", raising=False)
    monkeypatch.setattr(
        live_dashboard,
        "read_repo_env_values",
        lambda _names: {"BENCHMARK_PAID_CALL_MAX_ACTIVE": "3"},
    )
    monkeypatch.chdir(tmp_path)
    set_paid_call_policy(64, lease_dir=tmp_path / "leases")

    with paid_call_lease(
        lease_dir=tmp_path / "leases",
        model="google/gemini-flash",
        role="model_under_test",
        module="aita",
        max_active_calls=3,
    ):
        data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["paid_call_active_count"] == 1
    assert data["summary"]["paid_call_max_active"] == 3
    assert data["summary"]["paid_call_rate_limit_cooldown_count"] == 0
    assert data["paid_call_leases"]["max_active_calls"] == 3
    assert data["paid_call_leases"]["capacity"]["policy_limit"] == 64
    assert data["paid_call_leases"]["capacity"]["effective_limit"] == 3
    assert data["paid_call_leases"]["capacity"]["effective_limit_source"] == (
        "environment:BENCHMARK_PAID_CALL_MAX_ACTIVE"
    )
    assert data["paid_call_leases"]["active_leases"][0]["model"] == "google/gemini-flash"


def test_operational_activity_is_attributed_from_live_leases_not_stale_scheduler_counts(tmp_path, monkeypatch):
    run_dir = tmp_path / "active-run" / "sus"
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    set_paid_call_policy(8, lease_dir=lease_dir)
    now = live_dashboard.utc_now()
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "updated_at": now,
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "active-run",
            "contract_scope": "module",
            "modules": [
                {
                    "module": "sus",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": f"sus:{index}"} for index in range(8)],
                }
            ],
        },
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-active",
            "state": "running",
            "run_id": "active-run",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "updated_at": now,
            "progress": {
                "expected_units": 8,
                "completed_units": 0,
                "remaining_units": 8,
                "active_units": 8,
            },
        },
    )

    with paid_call_lease(
        lease_dir=lease_dir,
        model="test/model",
        role="model_under_test",
        module="sus",
        run_id="active-run",
        max_active_calls=8,
    ):
        data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    generating = next(
        stage for stage in data["operational_queue"]["stages"] if stage["id"] == "generating"
    )
    assert generating["items"][0]["active_units"] == 1
    assert generating["active_units"] == 1
    assert data["operational_queue"]["active_units"] == 1


def test_build_dashboard_data_surfaces_paid_call_rate_limit_cooldowns(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    record_rate_limit_cooldown(
        lease_dir=tmp_path / "leases",
        provider="openrouter",
        model="google/gemini-flash",
        headers={"Retry-After": "20"},
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["paid_call_rate_limit_cooldown_count"] == 1
    assert data["summary"]["paid_call_next_cooldown_seconds"] > 0
    assert data["paid_call_leases"]["rate_limit_cooldowns"][0]["provider"] == "openrouter"


def test_build_dashboard_data_shows_configured_paid_call_cap_without_active_leases(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "empty-leases"))
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "2")

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["paid_call_active_count"] == 0
    assert data["summary"]["paid_call_max_active"] == 2


def test_build_dashboard_data_groups_module_statuses(tmp_path):
    run_dir = tmp_path / "release-smoke" / "aita"
    transcript_path = run_dir / "gemini-flash_item0_side_a.json"
    _write_json(
        transcript_path,
        {
            "model": "gemini-flash",
            "item_idx": 0,
            "side": "side_a",
            "turns": [
                {
                    "turn": 1,
                    "user_message": "u1",
                    "model_response": "This is the latest saved response.",
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-ledger-v1",
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "metadata": {"models": ["gemini-flash"]},
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:03:20+00:00",
                "event": "conversation_started",
                "planned_turns": 5,
                "transcript_path": str(transcript_path),
            },
            {
                "timestamp": "2026-05-21T23:03:25+00:00",
                "event": "turn_saved",
                "turn": 1,
                "transcript_path": str(transcript_path),
            },
            {"timestamp": "2026-05-21T23:03:55+00:00", "event": "conversation_completed"},
            {"timestamp": "2026-05-21T23:04:29+00:00", "event": "score_saved"},
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 1
    assert data["operator"]["model_groups"]
    assert data["operator"]["models"]
    assert data["operator"]["commands"]
    assert any(
        command["title"] == "OpenRouter model/pricing preflight"
        for command in data["operator"]["commands"]
    )
    assert data["groups"][0]["run_id"] == "release-smoke"
    assert data["groups"][0]["statuses"] == {"completed": 1}
    module = data["groups"][0]["modules"][0]
    assert module["module"] == "aita"
    assert module["progress"]["percent"] == 100
    assert module["progress"]["turn_saved"] == 1
    assert module["progress"]["planned_turns"] == 5
    assert module["elapsed"] == "1m 15s"
    assert module["latest_transcript"]["model_response"] == "This is the latest saved response."
    turn_pair = next(item for item in data["evidence_feed"] if item["kind"] == "turn_pair")
    assert turn_pair["user_message"] == "u1"
    assert turn_pair["model_response"] == "This is the latest saved response."
    assert turn_pair["checks"]["assistant_ok"] is True
    assert turn_pair["stage"] == "generating"
    evidence_events = [item for item in data["evidence_feed"] if item["kind"] == "event"]
    score_event = next(item for item in evidence_events if item["event"] == "score_saved")
    assert score_event["stage"] == "score_ready"


def test_evidence_feed_preserves_event_stage_after_scoring_completes(tmp_path):
    run_dir = tmp_path / "completed-scored-run" / "aita"
    transcript_path = run_dir / "gemini-flash_item1_side_a.json"
    _write_json(
        transcript_path,
        {
            "model": "gemini-flash",
            "item_idx": 1,
            "side": "side_a",
            "turns": [
                {"turn": 1, "user_message": "first pressure", "model_response": "first answer"},
                {"turn": 2, "user_message": "second pressure", "model_response": "second answer"},
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-ledger-v1",
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "metadata": {"models": ["gemini-flash"]},
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:03:20+00:00",
                "stage": "generation",
                "event": "turn_saved",
                "turn": 1,
                "transcript_path": str(transcript_path),
            },
            {
                "timestamp": "2026-05-21T23:03:25+00:00",
                "stage": "generation",
                "event": "conversation_completed",
                "transcript_path": str(transcript_path),
            },
            {
                "timestamp": "2026-05-21T23:04:00+00:00",
                "stage": "scoring",
                "event": "judge_result_parsed",
                "model": "gemini-flash",
                "judge_model": "judge/model",
                "dimension": "integrity",
                "judge_result": 2,
                "max_score": 2,
                "item_idx": 1,
                "test_type": "pickside",
            },
            {
                "timestamp": "2026-05-21T23:04:05+00:00",
                "stage": "scoring",
                "event": "score_saved",
                "score_path": str(run_dir / "gemini-flash_item1_scores.json"),
            },
            {
                "timestamp": "2026-05-21T23:04:20+00:00",
                "stage": "scoring",
                "event": "final_results_saved",
            },
            {
                "timestamp": "2026-05-21T23:04:29+00:00",
                "stage": "scoring",
                "event": "stage_completed",
                "status": "completed",
                "validity": "score_ready",
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    stages = {(item["kind"], item.get("event"), item.get("turn")): item["stage"] for item in data["evidence_feed"]}

    assert stages[("turn_pair", None, 1)] == "generating"
    assert stages[("turn_pair", None, 2)] == "generating"
    assert stages[("event", "judge_result_parsed", None)] == "judging"
    assert stages[("event", "score_saved", None)] == "score_ready"
    assert stages[("event", "final_results_saved", None)] == "score_ready"
    assert stages[("event", "stage_completed", None)] == "score_ready"
    judge_result = next(
        item
        for item in data["evidence_feed"]
        if item.get("event") == "judge_result_parsed"
    )
    assert judge_result["model"] == "gemini-flash"
    assert judge_result["judge_model"] == "judge/model"
    assert judge_result["dimension"] == "integrity"
    assert judge_result["judge_result"] == 2
    assert judge_result["max_score"] == 2


def test_evidence_feed_keeps_score_writes_as_cumulative_ready_evidence(tmp_path):
    run_dir = tmp_path / "scoring-run" / "epistemic"
    now = live_dashboard.utc_now()
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "epistemic",
            "stage": "scoring",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": now,
            "updated_at": now,
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": now,
                "stage": "scoring",
                "event": "score_saved",
                "model": "condition-low",
                "item_idx": index,
                "test_type": "pickside",
                "score_path": str(run_dir / f"condition-low_item{index}_scores.json"),
            }
            for index in range(12)
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path, event_limit=4))
    score_writes = [
        item
        for item in data["evidence_feed"]
        if item["kind"] == "event" and item.get("event") == "score_saved"
    ]

    assert len(score_writes) == 12
    assert {item["stage"] for item in score_writes} == {"score_ready"}


def test_build_dashboard_data_uses_full_ledger_for_progress_and_latest_turn(tmp_path):
    run_dir = tmp_path / "long-aita-run" / "aita"
    transcript_path = run_dir / "gemini-flash_item24_side_a.json"
    _write_json(
        transcript_path,
        {
            "model": "gemini-flash",
            "item_idx": 24,
            "side": "side_a",
            "turns": [
                {
                    "turn": 5,
                    "user_message": "latest user pressure",
                    "model_response": "latest saved model response",
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-ledger-v1",
            "module": "aita",
            "stage": "scoring",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "updated_at": "2099-05-21T23:05:00+00:00",
            "metadata": {"models": ["gemini-flash"]},
        },
    )
    events = []
    for index in range(40):
        events.append(
            {
                "timestamp": "2026-05-21T23:03:20+00:00",
                "sequence": index + 1,
                "event": "conversation_started",
                "planned_turns": 5,
            }
        )
    for index in range(125):
        events.append(
            {
                "timestamp": "2026-05-21T23:03:30+00:00",
                "sequence": 41 + index,
                "event": "turn_saved",
                "turn": (index % 5) + 1,
                "transcript_path": str(transcript_path),
            }
        )
    for index in range(20):
        events.append(
            {
                "timestamp": "2026-05-21T23:04:30+00:00",
                "sequence": 166 + index,
                "event": "score_saved",
            }
        )
    _append_events(run_dir / "RUN_EVENTS.jsonl", events)

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path, event_limit=10))
    module = data["groups"][0]["modules"][0]

    assert module["recent_events"][0]["event"] == "score_saved"
    assert module["progress"]["source"] == "full_ledger_events"
    assert module["progress"]["planned_turns"] == 200
    assert module["progress"]["turn_saved"] == 125
    assert module["progress"]["scores_saved"] == 20
    assert module["progress"]["percent"] == 60
    assert module["latest_transcript"]["model_response"] == "latest saved model response"


def test_build_dashboard_data_reads_sus_conversation_bundle_preview(tmp_path):
    run_dir = tmp_path / "sus-smoke" / "sus"
    conversations_path = run_dir / "sus-bench-20260527-010000-conversations.json"
    _write_json(
        conversations_path,
        [
            {
                "model": "google/gemini-3-flash-preview",
                "scenario": "bridge_heights",
                "run_number": 1,
                "conversation": [
                    {"role": "user", "content": "Initial bridge question"},
                    {"role": "assistant", "content": "Initial safe answer"},
                    {"role": "user", "content": "Latest reframed pressure"},
                    {"role": "assistant", "content": "Latest SUS model response"},
                ],
            }
        ],
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "schema_version": "benchmark-run-ledger-v1",
            "module": "sus",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": "2026-05-27T08:00:00+00:00",
            "updated_at": "2026-05-27T08:01:00+00:00",
            "results_path": str(run_dir / "sus-bench-20260527-010000.json"),
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-27T08:01:00+00:00",
                "event": "stage_completed",
                "status": "completed",
            }
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    module = data["groups"][0]["modules"][0]
    assert module["latest_transcript"]["user_message"] == "Latest reframed pressure"
    assert module["latest_transcript"]["model_response"] == "Latest SUS model response"
    assert module["latest_transcript"]["test_type"] == "bridge_heights"
    assert module["latest_transcript"]["item_idx"] == 1


def test_dashboard_renders_provider_refusal_as_event_not_turn_pair(tmp_path):
    run_dir = tmp_path / "sus-refusal" / "sus"
    conversations_path = run_dir / "sus-refusal-conversations.json"
    _write_json(
        conversations_path,
        [
            {
                "model": "test/model",
                "scenario": "bridge_heights",
                "run_number": 1,
                "conversation": [{"role": "user", "content": "Bridge prompt"}],
                "turn_outcomes": [
                    {
                        "type": "provider_refusal",
                        "stop_reason": "refusal",
                        "timestamp": "2026-07-14T12:00:00+00:00",
                    }
                ],
            }
        ],
    )

    items = live_dashboard._build_evidence_items_from_sus_conversations_file(
        conversations_path,
        group="sus-refusal",
        module_path="sus",
        status={"module": "sus", "status": "completed", "validity": "not_score_ready"},
        fallback_timestamp="2026-07-14T12:00:01+00:00",
        limit=4,
    )

    assert not [item for item in items if item["kind"] == "turn_pair"]
    refusal = next(item for item in items if item["kind"] == "turn_outcome")
    assert refusal["event"] == "provider_refusal"
    assert refusal["stop_reason"] == "refusal"
    assert refusal["problem"] == "Provider refusal (stop reason: refusal)"


def test_build_dashboard_data_surfaces_cost_when_status_has_cost(tmp_path):
    run_dir = tmp_path / "run-1" / "sus"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "run",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cost": {
                "total_cost_usd": 0.01531,
                "total_calls": 6,
                "tokens_in": 10,
                "tokens_out": 5,
                "credit_remaining_usd": 0.75,
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert data["groups"][0]["cost_total_usd"] == 0.0153
    assert module["cost"] == {
        "total_cost_usd": 0.0153,
        "reported_cost_usd": 0,
        "estimated_cost_usd": 0,
        "unclassified_cost_usd": 0.0153,
        "total_calls": 6,
        "tokens": 15,
        "tokens_in": 10,
        "tokens_out": 5,
        "thinking_tokens_out": 0,
        "billable_tokens_out": 5,
        "billable_tokens": 15,
        "credit_remaining_usd": 0.75,
    }
    assert module["spend_guard"]["severity"] == "attention"
    assert module["spend_guard"]["label"] == "low credit"
    assert data["summary"]["spend_attention_count"] == 1


def test_build_dashboard_data_flags_unpriced_calls_before_more_spend(tmp_path):
    run_dir = tmp_path / "run-unpriced" / "sus"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "run",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cost": {
                "total_cost_usd": 0.0,
                "total_calls": 3,
                "tokens_in": 100,
                "tokens_out": 20,
                "unknown_cost_calls": 3,
                "unknown_cost_by_model": {"claude-opus-5": 3},
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert module["cost"]["unknown_cost_calls"] == 3
    assert module["cost"]["unknown_cost_by_model"] == {"claude-opus-5": 3}
    assert module["spend_guard"]["severity"] == "attention"
    assert module["spend_guard"]["label"] == "unpriced calls"
    assert "Stop before more paid calls" in module["spend_guard"]["detail"]
    assert data["summary"]["spend_attention_count"] == 1


def test_build_dashboard_data_demotes_adapter_only_unpriced_calls(tmp_path):
    run_dir = tmp_path / "run-adapter" / "epis"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-adapter",
            "contract_scope": "module",
            "lifecycle_state": "prepared",
            "expected_models": [
                {
                    "key": "private-model",
                    "model_id": "lab/private-model",
                    "condition_metadata": {"adapter_profile": "private_served_endpoint"},
                }
            ],
            "modules": [
                {
                    "module": "epistemic",
                    "expected_units": [{"unit_id": "epis:private-model:item0:side_a"}],
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "epistemic",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {"models": ["private-model"]},
            "cost": {
                "total_cost_usd": 1.25,
                "reported_cost_usd": 1.25,
                "total_calls": 4,
                "unknown_cost_calls": 2,
                "usage_by_role": {
                    "model_under_test": {
                        "calls": 4,
                        "reported_calls": 2,
                        "unknown_cost_calls": 2,
                    }
                },
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert module["spend_guard"]["severity"] == "info"
    assert module["spend_guard"]["kind"] == "adapter_pricing_partial"
    assert module["spend_guard"]["label"] == "adapter pricing partial"
    assert "does not affect benchmark evidence or scoring" in module["spend_guard"]["detail"]
    assert data["summary"]["spend_attention_count"] == 0


def test_dashboard_contract_summary_surfaces_pre_run_cost_estimate(tmp_path):
    run_dir = tmp_path / "run-1" / "aita"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "contract_scope": "module",
            "modules": [{"module": "aita", "expected_units": [{"unit_id": "u1"}]}],
            "call_plan": {
                "schema_version": "benchmark-cost-estimate-v1",
                "total_calls": {"low": 10, "expected": 12, "high": 15},
                "lines": [{"role": "judge", "model": "judge/a"}],
            },
            "cost_estimate": {
                "schema_version": "benchmark-cost-estimate-v1",
                "state": "estimated",
                "total_cost_usd": {"low": 1.0, "expected": 2.0, "high": 5.0},
                "cost_by_stage": {
                    "generation": {"low": 0.5, "expected": 1.25, "high": 3.0},
                    "scoring": {"low": 0.5, "expected": 0.75, "high": 2.0},
                },
                "lines": [{"large": "detail should not enter the summary"}],
            },
        },
    )

    contract = build_dashboard_data(DashboardOptions(results_root=tmp_path))["contracts"][0]

    assert contract["call_plan"] == {
        "schema_version": "benchmark-cost-estimate-v1",
        "total_calls": {"low": 10, "expected": 12, "high": 15},
        "line_count": 1,
        "scoring_judge_calls_expected": 0,
    }
    assert contract["cost_estimate"]["state"] == "estimated"
    assert contract["cost_estimate"]["total_cost_usd"]["expected"] == 2.0
    assert "lines" not in contract["cost_estimate"]


def test_completed_generation_waiting_for_scores_lands_in_needs_scoring(tmp_path):
    run_dir = tmp_path / "run-1" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "contract_scope": "module",
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [],
                    "expected_artifacts": [
                        {
                            "kind": "run_status",
                            "path": "RUN_STATUS.json",
                            "required_for": "diagnostic",
                        },
                        {
                            "kind": "final_results",
                            "path": "FINAL_RESULTS.json",
                            "required_for": "promotion",
                        },
                    ],
                }
            ],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}

    assert module["severity"] == "idle"
    assert module["attention"] is None
    assert module["score_state"]["kind"] == "needs_scoring"
    assert data["summary"]["attention_count"] == 0
    assert data["summary"]["contract_attention_count"] == 0
    assert lanes["needs_scoring"]["items"][0]["run_id"] == "run-1"
    assert lanes["attention"]["count"] == 0


def test_operational_views_exclude_uncontracted_nested_rescore_artifacts(tmp_path):
    run_dir = tmp_path / "run-1" / "aita"
    supplemental_dir = run_dir / "high-only-scoring"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "contract_scope": "module",
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": "side-a"}, {"unit_id": "side-b"}],
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "updated_at": "2026-07-30T12:00:00+00:00",
        },
    )
    _write_json(
        supplemental_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "updated_at": "2026-07-30T12:01:00+00:00",
        },
    )
    _append_events(
        supplemental_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-07-30T12:01:00+00:00",
                "sequence": 1,
                "event": "score_saved",
                "model": "test-model",
            }
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    group = data["groups"][0]
    modules = {module["module_path"]: module for module in group["modules"]}
    operational_items = [
        item
        for stage in data["operational_queue"]["stages"]
        for item in stage["items"]
    ]
    flow_items = [
        item
        for lane in data["flow"]["lanes"]
        for item in lane["items"]
    ]

    assert modules["aita"]["contract_membership"] == "contract"
    assert modules["aita/high-only-scoring"]["contract_membership"] == "supplemental"
    assert group["supplemental_module_count"] == 1
    assert data["operational_queue"]["total_units"] == 2
    assert {item["module_path"] for item in operational_items} == {"aita"}
    assert {item["module_path"] for item in flow_items} == {"aita"}


def test_needs_scoring_keeps_generation_and_scoring_progress_separate(tmp_path):
    run_dir = tmp_path / "run-1" / "sus"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "generation",
            "status": "completed",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "run-1",
            "contract_scope": "module",
            "modules": [
                {
                    "module": "sus",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": "run-a"}, {"unit_id": "run-b"}],
                }
            ],
            "call_plan": {
                "lines": [
                    {
                        "stage": "scoring",
                        "role": "judge",
                        "operation": "sus_post_analysis",
                        "calls": {"expected": 2},
                    },
                    {
                        "stage": "scoring",
                        "role": "judge",
                        "operation": "sus_post_analysis",
                        "calls": {"expected": 2},
                    },
                    {
                        "stage": "scoring",
                        "role": "judge",
                        "operation": "sus_post_analysis",
                        "calls": {"expected": 2},
                    },
                ]
            },
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {"timestamp": "2026-05-21T23:04:20+00:00", "event": "run_completed", "unit_id": "run-a"},
            {"timestamp": "2026-05-21T23:04:30+00:00", "event": "run_completed", "unit_id": "run-b"},
        ],
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-completed",
            "state": "needs_scoring",
            "run_id": "run-1",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "state_dir": str(run_dir),
            "updated_at": "2026-05-21T23:04:31+00:00",
            "progress": {
                "expected_units": 2,
                "completed_units": 2,
                "remaining_units": 0,
                "active_units": 3,
                "percent": 100,
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}
    operational = {stage["id"]: stage for stage in data["operational_queue"]["stages"]}

    assert lanes["needs_scoring"]["items"][0]["complete_units"] == 2
    item = operational["needs_scoring"]["items"][0]
    assert item["generation_expected_units"] == 2
    assert item["generated_units"] == 2
    assert item["score_expected_units"] == 2
    assert item["score_completed_units"] == 0
    assert item["judge_calls_expected"] == 6
    assert item["judge_calls_completed"] == 0
    assert item["completed_units"] == 0
    assert item["active_units"] == 0
    assert data["operational_queue"]["generation_completed_units"] == 2
    assert data["operational_queue"]["score_completed_units"] == 0
    assert data["operational_queue"]["judge_calls_expected"] == 6
    assert data["operational_queue"]["active_units"] == 0


def test_epistemic_sample_summary_names_cases_conversations_and_test_types():
    contract = {
        "identity": {
            "sample_spec": {
                "items": {
                    "delusion": [{"position": 0}, {"position": 1}],
                    "mirror": [{"position": 0}, {"position": 1}],
                    "pickside": [{"position": 0}, {"position": 1}],
                },
                "test_types": ["delusion", "mirror", "pickside"],
            }
        }
    }

    assert live_dashboard._sample_summary(
        contract,
        {"module": "epistemic", "expected_units": 10},
    ) == "n=10 conversations · 6 scored cases · 3 test types"


def test_scoring_queue_uses_unique_score_artifacts_and_condition_models(tmp_path):
    run_dir = tmp_path / "scoring-run" / "epistemic"
    now = live_dashboard.utc_now()
    expected_units = []
    for item_idx in range(2):
        for side in ("side_a", "side_b"):
            expected_units.append(
                {
                    "unit_id": f"epis:condition-low:item{item_idx}:{side}",
                    "model_key": "condition-low",
                    "model_id": "provider/base-model",
                    "item_idx": item_idx,
                    "test_type": "pickside",
                    "side": side,
                    "expected_transcript_path": f"condition-low_item{item_idx}_{side}.json",
                    "expected_score_path": f"condition-low_item{item_idx}_scores.json",
                }
            )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "scoring-run",
            "contract_scope": "module",
            "modules": [
                {
                    "module": "epistemic",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": expected_units,
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "epistemic",
            "stage": "scoring",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": now,
            "updated_at": now,
        },
    )
    events = [
        {
            "timestamp": now,
            "event": "conversation_completed",
            "model": "condition-low",
            "item_idx": unit["item_idx"],
            "test_type": unit["test_type"],
            "side": unit["side"],
        }
        for unit in expected_units
    ]
    events.append(
        {
            "timestamp": now,
            "stage": "scoring",
            "event": "score_saved",
            "model": "condition-low",
            "item_idx": 0,
            "test_type": "pickside",
            "score_path": str(run_dir / "condition-low_item0_scores.json"),
        }
    )
    _append_events(run_dir / "RUN_EVENTS.jsonl", events)
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-scoring",
            "state": "scoring",
            "run_id": "scoring-run",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "updated_at": now,
            "progress": {
                "expected_units": 4,
                "completed_units": 4,
                "remaining_units": 0,
                "active_units": 0,
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    scoring = next(
        stage for stage in data["operational_queue"]["stages"] if stage["id"] == "scoring"
    )
    item = scoring["items"][0]

    assert item["generation_expected_units"] == 4
    assert item["generated_units"] == 4
    assert item["expected_score_units"] == 2
    assert item["scored_units"] == 1
    assert item["expected_units"] == 2
    assert item["completed_units"] == 1
    assert len(item["models"]) == 1
    assert item["models"][0] == {
        "id": "condition-low",
        "label": "condition-low",
        "model_id": "provider/base-model",
        "expected_units": 4,
        "expected_score_units": 2,
        "completed_units": 4,
        "scored_units": 1,
        "active_units": 0,
    }


def test_score_ready_legacy_run_uses_completed_units_as_expected_floor(tmp_path):
    run_dir = tmp_path / "legacy-scored" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "updated_at": "2026-05-21T23:04:30+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "legacy-scored",
            "contract_scope": "module",
            "modules": [{"module": "aita", "stage": "generation", "expected_units": []}],
        },
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-legacy",
            "state": "score_ready",
            "run_id": "legacy-scored",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "updated_at": "2026-05-21T23:04:31+00:00",
            "progress": {
                "expected_units": 0,
                "completed_units": 5,
                "remaining_units": 0,
                "active_units": 0,
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    scored = next(
        stage for stage in data["operational_queue"]["stages"] if stage["id"] == "score_ready"
    )
    item = scored["items"][0]

    assert item["completed_units"] == 5
    assert item["expected_units"] == 5
    assert item["progress_percent"] == 100.0
    assert scored["completed_units"] == scored["expected_units"] == 5


def test_build_dashboard_data_treats_stale_running_ledgers_as_attention(tmp_path):
    run_dir = tmp_path / "stale-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": "2000-01-01T00:00:00+00:00",
            "updated_at": "2000-01-01T00:00:00+00:00",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert data["summary"]["running_count"] == 0
    assert data["summary"]["attention_count"] == 1
    assert module["stale_running"] is True
    assert module["severity"] == "attention"
    assert module["attention"]["title"] == "Stale running ledger"
    assert module["score_state"]["label"] == "Not scoreable"


def test_build_dashboard_data_excludes_rejected_disposition_from_active_attention(tmp_path):
    run_dir = tmp_path / "bad-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "failure_reason": "provider returned 429 before completion",
            "counters": {"turns_completed": 2, "turns_expected": 10},
        },
    )
    _write_json(
        run_dir / "RUN_DISPOSITION.json",
        {
            "schema_version": "benchmark-run-disposition-v1",
            "disposition": "rejected_from_analysis",
            "reason": "openrouter_429_incomplete_generation",
            "eligible_for_scoring": False,
            "eligible_for_promotion": False,
            "decided_at": "2026-05-21T23:05:00+00:00",
            "decided_by": "dashboard_operator",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}

    assert module["analysis_state"] == "rejected_from_analysis"
    assert module["severity"] == "rejected"
    assert module["attention"] is None
    assert module["eligible_for_scoring"] is False
    assert data["summary"]["attention_count"] == 0
    assert data["summary"]["failed_count"] == 0
    assert data["summary"]["rejected_count"] == 1
    assert lanes["attention"]["count"] == 0
    assert lanes["rejected"]["count"] == 1


def test_build_dashboard_data_surfaces_attention_and_latest_events(tmp_path):
    run_dir = tmp_path / "release-smoke" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "failed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "failure_reason": "sample_a.json: 0/5 turns (model failed)",
            "incomplete_conversations": [
                "sample_a.json: 0/5 turns (model failed)",
                "sample_b.json: 0/5 turns (model failed)",
            ],
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:03:20+00:00",
                "sequence": 1,
                "event": "conversation_started",
                "planned_turns": 5,
            },
            {
                "timestamp": "2026-05-21T23:04:00+00:00",
                "sequence": 2,
                "event": "conversation_incomplete",
                "planned_turns": 5,
                "turns": 0,
                "failure_stage": "model",
                "failure_reason": "model failed",
            },
            {
                "timestamp": "2026-05-21T23:04:30+00:00",
                "sequence": 3,
                "event": "stage_failed",
                "status": "failed_incomplete",
                "failure_reason": "sample_a.json: 0/5 turns (model failed)",
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert data["summary"]["attention_count"] == 1
    assert data["summary"]["failed_count"] == 1
    assert data["latest_events"][0]["event"] == "stage_failed"
    assert module["severity"] == "attention"
    assert module["score_state"]["label"] == "Score blocked"
    assert module["progress"]["conversations_incomplete"] == 1
    assert module["attention"]["title"] == "Incomplete generation"
    assert module["attention"]["incomplete_count"] == 2
    assert module["attention"]["incomplete_examples"] == [
        "sample_a.json: 0/5 turns (model failed)",
        "sample_b.json: 0/5 turns (model failed)",
    ]


def test_build_dashboard_data_surfaces_classified_failure_guidance(tmp_path):
    run_dir = tmp_path / "classified-failure" / "aita"
    reason = "Adapter rejected backend analysis failure: request setting disabled"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_invalid",
            "validity": "not_score_ready",
            "updated_at": "2026-07-30T12:04:00+00:00",
            "failure_reason": reason,
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-07-30T12:03:59+00:00",
                "sequence": 1,
                "event": "attempt_failure_classified",
                "model": "claude-opus-5",
                "evidence_class": "instrument_defect",
                "category": "payload",
                "action": "halt",
                "provider": "anthropic",
                "provider_code": "adapter_backend_analysis_failure",
                "retry_policy_kind": "terminal",
                "failure_reason": reason,
            },
            {
                "timestamp": "2026-07-30T12:04:00+00:00",
                "sequence": 2,
                "event": "stage_failed",
                "status": "failed_invalid",
                "failure_reason": reason,
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]
    classified = next(
        item
        for item in data["evidence_feed"]
        if item.get("event") == "attempt_failure_classified"
    )

    assert module["attention"]["classification"] == {
        "evidence_class": "instrument_defect",
        "category": "payload",
        "action": "halt",
        "provider": "anthropic",
        "provider_code": "adapter_backend_analysis_failure",
        "retry_policy_kind": "terminal",
        "label": "Instrument defect / payload",
    }
    assert "Stop paid calls" in module["attention"]["action"]
    assert module["score_state"]["label"] == "Instrument defect"
    assert classified["evidence_class"] == "instrument_defect"
    assert classified["category"] == "payload"
    assert classified["provider_code"] == "adapter_backend_analysis_failure"


def test_build_dashboard_data_classifies_legacy_adapter_failure_marker(tmp_path):
    run_dir = tmp_path / "legacy-adapter-failure" / "aita"
    reason = (
        "Error code: 502 - {'error': 'Adapter rejected backend analysis failure', "
        "'code': 'adapter_backend_analysis_failure', "
        "'benchmark_action': 'stop_run_preserve_artifacts'}"
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "updated_at": "2026-06-26T20:15:31+00:00",
            "failure_reason": reason,
            "incomplete_conversations": [reason],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert module["attention"]["classification"]["evidence_class"] == "instrument_defect"
    assert module["attention"]["classification"]["category"] == "adapter_backend_analysis_failure"
    assert module["score_state"]["label"] == "Instrument defect"
    assert "preflight" in module["score_state"]["action"]


def test_build_dashboard_data_explains_rate_limited_incomplete_generation(tmp_path):
    run_dir = tmp_path / "rate-limited" / "aita"
    reason = (
        "gemini-flash_item508_side_a.json: 4/5 turns "
        "(Error code: 429 - Rate limit exceeded: @ratelimit/too-many-requests.)"
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "failed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "failure_reason": reason,
            "incomplete_conversations": [reason],
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:04:30+00:00",
                "sequence": 1,
                "event": "stage_failed",
                "status": "failed_incomplete",
                "failure_reason": reason,
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert module["attention"]["title"] == "Rate-limited incomplete generation"
    assert module["score_state"]["label"] == "Rate-limited"
    assert "resume generation" in module["score_state"]["action"]


def test_build_dashboard_data_surfaces_exhausted_provider_credits(tmp_path):
    run_dir = tmp_path / "credit-exhausted" / "aita"
    reason = "Provider error: insufficient_quota; credit balance is exhausted."
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "failed_billing",
            "validity": "not_score_ready",
            "started_at": "2026-07-30T12:00:00+00:00",
            "failed_at": "2026-07-30T12:04:00+00:00",
            "updated_at": "2026-07-30T12:04:00+00:00",
            "failure_reason": reason,
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-07-30T12:04:00+00:00",
                "sequence": 1,
                "event": "stage_failed",
                "stage": "scoring",
                "status": "failed_billing",
                "failure_reason": reason,
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    module = data["groups"][0]["modules"][0]

    assert data["summary"]["attention_count"] == 1
    assert module["severity"] == "attention"
    assert module["attention"]["title"] == "Credits exhausted - refill required"
    assert module["score_state"]["label"] == "Credits exhausted"
    assert "Refill the provider account" in module["score_state"]["action"]
    assert "same prepared contract" in module["score_state"]["action"]


def test_build_dashboard_data_redacts_legacy_provider_ids(tmp_path):
    run_dir = tmp_path / "legacy-failure" / "aita"
    fake_user_id = "user_" + ("C" * 32)
    raw = (
        "Error code: 400 - {'error': {'message': 'bad model'}, "
        f"'user_id': '{fake_user_id}'}}"
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "failed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "failure_reason": raw,
            "incomplete_conversations": [raw],
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:04:30+00:00",
                "sequence": 1,
                "event": "stage_failed",
                "failure_reason": raw,
            },
        ],
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    rendered = json.dumps(data)

    assert fake_user_id not in rendered
    assert "'<redacted>'" in rendered
    assert "400" in data["groups"][0]["modules"][0]["attention"]["reason"]
    assert data["latest_events"][0]["failure_reason"].endswith("'user_id': '<redacted>'}")


def test_build_dashboard_data_surfaces_contract_and_control_without_status(tmp_path):
    run_dir = tmp_path / "contract-only-run"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "contract-only-run",
            "created_at": "2026-05-21T23:03:15+00:00",
            "contract_scope": "run_group",
            "identity": build_provenance_identity(
                benchmark_family_id="aita",
                benchmark_spec={
                    "module": "aita",
                    "module_version": "test",
                    "prompt_hashes": {"seeker": "abc"},
                    "score_dimensions": ["outcome_a"],
                },
                sample_spec={"item_indices": [0], "sides_by_item": {"0": ["side_a"]}},
                judge_panel={"judges": [{"role": "primary", "model_id": "judge/model"}]},
                model_conditions=[
                    {"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}
                ],
                execution={"run_id": "contract-only-run", "runner": "test"},
            ),
            "expected_models": [
                {"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}
            ],
            "expected_judges": [{"role": "primary", "model_id": "judge/model"}],
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "output_dir": "aita",
                    "expected_units": [
                        {
                            "unit_id": "aita:gemini-flash:item0:side_a",
                            "model_key": "gemini-flash",
                            "planned_turns": 5,
                            "expected_transcript_path": "sample.json",
                        }
                    ],
                    "expected_artifacts": [
                        {
                            "kind": "run_status",
                            "path": "RUN_STATUS.json",
                            "required_for": "diagnostic",
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_CONTROL.json",
        {
            "schema_version": "benchmark-run-control-v1",
            "state": "requested",
            "action": "stop_before_next_paid_call",
            "reason": "operator saw provider failures",
            "requested_by": "dashboard",
            "updated_at": "2026-05-21T23:05:15+00:00",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 0
    assert data["groups"][0]["contract_only"] is True
    assert data["contracts"][0]["run_id"] == "contract-only-run"
    assert data["contracts"][0]["expected_models"][0]["model_id"] == "google/gemini-3-flash-preview"
    assert data["contracts"][0]["identity"]["benchmark_family_id"] == "aita"
    assert data["contracts"][0]["provenance"]["benchmark_family_id"] == "aita"
    assert data["contracts"][0]["provenance"]["comparison_spec_hash"]
    assert data["contracts"][0]["provenance"]["model_conditions_hash"]
    assert data["contracts"][0]["provenance"]["run_execution_hash"]
    assert data["contracts"][0]["attention"] is True
    assert data["contracts"][0]["control"]["active"] is True
    assert data["contracts"][0]["control"]["label"] == "Stop before next paid call"
    assert data["summary"]["contract_count"] == 1
    assert data["summary"]["contract_attention_count"] == 1
    assert data["summary"]["contract_expected_units"] == 1
    assert data["summary"]["active_control_count"] == 1


def test_build_dashboard_data_groups_runtime_contract_by_path_when_run_id_differs(tmp_path):
    run_dir = tmp_path / "prepared-run" / "sus"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "run",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "runtime-sus-id",
            "contract_scope": "module",
            "identity": build_provenance_identity(
                benchmark_family_id="sus",
                benchmark_spec={"module": "sus"},
                sample_spec={"scenario_ids": ["bridge_heights"], "runs": 1},
                judge_panel={"panel": ["judge/model"]},
                model_conditions=[{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
                execution={"run_id": "runtime-sus-id"},
            ),
            "expected_models": [
                {"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}
            ],
            "modules": [{"module": "sus", "stage": "run", "output_dir": ".", "expected_units": []}],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    groups = {group["run_id"]: group for group in data["groups"]}
    assert set(groups) == {"prepared-run"}
    assert groups["prepared-run"]["statuses"] == {"completed": 1}
    assert groups["prepared-run"]["contracts"][0]["run_id"] == "runtime-sus-id"
    assert groups["prepared-run"]["contracts"][0]["path_group_id"] == "prepared-run"


def test_build_dashboard_data_reconciles_thin_runtime_contract_from_run_plan(tmp_path):
    run_root = tmp_path / "frontier-smoke"
    run_dir = run_root / "epis"
    _write_json(
        run_root / "RUN_PLAN.json",
        {
            "schema_version": "benchmark-run-plan-v1",
            "run_id": "frontier-smoke",
            "lifecycle_state": "prepared",
            "model_selector": "gemini-flash",
            "judge_set": "frontier",
            "modules": [
                {
                    "module": "epis",
                    "output_dir": str(run_dir),
                    "contract_path": str(run_dir / "RUN_CONTRACT.json"),
                    "expected_units": 5,
                    "model_selector": "gemini-flash",
                    "judge_set": "frontier",
                    "execute_command": "python -m epis_bench run",
                    "score_command": "python -m epis_bench score",
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "epistemic",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "frontier-smoke",
            "contract_scope": "module",
            "expected_models": [
                {"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}
            ],
            "modules": [
                {
                    "module": "epistemic",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": f"epis:gemini-flash:{idx}"} for idx in range(5)],
                }
            ],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    group = next(group for group in data["groups"] if group["run_id"] == "frontier-smoke")
    contract = group["contracts"][0]
    assert contract["model_selector"] == "gemini-flash"
    assert contract["judge_set"] == "frontier"
    assert contract["plan_reconciled"] is True
    assert "model_selector" in contract["plan_reconciled_fields"]
    assert "judge_set" in contract["plan_reconciled_fields"]

    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}
    item = lanes["score_ready"]["items"][0]
    assert item["run_id"] == "frontier-smoke"
    assert item["model_selector"] == "gemini-flash"
    assert item["judge_summary"] == "frontier"
    assert item["expected_units"] == 5


def test_summary_latest_elapsed_ignores_older_stale_running_max(tmp_path):
    stale_dir = tmp_path / "old-stale-running" / "sus"
    _write_json(
        stale_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "started_at": "2026-05-20T20:28:23+00:00",
            "updated_at": "2026-05-20T20:32:46+00:00",
        },
    )
    recent_dir = tmp_path / "recent-score-ready" / "aita"
    _write_json(
        recent_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-06-09T23:56:00+00:00",
            "completed_at": "2026-06-09T23:57:11+00:00",
            "updated_at": "2026-06-09T23:57:11+00:00",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["active_elapsed"] == "none"
    assert data["summary"]["latest_elapsed"] == "1m 11s"
    assert data["summary"]["latest_run_id"] == "recent-score-ready"
    assert data["summary"]["suite_elapsed"] != data["summary"]["latest_elapsed"]


def test_latest_run_id_prefers_running_group_over_newer_idle_group(tmp_path):
    _write_running_group_with_newer_prepared_contract(tmp_path)

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["latest_run_id"] == "older-running"


def test_flow_lanes_include_running_group_when_newer_prepared_exists(tmp_path):
    _write_running_group_with_newer_prepared_contract(tmp_path)

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}

    assert any(item["run_id"] == "older-running" for item in lanes["generating"]["items"])
    assert data["summary"]["latest_run_id"] == "older-running"


def test_build_dashboard_data_builds_operator_flow_lanes(tmp_path):
    prepared_dir = tmp_path / "prepared-local-endpoint" / "sus"
    _write_json(
        prepared_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "prepared-local-endpoint",
            "lifecycle_state": "prepared",
            "judge_set": "calibration",
            "model_selector": "group:local_endpoint_smoke",
            "execute_command": "python -m sus_bench run --output prepared-local-endpoint/sus",
            "identity": build_provenance_identity(
                benchmark_family_id="sus",
                benchmark_spec={"module": "sus"},
                sample_spec={"scenario_ids": ["bridge_heights"], "runs": 1},
                judge_panel={"panel": ["judge/model"]},
                model_conditions=[{"key": "local-openai-compatible", "model_id": "local/example-model"}],
                execution={"run_id": "prepared-local-endpoint"},
            ),
            "expected_models": [{"key": "local-openai-compatible", "model_id": "local/example-model"}],
            "expected_judges": [{"role": "analyzer", "model_id": "google/gemini-3-flash-preview"}],
            "modules": [
                {
                    "module": "sus",
                    "stage": "run",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": "sus:local-openai-compatible:bridge_heights:run1"}],
                }
            ],
        },
    )
    completed_dir = tmp_path / "completed-raw" / "sus"
    _write_json(
        completed_dir / "RUN_STATUS.json",
        {
            "module": "sus",
            "stage": "run",
            "status": "completed",
            "validity": "score_ready",
            "started_at": "2026-05-21T23:03:15+00:00",
            "completed_at": "2026-05-21T23:04:30+00:00",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "cost": {"total_cost_usd": 0.1},
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}

    assert data["flow"]["counts"]["prepared"] == 1
    assert data["flow"]["counts"]["score_ready"] == 1
    assert data["flow"]["group_counts"]["prepared"] == 1
    assert data["flow"]["group_counts"]["score_ready"] == 1
    assert data["flow"]["unit_counts"]["prepared"] == 1
    assert lanes["prepared"]["items"][0]["run_id"] == "prepared-local-endpoint"
    assert lanes["prepared"]["items"][0]["title"] == "SUS · bridge_heights · r1"
    assert lanes["prepared"]["items"][0]["model_summary"] == "Provider-routed · 1 model"
    assert lanes["prepared"]["items"][0]["model_names"] == ["local-openai-compatible"]
    assert lanes["prepared"]["items"][0]["judge_summary"] == "calibration · analyzer"
    assert "suite_tools.scheduler run" in lanes["prepared"]["items"][0]["execute_command"]
    assert "RUN_CONTRACT.json" in lanes["prepared"]["items"][0]["execute_command"]
    assert lanes["prepared"]["items"][0]["benchmark_condition_hash"]
    assert lanes["prepared"]["items"][0]["comparison_spec_hash"]
    assert lanes["score_ready"]["items"][0]["run_id"] == "completed-raw"
    assert lanes["score_ready"]["items"][0]["cost_total_usd"] == 0.1


def test_build_dashboard_data_surfaces_scheduler_queue_and_eta(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "2")
    run_dir = tmp_path / "scheduled-run" / "sus"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "scheduled-run",
            "lifecycle_state": "prepared",
            "execute_command": "python -m sus_bench run --output scheduled-run/sus",
            "identity": build_provenance_identity(
                benchmark_family_id="sus",
                benchmark_spec={"module": "sus"},
                sample_spec={"scenario_ids": ["bridge_heights"], "runs": 1},
                judge_panel={"panel": ["judge/model"]},
                model_conditions=[{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
                execution={"run_id": "scheduled-run"},
            ),
            "expected_models": [{"key": "gemini-flash", "model_id": "google/gemini-3-flash-preview"}],
            "modules": [
                {
                    "module": "sus",
                    "stage": "run",
                    "output_dir": ".",
                    "expected_units": [{"unit_id": "sus:gemini:bridge_heights:run1"}],
                }
            ],
        },
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-test",
            "state": "queued",
            "run_id": "scheduled-run",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "state_dir": str(run_dir),
            "command": "python -m sus_bench run --output scheduled-run/sus",
            "updated_at": "2026-05-21T23:03:15+00:00",
            "settings": {"max_active_calls": 6},
            "progress": {
                "expected_units": 4,
                "completed_units": 2,
                "remaining_units": 2,
                "active_units": 2,
                "effective_parallelism": 2,
                "percent": 50,
                "eta_seconds": 60,
                "eta_basis": "completed-unit-average",
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    lanes = {lane["id"]: lane for lane in data["flow"]["lanes"]}
    item = lanes["queued"]["items"][0]

    assert data["summary"]["scheduler_count"] == 1
    assert data["summary"]["scheduler_queued_count"] == 1
    assert data["summary"]["paid_call_max_active"] == 2
    assert data["schedulers"][0]["max_active_calls"] == 6
    assert lanes["queued"]["count"] == 1
    assert lanes["queued"]["group_count"] == 1
    assert lanes["queued"]["unit_count"] == 2
    assert lanes["queued"]["expected_units"] == 4
    assert item["scheduler_state"] == "queued"
    assert item["scheduler_eta_seconds"] == 60
    assert item["max_active_calls"] == 6
    assert item["complete_units"] == 2
    assert item["expected_units"] == 4
    operational = {stage["id"]: stage for stage in data["operational_queue"]["stages"]}
    assert data["operational_queue"]["total_units"] == 4
    assert data["operational_queue"]["generated_units"] == 2
    assert data["operational_queue"]["leases"]["cap"] == 2
    assert operational["queued"]["units"] == 2
    assert operational["queued"]["completed_units"] == 2
    assert operational["queued"]["expected_units"] == 4


def test_build_dashboard_data_ignores_stale_scheduler_attention_after_score_ready(tmp_path):
    run_dir = tmp_path / "rescored-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
            "updated_at": "2026-05-28T08:03:41+00:00",
        },
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-stale",
            "state": "attention",
            "run_id": "rescored-run",
            "state_dir": str(run_dir),
            "updated_at": "2026-05-28T07:57:58+00:00",
            "runner": {
                "status": "failed_incomplete",
                "stage": "scoring",
                "validity": "not_score_ready",
                "failure_reason": "AITA transcripts are not score-ready",
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["attention_count"] == 0
    assert data["summary"]["scheduler_attention_count"] == 0
    assert data["schedulers"][0]["raw_state"] == "attention"
    assert data["schedulers"][0]["state"] == "score_ready"
    assert data["schedulers"][0]["runner"]["validity"] == "score_ready"


def test_build_dashboard_data_builds_operational_attention_from_scheduler_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", "2")
    run_dir = tmp_path / "partial-run" / "aita"
    _write_json(
        run_dir / "RUN_CONTRACT.json",
        {
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "partial-run",
            "identity": build_provenance_identity(
                benchmark_family_id="aita",
                benchmark_spec={"module": "aita"},
                sample_spec={"item_count": 2},
                judge_panel={"panel": ["judge/model"]},
                model_conditions=[
                    {"key": "model-one", "model_id": "therapeutic-harness/model-one"},
                    {"key": "model-two", "model_id": "therapeutic-harness/model-two"},
                ],
                execution={"run_id": "partial-run"},
            ),
            "expected_models": [
                {"key": "model-one", "model_id": "therapeutic-harness/model-one"},
                {"key": "model-two", "model_id": "therapeutic-harness/model-two"},
            ],
            "modules": [
                {
                    "module": "aita",
                    "stage": "generation",
                    "output_dir": ".",
                    "expected_units": [
                        {
                            "unit_id": "aita:model-one:item0:side_a",
                            "model_key": "model-one",
                            "model_id": "therapeutic-harness/model-one",
                            "item_idx": 0,
                            "side": "side_a",
                            "expected_transcript_path": "model-one_item0_side_a.json",
                        },
                        {
                            "unit_id": "aita:model-one:item0:side_b",
                            "model_key": "model-one",
                            "model_id": "therapeutic-harness/model-one",
                            "item_idx": 0,
                            "side": "side_b",
                            "expected_transcript_path": "model-one_item0_side_b.json",
                        },
                        {
                            "unit_id": "aita:model-two:item1:side_a",
                            "model_key": "model-two",
                            "model_id": "therapeutic-harness/model-two",
                            "item_idx": 1,
                            "side": "side_a",
                            "expected_transcript_path": "model-two_item1_side_a.json",
                        },
                        {
                            "unit_id": "aita:model-two:item1:side_b",
                            "model_key": "model-two",
                            "model_id": "therapeutic-harness/model-two",
                            "item_idx": 1,
                            "side": "side_b",
                            "expected_transcript_path": "model-two_item1_side_b.json",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "updated_at": "2026-05-21T23:04:30+00:00",
            "incomplete_conversations": ["model-two_item1_side_b.json: 2/5 turns"],
        },
    )
    _append_events(
        run_dir / "RUN_EVENTS.jsonl",
        [
            {
                "timestamp": "2026-05-21T23:03:20+00:00",
                "event": "conversation_completed",
                "model": "model-one",
                "model_id": "therapeutic-harness/model-one",
                "item_idx": 0,
                "side": "side_a",
            },
            {
                "timestamp": "2026-05-21T23:03:30+00:00",
                "event": "conversation_completed",
                "model": "model-two",
                "model_id": "therapeutic-harness/model-two",
                "item_idx": 1,
                "side": "side_a",
            },
            {"timestamp": "2026-05-21T23:04:30+00:00", "event": "stage_failed"},
        ],
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-partial",
            "state": "attention",
            "run_id": "partial-run",
            "contract_path": str(run_dir / "RUN_CONTRACT.json"),
            "state_dir": str(run_dir),
            "updated_at": "2026-05-21T23:04:31+00:00",
            "settings": {"max_active_calls": 8},
            "progress": {
                "expected_units": 4,
                "completed_units": 2,
                "remaining_units": 2,
                "active_units": 1,
                "percent": 50,
            },
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    operational = {stage["id"]: stage for stage in data["operational_queue"]["stages"]}
    attention = operational["attention"]
    item = attention["items"][0]

    assert data["summary"]["scheduler_completed_units"] == 2
    assert data["summary"]["scheduler_attention_units"] == 1
    assert data["operational_queue"]["generated_units"] == 2
    assert data["operational_queue"]["leases"] == {
        "active": 0,
        "cap": 2,
        "registry_cap": 2,
        "source": "policy",
    }
    assert attention["units"] == 1
    assert attention["completed_units"] == 2
    assert attention["expected_units"] == 4
    assert item["stage"] == "attention"
    assert item["completed_units"] == 2
    assert item["expected_units"] == 4
    assert item["active_units"] == 0
    assert attention["active_units"] == 0
    assert data["operational_queue"]["active_units"] == 0
    models = {model["id"]: model for model in item["models"]}
    assert models["model-one"]["expected_units"] == 2
    assert models["model-one"]["completed_units"] == 1
    assert models["model-two"]["expected_units"] == 2
    assert models["model-two"]["completed_units"] == 1


def test_build_dashboard_data_ignores_rejected_scheduler_attention(tmp_path):
    run_dir = tmp_path / "rejected-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "failure_reason": "Error code: 429 - Rate limit exceeded",
            "updated_at": "2026-05-28T08:03:41+00:00",
        },
    )
    _write_json(
        run_dir / "RUN_DISPOSITION.json",
        {
            "schema_version": "benchmark-run-disposition-v1",
            "disposition": "rejected_from_analysis",
            "reason": "operator_rejected_malformed_or_incomplete_run",
            "eligible_for_scoring": False,
            "eligible_for_promotion": False,
        },
    )
    _write_json(
        run_dir / "SCHEDULER_STATUS.json",
        {
            "schema_version": "benchmark-scheduler-v1",
            "scheduler_id": "scheduler-rejected",
            "state": "attention",
            "run_id": "rejected-run",
            "state_dir": str(run_dir),
            "updated_at": "2026-05-28T07:57:58+00:00",
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["summary"]["attention_count"] == 0
    assert data["summary"]["scheduler_attention_count"] == 0
    assert data["schedulers"][0]["raw_state"] == "attention"
    assert data["schedulers"][0]["state"] == "rejected"
    assert data["schedulers"][0]["analysis_state"] == "rejected_from_analysis"


def test_render_html_contains_polling_dashboard_hooks():
    html = render_html(title="Test Dashboard", poll_ms=1234)
    dashboard_css = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.css").read_text()
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()
    theme_init_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "theme-init.js").read_text()

    html_snippets = [
        "Test Dashboard",
        "/assets/dashboard.css",
        '<script src="/assets/theme-init.js"></script>',
        "themeToggle",
        "top-menu",
        "brand-shield",
        "brandShield",
        "Anti-sycophancy shield",
        "/assets/anti-sycophancy-logo-static.png",
        "/assets/anti-sycophancy-logo-running.gif",
        "data-running-src",
        "quick-actions",
        "top-status",
        "topScopeControl",
        "Benchmark run scope",
        "top-scope-label",
        'data-short="Stage"',
        ">Stage</span>",
        "topComplete",
        "topElapsedLabel",
        "topElapsed",
        "data-quick-stage=\"attention\"",
        "top-stat-button attention",
        "data-quick-action=\"follow-live\"",
        "Pause feed",
        "copyPanel",
        "Select text",
        'data-poll-ms="1234"',
        '<script src="/assets/dashboard.js" defer></script>',
    ]
    for snippet in html_snippets:
        assert snippet in html, snippet

    html_absent_snippets = [
        "__BRAND_LOGO__",
        "__POLL_MS__",
        "const pollMs = 1234",
        "DASHBOARD_CONFIG",
        "<script>window.",
        "topWaiting",
        "topSpend",
        "/Users/me",
        "ericode",
    ]
    for snippet in html_absent_snippets:
        assert snippet not in html, snippet
    assert "<script>" not in html
    assert "benchmarkDashboardTheme" in theme_init_js
    assert "prefers-color-scheme: dark" in theme_init_js

    css_snippets = [
        "--font-serif: Georgia",
        "--font-sans: system-ui",
        "data-theme",
        "repeat(auto-fit, minmax(min(100%, 250px), 1fr))",
        "overflow-x: hidden",
        "@media (max-width: 900px)",
        "@media (max-width: 640px)",
        "@media (max-width: 440px)",
        '"brand scope actions theme"',
        '"scope scope"',
        'grid-template-areas:\n\t          "brand theme"\n\t          "scope scope"\n\t          "status status"\n\t          "actions actions"',
        "flex-wrap: wrap",
        "brand-logo-gemini",
        "border: 1px solid var(--color-line-soft)",
        "stage-judging",
        "--color-model-6",
        ".capacity-source",
        ".evidence-trace-legend-line",
        ".credit-alert",
        ".credit-alert.attention",
        'html[data-theme="dark"] .stage-select',
        ".stage-select option",
        ".failure-classification",
        ".failure-detail",
        ".hud-spend",
        "minmax(300px, 0.9fr)",
    ]
    for snippet in css_snippets:
        assert snippet in dashboard_css, snippet

    css_absent_snippets = [
        "Instrument Serif",
        "Source Serif 4",
        "grid-auto-flow: column",
        "scroll-behavior: smooth",
        "overflow-x: auto",
    ]
    for snippet in css_absent_snippets:
        assert snippet not in dashboard_css, snippet

    js_snippets = [
        "const pollMs = Number(document.body.dataset.pollMs || 2500)",
        "/api/runs",
        "RUN_CONTRACT.json",
        "RUN_CONTROL.json",
        "Run Contract",
        "function renderContractCostEstimate",
        "Planning estimate only",
        "Generation estimate",
        "Scoring estimate",
        "Model-under-test estimate",
        "Run-time support estimate",
        "Judge-call estimate",
        "Contract gaps",
        "Spend guard",
        "prepared groups",
        "module card",
        "work units",
        "Collapsed by default",
        "Model lock",
        "Comparable identity",
        "Benchmark condition",
        "Exact runset",
        "Sample condition",
        "Benchmark spec",
        "Model batch",
        "Batch condition",
        "Run execution",
        "Contract integrity",
        "Copy benchmark condition",
        "Copy exact runset",
        "Copy scheduler command",
        "Run Flow",
        "Rejected",
        "Prepared",
        "Queued",
        "brandShield.setAttribute('src', nextSrc)",
        "Needs Scoring",
        "Scheduler ETA",
        "Loaded elapsed",
        "Run control status",
        "Work queue",
        "Live evidence feed",
        "Jump to live",
        "evidenceTraceCanvas",
        "Evidence trace window",
        "Evidence content view",
        "canvas.style.width = '100%'",
        "const feedEntries = visibleEntries.slice().reverse()",
        "feedEntries.map((entry) => renderEvidenceItem(entry.item, entry.key))",
        "const DEFAULT_EVIDENCE_TRACE_WINDOW = '100'",
        "let evidenceAutoFollow = true",
        "let evidenceTraceAutoFollow = true",
        "let evidenceTraceWindow = DEFAULT_EVIDENCE_TRACE_WINDOW",
	        "let suppressFreshOnNextRender = false",
        "let pendingEvidenceLiveSnap = false",
        "let evidenceContentFilter = 'all'",
        "evidenceContentButton('text', 'Text')",
        "evidenceContentButton('writes', 'Writes')",
        "function resetEvidenceFiltersToAll",
        "function syncEvidenceDefaultsForRun",
        "function setEvidenceContentFilter",
        "data-evidence-window=\"${esc(value)}\"",
        "evidenceTraceWindowButton('all', 'All')",
        "evidenceTraceWindowButton('50', '50')",
        "function drawEvidenceTrace",
        "function selectEvidenceTraceKey",
        "function setEvidenceTraceWindow",
        "function scrollEvidenceTraceToLive",
        "pendingEvidenceLiveSnap = true",
        "feed.scrollTo({top: Math.max(0, targetTop), behavior: 'auto'})",
        "atLive: feed.scrollTop <= 36",
        "feed.scrollTop = 0",
	        "function evidenceTraceMode()",
	        "return 'horizontal'",
        "data-evidence-key",
        "syncTopScopeControl",
        "function sizeEvidenceTraceCanvas",
        "operational_queue",
        "function scopedOperationalQueue",
        "function scopedCostBreakdown",
        "provider-reported",
        "basis unavailable",
        "Estimated spend to date",
        "function runTiming",
        "Run duration",
        "Not started",
        "const queue = scopedOperationalQueue(data)",
        "function queueWorkGroupCount",
        "data-queue-toggle",
        "queueExpansionState",
	        "currentLifecycleStage",
	        "updateTopSummary",
	        "hudMetric('Generation', generationProgress, 'transcripts')",
	        "hudMetric('Scoring', scoringProgress, 'result bundles')",
	        "hudMetric('Judge calls', judgeProgress, 'completed / planned')",
	        "hudMetric('Capacity', `${activeLeases} / ${maxLeases}`, capacityNote, 'hud-capacity')",
	        "hudMetric('Run integrity', attentionUnits ? `${attentionUnits} issues` : '0', attentionNote)",
	        "Spend provenance",
	        "function scopedUnknownCostSummary",
        "Advanced inspection",
        "Scoped run flow, diagnostics, artifact snapshots, and raw ledger groups.",
        "advanced-subpanel",
        "Diagnostics Summary",
        "scopedLatestEvents",
        "Follow live",
        "syncPersistentControls",
        "applyStageFilter",
        "activeStageFilter",
	        "data-stage-filter",
        "awaiting scoring approval",
        "result bundles",
        "transcripts complete",
        "Saved turns and judge writes for the selected scope.",
        "Paid calls",
        "Radar",
	        "function capacitySourceLabel",
        ".env ceiling",
        "queueAccentClass",
        "(attention || {}).classification",
        "failure-classification",
        "provider_code",
        "const EVIDENCE_TRACE_MARK_FRACTION = 0.82",
        "const fullBarHeight = Math.max(14, Math.round(laneHeight * EVIDENCE_TRACE_MARK_FRACTION))",
        "function evidenceTraceModule(item)",
        "function evidenceTraceModuleColor(item)",
        "const freshPoints = points.filter((point) => point.fresh)",
        "const arrivalPoint = freshPoints[freshPoints.length - 1] || null",
        "context.moveTo(arrivalPoint.x, laneTop)",
        "`+${freshPoints.length} new · ${arrivalModule.label}`",
        "evidenceTraceEnterProgress",
        "evidenceTraceMotionReduced",
        "if (!suppressFreshOnNextRender && !firstEvidencePaint",
        "if (evidenceLoaded) suppressFreshOnNextRender = false",
        "suppressFreshOnNextRender = true",
        "sampledEvidenceTraceEntries",
        "evidence-trace-model-legend",
        "context.fillStyle = stageColor",
        "context.fillStyle = moduleColor",
        "Evidence event trace; line color indicates stage and top marker indicates benchmark module",
            "role=\"slider\"",
            "function ensureEvidenceDetails",
            "function ensureContractDetails",
            "function hydrateDashboardDetails",
            "await hydrateDashboardDetails(data)",
            "function renderCreditAlert",
            "Credits exhausted",
            "Credits running low",
            "Spend tracking incomplete",
            "Spend provenance",
            "Refill required",
            "'/api/evidence'",
            "'/api/contracts'",
            "data?.run_index || data?.groups || []",
            "const runsUrl = detailUrl('/api/runs', {scope: evidenceRunScope})",
            "runsEtag = '';",
            "data-selected=\"${selected}\"",
            "aria-current=\"${selected}\"",
            "function workingRawModules",
        "Run integrity and spend provenance",
        "evidence_feed",
        "evidenceAutoFollow",
        "renderMarkdown",
        "unwrapMessageContent",
        "brandMarks",
        "modelShortCode",
        "renderModelChip",
        "markdown-text",
        "model-chip",
        "provider text",
        "Primary Module Watch",
        "Latest user pressure",
        "Latest model response",
        "Recent ledger writes",
        "full-ledger progress",
        "Attention Queue",
        "Latest Events",
        "Raw Run Groups",
        "Operator tools",
        "Run builder, model selectors, judge sets, and CLI helpers.",
        "command-center",
        'data-run-builder="${esc(name)}"',
        "readRunBuilderState",
        "buildRunBuilderOutput",
        "data-run-builder-action=\"all-model-groups\"",
        "All modules",
        "Benchmarks",
        "Model groups",
        "Model groups are presets from suite_models.yaml",
        "renderModelGroupOption",
        "builder-model-group",
        "builder-group-models",
        "Judge panel",
        "Stage and size",
        "Names and limits",
        "repository root that contains suite_models.yaml",
        "Generated CLI",
        "runBuilderCli",
        "Agent prompt",
        "runBuilderPrompt",
        "Advanced CLI",
        "Module Snapshots",
        "Active elapsed",
        "Rate-limit pause",
        "Latest saved turn",
        "Raw command cards",
        "Score state",
        "Copy triage commands",
        "Copy failure",
        "Copy manually",
        "openDetails",
        "'If-None-Match': runsEtag",
        "response.status === 304",
        "new AbortController()",
        "controller.abort()",
        "signal: controller.signal",
        "window.clearTimeout(requestTimeout)",
        "window.setTimeout(refresh, nextRefreshDelay)",
    ]
    for snippet in js_snippets:
        assert snippet in dashboard_js, snippet

    js_absent_snippets = [
        "__POLL_MS__",
        "const pollMs = 1234",
        "const desiredWidth =",
        "data-trace-live",
        "Trace live",
        "feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 36",
        "Reset + replay",
        "queue-conveyor",
        "topWaiting",
        "topSpend",
        "Score-ready",
        "Score-Ready",
        "terminal unit",
        "quick-action stage-attention",
        "data-quick-stage=\"active\"",
        "data-quick-stage=\"score_ready\"",
        "data-quick-action=\"copy-root\"",
        "data-quick-action=\"commands\"",
        "queue-throughput",
        "setInterval(refresh, pollMs)",
        "dashboardSignature",
	        "evidenceTraceKind",
	        "drawEvidenceBead",
	        "Evidence bead timeline",
	        "function evidenceTraceWeight",
	        "function drawEvidenceTraceVertical",
	        "canvas.dataset.mode === 'vertical'",
	        "context.setLineDash([3, 3])",
	        "Evidence event trace; line color indicates model and base color indicates stage",
        "aria-selected=\"${selected}\"",
        "/Users/me",
        "ericode",
    ]
    for snippet in js_absent_snippets:
        assert snippet not in dashboard_js, snippet


def test_dashboard_poll_hydrates_details_before_one_render():
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()
    refresh_body = dashboard_js.split("async function refresh()", 1)[1]
    refresh_body = refresh_body.split("\n    refresh();", 1)[0]

    hydrate_offset = refresh_body.index("await hydrateDashboardDetails(data);")
    render_offset = refresh_body.index("render(data);", hydrate_offset)

    assert hydrate_offset < render_offset
    assert "renderOnComplete: false" in dashboard_js
    assert "ensureEvidenceDetails(data);\n" not in dashboard_js
    assert dashboard_js.count("app.innerHTML =") == 1
    assert "app.dataset.paintCount" in dashboard_js


def test_disposition_browser_action_requires_exact_confirmation_and_csrf_header():
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()

    assert "const csrfToken = document.body.dataset.csrfToken || '';" in dashboard_js  # noqa: release-audit-fixture
    assert 'window.prompt(' in dashboard_js
    assert 'confirmation !== `${confirmedAction} ${runId}`' in dashboard_js
    assert "'X-Benchmark-CSRF': csrfToken" in dashboard_js


def test_dashboard_scope_defaults_to_workflow_and_preserves_open_select():
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()

    assert "let evidenceRunScope = 'workflow:active';" in dashboard_js
    assert "Current workflow · ${workflow.member_count" in dashboard_js
    assert "document.activeElement === existing" in dashboard_js
    assert "currentOptions.forEach((option, index)" in dashboard_js


def test_advanced_dashboard_spend_uses_selected_run_scope():
    dashboard_js = (live_dashboard.DASHBOARD_ASSETS_DIR / "dashboard.js").read_text()
    render_body = dashboard_js.split("function render(data)", 1)[1].split("async function refresh", 1)[0]

    assert "const scopedCost = scopedCostTotal(data);" in render_body
    assert "money(summary.tracked_cost_total_usd)" not in render_body


def test_dashboard_handler_supports_head_health_checks(tmp_path):
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {"options": DashboardOptions(results_root=tmp_path), "page_title": "Test", "poll_ms": 1234},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        request = Request(f"http://127.0.0.1:{port}/api/runs", method="HEAD")
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/json; charset=utf-8"
            assert response.read() == b""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_returns_304_for_unchanged_etag(tmp_path):
    _write_json(tmp_path / "run-1" / "RUN_STATUS.json", {"status": "prepared"})
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {"options": DashboardOptions(results_root=tmp_path), "page_title": "Test", "poll_ms": 1234},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/api/runs"
        with urlopen(url, timeout=5) as response:
            etag = response.headers["ETag"]
            assert response.status == 200
            assert response.read()

        request = Request(url, headers={"If-None-Match": etag})
        try:
            urlopen(request, timeout=5)
            raise AssertionError("expected HTTP 304 for unchanged dashboard sources")
        except HTTPError as exc:
            assert exc.code == 304
            assert exc.headers["ETag"] == etag
            assert exc.read() == b""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_serves_background_store_without_request_time_scan(tmp_path, monkeypatch):
    store = DashboardStore(
        results_root=tmp_path,
        build_summary=lambda: {
            "schema_version": "test",
            "groups": [],
            "contracts": [],
            "plans": [],
            "schedulers": [],
            "flow": {"lanes": []},
            "summary": {},
            "evidence_feed": [],
        },
        source_revision=lambda: "r1",
    )
    store.refresh()
    monkeypatch.setattr(
        live_dashboard,
        "_dashboard_source_revision",
        lambda _root: (_ for _ in ()).throw(AssertionError("request-time scan")),
    )
    handler = type(
        "StoredDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "store": store,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=5) as response:
            payload = json.loads(response.read())

        assert payload["source_revision"] == "r1"
        assert payload["refreshing"] is False
        assert payload["refresh_error"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_scopes_summary_query_and_etag(tmp_path):
    store = DashboardStore(
        results_root=tmp_path,
        build_summary=lambda: {
            "schema_version": "test",
            "groups": [
                {"run_id": "run-new", "modules": [{"module": "aita"}]},
                {"run_id": "run-old", "modules": [{"module": "sus"}]},
            ],
            "contracts": [],
            "plans": [],
            "schedulers": [],
            "flow": {"lanes": []},
            "summary": {"latest_run_id": "run-new"},
            "evidence_feed": [],
        },
        source_revision=lambda: "r1",
    )
    store.refresh()
    handler = type(
        "ScopedStoredDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "store": store,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/runs?scope=latest", timeout=5) as response:
            latest_etag = response.headers["ETag"]
            latest = json.loads(response.read())
        with urlopen(f"http://127.0.0.1:{port}/api/runs?scope=run-old", timeout=5) as response:
            old_etag = response.headers["ETag"]
            old = json.loads(response.read())

        assert [group["run_id"] for group in latest["groups"]] == ["run-new"]
        assert [group["run_id"] for group in old["groups"]] == ["run-old"]
        assert latest_etag != old_etag
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_detail_endpoints_are_query_scoped_and_cacheable(tmp_path):
    _write_json(
        tmp_path / "run-1" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    )
    _append_events(
        tmp_path / "run-1" / "aita" / "RUN_EVENTS.jsonl",
        [{"timestamp": "2026-01-01T00:00:01+00:00", "event": "stage_started", "stage": "generation"}],
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {"options": DashboardOptions(results_root=tmp_path), "page_title": "Test", "poll_ms": 1234},
    )
    DashboardHandler._runs_cache.clear()
    DashboardHandler._runs_builds_in_progress.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/api/evidence?scope=run-1&stage=all&content=all&window=25"
        with urlopen(url, timeout=5) as response:
            etag = response.headers["ETag"]
            payload = json.loads(response.read())
            assert response.status == 200
            assert payload["resolved_scope"] == "run-1"
            assert len(payload["items"]) == 1

        request = Request(url, headers={"If-None-Match": etag})
        try:
            urlopen(request, timeout=5)
            raise AssertionError("expected HTTP 304 for unchanged scoped evidence")
        except HTTPError as exc:
            assert exc.code == 304
            assert exc.headers["ETag"] == etag
            assert exc.read() == b""

        with urlopen(f"http://127.0.0.1:{port}/api/contracts?scope=run-1", timeout=5) as response:
            contracts = json.loads(response.read())
            assert contracts["resolved_scope"] == "run-1"
            assert contracts["contracts"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_evidence_endpoint_etag_changes_when_ledger_appends(tmp_path):
    run_dir = tmp_path / "run-1" / "aita"
    status_path = run_dir / "RUN_STATUS.json"
    events_path = run_dir / "RUN_EVENTS.jsonl"
    _write_json(
        status_path,
        {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
            "updated_at": "2026-07-30T12:00:00+00:00",
        },
    )
    _append_events(
        events_path,
        [
            {
                "timestamp": "2026-07-30T12:00:00+00:00",
                "sequence": 1,
                "event": "stage_started",
                "stage": "generation",
            }
        ],
    )
    handler = type(
        "LiveUpdateDashboardHandler",
        (DashboardHandler,),
        {"options": DashboardOptions(results_root=tmp_path), "page_title": "Test", "poll_ms": 1234},
    )
    DashboardHandler._runs_cache.clear()
    DashboardHandler._runs_builds_in_progress.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_address[1]}"
            "/api/evidence?scope=run-1&stage=all&content=all&window=25"
        )
        with urlopen(url, timeout=5) as response:
            old_etag = response.headers["ETag"]
            old_payload = json.loads(response.read())

        with events_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": "2026-07-30T12:00:01+00:00",
                        "sequence": 2,
                        "event": "conversation_started",
                        "stage": "generation",
                        "model": "test-model",
                    }
                )
                + "\n"
            )
        _write_json(
            status_path,
            {
                "module": "aita",
                "stage": "generation",
                "status": "running",
                "validity": "not_score_ready",
                "updated_at": "2026-07-30T12:00:01+00:00",
            },
        )

        request = Request(url, headers={"If-None-Match": old_etag})
        with urlopen(request, timeout=5) as response:
            new_etag = response.headers["ETag"]
            new_payload = json.loads(response.read())

        assert new_etag != old_etag
        assert old_payload["total_count"] == 1
        assert new_payload["total_count"] == 2
        assert new_payload["items"][-1]["event"] == "conversation_started"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_serves_dashboard_assets(tmp_path):
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {"options": DashboardOptions(results_root=tmp_path), "page_title": "Test", "poll_ms": 1234},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/assets/anti-sycophancy-logo-running.gif", timeout=5) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/gif"
            assert body.startswith(b"GIF89a")
        with urlopen(f"http://127.0.0.1:{port}/assets/dashboard.js", timeout=5) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
            assert b"const pollMs = Number(document.body.dataset.pollMs || 2500)" in body
            assert b"__POLL_MS__" not in body
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["Content-Security-Policy"] == (
                "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; font-src 'self'; "
                "img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            )
        with urlopen(f"http://127.0.0.1:{port}/assets/theme-init.js", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
            assert b"benchmarkDashboardTheme" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_html_has_no_third_party_network_origins():
    page = render_html()

    assert "fonts.googleapis.com" not in page
    assert "fonts.gstatic.com" not in page
    assert "https://" not in page


def test_dashboard_handler_writes_disposition_sidecar(tmp_path):
    run_dir = tmp_path / "bad-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
            "failure_reason": "provider returned 429 before completion",
        },
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "local:test-operator",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "status_path": str(run_dir / "RUN_STATUS.json"),
                "disposition": "rejected_from_analysis",
                "reason": "openrouter_429_incomplete_generation",
                "notes": "Excluded from scored analysis; rerun with corrected pacing.",
            }
        ).encode()
        request = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Benchmark-CSRF": "test-csrf-token",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            payload = json.loads(response.read())

        disposition = json.loads((run_dir / "RUN_DISPOSITION.json").read_text())
        assert payload["ok"] is True
        assert disposition["disposition"] == "rejected_from_analysis"
        assert disposition["eligible_for_scoring"] is False
        assert disposition["decided_by"] == "local:test-operator"
        assert disposition["source_status"]["status"] == "failed_incomplete"
        events = [
            json.loads(line)
            for line in (run_dir / "RUN_DISPOSITION_EVENTS.jsonl").read_text().splitlines()
            if line
        ]
        assert events[-1]["operator_id"] == "local:test-operator"
        assert events[-1]["action"] == "reject"

        data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
        assert data["summary"]["attention_count"] == 0
        assert data["summary"]["rejected_count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_refuses_to_reject_completed_run(tmp_path):
    run_dir = tmp_path / "good-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "scoring",
            "status": "completed",
            "validity": "score_ready",
        },
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "local:test-operator",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/disposition",
            data=json.dumps(
                {
                    "status_path": str(run_dir / "RUN_STATUS.json"),
                    "disposition": "rejected_from_analysis",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Benchmark-CSRF": "test-csrf-token",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=5)
        assert exc_info.value.code == 400
        assert not (run_dir / "RUN_DISPOSITION.json").exists()
        assert not (run_dir / "RUN_DISPOSITION_EVENTS.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_accepts_display_safe_results_status_path(tmp_path):
    results_root = tmp_path / "results" / "testing"
    run_dir = results_root / "bad-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
        },
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=results_root),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "local:test-operator",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "status_path": "results/testing/bad-run/aita/RUN_STATUS.json",
                "disposition": "rejected_from_analysis",
            }
        ).encode()
        request = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Benchmark-CSRF": "test-csrf-token",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            payload = json.loads(response.read())

        disposition = json.loads((run_dir / "RUN_DISPOSITION.json").read_text())
        assert payload["ok"] is True
        assert disposition["disposition"] == "rejected_from_analysis"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_handler_rejects_non_local_host_and_wrong_content_type(tmp_path):
    run_dir = tmp_path / "bad-run" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
        },
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "local:test-operator",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "status_path": str(run_dir / "RUN_STATUS.json"),
                "disposition": "rejected_from_analysis",
                "reason": "csrf-probe",
            }
        ).encode()

        # DNS-rebinding style request: non-local Host header must be rejected.
        rebind = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={"Content-Type": "application/json", "Host": "evil.example.com"},
            method="POST",
        )
        try:
            urlopen(rebind, timeout=5)
            raise AssertionError("expected HTTP 403 for non-local Host")
        except HTTPError as exc:
            assert exc.code == 403

        # CSRF-style no-preflight POST (text/plain) must be rejected.
        csrf = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={"Content-Type": "text/plain", "X-Benchmark-CSRF": "test-csrf-token"},
            method="POST",
        )
        try:
            urlopen(csrf, timeout=5)
            raise AssertionError("expected HTTP 415 for non-JSON content type")
        except HTTPError as exc:
            assert exc.code == 415

        # Rebinding guard also covers the read API (full transcripts).
        rebind_get = Request(
            f"http://127.0.0.1:{port}/api/runs",
            headers={"Host": "evil.example.com"},
        )
        try:
            urlopen(rebind_get, timeout=5)
            raise AssertionError("expected HTTP 403 for non-local Host on /api/runs")
        except HTTPError as exc:
            assert exc.code == 403

        assert not (run_dir / "RUN_DISPOSITION.json").exists()

        # Favicon is served (was a 404 in every browser session).
        with urlopen(f"http://127.0.0.1:{port}/favicon.ico", timeout=5) as response:
            assert response.status == 200
            assert response.headers.get("Content-Type") == "image/png"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10"])
def test_run_server_rejects_non_loopback_before_store_or_bind(tmp_path, monkeypatch, host):
    def unexpected_store(*args, **kwargs):
        raise AssertionError("DashboardStore must not be created for a non-loopback bind")

    monkeypatch.setattr(live_dashboard, "DashboardStore", unexpected_store)

    with pytest.raises(ValueError, match="loopback"):
        run_server(
            host=host,
            port=0,
            options=DashboardOptions(results_root=tmp_path),
            title="Test",
            poll_ms=1000,
        )


@pytest.mark.parametrize("method", ["HEAD", "GET", "POST"])
def test_dashboard_rejects_nonlocal_host_for_every_method(tmp_path, method):
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "local:test-operator",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        request_kwargs = {"headers": {"Host": "evil.example"}, "method": method}
        if method == "POST":
            request_kwargs["data"] = b"{}"
        request = Request(f"http://127.0.0.1:{port}/api/runs", **request_kwargs)
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=5)
        assert exc_info.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_rejects_nonloopback_peer_address():
    handler = object.__new__(DashboardHandler)
    handler.client_address = ("192.0.2.10", 43210)

    assert handler._peer_is_local() is False


def test_disposition_requires_csrf_and_appends_attributed_event(tmp_path):
    run_dir = tmp_path / "run-1" / "aita"
    _write_json(
        run_dir / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "failed_incomplete",
            "validity": "not_score_ready",
        },
    )
    handler = type(
        "TestDashboardHandler",
        (DashboardHandler,),
        {
            "options": DashboardOptions(results_root=tmp_path),
            "page_title": "Test",
            "poll_ms": 1234,
            "csrf_token": "test-csrf-token",
            "operator_id": "reviewer-1",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "status_path": str(run_dir / "RUN_STATUS.json"),
                "disposition": "rejected_from_analysis",
            }
        ).encode()
        missing_csrf = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(missing_csrf, timeout=5)
        assert exc_info.value.code == 403

        accepted = Request(
            f"http://127.0.0.1:{port}/api/disposition",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Benchmark-CSRF": "test-csrf-token",
            },
            method="POST",
        )
        with urlopen(accepted, timeout=5) as response:
            assert response.status == 200

        events = [
            json.loads(line)
            for line in (run_dir / "RUN_DISPOSITION_EVENTS.jsonl").read_text().splitlines()
            if line
        ]
        assert events[-1]["operator_id"] == "reviewer-1"
        assert events[-1]["action"] == "reject"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_eligibility_is_derived_from_ledger_not_stale_sidecar(tmp_path):
    cases = {
        "rejected": ("failed_incomplete", "not_score_ready", "rejected_from_analysis"),
        "generation-complete": ("completed", "not_score_ready", "candidate"),
        "score-ready": ("completed", "score_ready", "candidate"),
    }
    for run_id, (status, validity, disposition) in cases.items():
        run_dir = tmp_path / run_id / "aita"
        _write_json(
            run_dir / "RUN_STATUS.json",
            {"module": "aita", "stage": "generation", "status": status, "validity": validity},
        )
        _write_json(
            run_dir / "RUN_DISPOSITION.json",
            {
                "schema_version": "benchmark-run-disposition-v1",
                "disposition": disposition,
                "eligible_for_generation": True,
                "eligible_for_scoring": True,
                "eligible_for_promotion": True,
            },
        )

    modules = {
        module["group"]: module
        for group in build_dashboard_data(DashboardOptions(results_root=tmp_path))["groups"]
        for module in group["modules"]
    }

    assert modules["rejected"]["eligible_for_generation"] is False
    assert modules["rejected"]["eligible_for_scoring"] is False
    assert modules["generation-complete"]["eligible_for_generation"] is True
    assert modules["generation-complete"]["eligible_for_scoring"] is False
    assert modules["score-ready"]["eligible_for_generation"] is True
    assert modules["score-ready"]["eligible_for_scoring"] is True


@pytest.mark.parametrize(
    ("status", "validity", "allowed"),
    [
        ("failed_incomplete", "not_score_ready", True),
        ("failed_invalid", "not_score_ready", True),
        ("failed_scoring", "not_score_ready", True),
        ("completed", "not_score_ready", False),
        ("completed", "score_ready", False),
        ("running", "not_score_ready", False),
        ("stopped", "not_score_ready", False),
    ],
)
def test_rejection_is_limited_to_failed_nonpublishable_runs(status, validity, allowed):
    assert live_dashboard._status_allows_analysis_rejection(
        {"status": status, "validity": validity}
    ) is allowed


def test_disposition_event_and_snapshot_updates_are_serialized(tmp_path, monkeypatch):
    run_dir = tmp_path / "run" / "aita"
    event_path = run_dir / "RUN_DISPOSITION_EVENTS.jsonl"
    snapshot_path = run_dir / "RUN_DISPOSITION.json"
    first_append_entered = threading.Event()
    release_first_append = threading.Event()
    real_append = live_dashboard._append_disposition_event
    append_count = 0
    count_lock = threading.Lock()

    def blocking_append(path, event):
        nonlocal append_count
        real_append(path, event)
        with count_lock:
            append_count += 1
            is_first = append_count == 1
        if is_first:
            first_append_entered.set()
            assert release_first_append.wait(timeout=5)

    monkeypatch.setattr(live_dashboard, "_append_disposition_event", blocking_append)

    first = threading.Thread(
        target=live_dashboard._persist_disposition_update,
        args=(event_path, {"action": "reject"}, snapshot_path, {"disposition": "rejected_from_analysis"}),
    )
    second = threading.Thread(
        target=live_dashboard._persist_disposition_update,
        args=(event_path, {"action": "restore"}, snapshot_path, {"disposition": "candidate"}),
    )
    first.start()
    assert first_append_entered.wait(timeout=5)
    second.start()
    release_first_append.set()
    first.join(timeout=5)
    second.join(timeout=5)

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    snapshot = json.loads(snapshot_path.read_text())
    assert events[-1]["action"] == "restore"
    assert snapshot["disposition"] == "candidate"


# --- Stable "latest" scope selection (Fix 1) ------------------------------


def test_select_latest_group_prefers_earliest_started_running_group():
    grouped = [
        {"run_id": "late", "running_count": 1, "started_at": "2026-07-16T10:00:00Z", "updated_at": "2026-07-16T12:00:05Z"},
        {"run_id": "early", "running_count": 2, "started_at": "2026-07-16T09:00:00Z", "updated_at": "2026-07-16T12:00:00Z"},
        {"run_id": "idle", "running_count": 0, "started_at": "2026-07-16T08:00:00Z", "updated_at": "2026-07-16T13:00:00Z"},
    ]
    # Even though 'late' updated most recently and 'idle' is newest overall,
    # the earliest-started RUNNING group is the sticky anchor.
    assert live_dashboard._select_latest_group(grouped)["run_id"] == "early"
    # Selection is order-independent (stable regardless of sort direction).
    assert live_dashboard._select_latest_group(list(reversed(grouped)))["run_id"] == "early"


def test_select_latest_group_tie_breaks_on_run_id():
    grouped = [
        {"run_id": "b-run", "running_count": 1, "started_at": "2026-07-16T09:00:00Z"},
        {"run_id": "a-run", "running_count": 1, "started_at": "2026-07-16T09:00:00Z"},
    ]
    assert live_dashboard._select_latest_group(grouped)["run_id"] == "a-run"


def test_select_latest_group_falls_back_to_recent_when_none_running():
    grouped = [
        {"run_id": "newest", "running_count": 0, "updated_at": "2026-07-16T13:00:00Z"},
        {"run_id": "older", "running_count": 0, "updated_at": "2026-07-16T10:00:00Z"},
    ]
    assert live_dashboard._select_latest_group(grouped)["run_id"] == "newest"
    assert live_dashboard._select_latest_group([]) is None


# --- Skip _archive dirs (Fix 2) -------------------------------------------


def test_path_has_archived_segment(tmp_path):
    root = tmp_path
    assert live_dashboard._path_has_archived_segment(
        root / "run-1" / "_archive_64k_cap_20260716" / "aita" / "RUN_STATUS.json", root
    )
    assert not live_dashboard._path_has_archived_segment(
        root / "run-1" / "aita" / "RUN_STATUS.json", root
    )


def test_build_dashboard_data_skips_archive_directories(tmp_path):
    _write_json(
        tmp_path / "run-1" / "aita" / "RUN_STATUS.json",
        {"module": "aita", "stage": "generation", "status": "completed", "validity": "score_ready"},
    )
    # A retired collection parked under _archive_* must not surface as a module.
    _write_json(
        tmp_path / "run-1" / "_archive_invalid_endpoint_20260716" / "aita" / "RUN_STATUS.json",
        {"module": "aita", "stage": "generation", "status": "failed", "validity": "not_score_ready"},
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert data["module_count"] == 1
    module_paths = [module["module_path"] for group in data["groups"] for module in group["modules"]]
    assert module_paths == ["aita"]
    assert not any("_archive" in path for path in module_paths)


# --- Logical-run family view (Fix 4) --------------------------------------


def test_prereg_families_requires_two_members():
    grouped = [
        {"run_id": "a", "prereg_sha256": "sha-shared"},
        {"run_id": "b", "prereg_sha256": "sha-shared"},
        {"run_id": "solo", "prereg_sha256": "sha-solo"},
        {"run_id": "none", "prereg_sha256": None},
    ]
    families = live_dashboard._prereg_families(grouped)
    assert len(families) == 1
    family = families[0]
    assert family["prereg_sha256"] == "sha-shared"
    assert family["key"] == "family:sha-shared"
    assert family["member_run_ids"] == ["a", "b"]
    assert family["member_count"] == 2


def test_build_dashboard_data_surfaces_prereg_family(tmp_path):
    sha = "f264eeae5a52f37b6b33d52266d7aaece9453af782796967acd3f51cfbafc5ed"
    for run_id in ("gpt56-sol", "gpt56-luna"):
        _write_json(tmp_path / run_id / "PREREG_FREEZE.json", {"prereg_sha256_current": sha})
        _write_json(
            tmp_path / run_id / "aita" / "RUN_STATUS.json",
            {"module": "aita", "stage": "generation", "status": "running", "validity": "not_score_ready"},
        )
    # A lone run with a different freeze must not form a family.
    _write_json(tmp_path / "gpt56-solo" / "PREREG_FREEZE.json", {"prereg_sha256_current": "other"})
    _write_json(
        tmp_path / "gpt56-solo" / "aita" / "RUN_STATUS.json",
        {"module": "aita", "stage": "generation", "status": "running", "validity": "not_score_ready"},
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))

    assert len(data["families"]) == 1
    assert data["families"][0]["prereg_sha256"] == sha
    assert data["families"][0]["member_run_ids"] == ["gpt56-luna", "gpt56-sol"]


def test_build_dashboard_data_surfaces_active_companion_workflow(tmp_path):
    results_root = tmp_path / "results" / "prepared"
    for run_id in ("run-a", "run-b"):
        _write_json(
            results_root / run_id / "sus" / "RUN_STATUS.json",
            {"module": "sus", "stage": "generation", "status": "running", "validity": "not_score_ready"},
        )
    companion_root = tmp_path / ".benchmark-companion"
    _write_json(companion_root / "ACTIVE.json", {"workflow_id": "current-collection"})
    _write_json(
        companion_root / "current-collection" / "RESUME.json",
        {
            "workflow_id": "current-collection",
            "runs": [
                {"run_id": "run-a", "run_dir": str(results_root / "run-a" / "sus")},
                {"run_id": "run-b", "run_dir": str(results_root / "run-b" / "sus")},
            ],
        },
    )

    data = build_dashboard_data(DashboardOptions(results_root=results_root))

    assert data["workflows"] == [
        {
            "key": "workflow:active",
            "workflow_id": "current-collection",
            "member_run_ids": ["run-a", "run-b"],
            "member_count": 2,
        }
    ]
    summary = live_dashboard._dashboard_summary_data(data, scope="workflow:active")
    assert sorted(group["run_id"] for group in summary["groups"]) == ["run-a", "run-b"]


def test_workflow_evidence_detail_walks_only_member_runs(tmp_path, monkeypatch):
    for run_id in ("run-a", "run-b", "outside"):
        run_dir = tmp_path / run_id / "sus"
        _write_json(
            run_dir / "RUN_STATUS.json",
            {"module": "sus", "stage": "generation", "status": "running", "validity": "not_score_ready"},
        )
        _append_events(
            run_dir / "RUN_EVENTS.jsonl",
            [{"timestamp": "2026-01-01T00:00:01+00:00", "event": "stage_completed"}],
        )
    visited = []
    original = live_dashboard._evidence_items_from_module

    def counted(*args, **kwargs):
        visited.append(kwargs["output_dir"])
        return original(*args, **kwargs)

    monkeypatch.setattr(live_dashboard, "_evidence_items_from_module", counted)
    summary_data = {
        "summary": {"latest_run_id": "outside"},
        "workflows": [
            {
                "key": "workflow:active",
                "workflow_id": "collection",
                "member_run_ids": ["run-a", "run-b"],
                "member_count": 2,
            }
        ],
    }

    payload = live_dashboard._build_dashboard_evidence_detail(
        summary_data,
        results_root=tmp_path,
        scope="workflow:active",
        stage="all",
        content="all",
        window="100",
        module="",
    )

    assert set(visited) == {tmp_path / "run-a" / "sus", tmp_path / "run-b" / "sus"}
    assert {item["group"] for item in payload["items"]} == {"run-a", "run-b"}


def test_dashboard_summary_family_scope_matches_all_members(tmp_path):
    sha = "shared-sha-value"
    for run_id in ("run-a", "run-b"):
        _write_json(tmp_path / run_id / "PREREG_FREEZE.json", {"prereg_sha256_current": sha})
        _write_json(
            tmp_path / run_id / "aita" / "RUN_STATUS.json",
            {"module": "aita", "stage": "generation", "status": "running", "validity": "not_score_ready"},
        )
    _write_json(tmp_path / "run-c" / "PREREG_FREEZE.json", {"prereg_sha256_current": "different"})
    _write_json(
        tmp_path / "run-c" / "aita" / "RUN_STATUS.json",
        {"module": "aita", "stage": "generation", "status": "running", "validity": "not_score_ready"},
    )

    data = build_dashboard_data(DashboardOptions(results_root=tmp_path))
    summary = live_dashboard._dashboard_summary_data(data, scope=f"family:{sha}")

    assert summary["resolved_scope"] == f"family:{sha}"
    scoped_run_ids = sorted(group["run_id"] for group in summary["groups"])
    assert scoped_run_ids == ["run-a", "run-b"]


def test_event_progress_counts_reused_units_once():
    from suite_tools.live_dashboard import _event_progress
    events = [{"event": "conversation_completed", "unit_id": f"u{i}", "attempt_number": 1, "planned_turns": 5}
              for i in range(3)]
    events += [{"event": "conversation_reused", "unit_id": f"u{i}", "attempt_number": 2} for i in range(3)]
    events += [{"event": "conversation_completed", "unit_id": f"u{i}", "attempt_number": 2} for i in range(3, 8)]
    assert _event_progress(events, {"status": "running"})["conversations_completed"] == 8


def test_run_server_exits_clearly_when_port_already_in_use(tmp_path):
    """run_server must fail fast with a clear error suggesting --port when another
    process holds the port.  A silent bind (or an error with no actionable advice)
    caused the cold-start simulation to receive the wrong application's HTML with
    no warning — the user discovered the conflict only after inspecting page content.
    """
    # Grab a free OS port and keep it occupied during the test.
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupant.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupant.bind(("127.0.0.1", 0))
    occupied_port = occupant.getsockname()[1]
    occupant.listen(1)
    try:
        import pytest

        with pytest.raises(SystemExit) as exc_info:
            run_server(
                host="127.0.0.1",
                port=occupied_port,
                options=DashboardOptions(results_root=tmp_path),
                title="Test",
                poll_ms=2500,
            )
        # The error message must tell the user how to resolve the conflict.
        error_text = str(exc_info.value)
        assert "--port" in error_text, f"Expected '--port' hint in error: {error_text!r}"
    finally:
        occupant.close()


def test_dashboard_rejects_lifecycle_parent_as_results_root(tmp_path):
    broad_root = tmp_path / "results"
    _write_json(
        broad_root / "prepared" / "run-1" / "aita" / "RUN_STATUS.json",
        {
            "module": "aita",
            "stage": "generation",
            "status": "running",
            "validity": "not_score_ready",
        },
    )

    error = live_dashboard._dashboard_results_root_error(broad_root)

    assert error is not None
    assert "results/prepared" in error
    assert "pseudo-runs" in error
