"""Integration tests for epistemic sycophancy runner."""

from __future__ import annotations

import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from epis_bench import cli as epis_cli
from epis_bench import runner
from epis_bench.runner import load_items, prepare_side_item, run_conversation
from suite_tools.call_diagnostics import diagnose_call_journal
from suite_tools.provider_client import (
    ProviderApiError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
)
from suite_tools.paid_call_lease import set_paid_call_policy
from suite_tools.run_contract import STOP_BEFORE_NEXT_PAID_CALL, write_run_control
from suite_tools.run_monitor import RunMonitor


def test_cli_rejects_literal_api_keys_and_accepts_environment_variable_names(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        ["epis-bench", "run", "--api-key", "literal-secret"],
    )
    with pytest.raises(SystemExit):
        epis_cli.main()

    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args))
    monkeypatch.setattr(
        "sys.argv",
        [
            "epis-bench",
            "run",
            "--output",
            str(tmp_path),
            "--api-key-env",
            "LOCAL_OPENAI_COMPATIBLE_API_KEY",
        ],
    )
    epis_cli.main()
    assert captured[0].api_key_env == "LOCAL_OPENAI_COMPATIBLE_API_KEY"
    assert not hasattr(captured[0], "api_key")


def test_runner_resolves_named_loopback_credential_without_creating_a_client(monkeypatch):
    monkeypatch.setenv("LOCAL_OPENAI_COMPATIBLE_API_KEY", "adapter-test-key")

    value, env_name, explicit = runner._argument_credential(
        SimpleNamespace(api_key_env="LOCAL_OPENAI_COMPATIBLE_API_KEY"),
        base_url="http://127.0.0.1:9999/v1",
    )

    assert value == "adapter-test-key"
    assert env_name == "LOCAL_OPENAI_COMPATIBLE_API_KEY"
    assert explicit is False


def test_run_rejects_prepared_config_drift_before_key_or_provider_preflight(
    tmp_path,
    monkeypatch,
):
    from suite_tools.run_contract import PreparedConfigProvenanceError

    monkeypatch.setattr(
        runner,
        "validate_run_prepared_config_before_spend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PreparedConfigProvenanceError("digest changed")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_preflight_openrouter_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("key preflight must not run")
        ),
    )
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.run(SimpleNamespace(
            output=str(tmp_path / "prepared" / "epis"),
            config=str(tmp_path / "prepared" / "_configs" / "models.yaml"),
            model=None,
            models="all",
            base_url=None,
        ))

    assert exc_info.value.code == 2
    status = json.loads(
        (tmp_path / "prepared" / "epis" / "RUN_STATUS.json").read_text()
    )
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_prepared_epistemic_unit_drift_before_key_preflight(
    tmp_path,
    monkeypatch,
):
    from suite_tools.prepare_run import prepare_epis_run

    run_group = tmp_path / "prepared"
    contract_path = prepare_epis_run(
        run_id="epis-unit-drift",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items=1,
        types="delusion,pickside,mirror",
        selection=str(
            Path(__file__).resolve().parents[1] / "data" / "calibration-selection.yaml"
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_items",
        lambda *_args, **_kwargs: {"delusion": [{"statement": "changed"}]},
    )
    monkeypatch.setattr(
        runner,
        "_preflight_openrouter_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("key preflight must not run")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.run(SimpleNamespace(
            output=str(contract_path.parent),
            config=str(run_group / "_configs" / "calibration" / "epis-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            types="delusion,pickside,mirror",
            items=1,
            data_dir=None,
            selection=None,
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_missing_preflight_receipt_before_key_or_client(
    tmp_path,
    monkeypatch,
):
    from suite_tools.prepare_run import prepare_epis_run

    run_group = tmp_path / "prepared"
    selection = Path(__file__).resolve().parents[1] / "data" / "calibration-selection.yaml"
    contract_path = prepare_epis_run(
        run_id="epis-missing-preflight",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items=1,
        types="delusion,pickside,mirror",
        selection=str(selection),
    )
    monkeypatch.setattr(
        runner,
        "_preflight_openrouter_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("key preflight must not run")
        ),
    )
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.run(SimpleNamespace(
            output=str(contract_path.parent),
            config=str(run_group / "_configs" / "calibration" / "epis-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            types="delusion,pickside,mirror",
            items=1,
            data_dir=None,
            selection=str(selection),
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "preflight_receipt_admission"


@pytest.mark.parametrize("model", [None, "unknown/model"])
def test_runtime_base_url_cannot_send_openrouter_key_to_custom_remote_host(
    tmp_path,
    monkeypatch,
    model,
):
    monkeypatch.setattr(
        runner,
        "load_models",
        lambda _config: {
            "loaded": {
                "model_id": "provider/model",
                "label": "Loaded",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "api_key": "operator-openrouter-key",
                "max_parallel": 1,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "load_items",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("data/provider setup must not continue")
        ),
    )
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created")
        ),
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY.*refusing"):
        runner.run(SimpleNamespace(
            output=str(tmp_path / "run"),
            config="models.yaml",
            model=model,
            models="all",
            base_url="https://attacker.example/v1",
            api_key=None,
            types="delusion",
            items=1,
            data_dir=None,
            selection=None,
        ))

SYCO_BENCH_PATH = Path(__file__).parent.parent / "data" / "syco_bench"
SYCO_BENCH_AVAILABLE = SYCO_BENCH_PATH.exists()

# Pinned sha256 of the vendored Syco-Bench CSVs (upstream MIT-0, commit
# 5ecc005b2a8cd429c5073763cd4714043123c2cd). Items are selected by CSV row
# index, so any silent edit or reorder of these files corrupts item identity.
VENDORED_CSV_SHA256 = {
    "delusion.csv": "fbf614631b4a6202649554205ada62e7cf9cd12d0d1ce019ad683c82cfa7d236",
    "pickside.csv": "128633e1c7090c1b6ab36abaee4a6275698d6967a23f350e0112ecd0e8d9423b",
    "mirror.csv": "656647d1101984216d8c19bd65990f0d778880e673cc956ef30761420c2a4186",
}


def test_generation_workers_cannot_exceed_authoritative_global_limit(tmp_path, monkeypatch):
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    set_paid_call_policy(1, lease_dir=lease_dir)
    active = 0
    max_seen = 0
    lock = threading.Lock()

    def fake_conversation(*args, **kwargs):
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"completed": True, "turns": []}

    monkeypatch.setattr(runner, "run_conversation", fake_conversation)
    runner.run_model_all_items(
        "model",
        {"delusion": [{"statement": "a"}, {"statement": "b"}]},
        tmp_path,
        "client",
        {"model": {"label": "Model", "max_parallel": 8}},
    )

    assert max_seen == 1


def test_preflight_openrouter_key_raises_on_empty(monkeypatch):
    monkeypatch.setattr(runner, "OPENROUTER_KEY", "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(runner, "load_repo_env_files", lambda: None)
    with pytest.raises(SystemExit) as exc:
        runner._preflight_openrouter_key("")
    assert exc.value.code == 1


def test_preflight_openrouter_key_error_names_correct_var(monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_repo_env_files", lambda: None)
    with pytest.raises(SystemExit):
        runner._preflight_openrouter_key("")
    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err


def test_preflight_openrouter_key_passes_with_valid_key():
    # Should not raise
    runner._preflight_openrouter_key("sk-" + "or-v1-fake-key")


def test_empty_generation_batch_returns_before_resolving_parallelism(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "effective_paid_call_parallelism",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("empty work must not resolve paid-call parallelism")
        ),
    )

    assert runner.run_model_all_items(
        "model",
        {},
        tmp_path,
        "client",
        {"model": {"label": "Model", "max_parallel": 8}},
    ) == []


class TestVendoredSycoBenchData:
    def test_default_data_dir_is_vendored_inside_the_repo(self):
        bench_root = Path(runner.__file__).resolve().parent.parent
        data_dir = runner.SYCO_BENCH_DATA.resolve()
        assert data_dir.is_relative_to(bench_root), (
            "Epis source items must be vendored in-repo, not loaded from an "
            "unpinned external clone (row-index selection breaks on upstream "
            "reorders)."
        )

    def test_vendored_csvs_exist_and_match_pinned_hashes(self):
        import hashlib

        for name, expected in VENDORED_CSV_SHA256.items():
            path = runner.SYCO_BENCH_DATA / name
            assert path.exists(), f"vendored CSV missing: {path}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected, (
                f"{name} drifted from the pinned upstream content; "
                "row-index item selection is no longer trustworthy"
            )

    def test_vendored_data_carries_provenance_and_license(self):
        provenance = runner.SYCO_BENCH_DATA / "PROVENANCE.md"
        assert provenance.exists()
        text = provenance.read_text()
        assert "5ecc005b2a8cd429c5073763cd4714043123c2cd" in text
        assert "timfduffy/syco-bench" in text
        assert (runner.SYCO_BENCH_DATA / "LICENSE").exists()

    def test_load_items_uses_vendored_data_by_default(self):
        items = runner.load_items(
            ["delusion", "pickside", "mirror"],
            items_per_type=4,
            selection_path=str(
                Path(runner.__file__).resolve().parent.parent / "data" / "selection.yaml"
            ),
        )
        assert {len(v) for v in items.values()} == {4}


def test_run_generation_completion_waits_for_scoring(tmp_path, monkeypatch):
    models = {
        "gemini-flash": {
            "model_id": "google/gemini-3-flash-preview",
            "label": "Gemini 3 Flash",
            "base_url": "https://openrouter.ai/api/v1",
            "max_parallel": 1,
        }
    }
    monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(runner, "load_models", lambda config: dict(models))
    monkeypatch.setattr(
        runner,
        "load_items",
        lambda test_types, items, data_dir=None, selection_path=None: {
            "delusion": [{"statement": "The moon is made of cheese."}]
        },
    )
    monkeypatch.setattr(
        runner,
        "run_model_all_items",
        lambda *args, **kwargs: [{"completed": True}],
    )
    monkeypatch.setattr(runner, "find_incomplete_conversations", lambda output_dir, **kwargs: [])

    runner.run(SimpleNamespace(
        config="missing.yaml",
        output=str(tmp_path),
        types="delusion",
        items=1,
        data_dir=None,
        selection=None,
        models="gemini-flash",
        model=None,
        base_url=None,
        api_key="fake",
    ))

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]

    assert status["status"] == "completed"
    assert status["stage"] == "generation"
    assert status["validity"] == "not_score_ready"
    assert events[-1]["event"] == "stage_completed"
    assert events[-1]["validity"] == "not_score_ready"


def test_parallel_model_batches_respect_global_worker_limit(tmp_path, monkeypatch):
    models = {
        key: {
            "model_id": f"test/{key}",
            "label": key,
            "base_url": "https://openrouter.ai/api/v1",
            "max_parallel": 4,
        }
        for key in ("one", "two", "three")
    }
    set_paid_call_policy(1, lease_dir=tmp_path / "leases")
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(runner, "load_models", lambda config: dict(models))
    monkeypatch.setattr(
        runner,
        "load_items",
        lambda *args, **kwargs: {"delusion": [{"statement": "test"}]},
    )
    active = 0
    max_seen = 0
    guard = threading.Lock()

    def fake_model_batch(*args, **kwargs):
        nonlocal active, max_seen
        with guard:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return []

    monkeypatch.setattr(runner, "run_model_all_items", fake_model_batch)
    monkeypatch.setattr(runner, "find_incomplete_conversations", lambda *args, **kwargs: [])

    runner.run(SimpleNamespace(
        config="missing.yaml", output=str(tmp_path / "run"), types="delusion",
        items=1, data_dir=None, selection=None, models="all", model=None,
        base_url=None, api_key="fake",
    ))

    assert max_seen == 1


def test_run_conversation_records_provider_failure_reason(tmp_path, monkeypatch):
    class Provider502(Exception):
        status_code = 502

        def __str__(self):
            return "provider returned 502 before first model turn"

    class FakeCompletions:
        def create(self, *args, **kwargs):
            raise Provider502()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    monitor = RunMonitor(tmp_path, module="epistemic", stage="generation")

    with pytest.raises(RuntimeError, match="provider returned 502"):
        runner.run_conversation(
            "gemini-flash",
            {"statement": "The moon is made of cheese."},
            0,
            "delusion",
            "side_a",
            tmp_path,
            MagicMock(),
            {
                "gemini-flash": {
                    "label": "Gemini 3 Flash",
                    "model_id": "google/gemini-3-flash-preview",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "fake",
                    "max_parallel": 1,
                }
            },
            monitor,
        )

    transcript = json.loads((tmp_path / "gemini-flash_item0_delusion_side_a.json").read_text())
    assert transcript["completed"] is False
    assert transcript["actual_num_turns"] == 0
    assert transcript["failure_stage"] == "model"
    assert transcript["failure_turn"] == 1
    assert transcript["failure_reason"] == "provider returned 502 before first model turn"
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    failed = [event for event in events if event["event"] == "conversation_failed"]
    assert failed[-1]["failure_status"] == "failed_provider"
    assert failed[-1]["failure_stage"] == "model"


@pytest.mark.skipif(not SYCO_BENCH_AVAILABLE, reason="Syco-Bench data not found")
class TestLoadItems:
    def test_loads_all_types(self):
        items = load_items(["delusion", "pickside", "mirror"], items_per_type=2, data_dir=SYCO_BENCH_PATH)
        assert "delusion" in items
        assert "pickside" in items
        assert "mirror" in items

    def test_correct_count(self):
        items = load_items(["delusion", "pickside", "mirror"], items_per_type=2, data_dir=SYCO_BENCH_PATH)
        assert len(items["delusion"]) == 2
        assert len(items["pickside"]) == 2

        assert len(items["mirror"]) == 2

    def test_delusion_has_statement(self):
        items = load_items(["delusion"], items_per_type=1, data_dir=SYCO_BENCH_PATH)
        assert "statement" in items["delusion"][0]

    def test_pickside_has_columns(self):
        items = load_items(["pickside"], items_per_type=1, data_dir=SYCO_BENCH_PATH)
        item = items["pickside"][0]
        assert "statement1" in item
        assert "statement2" in item

    def test_mirror_has_columns(self):
        items = load_items(["mirror"], items_per_type=1, data_dir=SYCO_BENCH_PATH)
        item = items["mirror"][0]
        for key in ("topic", "stance1", "stance2", "userstance1", "userstance2"):
            assert key in item, f"Missing key: {key}"

    def test_load_correct_max_count(self):
        items = load_items(["delusion"], items_per_type=100, data_dir=SYCO_BENCH_PATH)
        assert len(items["delusion"]) == 20

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown test type"):
            load_items(["nonexistent"], data_dir=SYCO_BENCH_PATH)


class TestModelDefaults:
    def test_default_models_use_current_claude_slugs(self, tmp_path):
        models = runner.load_models(str(tmp_path / "missing.yaml"))

        assert models["sonnet-4-6"]["model_id"] == "anthropic/claude-sonnet-4.6"
        assert models["opus-4-6"]["model_id"] == "anthropic/claude-opus-4.6"
        assert all(model["condition_id"] for model in models.values())
        assert all(model["condition_hash"].startswith("sha256:") for model in models.values())
        assert all(model["route_hash"].startswith("sha256:") for model in models.values())

    def test_custom_target_key_is_not_reused_for_openrouter_support(
        self,
        monkeypatch,
    ):
        support_clients = []
        monkeypatch.setattr(runner, "_openrouter_key", lambda: "support-openrouter-key")
        monkeypatch.setattr(runner, "_preflight_openrouter_key", lambda key: None)
        monkeypatch.setattr(
            runner,
            "OpenAI",
            lambda **kwargs: support_clients.append(kwargs) or SimpleNamespace(),
        )

        runner._openrouter_support_client()

        assert support_clients == [{
            "api_key": "support-openrouter-key",
            "base_url": "https://openrouter.ai/api/v1",
        }]

    def test_load_models_discovers_repo_env_key_after_import(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(runner, "OPENROUTER_KEY", "")

        def fake_load_env():
            os.environ.setdefault("OPENROUTER_API_KEY", "key-from-env-file")

        monkeypatch.setattr(runner, "load_repo_env_files", fake_load_env)
        config_path = tmp_path / "models.yaml"
        config_path.write_text(
            "\n".join([
                "models:",
                "  gemini-flash:",
                "    model_id: google/gemini-3-flash-preview",
            ])
        )

        models = runner.load_models(config_path)

        assert runner.OPENROUTER_KEY == "key-from-env-file"
        assert models["gemini-flash"]["api_key"] == "key-from-env-file"

    def test_load_models_preserves_native_provider_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setattr(runner, "load_repo_env_files", lambda: None)
        config_path = tmp_path / "models.yaml"
        config_path.write_text(
            "\n".join([
                "models:",
                "  opus-native:",
                "    model_id: claude-opus-4-8",
                "    label: Opus Native",
                "    base_url: https://api.anthropic.com/v1/messages",
                "    provider_api: anthropic_messages",
                "    api_key_env: ANTHROPIC_API_KEY",
                "    request_options:",
                "      output_config:",
                "        effort: high",
            ])
        )

        models = runner.load_models(config_path)

        assert models["opus-native"]["provider_api"] == "anthropic_messages"
        assert models["opus-native"]["base_url"] == "https://api.anthropic.com/v1/messages"
        assert models["opus-native"]["api_key"] == "anthropic-key"
        assert models["opus-native"]["request_options"]["output_config"]["effort"] == "high"

    def test_load_models_preserves_direct_judge_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setattr(runner, "load_repo_env_files", lambda: None)
        config_path = tmp_path / "models.yaml"
        config_path.write_text(
            "\n".join([
                "judge:",
                "  model_id: gpt-5.5",
                "  primary_config:",
                "    model_id: gpt-5.5",
                "    base_url: https://api.openai.com/v1",
                "    provider_api: openai_compatible",
                "    api_key_env: OPENAI_API_KEY",
                "    request_options:",
                "      reasoning_effort: high",
                "models:",
                "  gemini-flash:",
                "    model_id: google/gemini-3-flash-preview",
            ])
        )

        runner.load_models(config_path)

        assert runner.JUDGE_MODEL == "gpt-5.5"
        assert runner.JUDGE_CONFIG["base_url"] == "https://api.openai.com/v1"
        assert runner.JUDGE_CONFIG["provider_api"] == "openai_compatible"
        assert runner.JUDGE_CONFIG["api_key_env"] == "OPENAI_API_KEY"
        assert runner.JUDGE_CONFIG["request_options"]["reasoning_effort"] == "high"

    def test_direct_openai_judge_call_uses_supported_payload_shape(self):
        from epis_bench import scoring

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="2"))])

        # Real OpenAI SDK clients expose base_url on the client object, not on
        # chat.completions; the fake must match so the production path is hit.
        client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            chat=SimpleNamespace(completions=FakeCompletions()),
        )
        scoring.set_judge_request_options("gpt-5.5", {"reasoning_effort": "high"})

        assert scoring._call_judge(client, "gpt-5.5", "score this", max_score=2, max_tokens=128) == 2
        assert "temperature" not in captured
        assert "max_tokens" not in captured
        assert captured["max_completion_tokens"] == 128
        assert captured["extra_body"]["reasoning_effort"] == "high"

    def test_direct_openai_judge_reuses_configured_token_cap(self):
        from epis_bench import scoring

        captured = []

        class FakeCompletions:
            def create(self, **kwargs):
                captured.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="2"))]
                )

        client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            chat=SimpleNamespace(completions=FakeCompletions()),
        )
        request_options = {"max_tokens": 8192, "reasoning_effort": "high"}
        scoring.set_judge_request_options("gpt-5.5", request_options)

        try:
            for _ in range(2):
                assert (
                    scoring._call_judge(
                        client,
                        "gpt-5.5",
                        "score this",
                        max_score=2,
                        max_tokens=128,
                    )
                    == 2
                )
        finally:
            scoring.set_judge_request_options("gpt-5.5", None)

        assert [call["max_completion_tokens"] for call in captured] == [8192, 8192]
        assert request_options == {"max_tokens": 8192, "reasoning_effort": "high"}

    def test_panel_aggregation_preserves_per_judge_scores(self):
        judge_specs = [
            {"model_id": "judge-a", "config": {"condition_id": "a"}},
            {"model_id": "judge-b", "config": {"condition_id": "b"}},
            {"model_id": "judge-c", "config": {"condition_id": "c"}},
        ]
        aggregate = runner._aggregate_panel_scores(
            [
                {"persistence": 1, "integrity": 2, "missing_scores": []},
                {"persistence": 0, "integrity": 0, "missing_scores": []},
                {"persistence": 0, "integrity": 1, "missing_scores": []},
            ],
            judge_specs=judge_specs,
        )

        assert aggregate["persistence"] == 0.333
        assert aggregate["integrity"] == 1
        assert aggregate["endpoint_shift"] is True
        assert aggregate["integrity_break"] is False
        assert aggregate["primary_failure"] is True
        assert aggregate["panel_failure_votes"]["hard_failure"]["failed"] == 2
        assert aggregate["judge_panel"] == ["judge-a", "judge-b", "judge-c"]
        assert aggregate["judge_configs"] == [{"condition_id": "a"}, {"condition_id": "b"}, {"condition_id": "c"}]
        assert len(aggregate["judge_scores"]) == 3
        assert aggregate["judge_aggregation"]["method"] == "full_judge_panel_mean"

    def test_panel_aggregation_rejects_incomplete_panel(self):
        judge_specs = [
            {"model_id": "judge-a", "config": {"condition_id": "a"}},
            {"model_id": "judge-b", "config": {"condition_id": "b"}},
        ]

        with pytest.raises(runner.JudgePanelIncompleteError) as exc:
            runner._aggregate_panel_scores(
                [
                    {
                        "persistence": 1,
                        "integrity": 2,
                        "missing_scores": [],
                    }
                ],
                judge_specs=judge_specs,
                expected_dimensions=["persistence", "integrity"],
                item_key="test-model_item0_delusion",
            )

        payload = exc.value.to_status_payload()
        assert payload["judge_panel_complete"] is False
        assert payload["expected_judges"] == ["judge-a", "judge-b"]
        assert payload["successful_judges"] == ["judge-a"]
        assert payload["missing_judges"] == ["judge-b"]
        assert payload["rerun_recommended"] is True


class TestAdapterIntegrity:
    def test_api_call_raises_on_adapter_integrity_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception(
            "Error code: 502 - {'error': 'Adapter rejected malformed backend response wrapper'}"
        )

        with pytest.raises(runner.AdapterIntegrityError):
            runner.api_call(client, "private-endpoint/harness-gemini-3-1-pro", [])

    def test_api_call_raises_on_generic_backend_error_text(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        client = MagicMock()
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content="I apologize, but I encountered an error processing your message. Please try again."
                )
            )
        ]
        client.chat.completions.create.return_value = response

        with pytest.raises(runner.AdapterIntegrityError, match="benchmark-invalid error text"):
            runner.api_call(client, "private-endpoint/harness-gemini-3-1-pro", [], retries=1)

    def test_api_call_does_not_retry_invalid_model_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

        class BadRequest(Exception):
            status_code = 400

            def __str__(self):
                return "openrouter/invalid is not a valid model ID"

        client = MagicMock()
        client.chat.completions.create.side_effect = BadRequest()
        sleep = MagicMock()
        monkeypatch.setattr(runner.time, "sleep", sleep)

        with pytest.raises(runner.FatalBenchmarkApiError, match="not a valid model ID"):
            runner.api_call(client, "openrouter/invalid", [], retries=3)

        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()

    def test_api_call_does_not_retry_provider_refusal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        client = MagicMock()
        client.chat.completions.create.side_effect = ProviderRefusalError(
            "Anthropic native provider refusal; stop_reason=refusal"
        )
        sleep = MagicMock()
        monkeypatch.setattr(runner.time, "sleep", sleep)

        with pytest.raises(ProviderRefusalError, match="stop_reason=refusal"):
            runner.api_call(client, "anthropic/claude-fable-5", [], retries=3)

        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()

    def test_api_call_passes_request_options_as_extra_body(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        content = runner.api_call(
            client,
            "anthropic/claude-opus-4.8",
            [{"role": "user", "content": "hello"}],
            request_options={
                "reasoning": {"enabled": True, "exclude": True},
                "verbosity": "high",
            },
        )

        assert content == "ok"
        assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
            "reasoning": {"enabled": True, "exclude": True},
            "verbosity": "high",
        }

    def test_api_call_reuses_direct_openai_request_options_without_losing_cap(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        captured = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                captured.append(kwargs)
                return response

        client = SimpleNamespace(
            base_url="https://api.openai.com/v1/responses",
            chat=SimpleNamespace(completions=FakeCompletions()),
        )
        request_options = {"max_tokens": 128000, "reasoning_effort": "max"}
        monitor = runner.RunMonitor(
            tmp_path / "run",
            module="epistemic",
            stage="generation",
        )

        for _ in range(2):
            assert (
                runner.api_call(
                    client,
                    "gpt-5.6-sol",
                    [{"role": "user", "content": "hello"}],
                    max_tokens=1000,
                    retries=1,
                    monitor=monitor,
                    role="model_under_test",
                    request_options=request_options,
                    request_context={
                        "condition_id": "gpt-5-6-sol-openai-native-max",
                        "model_key": "gpt-5-6-sol-native-max",
                    },
                )
                == "ok"
            )

        assert [call["max_completion_tokens"] for call in captured] == [
            128000,
            128000,
        ]
        assert request_options == {
            "max_tokens": 128000,
            "reasoning_effort": "max",
        }
        receipts = [
            json.loads(line)
            for line in monitor.events_path.read_text().splitlines()
            if '"event": "effective_request"' in line
        ]
        assert [receipt["effective_max_output_tokens"] for receipt in receipts] == [
            128000,
            128000,
        ]
        assert {receipt["condition_id"] for receipt in receipts} == {
            "gpt-5-6-sol-openai-native-max"
        }
        diagnostics = diagnose_call_journal(tmp_path / "run")
        assert diagnostics["attempt_count"] == 2
        assert diagnostics["closed_count"] == 2
        assert diagnostics["failure_count"] == 0

    def test_api_call_generation_timeout_defaults_above_adapter_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.delenv("BENCHMARK_GENERATION_TIMEOUT_SECONDS", raising=False)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        assert runner.api_call(client, "private-endpoint/harness", [], retries=1) == "ok"

        assert client.chat.completions.create.call_args.kwargs["timeout"] == 150

    def test_api_call_generation_timeout_can_be_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_GENERATION_TIMEOUT_SECONDS", "180")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        assert runner.api_call(client, "private-endpoint/harness", [], retries=1) == "ok"

        assert client.chat.completions.create.call_args.kwargs["timeout"] == 180

    def test_api_call_normalizes_direct_openai_gpt5_token_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        captured = {}
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return response

        client = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            chat=SimpleNamespace(completions=FakeCompletions()),
        )

        content = runner.api_call(
            client,
            "gpt-5.5",
            [{"role": "user", "content": "hello"}],
            max_tokens=321,
            retries=1,
        )

        assert content == "ok"
        assert captured["max_completion_tokens"] == 321
        assert "max_tokens" not in captured

    def test_make_client_disables_sdk_retries(self, monkeypatch):
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return "client"

        monkeypatch.setattr(runner, "OpenAI", fake_openai)

        assert runner.make_client({"api_key": "key", "base_url": "http://localhost:9999/v1"}) == "client"
        assert captured["max_retries"] == 0

    def test_configured_custom_key_never_falls_back_to_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-leak")
        monkeypatch.delenv("CUSTOM_MODEL_KEY", raising=False)
        monkeypatch.setenv("BENCHMARK_ALLOWED_ENDPOINT_HOSTS", "models.example")

        with pytest.raises(ValueError, match=r"\$CUSTOM_MODEL_KEY"):
            runner._api_key_for_config(
                {
                    "api_key_env": "CUSTOM_MODEL_KEY",
                    "base_url": "https://models.example/v1",
                }
            )

    def test_run_conversation_obeys_control_stop_before_paid_call(self, tmp_path, monkeypatch):
        write_run_control(tmp_path, action=STOP_BEFORE_NEXT_PAID_CALL, reason="operator stop")
        monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")
        api_call = MagicMock(side_effect=AssertionError("paid call should not run"))
        monkeypatch.setattr(runner, "api_call", api_call)

        with pytest.raises(runner.RunControlStopRequested):
            run_conversation(
                "gemini-flash",
                {"statement": "test"},
                0,
                "delusion",
                "side_a",
                tmp_path,
                MagicMock(),
                {
                    "gemini-flash": {
                        "model_id": "google/gemini-3-flash-preview",
                        "label": "Gemini Flash",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_key": "fake",
                        "max_parallel": 1,
                    }
                },
            )

        api_call.assert_not_called()

    def test_run_conversation_checks_payload_blindness_before_paid_call(self, tmp_path, monkeypatch):
        events = []
        monitor = SimpleNamespace(record=lambda event, **fields: events.append(event))
        monkeypatch.setattr(runner, "NUM_TURNS", {"delusion": 1})
        monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")
        monkeypatch.setattr(
            runner,
            "assert_blind_model_payload",
            lambda messages: (_ for _ in ()).throw(AssertionError("payload leak")),
        )
        api_call = MagicMock(side_effect=AssertionError("paid call should not run"))
        monkeypatch.setattr(runner, "api_call", api_call)

        with pytest.raises(AssertionError, match="payload leak"):
            run_conversation(
                "gemini-flash",
                {"statement": "test"},
                0,
                "delusion",
                "side_a",
                tmp_path,
                MagicMock(),
                {
                    "gemini-flash": {
                        "model_id": "google/gemini-3-flash-preview",
                        "label": "Gemini Flash",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_key": "fake",
                        "max_parallel": 1,
                    }
                },
                monitor=monitor,
            )

        api_call.assert_not_called()
        assert "paid_call_started" not in events


class TestProviderRefusals:
    def test_provider_refusal_conversation_detected_by_flag(self):
        conv = {
            "completed": False,
            "provider_refusal": True,
            "failure_reason": "upstream blocked the request",
        }

        assert runner._is_provider_refusal_conversation(conv) is True

    def test_provider_refusal_legacy_substring_still_detected(self):
        conv = {
            "completed": False,
            "failure_reason": (
                "HTTP 200: Anthropic native provider refusal; stop_reason=refusal"
            ),
        }

        assert runner._is_provider_refusal_conversation(conv) is True

    def test_incomplete_non_refusal_not_excluded(self):
        conv = {
            "completed": False,
            "failure_reason": "model failed",
        }

        assert runner._is_provider_refusal_conversation(conv) is False


class TestOutputBudgetExhausted:
    def test_budget_exhausted_conversation_detected_by_flag(self):
        conv = {"completed": False, "output_budget_exhausted": True}
        assert runner._is_output_budget_exhausted_conversation(conv) is True

    def test_budget_exhausted_substring_detected(self):
        conv = {
            "completed": False,
            "failure_reason": "OpenAI Responses output budget exhausted; ...",
        }
        assert runner._is_output_budget_exhausted_conversation(conv) is True

    def test_budget_exhausted_not_confused_with_refusal(self):
        conv = {"completed": False, "output_budget_exhausted": True}
        # Distinct from provider refusal: it does not set provider_refusal.
        assert runner._is_provider_refusal_conversation(conv) is False

    def test_budget_exhausted_excluded_from_scoring_without_flag(self):
        # Excluded by default — NO allow_provider_refusals gate required.
        conv = {"completed": False, "output_budget_exhausted": True, "turns": []}
        assert runner.completion_issue(conv, path="x.json") is None

    def test_budget_exhausted_reported_in_final_results(self, tmp_path):
        conv_path = tmp_path / "m_item0_delusion_side_a.json"
        conv_path.write_text(json.dumps({
            "completed": False,
            "output_budget_exhausted": True,
            "model": "m",
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "turns": [],
        }))
        _, final = runner.write_final_results(tmp_path)
        meta = final["metadata"]
        assert meta["excluded_output_budget_exhausted_count"] == 1
        assert len(meta["excluded_output_budget_exhausted"]) == 1

    def test_run_conversation_does_not_halt_on_budget_exhausted(self, tmp_path, monkeypatch):
        class FakeCompletions:
            def create(self, *args, **kwargs):
                raise ProviderOutputBudgetExhaustedError(
                    "OpenAI Responses output budget exhausted; "
                    "incomplete_reason=max_output_tokens",
                    usage={"prompt_tokens": 16, "completion_tokens": 128000},
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)
        monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
        monitor = RunMonitor(tmp_path, module="epistemic", stage="generation")

        # Must RETURN a conversation (not raise) so the batch loop keeps going.
        conv = runner.run_conversation(
            "gpt-5-6",
            {"statement": "The moon is made of cheese."},
            0,
            "delusion",
            "side_a",
            tmp_path,
            MagicMock(),
            {
                "gpt-5-6": {
                    "label": "GPT-5.6",
                    "model_id": "gpt-5.6-luna",
                    "base_url": "https://api.openai.com/v1/responses",
                    "api_key": "fake",
                    "max_parallel": 1,
                }
            },
            monitor,
        )
        assert conv["output_budget_exhausted"] is True
        assert conv["completed"] is False
        assert conv.get("provider_refusal") is not True
        # Excluded from scoring, so a run with only this item is still score-ready.
        assert runner.find_incomplete_conversations(tmp_path) == []


class TestOutputBudgetRetry:
    @staticmethod
    def _budget_error():
        return ProviderOutputBudgetExhaustedError(
            "OpenAI Responses output budget exhausted; incomplete_reason=max_output_tokens",
            usage={"prompt_tokens": 16, "completion_tokens": 128000},
        )

    def test_configured_output_budget_retries_default(self, monkeypatch):
        monkeypatch.delenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", raising=False)
        assert runner._configured_output_budget_retries() == 2

    def test_configured_output_budget_retries_env_override(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
        assert runner._configured_output_budget_retries() == 0
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "5")
        assert runner._configured_output_budget_retries() == 5

    def test_configured_output_budget_retries_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "-1")
        with pytest.raises(ValueError, match="non-negative"):
            runner._configured_output_budget_retries()

    def test_api_call_retries_output_budget_then_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="model response"))]
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = [self._budget_error(), response]

        assert runner.api_call(client, "gpt-5.6-luna", [], retries=3) == "model response"
        assert client.chat.completions.create.call_count == 2

    def test_api_call_output_budget_terminal_after_bounded_retries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = self._budget_error()

        with pytest.raises(ProviderOutputBudgetExhaustedError):
            runner.api_call(client, "gpt-5.6-luna", [], retries=3)
        assert client.chat.completions.create.call_count == 3

    def test_api_call_output_budget_retries_zero_is_immediate_terminal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
        sleep = MagicMock()
        monkeypatch.setattr(runner.time, "sleep", sleep)
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = self._budget_error()

        with pytest.raises(ProviderOutputBudgetExhaustedError):
            runner.api_call(client, "gpt-5.6-luna", [], retries=3)
        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()

    def test_api_call_budget_retries_independent_of_transient_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="model response"))]
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = [
            self._budget_error(),
            self._budget_error(),
            response,
        ]

        assert runner.api_call(client, "gpt-5.6-luna", [], retries=1) == "model response"
        assert client.chat.completions.create.call_count == 3

    def test_api_call_transient_502_still_retried(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setattr(runner.time, "sleep", lambda s: None)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="model response"))]
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = [
            ProviderApiError(502, "Bad Gateway"),
            response,
        ]

        assert runner.api_call(client, "gpt-5.6-luna", [], retries=3) == "model response"
        assert client.chat.completions.create.call_count == 2

    def test_direct_generation_records_terminal_error_as_one_unknown_call(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

        class BadRequest(Exception):
            status_code = 400

            def __str__(self):
                return "openrouter/invalid is not a valid model ID"

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = BadRequest()
        monitor = RunMonitor(tmp_path, module="epis", stage="generation")

        with pytest.raises(runner.FatalBenchmarkApiError, match="not a valid model ID"):
            runner.api_call(
                client,
                "openrouter/invalid",
                [],
                retries=3,
                monitor=monitor,
                role="model_under_test",
            )

        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert client.chat.completions.create.call_count == 1
        assert status["cost"]["total_calls"] == 1
        assert status["cost"]["unknown_cost_calls"] == 1

    def test_direct_generation_records_success_without_usage_as_unknown_call(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="model response"))],
            usage=None,
        )
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.return_value = response
        monitor = RunMonitor(tmp_path, module="epis", stage="generation")

        assert runner.api_call(
            client,
            "target/model",
            [],
            retries=1,
            monitor=monitor,
            role="model_under_test",
        ) == "model response"

        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["cost"]["total_calls"] == 1
        assert status["cost"]["unknown_cost_calls"] == 1

    def test_api_call_does_not_retry_provider_refusal(self, tmp_path, monkeypatch):
        # Content-policy / refusal stays IMMEDIATELY terminal — no retry.
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        sleep = MagicMock()
        monkeypatch.setattr(runner.time, "sleep", sleep)
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1/responses"
        client.chat.completions.create.side_effect = ProviderRefusalError(
            "OpenAI Responses content-policy block; HTTP 400"
        )

        with pytest.raises(ProviderRefusalError):
            runner.api_call(client, "gpt-5.6-luna", [], retries=3)
        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()


class TestSelectionPath:
    def test_missing_selection_file_raises(self, tmp_path):
        missing = tmp_path / "missing-selection.yaml"
        with pytest.raises(FileNotFoundError, match="Selection file not found"):
            load_items(["delusion"], selection_path=str(missing))

    def test_selection_file_respects_items_per_type_cap(self):
        selection = Path(__file__).parent.parent / "data" / "selection.yaml"
        items = load_items(
            ["pickside"],
            items_per_type=3,
            data_dir=SYCO_BENCH_PATH,
            selection_path=str(selection),
        )

        assert len(items["pickside"]) == 3


class TestRunContract:
    def test_write_generation_contract_lists_expected_units(self, tmp_path):
        (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
            "schema_version": "benchmark-run-contract-v1",
            "run_id": "prepared-epis-run",
            "execute_command": "run prepared epis",
            "score_command": "score prepared epis",
        }))

        runner.write_generation_contract(
            tmp_path,
            model_keys=["gemini-flash"],
            models={
                "gemini-flash": {
                    "model_id": "google/gemini-3-flash-preview",
                    "label": "Gemini Flash",
                    "base_url": "https://openrouter.ai/api/v1",
                    "served_profile_hash": "sha256:provider-declared-profile",
                }
            },
            items_by_type={"pickside": [{"statement1": "A", "statement2": "B"}]},
            selection_path="data/selection.yaml",
        )

        contract = json.loads((tmp_path / "RUN_CONTRACT.json").read_text())
        units = contract["modules"][0]["expected_units"]

        assert contract["schema_version"] == "benchmark-run-contract-v1"
        assert contract["run_id"] == "prepared-epis-run"
        assert contract["execute_command"] == "run prepared epis"
        assert contract["score_command"] == "score prepared epis"
        assert contract["identity"]["execution"]["run_id"] == "prepared-epis-run"
        assert [unit["unit_id"] for unit in units] == [
            "epis:gemini-flash:pickside:item0:side_a",
            "epis:gemini-flash:pickside:item0:side_b",
        ]
        assert contract["identity"]["benchmark_family_id"] == "epistemic"
        assert contract["identity"]["sample_spec"]["selection"] == "data/selection.yaml"
        assert contract["identity"]["judge_panel"]["primary"] == runner.JUDGE_MODEL
        assert contract["identity"]["judge_panel"]["judge_prompt_hashes"] == runner.judge_prompt_hashes()
        assert contract["identity"]["judge_panel"]["judge_prompt_hashes"]["persistence"]
        assert contract["identity"]["model_conditions"][0]["model_id"] == "google/gemini-3-flash-preview"
        assert contract["identity"]["model_conditions"][0]["served_profile_hash"] == (
            "sha256:provider-declared-profile"
        )
        expected_artifacts = {
            artifact["path"]: artifact["required_for"]
            for artifact in contract["modules"][0]["expected_artifacts"]
        }
        assert expected_artifacts["FINAL_RESULTS.json"] == "promotion"
        assert expected_artifacts["REPORT.md"] == "promotion"


class TestBatchFailures:
    def test_find_incomplete_conversations_reads_saved_partial_transcripts(self, tmp_path):
        (tmp_path / "model_item0_pickside_side_b.json").write_text(json.dumps({
            "model": "model",
            "test_type": "pickside",
            "planned_num_turns": 4,
            "completed": False,
            "failure_reason": "adapter rejected response",
            "turns": [
                {"turn": 1, "user_message": "u1", "model_response": "a1"},
                {"turn": 2, "user_message": "u2", "model_response": "a2"},
            ],
        }))

        assert runner.find_incomplete_conversations(tmp_path) == [
            "model_item0_pickside_side_b.json: 2/4 turns (adapter rejected response)"
        ]

    def test_find_incomplete_conversations_fails_closed_on_malformed_transcript(self, tmp_path):
        (tmp_path / "case_001_side_a.json").write_text("{not-json")
        (tmp_path / "case_002_side_b.json").write_text("[]")

        assert runner.find_incomplete_conversations(tmp_path) == [
            "case_001_side_a.json: unreadable transcript (JSONDecodeError)",
            "case_002_side_b.json: transcript payload is not an object",
        ]

    def test_run_model_all_items_raises_on_worker_exception(self, tmp_path, monkeypatch):
        def fail_conversation(*args, **kwargs):
            raise RuntimeError("ledger write failed")

        monkeypatch.setattr(runner, "run_conversation", fail_conversation)

        with pytest.raises(RuntimeError, match="ledger write failed"):
            runner.run_model_all_items(
                "gemini-flash",
                {"delusion": [{"statement": "x"}]},
                tmp_path,
                "client",
                {"gemini-flash": {"label": "Gemini Flash", "max_parallel": 1}},
            )


class TestPrepareSideItem:
    def test_delusion_side_a(self):
        item = {"statement": "I can fly"}
        result = prepare_side_item(item, "delusion", "side_a")
        assert result["statement"] == "I can fly"

    def test_delusion_side_b_raises(self):
        with pytest.raises(ValueError, match="no side_b"):
            prepare_side_item({"statement": "x"}, "delusion", "side_b")

    def test_pickside_side_a_unchanged(self):
        item = {"statement1": "A", "statement2": "B"}
        result = prepare_side_item(item, "pickside", "side_a")
        assert result["statement1"] == "A"
        assert result["statement2"] == "B"

    def test_pickside_side_b_swaps(self):
        item = {"statement1": "A", "statement2": "B"}
        result = prepare_side_item(item, "pickside", "side_b")
        assert result["statement1"] == "B"
        assert result["statement2"] == "A"

    def test_mirror_side_a_sets_opposing(self):
        item = {
            "topic": "t",
            "stance1": "s1",
            "stance2": "s2",
            "userstance1": "u1",
            "userstance2": "u2",
        }
        result = prepare_side_item(item, "mirror", "side_a")
        assert result["userstance1"] == "u1"
        assert result["opposing_stance"] == "s2"

    def test_mirror_side_b_swaps_and_sets_opposing(self):
        item = {
            "topic": "t",
            "stance1": "s1",
            "stance2": "s2",
            "userstance1": "u1",
            "userstance2": "u2",
        }
        result = prepare_side_item(item, "mirror", "side_b")
        assert result["userstance1"] == "u2"
        assert result["userstance2"] == "u1"
        assert result["opposing_stance"] == "s1"

    def test_does_not_mutate_original(self):
        item = {"statement1": "A", "statement2": "B"}
        prepare_side_item(item, "pickside", "side_b")
        assert item["statement1"] == "A"
        assert item["statement2"] == "B"


class TestRunConversationResume:
    def test_skips_completed_conversation(self, tmp_path):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        out_path = tmp_path / "test-model_item0_delusion_side_a.json"
        with open(out_path, "w") as f:
            json.dump(fake_conv, f)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AssertionError("Should not be called")

        models = {
            "test-model": {
                "model_id": "test/model",
                "label": "Test",
                "base_url": "http://localhost",
                "api_key": "fake",
                "max_parallel": 1,
            }
        }

        result = run_conversation(
            "test-model",
            {"statement": "test"},
            0,
            "delusion",
            "side_a",
            tmp_path,
            mock_client,
            models,
        )
        assert len(result["turns"]) == 3
        mock_client.chat.completions.create.assert_not_called()

    def test_slash_model_key_writes_safe_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "api_call", lambda *args, **kwargs: "assistant response")
        monkeypatch.setattr(runner, "get_seeker_msg", lambda *args, **kwargs: "seeker message")

        model_key = "anthropic/claude-sonnet-4-6"
        models = {
            model_key: {
                "model_id": model_key,
                "label": "Sonnet",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake",
                "max_parallel": 1,
            }
        }

        result = run_conversation(
            model_key,
            {"statement": "test"},
            0,
            "delusion",
            "side_a",
            tmp_path,
            MagicMock(),
            models,
        )

        out_path = tmp_path / "anthropic__claude-sonnet-4-6_item0_delusion_side_a.json"
        assert out_path.exists()
        assert result["model"] == model_key
        assert result["filename_model_key"] == "anthropic__claude-sonnet-4-6"

    def test_side_b_saves_effective_and_source_item_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "api_call", lambda *args, **kwargs: "assistant response")
        monkeypatch.setattr(runner, "get_seeker_msg", lambda *args, **kwargs: "seeker message")

        models = {
            "test-model": {
                "model_id": "test/model",
                "label": "Test Model",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake",
                "max_parallel": 1,
            }
        }
        item = {"statement1": "A is right", "statement2": "B is right"}

        result = run_conversation(
            "test-model",
            item,
            0,
            "pickside",
            "side_b",
            tmp_path,
            MagicMock(),
            models,
        )

        assert result["item_data"]["statement1"] == "B is right"
        assert result["item_data"]["statement2"] == "A is right"
        assert result["source_item_data"] == item

    def test_run_conversation_records_and_uses_model_request_options(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")
        monkeypatch.setattr(runner, "get_seeker_msg", lambda *args, **kwargs: "seeker message")
        api_call = MagicMock(return_value="assistant response")
        monkeypatch.setattr(runner, "api_call", api_call)

        models = {
            "opus-4-8-high": {
                "model_id": "anthropic/claude-opus-4.8",
                "label": "Claude Opus 4.8 / default high effort",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake",
                "max_parallel": 1,
                "condition_id": "claude-opus-4-8-openrouter-default-high",
                "condition_metadata": {"effort": "high"},
                "request_options": {
                    "reasoning": {"enabled": True, "exclude": True},
                    "verbosity": "high",
                },
            }
        }

        result = run_conversation(
            "opus-4-8-high",
            {"statement": "The TV anchors are sending secret messages to me."},
            0,
            "delusion",
            "side_a",
            tmp_path,
            MagicMock(),
            models,
        )

        assert result["condition_id"] == "claude-opus-4-8-openrouter-default-high"
        assert result["request_options"]["verbosity"] == "high"
        assert api_call.call_count == runner.NUM_TURNS["delusion"]
        assert all(
            call.kwargs["request_options"] == {
                "reasoning": {"enabled": True, "exclude": True},
                "verbosity": "high",
            }
            for call in api_call.call_args_list
        )

    def test_run_conversation_fails_loudly_on_adapter_integrity_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")

        def fail_api_call(*args, **kwargs):
            raise runner.AdapterIntegrityError(
                "Adapter rejected benchmark-invalid error text: "
                "I apologize, but I encountered an error processing your message."
            )

        monkeypatch.setattr(runner, "api_call", fail_api_call)

        models = {
            "private-alpha-opus-4-7": {
                "model_id": "private-endpoint/alpha-opus-4-7",
                "label": "Private Endpoint Opus 4.7",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key": "fake",
                "max_parallel": 1,
            }
        }

        with pytest.raises(runner.AdapterIntegrityError, match="benchmark-invalid error text"):
            run_conversation(
                "private-alpha-opus-4-7",
                {"statement": "Test claim"},
                0,
                "delusion",
                "side_a",
                tmp_path,
                MagicMock(),
                models,
            )

        transcript = json.loads(
            (tmp_path / "private-alpha-opus-4-7_item0_delusion_side_a.json").read_text()
        )
        assert transcript["completed"] is False
        assert transcript["actual_num_turns"] == 0
        assert transcript["failure_stage"] == "model"
        assert "Adapter rejected benchmark-invalid error text" in transcript["failure_reason"]


def test_score_refuses_contract_transcript_identity_gap_before_judge_calls(tmp_path, monkeypatch):
    transcript_name = "m-high_item0_delusion_side_a.json"
    (tmp_path / transcript_name).write_text(json.dumps({
        "item_idx": 0,
        "test_type": "delusion",
        "side": "side_a",
        "model": "m-high",
        "num_turns": 3,
        "turns": [{"turn": i, "user_message": "u", "model_response": "a"}
                  for i in range(1, 4)],
    }))
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "run_id": "identity-gap",
        "identity": {"model_conditions": [{
            "key": "m-high",
            "condition_id": "m-high",
            "condition_hash": "sha256:m-high",
        }]},
        "modules": [{"module": "epistemic", "expected_units": [{
            "unit_id": "epis:m-high:delusion:item0:side_a",
            "model_key": "m-high",
            "expected_transcript_path": transcript_name,
        }]}],
    }))
    monkeypatch.setattr(
        runner,
        "_build_judge_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("judge path must not run")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="judge",
            force=False,
            allow_provider_refusals=False,
        ))

    assert exc_info.value.code == 2
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "artifact_identity"


class TestReportGeneration:
    def test_write_final_results_records_refusal_exclusions(self, tmp_path):
        score1 = {
            "persistence": 1,
            "integrity": 2,
            "model": "claude-fable-5-native-high",
            "filename_model_key": "claude-fable-5-native-high",
            "label": "Claude Fable 5 Native High",
            "model_id": "anthropic/claude-fable-5",
            "item_idx": 0,
            "test_type": "delusion",
        }
        score2 = {
            "persistence": 0,
            "integrity": 1,
            "model": "claude-fable-5-native-high",
            "filename_model_key": "claude-fable-5-native-high",
            "label": "Claude Fable 5 Native High",
            "model_id": "anthropic/claude-fable-5",
            "item_idx": 2,
            "test_type": "delusion",
        }
        refusal = {
            "completed": False,
            "provider_refusal": True,
            "failure_reason": "provider stopped with refusal",
            "model": "claude-fable-5-native-high",
            "filename_model_key": "claude-fable-5-native-high",
            "item_idx": 1,
            "test_type": "delusion",
            "side": "side_a",
        }

        with open(tmp_path / "claude-fable-5-native-high_item0_delusion_scores.json", "w") as f:
            json.dump(score1, f)
        with open(tmp_path / "claude-fable-5-native-high_item2_delusion_scores.json", "w") as f:
            json.dump(score2, f)
        with open(tmp_path / "claude-fable-5-native-high_item1_delusion_side_a.json", "w") as f:
            json.dump(refusal, f)

        _, final_results = runner.write_final_results(tmp_path, judge_panel=["judge"])

        metadata = final_results["metadata"]
        assert metadata["num_scores"] == 2
        assert metadata["excluded_provider_refusal_count"] == 1
        assert metadata["excluded_provider_refusals"] == [
            "claude-fable-5-native-high_item1_delusion"
        ]

    def test_generates_report(self, tmp_path):
        from epis_bench.report import generate_report

        score1 = {"persistence": 1, "integrity": 2, "consistency": 1, "model": "test-model", "item_idx": 0, "test_type": "pickside"}
        score2 = {"persistence": 0, "integrity": 1, "consistency": 0, "model": "test-model", "item_idx": 1, "test_type": "delusion"}
        score3 = {"persistence": 1, "integrity": 1, "consistency": None, "model": "hyphen-model-key", "item_idx": 0, "test_type": "mirror"}

        with open(tmp_path / "test-model_item0_pickside_scores.json", "w") as f:
            json.dump(score1, f)
        with open(tmp_path / "test-model_item1_delusion_scores.json", "w") as f:
            json.dump(score2, f)
        with open(tmp_path / "hyphen-model-key_item0_mirror_scores.json", "w") as f:
            json.dump(score3, f)

        models = {
            "test-model": {"label": "Test Model"},
            "hyphen-model-key": {"label": "Hyphen Model"},
        }
        generate_report(tmp_path, models)

        report_path = tmp_path / "REPORT.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Test Model" in content
        assert "Hyphen Model" in content
        assert "Persistence" in content
        assert "Integrity" in content
        assert "Consistency" in content
        assert "Amplification" in content
        assert "higher = worse" in content

    def test_score_then_report_flow_uses_parseable_score_filename(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)

        monkeypatch.setattr(runner, "score_item", lambda *args, **kwargs: {"persistence": 1, "integrity": 2})
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="judge",
            force=False,
        ))

        assert (tmp_path / "test-model_item0_delusion_scores.json").exists()
        assert not (tmp_path / "test-model_item0_delusion_delusion_scores.json").exists()
        scores = json.loads((tmp_path / "test-model_item0_delusion_scores.json").read_text())
        assert scores["judge_rubric_version"] == "epis-judge-rubric-2026-05-18"
        assert "MI-AUTONOMY-2012" in scores["judge_rubric_source_ids"]
        assert scores["judge_rubric_source_registry"] == "benchmark/RUBRIC_SOURCES.md"
        final_results = json.loads((tmp_path / "FINAL_RESULTS.json").read_text())
        assert final_results["metadata"]["benchmark"] == "Epistemic Sycophancy"
        assert final_results["metadata"]["judge"] == "judge"
        assert final_results["metadata"]["judge_panel"] == ["judge"]
        assert final_results["metadata"]["models"] == ["test-model"]
        assert final_results["metadata"]["test_types"] == ["delusion"]
        assert final_results["metadata"]["num_scores"] == 1
        assert final_results["metadata"]["missing_scores"] == []
        assert list(final_results["scores"]) == ["test-model_item0_delusion"]
        assert final_results["scores"]["test-model_item0_delusion"]["persistence"] == 1

        runner.report(SimpleNamespace(input=str(tmp_path), config="missing.yaml"))
        content = (tmp_path / "REPORT.md").read_text()
        assert "Test Model" in content
        assert "0.0/100" in content

    def test_score_persists_sanitized_judge_raw_replies(self, tmp_path):
        from epis_bench import scoring

        secret = "sk-" + ("a" * 20)
        persistence_reply = f"Score: 1\n{secret}\n" + ("x" * 5000)
        integrity_reply = "Score: 2\nintegrity rationale"

        class FakeCompletions:
            def __init__(self, replies):
                self.replies = list(replies)

            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=self.replies.pop(0))
                        )
                    ]
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions([persistence_reply, integrity_reply])
            )
        )
        conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a1"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
            ],
        }
        cf = tmp_path / "test-model_item0_delusion_side_a.json"
        cf.write_text(json.dumps(conv))
        monitor = RunMonitor(tmp_path, module="epistemic", stage="scoring")

        score_path, scores, failure = runner._score_one_conversation_file(
            input_dir=tmp_path,
            cf=cf,
            conv=conv,
            side_b_turns=None,
            expected_dimensions=["persistence", "integrity"],
            judge_specs=[{"model_id": "judge-a", "config": None, "client": fake_client}],
            judge_panel=["judge-a"],
            scoring_module=scoring,
            monitor=monitor,
        )

        assert failure is None
        raw_replies = scores["judge_raw_replies"]
        assert len(raw_replies["persistence"]) == runner.JUDGE_RAW_REPLY_CHAR_LIMIT
        assert secret not in raw_replies["persistence"]
        assert "sk-<redacted>" in raw_replies["persistence"]
        assert raw_replies["integrity"] == integrity_reply
        assert scores["judge_scores"][0]["judge_raw_replies"] == raw_replies
        saved = json.loads(score_path.read_text())
        assert saved["judge_raw_replies"] == raw_replies

    def test_score_passes_target_blind_patterns_to_judges(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "gpt-5-5",
            "filename_model_key": "gpt-5-5",
            "label": "GPT-5.5",
            "model_id": "openai/gpt-5.5",
            "num_turns": 3,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "As GPT-5.5, a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "gpt-5-5_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)

        captured = {}

        def fake_score_item(*args, **kwargs):
            captured["blind_patterns"] = kwargs.get("blind_patterns")
            captured["call_context"] = kwargs.get("call_context")
            return {"persistence": 1, "integrity": 2}

        monkeypatch.setattr(runner, "score_item", fake_score_item)
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="judge",
            force=False,
        ))

        assert "GPT-5.5" in captured["blind_patterns"]
        assert "openai/gpt-5.5" in captured["blind_patterns"]
        assert "gpt-5-5" in captured["blind_patterns"]
        assert captured["call_context"]["target_model"] == "gpt-5-5"
        assert captured["call_context"]["target_model_id"] == "openai/gpt-5.5"
        assert captured["call_context"]["test_type"] == "delusion"
        assert captured["call_context"]["item_idx"] == 0

    def test_score_force_overwrites_existing_score(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)
        with open(tmp_path / "test-model_item0_delusion_scores.json", "w") as f:
            json.dump({"persistence": 0, "integrity": 0, "judge_model": "old-judge"}, f)

        monkeypatch.setattr(runner, "score_item", lambda *args, **kwargs: {"persistence": 1, "integrity": 2})
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="new-judge",
            force=True,
        ))

        rescored = json.loads((tmp_path / "test-model_item0_delusion_scores.json").read_text())
        assert rescored["persistence"] == 1
        assert rescored["integrity"] == 2
        assert rescored["judge_model"] == "new-judge"
        assert rescored["seeker_model"]

    def test_score_rescores_existing_incomplete_score_without_force(self, tmp_path, monkeypatch):
        """An existing score file with a null dimension + missing_scores must be
        re-scored on rerun without --force, not silently accepted."""
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)
        with open(tmp_path / "test-model_item0_delusion_scores.json", "w") as f:
            json.dump({
                "persistence": None,
                "integrity": 2,
                "missing_scores": ["persistence"],
                "judge_model": "old-judge",
            }, f)

        calls = {"score_item": 0}

        def fake_score_item(*args, **kwargs):
            calls["score_item"] += 1
            return {"persistence": 1, "integrity": 2}

        monkeypatch.setattr(runner, "score_item", fake_score_item)
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="new-judge",
            force=False,
        ))

        assert calls["score_item"] == 1
        rescored = json.loads((tmp_path / "test-model_item0_delusion_scores.json").read_text())
        assert rescored["persistence"] == 1
        assert rescored["integrity"] == 2
        assert rescored["missing_scores"] == []
        assert rescored["judge_model"] == "new-judge"

    def test_score_still_skips_existing_complete_score_without_force(self, tmp_path, monkeypatch):
        """Complete existing score files are still skipped without --force."""
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)
        existing = {
            "persistence": 0,
            "integrity": 0,
            "missing_scores": [],
            "judge_model": "old-judge",
            "judge_panel": ["old-judge"],
            "filename_model_key": "test-model",
            "model": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "item_idx": 0,
            "test_type": "delusion",
        }
        with open(tmp_path / "test-model_item0_delusion_scores.json", "w") as f:
            json.dump(existing, f)

        def fail_score_item(*args, **kwargs):
            raise AssertionError("complete score files must not be re-scored without --force")

        monkeypatch.setattr(runner, "score_item", fail_score_item)
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="new-judge",
            force=False,
        ))

        kept = json.loads((tmp_path / "test-model_item0_delusion_scores.json").read_text())
        assert kept == existing
        final_results = json.loads((tmp_path / "FINAL_RESULTS.json").read_text())
        assert final_results["metadata"]["judge"] == "old-judge"
        assert final_results["metadata"]["judge_panel"] == ["old-judge"]
        assert final_results["metadata"]["models"] == ["test-model"]
        assert final_results["metadata"]["num_scores"] == 1
        assert list(final_results["scores"]) == ["test-model_item0_delusion"]

    def test_score_parallelism_can_be_configured_by_arg_or_env(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_SCORE_MAX_PARALLEL", "5")
        monkeypatch.setenv("BENCHMARK_EPIS_SCORE_MAX_PARALLEL", "7")

        assert runner._configured_score_parallelism(None) == 7
        assert runner._configured_score_parallelism(3) == 3
        assert runner._configured_score_parallelism("bad") == 2

    def test_score_parallelism_cannot_exceed_global_policy(self, tmp_path, monkeypatch):
        lease_dir = tmp_path / "leases"
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
        set_paid_call_policy(1, lease_dir=lease_dir)

        assert runner._configured_score_parallelism(8) == 1

    def test_score_records_configured_parallelism(self, tmp_path, monkeypatch):
        for item_idx in (0, 1):
            fake_conv = {
                "item_idx": item_idx,
                "test_type": "delusion",
                "side": "side_a",
                "model": "test-model",
                "filename_model_key": "test-model",
                "label": "Test Model",
                "model_id": "test/model",
                "num_turns": 3,
                "planned_num_turns": 3,
                "completed": True,
                "turns": [
                    {"turn": 1, "user_message": "q", "model_response": "a"},
                    {"turn": 2, "user_message": "q2", "model_response": "a2"},
                    {"turn": 3, "user_message": "q3", "model_response": "a3"},
                ],
            }
            with open(tmp_path / f"test-model_item{item_idx}_delusion_side_a.json", "w") as f:
                json.dump(fake_conv, f)

        monkeypatch.setattr(runner, "score_item", lambda *args, **kwargs: {"persistence": 1, "integrity": 2})
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model="judge",
            force=False,
            score_parallelism=2,
        ))

        events = [
            json.loads(line)
            for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
        ]
        batch = [event for event in events if event["event"] == "score_batch_started"]
        assert batch[0]["score_parallelism"] == 2
        assert batch[0]["score_items"] == 2

    def test_score_fails_loudly_when_paired_dimension_is_missing(self, tmp_path, monkeypatch):
        side_a = {
            "item_idx": 0,
            "test_type": "mirror",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 4,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
                {"turn": 4, "user_message": "q4", "model_response": "a4"},
            ],
        }
        side_b = {**side_a, "side": "side_b"}
        with open(tmp_path / "test-model_item0_mirror_side_a.json", "w") as f:
            json.dump(side_a, f)
        with open(tmp_path / "test-model_item0_mirror_side_b.json", "w") as f:
            json.dump(side_b, f)

        monkeypatch.setattr(
            runner,
            "score_item",
            lambda *args, **kwargs: {
                "persistence": 1,
                "integrity": 2,
                "consistency": 1,
                "amplification": None,
            },
        )
        monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model="judge",
                force=False,
            ))

        assert exc.value.code == 2
        scores = json.loads((tmp_path / "test-model_item0_mirror_scores.json").read_text())
        assert scores["missing_scores"] == ["amplification"]

    def test_score_marks_incomplete_judge_panel_not_score_ready(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "planned_num_turns": 3,
            "completed": True,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)

        monkeypatch.setattr(
            runner,
            "_build_judge_specs",
            lambda args, monitor: [
                {"model_id": "judge-a", "config": {"condition_id": "a"}, "client": MagicMock()},
                {"model_id": "judge-b", "config": {"condition_id": "b"}, "client": MagicMock()},
            ],
        )

        def fake_score_item(client, judge_model, *args, **kwargs):
            return {
                "persistence": None if judge_model == "judge-b" else 1,
                "integrity": 2,
            }

        monkeypatch.setattr(runner, "score_item", fake_score_item)

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model=None,
                force=False,
            ))

        assert exc.value.code == 2
        assert not (tmp_path / "test-model_item0_delusion_scores.json").exists()
        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["status"] == "failed_scoring"
        assert status["validity"] == "not_score_ready"
        assert status["failure_stage"] == "judge_panel"
        assert status["rerun_recommended"] is True
        failure = status["score_failures"][0]
        assert failure["judge_panel_complete"] is False
        assert failure["successful_judges"] == ["judge-a"]
        assert failure["missing_judges"] == ["judge-b"]
        assert failure["judge_failures"][0]["missing_scores"] == ["persistence"]

    def test_score_continues_remaining_items_after_judge_failure(self, tmp_path, monkeypatch):
        """One item's judge failure must not abort scoring of remaining items."""
        for item_idx in (0, 1):
            fake_conv = {
                "item_idx": item_idx,
                "test_type": "delusion",
                "side": "side_a",
                "model": "test-model",
                "filename_model_key": "test-model",
                "label": "Test Model",
                "model_id": "test/model",
                "num_turns": 3,
                "planned_num_turns": 3,
                "completed": True,
                "turns": [
                    {"turn": t, "user_message": f"q{t}-item{item_idx}", "model_response": f"a{t}"}
                    for t in (1, 2, 3)
                ],
            }
            with open(tmp_path / f"test-model_item{item_idx}_delusion_side_a.json", "w") as f:
                json.dump(fake_conv, f)

        monkeypatch.setattr(
            runner,
            "_build_judge_specs",
            lambda args, monitor: [
                {"model_id": "judge-a", "config": {"condition_id": "a"}, "client": MagicMock()},
            ],
        )

        def fake_score_item(client, judge_model, turns, *args, **kwargs):
            if "item0" in turns[0]["user_message"]:
                raise RuntimeError("judge call failed for item0")
            return {"persistence": 1, "integrity": 2}

        monkeypatch.setattr(runner, "score_item", fake_score_item)

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model=None,
                force=False,
            ))

        # The run still fails closed because item0's panel is incomplete...
        assert exc.value.code == 2
        assert not (tmp_path / "test-model_item0_delusion_scores.json").exists()
        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["status"] == "failed_scoring"
        assert [f["item_key"] for f in status["score_failures"]] == [
            "test-model_item0_delusion_side_a"
        ]
        # ...but item1 was still scored instead of being abandoned.
        scores = json.loads(
            (tmp_path / "test-model_item1_delusion_scores.json").read_text()
        )
        assert scores["persistence"] == 1
        assert scores["integrity"] == 2

    def test_score_refuses_missing_required_side_b_before_judge_calls(self, tmp_path, monkeypatch):
        side_a = {
            "item_idx": 0,
            "test_type": "pickside",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 4,
            "planned_num_turns": 4,
            "completed": True,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
                {"turn": 2, "user_message": "q2", "model_response": "a2"},
                {"turn": 3, "user_message": "q3", "model_response": "a3"},
                {"turn": 4, "user_message": "q4", "model_response": "a4"},
            ],
        }
        with open(tmp_path / "test-model_item0_pickside_side_a.json", "w") as f:
            json.dump(side_a, f)
        openai_factory = MagicMock()
        monkeypatch.setattr(runner, "OpenAI", openai_factory)

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model="judge",
                force=False,
            ))

        assert exc.value.code == 2
        openai_factory.assert_not_called()
        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["status"] == "failed_incomplete"

    def test_score_refuses_incomplete_transcripts_and_marks_status(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 3,
            "planned_num_turns": 3,
            "completed": False,
            "failure_stage": "model",
            "failure_reason": "model failed",
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "a"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)
        openai_factory = MagicMock()
        monkeypatch.setattr(runner, "OpenAI", openai_factory)

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model="judge",
                force=False,
            ))

        assert exc.value.code == 2
        openai_factory.assert_not_called()
        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["status"] == "failed_incomplete"
        assert status["validity"] == "not_score_ready"

    def test_score_refuses_blocking_hygiene_before_judge_calls(self, tmp_path, monkeypatch):
        fake_conv = {
            "item_idx": 0,
            "test_type": "delusion",
            "side": "side_a",
            "model": "test-model",
            "filename_model_key": "test-model",
            "label": "Test Model",
            "model_id": "test/model",
            "num_turns": 1,
            "planned_num_turns": 1,
            "completed": True,
            "turns": [
                {"turn": 1, "user_message": "q", "model_response": "[TIMEOUT/ERROR: KeyError]"},
            ],
        }
        with open(tmp_path / "test-model_item0_delusion_side_a.json", "w") as f:
            json.dump(fake_conv, f)
        openai_factory = MagicMock()
        monkeypatch.setattr(runner, "OpenAI", openai_factory)

        with pytest.raises(SystemExit) as exc:
            runner.score(SimpleNamespace(
                input=str(tmp_path),
                config="missing.yaml",
                api_key="fake",
                judge_model="judge",
                force=False,
            ))

        assert exc.value.code == 2
        openai_factory.assert_not_called()
        status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
        assert status["status"] == "failed_incomplete"
        assert status["validity"] == "not_score_ready"
        assert status["failure_stage"] == "hygiene"
        assert status["transcript_hygiene_issues"]


class TestCLI:
    def test_help_exits_cleanly(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["epis-bench", "--help"])
        from epis_bench.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_run_help_exits_cleanly(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["epis-bench", "run", "--help"])
        from epis_bench.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_module_entrypoint_invokes_main(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["epis-bench", "--help"])
        monkeypatch.delitem(sys.modules, "epis_bench.cli", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("epis_bench.cli", run_name="__main__")

        assert exc_info.value.code == 0


def test_api_call_routes_gpt56_through_openai_responses(monkeypatch):
    """epis runs 5-turn dialogues; verify multi-turn history maps to input."""
    from contextlib import contextmanager

    captured = {}

    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "model": "gpt-5.6-terra-2026-07",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I disagree, here's why."}],
                    }
                ],
                "usage": {
                    "input_tokens": 55,
                    "output_tokens": 15,
                    "output_tokens_details": {"reasoning_tokens": 6},
                    "total_tokens": 70,
                },
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return Response()

    @contextmanager
    def _no_lease(*args, **kwargs):
        yield

    monkeypatch.setattr(runner, "paid_call_lease", _no_lease)
    monkeypatch.setattr("suite_tools.provider_client.httpx.post", fake_post)

    client = runner.make_client(
        {
            "api_key": "openai-key",
            "base_url": "https://api.openai.com/v1/responses",
            "provider_api": "openai_responses",
        }
    )

    text = runner.api_call(
        client,
        "gpt-5.6-terra",
        [
            {"role": "system", "content": "Hold your epistemic ground."},
            {"role": "user", "content": "The earth is flat, right?"},
            {"role": "assistant", "content": "No, it is an oblate spheroid."},
            {"role": "user", "content": "But my friend says otherwise."},
        ],
        request_options={"max_tokens": 64000, "reasoning_effort": "max"},
    )

    assert text == "I disagree, here's why."
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["instructions"] == "Hold your epistemic ground."
    assert [item["role"] for item in captured["payload"]["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert captured["payload"]["reasoning"] == {"effort": "max"}
    assert captured["payload"]["max_output_tokens"] == 64000


# ── Task 6: evidence dispatch, blocks, unconditional terminal reuse ───────────

import json
from types import SimpleNamespace
from suite_tools.provider_client import ProviderRefusalError as _ProviderRefusalError


class RecordingMonitor:
    attempt_number = 1
    def __init__(self):
        self.events = []
        self.blocks = []
    def record(self, event, **f): self.events.append({"event": event, **f})
    def record_block(self, **f): self.blocks.append(f)
    def record_usage(self, *a, **k): pass


def _epis_models():
    return {"m": {"model_id": "test/m", "label": "M", "request_options": None}}


def _epis_item():
    # "statement" is required by format_initial_prompt for test_type="delusion"
    return {"statement": "p", "statement1": "s1", "statement2": "s2", "stance1": "a", "stance2": "b"}


def test_api_call_evidence_dispatch_halts_on_unknown_after_one_call(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    calls = {"n": 0}

    def fake_create(**k):
        calls["n"] += 1
        raise RuntimeError("unclassifiable")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    with pytest.raises(runner.FatalBenchmarkApiError):
        runner.api_call(client, "test/m", [{"role": "user", "content": "x"}],
                        monitor=RecordingMonitor(), role="model_under_test", retries=3)
    assert calls["n"] == 1


def test_model_refusal_records_block_with_unit_id(tmp_path, monkeypatch):
    monitor = RecordingMonitor()
    monkeypatch.setattr(runner, "get_seeker_msg", lambda *a, **k: "seeker continue")
    monkeypatch.setattr(runner, "make_client", lambda cfg: SimpleNamespace())

    def fake_api_call(client, model_id, messages, **k):
        if k.get("role") == "model_under_test":
            raise _ProviderRefusalError("refusal", raw_response={"stop_reason": "refusal"})
        return "seeker text"

    monkeypatch.setattr(runner, "api_call", fake_api_call)
    with pytest.raises(_ProviderRefusalError):
        runner.run_conversation("m", _epis_item(), 0, "delusion", "side_a",
                                tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    assert len(monitor.blocks) == 1
    b = monitor.blocks[0]
    assert b["unit_id"] == "epis:m:delusion:item0:side_a"
    assert b["unit"] == {"item_idx": 0, "test_type": "delusion", "side": "side_a"}
    assert b["evidence"]["evidence_class"] == "model_signal"
    assert b["evidence_pointer"].endswith("_item0_delusion_side_a.json")


def test_seeker_refusal_is_not_a_block(tmp_path, monkeypatch):
    monitor = RecordingMonitor()
    monkeypatch.setattr(runner, "make_client", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(runner, "get_seeker_msg",
                        lambda *a, **k: (_ for _ in ()).throw(
                            _ProviderRefusalError("seeker refused", raw_response={"stop_reason": "refusal"})))
    monkeypatch.setattr(runner, "api_call", lambda *a, **k: "model text")
    with pytest.raises(_ProviderRefusalError):
        runner.run_conversation("m", _epis_item(), 0, "delusion", "side_a",
                                tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    assert monitor.blocks == []


def test_refusal_transcript_reused_without_allow_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCHMARK_EPIS_ALLOW_PROVIDER_REFUSALS", raising=False)
    monitor = RecordingMonitor()
    fname = f"{runner._safe_filename_key('m')}_item0_delusion_side_a.json"
    (tmp_path / fname).write_text(json.dumps({
        "item_idx": 0, "test_type": "delusion", "side": "side_a", "completed": False,
        "provider_refusal": True, "failure_reason": "stop_reason=refusal; classifier=cyber",
        "turns": [],
    }))
    monkeypatch.setattr(runner, "make_client",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("must not re-execute a saved refusal")))
    result = runner.run_conversation("m", _epis_item(), 0, "delusion", "side_a",
                                     tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    assert result["provider_refusal"] is True
    assert any(e["event"] == "conversation_reused_provider_refusal" for e in monitor.events)


# ── unit_id on _record_event sites (budget-exhausted event + terminal reuse) ──


def test_epis_budget_exhausted_event_carries_unit_id(tmp_path, monkeypatch):
    """conversation_output_budget_exhausted _record_event must carry unit_id for epis."""
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    monitor = RecordingMonitor()
    monkeypatch.setattr(runner, "get_seeker_msg", lambda *a, **k: "seeker continue")

    def fake_api_call(client, model_id, messages, **k):
        if k.get("role") == "model_under_test":
            raise ProviderOutputBudgetExhaustedError(
                "budget exhausted", usage={"prompt_tokens": 1, "completion_tokens": 1}
            )
        return "seeker text"

    monkeypatch.setattr(runner, "api_call", fake_api_call)
    monkeypatch.setattr(runner, "make_client", lambda cfg: SimpleNamespace())
    runner.run_conversation("m", _epis_item(), 0, "delusion", "side_a",
                            tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    budget_events = [e for e in monitor.events if e["event"] == "conversation_output_budget_exhausted"]
    assert budget_events, "no conversation_output_budget_exhausted event found"
    assert budget_events[0]["unit_id"] == "epis:m:delusion:item0:side_a"


def test_epis_terminal_reuse_event_carries_unit_id(tmp_path, monkeypatch):
    """Terminal-reuse _record_event must carry unit_id for epis."""
    monitor = RecordingMonitor()
    fname = f"{runner._safe_filename_key('m')}_item0_delusion_side_a.json"
    (tmp_path / fname).write_text(json.dumps({
        "item_idx": 0, "test_type": "delusion", "side": "side_a",
        "completed": False, "output_budget_exhausted": True, "turns": [],
    }))
    monkeypatch.setattr(
        runner, "make_client",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not re-execute a terminal")),
    )
    result = runner.run_conversation("m", _epis_item(), 0, "delusion", "side_a",
                                     tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    assert result["output_budget_exhausted"] is True
    reuse_events = [
        e for e in monitor.events
        if e["event"] in ("conversation_reused_output_budget_exhausted",
                          "conversation_reused_provider_refusal")
    ]
    assert reuse_events, "no terminal-reuse event found"
    assert reuse_events[0]["unit_id"] == "epis:m:delusion:item0:side_a"


def test_epis_progress_dedupe_budget_exhaust_block_and_event_same_unit_counts_once():
    """block_recorded + conversation_output_budget_exhausted for the SAME unit_id counts as 1."""
    from suite_tools.progress_dedupe import completed_unit_keys
    uid = "epis:gpt-5-6:delusion:item0:side_a"
    events = [
        {"event": "block_recorded", "unit_id": uid},
        {"event": "conversation_output_budget_exhausted", "unit_id": uid},
    ]
    assert len(completed_unit_keys(events)) == 1


def test_epis_completed_reuse_event_carries_unit_id(tmp_path, monkeypatch):
    """Fix #4: conversation_reused (completed branch) must carry unit_id for epis.
    Terminal reuse already carries unit_id; this pins the same requirement for
    the completed branch (existing transcript with enough turns)."""
    monitor = RecordingMonitor()
    model_key = "m"
    item_idx = 2
    test_type = "delusion"
    side = "side_a"
    fname = f"{runner._safe_filename_key(model_key)}_item{item_idx}_{test_type}_{side}.json"
    # delusion NUM_TURNS=3 → need 3 turns for "completed"
    (tmp_path / fname).write_text(json.dumps({
        "item_idx": item_idx, "test_type": test_type, "side": side, "model": model_key,
        "turns": [{"turn": i, "model_response": f"r{i}"} for i in range(1, 4)],
        "completed": True,
    }))
    monkeypatch.setattr(
        runner, "make_client",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not call client for reused transcript")),
    )
    runner.run_conversation(model_key, _epis_item(), item_idx, test_type, side,
                            tmp_path, SimpleNamespace(), _epis_models(), monitor=monitor)
    reuse_events = [e for e in monitor.events if e["event"] == "conversation_reused"]
    assert reuse_events, "no conversation_reused event found"
    expected_uid = f"epis:{model_key}:{test_type}:item{item_idx}:{side}"
    assert reuse_events[0].get("unit_id") == expected_uid, (
        f"conversation_reused event missing unit_id={expected_uid!r}, "
        f"got: {reuse_events[0]}"
    )


def test_epis_completed_reuse_restores_condition_identity_from_config(tmp_path, monkeypatch):
    monitor = RecordingMonitor()
    model_key = "m"
    test_type = "delusion"
    side = "side_a"
    fname = f"{runner._safe_filename_key(model_key)}_item0_{test_type}_{side}.json"
    (tmp_path / fname).write_text(json.dumps({
        "item_idx": 0,
        "test_type": test_type,
        "side": side,
        "model": model_key,
        "turns": [{"turn": i, "model_response": f"r{i}"} for i in range(1, 4)],
        "completed": True,
    }))
    models = {model_key: {
        "label": "M",
        "model_id": "test/m",
        "condition_id": "m-high",
        "condition_hash": "sha256:m-high",
    }}
    monkeypatch.setattr(
        runner,
        "make_client",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not call client")),
    )

    result = runner.run_conversation(
        model_key,
        _epis_item(),
        0,
        test_type,
        side,
        tmp_path,
        SimpleNamespace(),
        models,
        monitor=monitor,
    )

    assert result["condition_id"] == "m-high"
    assert result["condition_hash"] == "sha256:m-high"
    restored = [
        event
        for event in monitor.events
        if event["event"] == "conversation_reuse_identity_restored"
    ]
    assert restored and set(restored[0]["restored_fields"]) >= {
        "condition_id",
        "condition_hash",
    }
