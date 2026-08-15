import json
import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aita_bench import runner
from aita_bench import cli as aita_cli
from suite_tools.call_diagnostics import diagnose_call_journal
from suite_tools.provider_client import (
    ProviderApiError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
)
from suite_tools.paid_call_lease import set_paid_call_policy
from suite_tools.run_contract import STOP_BEFORE_NEXT_PAID_CALL, load_run_contract, write_run_control
from suite_tools.run_monitor import MonitoredOpenAIClient, RunMonitor


def test_score_cli_default_does_not_create_a_prepared_route_override(
    tmp_path,
    monkeypatch,
):
    captured = []
    monkeypatch.setattr(runner, "score", lambda args: captured.append(args))
    monkeypatch.setattr(
        "sys.argv",
        ["aita-bench", "score", "--input", str(tmp_path)],
    )

    aita_cli.main()

    assert len(captured) == 1
    assert captured[0].judge_base_url is None


def test_cli_rejects_literal_api_keys_and_accepts_environment_variable_names(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        ["aita-bench", "run", "--api-key", "literal-secret"],
    )
    with pytest.raises(SystemExit):
        aita_cli.main()

    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args))
    monkeypatch.setattr(
        "sys.argv",
        [
            "aita-bench",
            "run",
            "--output",
            str(tmp_path),
            "--api-key-env",
            "LOCAL_OPENAI_COMPATIBLE_API_KEY",
        ],
    )
    aita_cli.main()
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


def test_run_cli_accepts_sealed_pack_path_without_accepting_key_material_in_argv(
    tmp_path,
    monkeypatch,
):
    captured = []
    envelope = tmp_path / "synthetic.envelope.json"
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args))
    monkeypatch.setattr(
        "sys.argv",
        ["aita-bench", "run", "--dataset-mode", "nta-paired", "--sealed-pack", str(envelope)],
    )

    aita_cli.main()

    assert captured[0].sealed_pack == str(envelope)
    assert captured[0].sealed_key_part_b_from_env is False
    assert not hasattr(captured[0], "sealed_pack_key_part_b")


def test_run_records_sealed_pack_admission_failure_before_provider_use(tmp_path, monkeypatch):
    envelope = tmp_path / "broken.envelope.json"
    envelope.write_text('{"schema_version":"not-a-pack"}\n')
    monkeypatch.setattr(
        runner,
        "load_models",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("models must not load before sealed pack admission")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.run(SimpleNamespace(
            config="missing.yaml",
            output=str(tmp_path / "run"),
            dataset_mode="nta-paired",
            items="20",
            models="all",
            model=None,
            base_url=None,
            api_key=None,
            data=None,
            og_data=None,
            flip_data=None,
            paired_labels=None,
            item_selection=None,
            sealed_pack=str(envelope),
            sealed_pack_key_part_b="P" * 21,
            sealed_key_part_b_from_env=False,
            allow_sample_fallback=False,
        ))

    assert exc_info.value.code == 2
    status = json.loads((tmp_path / "run" / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "sealed_pack_admission"


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
            output=str(tmp_path / "prepared" / "aita"),
            config=str(tmp_path / "prepared" / "_configs" / "models.yaml"),
            model=None,
            models="all",
            base_url=None,
        ))

    assert exc_info.value.code == 2
    status = json.loads(
        (tmp_path / "prepared" / "aita" / "RUN_STATUS.json").read_text()
    )
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_prepared_aita_unit_drift_before_key_preflight(tmp_path, monkeypatch):
    from suite_tools.prepare_run import prepare_aita_run

    run_group = tmp_path / "prepared"
    contract_path = prepare_aita_run(
        run_id="aita-unit-drift",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
    )
    monkeypatch.setattr(
        runner,
        "load_yta_synthflip_items",
        lambda _args: ([0], {0: {"original": "changed", "flipped": "changed"}}),
    )
    monkeypatch.setattr(runner, "build_dataset_manifest", lambda *_args, **_kwargs: {})
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
            config=str(run_group / "_configs" / "calibration" / "aita-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            dataset_mode="yta-synthflip",
            data=str(
                Path(__file__).resolve().parents[1] / "data" / "AITA-YTA_sample.csv"
            ),
            items="1",
            item_selection=None,
            allow_sample_fallback=True,
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_prepared_dataset_manifest_drift_before_model_loading(tmp_path, monkeypatch):
    from suite_tools.prepare_run import prepare_aita_run

    run_group = tmp_path / "prepared"
    contract_path = prepare_aita_run(
        run_id="aita-manifest-drift",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
    )
    real_build_manifest = runner.build_dataset_manifest

    def drifted_manifest(*args, **kwargs):
        manifest = real_build_manifest(*args, **kwargs)
        manifest["files"][0]["sha256"] = "0" * 64
        return manifest

    monkeypatch.setattr(runner, "build_dataset_manifest", drifted_manifest)
    monkeypatch.setattr(
        runner,
        "load_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model config must not load after dataset provenance drift")
        ),
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
            config=str(run_group / "_configs" / "calibration" / "aita-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            dataset_mode="yta-synthflip",
            data=str(
                Path(__file__).resolve().parents[1] / "data" / "AITA-YTA_sample.csv"
            ),
            items="1",
            item_selection=None,
            allow_sample_fallback=True,
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_resealed_prepared_pack_with_same_prompts_before_model_loading(
    tmp_path,
    monkeypatch,
):
    from suite_tools.prepare_run import prepare_aita_run
    from suite_tools.sealed_pack import seal_files

    files = {
        "DATASET_CARD.md": b"SYNTHETIC DATA CARD v1\n",
        "MANIFEST.json": b'{"schema":"synthetic-v1"}\n',
        "flip.csv": b"id,flipped_story\nsynthetic-pair,synthetic reversal\n",
        "flip.labels.json": b'{"labels":{"synthetic-pair":"YTA"}}\n',
        "og.csv": b"id,original_post\nsynthetic-pair,synthetic original\n",
        "selection.yaml": b"items:\n  - index: 0\n    pair_id: synthetic-pair\n",
    }
    original = seal_files(
        files,
        pack_id="synthetic-aita-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(range(32)),
        nonce=bytes(range(12)),
    )
    envelope_path = tmp_path / "synthetic.envelope.json"
    ciphertext_path = tmp_path / "synthetic.sealed"
    envelope_path.write_text(
        json.dumps(dict(original.envelope, ciphertext_file=ciphertext_path.name))
    )
    ciphertext_path.write_bytes(original.ciphertext)

    run_group = tmp_path / "prepared"
    contract_path = prepare_aita_run(
        run_id="aita-resealed-pack",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="nta-paired",
        sealed_pack=str(envelope_path),
        sealed_pack_key_part_b=original.key_part_b,
    )

    replacement_files = dict(files)
    replacement_files["MANIFEST.json"] = b'{"schema":"synthetic-v2"}\n'
    replacement = seal_files(
        replacement_files,
        pack_id="synthetic-aita-pack",
        pack_version="v1",
        pair_count=1,
        key=bytes(reversed(range(32))),
        nonce=bytes(reversed(range(12))),
    )
    envelope_path.write_text(
        json.dumps(dict(replacement.envelope, ciphertext_file=ciphertext_path.name))
    )
    ciphertext_path.write_bytes(replacement.ciphertext)
    monkeypatch.setattr(
        runner,
        "load_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model config must not load after sealed-pack drift")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.run(SimpleNamespace(
            output=str(contract_path.parent),
            config=str(run_group / "_configs" / "calibration" / "aita-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            dataset_mode="nta-paired",
            data=None,
            og_data=None,
            flip_data=None,
            paired_labels=None,
            item_selection=None,
            sealed_pack=str(envelope_path),
            sealed_pack_key_part_b=replacement.key_part_b,
            sealed_key_part_b_from_env=False,
            items="1",
            allow_sample_fallback=False,
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "prepared_config_provenance"


def test_run_rejects_missing_preflight_receipt_before_key_or_client(tmp_path, monkeypatch):
    from suite_tools.prepare_run import prepare_aita_run

    run_group = tmp_path / "prepared"
    contract_path = prepare_aita_run(
        run_id="aita-missing-preflight",
        output_root=run_group,
        suite_config_path=Path(__file__).resolve().parents[2] / "suite_models.yaml",
        model_selector="group:calibration_smoke",
        judge_set="calibration",
        items="1",
        dataset_mode="yta-synthflip",
        allow_sample_fallback=True,
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
            config=str(run_group / "_configs" / "calibration" / "aita-models.yaml"),
            model=None,
            models="all",
            base_url=None,
            api_key="fake",
            dataset_mode="yta-synthflip",
            data=str(
                Path(__file__).resolve().parents[1] / "data" / "AITA-YTA_sample.csv"
            ),
            items="1",
            item_selection=None,
            allow_sample_fallback=True,
        ))

    assert exc_info.value.code == 2
    status = json.loads((contract_path.parent / "RUN_STATUS.json").read_text())
    assert status["failure_stage"] == "preflight_receipt_admission"


@pytest.fixture(autouse=True)
def default_verdict_judge(monkeypatch):
    from aita_bench import scoring

    def fake_verdict_alignment(*args, **kwargs):
        ground_truth = kwargs.get("ground_truth")
        if ground_truth is None and len(args) >= 4:
            ground_truth = args[3]
        verdict = "YTA" if ground_truth == "YTA" else "NTA" if ground_truth == "NTA" else "UNCLEAR"
        return {
            "verdict": verdict,
            "alignment": 1 if ground_truth in {"NTA", "YTA"} else None,
            "evidence": f"test verdict {verdict}",
            "rationale": "test fixture verdict",
            "confidence": "high",
        }

    monkeypatch.setattr(scoring, "score_verdict_alignment", fake_verdict_alignment)


def test_score_applies_judge_model_override(tmp_path, monkeypatch):
    conv = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gpt-5-5",
        "label": "GPT-5.5",
        "model_id": "openai/gpt-5.5",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gpt-5-5_item0_side_a.json").write_text(json.dumps(conv))

    captured = {}

    def fake_score_all(
        output_dir,
        items,
        or_client,
        models,
        judge_model=None,
        monitor=None,
        force=False,
        score_parallelism=None,
        judge_specs=None,
    ):
        captured["judge_model"] = judge_model
        captured["judge_panel"] = [spec["model_id"] for spec in (judge_specs or [])]
        captured["items"] = list(items)
        captured["models"] = list(models.keys())
        captured["force"] = force
        captured["score_parallelism"] = score_parallelism
        return {
            ("gpt-5-5", 0): {
                "outcome_a": 2,
                "resistance_a": 1,
                "therapeutic_a": 3,
                "judge_model": judge_model,
            }
        }

    monkeypatch.setattr(runner, "score_all", fake_score_all)
    monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

    runner.score(SimpleNamespace(
        input=str(tmp_path),
        config="missing.yaml",
        judge_model="openai/gpt-5.5",
        api_key="fake",
        judge_base_url="https://example.test/v1",
    ))

    assert captured["judge_model"] == "openai/gpt-5.5"
    assert captured["judge_panel"] == ["openai/gpt-5.5"]
    assert captured["items"] == [0]
    assert captured["models"] == ["gpt-5-5"]
    assert captured["force"] is False

    final = json.loads((tmp_path / "FINAL_RESULTS.json").read_text())
    assert final["metadata"]["judge"] == "openai/gpt-5.5"
    assert final["metadata"]["missing_scores"] == []


def test_score_refuses_contract_transcript_identity_gap_before_judge_calls(tmp_path, monkeypatch):
    transcript_name = "m-high_item0_side_a.json"
    (tmp_path / transcript_name).write_text(json.dumps({
        "item_idx": 0,
        "side": "side_a",
        "model": "m-high",
        "turns": [{"turn": 1, "user_message": "u", "model_response": "a"}],
    }))
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "run_id": "identity-gap",
        "identity": {"model_conditions": [{
            "key": "m-high",
            "condition_id": "m-high",
            "condition_hash": "sha256:m-high",
        }]},
        "modules": [{"module": "aita", "expected_units": [{
            "unit_id": "aita:m-high:item0:side_a",
            "model_key": "m-high",
            "expected_transcript_path": transcript_name,
        }]}],
    }))
    monkeypatch.setattr(
        runner,
        "score_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("judge path must not run")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="judge",
            api_key="fake",
            judge_base_url=None,
        ))

    assert exc_info.value.code == 2
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "artifact_identity"


def test_score_refuses_stale_judge_provenance_before_judge_calls(tmp_path, monkeypatch):
    transcript_name = "m-high_item0_side_a.json"
    (tmp_path / transcript_name).write_text(json.dumps({
        "item_idx": 0,
        "side": "side_a",
        "model": "m-high",
        "condition_id": "m-high",
        "condition_hash": "sha256:m-high",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }))
    stale_hashes = runner.judge_prompt_hashes()
    stale_hashes["outcome"] = "stale-prompt-hash"
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "run_id": "stale-judge",
        "identity": {
            "model_conditions": [{
                "key": "m-high",
                "condition_id": "m-high",
                "condition_hash": "sha256:m-high",
            }],
            "judge_panel": {
                "primary": "judge",
                "panel": ["judge"],
                "configs": [{"model_id": "judge", "provider_api": "openai_compatible"}],
                "judge_prompt_hashes": stale_hashes,
                "rubric_version": runner.JUDGE_RUBRIC_VERSION,
                "rubric_source_ids": list(runner.JUDGE_RUBRIC_SOURCE_IDS),
                "rubric_source_registry": runner.JUDGE_SOURCE_REGISTRY,
            },
        },
        "modules": [{"module": "aita", "expected_units": [{
            "unit_id": "aita:m-high:item0:side_a",
            "model_key": "m-high",
            "expected_transcript_path": transcript_name,
        }]}],
    }))
    monkeypatch.setattr(
        runner,
        "_build_judge_specs",
        lambda *_args, **_kwargs: [{
            "model_id": "judge",
            "config": {"model_id": "judge", "provider_api": "openai_compatible"},
            "client": MagicMock(),
        }],
    )
    monkeypatch.setattr(
        runner,
        "score_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("judge calls must not run")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model=None,
            api_key="fake",
            judge_base_url=None,
            force=False,
            score_parallelism=1,
        ))

    assert exc_info.value.code == 2
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_invalid"
    assert status["failure_stage"] == "judge_provenance"


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
    monkeypatch.setattr(runner, "select_model_keys", lambda args, loaded: (["gemini-flash"], loaded))
    monkeypatch.setattr(
        runner,
        "load_nta_paired_items",
        lambda args: (
            [0],
            {
                0: {
                    "original": "original story",
                    "pair_id": "pair-1",
                    "side_a_ground_truth": "NTA",
                    "side_b_ground_truth": "YTA",
                }
            },
            {0: "flipped story"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_model_all_items",
        lambda *args, **kwargs: [{"completed": True}, {"completed": True}],
    )
    monkeypatch.setattr(runner, "find_incomplete_conversations", lambda output_dir: [])

    runner.run(SimpleNamespace(
        config="missing.yaml",
        output=str(tmp_path),
        dataset_mode="nta-paired",
        items="1",
        models="gemini-flash",
        model=None,
        base_url=None,
        api_key="fake",
        data=None,
        og_data=None,
        flip_data=None,
        allow_sample_fallback=False,
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


def test_sealed_pack_run_does_not_write_redundant_plaintext_flip_cache(tmp_path, monkeypatch):
    models = {
        "sealed-model": {
            "model_id": "test/sealed-model",
            "label": "Sealed Model",
            "base_url": "https://openrouter.ai/api/v1",
            "max_parallel": 1,
        }
    }
    monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(runner, "load_models", lambda config: dict(models))
    monkeypatch.setattr(
        runner,
        "select_model_keys",
        lambda args, loaded: (["sealed-model"], loaded),
    )

    def load_sealed(args):
        args._sealed_pack_context = {
            "pack_id": "synthetic-sealed-pack",
            "pack_version": "v1",
            "pair_count": 1,
            "ciphertext_sha256": "a" * 64,
            "plaintext_identity_sha256": "b" * 64,
            "key_scheme": "public-base64url-split-22-21-v1",
            "file_hashes": {},
        }
        return (
            [0],
            {0: {
                "original": "SEALED ORIGINAL SENTINEL",
                "pair_id": "pair-1",
                "side_a_ground_truth": "NTA",
                "side_b_ground_truth": "YTA",
            }},
            {0: "SEALED REVERSAL SENTINEL"},
        )

    monkeypatch.setattr(runner, "load_nta_paired_items", load_sealed)
    monkeypatch.setattr(
        runner,
        "run_model_all_items",
        lambda *args, **kwargs: [{"completed": True}, {"completed": True}],
    )
    monkeypatch.setattr(runner, "find_incomplete_conversations", lambda output_dir: [])

    runner.run(SimpleNamespace(
        config="missing.yaml",
        output=str(tmp_path),
        dataset_mode="nta-paired",
        items="1",
        models="sealed-model",
        model=None,
        base_url=None,
        api_key="fake",
        data=None,
        og_data=None,
        flip_data=None,
        paired_labels=None,
        item_selection=None,
        sealed_pack="synthetic.envelope.json",
        sealed_key_part_b_from_env=False,
        allow_sample_fallback=False,
    ))

    assert not (tmp_path / "flip_item0.json").exists()


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
    monkeypatch.setattr(runner, "select_model_keys", lambda args, loaded: (list(models), loaded))
    monkeypatch.setattr(
        runner,
        "load_nta_paired_items",
        lambda args: (
            [0],
            {0: {"original": "story", "pair_id": "pair", "side_a_ground_truth": "NTA", "side_b_ground_truth": "YTA"}},
            {0: "flipped story"},
        ),
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
    monkeypatch.setattr(runner, "find_incomplete_conversations", lambda output_dir: [])

    runner.run(SimpleNamespace(
        config="missing.yaml", output=str(tmp_path / "run"), dataset_mode="nta-paired",
        items="1", models="all", model=None, base_url=None, api_key="fake",
        data=None, og_data=None, flip_data=None, allow_sample_fallback=False,
    ))

    assert max_seen == 1


def test_yta_synthflip_contract_expects_generated_side_b(tmp_path):
    items = {
        0: {
            "original": "original story",
            "dataset_mode": "yta-synthflip",
            "side_a_ground_truth": "YTA",
            "side_b_ground_truth": "synthetic_reversal",
        }
    }
    models = {
        "gemini-flash": {
            "model_id": "google/gemini-3-flash-preview",
            "label": "Gemini 3 Flash",
            "base_url": "https://openrouter.ai/api/v1",
            "max_parallel": 1,
        }
    }
    manifest = runner.build_dataset_manifest(
        SimpleNamespace(items="1", data=None, item_selection=None, allow_sample_fallback=False),
        "yta-synthflip",
        [0],
        items,
        {},
    )

    runner.write_generation_contract(
        tmp_path,
        model_keys=["gemini-flash"],
        models=models,
        item_indices=[0],
        flips={},
        dataset_mode="yta-synthflip",
        items=items,
        dataset_manifest=manifest,
    )

    contract = load_run_contract(tmp_path)
    units = contract["modules"][0]["expected_units"]

    assert {unit["side"] for unit in units} == {"side_a", "side_b"}
    assert contract["modules"][0]["dataset_manifest"]["selected_items"][0]["sides"] == ["side_a", "side_b"]
    assert contract["identity"]["sample_spec"]["sides_by_item"] == {"0": ["side_a", "side_b"]}
    assert contract["identity"]["sample_spec"]["expected_flip_item_indices"] == [0]
    assert contract["identity"]["judge_panel"]["seeker"] == runner.SEEKER_MODEL
    assert contract["identity"]["judge_panel"]["flip_generator"] == runner.FLIP_MODEL


def test_explicit_model_overrides_models_all_default():
    models = {
        "gemini-flash": {
            "model_id": "google/gemini-3-flash-preview",
            "label": "Gemini 3 Flash",
        },
        "gpt-5-5": {
            "model_id": "openai/gpt-5.5",
            "label": "GPT-5.5",
        },
    }

    keys, resolved = runner.select_model_keys(SimpleNamespace(
        model="private-endpoint/harness-gemini-3-1-pro",
        models="all",
        base_url="http://127.0.0.1:9999/v1",
        api_key="adapter",
    ), models, openrouter_key="unused")

    assert keys == ["private-endpoint-harness-gemini-3-1-pro"]
    assert list(resolved.keys()) == [
        "gemini-flash",
        "gpt-5-5",
        "private-endpoint-harness-gemini-3-1-pro",
    ]
    assert resolved["private-endpoint-harness-gemini-3-1-pro"]["model_id"] == "private-endpoint/harness-gemini-3-1-pro"
    assert resolved["private-endpoint-harness-gemini-3-1-pro"]["base_url"] == "http://127.0.0.1:9999/v1"
    assert resolved["private-endpoint-harness-gemini-3-1-pro"]["api_key"] == "adapter"


def test_explicit_route_and_key_override_matching_configured_model():
    models = {
        "configured": {
            "model_id": "same/model",
            "label": "Configured",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_api": "openai_compatible",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_key": "openrouter-key",
        }
    }

    keys, resolved = runner.select_model_keys(
        SimpleNamespace(
            model="same/model",
            models="all",
            base_url="http://127.0.0.1:9999/v1",
            api_key="adapter-key",
        ),
        models,
        openrouter_key="openrouter-key",
    )

    assert keys == ["configured"]
    assert resolved["configured"]["base_url"] == "http://127.0.0.1:9999/v1"
    assert resolved["configured"]["provider_api"] == "openai_compatible"
    assert resolved["configured"]["api_key"] == "adapter-key"
    assert resolved["configured"]["api_key_env"] is None
    assert resolved["configured"]["credential_explicit"] is True
    assert resolved["configured"]["condition_hash"].startswith("sha256:")


def test_unknown_model_cannot_send_openrouter_key_to_custom_remote_host(monkeypatch):
    monkeypatch.setattr(
        runner,
        "make_provider_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created")
        ),
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY.*refusing"):
        runner.select_model_keys(
            SimpleNamespace(
                model="unknown/model",
                models="all",
                base_url="https://attacker.example/v1",
                api_key=None,
            ),
            {},
            openrouter_key="operator-openrouter-key",
        )


def test_unknown_model_allows_explicit_key_for_custom_remote_host():
    keys, resolved = runner.select_model_keys(
        SimpleNamespace(
            model="unknown/model",
            models="all",
            base_url="https://models.example/v1",
            api_key="explicit-custom-key",
        ),
        {},
        openrouter_key="must-not-be-used",
    )

    assert keys == ["unknown-model"]
    assert resolved["unknown-model"]["credential_explicit"] is True
    assert resolved["unknown-model"]["api_key"] == "explicit-custom-key"
    assert resolved["unknown-model"]["condition_id"] == "unknown-model"
    assert resolved["unknown-model"]["condition_hash"].startswith("sha256:")
    assert resolved["unknown-model"]["route_hash"].startswith("sha256:")


def test_unknown_model_with_provider_suffix_gets_portable_artifact_key():
    keys, resolved = runner.select_model_keys(
        SimpleNamespace(
            model="vendor/model:free",
            models="all",
            base_url="https://openrouter.ai/api/v1",
            api_key="explicit-key",
        ),
        {},
        openrouter_key="unused",
    )

    assert keys == ["vendor-model-free"]
    assert resolved["vendor-model-free"]["model_id"] == "vendor/model:free"


@pytest.mark.parametrize(
    "unsafe_key",
    ["/tmp/external", "../escape", "nested/key", ".", "", "-leading"],
)
def test_load_models_rejects_nonportable_artifact_keys_before_client_use(
    tmp_path,
    monkeypatch,
    unsafe_key,
):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(json.dumps({
        "models": {
            unsafe_key: {
                "model_id": "provider/model",
                "base_url": "https://openrouter.ai/api/v1",
            }
        }
    }))
    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-key")
    monkeypatch.setattr(
        runner,
        "make_provider_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created")
        ),
    )

    with pytest.raises(ValueError, match="model keys.*path separator"):
        runner.load_models(config_path)


@pytest.mark.parametrize("selector", ["../escape", "/tmp/external", "nested/key", ""])
def test_explicit_model_key_list_rejects_artifact_path_escape(selector):
    with pytest.raises(ValueError, match="model keys.*path separator"):
        runner.select_model_keys(
            SimpleNamespace(model=None, models=selector, base_url=None, api_key=None),
            {"normal-model": {"model_id": "provider/model"}},
            openrouter_key="unused",
        )


def test_custom_target_key_is_not_reused_for_openrouter_support(monkeypatch):
    support_clients = []
    monkeypatch.setattr(runner, "_openrouter_key", lambda: "support-openrouter-key")
    monkeypatch.setattr(runner, "_preflight_openrouter_key", lambda key: None)
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda **kwargs: support_clients.append(kwargs) or SimpleNamespace(),
    )

    _keys, resolved = runner.select_model_keys(
        SimpleNamespace(
            model="custom/model",
            models="all",
            base_url="http://127.0.0.1:9999/v1",
            api_key="target-private-key",
        ),
        {},
        openrouter_key="support-openrouter-key",
    )
    runner._openrouter_support_client()

    assert resolved["custom-model"]["api_key"] == "target-private-key"
    assert support_clients == [{
        "api_key": "support-openrouter-key",
        "base_url": "https://openrouter.ai/api/v1",
    }]


def test_default_models_use_current_claude_slugs(tmp_path):
    models = runner.load_models(tmp_path / "missing.yaml")

    assert models["sonnet-4-6"]["model_id"] == "anthropic/claude-sonnet-4.6"
    assert models["opus-4-6"]["model_id"] == "anthropic/claude-opus-4.6"
    assert all(model["condition_id"] for model in models.values())
    assert all(model["condition_hash"].startswith("sha256:") for model in models.values())
    assert all(model["route_hash"].startswith("sha256:") for model in models.values())


def test_load_models_discovers_repo_env_key_after_import(tmp_path, monkeypatch):
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


def test_load_models_preserves_native_provider_metadata(tmp_path, monkeypatch):
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


def test_load_models_preserves_direct_judge_config(tmp_path, monkeypatch):
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


def test_direct_openai_judge_call_uses_supported_payload_shape(tmp_path, monkeypatch):
    from aita_bench import scoring

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    captured = {}

    class FakeCompletions:
        base_url = "https://api.openai.com/v1"

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    scoring._create_judge_completion(
        client,
        model="gpt-5.5",
        messages=[{"role": "user", "content": "score"}],
        max_tokens=128,
        timeout=120,
        extra_body={"reasoning_effort": "high"},
    )

    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 128
    assert captured["extra_body"]["reasoning_effort"] == "high"


def test_judge_completion_without_usage_counts_unknown_cost(tmp_path, monkeypatch):
    from aita_bench import scoring
    from suite_tools.run_monitor import MonitoredOpenAIClient, RunMonitor

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

    class FakeCompletions:
        base_url = "https://api.openai.com/v1"

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="1"))],
                usage=None,
            )

    raw = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monitor = RunMonitor(tmp_path / "run", module="aita", stage="scoring")
    client = MonitoredOpenAIClient(raw, monitor, role="judge")

    scoring._create_judge_completion(
        client,
        model="gpt-5.5",
        messages=[{"role": "user", "content": "score"}],
        max_tokens=32,
        timeout=120,
        extra_body={},
    )

    cost = json.loads((tmp_path / "run" / "RUN_STATUS.json").read_text())["cost"]
    assert cost["total_calls"] == 1
    assert cost["unknown_cost_calls"] == 1


def test_judge_retry_records_failed_and_successful_physical_calls_once(tmp_path, monkeypatch):
    from aita_bench import scoring
    from suite_tools.run_monitor import MonitoredOpenAIClient, RunMonitor

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(scoring.time, "sleep", lambda _: None)

    class FakeCompletions:
        base_url = "https://api.openai.com/v1"

        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider failure")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="1"))],
                usage=None,
            )

    completions = FakeCompletions()
    raw = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=completions),
    )
    monitor = RunMonitor(tmp_path / "run", module="aita", stage="scoring")
    client = MonitoredOpenAIClient(raw, monitor, role="judge")

    assert scoring._call_judge(client, "gpt-5.5", "score", retries=2) == 1
    cost = json.loads((tmp_path / "run" / "RUN_STATUS.json").read_text())["cost"]
    assert completions.calls == 2
    assert cost["total_calls"] == 2
    assert cost["unknown_cost_calls"] == 2


def test_direct_openai_judge_reuses_configured_token_cap(tmp_path, monkeypatch):
    from aita_bench import scoring

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    captured = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"score": 1}')
                    )
                ]
            )

    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    request_options = {"max_tokens": 8192, "reasoning_effort": "high"}
    scoring.set_judge_request_options("gpt-5.5", request_options)

    try:
        for _ in range(2):
            assert scoring._call_judge_json(
                client,
                "gpt-5.5",
                "score",
                retries=1,
                max_tokens=900,
            ) == {"score": 1}
    finally:
        scoring.set_judge_request_options("gpt-5.5", None)

    assert [call["max_completion_tokens"] for call in captured] == [8192, 8192]
    assert request_options == {"max_tokens": 8192, "reasoning_effort": "high"}


def test_aita_panel_aggregation_preserves_per_judge_scores():
    judge_specs = [
        {"model_id": "judge-a", "config": {"condition_id": "a"}},
        {"model_id": "judge-b", "config": {"condition_id": "b"}},
        {"model_id": "judge-c", "config": {"condition_id": "c"}},
    ]
    aggregate = runner._aggregate_panel_scores(
        [
            {
                "verdict_a": "NTA",
                "verdict_b": "NTA",
                "verdict_alignment_a": 1,
                "verdict_alignment_b": 0,
                "paired_verdict_alignment": 0,
                "moral_sycophancy_pattern": "both_nta",
                "outcome_a": 2,
                "therapeutic_a": 3,
                "missing_scores": [],
            },
            {
                "verdict_a": "NTA",
                "verdict_b": "YTA",
                "verdict_alignment_a": 1,
                "verdict_alignment_b": 1,
                "paired_verdict_alignment": 1,
                "moral_sycophancy_pattern": "side_a_nta_side_b_yta",
                "outcome_a": 1,
                "therapeutic_a": 2,
                "missing_scores": [],
            },
            {
                "verdict_a": "NTA",
                "verdict_b": "YTA",
                "verdict_alignment_a": 1,
                "verdict_alignment_b": 1,
                "paired_verdict_alignment": 1,
                "moral_sycophancy_pattern": "side_a_nta_side_b_yta",
                "outcome_a": 1,
                "therapeutic_a": 1,
                "missing_scores": [],
            },
        ],
        judge_specs=judge_specs,
    )

    assert aggregate["outcome_a"] == 1.333
    assert aggregate["therapeutic_a"] == 2
    assert aggregate["verdict_b"] == "YTA"
    assert aggregate["verdict_alignment_b"] == 0.667
    assert aggregate["verdict_alignment_b_majority"] == 1
    assert aggregate["paired_verdict_alignment_majority"] == 1
    assert aggregate["primary_failure"] is False
    assert aggregate["moral_sycophancy_pattern"] == "side_a_nta_side_b_yta"
    assert aggregate["judge_panel"] == ["judge-a", "judge-b", "judge-c"]
    assert aggregate["judge_configs"] == [{"condition_id": "a"}, {"condition_id": "b"}, {"condition_id": "c"}]
    assert len(aggregate["judge_scores"]) == 3
    assert aggregate["judge_aggregation"]["method"] == "full_judge_panel_mean"


def test_aita_panel_aggregation_rejects_incomplete_panel():
    judge_specs = [
        {"model_id": "judge-a", "config": {"condition_id": "a"}},
        {"model_id": "judge-b", "config": {"condition_id": "b"}},
    ]

    with pytest.raises(runner.JudgePanelIncompleteError) as exc:
        runner._aggregate_panel_scores(
            [
                {
                    "verdict_a": "NTA",
                    "verdict_alignment_a": 1,
                    "outcome_a": 2,
                    "therapeutic_a": 3,
                    "missing_scores": [],
                }
            ],
            judge_specs=judge_specs,
            expected_dimensions=["verdict_alignment_a", "outcome_a", "therapeutic_a"],
            item_key="gemini-flash_item0",
        )

    payload = exc.value.to_status_payload()
    assert payload["judge_panel_complete"] is False
    assert payload["expected_judges"] == ["judge-a", "judge-b"]
    assert payload["successful_judges"] == ["judge-a"]
    assert payload["missing_judges"] == ["judge-b"]
    assert payload["rerun_recommended"] is True


def test_aita_panel_no_majority_is_inconclusive_and_ambiguous():
    judge_specs = [
        {"model_id": "judge-a", "config": {"condition_id": "a"}},
        {"model_id": "judge-b", "config": {"condition_id": "b"}},
    ]
    aggregate = runner._aggregate_panel_scores(
        [
            {
                "verdict_a": "NTA",
                "verdict_b": "YTA",
                "verdict_alignment_a": 1,
                "verdict_alignment_b": 1,
                "paired_verdict_alignment": 1,
                "missing_scores": [],
            },
            {
                "verdict_a": "YTA",
                "verdict_b": "NTA",
                "verdict_alignment_a": 0,
                "verdict_alignment_b": 0,
                "paired_verdict_alignment": 0,
                "missing_scores": [],
            },
        ],
        judge_specs=judge_specs,
    )

    assert aggregate["verdict_a"] == runner.PANEL_NO_MAJORITY
    assert aggregate["verdict_alignment_a"] == 0.5
    assert aggregate["verdict_alignment_a_majority"] is None
    assert aggregate["paired_verdict_alignment_majority"] is None
    assert aggregate["primary_failure"] is None
    assert aggregate["moral_sycophancy_pattern"] == "ambiguous"


def test_api_call_raises_on_adapter_integrity_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception(
        "Error code: 502 - {'error': 'Adapter rejected backend analysis failure'}"
    )

    with pytest.raises(runner.AdapterIntegrityError):
        runner.api_call(client, "private-endpoint/harness-gemini-3-1-pro", [])


def test_api_call_raises_on_generic_backend_error_text(tmp_path, monkeypatch):
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


def test_api_call_does_not_retry_invalid_model_errors(tmp_path, monkeypatch):
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


def test_api_call_passes_request_options_as_extra_body(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response

    content = runner.api_call(
        client,
        "anthropic/claude-opus-4.7",
        [{"role": "user", "content": "prompt"}],
        request_options={
            "reasoning": {"enabled": True, "exclude": True},
            "verbosity": "high",
        },
    )

    assert content == "model response"
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning": {"enabled": True, "exclude": True},
        "verbosity": "high",
    }


def test_api_call_reuses_direct_openai_request_options_without_losing_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    captured = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="model response"))]
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
    monitor = runner.RunMonitor(tmp_path / "run", module="aita", stage="generation")

    for _ in range(2):
        assert runner.api_call(
            client,
            "gpt-5.6-sol",
            [{"role": "user", "content": "prompt"}],
            max_tokens=1000,
            retries=1,
            monitor=monitor,
            role="model_under_test",
            request_options=request_options,
            request_context={
                "condition_id": "gpt-5-6-sol-openai-native-max",
                "model_key": "gpt-5-6-sol-native-max",
            },
        ) == "model response"

    assert [call["max_completion_tokens"] for call in captured] == [
        128000,
        128000,
    ]
    assert request_options == {"max_tokens": 128000, "reasoning_effort": "max"}
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


def test_api_call_generation_timeout_defaults_above_adapter_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.delenv("BENCHMARK_GENERATION_TIMEOUT_SECONDS", raising=False)
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response

    assert runner.api_call(client, "private-endpoint/harness", [], retries=1) == "model response"

    assert client.chat.completions.create.call_args.kwargs["timeout"] == 150


def test_api_call_generation_timeout_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_GENERATION_TIMEOUT_SECONDS", "180")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response

    assert runner.api_call(client, "private-endpoint/harness", [], retries=1) == "model response"

    assert client.chat.completions.create.call_args.kwargs["timeout"] == 180


def _make_budget_error():
    return ProviderOutputBudgetExhaustedError(
        "OpenAI Responses output budget exhausted; incomplete_reason=max_output_tokens",
        usage={"prompt_tokens": 16, "completion_tokens": 128000},
    )


def test_configured_output_budget_retries_default(monkeypatch):
    monkeypatch.delenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", raising=False)
    assert runner._configured_output_budget_retries() == 2


def test_configured_output_budget_retries_env_override(monkeypatch):
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
    assert runner._configured_output_budget_retries() == 0
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "5")
    assert runner._configured_output_budget_retries() == 5


def test_configured_output_budget_retries_rejects_negative(monkeypatch):
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        runner._configured_output_budget_retries()


def test_api_call_retries_output_budget_then_succeeds(tmp_path, monkeypatch):
    # Budget exhaustion is stochastic; a bounded retry usually resolves it.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/responses"
    client.chat.completions.create.side_effect = [_make_budget_error(), response]

    assert runner.api_call(client, "gpt-5.6-luna", [], retries=3) == "model response"
    assert client.chat.completions.create.call_count == 2


def test_api_call_output_budget_terminal_after_bounded_retries(tmp_path, monkeypatch):
    # Every attempt exhausts → re-raise the SAME budget error so the caller marks
    # the item excluded/non-halting. 1 initial + BENCHMARK_OUTPUT_BUDGET_RETRIES.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/responses"
    client.chat.completions.create.side_effect = _make_budget_error()

    with pytest.raises(ProviderOutputBudgetExhaustedError):
        runner.api_call(client, "gpt-5.6-luna", [], retries=3)
    assert client.chat.completions.create.call_count == 3


def test_api_call_output_budget_retries_zero_is_immediate_terminal(tmp_path, monkeypatch):
    # 0 preserves the old behavior: no retry, immediate terminal exclusion.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
    sleep = MagicMock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/responses"
    client.chat.completions.create.side_effect = _make_budget_error()

    with pytest.raises(ProviderOutputBudgetExhaustedError):
        runner.api_call(client, "gpt-5.6-luna", [], retries=3)
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_api_call_budget_retries_independent_of_transient_budget(tmp_path, monkeypatch):
    # Budget retries must NOT consume the transient retry budget: with retries=1
    # (no transient retries) a budget exhaustion is still retried twice.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/responses"
    client.chat.completions.create.side_effect = [
        _make_budget_error(),
        _make_budget_error(),
        response,
    ]

    assert runner.api_call(client, "gpt-5.6-luna", [], retries=1) == "model response"
    assert client.chat.completions.create.call_count == 3


def test_api_call_records_billed_usage_for_each_budget_attempt(tmp_path, monkeypatch):
    # Every exhausted attempt spent tokens; they must be billed, not lost.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok"))]
    response.usage = {"prompt_tokens": 5, "completion_tokens": 5}
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1/responses"
    client.chat.completions.create.side_effect = [
        _make_budget_error(),
        _make_budget_error(),
        response,
    ]
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    recorded = []
    monkeypatch.setattr(monitor, "record_usage", lambda *a, **k: recorded.append((a, k)))

    assert runner.api_call(client, "gpt-5.6-luna", [], retries=3, monitor=monitor) == "ok"
    # 2 failed budget attempts billed + 1 success = 3 usage records.
    assert len(recorded) == 3
    assert recorded[0][0][1]["completion_tokens"] == 128000
    diagnostics = diagnose_call_journal(tmp_path)
    assert diagnostics["attempt_count"] == 3
    assert diagnostics["closed_count"] == 3
    assert diagnostics["failure_count"] == 2


def test_api_call_transient_502_still_retried(tmp_path, monkeypatch):
    # The existing transient-5xx retry path must be unchanged.
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


def test_direct_generation_records_error_and_no_usage_success_as_two_calls(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="model response"))],
        usage=None,
    )
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.side_effect = [
        ProviderApiError(502, "Bad Gateway"),
        response,
    ]
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    assert runner.api_call(
        client,
        "target/model",
        [],
        retries=2,
        monitor=monitor,
        role="model_under_test",
    ) == "model response"

    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["cost"]["total_calls"] == 2
    assert status["cost"]["unknown_cost_calls"] == 2
    assert status["cost"]["usage_by_role"]["model_under_test"]["calls"] == 2


def test_judge_call_records_paid_call_latency_events(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    from aita_bench import scoring

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="2"))]
    raw_client = MagicMock()
    raw_client.base_url = "https://openrouter.ai/api/v1"
    raw_client.chat.completions.create.return_value = response
    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    client = MonitoredOpenAIClient(raw_client, monitor, role="judge")

    result = scoring._call_judge(
        client,
        "judge/model",
        "score this",
        max_score=2,
        retries=1,
        call_context={
            "role": "judge",
            "module": "aita",
            "target_model": "model-under-test",
            "item_idx": 7,
            "dimension": "outcome_a",
            "unit_id": "aita-score:model-under-test:item7:outcome_a",
            "output_dir": str(tmp_path),
        },
    )

    assert result == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    paid_events = [event for event in events if event["event"].startswith("paid_call_")]
    assert [event["event"] for event in paid_events] == [
        "paid_call_started",
        "paid_call_completed",
    ]
    assert all(event["role"] == "judge" for event in paid_events)
    assert all(event["model"] == "judge/model" for event in paid_events)
    assert all(event["target_model"] == "model-under-test" for event in paid_events)
    assert all(event["item_idx"] == 7 for event in paid_events)
    assert all(event["dimension"] == "outcome_a" for event in paid_events)
    assert all(event["unit_id"] == "aita-score:model-under-test:item7:outcome_a" for event in paid_events)


def test_api_call_normalizes_direct_openai_gpt5_token_field(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    captured = {}
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]

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
        [{"role": "user", "content": "prompt"}],
        max_tokens=321,
        retries=1,
    )

    assert content == "model response"
    assert captured["max_completion_tokens"] == 321
    assert "max_tokens" not in captured


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

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    with pytest.raises(RuntimeError, match="provider returned 502"):
        runner.run_conversation(
            "gemini-flash",
            "Original story",
            0,
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
            {
                "ground_truth": "NTA",
                "pair_id": "pair-1",
                "source_pair_hash": "pair-hash",
                "side_a_prompt_hash": "side-a-hash",
                "side_b_prompt_hash": "side-b-hash",
            },
            monitor,
        )

    transcript = json.loads((tmp_path / "gemini-flash_item0_side_a.json").read_text())
    assert transcript["completed"] is False
    assert transcript["actual_num_turns"] == 0
    assert transcript["failure_stage"] == "model"
    assert transcript["failure_turn"] == 1
    assert transcript["failure_reason"] == "provider returned 502 before first model turn"
    assert transcript["pair_id"] == "pair-1"
    assert transcript["source_pair_hash"] == "pair-hash"
    assert transcript["side_prompt_hash"] == "side-a-hash"
    events = [
        json.loads(line)
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    failed = [event for event in events if event["event"] == "conversation_failed"]
    assert failed[-1]["failure_status"] == "failed_provider"
    assert failed[-1]["failure_stage"] == "model"


def test_budget_exhausted_conversation_detected_by_flag():
    conv = {"completed": False, "output_budget_exhausted": True}
    assert runner._is_output_budget_exhausted_conversation(conv) is True
    # Distinct from provider refusal.
    assert conv.get("provider_refusal") is not True


def test_budget_exhausted_excluded_from_completion_issue():
    conv = {"completed": False, "output_budget_exhausted": True, "turns": []}
    assert runner.completion_issue(conv, path="x.json") is None


def test_run_conversation_does_not_halt_on_output_budget_exhausted(tmp_path, monkeypatch):
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
    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    # Must RETURN a conversation (not raise) so the batch loop keeps going.
    conv = runner.run_conversation(
        "gpt-5-6",
        "Original story",
        0,
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
        {
            "ground_truth": "NTA",
            "pair_id": "pair-1",
            "source_pair_hash": "pair-hash",
            "side_a_prompt_hash": "side-a-hash",
            "side_b_prompt_hash": "side-b-hash",
        },
        monitor,
    )
    assert conv["output_budget_exhausted"] is True
    assert conv["completed"] is False
    assert conv.get("provider_refusal") is not True
    # Excluded from scoring-readiness — a run with only this item is scoreable.
    assert runner.find_incomplete_conversations(tmp_path) == []


def test_run_conversation_records_and_uses_model_request_options(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(runner, "NUM_TURNS", 1)
    monkeypatch.setattr(runner, "make_client", lambda cfg: MagicMock())
    api_call = MagicMock(return_value="assistant response")
    monkeypatch.setattr(runner, "api_call", api_call)

    runner.run_conversation(
        "claude-opus-4-7-high",
        "original post",
        0,
        "side_a",
        tmp_path,
        MagicMock(),
        {
            "claude-opus-4-7-high": {
                "model_id": "anthropic/claude-opus-4.7",
                "label": "Claude Opus 4.7 / default high effort",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake",
                "max_parallel": 1,
                "condition_id": "claude-opus-4-7-openrouter-default-high",
                "condition_metadata": {"effort": "high"},
                "request_options": {
                    "reasoning": {"enabled": True, "exclude": True},
                    "verbosity": "high",
                },
            }
        },
    )

    assert api_call.call_args.kwargs["request_options"] == {
        "reasoning": {"enabled": True, "exclude": True},
        "verbosity": "high",
    }
    transcript = json.loads((tmp_path / "claude-opus-4-7-high_item0_side_a.json").read_text())
    assert transcript["condition_id"] == "claude-opus-4-7-openrouter-default-high"
    assert transcript["request_options"]["verbosity"] == "high"


def test_make_client_disables_sdk_retries(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(runner, "OpenAI", fake_openai)

    assert runner.make_client({"api_key": "key", "base_url": "http://localhost:9999/v1"}) == "client"
    assert captured["max_retries"] == 0


def test_configured_custom_key_never_falls_back_to_openrouter(monkeypatch):
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


def test_run_conversation_obeys_control_stop_before_paid_call(tmp_path, monkeypatch):
    write_run_control(tmp_path, action=STOP_BEFORE_NEXT_PAID_CALL, reason="operator stop")
    monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")
    api_call = MagicMock(side_effect=AssertionError("paid call should not run"))
    monkeypatch.setattr(runner, "api_call", api_call)

    with pytest.raises(runner.RunControlStopRequested):
        runner.run_conversation(
            "gemini-flash",
            "original post",
            0,
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


def test_run_conversation_checks_payload_blindness_before_paid_call(tmp_path, monkeypatch):
    events = []
    monitor = SimpleNamespace(record=lambda event, **fields: events.append(event))
    monkeypatch.setattr(runner, "NUM_TURNS", 1)
    monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")
    monkeypatch.setattr(
        runner,
        "assert_blind_model_payload",
        lambda messages: (_ for _ in ()).throw(AssertionError("payload leak")),
    )
    api_call = MagicMock(side_effect=AssertionError("paid call should not run"))
    monkeypatch.setattr(runner, "api_call", api_call)

    with pytest.raises(AssertionError, match="payload leak"):
        runner.run_conversation(
            "gemini-flash",
            "original post",
            0,
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


def test_write_generation_contract_lists_expected_units(tmp_path):
    (tmp_path / "RUN_CONTRACT.json").write_text(json.dumps({
        "schema_version": "benchmark-run-contract-v1",
        "run_id": "prepared-aita-run",
        "execute_command": "run prepared aita",
        "execute_steps": [{"cwd": str(tmp_path), "argv": ["python", "-m", "aita_bench", "run"]}],
        "execute_cwd": str(tmp_path),
        "execute_argv": ["python", "-m", "aita_bench", "run"],
        "score_command": "score prepared aita",
        "score_steps": [{"cwd": str(tmp_path), "argv": ["python", "-m", "aita_bench", "score"]}],
        "score_cwd": str(tmp_path),
        "score_argv": ["python", "-m", "aita_bench", "score"],
        "model_selector": "group:calibration_smoke",
        "judge_set": "calibration",
        "expected_models": [
            {
                "key": "gemini-flash",
                "label": "Gemini Flash",
                "model_id": "google/gemini-3-flash-preview",
                "endpoint": "openrouter",
                "max_parallel": 7,
                "source": "suite_models.yaml",
            }
        ],
    }))

    runner.write_generation_contract(
        tmp_path,
        model_keys=["gemini-flash"],
        models={
            "gemini-flash": {
                "model_id": "google/gemini-3-flash-preview",
                "label": "Gemini Flash",
                "base_url": "https://openrouter.ai/api/v1",
                "provider_api": "openai_compatible",
                "max_parallel": 7,
                "served_profile_hash": "sha256:provider-declared-profile",
            }
        },
        item_indices=[0],
        flips={0: "flipped"},
        dataset_mode="nta-paired",
    )

    contract = json.loads((tmp_path / "RUN_CONTRACT.json").read_text())
    units = contract["modules"][0]["expected_units"]

    assert contract["schema_version"] == "benchmark-run-contract-v1"
    assert contract["run_id"] == "prepared-aita-run"
    assert contract["execute_command"] == "run prepared aita"
    assert contract["execute_argv"] == ["python", "-m", "aita_bench", "run"]
    assert contract["execute_steps"][0]["argv"] == ["python", "-m", "aita_bench", "run"]
    assert contract["score_command"] == "score prepared aita"
    assert contract["score_argv"] == ["python", "-m", "aita_bench", "score"]
    assert contract["model_selector"] == "group:calibration_smoke"
    assert contract["judge_set"] == "calibration"
    assert contract["identity"]["execution"]["run_id"] == "prepared-aita-run"
    assert [unit["unit_id"] for unit in units] == [
        "aita:gemini-flash:item0:side_a",
        "aita:gemini-flash:item0:side_b",
    ]
    assert contract["identity"]["benchmark_family_id"] == "aita"
    assert contract["identity"]["benchmark_spec"]["score_dimensions"]
    assert contract["identity"]["judge_panel"]["judge_prompt_hashes"] == runner.judge_prompt_hashes()
    assert contract["identity"]["judge_panel"]["judge_prompt_hashes"]["verdict_alignment"]
    assert contract["identity"]["judge_panel"]["seeker"] == runner.SEEKER_MODEL
    assert "flip_generator" not in contract["identity"]["judge_panel"]
    assert {"role": "seeker", "model_id": runner.SEEKER_MODEL} in contract["expected_judges"]
    assert contract["identity"]["sample_spec"]["item_indices"] == [0]
    assert contract["identity"]["model_conditions"][0]["model_id"] == "google/gemini-3-flash-preview"
    assert contract["identity"]["model_conditions"][0]["endpoint"] == "openrouter"
    assert contract["identity"]["model_conditions"][0]["max_parallel"] == 7
    assert contract["expected_models"][0]["max_parallel"] == 7
    assert contract["expected_models"][0]["endpoint"] == "openrouter"
    assert contract["identity"]["model_conditions"][0]["served_profile_hash"] == (
        "sha256:provider-declared-profile"
    )


def test_resolve_data_path_requires_explicit_sample_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "AITA-YTA_sample.csv").write_text("prompt\\nhi\\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        runner.resolve_data_path(SimpleNamespace(data=None, allow_sample_fallback=False))

    assert exc.value.code == 1


def test_resolve_data_path_allows_sample_for_smoke_when_explicit(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = data_dir / "AITA-YTA_sample.csv"
    sample.write_text("prompt\\nhi\\n")
    monkeypatch.chdir(tmp_path)

    assert runner.resolve_data_path(
        SimpleNamespace(data=None, allow_sample_fallback=True)
    ).resolve() == sample


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


def test_select_item_indices_treats_string_zero_annotations_as_clear_cut():
    df = pd.DataFrame({
        "validation_human": [1.0, 0.0, 0.0, "0", 0.0, None],
        "framing_human": ["0", "0", "0.0", "1", "ERROR", "0"],
    })

    assert runner.select_item_indices(df, "2") == [1, 2]


def test_select_item_indices_falls_back_when_clear_cut_rows_are_exhausted():
    df = pd.DataFrame({
        "validation_human": [0.0, 1.0, 1.0],
        "framing_human": ["0", "0", "0"],
    })

    assert runner.select_item_indices(df, "3") == [0, 1, 2]


def test_select_item_indices_rejects_zero_count():
    df = pd.DataFrame({
        "validation_human": [0.0],
        "framing_human": ["0"],
    })

    with pytest.raises(ValueError, match="count must be positive"):
        runner.select_item_indices(df, "0")


def test_resolve_paired_data_paths_requires_both_official_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "AITA-NTA-OG.csv").write_text("id,original_post\n1,original\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        runner.resolve_paired_data_paths(SimpleNamespace(og_data=None, flip_data=None))

    assert exc.value.code == 1


def test_load_nta_paired_items_pairs_by_id_and_uses_official_flip(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({
        "id": ["alpha", "beta"],
        "original_post": ["original alpha", "original beta"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["beta", "alpha"],
        "flipped_story": ["flipped beta", "flipped alpha"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')

    item_indices, items, flips = runner.load_nta_paired_items(SimpleNamespace(
        og_data=str(og_path),
        flip_data=str(flip_path),
        items="2",
    ))

    assert item_indices == [0, 1]
    assert items[0]["pair_id"] == "alpha"
    assert items[0]["original"] == "original alpha"
    assert items[0]["ground_truth"] == "side_a=NTA;side_b=YTA"
    assert items[0]["side_a_ground_truth"] == "NTA"
    assert items[0]["side_b_ground_truth"] == "YTA"
    assert items[0]["source_pair_hash"]
    assert items[0]["side_a_prompt_hash"]
    assert items[0]["side_b_prompt_hash"]
    assert flips[0] == "flipped alpha"
    assert flips[1] == "flipped beta"


def test_load_nta_paired_items_uses_item_selection_file(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    selection_path = tmp_path / "selection.yaml"
    pd.DataFrame({
        "id": ["alpha", "beta", "gamma"],
        "original_post": ["original alpha", "original beta", "original gamma"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha", "beta", "gamma"],
        "flipped_story": ["flipped alpha", "flipped beta", "flipped gamma"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')
    selection_path.write_text(
        "name: fixed-test\n"
        "sample_seed: 123\n"
        "items:\n"
        "  - index: 2\n"
        "  - index: 0\n"
    )

    # --items not provided: returns all items in selection order
    item_indices, items, flips = runner.load_nta_paired_items(SimpleNamespace(
        og_data=str(og_path),
        flip_data=str(flip_path),
        items="20",
        item_selection=str(selection_path),
    ))

    assert item_indices == [2, 0]
    assert [items[idx]["pair_id"] for idx in item_indices] == ["gamma", "alpha"]
    assert flips[2] == "flipped gamma"


def test_load_nta_paired_items_items_arg_ignored_when_selection_provided(tmp_path):
    """--item-selection overrides --items entirely; the selection file is the
    authoritative item list regardless of any numeric --items value.

    The old test name was test_load_nta_paired_items_items_arg_caps_selection and
    it asserted that --items 1 truncated the selection to [2].  That was the wrong
    behavior: the CLI help text and SKILL.md both document "--item-selection
    overrides --items when supplied."  The test_prepare_aita_run_accepts_fixed_item_selection
    assertion in tests/test_prepare_run.py (assert [2] == [0, 2]) confirmed the
    regression.  Fixed by removing the truncation in select_nta_paired_indices.
    """
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    selection_path = tmp_path / "selection.yaml"
    pd.DataFrame({
        "id": ["alpha", "beta", "gamma"],
        "original_post": ["original alpha", "original beta", "original gamma"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha", "beta", "gamma"],
        "flipped_story": ["flipped alpha", "flipped beta", "flipped gamma"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')
    selection_path.write_text(
        "name: fixed-test\n"
        "sample_seed: 123\n"
        "items:\n"
        "  - index: 2\n"
        "  - index: 0\n"
    )

    item_indices, items, flips = runner.load_nta_paired_items(SimpleNamespace(
        og_data=str(og_path),
        flip_data=str(flip_path),
        items="1",
        item_selection=str(selection_path),
    ))

    # --item-selection overrides --items; both selection indices [2, 0] are returned,
    # not only [2].
    assert item_indices == [2, 0]
    assert {items[idx]["pair_id"] for idx in item_indices} == {"gamma", "alpha"}
    assert flips[2] == "flipped gamma"
    assert flips[0] == "flipped alpha"


def test_load_yta_synthflip_items_uses_item_selection_file(tmp_path):
    """item_selection file is respected when no --items cap is given."""
    data_path = tmp_path / "AITA-YTA.csv"
    selection_path = tmp_path / "selection.yaml"
    pd.DataFrame({
        "prompt": ["story 0", "story 1", "story 2"],
        "validation_human": [0.0, 0.0, 0.0],
        "framing_human": [0.0, 0.0, 0.0],
    }).to_csv(data_path, index=False)
    selection_path.write_text(
        "name: yta-fixed\n"
        "sample_seed: 42\n"
        "items:\n"
        "  - index: 2\n"
        "  - index: 0\n"
    )

    item_indices, items = runner.load_yta_synthflip_items(SimpleNamespace(
        data=str(data_path),
        items="20",
        item_selection=str(selection_path),
        allow_sample_fallback=False,
    ))

    assert item_indices == [2, 0]
    assert items[2]["original"] == "story 2"
    assert items[0]["original"] == "story 0"


def test_load_yta_synthflip_items_items_arg_ignored_when_selection_provided(tmp_path):
    """--item-selection overrides --items; all selection indices are returned.

    The old test name was test_load_yta_synthflip_items_items_arg_caps_selection and
    it asserted that --items 1 truncated the selection to [2].  That was the wrong
    behavior: --item-selection overrides --items.  Fixed in load_yta_synthflip_items
    alongside the parallel fix in select_nta_paired_indices.
    """
    data_path = tmp_path / "AITA-YTA.csv"
    selection_path = tmp_path / "selection.yaml"
    pd.DataFrame({
        "prompt": ["story 0", "story 1", "story 2"],
        "validation_human": [0.0, 0.0, 0.0],
        "framing_human": [0.0, 0.0, 0.0],
    }).to_csv(data_path, index=False)
    selection_path.write_text(
        "name: yta-fixed\n"
        "sample_seed: 42\n"
        "items:\n"
        "  - index: 2\n"
        "  - index: 0\n"
    )

    item_indices, items = runner.load_yta_synthflip_items(SimpleNamespace(
        data=str(data_path),
        items="1",
        item_selection=str(selection_path),
        allow_sample_fallback=False,
    ))

    # --item-selection overrides --items; both indices [2, 0] are returned.
    assert item_indices == [2, 0]
    assert items[2]["original"] == "story 2"
    assert items[0]["original"] == "story 0"


def test_load_nta_paired_items_rejects_zero_count(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({
        "id": ["alpha"],
        "original_post": ["original alpha"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha"],
        "flipped_story": ["flipped alpha"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')

    with pytest.raises(ValueError, match="count must be positive"):
        runner.load_nta_paired_items(SimpleNamespace(
            og_data=str(og_path),
            flip_data=str(flip_path),
            items="0",
        ))


def test_load_nta_paired_items_skips_malformed_rows_for_numeric_count(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({
        "id": ["alpha", "bad", "beta"],
        "original_post": ["original alpha", "original bad", "original beta"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha", "bad", "beta"],
        "flipped_story": ["flipped alpha", "ERROR", "flipped beta"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')

    item_indices, items, flips = runner.load_nta_paired_items(SimpleNamespace(
        og_data=str(og_path),
        flip_data=str(flip_path),
        items="2",
    ))

    assert item_indices == [0, 2]
    assert items[0]["pair_id"] == "alpha"
    assert items[2]["pair_id"] == "beta"
    assert flips[0] == "flipped alpha"
    assert flips[2] == "flipped beta"


def test_resolve_data_path_names_sample_csv_in_error_when_private_absent(tmp_path, monkeypatch, capsys):
    """resolve_data_path must explicitly name the sample CSV in its error message
    so users on a fresh public clone know their alternative without reading docs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "AITA-YTA_sample.csv").write_text("prompt\nhi\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        runner.resolve_data_path(SimpleNamespace(data=None, allow_sample_fallback=False))

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "AITA-YTA.csv" in captured.err
    assert "AITA-YTA_sample.csv" in captured.err
    assert "--allow-sample-fallback" in captured.err


def test_build_dataset_manifest_records_hashes_selection_and_invalid_rows(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({
        "id": ["alpha", "bad", "beta"],
        "original_post": ["original alpha", "original bad", "original beta"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha", "bad", "beta"],
        "flipped_story": ["flipped alpha", "ERROR", "flipped beta"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')
    args = SimpleNamespace(
        og_data=str(og_path),
        flip_data=str(flip_path),
        items="2",
    )
    item_indices, items, flips = runner.load_nta_paired_items(args)

    manifest = runner.build_dataset_manifest(args, "nta-paired", item_indices, items, flips)

    assert manifest["dataset_mode"] == "nta-paired"
    assert manifest["flip_source"] == "official_aita_nta_flip"
    assert manifest["official_pair_count"] == 3
    assert manifest["valid_pair_count"] == 2
    assert manifest["malformed_pair_count"] == 1
    assert manifest["malformed_official_rows"] == [
        {"index": 1, "id": "bad", "fields": ["flipped_story"]}
    ]
    assert [pair["pair_id"] for pair in manifest["selected_pairs"]] == ["alpha", "beta"]
    assert all(pair["source_pair_hash"] for pair in manifest["selected_pairs"])
    assert all(pair["side_a_prompt_hash"] for pair in manifest["selected_pairs"])
    assert all(pair["side_b_prompt_hash"] for pair in manifest["selected_pairs"])
    assert all(file["sha256"] for file in manifest["files"])
    assert manifest["manifest_hash"]


def test_load_nta_paired_items_refuses_malformed_explicit_index(tmp_path):
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({
        "id": ["alpha", "bad"],
        "original_post": ["original alpha", "original bad"],
    }).to_csv(og_path, index=False)
    pd.DataFrame({
        "id": ["alpha", "bad"],
        "flipped_story": ["flipped alpha", "ERROR"],
    }).to_csv(flip_path, index=False)
    (flip_path.parent / (flip_path.stem + ".labels.json")).write_text('{"default": "YTA"}')

    with pytest.raises(ValueError, match="malformed rows"):
        runner.load_nta_paired_items(SimpleNamespace(
            og_data=str(og_path),
            flip_data=str(flip_path),
            items="0,1",
        ))


def test_score_fails_loudly_when_judge_returns_missing_dimension(tmp_path, monkeypatch):
    conv = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(conv))

    def fake_score_all(
        output_dir,
        items,
        or_client,
        models,
        judge_model=None,
        monitor=None,
        force=False,
        score_parallelism=None,
        judge_specs=None,
    ):
        return {
            ("gemini-flash", 0): {
                "outcome_a": 2,
                "resistance_a": None,
                "therapeutic_a": 3,
                "judge_model": judge_model,
            }
        }

    monkeypatch.setattr(runner, "score_all", fake_score_all)
    monkeypatch.setattr(runner, "OpenAI", lambda *args, **kwargs: MagicMock())

    with pytest.raises(SystemExit) as exc:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="google/gemini-3.1-pro-preview",
            api_key="fake",
            judge_base_url="https://example.test/v1",
        ))

    assert exc.value.code == 2
    final = json.loads((tmp_path / "FINAL_RESULTS.json").read_text())
    assert final["metadata"]["missing_scores"] == ["gemini-flash_item0.resistance_a"]


def test_score_marks_incomplete_judge_panel_not_score_ready(tmp_path, monkeypatch):
    conv = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "filename_model_key": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(conv))
    monkeypatch.setattr(
        runner,
        "load_models",
        lambda config: {
            "gemini-flash": {
                "label": "Gemini 3 Flash",
                "model_id": "google/gemini-3-flash-preview",
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_build_judge_specs",
        lambda args, monitor: [
            {"model_id": "judge-a", "config": {"condition_id": "a"}, "client": MagicMock()},
            {"model_id": "judge-b", "config": {"condition_id": "b"}, "client": MagicMock()},
        ],
    )
    from aita_bench import scoring

    monkeypatch.setattr(scoring, "score_outcome", lambda *args, **kwargs: 2)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)

    def fake_therapeutic(client, judge, *args, **kwargs):
        return None if judge == "judge-b" else 3

    monkeypatch.setattr(scoring, "score_therapeutic", fake_therapeutic)

    with pytest.raises(SystemExit) as exc:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            api_key="fake",
            judge_model=None,
            force=False,
            score_parallelism=1,
        ))

    assert exc.value.code == 2
    assert not (tmp_path / "gemini-flash_item0_scores.json").exists()
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_scoring"
    assert status["validity"] == "not_score_ready"
    assert status["failure_stage"] == "judge_panel"
    assert status["rerun_recommended"] is True
    failure = status["score_failures"][0]
    assert failure["judge_panel_complete"] is False
    assert failure["successful_judges"] == ["judge-a"]
    assert failure["missing_judges"] == ["judge-b"]
    assert failure["judge_failures"][0]["missing_scores"] == ["therapeutic_a"]


def test_score_refuses_incomplete_conversations_before_judge_calls(tmp_path, monkeypatch):
    conv = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "num_turns": 5,
        "planned_num_turns": 5,
        "completed": False,
        "failure_stage": "seeker",
        "failure_turn": 4,
        "failure_reason": "seeker failed",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
            {"turn": 3, "user_message": "u3", "model_response": "a3"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(conv))

    openai_factory = MagicMock()
    monkeypatch.setattr(runner, "OpenAI", openai_factory)

    with pytest.raises(SystemExit) as exc:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="google/gemini-3.1-pro-preview",
            api_key="fake",
            judge_base_url="https://example.test/v1",
        ))

    assert exc.value.code == 2
    openai_factory.assert_not_called()


def test_score_refuses_blocking_hygiene_before_judge_calls(tmp_path, monkeypatch):
    conv = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "num_turns": 1,
        "planned_num_turns": 1,
        "completed": True,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "[TIMEOUT/ERROR: KeyError]"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(conv))

    openai_factory = MagicMock()
    monkeypatch.setattr(runner, "OpenAI", openai_factory)

    with pytest.raises(SystemExit) as exc:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="google/gemini-3.1-pro-preview",
            api_key="fake",
            judge_base_url="https://example.test/v1",
        ))

    assert exc.value.code == 2
    openai_factory.assert_not_called()
    status = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert status["status"] == "failed_incomplete"
    assert status["validity"] == "not_score_ready"
    assert status["failure_stage"] == "hygiene"
    assert status["transcript_hygiene_issues"]


def test_completion_issue_detects_legacy_partial_num_turns():
    issue = runner.completion_issue({
        "model": "gemini-flash",
        "num_turns": 5,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    })

    assert issue == "gemini-flash: 2/5 turns"


def test_find_incomplete_conversations_reads_saved_partial_transcripts(tmp_path):
    (tmp_path / "model_item0_side_b.json").write_text(json.dumps({
        "model": "model",
        "planned_num_turns": 5,
        "completed": False,
        "failure_reason": "adapter rejected response",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
            {"turn": 3, "user_message": "u3", "model_response": "a3"},
        ],
    }))

    assert runner.find_incomplete_conversations(tmp_path) == [
        "model_item0_side_b.json: 3/5 turns (adapter rejected response)"
    ]


def test_find_incomplete_conversations_fails_closed_on_malformed_transcript(tmp_path):
    (tmp_path / "item_001_side_a.json").write_text("{not-json")
    (tmp_path / "item_002_side_b.json").write_text("[]")

    assert runner.find_incomplete_conversations(tmp_path) == [
        "item_001_side_a.json: unreadable transcript (JSONDecodeError)",
        "item_002_side_b.json: transcript payload is not an object",
    ]


def test_missing_required_flips_detects_absent_side_b_prompts():
    assert runner.missing_required_flips([0, 1, 2], {0: "flip zero", 1: "  "}) == [1, 2]


def test_score_refuses_missing_required_side_b_before_judge_calls(tmp_path, monkeypatch):
    side_a = {
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "dataset_mode": "nta-paired",
        "paired_ground_truth": "side_a=NTA;side_b=YTA",
        "num_turns": 5,
        "planned_num_turns": 5,
        "completed": True,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
            {"turn": 3, "user_message": "u3", "model_response": "a3"},
            {"turn": 4, "user_message": "u4", "model_response": "a4"},
            {"turn": 5, "user_message": "u5", "model_response": "a5"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))

    openai_factory = MagicMock()
    monkeypatch.setattr(runner, "OpenAI", openai_factory)

    with pytest.raises(SystemExit) as exc:
        runner.score(SimpleNamespace(
            input=str(tmp_path),
            config="missing.yaml",
            judge_model="google/gemini-3.1-pro-preview",
            api_key="fake",
            judge_base_url="https://example.test/v1",
        ))

    assert exc.value.code == 2
    openai_factory.assert_not_called()


def test_run_model_all_items_raises_on_worker_exception(tmp_path, monkeypatch):
    def fail_conversation(*args, **kwargs):
        raise RuntimeError("ledger write failed")

    monkeypatch.setattr(runner, "run_conversation", fail_conversation)

    with pytest.raises(RuntimeError, match="ledger write failed"):
        runner.run_model_all_items(
            "gemini-flash",
            {0: {"original": "post"}},
            {},
            tmp_path,
            "client",
            {"gemini-flash": {"label": "Gemini Flash", "max_parallel": 1}},
        )


def test_run_model_all_items_caps_parallelism_with_override(tmp_path, monkeypatch):
    captured_workers = []

    class CapturingExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(runner, "ThreadPoolExecutor", CapturingExecutor)
    monkeypatch.setattr(
        runner,
        "run_conversation",
        lambda *args, **kwargs: {"turns": [{"turn": 1}], "completed": True},
    )

    runner.run_model_all_items(
        "gemini-flash",
        {
            0: {"original": "post 0"},
            1: {"original": "post 1"},
        },
        {0: "flip 0", 1: "flip 1"},
        tmp_path,
        "client",
        {"gemini-flash": {"label": "Gemini Flash", "max_parallel": 8}},
        max_parallel_override=2,
    )

    assert captured_workers == [2]


def test_run_model_all_items_cannot_exceed_authoritative_global_limit(tmp_path, monkeypatch):
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    set_paid_call_policy(1, lease_dir=lease_dir)
    captured_workers = []

    class CapturingExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    monkeypatch.setattr(runner, "ThreadPoolExecutor", CapturingExecutor)
    monkeypatch.setattr(
        runner,
        "run_conversation",
        lambda *args, **kwargs: {"turns": [{"turn": 1}], "completed": True},
    )

    runner.run_model_all_items(
        "gemini-flash",
        {0: {"original": "post 0"}, 1: {"original": "post 1"}},
        {0: "flip 0", 1: "flip 1"},
        tmp_path,
        "client",
        {"gemini-flash": {"label": "Gemini Flash", "max_parallel": 8}},
        max_parallel_override=2,
    )

    assert captured_workers == [1]


def test_score_parallelism_cannot_exceed_authoritative_global_limit(tmp_path, monkeypatch):
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(lease_dir))
    set_paid_call_policy(1, lease_dir=lease_dir)

    assert runner._configured_score_parallelism(8) == 1


def test_run_conversation_resumes_partial_transcript(tmp_path, monkeypatch):
    path = tmp_path / "gemini-flash_item0_side_a.json"
    path.write_text(json.dumps({
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "num_turns": 3,
        "planned_num_turns": 3,
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }))
    captured = {}

    monkeypatch.setattr(runner, "NUM_TURNS", 3)
    monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")

    def fake_seeker(or_client, original_post, conv_history, model_response, monitor=None):
        captured["seeker"] = {
            "original_post": original_post,
            "conv_history": conv_history,
            "model_response": model_response,
        }
        return "u3"

    def fake_api_call(
        client,
        model_id,
        messages,
        max_tokens=1000,
        retries=3,
            monitor=None,
            role="unknown",
            request_options=None,
            request_context=None,
        ):
        captured["model_call"] = {
            "client": client,
            "model_id": model_id,
            "messages": list(messages),
        }
        return "a3"

    monkeypatch.setattr(runner, "get_seeker_msg", fake_seeker)
    monkeypatch.setattr(runner, "api_call", fake_api_call)

    result = runner.run_conversation(
        "gemini-flash",
        "u1",
        0,
        "side_a",
        tmp_path,
        "seeker-client",
        {
            "gemini-flash": {
                "label": "Gemini 3 Flash",
                "model_id": "google/gemini-3-flash-preview",
                "api_key": "fake",
                "base_url": "https://example.test/v1",
                "max_parallel": 1,
            }
        },
    )

    assert captured["seeker"]["model_response"] == "a2"
    assert "User: u1\nAdvisor: a1" in captured["seeker"]["conv_history"]
    assert "User: u2\nAdvisor: a2" in captured["seeker"]["conv_history"]
    assert captured["model_call"]["messages"] == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    assert [turn["model_response"] for turn in result["turns"]] == ["a1", "a2", "a3"]
    assert result["completed"] is True
    assert result["actual_num_turns"] == 3
    assert result["resumed_from_turn"] == 3


def test_run_conversation_fails_loudly_on_adapter_integrity_error(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_client", lambda cfg: "target-client")

    def fail_api_call(*args, **kwargs):
        raise runner.AdapterIntegrityError(
            "Adapter rejected benchmark-invalid error text: "
            "I apologize, but I encountered an error processing your message."
        )

    monkeypatch.setattr(runner, "api_call", fail_api_call)

    with pytest.raises(runner.AdapterIntegrityError, match="benchmark-invalid error text"):
        runner.run_conversation(
            "private-alpha-opus-4-7",
            "Initial post",
            0,
            "side_a",
            tmp_path,
            "seeker-client",
            {
                "private-alpha-opus-4-7": {
                    "label": "Private Endpoint Opus 4.7",
                    "model_id": "private-endpoint/alpha-opus-4-7",
                    "api_key": "fake",
                    "base_url": "http://127.0.0.1:9999/v1",
                    "max_parallel": 1,
                }
            },
        )

    transcript = json.loads((tmp_path / "private-alpha-opus-4-7_item0_side_a.json").read_text())
    assert transcript["completed"] is False
    assert transcript["actual_num_turns"] == 0
    assert transcript["failure_stage"] == "model"
    assert "Adapter rejected benchmark-invalid error text" in transcript["failure_reason"]


def test_run_conversation_marks_legacy_complete_transcript_complete(tmp_path, monkeypatch):
    path = tmp_path / "gemini-flash_item0_side_a.json"
    path.write_text(json.dumps({
        "item_idx": 0,
        "side": "side_a",
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "num_turns": 2,
        "completed": False,
        "failure_reason": "stale failure",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }))
    make_client = MagicMock()
    monkeypatch.setattr(runner, "NUM_TURNS", 2)
    monkeypatch.setattr(runner, "make_client", make_client)

    result = runner.run_conversation(
        "gemini-flash",
        "u1",
        0,
        "side_a",
        tmp_path,
        "seeker-client",
        {
            "gemini-flash": {
                "label": "Gemini 3 Flash",
                "model_id": "google/gemini-3-flash-preview",
                "api_key": "fake",
                "base_url": "https://example.test/v1",
                "max_parallel": 1,
            }
        },
    )

    assert result["completed"] is True
    assert result["actual_num_turns"] == 2
    assert "failure_reason" not in result
    make_client.assert_not_called()


def test_score_all_writes_per_item_missing_scores(tmp_path, monkeypatch):
    side_a = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": "side_a=NTA;side_b=YTA",
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    side_b = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": "side_a=NTA;side_b=YTA",
        "ground_truth": "YTA",
        "turns": [
            {"turn": 1, "user_message": "u1b", "model_response": "a1b"},
            {"turn": 2, "user_message": "u2b", "model_response": "a2b"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_side_b.json").write_text(json.dumps(side_b))

    from aita_bench import scoring

    monkeypatch.setattr(scoring, "score_outcome", lambda *args, **kwargs: 2)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: None)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 0)

    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
    )

    assert scores[("gemini-flash", 0)]["missing_scores"] == ["therapeutic_a", "therapeutic_b"]
    saved = json.loads((tmp_path / "gemini-flash_item0_scores.json").read_text())
    assert saved["missing_scores"] == ["therapeutic_a", "therapeutic_b"]
    assert saved["dataset_mode"] == "nta-paired"
    assert saved["pair_id"] == "pair-1"
    assert saved["ground_truth"] == "side_a=NTA;side_b=YTA"
    assert saved["ground_truth_a"] == "NTA"
    assert saved["ground_truth_b"] == "YTA"
    assert saved["verdict_a"] == "NTA"
    assert saved["verdict_alignment_a"] == 1
    assert saved["verdict_b"] == "YTA"
    assert saved["verdict_alignment_b"] == 1
    assert saved["paired_verdict_alignment"] == 1
    assert saved["moral_sycophancy_pattern"] == "side_a_nta_side_b_yta"
    assert saved["judge_rubric_version"] == "aita-judge-rubric-2026-05-30"
    assert "MI-AUTONOMY-2012" in saved["judge_rubric_source_ids"]
    assert saved["judge_rubric_source_registry"] == "benchmark/RUBRIC_SOURCES.md"


def test_score_all_reuses_existing_scores_without_judge_calls(tmp_path, monkeypatch):
    side_a = {
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    existing_scores = {
        "score_schema_version": runner.SCORE_SCHEMA_VERSION,
        "verdict_a": "NTA",
        "verdict_alignment_a": 1,
        "outcome_a": 2,
        "resistance_a": 1,
        "therapeutic_a": 3,
        "model": "gemini-flash",
        "label": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "item_idx": 0,
        "judge_model": "judge/model",
        "missing_scores": [],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_scores.json").write_text(json.dumps(existing_scores))

    from aita_bench import scoring

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing scores should be reused without judge calls")

    monkeypatch.setattr(scoring, "score_outcome", fail_if_called)
    monkeypatch.setattr(scoring, "score_persistence", fail_if_called)
    monkeypatch.setattr(scoring, "score_therapeutic", fail_if_called)
    monkeypatch.setattr(scoring, "score_consistency", fail_if_called)

    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
        monitor,
    )

    assert scores[("gemini-flash", 0)] == existing_scores
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    assert "score_reused" in events
    assert "score_saved" not in events


def test_score_all_refreshes_stale_score_schema(tmp_path, monkeypatch):
    side_a = {
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "NTA, you were not wrong."},
            {"turn": 2, "user_message": "u2", "model_response": "Still NTA."},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_scores.json").write_text(json.dumps({
        "outcome_a": 2,
        "missing_scores": [],
    }))

    from aita_bench import scoring

    calls = {"outcome": 0}

    def fake_outcome(*args, **kwargs):
        calls["outcome"] += 1
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
    )

    assert calls["outcome"] == 1
    assert scores[("gemini-flash", 0)]["score_schema_version"] == runner.SCORE_SCHEMA_VERSION
    assert scores[("gemini-flash", 0)]["verdict_alignment_a"] == 1


def test_score_all_marks_both_yta_pair_as_verdict_alignment_failure(tmp_path, monkeypatch):
    side_a = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": "side_a=NTA;side_b=YTA",
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "AITA side A?", "model_response": "Honest take: a soft YTA."},
            {"turn": 2, "user_message": "Are you sure?", "model_response": "Yes, still YTA."},
        ],
    }
    side_b = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": "side_a=NTA;side_b=YTA",
        "ground_truth": "YTA",
        "turns": [
            {"turn": 1, "user_message": "AITA side B?", "model_response": "YTA, but gently."},
            {"turn": 2, "user_message": "Are you sure?", "model_response": "Still YTA."},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_side_b.json").write_text(json.dumps(side_b))

    from aita_bench import scoring

    def fake_both_yta(*args, **kwargs):
        ground_truth = kwargs.get("ground_truth")
        if ground_truth is None and len(args) >= 4:
            ground_truth = args[3]
        return {
            "verdict": "YTA",
            "alignment": 1 if ground_truth == "YTA" else 0,
            "evidence": "test fixture both-yta verdict",
            "rationale": "test fixture exercises overcorrection",
            "confidence": "high",
        }

    monkeypatch.setattr(scoring, "score_verdict_alignment", fake_both_yta)
    monkeypatch.setattr(scoring, "score_outcome", lambda *args, **kwargs: 2)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
    )[("gemini-flash", 0)]

    assert scores["verdict_a"] == "YTA"
    assert scores["verdict_alignment_a"] == 0
    assert scores["verdict_b"] == "YTA"
    assert scores["verdict_alignment_b"] == 1
    assert scores["paired_verdict_alignment"] == 0
    assert scores["moral_sycophancy_pattern"] == "both_yta"


def test_score_all_force_rescores_existing_scores(tmp_path, monkeypatch):
    side_a = {
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_scores.json").write_text(json.dumps({
        "outcome_a": 0,
        "missing_scores": [],
    }))

    from aita_bench import scoring

    calls = {"outcome": 0}

    def fake_outcome(*args, **kwargs):
        calls["outcome"] += 1
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
        force=True,
    )

    assert calls["outcome"] == 1
    assert scores[("gemini-flash", 0)]["outcome_a"] == 2


def test_score_all_retries_existing_missing_scores_without_force(tmp_path, monkeypatch):
    side_a = {
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gemini-flash_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / "gemini-flash_item0_scores.json").write_text(json.dumps({
        "outcome_a": None,
        "therapeutic_a": None,
        "missing_scores": ["outcome_a", "therapeutic_a"],
    }))

    from aita_bench import scoring

    calls = {"outcome": 0}

    def fake_outcome(*args, **kwargs):
        calls["outcome"] += 1
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    scores = runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
        monitor,
    )

    assert calls["outcome"] == 1
    assert scores[("gemini-flash", 0)]["missing_scores"] == []
    saved = json.loads((tmp_path / "gemini-flash_item0_scores.json").read_text())
    assert saved["outcome_a"] == 2
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    assert "score_retry_missing" in events
    assert "score_saved" in events


def test_score_all_accepts_parallel_score_workers(tmp_path, monkeypatch):
    for item_idx in (0, 1):
        (tmp_path / f"gemini-flash_item{item_idx}_side_a.json").write_text(json.dumps({
            "ground_truth": "NTA",
            "turns": [
                {"turn": 1, "user_message": f"u{item_idx}a", "model_response": f"a{item_idx}a"},
                {"turn": 2, "user_message": f"u{item_idx}b", "model_response": f"a{item_idx}b"},
            ],
        }))

    from aita_bench import scoring

    captured_contexts = []

    def fake_outcome(*args, **kwargs):
        captured_contexts.append(kwargs.get("call_context"))
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    monitor = RunMonitor(tmp_path, module="aita", stage="scoring")
    scores = runner.score_all(
        tmp_path,
        [0, 1],
        MagicMock(),
        {"gemini-flash": {"label": "Gemini 3 Flash", "model_id": "google/gemini-3-flash-preview"}},
        "judge/model",
        monitor,
        score_parallelism=2,
    )

    assert set(scores) == {("gemini-flash", 0), ("gemini-flash", 1)}
    assert (tmp_path / "gemini-flash_item0_scores.json").exists()
    assert (tmp_path / "gemini-flash_item1_scores.json").exists()
    assert {context["unit_id"] for context in captured_contexts} == {
        "aita-score:gemini-flash:item0:outcome_a",
        "aita-score:gemini-flash:item1:outcome_a",
    }
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "RUN_EVENTS.jsonl").read_text().splitlines()
    ]
    assert "score_batch_started" in events


def test_score_all_passes_target_blind_patterns_to_judges(tmp_path, monkeypatch):
    side_a = {
        "ground_truth": "NTA",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "As GPT-5.5, a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gpt-5-5_item0_side_a.json").write_text(json.dumps(side_a))

    from aita_bench import scoring

    captured = {}

    def fake_outcome(*args, **kwargs):
        captured["blind_patterns"] = kwargs.get("blind_patterns")
        captured["ground_truth"] = kwargs.get("ground_truth")
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gpt-5-5": {"label": "GPT-5.5", "model_id": "openai/gpt-5.5"}},
        "judge/model",
    )

    assert "GPT-5.5" in captured["blind_patterns"]
    assert "openai/gpt-5.5" in captured["blind_patterns"]
    assert "gpt-5.5" in captured["blind_patterns"]
    assert captured["ground_truth"] == "NTA"


def test_score_all_prefers_transcript_metadata_for_blinding_and_scores(tmp_path, monkeypatch):
    side_a = {
        "model": "gemini-3-1-pro",
        "label": "Gemini 3.1 Pro",
        "model_id": "google/gemini-3.1-pro-preview",
        "turns": [
            {"turn": 1, "user_message": "u1", "model_response": "a1"},
            {"turn": 2, "user_message": "u2", "model_response": "a2"},
        ],
    }
    (tmp_path / "gemini-3-1-pro_item0_side_a.json").write_text(json.dumps(side_a))

    from aita_bench import scoring

    captured = {}

    def fake_outcome(*args, **kwargs):
        captured["blind_patterns"] = kwargs.get("blind_patterns")
        return 2

    monkeypatch.setattr(scoring, "score_outcome", fake_outcome)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)

    runner.score_all(
        tmp_path,
        [0],
        MagicMock(),
        {"gemini-3-1-pro": {"label": "gemini-3-1-pro", "model_id": "gemini-3-1-pro"}},
        "judge/model",
    )

    assert "Gemini 3.1 Pro" in captured["blind_patterns"]
    assert "google/gemini-3.1-pro-preview" in captured["blind_patterns"]

    saved = json.loads((tmp_path / "gemini-3-1-pro_item0_scores.json").read_text())
    assert saved["label"] == "Gemini 3.1 Pro"
    assert saved["model_id"] == "google/gemini-3.1-pro-preview"


def _write_single_judge_pair_artifacts(tmp_path, model_key, *, ground_truth_a="NTA", ground_truth_b="YTA"):
    side_a = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": f"side_a={ground_truth_a};side_b={ground_truth_b}",
        "ground_truth": ground_truth_a,
        "turns": [
            {"turn": 1, "user_message": "AITA side A?", "model_response": "NTA here."},
            {"turn": 2, "user_message": "Are you sure?", "model_response": "Still NTA."},
        ],
    }
    side_b = {
        "dataset_mode": "nta-paired",
        "pair_id": "pair-1",
        "paired_ground_truth": f"side_a={ground_truth_a};side_b={ground_truth_b}",
        "ground_truth": ground_truth_b,
        "turns": [
            {"turn": 1, "user_message": "AITA side B?", "model_response": "YTA, gently."},
            {"turn": 2, "user_message": "Are you sure?", "model_response": "Still YTA."},
        ],
    }
    (tmp_path / f"{model_key}_item0_side_a.json").write_text(json.dumps(side_a))
    (tmp_path / f"{model_key}_item0_side_b.json").write_text(json.dumps(side_b))


def _patch_complete_scalar_scores(monkeypatch):
    from aita_bench import scoring

    monkeypatch.setattr(scoring, "score_outcome", lambda *args, **kwargs: 2)
    monkeypatch.setattr(scoring, "score_persistence", lambda *args, **kwargs: 1)
    monkeypatch.setattr(scoring, "score_therapeutic", lambda *args, **kwargs: 3)
    monkeypatch.setattr(scoring, "score_consistency", lambda *args, **kwargs: 1)


def test_single_judge_scores_emit_majority_keys(tmp_path, monkeypatch):
    """Single-judge scoring must write `*_majority` keys so report rendering
    (which reads the majority keys for primary binary fields) stays populated."""
    _write_single_judge_pair_artifacts(tmp_path, "modelx")

    _patch_complete_scalar_scores(monkeypatch)

    _, _, scores = runner._score_one_item_single(
        tmp_path, 0, "modelx", {"label": "Model X"}, MagicMock(), "judge/model",
    )

    for field in runner.PANEL_BINARY_PRIMARY_FIELDS:
        assert f"{field}_majority" in scores
        assert scores[f"{field}_majority"] == scores.get(field)
    assert scores["verdict_alignment_a_majority"] == 1
    assert scores["paired_verdict_alignment_majority"] == 1

    saved = json.loads((tmp_path / "modelx_item0_scores.json").read_text())
    for field in runner.PANEL_BINARY_PRIMARY_FIELDS:
        assert saved[f"{field}_majority"] == saved.get(field)


def test_single_judge_scores_persist_sanitized_raw_judge_replies(tmp_path, monkeypatch):
    from aita_bench import scoring

    _write_single_judge_pair_artifacts(tmp_path, "modelx")
    client = MagicMock()
    client.base_url = "https://example.test/v1"

    def response(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    long_therapeutic = "Score: 3 " + ("x" * 4100)
    client.chat.completions.create.side_effect = [
        response('Score: 2 api_key: "fake-token"'),
        response("Score: 1"),
        response(long_therapeutic),
        response("Score: 2"),
        response("Score: 1"),
        response("Score: 3"),
        response("Score: 1"),
    ]

    @contextmanager
    def fake_lease(**kwargs):
        yield object()

    monkeypatch.setattr(scoring, "paid_call_lease", fake_lease)

    _, _, scores = runner._score_one_item_single(
        tmp_path, 0, "modelx", {"label": "Model X"}, client, "judge/model",
    )

    replies = scores["judge_raw_replies"]
    assert "<redacted>" in replies["outcome_a"]
    assert "fake-token" not in replies["outcome_a"]
    assert len(replies["therapeutic_a"]) == runner.JUDGE_RAW_REPLY_CHAR_LIMIT

    saved = json.loads((tmp_path / "modelx_item0_scores.json").read_text())
    assert saved["judge_raw_replies"] == replies


def test_paired_verdict_alignment_is_one_when_both_sides_align(tmp_path, monkeypatch):
    _write_single_judge_pair_artifacts(tmp_path, "modelx")
    _patch_complete_scalar_scores(monkeypatch)

    _, _, scores = runner._score_one_item_single(
        tmp_path, 0, "modelx", {"label": "Model X"}, MagicMock(), "judge/model",
    )

    assert scores["paired_verdict_alignment"] == 1
    assert "paired_verdict_alignment" not in scores["missing_scores"]


def test_paired_verdict_alignment_none_not_missing_for_one_sided_labels(tmp_path, monkeypatch):
    _write_single_judge_pair_artifacts(tmp_path, "modelx", ground_truth_b="NAH")
    _patch_complete_scalar_scores(monkeypatch)

    _, _, scores = runner._score_one_item_single(
        tmp_path, 0, "modelx", {"label": "Model X"}, MagicMock(), "judge/model",
    )

    assert scores["verdict_alignment_b"] is None
    assert scores["paired_verdict_alignment"] is None
    assert "paired_verdict_alignment" not in scores["missing_scores"]


def test_paired_verdict_alignment_none_missing_when_both_sides_labeled(tmp_path, monkeypatch):
    from aita_bench import scoring

    _write_single_judge_pair_artifacts(tmp_path, "modelx")
    _patch_complete_scalar_scores(monkeypatch)

    def fake_verdict_alignment(*args, **kwargs):
        ground_truth = kwargs.get("ground_truth")
        if ground_truth is None and len(args) >= 4:
            ground_truth = args[3]
        return {
            "verdict": "YTA" if ground_truth == "YTA" else "NTA",
            "alignment": None if ground_truth == "YTA" else 1,
            "evidence": "test fixture",
            "rationale": "test fixture",
            "confidence": "high",
        }

    monkeypatch.setattr(scoring, "score_verdict_alignment", fake_verdict_alignment)

    _, _, scores = runner._score_one_item_single(
        tmp_path, 0, "modelx", {"label": "Model X"}, MagicMock(), "judge/model",
    )

    assert scores["verdict_alignment_a"] == 1
    assert scores["verdict_alignment_b"] is None
    assert scores["paired_verdict_alignment"] is None
    assert "paired_verdict_alignment" in scores["missing_scores"]


def test_report_primary_columns_populated_for_single_judge_artifacts(tmp_path, monkeypatch):
    """End-to-end report over a real single-judge scores file (no hand-fed
    `*_majority` keys): primary verdict columns must not render as `---`."""
    _write_single_judge_pair_artifacts(tmp_path, "gemini-flash")

    _patch_complete_scalar_scores(monkeypatch)

    runner._score_one_item_single(
        tmp_path, 0, "gemini-flash", {"label": "Gemini 3 Flash"}, MagicMock(), "judge/model",
    )

    runner.generate_report(tmp_path, {"gemini-flash": {"label": "Gemini 3 Flash"}})

    report = (tmp_path / "REPORT.md").read_text()
    row = next(line for line in report.splitlines() if line.startswith("| Gemini 3 Flash |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    # Verdict A, Verdict B, Pair Verdict, Pair Verdict Rate are columns 1-4.
    verdict_a, verdict_b, pair_verdict, pair_rate = cells[1:5]
    assert verdict_a != "---"
    assert verdict_b != "---"
    assert pair_verdict != "---"
    assert pair_rate != "---"
    assert verdict_a == "1/1"
    assert pair_verdict == "1/1"


def test_report_includes_side_b_resistance_and_therapeutic_scores(tmp_path):
    (tmp_path / "gemini-flash_item0_scores.json").write_text(json.dumps({
        "verdict_alignment_a": 1,
        "verdict_alignment_b": 0,
        "paired_verdict_alignment": 0,
        "verdict_alignment_a_majority": 1,
        "verdict_alignment_b_majority": 0,
        "paired_verdict_alignment_majority": 0,
        "primary_failure": True,
        "moral_sycophancy_pattern": "both_nta",
        "outcome_a": 2,
        "outcome_b": 1,
        "resistance_a": 1,
        "resistance_b": 0,
        "therapeutic_a": 3,
        "therapeutic_b": 1,
        "consistency": 0,
    }))

    runner.generate_report(tmp_path, {"gemini-flash": {"label": "Gemini 3 Flash"}})

    report = (tmp_path / "REPORT.md").read_text()
    assert "Resistance B" in report
    assert "Therapeutic B" in report
    assert "Verdict A" in report
    assert "Both NTA" in report
    assert "Correct Pair" in report
    assert "Inverted" in report
    assert "| Gemini 3 Flash | 1/1 | 0/1 | 0/1 | 0.0% [0.0, 79.3] | 0 | 0 | 1 | 0 | 0 | 2.00 | 1.00 | 1.00 | 0.00 | 3.00 | 1.00 | 0.00 |" in report


def test_nta_paired_requires_explicit_label_answer_key(tmp_path):
    """No implicit YTA default: nta-paired must ship a label answer key."""
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({"id": ["a"], "original_post": ["o a"]}).to_csv(og_path, index=False)
    pd.DataFrame({"id": ["a"], "flipped_story": ["f a"]}).to_csv(flip_path, index=False)
    # No labels file present anywhere.
    with pytest.raises(ValueError, match="requires an explicit flip-label answer key"):
        runner.load_nta_paired_items(SimpleNamespace(
            og_data=str(og_path), flip_data=str(flip_path), items="1",
        ))


def test_nta_paired_esh_label_excludes_verdict_alignment(tmp_path):
    """An ESH-labelled flip carries side_b_ground_truth=ESH so verdict_alignment
    returns None (excluded), while YTA flips score normally."""
    og_path = tmp_path / "AITA-NTA-OG.csv"
    flip_path = tmp_path / "AITA-NTA-FLIP.csv"
    pd.DataFrame({"id": ["yta1", "esh1"], "original_post": ["o1", "o2"]}).to_csv(og_path, index=False)
    pd.DataFrame({"id": ["yta1", "esh1"], "flipped_story": ["f1", "f2"]}).to_csv(flip_path, index=False)
    (tmp_path / "AITA-NTA-FLIP.labels.json").write_text(
        '{"labels": {"yta1": "YTA", "esh1": "ESH"}}'
    )
    _, items, _ = runner.load_nta_paired_items(SimpleNamespace(
        og_data=str(og_path), flip_data=str(flip_path), items="2",
    ))
    by_id = {it["pair_id"]: it for it in items.values()}
    assert by_id["yta1"]["side_b_ground_truth"] == "YTA"
    assert by_id["esh1"]["side_b_ground_truth"] == "ESH"
    from aita_bench.verdicts import verdict_alignment
    assert verdict_alignment("ESH", "ESH") is None   # excluded from alignment
    assert verdict_alignment("YTA", "YTA") == 1


def test_api_call_does_not_retry_provider_refusal(tmp_path, monkeypatch):
    from suite_tools.provider_client import ProviderRefusalError

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    client = MagicMock()
    client.chat.completions.create.side_effect = ProviderRefusalError(
        "OpenAI native provider refusal; stop_reason=refusal"
    )
    sleep = MagicMock()
    monkeypatch.setattr(runner.time, "sleep", sleep)

    with pytest.raises(ProviderRefusalError, match="stop_reason=refusal"):
        runner.api_call(client, "openai/gpt-5.6-luna", [], retries=3)

    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_run_conversation_marks_provider_refusal_as_first_class_outcome(tmp_path, monkeypatch):
    from suite_tools.provider_client import ProviderRefusalError

    calls = {"n": 0}

    class RefusingCompletions:
        def create(self, *args, **kwargs):
            calls["n"] += 1
            raise ProviderRefusalError("OpenAI native provider refusal; stop_reason=refusal")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=RefusingCompletions()))
    monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    with pytest.raises(ProviderRefusalError, match="stop_reason=refusal"):
        runner.run_conversation(
            "gpt-56-luna",
            "Original story",
            0,
            "side_a",
            tmp_path,
            MagicMock(),
            {
                "gpt-56-luna": {
                    "label": "GPT-5.6 Luna",
                    "model_id": "gpt-5.6-luna",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "fake",
                    "max_parallel": 1,
                }
            },
            {
                "ground_truth": "NTA",
                "pair_id": "pair-1",
                "source_pair_hash": "pair-hash",
                "side_a_prompt_hash": "side-a-hash",
                "side_b_prompt_hash": "side-b-hash",
            },
            monitor,
        )

    assert calls["n"] == 1
    transcript = json.loads((tmp_path / "gpt-56-luna_item0_side_a.json").read_text())
    assert transcript["completed"] is False
    assert transcript["provider_refusal"] is True
    assert transcript["failure_stage"] == "model"
    assert transcript["failure_stage_detail"] == "provider_refusal"


def test_api_call_routes_gpt56_through_openai_responses(tmp_path, monkeypatch):
    """aita drives multi-turn dialogues; verify history maps to Responses input."""
    from contextlib import contextmanager

    captured = {}

    class Response:
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {
                "model": "gpt-5.6-sol-2026-07",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "NTA, clearly."}],
                    },
                ],
                "usage": {
                    "input_tokens": 40,
                    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                    "output_tokens": 12,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 52,
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
        "gpt-5.6-sol",
        [
            {"role": "system", "content": "You judge AITA posts."},
            {"role": "user", "content": "AITA for leaving early?"},
            {"role": "assistant", "content": "Tell me more."},
            {"role": "user", "content": "It was my sister's wedding."},
        ],
        request_options={"max_tokens": 64000, "reasoning_effort": "max"},
    )

    assert text == "NTA, clearly."
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["instructions"] == "You judge AITA posts."
    assert [item["role"] for item in captured["payload"]["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert captured["payload"]["input"][1]["content"][0]["type"] == "output_text"
    assert captured["payload"]["reasoning"] == {"effort": "max"}
    assert captured["payload"]["max_output_tokens"] == 64000


# --- Task 6: evidence-first api_call dispatch + live BLOCKS ledger ---------


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_unknown_error_halts_without_retry(tmp_path, monkeypatch):
    from aita_bench import runner as runner_module

    # Isolate the paid-call lease so the halt path never touches the repo dir.
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    calls = {"count": 0}

    class InscrutableClient:
        base_url = "https://api.example.test/v1"

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls["count"] += 1
                    raise RuntimeError("something inscrutable happened")

    monkeypatch.setattr(runner_module.time, "sleep", lambda *_: None)
    with pytest.raises(runner_module.FatalBenchmarkApiError):
        runner_module.api_call(
            InscrutableClient(), "test-model",
            [{"role": "user", "content": "hi"}], retries=3,
        )
    assert calls["count"] == 1, "unknown errors must halt on first occurrence"


def test_api_call_retries_connect_timeout_then_succeeds(tmp_path, monkeypatch):
    # plan 014 §6: connect-timeouts are retryable (nothing was billed).
    import httpx

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="model response"))]
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.side_effect = [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectTimeout("connect timed out"),
        response,
    ]

    assert runner.api_call(client, "test-model", [], retries=3) == "model response"
    assert client.chat.completions.create.call_count == 3


def test_api_call_read_timeout_is_terminal_owed(tmp_path, monkeypatch):
    # plan 014 §6: read-timeouts stay terminal (tokens may have been spent).
    import httpx

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    sleep = MagicMock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.side_effect = httpx.ReadTimeout("read timed out")

    with pytest.raises(runner.FatalBenchmarkApiError):
        runner.api_call(client, "test-model", [], retries=3)
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_api_call_rate_limited_retries_then_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(runner.time, "sleep", lambda *_: None)

    class RateLimited(Exception):
        status_code = 429

        def __str__(self):
            return "429 Too Many Requests"

    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.side_effect = RateLimited()

    with pytest.raises(RuntimeError) as exc:
        runner.api_call(client, "test-model", [], retries=3)
    assert not isinstance(exc.value, runner.FatalBenchmarkApiError)
    assert client.chat.completions.create.call_count == 3


def _refusal_run_conversation(tmp_path, monkeypatch, refusal_error, model_key):
    """Drive the runner's terminal-refusal path with the existing single-item
    run_conversation harness (the one the provider-refusal tests use)."""
    class RefusingCompletions:
        def create(self, *args, **kwargs):
            raise refusal_error

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=RefusingCompletions()))
    monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    with pytest.raises(ProviderRefusalError):
        runner.run_conversation(
            model_key,
            "Original story",
            0,
            "side_a",
            tmp_path,
            MagicMock(),
            {
                model_key: {
                    "label": "GPT-5.6 Luna",
                    "model_id": "gpt-5.6-luna",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "fake",
                    "max_parallel": 1,
                }
            },
            {
                "ground_truth": "NTA",
                "pair_id": "pair-1",
                "source_pair_hash": "pair-hash",
                "side_a_prompt_hash": "side-a-hash",
                "side_b_prompt_hash": "side-b-hash",
            },
            monitor,
        )
    return monitor


def test_terminal_refusal_writes_exactly_one_block(tmp_path, monkeypatch):
    from suite_tools.provider_client import ProviderRefusalError as _Refusal

    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    model_key = "gpt-56-luna"
    refusal = _Refusal(
        "OpenAI native provider refusal; stop_reason=refusal",
        raw_response={"error": {"code": "cyber_policy"}},
    )

    _refusal_run_conversation(tmp_path, monkeypatch, refusal, model_key)

    expected_transcript_name = f"{model_key}_item0_side_a.json"
    blocks = _read_jsonl(tmp_path / "BLOCKS.jsonl")
    assert len(blocks) == 1
    assert blocks[0]["category"] == "cyber_policy"
    assert blocks[0]["evidence_class"] == "model_signal"
    assert blocks[0]["evidence_pointer"] == expected_transcript_name
    assert blocks[0]["attempt_number"] == 1
    assert blocks[0]["unit"] == {"item_idx": 0, "side": "side_a"}

    conv = json.loads((tmp_path / expected_transcript_name).read_text())
    assert conv["provider_refusal"] is True
    assert conv["attempt_number"] == 1


def test_budget_exhaustion_writes_block_after_bounded_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    model_key = "gpt-5-6"

    class ExhaustingCompletions:
        def create(self, *args, **kwargs):
            raise ProviderOutputBudgetExhaustedError(
                "OpenAI Responses output budget exhausted; "
                "incomplete_reason=max_output_tokens",
                usage={"prompt_tokens": 16, "completion_tokens": 128000},
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=ExhaustingCompletions()))
    monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")

    conv = runner.run_conversation(
        model_key,
        "Original story",
        0,
        "side_a",
        tmp_path,
        MagicMock(),
        {
            model_key: {
                "label": "GPT-5.6",
                "model_id": "gpt-5.6-luna",
                "base_url": "https://api.openai.com/v1/responses",
                "api_key": "fake",
                "max_parallel": 1,
            }
        },
        {
            "ground_truth": "NTA",
            "pair_id": "pair-1",
            "source_pair_hash": "pair-hash",
            "side_a_prompt_hash": "side-a-hash",
            "side_b_prompt_hash": "side-b-hash",
        },
        monitor,
    )

    assert conv["output_budget_exhausted"] is True
    assert conv.get("provider_refusal") is not True
    expected_transcript_name = f"{model_key}_item0_side_a.json"
    blocks = _read_jsonl(tmp_path / "BLOCKS.jsonl")
    assert len(blocks) == 1
    assert blocks[0]["category"] == "output_budget_exhausted"
    assert blocks[0]["evidence_class"] == "model_signal"
    assert blocks[0]["evidence_pointer"] == expected_transcript_name
    assert conv["attempt_number"] == 1


def test_api_call_content_filter_finish_reason_bounded_retry_then_refusal(monkeypatch):
    """Plan 020 D8/T3: choices[0].finish_reason='content_filter' maps to top-level
    finish_reason in the raw dict → classify_payload Rule 5 → bounded_retry(max_retries=1).
    Executor allows exactly 1 retry (2 total paid calls) then terminates with ProviderRefusalError.
    The transient 'attempt' counter is NOT incremented on content-block retries (independent
    executor counter).  Old behavior (immediate terminal on first call) was pre-T3."""
    from aita_bench import runner
    from suite_tools.provider_client import ProviderRefusalError
    from types import SimpleNamespace
    import pytest
    calls = {"n": 0}

    class Msg:
        content = ""

    def fake_create(*a, **k):
        calls["n"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=Msg(), finish_reason="content_filter")], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(runner, "make_client", lambda cfg: client)
    with pytest.raises(ProviderRefusalError):
        runner.api_call(client, "test/model", [{"role": "user", "content": "hi"}], retries=3)
    # bounded_retry(1): 1 retry allowed → 2 paid calls total before terminal raise
    assert calls["n"] == 2


# ── Task 6: terminal reuse + unit_id on block sites ──────────────────────────

def test_aita_reuses_terminal_refusal_transcript(tmp_path, monkeypatch):
    events = []
    monitor = SimpleNamespace(record=lambda e, **f: events.append({"event": e, **f}), attempt_number=2)
    out = tmp_path / "m_item0_side_a.json"
    out.write_text(json.dumps({"item_idx": 0, "side": "side_a", "turns": [],
                               "completed": False, "provider_refusal": True}))
    models = {"m": {"label": "M", "model_id": "test/m"}}
    monkeypatch.setattr(runner, "make_client",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("must not re-execute a saved refusal")))
    result = runner.run_conversation("m", "post", 0, "side_a", tmp_path,
                                     SimpleNamespace(), models, monitor=monitor)
    assert result["provider_refusal"] is True
    assert any(e["event"] == "conversation_reused_provider_refusal" for e in events)


def test_aita_refusal_block_carries_unit_id(tmp_path, monkeypatch):
    # R3-3: AITA's refusal record_block site (runner.py ~L1424) must carry unit_id.
    blocks = []
    monitor = SimpleNamespace(record=lambda e, **f: None,
                              record_block=lambda **f: blocks.append(f),
                              record_usage=lambda *a, **k: None, attempt_number=1)
    models = {"m": {"label": "M", "model_id": "test/m"}}
    monkeypatch.setattr(runner, "make_client", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(runner, "get_seeker_msg", lambda *a, **k: "seeker continue")

    def fake_api_call(client, model_id, messages, **k):
        raise ProviderRefusalError("refusal", raw_response={"stop_reason": "refusal"})

    monkeypatch.setattr(runner, "api_call", fake_api_call)
    with pytest.raises(ProviderRefusalError):
        runner.run_conversation("m", "post", 0, "side_a", tmp_path,
                                SimpleNamespace(), models, monitor=monitor)
    assert len(blocks) == 1
    assert blocks[0]["unit_id"] == "aita:m:item0:side_a"


def test_aita_budget_block_carries_unit_id(tmp_path, monkeypatch):
    # R3-3: AITA's budget record_block site (_mark_output_budget_exhausted) must carry unit_id.
    blocks = []
    monitor = SimpleNamespace(record=lambda e, **f: None,
                              record_block=lambda **f: blocks.append(f),
                              record_usage=lambda *a, **k: None, attempt_number=1)
    models = {"m": {"label": "M", "model_id": "test/m"}}
    monkeypatch.setattr(runner, "make_client", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(runner, "get_seeker_msg", lambda *a, **k: "seeker continue")

    def fake_api_call(client, model_id, messages, **k):
        raise ProviderOutputBudgetExhaustedError("budget", usage={"total_tokens": 1})

    monkeypatch.setattr(runner, "api_call", fake_api_call)
    runner.run_conversation("m", "post", 0, "side_a", tmp_path,
                            SimpleNamespace(), models, monitor=monitor)
    assert len(blocks) == 1
    assert blocks[0]["unit_id"] == "aita:m:item0:side_a"


# ── unit_id on _record_event sites (budget-exhausted event + terminal reuse) ──


def test_aita_budget_exhausted_event_carries_unit_id(tmp_path, monkeypatch):
    """conversation_output_budget_exhausted _record_event must carry unit_id."""
    monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "0")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    model_key = "gpt-5-6"

    class ExhaustingCompletions:
        def create(self, *args, **kwargs):
            raise ProviderOutputBudgetExhaustedError(
                "budget exhausted", usage={"prompt_tokens": 1, "completion_tokens": 1}
            )

    monkeypatch.setattr(runner, "make_client",
                        lambda cfg: SimpleNamespace(
                            chat=SimpleNamespace(completions=ExhaustingCompletions()),
                            base_url="https://api.openai.com/v1/responses",
                        ))

    monitor = RunMonitor(tmp_path, module="aita", stage="generation")
    runner.run_conversation(
        model_key, "story", 0, "side_a", tmp_path, MagicMock(),
        {
            model_key: {
                "label": "GPT-5.6",
                "model_id": "gpt-5.6-luna",
                "base_url": "https://api.openai.com/v1/responses",
                "api_key": "fake",
                "max_parallel": 1,
            }
        },
        {
            "ground_truth": "NTA",
            "pair_id": "pair-1",
            "source_pair_hash": "s",
            "side_a_prompt_hash": "ha",
            "side_b_prompt_hash": "hb",
        },
        monitor,
    )

    events = _read_jsonl(tmp_path / "RUN_EVENTS.jsonl")
    budget_events = [e for e in events if e["event"] == "conversation_output_budget_exhausted"]
    assert budget_events, "no conversation_output_budget_exhausted event found"
    assert budget_events[0]["unit_id"] == f"aita:{model_key}:item0:side_a"


def test_aita_terminal_reuse_event_carries_unit_id(tmp_path, monkeypatch):
    """Terminal-reuse _record_event must carry unit_id."""
    events = []
    monitor = SimpleNamespace(
        record=lambda e, **f: events.append({"event": e, **f}),
        attempt_number=2,
    )
    out = tmp_path / "m_item0_side_a.json"
    out.write_text(json.dumps({
        "item_idx": 0, "side": "side_a", "turns": [],
        "completed": False, "output_budget_exhausted": True,
    }))
    models = {"m": {"label": "M", "model_id": "test/m"}}
    monkeypatch.setattr(
        runner, "make_client",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not re-execute a terminal")),
    )
    runner.run_conversation("m", "post", 0, "side_a", tmp_path,
                            SimpleNamespace(), models, monitor=monitor)
    reuse_events = [
        e for e in events
        if e["event"] in ("conversation_reused_output_budget_exhausted",
                          "conversation_reused_provider_refusal")
    ]
    assert reuse_events, "no terminal-reuse event found"
    assert reuse_events[0]["unit_id"] == "aita:m:item0:side_a"


def test_progress_dedupe_budget_exhaust_block_and_event_same_unit_counts_once():
    """block_recorded + conversation_output_budget_exhausted for the SAME unit_id counts as 1."""
    from suite_tools.progress_dedupe import completed_unit_keys
    uid = "aita:gpt-5-6:item0:side_a"
    events = [
        {"event": "block_recorded", "unit_id": uid},
        {"event": "conversation_output_budget_exhausted", "unit_id": uid},
    ]
    assert len(completed_unit_keys(events)) == 1


def test_aita_completed_reuse_event_carries_unit_id(tmp_path, monkeypatch):
    """Fix #4: conversation_reused (completed branch) must carry unit_id.
    Terminal reuse already carries unit_id; this pins the same requirement for
    the completed branch (existing transcript with enough turns)."""
    events = []
    monitor = SimpleNamespace(
        record=lambda e, **f: events.append({"event": e, **f}),
        attempt_number=2,
    )
    model_key = "m"
    item_idx = 3
    side = "side_b"
    out = tmp_path / f"{model_key}_item{item_idx}_{side}.json"
    # 5-turn completed transcript (NUM_TURNS=5 for aita)
    out.write_text(json.dumps({
        "item_idx": item_idx, "side": side, "model": model_key,
        "turns": [{"turn": i, "model_response": f"r{i}"} for i in range(1, 6)],
        "completed": True,
    }))
    models = {model_key: {"label": "M", "model_id": "test/m"}}
    monkeypatch.setattr(
        runner, "make_client",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must not call client for reused transcript")),
    )
    runner.run_conversation(model_key, "post", item_idx, side, tmp_path,
                            SimpleNamespace(), models, monitor=monitor)
    reuse_events = [e for e in events if e["event"] == "conversation_reused"]
    assert reuse_events, "no conversation_reused event found"
    expected_uid = f"aita:{model_key}:item{item_idx}:{side}"
    assert reuse_events[0].get("unit_id") == expected_uid, (
        f"conversation_reused event missing unit_id={expected_uid!r}, "
        f"got: {reuse_events[0]}"
    )


def test_aita_completed_reuse_restores_condition_identity_from_config(tmp_path, monkeypatch):
    events = []
    monitor = SimpleNamespace(
        record=lambda e, **f: events.append({"event": e, **f}),
        attempt_number=2,
    )
    model_key = "m"
    out = tmp_path / f"{model_key}_item0_side_a.json"
    out.write_text(json.dumps({
        "item_idx": 0,
        "side": "side_a",
        "model": model_key,
        "turns": [{"turn": i, "model_response": f"r{i}"} for i in range(1, 6)],
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
        "post",
        0,
        "side_a",
        tmp_path,
        SimpleNamespace(),
        models,
        monitor=monitor,
    )

    assert result["condition_id"] == "m-high"
    assert result["condition_hash"] == "sha256:m-high"
    restored = [e for e in events if e["event"] == "conversation_reuse_identity_restored"]
    assert restored and set(restored[0]["restored_fields"]) >= {
        "condition_id",
        "condition_hash",
    }
