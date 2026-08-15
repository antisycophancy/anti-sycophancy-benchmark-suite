import os
from pathlib import Path

import suite_tools.env as env_module
from suite_tools.env import (
    BENCHMARK_SERVICE_KEY_ENVS,
    PROVIDER_KEY_ENVS,
    load_repo_env_files,
    default_env_paths,
    read_repo_env_values,
)


def test_load_repo_env_files_reads_simple_values_without_overriding(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "OPENROUTER_API_KEY=from-file",
            "export PRIVATE_ADAPTER_API_KEY='adapter-key'",
            "INVALID-KEY=ignored",
            "EMPTY_PLACEHOLDER=",
            "EXISTING=value-from-file",
            "BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS=1",
        ])
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PRIVATE_ADAPTER_API_KEY", raising=False)
    monkeypatch.delenv("EMPTY_PLACEHOLDER", raising=False)
    monkeypatch.setenv("EXISTING", "already-set")
    monkeypatch.delenv("BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS", raising=False)

    loaded = load_repo_env_files([env_path])

    assert loaded == [env_path]
    assert os.environ["OPENROUTER_API_KEY"] == "from-file"
    assert os.environ["PRIVATE_ADAPTER_API_KEY"] == "adapter-key"
    assert os.environ["EXISTING"] == "already-set"
    assert "EMPTY_PLACEHOLDER" not in os.environ
    assert "INVALID-KEY" not in os.environ
    assert "BENCHMARK_ALLOW_ARBITRARY_CONTRACT_COMMANDS" not in os.environ


def test_default_env_paths_ignore_outside_cwd_and_repo_parent(tmp_path, monkeypatch):
    repo = tmp_path / "release" / "benchmark"
    outside = tmp_path / "untrusted"
    repo.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setattr(env_module, "REPO_ROOT", repo)
    monkeypatch.setattr(env_module, "suite_root", lambda module: repo / module)

    paths = default_env_paths(cwd=outside)

    assert outside / ".env" not in paths
    assert repo.parent / ".env" not in paths
    assert repo / ".env" in paths


def test_load_repo_env_files_does_not_import_unlisted_behavior_flags(
    tmp_path,
    monkeypatch,
):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENROUTER_API_KEY=provider-key\n"
        "BENCHMARK_PAID_CALL_MAX_ACTIVE=3\n"
        "BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS=1\n"
        "PYTHONPATH=/tmp/attacker\n"
    )
    for name in (
        "OPENROUTER_API_KEY",
        "BENCHMARK_PAID_CALL_MAX_ACTIVE",
        "BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS",
        "PYTHONPATH",
    ):
        monkeypatch.delenv(name, raising=False)

    load_repo_env_files([env_path])

    assert os.environ["OPENROUTER_API_KEY"] == "provider-key"
    assert os.environ["BENCHMARK_PAID_CALL_MAX_ACTIVE"] == "3"
    assert "BENCHMARK_ALLOW_LEGACY_SHELL_CONTRACTS" not in os.environ
    assert "PYTHONPATH" not in os.environ


def test_read_repo_env_values_is_read_only_and_preserves_environment_precedence(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BENCHMARK_PAID_CALL_MAX_ACTIVE=3\n"
        "BENCHMARK_MAX_ACTIVE_CALLS=64\n"
        "OPENROUTER_API_KEY=do-not-return\n"
    )
    monkeypatch.setenv("BENCHMARK_MAX_ACTIVE_CALLS", "16")
    monkeypatch.delenv("BENCHMARK_PAID_CALL_MAX_ACTIVE", raising=False)

    values = read_repo_env_values(
        ("BENCHMARK_PAID_CALL_MAX_ACTIVE", "BENCHMARK_MAX_ACTIVE_CALLS"),
        [env_path],
    )

    assert values == {
        "BENCHMARK_PAID_CALL_MAX_ACTIVE": "3",
        "BENCHMARK_MAX_ACTIVE_CALLS": "16",
    }
    assert "BENCHMARK_PAID_CALL_MAX_ACTIVE" not in os.environ
    assert "OPENROUTER_API_KEY" not in values


def test_blank_module_env_does_not_shadow_root_provider_key(tmp_path, monkeypatch):
    module_env = tmp_path / "module.env"
    root_env = tmp_path / "root.env"
    module_env.write_text("OPENROUTER_API_KEY=\n")
    root_env.write_text("OPENROUTER_API_KEY=from-root\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_repo_env_files([module_env, root_env])

    assert loaded == [module_env, root_env]
    assert os.environ["OPENROUTER_API_KEY"] == "from-root"


def test_blank_existing_env_does_not_shadow_repo_provider_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    loaded = load_repo_env_files([env_path])

    assert loaded == [env_path]
    assert os.environ["OPENROUTER_API_KEY"] == "from-file"


def test_known_provider_env_names_include_supported_and_future_direct_keys():
    assert "OPENROUTER_API_KEY" in PROVIDER_KEY_ENVS
    assert "OPENAI_API_KEY" in PROVIDER_KEY_ENVS
    assert "ANTHROPIC_API_KEY" in PROVIDER_KEY_ENVS
    assert "GOOGLE_API_KEY" in PROVIDER_KEY_ENVS
    assert "GEMINI_API_KEY" in PROVIDER_KEY_ENVS
    assert "LOCAL_OPENAI_COMPATIBLE_API_KEY" in BENCHMARK_SERVICE_KEY_ENVS
    assert "PRIVATE_ADAPTER_API_KEY" in BENCHMARK_SERVICE_KEY_ENVS


def test_sus_cli_uses_only_the_allowlisted_repo_env_loader():
    root = Path(__file__).resolve().parents[1]
    source = (root / "sus-bench" / "sus_bench" / "cli.py").read_text()

    assert "load_dotenv" not in source
    assert source.count("load_repo_env_files()") >= 2
