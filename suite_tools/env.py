"""Shared environment loading for benchmark CLIs.

The nested benchmark modules can be launched from their own package folders,
from the suite root, or by the scheduler. Keep API-key discovery consistent
without requiring callers to copy secrets into every module directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from suite_tools.suite_registry import REPO_ROOT, suite_root

_DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROVIDER_KEY_ENVS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "FIREWORKS_API_KEY",
    "PERPLEXITY_API_KEY",
)
BENCHMARK_SERVICE_KEY_ENVS = (
    "LOCAL_OPENAI_COMPATIBLE_API_KEY",
    "PRIVATE_ADAPTER_API_KEY",
    "ADAPTER_INBOUND_API_KEY",
)
SAFE_DOTENV_OPERATIONAL_ENVS = (
    "BENCHMARK_PAID_CALL_MAX_ACTIVE",
    "BENCHMARK_RATE_LIMIT_COOLDOWN_SECONDS",
    "BENCHMARK_RATE_LIMIT_MAX_COOLDOWN_SECONDS",
)


def default_env_paths(cwd: Path | None = None) -> list[Path]:
    """Return repo-local .env locations in lookup order."""
    base = (cwd or Path.cwd()).resolve()
    repo_root = REPO_ROOT.resolve()
    paths = [
        REPO_ROOT / ".env",
        REPO_ROOT / "adapter" / ".env",
        suite_root("aita") / ".env",
        suite_root("epistemic") / ".env",
        suite_root("sus") / ".env",
    ]
    if base.is_relative_to(repo_root):
        paths.insert(0, base / ".env")
    return paths


def _dotenv_key_is_loadable(key: str) -> bool:
    """Allow credentials and documented pacing only, never behavior switches."""
    if key in SAFE_DOTENV_OPERATIONAL_ENVS:
        return True
    return key.endswith("_API_KEY") or key.endswith("_ACCESS_TOKEN")


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def read_repo_env_values(
    names: Iterable[str],
    paths: Iterable[Path] | None = None,
) -> dict[str, str]:
    """Read selected repo-local values without mutating the process environment."""
    requested = {str(name) for name in names}
    values = {
        name: value
        for name in requested
        if (value := os.environ.get(name))
    }
    seen: set[Path] = set()
    for candidate in paths or default_env_paths():
        path = Path(candidate).expanduser()
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if key not in requested or key in values or not _DOTENV_KEY_RE.match(key):
                continue
            value = _parse_env_value(raw_value)
            if value:
                values[key] = value
    return values


def load_repo_env_files(paths: Iterable[Path] | None = None) -> list[Path]:
    """Load simple KEY=VALUE pairs from repo-local .env files.

    Existing process environment values win. The returned list is only file
    paths loaded, never secret values.
    """
    loaded: list[Path] = []
    seen: set[Path] = set()
    for candidate in paths or default_env_paths():
        path = Path(candidate).expanduser()
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if (
                not _DOTENV_KEY_RE.match(key)
                or not _dotenv_key_is_loadable(key)
                or os.environ.get(key)
            ):
                continue
            value = _parse_env_value(raw_value)
            if not value:
                continue
            os.environ[key] = value
        loaded.append(path)
    return loaded
