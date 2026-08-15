"""Render per-module model configs from the central suite config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_CONFIG = REPO_ROOT / "suite_models.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated_configs"
MODULES = ("sus", "aita", "epis")
MODEL_CONDITION_METADATA_FIELDS = (
    "provider_api",
    "route_hash",
    "condition_id",
    "condition_hash",
    "profile_id",
    "profile_hash",
    "parent_profile_id",
    "served_profile_id",
    "served_profile_hash",
    "system_fingerprint",
    "provider_condition_id",
    "provider_condition_hash",
    "provider_version",
    "condition_metadata",
    "request_options",
)


def route_identity_hash(provider_api: str | None, base_url: str | None) -> str | None:
    """Return an opaque hash of the normalized provider route.

    OpenAI-compatible base URLs and their explicit chat-completions URL denote
    the same route, so the standard suffix is removed before hashing.
    """
    if not base_url:
        return None
    parsed = urlsplit(str(base_url))
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError(f"Invalid provider route URL: {base_url}")
    port = parsed.port or (443 if scheme == "https" else 80 if scheme == "http" else None)
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    if str(provider_api or "openai_compatible") == "openai_compatible":
        suffix = "/chat/completions"
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
    payload = {
        "schema_version": "benchmark-provider-route-v1",
        "provider_api": str(provider_api or "openai_compatible"),
        "scheme": scheme,
        "host": hostname,
        "port": port,
        "path": path,
        "query": parsed.query,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_model_condition_identity(
    model: dict[str, Any],
    *,
    key: str,
    endpoint_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return a model condition with a deterministic identity if none is declared."""
    resolved = dict(model)
    resolved.setdefault("condition_id", key)
    route_hash = route_identity_hash(
        resolved.get("provider_api", "openai_compatible"),
        resolved.get("base_url"),
    )
    if route_hash is not None:
        resolved["route_hash"] = route_hash
    if resolved.get("condition_hash") and not force:
        return resolved
    payload = {
        "schema_version": "benchmark-model-condition-v1",
        "model_id": resolved.get("model_id") or resolved.get("id"),
        "endpoint": endpoint_name,
        "route_hash": route_hash,
        **{
            field: resolved[field]
            for field in MODEL_CONDITION_METADATA_FIELDS
            if field not in {"condition_id", "condition_hash"}
            and field in resolved
            and resolved[field] is not None
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    resolved["condition_hash"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return resolved


def load_suite_config(path: str | Path = DEFAULT_SUITE_CONFIG) -> dict[str, Any]:
    target = Path(path)
    return _load_suite_config(target, seen=set())


def _load_suite_config(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    target = path.resolve()
    if target in seen:
        raise ValueError(f"Suite config extends cycle detected at {path}")
    seen.add(target)

    with open(target) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Suite config must be a mapping: {path}")

    extends = config.pop("extends", None)
    if not extends:
        seen.remove(target)
        return config

    base_config: dict[str, Any] = {}
    extend_paths = extends if isinstance(extends, list) else [extends]
    for extend_path in extend_paths:
        if not isinstance(extend_path, str) or not extend_path.strip():
            raise ValueError(f"`extends` entries must be non-empty strings: {path}")
        base_path = Path(extend_path)
        if not base_path.is_absolute():
            base_path = target.parent / base_path
        base_config = _deep_merge(base_config, _load_suite_config(base_path, seen=seen))

    seen.remove(target)
    return _deep_merge(base_config, config)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a mapping")
    return value


def _require_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{label}` must be a non-empty string")


def _validate_agents_mapping(
    agents: dict[str, Any],
    label: str,
    *,
    required_roles: set[str] | None = None,
) -> None:
    if not isinstance(agents, dict):
        raise ValueError(f"`{label}` must be a mapping")
    for agent_name, agent in agents.items():
        if not isinstance(agent, dict):
            raise ValueError(f"`{label}.{agent_name}` must be a mapping")
        _require_non_empty_string(agent.get("model_id"), f"{label}.{agent_name}.model_id")
        role = agent.get("role")
        if role is not None:
            _require_non_empty_string(role, f"{label}.{agent_name}.role")
    if required_roles:
        missing = sorted(required_roles.difference(agents))
        if missing:
            raise ValueError(f"`{label}` is missing required agent roles: {', '.join(missing)}")


def _profile_agents(profile: Any, label: str) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError(f"`{label}` must be a mapping")
    description = profile.get("description")
    if description is not None:
        _require_non_empty_string(description, f"{label}.description")
    agents = profile.get("agents")
    if not isinstance(agents, dict):
        raise ValueError(f"`{label}.agents` must be a mapping")
    return agents


def resolve_agents(
    config: dict[str, Any],
    agent_profile: str | None = None,
    *,
    module: str | None = None,
) -> dict[str, Any]:
    """Return the effective agent map after applying an optional profile."""
    agents = {name: dict(agent) for name, agent in (config.get("agents") or {}).items()}
    module_agents = config.get("module_agents") or {}
    if module and module in module_agents:
        agents = _deep_merge(agents, module_agents[module])
    if not agent_profile:
        return agents

    profiles = config.get("agent_profiles") or {}
    if agent_profile not in profiles:
        raise KeyError(f"Unknown agent profile: {agent_profile}")
    profile_agents = _profile_agents(profiles[agent_profile], f"agent_profiles.{agent_profile}")
    return _deep_merge(agents, profile_agents)


def validate_suite_config(config: dict[str, Any]) -> list[str]:
    """Validate the central suite model config and return non-fatal warnings."""
    warnings: list[str] = []
    schema_version = config.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("`schema_version` must be an integer")

    defaults = _require_mapping(config, "defaults")
    endpoints = _require_mapping(config, "endpoints")
    models = _require_mapping(config, "models")
    model_groups = _require_mapping(config, "model_groups")
    judge_sets = _require_mapping(config, "judge_sets")
    judge_models = config.get("judge_models") or {}
    if not isinstance(judge_models, dict):
        raise ValueError("`judge_models` must be a mapping when present")
    agents = _require_mapping(config, "agents")
    _validate_agents_mapping(agents, "agents")
    module_agents = config.get("module_agents") or {}
    if not isinstance(module_agents, dict):
        raise ValueError("`module_agents` must be a mapping when present")
    agent_profiles = config.get("agent_profiles") or {}
    if not isinstance(agent_profiles, dict):
        raise ValueError("`agent_profiles` must be a mapping when present")
    agent_roles = set(agents)
    for module_name, overrides in module_agents.items():
        _require_non_empty_string(module_name, "module_agents module name")
        if module_name not in MODULES:
            raise ValueError(f"`module_agents` references unknown module: {module_name}")
        if not isinstance(overrides, dict):
            raise ValueError(f"`module_agents.{module_name}` must be a mapping")
        unknown_roles = sorted(set(overrides).difference(agent_roles))
        if unknown_roles:
            raise ValueError(
                f"`module_agents.{module_name}` references unknown roles: {', '.join(unknown_roles)}"
            )
        resolved_module_agents = _deep_merge(agents, overrides)
        _validate_agents_mapping(
            resolved_module_agents,
            f"module_agents.{module_name}",
            required_roles=agent_roles,
        )
    for profile_name, profile in agent_profiles.items():
        _require_non_empty_string(profile_name, "agent_profiles profile name")
        profile_agents = _profile_agents(profile, f"agent_profiles.{profile_name}")
        unknown_roles = sorted(set(profile_agents).difference(agent_roles))
        if unknown_roles:
            raise ValueError(
                f"`agent_profiles.{profile_name}.agents` references unknown roles: {', '.join(unknown_roles)}"
            )
        resolved_profile_agents = _deep_merge(agents, profile_agents)
        _validate_agents_mapping(
            resolved_profile_agents,
            f"agent_profiles.{profile_name}.agents",
            required_roles=agent_roles,
        )

    default_endpoint = defaults.get("endpoint")
    _require_non_empty_string(default_endpoint, "defaults.endpoint")
    if default_endpoint not in endpoints:
        raise ValueError(f"`defaults.endpoint` references unknown endpoint: {default_endpoint}")
    max_parallel = defaults.get("max_parallel")
    if not isinstance(max_parallel, int) or max_parallel < 1:
        raise ValueError("`defaults.max_parallel` must be a positive integer")

    for endpoint_name, endpoint in endpoints.items():
        if not isinstance(endpoint, dict):
            raise ValueError(f"`endpoints.{endpoint_name}` must be a mapping")
        provider_api = endpoint.get("provider_api", "openai_compatible")
        _require_non_empty_string(provider_api, f"endpoints.{endpoint_name}.provider_api")
        _require_non_empty_string(endpoint.get("api_key_env"), f"endpoints.{endpoint_name}.api_key_env")
        if provider_api == "openai_compatible":
            for field in ("openai_base_url", "chat_completions_url"):
                _require_non_empty_string(endpoint.get(field), f"endpoints.{endpoint_name}.{field}")
        elif provider_api == "anthropic_messages":
            _require_non_empty_string(endpoint.get("messages_url"), f"endpoints.{endpoint_name}.messages_url")
        elif provider_api == "gemini_generate_content":
            _require_non_empty_string(
                endpoint.get("generate_content_base_url"),
                f"endpoints.{endpoint_name}.generate_content_base_url",
            )
        elif provider_api == "openai_responses":
            _require_non_empty_string(
                endpoint.get("responses_url"),
                f"endpoints.{endpoint_name}.responses_url",
            )
        else:
            raise ValueError(f"`endpoints.{endpoint_name}.provider_api` is unsupported: {provider_api}")

    for judge_name, judge in judge_sets.items():
        if not isinstance(judge, dict):
            raise ValueError(f"`judge_sets.{judge_name}` must be a mapping")
        _require_non_empty_string(judge.get("primary"), f"judge_sets.{judge_name}.primary")
        panel = judge.get("panel", [judge["primary"]])
        if not isinstance(panel, list) or not panel:
            raise ValueError(f"`judge_sets.{judge_name}.panel` must be a non-empty list")
        for index, model_id in enumerate(panel):
            _require_non_empty_string(model_id, f"judge_sets.{judge_name}.panel[{index}]")

    for judge_key, judge_model in judge_models.items():
        if not isinstance(judge_model, dict):
            raise ValueError(f"`judge_models.{judge_key}` must be a mapping")
        _require_non_empty_string(judge_model.get("model_id"), f"judge_models.{judge_key}.model_id")
        endpoint_name = judge_model.get("endpoint", default_endpoint)
        _require_non_empty_string(endpoint_name, f"judge_models.{judge_key}.endpoint")
        if endpoint_name not in endpoints:
            raise ValueError(f"`judge_models.{judge_key}.endpoint` references unknown endpoint: {endpoint_name}")
        request_options = judge_model.get("request_options")
        if request_options is not None and not isinstance(request_options, dict):
            raise ValueError(f"`judge_models.{judge_key}.request_options` must be a mapping")

    seen_model_ids: dict[str, tuple[str, str | None]] = {}
    for model_key, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"`models.{model_key}` must be a mapping")
        _require_non_empty_string(model.get("model_id"), f"models.{model_key}.model_id")
        endpoint_name = model.get("endpoint", default_endpoint)
        _require_non_empty_string(endpoint_name, f"models.{model_key}.endpoint")
        if endpoint_name not in endpoints:
            raise ValueError(f"`models.{model_key}.endpoint` references unknown endpoint: {endpoint_name}")
        model_parallel = model.get("max_parallel", max_parallel)
        if not isinstance(model_parallel, int) or model_parallel < 1:
            raise ValueError(f"`models.{model_key}.max_parallel` must be a positive integer")
        request_options = model.get("request_options")
        if request_options is not None and not isinstance(request_options, dict):
            raise ValueError(f"`models.{model_key}.request_options` must be a mapping")
        current_condition = model.get("condition_id")
        previous_key, previous_condition = seen_model_ids.setdefault(
            model["model_id"],
            (model_key, current_condition if isinstance(current_condition, str) else None),
        )
        if previous_key != model_key and (
            not previous_condition
            or not isinstance(current_condition, str)
            or previous_condition == current_condition
        ):
            warnings.append(
                f"model_id `{model['model_id']}` is used by both `{previous_key}` and `{model_key}`"
            )

    for group_name, keys in model_groups.items():
        if not isinstance(keys, list) or not keys:
            raise ValueError(f"`model_groups.{group_name}` must be a non-empty list")
        for index, model_key in enumerate(keys):
            _require_non_empty_string(model_key, f"model_groups.{group_name}[{index}]")
            if model_key not in models:
                raise ValueError(f"`model_groups.{group_name}` references unknown model: {model_key}")

    return warnings


def expand_model_keys(config: dict[str, Any], selector: str = "all") -> list[str]:
    models = config.get("models", {})
    groups = config.get("model_groups", {})

    if selector == "all":
        return list(models.keys())
    if selector.startswith("group:"):
        group_name = selector.split(":", 1)[1]
        if group_name not in groups:
            raise KeyError(f"Unknown model group: {group_name}")
        return list(groups[group_name])
    return [key.strip() for key in selector.split(",") if key.strip()]


def _endpoint_for_model(config: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    endpoint_name = model.get("endpoint", config.get("defaults", {}).get("endpoint", "openrouter"))
    endpoints = config.get("endpoints", {})
    if endpoint_name not in endpoints:
        raise KeyError(f"Unknown endpoint: {endpoint_name}")
    endpoint = dict(endpoints[endpoint_name])
    endpoint["name"] = endpoint_name
    return endpoint


def _model_entry_from_table(
    config: dict[str, Any],
    key: str,
    module: str,
    *,
    table_name: str,
) -> dict[str, Any]:
    table = config.get(table_name, {})
    if key not in table:
        raise KeyError(f"Unknown {table_name} key: {key}")

    model = dict(table[key])
    endpoint = _endpoint_for_model(config, model)
    default_parallel = config.get("defaults", {}).get("max_parallel", 3)
    provider_api = endpoint.get("provider_api", "openai_compatible")
    if provider_api == "anthropic_messages":
        base_url = endpoint["messages_url"]
    elif provider_api == "gemini_generate_content":
        base_url = endpoint["generate_content_base_url"]
    elif provider_api == "openai_responses":
        # The Responses API uses one endpoint for every module (no separate
        # sus chat-completions URL), so route all modules to responses_url.
        base_url = endpoint["responses_url"]
    else:
        base_url = endpoint["chat_completions_url"] if module == "sus" else endpoint["openai_base_url"]

    entry = {
        "model_id": model["model_id"],
        "label": model.get("label", key),
        "base_url": base_url,
        "provider_api": provider_api,
        "api_key_env": model.get("api_key_env", endpoint.get("api_key_env", "OPENROUTER_API_KEY")),
        "max_parallel": model.get("max_parallel", default_parallel),
    }
    for field in MODEL_CONDITION_METADATA_FIELDS:
        if field in model:
            entry[field] = model[field]
    return ensure_model_condition_identity(
        entry,
        key=key,
        endpoint_name=endpoint["name"],
        force=True,
    )


def _module_model_entry(config: dict[str, Any], key: str, module: str) -> dict[str, Any]:
    return _model_entry_from_table(config, key, module, table_name="models")


def render_model_condition(config: dict[str, Any], key: str, module: str) -> dict[str, Any]:
    """Render one evaluated-model condition with resolved route and identity."""
    return _module_model_entry(config, key, module)


def _judge_set(config: dict[str, Any], name: str) -> dict[str, Any]:
    judge_sets = config.get("judge_sets", {})
    if name not in judge_sets:
        raise KeyError(f"Unknown judge set: {name}")
    return judge_sets[name]


def _legacy_openrouter_judge_config(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    endpoint = dict((config.get("endpoints") or {}).get("openrouter") or {})
    base_url = endpoint.get("openai_base_url", "https://openrouter.ai/api/v1")
    provider_api = endpoint.get("provider_api", "openai_compatible")
    return {
        "model_id": model_id,
        "label": model_id,
        "base_url": base_url,
        "provider_api": provider_api,
        "route_hash": route_identity_hash(provider_api, base_url),
        "api_key_env": endpoint.get("api_key_env", "OPENROUTER_API_KEY"),
        "condition_metadata": {
            "provider_route": "openrouter",
            "role": "judge",
            "effort": "legacy_default",
            "effort_policy": "module_default_judge_reasoning",
        },
    }


def _judge_config_entry(config: dict[str, Any], judge_key_or_model_id: str) -> dict[str, Any]:
    judge_models = config.get("judge_models") or {}
    if judge_key_or_model_id in judge_models:
        entry = _model_entry_from_table(
            config,
            judge_key_or_model_id,
            "judge",
            table_name="judge_models",
        )
        entry["key"] = judge_key_or_model_id
        return entry
    return _legacy_openrouter_judge_config(config, judge_key_or_model_id)


def render_module_config(
    config: dict[str, Any],
    module: str,
    *,
    judge_set: str = "calibration",
    model_selector: str = "all",
    agent_profile: str | None = None,
) -> dict[str, Any]:
    """Render one module's native models.yaml shape."""
    if module not in MODULES:
        raise ValueError(f"Unknown module: {module}")

    judges = _judge_set(config, judge_set)
    agents = resolve_agents(config, agent_profile, module=module)
    agent_profile_label = agent_profile or "default"
    keys = expand_model_keys(config, model_selector)
    judge_panel_keys = list(judges.get("panel", [judges["primary"]]))
    judge_configs = [_judge_config_entry(config, judge_key) for judge_key in judge_panel_keys]
    judge_panel = [judge_config["model_id"] for judge_config in judge_configs]
    primary_judge_config = _judge_config_entry(config, judges["primary"])

    if module == "sus":
        return {
            "agent_profile": agent_profile_label,
            "analyzer": agents["analyzer"]["model_id"],
            "judge_panel": judge_panel,
            "judge_set": judge_set,
            "judge_configs": judge_configs,
            "models": [
                {"id": entry.pop("model_id"), "key": key, **entry}
                for key, entry in ((k, _module_model_entry(config, k, module)) for k in keys)
            ],
        }

    rendered_models = {
        key: _module_model_entry(config, key, module)
        for key in keys
    }
    result: dict[str, Any] = {
        "agent_profile": agent_profile_label,
        "judge": {
            "model_id": primary_judge_config["model_id"],
            "panel": judge_panel,
            "provider": primary_judge_config.get("condition_metadata", {}).get("provider_route", "openrouter"),
            "judge_set": judge_set,
            "primary_config": primary_judge_config,
            "configs": judge_configs,
        },
        "seeker": {
            "model_id": agents["seeker"]["model_id"],
            "provider": "openrouter",
            "role": agents["seeker"].get("role"),
        },
        "models": rendered_models,
    }
    if module == "aita":
        result["flip_generator"] = {
            "model_id": agents["flip_generator"]["model_id"],
            "provider": "openrouter",
            "role": agents["flip_generator"].get("role"),
        }
    return result


def write_rendered_configs(
    *,
    suite_config_path: str | Path = DEFAULT_SUITE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    judge_set: str = "calibration",
    model_selector: str = "all",
    agent_profile: str | None = None,
    modules: tuple[str, ...] = MODULES,
) -> list[Path]:
    config = load_suite_config(suite_config_path)
    out = Path(output_dir) / judge_set
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for module in modules:
        rendered = render_module_config(
            config,
            module,
            judge_set=judge_set,
            model_selector=model_selector,
            agent_profile=agent_profile,
        )
        path = out / f"{module}-models.yaml"
        path.write_text(yaml.safe_dump(rendered, sort_keys=False))
        written.append(path)
    return written


def describe_config(config: dict[str, Any]) -> str:
    """Return a compact human-readable inventory for model config selection."""
    lines = ["Judge sets:"]
    for name, judge in sorted((config.get("judge_sets") or {}).items()):
        panel = ", ".join(judge.get("panel", [judge.get("primary", "")]))
        lines.append(f"  {name}: primary={judge.get('primary')} panel=[{panel}]")
    judge_models = config.get("judge_models") or {}
    if judge_models:
        lines.append("Judge models:")
        for key, judge_model in sorted(judge_models.items()):
            endpoint = judge_model.get("endpoint", config.get("defaults", {}).get("endpoint", "openrouter"))
            lines.append(f"  {key}: {judge_model.get('model_id')} ({endpoint})")
    lines.append("Agent profiles:")
    lines.append("  default: " + ", ".join(
        f"{name}={agent.get('model_id')}"
        for name, agent in sorted((config.get("agents") or {}).items())
        if isinstance(agent, dict)
    ))
    for name, profile in sorted((config.get("agent_profiles") or {}).items()):
        if not isinstance(profile, dict):
            continue
        agents = profile.get("agents") or {}
        lines.append("  " + name + ": " + ", ".join(
            f"{agent_name}={agent.get('model_id')}"
            for agent_name, agent in sorted(agents.items())
            if isinstance(agent, dict)
        ))
    module_agents = config.get("module_agents") or {}
    if module_agents:
        lines.append("Module agent overrides:")
        for module_name, agents in sorted(module_agents.items()):
            if not isinstance(agents, dict):
                continue
            lines.append("  " + module_name + ": " + ", ".join(
                f"{agent_name}={agent.get('model_id')}"
                for agent_name, agent in sorted(agents.items())
                if isinstance(agent, dict)
            ))
    lines.append("Model groups:")
    for name, keys in sorted((config.get("model_groups") or {}).items()):
        lines.append(f"  {name}: {', '.join(keys)}")
    lines.append("Models:")
    for key, model in sorted((config.get("models") or {}).items()):
        endpoint = model.get("endpoint", config.get("defaults", {}).get("endpoint", "openrouter"))
        lines.append(f"  {key}: {model.get('model_id')} ({endpoint})")
    return "\n".join(lines)


def config_inventory(config: dict[str, Any]) -> dict[str, Any]:
    """Return a stable machine-readable inventory for operators and agents."""
    defaults = dict(config.get("defaults") or {})
    endpoints = config.get("endpoints") or {}
    models = config.get("models") or {}
    judge_models = config.get("judge_models") or {}
    model_groups = config.get("model_groups") or {}
    judge_sets = config.get("judge_sets") or {}
    agents = config.get("agents") or {}
    module_agents = config.get("module_agents") or {}
    agent_profiles = config.get("agent_profiles") or {}

    default_parallel = defaults.get("max_parallel")
    default_endpoint = defaults.get("endpoint")

    return {
        "schema_version": config.get("schema_version"),
        "description": config.get("description"),
        "defaults": defaults,
        "endpoints": [
            {
                "name": name,
                "provider_api": endpoint.get("provider_api", "openai_compatible"),
                "openai_base_url": endpoint.get("openai_base_url"),
                "chat_completions_url": endpoint.get("chat_completions_url"),
                "messages_url": endpoint.get("messages_url"),
                "generate_content_base_url": endpoint.get("generate_content_base_url"),
                "api_key_env": endpoint.get("api_key_env"),
            }
            for name, endpoint in sorted(endpoints.items())
            if isinstance(endpoint, dict)
        ],
        "agents": [
            {
                "name": name,
                "model_id": agent.get("model_id"),
                "role": agent.get("role"),
            }
            for name, agent in sorted(agents.items())
            if isinstance(agent, dict)
        ],
        "agent_profiles": [
            {
                "name": name,
                "description": profile.get("description"),
                "agents": [
                    {
                        "name": agent_name,
                        "model_id": agent.get("model_id"),
                        "role": agent.get("role"),
                    }
                    for agent_name, agent in sorted((profile.get("agents") or {}).items())
                    if isinstance(agent, dict)
                ],
            }
            for name, profile in sorted(agent_profiles.items())
            if isinstance(profile, dict)
        ],
        "module_agents": [
            {
                "module": module_name,
                "agents": [
                    {
                        "name": agent_name,
                        "model_id": agent.get("model_id"),
                        "role": agent.get("role"),
                    }
                    for agent_name, agent in sorted(agents_for_module.items())
                    if isinstance(agent, dict)
                ],
            }
            for module_name, agents_for_module in sorted(module_agents.items())
            if isinstance(agents_for_module, dict)
        ],
        "judge_sets": [
            {
                "name": name,
                "description": judge.get("description"),
                "primary": judge.get("primary"),
                "panel": list(judge.get("panel", [judge.get("primary")]) or []),
            }
            for name, judge in sorted(judge_sets.items())
            if isinstance(judge, dict)
        ],
        "judge_models": [
            {
                "key": key,
                "model_id": judge_model.get("model_id"),
                "label": judge_model.get("label", key),
                "endpoint": judge_model.get("endpoint", default_endpoint),
                "condition_id": judge_model.get("condition_id"),
                "condition_metadata": judge_model.get("condition_metadata"),
                "request_options": judge_model.get("request_options"),
            }
            for key, judge_model in sorted(judge_models.items())
            if isinstance(judge_model, dict)
        ],
        "model_groups": [
            {
                "name": name,
                "models": list(keys),
                "model_count": len(keys),
            }
            for name, keys in sorted(model_groups.items())
            if isinstance(keys, list)
        ],
        "models": [
            {
                "key": key,
                "model_id": model.get("model_id"),
                "label": model.get("label", key),
                "endpoint": model.get("endpoint", default_endpoint),
                "max_parallel": model.get("max_parallel", default_parallel),
                "api_key_env": model.get("api_key_env"),
                **{
                    field: model[field]
                    for field in MODEL_CONDITION_METADATA_FIELDS
                    if field in model
                },
            }
            for key, model in sorted(models.items())
            if isinstance(model, dict)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render module models.yaml files from suite_models.yaml.")
    parser.add_argument("--config", default=str(DEFAULT_SUITE_CONFIG), help="Path to suite_models.yaml.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated configs.")
    parser.add_argument("--judge-set", default="calibration", help="Judge set from suite_models.yaml.")
    parser.add_argument(
        "--models",
        default="all",
        help="Model selector: all, group:<name>, or comma-separated model keys.",
    )
    parser.add_argument("--agent-profile", help="Agent profile from suite_models.yaml.")
    parser.add_argument(
        "--module",
        choices=(*MODULES, "all"),
        default="all",
        help="Module config to render.",
    )
    parser.add_argument("--list", action="store_true", help="List configured judge sets, model groups, and models.")
    parser.add_argument("--validate", action="store_true", help="Validate the suite config and exit.")
    parser.add_argument("--output-json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_suite_config(args.config)
    warnings = validate_suite_config(config)
    if args.list:
        if args.output_json:
            print(json.dumps(config_inventory(config), indent=2, sort_keys=True))
        else:
            print(describe_config(config))
        return 0
    if args.validate:
        if args.output_json:
            print(
                json.dumps(
                    {"config_path": args.config, "valid": True, "warnings": warnings},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"{args.config} OK")
            for warning in warnings:
                print(f"WARNING: {warning}")
        return 0
    modules = MODULES if args.module == "all" else (args.module,)
    written = write_rendered_configs(
        suite_config_path=args.config,
        output_dir=args.output_dir,
        judge_set=args.judge_set,
        model_selector=args.models,
        agent_profile=args.agent_profile,
        modules=modules,
    )
    if args.output_json:
        print(
            json.dumps(
                {
                    "config_path": args.config,
                    "judge_set": args.judge_set,
                    "model_selector": args.models,
                    "agent_profile": args.agent_profile or "default",
                    "modules": list(modules),
                    "written": [str(path) for path in written],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for path in written:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
