"""Standing preflight gate for benchmark model conditions.

Given prepared run directories or ``suite_models.yaml`` model-group names, this
tool probes every unique ``(model_id, effort, endpoint)`` cell with a minimal
request and classifies the response:

* HTTP 200            -> PASS (the provider accepts the parameter combination)
* HTTP 400            -> FAIL (invalid/unsupported parameter, e.g. an effort the
                        endpoint rejects). The failing condition is listed.
* anything else / no  -> ERROR (could not confirm acceptance; treated as a gate
  API key                failure so a broken cell never slips into a paid run)

Every target makes a network request. The bundled local reference adapter probes
are free; remote or proxy targets may bill according to their provider. The
probe uses ``max_output_tokens<=16`` to keep remote cost small, but the harness
does not promise a fixed price. Any non-PASS cell yields a nonzero exit.

This module contains only pure request-building / classification logic plus a
thin ``httpx.post`` seam (``poster``) so the accounting paths can be unit tested
with mocked HTTP. The orchestrator runs the live probe after preparation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlsplit

import httpx

from suite_tools.credential_policy import destination_policy_error
from suite_tools.env import load_repo_env_files
from suite_tools.model_config import (
    DEFAULT_SUITE_CONFIG,
    load_suite_config,
    render_module_config,
    route_identity_hash,
)
from suite_tools.run_contract import (
    stable_json_hash,
    validate_run_prepared_config_before_spend,
)
from suite_tools.run_monitor import atomic_write_json
from suite_tools.provider_client import (
    build_gemini_generate_content_payload,
    is_openai_native_url,
    model_uses_max_completion_tokens,
    normalize_chat_payload_for_provider,
)

_PROBE_MAX_OUTPUT_TOKENS = 16
_ANTHROPIC_VERSION = "2023-06-01"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
_RECEIPT_FILENAME = "PREFLIGHT_RECEIPT.json"
_RECEIPT_SCHEMA_VERSION = "benchmark-preflight-receipt-v1"
_REPORT_SCHEMA_VERSION = "benchmark-preflight-report-v1"
PREFLIGHT_RECEIPT_TTL_SECONDS = 6 * 60 * 60
PREFLIGHT_RECEIPT_CLOCK_SKEW_SECONDS = 5 * 60
_UNKNOWN_USAGE = {
    "state": "unknown",
    "source": "unknown",
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
}
_UNKNOWN_COST = {"state": "unknown", "usd": None, "source": "unknown"}
_SAFE_REASON_CODES = {
    "accepted",
    "destination_policy",
    "invalid_endpoint_origin",
    "malformed_response",
    "missing_api_key",
    "missing_endpoint_origin",
    "request_error",
    "unexpected_status",
    "unsupported_parameter",
    "unknown",
}


@dataclass(frozen=True)
class ProbeTarget:
    model_id: str
    effort: str | None
    provider_api: str
    base_url: str
    api_key_env: str
    role: str = "model_under_test"
    condition_ids: tuple[str, ...] = ()
    request_options: dict[str, Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def key(self) -> tuple[str, str, str | None, str, str, str, str]:
        return (
            self.role,
            self.model_id,
            self.effort,
            self.provider_api,
            self.base_url,
            self.api_key_env,
            stable_json_hash(self.request_options or {}),
        )


@dataclass
class ProbeResult:
    target: ProbeTarget
    status: str  # "PASS" | "FAIL" | "ERROR"
    http_status: int | None
    reason: str
    reason_code: str = "unknown"
    usage: dict[str, Any] = field(default_factory=lambda: dict(_UNKNOWN_USAGE))
    cost: dict[str, Any] = field(default_factory=lambda: dict(_UNKNOWN_COST))


@dataclass(frozen=True)
class PreparedRunContext:
    run_dir: Path
    run_id: str
    module: str | None
    targets: tuple[ProbeTarget, ...]
    prepared_config: dict[str, Any]
    contract_provenance: dict[str, Any]
    contract_provenance_fingerprint: str
    snapshot_hash: str


class PreflightReceiptValidationError(ValueError):
    """Raised when durable preflight evidence cannot admit paid work."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(sorted(set(str(issue) for issue in issues)))
        super().__init__("preflight receipt invalid: " + "; ".join(self.issues))


def _effort_for_entry(entry: dict[str, Any]) -> str | None:
    request_options = entry.get("request_options")
    if isinstance(request_options, dict):
        if request_options.get("reasoning_effort") is not None:
            return str(request_options["reasoning_effort"])
        reasoning = request_options.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            return str(reasoning["effort"])
        output_config = request_options.get("output_config")
        if isinstance(output_config, dict) and output_config.get("effort") is not None:
            return str(output_config["effort"])
    metadata = entry.get("condition_metadata")
    if isinstance(metadata, dict) and metadata.get("effort") is not None:
        return str(metadata["effort"])
    return None


def _condition_id_for_entry(entry: dict[str, Any]) -> str | None:
    value = entry.get("condition_id") or entry.get("id") or entry.get("label")
    return str(value) if value else None


def _targets_from_entries(entries: Iterable[dict[str, Any]]) -> list[ProbeTarget]:
    grouped: dict[tuple, dict[str, Any]] = {}
    for entry in entries:
        model_id = entry.get("model_id") or entry.get("id")
        if not model_id:
            continue
        provider_api = entry.get("provider_api", "openai_compatible")
        base_url = entry.get("base_url", "")
        effort = _effort_for_entry(entry)
        api_key_env = entry.get("api_key_env", "OPENROUTER_API_KEY")
        role = str(entry.get("_preflight_role") or "model_under_test")
        raw_request_options = entry.get("request_options")
        if raw_request_options is not None and not isinstance(raw_request_options, dict):
            raise ValueError(
                f"preflight request_options must be a mapping for {model_id}"
            )
        request_options = copy.deepcopy(raw_request_options or {})
        request_controls_hash = stable_json_hash(request_options)
        key = (
            role,
            str(model_id),
            effort,
            str(provider_api),
            str(base_url),
            str(api_key_env),
            request_controls_hash,
        )
        bucket = grouped.setdefault(
            key,
            {
                "api_key_env": api_key_env,
                "condition_ids": [],
                "request_options": request_options,
            },
        )
        condition_id = _condition_id_for_entry(entry)
        if condition_id and condition_id not in bucket["condition_ids"]:
            bucket["condition_ids"].append(condition_id)

    targets: list[ProbeTarget] = []
    for (
        role,
        model_id,
        effort,
        provider_api,
        base_url,
        api_key_env,
        _request_controls_hash,
    ), bucket in grouped.items():
        targets.append(
            ProbeTarget(
                model_id=model_id,
                effort=effort,
                provider_api=provider_api,
                base_url=base_url,
                api_key_env=api_key_env,
                role=role,
                condition_ids=tuple(bucket["condition_ids"]),
                request_options=copy.deepcopy(bucket["request_options"]),
            )
        )
    return targets


def _openrouter_support_entry(model_id: Any, role: str) -> dict[str, Any] | None:
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    return {
        "model_id": model_id,
        "condition_id": f"support:{role}",
        "provider_api": "openai_compatible",
        "base_url": _OPENROUTER_BASE_URL,
        "api_key_env": _OPENROUTER_API_KEY_ENV,
        "_preflight_role": role,
    }


def _entries_from_rendered_config(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project every paid role from one rendered module config."""
    entries: list[dict[str, Any]] = []
    models = data.get("models")
    if isinstance(models, dict):
        model_entries = [value for value in models.values() if isinstance(value, dict)]
    elif isinstance(models, list):
        model_entries = [value for value in models if isinstance(value, dict)]
    else:
        model_entries = []
    entries.extend({**entry, "_preflight_role": "model_under_test"} for entry in model_entries)

    judge = data.get("judge") if isinstance(data.get("judge"), dict) else {}
    judge_configs = data.get("judge_configs")
    if not isinstance(judge_configs, list):
        judge_configs = judge.get("configs")
    if not isinstance(judge_configs, list) or not judge_configs:
        primary = judge.get("primary_config")
        judge_configs = [primary] if isinstance(primary, dict) else []
    entries.extend(
        {**entry, "_preflight_role": "judge"}
        for entry in judge_configs
        if isinstance(entry, dict)
    )

    analyzer = data.get("analyzer")
    analyzer_entry = _openrouter_support_entry(
        analyzer.get("model_id") if isinstance(analyzer, dict) else analyzer,
        "analyzer",
    )
    if analyzer_entry:
        if isinstance(analyzer, dict):
            analyzer_entry.update(analyzer)
            analyzer_entry["_preflight_role"] = "analyzer"
        entries.append(analyzer_entry)

    for config_key, role in (("seeker", "seeker"), ("flip_generator", "flip_generator")):
        support = data.get(config_key)
        model_id = support.get("model_id") if isinstance(support, dict) else support
        support_entry = _openrouter_support_entry(model_id, role)
        if support_entry:
            if isinstance(support, dict):
                support_entry.update(support)
                support_entry["_preflight_role"] = role
            entries.append(support_entry)
    return entries


def collect_targets_from_groups(
    config: dict[str, Any], groups: Iterable[str]
) -> list[ProbeTarget]:
    """Return deduped probe targets for the given model-group names.

    Rendering all modules resolves evaluated models and every configured paid
    support role into the exact routes each runner would use.
    """
    entries: list[dict[str, Any]] = []
    for group in groups:
        for module in ("sus", "aita", "epis"):
            rendered = render_module_config(
                config, module=module, model_selector=f"group:{group}"
            )
            entries.extend(_entries_from_rendered_config(rendered))
    return _targets_from_entries(entries)


def collect_targets_from_run_dir(run_dir: str | Path) -> list[ProbeTarget]:
    """Return deduped probe targets from a prepared run's model config files."""
    import yaml

    root = Path(run_dir)
    trust_root = (root.parent if (root / "RUN_CONTRACT.json").is_file() else root).resolve()
    entries: list[dict[str, Any]] = []
    # Accept legacy model_config files and current _configs/*-models output.
    patterns = (
        "*model_config*.yaml",
        "*model_config*.yml",
        "*model_config*.json",
        "*-models*.yaml",
        "*-models*.yml",
        "*-models*.json",
    )
    candidates: set[Path] = set()

    def add_trusted_candidate(path: Path, *, contract_artifact: bool = False) -> None:
        try:
            resolved = path.resolve()
        except OSError as exc:
            raise ValueError(f"preflight could not resolve model config path: {path}") from exc
        if not resolved.is_relative_to(trust_root):
            if contract_artifact:
                raise ValueError(
                    "rendered model config is outside the prepared run group: "
                    f"{path}"
                )
            return
        if resolved.is_file():
            candidates.add(resolved)

    contract_path = root / "RUN_CONTRACT.json"
    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed RUN_CONTRACT.json; refusing config scan") from exc
        validate_run_prepared_config_before_spend(root)
        rendered_artifacts = []
        for module in contract.get("modules") or []:
            if not isinstance(module, dict):
                continue
            for artifact in module.get("expected_artifacts") or []:
                if not isinstance(artifact, dict) or artifact.get("kind") != "rendered_models":
                    continue
                rendered_artifacts.append(artifact)
        if rendered_artifacts:
            if len(rendered_artifacts) != 1:
                raise ValueError(
                    "prepared contract must declare exactly one rendered model config"
                )
            artifact = rendered_artifacts[0]
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("rendered model config artifact path is missing")
            candidate = Path(raw_path)
            resolved_options = (
                [candidate]
                if candidate.is_absolute()
                else [
                    root / candidate,
                    root.parent / candidate,
                    DEFAULT_SUITE_CONFIG.parent / candidate,
                ]
            )
            existing_trusted = set()
            escaped = False
            trusted_option_seen = False
            for path in resolved_options:
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not resolved.is_relative_to(trust_root):
                    escaped = True
                    continue
                trusted_option_seen = True
                if resolved.is_file():
                    existing_trusted.add(resolved)
            if len(existing_trusted) != 1:
                if escaped and not trusted_option_seen:
                    raise ValueError(
                        "rendered model config is outside the prepared run group: "
                        f"{raw_path}"
                    )
                raise ValueError(
                    "rendered model config is missing or ambiguous: " f"{raw_path}"
                )
            candidates = existing_trusted
        else:
            # Legacy contracts did not declare their rendered config artifact.
            for pattern in patterns:
                for path in root.rglob(pattern):
                    add_trusted_candidate(path)
    else:
        # Legacy/unprepared directory mode: accept only configs found under the
        # caller-selected directory.
        for pattern in patterns:
            for path in root.rglob(pattern):
                add_trusted_candidate(path)
    for path in sorted(candidates):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        entries.extend(_entries_from_rendered_config(data))
    return _targets_from_entries(entries)


def build_probe_request(
    target: ProbeTarget, *, api_key: str
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Build a low-output probe with the target's exact request controls."""
    if target.request_options is not None and not isinstance(target.request_options, dict):
        raise ValueError("preflight request_options must be a mapping")
    request_options = copy.deepcopy(target.request_options or {})

    if target.provider_api == "openai_responses":
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            request_options.pop(key, None)
        body_effort = request_options.pop("reasoning_effort", None)
        reasoning_option = request_options.pop("reasoning", None)
        if body_effort is None and isinstance(reasoning_option, dict):
            body_effort = reasoning_option.get("effort")
        payload: dict[str, Any] = {
            "model": target.model_id,
            "input": "ping",
            "max_output_tokens": _PROBE_MAX_OUTPUT_TOKENS,
        }
        if body_effort is not None:
            payload["reasoning"] = {"effort": str(body_effort)}
        for key, value in request_options.items():
            payload.setdefault(key, value)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return target.base_url, payload, headers

    if target.provider_api == "anthropic_messages":
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            request_options.pop(key, None)
        payload = {
            "model": target.model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": _PROBE_MAX_OUTPUT_TOKENS,
        }
        payload.update(request_options)
        payload["max_tokens"] = _PROBE_MAX_OUTPUT_TOKENS
        payload = normalize_chat_payload_for_provider(
            payload,
            base_url=target.base_url,
        )
        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        return target.base_url, payload, headers

    if target.provider_api == "gemini_generate_content":
        root = target.base_url.rstrip("/")
        if not root.endswith(":generateContent"):
            root = f"{root}/models/{quote(target.model_id, safe='')}:generateContent"
        payload = build_gemini_generate_content_payload(
            [{"role": "user", "content": "ping"}],
            max_tokens=_PROBE_MAX_OUTPUT_TOKENS,
            request_options=request_options,
        )
        payload.setdefault("generationConfig", {})[
            "maxOutputTokens"
        ] = _PROBE_MAX_OUTPUT_TOKENS
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        return root, payload, headers

    # openai_compatible (OpenRouter or direct OpenAI chat completions)
    is_direct_openai = is_openai_native_url(target.base_url)
    requested_completion_cap = "max_completion_tokens" in request_options
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        request_options.pop(key, None)
    payload = {
        "model": target.model_id,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload.update(request_options)
    if is_direct_openai and model_uses_max_completion_tokens(target.model_id):
        payload["max_completion_tokens"] = _PROBE_MAX_OUTPUT_TOKENS
    elif requested_completion_cap:
        payload["max_completion_tokens"] = _PROBE_MAX_OUTPUT_TOKENS
    else:
        payload["max_tokens"] = _PROBE_MAX_OUTPUT_TOKENS
    payload = normalize_chat_payload_for_provider(payload, base_url=target.base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = target.base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    return url, payload, headers


def _error_snippet(response: Any) -> str:
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
    text = getattr(response, "text", "") or ""
    return str(text)[:300]


def _destination_policy_error(
    target: ProbeTarget,
    allowed_endpoint_hosts: Iterable[str] = (),
) -> str | None:
    """Prevent a model config from routing an environment secret off-origin."""
    error = destination_policy_error(
        target.api_key_env,
        target.base_url,
        allowed_endpoint_hosts,
    )
    if error and "BENCHMARK_ALLOWED_ENDPOINT_HOSTS" in error:
        return error + " or --allow-endpoint-host"
    return error


def _endpoint_origin_error(target: ProbeTarget) -> tuple[str, str] | None:
    raw = target.base_url
    if not isinstance(raw, str) or not raw.strip():
        return "missing_endpoint_origin", "endpoint origin is missing"
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "invalid_endpoint_origin", "endpoint origin is malformed"
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in hostname)
        or port == 0
    ):
        return "invalid_endpoint_origin", "endpoint origin must be an HTTP(S) URL"
    if parsed.username is not None or parsed.password is not None:
        return "invalid_endpoint_origin", "endpoint origin must not contain credentials"
    return None


def _response_payload(response: Any) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _valid_success_payload(target: ProbeTarget, payload: dict[str, Any]) -> bool:
    if target.provider_api == "openai_responses":
        return payload.get("status") in {"completed", "incomplete"}
    if target.provider_api == "anthropic_messages":
        content = payload.get("content")
        return isinstance(content, list) and bool(content)
    if target.provider_api == "gemini_generate_content":
        candidates = payload.get("candidates")
        return isinstance(candidates, list) and bool(candidates)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    return isinstance(message, dict) and (
        "content" in message or "refusal" in message
    )


def _finite_nonnegative_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def _token_count(value: Any) -> int | None:
    number = _finite_nonnegative_number(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _response_accounting(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = payload.get("usage")
    usage_source = "response.usage"
    if not isinstance(usage, dict):
        usage = payload.get("usageMetadata")
        usage_source = "response.usageMetadata"
    if not isinstance(usage, dict):
        return dict(_UNKNOWN_USAGE), dict(_UNKNOWN_COST)

    input_tokens = next(
        (
            count
            for key in ("prompt_tokens", "input_tokens", "promptTokenCount")
            if (count := _token_count(usage.get(key))) is not None
        ),
        None,
    )
    output_tokens = next(
        (
            count
            for key in ("completion_tokens", "output_tokens", "candidatesTokenCount")
            if (count := _token_count(usage.get(key))) is not None
        ),
        None,
    )
    total_tokens = next(
        (
            count
            for key in ("total_tokens", "totalTokenCount")
            if (count := _token_count(usage.get(key))) is not None
        ),
        None,
    )
    if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
        usage_receipt = {
            "state": "reported",
            "source": usage_source,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    else:
        usage_receipt = dict(_UNKNOWN_USAGE)

    reported_cost = _finite_nonnegative_number(usage.get("cost"))
    estimated_cost = _finite_nonnegative_number(usage.get("estimated_cost"))
    if reported_cost is not None:
        cost_receipt = {
            "state": "reported",
            "usd": float(reported_cost),
            "source": "response.usage.cost",
        }
    elif estimated_cost is not None:
        cost_receipt = {
            "state": "estimated",
            "usd": float(estimated_cost),
            "source": "response.usage.estimated_cost",
        }
    else:
        cost_receipt = dict(_UNKNOWN_COST)
    return usage_receipt, cost_receipt


def probe_target(
    target: ProbeTarget,
    *,
    api_key: str,
    poster: Callable[..., Any] = httpx.post,
    timeout: int = 60,
    allowed_endpoint_hosts: Iterable[str] = (),
) -> ProbeResult:
    origin_error = _endpoint_origin_error(target)
    if origin_error:
        code, reason = origin_error
        return ProbeResult(target, "ERROR", None, reason, reason_code=code)
    if not api_key:
        return ProbeResult(
            target,
            "ERROR",
            None,
            f"missing API key ${target.api_key_env}; cannot probe",
            reason_code="missing_api_key",
        )
    policy_error = _destination_policy_error(target, allowed_endpoint_hosts)
    if policy_error:
        return ProbeResult(
            target, "ERROR", None, policy_error, reason_code="destination_policy"
        )
    url, payload, headers = build_probe_request(target, api_key=api_key)
    try:
        response = poster(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:  # network failure -> cannot confirm acceptance
        return ProbeResult(
            target,
            "ERROR",
            None,
            f"probe request failed: {exc}",
            reason_code="request_error",
        )

    status_code = getattr(response, "status_code", None)
    response_payload = _response_payload(response)
    if response_payload is None:
        usage, cost = dict(_UNKNOWN_USAGE), dict(_UNKNOWN_COST)
    else:
        usage, cost = _response_accounting(response_payload)
    if status_code == 200:
        if response_payload is None or not _valid_success_payload(target, response_payload):
            return ProbeResult(
                target,
                "ERROR",
                200,
                "provider returned a malformed success response",
                reason_code="malformed_response",
                usage=usage,
                cost=cost,
            )
        return ProbeResult(
            target,
            "PASS",
            200,
            "accepted",
            reason_code="accepted",
            usage=usage,
            cost=cost,
        )
    if status_code == 400:
        return ProbeResult(
            target,
            "FAIL",
            400,
            _error_snippet(response),
            reason_code="unsupported_parameter",
            usage=usage,
            cost=cost,
        )
    return ProbeResult(
        target,
        "ERROR",
        status_code,
        f"unexpected status {status_code}: {_error_snippet(response)}",
        reason_code="unexpected_status",
        usage=usage,
        cost=cost,
    )


def run_preflight(
    targets: Iterable[ProbeTarget],
    *,
    poster: Callable[..., Any] = httpx.post,
    env: dict[str, str] | None = None,
    allowed_endpoint_hosts: Iterable[str] = (),
) -> tuple[list[ProbeResult], int]:
    environ = env if env is not None else os.environ
    results: list[ProbeResult] = []
    for target in targets:
        origin_error = _endpoint_origin_error(target)
        if origin_error:
            code, reason = origin_error
            results.append(
                ProbeResult(target, "ERROR", None, reason, reason_code=code)
            )
            continue
        policy_error = _destination_policy_error(target, allowed_endpoint_hosts)
        if policy_error:
            results.append(
                ProbeResult(
                    target,
                    "ERROR",
                    None,
                    policy_error,
                    reason_code="destination_policy",
                )
            )
            continue
        api_key = environ.get(target.api_key_env, "")
        if not api_key:
            results.append(
                ProbeResult(
                    target,
                    "ERROR",
                    None,
                    f"missing API key ${target.api_key_env}; cannot probe",
                    reason_code="missing_api_key",
                )
            )
            continue
        results.append(
            probe_target(
                target,
                api_key=api_key,
                poster=poster,
                allowed_endpoint_hosts=allowed_endpoint_hosts,
            )
        )
    exit_code = 0 if all(r.status == "PASS" for r in results) else 1
    return results, exit_code


def _format_result(result: ProbeResult) -> str:
    t = result.target
    conditions = ",".join(t.condition_ids) if t.condition_ids else "-"
    return (
        f"[{result.status}] role={t.role} {t.model_id} effort={t.effort} "
        f"api={t.provider_api} http={result.http_status} "
        f"conditions={conditions} :: {result.reason}"
    )


def _target_snapshot(target: ProbeTarget) -> dict[str, Any]:
    """Return the full target identity used only as hash input."""
    return {
        "role": target.role,
        "model_id": target.model_id,
        "effort": target.effort,
        "provider_api": target.provider_api,
        "base_url": target.base_url,
        "api_key_env": target.api_key_env,
        "condition_ids": list(target.condition_ids),
        "request_options": copy.deepcopy(target.request_options or {}),
    }


def _safe_target_identity(target: ProbeTarget) -> dict[str, Any]:
    """Project a target without endpoint details or credential identifiers."""
    try:
        route_hash = route_identity_hash(target.provider_api, target.base_url)
    except ValueError:
        route_hash = None
    return {
        "role": target.role,
        "model_id": target.model_id,
        "effort": target.effort,
        "provider_api": target.provider_api,
        "route_hash": route_hash,
        "credential_ref_hash": stable_json_hash({"api_key_env": target.api_key_env}),
        "request_controls_hash": stable_json_hash(target.request_options or {}),
        "condition_ids": list(target.condition_ids),
    }


def _safe_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict) or usage.get("state") != "reported":
        return dict(_UNKNOWN_USAGE)
    source = usage.get("source")
    if source not in {"response.usage", "response.usageMetadata"}:
        return dict(_UNKNOWN_USAGE)
    values = {
        key: _token_count(usage.get(key))
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    if all(value is None for value in values.values()):
        return dict(_UNKNOWN_USAGE)
    return {"state": "reported", "source": source, **values}


def _safe_cost(cost: Any) -> dict[str, Any]:
    if not isinstance(cost, dict):
        return dict(_UNKNOWN_COST)
    state = cost.get("state")
    source = cost.get("source")
    allowed_sources = {
        "reported": "response.usage.cost",
        "estimated": "response.usage.estimated_cost",
    }
    if state not in allowed_sources or source != allowed_sources[state]:
        return dict(_UNKNOWN_COST)
    value = _finite_nonnegative_number(cost.get("usd"))
    if value is None:
        return dict(_UNKNOWN_COST)
    return {"state": state, "usd": float(value), "source": source}


def _safe_result_row(target: ProbeTarget, result: ProbeResult) -> dict[str, Any]:
    status = result.status if result.status in {"PASS", "FAIL", "ERROR"} else "ERROR"
    reason_code = (
        result.reason_code if result.reason_code in _SAFE_REASON_CODES else "unknown"
    )
    http_status = result.http_status
    if isinstance(http_status, bool) or not isinstance(http_status, int):
        http_status = None
    return {
        **_safe_target_identity(target),
        "status": status,
        "reason_code": reason_code,
        "http_status": http_status,
        "usage": _safe_usage(result.usage),
        "cost": _safe_cost(result.cost),
    }


def _target_set_hash(targets: Iterable[ProbeTarget]) -> str:
    rows = [_target_snapshot(target) for target in targets]
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    return stable_json_hash(rows)


def collect_prepared_run_context(run_dir: str | Path) -> PreparedRunContext:
    """Authenticate a current prepared module and capture its probe target set."""
    root = Path(run_dir).resolve()
    contract_path = root / "RUN_CONTRACT.json"
    prepared_config = validate_run_prepared_config_before_spend(root)
    if not isinstance(prepared_config, dict) or prepared_config.get("verified") is not True:
        raise ValueError(
            "durable preflight receipts require a current provenance-bound prepared contract"
        )
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("RUN_CONTRACT.json is unreadable") from exc
    if not isinstance(contract, dict):
        raise ValueError("RUN_CONTRACT.json is not an object")
    provenance = contract.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("prepared contract provenance is missing")

    targets = tuple(collect_targets_from_run_dir(root))
    if not targets:
        raise ValueError("prepared contract has no paid role targets to preflight")
    module_names = [
        str(module.get("module"))
        for module in contract.get("modules") or []
        if isinstance(module, dict) and module.get("module")
    ]
    if len(set(module_names)) != 1:
        raise ValueError("prepared module contract must identify exactly one module")
    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("prepared module contract run_id is missing")

    provenance_fingerprint = stable_json_hash(provenance)
    snapshot_hash = stable_json_hash({
        "run_id": run_id,
        "module": module_names[0],
        "prepared_config": prepared_config,
        "contract_provenance_fingerprint": provenance_fingerprint,
        "targets": [_target_snapshot(target) for target in targets],
    })
    return PreparedRunContext(
        run_dir=root,
        run_id=run_id,
        module=module_names[0],
        targets=targets,
        prepared_config=dict(prepared_config),
        contract_provenance=dict(provenance),
        contract_provenance_fingerprint=provenance_fingerprint,
        snapshot_hash=snapshot_hash,
    )


def preflight_receipt_fingerprint(receipt: dict[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_fingerprint", None)
    return stable_json_hash(payload)


def write_preflight_receipt(
    context: PreparedRunContext,
    results: Iterable[ProbeResult],
) -> Path:
    """Persist a prompt-free receipt after re-authenticating the prepared input."""
    fresh_context = collect_prepared_run_context(context.run_dir)
    if fresh_context.snapshot_hash != context.snapshot_hash:
        raise ValueError("prepared contract, config, or preflight target set changed during probe")

    by_key: dict[tuple[Any, ...], ProbeResult] = {}
    for result in results:
        key = result.target.key()
        if key in by_key:
            raise ValueError("preflight results contain a duplicate role target")
        by_key[key] = result
    rows: list[dict[str, Any]] = []
    for target in context.targets:
        result = by_key.get(target.key())
        if result is None:
            raise ValueError(
                f"preflight result is missing for role={target.role} model={target.model_id}"
            )
        rows.append(_safe_result_row(target, result))

    summary = {
        status.lower(): sum(1 for row in rows if row["status"] == status)
        for status in ("PASS", "FAIL", "ERROR")
    }
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "module": context.module,
        "prepared_config": dict(context.prepared_config),
        "contract_provenance_fingerprint": context.contract_provenance_fingerprint,
        "target_set_hash": _target_set_hash(context.targets),
        "summary": {**summary, "total": len(rows)},
        "results": rows,
    }
    receipt["receipt_fingerprint"] = preflight_receipt_fingerprint(receipt)
    receipt_path = context.run_dir / _RECEIPT_FILENAME
    atomic_write_json(receipt_path, receipt)
    return receipt_path


def _parse_receipt_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def validate_preflight_receipt_before_spend(
    run_dir: str | Path,
    *,
    now: datetime | None = None,
    ttl_seconds: int = PREFLIGHT_RECEIPT_TTL_SECONDS,
) -> dict[str, Any]:
    """Authenticate current role-target PASS evidence before paid work.

    The receipt is local integrity evidence, not a provider signature. This
    validator therefore re-authenticates the current contract and rendered
    config, recomputes the complete target set, and compares every receipt row
    rather than trusting the receipt fingerprint by itself.
    """
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise ValueError("preflight receipt TTL must be a positive integer")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("preflight receipt validation time must be timezone-aware")
    current_time = current_time.astimezone(timezone.utc)

    try:
        context = collect_prepared_run_context(run_dir)
    except ValueError as exc:
        raise PreflightReceiptValidationError([str(exc)]) from exc
    receipt_path = context.run_dir / _RECEIPT_FILENAME
    if receipt_path.is_symlink():
        raise PreflightReceiptValidationError(["receipt must be a regular local file"])
    if not receipt_path.is_file():
        raise PreflightReceiptValidationError(["receipt is missing"])
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightReceiptValidationError(["receipt is unreadable"]) from exc
    if not isinstance(receipt, dict):
        raise PreflightReceiptValidationError(["receipt is not an object"])

    issues: list[str] = []
    expected_top_fields = {
        "schema_version",
        "generated_at",
        "run_id",
        "module",
        "prepared_config",
        "contract_provenance_fingerprint",
        "target_set_hash",
        "summary",
        "results",
        "receipt_fingerprint",
    }
    if set(receipt) != expected_top_fields:
        issues.append("receipt fields do not match the supported schema")
    if receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION:
        issues.append("receipt schema_version is unsupported")
    if receipt.get("receipt_fingerprint") != preflight_receipt_fingerprint(receipt):
        issues.append("receipt fingerprint does not authenticate its contents")
    if receipt.get("run_id") != context.run_id:
        issues.append("receipt run_id does not match the current contract")
    if receipt.get("module") != context.module:
        issues.append("receipt module does not match the current contract")
    if receipt.get("prepared_config") != context.prepared_config:
        issues.append("receipt prepared-config binding does not match current authenticated bytes")
    if (
        receipt.get("contract_provenance_fingerprint")
        != context.contract_provenance_fingerprint
    ):
        issues.append("receipt contract-provenance fingerprint does not match")
    expected_target_set_hash = _target_set_hash(context.targets)
    if receipt.get("target_set_hash") != expected_target_set_hash:
        issues.append("receipt target-set hash does not match current role targets")

    generated_at = _parse_receipt_time(receipt.get("generated_at"))
    age_seconds: float | None = None
    if generated_at is None:
        issues.append("receipt generated_at is missing or malformed")
    else:
        age_seconds = (current_time - generated_at).total_seconds()
        if age_seconds < -PREFLIGHT_RECEIPT_CLOCK_SKEW_SECONDS:
            issues.append("receipt timestamp is too far in the future")
        elif age_seconds > ttl_seconds:
            issues.append(
                f"receipt is stale; maximum age is {ttl_seconds} seconds"
            )

    expected_identities = {
        stable_json_hash(_safe_target_identity(target)): target
        for target in context.targets
    }
    raw_results = receipt.get("results")
    seen_identities: set[str] = set()
    if not isinstance(raw_results, list):
        issues.append("receipt results must be a list")
        raw_results = []
    if len(raw_results) != len(expected_identities):
        issues.append("receipt must contain exactly one PASS per current role target")
    expected_result_fields = {
        "role",
        "model_id",
        "effort",
        "provider_api",
        "route_hash",
        "credential_ref_hash",
        "request_controls_hash",
        "condition_ids",
        "status",
        "reason_code",
        "http_status",
        "usage",
        "cost",
    }
    identity_fields = (
        "role",
        "model_id",
        "effort",
        "provider_api",
        "route_hash",
        "credential_ref_hash",
        "request_controls_hash",
        "condition_ids",
    )
    for index, row in enumerate(raw_results):
        if not isinstance(row, dict):
            issues.append(f"receipt result[{index}] is not an object")
            continue
        if set(row) != expected_result_fields:
            issues.append(f"receipt result[{index}] fields do not match the schema")
        identity = {field: row.get(field) for field in identity_fields}
        identity_hash = stable_json_hash(identity)
        if identity_hash in seen_identities:
            issues.append("receipt must contain exactly one PASS per current role target")
            continue
        seen_identities.add(identity_hash)
        if identity_hash not in expected_identities:
            issues.append(f"receipt result[{index}] target identity is not current")
        if row.get("status") != "PASS":
            issues.append(f"receipt result[{index}] must be PASS")
        if row.get("http_status") != 200 or row.get("reason_code") != "accepted":
            issues.append(f"receipt result[{index}] does not prove an accepted HTTP 200")
        if row.get("usage") != _safe_usage(row.get("usage")):
            issues.append(f"receipt result[{index}] usage provenance is malformed")
        if row.get("cost") != _safe_cost(row.get("cost")):
            issues.append(f"receipt result[{index}] cost provenance is malformed")
    if seen_identities != set(expected_identities):
        issues.append("receipt must contain exactly one PASS per current role target")

    expected_summary = {
        "pass": len(expected_identities),
        "fail": 0,
        "error": 0,
        "total": len(expected_identities),
    }
    if receipt.get("summary") != expected_summary:
        issues.append("receipt summary does not match current PASS results")
    if issues:
        raise PreflightReceiptValidationError(issues)

    return {
        "verified": True,
        "path": str(receipt_path),
        "generated_at": receipt["generated_at"],
        "age_seconds": max(0.0, float(age_seconds or 0.0)),
        "ttl_seconds": ttl_seconds,
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "contract_provenance_fingerprint": context.contract_provenance_fingerprint,
        "target_set_hash": expected_target_set_hash,
        "target_count": len(expected_identities),
    }


def validate_preflight_receipt_for_prepared_config(
    run_dir: str | Path,
    prepared_config_receipt: dict[str, Any] | bool,
) -> dict[str, Any] | bool:
    """Require admission only when config validation identified a current run.

    Runtime-owned and legacy v1/v3 contracts return ``False`` from prepared
    config validation and retain their existing compatibility behavior. A
    current provenance-bound prepared config must always carry a valid receipt.
    """
    if not (
        isinstance(prepared_config_receipt, dict)
        and prepared_config_receipt.get("verified") is True
    ):
        return False
    return validate_preflight_receipt_before_spend(run_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", action="append", default=[], dest="groups",
        help="suite_models.yaml model-group name (repeatable)",
    )
    parser.add_argument(
        "--run-dir", action="append", default=[], dest="run_dirs",
        help="prepared run directory to scan for model configs (repeatable)",
    )
    parser.add_argument(
        "--suite-config", default=str(DEFAULT_SUITE_CONFIG),
        help="path to suite_models.yaml",
    )
    parser.add_argument(
        "--allow-endpoint-host",
        action="append",
        default=[],
        help=(
            "Explicitly trust one HTTPS hostname for a custom API-key variable. "
            "Repeat for multiple remote custom endpoints. Official provider keys "
            "remain bound to their canonical hosts."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a prompt-free machine-readable result report",
    )
    return parser


def collect_targets(args: argparse.Namespace) -> list[ProbeTarget]:
    targets: list[ProbeTarget] = []
    if args.groups:
        config = load_suite_config(args.suite_config)
        targets.extend(collect_targets_from_groups(config, args.groups))
    for run_dir in args.run_dirs:
        targets.extend(collect_targets_from_run_dir(run_dir))
    # Final dedup across sources.
    return _dedup(targets)


def _dedup(targets: Iterable[ProbeTarget]) -> list[ProbeTarget]:
    seen: dict[tuple, ProbeTarget] = {}
    for target in targets:
        existing = seen.get(target.key())
        if existing is None:
            seen[target.key()] = target
            continue
        condition_ids = list(existing.condition_ids)
        condition_ids.extend(
            condition_id
            for condition_id in target.condition_ids
            if condition_id not in condition_ids
        )
        seen[target.key()] = ProbeTarget(
            model_id=existing.model_id,
            effort=existing.effort,
            provider_api=existing.provider_api,
            base_url=existing.base_url,
            api_key_env=existing.api_key_env,
            role=existing.role,
            condition_ids=tuple(condition_ids),
            request_options=copy.deepcopy(existing.request_options or {}),
        )
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contexts: list[PreparedRunContext] = []
    try:
        targets = collect_targets(args)
        for run_dir in args.run_dirs:
            prepared_config = validate_run_prepared_config_before_spend(run_dir)
            if isinstance(prepared_config, dict) and prepared_config.get("verified") is True:
                contexts.append(collect_prepared_run_context(run_dir))
    except ValueError as exc:
        if args.json:
            print(json.dumps({
                "schema_version": _REPORT_SCHEMA_VERSION,
                "status": "ERROR",
                "reason_code": "input_authentication_failed",
            }, sort_keys=True))
        else:
            print(f"preflight: {exc}", file=sys.stderr)
        return 2
    if not targets:
        if args.json:
            print(json.dumps({
                "schema_version": _REPORT_SCHEMA_VERSION,
                "status": "ERROR",
                "reason_code": "no_probe_targets",
            }, sort_keys=True))
        else:
            print("preflight: no probe targets resolved from the given inputs", file=sys.stderr)
        return 2
    # Resolve and authenticate prepared inputs before reading any credentials.
    # Probes still load repo-local environment values without overriding an
    # already-set variable once the target set is trusted.
    load_repo_env_files()
    results, exit_code = run_preflight(
        targets,
        allowed_endpoint_hosts=args.allow_endpoint_host,
    )
    receipt_rows: list[dict[str, Any]] = []
    try:
        for context in contexts:
            receipt_path = write_preflight_receipt(context, results)
            receipt = json.loads(receipt_path.read_text())
            receipt_rows.append({
                "path": str(receipt_path),
                "receipt_fingerprint": receipt["receipt_fingerprint"],
            })
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        exit_code = 2
        if not args.json:
            print(f"preflight: receipt persistence failed: {exc}", file=sys.stderr)

    passed = sum(1 for r in results if r.status == "PASS")
    if args.json:
        summary = {
            status.lower(): sum(1 for result in results if result.status == status)
            for status in ("PASS", "FAIL", "ERROR")
        }
        print(json.dumps({
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if exit_code == 0 else "ERROR",
            "exit_code": exit_code,
            "summary": {**summary, "total": len(results)},
            "results": [
                _safe_result_row(result.target, result) for result in results
            ],
            "receipts": receipt_rows,
        }, sort_keys=True))
    else:
        for result in results:
            stream = sys.stdout if result.status == "PASS" else sys.stderr
            print(_format_result(result), file=stream)
        print(
            f"preflight: {passed}/{len(results)} cells PASS; exit={exit_code}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
