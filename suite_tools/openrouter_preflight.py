"""OpenRouter model and pricing preflight for the benchmark suite.

The command validates configured OpenRouter slugs against OpenRouter's model
catalog and reports pricing metadata when the catalog provides it. It is
advisory: actual paid-run accounting should still use returned ``usage.cost``
or generation stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from suite_tools.env import load_repo_env_files
from suite_tools.model_config import DEFAULT_SUITE_CONFIG, load_suite_config, validate_suite_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_KEY_URL = "https://openrouter.ai/api/v1/key"
DEFAULT_CACHE_PATH = REPO_ROOT / ".cache" / "openrouter-models.json"
MAX_PREFLIGHT_RESPONSE_BYTES = 16 * 1024 * 1024
PRICING_SNAPSHOT_SCHEMA_VERSION = "benchmark-pricing-snapshot-v1"
OPENROUTER_PRICING_UNITS = "per_token"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep OpenRouter credentials on the validated request origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_openrouter_url(url: str) -> str:
    """Reject local-file, credential-bearing, and non-OpenRouter preflight URLs."""
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OpenRouter preflight URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "openrouter.ai"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ValueError("OpenRouter preflight URL must use https://openrouter.ai")
    return url


def _open_openrouter_json(url: str, *, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(validate_openrouter_url(url), headers=headers)
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(MAX_PREFLIGHT_RESPONSE_BYTES + 1)
    if len(raw) > MAX_PREFLIGHT_RESPONSE_BYTES:
        raise ValueError("OpenRouter preflight response exceeds the size limit")
    return json.loads(raw.decode("utf-8"))


def _add_ref(refs: dict[str, set[str]], model_id: str | None, source: str) -> None:
    if isinstance(model_id, str) and model_id.strip():
        refs.setdefault(model_id.strip(), set()).add(source)


def collect_openrouter_refs(config: dict[str, Any]) -> tuple[dict[str, set[str]], list[str]]:
    """Return OpenRouter model ids to validate, plus non-OpenRouter skips."""
    default_endpoint = config.get("defaults", {}).get("endpoint", "openrouter")
    judge_models = config.get("judge_models") or {}
    refs: dict[str, set[str]] = {}
    skipped: list[str] = []

    def _add_judge_ref(ref: str | None, source: str) -> None:
        """Resolve a judge ref: judge_models aliases carry their own endpoint."""
        if not isinstance(ref, str) or not ref.strip():
            return
        alias = judge_models.get(ref.strip())
        if isinstance(alias, dict):
            endpoint = alias.get("endpoint", default_endpoint)
            if endpoint == "openrouter":
                _add_ref(refs, alias.get("model_id"), source)
            else:
                skipped.append(f"{source}: {ref.strip()} -> {alias.get('model_id')} ({endpoint})")
            return
        _add_ref(refs, ref, source)

    for agent_name, agent in (config.get("agents") or {}).items():
        _add_ref(refs, agent.get("model_id"), f"agents.{agent_name}")

    for module_name, module_agents in (config.get("module_agents") or {}).items():
        if not isinstance(module_agents, dict):
            continue
        for agent_name, agent in module_agents.items():
            if isinstance(agent, dict):
                _add_ref(
                    refs,
                    agent.get("model_id"),
                    f"module_agents.{module_name}.{agent_name}",
                )

    for profile_name, profile in (config.get("agent_profiles") or {}).items():
        if not isinstance(profile, dict):
            continue
        for agent_name, agent in (profile.get("agents") or {}).items():
            if isinstance(agent, dict):
                _add_ref(
                    refs,
                    agent.get("model_id"),
                    f"agent_profiles.{profile_name}.agents.{agent_name}",
                )

    for judge_name, judge in (config.get("judge_sets") or {}).items():
        _add_judge_ref(judge.get("primary"), f"judge_sets.{judge_name}.primary")
        for index, model_id in enumerate(judge.get("panel") or []):
            _add_judge_ref(model_id, f"judge_sets.{judge_name}.panel[{index}]")

    for model_key, model in (config.get("models") or {}).items():
        endpoint = model.get("endpoint", default_endpoint)
        model_id = model.get("model_id")
        if endpoint == "openrouter":
            _add_ref(refs, model_id, f"models.{model_key}")
        else:
            skipped.append(f"models.{model_key}: {model_id} ({endpoint})")

    return refs, skipped


def load_catalog_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fetch_catalog(url: str = DEFAULT_CATALOG_URL, *, timeout: float = 20) -> dict[str, Any]:
    headers = {"User-Agent": "sus-unified-benchmark-suite/openrouter-preflight"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _open_openrouter_json(url, headers=headers, timeout=timeout)


def fetch_key_info(url: str = DEFAULT_KEY_URL, *, timeout: float = 20) -> dict[str, Any]:
    """Fetch current OpenRouter key/account usage metadata without generations."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "sus-unified-benchmark-suite/openrouter-preflight",
    }
    return _open_openrouter_json(url, headers=headers, timeout=timeout)


def sanitize_key_info(payload: dict[str, Any]) -> dict[str, Any]:
    """Return non-secret OpenRouter key telemetry for preflight reports."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        return {"available": False, "error": "OpenRouter key response did not contain an object"}

    fields = (
        "limit",
        "limit_reset",
        "limit_remaining",
        "include_byok_in_limit",
        "usage",
        "usage_daily",
        "usage_weekly",
        "usage_monthly",
        "byok_usage",
        "byok_usage_daily",
        "byok_usage_weekly",
        "byok_usage_monthly",
        "is_free_tier",
    )
    result = {
        "available": True,
        "label_present": isinstance(data.get("label"), str) and bool(data.get("label")),
        **{field: data.get(field) for field in fields if field in data},
        "rate_limit_deprecated_present": isinstance(data.get("rate_limit"), dict),
    }
    remaining = result.get("limit_remaining")
    result["credit_limit_exhausted"] = isinstance(remaining, (int, float)) and remaining <= 0
    result["credit_limit_low"] = isinstance(remaining, (int, float)) and 0 < remaining <= 5
    return result


def catalog_by_id(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = catalog.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenRouter catalog must contain a `data` list")
    by_id: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        for key in ("id", "canonical_slug"):
            value = item.get(key)
            if isinstance(value, str) and value:
                by_id.setdefault(value, item)
    return by_id


def _parse_decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid pricing value for {label}: {value}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(
            f"Invalid pricing value for {label}: expected non-negative finite value"
        )
    return parsed


def parse_price_overrides(values: list[str]) -> dict[str, dict[str, str]]:
    """Parse repeated model_id:prompt:completion CLI overrides."""
    overrides: dict[str, dict[str, str]] = {}
    for value in values:
        parts = value.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid --price-override `{value}`; expected model_id:prompt:completion")
        completion = parts[-1]
        prompt = parts[-2]
        model_id = ":".join(parts[:-2])
        _parse_decimal(prompt, f"{model_id}.prompt")
        _parse_decimal(completion, f"{model_id}.completion")
        overrides[model_id] = {"prompt": prompt, "completion": completion}
    return overrides


def _pricing_for(model_id: str, catalog_entry: dict[str, Any], overrides: dict[str, dict[str, str]]) -> dict[str, str]:
    if model_id in overrides:
        return overrides[model_id]
    pricing = catalog_entry.get("pricing") or {}
    if not isinstance(pricing, dict):
        return {}
    return {key: str(value) for key, value in pricing.items() if value is not None}


def validate_openrouter_catalog(
    config: dict[str, Any],
    catalog: dict[str, Any],
    *,
    price_overrides: dict[str, dict[str, str]] | None = None,
    strict_pricing: bool = False,
) -> dict[str, Any]:
    """Validate suite OpenRouter refs against a catalog response."""
    validate_suite_config(config)
    refs, skipped = collect_openrouter_refs(config)
    by_id = catalog_by_id(catalog)
    overrides = price_overrides or {}
    checked: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for model_id in sorted(refs):
        sources = sorted(refs[model_id])
        entry = by_id.get(model_id)
        if not entry:
            errors.append(f"Missing OpenRouter model `{model_id}` used by {', '.join(sources)}")
            continue
        pricing = _pricing_for(model_id, entry, overrides)
        missing_price_fields = [field for field in ("prompt", "completion") if field not in pricing]
        if missing_price_fields:
            message = (
                f"No OpenRouter prompt/completion pricing for `{model_id}` "
                f"({', '.join(missing_price_fields)} missing)"
            )
            if strict_pricing:
                errors.append(message)
            else:
                warnings.append(message)
        else:
            _parse_decimal(pricing["prompt"], f"{model_id}.prompt")
            _parse_decimal(pricing["completion"], f"{model_id}.completion")
        checked.append(
            {
                "model_id": model_id,
                "name": entry.get("name"),
                "sources": sources,
                "pricing": pricing,
                "context_length": entry.get("context_length"),
            }
        )

    return {
        "schema_version": PRICING_SNAPSHOT_SCHEMA_VERSION,
        "units": OPENROUTER_PRICING_UNITS,
        "provider": "openrouter",
        "generated_at": _utc_now(),
        "checked": checked,
        "skipped": skipped,
        "warnings": warnings,
        "errors": errors,
        "catalog_count": len(by_id),
    }


def _load_catalog(args: argparse.Namespace) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    if args.catalog_file:
        return load_catalog_file(args.catalog_file), str(args.catalog_file), warnings
    try:
        catalog = fetch_catalog(args.catalog_url, timeout=args.timeout)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if args.cache and Path(args.cache).exists():
            warnings.append(f"Live catalog fetch failed; using cache {args.cache}: {exc}")
            return load_catalog_file(args.cache), str(args.cache), warnings
        raise
    if args.cache:
        cache_path = Path(args.cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return catalog, args.catalog_url, warnings


def _load_key_info(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if args.skip_key_info:
        return {"available": False, "reason": "skipped"}, warnings
    if not os.environ.get("OPENROUTER_API_KEY"):
        return {"available": False, "reason": "OPENROUTER_API_KEY not set"}, warnings
    try:
        return sanitize_key_info(fetch_key_info(args.key_url, timeout=args.timeout)), warnings
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        warnings.append(f"OpenRouter key info fetch failed: {exc}")
        return {"available": False, "reason": "fetch_failed"}, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate suite OpenRouter model slugs and pricing metadata.")
    parser.add_argument("--config", default=str(DEFAULT_SUITE_CONFIG), help="Path to suite_models.yaml.")
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL, help="OpenRouter models endpoint.")
    parser.add_argument("--key-url", default=DEFAULT_KEY_URL, help="OpenRouter current key endpoint.")
    parser.add_argument("--catalog-file", help="Use a saved catalog JSON file instead of the network.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Catalog cache path. Set empty to disable.")
    parser.add_argument("--timeout", type=float, default=20, help="Network timeout in seconds.")
    parser.add_argument(
        "--price-override",
        action="append",
        default=[],
        help="Manual estimate override as model_id:prompt:completion. May be repeated.",
    )
    parser.add_argument("--strict-pricing", action="store_true", help="Treat missing pricing as an error.")
    parser.add_argument("--skip-key-info", action="store_true", help="Skip OpenRouter key usage/limit telemetry.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_repo_env_files()
    if args.cache == "":
        args.cache = None
    config = load_suite_config(args.config)
    overrides = parse_price_overrides(args.price_override)
    try:
        catalog, source, fetch_warnings = _load_catalog(args)
        report = validate_openrouter_catalog(
            config,
            catalog,
            price_overrides=overrides,
            strict_pricing=args.strict_pricing,
        )
        key_info, key_warnings = _load_key_info(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = {"ok": not report["errors"], "catalog_source": source, "key_info": key_info, **report}
    report["warnings"] = [*fetch_warnings, *key_warnings, *report["warnings"]]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"OpenRouter catalog: {report['catalog_count']} known ids from {source}")
        print(f"Checked OpenRouter refs: {len(report['checked'])}")
        print(f"Skipped non-OpenRouter refs: {len(report['skipped'])}")
        if report["key_info"].get("available"):
            key = report["key_info"]
            limit = key.get("limit")
            remaining = key.get("limit_remaining")
            usage_daily = key.get("usage_daily")
            byok = key.get("byok_usage")
            print(
                "OpenRouter key: "
                f"remaining={remaining if remaining is not None else 'unlimited'} "
                f"limit={limit if limit is not None else 'unlimited'} "
                f"daily_usage={usage_daily if usage_daily is not None else 'unknown'} "
                f"byok_usage={byok if byok is not None else 'unknown'}"
            )
            if key.get("rate_limit_deprecated_present"):
                print("OpenRouter key: deprecated rate_limit object present; ignored.")
        else:
            print(f"OpenRouter key: {report['key_info'].get('reason', 'unavailable')}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
